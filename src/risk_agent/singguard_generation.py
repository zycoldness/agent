"""Generate one SingGuard SFT example by executing real sequential tool calls."""

from __future__ import annotations

import json
import hashlib
import os
import re
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from risk_agent.singguard import (
    ActivePolicy,
    Message,
    ModerationSample,
    ParsedCompletion,
    build_initial_messages,
    render_training_row,
    validate_completion,
)
from risk_agent.singguard_tools import (
    ToolCall,
    ToolEnvironment,
    ToolResult,
    tool_declarations,
    tools_json,
)
from risk_agent.teacher import (
    GeminiTeacher,
    ProviderFailureDiagnostic,
    TeacherBudget,
    TeacherBudgetExceeded,
    TeacherRequestError,
    TeacherUsage,
    _UsageAccumulator,
)


class AgentTurn(BaseModel):
    """Exactly one teacher action: call a tool or finish."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str | None = None
    tool_call: ToolCall | None = None

    @model_validator(mode="after")
    def require_one_action(self) -> "AgentTurn":
        if (self.content is None) == (self.tool_call is None):
            raise ValueError("agent turn must contain exactly one action")
        return self


class AgentProvider(Protocol):
    """Stateful provider session for one generated sample."""

    def start(
        self,
        *,
        system: str,
        user: str,
        tool_specs: tuple[dict[str, object], ...],
    ) -> AgentTurn: ...

    def continue_with_tool_result(
        self,
        *,
        call: ToolCall,
        result: ToolResult,
    ) -> AgentTurn: ...


class GeneratedExample(BaseModel):
    """Validated teacher completion and its real tool trajectory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_messages: tuple[Message, Message]
    trajectory: tuple[Message, ...]
    completion: str
    parsed: ParsedCompletion
    tools_json: str | None = None


class GenerationRejected(ValueError):
    """One candidate failed deterministic validation."""

    def __init__(
        self,
        code: str,
        *,
        candidate: str | None = None,
        trajectory: tuple[Message, ...] = (),
    ) -> None:
        super().__init__(code)
        self.code = code
        self.candidate = candidate
        self.trajectory = trajectory


_REPAIRABLE_CODES = {
    "output_grammar",
    "answer_format",
    "slow_structure",
    "slow_rule_evidence",
    "slow_rule_order",
    "slow_verdict_consistency",
}
_REPAIR_REQUIREMENTS = {
    "output_grammar": (
        "Output exactly one label on the first line: safe or unsafe. Never output "
        "the literal alternatives together, and preserve every required tag."
    ),
    "answer_format": (
        "A safe result must answer exactly Safe. An unsafe result must list each "
        "triggered active-rule title exactly once in policy order."
    ),
    "slow_structure": (
        "Preserve all three named reasoning steps and the reasoning and answer tags."
    ),
    "slow_rule_evidence": (
        "Every HIT, NOT HIT, and OBSERVED BUT NOT UNSAFE rule line must include "
        "non-empty evidence after the verdict period. A bare NOT APPLICABLE verdict "
        "is valid."
    ),
    "slow_rule_order": "Check every active rule exactly once in policy order.",
    "slow_verdict_consistency": (
        "Change only the first-line label and answer as needed to match the existing "
        "HIT verdicts. Do not change any Step 2 verdict or its evidence."
    ),
}


class GeminiAgentProvider(GeminiTeacher):
    """Stateful Gemini chat adapter using native sequential function calls."""

    def __init__(
        self,
        *,
        model: str,
        client_factory: Callable[[], object] | None = None,
        types_module: object | None = None,
        temperature: float = 0.2,
        max_attempts: int = 3,
        initial_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 30.0,
        request_timeout_seconds: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
        max_output_tokens: int = 4096,
        budget: TeacherBudget | None = None,
        event_sink: Callable[[dict[str, object]], object] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            response_schema={"type": "object"},
            temperature=temperature,
            client_factory=client_factory,
            max_attempts=max_attempts,
            initial_backoff_seconds=initial_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
            request_timeout_seconds=request_timeout_seconds,
            sleep=sleep,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
            max_output_tokens=max_output_tokens,
            budget=budget,
        )
        self._types_module = types_module
        self._client: object | None = None
        self._config: object | None = None
        self._history: list[object] = []
        self._context_text = ""
        self._event_sink = event_sink
        self._sample_id: str | None = None

    def set_event_sink(
        self,
        event_sink: Callable[[dict[str, object]], object] | None,
    ) -> None:
        self._event_sink = event_sink

    def begin_sample(self, sample_id: str) -> None:
        self._sample_id = sample_id

    def diagnostic_metadata(self) -> dict[str, object]:
        try:
            sdk_version = version("google-genai")
        except PackageNotFoundError:
            sdk_version = "unknown"
        use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").casefold()
        return {
            "name": "gemini",
            "model": self.model,
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
            "sdk": "google-genai",
            "sdk_version": sdk_version,
            "backend": (
                "vertex_ai"
                if use_vertex in {"1", "true", "yes", "on"}
                else "developer_api"
            ),
        }

    def _emit(self, event: str, **fields: object) -> None:
        if self._event_sink is None:
            return
        payload: dict[str, object] = {
            "event": event,
            "provider": "gemini",
            "model": self.model,
        }
        if self._sample_id is not None:
            payload["sample_id"] = self._sample_id
        payload.update(fields)
        try:
            self._event_sink(payload)
        except Exception:
            pass

    @staticmethod
    def _failure_diagnostic(
        error: Exception,
        *,
        stage: str,
        attempts: int,
    ) -> ProviderFailureDiagnostic:
        code = getattr(error, "code", None)
        if isinstance(code, bool) or not isinstance(code, int):
            code = None
        status = getattr(error, "status", None)
        if not (
            isinstance(status, str)
            and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", status)
        ):
            status = None
        exception_type = type(error).__name__
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", exception_type):
            exception_type = "Exception"
        return ProviderFailureDiagnostic(
            stage=stage,
            exception_type=exception_type,
            http_code=code,
            provider_status=status,
            attempts=attempts,
        )

    @staticmethod
    def _retryable_provider_error(
        error: Exception,
        diagnostic: ProviderFailureDiagnostic,
    ) -> bool:
        if diagnostic.http_code is not None:
            return diagnostic.http_code in {408, 409, 425, 429} or (
                500 <= diagnostic.http_code <= 599
            )
        if diagnostic.provider_status is not None:
            return diagnostic.provider_status in {
                "ABORTED",
                "DEADLINE_EXCEEDED",
                "INTERNAL",
                "RESOURCE_EXHAUSTED",
                "UNAVAILABLE",
            }
        exception_type = type(error).__name__.casefold()
        return isinstance(error, (ConnectionError, TimeoutError)) or any(
            marker in exception_type
            for marker in ("connection", "timeout", "transport")
        )

    def _types(self) -> object:
        if self._types_module is not None:
            return self._types_module
        try:
            from google.genai import types
        except ImportError:
            raise RuntimeError(
                "Gemini function calling requires the optional 'teacher' dependency"
            ) from None
        return types

    def _send(self, message: object, *, stage: str) -> object:
        if self._client is None or self._config is None:
            raise RuntimeError("Gemini agent session has not been started")
        request_contents = [*self._history, message]
        aggregate = _UsageAccumulator(
            "gemini",
            self.model,
            self._input_cost_per_million,
            self._output_cost_per_million,
        )
        worst_case_cost = self._worst_case_cost(self._context_text)
        for attempt in range(1, self._max_attempts + 1):
            reservation = 0.0
            if self._budget is not None:
                reservation = self._budget.reserve_request(
                    worst_case_cost_usd=worst_case_cost
                )
            self._emit(
                "provider_request_started",
                stage=stage,
                attempt=attempt,
                max_attempts=self._max_attempts,
            )
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=request_contents,
                    config=self._config,
                )
            except Exception as error:
                usage = aggregate.add(None, None)
                diagnostic = self._failure_diagnostic(
                    error,
                    stage=stage,
                    attempts=attempt,
                )
                retryable = self._retryable_provider_error(error, diagnostic)
                self._emit(
                    "provider_request_failed",
                    **diagnostic.as_dict(),
                    attempt=attempt,
                    max_attempts=self._max_attempts,
                    retryable=retryable,
                )
                if self._budget is not None:
                    self._budget.record_attempt(
                        TeacherUsage(
                            provider="gemini",
                            model=self.model,
                            accounting_complete=False,
                        ),
                        reserved_cost_usd=reservation,
                    )
                if retryable and attempt < self._max_attempts:
                    delay = self._retry_delay(attempt)
                    self._emit(
                        "provider_retry_scheduled",
                        stage=stage,
                        attempt=attempt,
                        next_attempt=attempt + 1,
                        delay_seconds=delay,
                    )
                    self._sleep(delay)
                    continue
                raise TeacherRequestError(
                    f"Gemini agent request failed after {attempt} attempts",
                    usage,
                    diagnostic,
                ) from None
            usage = self._normalize_response_usage(response)
            aggregate.add(
                usage.input_tokens if usage.accounting_complete else None,
                usage.output_tokens if usage.accounting_complete else None,
                explicit_cost=usage.estimated_cost_usd,
            )
            if self._budget is not None:
                self._budget.record_attempt(
                    usage,
                    reserved_cost_usd=reservation,
                )
            self._emit(
                "provider_request_succeeded",
                stage=stage,
                attempt=attempt,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                accounting_complete=usage.accounting_complete,
            )
            try:
                response_content = response.candidates[0].content
            except Exception:
                raise ValueError("Gemini response content is unavailable") from None
            if response_content is None:
                raise ValueError("Gemini response content is unavailable")
            # Preserve the exact model content so Gemini 3 thought signatures survive.
            self._history = [*request_contents, response_content]
            return response
        raise RuntimeError("unreachable Gemini retry state")

    @staticmethod
    def _to_turn(response: object) -> AgentTurn:
        try:
            function_calls = tuple(getattr(response, "function_calls", None) or ())
        except Exception:
            raise ValueError("Gemini response function calls are unreadable") from None
        if len(function_calls) > 1:
            raise ValueError("parallel Gemini tool calls are not supported")
        if function_calls:
            call = function_calls[0]
            try:
                return AgentTurn(
                    tool_call=ToolCall(
                        name=str(call.name),
                        arguments=dict(call.args or {}),
                    )
                )
            except Exception:
                raise ValueError("Gemini returned an invalid tool call") from None
        try:
            text = response.text
        except Exception:
            text = None
        if not isinstance(text, str) or not text:
            raise ValueError("Gemini returned neither a tool call nor final text")
        return AgentTurn(content=text)

    def start(
        self,
        *,
        system: str,
        user: str,
        tool_specs: tuple[dict[str, object], ...],
    ) -> AgentTurn:
        types = self._types()
        config_kwargs: dict[str, object] = {
            "system_instruction": system,
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
        }
        if tool_specs:
            config_kwargs["tools"] = [
                types.Tool(function_declarations=list(tool_specs))
            ]
        try:
            self._client = self._new_client()
            self._config = types.GenerateContentConfig(**config_kwargs)
            self._history = []
            user_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=user)],
            )
        except Exception as error:
            diagnostic = self._failure_diagnostic(
                error,
                stage="client_init",
                attempts=0,
            )
            self._emit("provider_client_init_failed", **diagnostic.as_dict())
            raise TeacherRequestError(
                "Gemini agent client initialization failed",
                TeacherUsage(
                    provider="gemini",
                    model=self.model,
                    request_count=0,
                    accounting_complete=False,
                ),
                diagnostic,
            ) from None
        self._context_text = f"{system}\n{user}"
        response = self._send(user_content, stage="initial")
        return self._to_turn(response)

    def continue_with_tool_result(
        self,
        *,
        call: ToolCall,
        result: ToolResult,
    ) -> AgentTurn:
        types = self._types()
        payload = {"status": result.status, **result.payload}
        self._context_text += (
            "\n" + _tool_call_content(call) + "\n" + result.to_content()
        )
        part = types.Part.from_function_response(
            name=call.name,
            response=payload,
        )
        tool_content = types.Content(role="tool", parts=[part])
        response = self._send(tool_content, stage="tool_response")
        return self._to_turn(response)

    def repair(
        self,
        *,
        system: str,
        user: str,
        candidate: str,
        validation_code: str,
    ) -> AgentTurn:
        """Make one fresh, tool-free request for format or trace consistency."""

        types = self._types()
        if validation_code == "slow_verdict_consistency":
            consistency_guard = (
                "You may change the first-line label and answer so they match the "
                "existing HIT verdicts, but you must not change any Step 2 verdict "
                "or its evidence. "
            )
        else:
            consistency_guard = (
                "Do not change the moderation label or triggered rule, and do not "
                "change any Step 2 verdict. "
            )
        repair_instruction = (
            "\n\n# Output Repair\n"
            "The previous candidate failed deterministic output validation. The only "
            "permitted changes are output serialization and the minimum wording needed "
            "to fill the exact Output Format structure above. "
            f"{consistency_guard}Do not call tools, add commentary, "
            "omit required steps, or introduce policy categories that are not active."
        )
        repair_user = (
            f"{user}\n\n[validation_error]: {validation_code}"
            f"\n[repair_requirement]: {_REPAIR_REQUIREMENTS.get(validation_code, 'Re-emit the exact required output without changing its classification.')}"
            f"\n[rejected_candidate]:\n{candidate}"
        )
        try:
            self._client = self._new_client()
            self._config = types.GenerateContentConfig(
                system_instruction=system + repair_instruction,
                temperature=0.0,
                max_output_tokens=self._max_output_tokens,
            )
            self._history = []
            repair_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=repair_user)],
            )
        except Exception as error:
            diagnostic = self._failure_diagnostic(
                error,
                stage="repair_init",
                attempts=0,
            )
            self._emit("provider_client_init_failed", **diagnostic.as_dict())
            raise TeacherRequestError(
                "Gemini repair client initialization failed",
                TeacherUsage(
                    provider="gemini",
                    model=self.model,
                    request_count=0,
                    accounting_complete=False,
                ),
                diagnostic,
            ) from None
        self._context_text = f"{system}{repair_instruction}\n{repair_user}"
        return self._to_turn(self._send(repair_content, stage="repair"))


def _tool_call_content(call: ToolCall) -> str:
    return json.dumps(
        {"name": call.name, "arguments": call.arguments},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validation_code(error: ValueError) -> str:
    message = str(error)
    if (
        "must belong to the active policy" in message
        or "checks an inactive policy rule" in message
    ):
        return "inactive_answer"
    if "missing or duplicate titles" in message or "must answer Safe" in message:
        return "answer_format"
    if "missing the content summary" in message or "required reasoning steps" in message:
        return "slow_structure"
    if "reasoning steps must not be empty" in message:
        return "slow_structure"
    if "only active-rule lines" in message or "invalid evidence" in message:
        return "slow_rule_evidence"
    if "must not contain a hit" in message or "match every HIT rule" in message:
        return "slow_verdict_consistency"
    if "policy order" in message:
        return "slow_rule_order"
    if "grammar" in message:
        return "output_grammar"
    return "completion_validation"


def _repair_failure_code(error: ValueError) -> str:
    message = str(error)
    if message.startswith("Gemini ") or message.startswith("repair "):
        return "provider_turn_invalid"
    return _validation_code(error)


def _validate_candidate(
    completion: str,
    *,
    policy: ActivePolicy,
    sample: ModerationSample,
) -> ParsedCompletion:
    return validate_completion(
        completion,
        thinking_type=sample.thinking_type,
        active_titles=tuple(rule.title for rule in policy.rules),
    )


def _semantic_mismatch(
    parsed: ParsedCompletion,
    *,
    policy: ActivePolicy,
    sample: ModerationSample,
) -> str | None:
    if sample.expected_label is None:
        return None
    active_titles = {rule.title for rule in policy.rules}
    if any(answer not in active_titles for answer in sample.expected_answers):
        raise ValueError("expected answer does not belong to the active policy")
    if parsed.label != sample.expected_label:
        return "semantic_label_mismatch"
    if sample.expected_label == "unsafe" and parsed.answers != sample.expected_answers:
        return "semantic_answer_mismatch"
    return None


def generate_example(
    policy: ActivePolicy,
    sample: ModerationSample,
    provider: AgentProvider,
    environment: ToolEnvironment,
    *,
    max_tool_calls: int = 2,
    event_sink: Callable[[dict[str, object]], object] | None = None,
) -> GeneratedExample:
    """Execute the complete prompt and preserve only locally produced tool results."""

    if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int):
        raise ValueError("max_tool_calls must be an integer")
    if not 0 <= max_tool_calls <= 2:
        raise ValueError("max_tool_calls must be between zero and two")
    if len(sample.tool_names) != len(set(sample.tool_names)):
        raise ValueError("sample tool names must be unique")
    if sample.tool_policy == "required" and len(sample.tool_names) > max_tool_calls:
        raise ValueError("required tool sequence exceeds the tool call limit")
    if sample.expected_label == "unsafe":
        active_titles = {rule.title for rule in policy.rules}
        if any(answer not in active_titles for answer in sample.expected_answers):
            raise ValueError("expected answer does not belong to the active policy")
    serialized_tools = tools_json(sample.tool_names) if sample.tool_names else None
    declarations = tool_declarations(sample.tool_names)
    initial = build_initial_messages(policy, sample)
    try:
        turn = provider.start(
            system=initial[0].content,
            user=initial[1].content,
            tool_specs=declarations,
        )
    except GenerationRejected:
        raise
    except ValueError:
        raise GenerationRejected("provider_turn_invalid") from None
    if not isinstance(turn, AgentTurn):
        raise GenerationRejected("provider_turn_invalid")
    trajectory: list[Message] = []
    calls = 0

    while turn.tool_call is not None:
        call = turn.tool_call
        call_message = Message(role="tool_call", content=_tool_call_content(call))
        rejected_trajectory = (*trajectory, call_message)
        if event_sink is not None:
            event_sink(
                {
                    "event": "tool_call_received",
                    "sample_id": sample.sample_id,
                    "tool_name": call.name,
                    "call_index": calls + 1,
                }
            )
        if call.name not in sample.tool_names:
            raise GenerationRejected(
                "tool_not_enabled",
                trajectory=rejected_trajectory,
            )
        if calls >= max_tool_calls:
            raise GenerationRejected(
                "tool_call_limit",
                trajectory=rejected_trajectory,
            )
        if sample.tool_policy == "required":
            if calls >= len(sample.tool_names):
                raise GenerationRejected(
                    "required_tool_extra",
                    trajectory=rejected_trajectory,
                )
            if call.name != sample.tool_names[calls]:
                raise GenerationRejected(
                    "required_tool_order",
                    trajectory=rejected_trajectory,
                )
        result = environment.execute(call)
        if event_sink is not None:
            event_sink(
                {
                    "event": "tool_result_created",
                    "sample_id": sample.sample_id,
                    "tool_name": call.name,
                    "call_index": calls + 1,
                    "status": result.status,
                }
            )
        trajectory.extend(
            (
                call_message,
                Message(role="tool_response", content=result.to_content()),
            )
        )
        calls += 1
        if result.status == "error":
            raise GenerationRejected(
                "tool_arguments_invalid",
                trajectory=tuple(trajectory),
            )
        if sample.tool_policy == "required" and result.status != "ok":
            raise GenerationRejected(
                "required_tool_unsuccessful",
                trajectory=tuple(trajectory),
            )
        try:
            turn = provider.continue_with_tool_result(call=call, result=result)
        except GenerationRejected:
            raise
        except ValueError:
            raise GenerationRejected(
                "provider_turn_invalid",
                trajectory=tuple(trajectory),
            ) from None
        if not isinstance(turn, AgentTurn):
            raise GenerationRejected(
                "provider_turn_invalid",
                trajectory=tuple(trajectory),
            )

    completion = turn.content
    if completion is None:
        raise ValueError("agent did not produce a final completion")
    if sample.tool_policy == "required" and calls != len(sample.tool_names):
        raise GenerationRejected(
            "required_tool_missing",
            candidate=completion,
            trajectory=tuple(trajectory),
        )
    if event_sink is not None:
        event_sink(
            {
                "event": "candidate_received",
                "sample_id": sample.sample_id,
                "thinking_type": sample.thinking_type,
                "tool_call_count": calls,
            }
        )
    try:
        parsed = _validate_candidate(completion, policy=policy, sample=sample)
    except ValueError as error:
        raise GenerationRejected(
            _validation_code(error),
            candidate=completion,
            trajectory=tuple(trajectory),
        ) from None
    semantic_code = _semantic_mismatch(parsed, policy=policy, sample=sample)
    if semantic_code is not None:
        raise GenerationRejected(
            semantic_code,
            candidate=completion,
            trajectory=tuple(trajectory),
        )
    return GeneratedExample(
        initial_messages=initial,
        trajectory=tuple(trajectory),
        completion=completion,
        parsed=parsed,
        tools_json=serialized_tools,
    )


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class _SafeEventLog:
    """Append-only operational events containing allowlisted metadata only."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.path.touch(exist_ok=True)
        except OSError:
            raise ValueError("cannot initialize event log") from None

    def write(self, payload: dict[str, object]) -> None:
        event = {
            "schema": "singguard-event-v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(
                        event,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
        except (OSError, TypeError, ValueError):
            raise RuntimeError("cannot write safe event log") from None


def _atomic_jsonl(path: Path, rows: Sequence[object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, path)


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sanitize_candidate(candidate: str | None, *, limit: int = 2048) -> str | None:
    if candidate is None:
        return None
    sanitized = re.sub(
        r"(?i)\b(api[_-]?key|access[_-]?token|token|secret)\s*[:=]\s*[^\s,;]+",
        lambda match: f"{match.group(1)}=[REDACTED]",
        candidate,
    )
    sanitized = re.sub(
        r"(?i)\bbearer\s+[a-z0-9._~+/=-]+",
        "Bearer [REDACTED]",
        sanitized,
    )
    sanitized = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[REDACTED_EMAIL]",
        sanitized,
        flags=re.IGNORECASE,
    )
    if len(sanitized) > limit:
        sanitized = sanitized[:limit] + "...[truncated]"
    return sanitized


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError(f"cannot read resume artifact {path.name}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"resume artifact {path.name} must contain an object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        raise ValueError(f"cannot read resume artifact {path.name}") from None
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"resume artifact {path.name} must contain objects")
    return rows


def _restore_budget(budget: TeacherBudget, payload: object) -> None:
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint has invalid budget accounting")
    request_count = payload.get("request_count")
    input_tokens = payload.get("input_tokens")
    output_tokens = payload.get("output_tokens")
    estimated_cost = payload.get("estimated_cost_usd")
    accounting_complete = payload.get("accounting_complete")
    if not (
        all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (request_count, input_tokens, output_tokens)
        )
        and isinstance(estimated_cost, (int, float))
        and not isinstance(estimated_cost, bool)
        and estimated_cost >= 0
        and isinstance(accounting_complete, bool)
    ):
        raise ValueError("resume checkpoint has invalid budget accounting")
    if budget.max_requests is not None and request_count > budget.max_requests:
        raise ValueError("resume request budget is below prior usage")
    if (
        budget.max_estimated_cost_usd is not None
        and float(estimated_cost) > budget.max_estimated_cost_usd
    ):
        raise ValueError("resume cost budget is below prior usage")
    budget.request_count = request_count
    budget.input_tokens = input_tokens
    budget.output_tokens = output_tokens
    budget.estimated_cost_usd = float(estimated_cost)
    budget.accounting_complete = accounting_complete


def _load_jsonl(path: Path, model: type[BaseModel], label: str) -> tuple[BaseModel, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        raise ValueError(f"cannot read {label} JSONL") from None
    records: list[BaseModel] = []
    try:
        for line in lines:
            if line.strip():
                records.append(model.model_validate_json(line))
    except (ValidationError, ValueError):
        raise ValueError(f"{label} JSONL contains an invalid record") from None
    if not records:
        raise ValueError(f"{label} JSONL is empty")
    return tuple(records)


def load_active_policies(path: Path) -> tuple[ActivePolicy, ...]:
    records = tuple(_load_jsonl(path, ActivePolicy, "active policy"))
    if len({record.policy_id for record in records}) != len(records):
        raise ValueError("active policy IDs must be unique")
    return records


def load_moderation_samples(path: Path) -> tuple[ModerationSample, ...]:
    records = tuple(_load_jsonl(path, ModerationSample, "content sample"))
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("sample IDs must be unique")
    return records


def run_generation_batch(
    *,
    policies: Sequence[ActivePolicy],
    samples: Sequence[ModerationSample],
    provider: AgentProvider,
    environment: ToolEnvironment,
    output_dir: Path,
    budget: TeacherBudget,
    max_tool_calls: int = 2,
    resume: bool = False,
    progress: Callable[[dict[str, object]], object] | None = None,
) -> dict[str, object]:
    """Generate one immutable ms-swift batch from reviewed prompts."""

    if output_dir.exists() and not resume:
        raise ValueError("output directory must not exist")
    if resume and not output_dir.is_dir():
        raise ValueError("resume output directory does not exist")
    policy_by_id = {policy.policy_id: policy for policy in policies}
    if len(policy_by_id) != len(policies):
        raise ValueError("active policy IDs must be unique")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("sample IDs must be unique")
    if any(sample.policy_id not in policy_by_id for sample in samples):
        raise ValueError("sample references an unknown active policy")
    if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int):
        raise ValueError("max_tool_calls must be an integer")
    if not 0 <= max_tool_calls <= 2:
        raise ValueError("max_tool_calls must be between zero and two")
    for sample in samples:
        if sample.tool_policy == "required" and len(sample.tool_names) > max_tool_calls:
            raise ValueError("required tool sequence exceeds the tool call limit")
        if sample.tool_names:
            tools_json(sample.tool_names)
        if sample.expected_label == "unsafe":
            policy_titles = tuple(
                rule.title for rule in policy_by_id[sample.policy_id].rules
            )
            expected_in_policy_order = tuple(
                title for title in policy_titles if title in sample.expected_answers
            )
            if expected_in_policy_order != sample.expected_answers:
                raise ValueError(
                    "expected answers must belong to the active policy in policy order"
                )
    diagnostic_metadata = getattr(provider, "diagnostic_metadata", None)
    if callable(diagnostic_metadata):
        provider_info = dict(diagnostic_metadata())
    else:
        provider_model = getattr(provider, "model", None)
        provider_info = {"name": type(provider).__name__}
        if isinstance(provider_model, str) and provider_model:
            provider_info["model"] = provider_model
    fingerprints: dict[str, object] = {
        "contract_version": "singguard-active-policy-v4",
        "policy_sha256": _fingerprint(
            [policy.model_dump(mode="json") for policy in policies]
        ),
        "sample_sha256": _fingerprint(
            [sample.model_dump(mode="json") for sample in samples]
        ),
        "tool_environment_sha256": environment.fingerprint,
        "prompt_sha256": _fingerprint(
            [
                build_initial_messages(policy_by_id[sample.policy_id], sample)[
                    0
                ].content
                for sample in samples
            ]
        ),
        "generation_sha256": _fingerprint(
            {
                "max_tool_calls": max_tool_calls,
                "provider": provider_info,
            }
        ),
    }
    if resume:
        checkpoint = _read_json(output_dir / "checkpoint.json")
        for name, expected in fingerprints.items():
            if checkpoint.get(name) != expected:
                raise ValueError(f"resume {name} does not match")
        completed = checkpoint.get("completed_samples")
        if not isinstance(completed, int) or not 0 <= completed <= len(samples):
            raise ValueError("resume checkpoint has invalid completed sample count")
        accepted_rows = _read_jsonl(output_dir / "train.jsonl")
        rejected_rows = _read_jsonl(output_dir / "rejected.jsonl")
        if len(accepted_rows) + len(rejected_rows) != completed:
            raise ValueError("resume artifact row count does not match checkpoint")
        repaired = checkpoint.get("repaired_samples", 0)
        if not isinstance(repaired, int) or not 0 <= repaired <= len(accepted_rows):
            raise ValueError("resume checkpoint has invalid repaired sample count")
        repair_attempts = checkpoint.get("repair_attempts", 0)
        if not isinstance(repair_attempts, int) or repair_attempts < repaired:
            raise ValueError("resume checkpoint has invalid repair attempt count")
        _restore_budget(budget, checkpoint.get("budget"))
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        accepted_rows = []
        rejected_rows = []
        completed = 0
        repaired = 0
        repair_attempts = 0
        _atomic_jsonl(output_dir / "train.jsonl", accepted_rows)
        _atomic_jsonl(output_dir / "rejected.jsonl", rejected_rows)
        _atomic_json(
            output_dir / "checkpoint.json",
            {
                "completed_samples": completed,
                "repaired_samples": repaired,
                "repair_attempts": repair_attempts,
                "budget": budget.as_dict(),
                **fingerprints,
            },
        )
    event_log = _SafeEventLog(output_dir / "events.jsonl")
    set_event_sink = getattr(provider, "set_event_sink", None)
    if callable(set_event_sink):
        set_event_sink(event_log.write)
    event_log.write(
        {
            "event": "batch_resumed" if resume else "batch_started",
            "planned_samples": len(samples),
            "completed_samples": completed,
            **provider_info,
        }
    )
    stopped_reason: str | None = None
    provider_failure: dict[str, object] | None = None

    def emit(phase: str, sample_id: str | None = None) -> None:
        if progress is None:
            return
        event: dict[str, object] = {
            "phase": phase,
            "completed": completed,
            "total": len(samples),
            "accepted": len(accepted_rows),
            "rejected": len(rejected_rows),
            "requests": budget.request_count,
        }
        if sample_id is not None:
            event["sample_id"] = sample_id
        try:
            progress(event)
        except Exception:
            pass

    emit("start")
    for sample in samples[completed:]:
        begin_sample = getattr(provider, "begin_sample", None)
        if callable(begin_sample):
            begin_sample(sample.sample_id)
        event_log.write(
            {
                "event": "sample_started",
                "sample_id": sample.sample_id,
                "thinking_type": sample.thinking_type,
                "tool_policy": sample.tool_policy,
                "configured_tool_count": len(sample.tool_names),
            }
        )
        accepted_before = len(accepted_rows)
        rejected_before = len(rejected_rows)
        sample_repaired = False
        emit("generate", sample.sample_id)
        try:
            generated = generate_example(
                policy_by_id[sample.policy_id],
                sample,
                provider,
                environment,
                max_tool_calls=max_tool_calls,
                event_sink=event_log.write,
            )
        except GenerationRejected as error:
            event_log.write(
                {
                    "event": "candidate_rejected",
                    "sample_id": sample.sample_id,
                    "code": error.code,
                    "repairable": error.code in _REPAIRABLE_CODES,
                    "tool_call_count": sum(
                        message.role == "tool_call" for message in error.trajectory
                    ),
                }
            )
            repair = getattr(provider, "repair", None)
            repaired_candidate: str | None = None
            repair_code: str | None = None
            if (
                error.code in _REPAIRABLE_CODES
                and callable(repair)
                and error.candidate is not None
            ):
                repair_attempts += 1
                initial = build_initial_messages(
                    policy_by_id[sample.policy_id], sample
                )
                try:
                    repair_turn = repair(
                        system=initial[0].content,
                        user=initial[1].content,
                        candidate=error.candidate,
                        validation_code=error.code,
                    )
                    if not isinstance(repair_turn, AgentTurn):
                        raise ValueError("repair provider returned an invalid turn")
                    if repair_turn.tool_call is not None or repair_turn.content is None:
                        raise ValueError("repair must return final text without tools")
                    repaired_candidate = repair_turn.content
                    repaired_parsed = _validate_candidate(
                        repaired_candidate,
                        policy=policy_by_id[sample.policy_id],
                        sample=sample,
                    )
                    repair_code = _semantic_mismatch(
                        repaired_parsed,
                        policy=policy_by_id[sample.policy_id],
                        sample=sample,
                    )
                except TeacherBudgetExceeded as budget_error:
                    stopped_reason = budget_error.reason
                    break
                except TeacherRequestError as request_error:
                    stopped_reason = "provider_error"
                    provider_failure = (
                        request_error.diagnostic.as_dict()
                        if request_error.diagnostic is not None
                        else {
                            "stage": "unknown",
                            "exception_type": "TeacherRequestError",
                            "http_code": None,
                            "provider_status": None,
                            "attempts": 0,
                        }
                    )
                    event_log.write(
                        {
                            "event": "batch_stopped",
                            "sample_id": sample.sample_id,
                            "reason": stopped_reason,
                            **provider_failure,
                        }
                    )
                    break
                except ValueError as repair_error:
                    repair_code = _repair_failure_code(repair_error)
                else:
                    if repair_code is None:
                        accepted_rows.append(
                            render_training_row(
                                initial,
                                repaired_candidate,
                                tools_json=(
                                    tools_json(sample.tool_names)
                                    if sample.tool_names
                                    else None
                                ),
                                trajectory=error.trajectory,
                            )
                        )
                        repaired += 1
                        sample_repaired = True
                        event_log.write(
                            {
                                "event": "repair_succeeded",
                                "sample_id": sample.sample_id,
                            }
                        )
                if repair_code is not None:
                    event_log.write(
                        {
                            "event": "repair_failed",
                            "sample_id": sample.sample_id,
                            "code": repair_code,
                            "candidate_present": repaired_candidate is not None,
                        }
                    )
            if repaired_candidate is None or repair_code is not None:
                codes = [error.code]
                if repair_code is not None:
                    codes.append(f"repair_{repair_code}")
                rejected_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "codes": codes,
                        "candidate": _sanitize_candidate(error.candidate),
                        "repaired_candidate": _sanitize_candidate(repaired_candidate),
                        "trajectory": [
                            message.model_dump(mode="json")
                            for message in error.trajectory
                        ],
                    }
                )
        except TeacherBudgetExceeded as error:
            stopped_reason = error.reason
            event_log.write(
                {
                    "event": "batch_stopped",
                    "sample_id": sample.sample_id,
                    "reason": stopped_reason,
                }
            )
            break
        except TeacherRequestError as error:
            stopped_reason = "provider_error"
            provider_failure = (
                error.diagnostic.as_dict()
                if error.diagnostic is not None
                else {
                    "stage": "unknown",
                    "exception_type": "TeacherRequestError",
                    "http_code": None,
                    "provider_status": None,
                    "attempts": 0,
                }
            )
            event_log.write(
                {
                    "event": "batch_stopped",
                    "sample_id": sample.sample_id,
                    "reason": stopped_reason,
                    **provider_failure,
                }
            )
            break
        else:
            accepted_rows.append(
                render_training_row(
                    generated.initial_messages,
                    generated.completion,
                    tools_json=generated.tools_json,
                    trajectory=generated.trajectory,
                )
            )
        if len(accepted_rows) > accepted_before:
            event_log.write(
                {
                    "event": "sample_accepted",
                    "sample_id": sample.sample_id,
                    "repaired": sample_repaired,
                }
            )
        elif len(rejected_rows) > rejected_before:
            event_log.write(
                {
                    "event": "sample_rejected",
                    "sample_id": sample.sample_id,
                    "codes": rejected_rows[-1].get("codes", []),
                }
            )
        completed += 1
        _atomic_jsonl(output_dir / "train.jsonl", accepted_rows)
        _atomic_jsonl(output_dir / "rejected.jsonl", rejected_rows)
        _atomic_json(
            output_dir / "checkpoint.json",
            {
                "completed_samples": completed,
                "repaired_samples": repaired,
                "repair_attempts": repair_attempts,
                "budget": budget.as_dict(),
                **fingerprints,
            },
        )
        emit("sample_complete", sample.sample_id)

    status = "complete" if stopped_reason is None and completed == len(samples) else "incomplete"
    rejected_ids = {
        row["sample_id"]
        for row in rejected_rows
        if isinstance(row.get("sample_id"), str)
    }
    accepted_samples = tuple(
        sample for sample in samples[:completed] if sample.sample_id not in rejected_ids
    )
    if len(accepted_samples) != len(accepted_rows):
        raise RuntimeError("accepted and rejected batch accounting is inconsistent")
    expected_samples = sum(
        sample.expected_label is not None for sample in samples[:completed]
    )
    expected_accepted = sum(
        sample.expected_label is not None for sample in accepted_samples
    )
    required_tool_samples = sum(
        sample.tool_policy == "required" for sample in samples[:completed]
    )
    required_tool_accepted = sum(
        sample.tool_policy == "required" for sample in accepted_samples
    )
    if status != "complete":
        quality_status = "incomplete"
    elif expected_samples == 0:
        quality_status = "not_evaluated"
    elif (
        expected_accepted == expected_samples
        and required_tool_accepted == required_tool_samples
    ):
        quality_status = "pass"
    else:
        quality_status = "fail"
    accepted_tool_call_count = sum(
        message.get("role") == "tool_call"
        for row in accepted_rows
        for message in row.get("messages", [])
        if isinstance(message, dict)
    )
    rejected_tool_call_count = sum(
        message.get("role") == "tool_call"
        for row in rejected_rows
        for message in row.get("trajectory", [])
        if isinstance(message, dict)
    )
    manifest: dict[str, object] = {
        "schema": "singguard-active-policy-batch-v2",
        "status": status,
        "reason": stopped_reason,
        "provider": provider_info,
        "event_log": "events.jsonl",
        "planned_samples": len(samples),
        "completed_samples": completed,
        "accepted_samples": len(accepted_rows),
        "repaired_samples": repaired,
        "repair_attempts": repair_attempts,
        "rejected_samples": len(rejected_rows),
        "thinking_type_counts": {
            mode: sum(sample.thinking_type == mode for sample in samples[:completed])
            for mode in ("fast", "slow")
        },
        "accepted_thinking_type_counts": {
            mode: sum(sample.thinking_type == mode for sample in accepted_samples)
            for mode in ("fast", "slow")
        },
        "required_tool_samples": required_tool_samples,
        "required_tool_accepted": required_tool_accepted,
        "attempted_tool_call_count": (
            accepted_tool_call_count + rejected_tool_call_count
        ),
        "accepted_tool_call_count": accepted_tool_call_count,
        "quality_gate": {
            "status": quality_status,
            "expected_samples": expected_samples,
            "expected_accepted": expected_accepted,
            "required_tool_samples": required_tool_samples,
            "required_tool_accepted": required_tool_accepted,
        },
        "budget": budget.as_dict(),
    }
    if provider_failure is not None:
        manifest["provider_failure"] = provider_failure
    _atomic_json(output_dir / "manifest.json", manifest)
    _atomic_json(
        output_dir / "checkpoint.json",
        {
            "completed_samples": completed,
            "repaired_samples": repaired,
            "repair_attempts": repair_attempts,
            "budget": budget.as_dict(),
            **fingerprints,
        },
    )
    event_log.write(
        {
            "event": "batch_complete" if status == "complete" else "batch_incomplete",
            "reason": stopped_reason,
            "completed_samples": completed,
            "accepted_samples": len(accepted_rows),
            "rejected_samples": len(rejected_rows),
            "request_count": budget.request_count,
        }
    )
    emit("complete" if status == "complete" else "incomplete")
    return manifest
