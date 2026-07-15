"""Generate one SingGuard SFT example by executing real sequential tool calls."""

from __future__ import annotations

import json
import hashlib
import os
import re
import time
from collections.abc import Callable, Sequence
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
from risk_agent.singguard_prompts import prompt_sha256
from risk_agent.teacher import (
    GeminiTeacher,
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
    "slow_rule_order",
    "completion_validation",
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
        self._chat: object | None = None
        self._context_text = ""

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

    def _send(self, message: object) -> object:
        if self._chat is None:
            raise RuntimeError("Gemini agent chat has not been started")
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
            try:
                response = self._chat.send_message(message)
            except Exception:
                usage = aggregate.add(None, None)
                if self._budget is not None:
                    self._budget.record_attempt(
                        TeacherUsage(
                            provider="gemini",
                            model=self.model,
                            accounting_complete=False,
                        ),
                        reserved_cost_usd=reservation,
                    )
                if attempt < self._max_attempts:
                    self._sleep(self._retry_delay(attempt))
                    continue
                raise TeacherRequestError(
                    f"Gemini agent request failed after {self._max_attempts} attempts",
                    usage,
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
            client = self._new_client()
            self._chat = client.chats.create(
                model=self.model,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception:
            raise TeacherRequestError(
                "Gemini agent client initialization failed",
                TeacherUsage(
                    provider="gemini",
                    model=self.model,
                    request_count=0,
                    accounting_complete=False,
                ),
            ) from None
        self._context_text = f"{system}\n{user}"
        response = self._send(user)
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
        response = self._send(part)
        return self._to_turn(response)

    def repair(
        self,
        *,
        system: str,
        user: str,
        candidate: str,
        validation_code: str,
    ) -> AgentTurn:
        """Make one fresh, tool-free request that only repairs output grammar."""

        types = self._types()
        repair_instruction = (
            "\n\n# Output Repair\n"
            "The previous candidate failed deterministic output validation. The only "
            "permitted changes are output serialization and the minimum wording needed "
            "to fill the exact Output Format structure above. Do not change the "
            "moderation label or triggered rule. Do not call tools, add commentary, "
            "omit required steps, or introduce policy categories that are not active."
        )
        repair_user = (
            f"{user}\n\n[validation_error]: {validation_code}"
            f"\n[rejected_candidate]:\n{candidate}"
        )
        try:
            client = self._new_client()
            self._chat = client.chats.create(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=system + repair_instruction,
                    temperature=0.0,
                    max_output_tokens=self._max_output_tokens,
                ),
            )
        except Exception:
            raise TeacherRequestError(
                "Gemini repair client initialization failed",
                TeacherUsage(
                    provider="gemini",
                    model=self.model,
                    request_count=0,
                    accounting_complete=False,
                ),
            ) from None
        self._context_text = f"{system}{repair_instruction}\n{repair_user}"
        return self._to_turn(self._send(repair_user))


def _tool_call_content(call: ToolCall) -> str:
    return json.dumps(
        {"name": call.name, "arguments": call.arguments},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validation_code(error: ValueError) -> str:
    message = str(error)
    if "active policy" in message:
        return "inactive_answer"
    if "policy order" in message:
        return "slow_rule_order"
    if "grammar" in message:
        return "output_grammar"
    return "completion_validation"


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
    turn = provider.start(
        system=initial[0].content,
        user=initial[1].content,
        tool_specs=declarations,
    )
    trajectory: list[Message] = []
    calls = 0

    while turn.tool_call is not None:
        call = turn.tool_call
        if call.name not in sample.tool_names:
            raise ValueError("requested tool is not enabled for this sample")
        if calls >= max_tool_calls:
            raise ValueError("tool call limit exceeded")
        if sample.tool_policy == "required":
            if calls >= len(sample.tool_names):
                raise GenerationRejected(
                    "required_tool_extra",
                    trajectory=tuple(trajectory),
                )
            if call.name != sample.tool_names[calls]:
                raise GenerationRejected(
                    "required_tool_order",
                    trajectory=tuple(trajectory),
                )
        result = environment.execute(call)
        trajectory.extend(
            (
                Message(role="tool_call", content=_tool_call_content(call)),
                Message(role="tool_response", content=result.to_content()),
            )
        )
        calls += 1
        turn = provider.continue_with_tool_result(call=call, result=result)

    completion = turn.content
    if completion is None:
        raise ValueError("agent did not produce a final completion")
    if sample.tool_policy == "required" and calls != len(sample.tool_names):
        raise GenerationRejected(
            "required_tool_missing",
            candidate=completion,
            trajectory=tuple(trajectory),
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
    fingerprints: dict[str, object] = {
        "policy_sha256": _fingerprint(
            [policy.model_dump(mode="json") for policy in policies]
        ),
        "sample_sha256": _fingerprint(
            [sample.model_dump(mode="json") for sample in samples]
        ),
        "tool_environment_sha256": environment.fingerprint,
        "prompt_sha256": prompt_sha256("guard"),
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
        repaired = checkpoint.get("repaired_samples", 0)
        if not isinstance(repaired, int) or not 0 <= repaired <= len(accepted_rows):
            raise ValueError("resume checkpoint has invalid repaired sample count")
        _restore_budget(budget, checkpoint.get("budget"))
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        accepted_rows = []
        rejected_rows = []
        completed = 0
        repaired = 0
        _atomic_jsonl(output_dir / "train.jsonl", accepted_rows)
        _atomic_jsonl(output_dir / "rejected.jsonl", rejected_rows)
        _atomic_json(
            output_dir / "checkpoint.json",
            {
                "completed_samples": completed,
                "repaired_samples": repaired,
                "budget": budget.as_dict(),
                **fingerprints,
            },
        )
    stopped_reason: str | None = None

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
        emit("generate", sample.sample_id)
        try:
            generated = generate_example(
                policy_by_id[sample.policy_id],
                sample,
                provider,
                environment,
                max_tool_calls=max_tool_calls,
            )
        except GenerationRejected as error:
            repair = getattr(provider, "repair", None)
            repaired_candidate: str | None = None
            repair_code: str | None = None
            if (
                error.code in _REPAIRABLE_CODES
                and callable(repair)
                and error.candidate is not None
            ):
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
                except TeacherRequestError:
                    stopped_reason = "provider_error"
                    break
                except ValueError as repair_error:
                    repair_code = _validation_code(repair_error)
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
            break
        except TeacherRequestError:
            stopped_reason = "provider_error"
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
        completed += 1
        _atomic_jsonl(output_dir / "train.jsonl", accepted_rows)
        _atomic_jsonl(output_dir / "rejected.jsonl", rejected_rows)
        _atomic_json(
            output_dir / "checkpoint.json",
            {
                "completed_samples": completed,
                "repaired_samples": repaired,
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
    manifest: dict[str, object] = {
        "schema": "singguard-active-policy-batch-v1",
        "status": status,
        "reason": stopped_reason,
        "planned_samples": len(samples),
        "completed_samples": completed,
        "accepted_samples": len(accepted_rows),
        "repaired_samples": repaired,
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
        "tool_call_count": sum(
            message["role"] == "tool_call"
            for row in accepted_rows
            for message in row["messages"]
        ),
        "quality_gate": {
            "status": quality_status,
            "expected_samples": expected_samples,
            "expected_accepted": expected_accepted,
            "required_tool_samples": required_tool_samples,
            "required_tool_accepted": required_tool_accepted,
        },
        "budget": budget.as_dict(),
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    _atomic_json(
        output_dir / "checkpoint.json",
        {
            "completed_samples": completed,
            "repaired_samples": repaired,
            "budget": budget.as_dict(),
            **fingerprints,
        },
    )
    emit("complete" if status == "complete" else "incomplete")
    return manifest
