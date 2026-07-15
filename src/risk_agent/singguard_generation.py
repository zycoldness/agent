"""Generate one SingGuard SFT example by executing real sequential tool calls."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, ConfigDict, model_validator

from risk_agent.singguard import (
    ActivePolicy,
    Message,
    ModerationSample,
    ParsedCompletion,
    build_initial_messages,
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
    TeacherBudget,
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


def _tool_call_content(call: ToolCall) -> str:
    return json.dumps(
        {"name": call.name, "arguments": call.arguments},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


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
    parsed = validate_completion(
        completion,
        thinking_type=sample.thinking_type,
        active_titles=tuple(rule.title for rule in policy.rules),
    )
    return GeneratedExample(
        initial_messages=initial,
        trajectory=tuple(trajectory),
        completion=completion,
        parsed=parsed,
        tools_json=serialized_tools,
    )
