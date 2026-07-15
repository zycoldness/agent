"""Gemini-independent tests for real sequential tool trajectories."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from risk_agent.contracts import PolicyRule
from risk_agent.singguard import ActivePolicy, ModerationSample
from risk_agent.singguard_tools import ToolCall, ToolEnvironment, ToolResult


class FakeAgentProvider:
    def __init__(self, turns: list[object]) -> None:
        self._turns = deque(turns)
        self.received_results: list[dict[str, object]] = []

    def start(
        self,
        *,
        system: str,
        user: str,
        tool_specs: tuple[dict[str, object], ...],
    ) -> object:
        assert system.startswith("# Task")
        assert user.startswith("[user]:")
        return self._turns.popleft()

    def continue_with_tool_result(
        self,
        *,
        call: ToolCall,
        result: ToolResult,
    ) -> object:
        self.received_results.append(
            {"status": result.status, **result.payload}
        )
        return self._turns.popleft()


def _policy() -> ActivePolicy:
    return ActivePolicy(
        policy_id="commerce-v1",
        rules=(
            PolicyRule(
                rule_id="LEAD-001",
                title="Off-Platform Solicitation",
                text="Do not redirect users to prohibited private channels.",
            ),
        ),
    )


def _environment() -> ToolEnvironment:
    return ToolEnvironment.load(Path("data/tool_env"))


def test_agent_executes_two_tools_then_records_final_completion() -> None:
    from risk_agent.singguard_generation import AgentTurn, generate_example

    sample = ModerationSample(
        sample_id="sample-1",
        policy_id="commerce-v1",
        thinking_type="fast",
        query="Contact w-h-a-t-s-a-p-p:user123 for the private price.",
        tool_names=("inspect_destination", "get_content_context"),
    )
    provider = FakeAgentProvider(
        turns=[
            AgentTurn(
                tool_call=ToolCall(
                    name="inspect_destination",
                    arguments={"indicator": "w-h-a-t-s-a-p-p:user123"},
                )
            ),
            AgentTurn(
                tool_call=ToolCall(
                    name="get_content_context",
                    arguments={"content_id": "content-0001"},
                )
            ),
            AgentTurn(
                content="unsafe\n<answer>Off-Platform Solicitation</answer>"
            ),
        ]
    )

    generated = generate_example(
        _policy(),
        sample,
        provider,
        _environment(),
        max_tool_calls=2,
    )

    assert [message.role for message in generated.trajectory] == [
        "tool_call",
        "tool_response",
        "tool_call",
        "tool_response",
    ]
    assert generated.completion.startswith("unsafe")
    assert provider.received_results[0]["status"] == "ok"
    assert provider.received_results[1]["result"]["content_id"] == "content-0001"


def test_agent_accepts_direct_no_tool_completion() -> None:
    from risk_agent.singguard_generation import AgentTurn, generate_example

    sample = ModerationSample(
        sample_id="sample-2",
        policy_id="commerce-v1",
        thinking_type="fast",
        query="This is a neutral statement.",
    )
    provider = FakeAgentProvider(
        turns=[AgentTurn(content="safe\n<answer>Safe</answer>")]
    )

    generated = generate_example(
        _policy(),
        sample,
        provider,
        _environment(),
    )

    assert generated.trajectory == ()
    assert generated.parsed.label == "safe"


def test_agent_rejects_tool_not_enabled_for_sample() -> None:
    from risk_agent.singguard_generation import AgentTurn, generate_example

    sample = ModerationSample(
        sample_id="sample-3",
        policy_id="commerce-v1",
        thinking_type="fast",
        query="Example.",
        tool_names=("search_cases",),
    )
    provider = FakeAgentProvider(
        turns=[
            AgentTurn(
                tool_call=ToolCall(
                    name="inspect_destination",
                    arguments={"indicator": "example.test"},
                )
            )
        ]
    )

    with pytest.raises(ValueError, match="not enabled"):
        generate_example(_policy(), sample, provider, _environment())


def test_agent_rejects_third_tool_call() -> None:
    from risk_agent.singguard_generation import AgentTurn, generate_example

    sample = ModerationSample(
        sample_id="sample-4",
        policy_id="commerce-v1",
        thinking_type="fast",
        query="Example.",
        tool_names=("search_cases",),
    )
    call = ToolCall(
        name="search_cases",
        arguments={"query": "private messaging"},
    )
    provider = FakeAgentProvider(
        turns=[
            AgentTurn(tool_call=call),
            AgentTurn(tool_call=call),
            AgentTurn(tool_call=call),
        ]
    )

    with pytest.raises(ValueError, match="tool call limit"):
        generate_example(
            _policy(),
            sample,
            provider,
            _environment(),
            max_tool_calls=2,
        )


def test_gemini_adapter_maps_function_call_and_response() -> None:
    from risk_agent.singguard_generation import GeminiAgentProvider
    from risk_agent.teacher import TeacherBudget

    responses = deque(
        [
            SimpleNamespace(
                function_calls=[
                    SimpleNamespace(
                        name="search_cases",
                        args={"query": "guaranteed claim"},
                    )
                ],
                text=None,
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10,
                    candidates_token_count=3,
                ),
            ),
            SimpleNamespace(
                function_calls=[],
                text="safe\n<answer>Safe</answer>",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=12,
                    candidates_token_count=4,
                ),
            ),
        ]
    )
    sent: list[object] = []
    captured: dict[str, object] = {}

    class Chat:
        def send_message(self, message):
            sent.append(message)
            return responses.popleft()

    class Chats:
        def create(self, **kwargs):
            captured.update(kwargs)
            return Chat()

    class Part:
        @staticmethod
        def from_function_response(*, name, response):
            return {"name": name, "response": response}

    types_module = SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: kwargs,
        Tool=lambda **kwargs: kwargs,
        Part=Part,
    )
    budget = TeacherBudget(max_requests=2)
    provider = GeminiAgentProvider(
        model="gemini-test",
        client_factory=lambda: SimpleNamespace(chats=Chats()),
        types_module=types_module,
        budget=budget,
        max_attempts=1,
    )

    first = provider.start(
        system="system prompt",
        user="[user]: content",
        tool_specs=(
            {
                "name": "search_cases",
                "description": "Search cases.",
                "parameters": {"type": "object", "properties": {}},
            },
        ),
    )
    second = provider.continue_with_tool_result(
        call=first.tool_call,
        result=ToolResult(status="ok", payload={"results": []}),
    )

    assert first.tool_call.name == "search_cases"
    assert second.content == "safe\n<answer>Safe</answer>"
    assert sent[-1] == {
        "name": "search_cases",
        "response": {"status": "ok", "results": []},
    }
    assert captured["config"]["system_instruction"] == "system prompt"
    assert budget.as_dict()["request_count"] == 2
