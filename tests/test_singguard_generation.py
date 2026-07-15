"""Gemini-independent tests for real sequential tool trajectories."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from risk_agent.contracts import PolicyRule
from risk_agent.singguard import ActivePolicy, ModerationSample
from risk_agent.singguard_tools import ToolCall, ToolEnvironment, ToolResult
from risk_agent.teacher import TeacherBudgetExceeded


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


def test_bundled_smoke_plan_has_complete_hidden_oracles_and_tool_targets() -> None:
    from risk_agent.singguard_generation import load_moderation_samples

    samples = load_moderation_samples(Path("data/content_samples.jsonl"))

    assert len(samples) == 6
    assert all(sample.expected_label is not None for sample in samples)
    assert sum(sample.thinking_type == "slow" for sample in samples) == 4
    assert sum(sample.tool_policy == "required" for sample in samples) == 3
    assert all(
        sample.tool_names for sample in samples if sample.tool_policy == "required"
    )


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


def test_agent_rejects_final_answer_before_required_tool_call() -> None:
    from risk_agent.singguard_generation import (
        AgentTurn,
        GenerationRejected,
        generate_example,
    )

    sample = ModerationSample(
        sample_id="sample-required",
        policy_id="commerce-v1",
        thinking_type="fast",
        query="Message me privately.",
        tool_names=("inspect_destination",),
        tool_policy="required",
    )
    provider = FakeAgentProvider(
        turns=[
            AgentTurn(
                content="unsafe\n<answer>Off-Platform Solicitation</answer>"
            )
        ]
    )

    with pytest.raises(GenerationRejected) as captured:
        generate_example(_policy(), sample, provider, _environment())

    assert captured.value.code == "required_tool_missing"


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


def test_gemini_repair_receives_exact_format_only_contract() -> None:
    from risk_agent.singguard_generation import GeminiAgentProvider
    from risk_agent.singguard_prompts import render_guard_prompt
    from risk_agent.teacher import TeacherBudget

    captured: dict[str, object] = {}
    sent: list[object] = []

    class Chat:
        def send_message(self, message):
            sent.append(message)
            return SimpleNamespace(
                function_calls=[],
                text="safe\n<answer>Safe</answer>",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=8,
                    candidates_token_count=3,
                ),
            )

    class Chats:
        def create(self, **kwargs):
            captured.update(kwargs)
            return Chat()

    types_module = SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: kwargs,
    )
    provider = GeminiAgentProvider(
        model="gemini-test",
        client_factory=lambda: SimpleNamespace(chats=Chats()),
        types_module=types_module,
        budget=TeacherBudget(max_requests=1),
        max_attempts=1,
    )
    system = render_guard_prompt(_policy().rules, thinking_type="fast")

    provider.repair(
        system=system,
        user="[user]: neutral",
        candidate="The content is safe.",
        validation_code="output_grammar",
    )

    repair_system = captured["config"]["system_instruction"]
    assert "The only permitted changes are output serialization" in repair_system
    assert "Do not change the moderation label or triggered rule" in repair_system
    assert "[validation_error]: output_grammar" in sent[0]
    assert "[rejected_candidate]:\nThe content is safe." in sent[0]
    assert "tools" not in captured["config"]


def test_batch_writes_accepted_rows_and_specific_rejection(tmp_path: Path) -> None:
    from risk_agent.singguard_generation import AgentTurn, run_generation_batch
    from risk_agent.teacher import TeacherBudget

    samples = (
        ModerationSample(
            sample_id="accepted-safe",
            policy_id="commerce-v1",
            thinking_type="fast",
            query="Neutral content.",
        ),
        ModerationSample(
            sample_id="accepted-unsafe",
            policy_id="commerce-v1",
            thinking_type="fast",
            query="Message me privately.",
        ),
        ModerationSample(
            sample_id="rejected-inactive",
            policy_id="commerce-v1",
            thinking_type="fast",
            query="Unknown category.",
        ),
    )
    provider = FakeAgentProvider(
        turns=[
            AgentTurn(content="safe\n<answer>Safe</answer>"),
            AgentTurn(
                content="unsafe\n<answer>Off-Platform Solicitation</answer>"
            ),
            AgentTurn(content="unsafe\n<answer>Unknown Rule</answer>"),
        ]
    )
    output = tmp_path / "batch"

    manifest = run_generation_batch(
        policies=(_policy(),),
        samples=samples,
        provider=provider,
        environment=_environment(),
        output_dir=output,
        budget=TeacherBudget(max_requests=10),
    )

    train_rows = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    rejected_rows = [json.loads(line) for line in (output / "rejected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["planned_samples"] == 3
    assert manifest["accepted_samples"] == 2
    assert manifest["rejected_samples"] == 1
    assert train_rows[0]["messages"][-1]["role"] == "assistant"
    assert rejected_rows[0]["sample_id"] == "rejected-inactive"
    assert rejected_rows[0]["codes"] == ["inactive_answer"]


def test_incomplete_batch_resumes_without_repeating_completed_sample(tmp_path: Path) -> None:
    from risk_agent.singguard_generation import AgentTurn, run_generation_batch
    from risk_agent.teacher import TeacherBudget

    samples = (
        ModerationSample(
            sample_id="first",
            policy_id="commerce-v1",
            thinking_type="fast",
            query="First.",
        ),
        ModerationSample(
            sample_id="second",
            policy_id="commerce-v1",
            thinking_type="fast",
            query="Second.",
        ),
    )

    class StopAfterOne(FakeAgentProvider):
        def start(self, **kwargs):
            if not self._turns:
                raise TeacherBudgetExceeded("max_requests")
            return super().start(**kwargs)

    output = tmp_path / "resumable"
    first_manifest = run_generation_batch(
        policies=(_policy(),),
        samples=samples,
        provider=StopAfterOne(
            turns=[AgentTurn(content="safe\n<answer>Safe</answer>")]
        ),
        environment=_environment(),
        output_dir=output,
        budget=TeacherBudget(max_requests=10),
    )
    second_manifest = run_generation_batch(
        policies=(_policy(),),
        samples=samples,
        provider=FakeAgentProvider(
            turns=[AgentTurn(content="safe\n<answer>Safe</answer>")]
        ),
        environment=_environment(),
        output_dir=output,
        budget=TeacherBudget(max_requests=10),
        resume=True,
    )

    train_rows = (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert first_manifest["status"] == "incomplete"
    assert first_manifest["completed_samples"] == 1
    assert second_manifest["status"] == "complete"
    assert len(train_rows) == 2


def test_batch_repairs_one_invalid_completion_without_replaying_tools(tmp_path: Path) -> None:
    from risk_agent.singguard_generation import AgentTurn, run_generation_batch
    from risk_agent.teacher import TeacherBudget

    sample = ModerationSample(
        sample_id="repair-me",
        policy_id="commerce-v1",
        thinking_type="fast",
        query="Message me privately.",
    )

    class RepairingProvider(FakeAgentProvider):
        def __init__(self) -> None:
            super().__init__([AgentTurn(content="The content is unsafe.")])
            self.repair_calls: list[dict[str, str]] = []

        def repair(self, *, system, user, candidate, validation_code):
            self.repair_calls.append(
                {
                    "system": system,
                    "user": user,
                    "candidate": candidate,
                    "validation_code": validation_code,
                }
            )
            return AgentTurn(
                content="unsafe\n<answer>Off-Platform Solicitation</answer>"
            )

    provider = RepairingProvider()
    output = tmp_path / "repair"
    manifest = run_generation_batch(
        policies=(_policy(),),
        samples=(sample,),
        provider=provider,
        environment=_environment(),
        output_dir=output,
        budget=TeacherBudget(max_requests=10),
    )

    row = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    assert manifest["accepted_samples"] == 1
    assert manifest["repaired_samples"] == 1
    assert row["messages"][-1]["content"].startswith("unsafe")
    assert provider.repair_calls[0]["candidate"] == "The content is unsafe."
    assert provider.repair_calls[0]["validation_code"] == "output_grammar"


def test_batch_rejects_semantic_label_mismatch_without_format_repair(tmp_path: Path) -> None:
    from risk_agent.singguard_generation import AgentTurn, run_generation_batch
    from risk_agent.teacher import TeacherBudget

    sample = ModerationSample(
        sample_id="known-unsafe",
        policy_id="commerce-v1",
        thinking_type="fast",
        query="Message me privately to place the order.",
        expected_label="unsafe",
        expected_answers=("Off-Platform Solicitation",),
    )

    class SemanticProvider(FakeAgentProvider):
        def repair(self, **kwargs):
            raise AssertionError("semantic mismatches must not be format-repaired")

    output = tmp_path / "semantic-reject"
    manifest = run_generation_batch(
        policies=(_policy(),),
        samples=(sample,),
        provider=SemanticProvider(
            [AgentTurn(content="safe\n<answer>Safe</answer>")]
        ),
        environment=_environment(),
        output_dir=output,
        budget=TeacherBudget(max_requests=10),
    )

    rejected = json.loads(
        (output / "rejected.jsonl").read_text(encoding="utf-8")
    )
    assert manifest["accepted_samples"] == 0
    assert rejected["codes"] == ["semantic_label_mismatch"]
    assert "expected_label" not in rejected["candidate"]
    assert manifest["quality_gate"]["status"] == "fail"


def test_rejected_candidate_is_redacted_and_truncated(tmp_path: Path) -> None:
    from risk_agent.singguard_generation import AgentTurn, run_generation_batch
    from risk_agent.teacher import TeacherBudget

    sample = ModerationSample(
        sample_id="reject-secret",
        policy_id="commerce-v1",
        thinking_type="fast",
        query="Neutral.",
    )
    candidate = "api_key=top-secret user@example.com " + "x" * 5000
    output = tmp_path / "reject"
    run_generation_batch(
        policies=(_policy(),),
        samples=(sample,),
        provider=FakeAgentProvider([AgentTurn(content=candidate)]),
        environment=_environment(),
        output_dir=output,
        budget=TeacherBudget(max_requests=10),
    )

    rejected = json.loads(
        (output / "rejected.jsonl").read_text(encoding="utf-8")
    )
    assert "top-secret" not in rejected["candidate"]
    assert "user@example.com" not in rejected["candidate"]
    assert rejected["candidate"].endswith("...[truncated]")


def test_manifest_counts_accepted_modes_and_required_tool_coverage(tmp_path: Path) -> None:
    from risk_agent.singguard_generation import AgentTurn, run_generation_batch
    from risk_agent.teacher import TeacherBudget

    sample = ModerationSample(
        sample_id="required-slow",
        policy_id="commerce-v1",
        thinking_type="slow",
        query="Contact w-h-a-t-s-a-p-p:user123.",
        tool_names=("inspect_destination",),
        tool_policy="required",
        expected_label="unsafe",
        expected_answers=("Off-Platform Solicitation",),
    )
    completion = (
        "unsafe\n<reasoning>\n"
        "[Step 1] Content Summary\nA private destination is supplied.\n\n"
        "[Step 2] Check Risk Categories\n"
        "- Off-Platform Solicitation: HIT. The destination is off platform.\n\n"
        "[Step 3] Final Judgment\nThe active rule is violated.\n"
        "</reasoning>\n<answer>Off-Platform Solicitation</answer>"
    )
    provider = FakeAgentProvider(
        [
            AgentTurn(
                tool_call=ToolCall(
                    name="inspect_destination",
                    arguments={"indicator": "w-h-a-t-s-a-p-p:user123"},
                )
            ),
            AgentTurn(content=completion),
        ]
    )
    output = tmp_path / "manifest-metrics"

    manifest = run_generation_batch(
        policies=(_policy(),),
        samples=(sample,),
        provider=provider,
        environment=_environment(),
        output_dir=output,
        budget=TeacherBudget(max_requests=10),
    )

    assert manifest["accepted_thinking_type_counts"] == {"fast": 0, "slow": 1}
    assert manifest["required_tool_samples"] == 1
    assert manifest["required_tool_accepted"] == 1
    assert manifest["tool_call_count"] == 1
    assert manifest["quality_gate"] == {
        "status": "pass",
        "expected_samples": 1,
        "expected_accepted": 1,
        "required_tool_samples": 1,
        "required_tool_accepted": 1,
    }
