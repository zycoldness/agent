"""Bootstrap package contract tests."""

import pytest
from pydantic import ValidationError

import risk_agent
from risk_agent.contracts import Action, Decision, PolicyRule, Task


def test_package_marker_is_importable() -> None:
    """The research environment exposes its package marker."""
    assert risk_agent.__doc__


def test_action_parses_case_search_json() -> None:
    action = Action.model_validate_json(
        '{"tool":"search_case","arguments":{"query":"减肥"}}'
    )

    assert action.tool == "search_case"
    assert action.arguments == {"query": "减肥"}


def test_action_arguments_are_immutable() -> None:
    action = Action(tool="search_case", arguments={"query": "减肥"})

    with pytest.raises(TypeError):
        action.arguments["query"] = "changed"


def test_action_rejects_unknown_tool() -> None:
    with pytest.raises(ValidationError):
        Action(tool="unknown", arguments={})


def test_action_rejects_non_object_arguments() -> None:
    with pytest.raises(ValidationError):
        Action(tool="search_case", arguments="减肥")


def test_decision_preserves_evidence_ids() -> None:
    decision = Decision(
        label="unsafe",
        rule_id="AD-001",
        evidence_ids=("ocr-1",),
        confidence=0.9,
    )

    assert decision.evidence_ids == ("ocr-1",)


def test_task_includes_full_active_policy() -> None:
    rule = PolicyRule(rule_id="AD-001", title="Advertising", text="No deceptive claims")
    task = Task(
        asset_id="asset-1",
        policy_version="2026-07",
        active_policy=(rule,),
        initial_observation="Claims instant weight loss",
    )

    assert task.active_policy[0].rule_id == "AD-001"


def test_task_rejects_max_turns_above_limit() -> None:
    with pytest.raises(ValidationError):
        Task(
            asset_id="asset-1",
            policy_version="2026-07",
            active_policy=(),
            initial_observation="No observation",
            max_turns=4,
        )
