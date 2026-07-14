"""Tests for the closed-world, policy-scoped risk environment."""

import json

import pytest

from risk_agent.contracts import Evidence, Oracle, PolicyRule, Task
from risk_agent.environment import RiskEnvironment
from risk_agent.stores import CaseStore, EvidenceStore


def make_env(*, max_turns: int = 3) -> RiskEnvironment:
    """Build a task whose oracle-only strings make accidental leakage visible."""

    task = Task(
        asset_id="asset-1",
        policy_version="v1",
        active_policy=(
            PolicyRule(rule_id="AD-1", title="Claims", text="Do not guarantee results"),
            PolicyRule(rule_id="AD-2", title="Disclosure", text="Show required disclosure"),
        ),
        initial_observation="OCR: guaranteed result in seven days",
        max_turns=max_turns,
    )
    return RiskEnvironment(
        task,
        CaseStore(
            [
                {
                    "case_id": "case-1",
                    "text": "A historical claim about seven-day results",
                    "label": "unsafe",
                    "rule_id": "ORACLE-RULE",
                }
            ]
        ),
        EvidenceStore(
            [
                Evidence(
                    evidence_id="ocr-1",
                    asset_id="asset-1",
                    kind="ocr",
                    content="guaranteed result in seven days",
                ),
                Evidence(
                    evidence_id="other-ocr",
                    asset_id="other-asset",
                    kind="ocr",
                    content="must never be returned",
                ),
            ]
        ),
        Oracle(
            asset_id="asset-1",
            policy_version="v1",
            label="unsafe",
            rule_id="AD-1",
            risk_level="P0",
            next_action="oracle-only-next-action",
        ),
    )


def action(tool: str, arguments: dict[str, object]) -> str:
    return json.dumps({"tool": tool, "arguments": arguments})


def assert_no_oracle_contents(result: object) -> None:
    rendered = str(result)
    assert "oracle-only-next-action" not in rendered
    assert "ORACLE-RULE" not in rendered
    assert "P0" not in rendered


def test_reset_injects_all_active_policy_and_initial_observation() -> None:
    observation = make_env().reset()

    assert "[AD-1] Claims: Do not guarantee results" in observation
    assert "[AD-2] Disclosure: Show required disclosure" in observation
    assert "OCR: guaranteed result in seven days" in observation


@pytest.mark.parametrize(
    "serialized",
    [
        "not-json",
        "[]",
        action("get_rule_detail", {}),
        action("get_rule_detail", {"rule_id": 3}),
        action("not_a_tool", {}),
    ],
)
def test_malformed_or_unknown_actions_terminate_without_oracle_leakage(serialized: str) -> None:
    result = make_env().step(serialized)

    assert result.done is True
    assert result.info["status"] == "invalid_action"
    assert_no_oracle_contents(result)


def test_rule_lookup_returns_only_a_rule_from_active_policy() -> None:
    result = make_env().step(action("get_rule_detail", {"rule_id": "AD-1"}))

    assert result.done is False
    assert result.reward == -0.08
    assert json.loads(result.observation)["rule_id"] == "AD-1"
    assert_no_oracle_contents(result)


def test_hidden_rule_lookup_terminates_without_oracle_leakage() -> None:
    result = make_env().step(action("get_rule_detail", {"rule_id": "ORACLE-RULE"}))

    assert result.done is True
    assert result.info["status"] == "invalid_action"
    assert_no_oracle_contents(result)


def test_case_search_returns_sanitized_store_results_only() -> None:
    result = make_env().step(action("search_case", {"query": "seven-day", "top_k": 1}))

    rows = json.loads(result.observation)
    assert rows == [{"case_id": "case-1", "text": "A historical claim about seven-day results"}]
    assert result.reward == -0.08
    assert_no_oracle_contents(result)


def test_evidence_inspection_is_asset_scoped() -> None:
    env = make_env()
    result = env.step(action("inspect_evidence", {"kinds": ["ocr"]}))

    assert json.loads(result.observation) == [
        {
            "evidence_id": "ocr-1",
            "asset_id": "asset-1",
            "kind": "ocr",
            "content": "guaranteed result in seven days",
        }
    ]
    assert_no_oracle_contents(result)


@pytest.mark.parametrize(
    "arguments",
    [
        {"label": "unsafe", "rule_id": "HIDDEN", "confidence": 0.9},
        {"label": "unsafe", "rule_id": "AD-1", "evidence_ids": ["ocr-1"], "confidence": 0.9},
        {"label": "not-a-label", "rule_id": "AD-1", "confidence": 0.9},
    ],
)
def test_final_decision_rejects_hidden_invalid_or_removed_fields(arguments: dict[str, object]) -> None:
    result = make_env().step(action("final_decision", arguments))

    assert result.done is True
    assert result.info["status"] == "invalid_action"
    assert_no_oracle_contents(result)


def test_final_decision_scores_label_and_rule() -> None:
    env = make_env()
    env.reset()
    tool_result = env.step(action("inspect_evidence", {"kinds": ["ocr"]}))
    result = env.step(
        action(
            "final_decision",
            {
                "label": "unsafe",
                "rule_id": "AD-1",
                "confidence": 0.9,
            },
        )
    )

    assert tool_result.reward == -0.08
    assert result.done is True
    assert result.info["status"] == "final"
    assert result.reward == 2.0


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "label": "unsafe",
            "rule_id": "AD-1",
            "confidence": True,
        },
        {
            "label": "unsafe",
            "rule_id": "AD-1",
            "confidence": "0.5",
        },
        {
            "label": "unsafe",
            "rule_id": "AD-1",
            "confidence": 0.5,
            "unexpected": "must be rejected",
        },
    ],
)
def test_final_decision_rejects_coercions_and_unknown_fields(arguments: dict[str, object]) -> None:
    result = make_env().step(action("final_decision", arguments))

    assert result.done is True
    assert result.info["status"] == "invalid_action"
    assert_no_oracle_contents(result)


def test_tool_on_last_turn_ends_with_deterministic_non_oracle_status() -> None:
    env = make_env(max_turns=2)
    env.reset()
    first = env.step(action("search_case", {"query": "claim", "top_k": 1}))
    last = env.step(action("get_rule_detail", {"rule_id": "AD-1"}))

    assert first.done is False
    assert first.info["status"] == "tool"
    assert last.done is True
    assert last.info["status"] == "turn_limit"
    assert last.reward == -0.08
    assert_no_oracle_contents(last)


def test_step_after_termination_is_invalid_and_does_not_reopen_environment() -> None:
    env = make_env(max_turns=1)
    env.step(action("search_case", {"query": "claim", "top_k": 1}))
    result = env.step(action("get_rule_detail", {"rule_id": "AD-1"}))

    assert result.done is True
    assert result.info["status"] == "invalid_action"
    assert_no_oracle_contents(result)
