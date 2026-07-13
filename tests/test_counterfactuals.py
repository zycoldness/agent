"""Tests for same-asset dynamic-policy counterfactual fixtures."""

import pytest

from risk_agent.contracts import PolicyRule
from risk_agent.counterfactuals import PolicyShiftRow, build_policy_shift_tasks


def test_same_asset_has_different_oracles_when_active_rule_is_removed() -> None:
    rule = PolicyRule(rule_id="AD-1", title="夸大", text="不得保证减重")

    rows = build_policy_shift_tasks("a1", "OCR: 保证减重十斤", rule)

    assert all(isinstance(row, PolicyShiftRow) for row in rows)
    assert {(row.task.policy_version, row.oracle.label) for row in rows} == {
        ("with-rule", "unsafe"),
        ("without-rule", "safe"),
    }
    assert {row.task.asset_id for row in rows} == {"a1"}
    assert {row.task.initial_observation for row in rows} == {"OCR: 保证减重十斤"}
    assert rows[0].task.active_policy == (rule,)
    assert rows[0].oracle.rule_id == "AD-1"
    assert rows[1].task.active_policy == ()
    assert rows[1].oracle.rule_id is None


def test_policy_shift_rows_are_immutable() -> None:
    rule = PolicyRule(rule_id="AD-1", title="夸大", text="不得保证减重")
    row = build_policy_shift_tasks("a1", "OCR", rule)[0]

    with pytest.raises(AttributeError):
        row.task = row.task
