"""Tests for same-asset dynamic-policy counterfactual fixtures."""

import pytest

from risk_agent.contracts import PolicyRule
from risk_agent.counterfactuals import (
    PolicyOutcome,
    PolicyShiftRow,
    build_exemption_tasks,
    build_policy_shift_rows,
    build_policy_shift_tasks,
    build_rule_addition_tasks,
    build_rule_removal_tasks,
    build_rule_rewrite_tasks,
)


def test_same_asset_has_different_oracles_when_active_rule_is_removed() -> None:
    rule = PolicyRule(rule_id="AD-1", title="Claims", text="Do not guarantee weight loss")

    rows = build_policy_shift_tasks("a1", "OCR: guaranteed weight loss", rule)

    assert all(isinstance(row, PolicyShiftRow) for row in rows)
    assert [row.oracle.label for row in rows] == ["unsafe", "safe"]
    assert rows[0].task.policy_version != rows[1].task.policy_version
    assert rows[0].group_id == rows[1].group_id
    assert rows[0].group_id
    assert {row.task.asset_id for row in rows} == {"a1"}
    assert {row.task.initial_observation for row in rows} == {"OCR: guaranteed weight loss"}
    assert rows[0].task.active_policy == (rule,)
    assert rows[0].oracle.rule_id == "AD-1"
    assert rows[1].task.active_policy == ()
    assert rows[1].oracle.rule_id is None


def test_legacy_fixed_policy_versions_require_an_explicit_opt_in() -> None:
    rule = PolicyRule(rule_id="AD-1", title="Claims", text="Do not guarantee weight loss")

    rows = build_policy_shift_tasks("a1", "OCR", rule, legacy_policy_versions=True)

    assert [row.task.policy_version for row in rows] == ["with-rule", "without-rule"]


def test_policy_shift_rows_are_immutable() -> None:
    rule = PolicyRule(rule_id="AD-1", title="Claims", text="Do not guarantee weight loss")
    row = build_policy_shift_tasks("a1", "OCR", rule)[0]

    with pytest.raises(AttributeError):
        row.task = row.task


def test_generic_builder_uses_complete_policies_explicit_outcomes_and_unique_versions() -> None:
    unrelated = PolicyRule(rule_id="DISC-1", title="Disclosure", text="Show a disclosure")
    matching = PolicyRule(rule_id="AD-1", title="Claims", text="Do not guarantee weight loss")

    rows = build_policy_shift_rows(
        "a1",
        "OCR: guaranteed weight loss",
        before_policy=(unrelated,),
        after_policy=(unrelated, matching),
        before_outcome=PolicyOutcome(label="safe"),
        after_outcome=PolicyOutcome(label="unsafe", rule_id="AD-1"),
        transformation="rule_addition",
    )
    repeated_rows = build_policy_shift_rows(
        "a1",
        "OCR: guaranteed weight loss",
        before_policy=(unrelated,),
        after_policy=(unrelated, matching),
        before_outcome=PolicyOutcome(label="safe"),
        after_outcome=PolicyOutcome(label="unsafe", rule_id="AD-1"),
        transformation="rule_addition",
    )

    assert [row.oracle.label for row in rows] == ["safe", "unsafe"]
    assert rows[0].task.active_policy == (unrelated,)
    assert rows[1].task.active_policy == (unrelated, matching)
    assert all(row.transformation == "rule_addition" for row in rows)
    assert rows[0].group_id == rows[1].group_id
    assert rows[0].task.policy_version != rows[1].task.policy_version
    assert rows[0].task.policy_version == repeated_rows[0].task.policy_version
    assert rows[1].task.policy_version == repeated_rows[1].task.policy_version
    assert all(row.task.policy_version == row.oracle.policy_version for row in rows)


def test_named_helpers_cover_addition_removal_rewrite_and_exemption_with_unrelated_rules() -> None:
    unrelated = PolicyRule(rule_id="DISC-1", title="Disclosure", text="Show a disclosure")
    rule = PolicyRule(
        rule_id="AD-1",
        title="Claims",
        text="Do not guarantee weight loss",
        exceptions=("existing",),
    )
    rewritten = PolicyRule(
        rule_id="AD-2", title="Claims", text="Allow general lifestyle descriptions"
    )

    additions = build_rule_addition_tasks("a1", "OCR", rule, unrelated_rules=(unrelated,))
    removals = build_rule_removal_tasks("a1", "OCR", rule, unrelated_rules=(unrelated,))
    rewrites = build_rule_rewrite_tasks("a1", "OCR", rule, rewritten, unrelated_rules=(unrelated,))
    exemptions = build_exemption_tasks(
        "a1", "OCR", rule, "medical-ad", unrelated_rules=(unrelated,)
    )

    assert [row.oracle.label for row in additions] == ["safe", "unsafe"]
    assert [row.oracle.label for row in removals] == ["unsafe", "safe"]
    assert [row.oracle.label for row in rewrites] == ["unsafe", "safe"]
    assert [row.oracle.label for row in exemptions] == ["unsafe", "safe"]
    assert [row.transformation for row in additions] == ["rule_addition"] * 2
    assert [row.transformation for row in removals] == ["rule_removal"] * 2
    assert [row.transformation for row in rewrites] == ["rule_rewrite"] * 2
    assert [row.transformation for row in exemptions] == ["exemption"] * 2
    all_rows = additions + removals + rewrites + exemptions
    assert all(row.task.active_policy[0] == unrelated for row in all_rows)
    assert exemptions[1].task.active_policy[-1].exceptions == ("existing", "medical-ad")
    assert all(row.task.policy_version == row.oracle.policy_version for row in all_rows)
    assert len({rows[0].group_id for rows in (additions, removals, rewrites, exemptions)}) == 4


def test_generic_builder_honors_caller_supplied_policy_versions() -> None:
    rule = PolicyRule(rule_id="AD-1", title="Claims", text="Do not guarantee weight loss")

    rows = build_policy_shift_rows(
        "a1",
        "OCR",
        before_policy=(rule,),
        after_policy=(),
        before_outcome=PolicyOutcome(label="unsafe", rule_id="AD-1"),
        after_outcome=PolicyOutcome(label="safe"),
        transformation="rule_removal",
        before_policy_version="ad-v1",
        after_policy_version="ad-v2",
    )

    assert [row.task.policy_version for row in rows] == ["ad-v1", "ad-v2"]
    assert [row.oracle.policy_version for row in rows] == ["ad-v1", "ad-v2"]
