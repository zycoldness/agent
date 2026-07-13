"""Fixtures for testing whether a decision follows the active policy."""

from dataclasses import dataclass

from risk_agent.contracts import Oracle, PolicyRule, Task


@dataclass(frozen=True)
class PolicyShiftRow:
    """One policy-conditioned task and its isolated reference assessment."""

    task: Task
    oracle: Oracle


def build_policy_shift_tasks(
    asset_id: str,
    observation: str,
    matching_rule: PolicyRule,
) -> tuple[PolicyShiftRow, PolicyShiftRow]:
    """Create same-asset unsafe/safe examples by adding or removing one rule."""

    if not isinstance(matching_rule, PolicyRule):
        raise TypeError("matching_rule must be a PolicyRule")

    with_rule = Task(
        asset_id=asset_id,
        policy_version="with-rule",
        active_policy=(matching_rule,),
        initial_observation=observation,
    )
    without_rule = Task(
        asset_id=asset_id,
        policy_version="without-rule",
        active_policy=(),
        initial_observation=observation,
    )
    return (
        PolicyShiftRow(
            with_rule,
            Oracle(
                asset_id=asset_id,
                policy_version="with-rule",
                label="unsafe",
                rule_id=matching_rule.rule_id,
            ),
        ),
        PolicyShiftRow(
            without_rule,
            Oracle(
                asset_id=asset_id,
                policy_version="without-rule",
                label="safe",
            ),
        ),
    )
