"""Exact metrics for policy-conditioned risk decisions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any, TypeAlias

from risk_agent.contracts import Oracle, Task
from risk_agent.counterfactuals import PolicyShiftRow

LegacyDecisionRow: TypeAlias = tuple[Task, Oracle, Mapping[str, Any]]
GroupedDecisionRow: TypeAlias = tuple[Task, Oracle, Mapping[str, Any], str]


@dataclass(frozen=True)
class EvaluationRow:
    """A scored prediction, optionally tied to one exact counterfactual pair."""

    task: Task
    oracle: Oracle
    decision: Mapping[str, Any]
    group_id: str | None = None

    @classmethod
    def from_policy_shift(
        cls, row: PolicyShiftRow, decision: Mapping[str, Any]
    ) -> EvaluationRow:
        """Carry the fixture's unambiguous before/after identity into scoring."""

        if not isinstance(row, PolicyShiftRow) or not row.group_id:
            raise ValueError("policy shift rows require a non-empty group_id for evaluation")
        return cls(row.task, row.oracle, decision, group_id=row.group_id)


EvaluationInput: TypeAlias = EvaluationRow | LegacyDecisionRow | GroupedDecisionRow


def _validate_row(row: EvaluationInput) -> EvaluationRow:
    """Normalize typed and legacy evaluator input while retaining group identity."""

    if isinstance(row, EvaluationRow):
        normalized = row
    elif isinstance(row, tuple) and len(row) in {3, 4}:
        task, oracle, decision = row[:3]
        group_id = row[3] if len(row) == 4 else None
        normalized = EvaluationRow(task, oracle, decision, group_id=group_id)
    else:
        raise TypeError("each evaluation row must be an EvaluationRow or a 3/4-item tuple")

    if not isinstance(normalized.task, Task) or not isinstance(normalized.oracle, Oracle):
        raise TypeError("each evaluation row must contain a Task and Oracle")
    if not isinstance(normalized.decision, Mapping):
        raise TypeError("decision must be a mapping")
    if normalized.group_id is not None and (
        not isinstance(normalized.group_id, str) or not normalized.group_id.strip()
    ):
        raise ValueError("group_id must be a non-empty string when supplied")
    if (normalized.task.asset_id, normalized.task.policy_version) != (
        normalized.oracle.asset_id,
        normalized.oracle.policy_version,
    ):
        raise ValueError("task and oracle must have the same asset and policy version")
    return normalized


def _decision_label(decision: Mapping[str, Any]) -> str:
    """Return the one valid binary decision label used by every metric."""

    label = decision.get("label")
    if label not in {"safe", "unsafe"}:
        raise ValueError("decision label must be 'safe' or 'unsafe'")
    return label


def _directionally_matches(
    left: tuple[Oracle, str], right: tuple[Oracle, str]
) -> bool:
    """Require the predicted before/after labels to match the oracle direction."""

    left_oracle, left_label = left
    right_oracle, right_label = right
    return (left_label, right_label) == (left_oracle.label, right_oracle.label)


def _policy_following_accuracy(rows: list[EvaluationRow], labels: list[str]) -> float:
    """Score directional policy following over explicit before/after fixture pairs.

    ``EvaluationRow`` and 4-item tuples use their ``group_id`` and must form
    exactly one same-asset pair.  The legacy 3-item tuple accepted by the
    original plan has no group metadata, so it falls back to one asset-level
    group with all pairwise directional comparisons.  Singletons fall back to
    label correctness because no policy transition is observable.
    """

    grouped: dict[tuple[str, str], list[tuple[EvaluationRow, str]]] = defaultdict(list)
    for row, label in zip(rows, labels, strict=True):
        key = ("explicit", row.group_id) if row.group_id is not None else ("legacy", row.oracle.asset_id)
        grouped[key].append((row, label))

    comparisons: list[bool] = []
    for (group_kind, group_id), group in grouped.items():
        if group_kind == "explicit":
            if len(group) != 2:
                raise ValueError(f"counterfactual group {group_id!r} must contain exactly two rows")
            if len({row.oracle.asset_id for row, _ in group}) != 1:
                raise ValueError(f"counterfactual group {group_id!r} must contain one asset")
            left, right = group
            comparisons.append(
                _directionally_matches(
                    (left[0].oracle, left[1]), (right[0].oracle, right[1])
                )
            )
            continue

        if len(group) == 1:
            row, label = group[0]
            comparisons.append(label == row.oracle.label)
            continue
        for left, right in combinations(group, 2):
            comparisons.append(
                _directionally_matches(
                    (left[0].oracle, left[1]), (right[0].oracle, right[1])
                )
            )
    return sum(comparisons) / len(comparisons)


def evaluate_decisions(rows: Iterable[EvaluationInput]) -> dict[str, float]:
    """Report exact label, policy-following, rule, and evidence metrics."""

    prepared_rows = [_validate_row(row) for row in rows]
    if not prepared_rows:
        raise ValueError("evaluate_decisions requires at least one row")

    labels: list[str] = []
    correct_labels: list[bool] = []
    correct_rules: list[bool] = []
    for row in prepared_rows:
        label = _decision_label(row.decision)
        labels.append(label)
        correct_labels.append(row.oracle.label == label)
        correct_rules.append(row.oracle.rule_id == row.decision.get("rule_id"))

    total = len(prepared_rows)
    return {
        "label_accuracy": sum(correct_labels) / total,
        "policy_following_accuracy": _policy_following_accuracy(prepared_rows, labels),
        "rule_exact_match": sum(correct_rules) / total,
    }
