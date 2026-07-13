"""Exact metrics for policy-conditioned risk decisions."""

from collections.abc import Iterable, Mapping
from typing import Any

from risk_agent.contracts import Oracle, Task


def _validate_row(
    row: tuple[Task, Oracle, Mapping[str, Any]],
) -> tuple[Task, Oracle, Mapping[str, Any]]:
    """Validate the small evaluator input contract before scoring a row."""

    if not isinstance(row, tuple) or len(row) != 3:
        raise TypeError("each evaluation row must be a (Task, Oracle, decision) tuple")
    task, oracle, decision = row
    if not isinstance(task, Task) or not isinstance(oracle, Oracle):
        raise TypeError("each evaluation row must contain a Task and Oracle")
    if not isinstance(decision, Mapping):
        raise TypeError("decision must be a mapping")
    if (task.asset_id, task.policy_version) != (oracle.asset_id, oracle.policy_version):
        raise ValueError("task and oracle must have the same asset and policy version")
    return task, oracle, decision


def _evidence_matches(oracle: Oracle, decision: Mapping[str, Any]) -> bool:
    """Compare evidence support as an unordered exact set."""

    evidence_ids = decision.get("evidence_ids", ())
    if isinstance(evidence_ids, (str, bytes)) or not isinstance(evidence_ids, Iterable):
        raise TypeError("decision evidence_ids must be an iterable of IDs")
    try:
        return set(oracle.evidence_ids) == set(evidence_ids)
    except TypeError as error:
        raise TypeError("decision evidence_ids must contain hashable IDs") from error


def evaluate_decisions(
    rows: Iterable[tuple[Task, Oracle, Mapping[str, Any]]],
) -> dict[str, float]:
    """Report exact label, policy-following, rule, and evidence metrics.

    A task already injects its full active policy, so following that policy is
    measured by whether the predicted label equals the oracle label for this
    exact task-policy pairing.
    """

    prepared_rows = [_validate_row(row) for row in rows]
    if not prepared_rows:
        raise ValueError("evaluate_decisions requires at least one row")

    correct_labels: list[bool] = []
    correct_rules: list[bool] = []
    correct_evidence: list[bool] = []
    for _, oracle, decision in prepared_rows:
        label_matches = oracle.label == decision.get("label")
        correct_labels.append(label_matches)
        correct_rules.append(oracle.rule_id == decision.get("rule_id"))
        correct_evidence.append(_evidence_matches(oracle, decision))

    total = len(prepared_rows)
    label_accuracy = sum(correct_labels) / total
    return {
        "label_accuracy": label_accuracy,
        "policy_following_accuracy": label_accuracy,
        "rule_exact_match": sum(correct_rules) / total,
        "evidence_exact_match": sum(correct_evidence) / total,
    }
