"""Exact metrics for policy-conditioned risk decisions."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from itertools import combinations
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
    if isinstance(evidence_ids, (Mapping, str, bytes)) or not isinstance(evidence_ids, Iterable):
        raise TypeError("decision evidence_ids must be an iterable of IDs")
    try:
        submitted_ids = tuple(evidence_ids)
    except TypeError as error:
        raise TypeError("decision evidence_ids must be iterable") from error
    if any(not isinstance(evidence_id, str) for evidence_id in submitted_ids):
        raise TypeError("decision evidence_ids must contain only string IDs")
    return set(oracle.evidence_ids) == set(submitted_ids)


def _decision_label(decision: Mapping[str, Any]) -> str:
    """Return the one valid binary decision label used by every metric."""

    label = decision.get("label")
    if label not in {"safe", "unsafe"}:
        raise ValueError("decision label must be 'safe' or 'unsafe'")
    return label


def _policy_following_accuracy(
    rows: list[tuple[Task, Oracle, Mapping[str, Any]]],
    labels: list[str],
) -> float:
    """Score whether each asset's predicted label changes with its policy.

    For a counterfactual group, every pair of policy variants must agree on
    whether the label changed.  A singleton has no policy comparison, so it
    falls back to ordinary label correctness rather than being silently ignored.
    """

    grouped: dict[str, list[tuple[Oracle, str]]] = defaultdict(list)
    for (_, oracle, _), label in zip(rows, labels, strict=True):
        grouped[oracle.asset_id].append((oracle, label))

    comparisons: list[bool] = []
    for group in grouped.values():
        if len(group) == 1:
            oracle, label = group[0]
            comparisons.append(label == oracle.label)
            continue
        for (left_oracle, left_label), (right_oracle, right_label) in combinations(group, 2):
            oracle_changed = left_oracle.label != right_oracle.label
            predicted_changed = left_label != right_label
            comparisons.append(predicted_changed == oracle_changed)
    return sum(comparisons) / len(comparisons)


def evaluate_decisions(
    rows: Iterable[tuple[Task, Oracle, Mapping[str, Any]]],
) -> dict[str, float]:
    """Report exact label, policy-following, rule, and evidence metrics.

    Policy following is evaluated over same-asset policy variants: predictions
    must change (or remain unchanged) whenever the matching oracle labels do.
    """

    prepared_rows = [_validate_row(row) for row in rows]
    if not prepared_rows:
        raise ValueError("evaluate_decisions requires at least one row")

    labels: list[str] = []
    correct_labels: list[bool] = []
    correct_rules: list[bool] = []
    correct_evidence: list[bool] = []
    for _, oracle, decision in prepared_rows:
        label = _decision_label(decision)
        labels.append(label)
        label_matches = oracle.label == label
        correct_labels.append(label_matches)
        correct_rules.append(oracle.rule_id == decision.get("rule_id"))
        correct_evidence.append(_evidence_matches(oracle, decision))

    total = len(prepared_rows)
    label_accuracy = sum(correct_labels) / total
    return {
        "label_accuracy": label_accuracy,
        "policy_following_accuracy": _policy_following_accuracy(prepared_rows, labels),
        "rule_exact_match": sum(correct_rules) / total,
        "evidence_exact_match": sum(correct_evidence) / total,
    }
