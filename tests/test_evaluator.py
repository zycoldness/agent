"""Tests for exact, policy-conditioned decision metrics."""

import pytest

from risk_agent.contracts import Oracle, PolicyRule, Task
from risk_agent.evaluator import evaluate_decisions


def make_task_and_oracle() -> tuple[Task, Oracle]:
    task = Task(
        asset_id="a1",
        policy_version="v1",
        active_policy=(PolicyRule(rule_id="AD-1", title="夸大", text="不得保证减重"),),
        initial_observation="OCR: 保证减重十斤",
    )
    oracle = Oracle(
        asset_id="a1",
        policy_version="v1",
        label="unsafe",
        rule_id="AD-1",
        evidence_ids=("ocr-1", "frame-1"),
    )
    return task, oracle


def test_evaluation_reports_exact_label_rule_and_evidence_metrics() -> None:
    task, oracle = make_task_and_oracle()

    report = evaluate_decisions(
        [
            (
                task,
                oracle,
                {
                    "label": "unsafe",
                    "rule_id": "AD-1",
                    "evidence_ids": ["frame-1", "ocr-1"],
                },
            ),
            (
                task,
                oracle,
                {
                    "label": "safe",
                    "rule_id": None,
                    "evidence_ids": ["ocr-1"],
                },
            ),
        ]
    )

    assert report == {
        "label_accuracy": 0.5,
        "policy_following_accuracy": 0.5,
        "rule_exact_match": 0.5,
        "evidence_exact_match": 0.5,
    }


def test_evaluation_treats_evidence_as_an_exact_set() -> None:
    task, oracle = make_task_and_oracle()

    report = evaluate_decisions(
        [(task, oracle, {"label": "unsafe", "rule_id": "AD-1", "evidence_ids": ["ocr-1", "ocr-1", "frame-1"]})]
    )

    assert report["evidence_exact_match"] == 1.0


def test_evaluation_rejects_empty_or_misaligned_rows() -> None:
    task, oracle = make_task_and_oracle()
    other_task = task.model_copy(update={"policy_version": "v2"})

    with pytest.raises(ValueError, match="at least one"):
        evaluate_decisions([])
    with pytest.raises(ValueError, match="same asset and policy version"):
        evaluate_decisions([(other_task, oracle, {"label": "unsafe"})])


def test_evaluation_rejects_non_mapping_decisions_and_non_iterable_evidence() -> None:
    task, oracle = make_task_and_oracle()

    with pytest.raises(TypeError, match="mapping"):
        evaluate_decisions([(task, oracle, "unsafe")])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="evidence_ids"):
        evaluate_decisions([(task, oracle, {"label": "unsafe", "evidence_ids": 3})])
