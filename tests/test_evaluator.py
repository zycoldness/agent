"""Tests for exact, policy-conditioned decision metrics."""

import pytest

from risk_agent.contracts import Oracle, PolicyRule, Task
from risk_agent.counterfactuals import PolicyOutcome, build_policy_shift_rows
from risk_agent.evaluator import EvaluationRow, evaluate_decisions


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
    safe_task = task.model_copy(update={"policy_version": "v2", "active_policy": ()})
    safe_oracle = oracle.model_copy(
        update={"policy_version": "v2", "label": "safe", "rule_id": None, "evidence_ids": ()}
    )

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
                safe_task,
                safe_oracle,
                {
                    "label": "unsafe",
                    "rule_id": "AD-1",
                    "evidence_ids": ["ocr-1"],
                },
            ),
        ]
    )

    assert report == {
        "label_accuracy": 0.5,
        "policy_following_accuracy": 0.0,
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
    with pytest.raises(TypeError, match="evidence_ids"):
        evaluate_decisions([(task, oracle, {"label": "unsafe", "evidence_ids": {"ocr-1": "mapped"}})])


def test_policy_following_requires_the_same_label_change_as_the_oracle() -> None:
    unsafe_task, unsafe_oracle = make_task_and_oracle()
    safe_task = unsafe_task.model_copy(update={"policy_version": "v2", "active_policy": ()})
    safe_oracle = unsafe_oracle.model_copy(
        update={"policy_version": "v2", "label": "safe", "rule_id": None, "evidence_ids": ()}
    )

    report = evaluate_decisions(
        [
            (unsafe_task, unsafe_oracle, {"label": "unsafe", "rule_id": "AD-1", "evidence_ids": []}),
            (safe_task, safe_oracle, {"label": "unsafe", "rule_id": None, "evidence_ids": []}),
        ]
    )

    assert report["label_accuracy"] == 0.5
    assert report["policy_following_accuracy"] == 0.0


def test_policy_following_for_singleton_assets_falls_back_to_label_correctness() -> None:
    task, oracle = make_task_and_oracle()

    report = evaluate_decisions([(task, oracle, {"label": "safe", "rule_id": None, "evidence_ids": []})])

    assert report["label_accuracy"] == 0.0
    assert report["policy_following_accuracy"] == 0.0


def test_policy_following_rejects_a_fully_inverted_counterfactual_pair() -> None:
    unsafe_task, unsafe_oracle = make_task_and_oracle()
    safe_task = unsafe_task.model_copy(update={"policy_version": "v2", "active_policy": ()})
    safe_oracle = unsafe_oracle.model_copy(
        update={"policy_version": "v2", "label": "safe", "rule_id": None, "evidence_ids": ()}
    )

    report = evaluate_decisions(
        [
            (unsafe_task, unsafe_oracle, {"label": "safe", "rule_id": None, "evidence_ids": []}, "pair-1"),
            (safe_task, safe_oracle, {"label": "unsafe", "rule_id": "AD-1", "evidence_ids": []}, "pair-1"),
        ]
    )

    assert report["label_accuracy"] == 0.0
    assert report["policy_following_accuracy"] == 0.0


def test_explicit_groups_keep_independent_same_asset_transformations_separate() -> None:
    rule = PolicyRule(rule_id="AD-1", title="Claims", text="Do not guarantee weight loss")
    first_pair = build_policy_shift_rows(
        "a1",
        "OCR",
        before_policy=(rule,),
        after_policy=(),
        before_outcome=PolicyOutcome(label="unsafe", rule_id="AD-1"),
        after_outcome=PolicyOutcome(label="safe"),
        transformation="rule_removal",
    )
    second_pair = build_policy_shift_rows(
        "a1",
        "OCR",
        before_policy=(rule,),
        after_policy=(),
        before_outcome=PolicyOutcome(label="unsafe", rule_id="AD-1"),
        after_outcome=PolicyOutcome(label="safe"),
        transformation="exemption",
    )

    report = evaluate_decisions(
        [
            EvaluationRow.from_policy_shift(first_pair[0], {"label": "safe", "evidence_ids": []}),
            EvaluationRow.from_policy_shift(first_pair[1], {"label": "unsafe", "evidence_ids": []}),
            EvaluationRow.from_policy_shift(second_pair[0], {"label": "unsafe", "evidence_ids": []}),
            EvaluationRow.from_policy_shift(second_pair[1], {"label": "safe", "evidence_ids": []}),
        ]
    )

    assert report["policy_following_accuracy"] == 0.5
