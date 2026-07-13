"""Tests for leak-free two-hop teacher SFT synthesis."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from risk_agent.contracts import Evidence, Oracle, PolicyRule, Task
from risk_agent.stores import CaseStore, EvidenceStore
from risk_agent.synthesis import generate_teacher_sample
from risk_agent.teacher import CallableTeacher


def _task() -> Task:
    return Task(
        asset_id="asset-1",
        policy_version="v1",
        active_policy=(
            PolicyRule(
                rule_id="AD-001",
                title="Absolute efficacy claim",
                text="Do not guarantee a weight-loss result.",
            ),
        ),
        initial_observation="OCR: Guaranteed to lose ten pounds in seven days.",
        max_turns=2,
    )


def _oracle() -> Oracle:
    return Oracle(
        asset_id="asset-1",
        policy_version="v1",
        label="unsafe",
        rule_id="AD-001",
        evidence_ids=("ocr-1",),
        risk_level="P1",
        next_action="block",
    )


def _stores() -> tuple[CaseStore, EvidenceStore]:
    return (
        CaseStore([{"case_id": "case-1", "text": "Public case about a guaranteed result."}]),
        EvidenceStore(
            [
                Evidence(
                    evidence_id="ocr-1",
                    asset_id="asset-1",
                    kind="ocr",
                    content="Guaranteed to lose ten pounds in seven days.",
                )
            ]
        ),
    )


def _teacher(outputs):
    remaining = iter(outputs)
    return CallableTeacher(lambda _request: next(remaining), model="fake-teacher")


def test_direct_teacher_decision_exports_only_rebuilt_student_messages():
    teacher = _teacher(
        [
            {
                "tool": "final_decision",
                "arguments": {
                    "label": "unsafe",
                    "rule_id": "AD-001",
                    "evidence_ids": ["ocr-1"],
                    "confidence": 0.8,
                    "risk_level": "P1",
                    "next_action": "block",
                },
            }
        ]
    )
    cases, evidence = _stores()

    result = generate_teacher_sample(_task(), _oracle(), cases, evidence, teacher)

    assert result.row["messages"][0]["role"] == "system"
    assert "oracle" not in json.dumps(result.row, ensure_ascii=False).casefold()
    assert "teacher" not in json.dumps(result.row["messages"], ensure_ascii=False).casefold()
    assert result.metadata["trajectory_type"] == "direct"
    assert result.metadata["teacher"]["request_count"] == 1
    assert "prompt" not in json.dumps(result.metadata).casefold()


def test_lookup_observation_is_recomputed_locally_and_teacher_observation_is_ignored():
    requests = []

    def fake(request):
        requests.append(request)
        if len(requests) == 1:
            return {
                "tool": "inspect_evidence",
                "arguments": {"kinds": ["ocr"]},
                "observation": {"label": "unsafe", "secret": "do not trust"},
            }
        return {
            "tool": "final_decision",
            "arguments": {
                "label": "unsafe",
                "rule_id": "AD-001",
                "evidence_ids": ["ocr-1"],
                "confidence": 0.9,
                "risk_level": "P1",
                "next_action": "block",
            },
        }

    cases, evidence = _stores()
    result = generate_teacher_sample(
        _task(), _oracle(), cases, evidence, CallableTeacher(fake, model="fake-teacher")
    )

    observation = result.row["messages"][3]["content"]
    assert "do not trust" not in observation
    assert json.loads(observation.removeprefix("Tool observation: ")) == [
        {
            "evidence_id": "ocr-1",
            "asset_id": "asset-1",
            "kind": "ocr",
            "content": "Guaranteed to lose ten pounds in seven days.",
        }
    ]
    assert requests[1]["local_observation"][0]["evidence_id"] == "ocr-1"
    assert result.metadata["trajectory_type"] == "one_lookup"
    assert result.metadata["teacher"]["request_count"] == 2


def test_second_hop_must_be_a_final_decision():
    teacher = _teacher(
        [
            {"tool": "inspect_evidence", "arguments": {"kinds": ["ocr"]}},
            {"tool": "search_case", "arguments": {"query": "claim"}},
        ]
    )
    cases, evidence = _stores()

    with pytest.raises(ValueError, match="second teacher action must be final_decision"):
        generate_teacher_sample(_task(), _oracle(), cases, evidence, teacher)


@pytest.mark.parametrize(
    "lookup",
    [
        {"tool": "get_rule_detail", "arguments": {"rule_id": "AD-001"}},
        {"tool": "search_case", "arguments": {"query": "guaranteed result", "top_k": 1}},
    ],
)
def test_rule_and_case_lookups_are_recomputed_for_safe_oracles_without_evidence(lookup):
    safe_oracle = Oracle(
        asset_id="asset-1",
        policy_version="v1",
        label="safe",
        evidence_ids=(),
        next_action="allow",
    )
    teacher = _teacher(
        [
            lookup,
            {
                "tool": "final_decision",
                "arguments": {
                    "label": "safe",
                    "rule_id": None,
                    "evidence_ids": [],
                    "confidence": 1.0,
                    "next_action": "allow",
                },
            },
        ]
    )
    cases, evidence = _stores()

    result = generate_teacher_sample(_task(), safe_oracle, cases, evidence, teacher)

    assert json.loads(result.row["messages"][2]["content"])["tool"] == lookup["tool"]
    assert result.row["messages"][3]["content"].startswith("Tool observation: ")


@pytest.mark.parametrize(
    "final_arguments",
    [
        {"label": "safe", "rule_id": None, "evidence_ids": [], "confidence": 1.0},
        {"label": "unsafe", "rule_id": "AD-999", "evidence_ids": ["ocr-1"], "confidence": 1.0},
        {"label": "unsafe", "rule_id": "AD-001", "evidence_ids": ["made-up"], "confidence": 1.0},
    ],
)
def test_teacher_final_must_agree_with_oracle_active_rule_and_observed_evidence(final_arguments):
    teacher = _teacher(
        [
            {"tool": "inspect_evidence", "arguments": {"kinds": ["ocr"]}},
            {"tool": "final_decision", "arguments": final_arguments},
        ]
    )
    cases, evidence = _stores()

    with pytest.raises(ValueError, match="teacher final decision"):
        generate_teacher_sample(_task(), _oracle(), cases, evidence, teacher)


def test_unknown_first_tool_is_rejected():
    cases, evidence = _stores()

    with pytest.raises(ValueError, match="valid JSON action"):
        generate_teacher_sample(
            _task(),
            _oracle(),
            cases,
            evidence,
            _teacher([{"tool": "browse_web", "arguments": {"query": "answer"}}]),
        )


def test_cli_requires_explicit_external_data_acknowledgement(tmp_path: Path):
    output = tmp_path / "out.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_teacher_sft.py",
            "data/fixtures/tasks.jsonl",
            "data/fixtures/oracle.jsonl",
            "data/fixtures/cases.jsonl",
            "data/fixtures/evidence.jsonl",
            str(output),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--allow-external-data" in result.stderr
    assert not output.exists()


def test_cli_rejects_non_allowlisted_data_classification_before_provider_call(tmp_path: Path):
    output = tmp_path / "out.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_teacher_sft.py",
            "data/fixtures/tasks.jsonl",
            "data/fixtures/oracle.jsonl",
            "data/fixtures/cases.jsonl",
            "data/fixtures/evidence.jsonl",
            str(output),
            "--allow-external-data",
            "--data-classification",
            "confidential",
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "synthetic or public" in result.stderr
    assert not output.exists()


def test_cli_rejects_output_that_overwrites_an_input(tmp_path: Path):
    task_path = tmp_path / "tasks.jsonl"
    oracle_path = tmp_path / "oracle.jsonl"
    cases_path = tmp_path / "cases.jsonl"
    evidence_path = tmp_path / "evidence.jsonl"
    task_path.write_text(_task().model_dump_json() + "\n", encoding="utf-8")
    oracle_path.write_text(_oracle().model_dump_json() + "\n", encoding="utf-8")
    cases_path.write_text('{"case_id":"case-1","text":"public"}\n', encoding="utf-8")
    evidence_path.write_text(
        Evidence(
            evidence_id="ocr-1", asset_id="asset-1", kind="ocr", content="claim"
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_teacher_sft.py",
            str(task_path),
            str(oracle_path),
            str(cases_path),
            str(evidence_path),
            str(task_path),
            "--allow-external-data",
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "must not overwrite" in result.stderr
    assert task_path.read_text(encoding="utf-8") == _task().model_dump_json() + "\n"
