"""Tests for leak-free, policy-conditioned SFT serialization."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from risk_agent.contracts import Evidence, Oracle, PolicyRule, Task
from risk_agent.sft_export import export_track_a, export_trajectory as _export_trajectory
from risk_agent.stores import CaseStore, EvidenceStore


def _task(*, max_turns: int = 3) -> Task:
    return Task(
        asset_id="asset-1",
        policy_version="policy-v1",
        active_policy=(
            PolicyRule(
                rule_id="AD-001",
                title="Absolute efficacy claim",
                text="Do not guarantee a weight-loss result.",
            ),
        ),
        initial_observation="OCR: Guaranteed to lose ten pounds in seven days.",
        max_turns=max_turns,
    )


def _oracle() -> Oracle:
    return Oracle(
        asset_id="asset-1",
        policy_version="policy-v1",
        label="unsafe",
        rule_id="AD-001",
        evidence_ids=("ocr-1",),
        risk_level="P1",
        next_action="block",
    )


def _action(tool: str, arguments: dict[str, object]) -> str:
    return json.dumps({"tool": tool, "arguments": arguments}, ensure_ascii=False)


def _rule_observation(rule: PolicyRule | None = None) -> str:
    rule = rule or _task().active_policy[0]
    return json.dumps(rule.model_dump(mode="json"), ensure_ascii=False)


def _case_store() -> CaseStore:
    return CaseStore(
        [
            {
                "case_id": "case-1",
                "text": "Synthetic public case text about a guaranteed weight-loss claim.",
            }
        ]
    )


def _evidence_store() -> EvidenceStore:
    return EvidenceStore(
        [
            Evidence(
                evidence_id="ocr-1",
                asset_id="asset-1",
                kind="ocr",
                content="Guaranteed to lose ten pounds in seven days.",
            )
        ]
    )


def export_trajectory(task: Task, oracle: Oracle, steps):
    """Exercise exports with the trusted closed-world stores every time."""

    return _export_trajectory(task, oracle, steps, _case_store(), _evidence_store())


def test_track_a_injects_full_policy_and_only_generates_final_action():
    row = export_track_a(_task(), _oracle())

    assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
    assert "[AD-001] Absolute efficacy claim: Do not guarantee a weight-loss result." in row["messages"][0]["content"]
    assert row["messages"][1]["content"] == "OCR: Guaranteed to lose ten pounds in seven days."
    assert json.loads(row["messages"][2]["content"]) == {
        "tool": "final_decision",
        "arguments": {
            "label": "unsafe",
            "rule_id": "AD-001",
            "evidence_ids": ["ocr-1"],
            "confidence": 1.0,
            "risk_level": "P1",
            "next_action": "block",
        },
    }


def test_trajectory_serializes_one_valid_nonfinal_action_per_tool_turn():
    row = export_trajectory(
        _task(),
        _oracle(),
        [
            (_action("get_rule_detail", {"rule_id": "AD-001"}), _rule_observation()),
            (
                _action("inspect_evidence", {"kinds": ["ocr"]}),
                '[{"evidence_id":"ocr-1","asset_id":"asset-1","kind":"ocr","content":"Guaranteed to lose ten pounds in seven days."}]',
            ),
        ],
    )

    assistants = row["messages"][2::2]
    assert [json.loads(message["content"])["tool"] for message in assistants] == [
        "get_rule_detail",
        "inspect_evidence",
        "final_decision",
    ]
    assert all(set(json.loads(message["content"])) == {"tool", "arguments"} for message in assistants)
    assert row["messages"][3]["role"] == "user"
    assert row["messages"][3]["content"].startswith("Tool observation: ")


def test_trajectory_rejects_oracle_fields_in_a_tool_observation():
    with pytest.raises(ValueError, match="oracle field"):
        export_trajectory(
            _task(),
            _oracle(),
            [
                (
                    _action("search_case", {"query": "weight loss", "top_k": 1}),
                    '{"label":"unsafe","case_id":"case-1","text":"fact only"}',
                ),
            ],
        )


def test_trajectory_rejects_nested_oracle_object_in_a_tool_observation():
    with pytest.raises(ValueError, match="oracle field"):
        export_trajectory(
            _task(),
            _oracle(),
            [
                (
                    _action("search_case", {"query": "weight loss", "top_k": 1}),
                    '{"case_id":"case-1","oracle":{}}',
                ),
            ],
        )


@pytest.mark.parametrize(
    "observation",
    [
        "Oracle outcome: unsafe under AD-001.",
        '{"verdict":"unsafe"}',
        '[{"case_id":"case-1","text":"fact only","gold_evidence":["ocr-1"]}]',
    ],
)
def test_trajectory_rejects_free_text_and_gold_style_tool_observations(observation: str):
    with pytest.raises(ValueError, match="tool observation"):
        export_trajectory(
            _task(),
            _oracle(),
            [(_action("search_case", {"query": "weight loss", "top_k": 1}), observation)],
        )


def test_trajectory_rejects_rule_detail_that_does_not_match_requested_active_rule():
    wrong_rule = PolicyRule(rule_id="AD-999", title="Other", text="Other text")
    with pytest.raises(ValueError, match="active policy rule"):
        export_trajectory(
            _task(),
            _oracle(),
            [(_action("get_rule_detail", {"rule_id": "AD-001"}), _rule_observation(wrong_rule))],
        )


def test_trajectory_rejects_noncanonical_rule_detail_field_types():
    malformed_rule = _task().active_policy[0].model_dump(mode="json")
    malformed_rule["priority"] = "100"
    with pytest.raises(ValueError, match="active policy rule"):
        export_trajectory(
            _task(),
            _oracle(),
            [
                (
                    _action("get_rule_detail", {"rule_id": "AD-001"}),
                    json.dumps(malformed_rule),
                )
            ],
        )


def test_trajectory_rejects_evidence_outside_requested_asset_or_kind():
    with pytest.raises(ValueError, match="requested asset and kinds"):
        export_trajectory(
            _task(),
            _oracle(),
            [
                (
                    _action("inspect_evidence", {"kinds": ["ocr"]}),
                    '[{"evidence_id":"ocr-1","asset_id":"asset-2","kind":"metadata","content":"Guaranteed"}]',
                )
            ],
        )


def test_trajectory_rejects_case_result_with_unapproved_extra_field():
    with pytest.raises(ValueError, match="sanitized case"):
        export_trajectory(
            _task(),
            _oracle(),
            [
                (
                    _action("search_case", {"query": "weight loss", "top_k": 1}),
                    '[{"case_id":"case-1","text":"fact only","source_type":"case"}]',
                )
            ],
        )


def test_trajectory_rejects_structurally_valid_case_not_returned_by_case_store():
    with pytest.raises(ValueError, match="deterministic case-store result"):
        export_trajectory(
            _task(),
            _oracle(),
            [
                (
                    _action("search_case", {"query": "weight loss", "top_k": 1}),
                    '[{"case_id":"case-1","text":"Oracle says this is unsafe."}]',
                )
            ],
        )


def test_trajectory_rejects_structurally_valid_evidence_not_returned_by_evidence_store():
    with pytest.raises(ValueError, match="deterministic evidence-store result"):
        export_trajectory(
            _task(),
            _oracle(),
            [
                (
                    _action("inspect_evidence", {"kinds": ["ocr"]}),
                    '[{"evidence_id":"ocr-1","asset_id":"asset-1","kind":"ocr","content":"Oracle says unsafe."}]',
                )
            ],
        )


def test_trajectory_accepts_and_canonicalizes_real_closed_world_store_results():
    cases = _case_store().search("weight loss", top_k=1)
    evidence = [item.model_dump(mode="json") for item in _evidence_store().inspect("asset-1", {"ocr"})]

    row = export_trajectory(
        _task(),
        _oracle(),
        [
            (_action("search_case", {"query": "weight loss", "top_k": 1}), json.dumps(cases)),
            (_action("inspect_evidence", {"kinds": ["ocr"]}), json.dumps(evidence)),
        ],
    )

    assert row["messages"][3]["content"] == "Tool observation: " + json.dumps(cases, separators=(",", ":"))
    assert row["messages"][5]["content"] == "Tool observation: " + json.dumps(evidence, separators=(",", ":"))


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ([("not-json", "observation")], "valid JSON action"),
        ([( _action("final_decision", {"label": "safe", "confidence": 1.0}), "observation")], "must not be final_decision"),
        ([( _action("get_rule_detail", {"rule_id": "AD-001"}), "one"), (_action("search_case", {"query": "claim"}), "two")], "max_turns"),
    ],
)
def test_trajectory_rejects_invalid_action_or_turn_shape(steps, message):
    with pytest.raises(ValueError, match=message):
        export_trajectory(_task(max_turns=2), _oracle(), steps)


def test_export_rejects_misaligned_task_and_oracle():
    with pytest.raises(ValueError, match="oracle must match"):
        export_track_a(
            _task(),
            Oracle(asset_id="other", policy_version="policy-v1", label="safe"),
        )


def test_cli_validates_every_record_before_creating_output(tmp_path: Path):
    tasks = tmp_path / "tasks.jsonl"
    oracle = tmp_path / "oracle.jsonl"
    output = tmp_path / "nested" / "out.jsonl"
    task = _task().model_dump(mode="json")
    second = {**task, "asset_id": "asset-2"}
    tasks.write_text(
        "\n".join(json.dumps(item) for item in (task, second)) + "\n",
        encoding="utf-8",
    )
    oracle.write_text(_oracle().model_dump_json() + "\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/export_sft.py", "track_a", str(tasks), str(oracle), str(output)],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "missing oracle" in result.stderr
    assert not output.exists()
    assert not output.parent.exists()


def test_cli_exports_deterministic_track_b_jsonl(tmp_path: Path):
    input_path = tmp_path / "trajectory.jsonl"
    oracle_path = tmp_path / "oracle.jsonl"
    cases_path = tmp_path / "cases.jsonl"
    evidence_path = tmp_path / "evidence.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "task": _task().model_dump(mode="json"),
                "steps": [
                    {
                        "action": _action("get_rule_detail", {"rule_id": "AD-001"}),
                        "observation": _rule_observation(),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    oracle_path.write_text(_oracle().model_dump_json() + "\n", encoding="utf-8")
    cases_path.write_text(
        json.dumps({"case_id": "case-1", "text": "Synthetic public case text about a guaranteed weight-loss claim."}) + "\n",
        encoding="utf-8",
    )
    evidence_path.write_text(_evidence_store()._evidence[0].model_dump_json() + "\n", encoding="utf-8")

    command = [
        sys.executable,
        "scripts/export_sft.py",
        "track_b",
        str(input_path),
        str(oracle_path),
        str(output_path),
        "--cases",
        str(cases_path),
        "--evidence",
        str(evidence_path),
    ]
    subprocess.run(command, cwd=Path(__file__).parents[1], check=True)
    first = output_path.read_text(encoding="utf-8")
    subprocess.run(command, cwd=Path(__file__).parents[1], check=True)

    assert output_path.read_text(encoding="utf-8") == first
    row = json.loads(first)
    assert row["messages"][0]["role"] == "system"
    assert json.loads(row["messages"][-1]["content"])["tool"] == "final_decision"


@pytest.mark.parametrize("output_name", ["tasks.jsonl", "oracle.jsonl"])
def test_cli_rejects_output_path_that_resolves_to_an_input_path(tmp_path: Path, output_name: str):
    tasks = tmp_path / "tasks.jsonl"
    oracle = tmp_path / "oracle.jsonl"
    tasks.write_text(_task().model_dump_json() + "\n", encoding="utf-8")
    oracle.write_text(_oracle().model_dump_json() + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_sft.py",
            "track_a",
            str(tasks),
            str(oracle),
            str(tmp_path / "." / output_name),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "must not be the same file" in result.stderr
    assert tasks.read_text(encoding="utf-8") == _task().model_dump_json() + "\n"
    assert oracle.read_text(encoding="utf-8") == _oracle().model_dump_json() + "\n"


def test_synthetic_fixture_task_and_oracle_files_are_parseable_and_aligned():
    root = Path(__file__).parents[1] / "data" / "fixtures"
    policy_document = yaml.safe_load((root / "policies.yaml").read_text(encoding="utf-8"))
    tasks = [Task.model_validate_json(line) for line in (root / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    oracles = [Oracle.model_validate_json(line) for line in (root / "oracle.jsonl").read_text(encoding="utf-8").splitlines()]
    cases = [json.loads(line) for line in (root / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    evidence = [Evidence.model_validate_json(line) for line in (root / "evidence.jsonl").read_text(encoding="utf-8").splitlines()]

    assert policy_document["data_classification"] == "synthetic_public"
    assert {(item.asset_id, item.policy_version) for item in tasks} == {
        (item.asset_id, item.policy_version) for item in oracles
    }
    assert all(set(item) == {"case_id", "text"} for item in cases)
    assert all(item.asset_id == "synthetic-asset-1" for item in evidence)
