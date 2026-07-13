"""Track A GRPO/OPSD bundle tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from risk_agent.contracts import Oracle, PolicyRule, Task
from risk_agent.ms_swift_rl import prepare_rl_bundle
from risk_agent.ms_swift_sft import SplitRatios, _split_for


def _sources(tmp_path: Path, count: int = 8) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tasks = tmp_path / "tasks.jsonl"
    oracles = tmp_path / "oracles.jsonl"
    task_lines, oracle_lines = [], []
    for index in range(count):
        asset_id = f"asset-{index}"
        task_lines.append(Task(
            asset_id=asset_id,
            policy_version="policy-v1",
            active_policy=(PolicyRule(rule_id="R-1", title="Rule", text="No claim"),),
            initial_observation=f"material {index}",
        ).model_dump_json())
        oracle_lines.append(Oracle(
            asset_id=asset_id,
            policy_version="policy-v1",
            label="unsafe" if index % 2 else "safe",
            rule_id="R-1" if index % 2 else None,
            evidence_ids=(f"ev-{index}",) if index % 2 else (),
            risk_level="P1" if index % 2 else None,
        ).model_dump_json())
    tasks.write_text("\n".join(task_lines) + "\n", encoding="utf-8")
    oracles.write_text("\n".join(oracle_lines) + "\n", encoding="utf-8")
    return tasks, oracles


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("mode", ["grpo", "opsd"])
def test_rl_bundle_is_group_split_like_sft_and_prompt_never_contains_answer(
    tmp_path: Path, mode: str
) -> None:
    tasks, oracles = _sources(tmp_path)
    output = tmp_path / mode
    prepared = prepare_rl_bundle(mode, tasks, oracles, output, seed=7)
    manifest = prepared["manifest"]

    assert manifest["mode"] == mode
    assert manifest["track"] == "track_a"
    assert manifest["split_algorithm"] == "sha256-seed-null-asset-id"
    expected = {name: set() for name in ("train", "dev", "holdout")}
    for index in range(8):
        asset_id = f"asset-{index}"
        expected[_split_for(asset_id, 7, SplitRatios())].add(asset_id)
    assert {k: set(v) for k, v in manifest["split_asset_ids"].items()} == expected
    assert manifest["asset_policy_versions"]["asset-0"] == ["policy-v1"]
    assert len(prepared["manifest_sha256"]) == 64
    assert prepared["manifest_sha256"] == hashlib.sha256(
        (output / "manifest.json").read_bytes()
    ).hexdigest()

    all_rows = sum((_rows(output / f"{name}.jsonl") for name in expected), [])
    assert len(all_rows) == 8
    for row in all_rows:
        assert [message["role"] for message in row["messages"]] == ["system", "user"]
        if mode == "grpo":
            assert set(row) == {"messages", "solution"}
            solution = json.loads(row["solution"])
            assert solution["tool"] == "final_decision"
        else:
            assert set(row) == {"messages", "teacher_prompt"}
            teacher = json.loads(row["teacher_prompt"])
            assert set(teacher) == {"student_observation", "privileged_solution", "instruction"}
            assert teacher["student_observation"] == row["messages"][1]["content"]
            assert "Active policy" not in row["teacher_prompt"]


def test_rl_manifest_hashes_exact_bytes_and_contains_policy_metadata_only(tmp_path: Path) -> None:
    tasks, oracles = _sources(tmp_path, 1)
    output = tmp_path / "bundle"
    prepared = prepare_rl_bundle(
        "grpo", tasks, oracles, output,
        ratios=SplitRatios(train=1, dev=0, holdout=0), seed=11,
    )
    manifest = prepared["manifest"]

    payload = (output / "train.jsonl").read_bytes()
    assert manifest["files"]["train.jsonl"] == {
        "bytes": len(payload), "records": 1, "sha256": hashlib.sha256(payload).hexdigest()
    }
    row = _rows(output / "train.jsonl")[0]
    assert set(row) == {"messages", "solution"}
    assert "asset_id" not in row and "policy_version" not in row


@pytest.mark.parametrize("mode", ["grpo", "opsd"])
def test_rl_bundle_preserves_images_for_qwen3_vl(tmp_path: Path, mode: str) -> None:
    tasks, oracles = _sources(tmp_path, 1)
    task = json.loads(tasks.read_text(encoding="utf-8"))
    task["initial_observation"] = "<image>\nReview this advertisement."
    task["images"] = ["assets/ad.jpg"]
    tasks.write_text(json.dumps(task) + "\n", encoding="utf-8")

    output = tmp_path / mode
    prepare_rl_bundle(
        mode,
        tasks,
        oracles,
        output,
        ratios=SplitRatios(train=1, dev=0, holdout=0),
    )

    row = _rows(output / "train.jsonl")[0]
    assert row["images"] == ["assets/ad.jpg"]


def test_rl_bundle_rejects_existing_output_mismatch_and_invalid_mode(tmp_path: Path) -> None:
    tasks, oracles = _sources(tmp_path, 1)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="must not exist"):
        prepare_rl_bundle("grpo", tasks, oracles, existing)
    with pytest.raises(ValueError, match="mode"):
        prepare_rl_bundle("bad", tasks, oracles, tmp_path / "bad")

    oracle = json.loads(oracles.read_text(encoding="utf-8"))
    oracle["asset_id"] = "other"
    oracles.write_text(json.dumps(oracle) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing oracle"):
        prepare_rl_bundle("grpo", tasks, oracles, tmp_path / "mismatch")


def test_rl_bundle_rejects_duplicate_json_and_oracle_rule_not_active(tmp_path: Path) -> None:
    tasks, oracles = _sources(tmp_path, 1)
    original = tasks.read_text(encoding="utf-8").strip()
    tasks.write_text(original[:-1] + ',"asset_id":"duplicate"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        prepare_rl_bundle("grpo", tasks, oracles, tmp_path / "dup")

    tasks, oracles = _sources(tmp_path / "second", 1)
    oracle = json.loads(oracles.read_text(encoding="utf-8"))
    oracle["rule_id"] = "R-not-active"
    oracles.write_text(json.dumps(oracle) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="active_policy"):
        prepare_rl_bundle("opsd", tasks, oracles, tmp_path / "bad-rule")


def test_prepare_rl_cli_emits_manifest_without_traceback(tmp_path: Path) -> None:
    tasks, oracles = _sources(tmp_path, 1)
    output = tmp_path / "cli"
    result = subprocess.run(
        [sys.executable, "scripts/prepare_ms_swift_rl.py", "grpo", str(tasks), str(oracles),
         str(output), "--train-ratio", "1", "--dev-ratio", "0", "--holdout-ratio", "0"],
        cwd=Path(__file__).parents[1], text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["manifest"]["mode"] == "grpo"
    assert len(payload["manifest_sha256"]) == 64


def test_same_asset_multiple_policies_is_deduplicated_in_split_manifest(tmp_path: Path) -> None:
    tasks, oracles = _sources(tmp_path, 1)
    task = json.loads(tasks.read_text(encoding="utf-8"))
    oracle = json.loads(oracles.read_text(encoding="utf-8"))
    task["policy_version"] = "policy-v2"
    oracle["policy_version"] = "policy-v2"
    tasks.write_text(tasks.read_text(encoding="utf-8") + json.dumps(task) + "\n", encoding="utf-8")
    oracles.write_text(oracles.read_text(encoding="utf-8") + json.dumps(oracle) + "\n", encoding="utf-8")
    prepared = prepare_rl_bundle("grpo", tasks, oracles, tmp_path / "bundle")
    manifest = prepared["manifest"]
    assert manifest["asset_policy_versions"] == {"asset-0": ["policy-v1", "policy-v2"]}
    assert sum(manifest["asset_group_counts"].values()) == 1
    assert sum(len(ids) for ids in manifest["split_asset_ids"].values()) == 1


def test_prepare_rejects_huge_seed_and_ratio_cleanly(tmp_path: Path) -> None:
    tasks, oracles = _sources(tmp_path, 1)
    with pytest.raises(ValueError, match="seed"):
        prepare_rl_bundle("grpo", tasks, oracles, tmp_path / "seed", seed=10**10000)
    with pytest.raises(ValueError, match="ratios"):
        prepare_rl_bundle("grpo", tasks, oracles, tmp_path / "ratio",
                          ratios=SplitRatios(train=10**10000, dev=0, holdout=0))
