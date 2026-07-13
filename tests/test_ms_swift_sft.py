"""Tests for preparing leak-free ms-swift SFT bundles."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from risk_agent.contracts import Oracle, PolicyRule, Task
from risk_agent.ms_swift_sft import SplitRatios, prepare_sft_bundle


def _task(asset_id: str, policy_version: str) -> Task:
    return Task(
        asset_id=asset_id,
        policy_version=policy_version,
        active_policy=(
            PolicyRule(
                rule_id="AD-001",
                title="Absolute claim",
                text="Do not guarantee an outcome.",
            ),
        ),
        initial_observation=f"OCR for {asset_id}",
    )


def _oracle(asset_id: str, policy_version: str) -> Oracle:
    return Oracle(
        asset_id=asset_id,
        policy_version=policy_version,
        label="unsafe",
        rule_id="AD-001",
        evidence_ids=(f"evidence-{asset_id}",),
    )


def _write_track_a_sources(tmp_path: Path, keys: list[tuple[str, str]]) -> tuple[Path, Path]:
    tasks = tmp_path / "tasks.jsonl"
    oracles = tmp_path / "oracles.jsonl"
    tasks.write_text(
        "".join(_task(asset, policy).model_dump_json() + "\n" for asset, policy in keys),
        encoding="utf-8",
    )
    oracles.write_text(
        "".join(_oracle(asset, policy).model_dump_json() + "\n" for asset, policy in keys),
        encoding="utf-8",
    )
    return tasks, oracles


def _read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prepare_bundle_groups_every_policy_variant_of_an_asset_in_one_split(tmp_path: Path) -> None:
    keys = [(f"asset-{index}", f"policy-{variant}") for index in range(12) for variant in range(2)]
    tasks, oracles = _write_track_a_sources(tmp_path, keys)

    manifest = prepare_sft_bundle(
        "track_a",
        tasks,
        oracles,
        tmp_path / "bundle",
        ratios=SplitRatios(train=0.5, dev=0.25, holdout=0.25),
        seed=17,
    )

    observed: dict[str, set[str]] = {}
    for split in ("train", "dev", "holdout"):
        for row in _read_rows(tmp_path / "bundle" / f"{split}.jsonl"):
            user_content = row["messages"][1]["content"]
            asset_id = user_content.removeprefix("OCR for ")
            observed.setdefault(asset_id, set()).add(split)
    assert observed and all(len(splits) == 1 for splits in observed.values())
    assert sum(manifest["asset_group_counts"].values()) == 12
    assert sum(manifest["split_counts"].values()) == 24


def test_prepare_bundle_is_deterministic_and_training_rows_only_contain_messages(tmp_path: Path) -> None:
    keys = [(f"asset-{index}", "policy-v1") for index in range(8)]
    tasks, oracles = _write_track_a_sources(tmp_path, keys)

    first = prepare_sft_bundle("track_a", tasks, oracles, tmp_path / "one", seed=901)
    second = prepare_sft_bundle("track_a", tasks, oracles, tmp_path / "two", seed=901)

    for split in ("train", "dev", "holdout"):
        one = (tmp_path / "one" / f"{split}.jsonl").read_bytes()
        two = (tmp_path / "two" / f"{split}.jsonl").read_bytes()
        assert one == two
        assert all(set(row) == {"messages"} for row in _read_rows(tmp_path / "one" / f"{split}.jsonl"))

    manifest = json.loads((tmp_path / "one" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == first
    assert second == first
    assert manifest["track"] == "track_a"


def test_prepare_bundle_allows_empty_splits_and_reports_them(tmp_path: Path) -> None:
    tasks, oracles = _write_track_a_sources(tmp_path, [("only-asset", "policy-v1")])

    manifest = prepare_sft_bundle(
        "track_a",
        tasks,
        oracles,
        tmp_path / "bundle",
        ratios=SplitRatios(train=1.0, dev=0.0, holdout=0.0),
    )

    assert manifest["split_counts"] == {"train": 1, "dev": 0, "holdout": 0}
    assert manifest["asset_group_counts"] == {"train": 1, "dev": 0, "holdout": 0}


def test_manifest_only_keeps_split_metadata_needed_for_experiments(tmp_path: Path) -> None:
    tasks, oracles = _write_track_a_sources(tmp_path, [("asset-1", "policy-v1")])

    manifest = prepare_sft_bundle(
        "track_a",
        tasks,
        oracles,
        tmp_path / "bundle",
        ratios=SplitRatios(train=1, dev=0, holdout=0),
    )

    assert set(manifest) == {
        "track",
        "seed",
        "ratios",
        "split_counts",
        "asset_group_counts",
    }
    assert (tmp_path / "bundle" / "dev.jsonl").read_bytes() == b""


@pytest.mark.parametrize(
    "ratios",
    [
        SplitRatios(train=float("nan"), dev=0.1, holdout=0.9),
        SplitRatios(train=0.8, dev=0.3, holdout=-0.1),
        SplitRatios(train=0.8, dev=0.1, holdout=0.2),
    ],
)
def test_prepare_bundle_rejects_nonfinite_negative_or_nonunit_ratios(
    tmp_path: Path, ratios: SplitRatios
) -> None:
    tasks, oracles = _write_track_a_sources(tmp_path, [("asset-1", "policy-v1")])

    with pytest.raises(ValueError, match="ratios"):
        prepare_sft_bundle("track_a", tasks, oracles, tmp_path / "bundle", ratios=ratios)

    assert not (tmp_path / "bundle").exists()


def test_prepare_bundle_rejects_duplicate_task_key_and_missing_oracle_before_output(tmp_path: Path) -> None:
    tasks, oracles = _write_track_a_sources(tmp_path, [("asset-1", "policy-v1")])
    task_line = tasks.read_text(encoding="utf-8")
    tasks.write_text(task_line + task_line, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate task"):
        prepare_sft_bundle("track_a", tasks, oracles, tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()

    tasks.write_text(_task("asset-2", "policy-v1").model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing oracle"):
        prepare_sft_bundle("track_a", tasks, oracles, tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


def test_prepare_bundle_rejects_duplicate_json_keys_and_extra_input_fields(tmp_path: Path) -> None:
    tasks, oracles = _write_track_a_sources(tmp_path, [("asset-1", "policy-v1")])
    record = json.loads(tasks.read_text(encoding="utf-8"))
    tasks.write_text(
        json.dumps(record)[:-1] + ',"asset_id":"asset-2"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        prepare_sft_bundle("track_a", tasks, oracles, tmp_path / "duplicate-key")

    record["unexpected"] = "must fail closed"
    tasks.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected fields"):
        prepare_sft_bundle("track_a", tasks, oracles, tmp_path / "extra")


def test_prepare_bundle_rejects_existing_output(tmp_path: Path) -> None:
    tasks, oracles = _write_track_a_sources(tmp_path, [("asset-1", "policy-v1")])
    output = tmp_path / "bundle"
    output.mkdir()

    with pytest.raises(ValueError, match="must not exist"):
        prepare_sft_bundle("track_a", tasks, oracles, output)


def test_prepare_track_b_uses_validated_stores_and_only_emits_messages(tmp_path: Path) -> None:
    task = _task("asset-1", "policy-v1")
    trajectory = tmp_path / "trajectory.jsonl"
    oracles = tmp_path / "oracles.jsonl"
    cases = tmp_path / "cases.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "task": task.model_dump(mode="json"),
                "steps": [
                    {
                        "action": json.dumps(
                            {"tool": "get_rule_detail", "arguments": {"rule_id": "AD-001"}}
                        ),
                        "observation": json.dumps(task.active_policy[0].model_dump(mode="json")),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    oracles.write_text(_oracle("asset-1", "policy-v1").model_dump_json() + "\n", encoding="utf-8")
    cases.write_text(json.dumps({"case_id": "case-1", "text": "sanitized public case"}) + "\n", encoding="utf-8")
    evidence.write_text(
        json.dumps(
            {
                "evidence_id": "evidence-asset-1",
                "asset_id": "asset-1",
                "kind": "ocr",
                "content": "OCR for asset-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = prepare_sft_bundle(
        "track_b",
        trajectory,
        oracles,
        tmp_path / "bundle",
        cases_path=cases,
        evidence_path=evidence,
        ratios=SplitRatios(train=1.0, dev=0.0, holdout=0.0),
    )

    assert manifest["track"] == "track_b"
    assert manifest["split_counts"] == {"train": 1, "dev": 0, "holdout": 0}
    assert set(_read_rows(tmp_path / "bundle" / "train.jsonl")[0]) == {"messages"}


def test_prepare_cli_builds_a_bundle_with_explicit_ratios(tmp_path: Path) -> None:
    tasks, oracles = _write_track_a_sources(tmp_path, [("asset-1", "policy-v1")])
    output = tmp_path / "cli-bundle"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_ms_swift_sft.py",
            "track_a",
            str(tasks),
            str(oracles),
            str(output),
            "--train-ratio",
            "1",
            "--dev-ratio",
            "0",
            "--holdout-ratio",
            "0",
            "--seed",
            "7",
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["seed"] == 7
    assert manifest["split_counts"] == {"train": 1, "dev": 0, "holdout": 0}
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest
