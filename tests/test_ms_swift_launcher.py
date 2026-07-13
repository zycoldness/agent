"""Tests for safe, reproducible ms-swift SFT launch planning."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from risk_agent.contracts import Oracle, PolicyRule, Task
from risk_agent.ms_swift_launcher import (
    load_sft_config,
    prepare_sft_launch,
    run_sft_launch,
    validate_ms_swift_runtime,
)
from risk_agent.ms_swift_sft import SplitRatios, prepare_sft_bundle


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "ms_swift" / "sft_qwen3_1_7b_lora.yaml"


def _bundle(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    task = Task(
        asset_id="asset-1",
        policy_version="policy-v1",
        active_policy=(PolicyRule(rule_id="R-1", title="Rule", text="Rule text"),),
        initial_observation="OCR text",
    )
    oracle = Oracle(asset_id="asset-1", policy_version="policy-v1", label="unsafe", rule_id="R-1")
    tasks = tmp_path / "tasks.jsonl"
    oracles = tmp_path / "oracles.jsonl"
    tasks.write_text(task.model_dump_json() + "\n", encoding="utf-8")
    oracles.write_text(oracle.model_dump_json() + "\n", encoding="utf-8")
    output = tmp_path / "bundle"
    prepare_sft_bundle(
        "track_a",
        tasks,
        oracles,
        output,
        ratios=SplitRatios(train=1.0, dev=0.0, holdout=0.0),
    )
    return output


def test_official_qwen_lora_config_has_safe_single_h20_defaults() -> None:
    config = load_sft_config(CONFIG)

    assert config["model"] == "Qwen/Qwen3-1.7B"
    assert config["tuner_type"] == "lora"
    assert config["torch_dtype"] == "bfloat16"
    assert config["max_length"] == 4096
    assert config["per_device_train_batch_size"] == 1
    assert config["gradient_accumulation_steps"] >= 8
    assert config["gradient_checkpointing"] is True
    assert config["strict"] is True
    assert config["dataset"] == "__TRAIN_DATASET__"
    assert config["val_dataset"] == "__VAL_DATASET__"
    assert config["output_dir"] == "__OUTPUT_DIR__"
    assert "deepspeed" not in config
    assert "vllm" not in config
    assert "quant_bits" not in config
    assert "train_type" not in config


def test_train_extra_pins_the_supported_ms_swift_minor() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["optional-dependencies"]["train"] == ["ms-swift>=4.3,<4.4"]


def test_dry_run_validates_bundle_and_renders_config_without_runtime_import(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    plan = prepare_sft_launch(bundle, CONFIG, tmp_path / "runs" / "smoke", device="2")

    assert plan.argv[:3] == ("swift", "sft", "--config")
    assert plan.argv[3] == "<generated-config>"
    assert plan.env == {"CUDA_VISIBLE_DEVICES": "2", "NPROC_PER_NODE": "1"}
    assert plan.config["dataset"] == [str((bundle / "train.jsonl").resolve())]
    assert plan.config["val_dataset"] == []
    assert plan.config["output_dir"] == str((tmp_path / "runs" / "smoke").resolve())
    assert plan.config["model"] == "Qwen/Qwen3-1.7B"


def test_launch_plan_accepts_model_override_and_rejects_path_injection(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    plan = prepare_sft_launch(
        bundle,
        CONFIG,
        tmp_path / "run",
        model="/models/local-qwen",
    )
    assert plan.config["model"] == "/models/local-qwen"

    with pytest.raises(ValueError, match="device"):
        prepare_sft_launch(bundle, CONFIG, tmp_path / "bad", device="0;touch /tmp/pwned")


def test_launch_plan_rejects_tampered_or_incomplete_bundle(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "train.jsonl").write_text('{"messages":[]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        prepare_sft_launch(bundle, CONFIG, tmp_path / "run")

    bundle = _bundle(tmp_path / "second")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "partial"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="complete"):
        prepare_sft_launch(bundle, CONFIG, tmp_path / "run-2")


def test_launch_plan_rejects_empty_train_split(tmp_path: Path) -> None:
    task = Task(
        asset_id="asset-1",
        policy_version="policy-v1",
        active_policy=(PolicyRule(rule_id="R-1", title="Rule", text="Rule text"),),
        initial_observation="OCR text",
    )
    oracle = Oracle(asset_id="asset-1", policy_version="policy-v1", label="unsafe", rule_id="R-1")
    tasks = tmp_path / "tasks.jsonl"
    oracles = tmp_path / "oracles.jsonl"
    tasks.write_text(task.model_dump_json() + "\n", encoding="utf-8")
    oracles.write_text(oracle.model_dump_json() + "\n", encoding="utf-8")
    bundle = tmp_path / "empty-train"
    prepare_sft_bundle(
        "track_a",
        tasks,
        oracles,
        bundle,
        ratios=SplitRatios(train=0.0, dev=0.0, holdout=1.0),
    )

    with pytest.raises(ValueError, match="train split must not be empty"):
        prepare_sft_launch(bundle, CONFIG, tmp_path / "run")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest["sources"]["input"].update({"sha256": "bad"}), "source metadata"),
        (lambda manifest: manifest["asset_group_counts"].update({"train": -1}), "asset group counts"),
        (lambda manifest: manifest.update({"unexpected": True}), "schema"),
    ],
)
def test_launch_plan_rejects_noncanonical_manifest_metadata(tmp_path: Path, mutation, message: str) -> None:
    bundle = _bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        prepare_sft_launch(bundle, CONFIG, tmp_path / "run")


def test_config_rejects_duplicate_unknown_and_obsolete_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("model: one\nmodel: two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_sft_config(duplicate)

    for key in ("unknown_option", "train_type"):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document[key] = "lora"
        invalid = tmp_path / f"{key}.yaml"
        invalid.write_text(yaml.safe_dump(document), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown config keys"):
            load_sft_config(invalid)


@pytest.mark.parametrize("version", ["4.2.9", "4.4.0", "5.0.0"])
def test_runtime_rejects_ms_swift_outside_supported_minor(version: str) -> None:
    with pytest.raises(RuntimeError, match=r">=4\.3,<4\.4"):
        validate_ms_swift_runtime(lambda _: version, lambda _: "C:/bin/swift.exe")


def test_runtime_accepts_43_and_requires_swift_executable() -> None:
    assert validate_ms_swift_runtime(lambda _: "4.3.2", lambda _: "C:/bin/swift.exe") == "C:/bin/swift.exe"
    with pytest.raises(RuntimeError, match="executable"):
        validate_ms_swift_runtime(lambda _: "4.3.2", lambda _: None)


def test_real_runner_uses_argv_environment_and_cleans_generated_config(tmp_path: Path) -> None:
    plan = prepare_sft_launch(_bundle(tmp_path), CONFIG, tmp_path / "run", device="3")
    observed: dict[str, object] = {}

    def fake_runner(argv, **kwargs):
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        config_path = Path(argv[3])
        observed["config_path"] = config_path
        observed["rendered"] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(argv, 0)

    result = run_sft_launch(
        plan,
        version_reader=lambda _: "4.3.1",
        executable_finder=lambda _: "C:/bin/swift.exe",
        runner=fake_runner,
    )

    assert result.returncode == 0
    assert observed["argv"][:3] == ["C:/bin/swift.exe", "sft", "--config"]
    assert observed["rendered"] == plan.config
    kwargs = observed["kwargs"]
    assert kwargs["shell"] is False
    assert kwargs["check"] is True
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert kwargs["env"]["NPROC_PER_NODE"] == "1"
    assert not observed["config_path"].exists()


def test_launcher_cli_dry_run_prints_full_plan_without_ms_swift(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/launch_ms_swift_sft.py",
            str(bundle),
            "--config",
            str(CONFIG),
            "--output-dir",
            str(tmp_path / "run"),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["argv"] == ["swift", "sft", "--config", "<generated-config>"]
    assert payload["rendered_config"]["dataset"] == [str((bundle / "train.jsonl").resolve())]
    assert payload["rendered_config"]["val_dataset"] == []
    assert payload["rendered_config"]["eval_strategy"] == "no"
