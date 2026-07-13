"""Safe ms-swift 4.3 Track A GRPO/OPSD planning tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from risk_agent.contracts import Oracle, PolicyRule, Task
from risk_agent.ms_swift_rl import prepare_rl_bundle
from risk_agent.ms_swift_rl_launcher import load_rl_config, prepare_rl_launch, run_rl_launch
from risk_agent.ms_swift_sft import SplitRatios


ROOT = Path(__file__).parents[1]
GRPO = ROOT / "configs/ms_swift/grpo_track_a_qwen3_1_7b_lora.yaml"
OPSD = ROOT / "configs/ms_swift/opsd_track_a_qwen3_1_7b_lora.yaml"
PLUGIN = ROOT / "plugins/ms_swift_risk_rewards.py"


def _bundle(tmp_path: Path, mode: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    task = Task(asset_id="a", policy_version="v", active_policy=(
        PolicyRule(rule_id="R", title="Rule", text="Text"),
    ), initial_observation="material")
    oracle = Oracle(asset_id="a", policy_version="v", label="unsafe", rule_id="R")
    tasks, oracles = tmp_path / "tasks.jsonl", tmp_path / "oracles.jsonl"
    tasks.write_text(task.model_dump_json() + "\n", encoding="utf-8")
    oracles.write_text(oracle.model_dump_json() + "\n", encoding="utf-8")
    output = tmp_path / "bundle"
    prepare_rl_bundle(mode, tasks, oracles, output,
                      ratios=SplitRatios(train=1, dev=0, holdout=0))
    return output


def test_official_grpo_config_has_compatible_single_h20_baseline() -> None:
    config = load_rl_config(GRPO)
    assert config["rlhf_type"] == "grpo"
    assert config["tuner_type"] == "lora"
    assert config["torch_dtype"] == "bfloat16"
    assert config["use_vllm"] is False
    assert config["remove_unused_columns"] is False
    assert config["num_generations"] == 4
    assert config["per_device_train_batch_size"] % config["num_generations"] == 0
    assert config["reward_funcs"] == [
        "risk_format_v1", "risk_label_exact_v1", "risk_rule_exact_v1", "risk_evidence_exact_v1"
    ]
    assert len(config["reward_weights"]) == 4
    assert config["dataset"] == "__TRAIN_DATASET__"
    assert config["model"] == "__MODEL__"
    assert config["adapters"] == "__ADAPTER__"
    assert config["external_plugins"] == "__EXTERNAL_PLUGIN__"


def test_official_opsd_config_is_dynamic_self_teacher_without_second_model() -> None:
    config = load_rl_config(OPSD)
    assert config["rlhf_type"] == "gkd"
    assert config["lmbda"] == 1.0
    assert config["seq_kd"] is False
    assert config["gkd_logits_topk"] == 64
    assert "teacher_model" not in config
    assert "teacher_model_server" not in config
    assert config["use_vllm"] is False
    assert config["remove_unused_columns"] is False


def test_opsd_dry_run_does_not_require_reward_plugin(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "opsd")
    plan = prepare_rl_launch(bundle, OPSD, tmp_path / "run", model="m", adapter="a")
    assert "external_plugins" not in plan.config


@pytest.mark.parametrize(("mode", "config"), [("grpo", GRPO), ("opsd", OPSD)])
def test_dry_run_renders_full_rlhf_plan_without_importing_swift(
    tmp_path: Path, mode: str, config: Path
) -> None:
    bundle = _bundle(tmp_path, mode)
    output = tmp_path / "run"
    plan = prepare_rl_launch(
        bundle, config, output,
        model="Qwen/Qwen3-1.7B", adapter="/checkpoints/sft", plugin=PLUGIN, device="2",
    )
    assert plan.argv == ("swift", "rlhf", "<generated-config>")
    assert plan.env == {"CUDA_VISIBLE_DEVICES": "2", "NPROC_PER_NODE": "1"}
    assert plan.config["dataset"] == [str((bundle / "train.jsonl").resolve())]
    assert plan.config["val_dataset"] == []
    assert plan.config["output_dir"] == str(output.resolve())
    assert plan.config["model"] == "Qwen/Qwen3-1.7B"
    assert plan.config["adapters"] == ["/checkpoints/sft"]
    if mode == "grpo":
        assert plan.config["external_plugins"] == [str(PLUGIN.resolve())]
    else:
        assert "external_plugins" not in plan.config


def test_launcher_rejects_mode_mismatch_tamper_or_existing_output(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "grpo")
    kwargs = dict(model="m", adapter="a", plugin=PLUGIN)
    with pytest.raises(ValueError, match="mode"):
        prepare_rl_launch(bundle, OPSD, tmp_path / "x", **kwargs)
    (bundle / "train.jsonl").write_text('{}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        prepare_rl_launch(bundle, GRPO, tmp_path / "y", **kwargs)

    bundle = _bundle(tmp_path / "second", "grpo")
    existing = tmp_path / "run"
    existing.mkdir()
    with pytest.raises(ValueError, match="must not exist"):
        prepare_rl_launch(bundle, GRPO, existing, **kwargs)


@pytest.mark.parametrize("mode", ["grpo", "opsd"])
def test_launcher_rejects_self_signed_illegal_or_leaking_rows(tmp_path: Path, mode: str) -> None:
    bundle = _bundle(tmp_path, mode)
    row = {"messages": [
        {"role": "system", "content": "x"},
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "oracle leak"},
    ]}
    row["solution" if mode == "grpo" else "teacher_prompt"] = "x"
    payload = (json.dumps(row, separators=(",", ":")) + "\n").encode()
    (bundle / "train.jsonl").write_bytes(payload)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["train.jsonl"].update(
        bytes=len(payload), records=1, sha256=hashlib.sha256(payload).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="messages"):
        prepare_rl_launch(bundle, GRPO if mode == "grpo" else OPSD, tmp_path / "run",
                          model="m", adapter="a", plugin=PLUGIN)


def test_config_rejects_duplicate_unknown_wrong_types_and_teacher_model(tmp_path: Path) -> None:
    duplicate = tmp_path / "dup.yaml"
    duplicate.write_text("rlhf_type: grpo\nrlhf_type: gkd\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_rl_config(duplicate)
    for key, value in (("unknown", 1), ("num_generations", "4"), ("teacher_model", "m")):
        document = yaml.safe_load(GRPO.read_text(encoding="utf-8"))
        document[key] = value
        path = tmp_path / f"{key}.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        with pytest.raises(ValueError, match="config"):
            load_rl_config(path)


def test_real_run_snapshots_exact_bytes_and_uses_shell_false(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "grpo")
    original = (bundle / "train.jsonl").read_bytes()
    seen: dict[str, object] = {}

    def tamper(_: dict[str, object]) -> None:
        (bundle / "train.jsonl").write_text("{}\n", encoding="utf-8")

    def runner(argv, **kwargs):
        seen["argv"], seen["kwargs"] = argv, kwargs
        rendered = yaml.safe_load(Path(argv[2]).read_text(encoding="utf-8"))
        seen["config"] = Path(argv[2])
        seen["train"] = Path(rendered["dataset"][0])
        seen["plugin"] = Path(rendered["external_plugins"][0])
        seen["bytes"] = seen["train"].read_bytes()
        return subprocess.CompletedProcess(argv, 0)

    result = run_rl_launch(
        bundle, GRPO, tmp_path / "run", model="m", adapter="a", plugin=PLUGIN,
        version_reader=lambda _: "4.3.2", executable_finder=lambda _: "C:/bin/swift.exe",
        runner=runner, on_execute=tamper,
    )
    assert result.returncode == 0
    assert seen["argv"][:2] == ["C:/bin/swift.exe", "rlhf"]
    assert seen["kwargs"]["shell"] is False
    assert seen["bytes"] == original
    assert not seen["config"].exists() and not seen["train"].exists() and not seen["plugin"].exists()


def test_rl_launcher_cli_dry_run_prints_plan(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "grpo")
    result = subprocess.run(
        [sys.executable, "scripts/launch_ms_swift_rl.py", str(bundle), "--config", str(GRPO),
         "--output-dir", str(tmp_path / "run"), "--model", "m", "--adapter", "a",
         "--plugin", str(PLUGIN), "--dry-run"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["argv"] == ["swift", "rlhf", "<generated-config>"]
