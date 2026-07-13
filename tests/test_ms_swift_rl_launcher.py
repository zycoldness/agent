"""Safe ms-swift 4.3 Track A GRPO/OPSD planning tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import risk_agent.ms_swift_rl_launcher as rl_launcher
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


def _digest(bundle: Path) -> str:
    return hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()


def _plugin_digest(path: Path = PLUGIN) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    assert config["reward_weights"] == [0.05, 0.6, 0.2, 0.15]
    assert config["reward_weights"][1] > sum(config["reward_weights"][0:1] + config["reward_weights"][2:])
    assert config["dataset"] == "__TRAIN_DATASET__"
    assert config["model"] == "__MODEL__"
    assert config["adapters"] == "__ADAPTER__"
    assert config["ref_adapters"] == "__ADAPTER__"
    assert config["external_plugins"] == "__EXTERNAL_PLUGIN__"


def test_official_opsd_config_is_dynamic_self_teacher_without_second_model() -> None:
    config = load_rl_config(OPSD)
    assert config["rlhf_type"] == "gkd"
    assert config["lmbda"] == 1.0
    assert config["seq_kd"] is False
    assert config["gkd_logits_topk"] == 64
    assert "teacher_model" not in config
    assert "teacher_model_server" not in config
    assert config["use_vllm"] is True
    assert config["remove_unused_columns"] is False
    assert config["vllm_mode"] == "colocate"
    assert 0 < config["vllm_gpu_memory_utilization"] <= 0.5
    assert config["sleep_level"] == 1
    assert config["offload_model"] is True
    assert config["offload_optimizer"] is True


def test_opsd_dry_run_does_not_require_reward_plugin(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "opsd")
    plan = prepare_rl_launch(bundle, OPSD, tmp_path / "run", model="m", adapter="a",
                             expected_manifest_sha256=_digest(bundle))
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
        expected_manifest_sha256=_digest(bundle),
        expected_plugin_sha256=_plugin_digest() if mode == "grpo" else None,
    )
    assert plan.argv == ("swift", "rlhf", "<generated-config>")
    assert plan.env == {"CUDA_VISIBLE_DEVICES": "2", "NPROC_PER_NODE": "1"}
    assert plan.config["dataset"] == [str((bundle / "train.jsonl").resolve())]
    assert plan.config["val_dataset"] == []
    assert plan.config["output_dir"] == str(output.resolve())
    assert plan.config["model"] == "Qwen/Qwen3-1.7B"
    assert plan.config["adapters"] == ["/checkpoints/sft"]
    if mode == "grpo":
        assert plan.config["ref_adapters"] == ["/checkpoints/sft"]
    if mode == "grpo":
        assert plan.config["external_plugins"] == [str(PLUGIN.resolve())]
    else:
        assert "external_plugins" not in plan.config


def test_launcher_rejects_mode_mismatch_tamper_or_existing_output(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "grpo")
    kwargs = dict(model="m", adapter="a", plugin=PLUGIN,
                  expected_manifest_sha256=_digest(bundle),
                  expected_plugin_sha256=_plugin_digest())
    with pytest.raises(ValueError, match="mode"):
        prepare_rl_launch(bundle, OPSD, tmp_path / "x", **kwargs)
    (bundle / "train.jsonl").write_text('{}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        prepare_rl_launch(bundle, GRPO, tmp_path / "y", **kwargs)

    bundle = _bundle(tmp_path / "second", "grpo")
    kwargs["expected_manifest_sha256"] = _digest(bundle)
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
                          model="m", adapter="a", plugin=PLUGIN,
                          expected_manifest_sha256=_digest(bundle),
                          expected_plugin_sha256=_plugin_digest() if mode == "grpo" else None)


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


def test_config_and_cli_reject_huge_epoch_without_traceback(tmp_path: Path) -> None:
    document = yaml.safe_load(GRPO.read_text(encoding="utf-8"))
    document["num_train_epochs"] = 10**400
    config = tmp_path / "huge-epoch.yaml"
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid RL config"):
        load_rl_config(config)

    bundle = _bundle(tmp_path / "data", "grpo")
    result = subprocess.run(
        [sys.executable, "scripts/launch_ms_swift_rl.py", str(bundle), "--config", str(config),
         "--output-dir", str(tmp_path / "run"), "--model", "m", "--adapter", "a",
         "--plugin", str(PLUGIN), "--expected-manifest-sha256", _digest(bundle),
         "--expected-plugin-sha256", _plugin_digest(), "--dry-run"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_grpo_requires_trusted_plugin_digest_and_rejects_wrong_digest(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "grpo")
    with pytest.raises(ValueError, match="plugin digest"):
        prepare_rl_launch(bundle, GRPO, tmp_path / "missing", model="m", adapter="a",
                          plugin=PLUGIN, expected_manifest_sha256=_digest(bundle))
    with pytest.raises(ValueError, match="trusted plugin digest"):
        prepare_rl_launch(bundle, GRPO, tmp_path / "wrong", model="m", adapter="a",
                          plugin=PLUGIN, expected_manifest_sha256=_digest(bundle),
                          expected_plugin_sha256="0" * 64)


def test_same_inode_same_size_plugin_overwrite_is_rejected_by_trusted_digest(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "grpo")
    plugin = tmp_path / "plugin.py"
    original = PLUGIN.read_bytes()
    plugin.write_bytes(original)
    trusted = _plugin_digest(plugin)
    before = plugin.stat()
    plugin.write_bytes(b"#" * len(original))
    after = plugin.stat()
    assert (before.st_ino, before.st_size) == (after.st_ino, after.st_size)
    with pytest.raises(ValueError, match="trusted plugin digest"):
        prepare_rl_launch(bundle, GRPO, tmp_path / "run", model="m", adapter="a",
                          plugin=plugin, expected_manifest_sha256=_digest(bundle),
                          expected_plugin_sha256=trusted)


def test_parent_replacement_between_render_and_reserve_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path / "data", "grpo")
    parent = tmp_path / "runs"
    parent.mkdir()
    displaced = tmp_path / "original-runs"
    original_render = rl_launcher._render

    def render_then_replace(*args, **kwargs):
        plan = original_render(*args, **kwargs)
        parent.rename(displaced)
        parent.mkdir()
        (parent / "competitor.txt").write_text("keep", encoding="utf-8")
        return plan

    monkeypatch.setattr(rl_launcher, "_render", render_then_replace)
    called = False

    def runner(argv, **kwargs):
        nonlocal called
        called = True
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(RuntimeError, match="parent.*identity changed"):
        run_rl_launch(
            bundle, GRPO, parent / "run", model="m", adapter="a", plugin=PLUGIN,
            expected_manifest_sha256=_digest(bundle), expected_plugin_sha256=_plugin_digest(),
            version_reader=lambda _: "4.3.2",
            executable_finder=lambda _: "C:/bin/swift.exe", runner=runner,
        )
    assert called is False
    assert (parent / "competitor.txt").read_text(encoding="utf-8") == "keep"


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
        expected_manifest_sha256=_digest(bundle),
        expected_plugin_sha256=_plugin_digest(),
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
         "--plugin", str(PLUGIN), "--expected-manifest-sha256", _digest(bundle),
         "--expected-plugin-sha256", _plugin_digest(), "--dry-run"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["argv"] == ["swift", "rlhf", "<generated-config>"]


def test_external_manifest_digest_rejects_fully_self_signed_tamper(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "grpo")
    trusted = _digest(bundle)
    train = bundle / "train.jsonl"
    row = json.loads(train.read_text(encoding="utf-8"))
    row["solution"] = row["solution"].replace('"unsafe"', '"safe"')
    payload = (json.dumps(row, separators=(",", ":")) + "\n").encode()
    train.write_bytes(payload)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["train.jsonl"].update(
        bytes=len(payload), records=1, sha256=hashlib.sha256(payload).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="trusted manifest digest"):
        prepare_rl_launch(bundle, GRPO, tmp_path / "run", model="m", adapter="a",
                          plugin=PLUGIN, expected_manifest_sha256=trusted,
                          expected_plugin_sha256=_plugin_digest())


def test_opsd_self_signed_teacher_prompt_must_bind_student_and_privileged_solution(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "opsd")
    train = bundle / "train.jsonl"
    row = json.loads(train.read_text(encoding="utf-8"))
    teacher = json.loads(row["teacher_prompt"])
    teacher["student_observation"] = "different observation"
    row["teacher_prompt"] = json.dumps(teacher, separators=(",", ":"))
    payload = (json.dumps(row, separators=(",", ":")) + "\n").encode()
    train.write_bytes(payload)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["train.jsonl"].update(
        bytes=len(payload), records=1, sha256=hashlib.sha256(payload).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="teacher_prompt"):
        prepare_rl_launch(bundle, OPSD, tmp_path / "run", model="m", adapter="a",
                          expected_manifest_sha256=_digest(bundle))


def test_plugin_symlink_is_rejected(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "grpo")
    link = tmp_path / "plugin.py"
    try:
        link.symlink_to(PLUGIN)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="symbolic link"):
        prepare_rl_launch(bundle, GRPO, tmp_path / "run", model="m", adapter="a",
                          plugin=link, expected_manifest_sha256=_digest(bundle),
                          expected_plugin_sha256=_plugin_digest())


def test_real_run_reserves_output_and_cleans_empty_reservation_on_failure(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "grpo")
    output = tmp_path / "run"

    def runner(argv, **kwargs):
        assert output.is_dir()
        raise subprocess.CalledProcessError(1, argv)

    with pytest.raises(subprocess.CalledProcessError):
        run_rl_launch(bundle, GRPO, output, model="m", adapter="a", plugin=PLUGIN,
                      expected_manifest_sha256=_digest(bundle),
                      expected_plugin_sha256=_plugin_digest(),
                      version_reader=lambda _: "4.3.2",
                      executable_finder=lambda _: "C:/bin/swift.exe", runner=runner)
    assert not output.exists()


def test_output_identity_swap_is_detected_before_exec_and_replacement_preserved(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "grpo")
    output, displaced = tmp_path / "run", tmp_path / "reserved"
    called = False

    def swap(_: dict[str, object]) -> None:
        output.rename(displaced)
        output.mkdir()
        (output / "competitor.txt").write_text("keep", encoding="utf-8")

    def runner(argv, **kwargs):
        nonlocal called
        called = True
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(RuntimeError, match="identity changed"):
        run_rl_launch(bundle, GRPO, output, model="m", adapter="a", plugin=PLUGIN,
                      expected_manifest_sha256=_digest(bundle),
                      expected_plugin_sha256=_plugin_digest(),
                      version_reader=lambda _: "4.3.2",
                      executable_finder=lambda _: "C:/bin/swift.exe", runner=runner,
                      on_execute=swap)
    assert called is False
    assert (output / "competitor.txt").read_text(encoding="utf-8") == "keep"


def test_huge_manifest_ratio_is_controlled_value_error(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "grpo")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["ratios"] = {"train": 10**400, "dev": 0, "holdout": 0}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="ratios"):
        prepare_rl_launch(bundle, GRPO, tmp_path / "run", model="m", adapter="a",
                          plugin=PLUGIN, expected_manifest_sha256=_digest(bundle),
                          expected_plugin_sha256=_plugin_digest())


def test_repository_fixture_prepares_then_dry_runs(tmp_path: Path) -> None:
    prepared = prepare_rl_bundle(
        "grpo", ROOT / "data/fixtures/tasks.jsonl", ROOT / "data/fixtures/oracle.jsonl",
        tmp_path / "bundle", ratios=SplitRatios(train=1, dev=0, holdout=0),
    )
    plan = prepare_rl_launch(
        tmp_path / "bundle", GRPO, tmp_path / "run", model="m", adapter="a", plugin=PLUGIN,
        expected_manifest_sha256=prepared["manifest_sha256"],
        expected_plugin_sha256=_plugin_digest(),
    )
    assert plan.argv[:2] == ("swift", "rlhf")
