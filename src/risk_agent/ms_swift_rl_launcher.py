"""Safely validate and launch single-device ms-swift 4.3 Track A RL."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from risk_agent.ms_swift_launcher import (
    LaunchPlan,
    _UniqueKeyLoader,
    _strict_json_loads,
    _validate_override,
    _write_private_file,
    validate_ms_swift_runtime,
)
from risk_agent.rl_rewards import _decision


_SPLIT_FILES = ("train.jsonl", "dev.jsonl", "holdout.jsonl")
_MANIFEST_FIELDS = frozenset({
    "schema_version", "format", "status", "track", "mode", "seed", "ratios",
    "split_algorithm", "sources", "split_counts", "asset_group_counts",
    "split_asset_ids", "asset_policy_versions", "files",
})
_COMMON = frozenset({
    "rlhf_type", "model", "adapters", "tuner_type", "torch_dtype", "dataset",
    "val_dataset", "output_dir", "use_vllm", "remove_unused_columns", "max_completion_length",
    "per_device_train_batch_size", "per_device_eval_batch_size",
    "gradient_accumulation_steps", "gradient_checkpointing", "learning_rate",
    "num_train_epochs", "lora_rank", "lora_alpha", "target_modules", "logging_steps",
    "save_steps", "eval_steps", "save_total_limit", "warmup_ratio", "report_to", "strict",
})
_GRPO = _COMMON | frozenset({
    "external_plugins", "reward_funcs", "reward_weights", "num_generations", "max_prompt_length",
    "ref_adapters",
})
_OPSD = _COMMON | frozenset({
    "lmbda", "beta", "seq_kd", "gkd_logits_topk", "temperature", "vllm_mode",
    "vllm_gpu_memory_utilization", "sleep_level", "offload_model", "offload_optimizer",
})
_REWARDS = [
    "risk_format_v1", "risk_label_exact_v1", "risk_rule_exact_v1", "risk_evidence_exact_v1"
]


@dataclass(frozen=True)
class _Bundle:
    root: Path
    manifest: dict[str, Any]
    split_bytes: dict[str, bytes]


def _is_int(value: object, minimum: int = 1) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and minimum <= value <= (2**63 - 1)
    )


def _is_float(value: object, *, low: float = 0, high: float = 1, high_open: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
        return False
    return value >= low and (value < high if high_open else value <= high)


def load_rl_config(path: Path) -> dict[str, Any]:
    """Load one exact GRPO or dynamic OPSD template."""

    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read RL config: {path}") from error
    if not isinstance(document, dict):
        raise ValueError("RL config must be a mapping")
    rlhf_type = document.get("rlhf_type")
    allowed = _GRPO if rlhf_type == "grpo" else _OPSD if rlhf_type == "gkd" else frozenset()
    if not allowed or set(document) != allowed:
        raise ValueError("RL config has unknown or missing config keys")
    literals = {
        "model": "__MODEL__", "adapters": "__ADAPTER__", "tuner_type": "lora",
        "torch_dtype": "bfloat16", "dataset": "__TRAIN_DATASET__",
        "val_dataset": "__VAL_DATASET__", "output_dir": "__OUTPUT_DIR__",
        "target_modules": "all-linear", "report_to": "none", "strict": True,
        "gradient_checkpointing": True, "remove_unused_columns": False,
    }
    if any(document.get(key) != expected or type(document.get(key)) is not type(expected)
           for key, expected in literals.items()):
        raise ValueError("invalid RL config values or types")
    integer_keys = {
        "max_completion_length", "per_device_train_batch_size", "per_device_eval_batch_size",
        "gradient_accumulation_steps", "lora_rank", "lora_alpha", "logging_steps",
        "save_steps", "eval_steps", "save_total_limit",
    }
    if any(not _is_int(document[key]) for key in integer_keys):
        raise ValueError("invalid RL config values or types")
    if not _is_float(document["learning_rate"], low=0, high=1) or document["learning_rate"] == 0:
        raise ValueError("invalid RL config values or types")
    if not _is_float(document["warmup_ratio"], low=0, high=1, high_open=True):
        raise ValueError("invalid RL config values or types")
    epochs = document["num_train_epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, (int, float)) or not math.isfinite(float(epochs)) or epochs <= 0:
        raise ValueError("invalid RL config values or types")
    if rlhf_type == "grpo":
        if document["use_vllm"] is not False or document["ref_adapters"] != "__ADAPTER__":
            raise ValueError("invalid GRPO config")
        if document["external_plugins"] != "__EXTERNAL_PLUGIN__":
            raise ValueError("invalid GRPO config")
        if document["reward_funcs"] != _REWARDS:
            raise ValueError("invalid GRPO config")
        weights = document["reward_weights"]
        if (not isinstance(weights, list) or len(weights) != len(_REWARDS)
                or any(not isinstance(x, float) or not math.isfinite(x) or x < 0 for x in weights)):
            raise ValueError("invalid GRPO config")
        if weights != [0.05, 0.6, 0.2, 0.15] or weights[1] <= sum(weights[:1] + weights[2:]):
            raise ValueError("invalid GRPO reward weights")
        for key in ("num_generations", "max_prompt_length"):
            if not _is_int(document[key]):
                raise ValueError("invalid GRPO config")
        if document["per_device_train_batch_size"] % document["num_generations"]:
            raise ValueError("GRPO train batch must be divisible by num_generations")
    else:
        if (
            document["use_vllm"] is not True
            or document["vllm_mode"] != "colocate"
            or not _is_float(document["vllm_gpu_memory_utilization"], low=0, high=0.5)
            or document["vllm_gpu_memory_utilization"] == 0
            or document["sleep_level"] != 1
            or document["offload_model"] is not True
            or document["offload_optimizer"] is not True
        ):
            raise ValueError("invalid OPSD colocate vLLM config")
        if not _is_float(document["lmbda"]) or not _is_float(document["beta"]):
            raise ValueError("invalid OPSD config")
        if type(document["seq_kd"]) is not bool or document["seq_kd"] is not False:
            raise ValueError("invalid OPSD config")
        if not _is_int(document["gkd_logits_topk"]):
            raise ValueError("invalid OPSD config")
        if not _is_float(document["temperature"], low=0, high=float("inf")) or document["temperature"] == 0:
            raise ValueError("invalid OPSD config")
    return document


def _validate_messages(messages: object, filename: str) -> None:
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"RL messages must contain system and user only: {filename}")
    for message, role in zip(messages, ("system", "user"), strict=True):
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message.get("role") != role or not isinstance(message.get("content"), str)
                or not message["content"]):
            raise ValueError(f"RL messages schema is invalid: {filename}")


def _validate_row(row: object, mode: str, filename: str) -> None:
    expected = {"messages", "solution"} if mode == "grpo" else {"messages", "teacher_prompt"}
    if not isinstance(row, dict) or set(row) != expected:
        raise ValueError(f"RL row schema is invalid: {filename}")
    _validate_messages(row["messages"], filename)
    if mode == "grpo":
        if _decision(row["solution"]) is None:
            raise ValueError(f"RL solution schema is invalid: {filename}")
    else:
        if not isinstance(row["teacher_prompt"], str) or not row["teacher_prompt"]:
            raise ValueError(f"RL teacher_prompt schema is invalid: {filename}")
        teacher = _strict_json_loads(row["teacher_prompt"], f"RL teacher_prompt in {filename}")
        if (
            not isinstance(teacher, dict)
            or set(teacher) != {"student_observation", "privileged_solution", "instruction"}
            or teacher["student_observation"] != row["messages"][1]["content"]
            or teacher["instruction"] != "Produce exactly the best final_decision JSON action."
            or _decision(teacher["privileged_solution"]) is None
        ):
            raise ValueError(f"RL teacher_prompt schema or binding is invalid: {filename}")


def _verify_bundle(bundle_dir: Path, expected_manifest_sha256: str) -> _Bundle:
    if not isinstance(expected_manifest_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", expected_manifest_sha256
    ) is None:
        raise ValueError("expected trusted manifest digest must be 64 lowercase hex characters")
    try:
        root = bundle_dir.resolve(strict=True)
    except OSError as error:
        raise ValueError("cannot resolve RL bundle") from error
    if not root.is_dir():
        raise ValueError("RL bundle path must be a directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("RL manifest must not be a symbolic link")
    try:
        manifest_payload = manifest_path.read_bytes()
        manifest_text = manifest_payload.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("RL manifest is missing or invalid") from error
    if hashlib.sha256(manifest_payload).hexdigest() != expected_manifest_sha256:
        raise ValueError("RL bundle does not match the trusted manifest digest")
    manifest = _strict_json_loads(manifest_text, "RL manifest")
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("RL manifest has invalid schema")
    if (manifest["schema_version"] != 1 or manifest["format"] != "ms-swift-track-a-rl-jsonl"
            or manifest["status"] != "complete" or manifest["track"] != "track_a"
            or manifest["mode"] not in ("grpo", "opsd")):
        raise ValueError("RL manifest schema, track, mode, or status is invalid")
    if not _is_int(manifest["seed"], minimum=0):
        raise ValueError("RL manifest seed is invalid")
    if manifest["split_algorithm"] != "sha256-seed-null-asset-id":
        raise ValueError("RL split algorithm is invalid")
    ratios = manifest["ratios"]
    try:
        ratio_invalid = (
            not isinstance(ratios, dict)
            or set(ratios) != {"train", "dev", "holdout"}
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(float(v))
                or v < 0
                for v in ratios.values()
            )
            or not math.isclose(sum(ratios.values()), 1.0)
        )
    except (OverflowError, TypeError, ValueError):
        ratio_invalid = True
    if ratio_invalid:
        raise ValueError("RL ratios are invalid")
    sources = manifest["sources"]
    if not isinstance(sources, dict) or set(sources) != {"input", "oracle"}:
        raise ValueError("RL source metadata is invalid")
    for source in sources.values():
        if (not isinstance(source, dict) or set(source) != {"bytes", "sha256"}
                or not _is_int(source["bytes"], 0)
                or not isinstance(source["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])):
            raise ValueError("RL source metadata is invalid")
    for key in ("split_counts", "asset_group_counts"):
        counts = manifest[key]
        if (not isinstance(counts, dict) or set(counts) != {"train", "dev", "holdout"}
                or any(not _is_int(v, 0) for v in counts.values())):
            raise ValueError(f"RL {key} is invalid")
    asset_ids = manifest["split_asset_ids"]
    if (not isinstance(asset_ids, dict) or set(asset_ids) != {"train", "dev", "holdout"}
            or any(not isinstance(v, list) or v != sorted(set(v)) or any(not isinstance(x, str) or not x for x in v)
                   for v in asset_ids.values())):
        raise ValueError("RL split asset ids are invalid")
    if set(asset_ids["train"]) & set(asset_ids["dev"]) or set(asset_ids["train"]) & set(asset_ids["holdout"]) or set(asset_ids["dev"]) & set(asset_ids["holdout"]):
        raise ValueError("RL asset groups overlap")
    policy_versions = manifest["asset_policy_versions"]
    all_assets = set().union(*(set(values) for values in asset_ids.values()))
    if (
        not isinstance(policy_versions, dict)
        or set(policy_versions) != all_assets
        or any(
            not isinstance(asset, str)
            or not asset
            or not isinstance(versions, list)
            or versions != sorted(set(versions))
            or not versions
            or any(not isinstance(version, str) or not version for version in versions)
            for asset, versions in policy_versions.items()
        )
        or any(manifest["asset_group_counts"][split] != len(asset_ids[split]) for split in asset_ids)
    ):
        raise ValueError("RL asset/policy group metadata is invalid")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(_SPLIT_FILES):
        raise ValueError("RL files metadata is invalid")
    split_bytes: dict[str, bytes] = {}
    for filename in _SPLIT_FILES:
        entry = files[filename]
        if (not isinstance(entry, dict) or set(entry) != {"bytes", "records", "sha256"}
                or not _is_int(entry["bytes"], 0) or not _is_int(entry["records"], 0)
                or not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
            raise ValueError("RL files metadata is invalid")
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"RL bundle file is missing or unsafe: {filename}")
        payload = path.read_bytes()
        if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"RL bundle file hash or size mismatch: {filename}")
        try:
            lines = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ValueError(f"RL bundle file must be UTF-8: {filename}") from error
        split = filename.removesuffix(".jsonl")
        if len(lines) != entry["records"] or len(lines) != manifest["split_counts"][split]:
            raise ValueError(f"RL record count mismatch: {filename}")
        for line in lines:
            _validate_row(_strict_json_loads(line, f"RL row in {filename}"), manifest["mode"], filename)
        split_bytes[filename] = payload
    if manifest["split_counts"]["train"] == 0:
        raise ValueError("RL train split must not be empty")
    return _Bundle(root, manifest, split_bytes)


def _read_plugin(path: Path) -> tuple[Path, bytes]:
    if path.is_symlink():
        raise ValueError("external plugin must not be a symbolic link")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError("cannot resolve external plugin") from error
    if not resolved.is_file() or resolved.suffix != ".py":
        raise ValueError("external plugin must be a regular Python file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("cannot safely open external plugin") from error
    try:
        held = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(held.st_mode) or not os.path.samestat(held, current):
            raise ValueError("external plugin identity changed or is unsafe")
        if held.st_size > 1024 * 1024:
            raise ValueError("external plugin exceeds one MiB")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read()
        if len(payload) != held.st_size:
            raise ValueError("external plugin changed while being read")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return resolved, payload


def _render(bundle: _Bundle, template: dict[str, Any], output_dir: Path, *, model: str,
            adapter: str, plugin_path: Path | None, train: Path, dev: Path, device: str) -> LaunchPlan:
    expected = "grpo" if template["rlhf_type"] == "grpo" else "opsd"
    if bundle.manifest["mode"] != expected:
        raise ValueError("RL bundle mode does not match config mode")
    for value, name in ((model, "model"), (adapter, "adapter")):
        _validate_override(value, name)
    if not re.fullmatch(r"[0-9]+", device):
        raise ValueError("device must be one non-negative CUDA device index")
    output = output_dir.resolve(strict=False)
    if output.exists():
        raise ValueError("training output directory must not exist")
    if output == bundle.root or bundle.root in output.parents:
        raise ValueError("training output directory must be outside the bundle")
    rendered = dict(template)
    rendered.update(model=model, adapters=[adapter], dataset=[str(train)], output_dir=str(output))
    if bundle.manifest["split_counts"]["dev"]:
        rendered.update(val_dataset=[str(dev)], eval_strategy="steps")
    else:
        rendered.update(val_dataset=[], eval_strategy="no")
    if expected == "grpo":
        assert plugin_path is not None
        rendered["external_plugins"] = [str(plugin_path)]
        rendered["ref_adapters"] = [adapter]
    return LaunchPlan(("swift", "rlhf", "<generated-config>"),
                      {"CUDA_VISIBLE_DEVICES": device, "NPROC_PER_NODE": "1"}, rendered)


def prepare_rl_launch(
    bundle_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    model: str,
    adapter: str,
    expected_manifest_sha256: str,
    plugin: Path | None = None,
    device: str = "0",
) -> LaunchPlan:
    bundle = _verify_bundle(bundle_dir, expected_manifest_sha256)
    template = load_rl_config(config_path)
    if template["rlhf_type"] == "grpo" and plugin is None:
        raise ValueError("GRPO requires an external reward plugin")
    plugin_path = _read_plugin(plugin)[0] if plugin is not None and template["rlhf_type"] == "grpo" else None
    return _render(bundle, template, output_dir, model=model, adapter=adapter,
                   plugin_path=plugin_path, train=bundle.root / "train.jsonl",
                   dev=bundle.root / "dev.jsonl", device=device)


def _reserve_output(output_dir: Path) -> os.stat_result:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as error:
        raise ValueError("training output directory must not exist") from error
    identity = os.stat(output_dir, follow_symlinks=False)
    if os.name == "posix":
        output_dir.chmod(0o700)
    return identity


def _require_output_identity(output_dir: Path, identity: os.stat_result) -> None:
    try:
        matches = os.path.samestat(identity, os.stat(output_dir, follow_symlinks=False))
    except OSError:
        matches = False
    if not matches:
        raise RuntimeError("training output reservation identity changed")


def _cleanup_empty_reservation(output_dir: Path, identity: os.stat_result) -> None:
    try:
        _require_output_identity(output_dir, identity)
        output_dir.rmdir()
    except (OSError, RuntimeError):
        # Preserve any path that changed identity or gained trainer/audit files.
        return


def run_rl_launch(
    bundle_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    model: str,
    adapter: str,
    expected_manifest_sha256: str,
    plugin: Path | None = None,
    device: str = "0",
    version_reader: Callable[[str], str] = importlib.metadata.version,
    executable_finder: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    on_execute: Callable[[dict[str, object]], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    bundle = _verify_bundle(bundle_dir, expected_manifest_sha256)
    template = load_rl_config(config_path)
    if template["rlhf_type"] == "grpo" and plugin is None:
        raise ValueError("GRPO requires an external reward plugin")
    plugin_payload = _read_plugin(plugin)[1] if plugin is not None and template["rlhf_type"] == "grpo" else None
    executable = validate_ms_swift_runtime(version_reader, executable_finder)
    with tempfile.TemporaryDirectory(prefix="risk-agent-ms-swift-rl-") as temporary:
        root = Path(temporary)
        train, dev = root / "train.jsonl", root / "dev.jsonl"
        _write_private_file(train, bundle.split_bytes["train.jsonl"])
        _write_private_file(dev, bundle.split_bytes["dev.jsonl"])
        plugin_snapshot = None
        if plugin_payload is not None:
            plugin_snapshot = root / "risk_rewards.py"
            _write_private_file(plugin_snapshot, plugin_payload)
        plan = _render(bundle, template, output_dir, model=model, adapter=adapter,
                       plugin_path=plugin_snapshot, train=train, dev=dev, device=device)
        config_snapshot = root / "rlhf.yaml"
        _write_private_file(config_snapshot, yaml.safe_dump(plan.config, allow_unicode=True,
                                                            sort_keys=False).encode("utf-8"))
        argv = [executable, "rlhf", str(config_snapshot)]
        execution = {"argv": argv, "env": plan.env, "rendered_config": plan.config}
        identity = _reserve_output(Path(plan.config["output_dir"]))
        try:
            if on_execute:
                on_execute(execution)
            _require_output_identity(Path(plan.config["output_dir"]), identity)
            environment = os.environ.copy()
            environment.update(plan.env)
            return runner(argv, env=environment, text=True, check=True, shell=False)
        except BaseException:
            _cleanup_empty_reservation(Path(plan.config["output_dir"]), identity)
            raise
