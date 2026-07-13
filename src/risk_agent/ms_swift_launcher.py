"""Validate and launch a single-device ms-swift 4.3 SFT run without a shell."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml


_CONFIG_REQUIRED = frozenset(
    {
        "model",
        "tuner_type",
        "torch_dtype",
        "dataset",
        "val_dataset",
        "output_dir",
        "max_length",
        "num_train_epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "learning_rate",
        "lora_rank",
        "lora_alpha",
        "target_modules",
        "logging_steps",
        "save_steps",
        "eval_steps",
        "save_total_limit",
        "warmup_ratio",
        "dataloader_num_workers",
        "dataset_num_proc",
        "strict",
        "report_to",
    }
)
_CONFIG_RUNTIME_ONLY = frozenset({"eval_strategy"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "format",
        "status",
        "track",
        "seed",
        "ratios",
        "sources",
        "split_counts",
        "asset_group_counts",
        "files",
    }
)
_SPLIT_FILES = ("train.jsonl", "dev.jsonl", "holdout.jsonl")


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("config keys must be strings")
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class LaunchPlan:
    """Fully rendered dry-run plan; no runtime dependency has been imported."""

    argv: tuple[str, ...]
    env: dict[str, str]
    config: dict[str, Any]

    def as_dict(self) -> dict[str, object]:
        return {
            "argv": list(self.argv),
            "env": self.env,
            "rendered_config": self.config,
        }


def load_sft_config(path: Path) -> dict[str, Any]:
    """Load one exact, auditable ms-swift 4.3 SFT config template."""

    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read SFT config: {path}") from error
    if not isinstance(document, dict):
        raise ValueError("SFT config must be a mapping")
    unknown = set(document) - _CONFIG_REQUIRED
    missing = _CONFIG_REQUIRED - set(document)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"missing config keys: {', '.join(sorted(missing))}")
    if document["tuner_type"] != "lora":
        raise ValueError("tuner_type must be lora")
    if document["torch_dtype"] != "bfloat16":
        raise ValueError("torch_dtype must be bfloat16")
    if document["strict"] is not True:
        raise ValueError("strict must be true")
    placeholders = {
        "dataset": "__TRAIN_DATASET__",
        "val_dataset": "__VAL_DATASET__",
        "output_dir": "__OUTPUT_DIR__",
    }
    for key, expected in placeholders.items():
        if document[key] != expected:
            raise ValueError(f"{key} must use the explicit {expected} placeholder")
    forbidden = {"deepspeed", "vllm", "quant_bits", "train_type"}
    if forbidden & set(document):
        raise ValueError("unknown config keys include an unsupported training option")
    return document


def _load_manifest(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("bundle manifest must not be a symbolic link")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("bundle manifest is missing or invalid") from error
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("bundle manifest has an invalid schema")
    if manifest["schema_version"] != 1 or manifest["format"] != "ms-swift-messages-jsonl":
        raise ValueError("bundle manifest has an unsupported schema version or format")
    if manifest["status"] != "complete":
        raise ValueError("bundle manifest status must be complete")
    if manifest["track"] not in {"track_a", "track_b"}:
        raise ValueError("bundle manifest track is invalid")
    if isinstance(manifest["seed"], bool) or not isinstance(manifest["seed"], int):
        raise ValueError("bundle manifest seed is invalid")
    ratios = manifest["ratios"]
    if not isinstance(ratios, dict) or set(ratios) != {"train", "dev", "holdout"}:
        raise ValueError("bundle manifest ratios are invalid")
    ratio_values = tuple(ratios[name] for name in ("train", "dev", "holdout"))
    if (
        any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in ratio_values)
        or any(not math.isfinite(float(value)) or value < 0 or value > 1 for value in ratio_values)
        or not math.isclose(sum(ratio_values), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("bundle manifest ratios are invalid")
    sources = manifest["sources"]
    expected_sources = {"input", "oracle"}
    if manifest["track"] == "track_b":
        expected_sources.update({"cases", "evidence"})
    if not isinstance(sources, dict) or set(sources) != expected_sources:
        raise ValueError("bundle source metadata is invalid")
    for source in sources.values():
        if (
            not isinstance(source, dict)
            or set(source) != {"bytes", "sha256"}
            or isinstance(source["bytes"], bool)
            or not isinstance(source["bytes"], int)
            or source["bytes"] < 0
            or not isinstance(source["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None
        ):
            raise ValueError("bundle source metadata is invalid")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(_SPLIT_FILES):
        raise ValueError("bundle manifest files are invalid")
    split_counts = manifest["split_counts"]
    if not isinstance(split_counts, dict) or set(split_counts) != {"train", "dev", "holdout"}:
        raise ValueError("bundle split counts are invalid")
    group_counts = manifest["asset_group_counts"]
    if not isinstance(group_counts, dict) or set(group_counts) != {"train", "dev", "holdout"}:
        raise ValueError("bundle asset group counts are invalid")
    for counts, message in (
        (split_counts, "bundle split counts are invalid"),
        (group_counts, "bundle asset group counts are invalid"),
    ):
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts.values()
        ):
            raise ValueError(message)
    for file_name in _SPLIT_FILES:
        entry = files[file_name]
        if not isinstance(entry, dict) or set(entry) != {"bytes", "records", "sha256"}:
            raise ValueError("bundle file metadata is invalid")
        if (
            isinstance(entry["bytes"], bool)
            or not isinstance(entry["bytes"], int)
            or entry["bytes"] < 0
            or isinstance(entry["records"], bool)
            or not isinstance(entry["records"], int)
            or entry["records"] < 0
            or not isinstance(entry["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
        ):
            raise ValueError("bundle file metadata is invalid")
        path = bundle_dir / file_name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"bundle file is missing or unsafe: {file_name}")
        payload = path.read_bytes()
        if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"bundle file hash or size mismatch: {file_name}")
        lines = payload.decode("utf-8").splitlines()
        if len(lines) != entry["records"] or entry["records"] != split_counts[file_name.removesuffix(".jsonl")]:
            raise ValueError(f"bundle file record count mismatch: {file_name}")
        for line in lines:
            row = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
            if not isinstance(row, dict) or set(row) != {"messages"}:
                raise ValueError(f"bundle training row schema is invalid: {file_name}")
    return manifest


def _validate_override(value: str, name: str) -> None:
    if not value or value != value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} override is invalid")


def prepare_sft_launch(
    bundle_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    model: str | None = None,
    device: str = "0",
) -> LaunchPlan:
    """Verify a bundle and render an in-memory dry-run launch plan."""

    if re.fullmatch(r"[0-9]+", device) is None:
        raise ValueError("device must be one non-negative CUDA device index")
    bundle_dir = bundle_dir.resolve(strict=True)
    manifest = _load_manifest(bundle_dir)
    if manifest["split_counts"]["train"] == 0:
        raise ValueError("train split must not be empty")
    template = load_sft_config(config_path)
    try:
        output = output_dir.resolve(strict=False)
    except OSError as error:
        raise ValueError("cannot resolve output directory") from error
    if output == bundle_dir or bundle_dir in output.parents:
        raise ValueError("training output directory must be outside the input bundle")
    rendered = dict(template)
    rendered["dataset"] = [str(bundle_dir / "train.jsonl")]
    dev_count = manifest["split_counts"]["dev"]
    if dev_count:
        rendered["val_dataset"] = [str(bundle_dir / "dev.jsonl")]
        rendered["eval_strategy"] = "steps"
    else:
        rendered["val_dataset"] = []
        rendered["eval_strategy"] = "no"
    rendered["output_dir"] = str(output)
    if model is not None:
        _validate_override(model, "model")
        rendered["model"] = model
    return LaunchPlan(
        argv=("swift", "sft", "--config", "<generated-config>"),
        env={"CUDA_VISIBLE_DEVICES": device, "NPROC_PER_NODE": "1"},
        config=rendered,
    )


def validate_ms_swift_runtime(
    version_reader: Callable[[str], str] = importlib.metadata.version,
    executable_finder: Callable[[str], str | None] = shutil.which,
) -> str:
    """Require the pinned ms-swift 4.3 minor and a CLI executable."""

    try:
        version = version_reader("ms-swift")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("ms-swift>=4.3,<4.4 is not installed") from error
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?(?:[.+-].*)?", version)
    if match is None or (int(match.group(1)), int(match.group(2))) != (4, 3):
        raise RuntimeError(f"ms-swift>=4.3,<4.4 is required; found {version!r}")
    executable = executable_finder("swift")
    if not executable:
        raise RuntimeError("swift executable was not found on PATH")
    return executable


def run_sft_launch(
    plan: LaunchPlan,
    *,
    version_reader: Callable[[str], str] = importlib.metadata.version,
    executable_finder: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    on_execute: Callable[[dict[str, object]], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Materialize the config only for the subprocess lifetime and run without a shell."""

    executable = validate_ms_swift_runtime(version_reader, executable_finder)
    with tempfile.TemporaryDirectory(prefix="risk-agent-ms-swift-") as temporary:
        rendered_path = Path(temporary) / "sft.yaml"
        rendered_path.write_text(
            yaml.safe_dump(plan.config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        argv = [executable, "sft", "--config", str(rendered_path)]
        environment = os.environ.copy()
        environment.update(plan.env)
        if on_execute is not None:
            on_execute(
                {
                    "argv": argv,
                    "env": plan.env,
                    "rendered_config": plan.config,
                }
            )
        return runner(
            argv,
            env=environment,
            text=True,
            check=True,
            shell=False,
        )
