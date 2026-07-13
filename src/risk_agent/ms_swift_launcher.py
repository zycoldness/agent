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
from typing import Any, Callable, Literal

import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    ValidationError,
    field_validator,
)


_SUPPORTED_MS_SWIFT = SpecifierSet(">=4.3,<4.4")
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


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_loads(text: str, context: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} is not valid JSON") from error


class _SFTTemplate(BaseModel):
    """Exact schema for the only supported ms-swift SFT template."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    model: str
    tuner_type: Literal["lora"]
    torch_dtype: Literal["bfloat16"]
    dataset: Literal["__TRAIN_DATASET__"]
    val_dataset: Literal["__VAL_DATASET__"]
    output_dir: Literal["__OUTPUT_DIR__"]
    max_length: StrictInt = Field(gt=0, le=32768)
    num_train_epochs: StrictInt | StrictFloat = Field(gt=0, le=100)
    per_device_train_batch_size: StrictInt = Field(gt=0, le=1024)
    per_device_eval_batch_size: StrictInt = Field(gt=0, le=1024)
    gradient_accumulation_steps: StrictInt = Field(gt=0, le=65536)
    gradient_checkpointing: StrictBool
    learning_rate: StrictFloat = Field(gt=0, le=1, allow_inf_nan=False)
    lora_rank: StrictInt = Field(gt=0, le=4096)
    lora_alpha: StrictInt = Field(gt=0, le=65536)
    target_modules: Literal["all-linear"]
    logging_steps: StrictInt = Field(gt=0)
    save_steps: StrictInt = Field(gt=0)
    eval_steps: StrictInt = Field(gt=0)
    save_total_limit: StrictInt = Field(gt=0)
    warmup_ratio: StrictFloat = Field(ge=0, lt=1, allow_inf_nan=False)
    dataloader_num_workers: StrictInt = Field(ge=0, le=1024)
    dataset_num_proc: StrictInt = Field(ge=0, le=1024)
    strict: Literal[True]
    report_to: Literal["none"]

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if (
            not value.strip()
            or value != value.strip()
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("model must be a non-empty trimmed string")
        return value

    @field_validator("num_train_epochs")
    @classmethod
    def validate_epochs_are_finite(cls, value: int | float) -> int | float:
        if not math.isfinite(float(value)):
            raise ValueError("num_train_epochs must be finite")
        return value


@dataclass(frozen=True)
class LaunchPlan:
    """Fully rendered dry-run plan; it is not a credential for a real run."""

    argv: tuple[str, ...]
    env: dict[str, str]
    config: dict[str, Any]

    def as_dict(self) -> dict[str, object]:
        return {
            "argv": list(self.argv),
            "env": self.env,
            "rendered_config": self.config,
        }


@dataclass(frozen=True)
class _VerifiedBundle:
    root: Path
    manifest: dict[str, Any]
    split_bytes: dict[str, bytes]


def load_sft_config(path: Path) -> dict[str, Any]:
    """Load one exact, strongly typed ms-swift 4.3 SFT config template."""

    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read SFT config: {path}") from error
    if not isinstance(document, dict):
        raise ValueError("SFT config must be a mapping")
    allowed = set(_SFTTemplate.model_fields)
    unknown = set(document) - allowed
    missing = allowed - set(document)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"missing config keys: {', '.join(sorted(missing))}")
    try:
        validated = _SFTTemplate.model_validate(document)
    except ValidationError as error:
        raise ValueError("invalid SFT config values or types") from error
    return validated.model_dump(mode="python")


def _validate_messages_row(row: Any, file_name: str) -> None:
    if not isinstance(row, dict) or set(row) != {"messages"}:
        raise ValueError(f"bundle messages row schema is invalid: {file_name}")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) < 3 or len(messages) % 2 == 0:
        raise ValueError(f"bundle messages must be a non-empty odd-length conversation: {file_name}")
    expected_roles = ["system"] + [
        "user" if index % 2 else "assistant" for index in range(1, len(messages))
    ]
    for index, (message, expected_role) in enumerate(zip(messages, expected_roles, strict=True)):
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message.get("role") != expected_role
            or not isinstance(message.get("content"), str)
            or not message["content"]
        ):
            raise ValueError(
                f"bundle messages item or role order is invalid at {file_name}:{index + 1}"
            )


def _read_and_validate_split(
    bundle_dir: Path,
    file_name: str,
    entry: dict[str, Any],
    expected_records: int,
) -> bytes:
    path = bundle_dir / file_name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"bundle file is missing or unsafe: {file_name}")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read bundle file: {file_name}") from error
    if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise ValueError(f"bundle file hash or size mismatch: {file_name}")
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"bundle file must be UTF-8: {file_name}") from error
    if len(lines) != entry["records"] or entry["records"] != expected_records:
        raise ValueError(f"bundle file record count mismatch: {file_name}")
    for line in lines:
        if not line.strip():
            raise ValueError(f"bundle messages row must not be blank: {file_name}")
        row = _strict_json_loads(line, f"bundle messages row in {file_name}")
        _validate_messages_row(row, file_name)
    return payload


def _verify_bundle(bundle_dir: Path) -> _VerifiedBundle:
    try:
        root = bundle_dir.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"cannot resolve bundle directory: {bundle_dir}") from error
    if not root.is_dir():
        raise ValueError("bundle path must be a directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("bundle manifest must not be a symbolic link")
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("bundle manifest is missing or invalid") from error
    manifest = _strict_json_loads(manifest_text, "bundle manifest")
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
    split_counts = manifest["split_counts"]
    group_counts = manifest["asset_group_counts"]
    if not isinstance(files, dict) or set(files) != set(_SPLIT_FILES):
        raise ValueError("bundle manifest files are invalid")
    for counts, message in (
        (split_counts, "bundle split counts are invalid"),
        (group_counts, "bundle asset group counts are invalid"),
    ):
        if (
            not isinstance(counts, dict)
            or set(counts) != {"train", "dev", "holdout"}
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counts.values()
            )
        ):
            raise ValueError(message)

    split_bytes: dict[str, bytes] = {}
    for file_name in _SPLIT_FILES:
        entry = files[file_name]
        if (
            not isinstance(entry, dict)
            or set(entry) != {"bytes", "records", "sha256"}
            or isinstance(entry["bytes"], bool)
            or not isinstance(entry["bytes"], int)
            or entry["bytes"] < 0
            or isinstance(entry["records"], bool)
            or not isinstance(entry["records"], int)
            or entry["records"] < 0
            or not isinstance(entry["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
        ):
            raise ValueError("bundle file metadata is invalid")
        split = file_name.removesuffix(".jsonl")
        split_bytes[file_name] = _read_and_validate_split(
            root, file_name, entry, split_counts[split]
        )
    if split_counts["train"] == 0:
        raise ValueError("train split must not be empty")
    return _VerifiedBundle(root=root, manifest=manifest, split_bytes=split_bytes)


def _validate_override(value: str, name: str) -> None:
    if not value or value != value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} override is invalid")


def _render_plan(
    verified: _VerifiedBundle,
    template: dict[str, Any],
    output_dir: Path,
    *,
    model: str | None,
    device: str,
    train_path: Path,
    dev_path: Path,
) -> LaunchPlan:
    if re.fullmatch(r"[0-9]+", device) is None:
        raise ValueError("device must be one non-negative CUDA device index")
    try:
        output = output_dir.resolve(strict=False)
    except OSError as error:
        raise ValueError("cannot resolve output directory") from error
    if output == verified.root or verified.root in output.parents:
        raise ValueError("training output directory must be outside the input bundle")
    rendered = dict(template)
    rendered["dataset"] = [str(train_path)]
    if verified.manifest["split_counts"]["dev"]:
        rendered["val_dataset"] = [str(dev_path)]
        rendered["eval_strategy"] = "steps"
    else:
        rendered["val_dataset"] = []
        rendered["eval_strategy"] = "no"
    rendered["output_dir"] = str(output)
    if model is not None:
        _validate_override(model, "model")
        rendered["model"] = model
    return LaunchPlan(
        argv=("swift", "sft", "<generated-config>"),
        env={"CUDA_VISIBLE_DEVICES": device, "NPROC_PER_NODE": "1"},
        config=rendered,
    )


def prepare_sft_launch(
    bundle_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    model: str | None = None,
    device: str = "0",
) -> LaunchPlan:
    """Verify current bundle bytes and render a non-executable dry-run plan."""

    verified = _verify_bundle(bundle_dir)
    template = load_sft_config(config_path)
    return _render_plan(
        verified,
        template,
        output_dir,
        model=model,
        device=device,
        train_path=verified.root / "train.jsonl",
        dev_path=verified.root / "dev.jsonl",
    )


def validate_ms_swift_runtime(
    version_reader: Callable[[str], str] = importlib.metadata.version,
    executable_finder: Callable[[str], str | None] = shutil.which,
) -> str:
    """Require a stable public ms-swift release in the pinned 4.3 minor."""

    try:
        raw_version = version_reader("ms-swift")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("ms-swift>=4.3,<4.4 is not installed") from error
    try:
        version = Version(raw_version)
    except InvalidVersion as error:
        raise RuntimeError(f"ms-swift>=4.3,<4.4 is required; found {raw_version!r}") from error
    if (
        version not in _SUPPORTED_MS_SWIFT
        or version.is_prerelease
        or version.is_devrelease
        or version.local is not None
    ):
        raise RuntimeError(f"ms-swift>=4.3,<4.4 is required; found {raw_version!r}")
    executable = executable_finder("swift")
    if not executable:
        raise RuntimeError("swift executable was not found on PATH")
    return executable


def _write_private_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def run_sft_launch(
    bundle_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    model: str | None = None,
    device: str = "0",
    version_reader: Callable[[str], str] = importlib.metadata.version,
    executable_finder: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    on_execute: Callable[[dict[str, object]], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Reverify bytes, snapshot them privately, and run only from the snapshot."""

    verified = _verify_bundle(bundle_dir)
    template = load_sft_config(config_path)
    executable = validate_ms_swift_runtime(version_reader, executable_finder)
    with tempfile.TemporaryDirectory(prefix="risk-agent-ms-swift-") as temporary:
        snapshot_dir = Path(temporary)
        train_snapshot = snapshot_dir / "train.jsonl"
        dev_snapshot = snapshot_dir / "dev.jsonl"
        _write_private_file(train_snapshot, verified.split_bytes["train.jsonl"])
        _write_private_file(dev_snapshot, verified.split_bytes["dev.jsonl"])
        plan = _render_plan(
            verified,
            template,
            output_dir,
            model=model,
            device=device,
            train_path=train_snapshot,
            dev_path=dev_snapshot,
        )
        rendered_path = snapshot_dir / "sft.yaml"
        rendered_payload = yaml.safe_dump(
            plan.config, allow_unicode=True, sort_keys=False
        ).encode("utf-8")
        _write_private_file(rendered_path, rendered_payload)
        argv = [executable, "sft", str(rendered_path)]
        environment = os.environ.copy()
        environment.update(plan.env)
        execution = {
            "argv": argv,
            "env": plan.env,
            "rendered_config": plan.config,
        }
        if on_execute is not None:
            on_execute(execution)
        return runner(
            argv,
            env=environment,
            text=True,
            check=True,
            shell=False,
        )
