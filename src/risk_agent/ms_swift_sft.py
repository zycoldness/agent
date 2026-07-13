"""Prepare deterministic, leak-free messages JSONL bundles for ms-swift SFT."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from risk_agent.contracts import Evidence, Oracle, Task
from risk_agent.sft_export import export_track_a, export_trajectory
from risk_agent.stores import CaseStore, EvidenceStore


Track = Literal["track_a", "track_b"]
_SPLITS = ("train", "dev", "holdout")
_TASK_REQUIRED = frozenset({"asset_id", "policy_version", "active_policy", "initial_observation"})
_TASK_OPTIONAL = frozenset({"max_turns", "images", "videos"})
_RULE_REQUIRED = frozenset({"rule_id", "title", "text"})
_RULE_OPTIONAL = frozenset({"exceptions", "priority"})
_ORACLE_REQUIRED = frozenset({"asset_id", "policy_version", "label"})
_ORACLE_OPTIONAL = frozenset({"rule_id", "evidence_ids", "risk_level", "next_action"})


@dataclass(frozen=True)
class SplitRatios:
    """Requested group-level split probabilities."""

    train: float = 0.8
    dev: float = 0.1
    holdout: float = 0.1


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _read_jsonl(path: Path, name: str) -> tuple[list[dict[str, Any]], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {name}: {path}") from error
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8") from error
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"{name} must contain at least one record")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{name} line {number} must not be blank")
        try:
            record = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_nonfinite_constant,
            )
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} line {number} is not valid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"{name} line {number} must be a JSON object")
        records.append(record)
    return records, payload


def _require_fields(
    record: dict[str, Any], required: frozenset[str], optional: frozenset[str], context: str
) -> None:
    missing = required - set(record)
    extra = set(record) - required - optional
    if missing:
        raise ValueError(f"{context} missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"{context} has unexpected fields: {', '.join(sorted(extra))}")


def _validate_task_shape(record: dict[str, Any], context: str) -> None:
    _require_fields(record, _TASK_REQUIRED, _TASK_OPTIONAL, context)
    rules = record.get("active_policy")
    if not isinstance(rules, list):
        raise ValueError(f"{context} active_policy must be a list")
    for index, rule in enumerate(rules, start=1):
        if not isinstance(rule, dict):
            raise ValueError(f"{context} rule {index} must be an object")
        _require_fields(rule, _RULE_REQUIRED, _RULE_OPTIONAL, f"{context} rule {index}")


def _parse_task(record: dict[str, Any], context: str) -> Task:
    _validate_task_shape(record, context)
    try:
        return Task.model_validate(record)
    except ValidationError as error:
        raise ValueError(f"{context} is invalid") from error


def _load_oracles(path: Path) -> tuple[dict[tuple[str, str], Oracle], bytes]:
    records, payload = _read_jsonl(path, "oracle input")
    result: dict[tuple[str, str], Oracle] = {}
    for number, record in enumerate(records, start=1):
        _require_fields(record, _ORACLE_REQUIRED, _ORACLE_OPTIONAL, f"oracle input line {number}")
        try:
            oracle = Oracle.model_validate(record)
        except ValidationError as error:
            raise ValueError(f"oracle input line {number} is invalid") from error
        key = (oracle.asset_id, oracle.policy_version)
        if key in result:
            raise ValueError(f"duplicate oracle record for {key[0]!r}, {key[1]!r}")
        result[key] = oracle
    return result, payload


def _load_stores(cases_path: Path, evidence_path: Path) -> tuple[CaseStore, EvidenceStore, bytes, bytes]:
    case_records, case_payload = _read_jsonl(cases_path, "case input")
    cases: list[dict[str, str]] = []
    for number, record in enumerate(case_records, start=1):
        if set(record) != {"case_id", "text"}:
            raise ValueError(f"case input line {number} has unexpected fields")
        if not all(isinstance(record[key], str) for key in record):
            raise ValueError(f"case input line {number} is invalid")
        cases.append(record)  # type: ignore[arg-type]

    evidence_records, evidence_payload = _read_jsonl(evidence_path, "evidence input")
    evidence: list[Evidence] = []
    for number, record in enumerate(evidence_records, start=1):
        if set(record) != {"evidence_id", "asset_id", "kind", "content"}:
            raise ValueError(f"evidence input line {number} has unexpected fields")
        try:
            evidence.append(Evidence.model_validate(record))
        except ValidationError as error:
            raise ValueError(f"evidence input line {number} is invalid") from error
    try:
        return CaseStore(cases), EvidenceStore(evidence), case_payload, evidence_payload
    except ValueError as error:
        raise ValueError("case or evidence input is invalid") from error


def _validate_ratios(ratios: SplitRatios) -> None:
    values = (ratios.train, ratios.dev, ratios.holdout)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError("split ratios must be finite numbers")
    try:
        invalid = any(not math.isfinite(float(value)) or value < 0 or value > 1 for value in values)
    except (OverflowError, ValueError):
        invalid = True
    if invalid:
        raise ValueError("split ratios must be finite numbers from zero to one")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("split ratios must sum to one")


def _split_for(asset_id: str, seed: int, ratios: SplitRatios) -> str:
    digest = hashlib.sha256(f"{seed}\0{asset_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest, "big") / (1 << 256)
    if value < ratios.train:
        return "train"
    if value < ratios.train + ratios.dev:
        return "dev"
    return "holdout"


def _source_fingerprint(payload: bytes) -> dict[str, object]:
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _ensure_distinct_sources(paths: list[Path]) -> None:
    identities: set[Path] = set()
    for path in paths:
        try:
            identity = path.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"cannot resolve source file: {path}") from error
        if identity in identities:
            raise ValueError("all inputs must be distinct source files")
        identities.add(identity)


def _build_rows(
    track: Track,
    input_path: Path,
    oracle_path: Path,
    cases_path: Path | None,
    evidence_path: Path | None,
) -> tuple[list[tuple[str, str, dict[str, object]]], dict[str, dict[str, object]]]:
    oracles, oracle_payload = _load_oracles(oracle_path)
    input_records, input_payload = _read_jsonl(
        input_path, "task input" if track == "track_a" else "trajectory input"
    )
    sources = {
        "input": _source_fingerprint(input_payload),
        "oracle": _source_fingerprint(oracle_payload),
    }
    case_store: CaseStore | None = None
    evidence_store: EvidenceStore | None = None
    if track == "track_b":
        if cases_path is None or evidence_path is None:
            raise ValueError("track_b requires cases_path and evidence_path")
        case_store, evidence_store, case_payload, evidence_payload = _load_stores(
            cases_path, evidence_path
        )
        sources["cases"] = _source_fingerprint(case_payload)
        sources["evidence"] = _source_fingerprint(evidence_payload)
    elif cases_path is not None or evidence_path is not None:
        raise ValueError("cases_path and evidence_path are only valid for track_b")

    task_keys: set[tuple[str, str]] = set()
    rows: list[tuple[str, str, dict[str, object]]] = []
    for number, record in enumerate(input_records, start=1):
        steps: list[tuple[str, str]] = []
        if track == "track_a":
            task = _parse_task(record, f"task input line {number}")
        else:
            if set(record) != {"task", "steps"}:
                raise ValueError(
                    f"trajectory input line {number} has unexpected fields or missing fields"
                )
            if not isinstance(record["task"], dict):
                raise ValueError(f"trajectory input line {number} task must be an object")
            task = _parse_task(record["task"], f"trajectory input line {number} task")
            raw_steps = record["steps"]
            if not isinstance(raw_steps, list):
                raise ValueError(f"trajectory input line {number} steps must be a list")
            for index, step in enumerate(raw_steps, start=1):
                if not isinstance(step, dict) or set(step) != {"action", "observation"}:
                    raise ValueError(
                        f"trajectory input line {number} step {index} has unexpected fields"
                    )
                if not isinstance(step["action"], str) or not isinstance(step["observation"], str):
                    raise ValueError(f"trajectory input line {number} step {index} is invalid")
                steps.append((step["action"], step["observation"]))
        key = (task.asset_id, task.policy_version)
        if key in task_keys:
            raise ValueError(f"duplicate task record for {key[0]!r}, {key[1]!r}")
        task_keys.add(key)
        oracle = oracles.get(key)
        if oracle is None:
            raise ValueError(f"missing oracle for {key[0]!r}, {key[1]!r}")
        if track == "track_a":
            row = export_track_a(task, oracle)
        else:
            assert case_store is not None and evidence_store is not None
            row = export_trajectory(task, oracle, steps, case_store, evidence_store)
        rows.append((task.asset_id, task.policy_version, row))

    unmatched = set(oracles) - task_keys
    if unmatched:
        first = sorted(unmatched)[0]
        raise ValueError(f"unmatched oracle record for {first[0]!r}, {first[1]!r}")
    return rows, sources


def prepare_sft_bundle(
    track: Track,
    input_path: Path,
    oracle_path: Path,
    output_dir: Path,
    *,
    cases_path: Path | None = None,
    evidence_path: Path | None = None,
    ratios: SplitRatios = SplitRatios(),
    seed: int = 42,
) -> dict[str, object]:
    """Validate all sources and exclusively publish a split messages bundle."""

    output_dir = output_dir.absolute()
    if track not in ("track_a", "track_b"):
        raise ValueError("track must be track_a or track_b")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    _validate_ratios(ratios)
    source_paths = [input_path, oracle_path]
    if cases_path is not None:
        source_paths.append(cases_path)
    if evidence_path is not None:
        source_paths.append(evidence_path)
    _ensure_distinct_sources(source_paths)
    if output_dir.exists():
        raise ValueError("output directory must not exist")
    try:
        output_identity = output_dir.resolve(strict=False)
    except OSError as error:
        raise ValueError(f"cannot resolve output directory: {output_dir}") from error
    if output_identity in {path.resolve(strict=True) for path in source_paths}:
        raise ValueError("output path must not alias a source file")

    rows, sources = _build_rows(track, input_path, oracle_path, cases_path, evidence_path)
    split_rows: dict[str, list[dict[str, object]]] = {name: [] for name in _SPLITS}
    split_assets: dict[str, set[str]] = {name: set() for name in _SPLITS}
    for asset_id, _policy_version, row in rows:
        split = _split_for(asset_id, seed, ratios)
        split_rows[split].append(row)
        split_assets[split].add(asset_id)

    payloads = {f"{split}.jsonl": _jsonl_bytes(split_rows[split]) for split in _SPLITS}
    files = {
        name: {
            "bytes": len(payload),
            "records": len(split_rows[name.removesuffix(".jsonl")]),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in payloads.items()
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "format": "ms-swift-messages-jsonl",
        "status": "complete",
        "track": track,
        "seed": seed,
        "ratios": {
            "train": ratios.train,
            "dev": ratios.dev,
            "holdout": ratios.holdout,
        },
        "sources": sources,
        "split_counts": {name: len(split_rows[name]) for name in _SPLITS},
        "asset_group_counts": {name: len(split_assets[name]) for name in _SPLITS},
        "files": files,
    }
    manifest_payload = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ValueError("output directory must not exist") from error
    created_identity = os.stat(output_dir, follow_symlinks=False)
    if os.name == "posix":
        output_dir.chmod(0o700)
    _publish_reserved_bundle(
        output_dir,
        created_identity,
        payloads,
        manifest_payload,
    )
    return manifest


def _publish_reserved_bundle(
    output_dir: Path,
    created_identity: os.stat_result,
    payloads: dict[str, bytes],
    manifest_payload: bytes,
) -> None:
    """Write split members and then the manifest without trusting the path again."""

    ordered_members = (*payloads.items(), ("manifest.json", manifest_payload))
    if os.name == "posix":
        directory_flags = os.O_RDONLY
        directory_flags |= getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        output_fd = os.open(output_dir, directory_flags)
        try:
            held_identity = os.fstat(output_fd)
            if not os.path.samestat(created_identity, held_identity):
                raise RuntimeError("SFT bundle reservation path identity changed")
            for filename, payload in ordered_members:
                _require_reservation_identity(output_dir, held_identity)
                _write_reserved_member(output_dir, output_fd, filename, payload)
                _require_reservation_identity(output_dir, held_identity)
        finally:
            os.close(output_fd)
        return

    for filename, payload in ordered_members:
        # Python does not expose Windows directory-relative create.  Recheck
        # twice before opening so a replacement triggered by the first probe
        # is detected before any member is written, and check again after.
        _require_reservation_identity(output_dir, created_identity)
        _require_reservation_identity(output_dir, created_identity)
        _write_reserved_member(output_dir, None, filename, payload)
        _require_reservation_identity(output_dir, created_identity)


def _require_reservation_identity(output_dir: Path, identity: os.stat_result) -> None:
    try:
        matches = os.path.samestat(identity, os.stat(output_dir, follow_symlinks=False))
    except OSError:
        matches = False
    if not matches:
        raise RuntimeError("SFT bundle reservation path identity changed")


def _write_reserved_member(
    output_dir: Path,
    output_fd: int | None,
    filename: str,
    payload: bytes,
) -> None:
    """Exclusively create a member relative to the held directory on POSIX."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    if output_fd is None:
        descriptor = os.open(output_dir / filename, flags, 0o600)
    else:
        descriptor = os.open(filename, flags, 0o600, dir_fd=output_fd)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
