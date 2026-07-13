"""Convert risk tasks and oracles into ms-swift SFT JSONL files."""

from __future__ import annotations

import hashlib
import json
import math
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


def _read_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read UTF-8 {name}: {path}") from error
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
    return records


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


def _load_oracles(path: Path) -> dict[tuple[str, str], Oracle]:
    records = _read_jsonl(path, "oracle input")
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
    return result


def _load_stores(cases_path: Path, evidence_path: Path) -> tuple[CaseStore, EvidenceStore]:
    case_records = _read_jsonl(cases_path, "case input")
    cases: list[dict[str, str]] = []
    for number, record in enumerate(case_records, start=1):
        if set(record) != {"case_id", "text"}:
            raise ValueError(f"case input line {number} has unexpected fields")
        if not all(isinstance(record[key], str) for key in record):
            raise ValueError(f"case input line {number} is invalid")
        cases.append(record)  # type: ignore[arg-type]

    evidence_records = _read_jsonl(evidence_path, "evidence input")
    evidence: list[Evidence] = []
    for number, record in enumerate(evidence_records, start=1):
        if set(record) != {"evidence_id", "asset_id", "kind", "content"}:
            raise ValueError(f"evidence input line {number} has unexpected fields")
        try:
            evidence.append(Evidence.model_validate(record))
        except ValidationError as error:
            raise ValueError(f"evidence input line {number} is invalid") from error
    try:
        return CaseStore(cases), EvidenceStore(evidence)
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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _build_rows(
    track: Track,
    input_path: Path,
    oracle_path: Path,
    cases_path: Path | None,
    evidence_path: Path | None,
) -> list[tuple[str, str, dict[str, object]]]:
    oracles = _load_oracles(oracle_path)
    input_records = _read_jsonl(
        input_path, "task input" if track == "track_a" else "trajectory input"
    )
    case_store: CaseStore | None = None
    evidence_store: EvidenceStore | None = None
    if track == "track_b":
        if cases_path is None or evidence_path is None:
            raise ValueError("track_b requires cases_path and evidence_path")
        case_store, evidence_store = _load_stores(cases_path, evidence_path)
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
    return rows


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
    """Validate inputs and write train, dev and holdout JSONL files."""

    output_dir = output_dir.absolute()
    if track not in ("track_a", "track_b"):
        raise ValueError("track must be track_a or track_b")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    _validate_ratios(ratios)
    if output_dir.exists():
        raise ValueError("output directory must not exist")

    rows = _build_rows(track, input_path, oracle_path, cases_path, evidence_path)
    split_rows: dict[str, list[dict[str, object]]] = {name: [] for name in _SPLITS}
    split_assets: dict[str, set[str]] = {name: set() for name in _SPLITS}
    for asset_id, _policy_version, row in rows:
        split = _split_for(asset_id, seed, ratios)
        split_rows[split].append(row)
        split_assets[split].add(asset_id)

    manifest: dict[str, object] = {
        "track": track,
        "seed": seed,
        "ratios": {
            "train": ratios.train,
            "dev": ratios.dev,
            "holdout": ratios.holdout,
        },
        "split_counts": {name: len(split_rows[name]) for name in _SPLITS},
        "asset_group_counts": {name: len(split_assets[name]) for name in _SPLITS},
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ValueError("output directory must not exist") from error
    for split in _SPLITS:
        _write_jsonl(output_dir / f"{split}.jsonl", split_rows[split])
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest
