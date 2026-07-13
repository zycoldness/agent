"""Export validated policy-conditioned SFT conversations as JSONL."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from risk_agent.contracts import Oracle, Task
from risk_agent.sft_export import export_track_a, export_trajectory


def _read_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read {name}: {path}") from error
    if not lines:
        raise ValueError(f"{name} must contain at least one record")

    records: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{name} line {number} must not be blank")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} line {number} is not valid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"{name} line {number} must be a JSON object")
        records.append(record)
    return records


def _ensure_distinct_output_path(output_path: Path, *input_paths: Path) -> None:
    """Reject aliases that would overwrite either validated source file."""

    try:
        output_resolved = output_path.resolve(strict=False)
    except OSError as error:
        raise ValueError(f"cannot resolve output path: {output_path}") from error
    for input_path in input_paths:
        try:
            same_resolved_path = output_resolved == input_path.resolve(strict=False)
            same_existing_file = (
                output_path.exists() and input_path.exists() and output_path.samefile(input_path)
            )
        except OSError as error:
            raise ValueError(f"cannot compare output path with input path: {input_path}") from error
        if same_resolved_path or same_existing_file:
            raise ValueError("output path must not be the same file as an input or oracle path")


def _load_oracles(path: Path) -> dict[tuple[str, str], Oracle]:
    oracles: dict[tuple[str, str], Oracle] = {}
    for number, record in enumerate(_read_jsonl(path, "oracle input"), start=1):
        try:
            oracle = Oracle.model_validate(record)
        except ValidationError as error:
            raise ValueError(f"oracle input line {number} is invalid") from error
        key = (oracle.asset_id, oracle.policy_version)
        if key in oracles:
            raise ValueError(f"duplicate oracle record for {key[0]!r}, {key[1]!r}")
        oracles[key] = oracle
    return oracles


def _task_from_record(record: dict[str, Any], number: int, mode: str) -> tuple[Task, list[tuple[str, str]]]:
    if mode == "track_a":
        try:
            return Task.model_validate(record), []
        except ValidationError as error:
            raise ValueError(f"task input line {number} is invalid") from error

    if set(record) != {"task", "steps"}:
        raise ValueError(f"trajectory input line {number} must contain exactly task and steps")
    try:
        task = Task.model_validate(record["task"])
    except ValidationError as error:
        raise ValueError(f"trajectory input line {number} has an invalid task") from error
    raw_steps = record["steps"]
    if not isinstance(raw_steps, list):
        raise ValueError(f"trajectory input line {number} steps must be a list")
    steps: list[tuple[str, str]] = []
    for index, step in enumerate(raw_steps, start=1):
        if not isinstance(step, dict) or set(step) != {"action", "observation"}:
            raise ValueError(f"trajectory input line {number} step {index} must contain action and observation")
        steps.append((step["action"], step["observation"]))
    return task, steps


def build_rows(mode: str, input_path: Path, oracle_path: Path) -> list[dict[str, object]]:
    """Fully validate inputs and construct rows before touching the output path."""

    oracles = _load_oracles(oracle_path)
    tasks: set[tuple[str, str]] = set()
    rows: list[dict[str, object]] = []
    input_name = "task input" if mode == "track_a" else "trajectory input"
    for number, record in enumerate(_read_jsonl(input_path, input_name), start=1):
        task, steps = _task_from_record(record, number, mode)
        key = (task.asset_id, task.policy_version)
        if key in tasks:
            raise ValueError(f"duplicate task record for {key[0]!r}, {key[1]!r}")
        tasks.add(key)
        oracle = oracles.get(key)
        if oracle is None:
            raise ValueError(f"missing oracle for {key[0]!r}, {key[1]!r}")
        row = export_track_a(task, oracle) if mode == "track_a" else export_trajectory(task, oracle, steps)
        rows.append(row)

    unmatched = sorted(set(oracles) - tasks)
    if unmatched:
        asset_id, policy_version = unmatched[0]
        raise ValueError(f"unmatched oracle record for {asset_id!r}, {policy_version!r}")
    return rows


def write_jsonl_atomically(rows: list[dict[str, object]], output_path: Path) -> None:
    """Replace the output only after every complete row has been serialized."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
                handle.write("\n")
        os.replace(temporary_name, output_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("track_a", "track_b"))
    parser.add_argument("input_path", type=Path)
    parser.add_argument("oracle_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args(argv)
    try:
        _ensure_distinct_output_path(args.output_path, args.input_path, args.oracle_path)
        rows = build_rows(args.mode, args.input_path, args.oracle_path)
        write_jsonl_atomically(rows, args.output_path)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
