"""Generate audited, bounded two-hop SFT candidates with Gemini."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

try:
    from scripts.export_sft import _load_oracles, _load_stores, _read_jsonl, write_jsonl_atomically
except ModuleNotFoundError:  # Direct execution sets scripts/ as sys.path[0].
    from export_sft import _load_oracles, _load_stores, _read_jsonl, write_jsonl_atomically
from risk_agent.contracts import Task
from risk_agent.synthesis import generate_teacher_sample
from risk_agent.teacher import (
    GeminiTeacher,
    Teacher,
    TeacherBudget,
    TeacherBudgetExceeded,
    TeacherRequestError,
)


@dataclass(frozen=True)
class BatchOutcome:
    complete: bool
    completed_records: int
    reason: str | None = None


def _sidecar_paths(output: Path) -> tuple[Path, Path]:
    return Path(str(output) + ".partial"), Path(str(output) + ".checkpoint.json")


def _reject_input_overwrite(output: Path, inputs: list[Path]) -> None:
    partial, checkpoint = _sidecar_paths(output)
    for candidate in (output, partial, checkpoint):
        candidate_resolved = candidate.resolve(strict=False)
        for input_path in inputs:
            if candidate_resolved == input_path.resolve(strict=False):
                raise ValueError("output path must not overwrite an input file")
            if candidate.exists() and input_path.exists() and candidate.samefile(input_path):
                raise ValueError("output path must not overwrite an input file")
    if output.exists() or partial.exists() or checkpoint.exists():
        raise ValueError("output and checkpoint paths must not already exist")


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _write_json_atomically(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _checkpoint(
    output: Path,
    rows: list[dict[str, object]],
    budget: TeacherBudget,
    *,
    total_records: int,
    classification: str,
    status: str,
    reason: str | None,
) -> None:
    partial, checkpoint = _sidecar_paths(output)
    if status != "complete" and rows:
        write_jsonl_atomically(rows, partial)
    manifest: dict[str, object] = {
        "status": status,
        "reason": reason,
        "completed_records": len(rows),
        "total_records": total_records,
        "data_classification": classification,
        "external_data_scope": ["task", "oracle", "selected_cases", "selected_evidence"],
        "teacher_usage": budget.as_dict(),
        "partial_path": partial.name if partial.exists() else None,
        "partial_sha256": _sha256(partial),
        "final_sha256": _sha256(output),
    }
    _write_json_atomically(manifest, checkpoint)


def _load_validated_tasks(tasks_path: Path, oracle_path: Path) -> tuple[list[Task], dict]:
    records = _read_jsonl(tasks_path, "task input")
    oracles = _load_oracles(oracle_path)
    tasks: list[Task] = []
    seen: set[tuple[str, str]] = set()
    for number, record in enumerate(records, start=1):
        try:
            task = Task.model_validate(record)
        except ValidationError as error:
            raise ValueError(f"task input line {number} is invalid") from error
        key = (task.asset_id, task.policy_version)
        if key in seen:
            raise ValueError(f"duplicate task record for {key[0]!r}, {key[1]!r}")
        if key not in oracles:
            raise ValueError(f"missing oracle for {key[0]!r}, {key[1]!r}")
        seen.add(key)
        tasks.append(task)
    unmatched = sorted(set(oracles) - seen)
    if unmatched:
        asset_id, policy_version = unmatched[0]
        raise ValueError(f"unmatched oracle for {asset_id!r}, {policy_version!r}")
    return tasks, oracles


def run_batch(
    tasks_path: Path,
    oracle_path: Path,
    cases_path: Path,
    evidence_path: Path,
    output_path: Path,
    *,
    teacher: Teacher,
    budget: TeacherBudget,
    data_classification: str,
    max_records: int | None = None,
) -> BatchOutcome:
    """Run one audited batch; final JSONL exists only when every input completes."""

    if data_classification not in {"synthetic", "public"}:
        raise ValueError("data classification must be synthetic or public")
    if max_records is not None and (
        isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 0
    ):
        raise ValueError("max_records must be a non-negative integer")
    inputs = [tasks_path, oracle_path, cases_path, evidence_path]
    _reject_input_overwrite(output_path, inputs)
    tasks, oracles = _load_validated_tasks(tasks_path, oracle_path)
    case_store, evidence_store = _load_stores(cases_path, evidence_path)
    rows: list[dict[str, object]] = []

    for task in tasks:
        if max_records is not None and len(rows) >= max_records:
            _checkpoint(
                output_path,
                rows,
                budget,
                total_records=len(tasks),
                classification=data_classification,
                status="incomplete",
                reason="max_records",
            )
            return BatchOutcome(False, len(rows), "max_records")
        try:
            result = generate_teacher_sample(
                task,
                oracles[(task.asset_id, task.policy_version)],
                case_store,
                evidence_store,
                teacher,
                data_classification=data_classification,
            )
        except TeacherBudgetExceeded as error:
            _checkpoint(
                output_path,
                rows,
                budget,
                total_records=len(tasks),
                classification=data_classification,
                status="incomplete",
                reason=error.reason,
            )
            return BatchOutcome(False, len(rows), error.reason)
        except TeacherRequestError:
            _checkpoint(
                output_path,
                rows,
                budget,
                total_records=len(tasks),
                classification=data_classification,
                status="incomplete",
                reason="teacher_request_failed",
            )
            return BatchOutcome(False, len(rows), "teacher_request_failed")
        except (RuntimeError, ValueError):
            _checkpoint(
                output_path,
                rows,
                budget,
                total_records=len(tasks),
                classification=data_classification,
                status="incomplete",
                reason="invalid_teacher_response",
            )
            return BatchOutcome(False, len(rows), "invalid_teacher_response")
        rows.append({**result.row, "metadata": result.metadata})
        _checkpoint(
            output_path,
            rows,
            budget,
            total_records=len(tasks),
            classification=data_classification,
            status="in_progress",
            reason=None,
        )

    write_jsonl_atomically(rows, output_path)
    partial, _checkpoint_path = _sidecar_paths(output_path)
    partial.unlink(missing_ok=True)
    _checkpoint(
        output_path,
        rows,
        budget,
        total_records=len(tasks),
        classification=data_classification,
        status="complete",
        reason=None,
    )
    return BatchOutcome(True, len(rows))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks_path", type=Path)
    parser.add_argument("oracle_path", type=Path)
    parser.add_argument("cases_path", type=Path)
    parser.add_argument("evidence_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--data-classification", default="synthetic")
    parser.add_argument(
        "--allow-external-data",
        action="store_true",
        help=(
            "acknowledge sending Task, Oracle, and any selected sanitized case/evidence "
            "records to Gemini"
        ),
    )
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--max-requests", type=int, default=2000)
    parser.add_argument(
        "--max-estimated-cost",
        type=float,
        help="hard USD cap enforced by worst-case input/output cost reservation",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--initial-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--max-backoff-seconds", type=float, default=30.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="finite per-request generation limit used in cost reservation",
    )
    parser.add_argument("--input-cost-per-million", type=float)
    parser.add_argument("--output-cost-per-million", type=float)
    return parser


def _finite_optional(value: float | None, name: str) -> float | None:
    if value is not None and (not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if not args.allow_external_data:
            raise ValueError(
                "Gemini synthesis sends Task, Oracle, and selected case/evidence records; "
                "explicit --allow-external-data acknowledgement is required"
            )
        if args.data_classification not in {"synthetic", "public"}:
            raise ValueError("data classification must be synthetic or public")
        _finite_optional(args.max_estimated_cost, "max_estimated_cost")
        if args.max_estimated_cost is not None and (
            args.input_cost_per_million is None or args.output_cost_per_million is None
        ):
            raise ValueError("cost budget requires both input and output token prices")
        budget = TeacherBudget(
            max_requests=args.max_requests,
            max_estimated_cost_usd=args.max_estimated_cost,
        )
        teacher = GeminiTeacher(
            model=args.model,
            max_attempts=args.max_attempts,
            initial_backoff_seconds=args.initial_backoff_seconds,
            max_backoff_seconds=args.max_backoff_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
            max_output_tokens=args.max_output_tokens,
            input_cost_per_million=args.input_cost_per_million,
            output_cost_per_million=args.output_cost_per_million,
            budget=budget,
        )
        outcome = run_batch(
            args.tasks_path,
            args.oracle_path,
            args.cases_path,
            args.evidence_path,
            args.output_path,
            teacher=teacher,
            budget=budget,
            data_classification=args.data_classification,
            max_records=args.max_records,
        )
        if not outcome.complete:
            raise ValueError(
                f"batch incomplete ({outcome.reason}); inspect the checkpoint manifest"
            )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
