"""Generate bounded two-hop SFT candidates with an external Gemini teacher."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError

from export_sft import _load_oracles, _load_stores, _read_jsonl, write_jsonl_atomically
from risk_agent.contracts import Task
from risk_agent.synthesis import generate_teacher_sample
from risk_agent.teacher import GeminiTeacher


def _reject_input_overwrite(output: Path, inputs: list[Path]) -> None:
    output_resolved = output.resolve(strict=False)
    for input_path in inputs:
        if output_resolved == input_path.resolve(strict=False):
            raise ValueError("output path must not overwrite an input file")
        if output.exists() and input_path.exists() and output.samefile(input_path):
            raise ValueError("output path must not overwrite an input file")


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
        help="explicitly acknowledge that task and oracle data will be sent to Gemini",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--initial-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--input-cost-per-million", type=float)
    parser.add_argument("--output-cost-per-million", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if not args.allow_external_data:
            raise ValueError(
                "Gemini synthesis requires explicit --allow-external-data acknowledgement"
            )
        if args.data_classification not in {"synthetic", "public"}:
            raise ValueError("data classification must be synthetic or public")
        input_paths = [args.tasks_path, args.oracle_path, args.cases_path, args.evidence_path]
        _reject_input_overwrite(args.output_path, input_paths)

        task_records = _read_jsonl(args.tasks_path, "task input")
        oracles = _load_oracles(args.oracle_path)
        case_store, evidence_store = _load_stores(args.cases_path, args.evidence_path)
        teacher = GeminiTeacher(
            model=args.model,
            max_attempts=args.max_attempts,
            initial_backoff_seconds=args.initial_backoff_seconds,
            input_cost_per_million=args.input_cost_per_million,
            output_cost_per_million=args.output_cost_per_million,
        )

        rows: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for number, record in enumerate(task_records, start=1):
            try:
                task = Task.model_validate(record)
            except ValidationError as error:
                raise ValueError(f"task input line {number} is invalid") from error
            key = (task.asset_id, task.policy_version)
            if key in seen:
                raise ValueError(f"duplicate task record for {key[0]!r}, {key[1]!r}")
            seen.add(key)
            oracle = oracles.get(key)
            if oracle is None:
                raise ValueError(f"missing oracle for {key[0]!r}, {key[1]!r}")
            result = generate_teacher_sample(
                task,
                oracle,
                case_store,
                evidence_store,
                teacher,
                data_classification=args.data_classification,
            )
            rows.append({**result.row, "metadata": result.metadata})

        unmatched = sorted(set(oracles) - seen)
        if unmatched:
            asset_id, policy_version = unmatched[0]
            raise ValueError(f"unmatched oracle for {asset_id!r}, {policy_version!r}")
        write_jsonl_atomically(rows, args.output_path)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

