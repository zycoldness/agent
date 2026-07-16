"""Generate English SingGuard query content with a Gemini teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import ValidationError

from risk_agent.singguard_generation import load_active_policies
from risk_agent.singguard_query_generation import (
    CONTENT_BATCH_SCHEMA,
    SEMANTIC_REVIEW_SCHEMA,
    plan_blueprints,
    run_query_batch,
)
from risk_agent.singguard_sources import SeedRecord
from risk_agent.teacher import GeminiTeacher, TeacherBudget


class ProgressBar:
    """Compact stderr progress; stdout remains one machine-readable manifest."""

    def __init__(
        self,
        total: int,
        *,
        stream: Callable[[str], object] = sys.stderr.write,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.total = total
        self._stream = stream
        self._clock = clock
        self._started: float | None = None

    def __call__(self, event: Mapping[str, object]) -> None:
        now = self._clock()
        if self._started is None:
            self._started = now
        accepted = int(event.get("accepted", 0))
        rejected = int(event.get("rejected", 0))
        completed = min(self.total, accepted + rejected)
        ratio = completed / self.total
        filled = round(ratio * 24)
        elapsed = max(0.0, now - self._started)
        pending = int(event.get("pending", self.total - completed))
        status = str(event.get("status", "working"))
        line = (
            f"\r[{'#' * filled}{'-' * (24 - filled)}] "
            f"{completed}/{self.total} {ratio * 100:5.1f}% "
            f"accepted={accepted} rejected={rejected} pending={pending} "
            f"elapsed={elapsed:0.1f}s status={status}"
        )
        if completed == self.total or status == "incomplete":
            line += "\n"
        self._stream(line)


def _stable_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_seeds(path: Path | None) -> tuple[SeedRecord, ...]:
    if path is None:
        return ()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        records = tuple(
            SeedRecord.model_validate_json(line)
            for line in lines
            if line.strip()
        )
    except (OSError, ValidationError, ValueError):
        raise ValueError("seed JSONL contains an invalid record") from None
    unique_records: list[SeedRecord] = []
    seen_hashes: set[str] = set()
    for record in records:
        if record.content_hash in seen_hashes:
            continue
        seen_hashes.add(record.content_hash)
        unique_records.append(record)
    duplicate_count = len(records) - len(unique_records)
    if duplicate_count:
        sys.stderr.write(
            f"warning: ignored {duplicate_count} duplicate seed rows by content_hash\n"
        )
    return tuple(unique_records)


def _write_dry_plan(
    *,
    output_dir: Path,
    policies,
    seeds: tuple[SeedRecord, ...],
    count: int,
    seed: int,
) -> dict[str, object]:
    if output_dir.exists():
        raise ValueError("output directory must not exist")
    plan = plan_blueprints(policies, count=count, seed=seed, seed_records=seeds)
    plan_bytes = b"".join(
        _stable_json(item.model_dump(mode="json")) for item in plan
    )
    output_dir.mkdir(parents=True)
    (output_dir / "plan.jsonl").write_bytes(plan_bytes)
    manifest: dict[str, object] = {
        "schema": "singguard-query-plan-v1",
        "status": "planned",
        "count": len(plan),
        "seed": seed,
        "label_counts": {
            label: sum(item.intended_label == label for item in plan)
            for label in ("safe", "unsafe")
        },
        "shape_counts": {
            shape: sum(item.conversation_shape == shape for item in plan)
            for shape in ("query", "query_response")
        },
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
    }
    (output_dir / "manifest.json").write_bytes(_stable_json(manifest))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("active_policies", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seeds", type=Path)
    parser.add_argument("--count", type=int, choices=(100, 500, 2000), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument(
        "--max-attempts-per-blueprint", type=int, choices=(1, 2, 3), default=3
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_GENERATOR_MODEL")
        or os.environ.get("GEMINI_MODEL"),
    )
    parser.add_argument("--provider-attempts", type=int, choices=(1, 2, 3, 4, 5), default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-requests", type=int, default=1000)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.dry_run and not args.model:
        parser.error("--model or GEMINI_GENERATOR_MODEL is required")
    if args.dry_run and args.resume:
        parser.error("--dry-run cannot be combined with --resume")
    try:
        policies = load_active_policies(args.active_policies)
        seeds = _load_seeds(args.seeds)
        if args.dry_run:
            manifest = _write_dry_plan(
                output_dir=args.output_dir,
                policies=policies,
                seeds=seeds,
                count=args.count,
                seed=args.seed,
            )
        else:
            budget = TeacherBudget(max_requests=args.max_requests)
            teacher = GeminiTeacher(
                model=args.model,
                response_schema=CONTENT_BATCH_SCHEMA,
                temperature=0.8,
                max_attempts=args.provider_attempts,
                request_timeout_seconds=args.request_timeout,
                max_output_tokens=args.max_output_tokens,
                budget=budget,
            )
            verifier = GeminiTeacher(
                model=args.model,
                response_schema=SEMANTIC_REVIEW_SCHEMA,
                temperature=0.0,
                max_attempts=args.provider_attempts,
                request_timeout_seconds=args.request_timeout,
                max_output_tokens=args.max_output_tokens,
                budget=budget,
            )
            manifest = run_query_batch(
                policies=policies,
                seed_records=seeds,
                output_dir=args.output_dir,
                count=args.count,
                seed=args.seed,
                teacher=teacher,
                verifier=verifier,
                batch_size=args.batch_size,
                max_attempts_per_blueprint=args.max_attempts_per_blueprint,
                resume=args.resume,
                progress=ProgressBar(args.count),
            )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return 0 if manifest.get("status") in {"complete", "planned"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
