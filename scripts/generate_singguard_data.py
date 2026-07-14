"""Generate an audited English text-only SingGuard dataset with Gemini."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import ValidationError

from risk_agent.singguard_prompts import prompt_sha256
from risk_agent.singguard_sources import SeedRecord
from risk_agent.singguard_synthesis import (
    GENERATOR_RESPONSE_SCHEMA,
    VERIFIER_RESPONSE_SCHEMA,
    AnchorBlueprint,
    plan_blueprints,
    run_singguard_batch,
)
from risk_agent.teacher import GeminiTeacher, TeacherBudget


class ProgressBar:
    """Small stderr progress renderer that keeps stdout machine-readable."""

    def __init__(
        self,
        *,
        total: int,
        stream: Callable[[str], object] = sys.stderr.write,
        clock: Callable[[], float] = time.monotonic,
        width: int = 24,
    ) -> None:
        self.total = max(1, total)
        self._stream = stream
        self._clock = clock
        self._width = width
        self._started: float | None = None

    def __call__(self, event: Mapping[str, object]) -> None:
        now = self._clock()
        if self._started is None:
            self._started = now
        completed = min(self.total, max(0, int(event.get("completed", 0))))
        ratio = completed / self.total
        filled = round(ratio * self._width)
        bar = "#" * filled + "-" * (self._width - filled)
        elapsed = max(0.0, now - self._started)
        if completed:
            eta_seconds = elapsed / completed * (self.total - completed)
            eta = _duration(eta_seconds)
        else:
            eta = "--"
        phase = str(event.get("phase", "working"))
        line = (
            f"\r[{bar}] {completed:>{len(str(self.total))}}/{self.total} "
            f"{ratio * 100:5.1f}% {phase:<15} "
            f"accepted={int(event.get('accepted', 0))} "
            f"rejected={int(event.get('rejected', 0))} "
            f"requests={int(event.get('requests', 0))} "
            f"elapsed={_duration(elapsed)} ETA={eta}"
        )
        if phase in {"complete", "incomplete", "stopped"}:
            line += "\n"
        self._stream(line)


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{second:02d}"


def _write_jsonl(path: Path, rows: tuple[AnchorBlueprint, ...]) -> None:
    path.write_text(
        "".join(item.model_dump_json() + "\n" for item in rows),
        encoding="utf-8",
    )


def _load_seeds(path: Path) -> tuple[SeedRecord, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        raise ValueError("cannot read the seed JSONL") from None
    records: list[SeedRecord] = []
    try:
        for line in lines:
            if line.strip():
                records.append(SeedRecord.model_validate_json(line))
    except (ValidationError, ValueError):
        raise ValueError("seed JSONL contains an invalid record") from None
    if not records:
        raise ValueError("seed JSONL is empty")
    return tuple(records)


def _assign_seeds(
    plan: tuple[AnchorBlueprint, ...], records: tuple[SeedRecord, ...]
) -> tuple[tuple[AnchorBlueprint, ...], dict[str, str], dict[str, str]]:
    assigned: list[AnchorBlueprint] = []
    texts: dict[str, str] = {}
    licenses: dict[str, str] = {}
    cursor = 0
    for item in plan:
        if not item.use_open_seed:
            assigned.append(item)
            continue
        record = records[cursor % len(records)]
        cursor += 1
        item = item.model_copy(update={"source_id": f"{record.source}:{record.source_id}"})
        assigned.append(item)
        texts[item.anchor_id] = record.text
        licenses[item.anchor_id] = record.license
    return tuple(assigned), texts, licenses


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--anchors", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generator-model", default=os.environ.get("GEMINI_GENERATOR_MODEL"))
    parser.add_argument("--verifier-model", default=os.environ.get("GEMINI_VERIFIER_MODEL"))
    parser.add_argument("--seeds", type=Path)
    parser.add_argument("--allow-external-data", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--provider-attempts", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-requests", type=int, default=800)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--input-cost-per-million", type=float)
    parser.add_argument("--output-cost-per-million", type=float)
    parser.add_argument("--max-cost-usd", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.output_dir.exists() and not args.resume:
        parser.error("output directory must not exist")
    if args.resume and args.plan_only:
        parser.error("--resume cannot be combined with --plan-only")
    try:
        plan = plan_blueprints(args.anchors, seed=args.seed)
    except ValueError as error:
        parser.error(str(error))

    if args.plan_only:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _write_jsonl(args.output_dir / "plan.jsonl", plan)
        manifest = {
            "schema": "singguard-plan-v1",
            "status": "planned",
            "planned_anchors": len(plan),
            "seed": args.seed,
            "prompt_hashes": {
                name: prompt_sha256(name)
                for name in ("guard", "agent", "generator", "verifier")
            },
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
        return 0

    if not args.generator_model or not args.verifier_model:
        parser.error("real generation requires --generator-model and --verifier-model")
    if args.seeds and not args.allow_external_data:
        parser.error("--seeds requires --allow-external-data")
    if args.allow_external_data and not args.seeds:
        parser.error("--allow-external-data requires --seeds")
    if (args.input_cost_per_million is None) != (args.output_cost_per_million is None):
        parser.error("both input and output prices are required together")
    if args.max_cost_usd is not None and args.input_cost_per_million is None:
        parser.error("--max-cost-usd requires input and output prices")

    seed_texts: dict[str, str] = {}
    seed_licenses: dict[str, str] = {}
    if args.seeds:
        try:
            plan, seed_texts, seed_licenses = _assign_seeds(plan, _load_seeds(args.seeds))
        except ValueError as error:
            parser.error(str(error))

    budget = TeacherBudget(
        max_requests=args.max_requests,
        max_estimated_cost_usd=args.max_cost_usd,
    )
    common = {
        "max_attempts": args.provider_attempts,
        "request_timeout_seconds": args.request_timeout,
        "input_cost_per_million": args.input_cost_per_million,
        "output_cost_per_million": args.output_cost_per_million,
        "max_output_tokens": args.max_output_tokens,
        "budget": budget,
    }
    generator = GeminiTeacher(
        model=args.generator_model,
        response_schema=GENERATOR_RESPONSE_SCHEMA,
        temperature=0.7,
        **common,
    )
    verifier = GeminiTeacher(
        model=args.verifier_model,
        response_schema=VERIFIER_RESPONSE_SCHEMA,
        temperature=0.0,
        **common,
    )
    try:
        manifest = run_singguard_batch(
            plan,
            generator=generator,
            verifier=verifier,
            budget=budget,
            output_dir=args.output_dir,
            seed=args.seed,
            seed_texts=seed_texts,
            seed_licenses=seed_licenses,
            max_retries=args.max_retries,
            pilot=args.pilot,
            resume=args.resume,
            progress=ProgressBar(total=len(plan)),
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
