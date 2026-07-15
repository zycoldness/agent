"""Generate ms-swift SingGuard data from complete active-policy prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from risk_agent.singguard_generation import (
    GeminiAgentProvider,
    load_active_policies,
    load_moderation_samples,
    run_generation_batch,
)
from risk_agent.singguard_tools import ToolEnvironment
from risk_agent.teacher import TeacherBudget


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
        eta = _duration(elapsed / completed * (self.total - completed)) if completed else "--"
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("active_policies", type=Path)
    parser.add_argument("content_samples", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_GENERATOR_MODEL")
        or os.environ.get("GEMINI_MODEL"),
    )
    parser.add_argument("--tool-env", type=Path, default=Path("data/tool_env"))
    parser.add_argument("--max-tool-calls", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--provider-attempts", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-requests", type=int, default=500)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--input-cost-per-million", type=float)
    parser.add_argument("--output-cost-per-million", type=float)
    parser.add_argument("--max-cost-usd", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model or GEMINI_GENERATOR_MODEL is required")
    if (args.input_cost_per_million is None) != (args.output_cost_per_million is None):
        parser.error("both input and output prices are required together")
    if args.max_cost_usd is not None and args.input_cost_per_million is None:
        parser.error("--max-cost-usd requires input and output prices")
    try:
        policies = load_active_policies(args.active_policies)
        samples = load_moderation_samples(args.content_samples)
        environment = ToolEnvironment.load(args.tool_env)
        budget = TeacherBudget(
            max_requests=args.max_requests,
            max_estimated_cost_usd=args.max_cost_usd,
        )
        provider = GeminiAgentProvider(
            model=args.model,
            max_attempts=args.provider_attempts,
            request_timeout_seconds=args.request_timeout,
            input_cost_per_million=args.input_cost_per_million,
            output_cost_per_million=args.output_cost_per_million,
            max_output_tokens=args.max_output_tokens,
            budget=budget,
        )
        manifest = run_generation_batch(
            policies=policies,
            samples=samples,
            provider=provider,
            environment=environment,
            output_dir=args.output_dir,
            budget=budget,
            max_tool_calls=args.max_tool_calls,
            resume=args.resume,
            progress=ProgressBar(total=len(samples)),
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
