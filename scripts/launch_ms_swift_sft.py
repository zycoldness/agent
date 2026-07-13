"""Validate, inspect, or launch a single-device ms-swift 4.3 SFT run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from risk_agent.ms_swift_launcher import prepare_sft_launch, run_sft_launch


def _print_plan(plan: dict[str, object]) -> None:
    print(
        json.dumps(
            plan,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = prepare_sft_launch(
            args.bundle_dir,
            args.config,
            args.output_dir,
            model=args.model,
            device=args.device,
        )
        if args.dry_run:
            _print_plan(plan.as_dict())
            return 0
        run_sft_launch(
            args.bundle_dir,
            args.config,
            args.output_dir,
            model=args.model,
            device=args.device,
            on_execute=_print_plan,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
