"""Validate, inspect, or launch one single-device ms-swift 4.3 RL run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from risk_agent.ms_swift_rl_launcher import prepare_rl_launch, run_rl_launch


def _print(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.dry_run:
            _print(prepare_rl_launch(
                args.bundle_dir, args.config, args.output_dir, model=args.model,
                adapter=args.adapter, plugin=args.plugin, device=args.device,
                expected_manifest_sha256=args.expected_manifest_sha256,
            ).as_dict())
            return 0
        run_rl_launch(
            args.bundle_dir, args.config, args.output_dir, model=args.model,
            adapter=args.adapter, plugin=args.plugin, device=args.device, on_execute=_print,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
