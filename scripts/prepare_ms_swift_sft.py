"""Prepare a deterministic ms-swift messages JSONL SFT bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from risk_agent.ms_swift_sft import SplitRatios, prepare_sft_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", choices=("track_a", "track_b"))
    parser.add_argument("input_path", type=Path)
    parser.add_argument("oracle_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--holdout-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_sft_bundle(
            args.track,
            args.input_path,
            args.oracle_path,
            args.output_dir,
            cases_path=args.cases,
            evidence_path=args.evidence,
            ratios=SplitRatios(
                train=args.train_ratio,
                dev=args.dev_ratio,
                holdout=args.holdout_ratio,
            ),
            seed=args.seed,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
