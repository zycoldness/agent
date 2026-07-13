"""Normalize user-supplied public benchmark and governed regulatory inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from risk_agent.public_data import run_public_data_import


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="governed public-source YAML config")
    parser.add_argument("output_dir", type=Path, help="new normalized output directory")
    arguments = parser.parse_args()

    try:
        report = run_public_data_import(arguments.config, arguments.output_dir)
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        message = str(error).splitlines()[0][:200]
        print(f"error: {message}", file=sys.stderr)
        return 2
    print(f"public assets: {report['public_asset_count']}")
    print(f"sanitized cases: {report['sanitized_case_count']}")
    print(f"quarantined cases: {report['quarantined_case_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
