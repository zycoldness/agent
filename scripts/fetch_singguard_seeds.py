"""Fetch governed open seeds for English SingGuard synthesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from risk_agent.singguard_sources import fetch_configured_seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = fetch_configured_seeds(args.catalog, args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
