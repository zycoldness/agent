"""Fetch only the fully validated public URLs in a source manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from risk_agent.crawler import fetch, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="YAML source manifest")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    arguments = parser.parse_args()

    # load_manifest validates every URL before this loop can issue a request.
    sources = load_manifest(arguments.manifest)
    for source in sources:
        print(fetch(source, arguments.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
