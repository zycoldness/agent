"""Prepare deterministic, leak-separated Track A GRPO and OPSD bundles."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from risk_agent.ms_swift_sft import (
    SplitRatios,
    _build_rows,
    _ensure_distinct_sources,
    _jsonl_bytes,
    _publish_reserved_bundle,
    _split_for,
    _validate_ratios,
)


RLMode = Literal["grpo", "opsd"]
_SPLITS = ("train", "dev", "holdout")


def _opsd_teacher_prompt(system: str, user: str, solution: str) -> str:
    return (
        f"{system}\n\nStudent input:\n{user}\n\n"
        f"Privileged reference final decision (teacher branch only):\n{solution}\n\n"
        "Produce the best final_decision JSON action."
    )


def prepare_rl_bundle(
    mode: RLMode,
    task_path: Path,
    oracle_path: Path,
    output_dir: Path,
    *,
    ratios: SplitRatios = SplitRatios(),
    seed: int = 42,
) -> dict[str, object]:
    """Validate sources and exclusively publish prompt-only Track A RL rows."""

    if mode not in ("grpo", "opsd"):
        raise ValueError("mode must be grpo or opsd")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    _validate_ratios(ratios)
    _ensure_distinct_sources([task_path, oracle_path])
    output_dir = output_dir.absolute()
    if output_dir.exists():
        raise ValueError("output directory must not exist")

    exported, sources = _build_rows("track_a", task_path, oracle_path, None, None)
    split_rows: dict[str, list[dict[str, object]]] = {name: [] for name in _SPLITS}
    split_assets: dict[str, list[str]] = {name: [] for name in _SPLITS}
    for asset_id, sft_row in exported:
        messages = sft_row["messages"]
        prompt = messages[:2]
        solution = messages[2]["content"]
        if mode == "grpo":
            row: dict[str, object] = {"messages": prompt, "solution": solution}
        else:
            row = {
                "messages": prompt,
                "teacher_prompt": _opsd_teacher_prompt(
                    prompt[0]["content"], prompt[1]["content"], solution
                ),
            }
        split = _split_for(asset_id, seed, ratios)
        split_rows[split].append(row)
        split_assets[split].append(asset_id)

    payloads = {f"{name}.jsonl": _jsonl_bytes(split_rows[name]) for name in _SPLITS}
    files = {
        filename: {
            "bytes": len(payload),
            "records": len(split_rows[filename.removesuffix(".jsonl")]),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for filename, payload in payloads.items()
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "format": "ms-swift-track-a-rl-jsonl",
        "status": "complete",
        "track": "track_a",
        "mode": mode,
        "seed": seed,
        "ratios": {"train": ratios.train, "dev": ratios.dev, "holdout": ratios.holdout},
        "split_algorithm": "sha256-seed-null-asset-id",
        "sources": sources,
        "split_counts": {name: len(split_rows[name]) for name in _SPLITS},
        "asset_group_counts": {name: len(set(split_assets[name])) for name in _SPLITS},
        "split_asset_ids": {name: sorted(split_assets[name]) for name in _SPLITS},
        "files": files,
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ValueError("output directory must not exist") from error
    identity = os.stat(output_dir, follow_symlinks=False)
    if os.name == "posix":
        output_dir.chmod(0o700)
    _publish_reserved_bundle(output_dir, identity, payloads, manifest_payload)
    return manifest
