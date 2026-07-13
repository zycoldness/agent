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


def _opsd_teacher_prompt(user: str, solution: str) -> str:
    return json.dumps(
        {
            "student_observation": user,
            "privileged_solution": solution,
            "instruction": "Produce exactly the best final_decision JSON action.",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (2**63 - 1):
        raise ValueError("seed must be an integer from zero to 2^63-1")
    _validate_ratios(ratios)
    _ensure_distinct_sources([task_path, oracle_path])
    output_dir = output_dir.absolute()
    if output_dir.exists():
        raise ValueError("output directory must not exist")

    exported, sources = _build_rows("track_a", task_path, oracle_path, None, None)
    split_rows: dict[str, list[dict[str, object]]] = {name: [] for name in _SPLITS}
    split_assets: dict[str, set[str]] = {name: set() for name in _SPLITS}
    asset_policy_versions: dict[str, set[str]] = {}
    for asset_id, policy_version, sft_row in exported:
        messages = sft_row["messages"]
        prompt = messages[:2]
        solution = messages[2]["content"]
        if mode == "grpo":
            row: dict[str, object] = {"messages": prompt, "solution": solution}
        else:
            row = {
                "messages": prompt,
                "teacher_prompt": _opsd_teacher_prompt(prompt[1]["content"], solution),
            }
        for field in ("images", "videos"):
            if field in sft_row:
                row[field] = sft_row[field]
        split = _split_for(asset_id, seed, ratios)
        split_rows[split].append(row)
        split_assets[split].add(asset_id)
        asset_policy_versions.setdefault(asset_id, set()).add(policy_version)

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
        "asset_group_counts": {name: len(split_assets[name]) for name in _SPLITS},
        "split_asset_ids": {name: sorted(split_assets[name]) for name in _SPLITS},
        "asset_policy_versions": {
            asset_id: sorted(versions) for asset_id, versions in sorted(asset_policy_versions.items())
        },
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
    return {
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }
