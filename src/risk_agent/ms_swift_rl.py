"""Convert Track A SFT rows into ms-swift GRPO or OPSD JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from risk_agent.ms_swift_sft import (
    SplitRatios,
    _build_rows,
    _split_for,
    _validate_ratios,
    _write_jsonl,
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
    """Write prompt-only rows while keeping Oracle data out of student messages."""

    if mode not in ("grpo", "opsd"):
        raise ValueError("mode must be grpo or opsd")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (2**63 - 1):
        raise ValueError("seed must be an integer from zero to 2^63-1")
    _validate_ratios(ratios)
    output_dir = output_dir.absolute()
    if output_dir.exists():
        raise ValueError("output directory must not exist")

    exported = _build_rows("track_a", task_path, oracle_path, None, None)
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

    manifest: dict[str, object] = {
        "track": "track_a",
        "mode": mode,
        "seed": seed,
        "ratios": {"train": ratios.train, "dev": ratios.dev, "holdout": ratios.holdout},
        "split_counts": {name: len(split_rows[name]) for name in _SPLITS},
        "asset_group_counts": {name: len(split_assets[name]) for name in _SPLITS},
        "split_asset_ids": {name: sorted(split_assets[name]) for name in _SPLITS},
        "asset_policy_versions": {
            asset_id: sorted(versions) for asset_id, versions in sorted(asset_policy_versions.items())
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ValueError("output directory must not exist") from error
    for split in _SPLITS:
        _write_jsonl(output_dir / f"{split}.jsonl", split_rows[split])
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest
