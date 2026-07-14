"""Prepare validated SingGuard examples for direct ms-swift SFT commands."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from risk_agent.ms_swift_sft import (
    SplitRatios,
    _read_jsonl,
    _split_for,
    _validate_ratios,
    _write_jsonl,
)
from risk_agent.singguard import SingGuardExample, render_sft_row


_SPLITS = ("train", "dev", "holdout")


def prepare_singguard_bundle(
    input_path: Path,
    output_dir: Path,
    *,
    ratios: SplitRatios = SplitRatios(),
    seed: int = 42,
) -> dict[str, object]:
    """Validate, group-split, and render one SingGuard JSONL dataset."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (2**63 - 1):
        raise ValueError("seed must be an integer from zero to 2^63-1")
    _validate_ratios(ratios)
    output_dir = output_dir.absolute()
    if output_dir.exists():
        raise ValueError("output directory must not exist")

    examples: list[SingGuardExample] = []
    keys: set[tuple[str, str, str]] = set()
    for number, record in enumerate(_read_jsonl(input_path, "SingGuard input"), start=1):
        try:
            example = SingGuardExample.model_validate(record)
        except ValidationError as error:
            raise ValueError(f"SingGuard input line {number} is invalid") from error
        key = (
            example.content.sample_id,
            example.policy.view_id,
            example.thinking_type,
        )
        if key in keys:
            raise ValueError(f"duplicate SingGuard example at line {number}")
        keys.add(key)
        examples.append(example)

    split_rows: dict[str, list[dict[str, object]]] = {name: [] for name in _SPLITS}
    split_groups: dict[str, set[str]] = {name: set() for name in _SPLITS}
    transition_counts: Counter[str] = Counter()
    thinking_type_counts: Counter[str] = Counter()
    for example in examples:
        split = _split_for(example.content.split_group, seed, ratios)
        split_rows[split].append(render_sft_row(example))
        split_groups[split].add(example.content.split_group)
        transition_counts[example.policy.transition] += 1
        thinking_type_counts[example.thinking_type] += 1

    manifest: dict[str, object] = {
        "schema": "singguard-v1",
        "seed": seed,
        "ratios": {"train": ratios.train, "dev": ratios.dev, "holdout": ratios.holdout},
        "split_counts": {name: len(split_rows[name]) for name in _SPLITS},
        "split_group_counts": {name: len(split_groups[name]) for name in _SPLITS},
        "transition_counts": dict(sorted(transition_counts.items())),
        "thinking_type_counts": dict(sorted(thinking_type_counts.items())),
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
