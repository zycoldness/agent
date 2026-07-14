"""ms-swift bundle preparation for SingGuard examples."""

import json
import subprocess
import sys
from pathlib import Path

from risk_agent.contracts import PolicyRule
from risk_agent.ms_swift_sft import SplitRatios
from risk_agent.singguard import (
    ContentSample,
    PolicyView,
    SingGuardAnnotation,
    SingGuardExample,
)


def _example(view_id: str, transition: str) -> SingGuardExample:
    rule = PolicyRule(
        rule_id="AD-001",
        title="Absolute efficacy claim",
        text="Do not guarantee a weight-loss result.",
    )
    return SingGuardExample(
        content=ContentSample(
            sample_id="asset-1",
            split_group="anchor-1",
            query="Guaranteed to lose ten pounds in seven days.",
        ),
        policy=PolicyView(
            view_id=view_id,
            active_policy=(rule,),
            style="full",
            transition=transition,
        ),
        annotation=SingGuardAnnotation(
            label="unsafe",
            rule_title="Absolute efficacy claim",
        ),
        thinking_type="fast",
    )


def test_bundle_keeps_every_policy_view_of_one_anchor_in_one_split(tmp_path) -> None:
    from risk_agent.ms_swift_singguard import prepare_singguard_bundle

    source = tmp_path / "examples.jsonl"
    source.write_text(
        _example("full", "unsafe_to_unsafe").model_dump_json()
        + "\n"
        + _example("rewritten", "unsafe_to_unsafe").model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    manifest = prepare_singguard_bundle(
        source,
        tmp_path / "bundle",
        ratios=SplitRatios(train=0, dev=1, holdout=0),
        seed=7,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "bundle" / "dev.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert manifest["split_counts"] == {"train": 0, "dev": 2, "holdout": 0}
    assert manifest["split_group_counts"] == {"train": 0, "dev": 1, "holdout": 0}
    assert manifest["transition_counts"] == {"unsafe_to_unsafe": 2}
    assert all(set(row) == {"messages"} for row in rows)


def test_smoke_fixture_covers_all_four_policy_transitions_without_empty_policy(tmp_path) -> None:
    from risk_agent.ms_swift_singguard import prepare_singguard_bundle

    source = Path(__file__).parents[1] / "data" / "fixtures" / "singguard_examples.jsonl"
    manifest = prepare_singguard_bundle(
        source,
        tmp_path / "bundle",
        ratios=SplitRatios(train=1, dev=0, holdout=0),
    )

    assert manifest["transition_counts"] == {
        "safe_to_safe": 1,
        "safe_to_unsafe": 1,
        "unsafe_to_safe": 1,
        "unsafe_to_unsafe": 1,
    }
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    assert all(record["policy"]["active_policy"] for record in records)


def test_prepare_cli_builds_a_singguard_bundle(tmp_path) -> None:
    repository = Path(__file__).parents[1]
    source = repository / "data" / "fixtures" / "singguard_examples.jsonl"
    output = tmp_path / "bundle"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_singguard_sft.py",
            str(source),
            str(output),
            "--train-ratio",
            "1",
            "--dev-ratio",
            "0",
            "--holdout-ratio",
            "0",
            "--seed",
            "7",
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["seed"] == 7
    assert manifest["split_counts"] == {"train": 4, "dev": 0, "holdout": 0}
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest
