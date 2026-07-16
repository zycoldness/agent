import json
from pathlib import Path

import pytest


def test_cli_requires_model_unless_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.generate_singguard_queries import main

    monkeypatch.delenv("GEMINI_GENERATOR_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    with pytest.raises(SystemExit) as error:
        main(["data/active_policies.jsonl", "out", "--count", "100"])
    assert error.value.code == 2


def test_cli_dry_run_writes_balanced_plan(tmp_path: Path, capsys) -> None:
    from scripts.generate_singguard_queries import main

    output = tmp_path / "plan"
    assert main([
        "data/active_policies.jsonl",
        str(output),
        "--count", "100",
        "--seed", "42",
        "--dry-run",
    ]) == 0

    rows = [json.loads(line) for line in (output / "plan.jsonl").read_text().splitlines()]
    manifest = json.loads((output / "manifest.json").read_text())
    stdout = json.loads(capsys.readouterr().out)
    assert len(rows) == 100
    assert len({row["blueprint_id"] for row in rows}) == 100
    assert manifest["label_counts"] == {"safe": 50, "unsafe": 50}
    assert manifest["shape_counts"] == {"query": 70, "query_response": 30}
    assert stdout == manifest


def test_cli_help_lists_minimal_query_options(capsys) -> None:
    from scripts.generate_singguard_queries import main

    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--count {100,500,2000}" in help_text
    assert "--dry-run" in help_text
    assert "--seeds" in help_text
