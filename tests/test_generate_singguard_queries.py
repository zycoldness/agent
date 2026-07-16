import json
from pathlib import Path

import pytest


def test_load_seeds_keeps_first_row_for_duplicate_content_hash(
    tmp_path: Path, capsys
) -> None:
    from scripts.generate_singguard_queries import _load_seeds

    content_hash = "a" * 64
    common = {
        "provenance_url": "https://example.com/dataset",
        "license": "CC-BY-4.0",
        "usage_scope": "research",
        "source_role": "query_seed",
        "text": "The same normalized seed content",
        "source_label": None,
        "content_hash": content_hash,
        "retrieved_at": "2026-07-16T00:00:00Z",
        "adapter_version": "v1",
    }
    path = tmp_path / "seeds.jsonl"
    rows = [
        {**common, "source": "first", "source_id": "1"},
        {**common, "source": "second", "source_id": "2"},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    records = _load_seeds(path)

    assert len(records) == 1
    assert records[0].source == "first"
    assert "ignored 1 duplicate seed rows" in capsys.readouterr().err


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


def test_cli_wires_separate_generator_and_blind_semantic_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import scripts.generate_singguard_queries as cli

    created: list[tuple[object, dict[str, object]]] = []

    class FixtureTeacher:
        def generate(self, _request):  # pragma: no cover - wiring only
            raise AssertionError("provider must not be called by wiring test")

    def teacher_factory(**kwargs):
        teacher = FixtureTeacher()
        created.append((teacher, kwargs))
        return teacher

    captured: dict[str, object] = {}

    def fake_run_query_batch(**kwargs):
        captured.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "GeminiTeacher", teacher_factory)
    monkeypatch.setattr(cli, "run_query_batch", fake_run_query_batch)
    output = tmp_path / "query"

    assert cli.main([
        "data/active_policies.jsonl",
        str(output),
        "--count", "100",
        "--model", "fixture-model",
    ]) == 0

    assert len(created) == 2
    assert created[0][1]["response_schema"] == cli.CONTENT_BATCH_SCHEMA
    assert created[1][1]["response_schema"] == cli.SEMANTIC_REVIEW_SCHEMA
    assert captured["teacher"] is created[0][0]
    assert captured["verifier"] is created[1][0]
    capsys.readouterr()
