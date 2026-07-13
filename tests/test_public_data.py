"""Tests for governed public benchmark and regulatory-source imports."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys

import httpx
import pytest
from pydantic import ValidationError
import yaml

from risk_agent.crawler import Source, fetch
from risk_agent.public_data import (
    MAX_IMPORT_RECORDS,
    PublicAsset,
    SanitizedCase,
    import_mm_safety_bench,
    import_regulatory_crawler_artifact,
    import_regulatory_html,
    run_public_data_import,
)


MM_SOURCE_URL = "https://github.com/isXinLiu/MM-SafetyBench"
RETRIEVED_AT = "2026-07-13T00:00:00Z"


def _write_mm_fixture(root: Path, *, include_images: bool = True) -> None:
    questions = root / "data" / "processed_questions"
    questions.mkdir(parents=True)
    scenario = "01-Illegal_Activitiy"
    (questions / f"{scenario}.json").write_text(
        json.dumps(
            {
                "0": {
                    "Question": "original zero",
                    "Rephrased Question": "typo prompt zero",
                    "Rephrased Question(SD)": "sd prompt zero",
                },
                "1": {
                    "Question": "original one",
                    "Rephrased Question": "typo prompt one",
                    "Rephrased Question(SD)": "sd prompt one",
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "TinyVersion_ID_List.json").write_text(
        json.dumps([{"Scenario": scenario, "Sampled_ID_List": [1]}]),
        encoding="utf-8",
    )
    if include_images:
        for variant in ("SD", "SD_TYPO", "TYPO"):
            image = root / "data" / "imgs" / scenario / variant / "1.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"{variant}-image".encode())


def test_public_asset_is_immutable_and_contains_no_oracle_contract() -> None:
    asset = PublicAsset(
        source_dataset="MM-SafetyBench",
        source_item_id="01-Illegal_Activitiy:1:SD",
        scenario="01-Illegal_Activitiy",
        prompt="sd prompt one",
        media_paths=("data/imgs/01-Illegal_Activitiy/SD/1.jpg",),
        media_status="available",
        content_hash="a" * 64,
        license_id="CC-BY-NC-4.0",
        usage_scope="smoke_only",
        source_url=MM_SOURCE_URL,
        retrieved_at=RETRIEVED_AT,
        split_group="MM-SafetyBench:01-Illegal_Activitiy:1",
    )

    assert asset.data_classification == "public"
    assert "label" not in PublicAsset.model_fields
    assert "oracle" not in PublicAsset.model_fields
    with pytest.raises(ValidationError, match="frozen"):
        asset.prompt = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("content_hash", "not-a-sha256"),
        ("source_url", "http://example.invalid/data"),
        ("retrieved_at", "not-a-timestamp"),
        ("media_paths", ("../outside.jpg",)),
    ],
)
def test_public_asset_rejects_invalid_provenance_fields(field: str, value: object) -> None:
    row = {
        "source_dataset": "MM-SafetyBench",
        "source_item_id": "01-Illegal_Activitiy:1:SD",
        "scenario": "01-Illegal_Activitiy",
        "prompt": "prompt",
        "media_paths": ("data/imgs/01-Illegal_Activitiy/SD/1.jpg",),
        "media_status": "available",
        "content_hash": "a" * 64,
        "license_id": "CC-BY-NC-4.0",
        "usage_scope": "smoke_only",
        "source_url": MM_SOURCE_URL,
        "retrieved_at": RETRIEVED_AT,
        "split_group": "MM-SafetyBench:01-Illegal_Activitiy:1",
    }
    row[field] = value

    with pytest.raises(ValidationError):
        PublicAsset.model_validate(row)


def test_mm_importer_uses_tiny_ids_and_emits_three_deterministic_variants(tmp_path: Path) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)

    first = import_mm_safety_bench(
        repo,
        source_url=MM_SOURCE_URL,
        retrieved_at=RETRIEVED_AT,
        use_tiny=True,
        max_records=10,
    )
    second = import_mm_safety_bench(
        repo,
        source_url=MM_SOURCE_URL,
        retrieved_at=RETRIEVED_AT,
        use_tiny=True,
        max_records=10,
    )

    assert first == second
    assert [asset.source_item_id for asset in first] == [
        "01-Illegal_Activitiy:1:SD",
        "01-Illegal_Activitiy:1:SD_TYPO",
        "01-Illegal_Activitiy:1:TYPO",
    ]
    assert [asset.prompt for asset in first] == [
        "sd prompt one",
        "typo prompt one",
        "typo prompt one",
    ]
    assert all(asset.media_status == "available" for asset in first)
    assert all(asset.license_id == "CC-BY-NC-4.0" for asset in first)
    assert all(asset.usage_scope == "smoke_only" for asset in first)
    assert len({asset.content_hash for asset in first}) == 3
    assert {asset.split_group for asset in first} == {
        "MM-SafetyBench:01-Illegal_Activitiy:1"
    }


def test_mm_importer_requires_explicit_opt_in_for_missing_images(tmp_path: Path) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo, include_images=False)

    with pytest.raises(ValueError, match="missing MM-SafetyBench image"):
        import_mm_safety_bench(
            repo,
            source_url=MM_SOURCE_URL,
            retrieved_at=RETRIEVED_AT,
            use_tiny=True,
            max_records=10,
        )

    records = import_mm_safety_bench(
        repo,
        source_url=MM_SOURCE_URL,
        retrieved_at=RETRIEVED_AT,
        use_tiny=True,
        allow_missing_media=True,
        max_records=10,
    )

    assert len(records) == 3
    assert all(record.media_status == "missing" for record in records)
    assert not (repo / records[0].media_paths[0]).exists()


def test_mm_importer_applies_allowlist_then_deterministic_record_cap(tmp_path: Path) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)

    records = import_mm_safety_bench(
        repo,
        source_url=MM_SOURCE_URL,
        retrieved_at=RETRIEVED_AT,
        use_tiny=True,
        scenario_allowlist={"01-Illegal_Activitiy"},
        max_records=2,
    )

    assert [record.source_item_id for record in records] == [
        "01-Illegal_Activitiy:1:SD",
        "01-Illegal_Activitiy:1:SD_TYPO",
    ]


@pytest.mark.parametrize("max_records", [0, -1, MAX_IMPORT_RECORDS + 1])
def test_mm_importer_rejects_invalid_record_caps(tmp_path: Path, max_records: int) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)

    with pytest.raises(ValueError, match="max_records"):
        import_mm_safety_bench(
            repo,
            source_url=MM_SOURCE_URL,
            retrieved_at=RETRIEVED_AT,
            max_records=max_records,
        )


def test_mm_importer_rejects_unknown_allowlisted_scenario(tmp_path: Path) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)

    with pytest.raises(ValueError, match="unknown scenario"):
        import_mm_safety_bench(
            repo,
            source_url=MM_SOURCE_URL,
            retrieved_at=RETRIEVED_AT,
            scenario_allowlist={"99-Unknown"},
            max_records=10,
        )


def test_mm_importer_rejects_tiny_ids_absent_from_questions(tmp_path: Path) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)
    (repo / "TinyVersion_ID_List.json").write_text(
        json.dumps(
            [
                {
                    "Scenario": "01-Illegal_Activitiy",
                    "Sampled_ID_List": [999],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown question id"):
        import_mm_safety_bench(
            repo,
            source_url=MM_SOURCE_URL,
            retrieved_at=RETRIEVED_AT,
            max_records=10,
        )


REGULATORY_HTML = """\
<html><body>
  <script>window.secret = 'ignore me';</script>
  <article>
    <p>案例一：某食品广告宣称普通食品能够治疗疾病，监管部门依法查处。</p>
    <p>案例一：某食品广告宣称普通食品能够治疗疾病，监管部门依法查处。</p>
    <li>案例二：直播间使用无法核验的功效保证，并引导消费者下单购买。</li>
  </article>
</body></html>
"""


def test_regulatory_html_import_sanitizes_deduplicates_and_quarantines_unapproved_source() -> None:
    cases = import_regulatory_html(
        REGULATORY_HTML.encode("utf-8"),
        source_dataset="SAMR-public-cases",
        source_url="https://www.samr.gov.cn/example/cases.html",
        retrieved_at=RETRIEVED_AT,
        license_id="RIGHTS-REVIEW-REQUIRED",
        usage_scope="research_only",
        license_review_status="review_required",
        terms_review_status="review_required",
        max_records=10,
    )

    assert len(cases) == 2
    assert all(isinstance(case, SanitizedCase) for case in cases)
    assert all(case.publication_status == "quarantined" for case in cases)
    assert all(case.data_classification == "public" for case in cases)
    assert all(
        case.source_content_hash
        == hashlib.sha256(REGULATORY_HTML.encode("utf-8")).hexdigest()
        for case in cases
    )
    assert all("<" not in case.text and "window.secret" not in case.text for case in cases)
    assert len({case.case_id for case in cases}) == 2
    assert "label" not in SanitizedCase.model_fields
    assert "rule_id" not in SanitizedCase.model_fields
    assert "oracle" not in SanitizedCase.model_fields


def test_regulatory_html_is_publishable_only_after_both_reviews_are_approved() -> None:
    cases = import_regulatory_html(
        REGULATORY_HTML,
        source_dataset="approved-public-cases",
        source_url="https://example.gov.cn/cases.html",
        retrieved_at=RETRIEVED_AT,
        license_id="PUBLIC-NOTICE-APPROVED",
        usage_scope="research_only",
        license_review_status="approved",
        terms_review_status="approved",
        max_records=1,
    )

    assert len(cases) == 1
    assert cases[0].publication_status == "published"


@pytest.mark.parametrize("license_id", ["", "unknown", "not_reviewed", "TBD"])
def test_regulatory_import_rejects_placeholder_licenses(license_id: str) -> None:
    with pytest.raises(ValueError, match="license_id"):
        import_regulatory_html(
            REGULATORY_HTML,
            source_dataset="cases",
            source_url="https://example.gov.cn/cases.html",
            retrieved_at=RETRIEVED_AT,
            license_id=license_id,
            usage_scope="research_only",
            license_review_status="review_required",
            terms_review_status="review_required",
            max_records=1,
        )


def test_regulatory_crawler_import_requires_an_intact_completed_artifact(tmp_path: Path) -> None:
    source = Source(
        url="https://example.gov.cn/cases.html",
        allowed_domains=("example.gov.cn",),
        source_type="public_case",
        license="RIGHTS-REVIEW-REQUIRED",
        terms_review="review_required",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=REGULATORY_HTML.encode("utf-8"))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        raw_path = fetch(source, tmp_path, client=client, retrieved_at=RETRIEVED_AT)

    cases = import_regulatory_crawler_artifact(
        source,
        tmp_path,
        source_dataset="governed-cases",
        license_id="RIGHTS-REVIEW-REQUIRED",
        usage_scope="research_only",
        license_review_status="review_required",
        terms_review_status="review_required",
        max_records=10,
    )
    assert len(cases) == 2
    assert all(case.retrieved_at == RETRIEVED_AT for case in cases)

    raw_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="complete crawler artifact"):
        import_regulatory_crawler_artifact(
            source,
            tmp_path,
            source_dataset="governed-cases",
            license_id="RIGHTS-REVIEW-REQUIRED",
            usage_scope="research_only",
            license_review_status="review_required",
            terms_review_status="review_required",
            max_records=10,
        )


def _write_import_config(path: Path, repo: Path, html_path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "max_records": 20,
                "mm_safety_bench": {
                    "repo_root": str(repo),
                    "use_tiny": True,
                    "allow_missing_media": False,
                    "scenario_allowlist": ["01-Illegal_Activitiy"],
                    "source_url": MM_SOURCE_URL,
                    "retrieved_at": RETRIEVED_AT,
                    "license_id": "CC-BY-NC-4.0",
                    "usage_scope": "smoke_only",
                },
                "regulatory_cases": [
                    {
                        "input_type": "local_html",
                        "local_html_path": str(html_path),
                        "source_dataset": "SAMR-public-cases",
                        "source_url": "https://www.samr.gov.cn/example/cases.html",
                        "retrieved_at": RETRIEVED_AT,
                        "license_id": "RIGHTS-REVIEW-REQUIRED",
                        "usage_scope": "research_only",
                        "license_review_status": "review_required",
                        "terms_review_status": "review_required",
                        "max_records": 10,
                    }
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def test_public_data_bundle_writes_normalized_outputs_report_and_completion_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)
    html_path = tmp_path / "samr.html"
    html_path.write_text(REGULATORY_HTML, encoding="utf-8")
    config = tmp_path / "public-sources.yaml"
    _write_import_config(config, repo, html_path)
    output = tmp_path / "normalized"

    report = run_public_data_import(config, output)

    asset_lines = (output / "public_assets.jsonl").read_text(encoding="utf-8").splitlines()
    case_lines = (output / "sanitized_cases.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(asset_lines) == 3
    assert len(case_lines) == 2
    assert report["public_asset_count"] == 3
    assert report["sanitized_case_count"] == 2
    assert report["missing_media_count"] == 0
    assert report["quarantined_case_count"] == 2
    assert report["official_evaluation_split"] is False
    assert "not an official" in report["split_note"]

    manifest = json.loads((output / "import_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["counts"] == {"public_assets": 3, "sanitized_cases": 2}
    for filename in ("public_assets.jsonl", "sanitized_cases.jsonl", "import_report.json"):
        expected = hashlib.sha256((output / filename).read_bytes()).hexdigest()
        assert manifest["files"][filename]["sha256"] == expected


def test_public_data_import_refuses_input_output_aliases_and_existing_outputs(tmp_path: Path) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)
    html_path = tmp_path / "samr.html"
    html_path.write_text(REGULATORY_HTML, encoding="utf-8")
    config = tmp_path / "public-sources.yaml"
    _write_import_config(config, repo, html_path)

    with pytest.raises(ValueError, match="overlap|alias"):
        run_public_data_import(config, repo)
    assert not (repo / "public_assets.jsonl").exists()

    output = tmp_path / "normalized"
    run_public_data_import(config, output)
    with pytest.raises(FileExistsError, match="already exists"):
        run_public_data_import(config, output)


def test_public_data_import_rejects_duplicate_case_content_before_writing(tmp_path: Path) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)
    html_path = tmp_path / "samr.html"
    html_path.write_text(REGULATORY_HTML, encoding="utf-8")
    config = tmp_path / "public-sources.yaml"
    _write_import_config(config, repo, html_path)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["regulatory_cases"].append(dict(document["regulatory_cases"][0]))
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    output = tmp_path / "normalized"

    with pytest.raises(ValueError, match="duplicate.*case"):
        run_public_data_import(config, output)

    assert not output.exists()


def test_public_data_cli_runs_without_network_and_respects_hard_record_limit(tmp_path: Path) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)
    html_path = tmp_path / "samr.html"
    html_path.write_text(REGULATORY_HTML, encoding="utf-8")
    config = tmp_path / "public-sources.yaml"
    _write_import_config(config, repo, html_path)
    output = tmp_path / "normalized"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_public_data.py",
            str(config),
            str(output),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "public assets: 3" in result.stdout
    assert (output / "import_manifest.json").is_file()

    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["max_records"] = MAX_IMPORT_RECORDS + 1
    too_large = tmp_path / "too-large.yaml"
    too_large.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="max_records"):
        run_public_data_import(too_large, tmp_path / "unused")
