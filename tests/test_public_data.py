"""Tests for governed public benchmark and regulatory-source imports."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import httpx
import pytest
from pydantic import ValidationError
import yaml

from risk_agent.crawler import Source, completion_path_for, fetch, metadata_path_for
from risk_agent.public_data import (
    MAX_IMPORT_RECORDS,
    MM_SAFETY_BENCH_RESTRICTIONS,
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

    forbidden = asset.model_dump()
    forbidden["oracle"] = {"label": "unsafe"}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PublicAsset.model_validate(forbidden)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("content_hash", "not-a-sha256"),
        ("source_url", "http://example.invalid/data"),
        ("source_url", "https://127.0.0.1/data"),
        ("source_url", "https://2130706433/data"),
        ("source_url", "https://example.invalid:8443/data"),
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
    assert all(
        asset.license_restrictions
        == (
            "CC-BY-NC-4.0 non-commercial research use only",
            "Upstream GPT-4 license restrictions apply",
            "Upstream Stable Diffusion license restrictions apply",
        )
        for asset in first
    )
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


def test_mm_importer_rejects_noncanonical_source_and_invalid_scenario_or_item_ids(tmp_path: Path) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)
    with pytest.raises(ValueError, match="canonical MM-SafetyBench"):
        import_mm_safety_bench(
            repo,
            source_url=f"{MM_SOURCE_URL}/",
            retrieved_at=RETRIEVED_AT,
            max_records=10,
        )

    questions = repo / "data" / "processed_questions"
    original = questions / "01-Illegal_Activitiy.json"
    original.rename(questions / "bad scenario.json")
    with pytest.raises(ValueError, match="scenario"):
        import_mm_safety_bench(
            repo,
            source_url=MM_SOURCE_URL,
            retrieved_at=RETRIEVED_AT,
            use_tiny=False,
            allow_missing_media=True,
            max_records=10,
        )

    (questions / "bad scenario.json").rename(original)
    original.write_text(
        json.dumps(
            {
                "../escape": {
                    "Rephrased Question": "prompt",
                    "Rephrased Question(SD)": "prompt",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="question id"):
        import_mm_safety_bench(
            repo,
            source_url=MM_SOURCE_URL,
            retrieved_at=RETRIEVED_AT,
            use_tiny=False,
            allow_missing_media=True,
            max_records=10,
        )


def test_mm_importer_rejects_symlinked_checkout_components(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)
    questions_dir = repo / "data" / "processed_questions"
    original = Path.is_symlink

    def fake_is_symlink(path: Path) -> bool:
        return path == questions_dir or original(path)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    with pytest.raises(ValueError, match="symlink"):
        import_mm_safety_bench(
            repo,
            source_url=MM_SOURCE_URL,
            retrieved_at=RETRIEVED_AT,
            max_records=10,
        )


def test_mm_importer_bounds_each_file_and_aggregate_bytes(tmp_path: Path) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)
    with pytest.raises(ValueError, match="byte limit"):
        import_mm_safety_bench(
            repo,
            source_url=MM_SOURCE_URL,
            retrieved_at=RETRIEVED_AT,
            max_file_bytes=8,
            max_total_bytes=1024,
            max_records=10,
        )
    with pytest.raises(ValueError, match="total byte limit"):
        import_mm_safety_bench(
            repo,
            source_url=MM_SOURCE_URL,
            retrieved_at=RETRIEVED_AT,
            max_file_bytes=1024,
            max_total_bytes=64,
            max_records=10,
        )


def test_mm_importer_rejects_duplicate_tiny_scenarios_and_json_keys(tmp_path: Path) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)
    scenario = "01-Illegal_Activitiy"
    (repo / "TinyVersion_ID_List.json").write_text(
        json.dumps(
            [
                {"Scenario": scenario, "Sampled_ID_List": [1]},
                {"Scenario": scenario, "Sampled_ID_List": [1]},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate scenario"):
        import_mm_safety_bench(
            repo,
            source_url=MM_SOURCE_URL,
            retrieved_at=RETRIEVED_AT,
            max_records=10,
        )

    (repo / "TinyVersion_ID_List.json").write_text(
        json.dumps([{"Scenario": scenario, "Sampled_ID_List": [1]}]),
        encoding="utf-8",
    )
    row = '{"Rephrased Question":"a","Rephrased Question(SD)":"b"}'
    (repo / "data" / "processed_questions" / f"{scenario}.json").write_text(
        f'{{"1":{row},"1":{row}}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
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
        allowed_domains=("www.samr.gov.cn",),
        retrieved_at=RETRIEVED_AT,
        license_id="RIGHTS-REVIEW-REQUIRED",
        usage_scope="research_only",
        license_review_status="review_required",
        terms_review_status="review_required",
        content_review_status="pending",
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


def test_regulatory_html_requires_license_terms_and_content_approval_to_publish() -> None:
    pending_cases = import_regulatory_html(
        REGULATORY_HTML,
        source_dataset="approved-public-cases",
        source_url="https://example.gov.cn/cases.html",
        allowed_domains=("example.gov.cn",),
        retrieved_at=RETRIEVED_AT,
        license_id="PUBLIC-NOTICE-APPROVED",
        usage_scope="research_only",
        license_review_status="approved",
        terms_review_status="approved",
        max_records=1,
    )
    assert pending_cases[0].publication_status == "quarantined"
    invalid_published = pending_cases[0].model_dump()
    invalid_published["publication_status"] = "published"
    with pytest.raises(ValidationError, match="approved license, terms, and content"):
        SanitizedCase.model_validate(invalid_published)
    forbidden_case = pending_cases[0].model_dump()
    forbidden_case["label"] = "unsafe"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SanitizedCase.model_validate(forbidden_case)

    cases = import_regulatory_html(
        REGULATORY_HTML,
        source_dataset="approved-public-cases",
        source_url="https://example.gov.cn/cases.html",
        allowed_domains=("example.gov.cn",),
        retrieved_at=RETRIEVED_AT,
        license_id="PUBLIC-NOTICE-APPROVED",
        usage_scope="research_only",
        license_review_status="approved",
        terms_review_status="approved",
        content_review_status="approved",
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
            allowed_domains=("example.gov.cn",),
            retrieved_at=RETRIEVED_AT,
            license_id=license_id,
            usage_scope="research_only",
            license_review_status="review_required",
            terms_review_status="review_required",
            content_review_status="pending",
            max_records=1,
        )


def test_regulatory_crawler_import_requires_an_intact_completed_artifact(tmp_path: Path) -> None:
    source = Source(
        url="https://example.gov.cn/cases.html",
        allowed_domains=("example.gov.cn",),
        source_type="public_case",
        license="RIGHTS-REVIEW-REQUIRED",
        license_review_status="pending",
        terms_review_status="pending",
        content_review_status="pending",
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
        max_records=10,
    )
    assert len(cases) == 2
    assert all(case.retrieved_at == RETRIEVED_AT for case in cases)
    assert all(case.publication_status == "quarantined" for case in cases)
    assert all(case.content_review_status == "pending" for case in cases)

    raw_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="complete crawler artifact"):
        import_regulatory_crawler_artifact(
            source,
            tmp_path,
            source_dataset="governed-cases",
            max_records=10,
        )


def test_regulatory_crawler_import_rejects_missing_governance_and_caller_laundering(tmp_path: Path) -> None:
    source = Source(
        url="https://example.gov.cn/cases.html",
        allowed_domains=("example.gov.cn",),
        source_type="public_case",
        license="RIGHTS-REVIEW-REQUIRED",
        license_review_status="pending",
        terms_review_status="pending",
        content_review_status="pending",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=REGULATORY_HTML.encode("utf-8"))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetch(source, tmp_path, client=client, retrieved_at=RETRIEVED_AT)

    with pytest.raises(ValueError, match="does not match hash-bound"):
        import_regulatory_crawler_artifact(
            source,
            tmp_path,
            source_dataset="cases",
            license_review_status="approved",
            max_records=10,
        )

    metadata_path = metadata_path_for(source, tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("content_review_status")
    metadata_bytes = (
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    metadata_path.write_bytes(metadata_bytes)
    marker_path = completion_path_for(source, tmp_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["metadata_sha256"] = hashlib.sha256(metadata_bytes).hexdigest()
    marker_path.write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a complete crawler artifact"):
        import_regulatory_crawler_artifact(
            source,
            tmp_path,
            source_dataset="cases",
            max_records=10,
        )


def test_regulatory_crawler_import_consumes_raw_artifact_once(tmp_path: Path, monkeypatch) -> None:
    source = Source(
        url="https://example.gov.cn/cases.html",
        allowed_domains=("example.gov.cn",),
        source_type="public_case",
        license="RIGHTS-REVIEW-REQUIRED",
        license_review_status="pending",
        terms_review_status="pending",
        content_review_status="pending",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=REGULATORY_HTML.encode("utf-8"))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        raw_path = fetch(source, tmp_path, client=client, retrieved_at=RETRIEVED_AT)

    from risk_agent import crawler

    original_open = crawler.os.open
    raw_open_count = 0

    def counted_open(path, *args, **kwargs):
        nonlocal raw_open_count
        is_direct_open = Path(path) == raw_path
        is_posix_openat = kwargs.get("dir_fd") is not None and Path(path) == Path(raw_path.name)
        if is_direct_open or is_posix_openat:
            raw_open_count += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(crawler.os, "open", counted_open)
    cases = import_regulatory_crawler_artifact(
        source,
        tmp_path,
        source_dataset="cases",
        max_records=10,
    )

    assert len(cases) == 2
    assert raw_open_count == 1


def test_regulatory_crawler_import_rejects_placeholder_governance_values(tmp_path: Path) -> None:
    source = Source(
        url="https://example.gov.cn/cases.html",
        allowed_domains=("example.gov.cn",),
        source_type="unknown",
        license="RIGHTS-REVIEW-REQUIRED",
        license_review_status="approved",
        terms_review_status="approved",
        content_review_status="approved",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=REGULATORY_HTML.encode("utf-8"))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetch(source, tmp_path, client=client, retrieved_at=RETRIEVED_AT)

    with pytest.raises(ValueError, match="source_type"):
        import_regulatory_crawler_artifact(
            source,
            tmp_path,
            source_dataset="cases",
            max_records=10,
        )


def _write_import_config(path: Path, repo: Path, html_path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "max_records": 20,
                "max_file_bytes": 1048576,
                "max_total_bytes": 8388608,
                "mm_safety_bench": {
                    "repo_root": str(repo),
                    "use_tiny": True,
                    "allow_missing_media": False,
                    "scenario_allowlist": ["01-Illegal_Activitiy"],
                    "source_url": MM_SOURCE_URL,
                    "retrieved_at": RETRIEVED_AT,
                    "license_id": "CC-BY-NC-4.0",
                    "license_restrictions": [
                        "CC-BY-NC-4.0 non-commercial research use only",
                        "Upstream GPT-4 license restrictions apply",
                        "Upstream Stable Diffusion license restrictions apply",
                    ],
                    "usage_scope": "smoke_only",
                },
                "regulatory_cases": [
                    {
                        "input_type": "local_html",
                        "local_html_path": str(html_path),
                        "source_dataset": "SAMR-public-cases",
                        "source_url": "https://www.samr.gov.cn/example/cases.html",
                        "allowed_domains": ["www.samr.gov.cn"],
                        "retrieved_at": RETRIEVED_AT,
                        "license_id": "RIGHTS-REVIEW-REQUIRED",
                        "usage_scope": "research_only",
                        "license_review_status": "review_required",
                        "terms_review_status": "review_required",
                        "content_review_status": "pending",
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
    assert report["licenses"][0]["license_restrictions"]

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
    output.mkdir()
    with pytest.raises(FileExistsError, match="destination already exists"):
        run_public_data_import(config, output)
    output.rmdir()
    run_public_data_import(config, output)
    with pytest.raises(FileExistsError, match="already exists"):
        run_public_data_import(config, output)


def test_public_data_bundle_publishes_completion_manifest_last(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)
    html_path = tmp_path / "samr.html"
    html_path.write_text(REGULATORY_HTML, encoding="utf-8")
    config = tmp_path / "public-sources.yaml"
    _write_import_config(config, repo, html_path)
    output = tmp_path / "normalized"

    from risk_agent import public_data

    original_replace = public_data.os.replace
    published: list[str] = []

    def record_replace(source, destination, **kwargs):
        published.append(Path(destination).name)
        return original_replace(source, destination, **kwargs)

    monkeypatch.setattr(public_data.os, "replace", record_replace)
    run_public_data_import(config, output)

    assert (output / "import_manifest.json").is_file()
    assert published[-1] == "import_manifest.json"


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


def test_public_data_import_applies_global_record_cap_incrementally(tmp_path: Path) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)
    html_path = tmp_path / "samr.html"
    html_path.write_text(REGULATORY_HTML, encoding="utf-8")
    config = tmp_path / "public-sources.yaml"
    _write_import_config(config, repo, html_path)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["max_records"] = 4
    config.write_text(yaml.safe_dump(document), encoding="utf-8")

    report = run_public_data_import(config, tmp_path / "normalized")

    assert report["public_asset_count"] == 3
    assert report["sanitized_case_count"] == 1
    assert report["truncated"] is True
    assert report["truncated_sources"] == ["SAMR-public-cases"]
    manifest = json.loads(
        (tmp_path / "normalized" / "import_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["truncation"]["truncated"] is True


def test_mm_only_bundle_reports_mm_safety_bench_truncation(tmp_path: Path) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)
    config = tmp_path / "public-sources.yaml"
    document = {
        "version": 2,
        "max_records": 1,
        "max_file_bytes": 1048576,
        "max_total_bytes": 8388608,
        "mm_safety_bench": {
            "repo_root": str(repo),
            "use_tiny": True,
            "allow_missing_media": False,
            "scenario_allowlist": ["01-Illegal_Activitiy"],
            "source_url": MM_SOURCE_URL,
            "retrieved_at": RETRIEVED_AT,
            "license_id": "CC-BY-NC-4.0",
            "license_restrictions": list(MM_SAFETY_BENCH_RESTRICTIONS),
            "usage_scope": "smoke_only",
        },
        "regulatory_cases": [],
    }
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    output = tmp_path / "normalized"

    report = run_public_data_import(config, output)

    assert report["public_asset_count"] == 1
    assert report["truncated"] is True
    assert report["truncated_sources"] == ["MM-SafetyBench"]
    manifest = json.loads((output / "import_manifest.json").read_text(encoding="utf-8"))
    assert manifest["truncation"]["truncated_sources"] == ["MM-SafetyBench"]


def test_public_data_import_rejects_duplicate_yaml_keys_and_unknown_source_fields(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("version: 1\nmax_records: 1\nmax_records: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        run_public_data_import(duplicate, tmp_path / "duplicate-output")

    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)
    config = tmp_path / "unknown.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "max_records": 3,
                "max_file_bytes": 1024,
                "max_total_bytes": 8192,
                "mm_safety_bench": {
                    "repo_root": str(repo),
                    "source_url": MM_SOURCE_URL,
                    "retrieved_at": RETRIEVED_AT,
                    "license_id": "CC-BY-NC-4.0",
                    "usage_scope": "smoke_only",
                    "unexpected": "must fail closed",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown.*MM-SafetyBench"):
        run_public_data_import(config, tmp_path / "unknown-output")


def test_public_data_config_v1_has_explicit_migration_error(tmp_path: Path) -> None:
    config = tmp_path / "v1.yaml"
    config.write_text(
        "version: 1\nmax_records: 1\nmax_file_bytes: 100\nmax_total_bytes: 100\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="version 1.*migrate.*version 2"):
        run_public_data_import(config, tmp_path / "out")


def test_all_regulatory_rows_use_discriminated_schema_before_record_cap(tmp_path: Path) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)
    html_path = tmp_path / "samr.html"
    html_path.write_text(REGULATORY_HTML, encoding="utf-8")
    config = tmp_path / "public-sources.yaml"
    _write_import_config(config, repo, html_path)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["max_records"] = 3
    invalid_row = dict(document["regulatory_cases"][0])
    invalid_row["crawler_output_dir"] = "irrelevant-for-local-html"
    document["regulatory_cases"].append(invalid_row)
    config.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError, match="not allowed for local_html"):
        run_public_data_import(config, tmp_path / "out")


def test_output_reservation_race_never_deletes_competing_directory(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)
    html_path = tmp_path / "samr.html"
    html_path.write_text(REGULATORY_HTML, encoding="utf-8")
    config = tmp_path / "public-sources.yaml"
    _write_import_config(config, repo, html_path)
    output = tmp_path / "normalized"
    original_mkdir = __import__("os").mkdir

    def racing_mkdir(path, *args, **kwargs):
        if Path(path) == output:
            original_mkdir(path)
            (output / "competitor.txt").write_text("keep", encoding="utf-8")
            raise FileExistsError("competitor won")
        return original_mkdir(path, *args, **kwargs)

    from risk_agent import public_data

    monkeypatch.setattr(public_data.os, "mkdir", racing_mkdir)
    with pytest.raises(FileExistsError, match="competitor won"):
        run_public_data_import(config, output)

    assert (output / "competitor.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX dir_fd rename semantics")
def test_output_fd_never_writes_or_deletes_path_swapped_after_identity_check(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "MM-SafetyBench"
    _write_mm_fixture(repo)
    html_path = tmp_path / "samr.html"
    html_path.write_text(REGULATORY_HTML, encoding="utf-8")
    config = tmp_path / "public-sources.yaml"
    _write_import_config(config, repo, html_path)
    output = tmp_path / "normalized"
    displaced = tmp_path / "owned-reservation"

    from risk_agent import public_data

    original_samestat = public_data.os.path.samestat
    swapped = False

    def swap_after_samestat(first, second):
        nonlocal swapped
        result = original_samestat(first, second)
        if result and not swapped:
            output.rename(displaced)
            output.mkdir()
            (output / "competitor.txt").write_text("keep", encoding="utf-8")
            swapped = True
        return result

    monkeypatch.setattr(public_data.os.path, "samestat", swap_after_samestat)
    with pytest.raises(RuntimeError, match="reservation path identity changed"):
        run_public_data_import(config, output)

    assert (output / "competitor.txt").read_text(encoding="utf-8") == "keep"
    assert sorted(path.name for path in output.iterdir()) == ["competitor.txt"]


def test_public_data_cli_reports_sanitized_validation_error_without_traceback(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    secret = "secret-query-value"
    config.write_text(
        f"version: 1\nmax_records: 0\nsource_url: https://example.test/?token={secret}\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/import_public_data.py", str(config), str(tmp_path / "out")],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr
    assert secret not in result.stderr
