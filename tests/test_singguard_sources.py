"""Governed open-seed adapters for SingGuard synthesis."""

import csv
import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest


RETRIEVED_AT = "2026-07-14T00:00:00Z"


def _zip_bytes(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _youtube_csv() -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=("COMMENT_ID", "AUTHOR", "DATE", "CONTENT", "CLASS"),
    )
    writer.writeheader()
    writer.writerow(
        {
            "COMMENT_ID": "yt-1",
            "AUTHOR": "creator",
            "DATE": "2026-01-01",
            "CONTENT": "Message my private account for the offer.",
            "CLASS": "1",
        }
    )
    writer.writerow(
        {
            "COMMENT_ID": "yt-1",
            "AUTHOR": "second-creator",
            "DATE": "2026-01-02",
            "CONTENT": "A second comment that reuses the upstream identifier.",
            "CLASS": "0",
        }
    )
    return buffer.getvalue()


def test_catalog_exposes_only_enabled_commercially_identified_adapters() -> None:
    from risk_agent.singguard_sources import load_source_catalog

    catalog = load_source_catalog(
        Path(__file__).parents[1] / "configs" / "singguard_sources.yaml"
    )

    enabled = {source.source for source in catalog if source.enabled}
    assert enabled == {"uci_sms_spam", "uci_youtube_spam"}
    assert all(source.usage_scope == "research_only" for source in catalog)
    assert all(source.license for source in catalog)


def test_fetch_normalizes_sms_and_youtube_archives_atomically(tmp_path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    sms = _zip_bytes(
        {
            "SMSSpamCollection": (
                "spam\tText +44 7700 900123 or winner@example.com for a prize.\n"
                "ham\tSee you tomorrow.\n"
            )
        }
    )
    youtube = _zip_bytes(
        {
            "Youtube01-Psy.csv": _youtube_csv(),
            "Youtube02-KatyPerry.csv": _youtube_csv(),
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = sms if "228" in str(request.url) else youtube
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        manifest = fetch_configured_seeds(
            Path(__file__).parents[1] / "configs" / "singguard_sources.yaml",
            tmp_path / "seeds",
            client=client,
            retrieved_at=RETRIEVED_AT,
        )

    rows = [
        json.loads(line)
        for line in (tmp_path / "seeds" / "seeds.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(rows) == 6
    assert {row["source"] for row in rows} == {"uci_sms_spam", "uci_youtube_spam"}
    assert all(row["license"] == "CC-BY-4.0" for row in rows)
    assert all(row["usage_scope"] == "research_only" for row in rows)
    assert all(row["retrieved_at"] == RETRIEVED_AT for row in rows)
    assert all(len(row["content_hash"]) == 64 for row in rows)
    sms_spam = next(
        row
        for row in rows
        if row["source"] == "uci_sms_spam" and row["source_label"] == "spam"
    )
    assert "[PHONE]" in sms_spam["text"]
    assert "[EMAIL]" in sms_spam["text"]
    assert "7700" not in sms_spam["text"]
    assert len({(row["source"], row["source_id"]) for row in rows}) == 6
    assert manifest["record_count"] == 6
    assert manifest["source_counts"] == {"uci_sms_spam": 2, "uci_youtube_spam": 4}
    assert manifest["usage_scope"] == "research_only"
    assert manifest["seeds_sha256"]


def test_fetch_rejects_zip_path_traversal(tmp_path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    malicious = _zip_bytes({"../SMSSpamCollection": "spam\tbad\n"})

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=malicious)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="unsafe ZIP member"):
            fetch_configured_seeds(
                Path(__file__).parents[1] / "configs" / "singguard_sources.yaml",
                tmp_path / "seeds",
                client=client,
                retrieved_at=RETRIEVED_AT,
            )

    assert not (tmp_path / "seeds").exists()


def test_enabled_unknown_license_is_rejected_before_network_call(tmp_path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    config = tmp_path / "sources.yaml"
    config.write_text(
        """version: singguard-sources-v1
sources:
  - source: unknown
    dataset_url: https://example.com/dataset
    artifact_url: https://example.com/data.zip
    license: UNKNOWN
    usage_scope: research_only
    source_role: style_seed
    adapter: uci_sms_zip
    enabled: true
""",
        encoding="utf-8",
    )
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, content=b"")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="allowlisted license"):
            fetch_configured_seeds(
                config,
                tmp_path / "seeds",
                client=client,
                retrieved_at=RETRIEVED_AT,
            )

    assert called is False
