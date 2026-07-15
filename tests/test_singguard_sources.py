"""Governed open-seed adapters for SingGuard synthesis."""

import builtins
import csv
import hashlib
import io
import json
import math
import sys
import types
import zipfile
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError


RETRIEVED_AT = "2026-07-14T00:00:00Z"
CATALOG = Path(__file__).parents[1] / "configs" / "singguard_sources.yaml"


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


def _read_rows(output_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (output_dir / "seeds.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _write_catalog(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "sources.yaml"
    path.write_text(
        "version: singguard-sources-v1\nsources:\n" + source,
        encoding="utf-8",
    )
    return path


def _external_spec(**updates: object) -> dict[str, object]:
    spec: dict[str, object] = {
        "source": "external",
        "dataset_url": "https://example.com/dataset",
        "artifact_url": "https://example.com/artifact",
        "license": "CC-BY-4.0",
        "usage_scope": "research_only",
        "source_role": "hard_negative",
        "adapter": "hf_aegis_v2",
        "enabled": True,
        "dataset_id": "org/data",
        "revision": "a" * 40,
        "split": "train",
        "max_records": 10,
    }
    spec.update(updates)
    return spec


def test_catalog_enables_exactly_five_pinned_permissive_sources() -> None:
    from risk_agent.singguard_sources import load_source_catalog

    catalog = load_source_catalog(CATALOG)
    by_name = {source.source: source for source in catalog}

    assert {source.source for source in catalog if source.enabled} == {
        "uci_sms_spam",
        "uci_youtube_spam",
        "nemotron_aegis_v2",
        "civil_comments",
        "amazon_esci",
    }
    aegis = by_name["nemotron_aegis_v2"]
    assert aegis.dataset_id == "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
    assert aegis.revision == "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"
    assert aegis.split == "train"
    assert aegis.max_records == 2000
    assert aegis.license == "CC-BY-4.0"
    assert aegis.adapter == "hf_aegis_v2"
    assert by_name["civil_comments"].dataset_id == "google/civil_comments"
    assert by_name["civil_comments"].revision == "f2970eb3a55777454c94069077cc8d9b5866312d"
    assert by_name["civil_comments"].split == "train"
    assert by_name["civil_comments"].max_records == 2000
    assert by_name["civil_comments"].license == "CC0-1.0"
    esci = by_name["amazon_esci"]
    assert esci.dataset_id == "parquet"
    assert esci.revision == "7916cdf6ab75a462e77f20ab40428a10923998d5"
    assert esci.split == "train"
    assert esci.max_records == 1000
    assert esci.license == "Apache-2.0"
    assert esci.artifact_url == (
        "https://media.githubusercontent.com/media/amazon-science/esci-data/"
        "7916cdf6ab75a462e77f20ab40428a10923998d5/shopping_queries_dataset/"
        "shopping_queries_dataset_examples.parquet"
    )
    assert by_name["wildguardmix"].enabled is False
    assert by_name["toxic_chat"].enabled is False
    assert by_name["beavertails"].enabled is False
    assert by_name["ftc_fda_public_records"].enabled is False


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
            CATALOG,
            tmp_path / "seeds",
            client=client,
            retrieved_at=RETRIEVED_AT,
            enabled_sources=("uci_sms_spam", "uci_youtube_spam"),
        )

    rows = _read_rows(tmp_path / "seeds")
    assert len(rows) == 4
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
    assert len({(row["source"], row["source_id"]) for row in rows}) == 4
    assert manifest["source_counts"] == {"uci_sms_spam": 2, "uci_youtube_spam": 2}
    assert manifest["source_snapshots"] == [
        {
            "adapter": "uci_sms_zip",
            "artifact_url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
            "dataset_id": None,
            "license": "CC-BY-4.0",
            "max_records": 5000,
            "revision": None,
            "source": "uci_sms_spam",
            "split": None,
        },
        {
            "adapter": "uci_youtube_zip",
            "artifact_url": "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip",
            "dataset_id": None,
            "license": "CC-BY-4.0",
            "max_records": 5000,
            "revision": None,
            "source": "uci_youtube_spam",
            "split": None,
        },
    ]
    assert manifest["seeds_sha256"]


def test_injected_aegis_normalizes_response_and_skips_redacted(tmp_path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    rows = [
        {
            "id": "a-1",
            "prompt": "  Explain   photosynthesis ",
            "response": None,
            "prompt_label": " safe ",
            "response_label": "unsafe",
            "violated_categories": ["irrelevant"],
        },
        {
            "id": "a-redacted",
            "prompt": " REDACTED ",
            "response": "must not appear",
            "prompt_label": "unsafe",
        },
        {
            "id": 42,
            "prompt": "Question",
            "response": "  Answer  text ",
            "prompt_label": " Unsafe ",
            "response_label": "safe",
        },
    ]

    manifest = fetch_configured_seeds(
        CATALOG,
        tmp_path / "seeds",
        retrieved_at=RETRIEVED_AT,
        dataset_loader=lambda spec: rows,
        enabled_sources=("nemotron_aegis_v2",),
    )

    normalized = _read_rows(tmp_path / "seeds")
    assert [(row["source_id"], row["text"], row["source_label"]) for row in normalized] == [
        ("a-1", "Explain photosynthesis", "safe"),
        (
            "42",
            '{"query":"Question","response":"Answer text"}',
            "unsafe",
        ),
    ]
    assert manifest["record_count"] == 2
    assert all(set(row) == {
        "adapter_version", "content_hash", "license", "provenance_url", "retrieved_at",
        "source", "source_id", "source_label", "source_role", "text", "usage_scope"
    } for row in normalized)


def test_aegis_response_pii_redaction_preserves_valid_json(tmp_path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    fetch_configured_seeds(
        CATALOG,
        tmp_path / "seeds",
        retrieved_at=RETRIEVED_AT,
        dataset_loader=lambda spec: [
            {
                "id": "pii",
                "prompt": "Email me at user@example.com",
                "response": "Open https://example.com/private",
                "prompt_label": "safe",
            }
        ],
        enabled_sources=("nemotron_aegis_v2",),
    )

    [row] = _read_rows(tmp_path / "seeds")
    assert json.loads(row["text"]) == {
        "query": "Email me at [EMAIL]",
        "response": "Open [URL]",
    }


def test_civil_comments_uses_coarse_label_and_stable_id_or_input_ordinal(tmp_path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    upstream = [
        {"id": "cc-1", "text": "First", "toxicity": 0.4999, "identity_attack": 1.0},
        {"id": "  ", "text": "Boundary", "toxicity": 0.5, "severe_toxicity": 0.9},
        {"text": "Last", "toxicity": 1, "obscene": 0.8},
    ]
    fetch_configured_seeds(
        CATALOG,
        tmp_path / "seeds",
        retrieved_at=RETRIEVED_AT,
        dataset_loader=lambda spec: upstream,
        enabled_sources=("civil_comments",),
    )

    rows = _read_rows(tmp_path / "seeds")
    assert [(row["source_id"], row["source_label"]) for row in rows] == [
        ("cc-1", "non_toxic"),
        ("civil-row-000000002", "toxic"),
        ("civil-row-000000003", "toxic"),
    ]
    assert all("identity_attack" not in row and "severe_toxicity" not in row for row in rows)


@pytest.mark.parametrize("toxicity", [math.nan, math.inf, -0.01, 1.01, True, "0.5"])
def test_civil_comments_rejects_invalid_toxicity(tmp_path, toxicity: object) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    with pytest.raises(ValueError, match="toxicity"):
        fetch_configured_seeds(
            CATALOG,
            tmp_path / "seeds",
            retrieved_at=RETRIEVED_AT,
            dataset_loader=lambda spec: [{"text": "row", "toxicity": toxicity}],
            enabled_sources=("civil_comments",),
        )
    assert not (tmp_path / "seeds").exists()


def test_esci_filters_and_deduplicates_canonical_queries(tmp_path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    upstream = [
        {"query": "Ｃａｆｅ   Chair", "product_locale": "US", "split": "TRAIN", "esci_label": "E"},
        {"query": "cafe chair", "product_locale": "us", "split": "train", "esci_label": "I"},
        {"query": "Ignored locale", "product_locale": "es", "split": "train"},
        {"query": "Ignored split", "product_locale": "us", "split": "test"},
        {"query": "Desk lamp", "product_locale": " us ", "split": " train ", "esci_label": "C"},
    ]
    fetch_configured_seeds(
        CATALOG,
        tmp_path / "seeds",
        retrieved_at=RETRIEVED_AT,
        dataset_loader=lambda spec: upstream,
        enabled_sources=("amazon_esci",),
    )

    rows = _read_rows(tmp_path / "seeds")
    canonical = "cafe chair"
    assert [(row["source_id"], row["text"], row["source_label"]) for row in rows] == [
        (
            f"esci-query-{hashlib.sha256(canonical.encode()).hexdigest()[:16]}",
            "Ｃａｆｅ Chair",
            None,
        ),
        (
            f"esci-query-{hashlib.sha256(b'desk lamp').hexdigest()[:16]}",
            "Desk lamp",
            None,
        ),
    ]
    assert all("esci_label" not in row for row in rows)


@pytest.mark.parametrize(
    ("source_name", "rows"),
    [
        ("nemotron_aegis_v2", [{"id": [], "prompt": "valid", "prompt_label": "safe"}]),
        ("nemotron_aegis_v2", [{"id": "id", "prompt": {"bad": "shape"}, "prompt_label": "safe"}]),
        ("nemotron_aegis_v2", [{"id": "id", "prompt": "valid", "response": ["bad"]}]),
        ("civil_comments", [{"text": ["bad"], "toxicity": 0.2}]),
        ("amazon_esci", [{"query": {"bad": "shape"}, "product_locale": "us", "split": "train"}]),
    ],
)
def test_external_sources_reject_schema_drift(tmp_path, source_name: str, rows: list[dict[str, object]]) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    with pytest.raises(ValueError, match="row|scalar|text|prompt|response|query|id"):
        fetch_configured_seeds(
            CATALOG,
            tmp_path / "seeds",
            retrieved_at=RETRIEVED_AT,
            dataset_loader=lambda spec: rows,
            enabled_sources=(source_name,),
        )
    assert not (tmp_path / "seeds").exists()


def test_max_records_counts_accepted_rows_without_overconsuming(tmp_path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    catalog = _write_catalog(
        tmp_path,
        """  - source: limited
    dataset_url: https://example.com/aegis
    artifact_url: https://example.com/aegis
    license: CC-BY-4.0
    usage_scope: research_only
    source_role: hard_negative
    adapter: hf_aegis_v2
    enabled: true
    dataset_id: org/aegis
    revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    split: train
    max_records: 2
""",
    )

    def rows():
        yield {"id": "skip", "prompt": "REDACTED", "prompt_label": "unsafe"}
        yield {"id": "one", "prompt": "one", "prompt_label": "safe"}
        yield {"id": "two", "prompt": "two", "prompt_label": "safe"}
        raise AssertionError("loader iterator was over-consumed")

    manifest = fetch_configured_seeds(
        catalog,
        tmp_path / "seeds",
        retrieved_at=RETRIEVED_AT,
        dataset_loader=lambda spec: rows(),
    )
    assert manifest["record_count"] == 2


def test_sms_max_records_stops_before_later_malformed_rows(tmp_path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    catalog = _write_catalog(
        tmp_path,
        """  - source: limited_sms
    dataset_url: https://example.com/sms
    artifact_url: https://example.com/sms.zip
    license: CC-BY-4.0
    usage_scope: research_only
    source_role: style_seed
    adapter: uci_sms_zip
    enabled: true
    max_records: 2
""",
    )
    payload = _zip_bytes(
        {
            "SMSSpamCollection": (
                "ham\tfirst accepted\n"
                "\n"
                "spam\tsecond accepted\n"
                "this malformed row must not be consumed\n"
            )
        }
    )

    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=payload)
        )
    ) as client:
        manifest = fetch_configured_seeds(
            catalog,
            tmp_path / "seeds",
            client=client,
            retrieved_at=RETRIEVED_AT,
        )

    assert manifest["record_count"] == 2
    assert [row["source_id"] for row in _read_rows(tmp_path / "seeds")] == [
        "sms-000001",
        "sms-000003",
    ]


def test_youtube_max_records_spans_members_and_stops_rows_and_members(tmp_path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    catalog = _write_catalog(
        tmp_path,
        """  - source: limited_youtube
    dataset_url: https://example.com/youtube
    artifact_url: https://example.com/youtube.zip
    license: CC-BY-4.0
    usage_scope: research_only
    source_role: style_seed
    adapter: uci_youtube_zip
    enabled: true
    max_records: 2
""",
    )
    payload = _zip_bytes(
        {
            "Youtube01.csv": "COMMENT_ID,CONTENT,CLASS\none,first accepted,0\n",
            "Youtube02.csv": (
                "COMMENT_ID,CONTENT,CLASS\n"
                "two,second accepted,1\n"
                "three,,0\n"
            ),
            "Youtube03.csv": "WRONG,COLUMNS\nnot,consumed\n",
        }
    )

    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=payload)
        )
    ) as client:
        manifest = fetch_configured_seeds(
            catalog,
            tmp_path / "seeds",
            client=client,
            retrieved_at=RETRIEVED_AT,
        )

    assert manifest["record_count"] == 2
    assert [row["text"] for row in _read_rows(tmp_path / "seeds")] == [
        "first accepted",
        "second accepted",
    ]


@pytest.mark.parametrize(
    "updates",
    [
        {"revision": None},
        {"revision": "A" * 40},
        {"revision": "a" * 39},
        {"split": "validation"},
        {"split": " train "},
        {"dataset_id": " "},
        {"adapter": "esci_query_parquet", "dataset_id": "not-parquet"},
        {"adapter": "esci_query_parquet", "dataset_id": "parquet", "artifact_url": "http://example.com/x"},
        {"adapter": "esci_query_parquet", "dataset_id": "parquet", "artifact_url": "https:///x"},
    ],
)
def test_external_source_specs_require_pinned_train_schema(updates: dict[str, object]) -> None:
    from risk_agent.singguard_sources import SourceSpec

    with pytest.raises(ValidationError):
        SourceSpec.model_validate(_external_spec(**updates))


def test_source_spec_rejects_blank_optional_strings_and_bad_max_records() -> None:
    from risk_agent.singguard_sources import SourceSpec

    for updates in ({"revision": " "}, {"split": ""}, {"dataset_id": "\t"}, {"max_records": 0}):
        with pytest.raises(ValidationError):
            SourceSpec.model_validate(_external_spec(**updates))


def test_default_loader_uses_streaming_revision_and_esci_media_file(monkeypatch) -> None:
    from risk_agent.singguard_sources import SourceSpec, _default_dataset_loader

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    sentinel = object()

    def load_dataset(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    aegis = SourceSpec.model_validate(_external_spec())
    esci = SourceSpec.model_validate(
        _external_spec(
            adapter="esci_query_parquet",
            dataset_id="parquet",
            artifact_url="https://example.com/examples.parquet",
        )
    )

    assert _default_dataset_loader(aegis) is sentinel
    assert _default_dataset_loader(esci) is sentinel
    assert calls == [
        (("org/data",), {"split": "train", "revision": "a" * 40, "streaming": True}),
        (
            ("parquet",),
            {
                "data_files": {"train": "https://example.com/examples.parquet"},
                "split": "train",
                "streaming": True,
            },
        ),
    ]


def test_default_loader_has_clear_missing_dependency_error(monkeypatch) -> None:
    from risk_agent.singguard_sources import SourceSpec, _default_dataset_loader

    spec = SourceSpec.model_validate(_external_spec())
    real_import = builtins.__import__

    def missing(name: str, *args: object, **kwargs: object) -> object:
        if name == "datasets":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(RuntimeError, match=r"pip install -e '\.\[sources\]'"):
        _default_dataset_loader(spec)


def test_enabled_sources_rejects_unknown_and_duplicate_before_calls(tmp_path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    called = False

    def loader(spec):
        nonlocal called
        called = True
        return []

    for selection, match in [
        (("missing",), "unknown"),
        (("civil_comments", "civil_comments"), "duplicate"),
    ]:
        with pytest.raises(ValueError, match=match):
            fetch_configured_seeds(
                CATALOG,
                tmp_path / ("out-" + match),
                dataset_loader=loader,
                enabled_sources=selection,
            )
    assert called is False


def test_enabled_sources_rejects_known_disabled_source_before_calls(tmp_path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    calls: list[str] = []
    with pytest.raises(ValueError, match="disabled"):
        fetch_configured_seeds(
            CATALOG,
            tmp_path / "seeds",
            dataset_loader=lambda spec: calls.append("loader") or [],
            enabled_sources=("wildguardmix",),
            retrieved_at=RETRIEVED_AT,
        )

    assert calls == []
    assert not (tmp_path / "seeds").exists()


def test_explicit_selection_calls_only_selected_external_loader(tmp_path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    calls: list[str] = []

    def loader(spec):
        calls.append(spec.source)
        return [{"text": "civil", "toxicity": 0.1}]

    fetch_configured_seeds(
        CATALOG,
        tmp_path / "seeds",
        dataset_loader=loader,
        enabled_sources=("civil_comments",),
        retrieved_at=RETRIEVED_AT,
    )
    assert calls == ["civil_comments"]


def test_license_is_rejected_before_network_or_loader_call(tmp_path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    config = _write_catalog(
        tmp_path,
        """  - source: unknown
    dataset_url: https://example.com/dataset
    artifact_url: https://example.com/data.zip
    license: UNKNOWN
    usage_scope: research_only
    source_role: style_seed
    adapter: uci_sms_zip
    enabled: true
""",
    )
    calls: list[str] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls.append("network")
        return httpx.Response(200, content=b"")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="allowlisted license"):
            fetch_configured_seeds(
                config,
                tmp_path / "seeds",
                client=client,
                dataset_loader=lambda spec: calls.append("loader") or [],
                retrieved_at=RETRIEVED_AT,
            )
    assert calls == []


def test_cross_source_content_is_deduplicated_first_and_manifest_is_deterministic(tmp_path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    def loader(spec):
        if spec.source == "nemotron_aegis_v2":
            return [{"id": "a", "prompt": "Same   content", "prompt_label": "safe"}]
        return [
            {"id": "c", "text": "Same content", "toxicity": 0.2},
            {"id": "c2", "text": "Different", "toxicity": 0.5},
        ]

    manifests = []
    rows_by_run = []
    for name in ("first", "second"):
        output = tmp_path / name
        manifests.append(
            fetch_configured_seeds(
                CATALOG,
                output,
                retrieved_at=RETRIEVED_AT,
                dataset_loader=loader,
                enabled_sources=("nemotron_aegis_v2", "civil_comments"),
            )
        )
        rows_by_run.append(_read_rows(output))

    assert rows_by_run[0] == rows_by_run[1]
    assert manifests[0] == manifests[1]
    assert [(row["source"], row["source_id"]) for row in rows_by_run[0]] == [
        ("nemotron_aegis_v2", "a"),
        ("civil_comments", "c2"),
    ]
    assert len({row["content_hash"] for row in rows_by_run[0]}) == 2
    assert len({(row["source"], row["source_id"]) for row in rows_by_run[0]}) == 2
    snapshots = manifests[0]["source_snapshots"]
    assert snapshots[0]["source"] == "nemotron_aegis_v2"
    assert set(snapshots[0]) == {
        "source", "adapter", "dataset_id", "revision", "split", "artifact_url",
        "max_records", "license"
    }
    assert "Same content" not in json.dumps(manifests[0])


def test_fetch_rejects_duplicate_source_ids_and_leaves_output_absent(tmp_path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    with pytest.raises(ValueError, match="duplicate source IDs"):
        fetch_configured_seeds(
            CATALOG,
            tmp_path / "seeds",
            retrieved_at=RETRIEVED_AT,
            dataset_loader=lambda spec: [
                {"id": "same", "prompt": "one", "prompt_label": "safe"},
                {"id": "same", "prompt": "two", "prompt_label": "safe"},
            ],
            enabled_sources=("nemotron_aegis_v2",),
        )
    assert not (tmp_path / "seeds").exists()


def test_fetch_rejects_zip_path_traversal(tmp_path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    malicious = _zip_bytes({"../SMSSpamCollection": "spam\tbad\n"})

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=malicious)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="unsafe ZIP member"):
            fetch_configured_seeds(
                CATALOG,
                tmp_path / "seeds",
                client=client,
                retrieved_at=RETRIEVED_AT,
                enabled_sources=("uci_sms_spam",),
            )
    assert not (tmp_path / "seeds").exists()
