import pytest

from risk_agent.contracts import Evidence
from risk_agent.stores import CaseStore, EvidenceStore


def test_case_store_discards_oracle_fields_from_search_results() -> None:
    store = CaseStore(
        [
            {
                "case_id": "case-1",
                "text": "七天瘦十斤",
                "label": "unsafe",
                "rule_id": "R-WEIGHT",
                "oracle": {"label": "unsafe"},
                "evidence_labels": ["OCR-1"],
                "extra": "hidden",
            }
        ]
    )

    results = store.search("瘦十斤", top_k=1)

    assert results == [{"case_id": "case-1", "text": "七天瘦十斤"}]
    assert "label" not in results[0]
    assert "rule_id" not in results[0]
    assert "oracle" not in str(results[0])
    assert "unsafe" not in str(results[0])


def test_case_store_returns_no_rows_when_top_k_is_zero() -> None:
    store = CaseStore([{"case_id": "case-1", "text": "七天瘦十斤"}])

    assert store.search("瘦十斤", top_k=0) == []


def test_case_store_search_order_is_independent_of_input_order() -> None:
    rows = [
        {"case_id": "case-b", "text": "claim promotion"},
        {"case_id": "case-a", "text": "claim promotion"},
    ]

    forward = CaseStore(rows).search("claim", top_k=2)
    reversed_rows = CaseStore(reversed(rows)).search("claim", top_k=2)

    assert forward == reversed_rows == [
        {"case_id": "case-a", "text": "claim promotion"},
        {"case_id": "case-b", "text": "claim promotion"},
    ]


def test_case_store_rejects_duplicate_case_ids() -> None:
    with pytest.raises(ValueError, match="duplicate case_id"):
        CaseStore(
            [
                {"case_id": "case-1", "text": "first"},
                {"case_id": "case-1", "text": "second"},
            ]
        )


@pytest.mark.parametrize("query", [None, 1, True])
def test_case_store_rejects_non_string_queries(query: object) -> None:
    store = CaseStore([{"case_id": "case-1", "text": "claim"}])

    with pytest.raises(ValueError, match="query must be a string"):
        store.search(query, top_k=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [True, False, 1.0, "1", None])
def test_case_store_rejects_invalid_top_k(top_k: object) -> None:
    store = CaseStore([{"case_id": "case-1", "text": "claim"}])

    with pytest.raises(ValueError, match="top_k must be an integer"):
        store.search("claim", top_k=top_k)  # type: ignore[arg-type]


def test_evidence_store_filters_by_asset_and_kind_in_insertion_order() -> None:
    store = EvidenceStore(
        [
            Evidence(evidence_id="ocr-1", asset_id="asset-1", kind="ocr", content="文字"),
            Evidence(evidence_id="asr-1", asset_id="asset-1", kind="asr", content="语音"),
            Evidence(evidence_id="ocr-2", asset_id="asset-2", kind="ocr", content="其他"),
        ]
    )

    assert store.inspect("asset-1", {"ocr"}) == [
        Evidence(evidence_id="ocr-1", asset_id="asset-1", kind="ocr", content="文字")
    ]
