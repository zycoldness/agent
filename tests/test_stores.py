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
