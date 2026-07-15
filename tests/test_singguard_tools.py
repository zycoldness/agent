"""Deterministic local tools for SingGuard teacher trajectories."""

from __future__ import annotations

import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _tool_environment(tmp_path: Path) -> Path:
    _write_jsonl(
        tmp_path / "cases.jsonl",
        [
            {
                "case_id": "case-1",
                "text": "Guaranteed weight loss in seven days",
                "summary": "A fixed weight-loss result was guaranteed.",
                "source_type": "public_regulatory_case",
                "source_id": "source-1",
            },
            {
                "case_id": "case-2",
                "text": "Official support order status",
                "summary": "Support discussed an existing order.",
                "source_type": "synthetic_fixture",
                "source_id": "source-2",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "claim_evidence.jsonl",
        [
            {
                "evidence_id": "evidence-1",
                "claim": "lose ten pounds in seven days",
                "product_type": "weight-loss supplement",
                "stance": "unsupported",
                "summary": "No accepted source supports a guaranteed fixed result.",
                "source_type": "public_guidance",
                "source_id": "guidance-1",
            },
            {
                "evidence_id": "evidence-2",
                "claim": "reduces charging time by ten percent",
                "product_type": "charging accessory",
                "stance": "insufficient",
                "summary": "An incomplete charging test.",
                "source_type": "synthetic_fixture",
                "source_id": "guidance-2",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "destinations.jsonl",
        [
            {
                "destination_id": "destination-1",
                "indicators": ["whatsapp:user123"],
                "destination_type": "off_platform_messaging",
                "risk_signals": ["obfuscated platform name"],
                "source_type": "synthetic_fixture",
                "source_id": "destination-source-1",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "content_history.jsonl",
        [
            {
                "content_id": "content-1",
                "recent_contents": ["DM me for a private price."],
                "account_signals": ["repeated off-platform solicitation"],
                "source_type": "synthetic_fixture",
                "source_id": "history-source-1",
            }
        ],
    )
    return tmp_path


def test_search_cases_returns_compact_deterministic_json(tmp_path: Path) -> None:
    from risk_agent.singguard_tools import ToolCall, ToolEnvironment

    environment = ToolEnvironment.load(_tool_environment(tmp_path))
    call = ToolCall(
        name="search_cases",
        arguments={"query": "guaranteed weight loss", "top_k": 1},
    )

    first = environment.execute(call)
    second = environment.execute(call)

    assert first == second
    assert first.status == "ok"
    assert json.loads(first.to_content()) == {
        "status": "ok",
        "results": [
            {
                "case_id": "case-1",
                "summary": "A fixed weight-loss result was guaranteed.",
                "source_id": "source-1",
                "source_type": "public_regulatory_case",
            }
        ],
    }


def test_invalid_tool_arguments_return_small_error_response(tmp_path: Path) -> None:
    from risk_agent.singguard_tools import ToolCall, ToolEnvironment

    environment = ToolEnvironment.load(_tool_environment(tmp_path))

    result = environment.execute(
        ToolCall(name="search_cases", arguments={"query": "claim", "top_k": 99})
    )

    assert json.loads(result.to_content()) == {
        "status": "error",
        "error": "invalid_arguments",
    }


def test_destination_lookup_normalizes_obfuscated_platform_name(tmp_path: Path) -> None:
    from risk_agent.singguard_tools import ToolCall, ToolEnvironment

    environment = ToolEnvironment.load(_tool_environment(tmp_path))

    result = environment.execute(
        ToolCall(
            name="inspect_destination",
            arguments={"indicator": "w-h-a-t-s-a-p-p : user123"},
        )
    )

    assert result.status == "ok"
    assert result.payload["result"]["destination_id"] == "destination-1"


def test_destination_lookup_reports_not_found_for_changed_identifier(tmp_path: Path) -> None:
    from risk_agent.singguard_tools import ToolCall, ToolEnvironment

    environment = ToolEnvironment.load(_tool_environment(tmp_path))

    result = environment.execute(
        ToolCall(
            name="inspect_destination",
            arguments={"indicator": "w-h-a-t-a-p-p:user123"},
        )
    )

    assert json.loads(result.to_content()) == {
        "status": "not_found",
        "result": None,
    }


def test_empty_search_result_reports_not_found(tmp_path: Path) -> None:
    from risk_agent.singguard_tools import ToolCall, ToolEnvironment

    environment = ToolEnvironment.load(_tool_environment(tmp_path))
    result = environment.execute(
        ToolCall(name="search_cases", arguments={"query": "unmatched-zebra-token"})
    )

    assert result.status == "not_found"
    assert result.payload == {"results": []}


def test_tool_environment_rejects_duplicate_normalized_indicators(tmp_path: Path) -> None:
    import pytest

    from risk_agent.singguard_tools import ToolEnvironment

    root = _tool_environment(tmp_path)
    with (root / "destinations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "destination_id": "destination-2",
                    "indicators": ["w-h-a-t-s-a-p-p:user123"],
                    "destination_type": "different_destination",
                    "risk_signals": [],
                    "source_type": "synthetic_fixture",
                    "source_id": "destination-source-2",
                }
            )
            + "\n"
        )

    with pytest.raises(ValueError, match="duplicate normalized destination indicator"):
        ToolEnvironment.load(root)


def test_tool_environment_rejects_duplicate_record_ids(tmp_path: Path) -> None:
    import pytest

    from risk_agent.singguard_tools import ToolEnvironment

    root = _tool_environment(tmp_path)
    with (root / "content_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "content_id": "content-1",
                    "recent_contents": ["Conflicting duplicate."],
                    "account_signals": [],
                    "source_type": "synthetic_fixture",
                    "source_id": "history-source-duplicate",
                }
            )
            + "\n"
        )

    with pytest.raises(ValueError, match="duplicate content IDs"):
        ToolEnvironment.load(root)


def test_tool_environment_rejects_indicator_without_searchable_content(
    tmp_path: Path,
) -> None:
    import pytest

    from risk_agent.singguard_tools import ToolEnvironment

    root = _tool_environment(tmp_path)
    destinations = [
        {
            "destination_id": "destination-empty",
            "indicators": ["---"],
            "destination_type": "unknown",
            "risk_signals": [],
            "source_type": "synthetic_fixture",
            "source_id": "destination-source-empty",
        }
    ]
    _write_jsonl(root / "destinations.jsonl", destinations)

    with pytest.raises(ValueError, match="empty normalized destination indicator"):
        ToolEnvironment.load(root)


def test_tools_json_uses_ms_swift_function_schema() -> None:
    from risk_agent.singguard_tools import tools_json

    payload = json.loads(tools_json(("search_cases", "inspect_destination")))

    assert [item["function"]["name"] for item in payload] == [
        "search_cases",
        "inspect_destination",
    ]
    assert all(item["type"] == "function" for item in payload)


def test_verify_claim_drops_records_with_only_one_shared_token(tmp_path: Path) -> None:
    from risk_agent.singguard_tools import ToolCall, ToolEnvironment

    environment = ToolEnvironment.load(_tool_environment(tmp_path))
    result = environment.execute(
        ToolCall(
            name="verify_claim",
            arguments={
                "claim": "lose ten pounds in seven days",
                "product_type": "weight-loss supplement",
            },
        )
    )

    assert [item["evidence_id"] for item in result.payload["results"]] == [
        "evidence-1"
    ]
