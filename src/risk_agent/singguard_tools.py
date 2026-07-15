"""Small deterministic tools used by Gemini teacher trajectories."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ToolCall(BaseModel):
    """Compact sequential tool request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    arguments: dict[str, object]


class ToolResult(BaseModel):
    """Compact deterministic tool response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["ok", "error"]
    payload: dict[str, object]

    def to_content(self) -> str:
        return json.dumps(
            {"status": self.status, **self.payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


class _CaseRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_id: str = Field(min_length=1)


class _ClaimRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    product_type: str = Field(min_length=1)
    stance: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_id: str = Field(min_length=1)


class _DestinationRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    destination_id: str = Field(min_length=1)
    indicators: tuple[str, ...] = Field(min_length=1)
    destination_type: str = Field(min_length=1)
    risk_signals: tuple[str, ...]
    source_type: str = Field(min_length=1)
    source_id: str = Field(min_length=1)


class _HistoryRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    content_id: str = Field(min_length=1)
    recent_contents: tuple[str, ...]
    account_signals: tuple[str, ...]
    source_type: str = Field(min_length=1)
    source_id: str = Field(min_length=1)


class _SearchCasesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=5)


class _VerifyClaimArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1)
    product_type: str | None = None
    top_k: int = Field(default=3, ge=1, le=5)


class _InspectDestinationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indicator: str = Field(min_length=1)


class _ContentContextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_id: str = Field(min_length=1)


def _function(
    name: str,
    description: str,
    properties: dict[str, object],
    required: list[str],
) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SPECS: dict[str, dict[str, object]] = {
    "search_cases": _function(
        "search_cases",
        "Search similar moderation or public regulatory cases. Returns evidence, not a safety label.",
        {
            "query": {"type": "string", "description": "Concise content or behavior to match."},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        ["query"],
    ),
    "verify_claim": _function(
        "verify_claim",
        "Search trusted claim evidence. Returns support status and sources, not a safety label.",
        {
            "claim": {"type": "string", "description": "The exact claim to verify."},
            "product_type": {"type": "string", "description": "Optional product category."},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        ["claim"],
    ),
    "inspect_destination": _function(
        "inspect_destination",
        "Normalize and inspect a link, handle, QR destination, or obfuscated contact indicator.",
        {
            "indicator": {"type": "string", "description": "The destination indicator from the content."},
        },
        ["indicator"],
    ),
    "get_content_context": _function(
        "get_content_context",
        "Return deterministic account and related-content history for one content ID.",
        {
            "content_id": {"type": "string", "description": "Stable content identifier."},
        },
        ["content_id"],
    ),
}


def tools_json(names: tuple[str, ...]) -> str:
    """Serialize allowlisted tools in the ms-swift agent dataset format."""

    if len(names) != len(set(names)):
        raise ValueError("tool names must be unique")
    try:
        payload = [TOOL_SPECS[name] for name in names]
    except KeyError:
        raise ValueError("unknown tool name") from None
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def tool_declarations(names: tuple[str, ...]) -> tuple[dict[str, object], ...]:
    """Return Gemini function declarations without the ms-swift wrappers."""

    return tuple(dict(TOOL_SPECS[name]["function"]) for name in names)


def _read_records(path: Path, model: type[BaseModel]) -> tuple[BaseModel, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        raise ValueError(f"cannot read tool store {path.name}") from None
    records: list[BaseModel] = []
    try:
        for line in lines:
            if line.strip():
                records.append(model.model_validate_json(line))
    except (ValidationError, ValueError):
        raise ValueError(f"tool store {path.name} contains an invalid record") from None
    return tuple(records)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.casefold()))


def _rank(
    query: str,
    records: tuple[BaseModel, ...],
    *,
    text: Any,
    stable_id: Any,
    top_k: int,
    minimum_score: int = 1,
) -> list[BaseModel]:
    query_tokens = _tokens(query)
    scored = [
        (len(query_tokens & _tokens(text(record))), stable_id(record), record)
        for record in records
    ]
    return [
        record
        for score, _identifier, record in sorted(scored, key=lambda item: (-item[0], item[1]))
        if score >= minimum_score
    ][:top_k]


def _normalize_indicator(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


class ToolEnvironment:
    """Validated in-memory snapshot of the local tool stores."""

    def __init__(
        self,
        *,
        cases: tuple[_CaseRecord, ...],
        claims: tuple[_ClaimRecord, ...],
        destinations: tuple[_DestinationRecord, ...],
        histories: tuple[_HistoryRecord, ...],
    ) -> None:
        self._cases = cases
        self._claims = claims
        self._destinations = destinations
        self._histories = histories
        self._destination_index = {
            _normalize_indicator(indicator): record
            for record in destinations
            for indicator in record.indicators
        }
        self._history_index = {record.content_id: record for record in histories}

    @classmethod
    def load(cls, root: Path) -> "ToolEnvironment":
        return cls(
            cases=tuple(_read_records(root / "cases.jsonl", _CaseRecord)),
            claims=tuple(_read_records(root / "claim_evidence.jsonl", _ClaimRecord)),
            destinations=tuple(_read_records(root / "destinations.jsonl", _DestinationRecord)),
            histories=tuple(_read_records(root / "content_history.jsonl", _HistoryRecord)),
        )

    @staticmethod
    def _error(code: str) -> ToolResult:
        return ToolResult(status="error", payload={"error": code})

    def execute(self, call: ToolCall) -> ToolResult:
        try:
            if call.name == "search_cases":
                return self._search_cases(_SearchCasesArgs.model_validate(call.arguments))
            if call.name == "verify_claim":
                return self._verify_claim(_VerifyClaimArgs.model_validate(call.arguments))
            if call.name == "inspect_destination":
                return self._inspect_destination(
                    _InspectDestinationArgs.model_validate(call.arguments)
                )
            if call.name == "get_content_context":
                return self._get_content_context(
                    _ContentContextArgs.model_validate(call.arguments)
                )
        except ValidationError:
            return self._error("invalid_arguments")
        return self._error("unknown_tool")

    def _search_cases(self, args: _SearchCasesArgs) -> ToolResult:
        records = _rank(
            args.query,
            self._cases,
            text=lambda record: record.text,
            stable_id=lambda record: record.case_id,
            top_k=args.top_k,
        )
        return ToolResult(
            status="ok",
            payload={
                "results": [
                    {
                        "case_id": record.case_id,
                        "summary": record.summary,
                        "source_id": record.source_id,
                        "source_type": record.source_type,
                    }
                    for record in records
                ]
            },
        )

    def _verify_claim(self, args: _VerifyClaimArgs) -> ToolResult:
        query = " ".join(part for part in (args.claim, args.product_type) if part)
        records = _rank(
            query,
            self._claims,
            text=lambda record: f"{record.claim} {record.product_type}",
            stable_id=lambda record: record.evidence_id,
            top_k=args.top_k,
            minimum_score=2,
        )
        return ToolResult(
            status="ok",
            payload={
                "results": [
                    {
                        "evidence_id": record.evidence_id,
                        "stance": record.stance,
                        "summary": record.summary,
                        "source_id": record.source_id,
                        "source_type": record.source_type,
                    }
                    for record in records
                ]
            },
        )

    def _inspect_destination(self, args: _InspectDestinationArgs) -> ToolResult:
        record = self._destination_index.get(_normalize_indicator(args.indicator))
        if record is None:
            return ToolResult(status="ok", payload={"result": None})
        return ToolResult(
            status="ok",
            payload={
                "result": {
                    "destination_id": record.destination_id,
                    "destination_type": record.destination_type,
                    "risk_signals": list(record.risk_signals),
                    "source_id": record.source_id,
                    "source_type": record.source_type,
                }
            },
        )

    def _get_content_context(self, args: _ContentContextArgs) -> ToolResult:
        record = self._history_index.get(args.content_id)
        if record is None:
            return ToolResult(status="ok", payload={"result": None})
        return ToolResult(
            status="ok",
            payload={
                "result": {
                    "content_id": record.content_id,
                    "recent_contents": list(record.recent_contents),
                    "account_signals": list(record.account_signals),
                    "source_id": record.source_id,
                    "source_type": record.source_type,
                }
            },
        )
