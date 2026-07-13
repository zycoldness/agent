"""Read-only stores for sanitized historical cases and source evidence."""

import re
from collections.abc import Iterable, Mapping

from risk_agent.contracts import Evidence


def _tokens(text: str) -> set[str]:
    """Extract word tokens and individual CJK characters for overlap ranking."""

    return set(re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]", text.casefold()))


class CaseStore:
    """Searchable case text that deliberately excludes evaluation metadata."""

    def __init__(self, rows: Iterable[Mapping[str, object]]) -> None:
        sanitized_rows: list[dict[str, str]] = []
        case_ids: set[str] = set()
        for row in rows:
            case_id = str(row["case_id"])
            if case_id in case_ids:
                raise ValueError(f"duplicate case_id: {case_id}")
            case_ids.add(case_id)
            sanitized_rows.append({"case_id": case_id, "text": str(row["text"])})
        self._rows = tuple(sanitized_rows)

    def search(self, query: str, top_k: int) -> list[dict[str, str]]:
        """Return the highest-overlap sanitized cases, with a stable case-id tie-break."""

        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer")
        if top_k <= 0:
            return []

        query_tokens = _tokens(query)
        ranked = sorted(
            self._rows,
            key=lambda row: (
                -len(query_tokens & _tokens(row["text"])),
                row["case_id"],
            ),
        )
        return [dict(row) for row in ranked[:top_k]]


class EvidenceStore:
    """Evidence lookup preserving original source insertion order."""

    def __init__(self, evidence: Iterable[Evidence]) -> None:
        self._evidence = tuple(evidence)

    def inspect(self, asset_id: str, kinds: set[str]) -> list[Evidence]:
        """Return selected evidence for an asset in source insertion order."""

        return [
            item
            for item in self._evidence
            if item.asset_id == asset_id and item.kind in kinds
        ]
