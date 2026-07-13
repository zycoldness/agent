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
        self._rows = tuple(
            {"case_id": str(row["case_id"]), "text": str(row["text"])} for row in rows
        )

    def search(self, query: str, top_k: int) -> list[dict[str, str]]:
        """Return the highest-overlap sanitized cases, with a stable case-id tie-break."""

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
