"""Pure, fail-closed rewards for strict final content-risk decisions."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


_ARG_REQUIRED = frozenset({"label", "rule_id", "confidence"})
_ARG_OPTIONAL = frozenset({"risk_level", "next_action"})


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _decision(value: object) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict) or set(parsed) != {"tool", "arguments"}:
        return None
    if parsed["tool"] != "final_decision" or not isinstance(parsed["arguments"], dict):
        return None
    args = parsed["arguments"]
    if not _ARG_REQUIRED.issubset(args) or set(args) - _ARG_REQUIRED - _ARG_OPTIONAL:
        return None
    if args["label"] not in ("safe", "unsafe"):
        return None
    if args["rule_id"] is not None and not isinstance(args["rule_id"], str):
        return None
    confidence = args["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        return None
    if "risk_level" in args and args["risk_level"] not in (None, "P0", "P1", "P2", "P3"):
        return None
    if "next_action" in args and args["next_action"] is not None and not isinstance(args["next_action"], str):
        return None
    return args


def _batch(
    completions: object,
    solution: object,
    component: Callable[[dict[str, Any], dict[str, Any]], bool] | None,
) -> list[float]:
    if not isinstance(completions, list):
        return []
    zeros = [0.0] * len(completions)
    if not isinstance(solution, list) or len(solution) != len(completions):
        return zeros
    result: list[float] = []
    for completion, target in zip(completions, solution):
        predicted = _decision(completion)
        expected = _decision(target)
        valid = predicted is not None and expected is not None
        result.append(float(valid and (component is None or component(predicted, expected))))
    return result


def format_reward(completions: object, *, solution: object = None, **kwargs: object) -> list[float]:
    return _batch(completions, solution, None)


def label_exact_reward(completions: object, *, solution: object = None, **kwargs: object) -> list[float]:
    return _batch(completions, solution, lambda got, want: got["label"] == want["label"])


def rule_exact_reward(completions: object, *, solution: object = None, **kwargs: object) -> list[float]:
    return _batch(
        completions,
        solution,
        lambda got, want: got["label"] == want["label"] and got["rule_id"] == want["rule_id"],
    )
