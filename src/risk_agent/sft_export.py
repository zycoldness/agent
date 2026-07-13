"""Leak-free serialization of policy-conditioned SFT conversations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from risk_agent.contracts import Action, Oracle, Task
from risk_agent.policy import render_active_policy


SYSTEM_SUFFIX = (
    "Return exactly one JSON action on every assistant turn. "
    'The only valid format is {"tool": "...", "arguments": {...}}.'
)
_ORACLE_ONLY_OBSERVATION_KEYS = frozenset({"label", "evidence_ids", "risk_level", "next_action"})
_ALLOWED_EVIDENCE_KINDS = frozenset({"ocr", "asr", "frame", "metadata", "case"})


def _compact_json(value: Any) -> str:
    """Produce deterministic JSON suitable for a single chat message."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _system(task: Task) -> str:
    return f"{render_active_policy(task.active_policy)}\n\n{SYSTEM_SUFFIX}"


def _validate_pair(task: Task, oracle: Oracle) -> None:
    if (task.asset_id, task.policy_version) != (oracle.asset_id, oracle.policy_version):
        raise ValueError("oracle must match the task asset_id and policy_version")
    active_rule_ids = {rule.rule_id for rule in task.active_policy}
    if oracle.rule_id is not None and oracle.rule_id not in active_rule_ids:
        raise ValueError("oracle rule_id must be present in the task active_policy")


def _final_action(oracle: Oracle) -> str:
    """Create the sole message that is permitted to contain oracle data."""

    arguments: dict[str, Any] = {
        "label": oracle.label,
        "rule_id": oracle.rule_id,
        "evidence_ids": list(oracle.evidence_ids),
        "confidence": 1.0,
    }
    if oracle.risk_level is not None:
        arguments["risk_level"] = oracle.risk_level
    if oracle.next_action is not None:
        arguments["next_action"] = oracle.next_action
    return _compact_json({"tool": "final_decision", "arguments": arguments})


def _validate_tool_arguments(action: Action) -> None:
    """Keep demonstrations executable by the closed-world environment."""

    arguments = action.arguments
    if action.tool == "get_rule_detail":
        if set(arguments) != {"rule_id"} or not isinstance(arguments.get("rule_id"), str) or not arguments["rule_id"].strip():
            raise ValueError("get_rule_detail requires exactly one non-empty string rule_id")
        return
    if action.tool == "search_case":
        if not set(arguments).issubset({"query", "top_k"}):
            raise ValueError("search_case only accepts query and top_k")
        query = arguments.get("query")
        top_k = arguments.get("top_k", 3)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search_case requires a non-empty string query")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 5:
            raise ValueError("search_case top_k must be an integer from 1 to 5")
        return
    if action.tool == "inspect_evidence":
        kinds = arguments.get("kinds")
        if (
            set(arguments) != {"kinds"}
            or not isinstance(kinds, tuple)
            or not kinds
            or any(not isinstance(kind, str) for kind in kinds)
            or not set(kinds).issubset(_ALLOWED_EVIDENCE_KINDS)
        ):
            raise ValueError("inspect_evidence requires non-empty allowed kinds")
        return
    raise ValueError("trajectory tool actions must not be final_decision")


def _normalize_tool_action(action_text: object) -> str:
    if not isinstance(action_text, str):
        raise ValueError("trajectory actions must be JSON strings")
    try:
        payload = json.loads(action_text)
    except json.JSONDecodeError as error:
        raise ValueError("trajectory actions must be valid JSON action objects") from error
    if not isinstance(payload, dict) or set(payload) != {"tool", "arguments"}:
        raise ValueError("trajectory actions must be one JSON action with tool and arguments")
    try:
        action = Action.model_validate(payload)
    except ValidationError as error:
        raise ValueError("trajectory actions must be valid JSON action objects") from error
    _validate_tool_arguments(action)
    return _compact_json(action.model_dump(mode="json"))


def _contains_oracle_only_field(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            key in _ORACLE_ONLY_OBSERVATION_KEYS or _contains_oracle_only_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_oracle_only_field(item) for item in value)
    return False


def _normalize_observation(observation: object) -> str:
    if not isinstance(observation, str):
        raise ValueError("trajectory observations must be strings")
    try:
        parsed = json.loads(observation)
    except json.JSONDecodeError:
        return observation
    if _contains_oracle_only_field(parsed):
        raise ValueError("tool observation must not contain an oracle field")
    return observation


def export_track_a(task: Task, oracle: Oracle) -> dict[str, list[dict[str, str]]]:
    """Export a no-tool guard demonstration with an oracle-derived final action."""

    _validate_pair(task, oracle)
    return {
        "messages": [
            {"role": "system", "content": _system(task)},
            {"role": "user", "content": task.initial_observation},
            {"role": "assistant", "content": _final_action(oracle)},
        ]
    }


def export_track_b(
    task: Task,
    oracle: Oracle,
    action: str,
    observation: str,
) -> dict[str, list[dict[str, str]]]:
    """Export a one-tool-turn trajectory; retained for simple data generators."""

    return export_trajectory(task, oracle, [(action, observation)])


def export_trajectory(
    task: Task,
    oracle: Oracle,
    steps: Sequence[tuple[str, str]],
) -> dict[str, list[dict[str, str]]]:
    """Export a bounded tool trajectory followed by exactly one final action."""

    _validate_pair(task, oracle)
    if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence):
        raise ValueError("trajectory steps must be a sequence of action-observation pairs")
    if len(steps) > task.max_turns - 1:
        raise ValueError("trajectory exceeds task max_turns after reserving the final decision")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _system(task)},
        {"role": "user", "content": task.initial_observation},
    ]
    for step in steps:
        if not isinstance(step, (tuple, list)) or len(step) != 2:
            raise ValueError("each trajectory step must be an action-observation pair")
        action_text, observation = step
        messages.append({"role": "assistant", "content": _normalize_tool_action(action_text)})
        messages.append(
            {"role": "user", "content": f"Tool observation: {_normalize_observation(observation)}"}
        )
    messages.append({"role": "assistant", "content": _final_action(oracle)})
    return {"messages": messages}
