"""Leak-free serialization of policy-conditioned SFT conversations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from risk_agent.contracts import Action, Evidence, Oracle, PolicyRule, Task
from risk_agent.policy import render_active_policy
from risk_agent.stores import CaseStore, EvidenceStore


SYSTEM_SUFFIX = (
    "Return exactly one JSON action on every assistant turn. "
    'The only valid format is {"tool": "...", "arguments": {...}}.'
)
_FORBIDDEN_OBSERVATION_KEYS = frozenset(
    {"oracle", "label", "evidence_ids", "risk_level", "next_action", "verdict", "decision"}
)
_ALLOWED_EVIDENCE_KINDS = frozenset({"ocr", "asr", "frame", "metadata", "case"})


def _compact_json(value: Any) -> str:
    """Produce deterministic JSON suitable for a single chat message."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_loads(value: str) -> Any:
    return json.loads(
        value,
        object_pairs_hook=_reject_duplicate_json_pairs,
        parse_constant=_reject_nonfinite_json_constant,
    )


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


def _parse_tool_action(action_text: object) -> Action:
    if not isinstance(action_text, str):
        raise ValueError("trajectory actions must be JSON strings")
    try:
        payload = _strict_json_loads(action_text)
    except json.JSONDecodeError as error:
        raise ValueError("trajectory actions must be valid JSON action objects") from error
    if not isinstance(payload, dict) or set(payload) != {"tool", "arguments"}:
        raise ValueError("trajectory actions must be one JSON action with tool and arguments")
    try:
        action = Action.model_validate(payload)
    except ValidationError as error:
        raise ValueError("trajectory actions must be valid JSON action objects") from error
    _validate_tool_arguments(action)
    return action


def _contains_forbidden_observation_field(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            isinstance(key, str)
            and (key.casefold() in _FORBIDDEN_OBSERVATION_KEYS or key.casefold().startswith("gold"))
            or _contains_forbidden_observation_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_observation_field(item) for item in value)
    return False


def _parse_json_observation(observation: object) -> object:
    if not isinstance(observation, str):
        raise ValueError("trajectory observations must be strings")
    try:
        return _strict_json_loads(observation)
    except json.JSONDecodeError as error:
        raise ValueError("tool observation must be valid JSON") from error


def _normalize_rule_detail(task: Task, action: Action, observation: object) -> str:
    if not isinstance(observation, Mapping) or set(observation) != {
        "rule_id",
        "title",
        "text",
        "exceptions",
        "priority",
    }:
        raise ValueError("tool observation must be one active policy rule matching requested rule_id")
    if (
        not isinstance(observation["rule_id"], str)
        or not isinstance(observation["title"], str)
        or not isinstance(observation["text"], str)
        or not isinstance(observation["exceptions"], list)
        or any(not isinstance(item, str) for item in observation["exceptions"])
        or isinstance(observation["priority"], bool)
        or not isinstance(observation["priority"], int)
    ):
        raise ValueError("tool observation must be one active policy rule matching requested rule_id")
    try:
        rule = PolicyRule.model_validate(observation)
    except ValidationError as error:
        raise ValueError("tool observation must be one active policy rule matching requested rule_id") from error
    requested_rule_id = action.arguments["rule_id"]
    expected = next((item for item in task.active_policy if item.rule_id == requested_rule_id), None)
    if expected is None or rule != expected:
        raise ValueError("tool observation must be one active policy rule matching requested rule_id")
    return _compact_json(rule.model_dump(mode="json"))


def _normalize_case_results(case_store: CaseStore, action: Action, observation: object) -> str:
    if not isinstance(observation, list):
        raise ValueError("tool observation must contain sanitized case records only")
    top_k = action.arguments.get("top_k", 3)
    if len(observation) > top_k:
        raise ValueError("tool observation must not contain more cases than requested")
    normalized: list[dict[str, str]] = []
    case_ids: set[str] = set()
    for row in observation:
        if not isinstance(row, Mapping) or set(row) != {"case_id", "text"}:
            raise ValueError("tool observation must contain sanitized case records only")
        case_id, text = row["case_id"], row["text"]
        if not isinstance(case_id, str) or not case_id or not isinstance(text, str):
            raise ValueError("tool observation must contain sanitized case records only")
        if case_id in case_ids:
            raise ValueError("tool observation must not repeat case_id values")
        case_ids.add(case_id)
        normalized.append({"case_id": case_id, "text": text})
    expected = case_store.search(action.arguments["query"], action.arguments.get("top_k", 3))
    if normalized != expected:
        raise ValueError("tool observation must match the deterministic case-store result")
    return _compact_json(expected)


def _normalize_evidence(
    task: Task,
    evidence_store: EvidenceStore,
    action: Action,
    observation: object,
) -> str:
    expected_keys = {"evidence_id", "asset_id", "kind", "content"}
    if not isinstance(observation, list):
        raise ValueError("tool observation evidence must match requested asset and kinds")
    requested_kinds = set(action.arguments["kinds"])
    normalized: list[dict[str, object]] = []
    evidence_ids: set[str] = set()
    for item in observation:
        if not isinstance(item, Mapping) or set(item) != expected_keys:
            raise ValueError("tool observation evidence must match requested asset and kinds")
        if any(not isinstance(item[key], str) for key in expected_keys):
            raise ValueError("tool observation evidence must match requested asset and kinds")
        try:
            evidence = Evidence.model_validate(item)
        except ValidationError as error:
            raise ValueError("tool observation evidence must match requested asset and kinds") from error
        if evidence.asset_id != task.asset_id or evidence.kind not in requested_kinds:
            raise ValueError("tool observation evidence must match requested asset and kinds")
        if evidence.evidence_id in evidence_ids:
            raise ValueError("tool observation must not repeat evidence_id values")
        evidence_ids.add(evidence.evidence_id)
        normalized.append(evidence.model_dump(mode="json"))
    expected = [
        item.model_dump(mode="json")
        for item in evidence_store.inspect(task.asset_id, set(action.arguments["kinds"]))
    ]
    if normalized != expected:
        raise ValueError("tool observation must match the deterministic evidence-store result")
    return _compact_json(expected)


def _normalize_observation(
    task: Task,
    case_store: CaseStore,
    evidence_store: EvidenceStore,
    action: Action,
    raw_observation: object,
) -> str:
    observation = _parse_json_observation(raw_observation)
    if _contains_forbidden_observation_field(observation):
        raise ValueError("tool observation must not contain an oracle field")
    if action.tool == "get_rule_detail":
        return _normalize_rule_detail(task, action, observation)
    if action.tool == "search_case":
        return _normalize_case_results(case_store, action, observation)
    if action.tool == "inspect_evidence":
        return _normalize_evidence(task, evidence_store, action, observation)
    raise ValueError("trajectory tool actions must not be final_decision")


def _row(task: Task, messages: list[dict[str, str]]) -> dict[str, object]:
    row: dict[str, object] = {"messages": messages}
    if task.images:
        row["images"] = list(task.images)
    if task.videos:
        row["videos"] = list(task.videos)
    return row


def export_track_a(task: Task, oracle: Oracle) -> dict[str, object]:
    """Export a no-tool guard demonstration with an oracle-derived final action."""

    _validate_pair(task, oracle)
    return _row(
        task,
        [
            {"role": "system", "content": _system(task)},
            {"role": "user", "content": task.initial_observation},
            {"role": "assistant", "content": _final_action(oracle)},
        ],
    )


def export_track_b(
    task: Task,
    oracle: Oracle,
    action: str,
    observation: str,
    case_store: CaseStore,
    evidence_store: EvidenceStore,
) -> dict[str, object]:
    """Export a one-tool-turn trajectory; retained for simple data generators."""

    return export_trajectory(task, oracle, [(action, observation)], case_store, evidence_store)


def export_trajectory(
    task: Task,
    oracle: Oracle,
    steps: Sequence[tuple[str, str]],
    case_store: CaseStore,
    evidence_store: EvidenceStore,
) -> dict[str, object]:
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
        action = _parse_tool_action(action_text)
        messages.append({"role": "assistant", "content": _compact_json(action.model_dump(mode="json"))})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Tool observation: "
                    f"{_normalize_observation(task, case_store, evidence_store, action, observation)}"
                ),
            }
        )
    messages.append({"role": "assistant", "content": _final_action(oracle)})
    return _row(task, messages)
