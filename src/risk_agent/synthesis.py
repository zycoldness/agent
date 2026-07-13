"""Generate leak-free SFT rows from privileged structured teacher candidates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from risk_agent.contracts import Action, Decision, Oracle, Task
from risk_agent.sft_export import export_track_a, export_trajectory
from risk_agent.stores import CaseStore, EvidenceStore
from risk_agent.teacher import Teacher, TeacherReply, TeacherUsage


_LOOKUP_TOOLS = frozenset({"get_rule_detail", "search_case", "inspect_evidence"})
_FINAL_KEYS = frozenset(
    {"label", "rule_id", "evidence_ids", "confidence", "risk_level", "route", "next_action"}
)


@dataclass(frozen=True)
class SynthesisResult:
    """A student-safe SFT row plus prompt-free provenance metadata."""

    row: dict[str, object]
    metadata: dict[str, object]


def _parse_action(payload: Mapping[str, Any]) -> Action:
    if set(payload) - {"tool", "arguments", "observation"}:
        raise ValueError("teacher response must be one valid JSON action")
    canonical = {key: payload[key] for key in ("tool", "arguments") if key in payload}
    if set(canonical) != {"tool", "arguments"}:
        raise ValueError("teacher response must be one valid JSON action")
    try:
        return Action.model_validate(canonical)
    except ValidationError as error:
        raise ValueError("teacher response must be one valid JSON action") from error


def _teacher_request(
    task: Task,
    oracle: Oracle,
    *,
    phase: str,
    first_action: dict[str, object] | None = None,
    local_observation: object | None = None,
) -> dict[str, object]:
    request: dict[str, object] = {
        "instruction": (
            "Return one JSON action only. First hop may be final_decision or one allowed lookup. "
            "Second hop must be final_decision. Follow the supplied oracle exactly and never invent evidence."
        ),
        "phase": phase,
        "task": task.model_dump(mode="json"),
        "oracle": oracle.model_dump(mode="json"),
    }
    if first_action is not None:
        request["first_action"] = first_action
    if local_observation is not None:
        request["local_observation"] = local_observation
    return request


def _execute_lookup(
    task: Task,
    action: Action,
    case_store: CaseStore,
    evidence_store: EvidenceStore,
) -> tuple[object, set[str]]:
    arguments = action.arguments
    if action.tool == "get_rule_detail":
        if set(arguments) != {"rule_id"} or not isinstance(arguments.get("rule_id"), str):
            raise ValueError("invalid get_rule_detail teacher action")
        rule = next((item for item in task.active_policy if item.rule_id == arguments["rule_id"]), None)
        if rule is None:
            raise ValueError("teacher lookup rule must be active")
        return rule.model_dump(mode="json"), set()
    if action.tool == "search_case":
        if not set(arguments).issubset({"query", "top_k"}):
            raise ValueError("invalid search_case teacher action")
        query = arguments.get("query")
        top_k = arguments.get("top_k", 3)
        if (
            not isinstance(query, str)
            or not query.strip()
            or isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 5
        ):
            raise ValueError("invalid search_case teacher action")
        return case_store.search(query, top_k), set()
    if action.tool == "inspect_evidence":
        kinds = arguments.get("kinds")
        if (
            set(arguments) != {"kinds"}
            or not isinstance(kinds, tuple)
            or not kinds
            or not set(kinds).issubset({"ocr", "asr", "frame", "metadata", "case"})
        ):
            raise ValueError("invalid inspect_evidence teacher action")
        evidence = evidence_store.inspect(task.asset_id, set(kinds))
        return (
            [item.model_dump(mode="json") for item in evidence],
            {item.evidence_id for item in evidence},
        )
    raise ValueError("first teacher action must be final_decision or one allowed lookup tool")


def _validate_final(
    task: Task,
    oracle: Oracle,
    action: Action,
    *,
    observed_evidence_ids: set[str] | None,
) -> None:
    if action.tool != "final_decision":
        raise ValueError("second teacher action must be final_decision")
    if not set(action.arguments).issubset(_FINAL_KEYS):
        raise ValueError("teacher final decision has unknown arguments")
    try:
        decision = Decision.model_validate(dict(action.arguments), strict=True)
    except ValidationError as error:
        raise ValueError("teacher final decision is invalid") from error
    active_rule_ids = {rule.rule_id for rule in task.active_policy}
    if decision.rule_id is not None and decision.rule_id not in active_rule_ids:
        raise ValueError("teacher final decision cites a non-active rule")
    if (
        decision.label != oracle.label
        or decision.rule_id != oracle.rule_id
        or tuple(decision.evidence_ids) != tuple(oracle.evidence_ids)
        or decision.risk_level != oracle.risk_level
        or decision.next_action != oracle.next_action
    ):
        raise ValueError("teacher final decision must agree with the oracle")
    if observed_evidence_ids is not None and not set(decision.evidence_ids).issubset(observed_evidence_ids):
        raise ValueError("teacher final decision cites unobserved evidence")


def _combine_usage(replies: list[TeacherReply]) -> dict[str, object]:
    usages = [reply.usage for reply in replies]
    input_tokens = sum(item.input_tokens for item in usages if item.input_tokens is not None)
    output_tokens = sum(item.output_tokens for item in usages if item.output_tokens is not None)
    costs = [item.estimated_cost_usd for item in usages]
    return TeacherUsage(
        provider=usages[0].provider,
        model=usages[0].model,
        request_count=sum(item.request_count for item in usages),
        input_tokens=input_tokens if all(item.input_tokens is not None for item in usages) else None,
        output_tokens=output_tokens if all(item.output_tokens is not None for item in usages) else None,
        estimated_cost_usd=sum(costs) if all(item is not None for item in costs) else None,
    ).as_dict()


def generate_teacher_sample(
    task: Task,
    oracle: Oracle,
    case_store: CaseStore,
    evidence_store: EvidenceStore,
    teacher: Teacher,
    *,
    data_classification: str = "synthetic",
) -> SynthesisResult:
    """Ask a privileged teacher for at most one lookup, then rebuild student data locally."""

    if data_classification not in {"synthetic", "public"}:
        raise ValueError("data classification must be synthetic or public")
    if (task.asset_id, task.policy_version) != (oracle.asset_id, oracle.policy_version):
        raise ValueError("oracle must match task")

    first_reply = teacher.generate(_teacher_request(task, oracle, phase="first"))
    first_action = _parse_action(first_reply.payload)
    replies = [first_reply]
    if first_action.tool == "final_decision":
        _validate_final(task, oracle, first_action, observed_evidence_ids=None)
        row = export_track_a(task, oracle)
        trajectory_type = "direct"
    else:
        if task.max_turns < 2 or first_action.tool not in _LOOKUP_TOOLS:
            raise ValueError("first teacher action must be final_decision or one allowed lookup tool")
        local_observation, observed_ids = _execute_lookup(task, first_action, case_store, evidence_store)
        second_reply = teacher.generate(
            _teacher_request(
                task,
                oracle,
                phase="second_final",
                first_action=first_action.model_dump(mode="json"),
                local_observation=local_observation,
            )
        )
        replies.append(second_reply)
        second_action = _parse_action(second_reply.payload)
        _validate_final(task, oracle, second_action, observed_evidence_ids=observed_ids)
        action_text = json.dumps(
            first_action.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
        )
        observation_text = json.dumps(local_observation, ensure_ascii=False, separators=(",", ":"))
        row = export_trajectory(
            task,
            oracle,
            [(action_text, observation_text)],
            case_store,
            evidence_store,
        )
        trajectory_type = "one_lookup"

    return SynthesisResult(
        row=row,
        metadata={
            "data_classification": data_classification,
            "trajectory_type": trajectory_type,
            "teacher": _combine_usage(replies),
        },
    )

