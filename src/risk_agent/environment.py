"""A bounded closed-world environment for policy-conditioned risk decisions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from risk_agent.contracts import Action, Decision, Oracle, Task
from risk_agent.policy import render_active_policy
from risk_agent.stores import CaseStore, EvidenceStore


@dataclass(frozen=True)
class StepResult:
    """The observation and transition metadata produced by one environment step."""

    observation: str
    reward: float
    done: bool
    info: dict[str, object]


class RiskEnvironment:
    """Expose only policy-scoped tools; retain oracle data solely for final scoring."""

    _TOOL_COST = -0.08
    _INVALID_ACTION_COST = -0.5
    _ALLOWED_EVIDENCE_KINDS = frozenset({"ocr", "asr", "frame", "metadata", "case"})
    _DECISION_ARGUMENT_KEYS = frozenset(
        {
            "label",
            "rule_id",
            "evidence_ids",
            "confidence",
            "risk_level",
            "route",
            "next_action",
        }
    )

    def __init__(
        self,
        task: Task,
        case_store: CaseStore,
        evidence_store: EvidenceStore,
        oracle: Oracle,
    ) -> None:
        if (oracle.asset_id, oracle.policy_version) != (task.asset_id, task.policy_version):
            raise ValueError("oracle must belong to the task asset and policy version")

        self.task = task
        self.case_store = case_store
        self.evidence_store = evidence_store
        self.oracle = oracle
        self.turns = 0
        self.observed_evidence_ids: set[str] = set()
        self._done = False

    def reset(self) -> str:
        """Start a fresh trajectory with the entire active policy in context."""

        self.turns = 0
        self.observed_evidence_ids.clear()
        self._done = False
        return f"{render_active_policy(self.task.active_policy)}\n\nAsset:\n{self.task.initial_observation}"

    def step(self, action_text: str) -> StepResult:
        """Execute exactly one JSON action, terminating safely on malformed requests."""

        if self._done:
            return self._invalid_action()

        action = self._parse_action(action_text)
        if action is None:
            return self._invalid_action()

        self.turns += 1
        if action.tool == "get_rule_detail":
            return self._get_rule_detail(action.arguments)
        if action.tool == "search_case":
            return self._search_case(action.arguments)
        if action.tool == "inspect_evidence":
            return self._inspect_evidence(action.arguments)
        return self._finalize(action.arguments)

    def _parse_action(self, action_text: str) -> Action | None:
        if not isinstance(action_text, str):
            return None
        try:
            payload = json.loads(action_text)
            return Action.model_validate(payload)
        except (TypeError, ValueError, ValidationError, json.JSONDecodeError):
            return None

    def _get_rule_detail(self, arguments: Mapping[str, Any]) -> StepResult:
        if set(arguments) != {"rule_id"} or not isinstance(arguments.get("rule_id"), str):
            return self._invalid_action()

        rule_id = arguments["rule_id"]
        rule = next(
            (candidate for candidate in self.task.active_policy if candidate.rule_id == rule_id),
            None,
        )
        if rule is None:
            return self._invalid_action()

        return self._tool_result(rule.model_dump_json())

    def _search_case(self, arguments: Mapping[str, Any]) -> StepResult:
        if not set(arguments).issubset({"query", "top_k"}):
            return self._invalid_action()
        query = arguments.get("query")
        top_k = arguments.get("top_k", 3)
        if not isinstance(query, str) or not query.strip():
            return self._invalid_action()
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 5:
            return self._invalid_action()

        try:
            rows = self.case_store.search(query, top_k)
        except ValueError:
            return self._invalid_action()
        return self._tool_result(json.dumps(rows, ensure_ascii=False, sort_keys=True))

    def _inspect_evidence(self, arguments: Mapping[str, Any]) -> StepResult:
        if set(arguments) != {"kinds"}:
            return self._invalid_action()
        kinds = arguments.get("kinds")
        if (
            not isinstance(kinds, tuple)
            or not kinds
            or any(not isinstance(kind, str) for kind in kinds)
            or not set(kinds).issubset(self._ALLOWED_EVIDENCE_KINDS)
        ):
            return self._invalid_action()

        evidence = self.evidence_store.inspect(self.task.asset_id, set(kinds))
        self.observed_evidence_ids.update(item.evidence_id for item in evidence)
        return self._tool_result(
            json.dumps(
                [item.model_dump(mode="json") for item in evidence],
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    def _finalize(self, arguments: Mapping[str, Any]) -> StepResult:
        if not set(arguments).issubset(self._DECISION_ARGUMENT_KEYS):
            return self._invalid_action()
        try:
            decision = Decision.model_validate(dict(arguments), strict=True)
        except ValidationError:
            return self._invalid_action()

        active_rule_ids = {rule.rule_id for rule in self.task.active_policy}
        if decision.rule_id is not None and decision.rule_id not in active_rule_ids:
            return self._invalid_action()
        if not set(decision.evidence_ids).issubset(self.observed_evidence_ids):
            return self._invalid_action()

        self._done = True
        reward = float(decision.label == self.oracle.label)
        reward += 0.6 * float(decision.rule_id == self.oracle.rule_id)
        reward += 0.4 * float(bool(set(decision.evidence_ids) & set(self.oracle.evidence_ids)))
        return StepResult(
            observation="Assessment complete.",
            reward=reward,
            done=True,
            info={"status": "final", "decision": decision.model_dump(mode="json")},
        )

    def _tool_result(self, observation: str) -> StepResult:
        reached_turn_limit = self.turns >= self.task.max_turns
        if reached_turn_limit:
            self._done = True
        return StepResult(
            observation=observation,
            reward=self._TOOL_COST,
            done=reached_turn_limit,
            info={"status": "turn_limit" if reached_turn_limit else "tool"},
        )

    def _invalid_action(self) -> StepResult:
        self._done = True
        return StepResult(
            observation="Invalid action.",
            reward=self._INVALID_ACTION_COST,
            done=True,
            info={"status": "invalid_action"},
        )
