"""Fixtures for testing whether decisions change with the active policy."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from risk_agent.contracts import Oracle, PolicyRule, Task

PolicyTransformation = Literal[
    "rule_addition",
    "rule_removal",
    "rule_rewrite",
    "exemption",
]
_TRANSFORMATIONS = frozenset({"rule_addition", "rule_removal", "rule_rewrite", "exemption"})


@dataclass(frozen=True)
class PolicyOutcome:
    """An explicit expected assessment under one complete active policy."""

    label: Literal["safe", "unsafe"]
    rule_id: str | None = None
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.label not in {"safe", "unsafe"}:
            raise ValueError("outcome label must be 'safe' or 'unsafe'")
        if self.rule_id is not None and not isinstance(self.rule_id, str):
            raise TypeError("outcome rule_id must be a string or None")
        if isinstance(self.evidence_ids, (Mapping, str, bytes)) or not isinstance(
            self.evidence_ids, Iterable
        ):
            raise TypeError("outcome evidence_ids must be an iterable of string IDs")
        evidence_ids = tuple(self.evidence_ids)
        if any(not isinstance(evidence_id, str) for evidence_id in evidence_ids):
            raise TypeError("outcome evidence_ids must contain only string IDs")
        object.__setattr__(self, "evidence_ids", evidence_ids)


@dataclass(frozen=True)
class PolicyShiftRow:
    """One named policy variant and its isolated reference assessment."""

    task: Task
    oracle: Oracle
    transformation: PolicyTransformation = "rule_removal"
    variant: Literal["before", "after"] = "before"
    group_id: str | None = None


def _normalize_policy(policy: Iterable[PolicyRule], *, field_name: str) -> tuple[PolicyRule, ...]:
    """Freeze a complete policy and reject ambiguous duplicate rule IDs."""

    if isinstance(policy, (str, bytes)) or not isinstance(policy, Iterable):
        raise TypeError(f"{field_name} must be an iterable of PolicyRule values")
    normalized = tuple(policy)
    if any(not isinstance(rule, PolicyRule) for rule in normalized):
        raise TypeError(f"{field_name} must contain only PolicyRule values")
    rule_ids = [rule.rule_id for rule in normalized]
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError(f"{field_name} must not contain duplicate rule IDs")
    return normalized


def _require_policy_rule(rule: PolicyRule, *, field_name: str) -> PolicyRule:
    if not isinstance(rule, PolicyRule):
        raise TypeError(f"{field_name} must be a PolicyRule")
    return rule


def _validate_outcome(policy: tuple[PolicyRule, ...], outcome: PolicyOutcome, *, field_name: str) -> None:
    if not isinstance(outcome, PolicyOutcome):
        raise TypeError(f"{field_name} must be a PolicyOutcome")
    rule_ids = {rule.rule_id for rule in policy}
    if outcome.rule_id is not None and outcome.rule_id not in rule_ids:
        raise ValueError(f"{field_name}.rule_id must belong to its active policy")


def _policy_version(
    transformation: PolicyTransformation,
    variant: Literal["before", "after"],
    policy: tuple[PolicyRule, ...],
) -> str:
    """Make reproducible, collision-resistant policy versions for generated rows."""

    serialized_policy = json.dumps(
        [rule.model_dump(mode="json") for rule in policy],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(serialized_policy.encode("utf-8")).hexdigest()[:12]
    return f"{transformation}-{variant}-{fingerprint}"


def _resolve_policy_version(
    supplied_version: str | None,
    transformation: PolicyTransformation,
    variant: Literal["before", "after"],
    policy: tuple[PolicyRule, ...],
) -> str:
    if supplied_version is None:
        return _policy_version(transformation, variant, policy)
    if not isinstance(supplied_version, str) or not supplied_version.strip():
        raise ValueError("policy versions must be non-empty strings")
    return supplied_version


def _group_id(
    *,
    asset_id: str,
    observation: str,
    transformation: PolicyTransformation,
    before_policy: tuple[PolicyRule, ...],
    after_policy: tuple[PolicyRule, ...],
    before_outcome: PolicyOutcome,
    after_outcome: PolicyOutcome,
) -> str:
    """Return one reproducible identity joining exactly this before/after pair."""

    payload = {
        "asset_id": asset_id,
        "observation": observation,
        "transformation": transformation,
        "before_policy": [rule.model_dump(mode="json") for rule in before_policy],
        "after_policy": [rule.model_dump(mode="json") for rule in after_policy],
        "before_outcome": {
            "label": before_outcome.label,
            "rule_id": before_outcome.rule_id,
            "evidence_ids": before_outcome.evidence_ids,
        },
        "after_outcome": {
            "label": after_outcome.label,
            "rule_id": after_outcome.rule_id,
            "evidence_ids": after_outcome.evidence_ids,
        },
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return f"{transformation}-group-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _build_row(
    *,
    asset_id: str,
    observation: str,
    policy: tuple[PolicyRule, ...],
    policy_version: str,
    outcome: PolicyOutcome,
    transformation: PolicyTransformation,
    variant: Literal["before", "after"],
    group_id: str,
) -> PolicyShiftRow:
    task = Task(
        asset_id=asset_id,
        policy_version=policy_version,
        active_policy=policy,
        initial_observation=observation,
    )
    oracle = Oracle(
        asset_id=asset_id,
        policy_version=policy_version,
        label=outcome.label,
        rule_id=outcome.rule_id,
        evidence_ids=outcome.evidence_ids,
    )
    return PolicyShiftRow(
        task,
        oracle,
        transformation=transformation,
        variant=variant,
        group_id=group_id,
    )


def build_policy_shift_rows(
    asset_id: str,
    observation: str,
    *,
    before_policy: Iterable[PolicyRule],
    after_policy: Iterable[PolicyRule],
    before_outcome: PolicyOutcome,
    after_outcome: PolicyOutcome,
    transformation: PolicyTransformation,
    before_policy_version: str | None = None,
    after_policy_version: str | None = None,
) -> tuple[PolicyShiftRow, PolicyShiftRow]:
    """Build one explicit before/after policy transformation for the same asset.

    Callers provide full policies and labels rather than relying on hidden policy
    semantics.  Default policy versions are deterministic fingerprints of each
    transformation and policy; callers may instead pass their own versions.
    """

    if transformation not in _TRANSFORMATIONS:
        raise ValueError(f"unsupported policy transformation: {transformation!r}")
    before = _normalize_policy(before_policy, field_name="before_policy")
    after = _normalize_policy(after_policy, field_name="after_policy")
    _validate_outcome(before, before_outcome, field_name="before_outcome")
    _validate_outcome(after, after_outcome, field_name="after_outcome")
    before_version = _resolve_policy_version(
        before_policy_version, transformation, "before", before
    )
    after_version = _resolve_policy_version(after_policy_version, transformation, "after", after)
    if before_version == after_version:
        raise ValueError("before and after policy versions must differ")
    group_id = _group_id(
        asset_id=asset_id,
        observation=observation,
        transformation=transformation,
        before_policy=before,
        after_policy=after,
        before_outcome=before_outcome,
        after_outcome=after_outcome,
    )
    return (
        _build_row(
            asset_id=asset_id,
            observation=observation,
            policy=before,
            policy_version=before_version,
            outcome=before_outcome,
            transformation=transformation,
            variant="before",
            group_id=group_id,
        ),
        _build_row(
            asset_id=asset_id,
            observation=observation,
            policy=after,
            policy_version=after_version,
            outcome=after_outcome,
            transformation=transformation,
            variant="after",
            group_id=group_id,
        ),
    )


def build_rule_addition_tasks(
    asset_id: str,
    observation: str,
    added_rule: PolicyRule,
    *,
    unrelated_rules: Iterable[PolicyRule] = (),
    before_policy_version: str | None = None,
    after_policy_version: str | None = None,
) -> tuple[PolicyShiftRow, PolicyShiftRow]:
    """Create a safe-to-unsafe pair after adding a matching rule."""

    added_rule = _require_policy_rule(added_rule, field_name="added_rule")
    unrelated = _normalize_policy(unrelated_rules, field_name="unrelated_rules")
    return build_policy_shift_rows(
        asset_id,
        observation,
        before_policy=unrelated,
        after_policy=(*unrelated, added_rule),
        before_outcome=PolicyOutcome(label="safe"),
        after_outcome=PolicyOutcome(label="unsafe", rule_id=added_rule.rule_id),
        transformation="rule_addition",
        before_policy_version=before_policy_version,
        after_policy_version=after_policy_version,
    )


def build_rule_removal_tasks(
    asset_id: str,
    observation: str,
    removed_rule: PolicyRule,
    *,
    unrelated_rules: Iterable[PolicyRule] = (),
    before_policy_version: str | None = None,
    after_policy_version: str | None = None,
) -> tuple[PolicyShiftRow, PolicyShiftRow]:
    """Create an unsafe-to-safe pair after removing a matching rule."""

    removed_rule = _require_policy_rule(removed_rule, field_name="removed_rule")
    unrelated = _normalize_policy(unrelated_rules, field_name="unrelated_rules")
    return build_policy_shift_rows(
        asset_id,
        observation,
        before_policy=(*unrelated, removed_rule),
        after_policy=unrelated,
        before_outcome=PolicyOutcome(label="unsafe", rule_id=removed_rule.rule_id),
        after_outcome=PolicyOutcome(label="safe"),
        transformation="rule_removal",
        before_policy_version=before_policy_version,
        after_policy_version=after_policy_version,
    )


def build_rule_rewrite_tasks(
    asset_id: str,
    observation: str,
    original_rule: PolicyRule,
    rewritten_rule: PolicyRule,
    *,
    unrelated_rules: Iterable[PolicyRule] = (),
    before_policy_version: str | None = None,
    after_policy_version: str | None = None,
) -> tuple[PolicyShiftRow, PolicyShiftRow]:
    """Create an unsafe-to-safe pair when a matching rule is rewritten."""

    original_rule = _require_policy_rule(original_rule, field_name="original_rule")
    rewritten_rule = _require_policy_rule(rewritten_rule, field_name="rewritten_rule")
    unrelated = _normalize_policy(unrelated_rules, field_name="unrelated_rules")
    return build_policy_shift_rows(
        asset_id,
        observation,
        before_policy=(*unrelated, original_rule),
        after_policy=(*unrelated, rewritten_rule),
        before_outcome=PolicyOutcome(label="unsafe", rule_id=original_rule.rule_id),
        after_outcome=PolicyOutcome(label="safe"),
        transformation="rule_rewrite",
        before_policy_version=before_policy_version,
        after_policy_version=after_policy_version,
    )


def build_exemption_tasks(
    asset_id: str,
    observation: str,
    matching_rule: PolicyRule,
    exemption: str,
    *,
    unrelated_rules: Iterable[PolicyRule] = (),
    before_policy_version: str | None = None,
    after_policy_version: str | None = None,
) -> tuple[PolicyShiftRow, PolicyShiftRow]:
    """Create an unsafe-to-safe pair after adding an explicit rule exemption."""

    matching_rule = _require_policy_rule(matching_rule, field_name="matching_rule")
    if not isinstance(exemption, str) or not exemption.strip():
        raise ValueError("exemption must be a non-empty string")
    if exemption in matching_rule.exceptions:
        raise ValueError("exemption must add a new exception to the matching rule")
    unrelated = _normalize_policy(unrelated_rules, field_name="unrelated_rules")
    exempted_rule = matching_rule.model_copy(
        update={"exceptions": (*matching_rule.exceptions, exemption)}
    )
    return build_policy_shift_rows(
        asset_id,
        observation,
        before_policy=(*unrelated, matching_rule),
        after_policy=(*unrelated, exempted_rule),
        before_outcome=PolicyOutcome(label="unsafe", rule_id=matching_rule.rule_id),
        after_outcome=PolicyOutcome(label="safe"),
        transformation="exemption",
        before_policy_version=before_policy_version,
        after_policy_version=after_policy_version,
    )


def build_policy_shift_tasks(
    asset_id: str,
    observation: str,
    matching_rule: PolicyRule,
    *,
    legacy_policy_versions: bool = False,
) -> tuple[PolicyShiftRow, PolicyShiftRow]:
    """Return the legacy removal pair with the plan's fixed version names.

    By default this wrapper uses the same deterministic policy fingerprints as
    the newer builders.  Set ``legacy_policy_versions=True`` only when loading
    an older fixture that requires ``with-rule`` and ``without-rule``.
    """

    if legacy_policy_versions:
        return build_rule_removal_tasks(
            asset_id,
            observation,
            matching_rule,
            before_policy_version="with-rule",
            after_policy_version="without-rule",
        )
    return build_rule_removal_tasks(asset_id, observation, matching_rule)
