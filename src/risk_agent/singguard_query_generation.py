"""Immutable query blueprints and deterministic release quota planning."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from fractions import Fraction
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from risk_agent.contracts import PolicyRule
from risk_agent.singguard import ActivePolicy
from risk_agent.singguard_sources import SeedRecord


ThinkingType = Literal["fast", "slow"]
ConversationShape = Literal["query", "query_response"]
ContentForm = Literal[
    "short_ad",
    "social_post",
    "livestream_pitch",
    "product_listing",
    "comment",
    "private_message",
    "support_exchange",
    "search_or_neutral",
]
Tone = Literal[
    "formal",
    "colloquial",
    "promotional",
    "urgent",
    "testimonial",
    "technical",
    "humorous",
    "neutral",
]
LengthBin = Literal["headline", "short", "medium", "long"]
Difficulty = Literal["explicit", "paraphrased", "implicit", "exception"]
NoiseProfile = Literal["none", "spelling", "emoji", "punctuation", "obfuscation"]


class SourceRef(BaseModel):
    """Minimal provenance for one governed open-data guidance seed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    content_hash: str = Field(min_length=64, max_length=64)


class QueryBlueprint(BaseModel):
    """One immutable content anchor to be generated in a later pipeline stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    blueprint_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    active_rule_ids: tuple[str, ...] = Field(min_length=1)
    intended_label: Literal["safe", "unsafe"]
    primary_answer: str | None = None
    intended_answers: tuple[str, ...] = ()
    conversation_shape: ConversationShape
    content_form: ContentForm
    thinking_type: ThinkingType
    difficulty: Difficulty
    tone: Tone
    noise_profile: NoiseProfile
    length_bin: LengthBin
    tool_capable: bool
    source_ref: SourceRef | None = None

    @model_validator(mode="after")
    def validate_intended_decision(self) -> "QueryBlueprint":
        if not self.blueprint_id.strip() or not self.family_id.strip():
            raise ValueError("blueprint and family IDs must be non-blank")
        if self.blueprint_id == self.family_id:
            raise ValueError("blueprint and family IDs must use distinct namespaces")
        if not self.policy_id.strip():
            raise ValueError("policy ID must be non-blank")
        if any(not rule_id.strip() for rule_id in self.active_rule_ids):
            raise ValueError("active rule IDs must be non-blank")
        if len(self.active_rule_ids) != len(set(self.active_rule_ids)):
            raise ValueError("active rule IDs must be unique")
        if len(self.intended_answers) != len(set(self.intended_answers)):
            raise ValueError("intended answers must be unique")
        if any(answer not in self.active_rule_ids for answer in self.intended_answers):
            raise ValueError("intended answers must be active rule IDs")
        ordered_answers = tuple(
            rule_id for rule_id in self.active_rule_ids if rule_id in self.intended_answers
        )
        if self.intended_answers != ordered_answers:
            raise ValueError("intended answers must follow active-policy order")
        if self.intended_label == "safe":
            if self.primary_answer is not None or self.intended_answers:
                raise ValueError("safe blueprints cannot have intended answers")
        elif (
            self.primary_answer is None
            or not self.intended_answers
            or self.primary_answer not in self.intended_answers
        ):
            raise ValueError(
                "unsafe blueprints require a primary answer among intended answers"
            )
        return self


_SUPPORTED_COUNTS = frozenset((100, 500, 2_000))
_BASE_COUNT = 2_000
_LABEL_QUOTAS = {"safe": 1_000, "unsafe": 1_000}
_SHAPE_QUOTAS = {"query": 1_400, "query_response": 600}
_THINKING_QUOTAS = {"fast": 1_400, "slow": 600}
_SOURCE_QUOTAS = {"synthetic": 1_400, "governed": 600}
_FORM_QUOTAS = {
    "short_ad": 400,
    "social_post": 300,
    "livestream_pitch": 300,
    "product_listing": 300,
    "comment": 200,
    "private_message": 200,
    "support_exchange": 200,
    "search_or_neutral": 100,
}
_DIFFICULTY_QUOTAS = {
    "explicit": 500,
    "paraphrased": 600,
    "implicit": 400,
    "exception": 500,
}
_TONE_QUOTAS = {
    "formal": 250,
    "colloquial": 250,
    "promotional": 250,
    "urgent": 250,
    "testimonial": 250,
    "technical": 250,
    "humorous": 250,
    "neutral": 250,
}
_NOISE_QUOTAS = {
    "none": 400,
    "spelling": 400,
    "emoji": 400,
    "punctuation": 400,
    "obfuscation": 400,
}
_LENGTH_QUOTAS = {"headline": 500, "short": 500, "medium": 500, "long": 500}


T = TypeVar("T")


def _largest_remainder(quotas: Mapping[T, int], count: int) -> dict[T, int]:
    """Scale 2000-row quotas exactly, breaking equal remainders by declaration order."""

    if sum(quotas.values()) != _BASE_COUNT:
        raise RuntimeError("internal quota table does not total 2000")
    exact = {key: Fraction(value * count, _BASE_COUNT) for key, value in quotas.items()}
    allocated = {key: int(value) for key, value in exact.items()}
    remaining = count - sum(allocated.values())
    order = sorted(
        quotas,
        key=lambda key: exact[key] - allocated[key],
        reverse=True,
    )
    for key in order[:remaining]:
        allocated[key] += 1
    if sum(allocated.values()) != count:
        raise RuntimeError("internal largest-remainder allocation failed")
    return allocated


def _shuffled_values(
    quotas: Mapping[T, int], count: int, rng: random.Random
) -> list[T]:
    values = [
        value
        for value, allocation in _largest_remainder(quotas, count).items()
        for _ in range(allocation)
    ]
    if len(values) != count:
        raise RuntimeError("internal quota expansion produced the wrong row count")
    rng.shuffle(values)
    return values


def _validate_policies(
    policies: tuple[ActivePolicy, ...],
) -> tuple[tuple[str, ...], dict[str, tuple[ActivePolicy, ...]]]:
    if not policies:
        raise ValueError("at least one active policy is required")
    if any(not isinstance(policy, ActivePolicy) for policy in policies):
        raise TypeError("policies must contain only ActivePolicy values")
    policy_ids = [policy.policy_id for policy in policies]
    if any(not policy_id.strip() for policy_id in policy_ids):
        raise ValueError("active policy IDs must be non-blank")
    if len(policy_ids) != len(set(policy_ids)):
        raise ValueError("duplicate policy IDs are not allowed")

    definitions: dict[str, PolicyRule] = {}
    owners: dict[str, list[ActivePolicy]] = {}
    ordered_rule_ids: list[str] = []
    for policy in policies:
        if not policy.rules:
            raise ValueError(f"active policy {policy.policy_id!r} has no rules")
        local_ids: set[str] = set()
        for rule in policy.rules:
            if not isinstance(rule, PolicyRule):
                raise TypeError("active policies must contain only PolicyRule values")
            if not rule.rule_id.strip() or not rule.title.strip() or not rule.text.strip():
                raise ValueError("active policy rule fields must be non-blank")
            if rule.rule_id in local_ids:
                raise ValueError("active policy contains duplicate rule IDs")
            local_ids.add(rule.rule_id)
            existing = definitions.get(rule.rule_id)
            if existing is not None and existing != rule:
                raise ValueError(f"inconsistent rule ID {rule.rule_id!r} across policies")
            if existing is None:
                definitions[rule.rule_id] = rule
                ordered_rule_ids.append(rule.rule_id)
            owners.setdefault(rule.rule_id, []).append(policy)
    if not ordered_rule_ids:
        raise ValueError("active policies must provide at least one rule")
    return tuple(ordered_rule_ids), {
        rule_id: tuple(rule_owners) for rule_id, rule_owners in owners.items()
    }


def _validate_seeds(seed_records: tuple[SeedRecord, ...]) -> None:
    if any(not isinstance(record, SeedRecord) for record in seed_records):
        raise TypeError("seed_records must contain only SeedRecord values")
    keys: set[tuple[str, str]] = set()
    content_hashes: set[str] = set()
    for record in seed_records:
        key = (record.source, record.source_id)
        if key in keys:
            raise ValueError("duplicate seed source/source_id pair")
        keys.add(key)
        if record.content_hash in content_hashes:
            raise ValueError("duplicate seed content hash")
        content_hashes.add(record.content_hash)


def _balanced_rule_schedule(
    rule_ids: tuple[str, ...], count: int, rng: random.Random
) -> list[str]:
    if count < len(rule_ids):
        raise ValueError(
            "release does not contain enough unsafe rows to cover every active rule"
        )
    quotient, remainder = divmod(count, len(rule_ids))
    schedule = [
        rule_id
        for index, rule_id in enumerate(rule_ids)
        for _ in range(quotient + (index < remainder))
    ]
    if len(schedule) != count:
        raise RuntimeError("internal rule balancing produced the wrong row count")
    rng.shuffle(schedule)
    return schedule


def _balanced_policy_schedule(
    policies: tuple[ActivePolicy, ...], count: int, rng: random.Random
) -> list[ActivePolicy]:
    quotient, remainder = divmod(count, len(policies))
    schedule = [
        policy
        for index, policy in enumerate(policies)
        for _ in range(quotient + (index < remainder))
    ]
    rng.shuffle(schedule)
    return schedule


def _plan_fingerprint(
    policies: tuple[ActivePolicy, ...], seed_records: tuple[SeedRecord, ...]
) -> str:
    payload = {
        "policies": [policy.model_dump(mode="json") for policy in policies],
        "seeds": [
            {
                "source": record.source,
                "source_id": record.source_id,
                "content_hash": record.content_hash,
            }
            for record in seed_records
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stable_id(
    namespace: str,
    *,
    plan_fingerprint: str,
    seed: int,
    count: int,
    index: int,
) -> str:
    payload = (
        f"singguard-query-v1\0{namespace}\0{plan_fingerprint}\0{seed}\0{count}\0{index}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"sg-{namespace}-{digest}"


def plan_blueprints(
    policies: tuple[ActivePolicy, ...],
    *,
    count: int,
    seed: int,
    seed_records: tuple[SeedRecord, ...],
) -> tuple[QueryBlueprint, ...]:
    """Plan one deterministic, quota-controlled SingGuard query release."""

    if count not in _SUPPORTED_COUNTS:
        raise ValueError("supported release sizes are 100, 500, and 2000")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    rule_ids, rule_owners = _validate_policies(policies)
    _validate_seeds(seed_records)
    plan_fingerprint = _plan_fingerprint(policies, seed_records)

    rng = random.Random(seed)
    labels = _shuffled_values(_LABEL_QUOTAS, count, rng)
    shapes = _shuffled_values(_SHAPE_QUOTAS, count, rng)
    thinking_types = _shuffled_values(_THINKING_QUOTAS, count, rng)
    forms = _shuffled_values(_FORM_QUOTAS, count, rng)
    difficulties = _shuffled_values(_DIFFICULTY_QUOTAS, count, rng)
    tones = _shuffled_values(_TONE_QUOTAS, count, rng)
    noise_profiles = _shuffled_values(_NOISE_QUOTAS, count, rng)
    length_bins = _shuffled_values(_LENGTH_QUOTAS, count, rng)

    unsafe_count = labels.count("unsafe")
    primary_rules = _balanced_rule_schedule(rule_ids, unsafe_count, rng)
    safe_policies = _balanced_policy_schedule(policies, count - unsafe_count, rng)

    source_quota = _largest_remainder(_SOURCE_QUOTAS, count)["governed"]
    governed_count = min(source_quota, len(seed_records))
    selected_seeds = list(seed_records)
    rng.shuffle(selected_seeds)
    selected_seeds = selected_seeds[:governed_count]
    source_refs: list[SourceRef | None] = [
        SourceRef(
            source=record.source,
            source_id=record.source_id,
            content_hash=record.content_hash,
        )
        for record in selected_seeds
    ]
    source_refs.extend([None] * (count - governed_count))
    rng.shuffle(source_refs)

    primary_owner_offsets = {rule_id: 0 for rule_id in rule_ids}
    rows: list[QueryBlueprint] = []
    unsafe_offset = 0
    safe_offset = 0
    for index, label in enumerate(labels):
        if label == "unsafe":
            primary_answer = primary_rules[unsafe_offset]
            unsafe_offset += 1
            owners = rule_owners[primary_answer]
            owner_offset = primary_owner_offsets[primary_answer]
            policy = owners[owner_offset % len(owners)]
            primary_owner_offsets[primary_answer] = owner_offset + 1
            intended_answers = (primary_answer,)
        else:
            primary_answer = None
            intended_answers = ()
            policy = safe_policies[safe_offset]
            safe_offset += 1
        active_rule_ids = tuple(rule.rule_id for rule in policy.rules)
        rows.append(
            QueryBlueprint(
                blueprint_id=_stable_id(
                    "blueprint",
                    plan_fingerprint=plan_fingerprint,
                    seed=seed,
                    count=count,
                    index=index,
                ),
                family_id=_stable_id(
                    "family",
                    plan_fingerprint=plan_fingerprint,
                    seed=seed,
                    count=count,
                    index=index,
                ),
                policy_id=policy.policy_id,
                active_rule_ids=active_rule_ids,
                intended_label=label,
                primary_answer=primary_answer,
                intended_answers=intended_answers,
                conversation_shape=shapes[index],
                content_form=forms[index],
                thinking_type=thinking_types[index],
                difficulty=difficulties[index],
                tone=tones[index],
                noise_profile=noise_profiles[index],
                length_bin=length_bins[index],
                tool_capable=thinking_types[index] == "slow",
                source_ref=source_refs[index],
            )
        )

    if unsafe_offset != unsafe_count or safe_offset != count - unsafe_count:
        raise RuntimeError("internal label assignment did not consume its schedules")
    return tuple(rows)
