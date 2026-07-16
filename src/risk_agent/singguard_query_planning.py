"""Immutable query blueprints and deterministic release quota planning."""

from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from collections.abc import Mapping, Sequence
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
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class QueryBlueprint(BaseModel):
    """One immutable content anchor to be generated in a later pipeline stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    blueprint_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    active_rule_ids: tuple[str, ...] = Field(min_length=1)
    active_rule_titles: tuple[str, ...] = Field(min_length=1)
    intended_label: Literal["safe", "unsafe"]
    primary_rule_id: str | None = None
    primary_answer: str | None = None
    intended_answers: tuple[str, ...] = ()
    target_exception_rule_id: str | None = None
    target_exception: str | None = None
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
        if len(self.active_rule_titles) != len(self.active_rule_ids):
            raise ValueError("active rule IDs and titles must have equal length")
        if any(not title.strip() for title in self.active_rule_titles):
            raise ValueError("active rule titles must be non-blank")
        if len(self.active_rule_titles) != len(set(self.active_rule_titles)):
            raise ValueError("active rule titles must be unique")
        if len(self.intended_answers) != len(set(self.intended_answers)):
            raise ValueError("intended answers must be unique")
        if any(answer not in self.active_rule_titles for answer in self.intended_answers):
            raise ValueError("intended answers must be active rule titles")
        ordered_answers = tuple(
            title for title in self.active_rule_titles if title in self.intended_answers
        )
        if self.intended_answers != ordered_answers:
            raise ValueError("intended answers must follow active-policy order")
        if (self.target_exception_rule_id is None) != (self.target_exception is None):
            raise ValueError("exception target rule and text must be provided together")
        if self.target_exception_rule_id is not None:
            if self.intended_label != "safe" or self.difficulty != "exception":
                raise ValueError("exception targets require a safe exception blueprint")
            if self.target_exception_rule_id not in self.active_rule_ids:
                raise ValueError("exception target rule must belong to the active policy")
            if not self.target_exception.strip():
                raise ValueError("exception target text must be non-blank")
        if self.intended_label == "safe":
            if (
                self.primary_rule_id is not None
                or self.primary_answer is not None
                or self.intended_answers
            ):
                raise ValueError("safe blueprints cannot have intended answers")
        else:
            if self.primary_rule_id not in self.active_rule_ids:
                raise ValueError("unsafe blueprints require an active primary rule ID")
            primary_index = self.active_rule_ids.index(self.primary_rule_id)
            expected_primary = self.active_rule_titles[primary_index]
            if self.primary_answer != expected_primary:
                raise ValueError("primary answer must be the primary rule's exact title")
            if not self.intended_answers or self.primary_answer not in self.intended_answers:
                raise ValueError(
                    "unsafe blueprints require a primary answer among intended answers"
                )
        return self


_SUPPORTED_COUNTS = frozenset((100, 500, 2_000))
_BASE_COUNT = 2_000
_LABEL_QUOTAS = {"safe": 1_000, "unsafe": 1_000}
_SHAPE_QUOTAS = {"query": 1_400, "query_response": 600}
_THINKING_QUOTAS = {"fast": 1_400, "slow": 600}
_GOVERNED_SOURCE_QUOTAS = {
    "uci_sms_spam": 200,
    "uci_youtube_spam": 100,
    "nemotron_aegis_v2": 120,
    "civil_comments": 100,
    "amazon_esci": 80,
}
_SOURCE_FORM_COMPATIBILITY = {
    "uci_sms_spam": frozenset(
        ("short_ad", "private_message", "support_exchange", "social_post")
    ),
    "uci_youtube_spam": frozenset(
        ("comment", "social_post", "livestream_pitch", "short_ad")
    ),
    "nemotron_aegis_v2": frozenset(ContentForm.__args__),
    "civil_comments": frozenset(
        ("comment", "social_post", "livestream_pitch", "support_exchange")
    ),
    "amazon_esci": frozenset(
        (
            "product_listing",
            "search_or_neutral",
            "short_ad",
            "social_post",
            "livestream_pitch",
        )
    ),
}
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


def _apportion(weights: Mapping[T, int], count: int) -> dict[T, int]:
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise RuntimeError("internal quota weights must have a positive total")
    exact = {key: Fraction(value * count, total_weight) for key, value in weights.items()}
    allocated = {key: int(value) for key, value in exact.items()}
    remaining = count - sum(allocated.values())
    order = sorted(
        weights,
        key=lambda key: exact[key] - allocated[key],
        reverse=True,
    )
    for key in order[:remaining]:
        allocated[key] += 1
    if sum(allocated.values()) != count:
        raise RuntimeError("internal largest-remainder allocation failed")
    return allocated


def _largest_remainder(quotas: Mapping[T, int], count: int) -> dict[T, int]:
    """Scale 2000-row quotas exactly, breaking equal remainders by declaration order."""

    if sum(quotas.values()) != _BASE_COUNT:
        raise RuntimeError("internal quota table does not total 2000")
    return _apportion(quotas, count)


def _scale_partial_quotas(quotas: Mapping[T, int], count: int) -> dict[T, int]:
    exact = {key: Fraction(value * count, _BASE_COUNT) for key, value in quotas.items()}
    allocated = {key: int(value) for key, value in exact.items()}
    target_total = int(Fraction(sum(quotas.values()) * count, _BASE_COUNT))
    remaining = target_total - sum(allocated.values())
    order = sorted(
        quotas,
        key=lambda key: exact[key] - allocated[key],
        reverse=True,
    )
    for key in order[:remaining]:
        allocated[key] += 1
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


def _values_balanced_across_labels(
    quotas: Mapping[T, int], count: int, rng: random.Random
) -> dict[str, list[T]]:
    scaled = _largest_remainder(quotas, count)
    allocations = {
        "safe": {value: amount // 2 for value, amount in scaled.items()},
        "unsafe": {value: amount // 2 for value, amount in scaled.items()},
    }
    totals = {label: sum(values.values()) for label, values in allocations.items()}
    for value, amount in scaled.items():
        if amount % 2:
            label = min(("safe", "unsafe"), key=lambda item: (totals[item], item != "safe"))
            allocations[label][value] += 1
            totals[label] += 1
    expected_labels = _largest_remainder(_LABEL_QUOTAS, count)
    if totals != expected_labels:
        raise RuntimeError("internal label crossing could not satisfy exact marginals")
    pools = {
        label: [
            value
            for value, amount in label_allocations.items()
            for _ in range(amount)
        ]
        for label, label_allocations in allocations.items()
    }
    for pool in pools.values():
        rng.shuffle(pool)
    return pools


def _align_label_balanced_values(
    labels: Sequence[str], quotas: Mapping[T, int], count: int, rng: random.Random
) -> list[T]:
    pools = _values_balanced_across_labels(quotas, count, rng)
    offsets = {"safe": 0, "unsafe": 0}
    values: list[T] = []
    for label in labels:
        values.append(pools[label][offsets[label]])
        offsets[label] += 1
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
        if record.source not in _GOVERNED_SOURCE_QUOTAS:
            raise ValueError(f"unknown governed source {record.source!r}")
        key = (record.source, record.source_id)
        if key in keys:
            raise ValueError("duplicate seed source/source_id pair")
        keys.add(key)
        if record.content_hash in content_hashes:
            raise ValueError("duplicate seed content hash")
        content_hashes.add(record.content_hash)
        expected_hash = hashlib.sha256(record.text.encode("utf-8")).hexdigest()
        if record.content_hash != expected_hash:
            raise ValueError("seed content_hash must be lowercase SHA256 of exact text")


def _balanced_label_allocations(category_counts: Mapping[str, int]) -> dict[str, int]:
    safe_counts = {
        category: amount // 2 for category, amount in category_counts.items()
    }
    target_safe = sum(category_counts.values()) // 2
    remaining = target_safe - sum(safe_counts.values())
    for category, amount in category_counts.items():
        if remaining and amount % 2:
            safe_counts[category] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("internal source-label allocation failed")
    return safe_counts


def _label_thinking_cell_counts(
    category_counts: Mapping[str, int],
) -> dict[str, dict[tuple[str, str], int]]:
    safe_counts = _balanced_label_allocations(category_counts)
    total = sum(category_counts.values())
    target_fast = _apportion(_THINKING_QUOTAS, total)["fast"]
    exact_fast = {
        source: Fraction(amount * _THINKING_QUOTAS["fast"], _BASE_COUNT)
        for source, amount in category_counts.items()
    }
    fast_counts = {source: int(value) for source, value in exact_fast.items()}
    remaining_fast = target_fast - sum(fast_counts.values())
    fast_remainder_order = sorted(
        category_counts,
        key=lambda source: exact_fast[source] - fast_counts[source],
        reverse=True,
    )
    for source in fast_remainder_order[:remaining_fast]:
        fast_counts[source] += 1
    safe_fast_target = _apportion(_THINKING_QUOTAS, sum(safe_counts.values()))["fast"]
    safe_fast_values: dict[str, int] = {}
    safe_fast_bounds: dict[str, tuple[int, int]] = {}
    safe_fast_ideals: dict[str, Fraction] = {}
    for category, amount in category_counts.items():
        safe = safe_counts[category]
        fast = fast_counts[category]
        slow = amount - fast
        lower = max(0, safe - slow)
        upper = min(safe, fast)
        ideal = Fraction(safe * fast, amount) if amount else Fraction(0)
        safe_fast_values[category] = max(lower, min(int(ideal), upper))
        safe_fast_bounds[category] = (lower, upper)
        safe_fast_ideals[category] = ideal
    while sum(safe_fast_values.values()) < safe_fast_target:
        candidates = [
            category
            for category, (_, upper) in safe_fast_bounds.items()
            if safe_fast_values[category] < upper
        ]
        if not candidates:
            raise RuntimeError("internal label-thinking allocation is infeasible")
        category = max(
            candidates,
            key=lambda item: safe_fast_ideals[item] - safe_fast_values[item],
        )
        safe_fast_values[category] += 1
    while sum(safe_fast_values.values()) > safe_fast_target:
        candidates = [
            category
            for category, (lower, _) in safe_fast_bounds.items()
            if safe_fast_values[category] > lower
        ]
        if not candidates:
            raise RuntimeError("internal label-thinking allocation is infeasible")
        category = min(
            candidates,
            key=lambda item: safe_fast_ideals[item] - safe_fast_values[item],
        )
        safe_fast_values[category] -= 1

    result: dict[str, dict[tuple[str, str], int]] = {}
    for category, amount in category_counts.items():
        safe = safe_counts[category]
        unsafe = amount - safe
        fast = fast_counts[category]
        safe_fast = safe_fast_values[category]
        result[category] = {
            ("safe", "fast"): safe_fast,
            ("safe", "slow"): safe - safe_fast,
            ("unsafe", "fast"): fast - safe_fast,
            ("unsafe", "slow"): unsafe - (fast - safe_fast),
        }
        if any(value < 0 for value in result[category].values()):
            raise RuntimeError("internal label-thinking crossing produced a negative cell")
    return result


def _align_label_thinking_values(
    labels: Sequence[str],
    thinking_types: Sequence[str],
    quotas: Mapping[str, int],
    count: int,
    rng: random.Random,
) -> list[str]:
    category_counts = _largest_remainder(quotas, count)
    cell_counts = _label_thinking_cell_counts(category_counts)
    pools = {
        cell: [
            category
            for category, category_cells in cell_counts.items()
            for _ in range(category_cells[cell])
        ]
        for cell in (
            ("safe", "fast"),
            ("safe", "slow"),
            ("unsafe", "fast"),
            ("unsafe", "slow"),
        )
    }
    for pool in pools.values():
        rng.shuffle(pool)
    offsets = {cell: 0 for cell in pools}
    values: list[str] = []
    for label, thinking_type in zip(labels, thinking_types, strict=True):
        cell = (label, thinking_type)
        values.append(pools[cell][offsets[cell]])
        offsets[cell] += 1
    return values


def _allocate_source_forms(
    demands: Mapping[str, int], capacities: Mapping[str, int]
) -> dict[str, dict[str, int]]:
    source_node = ("terminal", "source")
    sink_node = ("terminal", "sink")
    adjacency: dict[tuple[str, str], list[tuple[str, str]]] = {}
    residual: dict[tuple[tuple[str, str], tuple[str, str]], int] = {}

    def add_edge(
        start: tuple[str, str], end: tuple[str, str], capacity: int
    ) -> None:
        adjacency.setdefault(start, []).append(end)
        adjacency.setdefault(end, []).append(start)
        residual[(start, end)] = capacity
        residual[(end, start)] = 0

    for source, demand in demands.items():
        source_demand_node = ("source", source)
        add_edge(source_node, source_demand_node, demand)
        for form in capacities:
            if form in _SOURCE_FORM_COMPATIBILITY[source]:
                add_edge(source_demand_node, ("form", form), demand)
    for form, capacity in capacities.items():
        add_edge(("form", form), sink_node, capacity)

    assigned = 0
    while True:
        parents: dict[tuple[str, str], tuple[str, str] | None] = {source_node: None}
        queue = deque((source_node,))
        while queue and sink_node not in parents:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in parents and residual[(node, neighbor)] > 0:
                    parents[neighbor] = node
                    queue.append(neighbor)
        if sink_node not in parents:
            break
        path_capacity = sum(demands.values())
        node = sink_node
        while parents[node] is not None:
            parent = parents[node]
            path_capacity = min(path_capacity, residual[(parent, node)])
            node = parent
        node = sink_node
        while parents[node] is not None:
            parent = parents[node]
            residual[(parent, node)] -= path_capacity
            residual[(node, parent)] += path_capacity
            node = parent
        assigned += path_capacity

    if assigned != sum(demands.values()):
        raise RuntimeError(
            "source/form compatibility cannot satisfy governed quota in stratum"
        )
    return {
        source: {
            form: demand - residual[(('source', source), ('form', form))]
            for form in capacities
            if form in _SOURCE_FORM_COMPATIBILITY[source]
        }
        for source, demand in demands.items()
    }


def _assign_source_refs(
    *,
    seed_records: tuple[SeedRecord, ...],
    count: int,
    labels: Sequence[str],
    thinking_types: Sequence[str],
    forms: Sequence[str],
    rng: random.Random,
) -> list[SourceRef | None]:
    scaled_quotas = _scale_partial_quotas(_GOVERNED_SOURCE_QUOTAS, count)
    grouped: dict[str, list[SeedRecord]] = {
        source: [] for source in _GOVERNED_SOURCE_QUOTAS
    }
    for record in seed_records:
        grouped[record.source].append(record)
    selected: dict[str, list[SeedRecord]] = {}
    for source, records in grouped.items():
        rng.shuffle(records)
        selected[source] = records[: scaled_quotas[source]]
    source_counts = {source: len(records) for source, records in selected.items()}
    cell_counts = _label_thinking_cell_counts(source_counts)
    refs: list[SourceRef | None] = [None] * count
    available = set(range(count))
    record_offsets = {source: 0 for source in selected}
    cells = (("safe", "fast"), ("safe", "slow"), ("unsafe", "fast"), ("unsafe", "slow"))
    for cell in cells:
        label, thinking_type = cell
        candidates = [
            index
            for index in available
            if labels[index] == label and thinking_types[index] == thinking_type
        ]
        capacities = {
            form: sum(forms[index] == form for index in candidates)
            for form in _FORM_QUOTAS
        }
        demands = {
            source: source_cells[cell] for source, source_cells in cell_counts.items()
        }
        allocation = _allocate_source_forms(demands, capacities)
        indices_by_form = {
            form: [index for index in candidates if forms[index] == form]
            for form in _FORM_QUOTAS
        }
        for form_indices in indices_by_form.values():
            rng.shuffle(form_indices)
        for source, form_counts in allocation.items():
            for form, amount in form_counts.items():
                for _ in range(amount):
                    index = indices_by_form[form].pop()
                    record = selected[source][record_offsets[source]]
                    record_offsets[source] += 1
                    refs[index] = SourceRef(
                        source=record.source,
                        source_id=record.source_id,
                        content_hash=record.content_hash,
                    )
                    available.remove(index)
    if any(record_offsets[source] != len(records) for source, records in selected.items()):
        raise RuntimeError("internal source assignment left records unused")
    return refs


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


def _balanced_schedule(
    values: Sequence[T], count: int, rng: random.Random, *, shortage_message: str
) -> list[T]:
    if count < len(values):
        raise ValueError(shortage_message)
    quotient, remainder = divmod(count, len(values))
    schedule = [
        value
        for index, value in enumerate(values)
        for _ in range(quotient + (index < remainder))
    ]
    rng.shuffle(schedule)
    return schedule


def _stable_id(
    namespace: str,
    *,
    semantic_fields: Mapping[str, object],
    policy_fingerprint: str,
    ordinal: int,
) -> str:
    payload = {
        "planner_contract": "singguard-query-blueprint-v2",
        "namespace": namespace,
        "ordinal": ordinal,
        "policy_fingerprint": policy_fingerprint,
        "semantics": semantic_fields,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
    return f"sg-{namespace}-{digest}"


def plan_blueprints(
    policies: tuple[ActivePolicy, ...],
    *,
    count: int,
    seed: int,
    seed_records: tuple[SeedRecord, ...],
) -> tuple[QueryBlueprint, ...]:
    """Plan one deterministic, quota-controlled SingGuard query release."""

    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count not in _SUPPORTED_COUNTS:
        raise ValueError("supported release sizes are 100, 500, and 2000")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    rule_ids, rule_owners = _validate_policies(policies)
    _validate_seeds(seed_records)
    policy_fingerprints = {
        policy.policy_id: hashlib.sha256(
            json.dumps(
                policy.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for policy in policies
    }

    rng = random.Random(seed)
    labels = _shuffled_values(_LABEL_QUOTAS, count, rng)
    shapes = _align_label_balanced_values(labels, _SHAPE_QUOTAS, count, rng)
    thinking_types = _align_label_balanced_values(
        labels, _THINKING_QUOTAS, count, rng
    )
    forms = _align_label_thinking_values(
        labels, thinking_types, _FORM_QUOTAS, count, rng
    )
    difficulties = _align_label_balanced_values(
        labels, _DIFFICULTY_QUOTAS, count, rng
    )
    tones = _align_label_balanced_values(labels, _TONE_QUOTAS, count, rng)
    noise_profiles = _align_label_balanced_values(
        labels, _NOISE_QUOTAS, count, rng
    )
    length_bins = _align_label_balanced_values(
        labels, _LENGTH_QUOTAS, count, rng
    )

    unsafe_count = labels.count("unsafe")
    primary_rules = _balanced_rule_schedule(rule_ids, unsafe_count, rng)
    primary_owner_offsets = {rule_id: 0 for rule_id in rule_ids}
    unsafe_policies: list[ActivePolicy] = []
    for rule_id in primary_rules:
        owners = rule_owners[rule_id]
        owner_offset = primary_owner_offsets[rule_id]
        unsafe_policies.append(owners[owner_offset % len(owners)])
        primary_owner_offsets[rule_id] = owner_offset + 1
    documented_exceptions: list[tuple[str, str]] = []
    seen_exceptions: set[tuple[str, str]] = set()
    for policy in policies:
        for rule in policy.rules:
            for exception in rule.exceptions:
                pair = (rule.rule_id, exception)
                if pair not in seen_exceptions:
                    documented_exceptions.append(pair)
                    seen_exceptions.add(pair)
    safe_exception_count = sum(
        label == "safe" and difficulty == "exception"
        for label, difficulty in zip(labels, difficulties, strict=True)
    )
    exception_schedule = (
        _balanced_schedule(
            documented_exceptions,
            safe_exception_count,
            rng,
            shortage_message=(
                "release does not contain enough safe exception rows to cover every "
                "documented exception"
            ),
        )
        if documented_exceptions
        else []
    )
    generic_safe_count = count - unsafe_count - (
        safe_exception_count if documented_exceptions else 0
    )
    safe_policies = _balanced_policy_schedule(policies, generic_safe_count, rng)

    source_refs = _assign_source_refs(
        seed_records=seed_records,
        count=count,
        labels=labels,
        thinking_types=thinking_types,
        forms=forms,
        rng=rng,
    )

    exception_owner_offsets = {pair: 0 for pair in documented_exceptions}
    rows: list[QueryBlueprint] = []
    unsafe_offset = 0
    safe_offset = 0
    generic_safe_offset = 0
    exception_offset = 0
    for index, label in enumerate(labels):
        if label == "unsafe":
            primary_rule_id = primary_rules[unsafe_offset]
            policy = unsafe_policies[unsafe_offset]
            unsafe_offset += 1
            primary_answer = next(
                rule.title for rule in policy.rules if rule.rule_id == primary_rule_id
            )
            intended_answers = (primary_answer,)
            target_exception_rule_id = None
            target_exception = None
        else:
            primary_rule_id = None
            primary_answer = None
            intended_answers = ()
            if difficulties[index] == "exception" and documented_exceptions:
                target_exception_rule_id, target_exception = exception_schedule[
                    exception_offset
                ]
                exception_offset += 1
                pair = (target_exception_rule_id, target_exception)
                owners = rule_owners[target_exception_rule_id]
                owner_offset = exception_owner_offsets[pair]
                policy = owners[owner_offset % len(owners)]
                exception_owner_offsets[pair] = owner_offset + 1
            else:
                target_exception_rule_id = None
                target_exception = None
                policy = safe_policies[generic_safe_offset]
                generic_safe_offset += 1
            safe_offset += 1
        active_rule_ids = tuple(rule.rule_id for rule in policy.rules)
        active_rule_titles = tuple(rule.title for rule in policy.rules)
        semantic_fields: dict[str, object] = {
            "policy_id": policy.policy_id,
            "active_rule_ids": active_rule_ids,
            "active_rule_titles": active_rule_titles,
            "intended_label": label,
            "primary_rule_id": primary_rule_id,
            "primary_answer": primary_answer,
            "intended_answers": intended_answers,
            "target_exception_rule_id": target_exception_rule_id,
            "target_exception": target_exception,
            "conversation_shape": shapes[index],
            "content_form": forms[index],
            "thinking_type": thinking_types[index],
            "difficulty": difficulties[index],
            "tone": tones[index],
            "noise_profile": noise_profiles[index],
            "length_bin": length_bins[index],
            "tool_capable": False,
            "source_ref": (
                source_refs[index].model_dump(mode="json")
                if source_refs[index] is not None
                else None
            ),
        }
        rows.append(
            QueryBlueprint(
                blueprint_id=_stable_id(
                    "blueprint",
                    semantic_fields=semantic_fields,
                    policy_fingerprint=policy_fingerprints[policy.policy_id],
                    ordinal=index,
                ),
                family_id=_stable_id(
                    "family",
                    semantic_fields=semantic_fields,
                    policy_fingerprint=policy_fingerprints[policy.policy_id],
                    ordinal=index,
                ),
                **semantic_fields,
            )
        )

    if unsafe_offset != unsafe_count or safe_offset != count - unsafe_count:
        raise RuntimeError("internal label assignment did not consume its schedules")
    if generic_safe_offset != generic_safe_count:
        raise RuntimeError("internal generic-safe assignment left policies unused")
    if documented_exceptions and exception_offset != safe_exception_count:
        raise RuntimeError("internal exception assignment left targets unused")
    return tuple(rows)
