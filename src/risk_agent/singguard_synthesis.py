"""Deterministic planning and policy compilation for SingGuard synthesis."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from risk_agent.contracts import PolicyRule
from risk_agent.singguard import (
    PolicyTransition,
    PolicyView,
    SingGuardAnnotation,
    ThinkingType,
)


_RULE_CATALOG_PATH = Path(__file__).resolve().parents[2] / "policies" / "singguard_rules_v1.yaml"

RISK_DOMAINS = (
    "deceptive_efficacy",
    "off_platform_solicitation",
    "fraud_scams",
    "sexual_exploitation",
    "violence_crime",
    "hate_harassment",
    "self_harm_danger",
    "cyber_privacy_agent",
)
TRANSITIONS: tuple[PolicyTransition, ...] = (
    "unsafe_to_unsafe",
    "unsafe_to_safe",
    "safe_to_unsafe",
    "safe_to_safe",
)
STYLE_WEIGHTS = {
    "short_ad": 20,
    "social_post": 15,
    "livestream_pitch": 15,
    "product_listing": 15,
    "comment_or_review": 10,
    "sms_or_private_message": 10,
    "support_dialogue": 10,
    "neutral_information": 5,
}
_TONES = (
    "formal",
    "colloquial",
    "promotional",
    "urgent",
    "testimonial",
    "technical",
    "humorous",
    "neutral",
)
_DIFFICULTIES = ("explicit", "paraphrased", "implicit", "borderline_exception")
_LENGTH_WEIGHTS = {"headline": 15, "short": 35, "medium": 35, "long_dialogue": 15}
_NOISE_WEIGHTS = {"clean": 70, "light": 20, "obfuscated": 10}
_TRANSFORMATIONS: dict[PolicyTransition, tuple[str, ...]] = {
    "unsafe_to_unsafe": ("rewrite", "merge"),
    "unsafe_to_safe": ("remove", "narrow", "exception"),
    "safe_to_unsafe": ("add", "broaden"),
    "safe_to_safe": ("rewrite", "merge"),
}
_EXPECTED_LABELS: dict[PolicyTransition, tuple[str, str]] = {
    "unsafe_to_unsafe": ("unsafe", "unsafe"),
    "unsafe_to_safe": ("unsafe", "safe"),
    "safe_to_unsafe": ("safe", "unsafe"),
    "safe_to_safe": ("safe", "safe"),
}


class CatalogRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    exceptions: tuple[str, ...] = ()
    subtypes: tuple[str, ...] = Field(min_length=1)
    unsafe_facts: tuple[str, ...] = Field(min_length=1)
    safe_facts: tuple[str, ...] = Field(min_length=1)

    def as_policy_rule(self) -> PolicyRule:
        return PolicyRule(
            rule_id=self.rule_id,
            title=self.title,
            text=self.text,
            exceptions=self.exceptions,
        )


class AnchorBlueprint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    anchor_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    risk_domain: str = Field(min_length=1)
    subtype: str = Field(min_length=1)
    input_style: str = Field(min_length=1)
    tone: str = Field(min_length=1)
    length_bin: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    conversation_type: Literal["query", "query_response"]
    transition: PolicyTransition
    transformation: Literal["add", "remove", "narrow", "broaden", "rewrite", "merge", "exception"]
    policy_size: int = Field(ge=3, le=8)
    before_thinking_type: ThinkingType
    after_thinking_type: ThinkingType
    noise_profile: str = Field(min_length=1)
    intended_facts: tuple[str, ...] = Field(min_length=1)
    use_open_seed: bool
    source_id: str | None = None


class PolicyPair(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    before: PolicyView
    after: PolicyView
    before_annotation: SingGuardAnnotation
    after_annotation: SingGuardAnnotation

    @model_validator(mode="after")
    def validate_transition(self) -> "PolicyPair":
        expected = _EXPECTED_LABELS[self.before.transition]
        if self.after.transition != self.before.transition:
            raise ValueError("policy pair transitions must match")
        actual = (self.before_annotation.label, self.after_annotation.label)
        if actual != expected:
            raise ValueError("policy pair labels do not match the ordered transition")
        return self


def load_rule_catalog(path: Path = _RULE_CATALOG_PATH) -> tuple[CatalogRule, ...]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        raise ValueError("cannot read SingGuard rule catalog") from None
    if not isinstance(payload, dict) or payload.get("version") != "singguard-rules-v1":
        raise ValueError("unsupported SingGuard rule catalog")
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list):
        raise ValueError("SingGuard rule catalog must contain a rules list")
    rules = tuple(CatalogRule.model_validate(item) for item in raw_rules)
    if {rule.domain for rule in rules} != set(RISK_DOMAINS):
        raise ValueError("SingGuard rule catalog must contain exactly the required domains")
    if len({rule.rule_id for rule in rules}) != len(rules):
        raise ValueError("SingGuard rule catalog contains duplicate rule IDs")
    if len({rule.title for rule in rules}) != len(rules):
        raise ValueError("SingGuard rule catalog contains duplicate rule titles")
    return rules


def _allocate(count: int, weights: dict[str, int]) -> list[str]:
    total = sum(weights.values())
    raw = {key: count * weight / total for key, weight in weights.items()}
    allocated = {key: int(value) for key, value in raw.items()}
    remainder = count - sum(allocated.values())
    order = sorted(weights, key=lambda key: (-(raw[key] - allocated[key]), list(weights).index(key)))
    for key in order[:remainder]:
        allocated[key] += 1
    return [key for key in weights for _ in range(allocated[key])]


def _shuffled(values: list[str], rng: random.Random) -> list[str]:
    result = list(values)
    rng.shuffle(result)
    return result


def plan_blueprints(count: int, *, seed: int) -> tuple[AnchorBlueprint, ...]:
    """Create a complete deterministic quota plan before provider calls."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 32:
        raise ValueError("anchor count must be an integer of at least 32")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (2**63 - 1):
        raise ValueError("seed must be an integer from zero to 2^63-1")
    catalog = load_rule_catalog()
    by_domain = {rule.domain: rule for rule in catalog}
    rng = random.Random(seed)

    domains = _shuffled(_allocate(count, {item: 1 for item in RISK_DOMAINS}), rng)
    styles = _shuffled(_allocate(count, STYLE_WEIGHTS), rng)
    transitions = _shuffled(_allocate(count, {item: 1 for item in TRANSITIONS}), rng)
    conversations = _shuffled(_allocate(count, {"query": 70, "query_response": 30}), rng)
    tones = _shuffled(_allocate(count, {item: 1 for item in _TONES}), rng)
    lengths = _shuffled(_allocate(count, _LENGTH_WEIGHTS), rng)
    difficulties = _shuffled(_allocate(count, {item: 1 for item in _DIFFICULTIES}), rng)
    noise = _shuffled(_allocate(count, _NOISE_WEIGHTS), rng)
    sources = _shuffled(_allocate(count, {"synthetic": 70, "open": 30}), rng)
    modes = _shuffled(_allocate(count * 2, {"fast": 70, "slow": 30}), rng)
    policy_sizes = _shuffled(_allocate(count, {str(size): 1 for size in range(3, 9)}), rng)

    transformation_indexes = {transition: 0 for transition in TRANSITIONS}
    domain_indexes = {domain: 0 for domain in RISK_DOMAINS}
    blueprints: list[AnchorBlueprint] = []
    for index in range(count):
        domain = domains[index]
        transition = transitions[index]
        rule = by_domain[domain]
        domain_index = domain_indexes[domain]
        domain_indexes[domain] += 1
        subtype = rule.subtypes[domain_index % len(rule.subtypes)]
        transformations = _TRANSFORMATIONS[transition]
        transformation_index = transformation_indexes[transition]
        transformation_indexes[transition] += 1
        transformation = transformations[transformation_index % len(transformations)]
        use_safe_fact = transition == "safe_to_safe" or transformation in {"narrow", "exception"}
        facts = rule.safe_facts if use_safe_fact else rule.unsafe_facts
        fact = facts[domain_index % len(facts)]
        difficulty = "implicit" if transformation == "broaden" else difficulties[index]
        anchor_id = f"sg-{seed:016x}-{index:05d}"
        blueprints.append(
            AnchorBlueprint(
                anchor_id=anchor_id,
                family_id=anchor_id,
                risk_domain=domain,
                subtype=subtype,
                input_style=styles[index],
                tone=tones[index],
                length_bin=lengths[index],
                difficulty=difficulty,
                conversation_type=conversations[index],
                transition=transition,
                transformation=transformation,
                policy_size=int(policy_sizes[index]),
                before_thinking_type=modes[index * 2],
                after_thinking_type=modes[index * 2 + 1],
                noise_profile=noise[index],
                intended_facts=(fact,),
                use_open_seed=sources[index] == "open",
            )
        )
    return tuple(blueprints)


def _stable_offset(value: str, modulus: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def _variant(rule: PolicyRule, kind: str, *, stage: str) -> PolicyRule:
    if kind == "narrow" and stage == "after":
        return rule.model_copy(
            update={
                "text": rule.text + " This narrowed version applies only to explicit first-person directives or guarantees.",
                "exceptions": (*rule.exceptions, "Indirect, quoted, or neutral discussion is outside this narrowed version."),
            }
        )
    if kind == "exception" and stage == "after":
        return rule.model_copy(
            update={
                "exceptions": (*rule.exceptions, "The specific benign or support context in this policy version is allowed."),
            }
        )
    if kind == "rewrite" and stage == "after":
        return rule.model_copy(update={"text": "This equivalent wording prohibits the same conduct: " + rule.text})
    if kind == "merge" and stage == "after":
        return rule.model_copy(update={"text": rule.text + " This consolidated rule also covers closely related attempts."})
    if kind == "broaden" and stage == "before":
        return rule.model_copy(
            update={
                "text": rule.text + " This earlier narrow version applies only when the prohibited intent is stated explicitly."
            }
        )
    return rule


def compile_policy_pair(
    blueprint: AnchorBlueprint,
    *,
    catalog: tuple[CatalogRule, ...] | None = None,
) -> PolicyPair:
    """Compile ordered before/after active policies and local label oracles."""

    catalog = catalog or load_rule_catalog()
    target_catalog = next(rule for rule in catalog if rule.domain == blueprint.risk_domain)
    target = target_catalog.as_policy_rule()
    distractors = [rule.as_policy_rule() for rule in catalog if rule.rule_id != target.rule_id]
    offset = _stable_offset(blueprint.anchor_id, len(distractors))
    distractors = distractors[offset:] + distractors[:offset]
    before_label, after_label = _EXPECTED_LABELS[blueprint.transition]

    def build(stage: Literal["before", "after"], label: str) -> tuple[tuple[PolicyRule, ...], str | None]:
        include_target = label == "unsafe"
        if blueprint.transformation in {"narrow", "exception", "broaden"}:
            include_target = True
        if blueprint.transition == "safe_to_safe":
            include_target = False
        stage_target = _variant(target, blueprint.transformation, stage=stage)
        rules = ([stage_target] if include_target else []) + distractors
        active = tuple(rules[: blueprint.policy_size])
        answer = stage_target.title if label == "unsafe" else None
        if label == "unsafe" and stage_target.rule_id not in {rule.rule_id for rule in active}:
            raise ValueError("unsafe policy view must contain the target rule")
        return active, answer

    before_rules, before_answer = build("before", before_label)
    after_rules, after_answer = build("after", after_label)
    transition = blueprint.transition
    return PolicyPair(
        before=PolicyView(
            view_id=f"{blueprint.anchor_id}-before",
            active_policy=before_rules,
            style="full",
            transition=transition,
            stage="before",
        ),
        after=PolicyView(
            view_id=f"{blueprint.anchor_id}-after",
            active_policy=after_rules,
            style="rewritten" if blueprint.transformation in {"narrow", "broaden", "rewrite", "merge", "exception"} else "full",
            transition=transition,
            stage="after",
        ),
        before_annotation=SingGuardAnnotation(label=before_label, rule_title=before_answer),
        after_annotation=SingGuardAnnotation(label=after_label, rule_title=after_answer),
    )
