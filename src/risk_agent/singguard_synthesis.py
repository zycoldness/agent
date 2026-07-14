"""Deterministic planning and policy compilation for SingGuard synthesis."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from risk_agent.contracts import PolicyRule
from risk_agent.singguard import (
    ContentSample,
    PolicyTransition,
    PolicyView,
    RuleCheck,
    SingGuardAnnotation,
    SingGuardExample,
    ThinkingType,
)
from risk_agent.singguard_prompts import load_prompt, prompt_sha256
from risk_agent.teacher import (
    Teacher,
    TeacherBudget,
    TeacherBudgetExceeded,
    TeacherRequestError,
    TeacherUsage,
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


class GeneratedContent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1)
    response: str | None = None
    style: str = Field(min_length=1)
    risk_cues: tuple[str, ...] = ()
    benign_cues: tuple[str, ...] = ()


class VerifiedView(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    opaque_id: str = Field(min_length=1)
    label: Literal["safe", "unsafe"]
    rule_title: str | None = None
    evidence_quote: str | None = None
    confidence: float = Field(ge=0, le=1)
    ambiguous: bool
    summary: str | None = None
    checks: tuple[RuleCheck, ...] = ()

    @model_validator(mode="after")
    def validate_decision_fields(self) -> "VerifiedView":
        if self.label == "safe" and (
            self.rule_title is not None or self.evidence_quote is not None
        ):
            raise ValueError("safe verifier view must not name a rule or unsafe evidence")
        if self.label == "unsafe" and (
            not self.rule_title or not self.evidence_quote
        ):
            raise ValueError("unsafe verifier view requires a rule and literal evidence")
        return self


class VerifierResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    views: tuple[VerifiedView, ...] = Field(min_length=2, max_length=2)
    style: str = Field(min_length=1)
    naturalness: int = Field(ge=1, le=5)
    template_like: bool
    issues: tuple[str, ...] = ()

    @model_validator(mode="after")
    def reject_duplicate_views(self) -> "VerifierResult":
        ids = [view.opaque_id for view in self.views]
        if len(ids) != len(set(ids)):
            raise ValueError("verifier response repeats an opaque view ID")
        return self


GENERATOR_RESPONSE_SCHEMA = GeneratedContent.model_json_schema()
VERIFIER_RESPONSE_SCHEMA = VerifierResult.model_json_schema()


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


def build_generator_request(
    blueprint: AnchorBlueprint,
    *,
    seed_text: str | None = None,
) -> dict[str, object]:
    """Build a content-only request with no final label or policy transition."""

    constraints: dict[str, object] = {
        "risk_domain": blueprint.risk_domain,
        "subtype": blueprint.subtype,
        "platform_style": blueprint.input_style,
        "tone": blueprint.tone,
        "length_bin": blueprint.length_bin,
        "difficulty": blueprint.difficulty,
        "conversation_type": blueprint.conversation_type,
        "noise_profile": blueprint.noise_profile,
        "intended_facts": list(blueprint.intended_facts),
    }
    request: dict[str, object] = {
        "instruction": load_prompt("generator"),
        "prompt_version": "singguard_generator_v1",
        "prompt_sha256": prompt_sha256("generator"),
        "constraints": constraints,
    }
    if seed_text is not None:
        request["public_seed_text"] = seed_text
    return request


def generate_content(
    blueprint: AnchorBlueprint,
    teacher: Teacher,
    *,
    seed_text: str | None = None,
) -> tuple[GeneratedContent, TeacherUsage]:
    """Ask one structured teacher to realize a blueprint without classifying it."""

    reply = teacher.generate(build_generator_request(blueprint, seed_text=seed_text))
    try:
        content = GeneratedContent.model_validate(reply.payload)
    except Exception:
        raise ValueError("generator response is not valid SingGuard content") from None
    if content.style != blueprint.input_style:
        raise ValueError("generator response does not match the planned style")
    if blueprint.conversation_type == "query" and content.response is not None:
        raise ValueError("query-only blueprint must not contain a response")
    if blueprint.conversation_type == "query_response" and not content.response:
        raise ValueError("query-response blueprint requires a response")
    return content, reply.usage


def _opaque_id(anchor_id: str, stage: str, shuffle_seed: int) -> str:
    payload = f"{anchor_id}:{stage}:{shuffle_seed}".encode("utf-8")
    return "view-" + hashlib.sha256(payload).hexdigest()[:16]


def build_verifier_request(
    blueprint: AnchorBlueprint,
    pair: PolicyPair,
    content: GeneratedContent,
    *,
    shuffle_seed: int,
) -> tuple[dict[str, object], dict[str, Literal["before", "after"]]]:
    """Build a blind request and keep its opaque-to-stage map local."""

    if isinstance(shuffle_seed, bool) or not isinstance(shuffle_seed, int):
        raise ValueError("shuffle_seed must be an integer")
    view_specs: list[tuple[str, Literal["before", "after"], PolicyView, ThinkingType]] = []
    for stage, view, mode in (
        ("before", pair.before, blueprint.before_thinking_type),
        ("after", pair.after, blueprint.after_thinking_type),
    ):
        opaque = _opaque_id(blueprint.anchor_id, stage, shuffle_seed)
        view_specs.append((opaque, stage, view, mode))
    random.Random(shuffle_seed).shuffle(view_specs)
    mapping = {opaque: stage for opaque, stage, _view, _mode in view_specs}
    policy_views = [
        {
            "opaque_id": opaque,
            "thinking_type": mode,
            "active_policy": [
                rule.model_dump(mode="json") for rule in view.active_policy
            ],
        }
        for opaque, _stage, view, mode in view_specs
    ]
    request: dict[str, object] = {
        "instruction": load_prompt("verifier"),
        "prompt_version": "singguard_verifier_v1",
        "prompt_sha256": prompt_sha256("verifier"),
        "content": {"query": content.query, "response": content.response},
        "declared_style": blueprint.input_style,
        "policy_views": policy_views,
    }
    return request, mapping


def verify_content(
    blueprint: AnchorBlueprint,
    pair: PolicyPair,
    content: GeneratedContent,
    teacher: Teacher,
    *,
    shuffle_seed: int,
) -> tuple[VerifierResult, dict[str, Literal["before", "after"]], TeacherUsage]:
    """Ask a stateless teacher to judge opaque policy views independently."""

    request, mapping = build_verifier_request(
        blueprint,
        pair,
        content,
        shuffle_seed=shuffle_seed,
    )
    reply = teacher.generate(request)
    try:
        verdict = VerifierResult.model_validate(reply.payload)
    except Exception:
        raise ValueError("verifier response is not valid SingGuard verification") from None
    if {view.opaque_id for view in verdict.views} != set(mapping):
        raise ValueError("verifier response does not cover the requested opaque views")
    stage_views = {"before": pair.before, "after": pair.after}
    for verified in verdict.views:
        policy = stage_views[mapping[verified.opaque_id]]
        active_titles = {rule.title for rule in policy.active_policy}
        if verified.rule_title is not None and verified.rule_title not in active_titles:
            raise ValueError("verifier rule title must belong to the active policy")
    return verdict, mapping, reply.usage


def examples_from_verifier(
    blueprint: AnchorBlueprint,
    pair: PolicyPair,
    content: GeneratedContent,
    verdict: VerifierResult,
    opaque_to_stage: dict[str, Literal["before", "after"]],
) -> tuple[SingGuardExample, SingGuardExample]:
    """Convert only exact oracle agreement into canonical source examples."""

    by_stage = {
        opaque_to_stage[view.opaque_id]: view
        for view in verdict.views
        if view.opaque_id in opaque_to_stage
    }
    if set(by_stage) != {"before", "after"}:
        raise ValueError("verifier result does not map to both policy stages")
    content_sample = ContentSample(
        sample_id=blueprint.anchor_id,
        split_group=blueprint.family_id,
        query=content.query,
        response=content.response,
    )
    examples: list[SingGuardExample] = []
    for stage, policy, oracle, mode in (
        ("before", pair.before, pair.before_annotation, blueprint.before_thinking_type),
        ("after", pair.after, pair.after_annotation, blueprint.after_thinking_type),
    ):
        verified = by_stage[stage]
        if (verified.label, verified.rule_title) != (oracle.label, oracle.rule_title):
            raise ValueError("verifier disagrees with the local policy oracle")
        annotation = oracle
        if mode == "slow":
            annotation = SingGuardAnnotation(
                label=oracle.label,
                rule_title=oracle.rule_title,
                summary=verified.summary,
                checks=verified.checks,
            )
        examples.append(
            SingGuardExample(
                content=content_sample,
                policy=policy,
                annotation=annotation,
                thinking_type=mode,
            )
        )
    return examples[0], examples[1]


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if isinstance(row, BaseModel):
                row = row.model_dump(mode="json")
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError(f"cannot read resume artifact {path.name}") from None


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        raise ValueError(f"cannot read resume artifact {path.name}") from None
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"resume artifact {path.name} must contain JSON objects")
    return rows


def _run_fingerprints(
    plan: Sequence[AnchorBlueprint],
    seed_texts: Mapping[str, str],
    seed_licenses: Mapping[str, str],
) -> dict[str, object]:
    return {
        "plan_sha256": _fingerprint([item.model_dump(mode="json") for item in plan]),
        "prompt_sha256": {
            name: prompt_sha256(name) for name in ("guard", "generator", "verifier")
        },
        "source_assignment_sha256": _fingerprint(
            [
                {
                    "anchor_id": item.anchor_id,
                    "source_id": item.source_id,
                    "text_sha256": hashlib.sha256(
                        seed_texts.get(item.anchor_id, "").encode("utf-8")
                    ).hexdigest(),
                    "license": seed_licenses.get(item.anchor_id),
                }
                for item in plan
            ]
        ),
    }


def _restore_budget(budget: TeacherBudget, payload: object) -> None:
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint has invalid budget accounting")
    values = (
        payload.get("request_count"),
        payload.get("input_tokens"),
        payload.get("output_tokens"),
    )
    cost = payload.get("estimated_cost_usd")
    accounting_complete = payload.get("accounting_complete")
    if not (
        all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values)
        and isinstance(cost, (int, float))
        and not isinstance(cost, bool)
        and cost >= 0
        and isinstance(accounting_complete, bool)
    ):
        raise ValueError("resume checkpoint has invalid budget accounting")
    if budget.max_requests is not None and values[0] > budget.max_requests:
        raise ValueError("resume request budget is below prior usage")
    if budget.max_estimated_cost_usd is not None and float(cost) > budget.max_estimated_cost_usd:
        raise ValueError("resume cost budget is below prior usage")
    budget.request_count, budget.input_tokens, budget.output_tokens = values
    budget.estimated_cost_usd = float(cost)
    budget.accounting_complete = accounting_complete


def _coverage_complete(
    planned: Sequence[AnchorBlueprint], accepted: Sequence[AnchorBlueprint]
) -> bool:
    for field in ("risk_domain", "transition", "input_style"):
        if {getattr(item, field) for item in accepted} != {
            getattr(item, field) for item in planned
        }:
            return False
    return True


def _verifier_agrees_with_oracle(
    pair: PolicyPair,
    verdict: VerifierResult,
    mapping: Mapping[str, Literal["before", "after"]],
) -> bool:
    by_stage = {mapping[view.opaque_id]: view for view in verdict.views if view.opaque_id in mapping}
    return set(by_stage) == {"before", "after"} and all(
        (by_stage[stage].label, by_stage[stage].rule_title)
        == (oracle.label, oracle.rule_title)
        for stage, oracle in (
            ("before", pair.before_annotation),
            ("after", pair.after_annotation),
        )
    )


def run_singguard_batch(
    plan: Sequence[AnchorBlueprint],
    *,
    generator: Teacher,
    verifier: Teacher,
    budget: TeacherBudget,
    output_dir: Path,
    seed: int,
    seed_texts: Mapping[str, str] | None = None,
    seed_licenses: Mapping[str, str] | None = None,
    max_retries: int = 2,
    pilot: bool = False,
    resume: bool = False,
    progress: Callable[[dict[str, object]], object] | None = None,
) -> dict[str, object]:
    """Generate, blind-verify, gate, audit, and export one immutable batch."""

    from risk_agent.ms_swift_singguard import prepare_singguard_bundle
    from risk_agent.singguard_quality import (
        build_quality_report,
        gate_candidate,
        stratified_review_ids,
    )

    output_dir = output_dir.absolute()
    if output_dir.exists() and not resume:
        raise ValueError("output directory must not exist")
    if resume and not output_dir.is_dir():
        raise ValueError("resume output directory does not exist")
    if not plan:
        raise ValueError("plan must not be empty")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or not 0 <= max_retries <= 2:
        raise ValueError("max_retries must be an integer from zero to two")
    if len({item.anchor_id for item in plan}) != len(plan):
        raise ValueError("plan contains duplicate anchor IDs")

    seed_texts = dict(seed_texts or {})
    seed_licenses = dict(seed_licenses or {})
    fingerprints = _run_fingerprints(plan, seed_texts, seed_licenses)
    catalog = load_rule_catalog()
    if resume:
        if (output_dir / "ms_swift").exists():
            raise ValueError("completed output cannot be resumed")
        checkpoint = _read_json(output_dir / "checkpoint.json")
        if not isinstance(checkpoint, dict):
            raise ValueError("resume checkpoint must be a JSON object")
        for name, expected in fingerprints.items():
            if checkpoint.get(name) != expected:
                label = "plan fingerprint" if name == "plan_sha256" else f"{name} fingerprint"
                raise ValueError(f"resume {label} does not match")
        completed_count = checkpoint.get("completed_anchors")
        if not isinstance(completed_count, int) or not 0 <= completed_count <= len(plan):
            raise ValueError("resume checkpoint has invalid completed anchor count")
        accepted_rows = _read_jsonl(output_dir / "accepted.jsonl")
        rejected_rows = _read_jsonl(output_dir / "rejected.jsonl")
        completed_ids = {item.anchor_id for item in plan[:completed_count]}
        accepted_rows = [
            row for row in accepted_rows if row.get("anchor_id") in completed_ids
        ]
        rejected_rows = [
            row for row in rejected_rows if row.get("anchor_id") in completed_ids
        ]
        try:
            accepted_blueprints = [
                AnchorBlueprint.model_validate(row["blueprint"]) for row in accepted_rows
            ]
            existing_texts = [
                _combined_content_for_batch(GeneratedContent.model_validate(row["content"]))
                for row in accepted_rows
            ]
        except (KeyError, TypeError, ValueError):
            raise ValueError("resume accepted artifact is invalid") from None
        first_pass_agreements = sum(
            int(row.get("attempt") == 1 and row.get("oracle_agreement") is True)
            for row in rejected_rows
        ) + sum(int(row.get("attempt") == 1) for row in accepted_rows)
        near_duplicate_rejections = sum(
            int("near_duplicate" in row.get("codes", [])) for row in rejected_rows
        )
        _restore_budget(budget, checkpoint.get("budget"))
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        _atomic_jsonl(output_dir / "plan.jsonl", list(plan))
        accepted_rows = []
        accepted_blueprints = []
        rejected_rows = []
        existing_texts = []
        first_pass_agreements = 0
        near_duplicate_rejections = 0
        completed_count = 0
    stopped_reason: str | None = None

    def emit(phase: str, *, anchor_id: str | None = None, attempt: int | None = None) -> None:
        if progress is None:
            return
        event: dict[str, object] = {
            "phase": phase,
            "completed": completed_count,
            "total": len(plan),
            "accepted": len(accepted_rows),
            "rejected": len(rejected_rows),
            "requests": budget.request_count,
        }
        if anchor_id is not None:
            event["anchor_id"] = anchor_id
        if attempt is not None:
            event["attempt"] = attempt
        try:
            progress(event)
        except Exception:
            pass

    emit("start")

    for plan_index in range(completed_count, len(plan)):
        blueprint = plan[plan_index]
        pair = compile_policy_pair(blueprint, catalog=catalog)
        accepted = False
        for attempt in range(max_retries + 1):
            try:
                emit("generate", anchor_id=blueprint.anchor_id, attempt=attempt + 1)
                content, _generator_usage = generate_content(
                    blueprint,
                    generator,
                    seed_text=seed_texts.get(blueprint.anchor_id),
                )
                emit("verify", anchor_id=blueprint.anchor_id, attempt=attempt + 1)
                verdict, opaque_to_stage, _verifier_usage = verify_content(
                    blueprint,
                    pair,
                    content,
                    verifier,
                    shuffle_seed=seed + plan_index * 7 + attempt,
                )
                if attempt == 0 and _verifier_agrees_with_oracle(
                    pair, verdict, opaque_to_stage
                ):
                    first_pass_agreements += 1
                oracle_agreement = _verifier_agrees_with_oracle(
                    pair, verdict, opaque_to_stage
                )
                gate = gate_candidate(
                    blueprint,
                    pair,
                    content,
                    verdict,
                    opaque_to_stage,
                    existing_texts=existing_texts,
                    source_text=seed_texts.get(blueprint.anchor_id),
                )
                codes = gate.codes
            except (ValueError, KeyError, TypeError):
                codes = ("schema",)
                content = None
                verdict = None
                opaque_to_stage = {}
            except (TeacherRequestError, TeacherBudgetExceeded) as error:
                stopped_reason = (
                    error.reason if isinstance(error, TeacherBudgetExceeded) else "provider_error"
                )
                break

            if not codes and content is not None and verdict is not None:
                examples = examples_from_verifier(
                    blueprint,
                    pair,
                    content,
                    verdict,
                    opaque_to_stage,
                )
                row = {
                    "anchor_id": blueprint.anchor_id,
                    "attempt": attempt + 1,
                    "oracle_agreement": True,
                    "blueprint": blueprint.model_dump(mode="json"),
                    "content": content.model_dump(mode="json"),
                    "examples": [example.model_dump(mode="json") for example in examples],
                    "source": {
                        "used": blueprint.anchor_id in seed_texts,
                        "license": seed_licenses.get(blueprint.anchor_id),
                    },
                }
                accepted_rows.append(row)
                accepted_blueprints.append(blueprint)
                existing_texts.append(_combined_content_for_batch(content))
                accepted = True
                break

            rejected_rows.append(
                {
                    "anchor_id": blueprint.anchor_id,
                    "attempt": attempt + 1,
                    "codes": list(codes),
                    "oracle_agreement": oracle_agreement if verdict is not None else False,
                }
            )
            near_duplicate_rejections += int("near_duplicate" in codes)
            if attempt < max_retries:
                emit("retry", anchor_id=blueprint.anchor_id, attempt=attempt + 2)

        if not stopped_reason:
            completed_count = plan_index + 1
        _atomic_jsonl(output_dir / "accepted.jsonl", accepted_rows)
        _atomic_jsonl(output_dir / "rejected.jsonl", rejected_rows)
        _atomic_json(
            output_dir / "checkpoint.json",
            {
                "schema": "singguard-checkpoint-v1",
                "completed_anchors": completed_count,
                "accepted_anchors": len(accepted_rows),
                "status": "incomplete" if stopped_reason else "running",
                "budget": budget.as_dict(),
                **fingerprints,
            },
        )
        if stopped_reason:
            emit("stopped", anchor_id=blueprint.anchor_id)
            break
        emit("anchor_complete", anchor_id=blueprint.anchor_id)

    rejection_codes = tuple(tuple(row["codes"]) for row in rejected_rows)
    licenses = tuple(
        license_name
        for row in accepted_rows
        if (license_name := row["source"]["license"]) is not None
    )
    report = build_quality_report(
        plan,
        accepted_blueprints,
        rejection_codes=rejection_codes,
        first_pass_agreements=first_pass_agreements,
        first_pass_attempts=completed_count,
        near_duplicate_rejections=near_duplicate_rejections,
        licenses=licenses,
    )
    coverage = _coverage_complete(plan, accepted_blueprints)
    report["pilot_gates"]["near_duplicate_rate"] = (
        near_duplicate_rejections / len(plan) <= 0.05
    )
    report["pilot_gates"]["quota_coverage"] = coverage
    _atomic_json(output_dir / "quality_report.json", report)
    review_ids = set(stratified_review_ids(accepted_blueprints, fraction=0.10, seed=seed))
    _atomic_jsonl(
        output_dir / "review_sample.jsonl",
        [row for row in accepted_rows if row["anchor_id"] in review_ids],
    )

    gates_pass = all(report["pilot_gates"].values()) if pilot else True
    complete = (
        stopped_reason is None
        and completed_count == len(plan)
        and bool(accepted_rows)
        and gates_pass
    )
    if complete:
        examples_path = output_dir / ".accepted_examples.jsonl"
        _atomic_jsonl(
            examples_path,
            [example for row in accepted_rows for example in row["examples"]],
        )
        prepare_singguard_bundle(examples_path, output_dir / "ms_swift", seed=seed)
        examples_path.unlink()

    manifest: dict[str, object] = {
        "schema": "singguard-batch-v1",
        "status": "complete" if complete else "incomplete",
        "reason": stopped_reason if stopped_reason else (None if gates_pass else "quality_gates"),
        "seed": seed,
        **fingerprints,
        "planned_anchors": len(plan),
        "accepted_anchors": len(accepted_rows),
        "accepted_examples": len(accepted_rows) * 2,
        "rejected_attempts": len(rejected_rows),
        "budget": budget.as_dict(),
        "artifacts": {
            name: _sha256_file(output_dir / name)
            for name in (
                "plan.jsonl",
                "accepted.jsonl",
                "rejected.jsonl",
                "quality_report.json",
                "review_sample.jsonl",
            )
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    _atomic_json(
        output_dir / "checkpoint.json",
        {
            "schema": "singguard-checkpoint-v1",
            "completed_anchors": completed_count,
            "accepted_anchors": len(accepted_rows),
            "status": manifest["status"],
            "budget": budget.as_dict(),
            **fingerprints,
        },
    )
    emit("complete" if complete else "incomplete")
    return manifest


def _combined_content_for_batch(content: GeneratedContent) -> str:
    return "\n".join(part for part in (content.query, content.response) if part)
