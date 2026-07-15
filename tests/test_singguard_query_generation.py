"""Deterministic quota planning for SingGuard query synthesis."""

from __future__ import annotations

import hashlib
from collections import Counter

import pytest
from pydantic import ValidationError

from risk_agent.contracts import PolicyRule
from risk_agent.singguard import ActivePolicy
from risk_agent.singguard_sources import SeedRecord


def _policies() -> tuple[ActivePolicy, ...]:
    return (
        ActivePolicy(
            policy_id="commerce-v1",
            rules=(
                PolicyRule(
                    rule_id="R1",
                    title="Deception",
                    text="No deceptive claims.",
                    exceptions=("Educational context", "Satirical context"),
                ),
                PolicyRule(rule_id="R2", title="Solicitation", text="No risky solicitation."),
                PolicyRule(rule_id="R3", title="Impersonation", text="No impersonation."),
            ),
        ),
        ActivePolicy(
            policy_id="safety-v1",
            rules=(
                PolicyRule(
                    rule_id="R4",
                    title="Danger",
                    text="No dangerous advice.",
                    exceptions=("Emergency prevention",),
                ),
                PolicyRule(rule_id="R5", title="Abuse", text="No targeted abuse."),
            ),
        ),
    )


_SOURCE_WEIGHTS = {
    "uci_sms_spam": 200,
    "uci_youtube_spam": 100,
    "nemotron_aegis_v2": 120,
    "civil_comments": 100,
    "amazon_esci": 80,
}


def _seed_record(number: int, *, source: str = "nemotron_aegis_v2") -> SeedRecord:
    text = f"Governed {source} example {number}"
    return SeedRecord(
        source=source,
        source_id=f"{source}-row-{number:04d}",
        provenance_url="https://example.test/dataset",
        license="CC-BY-4.0",
        usage_scope="research_only",
        source_role="style_seed",
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        retrieved_at="2026-07-15T00:00:00Z",
    )


def _seed_records(count: int) -> tuple[SeedRecord, ...]:
    allocations = {
        source: quota * count // 600 for source, quota in _SOURCE_WEIGHTS.items()
    }
    remaining = count - sum(allocations.values())
    remainder_order = sorted(
        _SOURCE_WEIGHTS,
        key=lambda source: _SOURCE_WEIGHTS[source] * count % 600,
        reverse=True,
    )
    for source in remainder_order[:remaining]:
        allocations[source] += 1
    return tuple(
        _seed_record(number, source=source)
        for source, allocation in allocations.items()
        for number in range(allocation)
    )


def _plan(
    *, count: int = 2_000, seed: int = 41, seed_count: int = 700
):
    from risk_agent.singguard_query_generation import plan_blueprints

    return plan_blueprints(
        _policies(),
        count=count,
        seed=seed,
        seed_records=_seed_records(seed_count),
    )


def test_plan_matches_all_exact_2000_row_quotas() -> None:
    rows = _plan()

    assert len(rows) == 2_000
    assert Counter(row.intended_label for row in rows) == {"safe": 1_000, "unsafe": 1_000}
    assert Counter(row.conversation_shape for row in rows) == {
        "query": 1_400,
        "query_response": 600,
    }
    assert Counter(row.thinking_type for row in rows) == {"fast": 1_400, "slow": 600}
    assert Counter(row.content_form for row in rows) == {
        "short_ad": 400,
        "social_post": 300,
        "livestream_pitch": 300,
        "product_listing": 300,
        "comment": 200,
        "private_message": 200,
        "support_exchange": 200,
        "search_or_neutral": 100,
    }
    assert Counter(row.difficulty for row in rows) == {
        "explicit": 500,
        "paraphrased": 600,
        "implicit": 400,
        "exception": 500,
    }
    assert Counter(row.tone for row in rows) == {
        "formal": 250,
        "colloquial": 250,
        "promotional": 250,
        "urgent": 250,
        "testimonial": 250,
        "technical": 250,
        "humorous": 250,
        "neutral": 250,
    }
    assert Counter(row.noise_profile for row in rows) == {
        "none": 400,
        "spelling": 400,
        "emoji": 400,
        "punctuation": 400,
        "obfuscation": 400,
    }
    assert Counter(row.length_bin for row in rows) == {
        "headline": 500,
        "short": 500,
        "medium": 500,
        "long": 500,
    }
    assert all(row.tool_capable is False for row in rows)
    assert Counter(row.source_ref is None for row in rows) == {True: 1_400, False: 600}


def test_axes_are_balanced_within_each_label_and_label_thinking_is_crossed() -> None:
    rows = _plan(seed=3_685)

    assert Counter((row.intended_label, row.thinking_type) for row in rows) == {
        ("safe", "fast"): 700,
        ("safe", "slow"): 300,
        ("unsafe", "fast"): 700,
        ("unsafe", "slow"): 300,
    }
    for attribute in (
        "content_form",
        "difficulty",
        "tone",
        "noise_profile",
        "length_bin",
    ):
        by_label_and_value = Counter(
            (row.intended_label, getattr(row, attribute)) for row in rows
        )
        for value in {getattr(row, attribute) for row in rows}:
            safe_count = by_label_and_value[("safe", value)]
            unsafe_count = by_label_and_value[("unsafe", value)]
            assert abs(safe_count - unsafe_count) <= 1


def test_identical_inputs_and_seed_produce_identical_immutable_blueprints() -> None:
    first = _plan(count=100, seed=13)
    second = _plan(count=100, seed=13)

    assert first == second
    with pytest.raises(ValidationError, match="frozen"):
        first[0].tone = "formal"


def test_different_seed_changes_plan_without_changing_quotas() -> None:
    first = _plan(count=500, seed=13)
    second = _plan(count=500, seed=14)

    first_metadata = [
        row.model_dump(exclude={"blueprint_id", "family_id"}) for row in first
    ]
    second_metadata = [
        row.model_dump(exclude={"blueprint_id", "family_id"}) for row in second
    ]
    assert first_metadata != second_metadata
    for attribute in (
        "intended_label",
        "conversation_shape",
        "thinking_type",
        "content_form",
        "difficulty",
    ):
        assert Counter(getattr(row, attribute) for row in first) == Counter(
            getattr(row, attribute) for row in second
        )
    assert Counter(row.source_ref is None for row in first) == Counter(
        row.source_ref is None for row in second
    )


def test_blueprint_and_family_ids_are_unique_stable_and_distinct() -> None:
    rows = _plan(count=500, seed=9)

    assert len({row.blueprint_id for row in rows}) == 500
    assert len({row.family_id for row in rows}) == 500
    assert all(row.blueprint_id != row.family_id for row in rows)
    assert [row.blueprint_id for row in rows] == [
        row.blueprint_id for row in _plan(count=500, seed=9)
    ]


def test_ids_change_when_policy_semantics_change() -> None:
    from risk_agent.singguard_query_generation import plan_blueprints

    changed = tuple(
        policy.model_copy(
            update={
                "rules": tuple(
                    rule.model_copy(update={"title": f"Changed {rule.title}"})
                    for rule in policy.rules
                )
            }
        )
        if policy.policy_id == "commerce-v1"
        else policy
        for policy in _policies()
    )

    original_rows = plan_blueprints(
        _policies(), count=100, seed=3, seed_records=()
    )
    changed_rows = plan_blueprints(changed, count=100, seed=3, seed_records=())

    assert [row.blueprint_id for row in original_rows] != [
        row.blueprint_id for row in changed_rows
    ]
    for original, updated in zip(original_rows, changed_rows, strict=True):
        if original.policy_id == "safety-v1":
            assert original.blueprint_id == updated.blueprint_id


def test_changing_seed_text_and_hash_changes_only_the_affected_semantic_id() -> None:
    from risk_agent.singguard_query_generation import plan_blueprints

    original_seed = _seed_record(1)
    changed_text = f"{original_seed.text} changed"
    changed_seed = original_seed.model_copy(
        update={
            "text": changed_text,
            "content_hash": hashlib.sha256(changed_text.encode()).hexdigest(),
        }
    )
    original = plan_blueprints(
        _policies(), count=100, seed=8, seed_records=(original_seed,)
    )
    changed = plan_blueprints(
        _policies(), count=100, seed=8, seed_records=(changed_seed,)
    )

    changed_offsets = [
        index
        for index, (before, after) in enumerate(zip(original, changed, strict=True))
        if before.blueprint_id != after.blueprint_id
    ]
    assert len(changed_offsets) == 1
    affected = changed_offsets[0]
    assert original[affected].source_ref.source_id == original_seed.source_id
    assert changed[affected].source_ref.content_hash == changed_seed.content_hash


def test_unsafe_primary_rules_are_balanced_and_answers_follow_policy_order() -> None:
    rows = _plan()
    policies = {policy.policy_id: policy for policy in _policies()}
    unsafe = [row for row in rows if row.intended_label == "unsafe"]
    counts = Counter(row.primary_rule_id for row in unsafe)

    assert set(counts) == {"R1", "R2", "R3", "R4", "R5"}
    assert max(counts.values()) - min(counts.values()) <= 1
    for row in unsafe:
        policy_order = tuple(rule.rule_id for rule in policies[row.policy_id].rules)
        title_order = tuple(rule.title for rule in policies[row.policy_id].rules)
        assert row.active_rule_ids == policy_order
        assert row.active_rule_titles == title_order
        assert row.primary_rule_id in policy_order
        primary_index = policy_order.index(row.primary_rule_id)
        assert row.primary_answer == title_order[primary_index]
        assert row.intended_answers
        assert row.primary_answer in row.intended_answers
        assert row.intended_answers == tuple(
            title for title in title_order if title in row.intended_answers
        )


def test_planned_answer_titles_integrate_with_semantic_validation() -> None:
    from risk_agent.singguard import ModerationSample, ParsedCompletion
    from risk_agent.singguard_generation import _semantic_mismatch

    rows = _plan(count=100, seed_count=0)
    row = next(item for item in rows if item.intended_label == "unsafe")
    policy = next(item for item in _policies() if item.policy_id == row.policy_id)
    sample = ModerationSample(
        sample_id=row.blueprint_id,
        policy_id=row.policy_id,
        thinking_type=row.thinking_type,
        query="Planned content anchor.",
        expected_label=row.intended_label,
        expected_answers=row.intended_answers,
    )
    parsed = ParsedCompletion(
        label="unsafe",
        answers=row.intended_answers,
        original="unsafe\n<answer>planned</answer>",
    )

    assert row.primary_rule_id in row.active_rule_ids
    assert row.primary_answer in row.active_rule_titles
    assert _semantic_mismatch(parsed, policy=policy, sample=sample) is None


def test_plan_rejects_too_few_unsafe_rows_to_cover_every_active_rule() -> None:
    from risk_agent.singguard_query_generation import plan_blueprints

    policy = ActivePolicy(
        policy_id="too-many-rules",
        rules=tuple(
            PolicyRule(
                rule_id=f"R{index:02d}",
                title=f"Rule {index}",
                text=f"Rule {index} text.",
            )
            for index in range(51)
        ),
    )

    with pytest.raises(ValueError, match="enough unsafe rows to cover every active rule"):
        plan_blueprints((policy,), count=100, seed=1, seed_records=())


def test_safe_rows_have_no_intended_answers_or_primary_answer() -> None:
    safe = [row for row in _plan(count=500) if row.intended_label == "safe"]

    assert safe
    assert all(row.intended_answers == () for row in safe)
    assert all(row.primary_rule_id is None for row in safe)
    assert all(row.primary_answer is None for row in safe)
    assert all(row.tool_capable is False for row in safe)


def test_safe_exception_rows_cover_documented_exceptions_evenly() -> None:
    rows = _plan(count=100, seed_count=0)
    policies = {policy.policy_id: policy for policy in _policies()}
    exception_rows = [
        row
        for row in rows
        if row.intended_label == "safe" and row.difficulty == "exception"
    ]
    expected_pairs = {
        (rule.rule_id, exception)
        for policy in _policies()
        for rule in policy.rules
        for exception in rule.exceptions
    }
    counts = Counter(
        (row.target_exception_rule_id, row.target_exception) for row in exception_rows
    )

    assert set(counts) == expected_pairs
    assert max(counts.values()) - min(counts.values()) <= 1
    for row in exception_rows:
        policy = policies[row.policy_id]
        rule = next(
            rule for rule in policy.rules if rule.rule_id == row.target_exception_rule_id
        )
        assert row.target_exception in rule.exceptions


def test_exception_difficulty_remains_generic_when_policy_has_no_exceptions() -> None:
    from risk_agent.singguard_query_generation import plan_blueprints

    policy = ActivePolicy(
        policy_id="no-exceptions",
        rules=(PolicyRule(rule_id="ONLY", title="Only", text="Only rule."),),
    )
    rows = plan_blueprints((policy,), count=100, seed=1, seed_records=())

    exception_rows = [row for row in rows if row.difficulty == "exception"]
    assert exception_rows
    assert all(row.target_exception_rule_id is None for row in exception_rows)
    assert all(row.target_exception is None for row in exception_rows)


def test_plan_rejects_too_few_safe_exception_slots_for_documented_pairs() -> None:
    from risk_agent.singguard_query_generation import plan_blueprints

    policy = ActivePolicy(
        policy_id="many-exceptions",
        rules=(
            PolicyRule(
                rule_id="ONLY",
                title="Only",
                text="Only rule.",
                exceptions=tuple(f"Exception {index}" for index in range(13)),
            ),
        ),
    )

    with pytest.raises(ValueError, match="safe exception rows to cover every"):
        plan_blueprints((policy,), count=100, seed=1, seed_records=())


@pytest.mark.parametrize(("count", "expected"), ((100, 10), (500, 50), (2_000, 200)))
def test_multi_risk_rows_use_exactly_two_policy_ordered_titles(
    count: int, expected: int
) -> None:
    rows = _plan(count=count, seed_count=0)
    multi = [row for row in rows if len(row.intended_answers) == 2]

    assert len(multi) == expected
    for row in multi:
        assert len(set(row.intended_answers)) == 2
        assert row.primary_answer in row.intended_answers
        assert row.intended_answers == tuple(
            title for title in row.active_rule_titles if title in row.intended_answers
        )


def test_multi_risk_quota_degrades_when_no_policy_has_two_rules() -> None:
    from risk_agent.singguard_query_generation import plan_blueprints

    policies = tuple(
        ActivePolicy(
            policy_id=f"single-{index}",
            rules=(
                PolicyRule(
                    rule_id=f"R{index}", title=f"Rule {index}", text="Rule text."
                ),
            ),
        )
        for index in range(2)
    )

    rows = plan_blueprints(policies, count=100, seed=1, seed_records=())

    assert all(len(row.intended_answers) <= 1 for row in rows)


def test_at_least_quota_seeds_assigns_exactly_600_unique_references() -> None:
    rows = _plan(seed_count=650)
    refs = [row.source_ref for row in rows if row.source_ref is not None]

    assert len(refs) == 600
    assert len({(ref.source, ref.source_id) for ref in refs}) == 600
    assert all(ref.content_hash for ref in refs)


@pytest.mark.parametrize(
    ("count", "expected"),
    (
        (
            100,
            {
                "uci_sms_spam": 10,
                "uci_youtube_spam": 5,
                "nemotron_aegis_v2": 6,
                "civil_comments": 5,
                "amazon_esci": 4,
            },
        ),
        (
            500,
            {
                "uci_sms_spam": 50,
                "uci_youtube_spam": 25,
                "nemotron_aegis_v2": 30,
                "civil_comments": 25,
                "amazon_esci": 20,
            },
        ),
        (2_000, _SOURCE_WEIGHTS),
    ),
)
def test_exact_approved_governed_source_mix(count: int, expected: dict[str, int]) -> None:
    rows = _plan(count=count, seed_count=700)

    assert Counter(
        row.source_ref.source for row in rows if row.source_ref is not None
    ) == expected


def test_governed_sources_are_crossed_with_label_and_thinking() -> None:
    rows = _plan(count=100, seed=3_685, seed_count=100)
    governed = [row for row in rows if row.source_ref is not None]

    assert Counter(row.intended_label for row in governed) == {"safe": 15, "unsafe": 15}
    expected_thinking = {
        "uci_sms_spam": {"fast": 7, "slow": 3},
        "uci_youtube_spam": {"fast": 4, "slow": 1},
        "nemotron_aegis_v2": {"fast": 4, "slow": 2},
        "civil_comments": {"fast": 4, "slow": 1},
        "amazon_esci": {"fast": 3, "slow": 1},
    }
    for source, thinking_counts in expected_thinking.items():
        source_rows = [row for row in governed if row.source_ref.source == source]
        assert Counter(row.thinking_type for row in source_rows) == thinking_counts
        label_counts = Counter(row.intended_label for row in source_rows)
        assert abs(label_counts["safe"] - label_counts["unsafe"]) <= 1


def test_governed_sources_are_assigned_only_to_compatible_forms() -> None:
    allowed = {
        "uci_sms_spam": {
            "short_ad",
            "private_message",
            "support_exchange",
            "social_post",
        },
        "uci_youtube_spam": {
            "comment",
            "social_post",
            "livestream_pitch",
            "short_ad",
        },
        "nemotron_aegis_v2": {
            "short_ad",
            "social_post",
            "livestream_pitch",
            "product_listing",
            "comment",
            "private_message",
            "support_exchange",
            "search_or_neutral",
        },
        "civil_comments": {
            "comment",
            "social_post",
            "livestream_pitch",
            "support_exchange",
        },
        "amazon_esci": {
            "product_listing",
            "search_or_neutral",
            "short_ad",
            "social_post",
            "livestream_pitch",
        },
    }

    for row in _plan(seed=3_685):
        if row.source_ref is not None:
            assert row.content_form in allowed[row.source_ref.source]


def test_one_source_shortfall_is_filled_synthetically_not_from_other_sources() -> None:
    from risk_agent.singguard_query_generation import plan_blueprints

    seeds = tuple(
        _seed_record(index, source="uci_sms_spam") for index in range(100)
    )
    rows = plan_blueprints(_policies(), count=100, seed=5, seed_records=seeds)

    refs = [row.source_ref for row in rows if row.source_ref is not None]
    assert len(refs) == 10
    assert {ref.source for ref in refs} == {"uci_sms_spam"}


def test_unknown_governed_source_is_rejected() -> None:
    from risk_agent.singguard_query_generation import plan_blueprints

    with pytest.raises(ValueError, match="unknown governed source"):
        plan_blueprints(
            _policies(),
            count=100,
            seed=1,
            seed_records=(_seed_record(1, source="unapproved_source"),),
        )


@pytest.mark.parametrize(
    ("count", "seed_count", "expected_governed"),
    ((100, 45, 30), (500, 200, 150)),
)
def test_governed_source_quota_scales_for_smaller_releases(
    count: int, seed_count: int, expected_governed: int
) -> None:
    rows = _plan(count=count, seed_count=seed_count)

    assert sum(row.source_ref is not None for row in rows) == expected_governed


def test_no_seeds_degrades_to_all_synthetic() -> None:
    rows = _plan(seed_count=0)

    assert all(row.source_ref is None for row in rows)


def test_fewer_than_quota_seeds_are_each_used_at_most_once() -> None:
    rows = _plan(seed_count=37)
    refs = [row.source_ref for row in rows if row.source_ref is not None]

    assert len(refs) == 37
    assert len({(ref.source, ref.source_id) for ref in refs}) == 37


@pytest.mark.parametrize(
    ("count", "expected"),
    (
        (
            100,
            {
                "labels": {"safe": 50, "unsafe": 50},
                "shapes": {"query": 70, "query_response": 30},
                "thinking": {"fast": 70, "slow": 30},
                "forms": {
                    "short_ad": 20,
                    "social_post": 15,
                    "livestream_pitch": 15,
                    "product_listing": 15,
                    "comment": 10,
                    "private_message": 10,
                    "support_exchange": 10,
                    "search_or_neutral": 5,
                },
                "difficulty": {
                    "explicit": 25,
                    "paraphrased": 30,
                    "implicit": 20,
                    "exception": 25,
                },
            },
        ),
        (
            500,
            {
                "labels": {"safe": 250, "unsafe": 250},
                "shapes": {"query": 350, "query_response": 150},
                "thinking": {"fast": 350, "slow": 150},
                "forms": {
                    "short_ad": 100,
                    "social_post": 75,
                    "livestream_pitch": 75,
                    "product_listing": 75,
                    "comment": 50,
                    "private_message": 50,
                    "support_exchange": 50,
                    "search_or_neutral": 25,
                },
                "difficulty": {
                    "explicit": 125,
                    "paraphrased": 150,
                    "implicit": 100,
                    "exception": 125,
                },
            },
        ),
    ),
)
def test_supported_smaller_counts_scale_quotas_exactly(count: int, expected: dict) -> None:
    rows = _plan(count=count)

    assert Counter(row.intended_label for row in rows) == expected["labels"]
    assert Counter(row.conversation_shape for row in rows) == expected["shapes"]
    assert Counter(row.thinking_type for row in rows) == expected["thinking"]
    assert Counter(row.content_form for row in rows) == expected["forms"]
    assert Counter(row.difficulty for row in rows) == expected["difficulty"]


@pytest.mark.parametrize("count", (0, 99, 101, 1_000, 2_001))
def test_unsupported_counts_are_rejected(count: int) -> None:
    from risk_agent.singguard_query_generation import plan_blueprints

    with pytest.raises(ValueError, match="supported release sizes"):
        plan_blueprints(_policies(), count=count, seed=1, seed_records=())


@pytest.mark.parametrize("count", (True, 100.0, "100"))
def test_count_must_be_an_integer_not_bool_or_coercible_value(count: object) -> None:
    from risk_agent.singguard_query_generation import plan_blueprints

    with pytest.raises(TypeError, match="count must be an integer"):
        plan_blueprints(_policies(), count=count, seed=1, seed_records=())


def test_empty_and_duplicate_policy_ids_are_rejected() -> None:
    from risk_agent.singguard_query_generation import plan_blueprints

    with pytest.raises(ValueError, match="at least one active policy"):
        plan_blueprints((), count=100, seed=1, seed_records=())
    duplicate = (_policies()[0], _policies()[0])
    with pytest.raises(ValueError, match="duplicate policy IDs"):
        plan_blueprints(duplicate, count=100, seed=1, seed_records=())


def test_inconsistent_rule_reuse_across_policies_is_rejected() -> None:
    from risk_agent.singguard_query_generation import plan_blueprints

    policies = (
        ActivePolicy(
            policy_id="one",
            rules=(PolicyRule(rule_id="SHARED", title="Original", text="Original text."),),
        ),
        ActivePolicy(
            policy_id="two",
            rules=(PolicyRule(rule_id="SHARED", title="Changed", text="Changed text."),),
        ),
    )

    with pytest.raises(ValueError, match="inconsistent rule ID"):
        plan_blueprints(policies, count=100, seed=1, seed_records=())


def test_invalid_or_duplicate_seed_records_are_rejected() -> None:
    from risk_agent.singguard_query_generation import plan_blueprints

    seed_record = _seed_record(1)
    with pytest.raises(ValueError, match="duplicate seed"):
        plan_blueprints(
            _policies(), count=100, seed=1, seed_records=(seed_record, seed_record)
        )
    duplicate_content = _seed_record(2).model_copy(
        update={"content_hash": seed_record.content_hash}
    )
    with pytest.raises(ValueError, match="duplicate seed content"):
        plan_blueprints(
            _policies(),
            count=100,
            seed=1,
            seed_records=(seed_record, duplicate_content),
        )
    with pytest.raises(TypeError, match="SeedRecord"):
        plan_blueprints(_policies(), count=100, seed=1, seed_records=({},))


def test_planner_rejects_seed_hash_that_does_not_match_exact_text() -> None:
    from risk_agent.singguard_query_generation import plan_blueprints

    placeholder_hash = "x" * 64
    seed_record = _seed_record(1).model_copy(
        update={"content_hash": placeholder_hash}
    )

    with pytest.raises(ValueError, match="lowercase SHA256"):
        plan_blueprints(_policies(), count=100, seed=1, seed_records=(seed_record,))


def test_source_ref_is_frozen_and_forbids_extra_fields() -> None:
    from risk_agent.singguard_query_generation import SourceRef

    source_ref = SourceRef(
        source="open-corpus",
        source_id="row-1",
        content_hash="0" * 64,
    )

    with pytest.raises(ValidationError, match="frozen"):
        source_ref.source_id = "changed"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SourceRef(
            source="open-corpus",
            source_id="row-1",
            content_hash="0" * 64,
            unexpected="value",
        )


def test_query_blueprint_forbids_extra_fields() -> None:
    from risk_agent.singguard_query_generation import QueryBlueprint

    payload = _plan(count=100, seed_count=0)[0].model_dump()
    payload["unexpected"] = "value"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        QueryBlueprint.model_validate(payload)


def test_query_blueprint_accepts_policy_ordered_multi_answers_and_rejects_reverse() -> None:
    from risk_agent.singguard_query_generation import QueryBlueprint

    row = next(
        row
        for row in _plan(count=100, seed_count=0)
        if row.intended_label == "unsafe" and len(row.active_rule_ids) >= 2
    )
    ordered = row.active_rule_titles[:2]
    payload = row.model_dump()
    payload.update(
        primary_rule_id=row.active_rule_ids[0],
        primary_answer=ordered[0],
        intended_answers=ordered,
    )

    blueprint = QueryBlueprint.model_validate(payload)

    assert blueprint.intended_answers == ordered
    payload["intended_answers"] = tuple(reversed(ordered))
    with pytest.raises(ValidationError, match="active-policy order"):
        QueryBlueprint.model_validate(payload)


def test_identical_global_rule_reuse_across_policies_is_accepted_and_balanced() -> None:
    from risk_agent.singguard_query_generation import plan_blueprints

    shared = PolicyRule(
        rule_id="SHARED", title="Shared rule", text="Shared rule text."
    )
    policies = (
        ActivePolicy(
            policy_id="composition-one",
            rules=(shared,),
        ),
        ActivePolicy(
            policy_id="composition-two",
            rules=(
                shared,
                PolicyRule(rule_id="TWO", title="Rule two", text="Rule two text."),
            ),
        ),
        ActivePolicy(
            policy_id="composition-three",
            rules=(
                PolicyRule(rule_id="ONE", title="Rule one", text="Rule one text."),
            ),
        ),
    )

    rows = plan_blueprints(policies, count=100, seed=7, seed_records=())
    primary_counts = Counter(
        row.primary_rule_id for row in rows if row.intended_label == "unsafe"
    )

    assert set(primary_counts) == {"SHARED", "ONE", "TWO"}
    assert max(primary_counts.values()) - min(primary_counts.values()) <= 1
    shared_rows = [row for row in rows if row.primary_rule_id == "SHARED"]
    owner_counts = Counter(row.policy_id for row in shared_rows)
    assert abs(owner_counts["composition-one"] - owner_counts["composition-two"]) <= 1
    repeated = plan_blueprints(policies, count=100, seed=7, seed_records=())
    assert [row.policy_id for row in shared_rows] == [
        row.policy_id for row in repeated if row.primary_rule_id == "SHARED"
    ]
