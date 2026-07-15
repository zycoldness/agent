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
                PolicyRule(rule_id="R1", title="Deception", text="No deceptive claims."),
                PolicyRule(rule_id="R2", title="Solicitation", text="No risky solicitation."),
                PolicyRule(rule_id="R3", title="Impersonation", text="No impersonation."),
            ),
        ),
        ActivePolicy(
            policy_id="safety-v1",
            rules=(
                PolicyRule(rule_id="R4", title="Danger", text="No dangerous advice."),
                PolicyRule(rule_id="R5", title="Abuse", text="No targeted abuse."),
            ),
        ),
    )


def _seed_record(number: int) -> SeedRecord:
    text = f"Governed example {number}"
    return SeedRecord(
        source="open-corpus",
        source_id=f"row-{number:04d}",
        provenance_url="https://example.test/dataset",
        license="CC-BY-4.0",
        usage_scope="research_only",
        source_role="style_seed",
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        retrieved_at="2026-07-15T00:00:00Z",
    )


def _plan(
    *, count: int = 2_000, seed: int = 41, seed_count: int = 700
):
    from risk_agent.singguard_query_generation import plan_blueprints

    return plan_blueprints(
        _policies(),
        count=count,
        seed=seed,
        seed_records=tuple(_seed_record(index) for index in range(seed_count)),
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
    assert Counter(row.source_ref is None for row in rows) == {True: 1_400, False: 600}


def test_identical_inputs_and_seed_produce_identical_immutable_blueprints() -> None:
    first = _plan(count=100, seed=13)
    second = _plan(count=100, seed=13)

    assert first == second
    with pytest.raises(ValidationError, match="frozen"):
        first[0].tone = "formal"


def test_different_seed_changes_plan_without_changing_quotas() -> None:
    first = _plan(count=500, seed=13)
    second = _plan(count=500, seed=14)

    assert first != second
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


def test_ids_change_when_policy_content_changes() -> None:
    from risk_agent.singguard_query_generation import plan_blueprints

    changed = tuple(
        policy.model_copy(
            update={
                "rules": tuple(
                    rule.model_copy(update={"text": f"Changed: {rule.text}"})
                    for rule in policy.rules
                )
            }
        )
        for policy in _policies()
    )

    original_rows = plan_blueprints(
        _policies(), count=100, seed=3, seed_records=()
    )
    changed_rows = plan_blueprints(changed, count=100, seed=3, seed_records=())

    assert [row.blueprint_id for row in original_rows] != [
        row.blueprint_id for row in changed_rows
    ]


def test_unsafe_primary_rules_are_balanced_and_answers_follow_policy_order() -> None:
    rows = _plan()
    policies = {policy.policy_id: policy for policy in _policies()}
    unsafe = [row for row in rows if row.intended_label == "unsafe"]
    counts = Counter(row.primary_answer for row in unsafe)

    assert set(counts) == {"R1", "R2", "R3", "R4", "R5"}
    assert max(counts.values()) - min(counts.values()) <= 1
    for row in unsafe:
        policy_order = tuple(rule.rule_id for rule in policies[row.policy_id].rules)
        assert row.active_rule_ids == policy_order
        assert row.primary_answer in policy_order
        assert row.intended_answers
        assert row.primary_answer in row.intended_answers
        assert row.intended_answers == tuple(
            rule_id for rule_id in policy_order if rule_id in row.intended_answers
        )


def test_safe_rows_have_no_intended_answers_or_primary_answer() -> None:
    safe = [row for row in _plan(count=500) if row.intended_label == "safe"]

    assert safe
    assert all(row.intended_answers == () for row in safe)
    assert all(row.primary_answer is None for row in safe)


def test_at_least_quota_seeds_assigns_exactly_600_unique_references() -> None:
    rows = _plan(seed_count=650)
    refs = [row.source_ref for row in rows if row.source_ref is not None]

    assert len(refs) == 600
    assert len({(ref.source, ref.source_id) for ref in refs}) == 600
    assert all(ref.content_hash for ref in refs)


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


def test_planner_trusts_validated_seed_hash_without_recomputing_text_integrity() -> None:
    from risk_agent.singguard_query_generation import plan_blueprints

    placeholder_hash = "x" * 64
    seed_record = _seed_record(1).model_copy(
        update={"content_hash": placeholder_hash}
    )

    rows = plan_blueprints(
        _policies(), count=100, seed=1, seed_records=(seed_record,)
    )

    assert [row.source_ref.content_hash for row in rows if row.source_ref] == [
        placeholder_hash
    ]
