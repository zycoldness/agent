"""Planning and policy compilation for English SingGuard synthesis."""

from collections import Counter


def test_planner_is_deterministic_and_balances_pilot_transitions() -> None:
    from risk_agent.singguard_synthesis import plan_blueprints

    first = plan_blueprints(100, seed=17)
    second = plan_blueprints(100, seed=17)

    assert first == second
    assert Counter(item.transition for item in first) == {
        "unsafe_to_unsafe": 25,
        "unsafe_to_safe": 25,
        "safe_to_unsafe": 25,
        "safe_to_safe": 25,
    }
    domain_counts = Counter(item.risk_domain for item in first)
    assert len(domain_counts) == 8
    assert max(domain_counts.values()) - min(domain_counts.values()) <= 1
    assert sum(item.before_thinking_type == "slow" for item in first) + sum(
        item.after_thinking_type == "slow" for item in first
    ) == 60


def test_full_plan_has_exact_domain_and_style_quotas() -> None:
    from risk_agent.singguard_synthesis import plan_blueprints

    plan = plan_blueprints(2000, seed=42)

    assert set(Counter(item.risk_domain for item in plan).values()) == {250}
    assert Counter(item.input_style for item in plan) == {
        "short_ad": 400,
        "social_post": 300,
        "livestream_pitch": 300,
        "product_listing": 300,
        "comment_or_review": 200,
        "sms_or_private_message": 200,
        "support_dialogue": 200,
        "neutral_information": 100,
    }
    assert Counter(item.conversation_type for item in plan) == {
        "query": 1400,
        "query_response": 600,
    }


def test_policy_compiler_produces_ordered_label_pairs_and_realistic_policy_sizes() -> None:
    from risk_agent.singguard_synthesis import compile_policy_pair, plan_blueprints

    expected = {
        "unsafe_to_unsafe": ("unsafe", "unsafe"),
        "unsafe_to_safe": ("unsafe", "safe"),
        "safe_to_unsafe": ("safe", "unsafe"),
        "safe_to_safe": ("safe", "safe"),
    }
    plan = plan_blueprints(100, seed=3)

    for transition, labels in expected.items():
        blueprint = next(item for item in plan if item.transition == transition)
        pair = compile_policy_pair(blueprint)
        assert (pair.before_annotation.label, pair.after_annotation.label) == labels
        assert pair.before.stage == "before"
        assert pair.after.stage == "after"
        for view in (pair.before, pair.after):
            assert 3 <= len(view.active_policy) <= 8
            assert len({rule.rule_id for rule in view.active_policy}) == len(view.active_policy)
            assert len({rule.title for rule in view.active_policy}) == len(view.active_policy)
        if pair.before_annotation.label == "unsafe":
            assert pair.before_annotation.rule_title in {
                rule.title for rule in pair.before.active_policy
            }
        if pair.after_annotation.label == "unsafe":
            assert pair.after_annotation.rule_title in {
                rule.title for rule in pair.after.active_policy
            }


def test_planner_rejects_too_small_or_invalid_batches() -> None:
    import pytest

    from risk_agent.singguard_synthesis import plan_blueprints

    for count in (True, -1, 0, 31):
        with pytest.raises(ValueError, match="at least 32"):
            plan_blueprints(count, seed=42)


def test_narrowing_changes_only_the_after_policy_boundary() -> None:
    from risk_agent.singguard_synthesis import compile_policy_pair, plan_blueprints

    blueprint = next(
        item
        for item in plan_blueprints(100, seed=5)
        if item.transition == "unsafe_to_safe" and item.transformation == "narrow"
    )
    pair = compile_policy_pair(blueprint)
    before_rule = next(rule for rule in pair.before.active_policy if rule.rule_id.endswith("-001"))
    after_rule = next(
        rule
        for rule in pair.after.active_policy
        if rule.rule_id == before_rule.rule_id
    )

    assert "This narrowed version" not in before_rule.text
    assert "This narrowed version" in after_rule.text


def test_broadening_blueprints_require_implicit_content() -> None:
    from risk_agent.singguard_synthesis import plan_blueprints

    broadened = [
        item for item in plan_blueprints(100, seed=9) if item.transformation == "broaden"
    ]

    assert broadened
    assert all(item.difficulty == "implicit" for item in broadened)
