"""SingGuard-specific contracts and Appendix-style SFT rendering."""

from risk_agent.contracts import PolicyRule


def test_build_prompt_injects_all_active_rules_once() -> None:
    from risk_agent.singguard import ActivePolicy, ModerationSample, build_initial_messages

    policy = ActivePolicy(
        policy_id="commerce-v1",
        rules=(
            PolicyRule(
                rule_id="EFF-001",
                title="Deceptive Efficacy",
                text="No guaranteed result.",
            ),
            PolicyRule(
                rule_id="LEAD-001",
                title="Off-Platform Solicitation",
                text="No private-channel redirection.",
            ),
        ),
    )
    sample = ModerationSample(
        sample_id="sample-1",
        policy_id="commerce-v1",
        thinking_type="slow",
        query="Guaranteed to lose ten pounds in seven days.",
    )

    messages = build_initial_messages(policy, sample)

    assert [message.role for message in messages] == ["system", "user"]
    assert messages[0].content.count("### Deceptive Efficacy") == 1
    assert messages[0].content.count("### Off-Platform Solicitation") == 1
    assert "<thinking_type>slow</thinking_type>" in messages[0].content
    assert messages[1].content == (
        "[user]: Guaranteed to lose ten pounds in seven days."
    )


def test_build_prompt_renders_query_response_conversation() -> None:
    from risk_agent.singguard import ActivePolicy, ModerationSample, build_initial_messages

    policy = ActivePolicy(
        policy_id="response-v1",
        rules=(
            PolicyRule(
                rule_id="FRAUD-001",
                title="Fraud",
                text="Do not facilitate fraud.",
            ),
        ),
    )
    sample = ModerationSample(
        sample_id="sample-2",
        policy_id="response-v1",
        thinking_type="fast",
        query="Is this prize message real?",
        response="Pay the processing fee to collect it.",
    )

    messages = build_initial_messages(policy, sample)

    assert messages[1].content == (
        "[user]: Is this prize message real?\n"
        "[assistant]: Pay the processing fee to collect it."
    )


def test_active_policy_rejects_duplicate_rule_titles() -> None:
    import pytest

    from risk_agent.singguard import ActivePolicy

    with pytest.raises(ValueError, match="duplicate rule titles"):
        ActivePolicy(
            policy_id="duplicates",
            rules=(
                PolicyRule(rule_id="A", title="Same", text="First."),
                PolicyRule(rule_id="B", title="Same", text="Second."),
            ),
        )


def test_fast_renderer_uses_active_rule_title_and_appendix_grammar() -> None:
    from risk_agent.singguard import (
        ContentSample,
        PolicyView,
        SingGuardAnnotation,
        SingGuardExample,
        render_sft_row,
    )

    rule = PolicyRule(
        rule_id="AD-001",
        title="Absolute efficacy claim",
        text="Do not guarantee a weight-loss result.",
    )
    example = SingGuardExample(
        content=ContentSample(
            sample_id="asset-1",
            split_group="asset-1",
            query="Guaranteed to lose ten pounds in seven days.",
        ),
        policy=PolicyView(
            view_id="asset-1-full",
            active_policy=(rule,),
            style="full",
            transition="unsafe_to_unsafe",
        ),
        annotation=SingGuardAnnotation(
            label="unsafe",
            rule_title="Absolute efficacy claim",
        ),
        thinking_type="fast",
    )

    row = render_sft_row(example)

    assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
    assert "<thinking_type>fast</thinking_type>" in row["messages"][0]["content"]
    assert "Absolute efficacy claim" in row["messages"][0]["content"]
    assert "Do not guarantee a weight-loss result." in row["messages"][0]["content"]
    assert "emit the triggered active rule in an <answer> tag" in row["messages"][0]["content"]
    assert row["messages"][1]["content"] == "[user]: Guaranteed to lose ten pounds in seven days."
    assert row["messages"][2]["content"] == (
        "unsafe\n<answer>Absolute efficacy claim</answer>"
    )


def test_slow_renderer_checks_every_active_rule_before_final_answer() -> None:
    from risk_agent.singguard import (
        ContentSample,
        PolicyView,
        RuleCheck,
        SingGuardAnnotation,
        SingGuardExample,
        render_sft_row,
    )

    efficacy = PolicyRule(
        rule_id="AD-001",
        title="Absolute efficacy claim",
        text="Do not guarantee a weight-loss result.",
    )
    disclosure = PolicyRule(
        rule_id="AD-002",
        title="Required disclosure",
        text="Disclose material limitations.",
    )
    example = SingGuardExample(
        content=ContentSample(
            sample_id="asset-1",
            split_group="asset-1",
            query="Guaranteed to lose ten pounds in seven days.",
        ),
        policy=PolicyView(
            view_id="asset-1-slow",
            active_policy=(efficacy, disclosure),
            style="full",
            transition="unsafe_to_unsafe",
        ),
        annotation=SingGuardAnnotation(
            label="unsafe",
            rule_title="Absolute efficacy claim",
            summary="The advertisement guarantees a quantified result within seven days.",
            checks=(
                RuleCheck(
                    rule_title="Absolute efficacy claim",
                    verdict="hit",
                    evidence="The phrase guarantees losing ten pounds in seven days.",
                ),
                RuleCheck(
                    rule_title="Required disclosure",
                    verdict="not_hit",
                    evidence="No separate disclosure violation is established by this sample.",
                ),
            ),
        ),
        thinking_type="slow",
    )

    row = render_sft_row(example)
    system = row["messages"][0]["content"]
    answer = row["messages"][2]["content"]

    assert "assess each active Risk Category one by one" in system
    assert answer.startswith("unsafe\n<reasoning>\n[Step 1] Content Summary")
    assert "- Absolute efficacy claim: HIT." in answer
    assert "- Required disclosure: NOT HIT." in answer
    assert "[Step 3] Final Judgment" in answer
    assert answer.endswith("</reasoning>\n<answer>Absolute efficacy claim</answer>")
