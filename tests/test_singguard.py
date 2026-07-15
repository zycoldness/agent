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


def test_validate_fast_completion_rejects_inactive_answer() -> None:
    import pytest

    from risk_agent.singguard import validate_completion

    with pytest.raises(ValueError, match="active policy"):
        validate_completion(
            "unsafe\n<answer>Unknown Rule</answer>",
            thinking_type="fast",
            active_titles=("Deceptive Efficacy",),
        )


def test_validate_fast_completion_preserves_multiple_active_answers() -> None:
    from risk_agent.singguard import validate_completion

    completion = (
        "unsafe\n<answer>Deceptive Efficacy\n"
        "Off-Platform Solicitation</answer>"
    )

    parsed = validate_completion(
        completion,
        thinking_type="fast",
        active_titles=("Deceptive Efficacy", "Off-Platform Solicitation"),
    )

    assert parsed.label == "unsafe"
    assert parsed.answers == (
        "Deceptive Efficacy",
        "Off-Platform Solicitation",
    )
    assert parsed.original == completion


def test_validate_slow_completion_requires_every_rule_in_order() -> None:
    from risk_agent.singguard import validate_completion

    completion = (
        "unsafe\n<reasoning>\n"
        "[Step 1] Content Summary\nA guaranteed claim.\n\n"
        "[Step 2] Check Risk Categories\n"
        "- Deceptive Efficacy: HIT. The text says guaranteed.\n"
        "- Off-Platform Solicitation: NOT HIT. No destination.\n\n"
        "[Step 3] Final Judgment\nThe first active rule is violated.\n"
        "</reasoning>\n<answer>Deceptive Efficacy</answer>"
    )

    parsed = validate_completion(
        completion,
        thinking_type="slow",
        active_titles=("Deceptive Efficacy", "Off-Platform Solicitation"),
    )

    assert parsed.label == "unsafe"
    assert parsed.answers == ("Deceptive Efficacy",)


def test_validate_slow_completion_rejects_reordered_rule_checks() -> None:
    import pytest

    from risk_agent.singguard import validate_completion

    completion = (
        "safe\n<reasoning>\n"
        "[Step 1] Content Summary\nA benign statement.\n\n"
        "[Step 2] Check Risk Categories\n"
        "- Rule B: NOT HIT. No evidence.\n"
        "- Rule A: NOT HIT. No evidence.\n\n"
        "[Step 3] Final Judgment\nNo active rule is violated.\n"
        "</reasoning>\n<answer>Safe</answer>"
    )

    with pytest.raises(ValueError, match="policy order"):
        validate_completion(
            completion,
            thinking_type="slow",
            active_titles=("Rule A", "Rule B"),
        )


def test_render_training_row_keeps_gemini_completion_verbatim() -> None:
    from risk_agent.singguard import (
        ActivePolicy,
        ModerationSample,
        build_initial_messages,
        render_training_row,
    )

    policy = ActivePolicy(
        policy_id="policy-1",
        rules=(PolicyRule(rule_id="A", title="Rule A", text="Rule A text."),),
    )
    sample = ModerationSample(
        sample_id="sample-1",
        policy_id="policy-1",
        thinking_type="fast",
        query="Example content.",
    )
    completion = "safe\n<answer>Safe</answer>"

    row = render_training_row(build_initial_messages(policy, sample), completion)

    assert row["messages"][-1] == {"role": "assistant", "content": completion}
    assert "tools" not in row
