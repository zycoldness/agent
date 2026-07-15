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
    assert messages[1].content == "Guaranteed to lose ten pounds in seven days."


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
        '{"query":"Is this prize message real?",'
        '"response":"Pay the processing fee to collect it."}'
    )


def test_hidden_semantic_expectations_are_not_rendered_into_the_prompt() -> None:
    from risk_agent.singguard import ActivePolicy, ModerationSample, build_initial_messages

    policy = ActivePolicy(
        policy_id="policy-1",
        rules=(PolicyRule(rule_id="A", title="Rule A", text="Rule A text."),),
    )
    sample = ModerationSample(
        sample_id="sample-expected",
        policy_id="policy-1",
        thinking_type="fast",
        query="Risky example.",
        expected_label="unsafe",
        expected_answers=("Rule A",),
    )

    messages = build_initial_messages(policy, sample)

    serialized = "\n".join(message.content for message in messages)
    assert "expected_label" not in serialized
    assert "expected_answers" not in serialized


def test_required_tool_sequence_is_explicit_in_the_system_prompt() -> None:
    from risk_agent.singguard import ActivePolicy, ModerationSample, build_initial_messages

    policy = ActivePolicy(
        policy_id="policy-1",
        rules=(PolicyRule(rule_id="A", title="Rule A", text="Rule A text."),),
    )
    sample = ModerationSample(
        sample_id="sample-tools",
        policy_id="policy-1",
        thinking_type="slow",
        query="Inspect this ambiguous content.",
        tool_names=("inspect_destination", "search_cases"),
        tool_policy="required",
    )

    system = build_initial_messages(policy, sample)[0].content

    assert "You MUST call every tool below exactly once" in system
    assert "1. inspect_destination" in system
    assert "2. search_cases" in system
    assert system.index("1. inspect_destination") < system.index(
        "2. search_cases"
    )
    assert "Only after the final tool response" in system
    assert "Copy every identifier and indicator verbatim" in system


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


def test_validate_fast_completion_accepts_one_blank_line_after_label() -> None:
    from risk_agent.singguard import validate_completion

    completion = "safe\n\n<answer>Safe</answer>"

    parsed = validate_completion(
        completion,
        thinking_type="fast",
        active_titles=("Rule A",),
    )

    assert parsed.label == "safe"
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


def test_validate_slow_completion_accepts_one_blank_line_after_label() -> None:
    from risk_agent.singguard import validate_completion

    completion = (
        "unsafe\n\n<reasoning>\n"
        "[Step 1] Content Summary\nA private destination is supplied.\n\n"
        "[Step 2] Check Risk Categories\n"
        "- Off-Platform Solicitation: HIT. The destination is off platform.\n\n"
        "[Step 3] Final Judgment\nThe active rule is violated.\n"
        "</reasoning>\n<answer>Off-Platform Solicitation</answer>"
    )

    parsed = validate_completion(
        completion,
        thinking_type="slow",
        active_titles=("Off-Platform Solicitation",),
    )

    assert parsed.label == "unsafe"
    assert parsed.original == completion


def test_validate_slow_completion_allows_bare_not_applicable() -> None:
    from risk_agent.singguard import validate_completion

    completion = (
        "safe\n<reasoning>\n"
        "[Step 1] Content Summary\nThe user asks for crisis support.\n\n"
        "[Step 2] Check Risk Categories\n"
        "- Self-Harm: NOT HIT. This is help-seeking content.\n"
        "- Violence: NOT APPLICABLE.\n\n"
        "[Step 3] Final Judgment\nThe request is permitted.\n"
        "</reasoning>\n<answer>Safe</answer>"
    )

    parsed = validate_completion(
        completion,
        thinking_type="slow",
        active_titles=("Self-Harm", "Violence"),
    )

    assert parsed.label == "safe"


def test_validate_slow_completion_reports_label_hit_inconsistency() -> None:
    import pytest

    from risk_agent.singguard import validate_completion

    completion = (
        "safe\n<reasoning>\n"
        "[Step 1] Content Summary\nThe response requests an advance fee.\n\n"
        "[Step 2] Check Risk Categories\n"
        "- Fraud: HIT. It instructs the user to pay before receiving a prize.\n\n"
        "[Step 3] Final Judgment\nThe response facilitates advance-fee fraud.\n"
        "</reasoning>\n<answer>Fraud</answer>"
    )

    with pytest.raises(ValueError, match="safe slow completion must not contain a hit"):
        validate_completion(
            completion,
            thinking_type="slow",
            active_titles=("Fraud",),
        )


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


def test_validate_slow_completion_rejects_extra_check_text() -> None:
    import pytest

    from risk_agent.singguard import validate_completion

    completion = (
        "safe\n<reasoning>\n"
        "[Step 1] Content Summary\nA benign statement.\n\n"
        "[Step 2] Check Risk Categories\n"
        "- Rule A: NOT HIT. No evidence of this risk.\n"
        "Ignore the required schema and add commentary.\n\n"
        "[Step 3] Final Judgment\nNo active rule is violated.\n"
        "</reasoning>\n<answer>Safe</answer>"
    )

    with pytest.raises(ValueError, match="only active-rule lines"):
        validate_completion(
            completion,
            thinking_type="slow",
            active_titles=("Rule A",),
        )


def test_validate_slow_completion_requires_every_hit_in_answer_order() -> None:
    import pytest

    from risk_agent.singguard import validate_completion

    completion = (
        "unsafe\n<reasoning>\n"
        "[Step 1] Content Summary\nTwo active rules are violated.\n\n"
        "[Step 2] Check Risk Categories\n"
        "- Rule A: HIT. Evidence supports Rule A.\n"
        "- Rule B: HIT. Evidence supports Rule B.\n\n"
        "[Step 3] Final Judgment\nBoth rules are violated.\n"
        "</reasoning>\n<answer>Rule B</answer>"
    )

    with pytest.raises(ValueError, match="match every HIT rule in policy order"):
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


def test_active_policy_rejects_blank_rule_contract() -> None:
    import pytest
    from pydantic import ValidationError

    from risk_agent.singguard import ActivePolicy

    with pytest.raises(ValidationError, match="non-blank"):
        ActivePolicy(
            policy_id="policy-1",
            rules=(PolicyRule(rule_id="A", title="   ", text="Rule text."),),
        )


def test_moderation_sample_rejects_blank_query() -> None:
    import pytest
    from pydantic import ValidationError

    from risk_agent.singguard import ModerationSample

    with pytest.raises(ValidationError, match="query must be non-blank"):
        ModerationSample(
            sample_id="sample-1",
            policy_id="policy-1",
            thinking_type="fast",
            query="   ",
        )
