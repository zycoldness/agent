"""SingGuard-specific contracts and Appendix-style SFT rendering."""

from risk_agent.contracts import PolicyRule


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
