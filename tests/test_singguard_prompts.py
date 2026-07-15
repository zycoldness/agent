"""Versioned prompt contracts shared by synthesis, training, and inference."""

from risk_agent.contracts import PolicyRule


def _rule() -> PolicyRule:
    return PolicyRule(
        rule_id="EFF-001",
        title="Guaranteed Outcome",
        text="Do not guarantee a specific health or weight-loss result.",
        exceptions=("Neutral reporting without endorsement is allowed.",),
    )


def test_runtime_policy_replaces_default_taxonomy() -> None:
    from risk_agent.singguard_prompts import prompt_sha256, render_guard_prompt

    prompt = render_guard_prompt((_rule(),), thinking_type="slow")

    assert "<thinking_type>slow</thinking_type>" in prompt
    assert "### Guaranteed Outcome" in prompt
    assert "Neutral reporting without endorsement is allowed." in prompt
    assert "A. Sexual Content Risk" not in prompt
    assert "{{" not in prompt
    assert len(prompt_sha256("guard")) == 64


def test_slow_prompt_exposes_the_exact_validator_grammar() -> None:
    from risk_agent.singguard_prompts import render_guard_prompt

    prompt = render_guard_prompt((_rule(),), thinking_type="slow")

    assert "<reasoning>" in prompt
    assert "[Step 1] Content Summary" in prompt
    assert "[Step 2] Check Risk Categories" in prompt
    assert "[Step 3] Final Judgment" in prompt
    assert (
        "- Guaranteed Outcome: <HIT | NOT HIT | NOT APPLICABLE | "
        "OBSERVED BUT NOT UNSAFE>. <evidence>"
    ) in prompt
    assert "Do not add text outside this structure" in prompt


def test_agent_prompt_extends_policy_core_with_bounded_tools() -> None:
    from risk_agent.singguard_prompts import render_agent_prompt

    prompt = render_agent_prompt((_rule(),))

    assert "### Guaranteed Outcome" in prompt
    assert "get_rule_detail" in prompt
    assert "inspect_evidence" in prompt
    assert "At most three assistant turns" in prompt
    assert '{"tool":"...","arguments":{...}}' in prompt


def test_generator_and_verifier_prompts_are_versioned_and_role_separated() -> None:
    from risk_agent.singguard_prompts import load_prompt, prompt_sha256

    generator = load_prompt("generator")
    verifier = load_prompt("verifier")

    assert "Do not classify the content" in generator
    assert "Do not infer an expected transition" in verifier
    assert generator != verifier
    assert prompt_sha256("generator") != prompt_sha256("verifier")
