"""Rendering helpers for the active policy available to an assessor."""

from risk_agent.contracts import PolicyRule


def render_active_policy(rules: tuple[PolicyRule, ...]) -> str:
    """Render the complete active policy in deterministic priority order."""

    lines = ["当前生效规则："]
    for rule in sorted(rules, key=lambda item: (item.priority, item.rule_id)):
        lines.append(f"[{rule.rule_id}] {rule.title}: {rule.text}")
        lines.extend(f"例外：{exception}" for exception in rule.exceptions)
    return "\n".join(lines)
