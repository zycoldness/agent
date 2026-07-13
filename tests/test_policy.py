from risk_agent.contracts import PolicyRule
from risk_agent.policy import render_active_policy


def test_render_active_policy_orders_rules_and_includes_exceptions() -> None:
    rules = (
        PolicyRule(
            rule_id="R2",
            title="后置规则",
            text="第二条内容",
            exceptions=("允许公益场景",),
            priority=20,
        ),
        PolicyRule(
            rule_id="R1",
            title="前置规则",
            text="第一条内容",
            priority=10,
        ),
    )

    assert render_active_policy(rules) == (
        "当前生效规则：\n"
        "[R1] 前置规则: 第一条内容\n"
        "[R2] 后置规则: 第二条内容\n"
        "例外：允许公益场景"
    )


def test_render_active_policy_returns_only_header_when_empty() -> None:
    assert render_active_policy(()) == "当前生效规则："
