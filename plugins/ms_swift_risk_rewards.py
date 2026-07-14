"""ms-swift 4.3 ORM registrations backed only by row-local ``solution``."""

from swift.rewards import ORM, orms

from risk_agent.rl_rewards import (
    format_reward,
    label_exact_reward,
    rule_exact_reward,
)


class _CoreORM(ORM):
    core = staticmethod(format_reward)

    def __call__(self, completions, solution=None, **kwargs):
        return self.core(completions, solution=solution, **kwargs)


class RiskFormatORM(_CoreORM):
    core = staticmethod(format_reward)


class RiskLabelExactORM(_CoreORM):
    core = staticmethod(label_exact_reward)


class RiskRuleExactORM(_CoreORM):
    core = staticmethod(rule_exact_reward)


_REGISTRATIONS = {
    "risk_format_v1": RiskFormatORM,
    "risk_label_exact_v1": RiskLabelExactORM,
    "risk_rule_exact_v1": RiskRuleExactORM,
}
_collisions = sorted(set(_REGISTRATIONS) & set(orms))
if _collisions:
    raise RuntimeError(f"reward ORM already registered: {', '.join(_collisions)}")
orms.update(_REGISTRATIONS)
