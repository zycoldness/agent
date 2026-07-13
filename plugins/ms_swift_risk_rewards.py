"""ms-swift 4.3 ORM registrations backed only by row-local ``solution``."""

from swift.rewards import ORM, orms

from risk_agent.rl_rewards import (
    evidence_exact_reward,
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


class RiskEvidenceExactORM(_CoreORM):
    core = staticmethod(evidence_exact_reward)


orms["risk_format_v1"] = RiskFormatORM
orms["risk_label_exact_v1"] = RiskLabelExactORM
orms["risk_rule_exact_v1"] = RiskRuleExactORM
orms["risk_evidence_exact_v1"] = RiskEvidenceExactORM
