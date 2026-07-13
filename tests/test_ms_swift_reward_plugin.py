"""The external plugin remains a thin, offline ms-swift adapter."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).parents[1]
PLUGIN = ROOT / "plugins" / "ms_swift_risk_rewards.py"
SOLUTION = '{"tool":"final_decision","arguments":{"label":"unsafe","rule_id":"R-1","evidence_ids":[],"confidence":1.0}}'


def test_plugin_registers_four_unique_orms_and_delegates_to_pure_core(monkeypatch) -> None:
    registry: dict[str, type] = {}
    swift = types.ModuleType("swift")
    rewards = types.ModuleType("swift.rewards")

    class ORM:
        def __init__(self, args=None, **kwargs):
            self.args = args

    rewards.ORM = ORM
    rewards.orms = registry
    swift.rewards = rewards
    monkeypatch.setitem(sys.modules, "swift", swift)
    monkeypatch.setitem(sys.modules, "swift.rewards", rewards)

    spec = importlib.util.spec_from_file_location("risk_rewards_test_plugin", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    assert set(registry) == {
        "risk_format_v1", "risk_label_exact_v1", "risk_rule_exact_v1", "risk_evidence_exact_v1"
    }
    assert registry["risk_label_exact_v1"]()([SOLUTION], solution=[SOLUTION]) == [1.0]
    source = PLUGIN.read_text(encoding="utf-8")
    assert "requests" not in source and "open(" not in source and "Oracle" not in source
