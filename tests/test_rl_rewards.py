"""Pure deterministic Track A reward tests."""

from __future__ import annotations

import math

import pytest

from risk_agent.rl_rewards import evidence_exact_reward, format_reward, label_exact_reward, rule_exact_reward


SOLUTION = '{"tool":"final_decision","arguments":{"label":"unsafe","rule_id":"R-1","evidence_ids":["e-1"],"confidence":1.0,"risk_level":"P1"}}'


@pytest.mark.parametrize("reward", [format_reward, label_exact_reward, rule_exact_reward, evidence_exact_reward])
def test_rewards_accept_strict_exact_decision_and_return_finite_floats(reward) -> None:
    values = reward([SOLUTION], solution=[SOLUTION])
    assert values == [1.0]
    assert all(isinstance(value, float) and math.isfinite(value) for value in values)


@pytest.mark.parametrize(
    "completion",
    [
        '{"tool":"final_decision","tool":"final_decision","arguments":{}}',
        SOLUTION + " trailing",
        SOLUTION.replace("1.0", "NaN"),
        SOLUTION[:-1] + ',"unknown":1}',
        '{"tool":"other","arguments":{"label":"unsafe","rule_id":"R-1","evidence_ids":["e-1"],"confidence":1}}',
    ],
)
def test_format_reward_rejects_duplicate_nonfinite_trailing_unknown_and_wrong_tool(completion: str) -> None:
    assert format_reward([completion], solution=[SOLUTION]) == [0.0]


def test_component_rewards_have_explicit_none_and_ordered_evidence_semantics() -> None:
    safe = '{"tool":"final_decision","arguments":{"label":"safe","rule_id":null,"evidence_ids":[],"confidence":1.0}}'
    missing_rule = '{"tool":"final_decision","arguments":{"label":"safe","evidence_ids":[],"confidence":1.0}}'
    reversed_evidence = SOLUTION.replace('["e-1"]', '["e-2","e-1"]')
    target_evidence = SOLUTION.replace('["e-1"]', '["e-1","e-2"]')

    assert rule_exact_reward([safe], solution=[safe]) == [1.0]
    assert rule_exact_reward([missing_rule], solution=[safe]) == [0.0]
    assert evidence_exact_reward([reversed_evidence], solution=[target_evidence]) == [0.0]


@pytest.mark.parametrize("bad_solution", [None, "x", [SOLUTION, SOLUTION], [1]])
def test_all_rewards_fail_closed_on_bad_kwargs_or_lengths(bad_solution) -> None:
    for reward in (format_reward, label_exact_reward, rule_exact_reward, evidence_exact_reward):
        assert reward([SOLUTION], solution=bad_solution) == [0.0]


def test_all_rewards_fail_closed_for_non_string_completion_without_throwing() -> None:
    for reward in (format_reward, label_exact_reward, rule_exact_reward, evidence_exact_reward):
        assert reward([None], solution=[SOLUTION]) == [0.0]
