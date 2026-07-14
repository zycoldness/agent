"""Planning and policy compilation for English SingGuard synthesis."""

from collections import Counter
import json


def test_planner_is_deterministic_and_balances_pilot_transitions() -> None:
    from risk_agent.singguard_synthesis import plan_blueprints

    first = plan_blueprints(100, seed=17)
    second = plan_blueprints(100, seed=17)

    assert first == second
    assert Counter(item.transition for item in first) == {
        "unsafe_to_unsafe": 25,
        "unsafe_to_safe": 25,
        "safe_to_unsafe": 25,
        "safe_to_safe": 25,
    }
    domain_counts = Counter(item.risk_domain for item in first)
    assert len(domain_counts) == 8
    assert max(domain_counts.values()) - min(domain_counts.values()) <= 1
    assert sum(item.before_thinking_type == "slow" for item in first) + sum(
        item.after_thinking_type == "slow" for item in first
    ) == 60


def test_full_plan_has_exact_domain_and_style_quotas() -> None:
    from risk_agent.singguard_synthesis import plan_blueprints

    plan = plan_blueprints(2000, seed=42)

    assert set(Counter(item.risk_domain for item in plan).values()) == {250}
    assert Counter(item.input_style for item in plan) == {
        "short_ad": 400,
        "social_post": 300,
        "livestream_pitch": 300,
        "product_listing": 300,
        "comment_or_review": 200,
        "sms_or_private_message": 200,
        "support_dialogue": 200,
        "neutral_information": 100,
    }
    assert Counter(item.conversation_type for item in plan) == {
        "query": 1400,
        "query_response": 600,
    }


def test_policy_compiler_produces_ordered_label_pairs_and_realistic_policy_sizes() -> None:
    from risk_agent.singguard_synthesis import compile_policy_pair, plan_blueprints

    expected = {
        "unsafe_to_unsafe": ("unsafe", "unsafe"),
        "unsafe_to_safe": ("unsafe", "safe"),
        "safe_to_unsafe": ("safe", "unsafe"),
        "safe_to_safe": ("safe", "safe"),
    }
    plan = plan_blueprints(100, seed=3)

    for transition, labels in expected.items():
        blueprint = next(item for item in plan if item.transition == transition)
        pair = compile_policy_pair(blueprint)
        assert (pair.before_annotation.label, pair.after_annotation.label) == labels
        assert pair.before.stage == "before"
        assert pair.after.stage == "after"
        for view in (pair.before, pair.after):
            assert 3 <= len(view.active_policy) <= 8
            assert len({rule.rule_id for rule in view.active_policy}) == len(view.active_policy)
            assert len({rule.title for rule in view.active_policy}) == len(view.active_policy)
        if pair.before_annotation.label == "unsafe":
            assert pair.before_annotation.rule_title in {
                rule.title for rule in pair.before.active_policy
            }
        if pair.after_annotation.label == "unsafe":
            assert pair.after_annotation.rule_title in {
                rule.title for rule in pair.after.active_policy
            }


def test_planner_rejects_too_small_or_invalid_batches() -> None:
    import pytest

    from risk_agent.singguard_synthesis import plan_blueprints

    for count in (True, -1, 0, 31):
        with pytest.raises(ValueError, match="at least 32"):
            plan_blueprints(count, seed=42)


def test_narrowing_changes_only_the_after_policy_boundary() -> None:
    from risk_agent.singguard_synthesis import compile_policy_pair, plan_blueprints

    blueprint = next(
        item
        for item in plan_blueprints(100, seed=5)
        if item.transition == "unsafe_to_safe" and item.transformation == "narrow"
    )
    pair = compile_policy_pair(blueprint)
    before_rule = next(rule for rule in pair.before.active_policy if rule.rule_id.endswith("-001"))
    after_rule = next(
        rule
        for rule in pair.after.active_policy
        if rule.rule_id == before_rule.rule_id
    )

    assert "This narrowed version" not in before_rule.text
    assert "This narrowed version" in after_rule.text


def test_broadening_blueprints_require_implicit_content() -> None:
    from risk_agent.singguard_synthesis import plan_blueprints

    broadened = [
        item for item in plan_blueprints(100, seed=9) if item.transformation == "broaden"
    ]

    assert broadened
    assert all(item.difficulty == "implicit" for item in broadened)


def test_generator_request_contains_constraints_but_no_final_label() -> None:
    from risk_agent.singguard_synthesis import build_generator_request, plan_blueprints

    blueprint = plan_blueprints(100, seed=11)[0]
    request = build_generator_request(blueprint, seed_text="A short public style seed.")
    serialized = json.dumps(request, sort_keys=True)

    assert blueprint.input_style in serialized
    assert blueprint.intended_facts[0] in serialized
    assert "A short public style seed." in serialized
    assert blueprint.transition not in serialized
    assert '"oracle":' not in serialized.lower()
    assert "rule_title" not in serialized


def test_blind_verifier_request_omits_oracle_transition_and_stage_names() -> None:
    from risk_agent.singguard_synthesis import (
        GeneratedContent,
        build_verifier_request,
        compile_policy_pair,
        plan_blueprints,
    )

    blueprint = next(
        item for item in plan_blueprints(100, seed=13) if item.transition == "unsafe_to_safe"
    )
    pair = compile_policy_pair(blueprint)
    content = GeneratedContent(
        query="A natural English sample.",
        style=blueprint.input_style,
    )

    request, opaque_to_stage = build_verifier_request(
        blueprint,
        pair,
        content,
        shuffle_seed=7,
    )
    serialized = json.dumps(request, sort_keys=True)

    for forbidden in (
        '"oracle":',
        '"transition":',
        "unsafe_to_safe",
        "source_label",
        "intended_facts",
        '"before"',
        '"after"',
    ):
        assert forbidden not in serialized
    assert set(opaque_to_stage.values()) == {"before", "after"}
    assert all(stage not in opaque_to_stage for stage in ("before", "after"))


def test_structured_generator_and_verifier_create_two_fast_source_examples() -> None:
    from risk_agent.singguard_synthesis import (
        examples_from_verifier,
        generate_content,
        plan_blueprints,
        compile_policy_pair,
        verify_content,
    )
    from risk_agent.teacher import CallableTeacher

    blueprint = next(
        item for item in plan_blueprints(100, seed=15) if item.transition == "unsafe_to_unsafe"
    ).model_copy(
        update={"before_thinking_type": "fast", "after_thinking_type": "fast"}
    )
    pair = compile_policy_pair(blueprint)
    generator = CallableTeacher(
        lambda _request: {
            "query": "Guaranteed results for every customer in seven days.",
            "response": None,
            "style": blueprint.input_style,
            "risk_cues": ["Guaranteed results"],
            "benign_cues": [],
        }
    )

    content, _generator_usage = generate_content(blueprint, generator)

    def verifier_call(request):
        views = []
        for view in request["policy_views"]:
            title = view["active_policy"][0]["title"]
            views.append(
                {
                    "opaque_id": view["opaque_id"],
                    "label": "unsafe",
                    "rule_title": title,
                    "evidence_quote": "Guaranteed results",
                    "confidence": 0.99,
                    "ambiguous": False,
                    "summary": None,
                    "checks": [],
                }
            )
        return {
            "views": views,
            "style": blueprint.input_style,
            "naturalness": 5,
            "template_like": False,
            "issues": [],
        }

    verifier = CallableTeacher(verifier_call)
    verdict, opaque_to_stage, _verifier_usage = verify_content(
        blueprint,
        pair,
        content,
        verifier,
        shuffle_seed=19,
    )
    examples = examples_from_verifier(
        blueprint,
        pair,
        content,
        verdict,
        opaque_to_stage,
    )

    assert len(examples) == 2
    assert [example.policy.stage for example in examples] == ["before", "after"]
    assert all(example.annotation.label == "unsafe" for example in examples)
    assert all(example.thinking_type == "fast" for example in examples)
    assert all(example.content.split_group == blueprint.family_id for example in examples)


def test_verifier_rejects_policy_external_rule_title() -> None:
    import pytest

    from risk_agent.singguard_synthesis import (
        GeneratedContent,
        compile_policy_pair,
        plan_blueprints,
        verify_content,
    )
    from risk_agent.teacher import CallableTeacher

    blueprint = plan_blueprints(100, seed=21)[0]
    pair = compile_policy_pair(blueprint)
    content = GeneratedContent(query="Natural content.", style=blueprint.input_style)

    def bad_verifier(request):
        return {
            "views": [
                {
                    "opaque_id": view["opaque_id"],
                    "label": "unsafe",
                    "rule_title": "Invented Rule",
                    "evidence_quote": "Natural content",
                    "confidence": 0.99,
                    "ambiguous": False,
                    "checks": [],
                }
                for view in request["policy_views"]
            ],
            "style": blueprint.input_style,
            "naturalness": 5,
            "template_like": False,
            "issues": [],
        }

    with pytest.raises(ValueError, match="active policy"):
        verify_content(
            blueprint,
            pair,
            content,
            CallableTeacher(bad_verifier),
            shuffle_seed=1,
        )


def test_batch_retries_one_candidate_and_exports_audited_bundle(tmp_path) -> None:
    import hashlib
    import json

    from risk_agent.singguard_synthesis import plan_blueprints, run_singguard_batch
    from risk_agent.teacher import CallableTeacher, TeacherBudget

    plan = tuple(
        item.model_copy(
            update={
                "transition": "unsafe_to_unsafe",
                "transformation": "rewrite",
                "before_thinking_type": "fast",
                "after_thinking_type": "fast",
                "conversation_type": "query",
            }
        )
        for item in plan_blueprints(32, seed=47)
    )
    generation_number = 0
    verifier_number = 0

    def generate(request):
        nonlocal generation_number
        generation_number += 1
        fingerprint = hashlib.sha256(
            f"{generation_number}:{json.dumps(request['constraints'], sort_keys=True)}".encode()
        ).hexdigest()
        return {
            "query": f"{fingerprint} Guaranteed outcome.",
            "response": None,
            "style": request["constraints"]["platform_style"],
            "risk_cues": ["Guaranteed outcome"],
            "benign_cues": [],
        }

    def verify(request):
        nonlocal verifier_number
        verifier_number += 1
        views = []
        for view in request["policy_views"]:
            views.append(
                {
                    "opaque_id": view["opaque_id"],
                    "label": "unsafe",
                    "rule_title": view["active_policy"][0]["title"],
                    "evidence_quote": "Guaranteed outcome",
                    "confidence": 0.50 if verifier_number == 1 else 0.99,
                    "ambiguous": False,
                    "summary": None,
                    "checks": [],
                }
            )
        return {
            "views": views,
            "style": request["declared_style"],
            "naturalness": 5,
            "template_like": False,
            "issues": [],
        }

    output = tmp_path / "pilot"
    budget = TeacherBudget(max_requests=100)
    result = run_singguard_batch(
        plan,
        generator=CallableTeacher(generate, budget=budget),
        verifier=CallableTeacher(verify, budget=budget),
        budget=budget,
        output_dir=output,
        seed=47,
        pilot=True,
    )

    expected = {
        "plan.jsonl",
        "accepted.jsonl",
        "rejected.jsonl",
        "quality_report.json",
        "review_sample.jsonl",
        "manifest.json",
        "checkpoint.json",
    }
    assert expected <= {path.name for path in output.iterdir()}
    assert {
        "train.jsonl",
        "dev.jsonl",
        "holdout.jsonl",
        "manifest.json",
    } == {path.name for path in (output / "ms_swift").iterdir()}
    assert result["status"] == "complete"
    assert result["accepted_anchors"] == 32
    assert result["accepted_examples"] == 64
    quality = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
    assert quality["first_pass_agreement_rate"] == 1.0
    rejected = [
        json.loads(line)
        for line in (output / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rejected[0]["codes"] == ["low_confidence"]
    assert "request" not in json.dumps(rejected).lower()
    source_rows = [
        json.loads(line)
        for line in (output / "accepted.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(source_rows) == 32
    assert all(len(row["examples"]) == 2 for row in source_rows)
    all_sft_rows = sum(
        len((output / "ms_swift" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines())
        for split in ("train", "dev", "holdout")
    )
    assert all_sft_rows == 64
    split_groups = {}
    for split in ("train", "dev", "holdout"):
        for line in (output / "ms_swift" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            user_text = row["messages"][1]["content"]
            split_groups.setdefault(user_text, set()).add(split)
    assert all(len(splits) == 1 for splits in split_groups.values())


def test_batch_refuses_existing_output_before_provider_calls(tmp_path) -> None:
    import pytest

    from risk_agent.singguard_synthesis import plan_blueprints, run_singguard_batch
    from risk_agent.teacher import CallableTeacher, TeacherBudget

    output = tmp_path / "existing"
    output.mkdir()
    calls = 0

    def provider(_request):
        nonlocal calls
        calls += 1
        return {}

    with pytest.raises(ValueError, match="must not exist"):
        run_singguard_batch(
            plan_blueprints(32, seed=1),
            generator=CallableTeacher(provider),
            verifier=CallableTeacher(provider),
            budget=TeacherBudget(max_requests=100),
            output_dir=output,
            seed=1,
        )
    assert calls == 0


def test_incomplete_batch_resumes_only_with_the_identical_plan(tmp_path) -> None:
    import hashlib
    import json
    import pytest

    from risk_agent.singguard_synthesis import plan_blueprints, run_singguard_batch
    from risk_agent.teacher import CallableTeacher, TeacherBudget

    plan = tuple(
        item.model_copy(
            update={
                "transition": "unsafe_to_unsafe",
                "transformation": "rewrite",
                "before_thinking_type": "fast",
                "after_thinking_type": "fast",
                "conversation_type": "query",
            }
        )
        for item in plan_blueprints(32, seed=53)
    )
    generation_number = 0

    def generate(request):
        nonlocal generation_number
        generation_number += 1
        fingerprint = hashlib.sha256(
            f"resume-{generation_number}:{json.dumps(request['constraints'], sort_keys=True)}".encode()
        ).hexdigest()
        return {
            "query": f"{fingerprint} Guaranteed outcome.",
            "response": None,
            "style": request["constraints"]["platform_style"],
        }

    def verify(request):
        return {
            "views": [
                {
                    "opaque_id": view["opaque_id"],
                    "label": "unsafe",
                    "rule_title": view["active_policy"][0]["title"],
                    "evidence_quote": "Guaranteed outcome",
                    "confidence": 0.99,
                    "ambiguous": False,
                }
                for view in request["policy_views"]
            ],
            "style": request["declared_style"],
            "naturalness": 5,
            "template_like": False,
        }

    output = tmp_path / "resumable"
    small_budget = TeacherBudget(max_requests=3)
    first = run_singguard_batch(
        plan,
        generator=CallableTeacher(generate, budget=small_budget),
        verifier=CallableTeacher(verify, budget=small_budget),
        budget=small_budget,
        output_dir=output,
        seed=53,
    )
    assert first["status"] == "incomplete"
    assert not (output / "ms_swift").exists()

    wrong = list(plan)
    wrong[0] = wrong[0].model_copy(update={"tone": "different"})
    blocked_budget = TeacherBudget(max_requests=100)
    with pytest.raises(ValueError, match="plan fingerprint"):
        run_singguard_batch(
            tuple(wrong),
            generator=CallableTeacher(generate, budget=blocked_budget),
            verifier=CallableTeacher(verify, budget=blocked_budget),
            budget=blocked_budget,
            output_dir=output,
            seed=53,
            resume=True,
        )

    with (output / "accepted.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"anchor_id": plan[1].anchor_id}) + "\n")
    with (output / "rejected.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "anchor_id": plan[1].anchor_id,
                    "attempt": 1,
                    "codes": ["schema"],
                    "oracle_agreement": False,
                }
            )
            + "\n"
        )

    resumed_budget = TeacherBudget(max_requests=100)
    second = run_singguard_batch(
        plan,
        generator=CallableTeacher(generate, budget=resumed_budget),
        verifier=CallableTeacher(verify, budget=resumed_budget),
        budget=resumed_budget,
        output_dir=output,
        seed=53,
        resume=True,
    )
    assert second["status"] == "complete"
    assert second["accepted_anchors"] == 32
    assert (output / "ms_swift" / "train.jsonl").exists()
