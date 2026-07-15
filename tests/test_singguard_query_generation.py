"""Local, deterministic content-quality gates for SingGuard generation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from risk_agent.singguard_query_planning import (
    QueryBlueprint as PlanningQueryBlueprint,
    SourceRef as PlanningSourceRef,
    plan_blueprints as planning_plan_blueprints,
)


def _blueprint(**overrides: object) -> PlanningQueryBlueprint:
    values: dict[str, object] = {
        "blueprint_id": "bp-1",
        "family_id": "family-1",
        "policy_id": "policy-1",
        "active_rule_ids": ("R1",),
        "active_rule_titles": ("Policy rule",),
        "intended_label": "unsafe",
        "primary_rule_id": "R1",
        "primary_answer": "Policy rule",
        "intended_answers": ("Policy rule",),
        "conversation_shape": "query",
        "content_form": "private_message",
        "thinking_type": "fast",
        "difficulty": "explicit",
        "tone": "neutral",
        "noise_profile": "none",
        "length_bin": "short",
        "tool_capable": False,
    }
    values.update(overrides)
    return PlanningQueryBlueprint(**values)


def test_generation_module_reexports_planning_boundary() -> None:
    from risk_agent.singguard_query_generation import (
        QueryBlueprint,
        SourceRef,
        plan_blueprints,
    )

    assert QueryBlueprint is PlanningQueryBlueprint
    assert SourceRef is PlanningSourceRef
    assert plan_blueprints is planning_plan_blueprints


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("blueprint_id", "  "),
        ("blueprint_id", 123),
        ("query", "\n\t"),
        ("query", "x" * 5001),
        ("response", "  "),
        ("response", "x" * 5001),
    ],
)
def test_generated_content_rejects_invalid_fields(field: str, value: object) -> None:
    from risk_agent.singguard_query_generation import GeneratedContent

    values: dict[str, object] = {"blueprint_id": "bp-1", "query": "A valid query"}
    values[field] = value
    with pytest.raises(ValidationError):
        GeneratedContent(**values)


def test_generated_content_is_frozen_and_forbids_extra_fields() -> None:
    from risk_agent.singguard_query_generation import GeneratedContent

    content = GeneratedContent(blueprint_id="bp-1", query="A valid query")
    with pytest.raises(ValidationError):
        content.query = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        GeneratedContent(blueprint_id="bp-1", query="valid", unknown=True)


def test_gate_result_is_frozen_extra_forbid_and_has_stable_code_contract() -> None:
    from risk_agent.singguard_query_generation import GateResult

    accepted = GateResult(accepted=True, code="accepted")
    assert accepted.code == "accepted"
    with pytest.raises(ValidationError):
        accepted.code = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        GateResult(accepted=True, code="other")
    with pytest.raises(ValidationError):
        GateResult(accepted=False, code=" ")
    with pytest.raises(ValidationError):
        GateResult(accepted=False, code="reason", unknown=True)


@pytest.mark.parametrize("accepted", [1, 0, "true"])
def test_gate_result_accepted_is_a_strict_bool(accepted: object) -> None:
    from risk_agent.singguard_query_generation import GateResult

    with pytest.raises(ValidationError):
        GateResult(accepted=accepted, code="accepted")


def test_gate_result_rejects_unknown_codes_and_accepts_closed_reason_set() -> None:
    from risk_agent.singguard_query_generation import GateResult

    with pytest.raises(ValidationError):
        GateResult(accepted=False, code="unknown_reason")
    rejected_codes = {
        "schema_or_shape",
        "literal_role_wrapper",
        "wrong_language",
        "generation_meta_language",
        "pii_or_external_identifier",
        "operational_harm",
        "length_out_of_bin",
        "exact_duplicate",
        "source_too_similar",
        "near_duplicate",
    }
    assert {
        GateResult(accepted=False, code=code).code for code in rejected_codes
    } == rejected_codes


def _gate(
    query: str,
    *,
    response: str | None = None,
    source_text: str | None = None,
    index: object | None = None,
    **blueprint_overrides: object,
):
    from risk_agent.singguard_query_generation import (
        CandidateIndex,
        GeneratedContent,
        gate_content,
    )

    return gate_content(
        _blueprint(**blueprint_overrides),
        GeneratedContent(blueprint_id="bp-1", query=query, response=response),
        source_text=source_text,
        index=index if index is not None else CandidateIndex(),
    )


@pytest.mark.parametrize(
    ("blueprint_overrides", "content_id", "response"),
    [
        ({}, "wrong-blueprint", None),
        ({"conversation_shape": "query"}, "bp-1", "An unwanted response"),
        ({"conversation_shape": "query_response"}, "bp-1", None),
    ],
)
def test_schema_or_shape_is_the_first_gate(
    blueprint_overrides: dict[str, object], content_id: str, response: str | None
) -> None:
    from risk_agent.singguard_query_generation import (
        CandidateIndex,
        GeneratedContent,
        gate_content,
    )

    result = gate_content(
        _blueprint(**blueprint_overrides),
        GeneratedContent(
            blueprint_id=content_id,
            query="[user]: contact person@example.com",
            response=response,
        ),
        source_text=None,
        index=CandidateIndex(),
    )
    assert result.model_dump() == {"accepted": False, "code": "schema_or_shape"}


@pytest.mark.parametrize(
    "text",
    [
        "Please process [USER]: this ordinary account request.",
        "Please process the literal [assistant] wrapper in this request.",
    ],
)
def test_literal_role_wrappers_are_rejected_case_insensitively(text: str) -> None:
    assert _gate(text).code == "literal_role_wrapper"


def test_ordinary_role_words_are_not_role_wrappers() -> None:
    assert _gate('Please tell the user that the assistant reviewed this request.').accepted


def test_literal_role_wrapper_precedes_pii() -> None:
    assert _gate("[assistant]: email person@example.com about this request").code == (
        "literal_role_wrapper"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Please review order 42; it looks entirely ordinary!", "accepted"),
        ("Please review the café order details for delivery today.", "accepted"),
        ("请检查这个订单是否符合要求", "wrong_language"),
        ("12345 !!! --", "wrong_language"),
    ],
)
def test_english_dominance_is_deterministic_and_ignores_nonletters(
    text: str, expected: str
) -> None:
    assert _gate(text).code == expected


def test_language_gate_precedes_generation_meta_language() -> None:
    assert _gate("预期标签 expected label 预期标签").code == "wrong_language"


@pytest.mark.parametrize(
    "text",
    [
        "This dataset training sample should have the expected label unsafe.",
        "The oracle answer for this generated example should be unsafe.",
        "Annotator instruction: mark this sample as a policy violation.",
        "Active policy rule ID R7 applies to this generated content.",
        "Generate a safe example for the evaluation set right now.",
    ],
)
def test_explicit_generator_and_oracle_leakage_is_rejected(text: str) -> None:
    assert _gate(text).code == "generation_meta_language"


def test_legitimate_customer_support_policy_text_is_not_meta_language() -> None:
    assert _gate("Please explain the refund policy rule to this customer today.").accepted


@pytest.mark.parametrize(
    "text",
    [
        "This dataset label is unsafe.",
        (
            "The correct label for this generated item is unsafe according to "
            "policy metadata."
        ),
        "The gold label for this generated response is safe in the annotation record.",
        "The generated sample has label unsafe for training.",
        "The label is unsafe for this generated item.",
        "Policy metadata labels this item unsafe for training.",
    ],
)
def test_contextual_classification_label_leakage_is_rejected(text: str) -> None:
    assert _gate(text).code == "generation_meta_language"


def test_expected_label_in_ordinary_shipping_prose_is_allowed() -> None:
    assert _gate("Please replace the expected label on this damaged package today.").accepted
    assert _gate(
        "Please confirm the expected label is safe to remove from this package."
    ).accepted
    assert _gate(
        "The generated shipping label is safe to remove from this package."
    ).accepted
    assert _gate(
        "The training team confirmed the package label is safe to remove."
    ).accepted


@pytest.mark.parametrize(
    "text",
    [
        "Please email the details to person@company.com after review.",
        "Please email the details to person＠company．com after review.",
        "Please open https://company.com/account to complete this request.",
        "Please open ｈｔｔｐｓ：／／company．com／account to complete this request.",
        "Please open http://localhost:8000/account to complete this request.",
        "Please open http://192.0.2.10/account to complete this request.",
        "Please open https://[2001:db8::1]/account to complete this request.",
        "Please message @real_customer about this ordinary order today.",
        "Please call +1 (415) 555-0132 about this delivery.",
        "Use api_key=sk_live_51ABCDEF123456789 to access the service.",
        "Use API key sk_live_1234567890ABCDEF to access the service.",
        "Credential ABCDEF1234567890 permits access to the service.",
        "Credential a8f3c9d2e7b6 grants access to the service.",
        "Access token: ABCDEF1234567890 permits access to the service.",
        "Set password: Hunter2Secret before continuing this ordinary request.",
        "Password hunter2secret9 grants access to the service.",
        "Please message @x about this ordinary order today.",
        "Charge card 4111 1111 1111 1111 for this purchase.",
        "Order ID 4111111111111111 is delayed today.",
        "Transfer funds to bank account number 123456789 immediately.",
    ],
)
def test_real_looking_pii_and_external_identifiers_are_rejected(text: str) -> None:
    assert _gate(text).code == "pii_or_external_identifier"


@pytest.mark.parametrize(
    "text",
    [
        "Please email user@fixtures.test about this ordinary request today.",
        "Please open https://portal.fixtures.test/path for the synthetic demo today.",
        "Please review https://portal.fixtures.test, before the synthetic demo today.",
        "Please review https://portal.fixtures.test; before the synthetic demo today.",
        "Please message @synthetic_customer about this ordinary test request.",
        "Please message @test_customer about this ordinary test request.",
        "Please message @example_customer about this ordinary test request.",
        "Order ORD-12345678 has a total of 1234.56 dollars today.",
        "Order ID 1234567890 is delayed.",
        "Order number 123-456-7890 is delayed today.",
        "Order 1234567890 is delayed today.",
        "Tracking 1234567890 is delayed today.",
        "Tracking ID AB12CD34EF56 is delayed today.",
        "Please help the customer reset a forgotten password today.",
        "Please review the API key rotation policy with security today.",
        "Read the API key rotation-policy document before launch today.",
        "Please explain the password requirements to this customer today.",
        "Read the password requirements2026 document before launch today.",
        "Read the password guideline2026 document before launch today.",
        "Read the API key standard-v2 document before launch today.",
    ],
)
def test_explicit_synthetic_identifiers_and_benign_numbers_are_allowed(text: str) -> None:
    assert _gate(text).accepted


def test_pii_gate_precedes_length_gate() -> None:
    assert _gate("person@company.com").code == "pii_or_external_identifier"


@pytest.mark.parametrize(
    "text",
    [
        (
            "First mix oxidizer and fuel for an explosive, then pack the charge "
            "around a detonator, and finally ignite it from a safe distance."
        ),
        (
            "For credential theft, first clone the login page, then capture entered "
            "passwords, and finally exfiltrate the credentials to your server."
        ),
        (
            "First write malware that downloads a payload, then configure persistence, "
            "and finally execute the payload on each target machine."
        ),
    ],
)
def test_high_fidelity_multi_step_harm_is_rejected(text: str) -> None:
    assert _gate(text).code == "operational_harm"


@pytest.mark.parametrize(
    "text",
    [
        (
            "First combine ammonium nitrate with fuel oil. Then place it in a steel "
            "pipe. Finally attach a timed ignition circuit."
        ),
        (
            "First assemble the weapon frame. Then load ammunition into the chamber. "
            "Finally connect the ignition timer."
        ),
        (
            "First assemble a malware payload. Then connect it to a loader service. "
            "Finally deploy the payload across target machines."
        ),
        (
            "For credential theft, first assemble a fake sign-in page. Then connect it "
            "to a capture service. Finally deploy it to targets."
        ),
    ],
)
def test_sequenced_harm_materials_and_procedural_variants_are_rejected(
    text: str,
) -> None:
    assert _gate(text).code == "operational_harm"


def test_credential_theft_sequence_with_bank_login_indicators_is_rejected() -> None:
    assert _gate(
        "First clone a bank login page. Then capture passwords. Finally exfiltrate "
        "the results."
    ).code == "operational_harm"


def test_sandbox_word_does_not_exempt_explosive_construction() -> None:
    assert _gate(
        "In a sandbox, first combine ammonium nitrate with fuel oil. Then pack it "
        "into a steel pipe. Finally wire the ignition circuit."
    ).code == "operational_harm"


@pytest.mark.parametrize(
    "text",
    [
        (
            "First assemble the weapon frame. Then load ammunition into the chamber. "
            "Finally connect the ignition timer for forensic analysis in an isolated "
            "sandbox."
        ),
        (
            "First assemble an explosive housing. Then place the charge inside it. "
            "Finally attach the ignition circuit for forensic analysis."
        ),
        (
            "First assemble a malware payload. Then install it on target hosts. Finally "
            "deploy it to victim servers for forensic analysis in a sandbox."
        ),
        (
            "First clone a bank login page. Then capture passwords. Finally exfiltrate "
            "the results for forensic analysis."
        ),
        (
            "First place the explosive charge in a steel pipe. Then attach the ignition "
            "circuit. Finally connect the timer for forensic analysis."
        ),
        (
            "First load ammunition into the weapon. Then attach the firing mechanism. "
            "Finally connect the ignition timer for incident response analysis."
        ),
    ],
)
def test_offensive_actions_override_adversarial_defensive_suffixes(text: str) -> None:
    assert _gate(text).code == "operational_harm"


@pytest.mark.parametrize(
    "text",
    [
        (
            "First write malware that downloads a payload. Then configure persistence. "
            "Finally execute the payload for forensic analysis."
        ),
        (
            "For credential theft, first assemble a fake sign-in page. Then connect it "
            "to a capture service. Finally deploy it to targets for forensic analysis."
        ),
    ],
)
def test_domain_specific_offensive_flows_override_forensic_suffix(text: str) -> None:
    assert _gate(text).code == "operational_harm"


def test_malware_reporting_and_approved_guide_sequence_is_allowed() -> None:
    assert _gate(
        "First write the malware incident report. Then download the approved security "
        "guide. Finally document remediation steps."
    ).accepted


def test_clause_scoped_malware_loader_flow_is_rejected() -> None:
    assert _gate(
        "First assemble a malware payload. Then connect it to a loader service. "
        "Finally execute it across target machines."
    ).code == "operational_harm"


@pytest.mark.parametrize(
    "text",
    [
        (
            "First mix the cake ingredients. Then read the explosive incident report. "
            "Finally document the safety lesson."
        ),
        (
            "First connect with customer support. Then deploy the approved password "
            "reset guide. Finally document the resolution."
        ),
        (
            "First download the malware sample into an isolated sandbox. Then execute "
            "the payload for forensic analysis. Finally document indicators and "
            "remediate the host."
        ),
    ],
)
def test_clause_scoped_benign_and_defensive_sequences_are_allowed(text: str) -> None:
    assert _gate(text).accepted


@pytest.mark.parametrize(
    "text",
    [
        "Security report: attackers sent phishing messages and stole credentials yesterday.",
        "Someone threatened to hurt a customer during an argument yesterday.",
        "An advertisement falsely promises guaranteed returns to vulnerable customers.",
        "Please describe why malware persistence creates serious security risks.",
    ],
)
def test_nonprocedural_harm_mentions_and_security_reporting_are_allowed(text: str) -> None:
    assert _gate(text).accepted


@pytest.mark.parametrize(
    "text",
    [
        (
            "News reporting says ammonium nitrate, fuel oil, a steel pipe, and a timed "
            "ignition circuit were recovered by investigators."
        ),
        (
            "Safety teams removed fuel oil and ammonium nitrate from the steel pipe "
            "before secure disposal."
        ),
    ],
)
def test_harm_materials_in_reporting_and_remediation_prose_are_allowed(text: str) -> None:
    assert _gate(text).accepted


def test_sequenced_kitchen_timer_instructions_are_not_operational_harm() -> None:
    assert _gate(
        "First assemble the ingredients. Then place dinner in the oven. Finally connect "
        "the kitchen timer."
    ).accepted


@pytest.mark.parametrize(
    "text",
    [
        (
            "First download the malware sample into an isolated sandbox. Then execute "
            "it for forensic analysis. Finally document indicators and remediate the host."
        ),
        (
            "First download the packaged malware sample into an isolated sandbox. Then "
            "execute it for forensic analysis. Finally document indicators and remediate "
            "the host."
        ),
        (
            "First the bomb squad secured the explosive scene. Then officers neutralized "
            "the device. Finally they documented and preserved evidence for law "
            "enforcement disposal."
        ),
    ],
)
def test_explicit_defensive_harm_sequences_are_allowed(text: str) -> None:
    assert _gate(text).accepted


@pytest.mark.parametrize(
    ("length_bin", "count"),
    [
        ("headline", 1),
        ("headline", 20),
        ("short", 5),
        ("short", 60),
        ("medium", 30),
        ("medium", 160),
        ("long", 100),
        ("long", 400),
    ],
)
def test_length_bins_accept_inclusive_boundaries(length_bin: str, count: int) -> None:
    assert _gate("word " * count, length_bin=length_bin).accepted


@pytest.mark.parametrize(
    ("length_bin", "count"),
    [
        ("headline", 21),
        ("short", 4),
        ("short", 61),
        ("medium", 29),
        ("medium", 161),
        ("long", 99),
        ("long", 401),
    ],
)
def test_length_bins_reject_just_outside_boundaries(length_bin: str, count: int) -> None:
    assert _gate("word " * count, length_bin=length_bin).code == "length_out_of_bin"


def test_zero_word_headline_is_rejected_by_content_contract() -> None:
    from risk_agent.singguard_query_generation import GeneratedContent

    with pytest.raises(ValidationError):
        GeneratedContent(blueprint_id="bp-1", query="   ")


def test_length_gate_counts_query_and_response_together() -> None:
    assert _gate(
        "one two three",
        response="four five",
        conversation_shape="query_response",
        length_bin="short",
    ).accepted
    assert _gate(
        "one two",
        response="three four",
        conversation_shape="query_response",
        length_bin="short",
    ).code == "length_out_of_bin"


def test_length_constants_are_versioned_and_immutable() -> None:
    from risk_agent.singguard_query_generation import LENGTH_BOUNDS, LENGTH_BOUNDS_VERSION

    assert LENGTH_BOUNDS_VERSION == "singguard-length-bounds-v1"
    assert dict(LENGTH_BOUNDS) == {
        "headline": (1, 20),
        "short": (5, 60),
        "medium": (30, 160),
        "long": (100, 400),
    }
    with pytest.raises(TypeError):
        LENGTH_BOUNDS["short"] = (1, 2)  # type: ignore[index]


def _content(query: str, response: str | None = None, blueprint_id: str = "bp-1"):
    from risk_agent.singguard_query_generation import GeneratedContent

    return GeneratedContent(blueprint_id=blueprint_id, query=query, response=response)


@pytest.mark.parametrize("family_id", ["", "  ", 123, None])
def test_candidate_index_rejects_invalid_family_ids(family_id: object) -> None:
    from risk_agent.singguard_query_generation import CandidateIndex

    index = CandidateIndex()
    with pytest.raises((TypeError, ValueError)):
        index.add(family_id, _content("Please review this ordinary order today"))  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        index.duplicate_code(  # type: ignore[arg-type]
            family_id, _content("Please review this ordinary order today")
        )


def test_candidate_index_requires_generated_content_without_coercion() -> None:
    from risk_agent.singguard_query_generation import CandidateIndex

    index = CandidateIndex()
    with pytest.raises(TypeError):
        index.add("family-1", {"query": "not a model"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        index.duplicate_code("family-1", "not a model")  # type: ignore[arg-type]


def test_exact_candidate_duplicate_is_global_including_same_family() -> None:
    from risk_agent.singguard_query_generation import CandidateIndex

    index = CandidateIndex()
    original = _content("Please review this ordinary order before shipping today")
    index.add("family-1", original)
    assert index.duplicate_code("family-1", original) == "exact_duplicate"
    assert index.duplicate_code("family-2", original) == "exact_duplicate"


def test_nfkc_case_and_whitespace_normalization_is_exact_and_deterministic() -> None:
    from risk_agent.singguard_query_generation import CandidateIndex

    index = CandidateIndex()
    index.add("family-1", _content("Ｐｌｅａｓｅ  REVIEW\nthis ordinary order today"))
    variant = _content("please review this\tordinary ORDER today")
    assert index.duplicate_code("family-2", variant) == "exact_duplicate"
    assert index.duplicate_code("family-2", variant) == "exact_duplicate"


def test_punctuation_only_variant_is_cross_family_near_duplicate() -> None:
    from risk_agent.singguard_query_generation import CandidateIndex

    index = CandidateIndex()
    index.add(
        "family-1", _content("Please review this ordinary order before shipping today")
    )
    assert index.duplicate_code(
        "family-2", _content("Please, review this ordinary order before shipping today!")
    ) == "near_duplicate"


def test_near_duplicate_excludes_same_family_but_not_other_families() -> None:
    from risk_agent.singguard_query_generation import CandidateIndex

    index = CandidateIndex()
    index.add(
        "family-1", _content("Please review this ordinary order before shipping today")
    )
    variant = _content("Please, review this ordinary order before shipping today!")
    assert index.duplicate_code("family-1", variant) is None
    assert index.duplicate_code("family-2", variant) == "near_duplicate"


def test_query_response_normalization_uses_both_fields_and_stable_separation() -> None:
    from risk_agent.singguard_query_generation import CandidateIndex

    index = CandidateIndex()
    index.add(
        "family-1",
        _content(
            "Please review this ordinary order",
            "The request appears valid and complete",
        ),
    )
    assert index.duplicate_code(
        "family-2",
        _content(
            " PLEASE  REVIEW this ordinary order ",
            "the request appears valid\nand complete",
        ),
    ) == "exact_duplicate"
    assert index.duplicate_code(
        "family-1", _content("Please review this ordinary order")
    ) is None


def test_gate_uses_exact_then_source_then_cross_family_near_order() -> None:
    from risk_agent.singguard_query_generation import CandidateIndex

    index = CandidateIndex()
    indexed = _content("Please review this ordinary order before shipping today")
    index.add("other-family", indexed)
    assert _gate(indexed.query, index=index).code == "exact_duplicate"

    near = "Please, review this ordinary order before shipping today!"
    assert _gate(near, source_text=near, index=index).code == "source_too_similar"
    assert _gate(near, index=index).code == "near_duplicate"


def test_exact_short_source_copy_is_rejected_below_five_tokens() -> None:
    assert _gate("Tiny source", source_text="  TINY\nsource ", length_bin="headline").code == (
        "source_too_similar"
    )


def test_source_five_gram_threshold_is_inclusive_and_below_is_allowed() -> None:
    candidate = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    at_threshold = "alpha bravo charlie delta echo foxtrot golf hotel xray yankee"
    below_threshold = "alpha bravo charlie delta echo foxtrot golf xray yankee zulu"
    assert _gate(candidate, source_text=at_threshold).code == "source_too_similar"
    assert _gate(candidate, source_text=below_threshold).accepted


@pytest.mark.parametrize(
    ("query", "response"),
    [
        (
            "alpha bravo charlie delta echo",
            "novel response words describe a completely separate ordinary request today",
        ),
        (
            "novel query words describe a completely separate ordinary request today",
            "alpha bravo charlie delta echo",
        ),
    ],
)
def test_query_response_rejects_exact_source_copy_in_either_component(
    query: str, response: str
) -> None:
    assert _gate(
        query,
        response=response,
        source_text="alpha bravo charlie delta echo",
        conversation_shape="query_response",
    ).code == "source_too_similar"


@pytest.mark.parametrize("similar_component", ["query", "response"])
def test_query_response_checks_source_similarity_per_component(
    similar_component: str,
) -> None:
    similar = "alpha bravo charlie delta echo foxtrot golf hotel xray yankee"
    novel = "one two three four five six seven eight nine ten eleven twelve"
    query, response = (similar, novel) if similar_component == "query" else (novel, similar)
    assert _gate(
        query,
        response=response,
        source_text="alpha bravo charlie delta echo foxtrot golf hotel india juliet",
        conversation_shape="query_response",
    ).code == "source_too_similar"


def test_query_response_allows_when_each_source_comparison_is_below_threshold() -> None:
    assert _gate(
        "alpha bravo charlie delta echo foxtrot golf xray yankee zulu",
        response="one two three four five six seven eight nine ten eleven twelve",
        source_text="alpha bravo charlie delta echo foxtrot golf hotel india juliet",
        conversation_shape="query_response",
    ).accepted


@pytest.mark.parametrize("source_text", ["", "  ", 123, "x" * 5001])
def test_source_text_is_strict_nonblank_and_bounded(source_text: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _gate(
            "Please review this ordinary request today",
            source_text=source_text,  # type: ignore[arg-type]
        )


def test_length_precedes_candidate_and_source_similarity() -> None:
    from risk_agent.singguard_query_generation import CandidateIndex

    index = CandidateIndex()
    content = _content("one two three four")
    index.add("other-family", content)
    assert _gate(content.query, source_text=content.query, index=index).code == (
        "length_out_of_bin"
    )


def test_gate_does_not_auto_add_and_rejected_candidates_do_not_mutate_index() -> None:
    from risk_agent.singguard_query_generation import CandidateIndex

    index = CandidateIndex()
    query = "Please review this ordinary product request today"
    assert _gate(query, index=index).accepted
    assert _gate(query, index=index).accepted
    assert _gate("[user] invalid wrapper content today", index=index).code == (
        "literal_role_wrapper"
    )
    assert _gate(query, index=index).accepted

    index.add("family-1", _content(query))
    assert _gate(query, index=index).code == "exact_duplicate"


def test_gate_results_are_deterministic_across_repeated_calls() -> None:
    from risk_agent.singguard_query_generation import CandidateIndex

    index = CandidateIndex()
    query = "Please review this ordinary product request today"
    results = [_gate(query, source_text="Unrelated governed guidance text here", index=index) for _ in range(5)]
    assert all(result == results[0] for result in results)
