"""Tests for bounded, injectable structured teacher providers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from risk_agent.teacher import (
    CallableTeacher,
    GeminiTeacher,
    TeacherReply,
    TeacherRequestError,
    TeacherBudget,
    TeacherBudgetExceeded,
    TeacherUsage,
)


def test_callable_teacher_accepts_a_structured_fake_without_external_dependencies():
    seen = []

    def fake(request):
        seen.append(request)
        return {"tool": "final_decision", "arguments": {"label": "safe", "confidence": 1.0}}

    reply = CallableTeacher(fake, model="fake-model").generate({"phase": "first"})

    assert reply.payload["tool"] == "final_decision"
    assert reply.usage == TeacherUsage(
        provider="callable",
        model="fake-model",
        request_count=1,
        accounting_complete=False,
    )
    assert seen == [{"phase": "first"}]


def test_callable_teacher_rejects_non_object_json():
    teacher = CallableTeacher(lambda _request: "[]")

    with pytest.raises(ValueError, match="JSON object"):
        teacher.generate({"phase": "first"})


def test_gemini_teacher_retries_with_bounded_backoff_and_reports_usage():
    attempts = []
    delays = []

    class Models:
        def generate_content(self, **kwargs):
            attempts.append(kwargs)
            if len(attempts) < 3:
                raise RuntimeError("transient")
            return SimpleNamespace(
                text='{"tool":"final_decision","arguments":{"label":"safe","confidence":1.0}}',
                usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=4),
            )

    client = SimpleNamespace(models=Models())
    teacher = GeminiTeacher(
        model="gemini-test",
        client_factory=lambda: client,
        max_attempts=3,
        initial_backoff_seconds=0.25,
        sleep=delays.append,
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )

    reply = teacher.generate({"phase": "first", "oracle": {"label": "safe"}})

    assert len(attempts) == 3
    assert delays == [0.25, 0.5]
    assert attempts[-1]["config"]["response_mime_type"] == "application/json"
    assert attempts[-1]["config"]["response_schema"]["required"] == ["tool", "arguments"]
    assert reply.usage.request_count == 3
    assert reply.usage.input_tokens == 10
    assert reply.usage.output_tokens == 4
    assert reply.usage.estimated_cost_usd == pytest.approx(18e-6)
    assert reply.usage.accounting_complete is False


def test_gemini_teacher_stops_after_configured_attempts_without_logging_secret(caplog):
    secret = "never-log-this-api-key"

    class Models:
        def generate_content(self, **_kwargs):
            raise RuntimeError(secret)

    teacher = GeminiTeacher(
        model="gemini-test",
        client_factory=lambda: SimpleNamespace(models=Models()),
        max_attempts=2,
        initial_backoff_seconds=0,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(TeacherRequestError, match="teacher request failed after 2 attempts") as caught:
        teacher.generate({"phase": "first"})

    assert secret not in caplog.text
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert caught.value.usage.request_count == 2
    assert caught.value.usage.accounting_complete is False


def test_teacher_reply_metadata_never_contains_prompt_or_credentials():
    reply = TeacherReply(
        payload={"tool": "final_decision", "arguments": {"label": "safe", "confidence": 1.0}},
        usage=TeacherUsage(provider="fake", model="fake"),
    )

    assert set(reply.usage.as_dict()) == {
        "provider",
        "model",
        "request_count",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "accounting_complete",
    }


def test_malformed_response_attempt_usage_is_aggregated_before_json_parsing():
    responses = iter(
        [
            SimpleNamespace(
                text="not-json",
                usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=2),
            ),
            SimpleNamespace(
                text='{"tool":"final_decision","arguments":{"label":"safe","confidence":1.0}}',
                usage_metadata=SimpleNamespace(prompt_token_count=7, candidates_token_count=3),
            ),
        ]
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_kwargs: next(responses))
    )
    teacher = GeminiTeacher(
        model="gemini-test",
        client_factory=lambda: client,
        max_attempts=2,
        initial_backoff_seconds=0,
        sleep=lambda _seconds: None,
        input_cost_per_million=1,
        output_cost_per_million=1,
    )

    reply = teacher.generate({"phase": "first"})

    assert reply.usage.request_count == 2
    assert reply.usage.input_tokens == 12
    assert reply.usage.output_tokens == 5
    assert reply.usage.estimated_cost_usd == pytest.approx(17e-6)
    assert reply.usage.accounting_complete is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("initial_backoff_seconds", -1.0),
        ("initial_backoff_seconds", float("nan")),
        ("max_backoff_seconds", float("inf")),
        ("request_timeout_seconds", 0.0),
        ("request_timeout_seconds", float("nan")),
        ("input_cost_per_million", -0.1),
        ("output_cost_per_million", float("inf")),
    ],
)
def test_gemini_teacher_rejects_nonfinite_or_out_of_range_numeric_settings(name, value):
    kwargs = {name: value}

    with pytest.raises(ValueError):
        GeminiTeacher(model="gemini-test", client_factory=lambda: object(), **kwargs)


def test_default_gemini_client_receives_finite_application_timeout(monkeypatch):
    captured = {}

    class Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_genai = SimpleNamespace(Client=Client)
    monkeypatch.setitem(__import__("sys").modules, "google", SimpleNamespace(genai=fake_genai))

    teacher = GeminiTeacher(model="gemini-test", request_timeout_seconds=12.5)
    teacher._new_client()

    assert captured["http_options"]["timeout"] == 12_500


def test_callable_provider_failure_does_not_retain_raw_exception_context():
    secret = "private-provider-secret"
    teacher = CallableTeacher(lambda _request: (_ for _ in ()).throw(RuntimeError(secret)))

    with pytest.raises(TeacherRequestError) as caught:
        teacher.generate({"phase": "first"})

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert secret not in repr(caught.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request_count": -1},
        {"input_tokens": -1},
        {"output_tokens": -1},
        {"estimated_cost_usd": float("nan")},
    ],
)
def test_teacher_usage_rejects_invalid_accounting_values(kwargs):
    with pytest.raises(ValueError):
        TeacherUsage(provider="fake", model="fake", **kwargs)


def test_cost_budget_fails_closed_when_provider_usage_has_no_cost():
    budget = TeacherBudget(max_estimated_cost_usd=1.0)
    reservation = budget.reserve_request(worst_case_cost_usd=0.5)

    with pytest.raises(TeacherBudgetExceeded, match="cost_accounting_incomplete"):
        budget.record_attempt(
            TeacherUsage(
                provider="fake",
                model="fake",
                request_count=1,
                input_tokens=10,
                output_tokens=2,
                estimated_cost_usd=None,
                accounting_complete=True,
            ),
            reserved_cost_usd=reservation,
        )


def test_client_initialization_failure_is_sanitized_and_counts_zero_requests():
    secret = "client-init-secret"
    teacher = GeminiTeacher(
        model="gemini-test",
        client_factory=lambda: (_ for _ in ()).throw(RuntimeError(secret)),
    )

    with pytest.raises(TeacherRequestError) as caught:
        teacher.generate({"phase": "first"})

    assert caught.value.usage.request_count == 0
    assert caught.value.usage.accounting_complete is False
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert secret not in str(caught.value)


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [
        (True, 1),
        ("1", 1),
        (-1, 1),
        (float("nan"), 1),
        (1_000_000_001, 1),
        (1, False),
    ],
)
def test_malformed_provider_usage_records_one_incomplete_attempt(input_tokens, output_tokens):
    response = SimpleNamespace(
        text='{"tool":"final_decision","arguments":{"label":"safe","confidence":1.0}}',
        usage_metadata=SimpleNamespace(
            prompt_token_count=input_tokens,
            candidates_token_count=output_tokens,
        ),
    )
    calls = 0

    def generate_content(**_kwargs):
        nonlocal calls
        calls += 1
        return response

    budget = TeacherBudget(max_requests=2)
    teacher = GeminiTeacher(
        model="gemini-test",
        client_factory=lambda: SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        ),
        max_attempts=1,
        budget=budget,
    )

    reply = teacher.generate({"phase": "first"})

    assert calls == 1
    assert reply.usage.request_count == 1
    assert reply.usage.input_tokens == 0
    assert reply.usage.output_tokens == 0
    assert reply.usage.accounting_complete is False
    assert budget.as_dict()["request_count"] == 1
    assert budget.as_dict()["accounting_complete"] is False


def test_cost_cap_fails_closed_on_malformed_usage_without_zero_cost_success():
    calls = 0

    def generate_content(**_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            text='{"tool":"final_decision","arguments":{"label":"safe","confidence":1.0}}',
            usage_metadata=SimpleNamespace(prompt_token_count=True, candidates_token_count=1),
        )

    budget = TeacherBudget(max_estimated_cost_usd=10.0)
    teacher = GeminiTeacher(
        model="gemini-test",
        client_factory=lambda: SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        ),
        input_cost_per_million=1.0,
        output_cost_per_million=1.0,
        max_output_tokens=10,
        budget=budget,
    )

    with pytest.raises(TeacherBudgetExceeded, match="cost_accounting_incomplete"):
        teacher.generate({"phase": "first"})

    assert calls == 1
    assert budget.as_dict()["request_count"] == 1
    assert budget.as_dict()["accounting_complete"] is False


@pytest.mark.parametrize(
    ("input_price", "output_price"),
    [(None, None), (0.0, 1.0), (1.0, 0.0)],
)
def test_cost_cap_rejects_unknown_or_zero_pricing_before_provider_call(input_price, output_price):
    calls = 0

    def generate_content(**_kwargs):
        nonlocal calls
        calls += 1

    budget = TeacherBudget(max_estimated_cost_usd=0.1)
    teacher = GeminiTeacher(
        model="gemini-test",
        client_factory=lambda: SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        ),
        input_cost_per_million=input_price,
        output_cost_per_million=output_price,
        budget=budget,
    )

    with pytest.raises(TeacherBudgetExceeded, match="cost_accounting_incomplete"):
        teacher.generate({"phase": "first"})

    assert calls == 0
    assert budget.as_dict()["request_count"] == 0


def test_hard_cost_cap_prevents_request_when_worst_case_reservation_is_too_large():
    calls = 0

    def generate_content(**_kwargs):
        nonlocal calls
        calls += 1

    budget = TeacherBudget(max_estimated_cost_usd=0.1)
    teacher = GeminiTeacher(
        model="gemini-test",
        client_factory=lambda: SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        ),
        input_cost_per_million=1_000_000.0,
        output_cost_per_million=1_000_000.0,
        max_output_tokens=1,
        budget=budget,
    )

    with pytest.raises(TeacherBudgetExceeded, match="max_estimated_cost"):
        teacher.generate({"phase": "first"})

    assert calls == 0
    assert budget.as_dict()["request_count"] == 0


def test_retry_reservation_cannot_overshoot_cost_cap():
    calls = 0
    seen_configs = []

    def generate_content(**kwargs):
        nonlocal calls
        calls += 1
        seen_configs.append(kwargs["config"])
        return SimpleNamespace(
            text="not-json",
            usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
        )

    budget = TeacherBudget(max_estimated_cost_usd=0.1)
    teacher = GeminiTeacher(
        model="gemini-test",
        client_factory=lambda: SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        ),
        max_attempts=3,
        initial_backoff_seconds=0,
        sleep=lambda _seconds: None,
        input_cost_per_million=0.000001,
        output_cost_per_million=60_000.0,
        max_output_tokens=1,
        budget=budget,
    )

    with pytest.raises(TeacherBudgetExceeded, match="max_estimated_cost"):
        teacher.generate({"phase": "first"})

    assert calls == 1
    assert seen_configs[0]["max_output_tokens"] == 1
    assert budget.as_dict()["request_count"] == 1
    assert budget.as_dict()["estimated_cost_usd"] == pytest.approx(0.06)


def test_usage_metadata_accessor_failure_is_treated_as_incomplete_without_raw_error():
    secret = "usage-property-secret"

    class Response:
        text = '{"tool":"final_decision","arguments":{"label":"safe","confidence":1.0}}'

        @property
        def usage_metadata(self):
            raise RuntimeError(secret)

    teacher = GeminiTeacher(
        model="gemini-test",
        client_factory=lambda: SimpleNamespace(
            models=SimpleNamespace(generate_content=lambda **_kwargs: Response())
        ),
        max_attempts=1,
    )

    reply = teacher.generate({"phase": "first"})

    assert reply.usage.request_count == 1
    assert reply.usage.accounting_complete is False
    assert secret not in repr(reply)
