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
    budget.reserve_request()

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
            )
        )
