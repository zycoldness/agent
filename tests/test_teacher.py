"""Tests for bounded, injectable structured teacher providers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from risk_agent.teacher import CallableTeacher, GeminiTeacher, TeacherReply, TeacherUsage


def test_callable_teacher_accepts_a_structured_fake_without_external_dependencies():
    seen = []

    def fake(request):
        seen.append(request)
        return {"tool": "final_decision", "arguments": {"label": "safe", "confidence": 1.0}}

    reply = CallableTeacher(fake, model="fake-model").generate({"phase": "first"})

    assert reply.payload["tool"] == "final_decision"
    assert reply.usage == TeacherUsage(provider="callable", model="fake-model", request_count=1)
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

    with pytest.raises(RuntimeError, match="teacher request failed after 2 attempts"):
        teacher.generate({"phase": "first"})

    assert secret not in caplog.text


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
    }

