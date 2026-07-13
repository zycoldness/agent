"""Structured teacher providers for offline SFT candidate generation.

Provider prompts may contain privileged supervision. Callers must never persist the
request; only the validated payload and aggregate usage metadata may leave this module.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class TeacherUsage:
    """Non-sensitive request and cost accounting metadata."""

    provider: str
    model: str
    request_count: int = 1
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "request_count": self.request_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


@dataclass(frozen=True)
class TeacherReply:
    """One structured candidate and its non-sensitive usage accounting."""

    payload: dict[str, Any]
    usage: TeacherUsage


class Teacher(Protocol):
    """Injected structured teacher interface used by the synthesis pipeline."""

    def generate(self, request: Mapping[str, Any]) -> TeacherReply:
        """Return one JSON object without persisting the privileged request."""


def _parse_object(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("teacher response must be a valid JSON object") from error
    if not isinstance(value, Mapping):
        raise ValueError("teacher response must be a JSON object")
    return dict(value)


class CallableTeacher:
    """Adapt an injected callable for deterministic tests or private providers."""

    def __init__(
        self,
        call: Callable[[Mapping[str, Any]], object],
        *,
        provider: str = "callable",
        model: str = "injected",
    ) -> None:
        self._call = call
        self._provider = provider
        self._model = model

    def generate(self, request: Mapping[str, Any]) -> TeacherReply:
        value = self._call(request)
        if isinstance(value, TeacherReply):
            return value
        return TeacherReply(
            payload=_parse_object(value),
            usage=TeacherUsage(provider=self._provider, model=self._model),
        )


_ACTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "enum": ["get_rule_detail", "search_case", "inspect_evidence", "final_decision"],
        },
        "arguments": {"type": "object"},
    },
    "required": ["tool", "arguments"],
    "additionalProperties": False,
}


class GeminiTeacher:
    """Lazily loaded Gemini JSON provider with bounded retries and accounting."""

    def __init__(
        self,
        *,
        model: str,
        client_factory: Callable[[], object] | None = None,
        max_attempts: int = 3,
        initial_backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
    ) -> None:
        if not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        if initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds must be non-negative")
        self.model = model
        self._client_factory = client_factory
        self._max_attempts = max_attempts
        self._initial_backoff_seconds = initial_backoff_seconds
        self._sleep = sleep
        self._input_cost_per_million = input_cost_per_million
        self._output_cost_per_million = output_cost_per_million

    def _new_client(self) -> object:
        if self._client_factory is not None:
            return self._client_factory()
        try:
            from google import genai
        except ImportError as error:
            raise RuntimeError(
                "Gemini support requires the optional 'teacher' dependency"
            ) from error
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return genai.Client(api_key=api_key) if api_key else genai.Client()

    def generate(self, request: Mapping[str, Any]) -> TeacherReply:
        client = self._new_client()
        prompt = json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "temperature": 0.2,
                        "candidate_count": 1,
                        "response_mime_type": "application/json",
                        "response_schema": _ACTION_SCHEMA,
                    },
                )
                payload = _parse_object(response.text or "")
                usage = getattr(response, "usage_metadata", None)
                input_tokens = getattr(usage, "prompt_token_count", None)
                output_tokens = getattr(usage, "candidates_token_count", None)
                cost = self._estimate_cost(input_tokens, output_tokens)
                return TeacherReply(
                    payload=payload,
                    usage=TeacherUsage(
                        provider="gemini",
                        model=self.model,
                        request_count=attempt,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        estimated_cost_usd=cost,
                    ),
                )
            except Exception as error:  # Provider SDK exceptions vary by release.
                last_error = error
                if attempt < self._max_attempts:
                    self._sleep(self._initial_backoff_seconds * (2 ** (attempt - 1)))
        raise RuntimeError(
            f"teacher request failed after {self._max_attempts} attempts"
        ) from last_error

    def _estimate_cost(self, input_tokens: object, output_tokens: object) -> float | None:
        if (
            not isinstance(input_tokens, int)
            or not isinstance(output_tokens, int)
            or self._input_cost_per_million is None
            or self._output_cost_per_million is None
        ):
            return None
        return (
            input_tokens * self._input_cost_per_million
            + output_tokens * self._output_cost_per_million
        ) / 1_000_000

