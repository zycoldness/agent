"""Structured, bounded teacher providers for offline SFT generation.

Provider prompts may contain privileged supervision. Callers must never persist a
request or raw provider error; only validated payloads and aggregate accounting may
leave this module.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


def _finite_nonnegative(value: float, name: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (maximum is not None and result > maximum):
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


@dataclass(frozen=True)
class TeacherUsage:
    """Non-sensitive aggregate request and cost accounting metadata."""

    provider: str
    model: str
    request_count: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None
    accounting_complete: bool = False

    def __post_init__(self) -> None:
        for name in ("request_count", "input_tokens", "output_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.estimated_cost_usd is not None:
            _finite_nonnegative(self.estimated_cost_usd, "estimated_cost_usd")

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "request_count": self.request_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "accounting_complete": self.accounting_complete,
        }


@dataclass(frozen=True)
class TeacherReply:
    """One structured candidate and its non-sensitive usage accounting."""

    payload: dict[str, Any]
    usage: TeacherUsage


class TeacherRequestError(RuntimeError):
    """Sanitized provider failure retaining accounting but no raw exception."""

    def __init__(self, message: str, usage: TeacherUsage) -> None:
        super().__init__(message)
        self.usage = usage


class TeacherBudgetExceeded(RuntimeError):
    """Raised before a known batch budget would be exceeded."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Teacher(Protocol):
    def generate(self, request: Mapping[str, Any]) -> TeacherReply:
        """Return one JSON object without persisting the privileged request."""


class _UsageAccumulator:
    def __init__(
        self,
        provider: str,
        model: str,
        input_price: float | None,
        output_price: float | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.input_price = input_price
        self.output_price = output_price
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.accounting_complete = True

    def add(
        self,
        input_tokens: object,
        output_tokens: object,
        *,
        explicit_cost: float | None = None,
    ) -> TeacherUsage:
        self.requests += 1
        valid_usage = (
            isinstance(input_tokens, int)
            and not isinstance(input_tokens, bool)
            and input_tokens >= 0
            and isinstance(output_tokens, int)
            and not isinstance(output_tokens, bool)
            and output_tokens >= 0
        )
        if valid_usage:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
        else:
            self.accounting_complete = False
        if explicit_cost is not None:
            self.cost += explicit_cost
        elif valid_usage and self.input_price is not None and self.output_price is not None:
            self.cost += (
                input_tokens * self.input_price + output_tokens * self.output_price
            ) / 1_000_000
        return self.snapshot()

    def snapshot(self) -> TeacherUsage:
        has_cost = self.input_price is not None and self.output_price is not None
        return TeacherUsage(
            provider=self.provider,
            model=self.model,
            request_count=self.requests,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            estimated_cost_usd=self.cost if has_cost else None,
            accounting_complete=self.accounting_complete,
        )


class TeacherBudget:
    """Shared attempt-level ledger used to enforce CLI batch budgets."""

    def __init__(
        self,
        *,
        max_requests: int | None = None,
        max_estimated_cost_usd: float | None = None,
    ) -> None:
        if max_requests is not None and (
            isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 0
        ):
            raise ValueError("max_requests must be a non-negative integer")
        if max_estimated_cost_usd is not None:
            max_estimated_cost_usd = _finite_nonnegative(
                max_estimated_cost_usd, "max_estimated_cost_usd"
            )
        self.max_requests = max_requests
        self.max_estimated_cost_usd = max_estimated_cost_usd
        self.request_count = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.estimated_cost_usd = 0.0
        self.reserved_cost_usd = 0.0
        self.accounting_complete = True

    def reserve_request(self, *, worst_case_cost_usd: float | None = None) -> float:
        if self.max_requests is not None and self.request_count >= self.max_requests:
            raise TeacherBudgetExceeded("max_requests")
        reservation = 0.0
        if self.max_estimated_cost_usd is not None:
            if not self.accounting_complete:
                raise TeacherBudgetExceeded("cost_accounting_incomplete")
            if worst_case_cost_usd is None:
                raise TeacherBudgetExceeded("cost_accounting_incomplete")
            reservation = _finite_nonnegative(worst_case_cost_usd, "worst_case_cost_usd")
            if reservation == 0:
                raise TeacherBudgetExceeded("cost_accounting_incomplete")
            if (
                self.estimated_cost_usd + self.reserved_cost_usd + reservation
                > self.max_estimated_cost_usd
            ):
                raise TeacherBudgetExceeded("max_estimated_cost")
        self.request_count += 1
        self.reserved_cost_usd += reservation
        return reservation

    def record_attempt(
        self,
        usage: TeacherUsage,
        *,
        reserved_cost_usd: float = 0.0,
    ) -> None:
        reserved_cost_usd = _finite_nonnegative(reserved_cost_usd, "reserved_cost_usd")
        self.reserved_cost_usd = max(0.0, self.reserved_cost_usd - reserved_cost_usd)
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.accounting_complete = self.accounting_complete and usage.accounting_complete
        if usage.estimated_cost_usd is None:
            if self.max_estimated_cost_usd is not None:
                self.accounting_complete = False
                raise TeacherBudgetExceeded("cost_accounting_incomplete")
        else:
            self.estimated_cost_usd += usage.estimated_cost_usd
            if (
                self.max_estimated_cost_usd is not None
                and usage.estimated_cost_usd > reserved_cost_usd
            ):
                self.accounting_complete = False
                raise TeacherBudgetExceeded("cost_reservation_exceeded")
        if (
            self.max_estimated_cost_usd is not None
            and self.estimated_cost_usd > self.max_estimated_cost_usd
        ):
            raise TeacherBudgetExceeded("max_estimated_cost")

    def as_dict(self) -> dict[str, object]:
        return {
            "request_count": self.request_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "reserved_cost_usd": self.reserved_cost_usd,
            "accounting_complete": self.accounting_complete,
        }


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
    """Adapt one injected request callable for deterministic tests/private providers."""

    def __init__(
        self,
        call: Callable[[Mapping[str, Any]], object],
        *,
        provider: str = "callable",
        model: str = "injected",
        budget: TeacherBudget | None = None,
        worst_case_cost_usd: float | None = None,
    ) -> None:
        self._call = call
        self._provider = provider
        self._model = model
        self._budget = budget
        self._worst_case_cost_usd = (
            None
            if worst_case_cost_usd is None
            else _finite_nonnegative(worst_case_cost_usd, "worst_case_cost_usd")
        )

    def generate(self, request: Mapping[str, Any]) -> TeacherReply:
        reservation = 0.0
        if self._budget is not None:
            reservation = self._budget.reserve_request(
                worst_case_cost_usd=self._worst_case_cost_usd
            )
        provider_failed = False
        try:
            value = self._call(request)
        except TeacherBudgetExceeded:
            raise
        except Exception:
            provider_failed = True
        if provider_failed:
            usage = TeacherUsage(provider=self._provider, model=self._model)
            if self._budget is not None:
                self._budget.record_attempt(usage, reserved_cost_usd=reservation)
            raise TeacherRequestError("teacher request failed after 1 attempt", usage) from None
        if not isinstance(value, TeacherReply):
            usage = TeacherUsage(provider=self._provider, model=self._model)
            if self._budget is not None:
                self._budget.record_attempt(usage, reserved_cost_usd=reservation)
            return TeacherReply(payload=_parse_object(value), usage=usage)
        reply = value
        if reply.usage.request_count != 1:
            raise ValueError("CallableTeacher replies must account for exactly one request")
        if self._budget is not None:
            self._budget.record_attempt(reply.usage, reserved_cost_usd=reservation)
        return reply


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
_MAX_USAGE_TOKENS = 1_000_000_000
_PROMPT_OVERHEAD_TOKENS = 4096
_IMAGE_MIME_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _prompt_and_image_paths(request: Mapping[str, Any]) -> tuple[str, list[Path]]:
    payload = dict(request)
    task = payload.get("task")
    image_paths: list[Path] = []
    if isinstance(task, Mapping) and task.get("images"):
        raw_images = task["images"]
        if isinstance(raw_images, (str, bytes)) or not isinstance(raw_images, (list, tuple)):
            raise ValueError("task images must be a list of local paths")
        if any(not isinstance(path, str) or not path for path in raw_images):
            raise ValueError("task images must be a list of local paths")
        image_paths = [Path(path) for path in raw_images]
        sanitized_task = dict(task)
        sanitized_task["images"] = [f"<attached_image_{index}>" for index in range(len(image_paths))]
        payload["task"] = sanitized_task
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), image_paths


def _gemini_contents(prompt: str, image_paths: list[Path]) -> tuple[object, int]:
    if not image_paths:
        return prompt, 0
    try:
        from google.genai import types
    except ImportError:
        raise RuntimeError("Gemini image support requires the optional 'teacher' dependency") from None
    parts: list[object] = []
    total_bytes = 0
    for path in image_paths:
        mime_type = _IMAGE_MIME_TYPES.get(path.suffix.lower())
        if mime_type is None:
            raise ValueError("teacher images must be JPEG, PNG, WebP, or GIF files")
        try:
            data = path.read_bytes()
        except OSError:
            raise ValueError("teacher image must be a readable local file") from None
        total_bytes += len(data)
        parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
    return [*parts, prompt], total_bytes


class GeminiTeacher:
    """Lazily loaded Gemini JSON provider with finite timeout and retries."""

    def __init__(
        self,
        *,
        model: str,
        response_schema: Mapping[str, object] | None = None,
        temperature: float = 0.2,
        client_factory: Callable[[], object] | None = None,
        max_attempts: int = 3,
        initial_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 30.0,
        request_timeout_seconds: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
        max_output_tokens: int = 2048,
        budget: TeacherBudget | None = None,
    ) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        self.model = model
        self._response_schema = dict(response_schema or _ACTION_SCHEMA)
        self._temperature = _finite_nonnegative(temperature, "temperature", maximum=2)
        self._client_factory = client_factory
        self._max_attempts = max_attempts
        self._initial_backoff_seconds = _finite_nonnegative(
            initial_backoff_seconds, "initial_backoff_seconds", maximum=300
        )
        self._max_backoff_seconds = _finite_nonnegative(
            max_backoff_seconds, "max_backoff_seconds", maximum=300
        )
        self._request_timeout_seconds = _finite_nonnegative(
            request_timeout_seconds, "request_timeout_seconds", maximum=600
        )
        if self._request_timeout_seconds == 0:
            raise ValueError("request_timeout_seconds must be positive")
        self._sleep = sleep
        self._input_cost_per_million = (
            None
            if input_cost_per_million is None
            else _finite_nonnegative(input_cost_per_million, "input_cost_per_million")
        )
        self._output_cost_per_million = (
            None
            if output_cost_per_million is None
            else _finite_nonnegative(output_cost_per_million, "output_cost_per_million")
        )
        if (self._input_cost_per_million is None) != (self._output_cost_per_million is None):
            raise ValueError("input and output prices must be supplied together")
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or not 1 <= max_output_tokens <= 65_536
        ):
            raise ValueError("max_output_tokens must be an integer from 1 to 65536")
        self._max_output_tokens = max_output_tokens
        self._budget = budget

    def _new_client(self) -> object:
        if self._client_factory is not None:
            return self._client_factory()
        try:
            from google import genai
        except ImportError:
            raise RuntimeError(
                "Gemini support requires the optional 'teacher' dependency"
            ) from None
        use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        api_key = None if use_vertex else (
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        http_options: dict[str, object] = {
            "timeout": int(self._request_timeout_seconds * 1000)
        }
        if use_vertex:
            http_options["api_version"] = "v1"
        kwargs: dict[str, object] = {"http_options": http_options}
        if api_key:
            kwargs["api_key"] = api_key
        return genai.Client(**kwargs)

    def generate(self, request: Mapping[str, Any]) -> TeacherReply:
        prompt, image_paths = _prompt_and_image_paths(request)
        try:
            contents, image_bytes = _gemini_contents(prompt, image_paths)
        except (OSError, RuntimeError, ValueError):
            raise TeacherRequestError(
                "teacher image preparation failed",
                TeacherUsage(
                    provider="gemini",
                    model=self.model,
                    request_count=0,
                    accounting_complete=False,
                ),
            ) from None
        worst_case_cost = self._worst_case_cost(prompt, image_bytes)
        aggregate = _UsageAccumulator(
            "gemini",
            self.model,
            self._input_cost_per_million,
            self._output_cost_per_million,
        )
        client_failed = False
        try:
            client = self._new_client()
        except Exception:
            client_failed = True
        if client_failed:
            raise TeacherRequestError(
                "teacher client initialization failed",
                TeacherUsage(
                    provider="gemini",
                    model=self.model,
                    request_count=0,
                    accounting_complete=False,
                ),
            ) from None
        for attempt in range(1, self._max_attempts + 1):
            reservation = 0.0
            if self._budget is not None:
                reservation = self._budget.reserve_request(
                    worst_case_cost_usd=worst_case_cost
                )
            provider_failed = False
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config={
                        "temperature": self._temperature,
                        "candidate_count": 1,
                        "response_mime_type": "application/json",
                        "response_schema": self._response_schema,
                        "max_output_tokens": self._max_output_tokens,
                    },
                )
            except Exception:
                provider_failed = True
            if provider_failed:
                missing = TeacherUsage(provider="gemini", model=self.model)
                aggregate.add(None, None)
                if self._budget is not None:
                    self._budget.record_attempt(
                        missing,
                        reserved_cost_usd=reservation,
                    )
                if attempt < self._max_attempts:
                    self._sleep(self._retry_delay(attempt))
                continue

            attempt_usage = self._normalize_response_usage(response)
            aggregate.add(
                attempt_usage.input_tokens if attempt_usage.accounting_complete else None,
                attempt_usage.output_tokens if attempt_usage.accounting_complete else None,
            )
            if self._budget is not None:
                self._budget.record_attempt(
                    attempt_usage,
                    reserved_cost_usd=reservation,
                )
            parse_failed = False
            try:
                payload = _parse_object(response.text or "")
            except Exception:
                parse_failed = True
            if not parse_failed:
                return TeacherReply(payload=payload, usage=aggregate.snapshot())
            if attempt < self._max_attempts:
                self._sleep(self._retry_delay(attempt))
        raise TeacherRequestError(
            f"teacher request failed after {self._max_attempts} attempts",
            aggregate.snapshot(),
        ) from None

    def _normalize_usage(self, metadata: object) -> TeacherUsage:
        try:
            input_tokens = getattr(metadata, "prompt_token_count")
            output_tokens = getattr(metadata, "candidates_token_count")
        except Exception:
            input_tokens = output_tokens = None
        valid = all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= _MAX_USAGE_TOKENS
            for value in (input_tokens, output_tokens)
        )
        if not valid:
            return TeacherUsage(provider="gemini", model=self.model, accounting_complete=False)
        cost = None
        if self._input_cost_per_million is not None and self._output_cost_per_million is not None:
            cost = (
                input_tokens * self._input_cost_per_million
                + output_tokens * self._output_cost_per_million
            ) / 1_000_000
        return TeacherUsage(
            provider="gemini",
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
            accounting_complete=True,
        )

    def _normalize_response_usage(self, response: object) -> TeacherUsage:
        try:
            metadata = getattr(response, "usage_metadata", None)
        except Exception:
            metadata = None
        return self._normalize_usage(metadata)

    def _worst_case_cost(self, prompt: str, image_bytes: int = 0) -> float | None:
        if self._budget is None or self._budget.max_estimated_cost_usd is None:
            return None
        if (
            self._input_cost_per_million is None
            or self._output_cost_per_million is None
            or self._input_cost_per_million <= 0
            or self._output_cost_per_million <= 0
        ):
            raise TeacherBudgetExceeded("cost_accounting_incomplete")
        input_upper_bound = len(prompt.encode("utf-8")) + image_bytes + _PROMPT_OVERHEAD_TOKENS
        return (
            input_upper_bound * self._input_cost_per_million
            + self._max_output_tokens * self._output_cost_per_million
        ) / 1_000_000

    def _retry_delay(self, attempt: int) -> float:
        return min(
            self._max_backoff_seconds,
            self._initial_backoff_seconds * (2 ** (attempt - 1)),
        )
