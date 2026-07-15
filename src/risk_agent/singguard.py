"""Minimal SingGuard-compatible data contracts and SFT rendering."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from risk_agent.contracts import PolicyRule
from risk_agent.singguard_prompts import render_guard_prompt


ThinkingType = Literal["fast", "slow"]


class ActivePolicy(BaseModel):
    """One runtime policy set containing one or more active rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_id: str = Field(min_length=1)
    rules: tuple[PolicyRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_rules(self) -> "ActivePolicy":
        if not self.policy_id.strip():
            raise ValueError("active policy ID must be non-blank")
        if any(
            not value.strip()
            for rule in self.rules
            for value in (rule.rule_id, rule.title, rule.text, *rule.exceptions)
        ):
            raise ValueError("active policy rule fields must be non-blank")
        rule_ids = [rule.rule_id for rule in self.rules]
        titles = [rule.title for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("active policy contains duplicate rule IDs")
        if len(titles) != len(set(titles)):
            raise ValueError("active policy contains duplicate rule titles")
        return self


class ModerationSample(BaseModel):
    """One conversation to execute under a referenced active policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    thinking_type: ThinkingType
    query: str = Field(min_length=1)
    response: str | None = None
    tool_names: tuple[str, ...] = ()
    tool_policy: Literal["auto", "required"] = "auto"
    expected_label: Literal["safe", "unsafe"] | None = None
    expected_answers: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_hidden_expectation(self) -> "ModerationSample":
        if not self.sample_id.strip() or not self.policy_id.strip():
            raise ValueError("sample and policy IDs must be non-blank")
        if not self.query.strip():
            raise ValueError("query must be non-blank")
        if any(not name.strip() for name in self.tool_names):
            raise ValueError("tool names must be non-blank")
        if len(self.tool_names) != len(set(self.tool_names)):
            raise ValueError("tool names must be unique")
        if any(not answer.strip() for answer in self.expected_answers):
            raise ValueError("expected answers must be non-blank")
        if self.tool_policy == "required" and not self.tool_names:
            raise ValueError("required tool policy needs at least one tool")
        if len(self.expected_answers) != len(set(self.expected_answers)):
            raise ValueError("expected answers must be unique")
        if self.expected_label is None and self.expected_answers:
            raise ValueError("expected answers require an expected label")
        if self.expected_label == "safe" and self.expected_answers:
            raise ValueError("safe expectations must not name a rule")
        if self.expected_label == "unsafe" and not self.expected_answers:
            raise ValueError("unsafe expectations must name an active rule")
        return self


class Message(BaseModel):
    """One portable ms-swift conversation message."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant", "tool_call", "tool_response"]
    content: str = Field(min_length=1)


class ParsedCompletion(BaseModel):
    """Validated metadata while preserving the teacher completion verbatim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: Literal["safe", "unsafe"]
    answers: tuple[str, ...] = Field(min_length=1)
    original: str = Field(min_length=1)


_FAST_COMPLETION_RE = re.compile(
    r"\A(safe|unsafe)\n{1,2}<answer>([^<]+)</answer>\Z",
    re.DOTALL,
)
_SLOW_COMPLETION_RE = re.compile(
    r"\A(safe|unsafe)\n{1,2}<reasoning>\n(.+)\n</reasoning>\n"
    r"<answer>([^<]+)</answer>\Z",
    re.DOTALL,
)
_VERDICTS = ("HIT", "NOT HIT", "NOT APPLICABLE", "OBSERVED BUT NOT UNSAFE")


def _answer_lines(raw: str) -> tuple[str, ...]:
    answers = tuple(line.strip() for line in raw.splitlines() if line.strip())
    if not answers or len(answers) != len(set(answers)):
        raise ValueError("completion answer contains missing or duplicate titles")
    return answers


def _validate_answers(
    label: str,
    answers: tuple[str, ...],
    active_titles: tuple[str, ...],
) -> None:
    if label == "safe":
        if answers != ("Safe",):
            raise ValueError("safe completion must answer Safe")
        return
    if "Safe" in answers or any(answer not in active_titles for answer in answers):
        raise ValueError("unsafe completion answer must belong to the active policy")


def validate_completion(
    completion: str,
    *,
    thinking_type: ThinkingType,
    active_titles: tuple[str, ...],
) -> ParsedCompletion:
    """Validate SingGuard grammar without changing Gemini's output."""

    if not active_titles or len(active_titles) != len(set(active_titles)):
        raise ValueError("active policy titles must be non-empty and unique")
    pattern = _FAST_COMPLETION_RE if thinking_type == "fast" else _SLOW_COMPLETION_RE
    match = pattern.fullmatch(completion)
    if match is None:
        raise ValueError(f"completion does not match {thinking_type} output grammar")
    label = match.group(1)
    reasoning = None if thinking_type == "fast" else match.group(2)
    answer_group = 2 if thinking_type == "fast" else 3
    answers = _answer_lines(match.group(answer_group))
    if reasoning is None:
        _validate_answers(label, answers, active_titles)

    if reasoning is not None:
        summary_marker = "[Step 1] Content Summary\n"
        checks_marker = "\n\n[Step 2] Check Risk Categories\n"
        judgment_marker = "\n\n[Step 3] Final Judgment\n"
        if not reasoning.startswith(summary_marker):
            raise ValueError("slow completion is missing the content summary")
        try:
            _summary, remainder = reasoning[len(summary_marker) :].split(
                checks_marker,
                1,
            )
            checks_text, judgment = remainder.split(judgment_marker, 1)
        except ValueError:
            raise ValueError("slow completion is missing required reasoning steps") from None
        if not _summary.strip() or not judgment.strip():
            raise ValueError("slow completion reasoning steps must not be empty")
        raw_check_lines = checks_text.splitlines()
        if any(line and not line.startswith("- ") for line in raw_check_lines):
            raise ValueError("slow completion Step 2 must contain only active-rule lines")
        check_lines = tuple(line for line in raw_check_lines if line)
        observed_titles: list[str] = []
        hit_titles: set[str] = set()
        for line in check_lines:
            matched_title = next(
                (
                    title
                    for title in active_titles
                    if line.startswith(f"- {title}: ")
                ),
                None,
            )
            if matched_title is None:
                raise ValueError("slow completion checks an inactive policy rule")
            suffix = line[len(f"- {matched_title}: ") :]
            if suffix == "NOT APPLICABLE.":
                verdict = "NOT APPLICABLE"
            else:
                verdict = next(
                    (item for item in _VERDICTS if suffix.startswith(f"{item}. ")),
                    None,
                )
            if verdict is None or (
                verdict != "NOT APPLICABLE"
                and not suffix[len(verdict) + 2 :].strip()
            ):
                raise ValueError("slow completion rule check has invalid evidence")
            observed_titles.append(matched_title)
            if verdict == "HIT":
                hit_titles.add(matched_title)
        if tuple(observed_titles) != active_titles:
            raise ValueError("slow completion must check rules in active policy order")
        if label == "safe" and hit_titles:
            raise ValueError("safe slow completion must not contain a hit")
        _validate_answers(label, answers, active_titles)
        expected_hit_answers = tuple(
            title for title in active_titles if title in hit_titles
        )
        if label == "unsafe" and answers != expected_hit_answers:
            raise ValueError(
                "unsafe slow completion answers must match every HIT rule in policy order"
            )

    return ParsedCompletion(
        label=label,
        answers=answers,
        original=completion,
    )


def render_training_row(
    initial_messages: tuple[Message, Message],
    completion: str,
    *,
    tools_json: str | None = None,
    trajectory: tuple[Message, ...] = (),
) -> dict[str, object]:
    """Render Gemini's validated completion and optional real tool trajectory."""

    messages = [
        message.model_dump(mode="json")
        for message in (*initial_messages, *trajectory)
    ]
    messages.append({"role": "assistant", "content": completion})
    row: dict[str, object] = {"messages": messages}
    if tools_json is not None:
        row["tools"] = tools_json
    return row


def build_initial_messages(
    policy: ActivePolicy,
    sample: ModerationSample,
) -> tuple[Message, Message]:
    """Render the exact active-policy prompt and target conversation."""

    if sample.policy_id != policy.policy_id:
        raise ValueError("sample policy_id does not match the active policy")
    lines = [f"[user]: {sample.query}"]
    if sample.response is not None:
        lines.append(f"[assistant]: {sample.response}")
    return (
        Message(
            role="system",
            content=render_guard_prompt(
                policy.rules,
                thinking_type=sample.thinking_type,
                required_tools=(
                    sample.tool_names if sample.tool_policy == "required" else ()
                ),
            ),
        ),
        Message(role="user", content="\n".join(lines)),
    )
