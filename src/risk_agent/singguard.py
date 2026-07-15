"""Minimal SingGuard-compatible data contracts and SFT rendering."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from risk_agent.contracts import PolicyRule
from risk_agent.singguard_prompts import render_guard_prompt


ThinkingType = Literal["fast", "slow"]
PolicyStyle = Literal["full", "summary", "title_only", "rewritten", "merged"]
PolicyTransition = Literal[
    "unsafe_to_unsafe",
    "unsafe_to_safe",
    "safe_to_unsafe",
    "safe_to_safe",
]


class ActivePolicy(BaseModel):
    """One runtime policy set containing one or more active rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_id: str = Field(min_length=1)
    rules: tuple[PolicyRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_rules(self) -> "ActivePolicy":
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


class Message(BaseModel):
    """One portable ms-swift conversation message."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant", "tool_call", "tool_response"]
    content: str = Field(min_length=1)


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
            ),
        ),
        Message(role="user", content="\n".join(lines)),
    )


class ContentSample(BaseModel):
    """One normalized moderation target before policy conditioning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(min_length=1)
    split_group: str = Field(min_length=1)
    query: str = Field(min_length=1)
    response: str | None = None
    images: tuple[str, ...] = ()
    videos: tuple[str, ...] = ()


class PolicyView(BaseModel):
    """One rendered view of the complete active policy for an anchor sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    view_id: str = Field(min_length=1)
    active_policy: tuple[PolicyRule, ...] = Field(min_length=1)
    style: PolicyStyle
    transition: PolicyTransition
    stage: Literal["before", "after"] = "after"

    @model_validator(mode="after")
    def reject_ambiguous_rules(self) -> "PolicyView":
        rule_ids = [rule.rule_id for rule in self.active_policy]
        titles = [rule.title for rule in self.active_policy]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("active policy must not repeat rule_id values")
        if len(titles) != len(set(titles)):
            raise ValueError("active policy must not repeat rule titles")
        return self


class RuleCheck(BaseModel):
    """One auditable judgment against an active rule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_title: str = Field(min_length=1)
    verdict: Literal["hit", "not_hit", "not_applicable", "observed_not_unsafe"]
    evidence: str = Field(min_length=1)


class SingGuardAnnotation(BaseModel):
    """Policy-conditioned final label and optional verified reasoning trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: Literal["safe", "unsafe"]
    rule_title: str | None = None
    summary: str | None = None
    checks: tuple[RuleCheck, ...] = ()


class SingGuardExample(BaseModel):
    """A complete, independently splittable SingGuard training example."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: ContentSample
    policy: PolicyView
    annotation: SingGuardAnnotation
    thinking_type: ThinkingType

    @model_validator(mode="after")
    def validate_active_answer(self) -> "SingGuardExample":
        active_titles = {rule.title for rule in self.policy.active_policy}
        answer = self.annotation.rule_title
        if self.annotation.label == "safe" and answer is not None:
            raise ValueError("safe annotation must not name a triggered rule")
        if self.annotation.label == "unsafe" and answer not in active_titles:
            raise ValueError("unsafe annotation rule_title must belong to the active policy")
        if self.thinking_type == "slow":
            if not self.annotation.summary:
                raise ValueError("slow annotation requires a content summary")
            expected_titles = [rule.title for rule in self.policy.active_policy]
            check_titles = [check.rule_title for check in self.annotation.checks]
            if check_titles != expected_titles:
                raise ValueError("slow annotation must check every active rule in policy order")
            hit_titles = {
                check.rule_title for check in self.annotation.checks if check.verdict == "hit"
            }
            if self.annotation.label == "unsafe" and answer not in hit_titles:
                raise ValueError("unsafe slow annotation must mark its answer rule as hit")
            if self.annotation.label == "safe" and hit_titles:
                raise ValueError("safe slow annotation must not contain a hit rule")
        return self


def _system_prompt(example: SingGuardExample) -> str:
    return render_guard_prompt(
        example.policy.active_policy,
        thinking_type=example.thinking_type,
    )


def _user_content(content: ContentSample) -> str:
    lines = [f"[user]: {content.query}"]
    if content.response is not None:
        lines.append(f"[assistant]: {content.response}")
    return "\n".join(lines)


def render_sft_row(example: SingGuardExample) -> dict[str, object]:
    """Render one ms-swift messages row under the SingGuard output grammar."""

    answer = example.annotation.rule_title or "Safe"
    assistant = f"{example.annotation.label}\n<answer>{answer}</answer>"
    if example.thinking_type == "slow":
        verdicts = {
            "hit": "HIT",
            "not_hit": "NOT HIT",
            "not_applicable": "NOT APPLICABLE",
            "observed_not_unsafe": "OBSERVED BUT NOT UNSAFE",
        }
        checks = "\n".join(
            f"- {check.rule_title}: {verdicts[check.verdict]}. {check.evidence}"
            for check in example.annotation.checks
        )
        final_judgment = (
            f"The content violates the active rule {answer}."
            if example.annotation.label == "unsafe"
            else "No active rule is violated, so the content is safe."
        )
        assistant = (
            f"{example.annotation.label}\n<reasoning>\n"
            f"[Step 1] Content Summary\n{example.annotation.summary}\n\n"
            f"[Step 2] Check Risk Categories\n{checks}\n\n"
            f"[Step 3] Final Judgment\n{final_judgment}\n"
            f"</reasoning>\n<answer>{answer}</answer>"
        )
    row: dict[str, object] = {
        "messages": [
            {"role": "system", "content": _system_prompt(example)},
            {"role": "user", "content": _user_content(example.content)},
            {"role": "assistant", "content": assistant},
        ]
    }
    if example.content.images:
        row["images"] = list(example.content.images)
    if example.content.videos:
        row["videos"] = list(example.content.videos)
    return row
