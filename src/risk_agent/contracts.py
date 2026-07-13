"""Immutable data contracts for policy-driven content-risk assessment."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PolicyRule(BaseModel):
    """One rule in the complete policy supplied with a task."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    title: str
    text: str
    exceptions: tuple[str, ...] = ()
    priority: int = 100


class Task(BaseModel):
    """A bounded assessment task containing its full active policy."""

    model_config = ConfigDict(frozen=True)

    asset_id: str
    policy_version: str
    active_policy: tuple[PolicyRule, ...]
    initial_observation: str
    max_turns: int = Field(default=3, ge=1, le=3)


class Evidence(BaseModel):
    """Evidence associated with an asset under review."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    asset_id: str
    kind: Literal["ocr", "asr", "frame", "metadata", "case"]
    content: str


class Oracle(BaseModel):
    """Reference assessment for an asset and policy version."""

    model_config = ConfigDict(frozen=True)

    asset_id: str
    policy_version: str
    label: Literal["safe", "unsafe"]
    rule_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    risk_level: Literal["P0", "P1", "P2", "P3"] | None = None
    next_action: str | None = None


class Action(BaseModel):
    """A tool request made while assessing an asset."""

    model_config = ConfigDict(frozen=True)

    tool: Literal["get_rule_detail", "search_case", "inspect_evidence", "final_decision"]
    arguments: dict[str, Any]


class Decision(BaseModel):
    """An immutable policy decision for an asset."""

    model_config = ConfigDict(frozen=True)

    label: Literal["safe", "unsafe"]
    rule_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["P0", "P1", "P2", "P3"] | None = None
    route: Literal["fast", "hybrid", "slow", "agent"] = "agent"
    next_action: str | None = None
