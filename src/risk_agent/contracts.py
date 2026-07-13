"""Immutable data contracts for policy-driven content-risk assessment."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


def _freeze(value: Any) -> Any:
    """Recursively freeze JSON-compatible tool arguments."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    raise TypeError(f"Action arguments must contain JSON-compatible values, not {type(value).__name__}")


def _thaw(value: Any) -> Any:
    """Recursively convert frozen argument containers to JSON containers."""

    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


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
    arguments: Mapping[str, Any]

    @field_validator("arguments", mode="after")
    @classmethod
    def freeze_arguments(cls, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """Make tool arguments deeply immutable after type validation."""

        try:
            return _freeze(arguments)
        except TypeError as error:
            raise ValueError(str(error)) from error

    @field_serializer("arguments")
    def serialize_arguments(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Expose frozen arguments as ordinary JSON-compatible containers."""

        return _thaw(arguments)


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
