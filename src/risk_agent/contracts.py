"""Immutable data contracts for policy-driven content-risk assessment."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FrozenDict(dict[str, Any]):
    """A dictionary that cannot be mutated after it is constructed."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FrozenDict is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _freeze(value: Any) -> Any:
    """Recursively replace mutable containers with immutable equivalents."""

    if isinstance(value, dict):
        return FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
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
    arguments: dict[str, Any]

    @field_validator("arguments", mode="after")
    @classmethod
    def freeze_arguments(cls, arguments: dict[str, Any]) -> FrozenDict:
        """Make tool arguments deeply immutable after type validation."""

        return _freeze(arguments)


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
