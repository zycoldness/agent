"""Load and render versioned SingGuard prompt contracts."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from risk_agent.contracts import PolicyRule


_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
_PROMPT_FILES = {
    "guard": "singguard_guard_v1.txt",
    "agent": "singguard_agent_v1.txt",
    "generator": "singguard_generator_v1.txt",
    "verifier": "singguard_verifier_v1.txt",
}


def _prompt_path(name: str) -> Path:
    try:
        filename = _PROMPT_FILES[name]
    except KeyError:
        raise ValueError(f"unknown SingGuard prompt {name!r}") from None
    return _PROMPT_DIR / filename


def load_prompt(name: str) -> str:
    """Read one allowlisted UTF-8 prompt template."""

    try:
        return _prompt_path(name).read_text(encoding="utf-8").rstrip()
    except OSError:
        raise RuntimeError(f"cannot read SingGuard {name} prompt") from None


def prompt_sha256(name: str) -> str:
    """Hash the exact versioned prompt bytes recorded by a data manifest."""

    try:
        payload = _prompt_path(name).read_bytes()
    except OSError:
        raise RuntimeError(f"cannot read SingGuard {name} prompt") from None
    return hashlib.sha256(payload).hexdigest()


def _render_policy(active_policy: tuple[PolicyRule, ...]) -> str:
    if not active_policy:
        raise ValueError("active policy must not be empty")
    blocks: list[str] = []
    for rule in active_policy:
        lines = [f"### {rule.title}", rule.text]
        lines.extend(f"Exception: {exception}" for exception in rule.exceptions)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _substitute(name: str, replacements: dict[str, str]) -> str:
    rendered = load_prompt(name)
    for placeholder, value in replacements.items():
        rendered = rendered.replace("{{" + placeholder + "}}", value)
    if re.search(r"\{\{[A-Z_]+\}\}", rendered):
        raise RuntimeError(f"unresolved placeholder in SingGuard {name} prompt")
    return rendered


def render_guard_prompt(
    active_policy: tuple[PolicyRule, ...],
    *,
    thinking_type: str,
) -> str:
    if thinking_type not in {"fast", "fast-slow", "slow"}:
        raise ValueError("thinking_type must be fast, fast-slow, or slow")
    return _substitute(
        "guard",
        {
            "ACTIVE_POLICY": _render_policy(active_policy),
            "THINKING_TYPE": thinking_type,
        },
    )


def render_agent_prompt(active_policy: tuple[PolicyRule, ...]) -> str:
    return _substitute("agent", {"ACTIVE_POLICY": _render_policy(active_policy)})
