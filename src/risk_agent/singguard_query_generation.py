"""Generated-content contracts and deterministic local quality gates."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from risk_agent.singguard import ActivePolicy
from risk_agent.singguard_query_planning import (
    QueryBlueprint,
    SourceRef,
    plan_blueprints,
)


MAX_CONTENT_CHARS = 5_000
MAX_SOURCE_CHARS = 5_000
CONTENT_CONTRACT_VERSION = "singguard-query-generator-v1"
ENGLISH_ALPHA_RATIO = 0.80
NEAR_DUPLICATE_THRESHOLD = 0.85
SOURCE_SIMILARITY_THRESHOLD = 0.50
LENGTH_BOUNDS_VERSION = "singguard-length-bounds-v1"
LENGTH_BOUNDS = MappingProxyType(
    {
        "headline": (1, 20),
        "short": (5, 60),
        "medium": (30, 160),
        "long": (100, 400),
    }
)
GateCode = Literal[
    "accepted",
    "schema_or_shape",
    "literal_role_wrapper",
    "wrong_language",
    "generation_meta_language",
    "pii_or_external_identifier",
    "operational_harm",
    "length_out_of_bin",
    "exact_duplicate",
    "source_too_similar",
    "near_duplicate",
]

CONTENT_BATCH_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["blueprint_id", "query", "response"],
                "properties": {
                    "blueprint_id": {"type": "string"},
                    "query": {"type": "string"},
                    "response": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                },
            },
        }
    },
}

_ROLE_WRAPPER_RE = re.compile(r"\[(?:user|assistant)\](?:\s*:)?", re.IGNORECASE)
_META_LANGUAGE_RES = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bdataset\s+(?:training\s+)?sample\b",
        r"\btraining\s+sample\b",
        r"\boracle\s+answer\b",
        r"\bannotat(?:or|ion)\s+instruction\b",
        r"\bactive[- ]policy\s+(?:rule\s+)?id\b",
        r"\brule[- ]id\s+metadata\b",
        r"\bgenerate\s+(?:an?\s+)?(?:safe|unsafe)\s+example\b",
    )
)
_LABEL_DECISION_RES = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        (
            r"\blabel(?:s|ed)?\b(?:\s+\w+){0,6}\s+"
            r"(?:(?:is|as)\s+|should\s+be\s+)?(?:safe|unsafe)\b"
        ),
        r"\b(?:safe|unsafe)\b(?:\s+\w+){0,6}\s+\blabel(?:s|ed)?\b",
    )
)
_LABEL_META_CONTEXT_RE = re.compile(
    r"\b(?:generated\s+(?:sample|example|item|output|response|content)|"
    r"training\s+(?:sample|data|dataset|example|corpus|label)|dataset|gold|"
    r"oracle|annotat(?:or|ion)|policy\s+metadata|"
    r"rule(?:[- ]id)?\s+metadata)\b",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])([A-Z0-9.!#$%&'*+/=?^_`{|}~-]+)@"
    r"([A-Z0-9-]+(?:\.[A-Z0-9-]+)+)\b",
    re.IGNORECASE,
)
_EXPLICIT_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>'\"\])}]+", re.IGNORECASE)
_DOMAIN_RE = re.compile(
    r"(?<![\w@.-])([A-Z0-9-]+(?:\.[A-Z0-9-]+)*\.[A-Z]{2,})(?![\w.-])",
    re.IGNORECASE,
)
_HANDLE_RE = re.compile(r"(?<![\w@])@([A-Z0-9_]{1,32})\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<![\w-])(?:\+?\d[\s().-]*){10,15}(?![\w-])")
_CARD_RE = re.compile(r"(?<![\w-])(?:\d[ -]?){13,19}(?![\w-])")
_ORDER_IDENTIFIER_RE = re.compile(
    r"\b(?:order|tracking)(?:\s+(?:id|number|no\.?|#))?\s*[:#=\-]?\s+"
    r"(?=[A-Z0-9-]{6,32}\b)(?=[A-Z0-9-]*\d)[A-Z0-9][A-Z0-9-]{5,31}\b",
    re.IGNORECASE,
)
_KNOWN_SECRET_PREFIX_RE = re.compile(
    r"\b(?:sk_(?:live|test)|pk_(?:live|test)|gh[pousr]_|AKIA)"
    r"[A-Z0-9_\-]{8,}\b",
    re.IGNORECASE,
)
_EXPLICIT_SECRET_VALUE_RE = re.compile(
    r"\b(?:api[_ -]?key|access[_ -]?token|credential|secret|password|passwd|pwd)\b"
    r"\s*(?:(?:is)\s+|[:=]\s*)[\"']?([A-Z0-9_./+\-]{6,})",
    re.IGNORECASE,
)
_BARE_SECRET_VALUE_RE = re.compile(
    r"\b(?:api[_ -]?key|access[_ -]?token|credential|secret|password|passwd|pwd)\b"
    r"\s+[\"']?([A-Z0-9_./+\-]{6,})",
    re.IGNORECASE,
)
_BENIGN_SECRET_TOKEN_RE = re.compile(
    r"^(?:requirements\d{2,4}|rotation-policy|policy|"
    r"(?:guidelines?|standards?|documents?|versions?)"
    r"(?:[-_.]?v?\d{1,4}(?:[._-]\d+)*)?)$",
    re.IGNORECASE,
)
_ACCOUNT_RE = re.compile(
    r"\b(?:bank\s+)?account\s+(?:number|no\.?|#)\s*[:=]?\s*\d{8,20}\b",
    re.IGNORECASE,
)
_SEQUENCE_RE = re.compile(r"\b(?:first|then|next|finally|step\s+\d+)\b", re.IGNORECASE)
_CLAUSE_SPLIT_RE = re.compile(
    r"[.!?;]+|\b(?:first|then|next|finally|step\s+\d+)\b[:,]?",
    re.IGNORECASE,
)
_CREDENTIAL_THEFT_LITERAL_RE = re.compile(r"\bcredential\s+theft\b", re.IGNORECASE)
_CREDENTIAL_OBJECT_RE = re.compile(
    r"\b(?:bank\s+login\s+page|login\s+page|sign-in\s+page|"
    r"passwords?|credentials?|capture\s+service)\b",
    re.IGNORECASE,
)
_CREDENTIAL_ATTACK_ACTION_RE = re.compile(
    r"\b(?:clone|assemble|capture|connect|exfiltrate|deploy)\w*\b",
    re.IGNORECASE,
)
_DEFENSIVE_CONTEXT_RE = re.compile(
    r"\b(?:isolated\s+sandbox|forensic\s+analysis|remediat(?:e|ion)|"
    r"incident[- ]response|law[- ]enforcement|bomb[- ]squad|police\s+response)\b",
    re.IGNORECASE,
)
_HARD_EXPLOSIVE_CONSTRUCTION_RE = re.compile(
    r"\b(?:mix|combine|pack|wire|ignite|detonate)\w*\b",
    re.IGNORECASE,
)
_EXPLOSIVE_OR_WEAPON_DOMAIN_RE = re.compile(
    r"\b(?:weapons?|explosive|detonator|ammonium\s+nitrate|fuel\s+oil|"
    r"steel\s+pipe|timed\s+ignition|ignition|ammunition)\b",
    re.IGNORECASE,
)
_EXPLOSIVE_ASSEMBLY_ACTION_RE = re.compile(
    r"\b(?:assemble|load|connect|place|attach|insert|seal)\w*\b",
    re.IGNORECASE,
)
_MALWARE_DOMAIN_RE = re.compile(
    r"\b(?:malware|ransomware|payload)\b",
    re.IGNORECASE,
)
_MALWARE_CREATION_RE = re.compile(
    r"\b(?:write|create|build)\w*\s+(?:(?:a|an|the)\s+)?"
    r"(?:malware|ransomware|payload)\b(?!\s+(?:incident\s+)?report\b)",
    re.IGNORECASE,
)
_MALWARE_ASSEMBLY_RE = re.compile(
    r"\bassemble\w*\s+(?:(?:a|an|the)\s+)?(?:malware|ransomware|payload)\b",
    re.IGNORECASE,
)
_MALWARE_PERSISTENCE_RE = re.compile(
    r"\bconfigure\w*\s+persistence\b",
    re.IGNORECASE,
)
_MALWARE_EXECUTE_PAYLOAD_RE = re.compile(
    r"\bexecute\w*\s+(?:the\s+)?payload\b",
    re.IGNORECASE,
)
_MALWARE_DEPLOYMENT_ACTION_RE = re.compile(
    r"\b(?:deploy|install)\w*\b",
    re.IGNORECASE,
)
_MALWARE_DEPLOYMENT_OBJECT_RE = re.compile(
    r"\b(?:deploy|install)\w*\s+(?:(?:a|an|the)\s+)?"
    r"(?:malware|ransomware|payload)\b",
    re.IGNORECASE,
)
_ATTACK_TARGET_RE = re.compile(r"\b(?:targets?|victims?|hosts?|servers?)\b", re.IGNORECASE)
_LOADER_CONNECT_RE = re.compile(
    r"\bconnect\w*(?:\s+\w+){0,5}\s+loader\s+service\b",
    re.IGNORECASE,
)
_EXECUTE_PRONOUN_TARGET_RE = re.compile(
    r"\bexecute\w*\s+(?:it|them|this|that)\b(?:\s+\w+){0,6}\s+"
    r"(?:targets?|victims?|hosts?|servers?|target\s+machines?)\b",
    re.IGNORECASE,
)
_MALWARE_REPORT_RE = re.compile(
    r"\b(?:malware|ransomware|payload)\s+(?:incident\s+)?report\b",
    re.IGNORECASE,
)
_MALWARE_SAMPLE_RE = re.compile(
    r"\b(?:packaged\s+)?(?:malware|ransomware|payload)\s+sample\b",
    re.IGNORECASE,
)
_DOWNLOAD_RE = re.compile(r"\bdownload\w*\b", re.IGNORECASE)
_EXECUTE_RE = re.compile(r"\bexecute\w*\b", re.IGNORECASE)


class GeneratedContent(BaseModel):
    """One provider-produced candidate tied to a planned blueprint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    blueprint_id: str = Field(min_length=1, strict=True)
    query: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS, strict=True)
    response: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_CONTENT_CHARS,
        strict=True,
    )

    @field_validator("blueprint_id", "query", "response")
    @classmethod
    def reject_blank_strings(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("generated content strings must be non-blank")
        return value


def parse_content_batch(
    payload: Mapping[str, object],
    *,
    expected_ids: tuple[str, ...],
) -> tuple[GeneratedContent, ...]:
    """Strictly validate one provider batch without reordering or coercion."""

    if type(expected_ids) is not tuple or not 1 <= len(expected_ids) <= 4:
        raise ValueError("expected_ids must contain 1..4 IDs")
    if (
        any(type(item_id) is not str or not item_id.strip() for item_id in expected_ids)
        or len(set(expected_ids)) != len(expected_ids)
    ):
        raise ValueError("every expected blueprint ID must occur exactly once")
    if not isinstance(payload, Mapping):
        raise ValueError("content batch must be a mapping")
    if set(payload) != {"items"}:
        raise ValueError("content batch must contain exactly the items property")
    raw_items = payload["items"]
    if type(raw_items) is not list:
        raise ValueError("content batch items must be a list")
    if not 1 <= len(raw_items) <= 4:
        raise ValueError("content batch items must contain 1..4 entries")

    parsed: list[GeneratedContent] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            raise ValueError("content batch contains an invalid item")
        try:
            parsed.append(GeneratedContent.model_validate(raw_item, strict=True))
        except ValidationError:
            raise ValueError("content batch contains an invalid item") from None

    actual_ids = tuple(item.blueprint_id for item in parsed)
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
        raise ValueError("every expected blueprint ID must occur exactly once")
    return tuple(parsed)


def _prompt_bytes() -> bytes:
    path = Path(__file__).resolve().parents[2] / "prompts" / "singguard_query_generator_v1.txt"
    try:
        return path.read_bytes()
    except OSError:
        raise RuntimeError("cannot load the SingGuard query generator prompt") from None


def _redact_source_seed(text: str) -> str:
    if type(text) is not str:
        raise TypeError("source text must be a string")
    redacted = re.sub(
        r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[EMAIL]",
        text,
    )
    redacted = re.sub(r"(?i)https?://\S+", "[URL]", redacted)
    redacted = re.sub(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)", "[PHONE]", redacted)
    redacted = " ".join(redacted.split())
    if not redacted or len(redacted) > MAX_SOURCE_CHARS:
        raise ValueError("source text must be nonblank and at most 5000 characters")
    return redacted


def _source_text_for(
    source_ref: SourceRef,
    *,
    source_texts: Mapping[object, str],
    source_id_counts: Mapping[str, int],
) -> str:
    tuple_key = (source_ref.source, source_ref.source_id)
    string_key = f"{source_ref.source}:{source_ref.source_id}"
    qualified_matches = [key for key in (tuple_key, string_key) if key in source_texts]
    if len(qualified_matches) > 1:
        raise ValueError("source text lookup contains colliding qualified keys")
    if qualified_matches:
        return _redact_source_seed(source_texts[qualified_matches[0]])
    if source_ref.source_id in source_texts:
        if source_id_counts[source_ref.source_id] != 1:
            raise ValueError("source text bare ID lookup is ambiguous")
        return _redact_source_seed(source_texts[source_ref.source_id])
    raise ValueError("source text is missing for a blueprint source reference")


def _policy_payload(policy: ActivePolicy) -> dict[str, object]:
    return {
        "policy_id": policy.policy_id,
        "rules": [
            {
                "rule_id": rule.rule_id,
                "title": rule.title,
                "description": rule.text,
                "allowed_exceptions": list(rule.exceptions),
                "priority": rule.priority,
            }
            for rule in policy.rules
        ],
    }


def _blueprint_payload(
    blueprint: QueryBlueprint,
    *,
    policy: ActivePolicy,
    source_texts: Mapping[object, str],
    source_id_counts: Mapping[str, int],
) -> dict[str, object]:
    expected_rule_ids = tuple(rule.rule_id for rule in policy.rules)
    expected_rule_titles = tuple(rule.title for rule in policy.rules)
    if (
        blueprint.active_rule_ids != expected_rule_ids
        or blueprint.active_rule_titles != expected_rule_titles
    ):
        raise ValueError("blueprint active rules do not match the resolved policy")

    intended_target: dict[str, object] = {
        "intended_label": blueprint.intended_label,
        "primary_answer": blueprint.primary_answer,
        "intended_answers": list(blueprint.intended_answers),
    }
    if blueprint.target_exception_rule_id is not None:
        rule_title = next(
            rule.title
            for rule in policy.rules
            if rule.rule_id == blueprint.target_exception_rule_id
        )
        intended_target["safe_exception_context"] = {
            "rule_title": rule_title,
            "exception": blueprint.target_exception,
            "hard_negative": True,
        }

    item: dict[str, object] = {
        "blueprint_id": blueprint.blueprint_id,
        "policy_id": blueprint.policy_id,
        "controls": {
            "conversation_shape": blueprint.conversation_shape,
            "content_form": blueprint.content_form,
            "tone": blueprint.tone,
            "length_bin": blueprint.length_bin,
            "difficulty": blueprint.difficulty,
            "noise_profile": blueprint.noise_profile,
            "thinking_type": blueprint.thinking_type,
            "tool_capable": blueprint.tool_capable,
        },
        "intended_target": intended_target,
    }
    if blueprint.source_ref is not None:
        item["source_seed"] = {
            "trust": "untrusted_quoted_data",
            "use": "style_inspiration_only",
            "text": _source_text_for(
                blueprint.source_ref,
                source_texts=source_texts,
                source_id_counts=source_id_counts,
            ),
        }
    return item


def build_content_request(
    blueprints: tuple[QueryBlueprint, ...],
    *,
    policies: tuple[ActivePolicy, ...],
    source_texts: Mapping[str, str],
) -> dict[str, object]:
    """Build a content-only Gemini request with no provider orchestration."""

    if type(blueprints) is not tuple or not 1 <= len(blueprints) <= 4:
        raise ValueError("content request must contain 1..4 blueprints")
    if any(not isinstance(blueprint, QueryBlueprint) for blueprint in blueprints):
        raise TypeError("blueprints must contain QueryBlueprint items")
    blueprint_ids = tuple(blueprint.blueprint_id for blueprint in blueprints)
    if len(set(blueprint_ids)) != len(blueprint_ids):
        raise ValueError("blueprint IDs must be unique")
    if type(policies) is not tuple or any(
        not isinstance(policy, ActivePolicy) for policy in policies
    ):
        raise TypeError("policies must contain ActivePolicy items")
    if not isinstance(source_texts, Mapping):
        raise TypeError("source_texts must be a mapping")

    policies_by_id: dict[str, list[ActivePolicy]] = {}
    for policy in policies:
        policies_by_id.setdefault(policy.policy_id, []).append(policy)
    selected: list[ActivePolicy] = []
    resolved: dict[str, ActivePolicy] = {}
    for blueprint in blueprints:
        matches = policies_by_id.get(blueprint.policy_id, [])
        if len(matches) != 1:
            raise ValueError("every blueprint policy_id must resolve exactly once")
        if blueprint.policy_id not in resolved:
            resolved[blueprint.policy_id] = matches[0]
            selected.append(matches[0])

    source_id_counts: dict[str, int] = {}
    seen_refs: set[tuple[str, str]] = set()
    for blueprint in blueprints:
        if blueprint.source_ref is None:
            continue
        ref_key = (blueprint.source_ref.source, blueprint.source_ref.source_id)
        if ref_key not in seen_refs:
            source_id_counts[blueprint.source_ref.source_id] = (
                source_id_counts.get(blueprint.source_ref.source_id, 0) + 1
            )
            seen_refs.add(ref_key)

    prompt_bytes = _prompt_bytes()
    prompt = prompt_bytes.decode("utf-8")
    return {
        "contract_version": CONTENT_CONTRACT_VERSION,
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "prompt": prompt,
        "active_policies": [_policy_payload(policy) for policy in selected],
        "items": [
            _blueprint_payload(
                blueprint,
                policy=resolved[blueprint.policy_id],
                source_texts=source_texts,
                source_id_counts=source_id_counts,
            )
            for blueprint in blueprints
        ],
        "output_contract": {
            "items": [
                {
                    "blueprint_id": "string",
                    "query": "string",
                    "response": "string|null",
                }
            ]
        },
        "response_schema": CONTENT_BATCH_SCHEMA,
    }


class GateResult(BaseModel):
    """Stable first-failure result from the local gate sequence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted: bool = Field(strict=True)
    code: GateCode

    @field_validator("code")
    @classmethod
    def reject_blank_code(cls, value: GateCode) -> GateCode:
        if not value.strip():
            raise ValueError("gate result code must be non-blank")
        return value

    @model_validator(mode="after")
    def validate_accepted_code(self) -> "GateResult":
        if self.accepted and self.code != "accepted":
            raise ValueError("accepted gate results must use the accepted code")
        if not self.accepted and self.code == "accepted":
            raise ValueError("rejected gate results cannot use the accepted code")
        return self


class CandidateIndex:
    """Accepted candidates used by duplicate gates."""

    def __init__(self) -> None:
        self._rows: list[tuple[str, str, frozenset[tuple[str, ...]]]] = []

    def add(self, family_id: str, content: GeneratedContent) -> None:
        """Index an explicitly accepted candidate without changing the candidate."""

        _validate_index_inputs(family_id, content)
        self._rows.append(
            (
                family_id,
                _canonical_candidate(content),
                _five_grams(_combined_text(content)),
            )
        )

    def duplicate_code(
        self, family_id: str, content: GeneratedContent
    ) -> str | None:
        """Return the stable duplicate reason, if any, without adding the candidate."""

        _validate_index_inputs(family_id, content)
        canonical = _canonical_candidate(content)
        grams = _five_grams(_combined_text(content))
        if any(row_canonical == canonical for _, row_canonical, _ in self._rows):
            return "exact_duplicate"
        if any(
            row_family != family_id
            and _jaccard(grams, row_grams) >= NEAR_DUPLICATE_THRESHOLD
            for row_family, _, row_grams in self._rows
        ):
            return "near_duplicate"
        return None


def _combined_text(content: GeneratedContent) -> str:
    if content.response is None:
        return content.query
    return f"{content.query}\n{content.response}"


def _validate_index_inputs(family_id: str, content: GeneratedContent) -> None:
    if type(family_id) is not str:
        raise TypeError("family_id must be a string")
    if not family_id.strip():
        raise ValueError("family_id must be non-blank")
    if not isinstance(content, GeneratedContent):
        raise TypeError("content must be GeneratedContent")


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _canonical_candidate(content: GeneratedContent) -> str:
    query = _normalize_text(content.query)
    if content.response is None:
        return f"query\x1f{query}\x1eresponse:none"
    response = _normalize_text(content.response)
    return f"query\x1f{query}\x1eresponse\x1f{response}"


def _five_grams(text: str) -> frozenset[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", _normalize_text(text))
    return frozenset(tuple(tokens[index : index + 5]) for index in range(len(tokens) - 4))


def _jaccard(
    left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]
) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _source_too_similar(source_text: str, content: GeneratedContent) -> bool:
    source_normalized = _normalize_text(source_text)
    source_grams = _five_grams(source_text)
    candidate_views = [content.query]
    if content.response is not None:
        candidate_views.append(content.response)
    candidate_views.append(_combined_text(content))
    return any(
        source_normalized == _normalize_text(candidate_view)
        or _jaccard(source_grams, _five_grams(candidate_view))
        >= SOURCE_SIMILARITY_THRESHOLD
        for candidate_view in candidate_views
    )


def _is_english_dominant(text: str) -> bool:
    alphabetic_count = sum(character.isalpha() for character in text)
    ascii_letter_count = sum(
        ("a" <= character <= "z") or ("A" <= character <= "Z")
        for character in text
    )
    return bool(ascii_letter_count) and (
        ascii_letter_count / alphabetic_count >= ENGLISH_ALPHA_RATIO
    )


def _contains_generation_meta_language(text: str) -> bool:
    if any(pattern.search(text) for pattern in _META_LANGUAGE_RES):
        return True
    return bool(_LABEL_META_CONTEXT_RE.search(text)) and any(
        pattern.search(text) for pattern in _LABEL_DECISION_RES
    )


def _rejected(code: GateCode) -> GateResult:
    return GateResult(accepted=False, code=code)


def _blank_spans(text: str, matches: list[re.Match[str]]) -> str:
    characters = list(text)
    for match in matches:
        characters[match.start() : match.end()] = " " * (match.end() - match.start())
    return "".join(characters)


def _reserved_test_host(host: str) -> bool:
    return host.casefold().rstrip(".").endswith(".test")


def _luhn_valid(digits: str) -> bool:
    checksum = 0
    parity = len(digits) % 2
    for index, character in enumerate(digits):
        value = int(character)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        checksum += value
    return checksum % 10 == 0


def _looks_like_secret_value(value: str) -> bool:
    """Distinguish compact secret values from narrow documentation/version tokens."""

    if _KNOWN_SECRET_PREFIX_RE.search(value):
        return True
    if _BENIGN_SECRET_TOKEN_RE.fullmatch(value):
        return False
    has_letters = any(character.isalpha() for character in value)
    has_digits = any(character.isdigit() for character in value)
    has_secret_punctuation = any(character in "_./+" for character in value)
    return len(value) >= 10 and has_letters and (has_digits or has_secret_punctuation)


def _contains_credential_secret(text: str) -> bool:
    if _KNOWN_SECRET_PREFIX_RE.search(text):
        return True
    return any(
        _looks_like_secret_value(match.group(1))
        for pattern in (_EXPLICIT_SECRET_VALUE_RE, _BARE_SECRET_VALUE_RE)
        for match in pattern.finditer(text)
    )


def _explicit_url_host(url: str) -> str | None:
    token = url.rstrip(",;.!?")
    candidate = token if "://" in token else f"http://{token}"
    try:
        return urlsplit(candidate).hostname
    except ValueError:
        return None


def _contains_external_identifier(text: str) -> bool:
    email_matches = list(_EMAIL_RE.finditer(text))
    if any(not _reserved_test_host(match.group(2)) for match in email_matches):
        return True
    scrubbed = _blank_spans(text, email_matches)

    url_matches = list(_EXPLICIT_URL_RE.finditer(scrubbed))
    if any(
        (host := _explicit_url_host(match.group())) is None
        or not _reserved_test_host(host)
        for match in url_matches
    ):
        return True
    scrubbed = _blank_spans(scrubbed, url_matches)

    domain_matches = list(_DOMAIN_RE.finditer(scrubbed))
    if any(not _reserved_test_host(match.group(1)) for match in domain_matches):
        return True
    scrubbed = _blank_spans(scrubbed, domain_matches)

    card_matches = list(_CARD_RE.finditer(scrubbed))
    if any(
        _luhn_valid("".join(character for character in match.group() if character.isdigit()))
        for match in card_matches
    ):
        return True

    order_matches = list(_ORDER_IDENTIFIER_RE.finditer(scrubbed))
    scrubbed = _blank_spans(scrubbed, order_matches)

    for match in _HANDLE_RE.finditer(scrubbed):
        handle = match.group(1).casefold()
        if not handle.startswith(("synthetic_", "test_", "example_")):
            return True
    if _contains_credential_secret(scrubbed) or _ACCOUNT_RE.search(scrubbed):
        return True
    if _PHONE_RE.search(scrubbed):
        return True
    return False


def _harm_clauses(text: str) -> tuple[str, ...]:
    return tuple(
        clause.strip(" ,:\n\t")
        for clause in _CLAUSE_SPLIT_RE.split(text)
        if clause.strip(" ,:\n\t")
    )


def _is_explosive_or_weapon_procedure(clauses: tuple[str, ...]) -> bool:
    relationship_count = sum(
        (
            len(_HARD_EXPLOSIVE_CONSTRUCTION_RE.findall(clause))
            + len(_EXPLOSIVE_ASSEMBLY_ACTION_RE.findall(clause))
        )
        if _EXPLOSIVE_OR_WEAPON_DOMAIN_RE.search(clause)
        else 0
        for clause in clauses
    )
    return relationship_count >= 2


def _is_credential_theft_procedure(
    text: str, clauses: tuple[str, ...]
) -> bool:
    literal_context = bool(_CREDENTIAL_THEFT_LITERAL_RE.search(text))
    relationship_count = sum(
        len(_CREDENTIAL_ATTACK_ACTION_RE.findall(clause))
        if literal_context or _CREDENTIAL_OBJECT_RE.search(clause)
        else 0
        for clause in clauses
    )
    return relationship_count >= 2


def _is_defensive_malware_sample(text: str, clauses: tuple[str, ...]) -> bool:
    if not (
        _MALWARE_SAMPLE_RE.search(text)
        and _DEFENSIVE_CONTEXT_RE.search(text)
        and _DOWNLOAD_RE.search(text)
        and _EXECUTE_RE.search(text)
    ):
        return False
    has_offensive_relationship = any(
        _MALWARE_CREATION_RE.search(clause)
        or _MALWARE_ASSEMBLY_RE.search(clause)
        or _MALWARE_PERSISTENCE_RE.search(clause)
        or _MALWARE_DEPLOYMENT_OBJECT_RE.search(clause)
        or _LOADER_CONNECT_RE.search(clause)
        or _EXECUTE_PRONOUN_TARGET_RE.search(clause)
        or (
            _MALWARE_EXECUTE_PAYLOAD_RE.search(clause)
            and _ATTACK_TARGET_RE.search(clause)
        )
        or (
            _MALWARE_DEPLOYMENT_ACTION_RE.search(clause)
            and _ATTACK_TARGET_RE.search(clause)
        )
        for clause in clauses
    )
    return not has_offensive_relationship


def _malware_relationship_count(clause: str) -> int:
    sample_download = bool(
        _MALWARE_SAMPLE_RE.search(clause) and _DOWNLOAD_RE.search(clause)
    )
    creation_or_assembly = bool(
        _MALWARE_CREATION_RE.search(clause) or _MALWARE_ASSEMBLY_RE.search(clause)
    )
    persistence = bool(_MALWARE_PERSISTENCE_RE.search(clause))
    targeted_execution = bool(
        _MALWARE_EXECUTE_PAYLOAD_RE.search(clause)
        or _EXECUTE_PRONOUN_TARGET_RE.search(clause)
    )
    loader_connection = bool(_LOADER_CONNECT_RE.search(clause))
    deployment = bool(
        _MALWARE_DEPLOYMENT_OBJECT_RE.search(clause)
        or (
            _MALWARE_DEPLOYMENT_ACTION_RE.search(clause)
            and _ATTACK_TARGET_RE.search(clause)
        )
    )
    return sum(
        (
            sample_download,
            creation_or_assembly,
            persistence,
            targeted_execution,
            loader_connection,
            deployment,
        )
    )


def _is_malware_procedure(text: str, clauses: tuple[str, ...]) -> bool:
    if _is_defensive_malware_sample(text, clauses):
        return False
    has_flow_context = any(
        _MALWARE_DOMAIN_RE.search(clause) and not _MALWARE_REPORT_RE.search(clause)
        for clause in clauses
    )
    if not has_flow_context:
        return False
    relationship_count = sum(_malware_relationship_count(clause) for clause in clauses)
    return relationship_count >= 2


def _is_operational_harm(text: str) -> bool:
    if len(_SEQUENCE_RE.findall(text)) < 2:
        return False
    clauses = _harm_clauses(text)
    return (
        _is_explosive_or_weapon_procedure(clauses)
        or _is_credential_theft_procedure(text, clauses)
        or _is_malware_procedure(text, clauses)
    )


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text, re.UNICODE))


def gate_content(
    blueprint: QueryBlueprint,
    content: GeneratedContent,
    *,
    source_text: str | None,
    index: CandidateIndex,
) -> GateResult:
    """Apply local checks in stable first-failure order."""

    if not isinstance(blueprint, QueryBlueprint):
        raise TypeError("blueprint must be a QueryBlueprint")
    if not isinstance(content, GeneratedContent):
        raise TypeError("content must be GeneratedContent")
    if not isinstance(index, CandidateIndex):
        raise TypeError("index must be CandidateIndex")
    if source_text is not None:
        if type(source_text) is not str:
            raise TypeError("source_text must be a string or None")
        if not source_text.strip():
            raise ValueError("source_text must be non-blank when supplied")
        if len(source_text) > MAX_SOURCE_CHARS:
            raise ValueError(f"source_text cannot exceed {MAX_SOURCE_CHARS} characters")

    if (
        content.blueprint_id != blueprint.blueprint_id
        or (blueprint.conversation_shape == "query" and content.response is not None)
        or (blueprint.conversation_shape == "query_response" and content.response is None)
    ):
        return _rejected("schema_or_shape")

    combined = _combined_text(content)
    scan_text = unicodedata.normalize("NFKC", combined)
    if _ROLE_WRAPPER_RE.search(scan_text):
        return _rejected("literal_role_wrapper")
    if not _is_english_dominant(scan_text):
        return _rejected("wrong_language")
    if _contains_generation_meta_language(scan_text):
        return _rejected("generation_meta_language")
    if _contains_external_identifier(scan_text):
        return _rejected("pii_or_external_identifier")
    if _is_operational_harm(scan_text):
        return _rejected("operational_harm")
    minimum, maximum = LENGTH_BOUNDS[blueprint.length_bin]
    if not minimum <= _word_count(scan_text) <= maximum:
        return _rejected("length_out_of_bin")
    duplicate_code = index.duplicate_code(blueprint.family_id, content)
    if duplicate_code == "exact_duplicate":
        return _rejected("exact_duplicate")
    if source_text is not None and _source_too_similar(source_text, content):
        return _rejected("source_too_similar")
    if duplicate_code == "near_duplicate":
        return _rejected("near_duplicate")
    return GateResult(accepted=True, code="accepted")


__all__ = [
    "GeneratedContent",
    "GateResult",
    "CandidateIndex",
    "ENGLISH_ALPHA_RATIO",
    "LENGTH_BOUNDS",
    "LENGTH_BOUNDS_VERSION",
    "MAX_SOURCE_CHARS",
    "NEAR_DUPLICATE_THRESHOLD",
    "SOURCE_SIMILARITY_THRESHOLD",
    "MAX_CONTENT_CHARS",
    "QueryBlueprint",
    "SourceRef",
    "plan_blueprints",
    "gate_content",
]
