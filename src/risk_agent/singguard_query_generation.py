"""Generated-content contracts and deterministic local quality gates."""

from __future__ import annotations

import re
import unicodedata
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from risk_agent.singguard_query_planning import (
    QueryBlueprint,
    SourceRef,
    plan_blueprints,
)


MAX_CONTENT_CHARS = 5_000
MAX_SOURCE_CHARS = 5_000
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
_URL_RE = re.compile(
    r"\b(?:https?://|www\.)([A-Z0-9-]+(?:\.[A-Z0-9-]+)+)(?::\d+)?(?:[/?#][^\s]*)?",
    re.IGNORECASE,
)
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
_HARM_DOMAIN_RE = re.compile(
    r"\b(?:weapons?|explosive|detonator|ammonium\s+nitrate|fuel\s+oil|"
    r"steel\s+pipe|timed\s+ignition|ignition(?:\s+(?:circuit|timer))?|"
    r"(?:detonation|explosive)\s+(?:fuse|timer)|ammunition|credential\s+theft|"
    r"clone\s+the\s+login|malware|ransomware|payload)\b",
    re.IGNORECASE,
)
_HARM_ACTION_RE = re.compile(
    r"\b(?:mix|combine|pack|place|attach|assemble|load|connect|wire|insert|seal|"
    r"ignite|detonate|clone|capture|exfiltrate|write|download|configure|execute|"
    r"deploy|install)\w*\b",
    re.IGNORECASE,
)
_CREDENTIAL_THEFT_DOMAIN_RE = re.compile(
    r"\b(?:bank\s+login\s+page|login\s+page|passwords?|credentials?)\b",
    re.IGNORECASE,
)
_CREDENTIAL_THEFT_ACTION_RE = re.compile(
    r"\b(?:clone|capture|exfiltrate)\w*\b",
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
_ASSEMBLE_RE = re.compile(r"\bassemble\w*\b", re.IGNORECASE)
_EXPLOSIVE_ASSEMBLY_ACTION_RE = re.compile(
    r"\b(?:load|connect|place|attach|insert|seal)\w*\b",
    re.IGNORECASE,
)
_MALWARE_DOMAIN_RE = re.compile(
    r"\b(?:malware|ransomware|payload)\b",
    re.IGNORECASE,
)
_MALWARE_DEPLOYMENT_RE = re.compile(r"\b(?:deploy|install)\w*\b", re.IGNORECASE)


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


class GateResult(BaseModel):
    """Stable first-failure result from the local gate sequence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted: bool
    code: str = Field(min_length=1, strict=True)

    @field_validator("code")
    @classmethod
    def reject_blank_code(cls, value: str) -> str:
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


def _rejected(code: str) -> GateResult:
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


def _contains_external_identifier(text: str) -> bool:
    email_matches = list(_EMAIL_RE.finditer(text))
    if any(not _reserved_test_host(match.group(2)) for match in email_matches):
        return True
    scrubbed = _blank_spans(text, email_matches)

    url_matches = list(_URL_RE.finditer(scrubbed))
    if any(not _reserved_test_host(match.group(1)) for match in url_matches):
        return True
    scrubbed = _blank_spans(scrubbed, url_matches)

    domain_matches = list(_DOMAIN_RE.finditer(scrubbed))
    if any(not _reserved_test_host(match.group(1)) for match in domain_matches):
        return True
    scrubbed = _blank_spans(scrubbed, domain_matches)

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
    return any(
        _luhn_valid("".join(character for character in match.group() if character.isdigit()))
        for match in _CARD_RE.finditer(scrubbed)
    )


def _is_operational_harm(text: str) -> bool:
    if len(_SEQUENCE_RE.findall(text)) < 2 or len(_HARM_ACTION_RE.findall(text)) < 2:
        return False
    has_harm_domain = bool(_HARM_DOMAIN_RE.search(text)) or (
        bool(_CREDENTIAL_THEFT_DOMAIN_RE.search(text))
        and bool(_CREDENTIAL_THEFT_ACTION_RE.search(text))
    )
    if not has_harm_domain:
        return False
    if _DEFENSIVE_CONTEXT_RE.search(text):
        is_explosive_construction = bool(
            _EXPLOSIVE_OR_WEAPON_DOMAIN_RE.search(text)
        ) and (
            bool(_HARD_EXPLOSIVE_CONSTRUCTION_RE.search(text))
            or (
                bool(_ASSEMBLE_RE.search(text))
                and bool(_EXPLOSIVE_ASSEMBLY_ACTION_RE.search(text))
            )
        )
        is_credential_theft = bool(_CREDENTIAL_THEFT_DOMAIN_RE.search(text)) and len(
            _CREDENTIAL_THEFT_ACTION_RE.findall(text)
        ) >= 2
        is_malware_deployment = bool(_MALWARE_DOMAIN_RE.search(text)) and bool(
            _MALWARE_DEPLOYMENT_RE.search(text)
        )
        if not (
            is_explosive_construction
            or is_credential_theft
            or is_malware_deployment
        ):
            return False
    return True


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
    if _ROLE_WRAPPER_RE.search(combined):
        return _rejected("literal_role_wrapper")
    if not _is_english_dominant(combined):
        return _rejected("wrong_language")
    if _contains_generation_meta_language(combined):
        return _rejected("generation_meta_language")
    if _contains_external_identifier(combined):
        return _rejected("pii_or_external_identifier")
    if _is_operational_harm(combined):
        return _rejected("operational_harm")
    minimum, maximum = LENGTH_BOUNDS[blueprint.length_bin]
    if not minimum <= _word_count(combined) <= maximum:
        return _rejected("length_out_of_bin")
    duplicate_code = index.duplicate_code(blueprint.family_id, content)
    if duplicate_code == "exact_duplicate":
        return _rejected("exact_duplicate")
    if source_text is not None:
        normalized_source = _normalize_text(source_text)
        normalized_candidate = _normalize_text(combined)
        if normalized_source == normalized_candidate or _jaccard(
            _five_grams(source_text), _five_grams(combined)
        ) >= SOURCE_SIMILARITY_THRESHOLD:
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
