"""A deliberately narrow crawler for approved, public research sources only."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import yaml


USER_AGENT = "risk-agent-research/0.1 (+contact-required)"
TIMEOUT_SECONDS = 20.0
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_ROBOTS_BYTES = 256 * 1024
_BLOCKED_ACCOUNT_SEGMENTS = frozenset({"account", "accounts", "auth", "login", "signin", "user", "users"})


@dataclass(frozen=True)
class Source:
    """One manually approved public URL from a source manifest.

    ``allowed_domains`` is intentionally an exact hostname allowlist; it does
    not authorize sibling or child subdomains.
    """

    url: str
    allowed_domains: tuple[str, ...]
    source_type: str = "unknown"
    license: str = "not_reviewed"
    terms_review: str = "not_reviewed"


@dataclass(frozen=True)
class SourceMetadata:
    """Provenance recorded beside a raw download, never training content."""

    url: str
    retrieved_at: str
    content_sha256: str
    source_type: str
    license: str
    terms_review: str
    raw_path: str


def validate_source(source: Source) -> str:
    """Validate a public source URL before any request is made."""

    parsed = urlparse(source.url)
    hostname = parsed.hostname
    allowed_domains = {domain.casefold() for domain in source.allowed_domains}
    if parsed.scheme.casefold() != "https":
        raise ValueError("source must use HTTPS")
    if not hostname or hostname.casefold() not in allowed_domains:
        raise ValueError("source must use an exact allowlisted hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("source URL must not contain credentials")
    if parsed.port not in (None, 443):
        raise ValueError("source URL must not use a non-standard port")
    if parsed.fragment:
        raise ValueError("source URL must not contain a fragment")
    path_segments = {segment.casefold() for segment in parsed.path.split("/") if segment}
    if path_segments & _BLOCKED_ACCOUNT_SEGMENTS:
        raise ValueError("source URL must not target a user-account page")
    if not all(isinstance(value, str) and value.strip() for value in (source.source_type, source.license, source.terms_review)):
        raise ValueError("source metadata fields must be non-empty strings")
    return source.url


def raw_path_for(source: Source, output_dir: Path) -> Path:
    """Return a filename derived only from the approved URL, not its path."""

    digest = hashlib.sha256(validate_source(source).encode("utf-8")).hexdigest()
    return _safe_child(output_dir, f"{digest}.raw")


def metadata_path_for(source: Source, output_dir: Path) -> Path:
    """Return the controlled provenance sidecar path for a source."""

    digest = hashlib.sha256(validate_source(source).encode("utf-8")).hexdigest()
    return _safe_child(output_dir / "metadata", f"{digest}.json")


def fetch(
    source: Source,
    output_dir: Path,
    *,
    client: httpx.Client | None = None,
    retrieved_at: str | None = None,
) -> Path:
    """Fetch one robots-approved public resource without processing its body.

    The resulting ``.raw`` file is the exact bounded response body.  The
    companion JSON file holds provenance only and must not be treated as
    training data.
    """

    url = validate_source(source)
    owns_client = client is None
    if client is None:
        client = httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=False, trust_env=False)
    try:
        _check_robots(client, url)
        body = _get_bounded(client, url, MAX_RESPONSE_BYTES)
    finally:
        if owns_client:
            client.close()

    raw_path = raw_path_for(source, output_dir)
    metadata_path = metadata_path_for(source, output_dir)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(body)
    metadata = SourceMetadata(
        url=url,
        retrieved_at=retrieved_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        content_sha256=hashlib.sha256(body).hexdigest(),
        source_type=source.source_type,
        license=source.license,
        terms_review=source.terms_review,
        raw_path=raw_path.name,
    )
    metadata_path.write_text(
        json.dumps(asdict(metadata), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return raw_path


def load_manifest(path: Path) -> tuple[Source, ...]:
    """Load and fully validate a YAML manifest before fetching any source."""

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML source manifest: {error}") from error
    if not isinstance(document, Mapping):
        raise ValueError("source manifest must be a mapping")
    allowed_domains = document.get("allowed_domains")
    source_rows = document.get("sources")
    if not isinstance(allowed_domains, list) or not allowed_domains or not all(
        isinstance(domain, str) and domain.strip() for domain in allowed_domains
    ):
        raise ValueError("source manifest requires non-empty allowed_domains")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("source manifest requires a non-empty sources list")

    domains = tuple(allowed_domains)
    sources = tuple(_source_from_row(row, domains) for row in source_rows)
    for source in sources:
        validate_source(source)
    return sources


def _source_from_row(row: Any, allowed_domains: tuple[str, ...]) -> Source:
    if isinstance(row, str):
        return Source(url=row, allowed_domains=allowed_domains)
    if not isinstance(row, Mapping):
        raise ValueError("each source must be a URL string or mapping")
    unknown_fields = set(row) - {"url", "source_type", "license", "terms_review"}
    if unknown_fields:
        raise ValueError(f"unknown source fields: {', '.join(sorted(map(str, unknown_fields)))}")
    url = row.get("url")
    if not isinstance(url, str):
        raise ValueError("each source mapping requires a string url")
    metadata = {name: row.get(name, default) for name, default in (
        ("source_type", "unknown"),
        ("license", "not_reviewed"),
        ("terms_review", "not_reviewed"),
    )}
    return Source(url=url, allowed_domains=allowed_domains, **metadata)


def _check_robots(client: httpx.Client, url: str) -> None:
    parsed = urlparse(url)
    robots_url = f"https://{parsed.hostname}/robots.txt"
    robots_body = _get_bounded(client, robots_url, MAX_ROBOTS_BYTES)
    try:
        robots_text = robots_body.decode("utf-8")
    except UnicodeDecodeError:
        robots_text = robots_body.decode("utf-8", errors="replace")
    parser = RobotFileParser()
    parser.parse(robots_text.splitlines())
    if not parser.can_fetch(USER_AGENT, url):
        raise PermissionError("robots policy disallows this URL")


def _get_bounded(client: httpx.Client, url: str, byte_limit: int) -> bytes:
    """Issue one stateless request and reject redirects or oversized bodies."""

    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    with client.stream("GET", url, headers=headers, follow_redirects=False, timeout=TIMEOUT_SECONDS) as response:
        if response.is_redirect:
            raise ValueError("redirect responses are not allowed")
        response.raise_for_status()
        content_length = response.headers.get("content-length")
        if content_length is not None and int(content_length) > byte_limit:
            raise ValueError(f"response exceeds size limit of {byte_limit} bytes")
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > byte_limit:
                raise ValueError(f"response exceeds size limit of {byte_limit} bytes")
            chunks.append(chunk)
    return b"".join(chunks)


def _safe_child(parent: Path, filename: str) -> Path:
    """Defend output handling even if the configured output path is unusual."""

    resolved_parent = parent.resolve()
    candidate = (resolved_parent / filename).resolve()
    try:
        candidate.relative_to(resolved_parent)
    except ValueError as error:  # pragma: no cover - fixed digest names make this defensive.
        raise ValueError("crawler output escaped the configured output directory") from error
    return candidate
