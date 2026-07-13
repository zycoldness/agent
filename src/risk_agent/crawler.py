"""A deliberately narrow crawler for approved, public research sources only."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
from ipaddress import ip_address
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from unicodedata import normalize
from urllib.parse import parse_qsl, unquote, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import yaml


USER_AGENT = "risk-agent-research/0.1 (+contact-required)"
TIMEOUT_SECONDS = 20.0
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_ROBOTS_BYTES = 256 * 1024
# We intentionally allow only public records.  These route segments are common
# authenticated/profile surfaces and are blocked wherever they occur in a URL.
_BLOCKED_ACCOUNT_ROUTE_SEGMENTS = frozenset(
    {
        "account",
        "accounts",
        "auth",
        "dashboard",
        "dashboards",
        "login",
        "profile",
        "profiles",
        "setting",
        "settings",
        "signin",
        "user",
        "users",
    }
)
_SENSITIVE_QUERY_PARAMETER_NAMES = frozenset(
    {
        "accesstoken",
        "apikey",
        "authorization",
        "authtoken",
        "bearer",
        "cookie",
        "credential",
        "jwt",
        "key",
        "password",
        "passwd",
        "secret",
        "session",
        "sessionid",
        "sig",
        "signature",
        "token",
    }
)


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
    if not hostname or _is_local_or_ip_literal(hostname):
        raise ValueError("source must not use a local or IP-literal hostname")
    if not hostname or hostname.casefold() not in allowed_domains:
        raise ValueError("source must use an exact allowlisted hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("source URL must not contain credentials")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("source URL contains an invalid port") from error
    if port not in (None, 443):
        raise ValueError("source URL must not use a non-standard port")
    if parsed.fragment:
        raise ValueError("source URL must not contain a fragment")
    path_segments = set(_normalized_path_segments(parsed.path))
    if path_segments & _BLOCKED_ACCOUNT_ROUTE_SEGMENTS:
        raise ValueError("source URL must not target a user-account page")
    for name, _ in parse_qsl(parsed.query, keep_blank_values=True):
        normalized_name = "".join(character for character in name.casefold() if character.isalnum())
        if normalized_name in _SENSITIVE_QUERY_PARAMETER_NAMES:
            raise ValueError("source URL must not contain a credential-bearing query parameter")
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
    output_dir, metadata_dir = _prepare_output_directories(output_dir)
    raw_path = raw_path_for(source, output_dir)
    metadata_path = _safe_child(metadata_dir, metadata_path_for(source, output_dir).name)
    _reject_symlink(raw_path)
    _reject_symlink(metadata_path)
    owns_client = client is None
    if client is None:
        client = httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=False, trust_env=False)
    try:
        _check_robots(client, url)
        body = _get_bounded(client, url, MAX_RESPONSE_BYTES)
    finally:
        if owns_client:
            client.close()

    metadata = SourceMetadata(
        url=url,
        retrieved_at=retrieved_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        content_sha256=hashlib.sha256(body).hexdigest(),
        source_type=source.source_type,
        license=source.license,
        terms_review=source.terms_review,
        raw_path=raw_path.name,
    )
    # Both files are staged in their verified direct directories and atomically
    # replaced, so a failed write cannot leave a truncated raw or sidecar file.
    _atomic_write(raw_path, body)
    _atomic_write(metadata_path, (json.dumps(asdict(metadata), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
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
    """Construct a controlled child path from a fixed, non-user filename."""

    if Path(filename).name != filename:
        raise ValueError("crawler output filename must not contain a path")
    return parent / filename


def _is_local_or_ip_literal(hostname: str) -> bool:
    """Reject local names and all numeric addresses, including IPv6 forms."""

    normalized_hostname = hostname.casefold().rstrip(".")
    if normalized_hostname == "localhost" or normalized_hostname.endswith(".localhost") or normalized_hostname.endswith(".local"):
        return True
    try:
        ip_address(normalized_hostname)
    except ValueError:
        return False
    return True


def _normalized_path_segments(path: str) -> tuple[str, ...]:
    """Decode and normalize route segments before matching account surfaces."""

    decoded = path
    # One decode matches typical server handling; repeat a bounded number of
    # times to fail closed on common double-encoded route bypasses.
    for _ in range(3):
        next_decoded = unquote(decoded)
        if next_decoded == decoded:
            break
        decoded = next_decoded
    return tuple(normalize("NFKC", segment).casefold() for segment in decoded.split("/") if segment)


def _prepare_output_directories(output_dir: Path) -> tuple[Path, Path]:
    """Create only direct output directories; never traverse symlinks."""

    direct_output_dir = _verify_direct_directory(Path(output_dir))
    metadata_dir = _verify_direct_directory(direct_output_dir / "metadata")
    return direct_output_dir, metadata_dir


def _verify_direct_directory(directory: Path) -> Path:
    direct_directory = directory.absolute()
    _reject_symlinked_ancestors(direct_directory)
    if direct_directory.exists() and not direct_directory.is_dir():
        raise ValueError("crawler output path must be a directory")
    direct_directory.mkdir(parents=True, exist_ok=True)
    _reject_symlinked_ancestors(direct_directory)
    if not direct_directory.is_dir():  # pragma: no cover - protects against a filesystem race.
        raise ValueError("crawler output path must be a directory")
    return direct_directory


def _reject_symlinked_ancestors(path: Path) -> None:
    candidate = path
    while True:
        if candidate.is_symlink():
            raise ValueError("crawler output directories must not be symlinks")
        if candidate == candidate.parent:
            return
        candidate = candidate.parent


def _reject_symlink(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("crawler output files must not be symlinks")


def _atomic_write(path: Path, content: bytes) -> None:
    """Write one output file atomically without following an existing link."""

    _reject_symlink(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _reject_symlinked_ancestors(path.parent)
        _reject_symlink(path)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
