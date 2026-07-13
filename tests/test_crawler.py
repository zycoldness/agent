"""Tests for the governed public-source crawler."""

import hashlib
import json
import os
from pathlib import Path

import httpx
import pytest

from risk_agent.crawler import (
    MAX_RESPONSE_BYTES,
    Source,
    completion_path_for,
    fetch,
    is_complete_artifact,
    load_manifest,
    validate_source,
)


def test_source_requires_https_and_exact_allowlisted_hostname():
    source = Source(
        url="https://example.gov.cn/case/1",
        allowed_domains=("example.gov.cn",),
    )

    assert validate_source(source) == "https://example.gov.cn/case/1"


@pytest.mark.parametrize(
    "url",
    [
        "http://example.gov.cn/case/1",
        "https://sub.example.gov.cn/case/1",
        "https://user:secret@example.gov.cn/case/1",
        "file:///etc/passwd",
        "https://example.gov.cn:8443/case/1",
        "https://example.gov.cn/profile/alice",
        "https://example.gov.cn/settings",
        "https://example.gov.cn/dashboard",
        "https://localhost/case/1",
        "https://internal.local/case/1",
        "https://127.0.0.1/case/1",
        "https://[::1]/case/1",
        "https://[2001:4860:4860::8888]/case/1",
        "https://127.1/case/1",
        "https://2130706433/case/1",
        "https://0177.0.0.1/case/1",
        "https://0x7f000001/case/1",
        "https://0x7f.0x0.0x0.0x1/case/1",
    ],
)
def test_source_rejects_non_allowlisted_or_dangerous_urls(url: str):
    source = Source(url=url, allowed_domains=("example.gov.cn",))

    with pytest.raises(ValueError, match="allowlisted|HTTPS|credentials|port|user-account|local|IP-literal|canonical"):
        validate_source(source)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.gov.cn/case/1?access_token=secret-value",
        "https://example.gov.cn/case/1?SESSION_ID=secret-value",
        "https://example.gov.cn/case/1?api-key=secret-value",
    ],
)
def test_source_rejects_credential_bearing_query_without_echoing_it(url: str):
    source = Source(url=url, allowed_domains=("example.gov.cn",))

    with pytest.raises(ValueError, match="credential-bearing query") as error:
        validate_source(source)

    assert "secret-value" not in str(error.value)


def test_source_allows_ordinary_public_query_parameters():
    source = Source(
        url="https://example.gov.cn/public-case.html?q=advertising&page=2",
        allowed_domains=("example.gov.cn",),
    )

    assert validate_source(source) == source.url


@pytest.mark.parametrize(
    "url",
    [
        "https://example.gov.cn/%6cogin",
        "https://example.gov.cn/%75ser/alice",
        "https://example.gov.cn/%61ccount",
        "https://example.gov.cn/%50%72%6f%46%69%6c%65/alice",
        "https://example.gov.cn/foo%EF%BC%8Flogin",
    ],
)
def test_source_normalizes_encoded_account_route_segments(url: str):
    source = Source(url=url, allowed_domains=("example.gov.cn",))

    with pytest.raises(ValueError, match="user-account"):
        validate_source(source)


def test_fetch_obeys_robots_and_writes_raw_bytes_and_metadata(tmp_path):
    requested: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=b"<html>public record</html>")

    source = Source(
        url="https://example.gov.cn/case/1",
        allowed_domains=("example.gov.cn",),
        source_type="public_case",
        license="public notice",
        terms_review="reviewed",
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        raw_path = fetch(source, tmp_path, client=client, retrieved_at="2026-07-13T00:00:00Z")

    digest = hashlib.sha256(source.url.encode("utf-8")).hexdigest()
    assert raw_path == tmp_path / f"{digest}.raw"
    assert raw_path.read_bytes() == b"<html>public record</html>"
    assert [str(request.url) for request in requested] == [
        "https://example.gov.cn/robots.txt",
        source.url,
    ]
    assert all(request.headers["user-agent"] for request in requested)

    metadata = json.loads((tmp_path / "metadata" / f"{digest}.json").read_text(encoding="utf-8"))
    assert metadata == {
        "content_sha256": hashlib.sha256(b"<html>public record</html>").hexdigest(),
        "license": "public notice",
        "raw_path": f"{digest}.raw",
        "retrieved_at": "2026-07-13T00:00:00Z",
        "source_type": "public_case",
        "terms_review": "reviewed",
        "url": source.url,
    }
    assert is_complete_artifact(source, tmp_path)
    assert completion_path_for(source, tmp_path).is_file()


def test_fetch_rejects_robots_denial_before_requesting_source(tmp_path):
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return httpx.Response(200, text="User-agent: *\nDisallow: /case/\n")

    source = Source(url="https://example.gov.cn/case/1", allowed_domains=("example.gov.cn",))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PermissionError, match="robots"):
            fetch(source, tmp_path, client=client)

    assert requested_paths == ["/robots.txt"]


def test_fetch_rejects_redirects_and_oversized_responses(tmp_path):
    source = Source(url="https://example.gov.cn/case/1", allowed_domains=("example.gov.cn",))

    def redirect_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(302, headers={"location": "https://elsewhere.invalid"})

    with httpx.Client(transport=httpx.MockTransport(redirect_handler)) as client:
        with pytest.raises(ValueError, match="redirect"):
            fetch(source, tmp_path, client=client)

    def large_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=b"x" * (MAX_RESPONSE_BYTES + 1))

    with httpx.Client(transport=httpx.MockTransport(large_handler)) as client:
        with pytest.raises(ValueError, match="size limit"):
            fetch(source, tmp_path, client=client)


def test_load_manifest_validates_all_sources_before_fetch(tmp_path):
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        """\
allowed_domains:
  - example.gov.cn
sources:
  - url: https://example.gov.cn/public-case.html
    source_type: public_case
    license: public notice
    terms_review: reviewed
  - https://attacker.invalid/not-allowed
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="allowlisted"):
        load_manifest(manifest)


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink privileges are not reliably available in CI")
def test_fetch_rejects_symlinked_output_paths_before_any_network_request(tmp_path):
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(real_output, target_is_directory=True)
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, text="User-agent: *\nAllow: /\n")

    source = Source(url="https://example.gov.cn/case/1", allowed_domains=("example.gov.cn",))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="symlink"):
            fetch(source, linked_output, client=client)

    assert not called


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink privileges are not reliably available in CI")
@pytest.mark.parametrize("target_kind", ["raw", "metadata"])
def test_fetch_rejects_preexisting_symlinked_raw_or_metadata_file(tmp_path, target_kind: str):
    source = Source(url="https://example.gov.cn/case/1", allowed_domains=("example.gov.cn",))
    raw_target = tmp_path / "sentinel"
    raw_target.write_text("do not overwrite", encoding="utf-8")
    digest = hashlib.sha256(source.url.encode("utf-8")).hexdigest()
    linked_path = tmp_path / f"{digest}.raw"
    if target_kind == "metadata":
        linked_path = tmp_path / "metadata" / f"{digest}.json"
        linked_path.parent.mkdir()
    linked_path.symlink_to(raw_target)

    with pytest.raises(ValueError, match="symlink"):
        fetch(source, tmp_path)
    assert raw_target.read_text(encoding="utf-8") == "do not overwrite"


def test_second_commit_failure_never_exposes_a_completed_artifact(tmp_path, monkeypatch):
    source = Source(url="https://example.gov.cn/case/1", allowed_domains=("example.gov.cn",))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=b"raw document")

    from risk_agent import crawler

    original_replace = crawler.os.replace
    calls = 0

    def fail_metadata_replace(source_path: Path, target_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected metadata commit failure")
        original_replace(source_path, target_path)

    monkeypatch.setattr(crawler.os, "replace", fail_metadata_replace)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OSError, match="injected metadata"):
            fetch(source, tmp_path, client=client)

    assert not completion_path_for(source, tmp_path).exists()
    assert not is_complete_artifact(source, tmp_path)
