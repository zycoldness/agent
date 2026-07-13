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
    metadata_path_for,
    read_complete_artifact,
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
        license_review_status="approved",
        terms_review_status="approved",
        content_review_status="approved",
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
        "license_review_status": "approved",
        "terms_review_status": "approved",
        "content_review_status": "approved",
        "raw_path": f"{digest}.raw",
        "retrieved_at": "2026-07-13T00:00:00Z",
        "source_type": "public_case",
        "url": source.url,
    }
    assert is_complete_artifact(source, tmp_path)
    assert completion_path_for(source, tmp_path).is_file()
    body, verified_metadata = read_complete_artifact(source, tmp_path)
    assert body == b"<html>public record</html>"
    assert verified_metadata["content_review_status"] == "approved"


def test_complete_artifact_reader_enforces_file_and_total_byte_limits(tmp_path):
    source = Source(
        url="https://example.gov.cn/case/1",
        allowed_domains=("example.gov.cn",),
        source_type="public_case",
        license="RIGHTS-REVIEW-REQUIRED",
        license_review_status="pending",
        terms_review_status="pending",
        content_review_status="pending",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=b"bounded raw artifact")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetch(source, tmp_path, client=client, retrieved_at="2026-07-13T00:00:00Z")

    with pytest.raises(ValueError, match="byte limit"):
        read_complete_artifact(source, tmp_path, max_file_bytes=8, max_total_bytes=1024)
    with pytest.raises(ValueError, match="total byte limit"):
        read_complete_artifact(source, tmp_path, max_file_bytes=1024, max_total_bytes=32)


def _rewrite_metadata_and_marker(source: Source, output_dir: Path, metadata_bytes: bytes) -> None:
    digest = hashlib.sha256(source.url.encode("utf-8")).hexdigest()
    metadata_path = output_dir / "metadata" / f"{digest}.json"
    marker_path = completion_path_for(source, output_dir)
    metadata_path.write_bytes(metadata_bytes)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["metadata_sha256"] = hashlib.sha256(metadata_bytes).hexdigest()
    marker_path.write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_complete_reader_binds_source_governance_to_hash_bound_metadata(tmp_path):
    source = Source(
        url="https://example.gov.cn/case/1",
        allowed_domains=("example.gov.cn",),
        source_type="public_case",
        license="PUBLIC-NOTICE",
        license_review_status="approved",
        terms_review_status="approved",
        content_review_status="approved",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=b"governed body")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetch(source, tmp_path, client=client, retrieved_at="2026-07-13T00:00:00Z")

    mismatched = Source(
        url=source.url,
        allowed_domains=source.allowed_domains,
        source_type="different_type",
        license=source.license,
        license_review_status=source.license_review_status,
        terms_review_status=source.terms_review_status,
        content_review_status=source.content_review_status,
    )
    with pytest.raises(ValueError, match="governance.*mismatch"):
        read_complete_artifact(mismatched, tmp_path)
    assert not is_complete_artifact(mismatched, tmp_path)


def test_complete_reader_rejects_extra_and_duplicate_metadata_keys(tmp_path):
    source = Source(
        url="https://example.gov.cn/case/1",
        allowed_domains=("example.gov.cn",),
        source_type="public_case",
        license="PUBLIC-NOTICE",
        license_review_status="pending",
        terms_review_status="pending",
        content_review_status="pending",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=b"governed body")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetch(source, tmp_path, client=client, retrieved_at="2026-07-13T00:00:00Z")

    metadata_path = metadata_path_for(source, tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["unexpected"] = "not allowed"
    _rewrite_metadata_and_marker(
        source,
        tmp_path,
        (json.dumps(metadata, sort_keys=True) + "\n").encode("utf-8"),
    )
    with pytest.raises(ValueError, match="metadata schema"):
        read_complete_artifact(source, tmp_path)

    metadata.pop("unexpected")
    normal = json.dumps(metadata, sort_keys=True)
    duplicate = normal.replace(
        '"url": ',
        f'"url": {json.dumps(source.url)}, "url": ',
        1,
    )
    _rewrite_metadata_and_marker(source, tmp_path, (duplicate + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        read_complete_artifact(source, tmp_path)


def test_complete_reader_rejects_extra_and_duplicate_marker_keys(tmp_path):
    source = Source(url="https://example.gov.cn/case/1", allowed_domains=("example.gov.cn",))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=b"governed body")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetch(source, tmp_path, client=client, retrieved_at="2026-07-13T00:00:00Z")

    marker_path = completion_path_for(source, tmp_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["unexpected"] = "not allowed"
    marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="completion marker.*schema"):
        read_complete_artifact(source, tmp_path)

    marker.pop("unexpected")
    normal = json.dumps(marker, sort_keys=True)
    duplicate = normal.replace(
        '"raw_path": ',
        f'"raw_path": {json.dumps(marker["raw_path"])}, "raw_path": ',
        1,
    )
    marker_path.write_text(duplicate + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        read_complete_artifact(source, tmp_path)


def test_read_budget_does_not_use_a_stat_then_open_path_race(tmp_path, monkeypatch):
    from risk_agent.crawler import ReadBudget

    target = tmp_path / "target.raw"
    replacement = tmp_path / "replacement.raw"
    backup = tmp_path / "backup.raw"
    target.write_bytes(b"original")
    replacement.write_bytes(b"replacement")
    original_stat = Path.stat
    raced = False

    def racing_stat(path: Path, *args, **kwargs):
        nonlocal raced
        result = original_stat(path, *args, **kwargs)
        if path == target and not raced:
            raced = True
            target.rename(backup)
            replacement.rename(target)
        return result

    monkeypatch.setattr(Path, "stat", racing_stat)
    consumed = ReadBudget(1024, 1024).read(target)

    assert consumed == b"original"
    assert not raced


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX dir_fd/openat semantics")
def test_read_budget_keeps_trusted_root_fd_when_root_path_is_replaced(tmp_path, monkeypatch):
    from risk_agent import crawler
    from risk_agent.crawler import ReadBudget

    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    (trusted_root / "nested").mkdir()
    target = trusted_root / "nested" / "artifact.raw"
    target.write_bytes(b"original")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "nested").mkdir()
    (replacement / "nested" / "artifact.raw").write_bytes(b"competitor")
    displaced = tmp_path / "displaced"
    original_open = crawler.os.open
    swapped = False

    def swap_after_root_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if Path(path) == trusted_root and kwargs.get("dir_fd") is None:
            descriptor = original_open(path, flags, *args, **kwargs)
            trusted_root.rename(displaced)
            replacement.rename(trusted_root)
            swapped = True
            return descriptor
        if Path(path) == target and not swapped:
            trusted_root.rename(displaced)
            replacement.rename(trusted_root)
            swapped = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(crawler.os, "open", swap_after_root_open)

    assert ReadBudget(1024, 1024).read(target, trusted_root=trusted_root) == b"original"
    assert swapped


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX symlink and openat semantics")
def test_read_budget_rejects_symlinked_intermediate_component(tmp_path):
    from risk_agent.crawler import ReadBudget

    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "artifact.raw").write_bytes(b"outside")
    (trusted_root / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="regular non-symlink|directory"):
        ReadBudget(1024, 1024).read(
            trusted_root / "nested" / "artifact.raw",
            trusted_root=trusted_root,
        )


@pytest.mark.skipif(os.name == "nt", reason="named pipes use different Windows APIs")
def test_read_budget_rejects_fifo_without_blocking(tmp_path):
    from risk_agent.crawler import ReadBudget

    fifo = tmp_path / "artifact.fifo"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="regular non-symlink"):
        ReadBudget(1024, 1024).read(fifo)


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


def test_load_manifest_rejects_duplicate_yaml_keys_and_source_urls(tmp_path):
    duplicate_key = tmp_path / "duplicate-key.yaml"
    duplicate_key.write_text(
        "allowed_domains:\n  - example.gov.cn\nsources:\n  - url: https://example.gov.cn/a\n    license: A\n    license: B\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_manifest(duplicate_key)

    duplicate_source = tmp_path / "duplicate-source.yaml"
    duplicate_source.write_text(
        "allowed_domains:\n  - example.gov.cn\nsources:\n  - https://example.gov.cn/a\n  - https://example.gov.cn/a\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate source URL"):
        load_manifest(duplicate_source)


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


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink privileges are not reliably available in CI")
def test_complete_check_rejects_symlinked_output_or_metadata_ancestors(tmp_path):
    source = Source(url="https://example.gov.cn/case/1", allowed_domains=("example.gov.cn",))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=b"raw document")

    output_dir = tmp_path / "output"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetch(source, output_dir, client=client)
    assert is_complete_artifact(source, output_dir)

    real_metadata = tmp_path / "real-metadata"
    (output_dir / "metadata").rename(real_metadata)
    (output_dir / "metadata").symlink_to(real_metadata, target_is_directory=True)
    assert not is_complete_artifact(source, output_dir)

    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(output_dir, target_is_directory=True)
    assert not is_complete_artifact(source, linked_output)
