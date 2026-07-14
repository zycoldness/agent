"""Governed open-source seed adapters for SingGuard synthesis."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field


_ALLOWED_ENABLED_LICENSES = {
    "Apache-2.0",
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-NC-4.0",
    "ODC-BY-1.0",
}
_DOWNLOAD_LIMIT = 50 * 1024 * 1024
_TEXT_LIMIT = 5_000


class SourceSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    dataset_url: str = Field(min_length=1)
    artifact_url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    usage_scope: str = Field(min_length=1)
    source_role: str = Field(min_length=1)
    adapter: str = Field(min_length=1)
    enabled: bool


class SeedRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    provenance_url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    usage_scope: str = Field(min_length=1)
    source_role: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=_TEXT_LIMIT)
    source_label: str | None = None
    content_hash: str = Field(min_length=64, max_length=64)
    retrieved_at: str = Field(min_length=1)
    adapter_version: str = "v1"


def load_source_catalog(path: Path) -> tuple[SourceSpec, ...]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        raise ValueError("cannot read SingGuard source catalog") from None
    if not isinstance(payload, dict) or payload.get("version") != "singguard-sources-v1":
        raise ValueError("unsupported SingGuard source catalog")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("SingGuard source catalog must contain sources")
    sources = tuple(SourceSpec.model_validate(item) for item in raw_sources)
    if len({source.source for source in sources}) != len(sources):
        raise ValueError("SingGuard source catalog contains duplicate source names")
    return sources


def _download(client: httpx.Client, url: str) -> bytes:
    payload = bytearray()
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                payload.extend(chunk)
                if len(payload) > _DOWNLOAD_LIMIT:
                    raise ValueError("SingGuard seed artifact exceeds the byte limit")
    except httpx.HTTPError:
        raise ValueError("cannot download SingGuard seed artifact") from None
    return bytes(payload)


def _safe_members(payload: bytes) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members: dict[str, bytes] = {}
            for info in archive.infolist():
                path = PurePosixPath(info.filename.replace("\\", "/"))
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("unsafe ZIP member in SingGuard seed artifact")
                if info.is_dir():
                    continue
                members[info.filename] = archive.read(info)
            return members
    except zipfile.BadZipFile:
        raise ValueError("invalid SingGuard seed ZIP artifact") from None


def _decode(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("cannot decode SingGuard seed text")


def _normalize_text(value: str) -> str:
    text = re.sub(
        r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[EMAIL]",
        value,
    )
    text = re.sub(r"(?i)https?://\S+", "[URL]", text)
    text = re.sub(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)", "[PHONE]", text)
    text = " ".join(text.split())
    if not text or len(text) > _TEXT_LIMIT:
        raise ValueError("SingGuard seed text is empty or oversized")
    return text


def _record(
    spec: SourceSpec,
    *,
    source_id: str,
    text: str,
    source_label: str | None,
    retrieved_at: str,
) -> SeedRecord:
    normalized = _normalize_text(text)
    return SeedRecord(
        source=spec.source,
        source_id=source_id,
        provenance_url=spec.dataset_url,
        license=spec.license,
        usage_scope=spec.usage_scope,
        source_role=spec.source_role,
        text=normalized,
        source_label=source_label,
        content_hash=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        retrieved_at=retrieved_at,
    )


def _parse_sms(spec: SourceSpec, payload: bytes, retrieved_at: str) -> list[SeedRecord]:
    members = _safe_members(payload)
    match = next(
        (data for name, data in members.items() if PurePosixPath(name).name == "SMSSpamCollection"),
        None,
    )
    if match is None:
        raise ValueError("SMS seed artifact lacks SMSSpamCollection")
    records: list[SeedRecord] = []
    for number, line in enumerate(_decode(match).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            label, text = line.split("\t", 1)
        except ValueError:
            raise ValueError("invalid SMS seed row") from None
        records.append(
            _record(
                spec,
                source_id=f"sms-{number:06d}",
                text=text,
                source_label=label.strip().lower(),
                retrieved_at=retrieved_at,
            )
        )
    return records


def _parse_youtube(spec: SourceSpec, payload: bytes, retrieved_at: str) -> list[SeedRecord]:
    members = _safe_members(payload)
    csv_members = [
        (name, data)
        for name, data in members.items()
        if PurePosixPath(name).name.lower().startswith("youtube")
        and PurePosixPath(name).suffix.lower() == ".csv"
    ]
    if not csv_members:
        raise ValueError("YouTube seed artifact lacks expected CSV files")
    records: list[SeedRecord] = []
    for filename, data in sorted(csv_members):
        reader = csv.DictReader(io.StringIO(_decode(data), newline=""))
        required = {"COMMENT_ID", "CONTENT", "CLASS"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("invalid YouTube seed CSV columns")
        for number, row in enumerate(reader, start=1):
            comment_id = (row.get("COMMENT_ID") or "").strip() or f"row-{number}"
            source_id = f"{PurePosixPath(filename).stem}:{number:06d}:{comment_id}"
            label = "spam" if (row.get("CLASS") or "").strip() == "1" else "ham"
            records.append(
                _record(
                    spec,
                    source_id=source_id,
                    text=row.get("CONTENT") or "",
                    source_label=label,
                    retrieved_at=retrieved_at,
                )
            )
    return records


def _write_jsonl_atomically(path: Path, rows: list[dict[str, object]]) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
                handle.write("\n")
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _fetch_with_client(
    specs: tuple[SourceSpec, ...],
    client: httpx.Client,
    retrieved_at: str,
) -> list[SeedRecord]:
    records: list[SeedRecord] = []
    for spec in specs:
        if not spec.enabled:
            continue
        if spec.license not in _ALLOWED_ENABLED_LICENSES:
            raise ValueError("enabled SingGuard sources require an allowlisted license")
        payload = _download(client, spec.artifact_url)
        if spec.adapter == "uci_sms_zip":
            records.extend(_parse_sms(spec, payload, retrieved_at))
        elif spec.adapter == "uci_youtube_zip":
            records.extend(_parse_youtube(spec, payload, retrieved_at))
        else:
            raise ValueError("enabled SingGuard source has an unsupported adapter")
    keys = [(record.source, record.source_id) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("SingGuard seeds contain duplicate source IDs")
    return records


def fetch_configured_seeds(
    catalog_path: Path,
    output_dir: Path,
    *,
    client: httpx.Client | None = None,
    retrieved_at: str | None = None,
) -> dict[str, object]:
    """Fetch enabled governed sources and atomically write normalized seeds."""

    specs = load_source_catalog(catalog_path)
    if output_dir.exists():
        raise ValueError("output directory must not exist")
    retrieved_at = retrieved_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if client is None:
        with httpx.Client(timeout=60, follow_redirects=True) as owned_client:
            records = _fetch_with_client(specs, owned_client, retrieved_at)
    else:
        records = _fetch_with_client(specs, client, retrieved_at)
    rows = [record.model_dump(mode="json") for record in records]
    source_counts = dict(sorted(Counter(record.source for record in records).items()))
    license_counts = dict(sorted(Counter(record.license for record in records).items()))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    seeds_path = output_dir / "seeds.jsonl"
    _write_jsonl_atomically(seeds_path, rows)
    manifest: dict[str, object] = {
        "schema": "singguard-seeds-v1",
        "retrieved_at": retrieved_at,
        "record_count": len(records),
        "source_counts": source_counts,
        "license_counts": license_counts,
        "usage_scope": "research_only",
        "seeds_sha256": hashlib.sha256(seeds_path.read_bytes()).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return manifest
