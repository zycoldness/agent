"""Governed open-source seed adapters for SingGuard synthesis."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
import zipfile
from collections.abc import Callable, Iterable, Mapping
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_ALLOWED_ENABLED_LICENSES = {
    "Apache-2.0",
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-NC-4.0",
    "ODC-BY-1.0",
}
_DOWNLOAD_LIMIT = 50 * 1024 * 1024
_TEXT_LIMIT = 5_000
_ARCHIVE_ADAPTERS = {"uci_sms_zip", "uci_youtube_zip"}
_EXTERNAL_ADAPTERS = {"hf_aegis_v2", "hf_civil_comments", "esci_query_parquet"}
_PIN_PATTERN = re.compile(r"[0-9a-f]{40}")


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
    dataset_id: str | None = None
    revision: str | None = None
    split: str | None = None
    max_records: int = Field(default=5000, ge=1, le=100_000)

    @field_validator("dataset_id", "revision", "split")
    @classmethod
    def _optional_string_is_nonblank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("optional source strings must be nonblank")
        return value

    @model_validator(mode="after")
    def _validate_external_adapter(self) -> SourceSpec:
        if self.adapter not in _EXTERNAL_ADAPTERS:
            return self
        if self.revision is None or _PIN_PATTERN.fullmatch(self.revision) is None:
            raise ValueError("external sources require a pinned lowercase 40-hex revision")
        if self.split != "train":
            raise ValueError("external sources require split train")
        if self.adapter in {"hf_aegis_v2", "hf_civil_comments"}:
            if self.dataset_id is None:
                raise ValueError("Hugging Face sources require dataset_id")
        elif self.dataset_id != "parquet":
            raise ValueError("ESCI requires dataset_id parquet")
        else:
            artifact_url = urlsplit(self.artifact_url)
            if artifact_url.scheme != "https" or not artifact_url.netloc:
                raise ValueError("ESCI requires an HTTPS artifact URL")
        return self


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


def _required_string(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ValueError(f"external source row requires scalar string {field}")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"external source row requires nonblank {field}")
    return normalized


def _optional_string(row: Mapping[str, object], field: str) -> str | None:
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"external source row {field} must be a scalar string")
    normalized = " ".join(value.split())
    return normalized or None


def _scalar_id(value: object) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return str(value)
    return None


def _mapping_row(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("external source loader rows must be mappings")
    return value


def _parse_aegis(
    spec: SourceSpec,
    upstream: Iterable[Mapping[str, object]],
    retrieved_at: str,
) -> list[SeedRecord]:
    records: list[SeedRecord] = []
    for raw_row in upstream:
        row = _mapping_row(raw_row)
        prompt = _required_string(row, "prompt")
        if prompt == "REDACTED":
            continue
        source_id = _scalar_id(row.get("id"))
        if source_id is None:
            raise ValueError("Aegis row requires a nonblank scalar id")
        response = _optional_string(row, "response")
        prompt_label = _optional_string(row, "prompt_label")
        text = prompt
        if response is not None:
            text = json.dumps(
                {
                    "query": _normalize_text(prompt),
                    "response": _normalize_text(response),
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        records.append(
            _record(
                spec,
                source_id=source_id,
                text=text,
                source_label=prompt_label.lower() if prompt_label is not None else None,
                retrieved_at=retrieved_at,
            )
        )
        if len(records) >= spec.max_records:
            return records
    return records


def _parse_civil_comments(
    spec: SourceSpec,
    upstream: Iterable[Mapping[str, object]],
    retrieved_at: str,
) -> list[SeedRecord]:
    records: list[SeedRecord] = []
    for ordinal, raw_row in enumerate(upstream, start=1):
        row = _mapping_row(raw_row)
        text = _required_string(row, "text")
        toxicity = row.get("toxicity")
        if (
            isinstance(toxicity, bool)
            or not isinstance(toxicity, (int, float))
            or not math.isfinite(toxicity)
            or not 0 <= toxicity <= 1
        ):
            raise ValueError("Civil Comments toxicity must be finite numeric in [0,1]")
        source_id = _scalar_id(row.get("id")) or f"civil-row-{ordinal:09d}"
        records.append(
            _record(
                spec,
                source_id=source_id,
                text=text,
                source_label="toxic" if toxicity >= 0.5 else "non_toxic",
                retrieved_at=retrieved_at,
            )
        )
        if len(records) >= spec.max_records:
            return records
    return records


def _parse_esci_queries(
    spec: SourceSpec,
    upstream: Iterable[Mapping[str, object]],
    retrieved_at: str,
) -> list[SeedRecord]:
    records: list[SeedRecord] = []
    canonical_queries: set[str] = set()
    for raw_row in upstream:
        row = _mapping_row(raw_row)
        locale_value = row.get("product_locale")
        split_value = row.get("split")
        if not isinstance(locale_value, str) or not isinstance(split_value, str):
            raise ValueError("ESCI row locale and split must be scalar strings")
        if locale_value.strip().casefold() != "us" or split_value.strip().casefold() != "train":
            continue
        query = _required_string(row, "query")
        canonical = " ".join(unicodedata.normalize("NFKC", query).split()).casefold()
        if canonical in canonical_queries:
            continue
        canonical_queries.add(canonical)
        query_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        records.append(
            _record(
                spec,
                source_id=f"esci-query-{query_hash[:16]}",
                text=query,
                source_label=None,
                retrieved_at=retrieved_at,
            )
        )
        if len(records) >= spec.max_records:
            return records
    return records


def _default_dataset_loader(spec: SourceSpec) -> Iterable[Mapping[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("external sources require pip install -e '.[sources]'") from None

    if spec.adapter in {"hf_aegis_v2", "hf_civil_comments"}:
        return load_dataset(
            spec.dataset_id,
            split="train",
            revision=spec.revision,
            streaming=True,
        )
    if spec.adapter == "esci_query_parquet":
        return load_dataset(
            "parquet",
            data_files={"train": spec.artifact_url},
            split="train",
            streaming=True,
        )
    raise ValueError("default dataset loader received a non-external adapter")


def _parse_external(
    spec: SourceSpec,
    upstream: Iterable[Mapping[str, object]],
    retrieved_at: str,
) -> list[SeedRecord]:
    if spec.adapter == "hf_aegis_v2":
        return _parse_aegis(spec, upstream, retrieved_at)
    if spec.adapter == "hf_civil_comments":
        return _parse_civil_comments(spec, upstream, retrieved_at)
    if spec.adapter == "esci_query_parquet":
        return _parse_esci_queries(spec, upstream, retrieved_at)
    raise ValueError("external loader received an unsupported adapter")


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
        if len(records) >= spec.max_records:
            return records
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
            if len(records) >= spec.max_records:
                return records
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


def _fetch_selected(
    specs: tuple[SourceSpec, ...],
    client: httpx.Client | None,
    retrieved_at: str,
    dataset_loader: Callable[[SourceSpec], Iterable[Mapping[str, object]]],
) -> list[SeedRecord]:
    records: list[SeedRecord] = []
    for spec in specs:
        if spec.adapter in _ARCHIVE_ADAPTERS:
            if client is None:
                raise RuntimeError("archive sources require an HTTP client")
            payload = _download(client, spec.artifact_url)
            if spec.adapter == "uci_sms_zip":
                records.extend(_parse_sms(spec, payload, retrieved_at))
            else:
                records.extend(_parse_youtube(spec, payload, retrieved_at))
        elif spec.adapter in _EXTERNAL_ADAPTERS:
            upstream = dataset_loader(spec)
            records.extend(_parse_external(spec, upstream, retrieved_at))
        else:
            raise ValueError("enabled SingGuard source has an unsupported adapter")
    keys = [(record.source, record.source_id) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("SingGuard seeds contain duplicate source IDs")
    unique_records: list[SeedRecord] = []
    content_hashes: set[str] = set()
    for record in records:
        if record.content_hash in content_hashes:
            continue
        content_hashes.add(record.content_hash)
        unique_records.append(record)
    return unique_records


def _select_sources(
    specs: tuple[SourceSpec, ...],
    enabled_sources: tuple[str, ...] | None,
) -> tuple[SourceSpec, ...]:
    if enabled_sources is None:
        return tuple(spec for spec in specs if spec.enabled)
    if len(enabled_sources) != len(set(enabled_sources)):
        raise ValueError("enabled_sources contains duplicate source names")
    by_name = {spec.source: spec for spec in specs}
    unknown = [name for name in enabled_sources if name not in by_name]
    if unknown:
        raise ValueError(f"enabled_sources contains unknown source: {unknown[0]}")
    disabled = [name for name in enabled_sources if not by_name[name].enabled]
    if disabled:
        raise ValueError(f"enabled_sources contains disabled source: {disabled[0]}")
    return tuple(by_name[name] for name in enabled_sources)


def _source_snapshots(specs: tuple[SourceSpec, ...]) -> list[dict[str, object]]:
    return [
        {
            "source": spec.source,
            "adapter": spec.adapter,
            "dataset_id": spec.dataset_id,
            "revision": spec.revision,
            "split": spec.split,
            "artifact_url": spec.artifact_url,
            "max_records": spec.max_records,
            "license": spec.license,
        }
        for spec in specs
    ]


def fetch_configured_seeds(
    catalog_path: Path,
    output_dir: Path,
    *,
    client: httpx.Client | None = None,
    retrieved_at: str | None = None,
    dataset_loader: Callable[[SourceSpec], Iterable[Mapping[str, object]]] | None = None,
    enabled_sources: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Fetch enabled governed sources and atomically write normalized seeds."""

    specs = load_source_catalog(catalog_path)
    if output_dir.exists():
        raise ValueError("output directory must not exist")
    selected_specs = _select_sources(specs, enabled_sources)
    if any(spec.license not in _ALLOWED_ENABLED_LICENSES for spec in selected_specs):
        raise ValueError("enabled SingGuard sources require an allowlisted license")
    retrieved_at = retrieved_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    loader = dataset_loader or _default_dataset_loader
    needs_http = any(spec.adapter in _ARCHIVE_ADAPTERS for spec in selected_specs)
    if client is None and needs_http:
        with httpx.Client(timeout=60, follow_redirects=True) as owned_client:
            records = _fetch_selected(selected_specs, owned_client, retrieved_at, loader)
    else:
        records = _fetch_selected(selected_specs, client, retrieved_at, loader)
    rows = [record.model_dump(mode="json") for record in records]
    source_counts = dict(sorted(Counter(record.source for record in records).items()))
    license_counts = dict(sorted(Counter(record.license for record in records).items()))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".tmp", dir=output_dir.parent)
    )
    try:
        seeds_path = staging_dir / "seeds.jsonl"
        _write_jsonl_atomically(seeds_path, rows)
        manifest: dict[str, object] = {
            "schema": "singguard-seeds-v1",
            "retrieved_at": retrieved_at,
            "record_count": len(records),
            "source_counts": source_counts,
            "license_counts": license_counts,
            "usage_scope": "research_only",
            "seeds_sha256": hashlib.sha256(seeds_path.read_bytes()).hexdigest(),
            "source_snapshots": _source_snapshots(selected_specs),
        }
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(staging_dir, output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    return manifest
