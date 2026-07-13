"""Governed normalization of public benchmark assets and regulatory cases."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Literal
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, field_validator
import yaml

from risk_agent.crawler import (
    Source,
    is_complete_artifact,
    metadata_path_for,
    raw_path_for,
)


MAX_IMPORT_RECORDS = 50_000
_PLACEHOLDER_LICENSE_IDS = frozenset({"", "unknown", "not_reviewed", "not-reviewed", "tbd"})
_REVIEW_STATUSES = frozenset({"approved", "review_required", "rejected"})


class PublicAsset(BaseModel):
    """A public source item without an inferred risk label or Oracle."""

    model_config = ConfigDict(frozen=True)

    source_dataset: str
    source_item_id: str
    scenario: str
    prompt: str
    media_paths: tuple[str, ...]
    media_status: Literal["available", "missing"]
    content_hash: str
    data_classification: Literal["public"] = "public"
    license_id: str
    usage_scope: Literal["smoke_only", "research_only"]
    source_url: str
    retrieved_at: str
    split_group: str

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        return _validate_sha256(value)

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        return _validate_public_url(value)

    @field_validator("retrieved_at")
    @classmethod
    def validate_retrieved_at(cls, value: str) -> str:
        return _validate_timestamp(value)

    @field_validator("media_paths")
    @classmethod
    def validate_media_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("media_paths must not be empty")
        for item in value:
            path = PurePosixPath(item)
            if not item or "\\" in item or path.is_absolute() or ".." in path.parts:
                raise ValueError("media_paths must contain safe repo-relative POSIX paths")
        return value


class SanitizedCase(BaseModel):
    """A factual public-case snippet with source and release provenance."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    text: str
    source_dataset: str
    source_item_id: str
    source_url: str
    retrieved_at: str
    source_content_hash: str
    content_hash: str
    data_classification: Literal["public"] = "public"
    license_id: str
    usage_scope: Literal["smoke_only", "research_only"]
    license_review_status: Literal["approved", "review_required", "rejected"]
    terms_review_status: Literal["approved", "review_required", "rejected"]
    publication_status: Literal["published", "quarantined"]

    @field_validator("content_hash", "source_content_hash", "source_item_id")
    @classmethod
    def validate_hash_fields(cls, value: str) -> str:
        return _validate_sha256(value)

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        return _validate_public_url(value)

    @field_validator("retrieved_at")
    @classmethod
    def validate_retrieved_at(cls, value: str) -> str:
        return _validate_timestamp(value)


def import_regulatory_html(
    html: bytes | str,
    *,
    source_dataset: str,
    source_url: str,
    retrieved_at: str,
    license_id: str,
    usage_scope: Literal["smoke_only", "research_only"],
    license_review_status: Literal["approved", "review_required", "rejected"],
    terms_review_status: Literal["approved", "review_required", "rejected"],
    max_records: int,
) -> tuple[SanitizedCase, ...]:
    """Normalize factual snippets from one already-approved local HTML input."""

    _validate_import_metadata(
        source_dataset=source_dataset,
        source_url=source_url,
        retrieved_at=retrieved_at,
        license_id=license_id,
        usage_scope=usage_scope,
        license_review_status=license_review_status,
        terms_review_status=terms_review_status,
        max_records=max_records,
    )
    raw_bytes = html if isinstance(html, bytes) else html.encode("utf-8")
    source_content_hash = hashlib.sha256(raw_bytes).hexdigest()
    soup = BeautifulSoup(raw_bytes, "html.parser")
    for unwanted in soup.select("script, style, noscript"):
        unwanted.decompose()
    containers = soup.select("article p, article li") or soup.select("p, li")
    snippets: list[str] = []
    seen_texts: set[str] = set()
    for container in containers:
        text = re.sub(r"\s+", " ", container.get_text(" ", strip=True)).strip()
        if len(text) < 20 or text in seen_texts:
            continue
        seen_texts.add(text)
        snippets.append(text)
        if len(snippets) == max_records:
            break

    publication_status = (
        "published"
        if license_review_status == "approved" and terms_review_status == "approved"
        else "quarantined"
    )
    cases: list[SanitizedCase] = []
    for text in snippets:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        source_item_id = content_hash
        cases.append(
            SanitizedCase(
                case_id=f"public-case-{content_hash[:24]}",
                text=text,
                source_dataset=source_dataset,
                source_item_id=source_item_id,
                source_url=source_url,
                retrieved_at=retrieved_at,
                source_content_hash=source_content_hash,
                content_hash=content_hash,
                license_id=license_id,
                usage_scope=usage_scope,
                license_review_status=license_review_status,
                terms_review_status=terms_review_status,
                publication_status=publication_status,
            )
        )
    return tuple(cases)


def import_regulatory_crawler_artifact(
    source: Source,
    crawler_output_dir: Path,
    *,
    source_dataset: str,
    license_id: str,
    usage_scope: Literal["smoke_only", "research_only"],
    license_review_status: Literal["approved", "review_required", "rejected"],
    terms_review_status: Literal["approved", "review_required", "rejected"],
    max_records: int,
) -> tuple[SanitizedCase, ...]:
    """Import only a crawler artifact whose completion marker and hashes validate."""

    crawler_output_dir = Path(crawler_output_dir)
    if not is_complete_artifact(source, crawler_output_dir):
        raise ValueError("regulatory input is not a complete crawler artifact")
    if source.license != license_id:
        raise ValueError("license_id must match the governed crawler metadata")
    if source.terms_review != terms_review_status:
        raise ValueError("terms_review_status must match the governed crawler metadata")
    metadata = json.loads(metadata_path_for(source, crawler_output_dir).read_text(encoding="utf-8"))
    retrieved_at = metadata.get("retrieved_at")
    if not isinstance(retrieved_at, str):
        raise ValueError("crawler metadata must contain retrieved_at")
    return import_regulatory_html(
        raw_path_for(source, crawler_output_dir).read_bytes(),
        source_dataset=source_dataset,
        source_url=source.url,
        retrieved_at=retrieved_at,
        license_id=license_id,
        usage_scope=usage_scope,
        license_review_status=license_review_status,
        terms_review_status=terms_review_status,
        max_records=max_records,
    )


def run_public_data_import(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Build and atomically publish one normalized public-data seed bundle."""

    config_path = Path(config_path)
    output_dir = Path(output_dir)
    document = _load_public_data_config(config_path)
    max_records = document["max_records"]
    _validate_record_cap(max_records)
    input_paths = _config_input_paths(config_path, document)
    _reject_input_output_overlap(input_paths, output_dir)
    _reject_existing_bundle_outputs(output_dir)

    assets: list[PublicAsset] = []
    mm_config = document.get("mm_safety_bench")
    if mm_config is not None:
        mm = _require_mapping(mm_config, "mm_safety_bench")
        license_id = _required_string(mm, "license_id")
        if license_id != "CC-BY-NC-4.0":
            raise ValueError("MM-SafetyBench license_id must be CC-BY-NC-4.0")
        usage_scope = _required_string(mm, "usage_scope")
        if usage_scope not in {"smoke_only", "research_only"}:
            raise ValueError("MM-SafetyBench usage_scope must be smoke_only or research_only")
        scenario_rows = mm.get("scenario_allowlist")
        if scenario_rows is not None and (
            not isinstance(scenario_rows, list)
            or not all(isinstance(item, str) and item for item in scenario_rows)
        ):
            raise ValueError("scenario_allowlist must be a list of scenario names")
        assets.extend(
            import_mm_safety_bench(
                Path(_required_string(mm, "repo_root")),
                source_url=_required_string(mm, "source_url"),
                retrieved_at=_required_string(mm, "retrieved_at"),
                use_tiny=_optional_bool(mm, "use_tiny", True),
                scenario_allowlist=set(scenario_rows) if scenario_rows is not None else None,
                allow_missing_media=_optional_bool(mm, "allow_missing_media", False),
                license_id=license_id,
                usage_scope=usage_scope,
                max_records=max_records,
            )
        )

    cases: list[SanitizedCase] = []
    regulatory_rows = document.get("regulatory_cases", [])
    if not isinstance(regulatory_rows, list):
        raise ValueError("regulatory_cases must be a list")
    for index, value in enumerate(regulatory_rows):
        row = _require_mapping(value, f"regulatory_cases[{index}]")
        row_limit = row.get("max_records", max_records)
        _validate_record_cap(row_limit)
        common = {
            "source_dataset": _required_string(row, "source_dataset"),
            "license_id": _required_string(row, "license_id"),
            "usage_scope": _required_string(row, "usage_scope"),
            "license_review_status": _required_string(row, "license_review_status"),
            "terms_review_status": _required_string(row, "terms_review_status"),
            "max_records": row_limit,
        }
        input_type = _required_string(row, "input_type")
        if input_type == "local_html":
            html_path = Path(_required_string(row, "local_html_path"))
            cases.extend(
                import_regulatory_html(
                    html_path.read_bytes(),
                    source_url=_required_string(row, "source_url"),
                    retrieved_at=_required_string(row, "retrieved_at"),
                    **common,
                )
            )
        elif input_type == "crawler_artifact":
            source_row = _require_mapping(row.get("source"), f"regulatory_cases[{index}].source")
            allowed_domains = source_row.get("allowed_domains")
            if not isinstance(allowed_domains, list) or not all(
                isinstance(item, str) and item for item in allowed_domains
            ):
                raise ValueError("crawler source allowed_domains must be a list of hostnames")
            source = Source(
                url=_required_string(source_row, "url"),
                allowed_domains=tuple(allowed_domains),
                source_type=_required_string(source_row, "source_type"),
                license=_required_string(source_row, "license"),
                terms_review=_required_string(source_row, "terms_review"),
            )
            cases.extend(
                import_regulatory_crawler_artifact(
                    source,
                    Path(_required_string(row, "crawler_output_dir")),
                    **common,
                )
            )
        else:
            raise ValueError("regulatory input_type must be local_html or crawler_artifact")

    _reject_duplicate_records(assets, cases)
    if len(assets) + len(cases) > max_records:
        raise ValueError("normalized records exceed configured max_records")
    assets.sort(key=lambda item: (item.source_dataset, item.source_item_id))
    cases.sort(key=lambda item: item.case_id)
    report = _build_import_report(assets, cases)
    _publish_public_data_bundle(output_dir, assets, cases, report)
    return report


def import_mm_safety_bench(
    repo_root: Path,
    *,
    source_url: str,
    retrieved_at: str,
    use_tiny: bool = True,
    scenario_allowlist: set[str] | frozenset[str] | None = None,
    allow_missing_media: bool = False,
    license_id: str = "CC-BY-NC-4.0",
    usage_scope: Literal["smoke_only", "research_only"] = "smoke_only",
    max_records: int,
) -> tuple[PublicAsset, ...]:
    """Import a bounded local MM-SafetyBench checkout without downloading media."""

    _validate_record_cap(max_records)
    if license_id != "CC-BY-NC-4.0":
        raise ValueError("MM-SafetyBench license_id must be CC-BY-NC-4.0")
    if usage_scope not in {"smoke_only", "research_only"}:
        raise ValueError("MM-SafetyBench usage_scope must be smoke_only or research_only")
    repo_root = Path(repo_root)
    questions_dir = repo_root / "data" / "processed_questions"
    if not questions_dir.is_dir():
        raise ValueError("MM-SafetyBench questions directory is missing")
    tiny_ids = _load_tiny_ids(repo_root / "TinyVersion_ID_List.json") if use_tiny else None
    question_files = {path.stem: path for path in questions_dir.glob("*.json") if path.is_file()}
    if not question_files:
        raise ValueError("MM-SafetyBench questions directory contains no scenario JSON files")
    selected_scenarios = set(question_files)
    if scenario_allowlist is not None:
        unknown_scenarios = set(scenario_allowlist) - selected_scenarios
        if unknown_scenarios:
            raise ValueError(f"unknown scenario in allowlist: {', '.join(sorted(unknown_scenarios))}")
        selected_scenarios &= set(scenario_allowlist)
    assets: list[PublicAsset] = []
    for scenario in sorted(selected_scenarios):
        questions_path = question_files[scenario]
        rows = _read_json_mapping(questions_path)
        if tiny_ids is not None and scenario not in tiny_ids:
            raise ValueError(f"tiny ID list has no entry for scenario: {scenario}")
        selected_ids = tiny_ids[scenario] if tiny_ids is not None else rows.keys()
        for item_id in sorted((str(item) for item in selected_ids), key=_item_id_sort_key):
            if item_id not in rows:
                raise ValueError(f"tiny ID list references unknown question id {item_id} in {scenario}")
            row = rows[item_id]
            for variant, prompt_field in (
                ("SD", "Rephrased Question(SD)"),
                ("SD_TYPO", "Rephrased Question"),
                ("TYPO", "Rephrased Question"),
            ):
                media_path = Path("data") / "imgs" / scenario / variant / f"{item_id}.jpg"
                absolute_media_path = repo_root / media_path
                if absolute_media_path.is_file():
                    media_status = "available"
                    media_sha256: str | None = hashlib.sha256(absolute_media_path.read_bytes()).hexdigest()
                elif allow_missing_media:
                    media_status = "missing"
                    media_sha256 = None
                else:
                    raise ValueError(f"missing MM-SafetyBench image: {media_path.as_posix()}")
                prompt_value = row.get(prompt_field) if isinstance(row, dict) else None
                if not isinstance(prompt_value, str) or not prompt_value.strip():
                    raise ValueError(f"invalid {prompt_field} for {scenario}:{item_id}")
                prompt = prompt_value.strip()
                source_item_id = f"{scenario}:{item_id}:{variant}"
                content_hash = _canonical_hash(
                    {
                        "source_dataset": "MM-SafetyBench",
                        "source_item_id": source_item_id,
                        "prompt": prompt,
                        "media_path": media_path.as_posix(),
                        "media_sha256": media_sha256,
                    }
                )
                assets.append(
                    PublicAsset(
                        source_dataset="MM-SafetyBench",
                        source_item_id=source_item_id,
                        scenario=scenario,
                        prompt=prompt,
                        media_paths=(media_path.as_posix(),),
                        media_status=media_status,
                        content_hash=content_hash,
                        license_id=license_id,
                        usage_scope=usage_scope,
                        source_url=source_url,
                        retrieved_at=retrieved_at,
                        split_group=f"MM-SafetyBench:{scenario}:{item_id}",
                    )
                )
                if len(assets) == max_records:
                    return tuple(assets)
    return tuple(assets)


def _load_tiny_ids(path: Path) -> dict[str, tuple[str, ...]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["Scenario"]): tuple(str(item) for item in row["Sampled_ID_List"])
        for row in rows
    }


def _read_json_mapping(path: Path) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, dict):
        raise ValueError(f"MM-SafetyBench question file must contain a mapping: {path.name}")
    return rows


def _item_id_sort_key(item_id: str) -> tuple[int, int | str]:
    return (0, int(item_id)) if item_id.isdigit() else (1, item_id)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_public_data_config(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid public data YAML: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("public data config must be a mapping")
    unknown = set(document) - {"version", "max_records", "mm_safety_bench", "regulatory_cases"}
    if unknown:
        raise ValueError(f"unknown public data config fields: {', '.join(sorted(map(str, unknown)))}")
    if document.get("version") != 1:
        raise ValueError("public data config version must be 1")
    if "max_records" not in document:
        raise ValueError("public data config requires max_records")
    return document


def _config_input_paths(config_path: Path, document: Mapping[str, Any]) -> tuple[Path, ...]:
    paths = [config_path]
    mm = document.get("mm_safety_bench")
    if isinstance(mm, Mapping) and isinstance(mm.get("repo_root"), str):
        paths.append(Path(mm["repo_root"]))
    regulatory_rows = document.get("regulatory_cases", [])
    if isinstance(regulatory_rows, list):
        for value in regulatory_rows:
            if not isinstance(value, Mapping):
                continue
            if value.get("input_type") == "local_html" and isinstance(value.get("local_html_path"), str):
                paths.append(Path(value["local_html_path"]))
            if value.get("input_type") == "crawler_artifact" and isinstance(value.get("crawler_output_dir"), str):
                paths.append(Path(value["crawler_output_dir"]))
    return tuple(paths)


def _reject_input_output_overlap(input_paths: tuple[Path, ...], output_dir: Path) -> None:
    output = output_dir.resolve(strict=False)
    for input_path in input_paths:
        source = input_path.resolve(strict=False)
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("public data output must not alias or overlap an input path")


def _reject_existing_bundle_outputs(output_dir: Path) -> None:
    if output_dir.is_symlink():
        raise ValueError("public data output directory must not be a symlink")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("public data output must be a directory")
    for filename in (
        "public_assets.jsonl",
        "sanitized_cases.jsonl",
        "import_report.json",
        "import_manifest.json",
    ):
        if (output_dir / filename).exists() or (output_dir / filename).is_symlink():
            raise FileExistsError(f"public data output already exists: {filename}")


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _required_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_bool(row: Mapping[str, Any], field: str, default: bool) -> bool:
    value = row.get(field, default)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _reject_duplicate_records(
    assets: list[PublicAsset], cases: list[SanitizedCase]
) -> None:
    asset_keys = [(item.source_dataset, item.source_item_id) for item in assets]
    if len(set(asset_keys)) != len(asset_keys):
        raise ValueError("duplicate public asset source item")
    case_ids = [item.case_id for item in cases]
    case_hashes = [item.content_hash for item in cases]
    if len(set(case_ids)) != len(case_ids) or len(set(case_hashes)) != len(case_hashes):
        raise ValueError("duplicate sanitized case content")


def _build_import_report(
    assets: list[PublicAsset], cases: list[SanitizedCase]
) -> dict[str, Any]:
    licenses = {
        (
            item.source_dataset,
            item.license_id,
            item.usage_scope,
            getattr(item, "license_review_status", "approved_by_dataset_license"),
            getattr(item, "terms_review_status", "dataset_license_applies"),
        )
        for item in [*assets, *cases]
    }
    return {
        "schema_version": 1,
        "public_asset_count": len(assets),
        "sanitized_case_count": len(cases),
        "missing_media_count": sum(item.media_status == "missing" for item in assets),
        "published_case_count": sum(item.publication_status == "published" for item in cases),
        "quarantined_case_count": sum(item.publication_status == "quarantined" for item in cases),
        "official_evaluation_split": False,
        "split_note": (
            "split_group is deterministic leakage-control metadata; it is not an official "
            "MM-SafetyBench evaluation split."
        ),
        "licenses": [
            {
                "source_dataset": dataset,
                "license_id": license_id,
                "usage_scope": usage_scope,
                "license_review_status": license_status,
                "terms_review_status": terms_status,
            }
            for dataset, license_id, usage_scope, license_status, terms_status in sorted(licenses)
        ],
    }


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[BaseModel]) -> bytes:
    return b"".join(
        (
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _publish_public_data_bundle(
    output_dir: Path,
    assets: list[PublicAsset],
    cases: list[SanitizedCase],
    report: dict[str, Any],
) -> None:
    output_dir = output_dir.absolute()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(tempfile.mkdtemp(prefix=".public-data-", dir=output_dir.parent))
    created_output = not output_dir.exists()
    moved: list[Path] = []
    try:
        payloads = {
            "public_assets.jsonl": _jsonl_bytes(assets),
            "sanitized_cases.jsonl": _jsonl_bytes(cases),
            "import_report.json": _json_bytes(report),
        }
        files: dict[str, dict[str, Any]] = {}
        for filename, payload in payloads.items():
            staged_path = staged_dir / filename
            staged_path.write_bytes(payload)
            files[filename] = {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "counts": {
                "public_assets": len(assets),
                "sanitized_cases": len(cases),
            },
            "files": files,
            "missing_media_count": report["missing_media_count"],
            "license_statuses": report["licenses"],
        }
        (staged_dir / "import_manifest.json").write_bytes(_json_bytes(manifest))
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename in (*payloads, "import_manifest.json"):
            target = output_dir / filename
            os.replace(staged_dir / filename, target)
            moved.append(target)
    except BaseException:
        for target in moved:
            target.unlink(missing_ok=True)
        if created_output and output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        raise
    finally:
        shutil.rmtree(staged_dir, ignore_errors=True)


def _validate_record_cap(max_records: Any) -> None:
    if isinstance(max_records, bool) or not isinstance(max_records, int) or not 1 <= max_records <= MAX_IMPORT_RECORDS:
        raise ValueError(f"max_records must be between 1 and {MAX_IMPORT_RECORDS}")


def _validate_import_metadata(
    *,
    source_dataset: str,
    source_url: str,
    retrieved_at: str,
    license_id: str,
    usage_scope: str,
    license_review_status: str,
    terms_review_status: str,
    max_records: int,
) -> None:
    if not isinstance(source_dataset, str) or not source_dataset.strip():
        raise ValueError("source_dataset must be a non-empty string")
    parsed_url = urlparse(source_url)
    if parsed_url.scheme != "https" or not parsed_url.hostname:
        raise ValueError("source_url must be a public HTTPS URL")
    if not isinstance(retrieved_at, str) or not retrieved_at.strip():
        raise ValueError("retrieved_at must be a non-empty timestamp")
    if not isinstance(license_id, str) or license_id.strip().casefold() in _PLACEHOLDER_LICENSE_IDS:
        raise ValueError("license_id must be explicit and reviewed or marked review-required")
    if usage_scope not in {"smoke_only", "research_only"}:
        raise ValueError("usage_scope must be smoke_only or research_only")
    if license_review_status not in _REVIEW_STATUSES:
        raise ValueError("license_review_status is required")
    if terms_review_status not in _REVIEW_STATUSES:
        raise ValueError("terms_review_status is required")
    if isinstance(max_records, bool) or not isinstance(max_records, int) or not 1 <= max_records <= MAX_IMPORT_RECORDS:
        raise ValueError(f"max_records must be between 1 and {MAX_IMPORT_RECORDS}")


def _validate_sha256(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("value must be a lowercase SHA-256 digest")
    return value


def _validate_public_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("source_url must be a public HTTPS URL without credentials")
    return value


def _validate_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("retrieved_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("retrieved_at must include a timezone")
    return value
