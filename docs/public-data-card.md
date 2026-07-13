# Public seed data card

This pipeline creates a small, governed public-data seed for plumbing tests. It does **not** create `Task`, `Oracle`, a SingGuard training set, or an official benchmark evaluation split.

## Sources and permitted use

### MM-SafetyBench

- Canonical source: <https://github.com/isXinLiu/MM-SafetyBench>
- Questions are read from a user-supplied local checkout under `data/processed_questions`.
- Images are expected under `data/imgs/{scenario}/{SD|SD_TYPO|TYPO}/{question_id}.jpg`.
- The importer never downloads images. Download and place them manually according to the upstream instructions, or explicitly set `allow_missing_media: true`.
- The upstream notice states CC BY-NC 4.0 research-only/non-commercial use and says upstream GPT-4 and Stable Diffusion license restrictions also apply. Every asset, report, and config records all three restrictions. This repository therefore defaults to `usage_scope: smoke_only`.
- `TinyVersion_ID_List.json` is supported. `split_group` groups the three variants of one question only for leakage control; it is **not an official MM-SafetyBench evaluation split**.

The normalized record deliberately contains no inferred `label`, `rule_id`, `Oracle`, or policy mapping. Benchmark scenarios are preserved as source metadata rather than converted into business risk labels.

### SAMR and CAC public pages

The example configuration identifies one SAMR page and one CAC page as public regulatory sources. Public visibility is not a license grant. Their terms and downstream reuse rights must be reviewed separately.

- Raw pages should normally enter through the governed crawler's hash-verified `raw + metadata + completion` transaction.
- A local HTML file is accepted only with an exact hostname allowlist, source URL, retrieval time, license ID, and explicit license/terms/content review statuses.
- Extracted `<article>` paragraphs/list items are whitespace-normalized and deduplicated. No rule or verdict is inferred.
- HTML tag removal is not PII sanitization. `content_review_status` defaults to `pending`; a case remains `quarantined` unless license, terms, and content review are all `approved`.
- Crawler imports derive governance from completion-hash-covered metadata. Caller values are expectations only; a mismatch is rejected, so editing config cannot promote an older artifact.

## Normalized outputs

`scripts/import_public_data.py` stages and publishes four files:

```text
public_assets.jsonl
sanitized_cases.jsonl
import_report.json
import_manifest.json
```

The command reserves a destination with an exclusive directory create, moves staged payloads into it, and publishes `import_manifest.json` last as the completion marker. On POSIX, both staging and destination directories remain open and every move is relative to those directory descriptors, so replacing the destination path cannot redirect writes. A failed reservation is deliberately left in place without path-based cleanup; inspect it and remove it manually before retrying with a new destination. On Windows, Python does not expose equivalent directory-relative rename handles: the implementation checks lstat/open/fstat or path identities and fails closed when a change is observed, but cannot promise the stronger POSIX ancestor-swap guarantee.

`import_manifest.json` records counts, output hashes, missing media, license status, and any source truncated or skipped by the global record cap. This includes `MM-SafetyBench` when additional candidate records exist after its cap. Consumers should require `status == "complete"` and verify the hashes.

Every public asset records:

- source dataset and source item ID;
- scenario and prompt;
- repo-relative media path(s) plus `available` or `missing` status;
- deterministic content hash;
- `data_classification: public`;
- license ID and usage scope;
- source URL and retrieval timestamp;
- deterministic leakage-control `split_group`.

Every sanitized case records the factual snippet, the raw-source and snippet content hashes, source/provenance, review statuses, and publication status. It intentionally excludes Oracle fields.

## Reproducible import

1. Copy the schema-v2 `configs/public_sources.example.yaml` and replace local paths/timestamps. Version 1 is rejected with an explicit migration error because v2 requires mutually exclusive `local_html` and `crawler_artifact` fields.
2. Review every license, terms, and content field. Leave regulatory records as `pending` until approval is documented.
3. Run:

```bash
python scripts/import_public_data.py \
  configs/public_sources.local.yaml \
  data/processed/public_seed
```

The command rejects path overlap, placeholder governance, invalid record/byte caps, duplicate YAML/JSON keys and records, noncanonical URLs, symlinked/escaping MM paths, missing images unless explicitly allowed, any existing destination, and incomplete/tampered artifacts. Every JSON, HTML, and media read is charged to per-file and aggregate byte budgets. POSIX reads open a trusted root directory and traverse each component with `dir_fd`/`O_NOFOLLOW`; Windows uses identity checks and the fail-closed limitation described above. It performs no network requests.

## Intended and prohibited uses

Intended:

- data-pipeline smoke tests;
- parser, multimodal loading, and SFT-format debugging;
- out-of-domain research evaluation within source licenses.

Not sufficient or permitted by this pipeline alone:

- claiming a reproduction of the SingGuard training distribution;
- treating source scenarios as content-risk Oracle labels;
- commercial use of CC BY-NC data;
- publishing regulatory snippets before explicit content/PII review;
- measuring business performance without an authorized, deidentified holdout.
