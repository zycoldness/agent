# Public seed data card

This pipeline creates a small, governed public-data seed for plumbing tests. It does **not** create `Task`, `Oracle`, a SingGuard training set, or an official benchmark evaluation split.

## Sources and permitted use

### MM-SafetyBench

- Canonical source: <https://github.com/isXinLiu/MM-SafetyBench>
- Questions are read from a user-supplied local checkout under `data/processed_questions`.
- Images are expected under `data/imgs/{scenario}/{SD|SD_TYPO|TYPO}/{question_id}.jpg`.
- The importer never downloads images. Download and place them manually according to the upstream instructions, or explicitly set `allow_missing_media: true`.
- The upstream dataset notice states CC BY-NC 4.0, research-only, non-commercial use. This repository therefore defaults it to `usage_scope: smoke_only`. Do not use it for commercial training or production decisions.
- `TinyVersion_ID_List.json` is supported. `split_group` groups the three variants of one question only for leakage control; it is **not an official MM-SafetyBench evaluation split**.

The normalized record deliberately contains no inferred `label`, `rule_id`, `Oracle`, or policy mapping. Benchmark scenarios are preserved as source metadata rather than converted into business risk labels.

### SAMR and CAC public pages

The example configuration identifies one SAMR page and one CAC page as public regulatory sources. Public visibility is not a license grant. Their terms and downstream reuse rights must be reviewed separately.

- Raw pages should normally enter through the governed crawler's hash-verified `raw + metadata + completion` transaction.
- A local HTML file is accepted only when the operator provides the public source URL, retrieval time, license ID, terms-review status, and license-review status.
- Extracted `<article>` paragraphs/list items are whitespace-normalized and deduplicated. No rule or verdict is inferred.
- Cases remain `quarantined` unless both `license_review_status` and `terms_review_status` are `approved`.

## Normalized outputs

`scripts/import_public_data.py` stages and publishes four files:

```text
public_assets.jsonl
sanitized_cases.jsonl
import_report.json
import_manifest.json
```

`import_manifest.json` is committed last and records record counts, output hashes, missing-media counts, and license status. Consumers should require `status == "complete"` and verify the listed SHA-256 hashes.

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

1. Copy `configs/public_sources.example.yaml` and replace local paths/timestamps.
2. Review every license and terms field. Leave regulatory records as `review_required` until approval is documented.
3. Run:

```bash
python scripts/import_public_data.py \
  configs/public_sources.local.yaml \
  data/processed/public_seed
```

The command rejects input/output path overlap, placeholder licenses, invalid record caps, duplicate assets/cases, missing MM-SafetyBench images unless explicitly allowed, pre-existing output files, and incomplete/tampered crawler artifacts. It performs no network requests.

## Intended and prohibited uses

Intended:

- data-pipeline smoke tests;
- parser, multimodal loading, and SFT-format debugging;
- out-of-domain research evaluation within source licenses.

Not sufficient or permitted by this pipeline alone:

- claiming a reproduction of the SingGuard training distribution;
- treating source scenarios as content-risk Oracle labels;
- commercial use of CC BY-NC data;
- publishing quarantined regulatory snippets;
- measuring business performance without an authorized, deidentified holdout.
