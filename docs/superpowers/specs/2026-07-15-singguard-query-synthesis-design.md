# SingGuard Query Synthesis Design

**Status:** Awaiting written review

**Date:** 2026-07-15

**Usage scope:** Internal research; the first release uses only CC0-1.0,
CC-BY-4.0, and Apache-2.0 sources

## 1. Objective

Build a simple, reproducible English text pipeline that creates 2,000 unique
moderation-content anchors for the existing active-policy SingGuard trace
generator. Gemini realizes controlled content blueprints. Governed open data is
used only as a fact or style seed and is never copied directly or treated as the
SingGuard oracle.

Each accepted anchor binds to exactly one existing active policy. The release
does not create before/after policy counterfactuals.

## 2. Scope and boundaries

The release includes:

- 2,000 accepted query or query-response anchors;
- the seven reviewed rules in `data/active_policies.jsonl`;
- deterministic coverage planning before provider calls;
- 70% fully synthetic and 30% seed-guided content;
- local PII, leakage, source-copy, and duplicate gates;
- independent trace generation with the complete active-policy prompt;
- provenance, rejection reasons, progress, budget, and checkpoint artifacts.

The release excludes:

- images, audio, and video;
- active-policy generation or policy counterfactuals;
- direct reuse of upstream safety labels as an oracle;
- non-commercial or gated datasets;
- production business data;
- expansion of the current synthetic tool evidence stores;
- changes to the Agent tool-selection or pre-tool-reasoning protocol.

The current tool environment has only eight records per table and cannot ground
2,000 diverse queries. Query synthesis may flag a sample as `tool_capable`, but
the first query release does not force a tool trajectory. Tool-capable families
are retained for a later Agent dataset after the evidence stores and hidden tool
selection contract are expanded.

## 3. Release composition

### 3.1 Core quotas

| Axis | Quota |
|---|---:|
| Accepted anchors | 2,000 |
| Safe | 1,000 |
| Unsafe | 1,000 |
| Query only | 1,400 |
| Query plus model response | 600 |
| Fast | 1,400 |
| Slow | 600 |
| Fully synthetic | 1,400 |
| Open-seed guided | 600 |

The 1,000 unsafe anchors are distributed as evenly as possible by primary rule
across the seven active rules: each rule is primary for 142 or 143 anchors.
Additional simultaneously violated rules do not consume another primary-rule
quota. Safe anchors include both
unrelated benign content and hard negatives exercising every documented rule
exception. Thinking mode, source type, style, and label are crossed so none can
become a shortcut for another.

### 3.2 Content-form quotas

| Content form | Anchors |
|---|---:|
| Short advertisement | 400 |
| Social post or influencer caption | 300 |
| Livestream or spoken sales pitch | 300 |
| Product title, bullets, or detail page | 300 |
| Comment, reply, or review | 200 |
| SMS or private message | 200 |
| Customer-support exchange | 200 |
| Search query or neutral informational content | 100 |

The planner also crosses formal, colloquial, promotional, urgent, testimonial,
technical, humorous, and neutral tones with four difficulty bins:

- 500 explicit examples;
- 600 paraphrased examples;
- 400 implicit or obfuscated examples;
- 500 exception or hard-negative examples.

Bounded noise includes spelling variation, emoji, abbreviation, punctuation,
spacing, and synthetic contact obfuscation. Noise is not applied to every
sample and is balanced across safe and unsafe labels.

## 4. Open-source seeds

### 4.1 Enabled sources

| Source | License | Planned use | Seed quota |
|---|---|---|---:|
| [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms) | CC-BY-4.0 | SMS, private-message, spam, scam, and solicitation style | 200 |
| [UCI YouTube Spam Collection](https://archive.ics.uci.edu/dataset/380/youtube%2Bspam%2Bcollection) | CC-BY-4.0 | Comment, link, promotion, and solicitation style | 100 |
| [NVIDIA Aegis 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | CC-BY-4.0 | General risk facts, exceptions, and hard negatives | 120 |
| [Google Civil Comments](https://huggingface.co/datasets/google/civil_comments) | CC0-1.0 | Comment style, harassment, identity attack, and benign comments | 100 |
| [Amazon ESCI](https://github.com/amazon-science/esci-data) | Apache-2.0 | Search, product-title, bullet, and detail-page style | 80 |

Only upstream training splits are eligible when a source defines official
train/test splits. Official test records are excluded to reduce evaluation
contamination. UCI records without official splits are assigned to immutable
families before the local train/dev/holdout split.

### 4.2 Deferred sources

WildGuardMix is gated and subject to the AI2 Responsible Use Guidelines.
ToxicChat and BeaverTails are CC-BY-NC-4.0. They remain catalog entries but are
disabled for this release. A top-level license on an aggregate dataset is not
assumed to replace the obligations of its upstream sources.

### 4.3 Seed governance

Every normalized seed records source name, upstream item ID, provenance URL,
license, usage scope, role, normalized text hash, retrieval time, adapter
version, and optional upstream label. The upstream label is used only for seed
filtering and is never copied into `expected_label` or `expected_answers`.

Real URLs, emails, phone numbers, handles, credentials, and obvious personal
identifiers are replaced before any seed is sent to Gemini. Raw artifacts,
normalized seeds, generated candidates, and accepted training data remain in
separate directories.

## 5. Data contracts

### 5.1 `QueryBlueprint`

The deterministic planner emits one immutable blueprint per requested family:

```text
blueprint_id
family_id
policy_id
intended_label
primary_answer (unsafe only)
intended_answers
thinking_type
conversation_shape
content_form
tone
length_bin
difficulty
noise_profile
tool_capable
source_ref (optional)
```

`intended_answers` is empty for safe examples and contains one or more exact
active-rule titles in active-policy order for unsafe multi-risk examples.
`primary_answer` is used only for quota accounting and must be present in the
ordered answer list. Gemini does not choose or rewrite the active policy.

### 5.2 `GeneratedContent`

Gemini returns only a strict JSON object:

```json
{"query":"...","response":null}
```

`response` may be a string only for a query-response blueprint. The generator
does not emit a label, rule title, reasoning trace, tool call, policy text, or
training message row.

### 5.3 Accepted moderation sample

After local gates, an accepted candidate becomes the existing
`ModerationSample` shape with hidden `expected_label` and `expected_answers`.
The query-only content remains raw text; query-response content is serialized by
the existing compact JSON renderer. Tool fields remain empty in this release.

## 6. Generation and validation flow

1. Fetch and normalize enabled source snapshots with the existing governed
   source catalog.
2. Create the complete 2,000-row quota plan from `--seed` before any Gemini
   request.
3. Select zero or one source seed for each blueprint. Source selection is
   deterministic and family-safe.
4. Send the active policy, intended semantic target, content controls, and
   optional redacted seed to the Gemini content-generator prompt.
5. Parse the strict content response and run local candidate gates.
6. Retry generation at most twice for a failed blueprint. Keep every failed
   attempt with a stable, sanitized reason code.
7. Pass accepted content candidates to the existing active-policy trace
   generator.
8. Accept the final SFT row only when the generated label and exact active-rule
   answers match the hidden blueprint expectation.
9. Split by `family_id`, never by individual rewritten sample.
10. Produce a stratified human-review file before training.

Content generation may batch up to four blueprints in one Gemini request to
control cost. Each returned item is parsed, gated, retried, and accounted for
independently. Trace generation remains one moderation conversation per request
because later tool-capable trajectories require independent conversation state.

## 7. Quality gates

### 7.1 Local gates

A candidate is rejected when any of the following holds:

- invalid schema, blank content, wrong conversation shape, or wrong language;
- literal `[user]` or `[assistant]` wrappers added by the generator;
- policy, label, oracle, annotation, dataset, or generation meta-language;
- a real-looking phone number, email, credential, account, or unapproved URL;
- disallowed high-fidelity operational harm instructions;
- an exact normalized match to a seed or previously accepted candidate;
- token five-gram Jaccard similarity of at least 0.50 to its source seed;
- token five-gram Jaccard similarity of at least 0.85 to an accepted candidate;
- length outside the blueprint bin.

Similarity thresholds are versioned and recorded in the manifest. Duplicate
families, not only duplicate rows, are kept in one split.

### 7.2 Semantic gate

The trace call receives the complete active policy but not blueprint metadata.
The row is rejected when its final label or exact unsafe answer set differs from
the hidden expectation. Slow traces must still satisfy the existing per-rule
evidence and verdict checks. The pipeline never silently changes the blueprint
oracle to agree with Gemini.

### 7.3 Human gate

The query stage writes `content_review_sample.jsonl` for early naturalness,
templating, copied-phrasing, and PII review. After trace generation, the trace
stage writes `review_sample.jsonl` containing at least 100 accepted rows joined
to their blueprint and provenance metadata. It is stratified by primary rule,
safe/unsafe label, source mode, content form, difficulty, thinking type, and
conversation shape. Final review checks policy correctness and trace grounding
in addition to the content checks.

## 8. Outputs and resumability

The query stage writes:

- `plan.jsonl`: immutable blueprints;
- `content_samples.jsonl`: accepted inputs for trace generation;
- `rejected.jsonl`: sanitized failed attempts and reason codes;
- `content_review_sample.jsonl`: stratified content-only review rows;
- `checkpoint.json`: completed blueprint IDs, attempts, and budget accounting;
- `manifest.json`: counts, quota coverage, hashes, source licenses, prompt/model
  versions, similarity thresholds, and quality-gate status;
- `events.jsonl`: operational metadata without raw prompts or credentials.

Resume requires exact matches for the plan hash, active-policy hash, source
snapshot hash, generator prompt hash, model contract, local-gate version, and
quota configuration. It never appends to an output generated under a different
contract.

## 9. Rollout gates

The full 2,000 anchors are generated in three fresh stages:

1. 100-row pilot: inspect all rejections and at least 30 accepted rows.
2. 500-row batch: verify quota coverage, source-copy rate, duplication, and
   label agreement.
3. 2,000-row release: create the 100-row stratified review sample and approve it
   before SFT.

The pipeline stops scaling when semantic acceptance is below 80%, any accepted
row contains real PII, source-copy acceptance is non-zero, near-duplicate
acceptance exceeds 3%, or any planned rule/style/label cell has less than 90%
of its target count after retries.

## 10. Implementation shape

Keep the code path small:

- retain `scripts/fetch_singguard_seeds.py` and extend only its source adapters;
- add one `scripts/generate_singguard_queries.py` entry point;
- add one focused query-planning/generation module;
- reuse the existing Gemini credentials, request budget, retry diagnostics,
  progress renderer, active-policy loader, and trace generator;
- do not add a framework or database service;
- keep network tests mocked and local gates deterministic.

The query command emits `content_samples.jsonl`, which becomes the second
positional input to the existing `scripts/generate_singguard_data.py` command.
The trace command is extended only enough to emit the final joined
`review_sample.jsonl`; this file is not used by ms-swift training.
