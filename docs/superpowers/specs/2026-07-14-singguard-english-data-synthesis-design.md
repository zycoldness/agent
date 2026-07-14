# SingGuard English Data Synthesis and Quality-Gate Design

**Status:** Pending written review  
**Date:** 2026-07-14  
**Usage scope:** Internal research only

## 1. Objective

Build a reproducible English-only, text-only data pipeline for the first meaningful SingGuard SFT experiment. The pipeline must produce policy-conditioned examples with diverse platform language, valid before/after policy counterfactuals, grounded fast or slow targets, source provenance, and machine-auditable rejection reasons.

The first accepted release contains:

- 2,000 unique content anchors;
- two policy views per anchor, one before and one after a policy transformation;
- 4,000 rendered SFT rows;
- eight risk domains;
- balanced `unsafe_to_unsafe`, `unsafe_to_safe`, `safe_to_unsafe`, and `safe_to_safe` transitions;
- 70% fast and 30% slow targets, stratified so thinking type is not a label shortcut;
- an 80/10/10 split by content family, never by rendered row.

This produces 1,600/200/200 anchors and 3,200/400/400 rendered rows in train/dev/holdout.

The initial release is an experimental dataset, not evidence of production moderation quality.

## 2. Non-goals

- Images and video are deferred until the English text pipeline passes its quality gates.
- Tool-use trajectories and Agentic RL data are not mixed into this Guard SFT release.
- `fast-slow` hybrid early-exit targets are deferred until an explicit confidence and escalation policy exists.
- The pipeline does not claim that model agreement is equivalent to human ground truth.
- A commercial-compatible artifact is not produced in this phase because research-only sources may be used.
- The pipeline does not copy entire open datasets into the training set.

## 3. Dataset composition

### 3.1 Risk domains

Each domain contributes 250 unique anchors and 500 rendered policy views.

| Domain | Required subtypes |
|---|---|
| Deceptive Efficacy and Commercial Claims | guaranteed outcome, guaranteed time, universal success, pseudo-clinical authority, permanent result, testimonial framed as proof, qualified safe claim |
| Off-Platform Solicitation and Spam | phone, email, WhatsApp/Telegram, QR reference, obfuscated handle, link-in-bio, move-to-DM request, legitimate support exception |
| Fraud and Scams | phishing, impersonation, advance-fee fraud, fake prize, investment scam, credential harvesting, benign fraud discussion |
| Sexual and Exploitative Content | explicit sexual material, sexual exploitation, minors, coercion, suggestive boundary content, clinical or educational exception |
| Violence, Weapons, and Criminal Facilitation | violent threat, weapon acquisition, criminal instruction, dangerous wrongdoing, news or prevention context |
| Hate, Harassment, and Abusive Content | protected-class attack, targeted harassment, dehumanization, bullying, counterspeech, quoted abuse |
| Self-Harm and Dangerous Behavior | encouragement, method seeking, dangerous challenge, recovery/support context, prevention content |
| Cyber, Privacy, and Agent Safety | hacking, credential theft, privacy exposure, information manipulation, prompt extraction, safeguard bypass, defensive exception |

The broad paper taxonomy is therefore represented, but heterogeneous categories such as “Unethical Behavior” are split into rules that can be evaluated and changed independently.

### 3.2 Policy transitions

Each anchor has a `before` and `after` policy view. A transition describes the ordered label pair, not two unrelated policies.

| Transition | Anchors | Rendered rows |
|---|---:|---:|
| `unsafe_to_unsafe` | 500 | 1,000 |
| `unsafe_to_safe` | 500 | 1,000 |
| `safe_to_unsafe` | 500 | 1,000 |
| `safe_to_safe` | 500 | 1,000 |

Policy transformations include rule addition, removal, narrowing, broadening, semantic rewrite, merge, and exception addition. Every active policy contains 3–8 rules. It includes the relevant rule when applicable and realistic distractors from other domains. A transformation must change a real policy boundary; swapping in a single unrelated rule is not sufficient for the full dataset.

### 3.3 Input style quotas

| Style | Share | Anchors |
|---|---:|---:|
| Short-form advertisement | 20% | 400 |
| Social post or influencer caption | 15% | 300 |
| Livestream or spoken sales pitch | 15% | 300 |
| Product title, bullets, or detail page | 15% | 300 |
| Comment, reply, or user review | 10% | 200 |
| SMS or private message | 10% | 200 |
| Customer-support dialogue | 10% | 200 |
| Neutral informational context | 5% | 100 |

Global style quotas are crossed with risk domain, label, transition, and thinking type by a deterministic quota planner. No style is allowed to become a proxy for one risk domain or label.

Additional controlled axes are:

- query-only 70%, query-plus-response 30%;
- headline, short, medium, and long/dialogue length bins;
- explicit, paraphrased, implicit, and borderline/exception difficulty;
- formal, colloquial, promotional, urgent, testimonial, technical, humorous, and neutral tones;
- bounded spelling errors, emoji, abbreviations, punctuation variation, and contact obfuscation;
- merchant, influencer, reviewer, support agent, ordinary user, and neutral narrator roles.

Input language should be diverse. Assistant output structure should not be diverse: labels, tags, reasoning steps, rule order, and verdict vocabulary remain canonical. Only summaries, evidence explanations, and concise final-judgment wording receive controlled lexical variation.

## 4. Source governance

### 4.1 Source roles

Open data is used for three different purposes and is not treated as a single label pool:

1. **Risk-fact seeds:** examples whose facts can help define a content blueprint.
2. **Style seeds:** platform-like language that informs tone and format but not the oracle.
3. **Hard negatives and evaluation candidates:** benign, quoted, educational, or borderline content used to reduce false positives.

Approximately 30% of anchors use an open-data seed; 70% use fully synthetic blueprints. Generated text is checked against its seed and rejected when it is too similar. Original upstream labels may filter candidates but never directly become the SingGuard oracle.

### 4.2 Initial source catalog

| Source | License/use | Primary role |
|---|---|---|
| UCI SMS Spam Collection | CC BY 4.0 | SMS and private-message solicitation style |
| UCI YouTube Spam Collection | CC BY 4.0 | social comment and spam style |
| Amazon ESCI | Apache-2.0 | product title, query, and listing style |
| Nemotron/Aegis Safety Dataset 2.0 | CC BY 4.0 | general safety seeds and hard negatives |
| WildGuardMix | ODC-BY, gated access | general safety and policy-boundary candidates |
| ToxicChat | CC BY-NC 4.0 | conversational toxicity candidates |
| BeaverTails | CC BY-NC 4.0 | safety-category and boundary candidates |
| Civil Comments | CC0-1.0 | ordinary comments, abuse, quotation, and counterspeech candidates |
| FTC/FDA public cases and warnings | rights review required per record | deceptive commercial-claim facts |

The source catalog records access date, upstream ID, URL, declared license, usage scope, content hash, and adapter version. Any non-commercial source marks the complete downstream release and trained adapter `research_only`. Unknown-license records are rejected before generation.

### 4.3 Unified seed contract

Each adapter emits a strict `SeedRecord` with:

```text
source
source_id
provenance_url
license
usage_scope
source_role
text
source_label (optional, never an oracle)
content_hash
retrieved_at
adapter_version
```

Raw source files, normalized seeds, generated candidates, and training bundles remain separate. Split membership is determined only after a family ID has been assigned.

## 5. Prompt architecture

The paper Appendix is the canonical policy-grounded classification source, but it is not one universal application prompt.

```text
Shared policy core
├── Guard prompt: no tools, fast/slow/fast-slow output contract
├── Agent prompt: bounded tool protocol and final decision
├── Generator prompt: realize an English content blueprint
└── Verifier prompt: independently judge shuffled policy views
```

The Guard prompt contains the paper’s task, thinking mode, runtime policy replacement, classification logic, output format, and conversation wrapper. When a runtime policy is supplied, it replaces the default taxonomy rather than being appended to it.

The Agent prompt reuses the policy core and additionally defines available tools, JSON action schema, a three-turn limit, tool-use necessity, observation trust boundaries, invalid/repeated calls, and termination behavior. Tool schemas are rendered from the same registry used by the environment. The first data release does not use the Agent prompt.

Generator and verifier prompts are separate and stateless. Each prompt is versioned, and every manifest stores the prompt version and SHA-256. The system instruction must be injected either through row messages or a tokenizer chat template, never both.

## 6. Data contracts

### 6.1 `AnchorBlueprint`

The deterministic planner creates a blueprint before any model request. It fixes:

- `anchor_id` and `family_id`;
- risk domain and subtype;
- source record, if any;
- input style, tone, length, role, difficulty, and noise profile;
- query-only or query-plus-response structure;
- policy transformation and ordered transition;
- before/after policy IDs;
- intended semantic facts and prohibited meta-language;
- thinking type assignment for each rendered view.

Gemini may realize the blueprint but may not change these fields.

### 6.2 `PolicyPair`

The local compiler builds before/after policies from versioned rule definitions. A pair contains:

- ordered `before` and `after` active policies;
- transformation type;
- deterministic before/after oracle labels;
- the active answer rule for each unsafe view;
- required per-rule verdicts for slow views;
- a justification showing how the transformation implies the label pair.

The compiler rejects internally inconsistent pairs before any provider call.

### 6.3 `GeneratedContent`

The generator returns only structured content realization:

- query;
- optional response;
- declared style and length bin;
- a short list of literal risk cues or benign cues for local checking;
- no policy label, transition name, or final SingGuard answer.

The generator does not emit final `messages` rows.

### 6.4 `VerifierVerdict`

The verifier receives the generated content and shuffled, opaque policy views. It does not receive oracle labels, transition type, source labels, blueprint intent, or generator reasoning. For each view it returns:

- label;
- active rule title or `null`;
- literal evidence quote for unsafe decisions;
- confidence;
- ambiguity flag;
- per-rule verdicts and concise explanations when the assigned mode is slow.

It also returns content-style category, naturalness score, template-likeness, and issue codes.

For accepted slow examples, the verifier’s grounded summary and explanations become trace fields. The local renderer, not Gemini, produces the canonical assistant text.

## 7. End-to-end flow

1. **Import seeds.** Source-specific adapters normalize records, validate licenses, hash content, and remove unusable records.
2. **Plan quotas.** A deterministic seed produces the complete blueprint plan before calls begin. The plan makes missing cells visible.
3. **Compile policies.** The local rule compiler constructs a valid ordered policy pair and oracle for each blueprint.
4. **Generate content.** Gemini realizes English query/response content under a strict JSON schema. It does not label the sample.
5. **Run local candidate gates.** Schema, English, length, prohibited meta-language, synthetic PII, source similarity, and duplicate checks run before verification.
6. **Blind verification.** Gemini independently classifies shuffled policy views and creates grounded slow fields where required.
7. **Compare with oracle.** Both view labels and unsafe answer rules must exactly match. Evidence quotes must be literal substrings. Slow checks must cover every active rule in order.
8. **Accept or reject.** Failed candidates are recorded with stable rejection codes. They are never silently relabeled or repaired into the accepted set.
9. **Retry or backfill.** At most two regeneration retries are allowed per blueprint. An unresolved blueprint is retained in the rejected set and replaced by a new planned anchor in the same quota cell.
10. **Split by family.** Families, seeds, near-duplicate groups, and policy pairs cannot cross train/dev/holdout boundaries.
11. **Render.** One canonical renderer produces Guard SFT messages JSONL and a manifest.
12. **Audit.** Corpus metrics and a deterministic stratified review sample are produced before training.

Generator and verifier requests may be batched for cost efficiency, but each item keeps an opaque ID and is validated independently. Provider retries, request budgets, atomic checkpoints, sanitized errors, text-only and future image inputs reuse the existing teacher-provider infrastructure.

## 8. Quality gates

### 8.1 Per-record hard gates

Every accepted source example must satisfy all of the following:

- strict schema validation and non-empty content;
- English-only target, allowing bounded product names, handles, and common borrowed terms;
- declared source and license provenance;
- 3–8 unique active rules with unique titles and IDs;
- a valid before/after policy transformation and transition;
- verifier label and answer rule exactly equal to the local oracle for both views;
- unsafe evidence quote is a literal content substring;
- safe result has no answer rule;
- `ambiguous=false` and verifier confidence at least 0.85;
- verifier naturalness at least 4/5 and `template_like=false`;
- slow summary is present and every active rule is checked in order; an unsafe view marks its answer rule `HIT`, while a safe view contains no `HIT` rule;
- no policy-external rule is introduced in reasoning;
- no real personal phone number, email, credential, or account is generated; reserved/synthetic contact details are used;
- harmful examples may state or request unsafe content but may not add novel, high-fidelity operational instructions for crime, self-harm, weapon construction, credential theft, or safeguard bypass;
- no generation meta-language or leaked oracle fields appear in student-visible text.

### 8.2 Corpus hard gates

- exactly 2,000 accepted unique anchors and 4,000 rendered rows;
- exact normalized duplicates: zero;
- cross-anchor character 5-gram Jaccard similarity at or above 0.82: rejected and grouped before splitting;
- generated-to-source similarity at or above 0.75: rejected when enough text exists for a meaningful comparison;
- four transitions balanced exactly at 500 anchors each;
- risk domains and style quotas filled exactly, except documented backfill rounding of at most one anchor per crossed cell;
- no identical normalized opening of meaningful length may exceed 1% of anchors;
- mode, label, style, domain, and transition contingency tables must not contain an unintended empty cell required by the plan;
- accepted examples have 100% blind-verifier agreement by construction;
- corpus Data Quality Score at least 90/100;
- all output files have hashes and match their manifest counts.

### 8.3 Pilot stop gates

The pipeline first runs a 100-anchor pilot. Full generation does not start unless:

- first-pass verifier agreement is at least 80%;
- final acceptance after allowed retries is at least 90%;
- near-duplicate rejection is at most 5%;
- every planned domain, transition, mode, and style cell is represented;
- no critical provenance, leakage, or PII failure is found in the review sample.

Failure pauses scaling and requires prompt, rule, or quota revision. It is not solved by weakening the thresholds.

## 9. Review and reporting

The pipeline emits:

- `accepted.jsonl` with source contracts, policies, oracle, verifier fields, and provenance;
- `rejected.jsonl` with candidate metadata and stable rejection codes;
- `train.jsonl`, `dev.jsonl`, and `holdout.jsonl` containing only student-visible messages;
- `manifest.json` with counts, hashes, prompt versions, model configuration, provider usage, licenses, and usage scope;
- `quality_report.json` with completeness, consistency, validity, uniqueness, provenance/timeliness, distributions, correlations, duplicate statistics, and rejection rates;
- `review_sample.jsonl`, a deterministic 10% stratified sample of accepted anchors plus unresolved disagreement and boundary cases.

Codex reviews the pilot report, the stratified sample, every unresolved disagreement category, and all boundary/exception slices before approving the full batch. This review is a model-assisted quality check, not a substitute for a domain-owner review before production use.

## 10. Failure handling and reproducibility

- CLI output directories must not already exist.
- Writes are atomic and checkpointed; a completed file is never presented when a batch is partial.
- Generation is deterministic with respect to quota plan, prompt version, local seed, source snapshot, and accepted provider responses.
- Provider failures, invalid JSON, budget exhaustion, license failure, semantic disagreement, ambiguity, trace grounding failure, duplicate detection, and quota backfill use separate rejection/status codes.
- Raw provider exceptions and credentials are never persisted.
- Retrying a completed item is idempotent by anchor ID and prompt hash.
- Changing a prompt, policy catalog, source adapter, verifier threshold, or duplicate threshold creates a new dataset version.

## 11. Testing strategy

Implementation follows test-driven development and includes:

- deterministic quota-planner distribution tests;
- policy-pair tests for all four transitions and seven transformation types;
- prompt snapshot tests against the canonical Guard and Agent contracts;
- an assertion that blind-verifier requests contain no oracle, transition, source label, or blueprint intent;
- schema and adversarial-response tests for generator and verifier output;
- evidence-quote, active-rule, slow-rule-order, and ambiguity rejection tests;
- source-license and research-only propagation tests;
- exact, near-duplicate, and source-copy rejection tests;
- family-safe split tests;
- atomic checkpoint, resume, budget, and retry tests;
- an end-to-end fake-provider pilot that produces accepted, rejected, audit, review, and ms-swift artifacts without network access;
- a final server smoke test with a small real Gemini batch before the 100-anchor pilot.

## 12. Implementation boundaries

The implementation reuses the existing Gemini provider, budget, retry, multimodal attachment, and atomic checkpoint primitives. It adds a separate SingGuard synthesis path rather than extending the current two-hop Agentic teacher-response schema.

The first implementation phase contains only:

1. source catalog and a small set of source adapters;
2. rule catalog, quota planner, and policy-pair compiler;
3. versioned prompt renderers;
4. generator/verifier schemas and batch orchestration;
5. deterministic quality gates, audit report, and review sampler;
6. export through the existing SingGuard ms-swift renderer.

Tool-SFT generation remains a later, separate phase using the Agent prompt and the local `RiskEnvironment` for real observations.
