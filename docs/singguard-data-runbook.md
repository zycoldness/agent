# SingGuard active-policy data runbook

This runbook produces an English, text-only SFT batch from reviewed active
policies and content prompts. Gemini supplies the supervision trace; local code
only executes tools and enforces deterministic contracts.

## 1. Inputs

`data/active_policies.jsonl` contains one policy set per line. A set may contain
one or many simultaneously active rules:

```json
{"policy_id":"commerce-v1","rules":[{"rule_id":"EFF-001","title":"Deceptive Efficacy","text":"Do not claim a guaranteed or medically unsupported outcome."}]}
```

`data/content_samples.jsonl` contains the target conversations:

```json
{"sample_id":"sample-1","policy_id":"commerce-v1","thinking_type":"slow","query":"Guaranteed results in seven days.","tool_names":["verify_claim"],"tool_policy":"required","expected_label":"unsafe","expected_answers":["Deceptive Efficacy"]}
```

Optional fields are `response`, `tool_names`, `tool_policy`, `expected_label`,
and `expected_answers`. `tool_policy=required` makes `tool_names` an exact,
ordered call sequence. Expectations are local quality-oracle fields: they are
hashed with the sample plan and checked after generation, but never sent to
Gemini or exported into `train.jsonl`. The first version does not use images.
Every `policy_id` must resolve to exactly one active policy set.

The deterministic tool environment is stored under `data/tool_env`:

- `cases.jsonl` for `search_cases`;
- `claim_evidence.jsonl` for `verify_claim`;
- `destinations.jsonl` for `inspect_destination`.

These records are fixtures, not production evidence.

## 2. Credentials

Developer API:

```bash
export GEMINI_API_KEY='set-this-only-in-the-shell'
```

Vertex AI:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=true
```

Do not write secrets into commands, JSONL, logs, or Git. The repository's
`configs/vertex_ai.env.example` intentionally leaves all values blank.

## 3. Validate locally

```bash
pip install -e '.[dev,teacher]'
python -m pytest -q
python scripts/generate_singguard_data.py --help
```

## 4. Generate a batch

```bash
export GEMINI_GENERATOR_MODEL='your-gemini-model'

rm -rf outputs/singguard-v1
python scripts/generate_singguard_data.py \
  data/active_policies.jsonl \
  data/content_samples.jsonl \
  outputs/singguard-v1 \
  --tool-env data/tool_env \
  --max-tool-calls 2 \
  --max-requests 500 \
  --max-output-tokens 4096 \
  --request-timeout 120
```

The progress bar is written to stderr. The manifest remains the only stdout JSON
object, so automation can parse it.

For each sample, the pipeline:

1. renders the complete SingGuard system prompt with all active rules;
2. gives the prompt and conversation to Gemini;
3. executes each allowed tool locally and sends the real result back to Gemini;
4. accepts at most two sequential calls;
5. enforces the complete ordered sequence for `tool_policy=required` samples;
6. validates the final `fast` or three-step `slow` completion, tolerating one
   blank line after the label while preserving the original text;
7. requires evidence for `HIT`, `NOT HIT`, and `OBSERVED BUT NOT UNSAFE`, while
   allowing a bare `NOT APPLICABLE` verdict;
8. makes at most one fresh, tool-free format/consistency-repair request;
9. checks hidden semantic expectations without exposing them to the model;
10. writes either an accepted training row or a sanitized rejection.

## 5. Resume safely

If the provider fails or the request/cost budget is reached, preserve the output
directory and repeat the same command with `--resume`. `--max-requests` and
`--max-cost-usd` are total batch limits, so a resumed limit must not be lower than
already recorded usage.

```bash
python scripts/generate_singguard_data.py \
  data/active_policies.jsonl \
  data/content_samples.jsonl \
  outputs/singguard-v1 \
  --tool-env data/tool_env \
  --max-tool-calls 2 \
  --max-requests 800 \
  --resume
```

Resume validates hashes of the normalized policies, samples, three tool tables,
and every fully rendered system prompt. It also checks that saved accepted and
rejected row counts match the checkpoint before making another provider request.
The provider/model generation contract and tool-call limit must also match. It
never silently mixes versions, generation settings, or truncated artifacts.

## 6. Inspect before training

Review:

- `manifest.json`: execution status, accepted/rejected/repaired counts,
  accepted-mode distribution, required-tool coverage, semantic `quality_gate`,
  budget accounting, attempted tool calls, and tool calls retained in accepted
  rows;
- `train.jsonl`: complete prompts, real trajectories, and original accepted
  Gemini text;
- `rejected.jsonl`: reason codes and bounded, redacted candidates;
- `checkpoint.json`: completed count, budget, and fingerprints.
- `events.jsonl`: fsync-backed structured events for each sample, Gemini request,
  retry, tool boundary, validation result, and terminal batch status.

When `reason=provider_error`, inspect `manifest.provider_failure` and the tail of
`events.jsonl`. Safe diagnostics include `stage` (`client_init`, `initial`,
`tool_response`, `repair_init`, or `repair`), exception type, HTTP code,
provider status, attempt number, retry delay, token accounting, model, backend,
and `google-genai` version. They intentionally exclude raw exception text,
request/response bodies, prompts, candidates, tool arguments/results, API keys,
project IDs, credential paths, and service-account data.
`repair_succeeded` and `repair_failed` events expose only a stable failure code
and whether a repaired candidate existed; they never include candidate text.

```bash
tail -n 30 outputs/singguard-v3/events.jsonl
jq '.provider, .provider_failure, .budget' outputs/singguard-v3/manifest.json
```

Minimum first-batch checks:

- `status` is `complete` and `quality_gate.status` is `pass`;
- all six smoke expectations are accepted;
- all three required-tool samples are accepted, `accepted_tool_call_count` is
  three, and `attempted_tool_call_count` is not smaller;
- every system prompt includes the intended complete active policy;
- safe/unsafe and `fast`/`slow` distributions match the reviewed sample plan;
- `slow` traces check every active rule in policy order;
- tool results are relevant, deterministic, and not copied from an oracle;
- repaired rows are manually inspected as a separate slice;
- near duplicates and templated wording are not dominating the batch.

`status=complete` means only that every planned row was processed. It does not
mean the batch is trainable; that decision belongs to `quality_gate` plus human
review. The CLI exits with code 2 for an incomplete run and code 3 when execution
completes but the semantic quality gate fails.

Do not hand-edit accepted rows. Correct prompts, policies, samples, or tool data
and create a new versioned output directory.

## 7. Train

```bash
TRAIN_DATA=outputs/singguard-v1/train.jsonl \
VAL_DATA= \
OUTPUT_DIR=outputs/qwen3_vl_8b_singguard_sft \
bash scripts/train_qwen3_vl_sft.sh
```

Keep a separate, approved holdout. Generated training rows are not a substitute
for real content-risk evaluation data.
