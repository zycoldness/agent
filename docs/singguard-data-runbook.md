# SingGuard English data pilot runbook

This run produces the first English, text-only, research-only, Guard-only data
batch. It is an experiment artifact, not production moderation evidence. Images
and agent tool traces are deliberately deferred until the text pipeline passes.

## 1. Install and verify

```bash
pip install -e '.[dev,teacher]'
python -m pytest -q
```

Use either Gemini Developer API credentials:

```bash
export GEMINI_API_KEY='set-this-on-the-server'
```

or Vertex AI credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=true
```

Never put credentials in a command, JSONL file, shell script, or Git commit.

## 2. Fetch governed public style seeds

```bash
rm -rf outputs/singguard-seeds
python scripts/fetch_singguard_seeds.py \
  configs/singguard_sources.yaml \
  outputs/singguard-seeds
```

The enabled first version fetches UCI SMS Spam and YouTube Spam data under
CC-BY-4.0. Seeds are style references only; their labels never become the local
oracle. Inspect `outputs/singguard-seeds/manifest.json` before continuing.

## 3. Freeze and inspect the quota plan

```bash
rm -rf outputs/singguard-plan
python scripts/generate_singguard_data.py outputs/singguard-plan \
  --anchors 100 \
  --seed 42 \
  --plan-only
```

`plan.jsonl` must contain 100 unique anchors spanning all eight domains, all four
policy transitions, and the planned style/difficulty mix. Plan-only never needs a
Gemini credential and never sends external data.

## 4. Run the real 100-anchor pilot

Set the two model names explicitly so the run is reproducible:

```bash
export GEMINI_GENERATOR_MODEL='your-generator-model'
export GEMINI_VERIFIER_MODEL='your-verifier-model'

rm -rf outputs/singguard-pilot-v1
python scripts/generate_singguard_data.py outputs/singguard-pilot-v1 \
  --anchors 100 \
  --seed 42 \
  --generator-model "$GEMINI_GENERATOR_MODEL" \
  --verifier-model "$GEMINI_VERIFIER_MODEL" \
  --seeds outputs/singguard-seeds/seeds.jsonl \
  --allow-external-data \
  --max-requests 800 \
  --max-output-tokens 4096 \
  --request-timeout 120 \
  --pilot
```

If the command stops because of a transient provider failure or request budget,
keep the directory and resume with the identical arguments plus a sufficiently
large total request limit:

```bash
python scripts/generate_singguard_data.py outputs/singguard-pilot-v1 \
  --anchors 100 \
  --seed 42 \
  --generator-model "$GEMINI_GENERATOR_MODEL" \
  --verifier-model "$GEMINI_VERIFIER_MODEL" \
  --seeds outputs/singguard-seeds/seeds.jsonl \
  --allow-external-data \
  --max-requests 1200 \
  --max-output-tokens 4096 \
  --request-timeout 120 \
  --pilot \
  --resume
```

Resume is rejected if the plan, prompt files, or seed assignments differ from the
checkpoint. Requests and credentials are never persisted.

## 5. Review before scaling

Inspect these files:

- `quality_report.json`: acceptance, first-pass verifier agreement, DQS,
  duplicate rate, coverage, distributions, and rejection reasons;
- `review_sample.jsonl`: deterministic 10% whole-anchor sample spanning all eight
  domains;
- `rejected.jsonl`: every failed attempt with local reason codes, but no provider
  prompt or credential;
- `ms_swift/train.jsonl`, `dev.jsonl`, and `holdout.jsonl`: two policy-conditioned
  rows per accepted anchor.

Stop and revise prompts/rules as a new dataset version if first-pass agreement is
below 80%, final acceptance is below 90%, near-duplicate rejection exceeds 5%,
DQS is below 90, quota coverage fails, or human review finds systematic errors.
Do not hand-edit accepted rows.

Only after approval, create a new directory for the full batch:

```bash
python scripts/generate_singguard_data.py outputs/singguard-2000-v1 \
  --anchors 2000 \
  --seed 42 \
  --generator-model "$GEMINI_GENERATOR_MODEL" \
  --verifier-model "$GEMINI_VERIFIER_MODEL" \
  --seeds outputs/singguard-seeds/seeds.jsonl \
  --allow-external-data \
  --max-requests 15000 \
  --max-output-tokens 4096 \
  --request-timeout 120
```

The expected upper target is 2,000 accepted anchors and 4,000 SFT rows. The
actual accepted count is reported rather than silently padding failed quota cells.
