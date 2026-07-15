# Dynamic-Policy Content Risk Agent

This repository is a compact research implementation of policy-conditioned
content moderation. The current main line uses the SingGuard prompt grammar,
Gemini-generated English supervision, optional real two-step tool trajectories,
and native ms-swift training commands.

It is not the official SingGuard implementation. The first release is text-only;
image data can be added after the text pipeline and evaluation gates are stable.

## What is implemented

- one complete active policy set per sample, containing one or more rules;
- SingGuard `fast` and `slow` output modes;
- direct Gemini generation from the full system prompt and conversation;
- zero to two real sequential tool calls with deterministic local results and an
  optional sample-level `required` call sequence;
- strict slow-format validation, hidden smoke-oracle checks, and one tool-free
  serialization repair attempt;
- redacted rejection artifacts, request/cost budgets, progress, and safe resume;
- ms-swift SFT, GRPO, and OPSD/GKD shell entry points.

Policy transitions and before/after counterfactual labels are intentionally not
part of this version.

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,teacher]'
python -m pytest -q
```

On the training server, an existing compatible ms-swift/vLLM environment only
needs the repository in editable mode:

```bash
pip install -e '.[dev]'
```

## Generate the first SFT batch

Use either `GEMINI_API_KEY`, or Vertex AI environment variables. A blank Vertex
template is provided at `configs/vertex_ai.env.example`; never commit credentials.

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

The result is directly trainable:

- `train.jsonl`: accepted ms-swift `messages` rows;
- `rejected.jsonl`: sanitized invalid candidates and deterministic reason codes;
- `checkpoint.json`: progress, budget accounting, and input/prompt fingerprints;
- `manifest.json`: execution status, accepted-mode/tool coverage, and a separate
  semantic `quality_gate` result; `attempted_tool_call_count` includes rejected
  trajectories while `accepted_tool_call_count` counts trainable rows only;
- `events.jsonl`: crash-safe, append-only operational events for provider
  attempts, retries, tool boundaries, validation, and per-sample outcomes.

Provider failures in the manifest and event log include only the request stage,
exception type, HTTP code/status, attempts, SDK version, backend type, and model
name. Prompts, candidates, tool arguments/results, raw provider bodies,
credentials, project IDs, and credential paths are deliberately excluded.

The bundled six-row smoke input contains hidden `expected_label` and
`expected_answers` fields. They are used only by local quality gates and are
never rendered into Gemini's prompt or the exported SFT row. Do not train unless
`manifest.json` reports `quality_gate.status=pass`.

If a request budget or provider failure interrupts the batch, rerun the identical
command with a sufficiently large total budget and `--resume`. Resume is refused
when policy, sample, tool-environment, fully rendered system prompts, checkpoint
counts, saved row counts, provider/model settings, or the tool-call limit differ.

See [docs/singguard-data-runbook.md](docs/singguard-data-runbook.md) for schemas,
review checks, and operational details.

## Tool-call format

The model emits one compact call:

```json
{"name":"verify_claim","arguments":{"claim":"Lose ten pounds in seven days"}}
```

The local environment returns one compact result:

```json
{"status":"ok","result":{"verdict":"unsupported"}}
```

Supported tools are `search_cases`, `verify_claim`, `inspect_destination`, and
`get_content_context`. The final answer is normal SingGuard text, not another
tool call. In ms-swift JSONL, calls and results use the `tool_call` and
`tool_response` roles, while the top-level `tools` field is a JSON string.
For a required trajectory, the sample lists tools in execution order and the
pipeline rejects missing, extra, or reordered calls.

## SFT

```bash
TRAIN_DATA=outputs/singguard-v1/train.jsonl \
VAL_DATA= \
OUTPUT_DIR=outputs/qwen3_vl_8b_singguard_sft \
bash scripts/train_qwen3_vl_sft.sh
```

The default training scripts target `Qwen/Qwen3-VL-8B-Instruct` with LoRA and
are intended to stay close to the official ms-swift command-line workflow.

## GRPO and OPSD

The existing Track A fixtures can still be used to smoke-test the training
commands:

```bash
python scripts/prepare_ms_swift_rl.py grpo \
  data/fixtures/tasks.jsonl data/fixtures/oracle.jsonl outputs/grpo \
  --train-ratio 1 --dev-ratio 0 --holdout-ratio 0

python scripts/prepare_ms_swift_rl.py opsd \
  data/fixtures/tasks.jsonl data/fixtures/oracle.jsonl outputs/opsd \
  --train-ratio 1 --dev-ratio 0 --holdout-ratio 0
```

Start the rollout service and then train:

```bash
bash scripts/start_qwen3_vl_rollout.sh

SFT_ADAPTER=outputs/qwen3_vl_8b_singguard_sft/best \
TRAIN_DATA=outputs/grpo/train.jsonl \
bash scripts/train_qwen3_vl_grpo.sh

SFT_ADAPTER=outputs/qwen3_vl_8b_singguard_sft/best \
TRAIN_DATA=outputs/opsd/train.jsonl \
bash scripts/train_qwen3_vl_opsd.sh
```

The OPSD path uses ms-swift GKD. The current divergence approximation keeps the
top 128 vocabulary entries instead of materializing the full vocabulary.

## Data boundary

`data/fixtures` and `data/tool_env` contain synthetic or de-identified development
records. Oracle data must never be placed in retrieval results or model prompts.
Production evaluation requires a separately approved, de-identified holdout that
is not committed to this repository.
