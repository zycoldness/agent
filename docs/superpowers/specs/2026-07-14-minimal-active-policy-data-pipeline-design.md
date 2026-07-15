# Minimal Active-Policy Data Pipeline

## Goal

Build a small, understandable pipeline that turns a complete SingGuard-style
prompt into ms-swift SFT data by asking Gemini to execute the prompt as the
teacher model.

Each example contains one conversation evaluated once under the active policy
shown in its system prompt. The active policy is a set containing one or more
rules. The first version does not create before/after policy pairs or
counterfactual label transitions.

## Non-goals

- No `safe_to_unsafe`, `unsafe_to_safe`, or other transition planning.
- No automatic rule broadening, narrowing, rewriting, or exception synthesis.
- No local semantic oracle that attempts to infer the label from keywords.
- No Python-generated chain of thought.
- No multimodal generation in the first text-only milestone.
- No full `fast-slow` inference router in the data generator.

## Inputs

### System-prompt template

Use the SingGuard default system prompt structure:

- task definition;
- `thinking_type` control;
- default taxonomy;
- optional runtime active policy;
- classification logic;
- output grammar;
- conversation content.

When a runtime active policy is supplied, it replaces the default taxonomy for
that example. It may contain one or many active rules.

### Active policies

`data/active_policies.jsonl` stores reusable active-policy sets. A record has:

```json
{
  "policy_id": "commerce-risk-v1",
  "rules": [
    {
      "title": "Deceptive Efficacy",
      "text": "Content must not guarantee a specific result or time to result."
    },
    {
      "title": "Off-Platform Solicitation",
      "text": "Content must not direct users to prohibited private channels."
    }
  ]
}
```

The order is part of the prompt. Dataset preparation may shuffle order across
examples, but it must not change rule text or meaning.

### Content samples

`data/content_samples.jsonl` stores text-only moderation targets:

```json
{
  "sample_id": "sample-000001",
  "policy_id": "commerce-risk-v1",
  "thinking_type": "slow",
  "query": "Guaranteed to lose ten pounds in seven days.",
  "response": null,
  "tools": []
}
```

Samples may come from reviewed public data or a separate Gemini content
synthesis pass. Content synthesis is upstream of this pipeline and does not
change how teacher traces are produced.

## Core data flow

```text
system template + active policy + sample + thinking_type + optional tools
                              |
                              v
                     complete student prompt
                              |
                              v
                       Gemini teacher executes
                              |
                              v
                    validate and save messages
                              |
                              v
                       ms-swift train.jsonl
```

The generator does not separately predict an oracle. Gemini's execution of the
complete prompt provides the assistant completion used for SFT.

## Ordinary fast and slow examples

For a fast example, Gemini must return:

```text
unsafe
<answer>Deceptive Efficacy</answer>
```

For a slow example, Gemini must return the opening decision, a teacher-written
rule-by-rule trace, and the final active-rule answer:

```text
unsafe
<reasoning>
[Step 1] Content Summary
...

[Step 2] Check Risk Categories
- Deceptive Efficacy: HIT. ...
- Off-Platform Solicitation: NOT HIT. ...

[Step 3] Final Judgment
...
</reasoning>
<answer>Deceptive Efficacy</answer>
```

The pipeline records Gemini's output verbatim after validation. It does not
write or replace the reasoning text locally.

If the content triggers multiple active rules, `<answer>` contains each matched
active-rule title on its own line. A safe result contains only `Safe`.

`fast-slow` remains a supported value in the system template, but the first SFT
dataset uses explicit `fast` and `slow` examples. Confidence-based early exit is
implemented later in the inference runner because it depends on the trained
model's safe/unsafe token probabilities.

## Tool-using examples

Tools are optional per sample. The first tool environment exposes a small set of
deterministic, local functions such as:

- `search_policy(query)`;
- `search_cases(query)`;

The tool descriptions are injected into the complete prompt. Gemini chooses a
tool and arguments. The local runner executes the tool, appends the real tool
result, and calls Gemini again. The loop ends on a final decision or after three
assistant turns.

```text
user content
  -> Gemini tool call
  -> local tool result
  -> Gemini optional second tool call
  -> local tool result
  -> Gemini final decision
```

Gemini must not invent tool results. Every recorded tool message comes from the
local executor. The complete multi-turn trajectory is stored in ms-swift
`messages` format, including tool schemas when required by ms-swift.

### Minimal tool protocol

Version 1 uses sequential tool calls only. A call is represented as:

```json
{"name":"search_cases","arguments":{"query":"guaranteed weight loss","top_k":3}}
```

A successful response is represented as:

```json
{"status":"ok","results":[{"case_id":"case-001","summary":"A fixed result was guaranteed."}]}
```

An execution failure is represented as:

```json
{"status":"error","error":"invalid_arguments"}
```

The exported ms-swift messages use `role: tool_call` and
`role: tool_response`, with each `content` value serialized as a JSON string.
The row-level `tools` field is also serialized as the JSON string required by
ms-swift. The Gemini adapter maps Gemini-native `functionCall` and
`functionResponse` parts to and from this internal representation.

Because calls are sequential and every response immediately follows its call,
version 1 does not add call IDs or parallel-call envelopes. The final decision
is a normal assistant message in the SingGuard output grammar, not a
`final_decision` function call.

The initial tool set is:

- `search_cases` for similar public moderation or regulatory cases;
- `verify_claim` for evidence relevant to efficacy and factual claims;
- `inspect_destination` for links, handles, and obfuscated off-platform
  destinations;

Tool results return evidence, not the final safe/unsafe label. This prevents the
student from learning to copy a label emitted by the environment.

## Output

An ordinary row contains system, user, and assistant messages:

```json
{
  "messages": [
    {"role": "system", "content": "<complete SingGuard system prompt>"},
    {"role": "user", "content": "<conversation content>"},
    {"role": "assistant", "content": "<Gemini completion>"}
  ]
}
```

A tool row additionally contains assistant tool calls and real tool results in
the ms-swift-supported message representation.

## Minimal validation

The local code performs only deterministic checks:

- JSON and message schema are valid;
- output follows the selected fast or slow grammar;
- every final unsafe answer names one or more active rules exactly;
- a safe answer uses `Safe`;
- slow output checks every active rule in prompt order;
- tool names and argument schemas are valid;
- all tool results were produced by the local executor;
- a trajectory contains at most three assistant turns;
- samples contain no disallowed PII and are not exact or near duplicates.

Invalid candidates are written to `rejected.jsonl` with their sanitized
candidate output and specific rejection codes. Retry receives the rejection
reason and may repair the same prompt once.

The report contains observable measurements only: accepted/rejected counts,
format pass rate, active-rule coverage, thinking-mode distribution, tool-use
distribution, duplicate rate, and rejection-code counts. It does not emit a
synthetic DQS score.

## File layout

```text
prompts/
  singguard_guard_v1.txt

data/
  active_policies.jsonl
  content_samples.jsonl

scripts/
  generate_singguard_data.py

outputs/<run>/
  train.jsonl
  rejected.jsonl
  manifest.json
```

Existing provider authentication, request budgets, resumability, progress
reporting, and Gemini structured-response support remain reusable, but the
counterfactual planner and policy-pair data contracts are removed from the
generation path.

## Error handling

- Provider errors use bounded retries and preserve resumable progress.
- Schema or grammar failures are recorded with a precise rejection code.
- Invalid tool calls are not executed and are recorded as rejected.
- Tool execution errors are returned to Gemini once so it can recover; repeated
  failures reject the trajectory.
- Budget exhaustion stops cleanly without publishing a partial run as complete.

## Verification strategy

Tests cover:

1. prompt assembly for one-rule and multi-rule active policies;
2. exact fast and slow output validation;
3. rejection of answers outside the active policy;
4. preservation of Gemini-written slow reasoning;
5. one-, two-, and three-turn tool trajectories with real tool results;
6. invalid tool names, arguments, and turn-limit enforcement;
7. direct ms-swift readability of ordinary and tool-using rows;
8. resume, budget, progress, and sanitized rejection artifacts.

A small pilot of 20-30 examples must be manually reviewed before scaling data
generation.
