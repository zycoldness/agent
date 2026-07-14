# SingGuard English Data Synthesis Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-command, audited Gemini pipeline that can produce a 100-anchor English pilot today and scale unchanged to 2,000 anchors / 4,000 ms-swift SFT rows.

**Architecture:** A deterministic planner fixes quotas, policy pairs, and local oracles before generation. Gemini first realizes content without labels, then a stateless blind verifier classifies shuffled policy views and supplies grounded slow-trace fields; local gates alone accept or reject rows and render the canonical SingGuard format. Open seeds, provider accounting, checkpoints, rejected rows, audit reports, review samples, and final bundles remain separate artifacts.

**Tech Stack:** Python 3.11+, Pydantic v2, PyYAML, httpx, google-genai through the existing `Teacher` abstraction, pytest, ms-swift messages JSONL.

---

## File map

| File | Responsibility |
|---|---|
| `prompts/singguard_guard_v1.txt` | Canonical no-tool Guard system template |
| `prompts/singguard_agent_v1.txt` | Guard core plus bounded JSON tool protocol |
| `prompts/singguard_generator_v1.txt` | English content-realization instruction |
| `prompts/singguard_verifier_v1.txt` | Blind policy-view judging instruction |
| `policies/singguard_rules_v1.yaml` | Eight-domain versioned rule catalog and exceptions |
| `configs/singguard_sources.yaml` | Source URLs, licenses, roles, and research scope |
| `src/risk_agent/singguard_prompts.py` | Prompt loading, substitution, and hashes |
| `src/risk_agent/singguard_synthesis.py` | Contracts, quota planning, policy pairs, provider requests, and candidate conversion |
| `src/risk_agent/singguard_sources.py` | Governed UCI seed download and normalization |
| `src/risk_agent/singguard_quality.py` | Record gates, duplicate checks, corpus report, and review sampler |
| `scripts/fetch_singguard_seeds.py` | Seed-fetch CLI |
| `scripts/generate_singguard_data.py` | Plan/generate/verify/retry/audit/export CLI |
| `tests/test_singguard_prompts.py` | Prompt snapshots and no-double-policy tests |
| `tests/test_singguard_synthesis.py` | Planner, compiler, blind request, and fake-provider tests |
| `tests/test_singguard_sources.py` | Source/license/zip parser tests |
| `tests/test_singguard_quality.py` | Hard-gate, duplicate, report, and sample tests |

## Task 1: Make Gemini JSON schemas configurable

**Files:**
- Modify: `src/risk_agent/teacher.py`
- Modify: `tests/test_teacher.py`

- [ ] **Step 1: Write the failing provider configuration test**

Append a test that constructs `GeminiTeacher` with a generator schema and temperature:

```python
def test_gemini_teacher_accepts_per_task_schema_and_temperature():
    captured = {}
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }

    def generate_content(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            text='{"query":"A natural English ad."}',
            usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=3),
        )

    teacher = GeminiTeacher(
        model="gemini-test",
        client_factory=lambda: SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        ),
        response_schema=schema,
        temperature=0.7,
    )
    teacher.generate({"instruction": "Generate content."})

    assert captured["config"]["response_schema"] == schema
    assert captured["config"]["temperature"] == 0.7
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python -I -m pytest tests/test_teacher.py::test_gemini_teacher_accepts_per_task_schema_and_temperature -q
```

Expected: fail because `GeminiTeacher.__init__` does not accept these arguments.

- [ ] **Step 3: Implement the backward-compatible options**

Add constructor parameters and validate them:

```python
response_schema: Mapping[str, object] | None = None,
temperature: float = 0.2,
```

Store a detached dictionary and finite temperature in `[0, 2]`:

```python
self._response_schema = dict(response_schema or _ACTION_SCHEMA)
self._temperature = _finite_nonnegative(temperature, "temperature", maximum=2)
```

Use them in `generate_content`:

```python
"temperature": self._temperature,
"response_schema": self._response_schema,
```

- [ ] **Step 4: Run provider tests and verify GREEN**

Run `python -I -m pytest tests/test_teacher.py -q` and expect all tests to pass.

- [ ] **Step 5: Commit**

```bash
git add src/risk_agent/teacher.py tests/test_teacher.py
git commit -m "feat: configure Gemini structured responses"
```

## Task 2: Version canonical Guard and Agent prompts

**Files:**
- Create: `prompts/singguard_guard_v1.txt`
- Create: `prompts/singguard_agent_v1.txt`
- Create: `prompts/singguard_generator_v1.txt`
- Create: `prompts/singguard_verifier_v1.txt`
- Create: `src/risk_agent/singguard_prompts.py`
- Create: `tests/test_singguard_prompts.py`
- Modify: `src/risk_agent/singguard.py`
- Modify: `tests/test_singguard.py`

- [ ] **Step 1: Write failing prompt tests**

Test the wished-for API:

```python
from risk_agent.contracts import PolicyRule
from risk_agent.singguard_prompts import prompt_sha256, render_agent_prompt, render_guard_prompt


def test_runtime_policy_replaces_default_taxonomy():
    rule = PolicyRule(rule_id="EFF-001", title="Guaranteed Outcome", text="No guaranteed result.")
    prompt = render_guard_prompt((rule,), thinking_type="slow")
    assert "<thinking_type>slow</thinking_type>" in prompt
    assert "### Guaranteed Outcome" in prompt
    assert "A. Sexual Content Risk" not in prompt
    assert len(prompt_sha256("guard")) == 64


def test_agent_prompt_extends_guard_core_with_bounded_tools():
    rule = PolicyRule(rule_id="EFF-001", title="Guaranteed Outcome", text="No guaranteed result.")
    prompt = render_agent_prompt((rule,))
    assert "get_rule_detail" in prompt
    assert "inspect_evidence" in prompt
    assert "At most three assistant turns" in prompt
    assert '{"tool":"...","arguments":{...}}' in prompt
```

- [ ] **Step 2: Run prompt tests and verify RED**

Run `python -I -m pytest tests/test_singguard_prompts.py -q` and expect import failure for `singguard_prompts`.

- [ ] **Step 3: Add exact versioned templates and renderer**

Use explicit placeholders only:

```text
{{THINKING_TYPE}}
{{ACTIVE_POLICY}}
```

`load_prompt(name)` reads a fixed allowlisted path relative to the repository root, rejects unresolved placeholders, and `prompt_sha256(name)` hashes the raw template bytes. `render_guard_prompt` renders policy titles, rule text, and exceptions. `render_agent_prompt` renders the same active policy plus tool instructions. No user-controlled filesystem path is accepted.

- [ ] **Step 4: Route existing SFT rendering through `render_guard_prompt`**

Replace the private `_system_prompt` body in `singguard.py` with:

```python
from risk_agent.singguard_prompts import render_guard_prompt


def _system_prompt(example: SingGuardExample) -> str:
    return render_guard_prompt(
        example.policy.active_policy,
        thinking_type=example.thinking_type,
    )
```

Keep assistant output grammar unchanged.

- [ ] **Step 5: Run prompt and renderer tests**

Run `python -I -m pytest tests/test_singguard_prompts.py tests/test_singguard.py -q`; expect all tests to pass.

- [ ] **Step 6: Commit**

```bash
git add prompts src/risk_agent/singguard_prompts.py src/risk_agent/singguard.py tests/test_singguard_prompts.py tests/test_singguard.py
git commit -m "feat: version SingGuard prompt contracts"
```

## Task 3: Add the eight-domain rule catalog, contracts, quota planner, and policy compiler

**Files:**
- Create: `policies/singguard_rules_v1.yaml`
- Create: `src/risk_agent/singguard_synthesis.py`
- Create: `tests/test_singguard_synthesis.py`
- Modify: `src/risk_agent/singguard.py`

- [ ] **Step 1: Write failing contract and planner tests**

Cover a 100-anchor pilot and 2,000-anchor full plan:

```python
def test_planner_is_deterministic_and_balances_pilot_transitions():
    first = plan_blueprints(100, seed=17)
    second = plan_blueprints(100, seed=17)
    assert first == second
    assert Counter(item.transition for item in first) == {
        "unsafe_to_unsafe": 25,
        "unsafe_to_safe": 25,
        "safe_to_unsafe": 25,
        "safe_to_safe": 25,
    }
    domain_counts = Counter(item.risk_domain for item in first)
    assert max(domain_counts.values()) - min(domain_counts.values()) <= 1


def test_full_plan_has_exact_domain_and_style_quotas():
    plan = plan_blueprints(2000, seed=42)
    assert set(Counter(item.risk_domain for item in plan).values()) == {250}
    assert Counter(item.input_style for item in plan) == EXPECTED_STYLE_COUNTS
```

Add a compiler test that asserts the actual ordered label pair for every transition and that each policy has 3–8 unique active rules.

- [ ] **Step 2: Run synthesis tests and verify RED**

Run `python -I -m pytest tests/test_singguard_synthesis.py -q`; expect import failure.

- [ ] **Step 3: Define strict contracts**

In `singguard_synthesis.py`, add frozen Pydantic models with `extra="forbid"`:

```python
class AnchorBlueprint(BaseModel):
    anchor_id: str
    family_id: str
    risk_domain: str
    subtype: str
    input_style: str
    tone: str
    length_bin: str
    difficulty: str
    conversation_type: Literal["query", "query_response"]
    transition: PolicyTransition
    transformation: Literal["add", "remove", "narrow", "broaden", "rewrite", "merge", "exception"]
    source_id: str | None = None


class PolicyPair(BaseModel):
    before: PolicyView
    after: PolicyView
    before_annotation: SingGuardAnnotation
    after_annotation: SingGuardAnnotation
```

Add `stage: Literal["before", "after"] = "after"` to `PolicyView` for backward compatibility with the smoke fixture.

- [ ] **Step 4: Add the rule catalog**

Create one primary rule per domain plus specific exception text and at least eight distractor rules. Every record has `rule_id`, `domain`, `title`, `text`, `exceptions`, and `priority`. `load_rule_catalog()` validates unique IDs/titles and exactly the eight required domains.

- [ ] **Step 5: Implement deterministic quota planning**

Build the complete Cartesian quota schedule with balanced round-robin assignment, then apply a seeded shuffle. Use integer quotas for the 2,000-anchor plan and largest-remainder allocation for other valid pilot sizes. Reject counts below 32 because they cannot exercise the required pilot cells.

- [ ] **Step 6: Implement policy pair compilation**

Compile the ordered transition as follows:

```python
expected = {
    "unsafe_to_unsafe": ("unsafe", "unsafe"),
    "unsafe_to_safe": ("unsafe", "safe"),
    "safe_to_unsafe": ("safe", "unsafe"),
    "safe_to_safe": ("safe", "safe"),
}[blueprint.transition]
```

Presence, removal, narrowed scope, broadened scope, equivalent rewrite, merged rule, or matching exception determines whether the target rule is active for the intended fact pattern. Fill each view to its planned size with deterministic distractors. Unsafe annotations name only the target active title; safe annotations name no rule.

- [ ] **Step 7: Run tests and commit**

Run `python -I -m pytest tests/test_singguard_synthesis.py tests/test_singguard.py -q`, then commit:

```bash
git add policies/singguard_rules_v1.yaml src/risk_agent/singguard_synthesis.py src/risk_agent/singguard.py tests/test_singguard_synthesis.py
git commit -m "feat: plan dynamic-policy SingGuard anchors"
```

## Task 4: Fetch and normalize the first governed open seeds

**Files:**
- Create: `configs/singguard_sources.yaml`
- Create: `src/risk_agent/singguard_sources.py`
- Create: `scripts/fetch_singguard_seeds.py`
- Create: `tests/test_singguard_sources.py`

- [ ] **Step 1: Write failing source tests with in-memory ZIP files**

Use `httpx.MockTransport` and `zipfile.ZipFile` to represent the UCI SMS and YouTube archives. Assert normalized rows contain:

```python
{
    "source": "uci_sms_spam",
    "source_id": "...",
    "provenance_url": "...",
    "license": "CC-BY-4.0",
    "usage_scope": "research_only",
    "source_role": "style_seed",
    "text": "...",
    "source_label": "spam",
    "content_hash": "...",
    "retrieved_at": "...",
    "adapter_version": "v1",
}
```

Test that unknown licenses, duplicate `(source, source_id)`, oversized records, malformed ZIP paths, and non-UTF-8/Latin-1 decode failures are rejected.

- [ ] **Step 2: Run source tests and verify RED**

Run `python -I -m pytest tests/test_singguard_sources.py -q`; expect import failure.

- [ ] **Step 3: Add the versioned source catalog**

List all sources from the design, but enable only the two UCI adapters in the first fetch command. Every entry includes `dataset_url`, `artifact_url`, `license`, `source_role`, `usage_scope`, `enabled`, and adapter name. Entries with NC material set `usage_scope: research_only`.

- [ ] **Step 4: Implement bounded UCI adapters and atomic output**

Use `httpx.Client`, a finite byte cap, ZIP member allowlists, the standard `csv` module, SHA-256 IDs, and atomic JSONL/manifest writes. Do not extract archives to disk. The CLI is:

```bash
python scripts/fetch_singguard_seeds.py \
  configs/singguard_sources.yaml \
  outputs/singguard-seeds
```

- [ ] **Step 5: Run source tests and commit**

Run `python -I -m pytest tests/test_singguard_sources.py -q`, then commit:

```bash
git add configs/singguard_sources.yaml src/risk_agent/singguard_sources.py scripts/fetch_singguard_seeds.py tests/test_singguard_sources.py
git commit -m "feat: import governed SingGuard seed data"
```

## Task 5: Generate content and blind-verifier traces through strict schemas

**Files:**
- Modify: `src/risk_agent/singguard_synthesis.py`
- Modify: `tests/test_singguard_synthesis.py`

- [ ] **Step 1: Write failing generator/verifier tests**

Test structured fakes through the real request builders. The verifier-request assertion is mandatory:

```python
request_text = json.dumps(build_verifier_request(blueprint, pair, content), sort_keys=True)
for forbidden in ("oracle", "transition", "unsafe_to_safe", "source_label", "intended_facts"):
    assert forbidden not in request_text
assert [view["opaque_id"] for view in request["policy_views"]] != ["before", "after"]
```

Test generator replies with unknown keys, empty query, wrong declared style, or meta-language are rejected. Test verifier replies with missing view IDs, duplicate IDs, rule titles outside the presented policy, nonliteral evidence, or confidence outside `[0, 1]` are rejected.

- [ ] **Step 2: Run focused tests and verify RED**

Run the named tests and expect missing request builders/contracts.

- [ ] **Step 3: Add strict response contracts and JSON schemas**

Define:

```python
class GeneratedContent(BaseModel):
    query: str
    response: str | None = None
    style: str
    risk_cues: tuple[str, ...] = ()
    benign_cues: tuple[str, ...] = ()


class VerifiedView(BaseModel):
    opaque_id: str
    label: Literal["safe", "unsafe"]
    rule_title: str | None
    evidence_quote: str | None
    confidence: float = Field(ge=0, le=1)
    ambiguous: bool
    summary: str | None = None
    checks: tuple[RuleCheck, ...] = ()


class VerifierResult(BaseModel):
    views: tuple[VerifiedView, ...]
    style: str
    naturalness: int = Field(ge=1, le=5)
    template_like: bool
    issues: tuple[str, ...] = ()
```

Export provider-compatible JSON schemas from `model_json_schema()`.

- [ ] **Step 4: Build isolated requests**

The generator request contains the generator prompt version, blueprint style constraints, intended facts, prohibited details, and optional seed text. It contains no final messages row.

The verifier request contains only content, randomly mapped opaque policy views, thinking modes, and the verifier prompt. It omits blueprint intent, source label, transition, and oracle. Keep the opaque mapping only in local memory.

- [ ] **Step 5: Convert an agreed verdict into two source examples**

Compare both verifier views to the local annotations. For slow views, use verifier summary/checks only after title/order/evidence validation. For fast views, discard any supplied reasoning. Return `SingGuardExample` records with the same `family_id`/split group.

- [ ] **Step 6: Run tests and commit**

Run `python -I -m pytest tests/test_singguard_synthesis.py tests/test_teacher.py -q`, then commit:

```bash
git add src/risk_agent/singguard_synthesis.py tests/test_singguard_synthesis.py
git commit -m "feat: generate and blind-verify SingGuard content"
```

## Task 6: Add deterministic quality gates and audit artifacts

**Files:**
- Create: `src/risk_agent/singguard_quality.py`
- Create: `tests/test_singguard_quality.py`

- [ ] **Step 1: Write failing hard-gate tests**

Cover:

- unsafe quote absent from query/response;
- safe verifier view containing a rule;
- slow checks missing, reordered, or containing a safe `HIT`;
- confidence below 0.85, ambiguity, naturalness below four, or template-like content;
- non-English content beyond the configured allowance;
- real-looking phone/email versus `555-01xx`, `example.test`, and synthetic handles;
- exact duplicate, 5-gram Jaccard `>=0.82`, and generated-to-source similarity `>=0.75`;
- generation meta-language and high-fidelity operational instruction markers.

- [ ] **Step 2: Run quality tests and verify RED**

Run `python -I -m pytest tests/test_singguard_quality.py -q`; expect import failure.

- [ ] **Step 3: Implement stable rejection codes**

Use a frozen result rather than raising for expected candidate failures:

```python
class GateResult(BaseModel):
    accepted: bool
    codes: tuple[str, ...]


def gate_candidate(...) -> GateResult:
    ...
```

Codes include `schema`, `non_english`, `meta_language`, `pii`, `oracle_disagreement`, `inactive_rule`, `evidence_not_literal`, `ambiguous`, `low_confidence`, `unnatural`, `template_like`, `slow_trace`, `exact_duplicate`, `near_duplicate`, and `source_copy`.

- [ ] **Step 4: Implement corpus reporting and deterministic review sampling**

`build_quality_report` returns counts and rates by domain, subtype, style, tone, length, difficulty, transition, transformation, label, stage, and thinking type; contingency empty cells; source/license counts; rejection codes; duplicate statistics; verifier agreement; and a DQS with explicit component scores. `stratified_review_sample(..., fraction=0.10, seed=...)` selects whole anchors and is repeatable.

- [ ] **Step 5: Run quality tests and commit**

Run `python -I -m pytest tests/test_singguard_quality.py -q`, then commit:

```bash
git add src/risk_agent/singguard_quality.py tests/test_singguard_quality.py
git commit -m "feat: gate and audit SingGuard synthesis"
```

## Task 7: Orchestrate retries, checkpoints, audit, and ms-swift export

**Files:**
- Create: `scripts/generate_singguard_data.py`
- Modify: `src/risk_agent/singguard_synthesis.py`
- Modify: `tests/test_singguard_synthesis.py`
- Modify: `src/risk_agent/ms_swift_singguard.py`
- Modify: `tests/test_singguard_bundle.py`

- [ ] **Step 1: Write a failing end-to-end fake-provider pilot**

Inject generator and verifier `CallableTeacher` instances. Generate at least 32 planned anchors with one rejected first attempt and a successful retry. Assert the output directory contains:

```text
plan.jsonl
accepted.jsonl
rejected.jsonl
quality_report.json
review_sample.jsonl
manifest.json
ms_swift/train.jsonl
ms_swift/dev.jsonl
ms_swift/holdout.jsonl
ms_swift/manifest.json
```

Assert every accepted anchor produces two rows, family pairs never cross splits, rejected attempts remain auditable, and no provider request or credential is persisted.

- [ ] **Step 2: Run the end-to-end test and verify RED**

Run the exact test and expect the orchestration API or CLI to be missing.

- [ ] **Step 3: Implement `run_singguard_batch`**

The API receives plan, optional seeds, two `Teacher` instances, shared `TeacherBudget`, output directory, maximum two retries, and thresholds. It writes an atomic checkpoint after each anchor. Existing output or input-overwrite paths fail before any provider request.

Expected candidate failures append a rejected-attempt record and retry the same quota cell. Provider/budget failures stop with `status=incomplete`; they do not create a final ms-swift directory.

- [ ] **Step 4: Enforce pilot gates before export**

For `--pilot`, require at least 80% first-pass agreement, at least 90% final acceptance, no more than 5% near-duplicate rejection, all planned cells represented, and DQS at least 90. Always write audit artifacts; create `ms_swift/` only when the gates pass.

- [ ] **Step 5: Implement the real CLI**

Expose:

```bash
python scripts/generate_singguard_data.py \
  outputs/singguard-pilot \
  --anchors 100 \
  --seed 42 \
  --generator-model "$GEMINI_GENERATOR_MODEL" \
  --verifier-model "$GEMINI_VERIFIER_MODEL" \
  --seeds outputs/singguard-seeds/seeds.jsonl \
  --allow-external-data \
  --pilot
```

Also expose provider retries, request timeout, max requests, max output tokens, optional cost prices/cap, and `--plan-only`. Credentials remain environment-only.

- [ ] **Step 6: Run end-to-end, bundle, and CLI tests**

Run:

```bash
python -I -m pytest tests/test_singguard_synthesis.py tests/test_singguard_quality.py tests/test_singguard_bundle.py -q
```

Expected: all pass without network access.

- [ ] **Step 7: Commit**

```bash
git add scripts/generate_singguard_data.py src/risk_agent/singguard_synthesis.py src/risk_agent/ms_swift_singguard.py tests/test_singguard_synthesis.py tests/test_singguard_bundle.py
git commit -m "feat: orchestrate SingGuard pilot generation"
```

## Task 8: Document and verify the same-day pilot run

**Files:**
- Modify: `README.md`
- Create: `docs/singguard-data-runbook.md`
- Modify: `tests/test_training_scripts.py`

- [ ] **Step 1: Add a failing documentation command smoke test**

Extract or invoke the CLI help and plan-only commands used by the runbook. Assert both exit zero and that plan-only produces exactly 100 blueprints without requiring Gemini credentials.

- [ ] **Step 2: Document four copy-paste commands**

The runbook contains:

1. install: `pip install -e '.[dev,teacher]'`;
2. fetch governed seeds;
3. run `--plan-only` and inspect quotas;
4. run the real 100-anchor pilot, inspect `quality_report.json` and `review_sample.jsonl`, then run the 2,000-anchor batch only after approval.

It explicitly says the first batch is English, text-only, research-only, Guard-only, and not production evidence.

- [ ] **Step 3: Run focused and full tests**

Run:

```bash
python -I -m pytest tests/test_singguard_prompts.py tests/test_singguard_sources.py tests/test_singguard_synthesis.py tests/test_singguard_quality.py tests/test_singguard_bundle.py tests/test_teacher.py -q
python -I -m pytest -q
git diff --check
```

Expected: zero failures and no whitespace errors.

- [ ] **Step 4: Run a local no-network pilot simulation**

Use the fake-provider fixture to generate all artifacts. Parse every JSONL line, compare manifest hashes/counts, and inspect at least one fast unsafe, fast safe, slow unsafe, and slow safe row.

- [ ] **Step 5: Commit and push**

```bash
git add README.md docs/singguard-data-runbook.md tests/test_training_scripts.py
git commit -m "docs: add SingGuard data pilot runbook"
git push origin feature/singguard-risk-agent
```

## Task 9: Produce the first real pilot data

**Files:**
- Generated outside git: `outputs/singguard-seeds/`
- Generated outside git: `outputs/singguard-pilot/`

- [ ] **Step 1: Verify credentials without printing them**

Run:

```bash
test -n "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" || test "${GOOGLE_GENAI_USE_VERTEXAI:-}" = true
```

Expected: exit zero without secret output.

- [ ] **Step 2: Fetch and review open seeds**

Run the governed seed command, then verify manifest license counts, hashes, and record totals. Do not proceed if any enabled source has an unknown license.

- [ ] **Step 3: Generate the 100-anchor real pilot**

Run the runbook command with finite `--max-requests` and cost/time bounds. Preserve incomplete checkpoints on provider failure; resume only with matching plan/prompt/source hashes.

- [ ] **Step 4: Apply the review gate**

Review `quality_report.json`, the 10% accepted sample, every rejection-code group, and all borderline/exception examples. If any pilot stop gate fails, revise prompt/rule/planner inputs and create a new dataset version rather than editing rows by hand.

- [ ] **Step 5: Approve or stop full generation**

Only after the pilot is approved, run `--anchors 2000` into a new output directory. The final 4,000-row bundle remains outside git; commit only source manifests, prompt/policy versions, the quality report, and a data card when their contents are safe to publish.
