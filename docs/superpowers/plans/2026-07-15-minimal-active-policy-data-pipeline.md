# Minimal Active-Policy Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace counterfactual SingGuard synthesis with a small pipeline that gives Gemini a complete active-policy prompt and records either its completion or a real local-tool trajectory as ms-swift SFT data.

**Architecture:** `singguard.py` owns contracts, prompt assembly, completion validation, and row rendering. `singguard_tools.py` owns deterministic local tools and the compact protocol. `singguard_generation.py` owns the provider-neutral loop, Gemini adapter, and batch artifacts. The CLI only loads files/configuration and invokes the core.

**Tech Stack:** Python 3.11, Pydantic 2, google-genai, ms-swift agent format, pytest.

---

## File map

- `src/risk_agent/singguard.py`: active policies, samples, prompt assembly, fast/slow validation, ms-swift rows.
- `src/risk_agent/singguard_tools.py`: tool schemas, JSONL stores, deterministic execution.
- `src/risk_agent/singguard_generation.py`: sequential tool loop, Gemini adapter, resume and artifacts.
- `scripts/generate_singguard_data.py`: minimal CLI and progress.
- `data/active_policies.jsonl`, `data/content_samples.jsonl`, `data/tool_env/*.jsonl`: reviewed inputs.
- `tests/test_singguard.py`, `tests/test_singguard_tools.py`, `tests/test_singguard_generation.py`: focused tests.

### Task 1: Model one active-policy set per sample

**Files:**
- Modify: `src/risk_agent/singguard.py`
- Modify: `tests/test_singguard.py`
- Create: `data/active_policies.jsonl`
- Create: `data/content_samples.jsonl`

- [ ] **Step 1: Write the failing prompt test**

```python
def test_build_prompt_injects_all_active_rules_once() -> None:
    policy = ActivePolicy(
        policy_id="commerce-v1",
        rules=(
            PolicyRule(rule_id="EFF-001", title="Deceptive Efficacy", text="No guaranteed result."),
            PolicyRule(rule_id="LEAD-001", title="Off-Platform Solicitation", text="No private-channel redirection."),
        ),
    )
    sample = ModerationSample(
        sample_id="sample-1",
        policy_id="commerce-v1",
        thinking_type="slow",
        query="Guaranteed to lose ten pounds in seven days.",
    )
    messages = build_initial_messages(policy, sample)
    assert [message.role for message in messages] == ["system", "user"]
    assert messages[0].content.count("### Deceptive Efficacy") == 1
    assert messages[0].content.count("### Off-Platform Solicitation") == 1
    assert "<thinking_type>slow</thinking_type>" in messages[0].content
```

Also cover query-response input, duplicate rule titles/IDs, and an invalid thinking mode.

- [ ] **Step 2: Verify the test fails**

Run `python -m pytest tests/test_singguard.py -q`.
Expected: missing `ActivePolicy`, `ModerationSample`, and `build_initial_messages`.

- [ ] **Step 3: Implement minimal contracts**

```python
ThinkingType = Literal["fast", "slow"]

class ActivePolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    policy_id: str = Field(min_length=1)
    rules: tuple[PolicyRule, ...] = Field(min_length=1)

class ModerationSample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    sample_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    thinking_type: ThinkingType
    query: str = Field(min_length=1)
    response: str | None = None
    tool_names: tuple[str, ...] = ()

class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    role: Literal["system", "user", "assistant", "tool_call", "tool_response"]
    content: str = Field(min_length=1)
```

Use the existing versioned `render_guard_prompt()`; do not add labels, transitions, or local reasoning.

- [ ] **Step 4: Add reviewed fixtures**

Add two policy sets and six samples covering one/many active rules, fast/slow, query-only, and query-response. Input samples contain no generated answer.

- [ ] **Step 5: Test and commit**

```bash
python -m pytest tests/test_singguard.py -q
git add src/risk_agent/singguard.py tests/test_singguard.py data/active_policies.jsonl data/content_samples.jsonl
git commit -m "refactor: model one active policy per sample"
```

### Task 2: Validate Gemini output without rewriting it

**Files:**
- Modify: `src/risk_agent/singguard.py`
- Modify: `tests/test_singguard.py`

- [ ] **Step 1: Write failing fast/slow tests**

```python
def test_fast_rejects_inactive_answer() -> None:
    with pytest.raises(ValueError, match="active policy"):
        validate_completion(
            "unsafe\\n<answer>Unknown Rule</answer>",
            thinking_type="fast",
            active_titles=("Deceptive Efficacy",),
        )

def test_slow_checks_each_rule_in_order() -> None:
    completion = (
        "unsafe\\n<reasoning>\\n[Step 1] Content Summary\\nA claim.\\n\\n"
        "[Step 2] Check Risk Categories\\n"
        "- Deceptive Efficacy: HIT. Guaranteed result.\\n"
        "- Off-Platform Solicitation: NOT HIT. No destination.\\n\\n"
        "[Step 3] Final Judgment\\nThe first rule is violated.\\n"
        "</reasoning>\\n<answer>Deceptive Efficacy</answer>"
    )
    parsed = validate_completion(
        completion,
        thinking_type="slow",
        active_titles=("Deceptive Efficacy", "Off-Platform Solicitation"),
    )
    assert parsed.answers == ("Deceptive Efficacy",)
```

Add safe, multiple-answer, duplicate-answer, missing-step, and reordered-rule tests.

- [ ] **Step 2: Verify failure**

Run `python -m pytest tests/test_singguard.py -q`.
Expected: `validate_completion` missing.

- [ ] **Step 3: Implement grammar checks**

Require exact line-one `safe|unsafe`, one answer block, active titles only, `Safe` alone for safe, no reasoning in fast mode, and the three named sections plus ordered active-rule checks in slow mode. Return parsed metadata while preserving Gemini's original completion byte-for-byte.

- [ ] **Step 4: Render ms-swift rows**

```python
def render_training_row(initial, completion, *, tools_json=None, trajectory=()):
    messages = [message.model_dump(mode="json") for message in (*initial, *trajectory)]
    messages.append({"role": "assistant", "content": completion})
    row = {"messages": messages}
    if tools_json is not None:
        row["tools"] = tools_json
    return row
```

- [ ] **Step 5: Test and commit**

```bash
python -m pytest tests/test_singguard.py -q
git add src/risk_agent/singguard.py tests/test_singguard.py
git commit -m "feat: validate Gemini SingGuard completions"
```

### Task 3: Build the compact local tool environment

**Files:**
- Create: `src/risk_agent/singguard_tools.py`
- Create: `tests/test_singguard_tools.py`
- Create: `data/tool_env/cases.jsonl`
- Create: `data/tool_env/claim_evidence.jsonl`
- Create: `data/tool_env/destinations.jsonl`
- Create: `data/tool_env/content_history.jsonl`

- [ ] **Step 1: Write failing protocol tests**

```python
def test_search_cases_returns_compact_json(tmp_path: Path) -> None:
    write_tool_stores(tmp_path, cases=[
        {"case_id": "case-1", "text": "Guaranteed weight loss", "summary": "Fixed result claim"}
    ])
    env = ToolEnvironment.load(tmp_path)
    result = env.execute(ToolCall(
        name="search_cases",
        arguments={"query": "guaranteed weight loss", "top_k": 1},
    ))
    assert result.status == "ok"
    assert json.loads(result.to_content())["results"][0]["case_id"] == "case-1"
```

Also reject unknown tools, missing arguments, `top_k > 5`, invalid content IDs, and repeated nondeterministic output.

- [ ] **Step 2: Verify import failure**

Run `python -m pytest tests/test_singguard_tools.py -q`.
Expected: module import failure.

- [ ] **Step 3: Implement the protocol**

```python
class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: Literal["search_cases", "verify_claim", "inspect_destination", "get_content_context"]
    arguments: dict[str, object]

class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    status: Literal["ok", "error"]
    payload: dict[str, object]

    def to_content(self) -> str:
        return json.dumps(
            {"status": self.status, **self.payload},
            ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        )
```

Define all four function declarations once and export the row-level ms-swift `tools` JSON string.

- [ ] **Step 4: Implement deterministic lookups**

Use normalized-token overlap plus stable ID tie-breaking for cases/evidence, normalized exact lookup for destinations, and exact `content_id` lookup for history. Return evidence and source IDs, never a final safe/unsafe label.

- [ ] **Step 5: Add tool data**

Add at least eight case rows, eight claim-evidence rows, eight destinations, and six histories. Include success, empty, benign-exception, and conflicting-evidence records with `source_type` and `source_id`.

- [ ] **Step 6: Test and commit**

```bash
python -m pytest tests/test_singguard_tools.py -q
git add src/risk_agent/singguard_tools.py tests/test_singguard_tools.py data/tool_env
git commit -m "feat: add deterministic risk-agent tools"
```

### Task 4: Execute sequential Gemini tool calls

**Files:**
- Create: `src/risk_agent/singguard_generation.py`
- Create: `tests/test_singguard_generation.py`
- Modify: `src/risk_agent/teacher.py`
- Modify: `tests/test_teacher.py`

- [ ] **Step 1: Write failing trajectory tests**

```python
def test_two_tools_then_final(policy, sample, environment) -> None:
    provider = FakeAgentProvider(turns=[
        AgentTurn(tool_call=ToolCall(name="inspect_destination", arguments={"indicator": "w-h-a-t-s-a-p-p:user123"})),
        AgentTurn(tool_call=ToolCall(name="get_content_context", arguments={"content_id": "content-1"})),
        AgentTurn(content="unsafe\\n<answer>Off-Platform Solicitation</answer>"),
    ])
    generated = generate_example(policy, sample, provider, environment, max_tool_calls=2)
    assert [message.role for message in generated.trajectory] == [
        "tool_call", "tool_response", "tool_call", "tool_response"
    ]
```

Also cover no tools, unavailable tools, invalid arguments, a third tool attempt, provider failure, and tool error recovery.

- [ ] **Step 2: Verify import failure**

Run `python -m pytest tests/test_singguard_generation.py -q`.
Expected: module import failure.

- [ ] **Step 3: Implement the provider-neutral loop**

```python
class AgentTurn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    content: str | None = None
    tool_call: ToolCall | None = None

class AgentProvider(Protocol):
    def start(self, *, system: str, user: str, tool_specs: tuple[dict[str, object], ...]) -> AgentTurn: ...
    def continue_with_tool_result(self, *, call: ToolCall, result: ToolResult) -> AgentTurn: ...
```

Require exactly one action per turn. Execute at most two allowlisted calls, append real `tool_call/tool_response` messages, and validate the final completion.

- [ ] **Step 4: Share Gemini client creation**

Extract the current Vertex/API-key/client-factory logic from `GeminiTeacher._new_client()` into `create_gemini_client()`. Prove existing JSON teacher tests remain unchanged.

- [ ] **Step 5: Implement Gemini native function calling**

Create a chat with `system_instruction`, function declarations, temperature, and max tokens. Map exactly one `response.function_calls` entry to the compact `ToolCall`; otherwise use `response.text`. Return results with:

```python
types.Part.from_function_response(
    name=call.name,
    response={"status": result.status, **result.payload},
)
```

Reject parallel calls in version 1. Account for every turn with the existing budget types and retry only the current provider request.

- [ ] **Step 6: Test and commit**

```bash
python -m pytest tests/test_teacher.py tests/test_singguard_generation.py -q
git add src/risk_agent/teacher.py src/risk_agent/singguard_generation.py tests/test_teacher.py tests/test_singguard_generation.py
git commit -m "feat: record real Gemini tool trajectories"
```

### Task 5: Replace the CLI and batch artifacts

**Files:**
- Rewrite: `scripts/generate_singguard_data.py`
- Modify: `tests/test_training_scripts.py`
- Modify: `tests/test_singguard_generation.py`

- [ ] **Step 1: Write failing CLI tests**

Require:

```bash
python scripts/generate_singguard_data.py \
  data/active_policies.jsonl data/content_samples.jsonl \
  outputs/singguard-pilot-v2 \
  --model gemini-model-name --tool-env data/tool_env --max-tool-calls 2
```

A fake batch must produce two accepted rows and one rejected row with a specific code such as `inactive_answer`.

- [ ] **Step 2: Verify old CLI failure**

Run `python -m pytest tests/test_training_scripts.py tests/test_singguard_generation.py -q`.
Expected: parsing/artifact failures.

- [ ] **Step 3: Implement loading and generation**

Load policies by unique ID, validate every sample reference, create one provider, and run each sample. Retain provider attempts, timeout, request/token/cost budgets, resume, and progress. Remove anchors, plan-only, seed assignment, counterfactual retries, and pilot DQS.

- [ ] **Step 4: Write artifacts atomically after every sample**

Write `train.jsonl`, sanitized `rejected.jsonl`, `checkpoint.json`, and `manifest.json`. Fingerprint policies, samples, tools, and prompt. Report only observable counts/distributions.

If completion validation fails, make one repair request containing the original complete prompt, the rejected completion, and the specific validation code. Validate the repaired completion once; if it still fails, write both attempts to the sanitized rejection record.

- [ ] **Step 5: Test and commit**

```bash
python -m pytest tests/test_training_scripts.py tests/test_singguard_generation.py -q
git add scripts/generate_singguard_data.py tests/test_training_scripts.py tests/test_singguard_generation.py
git commit -m "refactor: simplify SingGuard generation CLI"
```

### Task 6: Remove counterfactual code and update documentation

**Files:**
- Delete: `src/risk_agent/singguard_synthesis.py`
- Delete: `src/risk_agent/singguard_quality.py`
- Delete: `src/risk_agent/ms_swift_singguard.py`
- Delete: `tests/test_singguard_synthesis.py`
- Delete: `tests/test_singguard_quality.py`
- Delete: `tests/test_singguard_bundle.py`
- Modify: `README.md`
- Modify: `docs/singguard-data-runbook.md`
- Modify: `data/README.md`

- [ ] **Step 1: Search live imports**

Run `rg -n "singguard_synthesis|singguard_quality|ms_swift_singguard|plan_blueprints|compile_policy_pair" .`.
Expected: only obsolete files and old docs after the CLI replacement.

- [ ] **Step 2: Delete only obsolete counterfactual files**

Keep guard prompts, teacher client, public-data ingestion, crawlers, and training/RL/OPSD code.

- [ ] **Step 3: Publish one quick start**

Document API-key and Vertex credentials, the new command, compact protocol, output files, resume behavior, and manual review of a 20-30 sample pilot.

- [ ] **Step 4: Verify everything**

```bash
python -m pytest -q
python scripts/generate_singguard_data.py --help
git diff --check
```

Expected: tests pass, help exits zero, and diff check is silent.

- [ ] **Step 5: Commit**

```bash
git add -A src/risk_agent tests README.md docs/singguard-data-runbook.md data/README.md
git commit -m "docs: publish minimal active-policy workflow"
```

### Task 7: Run the server quality gate

**Files:**
- No repository changes unless the smoke test finds a defect.

- [ ] **Step 1: Install and test**

```bash
git pull
pip install -e '.[dev,teacher]'
python -m pytest -q
```

Expected: tests pass without changing the existing torch/vLLM/ms-swift stack.

- [ ] **Step 2: Generate 20-30 real Gemini samples**

Include at least five no-tool, five one-tool, and five two-tool examples.

- [ ] **Step 3: Inspect every row**

Verify the complete prompt, active rules, tool necessity, real tool results, ordered slow checks, and active final answers.

- [ ] **Step 4: Scale or stop**

Scale only with zero invalid trajectories, zero inactive answers, and no repeated systematic style/reasoning defect. Preserve artifacts and fix prompts/tool data otherwise.


