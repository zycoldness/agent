# SingGuard Query Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a governed, diverse 2,000-anchor English query corpus that feeds the existing active-policy SingGuard trace generator.

**Architecture:** A deterministic planner fixes label, active rule, content form, difficulty, source mode, and conversation shape before any provider call. Gemini realizes batches of at most four blueprints; local gates reject PII, meta-language, source copying, and duplicates before the existing blind trace call validates the hidden label and rule answers. Open datasets are streamed at pinned revisions and normalized through the existing seed contract.

**Tech Stack:** Python 3.11+, Pydantic v2, google-genai through the existing `GeminiTeacher`, Hugging Face `datasets` as an optional source dependency, httpx, pytest, ms-swift JSONL.

---

## File map

- Create `src/risk_agent/singguard_query_generation.py`: blueprint contracts, quota planning, content gates, Gemini request rendering, batch orchestration, checkpointing, and query-stage artifacts.
- Create `scripts/generate_singguard_queries.py`: one CLI for the 100/500/2000 query stages and progress output.
- Create `prompts/singguard_query_generator_v1.txt`: content-only Gemini contract.
- Create `tests/test_singguard_query_generation.py`: planner, gates, provider, resume, artifacts, and CLI-independent batch tests.
- Modify `src/risk_agent/singguard_sources.py`: pinned streaming adapters for Aegis, Civil Comments, and ESCI while retaining the two UCI adapters.
- Modify `configs/singguard_sources.yaml`: enable the five approved permissive sources with pinned revisions and record caps.
- Modify `pyproject.toml`: add the optional `sources` dependency.
- Modify `tests/test_singguard_sources.py`: source-row normalization, split enforcement, revision propagation, and dependency failure tests.
- Modify `src/risk_agent/singguard_generation.py`: optional joined final review sample only; do not change moderation or tool behavior.
- Modify `scripts/generate_singguard_data.py`: accept query metadata for the final joined review artifact.
- Modify `tests/test_singguard_generation.py`: prove review joining does not alter train rows or checkpoints.
- Modify `README.md`, `data/README.md`, and `docs/singguard-data-runbook.md`: exact pilot, scale, inspection, and trace commands.

### Task 1: Add immutable query blueprints and deterministic quota planning

**Files:**
- Create: `src/risk_agent/singguard_query_generation.py`
- Create: `tests/test_singguard_query_generation.py`

- [ ] **Step 1: Write failing contract and quota tests**

```python
import json
from pathlib import Path

import pytest


def _seed_records(count: int):
    from risk_agent.singguard_sources import SeedRecord

    return tuple(
        SeedRecord(
            source="fixture", source_id=f"seed-{index:04d}",
            provenance_url="https://example.test/dataset", license="CC-BY-4.0",
            usage_scope="research_only", source_role="style_seed",
            text=f"Natural source style record number {index}", source_label=None,
            content_hash=f"{index:064x}", retrieved_at="2026-07-15T00:00:00Z",
        )
        for index in range(count)
    )


def test_plan_2000_has_exact_release_quotas() -> None:
    from pathlib import Path
    from risk_agent.singguard_generation import load_active_policies
    from risk_agent.singguard_query_generation import plan_blueprints

    policies = load_active_policies(Path("data/active_policies.jsonl"))
    plan = plan_blueprints(
        policies, count=2000, seed=42, seed_records=_seed_records(600)
    )

    assert len(plan) == 2000
    assert len({item.blueprint_id for item in plan}) == 2000
    assert len({item.family_id for item in plan}) == 2000
    assert sum(item.intended_label == "safe" for item in plan) == 1000
    assert sum(item.conversation_shape == "query" for item in plan) == 1400
    assert sum(item.thinking_type == "fast" for item in plan) == 1400
    assert sum(item.source_ref is None for item in plan) == 1400
    assert all(not item.intended_answers for item in plan if item.intended_label == "safe")


def test_plan_is_seeded_and_primary_rule_balanced() -> None:
    from pathlib import Path
    from collections import Counter
    from risk_agent.singguard_generation import load_active_policies
    from risk_agent.singguard_query_generation import plan_blueprints

    policies = load_active_policies(Path("data/active_policies.jsonl"))
    seeds = _seed_records(600)
    first = plan_blueprints(policies, count=2000, seed=7, seed_records=seeds)
    second = plan_blueprints(policies, count=2000, seed=7, seed_records=seeds)
    assert first == second
    counts = Counter(
        item.primary_answer for item in first if item.intended_label == "unsafe"
    )
    assert max(counts.values()) - min(counts.values()) <= 1
    assert all(
        item.primary_answer in item.intended_answers
        for item in first
        if item.intended_label == "unsafe"
    )
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```bash
python -m pytest tests/test_singguard_query_generation.py -q
```

Expected: collection fails with `ModuleNotFoundError: risk_agent.singguard_query_generation`.

- [ ] **Step 3: Implement strict models and largest-remainder quota assignment**

Add these public contracts and planner entry point:

```python
class SourceRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    source: str
    source_id: str
    content_hash: str


class QueryBlueprint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    blueprint_id: str
    family_id: str
    policy_id: str
    intended_label: Literal["safe", "unsafe"]
    primary_answer: str | None = None
    intended_answers: tuple[str, ...] = ()
    thinking_type: Literal["fast", "slow"]
    conversation_shape: Literal["query", "query_response"]
    content_form: Literal[
        "short_ad", "social_post", "livestream_pitch", "product_listing",
        "comment", "private_message", "support_exchange", "search_or_neutral"
    ]
    tone: Literal[
        "formal", "colloquial", "promotional", "urgent",
        "testimonial", "technical", "humorous", "neutral"
    ]
    length_bin: Literal["headline", "short", "medium", "long"]
    difficulty: Literal["explicit", "paraphrased", "implicit", "exception"]
    noise_profile: Literal["none", "spelling", "emoji", "punctuation", "obfuscation"]
    tool_capable: bool = False
    source_ref: SourceRef | None = None


def plan_blueprints(
    policies: tuple[ActivePolicy, ...],
    *,
    count: int,
    seed: int,
    seed_records: tuple[SeedRecord, ...],
) -> tuple[QueryBlueprint, ...]:
    if count not in {100, 500, 2000}:
        raise ValueError("count must be one of 100, 500, or 2000")
    rng = random.Random(seed)
    policy_rules = [(policy, rule) for policy in policies for rule in policy.rules]
    labels = _allocate({"safe": 1, "unsafe": 1}, count)
    shapes = _allocate({"query": 7, "query_response": 3}, count)
    modes = _allocate({"fast": 7, "slow": 3}, count)
    forms = _allocate(CONTENT_FORM_WEIGHTS, count)
    difficulties = _allocate(DIFFICULTY_WEIGHTS, count)
    source_slots = _source_assignment(seed_records, count=count, rng=rng)
    return _zip_blueprints(
        policy_rules=policy_rules,
        labels=labels,
        shapes=shapes,
        modes=modes,
        forms=forms,
        difficulties=difficulties,
        source_slots=source_slots,
        rng=rng,
    )
```

Implement `_allocate` with integer floor plus deterministic largest remainders. Implement `_zip_blueprints` so unsafe primary rules are round-robin balanced, safe rows rotate through every policy and exception, IDs derive from `seed` and ordinal, and the final tuple is deterministically shuffled once.

- [ ] **Step 4: Run planner tests**

```bash
python -m pytest tests/test_singguard_query_generation.py -q
```

Expected: planner tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/risk_agent/singguard_query_generation.py tests/test_singguard_query_generation.py
git commit -m "feat: plan diverse SingGuard query blueprints"
```

### Task 2: Add deterministic content-quality gates and family-safe duplication checks

**Files:**
- Modify: `src/risk_agent/singguard_query_generation.py`
- Modify: `tests/test_singguard_query_generation.py`

- [ ] **Step 1: Write failing gate tests**

```python
def _source_ref():
    from risk_agent.singguard_query_generation import SourceRef

    return SourceRef(source="fixture", source_id="source-1", content_hash="a" * 64)


def _blueprint(blueprint_id: str = "bp-1", source_ref=None):
    from risk_agent.singguard_query_generation import QueryBlueprint

    return QueryBlueprint(
        blueprint_id=blueprint_id, family_id=blueprint_id, policy_id="commerce-risk-v1",
        intended_label="safe", primary_answer=None, intended_answers=(),
        thinking_type="fast", conversation_shape="query", content_form="short_ad",
        tone="neutral", length_bin="short", difficulty="exception",
        noise_profile="none", tool_capable=False, source_ref=source_ref,
    )


@pytest.mark.parametrize(
    ("query", "code"),
    [
        ("[user]: buy this now", "literal_role_wrapper"),
        ("Email real.person@example.com now", "pii_or_external_identifier"),
        ("This dataset label is unsafe", "generation_meta_language"),
    ],
)
def test_local_gate_rejects_content_shortcuts(query: str, code: str) -> None:
    from risk_agent.singguard_query_generation import CandidateIndex, GeneratedContent, gate_content

    result = gate_content(
        _blueprint(), GeneratedContent(blueprint_id="bp-1", query=query),
        source_text=None, index=CandidateIndex()
    )
    assert result.code == code


def test_local_gate_rejects_source_copy_and_near_duplicate() -> None:
    from risk_agent.singguard_query_generation import CandidateIndex, GeneratedContent, gate_content

    source = "urgent claim your reward by messaging this private account today"
    index = CandidateIndex()
    copied = gate_content(
        _blueprint(), GeneratedContent(blueprint_id="bp-1", query=source),
        source_text=source, index=index
    )
    assert copied.code == "source_too_similar"

    first = GeneratedContent(blueprint_id="bp-1", query="A qualified moisturizer may improve hydration for some users.")
    assert gate_content(_blueprint(), first, source_text=None, index=index).accepted
    index.add("family-1", first)
    duplicate = GeneratedContent(blueprint_id="bp-2", query="A qualified moisturizer may improve hydration for some users!")
    assert gate_content(_blueprint("bp-2"), duplicate, source_text=None, index=index).code == "near_duplicate"
```

- [ ] **Step 2: Run the new tests and verify missing-symbol failures**

```bash
python -m pytest tests/test_singguard_query_generation.py -q
```

Expected: imports fail for `CandidateIndex`, `GeneratedContent`, or `gate_content`.

- [ ] **Step 3: Implement canonicalization, five-gram similarity, and gate results**

```python
class GeneratedContent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    blueprint_id: str
    query: str = Field(min_length=1, max_length=5000)
    response: str | None = Field(default=None, max_length=5000)


class GateResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    accepted: bool
    code: str


def _fivegrams(text: str) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    return {tuple(tokens[i:i + 5]) for i in range(max(0, len(tokens) - 4))}


def _jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class CandidateIndex:
    def __init__(self) -> None:
        self._exact: set[str] = set()
        self._rows: list[tuple[str, set[tuple[str, ...]]]] = []

    def add(self, family_id: str, content: GeneratedContent) -> None:
        normalized = _normalized_content(content)
        self._exact.add(normalized)
        self._rows.append((family_id, _fivegrams(normalized)))

    def duplicate_code(self, family_id: str, content: GeneratedContent) -> str | None:
        normalized = _normalized_content(content)
        if normalized in self._exact:
            return "exact_duplicate"
        grams = _fivegrams(normalized)
        if any(other != family_id and _jaccard(grams, seen) >= 0.85 for other, seen in self._rows):
            return "near_duplicate"
        return None
```

Implement `gate_content` in a stable order: schema/shape, role wrappers, English alphabet ratio, meta-language, PII/external identifiers, length bin, exact duplicate, source similarity `>=0.50`, then candidate similarity `>=0.85`. Use reserved `.test` domains and explicit synthetic handles as the only allowed external identifiers.

- [ ] **Step 4: Run gate tests and the complete new module tests**

```bash
python -m pytest tests/test_singguard_query_generation.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/risk_agent/singguard_query_generation.py tests/test_singguard_query_generation.py
git commit -m "feat: gate SingGuard query quality and duplication"
```

### Task 3: Stream and normalize the three additional permissive seed sources

**Files:**
- Modify: `pyproject.toml`
- Modify: `configs/singguard_sources.yaml`
- Modify: `src/risk_agent/singguard_sources.py`
- Modify: `tests/test_singguard_sources.py`

- [ ] **Step 1: Write failing source-adapter tests with injected rows**

```python
def test_external_adapters_use_train_rows_and_normalize_expected_text(tmp_path: Path) -> None:
    from risk_agent.singguard_sources import fetch_configured_seeds

    rows = {
        "nemotron_aegis_v2": [
            {"id": "a1", "prompt": "Explain this fraud warning", "response": None,
             "prompt_label": "safe", "violated_categories": ""}
        ],
        "civil_comments": [
            {"id": "c1", "text": "A normal public comment", "toxicity": 0.0}
        ],
        "amazon_esci": [
            {"example_id": 7, "query": "moisturizer sensitive skin", "product_locale": "us",
             "split": "train"}
        ],
    }

    manifest = fetch_configured_seeds(
        Path("configs/singguard_sources.yaml"), tmp_path / "seeds",
        retrieved_at="2026-07-15T00:00:00Z",
        dataset_loader=lambda spec: iter(rows.get(spec.source, ())),
        enabled_sources=("nemotron_aegis_v2", "civil_comments", "amazon_esci"),
    )
    records = [json.loads(line) for line in (tmp_path / "seeds" / "seeds.jsonl").read_text().splitlines()]
    assert manifest["record_count"] == 3
    assert {row["source"] for row in records} == set(rows)
    assert all(row["license"] in {"CC-BY-4.0", "CC0-1.0", "Apache-2.0"} for row in records)
    assert all("test" not in row["source_id"] for row in records)


def test_external_source_requires_pinned_revision(tmp_path: Path) -> None:
    import yaml

    from risk_agent.singguard_sources import fetch_configured_seeds

    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        yaml.safe_dump({
            "version": "singguard-sources-v1",
            "sources": [{
                "source": "external", "dataset_url": "https://example.test/dataset",
                "artifact_url": "https://example.test/artifact", "license": "CC-BY-4.0",
                "usage_scope": "research_only", "source_role": "style_seed",
                "adapter": "hf_civil_comments", "enabled": True,
                "dataset_id": "example/dataset", "revision": None,
                "split": "train", "max_records": 10,
            }],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pinned revision"):
        fetch_configured_seeds(catalog, tmp_path / "out", dataset_loader=lambda spec: iter(()))
```

- [ ] **Step 2: Run source tests and verify schema/signature failures**

```bash
python -m pytest tests/test_singguard_sources.py -q
```

Expected: failures because `SourceSpec` rejects external fields and `fetch_configured_seeds` lacks `dataset_loader` and `enabled_sources`.

- [ ] **Step 3: Add the optional source dependency and pinned catalog fields**

```toml
[project.optional-dependencies]
dev = ["pytest>=8.3"]
sources = ["datasets>=3.0,<5"]
teacher = ["google-genai>=1.0"]
train = ["ms-swift>=4.3", "transformers>=4.57", "qwen-vl-utils>=0.0.14"]
```

Extend `SourceSpec`:

```python
dataset_id: str | None = None
revision: str | None = None
split: str | None = None
max_records: int = Field(default=5000, ge=1, le=100_000)
```

Pin these revisions in `configs/singguard_sources.yaml`:

```yaml
  - source: nemotron_aegis_v2
    dataset_id: nvidia/Aegis-AI-Content-Safety-Dataset-2.0
    revision: d86bb8bedff51d25ac834ab7838f1cc61acb7a2c
    split: train
    adapter: hf_aegis_v2
    max_records: 2000
    enabled: true
  - source: civil_comments
    dataset_id: google/civil_comments
    revision: f2970eb3a55777454c94069077cc8d9b5866312d
    split: train
    adapter: hf_civil_comments
    max_records: 2000
    enabled: true
  - source: amazon_esci
    dataset_id: parquet
    revision: 7916cdf6ab75a462e77f20ab40428a10923998d5
    split: train
    adapter: esci_query_parquet
    max_records: 1000
    enabled: true
```

For ESCI, set `artifact_url` to the revision-pinned official examples parquet under `shopping_queries_dataset/`; do not load the product table.

- [ ] **Step 4: Implement lazy streaming and source-specific normalization**

```python
def _default_dataset_loader(spec: SourceSpec) -> Iterator[Mapping[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("external sources require pip install -e '.[sources]'") from None
    if not spec.revision or not spec.split:
        raise ValueError("enabled external sources require a pinned revision and train split")
    if spec.adapter == "esci_query_parquet":
        dataset = load_dataset(
            "parquet", data_files={"train": spec.artifact_url}, split="train", streaming=True
        )
    else:
        dataset = load_dataset(
            spec.dataset_id, split=spec.split, revision=spec.revision, streaming=True
        )
    return iter(dataset)
```

Implement `_parse_external_rows` with these rules: Aegis uses `id`, `prompt`, optional `response`, and retains `prompt_label` only as `source_label`; skip `prompt == "REDACTED"`. Civil Comments uses `text` and a stable ordinal ID, storing only a coarse source label derived from toxicity for filtering. ESCI uses only rows where `product_locale == "us"` and `split == "train"`, deduplicates `query`, and never treats `esci_label` as safety supervision. Stop at `max_records` after accepted normalization.

- [ ] **Step 5: Run source and regression tests**

```bash
python -m pytest tests/test_singguard_sources.py tests/test_public_data.py -q
```

Expected: all selected tests pass without network access.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml configs/singguard_sources.yaml src/risk_agent/singguard_sources.py tests/test_singguard_sources.py
git commit -m "feat: add governed SingGuard seed streams"
```

### Task 4: Add the content-only Gemini request contract

**Files:**
- Create: `prompts/singguard_query_generator_v1.txt`
- Modify: `src/risk_agent/singguard_query_generation.py`
- Modify: `tests/test_singguard_query_generation.py`

- [ ] **Step 1: Write failing provider-contract tests**

```python
def test_content_request_contains_controls_but_not_upstream_label() -> None:
    from risk_agent.singguard_query_generation import build_content_request

    blueprint = _blueprint(source_ref=_source_ref())
    request = build_content_request(
        (blueprint,), policies=(_policy(),), source_texts={"source-1": "private message style"}
    )
    serialized = json.dumps(request)
    assert "private message style" in serialized
    assert "source_label" not in serialized
    assert "expected_label" not in serialized
    assert "Deceptive Efficacy" in serialized
    assert request["output_contract"] == {"items": [{"blueprint_id": "string", "query": "string", "response": "string|null"}]}


def test_parse_content_batch_requires_every_blueprint_once() -> None:
    from risk_agent.singguard_query_generation import parse_content_batch

    with pytest.raises(ValueError, match="exactly once"):
        parse_content_batch(
            {"items": [{"blueprint_id": "bp-1", "query": "One", "response": None}]},
            expected_ids=("bp-1", "bp-2"),
        )
```

- [ ] **Step 2: Run tests and verify missing-function failures**

```bash
python -m pytest tests/test_singguard_query_generation.py -q
```

Expected: imports fail for `build_content_request` and `parse_content_batch`.

- [ ] **Step 3: Write the generator prompt and strict batch schema**

The prompt must require natural English platform content, exact conversation shape, no label/reasoning/tool output, no real PII, no harmful operational detail, and no copying of the optional seed. It must say that every requested blueprint ID appears exactly once.

Add:

```python
CONTENT_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "blueprint_id": {"type": "string"},
                    "query": {"type": "string"},
                    "response": {"type": ["string", "null"]},
                },
                "required": ["blueprint_id", "query", "response"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def parse_content_batch(payload: Mapping[str, object], *, expected_ids: tuple[str, ...]) -> tuple[GeneratedContent, ...]:
    raw = payload.get("items")
    if not isinstance(raw, list):
        raise ValueError("content response must contain items")
    items = tuple(GeneratedContent.model_validate(item) for item in raw)
    if sorted(item.blueprint_id for item in items) != sorted(expected_ids):
        raise ValueError("content response must contain every blueprint exactly once")
    return items
```

`build_content_request` includes the complete selected policy, intended semantic target, primary rule, ordered intended rules, exception context for safe hard negatives, content controls, redacted source text, and the literal prompt version/hash. It omits upstream labels and source-only metadata.

- [ ] **Step 4: Run provider-contract tests**

```bash
python -m pytest tests/test_singguard_query_generation.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add prompts/singguard_query_generator_v1.txt src/risk_agent/singguard_query_generation.py tests/test_singguard_query_generation.py
git commit -m "feat: define Gemini query realization contract"
```

### Task 5: Orchestrate batched generation, retries, checkpoints, and query artifacts

**Files:**
- Modify: `src/risk_agent/singguard_query_generation.py`
- Modify: `tests/test_singguard_query_generation.py`

- [ ] **Step 1: Write failing end-to-end and resume tests with `CallableTeacher`**

```python
class RecordingTeacher:
    def __init__(self, *, stop_after: int | None = None) -> None:
        self.stop_after = stop_after
        self.calls: list[tuple[str, ...]] = []

    def generate(self, request):
        from risk_agent.teacher import TeacherBudgetExceeded, TeacherReply, TeacherUsage

        if self.stop_after is not None and len(self.calls) >= self.stop_after:
            raise TeacherBudgetExceeded("max_requests")
        blueprints = request["blueprints"]
        ids = tuple(item["blueprint_id"] for item in blueprints)
        self.calls.append(ids)
        items = [
            {
                "blueprint_id": item["blueprint_id"],
                "query": f"Natural platform content {item['blueprint_id']} with ordinary wording.",
                "response": (
                    "A natural model response for moderation."
                    if item["conversation_shape"] == "query_response" else None
                ),
            }
            for item in blueprints
        ]
        return TeacherReply(
            payload={"items": items},
            usage=TeacherUsage(provider="fixture", model="fixture"),
        )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_query_batch_writes_trainable_samples_and_auditable_manifest(tmp_path: Path) -> None:
    from risk_agent.singguard_query_generation import run_query_batch

    teacher = RecordingTeacher()
    manifest = run_query_batch(
        policies=(_policy(),), seed_records=(), output_dir=tmp_path / "query-batch",
        count=100, seed=42, teacher=teacher, batch_size=4, max_attempts_per_blueprint=3,
    )
    rows = _read_jsonl(tmp_path / "query-batch" / "content_samples.jsonl")
    assert manifest["status"] == "complete"
    assert len(rows) == 100
    assert all(not row.get("tool_names") for row in rows)
    assert manifest["contract_version"] == "singguard-query-v1"
    assert manifest["accepted"] == 100
    assert manifest["quota_coverage"]["label"] == {"safe": 50, "unsafe": 50}


def test_query_resume_never_regenerates_completed_blueprints(tmp_path: Path) -> None:
    first = RecordingTeacher(stop_after=2)
    output = tmp_path / "resume"
    partial = run_query_batch(
        policies=(_policy(),), seed_records=(), output_dir=output,
        count=100, seed=42, teacher=first, batch_size=4, max_attempts_per_blueprint=3,
    )
    assert partial["status"] == "incomplete"
    completed = set(json.loads((output / "checkpoint.json").read_text())["completed_ids"])

    resumed = RecordingTeacher()
    run_query_batch(
        policies=(_policy(),), seed_records=(), output_dir=output,
        count=100, seed=42, teacher=resumed,
        batch_size=4, max_attempts_per_blueprint=3, resume=True,
    )
    assert completed.isdisjoint({item for batch in resumed.calls for item in batch})
```

- [ ] **Step 2: Run tests and verify missing orchestration failure**

```bash
python -m pytest tests/test_singguard_query_generation.py -q
```

Expected: import fails for `run_query_batch`.

- [ ] **Step 3: Implement crash-safe orchestration**

Implement this public boundary:

```python
def run_query_batch(
    *,
    policies: tuple[ActivePolicy, ...],
    seed_records: tuple[SeedRecord, ...],
    output_dir: Path,
    count: int,
    seed: int,
    teacher: Teacher,
    batch_size: int = 4,
    max_attempts_per_blueprint: int = 3,
    resume: bool = False,
    progress: Callable[[Mapping[str, object]], object] | None = None,
) -> dict[str, object]:
```

Before calls, atomically write `plan.jsonl` and hashes. For each pending batch, call `teacher.generate(build_content_request(batch, policies=policies, source_texts=source_texts))`, parse each item independently, gate it, append accepted `ModerationSample` rows, and persist rejected attempts with only blueprint ID, attempt, and stable reason code. A provider or budget failure writes a consistent checkpoint and returns `status=incomplete`. Resume verifies policy, plan, source snapshot, prompt, gate version, batch size, and attempt-limit hashes before reading prior artifacts.

Write these files atomically after each completed batch: `content_samples.jsonl`, `sample_metadata.jsonl`, `rejected.jsonl`, `checkpoint.json`, and `manifest.json`. Append sanitized operational records to `events.jsonl`. Build `content_review_sample.jsonl` deterministically from accepted metadata and content using stratified round-robin selection.

- [ ] **Step 4: Run orchestration tests**

```bash
python -m pytest tests/test_singguard_query_generation.py -q
```

Expected: all tests pass, including interrupted resume.

- [ ] **Step 5: Commit**

```bash
git add src/risk_agent/singguard_query_generation.py tests/test_singguard_query_generation.py
git commit -m "feat: orchestrate resumable SingGuard query synthesis"
```

### Task 6: Add the minimal query-generation CLI and progress output

**Files:**
- Create: `scripts/generate_singguard_queries.py`
- Modify: `tests/test_singguard_query_generation.py`

- [ ] **Step 1: Write failing parser and dry-run tests**

```python
def test_query_cli_requires_model_unless_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.generate_singguard_queries import main

    monkeypatch.delenv("GEMINI_GENERATOR_MODEL", raising=False)
    with pytest.raises(SystemExit) as error:
        main(["data/active_policies.jsonl", "out", "--count", "100"])
    assert error.value.code == 2


def test_query_cli_dry_run_writes_plan_without_provider_calls(tmp_path: Path) -> None:
    from scripts.generate_singguard_queries import main

    code = main([
        "data/active_policies.jsonl", str(tmp_path / "plan"),
        "--count", "100", "--seed", "42", "--dry-run",
    ])
    assert code == 0
    assert sum(1 for _ in (tmp_path / "plan" / "plan.jsonl").open()) == 100
```

- [ ] **Step 2: Run tests and verify the missing-script failure**

```bash
python -m pytest tests/test_singguard_query_generation.py -q
```

Expected: import fails for `scripts.generate_singguard_queries`.

- [ ] **Step 3: Implement the CLI**

The parser has two positional arguments, `active_policies` and `output_dir`, and these options:

```python
parser.add_argument("--seeds", type=Path)
parser.add_argument("--count", type=int, choices=(100, 500, 2000), required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch-size", type=int, choices=(1, 2, 3, 4), default=4)
parser.add_argument("--max-attempts-per-blueprint", type=int, choices=(1, 2, 3), default=3)
parser.add_argument("--model", default=os.environ.get("GEMINI_GENERATOR_MODEL") or os.environ.get("GEMINI_MODEL"))
parser.add_argument("--resume", action="store_true")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--max-requests", type=int, default=1000)
parser.add_argument("--max-output-tokens", type=int, default=4096)
parser.add_argument("--request-timeout", type=float, default=120.0)
```

Dry-run calls only `plan_blueprints` and writes `plan.jsonl` plus a plan manifest. Normal mode constructs `GeminiTeacher(model=args.model, response_schema=CONTENT_BATCH_SCHEMA, temperature=0.8, budget=budget)` and calls `run_query_batch`. Reuse the compact stderr progress format from `generate_singguard_data.py`; keep stdout to one final manifest JSON object.

- [ ] **Step 4: Run CLI tests and inspect help**

```bash
python -m pytest tests/test_singguard_query_generation.py -q
python scripts/generate_singguard_queries.py --help
```

Expected: tests pass and help shows the two positional arguments and bounded count choices.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_singguard_queries.py tests/test_singguard_query_generation.py
git commit -m "feat: add SingGuard query synthesis CLI"
```

### Task 7: Join query provenance into a final trace review sample

**Files:**
- Modify: `src/risk_agent/singguard_generation.py`
- Modify: `scripts/generate_singguard_data.py`
- Modify: `tests/test_singguard_generation.py`

- [ ] **Step 1: Write failing review-join tests**

```python
def test_trace_batch_writes_joined_review_without_changing_train_rows(tmp_path: Path) -> None:
    from risk_agent.singguard_generation import AgentTurn, run_generation_batch
    from risk_agent.teacher import TeacherBudget

    sample = ModerationSample(
        sample_id="sample-1", policy_id="commerce-v1", thinking_type="fast",
        query="Message my private account to complete the order.", expected_label="unsafe",
        expected_answers=("Off-Platform Solicitation",),
    )
    provider = FakeAgentProvider([
        AgentTurn(content="unsafe\n<answer>Off-Platform Solicitation</answer>")
    ])
    metadata = {
        "sample-1": {
            "primary_answer": "Off-Platform Solicitation",
            "content_form": "short_ad", "source": "synthetic"
        }
    }
    output = tmp_path / "trace"
    manifest = run_generation_batch(
        policies=(_policy(),), samples=(sample,), provider=provider,
        environment=_environment(), output_dir=output,
        budget=TeacherBudget(max_requests=5), review_metadata=metadata,
    )
    review = [json.loads(line) for line in (output / "review_sample.jsonl").read_text().splitlines()]
    train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert len(review) == 1
    assert review[0]["sample_id"] == "sample-1"
    assert review[0]["training_row"] == train[0]
    assert "sample_id" not in train[0]
    assert manifest["review_sample_count"] == 1
```

- [ ] **Step 2: Run the test and verify the unexpected-keyword failure**

```bash
python -m pytest tests/test_singguard_generation.py::test_trace_batch_writes_joined_review_without_changing_train_rows -q
```

Expected: `run_generation_batch()` rejects `review_metadata`.

- [ ] **Step 3: Implement bounded metadata loading and stratified review output**

Add an optional `review_metadata: Mapping[str, Mapping[str, object]] | None = None` argument. Validate that keys exactly match known sample IDs when supplied, values contain only allowlisted planning/provenance fields, and no raw source seed text is present. After accepted rows are known, choose at most 100 rows by deterministic round-robin across primary answer, label, content form, source mode, thinking type, and conversation shape. Write joined rows only to `review_sample.jsonl`; keep `train.jsonl` byte-compatible with the existing ms-swift schema.

Add CLI support:

```python
parser.add_argument("--sample-metadata", type=Path)
```

Load `sample_metadata.jsonl` into a unique `sample_id` mapping and pass it to `run_generation_batch`.

- [ ] **Step 4: Run generation tests**

```bash
python -m pytest tests/test_singguard_generation.py -q
```

Expected: all tests pass and existing train rows remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/risk_agent/singguard_generation.py scripts/generate_singguard_data.py tests/test_singguard_generation.py
git commit -m "feat: emit stratified SingGuard trace review rows"
```

### Task 8: Document, verify, and run the 100-row pilot gate

**Files:**
- Modify: `README.md`
- Modify: `data/README.md`
- Modify: `docs/singguard-data-runbook.md`
- Modify: `tests/test_singguard_query_generation.py`

- [ ] **Step 1: Add a failing documentation-contract test**

```python
def test_runbook_contains_query_then_trace_commands() -> None:
    text = Path("docs/singguard-data-runbook.md").read_text(encoding="utf-8")
    assert "generate_singguard_queries.py" in text
    assert "--count 100" in text
    assert "--sample-metadata" in text
    assert "Do not train from the query pilot" in text
```

- [ ] **Step 2: Run the documentation test and verify failure**

```bash
python -m pytest tests/test_singguard_query_generation.py::test_runbook_contains_query_then_trace_commands -q
```

Expected: assertion failure because the runbook lacks the new commands.

- [ ] **Step 3: Add exact installation, fetch, query, trace, and inspection commands**

Document:

```bash
pip install -e '.[dev,teacher,sources]'

rm -rf outputs/singguard-seeds-v2
python scripts/fetch_singguard_seeds.py \
  configs/singguard_sources.yaml \
  outputs/singguard-seeds-v2

rm -rf outputs/singguard-query-pilot-v1
python scripts/generate_singguard_queries.py \
  data/active_policies.jsonl \
  outputs/singguard-query-pilot-v1 \
  --seeds outputs/singguard-seeds-v2/seeds.jsonl \
  --count 100 \
  --seed 42 \
  --batch-size 4 \
  --max-requests 100 \
  --max-output-tokens 4096 \
  --request-timeout 120

rm -rf outputs/singguard-trace-pilot-v1
python scripts/generate_singguard_data.py \
  data/active_policies.jsonl \
  outputs/singguard-query-pilot-v1/content_samples.jsonl \
  outputs/singguard-trace-pilot-v1 \
  --sample-metadata outputs/singguard-query-pilot-v1/sample_metadata.jsonl \
  --tool-env data/tool_env \
  --max-tool-calls 0 \
  --max-requests 500 \
  --max-output-tokens 4096 \
  --request-timeout 120
```

Explain that the 100-row pilot is entirely reviewed before scaling, query and trace outputs use fresh directories, and `--resume` is used only with the identical command contract.

- [ ] **Step 4: Run the complete local verification suite**

```bash
python -m pytest -q
python -m compileall -q src scripts
python scripts/generate_singguard_queries.py --help
python scripts/generate_singguard_data.py --help
git diff --check
```

Expected: zero test failures, compilation exits zero, both CLIs print help, and `git diff --check` prints nothing.

- [ ] **Step 5: Run a local dry plan before any network/provider call**

```bash
rm -rf outputs/singguard-query-dry-run
python scripts/generate_singguard_queries.py \
  data/active_policies.jsonl \
  outputs/singguard-query-dry-run \
  --count 100 \
  --seed 42 \
  --dry-run
```

Expected: `plan.jsonl` contains 100 unique rows and the manifest reports 50 safe, 50 unsafe, 70 query-only, and 30 query-response blueprints.

- [ ] **Step 6: Commit documentation and final verification changes**

```bash
git add README.md data/README.md docs/singguard-data-runbook.md tests/test_singguard_query_generation.py
git commit -m "docs: add SingGuard query synthesis runbook"
```

- [ ] **Step 7: Run the server pilot and enforce the stop gate**

Run the documented source fetch, 100-query generation, and 100-trace commands on the server. Do not scale when semantic acceptance is below 80%, accepted source-copy or real-PII count is non-zero, accepted near-duplicate rate exceeds 3%, or any planned cell is below 90% coverage. Inspect all query rejections, at least 30 accepted content rows, and the final `review_sample.jsonl` before authorizing the 500-row stage.
