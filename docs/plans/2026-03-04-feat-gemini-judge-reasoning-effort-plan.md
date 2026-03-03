---
title: "feat: Gemini judge + per-phase reasoning effort"
type: feat
status: active
date: 2026-03-04
origin: /Users/terry/.claude/plans/generic-pondering-otter.md
---

# feat: Gemini as Judge + Per-Phase Reasoning Effort

## Overview

Two orthogonal improvements to consilium:

1. **Gemini 3.1 Pro as judge** — All 6 frontier labs now participate. Claude Opus moves from judge-only to M2 panelist + critique role. Gemini (OR 5.0s) replaces Anthropic (OR direct) as judge — better representation, cheaper ($2/$12 vs $5/$25 Mtok), and removes the "conflict of interest" exclusion. Risk: Gemini trails Opus on synthesis quality (Elo 1317 vs 1606 on GDPval-AA) — mitigated by the Prescription Discipline prompt and collabeval critique.

2. **`--effort low|medium|high`** — Explicit per-phase thinking budget. Blind phase needs deep independent reasoning. Debate rounds are peer responses (lighter). Judge synthesis needs maximum depth. Default (no flag) = unchanged behaviour (provider defaults).

---

## Problem Statement / Motivation

**Gemini as judge:** Claude was judge purely to avoid conflict of interest with council panelists, not because it's the best judge for the task. Now that Gemini is a panelist (M2 slot), swapping puts all 6 frontier labs in the council and lets Claude's strong analytical reasoning contribute to deliberation rather than just synthesis. Critique role also moves from Gemini→Claude for true judge/critic independence.

**Per-phase effort:** All models currently use provider-default thinking budgets — no explicit control. This wastes compute in debate rounds (just peer responses) and under-invests in the blind phase (where independent reasoning matters most, per the Surowiecki/Delphi research). Per-phase degradation is a latency/quality trade-off the user can tune.

---

## Proposed Solution

### Feature A: Gemini Judge

**Constants** (`config.rs`):
```rust
// Before
pub const JUDGE_MODEL: &str = "anthropic/claude-opus-4-6";
pub const CRITIQUE_MODEL: &str = "google/gemini-3.1-pro-preview";

// After
pub const JUDGE_MODEL: &str = "google/gemini-3.1-pro-preview";
pub const CRITIQUE_MODEL: &str = "anthropic/claude-opus-4-6";  // swap for independence
```

**M2 slot** (`resolved_council()`, config.rs lines 124-127):
```rust
// Before: google/gemini-3.1-pro-preview, None fallback
// After:  anthropic/claude-opus-4-6, Some(("anthropic", "claude-opus-4-6")) fallback
let model_2 = env_override(CONSILIUM_MODEL_M2_ENV)
    .map(|v| leak_if_needed(v, "anthropic/claude-opus-4-6"))
    .unwrap_or("anthropic/claude-opus-4-6");
// Fallback in vec: Some(("anthropic", "claude-opus-4-6"))
```

**`query_judge`** (`api.rs` lines 17-38) — make model-aware:
```rust
// Before: always tries ANTHROPIC_API_KEY → query_anthropic() → fallback OR
// After: dispatch on model prefix

if model.contains("google/") || model.contains("gemini") {
    if let Ok(g_key) = std::env::var("GOOGLE_API_KEY") {
        let bare = model.strip_prefix("google/").unwrap_or(model);
        let resp = query_google_ai_studio(client, &g_key, bare, messages, max_tokens, timeout_secs, retries).await;
        if !resp.starts_with('[') { return resp; }
    }
} else if model.contains("anthropic/") || model.contains("claude") {
    if let Ok(ant_key) = std::env::var("ANTHROPIC_API_KEY") {
        let resp = query_anthropic(client, &ant_key, model, messages, max_tokens, timeout_secs, retries).await;
        if !resp.starts_with('[') { return resp; }
    }
}
// fallback
query_model(client, openrouter_api_key, model, messages, max_tokens, timeout_secs, retries, cost_tracker).await
```

**New "anthropic" branch in `query_model_with_fallback`** (`api.rs` ~line 963):
```rust
// NEW: For Claude panelists — try Anthropic API directly first
if let Some(("anthropic", ant_model)) = fallback {
    if let Some(akey) = anthropic_api_key {
        let resp = query_anthropic(client, akey, ant_model, messages, max_tokens, timeout_secs, retries).await;
        if !is_error_response(&resp) {
            return (name.to_string(), ant_model.to_string(), resp);
        }
        primary_response = Some(resp);
    }
}
```
Also add `anthropic_api_key: Option<&str>` to `query_model_with_fallback` signature and propagate from callers.

**`quick_models()` label fix** (`config.rs` lines 164-170):
```rust
// Before: hardcoded label "Claude"
// After: derive label from model string
let judge = resolved_judge_model();
let judge_label = display_name_from_model(&judge);
let judge_label = leak_if_needed(judge_label, "Judge");
let mut models: Vec<ModelEntry> = vec![(judge_label, leak_if_needed(judge, JUDGE_MODEL), None)];
```

**Dedup in `quick_models()`**: Now that Gemini is both judge (prepended) and M2 panelist (from council), quick mode would have Gemini twice. Fix: filter out the judge model from the council slice before extending.
```rust
let judge_model_id = resolved_judge_model();
models.extend(
    resolved_council().into_iter().filter(|(_, m, _)| *m != judge_model_id.as_str())
);
```

**`CLASSIFIER_MODEL`** (`config.rs` line 53) — comment says "same as judge" but is a separate constant. Update comment; leave value as `"anthropic/claude-opus-4-6"` (classifier should remain stable and fast, not track judge).

**`COUNCIL` const comment** — update: "Claude is judge-only" → "Gemini is judge; Claude is M2 panelist + critique"

**Prescription Discipline validation**: After implementation, run a test council and verify Gemini judge output has max 3 "Do Now" items and an explicit "Skip" section. The judge prompt already contains Prescription Discipline — test it holds for Gemini.

---

### Feature B: Per-Phase Reasoning Effort

**`ReasoningEffort` enum** (`config.rs`):
```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => None,
        }
    }
    pub fn step_down(self) -> Self {
        match self { Self::High => Self::Medium, _ => Self::Low }
    }
    pub fn as_str(self) -> &'static str {
        match self { Self::Low => "low", Self::Medium => "medium", Self::High => "high" }
    }
    /// Anthropic: budget_tokens must be strictly less than max_tokens
    pub fn anthropic_budget(self) -> u32 {
        match self { Self::Low => 1024, Self::Medium => 6000, Self::High => 16000 }
    }
    pub fn google_budget(self) -> i64 {
        match self { Self::Low => 512, Self::Medium => 4096, Self::High => 16000 }
    }
}
```

**`--effort` flag** (`cli.rs`):
```rust
/// Reasoning effort for thinking models: low, medium, high
/// Blind phase uses configured effort; debate steps down one level; judge always uses high
#[arg(long, help_heading = "Deliberation")]
pub effort: Option<String>,
```

**Signature changes** — add `effort: Option<ReasoningEffort>` to (~25 signatures):
- All 6 provider query fns: `query_openai`, `query_anthropic`, `query_google_ai_studio`, `query_xai`, `query_bigmodel`, `query_moonshot`
- `query_model` (OR path)
- `query_model_with_fallback` + `query_model_async`
- `run_parallel` + `run_parallel_with_different_messages`
- `query_judge` (internally always passes `Some(High)` to inner calls)
- Mode entry fns: `run_council`, `run_quick`, `run_discuss`, `run_oxford`, `run_redteam`, `run_premortem`, `run_forecast`

**Per-provider injection** (only when `is_thinking_model(model) && effort.is_some()`):

| Provider | Injection |
|----------|-----------|
| `query_openai` | `"reasoning": {"effort": effort.as_str()}` in Responses API body |
| `query_anthropic` | `"thinking": {"type": "enabled", "budget_tokens": N}` + bump `max_tokens = max_tokens.max(effort.anthropic_budget() + 2000)` |
| `query_google_ai_studio` | `body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": N}` |
| `query_xai` | `"reasoning_effort": effort.as_str()` |
| `query_bigmodel` | skip (no confirmed API param) |
| `query_moonshot` | skip (no confirmed API param) |
| `query_model` (OR) | `"reasoning": {"effort": effort.as_str()}` — OR passes through |

**Anthropic max_tokens bump is critical**: `budget_tokens` must be < `max_tokens`. Current blind/judge max_tokens is 1200-1500. At `High`, `anthropic_budget()=16000` — must bump `max_tokens` to at least 18000 before sending.

**Per-phase degradation in `council.rs`**:
```rust
let effort = cli.effort.as_deref().and_then(ReasoningEffort::from_str);

// Blind phase
run_parallel(..., effort, ...).await;

// Debate rounds (sequential query_model_async per panelist)
query_model_async(..., effort.map(|e| e.step_down()), ...).await;

// Judge — query_judge ignores passed effort, always uses High internally
query_judge(...).await;

// Critique — pass effort (full quality for critique)
query_model(..., effort, ...).await;
```

**Other modes** (quick, discuss, oxford, redteam, premortem, forecast): pass `effort` as-is (no per-phase degradation — single-phase structure).

---

## Technical Considerations

### Architecture Impacts

- **`query_model_with_fallback`** is already the widest signature in the codebase (14 params). Adding `effort` makes 15. This is within acceptable range — a `QueryOptions` refactor would be cleaner but is out of scope for this change.

- **`run_parallel_with_different_messages`** (used by redteam, oxford, premortem, forecast) also needs `effort` threaded through. Don't miss this — it's separate from `run_parallel`.

- **`CLASSIFIER_MODEL`** and `DISCUSS_HOST` are both `"anthropic/claude-opus-4-6"`. Neither is affected by judge swap — they go through OR only (`query_model`, not `query_judge`). No changes needed, but update `CLASSIFIER_MODEL` comment (no longer "same as judge").

- **Gemini as judge gets `GOOGLE_API_KEY` direct call first** → but benchmarks show OR (5.0s) is faster than AI Studio direct (8.3s) from HK. The `query_judge` native-first pattern is kept for correctness/resilience, but expect OR to win in practice unless on mainland China routes. This is consistent with the M2 slot having `None` fallback (OR-only) for Gemini as panelist.

### Performance Implications

- **`--effort high`** on judge: With Anthropic thinking, `max_tokens` will be bumped to ~18000+. This increases response time significantly for the critique path (now Claude at High). Profile on a real council before recommending High as default.
- **Debate step-down**: `Medium→Low` reduces thinking budget from 6000→1024 (Anthropic) tokens. Expect ~30-50% faster per-model debate calls.

### Security Considerations

- No new API keys needed — all 5 native keys already in keychain.
- `effort` is a local enum, no user-controlled string reaches API except through `as_str()` which returns only "low"|"medium"|"high".

---

## System-Wide Impact

### Interaction Graph

- `main.rs` parses CLI → passes to mode fn → `run_council()` receives effort
- `run_council()` → `run_parallel()` (blind) → `query_model_with_fallback()` → provider fns
- `run_council()` → `query_model_async()` (debate loop) → same chain
- `run_council()` → `query_judge()` → `query_google_ai_studio()` or `query_anthropic()` → OR fallback
- `run_council()` → `query_model()` (critique) → OR

### Error Propagation

- New `"anthropic"` branch in `query_model_with_fallback`: failure returns `primary_response` error string → falls through to OR (existing pattern, no new error paths)
- `query_judge` model-dispatch: unknown provider prefix falls through to OR (safe default)
- Anthropic thinking 400 (budget > max_tokens): bumping max_tokens preemptively prevents this

### State Lifecycle Risks

- No persistent state changes. Council session files write the transcript regardless.
- `cost_tracker` accumulates across all calls — new Claude M2 calls via OR will add to cost correctly.

### API Surface Parity

- `quick_models()` dedup logic: must filter judge model from council list to prevent Gemini appearing twice
- `run_parallel_with_different_messages` needs identical effort param to `run_parallel` — test both

---

## Acceptance Criteria

- [ ] `consilium --doctor` shows: M1=GPT, M2=Claude-Opus-4-6, Judge=Gemini-3.1-Pro
- [ ] `consilium --quick --no-save "test"` runs 6 models without duplication (no Gemini twice)
- [ ] `consilium --council --no-save "test"` runs with Gemini as judge, Claude as M2
- [ ] `consilium --council --effort low --no-save "test"` completes without error
- [ ] `consilium --council --effort high --no-save "test"` completes (Anthropic max_tokens bump works)
- [ ] Judge output has max 3 "Do Now" items (Prescription Discipline holds for Gemini)
- [ ] `cargo test` passes (62 existing tests + new effort tests)
- [ ] `cargo clippy` passes with 0 errors
- [ ] `CONSILIUM_MODEL_M2=some/other-model consilium --doctor` shows override correctly
- [ ] `CONSILIUM_MODEL_JUDGE=anthropic/claude-opus-4-6 consilium --council "test"` restores Opus as judge

---

## Implementation Phases

### Phase 1: Gemini Judge + Claude M2 (Feature A)
**Files:** `config.rs`, `api.rs`, `admin.rs`, `README.md`

1. Update `JUDGE_MODEL`, `CRITIQUE_MODEL` constants
2. Update `resolved_council()` M2 defaults + add `Some(("anthropic", "claude-opus-4-6"))` fallback
3. Make `query_judge` model-aware (Google/Anthropic dispatch)
4. Add `"anthropic"` branch to `query_model_with_fallback` + new `anthropic_api_key` param
5. Fix `quick_models()` label + dedup logic
6. Update `CLASSIFIER_MODEL` comment
7. Update `doctor()` in `admin.rs`
8. Update README council table

**Verify:** `cargo build --release && cargo test && cargo clippy && consilium --doctor`

### Phase 2: Per-Phase Effort (Feature B)
**Files:** `config.rs`, `cli.rs`, `api.rs`, `modes/council.rs` + other mode files

1. Add `ReasoningEffort` enum to `config.rs`
2. Add `--effort` flag to `cli.rs`
3. Add `effort: Option<ReasoningEffort>` to all query function signatures
4. Inject provider-specific effort in each query fn body (guarded by `is_thinking_model`)
5. Thread effort through `run_parallel`, `run_parallel_with_different_messages`, `query_model_async`
6. Thread per-phase degradation through `council.rs`
7. Pass `effort` as-is through other mode files

**Verify:** `cargo test && consilium --council --effort low "test" --no-save`

---

## Dependencies & Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Gemini judge over-aggregation (Prescription Discipline prompt may need tuning) | Medium | Test with a real council, compare Do Now list length vs Opus |
| Elo gap (Gemini 1317 vs Opus 1606 on GDPval-AA) — lower synthesis quality | Medium | `CONSILIUM_MODEL_JUDGE` env var lets user revert; document trade-off |
| Anthropic thinking budget > max_tokens → 400 error | High without fix | Preemptive `max_tokens` bump in `query_anthropic` when effort set |
| Gemini in quick_models twice (before dedup fix) | Certain without fix | Explicit dedup in `quick_models()` — test case needed |
| `run_parallel_with_different_messages` missing effort param | Certain without fix | Grep for all callers before marking Phase 2 done |
| xAI `reasoning_effort` field name unconfirmed | Low | Test with Grok-4 after implementation; fallback is OR reasoning.effort |

---

## Sources

### Internal References
- Approved plan: `/Users/terry/.claude/plans/generic-pondering-otter.md`
- API latency benchmark: `~/docs/solutions/consilium-api-latency-benchmark.md`
- Streaming gotchas: `~/docs/solutions/consilium-streaming-gotchas.md`
- Judge over-aggregation: `~/docs/solutions/ai-tooling/llm-council-judge-over-aggregation.md`
- Multi-agent deliberation research: `~/docs/solutions/multi-llm-deliberation-research.md`
- Cross-model routing: `~/docs/solutions/cross-model-routing-guide.md`
- `src/api.rs` line 19 (`query_judge`), line 921 (`query_model_with_fallback`)
- `src/config.rs` line 50 (`JUDGE_MODEL`), line 164 (`quick_models`)
- `src/modes/council.rs` line 487 (blind), line 993 (debate), line 1244 (judge)

### Key Design Decisions Carried From Approved Plan
- Critique swaps to Claude for judge/critic independence (not noted in learnings KB, design decision)
- `--effort` default = `None` (no behaviour change unless flag is passed)
- `query_judge` always forces `High` internally for judge, regardless of phase effort
- Debate step-down: `High→Medium`, `Medium→Low`, `Low→Low`
