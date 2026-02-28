---
title: "consilium v0.1.4: Context Compression, Feedback, Early Exit, Stats"
type: feat
status: active
date: 2026-02-28
---

# consilium v0.1.4: Four Improvements

## Overview

Four targeted improvements to consilium's deliberation pipeline and CLI experience:

1. **Context compression** for multi-round debates (summary checkpoints between rounds)
2. **`--feedback` flag** for post-session user ratings
3. **`--thorough` flag** to override early consensus exit
4. **Stats enrichment** with cost-per-mode breakdown, latency percentiles, top modes

## Problem Statement / Motivation

- **Context explosion**: Multi-round council debates pass the full conversation history to every model every round. By round 3 with 5 models, each API call sees ~15 prior responses. Token costs scale O(rounds x council_size x accumulated_history).
- **No quality signal**: Sessions are saved but there's no way to mark which were actually useful. Can't prioritize improvements without knowing what works.
- **Missing user control**: `detect_consensus()` triggers early exit automatically — no way to force full deliberation when you want thoroughness over speed.
- **Thin stats**: `--stats` shows per-mode averages but no percentiles, ranking, or trend data.

## Proposed Solution

### Feature 1: Context Compression (council + discuss only)

**Approach**: After each debate round, call a cheap model to compress the round's exchanges into a structured summary. Subsequent rounds see the summary instead of full transcripts. Full transcripts preserved for session save and judge synthesis.

**Integration point**: `council.rs` lines 705-829 (debate loop), specifically the conversation context building at lines 750-781.

**New function in `council.rs`**:
```rust
async fn compress_round_context(
    round_responses: &[(String, String)],
    question: &str,
    client: &Client,
    api_key: &str,
    cost_tracker: &CostTracker,
) -> String
```

**Prompt design**: Ask for a structured digest:
```
Summarize this debate round. For each speaker, capture:
1. Core position (1 sentence)
2. Key new argument or rebuttal (1 sentence)
3. Whether they agree/disagree with majority

Keep exact quotes only if they contain specific data points or citations.
```

**Model choice**: Use the cheapest available model via OpenRouter (e.g., `meta-llama/llama-3.3-70b-instruct` or similar). Compression call costs ~$0.005 but saves ~$0.10+ per subsequent round.

**Where it applies**:
- `council` mode: Compress after each debate round (not blind phase, not judge)
- `discuss` mode: Compress after each roundtable round
- **NOT** `oxford`, `redteam`, `socratic` — these need exact phrasing for adversarial precision

**Data flow**:
```
Round 1: Full blind claims → all models see full context
Round 1 complete → compress_round_context() → summary_r1
Round 2: Models see summary_r1 + blind_context (not full R1 transcripts)
Round 2 complete → compress_round_context() → summary_r2
Judge: Sees full transcripts (not summaries) for final synthesis
```

**Opt-out**: `--thorough` flag (Feature 3) disables compression too — full context preserved.

**Files changed**:
- `src/modes/council.rs`: Add `compress_round_context()`, modify debate loop context building
- `src/modes/discuss.rs`: Same pattern for roundtable rounds
- `src/api.rs`: Reuse `query_model()` with cheap model — no new API code needed

### Feature 2: `--feedback` Flag

**Approach**: After session completes, if `--feedback` is set, prompt user for a 1-5 rating on stdin. Store in history.jsonl via the existing `extra` field in `log_history()`.

**Integration point**: `main.rs` after `finish_session()` call (line ~224).

**Implementation**:
```rust
// In main.rs, after finish_session()
if args.feedback {
    eprint!("Rate this session (1-5): ");
    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_ok() {
        if let Ok(rating) = input.trim().parse::<u8>() {
            if (1..=5).contains(&rating) {
                // Append to history.jsonl
                append_feedback_to_history(&session_path, rating);
            }
        }
    }
}
```

**History update**: Rather than modifying the existing `log_history()` call (which has already fired), add a small `append_feedback_to_history()` function in `session.rs` that reads the last line of history.jsonl, adds the `"feedback"` field, and rewrites it. Simpler than threading feedback through the existing call chain.

**Stats integration**: Feature 4 reads this field for average satisfaction display.

**Files changed**:
- `src/cli.rs`: Add `pub feedback: bool` field with `#[arg(long)]`
- `src/main.rs`: Add feedback prompt after `finish_session()`
- `src/session.rs`: Add `append_feedback_to_history()` helper

### Feature 3: `--thorough` Override Flag

**Approach**: When set, skip consensus-based early exit AND context compression. Forces full rounds with full context.

**Integration point**: `council.rs` line 823-828 where `detect_consensus()` triggers `break`.

**Current code**:
```rust
let (converged, reason) = detect_consensus(&conversation, council_config, Some(current_challenger));
if converged {
    let _ = output.write_str(&format!(">>> CONSENSUS DETECTED ({reason}) - proceeding to judge\n\n"));
    break;
}
```

**New code**:
```rust
if !thorough {
    let (converged, reason) = detect_consensus(&conversation, council_config, Some(current_challenger));
    if converged {
        let _ = output.write_str(&format!(">>> CONSENSUS DETECTED ({reason}) - proceeding to judge\n\n"));
        break;
    }
}
```

**Pass-through**: `args.thorough` from cli.rs → main.rs → `run_council()` and `run_discuss()`. Both functions already have 15+ parameters, so one more bool is consistent with the existing pattern.

**Files changed**:
- `src/cli.rs`: Add `pub thorough: bool` field
- `src/main.rs`: Pass `args.thorough` to council/discuss calls
- `src/modes/council.rs`: Add `thorough: bool` parameter, gate consensus check + compression
- `src/modes/discuss.rs`: Same pattern

### Feature 4: Stats Enrichment

**Approach**: Enhance `admin.rs` `show_stats()` to add:
- Cost-per-mode breakdown (sorted by total cost descending)
- Latency percentiles (p50, p95)
- Most-used modes ranking
- Average feedback rating per mode (if feedback data exists)
- Last 7d / last 30d / all-time breakdown

**Integration point**: `admin.rs` lines 65-199 (`show_stats()`).

**Current output format**:
```
Mode        Sessions   Avg Cost   Total Cost   Avg Time
council          15    $0.42      $6.30        85s
```

**New output format**:
```
consilium stats — 142 sessions, $28.43 total

By mode (sorted by usage):
  council     89 sessions  $0.42 avg  $37.38 total  45s p50  120s p95
  quick       31 sessions  $0.01 avg   $0.31 total   8s p50   15s p95
  discuss     12 sessions  $0.28 avg   $3.36 total  38s p50   65s p95
  redteam      6 sessions  $0.19 avg   $1.14 total  25s p50   40s p95
  oxford       4 sessions  $0.35 avg   $1.40 total  50s p50   70s p95

Last 7 days: 18 sessions, $7.20
Last 30 days: 65 sessions, $18.50

Feedback: 3.8 avg (24 rated sessions)
```

**Percentile calculation**: Sort durations, index at `floor(len * 0.5)` and `floor(len * 0.95)`. No external deps needed.

**Files changed**:
- `src/admin.rs`: Rewrite `show_stats()` with enriched output

## Technical Considerations

- **Context compression model**: Must be cheap and fast. Use `query_model()` with a budget model (configurable constant in `config.rs`). If it fails, fall back to no compression (full context) — never block the debate.
- **Feedback stdin blocking**: `std::io::stdin().read_line()` blocks the tokio runtime but we're at the end of execution, so this is fine. No need for async stdin.
- **History.jsonl backward compatibility**: New fields (`feedback`, `rounds_completed`) are optional. Stats code must handle entries without them via `.and_then()`.
- **JSON trailing comma gotcha**: If compression model returns structured JSON, apply trailing-comma fix `re.sub(r",\s*([}\]])", r"\1", text)` — known LLM JSON gotcha from docs/solutions.
- **`--thorough` + `--deep`**: `--deep` already sets rounds to max(2, rounds). `--thorough` adds no-compression and no-early-exit on top. They compose naturally.

## Acceptance Criteria

- [ ] `consilium --council "test" --rounds 3` uses context compression after round 1+, visible cost reduction vs v0.1.3
- [ ] `consilium --council "test" --thorough --rounds 3` skips compression and consensus early exit
- [ ] `consilium --quick "test" --feedback` prompts for 1-5 rating after completion, rating appears in history.jsonl
- [ ] `consilium --feedback` with invalid input (empty, "abc", 0, 6) silently skips — no crash
- [ ] `consilium --stats` shows enriched output: percentiles, ranking, feedback averages
- [ ] `consilium --stats` works with history.jsonl entries from v0.1.3 (no feedback field) — backward compatible
- [ ] `cargo clippy` — zero new warnings
- [ ] `cargo test` — all existing tests pass + new tests for compression, thorough, feedback, stats
- [ ] Oxford/redteam/socratic modes unaffected by compression changes

## Success Metrics

- Multi-round council cost drops 30-50% with compression enabled
- Feedback data collection works for personal workflow tracking
- Stats output useful enough to inform feature prioritization

## Dependencies & Risks

- **Compression quality**: Bad summaries could degrade debate quality. Mitigated by using compression only for convergence modes and preserving full transcripts for judge.
- **Cheap model availability**: If the compression model is unavailable via OpenRouter, fallback to no compression. Must not break the session.
- **Additional latency**: Compression adds one API call per round (~2-3s). Acceptable given it saves much more on subsequent rounds.

## Implementation Notes for Delegation

This plan is designed for delegation to OpenCode/Gemini CLI. Key context:

1. **Rust regex crate**: No lookahead/lookbehind. Delegates commonly port Python patterns — always `cargo clippy` after.
2. **Output trait**: `&mut dyn Output` threads through all mode functions. New output must go through this trait.
3. **Error-as-string pattern**: `query_model()` returns `String`. Errors start with `[Error:`. No `Result<>` that forces early returns.
4. **CostTracker**: `AtomicU64` micro-dollars. Pass existing tracker to compression calls.
5. **Test location**: Unit tests in `#[cfg(test)]` modules alongside source. Integration tests in `tests/cli_test.rs`.
6. **Version**: Bump Cargo.toml to `0.1.4`. Update test in `tests/cli_test.rs` that checks version string.

## Delegation Strategy

Split into 4 independent tasks for parallel delegation:

| Task | Delegatee | Files | Complexity |
|------|-----------|-------|------------|
| Context compression | OpenCode or Gemini CLI | council.rs, discuss.rs, config.rs | High |
| --feedback flag | OpenCode (free) | cli.rs, main.rs, session.rs | Low |
| --thorough flag | OpenCode (free) | cli.rs, main.rs, council.rs, discuss.rs | Low |
| Stats enrichment | OpenCode (free) | admin.rs | Medium |

Tasks 2-4 are independent and can run in parallel. Task 1 depends on Task 3 (--thorough gates compression). Run Task 3 first or together with Task 1.

## Sources

- Council self-deliberation session: `~/notes/consilium - Self-Deliberation on Improvements.md`
- Cascading summarization pattern: `~/docs/solutions/cascading-llm-summarization.md`
- Multi-agent deliberation design: `~/docs/solutions/best-practices/multi-agent-deliberation-design.md`
- Judge over-aggregation gotcha: `~/docs/solutions/ai-tooling/llm-council-judge-over-aggregation.md`
- Streaming gotchas: `~/docs/solutions/consilium-streaming-gotchas.md`
