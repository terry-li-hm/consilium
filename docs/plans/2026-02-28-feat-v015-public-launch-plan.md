---
title: "feat: v0.1.5 public launch prep"
type: feat
status: active
date: 2026-02-28
---

# v0.1.5 — Public Launch Prep

Three independent tasks. All touch different files — safe to run in parallel.

## Task 1: Fix challenger-consensus early exit bug

**Problem:** `detect_consensus()` in `src/config.rs:151` excludes the challenger from the consensus count entirely. If 4/4 non-challengers agree, consensus triggers early exit — even if the challenger is actively dissenting with critical risk identification. For a deliberation tool, suppressing minority dissent is a correctness bug.

**Fix:** After checking non-challenger agreement, also check if the challenger's most recent response contains active disagreement language. If so, block early exit.

**File:** `src/config.rs`

**Implementation:**
1. In `detect_consensus()`, after the existing agreement checks return `(true, ...)`, add a challenger dissent check
2. If `current_challenger_idx` is Some, find the challenger's response in `recent`
3. Check for disagreement phrases: "i disagree", "i challenge", "this is wrong", "critical flaw", "fundamental problem", "overlooking", "must object"
4. If challenger is actively dissenting, return `(false, "challenger actively dissenting")`
5. Add 3-4 test cases in the existing `#[cfg(test)]` module:
   - Consensus blocked when challenger dissents
   - Consensus allowed when challenger also agrees
   - Consensus allowed when no challenger idx provided

**Lines:** ~30 new, ~10 modified

## Task 2: Add `--doctor` command

**Purpose:** First-run diagnostic. User runs `consilium --doctor`, gets a checklist of what's working.

**Files:** `src/cli.rs`, `src/main.rs`, `src/admin.rs`

**Implementation:**

In `src/cli.rs`, add to the admin commands section:
```rust
/// Run diagnostics (check API keys, connectivity, session directory)
#[arg(long)]
pub doctor: bool,
```
Update `is_admin_command()` to include `self.doctor`.

In `src/main.rs`, add dispatch after the other admin commands:
```rust
if args.doctor {
    admin::run_doctor().await;
    std::process::exit(0);
}
```
Note: `run_doctor` is async (makes HTTP request), so main already has `#[tokio::main]`.

In `src/admin.rs`, add `run_doctor()`:
1. Check `OPENROUTER_API_KEY` env var — print checkmark or X
2. Check `GOOGLE_API_KEY` env var — print checkmark or note "optional, Gemini fallback disabled"
3. Check session directory exists (`~/.consilium/sessions/`) — create if missing
4. Make one test request to OpenRouter: POST to `https://openrouter.ai/api/v1/chat/completions` with model `meta-llama/llama-3.3-70b-instruct`, message "ping", max_tokens 1. If 200, print checkmark. If 401, print "invalid key". If other error, print the error.
5. Print summary: "Ready to deliberate!" or "Fix the issues above before running."

Use crossterm for colored checkmarks (green check, red X) — the crate is already a dependency.

**Lines:** ~60 new

## Task 3: Update README with MoCo stat

**File:** `README.md`

Add a "Why multi-model?" section after "How it works", before "Models":

```markdown
## Why multi-model?

Research shows multi-model collaboration produces **18.5% better outcomes** than any single model working alone, through a mechanism called "collaborative emergence" — models surface insights that no individual model would reach. consilium's structured deliberation (blind → debate → judge) is designed to maximize this effect.

Paper: [Model Collaboration](https://arxiv.org/html/2601.21257v1) (Feng et al., 2025)
```

Also add `--doctor` to the session management section.

## Version bump

- `Cargo.toml`: 0.1.4 → 0.1.5
- `tests/cli_test.rs`: update version string check

## Verification

1. `cargo clippy` — zero errors
2. `cargo test` — all pass (including new consensus tests)
3. `consilium --doctor` — shows diagnostic output
4. `consilium --version` — shows 0.1.5
