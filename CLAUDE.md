# consilium — Project Instructions

## Overview

Multi-model deliberation CLI in Rust. Queries frontier LLMs via OpenRouter, runs structured debate, Claude Opus judges. ~7,100 lines across 14 source files.

## Build & Test

```bash
cargo build --release        # Binary: target/release/consilium
cargo test                   # 62 tests (58 unit + 4 integration)
cargo clippy                 # Must pass with 0 errors
```

The release binary is symlinked from `~/.local/bin/consilium`. After code changes, rebuild with `cargo build --release`.

## Architecture

- **Single tokio runtime** — all mode functions are `async fn`, one `#[tokio::main]`
- **Error-as-string** — `query_model()` returns `String`, errors start with `[Error:`. No `Result<>` that forces early returns. Partial failures don't abort sessions.
- **CostTracker** — `AtomicU64` storing micro-dollars, lock-free across tokio tasks
- **Output trait** — `StdoutOutput` / `TeeOutput` / `CompactTeeOutput` implementations, threaded through all mode functions. Lifecycle hooks: `begin_phase` owns `streaming_phase`; `begin_participant` must not reset it.
- **Manual SSE parsing** — ~40 lines in `api.rs`. Handles DeepSeek-R1 `<think>` blocks and OpenAI `reasoning_details`

## Key patterns

- **Prompts as functions** — `fn council_blind_system(name: &str) -> String` with `format!()`. Role library as match expression in `prompts.rs`.
- **Parallel queries** — `run_parallel()` spawns tokio tasks per panelist. `run_parallel_with_different_messages()` for per-panelist prompts.
- **Consensus detection** — excludes challenger from agreement counting. See `detect_consensus()` in `config.rs` with 14 test cases.
- **Rotating challenger** — `(challenger_idx + round_num) % council.len()`
- **LazyLock statics** for compiled regexes

## Gotchas

- **Rust `regex` crate has no lookahead/lookbehind.** `(?=...)`, `(?!...)` all fail. Use `(?:...)` or character classes instead.
- **`cargo build --release` must be re-run** after source changes for the symlinked binary to update.
- **Google AI Studio fallback** only activates for Gemini models. Message format conversion happens in `query_google_ai_studio()`.
- **SSE parsing** assumes OpenRouter format. Two reasoning token formats: DeepSeek-R1 (`<think>` in `delta.content`) and OpenAI (`delta.reasoning_details`).
- **Council composition changes require README update.** The models table in `README.md` must be kept in sync with `resolved_council()` in `config.rs`. The `COUNCIL` const is dead code — only `resolved_council()` matters at runtime.

## Module map

| File | Lines | Purpose |
|------|-------|---------|
| `config.rs` | 1097 | Constants, types, CostTracker, utility functions, 58 tests |
| `api.rs` | 1579 | HTTP clients, SSE streaming, parallel queries, retry, fallback |
| `prompts.rs` | 719 | All prompt templates (verbatim port from Python) |
| `session.rs` | 859 | Output trait + CompactTeeOutput, LiveWriter, session save/share/history |
| `modes/council.rs` | 1509 | Full council deliberation |
| `modes/discuss.rs` | 527 | Roundtable + socratic |
| `modes/oxford.rs` | 334 | Oxford debate |
| `modes/quick.rs` | 411 | Parallel streaming |
| `modes/redteam.rs` | 273 | Adversarial stress-test |
| `modes/premortem.rs` | 218 | Pre-mortem: assume failure, work backward |
| `modes/forecast.rs` | 219 | Superforecasting: probability estimates + reconciliation |
| `admin.rs` | 440 | Stats, sessions, view, search |
| `tui.rs` | 433 | Ratatui TUI (Flexoki dark) |
| `watch.rs` | 275 | Crossterm live watcher |
| `cli.rs` | 235 | Clap derive struct (~30 flags) |
| `main.rs` | 368 | Entry point + mode dispatch |

## Testing

Unit tests live alongside source in `#[cfg(test)]` modules. Integration tests in `tests/cli_test.rs` use `assert_cmd`.

Key test categories:
- Social context detection (8 cases)
- Consensus detection with challenger exclusion (14 cases)
- Speaker content sanitization (8 cases)
- Thinking model identification (8 cases)
- Challenger rotation (3 cases)
- SSE parsing (5 cases)
- CLI flag parsing (4 integration tests)

## Release workflow

1. Bump version in `Cargo.toml`
2. Update version string in `tests/cli_test.rs`
3. `cargo clippy && cargo test`
4. `cargo build --release`
5. Commit, push, `cargo publish`
6. Site: if landing page needs update, edit `consilium-site/public/index.html`, push (Vercel auto-deploys)

## Maintenance

Module map line counts are **auto-updated by pre-commit hook** (`scripts/update-module-map.py`). No manual step needed. New files must still be added to the map manually (with purpose description) — the hook only updates counts for existing entries.

## Delegation notes

This codebase was built via delegation to Codex and Gemini CLI. When delegating future work:
- Provide the `llms.txt` file for context
- Delegates commonly generate Python-style `(?=...)` lookahead regex — always `cargo clippy` after
- The `Output` trait (`&mut dyn Output`) threads through all mode functions — new modes must accept it
- Tasks touching different files can run in parallel; same-file tasks must be phased
- Gemini tends to touch more files than asked (bonus clippy fixes, global allows) — review diff scope
- Delegates don't write tests unless explicitly tasked — make it a separate delegation if needed

## API key architecture

- **`COUNCIL` const is dead code.** `resolved_council()` in `config.rs` is the runtime source of truth for council composition and fallback providers. Editing `COUNCIL` has zero runtime effect.
- **`query_judge()` reads `ANTHROPIC_API_KEY` from env directly** (not threaded through function signatures). This avoids cascading changes across 30+ callers. The pattern: leaf function reads `std::env::var("KEY")`, tries native API, falls back to OpenRouter if key absent or on error.
- **Native API fallback pattern (all providers):** each provider tries its own direct API first (Moonshot.cn, xAI, OpenAI, Google AI Studio, Zhipu, Anthropic), falls back to OpenRouter. Keys loaded from macOS Keychain in `.zshenv`.
