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

## Module map

| File | Lines | Purpose |
|------|-------|---------|
| `config.rs` | 812 | Constants, types, CostTracker, utility functions, 58 tests |
| `api.rs` | 753 | HTTP clients, SSE streaming, parallel queries, retry, fallback |
| `prompts.rs` | 582 | All prompt templates (verbatim port from Python) |
| `session.rs` | 805 | Output trait + CompactTeeOutput, LiveWriter, session save/share/history |
| `modes/council.rs` | 1310 | Full council deliberation |
| `modes/discuss.rs` | 478 | Roundtable + socratic |
| `modes/oxford.rs` | 307 | Oxford debate |
| `modes/quick.rs` | 241 | Parallel streaming |
| `modes/redteam.rs` | 256 | Adversarial stress-test |
| `admin.rs` | 488 | Stats, sessions, view, search |
| `tui.rs` | 420 | Ratatui TUI (Flexoki dark) |
| `watch.rs` | 275 | Crossterm live watcher |
| `cli.rs` | 175 | Clap derive struct (~30 flags) |
| `main.rs` | 235 | Entry point + mode dispatch |

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

## Delegation notes

This codebase was built via delegation to Codex and Gemini CLI. When delegating future work:
- Provide the `llms.txt` file for context
- Delegates commonly generate Python-style `(?=...)` lookahead regex — always `cargo clippy` after
- The `Output` trait (`&mut dyn Output`) threads through all mode functions — new modes must accept it
- Tasks touching different files can run in parallel; same-file tasks must be phased
- Gemini tends to touch more files than asked (bonus clippy fixes, global allows) — review diff scope
- Delegates don't write tests unless explicitly tasked — make it a separate delegation if needed
