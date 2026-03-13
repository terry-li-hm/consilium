# consilium Context
<!-- Updated: 2026-03-13 -->

## What it does
Multi-model deliberation CLI — queries frontier LLMs via OpenRouter, runs structured debate, Claude Opus judges.

## State
v0.5.3. Published on crates.io. Rust rewrite complete. All modes working (quick, council, discuss, oxford, redteam, premortem, forecast).

## Last session
Added `SilentOutput` for piped contexts (CC background tasks). When stdout is a pipe, consilium now suppresses output and prints only the session file path at exit — prevents pipe buffer overflow on long deliberations.

## Next
Implement LRN-20260312-001: auto-enable `--vault` when `--deep` or `--xpol` is passed.

## Open questions
None.
