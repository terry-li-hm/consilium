# Phase 4 Delegation Prompt: Remaining Modes

```
Phase 4 of consilium-rs: Implement oxford, redteam, discuss, and solo modes.

REFERENCE FILES (read these first):
- reference/oxford.py — Oxford debate (369 lines)
- reference/redteam.py — Red team stress test (283 lines)
- reference/discuss.py — Roundtable + Socratic discussion (308 lines)
- reference/solo.py — Solo council / self-debate (273 lines)

EXISTING CODE:
- src/config.rs — all types, model configs, CostTracker
- src/api.rs — query_model, run_parallel
- src/prompts.rs — all prompt functions
- src/modes/mod.rs — mode dispatch

TASK 1: src/modes/oxford.rs
Port oxford debate from reference/oxford.py:
- _assign_sides() — random shuffle of debaters
- run_oxford() — main orchestration:
  a. Motion generation (judge transforms question to debate motion)
  b. Random side assignment
  c. Prior (judge gives initial probability 0-100)
  d. Constructive speeches (parallel, 800 tokens)
  e. Rebuttals (parallel, 600 tokens)
  f. Closing statements (parallel, 400 tokens)
  g. Verdict (judge evaluates, prior→posterior, winner, margin)

TASK 2: src/modes/redteam.rs
Port red team from reference/redteam.py:
- run_redteam() — main orchestration:
  a. Host analysis (identifies attack vectors)
  b. Parallel attacks (all attackers, 600 tokens)
  c. Host deepening (identifies most dangerous attack)
  d. Sequential attacker deepening (cascading failures)
  e. Host triage (severity-ranks vulnerabilities)

TASK 3: src/modes/discuss.rs
Port discussion from reference/discuss.py:
- run_discuss() — main orchestration supporting both styles:
  a. style="roundtable": collaborative exploration
  b. style="socratic": examiner probes panelists
  c. Opening (host frames or poses questions)
  d. Panelist opening takes (parallel, 500 tokens)
  e. Discussion rounds (host steers, panelists respond)
  f. rounds=0 means unlimited until Ctrl+C (use tokio::select! + signal::ctrl_c())
  g. Closing (parallel panelist closing + host synthesis)

TASK 4: src/modes/solo.rs
Port solo council from reference/solo.py:
- run_solo() — main orchestration:
  a. Parse roles from --roles or use SOLO_DEFAULT_ROLES
  b. Blind phase (parallel, same model different roles, 500 tokens)
  c. Debate (sequential, 500 tokens, challenger = perspectives[1])
  d. Judge synthesis

TASK 5: Update src/modes/mod.rs
Export all modes. Add Mode enum with dispatch function.

TASK 6: Update src/main.rs
Wire all modes to CLI flags.

Implement fully. No stubs, no placeholder comments, no TODO markers.
Run `cargo check` and `cargo test` after changes.
```
