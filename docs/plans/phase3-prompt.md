# Phase 3 Delegation Prompt: Council Mode

```
Phase 3 of consilium-rs: Implement the full council deliberation mode.

REFERENCE FILES (read these first):
- reference/council.py — full council implementation (779 lines)
- reference/models.py — detect_consensus, parse_confidence, extract_structured_summary

EXISTING CODE (read to understand types/APIs):
- src/config.rs — SessionResult, ModelEntry, CostTracker, Message, detect_consensus, parse_confidence, sanitize_speaker_content, COUNCIL, JUDGE_MODEL, CRITIQUE_MODEL, EXTRACTION_MODEL
- src/api.rs — query_model, query_model_streaming, run_parallel, query_model_async
- src/prompts.rs — all prompt functions (council_blind_system, council_debate_system, etc.)
- src/session.rs — finish_session, save_session
- src/modes/mod.rs — mode dispatch

TASK: Create src/modes/council.rs

Port the full council mode from reference/council.py:

1. decompose_question() — breaks complex questions into sub-questions via judge model
2. run_blind_phase_parallel() — async parallel blind claims from all council members
3. run_xpol_phase_parallel() — cross-pollination (each model extends gaps)
4. run_followup_discussion() — focused topic discussion with first two models
5. run_council() — main orchestration:
   a. Blind phase: parallel claims
   b. Anonymous name mapping: display_names = {name: "Speaker {i+1}"}
   c. Cross-pollination (optional, if --xpol)
   d. Debate rounds with rotating challenger: (challenger_idx + round_num) % council_config.len()
   e. First speaker gets special prompt with blind context
   f. Challenger gets COUNCIL_CHALLENGER_ADDITION prompt
   g. Consensus detection via detect_consensus() — break early if converged
   h. Judge synthesis using ACH method (if --no-judge not set)
   i. CollabEval: critique (phase 2) + judge revision (phase 3)
   j. Structured extraction: _parse_recommendation_items (regex) + LLM extraction
   k. Followup discussion (optional, if --followup)
   l. Anonymous name reverse substitution at end
   m. Format output: prose, json, yaml

6. Port extract_structured_summary() and helpers:
   - _parse_recommendation_items() — regex extracts Do Now/Consider Later/Skip
   - _extract_for_llm() — extracts interpretive sections for LLM summarization
   - EXTRACTION_PROMPT constant (already in prompts.rs)

7. Update src/modes/mod.rs to export council mode
8. Update src/main.rs to dispatch --council to run_council

Key patterns:
- CostTracker.add() for cost tracking (not list append)
- query_model returns String, errors start with "[Error:" or "[No response"
- All prompt functions are in prompts.rs
- detect_consensus is in config.rs

Implement fully. No stubs, no placeholder comments, no TODO markers.
Run `cargo check` and `cargo test` after changes.
```
