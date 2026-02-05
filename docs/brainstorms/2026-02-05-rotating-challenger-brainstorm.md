# Brainstorm: Rotating Challenger for Sustained Disagreement

**Date:** 2026-02-05
**Status:** Ready for planning

## What We're Building

Modify the frontier-council deliberation architecture so that the challenger role rotates each round instead of only firing in Round 1. This ensures someone is always structurally incentivized to push back, preventing the premature convergence observed in current transcripts.

Additionally, strengthen the challenger prompt to produce sharper disagreement.

## Why This Approach

### Problem Observed

From transcript analysis:
- Best insights came from genuine pushback (Kimi's "stop weaving, start haunting", Gemini's "dismantle the OpenCode delegation")
- These contrarian takes got softened by Round 2
- Models converge quickly because LLMs are trained to agree
- Current challenger/advocate roles only fire in Round 1, exactly when they're least needed

### Why Rotating Challenger

| Alternative | Why Not |
|-------------|---------|
| Position Locking | Requires state tracking, feels artificial, models may defend positions they don't believe |
| Adversarial Pairing | Major redesign, doesn't work for all question types |
| Just fix the judge | Symptom not cause — the deliberation itself converges too fast |

Rotating challenger is:
- Minimal code change (move the `if round_num == 0` check)
- Immediate impact on deliberation dynamics
- Easy to measure (compare transcripts before/after)

## Key Decisions

1. **Rotation pattern:** Sequential through council order (Claude → GPT → Gemini → Grok → Kimi → repeat)
2. **Prompt strengthening:** Add explicit requirements to challenger prompt:
   - Must name one specific thing that would make the emerging consensus WRONG
   - Must identify the weakest assumption being made
   - Cannot use phrases like "building on" or "adding nuance"
3. **Merge advocate and challenger:** Remove the redundant devil's advocate role. One challenger role is enough.
4. **All rounds:** Challenger fires every round, not just Round 1

## Open Questions

1. Should challenger be excluded from consensus detection? (If the challenger is forced to disagree, they shouldn't count toward "4/5 agree")
2. Should we track which model was challenger in the output metadata?
3. Does strengthening the prompt risk making disagreement feel forced/artificial?

## Success Criteria

- Transcripts show sustained disagreement through Round 2+
- Contrarian perspectives survive to judge synthesis
- Judge explicitly notes unresolved tensions (may need separate prompt tweak)

## Next Steps

Run `/workflows:plan` to create implementation plan.
