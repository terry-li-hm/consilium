---
title: "feat: Rotating Challenger for Sustained Disagreement"
type: feat
date: 2026-02-05
brainstorm: docs/brainstorms/2026-02-05-rotating-challenger-brainstorm.md
---

# feat: Rotating Challenger for Sustained Disagreement

## Overview

Modify frontier-council so the challenger role rotates each round instead of only firing in Round 1. This ensures someone is always structurally incentivized to push back, preventing the premature convergence observed in transcripts.

## Problem Statement

From transcript analysis:
- Best insights came from genuine pushback (Kimi's "stop weaving, start haunting", Gemini's "dismantle delegation")
- These contrarian takes got softened by Round 2
- Current challenger/advocate roles only apply in `round_num == 0`
- Devil's advocate and challenger prompts are **nearly identical** — redundant complexity

## Proposed Solution

1. **Merge advocate and challenger** into single "challenger" role
2. **Rotate challenger each round** — Claude R1 → GPT R2 → Gemini R3 → Grok R4 → Kimi R5 → wrap
3. **Strengthen challenger prompt** with explicit anti-convergence requirements
4. **Exclude challenger from consensus detection** so forced disagreement doesn't block early exit

## Technical Approach

### Files to Modify

| File | Changes |
|------|---------|
| `council.py` | Remove `devils_advocate_addition`, modify role application logic, update consensus detection |
| `cli.py` | Deprecate `--advocate` with warning, update `--challenger` semantics |

### Implementation Steps

#### Step 1: Merge Advocate and Challenger Prompts

**File:** `council.py` lines 783-837

Remove `devils_advocate_addition` (lines 783-794). Keep only `challenger_addition` and strengthen it:

```python
# council.py ~line 826
challenger_addition = """

SPECIAL ROLE: You are the CHALLENGER for this round. Your job is to argue the CONTRARIAN position.

REQUIREMENTS:
1. You MUST explicitly DISAGREE with at least one major point from the other speakers
2. Identify the weakest assumption in the emerging consensus and attack it
3. Name ONE specific thing that would make the consensus WRONG
4. You CANNOT use phrases like "building on", "adding nuance", or "I largely agree"
5. If everyone is converging too fast, that's a red flag — find the hidden complexity

Even if you ultimately agree with the direction, you MUST articulate the strongest possible counter-argument.
If you can't find real disagreement, explain why the consensus might be groupthink."""
```

#### Step 2: Rotate Challenger Each Round

**File:** `council.py` lines 879-883

Current code:
```python
if idx == advocate_idx and round_num == 0:
    system_prompt += devils_advocate_addition

if idx == challenger_idx and round_num == 0:
    system_prompt += challenger_addition
```

Replace with:
```python
# Calculate rotating challenger for this round
if challenger_idx is not None:
    # Explicit --challenger sets starting point, then rotates
    current_challenger = (challenger_idx + round_num) % len(council_config)
else:
    # Default: start with Claude (index 0), rotate through council
    current_challenger = round_num % len(council_config)

if idx == current_challenger:
    system_prompt += challenger_addition
```

#### Step 3: Update Consensus Detection

**File:** `council.py` lines 545-564

Modify `detect_consensus` to accept and exclude challenger:

```python
def detect_consensus(
    conversation: list[tuple[str, str]],
    council_config: list,
    current_challenger_idx: int | None = None
) -> tuple[bool, str]:
    """Detect if council has converged. Returns (converged, reason)."""
    council_size = len(council_config)

    if len(conversation) < council_size:
        return False, "insufficient responses"

    recent = conversation[-council_size:]

    # Exclude challenger from consensus count
    if current_challenger_idx is not None:
        challenger_name = council_config[current_challenger_idx][0]
        recent = [(name, text) for name, text in recent if name != challenger_name]

    effective_size = len(recent)
    threshold = effective_size - 1  # Need all-but-one non-challengers to agree

    consensus_count = sum(1 for _, text in recent if "CONSENSUS:" in text.upper())
    if consensus_count >= threshold:
        return True, "explicit consensus signals"

    agreement_phrases = ["i agree with", "i concur", "we all agree", "consensus emerging"]
    agreement_count = sum(
        1 for _, text in recent
        if any(phrase in text.lower() for phrase in agreement_phrases)
    )
    if agreement_count >= threshold:
        return True, "agreement language detected"

    return False, "no consensus"
```

Update the call site (~line 943):
```python
current_challenger = (challenger_idx + round_num) % len(council_config) if challenger_idx is not None else round_num % len(council_config)
converged, reason = detect_consensus(conversation, council_config, current_challenger)
```

#### Step 4: Deprecate --advocate Flag

**File:** `cli.py` lines 100-105 and 212

Add deprecation warning:
```python
# cli.py ~line 212
if args.advocate:
    print("Warning: --advocate is deprecated. Use --challenger instead.", file=sys.stderr)
    # Map speaker number (1-5) to model name for backward compat
    model_names = [n for n, _, _ in COUNCIL]
    mapped_model = model_names[args.advocate - 1]
    print(f"  Mapping --advocate {args.advocate} to --challenger {mapped_model.lower()}", file=sys.stderr)
    if not args.challenger:
        args.challenger = mapped_model.lower()
```

#### Step 5: Update Transcript Output

Show challenger indicator in round headers:

```python
# council.py ~line 908 (in the speaker output section)
challenger_indicator = " (challenger)" if idx == current_challenger else ""
output_parts.append(f"### {name}{challenger_indicator}\n{response}")
```

### Function Signature Changes

**`run_council`** (lines 706-723):
- Remove `advocate_idx` parameter
- Keep `challenger_idx` (now means "starting challenger")

```python
def run_council(
    question: str,
    council_config: list[tuple[str, str, tuple[str, str] | None]],
    api_key: str,
    google_api_key: str | None = None,
    moonshot_api_key: str | None = None,
    rounds: int = 1,
    verbose: bool = True,
    anonymous: bool = True,
    blind: bool = True,
    context: str | None = None,
    social_mode: bool = False,
    persona: str | None = None,
    # advocate_idx removed
    domain: str | None = None,
    challenger_idx: int | None = None,  # Now means "starting challenger"
    format: str = "prose",
) -> tuple[str, list[str]]:
```

## Acceptance Criteria

### Functional Requirements
- [x] Challenger role rotates each round (R1: model 0, R2: model 1, etc.)
- [x] `--challenger X` sets starting point, then rotates
- [x] `--advocate` shows deprecation warning and maps to `--challenger`
- [x] Challenger excluded from consensus detection
- [x] Transcript shows which model is challenger each round

### Non-Functional Requirements
- [x] No breaking changes to existing scripts (deprecation, not removal)
- [x] Tests pass for new rotation logic
- [x] README updated with new behavior

## Success Metrics

Compare transcripts before/after:
- Sustained disagreement through Round 2+
- Contrarian perspectives survive to judge synthesis
- Judge notes unresolved tensions (may need separate prompt tweak)

## Testing Plan

### Unit Tests

Add to `tests/test_utils.py`:

```python
class TestRotatingChallenger:
    def test_challenger_rotates_default(self):
        """Challenger rotates through council order by default."""
        # R0: index 0, R1: index 1, R2: index 2...
        assert get_challenger_for_round(None, 0, 5) == 0
        assert get_challenger_for_round(None, 1, 5) == 1
        assert get_challenger_for_round(None, 4, 5) == 4
        assert get_challenger_for_round(None, 5, 5) == 0  # wraps

    def test_challenger_rotates_from_explicit(self):
        """Explicit --challenger sets starting point."""
        # --challenger gemini (index 2): R0=2, R1=3, R2=4, R3=0...
        assert get_challenger_for_round(2, 0, 5) == 2
        assert get_challenger_for_round(2, 1, 5) == 3
        assert get_challenger_for_round(2, 3, 5) == 0  # wraps

class TestConsensusWithChallenger:
    def test_consensus_excludes_challenger(self):
        """Challenger's agreement doesn't count toward consensus."""
        conversation = [
            ("Claude", "CONSENSUS: I agree"),
            ("GPT", "CONSENSUS: agreed"),
            ("Gemini", "CONSENSUS: yes"),  # challenger
            ("Grok", "CONSENSUS: agreed"),
            ("Kimi", "different view"),
        ]
        council_config = [("Claude",), ("GPT",), ("Gemini",), ("Grok",), ("Kimi",)]
        # Gemini (index 2) is challenger, excluded
        # 3 of 4 non-challengers agree = consensus
        converged, _ = detect_consensus(conversation, council_config, 2)
        assert converged
```

### Integration Test

```bash
# Run with 3 rounds, verify rotation in transcript
frontier-council "test question" --rounds 3 --output /tmp/test.md
grep -E "### .+ \(challenger\)" /tmp/test.md
# Should show 3 different models as challenger
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Forced disagreement feels artificial | Prompt says "even if you ultimately agree" — models can agree after challenging |
| Breaking scripts using `--advocate` | Deprecation warning + automatic mapping, not hard removal |
| Consensus detection edge cases | Thorough unit tests for threshold math |

## References

- Brainstorm: `docs/brainstorms/2026-02-05-rotating-challenger-brainstorm.md`
- Current challenger impl: `council.py:826-837`
- Current advocate impl: `council.py:783-794` (to be removed)
- Consensus detection: `council.py:545-564`
- Deliberation loop: `council.py:839-947`
