# Frontier Council

Multi-model deliberation for important decisions. 5 frontier LLMs debate a question, then a judge synthesizes consensus.

Inspired by [Andrej Karpathy's LLM Council](https://github.com/karpathy/llm-council), with added blind phase (anti-anchoring), explicit engagement requirements, devil's advocate role, and social calibration mode.

## Models

- Claude (claude-opus-4.5)
- GPT (gpt-5.2-pro)
- Gemini (gemini-3-pro-preview)
- Grok (grok-4)
- Kimi (kimi-k2.5)
- Judge: Claude Opus 4.5

## Installation

```bash
pip install frontier-council
```

Or with uv:
```bash
uv tool install frontier-council
```

## Setup

Set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

Optional fallback keys (for flaky models):
```bash
export GOOGLE_API_KEY=AIza...      # Gemini fallback
export MOONSHOT_API_KEY=sk-...     # Kimi fallback
```

## Usage

```bash
# Basic question
frontier-council "Should we use microservices or monolith?"

# With social calibration (for interview/networking questions)
frontier-council "What questions should I ask in the interview?" --social

# With persona context
frontier-council "Should I take the job?" --persona "builder who hates process work"

# Multiple rounds
frontier-council "Architecture decision" --rounds 3

# Save transcript
frontier-council "Career question" --output transcript.md

# Share via GitHub Gist
frontier-council "Important decision" --share

# List past sessions
frontier-council --sessions
```

All sessions are auto-saved to `~/.frontier-council/sessions/` for later review.

## Options

| Flag | Description |
|------|-------------|
| `--rounds N` | Number of deliberation rounds (default: 2, exits early on consensus) |
| `--output FILE` | Save transcript to file |
| `--named` | Let models see real names during deliberation (may increase bias) |
| `--no-blind` | Skip blind first-pass (faster, but first speaker anchors others) |
| `--context TEXT` | Context hint for judge (e.g., "architecture decision") |
| `--share` | Upload transcript to secret GitHub Gist |
| `--social` | Enable social calibration mode (auto-detected for interview/networking) |
| `--persona TEXT` | Context about the person asking |
| `--advocate N` | Which speaker (1-5) should be devil's advocate (default: random) |
| `--quiet` | Suppress progress output |
| `--sessions` | List recent saved sessions |
| `--no-save` | Don't auto-save transcript to ~/.frontier-council/sessions/ |

## How It Works

**Blind First-Pass (Anti-Anchoring):**
1. All models generate short "claim sketches" independently and in parallel
2. This prevents the "first speaker lottery" where whoever speaks first anchors the debate
3. Each model commits to an initial position before seeing any other responses

**Deliberation Protocol:**
1. All models see everyone's blind claims, then deliberate
2. Each model MUST explicitly AGREE, DISAGREE, or BUILD ON previous speakers by name
3. After each round, the system checks for consensus (4/5 agreement triggers early exit)
4. Judge synthesizes the full deliberation

**Anonymous Deliberation:**
- Models see each other as "Speaker 1", "Speaker 2", etc. during deliberation
- Prevents models from playing favorites based on vendor reputation
- Output transcript shows real model names for readability

## When to Use

Use the council when:
- Making an important decision that benefits from diverse perspectives
- You want models to actually debate, not just answer in parallel
- You need a synthesized recommendation, not raw comparison
- Exploring trade-offs where different viewpoints matter

Skip the council when:
- You're just thinking out loud (exploratory discussions)
- The answer depends on personal preference more than objective trade-offs
- Speed matters (council takes 60-90 seconds)

## Python API

```python
from frontier_council import run_council, COUNCIL
import os

api_key = os.environ["OPENROUTER_API_KEY"]

transcript, failed_models = run_council(
    question="Should we use microservices or monolith?",
    council_config=COUNCIL,
    api_key=api_key,
    rounds=2,
    verbose=True,
    social_mode=False,
)

print(transcript)
```

## License

MIT
