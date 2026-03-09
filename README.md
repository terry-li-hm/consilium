# consilium

Multi-model deliberation CLI. 5 frontier LLMs debate your question, then Claude Opus 4.6 judges and synthesizes a recommendation.

## How it works

1. **Blind phase** — Each model answers independently (no herding)
2. **Cross-pollination** — Models read all blind claims and investigate gaps
3. **Debate** — Structured rounds with a rotating challenger ensuring sustained disagreement
4. **Judge** — Claude Opus synthesizes using Analysis of Competing Hypotheses
5. **CollabEval** — Gemini critiques the judge's synthesis; judge revises
6. **Extraction** — Structured Do Now / Consider Later / Skip recommendations

Auto-routes by difficulty: simple questions get quick parallel queries (~$0.10), complex ones get full council deliberation (~$0.50).

## Why multi-model?

Multi-model collaboration produces better outcomes than any single model — but naive "ask multiple models and vote" fails. Most of the benefit from debate comes from majority voting alone; the iterative discussion is a [martingale](https://openreview.net/forum?id=iUjGNJzrF1) that doesn't improve expected correctness without targeted interventions. consilium implements those interventions:

| Design principle | What it prevents | Evidence |
|---|---|---|
| Cross-lab panel (5 labs) | Shared training misconceptions reinforcing wrong answers | [ReConcile, ACL 2024](https://arxiv.org/html/2309.13007v3) — +11.4% over homogeneous MAD |
| Blind first phase | Identity/authority anchoring (Compassion-Fade bias) | [CALM, 2024](https://llm-judge-bias.github.io/) |
| Anonymous labels (Speaker 1/2/3) | Judge favoring named model outputs | [IBC paper, 2024](https://arxiv.org/html/2510.07517v1) — conformity > obstinacy when identity known |
| Rotating challenger | Sycophancy cascades; models converging to majority | [DEBATE, ACL 2024](https://aclanthology.org/2024.findings-acl.112/) |
| Anti-capitulation prompt | Position changes without new evidence | [Peacemaker/Troublemaker, 2024](https://arxiv.org/html/2509.23055v1) — r=0.902 sycophancy↔wrong answer |
| 1–2 round cap | Sycophancy intensifies at round 3+ | [Peacemaker/Troublemaker, 2024](https://arxiv.org/html/2509.23055v1) |
| Judge outside panelist pool | Self-Enhancement (judge favoring same-lab outputs) | [CALM, 2024](https://llm-judge-bias.github.io/) |
| Confidence extraction for judge | Equal-weighting confident and uncertain positions | [ReConcile, ACL 2024](https://arxiv.org/html/2309.13007v3) |
| Gemini critique + judge revision | Fallacy-Oversight (worst judge bias, 0.566 score) | [CALM, 2024](https://llm-judge-bias.github.io/) |

## Models

| Role | Model | Lab |
|------|-------|-----|
| Panelist | GPT-5.2 Pro | OpenAI |
| Panelist | Claude Sonnet 4.6 | Anthropic |
| Panelist | Grok-4.20 | xAI |
| Panelist | DeepSeek V3.2 | DeepSeek |
| Panelist | GLM-5 | Zhipu |
| Judge | Gemini 3.1 Pro | Google |
| Critique | Claude Opus 4.6 | Anthropic |

5 distinct labs on the panel. The judge (Google) is not in the panelist pool.

## Install

```bash
# Build from source
cargo build --release

# Binary at target/release/consilium
# Optionally symlink:
ln -s $(pwd)/target/release/consilium ~/.local/bin/consilium
```

Requires [OpenRouter](https://openrouter.ai/) API key:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
export GOOGLE_API_KEY=AIza...  # optional, Gemini fallback
```

## Usage

```bash
# Auto-route (Opus picks the best mode)
consilium "Should I take this job offer?"

# Quick parallel — independent opinions, no debate (~$0.10)
consilium "Best practices for error handling in Rust?" --quick

# Full council with JSON output (~$0.50)
consilium "Microservices vs monolith for a 5-person startup?" --council --format json

# Deep — auto-decompose + 2 debate rounds (~$0.90)
consilium "How should banks govern AI agents?" --deep

# Oxford debate — binary for/against + verdict (~$0.40)
consilium "Should we adopt Kubernetes?" --oxford

# Red team — adversarial stress-test (~$0.20)
consilium "My plan: rewrite the backend in Rust over 6 months" --redteam

# Roundtable discussion (~$0.30)
consilium "Future of open-source AI models?" --discuss --rounds 2

# Socratic examination (~$0.30)
consilium "Is consciousness computable?" --socratic
```

## Modes

| Mode | Flag | Cost | Description |
|------|------|------|-------------|
| Auto | *(default)* | varies | Opus classifies and picks the best mode |
| Quick | `--quick` | ~$0.10 | Parallel queries, no debate |
| Council | `--council` | ~$0.50 | Full multi-round deliberation + judge |
| Deep | `--deep` | ~$0.90 | Council + decompose + 2 rounds |
| Oxford | `--oxford` | ~$0.40 | Binary for/against debate |
| Red Team | `--redteam` | ~$0.20 | Adversarial stress-test |
| Discuss | `--discuss` | ~$0.30 | Hosted roundtable |
| Socratic | `--socratic` | ~$0.30 | Assumption-probing examination |

## Key flags

```
--persona "context"         Personal context injected into prompts
--domain banking|healthcare Domain-specific regulatory context
--challenger gemini         Assign contrarian role (council mode)
--decompose                 Break question into sub-questions first
--xpol                      Cross-pollination phase (council mode)
--followup                  Interactive drill-down after synthesis (council mode)
--rounds N                  Rounds for discuss/socratic (0 = unlimited)
--effort low|medium|high    Per-phase reasoning effort for thinking models
--thorough                  Disable early consensus exit
--output file.md            Save transcript to file
--format json|yaml|prose    Output format
--share                     Upload to secret GitHub gist
--quiet                     Suppress live output (auto-enabled when not a TTY)
--no-save                   Don't auto-save session
```

## Session management

```bash
consilium --stats           # Cost breakdown by mode
consilium --sessions        # List recent sessions
consilium --view            # View latest in pager
consilium --view "career"   # View session matching term
consilium --search "AI"     # Search session content
consilium --watch           # Live tail (styled terminal)
consilium --tui             # Full TUI viewer
consilium --doctor          # Check API keys and connectivity
```

## Architecture

~6,100 lines of Rust. Single 4.7MB binary, ~50ms cold start.

- Single tokio runtime, async throughout
- SSE streaming with `<think>` block filtering (DeepSeek-R1, OpenAI reasoning)
- CostTracker via AtomicU64 (micro-dollars, lock-free across tasks)
- Output trait enables TeeOutput (stdout + live file) for watch/TUI
- LiveWriter with PID-based file management and stale cleanup

## Prior art

Rewritten from [consilium-py](https://github.com/terry-li-hm/consilium-py) (Python, 5,091 lines). Same CLI interface, same modes, same output format, same `~/.consilium/` session directory. Both versions can coexist and read each other's sessions.

## License

MIT
