"""Model configurations, query functions, and shared helpers."""

import asyncio
import httpx
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SessionResult:
    """Structured return from all mode functions."""
    transcript: str
    cost: float
    duration: float

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GOOGLE_AI_STUDIO_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Model configurations (all via OpenRouter, with Google AI Studio fallback for Gemini)
# Format: (name, openrouter_model, fallback) - fallback is (provider, model) or None
COUNCIL = [
    ("GPT", "openai/gpt-5.2-pro", None),
    ("Gemini", "google/gemini-3-pro-preview", ("google", "gemini-2.5-pro")),
    ("Grok", "x-ai/grok-4", None),
    ("DeepSeek", "deepseek/deepseek-r1", None),
    ("GLM", "z-ai/glm-5", None),
]

# Claude is judge-only (not in council) to avoid conflict of interest
JUDGE_MODEL = "anthropic/claude-opus-4-6"
# Critique model for CollabEval phase 2 (strongest analytical reasoner, not Claude)
CRITIQUE_MODEL = "google/gemini-3-pro-preview"
# Classification model for auto-routing — Haiku is fast (~0.5s) and accurate enough for 3-class
CLASSIFIER_MODEL = "anthropic/claude-haiku-4-5"

# Quick mode: council models + Claude (no judge conflict in quick mode)
QUICK_MODELS = [("Claude", JUDGE_MODEL, None)] + [(n, m, fb) for n, m, fb in COUNCIL]

# Discussion mode: 3 panelists + Claude as host
DISCUSS_MODELS = COUNCIL[:3]  # GPT, Gemini, Grok
DISCUSS_HOST = JUDGE_MODEL     # Claude hosts

# Red team mode: same 3-model panel, Claude hosts
REDTEAM_MODELS = COUNCIL[:3]  # GPT, Gemini, Grok

# Oxford debate: 2 debaters, Claude judges
OXFORD_MODELS = COUNCIL[:2]  # GPT, Gemini

# Thinking models - use non-streaming, higher tokens, longer timeout
THINKING_MODEL_SUFFIXES = {
    "claude-opus-4-6", "claude-opus-4.5",
    "gpt-5.2-pro", "gpt-5.2",
    "gemini-3-pro-preview",
    "grok-4",
    "deepseek-r1",
    "glm-5",
}

# Keywords that suggest social/conversational context (auto-detect)
SOCIAL_KEYWORDS = [
    "interview", "ask him", "ask her", "ask them", "question to ask",
    "networking", "outreach", "message", "email", "linkedin",
    "coffee chat", "informational", "reach out", "follow up",
    "what should i say", "how should i respond", "conversation",
]

# Extraction model for structured summaries
EXTRACTION_MODEL = "anthropic/claude-haiku-4-5"


def is_thinking_model(model: str) -> bool:
    """Check if model is a thinking model that doesn't stream well."""
    model_name = model.split("/")[-1].lower()
    return model_name in THINKING_MODEL_SUFFIXES


def detect_social_context(question: str) -> bool:
    """Auto-detect if the question is about social/conversational context."""
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in SOCIAL_KEYWORDS)


def classify_difficulty(
    question: str,
    api_key: str,
    cost_accumulator: list[float] | None = None,
) -> str:
    """Classify question difficulty for DAAO routing. Returns 'simple', 'moderate', or 'complex'."""
    messages = [
        {"role": "system", "content": """Classify this question's deliberation difficulty as exactly one of: simple, moderate, complex.

simple: Factual questions, straightforward comparisons, well-known answers, single-dimension questions
moderate: Multi-faceted trade-off analysis, technical decisions with 2-3 competing factors, "should I X or Y" choices
complex: High-stakes decisions with many interacting variables, deeply nuanced strategic/ethical questions, decisions with irreversible consequences

Respond with ONLY the single word: simple, moderate, or complex."""},
        {"role": "user", "content": question},
    ]
    response = query_model(
        api_key, CLASSIFIER_MODEL, messages,
        max_tokens=10, timeout=15.0,
        cost_accumulator=cost_accumulator,
    )
    result = response.strip().lower().rstrip(".")
    if result in ("simple", "moderate", "complex"):
        return result
    return "moderate"


def query_model(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 1500,
    timeout: float = 120.0,
    stream: bool = False,
    retries: int = 2,
    cost_accumulator: list[float] | None = None,
) -> str:
    """Query a model via OpenRouter with retry logic for flaky models."""
    if is_thinking_model(model):
        max_tokens = max(max_tokens, 2500)
        timeout = max(timeout, 180.0)

    if stream and not is_thinking_model(model):
        result = query_model_streaming(api_key, model, messages, max_tokens, timeout, cost_accumulator=cost_accumulator)
        if not result.startswith("["):
            return result
        print("(Streaming failed, retrying without streaming...)", flush=True)

    for attempt in range(retries + 1):
        try:
            response = httpx.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                },
                timeout=timeout,
            )
        except (httpx.RequestError, httpx.RemoteProtocolError) as e:
            if attempt < retries:
                continue
            return f"[Error: Connection failed for {model}: {e}]"

        if response.status_code != 200:
            if attempt < retries:
                continue
            return f"[Error: HTTP {response.status_code} from {model}]"

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError):
            if attempt < retries:
                continue
            return f"[Error: Invalid JSON response from {model}]"

        if "error" in data:
            if attempt < retries:
                continue
            return f"[Error: {data['error'].get('message', data['error'])}]"

        if "choices" not in data or not data["choices"]:
            if attempt < retries:
                continue
            return f"[Error: No response from {model}]"

        content = data["choices"][0]["message"]["content"]

        if not content or not content.strip():
            reasoning = data["choices"][0]["message"].get("reasoning", "")
            if reasoning and reasoning.strip():
                if attempt < retries:
                    continue
                return f"[Model still thinking - needs more tokens. Partial reasoning: {reasoning[:150]}...]"
            if attempt < retries:
                continue
            return f"[No response from {model} after {retries + 1} attempts]"

        if "<think>" in content:
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        if cost_accumulator is not None:
            usage = data.get("usage", {})
            cost = usage.get("cost")
            if cost is not None:
                cost_accumulator.append(float(cost))

        return content

    return f"[Error: Failed to get response from {model}]"


def query_google_ai_studio(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 8192,
    timeout: float = 120.0,
    retries: int = 2,
) -> str:
    """Query Google AI Studio directly (fallback for Gemini models)."""
    contents = []
    system_instruction = None

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_instruction = content
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})

    body = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
        }
    }
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    url = f"{GOOGLE_AI_STUDIO_URL}/{model}:generateContent?key={api_key}"

    for attempt in range(retries + 1):
        try:
            response = httpx.post(url, json=body, timeout=timeout)

            if response.status_code != 200:
                if attempt < retries:
                    continue
                return f"[Error: HTTP {response.status_code} from AI Studio {model}]"

            data = response.json()

            if "error" in data:
                if attempt < retries:
                    continue
                return f"[Error: {data['error'].get('message', data['error'])}]"

            candidates = data.get("candidates", [])
            if not candidates:
                if attempt < retries:
                    continue
                return f"[Error: No candidates from AI Studio {model}]"

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                if attempt < retries:
                    continue
                return f"[Error: No content from AI Studio {model}]"

            content = parts[0].get("text", "")
            if not content.strip():
                if attempt < retries:
                    continue
                return f"[No response from AI Studio {model} after {retries + 1} attempts]"

            return content

        except httpx.TimeoutException:
            if attempt < retries:
                continue
            return f"[Error: Timeout from AI Studio {model}]"
        except httpx.RequestError as e:
            if attempt < retries:
                continue
            return f"[Error: Request failed for AI Studio {model}]"

    return f"[Error: Failed to get response from AI Studio {model}]"


def query_model_streaming(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 1500,
    timeout: float = 120.0,
    cost_accumulator: list[float] | None = None,
) -> str:
    """Query a model with streaming output - prints tokens as they arrive."""
    import json as json_module

    full_content = []
    in_think_block = False
    error_msg = None

    try:
        with httpx.stream(
            "POST",
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": True,
            },
            timeout=timeout,
        ) as response:
            if response.status_code != 200:
                error_msg = f"[Error: HTTP {response.status_code} from {model}]"
            else:
                for line in response.iter_lines():
                    if not line or line.startswith(":"):
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            data = json_module.loads(data_str)
                            if "error" in data:
                                error_msg = f"[Error: {data['error'].get('message', data['error'])}]"
                                break

                            # Final chunk with usage/cost has empty choices
                            if cost_accumulator is not None and "usage" in data:
                                cost = data["usage"].get("cost")
                                if cost is not None:
                                    cost_accumulator.append(float(cost))

                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    if "<think>" in content:
                                        in_think_block = True
                                    if in_think_block:
                                        if "</think>" in content:
                                            in_think_block = False
                                            content = content.split("</think>", 1)[-1]
                                        else:
                                            continue

                                    if content:
                                        print(content, end="", flush=True)
                                        full_content.append(content)
                        except json_module.JSONDecodeError:
                            pass

    except httpx.TimeoutException:
        error_msg = f"[Error: Timeout from {model}]"
    except (httpx.RequestError, httpx.RemoteProtocolError) as e:
        error_msg = f"[Error: Connection failed for {model}: {e}]"

    print()

    if error_msg:
        print(error_msg)
        return error_msg

    if not full_content:
        empty_msg = f"[No response from {model}]"
        print(empty_msg)
        return empty_msg

    return "".join(full_content)


async def query_model_async(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict],
    name: str,
    fallback: tuple[str, str] | None = None,
    google_api_key: str | None = None,
    max_tokens: int = 500,
    retries: int = 2,
    cost_accumulator: list[float] | None = None,
) -> tuple[str, str, str]:
    """Async query for parallel phases. Returns (name, model_name, response)."""
    if is_thinking_model(model):
        max_tokens = max(max_tokens, 1500)

    model_name = model.split("/")[-1]

    for attempt in range(retries + 1):
        try:
            response = await client.post(
                OPENROUTER_URL,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                },
            )

            if response.status_code != 200:
                if attempt < retries:
                    continue
                break

            data = response.json()

            if "error" in data:
                if attempt < retries:
                    continue
                break

            if "choices" not in data or not data["choices"]:
                if attempt < retries:
                    continue
                break

            content = data["choices"][0]["message"]["content"]

            if not content or not content.strip():
                reasoning = data["choices"][0]["message"].get("reasoning", "")
                if reasoning and reasoning.strip():
                    if attempt < retries:
                        continue
                    return (name, model_name, f"[Model still thinking - increase max_tokens. Partial: {reasoning[:200]}...]")
                if attempt < retries:
                    continue
                break

            if "<think>" in content:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            if cost_accumulator is not None:
                usage = data.get("usage", {})
                cost = usage.get("cost")
                if cost is not None:
                    cost_accumulator.append(float(cost))

            return (name, model_name, content)

        except (httpx.RequestError, httpx.RemoteProtocolError):
            if attempt < retries:
                continue
            break

    # Try fallback (Google AI Studio)
    if fallback:
        fallback_provider, fallback_model = fallback
        if fallback_provider == "google" and google_api_key:
            response = query_google_ai_studio(google_api_key, fallback_model, messages, max_tokens=max_tokens)
            return (name, fallback_model, response)

    return (name, model_name, f"[No response from {model_name} after {retries + 1} attempts]")


async def run_parallel(
    panelists: list[tuple[str, str, tuple[str, str] | None]],
    messages: list[dict],
    api_key: str,
    google_api_key: str | None = None,
    max_tokens: int = 500,
    cost_accumulator: list[float] | None = None,
) -> list[tuple[str, str, str]]:
    """Parallel query panelists with shared messages. Returns [(name, model_name, response)]."""
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=180.0,
    ) as client:
        tasks = [
            query_model_async(
                client, model, messages, name, fallback,
                google_api_key, max_tokens=max_tokens,
                cost_accumulator=cost_accumulator,
            )
            for name, model, fallback in panelists
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    out = []
    for i, result in enumerate(results):
        name, model, fallback = panelists[i]
        model_name = model.split("/")[-1]
        if isinstance(result, Exception):
            out.append((name, model_name, f"[Error: {result}]"))
        else:
            out.append(result)
    return out


_CONFIDENCE_RE = re.compile(
    r'\*{0,2}Confidence\*{0,2}:?\s*(\d{1,2})\s*(?:/\s*10|out\s+of\s+10)',
    re.IGNORECASE,
)


def parse_confidence(response: str) -> int | None:
    """Extract Confidence: N/10 from a debate response."""
    match = _CONFIDENCE_RE.search(response)
    if match:
        value = int(match.group(1))
        return value if 0 <= value <= 10 else None
    return None


def sanitize_speaker_content(content: str) -> str:
    """Sanitize speaker content to prevent prompt injection."""
    sanitized = content.replace("SYSTEM:", "[SYSTEM]:")
    sanitized = sanitized.replace("INSTRUCTION:", "[INSTRUCTION]:")
    sanitized = sanitized.replace("IGNORE PREVIOUS", "[IGNORE PREVIOUS]")
    sanitized = sanitized.replace("OVERRIDE:", "[OVERRIDE]:")
    return sanitized


def detect_consensus(
    conversation: list[tuple[str, str]],
    council_config: list[tuple[str, str, tuple[str, str] | None]],
    current_challenger_idx: int | None = None,
) -> tuple[bool, str]:
    """Detect if council has converged. Returns (converged, reason).

    Excludes the current challenger from consensus count since they're
    structurally incentivized to disagree.
    """
    council_size = len(council_config)

    if len(conversation) < council_size:
        return False, "insufficient responses"

    recent = conversation[-council_size:]

    # Exclude challenger from consensus count
    if current_challenger_idx is not None:
        challenger_name = council_config[current_challenger_idx][0]
        recent = [(name, text) for name, text in recent if name != challenger_name]

    effective_size = len(recent)
    if effective_size == 0:
        return False, "no non-challenger responses"

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


EXTRACTION_PROMPT = """Extract a structured JSON summary from this judge synthesis.

Return ONLY valid JSON (no markdown fences, no commentary) matching this schema:

{
  "decision": "The core recommendation in 1-2 sentences",
  "confidence": "high|medium|low",
  "reasoning_summary": "2-3 sentence summary of why",
  "dissents": [{"model": "model name", "concern": "what they disagreed on"}],
  "action_items": [{"action": "specific action", "priority": "high|medium|low"}],
  "do_now": ["action 1", "action 2", "action 3"],
  "consider_later": ["item 1", "item 2"],
  "skip": ["dropped item with reason"]
}

Rules:
- decision: the judge's final recommendation, not a section heading
- do_now: max 3 items from the judge's "Do Now" section
- action_items: all concrete actions mentioned, with priority
- dissents: real disagreements with the model name that raised them
- If a field has no content, use an empty list []"""


def extract_structured_summary(
    judge_response: str,
    question: str,
    models_used: list[str],
    rounds: int,
    duration: float,
    cost: float,
    api_key: str | None = None,
    cost_accumulator: list[float] | None = None,
) -> dict:
    extracted = {}

    # Try LLM extraction if API key available
    if api_key:
        try:
            messages = [
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": judge_response},
            ]
            raw = query_model(api_key, EXTRACTION_MODEL, messages, max_tokens=800, timeout=30.0, cost_accumulator=cost_accumulator)
            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            extracted = json.loads(raw)
        except (json.JSONDecodeError, KeyError, IndexError):
            pass  # Fall through to fallback

    # Fallback: minimal extraction from text
    if not extracted:
        lines = judge_response.split('\n')
        decision = ""
        for line in lines:
            line_lower = line.lower()
            if 'recommend' in line_lower or 'decision:' in line_lower:
                decision = line.strip()
                break
        if not decision:
            for line in lines:
                if len(line.strip()) > 20:
                    decision = line.strip()
                    break
        extracted = {
            "decision": decision[:500] if decision else "See transcript for details",
            "confidence": "medium",
            "reasoning_summary": judge_response[:1000],
            "dissents": [],
            "action_items": [],
        }

    # Always add meta and question
    extracted["schema_version"] = "1.0"
    extracted["question"] = question
    extracted["meta"] = {
        "timestamp": datetime.now().isoformat(),
        "models_used": models_used,
        "rounds": rounds,
        "duration_seconds": duration,
        "estimated_cost_usd": cost,
    }
    return extracted
