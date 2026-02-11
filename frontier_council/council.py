"""Core council deliberation logic."""

import asyncio
import httpx
import json
import re
import time
import yaml
from datetime import datetime
from pathlib import Path

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GOOGLE_AI_STUDIO_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MOONSHOT_URL = "https://api.moonshot.cn/v1/chat/completions"

# Model configurations (all via OpenRouter, with fallbacks where available)
# Format: (name, openrouter_model, fallback) - fallback is (provider, model) or None
# Providers: "google" = AI Studio, "moonshot" = Moonshot API
COUNCIL = [
    ("GPT", "openai/gpt-5.2-pro", None),
    ("Gemini", "google/gemini-3-pro-preview", ("google", "gemini-2.5-pro")),
    ("Grok", "x-ai/grok-4", None),
    ("Kimi", "moonshotai/kimi-k2.5", ("moonshot", "kimi-k2.5")),
]

# Claude is judge-only (not in council) to avoid conflict of interest
JUDGE_MODEL = "anthropic/claude-opus-4.5"

# Domain-specific regulatory contexts
DOMAIN_CONTEXTS = {
    "banking": "You are operating in a banking/financial services regulatory environment. Consider: HKMA/MAS/FCA requirements, Model Risk Management (MRM) expectations, audit trail needs, BCBS 239 governance, explainability requirements, documentation standards, and regulatory scrutiny levels.",
    "healthcare": "You are operating in a healthcare regulatory environment. Consider: HIPAA constraints on PHI handling, FDA requirements for medical devices, clinical validation expectations, interoperability standards (FHIR), GxP compliance, and patient safety requirements.",
    "eu": "You are operating in the EU regulatory environment. Consider: GDPR data protection requirements, EU AI Act risk categorization, Digital Markets Act compliance, cross-border data transfer rules (Schrems II), and EU data localization expectations.",
    "fintech": "You are operating in a fintech regulatory environment. Consider: KYC/AML requirements, PSD2 banking regulations, e-money licensing expectations, payment services directive compliance, and financial consumer protection rules.",
    "bio": "You are operating in a biotech/pharma regulatory environment. Consider: FDA/EMA drug approval processes, GMP manufacturing requirements, clinical trial design expectations, pharmacovigilance obligations, and post-market surveillance requirements.",
}

# Keywords that suggest social/conversational context (auto-detect)
SOCIAL_KEYWORDS = [
    "interview", "ask him", "ask her", "ask them", "question to ask",
    "networking", "outreach", "message", "email", "linkedin",
    "coffee chat", "informational", "reach out", "follow up",
    "what should i say", "how should i respond", "conversation",
]

# Thinking models - use non-streaming, higher tokens, longer timeout
THINKING_MODEL_SUFFIXES = {
    "claude-opus-4.5",
    "gpt-5.2-pro",
    "gemini-3-pro-preview",
    "grok-4",
    "kimi-k2.5",
    "deepseek-r1",
    "o1-preview", "o1-mini", "o1",
    "o3-preview", "o3-mini", "o3",
}


def is_thinking_model(model: str) -> bool:
    """Check if model is a thinking model that doesn't stream well."""
    model_name = model.split("/")[-1].lower()
    return model_name in THINKING_MODEL_SUFFIXES


def detect_social_context(question: str) -> bool:
    """Auto-detect if the question is about social/conversational context."""
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in SOCIAL_KEYWORDS)


def query_model(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 1500,
    timeout: float = 120.0,
    stream: bool = False,
    retries: int = 2,
) -> str:
    """Query a model via OpenRouter with retry logic for flaky models."""
    if is_thinking_model(model):
        max_tokens = max(max_tokens, 2500)
        timeout = max(timeout, 180.0)

    if stream and not is_thinking_model(model):
        result = query_model_streaming(api_key, model, messages, max_tokens, timeout)
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

        data = response.json()

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
            return f"[Error: Request failed for AI Studio {model}: {e}]"

    return f"[Error: Failed to get response from AI Studio {model}]"


def query_moonshot(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 8192,
    timeout: float = 120.0,
    retries: int = 2,
) -> str:
    """Query Moonshot API directly (fallback for Kimi models)."""
    for attempt in range(retries + 1):
        try:
            response = httpx.post(
                MOONSHOT_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                },
                timeout=timeout,
            )

            if response.status_code != 200:
                if attempt < retries:
                    continue
                return f"[Error: HTTP {response.status_code} from Moonshot {model}]"

            data = response.json()

            if "error" in data:
                if attempt < retries:
                    continue
                return f"[Error: {data['error'].get('message', data['error'])}]"

            if "choices" not in data or not data["choices"]:
                if attempt < retries:
                    continue
                return f"[Error: No response from Moonshot {model}]"

            content = data["choices"][0]["message"]["content"]

            if not content or not content.strip():
                if attempt < retries:
                    continue
                return f"[No response from Moonshot {model} after {retries + 1} attempts]"

            return content

        except httpx.TimeoutException:
            if attempt < retries:
                continue
            return f"[Error: Timeout from Moonshot {model}]"
        except httpx.RequestError as e:
            if attempt < retries:
                continue
            return f"[Error: Request failed for Moonshot {model}: {e}]"

    return f"[Error: Failed to get response from Moonshot {model}]"


def query_model_streaming(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 1500,
    timeout: float = 120.0,
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
    moonshot_api_key: str | None = None,
    max_tokens: int = 500,
    retries: int = 2,
) -> tuple[str, str, str]:
    """Async query for parallel blind phase. Returns (name, model_name, response)."""
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

            return (name, model_name, content)

        except (httpx.RequestError, httpx.RemoteProtocolError):
            if attempt < retries:
                continue
            break

    # Try fallbacks synchronously
    if fallback:
        fallback_provider, fallback_model = fallback
        if fallback_provider == "google" and google_api_key:
            response = query_google_ai_studio(google_api_key, fallback_model, messages, max_tokens=max_tokens)
            return (name, fallback_model, response)
        elif fallback_provider == "moonshot" and moonshot_api_key:
            response = query_moonshot(moonshot_api_key, fallback_model, messages, max_tokens=max_tokens)
            return (name, fallback_model, response)

    return (name, model_name, f"[No response from {model_name} after {retries + 1} attempts]")


async def run_blind_phase_parallel(
    question: str,
    council_config: list[tuple[str, str, tuple[str, str] | None]],
    api_key: str,
    google_api_key: str | None = None,
    moonshot_api_key: str | None = None,
    verbose: bool = True,
    persona: str | None = None,
    domain_context: str = "",
    practical_mode: bool = False,
    practical_constraint: str = "",
) -> list[tuple[str, str, str]]:
    """Parallel blind first-pass: all models stake claims simultaneously."""
    blind_system = """You are participating in the BLIND PHASE of a council deliberation.

Stake your initial position on the question BEFORE seeing what others think.
This prevents anchoring bias.

Provide a CLAIM SKETCH (not a full response):
1. Your core position (1-2 sentences)
2. Top 3 supporting claims or considerations
3. Key assumption or uncertainty

Keep it concise (~100 words). The full deliberation comes later.
Prioritize PRACTICAL, ACTIONABLE advice over academic observations."""

    if practical_mode and practical_constraint:
        blind_system += practical_constraint

    if domain_context:
        blind_system += f"""

DOMAIN CONTEXT: {domain_context}

Apply this regulatory domain context to your analysis."""

    if persona:
        blind_system += f"""

IMPORTANT CONTEXT about the person asking:
{persona}

Factor this into your advice — don't just give strategically optimal answers, consider what fits THIS person."""

    if verbose:
        print("=" * 60)
        print("BLIND PHASE (independent claims)")
        print("=" * 60)
        print()

    messages = [
        {"role": "system", "content": blind_system},
        {"role": "user", "content": f"Question:\n\n{question}"},
    ]

    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120.0,
    ) as client:
        tasks = [
            query_model_async(
                client, model, messages, name, fallback,
                google_api_key, moonshot_api_key
            )
            for name, model, fallback in council_config
        ]

        if verbose:
            print("(querying all models in parallel...)")

        results = await asyncio.gather(*tasks, return_exceptions=True)

    blind_claims = []
    for i, result in enumerate(results):
        name, model, _ = council_config[i]
        model_name = model.split("/")[-1]

        if isinstance(result, Exception):
            blind_claims.append((name, model_name, f"[Error: {result}]"))
        else:
            blind_claims.append(result)

    if verbose:
        print()
        for name, model_name, claims in blind_claims:
            print(f"### {model_name} (blind)")
            print(claims)
            print()

    return blind_claims


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


EXTRACTION_MODEL = "anthropic/claude-haiku-4-5"

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
) -> dict:
    extracted = {}

    # Try LLM extraction if API key available
    if api_key:
        try:
            messages = [
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": judge_response},
            ]
            raw = query_model(api_key, EXTRACTION_MODEL, messages, max_tokens=800, timeout=30.0)
            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            extracted = json.loads(raw)
        except (json.JSONDecodeError, Exception):
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


def run_followup_discussion(
    question: str,
    topic: str,
    council_config: list[tuple[str, str, tuple[str, str] | None]],
    api_key: str,
    domain_context: str = "",
    social_mode: bool = False,
    persona: str | None = None,
    verbose: bool = True,
) -> str:
    """Run a focused followup discussion on a specific topic with 2 models. Returns the followup transcript."""
    # Use judge (Claude 0) and one other model (GPT 1) for followup
    followup_models = council_config[:2]  # Claude and GPT
    
    followup_transcript_parts = []
    
    if verbose:
        print()
        print("=" * 60)
        print(f"FOLLOWUP: {topic}")
        print("=" * 60)
        print()
    
    social_constraint = """

SOCIAL CALIBRATION: This is a social/conversational context (interview, networking, outreach).
Your output should feel natural in conversation - something you'd actually say over coffee.
Avoid structured, multi-part diagnostic questions that sound like interrogation.
Simple and human beats strategic and comprehensive. Optimize for being relatable, not thorough.""" if social_mode else ""
    
    followup_parts = [
        "You are participating in a FOCUSED FOLLOWUP discussion on a specific topic.",
        "",
        f"The main council has concluded, and we're now drilling down into:",
        f"TOPIC: {topic}",
        "",
        "Keep your response focused on this specific topic. Don't rehash the full council deliberation.",
        "Be concise and practical.",
        "",
    ]
    
    if social_constraint:
        followup_parts.append(social_constraint.strip())
    
    if persona:
        followup_parts.extend([
            "",
            "IMPORTANT CONTEXT about the person asking:",
            persona,
            "",
            "Factor this into your advice — don't just give strategically optimal answers, consider what fits THIS person.",
        ])
    
    if domain_context:
        followup_parts.extend([
            "",
            f"DOMAIN CONTEXT: {domain_context}",
            "",
            "Apply this regulatory domain context to your analysis.",
        ])
    
    followup_system = "\n".join(followup_parts)
    
    followup_transcript_parts.append(f"### Followup Discussion: {topic}\n")
    
    for i, (name, model, fallback) in enumerate(followup_models):
        messages = [
            {"role": "system", "content": followup_system},
            {"role": "user", "content": f"Original Question:\n\n{question}\n\nFocus your response on: {topic}"},
        ]
        
        if verbose:
            print(f"### {name}")
        
        response = query_model(api_key, model, messages, stream=verbose)
        
        if verbose:
            print()
        
        followup_transcript_parts.append(f"### {name}\n{response}\n")
    
    if verbose:
        print("=" * 60)
        print("FOLLOWUP COMPLETE")
        print("=" * 60)
        print()
    
    return "\n\n".join(followup_transcript_parts)


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
    practical_mode: bool = False,
    persona: str | None = None,
    domain: str | None = None,
    challenger_idx: int | None = None,  # Starting challenger index, rotates each round
    format: str = "prose",
) -> tuple[str, list[str]]:
    """Run the council deliberation. Returns (transcript, failed_models)."""

    start_time = time.time()
    
    domain_context = DOMAIN_CONTEXTS.get(domain, "") if domain else ""
    council_names = [name for name, _, _ in council_config]
    blind_claims = []
    failed_models = []

    practical_constraint = ""
    if practical_mode:
        practical_constraint = """

PRACTICAL MODE: Focus on actionable triggers and concrete decision rules.
- Every claim must end with a specific action or rule ("do X when Y")
- Avoid cognitive science, philosophy, abstract theory, or speculation about human cognition
- No "it depends" hedging — commit to a threshold or trigger
- If you can't make it actionable, skip it"""

    if blind:
        blind_claims = asyncio.run(run_blind_phase_parallel(
            question,
            council_config,
            api_key,
            google_api_key,
            moonshot_api_key,
            verbose,
            persona,
            domain_context,
            practical_mode,
            practical_constraint,
        ))
        for name, model_name, claims in blind_claims:
            if claims.startswith("["):
                failed_models.append(f"{model_name} (blind): {claims}")

    if anonymous:
        display_names = {name: f"Speaker {i+1}" for i, (name, _, _) in enumerate(council_config)}
    else:
        display_names = {name: name for name, _, _ in council_config}

    if verbose:
        print(f"Council members: {council_names}")
        if anonymous:
            print("(Models see each other as Speaker 1, 2, etc. to prevent bias)")
        print(f"Rounds: {rounds}")
        if domain:
            print(f"Domain context: {domain}")
        print(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        print()
        print("=" * 60)
        print("COUNCIL DELIBERATION")
        print("=" * 60)
        print()

    conversation = []
    output_parts = []
    current_round = 0

    if blind_claims:
        for name, model_name, claims in blind_claims:
            output_parts.append(f"### {model_name} (blind)\n{claims}")

    blind_context = ""
    if blind_claims:
        blind_lines = []
        for name, _, claims in blind_claims:
            dname = display_names[name]
            blind_lines.append(f"**{dname}**: {sanitize_speaker_content(claims)}")
        blind_context = "\n\n".join(blind_lines)

    social_constraint = """

SOCIAL CALIBRATION: This is a social/conversational context (interview, networking, outreach).
Your output should feel natural in conversation - something you'd actually say over coffee.
Avoid structured, multi-part diagnostic questions that sound like interrogation.
Simple and human beats strategic and comprehensive. Optimize for being relatable, not thorough."""

    first_speaker_with_blind = """You are {name}, speaking first in Round {round_num} of a council deliberation.

You've seen everyone's BLIND CLAIMS (their independent initial positions). Now engage:
1. Reference at least ONE other speaker's blind claim
2. Agree, disagree, or build on their position
3. Develop your own position further based on what you've learned

Be direct. Challenge weak arguments. Don't be sycophantic.
Prioritize PRACTICAL, ACTIONABLE advice over academic observations. Avoid jargon."""

    first_speaker_system = """You are {name}, speaking first in Round {round_num} of a council deliberation.

As the first speaker, stake a clear position on the question. Be specific and substantive so others can engage with your points.
Prioritize PRACTICAL, ACTIONABLE advice over academic observations. Avoid jargon.

End with 2-3 key claims that others should respond to."""

    council_system = """You are {name}, participating in Round {round_num} of a council deliberation.

REQUIREMENTS for your response:
1. Reference at least ONE previous speaker by name (e.g., "I agree with Speaker 1 that..." or "Speaker 2's point about X overlooks...")
2. State explicitly: AGREE, DISAGREE, or BUILD ON their specific point
3. Add ONE new consideration not yet raised
4. Keep response under 250 words — be concise and practical

If you fully agree with emerging consensus, say: "CONSENSUS: [the agreed position]"

Previous speakers this round: {previous_speakers}

Be direct. Challenge weak arguments. Don't be sycophantic.
Prioritize PRACTICAL, ACTIONABLE advice over academic observations. Avoid jargon."""

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

    for round_num in range(rounds):
        current_round = round_num + 1
        round_speakers = []
        for idx, (name, model, fallback) in enumerate(council_config):
            dname = display_names[name]

            if idx == 0 and round_num == 0:
                if blind_claims:
                    system_prompt = first_speaker_with_blind.format(name=dname, round_num=round_num + 1)
                else:
                    system_prompt = first_speaker_system.format(name=dname, round_num=round_num + 1)
            else:
                if round_speakers:
                    previous = ", ".join(round_speakers)
                else:
                    previous = ", ".join([display_names[n] for n, _, _ in council_config])
                system_prompt = council_system.format(
                    name=dname,
                    round_num=round_num + 1,
                    previous_speakers=previous
                )

            if domain_context:
                system_prompt += f"""

DOMAIN CONTEXT: {domain_context}

Apply this regulatory domain context to your analysis."""

            if social_mode:
                system_prompt += social_constraint

            if practical_mode:
                system_prompt += practical_constraint

            if persona:
                system_prompt += f"""

IMPORTANT CONTEXT about the person asking:
{persona}

Factor this into your advice — don't just give strategically optimal answers, consider what fits THIS person."""

            # Calculate rotating challenger for this round
            if challenger_idx is not None:
                # Explicit --challenger sets starting point, then rotates
                current_challenger = (challenger_idx + round_num) % len(council_config)
            else:
                # Default: start with Claude (index 0), rotate through council
                current_challenger = round_num % len(council_config)

            if idx == current_challenger:
                system_prompt += challenger_addition

            user_content = f"Question for the council:\n\n{question}"
            if blind_context:
                user_content += f"\n\n---\n\nBLIND CLAIMS (independent initial positions):\n\n{blind_context}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            for speaker, text in conversation:
                speaker_dname = display_names[speaker]
                sanitized_text = sanitize_speaker_content(text)
                messages.append({
                    "role": "assistant" if speaker == name else "user",
                    "content": f"[{speaker_dname}]: {sanitized_text}" if speaker != name else sanitized_text,
                })

            model_name = model.split("/")[-1]
            challenger_indicator = " (challenger)" if idx == current_challenger else ""

            if verbose:
                print(f"### {model_name}{challenger_indicator}")
                if is_thinking_model(model):
                    print("(thinking...)", flush=True)

            response = query_model(api_key, model, messages, stream=verbose)

            used_fallback = False
            if response.startswith("[") and fallback:
                fallback_provider, fallback_model = fallback

                if fallback_provider == "google" and google_api_key:
                    if verbose:
                        print(f"(OpenRouter failed, trying AI Studio fallback: {fallback_model}...)", flush=True)
                    response = query_google_ai_studio(google_api_key, fallback_model, messages)
                    used_fallback = True
                    model_name = fallback_model

                elif fallback_provider == "moonshot" and moonshot_api_key:
                    if verbose:
                        print(f"(OpenRouter failed, trying Moonshot fallback: {fallback_model}...)", flush=True)
                    response = query_moonshot(moonshot_api_key, fallback_model, messages)
                    used_fallback = True
                    model_name = fallback_model

            if verbose and (is_thinking_model(model) or used_fallback):
                print(response)

            if response.startswith("["):
                failed_models.append(f"{model_name}: {response}")

            conversation.append((name, response))
            round_speakers.append(dname)

            if verbose:
                print()

            output_parts.append(f"### {model_name}{challenger_indicator}\n{response}")

        # current_challenger already calculated in the speaker loop above
        converged, reason = detect_consensus(conversation, council_config, current_challenger)
        if converged:
            if verbose:
                print(f">>> CONSENSUS DETECTED ({reason}) - proceeding to judge\n")
            break

    # Judge synthesis
    context_hint = ""
    if context:
        context_hint = f"\n\nContext about this question: {context}\nConsider this context when weighing perspectives and forming recommendations."
    
    domain_hint = ""
    if domain_context:
        domain_hint = f"\n\nDOMAIN CONTEXT: {domain}\nConsider this regulatory domain context when weighing perspectives and forming recommendations."

    social_judge_section = ""
    if social_mode:
        social_judge_section = """

## Social Calibration Check
[Would the recommendation feel natural in conversation? Is it something you'd actually say, or does it sound like strategic over-optimization? If the council produced something too formal/structured, suggest a simpler, more human alternative.]"""

    judge_system = f"""You are the Judge (Claude), responsible for synthesizing the council's deliberation.{context_hint}{domain_hint}

You did NOT participate in the deliberation — you're seeing it fresh. This gives you objectivity.

After the council members have shared their perspectives, you:
1. Identify points of AGREEMENT across all members
2. Identify points of DISAGREEMENT and explain the different views
3. Add YOUR OWN perspective — what did the council miss? What's your independent take?
4. Provide a SYNTHESIS that integrates the council's views with your own
5. Give a final RECOMMENDATION based on everything
{"6. SOCIAL CALIBRATION: Check if the recommendation would feel natural in actual conversation" if social_mode else ""}

Format your response as:

## Points of Agreement
[What the council agrees on]

## Points of Disagreement
[Where views differ and why]

## Judge's Own Take
[Your independent perspective. What did the council miss or underweight? What would YOU add to this discussion?]

## Synthesis
[The integrated perspective, combining council views with your own]

## Recommendation
[Your final recommendation]
{social_judge_section}
Be balanced and fair. Acknowledge minority views. But don't be afraid to have your own opinion — you're the judge, not just a summarizer.{" For social contexts, prioritize natural/human output over strategic optimization." if social_mode else ""}

CRITICAL — PRESCRIPTION DISCIPLINE:
Your job is to FILTER, not aggregate. The council will generate many suggestions. Most are interesting but not necessary.

Rules:
- **Do Now** — MAX 3 items. For each, first argue AGAINST including it (cost, risk, unnecessary). Only include it if it survives your own counter-argument. If you can't argue against it, it's probably essential.
- **Consider Later** — Items that are interesting but not worth doing now
- **Skip** — Explicitly list council suggestions you're DROPPING and why

The council's gravitational pull is toward "add more." Your gravitational pull must be toward "do less." A recommendation with 6 action items is not a recommendation — it's a wish list.

Don't recommend building infrastructure for problems that don't exist yet."""

    if practical_mode:
        judge_system += """

PRACTICAL MODE: The council was asked for actionable triggers and concrete rules. Filter aggressively — drop any recommendation that is philosophical, abstract, or not immediately actionable. Every Do Now item must be a specific rule or trigger, not a habit or mindset shift."""

    deliberation_text = "\n\n".join(
        f"**{display_names[speaker]}**: {sanitize_speaker_content(text)}" for speaker, text in conversation
    )

    judge_messages = [
        {"role": "system", "content": judge_system},
        {"role": "user", "content": f"Question:\n{question}\n\n---\n\nCouncil Deliberation:\n\n{deliberation_text}"},
    ]

    judge_model_name = JUDGE_MODEL.split("/")[-1]

    if verbose:
        print(f"### Judge ({judge_model_name})")

    judge_response = query_model(api_key, JUDGE_MODEL, judge_messages, max_tokens=1200, stream=verbose)

    if verbose:
        print()

    output_parts.append(f"### Judge ({judge_model_name})\n{judge_response}")

    if format != 'prose':
        structured = extract_structured_summary(
            judge_response=judge_response,
            question=question,
            models_used=[name for name, _, _ in council_config],
            rounds=current_round if rounds > 0 else 1,
            duration=time.time() - start_time,
            cost=0.85,
            api_key=api_key,
        )
        
        if format == 'json':
            output_parts.append('\n\n---\n\n' + json.dumps(structured, indent=2, ensure_ascii=False))
        else:
            output_parts.append('\n\n---\n\n' + yaml.dump(structured, allow_unicode=True, default_flow_style=False))

    if anonymous:
        final_output = "\n\n".join(output_parts)
        for name, model, _ in council_config:
            anon_name = display_names[name]
            model_name = model.split("/")[-1]
            final_output = final_output.replace(f"### {anon_name}", f"### {model_name}")
            final_output = final_output.replace(f"[{anon_name}]", f"[{model_name}]")
            final_output = final_output.replace(f"**{anon_name}**", f"**{model_name}**")
            final_output = final_output.replace(f"with {anon_name}", f"with {model_name}")
            final_output = final_output.replace(f"{anon_name}'s", f"{model_name}'s")
        return final_output, failed_models

    return "\n\n".join(output_parts), failed_models
