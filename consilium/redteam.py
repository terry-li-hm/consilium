"""Red team mode: adversarial stress-test of a plan or decision."""

import asyncio
import time

import httpx

from .models import (
    DISCUSS_HOST,
    SessionResult,
    is_thinking_model,
    query_model,
    query_model_async,
    query_google_ai_studio,
    sanitize_speaker_content,
)
from .prompts import (
    DOMAIN_CONTEXTS,
    REDTEAM_HOST_ANALYSIS,
    REDTEAM_ATTACKER_SYSTEM,
    REDTEAM_HOST_DEEPEN,
    REDTEAM_ATTACKER_DEEPEN,
    REDTEAM_HOST_TRIAGE,
)


def run_redteam(
    question: str,
    panelists: list[tuple[str, str, tuple[str, str] | None]],
    api_key: str,
    google_api_key: str | None = None,
    verbose: bool = True,
    persona: str | None = None,
    domain: str | None = None,
) -> SessionResult:
    """Run red team mode: adversarial stress-test of a plan or decision. Returns SessionResult."""
    start_time = time.time()
    cost_accumulator: list[float] = []
    host_model = DISCUSS_HOST
    host_name = "Host (Claude)"

    domain_context = DOMAIN_CONTEXTS.get(domain, "") if domain else ""

    transcript_parts = []
    conversation_history: list[tuple[str, str]] = []

    def _persona_suffix() -> str:
        if persona:
            return f"\n\nContext about the person asking: {persona}"
        return ""

    def _domain_suffix() -> str:
        if domain_context:
            return f"\n\nDomain context: {domain_context}"
        return ""

    # === Phase 1: ANALYSIS ===
    if verbose:
        print("=" * 60)
        print("RED TEAM")
        print("=" * 60)
        print()

    # Host analysis
    analysis_system = REDTEAM_HOST_ANALYSIS + _persona_suffix() + _domain_suffix()
    analysis_messages = [
        {"role": "system", "content": analysis_system},
        {"role": "user", "content": question},
    ]

    if verbose:
        print("## Analysis\n")
        print(f"### {host_name}")

    host_analysis = query_model(
        api_key, host_model, analysis_messages,
        max_tokens=500, stream=verbose, cost_accumulator=cost_accumulator,
    )

    if verbose:
        print()

    transcript_parts.append(f"## Analysis\n\n### {host_name}\n{host_analysis}")
    conversation_history.append((host_name, host_analysis))

    # Parallel attacks
    if verbose:
        print("## Attacks\n")

    transcript_parts.append("## Attacks")

    # Build per-panelist messages for personalized attack prompts
    parallel_panelists = []
    parallel_messages_list = []
    for name, model, fallback in panelists:
        attacker_system = REDTEAM_ATTACKER_SYSTEM.format(
            name=name, host_analysis=sanitize_speaker_content(host_analysis),
        ) + _persona_suffix() + _domain_suffix()
        msgs = [
            {"role": "system", "content": attacker_system},
            {"role": "user", "content": f"Plan/decision to attack:\n\n{question}"},
        ]
        parallel_panelists.append((name, model, fallback))
        parallel_messages_list.append(msgs)

    async def _run_attacks():
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=180.0,
        ) as client:
            tasks = [
                query_model_async(
                    client, model, parallel_messages_list[i], name, fallback,
                    google_api_key,
                    max_tokens=600,
                    cost_accumulator=cost_accumulator,
                )
                for i, (name, model, fallback) in enumerate(parallel_panelists)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        out = []
        for i, result in enumerate(results):
            name, model, fallback = parallel_panelists[i]
            model_name = model.split("/")[-1]
            if isinstance(result, Exception):
                out.append((name, model_name, f"[Error: {result}]"))
            else:
                out.append(result)
        return out

    if verbose:
        print(f"(querying {len(panelists)} attackers in parallel...)")

    attack_results = asyncio.run(_run_attacks())

    for name, model_name, response in attack_results:
        if verbose:
            print(f"\n### {name}")
            print(response)
        transcript_parts.append(f"### {name}\n{response}")
        conversation_history.append((name, response))

    if verbose:
        print()

    # === Phase 2: DEEPENING ===
    if verbose:
        print("## Deepening\n")

    transcript_parts.append("## Deepening")

    # Host deepening prompt
    attacks_text = "\n\n".join(
        f"**{speaker}**: {sanitize_speaker_content(text)}"
        for speaker, text in conversation_history[1:]  # Skip host analysis
    )

    deepen_system = REDTEAM_HOST_DEEPEN + _persona_suffix()
    deepen_messages = [
        {"role": "system", "content": deepen_system},
        {"role": "user", "content": f"Plan/decision:\n{question}\n\nInitial attacks:\n\n{attacks_text}"},
    ]

    if verbose:
        print(f"### {host_name}")

    host_deepen = query_model(
        api_key, host_model, deepen_messages,
        max_tokens=300, stream=verbose, cost_accumulator=cost_accumulator,
    )

    if verbose:
        print()

    transcript_parts.append(f"### {host_name}\n{host_deepen}")
    conversation_history.append((host_name, host_deepen))

    # Sequential deepening from each attacker
    for i, (name, model, fallback) in enumerate(panelists):
        other_names = [n for n, _, _ in panelists if n != name]
        attacker_deepen_system = REDTEAM_ATTACKER_DEEPEN.format(
            name=name,
            other1=other_names[0] if len(other_names) > 0 else "another attacker",
            other2=other_names[1] if len(other_names) > 1 else "another attacker",
        ) + _persona_suffix() + _domain_suffix()

        full_history = "\n\n".join(
            f"**{speaker}**: {sanitize_speaker_content(text)}"
            for speaker, text in conversation_history
        )

        deepen_attacker_messages = [
            {"role": "system", "content": attacker_deepen_system},
            {"role": "user", "content": (
                f"Plan/decision:\n{question}\n\n"
                f"Discussion so far:\n\n{full_history}\n\n"
                "Find cascading/compound failures."
            )},
        ]

        if verbose:
            print(f"### {name}")
            if is_thinking_model(model):
                print("(thinking...)", flush=True)

        response = query_model(
            api_key, model, deepen_attacker_messages,
            stream=verbose, cost_accumulator=cost_accumulator,
        )

        # Handle fallback
        used_fallback = False
        if response.startswith("[") and fallback:
            fallback_provider, fallback_model = fallback
            if fallback_provider == "google" and google_api_key:
                if verbose:
                    print(f"(OpenRouter failed, trying AI Studio fallback: {fallback_model}...)", flush=True)
                response = query_google_ai_studio(google_api_key, fallback_model, deepen_attacker_messages)
                used_fallback = True

        if verbose and (is_thinking_model(model) or used_fallback):
            print(response)

        if verbose:
            print()

        transcript_parts.append(f"### {name}\n{response}")
        conversation_history.append((name, response))

    # === Phase 3: TRIAGE ===
    if verbose:
        print("## Triage\n")

    transcript_parts.append("## Triage")

    full_history = "\n\n".join(
        f"**{speaker}**: {sanitize_speaker_content(text)}"
        for speaker, text in conversation_history
    )

    triage_system = REDTEAM_HOST_TRIAGE + _persona_suffix() + _domain_suffix()
    triage_messages = [
        {"role": "system", "content": triage_system},
        {"role": "user", "content": f"Plan/decision:\n{question}\n\nFull red team discussion:\n\n{full_history}"},
    ]

    if verbose:
        print(f"### {host_name}")

    host_triage = query_model(
        api_key, host_model, triage_messages,
        max_tokens=800, stream=verbose, cost_accumulator=cost_accumulator,
    )

    if verbose:
        print()

    transcript_parts.append(f"### {host_name}\n{host_triage}")

    duration = time.time() - start_time
    total_cost = round(sum(cost_accumulator), 4) if cost_accumulator else 0.0

    if verbose:
        print(f"({duration:.1f}s, ~${total_cost:.2f})")

    return SessionResult(
        transcript="\n\n".join(transcript_parts),
        cost=total_cost,
        duration=duration,
    )
