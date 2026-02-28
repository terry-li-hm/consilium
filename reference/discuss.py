"""Discussion mode: hosted roundtable exploration."""

import asyncio
import time

from .models import (
    DISCUSS_HOST,
    SessionResult,
    query_model,
    query_google_ai_studio,
    run_parallel,
    sanitize_speaker_content,
)
from .prompts import (
    DOMAIN_CONTEXTS,
    DISCUSS_HOST_FRAMING,
    DISCUSS_PANELIST_SYSTEM,
    DISCUSS_HOST_STEER,
    DISCUSS_PANELIST_CLOSING,
    DISCUSS_HOST_CLOSING,
    SOCRATIC_HOST_OPENING,
    SOCRATIC_PANELIST_SYSTEM,
    SOCRATIC_HOST_PROBE,
    SOCRATIC_PANELIST_CLOSING,
    SOCRATIC_HOST_SYNTHESIS,
)


def run_discuss(
    question: str,
    panelists: list[tuple[str, str, tuple[str, str] | None]],
    api_key: str,
    google_api_key: str | None = None,
    verbose: bool = True,
    persona: str | None = None,
    domain: str | None = None,
    rounds: int = 2,
    style: str = "roundtable",
) -> SessionResult:
    """Run discussion mode: hosted roundtable exploration. Returns SessionResult.

    rounds: number of discussion rounds. 0 means unlimited (Ctrl+C to wrap up).
    style: "roundtable" (collaborative exploration) or "socratic" (probing questions).
    """
    start_time = time.time()
    cost_accumulator: list[float] = []
    host_model = DISCUSS_HOST
    is_socratic = style == "socratic"
    host_name = "Examiner (Claude)" if is_socratic else "Host (Claude)"

    domain_context = DOMAIN_CONTEXTS.get(domain, "") if domain else ""

    transcript_parts = []
    conversation_history: list[tuple[str, str]] = []

    def _build_history_text() -> str:
        return "\n\n".join(
            f"**{speaker}**: {sanitize_speaker_content(text)}"
            for speaker, text in conversation_history
        )

    def _persona_suffix() -> str:
        if persona:
            return f"\n\nContext about the person asking: {persona}"
        return ""

    def _domain_suffix() -> str:
        if domain_context:
            return f"\n\nDomain context: {domain_context}"
        return ""

    # === Phase 1: FRAMING / QUESTIONS ===
    if verbose:
        print("=" * 60)
        print("SOCRATIC EXAMINATION" if is_socratic else "ROUNDTABLE DISCUSSION")
        print("=" * 60)
        print()

    # Host framing / opening questions
    if is_socratic:
        framing_system = SOCRATIC_HOST_OPENING + _persona_suffix() + _domain_suffix()
    else:
        framing_system = DISCUSS_HOST_FRAMING + _persona_suffix() + _domain_suffix()
    framing_messages = [
        {"role": "system", "content": framing_system},
        {"role": "user", "content": question},
    ]

    opening_label = "Questions" if is_socratic else "Opening"
    if verbose:
        print(f"## {opening_label}\n")
        print(f"### {host_name}")

    host_framing = query_model(
        api_key, host_model, framing_messages,
        max_tokens=500, stream=verbose, cost_accumulator=cost_accumulator,
    )

    if verbose:
        print()

    transcript_parts.append(f"## {opening_label}\n\n### {host_name}\n{host_framing}")
    conversation_history.append((host_name, host_framing))

    # Panelist opening takes / answers (parallel)
    if is_socratic:
        opening_system = SOCRATIC_PANELIST_SYSTEM.format(
            name="a panelist", word_limit=200,
        ) + _persona_suffix() + _domain_suffix()
        opening_user = (
            f"Topic: {question}\n\n"
            f"The examiner asks:\n{sanitize_speaker_content(host_framing)}\n\n"
            "Answer each question directly."
        )
    else:
        opening_system = DISCUSS_PANELIST_SYSTEM.format(
            name="a panelist", other="the host", word_limit=150,
        ) + _persona_suffix() + _domain_suffix()
        opening_user = (
            f"Topic: {question}\n\n"
            f"The host opened with:\n{sanitize_speaker_content(host_framing)}\n\n"
            "Give your opening take."
        )

    opening_messages = [
        {"role": "system", "content": opening_system},
        {"role": "user", "content": opening_user},
    ]

    if verbose:
        print(f"(querying {len(panelists)} panelists in parallel...)")

    opening_results = asyncio.run(run_parallel(
        panelists, opening_messages, api_key, google_api_key,
        max_tokens=500, cost_accumulator=cost_accumulator,
        verbose=verbose,
    ))

    if is_socratic:
        transcript_parts.append("## Answers")

    for name, model_name, response in opening_results:
        transcript_parts.append(f"### {name}\n{response}")
        conversation_history.append((name, response))

    if verbose:
        print()

    # === Phase 2: DISCUSSION / PROBING ===
    round_label = "Probing Round" if is_socratic else "Round"
    history_label = "Examination" if is_socratic else "Discussion"
    round_num = 0
    try:
        while rounds == 0 or round_num < rounds:
            round_num += 1
            if verbose:
                print(f"## {round_label} {round_num}\n")

            transcript_parts.append(f"## {round_label} {round_num}")

            # Host steering / probing
            if is_socratic:
                steer_system = SOCRATIC_HOST_PROBE + _persona_suffix()
            else:
                steer_system = DISCUSS_HOST_STEER + _persona_suffix()
            steer_messages = [
                {"role": "system", "content": steer_system},
                {"role": "user", "content": f"Topic: {question}\n\n{history_label} so far:\n\n{_build_history_text()}"},
            ]

            if verbose:
                print(f"### {host_name}")

            host_steer = query_model(
                api_key, host_model, steer_messages,
                max_tokens=300, stream=verbose, cost_accumulator=cost_accumulator,
            )

            if verbose:
                print()

            transcript_parts.append(f"### {host_name}\n{host_steer}")
            conversation_history.append((host_name, host_steer))

            # Panelists respond sequentially (see full history)
            for name, model, fallback in panelists:
                if is_socratic:
                    panelist_system = SOCRATIC_PANELIST_SYSTEM.format(
                        name=name, word_limit=150,
                    ) + _persona_suffix() + _domain_suffix()
                else:
                    panelist_system = DISCUSS_PANELIST_SYSTEM.format(
                        name=name, other="the others", word_limit=150,
                    ) + _persona_suffix() + _domain_suffix()

                follow_up_cue = "The examiner just asked a follow-up. Answer directly." if is_socratic else "The host just asked a follow-up. Give your response."
                panelist_messages = [
                    {"role": "system", "content": panelist_system},
                    {"role": "user", "content": (
                        f"Topic: {question}\n\n"
                        f"{history_label} so far:\n\n{_build_history_text()}\n\n"
                        f"{follow_up_cue}"
                    )},
                ]

                if verbose:
                    print(f"### {name}")

                response = query_model(
                    api_key, model, panelist_messages,
                    stream=verbose, cost_accumulator=cost_accumulator,
                )

                # Handle fallback
                used_fallback = False
                if response.startswith("[") and fallback:
                    fallback_provider, fallback_model = fallback
                    if fallback_provider == "google" and google_api_key:
                        if verbose:
                            print(f"(OpenRouter failed, trying AI Studio fallback: {fallback_model}...)", flush=True)
                        response = query_google_ai_studio(google_api_key, fallback_model, panelist_messages)
                        used_fallback = True

                if verbose and used_fallback:
                    print(response)

                if verbose:
                    print()

                transcript_parts.append(f"### {name}\n{response}")
                conversation_history.append((name, response))

            # Show running stats in unlimited mode
            if verbose and rounds == 0:
                elapsed = time.time() - start_time
                cost_so_far = round(sum(cost_accumulator), 4) if cost_accumulator else 0.0
                print(f"  (round {round_num} done — {elapsed:.0f}s, ~${cost_so_far:.2f} — Ctrl+C to wrap up)\n")

    except KeyboardInterrupt:
        if verbose:
            print(f"\n\n(interrupted after {round_num} rounds, wrapping up...)\n")

    # === Phase 3: CLOSING ===
    if verbose:
        print("## Closing\n")

    transcript_parts.append("## Closing")

    # Panelist final takes (parallel)
    closing_prompt = SOCRATIC_PANELIST_CLOSING if is_socratic else DISCUSS_PANELIST_CLOSING
    closing_messages = [
        {"role": "system", "content": closing_prompt},
        {"role": "user", "content": f"Topic: {question}\n\nFull {history_label.lower()}:\n\n{_build_history_text()}"},
    ]

    if verbose:
        print(f"(querying {len(panelists)} panelists in parallel...)")

    closing_results = asyncio.run(run_parallel(
        panelists, closing_messages, api_key, google_api_key,
        max_tokens=300, cost_accumulator=cost_accumulator,
        verbose=verbose,
    ))

    for name, model_name, response in closing_results:
        transcript_parts.append(f"### {name}\n{response}")
        conversation_history.append((name, response))

    if verbose:
        print()

    # Host closing / synthesis
    synthesis_label = "Synthesis" if is_socratic else None
    closing_host_prompt = SOCRATIC_HOST_SYNTHESIS if is_socratic else DISCUSS_HOST_CLOSING
    closing_host_messages = [
        {"role": "system", "content": closing_host_prompt},
        {"role": "user", "content": f"Topic: {question}\n\nFull {history_label.lower()}:\n\n{_build_history_text()}"},
    ]

    if synthesis_label and verbose:
        print(f"## {synthesis_label}\n")
        transcript_parts.append(f"## {synthesis_label}")

    if verbose:
        print(f"### {host_name}")

    host_closing = query_model(
        api_key, host_model, closing_host_messages,
        max_tokens=400 if is_socratic else 300, stream=verbose, cost_accumulator=cost_accumulator,
    )

    if verbose:
        print()

    transcript_parts.append(f"### {host_name}\n{host_closing}")

    duration = time.time() - start_time
    total_cost = round(sum(cost_accumulator), 4) if cost_accumulator else 0.0

    if verbose:
        print(f"({duration:.1f}s, ~${total_cost:.2f})")

    return SessionResult(
        transcript="\n\n".join(transcript_parts),
        cost=total_cost,
        duration=duration,
    )
