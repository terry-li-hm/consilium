"""Oxford debate mode: structured for/against with rebuttals and verdict."""

import asyncio
import random
import time

import httpx

from .models import (
    JUDGE_MODEL,
    SessionResult,
    query_model,
    query_model_async,
    sanitize_speaker_content,
)
from .prompts import (
    DOMAIN_CONTEXTS,
    OXFORD_MOTION_TRANSFORM,
    OXFORD_CONSTRUCTIVE_SYSTEM,
    OXFORD_REBUTTAL_SYSTEM,
    OXFORD_CLOSING_SYSTEM,
    OXFORD_JUDGE_PRIOR,
    OXFORD_JUDGE_VERDICT,
)


def _assign_sides(
    debaters: list[tuple[str, str, tuple[str, str] | None]],
) -> tuple[tuple[str, str, tuple[str, str] | None], tuple[str, str, tuple[str, str] | None]]:
    """Random side assignment for Oxford debate."""
    sides = list(debaters)
    random.shuffle(sides)
    return sides[0], sides[1]


def run_oxford(
    question: str,
    debaters: list[tuple[str, str, tuple[str, str] | None]],
    api_key: str,
    google_api_key: str | None = None,
    verbose: bool = True,
    persona: str | None = None,
    domain: str | None = None,
    motion_override: str | None = None,
) -> SessionResult:
    """Run Oxford debate: structured for/against with rebuttals and verdict. Returns SessionResult."""
    start_time = time.time()
    cost_accumulator: list[float] = []
    judge_model = JUDGE_MODEL
    judge_name = "Judge (Claude)"

    domain_context = DOMAIN_CONTEXTS.get(domain, "") if domain else ""
    transcript_parts = []

    def _persona_suffix() -> str:
        if persona:
            return f"\n\nContext about the person asking: {persona}"
        return ""

    if verbose:
        print("=" * 60)
        print("OXFORD DEBATE")
        print("=" * 60)
        print()

    # === Phase 1: MOTION ===
    if motion_override:
        motion = motion_override
    else:
        motion_messages = [
            {"role": "system", "content": OXFORD_MOTION_TRANSFORM.format(question=question)},
            {"role": "user", "content": question},
        ]

        if verbose:
            print("## Motion\n")

        motion = query_model(
            api_key, judge_model, motion_messages,
            max_tokens=100, stream=verbose, cost_accumulator=cost_accumulator,
        )
        motion = motion.strip().strip('"')

        if verbose:
            print()

    transcript_parts.append(f"## Motion\n\n{motion}")

    # Assign sides randomly
    prop, opp = _assign_sides(debaters)
    prop_name, prop_model, prop_fallback = prop
    opp_name, opp_model, opp_fallback = opp

    if verbose:
        print(f"Proposition (FOR): {prop_name}")
        print(f"Opposition (AGAINST): {opp_name}")
        print()

    transcript_parts.append(f"**Proposition:** {prop_name} | **Opposition:** {opp_name}")

    # === Phase 2: PRIOR ===
    if verbose:
        print("## Prior\n")
        print(f"### {judge_name}")

    prior_messages = [
        {"role": "system", "content": OXFORD_JUDGE_PRIOR.format(motion=motion) + _persona_suffix()},
        {"role": "user", "content": f"Motion: {motion}"},
    ]

    prior_response = query_model(
        api_key, judge_model, prior_messages,
        max_tokens=200, stream=verbose, cost_accumulator=cost_accumulator,
    )

    if verbose:
        print()

    transcript_parts.append(f"## Prior\n\n### {judge_name}\n{prior_response}")

    # === Phase 3: CONSTRUCTIVE (parallel) ===
    if verbose:
        print("## Constructive Speeches\n")
        print("(both sides arguing in parallel...)", flush=True)

    transcript_parts.append("## Constructive Speeches")

    prop_system = OXFORD_CONSTRUCTIVE_SYSTEM.format(
        name=prop_name, side="FOR", motion=motion,
    ) + _persona_suffix()
    opp_system = OXFORD_CONSTRUCTIVE_SYSTEM.format(
        name=opp_name, side="AGAINST", motion=motion,
    ) + _persona_suffix()

    async def _run_constructives():
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=180.0,
        ) as client:
            async def _prop():
                result = await query_model_async(
                    client, prop_model,
                    [{"role": "system", "content": prop_system},
                     {"role": "user", "content": f"Argue FOR the motion: {motion}"}],
                    prop_name, fallback=prop_fallback, google_api_key=google_api_key,
                    max_tokens=800, cost_accumulator=cost_accumulator,
                )
                if verbose:
                    _, _, response = result
                    if not response.startswith("["):
                        print(f"\n### {prop_name} (Proposition)")
                        print(response)
                        print(flush=True)
                return result

            async def _opp():
                result = await query_model_async(
                    client, opp_model,
                    [{"role": "system", "content": opp_system},
                     {"role": "user", "content": f"Argue AGAINST the motion: {motion}"}],
                    opp_name, fallback=opp_fallback, google_api_key=google_api_key,
                    max_tokens=800, cost_accumulator=cost_accumulator,
                )
                if verbose:
                    _, _, response = result
                    if not response.startswith("["):
                        print(f"\n### {opp_name} (Opposition)")
                        print(response)
                        print(flush=True)
                return result

            return await asyncio.gather(_prop(), _opp(), return_exceptions=True)

    results = list(asyncio.run(_run_constructives()))
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            side_name = [prop_name, opp_name][i]
            results[i] = (side_name, "unknown", f"[Error: {r}]")
    (_, _, prop_constructive), (_, _, opp_constructive) = results

    if verbose:
        print()

    transcript_parts.append(f"### {prop_name} (Proposition)\n{prop_constructive}")
    transcript_parts.append(f"### {opp_name} (Opposition)\n{opp_constructive}")

    # === Phase 4: REBUTTAL (parallel — each rebuts the other's constructive) ===
    if verbose:
        print("## Rebuttals\n")
        print("(both sides rebutting in parallel...)", flush=True)

    transcript_parts.append("## Rebuttals")

    prop_rebuttal_system = OXFORD_REBUTTAL_SYSTEM.format(
        name=prop_name, side="FOR", motion=motion,
        opponent_argument=sanitize_speaker_content(opp_constructive),
    ) + _persona_suffix()

    opp_rebuttal_system = OXFORD_REBUTTAL_SYSTEM.format(
        name=opp_name, side="AGAINST", motion=motion,
        opponent_argument=sanitize_speaker_content(prop_constructive),
    ) + _persona_suffix()

    async def _run_rebuttals():
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=180.0,
        ) as client:
            async def _prop():
                result = await query_model_async(
                    client, prop_model,
                    [{"role": "system", "content": prop_rebuttal_system},
                     {"role": "user", "content": "Rebut the opposition's arguments."}],
                    prop_name, fallback=prop_fallback, google_api_key=google_api_key,
                    max_tokens=600, cost_accumulator=cost_accumulator,
                )
                if verbose:
                    _, _, response = result
                    if not response.startswith("["):
                        print(f"\n### {prop_name} (Proposition rebuttal)")
                        print(response)
                        print(flush=True)
                return result

            async def _opp():
                result = await query_model_async(
                    client, opp_model,
                    [{"role": "system", "content": opp_rebuttal_system},
                     {"role": "user", "content": "Rebut the proposition's arguments."}],
                    opp_name, fallback=opp_fallback, google_api_key=google_api_key,
                    max_tokens=600, cost_accumulator=cost_accumulator,
                )
                if verbose:
                    _, _, response = result
                    if not response.startswith("["):
                        print(f"\n### {opp_name} (Opposition rebuttal)")
                        print(response)
                        print(flush=True)
                return result

            return await asyncio.gather(_prop(), _opp(), return_exceptions=True)

    rebuttal_results = list(asyncio.run(_run_rebuttals()))
    for i, r in enumerate(rebuttal_results):
        if isinstance(r, Exception):
            side_name = [prop_name, opp_name][i]
            rebuttal_results[i] = (side_name, "unknown", f"[Error: {r}]")
    (_, _, prop_rebuttal), (_, _, opp_rebuttal) = rebuttal_results

    if verbose:
        print()

    transcript_parts.append(f"### {prop_name} (Proposition rebuttal)\n{prop_rebuttal}")
    transcript_parts.append(f"### {opp_name} (Opposition rebuttal)\n{opp_rebuttal}")

    # === Phase 5: CLOSING (parallel) ===
    if verbose:
        print("## Closing Statements\n")
        print("(both sides closing in parallel...)", flush=True)

    transcript_parts.append("## Closing Statements")

    prop_closing_system = OXFORD_CLOSING_SYSTEM.format(
        name=prop_name, side="FOR", motion=motion,
    )
    opp_closing_system = OXFORD_CLOSING_SYSTEM.format(
        name=opp_name, side="AGAINST", motion=motion,
    )

    async def _run_closings():
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=180.0,
        ) as client:
            async def _prop():
                result = await query_model_async(
                    client, prop_model,
                    [{"role": "system", "content": prop_closing_system},
                     {"role": "user", "content": "Give your closing statement."}],
                    prop_name, fallback=prop_fallback, google_api_key=google_api_key,
                    max_tokens=400, cost_accumulator=cost_accumulator,
                )
                if verbose:
                    _, _, response = result
                    if not response.startswith("["):
                        print(f"\n### {prop_name} (Proposition closing)")
                        print(response)
                        print(flush=True)
                return result

            async def _opp():
                result = await query_model_async(
                    client, opp_model,
                    [{"role": "system", "content": opp_closing_system},
                     {"role": "user", "content": "Give your closing statement."}],
                    opp_name, fallback=opp_fallback, google_api_key=google_api_key,
                    max_tokens=400, cost_accumulator=cost_accumulator,
                )
                if verbose:
                    _, _, response = result
                    if not response.startswith("["):
                        print(f"\n### {opp_name} (Opposition closing)")
                        print(response)
                        print(flush=True)
                return result

            return await asyncio.gather(_prop(), _opp(), return_exceptions=True)

    results = list(asyncio.run(_run_closings()))
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            side_name = [prop_name, opp_name][i]
            results[i] = (side_name, "unknown", f"[Error: {r}]")
    (_, _, prop_closing), (_, _, opp_closing) = results

    if verbose:
        print()

    transcript_parts.append(f"### {prop_name} (Proposition closing)\n{prop_closing}")
    transcript_parts.append(f"### {opp_name} (Opposition closing)\n{opp_closing}")

    # === Phase 6: VERDICT ===
    if verbose:
        print("## Verdict\n")
        print(f"### {judge_name}")

    transcript_parts.append("## Verdict")

    debate_transcript = (
        f"### Proposition ({prop_name}) — Constructive\n{sanitize_speaker_content(prop_constructive)}\n\n"
        f"### Opposition ({opp_name}) — Constructive\n{sanitize_speaker_content(opp_constructive)}\n\n"
        f"### Proposition ({prop_name}) — Rebuttal\n{sanitize_speaker_content(prop_rebuttal)}\n\n"
        f"### Opposition ({opp_name}) — Rebuttal\n{sanitize_speaker_content(opp_rebuttal)}\n\n"
        f"### Proposition ({prop_name}) — Closing\n{sanitize_speaker_content(prop_closing)}\n\n"
        f"### Opposition ({opp_name}) — Closing\n{sanitize_speaker_content(opp_closing)}"
    )

    verdict_system = OXFORD_JUDGE_VERDICT.format(
        motion=motion,
        proposition_name=prop_name,
        opposition_name=opp_name,
        debate_transcript=debate_transcript,
    ) + _persona_suffix()

    verdict = query_model(
        api_key, judge_model, [
            {"role": "system", "content": verdict_system},
            {"role": "user", "content": f"Judge this debate on: {motion}\n\nYour prior assessment:\n{sanitize_speaker_content(prior_response)}"},
        ],
        max_tokens=1000, stream=verbose, cost_accumulator=cost_accumulator,
    )

    if verbose:
        print()

    transcript_parts.append(f"### {judge_name}\n{verdict}")

    duration = time.time() - start_time
    total_cost = round(sum(cost_accumulator), 4) if cost_accumulator else 0.0

    if verbose:
        print(f"({duration:.1f}s, ~${total_cost:.2f})")

    return SessionResult(
        transcript="\n\n".join(transcript_parts),
        cost=total_cost,
        duration=duration,
    )
