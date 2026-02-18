"""Solo council mode: one model, structured deliberation with multiple perspectives."""

import asyncio
import time

import httpx

from .models import (
    JUDGE_MODEL,
    SessionResult,
    parse_confidence,
    query_model,
    query_model_async,
    sanitize_speaker_content,
)
from .prompts import (
    DOMAIN_CONTEXTS,
    ROLE_LIBRARY,
    SOLO_DEFAULT_ROLES,
    SOLO_BLIND_SYSTEM,
    SOLO_DEBATE_SYSTEM,
    SOLO_CHALLENGER_ADDITION,
    SOLO_JUDGE_SYSTEM,
)


def run_solo(
    question: str,
    api_key: str,
    verbose: bool = True,
    persona: str | None = None,
    domain: str | None = None,
    roles: list[str] | None = None,
) -> SessionResult:
    """Run solo council: structured deliberation with one model playing multiple perspectives. Returns SessionResult."""
    start_time = time.time()
    cost_accumulator: list[float] = []
    model = JUDGE_MODEL

    domain_context = DOMAIN_CONTEXTS.get(domain, "") if domain else ""

    # Resolve roles → perspectives using the tuned library, with generic fallback
    role_names = roles if roles and len(roles) >= 2 else list(SOLO_DEFAULT_ROLES)
    perspectives = []
    for role in role_names:
        description = ROLE_LIBRARY.get(role)
        if not description:
            # Case-insensitive lookup
            for lib_name, lib_desc in ROLE_LIBRARY.items():
                if lib_name.lower() == role.lower():
                    description = lib_desc
                    role = lib_name  # Use canonical casing
                    break
        if not description:
            # Generic fallback for unknown roles
            description = f"You approach this as a {role}. Bring your professional lens, domain expertise, and the specific concerns someone in your role would have. Be specific and opinionated — generic advice is worthless."
        perspectives.append((role, description))

    # Second perspective is always the challenger
    challenger_name = perspectives[1][0]

    transcript_parts = []

    def _suffix() -> str:
        parts = []
        if persona:
            parts.append(f"\n\nContext about the person asking: {persona}")
        if domain_context:
            parts.append(f"\n\nDomain context: {domain_context}")
        return "".join(parts)

    if verbose:
        print("=" * 60)
        print("SOLO COUNCIL")
        print("=" * 60)
        print()

    # === Phase 1: BLIND (parallel — independent perspectives) ===
    if verbose:
        print("## Blind Phase\n")

    transcript_parts.append("## Blind Phase")

    perspective_configs = []
    perspective_messages = []
    for name, description in perspectives:
        system = SOLO_BLIND_SYSTEM.format(name=name, description=description) + _suffix()
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        perspective_configs.append((name, model, None))
        perspective_messages.append(msgs)

    async def _run_blind():
        indexed_results: list[tuple[int, tuple[str, str, str] | Exception]] = []

        async def _query_and_print(idx, name, client):
            try:
                result = await query_model_async(
                    client, model, perspective_messages[idx], name, None, None,
                    max_tokens=500,
                    cost_accumulator=cost_accumulator,
                )
            except Exception as e:
                indexed_results.append((idx, e))
                return
            if verbose:
                _, _, response = result
                if not response.startswith("["):
                    print(f"\n### {name}")
                    print(response)
                    print(flush=True)
            indexed_results.append((idx, result))

        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=180.0,
        ) as client:
            tasks = [
                _query_and_print(i, name, client)
                for i, (name, _, _) in enumerate(perspective_configs)
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

        indexed_results.sort(key=lambda x: x[0])
        return indexed_results

    if verbose:
        print(f"(generating {len(perspectives)} perspectives in parallel...)")

    blind_results = asyncio.run(_run_blind())

    blind_claims: dict[str, str] = {}
    for idx, result in blind_results:
        name = perspectives[idx][0]
        if isinstance(result, Exception):
            response = f"[Error: {result}]"
        else:
            _, _, response = result

        blind_claims[name] = response
        transcript_parts.append(f"### {name}\n{response}")

    if verbose:
        print()

    # === Phase 2: DEBATE (sequential — each sees previous responses) ===
    if verbose:
        print("## Debate\n")

    transcript_parts.append("## Debate")

    blind_text = "\n\n".join(
        f"**{name}**: {sanitize_speaker_content(text)}"
        for name, text in blind_claims.items()
    )

    debate_responses: dict[str, str] = {}
    confidences: dict[str, list[int]] = {}

    for name, description in perspectives:
        system = SOLO_DEBATE_SYSTEM.format(name=name, description=description) + _suffix()

        # Second perspective is always the challenger
        is_challenger = (name == challenger_name)
        if is_challenger:
            system += SOLO_CHALLENGER_ADDITION

        debate_context = f"Blind phase perspectives:\n\n{blind_text}"
        if debate_responses:
            debate_text = "\n\n".join(
                f"**{n}**: {sanitize_speaker_content(t)}"
                for n, t in debate_responses.items()
            )
            debate_context += f"\n\nDebate responses so far:\n\n{debate_text}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question:\n\n{question}\n\n---\n\n{debate_context}"},
        ]

        challenger_tag = " (challenger)" if is_challenger else ""

        if verbose:
            print(f"### {name}{challenger_tag}")

        response = query_model(
            api_key, model, messages,
            stream=verbose, cost_accumulator=cost_accumulator,
        )

        if verbose:
            print()

        debate_responses[name] = response
        transcript_parts.append(f"### {name}{challenger_tag}\n{response}")

        conf = parse_confidence(response)
        if conf is not None:
            confidences.setdefault(name, []).append(conf)

    # Also parse blind phase confidence
    for name, text in blind_claims.items():
        conf = parse_confidence(text)
        if conf is not None:
            confidences.setdefault(name, []).insert(0, conf)

    # Confidence drift display
    if confidences and verbose:
        drift_parts = []
        for name, scores in confidences.items():
            if len(scores) >= 2:
                drift_parts.append(f"{name} {scores[0]}\u2192{scores[-1]}")
            elif scores:
                drift_parts.append(f"{name} {scores[0]}/10")
        if drift_parts:
            print(f"  Confidence: {', '.join(drift_parts)}")
            print()

    if confidences:
        drift_parts = []
        for name, scores in confidences.items():
            if len(scores) >= 2:
                drift_parts.append(f"{name} {scores[0]}\u2192{scores[-1]}")
            elif scores:
                drift_parts.append(f"{name} {scores[0]}/10")
        if drift_parts:
            transcript_parts.append(f"Confidence drift: {', '.join(drift_parts)}")

    # === Phase 3: SYNTHESIS ===
    if verbose:
        print("## Synthesis\n")

    transcript_parts.append("## Synthesis")

    full_deliberation = f"Blind phase:\n\n{blind_text}\n\nDebate:\n\n"
    full_deliberation += "\n\n".join(
        f"**{name}**: {sanitize_speaker_content(text)}"
        for name, text in debate_responses.items()
    )

    judge_system = SOLO_JUDGE_SYSTEM + _suffix()
    judge_messages = [
        {"role": "system", "content": judge_system},
        {"role": "user", "content": f"Question:\n{question}\n\n---\n\nFull deliberation:\n\n{full_deliberation}"},
    ]

    if verbose:
        print("### Judge (Claude)")

    judge_response = query_model(
        api_key, model, judge_messages,
        max_tokens=1200, stream=verbose, cost_accumulator=cost_accumulator,
    )

    if verbose:
        print()

    transcript_parts.append(f"### Judge (Claude)\n{judge_response}")

    duration = time.time() - start_time
    total_cost = round(sum(cost_accumulator), 4) if cost_accumulator else 0.0

    if verbose:
        print(f"({duration:.1f}s, ~${total_cost:.2f})")

    return SessionResult(
        transcript="\n\n".join(transcript_parts),
        cost=total_cost,
        duration=duration,
    )
