"""Full council deliberation mode: blind phase, debate rounds, judge synthesis, CollabEval."""

import asyncio
import json
import re
import time
import yaml

from .models import (
    COUNCIL,
    JUDGE_MODEL,
    CRITIQUE_MODEL,
    SessionResult,
    parse_confidence,
    query_model,
    query_google_ai_studio,
    run_parallel,
    sanitize_speaker_content,
    detect_consensus,
    extract_structured_summary,
)
from .prompts import (
    DOMAIN_CONTEXTS,
    COUNCIL_BLIND_SYSTEM,
    COUNCIL_FIRST_SPEAKER_WITH_BLIND,
    COUNCIL_FIRST_SPEAKER_SYSTEM,
    COUNCIL_DEBATE_SYSTEM,
    COUNCIL_CHALLENGER_ADDITION,
    COUNCIL_SOCIAL_CONSTRAINT,
)


def decompose_question(
    question: str,
    api_key: str,
    verbose: bool = True,
    cost_accumulator: list[float] | None = None,
) -> list[str]:
    """Break a complex question into 2-3 focused sub-questions."""
    judge_model_name = JUDGE_MODEL.split("/")[-1]
    if verbose:
        print(f"### Question Decomposition ({judge_model_name})")

    messages = [
        {
            "role": "system",
            "content": """Decompose the user's complex question into 2-3 focused, non-overlapping sub-questions.

Output STRICT JSON only: an array of strings.
No markdown, no prose, no explanation.
Each sub-question should be actionable for independent analysis.""",
        },
        {"role": "user", "content": f"Question:\n{question}"},
    ]

    response = query_model(
        api_key,
        JUDGE_MODEL,
        messages,
        max_tokens=300,
        stream=verbose,
        cost_accumulator=cost_accumulator,
    )

    if verbose:
        print()

    match = re.search(r"\[[\s\S]*\]", response)
    candidate = match.group(0) if match else response
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            sub_questions = [str(item).strip() for item in parsed if str(item).strip()]
            if len(sub_questions) >= 2:
                return sub_questions[:3]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    lines = [
        re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        for line in response.splitlines()
    ]
    fallback = [line for line in lines if line]
    if len(fallback) >= 2:
        return fallback[:3]

    return [question]


async def run_blind_phase_parallel(
    question: str,
    council_config: list[tuple[str, str, tuple[str, str] | None]],
    api_key: str,
    google_api_key: str | None = None,
    verbose: bool = True,
    persona: str | None = None,
    domain_context: str = "",
    sub_questions: list[str] | None = None,
    cost_accumulator: list[float] | None = None,
) -> list[tuple[str, str, str]]:
    """Parallel blind first-pass: all models stake claims simultaneously."""
    blind_system = COUNCIL_BLIND_SYSTEM

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

    user_content = f"Question:\n\n{question}"
    if sub_questions and len(sub_questions) > 1:
        numbered = "\n".join(f"{i}. {sq}" for i, sq in enumerate(sub_questions, 1))
        user_content += f"\n\nSub-questions to address:\n{numbered}"

    messages = [
        {"role": "system", "content": blind_system},
        {"role": "user", "content": user_content},
    ]

    if verbose:
        print("(querying all models in parallel...)")

    blind_claims = await run_parallel(
        council_config, messages, api_key, google_api_key,
        cost_accumulator=cost_accumulator,
        verbose=verbose,
    )

    if verbose:
        print()

    return blind_claims


def run_followup_discussion(
    question: str,
    topic: str,
    council_config: list[tuple[str, str, tuple[str, str] | None]],
    api_key: str,
    domain_context: str = "",
    social_mode: bool = False,
    persona: str | None = None,
    verbose: bool = True,
    cost_accumulator: list[float] | None = None,
) -> str:
    """Run a focused followup discussion on a specific topic with 2 models. Returns the followup transcript."""
    # Use first two council models (GPT and Gemini) for focused followup
    followup_models = council_config[:2]

    followup_transcript_parts = []

    if verbose:
        print()
        print("=" * 60)
        print(f"FOLLOWUP: {topic}")
        print("=" * 60)
        print()

    social_constraint = COUNCIL_SOCIAL_CONSTRAINT if social_mode else ""

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

        response = query_model(api_key, model, messages, stream=verbose, cost_accumulator=cost_accumulator)

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
    rounds: int = 1,
    verbose: bool = True,
    anonymous: bool = True,
    blind: bool = True,
    context: str | None = None,
    social_mode: bool = False,
    persona: str | None = None,
    domain: str | None = None,
    challenger_idx: int | None = None,
    format: str = "prose",
    collabeval: bool = True,
    judge: bool = True,
    sub_questions: list[str] | None = None,
) -> SessionResult:
    """Run the council deliberation. Returns SessionResult."""

    start_time = time.time()
    cost_accumulator: list[float] = []

    domain_context = DOMAIN_CONTEXTS.get(domain, "") if domain else ""
    council_names = [name for name, _, _ in council_config]
    blind_claims = []
    failed_models = []

    social_constraint = COUNCIL_SOCIAL_CONSTRAINT if social_mode else ""

    if blind:
        blind_claims = asyncio.run(run_blind_phase_parallel(
            question,
            council_config,
            api_key,
            google_api_key,
            verbose,
            persona,
            domain_context,
            sub_questions,
            cost_accumulator=cost_accumulator,
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
    confidences: dict[str, list[int]] = {}

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

    for round_num in range(rounds):
        current_round = round_num + 1
        round_speakers = []
        for idx, (name, model, fallback) in enumerate(council_config):
            dname = display_names[name]

            if idx == 0 and round_num == 0:
                if blind_claims:
                    system_prompt = COUNCIL_FIRST_SPEAKER_WITH_BLIND.format(name=dname, round_num=round_num + 1)
                else:
                    system_prompt = COUNCIL_FIRST_SPEAKER_SYSTEM.format(name=dname, round_num=round_num + 1)
            else:
                if round_speakers:
                    previous = ", ".join(round_speakers)
                else:
                    previous = ", ".join([display_names[n] for n, _, _ in council_config])
                system_prompt = COUNCIL_DEBATE_SYSTEM.format(
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

            if persona:
                system_prompt += f"""

IMPORTANT CONTEXT about the person asking:
{persona}

Factor this into your advice — don't just give strategically optimal answers, consider what fits THIS person."""

            # Calculate rotating challenger for this round
            if challenger_idx is not None:
                current_challenger = (challenger_idx + round_num) % len(council_config)
            else:
                current_challenger = round_num % len(council_config)

            if idx == current_challenger:
                system_prompt += COUNCIL_CHALLENGER_ADDITION

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

            response = query_model(api_key, model, messages, stream=verbose, cost_accumulator=cost_accumulator)

            used_fallback = False
            if response.startswith("[") and fallback:
                fallback_provider, fallback_model = fallback

                if fallback_provider == "google" and google_api_key:
                    if verbose:
                        print(f"(OpenRouter failed, trying AI Studio fallback: {fallback_model}...)", flush=True)
                    response = query_google_ai_studio(google_api_key, fallback_model, messages)
                    used_fallback = True
                    model_name = fallback_model

            if verbose and used_fallback:
                print(response)

            if response.startswith("["):
                failed_models.append(f"{model_name}: {response}")

            conversation.append((name, response))
            round_speakers.append(dname)

            conf = parse_confidence(response)
            if conf is not None:
                confidences.setdefault(name, []).append(conf)

            if verbose:
                print()

            output_parts.append(f"### {model_name}{challenger_indicator}\n{response}")

        converged, reason = detect_consensus(conversation, council_config, current_challenger)
        if converged:
            if verbose:
                print(f">>> CONSENSUS DETECTED ({reason}) - proceeding to judge\n")
            break

    # Confidence drift display
    if confidences and verbose:
        drift_parts = []
        for name, scores in confidences.items():
            model_name = next(m.split("/")[-1] for n, m, _ in council_config if n == name)
            if len(scores) >= 2:
                drift_parts.append(f"{model_name} {scores[0]}\u2192{scores[-1]}")
            elif scores:
                drift_parts.append(f"{model_name} {scores[0]}/10")
        if drift_parts:
            print(f"  Confidence: {', '.join(drift_parts)}")
            print()

    if confidences:
        drift_parts = []
        for name, scores in confidences.items():
            model_name = next(m.split("/")[-1] for n, m, _ in council_config if n == name)
            if len(scores) >= 2:
                drift_parts.append(f"{model_name} {scores[0]}\u2192{scores[-1]}")
            elif scores:
                drift_parts.append(f"{model_name} {scores[0]}/10")
        if drift_parts:
            output_parts.append(f"Confidence drift: {', '.join(drift_parts)}")

    deliberation_text = "\n\n".join(
        f"**{display_names[speaker]}**: {sanitize_speaker_content(text)}" for speaker, text in conversation
    )

    if not judge:
        # External judge mode: emit debate transcript + metadata, skip synthesis
        if verbose:
            print("(judge=False — skipping synthesis for external judge)")
            print()

        output_parts.append(f"## Council Deliberation Transcript\n\n{deliberation_text}")

        duration = time.time() - start_time
        total_cost = round(sum(cost_accumulator), 4) if cost_accumulator else 0.0
        meta = {
            "judge": "external",
            "question": question,
            "models_used": [name for name, _, _ in council_config],
            "rounds": current_round if rounds > 0 else 1,
            "duration_seconds": round(duration, 1),
            "estimated_cost_usd": total_cost,
        }
        if context:
            meta["context"] = context
        if domain:
            meta["domain"] = domain
        if persona:
            meta["persona"] = persona
        if social_mode:
            meta["social_mode"] = True

        output_parts.append('\n\n---\n\n' + json.dumps(meta, indent=2, ensure_ascii=False))

    else:
        # Internal judge synthesis
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

SYNTHESIS METHOD — Analysis of Competing Hypotheses:
Rather than seeking the consensus view, first list ALL plausible conclusions from the deliberation (typically 2-4). For each piece of evidence or argument raised by the council, evaluate how well it supports or undermines EACH hypothesis. Eliminate conclusions that are inconsistent with the strongest evidence. The surviving hypothesis is your recommendation.

CONVERGENCE SIGNAL:
When independent agents with different models and training reached the SAME conclusion in the blind phase, treat this as a multiplicatively strong signal — independent agreement from different priors is more reliable than the same conclusion repeated. Push your confidence further toward certainty than a simple average.

SYCOPHANCY CHECK:
Flag any agent that changed position during debate WITHOUT citing a specific new argument or piece of evidence. Position changes labeled POSITION CHANGE with clear reasoning are healthy. Unlabeled shifts toward consensus are sycophancy — discount these.

After applying this method, structure your response as:

## Competing Hypotheses
[List 2-4 plausible conclusions. For each, note which council arguments support/undermine it]

## Points of Agreement
[What the council agrees on — and whether that consensus should be trusted given the sycophancy check]

## Points of Disagreement
[Where views genuinely diverged and why — these often point to the crux]

## Judge's Own Take
[Your independent perspective. What did the council miss or underweight?]

## Synthesis
[The integrated perspective, combining council views with your own ACH analysis]

## Recommendation
[Your final recommendation]
{social_judge_section}
Be balanced and fair. Acknowledge minority views. But don't be afraid to have your own opinion — you're the judge, not just a summarizer. Critically evaluate each position — don't replicate or parrot the council's language.{" For social contexts, prioritize natural/human output over strategic optimization." if social_mode else ""}

CRITICAL — PRESCRIPTION DISCIPLINE:
Your job is to FILTER, not aggregate. The council will generate many suggestions. Most are interesting but not necessary.

Rules:
- **Do Now** — MAX 3 items. For each, first argue AGAINST including it (cost, risk, unnecessary). Only include it if it survives your own counter-argument. If you can't argue against it, it's probably essential.
- **Consider Later** — Items that are interesting but not worth doing now
- **Skip** — Explicitly list council suggestions you're DROPPING and why

The council's gravitational pull is toward "add more." Your gravitational pull must be toward "do less." A recommendation with 6 action items is not a recommendation — it's a wish list.

Don't recommend building infrastructure for problems that don't exist yet.

If this council revealed a reusable pattern about model reliability, user blind spots, or question framing, propose a candidate Static Note update at the end of your synthesis."""

        judge_messages = [
            {"role": "system", "content": judge_system},
            {"role": "user", "content": f"Question:\n{question}\n\n---\n\nCouncil Deliberation:\n\n{deliberation_text}"},
        ]

        judge_model_name = JUDGE_MODEL.split("/")[-1]

        if verbose:
            print(f"### Judge ({judge_model_name})")

        judge_response = query_model(api_key, JUDGE_MODEL, judge_messages, max_tokens=1200, stream=verbose, cost_accumulator=cost_accumulator)

        if verbose:
            print()

        output_parts.append(f"### Judge ({judge_model_name})\n{judge_response}")

        # CollabEval Phase 2-3: Critique + Revision (skipped when collabeval=False)
        if collabeval:
            critique_model_name = CRITIQUE_MODEL.split("/")[-1]
            critique_system = f"""You are an independent critic reviewing a judge's synthesis of a multi-model council deliberation.

Your job is to find WEAKNESSES in the judge's synthesis — not to agree with it.

Look for:
1. Points the judge dismissed too quickly or weighted incorrectly
2. Minority views that deserved more consideration
3. Logical gaps or unsupported leaps in the recommendation
4. Practical concerns the judge missed
5. Whether the "Do Now" items are truly the right priorities

Be specific and concise. Name the exact weakness and why it matters.
If the synthesis is genuinely strong, say so briefly — but try hard to find something.{f" Consider the {domain} regulatory context." if domain else ""}"""

            critique_messages = [
                {"role": "system", "content": critique_system},
                {"role": "user", "content": f"Question:\n{question}\n\nJudge's Synthesis:\n\n{judge_response}"},
            ]

            if verbose:
                print(f"### Critique ({critique_model_name})")

            critique_response = query_model(api_key, CRITIQUE_MODEL, critique_messages, max_tokens=800, stream=verbose, cost_accumulator=cost_accumulator)

            if verbose:
                print()

            output_parts.append(f"### Critique ({critique_model_name})\n{critique_response}")

            # CollabEval Phase 3: Judge revision
            if verbose:
                print(f"### Final Synthesis ({judge_model_name})")

            revision_messages = judge_messages + [
                {"role": "assistant", "content": judge_response},
                {"role": "user", "content": f"An independent critic has reviewed your synthesis:\n\n{critique_response}\n\nRevise your synthesis considering this critique. Keep what's right, fix what's wrong. If the critique raises valid points, integrate them. If not, explain briefly why you stand by your original position. Output your FINAL revised synthesis in the same format."},
            ]

            final_response = query_model(api_key, JUDGE_MODEL, revision_messages, max_tokens=1200, stream=verbose, cost_accumulator=cost_accumulator)

            if verbose:
                print()

            output_parts.append(f"### Final Synthesis ({judge_model_name})\n{final_response}")

            # Use the final revised synthesis for structured extraction
            judge_response = final_response

        if format != 'prose':
            structured = extract_structured_summary(
                judge_response=judge_response,
                question=question,
                models_used=[name for name, _, _ in council_config],
                rounds=current_round if rounds > 0 else 1,
                duration=time.time() - start_time,
                cost=0.0,
                api_key=api_key,
                cost_accumulator=cost_accumulator,
            )
            total_cost = round(sum(cost_accumulator), 4) if cost_accumulator else 0.0
            structured["meta"]["estimated_cost_usd"] = total_cost

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
        transcript = final_output
    else:
        transcript = "\n\n".join(output_parts)

    # Print failure summary
    if failed_models and verbose:
        print()
        print("=" * 60)
        print("MODEL FAILURES")
        print("=" * 60)
        for failure in failed_models:
            print(f"  - {failure}")
        working_count = len(council_config) - len(set(f.split(":")[0].split(" (")[0] for f in failed_models))
        print(f"\nCouncil ran with {working_count}/{len(council_config)} models")
        print("=" * 60)
        print()

    duration = time.time() - start_time
    total_cost = round(sum(cost_accumulator), 4) if cost_accumulator else 0.0

    if verbose:
        print(f"({duration:.1f}s, ~${total_cost:.2f})")

    return SessionResult(transcript=transcript, cost=total_cost, duration=duration)
