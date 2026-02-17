"""CLI entry point for consilium."""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_sessions_dir() -> Path:
    """Get the sessions directory, creating if needed."""
    sessions_dir = Path.home() / ".consilium" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def slugify(text: str, max_len: int = 40) -> str:
    """Convert text to a filename-safe slug."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    return text[:max_len].strip('-')

from .council import (
    COUNCIL,
    QUICK_MODELS,
    QUICK_MODELS_CHEAP,
    classify_difficulty,
    detect_social_context,
    run_council,
    run_quick,
    DOMAIN_CONTEXTS,
    run_followup_discussion,
)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Council - Multi-model deliberation for important decisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  consilium "Should we use microservices or monolith?"
  consilium "Career decision" --persona "builder who hates process work"
  consilium "Decision" --domain banking --followup --output counsel.md

Quick mode (parallel queries, no debate):
  consilium "What are the tradeoffs of SSR vs CSR?" --quick
  consilium "Summarize this error" --quick --cheap
  consilium "Compare Python and Rust" --quick --format json

Auto-route by difficulty (simple→quick, moderate→council, complex→council+critique):
  consilium "Should I take this job offer?" --auto
        """,
    )
    parser.add_argument("question", nargs="?", help="The question for the council to deliberate")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--output", "-o",
        help="Save transcript to file",
    )
    parser.add_argument(
        "--context", "-c",
        help="Context hint for the judge (e.g., 'architecture decision', 'ethics question')",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "yaml", "prose"],
        default="prose",
        help="Output format: json (machine-parseable), yaml (structured), prose (default)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Upload transcript to secret GitHub Gist and print URL",
    )
    parser.add_argument(
        "--persona", "-p",
        help="Context about the person asking (e.g., 'builder who hates process work')",
    )
    parser.add_argument(
        "--domain",
        help="Regulatory domain context (banking, healthcare, eu, fintech, bio)",
    )
    parser.add_argument(
        "--challenger",
        help="Which model should argue contrarian (gpt, gemini, grok, deepseek, glm). Default: gpt",
    )
    parser.add_argument(
        "--followup",
        action="store_true",
        help="Enable followup mode to drill into specific points after judge synthesis",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't auto-save transcript to ~/.consilium/sessions/",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: parallel queries, no debate/judge (like ask-llms)",
    )
    parser.add_argument(
        "--cheap",
        action="store_true",
        help="Use cheaper model tier (only valid with --quick)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-route based on question difficulty: simple→quick, moderate→council, complex→council+critique",
    )
    parser.add_argument(
        "--sessions",
        action="store_true",
        help="List recent sessions and exit",
    )
    args = parser.parse_args()

    # Handle --sessions flag
    if args.sessions:
        sessions_dir = get_sessions_dir()
        sessions = sorted(sessions_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not sessions:
            print("No sessions found.")
        else:
            print(f"Sessions in {sessions_dir}:\n")
            for s in sessions[:20]:  # Show last 20
                mtime = datetime.fromtimestamp(s.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                print(f"  {mtime}  {s.name}")
            if len(sessions) > 20:
                print(f"\n  ... and {len(sessions) - 20} more")
        sys.exit(0)

    # Require question for normal operation
    if not args.question:
        parser.error("the following arguments are required: question")

    # Validate --quick / --cheap / --auto flags
    if args.cheap and not args.quick:
        parser.error("--cheap requires --quick")
    if args.auto and args.quick:
        parser.error("--auto and --quick are mutually exclusive")

    if args.quick:
        debate_flags = []
        if args.challenger:
            debate_flags.append("--challenger")
        if args.followup:
            debate_flags.append("--followup")
        if debate_flags:
            parser.error(f"--quick is incompatible with: {', '.join(debate_flags)}")

    # Auto-detect social context
    social_mode = detect_social_context(args.question)
    if social_mode and not args.quiet:
        print("(Auto-detected social context — enabling social calibration)")
        print()

    # Validate and resolve domain
    domain_context = None
    if args.domain:
        if args.domain.lower() not in DOMAIN_CONTEXTS:
            print(f"Error: Unknown domain '{args.domain}'. Valid domains: {', '.join(DOMAIN_CONTEXTS.keys())}", file=sys.stderr)
            sys.exit(1)
        domain_context = args.domain.lower()

    # Resolve challenger model
    challenger_idx = None
    if args.challenger:
        challenger_lower = args.challenger.lower()
        model_name_map = {n.lower(): i for i, (n, _, _) in enumerate(COUNCIL)}
        if challenger_lower not in model_name_map:
            print(f"Error: Unknown model '{args.challenger}'. Valid models: {', '.join(n for n, _, _ in COUNCIL)}", file=sys.stderr)
            sys.exit(1)
        challenger_idx = model_name_map[challenger_lower]
    elif args.domain:
        challenger_idx = 0

    if not args.quiet and challenger_idx is not None:
        challenger_name = COUNCIL[challenger_idx][0]
        print(f"(Contrarian challenger: {challenger_name})")
        print()

    # Get API keys
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    google_api_key = os.environ.get("GOOGLE_API_KEY")

    # Auto mode: classify difficulty and route
    difficulty = None
    if args.auto:
        if not args.quiet:
            print("Classifying question difficulty...", flush=True)
        difficulty = classify_difficulty(args.question, api_key)
        if not args.quiet:
            print(f"Difficulty: {difficulty}")
            print()

        if difficulty == "simple":
            # Route to quick mode with expensive models
            args.quick = True
        # moderate/complex proceed to full council below
        # collabeval is set based on difficulty when calling run_council

    # Quick mode: parallel queries, no debate
    if args.quick:
        models = QUICK_MODELS_CHEAP if args.cheap else QUICK_MODELS
        tier = "cheap" if args.cheap else "expensive"

        if not args.quiet:
            print(f"Running quick council ({tier}, {len(models)} models)...")
            print()

        try:
            transcript = run_quick(
                question=args.question,
                models=models,
                api_key=api_key,
                verbose=not args.quiet,
                format=args.format,
            )

            # Save transcript to user-specified location
            if args.output:
                Path(args.output).write_text(transcript)
                if not args.quiet:
                    print(f"Transcript saved to: {args.output}")

            # Auto-save to sessions directory
            session_path = None
            if not args.no_save:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                slug = slugify(args.question)
                filename = f"{timestamp}-quick-{slug}.md"
                session_path = get_sessions_dir() / filename
                session_content = f"""# Quick Council Session

**Question:** {args.question}
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Mode:** quick, {tier}
**Models:** {', '.join(m.split('/')[-1] for _, m in models)}

---

{transcript}
"""
                session_path.write_text(session_content)
                if not args.quiet:
                    print(f"Session saved to: {session_path}")

            # Share via gist
            gist_url = None
            if args.share:
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(
                        mode='w', suffix='.md', prefix='council-quick-', delete=False
                    ) as f:
                        f.write(f"# Quick LLM Council\n\n")
                        f.write(f"**Question:** {args.question}\n\n")
                        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n")
                        f.write(transcript)
                        temp_path = f.name

                    result = subprocess.run(
                        ["gh", "gist", "create", temp_path, "--desc", f"Quick Council: {args.question[:50]}"],
                        capture_output=True, text=True
                    )
                    os.unlink(temp_path)

                    if result.returncode == 0:
                        gist_url = result.stdout.strip()
                        print(f"\nShared: {gist_url}")
                    else:
                        print(f"Gist creation failed: {result.stderr}", file=sys.stderr)
                except FileNotFoundError:
                    print("Error: 'gh' CLI not found. Install with: brew install gh", file=sys.stderr)

            # Log to history
            history_file = get_sessions_dir().parent / "history.jsonl"
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": args.question[:200],
                "mode": "quick",
                "tier": tier,
                "session": str(session_path) if session_path else None,
                "gist": gist_url,
                "models": [m.split("/")[-1] for _, m in models],
            }
            with open(history_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

        sys.exit(0)

    # Full council mode
    if not args.quiet:
        mode_parts = ["anonymous", "blind"]
        if social_mode:
            mode_parts.append("social")
        print(f"Running LLM Council ({', '.join(mode_parts)})...")
        if google_api_key:
            print("(Fallback enabled: Gemini→AI Studio)")
        print()

    try:
        if not args.quiet and args.persona:
            print(f"(Persona context: {args.persona})")
            print()

        # Show challenger
        if not args.quiet:
            active_challenger_idx = challenger_idx if challenger_idx is not None else 0
            active_challenger_name = COUNCIL[active_challenger_idx][0]
            print(f"(Challenger: {active_challenger_name})")
            print()

        # CollabEval: on by default, off for --auto moderate
        use_collabeval = not (args.auto and difficulty == "moderate")

        transcript, failed_models = run_council(
            question=args.question,
            council_config=COUNCIL,
            api_key=api_key,
            google_api_key=google_api_key,
            rounds=1,
            verbose=not args.quiet,
            anonymous=True,
            blind=True,
            context=args.context,
            social_mode=social_mode,
            practical_mode=True,
            persona=args.persona,
            domain=domain_context,
            challenger_idx=challenger_idx,
            format=args.format,
            collabeval=use_collabeval,
        )

        # Followup mode
        followup_transcript = ""
        if args.followup and not args.quiet:
            print("\n" + "=" * 60)
            print("Enter topic to explore further (or 'done'): ", end="", flush=True)
            topic = input().strip()

            if topic and topic.lower() != "done":
                domain_ctxt = DOMAIN_CONTEXTS.get(domain_context, "") if domain_context else ""
                followup_transcript = run_followup_discussion(
                    question=args.question,
                    topic=topic,
                    council_config=COUNCIL,
                    api_key=api_key,
                    domain_context=domain_ctxt,
                    social_mode=social_mode,
                    persona=args.persona,
                    verbose=not args.quiet,
                )
                transcript += "\n\n" + followup_transcript

        # Print failure summary
        if failed_models and not args.quiet:
            print()
            print("=" * 60)
            print("MODEL FAILURES")
            print("=" * 60)
            for failure in failed_models:
                print(f"  - {failure}")
            working_count = len(COUNCIL) - len(set(f.split(":")[0].split(" (")[0] for f in failed_models))
            print(f"\nCouncil ran with {working_count}/{len(COUNCIL)} models")
            print("=" * 60)
            print()

        # Save transcript to user-specified location
        if args.output:
            Path(args.output).write_text(transcript)
            if not args.quiet:
                print(f"Transcript saved to: {args.output}")

        # Auto-save to sessions directory
        session_path = None
        if not args.no_save:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            slug = slugify(args.question)
            filename = f"{timestamp}-{slug}.md"
            session_path = get_sessions_dir() / filename

            session_content = f"""# Council Session

**Question:** {args.question}
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Mode:** anonymous, blind{", social" if social_mode else ""}{f", auto ({difficulty})" if args.auto else ""}{", collabeval" if use_collabeval else ""}
{f"**Context:** {args.context}" if args.context else ""}
{f"**Persona:** {args.persona}" if args.persona else ""}

---

{transcript}
"""
            session_path.write_text(session_content)
            if not args.quiet:
                print(f"Session saved to: {session_path}")

        # Share via gist
        gist_url = None
        if args.share:
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.md', prefix='council-', delete=False
                ) as f:
                    f.write(f"# LLM Council Deliberation\n\n")
                    f.write(f"**Question:** {args.question}\n\n")
                    if args.context:
                        f.write(f"**Context:** {args.context}\n\n")
                    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n")
                    f.write(transcript)
                    temp_path = f.name

                result = subprocess.run(
                    ["gh", "gist", "create", temp_path, "--desc", f"LLM Council: {args.question[:50]}"],
                    capture_output=True, text=True
                )
                os.unlink(temp_path)

                if result.returncode == 0:
                    gist_url = result.stdout.strip()
                    print(f"\nShared: {gist_url}")
                else:
                    print(f"Gist creation failed: {result.stderr}", file=sys.stderr)
            except FileNotFoundError:
                print("Error: 'gh' CLI not found. Install with: brew install gh", file=sys.stderr)

        # Log to history
        history_file = get_sessions_dir().parent / "history.jsonl"
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": args.question[:200],
            "session": str(session_path) if session_path else None,
            "gist": gist_url,
            "context": args.context,
            "models": [name for name, _, _ in COUNCIL],
            "collabeval": use_collabeval,
        }
        if args.auto:
            log_entry["mode"] = "auto"
            log_entry["difficulty"] = difficulty
        with open(history_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
