"""CLI entry point for frontier-council."""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_sessions_dir() -> Path:
    """Get the sessions directory, creating if needed."""
    sessions_dir = Path.home() / ".frontier-council" / "sessions"
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
    detect_social_context,
    run_council,
)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Council - Multi-model deliberation for important decisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  frontier-council "Should we use microservices or monolith?"
  frontier-council "What questions should I ask?" --social
  frontier-council "Career decision" --persona "builder who hates process work"
  frontier-council "Architecture choice" --rounds 3 --output transcript.md
        """,
    )
    parser.add_argument("question", help="The question for the council to deliberate")
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of deliberation rounds (default: 2, exits early on consensus)",
    )
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
        "--named",
        action="store_true",
        help="Show real model names instead of anonymous Speaker 1, 2, etc.",
    )
    parser.add_argument(
        "--no-blind",
        action="store_true",
        help="Skip blind first-pass (faster, but more anchoring bias)",
    )
    parser.add_argument(
        "--context", "-c",
        help="Context hint for the judge (e.g., 'architecture decision', 'ethics question')",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Upload transcript to secret GitHub Gist and print URL",
    )
    parser.add_argument(
        "--social",
        action="store_true",
        help="Enable social calibration mode (for interview questions, outreach, networking)",
    )
    parser.add_argument(
        "--persona", "-p",
        help="Context about the person asking (e.g., 'builder who hates process work')",
    )
    parser.add_argument(
        "--advocate",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Which speaker (1-5) should be devil's advocate (default: random)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't auto-save transcript to ~/.frontier-council/sessions/",
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

    # Auto-detect social context if not explicitly set
    social_mode = args.social or detect_social_context(args.question)
    if social_mode and not args.social and not args.quiet:
        print("(Auto-detected social context - enabling social calibration mode)")
        print()

    # Get API keys
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    google_api_key = os.environ.get("GOOGLE_API_KEY")
    moonshot_api_key = os.environ.get("MOONSHOT_API_KEY")

    use_blind = not args.no_blind

    if not args.quiet:
        mode_parts = []
        mode_parts.append("named" if args.named else "anonymous")
        mode_parts.append("blind first-pass" if use_blind else "no blind phase")
        if social_mode:
            mode_parts.append("social calibration")
        print(f"Running LLM Council ({', '.join(mode_parts)})...")
        fallbacks = []
        if google_api_key:
            fallbacks.append("Gemini→AI Studio")
        if moonshot_api_key:
            fallbacks.append("Kimi→Moonshot")
        if fallbacks:
            print(f"(Fallbacks enabled: {', '.join(fallbacks)})")
        print()

    try:
        advocate_idx = (args.advocate - 1) if args.advocate else random.randint(0, len(COUNCIL) - 1)

        if not args.quiet and args.persona:
            print(f"(Persona context: {args.persona})")
            print()
        if not args.quiet:
            advocate_name = COUNCIL[advocate_idx][0]
            print(f"(Devil's advocate: {advocate_name})")
            print()

        transcript, failed_models = run_council(
            question=args.question,
            council_config=COUNCIL,
            api_key=api_key,
            google_api_key=google_api_key,
            moonshot_api_key=moonshot_api_key,
            rounds=args.rounds,
            verbose=not args.quiet,
            anonymous=not args.named,
            blind=use_blind,
            context=args.context,
            social_mode=social_mode,
            persona=args.persona,
            advocate_idx=advocate_idx,
        )

        # Print failure summary
        if failed_models and not args.quiet:
            print()
            print("=" * 60)
            print("⚠️  MODEL FAILURES")
            print("=" * 60)
            for failure in failed_models:
                print(f"  • {failure}")
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

            # Build full session content with metadata header
            session_content = f"""# Council Session

**Question:** {args.question}
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Rounds:** {args.rounds}
**Mode:** {"named" if args.named else "anonymous"}, {"blind" if use_blind else "no blind"}{", social" if social_mode else ""}
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
                    print(f"\n🔗 Shared: {gist_url}")
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
            "rounds": args.rounds,
            "blind": use_blind,
            "models": [name for name, _, _ in COUNCIL],
        }
        with open(history_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
