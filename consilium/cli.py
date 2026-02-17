"""CLI entry point for consilium."""

import argparse
import atexit
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


MODE_TITLES = {
    "quick": "Quick Council",
    "discuss": "Roundtable Discussion",
    "redteam": "Red Team",
    "socratic": "Socratic Examination",
    "oxford": "Oxford Debate",
    "solo": "Solo Council",
    "council": "Council Deliberation",
}


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


class LiveWriter:
    """Tee stdout to a file for live watching from another terminal."""

    def __init__(self, original, live_file):
        self._original = original
        self._file = live_file

    def write(self, data):
        self._original.write(data)
        if not self._file.closed:
            self._file.write(data)

    def flush(self):
        self._original.flush()
        if not self._file.closed:
            self._file.flush()

    def isatty(self):
        return self._original.isatty()

    @property
    def encoding(self):
        return self._original.encoding

    def fileno(self):
        return self._original.fileno()


def _save_session(
    question: str,
    transcript: str,
    mode: str,
    header_extra: str = "",
    no_save: bool = False,
    output: str | None = None,
    quiet: bool = False,
) -> Path | None:
    """Save transcript to user-specified location and auto-save to sessions directory."""
    if output:
        Path(output).write_text(transcript)
        if not quiet:
            print(f"Transcript saved to: {output}")

    if no_save:
        return None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = slugify(question)
    filename = f"{timestamp}-{mode}-{slug}.md" if mode != "council" else f"{timestamp}-{slug}.md"
    session_path = get_sessions_dir() / filename

    session_content = f"""# {MODE_TITLES.get(mode, 'Session')}

**Question:** {question}
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Mode:** {mode}
{header_extra}
---

{transcript}
"""
    session_path.write_text(session_content)
    if not quiet:
        print(f"Session saved to: {session_path}")

    return session_path


def _share_gist(question: str, transcript: str, mode: str, quiet: bool = False) -> str | None:
    """Upload transcript to secret GitHub Gist and return URL."""
    try:
        import tempfile
        title = MODE_TITLES.get(mode, "Session")
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', prefix=f'council-{mode}-', delete=False
        ) as f:
            f.write(f"# {title}\n\n")
            f.write(f"**Question:** {question}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n")
            f.write(transcript)
            temp_path = f.name

        result = subprocess.run(
            ["gh", "gist", "create", temp_path, "--desc", f"{title}: {question[:50]}"],
            capture_output=True, text=True
        )
        os.unlink(temp_path)

        if result.returncode == 0:
            gist_url = result.stdout.strip()
            print(f"\nShared: {gist_url}")
            return gist_url
        else:
            print(f"Gist creation failed: {result.stderr}", file=sys.stderr)
    except FileNotFoundError:
        print("Error: 'gh' CLI not found. Install with: brew install gh", file=sys.stderr)

    return None


def _log_history(
    question: str,
    mode: str,
    session_path: Path | None = None,
    gist_url: str | None = None,
    extra: dict | None = None,
    cost: float | None = None,
    duration: float | None = None,
) -> None:
    """Append entry to history.jsonl."""
    history_file = get_sessions_dir().parent / "history.jsonl"
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question[:200],
        "mode": mode,
        "cost": round(cost, 4) if cost is not None else None,
        "duration": round(duration, 1) if duration is not None else None,
        "session": str(session_path) if session_path else None,
        "gist": gist_url,
    }
    if extra:
        log_entry.update(extra)
    with open(history_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def main():
    from .models import (
        COUNCIL,
        QUICK_MODELS,
        DISCUSS_MODELS,
        REDTEAM_MODELS,
        OXFORD_MODELS,
        classify_difficulty,
        detect_social_context,
    )
    from .prompts import ROLE_LIBRARY, DOMAIN_CONTEXTS

    parser = argparse.ArgumentParser(
        description="Multi-model deliberation CLI. Auto-routes by difficulty, or pick a mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Default: auto-routes by question difficulty (simple→quick, complex→council).

Modes:
  (no flag)   Auto-route — Opus classifies, then runs quick or council
  --quick     Parallel queries, no debate (~$0.10)
  --council   Full multi-round debate + judge (~$0.50)
  --discuss   Hosted roundtable exploration (~$0.30)
  --socratic  Socratic variant of discuss: probing questions (~$0.30)
  --oxford    Binary for/against with rebuttals + verdict (~$0.40)
  --redteam   Adversarial stress-test of a plan (~$0.20)
  --solo      Claude debates itself in multiple roles (~$0.40)

Examples:
  consilium "Should we use microservices or monolith?"
  consilium "What are the tradeoffs of SSR vs CSR?" --quick
  consilium "Career decision" --council --persona "builder who hates process"
  consilium "Is AI consciousness possible?" --discuss --rounds 0
  consilium "Hire seniors or train juniors?" --socratic
  consilium "Should we use microservices?" --oxford
  consilium "My plan: migrate to microservices..." --redteam
  consilium "Pricing strategy" --solo --roles "investor,founder,customer"

Session management:
  consilium --stats                        # Cost breakdown by mode
  consilium --watch                        # Live tail from another tmux tab
  consilium --view                         # View latest session
  consilium --search "career"              # Search all sessions
        """,
    )
    parser.add_argument("question", nargs="?", help="The question for the council to deliberate")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--output", "-o", help="Save transcript to file")
    parser.add_argument("--context", "-c", help="Context hint for the judge")
    parser.add_argument("--format", "-f", choices=["json", "yaml", "prose"], default="prose", dest="output_format", help="Output format")
    parser.add_argument("--share", action="store_true", help="Upload transcript to secret GitHub Gist")
    parser.add_argument("--persona", "-p", help="Context about the person asking")
    parser.add_argument("--domain", help="Regulatory domain context (banking, healthcare, eu, fintech, bio)")
    parser.add_argument("--challenger", help="Which model should argue contrarian")
    parser.add_argument("--followup", action="store_true", help="Enable followup mode after judge synthesis")
    parser.add_argument("--no-save", action="store_true", help="Don't auto-save transcript")
    parser.add_argument("--quick", action="store_true", help="Quick mode: parallel queries, no debate/judge")
    parser.add_argument("--council", action="store_true", help="Full council: skip auto-routing, always run debate + judge")
    parser.add_argument("--discuss", action="store_true", help="Discussion mode: hosted roundtable exploration")
    parser.add_argument("--redteam", action="store_true", help="Red team mode: adversarial stress-test")
    parser.add_argument("--solo", action="store_true", help="Solo council: Claude debates itself")
    parser.add_argument("--socratic", action="store_true", help="Socratic mode: probing questions to expose assumptions")
    parser.add_argument("--oxford", action="store_true", help="Oxford debate: binary for/against with rebuttals and verdict")
    parser.add_argument("--motion", help="Override auto-generated debate motion for --oxford")
    parser.add_argument("--roles", help="Custom perspectives for --solo (comma-separated, >= 2)")
    parser.add_argument("--rounds", type=int, default=None, metavar="N", help="Rounds for --discuss or --socratic (0 = unlimited)")
    parser.add_argument("--list-roles", action="store_true", help="List available predefined roles")
    parser.add_argument("--stats", action="store_true", help="Show session analytics")
    parser.add_argument("--sessions", action="store_true", help="List recent sessions")
    parser.add_argument("--watch", action="store_true", help="Watch live council output (tail -F)")
    parser.add_argument("--view", nargs="?", const="latest", default=None, metavar="TERM", help="View a session in pager")
    parser.add_argument("--search", metavar="TERM", help="Search session content")
    args = parser.parse_args()

    # Handle --sessions
    if args.sessions:
        sessions_dir = get_sessions_dir()
        sessions = sorted(sessions_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not sessions:
            print("No sessions found.")
        else:
            print(f"Sessions in {sessions_dir}:\n")
            for s in sessions[:20]:
                mtime = datetime.fromtimestamp(s.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                print(f"  {mtime}  {s.name}")
            if len(sessions) > 20:
                print(f"\n  ... and {len(sessions) - 20} more")
        sys.exit(0)

    # Handle --list-roles
    if args.list_roles:
        print("Available predefined roles for --solo --roles:\n")
        for name, desc in ROLE_LIBRARY.items():
            short = desc[:80] + "..." if len(desc) > 80 else desc
            print(f"  {name:<20s} {short}")
        print(f"\nDefault: Advocate, Skeptic, Pragmatist")
        print(f"Unknown roles use a generic prompt. Any name works.")
        sys.exit(0)

    # Handle --watch
    if args.watch:
        live_link = get_sessions_dir().parent / "live.md"
        # Clean up stale symlink pointing to non-existent file
        if live_link.is_symlink() and not live_link.exists():
            live_link.unlink()
        if not live_link.exists() and not live_link.is_symlink():
            live_link.touch()
            print("(no active session — waiting for output...)", file=sys.stderr, flush=True)
        os.execvp("tail", ["tail", "-F", str(live_link)])

    # Handle --view
    if args.view is not None:
        sessions_dir = get_sessions_dir()
        sessions = sorted(sessions_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not sessions:
            print("No sessions found.")
            sys.exit(1)

        if args.view == "latest":
            target = sessions[0]
        else:
            term = args.view.lower()
            matches = [s for s in sessions if term in s.name.lower()]
            if not matches:
                matches = [s for s in sessions if term in s.read_text()[:2000].lower()]
            if not matches:
                print(f"No sessions matching '{args.view}'.")
                sys.exit(1)
            if len(matches) > 1:
                print(f"({len(matches)} matches, showing most recent)\n")
            target = matches[0]

        pager = os.environ.get("PAGER", "less")
        os.execvp(pager, [pager, str(target)])

    # Handle --search
    if args.search:
        sessions_dir = get_sessions_dir()
        sessions = sorted(sessions_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        term = args.search.lower()
        matches = []
        for s in sessions:
            if term in s.read_text()[:3000].lower():
                matches.append(s)

        if not matches:
            print(f"No sessions matching '{args.search}'.")
        else:
            print(f"Sessions matching '{args.search}':\n")
            for s in matches[:20]:
                mtime = datetime.fromtimestamp(s.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                content = s.read_text()[:500]
                q_match = re.search(r'\*\*Question:\*\* (.+)', content)
                question = q_match.group(1)[:60] if q_match else s.stem
                print(f"  {mtime}  {question}")
            if len(matches) > 20:
                print(f"\n  ... and {len(matches) - 20} more")
        sys.exit(0)

    # Handle --stats
    if args.stats:
        from collections import defaultdict
        history_file = get_sessions_dir().parent / "history.jsonl"
        if not history_file.exists():
            print("No history found.")
            sys.exit(0)

        entries = []
        for line in history_file.read_text().splitlines():
            if line.strip():
                entries.append(json.loads(line))

        if not entries:
            print("No sessions recorded.")
            sys.exit(0)

        by_mode = defaultdict(list)
        for e in entries:
            by_mode[e.get("mode", "unknown")].append(e)

        first = entries[0].get("timestamp", "?")[:10]
        last = entries[-1].get("timestamp", "?")[:10]

        print(f"Consilium Stats ({len(entries)} sessions, {first} — {last})\n")
        print(f"{'Mode':<14s} {'Sessions':>8s} {'Avg Cost':>10s} {'Total Cost':>12s} {'Avg Time':>10s}")

        total_cost = 0.0
        for mode in sorted(by_mode.keys()):
            mode_entries = by_mode[mode]
            count = len(mode_entries)
            costs = [e.get("cost") for e in mode_entries if e.get("cost") is not None]
            durations = [e.get("duration") for e in mode_entries if e.get("duration") is not None]

            avg_cost = f"${sum(costs)/len(costs):.2f}" if costs else "\u2014"
            sum_cost = sum(costs) if costs else 0
            total_cost += sum_cost
            total_cost_str = f"${sum_cost:.2f}" if costs else "\u2014"
            avg_dur = f"{sum(durations)/len(durations):.0f}s" if durations else "\u2014"

            print(f"{mode:<14s} {count:>8d} {avg_cost:>10s} {total_cost_str:>12s} {avg_dur:>10s}")

        print(f"\n{'Total':<14s} {len(entries):>8d} {'':>10s} ${total_cost:>11.2f}")

        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        recent = [e for e in entries if e.get("timestamp", "") >= cutoff]
        recent_cost = sum(e.get("cost", 0) or 0 for e in recent)
        print(f"\nLast 7 days: {len(recent)} sessions, ${recent_cost:.2f}")

        sys.exit(0)

    # Require question for normal operation
    if not args.question:
        parser.error("the following arguments are required: question")

    # Validate mode flags
    mode_flags = [f for f in ["--quick", "--council", "--discuss", "--redteam", "--solo", "--socratic", "--oxford"] if getattr(args, f.lstrip("-"))]
    if len(mode_flags) > 1:
        parser.error(f"{' and '.join(mode_flags)} are mutually exclusive")

    if args.quick:
        debate_flags = []
        if args.challenger:
            debate_flags.append("--challenger")
        if args.followup:
            debate_flags.append("--followup")
        if debate_flags:
            parser.error(f"--quick is incompatible with: {', '.join(debate_flags)}")

    for mode in ("discuss", "redteam", "solo", "socratic", "oxford"):
        if getattr(args, mode):
            incompatible = []
            if args.challenger:
                incompatible.append("--challenger")
            if args.followup:
                incompatible.append("--followup")
            if args.output_format != "prose":
                incompatible.append("--format")
            if incompatible:
                parser.error(f"--{mode} is incompatible with: {', '.join(incompatible)}")

    if hasattr(args, 'roles') and args.roles and not args.solo:
        parser.error("--roles requires --solo")

    if args.rounds is not None and not (args.discuss or args.socratic):
        parser.error("--rounds requires --discuss or --socratic")

    if args.motion and not args.oxford:
        parser.error("--motion requires --oxford")

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

    # Set up live file for watching from another terminal
    live_dir = get_sessions_dir().parent
    if not args.quiet:
        live_pid_path = live_dir / f"live-{os.getpid()}.md"
        live_link = live_dir / "live.md"
        _live_file = open(live_pid_path, "w")
        sys.stdout = LiveWriter(sys.stdout, _live_file)

        try:
            live_link.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            live_link.symlink_to(live_pid_path.name)
        except OSError:
            pass

        def _cleanup_live():
            _live_file.close()
            try:
                live_pid_path.unlink(missing_ok=True)
            except OSError:
                pass

        atexit.register(_cleanup_live)

    # Default: auto-route by difficulty (unless an explicit mode was chosen)
    difficulty = None
    explicit_mode = any(getattr(args, f) for f in ("quick", "council", "discuss", "redteam", "solo", "socratic", "oxford"))
    if not explicit_mode:
        if not args.quiet:
            print("Classifying question difficulty...", flush=True)
        difficulty = classify_difficulty(args.question, api_key)
        if not args.quiet:
            print(f"Difficulty: {difficulty}")
            print()

        if difficulty == "simple":
            args.quick = True

    try:
        # Quick mode
        if args.quick:
            from .quick import run_quick

            if not args.quiet:
                print(f"Running quick council ({len(QUICK_MODELS)} models)...")
                print()

            result = run_quick(
                question=args.question,
                models=QUICK_MODELS,
                api_key=api_key,
                google_api_key=google_api_key,
                verbose=not args.quiet,
                format=args.output_format,
            )

            session_path = _save_session(args.question, result.transcript, "quick",
                header_extra=f"**Models:** {', '.join(m.split('/')[-1] for _, m, _ in QUICK_MODELS)}",
                no_save=args.no_save, output=args.output, quiet=args.quiet)
            gist_url = _share_gist(args.question, result.transcript, "quick") if args.share else None
            _log_history(args.question, "quick", session_path, gist_url,
                {"models": [m.split("/")[-1] for _, m, _ in QUICK_MODELS]},
                cost=result.cost, duration=result.duration)
            sys.exit(0)

        # Discussion mode
        if args.discuss:
            from .discuss import run_discuss

            discuss_rounds = args.rounds if args.rounds is not None else 2
            if not args.quiet:
                rounds_label = "unlimited — Ctrl+C to wrap up" if discuss_rounds == 0 else f"{discuss_rounds} rounds"
                print(f"Running roundtable discussion ({len(DISCUSS_MODELS)} panelists, Claude hosting, {rounds_label})...")
                if args.persona:
                    print(f"(Persona context: {args.persona})")
                print()

            result = run_discuss(
                question=args.question,
                panelists=DISCUSS_MODELS,
                api_key=api_key,
                google_api_key=google_api_key,
                verbose=not args.quiet,
                persona=args.persona,
                domain=domain_context,
                rounds=discuss_rounds,
            )

            session_path = _save_session(args.question, result.transcript, "discuss",
                header_extra=f"**Panelists:** {', '.join(name for name, _, _ in DISCUSS_MODELS)}",
                no_save=args.no_save, output=args.output, quiet=args.quiet)
            gist_url = _share_gist(args.question, result.transcript, "discuss") if args.share else None
            _log_history(args.question, "discuss", session_path, gist_url,
                {"models": [name for name, _, _ in DISCUSS_MODELS]},
                cost=result.cost, duration=result.duration)
            sys.exit(0)

        # Red team mode
        if args.redteam:
            from .redteam import run_redteam

            if not args.quiet:
                print(f"Running red team ({len(REDTEAM_MODELS)} attackers, Claude hosting)...")
                if args.persona:
                    print(f"(Persona context: {args.persona})")
                print()

            result = run_redteam(
                question=args.question,
                panelists=REDTEAM_MODELS,
                api_key=api_key,
                google_api_key=google_api_key,
                verbose=not args.quiet,
                persona=args.persona,
                domain=domain_context,
            )

            session_path = _save_session(args.question, result.transcript, "redteam",
                header_extra=f"**Attackers:** {', '.join(name for name, _, _ in REDTEAM_MODELS)}",
                no_save=args.no_save, output=args.output, quiet=args.quiet)
            gist_url = _share_gist(args.question, result.transcript, "redteam") if args.share else None
            _log_history(args.question, "redteam", session_path, gist_url,
                {"models": [name for name, _, _ in REDTEAM_MODELS]},
                cost=result.cost, duration=result.duration)
            sys.exit(0)

        # Socratic mode (probing questions, via discuss engine)
        if args.socratic:
            from .discuss import run_discuss

            socratic_rounds = args.rounds if args.rounds is not None else 2
            if not args.quiet:
                rounds_label = "unlimited — Ctrl+C to wrap up" if socratic_rounds == 0 else f"{socratic_rounds} rounds"
                print(f"Running Socratic examination ({len(DISCUSS_MODELS)} panelists, Claude examining, {rounds_label})...")
                if args.persona:
                    print(f"(Persona context: {args.persona})")
                print()

            result = run_discuss(
                question=args.question,
                panelists=DISCUSS_MODELS,
                api_key=api_key,
                google_api_key=google_api_key,
                verbose=not args.quiet,
                persona=args.persona,
                domain=domain_context,
                rounds=socratic_rounds,
                style="socratic",
            )

            session_path = _save_session(args.question, result.transcript, "socratic",
                header_extra=f"**Panelists:** {', '.join(name for name, _, _ in DISCUSS_MODELS)}",
                no_save=args.no_save, output=args.output, quiet=args.quiet)
            gist_url = _share_gist(args.question, result.transcript, "socratic") if args.share else None
            _log_history(args.question, "socratic", session_path, gist_url,
                {"models": [name for name, _, _ in DISCUSS_MODELS]},
                cost=result.cost, duration=result.duration)
            sys.exit(0)

        # Oxford debate mode
        if args.oxford:
            from .oxford import run_oxford

            if not args.quiet:
                print(f"Running Oxford debate ({', '.join(name for name, _, _ in OXFORD_MODELS)} debating, Claude judging)...")
                if args.motion:
                    print(f"(Motion override: {args.motion})")
                if args.persona:
                    print(f"(Persona context: {args.persona})")
                print()

            result = run_oxford(
                question=args.question,
                debaters=OXFORD_MODELS,
                api_key=api_key,
                google_api_key=google_api_key,
                verbose=not args.quiet,
                persona=args.persona,
                domain=domain_context,
                motion_override=args.motion,
            )

            session_path = _save_session(args.question, result.transcript, "oxford",
                header_extra=f"**Debaters:** {', '.join(name for name, _, _ in OXFORD_MODELS)}",
                no_save=args.no_save, output=args.output, quiet=args.quiet)
            gist_url = _share_gist(args.question, result.transcript, "oxford") if args.share else None
            _log_history(args.question, "oxford", session_path, gist_url,
                {"models": [name for name, _, _ in OXFORD_MODELS]},
                cost=result.cost, duration=result.duration)
            sys.exit(0)

        # Solo council mode
        if args.solo:
            from .solo import run_solo

            custom_roles = [r.strip().title() for r in args.roles.split(",")] if args.roles else None

            if not args.quiet:
                if custom_roles:
                    print(f"Running solo council ({' / '.join(custom_roles)})...")
                else:
                    print("Running solo council (Advocate / Skeptic / Pragmatist)...")
                if args.persona:
                    print(f"(Persona context: {args.persona})")
                print()

            result = run_solo(
                question=args.question,
                api_key=api_key,
                verbose=not args.quiet,
                persona=args.persona,
                domain=domain_context,
                roles=custom_roles,
            )

            roles_label = " / ".join(custom_roles) if custom_roles else "Advocate / Skeptic / Pragmatist"
            session_path = _save_session(args.question, result.transcript, "solo",
                header_extra=f"**Model:** Claude ({roles_label})",
                no_save=args.no_save, output=args.output, quiet=args.quiet)
            gist_url = _share_gist(args.question, result.transcript, "solo") if args.share else None
            _log_history(args.question, "solo", session_path, gist_url,
                cost=result.cost, duration=result.duration)
            sys.exit(0)

        # Full council mode (explicit --council or auto-routed moderate/complex)
        from .council import run_council, run_followup_discussion

        if not args.quiet:
            mode_parts = ["anonymous", "blind"]
            if social_mode:
                mode_parts.append("social")
            print(f"Running LLM Council ({', '.join(mode_parts)})...")
            if google_api_key:
                print("(Fallback enabled: Gemini→AI Studio)")
            if args.persona:
                print(f"(Persona context: {args.persona})")
            active_challenger_idx = challenger_idx if challenger_idx is not None else 0
            active_challenger_name = COUNCIL[active_challenger_idx][0]
            print(f"(Challenger: {active_challenger_name})")
            print()

        # Skip collabeval for moderate auto-routed questions; use for complex or explicit --council
        use_collabeval = args.council or difficulty != "moderate"

        result = run_council(
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
            format=args.output_format,
            collabeval=use_collabeval,
        )

        transcript = result.transcript

        # Followup mode
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

        mode_label = f"anonymous, blind{', social' if social_mode else ''}{f', auto ({difficulty})' if difficulty else ''}{', collabeval' if use_collabeval else ''}"
        header_lines = f"**Mode:** {mode_label}"
        if args.context:
            header_lines += f"\n**Context:** {args.context}"
        if args.persona:
            header_lines += f"\n**Persona:** {args.persona}"

        session_path = _save_session(args.question, transcript, "council",
            header_extra=header_lines,
            no_save=args.no_save, output=args.output, quiet=args.quiet)
        gist_url = _share_gist(args.question, transcript, "council") if args.share else None

        history_extra = {
            "context": args.context,
            "models": [name for name, _, _ in COUNCIL],
            "collabeval": use_collabeval,
        }
        if difficulty:
            history_extra["difficulty"] = difficulty
        _log_history(args.question, "council", session_path, gist_url,
            history_extra, cost=result.cost, duration=result.duration)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
