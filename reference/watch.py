"""Rich terminal renderer for consilium --watch.

Follows a live markdown file (symlink) and renders structural elements
(headers, separators, status lines) with Rich formatting. Body text
passes through with lightweight inline markdown transforms.

Only affects the watch renderer — main stdout and saved files stay plain markdown.
"""

import os
import re
import sys
import time
from enum import Enum, auto
from pathlib import Path

from rich.console import Console
from rich.text import Text


class LineType(Enum):
    SEPARATOR = auto()
    PHASE_BANNER = auto()
    MODEL_HEADER = auto()
    SECTION_HEADER = auto()
    NOTICE = auto()
    STATUS = auto()
    CONFIDENCE = auto()
    STATS = auto()
    INFO = auto()
    FAILURE = auto()
    BODY = auto()


# Compiled patterns in priority order
_PATTERNS: list[tuple[LineType, re.Pattern]] = [
    (LineType.SEPARATOR, re.compile(r"^={50,}$")),
    (LineType.MODEL_HEADER, re.compile(r"^### (.+)")),
    (LineType.SECTION_HEADER, re.compile(r"^## (?!#)(.+)")),
    (LineType.NOTICE, re.compile(r"^>>> (.+)")),
    (LineType.STATS, re.compile(r"^\(\d+\.\d+s, ~?\$[\d.]+\)$")),
    (LineType.CONFIDENCE, re.compile(r"^\s*(?:\*\*)?Confidence:\s*(.+?)(?:\*\*)?$")),
    (LineType.STATUS, re.compile(
        r"^\((?:thinking|querying|generating|round \d+|"
        r"querying \d+ (?:models|panelists|attackers) in parallel|"
        r"round \d+ done|"
        r"Models see each other|"
        r"Fallback enabled|"
        r"Persona context|"
        r"Challenger|"
        r"Contrarian challenger|"
        r"Auto-detected)"
    )),
    (LineType.INFO, re.compile(
        r"^(?:Council members:|Rounds:|Running |Difficulty:|"
        r"Question:|Domain context:|Classifying question)"
    )),
]

# Inline markdown: **bold**
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def classify(line: str, prev_type: LineType | None) -> tuple[LineType, re.Match | None]:
    """Classify a line and return its type + regex match."""
    stripped = line.strip()
    if not stripped:
        return LineType.BODY, None

    # Phase banner: all-caps title after a separator
    # e.g. "BLIND PHASE (independent claims)", "COUNCIL DELIBERATION", "FOLLOWUP: topic"
    if prev_type == LineType.SEPARATOR and len(stripped) > 3:
        # Strip parenthetical or colon-suffix before checking case
        core = re.sub(r"\s*\(.*\)$", "", stripped)
        core = re.sub(r":.*$", "", core)
        if core.isupper():
            return LineType.PHASE_BANNER, None

    for line_type, pattern in _PATTERNS:
        m = pattern.match(stripped)
        if m:
            return line_type, m

    return LineType.BODY, None


def _inline_format(text: str) -> Text:
    """Apply lightweight inline markdown transforms to body text."""
    result = Text()
    # Handle bullet points
    bullet_match = re.match(r"^(\s*)([-*])\s+(.*)$", text)
    numbered_match = re.match(r"^(\s*)(\d+\.)\s+(.*)$", text)

    if bullet_match:
        indent, bullet, content = bullet_match.groups()
        result.append(indent)
        result.append(f"{bullet} ", style="dim")
        _append_bold(result, content)
        return result
    elif numbered_match:
        indent, num, content = numbered_match.groups()
        result.append(indent)
        result.append(f"{num} ", style="dim")
        _append_bold(result, content)
        return result

    _append_bold(result, text)
    return result


def _append_bold(target: Text, text: str) -> None:
    """Append text with **bold** segments rendered as Rich bold."""
    last = 0
    for m in _BOLD_RE.finditer(text):
        if m.start() > last:
            target.append(text[last:m.start()])
        target.append(m.group(1), style="bold")
        last = m.end()
    if last < len(text):
        target.append(text[last:])


class Renderer:
    """Renders classified lines to a Rich console."""

    def __init__(self, console: Console):
        self.console = console
        self.prev_type: LineType | None = None
        self.in_failure_block = False

    def render(self, line: str) -> None:
        """Classify and render a single complete line."""
        line_type, match = classify(line, self.prev_type)

        # Track failure block state
        if line_type == LineType.PHASE_BANNER and "FAILURE" in line.strip().upper():
            self.in_failure_block = True
        elif line_type == LineType.SEPARATOR and self.in_failure_block:
            # Second separator ends the failure block
            if self.prev_type != LineType.PHASE_BANNER:
                self.in_failure_block = False

        stripped = line.strip()
        c = self.console

        if line_type == LineType.SEPARATOR:
            c.rule(style="dim")

        elif line_type == LineType.PHASE_BANNER:
            c.rule(title=stripped, style="cyan bold")

        elif line_type == LineType.MODEL_HEADER:
            assert match is not None
            raw = match.group(1)
            # Split "model_name (role)" if present
            role_match = re.match(r"^(.+?)\s+\((.+)\)$", raw)
            text = Text()
            if role_match:
                name, role = role_match.groups()
                text.append(name, style="bold green")
                text.append(f" ({role})", style="dim")
            else:
                text.append(raw, style="bold green")
            c.print(text)

        elif line_type == LineType.SECTION_HEADER:
            assert match is not None
            c.print(Text(match.group(1), style="bold yellow"))

        elif line_type == LineType.NOTICE:
            assert match is not None
            c.print(Text(f">>> {match.group(1)}", style="bold magenta"))

        elif line_type == LineType.STATUS:
            c.print(Text(stripped, style="dim italic"))

        elif line_type == LineType.CONFIDENCE:
            c.print(Text(stripped, style="cyan"))

        elif line_type == LineType.STATS:
            c.print(Text(stripped, style="dim"))

        elif line_type == LineType.INFO:
            c.print(Text(stripped, style="dim"))

        elif self.in_failure_block:
            c.print(Text(stripped, style="red"))

        else:
            # Body text with inline formatting
            if not stripped:
                c.print()
            else:
                c.print(_inline_format(line.rstrip("\n")))

        self.prev_type = line_type


def _resolve_target(link_path: Path) -> Path | None:
    """Resolve a symlink to its target file, returning None if invalid."""
    try:
        if link_path.is_symlink():
            target = link_path.parent / os.readlink(link_path)
            if target.exists():
                return target
        elif link_path.exists():
            return link_path
    except OSError:
        pass
    return None


def watch_live(live_link: Path) -> None:
    """Follow a live markdown symlink and render with Rich formatting.

    Handles:
    - Symlink rotation (new council session starts)
    - Token-by-token streaming (partial lines flush immediately)
    - Clean Ctrl+C exit
    """
    console = Console(highlight=False)
    renderer = Renderer(console)

    current_target: Path | None = None
    fh = None
    partial_buf = ""
    flushed_partial = False

    try:
        while True:
            # Resolve symlink target
            new_target = _resolve_target(live_link)

            # Handle symlink rotation
            if new_target != current_target:
                if fh is not None:
                    fh.close()
                    fh = None
                    partial_buf = ""
                    flushed_partial = False

                if new_target is not None:
                    current_target = new_target
                    fh = open(current_target, "r")
                    # If we had a previous session, add visual separator
                    if renderer.prev_type is not None:
                        console.print()
                        console.rule(title="new session", style="bright_blue bold")
                        console.print()
                        renderer.prev_type = None
                        renderer.in_failure_block = False
                else:
                    current_target = None

            if fh is None:
                time.sleep(0.1)
                continue

            # Read available data — grab whatever's buffered
            chunk = fh.read(4096)

            if chunk:
                partial_buf += chunk

                # Process all complete lines in buffer
                while "\n" in partial_buf:
                    line, partial_buf = partial_buf.split("\n", 1)
                    if flushed_partial:
                        # Previously flushed partial — print suffix to finish
                        console.print(line)
                        flushed_partial = False
                    else:
                        renderer.render(line)

                # Flush remaining partial content immediately (token-by-token)
                if partial_buf and not flushed_partial:
                    console.print(partial_buf, end="")
                    flushed_partial = True
            else:
                # No new data — check for file/symlink changes
                if current_target and not current_target.exists():
                    if fh is not None:
                        fh.close()
                        fh = None
                    current_target = None
                    partial_buf = ""
                    flushed_partial = False

                time.sleep(0.05)

    except KeyboardInterrupt:
        # Clean exit
        if partial_buf:
            console.print(partial_buf)
        console.print()
    finally:
        if fh is not None:
            fh.close()
