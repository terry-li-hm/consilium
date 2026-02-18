"""Textual TUI for consilium --tui.

Rich-formatted live view with phase tracking, cost/time display,
and token-by-token streaming. Reads the same live.md symlink as --watch.
"""

import os
import re
import subprocess
import time
from pathlib import Path

from rich.markdown import Markdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.theme import Theme
from textual.widgets import Footer, RichLog, Static

from .watch import LineType, classify, _inline_format, _resolve_target

# Flexoki dark theme — https://stephango.com/flexoki
_FLEXOKI = Theme(
    name="flexoki-dark",
    primary="#DA702C",      # orange-400
    secondary="#4385BE",    # blue-400
    accent="#8B7EC8",       # purple-400
    foreground="#CECDC3",   # base-200
    background="#100F0F",   # black
    success="#879A39",      # green-400
    warning="#D0A215",      # yellow-400
    error="#D14D41",        # red-400
    surface="#1C1B1A",      # base-950
    panel="#282726",        # base-900
    dark=True,
    variables={
        "block-cursor-text-style": "none",
        "footer-key-foreground": "#DA702C",
        "input-selection-background": "#4385BE 35%",
    },
)


class PhaseBar(Static):
    """Top bar: current phase, elapsed time, running cost."""

    phase = reactive("waiting\u2026")
    elapsed = reactive(0.0)
    cost = reactive(0.0)

    def render(self) -> Text:
        mins, secs = divmod(int(self.elapsed), 60)
        t = Text()
        t.append(f" {self.phase} ", style="bold")
        t.append(" \u2502 ", style="dim")
        t.append(f"{mins}:{secs:02d}", style="bold")
        t.append(" \u2502 ", style="dim")
        t.append(f"${self.cost:.3f}", style="bold yellow")
        return t


class StreamLine(Static):
    """Preview area — shows buffered lines + partial token during streaming."""
    pass


class ConsiliumTUI(App):
    """Live viewer for consilium council sessions."""

    CSS = """
    PhaseBar {
        dock: top;
        height: 1;
        background: $surface;
    }
    #log {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    #stream {
        height: auto;
        max-height: 8;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("y", "yank", "Copy all"),
        Binding("g", "scroll_bottom", "Bottom"),
    ]

    def __init__(self, live_link: Path):
        super().__init__()
        self.live_link = live_link
        self.current_target: Path | None = None
        self.fh = None
        self.partial_buf = ""
        self.flushed_partial = False
        self.prev_type: LineType | None = None
        self.in_failure_block = False
        self.start_time: float | None = None
        self.all_text: list[str] = []
        self._stats_re = re.compile(r"\([\d.]+s,\s*~?\$([\d.]+)\)")
        self._body_buffer: list[str] = []
        self._in_model_block = False

    def compose(self) -> ComposeResult:
        yield PhaseBar()
        yield RichLog(id="log", highlight=False, markup=False, auto_scroll=True, wrap=True)
        yield StreamLine(id="stream")
        yield Footer()

    def on_mount(self) -> None:
        self.register_theme(_FLEXOKI)
        self.theme = "flexoki-dark"
        self.set_interval(0.05, self._poll)
        self.set_interval(1.0, self._tick_elapsed)

    def _tick_elapsed(self) -> None:
        if self.start_time is not None:
            self.query_one(PhaseBar).elapsed = time.monotonic() - self.start_time

    # ── body buffer ───────────────────────────────────────────────

    def _flush_body(self, log: RichLog) -> None:
        """Render buffered body content as Markdown into the main log."""
        if not self._body_buffer:
            self._in_model_block = False
            return
        content = "\n".join(self._body_buffer)
        self._body_buffer.clear()
        self._in_model_block = False
        if content.strip():
            log.write(Markdown(content))

    # ── line renderer ────────────────────────────────────────────────

    def _render_line(self, line: str) -> None:
        """Classify and render a complete line into the RichLog."""
        log = self.query_one("#log", RichLog)
        bar = self.query_one(PhaseBar)
        stripped = line.strip()
        line_type, match = classify(line, self.prev_type)

        # Failure-block tracking
        if line_type == LineType.PHASE_BANNER and "FAILURE" in stripped.upper():
            self.in_failure_block = True
        elif line_type == LineType.SEPARATOR and self.in_failure_block:
            if self.prev_type != LineType.PHASE_BANNER:
                self.in_failure_block = False

        # Update header from phase banners
        if line_type == LineType.PHASE_BANNER:
            bar.phase = stripped

        # Accumulate cost from stats lines
        cost_match = self._stats_re.search(stripped)
        if cost_match:
            bar.cost += float(cost_match.group(1))

        # Flush body buffer on any structural (non-body) line
        is_body = line_type in (LineType.BODY, LineType.CONFIDENCE)
        if not is_body and self._in_model_block:
            self._flush_body(log)

        # ── write to log ──
        if line_type == LineType.SEPARATOR:
            log.write(Text("\u2500" * 60, style="dim"))

        elif line_type == LineType.PHASE_BANNER:
            t = Text()
            t.append(f" {stripped} ", style="bold cyan reverse")
            log.write(t)

        elif line_type == LineType.MODEL_HEADER:
            assert match is not None
            raw = match.group(1)
            role_m = re.match(r"^(.+?)\s+\((.+)\)$", raw)
            self._in_model_block = True
            log.write(Text(""))
            t = Text()
            if role_m:
                t.append(role_m.group(1), style="bold green")
                t.append(f" ({role_m.group(2)})", style="dim")
            else:
                t.append(raw, style="bold green")
            log.write(t)

        elif line_type == LineType.SECTION_HEADER:
            assert match is not None
            log.write(Text(match.group(1), style="bold yellow"))

        elif line_type == LineType.NOTICE:
            assert match is not None
            log.write(Text(f">>> {match.group(1)}", style="bold magenta"))

        elif line_type == LineType.STATUS:
            log.write(Text(stripped, style="dim italic"))

        elif line_type == LineType.STATS:
            log.write(Text(stripped, style="dim"))

        elif line_type == LineType.INFO:
            log.write(Text(stripped, style="dim"))

        elif self.in_failure_block:
            log.write(Text(stripped, style="red"))

        elif self._in_model_block:
            # Buffer body content for markdown rendering when block completes
            self._body_buffer.append(line.rstrip("\n"))

        elif not stripped:
            log.write(Text(""))

        else:
            log.write(_inline_format(line.rstrip("\n")))

        self.prev_type = line_type
        self.all_text.append(line)

    # ── file polling ─────────────────────────────────────────────────

    def _poll(self) -> None:
        bar = self.query_one(PhaseBar)
        stream = self.query_one("#stream", StreamLine)
        log = self.query_one("#log", RichLog)

        # Pause auto-scroll when user has scrolled up; resume at bottom
        at_bottom = log.max_scroll_y <= 0 or log.scroll_y >= log.max_scroll_y - 2
        log.auto_scroll = at_bottom

        # Resolve symlink
        new_target = _resolve_target(self.live_link)

        # Handle rotation
        if new_target != self.current_target:
            # Flush any pending body content before closing
            if self._in_model_block:
                self._flush_body(log)
            if self.fh is not None:
                self.fh.close()
                self.fh = None
                self.partial_buf = ""
                self.flushed_partial = False

            if new_target is not None:
                self.current_target = new_target
                self.fh = open(self.current_target, "r")
                self.start_time = time.monotonic()
                bar.phase = "starting\u2026"
                bar.elapsed = 0.0
                bar.cost = 0.0
                if self.prev_type is not None:
                    log.write(Text(""))
                    log.write(Text(" new session ", style="bold bright_blue reverse"))
                    log.write(Text(""))
                    self.prev_type = None
                    self.in_failure_block = False
            else:
                self.current_target = None

        if self.fh is None:
            return

        # Read whatever's available
        chunk = self.fh.read(4096)

        if chunk:
            self.partial_buf += chunk

            # Process all complete lines
            while "\n" in self.partial_buf:
                line, self.partial_buf = self.partial_buf.split("\n", 1)
                self._render_line(line)
                self.flushed_partial = False

            # Update stream preview: show buffer tail + partial during model blocks
            if self._in_model_block and (self._body_buffer or self.partial_buf):
                preview_lines = self._body_buffer[-7:]
                preview = "\n".join(preview_lines)
                if self.partial_buf:
                    preview = f"{preview}\n{self.partial_buf}" if preview else self.partial_buf
                stream.update(preview)
                self.flushed_partial = bool(self.partial_buf)
            elif self.partial_buf:
                stream.update(self.partial_buf)
                self.flushed_partial = True
            else:
                stream.update("")
        else:
            # File may have disappeared
            if self.current_target and not self.current_target.exists():
                if self._in_model_block:
                    self._flush_body(log)
                if self.fh is not None:
                    self.fh.close()
                    self.fh = None
                self.current_target = None
                self.partial_buf = ""
                self.flushed_partial = False
                stream.update("")

    # ── actions ──────────────────────────────────────────────────────

    def action_scroll_bottom(self) -> None:
        """Jump to bottom and re-enable auto-scroll."""
        log = self.query_one("#log", RichLog)
        log.auto_scroll = True
        log.scroll_end(animate=False)

    def action_yank(self) -> None:
        """Copy full transcript to clipboard."""
        text = "\n".join(self.all_text)
        try:
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
            self.notify("Copied to clipboard")
        except (FileNotFoundError, subprocess.CalledProcessError):
            self.notify("Copy failed", severity="error")


def run_tui(live_link: Path) -> None:
    """Entry point — launch the TUI."""
    app = ConsiliumTUI(live_link)
    app.run()
