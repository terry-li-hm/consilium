"""Snapshot and functional tests for the TUI.

Feeds a known consilium transcript into ConsiliumTUI via a temp file
and verifies rendering (markdown blocks, phase bar, structural elements).

Uses anyio (already a textual dependency) for async test support.
"""

from pathlib import Path

import pytest
from textual.widgets import RichLog

from consilium.tui import ConsiliumTUI, PhaseBar

# A minimal consilium transcript covering key line types
SAMPLE_TRANSCRIPT = """\
======================================================
BLIND PHASE (independent claims)
======================================================
(querying 6 models in parallel...)

### claude-opus-4-6
Here is my response with **bold text** and a list:

- First item
- Second item
- Third item with **emphasis**

**Confidence: 8/10**

### gpt-5.2-pro
# A Heading

Some paragraph text here.

## Subheading

1. Numbered item one
2. Numbered item two

```python
def hello():
    print("world")
```

**Confidence: 9/10**

======================================================
JUDGMENT
======================================================
### claude-opus-4-6 (judge)
The council has deliberated.

(45.2s, ~$0.18)
"""


@pytest.fixture
def tui_with_transcript(tmp_path):
    """Create a TUI pointing at a temp file with known content."""
    transcript_file = tmp_path / "live.md"
    transcript_file.write_text(SAMPLE_TRANSCRIPT)
    live_link = tmp_path / "live"
    live_link.symlink_to(transcript_file)
    return ConsiliumTUI(live_link)


@pytest.mark.anyio
async def test_tui_renders_without_crash(tui_with_transcript):
    """Basic smoke test — app mounts and polls without error."""
    app = tui_with_transcript
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.5)
        log = app.query_one("#log", RichLog)
        assert len(app.all_text) > 0


@pytest.mark.anyio
async def test_phase_bar_updates(tui_with_transcript):
    """Phase bar should show the last phase banner and accumulate cost."""
    app = tui_with_transcript
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.5)
        bar = app.query_one(PhaseBar)
        assert "JUDGMENT" in bar.phase
        assert bar.cost == pytest.approx(0.18, abs=0.01)


@pytest.mark.anyio
async def test_all_text_captured(tui_with_transcript):
    """all_text should contain every line for clipboard yank."""
    app = tui_with_transcript
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.5)
        full = "\n".join(app.all_text)
        assert "claude-opus-4-6" in full
        assert "gpt-5.2-pro" in full
        assert "Confidence: 8/10" in full


@pytest.mark.anyio
async def test_markdown_blocks_flushed(tui_with_transcript):
    """Body content should be flushed as Markdown, not left in buffer."""
    app = tui_with_transcript
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.5)
        assert len(app._body_buffer) == 0
        assert not app._in_model_block


@pytest.mark.anyio
async def test_scroll_bottom_action(tui_with_transcript):
    """The 'g' keybinding should scroll to bottom."""
    app = tui_with_transcript
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.5)
        await pilot.press("g")
        log = app.query_one("#log", RichLog)
        assert log.auto_scroll is True
