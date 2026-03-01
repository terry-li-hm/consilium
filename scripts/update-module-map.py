#!/usr/bin/env python3
"""Update module map line counts in CLAUDE.md.

Run directly or via pre-commit hook. Updates all | `file.rs` | N | rows
in the module map table to reflect current wc -l counts.
"""
import re
import subprocess
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
claude_md = repo_root / "CLAUDE.md"

# Collect all .rs source files
rs_files = list(repo_root.glob("src/*.rs")) + list(repo_root.glob("src/modes/*.rs"))

result = subprocess.run(
    ["wc", "-l"] + rs_files,
    capture_output=True,
    text=True,
)

# Parse: "  123 /path/to/file.rs"
counts: dict[str, int] = {}
for line in result.stdout.strip().splitlines():
    parts = line.strip().split(None, 1)
    if len(parts) == 2 and not parts[1].endswith("total"):
        path = Path(parts[1])
        try:
            rel = path.relative_to(repo_root / "src")
            counts[str(rel)] = int(parts[0])
        except ValueError:
            pass

text = claude_md.read_text()


def replace_count(match: re.Match) -> str:
    filepath = match.group(1)   # e.g. "config.rs" or "modes/council.rs"
    suffix = match.group(2)     # " | description... |"
    new_count = counts.get(filepath, None)
    if new_count is None:
        return match.group(0)   # file not found, leave unchanged
    return f"| `{filepath}` | {new_count}{suffix}"


new_text = re.sub(
    r"\| `([^`]+\.rs)` \| \d+( \| [^|]+ \|)",
    replace_count,
    text,
)

if new_text != text:
    claude_md.write_text(new_text)
    print("Updated module map line counts in CLAUDE.md")
else:
    print("Module map line counts already up to date")
