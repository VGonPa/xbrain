"""Executable operator-contract guards for ``docs/digest-video.md``."""

from __future__ import annotations

import re
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _section(markdown: str, heading: str) -> str:
    """Return one level-3 markdown section, excluding the next peer section."""
    match = re.search(rf"^### {re.escape(heading)}\s*$", markdown, flags=re.MULTILINE)
    assert match is not None, f"missing documentation section: {heading}"
    tail = markdown[match.end() :]
    next_heading = re.search(r"^#{1,3} ", tail, flags=re.MULTILINE)
    return tail if next_heading is None else tail[: next_heading.start()]


def test_hollow_recovery_recipe_keeps_the_three_safety_flags_indivisible():
    """The public hollow-recovery recipe must never re-run ASR by accident.

    Inspect the command inside the named section, rather than accepting the
    tokens anywhere in the document: an unsafe recipe followed by explanatory
    prose that mentions ``--keep-transcript`` must still fail this guard.
    """
    markdown = (ROOT / "docs" / "digest-video.md").read_text(encoding="utf-8")
    section = _section(markdown, "Finding and re-digesting hollow items")
    bash_blocks = re.findall(r"```bash\n(.*?)```", section, flags=re.DOTALL)
    commands = [
        shlex.split(line)
        for block in bash_blocks
        for line in block.splitlines()
        if "digest-video" in line and not line.lstrip().startswith("#")
    ]

    assert len(commands) == 1, "the hollow-recovery section must expose one canonical command"
    command = commands[0]
    required = {"--frames", "--force", "--keep-transcript"}
    assert required <= set(command), (
        "the hollow-recovery command must keep --frames --force --keep-transcript together; "
        f"found: {shlex.join(command)}"
    )
