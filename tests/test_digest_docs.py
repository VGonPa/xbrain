"""Executable operator-contract guards for ``docs/digest-video.md``."""

from __future__ import annotations

import re
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHELL_LANGUAGES = {"", "bash", "sh", "shell"}


def _section(markdown: str, heading: str) -> str:
    """Return one level-3 section; headings inside code fences are content."""
    in_section = False
    fence: tuple[str, int] | None = None
    lines: list[str] = []

    for line in markdown.splitlines():
        fence_match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if fence_match is not None:
            marker = fence_match.group(1)
            if fence is None:
                fence = (marker[0], len(marker))
            elif marker[0] == fence[0] and len(marker) >= fence[1]:
                fence = None
            if in_section:
                lines.append(line)
            continue

        if fence is None and re.match(r"^#{1,3} ", line):
            if in_section:
                break
            in_section = line.strip() == f"### {heading}"
            continue

        if in_section:
            lines.append(line)

    assert in_section, f"missing documentation section: {heading}"
    return "\n".join(lines)


def _shell_blocks(markdown: str) -> list[str]:
    """Return shell-like fenced blocks, preserving their command text."""
    blocks: list[str] = []
    block: list[str] | None = None
    fence: tuple[str, int] | None = None

    for line in markdown.splitlines():
        fence_match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if fence_match is not None:
            marker = fence_match.group(1)
            if fence is None:
                fence = (marker[0], len(marker))
                language = fence_match.group(2).strip().split(maxsplit=1)
                block = [] if (not language or language[0].lower() in SHELL_LANGUAGES) else None
            elif marker[0] == fence[0] and len(marker) >= fence[1]:
                if block is not None:
                    blocks.append("\n".join(block))
                block = None
                fence = None
            continue

        if block is not None:
            block.append(line)

    return blocks


def _digest_video_commands(markdown: str) -> list[list[str]]:
    """Tokenise every digest-video command, honouring shell comments/continuations."""
    commands: list[list[str]] = []
    for block in _shell_blocks(markdown):
        logical_lines = block.replace("\\\n", " ").splitlines()
        for line in logical_lines:
            if "digest-video" not in line:
                continue
            tokens = shlex.split(line, comments=True)
            if "digest-video" in tokens:
                commands.append(tokens)
    return commands


def test_hollow_recovery_recipe_keeps_the_three_safety_flags_indivisible():
    """The public hollow-recovery recipe must never re-run ASR by accident.

    Inspect the command inside the named section, rather than accepting the
    tokens anywhere in the document: an unsafe recipe followed by explanatory
    prose that mentions ``--keep-transcript`` must still fail this guard.
    """
    markdown = (ROOT / "docs" / "digest-video.md").read_text(encoding="utf-8")
    section = _section(markdown, "Finding and re-digesting hollow items")
    commands = _digest_video_commands(section)

    assert len(commands) == 1, "the hollow-recovery section must expose one canonical command"
    command = commands[0]
    required = {"--frames", "--force", "--keep-transcript"}
    assert required <= set(command), (
        "the hollow-recovery command must keep --frames --force --keep-transcript together; "
        f"found: {shlex.join(command)}"
    )
