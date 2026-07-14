"""Guardrail: the quality gate must examine every Python file in the repo.

The gate once linted `src tests` only. `scripts/` — the X session-import scripts
users run, and the `xbrain-transcribe` / `xbrain-vision` executables `digest-video`
shells out to — was never opened by ruff, while the summary printed ALL CRITICAL
CHECKS PASSED. The green was reporting more coverage than it had.

Widening the targets fixes *that* instance. These tests pin the *rule*, so the
next instance cannot happen quietly. Three ways the gate can silently lose
coverage again, one test each:

1. **The two definitions drift.** The lint/format targets were written twice —
   in the poe tasks and again in `scripts/check.sh`. Two lists that "should"
   match are exactly what CLAUDE.md rule 5 was paid in blood for, so the fix is
   to have ONE definition (the poe tasks) and make `check.sh` invoke it.
   `test_check_sh_does_not_restate_the_ruff_targets` keeps it that way.

2. **`lint` and `format` drift from each other.** They are two separate strings
   in `pyproject.toml`; nothing makes them agree.

3. **A Python file exists that the gate never opens.** The general form of the
   original bug — a new top-level directory nobody adds to the targets, or (the
   case that actually bit us) a shebang-python executable with no `.py` suffix,
   which ruff does not discover when handed a directory and which only
   `extend-include` brings in.

Test 3 is the load-bearing one and it asserts *behaviour*, not spelling: it asks
ruff itself which files it would open (`--show-files`, using the targets the gate
really declares) and compares that against every Python file git tracks. Both
sides are derived — neither hardcodes `{src, tests, scripts}` — so the day someone
adds `tools/*.py` and forgets the gate, this goes red and names the file.

"Every Python file git tracks" is deliberately sourced from `git ls-files`: it
inherits the repo's own ignore conventions, so `.venv`, `__pycache__`, build
artefacts and sibling worktrees can never leak in and make this test noisy. A
guardrail people learn to ignore is worse than no guardrail.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHECK_SH = REPO_ROOT / "scripts" / "check.sh"


# ---------------------------------------------------------------------------
# Deriving the gate's declared scope (the ONE definition)
# ---------------------------------------------------------------------------


def _poe_tasks() -> dict[str, str]:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["tool"]["poe"]["tasks"]


def _ruff_targets(command: str) -> set[str]:
    """Positional path targets of a `ruff …` command string.

    `ruff check src tests scripts`        -> {src, tests, scripts}
    `ruff format --check src tests scripts` -> {src, tests, scripts}

    Flags and the subcommand are dropped; whatever is left is a path target.
    Parsed with shlex rather than `.split()` so a quoted path stays one token.
    """
    tokens = shlex.split(command)
    assert tokens and tokens[0] == "ruff", f"not a ruff command: {command!r}"
    return {t for t in tokens[1:] if not t.startswith("-") and t not in {"check", "format"}}


def _declared_targets() -> set[str]:
    """The single source of truth for what the ruff gate examines."""
    return _ruff_targets(_poe_tasks()["lint"])


# `ruff check src tests` / `ruff format --check src tests`, matched ANYWHERE in a
# shell line — not anchored to its start. The offenders this test exists to reject
# were written `if uv run ruff check src tests 2>&1; then`, so a start-anchored
# match would have found nothing and left this test green against the exact code it
# is meant to catch. Stops at the first shell operator so `2>&1`, `; then` and pipes
# are not mistaken for path targets.
_RUFF_CALL = re.compile(r"\bruff\s+(?:check|format)\b([^;|&<>\n]*)")


def _strip_shell_comment(line: str) -> str:
    """Drop a trailing `#` comment.

    check.sh's own header comment lists "ruff check, ruff format, mypy, bandit" as
    the critical checks. Without this, that prose matches _RUFF_CALL and gets
    reported as a hardcoded target — a guardrail that cries wolf on a comment is a
    guardrail someone deletes.
    """
    for index, char in enumerate(line):
        if char == "#" and (index == 0 or line[index - 1].isspace()):
            return line[:index]
    return line


def _hardcoded_ruff_targets_in(shell_line: str) -> set[str]:
    """Positional ruff targets spelled out in a shell line, if any."""
    match = _RUFF_CALL.search(_strip_shell_comment(shell_line))
    if not match:
        return set()
    return {
        token
        for token in shlex.split(match.group(1))
        if not token.startswith("-") and token not in {"check", "format"}
    }


# ---------------------------------------------------------------------------
# Deriving the repo's actual Python (the truth to compare against)
# ---------------------------------------------------------------------------


def _tracked_files() -> list[Path]:
    """Every file git tracks. Inherits .gitignore, so no build/venv noise."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout — cannot derive the tracked file list")
    return [Path(p) for p in result.stdout.split("\0") if p]


def _is_python(relpath: Path) -> bool:
    """Python source, including the extensionless `#!/usr/bin/env python3` kind.

    The second case is the whole reason this file exists: ruff does not discover
    it when handed a directory, so it went unlinted for as long as it existed.
    """
    if relpath.suffix in {".py", ".pyi"}:
        return True
    if relpath.suffix:
        return False
    try:
        first_line = (REPO_ROOT / relpath).open("rb").readline()
    except OSError:  # tracked but absent from the working tree
        return False
    return first_line.startswith(b"#!") and b"python" in first_line


def _repo_python_files() -> set[Path]:
    return {p for p in _tracked_files() if _is_python(p)}


def _files_ruff_would_open(targets: set[str]) -> set[Path]:
    """Ask ruff itself which files it would examine. Behaviour, not spelling."""
    pytest.importorskip("ruff", reason="ruff not installed — install dev deps")
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *sorted(targets), "--show-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"ruff --show-files failed:\n{result.stderr}"
    return {
        Path(line).resolve().relative_to(REPO_ROOT) for line in result.stdout.split("\n") if line
    }


# ---------------------------------------------------------------------------
# 1. One definition, not two that "should" match  (CLAUDE.md rule 5)
# ---------------------------------------------------------------------------


def test_check_sh_does_not_restate_the_ruff_targets():
    """`scripts/check.sh` must invoke the poe tasks, not re-spell their targets.

    When check.sh hardcoded `ruff check src tests` alongside the poe task's own
    copy, the two could drift — and a local `poe lint` and CI would then disagree,
    silently, about what "clean" means. One definition (the poe task); check.sh
    consumes it.
    """
    offenders = [
        line.strip()
        for line in CHECK_SH.read_text().splitlines()
        if _hardcoded_ruff_targets_in(line)
    ]
    assert not offenders, (
        "scripts/check.sh restates the ruff targets instead of invoking the poe "
        "tasks, so the two definitions can drift (CLAUDE.md rule 5). Replace with "
        "`uv run poe lint` / `uv run poe format`. Offending line(s):\n  " + "\n  ".join(offenders)
    )

    body = CHECK_SH.read_text()
    for task in ("lint", "format"):
        assert f"poe {task}" in body, (
            f"scripts/check.sh no longer runs `poe {task}` — the gate would stop "
            f"{task}-checking entirely, and the summary would still print PASS."
        )


def test_lint_and_format_examine_the_same_tree():
    """`lint` and `format` are two strings; nothing but this test makes them agree.

    A `lint` widened to a new directory while `format` is forgotten leaves that
    directory format-unchecked, with the gate green.
    """
    tasks = _poe_tasks()
    lint = _ruff_targets(tasks["lint"])
    fmt = _ruff_targets(tasks["format"])
    assert lint == fmt, (
        "the poe `lint` and `format` tasks examine different trees — "
        f"only in lint: {sorted(lint - fmt) or 'none'}; "
        f"only in format: {sorted(fmt - lint) or 'none'}"
    )


# ---------------------------------------------------------------------------
# 2. The scope covers every Python file that exists  (the original bug, generalised)
# ---------------------------------------------------------------------------


def test_ruff_examines_every_python_file_in_the_repo():
    """Ruff's real file list must equal every Python file git tracks.

    Both sides are derived, so this cannot be satisfied by restating a constant:
    the left side is what ruff *does* (asked via `--show-files`), the right side is
    what is *on disk* (asked via `git ls-files`).

    It goes red for both ways the gate loses coverage:
      * a directory of Python missing from the targets (`scripts/` — the original bug);
      * a shebang-python executable with no `.py` suffix, which ruff skips when handed
        a directory unless `extend-include` names it (`scripts/xbrain-transcribe`,
        `scripts/xbrain-vision` — the half of the original bug that the obvious fix
        would have missed).
    """
    on_disk = _repo_python_files()
    examined = _files_ruff_would_open(_declared_targets())

    unexamined = on_disk - examined
    assert not unexamined, (
        "the quality gate never opens these Python files, so the gate can print "
        "ALL CRITICAL CHECKS PASSED while they are broken:\n  "
        + "\n  ".join(str(p) for p in sorted(unexamined))
        + f"\n\nThe gate declares targets {sorted(_declared_targets())}. Either add the "
        "missing directory to the `lint` and `format` poe tasks, or — if the file is an "
        "extensionless `#!/usr/bin/env python3` executable — add it to `extend-include` "
        "under [tool.ruff], because ruff does not discover those from a directory."
    )
