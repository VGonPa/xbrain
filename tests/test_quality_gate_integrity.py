# tests/test_quality_gate_integrity.py
r"""The gate is the one thing nobody gates. This module gates it.

`scripts/check.sh` is a **required status check** on `develop` and `main`, with
`enforce_admins: true`. Everything else in this repo is defended by it. Nothing
defended it. Two sibling modules pin the gate's *edges* — `test_ci_workflow.py`
owns the workflow's triggers and job identity, `test_quality_gate_scope.py` owns
the ruff scope. This module owns the *middle*: what `check.sh` actually does, and
whether it still has teeth.

It matters more than it sounds. **A PR's `pull_request` run uses the HEAD version
of the workflow, not the base's** (measured). A PR that neuters the gate is judged
by the neutered gate. It absolves itself. Nothing upstream catches this class.

Fail-open vs fail-closed — the taxonomy that decides what is worth an assertion
-------------------------------------------------------------------------------
Not every way of breaking the gate is dangerous, and the safe ones deserve no test.

* **FAIL-CLOSED = safe. Do not spend assertions here.** Anything that stops the
  required check from *reporting* cannot ship a lie: GitHub never sees the check,
  waits for it forever, and the merge is `BLOCKED`. Measured on live probe PRs:
  deleting `quality.yml` → zero runs → BLOCKED. Renaming the job → the required
  context never appears → BLOCKED. `continue-on-error` at **job** level → the
  check run still reports `FAILURE` → BLOCKED. Path filters that skip the
  workflow → Pending forever → BLOCKED. The damage is a stuck PR, which is loud,
  and a human fixes it. **What can be REMOVED already guards itself.**

* **FAIL-OPEN = lethal. This is the entire job.** Anything that lets the check
  still say **PASS while testing less** manufactures false confidence and merges
  cleanly. Nobody is stuck, nobody looks, and the green badge now certifies less
  than it did. **What can be HOLLOWED OUT does not guard itself.**

The difference between the two can be a single level of YAML indentation:

    jobs.quality.continue-on-error: true          # check run: FAILURE → BLOCKED. Harmless.
    jobs.quality.steps[n].continue-on-error: true # check run: SUCCESS → CLEAN.  LETHAL.

Both measured against the live API with a genuinely failing gate. In the second
case the step ran `check.sh`, `check.sh` exited 1, and GitHub reported the step
`success`, the job `success`, the check run `SUCCESS`, and the PR `CLEAN`. The
failure existed nowhere except the raw log text.

The principle that receipt forces
---------------------------------
> **Never trust a reported conclusion. Assert on the SOURCE.**

We hold a receipt proving the conclusion field lies: a step that ran `exit 1` and
reported `"conclusion": "success"`. So a guard that checks "did every step
conclude successfully?" is defeated by the exact attack it exists to catch. Every
assertion below therefore parses `scripts/check.sh`, `.github/workflows/quality.yml`
and `pyproject.toml` and asserts on what they **say** — never on what a run
**reports**. Where a stronger answer is available, it asks the *tool* (pytest's
real collection), never the *verdict*.

What each fail-open edit costs, and where it is caught
-----------------------------------------------------
| Edit that keeps the check GREEN while testing less | Caught by |
|---|---|
| delete one `mark_failed "Tests"` call → test failures stop blocking | `test_every_critical_check_can_still_fail_the_gate` |
| hollow out `mark_failed()` → *all* critical checks become warnings | `test_mark_failed_still_records_the_failure` |
| final `exit 1` → `exit 0` → every check becomes decorative | `test_a_recorded_failure_still_exits_nonzero` |
| `COVERAGE_MIN=0` → the coverage floor removed | `test_coverage_floor_is_never_lowered` |
| lower `fail_under` only → the two floors drift; the laxer one wins | `test_the_coverage_floor_has_a_single_definition` |
| `--cov=src/xbrain/cli.py`, or `omit` a module → floor holds over a sliver | `test_coverage_measures_the_whole_package` |
| `pytest --ignore=…` / a narrowed `testpaths` → a suite goes dark | `test_the_gate_collects_every_test_file_in_the_repo` |
| `pytest -k …` / `--deselect …` → individual tests silently dropped | `test_the_gate_deselects_nothing` (+ the flag ban) |
| `mypy src/xbrain/cli.py`, `detect-secrets scan src/xbrain` → tool sees a sliver | `test_the_critical_tools_still_scan_the_whole_package` |
| `ignore_errors`/`exclude_dirs`/`-lll` → same narrowing, hidden in pyproject | `test_the_critical_tools_are_not_blinded_from_pyproject` |
| gut the job's `steps:`; or `bash scripts/check.sh \|\| true` | `test_the_workflow_runs_the_gate_and_lets_it_fail_the_job` |
| `continue-on-error` on the gate's **step** → red gate reported GREEN | `test_no_step_in_the_gate_job_may_continue_on_error` |

Behaviour, not spelling
-----------------------
Where an assertion can ask the tool itself, it does. The load-bearing test here
(`test_the_gate_collects_every_test_file_in_the_repo`) does not read the pytest
command and reason about it — it *runs* pytest with the exact arguments the gate
uses and compares the files pytest really collects against every test file git
tracks. Both sides are derived; neither hardcodes a list. Reformat the command,
move it to `poe test`, restructure `tests/` — all stay green. Skip a file by any
mechanism at all and it goes red naming the file. Same discipline for the shell
parsing: an early version of this file matched the word `pytest` inside check.sh's
own *error message* and tried to run the suite with `output"` as an argument, so
the parser is a real tokenizer now, not a regex over prose.

The honest limits — read these before trusting this module
----------------------------------------------------------
Two holes. Neither can be closed from inside this file, and pretending otherwise
would be worse than the holes.

**1. This file can be deleted in the same diff that weakens the gate.** These
tests run *inside* the gate they guard. Remove the guard and you remove the
failure. That is not fixable with more machinery: any test that watches the gate
can itself be deleted, and a test watching *that* test inherits the same hole.

**2. `continue-on-error` on the gate's own step masks this module's own red.**
`test_no_step_in_the_gate_job_may_continue_on_error` goes red the moment that line
appears — but the red arrives *through* `check.sh`, which the same line tells
GitHub to ignore. The test fails, `check.sh` exits 1, the step is forgiven, the
check run reports SUCCESS. My alarm rings inside the soundproofed room. It still
appears in the log and in the diff; it does **not** block the merge. On that one
attack this assertion is documentation with a test's syntax, and it is the single
most dangerous edit in the threat model. The only mechanical escape is a *second*
required check in a *separate job* — a step-level `continue-on-error` in the gate
job cannot mask a failure in a different job, and deleting that job outright is
fail-closed (the required context never reports → BLOCKED). That is repo
infrastructure, not a test, and it is not in this PR.

The principle that says where the regress *does* terminate:

> The escape from "nothing catches itself" is not another catcher. It is making
> the *absence* of the catcher fail closed. Where a guard's removal blocks the
> merge, the regress terminates. Where it only removes a test, it does not — and
> that residue is social, not mechanical.

Branch protection sits on the terminating side. This file cannot.

So why write it? Because it converts a **silent** weakening into a **visible**
one. Today `COVERAGE_MIN=0` is a one-character diff that nothing notices. After
this file it is a one-character diff *plus* the deletion of a test that says, in
words, what you are about to give up and why it mattered. The reviewer no longer
has to notice the absence of an alarm; they have to approve its removal. That is
the whole of what a test on this side of the line can buy — and it is worth buying.
"""

from __future__ import annotations

import fnmatch
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
CHECK_SH = REPO_ROOT / "scripts" / "check.sh"
PYPROJECT = REPO_ROOT / "pyproject.toml"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "quality.yml"

# The job whose check run branch protection requires. Its *identity* (id, name,
# triggers) is test_ci_workflow.py's subject; this module needs it only to reach the
# steps, because a job that reports `quality` without running the gate is a green light
# for untested code.
GATE_JOB = "quality"
GATE_SCRIPT = "scripts/check.sh"

# The package the gate exists to protect. Every critical tool must see all of it.
PACKAGE = "src/xbrain"

# The checks whose failure must EXIT THE GATE, not merely print a warning.
#
# check.sh has exactly one mechanism for this: a check that must block calls
# `mark_failed "<Label>"`, which appends to `FAILED_CHECKS`; the script's last statement
# exits 1 iff `FAILED_CHECKS` is non-empty. A check with no `mark_failed` call is
# warn-only by construction (vulture, interrogate, deptry — deliberately). So "is this
# check critical?" reduces to "does its failure branch call `mark_failed`?", and demoting
# a check to warn-only is exactly the deletion of that one call.
#
# The labels are the *arguments* to those calls — check.sh's own names for its checks.
CRITICAL_CHECKS = frozenset(
    {
        "Ruff",  # ruff check
        "Format",  # ruff format --check
        "Mypy",  # type errors
        "Radon",  # complexity grade D/E/F (grade C stays warn-only, by design)
        "Bandit",  # security findings
        "Secrets",  # NEW secrets not in .secrets.baseline
        "Tests",  # pytest exit code
        "Coverage",  # below COVERAGE_MIN, or unmeasurable
    }
)

# Critical tools whose PATH SCOPE nobody else guards, and the trees each must examine.
#
# ruff and ruff-format are absent on purpose: their scope is the whole subject of
# tests/test_quality_gate_scope.py, which asks ruff itself which files it opens. The
# warn-only tools (vulture, interrogate, deptry) are absent too — they cannot manufacture
# false confidence, because they never blocked anything.
CRITICAL_TOOL_SCOPES = {
    "mypy": (PACKAGE,),
    "bandit": (PACKAGE,),
    "radon": (PACKAGE,),
    # A secret can be committed into a test fixture or a helper script just as easily as
    # into the package — that is why the scan covers three trees, and why dropping two of
    # them is a one-word edit worth a test.
    "detect-secrets": (PACKAGE, "tests", "scripts"),
}

# Subcommand words that sit where a path would. `radon cc src/xbrain`, `detect-secrets
# scan src/…` — without this, `cc` and `scan` would be read as path targets.
TOOL_SUBCOMMANDS = frozenset({"cc", "mi", "raw", "hal", "scan", "check", "format", "audit"})

# The coverage floor as a RATCHET, not an equality.
#
# `>= FLOOR` and `== current` fail in opposite directions:
#   * `== current` breaks on every legitimate RAISE (78 -> 85), so the first person to
#     improve coverage gets a red test for their trouble, learns the test is noise, and
#     edits the constant reflexively. A guard people are trained to edit is not a guard.
#   * no test at all guards nothing: `COVERAGE_MIN=0` is the attack.
# A ratchet permits every move that makes the gate stricter and rejects every move that
# makes it laxer, which is exactly the property wanted. Raising the floor means raising
# this constant too — a deliberate, reviewable, one-line commit.
COVERAGE_FLOOR = 78

# Flags that make pytest run FEWER tests than the suite contains. `-x` / `--maxfail` are
# absent on purpose: they stop early on failure, which cannot turn a red gate green.
TEST_NARROWING_FLAGS = ("-k", "-m", "--deselect", "--ignore", "--ignore-glob", "--collect-only")

# Shell fragments that discard a command's exit status. `bash scripts/check.sh || true`
# runs all 11 checks, prints a red FAILED summary — and exits 0.
EXIT_STATUS_SWALLOWERS = ("|| true", "|| :", "|| exit 0", "; true", "set +e", "|| echo")


# ---------------------------------------------------------------------------
# Reading the three places the truth lives
# ---------------------------------------------------------------------------


def _strip_shell_comment(line: str) -> str:
    """Drop a trailing `#` comment.

    check.sh documents its own severities in prose ("CRITICAL (exit 1 on failure): ruff
    check, ruff format, mypy, bandit..."). Without this, that comment is parsed as if it
    were code, and the assertions below could be satisfied — or broken — by editing a
    comment.
    """
    for index, char in enumerate(line):
        if char == "#" and (index == 0 or line[index - 1].isspace()):
            return line[:index]
    return line


def _check_sh_code() -> str:
    """`scripts/check.sh` with its comments removed. Code only."""
    return "\n".join(_strip_shell_comment(line) for line in CHECK_SH.read_text().splitlines())


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _table(*path: str) -> dict:
    """A nested table from pyproject.toml, or {} if any level is absent."""
    node: dict = _pyproject()
    for key in path:
        node = node.get(key, {}) or {}
    return node


def _poe_tasks() -> dict[str, str]:
    return _table("tool", "poe", "tasks")


def _gate_job() -> dict:
    """The `quality` job. Identity is test_ci_workflow.py's subject; we want its steps."""
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8")).get("jobs") or {}
    assert GATE_JOB in jobs, (
        f"`{WORKFLOW.name}` has no job with id `{GATE_JOB}` (found: {sorted(jobs)}). Branch "
        f"protection on develop/main requires a check run with that exact name, so this "
        f"rename blocks every merge until the settings are changed to match — which is "
        f"loud, and therefore safe. This module needs the job only to reach its steps; the "
        f"job's identity is pinned in detail by tests/test_ci_workflow.py."
    )
    return jobs[GATE_JOB] or {}


# ---------------------------------------------------------------------------
# Parsing what check.sh hands each tool
# ---------------------------------------------------------------------------


def _shell_tokens(command: str) -> list[str] | None:
    """Tokenize one shell line, or None if it does not stand alone as shell.

    A regex is not good enough here, and the difference is not academic: the first
    version of this file matched the word `pytest` inside check.sh's own error string —
    `print_error "Coverage: could not parse coverage from pytest output"` — and tried to
    run the gate's suite with `output"` as an argument. A tokenizer knows a quoted string
    is one token, so a mention of a tool in prose can never masquerade as an invocation of
    it.

    `punctuation_chars` makes shlex split `|`, `&`, `;`, `<`, `>`, `(`, `)` into tokens of
    their own, which is what lets us see where a command ends.

    None means the line is not self-contained shell — check.sh embeds a multi-line
    `python3 -c "..."` program for the secrets diff, and its interior lines have unbalanced
    quotes. Callers must treat None as "cannot parse", never as "no invocation here": this
    parser must not have the fail-open flaw it exists to hunt.
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:
        return None


def _tool_args_in(command: str, tool: str) -> list[str] | None:
    """The arguments `command` hands `tool`, or None if it does not invoke it.

    Recognises the tool as a COMMAND WORD (`mypy`, `.venv/bin/mypy`, `python -m pytest`),
    never as a substring of prose. Argument collection stops at the first shell operator,
    and at an fd redirection (`2>&1`) — whose leading `2` would otherwise be read as a
    positional path and quietly narrow the very scope this module measures.
    """
    tokens = _shell_tokens(command)
    if tokens is None:
        assert not re.search(rf"\b{re.escape(tool)}\b", command), (
            f"A line of scripts/check.sh mentions `{tool}` but cannot be parsed as shell, so "
            f"this module cannot verify what the gate hands it:\n  {command.strip()}\n\n"
            f"Refusing to assume it is harmless. Simplify the line, or teach _shell_tokens "
            f"about it — do not leave an invocation that the guard cannot read."
        )
        return None

    for index, token in enumerate(tokens):
        if token != tool and not token.endswith(f"/{tool}"):
            continue
        args: list[str] = []
        rest = tokens[index + 1 :]
        for position, argument in enumerate(rest):
            following = rest[position + 1] if position + 1 < len(rest) else ""
            if argument[0] in "|&;<>()" or (argument.isdigit() and following[:1] in {"<", ">"}):
                break
            args.append(argument)
        return args
    return None


def _tool_invocations(tool: str) -> list[list[str]]:
    """Every argv the gate hands `tool` — directly, or through a poe task it calls.

    Resolving `poe <task>` matters: check.sh already delegates lint/format to poe tasks, so
    moving another tool behind one (`uv run poe test`) is a plausible refactor. It would
    put the arguments in pyproject.toml, out of reach of a check.sh-only parse — and a
    guard that a routine refactor silently blinds is worse than no guard.
    """
    tasks = _poe_tasks()
    invocations: list[list[str]] = []
    for line in _check_sh_code().splitlines():
        commands = [line]
        for task_name, task_command in tasks.items():
            if re.search(rf"\bpoe\s+{re.escape(task_name)}\b", line):
                commands.append(task_command)
        for command in commands:
            args = _tool_args_in(command, tool)
            if args is not None:
                invocations.append(args)
    return invocations


def _positional_paths(argv: list[str]) -> set[str]:
    """The path targets in `argv`: everything that is not a flag or a subcommand.

    A flag's *value* can slip through (`--min-confidence 80` contributes `80`). Harmless:
    every caller asks whether the required trees are PRESENT, never whether the set is
    exactly equal to something — so a spurious extra can never turn a red green.
    """
    return {
        token
        for token in argv
        if not token.startswith("-") and token not in TOOL_SUBCOMMANDS and "=" not in token
    }


# ---------------------------------------------------------------------------
# Deriving what is really on disk (the truth to compare against)
# ---------------------------------------------------------------------------


def _tracked_files() -> list[str]:
    """Every file git tracks. Inherits .gitignore, so no venv/worktree/pycache noise."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout — cannot derive the tracked file list")
    return [path for path in result.stdout.split("\0") if path]


def _tracked_test_files() -> set[str]:
    """Every test file git tracks, wherever it lives."""
    return {
        path
        for path in _tracked_files()
        if path.endswith(".py") and Path(path).name.startswith("test_")
    }


def _tracked_package_files() -> set[str]:
    """Every Python file of the package the gate exists to protect."""
    return {
        path for path in _tracked_files() if path.startswith(f"{PACKAGE}/") and path.endswith(".py")
    }


# ---------------------------------------------------------------------------
# 1. The critical checks are still critical
# ---------------------------------------------------------------------------


def _checks_that_call_mark_failed() -> set[str]:
    """Every label passed to `mark_failed` — i.e. every check that can block the gate.

    The function *definition* (`mark_failed() {`) has no quoted argument, so it cannot be
    mistaken for a call.
    """
    return set(re.findall(r'\bmark_failed\s+"([^"]+)"', _check_sh_code()))


def test_every_critical_check_can_still_fail_the_gate() -> None:
    """Each blocking check must still call `mark_failed`. Deleting the call demotes it.

    The cheapest possible way to gut the gate: remove one line and the check still RUNS,
    still prints its red ❌ in the summary — and no longer stops the merge. The badge says
    `quality ✓`, and nobody reads the log of a passing job.
    """
    demoted = sorted(CRITICAL_CHECKS - _checks_that_call_mark_failed())
    assert not demoted, (
        f"These checks no longer call `mark_failed`, so they can no longer fail the quality "
        f"gate: {demoted}.\n"
        f"\n"
        f"`mark_failed` is the ONLY thing that makes a check critical in scripts/check.sh: it "
        f"appends the label to FAILED_CHECKS, and the script exits 1 iff FAILED_CHECKS is "
        f"non-empty. Without the call the check still runs and still prints ❌ in the summary "
        f"— and the job goes GREEN anyway. `quality` is a required status check on develop and "
        f"main, so this does not stop a merge: it waves it through while reporting success.\n"
        f"\n"
        f"If a check is genuinely meant to become advisory (vulture, interrogate and deptry "
        f"already are), that is a deliberate loosening of the gate: argue it in the PR and "
        f"remove the label from CRITICAL_CHECKS here, so a reviewer sees the trade in the diff "
        f"instead of inferring it from a missing line."
    )


def test_mark_failed_still_records_the_failure() -> None:
    """`mark_failed` must still write to FAILED_CHECKS.

    Every `mark_failed "X"` call can stay exactly where it is while the function they all
    call is hollowed out to a no-op — and then NOTHING is critical any more, in a diff that
    touches three lines of a helper nobody rereads.
    """
    body = re.search(r"mark_failed\s*\(\)\s*\{(.*?)\n\}", _check_sh_code(), re.DOTALL)
    assert body, (
        "scripts/check.sh no longer defines `mark_failed()`. That function is the entire "
        "critical/warn-only mechanism of the gate: every blocking check calls it, and the "
        "script's exit code is decided by what it records."
    )
    assert "FAILED_CHECKS=" in body.group(1), (
        "`mark_failed()` no longer assigns FAILED_CHECKS, so every call to it is now a "
        "no-op.\n"
        "\n"
        "This is the widest fail-open edit available in the repo: ruff, format, mypy, radon, "
        "bandit, detect-secrets, pytest AND coverage all keep running, all keep printing ❌ on "
        "failure, and the gate exits 0 for every one of them — because FAILED_CHECKS stays "
        'empty and the final `if [ -n "$FAILED_CHECKS" ]` never fires. The required '
        "`quality` check goes green on a completely broken tree."
    )


def test_a_recorded_failure_still_exits_nonzero() -> None:
    """A non-empty FAILED_CHECKS must still make the script exit 1.

    The last lines of check.sh are the only place the accumulated failures turn into an
    exit code, and an exit code is the only thing GitHub reads. Flip the `1` to a `0` and
    every check above becomes decorative.
    """
    exit_block = re.search(
        r'if\s+\[\s+-n\s+"\$FAILED_CHECKS"\s+\]\s*;\s*then\s*\n\s*exit\s+1\b',
        _check_sh_code(),
    )
    assert exit_block, (
        'scripts/check.sh no longer ends with `if [ -n "$FAILED_CHECKS" ]; then exit 1`.\n'
        "\n"
        "That block IS the gate. Everything above it only ACCUMULATES failures into "
        "FAILED_CHECKS; this is where they become the non-zero exit code that GitHub turns "
        "into a red required check. Without it — or with `exit 0` in its place — check.sh "
        "prints a full red FAILED summary and reports SUCCESS, and `quality` stays green with "
        "the tree on fire.\n"
        "\n"
        "If you restructured the exit logic rather than removed it, update this assertion in "
        "the same PR and say in the description how the new form still exits non-zero on a "
        "recorded failure."
    )


# ---------------------------------------------------------------------------
# 2. The coverage floor has not been lowered
# ---------------------------------------------------------------------------


def _coverage_min_in_check_sh() -> int:
    match = re.search(r"^\s*COVERAGE_MIN=(\d+)\s*$", _check_sh_code(), re.MULTILINE)
    assert match, (
        "scripts/check.sh no longer defines `COVERAGE_MIN=<int>`. The coverage floor is what "
        "stops a PR from deleting tests: without it, a green pytest says nothing about how "
        "much of the package was actually executed."
    )
    return int(match.group(1))


def test_coverage_floor_is_never_lowered() -> None:
    """COVERAGE_MIN is a ratchet: it may go up, never down.

    `COVERAGE_MIN=0` is a one-character diff that turns the coverage check into a
    formality — the gate still prints a Coverage row, still shows a percentage, and passes
    at 3%.
    """
    current = _coverage_min_in_check_sh()
    assert current >= COVERAGE_FLOOR, (
        f"The coverage floor has been LOWERED: scripts/check.sh sets COVERAGE_MIN={current}, "
        f"below the ratchet of {COVERAGE_FLOOR}.\n"
        f"\n"
        f"Lowering the floor does not make the gate report a problem — it makes the gate stop "
        f"reporting one. Coverage still prints, `quality` still goes green, and the suite is "
        f"now allowed to cover {current}% of {PACKAGE} instead of {COVERAGE_FLOOR}%. Every "
        f"test deleted from here on is free.\n"
        f"\n"
        f"The usual reason to want this is a PR whose coverage dipped. Add the tests. If the "
        f"floor genuinely must move down — a large, deliberately untested module landing, say "
        f"— that is a decision, not a tweak: change COVERAGE_FLOOR here in the same PR, and "
        f"the diff will show a reviewer exactly how much verification the repo just gave up.\n"
        f"\n"
        f"Raising the floor is always welcome: raise COVERAGE_MIN, `fail_under` in "
        f"pyproject.toml and COVERAGE_FLOOR here, and this test ratchets with you."
    )


def test_the_coverage_floor_has_a_single_definition() -> None:
    """check.sh's COVERAGE_MIN and pyproject's `fail_under` must agree.

    The floor is enforced TWICE — by pytest-cov (which reads `[tool.coverage.report]
    fail_under`) and again by check.sh's own parse of the TOTAL line. Two numbers that
    "should" match are exactly what CLAUDE.md rule 5 was paid in blood for: lower only one
    and the gate's effective floor becomes whichever is laxer, while the other number still
    reads reassuringly in the diff.
    """
    in_script = _coverage_min_in_check_sh()
    in_pyproject = _table("tool", "coverage", "report").get("fail_under")
    assert in_pyproject is not None, (
        "pyproject.toml no longer sets `[tool.coverage.report] fail_under`, so pytest-cov "
        "stops enforcing the floor and check.sh's TOTAL-line parse is the only thing left "
        "holding it. Two enforcement points became one. Restore it, at the same value as "
        f"COVERAGE_MIN ({in_script})."
    )
    assert in_script == int(in_pyproject), (
        f"The coverage floor is defined twice and the two definitions have DRIFTED: "
        f"scripts/check.sh says COVERAGE_MIN={in_script}, pyproject.toml says "
        f"fail_under={in_pyproject}.\n"
        f"\n"
        f"Both are live: pytest-cov fails the test step at {in_pyproject}%, check.sh fails the "
        f"Coverage row at {in_script}%. The gate's real floor is therefore "
        f"min({in_script}, {in_pyproject}) = {min(in_script, int(in_pyproject))}%, while a "
        f"reviewer reading the other file sees {max(in_script, int(in_pyproject))}% and "
        f"believes it. Set both to the same number."
    )


# ---------------------------------------------------------------------------
# 3. The floor is measured over the WHOLE package
# ---------------------------------------------------------------------------


def _coverage_targets() -> tuple[set[str], bool]:
    """What `--cov` points at, and whether any bare `--cov` makes the question moot.

    A bare `--cov` (no `=value`) measures everything importable, so it cannot narrow
    anything — that is the `measures_everything` flag, and it short-circuits the test
    rather than producing a bogus red.
    """
    targets: set[str] = set()
    measures_everything = False
    for argv in _tool_invocations("pytest"):
        for index, token in enumerate(argv):
            if token.startswith("--cov="):
                targets.add(token.split("=", 1)[1])
            elif token == "--cov":
                following = argv[index + 1] if index + 1 < len(argv) else ""
                if following and not following.startswith("-"):
                    targets.add(following)
                else:
                    measures_everything = True
    return targets, measures_everything


def _is_under(relpath: str, roots: set[str]) -> bool:
    """Is `relpath` inside one of `roots`?

    A root may be spelled as a path (`src/xbrain`) or as the importable package name
    (`xbrain`) — both are valid for `--cov` and for `[tool.coverage.run] source`, and both
    mean the same tree. Accepting either keeps this test from firing on a legitimate
    respelling that changes no behaviour.
    """
    prefixes = {root.rstrip("/") for root in roots} | {f"src/{root.rstrip('/')}" for root in roots}
    return any(relpath == prefix or relpath.startswith(f"{prefix}/") for prefix in prefixes)


def test_coverage_measures_the_whole_package() -> None:
    """The coverage floor must be computed over every file in the package.

    A floor is only as honest as its denominator. `--cov=src/xbrain/cli.py` leaves
    COVERAGE_MIN at a reassuring 78 and applies it to ONE FILE — the other ~40 modules
    stop being measured at all, and the gate reports a percentage that is true of a sliver
    and false of the repo. `omit = ["*/verification/*"]` does the same thing from
    pyproject.toml, one line away from the number it invalidates, and makes coverage go UP.

    This is the attack that makes `test_coverage_floor_is_never_lowered` insufficient on
    its own: you do not need to touch the floor if you can shrink what it is measured over.
    """
    targets, measures_everything = _coverage_targets()
    source = set(_table("tool", "coverage", "run").get("source") or [])
    omit = list(_table("tool", "coverage", "run").get("omit") or [])
    package_files = _tracked_package_files()

    assert targets or measures_everything or source, (
        "Nothing tells coverage what to measure: the gate's pytest call has no `--cov`, and "
        "pyproject.toml declares no `[tool.coverage.run] source`. The Coverage row would "
        "report on whatever happened to get imported — or fail to parse at all."
    )

    def measured(relpath: str) -> bool:
        if targets and not _is_under(relpath, targets):
            return False
        if source and not _is_under(relpath, source):
            return False
        return not any(
            fnmatch.fnmatch(relpath, pattern) or fnmatch.fnmatch(f"/{relpath}", pattern)
            for pattern in omit
        )

    unmeasured = sorted(path for path in package_files if not measured(path))
    assert not unmeasured, (
        f"The coverage floor is computed over only part of the package. These "
        f"{len(unmeasured)} file(s) are in {PACKAGE} but coverage never measures them:\n  "
        + "\n  ".join(unmeasured[:25])
        + ("\n  …" if len(unmeasured) > 25 else "")
        + f"\n\nCOVERAGE_MIN still reads {_coverage_min_in_check_sh()}, and the gate still "
        f"prints a green Coverage row — but the percentage is now true of a subset and false "
        f"of the repo. Deleting every test for an unmeasured module would not move it. This "
        f"is a lower floor than lowering the floor, and it does not look like one in the "
        f"diff.\n\n"
        f"Sources of truth being read here: `--cov=` in the gate's pytest call "
        f"({sorted(targets) or 'none'}), `[tool.coverage.run] source` "
        f"({sorted(source) or 'none'}), `[tool.coverage.run] omit` ({omit or 'none'}).\n\n"
        f"If a module is genuinely untestable, exclude it with `# pragma: no cover` at the "
        f"line that needs it — visibly, where the code is — rather than deleting a whole tree "
        f"from the denominator."
    )


# ---------------------------------------------------------------------------
# 4. pytest runs the whole suite
# ---------------------------------------------------------------------------


def test_the_gate_still_runs_pytest() -> None:
    """The gate must invoke pytest at all. Everything else in this section assumes it."""
    assert _tool_invocations("pytest"), (
        "scripts/check.sh no longer invokes pytest — neither directly nor through a poe task. "
        "The gate would report on lint, types and security while running ZERO tests, and "
        "`quality` would still go green. If the tests moved to their own workflow, that "
        "workflow must become a required status check on develop and main BEFORE this one "
        "stops running them, or there is a window in which nothing runs the suite."
    )


def test_the_gate_declares_no_test_narrowing_flags() -> None:
    """No `-k` / `-m` / `--deselect` / `--ignore` on the gate's pytest, or in `addopts`.

    Companion to the behavioural tests below, which compare what pytest really collects: a
    `-k` that skips individual test functions without emptying any file is invisible to a
    file-set comparison, and this catches it by name. `addopts` is checked too — it is the
    quietest hiding place, because it is nowhere near the pytest command.
    """
    addopts = _table("tool", "pytest", "ini_options").get("addopts", "")
    sources = [("scripts/check.sh", argv) for argv in _tool_invocations("pytest")]
    sources.append(("pyproject.toml [tool.pytest.ini_options] addopts", shlex.split(addopts)))

    offenders = [
        f"{where}: {token}"
        for where, argv in sources
        for token in argv
        if any(token == flag or token.startswith(f"{flag}=") for flag in TEST_NARROWING_FLAGS)
    ]
    assert not offenders, (
        "The quality gate narrows its own test run:\n  "
        + "\n  ".join(offenders)
        + "\n\nEvery one of these makes pytest run FEWER tests while still exiting 0 and still "
        "printing a green summary — the gate reports the same PASS over a smaller suite, which "
        "is the most dangerous shape a broken gate can take. `--deselect` and `-k` are how a "
        "failing test gets 'fixed' at 2am; `--ignore` is how a whole module goes dark.\n\n"
        "If a test is genuinely broken, fix it — or delete it in a diff a reviewer can see. Do "
        "not hide it behind a flag in the gate that is supposed to be running it."
    )


def _collected_by(argv: list[str]) -> tuple[set[str], int]:
    """Ask pytest itself what it collects with `argv`. Returns (test files, deselected).

    Behaviour, not spelling: `argv` is the gate's REAL pytest arguments, so anything that
    narrows collection — a flag, a positional path, a `testpaths` in pyproject.toml, even a
    conftest hook that deselects — shows up here as a smaller answer. Nothing about the
    expected result is hardcoded.

    Two classes of argument are dropped, neither of which can change WHICH tests are
    collected:

    * `--cov*` — orthogonal to collection (nothing executes under `--collect-only`), and
      starting a second coverage session inside the one the gate is already running would
      be noise, not signal. What coverage measures has its own test above.
    * verbosity (`-q`, `-v`, …) — because we impose our own. The gate passes `-q`, and `-q`
      plus our `-q` is `-qq`, at which level pytest stops printing node ids and prints only
      a count. That silently returned an empty collection and made this test fail for a
      reason that had nothing to do with the gate.
    """
    noise = re.compile(r"^(--cov.*|-q+|--quiet|-v+|--verbose)$")
    args = [argument for argument in argv if not noise.match(argument)]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *args, "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"The quality gate's own pytest arguments cannot even COLLECT the suite:\n"
        f"  pytest {' '.join(args)}\n\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    files = {line.split("::", 1)[0] for line in result.stdout.splitlines() if "::" in line}
    deselected = re.search(r"\((\d+) deselected\)", result.stdout)
    return files, int(deselected.group(1)) if deselected else 0


def test_the_gate_collects_every_test_file_in_the_repo() -> None:
    """Pytest, run with the gate's real arguments, must collect every test file on disk.

    The load-bearing test of this module, and the only one that cannot be satisfied by
    restating a constant. The left side is what pytest DOES (asked via `--collect-only`,
    using the gate's own argv); the right side is what EXISTS (asked via `git ls-files`).

    Red for every mechanism that hides a test file, not only the ones we thought of:
    `--ignore`, `--ignore-glob`, a positional path, a narrowed `testpaths`, a `-k` that
    empties a file, a conftest that skips one. Green through any change that keeps running
    them all — reformat the command, move it to `poe test`, add a directory under `tests/`.
    """
    on_disk = _tracked_test_files()
    collected: set[str] = set()
    for argv in _tool_invocations("pytest"):
        collected |= _collected_by(argv)[0]

    missing = sorted(on_disk - collected)
    assert not missing, (
        "The quality gate never runs these test files, but they are in the repo:\n  "
        + "\n  ".join(missing)
        + "\n\nThe gate does not go red when they break — it does not open them. Every "
        "assertion in them is now decorative, and the `quality` check reports the same green "
        "over a suite that verifies strictly less. tests/test_guardrail_contract.py is the "
        "sharpest example of what that costs: it is itself a guardrail, so hiding it removes a "
        "guard AND the alarm that the guard was missing, in one line.\n\n"
        "Usual causes: an `--ignore=` / `--ignore-glob=` added to the pytest call in "
        "scripts/check.sh; a positional path narrowing it to one directory; a `testpaths` in "
        "pyproject.toml that no longer covers where the tests actually live.\n\n"
        "A test too slow or too flaky to run in the gate is a test to FIX or DELETE, visibly. "
        "It is not a test to leave in the tree looking like coverage it no longer provides."
    )


def test_the_gate_deselects_nothing() -> None:
    """The gate's pytest run must deselect zero tests.

    The other half of the file-set test: `-k "not slow"` or `--deselect
    tests/test_x.py::test_y` can remove hundreds of individual tests while every file still
    appears in the collection. pytest reports that as `1433/1587 tests collected (154
    deselected)` — a line nobody reads in a green log. Here it is an assertion.
    """
    for argv in _tool_invocations("pytest"):
        _, deselected = _collected_by(argv)
        assert deselected == 0, (
            f"The quality gate deselects {deselected} test(s) from its own run:\n"
            f"  pytest {' '.join(argv)}\n"
            f"\n"
            f"Those tests are in the repo, they are collected, and they are then thrown away "
            f"before they execute — so they cannot fail, cannot go red, and cannot stop a "
            f"merge. The gate prints the same green summary over {deselected} fewer assertions "
            f"than the tree claims to have.\n"
            f"\n"
            f"This is almost always a `-k`, a `-m` or a `--deselect` added to make a failing "
            f"test go away. Fix the test, or delete it where a reviewer can see it."
        )


# ---------------------------------------------------------------------------
# 5. The critical tools still look at the whole package
# ---------------------------------------------------------------------------


def test_the_critical_tools_still_scan_the_whole_package() -> None:
    """mypy, bandit, radon and detect-secrets must each be pointed at their whole tree.

    A tool aimed at one file is a tool that passes. `mypy src/xbrain/cli.py` type-checks
    1 of ~40 modules and prints "Success"; `detect-secrets scan src/xbrain` stops reading
    `tests/` and `scripts/`, where a committed key is just as real. In both cases the gate
    keeps its ✅ row, keeps its CRITICAL severity, keeps exiting 0 — and has stopped looking.

    Every invocation is checked, not merely one: check.sh calls radon twice (the grade
    check and the average), so a test satisfied by "some call covers the package" would be
    silently defeated by narrowing the one that actually gates.

    (ruff and ruff-format are deliberately absent — tests/test_quality_gate_scope.py owns
    their scope, and asks ruff itself which files it opens.)
    """
    for tool, required in sorted(CRITICAL_TOOL_SCOPES.items()):
        invocations = _tool_invocations(tool)
        assert invocations, (
            f"scripts/check.sh no longer invokes `{tool}`, so that check cannot fail — while "
            f"the gate's summary and its CRITICAL severity table still list it. A check that "
            f"does not run is a check that always passes."
        )
        for argv in invocations:
            paths = _positional_paths(argv)
            missing = [tree for tree in required if tree not in paths]
            assert not missing, (
                f"`{tool}` is no longer pointed at {missing}:\n"
                f"  {tool} {' '.join(argv)}\n"
                f"\n"
                f"It must examine {list(required)}. Narrowing a critical tool's scope does not "
                f"make the gate complain — it makes the gate stop looking, while the ✅ row, "
                f"the CRITICAL severity and the exit code all stay exactly as they were. This "
                f"is the same defect that put tests/test_quality_gate_scope.py in the repo: "
                f"ruff was linting `src tests` and never opened `scripts/`, and the gate "
                f"printed ALL CRITICAL CHECKS PASSED the whole time.\n"
                f"\n"
                f"If a file genuinely cannot satisfy the tool, suppress it AT THE FILE — a "
                f"`# type: ignore`, a `# nosec` — where a reviewer sees the exemption, instead "
                f"of removing the tree from the tool's sight."
            )


def test_the_critical_tools_are_not_blinded_from_pyproject() -> None:
    """The same narrowing, hidden one file away from the command that looks correct.

    `check.sh` can keep reading `mypy src/xbrain`, word for word, while pyproject.toml
    quietly says `ignore_errors = true` for half the package. The command in the diff looks
    untouched; the tool reports success; the gate is blind. Same for bandit's
    `exclude_dirs`, and for its severity threshold: `-lll` restricts it to HIGH-severity
    findings only, so every MEDIUM one it used to block now passes.
    """
    overrides = _pyproject().get("tool", {}).get("mypy", {}).get("overrides", []) or []
    blinded = [
        override.get("module")
        for override in overrides
        if override.get("ignore_errors") or override.get("follow_imports") == "skip"
    ]
    assert not blinded, (
        f"pyproject.toml turns mypy OFF for {blinded} via `[[tool.mypy.overrides]]`.\n"
        f"\n"
        f"`check.sh` still runs `mypy {PACKAGE}` and mypy still prints Success — over a "
        f"package it has been told to stop reading. The Mypy row stays ✅ and stays CRITICAL. "
        f"Fix the types, or add a `# type: ignore[code]` at the line that needs it, where the "
        f"exemption is visible next to the code it excuses."
    )

    excluded = _table("tool", "bandit").get("exclude_dirs") or []
    package_exclusions = [pattern for pattern in excluded if PACKAGE in pattern]
    assert not package_exclusions, (
        f"pyproject.toml excludes {package_exclusions} from bandit via `exclude_dirs`.\n"
        f"\n"
        f"The security scan of the package the gate exists to protect now skips part of it, "
        f"while `check.sh` still reads `bandit -r {PACKAGE}` and the Bandit row still reports "
        f"✅. Suppress the individual finding with `# nosec` and a reason, at the line."
    )

    for argv in _tool_invocations("bandit"):
        raised = [token for token in argv if re.fullmatch(r"-l{3,}", token)]
        assert not raised, (
            f"bandit's severity threshold has been raised to {raised}:\n"
            f"  bandit {' '.join(argv)}\n"
            f"\n"
            f"`-lll` reports HIGH-severity findings only. Every MEDIUM finding the gate used to "
            f"block — the level it has run at all along — now passes silently. The check still "
            f"runs, still says ✅, and has quietly stopped enforcing most of what it enforced."
        )


# ---------------------------------------------------------------------------
# 6. The workflow really runs the gate, and really lets it fail
# ---------------------------------------------------------------------------


def _gate_invocations_in_workflow() -> list[str]:
    """The `run:` commands of the `quality` job that actually execute the gate.

    Accepts the script by path, or via any poe task that runs it — derived from
    pyproject.toml, so `uv run poe check` counts without this test knowing the task's name.
    `- run: echo ok` matches nothing.
    """
    gate_tokens = [GATE_SCRIPT] + [
        f"poe {name}" for name, command in _poe_tasks().items() if GATE_SCRIPT in command
    ]
    return [
        run
        for step in (_gate_job().get("steps") or [])
        if (run := str(step.get("run", "")))
        if any(token in run for token in gate_tokens)
    ]


def test_the_workflow_runs_the_gate_and_lets_it_fail_the_job() -> None:
    """The `quality` job must run check.sh, in a form whose exit code can fail the job.

    Two fail-open edits, one property. A job that runs `- run: echo ok` and a job that runs
    `bash scripts/check.sh || true` are indistinguishable from outside: both publish a check
    run named `quality`, both are green, both verified nothing. The second is worse, because
    its log is full of reassuring ✅ output.
    """
    invocations = _gate_invocations_in_workflow()
    assert invocations, (
        f"No step in the `{GATE_JOB}` job runs `{GATE_SCRIPT}`.\n"
        f"\n"
        f"`{GATE_JOB}` is a REQUIRED status check on develop and main, with admins enforced. A "
        f"job with these steps still publishes that check, and still publishes it GREEN — "
        f"having run none of the 11 quality checks. A required check that cannot fail is worse "
        f"than no required check at all, because it is trusted: every merge from here on is "
        f"waved through by a green light that means nothing.\n"
        f"\n"
        f"Steps currently in the job: "
        f"{[s.get('name') or s.get('uses') or s.get('run') for s in _gate_job().get('steps') or []]}"
    )

    for run in invocations:
        swallowers = [fragment for fragment in EXIT_STATUS_SWALLOWERS if fragment in run]
        assert not swallowers, (
            f"The step running the quality gate discards its exit status ({swallowers}):\n"
            f"  {run.strip()}\n"
            f"\n"
            f"check.sh exits 1 when a critical check fails, and that exit code is the ONLY thing "
            f"GitHub reads. Swallow it and the gate still runs, still prints its red FAILED "
            f"summary into the log — and the job succeeds, so the required `{GATE_JOB}` check "
            f"goes green over failing tests, type errors or leaked secrets. The evidence is "
            f"right there in a log nobody opens, because the check passed."
        )


def test_no_step_in_the_gate_job_may_continue_on_error() -> None:
    """`continue-on-error` anywhere inside the `quality` job. The step case is LETHAL.

    Measured against the live API, with a genuinely failing gate:

        jobs.quality.continue-on-error: true          → check run FAILURE, PR BLOCKED
        jobs.quality.steps[gate].continue-on-error: 1 → check run SUCCESS, PR CLEAN

    The step form is the purest fail-open edit in the repository. Nothing else changes:
    check.sh runs all 11 checks, exits 1, prints the red FAILED summary. GitHub then reports
    the step `success`, the job `success` and the check run `SUCCESS`. The failure exists
    nowhere a machine can read it — only in the raw log text. Branch protection,
    `enforce_admins` and every test in this repo are bypassed by two words of YAML that read
    like a courtesy.

    The job form is harmless (the check still reports FAILURE, so the merge is blocked) and
    is banned anyway: the ONLY thing separating it from the lethal form is indentation, and
    "we meant the safe one" is not a property a reviewer can verify at a glance.

    Every step is walked, not just the gate's — a step that is decorative today (checkout,
    dependency install) is load-bearing tomorrow, and a `continue-on-error` left on it is a
    trap armed in advance.

    HONEST LIMIT — this assertion cannot fail closed against the attack it names.
    If `continue-on-error` is placed on the step that runs check.sh, this test goes red
    *inside* that step, and the same line tells GitHub to ignore the step's exit code. The
    red reaches the log and the diff; it does not reach the merge button. See the module
    docstring: closing that requires a second required check in a separate job, which is
    infrastructure, not a test.
    """
    job = _gate_job()
    offenders = []
    if job.get("continue-on-error"):
        offenders.append(f"the `{GATE_JOB}` job itself (fail-closed, but still forbidden)")
    for index, step in enumerate(job.get("steps") or []):
        if step.get("continue-on-error"):
            label = step.get("name") or step.get("run") or step.get("uses") or f"#{index}"
            offenders.append(f"step `{str(label).strip()}` (FAIL-OPEN — this is the lethal one)")

    assert not offenders, (
        "`continue-on-error` is set on: " + "; ".join(offenders) + ".\n"
        "\n"
        "On a STEP this is the quietest way to disable a required status check that exists. "
        "check.sh still runs every check, still exits 1, still prints the red FAILED summary — "
        "and GitHub reports the step, the job and the check run as SUCCESS, and the PR as "
        "mergeable. Verified against the live API. The failure is visible nowhere except the "
        "raw log.\n"
        "\n"
        "There is no legitimate use of `continue-on-error` in a gating job. If one specific "
        "check is too noisy to block, make THAT CHECK warn-only inside check.sh — drop its "
        "`mark_failed` call and its entry in CRITICAL_CHECKS above — visibly, in a diff a "
        "reviewer can weigh, instead of blinding the entire gate with one line of YAML."
    )
