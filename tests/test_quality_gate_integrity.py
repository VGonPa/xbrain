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

Two axes, not one: SCOPE and SENSITIVITY
----------------------------------------
Guarding what a tool LOOKS AT is half the job, and the weaker half. A tool aimed at the
whole package but configured to report nothing is, from the gate's point of view,
indistinguishable from a tool that found nothing:

    [tool.ruff.lint.per-file-ignores]
    "*" = ["ALL"]                  ->  ruff: "All checks passed!" on a file with unused imports

    [tool.mypy]
    ignore_errors = true           ->  mypy: "Success: no issues found in 48 source files"

    [tool.coverage.report]
    exclude_lines = ["."]          ->  coverage: TOTAL 0 statements, 100%

Each is two lines of `pyproject.toml`. None touches `check.sh`. None touches the workflow.
None narrows a directory — the targets still read `src tests scripts` — so every scope
assertion in this module AND in `test_quality_gate_scope.py` stays green while the tool is
switched off. The scope was guarded and the sensitivity was wide open; the coverage
denominator (`--cov=src/xbrain/cli.py`) is the same bug one tool over.

Enumerating config keys is a losing game against a surface this size. So each critical tool
is asked the only question that settles it — *given this repo's configuration exactly as the
gate loads it, do you still report an obviously broken file?* — with a canary built to be
unmissable for that tool and nothing else. The static key-checks that accompany the canaries
exist for their failure MESSAGES (they name the offending line); the canaries are what
actually close the vector.

Two lessons paid for while writing this, both from attacks that came back GREEN:

* **Guard the tool's real configuration, not a clean one.** `bandit -r src/xbrain -ll -q` —
  the way check.sh invokes it — does NOT read `[tool.bandit]` from pyproject (measured). An
  earlier version of this module asserted that `exclude_dirs` must not touch the package,
  and would have gone red, claiming "the security scan now skips part of the package", on an
  edit that does *nothing at all*. A guard that lies about the damage is worse than no guard:
  it teaches the next reader that the guard is noise. That assertion is now conditional on
  the config actually being live.
* **Patterns are not all globs.** ruff's `per-file-ignores` are globs, mypy's `exclude` is a
  REGEX, bandit's `exclude_dirs` are directory prefixes. A matcher that knew only `fnmatch`
  let two real attacks through — `fnmatch("src/xbrain/executors/api.py",
  "src/xbrain/executors")` is False.

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
| **SENSITIVITY** — `per-file-ignores "*" = ["ALL"]`, `select = []`, `ignore = ["ALL"]` | `test_ruff_still_reports_an_obvious_violation` |
| `[tool.ruff.format] exclude` → linter still sees all; formatter sees 65 of 113 | `test_ruff_format_examines_every_file_ruff_check_does` |
| `[tool.mypy] ignore_errors` / `disable_error_code` / `follow_imports = "skip"` | `test_mypy_still_reports_an_obvious_type_error` (+ the key check) |
| `mypy exclude` (a REGEX; the canary cannot see it — explicit files bypass it) | `test_mypy_is_not_globally_silenced` |
| bandit `-lll` / `-iii` / `--skip` / `-c pyproject.toml` + `skips` | `test_bandit_still_reports_an_obvious_finding` |
| `exclude_lines = ["."]` → 100% coverage over ZERO statements | `test_coverage_excludes_no_ordinary_code` |
| `coverage include = [one file]` → a whitelist; every `omit` still reads innocent | `test_coverage_measures_the_whole_package` |
| `detect-secrets --exclude-files` → all three trees still named, none scanned | `test_the_detect_secrets_scan_is_not_narrowed_by_flags` |
| radon's grade class `[D-F]` → `[F]` → grade-D complexity ships, `mark_failed` intact | `test_radon_still_fails_on_every_forbidden_complexity_grade` |
| `# ruff: noqa` / `# mypy: ignore-errors` at the top of a source file | `test_no_package_file_blanket_suppresses_a_critical_tool` |

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
most dangerous edit in the threat model.

**A second required check in a separate job does NOT fix this.** An earlier draft
of this docstring claimed it did — that a step-level `continue-on-error` in the
gate job cannot mask a failure in a *different* job, so a second job terminates
the regress. That is wrong, and the error is worth recording because it is the
seductive one. The attacker writes **one diff**, and nothing stops that diff from
adding `continue-on-error: true` to the second job's steps as well. Both jobs then
report `success` while both fail. **A second job is defeated by the same attack
applied twice.** Multiplication does not terminate a regress; it only raises its
price by one line of YAML per copy.

The reason is structural, and it generalises past `continue-on-error`: a PR's
`pull_request` run evaluates the **HEAD** version of the workflow. **Anything
defined in a file the PR can edit is a guard the PR can edit.** No arrangement of
in-repo jobs, workflows or tests escapes that, because they are all in-repo. The
only surfaces a PR cannot touch are server-side: branch-protection settings,
rulesets, required workflows, merge queue — and on this repo those are
unavailable (org/Enterprise-gated), while `pull_request_target` is declined as an
RCE surface. The one exception is CODEOWNERS, which GitHub reads from the BASE
branch — so a PR cannot edit its way out of a required code-owner review. But it
needs a *second human* to approve, which is not a mechanism. It is a person.

Which is the whole point, and the reason the principle below is stated the way it
is rather than as a recipe:

> The escape from "nothing catches itself" is not another catcher. It is making
> the *absence* of the catcher fail closed. Where a guard's removal blocks the
> merge, the regress terminates. Where it only removes a test, it does not — and
> that residue is social, not mechanical.

Branch protection sits on the terminating side. This file cannot, and neither can
any file next to it.

**3. What is deliberately NOT guarded here**, so nobody mistakes silence for
coverage:
* `.secrets.baseline` regenerated to allowlist a real committed secret. The
  baseline is the audit trail by construction — the entry appears in the diff —
  but no test distinguishes an audited false positive from a laundered key.
* `# noqa: E501` / `# type: ignore[code]` / `# nosec` with a code. Allowed on
  purpose: a named exemption at the line is a reviewable decision. Only the BARE,
  uncoded, whole-file forms are rejected.
* A `per-file-ignores` entry that blinds ruff for a TEST file only. It does not
  reach the package, so no canary fires. It weakens the lint of the test suite.
* A `conftest.py` hook that drops individual tests without calling
  `pytest_deselected`. Whole files going dark is caught; a hook that quietly
  removes a handful of items is not.

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
import tempfile
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
# `--co` is the alias for `--collect-only` and must be listed separately — a ban that
# names only the long form is a ban on the spelling, not on the behaviour.
TEST_NARROWING_FLAGS = (
    "-k",
    "-m",
    "--deselect",
    "--ignore",
    "--ignore-glob",
    "--collect-only",
    "--co",
)

# Flags that shrink the detect-secrets scan without touching its positional paths, so
# `test_the_critical_tools_still_scan_the_whole_package` would still see three trees.
SECRETS_NARROWING_FLAGS = ("--exclude-files", "--exclude-lines", "--exclude-secrets", "--word-list")

# ---------------------------------------------------------------------------
# CANARIES — the SENSITIVITY axis
#
# Guarding a tool's SCOPE is only half the job, and the weaker half. A tool aimed at the
# whole package but configured to report nothing is indistinguishable, from the gate's
# point of view, from a tool that found nothing:
#
#     [tool.ruff.lint.per-file-ignores]
#     "*" = ["ALL"]                      # ruff: "All checks passed!" — on a broken file
#
#     [tool.mypy]
#     ignore_errors = true               # mypy: "Success: no issues found in 48 files"
#
#     [tool.coverage.report]
#     exclude_lines = ["."]              # coverage: TOTAL 0 statements, 100%
#
# None of these touch check.sh. None narrow a directory. Every scope assertion above stays
# green. This is the same defect as the coverage-denominator one, one tool over: the scope
# is guarded and the sensitivity is wide open.
#
# The answer is not to enumerate config keys — that is a losing game against a config
# surface this large. It is to ask each tool the only question that matters: *given this
# repo's configuration exactly as the gate loads it, do you still report an obviously
# broken file?* Each canary below is unambiguously broken for exactly one tool. If the tool
# stays silent, it has been switched off, however that was spelled.
#
# The static checks that accompany them exist only for their FAILURE MESSAGES — they name
# the offending key. The canaries are what actually close the vector.
# ---------------------------------------------------------------------------

# A path inside the package that no real file occupies. Wildcard config (`"*"`, `src/*`)
# matches it; a legitimate, narrow per-file rule for some other file does not — which is
# exactly the discrimination these tests need.
CANARY_MODULE = f"{PACKAGE}/_gate_canary.py"

# F401 (unused import) + E401 (multiple imports on one line) + E711 (comparison to None):
# three rules spanning ruff's default E4/E7/F selection, so narrowing `select` to any one
# of those families still leaves the canary screaming.
RUFF_LINT_CANARY = "import os, sys\n\nx = 1 == None\n"
RUFF_FORMAT_CANARY = "x   =   1\ndef  f( a ):\n  return  a\n"

# Returning an `int` from a `-> str` function. If mypy cannot see this, mypy sees nothing.
MYPY_CANARY = "def gate_canary(value: int) -> str:\n    return value\n"

# B108 hardcoded_tmp_directory: MEDIUM severity, MEDIUM confidence — chosen deliberately.
# A HIGH-severity canary would still be reported under `-lll`, so the gate could be
# desensitised from MEDIUM to HIGH-only and the canary would never notice. A MEDIUM one
# goes silent the moment the threshold is raised, which is the whole point.
BANDIT_CANARY = 'GATE_CANARY_TMP = "/tmp/gate_canary"\n'

# Perfectly ordinary code. No coverage exclusion pattern may match any of it: a pattern
# that does is not excluding a special case, it is excluding the program.
ORDINARY_CODE_LINES = (
    "x = 1",
    "def f():",
    "    return value",
    "class Thing:",
    "    self.name = name",
)

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
    run, report = _table("tool", "coverage", "run"), _table("tool", "coverage", "report")
    targets, measures_everything = _coverage_targets()
    source = set(run.get("source") or []) | set(run.get("source_pkgs") or [])

    # BOTH tables can narrow, and BOTH are live (pytest-cov reads them from pyproject).
    # `omit` was the only one guarded before; `include` is the sharper weapon, because it is
    # a whitelist — `include = ["src/xbrain/cli.py"]` drops the other 47 modules in one line
    # while every `omit` still reads as innocent.
    omit = list(run.get("omit") or []) + list(report.get("omit") or [])
    include = list(run.get("include") or []) + list(report.get("include") or [])
    package_files = _tracked_package_files()

    assert targets or measures_everything or source, (
        "Nothing tells coverage what to measure: the gate's pytest call has no `--cov`, and "
        "pyproject.toml declares no `[tool.coverage.run] source`. The Coverage row would "
        "report on whatever happened to get imported — or fail to parse at all."
    )

    def _matches(relpath: str, patterns: list[str]) -> bool:
        return any(
            fnmatch.fnmatch(relpath, pattern) or fnmatch.fnmatch(f"/{relpath}", pattern)
            for pattern in patterns
        )

    def measured(relpath: str) -> bool:
        if targets and not _is_under(relpath, targets):
            return False
        if source and not _is_under(relpath, source):
            return False
        if include and not _matches(relpath, include):
            return False
        return not _matches(relpath, omit)

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
        f"({sorted(targets) or 'none'}), coverage `source` ({sorted(source) or 'none'}), "
        f"`omit` ({omit or 'none'}), `include` ({include or 'none'}) — the last two read from "
        f"BOTH [tool.coverage.run] and [tool.coverage.report], because both are live.\n\n"
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


def _run_tool(tool: str, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run a quality tool from the repo root, so it loads this repo's real configuration."""
    pytest.importorskip(tool.replace("-", "_"), reason=f"{tool} not installed — install dev deps")
    return subprocess.run(
        [sys.executable, "-m", tool, *args],
        cwd=REPO_ROOT,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def test_ruff_still_reports_an_obvious_violation() -> None:
    """Ruff, configured exactly as this repo configures it, must still fail a broken file.

    THE cheapest attack found on this gate, and nothing caught it before this test:

        [tool.ruff.lint.per-file-ignores]
        "*" = ["ALL"]

    Two lines in pyproject.toml. It does not touch check.sh. It does not touch the
    workflow. It does not narrow a single directory — the targets stay `src tests scripts`,
    so every scope assertion in this module and in test_quality_gate_scope.py stays green.
    Ruff then prints `All checks passed!` over a file containing unused imports, and the
    gate reports ✅ Ruff (linting) PASS on a tool that has been switched off.

    Asking ruff itself is the only defence that generalises. The canary is fed on stdin
    under a filename INSIDE the package, so ruff resolves it against this repo's real
    config: wildcard per-file-ignores, `ignore = ["ALL"]`, `select = []`, an `exclude` that
    swallows the package — all of them silence it, and it does not care which was used.

    A legitimate narrow suppression (per-file-ignores for one real test file) does not
    match the canary's path, so this stays green. A guard that fired on those would be
    noise, and noise gets deleted.
    """
    lint = _run_tool("ruff", ["check", "--stdin-filename", CANARY_MODULE, "-"], RUFF_LINT_CANARY)
    assert lint.returncode != 0, (
        f"RUFF IS BLIND. Handed a file with an unused import, two imports on one line and a "
        f"`== None` comparison, ruff — loaded with this repo's configuration — reported "
        f"nothing:\n\n  {RUFF_LINT_CANARY!r}\n  -> {lint.stdout.strip() or '(silence)'}\n"
        f"\n"
        f"The gate still runs ruff. Its targets still say `src tests scripts`. The summary "
        f"still prints ✅ Ruff (linting) PASS. It is checking nothing, and every scope test "
        f"in this repo is satisfied, because the scope is not what was broken — the "
        f"SENSITIVITY was.\n"
        f"\n"
        f"Look in pyproject.toml under [tool.ruff]: a `per-file-ignores` entry whose pattern "
        f'matches everything, `ignore = ["ALL"]`, an empty `select`, or an `exclude` that '
        f"swallows the package. Any one of them does this, and none of them look like an "
        f"attack in a diff.\n"
        f"\n"
        f"If a specific rule is genuinely unwanted, disable THAT RULE by name. If one file "
        f"needs an exemption, give it a `# noqa: <code>` at the line, where the reviewer can "
        f"see what is being excused."
    )


def _ruff_argv(subcommand: str) -> list[str]:
    """The gate's own `ruff check` / `ruff format` argv, resolved through the poe tasks."""
    for argv in _tool_invocations("ruff"):
        if argv[:1] == [subcommand]:
            return argv
    raise AssertionError(
        f"scripts/check.sh no longer runs `ruff {subcommand}` — neither directly nor via a poe "
        f"task. That check cannot fail, while the gate's summary still lists it as CRITICAL."
    )


def test_ruff_format_examines_every_file_ruff_check_does() -> None:
    """`ruff format` must open exactly as many files as `ruff check` does.

    The formatter has its own exclusion list. `[tool.ruff.format] exclude = ["src/xbrain/*"]`
    leaves `ruff check` — and therefore test_quality_gate_scope.py's `--show-files` audit —
    completely untouched, and silently drops the package from the FORMAT check. Measured:
    the gate goes from formatting 113 files to formatting 65, and `ruff format --check src
    tests scripts` exits 0. The Format row prints ✅. Formatting is no longer enforced.

    The stdin canary above cannot see this (ruff does not apply `format.exclude` to a file
    handed to it on stdin, with or without `--force-exclude` — measured), so this test asks
    the question the other way round: count the files each mode actually opens, and demand
    they agree. Both numbers come from ruff; neither is hardcoded.

    A TOP-LEVEL `exclude` shrinks both counts equally and slips past this — that one is
    caught behaviourally by test_quality_gate_scope.py, which compares ruff's file list
    against `git ls-files`. The two tests are complements: it owns "ruff opens every file",
    this owns "the formatter opens the same files the linter does".
    """
    linted = _run_tool("ruff", [*_ruff_argv("check"), "--show-files"])
    assert linted.returncode == 0, f"ruff --show-files failed:\n{linted.stderr}"
    lint_count = len([line for line in linted.stdout.splitlines() if line.strip()])

    formatted = _run_tool("ruff", _ruff_argv("format"))
    # "113 files already formatted" / "1 file would be reformatted, 112 files already formatted"
    format_count = sum(
        int(number)
        for number in re.findall(
            r"(\d+) files? (?:would be reformatted|already formatted|reformatted)",
            formatted.stdout,
        )
    )

    assert format_count == lint_count, (
        f"`ruff format` opens {format_count} files; `ruff check` opens {lint_count}. The "
        f"formatter is examining {lint_count - format_count} fewer file(s) than the linter.\n"
        f"\n"
        f"Almost certainly `[tool.ruff.format] exclude` in pyproject.toml. It is invisible to "
        f"every other guard in this repo: `ruff check` still opens everything, the poe targets "
        f"still read `src tests scripts`, and test_quality_gate_scope.py — which audits ruff's "
        f"file list — still passes, because it audits the LINTER. Meanwhile `ruff format "
        f"--check` exits 0 over unformatted code and the gate prints ✅ Ruff (format) PASS.\n"
        f"\n"
        f"The formatter and the linter must examine the same tree. If they legitimately must "
        f"not, that is a deliberate weakening of the gate and belongs in a PR that argues it."
    )


def _package_matching_patterns(patterns: list[str]) -> list[str]:
    """Which of `patterns` reach a real file of the package?

    Deliberately permissive, because the three tools that take exclusion patterns disagree
    about what a pattern IS: ruff's `per-file-ignores` are globs, mypy's `exclude` is a
    REGEX, and bandit's `exclude_dirs` are directory prefixes. A matcher that only knew
    fnmatch — as the first version of this helper did — missed both of the others:
    `fnmatch("src/xbrain/executors/api.py", "src/xbrain/executors")` is False, and so is
    `fnmatch(..., "src/xbrain/executors/.*")`. Two real attacks walked straight through.

    So a pattern "reaches the package" if it does so under ANY of the three semantics. The
    cost of being permissive is a theoretical false positive on a pattern that accidentally
    regex-matches; the cost of being precise-but-wrong is a guard that says PASS while the
    tool is blind. That trade is not close.
    """
    package_files = _tracked_package_files() | {CANARY_MODULE}
    reaching = []
    for pattern in patterns:
        text = str(pattern)
        try:
            regex = re.compile(text)
        except re.error:
            regex = None
        if any(
            fnmatch.fnmatch(path, text)
            or fnmatch.fnmatch(path, f"{text.rstrip('/')}/*")
            or path.startswith(f"{text.rstrip('/')}/")
            or (regex is not None and regex.search(path))
            for path in package_files
        ):
            reaching.append(text)
    return reaching


def test_ruff_is_not_blinded_by_wildcard_configuration() -> None:
    """The named-key half of the ruff canary. Same vector, a message that points at the line.

    The canary above already goes red for all of this. This test exists because
    "RUFF IS BLIND" is a worse thing to read at 2am than "your per-file-ignores entry `*`
    disables every rule for every file in the package".

    Deliberately narrow: it fires only on the patterns that blind the PACKAGE, and only when
    the codes include `ALL`. `"tests/test_cli.py" = ["E501"]` is a legitimate suppression and
    stays green — a guard people are trained to edit is not a guard.
    """
    lint = _table("tool", "ruff", "lint")
    top = _table("tool", "ruff")

    select = lint.get("select", top.get("select"))
    assert select is None or select, (
        "`[tool.ruff.lint] select = []` selects NO rules. Ruff runs, opens every file, and "
        "reports nothing it could possibly find, because nothing is enabled. The Ruff row "
        "stays ✅ and stays CRITICAL. Omit `select` to keep ruff's defaults (E4, E7, E9, F)."
    )

    ignored = list(lint.get("ignore") or top.get("ignore") or [])
    assert "ALL" not in ignored, (
        '`ignore = ["ALL"]` under [tool.ruff.lint] switches off every rule in the ruleset. '
        "Ruff still runs on every file in `src tests scripts` and still reports `All checks "
        "passed!`, whatever is in them. Ignore the specific rules you mean, by code."
    )

    per_file: dict[str, list[str]] = {
        **(top.get("per-file-ignores") or {}),
        **(lint.get("per-file-ignores") or {}),
        **(lint.get("extend-per-file-ignores") or {}),
    }
    blinding = {
        pattern: codes
        for pattern, codes in per_file.items()
        if "ALL" in (codes or []) and _package_matching_patterns([pattern])
    }
    assert not blinding, (
        f"These `per-file-ignores` patterns disable EVERY ruff rule for files inside "
        f"{PACKAGE}: {blinding}.\n"
        f"\n"
        f'`"*" = ["ALL"]` is the whole attack. It is two lines, it touches neither '
        f"check.sh nor the workflow, it narrows no directory — so every scope assertion in "
        f"this repo stays green — and ruff reports `All checks passed!` on a file with unused "
        f"imports. The gate keeps printing ✅ Ruff (linting) PASS over a tool that has been "
        f"turned off.\n"
        f"\n"
        f"Per-file-ignores are for narrow, named exemptions on specific files: "
        f'`"tests/conftest.py" = ["E402"]`. A pattern that matches the package plus the '
        f"code `ALL` is not an exemption, it is a mute button."
    )


def test_mypy_still_reports_an_obvious_type_error() -> None:
    """Mypy, configured exactly as this repo configures it, must still fail a broken file.

    `[tool.mypy] ignore_errors = true` — three words — makes mypy print
    `Success: no issues found in 48 source files`. The command in check.sh still reads
    `mypy src/xbrain`, unchanged and reassuring, in a diff that does not mention it.

    `--shadow-file` is what makes this honest: mypy type-checks the canary's CONTENT while
    believing it lives at a REAL module path in the package, so every per-module override,
    every `disable_error_code`, every global switch resolves exactly as it does in the real
    run. A canary in a temp directory would inherit none of that and would pass while the
    package was blind — a test that is green from birth, which is precisely the failure mode
    this repo keeps paying for.
    """
    victim = sorted(path for path in _tracked_package_files() if not path.endswith("__init__.py"))
    if not victim:
        pytest.skip("no package module to shadow")

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(MYPY_CANARY)
        canary = handle.name
    try:
        result = _run_tool("mypy", [victim[0], "--shadow-file", victim[0], canary])
    finally:
        Path(canary).unlink(missing_ok=True)

    assert result.returncode != 0, (
        f"MYPY IS BLIND. Asked to check a function that returns an `int` from a `-> str` "
        f"signature, mypy — loaded with this repo's configuration, at the real module path "
        f"`{victim[0]}` — found no problem:\n\n  {MYPY_CANARY!r}\n"
        f"  -> {result.stdout.strip() or '(silence)'}\n"
        f"\n"
        f"check.sh still runs `mypy {PACKAGE}`. The Mypy row still prints ✅ and is still "
        f"CRITICAL. It is type-checking nothing.\n"
        f"\n"
        f"Look in pyproject.toml under [tool.mypy]: `ignore_errors = true`, "
        f'`follow_imports = "skip"`, a `disable_error_code` list, or a `[[tool.mypy.'
        f"overrides]]` block covering the package. Each of them makes mypy report Success "
        f"over code it has been told not to read."
    )


def test_mypy_is_not_globally_silenced() -> None:
    """The named-key half of the mypy canary, plus the one vector the canary cannot see.

    `exclude` is that vector: mypy ignores `exclude` for files passed explicitly on the
    command line (so the shadow-file canary sails through) but honours it when crawling a
    DIRECTORY — which is exactly how check.sh invokes it (`mypy src/xbrain`). So the canary
    is blind here and this static check is the only thing standing in front of it. The two
    tests are complements, not duplicates.
    """
    mypy = _table("tool", "mypy")

    assert not mypy.get("ignore_errors"), (
        "`[tool.mypy] ignore_errors = true` switches mypy off for the whole package. It still "
        "runs, still reads all 48 files, and reports `Success: no issues found`. The Mypy row "
        "stays ✅ and stays CRITICAL. This is not a configuration choice; it is a mute button."
    )
    assert mypy.get("follow_imports") != "skip", (
        '`[tool.mypy] follow_imports = "skip"` makes mypy treat imported modules as `Any`, '
        "so type errors that cross a module boundary — which is most of them in this package "
        "— stop being detected. (`silent` is the safe setting and is allowed.)"
    )
    assert not mypy.get("disable_error_code"), (
        f"`[tool.mypy] disable_error_code = {mypy.get('disable_error_code')}` turns off entire "
        f"classes of type error for the whole package, silently, from a file nobody rereads. "
        f"If one line genuinely needs an exemption, `# type: ignore[code]` it AT THAT LINE, "
        f"where a reviewer sees what is being excused."
    )

    excluded = _package_matching_patterns([str(p) for p in (mypy.get("exclude") or [])])
    assert not excluded, (
        f"`[tool.mypy] exclude = {excluded}` removes part of {PACKAGE} from the type check.\n"
        f"\n"
        f"check.sh runs `mypy {PACKAGE}` — a DIRECTORY — and mypy honours `exclude` when it "
        f"crawls a directory. Those modules are simply not checked, while the command in "
        f"check.sh is untouched and the Mypy row still reports ✅."
    )

    blinded = [
        override.get("module")
        for override in (_pyproject().get("tool", {}).get("mypy", {}).get("overrides", []) or [])
        if override.get("ignore_errors") or override.get("follow_imports") == "skip"
    ]
    assert not blinded, (
        f"pyproject.toml turns mypy OFF for {blinded} via `[[tool.mypy.overrides]]`. check.sh "
        f"still runs `mypy {PACKAGE}` and mypy still prints Success — over modules it has been "
        f"told to stop reading."
    )


def _bandit_config_flags(argv: list[str]) -> list[str]:
    """Config files the gate hands bandit via `-c` / `--configfile`, if any."""
    return [
        argv[index + 1]
        for index, token in enumerate(argv[:-1])
        if token in {"-c", "--configfile", "--ini"}
    ]


def test_bandit_still_reports_an_obvious_finding(tmp_path: Path) -> None:
    """Bandit, run with the gate's OWN flags, must still report a MEDIUM-severity finding.

    Run with the gate's real argv — not a clean invocation — because the flags ARE the
    configuration here: `-lll` raises the threshold to HIGH-only, `-iii` to HIGH-confidence
    only, `--skip B108` drops the rule outright. Each leaves `bandit -r src/xbrain` looking
    correct and the Bandit row printing ✅ while most of what it used to block walks through.

    The canary is deliberately MEDIUM/MEDIUM. A HIGH-severity canary would still be reported
    under `-lll` and this test would have been green from birth against the very edit it is
    written to catch.
    """
    canary = tmp_path / "gate_canary.py"
    canary.write_text(BANDIT_CANARY)

    for argv in _tool_invocations("bandit"):
        # Swap the scan target for the canary; keep every other flag exactly as the gate has
        # it, including any `-c` config file, so the tool is configured identically.
        args = [str(canary) if token == PACKAGE else token for token in argv]
        result = _run_tool("bandit", args)
        assert result.returncode != 0, (
            f"BANDIT IS DESENSITISED. Handed a hardcoded `/tmp` path (B108 — MEDIUM severity, "
            f"MEDIUM confidence), bandit — run with the gate's own flags — reported nothing:\n"
            f"  bandit {' '.join(args)}\n  -> {result.stdout.strip()[:400] or '(silence)'}\n"
            f"\n"
            f"The gate still scans {PACKAGE}. The Bandit row still prints ✅ and is still "
            f"CRITICAL. It has stopped reporting the severity band it has enforced all along.\n"
            f"\n"
            f"Look at the invocation in scripts/check.sh: `-lll` (HIGH severity only), `-iii` "
            f"(HIGH confidence only), a `--skip B…` list — or a `-c pyproject.toml` that pulls "
            f"in a `[tool.bandit] skips`. Suppress the individual finding with `# nosec` and a "
            f"reason, at the line, where the exemption is visible."
        )


def _expand_grade_class(character_class: str) -> set[str]:
    """Expand a shell bracket class of complexity grades: `D-F` -> {D,E,F}, `DEF` -> {D,E,F}."""
    grades: set[str] = set()
    index = 0
    while index < len(character_class):
        if index + 2 < len(character_class) and character_class[index + 1] == "-":
            start, end = character_class[index], character_class[index + 2]
            grades |= {chr(code) for code in range(ord(start), ord(end) + 1)}
            index += 3
        else:
            grades.add(character_class[index])
            index += 1
    return grades


def test_radon_still_fails_on_every_forbidden_complexity_grade() -> None:
    """The radon threshold is a `grep` character class, and narrowing it is a one-key edit.

    check.sh decides that a function is too complex by grepping its own radon output for a
    grade in `[D-F]`. Change that class to `[F]` and grades D and E — the ones a real
    refactor actually produces — sail through. Radon still runs. It still prints the offending
    function. The Radon row still says ✅ and is still CRITICAL, and `mark_failed "Radon"` is
    still right there in the source, never reached.

    This is the severity-table attack (`test_every_critical_check_can_still_fail_the_gate`)
    hiding one level down, inside the condition rather than the branch — and it is invisible
    to a test that only checks the `mark_failed` call exists.

    The class is expanded rather than string-matched, so `[D-F]` and `[DEF]` both pass: this
    guards the RULE, not its spelling.
    """
    code = _check_sh_code()
    radon_section = code[code.find("RADON_OUTPUT=$(") : code.find('mark_failed "Radon"')]
    assert radon_section, (
        "The radon section of scripts/check.sh no longer looks the way this test expects — it "
        'could not find the radon invocation and the `mark_failed "Radon"` call that its '
        "grade check guards. Radon is a CRITICAL check (grades D/E/F block the merge). If you "
        "restructured it, re-point this test at the new condition in the same PR."
    )

    gate_line = next(
        (line for line in radon_section.splitlines() if "grep -qE" in line and " - [" in line),
        None,
    )
    assert gate_line, (
        "scripts/check.sh no longer decides the radon FAILURE with a `grep -qE ' - [D-F] …'` "
        "over radon's output. That grep is the entire D/E/F gate: without it, complexity is "
        "reported and never blocks."
    )

    found = re.search(r" - \[([A-Z-]+)\]", gate_line)
    grades = _expand_grade_class(found.group(1)) if found else set()
    missing = sorted({"D", "E", "F"} - grades)
    assert not missing, (
        f"The radon complexity gate no longer fails on grade(s) {missing}:\n"
        f"  {gate_line.strip()}\n"
        f"\n"
        f"check.sh's own header promises 'radon: D/E/F = critical (exit 1)'. This condition now "
        f"only catches {sorted(grades)}. A function at grade {missing[0]} is reported in the "
        f"log, printed in the summary — and merges, because the branch that calls "
        f'`mark_failed "Radon"` is never reached. The call is still in the file, which is '
        f"exactly what makes this invisible: every guard that checks 'is Radon critical?' by "
        f"looking for that call still says yes.\n"
        f"\n"
        f"D and E are the grades a genuine complexity regression produces. Narrowing the class "
        f"to F is not a tightening, it is a surrender."
    )


def test_bandit_config_file_if_loaded_does_not_narrow_the_scan() -> None:
    """`[tool.bandit]` in pyproject.toml is DEAD CONFIG today — and must stay that way.

    Measured: `bandit -r src/xbrain -ll -q`, which is exactly how check.sh invokes it, does
    NOT read pyproject.toml. Bandit only loads it when handed `-c pyproject.toml`. So the
    `exclude_dirs` and `skips` keys sitting in pyproject right now change nothing.

    That matters for what this test may honestly assert. An earlier version of this module
    asserted unconditionally that `exclude_dirs` must not touch the package — and would have
    gone red, with a message claiming "the security scan now skips part of the package", on
    an edit that in fact does nothing at all. A guard that lies about the damage is worse
    than no guard: it teaches the next reader that the guard is noise.

    So the assertion is conditional on the config actually being LIVE. The moment someone
    adds `-c pyproject.toml` to the invocation, those keys wake up — and this fires.
    """
    for argv in _tool_invocations("bandit"):
        if not _bandit_config_flags(argv):
            continue  # pyproject's [tool.bandit] is inert for this invocation
        bandit = _table("tool", "bandit")
        assert not bandit.get("skips"), (
            f"The gate now passes a config file to bandit (`{' '.join(argv)}`), which makes "
            f"`[tool.bandit] skips = {bandit.get('skips')}` LIVE. Those checks are no longer "
            f"run. Until this invocation changed, that key was inert — so the diff that "
            f"switched it on may look like it changed nothing. It disabled security rules."
        )
        excluded = _package_matching_patterns([str(p) for p in (bandit.get("exclude_dirs") or [])])
        assert not excluded, (
            f"The gate now passes a config file to bandit, which makes `[tool.bandit] "
            f"exclude_dirs = {excluded}` LIVE — and it removes part of {PACKAGE} from the "
            f"security scan, while `bandit -r {PACKAGE}` still reads correctly in check.sh."
        )


def test_the_detect_secrets_scan_is_not_narrowed_by_flags() -> None:
    """detect-secrets must not be handed an exclusion flag.

    `--exclude-files 'src/.*'` shrinks the scan without touching a single positional path,
    so the scope test above — which sees `src/xbrain tests scripts`, all three trees, intact
    — stays perfectly green while the scan reads almost nothing.
    """
    offenders = [
        f"{token} (in `detect-secrets {' '.join(argv)}`)"
        for argv in _tool_invocations("detect-secrets")
        for token in argv
        if any(token == flag or token.startswith(f"{flag}=") for flag in SECRETS_NARROWING_FLAGS)
    ]
    assert not offenders, (
        "The secrets scan is narrowed by an exclusion flag:\n  "
        + "\n  ".join(offenders)
        + "\n\nThe positional paths still name all three trees, so every scope assertion stays "
        "green — and the scan no longer reads them. A committed key in an excluded file is now "
        "invisible to a gate that reports ✅ Detect-secrets PASS.\n\n"
        "A known false positive belongs in `.secrets.baseline`, where it is audited, "
        "attributable and visible in the diff — not behind a regex in the command line."
    )


def test_coverage_excludes_no_ordinary_code() -> None:
    """No coverage exclusion pattern may match ordinary code.

    Measured, and it is the worst of the lot:

        [tool.coverage.report]
        exclude_lines = ["."]          ->  TOTAL  0  0  100%

    One character. Every line of the package matches `.`, so every line is excluded, so
    coverage reports **100%** over **zero statements** — comfortably above the 78% floor,
    which is still sitting there in check.sh reading exactly as it always did. Both of the
    coverage tests above are satisfied: the floor was not lowered, and every file is still
    "measured". There is simply nothing left in them to measure.

    The check is behavioural: each pattern is compiled and run against perfectly ordinary
    code. A pattern that matches `x = 1` is not excluding a special case, it is excluding the
    program. Legitimate patterns (`pragma: no cover`, `if TYPE_CHECKING:`,
    `raise NotImplementedError`, `\\.\\.\\.`) match none of it and stay green.
    """
    report = _table("tool", "coverage", "report")
    patterns = list(report.get("exclude_lines") or []) + list(report.get("exclude_also") or [])

    for pattern in patterns:
        try:
            compiled = re.compile(pattern)
        except re.error as error:
            pytest.fail(
                f"`[tool.coverage.report]` has an uncompilable exclusion pattern {pattern!r}: "
                f"{error}. Coverage will refuse to report, and check.sh will fail the gate on "
                f"an unparseable TOTAL — loudly, at least."
            )
        swallowed = [line for line in ORDINARY_CODE_LINES if compiled.search(line)]
        assert not swallowed, (
            f"The coverage exclusion pattern {pattern!r} matches ORDINARY CODE — "
            f"{swallowed} — so it does not exclude a special case, it excludes the program.\n"
            f"\n"
            f'`exclude_lines = ["."]` reports TOTAL 0 statements and 100% coverage. The 78% '
            f"floor in check.sh is untouched and still reads as protection; the denominator "
            f"underneath it is gone. Every other coverage test in this module passes, because "
            f"the floor really was not lowered and every file really is still 'measured' — "
            f"there is nothing left inside them to count.\n"
            f"\n"
            f"Exclusion patterns are for genuine special cases: `pragma: no cover`, "
            f"`if TYPE_CHECKING:`, `raise NotImplementedError`. If a real module cannot be "
            f"tested, say so at the line with `# pragma: no cover`, where a reviewer sees it."
        )


def test_no_package_file_blanket_suppresses_a_critical_tool() -> None:
    """No source file may switch a critical tool off for itself, wholesale.

    `# ruff: noqa` on line 1 disables every rule for that file. `# mypy: ignore-errors` does
    the same for types. Both are file-local, so no canary and no config check sees them —
    they are the in-source form of exactly the same attack, and they arrive in the diff of
    the file they excuse, which is the one place a reviewer might not be looking for a gate
    change.

    Codes are required, not banned: `# ruff: noqa: E501` is a narrow, reviewable exemption
    and stays green. It is the BARE, uncoded form — "ignore everything, forever, here" —
    that this rejects.
    """
    bare_ruff = re.compile(r"^#\s*ruff:\s*noqa\s*$", re.MULTILINE)
    bare_mypy = re.compile(r"^#\s*mypy:\s*ignore-errors\s*$", re.MULTILINE)

    offenders = []
    for relpath in sorted(_tracked_package_files()):
        try:
            text = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        except OSError:  # tracked but absent from the working tree
            continue
        if bare_ruff.search(text):
            offenders.append(f"{relpath}: `# ruff: noqa` — every lint rule, off, for this file")
        if bare_mypy.search(text):
            offenders.append(f"{relpath}: `# mypy: ignore-errors` — all type checking, off")

    assert not offenders, (
        "These files switch a critical tool off for themselves:\n  "
        + "\n  ".join(offenders)
        + "\n\nThe tool still runs. Its scope still covers the file. The gate still prints ✅ "
        "and keeps its CRITICAL severity — and the file is exempt from all of it. This is the "
        'per-file version of `per-file-ignores = {"*" = ["ALL"]}`, and it hides in the diff '
        "of the file it excuses.\n\n"
        "Name what you are suppressing: `# noqa: E501` at the line, `# type: ignore[arg-type]` "
        "at the line. An exemption a reviewer can weigh is a different thing from a mute button."
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
