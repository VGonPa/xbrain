# tests/test_ci_workflow.py
"""A push to `develop`/`main` must produce a passing-or-failing check run named `quality`.

That sentence — not "the workflow declares a push trigger" — is the property this file
exists to defend. Two things depend on it:

1. **Detection.** CLAUDE.md rule 4 ("A green PR against a moving `develop` is not a green
   `develop`") was paid for in blood on 2026-07-14. PR #94 changed `_source_text(item)` to
   `_source_text(item, target)`; PR #97 added a test calling `_source_text(item)`. Both
   green on their own branches, zero textual conflict, so git merged both happily — and
   `develop` was RED. CI never said a word: it only ran on `pull_request`, so the one
   commit nobody ever tested was the merge commit itself. A human found it by hand (#103).

2. **Branch protection.** `develop` and `main` require a status check named exactly
   `quality`. GitHub derives that name from the job's `name:` if it declares one, else
   from the **job id**. So the job's identity is load-bearing repo infrastructure: rename
   it and the required check never appears, GitHub waits for it forever, and **every merge
   is blocked permanently**. That deadlock is strictly worse than the bug this file was
   written to catch.

An earlier version of this file asserted only that a `push:` key existed with the right
`branches:`. Four independent edits killed the gate while it stayed green:

| Attack | Effect | Why the old test missed it |
|---|---|---|
| rename job `quality:` → `gate:` | required check never appears → **all merges deadlock** | never looked at `jobs` |
| `paths-ignore: ["**"]` on push | gate runs on no push at all → #103 reopened | only read `branches` |
| keep id, add `name: Gate` | check run is named `Gate` → **all merges deadlock** | job id alone is not the check name |
| `if: false` on the gate step | check `quality` reports **GREEN having run nothing** | never looked at the steps |

So the assertions below model the *effect*, not the spelling: would a normal merge commit
pushed to `develop` actually produce a check run named `quality` that actually executes the
gate? Reformatting the YAML while preserving that behaviour stays green; any edit that
breaks it goes red, and says why in the failure message.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

_WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "quality.yml"

# The branches that are merged INTO. A push to one of these IS a merge result.
_GATED_BRANCHES = ("develop", "main")

# The status check name that branch protection on develop/main requires. This string is
# NOT cosmetic: it is a contract with the repo settings. See the module docstring.
_REQUIRED_CHECK = "quality"

# The script the gate job must actually execute. A check run named `quality` that runs
# nothing is worse than no check at all — it reports green.
_GATE_SCRIPT = "scripts/check.sh"

# Events that must be able to produce the `quality` check run.
_GATING_EVENTS = ("push", "pull_request")

# A PR must be gated when it is opened and on every subsequent push to it. These are
# GitHub's defaults for `pull_request`; narrowing `types:` away from them silently drops
# the PR-head runs.
_REQUIRED_PR_TYPES = ("opened", "synchronize")


def _workflow() -> dict[Any, Any]:
    """Parse the workflow file."""
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))


def _triggers() -> Any:
    """Return the workflow's trigger block, surviving YAML's Norway problem.

    In YAML 1.1 the bare key `on:` is the BOOLEAN `True`, not the string `"on"` — so
    `yaml.safe_load(...)["on"]` raises `KeyError` on a perfectly valid workflow. Quoting
    the key in the file (`"on":`) would make it a string instead. GitHub accepts both
    spellings, so look the key up under both: the test must not break the day someone
    quotes — or unquotes — it, because that edit changes no behaviour.
    """
    workflow = _workflow()
    for key in (True, "on"):
        if key in workflow:
            return workflow[key]
    raise AssertionError(f"{_WORKFLOW.name} declares no trigger block at all")


def _event(event: str) -> dict[str, Any] | None:
    """Config for `event`, or None if the workflow does not fire on it at all.

    Normalises every shape GitHub accepts: a bare string (`on: push`), a list
    (`on: [push, pull_request]`), or a mapping with filters. The first two carry no
    filters, so they normalise to an empty config rather than to None.
    """
    triggers = _triggers()
    if isinstance(triggers, str):
        return {} if triggers == event else None
    if isinstance(triggers, list):
        return {} if event in triggers else None
    if event not in triggers:
        return None
    return triggers[event] or {}  # `push:` with an empty body == every branch


def _fires_on_branch(event: str, branch: str) -> bool:
    """Would `event` on `branch` trigger the workflow, per the branch filters?

    An absent `branches:` means GitHub runs the event on EVERY branch, which covers
    `branch` too — that is a pass, not a miss. `branches-ignore` is the inverse filter and
    is checked first: it is the other way to exclude a branch while `branches:` looks fine.
    """
    config = _event(event)
    if config is None:
        return False
    if branch in (config.get("branches-ignore") or []):
        return False
    branches = config.get("branches")
    return branches is None or branch in branches


def _gate_job() -> dict[str, Any]:
    """The job whose check run branch protection requires."""
    jobs = _workflow().get("jobs") or {}
    assert _REQUIRED_CHECK in jobs, (
        f"No job with id `{_REQUIRED_CHECK}` in {_WORKFLOW.name} (found: {sorted(jobs)}).\n"
        f"\n"
        f"STOP — renaming this job BRICKS THE REPOSITORY. `develop` and `main` are "
        f"protected by a required status check named exactly `{_REQUIRED_CHECK}`, and "
        f"GitHub takes that name from this job. Rename the job and the required check "
        f"never appears; GitHub waits for it forever and EVERY MERGE IS BLOCKED, "
        f"permanently. If you really must rename it, change the required status check in "
        f"the branch-protection settings FIRST, then this constant, then the job."
    )
    return jobs[_REQUIRED_CHECK] or {}


def _check_run_name() -> str:
    """The name GitHub will give this job's check run.

    It is the job's `name:` when one is declared, and the job id otherwise. This
    indirection is the subtle half of the deadlock: a job can keep the id
    `quality` and still publish its check run as `Gate`.
    """
    return str(_gate_job().get("name", _REQUIRED_CHECK))


def _gate_step() -> dict[str, Any]:
    """The step that actually executes the quality gate."""
    steps = _gate_job().get("steps") or []
    for step in steps:
        if _GATE_SCRIPT in str(step.get("run", "")):
            return step
    raise AssertionError(
        f"No step in the `{_REQUIRED_CHECK}` job runs `{_GATE_SCRIPT}`.\n"
        f"\n"
        f"The check run named `{_REQUIRED_CHECK}` is what branch protection trusts. If it "
        f"no longer runs the gate, it reports GREEN having verified nothing — a required "
        f"check that always passes is worse than no required check at all."
    )


def test_gate_publishes_the_required_status_check_name() -> None:
    """The check run must be named `quality` — branch protection requires that exact name.

    Covers BOTH ways to break the name: renaming the job id (caught in `_gate_job`) and
    keeping the id while overriding the display name with `name:` (caught here).
    """
    assert _check_run_name() == _REQUIRED_CHECK, (
        f"The `{_REQUIRED_CHECK}` job declares `name: {_check_run_name()}`, so its check "
        f"run is published as `{_check_run_name()}` — NOT `{_REQUIRED_CHECK}`.\n"
        f"\n"
        f"STOP — this BRICKS THE REPOSITORY. `develop` and `main` require a status check "
        f"named exactly `{_REQUIRED_CHECK}`. Under this name it never appears, GitHub "
        f"waits for it forever, and EVERY MERGE IS BLOCKED, permanently. The job id alone "
        f"is not enough: GitHub names the check run after `name:` whenever one is set."
    )


@pytest.mark.parametrize("branch", _GATED_BRANCHES)
def test_gate_runs_on_push_to_gated_branch(branch: str) -> None:
    """A push to develop/main IS a merge result — the gate must run on it (rule 4)."""
    assert _fires_on_branch("push", branch), (
        f"quality.yml does not run on `push` to `{branch}`, so the merge commit is never "
        f"tested. Two PRs, each green on its own branch and with zero textual conflict, "
        f"can still merge into a RED `{branch}` while CI stays silent — this is exactly "
        f"how #103 happened. See CLAUDE.md rule 4."
    )


@pytest.mark.parametrize("branch", _GATED_BRANCHES)
def test_gate_still_runs_on_pull_request(branch: str) -> None:
    """Merge-result coverage is ADDED to PR coverage, never swapped for it."""
    assert _fires_on_branch("pull_request", branch), (
        f"quality.yml no longer runs on `pull_request` to `{branch}`. Testing the merge "
        f"result does not replace testing the PR head: without this, a broken branch is "
        f"only caught AFTER it has already landed on `{branch}`."
    )


@pytest.mark.parametrize("event", _GATING_EVENTS)
def test_gate_trigger_declares_no_path_filter(event: str) -> None:
    """Neither trigger may carry a `paths` / `paths-ignore` filter.

    A path filter is the quietest way to kill this gate: `paths-ignore: ["**"]` leaves a
    perfectly innocent-looking `push:` block that fires on nothing.

    The rule here is deliberately absolute — no path filter at all — rather than an attempt
    to decide which globs are "safe". A merge commit may touch ANY set of files, including
    a set that any given filter excludes, so no non-trivial path filter can guarantee the
    gate runs on every merge result. Modelling glob semantics to prove otherwise would be
    far more complexity than the property is worth. If a path filter is ever genuinely
    wanted, that is a deliberate decision to weaken the gate, and it should be argued in a
    PR that also changes this test — not slipped in under a green suite.

    A skipped workflow is not a benign no-op either: the required `quality` check is never
    created, so the PR sits blocked on a check that will never report.
    """
    config = _event(event) or {}
    offenders = [key for key in ("paths", "paths-ignore") if key in config]
    assert not offenders, (
        f"The `{event}` trigger declares {offenders}. A path filter can exclude a merge "
        f'commit from the gate entirely — `paths-ignore: ["**"]` silently disables it '
        f"while the `branches:` list still looks correct, reopening #103. The gate must run "
        f"on EVERY {event} to {list(_GATED_BRANCHES)}, whatever files it touches."
    )


def test_pull_request_trigger_covers_normal_pr_activity() -> None:
    """A narrowed `types:` is another way to stop the PR-head runs without touching branches.

    GitHub's default `pull_request` types are opened/synchronize/reopened. Declaring
    `types: [labeled]` (or similar) leaves `branches:` intact while the gate stops running
    when a PR is opened or updated. An absent `types:` is the correct, default state.
    """
    types = (_event("pull_request") or {}).get("types")
    if types is None:
        return  # defaults already include the events we need
    missing = [t for t in _REQUIRED_PR_TYPES if t not in types]
    assert not missing, (
        f"The `pull_request` trigger narrows `types:` to {types}, dropping {missing}. The "
        f"gate would stop running when a PR is opened or pushed to, so the PR head goes "
        f"untested while `branches:` still looks correct."
    )


def test_gate_job_is_not_conditional() -> None:
    """The gate job must carry no `if:`.

    An `if:` on the job is a way to neuter the gate while the trigger block above it looks
    completely healthy — `if: github.event_name == 'pull_request'` would undo this entire
    PR. Worse, a job skipped by `if:` does not run the gate yet still resolves its check
    run, so branch protection can be satisfied by a check that verified nothing.
    """
    assert "if" not in _gate_job(), (
        f"The `{_REQUIRED_CHECK}` job declares `if: {_gate_job().get('if')!r}`. A condition "
        f"here can stop the gate running while the `on:` block still looks correct, and a "
        f"skipped job can still satisfy the required check — a green light for code nobody "
        f"tested."
    )


def test_gate_step_actually_runs_and_is_not_conditional() -> None:
    """The step running the gate must exist and must not be skippable.

    `if: false` on this one step is the most dangerous edit in this file's threat model:
    the job still succeeds, so the required `quality` check goes GREEN — having executed
    none of the 11 checks. A required check that cannot fail is worse than none, because
    it is trusted.
    """
    step = _gate_step()  # raises with an explanation if the gate script is not run at all
    assert "if" not in step, (
        f"The step running `{_GATE_SCRIPT}` declares `if: {step.get('if')!r}`. If it is "
        f"skipped, the job still SUCCEEDS and the required `{_REQUIRED_CHECK}` check "
        f"reports GREEN having run none of the quality checks. Branch protection would "
        f"then be waving through completely unverified code."
    )
