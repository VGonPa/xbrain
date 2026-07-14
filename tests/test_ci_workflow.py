# tests/test_ci_workflow.py
"""A push to `develop`/`main` must produce a passing-or-failing check run named `quality`.

That sentence — not "the workflow declares a push trigger" — is the property this file
exists to defend. Two things depend on it:

1. **Detection.** CLAUDE.md rule 4 ("A green PR against a moving `develop` is not a green
   `develop`") was paid for in blood on 2026-07-14 (#103). PR #94 changed
   `_source_text(item)` to `_source_text(item, target)`; PR #97 added a test calling
   `_source_text(item)`. Both were green, there was zero textual conflict, git merged both
   happily — and `develop` was RED.

   Be precise about WHY, because the obvious explanation is wrong: it is **not** that the
   merge went untested. `pull_request` already tests a merge — GitHub builds
   `refs/pull/N/merge` (the PR head merged into the base) and checks THAT out. The defect
   is that the merge ref is computed when the run is **triggered** and is never recomputed
   when the base moves under it. #97 landed at 14:01:08Z; #94 merged at 14:04:08Z on a
   green measured against a base that did not contain #97. Merging #97 does not re-trigger
   #94. So the merge that was *tested* was not the merge that *landed*. The failure mode is
   **staleness, not absence** — and nothing caught it, because nothing ran on the branch
   afterwards. A human found it by hand.

   Hence: `push` is the **detector** (every push to a gated branch IS the true merge
   result), and branch protection's `strict` is the **preventer** (it forces the merge ref
   to be recomputed against the current base before merging). Complementary, not redundant.

2. **Branch protection.** `develop` and `main` require a status check named exactly
   `quality`. GitHub derives that name from the job's `name:` if it declares one, else
   from the **job id**. So the job's identity is load-bearing repo infrastructure: rename
   it and the required check never appears, GitHub waits for it forever, and **every merge
   is blocked permanently**. That deadlock is strictly worse than the bug this file was
   written to catch.

3. **Alerting.** A red push run blocks nothing — the commit has already landed — so it must
   raise an alarm or it is detecting into a void. On 2026-07-14 `develop` sat red for
   9m15s, three commits took the red commit as their parent, and two agents opened
   duplicate hotfix PRs 31 seconds apart. Watching the Actions tab is not a control.

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

WHY A PR CANNOT BE TRUSTED TO POLICE ITS OWN GATE
-------------------------------------------------
A `pull_request` run uses the **HEAD version of the workflow**, not the base's. So a PR that
neuters the gate **is judged by the neutered gate**. It absolves itself. That is the whole
reason this file exists as a *test* — something that runs INSIDE the suite, on the PR's own
tree — rather than as a comment asking people to be careful.

THE TAXONOMY: FAIL-CLOSED vs FAIL-OPEN
--------------------------------------
Every way to kill this gate lands in one of two families, and they fail in OPPOSITE
directions. Sorting an attack into the wrong family is worse than not having the taxonomy,
so each line below is what was MEASURED against the live API, not what was assumed.

**FAIL-CLOSED (annoying, not dangerous).** The check stops *reporting PASS*, so GitHub
blocks the merge. A bad day; it cannot ship a lie.

  * delete the workflow / `jobs: {}` -> no run at all (measured: zero runs, PR `BLOCKED`)
  * rename the job, or give it a `name:` -> the required check never appears; PRs hang
    Pending forever, and with `enforce_admins` nobody can override — including the PR that
    would fix it. A hard LOCKOUT, but never a false green.
  * `branches-ignore`, `paths:` under `pull_request`, invalid `types:` under `push` -> the
    workflow is skipped. GitHub, verbatim: *"When a workflow is skipped due to path
    filtering, branch filtering or a commit message, checks associated with that workflow
    will remain in a 'Pending' state."*
  * `continue-on-error: true` -> see the instrument trap below. Also fail-closed.

**FAIL-OPEN (lethal).** The check still says **PASS while testing less**. It merges green
and manufactures false confidence — worse than no gate, because a decoration is trusted.

  * `paths-ignore` under `push` -> the merge-result run never happens; #103 reopens, silently.
  * `steps:` gutted to `echo ok` -> a green `quality` check that ran nothing.
  * `if: false` on the JOB -> GitHub, verbatim: *"if a job within a workflow is skipped due
    to a conditional, it will report its status as 'Success.'"* A skipped job is a GREEN job.
  * **`checkout` with an explicit `ref:`** -> the cleanest attack found against this file.
    The gate RUNS. All eleven checks PASS. The check run reports a truthful ✅ `quality`.
    Branch protection is satisfied. And it examined **the wrong tree** — it never looked at
    the code being merged. Not a broken gate: a working gate pointed at the wrong thing,
    with no visible symptom anywhere. `actions/checkout` defaults to `GITHUB_SHA`, which on
    a push to a gated branch IS the merge commit. That default is the entire mechanism of
    this workflow, and nothing but the test below pins it.

THE INSTRUMENT TRAP (read this before you "discover" a hole)
------------------------------------------------------------
`continue-on-error: true` was reported as a lethal fail-open hole — the reasoning being that
it gags every in-suite guard, this file included: the test fails, `check.sh` exits 1, the job
fails, and `continue-on-error` reports success anyway. Airtight, and **false**.

Measured on a throwaway PR whose gate was forced red, with the job declaring
`continue-on-error: true`:

    job's CHECK RUN    -> conclusion = FAILURE     <- branch protection reads THIS
    the WORKFLOW RUN   -> conclusion = success     <- `gh run list` shows THIS
    the PR             -> mergeStateStatus = BLOCKED

Two instruments, the same event, opposite answers. `continue-on-error` suppresses the
WORKFLOW RUN's conclusion; it does NOT suppress the JOB's check run, and branch protection
keys off the check run. The merge is blocked. The attack is fail-closed, and the guard is
not gagged.

Read the wrong instrument and you will invent a lethal hole that does not exist — which is
the same mistake, in the same PR, that produced a wrong account of #103 (`gh run list` said
`success` for a stale run) and a duplicate-issue bug (every GitHub issue LIST endpoint lags;
only fetch-by-number is strongly consistent). Three times, one lesson: **audit the instrument
before you cite it.** The gate is asserted below anyway — a repo that has to discover its
gate is decorative by pushing a bad commit has still had a bad day — but it is asserted as
fail-closed, which is what it is.
"""

import fnmatch
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

# The script that files/updates/closes the "branch is RED" issue. A push-triggered red run
# blocks nothing — the commit has already landed — so if it does not SPEAK, it detects into
# a void. On 2026-07-14 `develop` was red for 9m15s and three commits took the red commit as
# their parent, including two duplicate hotfixes opened 31s apart by two agents who each
# rediscovered the breakage by hand.
_ALERT_SCRIPT = "scripts/announce_red_branch.sh"

# The alert steps must be scoped to `push`: a failing PR run must NOT open issues (the PR's
# own red check is already visible to its author, and one issue per pushed PR commit is spam).
_PUSH_GUARD = "github.event_name == 'push'"


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


def _matches_any(patterns: list[str], branch: str) -> bool:
    """Does `branch` match any GitHub branch-filter pattern?

    GitHub's branch filters are GLOBS, not literals — so a membership test (`branch in
    patterns`) is not merely imprecise, it is exploitable: `branches-ignore: ["**"]`
    disables the trigger for every branch, and `"develop" in ["**"]` is False, so a
    membership test concludes the gate still runs. Measured: that exact edit left an earlier
    version of this file green with the gate dead.

    `fnmatch` treats `*` as matching across `/` where GitHub does not, which makes this
    slightly MORE eager to conclude "this branch is matched". For the two slash-free branch
    names we gate that distinction cannot arise, and erring toward "matched" fails safe in
    both directions here: an over-eager match on `branches` says the gate runs (it does), and
    on `branches-ignore` says it does not (which raises the alarm rather than suppressing it).
    """
    return any(fnmatch.fnmatch(branch, str(pattern)) for pattern in patterns)


def _fires_on_branch(event: str, branch: str) -> bool:
    """Would `event` on `branch` trigger the workflow, per the branch filters?

    GitHub forbids `branches` and `branches-ignore` on the same event, so this is a genuine
    three-way choice, not two independent filters:

    * `branches-ignore` present -> the event fires on everything EXCEPT what it matches.
    * `branches` present        -> the event fires ONLY on what it matches.
    * neither                   -> the event fires on every branch.

    The ordering is load-bearing. An earlier version read `branches`, found it absent, and
    concluded "absent filter means every branch" — which is true ONLY when `branches-ignore`
    is absent too. With `branches-ignore` present, an absent `branches` means the exact
    OPPOSITE, and the helper cheerfully reported that a suppressed gate was running. That
    latent bug is now fixed at the source rather than masked by the ban in
    `test_gate_trigger_declares_no_path_filter` and friends, so this helper stays correct if
    anyone ever relaxes those bans.
    """
    config = _event(event)
    if config is None:
        return False
    ignore = config.get("branches-ignore")
    if ignore is not None:
        return not _matches_any(ignore, branch)
    branches = config.get("branches")
    return branches is None or _matches_any(branches, branch)


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


def test_gate_declares_no_continue_on_error() -> None:
    """`continue-on-error` must appear nowhere in the gate job. FAIL-CLOSED, but still wrong.

    Be precise about the severity, because the obvious reading is wrong and this file is
    where people will come to check. `continue-on-error: true` does NOT mask the check.
    Measured against the live API on a PR whose gate was forced red:

        job's CHECK RUN  -> FAILURE   <- branch protection reads this; the merge is BLOCKED
        the WORKFLOW RUN -> success   <- `gh run list` reads this; hence the confusion

    So this is fail-CLOSED: it cannot ship a lie, and it does not gag this test. It is banned
    anyway. At the JOB level it makes the workflow run's conclusion lie, so anything watching
    runs rather than checks (a dashboard, a `gh run list` in a script, a human glancing at the
    Actions tab) is told everything is fine. At the STEP level it is worse — a failing step
    with `continue-on-error: true` does not fail its job, so the gate genuinely goes green
    while red underneath. A repo should not have to discover its gate is decorative by pushing
    a bad commit and watching what happens.
    """
    job = _gate_job()
    assert "continue-on-error" not in job, (
        f"The `{_REQUIRED_CHECK}` job declares `continue-on-error: "
        f"{job.get('continue-on-error')!r}`. Branch protection still blocks the merge (the "
        f"job's CHECK RUN reports failure even though the WORKFLOW RUN reports success), so "
        f"this is fail-closed rather than lethal — but every tool that reads workflow runs "
        f"instead of check runs is now being lied to. Delete it."
    )
    offenders = [
        step.get("name", step.get("run", "?"))
        for step in (job.get("steps") or [])
        if "continue-on-error" in step
    ]
    assert not offenders, (
        f"These steps in the `{_REQUIRED_CHECK}` job declare `continue-on-error`: "
        f"{offenders}. A failing STEP with `continue-on-error: true` does not fail its job — "
        f"so unlike the job-level form, this one really does publish a ✅ `{_REQUIRED_CHECK}` "
        f"check while the gate is red underneath it. Fail-OPEN. Delete it."
    )


def test_push_trigger_declares_no_activity_types() -> None:
    """`types:` is not a valid key for `push` — it makes the whole workflow file invalid.

    FAIL-CLOSED, and the least dangerous attack in this file: an invalid workflow never runs,
    so the required check is never published and every PR hangs Pending. That is a hard
    lockout — a thoroughly bad day — but it cannot ship a lie, which is the only thing that
    would be worse. Ranked last for that reason, and closed anyway.

    GitHub documents activity types as "Not applicable" to `push`. It is a typo away: the
    `pull_request` block DOES take `types:`, so copy-pasting it up one event is easy and
    silent.
    """
    config = _event("push") or {}
    assert "types" not in config, (
        f"The `push` trigger declares `types: {config.get('types')!r}`, which GitHub does not "
        f"accept for `push` ('Not applicable'). The workflow file is INVALID and will never "
        f"run at all — so the `{_REQUIRED_CHECK}` check is never published and every PR to "
        f"{list(_GATED_BRANCHES)} hangs Pending forever. `types:` belongs only on "
        f"`pull_request`."
    )


def test_checkout_takes_no_explicit_ref() -> None:
    """The gate must test the commit that triggered it — the merge result — not a fixed ref.

    THE LETHAL ONE. Every other attack in this file either blocks the merge (fail-closed) or
    leaves a visible scar: a job that vanished, a check that never reports, a suite that runs
    nothing. This one leaves NO symptom. The gate runs. All eleven checks execute and pass.
    The check run reports a perfectly truthful ✅ `quality`. Branch protection is satisfied and
    the merge goes through — and the gate examined **the wrong tree**. It never looked at the
    code being merged at all.

    It is not a broken gate. It is a working gate pointed at the wrong thing, which is the
    hardest failure to see and the easiest to trust.

    `actions/checkout` defaults to `GITHUB_SHA` — on a push to a gated branch, that IS the
    merge commit. That default is the entire mechanism of this workflow, and nothing pins it
    except this test.
    """
    checkout = [
        step
        for step in (_gate_job().get("steps") or [])
        if "actions/checkout" in str(step.get("uses", ""))
    ]
    assert checkout, (
        f"The `{_REQUIRED_CHECK}` job never checks out the repository, so it cannot be "
        f"running the gate against the merge result — or against anything."
    )
    for step in checkout:
        ref = (step.get("with") or {}).get("ref")
        assert ref is None, (
            f"`actions/checkout` pins `ref: {ref!r}`. The gate would then test THAT tree "
            f"instead of the commit that triggered the run. On a push to a gated branch the "
            f"triggering commit IS the merge result — testing it is the whole point of this "
            f"workflow. Remove the `ref:` and let checkout default to GITHUB_SHA."
        )


def _alert_steps() -> list[dict[str, Any]]:
    """Every step that runs the red-branch alert script."""
    steps = _gate_job().get("steps") or []
    return [s for s in steps if _ALERT_SCRIPT in str(s.get("run", ""))]


def test_red_branch_failure_is_announced() -> None:
    """A red `develop` must SPEAK. Detecting into a void is not detecting.

    A push run that goes red blocks nothing — the commit has already landed. If it does not
    raise an alarm, the only thing standing between a red `develop` and the next twenty
    commits built on top of it is somebody happening to look at the Actions tab. On
    2026-07-14 nobody did, for 9m15s.
    """
    on_failure = [s for s in _alert_steps() if "failure()" in str(s.get("if", ""))]
    assert on_failure, (
        f"The `{_REQUIRED_CHECK}` job has no step running `{_ALERT_SCRIPT}` under "
        f"`if: failure()`. A red push run would then block nothing and tell nobody: the bad "
        f"commit is already on the branch, and the next commits will take it as their "
        f"parent. That is exactly how 2026-07-14 produced two duplicate hotfixes, opened 31 "
        f"seconds apart by two agents who each rediscovered the same breakage by hand."
    )


def test_red_branch_alert_only_fires_on_push() -> None:
    """The alert must be scoped to `push` — a failing PR must not open issues.

    A PR's red check is already in front of its author, and a PR that is pushed to five
    times would file five issues. The alert exists for the one case where nothing else
    speaks: a merge result that is already on the branch.
    """
    steps = _alert_steps()
    # Assert the steps EXIST before asserting a property of them. `all(... for s in [])` is
    # vacuously true: without this line the test would pass on a workflow with no alert at
    # all — a green test for a feature that had been deleted (CLAUDE.md rule 1).
    assert steps, f"No step runs `{_ALERT_SCRIPT}`, so there is nothing to scope to `push`."
    unguarded = [s for s in steps if _PUSH_GUARD not in str(s.get("if", ""))]
    assert not unguarded, (
        f"An alert step is not guarded by `{_PUSH_GUARD}`: "
        f"{[s.get('name', s.get('run')) for s in unguarded]}. Without that guard a failing "
        f"PULL REQUEST run also files issues — spamming one per pushed commit for a failure "
        f"its author is already looking at."
    )


def test_red_branch_alert_is_resolved_when_the_branch_goes_green() -> None:
    """A green push must close the alert. A stale alert is a training exercise in ignoring it.

    The issue asserts a live fact — "this branch is red RIGHT NOW" — and a green push on the
    true merge result disproves it. Leaving it open would teach everyone that the alert is
    usually out of date, which is exactly how the Actions tab stopped being read.
    """
    on_success = [s for s in _alert_steps() if "success()" in str(s.get("if", ""))]
    assert on_success, (
        f"No step runs `{_ALERT_SCRIPT}` under `if: success()`, so the red-branch issue is "
        f"never closed when the branch recovers. The alert would stay open after the fix "
        f"landed — and an alert that is routinely stale is one nobody reads."
    )


def test_gate_job_may_file_the_red_branch_issue() -> None:
    """The job needs `issues: write` to alert, and `contents: read` to exist at all.

    Declaring a `permissions:` block sets every scope NOT listed to `none` — so omitting
    `contents: read` does not merely narrow the token, it stops `actions/checkout` cloning
    the repo and the gate cannot run at all. The two entries are asserted together because
    adding the first without the second bricks the job.

    `issues: write` on the built-in `GITHUB_TOKEN` is what lets `github-actions[bot]` file
    the alert with no PAT, no bot account, and no second human in the loop.
    """
    permissions = _gate_job().get("permissions") or {}
    assert permissions.get("issues") == "write", (
        f"The `{_REQUIRED_CHECK}` job does not declare `permissions: issues: write`, so the "
        f"built-in GITHUB_TOKEN cannot file the red-branch issue and the alert step fails "
        f"with 403 exactly when it is needed most — on a red `develop`."
    )
    assert permissions.get("contents") == "read", (
        f"The `{_REQUIRED_CHECK}` job declares a `permissions:` block without "
        f"`contents: read`. A permissions block sets every unlisted scope to `none`, so "
        f"`actions/checkout` can no longer clone the repo and the gate cannot run AT ALL."
    )
