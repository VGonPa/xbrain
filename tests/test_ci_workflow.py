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

A fifth arrived with the umbrella filter (c1d9f2b), and it deserves its own line because it
breaks nothing structural — it edits one list:

| Attack | Effect | Why the test missed it |
|---|---|---|
| `"VGonPa/umbrella-*"` → the two umbrella names the tests sample | every OTHER umbrella ungated; its child PRs go `CLEAN` with no check, then deadlock on the required context | the assertion asks "does THIS branch fire?", and a literal answers *yes* |

Measured 2026-09-02: that edit in both trigger lists left this file, then 21 assertions
strong, at **21 passed**. The
assertions were about two branches; the property is about a PATTERN, and no sample the filter
is allowed to contain can tell the two apart. `_UNENUMERATED_UMBRELLA_BRANCHES` and
`_assert_sample_is_unenumerable` are the repair — probe with names the filter may not name.

A sixth survived even that repair, because it attacked the INSTRUMENT rather than the filter:

| Attack | Effect | Why the test missed it |
|---|---|---|
| `"VGonPa/umbrella-*"` → `"*"` in both lists | GitHub's `*` does not cross `/`, so develop and main stay gated and EVERY umbrella is ungated | the matcher was `fnmatch`, whose `*` DOES cross `/` — it answered the same question as Actions with the opposite answer |

Measured 2026-09-02 on this tree: `branches: ["*"]` in both trigger lists against the guard as
it stood at 6ed64d4 reported **30 passed** — a fully green file over a workflow that gates no
umbrella whatsoever. The same mutant against the guard as it stands now reports **12 failed,
21 passed**, the same shape as deleting the pattern outright (also 12 failed). The repair is
`_github_filter_matches`, which models GitHub's `/` rule instead of inheriting `fnmatch`'s;
the reason a *sample* could never have caught this is that the samples were right and the
oracle reading them was wrong — rule 9, one level under the assertions.

WHY A PR CANNOT BE TRUSTED TO POLICE ITS OWN GATE
-------------------------------------------------
A `pull_request` run uses the **HEAD version of the workflow**, not the base's. So a PR that
neuters the gate **is judged by the neutered gate**. It absolves itself. That is the whole
reason this file exists as a *test* — something that runs INSIDE the suite, on the PR's own
tree — rather than as a comment asking people to be careful.

NEVER TRUST A REPORTED CONCLUSION. ASSERT ON THE SOURCE.
--------------------------------------------------------
This is the rule that governs every assertion below, and here is the receipt that buys it.
Measured on a throwaway PR (#129) with `continue-on-error: true` on the **gate step** and a
test rigged to fail, so `scripts/check.sh` genuinely exited 1:

    the raw log        -> "FAILED ... Critical issues in: Format, Tests"   <- GROUND TRUTH
    ---------------------------------------------------------------------------------
    the STEP           -> conclusion = success        <- the step that ran `exit 1`
    the JOB            -> conclusion = success
    the CHECK RUN      -> conclusion = SUCCESS        <- what branch protection reads
    the WORKFLOW RUN   -> conclusion = success
    the PR             -> mergeable = MERGEABLE, mergeStateStatus = CLEAN

A step ran `exit 1` and reported `conclusion: success`. **Every machine-readable surface said
green.** The failure existed nowhere but in log prose.

So a guard built on "did every step conclude success?" is defeated by the exact attack it
exists to catch — the conclusion field is the thing that lies. That is why every assertion in
this file parses `quality.yml` (and the gate script it names) and asserts on what they SAY.
Never on what a run REPORTS.

THE TAXONOMY: FAIL-CLOSED vs FAIL-OPEN
--------------------------------------
Every way to kill this gate lands in one of two families, and they fail in OPPOSITE
directions. Sorting an attack into the wrong family is worse than having no taxonomy at all,
so every line below is MEASURED against the live API, not reasoned about.

**FAIL-CLOSED (a bad day, not a disaster).** The check stops *reporting PASS*, so GitHub
blocks the merge. It cannot ship a lie.

  * delete the workflow / `jobs: {}` -> no run at all (measured: zero runs, PR `BLOCKED`)
  * rename the job, or give it a `name:` -> the required check never appears; PRs hang
    Pending forever, and with `enforce_admins` nobody can override — including the PR that
    would fix it. A hard LOCKOUT, but never a false green.
  * `branches-ignore`, `paths:` under `pull_request`, invalid `types:` under `push` -> the
    workflow is skipped. GitHub, verbatim: *"When a workflow is skipped due to path
    filtering, branch filtering or a commit message, checks associated with that workflow
    will remain in a 'Pending' state."*
  * `continue-on-error` **on the JOB** -> measured (#123): check run `FAILURE`, PR `BLOCKED`.
    It suppresses the WORKFLOW RUN's conclusion, not the JOB's check run — and protection
    keys off the check run. Banned anyway: it lies to every tool that reads runs rather than
    checks (a dashboard, a `gh run list` in a script, a human glancing at the Actions tab).

**FAIL-OPEN (lethal).** The check still says **PASS while testing less, or nothing**. It
merges green and manufactures false confidence — worse than no gate, because a decoration
is trusted.

  * `continue-on-error` **on a STEP** -> measured (#129, above): `CLEAN`, `MERGEABLE`, check
    run SUCCESS, while `check.sh` exited 1. **The worst attack in this repo.** Note what
    separates it from its fail-closed twin one bullet up: *indentation*. Same directive, same
    spelling, two lines apart — one blocks the merge, the other hands you a permanent
    unconditional green. That is why the assertion below walks the ENTIRE job rather than
    special-casing the gate step: any step can become load-bearing later.
  * `paths-ignore` under `push` -> the merge-result run never happens; #103 reopens, silently.
  * `steps:` gutted to `echo ok` -> a green `quality` check that ran nothing.
  * `if: false` on the JOB -> measured (#130): check run conclusion = **`skipped`**, and the
    PR goes `MERGEABLE` / **`CLEAN`**. Note that this is NOT what GitHub's own docs imply —
    they say a skipped job *"will report its status as 'Success'"*, and the API in fact
    reports `skipped`. The distinction does not save you: **branch protection treats a
    skipped required check as satisfied.** The merge is waved through either way. Cited from
    the docs before it was measured, and the docs were wrong about the mechanism while right
    about the consequence — which is exactly the kind of luck this file refuses to run on.
  * **`checkout` with an explicit `ref:`** -> the gate RUNS. All eleven checks PASS. The check
    run reports a *truthful* ✅ `quality`. And it examined **the wrong tree** — it never looked
    at the code being merged. Not a broken gate: a working gate pointed at the wrong thing,
    with no visible symptom anywhere. `actions/checkout` defaults to `GITHUB_SHA`, which on a
    push to a gated branch IS the merge commit. That default is the entire mechanism of this
    workflow, and nothing but the test below pins it.

THE BOUNDARY — the one sentence to remember if you remember nothing else:

    Removing the guard fails CLOSED. Hollowing it out while leaving the sign on the door
    fails OPEN — and GitHub will hand you a CLEAN merge state while it does.

Everything in the fail-open column has that shape: the check still reports, it just no longer
checks anything.

AUDIT THE INSTRUMENT BEFORE YOU CITE IT
---------------------------------------
Four times in one PR an instrument lied, and each lie was believed until it was measured:

  1. `gh run list` said `success` — for a STALE run. It produced a wrong published account
     of #103.
  2. Every GitHub issue LIST endpoint lags (only fetch-by-number is strongly consistent).
     "No issue exists" was false; it filed duplicate issues #113 and #114.
  3. The WORKFLOW RUN's conclusion said `success` where the CHECK RUN said `FAILURE`. Reading
     the wrong one invented a lethal hole that does not exist (job-level `continue-on-error`).
  4. And the reverse: the check run said `SUCCESS` where the LOG said the gate exited 1.
     Reading the reported conclusion MISSED the lethal hole that does exist (step-level).

(3) and (4) are the same directive, measured on the wrong placement. Being careful was not
enough; only running the experiment on the exact thing under test was.

  5. GitHub's OWN DOCS said a skipped job "will report its status as 'Success'". The API
     reports `skipped`. The consequence happened to be the same (branch protection waves it
     through) — so the classification survived on luck, not on evidence.

THE HOUSE RULE THIS BUYS
------------------------
Every cell of the table above carries a probe number, or it carries the words "reasoned, not
measured". Nothing in this file is asserted from a vendor's documentation, from a plausible
mechanism, or from a colleague's confident summary — including this author's. Two people on
this PR each measured ONE cell of a 2x2 and claimed the whole table; both were half right,
and the halves they got wrong were the lethal ones. If you add a row, run the probe against
a throwaway PR, read the CHECK RUN (not the workflow run, not the log), and close the probe
the moment you have read it — a probe carrying a gate-killing edit is one careless click from
a dead gate, so it never sits open.
"""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "quality.yml"

# The branches that are merged INTO. A push to one of these IS a merge result.
_GATED_BRANCHES = ("develop", "main")

# A large initiative is delivered as a sequence of small child PRs onto a long-lived UMBRELLA
# branch, and only the umbrella opens a PR to `develop`. Concrete branch names to test the
# `VGonPa/umbrella-*` filter with — globs, not literals, so a real name is what must match.
_UMBRELLA_BRANCHES = (
    "VGonPa/umbrella-knowledge-index-lexical",
    "VGonPa/umbrella-knowledge-embeddings",
)

# The umbrellas that do NOT exist yet — and that is the entire point of them.
#
# Every assertion about `_UMBRELLA_BRANCHES` above is satisfiable by LISTING those two names,
# because they are real. Measured 2026-09-02 on this tree: replace `"VGonPa/umbrella-*"` with
# the two literals `VGonPa/umbrella-knowledge-index-lexical` and
# `VGonPa/umbrella-knowledge-embeddings` in BOTH trigger lists, and this file — as it stood
# at c1d9f2b, 21 assertions — reports **21 passed**, while Plan 03's umbrella, Plan 04's,
# and every umbrella cut after them get
# no `quality` check at all. That is rule 11's FAIL-OPEN (the check still reports PASS while
# gating less) arrived at through rule 1's shape (the assertion satisfied for the wrong
# reason), and the enumerated names are structurally incapable of catching it, because they
# ARE the enumeration.
#
# So the probe has to be a name the author of the filter could not have written down.
# `_assert_sample_is_unenumerable` then refuses to let any of these appear in the filter, so
# "list them too" is not a way to pass — it is a way to go red, with a message saying so.
_UNENUMERATED_UMBRELLA_BRANCHES = (
    "VGonPa/umbrella-knowledge-lexical-rescue",  # Plan 02, re-cut under a name nobody chose yet
    "VGonPa/umbrella-knowledge-vector-index",  # Plan 03
    "VGonPa/umbrella-knowledge-hybrid-ranking",  # Plan 04
    "VGonPa/umbrella-2027-01-15-eval-harness",  # an awkward shape: dates, digits, many hyphens
)

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


def _github_filter_matches(pattern: str, branch: str) -> bool:
    """Does one GitHub branch-filter pattern match `branch`, by GitHub's rules and not fnmatch's?

    Exactly one distinction is modelled, and it is the whole reason this function exists
    instead of `fnmatch.fnmatch`: **`*` does not match `/`, and `**` does.** Every umbrella
    name is `VGonPa/umbrella-<something>` — one `/`, always — so under `fnmatch` the single
    edit `branches: ["*"]` matched every umbrella sample in this file and left the suite
    green over a workflow that, in production, would gate `develop`, gate `main`, and gate
    no umbrella at all. Pinned by
    `test_branch_filter_matching_models_githubs_glob_and_not_fnmatchs` and, end to end,
    by `test_a_filter_that_gates_no_umbrella_is_reported_as_gating_no_umbrella`.

    Everything else in the pattern is LITERAL. GitHub's filter syntax also gives meaning to
    `?`, `+` and `[]`, and this function gives them none: no filter in this repo uses one,
    and modelling semantics nobody here has measured would be inventing a second instrument
    to disagree with (rule 2). A filter that reaches for one is a deliberate change, and it
    arrives with this function or not at all.
    """
    regex: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern[i] == "*":
            if pattern[i + 1 : i + 2] == "*":
                regex.append(".*")  # `**` crosses `/`
                i += 2
            else:
                regex.append("[^/]*")  # `*` stops at `/` — the distinction that matters
                i += 1
        else:
            regex.append(re.escape(pattern[i]))
            i += 1
    return re.fullmatch("".join(regex), branch) is not None


def _matches_any(patterns: list[str], branch: str) -> bool:
    """Does `branch` match any GitHub branch-filter pattern?

    GitHub's branch filters are GLOBS, not literals — so a membership test (`branch in
    patterns`) is not merely imprecise, it is exploitable: `branches-ignore: ["**"]`
    disables the trigger for every branch, and `"develop" in ["**"]` is False, so a
    membership test concludes the gate still runs. Measured: that exact edit left an earlier
    version of this file green with the gate dead.

    The glob is then modelled by `_github_filter_matches`, not by `fnmatch`. `fnmatch` was
    the second version of this hole rather than the fix for the first: its `*` crosses `/`,
    so it answered the same question as GitHub with the OPPOSITE answer for every branch
    carrying a slash — which is every umbrella there will ever be.
    """
    return any(_github_filter_matches(str(pattern), branch) for pattern in patterns)


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


def _assert_sample_is_unenumerable(branch: str) -> None:
    """Fail unless `branch` is a legitimate probe for "the umbrella filter is a GLOB".

    `_fires_on_branch` cannot tell a glob from a literal — it answers "would this exact name
    fire?", and an enumeration of that exact name answers *yes*. So the discriminating power
    does not live in the assertion; it lives in the CHOICE OF SAMPLE, and that choice is what
    this function polices. Two preconditions, both load-bearing:

    1. **The filter must not NAME it.** A sample the filter lists is satisfied by enumeration
       and proves nothing about any other umbrella. Checked against the parsed `branches` list
       of every gating event rather than the raw file text, so documenting a branch name in a
       comment stays legal while listing it as a filter does not.
    2. **No second `/`.** GitHub's filter `*` does not match `/`, so `VGonPa/umbrella-*`
       genuinely does NOT gate `VGonPa/umbrella-knowledge/index`: such a name is ungated in
       production, and a probe demanding it be gated would be asserting the opposite of the
       live behaviour. `_github_filter_matches` now models that rule rather than inheriting
       `fnmatch`'s (where `*` crosses `/`), so the ban is no longer papering over an
       over-eager matcher — it is refusing a sample the correct matcher must answer `False`
       for. Kept, not relaxed: the two-segment shape is not part of the umbrella topology,
       and a probe that can only ever be red proves nothing about the filter.

    Gutting this function re-opens the enumeration hole in silence, so it has a positive
    control of its own: `test_the_unenumerable_precondition_can_actually_fail`.
    """
    for event in _GATING_EVENTS:
        patterns = (_event(event) or {}).get("branches") or []
        assert branch not in [str(pattern) for pattern in patterns], (
            f"`{branch}` is enumerated verbatim in {_WORKFLOW.name}'s `{event}.branches`, so "
            f"it can no longer prove anything: the assertions that use it are satisfied by "
            f"the literal, exactly as they were on 2026-09-02 when the two real umbrella "
            f"names were listed and all 21 assertions this file then carried stayed GREEN, "
            f"with every other umbrella ungated.\n"
            f"\n"
            f"Listing an umbrella one-by-one is the defect, not the fix. Restore the glob "
            f"(`VGonPa/umbrella-*`) and pick a probe name nobody has written down."
        )
    assert branch.count("/") == 1, (
        f"`{branch}` carries more than one `/`, which makes it useless as a probe and worse "
        f"than useless as reassurance. GitHub documents its filter `*` as not matching across "
        f"`/` while `fnmatch`'s does, so `_matches_any` would report this branch GATED where "
        f"GitHub skipped it — over-eagerness pointed at the exact property under test. "
        f"Umbrella probes are `VGonPa/umbrella-<one segment>`."
    )


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


@pytest.mark.parametrize("branch", _UMBRELLA_BRANCHES)
def test_gate_runs_on_pull_request_to_umbrella(branch: str) -> None:
    """A child PR onto an umbrella must produce the `quality` check — rule 14, and rule 12's trap.

    Rule 14: a PR whose base is a feature branch runs no gate at all, and
    `gh pr view --json mergeStateStatus` still says `CLEAN` — which means "nothing was
    required", not "everything required passed". The umbrella topology puts every child PR in
    exactly that position, so the trigger is what closes the hole.

    And it is not merely a detector here. The umbrella carries classic branch protection with
    `quality` as a REQUIRED context. Drop this branch filter and the required check is never
    created, so GitHub waits for it forever: every child PR becomes permanently unmergeable —
    the same shape as rule 12's approval trap, arrived at from the other side.
    """
    assert _fires_on_branch("pull_request", branch), (
        f"quality.yml does not run on `pull_request` to `{branch}`. Child PRs onto the umbrella "
        f"would produce NO check run, GitHub would report `CLEAN` for a PR nothing gated "
        f"(rule 14) — and because the umbrella's branch protection REQUIRES the `quality` "
        f"context, every child PR would then block forever on a check that never appears."
    )


@pytest.mark.parametrize("branch", _UMBRELLA_BRANCHES)
def test_gate_runs_on_push_to_umbrella(branch: str) -> None:
    """Rule 4 applies inside the umbrella too, and a push to it IS the merge result.

    Two child PRs, each green against the umbrella it was opened from and with zero textual
    conflict, can still merge into a RED umbrella — #103's mechanism, one level down. Nothing
    would notice until the umbrella opened its PR to `develop`, by which point the bisect
    surface is the whole initiative instead of one child.
    """
    assert _fires_on_branch("push", branch), (
        f"quality.yml does not run on `push` to `{branch}`, so the merge of a child PR into "
        f"the umbrella is never tested. Two children, each green on its own branch, can merge "
        f"into a red umbrella with no conflict for git to report. See CLAUDE.md rule 4."
    )


@pytest.mark.parametrize("branch", _UNENUMERATED_UMBRELLA_BRANCHES)
def test_pull_request_umbrella_filter_is_a_glob_not_an_enumeration(branch: str) -> None:
    """The `pull_request` umbrella filter must gate a SHAPE, not the umbrellas that exist today.

    The two tests above are real and they are also enumerable: they name two umbrellas that
    exist, so `branches: [develop, main, "VGonPa/umbrella-knowledge-index-lexical",
    "VGonPa/umbrella-knowledge-embeddings"]` satisfies both of them. Measured 2026-09-02 on
    this tree, that exact edit in BOTH trigger lists left this file — 21 assertions, at
    c1d9f2b — at **21 passed**: a
    green suite over a filter that gates two branches and abandons every other umbrella. The
    check kept reporting PASS while gating less, which is rule 11's fail-open, and it did so
    because the assertion was satisfied for the wrong reason, which is rule 1's.

    This test asks the question the other two cannot: does the filter cover an umbrella
    NOBODY WROTE DOWN? `_assert_sample_is_unenumerable` first proves the filter does not name
    this branch, so the only way to be gated is to be matched by a pattern — a glob.

    What goes red: enumerating literals; narrowing the glob to `VGonPa/umbrella-knowledge-*`;
    dropping the umbrella pattern from `pull_request` while keeping it under `push`; deleting
    it outright. What stays green: any spelling that still matches arbitrary umbrella names.
    """
    _assert_sample_is_unenumerable(branch)
    assert _fires_on_branch("pull_request", branch), (
        f"quality.yml does not run on `pull_request` to `{branch}` — an umbrella that appears "
        f"in NO branch filter. So the filter gates the umbrellas someone happened to list, "
        f"not the shape `VGonPa/umbrella-*`, and the next initiative to cut one gets child "
        f"PRs with no check run at all: GitHub reports `mergeStateStatus: CLEAN` for a PR "
        f"nothing gated (rule 14), and the umbrella's own branch protection — which REQUIRES "
        f"the `quality` context — then blocks every child of it forever, which is rule 12's "
        f"trap reached from the other side.\n"
        f"\n"
        f"Fix the FILTER, never this list: `pull_request.branches` must carry a pattern that "
        f"matches umbrella names nobody has chosen yet."
    )


@pytest.mark.parametrize("branch", _UNENUMERATED_UMBRELLA_BRANCHES)
def test_push_umbrella_filter_is_a_glob_not_an_enumeration(branch: str) -> None:
    """The same question for `push`, and it is a different failure — so it is a separate test.

    `pull_request` failing means child PRs are ungated or deadlocked. `push` failing means the
    merge of a child INTO the umbrella is never tested: rule 4 one level down, where two
    children each green against the umbrella they saw merge into a red umbrella with no
    textual conflict for git to report. The two triggers can be broken independently — a glob
    under one and literals under the other is a single-line edit — so asserting them together
    would hide which half died.
    """
    _assert_sample_is_unenumerable(branch)
    assert _fires_on_branch("push", branch), (
        f"quality.yml does not run on `push` to `{branch}` — an umbrella that appears in NO "
        f"branch filter. Any umbrella not written down by hand goes untested at its merges, "
        f"so two child PRs, each green against the umbrella they were opened from, can land "
        f"a RED umbrella that nothing notices until it opens its PR to `develop` — by which "
        f"point the bisect surface is the whole initiative. See CLAUDE.md rule 4."
    )


# The two tests above ask `_fires_on_branch`, and `_fires_on_branch` is only as truthful as
# the glob it models. That is where the guard was still blind. `fnmatch`'s `*` crosses `/`
# and GitHub's documented filter `*` does not, so `branches: [develop, main, "*"]` — or a
# bare `"*"` — in BOTH trigger lists left every assertion in this file GREEN (`fnmatch` says
# `*` matches `VGonPa/umbrella-knowledge-vector-index`) while GitHub Actions would have gated
# `develop`, gated `main`, and gated NOT ONE umbrella, because every umbrella name carries a
# `/`. That is rule 11's fail-open — the check still reports PASS while gating less — reached
# through rule 9: two instruments answering the same question with opposite answers, and this
# file reading the wrong one.
#
# The pair below is the negative control. The first pins the matcher itself; the second pins
# what the matcher is FOR, end to end, against a forged workflow carrying the exact two
# filters that must not be allowed to look gated.


def test_branch_filter_matching_models_githubs_glob_and_not_fnmatchs() -> None:
    """GitHub's filter `*` does not cross `/`. `fnmatch`'s does, and that gap is exploitable.

    Every umbrella name is `VGonPa/umbrella-<something>` — one `/`, always. So the single
    edit `branches: ["*"]` is the whole attack: it looks maximally permissive, it satisfies a
    `fnmatch`-backed matcher for every sample this file owns, and in production it gates the
    slash-free branches only. The umbrella tests would report the shape covered while no
    umbrella was covered at all.

    Only `*` and `**` are modelled. GitHub's `?`, `+` and `[]` are not, and they are treated
    as literal characters — nothing in this repo's filters uses them, and inventing semantics
    for them would be pinning behaviour nobody has measured. If one ever appears in a filter,
    that is a deliberate change and it should arrive with this function.
    """
    assert _matches_any(["*"], "develop"), "a slash-free branch IS matched by `*`"
    assert not _matches_any(["*"], "VGonPa/umbrella-knowledge-vector-index"), (
        "`*` was reported as matching a branch containing `/`. GitHub documents the opposite, "
        'so a filter of `["*"]` would gate develop/main and NO umbrella while this file '
        "called every umbrella gated — the fail-open this whole module exists to refuse."
    )
    assert _matches_any(["**"], "VGonPa/umbrella-knowledge-vector-index"), (
        "`**` is the spelling that DOES cross `/`; refusing it would make the matcher unable "
        "to see a real catch-all, which is the alarm-suppressing direction."
    )
    assert _matches_any(["VGonPa/umbrella-*"], "VGonPa/umbrella-knowledge-vector-index")
    assert not _matches_any(["VGonPa/umbrella-*"], "VGonPa/umbrella-knowledge/index"), (
        "the live glob must not be reported as covering a two-segment umbrella name: GitHub "
        "would skip it, and `_assert_sample_is_unenumerable` bans such probes for this reason"
    )
    assert not _matches_any(["VGonPa/umbrella-*"], "develop")
    assert not _matches_any(["develop"], "develop-2"), "a literal pattern stays anchored"


@pytest.mark.parametrize(
    ("branches", "what_is_wrong_with_it"),
    [
        ('[develop, main, "*"]', "the catch-all `*`, which GitHub does not apply across `/`"),
        ("[develop, main]", "no umbrella pattern at all"),
    ],
)
def test_a_filter_that_gates_no_umbrella_is_reported_as_gating_no_umbrella(
    branches: str,
    what_is_wrong_with_it: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The umbrella tests must be able to FAIL, and these are the two filters that must fail them.

    Deleting the pattern is the obvious edit and it was already caught. `["*"]` is the one
    that was not: it reads as "gate everything", it is shorter than what it replaces, and
    until the matcher modelled GitHub's `/` rule it turned this entire file green over a
    workflow that gated no umbrella whatsoever.

    Forged workflow, not the real one, and for the reason the neighbouring positive control
    gives: a control that fails against the live file cannot tell "the guard is broken" from
    "the filter is broken", which is rule 9 committed inside the file that documents rule 9.
    `develop` is asserted STILL gated in both cases, so a matcher that simply refused
    everything could not pass this test either.
    """
    forged = tmp_path / "quality.yml"
    forged.write_text(
        f"name: Quality\non:\n  push:\n    branches: {branches}\n"
        f"  pull_request:\n    branches: {branches}\njobs: {{}}\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "_WORKFLOW", forged)

    for event in _GATING_EVENTS:
        assert _fires_on_branch(event, "develop"), (
            f"the forged filter {branches} does gate `develop`; a control that reported "
            f"otherwise would be measuring a broken matcher, not a broken filter"
        )
        for branch in _UNENUMERATED_UMBRELLA_BRANCHES:
            assert not _fires_on_branch(event, branch), (
                f"`{event}.branches: {branches}` carries {what_is_wrong_with_it}, so GitHub "
                f"creates no `quality` check for `{branch}` — yet this file reports it gated. "
                f"With that answer, `test_{event}_umbrella_filter_is_a_glob_not_an_"
                f"enumeration` passes over an ungated umbrella: child PRs go `CLEAN` with "
                f"nothing required (rule 14) and then deadlock on the required context "
                f"(rule 12's trap from the other side)."
            )


def test_the_unenumerable_precondition_can_actually_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_assert_sample_is_unenumerable` must be able to REFUSE, or the two tests above are theatre.

    It is the only thing standing between "the filter gates a shape" and "the filter happens
    to list my four samples", and a precondition that cannot fire is decoration — rule 1: an
    assertion you have never watched fail is not protecting anything. Gut the helper to `pass`
    (the obvious way to make the umbrella tests green again without repairing the filter) and
    this test goes red.

    It runs against a FORGED workflow rather than the real one, and that is not squeamishness.
    An earlier draft used `develop` as the positive control because the live filter lists it —
    and `branches: ["*"]`, which lists nothing literally, turned this test red while reporting
    that the precondition was broken. It was not: the control was. A guard whose failure names
    the wrong surface is rule 9 committed inside the file that documents rule 9.

    Both directions are pinned, because a helper that refuses EVERYTHING would also make the
    umbrella tests unfalsifiable — loudly, but unfalsifiably.
    """
    forged = tmp_path / "quality.yml"
    forged.write_text(
        "name: Quality\n"
        "on:\n"
        '  push:\n    branches: [develop, "VGonPa/umbrella-listed-by-hand"]\n'
        '  pull_request:\n    branches: [develop, "VGonPa/umbrella-listed-by-hand"]\n'
        "jobs: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "_WORKFLOW", forged)

    with pytest.raises(AssertionError, match="enumerated verbatim"):
        _assert_sample_is_unenumerable("VGonPa/umbrella-listed-by-hand")
    with pytest.raises(AssertionError, match="more than one"):
        _assert_sample_is_unenumerable("VGonPa/umbrella-knowledge/index")
    _assert_sample_is_unenumerable("VGonPa/umbrella-not-listed-by-anyone")


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
    """The gate job must carry no `if:`. FAIL-OPEN — measured, not inferred.

    An `if:` on the job neuters the gate while the trigger block above it looks completely
    healthy: `if: github.event_name == 'pull_request'` would undo this entire workflow in one
    line. Measured on probe #130 with `if: false` on this job:

        check run `quality` -> conclusion = skipped
        the PR              -> mergeable = MERGEABLE, mergeStateStatus = CLEAN

    The check run resolves as `skipped` — not `success`, whatever the docs say — and **branch
    protection treats a skipped required check as satisfied**. The gate never runs, and the
    merge is waved through. Fail-open, and one word long.
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
    """`continue-on-error` must appear NOWHERE in the gate job — not on it, not on any step.

    The same directive, spelled the same way, does OPPOSITE things depending on how far it is
    indented. Both were measured against the live API on PRs whose gate was genuinely failing:

        on the JOB (#123)   -> check run FAILURE, PR BLOCKED         fail-CLOSED
        on a STEP (#129)    -> check run SUCCESS, PR CLEAN,          fail-OPEN, LETHAL
                               while `check.sh` exited 1

    The step-level form is the worst attack in this repo. A step ran `exit 1` and reported
    `conclusion: success`; the job, the check run, the workflow run and the PR ALL said green.
    The failure existed nowhere but in the log text. Two words of YAML turn the gate into a
    permanent, unconditional green that no machine-readable surface can distinguish from a
    real pass.

    The job-level form does not mask the check — protection keys off the check run, which
    still reports failure — but it is banned too: it makes the WORKFLOW RUN's conclusion lie,
    so every tool that reads runs rather than checks (a dashboard, a `gh run list` in a
    script, a human glancing at the Actions tab) is told everything is fine.

    Both bans walk the WHOLE job rather than special-casing the `Quality gate` step. A step
    that is decorative today (installing uv, setting up Python) is load-bearing the moment the
    gate depends on it, and a `continue-on-error` parked on it would be waiting.
    """
    job = _gate_job()
    assert "continue-on-error" not in job, (
        f"The `{_REQUIRED_CHECK}` job declares `continue-on-error: "
        f"{job.get('continue-on-error')!r}`. Measured: branch protection still BLOCKS the "
        f"merge (the job's check run reports failure even though the workflow run reports "
        f"success), so this is fail-closed rather than lethal — but every tool that reads "
        f"workflow runs instead of check runs is now being lied to. Delete it."
    )
    offenders = [
        step.get("name", step.get("run", "?"))
        for step in (job.get("steps") or [])
        if "continue-on-error" in step
    ]
    assert not offenders, (
        f"These steps in the `{_REQUIRED_CHECK}` job declare `continue-on-error`: "
        f"{offenders}.\n"
        f"\n"
        f"STOP — this is the most dangerous edit in this repository, and it is one "
        f"indentation level away from the harmless job-level form. MEASURED on a PR whose "
        f"gate genuinely failed: the step ran `exit 1`, and the step, the job, the check run, "
        f"the workflow run and the PR ALL reported success — mergeStateStatus CLEAN. The "
        f"required `{_REQUIRED_CHECK}` check becomes a permanent unconditional green that no "
        f"machine-readable surface can tell apart from a real pass. The failure survives only "
        f"in the raw log text, where nothing is looking."
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
