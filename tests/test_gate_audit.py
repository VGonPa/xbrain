# tests/test_gate_audit.py
"""The auditor's comparison logic: does what the gate REPORTED match what it actually DOES?

This is the logic behind `.github/workflows/gate-audit.yml`, extracted so it can be tested
at all — a GitHub Actions workflow cannot be run locally, but the decision it makes can.

THE QUESTION THIS ASKS, AND WHY IT IS THE RIGHT ONE
---------------------------------------------------
Every other guard in this repo (`tests/test_ci_workflow.py`) asks a SYNTAX question: does
`quality.yml` contain a forbidden key? That guard is real, and it is also structurally
incapable of catching the attack it was written for, because it runs INSIDE the gate it
guards. A PR's CI runs the PR's own HEAD version of the workflow. Add
`continue-on-error: true` to the `Quality gate` step and the gate still runs, the guard test
still FAILS — and `continue-on-error` swallows that failure too. The check run reports
SUCCESS, `mergeStateStatus` is CLEAN, and every API surface agrees the code is fine.
Measured on this repo, probes #121 and #125. The alarm rings inside the soundproofed room.

So this module asks an OUTCOME question instead, from outside the room:

    GitHub says the `quality` check for develop@SHA concluded `success`.
    I just ran `bash scripts/check.sh` on develop@SHA myself, honestly, with no
    `continue-on-error` anywhere and no `ref:` pointing at some other tree.
    It FAILED.
    Therefore the gate is LYING.

That comparison needs no keyword list. It catches the `continue-on-error` gag (#121/#125)
and the wrong-tree `checkout ref:` attack (#124) with the same assertion, and it would catch
the next trick nobody has thought of yet, because it never asks HOW the gate lied — only
whether what it said matches what is true.

THE ASYMMETRY THAT DECIDES WHAT COUNTS AS A LIE
-----------------------------------------------
Not every mismatch is an alarm. The two directions are not equivalent:

* reported green + audit RED  -> **LYING**. Unverified code merges on a green light. File it.
* reported red  + audit green -> merges are BLOCKED. Annoying, possibly flaky — but
  fail-CLOSED. Nothing unsafe can land. Do not file; a system that cries wolf about safe
  states teaches people to ignore it.

And "green" is not the string `"success"`. GitHub treats `neutral` and `skipped` as
satisfying a required check too — a job killed with `if: false` reports `skipped` and
branch protection merges right past it. So the predicate is not *"did it say success"* but
*"would this have blocked a merge?"*. `_blocks_merge` encodes exactly that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from xbrain.gate_audit import (
    NON_BLOCKING_CONCLUSIONS,
    Verdict,
    Violation,
    audit_workflow_source,
    classify,
    reported_gate,
    render_issue_body,
    should_file_issue,
)

# ---------------------------------------------------------------------------
# reported_gate: what did GitHub say about develop@SHA?
# ---------------------------------------------------------------------------


#: A real, public commit SHA from this repo's history — used as realistic test data.
#:
#: `# pragma: allowlist secret` because detect-secrets scores any 40-char hex string as a
#: "Hex High Entropy String". A git SHA is precisely that shape and is precisely not a secret;
#: every commit hash in the repo is world-readable. Audited, false positive.
#:
#: (Worth knowing: `detect-secrets scan` only reads GIT-TRACKED files, so this fires the
#: moment the file is `git add`ed and NOT before. A local `poe check` on a brand-new,
#: untracked file is a FALSE GREEN — which is how this one reached CI.)
_SHA = "9e9a5c028e48d92c71a407c9e227e79caeb54326"  # pragma: allowlist secret


def _run(name: str, conclusion: str | None, status: str = "completed", **extra: Any) -> Any:
    """One entry as the `/commits/{sha}/check-runs` endpoint returns it."""
    return {"name": name, "status": status, "conclusion": conclusion, **extra}


def test_reported_gate_reads_the_quality_check_run() -> None:
    """The happy path: one completed `quality` run, and we read its conclusion."""
    payload = {"check_runs": [_run("quality", "success")]}
    reported = reported_gate(payload)
    assert reported.conclusion == "success"
    assert reported.completed is True


def test_reported_gate_ignores_other_check_runs() -> None:
    """Only the check run branch protection actually requires is relevant."""
    payload = {"check_runs": [_run("codeql", "failure"), _run("quality", "success")]}
    assert reported_gate(payload).conclusion == "success"


def test_reported_gate_is_absent_when_the_gate_never_ran() -> None:
    """No `quality` run at all -> we know nothing. That is not the same as green."""
    payload = {"check_runs": [_run("codeql", "success")]}
    reported = reported_gate(payload)
    assert reported.found is False
    assert reported.conclusion is None


def test_reported_gate_takes_the_most_recent_rerun() -> None:
    """A re-run supersedes the original: branch protection honours the LATEST conclusion.

    Order in the API response is not contractually newest-first, so sort by `started_at`
    rather than trusting the list. Reading a stale first entry would let a red gate that
    was re-run green (or vice versa) be compared against the wrong answer.
    """
    payload = {
        "check_runs": [
            _run("quality", "failure", started_at="2026-07-14T09:00:00Z"),
            _run("quality", "success", started_at="2026-07-14T11:00:00Z"),
        ]
    }
    assert reported_gate(payload).conclusion == "success"


@pytest.mark.parametrize("status", ["queued", "in_progress"])
def test_reported_gate_in_flight_is_not_a_conclusion(status: str) -> None:
    """A gate still running has concluded nothing. Do not read `null` as a verdict."""
    payload = {"check_runs": [_run("quality", None, status=status)]}
    reported = reported_gate(payload)
    assert reported.found is True
    assert reported.completed is False


# ---------------------------------------------------------------------------
# classify: the comparison. This is the whole point of the layer.
# ---------------------------------------------------------------------------


def test_gate_reported_green_but_audit_failed_is_a_LIE() -> None:
    """THE case this entire layer exists to catch.

    `continue-on-error` on the gate step (probes #121/#125) and `checkout ref:` (probe #124)
    both land exactly here: GitHub reports SUCCESS, and an honest execution of the same gate
    on the same commit fails. Neither is identified by name — only by outcome.
    """
    reported = reported_gate({"check_runs": [_run("quality", "success")]})
    assert classify(reported, audit_passed=False) is Verdict.LYING


@pytest.mark.parametrize("conclusion", sorted(NON_BLOCKING_CONCLUSIONS))
def test_every_non_blocking_conclusion_can_be_a_lie(conclusion: str) -> None:
    """`skipped` and `neutral` satisfy branch protection just as `success` does.

    `if: false` on the job makes the check run report `skipped`, and GitHub merges straight
    past it. Comparing against the literal string `"success"` would miss that entirely.
    """
    reported = reported_gate({"check_runs": [_run("quality", conclusion)]})
    assert classify(reported, audit_passed=False) is Verdict.LYING


def test_gate_reported_red_and_audit_failed_is_merely_red() -> None:
    """Consistent red is NOT a lie. develop is broken and the gate is saying so.

    The `push`-triggered run already filed the red-branch issue for this commit. Filing a
    second, differently-worded issue for the same fact is how alert fatigue starts.
    """
    reported = reported_gate({"check_runs": [_run("quality", "failure")]})
    assert classify(reported, audit_passed=False) is Verdict.CONSISTENT_RED


def test_gate_reported_green_and_audit_passed_is_clean() -> None:
    """The normal, boring, overwhelmingly common outcome."""
    reported = reported_gate({"check_runs": [_run("quality", "success")]})
    assert classify(reported, audit_passed=True) is Verdict.CLEAN


def test_gate_reported_red_but_audit_passed_is_not_an_alarm() -> None:
    """Over-reporting failure is fail-CLOSED: merges are blocked, nothing unsafe lands.

    Usually a flake or a since-fixed transient. Worth printing, never worth an issue.
    """
    reported = reported_gate({"check_runs": [_run("quality", "failure")]})
    assert classify(reported, audit_passed=True) is Verdict.REPORTED_RED_AUDIT_GREEN


def test_a_missing_check_run_is_inconclusive_not_a_lie() -> None:
    """Absence of evidence is not evidence of a lie — the execution check must say nothing.

    The static audit is what catches a gate that never runs (a `paths:` filter, a deleted
    workflow). Conflating "I could not observe the gate" with "the gate lied" would make the
    auditor cry wolf every time it raced a fresh push.
    """
    reported = reported_gate({"check_runs": []})
    assert classify(reported, audit_passed=False) is Verdict.INCONCLUSIVE
    assert classify(reported, audit_passed=True) is Verdict.INCONCLUSIVE


# ---------------------------------------------------------------------------
# should_file_issue: which verdicts are worth waking someone for?
# ---------------------------------------------------------------------------


def test_only_a_lie_or_a_violation_files_an_issue() -> None:
    """Precisely two things wake a human: the gate lied, or the gate is disarmed."""
    assert should_file_issue(Verdict.LYING, []) is True
    assert should_file_issue(Verdict.CLEAN, [Violation("step-continue-on-error", "x")]) is True
    assert should_file_issue(Verdict.CLEAN, []) is False
    assert should_file_issue(Verdict.CONSISTENT_RED, []) is False
    assert should_file_issue(Verdict.REPORTED_RED_AUDIT_GREEN, []) is False
    assert should_file_issue(Verdict.INCONCLUSIVE, []) is False


def test_a_disarmed_gate_is_filed_even_while_it_still_passes() -> None:
    """THE reason the static audit exists alongside the execution check.

    Add `continue-on-error: true` on a day when develop happens to be green and the
    execution check sees no discrepancy at all — reported green, audit green, CLEAN. The
    gate is nonetheless already dead: the very next red commit sails through. Only reading
    the source catches a booby-trap that has not gone off yet.
    """
    violations = [Violation("step-continue-on-error", "Quality gate")]
    assert should_file_issue(Verdict.CLEAN, violations) is True


# ---------------------------------------------------------------------------
# audit_workflow_source: the static half. Names WHAT is wrong.
# ---------------------------------------------------------------------------

_HEALTHY = """
name: Quality
on:
  push:
    branches: [develop, main]
  pull_request:
    branches: [develop, main]
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Quality gate
        run: bash scripts/check.sh
"""


def _codes(source: str, tmp_path: Path) -> set[str]:
    """Violation codes raised against a workflow source."""
    path = tmp_path / "quality.yml"
    path.write_text(source, encoding="utf-8")
    return {v.code for v in audit_workflow_source(path)}


def test_a_healthy_gate_raises_nothing(tmp_path: Path) -> None:
    """The guard must be quiet on the real, current workflow shape, or it is noise."""
    assert _codes(_HEALTHY, tmp_path) == set()


def test_the_real_quality_workflow_is_clean(tmp_path: Path) -> None:
    """Run the auditor against the ACTUAL quality.yml in this repo, not just a fixture.

    A fixture can drift from reality. This asserts the thing the auditor will really read,
    on develop, is currently clean — so a red result in CI means the gate changed, not that
    the fixture is stale.
    """
    real = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "quality.yml"
    assert [v.code for v in audit_workflow_source(real)] == []


def test_continue_on_error_on_the_gate_step_is_caught(tmp_path: Path) -> None:
    """Probes #121/#125: the gag. Gate runs, gate fails, check run reports SUCCESS anyway."""
    source = _HEALTHY.replace(
        "        run: bash scripts/check.sh",
        "        run: bash scripts/check.sh\n        continue-on-error: true",
    )
    assert "step-continue-on-error" in _codes(source, tmp_path)


def test_continue_on_error_on_the_job_is_caught(tmp_path: Path) -> None:
    """Probe #119: at JOB level this is fail-closed (check reports FAILURE, merge BLOCKED).

    Banned anyway, and the distinction is the point: it is ONE INDENT away from the step
    form, which is catastrophic. A reviewer who has to work out which indent they are
    looking at will eventually get it wrong. Ban both; explain the difference in the message.
    """
    source = _HEALTHY.replace(
        "    runs-on: ubuntu-latest",
        "    runs-on: ubuntu-latest\n    continue-on-error: true",
    )
    assert "job-continue-on-error" in _codes(source, tmp_path)


def test_an_explicit_checkout_ref_is_caught(tmp_path: Path) -> None:
    """Probe #124: the wrong-tree attack. Everything passes — on a tree nobody asked about.

    Measured: the gate went green having run `1088 passed` where develop's suite has ~1600.
    Only the test COUNT gave it away, and nobody reads the count on a green run.
    """
    source = _HEALTHY.replace(
        "      - uses: actions/checkout@v4",
        "      - uses: actions/checkout@v4\n        with:\n          ref: main",
    )
    assert "checkout-ref" in _codes(source, tmp_path)


def test_renaming_the_job_off_quality_is_caught(tmp_path: Path) -> None:
    """The required check is named after the job. Rename it and the check never appears."""
    source = _HEALTHY.replace("  quality:", "  gate:")
    assert "gate-job-missing" in _codes(source, tmp_path)


def test_overriding_the_check_name_is_caught(tmp_path: Path) -> None:
    """Subtler than a rename: keep the id `quality`, publish the check run as `Gate`."""
    source = _HEALTHY.replace(
        "  quality:\n    runs-on: ubuntu-latest",
        "  quality:\n    name: Gate\n    runs-on: ubuntu-latest",
    )
    assert "check-renamed" in _codes(source, tmp_path)


@pytest.mark.parametrize("filter_key", ["paths", "paths-ignore"])
def test_a_path_filter_on_a_gating_trigger_is_caught(filter_key: str, tmp_path: Path) -> None:
    """A path-filtered workflow is SKIPPED, and a skipped workflow leaves the check Pending.

    Not green, not red — never published. Every PR then waits forever on a check that will
    not report, and with `enforce_admins` nobody can override, including the PR that would
    fix it. Hard lockout.
    """
    source = _HEALTHY.replace(
        "  push:\n    branches: [develop, main]",
        f"  push:\n    branches: [develop, main]\n    {filter_key}: ['src/**']",
    )
    assert "path-filter" in _codes(source, tmp_path)


def test_a_conditional_job_is_caught(tmp_path: Path) -> None:
    """A SKIPPED JOB REPORTS SUCCESS. `if: false` is a green light over an empty room."""
    source = _HEALTHY.replace(
        "    runs-on: ubuntu-latest",
        "    runs-on: ubuntu-latest\n    if: false",
    )
    assert "job-conditional" in _codes(source, tmp_path)


def test_a_conditional_gate_step_is_caught(tmp_path: Path) -> None:
    """Same silent green, one level down: skip the step, the job still succeeds."""
    source = _HEALTHY.replace(
        "        run: bash scripts/check.sh",
        "        if: false\n        run: bash scripts/check.sh",
    )
    assert "gate-step-conditional" in _codes(source, tmp_path)


def test_a_gate_that_no_longer_runs_check_sh_is_caught(tmp_path: Path) -> None:
    """The bluntest kill: keep the job, keep the name, replace the gate with `echo ok`."""
    source = _HEALTHY.replace("        run: bash scripts/check.sh", "        run: echo ok")
    assert "gate-step-missing" in _codes(source, tmp_path)


def test_a_deleted_workflow_is_caught(tmp_path: Path) -> None:
    """Deleting the file is the simplest disarm of all, and it must not read as 'clean'."""
    codes = {v.code for v in audit_workflow_source(tmp_path / "does-not-exist.yml")}
    assert codes == {"workflow-missing"}


def test_an_unparseable_workflow_is_caught(tmp_path: Path) -> None:
    """Invalid YAML never runs, so the required check is never published -> hard lockout."""
    assert _codes("jobs: [unclosed\n", tmp_path) == {"workflow-unparseable"}


def test_quoting_the_on_key_is_not_a_violation(tmp_path: Path) -> None:
    """YAML's Norway problem: bare `on:` parses as the BOOLEAN True, not the string "on".

    GitHub accepts `on:` and `"on":` identically. An auditor that understood only one
    spelling would either miss every path filter under the other, or scream at a
    behaviour-preserving reformat. Both spellings must audit the same.
    """
    assert _codes(_HEALTHY.replace("\non:", '\n"on":'), tmp_path) == set()


def test_a_path_filter_is_caught_under_the_quoted_on_key(tmp_path: Path) -> None:
    """The half of the Norway problem that actually bites: a real attack, quoted key."""
    source = _HEALTHY.replace("\non:", '\n"on":').replace(
        "  push:\n    branches: [develop, main]",
        "  push:\n    branches: [develop, main]\n    paths: ['src/**']",
    )
    assert "path-filter" in _codes(source, tmp_path)


# ---------------------------------------------------------------------------
# render_issue_body: the alarm has to be actionable at 3am
# ---------------------------------------------------------------------------


def test_the_issue_body_carries_the_evidence() -> None:
    """An alert without evidence is an alert people learn to close unread."""
    body = render_issue_body(
        verdict=Verdict.LYING,
        violations=[Violation("step-continue-on-error", "Quality gate")],
        sha=_SHA,
        reported="success",
        audit_passed=False,
        repo="VGonPa/xbrain",
        run_url="https://github.com/VGonPa/xbrain/actions/runs/1",
    )
    assert "9e9a5c02" in body  # which commit
    assert "success" in body  # what the gate claimed
    assert "step-continue-on-error" in body  # what is wrong, by name
    assert "actions/runs/1" in body  # the receipts


# ---------------------------------------------------------------------------
# main: the seam the workflow actually calls
# ---------------------------------------------------------------------------


def _invoke(
    tmp_path: Path, workflow_src: str, check_runs: Any, audit_result: str
) -> dict[str, str]:
    """Drive `main()` exactly as the workflow does, and return the outputs it emitted."""
    import json

    from xbrain.gate_audit import main

    workflow = tmp_path / "quality.yml"
    workflow.write_text(workflow_src, encoding="utf-8")
    runs = tmp_path / "check-runs.json"
    runs.write_text(json.dumps(check_runs), encoding="utf-8")
    body = tmp_path / "body.md"
    outputs = tmp_path / "outputs.txt"
    outputs.touch()

    main(
        [
            "--check-runs", str(runs),
            "--workflow", str(workflow),
            "--audit-result", audit_result,
            "--sha", _SHA,
            "--repo", "VGonPa/xbrain",
            "--run-url", "https://github.com/VGonPa/xbrain/actions/runs/1",
            "--body-out", str(body),
            "--github-output", str(outputs),
        ]
    )  # fmt: skip

    parsed = dict(
        line.split("=", 1)
        for line in outputs.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    if body.is_file():
        parsed["body"] = body.read_text(encoding="utf-8")
    return parsed


def test_main_reports_a_lying_gate(tmp_path: Path) -> None:
    """End to end: gate said success, honest execution failed -> file the issue."""
    out = _invoke(tmp_path, _HEALTHY, {"check_runs": [_run("quality", "success")]}, "failure")
    assert out["verdict"] == "lying"
    assert out["file_issue"] == "true"
    assert "9e9a5c02" in out["body"]


def test_main_is_quiet_on_a_healthy_repo(tmp_path: Path) -> None:
    """The everyday case must produce no issue and no noise."""
    out = _invoke(tmp_path, _HEALTHY, {"check_runs": [_run("quality", "success")]}, "success")
    assert out["verdict"] == "clean"
    assert out["file_issue"] == "false"


def test_main_is_quiet_when_develop_is_merely_red(tmp_path: Path) -> None:
    """Consistent red belongs to the push-triggered alert, not to this one."""
    out = _invoke(tmp_path, _HEALTHY, {"check_runs": [_run("quality", "failure")]}, "failure")
    assert out["verdict"] == "consistent_red"
    assert out["file_issue"] == "false"


def test_main_files_a_disarmed_gate_that_still_passes(tmp_path: Path) -> None:
    """Static half, end to end: booby-trap planted on a green day is still filed."""
    gagged = _HEALTHY.replace(
        "        run: bash scripts/check.sh",
        "        run: bash scripts/check.sh\n        continue-on-error: true",
    )
    out = _invoke(tmp_path, gagged, {"check_runs": [_run("quality", "success")]}, "success")
    assert out["file_issue"] == "true"
    assert "step-continue-on-error" in out["body"]


@pytest.mark.parametrize("result", ["cancelled", "skipped", ""])
def test_a_gate_run_we_never_OBSERVED_is_never_called_a_liar(tmp_path: Path, result: str) -> None:
    """`needs.<job>.result` is not a boolean. Reading it as one manufactures a false accusation.

    A GitHub job resolves to `success`, `failure`, `cancelled` or `skipped`. If the
    gate-execution job is cancelled — a killed runner, a cancelled workflow, a spot-instance
    eviction — then `result == 'cancelled'`, and the naive `passed = (result == 'success')`
    reads that as *the audit failed*. Against a healthy green gate that yields the verdict
    LYING, and the auditor files a public issue accusing the repo of shipping unverified code
    because a VM died.

    An auditor that cries wolf is worse than no auditor: it burns exactly the credibility it
    needs on the one day it is right. We did not OBSERVE the gate, so we say nothing.
    """
    out = _invoke(tmp_path, _HEALTHY, {"check_runs": [_run("quality", "success")]}, result)
    assert out["verdict"] == "inconclusive"
    assert out["file_issue"] == "false"


def test_only_a_positively_clean_audit_stands_down_the_alarm(tmp_path: Path) -> None:
    """Closing an open "the gate is LYING" issue is a claim, and it needs positive evidence.

    `file_issue == false` is NOT that evidence: it is also false when the audit was
    inconclusive, or when develop is merely red. Auto-closing on those would let a cancelled
    runner silently retract a live, correct accusation — the alarm would disarm itself the
    moment CI got flaky. Only a verdict of CLEAN with zero violations may stand it down.
    """
    clean = _invoke(tmp_path, _HEALTHY, {"check_runs": [_run("quality", "success")]}, "success")
    assert clean["clean"] == "true"

    for result, runs in (
        ("cancelled", [_run("quality", "success")]),  # never observed
        ("failure", [_run("quality", "failure")]),  # merely red
        ("success", []),  # gate never reported
    ):
        out = _invoke(tmp_path, _HEALTHY, {"check_runs": runs}, result)
        assert out["file_issue"] == "false", "must not accuse"
        assert out["clean"] == "false", "and must not exonerate either"
