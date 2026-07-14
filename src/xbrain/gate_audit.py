"""Does the `quality` gate actually DO what it REPORTS? The comparison, and the alarm.

Called by `.github/workflows/gate-audit.yml`. Everything here is pure: it parses JSON that
`gh` already fetched and YAML already on disk, decides, and writes a verdict. It performs no
network I/O and shells out to nothing, so it can be unit-tested — which a workflow cannot.

WHY A SEPARATE, SCHEDULED AUDITOR EXISTS AT ALL
-----------------------------------------------
`tests/test_ci_workflow.py` already forbids `continue-on-error` on the gate. It runs INSIDE
`scripts/check.sh`, which is what the gate runs. So on a PR that adds `continue-on-error:
true` to the `Quality gate` step: the guard test fires correctly, `check.sh` exits 1 — and
the same `continue-on-error` swallows THAT failure too. Measured on this repo (probes #121,
#125): the step reports `success`, the job reports `success`, the check run reports
`SUCCESS`, `mergeStateStatus` is `CLEAN`. Every API surface says green. The failure exists
only as text in a log nobody opens.

The alarm is inside the soundproofed room, because a PR's CI runs the PR's own HEAD copy of
the workflow. No in-repo test can escape that. Something outside the room has to listen.

This is also not only an attack. `continue-on-error: true` on a failing step is THE canonical
"make flaky CI stop blocking merges" edit. An agent told *"the build keeps failing, make it
pass"* reaches for it in good faith and produces a permanently dead gate wearing a green
badge. No adversary required.

THE TWO HALVES, AND WHY BOTH
----------------------------
* **Execution** (`classify`) asks the OUTCOME question: GitHub says the gate concluded
  `success` for develop@SHA; an honest run of that same gate on that same commit FAILED;
  therefore the gate is lying. It names no keyword, so it catches the gag (#121/#125), the
  wrong-tree `checkout ref:` (#124), and whatever is invented next — but only once the trap
  has actually gone off.

* **Static** (`audit_workflow_source`) asks the SYNTAX question, from outside. It catches a
  disarmed gate on a day when develop happens to be green, which execution cannot see:
  reported green, audit green, no discrepancy — and the gate is already dead, waiting for the
  next red commit to wave through.

Neither is redundant. Execution proves THAT something is wrong; static says WHAT.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

#: The job id in `quality.yml`, and the status check branch protection requires by that
#: exact name on `develop` and `main` (verified against the API, 2026-07-14).
REQUIRED_CHECK = "quality"

#: The gate itself. A `quality` job that does not run this is a decoration.
GATE_SCRIPT = "scripts/check.sh"

#: Check-run conclusions that do NOT block a merge.
#:
#: This is the subtle one, and getting it wrong would blind the whole audit. The question is
#: never "did the check say the literal string `success`" — it is "would this have stopped a
#: merge". GitHub treats `neutral` and `skipped` as satisfying a required status check, and a
#: job killed with `if: false` publishes its check run as `skipped`. Compare against
#: `"success"` alone and `if: false` walks straight past you.
NON_BLOCKING_CONCLUSIONS = frozenset({"success", "neutral", "skipped"})


class Verdict(StrEnum):
    """The outcome of comparing what the gate REPORTED against what it actually DOES."""

    #: Reported non-blocking, and an honest execution agrees. The boring, normal case.
    CLEAN = "clean"

    #: Reported non-blocking, honest execution FAILED. The gate is lying. This is the one.
    LYING = "lying"

    #: Reported blocking, honest execution failed. develop is red and the gate says so.
    #: Not a lie — and already alarmed by the `push`-triggered run. Do not double-file.
    CONSISTENT_RED = "consistent_red"

    #: Reported blocking, honest execution passed. Fail-CLOSED: merges are blocked, nothing
    #: unsafe can land. Usually a flake or a since-fixed transient. Print it; never file it.
    REPORTED_RED_AUDIT_GREEN = "reported_red_audit_green"

    #: No completed `quality` check run for this commit — nothing to compare against. The
    #: static audit still runs, and it is what catches a gate that never ran at all.
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class Reported:
    """What GitHub published for the `quality` check on a given commit."""

    found: bool
    status: str | None
    conclusion: str | None

    @property
    def completed(self) -> bool:
        """Has the gate finished? An in-flight run has concluded nothing."""
        return self.found and self.status == "completed"

    @property
    def blocks_merge(self) -> bool:
        """Would this conclusion have stopped a merge? The only question worth asking."""
        return self.completed and self.conclusion not in NON_BLOCKING_CONCLUSIONS


@dataclass(frozen=True)
class Violation:
    """A specific, named way `quality.yml` has been disarmed."""

    code: str
    detail: str


def reported_gate(payload: Any, check_name: str = REQUIRED_CHECK) -> Reported:
    """Read the `quality` check run out of a `/commits/{sha}/check-runs` response.

    Takes the MOST RECENT run by `started_at`, not the first in the list: a re-run supersedes
    the original and branch protection honours the latest conclusion. The API does not
    contractually order the array, so trusting its order would risk comparing an honest
    execution against a stale, superseded verdict.
    """
    runs = [r for r in (payload or {}).get("check_runs", []) if r.get("name") == check_name]
    if not runs:
        return Reported(found=False, status=None, conclusion=None)
    latest = max(runs, key=lambda r: str(r.get("started_at") or ""))
    return Reported(
        found=True,
        status=str(latest.get("status") or ""),
        conclusion=latest.get("conclusion"),
    )


def classify(reported: Reported, audit_passed: bool) -> Verdict:
    """Compare what the gate said with what it actually does. The heart of the layer."""
    if not reported.completed:
        return Verdict.INCONCLUSIVE
    if reported.blocks_merge:
        return Verdict.REPORTED_RED_AUDIT_GREEN if audit_passed else Verdict.CONSISTENT_RED
    return Verdict.CLEAN if audit_passed else Verdict.LYING


def should_file_issue(verdict: Verdict, violations: Sequence[Violation]) -> bool:
    """Only two things are worth waking a human for: the gate lied, or the gate is disarmed.

    Everything else — consistent red, a flaky over-report, an in-flight run — is either
    already alarmed elsewhere or fail-closed. An alert that fires on safe states is an alert
    people learn to close unread, which is exactly how the Actions tab stopped being watched.
    """
    return verdict is Verdict.LYING or bool(violations)


def should_stand_down(verdict: Verdict, violations: Sequence[Violation]) -> bool:
    """May a later audit CLOSE an open alarm? Only on positive evidence of health.

    Deliberately not `not should_file_issue(...)`. That would also be true when the audit was
    INCONCLUSIVE (a cancelled runner, an in-flight gate) or when develop is merely red —
    neither of which disproves a live "the gate is LYING" accusation. Closing is itself a
    CLAIM ("I checked, it is fine"), and it needs evidence, not merely the absence of a new
    accusation. Otherwise the first flaky run after a real lie would quietly retract it.
    """
    return verdict is Verdict.CLEAN and not violations


def audit_observed(job_result: str) -> bool | None:
    """Did the gate-execution job actually RUN? None when we cannot say.

    `needs.<job>.result` is one of `success`, `failure`, `cancelled`, `skipped` — NOT a
    boolean. Collapsing it with `passed = (result == "success")` silently reads `cancelled`
    as *the gate failed*, which against a healthy green gate yields the verdict LYING. The
    auditor would then file a public issue accusing the repository of merging unverified code
    because a runner was evicted. We report only what we observed.
    """
    if job_result == "success":
        return True
    if job_result == "failure":
        return False
    return None


# ---------------------------------------------------------------------------
# Static audit of the audited branch's quality.yml
#
# These rules deliberately restate what `tests/test_ci_workflow.py` already asserts. That is
# not duplication for its own sake: those tests run INSIDE the gate they defend, so a gagged
# gate silences them. These run from the auditor's own tree on the default branch, against
# the audited branch's file, read as DATA. Judge != party. The rules must exist in both
# places precisely because one of the two can be muzzled.
# ---------------------------------------------------------------------------


def _triggers(workflow: dict[Any, Any]) -> Any:
    """The `on:` block, surviving YAML's Norway problem.

    In YAML 1.1 a bare `on:` key parses as the BOOLEAN `True`, not the string `"on"`. GitHub
    accepts `on:` and `"on":` identically, so an auditor that knows only one spelling either
    misses every filter written under the other, or screams at a behaviour-preserving
    reformat. Look under both.
    """
    for key in (True, "on"):
        if key in workflow:
            return workflow[key] or {}
    return {}


def _gate_job(workflow: dict[Any, Any]) -> dict[str, Any] | None:
    """The job whose check run branch protection requires, or None if it is gone."""
    job = (workflow.get("jobs") or {}).get(REQUIRED_CHECK)
    return job or None if isinstance(job, dict) else None


def _check_triggers(workflow: dict[Any, Any]) -> list[Violation]:
    """A path filter SKIPS the workflow — and a skipped workflow leaves the check Pending.

    Not green, not red: never published. Every PR then waits forever on a check that will
    never report, and `enforce_admins` means nobody can override it — including the PR that
    would undo the filter. Hard lockout.
    """
    triggers = _triggers(workflow)
    if not isinstance(triggers, dict):
        return []
    out = []
    for event in ("push", "pull_request"):
        config = triggers.get(event)
        if not isinstance(config, dict):
            continue
        for key in ("paths", "paths-ignore"):
            if key in config:
                out.append(
                    Violation(
                        "path-filter",
                        f"`{event}` declares `{key}: {config[key]!r}`. A filtered-out run is "
                        f"SKIPPED, so the required `{REQUIRED_CHECK}` check is never "
                        f"published and every PR hangs Pending forever.",
                    )
                )
    return out


def _check_job_identity(workflow: dict[Any, Any]) -> list[Violation]:
    """GitHub names the check run after the job's `name:`, else its id. Both can brick it."""
    jobs = workflow.get("jobs") or {}
    if REQUIRED_CHECK not in jobs:
        return [
            Violation(
                "gate-job-missing",
                f"No job with id `{REQUIRED_CHECK}` (found: {sorted(map(str, jobs))}). Branch "
                f"protection requires a check named exactly `{REQUIRED_CHECK}`; without it "
                f"the check never appears and EVERY MERGE BLOCKS, permanently.",
            )
        ]
    job = _gate_job(workflow) or {}
    name = str(job.get("name", REQUIRED_CHECK))
    if name != REQUIRED_CHECK:
        return [
            Violation(
                "check-renamed",
                f"The `{REQUIRED_CHECK}` job declares `name: {name}`, so its check run is "
                f"published as `{name}` — not `{REQUIRED_CHECK}`. The required check never "
                f"appears and every merge blocks, permanently.",
            )
        ]
    return []


def _check_job_guards(workflow: dict[Any, Any]) -> list[Violation]:
    """`continue-on-error` and `if:` at JOB level. One is fail-closed; the other is not."""
    job = _gate_job(workflow)
    if job is None:
        return []
    out = []
    if "continue-on-error" in job:
        out.append(
            Violation(
                "job-continue-on-error",
                "The `quality` JOB declares `continue-on-error`. Measured (probe #119) this "
                "one is fail-closed — the check run still reports FAILURE and the merge is "
                "still BLOCKED. It is banned anyway because it is ONE INDENT from the step "
                "form, which is catastrophic, and no reviewer should have to count indents "
                "to tell a dead gate from a live one.",
            )
        )
    if "if" in job:
        out.append(
            Violation(
                "job-conditional",
                f"The `quality` job declares `if: {job['if']!r}`. A SKIPPED JOB REPORTS "
                f"SUCCESS — GitHub's words. The required check goes green having run nothing.",
            )
        )
    return out


def _label(step: dict[str, Any]) -> str:
    """A human-readable handle for a step, for the failure message."""
    return str(step.get("name") or step.get("uses") or str(step.get("run", "?"))[:40])


def _check_step_gag(steps: list[dict[str, Any]]) -> list[Violation]:
    """`continue-on-error` on a step: THE gag. The gate runs, fails, and reports SUCCESS."""
    return [
        Violation(
            "step-continue-on-error",
            f"Step `{_label(step)}` declares `continue-on-error`. THE gag (probes "
            f"#121/#125): the gate still runs, still FAILS — and the check run reports "
            f"SUCCESS anyway. `mergeStateStatus` reads CLEAN. Every API surface says green "
            f"while nothing was verified.",
        )
        for step in steps
        if "continue-on-error" in step
    ]


def _check_checkout_ref(steps: list[dict[str, Any]]) -> list[Violation]:
    """An explicit `ref:`: the gate passes honestly, against a tree nobody asked about."""
    return [
        Violation(
            "checkout-ref",
            f"Step `{_label(step)}` pins `ref: {(step.get('with') or {})['ref']!r}`. The gate "
            f"then runs, passes honestly, reports SUCCESS — against the WRONG TREE (probe "
            f"#124: `1088 passed`, where develop's suite has ~1600). The commit that "
            f"triggered the run is never examined at all.",
        )
        for step in steps
        if "actions/checkout" in str(step.get("uses", "")) and (step.get("with") or {}).get("ref")
    ]


def _check_gate_step(steps: list[dict[str, Any]]) -> list[Violation]:
    """The gate step must exist and must be unconditional. Either way out is a silent green."""
    gate = next((s for s in steps if GATE_SCRIPT in str(s.get("run", ""))), None)
    if gate is None:
        return [
            Violation(
                "gate-step-missing",
                f"No step runs `{GATE_SCRIPT}`. The check run branch protection trusts no "
                f"longer executes the gate: it reports GREEN having verified nothing.",
            )
        ]
    if "if" in gate:
        return [
            Violation(
                "gate-step-conditional",
                f"The gate step declares `if: {gate['if']!r}`. Skip it and the job still "
                f"SUCCEEDS — the required check goes green having run no checks at all.",
            )
        ]
    return []


def _check_steps(workflow: dict[Any, Any]) -> list[Violation]:
    """The step-level kills: the gag, the wrong tree, the skipped gate, the missing gate."""
    job = _gate_job(workflow)
    if job is None:
        return []
    steps = [s for s in (job.get("steps") or []) if isinstance(s, dict)]
    out: list[Violation] = []
    for check in (_check_step_gag, _check_checkout_ref, _check_gate_step):
        out.extend(check(steps))
    return out


def audit_workflow_source(path: Path) -> list[Violation]:
    """Statically audit the audited branch's `quality.yml` for every known way to disarm it.

    Read as DATA, from the auditor's own tree. Nothing here is executed.
    """
    if not path.is_file():
        return [
            Violation(
                "workflow-missing",
                f"`{path.name}` does not exist on the audited branch. Deleting the workflow "
                f"is the simplest disarm of all, and it must never read as 'clean'.",
            )
        ]
    try:
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [Violation("workflow-unparseable", f"`{path.name}` is not valid YAML: {exc}")]
    if not isinstance(workflow, dict):
        return [Violation("workflow-unparseable", f"`{path.name}` is not a YAML mapping.")]

    violations: list[Violation] = []
    for check in (_check_triggers, _check_job_identity, _check_job_guards, _check_steps):
        violations.extend(check(workflow))
    return violations


# ---------------------------------------------------------------------------
# The alarm
# ---------------------------------------------------------------------------

_HEADLINE = {
    Verdict.LYING: "`develop`'s quality gate is LYING",
    Verdict.CLEAN: "`develop`'s quality gate is DISARMED",
    Verdict.CONSISTENT_RED: "`develop`'s quality gate is DISARMED",
    Verdict.REPORTED_RED_AUDIT_GREEN: "`develop`'s quality gate is DISARMED",
    Verdict.INCONCLUSIVE: "`develop`'s quality gate is DISARMED",
}


def issue_title(verdict: Verdict) -> str:
    """Title of the alert. Stable per verdict class, because it is the idempotency anchor."""
    return f"🚨 {_HEADLINE[verdict]}"


def render_issue_body(
    *,
    verdict: Verdict,
    violations: Sequence[Violation],
    sha: str,
    reported: str | None,
    audit_passed: bool,
    repo: str,
    run_url: str,
) -> str:
    """The alarm, written to be actionable by someone who has never read this module."""
    short = sha[:8]
    commit_url = f"https://github.com/{repo}/commit/{sha}"
    said = f"`{reported}`" if reported else "_never reported_"
    found = "PASSED" if audit_passed else "**FAILED**"

    lines = [
        f"The scheduled audit disagrees with the `{REQUIRED_CHECK}` gate on `develop`.",
        "",
        "| | |",
        "|---|---|",
        f"| **Commit** | [`{short}`]({commit_url}) |",
        f"| **The gate REPORTED** | {said} |",
        f"| **An honest re-run FOUND** | {found} |",
        f"| **Verdict** | `{verdict.value}` |",
        f"| **Audit run** | [logs]({run_url}) |",
        "",
    ]

    if verdict is Verdict.LYING:
        lines += [
            "### The gate is reporting green on code it did not verify",
            "",
            f"This audit checked out `develop@{short}`, ran the real gate — "
            f"`bash {GATE_SCRIPT}` — in a job with **no** `continue-on-error` and **no** "
            f"pinned `ref:`, and it **failed**. GitHub nonetheless published {said} for the "
            f"required `{REQUIRED_CHECK}` check on that exact commit.",
            "",
            "Both known causes produce precisely this signature:",
            "",
            "* `continue-on-error: true` on the **gate step** — the gate runs, fails, and the "
            "check run reports `SUCCESS` anyway (probes #121, #125). Every API surface, "
            "including `mergeStateStatus: CLEAN`, says the code is fine.",
            "* `checkout` with an explicit **`ref:`** — the gate passes honestly, having "
            "examined a tree nobody asked about (probe #124).",
            "",
            f"**Anything merged into `develop` since {short} passed a check that was not "
            f"checking.** Treat those merges as unreviewed by CI until the gate is restored "
            f"and re-run.",
            "",
        ]

    if violations:
        lines += ["### What is wrong with `quality.yml`, by name", ""]
        lines += [f"- **`{v.code}`** — {v.detail}" for v in violations]
        lines += [""]
    elif verdict is Verdict.LYING:
        lines += [
            "### The static audit found nothing",
            "",
            "The execution check proves the gate is lying, but no *known* banned pattern "
            "explains it. This is a mechanism nobody has catalogued yet — read the diff of "
            f"`.github/workflows/{REQUIRED_CHECK}.yml` and `{GATE_SCRIPT}` by hand, and add "
            "the new pattern to `audit_workflow_source` once you find it.",
            "",
        ]

    lines += [
        "---",
        "",
        "<sub>Filed by the scheduled gate audit "
        "(`.github/workflows/gate-audit.yml`), which runs from the **default branch** and "
        "therefore cannot be edited by the pull requests it audits. A later clean audit "
        "closes this issue automatically — re-run the workflow via `workflow_dispatch` once "
        "you have pushed the fix, rather than waiting for tomorrow's cron.</sub>",
    ]
    return "\n".join(lines)


def _emit(path: str | None, **outputs: str) -> None:
    """Append `key=value` lines to a GitHub Actions output file, if we were given one."""
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        for key, value in outputs.items():
            handle.write(f"{key}={value}\n")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Wire up the inputs the workflow hands us."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-runs", required=True, type=Path, help="check-runs API JSON")
    parser.add_argument("--workflow", required=True, type=Path, help="audited quality.yml")
    parser.add_argument("--audit-result", required=True, help="result of the gate-execution job")
    parser.add_argument("--sha", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--body-out", type=Path, required=True)
    parser.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Compare, decide, and hand the workflow a verdict. Always exits 0.

    The workflow — not this module — is what turns a `lying` verdict into a red run, and it
    does so only AFTER the issue has been filed. Exiting non-zero here would kill the job
    before it could raise the alarm, which is the one failure this whole layer cannot afford.
    """
    args = _parse_args(argv)

    payload = json.loads(args.check_runs.read_text(encoding="utf-8"))
    reported = reported_gate(payload)
    observed = audit_observed(args.audit_result)

    # A gate we never watched run cannot be caught lying. `cancelled`/`skipped` means the
    # execution job never delivered a verdict, so neither do we.
    verdict = Verdict.INCONCLUSIVE if observed is None else classify(reported, observed)
    violations = audit_workflow_source(args.workflow)
    file_issue = should_file_issue(verdict, violations)

    print(f"commit          : {args.sha}")
    print(f"gate REPORTED   : {reported.conclusion or '(no completed run)'}")
    print(
        f"audit FOUND     : {'pass' if observed else 'FAIL' if observed is False else '(not observed: ' + args.audit_result + ')'}"
    )  # noqa: E501
    print(f"verdict         : {verdict.value}")
    for violation in violations:
        print(f"violation       : {violation.code} — {violation.detail}")
    print(f"file issue      : {file_issue}")

    body = render_issue_body(
        verdict=verdict,
        violations=violations,
        sha=args.sha,
        reported=reported.conclusion,
        audit_passed=bool(observed),
        repo=args.repo,
        run_url=args.run_url,
    )
    args.body_out.write_text(body, encoding="utf-8")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"## Gate audit — `{verdict.value}`\n\n{body}\n")

    _emit(
        args.github_output,
        verdict=verdict.value,
        file_issue="true" if file_issue else "false",
        clean="true" if should_stand_down(verdict, violations) else "false",
        title=issue_title(verdict),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
