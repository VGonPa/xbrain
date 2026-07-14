# tests/test_gate_audit_workflow.py
"""The auditor's own shape. Who audits the auditor? This file — and only this file.

`gate-audit.yml` is the last layer of the CI defence, and it is defended by exactly one
property, which is easy to destroy by accident while "tidying up":

    A `schedule`-triggered workflow runs from the DEFAULT BRANCH's version of the file.

This repo's default branch is `main` (verified against the API, 2026-07-14). So the auditor's
definition lives on `main`, out of reach of any PR into `develop` — which is precisely the
thing it audits. A PR cannot gag the workflow that is watching it. That is the fail-closed
property, and it is the same one `pull_request_target` would give us WITHOUT the fork-PR RCE
surface that made `pull_request_target` unacceptable on a public repo.

Everything below defends that property or the honesty of what runs under it:

* **Triggers are an ALLOWLIST, not a denylist.** Not "no `pull_request_target`" — that would
  pass a workflow someone added `issue_comment:` to. The assertion is that the trigger set is
  a SUBSET of {schedule, workflow_dispatch}: two events, neither of which a fork or an
  outsider can fire. Anything else, named or not, fails. A denylist of today's known-bad
  events would have to be updated every time GitHub ships a new one; an allowlist never does.

* **No `continue-on-error`, anywhere.** The auditor exists because that key can make a gate
  lie. An auditor that could be silenced the same way would be a comedy.

* **It EXECUTES, it does not merely read.** A static linter of `quality.yml` is not a judge;
  it is a second opinion from the same armchair. The audit must run `bash scripts/check.sh`
  itself, on develop's real tree, and compare the result it gets with the result GitHub
  published. Judge != party, and the judge must actually do the work.

* **The comparator is the AUDITOR's copy, not the audited branch's.** Subtle and load-bearing.
  The gate execution necessarily runs develop's `check.sh` (that IS the gate under audit —
  running our own copy would prove nothing about develop's). But the code that DECIDES whether
  the gate lied must come from `main`, or a PR into develop could simply edit the judge into
  saying "looks fine to me". So the audit job checks out its own tree (no `ref:`, hence the
  default branch) and reads develop's `quality.yml` through a SPARSE checkout — data, never
  code. This file asserts that split, because nothing else would notice if it collapsed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "gate-audit.yml"

# The only two events that fire this workflow. Both are trusted: `schedule` is fired by
# GitHub on a clock, `workflow_dispatch` requires write access to the repo. Neither can be
# triggered by a fork, an outside contributor, or the contents of a pull request.
_ALLOWED_EVENTS = frozenset({"schedule", "workflow_dispatch"})

# The audited branch, the gate script it must really execute, and the endpoint it must really
# ask. Spelled out so a failure message can say what was expected rather than "assert False".
_AUDITED_BRANCH = "develop"
_GATE_SCRIPT = "scripts/check.sh"
_CHECK_RUNS_ENDPOINT = "check-runs"


def _workflow() -> dict[Any, Any]:
    """Parse `gate-audit.yml`. A parse failure here IS a test failure: it would never run."""
    assert _WORKFLOW.is_file(), (
        f"{_WORKFLOW.name} does not exist.\n"
        f"\n"
        f"This is the only layer that can catch a `quality` gate which reports GREEN while "
        f"testing nothing. Deleting it restores the hole described in probes #121/#124/#125."
    )
    parsed = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), f"{_WORKFLOW.name} is not a YAML mapping."
    return parsed


def _triggers() -> Any:
    """The trigger block, surviving YAML's Norway problem (bare `on:` parses as `True`)."""
    workflow = _workflow()
    for key in (True, "on"):
        if key in workflow:
            return workflow[key]
    raise AssertionError(f"{_WORKFLOW.name} declares no trigger block at all")


def _event_names() -> set[str]:
    """Every event this workflow fires on, whatever shape the `on:` block is written in."""
    triggers = _triggers()
    if isinstance(triggers, str):
        return {triggers}
    if isinstance(triggers, list):
        return set(triggers)
    return set(triggers)


def _jobs() -> dict[str, dict[str, Any]]:
    """Every job, normalised to a mapping."""
    return {str(k): (v or {}) for k, v in (_workflow().get("jobs") or {}).items()}


def _all_steps() -> list[dict[str, Any]]:
    """Every step of every job, flattened."""
    return [step for job in _jobs().values() for step in (job.get("steps") or [])]


def _runs() -> str:
    """All shell in the workflow, concatenated — what the auditor actually executes."""
    return "\n".join(str(step.get("run", "")) for step in _all_steps())


# ---------------------------------------------------------------------------
# The fail-closed property: what can fire this, and what cannot
# ---------------------------------------------------------------------------


def test_the_audit_is_scheduled() -> None:
    """`schedule` is not a nicety — it is the entire security model.

    A scheduled workflow runs from the DEFAULT BRANCH's copy of the file. That, and only
    that, is why a PR into `develop` cannot edit the auditor that judges it. Swap this for
    `push` or `pull_request` and the auditor starts running from the PR's own HEAD — at
    which point it can be gagged by the very diff it is supposed to catch, and this whole
    layer becomes theatre.
    """
    assert "schedule" in _event_names(), (
        "`gate-audit.yml` declares no `schedule:` trigger.\n"
        "\n"
        "STOP. Scheduled runs execute the DEFAULT BRANCH's version of a workflow. That is "
        "the ONLY reason a PR into `develop` cannot silence this auditor. Any trigger that "
        "runs the PR's own copy of this file (`push`, `pull_request`) hands the accused "
        "control of the judge."
    )


def test_the_audit_can_be_run_on_demand() -> None:
    """`workflow_dispatch` is how you re-run the audit after fixing a lie, to close the issue.

    Without it, a fix waits up to 24h for the cron before the alert clears — and a stale
    alert is one people learn to scroll past.
    """
    assert "workflow_dispatch" in _event_names()


def test_no_fork_fireable_trigger_can_ever_run_the_audit() -> None:
    """An ALLOWLIST, deliberately — a denylist of today's bad events rots.

    `pull_request_target` would also give the fail-closed property (it too runs the base
    branch's copy), and it was DECLINED: it hands a write-scoped token to a workflow running
    in the context of a fork's PR, on a PUBLIC repo. That is the canonical fork-PR RCE
    surface. `schedule` gets the same guarantee for free, because no outsider can fire a
    clock.
    """
    events = _event_names()
    forbidden = events - _ALLOWED_EVENTS
    assert not forbidden, (
        f"`gate-audit.yml` fires on {sorted(forbidden)}, which is outside the allowlist "
        f"{sorted(_ALLOWED_EVENTS)}.\n"
        f"\n"
        f"This workflow holds `issues: write` and runs trusted code from the default branch. "
        f"It must be unreachable from anything a fork or an outside contributor can cause. "
        f"`pull_request_target` in particular is BANNED: base-branch write token + public "
        f"repo = fork-PR RCE. If you need a new trigger, prove no outsider can fire it, then "
        f"add it to the allowlist here — deliberately, not by accident."
    )


def test_the_cron_is_a_valid_five_field_expression() -> None:
    """A malformed cron does not error — the workflow simply never runs. Silent, permanent."""
    schedule = (_triggers() or {}).get("schedule")
    assert isinstance(schedule, list) and schedule, "`schedule:` must be a non-empty list."
    for entry in schedule:
        cron = str(entry.get("cron", ""))
        fields = cron.split()
        assert len(fields) == 5, (
            f"cron {cron!r} has {len(fields)} fields, not 5. GitHub silently never runs an "
            f"invalid schedule, so this auditor would simply cease to exist."
        )
        assert re.fullmatch(r"[\d*,\-/]+", "".join(fields)), (
            f"cron {cron!r} contains characters outside the POSIX 5-field syntax."
        )


# ---------------------------------------------------------------------------
# The auditor must not be gaggable by the trick it exists to catch
# ---------------------------------------------------------------------------


def test_the_auditor_declares_no_continue_on_error_anywhere() -> None:
    """The one key this whole layer exists because of. It must not appear in the auditor.

    `continue-on-error: true` on a step makes a FAILING step report SUCCESS. If the audit
    job carried it, the audit could fail to prove the gate is lying — and report success.
    The watchman must not be able to sleep through his own alarm.
    """
    offenders = [name for name, job in _jobs().items() if "continue-on-error" in job]
    assert not offenders, f"Jobs declaring `continue-on-error`: {offenders}. Delete it."

    bad_steps = [
        step.get("name", step.get("run", step.get("uses", "?")))
        for step in _all_steps()
        if "continue-on-error" in step
    ]
    assert not bad_steps, (
        f"Steps declaring `continue-on-error`: {bad_steps}.\n"
        f"\n"
        f"This is the exact key the audit exists to detect on `quality.yml`. An auditor that "
        f"swallows its own failures reports a clean bill of health for a gate it never "
        f"managed to check."
    )


# ---------------------------------------------------------------------------
# Judge != party, and the judge must EXECUTE, not read
# ---------------------------------------------------------------------------


def test_the_audit_actually_executes_the_real_gate() -> None:
    """Reading `quality.yml` is a second opinion. RUNNING `check.sh` is evidence.

    The static audit can only catch what someone thought to ban. Executing the gate and
    comparing the outcome catches the attack nobody has invented yet, because it never asks
    HOW the gate lied — only whether what it reported matches what is true.
    """
    assert _GATE_SCRIPT in _runs(), (
        f"No step in `gate-audit.yml` runs `{_GATE_SCRIPT}`.\n"
        f"\n"
        f"Without executing the gate itself, this workflow is a linter with a cron — it can "
        f"only ever find banned keywords, and both measured attacks (#121 gag, #124 wrong "
        f"tree) were designed to look fine to a linter."
    )


def test_the_audit_reads_what_the_gate_REPORTED() -> None:
    """Half the comparison. Without GitHub's own answer there is nothing to compare against."""
    assert _CHECK_RUNS_ENDPOINT in _runs(), (
        "No step queries the `check-runs` API. The audit compares what the gate REPORTED "
        "for develop's HEAD against what an honest execution FINDS. Drop the first half and "
        "you have merely re-run the tests: you would learn that develop is red, but never "
        "that the gate CLAIMED it was green."
    )


def test_the_gate_execution_job_audits_develop() -> None:
    """It must run the gate on the branch under audit, not on the auditor's own tree.

    Auditing `main` would be circular and useless: `main` is only ever reached THROUGH
    `develop`, so a gate weakened on `develop` is invisible from `main` until it has already
    been promoted.
    """
    assert _AUDITED_BRANCH in yaml.safe_dump(_workflow()), (
        f"`gate-audit.yml` never mentions `{_AUDITED_BRANCH}`. The audit must check out and "
        f"execute the gate on the branch it is auditing."
    )


def test_the_comparator_runs_from_the_auditors_own_tree() -> None:
    """THE subtlety. The judge's code must come from `main`, never from the accused branch.

    The gate EXECUTION runs develop's `check.sh` — it has to; that IS the gate under audit.
    But the code that decides *whether the gate lied* must not be develop's, or a PR could
    edit the judge as easily as it edits the gate, and we would be back where we started.

    Structurally: the audit job checks out its own tree with NO `ref:` (a scheduled run
    therefore gets the default branch), and reads develop's `quality.yml` through a separate
    checkout pinned to a `path:`. Data in, never code. This test asserts that split exists.
    """
    jobs = _jobs()
    audit_steps = [
        step
        for name, job in jobs.items()
        for step in (job.get("steps") or [])
        if "gate_audit" in str(step.get("run", ""))
    ]
    assert audit_steps, "No step runs the `gate_audit` comparator module."

    # The job holding the comparator must check out the auditor's own tree: a checkout with
    # no `ref:` and no `path:`, which on a scheduled run resolves to the default branch.
    audit_job = next(
        job
        for job in jobs.values()
        if any("gate_audit" in str(s.get("run", "")) for s in (job.get("steps") or []))
    )
    checkouts = [
        step
        for step in (audit_job.get("steps") or [])
        if "actions/checkout" in str(step.get("uses", ""))
    ]
    own_tree = [c for c in checkouts if not (c.get("with") or {}).get("ref")]
    assert own_tree, (
        "The audit job never checks out its OWN tree (a checkout with no `ref:`).\n"
        "\n"
        "On a scheduled run, a bare checkout resolves to the DEFAULT BRANCH — which is the "
        "only copy of the comparator a PR into `develop` cannot edit. If the comparator were "
        "run from the audited checkout instead, a PR could neuter the judge in the same diff "
        "that neuters the gate, and this entire layer would be decorative."
    )

    # Anything checked out FROM the audited branch must land in its own `path:`, so it can
    # never be mistaken for — or shadow — the auditor's own files.
    audited = [c for c in checkouts if (c.get("with") or {}).get("ref")]
    for checkout in audited:
        with_ = checkout.get("with") or {}
        assert with_.get("path"), (
            f"A checkout pinned to `ref: {with_.get('ref')!r}` declares no `path:`, so it "
            f"overwrites the auditor's own working tree — including the comparator. The "
            f"audited branch's code would then BE the judge."
        )


# ---------------------------------------------------------------------------
# Least privilege
# ---------------------------------------------------------------------------


def test_the_audit_may_file_an_issue() -> None:
    """Detection that cannot speak is detection into a void."""
    perms = [job.get("permissions") or {} for job in _jobs().values()]
    assert any(p.get("issues") == "write" for p in perms), (
        "No job declares `issues: write`, so the audit cannot file the alert when it catches "
        "the gate lying. A red run in the Actions tab that nobody is watching is not a "
        "control — that lesson cost 9m15s of red `develop` and two duplicate hotfix PRs."
    )


def test_the_audit_can_read_the_reported_check_run() -> None:
    """`checks: read` is what lets the built-in token see what the gate published."""
    perms = [job.get("permissions") or {} for job in _jobs().values()]
    assert any(p.get("checks") == "read" for p in perms)


def test_the_audit_never_takes_write_access_to_code() -> None:
    """It observes and it reports. It must never be able to change what it is auditing.

    An auditor with `contents: write` could, if compromised or simply buggy, rewrite the very
    gate it is judging. Read-only on code is what keeps its authority to accuse credible.
    """
    for name, job in _jobs().items():
        perms = job.get("permissions") or {}
        assert perms.get("contents", "read") == "read", (
            f"Job `{name}` takes `contents: {perms.get('contents')!r}`. The auditor must "
            f"never be able to write to the repository it audits."
        )
        assert "pull-requests" not in perms, (
            f"Job `{name}` takes `pull-requests` permission. The auditor files issues; it "
            f"does not touch PRs. Least privilege."
        )


def test_every_job_declares_its_permissions_explicitly() -> None:
    """An omitted `permissions:` block inherits the repo default, which may be write-all.

    Silence here is not "no permissions" — it is "whatever the repo settings happen to say",
    which can change under you without a diff to this file.
    """
    for name, job in _jobs().items():
        assert "permissions" in job, (
            f"Job `{name}` declares no `permissions:` block, so it inherits the repository "
            f"default — which is NOT guaranteed to be read-only and can be changed in "
            f"settings without any diff to this workflow. State the scopes explicitly."
        )


def test_the_audit_uses_no_secret_beyond_the_builtin_token() -> None:
    """No PAT, no bot identity, no new credential to rotate, revoke, or leak."""
    text = _WORKFLOW.read_text(encoding="utf-8")
    used = set(re.findall(r"secrets\.([A-Za-z_][A-Za-z0-9_]*)", text))
    assert used <= {"GITHUB_TOKEN"}, (
        f"`gate-audit.yml` references secrets beyond the built-in token: "
        f"{sorted(used - {'GITHUB_TOKEN'})}. A scheduled workflow with a PAT is a standing "
        f"credential with nobody watching it. The built-in GITHUB_TOKEN is scoped to this "
        f"run and dies with it."
    )
