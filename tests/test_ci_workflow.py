# tests/test_ci_workflow.py
"""The quality gate must run on the MERGE RESULT, not only on the PR head.

CLAUDE.md rule 4 — "A green PR against a moving `develop` is not a green `develop`" — was
paid for in blood on 2026-07-14. PR #94 changed `_source_text(item)` to
`_source_text(item, target)`; PR #97 added a test calling `_source_text(item)`. Both were
green on their own branches, and there was zero textual conflict, so git merged both
happily — and `develop` was RED. CI never said a word, because CI only ran on
`pull_request`: the one commit nobody ever tested was the merge commit itself. A human
found it by hand, hours later (PR #103).

Until this test existed, rule 4 lived only as prose that a human had to remember to obey.
This is the machine enforcing it. Every push to `develop`/`main` IS a merge result, so the
workflow must fire on `push` to both — while KEEPING the `pull_request` runs that catch a
bad branch before it ever lands.

The assertions are on the RULE, not on the file's wording: they ask "would the gate
actually run on a push to develop?". Reformatting the YAML while preserving the behaviour
stays green; any edit that stops the gate running on a merge result goes red.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

_WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "quality.yml"

# The branches that are merged INTO. A push to one of these is a merge result.
_GATED_BRANCHES = ("develop", "main")


def _triggers() -> Any:
    """Return the workflow's trigger block, surviving YAML's Norway problem.

    In YAML 1.1 the bare key `on:` is the BOOLEAN `True`, not the string `"on"` — so
    `yaml.safe_load(...)["on"]` raises `KeyError` on a perfectly valid workflow. Quoting
    the key in the file (`"on":`) would make it a string instead. GitHub Actions accepts
    both spellings, so this looks the key up under both and the test will not silently
    break the day someone quotes — or unquotes — it.
    """
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    for key in (True, "on"):
        if key in workflow:
            return workflow[key]
    raise AssertionError(f"{_WORKFLOW.name} declares no trigger block at all")


def _runs_on(event: str, branch: str) -> bool:
    """Would the gate actually run for `event` on `branch`?

    Handles every shape GitHub accepts for the trigger block: a bare string (`on: push`),
    a list (`on: [push, pull_request]`), or a mapping with optional filters. An absent
    `branches:` filter means GitHub runs the event on EVERY branch — which covers `branch`
    too, so that is a pass, not a miss. Asserting the effect rather than the syntax is what
    keeps this a rule test instead of a spelling test.
    """
    triggers = _triggers()
    if isinstance(triggers, str):  # `on: push`
        return triggers == event
    if isinstance(triggers, list):  # `on: [push, ...]` — no branch filter possible
        return event in triggers
    if event not in triggers:
        return False
    config = triggers[event] or {}  # `push:` with no body == run on every branch
    branches = config.get("branches")
    return branches is None or branch in branches


@pytest.mark.parametrize("branch", _GATED_BRANCHES)
def test_gate_runs_on_push_to_gated_branch(branch: str) -> None:
    """A push to develop/main IS a merge result — the gate must run on it (rule 4)."""
    assert _runs_on("push", branch), (
        f"quality.yml does not run on `push` to `{branch}`, so the merge commit is never "
        f"tested. Two PRs, each green on its own branch and with zero textual conflict, "
        f"can still merge into a RED `{branch}` while CI stays silent — this is exactly "
        f"how #103 happened. See CLAUDE.md rule 4."
    )


@pytest.mark.parametrize("branch", _GATED_BRANCHES)
def test_gate_still_runs_on_pull_request(branch: str) -> None:
    """Merge-result coverage is ADDED to PR coverage, never swapped for it."""
    assert _runs_on("pull_request", branch), (
        f"quality.yml no longer runs on `pull_request` to `{branch}`. Testing the merge "
        f"result does not replace testing the PR head: without this, a broken branch is "
        f"only caught AFTER it has already landed on `{branch}`."
    )
