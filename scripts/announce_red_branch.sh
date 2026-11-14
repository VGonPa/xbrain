#!/bin/bash
# Announce (or resolve) a RED gated branch by filing a GitHub issue.
#
# WHY THIS EXISTS
#
# A `push`-triggered gate run that goes red blocks nothing: the commit is already ON the
# branch. Branch protection guards the door, and this run happens after the horse has left.
# So if the red run does not SPEAK, it detects into a void.
#
# It did exactly that on 2026-07-14. `develop` was red for 9m15s; three commits took the red
# commit as their parent; and two agents opened DUPLICATE hotfix PRs 31 seconds apart, each
# having rediscovered the same breakage by hand. Nobody was watching the Actions tab, because
# watching the Actions tab is not a control.
#
# IDENTITY: uses the built-in GITHUB_TOKEN (`github-actions[bot]`) with `issues: write`.
# No PAT, no bot account, no second human. Do not "upgrade" this to a PAT.
#
# IDEMPOTENCY: one open issue per gated branch, keyed by the label `ci-red-<branch>`. If the
# branch is red across three pushes that is ONE issue with three comments, not three issues.
#
# Getting that right is harder than it looks, and the naive version was WRONG. Every GitHub
# issue LIST endpoint — label filter, search, and the plain REST list alike — is EVENTUALLY
# consistent. Measured on this repo: an issue created and then listed immediately does not
# appear for several seconds, while fetching it by number returns it at once. So a bare
# list-then-create races against its own write: two runs inside the lag window both conclude
# "no issue exists" and both file one. This is not hypothetical — the first version of this
# script filed issues #113 AND #114 for a single branch when `open` ran twice.
#
# Hence two defences, because neither alone suffices:
#   1. RETRY the lookup with a bounded backoff (open mode only), which absorbs the ordinary
#      index lag — the common case.
#   2. RECONCILE on every run: if more than one open issue carries the key, keep the oldest
#      and close the rest as duplicates. A true race can still slip past the retry; this
#      makes the state CONVERGE on the next run instead of compounding. Note the shape of
#      the fix — the system heals itself rather than trusting a check to have been right.
#
# Usage:  announce_red_branch.sh open|close
#   open  - branch is red: file the issue, or add a comment if it is already open
#   close - branch is green again: comment the fixing sha and close the issue, if any
#
# Env (all supplied by the workflow from trusted `github.*` context, never user input):
#   GH_TOKEN, REPO, BRANCH, SHA, RUN_URL

set -euo pipefail

MODE="${1:?usage: announce_red_branch.sh open|close}"
# Validate the mode BEFORE touching the API: a typo should fail instantly and loudly, not
# after two network round-trips, and certainly not by falling through to a no-op that leaves
# a red branch unannounced.
case "$MODE" in
open | close) ;;
*)
    echo "Unknown mode '${MODE}' (expected 'open' or 'close')" >&2
    exit 2
    ;;
esac

: "${GH_TOKEN:?GH_TOKEN is required}"
: "${REPO:?REPO is required}"
: "${BRANCH:?BRANCH is required}"
: "${SHA:?SHA is required}"
: "${RUN_URL:?RUN_URL is required}"

SHORT_SHA="${SHA:0:8}"
LABEL="ci-red-${BRANCH}"
COMMIT_URL="https://github.com/${REPO}/commit/${SHA}"

# The label is the idempotency key, so it must exist before it can be filtered on. Creating
# it is safe to repeat; it already existing is the normal case, not an error.
gh label create "$LABEL" \
    --repo "$REPO" \
    --color B60205 \
    --description "The ${BRANCH} branch is failing its quality gate" \
    >/dev/null 2>&1 || true

# Every open issue carrying the key, oldest first. Oldest-first matters: the oldest is the
# canonical one, so reconciliation is deterministic no matter which run observes the mess.
open_issues() {
    gh issue list \
        --repo "$REPO" \
        --label "$LABEL" \
        --state open \
        --limit 50 \
        --json number \
        --jq '[.[].number] | sort | .[]'
}

# Lookup that tolerates the list-endpoint lag. Only used by `open`, where a false "none"
# costs a duplicate issue. `close` skips it: by the time a branch goes green, the red run
# that filed the issue is a full gate-run (~73s) in the past, so the index has long settled,
# and paying 10s of backoff on every green push to learn "nothing to close" is pure waste.
open_issues_with_retry() {
    local found attempt
    for attempt in 1 2 3 4 5; do
        found=$(open_issues || true)
        if [ -n "$found" ]; then
            echo "$found"
            return 0
        fi
        [ "$attempt" -lt 5 ] && sleep 2
    done
    return 0 # genuinely none: the retries expired without ever seeing one
}

case "$MODE" in
open)
    EXISTING=$(open_issues_with_retry)
    if [ -n "$EXISTING" ]; then
        # Still red, on a new commit. Update the canonical issue — never open a second one.
        CANONICAL=$(echo "$EXISTING" | head -1)
        gh issue comment "$CANONICAL" --repo "$REPO" --body \
            "Still RED after [\`${SHORT_SHA}\`](${COMMIT_URL}) — [failing run](${RUN_URL})"
        echo "Updated existing red-${BRANCH} issue #${CANONICAL}"

        # RECONCILE: a race (or an older buggy run) may have left duplicates. Fold them into
        # the canonical issue so the branch converges on exactly one open alert.
        echo "$EXISTING" | tail -n +2 | while read -r dup; do
            [ -z "$dup" ] && continue
            gh issue comment "$dup" --repo "$REPO" --body \
                "Duplicate of #${CANONICAL}, which tracks this red \`${BRANCH}\`. Closing."
            gh issue close "$dup" --repo "$REPO" --reason "not planned"
            echo "Closed duplicate red-${BRANCH} issue #${dup}"
        done
        exit 0
    fi

    BODY=$(
        cat <<EOF
\`${BRANCH}\` is failing its quality gate.

| | |
|---|---|
| **Commit** | [\`${SHORT_SHA}\`](${COMMIT_URL}) |
| **Failing run** | [Quality gate](${RUN_URL}) |

This commit is **already on \`${BRANCH}\`**. Every commit branched from here inherits the
breakage, so this is urgent in a way a red PR is not.

**Before you fix it, comment here to claim it.** On 2026-07-14 two agents opened duplicate
hotfix PRs 31 seconds apart because neither knew the other had started. This issue is the
coordination point.

Then either revert the offending commit or fix forward. A green push to \`${BRANCH}\` closes
this issue automatically.

<sub>Filed automatically by the quality gate. Root cause of this class of failure:
[CLAUDE.md rule 4](https://github.com/${REPO}/blob/${BRANCH}/CLAUDE.md).</sub>
EOF
    )

    gh issue create \
        --repo "$REPO" \
        --title "🔴 \`${BRANCH}\` is RED at ${SHORT_SHA}" \
        --label "$LABEL" \
        --body "$BODY"
    echo "Filed a new red-${BRANCH} issue"
    ;;

close)
    # Auto-close on green. The issue asserts a live fact — "this branch is red RIGHT NOW" —
    # and a green push disproves it: the gate has just passed on the true merge result, which
    # IS the definition of the branch being green. Leaving it open after the fact would train
    # everyone to scroll past an alert that is usually stale, which is how the Actions tab
    # stopped being read in the first place. The audit trail survives in the closed issue.
    EXISTING=$(open_issues)
    if [ -z "$EXISTING" ]; then
        echo "No open red-${BRANCH} issue; nothing to close"
        exit 0
    fi
    # Close ALL of them, not just the canonical one: if a race ever left a duplicate behind,
    # a green branch must not leave a stale "branch is RED" issue open. An alert that is
    # sometimes wrong is an alert people learn to ignore.
    echo "$EXISTING" | while read -r issue; do
        [ -z "$issue" ] && continue
        gh issue comment "$issue" --repo "$REPO" --body \
            "✅ Green again as of [\`${SHORT_SHA}\`](${COMMIT_URL}) — [passing run](${RUN_URL})"
        gh issue close "$issue" --repo "$REPO" --reason completed
        echo "Closed red-${BRANCH} issue #${issue}"
    done
    ;;
esac
