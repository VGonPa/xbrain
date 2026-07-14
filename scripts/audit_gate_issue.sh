#!/bin/bash
# Raise (or stand down) the alarm when the scheduled audit catches the `quality` gate lying.
#
# WHY THIS IS A SEPARATE SCRIPT FROM announce_red_branch.sh
#
# They alarm on two different facts, and conflating them would blunt both:
#
#   announce_red_branch.sh : "`develop` is RED."          -> the code is broken.
#   this script            : "`develop`'s GATE is LYING." -> the code may be broken and
#                                                            NOTHING WILL TELL YOU.
#
# The second is strictly worse, and it needs its own issue, its own label and its own words.
# A red branch is loud: every subsequent PR sees it. A lying gate is silent by construction —
# every API surface reports green (measured: check run SUCCESS, mergeStateStatus CLEAN,
# probes #121/#125). Filing that under the same label as an ordinary red branch would bury
# the one alert nobody else can raise underneath the one everybody already knows about.
#
# IDENTITY: the built-in GITHUB_TOKEN (`github-actions[bot]`) with `issues: write`. No PAT,
# no bot account, no standing credential. A scheduled workflow holding a long-lived PAT is a
# key nobody is watching; the built-in token is scoped to the run and dies with it.
#
# IDEMPOTENCY: one open issue, keyed by the label `ci-gate-audit`. The audit runs daily, and
# a lying gate stays lying until someone fixes it — so the naive version files an issue every
# single day until you do. The defences here are lifted wholesale from announce_red_branch.sh
# because that script already paid for them: every GitHub issue LIST endpoint is EVENTUALLY
# CONSISTENT (measured on this repo — an issue created and immediately listed does not appear
# for several seconds, while fetching it by number returns it at once). A bare
# list-then-create races against its own write; that bug filed issues #113 AND #114 for a
# single red branch. Hence: RETRY the lookup with bounded backoff, and RECONCILE duplicates
# on every run so the state converges instead of compounding.
#
# Usage:  audit_gate_issue.sh open <title> <body-file>
#         audit_gate_issue.sh close
#
# Env (all from trusted `github.*` context — never from user input):
#   GH_TOKEN, REPO, SHA, RUN_URL

set -euo pipefail

MODE="${1:?usage: audit_gate_issue.sh open <title> <body-file> | close}"
case "$MODE" in
open | close) ;;
*)
    echo "Unknown mode '${MODE}' (expected 'open' or 'close')" >&2
    exit 2
    ;;
esac

: "${GH_TOKEN:?GH_TOKEN is required}"
: "${REPO:?REPO is required}"
: "${SHA:?SHA is required}"
: "${RUN_URL:?RUN_URL is required}"

SHORT_SHA="${SHA:0:8}"
LABEL="ci-gate-audit"
COMMIT_URL="https://github.com/${REPO}/commit/${SHA}"

gh label create "$LABEL" \
    --repo "$REPO" \
    --color B60205 \
    --description "The develop quality gate is reporting green without testing" \
    >/dev/null 2>&1 || true

# Oldest first: the oldest issue is the canonical one, so reconciliation is deterministic no
# matter which run happens to observe a mess.
open_issues() {
    gh issue list \
        --repo "$REPO" \
        --label "$LABEL" \
        --state open \
        --limit 50 \
        --json number \
        --jq '[.[].number] | sort | .[]'
}

# Absorbs the list-endpoint lag. Only `open` needs it: there, a false "none" costs a
# duplicate issue. `close` skips it — a false "none" there costs one extra day of a stale
# alert, and the next run fixes it.
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
    return 0
}

case "$MODE" in
open)
    TITLE="${2:?usage: audit_gate_issue.sh open <title> <body-file>}"
    BODY_FILE="${3:?usage: audit_gate_issue.sh open <title> <body-file>}"
    [ -f "$BODY_FILE" ] || {
        echo "Body file '${BODY_FILE}' does not exist" >&2
        exit 2
    }

    EXISTING=$(open_issues_with_retry)
    if [ -n "$EXISTING" ]; then
        # Still lying, on a new commit. Update the canonical issue — never open a second one.
        CANONICAL=$(echo "$EXISTING" | head -1)
        gh issue comment "$CANONICAL" --repo "$REPO" --body \
            "Still failing the audit at [\`${SHORT_SHA}\`](${COMMIT_URL}) — [audit run](${RUN_URL})"
        echo "Updated existing gate-audit issue #${CANONICAL}"

        # RECONCILE: fold any duplicate left by a race into the canonical issue.
        echo "$EXISTING" | tail -n +2 | while read -r dup; do
            [ -z "$dup" ] && continue
            gh issue comment "$dup" --repo "$REPO" --body \
                "Duplicate of #${CANONICAL}, which tracks this. Closing."
            gh issue close "$dup" --repo "$REPO" --reason "not planned"
            echo "Closed duplicate gate-audit issue #${dup}"
        done
        exit 0
    fi

    gh issue create \
        --repo "$REPO" \
        --title "$TITLE" \
        --label "$LABEL" \
        --body-file "$BODY_FILE"
    echo "Filed a new gate-audit issue"
    ;;

close)
    # AUTO-CLOSE ON A CLEAN AUDIT — a deliberate choice, and the arguments both ways:
    #
    # AGAINST: a lying gate is a security-relevant event, and closing the issue could erase a
    # signal someone still needed to read.
    #
    # FOR (and why we do it): the issue asserts a LIVE fact — "develop's gate is lying RIGHT
    # NOW". A clean audit disproves that fact: the gate was executed, honestly, on develop's
    # real HEAD, and it agrees with what GitHub published. Leaving the issue open after that
    # trains everyone to scroll past an alert that is usually stale — which is precisely how
    # the Actions tab stopped being read, and why announce_red_branch.sh exists at all.
    #
    # The audit trail is not lost: the issue is CLOSED, not deleted, it keeps every comment,
    # and the closing comment records the exact commit that proved the gate honest again.
    #
    # The caller passes `close` ONLY on a positively CLEAN verdict (`should_stand_down` in
    # xbrain.gate_audit) — never on merely "no new accusation". An inconclusive audit, a
    # cancelled runner, or a plainly red develop all leave the alarm standing, because none of
    # them is evidence that the gate is honest.
    #
    # NOTE ON LATENCY: the cron is daily, so a fix pushed at noon does not clear the alert
    # until tomorrow morning. That is why `workflow_dispatch` exists on this workflow — re-run
    # the audit by hand once the fix is on develop and the issue closes within the minute.
    EXISTING=$(open_issues)
    if [ -z "$EXISTING" ]; then
        echo "No open gate-audit issue; nothing to close"
        exit 0
    fi
    echo "$EXISTING" | while read -r issue; do
        [ -z "$issue" ] && continue
        gh issue comment "$issue" --repo "$REPO" --body \
            "✅ The gate is honest again as of [\`${SHORT_SHA}\`](${COMMIT_URL}): the audit ran \
\`scripts/check.sh\` on that commit itself and its result matches what the \`quality\` check \
reported — [audit run](${RUN_URL})"
        gh issue close "$issue" --repo "$REPO" --reason completed
        echo "Closed gate-audit issue #${issue}"
    done
    ;;
esac
