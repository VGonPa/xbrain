# AGENTS.md — xbrain

Instructions for coding agents working in this repository. **Claude Code reads `CLAUDE.md`; Codex
reads this file.** They must not diverge: everything below is either a pointer into `CLAUDE.md` or
the delivery topology, which both agents apply identically.

## Read `CLAUDE.md` first, in full

`CLAUDE.md` is the project contract: architecture, pipeline stages, conventions, and the
**"Rules paid for in blood"** — fifteen rules, each written because something shipped wrong while
the suite was green. They are not style advice. Apply them; do not admire them.

The five that bite an agent hardest:

- **Rule 1** — a test that passes before you write the fix is not a test. **Watch it go red first.**
- **Rule 2** — a metric that cannot come out any other way is not a measurement. State the
  population you measured on and how a different answer could have come out.
- **Rule 3** — nothing catches itself: judge ≠ party, and the judge must **execute**, not read.
- **Rule 4** — a green PR against a moving base is not a green base. Run the suite on the **merge
  result**, and read a check's **reported conclusion**, never the exit status of the command that
  printed it.
- **Rule 9/10** — name the surface you read. `gh pr checks` exits 0 on a failing check; a step that
  ran `exit 1` can report `"conclusion": "success"`; `continue-on-error` is harmless on the job and
  lethal on the gate step.

## Git workflow

- `develop` is the integration branch: `feature-branch → PR → develop`. Branch from `develop`,
  never from `main`; target every PR at `develop`.
- `develop → main` only via PR. Never merge or push directly to `main`.
- Never commit personal data: `auth/storage_state.json`, `data/`, `config.toml` — all gitignored.
- **Implementation plans never enter the repo.** They live in `zz-support-files/`, which is
  gitignored without exception. A finished spec may live in `docs/`; a plan may not.

## Local gate, before you claim anything is green

```bash
bash scripts/check.sh        # ruff · ruff format · mypy · bandit · detect-secrets · pytest · coverage
```

Read the **conclusion the script prints** (`ALL CRITICAL CHECKS PASSED`), not `$?` of the pipeline
that printed it. Coverage minimum is 78% globally and **90% for `src/xbrain/knowledge/`**. Radon
D/E/F fails; C warns.

## Delivering a large initiative: `develop` → umbrella → child PRs

**This is mandatory for any initiative that would otherwise be one large PR.** It is `CLAUDE.md`
rule 15, and the full procedure — PR matrix, worktree lifecycle, roles, blocking criteria — is in
`zz-support-files/docs/implementation-plans/2026-09-02-plan-entrega-atomica-umbrellas.md`.

### Why

A plan is a unit of product; **a PR is a unit of review**, and they are not the same size. Measured
on this repo, 2026-09-02: one plan delivered as one branch — **78 commits, 43 files,
+16,656/−902** — needed **nine review rounds and 615,938 bytes of review prose** (**9.0×** the plan
that merged) to converge, and round **8** still found **two new HIGHs** on a tree with `check.sh`
green and 98% coverage. The code was fine. **A review of 16,656 lines does not converge**: each
pass picks a subset, and the subset it did not pick stays unlooked-at.

### The topology

```
develop ──► VGonPa/umbrella-<name> ──► one PR back to develop
                ▲     ▲     ▲
              02.1 → 02.2 → 02.3   (sequential; each branches from the umbrella already
                                    containing its predecessors)
```

- A child PR targets **the umbrella**, never `develop`.
- Children are **sequential**, merged **one at a time**. Never two open at once.
- **Every child must be green by itself** (`scripts/check.sh` on the merge result). If a boundary
  cannot produce a coherent state, **move the boundary and document why** — never merge red, never
  `xfail` your way across a line, never split an acceptance criterion in half.
- Only the umbrella opens a PR to `develop`; its gate looks for **integration regressions**, not
  unit defects.
- **No recap.** A child PR cites the numbers of its predecessors instead of re-explaining them.

### The umbrella gets a real gate — mandatory, in this order

Rule 14: a PR onto a feature branch runs **no gate at all**, and GitHub reports
`mergeStateStatus: CLEAN` — which means *nothing was required*, not *everything passed*. Close it:

1. **PR 00 (governance), before the umbrella exists.** `.github/workflows/quality.yml` lists
   `"VGonPa/umbrella-*"` under **both** `push` and `pull_request`, and `tests/test_ci_workflow.py`
   gains the asserts that keep it there (rule 11: guard what can be hollowed out).
2. **Classic branch protection on the umbrella's EXACT name**, created **before the first child**:

   ```bash
   gh api -X PUT repos/VGonPa/xbrain/branches/<url-encoded-umbrella>/protection \
     -f 'required_status_checks[strict]=true' \
     -f 'required_status_checks[contexts][]=quality' \
     -F 'enforce_admins=true' \
     -F 'required_pull_request_reviews[required_approving_review_count]=0' \
     -F 'restrictions=null' -F 'allow_force_pushes=false' -F 'allow_deletions=false'
   ```

   **`approvals` must be 0** — one collaborator, and GitHub forbids self-approval, so a 1 blocks
   every child forever (rule 12). Available here, measured: the repo is **public**, the token has
   `admin`, and `develop` is itself a **non-default** branch carrying exactly this. `rulesets` is
   `[]` — rule 13's GHEC-only finding is about rulesets, not classic protection. Wildcards need
   GraphQL, so protect the **exact name**.
3. **Verify by API and read the response**, not the exit code. If it cannot be created, that is a
   **BLOCKER** — stop and say so. Do not substitute a rule someone has to remember.
4. **Remove the protection only after the final merge**, with the branch.

### `strict` does not tell you `develop` moved

`strict: true` compares a child to **its base, the umbrella**. It never looks at `develop`, which
can advance twenty commits with every child still `CLEAN`. So:

- **Checkpoint `git rev-parse origin/develop` before every child** against the SHA recorded when
  the umbrella was created.
- If it moved, integrate with a **sync child PR** (`02.4s`, `03.2s`…) whose only content is
  `git merge origin/develop`. **Never a direct push** (skips the gate you just installed) and
  **never a rebase** (rewrites merged children).

### Redistributing an already-implemented monolith

When the code exists and only its delivery is wrong, **do not reimplement it** — you would discard
every review finding already fixed, and reintroduce defects the plan does not describe.

- **Freeze the tip as a snapshot SHA** and treat it as read-only evidence: no rebase, no reset, no
  force-push, no deletion.
- **Branch the umbrella from the snapshot's historical base**, not from a moving `develop`.
- **Port bytes, do not retype them**: `git checkout <snapshot> -- <paths>` for new files;
  `git checkout -p <snapshot> -- <file>` for pre-existing files several boundaries touch.
  `cherry-pick` rarely helps — one commit usually spans three future PRs.
- Anything not portable verbatim is a **declared boundary adaptation**, named in the PR body.
- **Prove `tree == snapshot` before the first sync.** After a sync the only honest reference is the
  synthetic `git merge-tree --write-tree origin/develop <snapshot>`.
- **Check the suite did not shrink** against the snapshot's measured counts. Fewer tests with a
  green gate is rule 11's fail-open.

### Size is a signal, not a game

Under 400 lines is normal; 800–1,500 needs a written justification; over 1,500 is a failed cut
unless demonstrated otherwise. **Coherence beats the band** — splitting a service in half to hit a
number produces two PRs that must both be read to review either. Tests count and are not
discounted: they are exactly where rule 1 lives.

### Reviewing a child PR

Scale the panel to the PR: 3–4 lenses for XS purely-additive, 5 for S, 6–7 for M/L. **Reviewers
execute** — run `check.sh`, run each acceptance criterion, and **mutate**: break what a test claims
to protect and confirm it goes red. A test that survives its mutation protects nothing. A reviewer
does not edit code, commit, push, or open PRs; it delivers a report.

Use at least one reviewer on a **different model** from the implementer's. Three judges sharing one
model and one rubric are one sample drawn three times: their unanimity measures agreement, not
truth.
