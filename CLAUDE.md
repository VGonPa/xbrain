# CLAUDE.md — xbrain

Python CLI (`xbrain`) that extracts X bookmarks/tweets into a JSON store and
generates an Obsidian wiki.

## Stack
- Python 3.12+ (venv currently runs 3.13), `uv`, `pydantic` v2, `typer`, `playwright`, `trafilatura`, `pytest`.
- `uv pip install` needs `--index-url https://pypi.org/simple` to bypass the
  machine-wide private FITIZENS pip index.

## Architecture
- Pipeline — **six ordered stages**: `extract → fetch → vocab → enrich → topics → generate`,
  with `data/items.json` as the hub every stage reads and writes back. `sync` runs only the
  mechanical three (`extract → fetch → generate`); `vocab`/`enrich`/`topics` are the LLM
  stages and are run explicitly, on your own cadence. `media → describe` is a side-pipeline
  that feeds enrich/topics (below). `import-archive` and the backfill family
  (`refresh-quoted`, `refresh-media`, `download-videos`, `reextract`, `refetch-truncated`)
  sit OUTSIDE the pipeline — one-off repairs, never steps in it.
- `data/items.json` (dict keyed by tweet id) is the source of truth; markdown
  is derived. All stages are idempotent and incremental.

Everything else about a subsystem lives in exactly one document. Read its row
before you touch it:

| Subsystem | Read |
|---|---|
| Install, configure, run it end to end | [README.md](README.md) · [docs/tutorial.md](docs/tutorial.md) |
| Stages, artifacts, rubrics, executors, invariants | [ARCHITECTURE.md](ARCHITECTURE.md) |
| `extract` and quoted posts; the raw payloads (`reextract`, `payload-stats`); `refetch-truncated` | [extract](ARCHITECTURE.md#extract) · [payloads](ARCHITECTURE.md#payloads) · [refetch-truncated](ARCHITECTURE.md#refetch-truncated) |
| `fetch`, `validate_body`, `fetch --retry-failed` / `--revalidate` | [fetch](ARCHITECTURE.md#fetch) · [retry-failed and revalidate](ARCHITECTURE.md#fetch-retry-failed-and-revalidate) |
| X Articles: the `blocks` body, structured fetch, inline images and videos, blogpost render | [invariant 12](ARCHITECTURE.md#invariants) · [fetch](ARCHITECTURE.md#fetch) · [media](ARCHITECTURE.md#media) · [generate](ARCHITECTURE.md#generate) |
| `media` → `describe`; `refresh-quoted`, `refresh-media`, `download-videos` | [media](ARCHITECTURE.md#media) · [describe](ARCHITECTURE.md#describe) · [refresh-quoted](ARCHITECTURE.md#refresh-quoted) · [refresh-media](ARCHITECTURE.md#refresh-media) · [download-videos](ARCHITECTURE.md#download-videos) |
| Video: `list-videos` / `fetch-video`, `digest-video` (and `--frames`), `video-digest`, `redescribe-frames` | [list-videos / fetch-video](ARCHITECTURE.md#list-videos--fetch-video) · [digest-video](ARCHITECTURE.md#digest-video) · [video-digest](ARCHITECTURE.md#video-digest) · [redescribe-frames](ARCHITECTURE.md#redescribe-frames) · [docs/digest-video.md](docs/digest-video.md) |
| `vocab`, `enrich`, `topics`, `generate` (video digest section, verification badge) | [vocab](ARCHITECTURE.md#vocab) · [enrich](ARCHITECTURE.md#enrich) · [topics](ARCHITECTURE.md#topics) · [generate](ARCHITECTURE.md#generate) |
| Rubrics, and the shared on-screen-text fragment | [Rubrics](ARCHITECTURE.md#rubrics-the-prompt-layer) |
| Evidence surfaces (`evidence.py`), `verify` (`--audit`, `--write-verdicts`), `verify-entities` | [evidence](ARCHITECTURE.md#evidence) · [verify](ARCHITECTURE.md#verify) · [verify-entities](ARCHITECTURE.md#verify-entities) |
| The knowledge layer: contract, provenance, identity, chunking | [The knowledge layer](ARCHITECTURE.md#the-knowledge-layer) |
| The persistent index, the vector plane, the evaluation, the graph, the MCP server | [persistent index](ARCHITECTURE.md#the-persistent-index) · [vector plane](ARCHITECTURE.md#the-vector-plane-and-hybrid-retrieval) · [evaluation](ARCHITECTURE.md#the-evaluation-and-where-its-gate-really-reaches) · [graph](ARCHITECTURE.md#the-minimal-graph) · [MCP server](ARCHITECTURE.md#the-mcp-server) |
| Operating the index: commands, costs, limits, measured store and golden-set versions, the spec's §13 table | [docs/knowledge-index.md](docs/knowledge-index.md) · [measured versions](docs/knowledge-index.md#measured-versions) · [§13 table](docs/knowledge-index.md#the-specs-acceptance-criteria) |
| `mcp-serve`, and what an agent does with the answers | [docs/mcp.md](docs/mcp.md) · [docs/knowledge-for-agents.md](docs/knowledge-for-agents.md) |
| The embeddings bake-off and the graph threshold sweep | [docs/embeddings-bakeoff.md](docs/embeddings-bakeoff.md) · [docs/graph-threshold-sweep.md](docs/graph-threshold-sweep.md) |
| Something failing | [docs/troubleshooting.md](docs/troubleshooting.md) |

## Conventions
- TDD: every module has a `tests/test_*.py`. Run `uv run pytest -v`.
- The X GraphQL parser anchors on key names, not paths — X's private API drifts.
- Never commit personal data: `auth/storage_state.json`, `data/`, `config.toml`.
  All are gitignored.

## Git workflow
- `develop` is the integration branch: `feature-branch → PR → develop`. Branch
  from `develop` (never from `main`) and target every PR at `develop`.
- `develop → main` only via PR — never merge or push directly to `main`.

## Rules paid for in blood (2026-07-14: verification audit, then CI audit)

Fifteen PRs merged in one day (`gh pr list --state merged`, 2026-07-14), six agents.
Every rule below is here because we broke it and something shipped wrong while the suite
was green. They are ordered by how often they bit us. Apply them; do not admire them.

Rules 1–8 came out of the morning's audit of the data pipeline. Rules 9–13 came out of
the afternoon's audit of CI and branch protection, and each one was **measured against the
live GitHub API**, with a probe PR number as the receipt. Do not soften them.

### 1. A test that passes before you write the fix is not a test

Six times in one day, six different agents, the same defect: an assertion satisfied for
the wrong reason.

| The assertion | Why it passed anyway |
|---|---|
| `assert "NOT fetched" in source` | satisfied by the section **header** `[Links — content NOT fetched]`. The rule sentence it claimed to pin was unprotected — deleting it stayed green. |
| `assert "1 verdicts escritos" in output` | satisfied verbatim by `"0 de 1 verdicts escritos (1 omitidos: …)"`. The test for *one written* passed on *zero written*. |
| `assert stored_ids(tmp_path) == set()` | `tmp_path` was never passed to the function. It asserted that an empty directory is empty. |
| `assert evidence_text(i, t) == …evidence_surfaces(i, t)` | **both sides came from the same module.** It asserted `evidence.py` against itself — inside the PR written to end this class of test. |
| `assert "topic signal only" in payload` | already satisfied by the *links* rule; it said nothing about the *bookmark folder* it claimed to pin. |
| `assert checker.evidence_text is evidence.evidence_text` | once the checker delegated, the module attribute **is** the same object. The tautology reappears in disguise the moment you think you have killed it. |

**Do:** assert **where** a value lives and **which source** it came from — the label it
sits under, the shared constant it is *identical* to, the behaviour it produces through
the **public API**. Never that a string appears *somewhere*.

**And watch it go red first.** A green test before the fix exists is the only reliable
tell that it is testing nothing. If you cannot make it fail, you have not written a test.

### 2. A metric that cannot come out any other way is not a measurement

`"0 false flags on tweets under 200 chars"` — with a 265-char floor, nothing under 265
*can* be flagged. The number restated the constant. A disk-footprint table and a secrets
sweep were both computed on a fixture that contained **no tweets**. Three headline
numbers were retracted in one day.

**Do:** state the population you measured **on** and the way a different answer could
have come out. If neither exists, do not quote the number.

### 3. Nothing catches itself

Not one defect that mattered was found by CI, by review, or by the author re-reading.
Every one was found by **someone who did not write it, running it against real data**: an
attribution rule that let a false speaker through 8 judges out of 8; a thread served to
the judge as a fetched article; a cookie wall stored as evidence; a quoted post rendered
in the user's note as if the poster had written it; an entity checker with ~0% precision.

**Do:** judge ≠ party, and the judge must **execute**, not read. Run the thing against
the real store (read-only) before you claim it works.

**And the base case**, for when "what guards the guard?" threatens to regress forever: the
escape is **not another catcher**. It is making the **absence** of the catcher fail closed.
Where removing a guard blocks the merge, the regress terminates — see rule 11.

### 4. A green PR against a moving `develop` is not a green `develop`

One PR added a test calling `_source_text(item)`; another changed that signature. No
textual conflict. Both green on their own branches. **The merge was red** — nothing ever
ran the combination.

**Do:** before merging, run the suite on the **merge result**, not on your branch
(`git merge-tree --write-tree origin/develop HEAD` proves it merges; only running the
tests proves it works). And read a check's **reported conclusion** — never infer it from
the exit status of the command that printed it. A red check has already reached `develop`
that way.

**This is now mechanized** (PR #110 + branch protection). `quality.yml` runs on `push` to
`develop`/`main`, so the merge commit itself is finally tested — before this the repo had
**zero** `push` runs in its entire history, and `1209094`, the merge that broke `develop`,
carried `total_count: 0` check runs. Nobody had ever tested it. And `strict: true` forces
the merge ref to be recomputed against the current base *before* merging, so the stale
green cannot land in the first place. `push` is the detector; `strict` is the preventer.

**Caveat, and it is a sharp one:** the advice above — *read the reported conclusion* —
assumed the conclusion is the trustworthy surface. **Rule 10 is the case where it is not.**
A step that ran `exit 1` can report `"conclusion": "success"`. Read rule 9 before you trust
any conclusion field.

### 5. One definition, or five that silently diverge

"What counts as evidence" was written **five times by five hands** — the generator, the
generator rubrics, the judge's rubric, the judge, the checker. Every divergence produced
a confident wrong number with the suite green. Three people independently fixed the same
missing surface. The fix was ONE function (`evidence.evidence_surfaces`) plus a
cross-component test that fails when any consumer drifts.

**Do:** if two components must agree, bind them **in code** (one function, one constant,
one test asserting identity across all consumers) — never in prose, and never in two
lists that "should" match.

### 6. Repair the evidence, invalidate the derivative

Three PRs shipped a repair that fixed the source and left the summary, the digest and the
verdict standing — a full tweet sitting next to a summary of half of it, wearing a PASS
badge.

**Do:** any change to evidence must invalidate everything derived from it
(`contract_fingerprint` does this for verdicts: it hashes the output **and** the source
the judge read **and** the rubrics it applied).

And check the invalidation signal actually **reaches the population being repaired**. The
usual lever is `content.fetched_at` — and it cannot reach an item whose `content` is
`None`, because there is nothing to stamp.

Measured on the real store, each number with the definition it was measured under:

- **620** items are truncated (`looks_truncated`);
- of those, **526** have the truncated tweet as their **only** evidence — no article, no
  transcript, no thread. Repairing their text changes everything downstream;
- and **1,551 of 2,168 (72%)** carry **no `content` block at all**, so a `fetched_at`
  lever reaches none of them.

A repair whose staleness signal lives on the object it is *creating* reaches nobody. (The
figure originally quoted here — "432" — was stale: it predated a fix that moved detection
from 535 to 620, and nobody re-derived it when the population moved. Both numbers above
were re-derived from the store before being written down. That is the whole of rule 2.)

### 7. The cheapest verification layer is showing the user the evidence next to the claim

We built three judges, an independent auditor, a deterministic checker, and a contract to
bind them. Then we rendered the quoted post in the note — and a reader sees in two
seconds that a summary claiming *"his move to Anthropic"* sits above a quoted post that
never mentions Anthropic. No tokens, no threshold, no false positives.

**Do:** before building an instrument to detect a defect, ask whether **showing the
evidence** to the human would make the defect self-evident.

### 8. Once a review lands on a PR, its author owns the fix — unless reassigned out loud

Six times in one day, two agents built the same thing in parallel: the review fixes for one
PR (both versions complete, one thrown away), a `develop`-is-red hotfix (opened twice,
minutes apart), a whole feature re-implemented and opened as a duplicate PR **29 seconds
after the original merged** — and the same missing evidence surface was independently fixed
by three different people.

**Do:** when a review lands, the PR's author fixes it. Reassignment is stated explicitly,
to both agents. Before starting anything that someone else might already be doing, check
the actual remote state (`gh pr view`, `git ls-remote`) rather than the state you remember
— and note that a PR in **CONFLICTING** state runs **no checks at all**, so from the
outside it looks identical to a dead one.

### 9. Two instruments report the same event with opposite answers — name the surface

`gh run list` shows the **workflow run**. `gh pr view --json statusCheckRollup` shows the
job's **check run**. Branch protection reads the **check run**. A `continue-on-error` job
reports the workflow run as `success` and the check run as `FAILURE`. Read the wrong
instrument and you conclude the exact **opposite** of the truth.

This was the third costume the same bug wore in one day:

| The instrument | What it actually reports |
|---|---|
| `gh pr checks` | exits **0** on a failing check. Cost: a red merge (#96). |
| `git push … \| tail` | `$?` comes from **`tail`**, not from `push`. |
| a step that ran `exit 1` | `"conclusion": "success"` — see rule 10. |

**Do:** always name **which surface you read**. And **never trust a reported conclusion —
assert on the SOURCE.** We have a receipt that the conclusion field lies, so any guard that
verifies "every step concluded success" is defeated by the exact attack it exists to catch.
`tests/test_ci_workflow.py` and the `gate-integrity` suite parse `quality.yml` and
`check.sh` and assert on what they **say**.

### 10. The keyword does not have a column; the placement does

`continue-on-error: true` in two places, opposite outcomes:

| Where | Check run | Merge | |
|---|---|---|---|
| on the **job** (`jobs.quality.continue-on-error`) | `FAILURE` | `BLOCKED` | harmless *(probe #119)* |
| on the gate **step** | `SUCCESS`, `mergeStateStatus: CLEAN` | **merges** | **lethal** *(probes #121, #125)* |

On the step, the step that ran `exit 1` reports `success`, the job reports `success`, the
check reports `SUCCESS`. Two people each measured **one cell of a 2×2** and each claimed
the whole table.

**Do:** measure **the cell**, not the keyword.

### 11. Removing a guard fails closed; hollowing it out fails open

First, the mechanism that makes this the only distinction that matters: **a PR's CI runs
the HEAD version of the workflow, not the base's** *(measured: a probe step added on a
branch executed; and #112 — deleting `quality.yml` on a branch produced **zero** runs,
impossible if the base's workflow were used)*. **A PR that neuters the gate is judged by
the neutered gate. It absolves itself.** Which is why asserting on the workflow's
*triggers* alone was never going to be enough.

So every attack on the gate sorts into exactly two bins:

- **FAIL-CLOSED (safe)** — anything that stops the required check from **reporting**:
  deleting the workflow file *(probe #112: zero runs, `BLOCKED`)*; renaming the job (the
  required context `quality` never appears); `continue-on-error` on the job;
  `branches-ignore`; `paths:` under `pull_request`; an invalid `types:`. GitHub blocks the
  merge. **The change cannot ship a lie.** *(This bin assumes the PR targets `develop` or
  `main` — what blocks is branch protection's required check, and it exists only there. On
  any other base these cells flip into the column below: see rule 14.)*
- **FAIL-OPEN (lethal)** — anything that lets the check still say **PASS while testing
  less**: `continue-on-error` on the gate step; `checkout` with an explicit `ref:` *(probe
  #124 — the gate ran green **on the wrong tree**; the test count gave it away, `1088
  passed` where `develop`'s suite ran 1604 that day)*; gutted `steps:`; `COVERAGE_MIN=0`;
  `pytest --ignore=…`; demoting `Tests` to warn-only.

**Do: guard what can be hollowed out — what can be removed already guards itself.** This is
also the base case that terminates rule 3's regress: where a guard's *absence* fails closed,
you do not need a catcher for the catcher.

### 12. `required_approving_review_count` must stay **0**. Never set it to 1

`VGonPa` is the repo's **only collaborator** and authored all 76 PRs. GitHub forbids a PR
author from approving their own PR, and `can_approve_pull_request_reviews` is `false` for
`github-actions[bot]`. **No identity in this repository can satisfy a required approval.**

With `enforce_admins: true`, setting it to 1 makes **every PR permanently unmergeable, with
no bypass — including the PR that would undo it.** A reviewer proposed this today as its top
recommendation. It was thirty seconds from bricking the repo.

**Do:** leave it at 0. This is written down so the next agent does not helpfully re-propose it.

### 13. The honest residue — state it as social, do not dress it as mechanical

**Two lines in `.github/workflows/quality.yml` can make the gate lie** — a
`continue-on-error` on the gate step, or a pinned `checkout ref:` — and **nothing mechanical
prevents it on this repo.** Every escape was checked against the live API and is unavailable:
merge queue is **org-only**; the required-workflows ruleset is **GHEC/GHES-only**;
`file_path_restriction` rulesets are **Enterprise-only**; required approvals are impossible
(rule 12).

The one mechanism that would work — `pull_request_target`, whose definition comes from the
**base** branch, which the PR head therefore cannot edit — was **deliberately declined**: it
runs with the base's secrets and a write token, and on a **public** repo a single careless
future edit that makes it touch head code hands the repository to any stranger who opens a
fork PR. That is a permanent RCE surface bought to defend against an adversary who does not
exist.

**The guards catch mistakes, not attacks. The only backstop against an attack is that
someone reads the diff of `.github/`.** Say so plainly; do not claim the gate is airtight.

### 14. A PR onto a feature branch runs no gate at all — and GitHub calls it `CLEAN`

Rule 11 sorts every attack on the gate into two bins. **That table is conditional on
something it never says: the PR targets `develop` or `main`.** What blocks a merge is
branch protection's *required check*, and branch protection lives only on those two.

`quality.yml`'s trigger is `pull_request: branches: [develop, main]`. Point a PR at a
feature branch — a stacked PR, most obviously — and the workflow fires **zero** times.
No check is produced, so no check is missing, so nothing blocks.

Measured on **#151**, stacked on #149's branch while #149 was still open:

| Surface read | What it said |
|---|---|
| `gh run list --branch feat/90-pr2-redescribe-frames` | **no runs at all** |
| `gh pr view 151 --json statusCheckRollup` | `[]` |
| `gh pr view 151 --json mergeable,mergeStateStatus` | `MERGEABLE`, **`CLEAN`** |

`CLEAN` does not distinguish *everything required passed* from *nothing was required*.
That is rule 9 one level up: the instrument does not lie about a check, it lies **by the
absence of checks**. Retargeting the same PR (`gh pr edit 151 --base develop`) started the
gate — a `pull_request` run was in progress 30 s later — and it passed. No code changed,
only the base.

**Do: a stacked branch is a work container; a PR is a gate. Do not open them together.**
Let the work proceed on the stacked branch, and open the PR only once its base has landed
in `develop`. And read `required_status_checks.contexts` on the TARGET branch before
trusting any merge state.

**And check why you are stacking at all.** #151 was stacked because #149 sat merge-ready
and unmerged — a permission boundary that was never there. `develop` is the *integration*
branch: merging into it is the ordinary end of the loop, not a release. The cheapest fix
for a stacked PR is usually to merge the one below it.

### 15. A big initiative ships as child PRs onto a PROTECTED umbrella — never as one PR

A plan is a unit of product. **A PR is a unit of review**, and the two are not the same size.
Measured, 2026-09-02: Plan 02 of the knowledge initiative was implemented as one branch — **78
commits, 43 files, +16,656/−902** — and took **nine review rounds and 615,938 bytes of review
prose** (**9.0×** Plan 01, which merged) to converge. Round **8**, on a tree with `check.sh`
green and **98%** coverage in `knowledge/`, still surfaced **two NEW HIGHs**, one of them a
`build --force` that destroyed a good index and sealed it with exit 0. Nothing was wrong with the
code — `b61e04b` ends green at 2,549 tests. What was wrong is that **a review of 16,656 lines does
not converge**: each pass picks a subset, and the subset it did not pick stays unlooked-at.

**So: `develop` → `umbrella` → child PRs, merged one at a time.** A child PR targets the umbrella,
never `develop`; children are sequential, so each is read against a tree containing its
predecessors; **each must be green on its own**, and if a boundary cannot produce a coherent state,
move the boundary and write down why. Only the umbrella opens a PR to `develop`, and its gate looks
for integration regressions, not unit defects.

**The umbrella gets a REAL gate — this is not optional, and not a social rule.** Rule 14 says a PR
onto a feature branch runs nothing and GitHub still reports `CLEAN`. Close it, in this order:

1. `.github/workflows/quality.yml` lists `"VGonPa/umbrella-*"` under **both** `push` and
   `pull_request` (guarded by `tests/test_ci_workflow.py`);
2. **classic branch protection on the umbrella's EXACT name**, created **before the first child**
   and removed only after the final merge: `quality` required · `strict: true` ·
   `enforce_admins: true` · **approvals 0** (rule 12 — one collaborator, so a 1 blocks everything
   forever). Measured available here: the repo is **public** with `admin: true`, and `develop`
   itself is a **non-default** branch carrying exactly this. `rulesets` is `[]` — rule 13's
   GHEC-only finding is about rulesets, not about classic protection;
3. **verify by API and read the response**, not the exit code. If it cannot be created, that is a
   **BLOCKER**, not a compensation to remember.

**`strict` on the umbrella does NOT tell you `develop` moved** — it only compares a child to its
umbrella base. So **checkpoint `origin/develop`'s SHA before every child**, and when it has moved,
integrate it with a **sync child PR**: never a direct push (skips the gate) and never a rebase
(rewrites merged children). When a monolith is being **redistributed** rather than written, freeze
its tip as a snapshot, branch the umbrella from the snapshot's **historical base**, port bytes with
`git checkout <snapshot> -- <paths>` instead of retyping them, and prove `tree == snapshot` **before**
the first sync — after a sync the only honest reference is the synthetic `merge-tree` of current
`develop` + snapshot.

Full procedure, PR matrix and roles:
`zz-support-files/docs/implementation-plans/2026-09-02-plan-entrega-atomica-umbrellas.md`
(and `AGENTS.md`, which carries the same topology for Codex).

### Branch protection: the live settings, and what each one cost

Live on **`develop` and `main`** (`gh api repos/VGonPa/xbrain/branches/develop/protection`):

| Setting | What it prevents | What it costs |
|---|---|---|
| required check **`quality`** | a merge with no green gate | nothing |
| **PRs required** | pushing straight to `develop`/`main` | one PR per change |
| **`strict: true`** (up to date before merging) | the stale-green merge of rule 4 | a rebase when the base moves |
| **`enforce_admins: true`** | the owner waving his own change through | the owner waits like everyone else |
| no force pushes / no deletions | rewriting or vaporizing the integration branch | nothing |
| approvals **0** | — | **must stay 0 — see rule 12** |

`strict` was the contested one. The rationale for leaving it off (CI is slow, merges are
frequent, you would rebase all day) was **asserted, never measured** — and the measurement
pointed the other way: **CI median 73 s** *(measured: `[70,66,72,73,73,79,65,74,76,71]`)*
against a **median 7.6 min gap between merges**. The queue never forms.

And the coupling nobody guarded until today: **the required check's name comes from the job
id `quality`.** Rename the job and the required check **never appears** — every merge blocks,
forever. Change the branch-protection setting **first** if you ever must rename it. Guarded
by `tests/test_ci_workflow.py` (17 assertions); ruff's scope by `tests/test_quality_gate_scope.py`.
