# CLAUDE.md — xbrain

Python CLI (`xbrain`) that extracts X bookmarks/tweets into a JSON store and
generates an Obsidian wiki.

## Stack
- Python 3.12+ (venv currently runs 3.13), `uv`, `pydantic` v2, `typer`, `playwright`, `trafilatura`, `pytest`.
- `uv pip install` needs `--index-url https://pypi.org/simple` to bypass the
  machine-wide private FITIZENS pip index.

## Architecture
- Pipeline: `extract → import-archive → fetch → [enrich] → generate`.
- Quoted posts (a quote-tweet's third-party content): X embeds the quoted post — body
  AND author — in the same timeline payload as the tweet quoting it, so `extract`
  parses it out as a `ContentSourceSuccess(kind="quoted_tweet", author=…)` at **no
  extra network cost** (a `ContentSourceFailure` when X tombstones it / refuses it /
  hydrates nothing). It then reaches every LLM surface under ONE shared label —
  `executors.api.quoted_attribution` → `Quoted post — @handle (Name)` — read by the api
  prompt, the enrich worksheet and the judge's `_source_text`, so the attribution rule
  (**the poster is not the author of what they quote**) is enforceable. Backfill of
  already-stored items: `xbrain refresh-quoted --from-store` (offline join on
  `quoted_id`, repairs the 199 of 762 whose quoted post is already an item) then
  `xbrain refresh-quoted` (re-capture, for the rest). Both bump `content.fetched_at`,
  so `enrich` re-generates exactly the repaired summaries.
- Media side-pipeline: `media` (download photos) → `describe` (vision LLM);
  `refresh-media` re-captures X to backfill the playable video URL + bitrate +
  duration onto already-stored items (video-only, preserves photos/enrichment;
  destructive → auto-snapshot); `download-videos` then downloads the mp4 bytes
  for backfilled videos (mp4 only — HLS `.m3u8` needs ffmpeg and is a deferred
  follow-up; prints a ~GB size-gate, confirm unless `--yes`; destructive →
  auto-snapshot).
- Vision descriptions — pipeline integration (#34): content-bearing described-photo
  prose feeds **both** the `enrich` and `topics` LLM inputs, on **both** the API and
  the worksheet (`claude-code`/`manual`) tracks. `enrich`: the api executor splices an
  `Images in this post:` block (`executors/api.py:_user_prompt`); the enrich worksheet
  carries an `image_descriptions` field per item (reusing `_content_image_descriptions`,
  the same non-decorative seam — shared, not duplicated). `topics`: the api track appends
  the flat content-image list; the topic worksheet carries `image_descriptions` per topic
  from the `TopicInput` that `build_topic_inputs` already computes. Decoratives are
  filtered at the seam so avatars/memes add no topic noise. Wiring only: the
  descriptions flow whenever enrich/topics next run for an item. To propagate them
  onto ALREADY-enriched items (a one-time LLM cost, run separately): `xbrain vocab
  --regenerate` (clears enrichments) → `xbrain enrich` re-runs every item with its
  image descriptions; `xbrain topics --resynth` re-synthesizes overviews with the
  image + transcript evidence.
- Agent-driven video surface (fetch is mechanical, ML is external): `list-videos`
  is a **read-only** catalog of video media (`--json` → stable `{id, url, state,
  topic, size_bytes, mp4_url, text}` array; filters `--topic/--status/--max-size/
  --source/--limit`; no writes, no snapshot); `fetch-video --to <dir>` does an
  **ephemeral** mp4 fetch to `<dir>/<id>.mp4` (select by `--ids`/`--topic`),
  reusing `video_media` primitives — deliberately non-persisting: it does NOT
  mutate `items.json`, does NOT snapshot, and does NOT touch `data/media/`.
- Video digest: `digest-video` turns bookmarked videos into text — ephemeral
  fetch → **external** transcriber subprocess (`[transcribe].command`, default
  `parakeet-mlx`; NO MLX/ML in xbrain core) → attach the transcript to the item
  as a `ContentSourceSuccess(kind="x_video")` → discard the bytes. **Dedup by
  video identity** (the stable `amplify_video`/`ext_tw_video`/`tweet_video` id
  from the mp4 URL path, not the signed URL): N bookmarks of one video → one
  fetch+transcribe, all get the source. No-speech videos attach with empty text +
  `has_speech=False` (never a hard failure). Idempotent (skips items with a fresh
  `x_video` source unless `--force`); destructive → auto-snapshot.
- **Unfetched links carry their REASON (PR-I).** The shared `unfetched_links_note` builder now
  names WHY the content is missing ("the page no longer exists (HTTP 404)" vs "the page could not
  be extracted") — one builder, so all three LLM surfaces (api prompt · enrich worksheet · verify
  source) get it verbatim, and the judge can hold the generator to it. Naming the cause never
  licenses describing the content: the rule sentence is unconditional.
- **A wall is never evidence (`validate_body`).** The only content check used to be `if not text`
  — non-empty ⇒ success. Firecrawl RENDERS JavaScript and `js_required` means "downloadable but
  no extractable article", so a retry would very often "succeed" on a consent/login wall and hand
  back the banner. Accepting it flips the source to success → `links_content_unfetched` goes False
  → the `[Links — content NOT fetched]` block DISAPPEARS from all three surfaces → rubric-summary
  orders the generator to summarise "the article's substance" and the judge sees a
  `[Linked article]` and trusts it. A rendered Instagram wall even contains "Instagram", so the
  entity checker would call it GROUNDED. `validate_body` (length floor · wall + page-chrome
  markers · title≈bare-domain) records these as a `blocked_interstitial` FAILURE instead, at the
  PERSISTENCE boundary (`_safe_extract`), so no extractor — injected or future — can write a wall
  into the store. Verified against the real nature.com / reddit / instagram / twitch URLs the
  backfill would hit.
- **`fetch --retry-failed`.** `_should_refetch` retries only `_TRANSIENT_FAILURES`, so
  `js_required`/`empty_content` are treated as terminal and NEVER retried — yet those are exactly
  the two reasons `extract_article` escalates to the Firecrawl fallback, which returns None (and
  keeps the failure at `attempts=1`) when `FIRECRAWL_API_KEY` is unset. Every failure in the real
  corpus is at `attempts=1`: the fallback has never run. "trafilatura cannot do better" is not the
  same fact as "the pipeline cannot do better". `--retry-failed` targets only the failures a retry
  could repair (transient, plus fallback-eligible when the key is set), with `--dry-run`; it does
  NOT re-fetch what already succeeded, which is what `--force` does.
- Video digest — pipeline integration (PR3, + #75): the attached `x_video` transcript
  flows through the **existing** `enrich → topics → generate` steps, no new stage.
  `enrich` feeds the transcript (+ frame descriptions) into the item prompt (skips
  no-speech). **The two tracks differ on length:** the `api` executor splices a
  `Video transcript:` block capped at `TRANSCRIPT_CHAR_LIMIT`=12000 chars, while the
  worksheet (`claude-code`/`manual`) track sends the **FULL untruncated** transcript
  (`worksheet._video_transcript`) — a full-context agent judges it — plus a
  `video_frame_descriptions` field (what the video SHOWS — the slide descriptions,
  #75; the `api` track injects the same as a `Video frames:` block). `topics` folds a
  tighter per-video excerpt (`TOPIC_TRANSCRIPT_CHAR_LIMIT`=2000) into the synthesis
  prompt; `generate` renders a `## Video digest` section (or a one-line silent-video note).
  This is what fixes video items showing topic `—`. Re-enrichment trigger:
  `attach_transcript` bumps `content.fetched_at`, and `enrich` re-enriches any item
  whose `content.fetched_at > enriched.enriched_at`, so a transcript attached AFTER
  a tweet-only enrich is not treated as already-processed. **Re-enrich fires only on
  a *material* content change:** `fetch.fetch_item` preserves the prior `fetched_at`
  when a re-fetch reproduces the same source set — fingerprinted (`_source_signature`)
  as the whole source model minus fetch bookkeeping (`attempts`/`error`), a
  model-derived deny-list that captures every content field (incl. `title`) and fails
  safe — so a persistently-failing transient link, re-fetched every run by
  `fetch_pending` (which keys on source state, not time), does not burn one identical
  LLM call per cycle.
- Video digest — visual layer (PR4, `--frames`, opt-in): for slide-heavy talks,
  `digest-video --frames` extracts key slides via **external** `ffmpeg`
  (`video_frames.py`, scene detection + interval sampling so a static tail is still
  covered; NO ML/vision lib, Pillow only for edge-density classify), describes each
  via the **external** vision model (`vision.py`, `[vision].command`; mirrors
  `transcribe.py`, no bundled default), records the descriptions on the `x_video`
  source's optional `frames` list, and embeds the slide images into the note like
  downloaded photos (`_media/` mirroring). Content-aware: talking-head/interview
  videos are detected and the visual layer is skipped + logged (never a silent
  drop). Default off — a normal `digest-video` run never touches ffmpeg/vision.
- Video digest — long-form synthesis (`video-digest`, #44 / PR #78): a **separate**
  worksheet stage (not folded into `digest-video`) that reads the transcript + frame
  descriptions and writes a readable long-form digest ("what it is · key points · why
  it matters") to the `x_video` source's **additive `digest: str = ""`** field on
  `ContentSourceSuccess` (`""` = "no digest yet", so every pre-digest record loads
  unchanged). Worksheet flow like enrich (`--executor manual|claude-code`, reuses
  `[enrich].executor`; NO `api` track, NO config section of its own); `--apply`
  imports the filled worksheet, writes every `source.digest`, and **auto-snapshots**
  (the apply branch is the one that mutates `items.json`; export only writes the
  worksheet JSON). `generate` then renders the digest as the section HEADLINE,
  demoting the raw transcript + frames into a collapsible `<details>`
  (`i18n.Strings.video_evidence_header`); an empty `digest` falls back to the old
  inline raw layout (back-compat).
- Enrichment verification (`verify`, LLM-as-judge, #79 / PR #80): a **report-only**
  QA stage — an ensemble of LLM judges scores each enrichment output (`summary`,
  video `digest`, `topics`) for **faithfulness** (grounded in source?) + **adherence**
  (follows rubric?). `--target summary|digest|topics|all`; worksheet flow (`--executor
  manual|claude-code`, reuses `[enrich].executor`, no `api`); `--apply` accepts
  **multiple** worksheets (one per judge), aggregates them (faithfulness unforgiving:
  one judge's FAIL sinks the group), and writes `data/verify-report.{json,md}`.
  **Report-only by default — never mutates the store, never snapshots** (mirrors
  `cv-guardrail`). **Opt-in `--write-verdicts`** (only with `--apply`) persists each
  verdict onto `Item.verification` and auto-snapshots — see the badge bullet below;
  **`--audit`** runs the verifier-audit judge≠party re-check over the FAIL/divergent
  verdicts (`verification_audit.py`). **`--audit --apply … --write-verdicts` persists the
  MERGED, post-audit verdicts** — the audited verdict is the authoritative one, so a FAIL the
  auditor revoked never badges a note and a confirmed/auditor-added failure does. The write
  consumes `merge_audit`'s output (floor, confidence gate, mass-revocation guard, anti-washing
  all intact); it never re-derives a verdict. Three rules keep a persisted verdict from LYING:
  **`--write-verdicts` is incompatible with `--force`** (`--force` bypasses the already-audited
  guard, and each forced run re-renders the report from the merged records — so the FAIL set
  shrinks and N single-revoke runs would clear every FAIL without ever tripping the
  mass-revocation guard, which needs ≥2 FAILs; forced re-audits stay available report-only);
  an **absent `audits` key is not an empty audit** (it would pass every record through
  un-audited, persisting the PRE-audit aggregate), and a **write whose audit matched nothing**
  while consequential records remain is refused; and the **store is written BEFORE the report**,
  so a failed write never leaves the report marked `audited` (which would deadlock the retry
  behind the now-forbidden `--force`).
- X Articles as blogposts — model seam (#39 PR1): an `x_article`
  `ContentSourceSuccess` carries an additive, ordered `blocks: list[ArticleBlock]`
  body — `ArticleTextBlock` (`kind="text"`) + `ArticleImageBlock` (`kind="image"`,
  optional `alt`, `media` **wrapping the existing `MediaEntry` photo-state union**),
  discriminated on `kind`. Reusing `MediaEntry` means the photo download engine +
  path/timestamp validators + `_media/` mirror apply to article images with no new
  plumbing. `text` stays the flattened body (= concatenation of the text blocks) so
  `enrich`/`topics`/`generate`'s fallback consume it unchanged. Optional + additive
  (defaults to `[]`) → existing `items.json` loads unchanged, same as `frames`. The
  download walk (`media`, PR4) and the blogpost renderer (`generate`, PR5) complete
  the chain — a bookmarked Article renders end-to-end as an ordered blogpost note.
- X Articles — extract link synthesis (#39 PR2): `graphql._extract_article_link`
  detects a directly-bookmarked long-form Article (the `article` entity on the tweet
  result: `article.article_results.result.rest_id`, anchored via `_dig`) and
  synthesizes its canonical `https://x.com/i/article/<id>` `Link` (deduped against
  `entities.urls`) so the existing `fetch` x.com path fires for it — no routing/model
  change. A missing/malformed Article node degrades to no link (never a wrong one).
  Model-independent (uses the existing `Link`). Fixture is **constructed**, not a
  recorded live payload — validate the key path against a real capture before prod.
- X Articles — structured fetch (#39 PR3): `fetch_x._fetch_rendered` intercepts the
  article-content GraphQL (URL op-name contains `article`; same `page.on("response")`
  pattern as `_fetch_tweet`/`TweetDetail`) and `extract/article.parse_article_content_state`
  maps the Draft.js `content_state` into ordered `ArticleBlock`s (text runs +
  `MediaPhotoPending` inline images, in document order). `text` is set to the exact
  `"".join` of the text runs (enforced by a `ContentSourceSuccess` `model_validator`).
  On any interception/parse miss it degrades to the retained `trafilatura.extract`
  text-only path (`blocks=[]`); a truly empty article still records `empty_content`.
  `_attach_x_sources` bumps `fetched_at` only on a material `x_article` change (reusing
  `fetch._sources_materially_equal`) so a richer body re-triggers enrich. Fixture +
  op-name are **constructed/unconfirmed** — validate against a real capture (open-Q #4).
- X Articles — inline-image download (#39 PR4): `media.download_all` extends the photo
  walk to advance each `ArticleImageBlock.media` on an `x_article` source
  (`_iter_eligible_article_images` mirrors `_iter_eligible_attempts`), reusing the SAME
  `_download_one` engine/size-cascade/throttle/failure-classification — no new download
  loop. Bytes land at a **namespaced** `data/media/<id>/article/<n>.<ext>` (via
  `_local_path(..., subdir="article")`) so they never collide with the item's own
  `<id>/<n>` photos; the result is swapped **in place** onto `block.media` (safe —
  no `validate_assignment`, images don't affect `text`). Dedicated `MediaReport.article_images_*`
  counters + SUMMARY fields, incl. a dedicated `article_images_skipped` (distinct from the
  photo skip counter, never contaminated); the total-failure `RuntimeError` and `--limit`
  key on the **combined** photos+article totals (`--limit` threaded into the generator's
  top-of-iteration check, like the photo path, so a spent budget never miscounts skips).
  **`--force` decision (documented):** `fetch --force` rebuilds `x_article` with fresh
  `MediaPhotoPending`, so a forced re-fetch resets image state and the next `media` run
  re-downloads — the conscious "redo from scratch" choice (not carry-forward), consistent
  with `fetch --force`/photo `--force`.
- X Articles — blogpost render (#39 PR5): `generate._article_blocks_lines` renders an
  `x_article` source with non-empty `blocks` as an ordered blogpost under `## Content:
  <title>` — walking `source.blocks` IN AUTHORED ORDER: `ArticleTextBlock` → a body
  paragraph (the baked `\n\n` separator stripped via `removeprefix` so it never leaks
  as a stray blank line), `ArticleImageBlock` → an inline `![[_media/<id>/article/<n>]]`
  embed (alt + a described image's caption as `> …` lines; failed → `⚠ Imagen no
  disponible`; pending → silent), the SAME photo convention as `_render_media_lines`.
  `_mirror_item_article_images` copies the bytes into the vault via the shared
  `_mirror_file`, keyed by the STORED `local_path` (no per-source index recompute). An
  `x_article` with empty `blocks` (trafilatura fallback / pre-#39) renders the plain
  `source.text` — byte-unchanged, no regression. Deterministic + regen-stable.
- Verification badge — staleness-aware (#79, follow-up of the verification layer): opt-in
  `verify --apply --write-verdicts` (and `verify --audit --apply … --write-verdicts`, which
  persists the MERGED post-audit verdicts) persists each verdict onto the **additive** `Item.verification`
  field (`dict[str, VerificationVerdict]` keyed by target, defaults `{}` so legacy items load
  unchanged), each carrying a **sha256 `output_fingerprint` of the exact judged text** +
  `verified_at`; the write path auto-snapshots (`pre-verify-write-verdicts`) and echoes a
  written/skipped tally. Default `verify` stays report-only. **The judged fingerprint is stamped
  at worksheet EXPORT** (`export_verify_worksheet`) and threaded through the filled worksheet to
  the writer (`import_verify_fingerprints` → `apply_verdicts_to_store` stores it verbatim) —
  NEVER a write-time recompute against the live store, so a regen in the export→judge→write
  window can't bind a verdict to output it never judged. The SAME stamp rides through the audit
  window: `stamp_record_fingerprints` puts it on the report records → `export_audit_worksheet`
  copies it from the record (never re-fingerprints the live store) → `merge_audit` preserves it →
  the post-audit write reads it off the merged RECORDS (`record_fingerprints`), with the applied
  audit worksheet as a CROSS-CHECK only (`cross_check_fingerprints`: a disagreeing stamp DROPS the
  key fail-safe → the record is skipped, not badged). Deliberately **not a union** — nothing binds
  a worksheet to the report it is applied against (no run-id), so a union would let a stale
  worksheet SUPPLY a fingerprint the record never carried, binding a verdict to a text those
  judges never read. An unstamped record stays unwritable. `generate._verdict_badge` recomputes
  `verification.fingerprint_output` on the item's CURRENT output and renders a localised badge
  (❌ FAIL / ⚠️ REVIEW; PASS unbadged) **only when it matches the stored fingerprint** — a STALE
  verdict (output re-generated in EITHER window) is silently NOT badged, so a fixed output never
  shows a ❌. `fingerprint_output` is the single canonicalization shared by the export stamp + the
  reader; `verdict`/`faithfulness`/`adherence` are a shared `Verdict` Literal and
  `output_fingerprint` is `Field(pattern=...)`-hardened; labels via `i18n.Strings`.
- `data/items.json` (dict keyed by tweet id) is the source of truth; markdown
  is derived. All stages are idempotent and incremental.
- `enrich` is a stub — the LLM executor is intentionally in pause (spec §9).

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
  merge. **The change cannot ship a lie.**
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
