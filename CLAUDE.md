# CLAUDE.md — xbrain

Python CLI (`xbrain`) that extracts X bookmarks/tweets into a JSON store and
generates an Obsidian wiki.

## Stack
- Python 3.12+ (venv currently runs 3.13), `uv`, `pydantic` v2, `typer`, `playwright`, `trafilatura`, `pytest`.
- `uv pip install` needs `--index-url https://pypi.org/simple` to bypass the
  machine-wide private FITIZENS pip index.

## Architecture
- Pipeline: `extract → import-archive → fetch → [enrich] → generate`.
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
