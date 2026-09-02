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
  `x_video` source unless `--force`); destructive → auto-snapshot. **On a multilingual
  corpus point `[transcribe].command` at `scripts/xbrain-transcribe-auto` (#133), never
  straight at parakeet.** That wrapper is a LANGUAGE ROUTER: `ffmpeg` slices the first 30 s
  (`XBRAIN_ASR_DETECT_SECONDS`), a small whisper pass (`base`, ~19 s/clip, measured) reports
  the language it detected, then English goes to parakeet and EVERYTHING ELSE — including
  every uncertainty: ffmpeg missing, whisper failing, an unreadable result — goes to whisper.
  The costs are asymmetric in exactly one direction: **parakeet-tdt does not FAIL on Spanish
  audio, it INVENTS** — verified 17-jul-2026 against a real es-ES clip, it exits 0 and emits
  fluent English that never reproduces what was said. A noisy failure shows up in a log; this
  one passes the whole pipeline and lands in a note as a quotation, and by then you can no
  longer tell "the video said that" from "parakeet made it up". The backend has to be chosen
  BEFORE transcription, because it cannot be diagnosed after.
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
- Frame captions — verbatim on-screen text (#90): frame captions are the ONLY
  channel through which on-screen text (slide labels, code, chart axes) reaches
  the digest, and translating a NON-COGNATE label broke that channel — measured:
  `is_grounded("Self-Attention", 'una caja de "Auto-Atención"')` is `False`, so a
  digest correctly naming the label got reported as ungrounded by the #89
  checker. (A cognate pair like `Layer Norm` → `Norma de Capa` does NOT reproduce
  this — the checker's fuzzy match still grounds "Norm" against "Norma" — so
  don't cite that pair as the failure mode.) The rule now lives in ONE place —
  `rubrics/fragment-onscreen-text.md`, spliced by `load_rubric`'s
  `{onscreen_text_rule}` into BOTH `rubric-describe-frame.md` (video frames) and
  `rubric-describe-image.md` (tweet photos) — and reaches the video-frame path's
  EXTERNAL vision subprocess via the `XBRAIN_VISION_PROMPT` env var (the photo
  path calls the Anthropic API in-process, so it just gets the rubric text
  directly), so the documented argv contract `<command> [--model M] <image>`
  stays frozen and a third-party vision command keeps working — it just gets no
  caption discipline. Captions are stamped with
  `ContentSourceSuccess.caption_contract` (`models.FRAME_CAPTION_CONTRACT` =
  `"xbrain-frame-caption/v1"`) — at SOURCE level, and listed in
  `fetch._BOOKKEEPING_FIELDS`, because a stamp nested on `VideoFrame` would land
  INSIDE `_source_signature`'s flat top-level `exclude` (it does not descend into
  nested models) and make every re-stamp read as a material content change.
  `xbrain redescribe-frames` re-captions the frames ALREADY on disk
  (`data/media/<id>/frames/`) with zero network, zero ffmpeg and zero X: it
  skips any source whose `caption_contract` already matches (`--force` overrides),
  stamps a source current only when EVERY one of its frames re-described cleanly
  (a single un-re-described frame — e.g. a permanently missing image — leaves the
  WHOLE source stale forever, so every future run re-pays for its surviving
  frames too, and because a real vision model is non-deterministic that also
  keeps re-bumping `content.fetched_at` and re-triggering `enrich`/`video-digest`
  on that item), and bumps `content.fetched_at` only when a caption actually
  changed. A per-frame `VisionFailed` (bad exit / timeout / empty stdout) is
  logged and the old caption is kept, without aborting the run; a `VisionNotFound`
  (unconfigured `[vision].command`) is a config error and IS allowed to abort the
  run; and a run where every attempted frame failed raises `RuntimeError` instead
  of completing silently (dry runs never raise). **The photo half of this
  contract is not self-enforcing:** `describe._is_stale` keys staleness on the
  hand-set `[describe].version` tag (`config.py`), never on rubric content, so
  upgrading requires an explicit `[describe].version` bump + `xbrain describe`
  re-run, or every already-described photo keeps its old, mistranslated prose
  forever.
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
  body — a **three-variant** discriminated union on `kind`: `ArticleTextBlock`
  (`kind="text"`) + `ArticleImageBlock` (`kind="image"`, optional `alt`, `media` **wrapping
  the existing `MediaEntry` photo-state union**) + `ArticleVideoBlock` (`kind="video"`,
  #133). The video variant exists because **an article's `MEDIA` entity is not always a
  photo**: X embeds native video (`media_info.__typename == "ApiVideo"`) the same way, the
  parser resolved a photo URL or nothing, and every embedded video was therefore DROPPED from
  the body — invisible in the note, where the reader saw prose with a hole exactly where the
  author had put a demo. It wraps the SAME `MediaEntry` union (a `MediaVideoPending`), so the
  playable stream, the poster thumbnail, the bitrate and the duration all ride in the shape
  the rest of the pipeline already understands; the bytes are NOT downloaded and the speech is
  NOT transcribed by this variant (`digest-video` selects from item-level media, never from
  article bodies) — what it guarantees is that the video is no longer lost. Reusing
  `MediaEntry` means the photo download engine +
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
  maps the Draft.js `content_state` into ordered `ArticleBlock`s — text runs +
  `MediaPhotoPending` inline images + `MediaVideoPending` inline videos (#133), in document
  order, with the lead `cover_media` prepended as the first block. Photo-or-video is decided
  ONCE, at the media index (`_item_media`), and the video branch is checked FIRST because a
  video entry also carries a poster image: resolving that first is what dropped every embedded
  video. `text` is set to the exact
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
  disponible`; pending → silent), the SAME photo convention as `_render_media_lines`, and
  `ArticleVideoBlock` → `_article_video_lines`, mirroring the video convention (downloaded →
  local mp4 embed; failed → a visible `⚠`; **pending → a `🎥 Ver vídeo` link, deliberately NOT
  silent**, because no later pass advances an article video — `xbrain media` downloads photos
  — so silence would reproduce the very defect the variant exists to fix).
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
  **`contract_fingerprint` (`verification.py`) is the second, stronger stamp, and the one
  that decides whether a badge may paint at all** — the gate in `_verdict_badge` is
  `verdict_is_current`, i.e. THIS fingerprint, not `output_fingerprint` alone. A verdict is
  not a property of the output
  alone — it is the result of judging THAT output, against THAT source, under THOSE rubrics —
  so it hashes **three arms**: the OUTPUT text; the SOURCE the judge actually read *for that
  target* (`_source_text` = `evidence_surfaces` + the not-fetched markers, so a digest and a
  summary fingerprint different evidence); and the RUBRICS applied (`rubric_digest` =
  `rubric-verify` **plus** the target's generation rubric, since the adherence axis is judged
  against it). `output_fingerprint` hashed only the first, so what the judge reads and the
  rules it reads by could BOTH be rewritten without touching one output character, and every
  stored verdict still matched, still looked current, and still painted its badge — including
  verdicts issued under the contract measured letting a false attribution through 8 times out
  of 8. Now any change to any arm retires every affected verdict automatically, with nobody
  having to remember. And a verdict whose `contract_fingerprint` is `None` (stored before the
  stamp existed) is **permanently stale**: we cannot reconstruct what it was judged against,
  so it is retired, never grandfathered in.
- **The raw-payload layer (`payloads.py`, `payload-stats`, `reextract`) — `extract` is a
  re-runnable transformation over data we own.** Every tweet's raw GraphQL subtree is
  persisted gzipped at `data/payloads/<last-2-of-id>/<id>.json.gz`, scrubbed AT THE SEAM
  (inside `save_payload`, never left to a caller) against a frozenset of WHOLE credential key
  names — never substrings: the first version matched substrings and `auth` ate
  `author`/`author_id`/`authorship`, deleting an author block on write with the original
  already discarded. The failure this removes is exactly the shape of rule 6 (*repair the
  evidence, invalidate the derivative*): we read `legacy.full_text`, capped at 280 chars,
  while `note_tweet` sat unread in EVERY payload — and the fix was not a re-parse, it was a
  network round-trip to X: a logged-in browser, rate limits, and tweets that may since have
  been deleted or protected. With the payload on disk, a parse bug is repaired offline —
  `reextract` re-runs the parser over the whole corpus and PRINTS THE DIFF; only `--apply`
  writes it (auto-snapshot first). It refuses to let "cannot be re-extracted" look like
  "re-extracted cleanly": missing files, corrupt files and payloads that parse to NOTHING are
  each counted apart from coverage. `payload-stats` measures what is actually on disk,
  because the first disk figures quoted for this feature were computed on an X *Article*
  fixture that contains no tweets (rule 2).
- **`refetch-truncated` — the repair that costs a network round-trip, and a count you must
  not quote as a census.** `looks_truncated` (`extract/graphql.py`) detects X's 280-char cut
  by LENGTH: ≥274 chars unconditionally, plus 265–273 when the text does not end on a real
  terminator (`:` and `;` are not terminators). `--apply` is a REAL browser re-fetch —
  headful, human-paced, hours of it — auto-snapshotting first and checkpointing every **25**
  items (`every: int = 25`) plus once more through a `finally` on the way out, so an expiry
  that RAISES loses nothing and only a hard kill drops up to 24 repairs; without `--apply` it
  only reports and writes `data/truncated-items.json`. **Reach for `reextract` first:**
  measured 2026-08-30 on the live store (2,404 items), of the **707** items the detector
  flags only **5** have no stored payload — the other **702 re-parse offline, for free, with
  no network at all**. Corpus-wide the payload layer covers 2,360 of 2,404, so the "payloads
  are NOT persisted" premise — asserted in `items_needing_refetch`'s docstring and AGAIN in
  this command's own docstring (issue #142) — survives only for what predates persistence.
  **But 702 re-parses is a fact about where the BYTES live, not a count of repairs.** Compare
  `note_tweet.note_tweet_results.result.text` in each payload against the stored text and the
  work list splits four ways: **214 truncated** — the payload holds a longer body, and these
  are the real repairs, offline and free; **358 complete** — the long-form body is
  byte-identical to what is stored (358 of 358, median 721 chars, max 13,173), flagged on
  length alone; **122 undetermined** — no long-form body in the payload, so the store cannot
  decide them either way; **13 other** — 8 rewritten without lengthening, 5 with no payload.
  The undecided population is therefore **127**: the 122 plus those 5, which are undecided
  for the stronger reason that no evidence exists in either direction. That puts the real
  truncations at **214–341** and the false flags at **358–485** — two intervals of the same
  width, 127, the same uncertainty counted from either end. The 8 rewritten-without-
  lengthening sit OUTSIDE both and lean truncated (7 of the 8 carry a longer body once the
  trailing t.co is stripped, which is the string `looks_truncated` actually judges), so
  folding them in would raise the truncation ceiling to 349 and never the false-flag one.
  **`--apply` knows none of this:** `targets = items_needing_refetch(store)` is the whole 707
  and the loop re-fetches every one of them (only the WRITE is conditional), so as written it
  is a 707-item browser run in which at most **127** items can gain a character — the same
  127, because what the store cannot decide is exactly what the network would have to be
  asked — while 358 of those fetches re-download posts this measurement proves were already
  complete. That gap is the argument for triaging before you run it. **Do not tidy the 122
  away** by arguing they fitted inside 280 chars: tested against the 214 KNOWN truncations on
  both signatures a reader reaches for, they are indistinguishable — median stored prose 277
  against 277, all 122 inside the 265–292 band against 212 of the 214, a trailing self-link
  on 70% of them against 58%. Two readings survive and nothing in the store chooses between
  them: a truncation whose payload never carried the body, or a complete ~277-character post
  that ends in the author's own link. A discriminating test that fails to discriminate is a
  result, not a dead end. **And the re-parse log is not confirmation:** a dry `reextract`
  warns `tweet ... arrives TRUNCATED` for 702 of the 707, and for the **480** whose text it
  does not change that warning is guaranteed before it runs — same string in, same
  `looks_truncated` out, 480 of 480. Rule 2, tripped inside the repo that wrote it. The
  detector is deliberately biased towards flagging (a missed truncation is a fabrication kept
  forever; a false flag costs one re-fetch), so the number measures the flag, never the
  defect.
- **`verify-entities` / `entity_grounding.py` — the deterministic checker, and READ WHAT IT
  IS BLIND TO BEFORE QUOTING ANY NUMBER FROM IT.** Token-free, no model, so it cannot inherit
  the judge ensemble's blind spot (three judges sharing one model and one rubric are ONE
  sample drawn three times; unanimity there measures agreement, not truth), and it sweeps the
  whole corpus instead of sampling. With `--verdicts <verify-report.json>` it joins the
  flagged outputs against the judges' records and reports how many of them the ensemble passed
  UNANIMOUSLY. **That is a LOWER BOUND, and today it is structurally zero — because the two
  sets do not overlap.** Measured 2026-08-30 on the live data: `data/entity-report.md` carries
  **1,614** flagged rows (140 confident + 1,474 uncertain) while `data/verify-report.json`
  holds **14** records in total, 7 of them `summary`; the intersection with the 140 confident
  flags is **0**. The store is barely better — 70 stored `summary` verdicts across 2,404
  items, and exactly **1** of the 140 confident flags carries one. So the printed number
  measures COVERAGE of the judged population, not the ensemble's quality, and no other answer
  can come out until the two populations are made to overlap. Quoting it as a recall claim
  about the judges is the error rule 2 exists to stop. It checks ONE thing: that every proper
  noun in a generated output appears on an evidence surface its rubric declares (matching is
  variant-aware, because ASR mangles proper nouns and an exact matcher flagged exactly the
  names the generator got RIGHT — ~0% digest precision before that was fixed). **It never
  checks what is ASSERTED about an entity, and it never looks at a single NUMBER.** The
  module's own example: "Sam Altman dijo que despedirá a la mitad", against evidence where he
  discusses *hiring*, extracts `Sam Altman`, finds it grounded, and passes CLEAN. Every false
  attribution, invented mechanism and fabricated causal link has that shape, and **nothing in
  this repo has ever measured how often it happens** — the check that would find them is the
  one that cannot see them. Do NOT reach for the ~7-8% in `data/entity-precision.md` to fill
  the hole: that is the corrected rate of flagged **digests** that are genuinely ungrounded
  (24.9% of digests flagged × ~30% sampled precision), a statement about the outputs this
  check called DIRTY, digests only — and that file says so itself, *"es el suelo de un
  problema, no su medida"*. The blind spot has no number. An invented "92% en MMLU", a false
  date, a fabricated funding round: invisible, always. Lowercase and two-letter names are not
  extracted at all. **A clean verdict means "no unknown proper nouns". It does NOT mean "not
  hallucinated"** — no statement of the form
  "N% of the corpus is hallucination-free" is supported by this tool, and the most damaging
  hallucination for a knowledge base, a confident false claim about a real, correctly-named
  entity, is precisely the one it cannot see. Report-only: writes `entity-report.{json,md}`,
  never touches the store.
- **`evidence.py` is where rule 5's binding actually lives.** `evidence_surfaces(item,
  target)` is the single definition of what may support a claim, and it is
  TARGET-DEPENDENT — `digest` gets the video and the post it arrived in (author metadata ·
  tweet text · video title · transcript · frame descriptions); `summary`/`topics` get those
  PLUS the poster's own thread, the fetched article's title and body, and the image
  descriptions. Getting it wrong is a bug in both directions: judge a digest against the
  linked article and you excuse an invention its generator could never have sourced; judge a
  summary against the digest's narrower set and you flag the generator for using evidence it
  was correctly given. `evidence_text(item, target)` is the flattened string the deterministic
  checker consumes. **A LINK IS NEVER A SURFACE** — nothing derives from `item.links`, because
  a URL is topic signal and never a name: a summary in the corpus reconstructed a publication
  ("Axios") and a company ("Anthropic") out of the SLUG of a link that was never fetched.
  `tests/test_evidence_contract.py` asserts the generator, the rubric and the judge against
  this module BY IDENTITY, per target — add a surface to one consumer and forget the others
  and it goes red.
- **The knowledge layer (`src/xbrain/knowledge/`) is the READ contract, and it is read-only by
  construction** (spec «conocimiento verificable» §3, §4, §8). It projects `Item` + `Content` +
  `Enrichment` + `Topic` + `TopicPage` into `KnowledgeItem` · `KnowledgeSurface` ·
  `KnowledgeChunk` · `TopicRecord`, so a consumer never parses `items.json`, hunts a markdown
  heading, or guesses who wrote a quoted tweet. **It does not replace `evidence.py` and it is
  not the same contract** — the two differ on three axes ON PURPOSE: scope (all surfaces vs.
  target-dependent), truncation (never vs. `ARTICLE_CHAR_LIMIT`) and multiplicity (every
  source vs. the first — 119 items carry more than one). What they SHARE is the atomic walk,
  not the assembled block: `iter_content_sources` / `iter_described_photos` /
  `iter_video_frames` in `executors/api.py`, onto which the five enrichment selectors were
  re-expressed with no observable change. What BINDS them is not an identity assertion
  (tautology the moment delegation exists — rule 1) but **three totality tests**
  (`tests/test_knowledge_surface_coverage.py`) that go red when someone adds a `ContentKind`,
  a derived surface or an evidence key and forgets the other side; the `video_digest` deletion
  leaves the per-kind test GREEN and only the closure test fires. And the proof that the
  refactor did not MOVE `evidence.py` is `tests/test_evidence_characterization.py`, which pins
  the judge's source text and the full `contract_fingerprint` as hex literals: one byte of
  drift would have retired every stored verdict and deleted every badge from `generate` —
  rule 6 run backwards. **Provenance is a TYPE** (`Origin` → `TrustClass`, one total table),
  and `unknown` maps to `llm_synthesis` with `is_derived == True`: it fails closed, because
  `Topic.description` does not record whether it was written or generated and the two errors
  are not symmetric — treating a source as synthesis loses a citation, treating synthesis as a
  source manufactures one. **`verification` is NOT a field of `KnowledgeSurface`** and the
  absence is asserted: `surface_fingerprint` does not depend on the verdict, so a stored copy
  could never be invalidated and a FAIL revoked by `verify --audit` would keep being served as
  the old PASS; it is hydrated from the live store through the SAME freshness check
  `generate._verdict_badge` applies. **`source_key = sha1(kind\0url)[:12]`, never the index in
  `content.sources`** — `fetch` rewrites that order, and an index-keyed id would repoint stored
  chunks at a different body silently, because the id still resolves. **Atomic beats
  `MAX_CHARS`**: a quoted post has ONE author, so the P2 quoted post of 3,943 chars is emitted
  whole against a 2,000 ceiling — splitting it creates two fragments that no longer say whose
  words they are. `target` is a SOFT ceiling that paragraphs are PACKED into, and this was
  found by MEASURING, not by reading: one chunk per paragraph gave **30,449** chunks
  (`x_article` averaging 194 chars) against the plan's predicted 18–25k, and packing gives
  **18,319** (9,294 atomic + 9,025 splittable) — a 194-char chunk is bad retrieval before it
  is bad arithmetic, too small to judge a match and scattering one argument across a dozen ids.
  (The figure was **18,328 / 9,034** until `_absorb_scraps` merged the 9 chunks that sat below
  the floor; the commit that removed them said so and this line was not re-derived — rule 6 in
  miniature, in the file the repo says is read first and acted on. Re-derived 2026-08-31 on the
  same 2,404-item corpus, md5 `5aaf62f4…`. **30,449 is NOT re-derivable**: it measured the
  pre-packing implementation, which no longer exists, so read it as history, never as a figure
  you could reproduce today.) The chunker's parameters are ARGUMENTS, so the Plan-02 sweep
  cannot move the ranking fixture that pins today's behaviour.
- **`eval/golden-set.yaml` is TRACKED — the single exception to "nothing personal in Git" —
  and the loader has two stages because of it.** Untracked (it lived under `data/*`), the v3
  migration would have appeared in no diff, `xbrain eval` could never run in CI, and a case
  edited to turn a gate green would leave no history. It holds questions, ids and short
  identifying fragments; a 300-char ceiling on `expected_text`, checked in CI **against the
  real file**, keeps a corpus body out. `load_cases(path)` validates STRUCTURE without opening
  the store (so it runs in CI, where there is no `data/`); `resolve_cases(cases, store)` checks
  the ids (local, or CI against fixtures). Fusing them would make the very test that proves the
  evaluation runs in CI the first one that cannot. **Only an ENUMERATED case scores**: with
  `relevant_items: []` the recall@k is 0/0 and comes out 1.0 or 0.0 depending on the
  implementation — rule 2 — so unenumerated cases are archived as `scenarios` with their reason.
  Migration measured 2026-08-31 against 2,404 items: D1c/U2 enumerates to **exactly 12** (so the
  Plan-03 bake-off keeps its deciding stratum), P1 to **6** where the file said 5, U3 to **22**
  where it said 20 — the two moved because the corpus grew, which is why the figures are notes
  and never asserts. **`video_digest` still has NO case, and now the reason is measured**: of
  the 36 proper nouns appearing only in a digest, **22 LEAK** (a fuzzy variant sits in the
  transcript, sometimes ASR-mangled — "Johannes Trithemius" vs "johannes tritemius", ratio 0.97)
  and the other **14 have no support on any surface**, i.e. candidates for invention. The first
  group fails the anexo-A.3 leak rule; founding a case on the second would enshrine a possible
  hallucination as ground truth. Zero usable candidates — and the population measured is proper
  nouns, not all facts. **A case whose filters the strategy cannot apply is UNMEASURED, not
  0.0**: the FTS5 baseline pushes only `has_surfaces`/`origins` into `WHERE`, and the first real
  run reported `filtros: recall@10 = 0.0`, which reads as "retrieval failed at filtering" when
  the instrument does not exist yet (spec §8.6.8). The baseline is the SAME FTS5 the persisted
  index will use, on `sqlite3(":memory:")` — same DDL, same `unicode61 remove_diacritics 2`
  (no stemming: FTS5 has none multilingual, and the English one would wreck the Spanish half),
  same `bm25()`, same explicit `chunk_id` tie-break — so what dies later is where the database
  lives, not how it scores. **And that last clause was FALSE while the terms were ANDed.** The
  FTS5 default conjunction requires every word of a question inside ONE chunk: measured on the
  real corpus, 18 of the 21 scorable cases got back NOT ONE ROW, and the only three that
  retrieved anything were single-term `exacto` queries. So bm25 ranked nothing in 18 of 21
  cases, and the published `semantico: 0.0` / `cruzado_idioma: 0.0` / `topic: 0.0` measured the
  QUERY BUILDER while the execution report read them as an absence of vocabulary overlap. The
  connective is now a **disjunction** (`FTS_CONNECTIVE`, recorded in the ranking fixture beside
  the tokenizer): empty result sets 18/21 → 0/21, recall@10 0.1429 → 0.8099, MRR 0.1429 →
  0.7206, `exacto` unchanged and **no stratum regressed**, at p50 0.23 → 9.75 ms. A conjunction
  in front of bm25 is two retrieval models stacked: bm25 wants a wide candidate set and
  discriminates by IDF, and requiring every term does that by brute force *before* the scorer
  runs. **The limit that remains is that IDF is relative to THIS corpus**, so a word that reads
  as a function word can still be rare to the index and go undiscounted — `el` is 1 of 49
  fixture chunks (2.0 %) and **6,070 of 22,286** real ones (**27.2 %**), re-derived 2026-09-01
  on the SHIPPED chunker (v2, `800/0`, store sha256 `f76341a3…`), which is why a fixture query
  ranks it high and the real corpus does not. *(It read `5,748 of 18,319 (31.4 %)`, which was
  correct for the PROVISIONAL chunker v1 and for the store md5 `5aaf62f4…`; the chunker moved
  in this branch and the derived figure did not — rule 6. Read the old pair as history.)* That, and no stemming, is what Plan 03's vector
  layer has to beat. Picking between `OR`, minimum-should-match and per-term weighting is Plan
  02's sweep. **A threshold that reached no bucket is a FAILURE, not a pass**: `--min-recall`
  counts the comparisons it made and fails closed at zero, because `passed = not failures` let
  `--min-recall 1.0` exit 0 having scored nothing.
- `data/items.json` (dict keyed by tweet id) is the source of truth; markdown
  is derived. All stages are idempotent and incremental.
- `enrich` is the LLM stage that writes `Item.enriched` (`summary` · `topics` ·
  `primary_topic`), and it is **fully live** — three executor tracks (`api` in-process;
  `claude-code`/`manual` through the worksheet — the `ExecutorName` literal), ONE shared
  validator (`validate.validate_judgment`, against `guardrails.yaml` and the closed
  `vocab.yaml`), the re-enrichment trigger `_needs_reenrichment`
  (`content.fetched_at > enriched.enriched_at`), and video-transcript + image-description
  splicing on both tracks. Measured 2026-08-30 on the live `data/items.json` with
  `sum(1 for i in store.values() if i.enriched is not None)`: **2,325 of 2,404 items carry an
  `enriched` block.** The 79 that do not ARE how a different answer comes out — they were
  extracted since the last `enrich` run, so read the number as "the corpus is enriched to the
  last run", never as an invariant. An earlier reading the same day was 2,325 of 2,325: the
  enriched count did not move, `extract` did. The line this replaces ("`enrich` is a stub — the
  LLM executor is intentionally in pause (spec §9)") is retired: it was false for the entire
  life of the corpus it described, and it is the worst kind of wrong in this file, because
  this file is read first and acted on.

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
