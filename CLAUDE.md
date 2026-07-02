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
- Video digest — pipeline integration (PR3): the attached `x_video` transcript
  flows through the **existing** `enrich → topics → generate` steps, no new stage.
  `enrich` splices a `Video transcript:` block into the item prompt (skips
  no-speech; caps at `TRANSCRIPT_CHAR_LIMIT`=12000 chars); `topics` folds a tighter
  per-video excerpt (`TOPIC_TRANSCRIPT_CHAR_LIMIT`=2000) into the synthesis prompt;
  `generate` renders a `## Video digest` section (or a one-line silent-video note).
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
- X Articles as blogposts — model seam (#39 PR1): an `x_article`
  `ContentSourceSuccess` carries an additive, ordered `blocks: list[ArticleBlock]`
  body — `ArticleTextBlock` (`kind="text"`) + `ArticleImageBlock` (`kind="image"`,
  optional `alt`, `media` **wrapping the existing `MediaEntry` photo-state union**),
  discriminated on `kind`. Reusing `MediaEntry` means the photo download engine +
  path/timestamp validators + `_media/` mirror apply to article images with no new
  plumbing. `text` stays the flattened body (= concatenation of the text blocks) so
  `enrich`/`topics`/`generate`'s fallback consume it unchanged. Optional + additive
  (defaults to `[]`) → existing `items.json` loads unchanged, same as `frames`. The
  producer (`fetch`), download walk (`media`) and blogpost renderer (`generate`)
  land in the later #39 PRs; PR1 is model-only (no behaviour change).
- X Articles — extract link synthesis (#39 PR2): `graphql._extract_article_link`
  detects a directly-bookmarked long-form Article (the `article` entity on the tweet
  result: `article.article_results.result.rest_id`, anchored via `_dig`) and
  synthesizes its canonical `https://x.com/i/article/<id>` `Link` (deduped against
  `entities.urls`) so the existing `fetch` x.com path fires for it — no routing/model
  change. A missing/malformed Article node degrades to no link (never a wrong one).
  Model-independent (uses the existing `Link`). Fixture is **constructed**, not a
  recorded live payload — validate the key path against a real capture before prod.
- `data/items.json` (dict keyed by tweet id) is the source of truth; markdown
  is derived. All stages are idempotent and incremental.
- `enrich` is a stub — the LLM executor is intentionally in pause (spec §9).

## Conventions
- TDD: every module has a `tests/test_*.py`. Run `uv run pytest -v`.
- The X GraphQL parser anchors on key names, not paths — X's private API drifts.
- Never commit personal data: `auth/storage_state.json`, `data/`, `config.toml`.
  All are gitignored.

## Git workflow
- This repo has only `main`: `feature-branch → PR → main`.
