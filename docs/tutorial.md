# Tutorial — from zero to a searchable wiki

A worked, end-to-end walkthrough: install XBrain, turn *your* X bookmarks into an
Obsidian knowledge base, and digest a bookmarked talk into readable notes. Every
command is copy-paste; the → lines show what you should see.

New here? Do the [Quick start](../README.md#quick-start) first (install +
authenticate), then come back — this tutorial picks up from a logged-in install.

---

## 1. Confirm you're set up

```bash
uv run xbrain status
# → Items: 0
# →   con enlace: 0
# →   ...
```

An empty store with no error means config + auth are good. If `status` complains
about config, copy `config.toml.example` to `config.toml` and set your vault path
+ X handle. If it can't authenticate, re-run the cookie import (see
[Troubleshooting](troubleshooting.md#x-session-expired--auth-fails)).

## 2. Pull your posts and build the mechanical wiki

```bash
uv run xbrain sync        # extract (scrape X) + fetch (article bodies) + generate
uv run xbrain status
# → Items: 812
# →   con enlace: 143
# →   última extracción bookmarks: 2026-07-04 ...
```

`sync` scrapes your bookmarks + own tweets into `data/items.json`, fetches the
linked article bodies, and writes one markdown note per post into your vault.
Open the vault in Obsidian — you already have `items/*.md` and `_index.md`.

> `sync` runs **headful** by default (a visible Chromium) to look human; it
> paces itself and backs off on rate limits. First run scrolls your whole
> history, so it's the slow one.

## 3. Add the topic layer (the LLM stages)

The mechanical layers need no LLM. The *understanding* layers — a topic
vocabulary, per-post summaries + topics, and topic-page overviews — do:

```bash
uv run xbrain vocab       # induce ~45 topics from the corpus
uv run xbrain enrich      # summary + topics for each post
uv run xbrain topics      # write a topic page per cluster
uv run xbrain generate    # re-render the vault with the new layers
```

By default these use the **claude-code execution mode** (no API key, no cost):
each stage exports a worksheet you fill in a Claude Code session, then
`--apply`. To run them unattended with the API instead, add `--executor api`
(needs `ANTHROPIC_API_KEY`). See [Execution modes](../README.md#execution-modes).

Now your vault has three layers: `items/` (posts), `topics/` (thematic pages),
and `_index.md` (the map). Open `_index.md` in Obsidian and click into a topic.

## 4. Download the media

```bash
uv run xbrain media                 # download bookmarked photos
uv run xbrain download-videos --yes # download videos (prints a size gate first)
```

Photos embed under each post note. To make photos **searchable**, add vision
descriptions:

```bash
uv run xbrain describe --executor claude-code   # export a worksheet
# fill it in a Claude Code session, then:
uv run xbrain describe --apply data/describe-worksheet.json
uv run xbrain generate
```

Each photo now renders with a one-line caption under it — plain note text, so
Obsidian's search finds "that chart about pricing".

## 5. Digest a bookmarked video

This turns a saved talk into a readable, topic-linked note. It needs the local
tooling from [Local models for `digest-video`](../README.md#local-models-for-digest-video-apple-silicon)
(ffmpeg + parakeet-mlx, plus mlx-vlm for `--frames`). See the worked example in
[digest-video.md](digest-video.md).

```bash
# Transcript only (fast): every bookmarked video → an x_video transcript source
uv run xbrain digest-video --all-pending

# With the visual layer: also describe the slides of slide-heavy talks
uv run xbrain digest-video --all-pending --frames

# Turn the transcript (+ slides) into a readable long-form digest — worksheet flow,
# just like enrich: export → fill in a Claude Code session → apply.
uv run xbrain video-digest --executor claude-code
uv run xbrain video-digest --apply data/video-digest-worksheet.json

uv run xbrain generate
# → the video's note now leads with a readable "## Video digest"; the raw
#   transcript + slides are tucked into a collapsible "Frames + transcript" block
```

Skip the `video-digest` step and the note still renders — it just falls back to the
raw transcript inline, without the readable digest.

Optionally, sanity-check the LLM outputs with `verify` — an LLM-as-judge pass over
the summaries, digests and topics. It writes `data/verify-report.md` and **never
touches your store**:

```bash
uv run xbrain verify --target all --executor claude-code
uv run xbrain verify --apply data/verify-worksheet.json   # one worksheet per judge
```

## 6. See the whole corpus at a glance

`generate` also writes `dashboard.html` — a self-contained interactive dashboard
(counts, topics, authors, growth over time, photo thumbnails), with drill-down and
deep links back to each post + note. Open it from the **📊 Dashboard** link at the
top of `_index.md`, or directly in your browser:

```bash
# <vault>/<output_subdir>/dashboard.html — from your config.toml [paths]:
open ~/Documents/Vault/vault/learnings/x-knowledge/dashboard.html
```

## Keeping it fresh

Re-run periodically — everything is **incremental and idempotent**:

```bash
uv run xbrain sync        # pull new bookmarks/tweets, re-render
uv run xbrain enrich      # enrich only the new posts
uv run xbrain topics      # refresh topic pages
uv run xbrain generate
```

The markdown is **derived and disposable** — delete and regenerate any time. The
source of truth is `data/items.json` (snapshotted before every destructive op;
see [Snapshots & safety](../README.md#snapshots--safety)).

Stuck? → [Troubleshooting](troubleshooting.md).
