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
(ffmpeg + an ASR backend, plus mlx-vlm for `--frames`). See the worked example in
[digest-video.md](digest-video.md).

> **If any of your videos are not in English, set the transcriber first.**
> `parakeet-mlx` is English-only and does *not* fail on other languages — it
> invents fluent English and exits 0. Point `[transcribe].command` at
> `scripts/xbrain-transcribe-auto`, which detects the language and routes
> accordingly. See [Picking the transcriber](digest-video.md#picking-the-transcriber).

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

## 6. Check what the LLM wrote (optional)

Two QA passes. Both are **report-only** — neither touches your store — and they
are deliberately different instruments: one is a panel of LLM judges, the other
has no model in it at all.

**`verify`** scores each `summary` / `digest` / `topics` output for faithfulness
(did it invent facts the source doesn't support?) and rubric-adherence, and
writes `data/verify-report.md` worst-first. It is a worksheet flow, like
`enrich`:

```bash
uv run xbrain verify --target all --executor claude-code
# → <N> outputs exportados a data/verify-worksheet.json
# →   Copia N veces (una por juez), rellena `judgments` en cada una, y ejecuta:
# →     xbrain verify --apply ws1.json --apply ws2.json ...
```

Copy the exported worksheet **once per judge**, fill each one in its own Claude
Code session, then aggregate them in a single call:

```bash
uv run xbrain verify --apply ws1.json --apply ws2.json --apply ws3.json
```

How much this is worth depends on how many judges you actually run. One filled
worksheet is one opinion; the aggregation is what surfaces divergence between
judges, and divergence is the signal. Verdicts, `--audit`, `--write-verdicts`
and the badge-writing path have a large flag surface — see the `verify` row in
the README's [Commands](../README.md#commands) table before you use them.

**`verify-entities`** is the deterministic sweep: no LLM, no tokens, and the
whole corpus rather than a sample. It flags generated outputs containing proper
nouns that appear on none of the evidence surfaces:

```bash
uv run xbrain verify-entities --target digest
# → 51 outputs con entidades sin soporte (64 entidades); 0 de ellas con PASS UNÁNIME de los jueces.
# → + 116 outputs cuyo ÚNICO indicio es del tier incierto (mayúscula ambigua, …)
# → Report: data/entity-report.md
```

Counts come from one corpus (205 digests across 2,404 items, read 30-ago-2026);
yours will differ.

**That trailing zero is not a finding**, and it is the easiest number in this
tool to misread. The clause counts flagged outputs the judges had already passed
*unanimously* — the ensemble's false-negative floor. It can only be non-zero when
you pass `--verdicts`; without that flag there is no ensemble to join against and
the count is structurally `0`. Even *with* it this corpus still reports `0`, for
a second reason: only 39 digest verdicts exist across 205 digests, and none of
them lands on a flagged output. So the zero here measures **coverage, not
agreement** — it says the judges never looked at these, not that they cleared
them. Read it as a finding only when your verdict coverage is high enough for
the join to mean something.

Read the caveat before you read the report, because it is narrower than it
sounds. **It checks only that proper nouns appear somewhere on the evidence. It
never checks what is asserted about them, and it never looks at a number.** A
digest claiming Sam Altman said he'd fire half the company, against evidence
where he discusses hiring, finds `Sam Altman` on the transcript and passes
clean. So does an invented benchmark score, a wrong date, a fabricated funding
round. A clean verdict means "no unknown proper nouns" — never "not
hallucinated", and the most damaging hallucination for a knowledge base, a
confident false claim about a correctly-named real entity, is exactly the shape
it cannot see.

That is why the two passes are both here: the judges can read a claim but share
one blind spot, and this one cannot share it but cannot read a claim.
`--verdicts data/verify-report.json` is what actually joins them, and it is worth
running only once enough of the corpus carries verdicts — otherwise, as above,
you are measuring your own coverage.

## 7. See the whole corpus at a glance

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

## When something went wrong

Not part of the happy path. These repair a corpus you already have. Most report
before they write — run the dry form, read it, then re-run to apply.

```bash
# A parse bug was fixed: re-run the parser over the STORED payloads. Offline,
# no network, no re-scrape. Prints exactly what would change; --apply writes it.
uv run xbrain reextract
# → Dry run. Pass --apply to write.
uv run xbrain reextract --apply

# Posts whose text was cut at 280 chars on ingest (the generator then "finished"
# the sentence for them). The raw payloads aren't on disk for these, so applying
# re-fetches each one from X — slow, human-paced browser work.
uv run xbrain refetch-truncated
# → Dry run. Re-fetching requires the network: pass --apply.

# Quote-tweets stored without the post they quote. --from-store joins against
# items you already hold: no browser, instant, re-runnable. It has NO dry run —
# it writes on the spot (after snapshotting). Start here; run the full
# `xbrain refresh-quoted` afterwards for the ones it couldn't reach.
uv run xbrain refresh-quoted --from-store

# Links that failed to fetch, retried only where a retry could actually help.
uv run xbrain fetch --retry-failed --dry-run
uv run xbrain fetch --retry-failed
```

The applying form of each rewrites `items.json`, and auto-snapshots `data/`
before it does, so a bad run is undoable with `xbrain snapshot restore`.
`refetch-truncated` and `refresh-quoted` invalidate the summaries of the items
they repair — re-run `xbrain enrich` afterwards so those get written against the
evidence they were missing.

Stuck? → [Troubleshooting](troubleshooting.md).
