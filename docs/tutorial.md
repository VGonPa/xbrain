# Tutorial — from zero to a searchable wiki

A worked, end-to-end walkthrough: install XBrain, turn *your* X bookmarks into an
Obsidian knowledge base, digest a bookmarked talk into readable notes, then
search the corpus and hand it to an agent. Every command is copy-paste; the →
lines show what you should see.

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

An empty store with no error means the config loads. If `status` fails with
`No such file or directory: '…/config.toml'`, copy `config.toml.example` to
`config.toml` and set your vault path + X handle.

`status` says nothing about your login. It never reads
`auth/storage_state.json` and prints the same counts with or without it. The
session is first used by `sync`, in the next step. If there is no saved session,
`sync` stops with exit code 1 before it opens a browser or writes anything:

```
Error: No hay sesión guardada en …/auth/storage_state.json. Ejecuta `xbrain login`.
```

Re-run the cookie import from the Quick start. A session that exists but has
expired is a different failure: see
[Troubleshooting](troubleshooting.md#x-session-expired--auth-fails).

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

## 7. Index it and search it

Once `enrich` and `topics` have run, build the retrieval index. It is derived
from the store, so it costs nothing you cannot rebuild:

```bash
uv run xbrain index build
# → 2495 items · 45 topics · 10686 superficies · 23145 chunks · 2495 perfiles
# →   omitidos: decorative 14 · empty_text 0 · failed_sources 65 · no_speech 111
# →   2.2s
```

The outputs in this section come from one corpus of 2,495 posts (`store-2495` in
[Measured versions](knowledge-index.md#measured-versions)).

Now query it. Results come back grouped by item, each with the fragments that
matched and where they came from:

```bash
uv run xbrain search "transformer attention" --limit 5
# → 1. 2051242195298968041  @xiathis (xIA) · 2026-05-04
# →    · [video_transcript] origin=asr trust=machine_extracted · via lexical
# →      ly talking about changing here is you know, where the norms go, …
# →    → verifica con: xbrain get 2051242195298968041 --surface video_transcript
```

That last line is the habit worth forming. `origin=llm` on a match means **xbrain
wrote it**, not the author — so when a summary is what matched, `verifica con`
names the surface that can actually settle the claim, and `get` hands you the
source in full:

```bash
uv run xbrain get 2051242195298968041 --surface video_transcript
```

`get` reads the **live store**, never the index, so it keeps working with
`data/index/` deleted. Add `--json` to either command for the same content as a
stable document — the human view and the JSON are two renderings of one model.

Narrow with any of the eight filters (`--from`/`--to`, `--author`, `--mine`,
`--topic`, `--kind`, `--origin`, `--has-surface`); they are applied before
anything is scored:

```bash
uv run xbrain search "agents" --topic ai-agents --from 2026-01-01 --kind x_article
```

**The index does not update itself.** Any stage that writes the store leaves it
behind, and `search` says so rather than quietly serving stale evidence:

```bash
uv run xbrain index status    # what it holds, how far behind it is
uv run xbrain index update    # touch only what changed (0.8s when nothing did)
```

Three limits to know before you judge the results. There is **no stemming**:
`agente` and `agentes` are different words, and on this corpus their top tens
share zero items. There is **no phrase search**: a query of several words
matches any of them, quoted or not. And it is **lexical, not semantic**: it
finds proper nouns and figures, not conceptual similarity. Without a vector
plane (below), every response says so (`degraded: ["no_embeddings"]`). Details,
costs and the rest of the limits: [The knowledge index](knowledge-index.md).

### Search by meaning (optional, and not shown to beat word search)

A second, **opt-in** plane ranks passages by meaning. `--strategy vector` uses
it alone; `--strategy hybrid` fuses it with the word ranking. Without it, both
tell you so:

```bash
uv run xbrain search "transformer attention" --strategy hybrid
# → "transformer attention" · estrategia lexical
uv run xbrain search "transformer attention" --strategy vector
# → Error: `--strategy vector` necesita vectores y este índice no tiene plano vectorial: …
```

Know the result before you spend the time. The one embedding model measured did
**not** beat word search, and the comparison stopped after one of three
candidates ([bake-off](embeddings-bakeoff.md)). That is why `lexical` stays the
default.

The plane needs two things the Quick start did not install. First, `numpy`, in
xbrain's own environment. Without it the build below fails only after it has
deleted your index, and even word search refuses until a plain `index build`:

```bash
uv pip install -e ".[embeddings]" --index-url https://pypi.org/simple
```

Second, an embedder, a program xbrain runs as a subprocess. The reference one,
`scripts/xbrain-embed`, needs `sentence-transformers` in a Python environment of
its own: 793 MB installed, plus 458 MB of model on first use.

```bash
EMBED=~/.xbrain-embed
uv venv --python 3.12 $EMBED
VIRTUAL_ENV=$EMBED uv pip install --index-url https://pypi.org/simple sentence-transformers
```

In `config.toml`, name that environment's Python before the script. The
script's shebang runs the first `python3` on your `PATH`, which does not have
the library. Use absolute paths: the command runs without a shell, so `~` is not
expanded.

```toml
[embeddings]
command = "/Users/you/.xbrain-embed/bin/python /path/to/xbrain/scripts/xbrain-embed"
model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
batch_size = 1024
```

MiniLM is the model the bake-off measured, and it needs no query or passage
prefix. Each batch is one process that loads the model again, so the default
`batch_size` of 64 would load it about 350 times on this corpus.

Build both planes (the word index is rebuilt too), then query:

```bash
uv run xbrain index build --embeddings --force
# → 2495 items · 45 topics · 10686 superficies · 23145 chunks · 2495 perfiles
# →   omitidos: decorative 14 · empty_text 0 · failed_sources 65 · no_speech 111
# →   259.8s
uv run xbrain search "transformer attention" --strategy hybrid --limit 3
# → "transformer attention" · estrategia hybrid
# → 1. 2051242195298968041  @xiathis (xIA) · 2026-05-04
# →    · [video_transcript] origin=asr trust=machine_extracted · via lexical+vector
# → …
# → 3. 2063758059307175994  @kylejeong (Kyle Jeong) · 2026-06-07
# →    · [x_article] origin=source trust=primary_source · via lexical+vector
# →      Human attention is still roughly the same (attention spans may have gotten worse), …
```

`via` names the ranking that found each passage. Result 3 is about human
attention, not transformers, and both rankings found it: neither one rules out
the wrong sense of a word.

It costs minutes to build and seconds per query, because every batch and every
query starts the embedder and loads the model again. Here the build took 260 s
and 1.15 GB of memory on a 16 GB laptop that was already swapping; the
bake-off's figures and conditions are in its
[§6](embeddings-bakeoff.md#6-coste-indexación-disco-latencia-y-memoria). A
filter (`--topic`, `--from`, …) sends the query back to `lexical`, declaring
`vector_filters_unsupported`. Every failure and its message:
[The vector plane](knowledge-index.md#the-vector-plane-and-hybrid--opt-in-and-not-the-default).

### Look around a post: `graph-expand`

`index build` also writes a small graph of posts and topics. From any post:

```bash
uv run xbrain graph-expand --item 2063609922667815064
# → This edge reflects co-occurrence in this corpus, not a relationship in the world.
# → item:2063609922667815064 → topic:agentic-engineering
# → item:2063609922667815064 → topic:ai-agents
# → item:2063609922667815064 → topic:ai-coding
uv run xbrain graph-expand --item 2063609922667815064 --max-hops 2
# → …
# → item:2063609922667815064 → topic:agentic-engineering → topic:claude-code
# → …
# → item:2063609922667815064 → topic:agentic-engineering → item:1934807329989623905
```

Believe the first line. Two topics are linked because xbrain assigned both to
enough of *your* posts, and for no other reason. Use the graph to pick what to
read next, then `get` those posts. It does not improve search either: used to
re-rank results, it lifted 0 of the 33 that only it could reach
([graph sweep](graph-threshold-sweep.md)).

Unlike `search`, it will not answer from an index that is behind the store. It
stops with ``Ejecuta `xbrain index update`.``

### Hand it to an agent: MCP

`xbrain mcp-serve` offers `search`, `get` and `graph_expand` to Claude Code or
any other MCP client, with the same answers as `--json`. From your xbrain
checkout:

```bash
claude mcp add xbrain -- uv run --directory "$PWD" --extra mcp xbrain mcp-serve
claude mcp get xbrain
# → xbrain:
# →   Scope: Local config (private to you in this project)
# →   Status: ✔ Connected
```

`--extra mcp` installs the MCP SDK when the client starts the server; without
it, `mcp-serve` exits naming `uv pip install 'xbrain[mcp]'`. The server is
registered for the directory you ran `claude mcp add` in. Claude Desktop, the
error messages and what the server can reach: [xbrain over MCP](mcp.md). What
the agent should do with the answers:
[Consuming xbrain from an agent](knowledge-for-agents.md).

## 8. See the whole corpus at a glance

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
uv run xbrain sync          # pull new bookmarks/tweets, re-render
uv run xbrain enrich        # enrich only the new posts
uv run xbrain topics        # refresh topic pages
uv run xbrain generate
uv run xbrain index update  # put the search index back in step with the store
```

`index update` is last because every command above it writes the store. Skip it
and nothing breaks — `search` detects it and warns — but you will be searching
yesterday's corpus. If you built the vector plane, run
`uv run xbrain index build --embeddings --force` instead: `update` never
re-embeds, so new passages have no vector and `vector`/`hybrid` answers declare
`vector_plane_behind` until you rebuild.

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
# the sentence for them). Run `reextract` FIRST: payloads are persisted now, and
# roughly a third of these re-parse offline for free. Only what that leaves needs
# this, and applying it re-fetches each from X — slow, human-paced browser work.
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

Two things about `refetch-truncated`'s count before you plan a session around
it. Its detector decides on **length alone** — 274 characters or more is
truncated unconditionally — so the total is a work list biased towards flagging,
not a census: on one corpus (2,404 items, read 30-ago-2026) it flags 707, of
which only 146 are 290 characters or shorter. And a stored payload is not
automatically a free repair: 702 of those 707 have one, but a payload only helps
when X included the long-form body at capture time, so on that same corpus
`reextract` recovers the text of 222 of them and the remaining ~480 still need
the network. Run the `reextract` dry pass and read its count — that is the number
you actually get for free.

The applying form of each rewrites `items.json`, and auto-snapshots `data/`
before it does, so a bad run is undoable with `xbrain snapshot restore`.
`refetch-truncated` and `refresh-quoted` invalidate the summaries of the items
they repair — re-run `xbrain enrich` afterwards so those get written against the
evidence they were missing.

Stuck? → [Troubleshooting](troubleshooting.md).
