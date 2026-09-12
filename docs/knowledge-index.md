# The knowledge index — operating it

`xbrain index build` turns the store into a persistent SQLite/FTS5 index under
`data/index/`, and `xbrain search` reads it. `xbrain get` does not — it serves
evidence from the live store, and keeps working with `data/index/` deleted.
This page is the operational half: when to rebuild, what it costs, and what the
lexical baseline cannot do. The shape of it — planes, manifest, invalidation — is in
[ARCHITECTURE.md](../ARCHITECTURE.md#the-persistent-index).

Everything here was measured on one corpus and re-derived on 2026-09-12: **2,474
items, 45 topics**, macOS/APFS, a warm page cache. Your numbers will differ; the
commands that produce them are printed beside each one, so read these as a shape
rather than as a promise.

## The five commands

```bash
uv run xbrain index build     # build data/index/ from scratch and seal it
uv run xbrain index update    # touch only what changed since the last build
uv run xbrain index status    # what the index holds, and how far behind it is
uv run xbrain search "…"      # ranked items with citable fragments
uv run xbrain get <item-id>   # one item's evidence, from the STORE (never the index)
```

`build`, `update` and `status` write or read only `data/index/`. None of them
touches `items.json`, `vocab.yaml` or `topics.json`, and none of them takes a
snapshot, because there is nothing of yours to lose: **`data/index/` is derived
and reconstructible**, it is never versioned, and deleting it costs one
`build`.

`search` opens the database `mode=ro`, so a stray write is an error rather than
a silent repair; `get` opens no database at all. Neither calls an LLM and
neither touches the network.

## When to rebuild, and when to update

Indexing is **manual by design**. Nothing runs it for you, so the failure that
actually happens is *you ran `enrich` and did not reindex* — which is why the
index declares its own staleness instead of hoping you remember.

| You did this | Run this |
|---|---|
| `extract`, `fetch`, `enrich`, `topics`, `vocab`, `digest-video`, `describe` | `index update` |
| upgraded xbrain and `status` says the manifest is incompatible | `index build --force` |
| deleted `data/index/` | `index build` |
| nothing — you just want to know | `index status` |

`update` is the normal path. It compares each item's fingerprint against the
stored one and rewrites only what moved, inside a single transaction, so an
interrupted run leaves the index at its previous state rather than half-applied.

`build --force` is the sledgehammer: it deletes the manifest first and the
database second, so a rebuild interrupted halfway never leaves a manifest
standing over an empty base.

## What it costs

Measured with `time uv run xbrain <command>` on the corpus above, wall clock,
five runs for `search` and the median reported:

| Operation | Cost |
|---|---|
| `index build` (2,474 items → 22,933 chunks) | 3.1 s of work, 9.5 s wall including the store load |
| `index update`, nothing changed | 0.8 s of work, 1.4 s wall |
| `index status` | 1.8 s wall — it runs `PRAGMA quick_check`, which the query doors do not |
| `search`, `--limit 10` | 0.65 s wall (median of 5: 1.15 · 0.60 · 0.65 · 0.71 · 0.65) |

`search`'s wall clock is dominated by loading `data/items.json` (17.3 MiB here),
not by the query: verification verdicts are hydrated from the **live store**, so
every call reads it.

On disk:

| File | Size |
|---|---|
| `data/index/knowledge.db` | 52 MiB |
| `data/index/manifest.json` | 1.1 KiB |
| `data/items.json`, for scale | 17.3 MiB |

So the index costs roughly **3× the store it indexes**. Most of that is FTS5's
inverted index over two planes — chunk bodies and item profiles — plus the chunk
rows themselves.

What a build reports it skipped, on this corpus:

```
omitidos: decorative 14 · empty_text 0 · failed_sources 65 · no_speech 111
```

Those are not errors. A decorative image description, a silent video and a link
whose fetch failed carry no indexable prose, and indexing them would put empty
rows in front of a scorer.

## Reading a result

```bash
$ uv run xbrain search "transformer attention" --limit 2
"transformer attention" · estrategia lexical
· Estrategia léxica (sin embeddings): recupera nombres propios, cifras y frases
  exactas, no similitud conceptual.

1. 2051242195298968041  @xiathis (xIA) · 2026-05-04
   https://x.com/xiathis/status/2051242195298968041
   topics: llm-foundations, frontier-models
   resumen (llm): Clase de Stanford que recorre las decisiones de arquitectura…
   · [video_transcript] origin=asr trust=machine_extracted · via lexical
     ly talking about changing here is you know, where the norms go, or you know…
   → verifica con: xbrain get 2051242195298968041 --surface video_transcript
```

Three things in that block are the whole point of the layer.

**`origin` and `trust` travel with every fragment.** `origin=asr` means a machine
heard it, not that the speaker said it; `origin=llm` on a summary means xbrain
wrote it, not the author. The trust class is derived from the origin by one
total table, so a consumer never has to infer it from the surface name.

**The last line is the verification instruction.** When a match lands on a
summary — text xbrain generated — `verifica con` names the underlying surface
that can actually settle the claim. Run that `xbrain get` and you are reading
the source, not the paraphrase. When no such source exists the line says so —
`⚠ no_underlying_source: la coincidencia es texto derivado y este item no
conserva ninguna fuente primaria que la sustente` — instead of pointing you
back at the summary.

**`--json` is the same model, and the richer view.** The human view and `--json`
are two renderings of one `SearchResponse`; the renderer never reaches back into
the store or the index, so the two cannot disagree and anything you read above
is a field you can parse. The containment runs one way only: the human view
**selects**, and it turns some fields into Spanish prose (a `degraded` flag
becomes a sentence, an empty `verify_with` becomes the `no_underlying_source`
warning). The JSON below carries fields it never prints — `schema_version`, the
echoed `filters`, `manifest_version`, `built_at`, and per match the `chunk_id`,
`title`, `score`, `lexical_rank`, `vector_rank` and `locator`. Read the human
view to judge a result; parse `--json` to consume one:

```bash
uv run xbrain search "transformer attention" --limit 1 --json
# → {"schema_version": "2", "query": …, "strategy": "lexical",
#    "index": {"manifest_version": "4", "built_at": …,
#              "corrupt_chunks_excluded": 0, "degraded": ["no_embeddings"]},
#    "results": [{"rank": 1, "item_id": …, "matches": [{"chunk_id": …,
#       "origin": "asr", "trust_class": "machine_extracted",
#       "matched_by": ["lexical"], "lexical_rank": 1, "locator": {…}}],
#       "available_surfaces": […], "verify_with": ["video_transcript"]}]}
```

## `get` works with the index deleted

```bash
rm -rf data/index/
uv run xbrain get 1880184389218496770     # still works
uv run xbrain search "transformer"        # refuses, and names the fix
# → Error: No hay índice en …/data/index. Constrúyelo con `xbrain index build`.
```

That asymmetry is deliberate. `get` serves evidence from the live store, so an
index that could answer it would be a second copy of the corpus that nothing
invalidates — and the day the two disagreed there would be no way to know which
one the reader was shown.

### Ask for the surface you want

A bare `get` is an **index card, not a dump**: item metadata, topics, the
`summary` body, and the list of surfaces this item has. Every other body — the
article, the transcript, the thread, the frame descriptions — is asked for by
name:

```bash
uv run xbrain get <id>                              # metadata + topics + summary
uv run xbrain get <id> --surface external_article   # that body, whole
uv run xbrain get <id> --surface external_article --surface video_transcript
```

An unknown `--surface` is refused listing the ones the item actually has, so the
first command is also how you find out what the second can ask for. That is what
"complete evidence" means here: every surface is *reachable*, by selection and
by pagination — not everything at once.

### Paginating: the cursor is not the whole request

Over the character budget (`[index].get_char_budget`, 40,000 by default) the
bundle is truncated **and says so**, handing back a cursor. The cursor is an
offset into a sequence that **the request defined** — the chunks of the surfaces
you selected, in emitter order, or their ranking when you passed `--query` — and
the response does not carry that request back to you. So a continuation repeats
the original `--surface` flags, in the same order, and the same `--query`:

```bash
uv run xbrain get <id> --surface external_article --query "attention" --cursor q:12
```

Drop them and you resume inside a different sequence: the cursor either lands in
the default selection and returns an empty page, or is refused outright, because
the positional cursor (`<surface>:<chunk>`) and the ranked one (`q:<offset>`)
each refuse the other's shape by name rather than restarting at zero. You never
have to assemble that line yourself — the truncation warning prints the exact
command to run, flags and all:

```
⚠ Truncado. Continúa con: xbrain get <id> --surface external_article --cursor 0:7
```

`search` paginates the same way, for the same reason: its cursor (`s:<offset>`)
is a position in the ranking that the query **and its filters** define, so the
printed continuation repeats every filter and the page size beside `--cursor`.

## The eight filters

All eight of them are pushed into SQL before anything is scored, so a narrow
query is cheaper than a broad one rather than more expensive:

```bash
uv run xbrain search "agents" --from 2026-01-01 --to 2026-06-30
uv run xbrain search "agents" --mine              # shorthand for --source own_tweet
uv run xbrain search "agents" --author simonw
uv run xbrain search "agents" --topic ai-agents --topic multi-agent-systems
uv run xbrain search "agents" --kind x_article
uv run xbrain search "agents" --origin asr
uv run xbrain search "agents" --has-surface video_transcript
```

`--mine` and `--source` are mutually exclusive: two ways of setting one field is
a way of setting it to two values, and resolving that silently would hand back a
corpus nobody asked for. Unknown values are refused with the valid ones listed —
an unknown `--topic` prints the whole vocabulary.

**`xbrain eval` is a different surface, and it can push only two of the eight.**
The evaluation harness measures the baseline retriever, which supports
`has_surfaces` and `origins`; a golden-set case declaring any of the other six is
reported **unmeasured**, never `0.0`. A zero from a filter nobody applied reads
as "retrieval failed at filtering" when the truth is that the instrument was not
there.

## Known limits of the lexical baseline

These are declared, not discovered later. Plan 03's vector layer is what
addresses the first two.

**No stemming, and it is visible.** FTS5 has no multilingual stemmer, and the
English one would wreck the Spanish half of a bilingual corpus, so there is
none. Singular and plural are different words. Measured on this corpus, the top
ten for `agente` and for `agentes` share **0 items**; `transformer` and
`transformers` share 7. Search for the form you expect to be written, or for
both.

**Diacritics, on the other hand, fold.** The tokenizer is
`unicode61 remove_diacritics 2`, so `atencion` and `atención` return the same
ranking. If an accent seems to change your results, the cause is elsewhere —
usually a different word entirely.

**IDF is relative to this corpus.** bm25 discounts a term by how common it is
*here*, not in the language. `el` appears in 27.2 % of real chunks, so it is
discounted heavily; a word that reads like a function word to you may be rare to
the index and go undiscounted. This is also why a fixture-sized index ranks
differently from the real one.

**Lexical means lexical.** It retrieves proper nouns, figures and exact phrases,
not conceptual similarity. Every response says so: `degraded: ["no_embeddings"]`,
read off the manifest rather than hard-coded, so the day a vector backend writes
an `embeddings` block the flag stops appearing on its own.

**A strategy with no backend degrades, it does not refuse.** `--strategy vector`
answers with `lexical` and labels the gap `vector_not_implemented`, in the
response and in a warning line. A *typo* (`--strategy vectro`) is refused
instead: answering it with lexical results would turn a mistake into a
measurement.

**The staleness signal is cheap and falible, in one known direction.**
`index_behind_store` compares the `mtime_ns` and size of the three inputs against
what the manifest recorded. A `touch` with no edit is a false positive, and that
is accepted — a false positive costs one warning, a false negative costs serving
stale evidence as fresh. The blind spot is the mirror image: **a replacement of
exactly the same size with the mtime preserved** (`cp -p`, `rsync -a`, `tar -x`,
`unzip`, a restored backup) is invisible to it, deterministically and forever.
Nothing here promises freshness from `mtime` + size; the answer to *what
actually changed* is the deep fingerprint that `build`, `update` and `status`
compute. After any such restore, run `index status` — which compares
fingerprints — or just `index build --force`.

**The measured half of that sentence is macOS/APFS.** ext4's `mtime_ns`
granularity has not been measured here. The contract leans on the *size*, which
does not depend on the filesystem.

## Configuration

Everything has a default; the whole `[index]` section is optional. See
[`config.toml.example`](../config.toml.example).

```toml
[index]
# dir = "index"                 # under data/ — must resolve INSIDE data/
# max_matches_per_item = 3      # fragments one item may cite in a search
# get_char_budget = 40000       # per-response ceiling before truncate + cursor
```

`max_matches_per_item` is what stops a long transcript filling the top ten with
ten adjacent windows of itself. `dir` is validated at config load: an absolute
path, a `..` or a symlink that escapes `data/` is refused, because
`index build --force` deletes and recreates whatever it finds there.

---

Something broken? → [Troubleshooting](troubleshooting.md#the-knowledge-index).
