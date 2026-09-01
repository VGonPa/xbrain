# The knowledge index — operating it

`data/index/` is a **derived, reconstructible** artefact. It is never versioned, it is never a
second source of truth, and losing it costs one command. This page is how to run it, what it
costs, and — the part worth reading twice — what the lexical baseline **cannot** do.

```bash
xbrain index build            # from scratch (refuses to overwrite; use --force)
xbrain index update           # only what changed in the store
xbrain index status [--json]  # what it holds, and how far behind the store it is

xbrain search "agentes que evalúan su propio trabajo" [--json]
xbrain get <item-id> [--surface external_article] [--query "..."] [--json]
```

---

## When to reindex

Indexing is **manual by decision**. There is no daemon and no `launchd` job, because the spec
requires the cost to be measured before any of that is decided — and this page is that
measurement.

Reindex after anything that changes indexable text:

| you ran | what changed | run |
|---|---|---|
| `extract` / `fetch` / `import-archive` | new items, new bodies | `xbrain index update` |
| `enrich` | summaries and topic assignments | `xbrain index update` |
| `topics` | overviews and topic notes | `xbrain index update` |
| `digest-video` / `describe` / `redescribe-frames` | transcripts, captions, image prose | `xbrain index update` |
| `vocab` | topic descriptions — which enter every assigned item's PROFILE | `xbrain index update` (it rebuilds the profiles) |
| upgraded xbrain and `index update` refuses | the emitter, the chunker or the SCHEMA moved (schema **2** since round 02) | `xbrain index build --force` |

**You do not have to remember.** Two independent signals say so for you:

- every `search` compares the manifest's `mtime`+`size` of `data/items.json` against the file
  right now, and declares `index_behind_store` in the response — cheap enough to do on every
  query, and it still ANSWERS, because possibly-stale evidence is usable as long as it says so;
- `xbrain index status` loads the store and reports **how many** items changed, added or
  disappeared — a number, not a flag, because "something changed" does not distinguish a
  `touch` from a hundred re-enriched items.

A `touch` with no edit is a false positive on the cheap signal, and that is accepted: a false
positive costs one warning, a false negative costs serving stale evidence as fresh.

---

## What it costs — measured, on the real corpus

Measured 2026-09-01 on `data/items.json` with **2,404 items** (sha256 `f76341a3…`), 45 topics,
on an Apple-silicon laptop, with the shipped chunker (`target=800, overlap=0`, chunker v2).
**These are a photograph, not a constant.** Re-derive them; the corpus grows.

| measurement | value |
|---|---|
| `index build`, full | **1.37 s** median of 5 (1.20 – 1.54) |
| loading the store (not counted above) | 0.22 s |
| `index update`, 0 items changed | 0.15 s — 0 writes to the DATABASE (the manifest is always rewritten: it records the current cheap signal) |
| `index update`, 1 item changed | 0.16 s — 3 chunks out, 3 in |
| `index update`, 100 items changed (SYNTHETIC: 100 summaries rewritten in memory) | 0.26 s — 645 chunks out, 645 in |
| `data/index/knowledge.db` | **52.4 MB** (manifest 1 KB) |
| chunks | **22,286** — 10,160 surfaces, 2,404 profiles |
| chunks per item | median 3, mean 9.0, max 712 |
| `search` latency, 23 golden-set cases with their filters, **`--limit 10`** | p50 **27.7 ms**, p95 57.3 ms — medians of 3 passes (p50 27.6 / 28.1 / 27.7; max 67–97 ms) |
| the same, **`--limit 20`** (the depth the baseline runs at) | p50 **35.6 ms**, p95 69.8 ms — medians of 3 passes (max 76–109 ms) |
| cold `xbrain search … --json` (process start + config + store load + one query) | 0.77 – 0.86 s, 3 runs |
| omitted, by cause | 63 failed fetches · 108 silent videos · 14 decorative images · 0 empty |

`search` latency is index-open + score + fingerprint-verify + group + hydrate, with the store
**already loaded**, one warm-up pass first so the 52 MB database sits in the page cache, sqlite
3.51.2, load average **9.0** during the run (M-4: the row used to read `p50 26 ms · p95 43 ms ·
max 66 ms` with no `--limit`, no repetitions and no load stated — and `--limit` alone moves the
p50 by ~1.3×, so that figure is retired rather than reinterpreted). `search` hydrates
verification from the live store (a verdict copied into the index could never be invalidated),
so the store is not optional; the cold CLI figure above is what that costs end to end.

**Whether to automate this is now decidable and the answer looks like "not yet":** a full
rebuild costs under two seconds. There is no incremental-cost problem to solve, and the honest
follow-up question is not *how do we automate the update* but *should `enrich` and `topics`
simply call `index update` on their way out*.

### Chunks by surface

| surface | chunks | | surface | chunks |
|---|---:|---|---|---:|
| `external_article` | 5,919 | | `image_description` | 922 |
| `video_transcript` | 3,265 | | `quoted_post` | 797 |
| `x_article` | 3,234 | | `topic_note` | 526 |
| `post` | 2,404 | | `video_digest` | 441 |
| `summary` | 2,404 | | `topic_overview` | 132 |
| `video_frame` | 2,197 | | `topic_description` | 45 |

### Staleness after a typical `enrich` (Plan 02 §8.7)

Measured 2026-09-01 on the same store, from the `enriched.enriched_at` the store records: all
2,404 items carry an enrichment and they were (re)enriched on **11 distinct days**. Re-enriching
an item changes its fingerprint, and `index update` deletes and rewrites **every** chunk and the
profile of a changed item, so *chunks stale after that run* is the chunks those items own in
today's index (chunker v2):

| run (by day) | items | chunks they own today | | run | items | chunks |
|---|---:|---:|---|---|---:|---:|
| 2026-05-19 | 725 | 3,060 | | 2026-07-09 | 7 | 76 |
| 2026-05-23 | 14 | 34 | | 2026-07-10 | 20 | 446 |
| 2026-06-09 | 53 | 240 | | 2026-08-12 | **1,098** | **8,503** |
| 2026-06-25 | 22 | 244 | | 2026-08-30 | 88 | 1,857 |
| 2026-06-30 | 4 | 37 | | 2026-08-31 | 40 | 514 |
| 2026-07-05 | 333 | 6,572 | | | | |

Leaving out the 2026-08-12 backfill, a **typical run re-enriches 31 items (median of 10 runs;
4 – 333) and leaves 345 chunks stale (median; 34 – 6,572), i.e. 1.55 % of the 22,286** — well
inside what `update` handles in a fraction of a second. What `enrich` would process if run
today: **1** item (a re-enrichment), owning 3 chunks. The "100 items changed" row above is a
SYNTHETIC population and is labelled as such; it sits between a typical run and the largest.

### Interruption

`Ctrl-C` mid-build rolls the transaction back and writes **no manifest**, and it is the manifest
that carries the safety property: an index with no manifest is **refused** by every query rather
than answered partially. **This was only true for a FRESH build until round 02 (C-1):** `--force`
— the command every rebuild error recommends — kept the OLD manifest standing while the new
database was written, so an interrupted forced rebuild left a manifest every query accepted over
an empty base (`status` said nothing, `search` answered "no results"). A forced rebuild now
removes the manifest **before** the database. Re-measured through the CLI on the real corpus
(SIGINT sent from a driver; a shell's background job ignores it), three reachable states, all
failing closed:

| `SIGINT` at | what is left | `status` | `search` / `update` | recovery |
|---|---|---|---|---|
| 0.4 s — before `build()` runs (interpreter + store load) | the **previous** index, intact | healthy | answer normally | nothing to recover |
| 0.9 – 1.8 s — mid-transaction | database with **0** rows, **no manifest** | `incomplete`, names `xbrain index build` | exit 1, «No hay manifest…» | `xbrain index build` → 22,286 chunks |
| 2.0 s — after the commit, before the manifest | database with all **22,286** rows, **no manifest** | `incomplete` | exit 1 | same |

*(The F-13 paragraph this replaces measured a FRESH build at 0.55 s / 0.75 s and reported the
last two states; the first is specific to `--force`, whose previous index survives only if the
signal lands before the rebuild starts.)*

### The manifest describes the base, or the base is refused (C-3, A-3)

`counts` and `skipped` in the manifest are **read back from the database** by the same function
after a build and after an update — five `COUNT(*)` plus a `SUM` over three per-item omission
columns on `items` (schema **2**). Until round 02 an update carried `surfaces`, `skipped` and
`failed` over from the previous manifest and adjusted the rest by hand, which drifted (topic
chunks were added on every vocabulary rebuild and never subtracted), so `index status --json`
published the previous population as current. Because the manifest now describes the base by
construction, a base that contradicts it is DETECTED: `index update` refuses (`El índice no
contiene lo que su manifest declara (topics 0 != 45)` + the rebuild command) instead of silently
rewriting every item and dropping the topic plane — the recovery path the Claude gate measured
losing 45 topics, 616 surfaces and 703 chunks for good — and `index status` reports it as
incomplete, naming `xbrain index build --force`. `status` also applies the same version check
`search` and `update` do, so the three instruments agree on one state.

---

## The lexical baseline — the number Plan 03 has to beat

Measured on the same corpus, against the 23 scorable cases of `eval/golden-set.yaml`
(`xbrain eval`). All 23 score; none is unmeasurable, and none returned an empty result set.

| metric | mean |
|---|---:|
| `recall@1` | 0.6034 |
| `recall@5` | 0.7665 |
| `recall@10` | 0.8264 |
| `recall@20` | 0.8300 |
| `precision@10` | 0.3152 |
| `MRR` | 0.8179 |

| stratum | cases | recall@1 | recall@10 | precision@10 | MRR |
|---|---:|---:|---:|---:|---:|
| `cruzado_idioma` | 5 | 0.6091 | 0.6182 | 0.130 | 0.800 |
| `enterrado` | 8 | 0.3958 | 0.7396 | 0.269 | 0.601 |
| `exacto` | 5 | 0.3667 | 0.8000 | 0.620 | 0.622 |
| `filtros` | 2 | 0.4167 | 1.0000 | 1.000 | 1.000 |
| `multimodal` | 6 | 0.7500 | 0.8333 | 0.233 | 0.833 |
| `resumen` | 1 | 1.0000 | 1.0000 | 0.100 | 1.000 |
| `semantico` | 9 | 0.4680 | 0.6675 | 0.144 | 0.744 |
| `topic` | 3 | 1.0000 | 1.0000 | 0.100 | 1.000 |
| `expansion` | — | *sin cobertura* | | | |

`recall@k` counts deduplicated **owners**; `surface_recall@k` counts **chunks**. Under the same
`k` they do not measure the same population — compare each column with its own value in the next
run, never one with the other. A cell reading *sin cobertura* is **not a zero**: no case in that
bucket could measure that metric, and a zero there would claim retrieval failed where nobody
asked.

`expansion` has no cases because it is the stratum the minimal graph is for (Plan 04). `thread`
and `user_note` have **no data in the corpus at all**, so no case can exist for them.

**`filtros` publishes `recall@10 = precision@10 = 1.000` BY CONSTRUCTION, and that pair does
not measure ranking (F-11).** Its two cases define `relevant_items` as *exactly* the population
the filter selects (2 and 3 items), so a filter that reaches the backend returns that population
and nothing else. The number is not vacuous — it can come out near 0 if the filter never reaches
the `WHERE`, which is the defect that motivated the cases (rule 2 is satisfied: there is a way
for a different answer to come out) — but read it as *the filter runs*, never as *ranking is
perfect there*. Its effect on the aggregate, said out loud: `precision@10` 0.2500 → **0.3152**
and `recall@10` 0.8099 → **0.8264**.

---

## What the lexical baseline cannot do

Written down because a limit nobody records gets quietly attributed to the corpus instead of to
the retriever.

- **No stemming.** FTS5 has no multilingual stemmer and the English one would wreck the Spanish
  half of a bilingual corpus, so `agent` does not match `agents`. This is what a vector layer is
  for.
- **IDF is relative to THIS corpus.** A word that reads as a function word can still be rare to
  the index and therefore undiscounted. Measured 2026-09-01 on the shipped chunker
  (`target=800, overlap=0`, v2; 2,404 items, 22,286 chunks, store sha256 `f76341a3…`): `el`
  sits in **6,070 of the 22,286 chunks (27.2 %)** on the real corpus and in 1 of 49 in the test
  fixture — bm25 behaving correctly on each corpus it was given, and ranking `el` very
  differently in the two. *(This line read `5,748 (31 %)` until F-4: that is the figure for the
  PROVISIONAL chunker v1, `1200/150`, whose corpus is 18,320 chunks — a v1 number published
  under a v2 population. The chunker moved and the derived figure did not, which is rule 6.)*
- **The terms are ORed, not phrased.** Every term is quoted as an FTS5 string and joined with
  `OR`. A conjunction returned zero rows for 18 of 21 cases (measured), so the disjunction is
  what makes bm25 able to rank at all — but it also means no phrase query, and therefore no
  benefit from window overlap.
- **`KnowledgeSurface.language` is `None` on almost everything.** The only language the store
  records is `ContentSourceSuccess.language`, populated by transcriptions. It is never guessed,
  and there is **no language filter**.
- **No embeddings, and `strategy` says which retriever ran.** Every response declares
  `degraded: ["no_embeddings"]`. Asking for a strategy that has no backend
  (`vector`, `hybrid`, `hybrid_graph`) does not fail — spec §9.3 requires that *lexical sigue
  operativo* — it answers lexically, labels the response `strategy: "lexical"` and adds
  `<requested>_not_implemented` to `degraded`. `xbrain eval` does the same: the report is
  headed with the strategy that RAN and names the one that was asked for. Until F-2 was fixed
  the `degraded` flag was there but the `strategy` field echoed the request, so a `hybrid`
  answer produced by bm25 came back labelled `hybrid`, and `xbrain eval --strategy vector`
  published a report headed `vector`. A strategy that is in no contract at all
  (`--strategy banana`) is a validation error naming the valid ones.

---

## What a match and a bundle carry (round 02)

- **A `search` match names the SURFACE's author and locator (A-1).** `SearchMatch.attribution`
  is the quoted author on a `quoted_post` — never the poster — and `locator` is the surface's
  own (`source_index`, `content_kind`, the source URL; `media_index` on an image description)
  with the chunk's character range on top. The human view prints `autor: @handle (Name)` under a
  match whose author is not the item's. Until round 02 the index stored both and `search` threw
  them away: k07's quoted post came back with `attribution: null` under the poster's name.
- **`get --query` paginates with a cursor (A-2).** The ranking is deterministic, so the cursor is
  `q:<offset>` into it; `truncated: true` always comes with one, and the two cursor shapes
  (`q:<offset>` for a query, `<surface>:<chunk>` positional) refuse each other by name.
- **`get` keeps the ASR/VLM producer (A-4).** `transcribe_command` and `vision_command` travel in
  `QueryContext` from the same config definition the build uses, so a transcript's `producer` is
  the configured transcriber in `get` exactly as in the index.

## Troubleshooting the index

See [`docs/troubleshooting.md`](./troubleshooting.md#the-knowledge-index) for the error
messages and their fixes: index absent, manifest incompatible, corrupt chunks, index behind the
store, and a query that will not match an accent.

## Configuration

```toml
[index]
# dir                  = "index"   # under data/; raramente se toca
# max_matches_per_item = 3         # tope de matches por item en `search`
# get_char_budget      = 40000     # techo por respuesta de `get` antes de truncar + cursor
```

`dir` is a **name**, not a path: it is resolved under `data/` and proven contained there, so a
symlink cannot move the database outside the data root.
