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
| `enrich` | summaries and topic assignments — and with the assignments, the MEMBERS and the `stale` bit of every topic row they enter or leave (H1) | `xbrain index update` |
| `topics` | overviews and topic notes | `xbrain index update` |
| `digest-video` / `describe` / `redescribe-frames` | transcripts, captions, image prose | `xbrain index update` |
| `vocab` | topic descriptions — which enter every assigned item's PROFILE | `xbrain index update` (it rebuilds the profiles) |
| `refresh-quoted` / any repair of a source's author, title, language or URL that leaves its body untouched | the attribution and locator `search` serves on every match (A-1) | `xbrain index update` — since round 03 the item fingerprint covers every column `surfaces` stores, not the text alone (G-5); before, this repair left `update` at «0 cambiados» and `search` serving the old author |
| upgraded xbrain and `index update` refuses | the emitter, the chunker or the SCHEMA moved (schema **3** since round 07 — U-5, the fingerprint covers the whole served evidence; **2** since round 02) | `xbrain index build --force` |
| upgraded xbrain across round 03 and `index update` reports **every** item changed | the fingerprint's definition moved (G-5), so an index built before it compares unequal on all items — a ONE-TIME full rewrite (2,404 items, measured), after which the next `update` is back to 0 | nothing: let it run once |
| upgraded xbrain across round 05 and every `search` warns `index_behind_store` with `status` reporting 0 items and 0 topics changed | the manifest predates the three-file signal (P1a): its vocab/topics entries read back as zeros and compare unequal to the live files — the direction the signal fails in, towards the warning | `xbrain index update` once: it re-seals the manifest with the full signal |
| upgraded xbrain across round 04 and `index status` reports `N topics con miembros desfasados` with 0 items changed | an index updated by the pre-H1 code kept the topic rows of every topic an `enrich` moved an item into or out of; `status` now reads those rows back and compares them | `xbrain index update` — it rewrites exactly those rows (`N topics con miembros recalculados`) and nothing else |

**You do not have to remember.** Two independent signals say so for you:

- every `search` compares the manifest's `mtime`+`size` of `data/items.json`, `data/vocab.yaml`
  **and** `data/topics.json` against the files right now, and declares `index_behind_store` in
  the response — three `stat` calls, cheap enough to do on every query, and it still ANSWERS,
  because possibly-stale evidence is usable as long as it says so. *Three files since round 05
  (P1a): the signal watched `items.json` alone, and the two commands in the table that write
  the other two — `topics`, `vocab` — never touch it, so `search` after either answered over the
  old topic plane with nothing declared (the round-05 gate's probes B and C: a term added to a
  topic note or description, `items.json` byte-identical, 0 results, `degraded:
  ["no_embeddings"]`, while `status` reported `topics_changed=1`).*
- `xbrain index status` loads the store and reports **how many** items changed, added or
  disappeared — a number, not a flag, because "something changed" does not distinguish a
  `touch` from a hundred re-enriched items — and, since round 04, **how many topic rows** the
  base holds that are not what the store implies (`topics_changed`, H1).

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
| `index update`, 0 items changed | **0.43 s** median of 5 (0.42 – 0.46), re-measured 2026-09-02 at load 8.8–9.7 — 0 writes to the DATABASE (the manifest is always rewritten: it records the current cheap signal). *Was 0.15 s until round 06 made `update` pay `PRAGMA quick_check` (D-1) and round 07 the column check (U-4); the old row stood for two rounds after the cost moved — F7-4, rule 6 in the measurement table itself.* |
| `index update`, 1 item changed | **0.44 s** — 3 chunks out, 3 in (was 0.16 s, same population) |
| `index update`, 100 items changed (SYNTHETIC: 100 summaries rewritten in memory) | **0.57 s** — 645 chunks out, 645 in (was 0.26 s, same population) |
| cold `xbrain index status --json` / `index update --dry-run --json` / `search … --json` | 0.86–1.24 s / 0.85–0.88 s / 0.55–0.63 s wall, 3 runs each, 2026-09-02, load 8–10 |
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
4 – 725) and leaves 345 chunks stale (median; 34 – 6,572), i.e. 1.55 % of the 22,286** — well
inside what `update` handles in a fraction of a second. *(Until round 04 this line read
`4 – 333` for the same 10 runs — a range no population in the table produces: the 10 runs
include the 725-item day of 2026-05-19, and `4 – 333` is the range of the 9 runs that exclude it
too, whose median is 22, not 31 (H4). Re-derived 2026-09-02 on the same store: the median, the
chunk median and the 1.55 % are all figures of the 10-run population and stand; only the range
was wrong. The DECLARED population is kept and the range corrected, rather than the other way
round, because the text names one exclusion — the backfill — and excluding the 725-item day as
well would need a criterion the text never states; a range narrowed by an unstated exclusion is
the number rule 2 exists to stop. For the reader who wants the sensitivity: the 9-run
population gives a median of 22 items / 244 chunks, range 4 – 333 / 34 – 6,572, 1.09 %.)*
What `enrich` would process if run today: **1** item (a re-enrichment), owning 3 chunks. The
"100 items changed" row above is a SYNTHETIC population and is labelled as such; it sits
between a typical run and the largest.

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
columns on `items` (schema **2**; **3** since round 07, when the chunk fingerprint's definition moved — U-5 below). Until round 02 an update carried `surfaces`, `skipped` and
`failed` over from the previous manifest and adjusted the rest by hand, which drifted (topic
chunks were added on every vocabulary rebuild and never subtracted), so `index status --json`
published the previous population as current. Because the manifest now describes the base by
construction, a base that contradicts it is DETECTED: `index update` refuses (`El índice no
contiene lo que su manifest declara (topics 0 != 45)` + the rebuild command) instead of silently
rewriting every item and dropping the topic plane — the recovery path the Claude gate measured
losing 45 topics, 616 surfaces and 703 chunks for good — and `index status` reports it as
incomplete, naming `xbrain index build --force`. `status` also applies the same version check
`search` and `update` do, so the three instruments agree on one state.

**And `search` applies the same four version checks as `update` and `status` — schema, emitter, chunker, and since round 05 (M-1) the chunker PARAMETERS too.** Until then `open_for_query` compared `chunker_params` only when handed them and the query path never did: over a manifest whose parameters had moved, `update` refused and `status` said «ninguna consulta lo usará» while `search` answered — chunks cut differently under identical ids. The parameters now travel in `QueryContext` from the same `IndexOptions` the build uses.

**And since round 03, `search` runs the same count check (G-2, B-c).** Until then a query
compared versions and schema only: over a base amputated behind the manifest's back it answered
«Sin resultados» and declared nothing, while `update` and `status` refused it. The path that
actually produced that base was `index update --dry-run` over a deleted `knowledge.db` (52 MB, a
natural clean-up target; the 1 KB manifest stays): the write door created an EMPTY database
before the consistency check raised, and the next `search` exited 0 over zero rows. Now no door
but `build` creates the file, `update` refuses before touching the disk, and every `search` runs
five `COUNT(*)` (0.04 ms on the real index) against the manifest. With a manifest standing beside
a missing database the advice names `xbrain index build --force`, because plain `build` refuses
while a manifest exists — **on all three doors since round 07 (U-2)**: until then `status` tested
`exists()` by itself and read that state as an index never built, answering `incomplete: false`,
`+2404 nuevos` and «actualiza con `xbrain index update`», the advice `update` then refused (both
round-07 gates, independently). `status` now asks `require_database` like `search` and `update`
and publishes its sentence, and `get` — which reads the store — keeps answering.

### `status` runs `PRAGMA quick_check`; the open door does not (B-1)

The probes every door runs — page 1, `sqlite_master`, one `MATCH` per FTS plane — cost 0.01 ms
and see what a query would see first. They do not see a damaged page none of them touches: 16 KB
of `0xff` over pages 17–20 of the real index left `status` saying `incomplete: false` while
`quick_check` reported «btreeInitPage() returns error code 11» (the round-05 gate, B-1). A query
still fails closed the moment it reaches the page (G-4), so no instrument contradicted another;
but `status` is the explicit command an operator runs to find out, and since round 05 it runs
the whole-file check and declares the damage naming `xbrain index build --force`. `search` and
`update` keep the cheap probes. **Cost, on the 52 MB real index:** 155–167 ms measured by the
gate (3 runs, its load unstated); **362–850 ms, median 425 ms**, re-measured in round 05 (5 runs of
`PRAGMA quick_check(1)` over a read-only connection, load average **8.6**), which puts a cold
`xbrain index status` at ~1.0 s of wall clock against 0.78 s in round 04 — the same command, one
whole-file read heavier.

### The manifest describes the snapshot that was indexed, not the file at commit time (P1b)

The cheap signal sealed into the manifest is the one of the bytes the three inputs were **read
from**, never a `stat` of the paths taken after the rows were committed. Until round 05 `build`
and `update` stat'ed `items.json` at the END — a TOCTOU the round-05 gate reproduced (probe A): the
CLI loads the store, a save lands, the base is built from the old objects and the manifest
records the new file's mtime and size, so `search` compared equal signals and answered over
stale rows with `degraded: ["no_embeddings"]` while `status` counted `items_changed=1`.
`require_consistent` cannot see it: the counts agree. Now `load_index_inputs` reads each input
through its own handle, takes `fstat` of that handle **before** reading, and hands the signal
over with the objects (`IndexInputs`); `build`/`update` seal that one. The store's writers replace
files atomically, so the handle keeps the inode it opened and the signal describes exactly the
bytes parsed; a replacement that lands mid-load leaves the path on a newer inode, which every
later query reports as `index_behind_store`. A caller that passes no signal gets the paths
stat'ed before the first write — honest only when it wrote the files itself a moment ago.

### The topic plane follows the assignments, or `status` says it does not (H1)

`topics` stores each topic's primary and secondary members and its `stale` bit, and all three
are functions of the items' assignments — which `enrich` rewrites. Until round 04 `update`
decided the whole topic plane from the vocabulary and page fingerprints alone: an item moved
from one topic to another rewrote the item and `item_topics`, and left `topics` listing the
old members with the old `stale` bit under a manifest and a `status` that called the index
healthy (the round-04 gate reproduced it on the fixture: k02 moved to `ai-policy`,
`item_topics` said so, `topics` did not). CLAUDE.md rule 6, with the diagnostic instrument
lying on top. Now the topic row is ONE projection (`topic_row`) shared by the writer and a
comparator, the way the surface row is since G-5: `update` rewrites the rows whose members,
`stale` bit, description or synthesis are not what the store implies (`topics_refreshed`, a
row-only write — the topic surfaces and chunks depend on nothing in the membership and are
left alone; a full `topics_rebuilt` still follows a vocabulary or page change), and `status`
reads the rows back from the BASE and reports `topics_changed`, so an index whose item
fingerprints all match and whose topic plane is behind anyway is declared, not blessed.
`status` takes `vocab.yaml` and `topics.json` for that, exactly like `build` and `update`.
Neither `search` nor `get` read those columns today — both derive membership from the live
store — so the immediate consumer of the repair is the operator's instrument and Plan 04's
graph, which will.

### ONE function answers «does this manifest describe this base?» — and the manifest's nested schema is total (round 06: B1, D-1, F-1)

Six rounds closed the fail-open family one route at a time — an interrupted forced rebuild
(C-1), a missing table (C-2), counts the base did not hold (C-3), a dry run that created an
empty base (G-2), a cheap signal over one input of three (P1a) — and the round-06 gates found
the sixth and the seventh. **B1 (Codex, blind):** `Manifest.from_dict` checked the TOP-LEVEL
key set and cast what sat under it, so a manifest whose `counts` was `{}` loaded as compatible,
and the consistency check, which iterated whatever `counts` offered, compared nothing: with the
rows of `chunks` and `profiles` deleted, `status` said `incomplete: false` publishing
`chunks 0`, `search` answered zero results with `no_embeddings` and nothing else, and `update`
re-sealed the amputation as sound. **D-1 (Fable):** the root page of `items` overwritten (read
from `sqlite_master.rootpage`) made `status`, `search` and `update` a 61-line
`sqlite3.DatabaseError` traceback naming no command — the maintenance reads (`count_rows`,
`_stored_fingerprints`, `stored_topic_rows`) ran BEFORE `quick_check` and converted nothing —
while CLAUDE.md declared G-4 closed on the three; and with the root page of `chunks` damaged,
`update --dry-run` returned a normal report, because `COUNT(*)` is answered from an index.

Each earlier fix was right; each door kept its own reading of the question (CLAUDE.md rule 5).
Now there is one: `index_build.describe_base(connection, manifest, database, whole_file=…)`.
`status` reports its sentence as advice, `search` (`open_for_query`) and `update` raise it
through `require_consistent`, and a test replaces its answer with a sentinel and asserts the
three doors repeat it verbatim — a door that re-derives the question goes red (seen red with
the query door re-deriving). Three things, in order, and any `DatabaseError` of its reads IS
the answer: `PRAGMA quick_check` first, paid by `status` AND by `update` (it re-seals the
manifest and must not seal it over a torn page; `search` keeps the cheap probes by the B-1
decision — a query fails closed the moment it reaches the page, G-4); the five `COUNT(*)`
against the five planes the manifest is REQUIRED to declare; and the conversion of any
`sqlite3.DatabaseError` into the rebuild advice (`index_schema.reading_base`, wrapped around
every maintenance read). **Cost of `update`:** one `quick_check` more than before — 155–850 ms
on the 52 MB real index depending on load, the figures published under B-1.

The nested schema is **total and closed**: `counts` holds exactly the five planes
(`COUNT_PLANES`), `skipped` exactly the four causes (`SKIPPED_CAUSES`), `chunker_params`
exactly the fields of `ChunkerParams`, `store_signal` the two `items.json` entries with the four
vocab/topics entries optional (the documented pre-round-05 compatibility, now under test — F-1:
they read as zeros, the index is declared behind, one `update` clears it), every value a
non-negative integer (a string `"56"` or a JSON `true` is refused, not cast), unknown keys
refused, `embeddings` `null` or an object, `failed` a list of string maps, `built_at` an
instant. `manifest_mismatch` iterates the required planes, never `manifest.counts`; and
`write_manifest` round-trips the document through the same reader before one byte lands, so a
writer cannot seal what every door would refuse. Every malformed field is
`El manifest tiene el campo 'counts' malformado: … Reconstruye el índice con \`xbrain index
build --force\``.

---

## The lexical baseline — the number Plan 03 has to beat

Measured on the same corpus, against the 23 scorable cases of `eval/golden-set.yaml`
(`xbrain eval`). All 23 score; none is unmeasurable, and none returned an empty result set.
**Regenerated 2026-09-02 (round 07, U-6) with the depth counted in OWNERS**: until then
`evaluate` asked the index for `max(limit, max(ks))` CHUNKS and deduplicated owners afterwards,
so ten chunks dominated by one transcript held two owners and the real case F2 read
`recall@10 = 0.6667` with `--k 10` alone and `1.0` with `k=20` beside it (the `filtros` stratum
0.8333 → 1.0). Now the ranking is materialised until it holds the owners asked for, the report
publishes that depth (`limit`), and every `recall@k` at or below it is ONE number: re-derived at
depth 10, depth 20 and `--k 10` alone, the 23 per-case `recall@10` values are identical across
the three runs (0 cases differ). What the regeneration MOVED, said before the table:
`precision@10` 0.3152 → **0.3087**, because its denominator is now ten owners whenever ten exist
(it used to be however many owners ten chunks happened to hold); `MRR` 0.8179 → **0.8188** at
depth 20, because MRR is a reciprocal rank over the materialised list and is therefore bounded
by the published depth (0.8179 at depth 10, 0.8190 at depth 150 — read it with its `limit`).
Every `recall@k` in the tables below is unchanged from round 03's figures.

**What ranking this number measures, said before the number (G-6).** `xbrain eval` scores
`LexicalIndex.search` — the CHUNK plane, hits deduplicated by owner in rank order, a hit on a
topic surface counting as the owner `topic:<slug>`. That is NOT the ranking `xbrain search`
serves: the service expands each topic hit into up to `limit` supporting items (primary
first), appends profile-plane candidates after the chunk-matched ones, and can never return a
topic as a result. Re-derived 2026-09-01 on the same 23 cases at depth 10, harness against
service: the top-10 differs in **17 of 23** cases; `recall@10` read **0.8119** on the harness
and **0.6924** on the service *(both under the chunk-depth semantics retired by U-6 — the
harness figure at depth 10 owners is 0.8264; the service comparison has not been re-derived)* (the three `relevant_topics` cases cannot be hit by a service
that never returns a topic); on the 20 item-only cases the two are **0.7837** and **0.7962** —
the service is not worse, it is differently shaped — and **42.8 %** of the service's match
slots (140 of 327) are topic surfaces attached to expanded items. So the table below is the
retriever's number, the one the next retriever is compared against on the same plane; it is
not a measurement of what a `search` caller sees. **Decision recorded for Plan 03:** whether
the topic expansion should consume `max_matches_per_item` slots (today an expanded item can
carry three topic-surface matches and no match of its own), and whether the harness should
score the service's ranking beside the retriever's, are open, and the fusion sweep has to
settle both before publishing a `hybrid` number as comparable to this one.

| metric | mean (depth 20 owners, 2026-09-02) |
|---|---:|
| `recall@1` | 0.6034 |
| `recall@5` | 0.7665 |
| `recall@10` | 0.8264 |
| `recall@20` | 0.8300 |
| `precision@10` | 0.3087 *(0.3152 under chunk depth — see above)* |
| `MRR` | 0.8188 *(0.8179 at depth 10; bounded by the depth)* |

| stratum | cases | recall@1 | recall@10 | precision@10 | MRR |
|---|---:|---:|---:|---:|---:|
| `cruzado_idioma` | 5 | 0.6091 | 0.6182 | 0.1000 | 0.8000 |
| `enterrado` | 8 | 0.3958 | 0.7396 | 0.2500 | 0.6014 |
| `exacto` | 5 | 0.3667 | 0.8000 | 0.6200 | 0.6267 |
| `filtros` | 2 | 0.4167 | 1.0000 | 1.0000 | 1.0000 |
| `multimodal` | 6 | 0.7500 | 0.8333 | 0.2333 | 0.8370 |
| `resumen` | 1 | 1.0000 | 1.0000 | 0.1000 | 1.0000 |
| `semantico` | 9 | 0.4680 | 0.6675 | 0.1444 | 0.7444 |
| `topic` | 3 | 1.0000 | 1.0000 | 0.1000 | 1.0000 |
| `expansion` | — | *sin cobertura* | | | |

*(Depth 20 owners, 2026-09-02. Against round 03's table the `recall` columns are identical; the
`precision@10` and `MRR` cells moved on `cruzado_idioma`, `enterrado`, `exacto` and `multimodal`
for the two reasons stated above.)*

`recall@k` counts deduplicated **owners**; `surface_recall@k` counts **chunks**. Under the same
`k` they do not measure the same population — compare each column with its own value in the next
run, never one with the other. A cell reading *sin cobertura* is **not a zero**: no case in that
bucket could measure that metric, and a zero there would claim retrieval failed where nobody
asked.

`expansion` has no cases because it is the stratum the minimal graph is for (Plan 04). `thread`
and `user_note` have **no data in the corpus at all**, so no case can exist for them.

**By provenance (criterion 11 asks for strategy × stratum × PROVENANCE, and this axis was
computed and never published — B3, round 06).** Re-derived 2026-09-02 on the same corpus
(2,404 items, sha256 `f76341a3…`), through `evaluate` over `eval/golden-set.yaml`:

| provenance | scorable cases | recall@1 | recall@10 | precision@10 | MRR |
|---|---:|---:|---:|---:|---:|
| `construido` | 23 | 0.6034 | 0.8264 | 0.3087 | 0.8188 |
| `real` | 0 | *sin cobertura* | | | |

**Read the aggregate table above as a mean over 23 CONSTRUCTED cases**, not as a measurement of
questions Víctor has asked. The five `real` entries in the golden set (C1–C5) are archived as
`scenarios` with their reason — none has an enumerable, verified ground truth (Plan 01 §4.4,
B2) — so `real` scores nothing and is declared *sin cobertura* rather than invented; the file
itself says so (*«Las cinco D1 de este fichero son `construido`. Ningún plan las inventa como
`real`»*). Until two or three real questions are enumerated, the number Plan 03 has to beat is
the constructed one, and a `hybrid` figure compared against it inherits that population.

**`filtros` publishes `recall@10 = precision@10 = 1.000` BY CONSTRUCTION, and that pair does
not measure ranking (F-11).** Its two cases define `relevant_items` as *exactly* the population
the filter selects (2 and 3 items), so a filter that reaches the backend returns that population
and nothing else. The number is not vacuous — it can come out near 0 if the filter never reaches
the `WHERE`, which is the defect that motivated the cases (rule 2 is satisfied: there is a way
for a different answer to come out) — but read it as *the filter runs*, never as *ranking is
perfect there*. Its effect on the aggregate, said out loud (re-derived 2026-09-02 in owners):
`precision@10` 0.2429 → **0.3087** and `recall@10` 0.8099 → **0.8264**.

### The chunker sweep, regenerated in owners (Plan 02 §7, U-6)

`xbrain eval --strategy lexical --limit 10 --sweep-chunker 'target=800,1200,1600,2400
overlap=0,150,300'`, 2026-09-02, same corpus, 23 cases, **depth 10 owners** (published on the
report as `limit`):

| target | overlap | chunks | recall@10 | MRR |
|---:|---:|---:|---:|---:|
| 800 | 0 | 22,286 | **0.8264** | **0.8179** |
| 800 | 150 | 22,987 | 0.8264 | 0.7961 |
| 800 | 300 | 24,110 | 0.8264 | 0.7725 |
| 1200 | 0 | 18,036 | 0.8264 | 0.7667 |
| 1200 | 150 | 18,320 | 0.8264 | 0.7449 |
| 1200 | 300 | 18,696 | 0.8264 | 0.7449 |
| 2400 | 0 | 13,850 | 0.7955 | 0.7315 |
| 2400 | 150 | 13,912 | 0.7955 | 0.7305 |
| 2400 | 300 | 13,984 | 0.7955 | 0.7283 |
| 1600 | 0 | 15,905 | 0.7684 | 0.7710 |
| 1600 | 150 | 16,059 | 0.7684 | 0.7493 |
| 1600 | 300 | 16,245 | 0.7684 | 0.7493 |

**What this retracts.** The round-03 sweep, scored at a depth of ten CHUNKS, published 800/0
winning on `recall@10` (0.8119 against 0.8027 for the provisional 1200/150) with gains in exactly
the strata where chunking is supposed to matter (`enterrado` +2.1 pp, `semantico` +2.4 pp,
`cruzado_idioma` +0.9 pp). **In owners, that advantage does not exist**: `target=800` and
`target=1200` tie on `recall@10` at every overlap (0.8264), and the eight per-stratum `recall@10`
values of 800/0 and 1200/150 are identical cell for cell. The recall gain was a measurement of
the depth applied — a smaller target packs less of one owner into ten chunks — and not of the
chunker (rule 2, the gate's own reading). The `--limit` the CLI advertised was never passed to
the sweep, so `--limit 10` and `--limit 150` produced byte-identical reports; the same cell at
`--limit 150` now reads `recall@10 0.8264 · MRR 0.8190`.

**What survives, and the decision stands.** 800/0 still wins, by MRR — 0.8179 against 0.7449 —
and by `recall@1` (0.6034 against 0.4730, a figure that is depth-independent by construction:
one chunk is always one owner); the two larger targets lose on recall as before; overlap still
moves only MRR and never recall (this retriever cannot phrase). So `DEFAULT_CHUNKER_PARAMS`
stays `800/0` and `CHUNKER_VERSION` stays `v2`: the winner is the same, the reason is narrower
than the one first published, and Plan 03 inherits the narrower one.

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
- **And `get` names it on every surface AND every chunk it renders (H3, round 04).** The chunk
  branch of `render_get` printed type, offsets and origin only, under a bundle header naming the
  ITEM's author — so k07's quoted post, which arrives as a CHUNK when it is paged or prioritised
  by `--query`, sat under `@vgonpa` with nothing saying `@othervoice` wrote it, while the JSON was
  right. The rule is now ONE function for the search match, the surface header and the chunk
  header: the text's own author is named whenever it differs from the item's
  (`[quoted_post 0:188] origin=source · autor: @othervoice (Other Voice)`). The poster is not the
  author of what they quote.
- **`get --surface X` refuses a name the item lacks, even when another fetch failed (M-2,
  round 05).** `_select` used to answer with an EMPTY bundle whenever the item had ANY failed
  fetch — on the 62 of 2,404 real items with one, `get … --surface video_transcript` over a
  404'd article came back as `surfaces []` plus the article's failure, which reads as "the
  transcript failed". Each requested name is now compared with the surfaces the failed KIND
  would have produced: a failed one is answered with its failure, an absent one is refused
  naming what is available, and a mixed request is refused naming only the absent names
  (before, `--surface post --surface video_transcript` returned the post and dropped the
  other in silence).
- **`get --query` paginates with a cursor (A-2).** The ranking is deterministic, so the cursor is
  `q:<offset>` into it; `truncated: true` always comes with one, and the two cursor shapes
  (`q:<offset>` for a query, `<surface>:<chunk>` positional) refuse each other by name.
- **The human continuation repeats the request (H2, round 04).** A cursor is an offset into a
  sequence, and the sequence is defined by `--surface` and `--query`; the frozen bundle carries
  neither, so the line `get` printed — `xbrain get ID --cursor C` — resumed inside the DEFAULT
  selection and returned an empty page on the positional route, and was refused by the cursor
  decoder on the query route (the round-04 gate followed it literally). It now reads
  `xbrain get ID --surface … [--query '…'] --cursor C`, shell-quoted, runnable as printed; a CLI
  test follows each printed line page after page and reassembles the surface (positional) or
  the ranked chunk list (query). `--budget` is not repeated: it bounds a page and does not
  define the sequence.
- **`get` keeps the ASR/VLM producer (A-4) — and that producer is the CONFIGURED command, not
  necessarily the one that wrote the text (F7-7, round 07, declared, not fixed).**
  `transcribe_command` and `vision_command` travel in `QueryContext` from the same config
  definition the build uses, so a transcript's `producer` is the configured transcriber in `get`
  exactly as in the index. But the store's `x_video` source records no transcriber: measured on
  the real corpus, after changing `[transcribe].command` from `xbrain-transcribe-auto` to
  `whisper-large-v3`, `get` served `producer: whisper-large-v3` for a transcript parakeet wrote,
  with text, `surface_fingerprint` and `item_fingerprint` identical — a provenance claim the
  store cannot back. The fix is to stamp the producer on the source when `digest-video` attaches
  the transcript (the `caption_contract` pattern) and is a store change outside Plan 02; it is
  recorded as an open issue, and `KnowledgeSurface.producer`'s docstring says which two surface
  types carry this reading. The fingerprint deliberately does not hash it (a binary rename must
  not rewrite every ASR item), which is the right decision over the wrong data.
- **The spec's `matched_surface` is the contract's `surface_type` (M-6).** Spec §7.2 names the
  field `matched_surface` in its illustrative JSON and says the example *defines semantics, not
  final property names*; Plan 01 froze `SearchMatch.surface_type` at `schema_version: "1"` and
  Plan 02 §0 forbids renaming it. Read one as the other; do not look for `matched_surface`.
- **Every fragment carries the locator of the source whose text it delivers, built by ONE
  function for `search` and `get` (B2, round 06).** A `KnowledgeChunk` had no locator: it
  carried the OWNER's URL and offsets into a surface that, when `get` paginated or prioritised
  by `--query`, was not in the bundle at all (`surfaces: []`, the only shape a chunk exists in).
  The blind gate's reproduction on k03: `chunk.url` the poster's tweet, the essay's locator
  nowhere. `KnowledgeChunk.locator` is now REQUIRED — the surface's locator narrowed to the
  chunk's range by `chunking.fragment_locator`, the SAME function `search._match` applies —
  and two tests bind the services: the match and the chunk `get` serves for one `chunk_id`
  carry one locator and one attribution (the quoted author on k07), and a sentinel swapped
  into `fragment_locator` must reach every match and every chunk. It is a field spec §3.7.2
  required of the chunk from the start — its absence was a defect of the contract, not a
  property of it — **and it bumped `EvidenceBundle.schema_version` to `"2"` (U-1, round 07)**:
  round 06 kept `"1"` calling the key "additive", but every contract model is `extra="forbid"`,
  so the version-1 Pydantic consumer refuses a bundle carrying `locator` (`Extra inputs are not
  permitted`, measured against the exact `origin/develop` model) and the new consumer refuses a
  version-1 document (`Field required`) — two producers under one number that do not
  interoperate. The policy is written once in `contracts.py`: a key added to a frozen shape
  bumps the envelope that transports it, and the refusal then names the version. `SearchResponse`
  stays at `"1"` (`SearchMatch` always carried its locator); `xbrain knowledge inspect` reads the
  number off the contract (`EVIDENCE_SCHEMA_VERSION`) instead of stamping it by hand. Nothing at
  `"1"` is persisted anywhere, so there is no migration. `chunk.url` keeps its one meaning
  (where a human opens the owner).
- **A hit whose surface row cannot be resolved is excluded and counted (B-k, round 06).**
  `_match` used to FABRICATE a locator for it (`content_source`, the item's URL, no source
  index) — the one thing worse than a missing locator. It now counts in
  `corrupt_chunks_excluded` like a fingerprint that does not recompute: invariant 1 (every
  chunk resolves to a surface) has the shape of invariant 6.
- **The fingerprint covers EVERYTHING the index serves about a chunk, not the text alone (U-5,
  round 07 — gate Codex F4, reproduced on the real corpus).** `chunk_fingerprint` hashed
  `(surface_id, chunk_index, text)`, and the row served beside the text — `origin`,
  `trust_class`, `derived`, `surface_type`, the owner, the position, the surface row's
  attribution and locator — was bound to nothing. A quoted post by Josh Bryant rewritten in the
  base as the poster's own `summary`, `origin: llm`, `trust_class: llm_synthesis`, attributed to
  `@vgonpa` with a syntactically valid URL to the poster's page, was served by `search` with
  `corrupt_chunks_excluded: 0` while `status` called the index healthy: the attribution rule this
  repo paid for in blood, defeated by a valid-looking value. Now ONE projection —
  `chunking.chunk_evidence`, the second half of seam (b) — is what `_chunk` hashes at emission
  and what `verify_fingerprints` rebuilds from the served row (the locator narrowed through
  `fragment_locator`, the same function that builds the locator a match carries), so any arm
  rewritten is excluded and counted like a text that does not match its hash. Eleven per-field
  mutations pin it; the counter is asserted EXACT (`== 2` on two victims — F7-6: `int(bool(…))`
  had kept 495 tests green against `>= 1`). The physical schema is **3** for it: a v2 base's
  fingerprints were computed over the text alone, so the door refuses it by name instead of
  answering «22,286 chunks excluded». One `xbrain index build --force` on upgrade.
- **An author's name or handle is ONE printable line wherever a human sees it (D-3, round 06).**
  M-3 fenced the bodies; the fields beside them were left as they arrived, and a newline in
  `author.name` printed at column 0 a line byte-identical to a renderer header and another to
  a fence line — the G-7 forge through the header itself — while an `ESC[2K` in a handle
  reached the TTY through the result line. `render._author_label` is the only way an author
  is printed (bundle header, result line, `autor:` label), collapsing each field through
  `_one_line`; a sentinel test pins the three placements. Population today 0 of 2,404; the
  population is what X accepts.
- **`get` exposes the item's CURRENT verdicts, and a stale one is dropped (S-7, round 06, test
  only).** The conduct existed; no test guarded it — `verification={}` in `get` left 443 tests
  green — and what would vanish in silence is every FAIL a consumer of `get` would see. Now
  pinned: a FAIL whose contract fingerprint matches the current summary travels with its stamp;
  the same FAIL over a regenerated summary is `{}`.
- **The human view is NOT a surface for agents — use `--json` (G-7).** The JSON carries `origin`
  and `trust_class` beside every text, so nothing in a body can be mistaken for the frame. The
  human view of `get` is text all the way down, so since round 03 it FENCES every body line with
  `│ ` and collapses titles to one line: a quoted post containing a line identical to the
  renderer's own `[user_note] origin=user trust=user_text` header (reproduced by the round-04
  gate) stays visibly inside the body. **And since round 05 the body cannot erase that fence
  (M-3):** an `ESC[2K` / `ESC[1A` stored in a tweet reached the terminal intact under a
  pseudo-TTY (BEL even through a pipe) and could wipe the header or the fence above it, so every
  C0 control except TAB and LF, plus DEL and the C1 range, is removed from every body line,
  title, summary and excerpt — dropped, not escaped; the text is still shown whole. That is a
  courtesy to the reader, not a security boundary; an agent that parses the human view instead
  of the JSON is parsing untrusted text.

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
