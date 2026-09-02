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
| upgraded xbrain and `index update` refuses | the emitter, the chunker or the SCHEMA moved (schema **2** since round 02) | `xbrain index build --force` |
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
while a manifest exists.

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

---

## The lexical baseline — the number Plan 03 has to beat

Measured on the same corpus, against the 23 scorable cases of `eval/golden-set.yaml`
(`xbrain eval`). All 23 score; none is unmeasurable, and none returned an empty result set.

**What ranking this number measures, said before the number (G-6).** `xbrain eval` scores
`LexicalIndex.search` — the CHUNK plane, hits deduplicated by owner in rank order, a hit on a
topic surface counting as the owner `topic:<slug>`. That is NOT the ranking `xbrain search`
serves: the service expands each topic hit into up to `limit` supporting items (primary
first), appends profile-plane candidates after the chunk-matched ones, and can never return a
topic as a result. Re-derived 2026-09-01 on the same 23 cases at depth 10, harness against
service: the top-10 differs in **17 of 23** cases; `recall@10` reads **0.8119** on the harness
and **0.6924** on the service (the three `relevant_topics` cases cannot be hit by a service
that never returns a topic); on the 20 item-only cases the two are **0.7837** and **0.7962** —
the service is not worse, it is differently shaped — and **42.8 %** of the service's match
slots (140 of 327) are topic surfaces attached to expanded items. So the table below is the
retriever's number, the one the next retriever is compared against on the same plane; it is
not a measurement of what a `search` caller sees. **Decision recorded for Plan 03:** whether
the topic expansion should consume `max_matches_per_item` slots (today an expanded item can
carry three topic-surface matches and no match of its own), and whether the harness should
score the service's ranking beside the retriever's, are open, and the fusion sweep has to
settle both before publishing a `hybrid` number as comparable to this one.

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
- **`get` keeps the ASR/VLM producer (A-4).** `transcribe_command` and `vision_command` travel in
  `QueryContext` from the same config definition the build uses, so a transcript's `producer` is
  the configured transcriber in `get` exactly as in the index.
- **The spec's `matched_surface` is the contract's `surface_type` (M-6).** Spec §7.2 names the
  field `matched_surface` in its illustrative JSON and says the example *defines semantics, not
  final property names*; Plan 01 froze `SearchMatch.surface_type` at `schema_version: "1"` and
  Plan 02 §0 forbids renaming it. Read one as the other; do not look for `matched_surface`.
- **The human view is NOT a surface for agents — use `--json` (G-7).** The JSON carries `origin`
  and `trust_class` beside every text, so nothing in a body can be mistaken for the frame. The
  human view of `get` is text all the way down, so since round 03 it FENCES every body line with
  `│ ` and collapses titles to one line: a quoted post containing a line identical to the
  renderer's own `[user_note] origin=user trust=user_text` header (reproduced by the round-04
  gate) stays visibly inside the body. That is a courtesy to the reader, not a security boundary;
  an agent that parses the human view instead of the JSON is parsing untrusted text.

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
