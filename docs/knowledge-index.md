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
| upgraded xbrain and `index update` refuses | the emitter or the chunker moved | `xbrain index build --force` |

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
| `index update`, 0 items changed | 0.15 s |
| `index update`, 1 item changed | 0.16 s — 3 chunks out, 3 in |
| `index update`, 100 items changed | 0.26 s — 645 chunks out, 645 in |
| `data/index/knowledge.db` | **52.4 MB** (manifest 1 KB) |
| chunks | **22,286** — 10,160 surfaces, 2,404 profiles |
| chunks per item | median 3, mean 9.0, max 712 |
| `search` latency over the 23 golden-set cases | p50 **26 ms**, p95 43 ms, max 66 ms |
| omitted, by cause | 63 failed fetches · 108 silent videos · 14 decorative images · 0 empty |

`search` latency is index-open + score + fingerprint-verify + group + hydrate, with the store
**already loaded**. Add the 0.22 s store load for a cold CLI invocation: `search` hydrates
verification from the live store (a verdict copied into the index could never be invalidated),
so the store is not optional.

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

### Interruption

`Ctrl-C` mid-build rolls the transaction back and writes **no manifest**. Measured: after an
interrupt, `chunks` holds 0 rows and no manifest exists — and an index with no manifest is
**refused** by every query rather than answered partially, so an interruption can never leave a
small index that looks valid. `xbrain index status` reports it as incomplete and names
`xbrain index build`.

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

---

## What the lexical baseline cannot do

Written down because a limit nobody records gets quietly attributed to the corpus instead of to
the retriever.

- **No stemming.** FTS5 has no multilingual stemmer and the English one would wreck the Spanish
  half of a bilingual corpus, so `agent` does not match `agents`. This is what a vector layer is
  for.
- **IDF is relative to THIS corpus.** A word that reads as a function word can still be rare to
  the index and therefore undiscounted. Measured: `el` sits in 5,748 of the chunks (31 %) on the
  real corpus and in 1 of 43 in the test fixture — bm25 behaving correctly on each corpus it was
  given, and ranking `el` very differently in the two.
- **The terms are ORed, not phrased.** Every term is quoted as an FTS5 string and joined with
  `OR`. A conjunction returned zero rows for 18 of 21 cases (measured), so the disjunction is
  what makes bm25 able to rank at all — but it also means no phrase query, and therefore no
  benefit from window overlap.
- **`KnowledgeSurface.language` is `None` on almost everything.** The only language the store
  records is `ContentSourceSuccess.language`, populated by transcriptions. It is never guessed,
  and there is **no language filter**.
- **No embeddings.** Every response declares `degraded: ["no_embeddings"]` — so a consumer
  cannot read a lexical answer as a hybrid one.

---

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
