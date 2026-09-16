# The knowledge index — what it does, how it works, how to use it

This is the page to read without opening the code. It goes from building the
index to connecting an agent; the shape underneath — planes, manifest,
invalidation — is in [ARCHITECTURE.md](../ARCHITECTURE.md#the-knowledge-layer).

## In plain words

Your saved posts live in `data/items.json`, with everything xbrain gathered around
each one: the linked article, the thread, the quoted post, the video transcript,
the image descriptions, and the summary and topics it generated. That store is the
truth. The **knowledge index** is a search engine built next to it, so that you —
or an agent — can ask a question and get back **the exact passages** that answer
it, each labelled with where it came from.

| Piece | What it does | How it works |
|---|---|---|
| **The index** (`index build`) | Makes the corpus searchable | Cuts every text into passages of about 800 characters ("chunks"), keeps a fingerprint of each, and stores them in a SQLite file under `data/index/` with a full-text index (FTS5). A manifest records what was indexed and with which settings. |
| **`search`** | Finds the posts that match a question | Scores passages by the words they share with the question, rare words counting more (bm25), then groups them by post, at most three passages per post. |
| **`get`** | Hands over one post's evidence in full | Reads the live store, not the index, so it is never out of date. Long texts come in pages. |
| **Provenance labels** | Tells you who wrote each passage | Every passage says whether it was captured from the source, heard by speech recognition, seen by a vision model, or written by xbrain itself (a summary). |
| **`graph-expand`** | Shows which topics and posts sit around a post | A small graph of posts and topics, built from the topics xbrain assigned. Two topics are linked when enough posts carry both. |
| **Vector search** (opt-in) | Finds passages by meaning, not only by words | An external program you configure turns passages into vectors; `hybrid` merges both rankings. Off by default, and not shown to beat word search yet. |
| **`mcp-serve`** | Lets an agent use all of the above | A small server that offers `search`, `get` and `graph_expand` to Claude Code or Claude Desktop, with the same answers the CLI gives. |
| **`eval`** | Measures whether retrieval finds the right posts | Runs the questions in `eval/golden-set.yaml`, whose right answers are known, and reports recall per kind of question. |

Nothing on this path calls a generative model, nothing writes your store, and the
index can be deleted and rebuilt at any time. The one outside program it may
start is the embedder you configure for vector search — off by default.

### End to end

```bash
uv run xbrain index build                           # 1. build it (once)
uv run xbrain search "harness engineering"          # 2. ask
uv run xbrain get 2063609922667815064 --surface external_article   # 3. read the source
uv run xbrain graph-expand --item 2063609922667815064              # 4. look around
uv run xbrain index update                          # 5. after enrich/topics/fetch: catch up
claude mcp add xbrain -- uv run --directory "$PWD" --extra mcp xbrain mcp-serve   # 6. give it to an agent (run from the checkout)
```

Step 2 prints, for each post, the passages that matched, and a `verifica con:` line
naming the source to read when a match landed on text xbrain wrote; step 3 is how
you read it (a bare `get <id>` lists the surfaces an item has). Step 6 is
[docs/mcp.md](mcp.md), and what the agent should do with the answers is
[docs/knowledge-for-agents.md](knowledge-for-agents.md).

Unless a section says otherwise, the figures below were measured on one corpus on
2026-09-12: **2,474 items, 45 topics** (`store-2474` in
[Measured versions](#measured-versions)), macOS/APFS, a warm page cache, **before
the graph plane existed**. Your numbers will differ; the commands that produce them
are printed beside each one, so read these as a shape rather than as a promise.

## The seven commands

```bash
uv run xbrain index build     # build data/index/ from scratch and seal it
uv run xbrain index update    # touch only what changed since the last build
uv run xbrain index status    # what the index holds, and how far behind it is
uv run xbrain search "…"      # ranked items with citable fragments
uv run xbrain get <item-id>   # one item's evidence, from the STORE (never the index)
uv run xbrain graph-expand --item <item-id>   # topics and items around one item
uv run xbrain mcp-serve       # the same three queries, for an agent (stdio)
```

`build`, `update` and `status` WRITE only `data/index/`, and none of them takes a
snapshot, because there is nothing of yours to lose: **`data/index/` is derived
and reconstructible**, it is never versioned, and deleting it costs one
`build`. They do READ your store: every command above loads `items.json`,
`vocab.yaml` and `topics.json` through the same loader, which is why the `build`
below costs 9.5 s wall for 3.1 s of work. `status` goes further and WALKS what
it read — a fingerprint per item, plus the topic records the three inputs imply
— and that walk is what lets it answer *how many* items changed rather than
merely *something moved*.

`search` and `graph-expand` open the database `mode=ro`, so a stray write is an
error rather than a silent repair. `get` never opens `data/index/`: a bare `get` reads the live
store and opens no database at all, and `get --query` opens one that exists
nowhere on disk — a scratch `sqlite3(":memory:")` holding only that item's own
chunks, built to rank them with the same scorer `search` uses, and closed before
the call returns. Neither command calls an LLM and neither touches the network.

## When to rebuild, and when to update

Indexing is **manual by design**. Nothing runs it for you, so the failure that
actually happens is *you ran `enrich` and did not reindex* — which is why the
index declares its own staleness instead of hoping you remember.

| You did this | Run this |
|---|---|
| `extract`, `fetch`, `enrich`, `topics`, `vocab`, `digest-video`, `describe` | `index update` |
| upgraded xbrain and `status` says the manifest is incompatible | `index build --force` |
| deleted `data/index/` | `index build` |
| edited `[index].graph_*` in `config.toml` | `index update` — it rewrites the graph plane |
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

**The graph plane came later and is small.** Re-measured on `store-2495`, the
graph sweep's store ([Measured versions](#measured-versions)), with an index built
on 2026-09-16 at the default thresholds: `knowledge.db` is **53.6 MiB**, of which the
`graph_edges` table and its three indexes take **1.6 MiB** (`dbstat`), for 5,783
edges. The 52 MiB above predates that plane and that corpus. Timings were not
re-taken: the machine was swapping hard (load average above 70) and a wall clock
measured there describes the machine, not xbrain.

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
   · [video_transcript] origin=asr trust=machine_extracted · via lexical
     do kind of position dependence and incorporate information from other posi…
   · [video_transcript] origin=asr trust=machine_extracted · via lexical
     hyperparameter. Um I always found this to be very strange when sort of teac…
   → verifica con: xbrain get 2051242195298968041 --surface video_transcript

[extracto: se omite el resultado 2; una línea de truncado, cuando la hay, se
 imprime bajo la cabecera, encima del resultado 1, no aquí]
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
warning). The JSON below carries fields it never prints — `schema_version`,
`manifest_version`, `built_at`, and per match the `chunk_id`, `title`, `score`,
`lexical_rank`, `vector_rank` and `locator`. The echoed `filters` are **not**
among them: a truncated result spells every one back as a flag of the
continuation command it prints (`--topic`, `--kind`, `--origin`, …), because a
cursor is an offset into the ranking the query *and* the filters define, and a
continuation that dropped one would resume inside a different ranking. Read the
human view to judge a result; parse `--json` to consume one:

```bash
uv run xbrain search "transformer attention" --limit 1 --json
# → {"schema_version": "2", "query": …, "strategy": "lexical",
#    "index": {"manifest_version": "5", "built_at": …,
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
is a position in the ranking that the query, **its strategy and its filters**
define. The printed continuation repeats every filter and the page size beside
`--cursor`, but **not `--strategy`**. After a `--strategy hybrid` page, add it
yourself: without it the next page comes from the `lexical` ranking, which
repeats some results and never serves others that `hybrid` ranked. That
page's heading names `lexical`, so the switch is visible. The printed line
should carry the flag (backlog).

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

**`xbrain eval` pushes the same eight.** The evaluation harness used to walk the
corpus its own way, writing chunks and no metadata, so it could push only
`has_surfaces` and `origins` — and even that pair was a fabricated zero, because
its `surfaces` table held no rows for the `EXISTS` to match. It now builds
through the same writer `xbrain index build` drives, so the measured retriever is
the one `search` queries and all eight filters reach `WHERE`. The set is derived
from `SearchFilters`, not written out a second time, so a ninth filter cannot
leave the harness silently declaring eight.

**The rule that gap produced still stands, and it has a live instance again.** A
golden-set case whose filters a strategy cannot push is reported **unmeasured**,
never `0.0`. A zero from a filter nobody applied reads as "retrieval failed at
filtering" when the truth is that the instrument was not there. `lexical` scores
both `filtros` cases. The vector plane has **no filter columns**, so under
`--strategy vector` and `--strategy hybrid` those two cases come back unmeasured —
and `search`, with the plane and the command in place, answers a filtered
`vector`/`hybrid` request lexically, declaring it. Without them, `vector` is an
error, filters or not ([below](#when-the-vector-channel-cannot-run)).

## Measuring it: the golden set

`eval/golden-set.yaml` is a versioned list of questions whose right answers — item
ids enumerated by hand from the real corpus — are known. It is tracked in Git on
purpose (questions and ids, no corpus bodies), and CI validates its structure
without a store. Each case belongs to one or more **strata** (`exacto`,
`semantico`, `cruzado_idioma`, `filtros`, `enterrado`, `multimodal`, `expansion`,
…), because one global number hides the question types that fail.

```bash
uv run xbrain eval                                  # lexical, report in data/eval-report.{json,md}
uv run xbrain eval --strategy hybrid --embeddings-model <model>   # the bake-off
uv run xbrain eval --min-recall 0.5                 # exit non-zero if a bucket falls below
```

The report gives recall@k, MRR and nDCG per strategy × stratum × provenance. A
case the strategy cannot measure — a filter the vector plane cannot apply, a
stratum with no enumerated answers — is printed as **unmeasured**, never as `0.0`.
The report goes under `data/`, which is not tracked, because it quotes the corpus.

The lexical baseline to beat, as shipped (chunks of 800 characters, no overlap),
is `recall@10` **0.7391** · MRR **0.7357**, measured on `store-2474` (22,933
chunks) against `golden@d1423c8`, the tracked golden set, re-derived 2026-09-16
with `xbrain eval --sweep-chunker "target=800 overlap=0"`. Both versions are
spelled out in [Measured versions](#measured-versions). Older documents quote
**0.7395**: the same store and the same chunks against a golden set from before
`d1423c8`, when case U3 listed 22 relevant items instead of 24. Two versions
fit that description and both score 0.7395: `golden@a88c753` and
`golden@427fea9`, which differs from it only in the `expansion` stratum (its
labels and their notes). The MRR does not move. The recall moves with the
golden set as well as with the corpus, so quote it with both versions.

The two negative results are summarised in their sections below, and each one
states its own population. The embeddings bake-off was measured on `store-2474`
against `golden@427fea9` (commit `547a860`): its lexical row reads 0.7395, at a
depth of 20 rather than 10. The graph sweep used `store-2495`, 18 cases and
items served by `search` as its unit, and it says itself that its figures are not
comparable with either number.

## Measured versions

Every figure on this page, in [CLAUDE.md](../CLAUDE.md), in
[ARCHITECTURE.md](../ARCHITECTURE.md), in the [bake-off](embeddings-bakeoff.md)
and in the [graph sweep](graph-threshold-sweep.md) was measured on some version
of the golden set and some version of the store, and this section is the only
place those versions are written out: the other documents name a version and
link here instead of repeating its hash.

The two inputs are not checkable in the same places. The golden set is tracked
in Git, so its check runs on any clone that has the history. The store (`data/`)
is not, so its check only runs on the machine holding the named snapshot, from
the directory that contains `data/`. When no snapshot kept a store, its figures
cannot be re-derived anywhere, and the table keeps the row and says so.

Every check below was run on 2026-09-16 at `b7aa992`.

| Version | What it is | Where it is | sha256 | Check | State |
|---|---|---|---|---|---|
| `golden@427fea9` | `eval/golden-set.yaml` from `427fea9` up to `a88c753^`: U3 lists 22 relevant items, no `expansion` labels | Git, every commit in that range, the bake-off's `547a860` among them | `ed6dd760…` | `git show 547a860:eval/golden-set.yaml \| shasum -a 256` | reproduced |
| `golden@a88c753` | the same file from `a88c753` up to `d1423c8^`: `expansion` labels and their notes added, U3 still at 22 | Git | `ed590920…` | `git show a88c753:eval/golden-set.yaml \| shasum -a 256` | reproduced |
| `golden@d1423c8` | the same file from `d1423c8` on, U3 at 24. The tracked one today | Git, `d1423c8` to `b7aa992` | `bf9aad8f…` | `git show d1423c8:eval/golden-set.yaml \| shasum -a 256` | reproduced |
| `store-2404` | `data/items.json`, 2,404 items, the live store of 2026-09-01 to 09-03, quoted by sha256 | nowhere | `f76341a3…` | none | **not reproducible** |
| `store-2404` | `data/items.json`, 2,404 items, the live store of 2026-08-31, quoted by **md5** | nowhere | md5 `5aaf62f4…` | none | **not reproducible** |
| `store-2474` | `items.json`, 2,474 items. Live from 2026-09-11 17:34 to 2026-09-14 09:43 (local time; the snapshot keeps the file's mtime) | `data/snapshots/2026-09-14T07-43-40-952Z-pre-full-pipeline-20260914/` | `4fed54a0…` | in that directory: `shasum -a 256 items.json vocab.yaml topics.json` | reproduced |
| `store-2474` | its `topics.json` | the same snapshot, and every one up to `…08-47-12-722Z-pre-topics-resynth` | `7a40f4f1…` | the same command | reproduced |
| `store-2474`, `store-2495` | `vocab.yaml`, the same file in both stores and in `data/` today | every snapshot | `e73fbede…` | either store's command | reproduced |
| `store-2495` | `items.json`, 2,495 items. Live from 2026-09-14 10:46 to 2026-09-16 10:55 | `data/snapshots/2026-09-16T08-55-19-225Z-pre-full-pipeline-20260916/` (also in `…08-47-12-722Z-pre-topics-resynth`, beside the older `topics.json`) | `2773310f…` | in that directory: `shasum -a 256 items.json vocab.yaml topics.json` | reproduced |
| `store-2495` | its `topics.json`, re-synthesised on 2026-09-14. Also `data/topics.json` today | the same snapshot | `d2f46a72…` | the same command | reproduced |
| `index-2495` | `store_fingerprint` of `store-2495`, the value the sweep's manifest sealed | computed, not stored | `93f994d4…` | the fingerprint block below | reproduced |
| `index-2495` | `vocab_fingerprint` of the same inputs. Also in today's `data/index/manifest.json` | computed | `55da1032…` | the fingerprint block below | reproduced |
| `index-2495` | `topics_fingerprint` of the same inputs. Also in today's manifest | computed | `9af12df5…` | the fingerprint block below | reproduced |

**The full digests, one command each.** For the golden sets, from any clone:

```bash
for rev in 547a860 a88c753 d1423c8; do
  printf '%s  %s\n' "$(git show "${rev}:eval/golden-set.yaml" | shasum -a 256 | cut -d' ' -f1)" "$rev"
done
# ed6dd7600f946fba0d367eaa7bd020092820dc1684da79c74b820a91fe3fa319  547a860   golden@427fea9
# ed590920af894eb8014ce3fcf35707413352ae4d561213fad62dd293b72010d2  a88c753   golden@a88c753
# bf9aad8f6d73af6d6a1dcd83c103ebbf012955c5c590b10b8c375712258d4df4  d1423c8   golden@d1423c8
```

Keep the braces in `${rev}`. zsh reads `$rev:eval/…` as `$rev` plus the `:e`
modifier, so `git show` gets a broken path and fails, and the pipe prints
`e3b0c442…`, the hash of an empty string.

For the two stores, run this from the directory that contains `data/`; it prints
`OK` or `FAILED` for each file:

```bash
shasum -a 256 -c <<'EOF'
4fed54a0bee5e747fffa7efcace8502733defdc445d0e9a88194a9c293312cde  data/snapshots/2026-09-14T07-43-40-952Z-pre-full-pipeline-20260914/items.json
e73fbedecdcaf6a8cf9609fb72487481a1917b2611bf96c4529097dcf2cca595  data/snapshots/2026-09-14T07-43-40-952Z-pre-full-pipeline-20260914/vocab.yaml
7a40f4f12d285d44cb4205c0c85ce5c79448872c0fc3c7f92894b5d7ca28363e  data/snapshots/2026-09-14T07-43-40-952Z-pre-full-pipeline-20260914/topics.json
2773310f60bb0f453a90d047244ace091064bcbe0cf859962b6e1d1a5c051230  data/snapshots/2026-09-16T08-55-19-225Z-pre-full-pipeline-20260916/items.json
e73fbedecdcaf6a8cf9609fb72487481a1917b2611bf96c4529097dcf2cca595  data/snapshots/2026-09-16T08-55-19-225Z-pre-full-pipeline-20260916/vocab.yaml
d2f46a72c88413b6dbf62519f7b17fec79bb4aaa7493555c50b64cd3977303ee  data/snapshots/2026-09-16T08-55-19-225Z-pre-full-pipeline-20260916/topics.json
EOF
```

The three index fingerprints need Python. A fingerprint hashes what the index
emits from the store rather than the bytes of a file, so it depends on the code
too; at `b7aa992` this gives the same three values the sweep sealed at
`d1423c8`, and a later commit that bumps a projection version will give others.
Run it from the directory that contains `data/`, with `CHECKOUT` pointing at a
clone checked out at `b7aa992`:

```bash
uv run --project "$CHECKOUT" python - <<'PY'
from pathlib import Path
from xbrain.knowledge.index_build import (
    load_index_inputs, store_fingerprint, topics_fingerprint, vocab_fingerprint,
)
d = Path("data/snapshots/2026-09-16T08-55-19-225Z-pre-full-pipeline-20260916")
i = load_index_inputs(d / "items.json", d / "vocab.yaml", d / "topics.json")
print(store_fingerprint(i.store))       # 93f994d4ef1b6aa2ec66f2738b2c8898bb305b928c006ef8680a01c936ae8d76
print(vocab_fingerprint(i.vocab))       # 55da10325820ed0b5c026ac93d526b391c3dcb85c14474e9ba10f1ca979a1a28
print(topics_fingerprint(i.topic_pages))  # 9af12df51ff96edbacfadf025b0c82dc2bd05df4ea174588caf173b0b740d9c5
PY
```

**What was measured on each.** In the last column, *re-run* means the figure
came out identical on 2026-09-16 at `b7aa992`, and *recorded* means nobody re-ran
it, for the reason given.

| Population | Figures | Quoted in | 2026-09-16 |
|---|---|---|---|
| `store-2474` × `golden@d1423c8` | lexical `800/0`: `recall@10` 0.7391 · MRR 0.7357, depth 10, 22,933 chunks. `800/150`: 0.7391 · 0.7360 | this page, CLAUDE.md | re-run |
| `store-2474` × `golden@a88c753` | lexical `800/0`: 0.7395 · 0.7357 | this page, CLAUDE.md | re-run |
| `store-2474` × `golden@427fea9` | lexical `800/0`: 0.7395 · 0.7357. `800/150`: 0.7395 · 0.7360. The whole [bake-off](embeddings-bakeoff.md) | bake-off, this page, CLAUDE.md | lexical re-run, including the bake-off's lexical row at depth 20 (0.7395, uncut MRR 0.7366). Its per-stratum MRR is now printed as `mrr@10`, as the bake-off's §9 warns. Recorded: every MiniLM figure (it needs the embedder and weights the bake-off deleted) |
| `store-2474` alone | 10,570 surfaces · 22,933 chunks · 2,474 profiles. The costs in [What it costs](#what-it-costs), taken 2026-09-12, while this was the live store. `agente` and `agentes` share 0 of 10, and `transformer` and `transformers` share 7. `el` in 28.0 % of chunks (2026-09-13) | this page, CLAUDE.md | 22,933 chunks re-run. Recorded: wall-clock timings (they describe the machine) |
| `store-2495` × `golden@d1423c8` | the [graph sweep](graph-threshold-sweep.md): 16 cells, base 0.6296 on 18 cases, 33 `expansion` pairs, 0 of them lifted | graph sweep, this page | re-run with the sweep's §6 command, at `b7aa992` rather than the `d1423c8` it checks out: the 16 rows of its §3 table came out byte-identical, and so did the base, the `expansion` line and the verdict |
| `store-2495` × `golden@a88c753` | the sweep's first signature (base 0.6301, 31 pairs), and its two `expansion` classification runs (140 s and 144 s) with their 58 identical pairs | graph sweep | recorded |
| `store-2495` / `index-2495` | the graph plane at the default thresholds: 5,783 edges (2,495 `HAS_PRIMARY_TOPIC`, 3,128 `HAS_TOPIC`, 160 `CO_OCCURS_WITH` among 41 topics). `knowledge.db` 53.6 MiB, 1.6 MiB of it `graph_edges` with its three indexes (0.64 MiB the table alone) | this page, CLAUDE.md | recorded |
| `store-2404` | with the **sha256**, on 2026-09-01: `el` in 6,070 of 22,286 chunks (27.2 %), chunker v2 `800/0`. With the **md5**, on 2026-08-31: 18,319 chunks (9,294 atomic + 9,025 splittable), and `el` in 5,748 of 18,319 (31.4 %), chunker v1 | this page, CLAUDE.md, ARCHITECTURE.md, and comments in `src/` and `tests/` | cannot be re-run |

The live store today (2,519 items) carries none of these figures.

**Why `store-2404` cannot be reproduced.** Both digests name a 2,404-item store
that `data/` no longer holds. Snapshots begin on 2026-09-11, when the store
already had 2,474 items; the older backups next to it hold 2,130, the live store
2,519, and none of them hashes to either value. A search of the whole disk for
`items.json` copies turned up nothing else. Whether the md5 and the sha256 even
name the same bytes is beyond checking too, since there is no file left to hash
with both. So the figures stay where they are, dated and read as history: none
can be re-derived, and none may be re-stamped to a later store (CLAUDE.md,
rule 6).

**Keeping a measured store checkable.** These are ordinary `xbrain` snapshots,
and `xbrain snapshot prune` removes the oldest. Run with its default
`--keep-last 10` on today's 15, it would delete five, and the only copy of
`store-2474` is among them. Copy any snapshot a figure was measured on out of
`data/snapshots/` before you prune.

## Known limits of the lexical baseline

These are declared, not discovered later. The optional vector plane is designed
for the stemming and meaning gaps; whether it closes them on this corpus has not
been shown — the only candidate measured did not.

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
*here*, not in the language. `el` appears in roughly 27–28 % of real chunks (6,070
of 22,286 on `store-2404`, which [no copy kept](#measured-versions); 28.0 % on
`store-2474`, per the umbrella audit of 2026-09-13), so it is discounted heavily; a word that reads
like a function word to you may be rare to the index and go undiscounted. This is
also why a fixture-sized index ranks differently from the real one.

**Lexical means lexical.** It retrieves proper nouns, figures, handles and other
literal terms, not conceptual similarity. Every response over an index built
without `--embeddings` says so: `degraded: ["no_embeddings"]`, read off the
manifest's `embeddings` block rather than hard-coded, so an index built with
`--embeddings` stops declaring it on its own.

**There is no phrase search.** The query is split into terms, each quoted so that
punctuation stays literal (`@simonw`, `11.37%`), and the terms are joined with
`OR`: `"harness engineering"`, quotes included, asks for chunks with *harness* or
*engineering*, and bm25 usually ranks those with both higher. It does not require the two
words to be adjacent. The human view's `frases exactas` wording overstates this.

**A strategy you name is run or declared, never faked.** `hybrid` without a working
vector channel answers `lexical` and names the cause; `vector` without one is an
**error**, because you asked for vectors by name. `hybrid_graph` answers `lexical`
labelled `hybrid_graph_not_implemented` from `search`, from MCP and from
`xbrain eval --strategy hybrid_graph` — **by default**: the graph re-ranking
exists, behind a switch only the Python API turns on
(`search(..., graph_enabled=True)`). The one command that reaches it is the
threshold sweep, `xbrain eval --strategy hybrid_graph --sweep-graph …`; without
`--sweep-graph` the report says `strategy: lexical`. Switching it on also opens
the vector channel exactly as `hybrid` does
([below](#the-graph--opt-in-and-measured-negative)). A *typo*
(`--strategy vectro`) is refused: answering it with lexical results would turn a
mistake into a measurement.

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

## The vector plane and `hybrid` — opt-in, and not the default

Everything above is the lexical index, and it is what `search` runs unless you ask
for something else. Plan 03 adds a second, **optional** plane — one embedding per
distinct chunk text — and two strategies that read it: `vector` (the geometry alone)
and `hybrid` (bm25 and the geometry, fused). **`lexical` stays the default and
`hybrid` has not been promoted**: the bake-off that would justify it is incomplete
([below](#the-bake-off-incomplete)). Nothing in this section is needed to use `search`.

### What you need

**1 · `numpy`, through the `[embeddings]` extra.** It is deliberately not a runtime
dependency. From a checkout of this repo the extra is installed with
`uv sync --extra embeddings` — CI runs `uv sync --extra dev --extra embeddings --locked`.
`uv sync` removes packages its flags did not ask for, so a later plain `uv sync`
uninstalls `numpy` again. Without it, `import xbrain` and lexical `search` keep
working, and a query that needs the matrix fails naming
`uv pip install 'xbrain[embeddings]'` instead of raising an `ImportError`.

**2 · An embedder: any program that honours the contract.** xbrain carries no model
library. It runs `[embeddings].command` as a subprocess — split with `shlex`, never
through a shell — as `<command> [--model M]`, writes one JSON request to its stdin
and reads one JSON response from its stdout:

```
in:  {"schema_version": "1", "model": "…|null", "texts": ["…", "…"]}
out: {"schema_version": "1", "model": "…", "dimension": N, "normalized": true, "vectors": [[…], …]}
```

The reference backend is `scripts/xbrain-embed` (sentence-transformers, local:
nothing leaves the machine). Its dependencies are not in `pyproject.toml`, and its
shebang is `#!/usr/bin/env python3`, so a `command` naming the script alone runs
whichever `python3` comes first on `PATH`. Name the interpreter that has
`sentence-transformers` installed, as the bake-off did — its
[§9](embeddings-bakeoff.md#9-cómo-re-derivarlo) is the recipe that was actually
executed, environment and weights included. The wrapper's default model
(`intfloat/multilingual-e5-base`) is a starting point, **not** a model chosen by
evaluation.

Try the command by hand before indexing. The request below is yours, so whatever
the backend prints — a traceback included — is safe to read, which is not true of
a failure during a build (see [troubleshooting](troubleshooting.md#embedder--exited-n--its-stderr-is-not-repeated-here)):

```bash
printf '%s' '{"schema_version": "1", "model": null, "texts": ["hola"]}' \
  | /path/to/embedder-env/bin/python /path/to/xbrain/scripts/xbrain-embed
```

### Configuring it

```toml
[embeddings]
command = "/path/to/embedder-env/bin/python /path/to/xbrain/scripts/xbrain-embed"
# model = "intfloat/multilingual-e5-base"  # omit → the backend's own default
# batch_size = 64                           # texts per subprocess call (>= 1)
# timeout_seconds = 600                     # wall-clock cap per call (>= 1)
# query_prefix = "query: "                  # E5/BGE want these; most models want ""
# passage_prefix = "passage: "
```

| Key | Default | Read by |
|---|---|---|
| `command` | `""` — the plane is off | `index build --embeddings` and every `vector`/`hybrid` query |
| `model` | unset → the backend's default | `index build --embeddings` only |
| `batch_size` | `64` | `index build --embeddings` |
| `timeout_seconds` | `600` | the build and every query |
| `passage_prefix` | `""` | `index build --embeddings`, which records it in the manifest |
| `query_prefix` | `""` | recorded in the manifest by the build; **queries read the manifest's copy** |

The last column is the part to read twice. A query has to land in the geometry the
plane was written in, so `search` takes the **model and the query prefix from the
manifest**, not from `config.toml`. Editing `model` or a prefix changes nothing until
the next `index build --embeddings --force`, and a backend that answers a query with
a model other than the manifest's is refused even at the same dimension.

### Building it

```bash
uv run xbrain index build --embeddings            # no index yet
uv run xbrain index build --embeddings --force    # over an existing index
uv run xbrain index status                        # the `embeddings` line and the plane's verdict
```

What `--embeddings` does, in order:

1. refuses an empty `command` before anything is built;
2. sends **one probe batch**: the model, dimension and normalization the backend
   declares become the plane's spec, and a missing or non-executable binary fails
   here, before a byte is written;
3. builds the lexical plane exactly as a plain `build` does, and commits it;
4. embeds each **distinct** chunk text once — identical texts share one row, each
   chunk keeping its own id, owner, author and URL — in `batch_size` batches, each
   held to the probe's dimension and model. Every response is validated (count,
   shape, finite values, no zero vector, UTF-8, schema version) and re-normalized
   when it is not unit length;
5. writes `vectors.f32` (a float32 matrix) and `vectors.meta.json` beside
   `knowledge.db`, and only then seals the manifest, whose `embeddings` block is
   exactly the spec: `model`, `dimension`, `normalized`, `query_prefix`,
   `passage_prefix`.

A plain `build --force` without `--embeddings` deletes an existing plane and seals
`embeddings: null`: a matrix the manifest does not declare is refused, never queried.
`--embeddings --dry-run` does not call the embedder and, from the CLI, reports the
lexical counts only. The human build line prints no vector numbers; `--json` carries
`vector_chunks` and `vector_rows`.

**There is no vector-only rebuild.** `--embeddings --force` re-derives the lexical
plane from the store too, and the previous manifest and database are gone before the
first row is written. If the embedder then fails — a timeout, a crash, a batch of
another dimension or model — **no index answers, not even lexically**, until
`xbrain index build` runs again. Every refusal of the plane ends with that warning
spelled out. A fresh build whose embedder fails leaves the lexical rows committed and
**no manifest**, so every door refuses the directory: nothing is rolled back, whatever
Plan 03 §5 row 5 promised, and the tests pin the state that actually happens.

### Keeping it current: `update` does not re-embed

`xbrain index update` rewrites the chunks of every item that moved and **never calls
the embedder** — Plan 03 §2.3's "update re-embeds only the changed chunks" was not
built. What keeps that from being silent:

- `index update` reports the plane's debt (in `--json`: `vector_state`,
  `vector_missing` — chunks with no vector of their current text — and
  `vector_orphaned` — rows whose chunk is gone);
- `index status` names what is wrong with the plane on its `→` line, with the rebuild
  command: declared files that are gone, files the manifest does not declare, a meta
  it cannot load or `numpy` missing, chunks with no vector of their current text. It
  prints the reason, not a state name;
- a `vector`/`hybrid` query over a plane that is behind still runs the vector channel
  over the chunks it covers, **never serves a stale vector**, and declares
  `vector_plane_behind`.

So after `enrich`, `topics`, `fetch` or `digest-video`: `index update` keeps lexical
search current, and `index build --embeddings --force` is what brings vector coverage
back.

### Querying it

```bash
uv run xbrain search "…" --strategy hybrid
uv run xbrain search "…" --strategy vector --json
```

- **`vector`** embeds the query (one subprocess call, with the manifest's model and
  query prefix) and ranks chunks by cosine. It serves only what the geometry found.
- **`hybrid`** takes bm25's top chunks and the vector plane's top chunks — a fixed
  window of `FUSED_CHUNK_WINDOW` per channel, so every page is a slice of one
  ranking, and a channel that fills the window marks the response `truncated` —
  and fuses them by **Reciprocal Rank Fusion**:
  `score = Σ w_channel / (RRF_K + rank_channel)`. It then groups by item like
  `lexical`, and tops up from the profile plane; an item that arrives only by its
  profile comes back with `matches: []`, never with a fabricated excerpt.
- Both channels go through the **same fingerprint gate** the lexical door applies.
  The vector plane stores no text, owner, author or URL — only which chunk a row
  belongs to — so a vector hit is hydrated from the lexical plane's row.
- **Every fused match explains itself.** `matched_by` names the channels that found
  the chunk; `lexical_rank` and `vector_rank` are `null` when that channel did not
  find it, never `0`; the human view prints `via lexical+vector`. `score` is the RRF
  signal on `vector`/`hybrid` (bm25's on `lexical`): uncalibrated, and not a
  probability.
- `RRF_K = 60` and weights `1 / 1` in `knowledge/fusion.py` are the standard
  starting point, **not measured values**: `xbrain eval --strategy hybrid
  --sweep-fusion …` exists and has not been run on any candidate.

### When the vector channel cannot run

**The line that is not crossed: a response names `vector` or `hybrid` only when the
vector channel ran** — a plane the manifest declares, loaded under the manifest's own
spec, and the query vector in hand. Short of that, `hybrid` answers
`strategy: "lexical"`, `degraded` names the cause, and not one match carries `vector`
in `matched_by` or a `vector_rank`; `vector`, asked for by name, is an error. Two
situations are errors under **both** strategies, because degrading would hide a
misconfiguration that makes every later vector answer wrong.

Rows 1–6 are Plan 03 §5; the last three are what the code added.

| Situation | `index build --embeddings` | `search --strategy hybrid` | `search --strategy vector` |
|---|---|---|---|
| **1** · `[embeddings].command` empty | refused before building: `` `--embeddings` necesita un embedder: configura `[embeddings].command` … `` | `lexical` · `embeddings_not_configured`, and the line `⚠ Pediste hybrid y ha respondido lexical: [embeddings].command está vacío…` | **error** naming `[embeddings].command` |
| **2** · binary missing or not executable | the probe raises `embedder '…' not found — install it or set a valid [embeddings].command…` (or `could not be executed`); nothing written | `lexical` · `embedder_unavailable` | **error**: that same message |
| **3** · the manifest declares a plane whose files are gone | — | **error**: `no hay plano vectorial en … Reconstruye el índice con su plano vectorial: xbrain index build --embeddings --force …`; the query is not embedded | **error**: same |
| **4** · the backend returns another dimension, or declares another model | fails mid-build (`models are never mixed`) — with `--force`, no index until `index build` | **error**, never degraded: a backend serving another model is not a backend that is down | **error**: same |
| **5** · timeout, non-zero exit, unusable output | fails mid-build (`timed out after Ns`, `exited N`…): no manifest sealed; with `--force`, no index until `index build` | `lexical` · `embedder_unavailable` | **error**: the backend's own message |
| **6** · the index has no plane (built without `--embeddings`) | — | `lexical` · `no_embeddings` (plus `embeddings_not_configured` if the command is empty too); the embedder is not called | **error**: `…este índice no tiene plano vectorial: configura [embeddings].command … y ejecuta xbrain index build --embeddings --force…` |
| any filter (`--topic`, `--from`, `--kind`, …), plane and command in place | — | `lexical` · `vector_filters_unsupported` | `lexical` · `vector_filters_unsupported` |
| the plane is behind the lexical base | — | `hybrid` runs · `vector_plane_behind` | `vector` runs · `vector_plane_behind` |
| `numpy` not installed, plane in place | — | **error** naming `uv pip install 'xbrain[embeddings]'` | **error**: same |

Rows 3 and 5 of the build are where the code **diverges from the plan on purpose and
says so**: nothing is queried half-built, and nothing is rolled back either. The
filter row is the other one to know: `vector` is the one strategy that otherwise never
degrades, and a filtered `vector` request does. In the human view,
`vector_filters_unsupported` and `vector_plane_behind` have no sentence of their own
yet and print as the bare flag (`⚠ vector_filters_unsupported`); the header still says
`estrategia lexical` when the channel did not run.

### What it costs

No figure is republished here. The only measurement is the bake-off's, of one
candidate on one laptop that was already swapping; its
[§6](embeddings-bakeoff.md#6-coste-indexación-disco-latencia-y-memoria) states those
conditions beside every number. Two properties hold whatever the numbers turn out to be:

- the reference embedder serves **one request per process and loads the model every
  time**, and `search` calls it once per query — so on `vector`/`hybrid`, embedding the
  query dominates the latency, not retrieving. `hybrid` as a default would need a
  persistent embedder first, whichever model wins;
- the matrix holds `dimension × 4` bytes per **distinct** chunk text, plus
  `vectors.meta.json`, which maps every chunk id to its row.

### The bake-off: incomplete

[`docs/embeddings-bakeoff.md`](embeddings-bakeoff.md) is the measurement, with its
population, its conditions and its re-derivation. What it decides:

- **Plan 03 §13.8 — a bake-off with ≥ 3 candidates, winner and losers — does NOT
  PASS: 1 of 3.** `paraphrase-multilingual-MiniLM-L12-v2` was measured and lost under
  both strategies. `multilingual-e5-small` was interrupted mid-build by memory and
  disk pressure; `multilingual-e5-base` did not start; `bge-m3` and
  `jina-embeddings-v3` were not run.
- **There is no winner.** `hybrid` is not promoted, `lexical` stays the default, and
  the fusion constants did not move. That is "the cheap floor does not beat lexical",
  not "no model beats lexical": the candidates that could have were never measured.
- Closing Plan 03's §13.8 means running the remaining candidates on a machine without memory
  pressure, with `xbrain eval --strategy vector|hybrid --embeddings-model <model>`,
  following the bake-off's §9.

### What the vector plane does not solve

- **Filters.** The plane has no filter columns; with the plane and the command in
  place, a filtered request is answered lexically, and `eval` leaves the `filtros`
  cases unmeasured under `vector`/`hybrid`.
- **Coverage after `update`.** There is no incremental re-embed and no vector-only
  rebuild: `index build --embeddings --force` rebuilds both planes.
- **Query latency** with the reference embedder: a subprocess and a model load per query.
- **The profile plane is outside every published number.** `eval` scores chunk
  rankings under all three strategies, so `hybrid`'s profile top-up is not measured.
- **Real use.** Every scorable case in the golden set the bake-off measured is
  `construido`; the `real` provenance has no coverage.
- **Verification.** `matched_by` says which channel found a chunk, not whether the
  chunk supports a claim. That is still `xbrain get` and the `verifica con` line.

## The graph — opt-in, and measured negative

`index build` also writes a small graph into `knowledge.db`, next to the lexical
planes. It exists to **explain and explore** — which topics an item sits in, which
topics travel together, which items support that — and not, as measured, to
improve search.

### What is in it

- **Nodes:** items and topics. Nothing else.
- **`HAS_PRIMARY_TOPIC` / `HAS_TOPIC`:** one edge per (item, topic) that `enrich`
  assigned, primary kept apart from secondary. Never pruned.
- **`CO_OCCURS_WITH`:** topic ↔ topic, in both directions with the same weight,
  when at least `graph_min_shared_items` items carry both **and** their Jaccard
  index (shared items ÷ items carrying either) reaches `graph_min_weight`. Jaccard
  rather than a raw count, so a topic assigned to half the corpus does not
  co-occur with everything. Each topic then keeps its
  `graph_max_neighbors_per_node` strongest edges. Each edge keeps `shared_items`
  (the full count), up to 20 `supporting_item_ids`, its `method`
  (`topic-cooccurrence/v1`) and the fingerprints of what it was derived from.
- **No item → item edge exists**, and a `CHECK` on the table refuses one: two
  items are related only through a topic they share, so the path always shows
  which assignment connects them.

An edge means *xbrain assigned these topics together to these items of this
corpus*. It never means the concepts are related in the world, and the response
carries that in its data (`semantics: "co_occurrence_in_corpus"`), not only in
this sentence.

On `store-2495`, the sweep's store ([Measured versions](#measured-versions)), at
the default thresholds:
**5,783 edges** — 2,495 `HAS_PRIMARY_TOPIC`, 3,128 `HAS_TOPIC`, 160
`CO_OCCURS_WITH` among 41 topics — in **1.6 MiB** of the database.

### Keeping it current

The manifest seals a `graph` block — `algorithm_version`, the three thresholds and
the edge count. `index update` rewrites the whole graph plane whenever any item,
the vocabulary or the topic pages changed, and also when the thresholds or the
algorithm version in force differ from the sealed ones — a change that, on its
own, leaves the lexical planes untouched. A no-op update leaves `knowledge.db` byte-identical
and only reseals the manifest. **`index status`
does not report that last case yet**: after editing a threshold it says nothing is
behind, while `update` would rewrite the edges (backlog). Run `index update` after
changing `[index].graph_*`.

### Reading it: `graph-expand`

```bash
$ uv run xbrain graph-expand --item 2063609922667815064
This edge reflects co-occurrence in this corpus, not a relationship in the world.
item:2063609922667815064 → topic:agentic-engineering
item:2063609922667815064 → topic:ai-agents
item:2063609922667815064 → topic:ai-coding
```

The human view prints the disclaimer (in `[output].language`) and one path per
reached node. `--max-hops 2` reaches the co-occurring topics and the other items
of each topic (`item → topic → topic`, `item → topic → item`); `--max-neighbors N`
keeps each node's N strongest edges
(10 by default). `--json` returns the `GraphExpansionResponse`: nodes, edges and
paths, each edge with `relation`, `method`, `weight`, `shared_items` and
`supporting_item_ids`.

Two guarantees, both enforced when the response is built:

- **every path rests on ids that resolve in the live store** — an edge whose
  listed support has left the store makes the whole expansion refuse, it is never
  served half-true;
- **an index behind the store is refused**, not expanded: its edges may belong to
  a corpus that no longer exists. `search` only declares that state; `graph-expand`
  stops (`Ejecuta xbrain index update`).

One gap, in the backlog: an `--item` that does not exist returns exit 0 with a
one-node expansion, indistinguishable from an item with no topics.

### `hybrid_graph`, and why it is off

`hybrid_graph` runs `hybrid` (lexical plus the vector channel when it can open)
and then lets the graph re-order the page: items reachable from the best result
through its topics get an extra RRF term. **The graph re-orders, it never
admits**: a neighbour no channel scored is not a result.

It is **off by default** (`GRAPH_ENABLED_BY_DEFAULT = False`). Neither
`xbrain search`, nor MCP, nor a plain `xbrain eval --strategy hybrid_graph` can
switch it on: all three answer `lexical`, declaring
`hybrid_graph_not_implemented`. Only `search(..., graph_enabled=True)` turns it
on, and the only command that calls it that way is the sweep below,
`xbrain eval --strategy hybrid_graph --sweep-graph …`. Measuring `hybrid_graph`
at the applied threshold alone is therefore a one-cell sweep
(`--sweep-graph "min_shared_items=5 min_weight=0.05"`).

### The threshold sweep: a negative result

[graph-threshold-sweep.md](graph-threshold-sweep.md) measured all 16 cells of
`min_shared_items ∈ {2, 3, 5, 8}` × `min_weight ∈ {0.0, 0.02, 0.05, 0.10}` on the
golden set (18 measured cases on `store-2495` and `golden@d1423c8`, see
[Measured versions](#measured-versions); 2026-09-15). The result:

- **every cell made `recall@10` worse** than the ranking it re-orders (Δ between
  −0.12 and −0.18 on a base of 0.63), and every cell lost more than 3 pp of
  precision in at least one stratum;
- the `expansion` stratum was populated first — **33** relevant results that only
  a graph path could reach — and the graph lifted **0 of 33** into the top 10, in
  all 16 cells;
- so **`hybrid_graph` is not promoted** and `lexical` stays the default. The
  applied threshold, `5 / 0.05`, is simply the least damaging cell, applied
  because every build writes a graph.

The sweep's §4 explains the mechanism (the graph term's weight and the fixed
neighbour budget, not the threshold, decide the damage) and what was not swept.
Its base was lexical (no embedder configured), so the effect over a real fused
`hybrid` ranking is unmeasured.

## Serving it to an agent

`xbrain mcp-serve` offers `xbrain.search`, `xbrain.get` and `xbrain.graph_expand`
over stdio, returning the same models as `--json`, read-only, with retrieved text
labelled as untrusted data. Install and client configuration:
[docs/mcp.md](mcp.md). How an agent should use the answers:
[docs/knowledge-for-agents.md](knowledge-for-agents.md).

## The spec's acceptance criteria: 12 of 15 met

The design spec behind this page ends with fifteen acceptance criteria (its §13).
This is their state, and where to check each one. Test paths are under `tests/`.

| § | Criterion (spec §13, verbatim) | What proves it | State |
|---|---|---|---|
| 13.1 | un agente puede buscar por concepto, frase exacta, topic y filtros estructurados | Filters: `test_knowledge_lexical.py::test_every_declared_filter_is_actually_pushed_to_sql`. Topic: `test_knowledge_search_service.py::test_a_topic_note_match_returns_the_topics_supporting_items`. Concept: `test_knowledge_search_hybrid.py::test_vector_serves_only_what_the_vector_channel_found`. The gap: [There is no phrase search](#known-limits-of-the-lexical-baseline) | **NOT MET.** There is no exact-phrase search: `match_expression` (`lexical_fts.py`) joins the terms with `OR`, quotes included. Concept search exists only on the opt-in vector plane, and how good it is remains §13.5's open question. |
| 13.2 | puede recuperar la fuente real sin depender del summary | `test_knowledge_get_service.py::test_get_returns_the_whole_article_body_untruncated`, `::test_get_works_with_the_index_directory_deleted`; `test_knowledge_search_service.py::test_a_summary_match_points_at_the_underlying_article` | Met |
| 13.3 | cada fragmento expone procedencia, autoría y localizador | `test_knowledge_search_service.py::test_a_quoted_post_match_carries_the_quoted_author_not_the_poster`, `::test_a_match_locator_is_the_surface_locator_plus_the_character_range`; `test_mcp_prompt_injection.py::test_every_served_fragment_carries_its_three_labels` | Met |
| 13.4 | summaries, digests y topic syntheses están disponibles pero etiquetados como derivados | `test_knowledge_provenance.py::test_is_derived_is_true_exactly_for_machine_produced_text`, `::test_unknown_fails_closed_to_llm_synthesis`; `test_knowledge_search_service.py::test_a_derived_match_with_no_primary_source_says_so` | Met |
| 13.5 | la búsqueda textual y la vectorial se evalúan por separado y juntas | The instrument: `test_knowledge_evaluation.py::test_metrics_are_reported_per_stratum_and_provenance`, `::test_hybrid_fuses_both_channels_and_names_itself`. The evaluation: [`embeddings-bakeoff.md`](embeddings-bakeoff.md) §0 and §10 | **NOT MET.** The instrument exists; the evaluation does not. The bake-off Plan 03 owed this criterion (its own criterion 8: ≥ 3 candidates) measured **1 of 3** ([above](#the-bake-off-incomplete)). |
| 13.6 | el índice incremental detecta cambios y nunca sirve chunks stale | `test_knowledge_index_invalidation.py::test_update_touches_only_the_changed_item`; `test_knowledge_search_service.py::test_a_chunk_with_a_manipulated_fingerprint_is_excluded_and_counted`, `::test_editing_the_store_without_reindexing_declares_the_index_behind`; `test_knowledge_search_hybrid.py::test_a_vector_left_stale_by_update_is_not_served_and_the_response_says_so` | Met, by the spec's own definitions: a stale chunk is excluded and counted, and an index behind the store is declared. The cheap signal's blind spot is in [Known limits](#known-limits-of-the-lexical-baseline). |
| 13.7 | `search` agrupa matches sin ocultar la superficie que produjo cada uno | `test_knowledge_search_service.py::test_a_long_transcript_yields_one_result_with_at_most_three_matches`, `::test_a_primary_match_names_ITSELF_not_every_primary_surface_the_item_has`; `test_knowledge_search_hybrid.py::test_a_chunk_both_channels_found_is_explained_by_both` | Met |
| 13.8 | `get` puede entregar fuentes largas de manera selectiva/paginada | `test_knowledge_get_service.py::test_a_body_over_the_budget_is_paginated_not_cut`, `::test_the_cursor_continues_where_the_previous_call_stopped`, `::test_asking_for_a_surface_the_item_does_not_have_lists_what_it_does` | Met (this is the **spec's** §13.8, not the bake-off's; see below). |
| 13.9 | el grafo mínimo explica paths y conserva support ids | `test_knowledge_graph_service.py::test_every_path_carries_node_types_relation_method_weight_and_support`, `::test_every_served_path_rests_on_item_ids_that_resolve_in_the_live_store`; `test_knowledge_graph_build.py::test_no_item_to_item_edge_exists_in_the_schema` | Met |
| 13.10 | la expansión por grafo puede activarse o desactivarse y tiene una métrica incremental | `test_knowledge_graph_strategy.py::test_hybrid_graph_existe_es_desactivable_y_el_default_no_cambia`; `test_knowledge_graph_sweep.py::test_the_graph_sweep_publishes_the_expansion_population_its_useful_column_counts_from`; [`graph-threshold-sweep.md`](graph-threshold-sweep.md) §0 | Met. The switch is `search(..., graph_enabled=True)`, which the CLI reaches only through `eval --strategy hybrid_graph --sweep-graph`. The metric is Δ recall@10 against `hybrid` plus the `expansion` stratum, and its result is negative. |
| 13.11 | CLI JSON y MCP usan los mismos modelos y producen semántica equivalente | `test_mcp_cli_equivalence.py::test_mcp_and_cli_json_are_structurally_identical`; `test_mcp_server.py::test_the_output_schema_is_the_plan01_model_itself`, `::test_mcp_refuses_with_the_same_message_as_the_cli` | Met |
| 13.12 | query y retrieval funcionan sin llamada a un LLM generativo | `test_mcp_server.py::test_the_mcp_server_imports_nothing_that_speaks_to_the_network` covers the MCP adapter only. No test covers the CLI doors (see below). Spot check: importing `xbrain.knowledge.search_service`, `get_service`, `graph_service` and `xbrain.mcp_server` leaves `anthropic` out of `sys.modules`, because every in-process generative call imports it lazily. The external vision command, a subprocess, is started by `digest-video --frames` and `redescribe-frames`, never by a query door | Met |
| 13.13 | ningún artefacto personal o índice entra en Git | `test_knowledge_index_schema.py::test_the_index_directory_is_git_ignored`; `test_knowledge_goldenset.py::test_the_golden_set_is_tracked_and_the_reports_are_not`. For the rest: `git check-ignore --no-index config.toml auth/storage_state.json data/items.json` lists all three, and `git ls-files data auth` lists only the two `.gitkeep` | Met |
| 13.14 | README, tutorial, arquitectura y troubleshooting se actualizan con el código de cada plan | README *Search & retrieval*; ARCHITECTURE *The minimal graph* and *The MCP server*; troubleshooting *The knowledge index* | **NOT MET.** `docs/tutorial.md` was last updated in Plan 02 (02.15). It teaches neither `vector`/`hybrid`, `graph-expand` nor MCP. |
| 13.15 | los resultados negativos de evaluación se documentan en vez de ocultarse | [`embeddings-bakeoff.md`](embeddings-bakeoff.md) §0; [`graph-threshold-sweep.md`](graph-threshold-sweep.md) §0; [The graph — opt-in, and measured negative](#the-graph--opt-in-and-measured-negative) | Met |

**Which §13.** «§13.N» is ambiguous. The spec, Plan 01, Plan 02 and Plan 03 each
have a §13:

- only the spec's and Plan 03's list acceptance criteria (Plan 01's §13 is its
  quality gates, Plan 02's its documentation);
- Plan 04 has no §13: its criteria are its §11.

The table above is the **spec's** list. So «§13.8 — NO CUMPLE: 1 de 3» in the
bake-off is criterion 8 of **Plan 03**, the ≥ 3-candidate bake-off. What it leaves
unmet in the spec is §13.5. The spec's own §13.8 is `get`'s pagination, and it is
met. Likewise, the «§13.12» in `test_knowledge_cli.py` and
`test_knowledge_degradation.py` is Plan 03's criterion 12 (the `[embeddings]`
extra), not the spec's «sin llamada a un LLM generativo».

**A table, not a test.** Until Plan 04.8 this table was an executable test,
`tests/test_spec_closure.py`. It was removed as a scope decision: 1,127 lines of
machinery to guard fifteen sentences, and each of three review rounds found
another way to leave it green. The criteria's proofs are ordinary tests, and
deleting one already leaves the suite a test short, which is visible without
extra machinery. Nothing watches this table, so when a criterion changes state,
edit its row. Two tests left with that file: the check of the spec's §13.12 through
the CLI doors, and the `git check-ignore` check behind the spec's §13.13. The
spot checks in those two rows replace them.

## Configuration

Everything has a default; the whole `[index]` section is optional. The
`[embeddings]` section is optional too, and unset is a supported state:
[configuring it](#configuring-it) is above. See
[`config.toml.example`](../config.toml.example).

```toml
[index]
# dir = "index"                 # under data/ — must resolve INSIDE data/
# max_matches_per_item = 3      # fragments one item may cite in a search
# get_char_budget = 40000       # per-response ceiling before truncate + cursor
# graph_min_shared_items = 5    # a CO_OCCURS_WITH edge needs this many shared items…
# graph_min_weight = 0.05       # …and at least this Jaccard (both measured, see above)
# graph_max_neighbors_per_node = 10   # strongest co-occurrence edges kept per topic (not swept)
```

`max_matches_per_item` is what stops a long transcript filling the top ten with
ten adjacent windows of itself. `dir` is validated at config load: an absolute
path, a `..` or a symlink that escapes `data/` is refused, because
`index build --force` deletes and recreates whatever it finds there.

---

Something broken? → [Troubleshooting](troubleshooting.md#the-knowledge-index).
