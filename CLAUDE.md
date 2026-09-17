# CLAUDE.md — xbrain

Python CLI (`xbrain`) that extracts X bookmarks/tweets into a JSON store and
generates an Obsidian wiki.

## Stack
- Python 3.12+ (venv currently runs 3.13), `uv`, `pydantic` v2, `typer`, `playwright`, `trafilatura`, `pytest`.
- `uv pip install` needs `--index-url https://pypi.org/simple` to bypass the
  machine-wide private FITIZENS pip index.

## Architecture
- Pipeline — **six ordered stages**: `extract → fetch → vocab → enrich → topics → generate`,
  with `data/items.json` as the hub every stage reads and writes back. `sync` runs only the
  mechanical three (`extract → fetch → generate`); `vocab`/`enrich`/`topics` are the LLM
  stages and are run explicitly, on your own cadence. `media → describe` is a side-pipeline
  that feeds enrich/topics (below). `import-archive` and the backfill family
  (`refresh-quoted`, `refresh-media`, `download-videos`, `reextract`, `refetch-truncated`)
  sit OUTSIDE the pipeline — one-off repairs, never steps in it.
- **`eval/golden-set.yaml` is TRACKED — the single exception to "nothing personal in Git" —
  and the loader has two stages because of it.** Untracked (it lived under `data/*`), the v3
  migration would have appeared in no diff, `xbrain eval` could never run in CI, and a case
  edited to turn a gate green would leave no history. It holds questions, ids and short
  identifying fragments; a 300-char ceiling on `expected_text`, checked in CI **against the
  real file**, keeps a corpus body out. `load_cases(path)` validates STRUCTURE without opening
  the store (so it runs in CI, where there is no `data/`); `resolve_cases(cases, store)` checks
  the ids (local, or CI against fixtures). Fusing them would make the very test that proves the
  evaluation runs in CI the first one that cannot. **Only an ENUMERATED case scores**: with
  `relevant_items: []` the recall@k is 0/0 and comes out 1.0 or 0.0 depending on the
  implementation — rule 2 — so unenumerated cases are archived as `scenarios` with their reason.
  Migration measured 2026-08-31 against 2,404 items: D1c/U2 enumerates to **exactly 12** (so the
  Plan-03 bake-off keeps its deciding stratum), P1 to **6** where the file said 5, U3 to **22**
  where it said 20 — the two moved because the corpus grew, which is why the figures are notes
  and never asserts. **`video_digest` still has NO case, and now the reason is measured**: of
  the 36 proper nouns appearing only in a digest, **22 LEAK** (a fuzzy variant sits in the
  transcript, sometimes ASR-mangled — "Johannes Trithemius" vs "johannes tritemius", ratio 0.97)
  and the other **14 have no support on any surface**, i.e. candidates for invention. The first
  group fails the anexo-A.3 leak rule; founding a case on the second would enshrine a possible
  hallucination as ground truth. Zero usable candidates — and the population measured is proper
  nouns, not all facts. **A case whose filters the strategy cannot apply is UNMEASURED, not
  0.0**: under Plan 01 the FTS5 baseline pushed only `has_surfaces`/`origins` into `WHERE`, and
  the first real run reported `filtros: recall@10 = 0.0`, which reads as "retrieval failed at
  filtering" when the instrument does not exist yet (spec §8.6.8). **That gap is now closed and
  the RULE is what survives** (PR #179): the harness builds through `index_build`'s writer, the
  same one `xbrain index build` drives, so all **eight** filters of spec §7.2 are pushed and the
  two `filtros` cases are scored rather than reported unmeasured. `SUPPORTED_FILTERS` is derived
  from `SearchFilters.model_fields`, so a ninth filter added to the frozen contract cannot
  silently keep the set at eight. Read "only two" as history; do not quote it as a limit. The baseline is the SAME FTS5 the persisted
  index will use, on `sqlite3(":memory:")` — same DDL, same `unicode61 remove_diacritics 2`
  (no stemming: FTS5 has none multilingual, and the English one would wreck the Spanish half),
  same `bm25()`, same explicit `chunk_id` tie-break — so what dies later is where the database
  lives, not how it scores. **And that last clause was FALSE while the terms were ANDed.** The
  FTS5 default conjunction requires every word of a question inside ONE chunk: measured on the
  real corpus, 18 of the 21 scorable cases got back NOT ONE ROW, and the only three that
  retrieved anything were single-term `exacto` queries. So bm25 ranked nothing in 18 of 21
  cases, and the published `semantico: 0.0` / `cruzado_idioma: 0.0` / `topic: 0.0` measured the
  QUERY BUILDER while the execution report read them as an absence of vocabulary overlap. The
  connective is now a **disjunction** (`FTS_CONNECTIVE`, recorded in the ranking fixture beside
  the tokenizer): empty result sets 18/21 → 0/21, recall@10 0.1429 → 0.8099, MRR 0.1429 →
  0.7206, `exacto` unchanged and **no stratum regressed**, at p50 0.23 → 9.75 ms. A conjunction
  in front of bm25 is two retrieval models stacked: bm25 wants a wide candidate set and
  discriminates by IDF, and requiring every term does that by brute force *before* the scorer
  runs. **The limit that remains is that IDF is relative to THIS corpus**, so a word that reads
  as a function word can still be rare to the index and go undiscounted — `el` is 1 of 49
  fixture chunks (2.0 %) and **6,070 of 22,286** real ones (**27.2 %**), re-derived 2026-09-01
  on the then-shipped chunker (v2, `800/0`, `store-2404-0901`, not reproducible — see the versions
  table in docs/knowledge-index.md#measured-versions), which is why a fixture
  query ranks it high and the real corpus does not. **`CHUNKER_VERSION` is `v3` since Plan
  02.9**, and this count still stands: v3 changed the FINGERPRINT projection (the served title
  joined the hashed tuple) and not the cut, proven by the version-stripped ranking fixture
  being byte-identical across the bump — so the chunk ids moved and the chunk COUNT did not.
  Do not re-stamp a measured figure to a new version without that proof: which of the two a
  version bump touched is the whole question. *(It read `5,748 of 18,319 (31.4 %)`, which was
  correct for the PROVISIONAL chunker v1 and for `store-2404-0831`; the chunker moved in this
  branch, the store moved too (`store-2404-0901` is another file with the same 2,404 items), and
  the derived figure did not — rule 6. Read the old pair as history.)*
  **The pair Plan 03's vector layer has to beat is `recall@10` 0.7391 · MRR 0.7357** — NOT the
  `0.8099 / 0.7206` above, which measured the pre-#179 in-memory harness and is retired with
  it. Those are the SHIPPED `800/0` chunker's, scored through `index_build`'s writer by the
  harness in `evaluation.sweep_chunker` (`store-2474`, 22,933 chunks) against the TRACKED
  golden set, `golden@d1423c8` — re-derived 2026-09-16; every version named here is spelled
  out, hash and check command, in docs/knowledge-index.md#measured-versions. It read
  **0.7395** until then, and the documents that still quote 0.7395 (the bake-off among them)
  are right about THEIR pair: the same store and chunks against a golden set from before
  `d1423c8`, with U3 at 22 relevant instead of 24. There are TWO such sets, and both score
  0.7395: `d1423c8^` is `golden@a88c753`, and the bake-off's (`547a860`, before `a88c753`,
  which only added `expansion` stratum labels and their notes) is `golden@427fea9`. A recall is a fact about the corpus
  AND the case set; a figure that names only one of them is rule 2's missing population. Read
  `0.7357` as `800/0`'s OWN MRR and never as the winner's: the same sweep reports `800/150`
  tied at `recall@10` (0.7391; 0.7395 on `golden@427fea9`) and ahead on MRR at 0.7360, under
  both, so pairing the winner's recall with this MRR is rule 6 in one line. That, and no stemming, is what the vector layer has to beat — and the bake-off that tried
  has not beaten it (next bullet but one: incomplete, Plan 03 §13.8 NOT MET).
  Picking between `OR`, minimum-should-match and per-term weighting is Plan 02's sweep.
  **A threshold that reached no bucket is a FAILURE, not a pass**: `--min-recall`
  counts the comparisons it made and fails closed at zero, because `passed = not failures` let
  `--min-recall 1.0` exit 0 having scored nothing.
- **The persistent index (`data/index/`, `xbrain index build|update|status` · `search` · `get`)
  is DERIVED, and that is what licenses every refusal in it.** SQLite + FTS5, two planes —
  `chunks_fts` over fragment bodies (what a citation quotes) and `profiles_fts` over one
  retrieval profile per item (what answers a query whose words appear in no fragment; never
  served as a citation). Deleting the directory costs one `build`, so an incompatible manifest,
  a drifted column or a torn page is **refused whole** rather than queried partially — a partial
  answer over a schema this code no longer matches is a wrong answer wearing a right one's shape
  — and every incompatibility ends with the same sentence, `xbrain index build --force`.
  **`rowid` is an explicit `INTEGER PRIMARY KEY`**: with an implicit one SQLite may reuse a
  deleted row's rowid and external-content FTS5 would return the NEW chunk for the OLD word.
  **Two change signals, and confusing them is the trap.** Four DEEP fingerprints (item · store ·
  vocabulary · topics) drive `build`/`update`/`status` and answer *what changed*; one CHEAP
  `StoreSignal` — `mtime_ns` + size of the THREE inputs (`items.json`, `vocab.yaml`,
  `topics.json`, six required fields) — is three `os.stat`, so a query can afford it on every
  call and declare `index_behind_store`. Three inputs because comparing `items.json` alone left
  `xbrain topics` writing a new topic plane that every later `search` answered over silently.
  The cheap signal is **falible in one declared direction**: a `touch` with no edit is an
  accepted false positive (a false positive costs a warning, a false negative serves stale
  evidence as fresh), and a same-size replacement with the mtime preserved (`cp -p`, `rsync -a`,
  `unzip`, a restored backup) is invisible to it FOREVER. Nothing promises freshness from
  `mtime`+size. **The query door refuses and `index status` REPORTS** — same sentence, opposite
  behaviour, on purpose (rule 9): `status` is the instrument you run to find out, and it pays
  for `PRAGMA quick_check` (155–850 ms on the 52 MiB real index) which no query door can. The
  open door instead runs one trivial `MATCH` per FTS plane (0.01 ms each), because page-1 and
  `sqlite_master` reads touch no FTS5 shadow table: with `chunks_fts_data` dropped, `search`
  died in a traceback while `status` exited 0 and called the index healthy. **`search` filters
  BEFORE it scores** (all eight, incl. `content_kinds` and `has_surfaces`), excludes rows it
  cannot serve honestly — no locator, or a fingerprint that does not recompute over the served
  projection — into `corrupt_chunks_excluded` (named `stale_chunks_excluded` until someone
  noticed it SOUNDED like the staleness signal and MEASURED the consistency one), groups by item
  at `max_matches_per_item` (3), and hydrates `verification_status` from the **live store**, never
  the index. **`get` never reads the index at all** and works with `data/index/` deleted: an
  index able to answer it would be a second copy of the corpus that nothing invalidates.
  Degradations are a fixed-order tuple, DECLARED not simulated: `no_embeddings` is read off the
  manifest's `embeddings` block (not hard-coded — an index built with `--embeddings` stops
  declaring it by itself), a requested vector channel that did not run is named by cause (next
  bullet), `hybrid_graph` degrades to lexical labelled `hybrid_graph_not_implemented` while its
  switch is off (the default, and the only state `search`/MCP reach — next bullet), and a TYPO
  raises (a typo is not a degradation; answering it with lexical results would turn it into a
  measurement). *(This line said `--strategy vector` degrades labelled `vector_not_implemented`;
  true until Plan 03.6, false since — `vector` without vectors is now an error.)* `render.py` is the human view of the SAME
  response model `--json` serialises and reaches back into nothing. **Measured 2026-09-12 on the
  live corpus (`store-2474`: 2,474 items · 45 topics): 10,570 surfaces · 22,933 chunks · 2,474 profiles,
  `build` 3.1 s, no-op `update` 0.8 s, `search --limit 10` 0.65 s wall (median of 5, dominated by
  loading the 17.3 MiB `items.json` for verification hydration), `knowledge.db` 52 MiB ≈ 3× the
  store.** HISTORY since 04.2: those figures predate the graph plane. On the 2,495-item store of
  the graph sweep (`store-2495`, index rebuilt 2026-09-16) `knowledge.db` is 53.6 MiB, 1.6 MiB of
  it `graph_edges` WITH its three indexes (the table alone is 0.64 MiB; `dbstat`); timings were not re-taken (machine swapping). Re-derive it; it
  moves with the corpus. **Known limits, declared not discovered:** no
  stemming (FTS5 has none multilingual and the English one would wreck the Spanish half) — top
  tens for `agente` and `agentes` share **0 of 10** items, measured on the 2,474-item corpus, while
  `transformer`/`transformers` share 7 — and IDF is relative to THIS corpus. Diacritics DO fold
  (`unicode61 remove_diacritics 2`). And `xbrain eval` is **no longer a different filter surface** (PR #179):
  the harness stopped walking the corpus its own way and now builds through `index_build`'s
  writer, so it pushes the **same eight** filters `search` does and the two `filtros` cases of
  the golden set are scored. **The rule outlives the gap**: a case whose filters a strategy
  cannot apply is still **UNMEASURED, never 0.0** — a zero from a filter nobody applied reads as
  "retrieval failed at filtering" when the instrument was not there — and that rule has a live
  instance again: the vector plane has no filter columns, so under `--strategy vector|hybrid`
  the two `filtros` cases are UNMEASURED, and `search`, with plane and command in place, answers
  a filtered `vector`/`hybrid` request lexically, declaring `vector_filters_unsupported`. The earlier
  line here said the harness pushes only two; it was true until #179 and is now false.
  Operation: `docs/knowledge-index.md`.
- **The vector plane and `hybrid` (Plan 03) are OPT-IN, and `lexical` is still the default.**
  Embeddings follow the `transcribe`/`vision` shape: `[embeddings].command` is an EXTERNAL
  subprocess (`embeddings.py`: `shlex`, no shell, JSON on stdin/stdout, `schema_version` "1",
  every row validated, normalization verified rather than trusted) with **no bundled default** —
  the model is chosen by the golden set, not by a default — and `scripts/xbrain-embed` is only the
  reference backend. **The backend's stderr is never relayed**: a crashing embedder's traceback
  quotes the chunk it failed on, i.e. the corpus. `numpy` is the `[embeddings]` extra, imported
  lazily, so `import xbrain` works without it and a query that needs the matrix names
  `uv pip install -e '.[embeddings]'`. `xbrain index build --embeddings` writes
  `data/index/vectors.f32` + `vectors.meta.json` and a manifest `embeddings` block that IS
  `VectorSpec` (model · dimension · normalized · both prefixes), as DECLARED by a probe batch; a
  query reads model and query prefix **off the manifest**, never off `config.toml`. The plane
  stores no text, owner, author or URL (rows keyed by `sha256(text)`, `chunk_id → row`
  many-to-one), so a vector hit is hydrated through the lexical `chunks` row and the same
  fingerprint gate. **Three facts where the code is NOT Plan 03's text — do not "fix" the docs
  back:** there is **no vector-only rebuild** (`--embeddings --force` re-derives the lexical
  plane too, and a failed embedder then leaves NO index, not even lexical — `VECTOR_REBUILD_ADVICE`
  says so); **`index update` never re-embeds** (the plane goes `behind`, `status` says so, and a
  query skips stale vectors declaring `vector_plane_behind`); **a failed build rolls nothing back**
  (lexical rows committed, no manifest, every door refuses). `hybrid` fuses by RRF
  (`fusion.py`; `RRF_K`=60 and weights 1/1 are unswept starting points, read at call time) over a
  fixed `FUSED_CHUNK_WINDOW` per channel, and every match keeps `matched_by` / `lexical_rank` /
  `vector_rank` (`None` for a channel that did not find it, never 0). **The line not crossed:** a
  response says `vector`/`hybrid` only when the vector channel ran; otherwise `hybrid` answers
  `lexical` naming the cause (`embeddings_not_configured` · `embedder_unavailable` ·
  `no_embeddings` · `vector_filters_unsupported`) and `vector` is an ERROR, filters or not. Only
  with plane AND command in place does a filtered `vector` request answer lexically, declaring
  `vector_filters_unsupported` (the check order is `search_service._resolve_channel`). A declared
  plane the disk cannot serve, and a query vector of another dimension or another model, are
  errors under BOTH (`tests/test_knowledge_degradation.py`, one test per §5 row). Full matrix
  with the message each case prints: `docs/knowledge-index.md`. **The bake-off is INCOMPLETE and
  Plan 03's criterion §13.8 does NOT PASS** (`docs/embeddings-bakeoff.md`, measured 2026-09-13): 1 of the
  ≥ 3 candidates required — `paraphrase-multilingual-MiniLM-L12-v2`, measured and losing under
  both strategies; `multilingual-e5-small` interrupted by memory and disk pressure;
  `multilingual-e5-base`, `bge-m3` and `jina-embeddings-v3` never run. So `hybrid` is NOT promoted
  and the fusion constants did not move. Read it as "the cheap floor does not beat lexical",
  NEVER as "no model beats lexical". Quote none of its figures without its §1 (corpus sha256,
  golden-set version, one laptop already swapping); re-derive with its §9.
- **The minimal graph, MCP, and the spec closure (Plan 04) — the corpus/world line is DATA.**
  `index build` writes `graph_edges` beside the lexical planes: items and topics only,
  `HAS_PRIMARY_TOPIC`/`HAS_TOPIC` per assignment and a Jaccard-weighted `CO_OCCURS_WITH` between
  topics (thresholds `[index].graph_*`, applied `5 / 0.05`, cap 10; no item→item edge, a SQL
  `CHECK` refuses one). An edge is **co-occurrence in this corpus, never a relationship in the
  world**, and `GraphExpansionResponse` carries it as single-value `Literal`s (`semantics`,
  `disclaimer_key`) so it survives an agent summarising the prose away. `graph-expand` serves
  explicit paths whose support must resolve in the live store (else the whole expansion is
  refused) and REFUSES an index behind the store. `hybrid_graph` re-orders a `hybrid` page and
  never admits an unscored neighbour; `GRAPH_ENABLED_BY_DEFAULT = False` and only
  `search(..., graph_enabled=True)` switches it on — from the CLI, ONLY
  `eval --strategy hybrid_graph --sweep-graph …`; a plain `eval --strategy hybrid_graph` answers
  `lexical` + `hybrid_graph_not_implemented`, exactly like `search`. **Its sweep is NEGATIVE**
  (`docs/graph-threshold-sweep.md`, 2,495 items, 18 cases, 2026-09-15): all 16 cells lowered
  recall@10 and lifted **0 of 33** graph-only pairs — not promoted, `lexical` stays the default.
  `xbrain mcp-serve` (`mcp_server.py`, `[mcp]` extra, stdio) is a thin adapter over the three
  services with the CLI's loaders, derived schemas, the CLI's operator errors as `ToolError`,
  and `CORPUS_IS_DATA` on every tool; the trust boundary is DECLARED, not locked (no own network
  call; the only external process is `[embeddings].command`, off by default). Operation:
  `docs/mcp.md`, `docs/knowledge-for-agents.md`. **Spec §13 is a TABLE in
  `docs/knowledge-index.md`** (criterion · what proves it · state): 13 of 15 met; **§13.1** (no
  phrase search: terms are ORed) and **§13.5** (the bake-off above) are NOT. **§13.14** has been
  met since 06.3, when `docs/tutorial.md` caught up with Plans 03 and 04. Nothing checks the
  table: edit the row when a state changes. It was an executable test,
  `tests/test_spec_closure.py`, until 04.8 removed it as a SCOPE decision, not a quality one.
  It was 1,127 lines guarding fifteen sentences, and
  across three review rounds it produced six blockers of one family: each round found another
  way to leave it green. Do not rebuild it without a different approach. «§13.N» is ambiguous —
  the spec, Plan 01, Plan 02 and Plan 03 each have a §13 (criteria only in the spec and Plan 03; Plan 04's criteria are its **§11**):
  name which.
  **Backlog, written so it is not lost:** no CLI/MCP switch for `hybrid_graph`; `index status`
  silent when the sealed graph thresholds/version differ from config (`index_build.py`,
  `index_store.py`); `graph-expand` on an unknown id exits 0 with one node; the SQL `CHECK` has
  no test; `resolve_locator` has no consumer; `max_hops`/node caps not in `config.toml`;
  `graph-expand`'s human view is printed inline in `cli.py`, not by `render.py` (the
  disclaimer sentence itself lives in `i18n.Strings`); stale strings (`implementadas hoy: lexical`, "no tiene
  backend todavía"). The unknown-id exit, `resolve_locator`, the caps and the
  inline disclaimer are PR #193's unlabelled backlog — the «F3–F6 of 04.3», a mapping no written
  record confirms.
- `data/items.json` (dict keyed by tweet id) is the source of truth; markdown
  is derived. All stages are idempotent and incremental.

Everything else about a subsystem lives in exactly one document. Read its row
before you touch it:

| Subsystem | Read |
|---|---|
| Install, configure, run it end to end | [README.md](README.md) · [docs/tutorial.md](docs/tutorial.md) |
| Stages, artifacts, rubrics, executors, invariants | [ARCHITECTURE.md](ARCHITECTURE.md) |
| `extract` and quoted posts; the raw payloads (`reextract`, `payload-stats`); `refetch-truncated` | [extract](ARCHITECTURE.md#extract) · [payloads](ARCHITECTURE.md#payloads) · [refetch-truncated](ARCHITECTURE.md#refetch-truncated) |
| `fetch`, `validate_body`, `fetch --retry-failed` / `--revalidate` | [fetch](ARCHITECTURE.md#fetch) · [retry-failed and revalidate](ARCHITECTURE.md#fetch-retry-failed-and-revalidate) |
| X Articles: the `blocks` body, structured fetch, inline images and videos, blogpost render | [invariant 12](ARCHITECTURE.md#invariants) · [fetch](ARCHITECTURE.md#fetch) · [media](ARCHITECTURE.md#media) · [generate](ARCHITECTURE.md#generate) |
| `media` → `describe`; `refresh-quoted`, `refresh-media`, `download-videos` | [media](ARCHITECTURE.md#media) · [describe](ARCHITECTURE.md#describe) · [refresh-quoted](ARCHITECTURE.md#refresh-quoted) · [refresh-media](ARCHITECTURE.md#refresh-media) · [download-videos](ARCHITECTURE.md#download-videos) |
| Video: `list-videos` / `fetch-video`, `digest-video` (and `--frames`), `video-digest`, `redescribe-frames` | [list-videos / fetch-video](ARCHITECTURE.md#list-videos--fetch-video) · [digest-video](ARCHITECTURE.md#digest-video) · [video-digest](ARCHITECTURE.md#video-digest) · [redescribe-frames](ARCHITECTURE.md#redescribe-frames) · [docs/digest-video.md](docs/digest-video.md) |
| `vocab`, `enrich`, `topics`, `generate` (video digest section, verification badge) | [vocab](ARCHITECTURE.md#vocab) · [enrich](ARCHITECTURE.md#enrich) · [topics](ARCHITECTURE.md#topics) · [generate](ARCHITECTURE.md#generate) |
| Rubrics, and the shared on-screen-text fragment | [Rubrics](ARCHITECTURE.md#rubrics-the-prompt-layer) |
| Evidence surfaces (`evidence.py`), `verify` (`--audit`, `--write-verdicts`), `verify-entities` | [evidence](ARCHITECTURE.md#evidence) · [verify](ARCHITECTURE.md#verify) · [verify-entities](ARCHITECTURE.md#verify-entities) |
| The knowledge layer: contract, provenance, identity, chunking | [The knowledge layer](ARCHITECTURE.md#the-knowledge-layer) |
| Something failing | [docs/troubleshooting.md](docs/troubleshooting.md) |

## Conventions
- TDD: every module has a `tests/test_*.py`. Run `uv run pytest -v`.
- The X GraphQL parser anchors on key names, not paths — X's private API drifts.
- Never commit personal data: `auth/storage_state.json`, `data/`, `config.toml`.
  All are gitignored.

## Git workflow
- `develop` is the integration branch: `feature-branch → PR → develop`. Branch
  from `develop` (never from `main`) and target every PR at `develop`.
- `develop → main` only via PR — never merge or push directly to `main`.

## Rules paid for in blood (2026-07-14: verification audit, then CI audit)

Fifteen PRs merged in one day (`gh pr list --state merged`, 2026-07-14), six agents.
Every rule below is here because we broke it and something shipped wrong while the suite
was green. They are ordered by how often they bit us. Apply them; do not admire them.

Rules 1–8 came out of the morning's audit of the data pipeline. Rules 9–13 came out of
the afternoon's audit of CI and branch protection, and each one was **measured against the
live GitHub API**, with a probe PR number as the receipt. Do not soften them.

### 1. A test that passes before you write the fix is not a test

Six times in one day, six different agents, the same defect: an assertion satisfied for
the wrong reason.

| The assertion | Why it passed anyway |
|---|---|
| `assert "NOT fetched" in source` | satisfied by the section **header** `[Links — content NOT fetched]`. The rule sentence it claimed to pin was unprotected — deleting it stayed green. |
| `assert "1 verdicts escritos" in output` | satisfied verbatim by `"0 de 1 verdicts escritos (1 omitidos: …)"`. The test for *one written* passed on *zero written*. |
| `assert stored_ids(tmp_path) == set()` | `tmp_path` was never passed to the function. It asserted that an empty directory is empty. |
| `assert evidence_text(i, t) == …evidence_surfaces(i, t)` | **both sides came from the same module.** It asserted `evidence.py` against itself — inside the PR written to end this class of test. |
| `assert "topic signal only" in payload` | already satisfied by the *links* rule; it said nothing about the *bookmark folder* it claimed to pin. |
| `assert checker.evidence_text is evidence.evidence_text` | once the checker delegated, the module attribute **is** the same object. The tautology reappears in disguise the moment you think you have killed it. |

**Do:** assert **where** a value lives and **which source** it came from — the label it
sits under, the shared constant it is *identical* to, the behaviour it produces through
the **public API**. Never that a string appears *somewhere*.

**And watch it go red first.** A green test before the fix exists is the only reliable
tell that it is testing nothing. If you cannot make it fail, you have not written a test.

### 2. A metric that cannot come out any other way is not a measurement

`"0 false flags on tweets under 200 chars"` — with a 265-char floor, nothing under 265
*can* be flagged. The number restated the constant. A disk-footprint table and a secrets
sweep were both computed on a fixture that contained **no tweets**. Three headline
numbers were retracted in one day.

**Do:** state the population you measured **on** and the way a different answer could
have come out. If neither exists, do not quote the number.

### 3. Nothing catches itself

Not one defect that mattered was found by CI, by review, or by the author re-reading.
Every one was found by **someone who did not write it, running it against real data**: an
attribution rule that let a false speaker through 8 judges out of 8; a thread served to
the judge as a fetched article; a cookie wall stored as evidence; a quoted post rendered
in the user's note as if the poster had written it; an entity checker with ~0% precision.

**Do:** judge ≠ party, and the judge must **execute**, not read. Run the thing against
the real store (read-only) before you claim it works.

**And the base case**, for when "what guards the guard?" threatens to regress forever: the
escape is **not another catcher**. It is making the **absence** of the catcher fail closed.
Where removing a guard blocks the merge, the regress terminates — see rule 11.

### 4. A green PR against a moving `develop` is not a green `develop`

One PR added a test calling `_source_text(item)`; another changed that signature. No
textual conflict. Both green on their own branches. **The merge was red** — nothing ever
ran the combination.

**Do:** before merging, run the suite on the **merge result**, not on your branch
(`git merge-tree --write-tree origin/develop HEAD` proves it merges; only running the
tests proves it works). And read a check's **reported conclusion** — never infer it from
the exit status of the command that printed it. A red check has already reached `develop`
that way.

**This is now mechanized** (PR #110 + branch protection). `quality.yml` runs on `push` to
`develop`/`main`, so the merge commit itself is finally tested — before this the repo had
**zero** `push` runs in its entire history, and `1209094`, the merge that broke `develop`,
carried `total_count: 0` check runs. Nobody had ever tested it. And `strict: true` forces
the merge ref to be recomputed against the current base *before* merging, so the stale
green cannot land in the first place. `push` is the detector; `strict` is the preventer.

**Caveat, and it is a sharp one:** the advice above — *read the reported conclusion* —
assumed the conclusion is the trustworthy surface. **Rule 10 is the case where it is not.**
A step that ran `exit 1` can report `"conclusion": "success"`. Read rule 9 before you trust
any conclusion field.

### 5. One definition, or five that silently diverge

"What counts as evidence" was written **five times by five hands** — the generator, the
generator rubrics, the judge's rubric, the judge, the checker. Every divergence produced
a confident wrong number with the suite green. Three people independently fixed the same
missing surface. The fix was ONE function (`evidence.evidence_surfaces`) plus a
cross-component test that fails when any consumer drifts.

**Do:** if two components must agree, bind them **in code** (one function, one constant,
one test asserting identity across all consumers) — never in prose, and never in two
lists that "should" match.

### 6. Repair the evidence, invalidate the derivative

Three PRs shipped a repair that fixed the source and left the summary, the digest and the
verdict standing — a full tweet sitting next to a summary of half of it, wearing a PASS
badge.

**Do:** any change to evidence must invalidate everything derived from it
(`contract_fingerprint` does this for verdicts: it hashes the output **and** the source
the judge read **and** the rubrics it applied).

And check the invalidation signal actually **reaches the population being repaired**. The
usual lever is `content.fetched_at` — and it cannot reach an item whose `content` is
`None`, because there is nothing to stamp.

Measured on the real store, each number with the definition it was measured under:

- **620** items are truncated (`looks_truncated`);
- of those, **526** have the truncated tweet as their **only** evidence — no article, no
  transcript, no thread. Repairing their text changes everything downstream;
- and **1,551 of 2,168 (72%)** carry **no `content` block at all**, so a `fetched_at`
  lever reaches none of them.

A repair whose staleness signal lives on the object it is *creating* reaches nobody. (The
figure originally quoted here — "432" — was stale: it predated a fix that moved detection
from 535 to 620, and nobody re-derived it when the population moved. Both numbers above
were re-derived from the store before being written down. That is the whole of rule 2.)

### 7. The cheapest verification layer is showing the user the evidence next to the claim

We built three judges, an independent auditor, a deterministic checker, and a contract to
bind them. Then we rendered the quoted post in the note — and a reader sees in two
seconds that a summary claiming *"his move to Anthropic"* sits above a quoted post that
never mentions Anthropic. No tokens, no threshold, no false positives.

**Do:** before building an instrument to detect a defect, ask whether **showing the
evidence** to the human would make the defect self-evident.

### 8. Once a review lands on a PR, its author owns the fix — unless reassigned out loud

Six times in one day, two agents built the same thing in parallel: the review fixes for one
PR (both versions complete, one thrown away), a `develop`-is-red hotfix (opened twice,
minutes apart), a whole feature re-implemented and opened as a duplicate PR **29 seconds
after the original merged** — and the same missing evidence surface was independently fixed
by three different people.

**Do:** when a review lands, the PR's author fixes it. Reassignment is stated explicitly,
to both agents. Before starting anything that someone else might already be doing, check
the actual remote state (`gh pr view`, `git ls-remote`) rather than the state you remember
— and note that a PR in **CONFLICTING** state runs **no checks at all**, so from the
outside it looks identical to a dead one.

### 9. Two instruments report the same event with opposite answers — name the surface

`gh run list` shows the **workflow run**. `gh pr view --json statusCheckRollup` shows the
job's **check run**. Branch protection reads the **check run**. A `continue-on-error` job
reports the workflow run as `success` and the check run as `FAILURE`. Read the wrong
instrument and you conclude the exact **opposite** of the truth.

This was the third costume the same bug wore in one day:

| The instrument | What it actually reports |
|---|---|
| `gh pr checks` | exits **0** on a failing check. Cost: a red merge (#96). |
| `git push … \| tail` | `$?` comes from **`tail`**, not from `push`. |
| a step that ran `exit 1` | `"conclusion": "success"` — see rule 10. |

**Do:** always name **which surface you read**. And **never trust a reported conclusion —
assert on the SOURCE.** We have a receipt that the conclusion field lies, so any guard that
verifies "every step concluded success" is defeated by the exact attack it exists to catch.
`tests/test_ci_workflow.py` and the `gate-integrity` suite parse `quality.yml` and
`check.sh` and assert on what they **say**.

### 10. The keyword does not have a column; the placement does

`continue-on-error: true` in two places, opposite outcomes:

| Where | Check run | Merge | |
|---|---|---|---|
| on the **job** (`jobs.quality.continue-on-error`) | `FAILURE` | `BLOCKED` | harmless *(probe #119)* |
| on the gate **step** | `SUCCESS`, `mergeStateStatus: CLEAN` | **merges** | **lethal** *(probes #121, #125)* |

On the step, the step that ran `exit 1` reports `success`, the job reports `success`, the
check reports `SUCCESS`. Two people each measured **one cell of a 2×2** and each claimed
the whole table.

**Do:** measure **the cell**, not the keyword.

### 11. Removing a guard fails closed; hollowing it out fails open

First, the mechanism that makes this the only distinction that matters: **a PR's CI runs
the HEAD version of the workflow, not the base's** *(measured: a probe step added on a
branch executed; and #112 — deleting `quality.yml` on a branch produced **zero** runs,
impossible if the base's workflow were used)*. **A PR that neuters the gate is judged by
the neutered gate. It absolves itself.** Which is why asserting on the workflow's
*triggers* alone was never going to be enough.

So every attack on the gate sorts into exactly two bins:

- **FAIL-CLOSED (safe)** — anything that stops the required check from **reporting**:
  deleting the workflow file *(probe #112: zero runs, `BLOCKED`)*; renaming the job (the
  required context `quality` never appears); `continue-on-error` on the job;
  `branches-ignore`; `paths:` under `pull_request`; an invalid `types:`. GitHub blocks the
  merge. **The change cannot ship a lie.** *(This bin assumes the PR targets `develop` or
  `main` — what blocks is branch protection's required check, and it exists only there. On
  any other base these cells flip into the column below: see rule 14.)*
- **FAIL-OPEN (lethal)** — anything that lets the check still say **PASS while testing
  less**: `continue-on-error` on the gate step; `checkout` with an explicit `ref:` *(probe
  #124 — the gate ran green **on the wrong tree**; the test count gave it away, `1088
  passed` where `develop`'s suite ran 1604 that day)*; gutted `steps:`; `COVERAGE_MIN=0`;
  `pytest --ignore=…`; demoting `Tests` to warn-only.

**Do: guard what can be hollowed out — what can be removed already guards itself.** This is
also the base case that terminates rule 3's regress: where a guard's *absence* fails closed,
you do not need a catcher for the catcher.

### 12. `required_approving_review_count` must stay **0**. Never set it to 1

`VGonPa` is the repo's **only collaborator** and authored all 76 PRs. GitHub forbids a PR
author from approving their own PR, and `can_approve_pull_request_reviews` is `false` for
`github-actions[bot]`. **No identity in this repository can satisfy a required approval.**

With `enforce_admins: true`, setting it to 1 makes **every PR permanently unmergeable, with
no bypass — including the PR that would undo it.** A reviewer proposed this today as its top
recommendation. It was thirty seconds from bricking the repo.

**Do:** leave it at 0. This is written down so the next agent does not helpfully re-propose it.

### 13. The honest residue — state it as social, do not dress it as mechanical

**Two lines in `.github/workflows/quality.yml` can make the gate lie** — a
`continue-on-error` on the gate step, or a pinned `checkout ref:` — and **nothing mechanical
prevents it on this repo.** Every escape was checked against the live API and is unavailable:
merge queue is **org-only**; the required-workflows ruleset is **GHEC/GHES-only**;
`file_path_restriction` rulesets are **Enterprise-only**; required approvals are impossible
(rule 12).

The one mechanism that would work — `pull_request_target`, whose definition comes from the
**base** branch, which the PR head therefore cannot edit — was **deliberately declined**: it
runs with the base's secrets and a write token, and on a **public** repo a single careless
future edit that makes it touch head code hands the repository to any stranger who opens a
fork PR. That is a permanent RCE surface bought to defend against an adversary who does not
exist.

**The guards catch mistakes, not attacks. The only backstop against an attack is that
someone reads the diff of `.github/`.** Say so plainly; do not claim the gate is airtight.

### 14. A PR onto a feature branch runs no gate at all — and GitHub calls it `CLEAN`

Rule 11 sorts every attack on the gate into two bins. **That table is conditional on
something it never says: the PR targets `develop` or `main`.** What blocks a merge is
branch protection's *required check*, and branch protection lives only on those two.

`quality.yml`'s trigger is `pull_request: branches: [develop, main]`. Point a PR at a
feature branch — a stacked PR, most obviously — and the workflow fires **zero** times.
No check is produced, so no check is missing, so nothing blocks.

Measured on **#151**, stacked on #149's branch while #149 was still open:

| Surface read | What it said |
|---|---|
| `gh run list --branch feat/90-pr2-redescribe-frames` | **no runs at all** |
| `gh pr view 151 --json statusCheckRollup` | `[]` |
| `gh pr view 151 --json mergeable,mergeStateStatus` | `MERGEABLE`, **`CLEAN`** |

`CLEAN` does not distinguish *everything required passed* from *nothing was required*.
That is rule 9 one level up: the instrument does not lie about a check, it lies **by the
absence of checks**. Retargeting the same PR (`gh pr edit 151 --base develop`) started the
gate — a `pull_request` run was in progress 30 s later — and it passed. No code changed,
only the base.

**Do: a stacked branch is a work container; a PR is a gate. Do not open them together.**
Let the work proceed on the stacked branch, and open the PR only once its base has landed
in `develop`. And read `required_status_checks.contexts` on the TARGET branch before
trusting any merge state.

**And check why you are stacking at all.** #151 was stacked because #149 sat merge-ready
and unmerged — a permission boundary that was never there. `develop` is the *integration*
branch: merging into it is the ordinary end of the loop, not a release. The cheapest fix
for a stacked PR is usually to merge the one below it.

### 15. A big initiative ships as child PRs onto a PROTECTED umbrella — never as one PR

A plan is a unit of product. **A PR is a unit of review**, and the two are not the same size.
Measured, 2026-09-02: Plan 02 of the knowledge initiative was implemented as one branch — **78
commits, 43 files, +16,656/−902** — and took **nine review rounds and 615,938 bytes of review
prose** (**9.0×** Plan 01, which merged) to converge. Round **8**, on a tree with `check.sh`
green and **98%** coverage in `knowledge/`, still surfaced **two NEW HIGHs**, one of them a
`build --force` that destroyed a good index and sealed it with exit 0. Nothing was wrong with the
code — `b61e04b` ends green at 2,549 tests. What was wrong is that **a review of 16,656 lines does
not converge**: each pass picks a subset, and the subset it did not pick stays unlooked-at.

**So: `develop` → `umbrella` → child PRs, merged one at a time.** A child PR targets the umbrella,
never `develop`; children are sequential, so each is read against a tree containing its
predecessors; **each must be green on its own**, and if a boundary cannot produce a coherent state,
move the boundary and write down why. Only the umbrella opens a PR to `develop`, and its gate looks
for integration regressions, not unit defects.

**The umbrella gets a REAL gate — this is not optional, and not a social rule.** Rule 14 says a PR
onto a feature branch runs nothing and GitHub still reports `CLEAN`. Close it, in this order:

1. `.github/workflows/quality.yml` lists `"VGonPa/umbrella-*"` under **both** `push` and
   `pull_request` (guarded by `tests/test_ci_workflow.py`);
2. **classic branch protection on the umbrella's EXACT name**, created **before the first child**
   and removed only after the final merge: `quality` required · `strict: true` ·
   `enforce_admins: true` · **approvals 0** (rule 12 — one collaborator, so a 1 blocks everything
   forever). Measured available here: the repo is **public** with `admin: true`, and `develop`
   itself is a **non-default** branch carrying exactly this. `rulesets` is `[]` — rule 13's
   GHEC-only finding is about rulesets, not about classic protection;
3. **verify by API and read the response**, not the exit code. If it cannot be created, that is a
   **BLOCKER**, not a compensation to remember.

**`strict` on the umbrella does NOT tell you `develop` moved** — it only compares a child to its
umbrella base. So **checkpoint `origin/develop`'s SHA before every child**, and when it has moved,
integrate it with a **sync child PR**: never a direct push (skips the gate) and never a rebase
(rewrites merged children). When a monolith is being **redistributed** rather than written, freeze
its tip as a snapshot, branch the umbrella from the snapshot's **historical base**, port bytes with
`git checkout <snapshot> -- <paths>` instead of retyping them, and prove `tree == snapshot` **before**
the first sync — after a sync the only honest reference is the synthetic `merge-tree` of current
`develop` + snapshot.

Full procedure, PR matrix and roles:
`zz-support-files/docs/implementation-plans/2026-09-02-plan-entrega-atomica-umbrellas.md`
(and `AGENTS.md`, which carries the same topology for Codex).

### Branch protection: the live settings, and what each one cost

Live on **`develop` and `main`** (`gh api repos/VGonPa/xbrain/branches/develop/protection`):

| Setting | What it prevents | What it costs |
|---|---|---|
| required check **`quality`** | a merge with no green gate | nothing |
| **PRs required** | pushing straight to `develop`/`main` | one PR per change |
| **`strict: true`** (up to date before merging) | the stale-green merge of rule 4 | a rebase when the base moves |
| **`enforce_admins: true`** | the owner waving his own change through | the owner waits like everyone else |
| no force pushes / no deletions | rewriting or vaporizing the integration branch | nothing |
| approvals **0** | — | **must stay 0 — see rule 12** |

`strict` was the contested one. The rationale for leaving it off (CI is slow, merges are
frequent, you would rebase all day) was **asserted, never measured** — and the measurement
pointed the other way: **CI median 73 s** *(measured: `[70,66,72,73,73,79,65,74,76,71]`)*
against a **median 7.6 min gap between merges**. The queue never forms.

And the coupling nobody guarded until today: **the required check's name comes from the job
id `quality`.** Rename the job and the required check **never appears** — every merge blocks,
forever. Change the branch-protection setting **first** if you ever must rename it. Guarded
by `tests/test_ci_workflow.py` (17 assertions); ruff's scope by `tests/test_quality_gate_scope.py`.
