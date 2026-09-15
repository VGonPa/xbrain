"""The retrieval evaluation harness (spec §8).

WHAT IT EVALUATES. Whether xbrain retrieves the relevant items and surfaces — NOT the quality
of a written answer, because xbrain does not write one (spec §0.2, §8.1). The unit of ranking
is the chunk; the unit of scoring is the OWNER (an item or a topic), because spec §5.4 groups
results by item so that ten adjacent windows of one transcript cannot occupy the top ten.

THREE RULES, and they are the reason this module exists rather than a script:

1. **Never one global figure.** Spec §8.4 asks for strategy x stratum x provenance. One
   corpus-wide recall averages a 9-case stratum against a 2-case one and yields a number no
   decision can be made from — and hides the case where a layer helps exactly one stratum.
   The report has no top-level metric key to reach for.

2. **No coverage is NOT zero, at BOTH levels.** Spec §8.6.8: *failures and skips are
   published; zeros are never fabricated by mixing in unmeasured cases*. `expansion` has no
   mechanism until Plan 04; `thread` and `user_note` have zero instances in the corpus.
   Reporting them at 0.0 would claim the retriever failed at something nobody asked, and the
   figure would sit in a table looking exactly like a measurement.

   The BUCKET level was guarded from the start. The METRIC level was not (B1), and that is
   the level the defect actually lived at: a case that names no surface is not a case that
   failed `surface_recall`, and a case whose ground truth is surfaces only has a 0/0
   `recall@k`. Both returned a hard 0.0 and both entered the stratum mean. So a metric is
   `None` on the case when the case could not measure it, the bucket mean is taken over the
   members that carry it, and the bucket reports `NO_COVERAGE` for a metric none of them do.
   `measured` states each mean's denominator, because a mean over a silently smaller
   population is the same defect in its next costume (rule 2).

3. **Report-only.** Never writes `items.json`, never snapshots — the precedent `verify` sets
   by default and `cv-guardrail` follows. Asserted by hashing the file across a run.

THRESHOLDS ARE NOT INVENTED HERE. `threshold` defaults to `None`, which means "report, do not
judge". Spec §8.6 is explicit that observed values and merge thresholds are fixed AFTER the
baseline is run, not guessed in advance; a default number here would be exactly the metric
that cannot come out any other way (CLAUDE.md rule 2). A caller that wants a gate passes one.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from xbrain.knowledge.chunking import ChunkerParams, DEFAULT_CHUNKER_PARAMS, chunk_surfaces
from xbrain.knowledge.contracts import SearchFilters, Strategy, resolve_strategy
from xbrain.knowledge.goldenset import STRATA, GoldenCase, GoldenScenario
from xbrain.knowledge.index_schema import IndexError_, open_memory_index
from xbrain.knowledge.lexical import LexicalHit, LexicalIndex, distinct_owners
from xbrain.knowledge.models import KnowledgeChunk

if TYPE_CHECKING:  # pragma: no cover - annotations only; the vector path imports at call time
    from xbrain.knowledge.index_build import VectorBuild
    from xbrain.knowledge.index_store import OpenIndex
    from xbrain.knowledge.vector_index import VectorPlane
from xbrain.knowledge.surfaces import (
    article_block_texts,
    item_surfaces,
    item_topics,
    topic_surfaces,
)
from xbrain.models import Item, Topic, TopicPage
from xbrain.store import load_store, load_topic_pages

# The marker carried INSTEAD of a number by a bucket with no cases, and by a METRIC that no
# case in a bucket could measure (B1). A sentinel rather than a `None` or a zero, so it
# survives JSON and renders as words in the markdown. `None` is what a single CASE carries;
# this is what the AGGREGATE carries, because the aggregate is what gets published and read.
NO_COVERAGE = {"coverage": "sin cobertura"}

# Surfaces the emitter supports for which the corpus holds NO data (measured 2026-08-31), so
# the evaluation cannot have cases and does not invent any (spec §8.6.8). Declared, so their
# absence never reads as an oversight.
SURFACES_WITHOUT_DATA: tuple[str, ...] = ("thread", "user_note")

DEFAULT_KS: tuple[int, ...] = (1, 5, 10, 20)

# What `k` counts in each metric family (m6; PR #186, Codex F2). Published on every report and
# every comparison, because a denominator read in the wrong unit is a wrong number: `recall@10`
# is the first ten deduplicated OWNERS, `surface_recall@10` the first ten CHUNKS.
METRIC_UNITS: dict[str, str] = {
    "recall": "owners",
    "precision": "owners",
    "mrr": "owners",
    "ndcg": "owners",
    "surface_recall": "chunks",
}

# THE DEPTH IS COUNTED IN OWNERS (U-6, round 07 — gate Codex F5). Every metric here is defined
# over deduplicated owners (spec §5.4 groups by item), and `evaluate` asked the index for
# `max(limit, max(ks))` CHUNKS: seven windows of one transcript at the top meant ten chunks
# held two owners, so the real case F2 scored `recall@10 = 2/3` with `--k 10` alone and 1.0
# with `k=20` requested beside it (the default) — the `filtros` stratum moved 0.8333 -> 1.0
# on which neighbouring figure was asked for. Now the ranking is materialised until it holds
# the owners requested: `OWNER_CHUNK_MULTIPLIER` chunks per owner first, doubling while the
# result set is full and short of owners, up to `MAX_CHUNK_DEPTH` (an exhausted ranking is
# declared on the case, never read as «the owner was not there»). A prefix of a deeper FTS
# ranking is the shallower ranking — one total order, `chunk_id` tie-break — so `recall@k`
# for any k at or below the depth is one number, whatever else was asked for.
#
# THE LOOP LIVES IN `lexical` (M-4, round 08) and this module calls it, so the search service
# decides `truncated` over the SAME window this harness scores — ONE function, which is the
# rule-5 binding. That binding was ABSENT from this tree until this child: `_search` called
# `LexicalIndex.search(q, limit)` and `lexical.search_owners` had a single consumer, and the
# two therefore scored different retrievals under the same `k`. Restoring it is what makes a
# `recall@k` published here a statement about what `search` returns.
#
# AND THE TWO CONSTANTS ARE NOT RE-EXPORTED HERE. They were, as `evaluation.MAX_CHUNK_DEPTH`
# and `evaluation.OWNER_CHUNK_MULTIPLIER`, with no reader anywhere in `src/` or `tests/` — a
# SECOND BINDING of a constant, sitting inside the comment that argues for one, and a stale
# one at that: `tests/test_knowledge_evaluation.py` monkeypatches `lexical.MAX_CHUNK_DEPTH`,
# which an alias bound at import time never sees. `vulture` at confidence 80 does not flag a
# module-level name, so nothing was going to find them. Read them off `lexical`.

# Which of spec §7.2's eight filters each strategy can actually push into the backend.
#
# THIS TABLE IS THE DIFFERENCE BETWEEN A ZERO AND A GAP, and Plan 02 is what closed the gap.
#
# Under Plan 01 the baseline held only chunks and their surface metadata: no date, no author,
# no source, no content-kind column. Scoring a case whose filter nobody applied produced
# `filtros: recall@10 = 0.0` in this harness's first real-corpus run — a number that reads as
# "retrieval failed at filtering" when the truth was that the instrument did not exist yet.
# A fabricated zero, and precisely what spec §8.6.8 forbids, so those cases were reported
# UNMEASURED instead.
#
# The persisted schema has all eight columns and the harness now builds through the SAME
# writer as `index build`, so the set is `SearchFilters.model_fields` — derived from the
# frozen contract rather than written out again, which means a ninth filter added to the
# contract shows up here without anybody remembering to.
#
# WHAT THIS CHANGES IN THE PUBLISHED NUMBERS, said out loud: the two `filtros` cases of
# `eval/golden-set.yaml` (`source`+dates, `content_kinds`+dates) move from UNMEASURED to
# scored. Measured across the 23 versioned cases, the only filters any of them declares are
# `content_kinds` (1), `created_from` (2), `created_to` (2) and `source` (1) — so nothing else
# in this table moves a published figure today.
#
# `has_surfaces` IS THE ONE TO READ CAREFULLY, because the obvious reading of the repair is
# wrong in the reassuring direction. It was never a chunk-level restriction that needed
# correcting: `lexical._item_clauses` has always answered it with `EXISTS (SELECT 1 FROM
# surfaces …)`, exactly the contract's *the item HAS this surface*, and this child changed no
# executable line of `lexical.py`. What it was is worse than a no-op — it was a FABRICATED
# ZERO wearing the «supported» label, in the module that exists to keep a zero and a gap
# apart. The harness's old two-walk `build_index` wrote chunks and NO metadata, so its
# `surfaces` table held 0 rows; `has_surfaces` sat inside `SUPPORTED_FILTERS`, so a case
# declaring it would have been SCORED rather than reported unmeasured, against an `EXISTS`
# guaranteed to match nothing. Measured on `tests/fixtures/knowledge_corpus.json`, base
# against head: `surfaces` rows 0 -> 43, and `search("the", 50, has_surfaces=('post',))`
# 0 chunks -> 24. Nothing PUBLISHED moves, because no case declares it — but what made that
# safe was «no case reaches the filter», never «the filter was equivalent».
#
# THE VECTOR PAIR PUSHES NONE, AND THAT IS PLAN 03.7's HALF OF THE RULE. The plane has no filter
# columns and a filter applied after scoring is not a filter, which is why `search` refuses to
# run the vector channel under one (`search_service.VECTOR_FILTERS_UNSUPPORTED`). Here the
# same fact makes a filtered case UNMEASURED under `vector` and `hybrid` — never scored, never
# 0.0 — so `filtros` stays a lexical measurement and says so, instead of the bake-off quoting a
# number for filtering that no vector channel performed. Measured on the golden set: F1 and F2
# are the two cases this removes from the vector reports, and the only two.
#
# `hybrid_graph` runs the SAME channels as `hybrid` in `search` (Plan 04 §3.1,
# `search_service._VECTOR_STRATEGIES`) and the vector plane has no filter columns, so it pushes
# none either: declaring anything else would SCORE a filtered case instead of reporting it
# unmeasured. The `.get(strategy, frozenset())` fallback already gave that answer — the entry
# makes the declaration honest, not accidental. It is declared for what `search` runs: the
# HARNESS scores `hybrid_graph` under no model (`EMBEDDING_MODEL_STRATEGIES`), so a run asking
# for it executes `lexical`, and this entry is not consulted until the harness runs the graph.
SUPPORTED_FILTERS: dict[str, frozenset[str]] = {
    "lexical": frozenset(SearchFilters.model_fields),
    "vector": frozenset(),
    "hybrid": frozenset(),
    "hybrid_graph": frozenset(),
}


def unsupported_filters(filters: Any, strategy: str) -> tuple[str, ...]:
    """The filters this case declares that `strategy` cannot apply, in declaration order.

    A filter left at its default is not "declared", so a case with `filters: {}` is always
    measurable. Only a filter the case actually set, and the backend cannot honour, makes it
    unmeasurable.
    """
    supported = SUPPORTED_FILTERS.get(strategy, frozenset())
    declared = tuple(
        name
        for name, field_info in type(filters).model_fields.items()
        if getattr(filters, name) not in (None, (), field_info.default)
    )
    return tuple(name for name in declared if name not in supported)


@dataclass(frozen=True)
class Corpus:
    """Everything the harness needs to build an index, and where it came from.

    `source` is carried because CLAUDE.md rule 2 asks for the population a number was
    measured ON: a recall figure with no corpus beside it cannot be compared with the next
    run, and is exactly the kind of number that gets quoted after the corpus has moved.
    """

    items: dict[str, Item]
    vocab: list[Topic]
    topic_pages: dict[str, TopicPage]
    source: str


@dataclass(frozen=True)
class IndexStats:
    """Coverage of the indexed corpus (spec §8.4, last bullet).

    An index that quietly dropped every article would still score well on the post-only
    cases; only this line says why the rest went missing.
    """

    items: int
    topics: int
    surfaces: int
    chunks: int  # what the chunker EMITTED
    # Emitted but REFUSED by the index — a blank body, or a `chunk_id` already held. It was
    # called `empty_surfaces` and counted neither empty things nor surfaces: hardcoded to 0
    # in `corpus_chunks`, recomputed as `chunks - indexed` in `build_index`. Two different
    # wrong answers under one name, the discordance B3 renamed `stale_chunks_excluded` for.
    # Counting empty SURFACES honestly would be a constant anyway — the emitters drop a
    # blank surface at `_blank` — so the number could not come out any other way (rule 2).
    chunks_not_indexed: int


@dataclass(frozen=True)
class CaseResult:
    """One case's outcome, with the ranking that produced it.

    The retrieved owners are kept so a reader can see WHY a case scored what it did, rather
    than being handed a number to trust.
    """

    id: str
    provenance: str
    strata: tuple[str, ...]
    retrieved: tuple[str, ...]
    # `None` = NOT MEASURED for this case, never "scored zero" (B1). A case that names no
    # surface did not fail surface recall; a case with no relevant OWNER has a 0/0 recall.
    metrics: dict[str, float | None]
    # The retriever returned NO CHUNK AT ALL (M3). A 0.0 recall has two causes — the right
    # item ranked below k, or nothing matched — and one number cannot say which. On the real
    # corpus the second was the cause in 18 of 21 cases while the report read every one of
    # them as the first, which is how a query-construction defect got published as a semantic
    # result. Carried per case and counted per bucket so the two are never confused again.
    no_results: bool = False
    # The ranking hit `MAX_CHUNK_DEPTH` before holding the owners asked for (U-6): the
    # owner list is SHORT for a reason that is not the retriever's. Declared, never silent.
    depth_exhausted: bool = False


@dataclass(frozen=True)
class EvaluationReport:
    # THE STRATEGY THAT RAN, never the one that was asked for (F-2). `xbrain eval --strategy
    # vector` published a report headed `vector`, with 21 cases and `recall@10 = 0.8099`,
    # produced entirely by the lexical retriever — a metric whose label does not describe its
    # instrument, which is rule 2 and spec §8.6.8 in one line. `requested_strategy` keeps the
    # question that was asked, and `degraded` says why the answer came from somewhere else.
    strategy: str
    corpus: dict[str, Any]
    cases: tuple[CaseResult, ...]
    by_stratum: dict[str, Any]
    by_provenance: dict[str, Any]
    latency: dict[str, float]
    without_coverage: dict[str, list[str]]
    scenarios: tuple[GoldenScenario, ...] = ()
    unmeasured: tuple[dict[str, Any], ...] = ()
    threshold: float | None = None
    failures: tuple[str, ...] = ()
    index_stats: IndexStats | None = None
    requested_strategy: str = ""
    degraded: tuple[str, ...] = ()
    # The depth every case ran at, in OWNERS (U-6): a figure travels with the depth that
    # produced it, or it is a figure that cannot come out any other way (rule 2).
    limit: int = 0
    # WHAT PRODUCED THE VECTORS, and what indexing them cost (Plan 03 §3.2, §3.5). `None` on a
    # lexical report, never an empty block: a report that ran no model has no model to name,
    # and `{}` would read as a model whose every property went unrecorded.
    embeddings: dict[str, Any] | None = None
    indexing: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.requested_strategy:
            object.__setattr__(self, "requested_strategy", self.strategy)

    @property
    def passed(self) -> bool:
        """True when no bucket fell below the threshold — vacuously true with no threshold."""
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "requested_strategy": self.requested_strategy,
            "degraded": list(self.degraded),
            "embeddings": self.embeddings,
            "indexing": self.indexing,
            "limit": self.limit,
            "metric_units": dict(METRIC_UNITS),
            "corpus": self.corpus,
            "threshold": self.threshold,
            "passed": self.passed,
            "failures": list(self.failures),
            "by_stratum": self.by_stratum,
            "by_provenance": self.by_provenance,
            "latency": self.latency,
            "without_coverage": self.without_coverage,
            "unmeasured": [dict(entry) for entry in self.unmeasured],
            "cases": [
                {
                    "id": case.id,
                    "provenance": case.provenance,
                    "strata": list(case.strata),
                    "retrieved": list(case.retrieved),
                    "metrics": case.metrics,
                    "no_results": case.no_results,
                    "depth_exhausted": case.depth_exhausted,
                }
                for case in self.cases
            ],
            "scenarios": [
                {"id": s.id, "question": s.question, "provenance": s.provenance, "reason": s.reason}
                for s in self.scenarios
            ],
        }


def load_corpus(path: Path) -> Corpus:
    """Load a FIXTURE corpus — `{items, vocab, topics}` in one JSON file.

    Separate from `load_corpus_from_store` on purpose: the fixture is what CI can run against
    (there is no `data/` there), and the store is what the local run measures. Keeping the two
    entry points distinct is the same separation the golden-set loader makes, for the same
    reason — one of them must work without a corpus.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Corpus(
        items={k: Item.model_validate(v) for k, v in raw["items"].items()},
        vocab=[Topic.model_validate(v) for v in (raw.get("vocab") or {}).values()],
        topic_pages={k: TopicPage.model_validate(v) for k, v in (raw.get("topics") or {}).items()},
        source=str(path),
    )


def load_corpus_from_store(items_path: Path, vocab: list[Topic], topics_path: Path) -> Corpus:
    """Load the REAL corpus, read-only. Never called from a test (no `data/` in CI)."""
    return Corpus(
        items=load_store(items_path),
        vocab=vocab,
        topic_pages=load_topic_pages(topics_path),
        source=str(items_path),
    )


def corpus_chunks(
    corpus: Corpus, *, params: ChunkerParams = DEFAULT_CHUNKER_PARAMS
) -> tuple[list[KnowledgeChunk], int]:
    """Every chunk of every item and topic surface, and how many SURFACES were walked.

    NO PRODUCTION CALLER — A TEST-SIDE ORACLE, AND SAYING SO IS THE POINT. `build_index` used
    to call this and now drives `index_build.write_item`, the same writer `xbrain index build`
    uses, so nothing under `src/` reaches this function any more. It survives as the
    INDEPENDENT second computation `test_the_stat_for_refused_chunks_is_named_for_what_it_counts`
    checks the writer's counter arithmetic against: `stats.surfaces` and `stats.chunks` are
    tallies the writer increments, and a tally is worth nothing without something that counted
    the same corpus another way.

    WHAT IT IS NOT is the walk any guard should assert a chunking PROPERTY through. One did —
    the article block-boundary regression of m8 — and when `build_index` moved, that guard
    stayed here and stopped protecting anything: measured at `6b368e9`, deleting
    `blocks_by_surface_id=` from `index_build.write_item` left the whole suite green while
    deleting it here reddened the guard. The guard now asserts on the rows the writer
    persisted (`tests/test_knowledge_chunking.py::
    test_the_index_writer_chunks_an_article_on_its_block_boundaries`). A property of the
    chunking belongs on the walk that ships; only a COUNT belongs here, and only as the
    other side of a comparison.

    Returns the surface count rather than an `IndexStats`, because it has not indexed
    anything: assembling the stats here would force a `chunks_not_indexed` that could only
    ever be 0 — a fabricated constant in the one module whose job is not to fabricate any.
    `build_index` owns the stats, because `build_index` is what does the refusing.
    """
    chunks: list[KnowledgeChunk] = []
    surfaces = 0
    for item in corpus.items.values():
        emitted = item_surfaces(item)
        surfaces += len(emitted)
        chunks += list(
            chunk_surfaces(
                emitted,
                params=params,
                topics=item_topics(item),
                url=item.url,
                blocks_by_surface_id=article_block_texts(item),
            )
        )
    for topic in corpus.vocab:
        emitted = topic_surfaces(topic, corpus.topic_pages.get(topic.slug))
        surfaces += len(emitted)
        chunks += list(chunk_surfaces(emitted, params=params))
    return chunks, surfaces


def build_index(
    corpus: Corpus, *, params: ChunkerParams = DEFAULT_CHUNKER_PARAMS
) -> tuple[LexicalIndex, IndexStats]:
    """The lexical baseline over a whole corpus, plus what it covered.

    BUILT THROUGH `index_build`'s WRITER, on `sqlite3(":memory:")`. Not a second walk: the
    harness must measure the instrument `search` actually queries, and two walks that "should"
    emit the same corpus are the divergence CLAUDE.md rule 5 is about — the one that would
    have gone wrong first is the metadata, which is what makes six of the eight filters
    answerable at all.

    `chunks` is what the chunker EMITTED and `chunks_not_indexed` is the difference the index
    refused, so the two together say whether coverage is complete — one number that silently
    meant "indexed" could not.
    """
    from xbrain.knowledge.index_build import IndexOptions, WriteCounters, topic_membership
    from xbrain.knowledge.index_build import write_item as write_item_rows
    from xbrain.knowledge.index_build import write_topic as write_topic_rows

    index = LexicalIndex(open_memory_index())
    counters = WriteCounters()
    options = IndexOptions(params=params)
    for item_id in sorted(corpus.items):
        write_item_rows(index, corpus.items[item_id], corpus.vocab, counters, options=options)
    for topic in sorted(corpus.vocab, key=lambda t: t.slug):
        primary, secondary = topic_membership(corpus.items, topic.slug)
        write_topic_rows(
            index,
            topic,
            corpus.topic_pages.get(topic.slug),
            primary,
            secondary,
            counters,
            options=options,
        )
    index.connection.commit()
    return index, IndexStats(
        items=len(corpus.items),
        topics=len(corpus.vocab),
        surfaces=counters.surfaces,
        chunks=counters.chunks + counters.empty_text,
        chunks_not_indexed=counters.empty_text,
    )


def evaluate(
    cases: Sequence[GoldenCase],
    corpus: Corpus,
    *,
    strategy: str = "lexical",
    ks: tuple[int, ...] = DEFAULT_KS,
    threshold: float | None = None,
    scenarios: Sequence[GoldenScenario] = (),
    params: ChunkerParams = DEFAULT_CHUNKER_PARAMS,
    limit: int | None = None,
    vectors: VectorEvaluation | None = None,
) -> EvaluationReport:
    """Score every case and aggregate by stratum and by provenance.

    `vectors` is what makes `vector` and `hybrid` RUN (Plan 03.7): the model asked for, the
    embedder that serves it, and where the evaluation's own persisted index lives. Without it
    those two strategies resolve as they always did — degraded to `lexical` and declared — and
    with it they are scored over the SAME chunks the lexical baseline scores, through the
    search service's own fused window (`_retriever`). Passing it with `lexical` is refused: a
    model named on a report none of whose numbers it produced is F-2 in the other direction.

    `threshold`, when given, turns the report into a gate: any bucket whose `recall@max(ks)`
    falls below it becomes a named failure. When absent, the report only reports — spec §8.6
    fixes thresholds after the baseline, and a default here would be a number that could not
    come out any other way.

    `strategy` is what was ASKED FOR. What the report is labelled with is what RAN: a
    strategy declared in the frozen `Strategy` literal but with no backend degrades to
    `lexical` and the degradation is published beside the numbers (spec §9.3 — *lexical sigue
    operativo y el response declara estrategia degradada*), while a strategy that is in no
    contract at all raises. Echoing the request into the heading is how a lexical baseline got
    published as a vector measurement (F-2).

    `limit` is how many OWNERS the ranking is materialised to (U-6). It defaults to
    `max(ks)`, because asking for fewer results than the largest k being reported would
    make that k's recall a measurement of the LIMIT rather than of the retriever — a number
    that cannot come out any other way. A caller may raise it to see whether a miss is a
    ranking problem or an absence. It is published on the report.
    """
    executed, degraded = _resolve_strategy(strategy, vectors)
    depth = max(limit or 0, max(ks))
    index, stats = build_index(corpus, params=params)
    unmeasured: list[dict[str, Any]] = []
    timings: dict[str, list[float]] = {"total": [], "embedding": []}
    run: _VectorRun | None = None
    try:
        if vectors is not None:
            run = _open_vector_run(vectors, params=params)
        results = _score_cases(
            cases, executed, ks, depth, _retriever(index, run, executed, depth), unmeasured, timings
        )
    finally:
        # The vector run holds a read-only SQLite handle and a memory map of the matrix, and a
        # fusion sweep calls into the same plane once per cell: released here, with the
        # `:memory:` index, rather than whenever the locals happen to be reclaimed.
        if run is not None:
            run.close()
        # The `:memory:` index lives exactly as long as the scoring. `sweep_chunker` calls
        # this once per COMBINATION, so a twelve-cell sweep opened twelve handles and closed
        # none of them explicitly — release left to whenever the local was reclaimed.
        # Measured before this line, by capturing the index `build_index` returned and
        # querying it after `evaluate` had returned: `SELECT 1` succeeded, i.e. STILL OPEN.
        # (The `ResourceWarning: unclosed database` the snapshot's own note records did NOT
        # reproduce on this tree, so the assertion is on the connection state, which is the
        # surface that answers the question anyway — rule 9.)
        index.connection.close()

    by_stratum = _aggregate(results, STRATA, lambda case: case.strata)
    by_provenance = _aggregate(results, {"real", "construido"}, lambda case: (case.provenance,))
    failures = _failures(by_stratum, by_provenance, threshold, ks)
    return EvaluationReport(
        strategy=executed,
        requested_strategy=strategy,
        degraded=degraded,
        corpus={
            "source": corpus.source,
            "items": len(corpus.items),
            "topics": len(corpus.vocab),
            "surfaces": stats.surfaces,
            "chunks": stats.chunks,
            "chunks_not_indexed": stats.chunks_not_indexed,
        },
        cases=tuple(results),
        by_stratum=by_stratum,
        by_provenance=by_provenance,
        latency=_latency(timings),
        embeddings=run.embeddings if run is not None else None,
        indexing=run.indexing if run is not None else None,
        without_coverage={
            "strata": sorted(k for k, v in by_stratum.items() if v == NO_COVERAGE),
            "surfaces": list(SURFACES_WITHOUT_DATA),
        },
        unmeasured=tuple(unmeasured),
        scenarios=tuple(scenarios),
        threshold=threshold,
        failures=failures,
        index_stats=stats,
        limit=depth,
    )


def _search(
    index: LexicalIndex, case: GoldenCase, *, owners: int
) -> tuple[tuple[LexicalHit, ...], bool]:
    """Run one case's query, applying its filters BEFORE scoring, deep enough to hold
    `owners` distinct owners (spec §5.3, U-6). Returns `(hits, depth_exhausted)`.

    The filters a case declares are part of the case (spec §8.1) — v1 kept windows under a
    key no loader read, so a temporal case silently became an untemporal one and its result
    was reported as though the window had been applied.

    ALL EIGHT are passed now, unchanged, because the persisted schema can push all eight into
    `WHERE`. Passing `case.filters` whole rather than reconstructing a subset is what keeps
    `SUPPORTED_FILTERS` an honest declaration instead of a list that has to be kept in step
    with a second one here.

    The window is `LexicalIndex.search_owners` — ONE loop for the harness and for the
    search service (M-4, round 08), so what this harness scores at depth N is the window the
    service pages at depth N: `OWNER_CHUNK_MULTIPLIER` chunks per owner, doubling while the
    result set came back full and short of owners, bounded by `MAX_CHUNK_DEPTH`; reaching
    the bound short of owners is declared on the case.
    """
    return index.search_owners(case.query, owners, filters=case.filters)


def _owner_key(owner_type: str, owner_id: str) -> str:
    return f"{owner_type}:{owner_id}"


def _score(
    case: GoldenCase,
    hits: Sequence[LexicalHit],
    ks: tuple[int, ...],
    *,
    depth: int,
    depth_exhausted: bool = False,
) -> CaseResult:
    """Recall/precision/MRR over OWNERS, plus surface recall (spec §8.4).

    Owners, not chunks: spec §5.4 groups by item, so a transcript matching in six windows is
    one retrieved item, not six. Deduplicated by FIRST occurrence, which preserves the rank
    the best chunk earned. `retrieved` is the owner ranking up to `depth` — the owners the
    case was materialised to (U-6), so a reader sees the population every k was cut from.
    """
    ranked: list[str] = []
    for hit in hits:
        key = _owner_key(hit.owner_type, hit.owner_id)
        if key not in ranked:
            ranked.append(key)
    relevant = {_owner_key("item", i) for i in case.relevant_items}
    relevant |= {_owner_key("topic", t) for t in case.relevant_topics}

    metrics: dict[str, float | None] = {}
    for k in ks:
        top = ranked[:k]
        found = len(relevant & set(top))
        # 0/0 IS NOT 0.0 (B1.b). `load_cases` accepts a case whose ground truth is
        # `relevant_surfaces` only, and `relevant` is then empty. Returning 0.0 reported a
        # perfect rank-1 answer as a total recall failure — the fabricated zero of spec
        # §8.6.8, one level below the empty bucket the sentinel already covers. Precision
        # goes with it: with no relevant owner its numerator is 0 by construction, so the
        # number would restate the empty set rather than measure the retriever (rule 2).
        metrics[f"recall@{k}"] = found / len(relevant) if relevant else None
        # `top` EMPTY IS 0/0, not a precision of zero (M3) — the last instance of B1, and
        # the one that reached the published table: 18 of 21 cases on the real corpus, 86 %
        # of the `precision` column. Its numerator is 0 by construction, so the number
        # restates the empty set instead of measuring the retriever, which is exactly the
        # argument 3006c0c used one line above for `recall` and `mrr`.
        #
        # `recall@k` deliberately does NOT follow: its denominator is the case's known
        # relevant set, so "none of the two items that exist came back" is a real
        # measurement of a real failure. Different denominators, different answers.
        precision = None if not top else found / len(top)
        metrics[f"precision@{k}"] = precision if relevant else None
        # Binary, and `None` on the same 0/0 `recall` is `None` on (`_ndcg`).
        metrics[f"ndcg@{k}"] = _ndcg(ranked, relevant, k) if relevant else None
        metrics[f"surface_recall@{k}"] = _surface_recall(case, hits, k)
        # MRR CUT AT k, over the SAME owner prefix `recall@k` reads (PR #186, Codex F1). The
        # bare `mrr` below walks the WHOLE ranking the retriever returned, and that ranking's
        # length is the retriever's window, not the report's depth: the lexical owner loop
        # stops past `depth` owners, the fused window holds up to 1,000 chunks per channel.
        # Measured on the published bake-off reports (depth 20): V1's lexical `mrr` is 1/51
        # and its hybrid one 1/56, S8's vector one 1/268 — so a «hybrid empeora exacto»
        # decided on 0.0196 → 0.0179 was decided on ranks no report publishes.
        # A prefix of the ranking is the same under any depth ≥ k (U-6), so `mrr@k` is one
        # number per strategy and comparable across them; `compare_reports` reads this one.
        metrics[f"mrr@{k}"] = _mrr(ranked[:k], relevant) if relevant else None
    # Window-dependent: comparable only between reports that share the retriever's window
    # (one strategy, one depth — the two sweeps). Never across strategies; see `mrr@k`.
    metrics["mrr"] = _mrr(ranked, relevant) if relevant else None
    return CaseResult(
        id=case.id,
        provenance=case.provenance,
        strata=case.strata,
        retrieved=tuple(ranked[:depth]),
        metrics=metrics,
        no_results=not hits,
        depth_exhausted=depth_exhausted,
    )


def _surface_recall(case: GoldenCase, hits: Sequence[LexicalHit], k: int) -> float | None:
    """How many of the case's named surfaces appear among the top-k retrieved CHUNKS.

    Spec §8.4 asks for this alongside item recall because returning the right item through
    the wrong surface is a different, usually worse, answer: the evidence a consumer would
    open is not the evidence the fact is in. Item recall alone scores that as a success.

    THE UNIT IS THE CHUNK, and `recall@k`'s unit is the deduplicated OWNER (m6). Under one
    label `k` they therefore count different things: the ranking is materialised to k
    OWNERS (U-6), so `recall@10` is formed from however many chunks ten distinct owners
    take, while `surface_recall@10` never sees past the tenth chunk.
    The chunk is the right unit here — the question is whether the EVIDENCE surfaced, and a
    surface that arrived as the 30th chunk did not surface — but the two columns are not
    comparable to each other, only to their own value in the next run.

    `None`, never 0.0, when the case names no surface (B1.a): that case did not fail this
    metric, it did not measure it, and a 0.0 entered the stratum mean looking exactly like a
    measurement. Measured on the real corpus (2026-08-31, 2,404 items, 23 cases): three
    cases name no surface, and they depressed `enterrado`'s published `surface_recall@10`
    from 0.1667 to 0.125.
    """
    if not case.relevant_surfaces:
        return None
    wanted = {(s.owner_type, s.owner_id, s.surface_type) for s in case.relevant_surfaces}
    seen = {(hit.owner_type, hit.owner_id, hit.surface_type) for hit in hits[:k]}
    return len(wanted & seen) / len(wanted)


def _mrr(ranked: Sequence[str], relevant: set[str]) -> float:
    for position, key in enumerate(ranked, start=1):
        if key in relevant:
            return 1.0 / position
    return 0.0


def _aggregate(
    results: Sequence[CaseResult],
    buckets: Iterable[str],
    key: Any,
) -> dict[str, Any]:
    """Mean of each metric per bucket — or `NO_COVERAGE`, for the bucket AND per metric.

    This is rule 2 of the module docstring made mechanical, at BOTH levels. The empty bucket
    was always covered; the metric inside a non-empty bucket was not (B1). A case that names
    no surface contributed a hard 0.0 to `surface_recall`, and a case whose ground truth is
    surfaces only contributed a 0/0 `recall`. Both averaged in as though they were
    measurements, which is precisely the mixing spec §8.6.8 forbids.

    So the mean is taken over the members that ACTUALLY CARRY the metric, and the metric
    gets the same sentinel as an empty bucket when none of them do.

    `measured` ships beside the means because averaging a subset silently moves the
    population: `cases: 8` next to a mean over 6 of them is a second fabricated number, of
    the shape rule 2 exists to stop. `cases` is the bucket's size; `measured[name]` is the
    denominator that metric's mean was actually divided by.
    """
    out: dict[str, Any] = {}
    for bucket in sorted(buckets):
        members = [r for r in results if bucket in key(r)]
        out[bucket] = _bucket_means(members) if members else NO_COVERAGE
    return out


def _metric_names(members: Sequence[CaseResult]) -> list[str]:
    """Every metric the SCORER produced, in the order it produced them.

    DERIVED from the results, not restated (m10). The previous version built the names from
    its own list of f-strings while claiming "one list, so the aggregate and the per-case
    scoring cannot disagree about what exists" — and `_score` never called it. There were two
    lists that happened to agree. One direction was guarded by accident; the other was not,
    so a metric added to `_score` — spec §8.4 already anticipates `nDCG` "when grades exist" —
    would be dropped from the aggregate, from `measured` and from the published table in
    silence, while the docstring went on asserting it could not be. That is rule 5 in the
    module that cites it: bind them in code, or they are two lists.

    Now there is genuinely one source, and it is the scoring: the union preserves insertion
    order, and `_score` inserts per k, so the report groups a k's metrics together rather
    than a metric's ks. A union rather than the first member's keys, because a member missing
    a key must not delete that column for the whole bucket.
    """
    names: dict[str, None] = {}
    for member in members:
        names.update(dict.fromkeys(member.metrics))
    return list(names)


def _bucket_means(members: Sequence[CaseResult]) -> dict[str, Any]:
    """One non-empty bucket's means, each over the members that CARRY that metric."""
    metrics: dict[str, Any] = {}
    measured: dict[str, int] = {}
    for name in _metric_names(members):
        values = [value for m in members if (value := m.metrics.get(name)) is not None]
        measured[name] = len(values)
        metrics[name] = round(sum(values) / len(values), 4) if values else NO_COVERAGE
    metrics["cases"] = len(members)
    metrics["measured"] = measured
    # Beside `measured`, and for the same reason (rule 2): `measured` says how many members
    # carried a metric, this says how many were handed nothing to rank. A bucket reading 0.0
    # with `no_results == cases` is not a retriever that ranked badly.
    metrics["no_results"] = sum(1 for m in members if m.no_results)
    return metrics


def _failures(
    by_stratum: dict[str, Any],
    by_provenance: dict[str, Any],
    threshold: float | None,
    ks: tuple[int, ...],
) -> tuple[str, ...]:
    """Buckets below the threshold, each NAMED with its value — or the gate's own failure.

    "It failed" is not actionable. "stratum semantico: recall@1 = 0.5 < 1.0" tells the reader
    which bucket to look at and by how much — and, because buckets with no coverage carry the
    sentinel rather than a zero, an unmeasured stratum can never be reported as a failure.

    THE COMPARISONS ARE COUNTED, and zero of them is itself a failure (M2). Every skip above
    is right on its own: a bucket with no cases, and a metric no case in it measured, must
    not be named. But `passed` is `not failures`, so when the skips consume EVERY bucket the
    strictest threshold that exists comes out green having compared nothing — the FAIL-OPEN
    cell of CLAUDE.md rule 11, in the command whose acceptance criterion is "the evaluation
    can fail". Reproduced through the CLI on the real corpus: `--min-recall 1.0` over a
    golden set whose filters the baseline cannot apply exited 0 with `passed: true`.

    So the counter, and not a fabricated bucket: naming a stratum here would be B1 again in
    the opposite direction, inventing a failure for a population nobody measured. What
    failed is the GATE, and the failure says so.
    """
    if threshold is None:
        return ()
    metric = f"recall@{max(ks)}"
    failures = []
    comparisons = 0
    for label, buckets in (("stratum", by_stratum), ("provenance", by_provenance)):
        for name, values in buckets.items():
            if values == NO_COVERAGE:
                continue
            value = values.get(metric)
            # A metric no case in this bucket measured cannot be a failure of this bucket
            # (B1). Reading a missing metric as 0.0 would name a failure that measured
            # nothing — the fabricated zero wearing a gate's clothes.
            if value is None or value == NO_COVERAGE:
                continue
            comparisons += 1
            if value < threshold:
                failures.append(f"{label} {name}: {metric} = {value} < {threshold}")
    if not comparisons:
        failures.append(
            f"el umbral {threshold} no se comparó contra nada: 0 buckets con {metric} "
            "medido. La puerta no puede pasar sin haber medido (M2)"
        )
    return tuple(failures)


def _percentiles(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"p50_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(latencies)
    return {
        "p50_ms": round(ordered[int(len(ordered) * 0.50)], 3),
        "p95_ms": round(ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)], 3),
    }


def retriever_label(strategy: str, requested_strategy: str, degraded: Sequence[str]) -> str:
    """How every published figure names the retriever that produced it — ONE definition.

    Read by BOTH markdown renderers, because `xbrain eval` has two branches that publish a
    number and only one of them used to name its instrument: the ordinary report headed itself
    «`lexical` · solicitada `vector`, sin backend», while `--sweep-chunker` wrote a ranked
    table with no retriever named anywhere in it or in its JSON. Two renderers that "should"
    say the same thing about the same three fields are the divergence rule 5 is about, and the
    fix is a function rather than a second f-string.

    The degradation clause is what makes the label a measurement rather than an echo: a
    strategy with no backend runs as `lexical` and the sentence says whose figures these are
    (spec §9.3 — *el response declara estrategia degradada; no finge resultados vectoriales*).
    """
    label = f"`{strategy}`"
    if degraded:
        label += (
            f" · solicitada `{requested_strategy}`, sin backend "
            f"({', '.join(degraded)}): las cifras son del recuperador que SÍ corrió"
        )
    return label


def render_markdown(report: EvaluationReport) -> str:
    """The human report. Publishes failures and gaps, never fabricated zeros (spec §8.6.8)."""
    lines = [
        "# Evaluación de recuperación — "
        + retriever_label(report.strategy, report.requested_strategy, report.degraded),
        "",
        f"- Corpus: `{report.corpus['source']}` — {report.corpus['items']} items, "
        f"{report.corpus['topics']} topics, {report.corpus['surfaces']} superficies, "
        f"{report.corpus['chunks']} chunks emitidos, "
        f"{report.corpus.get('chunks_not_indexed', 0)} rechazados por el índice.",
        f"- Umbral: {report.threshold if report.threshold is not None else 'ninguno (solo informe)'}.",
        f"- Profundidad: {report.limit} owners por caso (U-6: la lista se materializa hasta"
        " tener esos owners, y cada `recall@k` sale de su prefijo).",
        f"- Latencia p50 {report.latency['p50_ms']} ms · p95 {report.latency['p95_ms']} ms.",
        *_vector_lines(report),
        "",
        "> Las cifras de arriba son una fotografía del corpus medido, no una constante del",
        "> producto. Vuelve a derivarlas al ejecutar (CLAUDE.md regla 2).",
        "",
        "> `recall@k` cuenta OWNERS deduplicados; `surface_recall@k` cuenta CHUNKS (m6). Bajo",
        "> la misma `k` no miden la misma población: la lista se materializa hasta k OWNERS,",
        "> así que `recall@10` puede formarse con más de diez chunks, mientras que",
        "> `surface_recall@10` nunca mira más allá del décimo. Compara cada columna con su",
        "> propio valor en la siguiente ejecución, no una con la otra.",
        "",
        "> Una celda `sin cobertura` NO es un cero: ningún caso del bucket pudo medir esa",
        "> métrica (spec §8.6.8). `measured` en el JSON lleva el denominador de cada media.",
        "",
        "> Entre paréntesis, el DENOMINADOR de cada media: los casos del bucket que midieron esa",
        "> métrica. `no medidos` son los casos del bucket que esta estrategia no puntuó (sus",
        "> filtros): no entran en NINGÚN denominador. Unidad de `k` entre corchetes.",
        "",
        "> `MRR@10` es el rango recíproco dentro de los mismos 10 owners que `recall@10`, y se",
        "> compara entre estrategias. El `mrr` sin corte del JSON recorre la ventana entera del",
        "> recuperador, que no es la misma en `lexical` que en `vector`/`hybrid`: no se compara.",
        "",
        "> `vacíos` cuenta los casos del bucket cuya consulta no recuperó NI UN CHUNK. Un 0,0",
        "> con `vacíos = casos` no dice que el recuperador ordenase mal: dice que no llegó a",
        "> ordenar nada. Sobre esos casos `precision@k` sale *no medida*, nunca 0,0 — su",
        "> numerador es 0 por construcción y repetiría el conjunto vacío (M3).",
        "",
    ]
    lines += _table("Por estrato", report.by_stratum, _unmeasured_counts(report, "strata"))
    lines += _table(
        "Por procedencia", report.by_provenance, _unmeasured_counts(report, "provenance")
    )
    lines += [
        "## Sin cobertura",
        "",
        "No se puntúan y **no se reportan como 0,0** (spec §8.6.8): un cero aquí diría que la",
        "recuperación falló donde nadie preguntó.",
        "",
        f"- Estratos sin casos medibles: {', '.join(report.without_coverage['strata']) or 'ninguno'}.",
        f"- Superficies sin datos en el corpus: {', '.join(report.without_coverage['surfaces'])}.",
        "",
    ]
    if report.unmeasured:
        lines += [
            "### Casos NO medidos (la estrategia no puede aplicar sus filtros)",
            "",
            "No puntúan: un 0,0 aquí diría que la recuperación falló, cuando lo que falta es el",
            "instrumento. Los filtros de fecha, autor, fuente y content kind llegan con el índice",
            "persistido del Plan 02.",
            "",
        ]
        lines += [
            f"- **{entry['id']}** ({', '.join(entry['strata'])}) — filtros sin soporte: "
            f"{', '.join(entry['unsupported_filters'])}."
            for entry in report.unmeasured
        ]
        lines.append("")
    if report.scenarios:
        lines += ["## Escenarios archivados (no puntúan)", ""]
        lines += [f"- **{s.id}** — {s.reason.strip()}" for s in report.scenarios]
        lines.append("")
    if report.failures:
        lines += ["## Fallos", ""] + [f"- {failure}" for failure in report.failures] + [""]
    return "\n".join(lines)


def _vector_lines(report: EvaluationReport) -> list[str]:
    """What produced the vectors and what they cost — on a report that ran a model, and only
    there: a lexical report has no model to name (Plan 03 §3.2, §3.5)."""
    if report.embeddings is None or report.indexing is None:
        return []
    spec, cost, latency = report.embeddings, report.indexing, report.latency
    built = (
        f"construido en {cost['seconds']} s" if cost["built"] else "reutilizado (no se re-embebió)"
    )
    return [
        f"- Modelo: `{spec['model']}` · dimensión {spec['dimension']} · normalizado "
        f"{spec['normalized']} · prefijos query {spec['query_prefix']!r} / passage "
        f"{spec['passage_prefix']!r} · {spec['command_version']}.",
        f"- Índice de evaluación: {built} · plano vectorial de {cost['vector_rows']} filas para "
        f"{cost['vector_chunks']} chunks, {cost['vector_bytes']} bytes en disco.",
        f"- La latencia incluye embeber la consulta: p50 {latency.get('embedding_p50_ms')} ms · "
        f"p95 {latency.get('embedding_p95_ms')} ms; recuperar: p50 "
        f"{latency.get('retrieval_p50_ms')} ms · p95 {latency.get('retrieval_p95_ms')} ms.",
    ]


# The columns of the human table, in order: the metric and the unit its `k` counts. `nDCG@10`
# is BINARY (no grades exist) and `superficies@10` is `surface_recall@10`, whose unit is the
# CHUNK, not the owner (m6) — spec §8.4 asks for both beside recall.
_TABLE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("recall@1", "recall@1", "recall"),
    ("recall@10", "recall@10", "recall"),
    ("precision@10", "precision@10", "precision"),
    ("mrr@10", "MRR@10", "mrr"),
    ("ndcg@10", "nDCG@10", "ndcg"),
    ("surface_recall@10", "superficies@10", "surface_recall"),
)


def _unmeasured_counts(report: EvaluationReport, field: str) -> dict[str, int]:
    """How many cases of each bucket the strategy did NOT score — outside every denominator."""
    counts: dict[str, int] = {}
    for entry in report.unmeasured:
        keys = entry.get(field) or ()
        for key in (keys,) if isinstance(keys, str) else keys:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _table(title: str, buckets: dict[str, Any], unmeasured: Mapping[str, int]) -> list[str]:
    header = " | ".join(f"{label} [{METRIC_UNITS[unit]}]" for _, label, unit in _TABLE_COLUMNS)
    lines = [
        f"## {title}",
        "",
        f"| bucket | casos | vacíos | no medidos | {header} |",
        "|---|---:|---:|---:|" + "---:|" * len(_TABLE_COLUMNS),
    ]
    for name, values in buckets.items():
        skipped = unmeasured.get(name, 0)
        if values == NO_COVERAGE:
            lines.append(
                f"| {name} | — | — | {skipped} | sin cobertura | sin cobertura | sin cobertura "
                "| — | — | — |"
            )
            continue
        cells = [_cell(values, metric) for metric, _, _ in _TABLE_COLUMNS]
        lines.append(
            f"| {name} | {values['cases']} | {values['no_results']} | {skipped} | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")
    return lines


def _cell(values: dict[str, Any], name: str) -> str:
    """One metric cell — words when nobody measured it, never a number (spec §8.6.8).

    The markdown is where a fabricated zero does its damage, because it is the surface that
    gets read and quoted. A `0.0` here is indistinguishable from a measured failure. A number
    carries its denominator in parentheses (PR #186, Codex F2): `cases` is the bucket, and a
    mean over fewer of them is a different population under the same row.
    """
    value = values.get(name)
    if value is None:
        return "—"
    if value == NO_COVERAGE:
        return "sin cobertura"
    return f"{value} ({values.get('measured', {}).get(name, '?')})"


# ---------------------------------------------------------------------------
# The chunker sweep (Plan 02 §7 · §15.12, the signed-measurement half; delivery row 02.13)
# ---------------------------------------------------------------------------
#
# WHY THE INSTRUMENT SHIPS EVEN THOUGH THE MEASUREMENT IS NOT A CI CHECK. Plan 02 §15
# declares criteria 11 and 12 *local measurements, not CI checks*, and they do not block a
# merge. That exempts the MEASUREMENT, not the INSTRUMENT: `800/0` and `CHUNKER_VERSION v2`
# are published in the README and in `docs/knowledge-index.md` as the number Plan 03 has to
# beat, and without this code that number cannot be re-derived by anyone. A figure whose
# instrument is absent is the difference CLAUDE.md rule 2 draws between a measurement and a
# number that restates a constant.
#
# THE SWEEP CHANGES `ChunkerParams` AS AN ARGUMENT AND NEVER THE MODULE CONSTANT (M7), which
# is the whole reason `chunk_surfaces` takes `params` at all: `tests/fixtures/
# knowledge_ranking.json` pins today's ranking by passing its OWN parameters, so a sweep that
# assigned `DEFAULT_CHUNKER_PARAMS` would break the fixture that exists to pin the ranking,
# and the comfortable repair would be to regenerate it — at which point it pins nothing.
#
# WHAT `limit` MEANS HERE: OWNERS, not rows. `evaluate` materialises the ranking until it
# holds that many distinct owners (U-6) and `_score` reads its prefix, so `report.limit` and
# every `recall@k` beneath it count the same population. The version of this block that
# shipped in the umbrella said the opposite — «the depth the RETRIEVER is asked for … as a
# row count» — because the owner-counted depth had been dropped from `evaluate` and the
# comment was corrected to match the code rather than the code to match the contract. Both
# are true statements about their own tree; only one of them describes a `recall@k` that does
# not move when a neighbouring k is requested beside it.


# The k the sweep ranks at when the caller names none. ONE definition, read by the function
# signature below and by the CLI's `_run_sweep`: a second literal in the command would be a
# default that could drift away from the one the report publishes (rule 5).
DEFAULT_SWEEP_K: int = 10


@dataclass(frozen=True)
class SweepRow:
    """One `(target, overlap)` combination and what it scored.

    `chunks` is carried beside the metrics because spec §13.15 asks for negative results to be
    published rather than hidden: when two combinations tie on recall, the tie-break is the
    one that produces FEWER chunks, and that only works if the count is in the table.
    """

    params: ChunkerParams
    chunks: int
    recall: float | None
    mrr: float | None
    by_stratum: dict[str, Any]
    # `recall@1` is depth-independent by construction (one chunk is always one owner) and is
    # the figure the real decision rested on (S-1, round 08); published on every row so a
    # reader can check the tie-break without re-running the sweep at another k.
    recall_at_1: float | None = None


@dataclass(frozen=True)
class SweepReport:
    """Every combination, best first, with the k the ranking was decided on and the DEPTH
    (in owners) every cell ran at (U-6) — the two numbers a reader needs to compare a cell
    with the next sweep's.

    A ranking is not guaranteed: a grid that resolved to no combination, and a table where no
    combination could be scored, both report `winner is None`. The caller decides what to do
    with that; what this class refuses to do is name a winner it does not have.
    """

    k: int
    rows: tuple[SweepRow, ...]
    limit: int = 0
    # THE RETRIEVER THAT RANKED THE CELLS, on the same three fields `EvaluationReport` carries
    # and for the same reason (F-2). `sweep_chunker` already called `evaluate(..., strategy=)`
    # once per cell, so `resolve_strategy` ran, the degradation was computed — and then thrown
    # away with the rest of the per-cell report. `xbrain eval --strategy vector
    # --sweep-chunker …` therefore published `data/eval-sweep.{json,md}` — the artefact Plan 03
    # has to beat — with no retriever named anywhere in it, while the NON-sweep branch of the
    # same command headed its report «`lexical` · solicitada `vector`, sin backend». One
    # command, two branches, and only one of them said which instrument produced the number.
    # `strategy` is what RAN, `requested_strategy` what was ASKED FOR, `degraded` why they
    # differ — never the request echoed back, which is the pretence spec §9.3 forbids.
    strategy: str = "lexical"
    requested_strategy: str = ""
    degraded: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.requested_strategy:
            object.__setattr__(self, "requested_strategy", self.strategy)

    @property
    def measured(self) -> bool:
        """Did ANY combination produce a score? A table of `sin cobertura` is not a ranking.

        Read off the rows' own `recall`, never off the rendered strings: `_number(None)` is the
        constant `"sin cobertura"`, so a predicate computed on the formatted table collapses
        every unscored row onto one value and reads as a tie (F2-3).
        """
        return any(row.recall is not None for row in self.rows)

    @property
    def winner(self) -> SweepRow | None:
        """The top row — and only when that row actually scored.

        The condition is the ROW's own state rather than `self.measured`, so a future change to
        the sort order cannot publish an unscored combination as the winner while the report
        still calls itself measured. Today the two coincide: measurability depends on the cases
        (a filter the strategy cannot push into `WHERE`), never on the chunk size, so every
        combination of one sweep is scorable or none is.
        """
        if not self.rows:
            return None
        top = self.rows[0]
        return top if top.recall is not None else None

    def to_dict(self) -> dict[str, Any]:
        winner = self.winner
        return {
            "k": self.k,
            "limit": self.limit,
            # Ahead of the numbers, like the non-sweep report's heading: a machine consumer
            # that reads `rows` without reading these is reading a ranking whose instrument it
            # never checked.
            "strategy": self.strategy,
            "requested_strategy": self.requested_strategy,
            "degraded": list(self.degraded),
            # The three keys a machine consumer needs to tell a RANKING from a table that could
            # not be ranked. `verdict` is the same sentence the markdown prints, taken from the
            # same function, so the human artefact and the machine one cannot disagree (rule 5).
            "measured": self.measured,
            "winner": (
                None
                if winner is None
                else {"target": winner.params.target, "overlap": winner.params.overlap}
            ),
            "verdict": _sweep_verdict(self),
            "rows": [
                {
                    "target": row.params.target,
                    "max_chars": row.params.max_chars,
                    "overlap": row.params.overlap,
                    "min_chars": row.params.min_chars,
                    "chunks": row.chunks,
                    f"recall@{self.k}": row.recall,
                    "recall@1": row.recall_at_1,
                    "mrr": row.mrr,
                    "by_stratum": row.by_stratum,
                }
                for row in self.rows
            ],
        }


def parse_sweep(values: Sequence[str]) -> dict[str, list[int]]:
    """`["target=800,1200", "overlap=0,150"]` -> `{"target": [800, 1200], ...}`.

    Whitespace inside one value is also split, so the plan's own syntax —
    `--sweep-chunker "target=800,1200,1600,2400 overlap=0,150,300"` — works as written when
    quoted, and so does one flag per axis. An unknown key is REFUSED rather than ignored: a
    typo that silently swept nothing would publish the default's numbers under the name of a
    sweep.
    """
    grid: dict[str, list[int]] = {}
    for value in values:
        for token in value.split():
            if "=" not in token:
                raise ValueError(f"Formato de barrido inválido: {token!r}. Usa `clave=v1,v2`.")
            key, raw = token.split("=", 1)
            if key not in {"target", "max_chars", "overlap", "min_chars"}:
                raise ValueError(
                    f"Eje de barrido desconocido: {key!r}. "
                    "Válidos: target, max_chars, overlap, min_chars."
                )
            grid[key] = [int(part) for part in raw.split(",") if part.strip()]
    return grid


def sweep_chunker(
    cases: Sequence[GoldenCase],
    corpus: Corpus,
    grid: Mapping[str, Sequence[int]],
    *,
    strategy: str = "lexical",
    k: int = DEFAULT_SWEEP_K,
    base: ChunkerParams = DEFAULT_CHUNKER_PARAMS,
    limit: int | None = None,
) -> SweepReport:
    """Score every combination in `grid` against the golden set (Plan 02 §7).

    THE CRITERION, IN ORDER (S-1, round 08): `recall@k` first, MRR second, FEWER CHUNKS last.
    Plan 02 §7 wrote «si empata, se escoge el que produzca menos chunks» and the README
    repeated it, while this function has ordered by MRR before the chunk count since before
    any measurement existed. A gate found the published winner contradicting the published
    rule: the written rule chose `1200/0`, the applied one `800/0`, and the tie-break was the
    whole decision. The rule that stands is this one, and the plan and the README now say it,
    with the reason: `recall@k` and MRR are both retrieval QUALITY — whether the relevant item
    is on the page, and where on it — and the consumer of `search` is an agent that reads the
    top of the page, so rank position is not a tie-breaking nicety; the chunk count is a COST
    (disk, build time) and a cost breaks a tie in quality only when quality is flat, which is
    what spec §13.15's «flat result» means. `recall@1` is published on every row because it is
    the depth-independent form of the same argument. The report names WHICH criterion decided,
    so the reader never infers it from the table.

    THE FIGURES THAT DECIDED IT ARE ROUND-08's, AND THEY ARE NOT RE-DERIVABLE HERE (rule 6).
    `MRR 0.8179 against 0.7667`, and the `1200/0` chunk count quoted beside them, come from
    the round-08 sweep — an earlier harness, an earlier chunker — and no instrument in this
    tree produces them. Quoting them under a «once U-6 counted the depth in owners» clause, as
    an earlier revision of this docstring did, re-dates a pre-U-6 measurement as a post-U-6
    one: the argument for the ORDER survives (a tie on `recall@k` broken by MRR rather than by
    cost), the two numbers are history and must be read as history.

    WHAT THE REPAIRED HARNESS MEASURES, with its population beside it (rule 2). Corpus
    `data/items.json`, 2,474 items, sha256 `4fed54a0…`; `eval/golden-set.yaml` as versioned;
    CPython 3.12.11, `PYTHONHASHSEED=0`; grid `target=800,1200,1600,2400 overlap=0,150,300` at
    `k=10`, depth 10 owners. The winner is **`800/150`** (23,651 chunks, `recall@10` 0.7395,
    MRR 0.7360), tied on `recall@10` with `800/0` (22,933 chunks, MRR 0.7357) and decided by
    MRR — the same criterion, a different winner. So `800/0` is not the winner under the
    instrument this harness is now; it was the winner under the one that counted its depth in
    CHUNKS. Every `800/0`-derived figure published before this child — `CHUNKER_VERSION`
    included — is a derivative of that older instrument and is retired with it. Re-derive on
    the corpus in front of you; these numbers move with it.

    A combination that scores nothing measurable sorts last instead of sorting first, which is
    what a `None` would do under a naive `max`. And when NO combination scored, the report has
    no winner at all — `sin cobertura` on every row is the absence of a ranking, never a tie
    between rows that were never compared.

    `limit` is the depth in OWNERS every cell runs at (U-6): the CLI's `--limit`, threaded
    through and published on the report. The first version of the sweep called `evaluate`
    with no depth and the CLI's `_run_sweep` never passed the option the command advertised,
    so `--limit 10` and `--limit 150` produced byte-identical reports on the real corpus.

    IT IS PUBLISHED AT THE VALUE THE RUN USED, not at the value asked for. `evaluate` clamps
    its own depth to `max(limit, max(ks))`, so a `limit` below `k` never reaches the index;
    clamping here too is what keeps `report.limit` from naming a depth no cell ran at. It
    defaults to `k`.
    """
    depth = max(limit if limit is not None else k, k)
    # RESOLVED ONCE, HERE, AND PUBLISHED ON THE REPORT (F-2, this pass). `evaluate` resolves it
    # per cell and the answer is identical for every cell — it depends on the strategy, never
    # on the chunker's parameters — so resolving it here costs nothing and gives the table
    # somewhere to carry it. It also makes an unknown strategy raise BEFORE the first cell is
    # scored, instead of after a full grid has been walked.
    executed, degraded = resolve_strategy(strategy)
    rows: list[SweepRow] = []
    for params in _combinations(grid, base):
        report = evaluate(cases, corpus, strategy=strategy, ks=(1, k), params=params, limit=depth)
        overall = _overall(report, k)
        first = _measured(report, "recall@1")
        rows.append(
            SweepRow(
                params=params,
                chunks=report.index_stats.chunks if report.index_stats else 0,
                recall=overall[0],
                mrr=overall[1],
                by_stratum=report.by_stratum,
                recall_at_1=sum(first) / len(first) if first else None,
            )
        )
    rows.sort(key=lambda row: (-(row.recall or -1.0), -(row.mrr or -1.0), row.chunks))
    return SweepReport(
        k=k,
        rows=tuple(rows),
        limit=depth,
        strategy=executed,
        requested_strategy=strategy,
        degraded=degraded,
    )


def _combinations(grid: Mapping[str, Sequence[int]], base: ChunkerParams) -> list[ChunkerParams]:
    """The cartesian product of the swept axes, with the unswept ones held at `base`.

    Deterministic order — the axes are sorted and each axis keeps the order it was given — so
    two runs of the same sweep produce the same table and a diff between them is readable.
    """
    axes = sorted(grid)
    combos = [dict[str, int]()]
    for axis in axes:
        combos = [{**combo, axis: value} for combo in combos for value in grid[axis]]
    return [
        ChunkerParams(
            target=combo.get("target", base.target),
            max_chars=combo.get("max_chars", base.max_chars),
            overlap=combo.get("overlap", base.overlap),
            min_chars=combo.get("min_chars", base.min_chars),
        )
        for combo in combos
    ]


def _overall(report: EvaluationReport, k: int) -> tuple[float | None, float | None]:
    """The mean `recall@k` and MRR over every SCORED case, or `(None, None)` if none scored.

    Computed over the cases rather than over the stratum means, because the strata have very
    different sizes and averaging the averages would weight a one-case stratum like a
    twelve-case one.
    """
    recalls = _measured(report, f"recall@{k}")
    mrrs = _measured(report, "mrr")
    return (
        sum(recalls) / len(recalls) if recalls else None,
        sum(mrrs) / len(mrrs) if mrrs else None,
    )


def _measured(report: EvaluationReport, metric: str) -> list[float]:
    """Every case that actually measured `metric`. A `None` is an ABSENCE, never a zero.

    Filtering here rather than defaulting to 0.0 is the same rule B1 established at bucket
    level, applied one layer down: a case that could not measure a metric must not drag the
    mean towards a number nobody observed.
    """
    values = []
    for case in report.cases:
        value = case.metrics.get(metric)
        if value is not None:
            values.append(float(value))
    return values


def render_sweep_markdown(report: SweepReport) -> str:
    """The sweep table, winner first (Plan 02 §7).

    Published even when flat, and the flatness is stated in the table rather than left for a
    reader to notice: spec §13.15 asks for negative results to be documented, and a sweep
    whose rows are indistinguishable is a result about the chunker, not a missing measurement.
    """
    lines = [
        # FIRST LINE, ahead of the depth and the criterion: the ranking is a statement about a
        # retriever, and a reader who takes the winner out of this table without knowing which
        # one ranked it has the F-2 defect in the artefact Plan 03 is measured against.
        "Recuperador: "
        + retriever_label(report.strategy, report.requested_strategy, report.degraded),
        f"Profundidad: {report.limit} owners por caso (U-6).",
        f"Criterio: recall@{report.k}, luego MRR, luego menos chunks (S-1).",
        f"| target | overlap | chunks | recall@{report.k} | recall@1 | MRR |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.rows:
        lines.append(
            f"| {row.params.target} | {row.params.overlap} | {row.chunks} "
            f"| {_number(row.recall)} | {_number(row.recall_at_1)} | {_number(row.mrr)} |"
        )
    # Emitted unconditionally: every table has a sentence now, including the two that have no
    # winner. The guard that used to stand here (`if report.winner is not None`) meant a sweep
    # with no rows published a bare header and said nothing at all about it.
    lines += ["", _sweep_verdict(report)]
    return "\n".join(lines)


def _sweep_verdict(report: SweepReport) -> str:
    """The one sentence a reader takes away, for EVERY shape of table.

    Three of the five cases are not rankings at all — no rows, nothing scored, a single cell —
    and each of them used to be reported as something it was not. They are answered here; the
    ranked table's own «which criterion decided» is `_decided_verdict`, kept apart so this
    function reads as the list of shapes a sweep can have.
    """
    # NO ROWS AT ALL. A grid whose axis resolved to no value swept nothing, and a header with
    # no table under it is not a result; saying so is cheaper than making a reader notice.
    if not report.rows:
        return (
            "SIN COMBINACIONES: el barrido no produjo ninguna fila, así que no hay ganador "
            "(¿un eje sin valores, `target=`?)."
        )
    winner = report.winner
    # ROWS, BUT NOTHING MEASURED (F2-3 of the final gate on #177). `distinct` below is built
    # out of `_number(...)` STRINGS and `_number(None)` is the constant `"sin cobertura"`, so N
    # unscored rows collapsed to ONE distinct value and took the flat branch: «PLANO: todas las
    # combinaciones puntúan igual» over a table where nothing was compared to anything. That is
    # declared deviation 3 — the one-row fake tie — one input class over, and the verdict is the
    # line a reader takes away. A tie is a RESULT about the chunker; this is its absence, and
    # the two now read differently.
    if winner is None:
        return (
            f"SIN MEDICIÓN: ninguna de las {len(report.rows)} combinaciones pudo puntuarse "
            f"(recall@{report.k} sin cobertura en todas), así que no hay ganador ni empate: "
            "el barrido no midió nada."
        )
    label = f"target={winner.params.target}, overlap={winner.params.overlap}"
    # A DEVIATION FROM THE SNAPSHOT, and the reason. `len(distinct) == 1` is true for a
    # ONE-ROW sweep as well, so a single-combination run always printed «PLANO: todas las
    # combinaciones puntúan igual» — a verdict about a tie, over a table with nothing to tie
    # against, that no input could have made say anything else. That is the shape CLAUDE.md
    # rule 2 rejects, inside the instrument that exists to publish an honest measurement.
    # A single cell is a measurement of one combination and says so.
    if len(report.rows) == 1:
        return f"UNA COMBINACIÓN: {label}; no hay barrido que comparar."
    return _decided_verdict(report, winner, label)


def _decided_verdict(report: SweepReport, winner: SweepRow, label: str) -> str:
    """Which criterion DECIDED, said in the report (S-1): a tie on `recall@k` is named, with
    the rows it spans, and the criterion that broke it is named with its two values.

    Reached only for a table with MORE THAN ONE row and a scored winner, so every branch below
    is a statement about a comparison that actually happened.
    """
    # Reached only with a winner, i.e. with at least one row SCORED — so a `sin cobertura`
    # that survives into this set is a partially-unmeasured table, never the empty one.
    distinct = {(_number(r.recall), _number(r.mrr)) for r in report.rows}
    if len(distinct) == 1:
        return "PLANO: todas las combinaciones puntúan igual; gana la que produce menos chunks."
    tied = [r for r in report.rows[1:] if _number(r.recall) == _number(winner.recall)]
    if not tied:
        return f"Gana {label}: decidió recall@{report.k} ({_number(winner.recall)})."
    names = ", ".join(f"{r.params.target}/{r.params.overlap}" for r in tied)
    runner_up = tied[0]
    if _number(runner_up.mrr) != _number(winner.mrr):
        return (
            f"Gana {label}: empate en recall@{report.k} ({_number(winner.recall)}) con {names}; "
            f"decidió MRR ({_number(winner.mrr)} frente a {_number(runner_up.mrr)}; "
            f"recall@1 {_number(winner.recall_at_1)} frente a {_number(runner_up.recall_at_1)})."
        )
    return (
        f"Gana {label}: empate en recall@{report.k} y en MRR con {names}; "
        f"decidió menos chunks ({winner.chunks} frente a {runner_up.chunks})."
    )


def _number(value: float | None) -> str:
    return "sin cobertura" if value is None else f"{value:.4f}"


# ---------------------------------------------------------------------------
# Plan 03.7 — `vector` and `hybrid` in the harness, and the bake-off's instruments
# ---------------------------------------------------------------------------
#
# ONE INDEX PER MODEL, WRITTEN BY THE BUILDER THAT WRITES THE REAL ONE. The vector strategies
# are scored over a persisted index `index_build.build(..., vectors=...)` writes — the writer
# and the plane `xbrain index build --embeddings` produce — under `data/eval-index/<model>/`,
# never over `data/index/`, which belongs to `search`. Its manifest is what TDD 23 reads.
#
# THE RANKING IS THE SEARCH SERVICE'S FUSED WINDOW, NOT A SECOND FUSION. `_fused_hits` calls
# `search_service._fused_window` and `_fuse_hits` — private, and imported anyway, because the
# alternative is a second copy of «which chunks each channel offers and how they are fused»:
# the five-hands divergence of rule 5, inside the module that exists to measure the first
# copy. What the harness does NOT take from the service is the item grouping: it scores OWNERS
# off the chunk ranking exactly as it scores the lexical baseline, so the three strategies are
# compared on one unit. And like the baseline it leaves the profile plane out, so `hybrid`'s
# profile fill (Plan 03 §4.3) is outside this measurement — declared, not forgotten.

# The strata the bake-off DECIDES on, and the one it must not break (Plan 03 §3.3; spec §8.6.3
# and §8.6.4). Named once, so the comparison and the published document read the same list.
DECISIVE_STRATA: tuple[str, ...] = ("semantico", "cruzado_idioma")
GUARDRAIL_STRATA: tuple[str, ...] = ("exacto",)

# How one case is retrieved: `(hits in rank order, depth exhausted, query-embedding ms)`.
Retrieval = tuple[Sequence[LexicalHit], bool, "float | None"]


def eval_index_dir(data_dir: Path, model: str) -> Path:
    """Where the evaluation index of `model` lives: `data/eval-index/<model>/`, one per model.

    One per model is what lets `vector` and `hybrid` share a plane built once while two
    candidates never share one. The directory is keyed by the name, and the manifest inside is
    checked against the request anyway (TDD 23), because a directory name proves nothing.
    """
    return data_dir / "eval-index" / re.sub(r"[^A-Za-z0-9._-]+", "__", model)


@dataclass(frozen=True)
class VectorEvaluation:
    """Everything `vector` and `hybrid` need that the corpus does not carry.

    `build` is the SAME `VectorBuild` `index build --embeddings` receives: its `spec` is what
    the backend DECLARED on a probe, so `spec.model` is the model that will produce the
    vectors, and `requested_model` is the one the report is about to be headed with. The two
    are compared before anything is written.

    The three input paths are here because a persisted index is sealed against the FILES it was
    read from (P1b), and the harness's `Corpus` carries objects, not files.
    """

    requested_model: str
    build: VectorBuild
    embed_query: Callable[[str], Sequence[float]]
    index_dir: Path
    items_path: Path
    vocab_path: Path
    topics_path: Path
    command: str = ""


class EmbeddingModelMismatch(ValueError):
    """The model that would produce the numbers is not the model asked for (TDD 23).

    A `ValueError`, so the CLI turns it into a clean exit 1. Never answered with a rebuild: the
    index it refuses may be someone's measurement of the other model.
    """


# WHICH STRATEGIES THE HARNESS SCORES WITH AN EMBEDDINGS MODEL — the only ones `--embeddings-model`
# pairs with (`require_vector_arguments`). Deliberately NOT `search_service._VECTOR_STRATEGIES`:
# that set answers «does `search` open the vector channel for this strategy?», this one «is a
# report headed by a model a measurement of THIS strategy?». The two agreed by accident, and one
# constant answering both flipped both pairings the moment Plan 04.4 put `hybrid_graph` in the
# first (rule 5): `evaluate(strategy="hybrid_graph", vectors=…)` published a report named
# `hybrid_graph` whose ranking was `vector`'s, chunk for chunk — `_fused_hits` fuses the lexical
# channel only for `hybrid` — and in which no graph ran, because the harness has no graph channel.
# `hybrid_graph` joins this set when the harness executes the graph over `hybrid`'s fusion, not
# before; until then, without a model it degrades to `lexical` and says so (`_resolve_strategy`).
EMBEDDING_MODEL_STRATEGIES: frozenset[str] = frozenset({"vector", "hybrid"})


def require_vector_arguments(strategy: str, model: str | None) -> None:
    """The ONE place `--strategy` and `--embeddings-model` are paired (Plan 03 §3.2).

    A vector strategy with no model has nothing to measure, and a model beside a strategy the
    harness does not score with one would head a report that is not that strategy's measurement:
    `lexical`, none of whose numbers the model produced, or `hybrid_graph`, whose graph the
    harness does not run. Paired against `EMBEDDING_MODEL_STRATEGIES`, never against the search
    service's vector set (see the constant), and the suggestion is built from that constant so it
    cannot enumerate anything else. Read by the CLI before anything is loaded — the embedder
    probe loads a model — and by `evaluate` for the half an API caller can reach.
    """
    if model is not None and strategy not in EMBEDDING_MODEL_STRATEGIES:
        usable = " o ".join(f"`--strategy {name}`" for name in sorted(EMBEDDING_MODEL_STRATEGIES))
        raise ValueError(
            f"`--embeddings-model {model}` no se puede medir con `--strategy {strategy}`: el "
            f"arnés de evaluación solo puntúa un modelo de embeddings con {usable}. Con otra "
            "estrategia el informe llevaría el nombre de ese modelo sobre un ranking que ese "
            "modelo no produjo, o que no es el de la estrategia pedida."
        )
    if model is None and strategy in EMBEDDING_MODEL_STRATEGIES:
        raise ValueError(
            f"`--strategy {strategy}` mide un modelo de embeddings y no se nombró ninguno: pasa "
            "`--embeddings-model <modelo>` (Plan 03 §3.2)."
        )


def _resolve_strategy(
    strategy: str, vectors: VectorEvaluation | None
) -> tuple[str, tuple[str, ...]]:
    """What runs: the contract's resolution without vectors, the requested strategy with them.

    A typo is refused AS a typo before the model is mentioned, so `--strategy lexcial
    --embeddings-model m` names the misspelling rather than a pairing nobody asked about.
    """
    if vectors is None:
        return resolve_strategy(strategy)
    resolve_strategy(strategy)
    require_vector_arguments(strategy, vectors.requested_model)
    return strategy, ()


@dataclass(frozen=True)
class _VectorRun:
    """An evaluation index proved queryable, its plane, and what the report records of both."""

    index: OpenIndex
    plane: VectorPlane
    embed_query: Callable[[str], Sequence[float]]
    embeddings: dict[str, Any]
    indexing: dict[str, Any]

    def close(self) -> None:
        self.plane.close()
        self.index.close()


def _open_vector_run(vectors: VectorEvaluation, *, params: ChunkerParams) -> _VectorRun:
    """Build the evaluation index of `vectors.requested_model`, or reuse it, and open it.

    FOUR OUTCOMES, in this order, and the order is what keeps a refusal from costing a build:

    1. the backend declares ANOTHER model than the one asked for -> refused, nothing written;
    2. the directory's manifest names ANOTHER model -> refused, the index left as it was;
    3. the index is current for this spec AND this store -> reused, `built: false`;
    4. anything else (absent, behind the store, another prefix, torn) -> rebuilt and timed.

    Reuse asks the two questions `search` asks — `index_behind_store` off the cheap signal, and
    `vector_verdict` on the plane's coverage of every chunk's CURRENT text — because a plane
    reused over a store that moved measures a corpus that is no longer there.

    `command_version` is recorded HERE and not in the index manifest, on purpose and declared:
    the manifest's `embeddings` block is `VectorSpec`, closed and validated since 03.4, and the
    embedder contract has no version query. What is recorded is the command that ran and the
    wire contract it spoke — the two things a re-run needs to reproduce the vectors.
    """
    from xbrain.embeddings import SCHEMA_VERSION
    from xbrain.knowledge.index_build import IndexOptions, build, load_index_inputs

    spec = vectors.build.spec
    if spec.model != vectors.requested_model:
        raise EmbeddingModelMismatch(
            f"se pidió medir `{vectors.requested_model}` y el embedder de `[embeddings].command` "
            f"declara `{spec.model}`: el informe llevaría el nombre de un modelo que no produjo "
            "sus vectores. Configura el embedder para servir el modelo pedido."
        )
    stored = _stored_model(vectors.index_dir)
    if stored is not None and stored != vectors.requested_model:
        raise EmbeddingModelMismatch(
            f"el índice de evaluación de {vectors.index_dir} declara el modelo `{stored}` en su "
            f"manifest y se pidió medir `{vectors.requested_model}`: jamás se comparan vectores "
            "de dos modelos, y reconstruir encima borraría en silencio la medición del otro. "
            "Usa otro directorio o bórralo a mano."
        )
    seconds: float | None = None
    if not _reusable(vectors, params):
        started = time.perf_counter()
        inputs = load_index_inputs(vectors.items_path, vectors.vocab_path, vectors.topics_path)
        build(
            vectors.index_dir,
            inputs,
            options=IndexOptions(params=params),
            force=True,
            vectors=vectors.build,
        )
        seconds = round(time.perf_counter() - started, 3)
    index, plane = _open_plane(vectors, params)
    command = vectors.command or "(embedder inyectado)"
    return _VectorRun(
        index=index,
        plane=plane,
        embed_query=vectors.embed_query,
        embeddings={
            "model": spec.model,
            "dimension": spec.dimension,
            "normalized": spec.normalized,
            "query_prefix": spec.query_prefix,
            "passage_prefix": spec.passage_prefix,
            "command_version": f"{command} · contrato del embedder v{SCHEMA_VERSION}",
        },
        indexing={
            "built": seconds is not None,
            "seconds": seconds,
            "vector_chunks": plane.chunk_count,
            "vector_rows": plane.row_count,
            "vector_bytes": _plane_bytes(vectors.index_dir),
        },
    )


def _stored_model(index_dir: Path) -> str | None:
    """The model an existing evaluation index's manifest declares, or `None` if it declares none.

    An unreadable manifest declares nothing anybody could have measured, so it is rebuilt over
    rather than refused — the refusal is reserved for a READABLE claim of another model.
    """
    from xbrain.knowledge.index_build import load_manifest, manifest_spec
    from xbrain.knowledge.index_schema import manifest_path

    if not manifest_path(index_dir).exists():
        return None
    try:
        spec = manifest_spec(load_manifest(index_dir))
    except IndexError_:
        return None
    return spec.model if spec is not None else None


def _reusable(vectors: VectorEvaluation, params: ChunkerParams) -> bool:
    """Whether the index on disk answers for THIS spec over THIS store, chunk for chunk."""
    from xbrain.knowledge.index_build import stored_chunk_texts, vector_verdict
    from xbrain.knowledge.index_store import open_for_query

    try:
        index = open_for_query(
            vectors.index_dir,
            vectors.items_path,
            vectors.vocab_path,
            vectors.topics_path,
            params=params,
        )
    except IndexError_:
        return False
    try:
        if "index_behind_store" in index.degraded:
            return False
        verdict = vector_verdict(
            vectors.index_dir,
            index.manifest,
            expected=vectors.build.spec,
            texts=stored_chunk_texts(index.lexical.connection),
        )
        return verdict.usable
    finally:
        index.close()


def _open_plane(vectors: VectorEvaluation, params: ChunkerParams) -> tuple[OpenIndex, VectorPlane]:
    from xbrain.knowledge.index_store import open_for_query
    from xbrain.knowledge.vector_index import load_vector_plane

    index = open_for_query(
        vectors.index_dir,
        vectors.items_path,
        vectors.vocab_path,
        vectors.topics_path,
        params=params,
    )
    try:
        return index, load_vector_plane(vectors.index_dir, expected=vectors.build.spec)
    except BaseException:
        index.close()
        raise


def _plane_bytes(index_dir: Path) -> int:
    """The plane's size on disk — BOTH files, read off the filesystem rather than multiplied."""
    from xbrain.knowledge.vector_index import VECTORS_FILENAME, VECTORS_META_FILENAME

    return sum(
        (index_dir / name).stat().st_size for name in (VECTORS_FILENAME, VECTORS_META_FILENAME)
    )


def _retriever(
    index: LexicalIndex, run: _VectorRun | None, strategy: str, depth: int
) -> Callable[[GoldenCase], Retrieval]:
    """How one case is retrieved under `strategy`: the lexical owner window, or the fused one."""
    if run is None:

        def lexical(case: GoldenCase) -> Retrieval:
            hits, exhausted = _search(index, case, owners=depth)
            return hits, exhausted, None

        return lexical

    def fused(case: GoldenCase) -> Retrieval:
        vector, embedding_ms = _embed_query(run, case)
        hits, exhausted = _fused_hits(run, case.query, vector, strategy, depth)
        return hits, exhausted, embedding_ms

    return fused


def _embed_query(run: _VectorRun, case: GoldenCase) -> tuple[tuple[float, ...], float]:
    """The query's vector and what embedding it cost, in milliseconds."""
    started = time.perf_counter()
    vector = tuple(float(value) for value in run.embed_query(case.query))
    return vector, (time.perf_counter() - started) * 1000


def _fused_hits(
    run: _VectorRun, query: str, vector: tuple[float, ...], strategy: str, depth: int
) -> tuple[list[LexicalHit], bool]:
    """The chunk ranking `search` fuses for `strategy`, and whether its bound cut the owners short.

    `_fused_window` reports a channel that FILLED `FUSED_CHUNK_WINDOW` — on the real corpus the
    vector channel always does — so «exhausted» here is that AND fewer owners than the depth:
    a full window that still holds the owners asked for did not cut the list (U-6).
    """
    from xbrain.knowledge.search_service import _fuse_hits, _fused_window, _VectorChannel

    channel = _VectorChannel(plane=run.plane, vector=vector, lexical=strategy == "hybrid")
    window, _excluded, full = _fused_window(run.index, channel, query)
    hits = [hit for hit, _ in _fuse_hits(window.fuses_lexical, window.lexical, window.vector)]
    return hits, full and distinct_owners(hits) < depth


def _score_cases(
    cases: Sequence[GoldenCase],
    strategy: str,
    ks: tuple[int, ...],
    depth: int,
    retrieve: Callable[[GoldenCase], Retrieval],
    unmeasured: list[dict[str, Any]],
    timings: dict[str, list[float]],
) -> list[CaseResult]:
    """Score every case `strategy` can apply; record the rest as unmeasured, and what each cost."""
    results: list[CaseResult] = []
    for case in cases:
        blocked = unsupported_filters(case.filters, strategy)
        if blocked:
            unmeasured.append(
                {
                    "id": case.id,
                    "strata": list(case.strata),
                    "provenance": case.provenance,
                    "unsupported_filters": list(blocked),
                    "reason": (
                        f"la estrategia `{strategy}` no puede aplicar {list(blocked)}; "
                        "puntuar el caso sería fabricar un cero (spec §8.6.8)"
                    ),
                }
            )
            continue
        started = time.perf_counter()
        hits, exhausted, embedding_ms = retrieve(case)
        timings["total"].append((time.perf_counter() - started) * 1000)
        if embedding_ms is not None:
            timings["embedding"].append(embedding_ms)
        results.append(_score(case, hits, ks, depth=depth, depth_exhausted=exhausted))
    return results


def _latency(timings: Mapping[str, list[float]]) -> dict[str, float]:
    """p50/p95 of the whole query — and, when the query had to be embedded, of each half.

    In production `search --strategy hybrid` pays the embedder on EVERY query, so that cost is
    part of the latency; it is split out because the two halves move for different reasons (a
    model and an index), and one blended figure could not say which one got slower.
    """
    total = list(timings["total"])
    latency = _percentiles(total)
    embedding = list(timings.get("embedding", ()))
    if embedding:
        retrieval = [whole - spent for whole, spent in zip(total, embedding, strict=True)]
        latency |= {f"embedding_{key}": value for key, value in _percentiles(embedding).items()}
        latency |= {f"retrieval_{key}": value for key, value in _percentiles(retrieval).items()}
    return latency


def _ndcg(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary nDCG@k over owners (spec §8.4): gain 1 for a relevant owner, 0 for anything else.

    BINARY because the golden set carries no relevance grades, and grades invented here would
    be ground truth the evaluation generated for itself (spec §8.3). The ideal ranking puts
    every relevant owner first, up to `k`.
    """
    gain = sum(
        1.0 / math.log2(position + 1)
        for position, key in enumerate(ranked[:k], start=1)
        if key in relevant
    )
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return gain / ideal


def compare_reports(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    k: int = DEFAULT_SWEEP_K,
    exclude: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Spec §8.6 gates 3 and 4, case by case: `candidate` against `baseline` (normally lexical).

    PAIRED, NEVER MEAN AGAINST MEAN. A stratum is compared over the cases BOTH reports measured
    `recall@k` on — named in `cases` — and every other member is named in `unpaired`. Two means
    over different populations under one stratum name is the rule-2 defect, and a vector report
    walks into it first: the filtered cases it cannot measure are the ones lexical scores best.

    A STRATUM WITH NOTHING PAIRED IS REFUSED (TDD 28, B2): `sin cobertura`, listed in
    `rejected_strata` with its reason, and any gate needing it fails naming it. Never `empata`:
    two absences are not an equality, and «hybrid no degrada exacto» over zero cases is the
    vacuous pass rule 11 calls fail-open.

    The verdict orders by `recall@k`, then `mrr@k` — the sweep's S-1 criterion — so a candidate
    that finds the same items and ranks them higher is `mejora`. `passes_gates` is true only when
    `exacto` is measured and not worse AND every decisive stratum is measured and better. It
    decides nothing by itself; the published document does, with these numbers beside it.

    THE MRR IS `mrr@k`, NEVER THE BARE `mrr` (PR #186, Codex F1): both halves of the verdict are
    read off the same top-k owner prefix, so they share one depth whatever window each
    retriever materialised. A case that does not carry `mrr@k` on both sides is UNPAIRED with
    its reason — the old `or 0.0` read a missing MRR as a measured zero.

    EVERY MEAN SHIPS WITH ITS DENOMINATOR AND ITS UNIT (Codex F2): `denominators` per stratum,
    `units` and each report's `depth` at the top, and `unpaired_reasons` naming why each
    member of the stratum stayed out — unmeasured by a strategy, a metric missing, or excluded.
    """
    metric = f"recall@{k}"
    rank_metric = f"mrr@{k}"
    # EXCLUSIONS ARE AN ARGUMENT, NOT A HAND-EDITED REPORT (Plan 03 §3.3-3.4). A case whose
    # ground truth no longer verifies on the corpus being measured does not decide; it is
    # carried into the output by name, with its reason, so the published verdict re-derives.
    excluded = dict(exclude or {})
    strata = {
        name: _compare_stratum(baseline, candidate, name, (metric, rank_metric), excluded)
        for name in (*GUARDRAIL_STRATA, *DECISIVE_STRATA)
    }
    rejected = {
        name: entry["reason"]
        for name, entry in strata.items()
        if entry["verdict"] == NO_COVERAGE["coverage"]
    }
    reasons = [f"{name}: {reason}" for name, reason in rejected.items()]
    for name in GUARDRAIL_STRATA:
        if strata[name]["verdict"] == "empeora":
            reasons.append(f"{name}: empeora — {_movement(strata[name], metric, rank_metric)}")
    for name in DECISIVE_STRATA:
        if strata[name]["verdict"] in {"empata", "empeora"}:
            reasons.append(
                f"{name}: no mejora ({strata[name]['verdict']}) — "
                f"{_movement(strata[name], metric, rank_metric)}"
            )
    return {
        "k": k,
        "metric": metric,
        "rank_metric": rank_metric,
        "units": {metric: METRIC_UNITS["recall"], rank_metric: METRIC_UNITS["mrr"]},
        "depth": {"baseline": baseline.get("limit"), "candidate": candidate.get("limit")},
        "baseline_strategy": baseline.get("strategy"),
        "candidate_strategy": candidate.get("strategy"),
        "excluded": excluded,
        "strata": strata,
        "rejected_strata": rejected,
        "passes_gates": not reasons,
        "reasons": reasons,
    }


def _compare_stratum(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    stratum: str,
    metrics: tuple[str, str],
    excluded: Mapping[str, str],
) -> dict[str, Any]:
    metric, rank_metric = metrics
    base = _stratum_metrics(baseline, stratum)
    cand = _stratum_metrics(candidate, stratum)
    # A case BOTH strategies left unmeasured is in neither `cases` list, only in `unmeasured`:
    # without this it vanished from the comparison instead of being named as outside it.
    members = base.keys() | cand.keys() | _unmeasured_members(baseline, candidate, stratum)
    left_out = {case_id: excluded[case_id] for case_id in sorted(members) if case_id in excluded}
    paired = sorted(
        case_id
        for case_id in base.keys() & cand.keys()
        if case_id not in excluded
        and all(side[case_id].get(name) is not None for side in (base, cand) for name in metrics)
    )
    unpaired = sorted(members - set(paired) - set(left_out))
    entry: dict[str, Any] = {
        "cases": paired,
        "unpaired": unpaired,
        "unpaired_reasons": {
            case_id: _unpaired_reason(case_id, baseline, candidate, base, cand, metrics)
            for case_id in unpaired
        },
        "excluded": left_out,
        "denominators": {metric: len(paired), rank_metric: len(paired)},
    }
    if not paired:
        named = [*unpaired, *left_out]
        outside = f" (fuera: {', '.join(named)})" if named else ""
        return entry | {
            "verdict": NO_COVERAGE["coverage"],
            "reason": (
                "ningún caso con verdad de terreno enumerada medido en las dos estrategias"
                f"{outside}: sin él no hay comparación, y dos ausencias no son un empate (B2)"
            ),
        }
    means = {
        side: {
            name: round(sum(float(values[c][name]) for c in paired) / len(paired), 4)
            for name in metrics
        }
        for side, values in (("baseline", base), ("candidate", cand))
    }
    before = (means["baseline"][metric], means["baseline"][rank_metric])
    after = (means["candidate"][metric], means["candidate"][rank_metric])
    verdict = "mejora" if after > before else "empeora" if after < before else "empata"
    per_case = {
        case_id: {
            side: {name: values[case_id][name] for name in metrics}
            for side, values in (("baseline", base), ("candidate", cand))
        }
        for case_id in paired
    }
    return entry | means | {"verdict": verdict, "reason": None, "per_case": per_case}


def _stratum_metrics(report: Mapping[str, Any], stratum: str) -> dict[str, Mapping[str, Any]]:
    return {
        str(case["id"]): case["metrics"]
        for case in report.get("cases", ())
        if stratum in case.get("strata", ())
    }


def _unmeasured_members(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], stratum: str
) -> set[str]:
    return {
        str(entry["id"])
        for report in (baseline, candidate)
        for entry in report.get("unmeasured", ())
        if stratum in entry.get("strata", ())
    }


def _unpaired_reason(
    case_id: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    base: Mapping[str, Mapping[str, Any]],
    cand: Mapping[str, Mapping[str, Any]],
    metrics: tuple[str, ...],
) -> str:
    """Why one member of a stratum is outside the paired denominator — the first cause found."""
    for side, report, values in (("baseline", baseline, base), ("candidate", candidate, cand)):
        label = f"{side} `{report.get('strategy')}`"
        if case_id not in values:
            skipped = next(
                (e for e in report.get("unmeasured", ()) if str(e.get("id")) == case_id), None
            )
            why = (skipped or {}).get("reason") or "el caso no está en el informe"
            return f"no medido en {label}: {why}"
        missing = [name for name in metrics if values[case_id].get(name) is None]
        if missing:
            return f"{', '.join(missing)} ausente o no medido en {label}"
    return "sin causa registrada"  # pragma: no cover - `paired` holds every other member


def _movement(entry: Mapping[str, Any], metric: str, rank_metric: str) -> str:
    return (
        f"{metric} {entry['baseline'][metric]} → {entry['candidate'][metric]}, "
        f"{rank_metric} {entry['baseline'][rank_metric]} → {entry['candidate'][rank_metric]} "
        f"sobre {', '.join(entry['cases'])} (n={len(entry['cases'])})"
    )


# The axes a fusion sweep may move, and the type each is read as: `RRF_K` is an integer in the
# formula, the weights are real.
FUSION_AXES: dict[str, type] = {"rrf_k": int, "w_lexical": float, "w_vector": float}


def parse_fusion_sweep(values: Sequence[str]) -> dict[str, list[float]]:
    """`["rrf_k=10,60 w_vector=0.5,1"]` -> `{"rrf_k": [10, 60], "w_vector": [0.5, 1.0]}`.

    The syntax of `parse_sweep`, and its refusal of an unknown axis: a typo that swept nothing
    would publish the constants in force as the winner of a sweep that never ran.
    """
    grid = _parse_axes(values, FUSION_AXES, "de fusión")
    _check_fusion_grid(grid)
    return grid


def _check_fusion_grid(grid: Mapping[str, Sequence[float]]) -> None:
    """Refuse what `fuse` cannot honour, before a single cell is scored."""
    unknown = sorted(set(grid) - set(FUSION_AXES))
    if unknown:
        raise ValueError(f"Ejes de barrido de fusión desconocidos: {unknown}.")
    for value in grid.get("rrf_k", ()):
        if value < 1:
            raise ValueError(
                f"rrf_k={value} no es válido: RRF divide por RRF_K + rango, y con RRF_K < 1 el "
                "primer puesto divide por cero o por un número negativo."
            )
    for axis in ("w_lexical", "w_vector"):
        for value in grid.get(axis, ()):
            if value < 0:
                raise ValueError(
                    f"{axis}={value} no es válido: un peso negativo PENALIZA que el canal "
                    "encuentre un chunk."
                )


@dataclass(frozen=True)
class FusionRow:
    """One `(RRF_K, w_lexical, w_vector)` cell and what `hybrid` scored under it."""

    rrf_k: int
    w_lexical: float
    w_vector: float
    recall: float | None
    mrr: float | None
    recall_at_1: float | None
    by_stratum: dict[str, Any]
    in_force: bool


@dataclass(frozen=True)
class FusionSweepReport:
    """Every cell, best first; the cell in force in `fusion.py` is always one of them."""

    k: int
    rows: tuple[FusionRow, ...]
    limit: int
    embeddings: dict[str, Any]
    indexing: dict[str, Any]
    unmeasured: tuple[str, ...] = ()

    @property
    def winner(self) -> FusionRow | None:
        """The top row, and only when it scored — the same guard as `SweepReport.winner`."""
        if not self.rows:
            return None
        top = self.rows[0]
        return top if top.recall is not None else None

    @property
    def moves(self) -> bool:
        """Whether the sweep beat the constants in force — the ONLY case `fusion.py` changes."""
        winner = self.winner
        return winner is not None and not winner.in_force

    def to_dict(self) -> dict[str, Any]:
        winner = self.winner
        return {
            "strategy": "hybrid",
            "k": self.k,
            "limit": self.limit,
            "embeddings": self.embeddings,
            "indexing": self.indexing,
            "unmeasured": list(self.unmeasured),
            "winner": (
                None
                if winner is None
                else {
                    "rrf_k": winner.rrf_k,
                    "w_lexical": winner.w_lexical,
                    "w_vector": winner.w_vector,
                }
            ),
            "moves": self.moves,
            "verdict": _fusion_verdict(self),
            "rows": [
                {
                    "rrf_k": row.rrf_k,
                    "w_lexical": row.w_lexical,
                    "w_vector": row.w_vector,
                    f"recall@{self.k}": row.recall,
                    "recall@1": row.recall_at_1,
                    "mrr": row.mrr,
                    "in_force": row.in_force,
                    "by_stratum": row.by_stratum,
                }
                for row in self.rows
            ],
        }


def sweep_fusion(
    cases: Sequence[GoldenCase],
    corpus: Corpus,
    vectors: VectorEvaluation,
    grid: Mapping[str, Sequence[float]],
    *,
    k: int = DEFAULT_SWEEP_K,
    limit: int | None = None,
    params: ChunkerParams = DEFAULT_CHUNKER_PARAMS,
) -> FusionSweepReport:
    """Score `hybrid` at every `(RRF_K, w_lexical, w_vector)` in `grid` (Plan 03 §4.1).

    ONE PLANE AND ONE EMBEDDING PER QUERY, WHATEVER THE GRID. The index is built or reused once
    and every scored query embedded once; each cell then only re-fuses. Unswept axes hold the
    values in force, and the cell IN FORCE is always scored — appended when the grid omits it —
    so «the sweep moves the constants» is always a comparison against a measured current.

    THE CRITERION is the chunker sweep's (S-1): `recall@k`, then MRR; and on a tie the cell in
    force wins, because a tie does not license an edit (spec §13.15 — a flat result is a
    result). `corpus` is accepted for symmetry with `evaluate` and names the population; the
    ranking reads the persisted evaluation index built from the same files.
    """
    from xbrain.knowledge import fusion

    _ = corpus
    _check_fusion_grid(grid)
    depth = max(limit if limit is not None else k, k)
    in_force = (fusion.RRF_K, fusion.CHANNEL_WEIGHTS["lexical"], fusion.CHANNEL_WEIGHTS["vector"])
    run = _open_vector_run(vectors, params=params)
    try:
        measured = [case for case in cases if not unsupported_filters(case.filters, "hybrid")]
        queries = {case.id: _embed_query(run, case)[0] for case in measured}
        rows = [
            _fusion_row(run, measured, queries, cell, in_force=in_force, k=k, depth=depth)
            for cell in _fusion_cells(grid, in_force)
        ]
    finally:
        run.close()
    rows.sort(
        key=lambda row: (
            -(row.recall if row.recall is not None else -1.0),
            -(row.mrr if row.mrr is not None else -1.0),
            not row.in_force,
            row.rrf_k,
            row.w_lexical,
            row.w_vector,
        )
    )
    scored = {case.id for case in measured}
    return FusionSweepReport(
        k=k,
        rows=tuple(rows),
        limit=depth,
        embeddings=run.embeddings,
        indexing=run.indexing,
        unmeasured=tuple(case.id for case in cases if case.id not in scored),
    )


def _fusion_cells(
    grid: Mapping[str, Sequence[float]], in_force: tuple[int, float, float]
) -> list[tuple[int, float, float]]:
    """The cartesian product in the order given, with the cell in force appended if absent."""
    axes = (
        grid.get("rrf_k", [in_force[0]]),
        grid.get("w_lexical", [in_force[1]]),
        grid.get("w_vector", [in_force[2]]),
    )
    cells = [(int(r), float(lex), float(vec)) for r, lex, vec in product(*axes)]
    if cells and in_force not in cells:
        cells.append(in_force)
    return cells


def _fusion_row(
    run: _VectorRun,
    cases: Sequence[GoldenCase],
    queries: Mapping[str, tuple[float, ...]],
    cell: tuple[int, float, float],
    *,
    in_force: tuple[int, float, float],
    k: int,
    depth: int,
) -> FusionRow:
    rrf_k, w_lexical, w_vector = cell
    with _fusion_constants(rrf_k, w_lexical, w_vector):
        results = []
        for case in cases:
            hits, exhausted = _fused_hits(run, case.query, queries[case.id], "hybrid", depth)
            results.append(_score(case, hits, (1, k), depth=depth, depth_exhausted=exhausted))
    # A `None` is an ABSENCE, never a zero — `_measured`'s rule, over results rather than a report.
    recalls = [value for r in results if (value := r.metrics.get(f"recall@{k}")) is not None]
    mrrs = [value for r in results if (value := r.metrics.get("mrr")) is not None]
    firsts = [value for r in results if (value := r.metrics.get("recall@1")) is not None]
    return FusionRow(
        rrf_k=rrf_k,
        w_lexical=w_lexical,
        w_vector=w_vector,
        recall=sum(recalls) / len(recalls) if recalls else None,
        mrr=sum(mrrs) / len(mrrs) if mrrs else None,
        recall_at_1=sum(firsts) / len(firsts) if firsts else None,
        by_stratum=_aggregate(results, STRATA, lambda case: case.strata),
        in_force=cell == in_force,
    )


@contextmanager
def _fusion_constants(rrf_k: int, w_lexical: float, w_vector: float) -> Iterator[None]:
    """Set `fusion.RRF_K` and `fusion.CHANNEL_WEIGHTS` for one cell, and put them BACK.

    Module state, deliberately: `fusion` reads both at CALL time precisely so a measured winner
    takes effect (its docstring), and a parameter added to `fuse` for the sweep's sake would be
    a second way to choose the constants, one `search` never uses. The restore is a `finally`
    because the other failure is silent: every later fusion in the process on the last cell's
    constants.
    """
    from xbrain.knowledge import fusion

    saved = (fusion.RRF_K, fusion.CHANNEL_WEIGHTS)
    fusion.RRF_K = rrf_k
    fusion.CHANNEL_WEIGHTS = {"lexical": w_lexical, "vector": w_vector}
    try:
        yield
    finally:
        fusion.RRF_K, fusion.CHANNEL_WEIGHTS = saved


def render_fusion_sweep_markdown(report: FusionSweepReport) -> str:
    """The fusion table, winner first, with the verdict that says whether `fusion.py` moves."""
    lines = [
        f"Recuperador: `hybrid` · modelo `{report.embeddings.get('model')}`",
        f"Profundidad: {report.limit} owners por caso (U-6).",
        f"Criterio: recall@{report.k}, luego MRR; en empate gana la combinación EN VIGOR "
        "(spec §13.15: un empate no mueve las constantes).",
        f"| RRF_K | w_lexical | w_vector | recall@{report.k} | recall@1 | MRR | en vigor |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report.rows:
        lines.append(
            f"| {row.rrf_k} | {row.w_lexical} | {row.w_vector} | {_number(row.recall)} "
            f"| {_number(row.recall_at_1)} | {_number(row.mrr)} | {'sí' if row.in_force else ''} |"
        )
    if report.unmeasured:
        lines += [
            "",
            f"No medidos (filtros que el plano no aplica): {', '.join(report.unmeasured)}.",
        ]
    lines += ["", _fusion_verdict(report)]
    return "\n".join(lines)


def _fusion_verdict(report: FusionSweepReport) -> str:
    if not report.rows:
        return "SIN COMBINACIONES: el barrido no produjo ninguna fila, así que no hay ganador."
    winner = report.winner
    if winner is None:
        return (
            f"SIN MEDICIÓN: ninguna de las {len(report.rows)} combinaciones pudo puntuarse, así "
            "que no hay ganador."
        )
    label = f"RRF_K={winner.rrf_k}, w_lexical={winner.w_lexical}, w_vector={winner.w_vector}"
    if not report.moves:
        return f"El barrido no mueve las constantes: gana la combinación en vigor ({label})."
    current = next(row for row in report.rows if row.in_force)
    return (
        f"El barrido MUEVE las constantes: gana {label} (recall@{report.k} "
        f"{_number(winner.recall)}, MRR {_number(winner.mrr)}) frente a la combinación en vigor "
        f"(recall@{report.k} {_number(current.recall)}, MRR {_number(current.mrr)})."
    )


# ---------------------------------------------------------------------------
# Plan 04.5 — the graph threshold sweep (Plan 04 §1.3, spec §14)
# ---------------------------------------------------------------------------
#
# «UMBRAL DE COOCURRENCIA DEL GRAFO MÍNIMO: SE MIDE CONTRA EXPANSIÓN ÚTIL/RUIDO» (spec §14). Plan
# 04 §1.3 sweeps `min_shared_items × min_weight` and asks each cell for its co-occurrence edges,
# its mean degree and the recall delta of `hybrid_graph` against `hybrid`; spec §8.4 adds the two
# figures that make «ruido» a number — the precision of the candidates only the graph brought into
# the page, and the places the direct results lost to make room for them.
#
# THROUGH `search`, NOT A SECOND STRATEGY. `hybrid_graph` exists in exactly one place,
# `search_service._graph_order`, and a copy of it here would be the rule-5 divergence inside the
# module that exists to measure the first copy. So each cell derives its graph plane with the
# writer `xbrain index build` uses — a build for the first cell, then `index_build.update`, which
# rewrites the graph alone when only a threshold moved — and scores what `search` serves. The unit
# is the ITEM, because that is what `search` pages: a case whose truth is a topic is unmeasured,
# and none of these figures is comparable with the owner-level ones `evaluate` publishes.
#
# THE BASE IS `hybrid` AS `search` ANSWERS IT OVER THIS INDEX. The sweep writes no vector plane —
# one per cell would re-embed the corpus for a threshold that does not touch a single vector — so
# `hybrid` answers `lexical` and `hybrid_graph` re-ranks that same ranking, each declaring why, and
# the report carries both declarations. The graph's term joins at item level over the rank the
# base served, so the delta is the graph's and nothing else's.
#
# THE RULE IS FIXED HERE, BEFORE ANY CELL IS READ (`rank_graph_rows`), and pinned by tests on
# constructed rows so that no measurement can move it.

# Plan 04 §3 defines «degradar materialmente la precisión» before measuring, «para que no se
# renegocie con el resultado delante»: a fall of MORE than 3 pp of `precision@10` in any stratum.
MATERIAL_PRECISION_DROP_PP: float = 3.0

# The axes a graph sweep may move: the two `build_graph_edges` thresholds `[index]` configures.
GRAPH_AXES: dict[str, type] = {"min_shared_items": int, "min_weight": float}

GRAPH_SWEEP_CRITERION = (
    f"descarta la combinación que pierde más de {MATERIAL_PRECISION_DROP_PP:g} pp de precisión "
    "en algún estrato (Plan 04 §3); entre las demás decide Δ recall frente a `hybrid`, luego "
    "menos ruido (entrantes no relevantes), luego menos degradación de los resultados directos, "
    "luego el grafo más disperso y, entre grafos idénticos, los umbrales menos restrictivos"
)

_CO_OCCURRENCE_EDGES_SQL = "SELECT COUNT(*) FROM graph_edges WHERE relation = 'CO_OCCURS_WITH'"
_TOPIC_NODES_SQL = (
    "SELECT COUNT(DISTINCT target) FROM graph_edges WHERE relation != 'CO_OCCURS_WITH'"
)


def parse_graph_sweep(values: Sequence[str]) -> dict[str, list[float]]:
    """`["min_shared_items=2,3 min_weight=0.0,0.05"]` -> `{"min_shared_items": [2, 3], ...}`.

    The syntax and the unknown-axis refusal of `parse_fusion_sweep`, through the same parser.
    """
    grid = _parse_axes(values, GRAPH_AXES, "del grafo")
    _check_graph_grid(grid)
    return grid


def _parse_axes(
    values: Sequence[str], axes: Mapping[str, type], noun: str
) -> dict[str, list[float]]:
    """`clave=v1,v2` tokens, each value read as its axis's type; an unknown axis is refused."""
    grid: dict[str, list[float]] = {}
    for value in values:
        for token in value.split():
            if "=" not in token:
                raise ValueError(f"Formato de barrido inválido: {token!r}. Usa `clave=v1,v2`.")
            key, raw = token.split("=", 1)
            reader = axes.get(key)
            if reader is None:
                raise ValueError(
                    f"Eje de barrido {noun} desconocido: {key!r}. Válidos: {', '.join(axes)}."
                )
            grid[key] = [reader(part) for part in raw.split(",") if part.strip()]
    return grid


def _check_graph_grid(grid: Mapping[str, Sequence[float]]) -> None:
    """Refuse a threshold `build_graph_edges` would not apply as written, before any build."""
    unknown = sorted(set(grid) - set(GRAPH_AXES))
    if unknown:
        raise ValueError(f"Ejes de barrido del grafo desconocidos: {unknown}.")
    for value in grid.get("min_shared_items", ()):
        if value < 1:
            raise ValueError(
                f"min_shared_items={value} no es válido: `build_graph_edges` trata un valor < 1 "
                "como 1, así que la celda mediría la de 1 con otro nombre."
            )
    for value in grid.get("min_weight", ()):
        if not 0 <= value <= 1:
            raise ValueError(
                f"min_weight={value} no es válido: el peso es un índice de Jaccard, en [0, 1], y "
                "fuera de ese rango la celda lo conserva todo o no conserva nada."
            )


@dataclass(frozen=True)
class GraphSweepRow:
    """One `(min_shared_items, min_weight)` cell: the graph it persisted and what the graph did.

    `entrants` are the items in `hybrid_graph`'s top k that `hybrid`'s top k did not hold — the
    candidates only the graph brought in, since the graph adds a term and never takes one away;
    `useful` is how many of them are relevant. `degradation` is the mean number of places a
    direct result (one already in `hybrid`'s top k) lost. `precision_drops` is, per stratum
    measured by both, how many percentage points of `precision@k` the graph cost (negative when
    it gained). All three are summed or averaged over the SAME measured cases as the recall.
    """

    min_shared_items: int
    min_weight: float
    edges: int
    mean_degree: float
    recall: float | None
    recall_delta: float | None
    entrants: int
    useful: int
    degradation: float | None
    precision_drops: dict[str, float]
    graph_ran: bool
    in_force: bool = False
    degraded: tuple[str, ...] = ()

    @property
    def noise(self) -> int:
        """Entrants that are not relevant: what the graph displaced the direct results for."""
        return self.entrants - self.useful

    @property
    def entrant_precision(self) -> float | None:
        """Spec §8.4's «precisión de candidatos añadidos exclusivamente por grafo». 0/0 is `None`."""
        return self.useful / self.entrants if self.entrants else None

    @property
    def rejected_for(self) -> tuple[str, ...]:
        """Why this cell cannot be a USEFUL winner — empty when nothing disqualifies it."""
        if not self.graph_ran:
            return ("el grafo no corrió: `hybrid_graph` respondió otra estrategia en algún caso",)
        return tuple(
            f"la precisión cae {drop:.2f} pp en `{stratum}` "
            f"(> {MATERIAL_PRECISION_DROP_PP:g} pp, Plan 04 §3)"
            for stratum, drop in sorted(self.precision_drops.items())
            if drop > MATERIAL_PRECISION_DROP_PP
        )


def rank_graph_rows(rows: Iterable[GraphSweepRow]) -> tuple[GraphSweepRow, ...]:
    """The cells in the order the rule prefers them — `GRAPH_SWEEP_CRITERION`, as a sort key.

    A cell where the graph did not run measured another strategy and ranks last. A cell losing
    more than `MATERIAL_PRECISION_DROP_PP` in any stratum ranks after every cell that does not.
    Within each group: the larger recall delta, then less noise, then less degradation, then the
    sparser graph, then the lower `min_shared_items` and `min_weight` — two cells that persisted
    the same graph measured the same thing, and the constraint that changed nothing is not the
    one to apply.
    """

    def key(row: GraphSweepRow) -> tuple[bool, bool, float, int, float, int, int, float]:
        return (
            not row.graph_ran,
            bool(row.rejected_for),
            -row.recall_delta if row.recall_delta is not None else math.inf,
            row.noise,
            row.degradation if row.degradation is not None else math.inf,
            row.edges,
            row.min_shared_items,
            row.min_weight,
        )

    return tuple(sorted(rows, key=key))


@dataclass(frozen=True)
class GraphSweepReport:
    """Every cell, in the rule's order, with the two retrievers it compared and on what."""

    k: int
    limit: int
    rows: tuple[GraphSweepRow, ...]
    base: dict[str, Any] = field(default_factory=dict)
    graph: dict[str, Any] = field(default_factory=dict)
    corpus: dict[str, Any] = field(default_factory=dict)
    measured_cases: tuple[str, ...] = ()
    unmeasured: tuple[dict[str, Any], ...] = ()

    @property
    def winner(self) -> GraphSweepRow | None:
        """The threshold to APPLY: the first cell, when the graph ran in it and it was scored.

        Applied even when it helps nothing — the index always builds a graph, so some threshold
        is always in force — which is why `useful` is a separate question.
        """
        if not self.rows:
            return None
        top = self.rows[0]
        return top if top.graph_ran and top.recall_delta is not None else None

    @property
    def useful(self) -> bool:
        """Whether the winner improves recall without a material loss of precision (Plan 04 §3)."""
        winner = self.winner
        return (
            winner is not None
            and not winner.rejected_for
            and winner.recall_delta is not None
            and winner.recall_delta > 0
        )

    @property
    def moves(self) -> bool:
        """Whether the winner is not the cell in force — the only case the defaults change."""
        winner = self.winner
        return winner is not None and not winner.in_force

    def to_dict(self) -> dict[str, Any]:
        winner = self.winner
        return {
            "strategy": "hybrid_graph",
            "k": self.k,
            "limit": self.limit,
            "base": self.base,
            "graph": self.graph,
            "corpus": self.corpus,
            "material_precision_drop_pp": MATERIAL_PRECISION_DROP_PP,
            "criterion": GRAPH_SWEEP_CRITERION,
            "measured_cases": list(self.measured_cases),
            "unmeasured": [dict(entry) for entry in self.unmeasured],
            "winner": (
                None
                if winner is None
                else {
                    "min_shared_items": winner.min_shared_items,
                    "min_weight": winner.min_weight,
                }
            ),
            "useful": self.useful,
            "moves": self.moves,
            "verdict": _graph_verdict(self),
            "rows": [
                {
                    "min_shared_items": row.min_shared_items,
                    "min_weight": row.min_weight,
                    "edges": row.edges,
                    "mean_degree": row.mean_degree,
                    f"recall@{self.k}": row.recall,
                    "recall_delta": row.recall_delta,
                    "entrants": row.entrants,
                    "useful": row.useful,
                    "noise": row.noise,
                    "entrant_precision": row.entrant_precision,
                    "degradation": row.degradation,
                    "precision_drops": row.precision_drops,
                    "rejected_for": list(row.rejected_for),
                    "graph_ran": row.graph_ran,
                    "in_force": row.in_force,
                    "degraded": list(row.degraded),
                }
                for row in self.rows
            ],
        }


@dataclass(frozen=True)
class _Rankings:
    """What `search` served for every measured case under one requested strategy."""

    requested: str
    ids: dict[str, tuple[str, ...]]
    strategies: tuple[str, ...]
    degraded: tuple[str, ...]

    def describe(self) -> dict[str, Any]:
        return {
            "requested_strategy": self.requested,
            "strategy": ", ".join(self.strategies) or self.requested,
            "degraded": list(self.degraded),
        }


@dataclass(frozen=True)
class _GraphCase:
    recall: float
    base_recall: float
    precision: float | None
    base_precision: float | None
    entrants: int
    useful: int
    lost: tuple[int, ...]


def sweep_graph(
    cases: Sequence[GoldenCase],
    grid: Mapping[str, Sequence[float]],
    *,
    items_path: Path,
    vocab_path: Path,
    topics_path: Path,
    index_dir: Path,
    k: int = DEFAULT_SWEEP_K,
    limit: int | None = None,
) -> GraphSweepReport:
    """Score `hybrid_graph` against `hybrid` at every `(min_shared_items, min_weight)` in `grid`.

    `index_dir` is the sweep's OWN index, rebuilt from the three inputs: never `data/index/`,
    which belongs to `search`. The store is read once, as one snapshot, and never written. The
    cell in force (`graph_build`'s defaults, read at call time) is always scored, appended when
    the grid omits it, so «the sweep moves the default» is a comparison against a measurement.
    `hybrid` is ranked once — no threshold touches it — and `hybrid_graph` once per cell.
    """
    from xbrain.knowledge import graph_build
    from xbrain.knowledge.index_build import load_index_inputs
    from xbrain.knowledge.search_service import QueryContext

    _check_graph_grid(grid)
    depth = max(limit if limit is not None else k, k)
    in_force = (
        int(graph_build.DEFAULT_GRAPH_MIN_SHARED_ITEMS),
        float(graph_build.DEFAULT_GRAPH_MIN_WEIGHT),
    )
    measured, unmeasured = _graph_population(cases)
    inputs = load_index_inputs(items_path, vocab_path, topics_path)
    context = QueryContext(
        store=inputs.store,
        vocab=inputs.vocab,
        topic_pages=inputs.topic_pages,
        index_dir=index_dir,
        items_path=items_path,
        vocab_path=vocab_path,
        topics_path=topics_path,
    )
    rows: list[GraphSweepRow] = []
    base: _Rankings | None = None
    corpus: dict[str, Any] = {}
    for position, cell in enumerate(_graph_cells(grid, in_force)):
        corpus = _derive_graph(index_dir, inputs, cell, rebuild=position == 0)
        base = base or _rankings(measured, context, depth, "hybrid")
        graph = _rankings(measured, context, depth, "hybrid_graph")
        stats = _graph_edge_stats(context)
        rows.append(
            _graph_row(cell, measured, base, graph, stats, in_force=in_force, k=k, depth=depth)
        )
    return GraphSweepReport(
        k=k,
        limit=depth,
        rows=rank_graph_rows(rows),
        base=_base_block(measured, base, k),
        graph=graph.describe() if rows else {"requested_strategy": "hybrid_graph"},
        corpus={"items_path": str(items_path), **corpus},
        measured_cases=tuple(case.id for case in measured),
        unmeasured=tuple(unmeasured),
    )


def _graph_population(
    cases: Sequence[GoldenCase],
) -> tuple[list[GoldenCase], list[dict[str, Any]]]:
    """The cases `search` can score for the graph, and every other one with its reason."""
    measured: list[GoldenCase] = []
    unmeasured: list[dict[str, Any]] = []
    for case in cases:
        blocked = unsupported_filters(case.filters, "hybrid_graph")
        if blocked:
            reason = (
                f"`hybrid_graph` no puede aplicar {list(blocked)}: puntuar el caso sería "
                "fabricar un cero (spec §8.6.8)"
            )
        elif not case.relevant_items:
            reason = (
                "la verdad del caso son topics y `search` sirve items: su recall sería 0/0, "
                "no 0,0 (spec §8.6.8)"
            )
        else:
            measured.append(case)
            continue
        unmeasured.append(
            {
                "id": case.id,
                "strata": list(case.strata),
                "provenance": case.provenance,
                "reason": reason,
            }
        )
    return measured, unmeasured


def _graph_cells(
    grid: Mapping[str, Sequence[float]], in_force: tuple[int, float]
) -> list[tuple[int, float]]:
    """The cartesian product in the order given, with the cell in force appended if absent."""
    axes = (
        grid.get("min_shared_items", [in_force[0]]),
        grid.get("min_weight", [in_force[1]]),
    )
    cells = [(int(shared), float(weight)) for shared, weight in product(*axes)]
    if cells and in_force not in cells:
        cells.append(in_force)
    return cells


def _derive_graph(
    index_dir: Path, inputs: Any, cell: tuple[int, float], *, rebuild: bool
) -> dict[str, Any]:
    """Write the graph plane for `cell` and PROVE it: the manifest must seal these thresholds.

    Without the proof a plane that failed to move would be measured under every cell's name,
    publishing one graph as a flat table of sixteen. Returns the fingerprints of what was built.
    """
    from xbrain.knowledge import index_build

    options = index_build.IndexOptions(graph_min_shared_items=cell[0], graph_min_weight=cell[1])
    if rebuild:
        index_build.build(index_dir, inputs, options=options, force=True)
    else:
        index_build.update(index_dir, inputs, options=options)
    manifest = index_build.load_manifest(index_dir)
    sealed = (manifest.graph["min_shared_items"], manifest.graph["min_weight"])
    if sealed != cell:
        raise ValueError(
            f"la combinación min_shared_items={cell[0]}, min_weight={cell[1]} no se midió: el "
            f"manifest de {index_dir} sella min_shared_items={sealed[0]}, "
            f"min_weight={sealed[1]}, y medir otra vez el mismo grafo publicaría una fila falsa."
        )
    return {
        "items": len(inputs.store),
        "topics": len(inputs.vocab),
        "store_fingerprint": manifest.store_fingerprint,
        "vocab_fingerprint": manifest.vocab_fingerprint,
        "topics_fingerprint": manifest.topics_fingerprint,
    }


def _rankings(
    cases: Sequence[GoldenCase], context: Any, depth: int, strategy: Strategy
) -> _Rankings:
    """The item ranking `search` serves each case at `depth`, and what it said about itself."""
    from xbrain.knowledge.search_service import search

    ids: dict[str, tuple[str, ...]] = {}
    strategies: dict[str, None] = {}
    degraded: dict[str, None] = {}
    for case in cases:
        response = search(
            case.query,
            context,
            filters=case.filters,
            limit=depth,
            strategy=strategy,
            graph_enabled=True,
        )
        ids[case.id] = tuple(result.item_id for result in response.results)
        strategies[response.strategy] = None
        degraded.update(dict.fromkeys(response.index.degraded))
    return _Rankings(strategy, ids, tuple(strategies), tuple(degraded))


def _graph_edge_stats(context: Any) -> tuple[int, float]:
    """The co-occurrence edges the cell PERSISTED, and their mean per topic node."""
    from xbrain.knowledge.index_store import open_for_query

    index = open_for_query(
        context.index_dir, context.items_path, context.vocab_path, context.topics_path
    )
    try:
        connection = index.lexical.connection
        edges = int(connection.execute(_CO_OCCURRENCE_EDGES_SQL).fetchone()[0])
        topics = int(connection.execute(_TOPIC_NODES_SQL).fetchone()[0])
    finally:
        index.close()
    return edges, (edges / topics if topics else 0.0)


def _graph_row(
    cell: tuple[int, float],
    cases: Sequence[GoldenCase],
    base: _Rankings,
    graph: _Rankings,
    stats: tuple[int, float],
    *,
    in_force: tuple[int, float],
    k: int,
    depth: int,
) -> GraphSweepRow:
    results = [
        _graph_case(case, base.ids[case.id], graph.ids[case.id], k=k, depth=depth) for case in cases
    ]
    recall = _mean([result.recall for result in results])
    base_recall = _mean([result.base_recall for result in results])
    return GraphSweepRow(
        min_shared_items=cell[0],
        min_weight=cell[1],
        edges=stats[0],
        mean_degree=stats[1],
        recall=recall,
        recall_delta=None if recall is None or base_recall is None else recall - base_recall,
        entrants=sum(result.entrants for result in results),
        useful=sum(result.useful for result in results),
        degradation=_mean([places for result in results for places in result.lost]),
        precision_drops=_precision_drops(cases, results),
        graph_ran=bool(cases) and graph.strategies == ("hybrid_graph",),
        in_force=cell == in_force,
        degraded=graph.degraded,
    )


def _graph_case(
    case: GoldenCase,
    base_ids: Sequence[str],
    graph_ids: Sequence[str],
    *,
    k: int,
    depth: int,
) -> _GraphCase:
    """One case's figures. A direct result pushed out of `depth` counts as place `depth + 1`."""
    relevant = set(case.relevant_items)
    base_top, graph_top = list(base_ids[:k]), list(graph_ids[:k])
    entrants = set(graph_top) - set(base_top)
    place = {item_id: rank for rank, item_id in enumerate(graph_ids[:depth], start=1)}
    return _GraphCase(
        recall=len(relevant & set(graph_top)) / len(relevant),
        base_recall=len(relevant & set(base_top)) / len(relevant),
        precision=len(relevant & set(graph_top)) / len(graph_top) if graph_top else None,
        base_precision=len(relevant & set(base_top)) / len(base_top) if base_top else None,
        entrants=len(entrants),
        useful=len(entrants & relevant),
        lost=tuple(
            max(0, place.get(item_id, depth + 1) - rank)
            for rank, item_id in enumerate(base_top, start=1)
        ),
    )


def _precision_drops(
    cases: Sequence[GoldenCase], results: Sequence[_GraphCase]
) -> dict[str, float]:
    """Percentage points of `precision@k` lost per stratum, over the cases measuring both."""
    pairs: dict[str, list[tuple[float, float]]] = {}
    for case, result in zip(cases, results, strict=True):
        if result.precision is None or result.base_precision is None:
            continue
        for stratum in case.strata:
            pairs.setdefault(stratum, []).append((result.base_precision, result.precision))
    return {
        stratum: round(sum(before - after for before, after in values) / len(values) * 100, 4)
        for stratum, values in sorted(pairs.items())
    }


def _base_block(cases: Sequence[GoldenCase], base: _Rankings | None, k: int) -> dict[str, Any]:
    """`hybrid` as it answered, with its recall over the same measured cases every row reads."""
    if base is None:
        return {"requested_strategy": "hybrid"}
    recalls = [
        len(set(case.relevant_items) & set(base.ids[case.id][:k])) / len(case.relevant_items)
        for case in cases
    ]
    return {**base.describe(), "recall": _mean(recalls)}


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def render_graph_sweep_markdown(report: GraphSweepReport) -> str:
    """The graph table, in the rule's order, EVERY cell included (Plan 04 §1.3), and the verdict."""
    base = report.base
    k = report.k
    lines = [
        "Recuperador: `hybrid_graph` sobre "
        + retriever_label(
            str(base.get("strategy", "hybrid")),
            str(base.get("requested_strategy", "hybrid")),
            tuple(base.get("degraded", ())),
        ),
        f"Base: `hybrid` · recall@{k} {_number(base.get('recall'))} sobre "
        f"{len(report.measured_cases)} casos medidos.",
        f"Profundidad: {report.limit} items por caso; un resultado directo expulsado de esa "
        f"profundidad cuenta como puesto {report.limit + 1}.",
        f"Criterio (fijado antes de medir): {GRAPH_SWEEP_CRITERION}.",
        f"| min_shared_items | min_weight | aristas | grado medio | recall@{k} | Δ recall@{k} "
        "| entrantes | útiles | ruido | precisión entrantes | degradación | descartada por "
        "| en vigor |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|:---:|",
    ]
    lines += [_graph_table_row(row) for row in report.rows]
    if report.unmeasured:
        lines += [
            "",
            "No medidos: "
            + "; ".join(f"{entry['id']} ({entry['reason']})" for entry in report.unmeasured)
            + ".",
        ]
    lines += ["", _graph_verdict(report)]
    return "\n".join(lines)


def _graph_table_row(row: GraphSweepRow) -> str:
    cells = [
        str(row.min_shared_items),
        str(row.min_weight),
        str(row.edges),
        f"{row.mean_degree:.2f}",
        _number(row.recall),
        _signed(row.recall_delta),
        str(row.entrants),
        str(row.useful),
        str(row.noise),
        _number(row.entrant_precision),
        _number(row.degradation),
        "; ".join(row.rejected_for),
        "sí" if row.in_force else "",
    ]
    return "| " + " | ".join(cells) + " |"


def _graph_verdict(report: GraphSweepReport) -> str:
    if not report.rows:
        return "SIN COMBINACIONES: el barrido no produjo ninguna fila, así que no hay umbral."
    winner = report.winner
    if winner is None:
        return (
            f"SIN MEDICIÓN: `hybrid_graph` no corrió en ninguna de las {len(report.rows)} "
            "combinaciones, así que no hay umbral que aplicar."
        )
    label = f"min_shared_items={winner.min_shared_items}, min_weight={winner.min_weight}"
    figures = (
        f"Δ recall@{report.k} {_signed(winner.recall_delta)}, ruido {winner.noise}, "
        f"degradación {_number(winner.degradation)} puestos"
    )
    ceiling = f"{MATERIAL_PRECISION_DROP_PP:g} pp"
    if report.useful:
        return (
            f"Gana {label}: {figures} frente a `hybrid`, sin perder más de {ceiling} de precisión "
            "en ningún estrato — cumple la regla de promoción del Plan 04 §3; promover "
            "`hybrid_graph` es una decisión aparte."
        )
    return (
        f"NINGUNA COMBINACIÓN APORTA: ninguna mejora recall@{report.k} frente a `hybrid` sin "
        f"perder más de {ceiling} de precisión en algún estrato. Se aplica {label}, la primera "
        f"por la regla ({figures}), porque el índice siempre construye un grafo; `hybrid_graph` "
        "NO se promueve (Plan 04 §3)."
    )


def _signed(value: float | None) -> str:
    return "sin cobertura" if value is None else f"{value:+.4f}"
