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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from xbrain.knowledge.chunking import ChunkerParams, DEFAULT_CHUNKER_PARAMS, chunk_surfaces
from xbrain.knowledge.contracts import SearchFilters, resolve_strategy
from xbrain.knowledge.goldenset import STRATA, GoldenCase, GoldenScenario
from xbrain.knowledge.index_schema import open_memory_index
from xbrain.knowledge.lexical import LexicalHit, LexicalIndex
from xbrain.knowledge.models import KnowledgeChunk
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
SUPPORTED_FILTERS: dict[str, frozenset[str]] = {
    "lexical": frozenset(SearchFilters.model_fields),
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
            "limit": self.limit,
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
) -> EvaluationReport:
    """Score every case and aggregate by stratum and by provenance.

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
    executed, degraded = resolve_strategy(strategy)
    depth = max(limit or 0, max(ks))
    index, stats = build_index(corpus, params=params)
    results: list[CaseResult] = []
    unmeasured: list[dict[str, Any]] = []
    latencies: list[float] = []
    try:
        for case in cases:
            blocked = unsupported_filters(case.filters, executed)
            if blocked:
                unmeasured.append(
                    {
                        "id": case.id,
                        "strata": list(case.strata),
                        "provenance": case.provenance,
                        "unsupported_filters": list(blocked),
                        "reason": (
                            f"la estrategia `{executed}` no puede aplicar {list(blocked)}; "
                            "puntuar el caso sería fabricar un cero (spec §8.6.8)"
                        ),
                    }
                )
                continue
            started = time.perf_counter()
            hits, exhausted = _search(index, case, owners=depth)
            latencies.append((time.perf_counter() - started) * 1000)
            results.append(_score(case, hits, ks, depth=depth, depth_exhausted=exhausted))
    finally:
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
        latency=_percentiles(latencies),
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
        metrics[f"surface_recall@{k}"] = _surface_recall(case, hits, k)
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
        "> `vacíos` cuenta los casos del bucket cuya consulta no recuperó NI UN CHUNK. Un 0,0",
        "> con `vacíos = casos` no dice que el recuperador ordenase mal: dice que no llegó a",
        "> ordenar nada. Sobre esos casos `precision@k` sale *no medida*, nunca 0,0 — su",
        "> numerador es 0 por construcción y repetiría el conjunto vacío (M3).",
        "",
    ]
    lines += _table("Por estrato", report.by_stratum)
    lines += _table("Por procedencia", report.by_provenance)
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


def _table(title: str, buckets: dict[str, Any]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| bucket | casos | vacíos | recall@1 | recall@10 | precision@10 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in buckets.items():
        if values == NO_COVERAGE:
            lines.append(f"| {name} | — | — | sin cobertura | sin cobertura | sin cobertura | — |")
            continue
        cells = [
            _cell(values, "recall@1"),
            _cell(values, "recall@10"),
            _cell(values, "precision@10"),
            _cell(values, "mrr"),
        ]
        lines.append(
            f"| {name} | {values['cases']} | {values['no_results']} | " + " | ".join(cells) + " |"
        )
    lines.append("")
    return lines


def _cell(values: dict[str, Any], name: str) -> str:
    """One metric cell — words when nobody measured it, never a number (spec §8.6.8).

    The markdown is where a fabricated zero does its damage, because it is the surface that
    gets read and quoted. A `0.0` here is indistinguishable from a measured failure.
    """
    value = values.get(name)
    if value is None:
        return "—"
    if value == NO_COVERAGE:
        return "sin cobertura"
    return str(value)


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
