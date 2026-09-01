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
from xbrain.knowledge.goldenset import STRATA, GoldenCase, GoldenScenario
from xbrain.knowledge.contracts import SearchFilters, resolve_strategy
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
# scored. No case in the golden set declares `has_surfaces`, so the previous code's mapping of
# it onto a chunk-level surface restriction — a different question from the contract's *the
# item HAS this surface* — was a no-op on the real data and its correction moves nothing.
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

    `limit` is how deep the retriever is asked to go. It defaults to `max(ks)`, because
    asking for fewer results than the largest k being reported would make that k's recall a
    measurement of the LIMIT rather than of the retriever — a number that cannot come out any
    other way. A caller may raise it to see whether a miss is a ranking problem or an absence.
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
            hits = _search(index, case, limit=depth)
            latencies.append((time.perf_counter() - started) * 1000)
            results.append(_score(case, hits, ks))
    finally:
        # The `:memory:` index lives exactly as long as the scoring (M-1). `sweep_chunker`
        # calls this once per combination; a twelve-cell sweep used to leak twelve handles.
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
    )


def _search(index: LexicalIndex, case: GoldenCase, limit: int) -> tuple[LexicalHit, ...]:
    """Run one case's query, applying its filters BEFORE scoring (spec §5.3).

    The filters a case declares are part of the case (spec §8.1) — v1 kept windows under a
    key no loader read, so a temporal case silently became an untemporal one and its result
    was reported as though the window had been applied.

    ALL EIGHT are passed now, unchanged, because the persisted schema can push all eight into
    `WHERE`. Passing `case.filters` whole rather than reconstructing a subset is what keeps
    `SUPPORTED_FILTERS` an honest declaration instead of a list that has to be kept in step
    with a second one here.
    """
    return index.search(case.query, limit=limit, filters=case.filters)


def _owner_key(owner_type: str, owner_id: str) -> str:
    return f"{owner_type}:{owner_id}"


def _score(case: GoldenCase, hits: Sequence[LexicalHit], ks: tuple[int, ...]) -> CaseResult:
    """Recall/precision/MRR over OWNERS, plus surface recall (spec §8.4).

    Owners, not chunks: spec §5.4 groups by item, so a transcript matching in six windows is
    one retrieved item, not six. Deduplicated by FIRST occurrence, which preserves the rank
    the best chunk earned.
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
        retrieved=tuple(ranked[: max(ks)]),
        metrics=metrics,
        no_results=not hits,
    )


def _surface_recall(case: GoldenCase, hits: Sequence[LexicalHit], k: int) -> float | None:
    """How many of the case's named surfaces appear among the top-k retrieved CHUNKS.

    Spec §8.4 asks for this alongside item recall because returning the right item through
    the wrong surface is a different, usually worse, answer: the evidence a consumer would
    open is not the evidence the fact is in. Item recall alone scores that as a success.

    THE UNIT IS THE CHUNK, and `recall@k`'s unit is the deduplicated OWNER (m6). Under one
    label `k` they therefore count different things: with `depth = max(limit, max(ks))`,
    `recall@10` can be formed from more than ten chunks (ten distinct owners may take more
    than ten hits to accumulate) while `surface_recall@10` never sees past the tenth chunk.
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


def render_markdown(report: EvaluationReport) -> str:
    """The human report. Publishes failures and gaps, never fabricated zeros (spec §8.6.8)."""
    heading = f"# Evaluación de recuperación — `{report.strategy}`"
    if report.degraded:
        heading += (
            f" · solicitada `{report.requested_strategy}`, sin backend "
            f"({', '.join(report.degraded)}): las cifras son del recuperador que SÍ corrió"
        )
    lines = [
        heading,
        "",
        f"- Corpus: `{report.corpus['source']}` — {report.corpus['items']} items, "
        f"{report.corpus['topics']} topics, {report.corpus['surfaces']} superficies, "
        f"{report.corpus['chunks']} chunks emitidos, "
        f"{report.corpus.get('chunks_not_indexed', 0)} rechazados por el índice.",
        f"- Umbral: {report.threshold if report.threshold is not None else 'ninguno (solo informe)'}.",
        f"- Latencia p50 {report.latency['p50_ms']} ms · p95 {report.latency['p95_ms']} ms.",
        "",
        "> Las cifras de arriba son una fotografía del corpus medido, no una constante del",
        "> producto. Vuelve a derivarlas al ejecutar (CLAUDE.md regla 2).",
        "",
        "> `recall@k` cuenta OWNERS deduplicados; `surface_recall@k` cuenta CHUNKS (m6). Bajo",
        "> la misma `k` no miden la misma población: con `depth = max(limit, max(ks))`, diez",
        "> owners distintos pueden necesitar más de diez chunks, mientras que",
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
# The chunker sweep (Plan 02 §7)
# ---------------------------------------------------------------------------
#
# Spec §5.2: the chunk size and overlap *se eligen con evaluación*. This is that evaluation.
#
# AND IT CANNOT TOUCH THE CHARACTERIZATION FIXTURE (M7). The sweep changes
# `DEFAULT_CHUNKER_PARAMS`; `tests/fixtures/knowledge_ranking.json` is built with parameters
# passed as an ARGUMENT, so the two never meet. That separation is the whole reason
# `ChunkerParams` is a parameter rather than a module constant — without it, the sweep would
# break the fixture that exists to pin the ranking, and the comfortable fix would be to
# regenerate it, at which point it pins nothing.


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


@dataclass(frozen=True)
class SweepReport:
    """Every combination, best first, with the k the ranking was decided on."""

    k: int
    rows: tuple[SweepRow, ...]

    @property
    def winner(self) -> SweepRow | None:
        return self.rows[0] if self.rows else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "rows": [
                {
                    "target": row.params.target,
                    "max_chars": row.params.max_chars,
                    "overlap": row.params.overlap,
                    "min_chars": row.params.min_chars,
                    "chunks": row.chunks,
                    f"recall@{self.k}": row.recall,
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
    k: int = 10,
    base: ChunkerParams = DEFAULT_CHUNKER_PARAMS,
) -> SweepReport:
    """Score every combination in `grid` against the golden set (Plan 02 §7).

    Ranked by `recall@k`, tie-broken by MRR and then by FEWER CHUNKS — spec §13.15 says a flat
    result is documented rather than hidden, and "if they tie, take the one that produces less
    index" is the honest tie-break rather than a preference dressed as a finding.

    A combination that scores nothing measurable sorts last instead of sorting first, which is
    what a `None` would do under a naive `max`.
    """
    rows: list[SweepRow] = []
    for params in _combinations(grid, base):
        report = evaluate(cases, corpus, strategy=strategy, ks=(k,), params=params)
        overall = _overall(report, k)
        rows.append(
            SweepRow(
                params=params,
                chunks=report.index_stats.chunks if report.index_stats else 0,
                recall=overall[0],
                mrr=overall[1],
                by_stratum=report.by_stratum,
            )
        )
    rows.sort(key=lambda row: (-(row.recall or -1.0), -(row.mrr or -1.0), row.chunks))
    return SweepReport(k=k, rows=tuple(rows))


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
        f"| target | overlap | chunks | recall@{report.k} | MRR |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report.rows:
        lines.append(
            f"| {row.params.target} | {row.params.overlap} | {row.chunks} "
            f"| {_number(row.recall)} | {_number(row.mrr)} |"
        )
    winner = report.winner
    if winner is not None:
        distinct = {(_number(r.recall), _number(r.mrr)) for r in report.rows}
        verdict = (
            "PLANO: todas las combinaciones puntúan igual; gana la que produce menos chunks."
            if len(distinct) == 1
            else f"Gana target={winner.params.target}, overlap={winner.params.overlap}."
        )
        lines += ["", verdict]
    return "\n".join(lines)


def _number(value: float | None) -> str:
    return "sin cobertura" if value is None else f"{value:.4f}"
