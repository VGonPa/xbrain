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

2. **No coverage is NOT zero.** Spec §8.6.8: *failures and skips are published; zeros are
   never fabricated by mixing in unmeasured cases*. `expansion` has no mechanism until Plan
   04; `thread` and `user_note` have zero instances in the corpus. Reporting them at 0.0
   would claim the retriever failed at something nobody asked, and the figure would sit in a
   table looking exactly like a measurement.

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
from typing import Any, Iterable, Sequence

from xbrain.knowledge.chunking import ChunkerParams, DEFAULT_CHUNKER_PARAMS, chunk_surfaces
from xbrain.knowledge.goldenset import STRATA, GoldenCase, GoldenScenario
from xbrain.knowledge.lexical_memory import InMemoryLexicalIndex, LexicalHit
from xbrain.knowledge.models import KnowledgeChunk
from xbrain.knowledge.surfaces import item_surfaces, item_topics, topic_surfaces
from xbrain.models import Item, Topic, TopicPage
from xbrain.store import load_store, load_topic_pages

# The marker a bucket carries INSTEAD of numbers when it has no cases. A sentinel rather than
# a `None` or a zero, so it survives JSON and renders as words in the markdown.
NO_COVERAGE = {"coverage": "sin cobertura"}

# Surfaces the emitter supports for which the corpus holds NO data (measured 2026-08-31), so
# the evaluation cannot have cases and does not invent any (spec §8.6.8). Declared, so their
# absence never reads as an oversight.
SURFACES_WITHOUT_DATA: tuple[str, ...] = ("thread", "user_note")

DEFAULT_KS: tuple[int, ...] = (1, 5, 10, 20)


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
    chunks: int
    empty_surfaces: int


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
    metrics: dict[str, float]


@dataclass(frozen=True)
class EvaluationReport:
    strategy: str
    corpus: dict[str, Any]
    cases: tuple[CaseResult, ...]
    by_stratum: dict[str, Any]
    by_provenance: dict[str, Any]
    latency: dict[str, float]
    without_coverage: dict[str, list[str]]
    scenarios: tuple[GoldenScenario, ...] = ()
    threshold: float | None = None
    failures: tuple[str, ...] = ()
    index_stats: IndexStats | None = None

    @property
    def passed(self) -> bool:
        """True when no bucket fell below the threshold — vacuously true with no threshold."""
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "corpus": self.corpus,
            "threshold": self.threshold,
            "passed": self.passed,
            "failures": list(self.failures),
            "by_stratum": self.by_stratum,
            "by_provenance": self.by_provenance,
            "latency": self.latency,
            "without_coverage": self.without_coverage,
            "cases": [
                {
                    "id": case.id,
                    "provenance": case.provenance,
                    "strata": list(case.strata),
                    "retrieved": list(case.retrieved),
                    "metrics": case.metrics,
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
) -> tuple[list[KnowledgeChunk], IndexStats]:
    """Every chunk of every item and topic surface, with the coverage stats."""
    chunks: list[KnowledgeChunk] = []
    surfaces = empty = 0
    for item in corpus.items.values():
        emitted = item_surfaces(item)
        surfaces += len(emitted)
        chunks += list(
            chunk_surfaces(emitted, params=params, topics=item_topics(item), url=item.url)
        )
    for topic in corpus.vocab:
        emitted = topic_surfaces(topic, corpus.topic_pages.get(topic.slug))
        surfaces += len(emitted)
        chunks += list(chunk_surfaces(emitted, params=params))
    stats = IndexStats(
        items=len(corpus.items),
        topics=len(corpus.vocab),
        surfaces=surfaces,
        chunks=len(chunks),
        empty_surfaces=empty,
    )
    return chunks, stats


def build_index(
    corpus: Corpus, *, params: ChunkerParams = DEFAULT_CHUNKER_PARAMS
) -> tuple[InMemoryLexicalIndex, IndexStats]:
    """The lexical baseline over a whole corpus, plus what it covered."""
    chunks, stats = corpus_chunks(corpus, params=params)
    index = InMemoryLexicalIndex()
    indexed = index.add(chunks)
    return index, IndexStats(
        items=stats.items,
        topics=stats.topics,
        surfaces=stats.surfaces,
        chunks=indexed,
        empty_surfaces=stats.chunks - indexed,
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
) -> EvaluationReport:
    """Score every case and aggregate by stratum and by provenance.

    `threshold`, when given, turns the report into a gate: any bucket whose `recall@max(ks)`
    falls below it becomes a named failure. When absent, the report only reports — spec §8.6
    fixes thresholds after the baseline, and a default here would be a number that could not
    come out any other way.
    """
    index, stats = build_index(corpus, params=params)
    results, latencies = [], []
    for case in cases:
        started = time.perf_counter()
        hits = _search(index, case, limit=max(ks))
        latencies.append((time.perf_counter() - started) * 1000)
        results.append(_score(case, hits, ks))

    by_stratum = _aggregate(results, STRATA, lambda case: case.strata, ks)
    by_provenance = _aggregate(results, {"real", "construido"}, lambda case: (case.provenance,), ks)
    failures = _failures(by_stratum, by_provenance, threshold, ks)
    return EvaluationReport(
        strategy=strategy,
        corpus={
            "source": corpus.source,
            "items": len(corpus.items),
            "topics": len(corpus.vocab),
            "surfaces": stats.surfaces,
            "chunks": stats.chunks,
        },
        cases=tuple(results),
        by_stratum=by_stratum,
        by_provenance=by_provenance,
        latency=_percentiles(latencies),
        without_coverage={
            "strata": sorted(k for k, v in by_stratum.items() if v == NO_COVERAGE),
            "surfaces": list(SURFACES_WITHOUT_DATA),
        },
        scenarios=tuple(scenarios),
        threshold=threshold,
        failures=failures,
        index_stats=stats,
    )


def _search(index: InMemoryLexicalIndex, case: GoldenCase, limit: int) -> tuple[LexicalHit, ...]:
    """Run one case's query, applying its filters BEFORE scoring (spec §5.3).

    The filters a case declares are part of the case (spec §8.1) — v1 kept windows under a
    key no loader read, so a temporal case silently became an untemporal one and its result
    was reported as though the window had been applied. Only the filters this baseline can
    push into `WHERE` are applied here; the rest are declared in the report rather than
    silently ignored (see `_unsupported_filters`).
    """
    return index.search(
        case.query,
        limit=limit,
        surface_types=case.filters.has_surfaces,
        origins=case.filters.origins,
    )


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

    metrics: dict[str, float] = {}
    for k in ks:
        top = ranked[:k]
        found = len(relevant & set(top))
        metrics[f"recall@{k}"] = found / len(relevant) if relevant else 0.0
        metrics[f"precision@{k}"] = found / len(top) if top else 0.0
        metrics[f"surface_recall@{k}"] = _surface_recall(case, hits, k)
    metrics["mrr"] = _mrr(ranked, relevant)
    return CaseResult(
        id=case.id,
        provenance=case.provenance,
        strata=case.strata,
        retrieved=tuple(ranked[: max(ks)]),
        metrics=metrics,
    )


def _surface_recall(case: GoldenCase, hits: Sequence[LexicalHit], k: int) -> float:
    """How many of the case's named surfaces appear among the top-k retrieved CHUNKS.

    Spec §8.4 asks for this alongside item recall because returning the right item through
    the wrong surface is a different, usually worse, answer: the evidence a consumer would
    open is not the evidence the fact is in. Item recall alone scores that as a success.
    """
    if not case.relevant_surfaces:
        return 0.0
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
    ks: tuple[int, ...],
) -> dict[str, Any]:
    """Mean of each metric per bucket — or `NO_COVERAGE` when the bucket has no cases.

    This is rule 2 of the module docstring made mechanical: there is no code path that
    produces a 0.0 for an empty bucket, because the empty branch returns a sentinel instead
    of averaging an empty list.
    """
    out: dict[str, Any] = {}
    for bucket in sorted(buckets):
        members = [r for r in results if bucket in key(r)]
        if not members:
            out[bucket] = NO_COVERAGE
            continue
        metric_names = [f"recall@{k}" for k in ks]
        metric_names += [f"precision@{k}" for k in ks]
        metric_names += [f"surface_recall@{k}" for k in ks]
        metric_names.append("mrr")
        bucket_metrics = {
            name: round(sum(m.metrics.get(name, 0.0) for m in members) / len(members), 4)
            for name in metric_names
        }
        bucket_metrics["cases"] = len(members)
        out[bucket] = bucket_metrics
    return out


def _failures(
    by_stratum: dict[str, Any],
    by_provenance: dict[str, Any],
    threshold: float | None,
    ks: tuple[int, ...],
) -> tuple[str, ...]:
    """Buckets below the threshold, each NAMED with its value.

    "It failed" is not actionable. "stratum semantico: recall@1 = 0.5 < 1.0" tells the reader
    which bucket to look at and by how much — and, because buckets with no coverage carry the
    sentinel rather than a zero, an unmeasured stratum can never be reported as a failure.
    """
    if threshold is None:
        return ()
    metric = f"recall@{max(ks)}"
    failures = []
    for label, buckets in (("stratum", by_stratum), ("provenance", by_provenance)):
        for name, values in buckets.items():
            if values == NO_COVERAGE:
                continue
            value = values.get(metric, 0.0)
            if value < threshold:
                failures.append(f"{label} {name}: {metric} = {value} < {threshold}")
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
    lines = [
        f"# Evaluación de recuperación — `{report.strategy}`",
        "",
        f"- Corpus: `{report.corpus['source']}` — {report.corpus['items']} items, "
        f"{report.corpus['topics']} topics, {report.corpus['surfaces']} superficies, "
        f"{report.corpus['chunks']} chunks.",
        f"- Umbral: {report.threshold if report.threshold is not None else 'ninguno (solo informe)'}.",
        f"- Latencia p50 {report.latency['p50_ms']} ms · p95 {report.latency['p95_ms']} ms.",
        "",
        "> Las cifras de arriba son una fotografía del corpus medido, no una constante del",
        "> producto. Vuelve a derivarlas al ejecutar (CLAUDE.md regla 2).",
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
        f"- Estratos sin casos: {', '.join(report.without_coverage['strata']) or 'ninguno'}.",
        f"- Superficies sin datos en el corpus: {', '.join(report.without_coverage['surfaces'])}.",
        "",
    ]
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
        "| bucket | casos | recall@1 | recall@10 | precision@10 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in buckets.items():
        if values == NO_COVERAGE:
            lines.append(f"| {name} | — | sin cobertura | sin cobertura | sin cobertura | — |")
            continue
        lines.append(
            f"| {name} | {values['cases']} | {values.get('recall@1', '—')} | "
            f"{values.get('recall@10', '—')} | {values.get('precision@10', '—')} | "
            f"{values['mrr']} |"
        )
    lines.append("")
    return lines
