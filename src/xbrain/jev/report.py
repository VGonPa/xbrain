"""Compare Jev's membership probabilities with the pipeline's `enrich` assignment at a
threshold.

Pure functions over the store and the side-car; the CLI and the dashboard both read through
here so the two never disagree on a number. NOTHING in this module asks Jev anything or
writes `items.json` — it only reads what the side-car already paid for.

TWO THINGS ARE DELIBERATELY NOT RE-IMPLEMENTED HERE. Currency is `assess.assessment_is_current`
alone (state + questions); `TopicAssessment.output_fingerprint` records which enrich assignment
existed AT ASK TIME and is informational — comparing it would retire an assessment the moment
the item was re-enriched, which is the one event this report exists to look at. And the bill is
`defaults.input_cost_usd`: `jev topics` and this report must quote the same number, and a
formula inlined at each call site is two definitions of it.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xbrain.jev.assess import assessment_is_current
from xbrain.jev.defaults import input_cost_usd, input_tokens_total, unpriced_providers
from xbrain.jev.models import TopicAssessment
from xbrain.models import Item, Topic
from xbrain.store import _atomic_write

#: `ItemComparison` fields holding `Pair` tuples. They are rendered by hand in the JSON
#: record (item id dropped — it is the record's own key), so `asdict` must not emit them.
_PAIR_FIELDS = frozenset({"doubtful", "missing"})


@dataclass(frozen=True)
class Pair:
    """One (item, topic) pair and the probability Jev gave it."""

    item_id: str
    slug: str
    noul: float


@dataclass(frozen=True)
class ItemComparison:
    """What Jev and `enrich` say about ONE item, at one threshold.

    `doubtful` and `missing` are the two DISAGREEMENTS, split by direction: a topic enrich
    assigned that Jev does not back, and a topic Jev backs that enrich did not assign.
    Neither is a verdict — Jev is a second opinion, not an oracle — which is why both are
    reported with the probability that produced them.
    """

    item_id: str
    assigned: tuple[str, ...]  # `enriched.topics`, in the order enrich stored them
    primary_topic: str | None
    jev_primary: str
    jev_confidence: float
    primary_rank: int | None  # 1-based rank of `primary_topic` in the Choice distribution
    doubtful: tuple[Pair, ...]  # assigned by enrich, noul < threshold — ascending noul
    missing: tuple[Pair, ...]  # not assigned, noul >= threshold — descending noul
    jev_assigned: tuple[str, ...]  # noul >= threshold, descending — Jev's own assignment
    #: Assigned topics that are NOT in `membership` — slugs Jev was never asked about.
    #:
    #: `enrich` validates topics against the vocabulary AT WRITE TIME, so an item enriched
    #: under an older `vocab.yaml` keeps a slug today's questions no longer carry. Such a
    #: pair is neither backed nor doubtful, and calling it either is a lie in a different
    #: direction: as backed it invents an endorsement Jev never gave, as doubtful it invents
    #: a doubt. It is its own bucket, and `summarize` reports it.
    unjudged: tuple[str, ...] = ()

    @property
    def primary_agrees(self) -> bool:
        """Whether Jev's Choice is the topic enrich called primary.

        False when enrich recorded no primary at all: "there is nothing to agree with" is
        not agreement, and folding the two together would let a corpus with no primaries
        report a perfect agreement rate.
        """
        return self.primary_topic is not None and self.jev_primary == self.primary_topic


def _pct(part: int, whole: int) -> float:
    """`part` of `whole` as a percentage to one decimal; 0.0 when there is no whole."""
    return round(part / whole * 100, 1) if whole else 0.0


def _primary_rank(primary_topic: str | None, probabilities: dict[str, float]) -> int | None:
    """1-based rank of `primary_topic` in the Choice distribution; `None` when it is absent.

    Ranked by descending probability, ties broken by option name so two runs of the same
    distribution can never report different ranks. The rank is what separates "Jev disagrees"
    from "Jev never considered it": a primary sitting at rank 2 of 30 is a close call, one
    sitting at rank 29 is a real conflict, and the bare boolean `primary_agrees` cannot tell
    them apart.
    """
    if primary_topic is None:
        return None
    order = [option for option, _ in sorted(probabilities.items(), key=lambda kv: (-kv[1], kv[0]))]
    return order.index(primary_topic) + 1 if primary_topic in order else None


def _doubtful(
    item_id: str, assigned: Sequence[str], membership: dict[str, float], threshold: float
) -> tuple[Pair, ...]:
    """Assigned topics Jev scored BELOW the threshold, weakest first.

    A slug absent from `membership` is skipped rather than scored 0.0: Jev was not asked
    about it (see `ItemComparison.unjudged`), and a fabricated zero would put it at the top
    of the "most doubtful" table as the strongest disagreement in the corpus.
    """
    pairs = (
        Pair(item_id, slug, membership[slug])
        for slug in assigned
        if slug in membership and membership[slug] < threshold
    )
    return tuple(sorted(pairs, key=lambda pair: (pair.noul, pair.slug)))


def _missing(
    item_id: str, assigned: Sequence[str], membership: dict[str, float], threshold: float
) -> tuple[Pair, ...]:
    """Topics Jev backs at or above the threshold that enrich did not assign, strongest first."""
    pairs = (
        Pair(item_id, slug, noul)
        for slug, noul in membership.items()
        if slug not in assigned and noul >= threshold
    )
    return tuple(sorted(pairs, key=lambda pair: (-pair.noul, pair.slug)))


def _unjudged(assigned: Sequence[str], membership: dict[str, float]) -> tuple[str, ...]:
    """Assigned topics absent from `membership` — the ones Jev was never asked about.

    Read straight off `membership`, never inferred from the buckets `compare_item` also
    builds: "Jev was not asked" is a fact about the ASK, and deriving it from the outcome
    would make it drift the day a bucket's rule changes.
    """
    return tuple(slug for slug in assigned if slug not in membership)


def _jev_assigned(membership: dict[str, float], threshold: float) -> tuple[str, ...]:
    """Jev's OWN assignment at this threshold: every topic at or above it, strongest first."""
    ranked = sorted(membership.items(), key=lambda kv: (-kv[1], kv[0]))
    return tuple(slug for slug, noul in ranked if noul >= threshold)


def compare_item(
    item: Item, assessment: TopicAssessment, threshold: float
) -> ItemComparison | None:
    """None when the item has no enrichment (nothing to compare against)."""
    if item.enriched is None:
        return None
    assigned = tuple(item.enriched.topics)
    membership = assessment.membership
    return ItemComparison(
        item_id=item.id,
        assigned=assigned,
        primary_topic=item.enriched.primary_topic,
        jev_primary=assessment.primary.choice,
        jev_confidence=assessment.primary.confidence,
        primary_rank=_primary_rank(item.enriched.primary_topic, assessment.primary.probabilities),
        doubtful=_doubtful(item.id, assigned, membership, threshold),
        missing=_missing(item.id, assigned, membership, threshold),
        jev_assigned=_jev_assigned(membership, threshold),
        unjudged=_unjudged(assigned, membership),
    )


def current_assessments(
    items: list[Item],
    assessments: dict[str, TopicAssessment],
    vocab: list[Topic],
    *,
    fallback: str,
    char_limit: int,
) -> list[tuple[Item, TopicAssessment]]:
    """The (item, assessment) pairs whose assessment is still current — stale ones are
    EXCLUDED, not folded in.

    Currency is decided by `assess.assessment_is_current`, which rebuilds today's questions
    from `vocab` + `fallback` and hashes them with the state. Re-deriving that rule here
    would be a second definition of "still valid", and the one that drifts is the one nobody
    re-computes. A stale record is dropped rather than reported, because a comparison against
    a question Jev is no longer asked is not a weaker signal — it is a wrong one.
    """
    pairs: list[tuple[Item, TopicAssessment]] = []
    for item in items:
        assessment = assessments.get(item.id)
        if assessment is not None and assessment_is_current(
            assessment, item, vocab, fallback=fallback, char_limit=char_limit
        ):
            pairs.append((item, assessment))
    return pairs


def _slug_counts(
    comparisons: list[ItemComparison],
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    """Per-slug `(assigned, doubtful, missing)` counters over every comparison."""
    assigned: Counter[str] = Counter(slug for c in comparisons for slug in c.assigned)
    doubtful: Counter[str] = Counter(pair.slug for c in comparisons for pair in c.doubtful)
    missing: Counter[str] = Counter(pair.slug for c in comparisons for pair in c.missing)
    return assigned, doubtful, missing


def _topic_row(
    slug: str, assigned: Counter[str], doubtful: Counter[str], missing: Counter[str]
) -> dict[str, Any]:
    """One `per_topic` row. `backed` excludes the doubtful ones AND the unjudged ones.

    Rows are built per VOCABULARY topic, so `assigned - doubtful` is safe here in a way it is
    not for the corpus-wide total: a slug that has left the vocabulary has no row at all.
    """
    count = assigned[slug]
    backed = count - doubtful[slug]
    return {
        "slug": slug,
        "assigned": count,
        "backed": backed,
        "backed_pct": _pct(backed, count),
        "missing": missing[slug],
    }


def _per_topic(comparisons: list[ItemComparison], vocab: list[Topic]) -> list[dict[str, Any]]:
    """One row per vocabulary topic, WORST BACKING FIRST — the reading order of the report.

    Sorted by `(backed_pct, slug)`: the question the report answers is "which topic is enrich
    assigning that Jev does not recognise", and a vocabulary-ordered table buries it.
    """
    assigned, doubtful, missing = _slug_counts(comparisons)
    rows = [_topic_row(topic.slug, assigned, doubtful, missing) for topic in vocab]
    return sorted(rows, key=lambda row: (row["backed_pct"], row["slug"]))


def _pair_totals(comparisons: list[ItemComparison]) -> dict[str, int]:
    """The (item, topic) pair counts, in both directions.

    `assigned`, `doubtful` and `unjudged` PARTITION the enrich side: `backed` is what is left
    once the other two are removed, never `assigned - doubtful` alone.
    """
    doubtful = sum(len(c.doubtful) for c in comparisons)
    unjudged = sum(len(c.unjudged) for c in comparisons)
    assigned = sum(len(c.assigned) for c in comparisons)
    return {
        "assigned": assigned,
        "doubtful": doubtful,
        "unjudged": unjudged,
        "backed": assigned - doubtful - unjudged,
        "missing": sum(len(c.missing) for c in comparisons),
        "jev": sum(len(c.jev_assigned) for c in comparisons),
        "jev_backed": sum(len(set(c.jev_assigned) & set(c.assigned)) for c in comparisons),
    }


def _primary_totals(comparisons: list[ItemComparison], slugs: set[str]) -> tuple[int, int]:
    """`(agreements, fallback picks)` over the comparisons.

    A primary outside the vocabulary is Jev answering "none of these", which is a different
    event from disagreeing with enrich's pick and is counted separately — a corpus where Jev
    keeps choosing the fallback says the vocabulary is missing a topic, not that enrich is
    wrong.
    """
    agree = sum(1 for c in comparisons if c.primary_agrees)
    fallback = sum(1 for c in comparisons if c.jev_primary not in slugs)
    return agree, fallback


def _judge_fields(assessments: tuple[TopicAssessment, ...]) -> dict[str, Any]:
    """Who answered, how much was sent and what it cost.

    A TUPLE, not an iterable: the three `jev.defaults` helpers each walk `assessments`, so a
    one-shot generator would be exhausted after the first and price the run at zero.

    Both "unknown" markers travel with their number, because neither zero can speak for
    itself. `input_tokens_unknown` counts records whose provider reported no usage — folding
    those into 0 reports a run that WAS paid for as free. `unpriced_providers` NAMES the
    judges `INPUT_USD_PER_MTOK` cannot price; they contribute 0.0 rather than borrowing
    another vendor's rate, and without the names a real zero and an unpriced one render
    identically.
    """
    tokens, unknown = input_tokens_total(assessments)
    return {
        "models": dict(sorted(Counter(a.model for a in assessments).items())),
        "providers": dict(sorted(Counter(a.provider for a in assessments).items())),
        "truncated": sum(1 for a in assessments if a.truncated),
        "input_tokens": tokens,
        "input_tokens_unknown": unknown,
        "cost_usd": round(input_cost_usd(assessments), 4),
        "unpriced_providers": list(unpriced_providers(assessments)),
    }


def summarize(
    pairs: list[tuple[Item, TopicAssessment]], vocab: list[Topic], threshold: float
) -> dict[str, Any]:
    """Every number the report and the dashboard quote, computed ONCE, from the same pairs.

    `items_assessed` counts the pairs handed in; `items_compared` counts the ones with an
    enrichment to compare against. They differ exactly by the items Jev has an opinion about
    that the pipeline has not enriched, and collapsing them would hide that population.
    """
    comparisons = [c for item, a in pairs if (c := compare_item(item, a, threshold)) is not None]
    assessments = tuple(assessment for _, assessment in pairs)
    totals = _pair_totals(comparisons)
    agree, fallback = _primary_totals(comparisons, {topic.slug for topic in vocab})
    return {
        "threshold": threshold,
        "items_assessed": len(pairs),
        "items_compared": len(comparisons),
        **_judge_fields(assessments),
        "assigned_pairs": totals["assigned"],
        "assigned_backed": totals["backed"],
        "assigned_unjudged": totals["unjudged"],
        "enrich_backed_pct": _pct(totals["backed"], totals["assigned"]),
        "jev_pairs": totals["jev"],
        "jev_backed": totals["jev_backed"],
        "jev_backed_pct": _pct(totals["jev_backed"], totals["jev"]),
        "doubtful_pairs": totals["doubtful"],
        "missing_pairs": totals["missing"],
        "primary_agree": agree,
        "primary_agree_pct": _pct(agree, len(comparisons)),
        "primary_fallback": fallback,
        "per_topic": _per_topic(comparisons, vocab),
    }


def _snippet(item: Item | None, width: int = 80) -> str:
    """The post's first `width` characters on one line, for a table cell.

    Whitespace is collapsed because a cell containing a newline or a pipe breaks the markdown
    table around it, and the snippet exists to let a reader recognise the post without
    opening it.
    """
    text = " ".join(item.text.split()) if item else ""
    return (text[: width - 1] + "…") if len(text) > width else text


def _counted(counts: dict[str, int]) -> str:
    """`name (n), name (n)` for a name→count map, sorted; an em dash when it is empty."""
    return ", ".join(f"{name} ({count})" for name, count in sorted(counts.items())) or "—"


def _cost_line(summary: dict[str, Any]) -> str:
    """The bill, with both markers that say which kind of zero this is."""
    line = f"Coste: {summary['input_tokens']} tokens de entrada"
    if summary["input_tokens_unknown"]:
        line += f" (+{summary['input_tokens_unknown']} sin recuento)"
    line += f" · ~{summary['cost_usd']} $"
    if summary["unpriced_providers"]:
        line += f" · sin tarifa: {', '.join(summary['unpriced_providers'])}"
    return line


def _headline(summary: dict[str, Any]) -> list[str]:
    """Title, what was asked, and the four numbers the whole report exists to produce."""
    return [
        f"# Jev · topics — {datetime.now(timezone.utc):%Y-%m-%d}",
        "",
        f"Umbral {summary['threshold']} · modelos: {_counted(summary['models'])} · "
        f"proveedores: {_counted(summary['providers'])}",
        f"Items evaluados: {summary['items_assessed']} · comparables: "
        f"{summary['items_compared']} · truncados: {summary['truncated']}",
        "",
        f"**Asignaciones de enrich respaldadas por Jev:** {summary['assigned_backed']} de "
        f"{summary['assigned_pairs']} ({summary['enrich_backed_pct']} %) · "
        f"**asignaciones de Jev que enrich tiene:** {summary['jev_backed']} de "
        f"{summary['jev_pairs']} ({summary['jev_backed_pct']} %)",
        f"**Dudosas:** {summary['doubtful_pairs']} · **sin juzgar:** "
        f"{summary['assigned_unjudged']} · **candidatas que faltan:** "
        f"{summary['missing_pairs']} · **primario coincide:** {summary['primary_agree']} "
        f"({summary['primary_agree_pct']} %) · **primario = fallback:** "
        f"{summary['primary_fallback']}",
        _cost_line(summary),
    ]


def _table(title: str, columns: Sequence[str], rows: list[str]) -> list[str]:
    """A titled markdown table, header included — always rendered, even with no rows.

    An empty table says "nothing disagreed here"; omitting the section entirely would read as
    "this report does not cover that", which is the opposite claim.
    """
    return ["", title, "", f"| {' | '.join(columns)} |", "|" + "---|" * len(columns), *rows]


def _pair_table(title: str, pairs: list[Pair], items_by_id: dict[str, Item], top: int) -> list[str]:
    """The `top` worst pairs as a table; the caller has already sorted them."""
    rows = [
        f"| {pair.item_id} | {pair.slug} | {pair.noul:.2f} | "
        f"{_snippet(items_by_id.get(pair.item_id))} |"
        for pair in pairs[:top]
    ]
    return _table(title, ("item", "topic", "noul", "texto"), rows)


def _disagreement_row(comparison: ItemComparison, items_by_id: dict[str, Item]) -> str:
    """One row of the primary-disagreement table, rank included."""
    return (
        f"| {comparison.item_id} | {comparison.primary_topic or '—'} | {comparison.jev_primary} "
        f"| {comparison.jev_confidence:.2f} | {comparison.primary_rank or '—'} | "
        f"{_snippet(items_by_id.get(comparison.item_id))} |"
    )


def _sorted_doubtful(comparisons: list[ItemComparison]) -> list[Pair]:
    """Every doubtful pair in the corpus, weakest first — the worst disagreements on top."""
    pairs = (pair for c in comparisons for pair in c.doubtful)
    return sorted(pairs, key=lambda pair: (pair.noul, pair.item_id))


def _sorted_missing(comparisons: list[ItemComparison]) -> list[Pair]:
    """Every missing candidate in the corpus, strongest first."""
    pairs = (pair for c in comparisons for pair in c.missing)
    return sorted(pairs, key=lambda pair: (-pair.noul, pair.item_id))


def render_report_markdown(
    summary: dict[str, Any],
    comparisons: list[ItemComparison],
    items_by_id: dict[str, Item],
    *,
    top: int = 20,
) -> str:
    """The human-readable report: the headline numbers, then the worst `top` of each table.

    TRUNCATED ON PURPOSE. The corpus is ~3,000 items and the JSON beside it carries every
    record, so this file is the thing a person reads — a table with 2,000 rows is not one.
    """
    disagree = [c for c in comparisons if not c.primary_agrees]
    lines = _headline(summary)
    lines += _table(
        "## Por topic (peor primero)",
        ("topic", "asignadas", "respaldadas", "%", "faltan"),
        [
            f"| {row['slug']} | {row['assigned']} | {row['backed']} | "
            f"{row['backed_pct']} | {row['missing']} |"
            for row in summary["per_topic"]
        ],
    )
    lines += _pair_table(
        f"## Dudosas (top {top}, noul ascendente)", _sorted_doubtful(comparisons), items_by_id, top
    )
    lines += _pair_table(
        f"## Candidatas que faltan (top {top}, noul descendente)",
        _sorted_missing(comparisons),
        items_by_id,
        top,
    )
    lines += _table(
        f"## Primario en desacuerdo (top {top})",
        ("item", "enrich", "Jev", "conf.", "rango del de enrich", "texto"),
        [_disagreement_row(c, items_by_id) for c in disagree[:top]],
    )
    return "\n".join(lines) + "\n"


def _record(comparison: ItemComparison) -> dict[str, Any]:
    """One compared item as JSON: every field, with the tuples flattened to lists.

    Built from `asdict` rather than field by field so a field added to `ItemComparison` shows
    up in the JSON — the dashboard's input — without a second edit here to remember. The two
    `Pair` fields are re-rendered by hand: their `item_id` is the record's own id repeated on
    every row.
    """
    return {
        **{k: v for k, v in asdict(comparison).items() if k not in _PAIR_FIELDS},
        "assigned": list(comparison.assigned),
        "jev_assigned": list(comparison.jev_assigned),
        "unjudged": list(comparison.unjudged),
        "doubtful": [{"slug": p.slug, "noul": p.noul} for p in comparison.doubtful],
        "missing": [{"slug": p.slug, "noul": p.noul} for p in comparison.missing],
        "primary_agrees": comparison.primary_agrees,
    }


def write_reports(
    summary: dict[str, Any],
    comparisons: list[ItemComparison],
    items_by_id: dict[str, Item],
    jev_dir: Path,
) -> tuple[Path, Path]:
    """`topics-report.json` (summary + one record per compared item) and `topics-report.md`.

    Two files, one computation: the markdown is what a person reads and the JSON is what the
    dashboard reads, and both are rendered from the SAME `summary` and `comparisons` so they
    cannot quote different numbers. Atomic, like every other write in the repo — a half-written
    report read by the dashboard is a wrong report, not a missing one.
    """
    jev_dir.mkdir(parents=True, exist_ok=True)
    json_path = jev_dir / "topics-report.json"
    md_path = jev_dir / "topics-report.md"
    payload = {"summary": summary, "items": [_record(c) for c in comparisons]}
    _atomic_write(json_path, json.dumps(payload, indent=2, ensure_ascii=False))
    _atomic_write(md_path, render_report_markdown(summary, comparisons, items_by_id))
    return json_path, md_path
