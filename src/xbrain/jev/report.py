"""Compare Jev's membership probabilities with the pipeline's `enrich` assignment at a
threshold.

Pure functions over the store and the side-car. `xbrain jev report` and `xbrain jev dashboard`
both read through here, so the two can never disagree on a number. NOTHING in this module asks
Jev anything or writes `items.json` — it only reads what the side-car already paid for.

THREE THINGS ARE DELIBERATELY NOT RE-IMPLEMENTED HERE.

* Currency is `assess.current_pairs` alone (state + questions, digest hoisted once).
  `TopicAssessment.output_fingerprint` records which enrich assignment existed AT ASK TIME and
  is informational — comparing it would retire an assessment the moment the item was
  re-enriched, which is the one event this report exists to look at.
* The bill is `defaults.input_cost_usd`, and the SENTENCE that quotes it is
  `defaults.jev_cost_fragment`: `jev topics` and this report must quote the same number in the
  same words, and a formula or a format inlined at each call site is two definitions of it.
* Spanish agreement is `defaults.plural`. "1 filas más" reads as a bug in the counting.

NOTHING IS DROPPED WITHOUT A COUNTER. Stale and orphaned side-car records are excluded from the
comparison and COUNTED in the summary (`assessments_stale`, `assessments_orphaned`): a full
side-car retired by a vocabulary edit must never render identically to a side-car nobody ever
wrote, because the second reading sends an operator to re-pay for the whole corpus.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xbrain.jev.assess import current_pairs
from xbrain.jev.defaults import (
    input_cost_usd,
    input_tokens_total,
    jev_cost_fragment,
    plural,
    tokens_cost_usd,
    unpriced,
    unpriced_providers,
)
from xbrain.jev.models import JevRun, TopicAssessment
from xbrain.models import Item, Topic, _require_utc_aware
from xbrain.store import _atomic_write

#: `ItemComparison` fields the JSON record does not take from `asdict`: the two holding `Pair`
#: tuples are rendered by hand (item id dropped — the record already carries it as a field),
#: and `threshold` is the summary's, the same on every record.
_RECORD_SKIP = frozenset({"doubtful", "missing", "threshold"})

#: Why a primary does not coincide, in CLASSIFICATION PRIORITY order — the first that
#: applies wins, so the five sections partition the non-agreeing comparisons. The enrich-side
#: facts come first because they are the ones an operator acts on (re-enrich, or put the slug
#: back in `vocab.yaml`); Jev's own answer next; the provider's omission last.
_PRIMARY_REASONS: tuple[tuple[str, str], ...] = (
    ("sin_primario", "Sin primario en enrich"),
    ("sin_juzgar", "Primario sin juzgar (salió del vocabulario)"),
    ("fallback", "Jev eligió el fallback"),
    ("sin_rango", "Primario ausente de la distribución de Jev"),
    ("desacuerdo", "Desacuerdo real"),
)


@dataclass(frozen=True)
class Pair:
    """One (item, topic) pair and the probability Jev gave it."""

    item_id: str
    slug: str
    noul: float


@dataclass(frozen=True)
class Band:
    """One range of Jev's probability on a disagreement: `lo <= p < hi`, and `p == hi` too
    when `top` (the last band reaches 1.0). `kind` is the direction it splits — `enrich_only`
    (enrich assigns, Jev below the threshold) or `jev_only` (Jev at or above it, enrich did
    not assign)."""

    kind: str
    key: str
    label: str
    lo: float
    hi: float
    #: The summary total this band's pairs add up to, with the other bands of its kind.
    total: str
    top: bool = False

    def holds(self, noul: float) -> bool:
        return self.lo <= noul and (noul < self.hi or (self.top and noul == self.hi))


#: THE band edges: how sure Jev was on each disagreement, in the words the page uses. `None`
#: is the threshold, which ends the enrich-side bands and starts the Jev-side ones. Defined
#: here once, shipped in the summary with each band's counts, and stated on the page. The
#: labels are part of the report's contract (they travel in `topics-report.json`), in Spanish
#: like every other operator-facing word, and they read right for a HIGH threshold such as the
#: default 0.85 — at 0.4 "entre 0,5 y el umbral" is a band that is simply dropped.
_BAND_SPECS: tuple[tuple[str, str, str, float | None, float | None], ...] = (
    ("enrich_only", "e-lo", "Jev lo descarta claramente", 0.0, 0.2),
    ("enrich_only", "e-mid", "Jev lo ve poco probable", 0.2, 0.5),
    ("enrich_only", "e-near", "Jev duda: entre 0,5 y el umbral", 0.5, None),
    ("jev_only", "j-near", "Jev lo ve por encima del umbral, sin mucho margen", None, 0.95),
    ("jev_only", "j-hi", "Jev lo ve claramente", 0.95, 1.0),
)
#: The summary total each kind of band splits.
_BAND_TOTALS = {"enrich_only": "doubtful_pairs", "jev_only": "missing_pairs"}


def confidence_bands(threshold: float) -> tuple[Band, ...]:
    """The bands at this threshold: enrich-side ones cut at it from above, Jev-side ones from
    below. A band the threshold leaves no room for (0.5–threshold at a threshold of 0.4) is
    dropped rather than shipped with an empty or inverted range.

    Only the Jev side's last band is `top` (it holds 1.0): an enrich-side band ending at a
    threshold of 1.0 still excludes 1.0, which is at the threshold and so not a doubt. At that
    threshold the top band is the single point 1.0 — the one band allowed `lo == hi`."""
    bands = []
    for kind, key, label, lo, hi in _BAND_SPECS:
        low, high = _band_edges(kind, lo, hi, threshold)
        top = kind == "jev_only" and high == 1.0
        if low < high or (top and low == high):
            bands.append(Band(kind, key, label, low, high, _BAND_TOTALS[kind], top=top))
    return tuple(bands)


def _band_edges(
    kind: str, lo: float | None, hi: float | None, threshold: float
) -> tuple[float, float]:
    """One spec's edges at this threshold (`None` = the threshold): enrich-side bands are cut
    at it from above, Jev-side ones from below."""
    if kind == "enrich_only":
        return lo or 0.0, min(threshold if hi is None else hi, threshold)
    return max(threshold if lo is None else lo, threshold), hi or 1.0


def band_of(noul: float, kind: str, bands: Sequence[Band]) -> Band:
    """The `kind` band holding `noul`. A probability none holds is REFUSED, never dropped: a
    disagreement missing from every band would make the bands sum short of their total."""
    for band in bands:
        if band.kind == kind and band.holds(noul):
            return band
    # English: no stored answer reaches here (a membership probability is in [0, 1] and the
    # bands cover it); only a caller handing in the wrong kind or bands can.
    raise ValueError(f"probability {noul} is in no {kind} band")


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
    #: The threshold this comparison was made at. Everything that re-reads the comparison at a
    #: threshold (the confidence bands) takes it from here, so it cannot be handed another.
    threshold: float
    #: Assigned topics that are NOT in `membership` — slugs Jev was never asked about.
    #:
    #: `enrich` validates topics against the vocabulary AT WRITE TIME, so an item enriched
    #: under an older `vocab.yaml` keeps a slug today's questions no longer carry. Such a
    #: pair is neither backed nor doubtful, and calling it either is a lie in a different
    #: direction: as backed it invents an endorsement Jev never gave, as doubtful it invents
    #: a doubt. It is its own bucket, and `summarize` reports it.
    unjudged: tuple[str, ...] = ()
    #: Whether `primary_topic` is itself a topic Jev was never asked about.
    #:
    #: The same fact as `unjudged`, on the primary axis. Without it, a primary that left the
    #: vocabulary reads as "Jev picked something else" in the headline agreement rate, while
    #: the identical fact about a membership topic gets its own bucket — the asymmetry
    #: `unjudged` exists to remove. `primary_rank` is `None` in this case too, but that is
    #: also `None` when the provider simply omitted an option from its distribution, so it
    #: cannot carry the meaning on its own.
    primary_unjudged: bool = False

    @property
    def primary_agrees(self) -> bool:
        """Whether Jev's Choice is the topic enrich called primary.

        False when enrich recorded no primary at all: "there is nothing to agree with" is
        not agreement, and folding the two together would let a corpus with no primaries
        report a perfect agreement rate.

        False too when the primary left the vocabulary (`primary_unjudged`): Jev was never asked
        about it, so the one way it can "match" is an old slug spelled like the fallback, where
        "never asked" and "none of these" are the same string and opposite facts.
        """
        return (
            self.primary_topic is not None
            and not self.primary_unjudged
            and self.jev_primary == self.primary_topic
        )

    @property
    def enrich_only(self) -> tuple[Pair, ...]:
        """Disagreement 1 of 3 — enrich assigns it and Jev does not back it (`doubtful`)."""
        return self.doubtful

    @property
    def jev_only(self) -> tuple[Pair, ...]:
        """Disagreement 2 of 3 — Jev backs it and enrich did not assign it (`missing`)."""
        return self.missing

    @property
    def primary_differs(self) -> bool:
        """Disagreement 3 of 3 — the primary topic is not the one Jev chose.

        An item enrich left WITHOUT a primary counts: there is nothing for Jev to agree with,
        and that is itself a thing to fix. (`primary_agrees` is False for it for the same
        reason.)
        """
        return not self.primary_agrees

    @property
    def disagreements(self) -> int:
        """How much this post disagrees: each enrich-only topic, each Jev-only topic, and the
        primary when it differs. THE definition every surface sorts and filters on."""
        return len(self.enrich_only) + len(self.jev_only) + int(self.primary_differs)

    @property
    def primary_unranked(self) -> bool:
        """Jev was asked about enrich's primary, and then left it out of its own distribution.

        `PrimaryChoice` deliberately does not enforce full coverage of the option set ("read a
        option's probability with `.get(option, 0.0)`"), so a provider may score a slug in
        `membership` and omit it from the Choice. `primary_rank` is `None` for that item — the
        same `—` it renders for "no primary at all" and for `primary_unjudged` — which makes the
        one case where Jev DEMONSTRABLY never considered the topic indistinguishable from the
        two where it was never asked. This is the third state, named, so it gets a bucket
        instead of quietly depressing the agreement rate.
        """
        return (
            self.primary_topic is not None
            and not self.primary_unjudged
            and self.primary_rank is None
        )


def _pct(part: int, whole: int) -> float:
    """`part` of `whole` as a percentage to one decimal; 0.0 when there is no whole.

    DISPLAY ONLY. One decimal is what a reader wants, and it is also lossy at both ends —
    2499 of 2500 renders `100.0`, 1 of 2500 renders `0.0`. Nothing may SORT on this value;
    `_backing_order` keys on the exact ratio for exactly that reason.
    """
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


def _primary_unjudged(primary_topic: str | None, membership: dict[str, float]) -> bool:
    """Whether enrich's primary is a topic absent from `membership` — never asked about.

    Read off `membership`, like `_unjudged`: no primary at all is not "unjudged", it is
    nothing to compare, which `primary_agrees` already reports as False.
    """
    return primary_topic is not None and primary_topic not in membership


def jev_assigned(membership: dict[str, float], threshold: float) -> tuple[str, ...]:
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
        jev_assigned=jev_assigned(membership, threshold),
        threshold=threshold,
        unjudged=_unjudged(assigned, membership),
        primary_unjudged=_primary_unjudged(item.enriched.primary_topic, membership),
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

    Currency is decided by `assess.current_pairs`, which builds today's questions ONCE for the
    whole side-car and hashes them with each item's state. Re-deriving that rule here would be a
    second definition of "still valid", and the one that drifts is the one nobody re-computes.
    A stale record is dropped rather than reported, because a comparison against a question Jev
    is no longer asked is not a weaker signal — it is a wrong one.

    THIS WRAPPER DISCARDS THE DROP COUNTS. Call `assess.current_pairs` directly when they
    matter — `_summarize` reports `assessments_stale` and `assessments_orphaned`, and a caller
    that goes through here and then passes `stale=0` is telling the report nothing was dropped.
    """
    return list(
        current_pairs(items, assessments, vocab, fallback=fallback, char_limit=char_limit).pairs
    )


def _slug_counts(comparisons: list[ItemComparison]) -> dict[str, Counter[str]]:
    """Per-slug counters over every comparison: `assigned` / `doubtful` / `missing` /
    `unjudged`, how often each side picked the slug as primary and how often both did
    (`_primary_counts`: `enrich_primary`, `jev_primary`, `primary_both`). The posts behind
    `primary_both` are `post_sets`' `pd`; those behind the bands, its `bands`."""
    return {
        "assigned": Counter(slug for c in comparisons for slug in c.assigned),
        "doubtful": Counter(pair.slug for c in comparisons for pair in c.doubtful),
        "missing": Counter(pair.slug for c in comparisons for pair in c.missing),
        "unjudged": Counter(slug for c in comparisons for slug in c.unjudged),
        **_primary_counts(comparisons),
    }


def _primary_counts(comparisons: list[ItemComparison]) -> dict[str, Counter[str]]:
    """Per-slug count of the posts each side picked it as primary on (enrich, then Jev), and
    of the posts both picked it on."""
    return {
        "enrich_primary": Counter(c.primary_topic for c in comparisons if c.primary_topic),
        "jev_primary": Counter(c.jev_primary for c in comparisons),
        "primary_both": Counter(c.jev_primary for c in comparisons if c.primary_agrees),
    }


def _topic_row(slug: str, counts: dict[str, Counter[str]]) -> dict[str, Any]:
    """One `per_topic` row: `assigned`, the three buckets that PARTITION it, `missing`,
    `disagreeing` (doubtful + missing) and the two primary counts.

    `backed = assigned - doubtful - unjudged`, the same arithmetic as the corpus-wide total in
    `_pair_totals`, and not because a vocabulary row can carry an unjudged pair today. It
    cannot: `questions.build_topic_questions` asks one Noul per vocabulary slug and
    `assess.parse_topic_result` refuses a partial answer set, so for a CURRENT assessment
    `membership`'s keys are exactly today's vocabulary and an unjudged slug has no row to land
    in. The term is here because that shelter depends entirely on the caller having filtered
    through `current_assessments` — `summarize` and `build_report` are public and take raw
    pairs — and a row that silently counts an unjudged pair as backed is the exact error
    `_pair_totals` exists to prevent one altitude up.
    """
    assigned = counts["assigned"][slug]
    doubtful = counts["doubtful"][slug]
    unjudged = counts["unjudged"][slug]
    return {
        "slug": slug,
        "assigned": assigned,
        "backed": assigned - doubtful - unjudged,
        # The other two terms of the partition ride WITH the row. They cost nothing —
        # `_slug_counts` already built both Counters — and a consumer that shows them per
        # topic otherwise displays a per-topic number whose only counterpart in this report
        # is a corpus-wide sum. A number nothing can be checked against is free to be wrong.
        "doubtful": doubtful,
        "unjudged": unjudged,
        "backed_pct": _pct(assigned - doubtful - unjudged, assigned),
        "missing": counts["missing"][slug],
        # The posts that disagree ABOUT this topic, both directions: enrich puts it and Jev
        # does not back it, or Jev backs it and enrich did not put it. A post is at most one
        # of the two for one topic, so the sum counts posts. The page's topic navigator.
        "disagreeing": doubtful + counts["missing"][slug],
        # How often each side made it THE topic of a post: the page's topic index puts the
        # two side by side, and a topic Jev never picks as primary is a finding of its own.
        "enrich_primary": counts["enrich_primary"][slug],
        "jev_primary": counts["jev_primary"][slug],
        # Both sides picked it: the diagonal of the primary cross. `enrich_primary` minus the
        # posts where enrich picked it and Jev did not (`primary_confusion`), counted directly.
        "primary_both": counts["primary_both"][slug],
    }


def _backing_order(row: dict[str, Any]) -> tuple[bool, float, str]:
    """`per_topic`'s sort key: never-assigned rows last, then the EXACT backing ratio.

    The exact ratio, never `backed_pct`. `_pct` rounds to one decimal, so a topic at 2499 of
    2500 ties with every perfect topic and is then ordered ALPHABETICALLY among them: in a
    30-topic vocabulary the one topic with a real disagreement lands below every perfect topic
    whose slug sorts earlier, in a section titled "peor primero". The rounded value is for the
    reader; the order is for the truth.
    """
    assigned = row["assigned"]
    ratio = row["backed"] / assigned if assigned else 0.0
    return (assigned == 0, ratio, row["slug"])


def _per_topic(comparisons: list[ItemComparison], vocab: list[Topic]) -> list[dict[str, Any]]:
    """One row per vocabulary topic, WORST BACKING FIRST — the reading order of the report.

    The question this table answers is "which topic is enrich assigning that Jev does not
    recognise", and a vocabulary-ordered table buries it. So: rows WITH assignments first,
    worst-backed first, then the never-assigned ones.

    Sorting purely on `backed_pct` would put every unused topic on top, because `_pct` scores
    a zero whole as 0.0 — in a 30-topic vocabulary with a long tail, the answer would sit
    below rows about nothing. A topic nobody assigned has no backing rate to be worst at.
    See `_backing_order` for why the key is the exact ratio and not the rendered percentage.
    """
    counts = _slug_counts(comparisons)
    rows = [_topic_row(topic.slug, counts) for topic in vocab]
    return sorted(rows, key=_backing_order)


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


def _post_totals(comparisons: list[ItemComparison]) -> dict[str, int]:
    """Posts with any disagreement, and the same posts split by KIND of disagreement (a post
    may count in several): the three `ItemComparison` properties `disagreements` sums."""
    return {
        "posts_with_disagreement": sum(1 for c in comparisons if c.disagreements),
        "posts_enrich_only": sum(1 for c in comparisons if c.enrich_only),
        "posts_jev_only": sum(1 for c in comparisons if c.jev_only),
        "posts_primary_differs": sum(1 for c in comparisons if c.primary_differs),
    }


def pair_key(enrich: str | None, jev: str | None) -> str:
    """`enrich~jev`, with `-` for "nothing": the key of a confusion pair in `post_sets` and in
    the page's URLs. Slugs are `[a-z0-9-]`, so neither `~` nor a lone `-` can be one."""
    return f"{enrich or '-'}~{jev or '-'}"


PairPosts = dict[tuple[str | None, str | None], list[str]]


def _topic_pairs(comparisons: list[ItemComparison]) -> PairPosts:
    """Every topic only enrich has × every topic only Jev has, per post (`None` for a side
    that has nothing), with the posts."""
    pairs: PairPosts = {}
    for c in comparisons:
        if not (c.doubtful or c.missing):
            continue
        only_enrich: list[str | None] = [pair.slug for pair in c.doubtful] or [None]
        only_jev: list[str | None] = [pair.slug for pair in c.missing] or [None]
        for enrich in only_enrich:
            for jev in only_jev:
                pairs.setdefault((enrich, jev), []).append(c.item_id)
    return pairs


def _primary_pairs(comparisons: list[ItemComparison]) -> PairPosts:
    """Enrich's primary × Jev's choice on every post where they differ, with the posts."""
    pairs: PairPosts = {}
    for c in comparisons:
        if c.primary_differs:
            pairs.setdefault((c.primary_topic, c.jev_primary), []).append(c.item_id)
    return pairs


def _pair_rows(pairs: PairPosts) -> list[dict[str, Any]]:
    """`{enrich, jev, posts}` rows, most posts first, ties by the two slugs (`None` first) so
    two runs order identically. Counts only: the posts are `post_sets`'."""
    rows = [{"enrich": e, "jev": j, "posts": len(ids)} for (e, j), ids in pairs.items()]
    return sorted(rows, key=lambda row: (-row["posts"], row["enrich"] or "", row["jev"] or ""))


def topic_confusion(comparisons: list[ItemComparison]) -> list[dict[str, Any]]:
    """What each side put INSTEAD, counted in posts: every topic only enrich has (`doubtful`) ×
    every topic only Jev has (`missing`) on the same post.

    A post where only one side has anything to its name pairs it with `None` — enrich put a
    topic Jev does not back and Jev put nothing in its place, or Jev added one without enrich
    having put anything it replaces. A post with two topics on one side is in two rows (a
    PRODUCT: 2 × 2 puts one post in four). The page's "se confunde con" list."""
    return _pair_rows(_topic_pairs(comparisons))


def primary_confusion(comparisons: list[ItemComparison]) -> list[dict[str, Any]]:
    """Enrich's primary × Jev's Choice on every post where they differ (`primary_differs`),
    counted in posts. `None` is a post enrich left without a primary; the fallback is named as
    Jev answered it. The rows' posts add up to `posts_primary_differs`."""
    return _pair_rows(_primary_pairs(comparisons))


def post_sets(comparisons: list[ItemComparison]) -> dict[str, dict[str, list[str]]]:
    """THE index of the posts behind every count the page opens: `cx` and `px` by `pair_key`
    (`topic_confusion`, `primary_confusion`), `pd` by topic (the posts both sides picked it as
    primary on, `per_topic.primary_both`) and `bands` by band key (`confidence_bands`).

    Kept OUT of the summary on purpose: the summary holds counts, which are what the JSON
    report is for and which stay small; these lists grow with every evaluated post, and only
    the page needs them (to open a pair's posts). `build_page_data` ships them; `write_reports`
    does not."""
    sets = {
        kind: {pair_key(e, j): sorted(ids) for (e, j), ids in pairs.items()}
        for kind, pairs in (("cx", _topic_pairs(comparisons)), ("px", _primary_pairs(comparisons)))
    }
    sets["pd"] = {slug: sorted(ids) for slug, ids in _agreeing_primaries(comparisons).items()}
    threshold = comparisons_threshold(comparisons)
    sets["bands"] = (
        {}
        if threshold is None
        else {
            band.key: sorted({pair.item_id for pair in pairs})
            for band, pairs in _band_pairs(comparisons, threshold).items()
        }
    )
    return sets


def comparisons_threshold(comparisons: Sequence[ItemComparison]) -> float | None:
    """The one threshold the comparisons were made at; `None` for none. Two thresholds in one
    list is a caller bug — the bands of one would be cut at the other's edges — and raises."""
    thresholds = {c.threshold for c in comparisons}
    if len(thresholds) > 1:
        raise ValueError(f"comparisons made at more than one threshold: {sorted(thresholds)}")
    return next(iter(thresholds), None)


def _band_pairs(comparisons: list[ItemComparison], threshold: float) -> dict[Band, list[Pair]]:
    """Every disagreeing (post, topic) pair in its band: `doubtful` by the enrich-side bands,
    `missing` by the Jev-side ones."""
    bands = confidence_bands(threshold)
    grouped: dict[Band, list[Pair]] = {band: [] for band in bands}
    for c in comparisons:
        for kind, pairs in (("enrich_only", c.doubtful), ("jev_only", c.missing)):
            for pair in pairs:
                grouped[band_of(pair.noul, kind, bands)].append(pair)
    return grouped


def confidence_rows(comparisons: list[ItemComparison], threshold: float) -> list[dict[str, Any]]:
    """How sure Jev was on each disagreement: one row per band, its edges and label, the
    pairs in it and the posts they are on (a post with two pairs in a band counts once).
    Counts only: the posts are `post_sets`'.

    `threshold` is the summary's; comparisons made at another one are refused rather than cut
    at edges they were not compared against."""
    made_at = comparisons_threshold(comparisons)
    if made_at is not None and made_at != threshold:
        raise ValueError(f"comparisons made at {made_at}, summary asked for {threshold}")
    return [
        {**asdict(band), "pairs": len(pairs), "posts": len({p.item_id for p in pairs})}
        for band, pairs in _band_pairs(comparisons, threshold).items()
    ]


def _agreeing_primaries(comparisons: list[ItemComparison]) -> dict[str, list[str]]:
    """The posts where both sides picked the same primary, by that topic."""
    both: dict[str, list[str]] = {}
    for c in comparisons:
        if c.primary_agrees:
            both.setdefault(c.jev_primary, []).append(c.item_id)
    return both


def chose_fallback(comparison: ItemComparison, slugs: set[str]) -> bool:
    """Whether Jev's primary is outside the vocabulary — the fallback ("none of these")."""
    return comparison.jev_primary not in slugs


def _primary_totals(comparisons: list[ItemComparison], slugs: set[str]) -> dict[str, int]:
    """Agreements, fallback picks and never-asked primaries over the comparisons.

    Four different events, four counters. `fallback` is Jev answering "none of these", which
    says the vocabulary is missing a topic rather than that enrich is wrong. `unjudged` is
    enrich's primary having left the vocabulary, which says nothing about either of them.
    `unranked` is the provider omitting the option from its own distribution. The last three
    are disagreements only in the sense that they are not agreements.

    THE DENOMINATOR KEEPS COUNTING THEM, deliberately: `primary_agree_pct` divides by every
    comparison, exactly as `enrich_backed_pct` divides by every assigned pair including the
    unjudged ones. Dropping them would let a corpus improve its agreement rate by losing
    vocabulary. The buckets make the REASON visible; they do not shrink the denominator.
    """
    return {
        "agree": sum(1 for c in comparisons if c.primary_agrees),
        "fallback": sum(1 for c in comparisons if chose_fallback(c, slugs)),
        "unjudged": sum(1 for c in comparisons if c.primary_unjudged),
        "unranked": sum(1 for c in comparisons if c.primary_unranked),
    }


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
        # `input_cost_usd` returns a genuine float even for an empty run, so this key's TYPE
        # cannot change with the contents of the side-car — a reader of the JSON gets a number
        # it can format the same way every time.
        "cost_usd": round(input_cost_usd(assessments), 4),
        "unpriced_providers": list(unpriced_providers(assessments)),
    }


def compare_all(
    pairs: list[tuple[Item, TopicAssessment]], threshold: float
) -> list[ItemComparison]:
    """Every COMPARABLE pair, compared once. Pairs with no enrichment drop out."""
    return [c for item, a in pairs if (c := compare_item(item, a, threshold)) is not None]


def build_report(
    pairs: list[tuple[Item, TopicAssessment]],
    vocab: list[Topic],
    threshold: float,
    *,
    now: datetime | None = None,
    stale: int = 0,
    orphans: int = 0,
    unassessed: int = 0,
) -> tuple[dict[str, Any], list[ItemComparison]]:
    """The summary AND the comparisons it was computed from — one comparison pass.

    THE ENTRY POINT. Every caller needs both halves: the summary carries the numbers, the
    comparisons carry the rows. Computing the comparisons twice is not just twice the work on a
    ~3,000-item corpus — it is a second call site that has to be handed the same threshold, and
    the day the two disagree the report's tables stop matching its own headline.

    `pairs` MUST already be filtered by `current_assessments` (or `assess.current_pairs`): every
    number below assumes each `membership` was produced by TODAY'S questions. Stale pairs do not
    merely age the report, they make `per_topic` count topics Jev was never asked about.

    `stale` and `orphans` are what that filter DROPPED. They default to 0 because a caller that
    filtered nothing dropped nothing — but a caller that filtered and then omits them is telling
    the report a retired side-car was empty, which is the one reading that costs money.

    `unassessed` is how many posts of the corpus have NO current answer — never asked, or
    stale. Only a caller that loaded the corpus knows it (`pairs` holds the answered ones).
    """
    comparisons = compare_all(pairs, threshold)
    summary = _summarize(comparisons, pairs, vocab, threshold, now, (stale, orphans, unassessed))
    return summary, comparisons


def summarize(
    pairs: list[tuple[Item, TopicAssessment]],
    vocab: list[Topic],
    threshold: float,
    *,
    now: datetime | None = None,
    stale: int = 0,
    orphans: int = 0,
    unassessed: int = 0,
) -> dict[str, Any]:
    """The summary alone, for a caller that does not need the comparisons.

    Prefer `build_report`: the CLI and the dashboard both want the rows as well, and this
    wrapper throws away a comparison pass they would only redo.
    """
    return build_report(
        pairs, vocab, threshold, now=now, stale=stale, orphans=orphans, unassessed=unassessed
    )[0]


def _summarize(
    comparisons: list[ItemComparison],
    pairs: list[tuple[Item, TopicAssessment]],
    vocab: list[Topic],
    threshold: float,
    now: datetime | None,
    dropped: tuple[int, int, int],
) -> dict[str, Any]:
    """Every number the report and the dashboard quote, computed ONCE, from the same pairs.

    `items_assessed` counts the pairs handed in; `items_compared` counts the ones with an
    enrichment to compare against. They differ exactly by the items Jev has an opinion about
    that the pipeline has not enriched, and collapsing them would hide that population.

    `assessments_stored` is the size of the side-car the pairs were filtered OUT of:
    `len(pairs) + stale + orphans`, a real partition rather than a number taken on trust.

    `generated_at` is stamped HERE rather than at render time so the JSON carries it too — a
    reader of the JSON otherwise cannot say how old the report it is showing is — and so
    `render_report_markdown` stays a pure function of its inputs. An injected `now` must be
    tz-aware for the same reason `TopicAssessment.asked_at` refuses a naive one: we do not
    coerce, because that masks the bug, and a consumer ageing the report against `utcnow` would
    silently read a local timestamp as UTC.
    """
    if now is not None:
        _require_utc_aware("now", now)
    stale, orphans, unassessed = dropped
    assessments = tuple(assessment for _, assessment in pairs)
    totals = _pair_totals(comparisons)
    primary = _primary_totals(comparisons, {topic.slug for topic in vocab})
    return {
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "threshold": threshold,
        "items_assessed": len(pairs),
        "items_compared": len(comparisons),
        # Posts with no current answer (never asked + stale): what `jev topics` asks next.
        "items_unassessed": unassessed,
        "assessments_stored": len(pairs) + stale + orphans,
        "assessments_stale": stale,
        "assessments_orphaned": orphans,
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
        "primary_agree": primary["agree"],
        "primary_agree_pct": _pct(primary["agree"], len(comparisons)),
        "primary_fallback": primary["fallback"],
        "primary_unjudged": primary["unjudged"],
        "primary_unranked": primary["unranked"],
        **_post_totals(comparisons),
        "per_topic": _per_topic(comparisons, vocab),
        "topic_confusion": topic_confusion(comparisons),
        "primary_confusion": primary_confusion(comparisons),
        "confidence_bands": confidence_rows(comparisons, threshold),
    }


def _run_row(run: JevRun) -> dict[str, Any]:
    """One logged pass as the page and the CLI read it, PRICED NOW from its tokens.

    The log stores tokens and never dollars, so this is where a pass acquires a cost — through
    `defaults.tokens_cost_usd`, the same formula a stored assessment is priced with.
    """
    by_provider = run.input_tokens_by_provider
    return {
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat(),
        "requests": run.requests,
        "ok": run.ok,
        "failed": run.failed,
        "input_tokens": run.input_tokens,
        "input_tokens_unknown": run.input_tokens_unknown,
        # `float(...)`: a pass where nothing answered sums an empty generator to `int 0`.
        "cost_usd": float(sum(tokens_cost_usd(tokens, p) for p, tokens in by_provider.items())),
        "providers": sorted(by_provider),
        "unpriced_providers": list(unpriced(by_provider)),
        "models": list(run.models),
        "interrupted": run.interrupted,
    }


def _history_total(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The sum of every priced pass — the "Total" of the cost strip."""

    def total(key: str) -> int:
        return sum(row[key] for row in rows)

    return {
        "runs": len(rows),
        "requests": total("requests"),
        "ok": total("ok"),
        "failed": total("failed"),
        "input_tokens": total("input_tokens"),
        "input_tokens_unknown": total("input_tokens_unknown"),
        "cost_usd": float(sum(row["cost_usd"] for row in rows)),
        "unpriced_providers": sorted({p for row in rows for p in row["unpriced_providers"]}),
    }


def bill(assessments: Sequence[TopicAssessment]) -> dict[str, Any]:
    """How many answers, their tokens and what they cost — UNROUNDED, for a surface to format.

    The summary's own `cost_usd` is rounded to four decimals for the JSON report; a page
    that shows a single post's share needs the exact figure.
    """
    tokens, unknown = input_tokens_total(assessments)
    return {
        "assessments": len(assessments),
        "input_tokens": tokens,
        "input_tokens_unknown": unknown,
        "cost_usd": input_cost_usd(assessments),
        "unpriced_providers": list(unpriced_providers(assessments)),
    }


def _out_of_log(runs: Sequence[JevRun], assessments: dict[str, TopicAssessment]) -> dict[str, Any]:
    """The stored records no logged pass covers — counted and priced, never dropped.

    A record is COVERED iff its `asked_at` falls inside some logged topics pass's
    [`started_at`, `finished_at`] (inclusive). Everything else is outside the log: records
    asked before the log existed, by a copy of xbrain that does not write it, in a pass
    whose line could not be appended, or in a pass killed by SIGTERM. Their bill is priced
    from their own stored tokens.
    """
    windows = [(run.started_at, run.finished_at) for run in runs]
    return bill(
        tuple(
            a
            for a in assessments.values()
            if not any(start <= a.asked_at <= end for start, end in windows)
        )
    )


def run_history(runs: Sequence[JevRun], assessments: dict[str, TopicAssessment]) -> dict[str, Any]:
    """What Jev topic passes have cost over time: every logged pass, their total, and what
    the log never saw.

    `runs` is `store.load_runs`, filtered here to `kind == "topics"` (the file is shared by
    every kind of pass); `assessments` is the RAW side-car, every record including stale
    and orphaned ones — they were all paid for.

    `out_of_log` is kept apart from the total rather than folded into it: the total is what
    the log recorded, and the side-car holds only the LATEST answer per item, so it cannot
    reconstruct the passes the log missed. It is counted and priced so a surface can say so
    instead of quoting a total that silently omits it.
    """
    topics_runs = [run for run in runs if run.kind == "topics"]
    rows = [_run_row(run) for run in topics_runs]
    return {
        "runs": sorted(rows, key=lambda row: row["started_at"], reverse=True),
        "total": _history_total(rows),
        "out_of_log": _out_of_log(topics_runs, assessments),
    }


def history_fragment(history: dict[str, Any]) -> str:
    """`N pasadas · M peticiones · <the shared cost sentence>`, plus what the log never saw.

    With no logged pass it says so — `sin pasadas registradas` — instead of quoting a
    `~0.0000 $` that reads as "free". What is outside the log is named AND priced through
    the same shared sentence.
    """
    total = history["total"]
    if total["runs"]:
        line = (
            f"{plural(total['runs'], 'pasada', 'pasadas')} · "
            f"{plural(total['requests'], 'petición', 'peticiones')} · "
            + jev_cost_fragment(
                total["input_tokens"],
                total["input_tokens_unknown"],
                total["cost_usd"],
                total["unpriced_providers"],
            )
        )
    else:
        line = "sin pasadas registradas"
    outside = history["out_of_log"]
    if outside["assessments"]:
        noun = plural(outside["assessments"], "evaluación", "evaluaciones")
        line += f" · {noun} fuera del registro de pasadas: " + jev_cost_fragment(
            outside["input_tokens"],
            outside["input_tokens_unknown"],
            outside["cost_usd"],
            outside["unpriced_providers"],
        )
    return line


def assessment_cost_usd(assessment: TopicAssessment) -> float | None:
    """What one stored answer cost; `None` when it cannot be said.

    Unknown usage (`input_tokens is None`) and an unpriced provider are both `None`, never
    `0.0`: a zero would read as "free" and would drag a mean down.
    """
    if assessment.input_tokens is None or unpriced([assessment.provider]):
        return None
    return tokens_cost_usd(assessment.input_tokens, assessment.provider)


def post_cost_view(assessments: Sequence[TopicAssessment]) -> dict[str, Any]:
    """The mean cost of one stored answer, over the ones that CAN be priced — and how many.

    `n` answers priced of `of`: without the count, a mean over 3 of 2,600 posts reads like a
    mean over all of them. `mean_usd` is `None` when nothing can be priced.
    """
    costs = [cost for a in assessments if (cost := assessment_cost_usd(a)) is not None]
    tokens = [a.input_tokens for a in assessments if a.input_tokens is not None]
    return {
        "mean_usd": sum(costs) / len(costs) if costs else None,
        "n": len(costs),
        "of": len(assessments),
        "unpriced_providers": list(unpriced_providers(assessments)),
        # Over the answers that REPORTED usage (`tokens_n` of `of`): an unknown count is not 0.
        "mean_tokens": sum(tokens) / len(tokens) if tokens else None,
        "tokens_n": len(tokens),
    }


def pass_estimate(per_post: dict[str, Any], posts: int) -> dict[str, Any]:
    """What asking about `posts` more posts would cost, from `post_cost_view`'s means — an
    ESTIMATE: the mean of the answers already paid for, times a count, at list price.

    `tokens` is the mean input tokens × `posts`, `usd` the mean priced cost × `posts`; either
    is `None` when there is nothing to take its mean over, never a 0 that reads as free.
    """
    mean_tokens, mean_usd = per_post["mean_tokens"], per_post["mean_usd"]
    return {
        "posts": posts,
        "tokens": None if mean_tokens is None else round(mean_tokens * posts),
        "usd": None if mean_usd is None else mean_usd * posts,
    }


def _escape_cell(text: str) -> str:
    r"""Make `text` safe to drop between two `|` in a markdown table row.

    THE ESCAPE CHARACTER FIRST, THEN THE PIPE, and the order is the whole point. Escaping only
    the pipe turns a post that already contains `a\|b` — a regex alternation, a shell escape, a
    Windows path — into `a\\|b`, and cmark-gfm's row scanner consumes `\` plus ONE following
    character: it eats both backslashes and the `|` is left as a LIVE cell delimiter. The row
    gains a column, every column after it shifts, and the table silently misreports itself.
    Doubling the backslash first means every `\|` the reader sees was written by us.

    Applied to every interpolated value, not just the post text: `Topic.slug` is
    pattern-constrained, but `Enrichment.primary_topic` is a bare `str` and
    `[jev].fallback_option` is a free config string — and an item whose primary left the
    vocabulary carries by construction a string today's validation never saw.
    """
    return text.replace("\\", "\\\\").replace("|", "\\|")


def _snippet(item: Item | None, width: int = 80) -> str:
    """The post on one line, cut to `width` (`width - 1` plus an ellipsis) and ESCAPED.

    Two different ways a post re-shapes the table it lands in, and both are handled here: a
    newline ends the row, and an unescaped `|` opens a new cell. The corpus is AI/dev posts
    from X, where `cat x | grep y` and `A | B` phrasing are ordinary text — `cost | benefit |
    ratio` turns a four-cell row into seven — so this is the common case, not a defensive
    flourish.

    The escape runs AFTER the cut, so the cut can never land between a backslash and its
    pipe and leave a dangling escape. It can push the cell past `width`, which is fine:
    `width` bounds the TEXT a reader has to scan, not the rendered cell.
    """
    text = " ".join(item.text.split()) if item else ""
    if len(text) > width:
        text = text[: width - 1] + "…"
    return _escape_cell(text)


def _counted(counts: dict[str, int]) -> str:
    """`name (n), name (n)` for a name→count map, sorted; an em dash when it is empty."""
    return ", ".join(f"{name} ({count})" for name, count in sorted(counts.items())) or "—"


def cost_fragment(summary: dict[str, Any]) -> str:
    """The bill, in the ONE Spanish sentence `jev topics` and `jev report` also print.

    PUBLIC because `cli._jev_report_line` prints the same segment: which summary keys feed the
    fragment is itself a definition, and two adapters are two of them.

    Formatted by `defaults.jev_cost_fragment`, never here: this is a RECAP of work `jev topics`
    already paid for, and a recap that prints a different figure — or the same marker in
    different words — from the bill it recaps is the one thing a recap must not do.
    """
    return jev_cost_fragment(
        summary["input_tokens"],
        summary["input_tokens_unknown"],
        summary["cost_usd"],
        summary["unpriced_providers"],
    )


def _tally(singular: str, plural_label: str, counts: dict[str, int]) -> str:
    """`modelo: x (1)` / `modelos: x (1), y (2)` — the label agrees with how many there are."""
    return f"{singular if len(counts) == 1 else plural_label}: {_counted(counts)}"


def _side_car_line(summary: dict[str, Any]) -> str:
    """How much of the side-car this report is actually about.

    `caducadas` is printed even at zero. It is the number that distinguishes "nobody has run
    `xbrain jev topics` yet" from "a vocabulary edit just retired every paid record you have",
    and those two readings differ by the price of re-assessing the corpus. `huérfanas` is rarer,
    so it appears only when it has something to say.
    """
    line = (
        f"Evaluaciones: {summary['items_assessed']} vigentes de "
        f"{summary['assessments_stored']} guardadas · "
        f"{plural(summary['assessments_stale'], 'caducada', 'caducadas')}"
    )
    if summary["assessments_orphaned"]:
        line += f" · {plural(summary['assessments_orphaned'], 'huérfana', 'huérfanas')}"
    return line


def _headline(summary: dict[str, Any]) -> list[str]:
    """Title, what was asked, and the headline counts.

    In reading order: how much of the side-car is current, agreement in both directions, the
    three disagreement buckets, the primary counters and the bill.
    """
    return [
        # ISO 8601 puts the calendar date in the first ten characters, so the title and the
        # JSON's `generated_at` can never name different days.
        f"# Jev · topics — {summary['generated_at'][:10]}",
        "",
        # Three decimals, like the noul column it is read against and like the page's own
        # umbral: a reader compares the two, and one at `0.85` beside a column at `0.850`
        # is the like-for-like comparison this report exists to make, broken.
        f"Umbral {summary['threshold']:.3f} · {_tally('modelo', 'modelos', summary['models'])} · "
        f"{_tally('proveedor', 'proveedores', summary['providers'])}",
        _side_car_line(summary),
        f"Items comparados: {summary['items_compared']} de {summary['items_assessed']} · "
        f"truncados: {summary['truncated']}",
        "",
        f"**Asignaciones de enrich respaldadas por Jev:** {summary['assigned_backed']} de "
        f"{summary['assigned_pairs']} ({summary['enrich_backed_pct']} %) · "
        f"**asignaciones de Jev que enrich tiene:** {summary['jev_backed']} de "
        f"{summary['jev_pairs']} ({summary['jev_backed_pct']} %)",
        f"**Dudosas:** {summary['doubtful_pairs']} · **sin juzgar:** "
        f"{summary['assigned_unjudged']} · **candidatas que faltan:** "
        f"{summary['missing_pairs']} · **primario coincide:** {summary['primary_agree']} "
        f"({summary['primary_agree_pct']} %) · **primario = fallback:** "
        f"{summary['primary_fallback']} · **primario sin juzgar:** "
        f"{summary['primary_unjudged']} · **primario sin rango:** "
        f"{summary['primary_unranked']}",
        f"Coste: {cost_fragment(summary)}",
    ]


def _table(title: str, columns: Sequence[str], rows: list[str]) -> list[str]:
    """A titled markdown table, header included — always rendered, even with no rows.

    An empty table says "nothing disagreed here"; omitting the section entirely would read as
    "this report does not cover that", which is the opposite claim.
    """
    return ["", title, "", f"| {' | '.join(columns)} |", "|" + "---|" * len(columns), *rows]


def _cut_note(shown: int, total: int) -> list[str]:
    """`… y N filas más` under a table the `top` cut.

    A section headed `top 20` reads the same whether there were 7 rows or 700. The JSON carries
    every one of them, which is exactly why the file a PERSON reads has to say that it does not
    — a silent drop in the artifact that exists to be read is the same failure as a silent drop
    in a number.
    """
    dropped = total - shown
    if dropped <= 0:
        return []
    return ["", f"_… y {plural(dropped, 'fila más', 'filas más')} (el JSON las lleva todas)._"]


def _pair_table(title: str, pairs: list[Pair], items_by_id: dict[str, Item], top: int) -> list[str]:
    """The `top` worst pairs as a table; the caller has already sorted them.

    THREE decimals, like `jev.html`, and for the reason the page was moved to three: these
    rows are read AGAINST the umbral — `## Dudosas` means "below it" — and at two decimals a
    pair at 0.8496 prints `0.85` under a headline reading `Umbral 0.85`, so the row denies
    the section it is in. One side-car must not quote two numbers for one value, and the
    `top` cut sorts the boundary rows away only on a corpus large enough to have twenty
    worse ones, so the contradiction surfaces exactly where every row gets read.

    `_disagreement_row`'s confidence stays at two: it is never compared against the umbral,
    and the page prints it at two as well.
    """
    rows = [
        f"| {pair.item_id} | {_escape_cell(pair.slug)} | {pair.noul:.3f} | "
        f"{_snippet(items_by_id.get(pair.item_id))} |"
        for pair in pairs[:top]
    ]
    return _table(title, ("item", "topic", "noul", "texto"), rows) + _cut_note(
        len(rows), len(pairs)
    )


def _disagreement_row(comparison: ItemComparison, items_by_id: dict[str, Item]) -> str:
    """One row of a primary-mismatch section. WHY it does not coincide is the section it is in,
    because `rango = —` means "never asked", "the provider omitted the option" AND "no primary
    at all" — it cannot carry the reason on its own."""
    return (
        f"| {comparison.item_id} | {_escape_cell(comparison.primary_topic or '—')} "
        f"| {_escape_cell(comparison.jev_primary)} | {comparison.jev_confidence:.2f} "
        f"| {comparison.primary_rank or '—'} | "
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


def _sorted_unjudged(comparisons: list[ItemComparison]) -> list[tuple[str, str]]:
    """Every `(item, topic)` pair Jev was never asked about, grouped by topic.

    Sorted by slug so the retired topics cluster: the decision this table exists to prompt —
    put the slug back in `vocab.yaml`, or re-enrich these items — is a per-topic one.
    """
    entries = ((c.item_id, slug) for c in comparisons for slug in c.unjudged)
    return sorted(entries, key=lambda entry: (entry[1], entry[0]))


def _unjudged_table(
    comparisons: list[ItemComparison], items_by_id: dict[str, Item], top: int
) -> list[str]:
    """The pairs the headline counts as `sin juzgar`, NAMED.

    The count alone cannot tell a reader WHICH topic to put back in `vocab.yaml`. The JSON
    record carries the slugs and the file a person reads did not, which made the one bucket
    that says "your vocabulary moved" the one bucket nobody could act on.
    """
    entries = _sorted_unjudged(comparisons)
    rows = [
        f"| {item_id} | {_escape_cell(slug)} | {_snippet(items_by_id.get(item_id))} |"
        for item_id, slug in entries[:top]
    ]
    title = f"## Asignaciones sin juzgar (top {top})"
    return _table(title, ("item", "topic", "texto"), rows) + _cut_note(len(rows), len(entries))


def _primary_reason(comparison: ItemComparison, slugs: set[str]) -> str:
    """Why this primary does not coincide — the FIRST reason in `_PRIMARY_REASONS` that applies.

    Priority, not exclusivity: an item can be both `sin juzgar` and answered with the fallback.
    Ordering the enrich-side facts first puts each item in the section whose remedy is the one
    the operator controls.
    """
    if comparison.primary_topic is None:
        return "sin_primario"
    if comparison.primary_unjudged:
        return "sin_juzgar"
    if comparison.jev_primary not in slugs:
        return "fallback"
    if comparison.primary_unranked:
        return "sin_rango"
    return "desacuerdo"


def _sorted_disagreements(comparisons: list[ItemComparison]) -> list[ItemComparison]:
    """Most consequential first: enrich's primary furthest DOWN Jev's own ranking.

    A primary at rank 29 of 30 is a real conflict; one at rank 2 is a close call, and the file
    shows only the first `top` rows — unsorted, those were simply the first `top` in
    `items.json` order. Unranked items (`—`) sort last within their section: there is no rank to
    be worst at, and they already have a section of their own.
    """
    return sorted(
        comparisons,
        key=lambda c: (c.primary_rank is None, -(c.primary_rank or 0), c.item_id),
    )


def _primary_section(
    title: str, comparisons: list[ItemComparison], items_by_id: dict[str, Item], top: int
) -> list[str]:
    """One reason's sub-table, with its own count in the heading."""
    ranked = _sorted_disagreements(comparisons)
    rows = [_disagreement_row(c, items_by_id) for c in ranked[:top]]
    return _table(
        f"### {title} ({len(comparisons)})",
        ("item", "enrich", "Jev", "conf.", "rango del de enrich", "texto"),
        rows,
    ) + _cut_note(len(rows), len(ranked))


def _primary_tables(
    comparisons: list[ItemComparison], slugs: set[str], items_by_id: dict[str, Item], top: int
) -> list[str]:
    """The primary mismatches, SPLIT BY REASON.

    One table headed "en desacuerdo" would say the opposite of what this module spends forty
    lines establishing: an item with no primary, and an item whose primary left the vocabulary,
    are not disagreements — "there is nothing to agree with" is not agreement, and neither is it
    a conflict. The headline counts those buckets two lines above; a single table would erase
    the distinction the headline just drew.

    THE SECTIONS AND THE HEADLINE COUNT DIFFERENTLY, on purpose. `_primary_totals`' counters are
    INDEPENDENT — an item whose primary left the vocabulary AND that Jev answered with the
    fallback is counted by both `primario sin juzgar` and `primario = fallback`, because each
    answers its own question about the corpus. The sections PARTITION: that item appears once,
    under the first reason in `_PRIMARY_REASONS` that applies, because a row a reader has to act
    on must not be filed in two places. So the section counts can sum to less than the headline
    counters, and that is not a discrepancy.
    """
    grouped: dict[str, list[ItemComparison]] = {key: [] for key, _ in _PRIMARY_REASONS}
    for comparison in comparisons:
        if not comparison.primary_agrees:
            grouped[_primary_reason(comparison, slugs)].append(comparison)
    lines = [
        "",
        f"## Primario que no coincide (top {top} por motivo)",
        "",
        "_Los contadores de la cabecera son independientes: un item cuyo primario salió del "
        "vocabulario y que además Jev resolvió con el fallback cuenta en los dos. Aquí cada "
        "item aparece UNA vez, bajo el primer motivo que le aplica, así que estas cifras "
        "pueden sumar menos que las de arriba._",
    ]
    for key, title in _PRIMARY_REASONS:
        lines += _primary_section(title, grouped[key], items_by_id, top)
    return lines


#: How the markdown names each kind of band — the Posts filter names.
_BAND_KIND_NAMES = {"enrich_only": "enrich asigna y Jev no", "jev_only": "Jev añadiría"}


def _band_range(band: dict[str, Any]) -> str:
    """A band's edges at three decimals, like the umbral: `< hi` for the enrich side's bottom
    band, `≥ lo` for the Jev side's top one, `lo – hi` otherwise (each band holds its lower
    edge)."""
    if band["kind"] == "enrich_only" and band["lo"] == 0:
        return f"< {band['hi']:.3f}"
    if band["top"]:
        return f"≥ {band['lo']:.3f}"
    return f"{band['lo']:.3f} – {band['hi']:.3f}"


def _bands_table(summary: dict[str, Any]) -> list[str]:
    """How sure Jev was on the disagreements: `confidence_bands`, one row per band."""
    rows = [
        f"| {_BAND_KIND_NAMES[b['kind']]} | {b['label']} | {_band_range(b)} | {b['pairs']} | "
        f"{b['posts']} |"
        for b in summary["confidence_bands"]
    ]
    return _table(
        "## Qué seguro estaba Jev en los desacuerdos",
        ("desacuerdo", "tramo", "probabilidad", "desacuerdos", "posts"),
        rows,
    )


def render_report_markdown(
    summary: dict[str, Any],
    comparisons: list[ItemComparison],
    items_by_id: dict[str, Item],
    *,
    top: int = 20,
) -> str:
    """The human-readable report: the headline numbers, then the worst `top` of each table.

    TRUNCATED ON PURPOSE. The corpus is ~3,000 items and the JSON beside it carries every
    record, so this file is the thing a person reads — a table with 2,000 rows is not one. Every
    table that was cut says so, in `_cut_note`.

    The vocabulary comes from `summary["per_topic"]`, which has exactly one row per vocabulary
    topic: the renderer needs the slug set to tell "Jev chose the fallback" from a real
    disagreement, and taking it from the summary keeps this function a pure function of its
    two inputs.
    """
    slugs = {row["slug"] for row in summary["per_topic"]}
    lines = _headline(summary)
    lines += _table(
        "## Por topic (peor primero)",
        ("topic", "asignadas", "respaldadas", "%", "candidatas"),
        [
            f"| {_escape_cell(row['slug'])} | {row['assigned']} | {row['backed']} | "
            f"{row['backed_pct']} | {row['missing']} |"
            for row in summary["per_topic"]
        ],
    )
    lines += _bands_table(summary)
    lines += _pair_table(
        f"## Dudosas (top {top}, noul ascendente)", _sorted_doubtful(comparisons), items_by_id, top
    )
    lines += _pair_table(
        f"## Candidatas que faltan (top {top}, noul descendente)",
        _sorted_missing(comparisons),
        items_by_id,
        top,
    )
    lines += _unjudged_table(comparisons, items_by_id, top)
    lines += _primary_tables(comparisons, slugs, items_by_id, top)
    return "\n".join(lines) + "\n"


def _record(comparison: ItemComparison) -> dict[str, Any]:
    """One compared item as JSON: every field, with the tuples flattened to lists.

    Built from `asdict` rather than field by field so a field added to `ItemComparison` shows
    up in the JSON — a reader of it gets the new field — without a second edit here to
    remember. The two `Pair` fields are re-rendered by hand: their `item_id` is the record's own
    id repeated on every row.
    """
    return {
        **{k: v for k, v in asdict(comparison).items() if k not in _RECORD_SKIP},
        "assigned": list(comparison.assigned),
        "jev_assigned": list(comparison.jev_assigned),
        "unjudged": list(comparison.unjudged),
        "doubtful": [{"slug": p.slug, "noul": p.noul} for p in comparison.doubtful],
        "missing": [{"slug": p.slug, "noul": p.noul} for p in comparison.missing],
        # Properties are not fields, so `asdict` cannot see them; they are the two per-item
        # reasons a reader of the JSON needs and would otherwise have to re-derive.
        "primary_agrees": comparison.primary_agrees,
        "primary_unranked": comparison.primary_unranked,
    }


def report_paths(jev_dir: Path) -> tuple[Path, Path]:
    """`(topics-report.json, topics-report.md)` under `jev_dir` — where `write_reports` writes,
    named once for every reader that points at them (the CLI, the Configuración tab)."""
    return jev_dir / "topics-report.json", jev_dir / "topics-report.md"


def write_reports(
    summary: dict[str, Any],
    comparisons: list[ItemComparison],
    items_by_id: dict[str, Item],
    jev_dir: Path,
) -> tuple[Path, Path]:
    """`topics-report.json` (summary + one record per compared item) and `topics-report.md`.

    Two files, one computation: the markdown is what a person reads and the JSON is what a
    program reads, and both are rendered from the SAME `summary` and `comparisons`, so they
    cannot quote different numbers.

    ATOMIC INDIVIDUALLY, NOT JOINTLY, and the difference is worth stating rather than leaving
    to be discovered. Each file goes through `store._atomic_write` (temp file, then
    `os.replace`), so neither can ever be read half-written and a failed write leaves the
    PREVIOUS good copy of that file untouched. But they are two renames: if the second dies,
    the JSON is the new run's and the markdown is still the previous run's. The JSON is written
    FIRST because it is the machine-readable one — a program comparing `generated_at` can see
    the pair is mismatched, while a person reading a stale markdown has no such signal.

    The directory is created if it does not exist: on a corpus assessed elsewhere, this report
    is the first thing that ever lands in `data/jev/`.
    """
    jev_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = report_paths(jev_dir)
    payload = {"summary": summary, "items": [_record(c) for c in comparisons]}
    _atomic_write(json_path, json.dumps(payload, indent=2, ensure_ascii=False))
    _atomic_write(md_path, render_report_markdown(summary, comparisons, items_by_id))
    return json_path, md_path
