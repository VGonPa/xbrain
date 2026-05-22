"""Structured diff between two snapshot data directories.

The `diff_snapshots` orchestrator answers one question: **what changed between
two states of `data/`?** It compares the source-of-truth artifacts (items.json
enrichment, vocab.yaml, topics.json) and produces a structured `DiffReport`
that the CLI renders as text or JSON.

The module is **pure I/O**: no `typer.echo`, no `print`, no CLI side-effects.
Inputs are two `Path`s pointing at *data directories* (the ones that hold
`items.json` / `vocab.yaml` / `topics.json` directly — a snapshot dir or the
live `data/` dir; the module does not distinguish). The CLI is the only thing
that knows what a "snapshot name" is and resolves it via
`xbrain.snapshot.snapshot_show`.

Overview drift uses a pure-Python TF cosine similarity (see `_tf_cosine`).
No new dependencies: scikit-learn or sentence-transformers would tax the
install for one CLI feature, and the offline-by-default invariant rules out
API-call embeddings. LLM-judged similarity is the explicit follow-up tied to
WS3 (#8) — out of scope for v1.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from xbrain.models import Item, Topic, TopicPage
from xbrain.rubrics import load_vocab
from xbrain.store import load_store, load_topic_pages

# --------------------------------------------------------------------- public models

#: Three-state classification of how much a topic-page overview moved.
#: `not_comparable` covers the case where one side has no topic-page entry.
OverviewFlag = Literal["identical", "similar", "different", "not_comparable"]

#: Topics with fewer than this many starting members do not get a growth/shrink
#: flag — a 2→3 jump is 50% growth but statistically meaningless on a small
#: topic. Hardcoded floor; promote to a parameter if user research demands.
_MIN_MEMBERS_FOR_GROWTH_FLAG = 5

#: Cosine similarity at or above this counts as "identical" (handles fp slop).
_IDENTICAL_SIMILARITY = 0.99

# Tokenizer: lowercase, ASCII letters/digits + Romance-language Latin
# accented characters (à-ÿ) so Spanish/French overviews tokenize correctly.
# Length filter happens in the helper — single-character tokens add noise.
_TOKEN_RE = re.compile(r"[a-zà-ÿ0-9]+")


class Transition(BaseModel):
    """One `primary_topic` reassignment between snapshot A and B.

    `from_topic` / `to_topic` carry `None` when the item had no enrichment on
    that side (e.g. unenriched in A, enriched in B → `from_topic=None`). The
    counter aggregates identical (from, to) pairs across all items.
    """

    from_topic: str | None
    to_topic: str | None
    count: int


class ItemsDiff(BaseModel):
    """Item-level reassignment summary across two snapshots."""

    count_a: int
    count_b: int
    count_in_both: int
    enriched_in_both: int
    reassigned: int
    reassigned_pct: float
    top_transitions: list[Transition] = Field(default_factory=list)


class TopicChange(BaseModel):
    """Per-topic membership shift and overview drift.

    A "member" is an item whose `primary_topic` equals this topic's slug. The
    set ops (`added`, `removed`, `unchanged`) operate on the item ids — they
    answer "which items entered or left this topic between A and B", not just
    "did the count move".
    """

    slug: str
    members_a: int
    members_b: int
    added: int
    removed: int
    unchanged: int
    growth_pct: float | None
    flagged_growth: bool
    overview_a: str | None = None
    overview_b: str | None = None
    overview_similarity: float | None = None
    overview_flag: OverviewFlag = "not_comparable"


class TopicsDiff(BaseModel):
    """All per-topic changes, keyed by slug (union of vocab A and vocab B)."""

    per_slug: dict[str, TopicChange]


class VocabDiff(BaseModel):
    """Slug-level set difference between two `vocab.yaml` files.

    `unchanged_count` carries the cardinality of slugs present in both vocabs
    — only the count is consumed downstream (text renderer, JSON snapshot),
    so the full list is not stored.
    """

    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    unchanged_count: int = 0


class DiffSummary(BaseModel):
    """Flat top-level counts — used by both the text header and JSON consumers."""

    items_a: int
    items_b: int
    items_in_both: int
    enriched_in_both: int
    reassigned: int
    reassigned_pct: float
    vocab_added: int
    vocab_removed: int
    topic_pages_a: int
    topic_pages_b: int


class DiffReport(BaseModel):
    """The full structured comparison between two data directories."""

    summary: DiffSummary
    items: ItemsDiff
    topics: TopicsDiff
    vocab: VocabDiff


# ----------------------------------------------------------------- public orchestrator


def diff_snapshots(
    a_dir: Path,
    b_dir: Path,
    *,
    overview_similarity_threshold: float = 0.7,
    growth_flag_threshold: float = 0.10,
    top_n_transitions: int = 10,
) -> DiffReport:
    """Compute the structured diff between two snapshot data directories.

    `a_dir` and `b_dir` are *data directories* — the ones holding `items.json`,
    `vocab.yaml`, `topics.json` directly. A snapshot directory and the live
    `data/` directory are the same shape; the CLI passes whichever the user
    asked for, and this function does not care which is which.

    The thresholds tune which transitions get flagged but do not affect the
    raw counts. Defaults match PRD §5.2.

    Raises:
        FileNotFoundError: if either directory does not exist on disk, or if
            both directories are completely empty (no `items.json`,
            `vocab.yaml` or `topics.json`). Without this guard a `diff` against
            an accidentally-deleted `data/` would silently report
            "everything removed" — surfacing as a clean error is safer.
        ValueError: a context-adding wrap around the corrupt-file errors
            raised by the underlying loaders, so the operator sees *which*
            file is the problem rather than a bare pydantic / json traceback.
    """
    for label, path in (("A", a_dir), ("B", b_dir)):
        if not path.exists():
            raise FileNotFoundError(f"diff side {label}: directory not found: {path}")

    def _load_or_explain(path: Path, loader):
        try:
            return loader(path)
        except (ValueError, OSError) as exc:  # pydantic ValidationError is a ValueError
            raise ValueError(f"failed to load {path}: {exc}") from exc

    items_a = _load_or_explain(a_dir / "items.json", load_store)
    items_b = _load_or_explain(b_dir / "items.json", load_store)
    vocab_a = _load_or_explain(a_dir / "vocab.yaml", load_vocab)
    vocab_b = _load_or_explain(b_dir / "vocab.yaml", load_vocab)
    pages_a = _load_or_explain(a_dir / "topics.json", load_topic_pages)
    pages_b = _load_or_explain(b_dir / "topics.json", load_topic_pages)

    a_empty = not (items_a or vocab_a or pages_a)
    b_empty = not (items_b or vocab_b or pages_b)
    if a_empty and b_empty:
        raise FileNotFoundError(
            f"Both diff sides are empty (no items.json / vocab.yaml / topics.json "
            f"present under {a_dir} or {b_dir}). Confirm the snapshot names and "
            "that data/ is populated."
        )

    items_diff = _compute_items_diff(items_a, items_b, top_n=top_n_transitions)
    vocab_diff = _compute_vocab_diff(vocab_a, vocab_b)
    topics_diff = _compute_topics_diff(
        items_a,
        items_b,
        pages_a,
        pages_b,
        vocab_a,
        vocab_b,
        growth_flag_threshold=growth_flag_threshold,
        overview_similarity_threshold=overview_similarity_threshold,
    )
    summary = DiffSummary(
        items_a=items_diff.count_a,
        items_b=items_diff.count_b,
        items_in_both=items_diff.count_in_both,
        enriched_in_both=items_diff.enriched_in_both,
        reassigned=items_diff.reassigned,
        reassigned_pct=items_diff.reassigned_pct,
        vocab_added=len(vocab_diff.added),
        vocab_removed=len(vocab_diff.removed),
        topic_pages_a=len(pages_a),
        topic_pages_b=len(pages_b),
    )
    return DiffReport(
        summary=summary,
        items=items_diff,
        topics=topics_diff,
        vocab=vocab_diff,
    )


# --------------------------------------------------------------------- compute pieces


def _compute_items_diff(
    items_a: dict[str, Item],
    items_b: dict[str, Item],
    *,
    top_n: int,
) -> ItemsDiff:
    """Reassignment count + top transitions, restricted to items in both.

    `reassigned` counts items present AND enriched on both sides whose
    `primary_topic` differs. Items added in B or un-enriched on either side
    do NOT count as reassignments — they show up in transitions and topic
    membership instead.
    """
    shared_ids = set(items_a) & set(items_b)
    transitions: list[tuple[str | None, str | None]] = []
    reassigned = 0
    enriched_in_both = 0
    for item_id in shared_ids:
        primary_a = _primary_topic(items_a[item_id])
        primary_b = _primary_topic(items_b[item_id])
        if primary_a is not None and primary_b is not None:
            enriched_in_both += 1
            if primary_a != primary_b:
                reassigned += 1
                transitions.append((primary_a, primary_b))
        elif primary_a != primary_b:
            # One side unenriched. Surface as a transition (None on one side)
            # but do NOT count as a reassignment — only judgment changes on
            # items judged on BOTH sides count.
            transitions.append((primary_a, primary_b))
    pct = (reassigned / enriched_in_both) if enriched_in_both else 0.0
    return ItemsDiff(
        count_a=len(items_a),
        count_b=len(items_b),
        count_in_both=len(shared_ids),
        enriched_in_both=enriched_in_both,
        reassigned=reassigned,
        reassigned_pct=pct,
        top_transitions=_top_transitions(transitions, top_n=top_n),
    )


def _compute_topics_diff(
    items_a: dict[str, Item],
    items_b: dict[str, Item],
    pages_a: dict[str, TopicPage],
    pages_b: dict[str, TopicPage],
    vocab_a: list[Topic],
    vocab_b: list[Topic],
    *,
    growth_flag_threshold: float,
    overview_similarity_threshold: float,
) -> TopicsDiff:
    """Per-slug membership and overview drift, keyed by the union of slugs."""
    membership_a = _membership_by_topic(items_a)
    membership_b = _membership_by_topic(items_b)

    # Union over: vocab slugs from both sides + any slug that appears as a
    # primary_topic in either store (defensive — items can reference a slug
    # that the vocab no longer carries).
    all_slugs: set[str] = {t.slug for t in vocab_a} | {t.slug for t in vocab_b}
    all_slugs |= set(membership_a) | set(membership_b)

    per_slug: dict[str, TopicChange] = {}
    for slug in sorted(all_slugs):
        member_ids_a = membership_a.get(slug, set())
        member_ids_b = membership_b.get(slug, set())
        added = len(member_ids_b - member_ids_a)
        removed = len(member_ids_a - member_ids_b)
        unchanged = len(member_ids_a & member_ids_b)
        growth_pct = _growth_pct(len(member_ids_a), len(member_ids_b))
        flagged_growth = (
            growth_pct is not None
            and len(member_ids_a) >= _MIN_MEMBERS_FOR_GROWTH_FLAG
            and abs(growth_pct) >= growth_flag_threshold
        )
        overview_a = pages_a[slug].overview if slug in pages_a else None
        overview_b = pages_b[slug].overview if slug in pages_b else None
        similarity, flag = _classify_overview(
            overview_a, overview_b, threshold=overview_similarity_threshold
        )
        per_slug[slug] = TopicChange(
            slug=slug,
            members_a=len(member_ids_a),
            members_b=len(member_ids_b),
            added=added,
            removed=removed,
            unchanged=unchanged,
            growth_pct=growth_pct,
            flagged_growth=flagged_growth,
            overview_a=overview_a,
            overview_b=overview_b,
            overview_similarity=similarity,
            overview_flag=flag,
        )
    return TopicsDiff(per_slug=per_slug)


def _compute_vocab_diff(vocab_a: list[Topic], vocab_b: list[Topic]) -> VocabDiff:
    """Slug set-difference between two vocab.yaml files (sorted output)."""
    slugs_a = {t.slug for t in vocab_a}
    slugs_b = {t.slug for t in vocab_b}
    return VocabDiff(
        added=sorted(slugs_b - slugs_a),
        removed=sorted(slugs_a - slugs_b),
        unchanged_count=len(slugs_a & slugs_b),
    )


# --------------------------------------------------------------------------- TF cosine


def _tf_cosine(text_a: str, text_b: str) -> float:
    """Return TF cosine similarity in [0.0, 1.0] for two short prose strings.

    Tokenization: lowercase, runs of ASCII or Romance-Latin letters/digits of
    length >= 2. With only two documents IDF degenerates, so this is plain TF
    cosine (not TF-IDF) — relative ordering across topic pairs is what matters
    for the `identical / similar / different` bucket assignment, not the
    absolute score.

    Edge cases:
      - Either text empty or contains no qualifying tokens → 0.0.
      - Identical token bags → 1.0.
      - Disjoint vocabularies → 0.0.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)
    shared = set(counter_a) & set(counter_b)
    if not shared:
        return 0.0
    numerator = sum(counter_a[t] * counter_b[t] for t in shared)
    norm_a = math.sqrt(sum(c * c for c in counter_a.values()))
    norm_b = math.sqrt(sum(c * c for c in counter_b.values()))
    return numerator / (norm_a * norm_b)


def _tokenize(text: str) -> list[str]:
    """Tokens of length >= 2, lowercased, ASCII + Latin-1 letters/digits."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2]


# ----------------------------------------------------------------------- renderers


def format_text(report: DiffReport) -> str:
    """Render the diff as a human-readable terminal report."""
    lines: list[str] = []
    lines.append("ITEMS")
    lines.append(f"  count A: {report.summary.items_a}")
    lines.append(f"  count B: {report.summary.items_b}")
    lines.append(f"  in both snapshots: {report.summary.items_in_both}")
    lines.append(f"  enriched on both sides: {report.summary.enriched_in_both}")
    lines.append(
        f"  primary_topic reassigned: {report.summary.reassigned} "
        f"({report.summary.reassigned_pct * 100:.1f}%)"
    )
    if report.items.top_transitions:
        lines.append("")
        lines.append("  Top transitions:")
        for transition in report.items.top_transitions:
            from_label = transition.from_topic if transition.from_topic is not None else "(none)"
            to_label = transition.to_topic if transition.to_topic is not None else "(none)"
            lines.append(f"    {from_label} -> {to_label}: {transition.count} items")
    lines.append("")
    lines.append("TOPICS")
    lines.extend(_format_topics_block(report.topics))
    lines.append("")
    lines.append("OVERVIEWS")
    lines.extend(_format_overviews_block(report.topics))
    lines.append("")
    lines.append("VOCAB")
    lines.append(
        f"  added ({len(report.vocab.added)}): "
        f"{', '.join(report.vocab.added) if report.vocab.added else '(none)'}"
    )
    lines.append(
        f"  removed ({len(report.vocab.removed)}): "
        f"{', '.join(report.vocab.removed) if report.vocab.removed else '(none)'}"
    )
    lines.append(f"  unchanged: {report.vocab.unchanged_count} slugs")
    return "\n".join(lines)


def format_json(report: DiffReport) -> str:
    """Render the diff as pretty JSON for machine consumption."""
    return report.model_dump_json(indent=2)


def _format_topics_block(topics: TopicsDiff) -> list[str]:
    """Render the per-topic membership rows, sorted by slug."""
    if not topics.per_slug:
        return ["  (no topics in either snapshot)"]
    rows: list[str] = []
    for slug in sorted(topics.per_slug):
        change = topics.per_slug[slug]
        flag = " [FLAG]" if change.flagged_growth else ""
        growth = "n/a" if change.growth_pct is None else f"{change.growth_pct * 100:+.1f}%"
        rows.append(
            f"  {slug}{flag}  A={change.members_a}  B={change.members_b}  "
            f"added={change.added}  removed={change.removed}  "
            f"unchanged={change.unchanged}  growth={growth}"
        )
    return rows


def _format_overviews_block(topics: TopicsDiff) -> list[str]:
    """Render the overview-drift rows, with a flagged-shifts sub-block."""
    if not topics.per_slug:
        return ["  (no overviews to compare)"]
    rows: list[str] = []
    flagged: list[tuple[str, float]] = []
    for slug in sorted(topics.per_slug):
        change = topics.per_slug[slug]
        if change.overview_flag == "not_comparable":
            rows.append(f"  {slug}: not comparable")
            continue
        sim = change.overview_similarity if change.overview_similarity is not None else 0.0
        rows.append(f"  {slug}: {change.overview_flag} (sim={sim:.2f})")
        if change.overview_flag == "different":
            flagged.append((slug, sim))
    if flagged:
        rows.append("")
        rows.append("  Sharp shifts (low similarity):")
        for slug, sim in flagged:
            rows.append(f"    {slug}  sim={sim:.2f}")
    return rows


# ------------------------------------------------------------------------- internals


def _primary_topic(item: Item) -> str | None:
    """Return the item's primary_topic, or None when unenriched."""
    if item.enriched is None:
        return None
    return item.enriched.primary_topic


def _membership_by_topic(items: dict[str, Item]) -> dict[str, set[str]]:
    """Group item ids by their primary_topic (only enriched items contribute)."""
    membership: dict[str, set[str]] = {}
    for item_id, item in items.items():
        primary = _primary_topic(item)
        if primary is None:
            continue
        membership.setdefault(primary, set()).add(item_id)
    return membership


def _growth_pct(members_a: int, members_b: int) -> float | None:
    """Relative growth from A to B; None when A is empty (undefined growth)."""
    if members_a == 0:
        return None
    return (members_b - members_a) / members_a


def _top_transitions(
    transitions: Iterable[tuple[str | None, str | None]],
    *,
    top_n: int,
) -> list[Transition]:
    """Aggregate (from, to) pairs and return the top N, with a stable order.

    Sort key: count desc, then `from_topic` asc, then `to_topic` asc. `None`
    is coalesced to the empty string for ordering purposes only — the surface
    representation keeps `None` intact.
    """
    counter: Counter[tuple[str | None, str | None]] = Counter(transitions)
    if not counter:
        return []
    pairs = sorted(
        counter.items(),
        key=lambda kv: (-kv[1], kv[0][0] or "", kv[0][1] or ""),
    )
    return [
        Transition(from_topic=from_t, to_topic=to_t, count=count)
        for (from_t, to_t), count in pairs[:top_n]
    ]


def _classify_overview(
    text_a: str | None,
    text_b: str | None,
    *,
    threshold: float,
) -> tuple[float | None, OverviewFlag]:
    """Compute similarity and bucket it into the OverviewFlag enum."""
    if text_a is None or text_b is None:
        return None, "not_comparable"
    similarity = _tf_cosine(text_a, text_b)
    if similarity >= _IDENTICAL_SIMILARITY:
        return similarity, "identical"
    if similarity >= threshold:
        return similarity, "similar"
    return similarity, "different"
