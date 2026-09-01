"""`search` as a SERVICE (spec §5.3, §5.4, §7.1, §7.2) — CLI and MCP are adapters over it.

Spec §7.1: *CLI JSON and MCP are adapters over the same application services. They do not
implement two searches.* The way two searches appear is never a decision — the second adapter
is written months later against whatever the first happened to emit — so the service returns
a `SearchResponse` and the adapters render it. Plan 04's MCP tool is then a consumer of this
module, not an author.

THE PIPELINE, five steps, each in its own function because Plan 02 §14 named this file the
natural candidate for a radon D:

    validate -> score (two planes) -> verify fingerprints -> group by item -> hydrate

**Validation refuses, it does not guess.** An empty query is an error, not an empty result:
an empty result set makes a claim about the corpus when the truth is that nothing was asked
(spec §9.3). An unknown topic is an error LISTING the valid slugs, because spec §3.7.5 says
topics and filters come from the store and are never invented from the query text — and
answering with zero results would be exactly that invention, silently.

**Two planes, not fused.** `chunks_fts` produces citable matches; `profiles_fts` produces
item candidates (spec §5.1). Their bm25 scores are computed over different corpora and have
no common scale, so ranking them in one sorted list would invent one. Chunk-matched items
come first — they carry evidence — and profile-only candidates follow. Fusion is Plan 03's
decision, with RRF and the golden set in front of it.

**A topic match expands to the topic's items.** `SearchResult` is item-shaped: it requires an
`item_id`, a `url`, an `author` and a `created_at`, none of which a `topic_note` has. So a
match on a topic surface is attached to that topic's SUPPORTING ITEMS, primary first — which
is what Plan 02 §4 asks for (*el resultado incluye los `supporting_item_ids` del topic, para
que el consumidor pueda saltar a items reales con `get`*) and is stronger than a list of ids,
because the items arrive hydrated.

**Verification comes from the LIVE STORE (M5).** Never from a column: `surface_fingerprint`
does not depend on the verdict, so a stored copy could not be invalidated when the verdict
changed and a `FAIL` revoked by `verify --audit` would keep being served as the old `PASS`.
The freshness check is `verification.verdict_is_current`, the same one
`generate._verdict_badge` applies.

**`verify_with` empty means `no_underlying_source`, and that is the whole statement.** Spec
§3.5: *si una superficie derivada no permite llegar a material sustentante, el resultado debe
decirlo*. A match on a primary surface names that surface; a match on a derived surface names
the item's primary surfaces; and an item that has none gets `()`. `verify_with == ()` is
therefore reachable in exactly one case, which makes the empty tuple the structural form of
the warning. `no_underlying_source()` is the predicate, `render.py` prints the word, and the
gap — that the FROZEN `SearchResult` has no `warnings` field for a JSON consumer to read it
from — is recorded in the execution report rather than closed by amending a contract Plan 01
froze on purpose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from xbrain.knowledge.contracts import (
    SearchFilters,
    SearchMatch,
    SearchResponse,
    SearchResult,
    Strategy,
    resolve_strategy,
)
from xbrain.knowledge.index_store import open_for_query, verify_fingerprints
from xbrain.knowledge.lexical import LexicalHit
from xbrain.knowledge.models import DerivedText, Locator, SurfaceType
from xbrain.knowledge.provenance import DEFAULT_EVIDENCE_CLASSES
from xbrain.knowledge.surfaces import (
    SURFACE_ORIGIN,
    hydrate_verification,
    item_topics,
    knowledge_item,
)
from xbrain.knowledge.provenance import ORIGIN_TRUST
from xbrain.models import Item, Topic, TopicPage

# How deep to go into the chunk plane before grouping. Grouping collapses many chunks into
# one item, so retrieving exactly `limit` chunks would return far fewer than `limit` items
# whenever one item matches well — the failure mode grouping exists to prevent, reappearing
# as a short answer. Four matches' worth of headroom per requested item is enough on the real
# corpus and is bounded so a `--limit 100` cannot ask for 1,200 rows.
CANDIDATE_MULTIPLIER = 4
MAX_CANDIDATES = 500

# WHICH SURFACE TO VERIFY A DERIVED CLAIM AGAINST, most decisive first.
#
# A summary is best checked against the source it summarised, so the content sources lead and
# the post trails: the post is what the item IS, but for a claim the summary made about a
# linked article, the article settles it and the tweet does not. One ordered list, in one
# place, rather than an ordering that emerges from whatever order the emitter happened to
# walk `content.sources` in.
VERIFY_PREFERENCE: tuple[SurfaceType, ...] = (
    "external_article",
    "x_article",
    "thread",
    "quoted_post",
    "video_transcript",
    "video_frame",
    "image_description",
    "post",
    "user_note",
)


@dataclass(frozen=True)
class QueryContext:
    """Everything a query needs that is not the query. Built once by the adapter.

    The STORE is in here, and it is not an optimisation detail: spec §3.7.7 makes `get` read
    the live store, and M5 makes `search` hydrate verification from it. A service that read
    metadata from the index and verdicts from the store would have two answers to *who wrote
    this*, which is the divergence rule 5 is about.
    """

    store: Mapping[str, Item]
    vocab: Sequence[Topic] = ()
    topic_pages: Mapping[str, TopicPage] = field(default_factory=dict)
    index_dir: Path = Path("data/index")
    items_path: Path = Path("data/items.json")
    vault_dir: Path | None = None
    language: str = "English"
    max_matches_per_item: int = 3


def search(
    query: str,
    context: QueryContext,
    *,
    filters: SearchFilters | None = None,
    limit: int = 10,
    strategy: Strategy = "lexical",
) -> SearchResponse:
    """Run one query end to end and return the frozen envelope (spec §7.2).

    Read-only throughout: the index is opened `mode=ro` and the store is never written. No
    network, no model call — spec §13.12 requires that `search` work without a single LLM.

    `strategy` is what the CALLER asked for; `SearchResponse.strategy` is what RAN. They
    differ whenever the requested retriever has no backend, and the difference is declared as
    a `<requested>_not_implemented` degradation rather than hidden — spec §9.3: *lexical
    sigue operativo y el response declara estrategia degradada; no finge resultados
    vectoriales.* Echoing the request back was that pretence in the one field that names the
    retriever (F-2).
    """
    filters = filters or SearchFilters()
    _validate(query, filters, limit, context)
    executed, strategy_degradation = resolve_strategy(strategy)
    index = open_for_query(context.index_dir, context.items_path)
    try:
        depth = min(limit * CANDIDATE_MULTIPLIER * context.max_matches_per_item, MAX_CANDIDATES)
        hits, excluded = verify_fingerprints(index.lexical.search(query, depth, filters=filters))
        profile_ids = [
            hit.item_id for hit in index.lexical.search_profiles(query, limit, filters=filters)
        ]
        grouped = _group_by_item(hits, context, limit=limit)
        _append_profile_candidates(grouped, profile_ids, context, limit=limit)
        results = tuple(
            _hydrate(rank, item_id, matches, context)
            for rank, (item_id, matches) in enumerate(list(grouped.items())[:limit], start=1)
        )
        return SearchResponse(
            query=query,
            strategy=executed,
            filters=filters,
            index=index.status_ref(
                corrupt_chunks_excluded=excluded, strategy_degradation=strategy_degradation
            ),
            results=results,
        )
    finally:
        index.close()


def no_underlying_source(result: SearchResult) -> bool:
    """True when a DERIVED match on this item leads to no recoverable source (spec §3.5).

    The predicate behind the `no_underlying_source` warning. `verify_with` is empty in
    exactly this case — a primary match names its own surface and a derived match names the
    item's primary surfaces — so the empty tuple is the structural statement and this is its
    name.
    """
    return bool(result.matches) and not result.verify_with


# ---------------------------------------------------------------------------
# 1 — validate
# ---------------------------------------------------------------------------


def _validate(query: str, filters: SearchFilters, limit: int, context: QueryContext) -> None:
    """Refuse what cannot be answered honestly, naming what would have been valid.

    An unknown topic is the interesting one. Answering it with zero results would be a claim
    about the corpus produced by a typo, and it would be indistinguishable from a topic that
    genuinely has no matches — so spec §3.7.5's *los topics no se inventan desde el texto del
    query* is enforced by listing the vocabulary instead.
    """
    if not query.strip():
        raise ValueError("La consulta está vacía. Escribe algo que buscar.")
    if limit <= 0:
        raise ValueError(f"--limit debe ser >= 1, recibido {limit}.")
    if filters.topics and context.vocab:
        known = {topic.slug for topic in context.vocab}
        unknown = [slug for slug in filters.topics if slug not in known]
        if unknown:
            raise ValueError(
                f"Topic(s) desconocido(s): {', '.join(unknown)}. "
                f"Los válidos son: {', '.join(sorted(known))}."
            )


# ---------------------------------------------------------------------------
# 4 — group by item (spec §5.4)
# ---------------------------------------------------------------------------


def _group_by_item(
    hits: Sequence[LexicalHit], context: QueryContext, *, limit: int
) -> dict[str, list[LexicalHit]]:
    """Collapse chunk hits into item buckets, capped at `max_matches_per_item`.

    Spec §5.4: grouping is what stops a long transcript filling the top ten with ten adjacent
    windows of itself. The cap is per item and the hits arrive already ranked, so the ones
    kept are the best ones; the item's own rank is its best chunk's, and the rest order
    inside it.

    A TOPIC-owned hit has no item to be grouped under, so it expands to the topic's
    supporting items (see `_topic_owners`).
    """
    grouped: dict[str, list[LexicalHit]] = {}
    for hit in hits:
        for item_id in _owners_of(hit, context, limit=limit):
            bucket = grouped.setdefault(item_id, [])
            if len(bucket) < context.max_matches_per_item:
                bucket.append(hit)
    return grouped


def _owners_of(hit: LexicalHit, context: QueryContext, *, limit: int) -> list[str]:
    """The item ids this hit belongs to. One for an item chunk, several for a topic chunk."""
    if hit.owner_type == "item":
        return [hit.owner_id] if hit.owner_id in context.store else []
    return _topic_owners(hit.owner_id, context, limit=limit)


def _topic_owners(slug: str, context: QueryContext, *, limit: int) -> list[str]:
    """The items a topic surface points at, PRIMARY first, capped at the response limit.

    Capped because a topic with 173 primary items would otherwise flood the answer with one
    note's worth of evidence. Primary first because a primary assignment is the stronger
    claim that the item is about the topic, and sorted within each group so two runs over the
    same store agree (spec §3.7.8).
    """
    primary = sorted(
        item_id
        for item_id, item in context.store.items()
        if item.enriched and item.enriched.primary_topic == slug
    )
    secondary = sorted(
        item_id
        for item_id, item in context.store.items()
        if item.enriched and slug in item.enriched.topics and item_id not in primary
    )
    return (primary + secondary)[:limit]


def _append_profile_candidates(
    grouped: dict[str, list[LexicalHit]],
    profile_ids: Sequence[str],
    context: QueryContext,
    *,
    limit: int,
) -> None:
    """Add profile-only items AFTER the chunk-matched ones, never interleaved.

    The two planes' bm25 scores are computed over different corpora, so there is no scale on
    which to compare them; appending is a declared ordering rather than an invented one.
    Chunk matches come first because they carry a citable excerpt, which is what a consumer
    can actually verify — a profile match says *this item is about that*, and the profile
    itself may never be quoted (spec §5.1.A).
    """
    for item_id in profile_ids:
        if len(grouped) >= limit:
            return
        if item_id not in grouped and item_id in context.store:
            grouped[item_id] = []


# ---------------------------------------------------------------------------
# 5 — hydrate (spec §5.4)
# ---------------------------------------------------------------------------


def _hydrate(
    rank: int, item_id: str, hits: Sequence[LexicalHit], context: QueryContext
) -> SearchResult:
    """One item's result: metadata, labelled context, matches, and what to ask `get` for.

    `summary` and `topics` travel as CONTEXT, and the summary carries its `origin` in the
    same object (`DerivedText`), which is invariant 2 of spec §3.7 made structural: the text
    cannot be read without the provenance that qualifies it.
    """
    item = context.store[item_id]
    projection = knowledge_item(item, vault_dir=context.vault_dir)
    verdicts = hydrate_verification(item, context.language)
    summary = None
    if item.enriched is not None and item.enriched.summary:
        summary = DerivedText(
            text=item.enriched.summary,
            origin=SURFACE_ORIGIN["summary"],
            verification_status=(verdicts["summary"].verdict if "summary" in verdicts else None),
        )
    matches = tuple(_match(position, hit) for position, hit in enumerate(hits, start=1))
    return SearchResult(
        rank=rank,
        item_id=item.id,
        url=item.url,
        author=item.author,
        created_at=item.created_at,
        summary=summary,
        topics=item_topics(item),
        matches=matches,
        available_surfaces=projection.available_surfaces,
        verify_with=_verify_with(matches, projection.available_surfaces),
    )


def _match(position: int, hit: LexicalHit) -> SearchMatch:
    """One chunk, with WHY it matched and where it came from (spec §5.3).

    `score` is bm25's, and bm25 is negative-is-better; it is carried unchanged and documented
    as a RANKING SIGNAL rather than a probability, which spec §5.3 requires — a fused rank
    has no calibrated scale, and neither does an unfused one.

    The locator is rebuilt from the chunk's own columns rather than read back from
    `surfaces.locator_json`: what a consumer needs to check this MATCH is the character range
    inside the surface, and the surface's own locator answers a different question.
    """
    return SearchMatch(
        chunk_id=hit.chunk_id,
        surface_type=hit.surface_type,
        origin=hit.origin,  # type: ignore[arg-type]
        trust_class=hit.trust_class,  # type: ignore[arg-type]
        derived=hit.derived,
        excerpt=hit.excerpt,
        matched_by=("lexical",),
        lexical_rank=position,
        score=hit.score,
        locator=Locator(
            kind="content_source" if hit.owner_type == "item" else "topic_page",
            url=hit.url,
            char_start=hit.char_start,
            char_end=hit.char_end,
        ),
    )


def _verify_with(
    matches: Sequence[SearchMatch], available: Sequence[SurfaceType]
) -> tuple[SurfaceType, ...]:
    """What to ask `get` for in order to check these matches (spec §5.4, §3.5).

    A PRIMARY match names its own surface: you verify a quoted post by fetching the quoted
    post. A DERIVED match — summary, digest, topic overview or note — names the item's
    primary surfaces instead, because spec §3.5 says a derived result is a discovery signal
    and the underlying source is what settles the claim.

    An item with no primary surface at all yields `()`, and that is the `no_underlying_source`
    state: it is reachable in exactly this one case, which is what makes the empty tuple an
    unambiguous statement rather than an absence.
    """
    primary_matched = [
        match.surface_type
        for match in matches
        if ORIGIN_TRUST[SURFACE_ORIGIN[match.surface_type]] in DEFAULT_EVIDENCE_CLASSES
    ]
    if primary_matched:
        return tuple(dict.fromkeys(primary_matched))
    candidates = {
        surface
        for surface in available
        if ORIGIN_TRUST[SURFACE_ORIGIN[surface]] in DEFAULT_EVIDENCE_CLASSES
    }
    return tuple(surface for surface in VERIFY_PREFERENCE if surface in candidates)


def supporting_item_ids(slug: str, context: QueryContext, *, limit: int = 10) -> tuple[str, ...]:
    """The items a topic surface is supported by — public, for `render` and for Plan 04.

    Exposed rather than left private because Plan 02 §4 requires the consumer to be able to
    jump from a topic match to real items, and the CLI's human output has to name them.
    """
    return tuple(_topic_owners(slug, context, limit=limit))
