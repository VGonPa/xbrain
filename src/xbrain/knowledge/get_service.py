"""`get` as a SERVICE (spec §3.7.7, §7.3) — the one that reads the STORE, never the index.

INVARIANT 7 OF SPEC §3.7 IS THE WHOLE POINT OF THIS MODULE: *`get` lee el store actual; el
índice no se convierte en una segunda fuente de verdad.* Its operational definition is a test
that DELETES `data/index/` and calls `get`, which still works — because everything here comes
from `load_store` and the emitter. An index that could answer `get` would be a copy of the
corpus that nothing invalidates, and the day the two disagreed there would be no way to know
which one a reader had been shown.

TRUNCATION NEVER CUTS A SURFACE'S TEXT, and this is not a stylistic choice. A
`KnowledgeSurface` carries a `fingerprint` over its own body, and `KnowledgeChunk` guarantees
`surface.text[char_start:char_end] == chunk.text`. Shortening a surface's `text` to fit a
budget would leave a fingerprint that no longer describes its own field — the verbatim claim
of spec §3.8, broken by the pagination. So the bundle has TWO fields and they mean two
different things:

* `surfaces` — delivered WHOLE. Text verbatim, fingerprint valid, nothing removed.
* `chunks` — the fragments delivered when a surface did not fit, or when a `query` asked for
  the parts of it that score. Each carries its own offsets and its own fingerprint, so a
  reader can still slice the stored surface and get back exactly what they were shown.

`truncated` plus a `cursor` says the rest exists (spec §9.3: never a silent cut). "Complete"
means every surface is reachable by selection and pagination, not three million tokens in one
call (spec §7.3).

THE `query` PATH USES THE SAME SCORER AS `search`, on an in-memory index built from this
item's own chunks. Not a second ranking function — CLAUDE.md rule 5 — and not the persisted
index either, because `get` must work with `data/index/` deleted. `lexical_fts` makes the two
identical by construction: same tokenizer, same columns, same `bm25()`, same tie-break.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial

from xbrain.knowledge.chunking import chunk_surfaces
from xbrain.knowledge.contracts import EvidenceBundle
from xbrain.knowledge.index_schema import open_memory_index
from xbrain.knowledge.lexical import LexicalIndex
from xbrain.knowledge.models import KnowledgeChunk, KnowledgeSurface, SurfaceType
from xbrain.knowledge.search_service import QueryContext
from xbrain.knowledge.surfaces import (
    article_block_texts,
    hydrate_verification,
    item_surfaces,
    item_topics,
    knowledge_item,
    topic_record,
)

# What `get` returns when the caller names no surface. Spec §7.3 / Plan 02 §5: metadata,
# topics, summary and the list of what is available — WITHOUT dumping the long bodies. The
# summary is the one body included because it is the item's own index card; every other
# surface is asked for by name, and `KnowledgeItem.available_surfaces` says which names work.
DEFAULT_SURFACES: tuple[SurfaceType, ...] = ("summary",)

# The default per-response ceiling (Plan 02 §5). 40,000 characters is roughly a long article
# plus its metadata: enough that the common case never paginates, small enough that a bundle
# cannot become the whole corpus in one call.
DEFAULT_CHAR_BUDGET = 40_000


@dataclass(frozen=True)
class GetLimits:
    """The response ceiling, and nothing else.

    A dataclass rather than a bare int because spec §7.3 names `limits` as a parameter and
    Plan 03/04 will add to it (a vector budget, an MCP payload cap); growing a dataclass is
    additive, widening an int is a signature change in every adapter.
    """

    char_budget: int = DEFAULT_CHAR_BUDGET


class UnknownSurfaceError(ValueError):
    """A surface was asked for that this item does not have and did not fail to fetch.

    Distinguished from a FAILED fetch on purpose: a failure is answered with the failure
    (spec §4), because "we tried and the server said 404" is information. "This item has no
    transcript" answered with an empty bundle would look like a transcript full of nothing.
    """


def get(
    item_id: str,
    context: QueryContext,
    *,
    surfaces: Sequence[SurfaceType] | None = None,
    query: str | None = None,
    limits: GetLimits | None = None,
    cursor: str | None = None,
) -> EvidenceBundle:
    """One item's evidence, read from the LIVE store (spec §7.3).

    Never opens `data/index/`. That is invariant 7 of spec §3.7, and the test that proves it
    deletes the index directory first.
    """
    limits = limits or GetLimits()
    item = context.store.get(item_id)
    if item is None:
        raise ValueError(
            f"No existe el item {item_id!r} en el store. "
            "Comprueba el id con `xbrain search` o `xbrain knowledge inspect <id>`."
        )
    # The SAME producers the build stamped (A-4): without them a transcript the index holds
    # with `producer="parakeet-mlx"` came back from `get` with `producer=None`.
    emitted = item_surfaces(
        item,
        transcribe_command=context.transcribe_command,
        vision_command=context.vision_command,
    )
    projection = knowledge_item(item, vault_dir=context.vault_dir)
    wanted = _select(surfaces, emitted, projection.available_surfaces, projection.failed_sources)

    if query:
        delivered_surfaces: tuple[KnowledgeSurface, ...] = ()
        chunks, truncated, next_cursor = _ranked_chunks(item, wanted, query, limits, cursor)
    else:
        delivered_surfaces, chunks, truncated, next_cursor = _paginate(item, wanted, limits, cursor)

    return EvidenceBundle(
        item=projection,
        topics=_topics(item, context),
        surfaces=delivered_surfaces,
        chunks=chunks,
        failures=projection.failed_sources,
        unfetched_links=projection.unfetched_links,
        verification=hydrate_verification(item, context.language),
        truncated=truncated,
        cursor=next_cursor,
    )


def _select(
    requested: Sequence[SurfaceType] | None,
    emitted: Sequence[KnowledgeSurface],
    available: Sequence[SurfaceType],
    failures: Sequence[object],
) -> tuple[KnowledgeSurface, ...]:
    """The surfaces to deliver, in emitter order, refusing a name the item cannot honour.

    A requested surface the item does not have is an ERROR listing what it does have — unless
    a fetch for it FAILED, in which case the bundle answers with the failure. Returning an
    empty bundle for both would make "we never had it" and "the server returned 404"
    indistinguishable, which is exactly the collapse `failed_sources` and `unfetched_links`
    exist to prevent (m7).
    """
    names = tuple(requested) if requested is not None else DEFAULT_SURFACES
    chosen = tuple(surface for surface in emitted if surface.surface_type in names)
    if requested is not None and not chosen and not failures:
        raise UnknownSurfaceError(
            f"Este item no tiene {', '.join(names)}. "
            f"Superficies disponibles: {', '.join(available) or '—'}."
        )
    return chosen


def _topics(item, context: QueryContext) -> tuple:
    """The item's topics as full `TopicRecord`s, in the item's own topic order.

    Records rather than slugs because spec §3.6 requires each topic layer to carry its own
    provenance: the description's producer is unrecorded (`unknown`, hence synthesis by the
    fail-closed rule) while the overview and notes are known LLM output, and a bare slug
    strips exactly that.
    """
    by_slug = {topic.slug: topic for topic in context.vocab}
    records = []
    for slug in item_topics(item):
        topic = by_slug.get(slug)
        if topic is None:
            continue
        primary, secondary = _membership(context.store, slug)
        records.append(topic_record(topic, context.topic_pages.get(slug), primary, secondary))
    return tuple(records)


def _membership(store: Mapping, slug: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    primary = tuple(
        sorted(i.id for i in store.values() if i.enriched and i.enriched.primary_topic == slug)
    )
    secondary = tuple(
        sorted(
            i.id
            for i in store.values()
            if i.enriched and slug in i.enriched.topics and i.id not in primary
        )
    )
    return primary, secondary


def _paginate(
    item,
    wanted: Sequence[KnowledgeSurface],
    limits: GetLimits,
    cursor: str | None,
) -> tuple[tuple[KnowledgeSurface, ...], tuple[KnowledgeChunk, ...], bool, str | None]:
    """Deliver whole surfaces while they fit; page the first one that does not.

    A surface is either delivered COMPLETE or not at all, because its `fingerprint` covers its
    own body and a shortened `text` would carry a fingerprint that no longer describes it.
    When one does not fit, its CHUNKS are delivered instead — they are the verbatim unit, each
    with its own offsets and fingerprint — up to the budget, and the cursor says where to
    resume.

    A surface LARGER than the entire budget is still delivered, chunk by chunk across
    successive calls: the alternative is a body that can never be read at all, which is the
    silent cut spec §9.3 forbids wearing a different hat.
    """
    start_surface, start_chunk = _decode(cursor)
    page = _Page(budget=limits.char_budget)

    for position, surface in enumerate(wanted):
        if position < start_surface:
            continue
        resume = start_chunk if position == start_surface else 0
        if resume == 0 and len(surface.text) <= page.budget:
            page.take_surface(surface)
            continue
        stopped = page.take_chunks(_chunks_of(item, surface), resume, partial(_encode, position))
        if stopped is not None:
            return page.surfaces_out(), page.chunks_out(), True, stopped
        if position + 1 < len(wanted):
            return page.surfaces_out(), page.chunks_out(), True, _encode(position + 1, 0)
    return page.surfaces_out(), page.chunks_out(), False, None


@dataclass
class _Page:
    """One response's worth of budget, and what has been spent on it so far.

    Mutable because it IS the accumulator; extracted from `_paginate` so the loop reads as
    the decision it makes ("whole, or in pieces, or stop") rather than as bookkeeping.
    """

    budget: int
    surfaces: list[KnowledgeSurface] = field(default_factory=list)
    chunks: list[KnowledgeChunk] = field(default_factory=list)

    def take_surface(self, surface: KnowledgeSurface) -> None:
        self.surfaces.append(surface)
        self.budget -= len(surface.text)

    def take_chunks(
        self, pieces: Sequence[KnowledgeChunk], resume: int, encode: Callable[[int], str]
    ) -> str | None:
        """Fill from `pieces`, returning the cursor if the budget ran out. None if it did not.

        `encode` turns the offset to resume from into the caller's cursor — positional for
        `_paginate`, `q:<offset>` for the ranked path — so ONE budget loop serves both and
        the query path cannot again end up with a truncation that names no continuation
        (A-2).

        The FIRST chunk of a page is always taken, however long it is: refusing it would
        return an empty page with a cursor pointing at the same place, and a consumer
        following that cursor would loop forever while looking like it was paginating.
        """
        for offset, chunk in enumerate(pieces[resume:], start=resume):
            if self.chunks and len(chunk.text) > self.budget:
                return encode(offset)
            self.chunks.append(chunk)
            self.budget -= len(chunk.text)
            if self.budget <= 0 and offset + 1 < len(pieces):
                return encode(offset + 1)
        return None

    def surfaces_out(self) -> tuple[KnowledgeSurface, ...]:
        return tuple(self.surfaces)

    def chunks_out(self) -> tuple[KnowledgeChunk, ...]:
        return tuple(self.chunks)


def _ranked_chunks(
    item, wanted: Sequence[KnowledgeSurface], query: str, limits: GetLimits, cursor: str | None
) -> tuple[tuple[KnowledgeChunk, ...], bool, str | None]:
    """The chunks of `wanted` that score for `query`, best first (spec §7.3), PAGED.

    THE SAME SCORER AS `search`, on an in-memory database built from this item's own chunks.
    Not a second ranking function (rule 5), and not the persisted index either — `get` must
    keep working with `data/index/` deleted, which is the test that defines invariant 7.
    `lexical_fts` makes the two identical by construction.

    Chunks rather than whole surfaces because prioritising INSIDE a long source is what the
    query is for; the surface it came from is still named on every chunk, so nothing is lost.

    THE CURSOR IS AN OFFSET INTO THE RANKING (A-2). The first version returned
    `(kept, True, None)` when the budget ran out and `get` ignored `cursor` whenever a
    `query` was given, so the query path declared the truncation and offered no way to
    continue — the human view printed `--cursor None`. The ranking is deterministic (same
    scorer, same tie-break, same chunks), so an offset resumes exactly where the previous
    page stopped; the pages are disjoint and reassemble the whole list.
    """
    by_id: dict[str, KnowledgeChunk] = {}
    for surface in wanted:
        for chunk in _chunks_of(item, surface):
            by_id[chunk.chunk_id] = chunk
    index = LexicalIndex(open_memory_index())
    try:
        index.add(list(by_id.values()))
        hits = index.search(query, max(len(by_id), 1))
    finally:
        # A scratch database per call, CLOSED per call (M-1): a long-lived adapter would
        # otherwise hold one handle per `get --query` it ever answered.
        index.connection.close()
    ordered = [by_id[hit.chunk_id] for hit in hits if hit.chunk_id in by_id]

    page = _Page(budget=limits.char_budget)
    stopped = page.take_chunks(ordered, _decode_query(cursor), _encode_query)
    return page.chunks_out(), stopped is not None, stopped


def _chunks_of(item, surface: KnowledgeSurface) -> tuple[KnowledgeChunk, ...]:
    """This surface's chunks, cut exactly as the index cuts them.

    Through `chunk_surfaces` with the article block map, so an X Article pages on the
    boundaries its author set rather than on the paragraph fallback — the same seam
    `index_build` uses, for the same reason.
    """
    return chunk_surfaces(
        (surface,),
        topics=item_topics(item),
        url=item.url,
        blocks_by_surface_id=article_block_texts(item),
    )


# The two cursor shapes. A positional cursor is `<surface>:<chunk>`; a ranked one is
# `q:<offset>`. They index DIFFERENT sequences — emitter order against a ranking that only
# exists for one query — so each decoder refuses the other's shape by name: a positional
# cursor applied to a ranking would resume at an unrelated place while looking like progress.
_QUERY_CURSOR_PREFIX = "q"


def _encode(surface_index: int, chunk_index: int) -> str:
    """The positional cursor: where the next call resumes. Opaque to the caller."""
    return f"{surface_index}:{chunk_index}"


def _encode_query(offset: int) -> str:
    """The ranked cursor: the offset into the query's ranking to resume from (A-2)."""
    return f"{_QUERY_CURSOR_PREFIX}:{offset}"


def _decode(cursor: str | None) -> tuple[int, int]:
    """Parse a positional cursor, refusing a malformed one rather than silently restarting.

    Restarting from the beginning on a bad cursor would loop a paginating consumer forever
    while looking like it was making progress.
    """
    if not cursor:
        return 0, 0
    head, _, tail = cursor.partition(":")
    if head == _QUERY_CURSOR_PREFIX:
        raise ValueError(
            f"Cursor inválido: {cursor!r} es un cursor de --query. "
            "Pásalo junto a la misma --query que lo devolvió."
        )
    return _component(head, cursor), _component(tail, cursor)


def _decode_query(cursor: str | None) -> int:
    """Parse a ranked cursor (A-2), refusing a positional one by name."""
    if not cursor:
        return 0
    head, _, tail = cursor.partition(":")
    if head != _QUERY_CURSOR_PREFIX:
        raise ValueError(
            f"Cursor inválido: {cursor!r} es un cursor de posición, no de --query. "
            "Usa el que devolvió la respuesta anterior a esta misma --query."
        )
    return _component(tail, cursor)


def _component(raw: str, cursor: str) -> int:
    """One non-negative integer of a cursor. A negative one is a restart in disguise."""
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            f"Cursor inválido: {cursor!r}. Usa el que devolvió la respuesta anterior."
        ) from error
    if value < 0:
        raise ValueError(f"Cursor inválido: {cursor!r}. Usa el que devolvió la respuesta anterior.")
    return value
