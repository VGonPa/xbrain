# tests/test_knowledge_get_service.py
"""`get` (Plan 02 §5, steps 21–25b).

THE DEFINING TEST OF THIS FILE IS THE ONE THAT DELETES `data/index/`. Invariant 7 of spec
§3.7 — *`get` lee el store actual; el índice no se convierte en una segunda fuente de
verdad* — has no meaning as prose; it has meaning as a test that removes the index and calls
`get` anyway. An index that could answer `get` would be a copy of the corpus that nothing
invalidates, and the day the two disagreed nobody could say which one a reader had seen.

THE SECOND THING THIS FILE PINS IS THAT TRUNCATION NEVER CUTS A SURFACE'S TEXT. A
`KnowledgeSurface` carries a fingerprint over its own body; shortening the body to fit a
budget would leave a fingerprint that no longer describes its own field, so the verbatim
claim of spec §3.8 would be broken by the pagination itself. Whole surfaces go in `surfaces`;
the fragments of one that did not fit go in `chunks`, each with its own offsets.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from xbrain.knowledge import index_build
from xbrain.knowledge.chunking import ChunkerParams, chunk_surfaces
from xbrain.knowledge.contracts import EvidenceBundle
from xbrain.knowledge.get_service import (
    DEFAULT_CHAR_BUDGET,
    DEFAULT_SURFACES,
    GetLimits,
    UnknownSurfaceError,
    get,
)
from xbrain.knowledge.index_schema import db_path, open_index
from xbrain.knowledge.models import KnowledgeSurface
from xbrain.knowledge.search_service import QueryContext
from xbrain.knowledge.surfaces import article_block_texts, item_surfaces, item_topics
from xbrain.models import Item, Topic, TopicPage
from xbrain.rubrics import save_vocab
from xbrain.store import save_store, save_topic_pages

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def corpus() -> tuple[dict[str, Item], list[Topic], dict[str, TopicPage]]:
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    return (
        {k: Item.model_validate(v) for k, v in raw["items"].items()},
        [Topic.model_validate(v) for v in raw["vocab"].values()],
        {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()},
    )


def _paths(data: Path) -> tuple[Path, Path, Path]:
    """The three inputs, in the order every door of this package takes them."""
    return data / "items.json", data / "vocab.yaml", data / "topics.json"


def _persist(data: Path, store, vocab, pages) -> None:
    """All THREE inputs on disk, through the store's own writers.

    The same helper `tests/test_knowledge_search_service.py` uses, for the same reason: the
    cheap signal covers `vocab.yaml` and `topics.json` as well as `items.json` (P1a), so a
    fixture that wrote only the first would exercise a store no door of this package accepts.
    """
    save_store(store, data / "items.json")
    save_vocab(vocab, data / "vocab.yaml")
    save_topic_pages(pages, data / "topics.json")


def _context(data: Path, store, vocab, pages, **overrides) -> QueryContext:
    return QueryContext(
        store=store,
        vocab=vocab,
        topic_pages=pages,
        index_dir=data / "index",
        items_path=data / "items.json",
        vocab_path=data / "vocab.yaml",
        topics_path=data / "topics.json",
        **overrides,
    )


@pytest.fixture()
def context(tmp_path: Path, corpus) -> QueryContext:
    """The live store, and an index built beside it that `get` must never read.

    The index is built — not omitted — on purpose: a fixture with no index would make every
    test in this file pass invariant 7 by accident, and the one test that DELETES the
    directory would delete nothing (rule 2). Here the index exists, is valid, and answers
    `search`; `get` still has to come back from the store alone.
    """
    store, vocab, pages = corpus
    data = tmp_path / "data"
    _persist(data, store, vocab, pages)
    index_build.build(data / "index", index_build.load_index_inputs(*_paths(data)))
    return _context(data, store, vocab, pages)


@pytest.fixture()
def long_article_context(tmp_path: Path, corpus) -> QueryContext:
    """The S2 case: a real 20k-character article, as a COMMITTED fixture (m-iv).

    The number asserted downstream is a property of
    `tests/fixtures/knowledge_long_article.json`, not of the corpus — which is why it can be
    asserted at all. A corpus figure in an assert would be a test that goes red when someone
    runs `enrich` (M6).

    NO INDEX IS BUILT HERE, and `index_dir` points at a directory that does not exist. Every
    pagination test below therefore runs with `data/index/` absent, which is invariant 7
    holding for the whole S2 block rather than for one test of it.
    """
    item = Item.model_validate(
        json.loads((FIXTURES / "knowledge_long_article.json").read_text(encoding="utf-8"))
    )
    _store, vocab, pages = corpus
    store = {item.id: item}
    data = tmp_path / "data"
    _persist(data, store, vocab, pages)
    assert not (data / "index").exists()
    return _context(data, store, vocab, pages)


# ---------------------------------------------------------------------------
# 21 — the index is not a second source of truth
# ---------------------------------------------------------------------------


def test_get_works_with_the_index_directory_deleted(context: QueryContext) -> None:
    """Step 21 / acceptance 9: the OPERATIONAL definition of invariant 7 (spec §3.7).

    Not "we do not import index_store" — that is a claim about the code. The index directory
    is REMOVED and `get` answers anyway. Seen red by reading any field from the index: the
    call raises `IndexMissingError` instead of returning a bundle.
    """
    shutil.rmtree(context.index_dir)
    assert not context.index_dir.exists()

    bundle = get("k03", context, surfaces=("external_article",))

    assert isinstance(bundle, EvidenceBundle)
    assert bundle.item.item_id == "k03"
    assert bundle.surfaces and bundle.surfaces[0].surface_type == "external_article"


def test_get_with_a_query_also_works_without_the_index(context: QueryContext) -> None:
    """The `query` path is the one that could most easily reach for the persisted index.

    It ranks with the SAME scorer — `lexical_fts` — on an in-memory database built from this
    item's own chunks, so it is one ranking function used twice rather than two that agree by
    coincidence (rule 5). Seen red by pointing it at `data/index/knowledge.db`.
    """
    shutil.rmtree(context.index_dir)
    bundle = get("k03", context, surfaces=("external_article",), query="Quillfeather")
    assert bundle.chunks
    assert "Quillfeather" in bundle.chunks[0].text


# ---------------------------------------------------------------------------
# 22 — the default is metadata, not a dump
# ---------------------------------------------------------------------------


def test_get_without_surfaces_does_not_dump_the_long_bodies(context: QueryContext) -> None:
    """Step 22 / Plan 02 §5: metadata, topics, summary and the LIST of what is available.

    Asserted by what is ABSENT — no article, no transcript — rather than by a length
    threshold, which would pass or fail on how long the fixture happens to be.
    """
    bundle = get("k03", context)
    assert {s.surface_type for s in bundle.surfaces} <= set(DEFAULT_SURFACES)
    assert "external_article" in bundle.item.available_surfaces, (
        "the body is withheld, but its NAME must be there or the caller cannot ask for it"
    )
    assert bundle.chunks == ()


def test_get_carries_the_topics_with_their_own_provenance(context: QueryContext) -> None:
    """Spec §3.6: each topic layer keeps its own provenance.

    The description is `unknown` (the vocabulary does not record whether it was written or
    generated) and therefore synthesis by the fail-closed rule; the overview is known LLM
    output. One `origin` for the whole record would have to lie about one of them.
    """
    bundle = get("k03", context)
    assert bundle.topics
    record = bundle.topics[0]
    assert record.description.origin == "unknown"
    assert record.overview is None or record.overview.origin == "llm"


def test_asking_for_a_surface_the_item_does_not_have_lists_what_it_does(
    context: QueryContext,
) -> None:
    """An empty bundle would make "we never had it" look like "it is empty" (spec §9.3)."""
    with pytest.raises(UnknownSurfaceError, match="Superficies disponibles"):
        get("k01", context, surfaces=("video_transcript",))


def test_getting_an_unknown_item_names_how_to_find_the_id(context: QueryContext) -> None:
    """An actionable error, never a `KeyError` from inside a service."""
    with pytest.raises(ValueError, match="xbrain search"):
        get("nope", context)


# ---------------------------------------------------------------------------
# 23 — the complete body (m-iv: a FIXTURE figure, not a corpus one)
# ---------------------------------------------------------------------------


def test_get_returns_the_whole_article_body_untruncated(
    long_article_context: QueryContext,
) -> None:
    """Step 23 / acceptance 9: the S2 article COMPLETE — 20,147 chars from the fixture.

    `ARTICLE_CHAR_LIMIT` truncates in `evidence.py` because that is a prompt; retrieval cannot
    (plan-00 §3). The number is read from the fixture rather than hard-coded twice, and the
    equality against the stored source text is what makes it a verbatim claim rather than a
    length claim. Seen red by applying any ceiling: the body comes back short.
    """
    item = next(iter(long_article_context.store.values()))
    source = next(s for s in item.content.sources if getattr(s, "kind", None) == "external_article")

    bundle = get(item.id, long_article_context, surfaces=("external_article",))

    (surface,) = bundle.surfaces
    assert surface.text == source.text
    assert len(surface.text) == 20147, (
        "this is a property of tests/fixtures/knowledge_long_article.json, not of the corpus"
    )
    assert bundle.truncated is False and bundle.cursor is None


# ---------------------------------------------------------------------------
# 24 — explicit truncation and a cursor that continues
# ---------------------------------------------------------------------------


def test_a_body_over_the_budget_is_paginated_not_cut(
    long_article_context: QueryContext,
) -> None:
    """Step 24 / spec §9.3: `truncated: true` plus a cursor, never a silent cut.

    The surface does not appear in `surfaces` at all, because a surface is delivered COMPLETE
    or not at all: its fingerprint covers its own body, and a shortened `text` would carry a
    fingerprint that no longer describes it. Its CHUNKS come instead, each with its own
    offsets and its own fingerprint.
    """
    item = next(iter(long_article_context.store.values()))
    bundle = get(
        item.id,
        long_article_context,
        surfaces=("external_article",),
        limits=GetLimits(char_budget=3000),
    )
    assert bundle.truncated is True and bundle.cursor
    assert bundle.surfaces == (), "a partial surface would carry a fingerprint that lies"
    assert bundle.chunks
    assert sum(len(c.text) for c in bundle.chunks) <= 3000 + max(len(c.text) for c in bundle.chunks)


def test_the_cursor_continues_where_the_previous_call_stopped(
    long_article_context: QueryContext,
) -> None:
    """A cursor that does not advance is a loop that looks like progress.

    Asserted by walking the WHOLE body across pages and reassembling it: the concatenation of
    every chunk delivered equals the stored source text, in order, with nothing repeated and
    nothing lost. That is the strongest form of "pagination is complete" — a page count would
    pass while dropping a paragraph.
    """
    item = next(iter(long_article_context.store.values()))
    source = next(s for s in item.content.sources if getattr(s, "kind", None) == "external_article")

    seen: list[str] = []
    cursor: str | None = None
    for _page in range(50):
        bundle = get(
            item.id,
            long_article_context,
            surfaces=("external_article",),
            limits=GetLimits(char_budget=3000),
            cursor=cursor,
        )
        seen += [chunk.text for chunk in bundle.chunks]
        seen += [surface.text for surface in bundle.surfaces]
        if not bundle.truncated:
            break
        assert bundle.cursor != cursor, "the cursor did not advance"
        cursor = bundle.cursor
    else:
        pytest.fail("pagination never terminated")

    assert "".join(seen) == source.text


def test_a_malformed_cursor_is_refused(long_article_context: QueryContext) -> None:
    """Restarting silently on a bad cursor loops a consumer forever while looking fine."""
    item = next(iter(long_article_context.store.values()))
    with pytest.raises(ValueError, match="Cursor"):
        get(item.id, long_article_context, surfaces=("external_article",), cursor="banana")


# ---------------------------------------------------------------------------
# B2 — out-of-range cursors are refused, not silently completed
# ---------------------------------------------------------------------------


def test_an_out_of_range_surface_cursor_is_refused(context: QueryContext) -> None:
    """B2 (gate review): a cursor whose surface index exceeds `len(wanted)` was returning
    `surfaces=(), chunks=(), truncated=False, cursor=None` — the signal for "complete" — when
    the consumer may have missed the whole corpus. Same family as a negative component (a
    restart in disguise) but positional rather than signed. Spec §9.3: "truncamiento explícito
    + cursor, nunca corte silencioso". The silent completion violated the explicit half.

    Reproduced on HEAD before the fix:
        get("k03", ctx, surfaces=("external_article",), cursor="1:0")
        → surfaces=(), chunks=(), truncated=False, cursor=None

    Now raises the same ValueError as a malformed or negative cursor.
    """
    with pytest.raises(ValueError, match="Cursor"):
        # k03 has exactly 1 external_article surface, so index 1 is out of range.
        get("k03", context, surfaces=("external_article",), cursor="1:0")


def test_an_out_of_range_chunk_cursor_on_the_final_surface_is_refused(
    context: QueryContext,
) -> None:
    """B2 (gate review): a cursor whose chunk index exceeds `len(pieces)` on the FINAL surface
    was returning an empty bundle with `truncated=False` — same silent completion as B2's
    surface case. Mid-pagination, a cursor past the current surface's chunks advances to the
    next surface; on the LAST surface it signals completion when chunks remain unvisited.

    Reproduced on HEAD before the fix:
        get("k03", ctx, surfaces=("external_article",), cursor="0:99999")
        → surfaces=(), chunks=(), truncated=False, cursor=None

    Now raises ValueError.
    """
    with pytest.raises(ValueError, match="Cursor"):
        # k03's external_article has far fewer than 99999 chunks at any reasonable budget.
        get(
            "k03",
            context,
            surfaces=("external_article",),
            limits=GetLimits(char_budget=100),  # force chunking
            cursor="0:99999",
        )


def test_an_out_of_range_query_cursor_is_refused(context: QueryContext) -> None:
    """B2 (gate review): a query cursor whose offset exceeds `len(ordered)` was returning
    `chunks=(), truncated=False, cursor=None` — the same silent completion as the positional
    cases. The ranking is deterministic, so an offset past the end is either a corrupted
    cursor or a replay against a smaller result set, and both are invalid.

    Reproduced on HEAD before the fix:
        get("k03", ctx, surfaces=("external_article",), query="Quillfeather", cursor="q:99999")
        → chunks=(), truncated=False, cursor=None

    Now raises ValueError.
    """
    with pytest.raises(ValueError, match="Cursor"):
        get(
            "k03",
            context,
            surfaces=("external_article",),
            query="Quillfeather",
            cursor="q:99999",
        )


# ---------------------------------------------------------------------------
# The query path (spec §7.3)
# ---------------------------------------------------------------------------


def test_a_query_prioritises_the_chunks_that_score(context: QueryContext) -> None:
    """Spec §7.3: *priorizar dentro de una fuente larga usando un query opcional.*

    THE CASE IS CONSTRUCTED, and it has to be. The obvious fixture —
    `knowledge_long_article.json` — cannot test this: its 20,147 characters are the NATO
    alphabet repeated, so every word appears in all 23 chunks and **zero** words are unique to
    one (measured). A query over it returns the same order as no query at all, and the test
    would be green while proving nothing (rule 2). The fixture is right for what it exists to
    pin — that a 20 k body comes back whole — and wrong for this.

    So the body below places a distinctive term in the LAST paragraph, and the assertion is
    against the POSITIONAL order: the ranked first chunk must not be the positional first
    chunk. A passthrough that ignored the query would fail on exactly that line.
    """
    item = context.store["k03"]
    body = "\n\n".join(
        ["Ordinary filler prose about evaluation harnesses. " * 12 for _ in range(6)]
        + ["The distinctive marker here is Zephyrine, and it appears nowhere else."]
    )
    sources = list(item.content.sources)
    sources[0] = sources[0].model_copy(update={"text": body})
    store = dict(context.store)
    store["k03"] = item.model_copy(
        update={"content": item.content.model_copy(update={"sources": sources})}
    )
    local = QueryContext(**{**context.__dict__, "store": store})

    # The chunker's own order, taken directly: a budgeted `get` would only show the first
    # page, and a term chosen from one page cannot discriminate ranked from unranked.
    surface = next(s for s in item_surfaces(store["k03"]) if s.surface_type == "external_article")
    positional = chunk_surfaces((surface,), url=store["k03"].url)
    assert len(positional) > 1, "the body must produce several chunks or nothing is ranked"
    assert "Zephyrine" not in positional[0].text, "the marker must not open the article"

    ranked = get("k03", local, surfaces=("external_article",), query="Zephyrine").chunks

    assert ranked, "the query path must return something"
    assert "Zephyrine" in ranked[0].text
    assert ranked[0].chunk_id != positional[0].chunk_id, (
        "the ranked first chunk equals the positional first chunk: the query changed nothing"
    )


# ---------------------------------------------------------------------------
# 25 / 25b — failures and unfetched links are STATE, not silence
# ---------------------------------------------------------------------------


def test_get_returns_a_failed_fetch_as_structured_state(context: QueryContext) -> None:
    """Step 25 / acceptance 10 / spec §4: a dead link is state, not a silence.

    `k11` has a 404. The failure travels with its `failure_reason`, so a consumer can tell
    "the server refused" from "we never tried" — which is the whole reason `failed_sources`
    and `unfetched_links` are two fields (m7).
    """
    bundle = get("k11", context)
    assert bundle.failures
    assert bundle.failures[0].failure_reason == "not_found"


def test_get_returns_unfetched_links_with_their_reason(context: QueryContext) -> None:
    """Step 25b (m7): a link with no body carries WHY, and carries no text.

    There is deliberately no text field on `UnfetchedLink`: naming the cause never licenses
    describing the content, and the absence of the field is the guardrail.
    """
    bundle = get("k11", context)
    assert bundle.unfetched_links
    link = bundle.unfetched_links[0]
    assert link.reason == "http_error"
    assert not hasattr(link, "text")


def test_asking_for_a_failed_surface_answers_with_the_failure(context: QueryContext) -> None:
    """The branch `_select` exists for: a FAILED fetch is answered, not refused.

    "This item has no article" and "the article returned 404" are different facts, and an
    error for both would collapse them.
    """
    bundle = get("k11", context, surfaces=("external_article",))
    assert bundle.failures
    assert bundle.surfaces == ()


def test_a_surface_the_item_lacks_is_refused_even_when_another_fetch_failed(
    context: QueryContext,
) -> None:
    """M-2 (gate Fable, round 05): `_select` checked `not failures` — ANY failure — while its
    docstring promised the exception «unless a fetch for IT failed». So on every item with
    one failed fetch (62 of 2,404 on the real store, measured 2026-09-02) asking for a
    surface it never had returned an EMPTY bundle with exit 0: `get k11 --surface
    video_transcript` → `surfaces []`, `chunks []`, `failures [external_article/not_found]`,
    and a consumer asking for the transcript concludes the transcript failed. That is
    precisely the collapse of "we never had it" into "it failed" that `failed_sources` and
    `unfetched_links` exist to prevent (m7).

    The requested names are now compared with the surface types the failed KINDS would have
    produced (`CONTENT_KIND_TO_SURFACE_TYPES`), name by name: a name that is neither emitted
    nor failed is refused listing what is available; a name that failed is answered with the
    failure. Both directions on k11, and the mixed request is refused naming ONLY the
    unknown name. Seen red before the fix: `get("k11", surfaces=("video_transcript",))`
    returned a bundle.
    """
    with pytest.raises(UnknownSurfaceError, match="no tiene video_transcript") as caught:
        get("k11", context, surfaces=("video_transcript",))
    assert "Superficies disponibles" in str(caught.value)

    answered = get("k11", context, surfaces=("external_article",))
    assert answered.surfaces == () and answered.chunks == ()
    assert [f.kind for f in answered.failures] == ["external_article"]

    with pytest.raises(UnknownSurfaceError, match="no tiene video_transcript\\.") as caught:
        get("k11", context, surfaces=("external_article", "video_transcript"))
    assert "no tiene external_article" not in str(caught.value), "the failed one is not unknown"


def test_a_request_mixing_a_present_and_an_absent_surface_is_refused_not_trimmed(
    context: QueryContext,
) -> None:
    """The same rule on an item with no failure at all: `("post", "video_transcript")` on k03
    used to return the post and drop the transcript in silence, because the refusal fired
    only when NOTHING was chosen. A partial answer that names no omission is the silent cut
    spec §9.3 forbids. Seen red before the fix: a bundle with one surface came back.
    """
    with pytest.raises(UnknownSurfaceError, match="no tiene video_transcript"):
        get("k03", context, surfaces=("post", "video_transcript"))
    assert get("k03", context, surfaces=("post",)).surfaces[0].surface_type == "post"


# ---------------------------------------------------------------------------
# 28 — read-only
# ---------------------------------------------------------------------------


def test_the_two_copies_of_the_failure_list_agree(context: QueryContext) -> None:
    """`EvidenceBundle.failures` and `KnowledgeItem.failed_sources` are BOTH in the frozen
    contract, and the service fills both from one projection.

    Two fields holding the same fact is the shape that drifts (rule 5). Plan 01 froze them
    and Plan 02 §0 forbids amending the contract here, so the redundancy is pinned by a test
    instead: the day a second writer fills one and forgets the other, this goes red. The same
    holds for `unfetched_links`.
    """
    bundle = get("k11", context)
    assert bundle.failures == bundle.item.failed_sources
    assert bundle.unfetched_links == bundle.item.unfetched_links


def test_get_does_not_touch_items_json(context: QueryContext) -> None:
    """Acceptance 13, hashed before and after."""
    import hashlib

    before = hashlib.sha256(context.items_path.read_bytes()).hexdigest()
    get("k03", context, surfaces=("external_article",))
    assert hashlib.sha256(context.items_path.read_bytes()).hexdigest() == before


def test_the_bundle_validates_against_the_frozen_schema(context: QueryContext) -> None:
    """`extra="forbid"` means a field the service invents fails construction."""
    bundle = get("k03", context, surfaces=("external_article",))
    assert EvidenceBundle.model_validate(bundle.model_dump()) == bundle


# ---------------------------------------------------------------------------
# A-2 — `--query` paginates with a cursor, never a silent cut
# ---------------------------------------------------------------------------


def test_a_query_over_the_budget_paginates_with_a_cursor_that_continues(
    long_article_context: QueryContext,
) -> None:
    """A-2 (round 02, both gates): `_ranked_chunks` returned `(kept, True, None)` when the
    budget ran out, and `get` ignored `cursor` whenever a `query` was given — so the query
    path said `truncated: true` and offered no way to continue, and the human view printed
    literally `--cursor None`. Spec §9.3: *truncamiento explícito + cursor, nunca corte
    silencioso*; the explicit half was there and the continuation was not. Measured on the
    20,147-char fixture article with `query="Alpha"` and a 1,000-char budget:
    `truncated=True, cursor=None, chunks_returned=1`.

    The ranked list is deterministic (same scorer, same in-memory index), so a cursor is an
    offset into it. Followed to the end, the pages are disjoint, each page advances, and
    their union IS the full ranked list a single unbounded call returns.

    Seen red before the fix: `truncated with no cursor`.
    """
    item = next(iter(long_article_context.store.values()))
    limits = GetLimits(char_budget=1000)
    cursor: str | None = None
    seen: list[str] = []
    for _ in range(100):
        bundle = get(
            item.id,
            long_article_context,
            surfaces=("external_article",),
            query="Alpha",
            limits=limits,
            cursor=cursor,
        )
        assert bundle.chunks, "a page with a cursor must deliver something"
        ids = [chunk.chunk_id for chunk in bundle.chunks]
        assert not set(ids) & set(seen), "a page repeated a chunk"
        seen += ids
        if not bundle.truncated:
            assert bundle.cursor is None
            break
        assert bundle.cursor is not None, "truncated with no cursor"
        assert bundle.cursor != cursor, "the cursor did not advance"
        cursor = bundle.cursor
    else:
        pytest.fail("the query path never finished paginating")

    assert len(seen) > 1, "the budget forced more than one page"
    whole = get(
        item.id,
        long_article_context,
        surfaces=("external_article",),
        query="Alpha",
        limits=GetLimits(char_budget=10**9),
    )
    assert seen == [chunk.chunk_id for chunk in whole.chunks], "the pages reassemble the ranking"


def test_a_query_cursor_is_refused_without_its_query_and_vice_versa(
    long_article_context: QueryContext,
) -> None:
    """The two cursor shapes are not interchangeable, and mixing them is an error rather
    than a silent restart — the loop-that-looks-like-progress `_decode` already refuses for
    a malformed positional cursor.
    """
    item = next(iter(long_article_context.store.values()))
    with pytest.raises(ValueError, match="query"):
        get(item.id, long_article_context, surfaces=("external_article",), cursor="q:3")
    with pytest.raises(ValueError, match="query"):
        get(
            item.id,
            long_article_context,
            surfaces=("external_article",),
            query="Alpha",
            cursor="0:3",
        )


# ---------------------------------------------------------------------------
# F7-7 (round 08) — `get` claims no producer the store cannot prove
# ---------------------------------------------------------------------------


def test_get_declares_no_producer_for_a_transcript_the_store_does_not_attribute(
    tmp_path: Path, corpus
) -> None:
    """The consumer-side half of the surfaces test: A-4 (round 02) made `get` serve the
    CONFIGURED transcriber as the transcript's `producer`, and F7-7 (round 07) measured
    what that means — change `[transcribe].command` and `get` serves a different producer
    for the same bytes, fingerprints identical. Gate Codex (round 08) read it against spec
    §3.4 as a provenance claim the store cannot back, and it is: a configured command is
    the transcriber this installation WOULD use, not the one that wrote the text.

    `QueryContext` no longer carries the two commands, so no adapter can reintroduce the
    claim; the transcript and the frame come out with `producer: None`, their origin
    still declared. Seen red on `36f694b`: `QueryContext` had the fields and `get` served
    `producer="review-transcriber"`.
    """
    from dataclasses import fields

    store, vocab, pages = corpus
    data = tmp_path / "data"
    data.mkdir()
    (data / "items.json").write_text(
        json.dumps({k: v.model_dump(mode="json") for k, v in store.items()}), encoding="utf-8"
    )
    assert {f.name for f in fields(QueryContext)}.isdisjoint(
        {"transcribe_command", "vision_command"}
    )
    context = QueryContext(
        store=store,
        vocab=vocab,
        topic_pages=pages,
        index_dir=data / "index",
        items_path=data / "items.json",
    )
    transcript = get("k08", context, surfaces=("video_transcript",)).surfaces[0]
    assert transcript.producer is None and transcript.origin == "asr"
    assert transcript.produced_at is not None
    frame = get("k08", context, surfaces=("video_frame",)).surfaces[0]
    assert frame.producer is None and frame.origin == "vlm"


# ---------------------------------------------------------------------------
# M-1 — the scratch database is closed
# ---------------------------------------------------------------------------


def test_get_with_a_query_closes_its_scratch_database(context: QueryContext) -> None:
    """M-1 (round 02, Codex F-08): `_ranked_chunks` opened `sqlite3(":memory:")` and never
    closed it; Python 3.13 reports the leak as a `ResourceWarning: unclosed database`, which
    the gate printed repeatedly during pytest. One call is harmless; a long-lived adapter
    (Plan 04's MCP server) accumulates handles for every `get --query`.

    Seen red before the fix: one `unclosed database` warning per call.
    """
    import gc
    import warnings

    # Reap what EARLIER tests left behind first, so the measured window holds only this
    # call's connections — in the full suite the first version caught a neighbour's leak.
    gc.collect()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        get("k03", context, surfaces=("external_article",), query="Alpha")
        gc.collect()
    leaks = [
        w
        for w in caught
        if issubclass(w.category, ResourceWarning) and "unclosed database" in str(w.message)
    ]
    assert not leaks, [str(w.message) for w in leaks]


# ---------------------------------------------------------------------------
# B2 (gate Codex, round 06) — a chunk `get` delivers names the source it came from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["paged", "query"])
def test_a_chunk_from_get_carries_the_locator_of_the_source_whose_text_it_delivers(
    context: QueryContext, route: str
) -> None:
    """The gate's reproduction, both routes. k03's external article paged at a 100-char
    budget, and prioritised by `--query Quillfeather`: `surfaces == ()` on both, and the
    chunk carried `url = https://x.com/vgonpa/status/k03` — the ITEM's — with offsets into
    a surface not in the bundle. `surface_id` did not substitute: it is opaque and holds
    neither `source_index` nor `content_kind` nor the source's URL.

    Every chunk now carries the SURFACE's locator narrowed to its own range: kind,
    `source_index`, `content_kind` and the source's URL, plus `char_start`/`char_end`.
    `chunk.url` keeps its one meaning — where a human opens the owner. Seen red before the
    fix: `AttributeError: 'KnowledgeChunk' object has no attribute 'locator'`.
    """
    item = context.store["k03"]
    source = next(s for s in item_surfaces(item) if s.surface_type == "external_article")
    kwargs = {"limits": GetLimits(char_budget=100)}
    if route == "query":
        kwargs["query"] = "Quillfeather"
    bundle = get("k03", context, surfaces=("external_article",), **kwargs)

    assert bundle.surfaces == (), "the route under test delivers chunks, not the surface"
    assert bundle.chunks
    for chunk in bundle.chunks:
        assert chunk.locator.kind == "content_source"
        assert chunk.locator.source_index == source.locator.source_index is not None
        assert chunk.locator.content_kind == "external_article"
        assert chunk.locator.url == source.locator.url == "https://example.org/essay"
        assert (chunk.locator.char_start, chunk.locator.char_end) == (
            chunk.char_start,
            chunk.char_end,
        )
        assert source.text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.url == item.url, "the owner's URL keeps its one meaning"


# ---------------------------------------------------------------------------
# S-7 (gate Fable, round 06) — the bundle's verification is the live store's, and current
# ---------------------------------------------------------------------------


def test_get_hydrates_a_current_verdict_and_drops_a_stale_one(context: QueryContext) -> None:
    """Spec §3.4 / Plan 01 §3.4 (M5): `verification` is exposed on the `EvidenceBundle`,
    hydrated from the live store with the freshness check `generate._verdict_badge`
    applies. The conduct existed and was correct; no test knew: `verification={}` in
    `get_service.get` left 443 tests green (the gate's mutation, reproduced), and what
    would vanish in silence is every FAIL a consumer of `get` would see on a summary a
    judge already called unfaithful.

    Two halves, one item: a verdict whose contract fingerprint matches the current output
    travels with its fingerprint; the same verdict over a regenerated summary is stale and
    the bundle carries `{}` — never the old PASS. Seen red under `verification={}`: the
    first assertion; under a hydration with no freshness check: the last.
    """
    from datetime import datetime, timezone

    from xbrain.models import VerificationVerdict
    from xbrain.verification import contract_fingerprint, fingerprint_output

    item = context.store["k03"]
    stamp = contract_fingerprint(item, "summary", context.language)
    assert stamp is not None
    judged = item.model_copy(
        update={
            "verification": {
                "summary": VerificationVerdict(
                    target="summary",
                    verdict="FAIL",
                    faithfulness="FAIL",
                    adherence="PASS",
                    output_fingerprint=fingerprint_output(item, "summary"),
                    contract_fingerprint=stamp,
                    verified_at=datetime.now(timezone.utc),
                )
            }
        }
    )
    judged_context = QueryContext(**{**context.__dict__, "store": {**context.store, "k03": judged}})

    bundle = get("k03", judged_context)
    assert set(bundle.verification) == {"summary"}
    assert bundle.verification["summary"].verdict == "FAIL"
    assert bundle.verification["summary"].contract_fingerprint == stamp

    regenerated = judged.model_copy(
        update={
            "enriched": judged.enriched.model_copy(
                update={"summary": judged.enriched.summary + " Y una frase nueva."}
            )
        }
    )
    stale_context = QueryContext(
        **{**context.__dict__, "store": {**context.store, "k03": regenerated}}
    )
    assert get("k03", stale_context).verification == {}, "a stale FAIL must not be shown as current"


# ---------------------------------------------------------------------------
# The two things this child OWNS that the ported blob did not: the chunker
# parameters `QueryContext` started carrying in 02.9, and the budget constant
# `config._index_settings` imports (R12).
# ---------------------------------------------------------------------------


def _indexed_chunk_ids(data: Path, item_id: str, surface_type: str) -> list[str]:
    """The chunk ids the persisted index holds for ONE item's surface, in its own order.

    Scoped by owner as well as by type, because `owner_id` and `surface_type` are two
    different narrowings and the corpus carries three external articles.
    """
    connection = open_index(db_path(data / "index"), read_only=True)
    try:
        return [
            row[0]
            for row in connection.execute(
                "SELECT chunk_id FROM chunks WHERE owner_type = 'item' AND owner_id = ? "
                "AND surface_type = ? ORDER BY chunk_index",
                (item_id, surface_type),
            )
        ]
    finally:
        connection.close()


@pytest.mark.parametrize("route", ["paged", "query"])
def test_get_cuts_with_the_chunker_parameters_its_context_carries(
    tmp_path: Path, corpus, route: str
) -> None:
    """`get` re-cuts the body it pages; `QueryContext.params` is what it must cut WITH.

    `chunk_id` is `<surface_id>:<chunk_index>:<chunker_version>` and carries no trace of the
    parameters, so a `get` that cut at the module default while the index was built at
    another setting would hand a consumer ids that resolve in the index to DIFFERENT bytes —
    the same name over a different cut, which is the one failure an id is supposed to make
    impossible. `params` reached `QueryContext` in 02.9 for the neighbouring reason (the door
    refuses a manifest whose parameters have moved); this is the second consumer, and rule 5
    says both hands must read the one field.

    Measured on a NON-default cut, because at `target=800` the two agree by coincidence and
    the assertion would restate the constant (rule 2): k03's 1,740-character article yields 4
    chunks at the default (423/423/423/471) and 9 at `target=200, max_chars=400`. The
    precondition below asserts that inequality rather than the two counts, so the test stays
    honest if the fixture's body ever moves.

    BOTH ROUTES (B1, gate review): the original test only exercised the paged route. The
    query route passes through `_ranked_chunks` → `_chunks_of(item, surface, params)`, and if
    the `params` argument were ever broken there, no test would catch it.

    Seen red with `params=DEFAULT_CHUNKER_PARAMS` hard-coded in `_chunks_of`: `get` returns
    2 chunk ids where the index holds 8, and the first id names 800 characters in one place
    and 200 in the other.
    """
    tight = ChunkerParams(target=200, max_chars=400)
    store, vocab, pages = corpus
    data = tmp_path / "data"
    _persist(data, store, vocab, pages)
    index_build.build(
        data / "index",
        index_build.load_index_inputs(*_paths(data)),
        options=index_build.IndexOptions(params=tight),
    )
    context = _context(data, store, vocab, pages, params=tight)

    default_cut = chunk_surfaces(
        tuple(s for s in item_surfaces(store["k03"]) if s.surface_type == "external_article"),
        topics=item_topics(store["k03"]),
        url=store["k03"].url,
        blocks_by_surface_id=article_block_texts(store["k03"]),
    )
    indexed = _indexed_chunk_ids(data, "k03", "external_article")
    assert len(indexed) > len(default_cut), (
        "the parameters must actually change the cut or this test restates the default"
    )

    body = next(s for s in item_surfaces(store["k03"]) if s.surface_type == "external_article").text
    served: list[tuple[str, str]] = []
    cursor: str | None = None
    # A query that appears across the fixture body — "Retrieval quality" appears in all
    # paragraphs of k03's article, so the query route returns multiple chunks.
    query_term = "Retrieval quality"
    for _page in range(50):
        kwargs: dict = {
            "surfaces": ("external_article",),
            "limits": GetLimits(char_budget=len(body) // 4),
            "cursor": cursor,
        }
        if route == "query":
            kwargs["query"] = query_term
        bundle = get("k03", context, **kwargs)
        assert bundle.surfaces == (), "the budget must force the chunk route or nothing is cut"
        served += [(chunk.chunk_id, chunk.text) for chunk in bundle.chunks]
        if not bundle.truncated:
            break
        cursor = bundle.cursor
    else:
        pytest.fail("pagination never terminated")

    # The PAGED route delivers ALL chunks in emitter order.
    # The QUERY route delivers only chunks that score for the query, ranked by relevance.
    # What BOTH must do is cut with `params`, not the module default — that's the invariant.
    served_ids = [chunk_id for chunk_id, _text in served]
    if route == "paged":
        # Paged route preserves order, so exact list comparison.
        assert served_ids == indexed
        # The first indexed chunk's TEXT under the tight params must differ from the default.
        assert dict(served)[indexed[0]] != default_cut[0].text
    else:
        # Query route: verify that returned chunks use the tight parameters' cut, not the
        # default. The key observable is chunk id membership: every returned chunk id must
        # exist in the index built with `tight` params, and each such chunk's text must match
        # the indexed text exactly — proving the query route used the same params.
        assert served, "the query route must return at least one chunk"
        for chunk_id, text in served:
            assert chunk_id in indexed, (
                f"query route returned chunk {chunk_id!r} not in indexed set — wrong params?"
            )
        # Additional: no chunk should be as long as the default cut's first chunk, which is
        # ~423 chars at default params vs ~200 chars at tight params.
        default_first_len = len(default_cut[0].text)
        for chunk_id, text in served:
            assert len(text) < default_first_len, (
                f"chunk {chunk_id!r} is {len(text)} chars, same as default cut — wrong params"
            )


def test_the_default_budget_is_the_one_get_applies_when_the_caller_names_none(
    long_article_context: QueryContext,
) -> None:
    """`DEFAULT_CHAR_BUDGET` is the configurable ceiling, and R12 makes it a public name.

    `config._index_settings` imports it in child 02.12 so `[index].char_budget` has a default
    to fall back on, and `load_config` runs on EVERY CLI invocation — so the export is load
    bearing for the whole CLI, not decoration.

    Asserted through behaviour rather than by reading the constant back, which would restate
    it (rule 1): a `get` that names no `limits` must equal one that names exactly this budget,
    the fixture's whole 20,147-character body must fit under it, and one character less must
    be enough to force the pagination — so the number is the ceiling that actually binds and
    not a field nothing consults.
    """
    item = next(iter(long_article_context.store.values()))
    kwargs = {"surfaces": ("external_article",)}

    implicit = get(item.id, long_article_context, **kwargs)
    explicit = get(
        item.id, long_article_context, limits=GetLimits(char_budget=DEFAULT_CHAR_BUDGET), **kwargs
    )
    assert implicit == explicit
    (surface,) = implicit.surfaces
    assert len(surface.text) <= DEFAULT_CHAR_BUDGET

    tighter = get(
        item.id,
        long_article_context,
        limits=GetLimits(char_budget=len(surface.text) - 1),
        **kwargs,
    )
    assert tighter.surfaces == () and tighter.truncated is True


def test_pagination_crosses_from_one_surface_to_the_next_without_mixing_them(
    context: QueryContext,
) -> None:
    """Two surfaces, one budget: the walk has to leave the first and arrive at the second.

    Three branches of `_paginate` exist only in this shape and no single-surface test reaches
    any of them: the page that EXHAUSTS a chunked surface with budget to spare and must point
    at the next one (`_encode(position + 1, 0)`), the page that then SKIPS the surfaces
    already delivered (`position < start_surface`), and the arrival that delivers the second
    surface whole. A cursor that named the exhausted surface again would re-serve its last
    chunks; one that named the next surface's first CHUNK would fragment a body that fits.

    Requested as `("external_article", "summary")` because `_select` preserves EMITTER order
    and the article has to come first for the crossing to happen at all — measured on the
    fixture: 4 chunks of 423/423/423/471 against a 900-character budget, so the crossing
    lands on page 3 rather than by luck on page 1.

    Asserted by REASSEMBLY, in both directions: the chunks concatenate to the stored article
    byte for byte, the summary arrives whole exactly once, and no chunk is served twice.
    """
    item = context.store["k03"]
    article = next(s for s in item_surfaces(item) if s.surface_type == "external_article")
    summary = next(s for s in item_surfaces(item) if s.surface_type == "summary")

    chunk_texts: list[str] = []
    chunk_ids: list[str] = []
    whole: list[KnowledgeSurface] = []
    pages = 0
    cursor: str | None = None
    for _page in range(20):
        bundle = get(
            "k03",
            context,
            surfaces=("external_article", "summary"),
            limits=GetLimits(char_budget=900),
            cursor=cursor,
        )
        pages += 1
        chunk_texts += [c.text for c in bundle.chunks]
        chunk_ids += [c.chunk_id for c in bundle.chunks]
        whole += list(bundle.surfaces)
        if not bundle.truncated:
            assert bundle.cursor is None
            break
        assert bundle.cursor is not None and bundle.cursor != cursor
        cursor = bundle.cursor
    else:
        pytest.fail("pagination never terminated")

    assert pages > 2, "the crossing must happen on a later page, not on the first"
    assert "".join(chunk_texts) == article.text
    assert len(set(chunk_ids)) == len(chunk_ids), "a chunk was served on two pages"
    assert whole == [summary], (
        "the short surface arrives WHOLE and exactly once — compared against the surface the "
        "emitter produced, so a shortened body goes red here and not only a missing one"
    )


def test_a_negative_cursor_component_is_refused_like_any_other_malformed_one(
    context: QueryContext,
) -> None:
    """`int("-1")` parses. A negative offset is a restart wearing a cursor's clothes.

    `_component` raises on `ValueError` — which `-1` never triggers — so without the sign
    check a consumer following a corrupted cursor resumes at `wanted[-1]`, the LAST surface,
    and calls it progress. Both positions are exercised because they are two calls to the
    same guard and a check on one of them would leave the other open.
    """
    for bad in ("-1:0", "0:-1"):
        with pytest.raises(ValueError, match="Cursor inválido"):
            get("k03", context, surfaces=("external_article",), cursor=bad)
    with pytest.raises(ValueError, match="Cursor inválido"):
        get("k03", context, surfaces=("external_article",), query="Quillfeather", cursor="q:-1")


def test_a_topic_the_vocabulary_no_longer_holds_is_skipped_not_raised(
    context: QueryContext,
) -> None:
    """`vocab.yaml` and `items.json` are edited independently, and `get` reads both.

    An enrichment naming a slug the vocabulary has since dropped is an ordinary state of the
    store — `vocab --regenerate` and a hand edit both produce it — and it must cost the
    caller that one topic record, never the whole bundle. The item keeps the topic in its own
    projection (`KnowledgeItem.topics` is what the STORE says), so the omission is visible
    rather than silent: the two lists differ, and they differ by exactly the dropped slug.

    Asserted with a SURVIVING topic beside the dropped one, because a bundle with no topics
    at all cannot distinguish "skipped one" from "gave up on the loop".
    """
    item = context.store["k03"]
    enriched = item.enriched
    assert enriched is not None
    both = item.model_copy(
        update={"enriched": enriched.model_copy(update={"topics": ["ai-policy", "gone-slug"]})}
    )
    local = QueryContext(**{**context.__dict__, "store": {**context.store, "k03": both}})

    bundle = get("k03", local)

    assert [record.slug for record in bundle.topics] == ["ai-policy"]
    assert "gone-slug" in bundle.item.topics, "the store's own claim is not edited away"
