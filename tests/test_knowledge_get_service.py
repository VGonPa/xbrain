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
from xbrain.knowledge.contracts import EvidenceBundle
from xbrain.knowledge.get_service import (
    DEFAULT_SURFACES,
    GetLimits,
    UnknownSurfaceError,
    get,
)
from xbrain.knowledge.search_service import QueryContext
from xbrain.models import Item, Topic, TopicPage

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def corpus() -> tuple[dict[str, Item], list[Topic], dict[str, TopicPage]]:
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    return (
        {k: Item.model_validate(v) for k, v in raw["items"].items()},
        [Topic.model_validate(v) for v in raw["vocab"].values()],
        {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()},
    )


@pytest.fixture()
def context(tmp_path: Path, corpus) -> QueryContext:
    store, vocab, pages = corpus
    data = tmp_path / "data"
    data.mkdir()
    (data / "items.json").write_text(
        json.dumps({k: v.model_dump(mode="json") for k, v in store.items()}), encoding="utf-8"
    )
    index_build.build(data / "index", store, vocab, pages, data / "items.json")
    return QueryContext(
        store=store,
        vocab=vocab,
        topic_pages=pages,
        index_dir=data / "index",
        items_path=data / "items.json",
    )


@pytest.fixture()
def long_article_context(tmp_path: Path, corpus) -> QueryContext:
    """The S2 case: a real 20k-character article, as a COMMITTED fixture (m-iv).

    The number below is a property of `tests/fixtures/knowledge_long_article.json`, not of the
    corpus — which is why it can be asserted at all. A corpus figure in an assert would be a
    test that goes red when someone runs `enrich` (M6).
    """
    item = Item.model_validate(
        json.loads((FIXTURES / "knowledge_long_article.json").read_text(encoding="utf-8"))
    )
    _store, vocab, pages = corpus
    store = {item.id: item}
    data = tmp_path / "data"
    data.mkdir()
    (data / "items.json").write_text(
        json.dumps({k: v.model_dump(mode="json") for k, v in store.items()}), encoding="utf-8"
    )
    return QueryContext(
        store=store,
        vocab=vocab,
        topic_pages=pages,
        index_dir=data / "index",
        items_path=data / "items.json",
    )


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
    from xbrain.knowledge.chunking import chunk_surfaces
    from xbrain.knowledge.surfaces import item_surfaces

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
    from xbrain.knowledge.surfaces import item_surfaces

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
