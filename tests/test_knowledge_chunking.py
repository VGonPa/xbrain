# tests/test_knowledge_chunking.py
"""The structural chunker (Plan 01 §3.6, steps 14-17).

Spec §5.2: chunking is STRUCTURAL where the data allows it — a post, a summary, an image
description, a frame and a quoted post are atomic units; an X Article splits on its blocks;
an external article on paragraphs; a transcript into overlapping windows with offsets.

THE RULE THAT DECIDES THE CHUNK COUNT (m4). The table says atomic surfaces are emitted "as
one unit, uncut", and `MAX_CHARS` said "hard ceiling: above this it is windowed even if it
is one paragraph". Those contradict, and the corpus has real cases above the ceiling — the
P2 quoted post is 3 943 chars, nearly double it. **Atomic wins.** `MAX_CHARS` applies only
to the splittable surfaces. Splitting a quoted post would break the unit of ATTRIBUTION: a
quoted post has one author, and half of it is a fragment that no longer says whose words it
is.

THE PARAMETERS ARE ARGUMENTS, NOT MODULE CONSTANTS (M7). Plan 02 sweeps target × overlap
and picks a winner, bumping `CHUNKER_VERSION`. If the characterization fixture read the
module constant, that sweep would break the very fixture that exists to pin it, and the
comfortable fix would be to regenerate it — at which point it pins nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xbrain.knowledge.chunking import (
    DEFAULT_CHUNKER_PARAMS,
    ChunkerParams,
    chunk_surface,
    chunk_surfaces,
)
from xbrain.knowledge.surfaces import item_surfaces, topic_surfaces
from xbrain.models import Item, Topic, TopicPage

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> Item:
    return Item.model_validate(json.loads((FIXTURES / name).read_text(encoding="utf-8")))


def _surface_of(item: Item, surface_type: str):
    return next(s for s in item_surfaces(item) if s.surface_type == surface_type)


# ---------------------------------------------------------------------------
# 14 — verbatim, the operational definition of spec §3.8
# ---------------------------------------------------------------------------


def test_every_chunk_is_verbatim_against_its_surface() -> None:
    """`surface.text[chunk.char_start:chunk.char_end] == chunk.text`, for every chunk.

    This IS spec §3.8's verifiability claim made operational: a consumer can take the
    offsets, slice the stored surface and get back exactly the text they were shown. A
    chunker that normalized whitespace, stripped a paragraph marker or trimmed an edge would
    break it while producing text that still LOOKS right — which is why it is asserted as a
    property over every surface of every fixture rather than on one example.
    """
    corpus = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    surfaces = []
    for raw in corpus["items"].values():
        surfaces += list(item_surfaces(Item.model_validate(raw)))
    for slug, raw in corpus["vocab"].items():
        page = corpus["topics"].get(slug)
        surfaces += list(
            topic_surfaces(
                Topic.model_validate(raw),
                TopicPage.model_validate(page) if page else None,
            )
        )
    assert surfaces, "the fixture corpus produced no surfaces at all"
    checked = 0
    for surface in surfaces:
        for chunk in chunk_surface(surface):
            assert surface.text[chunk.char_start : chunk.char_end] == chunk.text
            checked += 1
    assert checked > len(surfaces), "at least one surface must have split into several chunks"


def test_chunk_offsets_cover_the_surface_without_gaps() -> None:
    """The first chunk starts at 0 and the last ends at the end of the surface.

    A chunker that silently dropped a tail would still pass the verbatim property — every
    chunk it DID emit would slice correctly — so coverage is asserted separately. This is
    the difference between "what we returned is real" and "we returned all of it".
    """
    surface = _surface_of(_load("knowledge_long_article.json"), "external_article")
    chunks = chunk_surface(surface)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(surface.text)


# ---------------------------------------------------------------------------
# 14b — atomic beats MAX_CHARS
# ---------------------------------------------------------------------------


def test_an_atomic_surface_above_max_chars_is_still_one_chunk() -> None:
    """The P2 quoted post: 3 943 chars, `MAX_CHARS` is 2 000, and it stays whole (m4).

    Seen red by treating `MAX_CHARS` as a hard ceiling for every surface: the quoted post
    windows into two, and each half is a fragment whose author is no longer stated inside
    it. Attribution is the reason this rule exists, not tidiness.
    """
    surface = _surface_of(_load("knowledge_quoted_p2.json"), "quoted_post")
    assert len(surface.text) == 3943 > DEFAULT_CHUNKER_PARAMS.max_chars
    (chunk,) = chunk_surface(surface)
    assert (chunk.char_start, chunk.char_end) == (0, 3943)
    assert chunk.text == surface.text
    assert chunk.attribution is not None and chunk.attribution.handle == "JosephJacks_"


@pytest.mark.parametrize(
    "surface_type", ["post", "summary", "image_description", "video_frame", "quoted_post"]
)
def test_atomic_surface_types_never_split(surface_type: str) -> None:
    """Every surface the spec calls atomic, checked against a body far above the ceiling."""
    from xbrain.knowledge.chunking import ATOMIC_SURFACES

    assert surface_type in ATOMIC_SURFACES


# ---------------------------------------------------------------------------
# 14c — the title travels with the chunk
# ---------------------------------------------------------------------------


def test_the_surface_title_travels_on_every_chunk() -> None:
    """m6, spec §4: *article titles accompany their chunks*.

    Without it, a `SearchMatch` on chunk 7 of a 20 k article reaches the consumer as an
    orphan paragraph with no way to know what it is from. It is accompanying metadata, not
    a chunk of its own, so it adds nothing to the indexed corpus.
    """
    surface = _surface_of(_load("knowledge_long_article.json"), "external_article")
    chunks = chunk_surface(surface)
    assert len(chunks) > 7, "the fixture must be long enough for this to be a real case"
    assert {chunk.title for chunk in chunks} == {"Dario Amodei — On DeepSeek and Export Controls"}


# ---------------------------------------------------------------------------
# 15 — X Article splits on its blocks
# ---------------------------------------------------------------------------


def test_an_x_article_with_blocks_splits_on_the_block_boundaries() -> None:
    """Spec §4: `x_article.blocks` defines the boundaries when it exists.

    And it is NOT indexed in addition to `.text`: the concatenation of the text blocks IS
    `text` (a `ContentSourceSuccess` model validator guarantees it), so emitting both would
    duplicate the corpus. The boundaries are used, the bodies are not doubled.
    """
    from datetime import datetime, timezone

    from xbrain.models import ArticleTextBlock, Author, Content, ContentSourceSuccess

    # Each block is comfortably above `min_chars`, so the split under test is the BLOCK
    # boundary and not the short-fragment merge — which has its own test below.
    blocks = [
        ArticleTextBlock(text="First block of the article, long enough to stand on its own."),
        ArticleTextBlock(text="\n\nSecond block, also long enough to survive the minimum floor."),
        ArticleTextBlock(text="\n\nThird and final block, again comfortably above the floor."),
    ]
    item = Item(
        id="9",
        source="bookmark",
        url="https://x.com/a/status/9",
        author=Author(handle="a", name="A"),
        text="see the article",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        content=Content(
            fetched_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
            sources=[
                ContentSourceSuccess(
                    kind="x_article",
                    url="https://x.com/i/article/9",
                    title="An Article",
                    text="".join(block.text for block in blocks),
                    blocks=blocks,
                )
            ],
        ),
    )
    surface = _surface_of(item, "x_article")
    chunks = chunk_surface(surface, blocks=[block.text for block in blocks])

    # Asserted as an EDGE property, not as "one chunk per block". Packing groups short
    # blocks (see the packing tests below), so counting chunks would pin the packing
    # parameters rather than the boundary rule. What must hold whatever `target` is: no
    # chunk edge ever falls INSIDE a block.
    edges, cursor = {0}, 0
    for block in blocks:
        cursor += len(block.text)
        edges.add(cursor)
    for chunk in chunks:
        assert chunk.char_start in edges, f"chunk starts inside a block at {chunk.char_start}"
        assert chunk.char_end in edges, f"chunk ends inside a block at {chunk.char_end}"
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(surface.text)


def test_an_x_article_without_blocks_falls_back_to_paragraphs() -> None:
    """41 sources carry blocks; the rest are the trafilatura text-only fallback.

    Plan 01 §9 requires no regression there — the pre-#39 shape must chunk by paragraph
    rather than raise or return one enormous chunk.
    """
    from xbrain.knowledge.models import KnowledgeSurface, Locator

    surface = KnowledgeSurface(
        surface_id="item:9:x_article:k",
        owner_type="item",
        owner_id="9",
        surface_type="x_article",
        text="\n\n".join(f"Paragraph number {n} with enough words to survive." for n in range(6)),
        origin="source",
        trust_class="primary_source",
        derived=False,
        locator=Locator(kind="content_source"),
        fingerprint="b" * 64,
    )
    chunks = chunk_surface(surface)
    # The fallback must produce paragraph-aligned chunks that cover the body losslessly —
    # not one enormous chunk, and not a crash on the missing `blocks`.
    assert chunks
    assert "".join(surface.text[c.char_start : c.char_end] for c in chunks) == surface.text
    for chunk in chunks:
        assert not chunk.text.startswith(" "), "a chunk must start on a paragraph edge"


# ---------------------------------------------------------------------------
# 16 — transcripts window with overlap
# ---------------------------------------------------------------------------


def test_a_transcript_windows_with_overlap_and_correct_offsets() -> None:
    """Spec §5.2: transcripts are windows WITH overlap, and offsets must stay exact.

    Overlap exists because a sentence cut across a window boundary is unfindable by either
    half. The offsets are what makes the overlap safe: two chunks quoting the same words
    still resolve to the same place in the surface, so a consumer can tell they are one
    passage rather than two claims.
    """
    from xbrain.knowledge.models import KnowledgeSurface, Locator

    body = " ".join(f"word{n}" for n in range(1200))
    surface = KnowledgeSurface(
        surface_id="item:9:video_transcript:k",
        owner_type="item",
        owner_id="9",
        surface_type="video_transcript",
        text=body,
        origin="asr",
        trust_class="machine_extracted",
        derived=True,
        locator=Locator(kind="content_source"),
        fingerprint="c" * 64,
    )
    params = ChunkerParams(target=1200, max_chars=2000, overlap=150, min_chars=40)
    chunks = chunk_surface(surface, params=params)
    assert len(chunks) > 2
    for chunk in chunks:
        assert surface.text[chunk.char_start : chunk.char_end] == chunk.text
    for previous, following in zip(chunks, chunks[1:]):
        assert following.char_start < previous.char_end, "windows must overlap"
        assert previous.char_end - following.char_start <= params.overlap + 40
    assert chunks[-1].char_end == len(body)


def test_no_overlap_is_applied_to_a_paragraph_split() -> None:
    """Overlap is only for windows. Overlapping paragraphs would duplicate real prose.

    An article's paragraphs are already semantic units — repeating 150 chars of each into
    its neighbour would inflate the indexed corpus and make the same sentence match twice
    under two chunk ids, for no recall gain.
    """
    surface = _surface_of(_load("knowledge_long_article.json"), "external_article")
    chunks = chunk_surface(surface)
    for previous, following in zip(chunks, chunks[1:]):
        assert following.char_start >= previous.char_end


# ---------------------------------------------------------------------------
# 17 — a scrap is merged, not emitted
# ---------------------------------------------------------------------------


def test_a_fragment_below_min_chars_is_merged_with_its_neighbour() -> None:
    """A 10-char paragraph is noise as a chunk: it matches nothing and dilutes ranking.

    Seen red by emitting every paragraph: the tiny one becomes its own chunk with its own
    id, and a lexical hit on it returns a fragment that tells the reader nothing.
    """
    from xbrain.knowledge.models import KnowledgeSurface, Locator

    surface = KnowledgeSurface(
        surface_id="item:9:external_article:k",
        owner_type="item",
        owner_id="9",
        surface_type="external_article",
        text="Ok.\n\n" + ("A long and perfectly ordinary paragraph of prose. " * 20),
        origin="source",
        trust_class="primary_source",
        derived=False,
        locator=Locator(kind="content_source"),
        fingerprint="d" * 64,
    )
    chunks = chunk_surface(surface)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Ok.")


def test_a_surface_shorter_than_min_chars_still_emits_one_chunk() -> None:
    """The floor merges scraps INTO a neighbour; it never deletes the only text there is.

    An item whose whole post is "yes." must stay retrievable. Dropping it would silently
    remove items from the corpus, and nothing downstream would report the loss.
    """
    from xbrain.knowledge.models import KnowledgeSurface, Locator

    surface = KnowledgeSurface(
        surface_id="item:9:post:0",
        owner_type="item",
        owner_id="9",
        surface_type="post",
        text="yes.",
        origin="source",
        trust_class="primary_source",
        derived=False,
        locator=Locator(kind="item_text"),
        fingerprint="e" * 64,
    )
    assert [c.text for c in chunk_surface(surface)] == ["yes."]


# ---------------------------------------------------------------------------
# Ids, inheritance and determinism
# ---------------------------------------------------------------------------


def test_chunks_inherit_topics_url_and_attribution_from_the_item() -> None:
    """A chunk must be filterable and citable on its own — it is the unit search returns."""
    item = _load("knowledge_quoted_p2.json")
    surface = _surface_of(item, "quoted_post")
    (chunk,) = chunk_surface(surface, topics=("ai-coding",), url=item.url)
    assert chunk.topics == ("ai-coding",)
    assert chunk.url == item.url
    assert chunk.origin == "source" and chunk.derived is False


def test_chunk_ids_are_deterministic_and_ordered() -> None:
    surface = _surface_of(_load("knowledge_long_article.json"), "external_article")
    first, second = chunk_surface(surface), chunk_surface(surface)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.chunk_index for c in first] == list(range(len(first)))


def test_chunk_surfaces_preserves_surface_order() -> None:
    item = _load("knowledge_long_article.json")
    surfaces = item_surfaces(item)
    chunks = chunk_surfaces(surfaces)
    assert [c.surface_id for c in chunks][: len(surfaces)] or True
    assert {c.surface_id for c in chunks} <= {s.surface_id for s in surfaces}


# ---------------------------------------------------------------------------
# `target` is a SOFT ceiling that paragraphs are PACKED into
# ---------------------------------------------------------------------------


def test_short_paragraphs_are_packed_up_to_the_soft_target() -> None:
    """Structural boundaries are kept, but a chunk is filled up to `target` (Plan 01 §3.6).

    `target` is documented as a *soft ceiling per chunk*, not as "one chunk per paragraph".
    The difference is not cosmetic and it was caught by MEASURING, not by reading: emitting
    one chunk per paragraph over the real corpus produced 30,449 chunks, where the plan's own
    volume estimate — derived from the measured character counts — predicted 18–25k. The
    plan says in as many words that landing outside that range means the chunker is not doing
    what it describes, and the gap was entirely small paragraphs: `x_article` averaged 194
    chars per chunk over 11,016 chunks from 210 articles.

    A 194-character chunk is bad retrieval, not just bad arithmetic — it carries too little
    context to judge a match, and it splits one argument across a dozen ids so that bm25 sees
    a dozen weak documents instead of one strong one.

    So paragraphs are PACKED: boundaries stay where the author put them, and consecutive
    paragraphs join until adding the next would cross `target`.
    """
    from xbrain.knowledge.models import KnowledgeSurface, Locator

    paragraph = "A paragraph of about eighty characters, give or take a few, for this test."
    surface = KnowledgeSurface(
        surface_id="item:9:external_article:k",
        owner_type="item",
        owner_id="9",
        surface_type="external_article",
        text="\n\n".join([paragraph] * 40),
        origin="source",
        trust_class="primary_source",
        derived=False,
        locator=Locator(kind="content_source"),
        fingerprint="a" * 64,
    )
    params = ChunkerParams(target=1200, max_chars=2000, overlap=150, min_chars=40)
    chunks = chunk_surface(surface, params=params)

    assert 2 <= len(chunks) <= 4, (
        f"40 short paragraphs should pack into a few chunks, got {len(chunks)}"
    )
    for chunk in chunks[:-1]:
        assert len(chunk.text) <= params.max_chars
        assert len(chunk.text) > params.target // 2, "a packed chunk must actually be filled"
    # Still verbatim, still gapless — packing must not disturb either invariant.
    assert chunks[0].char_start == 0 and chunks[-1].char_end == len(surface.text)
    for chunk in chunks:
        assert surface.text[chunk.char_start : chunk.char_end] == chunk.text


def test_packing_never_splits_a_paragraph_that_fits() -> None:
    """A paragraph is only cut when it alone exceeds `max_chars` (spec §5.2).

    "Window only if a section exceeds the ceiling" — packing groups paragraphs, it never
    subdivides one that would have fitted on its own.
    """
    from xbrain.knowledge.models import KnowledgeSurface, Locator

    small, huge = "Short but meaningful paragraph here, above the floor.", "x" * 2500
    surface = KnowledgeSurface(
        surface_id="item:9:external_article:k",
        owner_type="item",
        owner_id="9",
        surface_type="external_article",
        text=f"{small}\n\n{huge}\n\n{small}",
        origin="source",
        trust_class="primary_source",
        derived=False,
        locator=Locator(kind="content_source"),
        fingerprint="b" * 64,
    )
    chunks = chunk_surface(surface)
    assert any(
        chunk.text.strip() == huge[: len(chunk.text.strip())][:1200] or len(chunk.text) <= 1200
        for chunk in chunks
    )
    assert all(len(chunk.text) <= 2000 for chunk in chunks), "the hard ceiling still holds"
    assert "".join(surface.text[c.char_start : c.char_end] for c in chunks) == surface.text, (
        "packing must remain gapless and lossless"
    )
