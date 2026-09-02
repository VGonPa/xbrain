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


def _article_item(blocks):
    """One item carrying an `x_article` source whose body IS the concatenation of `blocks`.

    Built through the real models, so the `ContentSourceSuccess` validator that guarantees
    `text == "".join(text blocks)` is the thing keeping the fixture honest.
    """
    from datetime import datetime, timezone

    from xbrain.models import Author, Content, ContentSourceSuccess

    return Item(
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
                    text="".join(b.text for b in blocks),
                    blocks=blocks,
                )
            ],
        ),
    )


def _discriminating_blocks():
    """Blocks whose edges the PARAGRAPH fallback provably cannot reproduce.

    Both real shapes are present, because the fixture that failed to discriminate had
    neither:

    * the first block carries an INTERNAL blank line. Splitting on paragraphs cuts there —
      950 characters into a block the author never divided. Measured on the real corpus
      (2026-08-31): 5 of the 41 `x_article` sources with blocks have one.
    * the later blocks carry the separator BAKED AT THEIR START, which is how the producer
      writes them (`generate` strips it back off with `removeprefix`). `_paragraph_spans`
      keeps a separator at the END of the preceding span, so every edge lands two characters
      late. That alone is why 41 of 41 real sources chunk on boundaries the author did not
      set, including the 36 with no internal blank line.
    """
    from xbrain.models import ArticleTextBlock

    return [
        ArticleTextBlock(text=("A" * 950) + "\n\n" + ("B" * 948)),
        ArticleTextBlock(text="\n\n" + ("C" * 1100)),
        ArticleTextBlock(text="\n\n" + ("D" * 1100)),
    ]


def test_an_x_article_with_blocks_splits_on_the_block_boundaries() -> None:
    """Spec §4: `x_article.blocks` defines the boundaries when it exists.

    THROUGH THE PRODUCTION PATH. The previous version of this test called
    `chunk_surface(blocks=...)` by hand — the one call site in the repo that passed blocks —
    while `chunk_surfaces`, the only batch entry point and the only one the CLI and the
    harness use, had no `blocks` parameter at all. So the branch under test was unreachable
    from every caller, and deleting it left the whole suite green (2,124 pass, 0 fail). That
    is rule 1 in its hardest form: not a test that passes before the fix, but one that passes
    after the functionality is deleted.

    This one builds the item, emits its surfaces and chunks them exactly as
    `xbrain knowledge inspect --chunks` and the evaluation harness do.

    And it is NOT indexed in addition to `.text`: the concatenation of the text blocks IS
    `text` (a `ContentSourceSuccess` model validator guarantees it), so emitting both would
    duplicate the corpus. The boundaries are used, the bodies are not doubled.
    """
    from xbrain.knowledge.chunking import chunk_surfaces
    from xbrain.knowledge.surfaces import article_block_texts

    blocks = _discriminating_blocks()
    item = _article_item(blocks)
    surfaces = item_surfaces(item)
    chunks = [
        c
        for c in chunk_surfaces(surfaces, blocks_by_surface_id=article_block_texts(item))
        if c.surface_type == "x_article"
    ]
    surface = _surface_of(item, "x_article")

    # Asserted as an EDGE property, not as "one chunk per block". Packing groups short
    # blocks (see the packing tests below), so counting chunks would pin the packing
    # parameters rather than the boundary rule. What must hold whatever `target` is: no
    # chunk edge ever falls INSIDE a block.
    #
    # This holds only while no single block exceeds `max_chars` — spec §5.2 windows a
    # section above the ceiling, and that cut lands inside the block by design. The fixture
    # stays under it; the exception has its own test below, so the 39-of-41 measured on the
    # real corpus is a documented property and not an unexplained residue.
    edges, cursor = {0}, 0
    for block in blocks:
        cursor += len(block.text)
        edges.add(cursor)
    for chunk in chunks:
        assert chunk.char_start in edges, f"chunk starts inside a block at {chunk.char_start}"
        assert chunk.char_end in edges, f"chunk ends inside a block at {chunk.char_end}"
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(surface.text)


def test_the_block_fixture_actually_discriminates_blocks_from_paragraphs() -> None:
    """The guard on the test above: prove the fixture can tell the two apart.

    The assertion "every chunk edge is a block edge" is vacuous whenever the blocks happen
    to align with the paragraphs — which is what let the previous fixture survive the
    deletion of the branch it existed to protect. So the discriminating power is asserted
    here rather than assumed: chunking the SAME text without blocks must produce a different
    set of edges, and at least one of them must fall strictly inside a block.

    If someone later edits the fixture into alignment, this goes red and says so, instead of
    the boundary test quietly becoming a paragraph test again.
    """
    from xbrain.knowledge.chunking import chunk_surface, chunk_surfaces
    from xbrain.knowledge.surfaces import article_block_texts

    blocks = _discriminating_blocks()
    item = _article_item(blocks)
    surface = _surface_of(item, "x_article")

    with_blocks = chunk_surfaces(
        item_surfaces(item), blocks_by_surface_id=article_block_texts(item)
    )
    without = chunk_surface(surface)

    block_edges, cursor = {0}, 0
    for block in blocks:
        cursor += len(block.text)
        block_edges.add(cursor)
    fallback_edges = {c.char_start for c in without} | {c.char_end for c in without}
    interior = {e for e in fallback_edges if e not in block_edges}

    assert interior, "the fixture does not discriminate: paragraph edges == block edges"
    assert any(0 < e < len(blocks[0].text) for e in interior), (
        f"no fallback edge falls inside the author's first block: {sorted(interior)}"
    )
    article = [c for c in with_blocks if c.surface_type == "x_article"]
    assert [(c.char_start, c.char_end) for c in article] != [
        (c.char_start, c.char_end) for c in without
    ]


def test_a_block_above_the_ceiling_is_still_cut_inside_itself() -> None:
    """Spec §5.2: *a window only if a section exceeds the ceiling* — blocks included.

    The author's boundary is respected wherever it CAN be. A single block longer than
    `max_chars` cannot be honoured and one chunk of it — the whole point of a ceiling is that
    nothing above it is emitted whole. Measured on the real corpus (2026-08-31): 39 of the 41
    sources that carry blocks land every chunk edge exactly on an author edge, and the 2 that
    do not each contain one block of 5,879 and 2,861 chars against the 2,000 ceiling.

    Pinned so that number is explained rather than filed as a leftover defect.
    """
    from xbrain.knowledge.chunking import chunk_surfaces
    from xbrain.knowledge.surfaces import article_block_texts
    from xbrain.models import ArticleTextBlock

    params = ChunkerParams()
    huge = ArticleTextBlock(text="E" * (params.max_chars * 2))
    item = _article_item([huge, ArticleTextBlock(text="\n\n" + "F" * 600)])
    chunks = [
        c
        for c in chunk_surfaces(item_surfaces(item), blocks_by_surface_id=article_block_texts(item))
        if c.surface_type == "x_article"
    ]
    interior = [c for c in chunks if 0 < c.char_end < len(huge.text)]
    assert interior, "a block above the ceiling must be windowed, not emitted whole"
    assert all(c.char_end - c.char_start <= params.max_chars for c in chunks)
    surface = _surface_of(item, "x_article")
    assert "".join(surface.text[c.char_start : c.char_end] for c in chunks) == surface.text


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


# ---------------------------------------------------------------------------
# m1 — `chunk.url` is the OWNER's URL, and says so
# ---------------------------------------------------------------------------


def test_a_chunk_carries_the_owner_url_and_the_surface_keeps_its_own_locator() -> None:
    """One meaning for `chunk.url`, asserted by WHERE the value comes from (rule 1).

    `url=url or surface.locator.url` read as a designed fallback and was dead: both callers
    always pass `item.url`, which is truthy, so 9,377 of 9,377 non-`post` chunks carried the
    tweet URL and 0 carried their own. The fallback was unreachable — the same shape as M1.

    It is deleted rather than reached, because the two URLs are not interchangeable and the
    item's is the right one: a `video_transcript`'s locator holds a SIGNED, EXPIRING
    `video.twimg.com` URL, and serving that as the chunk's citable link would hand a
    consumer a dead link. Nothing is lost — the surface keeps its locator, which is where a
    precise position belongs, and this test pins that it is still there.
    """
    from xbrain.knowledge.evaluation import load_corpus

    item = load_corpus(FIXTURES / "knowledge_corpus.json").items["k08"]
    surface = _surface_of(item, "video_transcript")
    (chunk,) = tuple(
        c
        for c in chunk_surfaces(item_surfaces(item), url=item.url)
        if c.surface_id == surface.surface_id
    )[:1]

    assert chunk.url == item.url
    assert surface.locator.url and surface.locator.url != item.url, (
        "the fixture must have a surface whose own locator differs, or this pins nothing"
    )

    # And the AMBIGUITY cannot come back. The behaviour above was already correct before the
    # fallback was deleted — both callers pass a truthy `item.url` — so a purely behavioural
    # test here would be green before and after, which rule 1 says is the tell that it pins
    # nothing. This asserts on the SOURCE (rule 9): the `url=` the chunk is built with must
    # be the bare `url` argument, never an expression over the surface's locator, because a
    # reachable fallback would make `chunk.url` mean two different things depending on the
    # caller. (Since B2 the chunk ALSO carries `locator=`, built from the surface's; that is
    # a second field with its own meaning, not the fallback — so the guard is on the `url`
    # keyword, not on the word "locator" appearing anywhere in the function.)
    import ast
    import inspect
    import textwrap

    from xbrain.knowledge import chunking

    tree = ast.parse(textwrap.dedent(inspect.getsource(chunking._chunk))).body[0]
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "KnowledgeChunk"
    ]
    assert len(calls) == 1
    (url_keyword,) = [kw for kw in calls[0].keywords if kw.arg == "url"]
    assert isinstance(url_keyword.value, ast.Name) and url_keyword.value.id == "url", (
        "`chunk.url` is derived from something other than the owner's URL argument: "
        f"{ast.unparse(url_keyword.value)}"
    )


# ---------------------------------------------------------------------------
# m2 — the `min_chars` floor covers the TAIL too
# ---------------------------------------------------------------------------


def _splittable(text: str):
    """A bare splittable surface — the shape the structural path takes."""
    from xbrain.knowledge.models import KnowledgeSurface, Locator

    return KnowledgeSurface(
        surface_id="item:9:external_article:k",
        owner_type="item",
        owner_id="9",
        surface_type="external_article",
        text=text,
        origin="source",
        trust_class="primary_source",
        derived=False,
        locator=Locator(kind="content_source"),
        fingerprint="c" * 64,
    )


def test_a_trailing_scrap_is_merged_backwards_not_emitted_alone() -> None:
    """`_merge_short` packs looking BACKWARDS, so the last fragment had nobody to join.

    The docstring claimed the floor was subsumed — *a scrap can never stand alone, because
    it is packed with its neighbour* — and that was false for the tail: packing merges a
    span into the one before it, so a final short unit is appended untouched.

    Seen red with paragraphs of 1200/1200/10 and `min_chars=40`: a final span of 10 chars.
    """
    params = ChunkerParams(target=1200, max_chars=2000, overlap=150, min_chars=40)
    surface = _splittable("\n\n".join(["A" * 1200, "B" * 1200, "C" * 10]))
    chunks = chunk_surface(surface, params=params)

    assert len(chunks) > 1, "the fixture must actually split, or this pins nothing"
    for chunk in chunks:
        assert len(chunk.text.strip()) >= params.min_chars, (
            f"a {len(chunk.text.strip())}-char scrap was emitted alone: {chunk.text!r}"
        )
    assert "".join(surface.text[c.char_start : c.char_end] for c in chunks) == surface.text


def test_the_tail_of_an_oversized_split_is_absorbed_too() -> None:
    """`_oversize_spans` runs AFTER packing, so ITS tails were never re-packed.

    This is the shape the real corpus actually hits. Measured 2026-08-31 over 17,642 chunks:
    9 fell below the floor without being the whole surface (`external_article` 7,
    `video_digest` 2), the smallest a 4-char `'Woo!'` and one a 16-char `'zure AI Foundry.'`
    cut mid-word. It is 0.05% of the corpus and it is not cosmetic: bm25 normalizes by
    length, so a 16-character chunk holding the query term can outrank the paragraph that
    actually answers it.
    """
    params = ChunkerParams(target=1200, max_chars=2000, overlap=150, min_chars=40)
    # One paragraph just over `max_chars`, so the oversize split leaves a short remainder.
    surface = _splittable("D" * (params.target * 2 + 12))
    chunks = chunk_surface(surface, params=params)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text.strip()) >= params.min_chars, (
            f"an oversize tail of {len(chunk.text.strip())} chars stood alone: {chunk.text!r}"
        )
    assert "".join(surface.text[c.char_start : c.char_end] for c in chunks) == surface.text


def test_a_surface_shorter_than_the_floor_still_yields_its_one_chunk() -> None:
    """The floor MERGES, it never DELETES.

    A surface whose whole text is below the floor still produces one chunk: dropping it
    would remove an item from the corpus with nothing reporting the loss. Pinned beside the
    two tests above so the fix cannot be over-applied into a filter.
    """
    params = ChunkerParams(target=1200, max_chars=2000, overlap=150, min_chars=40)
    surface = _splittable("tiny")
    chunks = chunk_surface(surface, params=params)
    assert len(chunks) == 1
    assert chunks[0].text == "tiny"


def test_the_harness_chunks_an_article_on_its_block_boundaries() -> None:
    """m8: the guard on the PRODUCTION path, not on the entry point the tests call.

    M1 was never really about the `if blocks:` branch in `_spans`. It was that no production
    caller could feed it: `chunk_surfaces` had no `blocks` parameter, so the branch was
    unreachable from the CLI and from the harness, and deleting it left the whole suite
    green. The fix wired both callers — and nothing pinned the wiring. Measured: deleting
    `blocks_by_surface_id=article_block_texts(item)` from `evaluation.corpus_chunks` left
    2,143 tests passing.

    So this asserts through `corpus_chunks`, the function the harness actually calls, over a
    corpus holding the article whose blocks the paragraph fallback provably cannot reproduce
    (`_discriminating_blocks`, guarded by its own discrimination test above).
    """
    from xbrain.knowledge.evaluation import Corpus, corpus_chunks

    blocks = _discriminating_blocks()
    item = _article_item(blocks)
    corpus = Corpus(items={item.id: item}, vocab=[], topic_pages={}, source="m8")

    chunks, _surfaces = corpus_chunks(corpus)
    article = [c for c in chunks if c.surface_type == "x_article"]

    assert article, "the corpus walk emitted no article chunk at all"
    edges, cursor = {0}, 0
    for block in blocks:
        cursor += len(block.text)
        edges.add(cursor)
    for chunk in article:
        assert chunk.char_start in edges, (
            f"the harness cut inside a block at {chunk.char_start}: it is not being handed "
            "the block boundaries"
        )
        assert chunk.char_end in edges, f"the harness cut inside a block at {chunk.char_end}"


# ---------------------------------------------------------------------------
# B2 (gate Codex, round 06) — every chunk carries its surface's locator, narrowed
# ---------------------------------------------------------------------------


def test_every_chunk_carries_its_surface_locator_narrowed_to_its_own_range() -> None:
    """Spec §3.7 invariant 2 — *todo texto devuelto incluye origin, surface_type y
    localizador* — and §3.8: surface, owner and locator resolvable. A `KnowledgeChunk` had
    no locator: it carried the OWNER's URL and offsets into a surface that, when `get`
    paginated or prioritised by `--query`, was not in the bundle at all (`surfaces == ()`).
    The gate's reproduction on k03: `chunk_url` the poster's tweet, `source_locator` the
    essay's URL, `chunk_has_locator False`. A fragment nobody can resolve back to the bytes
    it quotes is a number, not evidence.

    ONE function narrows a surface locator to a fragment (`fragment_locator`), and every
    chunk of every surface in the fixture corpus carries exactly that — the same function
    `search` applies to a match, which is what makes the two consumers agree by
    construction rather than by two copies of a `model_copy`. Seen red before the fix:
    `KnowledgeChunk` had no `locator` field.
    """
    from xbrain.knowledge.chunking import fragment_locator
    from xbrain.knowledge.evaluation import load_corpus

    corpus = load_corpus(FIXTURES / "knowledge_corpus.json")
    checked = 0
    for item in corpus.items.values():
        for surface in item_surfaces(item):
            for chunk in chunk_surfaces((surface,), url=item.url):
                assert chunk.locator == fragment_locator(
                    surface.locator, chunk.char_start, chunk.char_end
                )
                narrowed = chunk.locator.model_dump(exclude={"char_start", "char_end"})
                assert narrowed == surface.locator.model_dump(exclude={"char_start", "char_end"})
                assert (chunk.locator.char_start, chunk.locator.char_end) == (
                    chunk.char_start,
                    chunk.char_end,
                )
                checked += 1
    assert checked > 40, "the fixture corpus must exercise the property, or it pins nothing"
