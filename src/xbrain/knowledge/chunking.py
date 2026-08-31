"""The structural chunker — a surface in, its indexable fragments out (spec §5.2).

Chunking is STRUCTURAL wherever the data allows it, because the structure the author gave
the text is better than any window we could impose: an article's paragraphs, an X Article's
blocks, a topic's notes. Only a transcript — which has no paragraphs, being a machine's flat
rendering of speech — is windowed, and then with overlap, because a sentence cut across a
boundary is findable from neither half.

TWO RULES DECIDE ALMOST EVERYTHING HERE.

**Atomic beats `MAX_CHARS` (m4).** The plan's table calls a post, a summary, an image
description, a frame and a quoted post "one unit, uncut", while `MAX_CHARS` was described as
a hard ceiling above which anything is windowed. Those contradict, and the corpus has real
cases on the wrong side: the P2 quoted post is 3 943 chars against a 2 000 ceiling. Atomic
wins, and the reason is attribution, not tidiness — a quoted post has ONE author, and half
of it is a fragment that no longer says whose words it is. `MAX_CHARS` therefore applies
only to the splittable surfaces.

**The parameters are ARGUMENTS, not module constants (M7).** Plan 02 sweeps target ×
overlap, picks a winner and bumps `CHUNKER_VERSION`. If the characterization fixture read a
module constant, that sweep would break the very fixture that exists to pin the ranking —
and the comfortable fix would be to regenerate it, at which point it pins nothing. The
defaults live in `DEFAULT_CHUNKER_PARAMS`; the pinned test passes its own.

The values below are PROVISIONAL and declared as such (Plan 01 §3.6). They are not measured
— Plan 02 measures them.
"""

from __future__ import annotations

from dataclasses import dataclass

from xbrain.knowledge.ids import CHUNKER_VERSION, chunk_fingerprint, chunk_id
from xbrain.knowledge.models import KnowledgeChunk, KnowledgeSurface, SurfaceType
from xbrain.models import ARTICLE_PARAGRAPH_SEP, Author


@dataclass(frozen=True)
class ChunkerParams:
    """Provisional chunking parameters, passed explicitly so a sweep cannot move a pin."""

    target: int = 1200  # soft ceiling per chunk
    max_chars: int = 2000  # hard ceiling — SPLITTABLE surfaces only (see the module docstring)
    overlap: int = 150  # windows only; a paragraph split never overlaps
    min_chars: int = 40  # below this a fragment is merged into a neighbour, not emitted


DEFAULT_CHUNKER_PARAMS = ChunkerParams()

# Surfaces emitted whole, whatever their length. The unit of ATTRIBUTION (a quoted post, a
# frame caption, a photo description) and the unit of MEANING (a post, a summary, a topic
# note) are the same thing here, and splitting either produces fragments that no longer say
# who wrote them or what they are about.
ATOMIC_SURFACES: frozenset[SurfaceType] = frozenset(
    {
        "post",
        "summary",
        "image_description",
        "video_frame",
        "quoted_post",
        "topic_note",
        "topic_description",
        "user_note",
    }
)

# Surfaces with no internal structure to split on — a machine's flat rendering of speech. The
# only ones that get windows, and therefore the only ones that get overlap.
WINDOWED_SURFACES: frozenset[SurfaceType] = frozenset({"video_transcript"})


def chunk_surface(
    surface: KnowledgeSurface,
    *,
    params: ChunkerParams = DEFAULT_CHUNKER_PARAMS,
    topics: tuple[str, ...] = (),
    url: str | None = None,
    blocks: list[str] | None = None,
    chunker_version: str = CHUNKER_VERSION,
) -> tuple[KnowledgeChunk, ...]:
    """One surface's chunks, in order, each verbatim against the surface.

    `blocks` is the X-Article block boundary list when the source carries one (41 sources in
    the corpus). Spec §4 says the blocks define boundaries but are NOT indexed in addition to
    `.text`: the concatenation of the text blocks IS `text`, guaranteed by a
    `ContentSourceSuccess` model validator, so emitting both would duplicate the corpus.

    The invariant every caller may rely on, asserted as a property over the whole fixture
    corpus: `surface.text[chunk.char_start:chunk.char_end] == chunk.text`. That is spec
    §3.8's verifiability claim made operational — a consumer can slice the stored surface
    and get back exactly what they were shown.
    """
    spans = _spans(surface, params, blocks)
    return tuple(
        _chunk(surface, index, start, end, topics, url, chunker_version)
        for index, (start, end) in enumerate(spans)
    )


def chunk_surfaces(
    surfaces: tuple[KnowledgeSurface, ...],
    *,
    params: ChunkerParams = DEFAULT_CHUNKER_PARAMS,
    topics: tuple[str, ...] = (),
    url: str | None = None,
    chunker_version: str = CHUNKER_VERSION,
) -> tuple[KnowledgeChunk, ...]:
    """Every chunk of every surface, in surface order then chunk order.

    Deterministic across runs, which spec §3.7.8 needs for stable ordering under ties: an
    index built twice from the same store must produce the same ids in the same sequence, or
    a tie-break on `chunk_id` would silently reorder results between rebuilds.
    """
    chunks: list[KnowledgeChunk] = []
    for surface in surfaces:
        chunks += chunk_surface(
            surface, params=params, topics=topics, url=url, chunker_version=chunker_version
        )
    return tuple(chunks)


def _spans(
    surface: KnowledgeSurface, params: ChunkerParams, blocks: list[str] | None
) -> list[tuple[int, int]]:
    """The `(start, end)` spans this surface splits into.

    Offsets rather than strings all the way through, so "verbatim" is structural rather than
    something the emitter has to remember to preserve: the text of a chunk is only ever a
    slice, never a rebuilt string.
    """
    text = surface.text
    if not text:
        return []
    if surface.surface_type in ATOMIC_SURFACES:
        return [(0, len(text))]
    if surface.surface_type in WINDOWED_SURFACES:
        return _window_spans(len(text), params)
    if blocks:
        return _merge_short(_block_spans(blocks), text, params)
    return _merge_short(_paragraph_spans(text), text, params)


def _block_spans(blocks: list[str]) -> list[tuple[int, int]]:
    """Spans taken straight from the X Article's own block lengths.

    The blocks are contiguous and their concatenation is the surface text, so cumulative
    lengths are exact offsets — no searching, and no chance of a boundary landing one
    character off because a separator was counted twice.
    """
    spans, cursor = [], 0
    for block in blocks:
        spans.append((cursor, cursor + len(block)))
        cursor += len(block)
    return spans


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Spans split on the blank-line paragraph separator, keeping every character.

    The separator is `models.ARTICLE_PARAGRAPH_SEP` — the same constant the Article producer
    bakes in and the renderer strips — rather than a private `"\\n\\n"` literal, so the two
    cannot drift. Each separator is kept at the END of the preceding span: dropping it would
    leave gaps between the spans and break the coverage property.
    """
    spans, cursor = [], 0
    parts = text.split(ARTICLE_PARAGRAPH_SEP)
    for index, part in enumerate(parts):
        end = cursor + len(part) + (len(ARTICLE_PARAGRAPH_SEP) if index < len(parts) - 1 else 0)
        spans.append((cursor, end))
        cursor = end
    return spans


def _window_spans(length: int, params: ChunkerParams) -> list[tuple[int, int]]:
    """Overlapping windows over a flat body, covering it end to end.

    The stride is `target - overlap`, so consecutive windows share `overlap` characters and a
    sentence spanning a boundary is complete in at least one of them. The final window is
    clamped to the end of the text rather than emitted short, so the coverage property holds
    without a special case.
    """
    stride = max(params.target - params.overlap, 1)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + params.target, length)
        spans.append((start, end))
        if end == length:
            break
        start += stride
    return spans or [(0, length)]


def _merge_short(
    spans: list[tuple[int, int]], text: str, params: ChunkerParams
) -> list[tuple[int, int]]:
    """PACK consecutive structural units up to `target`, then split anything over `max_chars`.

    `target` is a SOFT ceiling per chunk, not "one chunk per paragraph", and the difference
    is load-bearing. Measured on the real corpus (2026-08-31, 2,404 items): emitting one
    chunk per paragraph produced **30,449** chunks where the plan's own volume estimate,
    derived from the measured character counts, predicted 18–25k — and the plan says landing
    outside that range means the chunker is not doing what it describes. The whole gap was
    small paragraphs: `x_article` averaged **194 chars** across 11,016 chunks from 210
    articles.

    A 194-character chunk is bad retrieval before it is bad arithmetic. It carries too little
    context for a reader to judge the match, and it scatters one argument across a dozen ids
    so bm25 sees a dozen weak documents instead of one strong one.

    Packing keeps the author's boundaries — a chunk always starts and ends on a paragraph
    edge — and simply stops adding paragraphs when the next one would cross `target`. The
    `min_chars` floor is subsumed: a scrap can never stand alone, because it is packed with
    its neighbour. It NEVER deletes — a surface whose entire text is below the floor still
    yields one chunk, since dropping it would remove an item from the corpus with nothing
    reporting the loss.
    """
    packed: list[tuple[int, int]] = []
    for start, end in spans:
        if not packed:
            packed.append((start, end))
            continue
        open_start, open_end = packed[-1]
        combined = end - open_start
        too_short = len(text[open_start:open_end].strip()) < params.min_chars
        if combined <= params.target or too_short:
            packed[-1] = (open_start, end)
        else:
            packed.append((start, end))

    out: list[tuple[int, int]] = []
    for start, end in packed:
        out += (
            _oversize_spans(start, end, params)
            if end - start > params.max_chars
            else [(start, end)]
        )
    return out


def _oversize_spans(start: int, end: int, params: ChunkerParams) -> list[tuple[int, int]]:
    """A single splittable unit longer than `max_chars`, cut into `target`-sized pieces.

    Spec §5.2: *a window only if a section exceeds the ceiling*. No overlap is applied — this
    is not a transcript, and overlapping real paragraphs would duplicate prose in the index
    and make the same sentence match twice under two ids for no recall gain.
    """
    spans, cursor = [], start
    while cursor < end:
        stop = min(cursor + params.target, end)
        spans.append((cursor, stop))
        cursor = stop
    return spans


def _chunk(
    surface: KnowledgeSurface,
    index: int,
    start: int,
    end: int,
    topics: tuple[str, ...],
    url: str | None,
    chunker_version: str,
) -> KnowledgeChunk:
    """One chunk, with everything it needs to be citable and filterable on its own.

    `title` travels WITH the chunk (m6, spec §4): a match on chunk 7 of a 20 k article would
    otherwise reach the consumer as an orphan paragraph. It is accompanying metadata, not a
    chunk of its own, so it adds nothing to the indexed corpus.

    Provenance is COPIED from the surface rather than re-derived. Re-deriving would be a
    second definition of the same fact (CLAUDE.md rule 5), and the one that mattered — the
    quoted post's third-party author — is exactly the one a second derivation would get
    wrong by falling back to the item.
    """
    text = surface.text[start:end]
    cid = chunk_id(surface.surface_id, index, chunker_version=chunker_version)
    attribution: Author | None = surface.attribution
    return KnowledgeChunk(
        chunk_id=cid,
        surface_id=surface.surface_id,
        owner_type=surface.owner_type,
        owner_id=surface.owner_id,
        surface_type=surface.surface_type,
        text=text,
        title=surface.title,
        chunk_index=index,
        char_start=start,
        char_end=end,
        origin=surface.origin,
        trust_class=surface.trust_class,
        derived=surface.derived,
        attribution=attribution,
        topics=topics,
        url=url or surface.locator.url,
        language=surface.language,
        fingerprint=chunk_fingerprint(
            surface.surface_id, index, text, chunker_version=chunker_version
        ),
    )
