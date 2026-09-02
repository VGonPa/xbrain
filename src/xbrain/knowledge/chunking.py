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

THE VALUES ARE MEASURED NOW, and they moved (Plan 02 §7). The Plan 01 provisional was
`target=1200, overlap=150`; the sweep over `target ∈ {800,1200,1600,2400} × overlap ∈
{0,150,300}` against the 23 scorable golden-set cases on the real 2,404-item corpus put
`target=800, overlap=0` first. **On MRR** — 0.8179 vs 0.7449 — and on `recall@1` (0.6034 vs
0.4730); the round-03 claim that it also won on `recall@10` (0.8119 vs 0.8027, with gains
in `enterrado`, `semantico` and `cruzado_idioma`) was an artefact of scoring at a depth of
ten CHUNKS, and it is retracted (U-6, round 07): with the depth counted in OWNERS the two
targets tie on `recall@10` at every overlap (0.8264) and their per-stratum recalls are
identical cell for cell. The winner is the same; the reason is narrower. The cost is
+21.6 % chunks (18,320 -> 22,286). The chunk-size distribution does NOT return to the
194-character pathology that motivated packing: the median moves 658 -> 670 and
`x_article` averages 661.

**THE OVERLAP AXIS WAS DECIDED BY A RETRIEVER THAT CANNOT USE IT, AND THAT IS A DECLARED
LIMIT, NOT A FINDING.** Overlap applies only to `video_transcript`, the one windowed surface,
and it exists so a sentence spanning a boundary is complete in at least one window. But
`lexical_fts.match_expression` quotes every TERM separately and never builds a phrase, so
this retriever cannot benefit from a whole sentence — measured, `recall@10` is IDENTICAL for
overlap 0 and 150 at every target, and only MRR moves (more overlap = more near-duplicate
chunks competing). The golden set also contains no case that requires a sentence to survive a
boundary. So the sweep chose 0 on the evidence it has, and Plan 03 MUST re-decide this axis
with the vector retriever in front of it: an embedding of a truncated sentence is a worse
vector, and that is a cost this measurement is blind to.
"""

from __future__ import annotations

from dataclasses import dataclass

from xbrain.knowledge.ids import CHUNKER_VERSION, chunk_fingerprint, chunk_id
from xbrain.knowledge.models import KnowledgeChunk, KnowledgeSurface, Locator, SurfaceType
from xbrain.models import ARTICLE_PARAGRAPH_SEP, Author


@dataclass(frozen=True)
class ChunkerParams:
    """Provisional chunking parameters, passed explicitly so a sweep cannot move a pin."""

    target: int = 800  # soft ceiling per chunk — MEASURED, see the module docstring
    max_chars: int = 2000  # hard ceiling — SPLITTABLE surfaces only (see the module docstring)
    overlap: int = 0  # windows only; a paragraph split never overlaps
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
    blocks_by_surface_id: dict[str, list[str]] | None = None,
    chunker_version: str = CHUNKER_VERSION,
) -> tuple[KnowledgeChunk, ...]:
    """Every chunk of every surface, in surface order then chunk order.

    Deterministic across runs, which spec §3.7.8 needs for stable ordering under ties: an
    index built twice from the same store must produce the same ids in the same sequence, or
    a tie-break on `chunk_id` would silently reorder results between rebuilds.

    `blocks_by_surface_id` is HOW THE X-ARTICLE BOUNDARIES REACH THIS MODULE. `chunk_surface`
    could always take `blocks`, but this — the only batch entry point, and the only one the
    CLI and the evaluation harness call — could not pass them, so the branch was unreachable
    from every caller and all 41 sources that carry blocks were chunked by the paragraph
    fallback. Build the map with `surfaces.article_block_texts(item)`; omitting it keeps the
    fallback, which is the right behaviour for a topic surface or a pre-#39 article.
    """
    blocks_by_surface_id = blocks_by_surface_id or {}
    chunks: list[KnowledgeChunk] = []
    for surface in surfaces:
        chunks += chunk_surface(
            surface,
            params=params,
            topics=topics,
            url=url,
            blocks=blocks_by_surface_id.get(surface.surface_id),
            chunker_version=chunker_version,
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
    edge — and simply stops adding paragraphs when the next one would cross `target`.

    THE FLOOR IS NOT SUBSUMED BY THE PACKING, and the docstring used to claim it was. Packing
    looks BACKWARDS — it merges a unit into the span before it — so the LAST unit has nobody
    to join and is appended however short it is; and `_oversize_spans` runs AFTER the packing,
    so the remainders it leaves are never re-packed at all. Measured on the real corpus
    (2026-08-31, 17,642 chunks): 9 chunks sat below the floor without being the whole surface
    (`external_article` 7, `video_digest` 2), the smallest a 4-char `'Woo!'` and one a 16-char
    `'zure AI Foundry.'` cut mid-word. 0.05% of the corpus, and not cosmetic: bm25 normalizes
    by length, so a 16-character chunk holding the query term can outrank the paragraph that
    answers it. `_absorb_scraps` closes it by merging forwards-produced scraps backwards.

    It NEVER deletes — a surface whose entire text is below the floor still yields one chunk,
    since dropping it would remove an item from the corpus with nothing reporting the loss.
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
    return _absorb_scraps(out, text, params)


def _absorb_scraps(
    spans: list[tuple[int, int]], text: str, params: ChunkerParams
) -> list[tuple[int, int]]:
    """Merge any span below `min_chars` into the one before it — the LAST pass, after all.

    Both producers above leave scraps the packing cannot reach: packing merges backwards, so
    a short final unit has no successor to absorb it, and `_oversize_spans` runs afterwards
    and its remainder is never re-packed.

    A scrap is merged into its PREDECESSOR, so the first span is never a scrap's victim and
    a surface that is entirely below the floor keeps its single chunk. The merge can push a
    chunk at most `min_chars` past `target`, which is far below `max_chars` — a 1,216-char
    chunk is a better document than a 1,200-char one plus a 16-char one.

    Coverage is preserved: the spans are contiguous, and merging two adjacent ones leaves
    them contiguous, so `surface.text[start:end] == chunk.text` still tiles the surface.
    """
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and len(text[start:end].strip()) < params.min_chars:
            merged[-1] = (merged[-1][0], end)
            continue
        merged.append((start, end))
    return merged


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


def fragment_locator(surface_locator: Locator, char_start: int, char_end: int) -> Locator:
    """The locator of a FRAGMENT of a surface: the surface's, narrowed to the range (seam b).

    ONE function for any served fragment — the chunk `get` delivers, the match `search`
    returns — because the attribution/locator family reappeared by four routes in five
    rounds (a search without the surfaces join, a fingerprint blind to the author, the human
    chunk header, the chunks of `get`), each time through a consumer building its own. The
    surface's locator says where the surface lives in the original data (source index and
    kind, media or frame index, the source's own URL); the range says which bytes of it
    this fragment is. Nothing else is derived, and nothing is invented: a surface with no
    resolvable locator yields no fragment (`index_store.resolvable_hits`), never a
    fabricated one.
    """
    return surface_locator.model_copy(update={"char_start": char_start, "char_end": char_end})


def chunk_evidence(
    *,
    surface_id: str,
    chunk_index: int,
    text: str,
    owner_type: str,
    owner_id: str,
    surface_type: str,
    origin: str,
    trust_class: str,
    derived: bool,
    char_start: int,
    char_end: int,
    attribution: Author | None,
    locator: Locator,
) -> tuple[str, ...]:
    """Everything the index SERVES about a chunk, as the ONE tuple its fingerprint hashes (U-5).

    The second half of seam (b), extended to integrity. `fragment_locator` is the one
    construction of a served fragment's locator; this is the one projection of a served
    fragment's EVIDENCE — the text, the surface it came from and its position in it, the
    owner the words are hydrated under, the provenance that qualifies them (`origin`,
    `trust_class`, `derived`, `surface_type`), the attribution that says whose words they
    are, and the narrowed locator that says where to check. `_chunk` hashes it at emission;
    `index_store.verify_fingerprints` rebuilds it from the served row — the chunk's columns,
    the surface row's attribution and locator, the locator narrowed through
    `fragment_locator` — and a row on which any arm was rewritten no longer recomputes: it
    is excluded and counted, exactly like a text that does not match its hash. Before this
    the fingerprint covered three of these fields and the other nine were served on trust
    (gate Codex F4: a quoted post served as the poster's own `summary`, `origin: llm`, with a
    valid URL to the poster's page and `corrupt_chunks_excluded: 0`).

    Attribution is hashed as its two stored columns, never as a model dump: the index keeps
    `handle` and `name` and rebuilds the `Author` from them, so a field added to `Author`
    upstream would otherwise make every fingerprint fail on the next query. `None` and an
    author with an empty name are distinct arms. The locator IS a model dump — it is stored
    as one (`locator_json`) and rebuilt by validation, and a round trip is byte-stable.
    """
    return (
        surface_id,
        str(chunk_index),
        text,
        owner_type,
        owner_id,
        surface_type,
        str(origin),
        str(trust_class),
        "1" if derived else "0",
        str(char_start),
        str(char_end),
        "author" if attribution is not None else "",
        attribution.handle if attribution is not None else "",
        attribution.name if attribution is not None else "",
        locator.model_dump_json(),
    )


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

    `url` is the OWNER's URL — where a human opens this chunk — and has exactly that one
    meaning. It used to read `url or surface.locator.url`, which looked like a designed
    fallback to the surface's own address and was dead: both callers always pass a truthy
    `item.url`, so 9,377 of 9,377 non-`post` chunks carried the tweet's URL and none carried
    their own. The fallback is deleted rather than reached, because the two are not
    interchangeable: a `video_transcript`'s locator holds a SIGNED, EXPIRING
    `video.twimg.com` URL, and a `quoted_post`'s would still be the poster's page. Serving
    either as the chunk's citable link buys nothing and rots.

    The precise position travels on `locator` (B2, round 06): `fragment_locator` narrows the
    surface's locator to this range. It used to stay on `surface.locator` alone — "which the
    consumer already receives" — and the consumer did NOT receive it whenever `get` delivered
    chunks instead of the surface, which is exactly when a chunk exists.
    """
    text = surface.text[start:end]
    cid = chunk_id(surface.surface_id, index, chunker_version=chunker_version)
    attribution: Author | None = surface.attribution
    locator = fragment_locator(surface.locator, start, end)
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
        url=url,
        locator=locator,
        language=surface.language,
        # Over the whole served evidence (U-5), through the projection the verifier shares.
        fingerprint=chunk_fingerprint(
            chunk_evidence(
                surface_id=surface.surface_id,
                chunk_index=index,
                text=text,
                owner_type=surface.owner_type,
                owner_id=surface.owner_id,
                surface_type=surface.surface_type,
                origin=surface.origin,
                trust_class=surface.trust_class,
                derived=surface.derived,
                char_start=start,
                char_end=end,
                attribution=attribution,
                locator=locator,
            ),
            chunker_version=chunker_version,
        ),
    )
