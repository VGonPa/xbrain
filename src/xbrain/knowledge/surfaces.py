"""The surface emitter — one item or topic in, its `KnowledgeSurface`s out (spec §3, §4).

THE DESIGN DECISION THAT GOVERNS THIS MODULE. `xbrain.evidence` already answers *what may
support a generated output*. Spec §2.2 says this contract must not replace it and must not
grow a second hand-written list. It also cannot simply call it, because the two differ on
three axes — ON PURPOSE:

| axis          | `evidence.evidence_surfaces`        | here                                |
|---------------|-------------------------------------|-------------------------------------|
| scope         | depends on the target               | every surface, no target            |
| truncation    | `ARTICLE_CHAR_LIMIT` on the body    | never (spec §2.2)                   |
| multiplicity  | the FIRST source of each kind       | all of them — 119 items have >1     |

So what is shared is the ATOMIC WALK, not the assembled block: `iter_content_sources`,
`iter_described_photos` and `iter_video_frames` in `executors/api.py`, which the enrichment
selectors were re-expressed onto. This module reads `.text` off those iterators without
recutting it. The proof that nothing moved on the other side is
`tests/test_evidence_characterization.py`, which pins the judge's source text as a literal.

The maps below are asserted TOTAL by `tests/test_knowledge_surface_coverage.py` — that is
what makes "the two definitions agree" a test rather than a sentence (CLAUDE.md rule 5).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from xbrain.executors.api import (
    _FAILURE_CLAUSE,
    fetched_link_sources,
    iter_content_sources,
    iter_described_photos,
    iter_video_frames,
)
from xbrain.knowledge.ids import (
    SINGLETON_SOURCE_KEY,
    content_source_keys,
    surface_fingerprint,
    surface_id,
    topic_id,
    video_frame_source_key,
)
from xbrain.knowledge.models import (
    UNFETCHED_REASON_BY_FAILURE,
    DerivedText,
    KnowledgeItem,
    KnowledgeSurface,
    Locator,
    SourceFailure,
    SurfaceType,
    TopicRecord,
    UnfetchedLink,
)
from xbrain.knowledge.provenance import ORIGIN_TRUST, Origin, is_derived
from xbrain.models import (
    LINK_CONTENT_KINDS,
    ArticleTextBlock,
    Author,
    ContentKind,
    ContentSourceFailure,
    ContentSourceSuccess,
    Item,
    Topic,
    TopicPage,
    VerificationVerdict,
)
from xbrain.notes_io import note_filename

# Which surfaces each content kind can produce. The value is a TUPLE, not a scalar: an
# `x_video` produces three (transcript, frames, digest), and a one-to-one map left two of
# them unprotected by the totality test — the subtle version of CLAUDE.md rule 1, where the
# assertion was right and the name overpromised.
CONTENT_KIND_TO_SURFACE_TYPES: dict[ContentKind, tuple[SurfaceType, ...]] = {
    "external_article": ("external_article",),
    "x_article": ("x_article",),
    "thread": ("thread",),
    "quoted_tweet": ("quoted_post",),
    "x_video": ("video_transcript", "video_frame", "video_digest"),
}

# Surfaces that do NOT come from a content source: the tweet itself, the photo descriptions,
# the enrichment, the user's own note and the three topic surfaces. Declared so the totality
# closure over `SurfaceType` can be written at all.
NON_CONTENT_SURFACES: frozenset[SurfaceType] = frozenset(
    {
        "post",
        "image_description",
        "summary",
        "user_note",
        "topic_description",
        "topic_overview",
        "topic_note",
    }
)

# Spec §4, the `origin` column, verbatim. `topic_description` is `unknown` because the
# vocabulary does not record whether a description was generated or hand-edited — and
# `ORIGIN_TRUST` then classifies `unknown` as synthesis, the fail-closed path.
SURFACE_ORIGIN: dict[SurfaceType, Origin] = {
    "post": "source",
    "external_article": "source",
    "x_article": "source",
    "thread": "source",
    "quoted_post": "source",
    "video_transcript": "asr",
    "video_frame": "vlm",
    "image_description": "vlm",
    "summary": "llm",
    "video_digest": "llm",
    "topic_overview": "llm",
    "topic_note": "llm",
    "topic_description": "unknown",
    "user_note": "user",
}

# The reach across to the OTHER contract. Every key of `evidence._SURFACES` is either mapped
# to the knowledge surfaces it corresponds to, or declared not to be one. Adding a surface
# for the judge without deciding here is red.
EVIDENCE_KEY_SURFACES: dict[str, tuple[SurfaceType, ...]] = {
    "tweet": ("post",),
    # ONE evidence key, TWO knowledge surfaces: `evidence` selects the article body through
    # `LINK_CONTENT_KINDS`, which is `{external_article, x_article}`.
    "article": ("external_article", "x_article"),
    "thread": ("thread",),
    "quoted": ("quoted_post",),
    "video_transcript": ("video_transcript",),
    "video_frames": ("video_frame",),
    "images": ("image_description",),
}

# Evidence keys that are METADATA, not bodies of text. They ride ON a knowledge surface —
# `KnowledgeSurface.attribution` and `KnowledgeSurface.title` — rather than becoming
# surfaces of their own, because a title is not a citable claim and an author is not a text.
NOT_A_KNOWLEDGE_SURFACE: frozenset[str] = frozenset({"author", "video_title", "article_title"})

# Emitted on every derived surface (spec §3.4's warning list). Machine-produced text is
# usable evidence but the response must say what produced it — an ASR transcript is not the
# speaker's words and a VLM description is not text that appeared in the image.
_MACHINE_WARNING: tuple[str, ...] = ("machine_generated",)


def _blank(text: str | None) -> bool:
    """A surface with nothing but whitespace is not a surface (Plan 01 §9)."""
    return not text or not text.strip()


def _surface(
    *,
    owner_type: str,
    owner_id: str,
    surface_type: SurfaceType,
    source_key: str,
    text: str,
    locator: Locator,
    title: str | None = None,
    attribution: Author | None = None,
    producer: str | None = None,
    produced_at: datetime | None = None,
    language: str | None = None,
) -> KnowledgeSurface:
    """Assemble one surface, deriving origin, trust class, derived-ness and fingerprint.

    Every surface goes through here so those four can never be set by hand at a call site
    and disagree with `SURFACE_ORIGIN` — the single-definition rule applied inside the
    module, not just across modules.
    """
    origin = SURFACE_ORIGIN[surface_type]
    sid = surface_id(owner_type, owner_id, surface_type, source_key)
    derived = is_derived(origin)
    return KnowledgeSurface(
        surface_id=sid,
        owner_type=owner_type,  # type: ignore[arg-type]
        owner_id=owner_id,
        surface_type=surface_type,
        text=text,
        title=title,
        origin=origin,
        trust_class=ORIGIN_TRUST[origin],
        derived=derived,
        attribution=attribution,
        producer=producer,
        produced_at=produced_at,
        locator=locator,
        fingerprint=surface_fingerprint(surface_type, origin, text),
        language=language,
        warnings=_MACHINE_WARNING if derived else (),  # type: ignore[arg-type]
    )


def item_surfaces(
    item: Item,
    *,
    transcribe_command: str | None = None,
    vision_command: str | None = None,
) -> tuple[KnowledgeSurface, ...]:
    """Every indexable surface of one item, in a deterministic order.

    Order: the post, then each content source in `content.sources` order (a video
    contributing its transcript, then its frames, then its digest), then the described
    photos in `item.media` order, then the summary, then the user's own note. Deterministic
    because two runs over the same store must produce the same ids and the same ranking
    (spec §3.7.8).

    `transcribe_command` and `vision_command` are the only producers that do NOT live in the
    store, so they are passed in. Which backend produced a transcript is not bookkeeping:
    CLAUDE.md records that parakeet does not fail on Spanish audio, it INVENTS — so a reader
    must be able to recover, after the fact, what wrote the words they are reading.
    """
    surfaces: list[KnowledgeSurface] = []
    fetched_at = item.content.fetched_at if item.content else None
    keys = content_source_keys(item)

    if not _blank(item.text):
        surfaces.append(
            _surface(
                owner_type="item",
                owner_id=item.id,
                surface_type="post",
                source_key=SINGLETON_SOURCE_KEY,
                text=item.text,
                attribution=item.author,
                locator=Locator(kind="item_text", url=item.url),
            )
        )

    for index, source in iter_content_sources(item, set(CONTENT_KIND_TO_SURFACE_TYPES)):
        surfaces += _content_source_surfaces(
            item,
            index,
            source,
            keys[index],
            fetched_at,
            transcribe_command=transcribe_command,
            vision_command=vision_command,
        )

    for media_index, photo in iter_described_photos(item):
        surfaces.append(
            _surface(
                owner_type="item",
                owner_id=item.id,
                surface_type="image_description",
                source_key=_photo_source_key(photo.url),
                text=photo.description,
                producer=photo.description_version,
                produced_at=photo.described_at,
                language=photo.description_lang,
                locator=Locator(kind="media", media_index=media_index, url=photo.url),
            )
        )

    surfaces += _enrichment_surfaces(item)
    return tuple(surfaces)


def article_block_texts(item: Item) -> dict[str, list[str]]:
    """`{surface_id: the ordered ArticleTextBlock bodies}` for this item's X Articles.

    THE SEAM THAT MAKES SPEC §4's BLOCK BOUNDARIES REACHABLE. The chunker has always known
    how to split an `x_article` on the boundaries its author set, but nothing in production
    could hand it the blocks: `chunk_surfaces` — the only batch entry point, and the only one
    the CLI and the evaluation harness call — had no parameter for them. So all 41 sources
    that carry blocks were chunked by the paragraph fallback, and 41 of 41 landed on
    boundaries the author did not set (measured 2026-08-31 on 2,404 items).

    The map is keyed by `surface_id` rather than by position in `content.sources` for the
    reason `ids.py` hashes `(kind, url)` instead of the index: `fetch` rewrites that list, and
    a position-keyed map would hand one source's blocks to another the first time two entries
    swapped — silently, because the lookup would still succeed.

    Only `ArticleTextBlock` contributes. An image or a video block carries no text and is not
    part of the flattened body, so counting it would push every later boundary off by the
    length of a caption that is not there.

    Blocks whose concatenation does NOT reproduce the surface text are dropped rather than
    used. The `ContentSourceSuccess` validator makes that unreachable for anything the
    producer wrote, but the offsets are the chunker's whole verbatim guarantee
    (`surface.text[start:end] == chunk.text`), so a disagreeing record degrades to the
    paragraph fallback instead of cutting the body at offsets that do not describe it.
    """
    out: dict[str, list[str]] = {}
    keys = content_source_keys(item)
    for index, source in iter_content_sources(item, {"x_article"}):
        texts = [b.text for b in source.blocks if isinstance(b, ArticleTextBlock)]
        if not texts or "".join(texts) != source.text:
            continue
        (surface_type,) = CONTENT_KIND_TO_SURFACE_TYPES[source.kind]
        out[surface_id("item", item.id, surface_type, keys[index])] = texts
    return out


def _enrichment_surfaces(item: Item) -> list[KnowledgeSurface]:
    """The `summary` and the user's own `user_note`, when the item carries them.

    `user_note` has **0 instances** in the corpus today (measured 2026-08-31) and is emitted
    anyway, because the model already allows it and the Plan 05 vault-tail adapter will feed
    the same surface type. Its attribution is the user — it is the one surface here whose
    words a human wrote — and its producer is nobody, because no component generated it.
    """
    if item.enriched is None:
        return []
    out: list[KnowledgeSurface] = []
    if not _blank(item.enriched.summary):
        out.append(
            _surface(
                owner_type="item",
                owner_id=item.id,
                surface_type="summary",
                source_key=SINGLETON_SOURCE_KEY,
                text=item.enriched.summary or "",
                producer=item.enriched.executor,
                produced_at=item.enriched.enriched_at,
                locator=Locator(kind="enrichment"),
            )
        )
    if not _blank(item.enriched.user_notes):
        out.append(
            _surface(
                owner_type="item",
                owner_id=item.id,
                surface_type="user_note",
                source_key=SINGLETON_SOURCE_KEY,
                text=item.enriched.user_notes or "",
                attribution=item.author,
                locator=Locator(kind="enrichment"),
            )
        )
    return out


def _photo_source_key(url: str) -> str:
    from xbrain.knowledge.ids import _sha1_12

    return _sha1_12(url)


def _content_source_surfaces(
    item: Item,
    index: int,
    source: ContentSourceSuccess,
    source_key: str,
    fetched_at: datetime | None,
    *,
    transcribe_command: str | None,
    vision_command: str | None,
) -> list[KnowledgeSurface]:
    """The surfaces one content source contributes.

    Split out because the video branch contributes three surface types and the function
    would otherwise be the one place in this package that radon would flag — and a chunker
    or emitter nobody can read is how a wrong locator survives review.
    """
    locator_base = dict(kind="content_source", source_index=index, content_kind=source.kind)
    out: list[KnowledgeSurface] = []

    if source.kind != "x_video":
        if _blank(source.text):
            return out
        (surface_type,) = CONTENT_KIND_TO_SURFACE_TYPES[source.kind]
        return [
            _surface(
                owner_type="item",
                owner_id=item.id,
                surface_type=surface_type,
                source_key=source_key,
                text=source.text,
                title=source.title,
                # THE attribution rule (#86, spec §3.7 invariant 3): a quoted post keeps the
                # THIRD PARTY's author. `source.author` is None for the poster's own thread
                # and for an article, where the item's author is not the body's author
                # either — so nothing is inherited by default.
                attribution=source.author,
                producer="fetch",
                produced_at=fetched_at,
                language=source.language,
                locator=Locator(**locator_base, url=source.url),  # type: ignore[arg-type]
            )
        ]

    # A no-speech video (107 of 259 in the corpus) has an EMPTY transcript. Emitting it would
    # put a zero-length body in the index under a label that promises speech.
    if not _blank(source.text):
        out.append(
            _surface(
                owner_type="item",
                owner_id=item.id,
                surface_type="video_transcript",
                source_key=source_key,
                text=source.text,
                title=source.title,
                producer=transcribe_command,
                produced_at=fetched_at,
                language=source.language,
                locator=Locator(**locator_base, url=source.url),  # type: ignore[arg-type]
            )
        )
    for _si, _src, frame_index, frame in iter_video_frames(item):
        if _si != index:
            continue
        out.append(
            _surface(
                owner_type="item",
                owner_id=item.id,
                surface_type="video_frame",
                source_key=video_frame_source_key(source_key, frame_index),
                text=frame.description,
                title=source.title,
                producer=vision_command,
                produced_at=fetched_at,
                locator=Locator(
                    kind="video_frame",
                    source_index=index,
                    content_kind=source.kind,
                    url=source.url,
                    frame_index=frame_index,
                    frame_timestamp=frame.timestamp,
                ),
            )
        )
    if not _blank(source.digest):
        out.append(
            _surface(
                owner_type="item",
                owner_id=item.id,
                surface_type="video_digest",
                source_key=source_key,
                text=source.digest,
                title=source.title,
                # The digest is written by the `video-digest` worksheet stage, whose executor
                # is not recorded on the source; `enriched.executor` would be a guess about a
                # different stage, so this stays honest at None.
                produced_at=fetched_at,
                locator=Locator(**locator_base, url=source.url),  # type: ignore[arg-type]
            )
        )
    return out


def item_topics(item: Item) -> tuple[str, ...]:
    """The item's topics, deduplicated, primary first.

    Deterministic on purpose: `KnowledgeChunk.topics` is copied from here and feeds a
    filter, so an order that depended on how the enrichment happened to be written would
    make two otherwise identical items rank differently.
    """
    if item.enriched is None:
        return ()
    ordered = [item.enriched.primary_topic, *item.enriched.topics]
    return tuple(dict.fromkeys(topic for topic in ordered if topic))


def _failed_sources(item: Item) -> tuple[SourceFailure, ...]:
    return tuple(
        SourceFailure(
            kind=source.kind,
            url=source.url,
            failure_reason=source.failure_reason,
            error=source.error,
            http_status=source.http_status,
        )
        for source in (item.content.sources if item.content else [])
        if isinstance(source, ContentSourceFailure)
    )


def _unfetched_links(item: Item) -> tuple[UnfetchedLink, ...]:
    """The URLs in `item.links` with no recovered body, each with its recorded reason (m7).

    Spec §4: these are metadata the consumer must SEE, together with why there is no body —
    the same thing `unfetched_links_note` tells the LLM surfaces. The reason is mapped from
    the store's own `FailureReason` through a table asserted total; it is never invented,
    and `detail` reuses `executors.api._FAILURE_CLAUSE` so the wording cannot drift from
    what the generators and the judge are told.

    THE KNOWN LIMIT, stated rather than hidden: a link is matched to a source by EXACT URL,
    and X's `t.co` shortening means `item.links` often carries a different string from the
    source that resolved it. `links_content_unfetched` sidesteps this by counting rather
    than matching, and this function is gated on it — so nothing is reported unless at least
    one body really is missing. Within that gate the error can only be over-reporting (a
    link whose body was fetched under a resolved URL listed as unattempted), which tells a
    consumer to go and check. The opposite error — staying silent about a URL with no body —
    is the one that lets a model reconstruct an article from its slug, which is precisely
    what PR-I exists to prevent.
    """
    if not item.links:
        return ()
    fetched_urls = {
        source.url for _i, source in iter_content_sources(item, LINK_CONTENT_KINDS) if source.text
    }
    failures = _link_failures(item)
    if fetched_link_sources(item) >= len(item.links) and not failures:
        return ()
    return tuple(
        _unfetched_link(link.url, failures.get(link.url))
        for link in item.links
        if link.url not in fetched_urls
    )


def _link_failures(item: Item) -> dict[str, ContentSourceFailure]:
    """`{url: failure}` for the LINK-content fetches that were attempted and failed.

    Only `LINK_CONTENT_KINDS`: a failed `thread` or `quoted_tweet` is not a link failure,
    and reporting one as the reason a URL has no body would name a cause that belongs to a
    different source — the same distinction `executors.api._failure_clause` already draws.
    """
    return {
        entry.url: entry
        for entry in (item.content.sources if item.content else [])
        if isinstance(entry, ContentSourceFailure) and entry.kind in LINK_CONTENT_KINDS
    }


def _unfetched_link(url: str, failure: ContentSourceFailure | None) -> UnfetchedLink:
    """One URL with no body, carrying its RECORDED reason or none at all.

    With no failure record there was no attempt, so there is no cause to name — and naming
    one would be the exact sin the note exists to forbid. `executors.api._failure_clause`
    already refuses to invent a cause it did not measure; `detail` reuses that module's
    `_FAILURE_CLAUSE` wording so the phrasing the consumer sees cannot drift from the
    phrasing the generators and the judge are given.
    """
    if failure is None:
        return UnfetchedLink(url=url, reason="not_attempted")
    return UnfetchedLink(
        url=url,
        reason=UNFETCHED_REASON_BY_FAILURE[failure.failure_reason],
        detail=_FAILURE_CLAUSE.get(failure.failure_reason),
    )


def _note_path(item: Item, vault_dir: Path | None) -> str | None:
    """The item's note inside the vault, or None when it has not been generated.

    Returns a path only for a note that EXISTS: a path to a note nobody generated is a
    broken promise to the consumer. Containment is checked with `is_relative_to` after
    `resolve()`, not by rejecting `..` — spec §10.6 asks that locators not escape the
    configured roots, and rejecting `..` does not prove containment: a symlink inside the
    vault passes that filter and still points outside.
    """
    if vault_dir is None:
        return None
    relative = Path("items") / note_filename(item)
    candidate = (vault_dir / relative).resolve()
    root = vault_dir.resolve()
    if not candidate.is_relative_to(root) or not candidate.exists():
        return None
    return relative.as_posix()


def knowledge_item(item: Item, *, vault_dir: Path | None = None) -> KnowledgeItem:
    """The read projection of one item (spec §3.2).

    `content_kinds` lists the kinds with a READABLE body — a failed fetch appears in
    `failed_sources` instead, because listing its kind as available would tell a consumer
    to ask `get` for a body that does not exist.
    """
    surfaces = item_surfaces(item)
    kinds = tuple(
        dict.fromkeys(
            source.kind
            for _i, source in iter_content_sources(item, set(CONTENT_KIND_TO_SURFACE_TYPES))
        )
    )
    return KnowledgeItem(
        item_id=item.id,
        source=item.source,
        url=item.url,
        author=item.author,
        created_at=item.created_at,
        captured_at=item.captured_at,
        primary_topic=item.enriched.primary_topic if item.enriched else None,
        topics=item_topics(item),
        available_surfaces=tuple(dict.fromkeys(s.surface_type for s in surfaces)),
        content_kinds=kinds,
        failed_sources=_failed_sources(item),
        unfetched_links=_unfetched_links(item),
        note_path=_note_path(item, vault_dir),
        bookmark_folder=item.bookmark_folder,
    )


def hydrate_verification(item: Item, language: str) -> dict[str, VerificationVerdict]:
    """The item's verdicts that are STILL CURRENT, read from the live store (M5).

    Not a field on `KnowledgeSurface`, and not a column the index may cache. A stored copy
    of a verdict cannot be invalidated when the verdict changes: `surface_fingerprint`
    hashes (version, surface type, origin, text) and does not depend on the verdict at all,
    so a FAIL revoked by `verify --audit` would keep being served as the PASS it used to be
    — CLAUDE.md rule 6 run backwards.

    The freshness check is `verification.verdict_is_current`, the SAME one
    `generate._verdict_badge` applies. A stale verdict is omitted entirely rather than
    downgraded, because "we have no current verdict" is true and "REVIEW" would not be.
    """
    from xbrain.verification import ALL_TARGETS, verdict_is_current

    return {
        target: item.verification[target]
        for target in ALL_TARGETS
        if target in item.verification and verdict_is_current(item, target, language)
    }


def topic_surfaces(topic: Topic, page: TopicPage | None) -> tuple[KnowledgeSurface, ...]:
    """The description, the overview and each note of one topic (spec §3.6).

    Three separate surfaces, not one blob, because they have different provenance: the
    description's producer is unrecorded (`unknown`), while the overview and the notes are
    known LLM synthesis. Collapsing them would erase exactly the distinction spec §3.6 asks
    the layer to preserve.
    """
    surfaces: list[KnowledgeSurface] = []
    if not _blank(topic.description):
        surfaces.append(
            _surface(
                owner_type="topic",
                owner_id=topic.slug,
                surface_type="topic_description",
                source_key=SINGLETON_SOURCE_KEY,
                text=topic.description,
                locator=Locator(kind="vocab"),
            )
        )
    if page is None:
        return tuple(surfaces)
    if not _blank(page.overview):
        surfaces.append(
            _surface(
                owner_type="topic",
                owner_id=topic.slug,
                surface_type="topic_overview",
                source_key=SINGLETON_SOURCE_KEY,
                text=page.overview,
                producer="topics",
                produced_at=page.synthesized_at,
                locator=Locator(kind="topic_page"),
            )
        )
    for note_index, note in enumerate(page.notes):
        if _blank(note):
            continue
        surfaces.append(
            _surface(
                owner_type="topic",
                owner_id=topic.slug,
                surface_type="topic_note",
                source_key=str(note_index),
                text=note,
                producer="topics",
                produced_at=page.synthesized_at,
                locator=Locator(kind="topic_page", note_index=note_index),
            )
        )
    return tuple(surfaces)


def topic_record(
    topic: Topic,
    page: TopicPage | None,
    primary_item_ids: tuple[str, ...],
    secondary_item_ids: tuple[str, ...],
) -> TopicRecord:
    """The common view of a topic (spec §3.6).

    `stale` is derived here from the live primary count against `post_count_at_synth` — the
    same derivation `topics.topics_needing_synth` performs. A topic with no page is stale by
    definition: an overview that was never written is missing, not empty, and telling a
    consumer otherwise would present an absence as a synthesis.
    """
    overview = page.overview if page else ""
    stale = page is None or len(primary_item_ids) != page.post_count_at_synth
    return TopicRecord(
        topic_id=topic_id(topic.slug),
        slug=topic.slug,
        # Each layer keeps ITS OWN provenance (spec §3.6). The vocabulary does not record
        # whether a description was written or generated, so it is `unknown` and fails closed
        # to synthesis; the overview and the notes are known LLM output. One `origin` for the
        # whole record would have to lie about one of them.
        description=DerivedText(text=topic.description, origin=SURFACE_ORIGIN["topic_description"]),
        overview=(
            DerivedText(text=overview, origin=SURFACE_ORIGIN["topic_overview"]) if page else None
        ),
        notes=(
            tuple(
                DerivedText(text=note, origin=SURFACE_ORIGIN["topic_note"]) for note in page.notes
            )
            if page
            else ()
        ),
        primary_item_ids=primary_item_ids,
        secondary_item_ids=secondary_item_ids,
        synthesized_at=page.synthesized_at if page else None,
        post_count_at_synth=page.post_count_at_synth if page else None,
        stale=stale,
        vocab_fingerprint=surface_fingerprint("topic_description", "unknown", topic.description),
        synthesis_fingerprint=(
            surface_fingerprint("topic_overview", "llm", overview) if page else None
        ),
    )
