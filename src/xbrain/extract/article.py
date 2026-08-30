"""Parse an X long-form Article's GraphQL payload into ordered `ArticleBlock`s.

X serialises a long-form Article body as a Draft.js `ContentState`: an ordered
list of `blocks` (paragraphs, headings, list items, plus `atomic` blocks that
reference inline media) and an `entityMap` that resolves each media reference.
This module turns that payload into the ordered `list[ArticleBlock]` carried on
`ContentSourceSuccess.blocks` (#39 PR3): text runs become `ArticleTextBlock`
(with `## `/`- ` markdown prefixes baked in for headings/lists), inline images
become `ArticleImageBlock`s wrapping a `MediaPhotoPending` (so the existing
`xbrain media` engine downloads them later, PR4), and inline VIDEO becomes an
`ArticleVideoBlock` wrapping a `MediaVideoPending`, IN DOCUMENT ORDER. The lead
`cover_media` (image or video) is prepended as the first block.

NOTHING IS DROPPED SILENTLY. Two whole classes of content used to be, and both
were invisible in the note rather than reported — measured over 23 real Articles:

  - **video**, because `MEDIA` was assumed to mean photo. An `ApiVideo` keeps its
    poster one level deeper (`media_info.preview_image.original_img_url`) and its
    bitrate under `bit_rate`, not the `bitrate` a tweet uses — so the lookup
    returned nothing and the block went to the drop log.
  - **entity-borne text**: 20 `MARKDOWN` (whole code listings), 48 `DIVIDER`
    (section rules), 1 `TWEET` (an embedded post). Their block is `atomic` with
    `text: " "`, which fails `.strip()`, so the "no text run ⇒ no content"
    assumption deleted them. `_entity_text` recovers these, and sweeps unknown
    entity types carrying `markdown`/`text`/`html` for the same reason the wall
    detector over-rejects: surfacing something skippable costs a line, dropping
    content is permanent and invisible.

VALIDATION / RESILIENCE: the key path is validated against three REAL captured
bookmarked-Article GraphQL payloads (`tests/test_article_real.py` +
`tests/fixtures/art-*.json`, #66). On the live shape the `entityMap` is a LIST
keyed by `entry.key`, a `MEDIA` entity resolves its CDN URL INDIRECTLY via a
sibling `media_entities[]` (`mediaItems[].mediaId` -> `media_id` ->
`media_info.original_img_url`), and the lead image lives in a separate
`cover_media` sibling. The parser anchors ONLY on stable key names and degrades
safely: a partial shape miss yields an image-less but text-complete body (with a
WARN), a wholesale miss yields `(None, [])`, so the caller
(`fetch_x._fetch_rendered`) falls back to trafilatura rather than crash — never a
partial/wrong block set masquerading as a complete body. The older CONSTRUCTED
shape (dict `entityMap`, URL directly on the entity `data`) is retained as a
defensive path (`tests/test_article.py`).

FLATTENED-BODY INVARIANT: the inter-paragraph separator (`\\n\\n`) is baked into
each non-first text run (after any `## `/`- ` prefix), so the source's flattened
`text` is the EXACT `"".join(b.text for b in blocks if isinstance(b,
ArticleTextBlock))` (the PR1 contract) AND still reads naturally for
`enrich`/`topics`.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeGuard

from xbrain.extract.video import build_video_media
from xbrain.models import (
    ARTICLE_PARAGRAPH_SEP,
    ArticleBlock,
    ArticleImageBlock,
    ArticleTextBlock,
    ArticleVideoBlock,
    MediaPhotoPending,
    MediaVideoPending,
)

# The two pending media states an article body can carry. `_media_index` resolves
# a `mediaId` to one of them, and `_media_block` wraps it in the matching block
# variant — so "is this a photo or a video" is decided ONCE, at the index, from
# the payload's own shape, not re-guessed at every use site.
MediaPending = MediaPhotoPending | MediaVideoPending

logger = logging.getLogger(__name__)

# Draft.js entity `type`s that denote an inline image (case-insensitive). A
# LINK / TWEET / MENTION entity is explicitly NOT one, so an inline-link
# paragraph keeps its text rather than being mistaken for an image.
_MEDIA_ENTITY_TYPES = frozenset({"IMAGE", "MEDIA"})
# Ordered key names that may carry an image's CDN URL directly on the entity /
# media-item `data` (the defensive/constructed shape; the live shape resolves
# via `media_entities` instead). Anchoring on the key name keeps it drift-tolerant.
_IMAGE_URL_KEYS = ("media_url_https", "mediaUrl", "media_url", "mediaURL", "url")
# Ordered key names that may carry an image's alt text.
_ALT_KEYS = ("altText", "alt_text", "alt", "description")
# Ordered key names under an entity's `data` that may carry its TEXT, for a block
# whose own `text` run is empty (see `_entity_text`). `markdown` first: it is the
# real key on the live shape, and the others are drift tolerance.
_ENTITY_TEXT_KEYS = ("markdown", "text", "html")

# Draft.js block `type`s whose text run carries a markdown prefix, baked into
# the run AFTER the `\n\n` separator so the flattened-text invariant still holds
# (`generate` strips only the leading separator, leaving the prefix to render).
_BLOCK_PREFIXES = {
    "header-one": "# ",
    "header-two": "## ",
    "header-three": "### ",
    "header-four": "#### ",
    "unordered-list-item": "- ",
    "ordered-list-item": "1. ",
    "blockquote": "> ",
}


def parse_article_content_state(payload: Any) -> tuple[str | None, list[ArticleBlock]]:
    """Map an X article GraphQL `payload` to `(title, ordered_blocks)`.

    Returns `(None, [])` when no usable `content_state` is found (missing /
    renamed / malformed) — the caller then routes to the trafilatura fallback.
    `title` may be `None` even when blocks are found (a title-less shape still
    yields a body). On a partial media-shape miss the body is returned WITH its
    text runs but WITHOUT the unresolved images (a WARN is logged), never a crash.
    """
    container, content_state = _find_article_container(payload)
    if content_state is None:
        return None, []
    raw_blocks = content_state.get("blocks")
    if not isinstance(raw_blocks, list):
        return None, []
    entity_by_key = _entity_by_key(content_state)
    media_index = _media_index(container)
    blocks = _build_blocks(raw_blocks, entity_by_key, media_index)
    blocks = _prepend_cover(blocks, container)
    _warn_if_media_unresolved(raw_blocks, entity_by_key, container, media_index, blocks)
    return _find_title(payload), blocks


def _coerce_content_state(value: Any) -> dict[str, Any] | None:
    """Return `value` as a Draft.js content_state dict, or None.

    Accepts either a dict or a JSON-encoded string (X commonly serialises
    content_state as a string on the wire). A dict qualifies only when it
    carries a `blocks` list with at least one Draft.js-looking entry — a strong,
    drift-tolerant signal that avoids mistaking an unrelated nested `blocks` key
    for the article body.
    """
    if isinstance(value, str):
        value = _load_json(value)
    if not isinstance(value, dict):
        return None
    return value if _looks_like_draftjs_blocks(value.get("blocks")) else None


def _load_json(value: str) -> Any:
    """Parse a JSON string, or None on failure (never raises).

    A content_state that arrives as a string but does not parse is a real
    serialization drift; log it at DEBUG so the specific "present but
    unparseable" case is diagnosable rather than indistinguishable from absent.
    """
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        logger.debug("article: a content_state string was not valid JSON; treating as absent")
        return None


def _looks_like_draftjs_blocks(blocks: Any) -> bool:
    """True when `blocks` is a non-empty list with ≥1 Draft.js-looking block.

    We do NOT require the FIRST entry to be valid — a stray garbage entry up
    front must not reject an otherwise-valid body — but at least one real block
    (`type`/`text` dict) must be present to gate against an unrelated `blocks`
    key masquerading as the article content_state.
    """
    if not isinstance(blocks, list) or not blocks:
        return False
    return any(isinstance(b, dict) and ("type" in b or "text" in b) for b in blocks)


def _find_article_container(node: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Locate the article container + its content_state anywhere in `node` (BFS).

    Returns `(container, content_state)` where `container` is the dict that HOLDS
    the `content_state` — on the real X shape it is the `article_results.result`
    node, so its sibling `media_entities` (inline-image CDN URLs) and
    `cover_media` (the lead image) are readable off it. When the response IS the
    content_state itself (a title-less body passed directly), the container is
    that same dict — its media siblings are simply absent (null-safe reads).

    Prefers an explicit `content_state` / `contentState` key at any level. Both
    elements are `None` on a missing/renamed path (degrade to the fallback).
    """
    queue: list[Any] = [node]
    while queue:
        current = queue.pop(0)
        if isinstance(current, dict):
            for key in ("content_state", "contentState"):
                if key in current:
                    coerced = _coerce_content_state(current[key])
                    if coerced is not None:
                        return current, coerced
            coerced = _coerce_content_state(current)
            if coerced is not None:
                return current, coerced
            queue.extend(current.values())
        elif isinstance(current, list):
            queue.extend(current)
    return None, None


def _entity_by_key(content_state: dict[str, Any]) -> dict[str, Any]:
    """Index the Draft.js `entityMap` by entity key, LIST or dict shape.

    The REAL X shape is a LIST of `{"key": <int|str>, "value": {type, data}}`;
    a block's `entityRanges[0].key` matches an element's `key` (NOT its list
    index), so we key by `str(entry["key"])`. The older CONSTRUCTED shape is a
    plain `{key: value}` dict — accepted verbatim as the defensive path. Any
    other shape yields an empty map (every atomic block then resolves to no
    image, and the body degrades to text).
    """
    raw = content_state.get("entityMap")
    if raw is None:
        raw = content_state.get("entity_map")
    if isinstance(raw, list):
        indexed: dict[str, Any] = {}
        for entry in raw:
            if isinstance(entry, dict) and "key" in entry and isinstance(entry.get("value"), dict):
                indexed[str(entry["key"])] = entry["value"]
        return indexed
    if isinstance(raw, dict):
        return raw
    return {}


def _media_info_url(node: Any) -> str | None:
    """The still-image CDN URL on a media node, or None.

    Two real shapes: a PHOTO stores it at `media_info.original_img_url`; a VIDEO
    (`__typename: "ApiVideo"`) has no such key and stores its poster one level
    deeper, at `media_info.preview_image.original_img_url`. Reading only the
    first is what dropped every embedded video from an article body — the
    lookup returned None, the entity resolved to nothing, and the block was
    logged as a drop.

    Shared by the inline `media_entities[]` index and the `cover_media` reader.
    Null-safe: a missing `media_info` / URL (or a non-http value) yields None.
    """
    if not isinstance(node, dict):
        return None
    info = node.get("media_info")
    if not isinstance(info, dict):
        return None
    for holder in (info, info.get("preview_image")):
        if isinstance(holder, dict):
            url = holder.get("original_img_url")
            if isinstance(url, str) and url.startswith("http"):
                return url
    return None


def _video_media(node: Any) -> MediaVideoPending | None:
    """Build a playable `MediaVideoPending` from a `media_entities[]` VIDEO entry.

    An article video carries the same `variants` array as a tweet video, just
    under `media_info` instead of `video_info` — so the entry is re-keyed and
    handed to the SHARED `build_video_media`, keeping the "highest-bitrate mp4,
    else the HLS manifest" rule in one place rather than reimplementing it here
    (that centralisation is `extract.video`'s stated reason to exist).

    Returns None for a photo (no `variants`), so the caller can fall through to
    the image path. An EMPTY `variants` list counts as "not a video" for the same
    reason: `build_video_media` falls back to `media_url_https` when it finds no
    playable stream, and here that key holds the POSTER — so a variant-less entry
    would yield a "video" whose url is a JPEG, rendered as a `🎥 Ver vídeo` link
    to a still. Falling through to the image path is both truthful and useful.
    """
    if not isinstance(node, dict):
        return None
    info = node.get("media_info")
    if not isinstance(info, dict) or not info.get("variants"):
        return None
    if not isinstance(info["variants"], list):
        return None
    return build_video_media(
        {
            "video_info": {**info, "variants": [_normalise_variant(v) for v in info["variants"]]},
            "media_url_https": _media_info_url(node),
        },
    )


def _normalise_variant(variant: Any) -> Any:
    """Rename an article variant's `bit_rate` to the `bitrate` `select_variant` reads.

    The two X shapes disagree on ONE character: a tweet's
    `video_info.variants[]` uses `bitrate`, an article's `media_info.variants[]`
    uses `bit_rate`. Without this, every article video silently resolved to the
    FIRST mp4 in the list (measured: a 480x270 stream where a 1920x1080 one was
    offered) because `max(..., key=lambda v: v.get("bitrate") or 0)` compared a
    list of zeros. It kept working, just badly — which is why it needed a real
    payload to catch.
    """
    if not isinstance(variant, dict) or "bitrate" in variant or "bit_rate" not in variant:
        return variant
    renamed = {k: v for k, v in variant.items() if k != "bit_rate"}
    renamed["bitrate"] = variant["bit_rate"]
    return renamed


def _media_index(container: dict[str, Any] | None) -> dict[str, MediaPending]:
    """Map `str(media_id)` → the resolved media from the container's `media_entities[]`.

    A `MEDIA` entity carries only a `mediaId`; the payload lives on the sibling
    `media_entities[]` array keyed by `media_id`. This builds that lookup so
    `_item_media` can turn a `mediaId` into a real photo OR video. Keys are
    stringified so an int `media_id` and a str `mediaId` still match. Video is
    checked FIRST: a video entry also has a poster image, and resolving it as a
    photo would silently demote a demo clip to a still frame. Null-safe: a
    missing / non-list `media_entities` yields an empty index.
    """
    index: dict[str, MediaPending] = {}
    if not isinstance(container, dict):
        return index
    entities = container.get("media_entities")
    if not isinstance(entities, list):
        return index
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        media_id = entity.get("media_id")
        if media_id is None:
            continue
        video = _video_media(entity)
        if video is not None:
            index[str(media_id)] = video
            continue
        url = _media_info_url(entity)
        if url:
            index[str(media_id)] = MediaPhotoPending(url=url)
    return index


def _build_blocks(
    raw_blocks: list[Any], entity_by_key: dict[str, Any], media_index: dict[str, MediaPending]
) -> list[ArticleBlock]:
    """Turn Draft.js blocks into ordered `ArticleBlock`s (media + text runs).

    A block that references a media entity is resolved to one or more image /
    video blocks (a `mediaItems` gallery yields one per resolvable item) and
    NEVER falls through to the text branch — so a caption-bearing atomic block
    whose media fails to resolve is logged as a drop rather than silently demoted
    to text. Headings/list items get their markdown prefix baked in.

    A block whose text lives on its ENTITY rather than in its `text` run
    (`MARKDOWN`, `DIVIDER`, `TWEET` — see `_entity_text`) is recovered before the
    empty-`text` drop: those are real prose, code blocks and embedded tweets, and
    treating "no text run" as "no content" is what deleted them from the body.
    """
    blocks: list[ArticleBlock] = []
    have_text = False

    def _append_text(value: str) -> None:
        nonlocal have_text
        separator = ARTICLE_PARAGRAPH_SEP if have_text else ""
        blocks.append(ArticleTextBlock(text=separator + value))
        have_text = True

    for raw in raw_blocks:
        if not isinstance(raw, dict):
            continue
        entity = _first_entity(raw, entity_by_key)
        if _is_media_entity(entity):
            media, unresolved = _resolve_media_blocks(entity, media_index)
            if media:
                blocks.extend(media)
                if unresolved:
                    _log_partial_gallery(len(media), unresolved)
            else:
                _log_dropped_block(raw, entity_by_key)
            continue
        text = raw.get("text")
        if isinstance(text, str) and text.strip():
            _append_text(_block_prefix(raw.get("type")) + text)
            continue
        recovered = _entity_text(entity)
        if recovered:
            _append_text(recovered)
            continue
        _log_dropped_block(raw, entity_by_key)
    return blocks


def _is_media_entity(entity: dict[str, Any] | None) -> TypeGuard[dict[str, Any]]:
    """True when `entity` is an inline-media entity (`IMAGE`/`MEDIA` type).

    A `TypeGuard` so callers narrow `entity` to a non-optional dict afterwards.
    """
    return isinstance(entity, dict) and str(entity.get("type", "")).upper() in _MEDIA_ENTITY_TYPES


def _media_block(media: MediaPending, alt: str | None) -> ArticleBlock:
    """Wrap a resolved photo/video in its matching block variant."""
    if isinstance(media, MediaVideoPending):
        return ArticleVideoBlock(media=media, alt=alt)
    return ArticleImageBlock(media=media, alt=alt)


def _resolve_media_blocks(
    entity: dict[str, Any], media_index: dict[str, MediaPending]
) -> tuple[list[ArticleBlock], int]:
    """Resolve a media entity to `(media_blocks, unresolved_item_count)`.

    Two shapes, tried in order: (1) the REAL X gallery — `data.mediaItems[]`,
    each item resolved via `_item_media` (its `mediaId` looked up in
    `media_index`, or a URL stored on the item), yielding ONE block per
    resolvable item so a multi-image gallery is not truncated to its first
    image; a `mediaItems` present but partially/wholly unresolvable does NOT fall
    through to a stray `data`-level URL (which could be a click-through link) —
    it reports the miss instead. (2) the defensive shape — no `mediaItems`, so a
    single URL stored directly on the entity `data` (`media_url_https`,
    `mediaUrl`, …), which is always a photo (a video is never expressed that way).
    """
    data = entity.get("data")
    if not isinstance(data, dict):
        return [], 0
    items = data.get("mediaItems")
    if isinstance(items, list) and items:
        blocks: list[ArticleBlock] = []
        unresolved = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            media = _item_media(item, media_index)
            if media is not None:
                blocks.append(_media_block(media, _alt_text(item) or _alt_text(data)))
            else:
                unresolved += 1
        return blocks, unresolved
    url = _find_url_by_key(data)
    if url:
        return [ArticleImageBlock(media=MediaPhotoPending(url=url), alt=_alt_text(data))], 0
    return [], 0


def _item_media(item: dict[str, Any], media_index: dict[str, MediaPending]) -> MediaPending | None:
    """One `mediaItems[i]`'s media: its `mediaId` in `media_index`, else a URL
    stored directly on the item (the defensive/constructed shape, always a photo)."""
    media_id = item.get("mediaId") or item.get("media_id")
    if media_id is not None:
        media = media_index.get(str(media_id))
        if media is not None:
            return media
    url = _find_url_by_key(item)
    return MediaPhotoPending(url=url) if url else None


def _entity_text(entity: dict[str, Any] | None) -> str | None:
    """Recover the text an entity carries when its BLOCK has no text run, else None.

    X puts several kinds of real content on the entity rather than in the Draft.js
    `text`; the referencing block is `atomic` with `text: " "`, which fails the
    `text.strip()` check and used to fall straight into the drop log. Measured on
    the real corpus over 23 Articles: 20 `MARKDOWN`, 1 `TWEET`, 48 `DIVIDER`.

      - `MARKDOWN` → `data.markdown`, a fenced code block. Whole listings were
        being deleted from articles that are largely ABOUT their code.
      - `TWEET` → `data.tweetId`, an embedded post. Only the pointer is in this
        payload (the quoted body is not), so the canonical URL is what there is
        to keep — and `x.com/i/status/<id>` is the same handle-less form
        `extract.graphql` already uses.
      - `DIVIDER` → an empty `data`: a horizontal rule, i.e. the author's own
        section break. `---` reproduces it exactly.

    The trailing generic sweep is deliberate drift-tolerance: an entity type we
    have never seen that carries `markdown`/`text`/`html` is recovered instead of
    dropped. The asymmetry is the same one that governs the wall detector —
    surfacing content we could have skipped costs a stray line; dropping content
    is invisible and permanent. Anything still unrecovered reaches
    `_log_dropped_block`, so a genuinely new shape is reported, never silent.
    """
    if not isinstance(entity, dict):
        return None
    entity_type = str(entity.get("type", "")).upper()
    data = entity.get("data")
    data = data if isinstance(data, dict) else {}
    if entity_type == "DIVIDER":
        return "---"
    if entity_type == "TWEET":
        tweet_id = data.get("tweetId") or data.get("tweet_id")
        return f"https://x.com/i/status/{tweet_id}" if tweet_id else None
    for key in _ENTITY_TEXT_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _block_prefix(block_type: Any) -> str:
    """The markdown prefix for a heading / list / quote block `type` (else "")."""
    if isinstance(block_type, str):
        return _BLOCK_PREFIXES.get(block_type, "")
    return ""


def _prepend_cover(
    blocks: list[ArticleBlock], container: dict[str, Any] | None
) -> list[ArticleBlock]:
    """Prepend the article's `cover_media` lead image as an `ArticleImageBlock`.

    The cover lives outside `content_state.blocks` entirely (a `cover_media`
    sibling), so it is added as the FIRST block — the lead image, before any text
    run. Null-safe: no cover (or no URL) leaves `blocks` untouched. Dedup: if the
    cover URL already appears inline, it is not emitted twice.
    """
    if not isinstance(container, dict):
        return blocks
    cover_media = container.get("cover_media")
    video = _video_media(cover_media)
    poster = _media_info_url(cover_media)
    cover: ArticleImageBlock | ArticleVideoBlock
    if video is not None:
        cover = ArticleVideoBlock(media=video, alt=None)
    elif poster:
        cover = ArticleImageBlock(media=MediaPhotoPending(url=poster), alt=None)
    else:
        return blocks
    # Dedup against the POSTER url for a video cover too: the same clip appearing
    # inline is the same lead media, however each occurrence resolved.
    identities = {poster, getattr(cover.media, "thumbnail_url", None), cover.media.url} - {None}
    for block in blocks:
        if isinstance(block, (ArticleImageBlock, ArticleVideoBlock)):
            existing = {getattr(block.media, "thumbnail_url", None), block.media.url}
            if existing & identities:
                return blocks
    return [cover, *blocks]


def _warn_if_media_unresolved(
    raw_blocks: list[Any],
    entity_by_key: dict[str, Any],
    container: dict[str, Any] | None,
    media_index: dict[str, MediaPending],
    blocks: list[ArticleBlock],
) -> None:
    """WARN when the payload carried media but the body resolved ZERO of it.

    The original #39 defect was exactly this: media present (atomic blocks +
    `media_entities` + `cover_media`) yet every image silently dropped. A
    genuinely text-only article (no atomic blocks, no media siblings) is NOT
    flagged — so this fires only on a real media-resolution drift, not on every
    prose-only piece.

    "Had media" is judged by the atomic blocks that reference a MEDIA/IMAGE
    ENTITY, not by the presence of any atomic block: `MARKDOWN`, `DIVIDER` and
    `TWEET` are atomic too, so an article that is all prose, code and section
    breaks used to trip this warning every single time — and a warning that
    cries wolf on ordinary input is one nobody reads when it is real.
    """
    if any(isinstance(b, (ArticleImageBlock, ArticleVideoBlock)) for b in blocks):
        return
    had_media_block = any(
        _is_media_entity(_first_entity(r, entity_by_key)) for r in raw_blocks if isinstance(r, dict)
    )
    cover = container.get("cover_media") if isinstance(container, dict) else None
    if had_media_block or media_index or _media_info_url(cover):
        logger.warning(
            "article: content_state has media indicators (atomic blocks / "
            "media_entities / cover_media) but resolved 0 images — media "
            "resolution may have drifted."
        )


def _log_partial_gallery(resolved: int, unresolved: int) -> None:
    """WARN that a multi-image gallery only partially resolved (some items lost)."""
    logger.warning(
        "article: a MEDIA block resolved %d image(s) but %d gallery item(s) had "
        "no URL — a media_entities key drift may be hiding images.",
        resolved,
        unresolved,
    )


def _log_dropped_block(raw: dict[str, Any], entity_by_key: dict[str, Any]) -> None:
    """WARN when a non-text block references an entity we could not render.

    A genuinely empty spacer block (no entity) is silent — only an entity-bearing
    block that produced no image is a real content drop worth surfacing.
    """
    entity = _first_entity(raw, entity_by_key)
    if entity is None:
        return
    data = entity.get("data")
    data_keys = sorted(data) if isinstance(data, dict) else type(data).__name__
    logger.warning(
        "article: dropped a non-text block (entity type=%r, data keys=%s) — no "
        "image URL resolved; a content_state key drift may be hiding an image.",
        entity.get("type"),
        data_keys,
    )


def _first_entity(raw: dict[str, Any], entity_by_key: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve the first entity referenced by `raw`'s entityRanges, or None."""
    ranges = raw.get("entityRanges") or raw.get("entity_ranges")
    if not isinstance(ranges, list) or not ranges:
        return None
    first = ranges[0]
    if not isinstance(first, dict) or first.get("key") is None:
        return None
    entity = entity_by_key.get(str(first["key"]))
    return entity if isinstance(entity, dict) else None


def _find_url_by_key(node: Any) -> str | None:
    """The image CDN URL, preferring the canonical key GLOBALLY.

    Searches the whole `node` tree once per key in `_IMAGE_URL_KEYS` priority
    order, so a deep `media_url_https` beats a shallow bare `url` (a bare `url`
    may be a link/thumbnail; `media_url_https` is the canonical full-size CDN
    photo PR4's size-cascade wants).
    """
    for key in _IMAGE_URL_KEYS:
        url = _first_http_value_for_key(node, key)
        if url:
            return url
    return None


def _first_http_value_for_key(node: Any, key: str) -> str | None:
    """First http(s) string stored under `key` anywhere in `node` (any depth)."""
    if isinstance(node, dict):
        value = node.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value
        for value in node.values():
            found = _first_http_value_for_key(value, key)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _first_http_value_for_key(value, key)
            if found:
                return found
    return None


def _alt_text(data: dict[str, Any]) -> str | None:
    """First non-empty alt-text string under `data`'s known alt keys, or None."""
    for key in _ALT_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _find_title(node: Any) -> str | None:
    """First non-blank `title` string in `node` (DFS, insertion order)."""
    if isinstance(node, dict):
        title = node.get("title")
        if isinstance(title, str) and title.strip():
            return title
        for value in node.values():
            found = _find_title(value)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_title(value)
            if found:
                return found
    return None
