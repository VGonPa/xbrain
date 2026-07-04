"""Parse an X long-form Article's GraphQL payload into ordered `ArticleBlock`s.

X serialises a long-form Article body as a Draft.js `ContentState`: an ordered
list of `blocks` (paragraphs, headings, list items, plus `atomic` blocks that
reference inline media) and an `entityMap` that resolves each media reference.
This module turns that payload into the ordered `list[ArticleBlock]` carried on
`ContentSourceSuccess.blocks` (#39 PR3): text runs become `ArticleTextBlock`
(with `## `/`- ` markdown prefixes baked in for headings/lists), inline images
become `ArticleImageBlock`s wrapping a `MediaPhotoPending` (so the existing
`xbrain media` engine downloads them later, PR4), IN DOCUMENT ORDER. The lead
`cover_media` image is prepended as the first block.

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

from xbrain.models import (
    ARTICLE_PARAGRAPH_SEP,
    ArticleBlock,
    ArticleImageBlock,
    ArticleTextBlock,
    MediaPhotoPending,
)

logger = logging.getLogger(__name__)

# Draft.js entity `type`s that denote an inline image (case-insensitive). A
# LINK / TWEET / MENTION entity is explicitly NOT one, so an inline-link
# paragraph keeps its text rather than being mistaken for an image.
_IMAGE_ENTITY_TYPES = frozenset({"IMAGE", "MEDIA"})
# Ordered key names that may carry an image's CDN URL directly on the entity /
# media-item `data` (the defensive/constructed shape; the live shape resolves
# via `media_entities` instead). Anchoring on the key name keeps it drift-tolerant.
_IMAGE_URL_KEYS = ("media_url_https", "mediaUrl", "media_url", "mediaURL", "url")
# Ordered key names that may carry an image's alt text.
_ALT_KEYS = ("altText", "alt_text", "alt", "description")

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
    _warn_if_media_unresolved(raw_blocks, container, media_index, blocks)
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
    """The `media_info.original_img_url` CDN URL on a media node, or None.

    Shared by the inline `media_entities[]` index and the `cover_media` reader.
    Null-safe: a missing `media_info` / URL (or a non-http value) yields None.
    """
    if not isinstance(node, dict):
        return None
    info = node.get("media_info")
    if isinstance(info, dict):
        url = info.get("original_img_url")
        if isinstance(url, str) and url.startswith("http"):
            return url
    return None


def _media_index(container: dict[str, Any] | None) -> dict[str, str]:
    """Map `str(media_id)` → CDN URL from the container's `media_entities[]`.

    A `MEDIA` entity carries only a `mediaId`; the CDN URL lives on the sibling
    `media_entities[]` array keyed by `media_id`. This builds that lookup so
    `_item_url` can turn a `mediaId` into a real image URL. Keys are stringified
    so an int `media_id` and a str `mediaId` still match. Null-safe: a missing /
    non-list `media_entities` yields an empty index.
    """
    index: dict[str, str] = {}
    if not isinstance(container, dict):
        return index
    entities = container.get("media_entities")
    if not isinstance(entities, list):
        return index
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        media_id = entity.get("media_id")
        url = _media_info_url(entity)
        if media_id is not None and url:
            index[str(media_id)] = url
    return index


def _build_blocks(
    raw_blocks: list[Any], entity_by_key: dict[str, Any], media_index: dict[str, str]
) -> list[ArticleBlock]:
    """Turn Draft.js blocks into ordered `ArticleBlock`s (images + text runs).

    A block that references an image entity is resolved to one or more
    `ArticleImageBlock`s (a `mediaItems` gallery yields one per resolvable item)
    and NEVER falls through to the text branch — so a caption-bearing atomic
    block whose media fails to resolve is logged as a drop rather than silently
    demoted to text. Headings/list items get their markdown prefix baked in.
    """
    blocks: list[ArticleBlock] = []
    have_text = False
    for raw in raw_blocks:
        if not isinstance(raw, dict):
            continue
        entity = _first_entity(raw, entity_by_key)
        if _is_image_entity(entity):
            images, unresolved = _resolve_image_blocks(entity, media_index)
            if images:
                blocks.extend(images)
                if unresolved:
                    _log_partial_gallery(len(images), unresolved)
            else:
                _log_dropped_block(raw, entity_by_key)
            continue
        text = raw.get("text")
        if isinstance(text, str) and text.strip():
            separator = ARTICLE_PARAGRAPH_SEP if have_text else ""
            blocks.append(ArticleTextBlock(text=separator + _block_prefix(raw.get("type")) + text))
            have_text = True
            continue
        _log_dropped_block(raw, entity_by_key)
    return blocks


def _is_image_entity(entity: dict[str, Any] | None) -> TypeGuard[dict[str, Any]]:
    """True when `entity` is an inline-image entity (`IMAGE`/`MEDIA` type).

    A `TypeGuard` so callers narrow `entity` to a non-optional dict afterwards.
    """
    return isinstance(entity, dict) and str(entity.get("type", "")).upper() in _IMAGE_ENTITY_TYPES


def _resolve_image_blocks(
    entity: dict[str, Any], media_index: dict[str, str]
) -> tuple[list[ArticleImageBlock], int]:
    """Resolve an image entity to `(image_blocks, unresolved_item_count)`.

    Two shapes, tried in order: (1) the REAL X gallery — `data.mediaItems[]`,
    each item resolved via `_item_url` (its `mediaId` looked up in `media_index`,
    or a URL stored on the item), yielding ONE `ArticleImageBlock` per resolvable
    item so a multi-image gallery is not truncated to its first image; a
    `mediaItems` present but partially/wholly unresolvable does NOT fall through
    to a stray `data`-level URL (which could be a click-through link) — it reports
    the miss instead. (2) the defensive shape — no `mediaItems`, so a single URL
    stored directly on the entity `data` (`media_url_https`, `mediaUrl`, …).
    """
    data = entity.get("data")
    if not isinstance(data, dict):
        return [], 0
    items = data.get("mediaItems")
    if isinstance(items, list) and items:
        images: list[ArticleImageBlock] = []
        unresolved = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            url = _item_url(item, media_index)
            if url:
                images.append(
                    ArticleImageBlock(
                        media=MediaPhotoPending(url=url), alt=_alt_text(item) or _alt_text(data)
                    )
                )
            else:
                unresolved += 1
        return images, unresolved
    url = _find_url_by_key(data)
    if url:
        return [ArticleImageBlock(media=MediaPhotoPending(url=url), alt=_alt_text(data))], 0
    return [], 0


def _item_url(item: dict[str, Any], media_index: dict[str, str]) -> str | None:
    """One `mediaItems[i]`'s CDN URL: its `mediaId` in `media_index`, else a URL
    stored directly on the item (the defensive/constructed shape)."""
    media_id = item.get("mediaId") or item.get("media_id")
    if media_id is not None:
        url = media_index.get(str(media_id))
        if url:
            return url
    return _find_url_by_key(item)


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
    url = _media_info_url(container.get("cover_media"))
    if not url:
        return blocks
    for block in blocks:
        if isinstance(block, ArticleImageBlock) and block.media.url == url:
            return blocks
    cover = ArticleImageBlock(media=MediaPhotoPending(url=url), alt=None)
    return [cover, *blocks]


def _warn_if_media_unresolved(
    raw_blocks: list[Any],
    container: dict[str, Any] | None,
    media_index: dict[str, str],
    blocks: list[ArticleBlock],
) -> None:
    """WARN when the payload carried media but the body resolved ZERO images.

    The original #39 defect was exactly this: media present (atomic blocks +
    `media_entities` + `cover_media`) yet every image silently dropped. A
    genuinely text-only article (no atomic blocks, no media siblings) is NOT
    flagged — so this fires only on a real media-resolution drift, not on every
    prose-only piece.
    """
    if any(isinstance(b, ArticleImageBlock) for b in blocks):
        return
    had_atomic = any(isinstance(r, dict) and r.get("type") == "atomic" for r in raw_blocks)
    cover = container.get("cover_media") if isinstance(container, dict) else None
    if had_atomic or media_index or _media_info_url(cover):
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
