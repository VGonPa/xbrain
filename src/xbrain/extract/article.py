"""Parse an X long-form Article's GraphQL payload into ordered `ArticleBlock`s.

X serialises a long-form Article body as a Draft.js `ContentState`: an ordered
list of `blocks` (paragraphs, plus `atomic` blocks that reference inline media)
and an `entityMap` that resolves each media reference. This module turns that
payload into the ordered `list[ArticleBlock]` carried on
`ContentSourceSuccess.blocks` (#39 PR3): text runs become `ArticleTextBlock`,
inline images become `ArticleImageBlock` wrapping a `MediaPhotoPending` (so the
existing `xbrain media` engine downloads them later, PR4), IN DOCUMENT ORDER.

FIXTURE PROVENANCE / RESILIENCE: the exact key path is pinned against a
CONSTRUCTED fixture (see `tests/test_article.py` / `tests/test_fetch_x.py`), not
a recorded live payload — validate against a real bookmarked-Article GraphQL
response before production reliance (RFC #39 open-Q #4). The parser therefore
anchors ONLY on stable key names and degrades to `(None, [])` on ANY shape drift
so the caller (`fetch_x._fetch_rendered`) falls back to trafilatura text rather
than crash — never a partial/wrong block set masquerading as a complete body.

FLATTENED-BODY INVARIANT: the inter-paragraph separator (`\\n\\n`) is baked into
each non-first text run, so the source's flattened `text` is the EXACT
`"".join(b.text for b in blocks if isinstance(b, ArticleTextBlock))` (the PR1
contract) AND still reads naturally for `enrich`/`topics`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

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
# Ordered key names that may carry an image's CDN URL, at any nesting depth of
# the entity `data` (X shapes vary: a flat `url`, a `mediaUrl`, or a nested
# `mediaItems[]`). Anchoring on the key name keeps the resolver drift-tolerant.
_IMAGE_URL_KEYS = ("media_url_https", "mediaUrl", "media_url", "mediaURL", "url")
# Ordered key names that may carry an image's alt text (top-level of `data`).
_ALT_KEYS = ("altText", "alt_text", "alt", "description")


def parse_article_content_state(payload: Any) -> tuple[str | None, list[ArticleBlock]]:
    """Map an X article GraphQL `payload` to `(title, ordered_blocks)`.

    Returns `(None, [])` when no usable `content_state` is found (missing /
    renamed / malformed) — the caller then routes to the trafilatura fallback.
    `title` may be `None` even when blocks are found (a title-less shape still
    yields a body).
    """
    content_state = _find_content_state(payload)
    if content_state is None:
        return None, []
    raw_blocks = content_state.get("blocks")
    if not isinstance(raw_blocks, list):
        return None, []
    entity_map = _entity_map(content_state)
    blocks = _build_blocks(raw_blocks, entity_map)
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
    """Parse a JSON string, or None on failure (never raises)."""
    try:
        return json.loads(value)
    except (ValueError, TypeError):
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


def _find_content_state(node: Any) -> dict[str, Any] | None:
    """Locate the content_state dict anywhere in `node` (BFS, key-anchored).

    Prefers an explicit `content_state` / `contentState` key at any level, but
    also accepts a node that IS a content_state (the response being the body
    itself). Null-safe: a missing/renamed path degrades to None.
    """
    queue: list[Any] = [node]
    while queue:
        current = queue.pop(0)
        if isinstance(current, dict):
            for key in ("content_state", "contentState"):
                if key in current:
                    coerced = _coerce_content_state(current[key])
                    if coerced is not None:
                        return coerced
            coerced = _coerce_content_state(current)
            if coerced is not None:
                return coerced
            queue.extend(current.values())
        elif isinstance(current, list):
            queue.extend(current)
    return None


def _entity_map(content_state: dict[str, Any]) -> dict[str, Any]:
    for key in ("entityMap", "entity_map"):
        value = content_state.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _build_blocks(raw_blocks: list[Any], entity_map: dict[str, Any]) -> list[ArticleBlock]:
    """Turn Draft.js blocks into ordered `ArticleBlock`s (images + text runs)."""
    blocks: list[ArticleBlock] = []
    have_text = False
    for raw in raw_blocks:
        if not isinstance(raw, dict):
            continue
        image = _image_block(raw, entity_map)
        if image is not None:
            blocks.append(image)
            continue
        text = raw.get("text")
        if isinstance(text, str) and text.strip():
            separator = ARTICLE_PARAGRAPH_SEP if have_text else ""
            blocks.append(ArticleTextBlock(text=separator + text))
            have_text = True
            continue
        # Neither an image nor a text run: if it referenced an entity, it is a
        # DROPPED media/atomic block (an embed/divider, or an image whose URL did
        # not resolve). Log it so a real-payload key drift is visible rather than
        # silently losing content (data-safety observability, #39 PR3 review).
        _log_dropped_block(raw, entity_map)
    return blocks


def _log_dropped_block(raw: dict[str, Any], entity_map: dict[str, Any]) -> None:
    """WARN when a non-text block references an entity we could not render.

    A genuinely empty spacer block (no entity) is silent — only an entity-bearing
    block that produced no image is a real content drop worth surfacing.
    """
    entity = _first_entity(raw, entity_map)
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


def _image_block(raw: dict[str, Any], entity_map: dict[str, Any]) -> ArticleImageBlock | None:
    """An `ArticleImageBlock` when `raw` references an inline image, else None."""
    entity = _first_entity(raw, entity_map)
    if entity is None:
        return None
    if str(entity.get("type", "")).upper() not in _IMAGE_ENTITY_TYPES:
        return None
    data = entity.get("data")
    if not isinstance(data, dict):
        return None
    url = _find_url_by_key(data)
    if not url:
        return None
    return ArticleImageBlock(media=MediaPhotoPending(url=url), alt=_alt_text(data))


def _first_entity(raw: dict[str, Any], entity_map: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve the first entity referenced by `raw`'s entityRanges, or None."""
    ranges = raw.get("entityRanges") or raw.get("entity_ranges")
    if not isinstance(ranges, list) or not ranges:
        return None
    first = ranges[0]
    if not isinstance(first, dict) or first.get("key") is None:
        return None
    entity = entity_map.get(str(first["key"]))
    return entity if isinstance(entity, dict) else None


def _find_url_by_key(node: Any) -> str | None:
    """The image CDN URL, preferring the canonical key GLOBALLY.

    Searches the whole entity `data` tree once per key in `_IMAGE_URL_KEYS`
    priority order, so a deep `media_url_https` beats a shallow bare `url`
    (a bare `url` may be a link/thumbnail; `media_url_https` is the canonical
    full-size CDN photo PR4's size-cascade wants).
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
