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
_PARAGRAPH_SEP = "\n\n"


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
    carries a non-empty `blocks` list whose first entry looks like a Draft.js
    block (`type`/`text`) — a strong, drift-tolerant signal that avoids
    mistaking an unrelated nested `blocks` key for the article body.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    if not isinstance(value, dict):
        return None
    blocks = value.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return None
    # At least one entry must look like a Draft.js block (a `type`/`text` dict).
    # We do NOT require the FIRST entry — a stray garbage entry up front must not
    # reject an otherwise-valid body — but at least one real block gates against
    # mistaking an unrelated `blocks` key for the article content_state.
    if any(isinstance(b, dict) and ("type" in b or "text" in b) for b in blocks):
        return value
    return None


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
            separator = _PARAGRAPH_SEP if have_text else ""
            blocks.append(ArticleTextBlock(text=separator + text))
            have_text = True
    return blocks


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
    """First http(s) string under an image-URL key, searched at any depth."""
    if isinstance(node, dict):
        for key in _IMAGE_URL_KEYS:
            value = node.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
        for value in node.values():
            found = _find_url_by_key(value)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_url_by_key(value)
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
