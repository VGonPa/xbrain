# tests/test_article_real.py
"""Ground-truth tests for the X-article parser against REAL captured payloads.

Unlike `tests/test_article.py` (which pins CONSTRUCTED Draft.js shapes), every
fixture here is the `article_results.result` subtree trimmed VERBATIM from a real
bookmarked-Article GraphQL response (`payload[1]`), committed under
`tests/fixtures/art-*.json`. They are the ground truth for #39/#66: X's real
`entityMap` is a LIST keyed by `entry.key`, a `MEDIA` entity resolves its CDN URL
indirectly via `media_entities[].media_info.original_img_url`, and the lead image
lives in a separate `cover_media` sibling. The parser must resolve inline + cover
images here, render `## ` headings and `- ` bullets, and keep the flattened-text
invariant — degrading to text-only (never crashing) on any shape miss.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xbrain.extract.article import parse_article_content_state
from xbrain.models import ArticleImageBlock, ArticleTextBlock, ContentSourceSuccess

_FIXTURES = Path(__file__).parent / "fixtures"

# Ground-truth image URLs (from the real captured payloads).
_OPENWIKI_COVER = "https://pbs.twimg.com/media/HMKNwxAbUAEMrOF.jpg"
_OPENWIKI_INLINE = "https://pbs.twimg.com/media/HMKNQeJbMAA9ljZ.jpg"
_WIKI_MEMORY_COVER = "https://pbs.twimg.com/media/HMEX45LWoAA7btu.jpg"


def _load(name: str) -> dict[str, Any]:
    """The trimmed `article_results.result` subtree of a real captured Article."""
    return json.loads((_FIXTURES / f"art-{name}.json").read_text(encoding="utf-8"))


def _image_urls(blocks: list) -> list[str]:
    """The `media.url` of every `ArticleImageBlock`, in document order."""
    return [b.media.url for b in blocks if isinstance(b, ArticleImageBlock)]


def test_openwiki_resolves_inline_media_from_entitymap_list():
    """art-OpenWiki: the atomic MEDIA block resolves to the inline CDN image.

    The entity is found in the LIST `entityMap` by `entry.key`, and its
    `mediaItems[0].mediaId` resolves against the sibling `media_entities[]`.
    The 7 LINK entities must NOT be mistaken for images.
    """
    _title, blocks = parse_article_content_state(_load("OpenWiki"))
    urls = _image_urls(blocks)
    assert _OPENWIKI_INLINE in urls
    # The inline image sits among the text runs, not at the very front (that is
    # the cover's slot), and is preceded + followed by text.
    inline_idx = next(
        i for i, b in enumerate(blocks) if isinstance(b, ArticleImageBlock) and b.media.url == _OPENWIKI_INLINE
    )
    assert any(isinstance(b, ArticleTextBlock) for b in blocks[:inline_idx])
    assert any(isinstance(b, ArticleTextBlock) for b in blocks[inline_idx + 1 :])
