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

import pytest

from xbrain.extract.article import parse_article_content_state
from xbrain.fetch_x import _flatten_blocks
from xbrain.models import (
    ARTICLE_PARAGRAPH_SEP,
    ArticleImageBlock,
    ArticleTextBlock,
    ArticleVideoBlock,
    ContentSourceSuccess,
    MediaVideoPending,
)

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
        i
        for i, b in enumerate(blocks)
        if isinstance(b, ArticleImageBlock) and b.media.url == _OPENWIKI_INLINE
    )
    assert any(isinstance(b, ArticleTextBlock) for b in blocks[:inline_idx])
    assert any(isinstance(b, ArticleTextBlock) for b in blocks[inline_idx + 1 :])


# Ground-truth image inventory per real Article (cover first, then inline, in
# document order). This is the #66 acceptance table, locked as a test.
_EXPECTED_IMAGES = [
    ("OpenWiki", [_OPENWIKI_COVER, _OPENWIKI_INLINE]),
    ("Wiki_Memory", [_WIKI_MEMORY_COVER]),
    ("Headcount_AI", []),
]


@pytest.mark.parametrize("name, expected", _EXPECTED_IMAGES)
def test_real_payload_image_inventory(name: str, expected: list[str]) -> None:
    """Every real Article resolves EXACTLY its ground-truth images, in order."""
    _title, blocks = parse_article_content_state(_load(name))
    assert _image_urls(blocks) == expected


@pytest.mark.parametrize(
    "name, cover", [("OpenWiki", _OPENWIKI_COVER), ("Wiki_Memory", _WIKI_MEMORY_COVER)]
)
def test_cover_media_is_the_lead_block(name: str, cover: str) -> None:
    """When a `cover_media` exists, it is prepended as the FIRST block."""
    _title, blocks = parse_article_content_state(_load(name))
    assert isinstance(blocks[0], ArticleImageBlock)
    assert blocks[0].media.url == cover


def test_link_entities_never_become_images() -> None:
    """art-OpenWiki has 7 LINK entities + 1 MEDIA; only MEDIA + cover are images."""
    _title, blocks = parse_article_content_state(_load("OpenWiki"))
    assert len(_image_urls(blocks)) == 2


def test_headings_and_bullets_render_as_markdown() -> None:
    """`header-two` → `## ` and `unordered-list-item` → `- `, with the `\\n\\n`
    separator applied BEFORE the prefix (an independent order check)."""
    _title, blocks = parse_article_content_state(_load("Wiki_Memory"))
    texts = [b.text for b in blocks if isinstance(b, ArticleTextBlock)]
    # A non-first heading run is "\n\n## …"; a non-first bullet run is "\n\n- …".
    assert any(t.startswith(ARTICLE_PARAGRAPH_SEP + "## ") for t in texts)
    assert any(t.startswith(ARTICLE_PARAGRAPH_SEP + "- ") for t in texts)


@pytest.mark.parametrize("name", ["OpenWiki", "Wiki_Memory", "Headcount_AI"])
def test_text_run_separator_structure(name: str) -> None:
    """The FIRST text run never leads with the `\\n\\n` separator; every
    subsequent run does. An INDEPENDENT structural check (not a tautology that
    re-derives `text` from the same blocks): a broken separator would fail it.
    """
    _title, blocks = parse_article_content_state(_load(name))
    runs = [b.text for b in blocks if isinstance(b, ArticleTextBlock)]
    assert runs, "every real fixture carries prose"
    assert not runs[0].startswith(ARTICLE_PARAGRAPH_SEP)
    assert all(r.startswith(ARTICLE_PARAGRAPH_SEP) for r in runs[1:])


@pytest.mark.parametrize("name", ["OpenWiki", "Wiki_Memory", "Headcount_AI"])
def test_flattened_text_validates_against_model(name: str) -> None:
    """The source built the way `fetch_x` builds it (`text = _flatten_blocks`)
    satisfies the `_text_matches_blocks` model validator on real payloads."""
    _title, blocks = parse_article_content_state(_load(name))
    flat = _flatten_blocks(blocks)
    source = ContentSourceSuccess(
        kind="x_article",
        url="https://x.com/i/article/1",
        text=flat,
        blocks=blocks,
        http_status=200,
        attempts=1,
    )
    assert source.text == flat


@pytest.mark.parametrize(
    "name, title",
    [
        ("OpenWiki", "Introducing OpenWiki, an open source agent for repo documentation"),
        ("Wiki_Memory", "Wiki Memory"),
        ("Headcount_AI", "The case for headcount in the age of AI"),
    ],
)
def test_real_payload_title_is_extracted(name: str, title: str) -> None:
    """The article title is pulled from the real payload (flows to the source)."""
    got, _blocks = parse_article_content_state(_load(name))
    assert got == title


def test_cover_equal_to_an_inline_image_is_deduped() -> None:
    """When `cover_media` resolves to a URL already present inline, it is NOT
    emitted twice — the lead image renders once, not doubled."""
    payload = _load("OpenWiki")
    payload["cover_media"]["media_info"]["original_img_url"] = _OPENWIKI_INLINE
    _title, blocks = parse_article_content_state(payload)
    urls = _image_urls(blocks)
    assert urls == [_OPENWIKI_INLINE]  # single image, no doubled lead


def test_media_tripwire_warns_when_media_present_but_zero_images(caplog) -> None:
    """Media indicators present (atomic block) but 0 images resolved → WARN —
    the exact regression signal the original #39 defect lacked."""
    payload = _load("OpenWiki")
    payload.pop("media_entities", None)
    payload.pop("cover_media", None)
    with caplog.at_level("WARNING", logger="xbrain.extract.article"):
        _title, blocks = parse_article_content_state(payload)
    assert _image_urls(blocks) == []
    assert any("resolved 0 images" in r.message for r in caplog.records)


def test_text_only_article_does_not_trip_the_media_warning(caplog) -> None:
    """A genuinely text-only article (no atomic/media siblings) is NOT flagged."""
    with caplog.at_level("WARNING", logger="xbrain.extract.article"):
        _title, blocks = parse_article_content_state(_load("Headcount_AI"))
    assert _image_urls(blocks) == []
    assert not any("resolved 0 images" in r.message for r in caplog.records)


def test_missing_media_entities_drops_image_but_keeps_text() -> None:
    """No `media_entities`/`cover_media` → images unresolved, text still complete.

    The degrade-not-crash guarantee: a real body whose media siblings are absent
    (or renamed) yields zero images but loses no text run — never a crash.
    """
    payload = _load("OpenWiki")
    payload.pop("media_entities", None)
    payload.pop("cover_media", None)
    _title, blocks = parse_article_content_state(payload)
    assert _image_urls(blocks) == []
    assert any(isinstance(b, ArticleTextBlock) for b in blocks)


@pytest.mark.parametrize(
    "payload", [{}, {"foo": "bar"}, [], None, {"content_state": {"blocks": []}}]
)
def test_foreign_payload_degrades_to_none_empty(payload: Any) -> None:
    """Any non-Article / empty-body shape degrades to `(None, [])`, never a crash."""
    assert parse_article_content_state(payload) == (None, [])


# --------------------------------------------------------------------------
# Embedded VIDEO and entity-borne text — real payloads for the block-drop fix.
#
# `art-Video_Embed` is trimmed from an Article whose body embeds two native X
# videos; `art-Rich_Entities` from one whose body carries `MARKDOWN` code blocks,
# `DIVIDER` rules and `TWEET` embeds. Both used to lose that content entirely:
# the parser resolved a photo URL or nothing, and every non-photo atomic block
# went to the drop log.
# --------------------------------------------------------------------------


def test_video_embed_resolves_native_videos_not_their_posters() -> None:
    """An article `MEDIA` entity can be an `ApiVideo` — it must stay a video.

    Its poster lives one level deeper than a photo's URL
    (`media_info.preview_image.original_img_url`), which is why the lookup
    returned None and the block was dropped. Resolving it as a PHOTO would be
    just as wrong: a demo clip silently demoted to a still frame.
    """
    _title, blocks = parse_article_content_state(_load("Video_Embed"))
    videos = [b for b in blocks if isinstance(b, ArticleVideoBlock)]

    assert len(videos) == 2
    assert _image_urls(blocks) == []  # the posters must NOT become image blocks
    for video in videos:
        assert isinstance(video.media, MediaVideoPending)
        # The playable stream, never the poster.
        assert video.media.url.startswith("https://video.twimg.com/")
        assert video.media.url.endswith(".mp4") or ".mp4?" in video.media.url
        assert (video.media.thumbnail_url or "").startswith("https://pbs.twimg.com/")
        assert video.media.duration_millis


def test_video_embed_picks_the_highest_bitrate_variant() -> None:
    """X spells it `bit_rate` here and `bitrate` on a tweet — one character apart.

    Reading the tweet key against an article payload made every bitrate compare
    as 0, so `max()` returned the FIRST mp4: a 480x270 stream where a 1920x1080
    one was on offer. It kept working, just badly.
    """
    _title, blocks = parse_article_content_state(_load("Video_Embed"))
    videos = [b for b in blocks if isinstance(b, ArticleVideoBlock)]

    best = max(videos, key=lambda b: b.media.bitrate or 0)
    assert best.media.bitrate == 10368000
    assert "1920x1080" in best.media.url


def test_video_blocks_sit_in_document_order_between_text() -> None:
    """The clip stays where the author put it, not appended at the end."""
    _title, blocks = parse_article_content_state(_load("Video_Embed"))
    first_video = next(i for i, b in enumerate(blocks) if isinstance(b, ArticleVideoBlock))

    assert any(isinstance(b, ArticleTextBlock) for b in blocks[:first_video])
    assert any(isinstance(b, ArticleTextBlock) for b in blocks[first_video + 1 :])


def test_rich_entities_recovers_markdown_dividers_and_tweets() -> None:
    """Entity-borne content survives: code blocks, rules and embedded posts."""
    _title, blocks = parse_article_content_state(_load("Rich_Entities"))
    texts = [b.text for b in blocks if isinstance(b, ArticleTextBlock)]
    body = "".join(texts)

    # Three embedded posts, each kept as its canonical handle-less URL.
    assert body.count("https://x.com/i/status/") == 3
    # And the article's own prose is still there, in order, around them.
    assert texts[0].startswith('"¿Podría Madrid')
    assert len(texts) == len(blocks)  # this fixture is text-only once recovered


def test_recovered_blocks_keep_the_flattened_text_invariant() -> None:
    """`text` must still equal the concatenation of the text blocks.

    The recovery adds NEW `ArticleTextBlock`s mid-body, so the `\\n\\n` separator
    bookkeeping has to hold for them too — otherwise the flattened body that
    `enrich`/`topics` read drifts from what the note renders.
    """
    for name in ("Rich_Entities", "Video_Embed"):
        _title, blocks = parse_article_content_state(_load(name))
        source = ContentSourceSuccess(
            kind="x_article",
            url="https://x.com/i/article/1",
            text=_flatten_blocks(blocks),
            blocks=blocks,
            http_status=200,
            attempts=1,
        )
        assert source.text == "".join(
            b.text for b in source.blocks if isinstance(b, ArticleTextBlock)
        )
        # The first text run never carries a leading separator.
        first = next(b for b in blocks if isinstance(b, ArticleTextBlock))
        assert not first.text.startswith(ARTICLE_PARAGRAPH_SEP)


def test_entity_borne_text_does_not_trip_the_media_warning(caplog) -> None:
    """`MARKDOWN`/`DIVIDER`/`TWEET` blocks are atomic but are NOT media.

    Counting any atomic block as "had media" made every prose-and-code article
    log a media-drift warning, and a warning that cries wolf on ordinary input
    is one nobody reads when it is real.
    """
    with caplog.at_level("WARNING", logger="xbrain.extract.article"):
        _title, blocks = parse_article_content_state(_load("Rich_Entities"))
    assert blocks
    assert not any("resolved 0 images" in r.message for r in caplog.records)
