# tests/test_article.py
"""Unit tests for the pure X-article content_state parser (#39 PR3).

FIXTURE PROVENANCE — READ THIS: every `content_state` payload below is
**CONSTRUCTED** from the documented Draft.js `ContentState` shape that X's
long-form Article editor serialises (ordered `blocks` + an `entityMap` of
inline media), NOT recorded from a live X GraphQL response. No real captured
article payload exists in the repo yet. The parser anchors ONLY on stable key
names (`content_state`/`contentState`, `blocks`, `entityMap`, `entityRanges`,
`type`, `text`) and degrades to `(None, [])` on any shape drift, so the fetch
stage routes to the trafilatura text fallback rather than crash. Pin these
shapes against a REAL bookmarked-Article payload before production reliance
(RFC #39 open-Q #4).
"""

from __future__ import annotations

import json

from xbrain.extract.article import parse_article_content_state
from xbrain.models import ArticleImageBlock, ArticleTextBlock, MediaPhotoPending

_IMAGE_URL = "https://pbs.twimg.com/media/ABC123.jpg"


def _content_state() -> dict:
    """A text → image → text article body (the canonical ordered case)."""
    return {
        "blocks": [
            {"key": "aaa", "text": "First paragraph.", "type": "unstyled", "entityRanges": []},
            {
                "key": "bbb",
                "text": " ",
                "type": "atomic",
                "entityRanges": [{"offset": 0, "length": 1, "key": 0}],
            },
            {"key": "ccc", "text": "Second paragraph.", "type": "unstyled", "entityRanges": []},
        ],
        "entityMap": {
            "0": {
                "type": "IMAGE",
                "mutability": "IMMUTABLE",
                "data": {"url": _IMAGE_URL, "altText": "a diagram"},
            }
        },
    }


def _payload(content_state, *, title: str = "The Long Read") -> dict:
    return {
        "data": {
            "article": {
                "article_results": {
                    "result": {
                        "__typename": "Article",
                        "rest_id": "1900000000000000000",
                        "title": title,
                        "content_state": content_state,
                    }
                }
            }
        }
    }


def test_parse_returns_ordered_text_image_text_blocks():
    title, blocks = parse_article_content_state(_payload(_content_state()))

    assert title == "The Long Read"
    assert len(blocks) == 3
    assert blocks[0] == ArticleTextBlock(text="First paragraph.")
    assert blocks[1] == ArticleImageBlock(media=MediaPhotoPending(url=_IMAGE_URL), alt="a diagram")
    # The inter-paragraph separator is baked into the (non-first) text run so
    # the flattened body is the exact "".join of the text-block texts.
    assert blocks[2] == ArticleTextBlock(text="\n\nSecond paragraph.")


def test_flattened_text_equals_concat_of_text_blocks():
    _title, blocks = parse_article_content_state(_payload(_content_state()))
    flattened = "".join(b.text for b in blocks if isinstance(b, ArticleTextBlock))
    assert flattened == "First paragraph.\n\nSecond paragraph."


def test_image_block_carries_pending_media_for_later_download():
    _title, blocks = parse_article_content_state(_payload(_content_state()))
    image = blocks[1]
    assert isinstance(image, ArticleImageBlock)
    # PR3 only ever emits a pending photo; PR4's engine advances it in place.
    assert isinstance(image.media, MediaPhotoPending)
    assert image.media.url == _IMAGE_URL


def test_content_state_accepts_a_json_encoded_string():
    # X commonly serialises content_state as a JSON string on the wire.
    payload = _payload(json.dumps(_content_state()))
    title, blocks = parse_article_content_state(payload)
    assert title == "The Long Read"
    assert [type(b).__name__ for b in blocks] == [
        "ArticleTextBlock",
        "ArticleImageBlock",
        "ArticleTextBlock",
    ]


def test_image_url_resolves_from_nested_media_items():
    nested_url = "https://pbs.twimg.com/media/NESTED.jpg"
    content_state = {
        "blocks": [
            {
                "key": "b",
                "text": " ",
                "type": "atomic",
                "entityRanges": [{"offset": 0, "length": 1, "key": 0}],
            }
        ],
        "entityMap": {
            "0": {
                "type": "MEDIA",
                "data": {
                    "mediaItems": [{"mediaId": "X", "mediaUrl": nested_url}],
                    "altText": "nested alt",
                },
            }
        },
    }
    _title, blocks = parse_article_content_state(_payload(content_state))
    assert len(blocks) == 1
    assert isinstance(blocks[0], ArticleImageBlock)
    assert blocks[0].media.url == nested_url
    assert blocks[0].alt == "nested alt"


def test_non_image_atomic_entity_is_skipped_and_inline_link_text_kept():
    content_state = {
        "blocks": [
            {
                "key": "embed",
                "text": " ",
                "type": "atomic",
                "entityRanges": [{"offset": 0, "length": 1, "key": 0}],
            },
            {
                "key": "link",
                "text": "see this link",
                "type": "unstyled",
                "entityRanges": [{"offset": 4, "length": 4, "key": 1}],
            },
        ],
        "entityMap": {
            "0": {"type": "TWEET", "data": {"tweetId": "123"}},
            "1": {"type": "LINK", "data": {"url": "https://example.com"}},
        },
    }
    _title, blocks = parse_article_content_state(_payload(content_state))
    # The embedded tweet is not an image (dropped); the link paragraph's text
    # is kept as a text run (a LINK entity is never mistaken for an image).
    assert blocks == [ArticleTextBlock(text="see this link")]


def test_image_only_article_yields_blocks_with_empty_flattened_text():
    content_state = {
        "blocks": [
            {
                "key": "b",
                "text": " ",
                "type": "atomic",
                "entityRanges": [{"offset": 0, "length": 1, "key": 0}],
            }
        ],
        "entityMap": {"0": {"type": "IMAGE", "data": {"url": _IMAGE_URL}}},
    }
    _title, blocks = parse_article_content_state(_payload(content_state))
    assert len(blocks) == 1
    assert isinstance(blocks[0], ArticleImageBlock)
    assert blocks[0].alt is None
    flattened = "".join(b.text for b in blocks if isinstance(b, ArticleTextBlock))
    assert flattened == ""


def test_missing_content_state_returns_empty():
    assert parse_article_content_state({"data": {"nothing": "here"}}) == (None, [])


def test_blank_payload_returns_empty():
    assert parse_article_content_state({}) == (None, [])
    assert parse_article_content_state("not a dict") == (None, [])


def test_blocks_not_a_list_degrades_to_empty():
    payload = _payload({"blocks": "not-a-list", "entityMap": {}})
    assert parse_article_content_state(payload) == (None, [])


def test_garbage_block_entries_are_skipped_without_crashing():
    content_state = {
        "blocks": [
            "not-a-dict",
            {"key": "ok", "text": "real text", "type": "unstyled"},
            {"no_text_key": True},
        ],
        "entityMap": {},
    }
    _title, blocks = parse_article_content_state(_payload(content_state))
    assert blocks == [ArticleTextBlock(text="real text")]


def test_atomic_image_with_missing_entity_is_dropped():
    # entityRanges points at a key absent from entityMap -> no image, no crash.
    content_state = {
        "blocks": [
            {
                "key": "b",
                "text": " ",
                "type": "atomic",
                "entityRanges": [{"offset": 0, "length": 1, "key": 9}],
            }
        ],
        "entityMap": {},
    }
    _title, blocks = parse_article_content_state(_payload(content_state))
    assert blocks == []


def test_content_state_passed_directly_without_wrapper():
    # Robust to the response being the content_state itself (no title around it).
    title, blocks = parse_article_content_state(_content_state())
    assert title is None
    assert [type(b).__name__ for b in blocks] == [
        "ArticleTextBlock",
        "ArticleImageBlock",
        "ArticleTextBlock",
    ]


def test_non_json_content_state_string_degrades_to_empty():
    # A content_state that is a string but not valid JSON degrades, no crash.
    assert parse_article_content_state(_payload("<<not json>>")) == (None, [])


def test_content_state_without_entity_map_key_parses_text():
    # No entityMap key at all -> the entity map is treated as empty; text runs
    # still parse (an atomic block simply resolves to no image).
    content_state = {"blocks": [{"key": "a", "text": "just text", "type": "unstyled"}]}
    _title, blocks = parse_article_content_state(_payload(content_state))
    assert blocks == [ArticleTextBlock(text="just text")]


def test_image_entity_with_non_dict_data_is_dropped():
    content_state = {
        "blocks": [
            {
                "key": "b",
                "text": " ",
                "type": "atomic",
                "entityRanges": [{"offset": 0, "length": 1, "key": 0}],
            }
        ],
        "entityMap": {"0": {"type": "IMAGE", "data": "not-a-dict"}},
    }
    assert parse_article_content_state(_payload(content_state)) == ("The Long Read", [])


def test_title_nested_inside_a_list_is_found():
    payload = {"items": [{"title": "Deep Title", "content_state": _content_state()}]}
    title, blocks = parse_article_content_state(payload)
    assert title == "Deep Title"
    assert len(blocks) == 3


def test_decoy_blocks_key_does_not_shadow_the_real_content_state():
    # A shallow, unrelated `blocks` key (NOT Draft.js-shaped) must not be
    # mistaken for the body — the real content_state (deeper) is still extracted.
    payload = {
        "data": {
            "blocks": ["just", "strings", 42],  # decoy: shallower, non-Draft.js
            "article": {
                "article_results": {"result": {"title": "Real", "content_state": _content_state()}}
            },
        }
    }
    title, blocks = parse_article_content_state(payload)
    assert title == "Real"
    assert [type(b).__name__ for b in blocks] == [
        "ArticleTextBlock",
        "ArticleImageBlock",
        "ArticleTextBlock",
    ]


def test_image_url_prefers_canonical_key_over_bare_url():
    # A canonical `media_url_https` anywhere in the entity data beats a shallow
    # bare `url` (which may be a link/thumbnail), so PR4's size cascade applies.
    canonical = "https://pbs.twimg.com/media/CANON.jpg"
    content_state = {
        "blocks": [
            {
                "key": "b",
                "text": " ",
                "type": "atomic",
                "entityRanges": [{"offset": 0, "length": 1, "key": 0}],
            }
        ],
        "entityMap": {
            "0": {
                "type": "IMAGE",
                "data": {
                    "url": "https://pbs.twimg.com/media/BARE.jpg",
                    "original": {"media_url_https": canonical},
                },
            }
        },
    }
    _title, blocks = parse_article_content_state(_payload(content_state))
    assert len(blocks) == 1
    assert isinstance(blocks[0], ArticleImageBlock)
    assert blocks[0].media.url == canonical


def test_dropped_media_block_is_logged(caplog):
    # An entity-bearing atomic block that resolves to no image is a real content
    # drop -> WARNING (a genuinely empty spacer stays silent).
    content_state = {
        "blocks": [
            {
                "key": "b",
                "text": " ",
                "type": "atomic",
                "entityRanges": [{"offset": 0, "length": 1, "key": 0}],
            }
        ],
        "entityMap": {"0": {"type": "IMAGE", "data": {"no_url_here": True}}},
    }
    with caplog.at_level("WARNING", logger="xbrain.extract.article"):
        _title, blocks = parse_article_content_state(_payload(content_state))
    assert blocks == []
    assert any("dropped a non-text block" in r.message for r in caplog.records)
