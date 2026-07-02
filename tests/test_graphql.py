# tests/test_graphql.py
import pytest

from xbrain.extract.graphql import parse_tweets

SAMPLE_RESPONSE = {
    "data": {
        "bookmark_timeline_v2": {
            "timeline": {
                "instructions": [
                    {
                        "type": "TimelineAddEntries",
                        "entries": [
                            {
                                "entryId": "tweet-111",
                                "content": {
                                    "itemContent": {
                                        "itemType": "TimelineTweet",
                                        "tweet_results": {
                                            "result": {
                                                "__typename": "Tweet",
                                                "rest_id": "111",
                                                "core": {
                                                    "user_results": {
                                                        "result": {
                                                            "legacy": {
                                                                "screen_name": "alice",
                                                                "name": "Alice",
                                                            }
                                                        }
                                                    }
                                                },
                                                "legacy": {
                                                    "full_text": "Great read https://t.co/abc",
                                                    "created_at": "Wed May 10 14:23:00 +0000 2026",
                                                    "entities": {
                                                        "urls": [
                                                            {
                                                                "expanded_url": "https://example.com/post",
                                                                "url": "https://t.co/abc",
                                                            }
                                                        ]
                                                    },
                                                },
                                            }
                                        },
                                    }
                                },
                            },
                            {
                                "entryId": "cursor-bottom-xyz",
                                "content": {"itemContent": {"itemType": "TimelineTimelineCursor"}},
                            },
                        ],
                    }
                ]
            }
        }
    }
}


def test_parse_tweets_extracts_timeline_tweet():
    items = parse_tweets(SAMPLE_RESPONSE, "bookmark")
    assert len(items) == 1
    item = items[0]
    assert item.id == "111"
    assert item.source == "bookmark"
    assert item.author.handle == "alice"
    assert item.text == "Great read https://t.co/abc"
    assert item.url == "https://x.com/alice/status/111"
    assert item.links[0].url == "https://example.com/post"
    assert item.links[0].domain == "example.com"
    assert item.created_at.year == 2026


def test_parse_tweets_ignores_non_tweet_entries():
    empty = {"data": {"timeline": {"instructions": []}}}
    assert parse_tweets(empty, "bookmark") == []


def test_parse_tweets_deduplicates_by_id():
    doubled = {"a": SAMPLE_RESPONSE, "b": SAMPLE_RESPONSE}
    assert len(parse_tweets(doubled, "bookmark")) == 1


def test_parse_tweets_detects_self_thread():
    sample = {
        "tweet_results": {
            "result": {
                "__typename": "Tweet",
                "rest_id": "222",
                "core": {
                    "user_results": {
                        "result": {"legacy": {"screen_name": "alice", "name": "Alice"}}
                    }
                },
                "legacy": {
                    "full_text": "thread tweet",
                    "created_at": "Wed May 10 14:23:00 +0000 2026",
                    "entities": {"urls": []},
                    "self_thread": {"id_str": "200"},
                },
            }
        }
    }
    items = parse_tweets(sample, "own_tweet")
    assert items[0].thread is not None
    assert items[0].thread.root_id == "200"


def _tweet_result(rest_id, handle, text, **legacy_extra):
    """Build a minimal `tweet_results.result` Tweet block for tests."""
    legacy = {
        "full_text": text,
        "created_at": "Wed May 10 14:23:00 +0000 2026",
        "entities": {"urls": []},
    }
    legacy.update(legacy_extra)
    return {
        "__typename": "Tweet",
        "rest_id": rest_id,
        "core": {
            "user_results": {"result": {"legacy": {"screen_name": handle, "name": handle.title()}}}
        },
        "legacy": legacy,
    }


def test_parse_tweets_skips_hydrated_quoted_tweet():
    quoted = _tweet_result("999", "bob", "quoted original tweet")
    bookmark = _tweet_result("333", "alice", "look at this")
    bookmark["legacy"]["quoted_status_result"] = {"result": quoted}
    response = {
        "data": {
            "bookmark_timeline_v2": {
                "timeline": {
                    "instructions": [
                        {
                            "type": "TimelineAddEntries",
                            "entries": [
                                {
                                    "entryId": "tweet-333",
                                    "content": {
                                        "itemContent": {
                                            "itemType": "TimelineTweet",
                                            "tweet_results": {"result": bookmark},
                                        }
                                    },
                                }
                            ],
                        }
                    ]
                }
            }
        }
    }
    items = parse_tweets(response, "bookmark")
    assert len(items) == 1
    assert items[0].id == "333"


def test_parse_tweets_reads_author_from_core_path():
    sample = {
        "tweet_results": {
            "result": {
                "__typename": "Tweet",
                "rest_id": "444",
                "core": {
                    "user_results": {"result": {"core": {"screen_name": "carol", "name": "Carol"}}}
                },
                "legacy": {
                    "full_text": "core-path author",
                    "created_at": "Wed May 10 14:23:00 +0000 2026",
                    "entities": {"urls": []},
                },
            }
        }
    }
    items = parse_tweets(sample, "bookmark")
    assert len(items) == 1
    assert items[0].author.handle == "carol"
    assert items[0].author.name == "Carol"


def test_parse_tweets_unwraps_visibility_envelope():
    sample = {
        "tweet_results": {
            "result": {
                "__typename": "TweetWithVisibilityResults",
                "tweet": _tweet_result("555", "dave", "limited visibility tweet"),
            }
        }
    }
    items = parse_tweets(sample, "bookmark")
    assert len(items) == 1
    assert items[0].id == "555"
    assert items[0].text == "limited visibility tweet"


# --- media: video variant selection (#40 part 1) ---------------------------


# Sentinel so a test can omit `duration_millis` entirely (an animated_gif
# entry has `video_info.variants` but no `duration_millis` key at all).
_OMIT = object()


def _video_legacy(
    variants: list[dict],
    *,
    media_type: str = "video",
    duration_millis: object = 30000,
) -> dict:
    """A `legacy` block carrying one video/animated_gif media entry whose poster
    is a fixed pbs.twimg.com image and whose playable URLs live in
    `video_info.variants` (the real X shape).

    `media_type` switches between `"video"` and `"animated_gif"`. Pass
    `duration_millis=_OMIT` to drop the key entirely (the animated_gif case)."""
    video_info: dict = {"variants": variants}
    if duration_millis is not _OMIT:
        video_info["duration_millis"] = duration_millis
    return {
        "extended_entities": {
            "media": [
                {
                    "type": media_type,
                    "media_url_https": "https://pbs.twimg.com/poster.jpg",
                    "video_info": video_info,
                }
            ]
        }
    }


def test_extract_media_video_picks_highest_bitrate_mp4():
    """A video entry stores the highest-bitrate mp4 from video_info.variants,
    not the poster image (media_url_https)."""
    from xbrain.extract.graphql import _extract_media
    from xbrain.models import MediaVideoPending

    legacy = _video_legacy(
        [
            {
                "bitrate": 256000,
                "content_type": "video/mp4",
                "url": "https://video.twimg.com/low.mp4?tag=12",
            },
            {
                "bitrate": 2176000,
                "content_type": "video/mp4",
                "url": "https://video.twimg.com/high.mp4?tag=12",
            },
            {
                "bitrate": 832000,
                "content_type": "video/mp4",
                "url": "https://video.twimg.com/mid.mp4?tag=12",
            },
            {
                "content_type": "application/x-mpegURL",
                "url": "https://video.twimg.com/playlist.m3u8?tag=12",
            },
        ]
    )

    media = _extract_media(legacy)

    assert len(media) == 1
    entry = media[0]
    assert isinstance(entry, MediaVideoPending)
    assert entry.url == "https://video.twimg.com/high.mp4?tag=12"


def test_extract_media_video_hls_only_falls_back_to_manifest():
    """With no progressive mp4 (HLS-only), store the m3u8 manifest URL rather
    than the poster, so the real stream is captured for a later ffmpeg pass."""
    from xbrain.extract.graphql import _extract_media
    from xbrain.models import MediaVideoPending

    legacy = _video_legacy(
        [
            {
                "content_type": "application/x-mpegURL",
                "url": "https://video.twimg.com/playlist.m3u8?tag=12",
            },
        ]
    )

    media = _extract_media(legacy)

    assert isinstance(media[0], MediaVideoPending)
    assert media[0].url == "https://video.twimg.com/playlist.m3u8?tag=12"


def test_extract_media_video_captures_poster_and_size_metadata():
    """The poster is kept as thumbnail_url, and the chosen variant's bitrate
    plus the clip duration are stored so size can be estimated without a
    download."""
    from xbrain.extract.graphql import _extract_media

    legacy = _video_legacy(
        [
            {
                "bitrate": 2176000,
                "content_type": "video/mp4",
                "url": "https://video.twimg.com/high.mp4?tag=12",
            },
        ],
        duration_millis=30000,
    )

    entry = _extract_media(legacy)[0]

    assert entry.thumbnail_url == "https://pbs.twimg.com/poster.jpg"
    assert entry.bitrate == 2176000
    assert entry.duration_millis == 30000


def test_extract_media_video_no_variants_falls_back_to_poster():
    """A video entry with no usable variants (empty list, no playable stream)
    falls back to the poster url so the item is not silently dropped; the poster
    is still kept as thumbnail_url and bitrate is None (nothing was chosen)."""
    from xbrain.extract.graphql import _extract_media
    from xbrain.models import MediaVideoPending

    legacy = _video_legacy([])

    entry = _extract_media(legacy)[0]

    assert isinstance(entry, MediaVideoPending)
    assert entry.url == "https://pbs.twimg.com/poster.jpg"
    assert entry.thumbnail_url == "https://pbs.twimg.com/poster.jpg"
    assert entry.bitrate is None


def test_extract_media_animated_gif_captures_mp4_without_duration():
    """An `animated_gif` entry is a single soundless mp4 with `bitrate: 0` and
    no `duration_millis`. The mp4 is captured, bitrate stays 0 (a real value,
    not "missing"), and duration_millis is None."""
    from xbrain.extract.graphql import _extract_media
    from xbrain.models import MediaVideoPending

    legacy = _video_legacy(
        [
            {
                "bitrate": 0,
                "content_type": "video/mp4",
                "url": "https://video.twimg.com/gif.mp4?tag=12",
            },
        ],
        media_type="animated_gif",
        duration_millis=_OMIT,
    )

    entry = _extract_media(legacy)[0]

    assert isinstance(entry, MediaVideoPending)
    assert entry.url == "https://video.twimg.com/gif.mp4?tag=12"
    assert entry.bitrate == 0
    assert entry.duration_millis is None


def test_extract_media_video_handles_null_and_missing_bitrate():
    """Variant selection must not crash when a variant has an explicit
    `"bitrate": null` (None) alongside variants with a missing bitrate key.
    `max(..., key=lambda v: v.get("bitrate", 0))` raises TypeError (None < int);
    a hardened key treats both null and missing as 0, and the variant carrying a
    real bitrate wins."""
    from xbrain.extract.graphql import _extract_media

    legacy = _video_legacy(
        [
            {
                "content_type": "video/mp4",
                "url": "https://video.twimg.com/missing.mp4?tag=12",
            },
            {
                "bitrate": None,
                "content_type": "video/mp4",
                "url": "https://video.twimg.com/null.mp4?tag=12",
            },
            {
                "bitrate": 832000,
                "content_type": "video/mp4",
                "url": "https://video.twimg.com/real.mp4?tag=12",
            },
        ]
    )

    entry = _extract_media(legacy)[0]

    assert entry.url == "https://video.twimg.com/real.mp4?tag=12"
    assert entry.bitrate == 832000


# --- extract: X Article entity -> canonical /i/article/<id> link (#39 PR2) ---
#
# FIXTURE PROVENANCE: the `article.article_results.result.rest_id` shape below
# is CONSTRUCTED from the documented X long-form-Article GraphQL shape, not
# recorded from a live bookmarks-with-Article payload (none was in the repo).
# The parser anchors ONLY on the stable key names `article` / `article_results`
# / `result` / `rest_id` (via the same null-safe `_dig` walk `_extract_author`
# uses), so a shape drift degrades to `None` (no link) rather than mis-parsing.
# Validate the exact key path against a real captured payload before relying on
# it in production (RFC #39 open-Q #4).


def _article_result_block(article_id: str, *, title: str = "The Long Read") -> dict:
    """The `article` block X attaches to a tweet result that carries a
    long-form Article. Only `rest_id` is load-bearing for PR2."""
    return {
        "article": {
            "article_results": {
                "result": {
                    "__typename": "Article",
                    "rest_id": article_id,
                    "title": title,
                }
            }
        }
    }


def _article_tweet(rest_id: str, article_id: str, *, handle: str = "alice", urls=None) -> dict:
    """A tweet result carrying a directly-bookmarked long-form Article."""
    tweet = _tweet_result(rest_id, handle, "check out my long read")
    tweet.update(_article_result_block(article_id))
    if urls is not None:
        tweet["legacy"]["entities"]["urls"] = urls
    return tweet


def test_extract_article_link_synthesizes_canonical_link():
    """A tweet carrying an Article entity yields the canonical
    `https://x.com/i/article/<rest_id>` Link on the stable x.com host."""
    from xbrain.extract.graphql import _extract_article_link
    from xbrain.models import Link

    link = _extract_article_link(_article_tweet("777", "1900000000000000000"))

    assert link == Link(url="https://x.com/i/article/1900000000000000000", domain="x.com")


def test_parse_tweets_attaches_article_link_to_item():
    """A directly-bookmarked Article surfaces the `/i/article/<id>` link on the
    parsed Item, so the existing `xbrain fetch` x.com path later fires for it."""
    sample = {"tweet_results": {"result": _article_tweet("777", "1900000000000000000")}}

    items = parse_tweets(sample, "bookmark")

    assert len(items) == 1
    article_links = [
        link for link in items[0].links if link.url == "https://x.com/i/article/1900000000000000000"
    ]
    assert len(article_links) == 1
    assert article_links[0].domain == "x.com"


def test_extract_article_link_not_duplicated_when_already_in_entities_urls():
    """If the tweet already surfaced the `/i/article/<id>` URL via
    `entities.urls`, the synthesized link is NOT double-added."""
    article_url = "https://x.com/i/article/555"
    tweet = _article_tweet(
        "333", "555", urls=[{"expanded_url": article_url, "url": "https://t.co/abc"}]
    )
    sample = {"tweet_results": {"result": tweet}}

    items = parse_tweets(sample, "bookmark")

    matching = [link for link in items[0].links if link.url == article_url]
    assert len(matching) == 1


def test_extract_article_link_none_for_plain_tweet():
    """A plain photo/video/text tweet (no Article entity) yields no link — the
    existing corpus is untouched (regression)."""
    from xbrain.extract.graphql import _extract_article_link

    assert _extract_article_link(_tweet_result("111", "alice", "just a normal tweet")) is None


@pytest.mark.parametrize(
    "article_block",
    [
        {},  # article present but empty
        "not-a-dict",  # article is a scalar (X drift)
        {"article_results": {}},  # no result node
        {"article_results": "nope"},  # article_results not a dict
        {"article_results": {"result": {}}},  # result carries no rest_id
        {"article_results": {"result": {"rest_id": ""}}},  # empty rest_id
        {"article_results": {"result": {"rest_id": None}}},  # null rest_id
    ],
)
def test_extract_article_link_malformed_returns_none(article_block):
    """A missing/renamed/malformed Article node degrades to None (no crash, no
    wrong link) — matching `_dig`'s null-safe walk."""
    from xbrain.extract.graphql import _extract_article_link

    tweet = _tweet_result("111", "alice", "text")
    tweet["article"] = article_block

    assert _extract_article_link(tweet) is None


# --- PR2 hardening round -----------------------------------------------------


def test_extract_article_link_dedups_noncanonical_url_variant():
    """A non-canonical variant of the article URL in `entities.urls`
    (twitter.com host / http scheme / trailing slash) suppresses the
    synthesized canonical link — no redundant re-fetch of the same Article."""
    variant = "http://twitter.com/i/article/555/"
    tweet = _article_tweet(
        "333", "555", urls=[{"expanded_url": variant, "url": "https://t.co/abc"}]
    )

    items = parse_tweets({"tweet_results": {"result": tweet}}, "bookmark")
    urls = [link.url for link in items[0].links]

    # the synthesized canonical link is NOT added (the variant already covers it)
    assert "https://x.com/i/article/555" not in urls
    # and the variant from entities.urls is still present
    assert variant in urls


def test_extract_article_link_rejects_non_article_typename():
    """A `result` with a non-Article `__typename` (e.g. a Card that happens to
    carry a `rest_id`) is not synthesized into an article link."""
    from xbrain.extract.graphql import _extract_article_link

    tweet = _tweet_result("111", "alice", "text")
    tweet["article"] = {"article_results": {"result": {"__typename": "Card", "rest_id": "999"}}}

    assert _extract_article_link(tweet) is None


@pytest.mark.parametrize(
    "rest_id",
    [
        {"x": "y"},  # dict — garbage-URL vector
        ["1", "2"],  # list — garbage-URL vector
        "abc",  # non-numeric text
        "12a3",  # mostly-numeric but not a clean id
        12345,  # int, not a string
        3.14,  # float
    ],
)
def test_extract_article_link_rejects_nonnumeric_rest_id(rest_id):
    """Only a numeric-string `rest_id` yields a link; a non-scalar or
    non-numeric id degrades to None (no crash, no garbage URL)."""
    from xbrain.extract.graphql import _extract_article_link

    tweet = _tweet_result("111", "alice", "text")
    tweet["article"] = {
        "article_results": {"result": {"__typename": "Article", "rest_id": rest_id}}
    }

    assert _extract_article_link(tweet) is None


def test_synthesized_article_link_routes_as_article():
    """Belt-and-suspenders: feeding the synthesizer's output into the fetch
    classifier yields `"article"`, so a future `/i/article/` rename can't
    silently break the extract→fetch routing contract."""
    from xbrain.extract.graphql import _extract_article_link
    from xbrain.fetch_x import _classify_x_url

    link = _extract_article_link(_article_tweet("777", "1900000000000000000"))

    assert link is not None
    assert _classify_x_url(link.url) == "article"
