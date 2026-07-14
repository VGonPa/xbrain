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


def test_iter_tweet_payloads_yields_the_full_subtree_per_tweet():
    """The persistence seam: (id, whole result subtree) per tweet — NOT the timeline
    envelope, which carries no item data and would duplicate ~20x per response."""
    from xbrain.extract.graphql import iter_tweet_payloads

    pairs = dict(iter_tweet_payloads(SAMPLE_RESPONSE))
    assert set(pairs) == {"111"}
    assert pairs["111"]["legacy"]["full_text"]  # the whole subtree, not a summary
    assert pairs["111"]["rest_id"] == "111"


def _quoted_author_block(handle: str, name: str) -> dict:
    return {"user_results": {"result": {"legacy": {"screen_name": handle, "name": name}}}}


def _quoting_tweet(quoted_result: dict | None, *, quoted_id: str | None = "222") -> dict:
    """Tweet 111 quoting tweet `quoted_id`; `quoted_result` is what X hydrated
    (None = X sent the id but embedded no post)."""
    tweet = _tweet_result("111", "alice", "Read this and you'll understand better this career move")
    if quoted_id is not None:
        tweet["legacy"]["quoted_status_id_str"] = quoted_id
    if quoted_result is not None:
        tweet["quoted_status_result"] = {"result": quoted_result}
    return tweet


def _quoted_post(**over) -> dict:
    quoted = _tweet_result("222", "karpathy", "I am leaving OpenAI.")
    quoted["core"] = _quoted_author_block("karpathy", "Andrej Karpathy")
    quoted.update(over)
    return quoted


def _timeline(tweet: dict) -> dict:
    """Wrap one tweet result in a bookmark-timeline envelope."""
    return {
        "data": {
            "bookmark_timeline_v2": {
                "timeline": {
                    "instructions": [
                        {
                            "type": "TimelineAddEntries",
                            "entries": [
                                {
                                    "entryId": f"tweet-{tweet['rest_id']}",
                                    "content": {
                                        "itemContent": {
                                            "itemType": "TimelineTweet",
                                            "tweet_results": {"result": tweet},
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


# --- Long-form posts: `legacy.full_text` is a 280-char TRUNCATION -------------------
#
# X caps `legacy.full_text` at 280 characters and appends a t.co self-link. For a long
# post the real body lives in `note_tweet.note_tweet_results.result.text`. Reading
# `full_text` hands the generator half a sentence — ending mid-word, mid-clause — and the
# rubric then tells it to summarise. It completes the sentence itself. That is a
# fabrication WE cause, at ingest, and it affects ~18% of the store (382 items); for 281
# of them the truncated tweet is the only evidence there is.


def _long_post_response(full_text: str, note_text: str | None) -> dict:
    """A bookmark payload whose tweet may carry a `note_tweet` long-form body.

    Built on the SHARED `_tweet_result` / `_timeline` helpers rather than a private copy of
    the envelope — two fixtures for one payload shape is how the two sides drift apart.
    """
    tweet = _tweet_result("999", "bob", full_text)
    if note_text is not None:
        tweet["note_tweet"] = {"note_tweet_results": {"result": {"id": "N1", "text": note_text}}}
    return _timeline(tweet)


def _only_source(tweet: dict):
    items = parse_tweets(_timeline(tweet), "bookmark")
    assert len(items) == 1
    item = items[0]
    assert item.content is not None, "the quoting item must carry a Content block"
    assert len(item.content.sources) == 1
    return item, item.content.sources[0]


def test_quoted_post_body_and_author_land_as_a_content_source():
    """The quoted post's TEXT and its AUTHOR are the evidence that was missing.

    The author matters as much as the body: the quoted post is a third party's,
    and naming the poster as its author is the attribution bug #86 guards against.
    """
    from xbrain.models import Author, ContentSourceSuccess

    item, source = _only_source(_quoting_tweet(_quoted_post()))

    assert isinstance(source, ContentSourceSuccess)
    assert source.kind == "quoted_tweet"
    assert source.text == "I am leaving OpenAI."
    assert source.author == Author(handle="karpathy", name="Andrej Karpathy")
    assert source.url == "https://x.com/karpathy/status/222"
    assert item.quoted_id == "222"


def test_the_quoted_source_carries_no_title():
    """`notes_io.note_title` takes the first source with a `title` — a quoted post
    has no title, and borrowing that field for the author would hijack note titles."""
    _, source = _only_source(_quoting_tweet(_quoted_post()))
    assert source.title is None


def test_the_quoted_source_is_not_a_link_content_kind():
    """It must never be mistaken for a fetched LINK body: a quoted post is not the
    content of any link the post points at."""
    from xbrain.executors.api import fetched_link_sources
    from xbrain.models import LINK_CONTENT_KINDS

    item, source = _only_source(_quoting_tweet(_quoted_post()))

    assert source.kind not in LINK_CONTENT_KINDS
    assert fetched_link_sources(item) == 0


def test_a_deleted_quoted_post_lands_as_a_not_found_failure():
    from xbrain.models import ContentSourceFailure

    tombstone = {"__typename": "TweetTombstone", "tombstone": {"text": {"text": "unavailable"}}}
    _, source = _only_source(_quoting_tweet(tombstone))

    assert isinstance(source, ContentSourceFailure)
    assert source.kind == "quoted_tweet"
    assert source.failure_reason == "not_found"
    assert source.url == "https://x.com/i/status/222"


def test_a_protected_quoted_post_lands_as_a_forbidden_failure():
    from xbrain.models import ContentSourceFailure

    unavailable = {"__typename": "TweetUnavailable", "reason": "Protected"}
    _, source = _only_source(_quoting_tweet(unavailable))

    assert isinstance(source, ContentSourceFailure)
    assert source.failure_reason == "forbidden"


def test_an_unhydrated_quoted_post_lands_as_an_empty_content_failure():
    """X sent the id but embedded no post — we know a quote exists and hold none
    of it. That is a FAILURE state, not the absence of a quote."""
    from xbrain.models import ContentSourceFailure

    _, source = _only_source(_quoting_tweet(None))

    assert isinstance(source, ContentSourceFailure)
    assert source.failure_reason == "empty_content"
    assert source.url == "https://x.com/i/status/222"


def test_a_quoted_post_with_no_body_and_no_author_is_an_empty_content_failure():
    from xbrain.models import ContentSourceFailure

    _, source = _only_source(_quoting_tweet({"__typename": "Tweet", "rest_id": "222"}))

    assert isinstance(source, ContentSourceFailure)
    assert source.failure_reason == "empty_content"


def test_a_tweet_that_quotes_nothing_gets_no_content_block():
    """No quote → no source, and crucially no EMPTY `Content` that would later read
    as "already fetched" to the link fetcher."""
    items = parse_tweets(_timeline(_tweet_result("111", "alice", "just a post")), "bookmark")
    assert items[0].quoted_id is None
    assert items[0].content is None


# ------------------------------------------- L6/M5: an honest record of what X did


def test_a_media_only_quoted_post_is_not_recorded_as_X_refusing_it():
    """X DID serve the post — it simply carries no text (a photo/video quote). Recording
    "X did not serve the quoted post" is a false statement about the world, in the very
    evidence store whose whole purpose is to stop us inventing facts."""
    from xbrain.models import ContentSourceFailure

    media_only = _quoted_post()
    media_only["legacy"]["full_text"] = ""
    _, source = _only_source(_quoting_tweet(media_only))

    assert isinstance(source, ContentSourceFailure)
    assert source.failure_reason == "empty_content"
    assert "no text" in (source.error or "")
    assert "did not serve" not in (source.error or "")


def test_a_quoted_post_with_a_body_but_no_author_still_ships_its_body():
    """The body is evidence even when X hydrates no author. Dropping it would throw away
    the cure because one field is missing; the attribution simply names nobody."""
    from xbrain.models import ContentSourceSuccess

    anonymous = _quoted_post()
    del anonymous["core"]
    _, source = _only_source(_quoting_tweet(anonymous))

    assert isinstance(source, ContentSourceSuccess)
    assert source.text == "I am leaving OpenAI."
    assert source.author is None


def test_long_post_uses_the_note_tweet_body_not_the_truncated_full_text():
    """The whole bug in one assertion: when X provides the long-form body, take it."""
    # A real long-form body is LONGER than the 280-char cut of itself — the fixture that
    # said otherwise described a shape X cannot produce.
    truncated = "The complete thought, " + "a" * 250 + " https://t.co/abc123"
    full = "The complete thought, " + "a" * 250 + " and the rest of it, ending properly."
    items = parse_tweets([_long_post_response(truncated, full)], "bookmark")
    assert len(items) == 1
    assert items[0].text == full  # identity, not a substring check
    assert "t.co" not in items[0].text


def test_a_short_post_still_uses_full_text():
    """No `note_tweet` → `full_text` IS the whole post. The fix must not disturb the 82%."""
    items = parse_tweets([_long_post_response("A short thought.", None)], "bookmark")
    assert items[0].text == "A short thought."


def test_an_empty_note_tweet_body_never_silently_replaces_the_tweet():
    """A present-but-empty long-form body must not blank the item: falling back to the
    truncated text is bad, but shipping an EMPTY tweet to the generator is worse."""
    items = parse_tweets([_long_post_response("Real text here.", "")], "bookmark")
    assert items[0].text == "Real text here."


# --- The ingest guard that would have caught this ------------------------------------


def test_truncation_detector_flags_x_s_long_post_signature():
    """X's signature is unmistakable and mechanical: ~280 characters of prose, cut
    mid-word, followed by an appended t.co self-link that is NOT among the tweet's own
    links. 382 items in the store carry it; 281 of them have no other evidence at all."""
    from xbrain.extract.graphql import looks_truncated

    body = "En este hilo explico por qué el modelo " * 7  # ~270 chars, cut mid-clause
    assert looks_truncated(body[:277] + " https://t.co/abc123", links=[])


def test_truncation_detector_does_not_flag_a_short_post_or_a_real_trailing_link():
    """A short post is not truncated, and a genuine link the tweet actually contains (it is
    in `links`) is not the appended self-link marker."""
    from xbrain.extract.graphql import looks_truncated

    assert not looks_truncated("A short thought.", links=[])
    real = "https://t.co/realone"
    assert not looks_truncated("Read this: " + real, links=[real])


def test_truncation_detector_flags_a_long_post_cut_without_any_link():
    """When in doubt, FLAG. A ~280-char text ending mid-word with no terminal punctuation is
    truncated even if the trailing link is absent — a detector that silently passes a
    genuinely truncated tweet is the failure mode that produced this bug."""
    from xbrain.extract.graphql import looks_truncated

    assert looks_truncated("palabra " * 35 + "cortado a mitad de", links=[])


def test_items_needing_refetch_selects_exactly_the_truncated_ones():
    """The backfill's selection is pure and testable; only the browser hop is not."""
    from datetime import datetime, timezone

    from xbrain.extract.graphql import items_needing_refetch
    from xbrain.models import Author, Item, Link

    def mk(item_id: str, text: str, links: list[str] | None = None) -> Item:
        return Item(
            id=item_id,
            source="bookmark",
            url=f"https://x.com/a/status/{item_id}",
            author=Author(handle="a", name="A"),
            text=text,
            created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            captured_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            links=[Link(url=u, domain="t.co") for u in (links or [])],
        )

    cut = "palabra " * 35 + "cortado a mitad de"
    store = {
        "1": mk("1", cut),  # truncated
        "2": mk("2", "Un post corto y completo."),  # fine
        "3": mk("3", "x" * 300 + " https://t.co/aaa"),  # truncated + appended self-link
    }
    assert [i.id for i in items_needing_refetch(store)] == ["1", "3"]


def test_a_long_QUOTED_post_uses_its_note_tweet_body_too():
    """N9, and it is live on `develop` right now.

    #98 ships the quoted post's body to fix 46% of all defects — and reads
    `legacy.full_text` DIRECTLY, so a long quoted post reaches the generator truncated at
    280 chars, mid-word. The exact failure #98 exists to fix, reintroduced through the back
    door. The quoted parse must go THROUGH the shared helper, not around it.
    """
    quoted = _tweet_result("222", "karpathy", "I am leaving " + "x" * 270 + " https://t.co/abc")
    quoted["note_tweet"] = {
        "note_tweet_results": {
            "result": {"text": "I am leaving " + "x" * 270 + " — and here is the whole reason."}
        }
    }
    quoted["core"] = _quoted_author_block("karpathy", "Andrej Karpathy")
    tweet = _tweet_result("111", "bob", "read this")
    tweet["quoted_status_result"] = {"result": quoted}
    tweet["legacy"]["quoted_status_id_str"] = "222"

    _item, source = _only_source(tweet)

    assert source.text.endswith("— and here is the whole reason.")
    assert "t.co" not in source.text


def test_a_note_tweet_shorter_than_full_text_never_wins():
    """N8. Unreachable in X's real schema — and this PR's entire thesis is that the
    unreachable shape is what bites you six months later."""
    from xbrain.extract.graphql import _tweet_text

    legacy = {"full_text": "the longer, real body of the post"}
    tweet = {"note_tweet": {"note_tweet_results": {"result": {"text": "short"}}}}
    assert _tweet_text(tweet, legacy) == "the longer, real body of the post"


# --- N2: the detector misses 72 of the worst-mutilated items --------------------------


def test_truncation_detector_flags_a_cut_that_ends_on_a_colon():
    """N2. `:` and `;` were in `_TERMINAL` — but **a colon is a CONTINUATION marker, the
    literal opposite of a sentence terminator.** And the punctuation escape overrode the
    LENGTH signal entirely, when X's cut is *defined* by prose length.

    These are the worst-mutilated items in the corpus: a numbered list cut at item 4, a
    promise of "the breakdown" with no breakdown. And after this PR lands,
    `refetch-truncated` is the ONLY repair path — so a miss here keeps its half-sentence
    forever.
    """
    from xbrain.extract.graphql import looks_truncated

    colon = "x" * 224 + " Here's how to write headers that convert like crazy:"
    assert len(colon) >= 274
    assert looks_truncated(colon, links=[])

    numbered = "y" * 250 + " how to select the best set of tools for your agent\n\n4."
    assert looks_truncated(numbered, links=[])


def test_truncation_detector_still_spares_a_short_post_that_ends_on_a_colon():
    """Length is the signal. A colon in a SHORT post is just a colon."""
    from xbrain.extract.graphql import looks_truncated

    assert not looks_truncated("Here is the list:", links=[])


def test_ingest_warns_when_a_tweet_arrives_truncated(caplog):
    """N7. I called `looks_truncated` "the ingest guard that would have caught this" — and
    never wired it into the extract path. Nothing warned at ingest, so if the `note_tweet`
    read ever regresses it goes silent again, exactly as before. A guard that is not called
    is a comment."""
    import logging

    with caplog.at_level(logging.WARNING):
        parse_tweets(_long_post_response("x" * 279 + " https://t.co/abc", None), "bookmark")

    assert any(
        "truncad" in r.message.lower() or "truncat" in r.message.lower() for r in caplog.records
    )
