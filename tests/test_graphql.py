# tests/test_graphql.py
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


def _video_legacy(variants: list[dict], *, duration_millis: int = 30000) -> dict:
    """A `legacy` block carrying one video media entry whose poster is a
    fixed pbs.twimg.com image and whose playable URLs live in
    `video_info.variants` (the real X shape)."""
    return {
        "extended_entities": {
            "media": [
                {
                    "type": "video",
                    "media_url_https": "https://pbs.twimg.com/poster.jpg",
                    "video_info": {
                        "duration_millis": duration_millis,
                        "variants": variants,
                    },
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
