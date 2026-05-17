# tests/test_archive.py
import json
import zipfile
from pathlib import Path

import pytest

from xbrain.archive import parse_archive
from xbrain.models import Author


def _make_archive(tmp_path: Path, filename: str = "data/tweets.js") -> Path:
    tweets = [
        {
            "tweet": {
                "id_str": "555",
                "created_at": "Wed May 10 14:23:00 +0000 2026",
                "full_text": "hello https://t.co/x",
                "entities": {
                    "urls": [{"expanded_url": "https://example.com/post"}]
                },
            }
        }
    ]
    body = "window.YTD.tweets.part0 = " + json.dumps(tweets)
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(filename, body)
    return zip_path


def _make_archive_from_tweets(
    tmp_path: Path, tweets: list, filename: str = "data/tweets.js"
) -> Path:
    """Build an archive ZIP from an arbitrary list of entries."""
    body = "window.YTD.tweets.part0 = " + json.dumps(tweets)
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(filename, body)
    return zip_path


def test_parse_archive_extracts_own_tweets(tmp_path: Path):
    author = Author(handle="vgonpa", name="Victor")
    items = parse_archive(_make_archive(tmp_path), author)
    assert len(items) == 1
    item = items[0]
    assert item.id == "555"
    assert item.source == "own_tweet"
    assert item.url == "https://x.com/vgonpa/status/555"
    assert item.links[0].domain == "example.com"


def test_parse_archive_handles_legacy_tweet_js_name(tmp_path: Path):
    author = Author(handle="vgonpa", name="Victor")
    items = parse_archive(_make_archive(tmp_path, "data/tweet.js"), author)
    assert items[0].id == "555"


def test_parse_archive_raises_on_missing_tweets_file(tmp_path: Path):
    author = Author(handle="vgonpa", name="Victor")
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("data/account.js", "window.YTD.account.part0 = []")
    with pytest.raises(ValueError):
        parse_archive(zip_path, author)


def test_parse_archive_raises_on_malformed_json(tmp_path: Path):
    author = Author(handle="vgonpa", name="Victor")
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("data/tweets.js", "window.YTD.tweets.part0 = [ {bad json")
    with pytest.raises(ValueError):
        parse_archive(zip_path, author)


def test_parse_archive_extracts_media(tmp_path: Path):
    author = Author(handle="vgonpa", name="Victor")
    tweets = [
        {
            "tweet": {
                "id_str": "777",
                "created_at": "Wed May 10 14:23:00 +0000 2026",
                "full_text": "with a photo",
                "extended_entities": {
                    "media": [
                        {
                            "type": "photo",
                            "media_url_https": "https://pbs.twimg.com/media/a.jpg",
                        }
                    ]
                },
            }
        }
    ]
    items = parse_archive(_make_archive_from_tweets(tmp_path, tweets), author)
    assert len(items) == 1
    assert len(items[0].media) == 1
    assert items[0].media[0].type == "photo"
    assert items[0].media[0].url == "https://pbs.twimg.com/media/a.jpg"


def test_parse_archive_skips_malformed_entry(tmp_path: Path):
    author = Author(handle="vgonpa", name="Victor")
    tweets = [
        {
            "tweet": {
                "id_str": "555",
                "created_at": "Wed May 10 14:23:00 +0000 2026",
                "full_text": "valid tweet",
            }
        },
        {"tweet": {"created_at": "Wed May 10 14:23:00 +0000 2026"}},
    ]
    items = parse_archive(_make_archive_from_tweets(tmp_path, tweets), author)
    assert len(items) == 1
    assert items[0].id == "555"
