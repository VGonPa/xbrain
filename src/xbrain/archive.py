"""Parse the official X data archive (tweets.js) into Item objects."""

from __future__ import annotations

import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from xbrain.extract.graphql import _parse_x_date
from xbrain.models import Author, Item, Link, Media, MediaEntry

logger = logging.getLogger(__name__)

_TWEET_FILES = ("data/tweets.js", "data/tweet.js")


def parse_archive(zip_path: Path, author: Author) -> list[Item]:
    """Extract all own tweets from an X data archive ZIP."""
    with zipfile.ZipFile(zip_path) as archive:
        tweets_file = _find_tweets_file(archive)
        raw = archive.read(tweets_file).decode("utf-8")
    items: list[Item] = []
    for entry in _parse_js_array(raw, tweets_file):
        item = _archive_tweet_to_item(entry, author)
        if item is not None:
            items.append(item)
    return items


def _find_tweets_file(archive: zipfile.ZipFile) -> str:
    names = set(archive.namelist())
    for candidate in _TWEET_FILES:
        if candidate in names:
            return candidate
    raise ValueError(f"No tweets file in archive (looked for {_TWEET_FILES})")


def _parse_js_array(raw: str, tweets_file: str) -> list:
    """Strip the `window.YTD...=` JS prefix and parse the JSON array body."""
    bracket = raw.find("[")
    if bracket == -1:
        raise ValueError(f"{tweets_file}: no JSON array found in archive tweets file")
    try:
        return json.loads(raw[bracket:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{tweets_file}: malformed JSON in archive tweets file: {exc}") from exc


def _extract_archive_links(tweet: dict[str, Any]) -> list[Link]:
    """Parse `entities.urls` from an archive tweet into Link objects.

    Each URL entity in the archive carries an `expanded_url`; we derive the
    domain from it and drop any entry that lacks an expanded URL.
    """
    return [
        Link(
            url=url_entity["expanded_url"],
            domain=urlparse(url_entity["expanded_url"]).netloc,
        )
        for url_entity in tweet.get("entities", {}).get("urls", [])
        if url_entity.get("expanded_url")
    ]


def _extract_archive_media(tweet: dict[str, Any]) -> list[MediaEntry]:
    """Parse media from `extended_entities.media`, falling back to `entities.media`.

    Normalises `type` to `"video"` (videos and animated GIFs) or `"photo"`, and
    picks the canonical URL (`media_url_https`, falling back to `expanded_url`).
    Entries missing both URLs are dropped.
    """
    media_entries = tweet.get("extended_entities", {}).get("media") or tweet.get(
        "entities", {}
    ).get("media", [])
    return [
        Media(
            type="video" if media_entity.get("type") in ("video", "animated_gif") else "photo",
            url=media_entity.get("media_url_https") or media_entity["expanded_url"],
        )
        for media_entity in media_entries
        if media_entity.get("media_url_https") or media_entity.get("expanded_url")
    ]


def _archive_tweet_to_item(entry: dict[str, Any], author: Author) -> Item | None:
    """Map one archive entry into an Item, returning None for malformed rows."""
    tweet = entry.get("tweet")
    if not isinstance(tweet, dict):
        logger.warning("archive entry missing 'tweet' object, skipping")
        return None
    rest_id = tweet.get("id_str")
    if not rest_id:
        logger.warning("archive tweet missing 'id_str', skipping")
        return None
    rest_id = str(rest_id)
    return Item(
        id=rest_id,
        source="own_tweet",
        url=f"https://x.com/{author.handle}/status/{rest_id}",
        author=author,
        text=tweet.get("full_text", ""),
        created_at=_parse_x_date(tweet.get("created_at")),
        captured_at=datetime.now(timezone.utc),
        media=_extract_archive_media(tweet),
        links=_extract_archive_links(tweet),
        quoted_id=None,
    )
