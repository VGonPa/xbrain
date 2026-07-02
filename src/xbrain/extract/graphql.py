"""Parse X (Twitter) internal GraphQL responses into Item objects."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterator
from urllib.parse import urlparse

from xbrain.extract.video import build_video_media
from xbrain.models import (
    Author,
    Item,
    Link,
    Media,
    MediaEntry,
    SourceName,
    ThreadInfo,
)

logger = logging.getLogger(__name__)

X_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"
_NESTED_TWEET_KEYS = ("quoted_status_result", "retweeted_status_result")


def parse_tweets(response: dict[str, Any], source: SourceName) -> list[Item]:
    """Extract every timeline tweet from one X GraphQL response.

    Walks the response tree recursively looking for `tweet_results` blocks,
    which is how X wraps timeline tweets in both the Bookmarks and UserTweets
    operations. Anchoring on the key name (stable) rather than a fixed path
    keeps the parser resilient to X restructuring the surrounding envelope.
    """
    items: list[Item] = []
    seen: set[str] = set()
    for result in _find_tweet_results(response):
        tweet = _unwrap(result)
        if tweet is None:
            continue
        rest_id = tweet.get("rest_id")
        if not rest_id or rest_id in seen:
            continue
        item = _tweet_to_item(tweet, source)
        if item is not None:
            seen.add(str(rest_id))
            items.append(item)
    return items


def _find_tweet_results(obj: Any) -> Iterator[dict[str, Any]]:
    """Yield every top-level `tweet_results.result` dict in the tree.

    Skips quoted/retweeted sub-trees so a nested hydrated tweet is not
    surfaced as a standalone timeline item.
    """
    if isinstance(obj, dict):
        results = obj.get("tweet_results")
        if isinstance(results, dict) and isinstance(results.get("result"), dict):
            yield results["result"]
        for key, value in obj.items():
            if key not in _NESTED_TWEET_KEYS:
                yield from _find_tweet_results(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _find_tweet_results(value)


def _dig(obj: Any, *keys: str) -> dict[str, Any]:
    """Walk nested dict keys, returning {} on any missing/null/non-dict node."""
    for key in keys:
        obj = obj.get(key) if isinstance(obj, dict) else None
    return obj if isinstance(obj, dict) else {}


def _unwrap(result: dict[str, Any]) -> dict[str, Any] | None:
    """Unwrap the TweetWithVisibilityResults envelope if present."""
    typename = result.get("__typename")
    if typename == "TweetWithVisibilityResults":
        inner = result.get("tweet")
        return inner if isinstance(inner, dict) else None
    if typename == "Tweet":
        return result
    return result if "legacy" in result else None


def _tweet_to_item(tweet: dict[str, Any], source: SourceName) -> Item | None:
    legacy = tweet.get("legacy")
    rest_id = tweet.get("rest_id")
    if not isinstance(legacy, dict) or not rest_id:
        return None
    author = _extract_author(tweet)
    if author is None:
        return None
    return Item(
        id=str(rest_id),
        source=source,
        url=f"https://x.com/{author.handle}/status/{rest_id}",
        author=author,
        text=legacy.get("full_text", ""),
        created_at=_parse_x_date(legacy.get("created_at")),
        captured_at=datetime.now(timezone.utc),
        media=_extract_media(legacy),
        links=_extract_links(legacy, tweet),
        quoted_id=legacy.get("quoted_status_id_str") or _quoted_id(tweet),
        thread=_thread_info(legacy),
    )


def _extract_author(tweet: dict[str, Any]) -> Author | None:
    user = _dig(tweet, "core", "user_results", "result")
    legacy = _dig(user, "legacy")
    core = _dig(user, "core")  # newer responses moved name/handle here
    handle = legacy.get("screen_name") or core.get("screen_name")
    name = legacy.get("name") or core.get("name")
    if not handle:
        return None
    return Author(handle=handle, name=name or handle)


def _extract_links(legacy: dict[str, Any], tweet: dict[str, Any]) -> list[Link]:
    """Every link on a tweet: the text URLs in `entities.urls` plus the
    synthesized canonical link for a directly-bookmarked long-form Article.

    The Article link is appended only when the tweet carries an Article entity
    and the URL is not already present (dedup against `entities.urls`), so a
    tweet that merely *links* an Article never double-adds it.
    """
    links: list[Link] = []
    for entry in legacy.get("entities", {}).get("urls", []):
        expanded = entry.get("expanded_url")
        if expanded:
            links.append(Link(url=expanded, domain=urlparse(expanded).netloc))
    article = _extract_article_link(tweet)
    if article is not None and article.url not in {link.url for link in links}:
        links.append(article)
    return links


def _extract_article_link(tweet: dict[str, Any]) -> Link | None:
    """Synthesize the canonical `/i/article/<id>` Link for a directly-bookmarked
    X long-form Article, or None when the tweet carries no Article entity.

    X attaches a long-form Article to its tweet result as an `article` block:
    `tweet["article"]["article_results"]["result"]` carries the Article's
    numeric `rest_id`. We anchor on those stable key names via `_dig` (the same
    null-safe walk `_extract_author` uses) rather than a fixed path, so an X
    shape drift degrades to None (no link) instead of mis-parsing into a wrong
    link. The `https://x.com/i/article/<rest_id>` URL is chosen so the existing
    `is_x_url` + `_classify_x_url` routing already fires the rendered-fetch path
    for it — no change to `fetch_x`.

    NOTE: the `article.article_results.result.rest_id` key path is pinned
    against a CONSTRUCTED fixture (see `tests/test_graphql.py`), not a recorded
    live payload; validate it against a real bookmarked-Article GraphQL
    response before production reliance (RFC #39 open-Q #4).
    """
    result = _dig(tweet, "article", "article_results", "result")
    rest_id = result.get("rest_id")
    if not rest_id:
        return None
    return Link(url=f"https://x.com/i/article/{rest_id}", domain="x.com")


def _extract_media(legacy: dict[str, Any]) -> list[MediaEntry]:
    entries = legacy.get("extended_entities", {}).get("media") or legacy.get("entities", {}).get(
        "media", []
    )
    media: list[MediaEntry] = []
    for entry in entries:
        if entry.get("type") in ("video", "animated_gif"):
            video = build_video_media(entry)
            if video is not None:
                media.append(video)
            continue
        url = entry.get("media_url_https") or entry.get("expanded_url")
        if url:
            media.append(Media(type="photo", url=url))
    return media


def _quoted_id(tweet: dict[str, Any]) -> str | None:
    quoted = _dig(tweet, "quoted_status_result", "result")
    rest_id = quoted.get("rest_id")
    return str(rest_id) if rest_id else None


def _thread_info(legacy: dict[str, Any]) -> ThreadInfo | None:
    """Detect a self-thread from X's `self_thread` marker in the legacy block."""
    self_thread = legacy.get("self_thread")
    if isinstance(self_thread, dict) and self_thread.get("id_str"):
        return ThreadInfo(root_id=str(self_thread["id_str"]))
    return None


def _parse_x_date(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.strptime(value, X_DATE_FORMAT)
    except ValueError:
        logger.warning("unparseable X date %r, using now()", value)
        return datetime.now(timezone.utc)
