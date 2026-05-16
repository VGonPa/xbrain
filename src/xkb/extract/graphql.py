"""Parse X (Twitter) internal GraphQL responses into Item objects."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator
from urllib.parse import urlparse

from xkb.models import Author, Item, Link, Media, ThreadInfo

X_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def parse_tweets(response: dict[str, Any], source: str) -> list[Item]:
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
    """Yield every `tweet_results.result` dict found anywhere in the tree."""
    if isinstance(obj, dict):
        results = obj.get("tweet_results")
        if isinstance(results, dict) and isinstance(results.get("result"), dict):
            yield results["result"]
        for value in obj.values():
            yield from _find_tweet_results(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _find_tweet_results(value)


def _unwrap(result: dict[str, Any]) -> dict[str, Any] | None:
    """Unwrap the TweetWithVisibilityResults envelope if present."""
    typename = result.get("__typename")
    if typename == "TweetWithVisibilityResults":
        inner = result.get("tweet")
        return inner if isinstance(inner, dict) else None
    if typename == "Tweet":
        return result
    return result if "legacy" in result else None


def _tweet_to_item(tweet: dict[str, Any], source: str) -> Item | None:
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
        links=_extract_links(legacy),
        quoted_id=legacy.get("quoted_status_id_str") or _quoted_id(tweet),
        thread=_thread_info(legacy),
    )


def _extract_author(tweet: dict[str, Any]) -> Author | None:
    user = tweet.get("core", {}).get("user_results", {}).get("result", {})
    legacy = user.get("legacy", {})
    core = user.get("core", {})  # newer responses moved name/handle here
    handle = legacy.get("screen_name") or core.get("screen_name")
    name = legacy.get("name") or core.get("name")
    if not handle:
        return None
    return Author(handle=handle, name=name or handle)


def _extract_links(legacy: dict[str, Any]) -> list[Link]:
    links: list[Link] = []
    for entry in legacy.get("entities", {}).get("urls", []):
        expanded = entry.get("expanded_url")
        if expanded:
            links.append(Link(url=expanded, domain=urlparse(expanded).netloc))
    return links


def _extract_media(legacy: dict[str, Any]) -> list[Media]:
    entries = (
        legacy.get("extended_entities", {}).get("media")
        or legacy.get("entities", {}).get("media", [])
    )
    media: list[Media] = []
    for entry in entries:
        kind = "video" if entry.get("type") in ("video", "animated_gif") else "photo"
        url = entry.get("media_url_https") or entry.get("expanded_url")
        if url:
            media.append(Media(type=kind, url=url))
    return media


def _quoted_id(tweet: dict[str, Any]) -> str | None:
    quoted = tweet.get("quoted_status_result", {}).get("result", {})
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
        return datetime.now(timezone.utc)
