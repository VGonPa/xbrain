"""Content-retention policy for the XBrain corpus.

XBrain keeps saved bookmarks that can become durable knowledge: long-form
articles and videos. Plain posts, image-only posts and own tweets are discarded
so downstream stages operate on the real library instead of social-feed noise.
"""

from __future__ import annotations

from urllib.parse import urlparse

from xbrain.models import (
    ContentSourceFailure,
    ContentSourceSuccess,
    Item,
    MediaVideoDownloaded,
    MediaVideoFailed,
    MediaVideoPending,
)

_ARTICLE_SOURCE_KINDS = frozenset({"external_article", "x_article"})
_VIDEO_MEDIA_TYPES = (MediaVideoPending, MediaVideoDownloaded, MediaVideoFailed)
_X_HOSTS = frozenset({"x.com", "www.x.com", "twitter.com", "www.twitter.com"})


def is_article_candidate_url(url: str) -> bool:
    """True when a link may produce article content.

    External URLs are candidates because `fetch_pending` can extract their body.
    X `/i/article/<id>` URLs are candidates for `fetch_x_articles`; ordinary
    tweet/status links are not, because the product no longer keeps post-only
    bookmarks.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in _X_HOSTS:
        return parsed.path.rstrip("/").startswith("/i/article/")
    return bool(host)


def has_video(item: Item) -> bool:
    """True when an item carries any video state worth keeping/retrying."""
    return any(isinstance(entry, _VIDEO_MEDIA_TYPES) for entry in item.media)


def has_article_candidate(item: Item) -> bool:
    """True when a bookmark has a link that can become article content."""
    return any(is_article_candidate_url(link.url) for link in item.links)


def has_article_source(item: Item) -> bool:
    """True when fetched content includes article success/failure evidence."""
    if item.content is None:
        return False
    return any(
        source.kind in _ARTICLE_SOURCE_KINDS
        and isinstance(source, (ContentSourceSuccess, ContentSourceFailure))
        for source in item.content.sources
    )


def should_keep_item(item: Item) -> bool:
    """Return whether an item belongs in the long-term XBrain corpus."""
    if item.source != "bookmark":
        return False
    return has_video(item) or has_article_candidate(item) or has_article_source(item)


def prune_store(store: dict[str, Item]) -> list[str]:
    """Remove non-library items from the store, returning removed ids."""
    removed = [item_id for item_id, item in store.items() if not should_keep_item(item)]
    for item_id in removed:
        del store[item_id]
    return removed


def retained_store(store: dict[str, Item]) -> dict[str, Item]:
    """Return the non-mutating retained-library view of a store."""
    return {item_id: item for item_id, item in store.items() if should_keep_item(item)}
