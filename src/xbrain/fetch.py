"""Fetch the external content linked from items.

v1 fetches external web articles via trafilatura. x.com links (X articles,
threads, quoted tweets) are recorded but not auto-extracted — see the plan's
scope notes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable
from urllib.parse import urlparse

import trafilatura

from xbrain.models import Content, ContentSource, Item

ArticleExtractor = Callable[[str], tuple[str | None, str | None]]

_X_HOSTS = {"x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"}


def extract_article(url: str) -> tuple[str | None, str | None]:
    """Download and extract a web article. Returns (title, text)."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        return None, None
    text = trafilatura.extract(downloaded)
    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title if metadata else None
    return title, text


def fetch_item(item: Item, extractor: ArticleExtractor = extract_article) -> Content:
    """Fetch every external link of an item into ContentSource entries."""
    sources: list[ContentSource] = []
    for link in item.links:
        if _is_x_url(link.url):
            sources.append(ContentSource(
                kind="x_article",
                url=link.url,
                ok=False,
                error="El contenido de x.com no se extrae automáticamente en v1.",
            ))
            continue
        try:
            title, text = extractor(link.url)
        except Exception as exc:  # noqa: BLE001 - one bad URL must not abort the batch
            sources.append(ContentSource(
                kind="external_article",
                url=link.url,
                ok=False,
                error=f"Error al descargar el artículo: {exc}",
            ))
            continue
        if text:
            sources.append(ContentSource(
                kind="external_article",
                url=link.url,
                title=title,
                text=text,
                ok=True,
            ))
        else:
            sources.append(ContentSource(
                kind="external_article",
                url=link.url,
                ok=False,
                error="No se pudo extraer el artículo.",
            ))
    return Content(fetched_at=datetime.now(timezone.utc), sources=sources)


def fetch_pending(
    store: dict[str, Item],
    since: datetime | None = None,
    until: datetime | None = None,
    force: bool = False,
    extractor: ArticleExtractor = extract_article,
) -> int:
    """Fetch content for items that have links and no content yet."""
    fetched = 0
    for item in store.values():
        if not item.links:
            continue
        if item.content is not None and not force:
            continue
        if since and item.created_at < since:
            continue
        if until and item.created_at > until:
            continue
        item.content = fetch_item(item, extractor)
        fetched += 1
    return fetched


def _is_x_url(url: str) -> bool:
    return (urlparse(url).hostname or "").lower() in _X_HOSTS
