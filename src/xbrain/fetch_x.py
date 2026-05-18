"""Fetch X (x.com) content linked from items.

`/status/` links are fetched by reusing the `TweetDetail` GraphQL interception
proven in `xbrain.extract.threads`; `/i/article/` and other x.com links are
fetched as Playwright-rendered HTML and run through trafilatura. A fetch failure
records the same structured evidence as `xbrain.fetch` (design §4, §15.2).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import urlparse

import trafilatura
from playwright.sync_api import BrowserContext, Response

from xbrain.extract.browser import is_logged_out, x_context
from xbrain.extract.graphql import parse_tweets
from xbrain.fetch import is_x_url
from xbrain.models import Content, ContentSource, Item

_SETTLE_MS = 4000
_STATUS_RE = re.compile(r"/status/(\d+)")

LinkFetcher = Callable[[str], ContentSource]


def _x_status_id(url: str) -> str | None:
    """The tweet id of an x.com `/status/<id>` URL, or None."""
    match = _STATUS_RE.search(urlparse(url).path)
    return match.group(1) if match else None


def _classify_x_url(url: str) -> Literal["status", "article", "other"]:
    """Classify an x.com URL: a tweet/thread, an X article, or anything else."""
    path = urlparse(url).path
    if _STATUS_RE.search(path):
        return "status"
    if path.startswith("/i/article/"):
        return "article"
    return "other"


def assemble_linked_thread(responses: list, anchor_id: str) -> tuple[str | None, str]:
    """From captured `TweetDetail` responses, concatenate the thread rooted at
    the linked tweet. Pure — no browser. Returns `(author_handle, text)`."""
    tweets: list[Item] = []
    seen: set[str] = set()
    for response in responses:
        for tweet in parse_tweets(response, "bookmark"):
            if tweet.id not in seen:
                seen.add(tweet.id)
                tweets.append(tweet)
    anchor = next((t for t in tweets if t.id == anchor_id), None)
    if anchor is None:
        return None, ""
    handle = anchor.author.handle
    thread = sorted((t for t in tweets if t.author.handle == handle), key=lambda t: t.created_at)
    return handle, "\n\n".join(t.text for t in thread)


def _fetch_tweet(context: BrowserContext, url: str) -> ContentSource:
    """Fetch a linked tweet/thread via TweetDetail interception."""
    captured: list[dict] = []
    page = context.new_page()

    def on_response(response: Response) -> None:
        if "/graphql/" in response.url and "TweetDetail" in response.url:
            try:
                captured.append(response.json())
            except Exception:  # noqa: BLE001 - ignore non-JSON / partial bodies
                pass

    page.on("response", on_response)
    try:
        try:
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(_SETTLE_MS)
        except Exception:  # noqa: BLE001 - navigation failure -> empty result
            return ContentSource(
                kind="x_article",
                url=url,
                ok=False,
                failure_reason="timeout",
                error="No se pudo cargar el tweet.",
                attempts=1,
            )
        if is_logged_out(page.url):
            raise RuntimeError("Sesión de X caducada. Ejecuta `xbrain login`.")
        _handle, text = assemble_linked_thread(captured, _x_status_id(url) or "")
    finally:
        page.close()
    if text:
        return ContentSource(kind="x_article", url=url, text=text, ok=True, attempts=1)
    return ContentSource(
        kind="x_article",
        url=url,
        ok=False,
        failure_reason="empty_content",
        error="No se pudo recuperar el contenido del tweet.",
        attempts=1,
    )


def _fetch_rendered(context: BrowserContext, url: str) -> ContentSource:
    """Fetch an X article (or other x.com page) as rendered HTML."""
    page = context.new_page()
    try:
        try:
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(_SETTLE_MS)
        except Exception:  # noqa: BLE001 - navigation failure -> empty result
            return ContentSource(
                kind="x_article",
                url=url,
                ok=False,
                failure_reason="timeout",
                error="No se pudo cargar el artículo de X.",
                attempts=1,
            )
        if is_logged_out(page.url):
            raise RuntimeError("Sesión de X caducada. Ejecuta `xbrain login`.")
        html = page.content()
    finally:
        page.close()
    text = trafilatura.extract(html)
    if text:
        return ContentSource(
            kind="x_article", url=url, text=text, ok=True, http_status=200, attempts=1
        )
    return ContentSource(
        kind="x_article",
        url=url,
        ok=False,
        failure_reason="empty_content",
        error="No se pudo extraer el contenido del artículo de X.",
        attempts=1,
    )


def _fetch_x_link(context: BrowserContext, url: str) -> ContentSource:
    """Fetch one x.com link, routed by URL kind."""
    if _classify_x_url(url) == "status":
        return _fetch_tweet(context, url)
    return _fetch_rendered(context, url)


def _needs_x_fetch(item: Item, force: bool) -> bool:
    if not any(is_x_url(link.url) for link in item.links):
        return False
    if force or item.content is None:
        return True
    return not any(s.kind == "x_article" for s in item.content.sources)


def _attach_x_sources(item: Item, sources: list[ContentSource]) -> None:
    """Replace the item's `x_article` sources, keeping every other kind."""
    if item.content is None:
        item.content = Content(fetched_at=datetime.now(timezone.utc), sources=list(sources))
    else:
        kept = [s for s in item.content.sources if s.kind != "x_article"]
        item.content.sources = kept + list(sources)


def fetch_x_articles(
    store: dict[str, Item],
    storage_state_path: Path | None,
    force: bool = False,
    *,
    link_fetcher: LinkFetcher | None = None,
) -> int:
    """Fetch x.com link content for every item that has one and needs it.

    `link_fetcher` is injected by tests; in production it is bound to a live
    Playwright context opened from `storage_state_path`.
    """
    pending = [item for item in store.values() if _needs_x_fetch(item, force)]
    if not pending:
        return 0

    def _process(fetcher: LinkFetcher) -> None:
        for item in pending:
            sources = [fetcher(link.url) for link in item.links if is_x_url(link.url)]
            _attach_x_sources(item, sources)

    if link_fetcher is not None:
        _process(link_fetcher)
    else:
        if storage_state_path is None:
            raise ValueError("storage_state_path is required without an injected link_fetcher")
        with x_context(storage_state_path) as context:
            _process(lambda url: _fetch_x_link(context, url))
    return len(pending)
