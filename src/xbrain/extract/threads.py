"""Expand X threads into concatenated text via the TweetDetail operation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import BrowserContext, Response

from xbrain.extract.browser import is_logged_out, x_context
from xbrain.extract.graphql import parse_tweets
from xbrain.models import (
    Content,
    ContentSource,
    ContentSourceFailure,
    ContentSourceSuccess,
    Item,
)

_SETTLE_MS = 4000


def assemble_thread(responses: list[dict], author_handle: str) -> str:
    """Concatenate, in chronological order, the thread tweets by one author.

    Pure function — no browser. This is the unit-tested core of expansion.
    """
    tweets = []
    seen: set[str] = set()
    for response in responses:
        for tweet in parse_tweets(response, "own_tweet"):
            if tweet.author.handle == author_handle and tweet.id not in seen:
                seen.add(tweet.id)
                tweets.append(tweet)
    tweets.sort(key=lambda tweet: tweet.created_at)
    return "\n\n".join(tweet.text for tweet in tweets)


def expand_threads(store: dict[str, Item], storage_state_path: Path, force: bool = False) -> int:
    """Fetch full thread text for every item flagged as a thread."""
    pending = [
        item
        for item in store.values()
        if item.thread is not None and not _already_expanded(item, force)
    ]
    if not pending:
        return 0
    with x_context(storage_state_path) as context:
        for item in pending:
            text = _fetch_thread_text(context, item)
            source: ContentSource
            if text:
                source = ContentSourceSuccess(kind="thread", url=item.url, text=text)
            else:
                source = ContentSourceFailure(
                    kind="thread",
                    url=item.url,
                    failure_reason="empty_content",
                    error="No se pudo recuperar el hilo.",
                )
            _attach_thread(item, source)
    return len(pending)


def _fetch_thread_text(context: BrowserContext, item: Item) -> str:
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
            page.goto(item.url, wait_until="domcontentloaded")
            page.wait_for_timeout(_SETTLE_MS)
        except Exception:  # noqa: BLE001 - navigation failure -> empty thread
            return ""
        if is_logged_out(page.url):
            raise RuntimeError("Sesión de X caducada. Ejecuta `xbrain login`.")
        return assemble_thread(captured, item.author.handle)
    finally:
        page.close()


def _already_expanded(item: Item, force: bool) -> bool:
    if force or item.content is None:
        return False
    return any(source.kind == "thread" for source in item.content.sources)


def _attach_thread(item: Item, source: ContentSource) -> None:
    if item.content is None:
        item.content = Content(fetched_at=datetime.now(timezone.utc), sources=[source])
    else:
        kept = [s for s in item.content.sources if s.kind != "thread"]
        item.content.sources = kept + [source]
