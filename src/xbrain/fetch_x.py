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

from xbrain.extract.article import parse_article_content_state
from xbrain.extract.browser import is_logged_out, x_context
from xbrain.extract.graphql import parse_tweets
from xbrain.fetch import _sources_materially_equal, is_x_url
from xbrain.models import (
    ArticleBlock,
    ArticleTextBlock,
    Content,
    ContentSource,
    ContentSourceFailure,
    ContentSourceSuccess,
    Item,
)

_SETTLE_MS = 4000
_STATUS_RE = re.compile(r"/[^/]+/status/(\d+)")
# The article body rides an X GraphQL operation whose name contains "article"
# (e.g. `TweetArticleContent`). We match on that stable URL substring — the same
# op-name-substring anchor `_fetch_tweet` uses for `TweetDetail` — rather than a
# pinned op name, so a minor op rename still captures the response. The exact op
# name is UNCONFIRMED against a live payload (RFC #39 open-Q #4); on any miss the
# parser yields no blocks and we degrade to the trafilatura text fallback.
_ARTICLE_GRAPHQL_HINT = "article"

LinkFetcher = Callable[[str], ContentSource]


def _x_status_id(url: str) -> str | None:
    """The tweet id of an x.com `/status/<id>` URL, or None."""
    match = _STATUS_RE.search(urlparse(url).path)
    return match.group(1) if match else None


def _classify_x_url(url: str) -> Literal["status", "article", "other"]:
    """Classify an x.com URL: a tweet/thread, an X article, or anything else."""
    path = urlparse(url).path
    # Check the article prefix before /status/ — an X article URL can itself
    # contain a `/status/<id>` segment and must not be misrouted as a tweet.
    if path.startswith("/i/article/"):
        return "article"
    if _STATUS_RE.search(path):
        return "status"
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
    """Fetch a linked tweet/thread via TweetDetail interception.

    The result is filed as `kind="x_article"` (not `thread`) so all x.com-link
    content shares one source kind for `_needs_x_fetch` / `_attach_x_sources`.
    """
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
            return ContentSourceFailure(
                kind="x_article",
                url=url,
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
        return ContentSourceSuccess(kind="x_article", url=url, text=text, attempts=1)
    return ContentSourceFailure(
        kind="x_article",
        url=url,
        failure_reason="empty_content",
        error="No se pudo recuperar el contenido del tweet.",
        attempts=1,
    )


def _is_article_graphql(response_url: str) -> bool:
    """True for a GraphQL response that may carry the article content_state."""
    return "/graphql/" in response_url and _ARTICLE_GRAPHQL_HINT in response_url.lower()


def _flatten_blocks(blocks: list[ArticleBlock]) -> str:
    """The flattened body: the concatenation of the text-run texts, in order.

    This is the PR1 `text`-is-flattened-body invariant — the separators live
    inside the text runs, so `enrich`/`topics` consume `text` unchanged.
    """
    return "".join(b.text for b in blocks if isinstance(b, ArticleTextBlock))


def _structured_article(captured: list[dict], url: str) -> ContentSourceSuccess | None:
    """Build a structured `x_article` success from captured GraphQL responses.

    Returns None when nothing parsed to blocks — the caller then falls back to
    trafilatura. A captured-but-empty parse is a fallback, never a crash and
    never a silent empty success.
    """
    for payload in captured:
        title, blocks = parse_article_content_state(payload)
        if blocks:
            return ContentSourceSuccess(
                kind="x_article",
                url=url,
                title=title,
                text=_flatten_blocks(blocks),
                blocks=blocks,
                http_status=200,
                attempts=1,
            )
    return None


def _fetch_rendered(context: BrowserContext, url: str) -> ContentSource:
    """Fetch an X article (or other x.com page).

    For an article URL, intercept the article-content GraphQL response (the same
    `page.on("response", …)` pattern `_fetch_tweet` uses for `TweetDetail`) and
    build an ordered `blocks` body. On any interception/parse miss, fall back to
    `trafilatura.extract(html)` — the text-only behaviour retained from before.
    """
    is_article = _classify_x_url(url) == "article"
    captured: list[dict] = []
    page = context.new_page()

    if is_article:

        def on_response(response: Response) -> None:
            if _is_article_graphql(response.url):
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
            return ContentSourceFailure(
                kind="x_article",
                url=url,
                failure_reason="timeout",
                error="No se pudo cargar el artículo de X.",
                attempts=1,
            )
        if is_logged_out(page.url):
            raise RuntimeError("Sesión de X caducada. Ejecuta `xbrain login`.")
        html = page.content()
    finally:
        page.close()

    if is_article:
        structured = _structured_article(captured, url)
        if structured is not None:
            return structured

    text = trafilatura.extract(html)
    if text:
        return ContentSourceSuccess(
            kind="x_article", url=url, text=text, http_status=200, attempts=1
        )
    return ContentSourceFailure(
        kind="x_article",
        url=url,
        failure_reason="empty_content",
        error="No se pudo extraer el contenido del artículo de X.",
        attempts=1,
    )


def _fetch_x_link(context: BrowserContext, url: str) -> ContentSource:
    """Fetch one x.com link, routed by URL kind."""
    if _classify_x_url(url) == "status":
        return _fetch_tweet(context, url)
    return _fetch_rendered(context, url)


def _needs_x_fetch(
    item: Item,
    force: bool,
    since: datetime | None = None,
    until: datetime | None = None,
) -> bool:
    if not any(is_x_url(link.url) for link in item.links):
        return False
    if since and item.created_at < since:
        return False
    if until and item.created_at > until:
        return False
    if force or item.content is None:
        return True
    return not any(s.kind == "x_article" for s in item.content.sources)


def _attach_x_sources(item: Item, sources: list[ContentSource]) -> None:
    """Replace the item's `x_article` sources, keeping every other kind.

    Advances `content.fetched_at` only on a MATERIAL change to the `x_article`
    source set (a first structured body, or a text change) — so the
    `enrich._needs_reenrichment` trigger fires when the body actually changed,
    while an idempotent re-fetch produces no LLM churn. This mirrors
    `fetch.fetch_item` and reuses its material fingerprint (`_sources_materially_equal`,
    a model-derived deny-list) rather than reimplementing it (#39 PR3).
    """
    now = datetime.now(timezone.utc)
    if item.content is None:
        item.content = Content(fetched_at=now, sources=list(sources))
        return
    old_x_sources = [s for s in item.content.sources if s.kind == "x_article"]
    kept = [s for s in item.content.sources if s.kind != "x_article"]
    item.content.sources = kept + list(sources)
    if not _sources_materially_equal(old_x_sources, list(sources)):
        item.content.fetched_at = now


def fetch_x_articles(
    store: dict[str, Item],
    storage_state_path: Path | None,
    force: bool = False,
    since: datetime | None = None,
    until: datetime | None = None,
    *,
    headless: bool = False,
    link_fetcher: LinkFetcher | None = None,
) -> int:
    """Fetch x.com link content for every item that has one and needs it.

    `link_fetcher` is injected by tests; in production it is bound to a live
    Playwright context opened from `storage_state_path`. `since`/`until` apply
    the same `created_at` date-window filter as `fetch.fetch_pending`.
    """
    pending = [item for item in store.values() if _needs_x_fetch(item, force, since, until)]
    if not pending:
        return 0

    def _process(fetcher: LinkFetcher) -> None:
        for item in pending:
            # Dedup x.com URLs so a repeated link yields a single source.
            urls = dict.fromkeys(link.url for link in item.links if is_x_url(link.url))
            sources = [fetcher(url) for url in urls]
            _attach_x_sources(item, sources)

    if link_fetcher is not None:
        _process(link_fetcher)
    else:
        if storage_state_path is None:
            raise ValueError("storage_state_path is required without an injected link_fetcher")
        with x_context(storage_state_path, headless=headless) as context:
            _process(lambda url: _fetch_x_link(context, url))
    return len(pending)
