"""Fetch the external content linked from items.

External (non-x.com) links are extracted with trafilatura; a failed fetch
records *structured evidence* (HTTP status + a categorised reason) so a broken
link is demonstrable, not assumed (design §4). x.com links are skipped here —
they are handled by `xbrain.fetch_x`.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import urlparse

import trafilatura

from xbrain.models import Content, ContentSource, FailureReason, Item

_X_HOSTS = {"x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"}
_UA = "Mozilla/5.0 (compatible; XBrain/1.0)"
_TIMEOUT = 20
_FIRECRAWL_URL = "https://api.firecrawl.dev/v1/scrape"
_FIRECRAWL_TIMEOUT = 60


@dataclass
class FetchResult:
    """The structured outcome of one content-extraction attempt."""

    title: str | None = None
    text: str | None = None
    http_status: int | None = None
    failure_reason: FailureReason | None = None
    error: str | None = None
    attempts: int = 1


ArticleExtractor = Callable[[str], FetchResult]


def is_x_url(url: str) -> bool:
    """True for a link whose host is x.com / twitter.com."""
    return (urlparse(url).hostname or "").lower() in _X_HOSTS


def _reason_for_status(code: int) -> FailureReason | None:
    """Map an HTTP status code to a failure category (None = not categorised)."""
    if code in (404, 410):
        return "not_found"
    if code in (401, 403):
        return "forbidden"
    if code in (402, 451):
        return "paywall"
    return None


def _categorize_url_error(exc: urllib.error.URLError) -> FailureReason | None:
    """Classify a network-level URLError as timeout / dns_error / uncategorised."""
    reason = getattr(exc, "reason", None)
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(reason, socket.gaierror):
        return "dns_error"
    return None


def _probe_status(
    url: str, opener: Callable = urllib.request.urlopen
) -> tuple[int | None, FailureReason | None, str]:
    """Probe a URL trafilatura could not download, to categorise the failure.

    Returns `(http_status, failure_reason, error_text)`. A *successful* probe
    means the page is reachable but trafilatura found no article — most likely
    JavaScript-rendered, so the reason is `js_required`.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with opener(req, timeout=_TIMEOUT) as resp:
            status = getattr(resp, "status", None) or 200
        return (
            status,
            "js_required",
            "Descargable pero sin artículo extraíble (posible JavaScript).",
        )
    except urllib.error.HTTPError as exc:
        return exc.code, _reason_for_status(exc.code), f"HTTP {exc.code}: {exc.reason}"
    except urllib.error.URLError as exc:
        return None, _categorize_url_error(exc), f"Error de red: {exc.reason}"
    except (TimeoutError, socket.timeout) as exc:
        return None, "timeout", f"Tiempo de espera agotado: {exc}"


def trafilatura_extract(
    url: str,
    *,
    fetch: Callable = trafilatura.fetch_url,
    extract: Callable = trafilatura.extract,
    prober: Callable = _probe_status,
) -> FetchResult:
    """Extract an article with trafilatura, categorising any failure."""
    downloaded = fetch(url)
    if downloaded is None:
        status, reason, error = prober(url)
        return FetchResult(http_status=status, failure_reason=reason, error=error, attempts=1)
    text = extract(downloaded)
    if not text:
        return FetchResult(
            http_status=200,
            failure_reason="empty_content",
            error="La página se descargó pero no tiene un artículo extraíble.",
            attempts=1,
        )
    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title if metadata else None
    return FetchResult(title=title, text=text, http_status=200, attempts=1)


def _firecrawl_extract(
    url: str, *, opener: Callable = urllib.request.urlopen
) -> FetchResult | None:
    """Second-attempt extraction via Firecrawl (renders JavaScript).

    Returns `None` when `FIRECRAWL_API_KEY` is unset — the fallback is optional,
    so XBrain users without a key simply do not get it.
    """
    key = os.environ.get("FIRECRAWL_API_KEY")
    if not key:
        return None
    body = json.dumps({"url": url, "formats": ["markdown"], "onlyMainContent": True}).encode()
    req = urllib.request.Request(
        _FIRECRAWL_URL,
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with opener(req, timeout=_FIRECRAWL_TIMEOUT) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return FetchResult(error=f"Firecrawl falló: {exc}", attempts=1)
    if payload.get("success") is False or payload.get("error"):
        return FetchResult(
            error=f"Firecrawl falló: {payload.get('error') or 'respuesta de error'}",
            attempts=1,
        )
    data = payload.get("data") or {}
    text = data.get("markdown")
    if text:
        title = (data.get("metadata") or {}).get("title")
        return FetchResult(title=title, text=text, http_status=200, attempts=1)
    return FetchResult(
        failure_reason="empty_content", error="Firecrawl no devolvió contenido.", attempts=1
    )


def extract_article(
    url: str,
    *,
    primary: Callable = trafilatura_extract,
    firecrawl: Callable = _firecrawl_extract,
) -> FetchResult:
    """Default extractor: trafilatura, then Firecrawl for JS-rendered pages."""
    result = primary(url)
    if result.text:
        return result
    if result.failure_reason not in ("js_required", "empty_content"):
        return result  # a hard failure (404, dns, ...) — Firecrawl will not help
    fallback = firecrawl(url)
    if fallback is None:
        return result  # Firecrawl not configured — keep the original evidence
    if fallback.text:
        fallback.attempts = result.attempts + 1
        return fallback
    result.attempts = result.attempts + 1  # both attempts exhausted; keep first evidence
    if fallback.error:
        result.error = f"{result.error or 'sin contenido'} | Firecrawl: {fallback.error}"
    return result


def fetch_item(item: Item, extractor: ArticleExtractor = extract_article) -> Content:
    """Build/refresh the `external_article` sources of an item.

    x.com links are skipped (see `xbrain.fetch_x`). Only `external_article`
    sources are rebuilt; every other source kind already on the item is
    preserved across a re-fetch.
    """
    new_sources: list[ContentSource] = []
    # Dedup: an item whose links repeat a URL must not yield duplicate sources.
    non_x_urls = dict.fromkeys(link.url for link in item.links if not is_x_url(link.url))
    for url in non_x_urls:
        try:
            result = extractor(url)
        except Exception as exc:  # noqa: BLE001 - one bad URL must not abort the batch
            new_sources.append(
                ContentSource(
                    kind="external_article",
                    url=url,
                    ok=False,
                    error=f"Error al descargar el artículo: {exc}",
                    attempts=1,
                )
            )
            continue
        if result.text:
            new_sources.append(
                ContentSource(
                    kind="external_article",
                    url=url,
                    title=result.title,
                    text=result.text,
                    ok=True,
                    http_status=result.http_status,
                    attempts=result.attempts,
                )
            )
        else:
            new_sources.append(
                ContentSource(
                    kind="external_article",
                    url=url,
                    ok=False,
                    error=result.error,
                    http_status=result.http_status,
                    failure_reason=result.failure_reason,
                    attempts=result.attempts,
                )
            )
    kept = [s for s in item.content.sources if s.kind != "external_article"] if item.content else []
    return Content(fetched_at=datetime.now(timezone.utc), sources=kept + new_sources)


def fetch_pending(
    store: dict[str, Item],
    since: datetime | None = None,
    until: datetime | None = None,
    force: bool = False,
    extractor: ArticleExtractor = extract_article,
) -> int:
    """Fetch external content for items that have non-x links and no content yet."""
    fetched = 0
    for item in store.values():
        # all links are x.com — handled by fetch_x
        if all(is_x_url(link.url) for link in item.links):
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
