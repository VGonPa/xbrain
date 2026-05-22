"""Fetch the external content linked from items.

External (non-x.com) links are extracted with trafilatura; a failed fetch
records *structured evidence* (HTTP status + a categorised reason) so a broken
link is demonstrable, not assumed (design §4). x.com links are skipped here —
they are handled by `xbrain.fetch_x`.

`FetchResult` is the in-memory return type of the extractors and is a tagged
union (`FetchSuccess | FetchFailure`) so callers cannot accidentally read a
success-only field off a failure record (or vice versa). The persisted
`ContentSource` is itself a tagged union — see `xbrain.models`.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Callable, Union
from urllib.parse import urlparse

import trafilatura
from pydantic import BaseModel

from xbrain.models import (
    Content,
    ContentSource,
    ContentSourceFailure,
    ContentSourceSuccess,
    FailureReason,
    Item,
)

_X_HOSTS = {"x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"}
_UA = "Mozilla/5.0 (compatible; XBrain/1.0)"
_TIMEOUT = 20
_FIRECRAWL_URL = "https://api.firecrawl.dev/v1/scrape"
_FIRECRAWL_TIMEOUT = 60


class FetchSuccess(BaseModel):
    """The successful outcome of one content-extraction attempt.

    `text` is required: a success without text is not a success. The type
    system enforces this — callers do not need to defensively check for
    `result.text is not None`.
    """

    title: str | None = None
    text: str
    http_status: int | None = None
    attempts: int = 1


class FetchFailure(BaseModel):
    """The failed outcome of one content-extraction attempt.

    `failure_reason` may be `None` when the failure has not yet been
    categorised (e.g. an uncaught network error). Callers that persist the
    result fall back to ``"unknown_error"`` — a transient bucket — to satisfy
    the required field on `ContentSourceFailure` while preserving the
    "uncategorised = retry-worthy" invariant from #19.
    """

    failure_reason: FailureReason | None = None
    error: str | None = None
    http_status: int | None = None
    attempts: int = 1


# The in-memory FetchResult — internal to this module. Never persisted (the
# wire format is `ContentSource`, which has its own discriminator). Callers
# narrow via `isinstance(result, FetchSuccess)` / `isinstance(result, FetchFailure)`.
FetchResult = Union[FetchSuccess, FetchFailure]
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
        return FetchFailure(http_status=status, failure_reason=reason, error=error, attempts=1)
    text = extract(downloaded)
    if not text:
        return FetchFailure(
            http_status=200,
            failure_reason="empty_content",
            error="La página se descargó pero no tiene un artículo extraíble.",
            attempts=1,
        )
    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title if metadata else None
    return FetchSuccess(title=title, text=text, http_status=200, attempts=1)


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
        return FetchFailure(error=f"Firecrawl falló: {exc}", attempts=1)
    if payload.get("success") is False or payload.get("error"):
        return FetchFailure(
            error=f"Firecrawl falló: {payload.get('error') or 'respuesta de error'}",
            attempts=1,
        )
    data = payload.get("data") or {}
    text = data.get("markdown")
    if text:
        title = (data.get("metadata") or {}).get("title")
        return FetchSuccess(title=title, text=text, http_status=200, attempts=1)
    return FetchFailure(
        failure_reason="empty_content",
        error="Firecrawl no devolvió contenido.",
        attempts=1,
    )


def extract_article(
    url: str,
    *,
    primary: Callable = trafilatura_extract,
    firecrawl: Callable = _firecrawl_extract,
) -> FetchResult:
    """Default extractor: trafilatura, then Firecrawl for JS-rendered pages.

    Switches on the `FetchResult` variant: a `FetchSuccess` is returned
    as-is. A `FetchFailure` whose `failure_reason` is in
    ``{"js_required", "empty_content"}`` triggers the Firecrawl fallback;
    anything else (404, dns, ...) is terminal and Firecrawl is not called.
    """
    result = primary(url)
    if isinstance(result, FetchSuccess):
        return result
    # mypy now narrows `result` to FetchFailure for the rest of this function.
    if result.failure_reason not in ("js_required", "empty_content"):
        return result  # hard failure (404, dns, ...) — Firecrawl will not help
    fallback = firecrawl(url)
    if fallback is None:
        return result  # Firecrawl not configured — keep the original evidence
    if isinstance(fallback, FetchSuccess):
        return fallback.model_copy(update={"attempts": result.attempts + 1})
    # Both attempts exhausted — merge evidence and bump the attempt counter.
    merged_error = (
        f"{result.error or 'sin contenido'} | Firecrawl: {fallback.error}"
        if fallback.error
        else result.error
    )
    return result.model_copy(update={"attempts": result.attempts + 1, "error": merged_error})


def _content_source_from(url: str, result: FetchResult) -> ContentSource:
    """Build the persisted `ContentSource` variant matching the fetch outcome.

    Maps `FetchSuccess` → `ContentSourceSuccess` and `FetchFailure` →
    `ContentSourceFailure`. A failure without a categorised reason falls back
    to ``"empty_content"`` — the catch-all bucket — because
    `ContentSourceFailure.failure_reason` is a required field (the type
    system says a failure without a reason is not demonstrable evidence).
    The free-form `error` string still carries the original explanation.
    """
    if isinstance(result, FetchSuccess):
        return ContentSourceSuccess(
            kind="external_article",
            url=url,
            title=result.title,
            text=result.text,
            http_status=result.http_status,
            attempts=result.attempts,
        )
    return ContentSourceFailure(
        kind="external_article",
        url=url,
        error=result.error,
        http_status=result.http_status,
        # `unknown_error` (a transient bucket) — NOT `empty_content` (terminal).
        # An uncategorised failure must stay self-healing on the next
        # `fetch_pending` run, mirroring the #19 invariant that a missing
        # `failure_reason` was retry-worthy by default.
        failure_reason=result.failure_reason or "unknown_error",
        attempts=result.attempts,
    )


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
                ContentSourceFailure(
                    kind="external_article",
                    url=url,
                    # Transient bucket: an uncaught extractor exception is
                    # almost always something that may succeed on the next
                    # run (network blip, intermittent SSL, …). #19 relied on
                    # this being retry-worthy; preserve that.
                    failure_reason="unknown_error",
                    error=f"Error al descargar el artículo: {exc}",
                    attempts=1,
                )
            )
            continue
        new_sources.append(_content_source_from(url, result))
    kept = [s for s in item.content.sources if s.kind != "external_article"] if item.content else []
    return Content(fetched_at=datetime.now(timezone.utc), sources=kept + new_sources)


# Failure reasons that justify an automatic retry on the next run. Everything
# else (`not_found`, `forbidden`, `paywall`, `js_required`, `empty_content`)
# is treated as terminal — only `--force` re-fetches those.
#
# `unknown_error` is the uncategorised-failure bucket — an extractor exception
# or any failure path that did not pin a specific reason. We retry by default:
# the alternative is silently classifying every uncaught failure as terminal,
# which would break the #19 invariant ("a failure without a categorised reason
# is anomalous; re-fetching gives it a chance to land on a known result").
_TRANSIENT_FAILURES: frozenset[FailureReason] = frozenset({"timeout", "dns_error", "unknown_error"})


def _should_refetch(content: Content | None, force: bool) -> bool:
    """Return True if `fetch_pending` should (re)fetch this item.

    - `content is None` (never fetched) → True.
    - `force=True` → True regardless of recorded state.
    - Otherwise, True only if every `external_article` source on `content`
      is a `ContentSourceFailure` whose `failure_reason` is in
      `_TRANSIENT_FAILURES`. A single successful source (any
      `ContentSourceSuccess`) or a categorised terminal failure (e.g.
      `not_found`, `paywall`) skips. No `external_article` sources at all
      → skip (there is nothing here for `fetch_pending`; the x.com sources
      are handled by `fetch_x`).

    The pre-#20 helper read `src.ok` / `src.failure_reason` as Optionals;
    after the tagged-union refactor the variant `isinstance` check is the
    single switch and `failure_reason` is required on the failure variant,
    so no `is None` special-case is needed. Pre-#20 records that lacked a
    categorised `failure_reason` are migrated to `timeout` by
    `_normalise_legacy_content_source` (see `xbrain.models`) — they
    therefore get one automatic retry under #19, matching the prior
    behaviour without an extra branch here.
    """
    if content is None:
        return True
    if force:
        return True
    external = [s for s in content.sources if s.kind == "external_article"]
    if not external:
        return False
    return all(
        isinstance(src, ContentSourceFailure) and src.failure_reason in _TRANSIENT_FAILURES
        for src in external
    )


def fetch_pending(
    store: dict[str, Item],
    since: datetime | None = None,
    until: datetime | None = None,
    force: bool = False,
    extractor: ArticleExtractor = extract_article,
) -> int:
    """Fetch external content for items that have non-x links and no content yet.

    A previous fetch whose only failures were transient (timeout, dns_error)
    is automatically retried — `--force` is only needed to retry terminal
    failures (404, paywall, …) or to re-hit already-successful items.
    """
    fetched = 0
    for item in store.values():
        # all links are x.com — handled by fetch_x
        if all(is_x_url(link.url) for link in item.links):
            continue
        if not _should_refetch(item.content, force):
            continue
        if since and item.created_at < since:
            continue
        if until and item.created_at > until:
            continue
        item.content = fetch_item(item, extractor)
        fetched += 1
    return fetched
