"""Fetch the external content linked from items.

External (non-x.com) links are extracted with trafilatura; a failed fetch
records *structured evidence* (HTTP status + a categorised reason) so a broken
link is demonstrable, not assumed (design §4). x.com links are skipped here —
they are handled by `xbrain.fetch_x`.

`FetchResult` is the in-memory return type of the extractors and is a tagged
union (`FetchSuccess | FetchFailure`) so callers cannot accidentally read a
success-only field off a failure record (or vice versa). The persisted
`ContentSource` is itself a tagged union — see `xbrain.models`.

**A non-empty body is not an article.** For a long time the only content check was
`if not text` — non-empty ⇒ success — and that is how YouTube's footer menu, a Cloudflare
challenge and a bare page title became `[Linked article]` evidence: 28 of the store's 189
fetched "articles" (14.8%), measured. The guardrail cannot fire for them (they are recorded
as successes), so the generator was ordered to "summarise the article's substance" while
being handed a footer menu. `validate_body` is the fix, and it sits at the PERSISTENCE
boundary (`_safe_extract`), so the ordinary `fetch` path is covered — not just the retry.

The wall detector is a heuristic, and its bias is deliberate and MEASURED, not assumed:

    Rejecting a good article merely leaves the honest failure we already had — the guardrail
    fires, the generator is told nothing is known, and nothing is lost but an opportunity.
    ACCEPTING a wall poisons the evidence: it silences the guardrail, grounds an entity in a
    cookie banner, and hands the judge a `[Linked article]` it will PASS.

The costs are asymmetric, so the bias is to over-reject. Tuned against the real corpus rather
than guessed: over its 189 successfully-fetched articles it rejects 28, and all 28 are junk —
**zero false rejects**. (An earlier list used a bare `"log in to"`, which is ordinary English
prose, and it wrongly rejected three real bodies — two GitHub READMEs and a docs page. A marker
that fires on prose is not a wall detector, it is a coin flip. Hence `_WALL_MARKERS` vs
`_CHROME_MARKERS`.)
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from collections import Counter
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

# Pacing between items in a `--retry-failed` run. These are other people's servers, and the
# corpus's one transient failure is a recorded HTTP 429 — a host that already rate-limited us.
_RETRY_DELAY_SECONDS = 1.0


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
    # The extractor could not be REACHED at all (network error, 503, rate-limit) — as opposed
    # to running and finding nothing. Never persisted; `extract_article` reads it to decide
    # whether the attempt counts. A transient outage must not permanently burn an item's retry.
    transport_error: bool = False


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
        return FetchFailure(error=f"Firecrawl falló: {exc}", attempts=1, transport_error=True)
    if payload.get("success") is False or payload.get("error"):
        return FetchFailure(
            error=f"Firecrawl falló: {payload.get('error') or 'respuesta de error'}",
            attempts=1,
            transport_error=True,  # it refused (5xx / 429) — it did not judge the page
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


# An article body shorter than this is not an article. Chrome, a paywall teaser or a consent
# shell lands here; a real piece does not. Failing CLOSED is the safe direction — a rejected
# body becomes a recorded failure, which keeps the guardrail firing, while a wrongly ACCEPTED
# body becomes evidence the judge will trust.
_MIN_ARTICLE_CHARS = 300

# WALL phrases: a page containing one of these IS a wall. Measured against the corpus's 189
# successfully-fetched articles — none of them contains any of these. Rejecting on a single hit
# is therefore safe.
#
# Deliberately specific. A bare "log in to" was the FIRST version of this list and it rejected
# three genuinely good bodies — the `open-wearables` README ("Log in to the developer portal"),
# this repo's own README ("log in to X in Chrome first") — because that phrase is ordinary
# English prose. A marker that fires on prose is not a wall detector, it is a coin flip.
_WALL_MARKERS = (
    "accept all cookies",
    "we value your privacy",
    "we and our partners store",
    "sign in to continue",
    "log in to continue",
    "log in to see",
    "log in to view",
    "log in or sign up",
    "create an account to see",
    "subscribe to continue",
    "enable javascript",
    "turn on javascript",
    "verify you are human",
    "checking your browser",
    "are you a robot",
    "limited support for css",
    "turn off compatibility mode",
    "doesn't work on your browser",
    "does not work on your browser",
)

# CHROME phrases: page furniture — a footer, a cookie line. A REAL article page can legitimately
# carry ONE (the Claude docs page prepends a cookie banner to genuine documentation; measured).
# But when several co-occur we did not capture the article, we captured the page furniture — and
# that body is not evidence, it is a surface a model can mine for claims the article never made.
# So: one is tolerated, TWO OR MORE is a rejection.
_CHROME_MARKERS = (
    "skip to main content",
    "cookie policy",
    "privacy policy",
    "manage preferences",
    "accept cookies",
    "your privacy",
)
_MAX_TOLERATED_CHROME = 1


def _interstitial_marker(text: str) -> str | None:
    """The wall evidence this body carries, if any — a wall phrase, or a pile-up of page chrome.

    ASYMMETRIC ON PURPOSE, and the direction matters: rejecting a good article merely leaves the
    honest failure we already had (the guardrail fires, the generator is told nothing is known,
    nothing is lost but an opportunity). ACCEPTING a wall poisons the evidence — it silences the
    guardrail, grounds an entity in a cookie banner, and hands the judge a `[Linked article]` it
    will PASS. So when in doubt, reject. The thresholds below are nevertheless tuned against the
    real corpus rather than guessed: they reject 0 of its 189 successfully-fetched articles.
    """
    lowered = text.lower()
    wall = next((m for m in _WALL_MARKERS if m in lowered), None)
    if wall:
        return wall
    chrome = [m for m in _CHROME_MARKERS if m in lowered]
    if len(chrome) > _MAX_TOLERATED_CHROME:
        return " + ".join(chrome)
    return None


def validate_body(url: str, result: FetchResult) -> FetchResult:
    """Refuse to call a cookie/login wall — or a body too thin to be an article — a SUCCESS.

    The only content check used to be `if not text`: non-empty ⇒ success. Firecrawl RENDERS
    JavaScript, and `js_required` means "downloadable but no extractable article" — the most
    enriched population there is for consent shells and SPA login walls. So a retry would very
    often "succeed" on exactly those pages and hand back the banner markdown.

    That is strictly worse than the honest failure it replaces: flipping the source to success
    makes `links_content_unfetched` False, which DELETES the `[Links — content NOT fetched]`
    block from all three LLM surfaces, whereupon `rubric-summary` orders the generator to
    summarise "the article's substance" and the judge sees a `[Linked article]` and trusts it.
    A rendered Instagram login wall even contains the word "Instagram", so the entity checker
    would call the entity GROUNDED. A wall is recorded as a failure, with a true cause.
    """
    if not isinstance(result, FetchSuccess):
        return result
    marker = _interstitial_marker(result.text)
    if marker:
        return FetchFailure(
            failure_reason="blocked_interstitial",
            error=f"La página devolvió un muro (cookies/login), no un artículo: {marker!r}.",
            http_status=result.http_status,
            attempts=result.attempts,
        )
    if len(result.text.strip()) < _MIN_ARTICLE_CHARS:
        return FetchFailure(
            failure_reason="blocked_interstitial",
            error=(
                f"El cuerpo extraído ({len(result.text.strip())} caracteres) es demasiado "
                "corto para ser un artículo."
            ),
            http_status=result.http_status,
            attempts=result.attempts,
        )
    if _title_is_bare_domain(url, result.title):
        return FetchFailure(
            failure_reason="blocked_interstitial",
            error=f"El título extraído ({result.title!r}) es el dominio, no un artículo.",
            http_status=result.http_status,
            attempts=result.attempts,
        )
    return result


def _title_is_bare_domain(url: str, title: str | None) -> bool:
    """A page whose whole title is its own domain ("Instagram", "twitch.tv") is a wall or a
    landing page, not an article."""
    if not title:
        return False
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    normalized = title.strip().lower()
    return normalized in {host, host.split(".")[0], host.removesuffix(".com")}


def extract_article(
    url: str,
    *,
    primary: Callable = trafilatura_extract,
    firecrawl: Callable = _firecrawl_extract,
) -> FetchResult:
    """Default extractor: trafilatura, then Firecrawl for JS-rendered pages.

    Switches on the `FetchResult` variant: a `FetchSuccess` is returned
    as-is. A `FetchFailure` whose `failure_reason` is in `_FALLBACK_ELIGIBLE`
    triggers the Firecrawl fallback; anything else (404, dns, ...) is terminal
    and Firecrawl is not called.

    Every accepted body passes `validate_body` first — a consent/login wall or a body too thin
    to be an article is recorded as a `blocked_interstitial` FAILURE, never as evidence. A wall
    seen by TRAFILATURA still escalates (Firecrawl may reach the article behind it); a wall
    seen by FIRECRAWL is the end of the road, and `attempts=2` then keeps it there.

    A Firecrawl TRANSPORT error (unreachable, 503, rate-limited) does NOT count as an attempt:
    the fallback never ran, so burning the item's one retry on an outage would strand it
    forever — `--retry-failed` would never select it again and the dry run would cheerfully
    report "Reintentables: 0".
    """
    result = validate_body(url, primary(url))
    if isinstance(result, FetchSuccess):
        return result
    # mypy now narrows `result` to FetchFailure for the rest of this function.
    if result.failure_reason not in _FALLBACK_ELIGIBLE:
        return result  # hard failure (404, dns, ...) — Firecrawl will not help
    fallback = firecrawl(url)
    if fallback is None:
        return result  # Firecrawl not configured — keep the original evidence
    if isinstance(fallback, FetchFailure) and fallback.transport_error:
        return result  # Firecrawl never ran — do NOT burn the retry on an outage
    fallback = validate_body(url, fallback)
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


def _utcnow() -> datetime:
    """The default `fetch_item` / `fetch_pending` clock (UTC-aware, injectable)."""
    return datetime.now(timezone.utc)


# Fields that are fetch *bookkeeping*, not material content: they can churn across
# a re-fetch that produced no real change (a trafilatura→Firecrawl retry bumps
# `attempts`; the free-form `error` string can be reworded). Everything else on a
# `ContentSource` is treated as material. This is a DENY-list on purpose — it is
# derived from the model, so a NEW content field (e.g. a future `summary`) is
# fingerprinted automatically instead of being silently dropped by a stale
# allow-list. It fails safe: forgetting to exclude a bookkeeping field costs at
# most one redundant re-enrichment, never a lost (stale-content) one.
_BOOKKEEPING_FIELDS = {"attempts", "error"}


def _source_signature(source: ContentSource) -> str:
    """A *material-content* fingerprint of one source: the whole model minus
    fetch bookkeeping (`attempts`/`error`).

    Two sources with equal signatures carry the same material content even if a
    re-fetch churned their bookkeeping. Deriving the fingerprint from the model
    (rather than a hand-picked field list) means every content-bearing field —
    `title`, `text`, `failure_reason`, `http_status`, the `x_video`
    transcript/`frames`, … — is compared automatically and safely. `exclude`ing a
    field absent on a given variant is a harmless no-op.
    """
    return source.model_dump_json(exclude=_BOOKKEEPING_FIELDS)


def _sources_materially_equal(old: list[ContentSource], new: list[ContentSource]) -> bool:
    """True when two source sets carry the same material content (order-insensitive)."""
    return Counter(map(_source_signature, old)) == Counter(map(_source_signature, new))


def _build_external_sources(item: Item, extractor: ArticleExtractor) -> list[ContentSource]:
    """Extract the `external_article` sources for an item's non-x links.

    Dedups repeated URLs and isolates a per-URL extractor exception as a
    transient (`unknown_error`) failure so one bad link cannot abort the batch.
    """
    sources: list[ContentSource] = []
    non_x_urls = dict.fromkeys(link.url for link in item.links if not is_x_url(link.url))
    for url in non_x_urls:
        sources.append(_content_source_from(url, _safe_extract(url, extractor)))
    return sources


def _safe_extract(url: str, extractor: ArticleExtractor) -> FetchResult:
    """Extract one URL, VALIDATED, with a per-URL exception isolated as a transient failure.

    Validation happens at this persistence boundary, not just inside `extract_article`: no
    extractor — injected, custom, or future — may write a cookie/login wall into the store as
    evidence. `validate_body` is idempotent on a failure, so the default path (which already
    validated) passes straight through.
    """
    try:
        return validate_body(url, extractor(url))
    except Exception as exc:  # noqa: BLE001 - one bad URL must not abort the batch
        return FetchFailure(
            failure_reason="unknown_error",
            error=f"Extractor falló: {exc}",
            attempts=1,
        )


def fetch_item(
    item: Item,
    extractor: ArticleExtractor = extract_article,
    *,
    now: Callable[[], datetime] = _utcnow,
) -> Content:
    """Build/refresh the `external_article` sources of an item.

    x.com links are skipped (see `xbrain.fetch_x`). Only `external_article`
    sources are rebuilt; every other source kind already on the item is
    preserved across a re-fetch.

    **`fetched_at` advances only on a material content change (#44 data-safety).**
    `fetch_pending` re-fetches a persistently-failing *transient* link on every
    run (its refetch decision keys on source STATE, not on `fetched_at`). If each
    identical re-fetch bumped `fetched_at`, `enrich._needs_reenrichment` would
    re-flag the item forever — one wasted, identical LLM call per stuck item per
    cycle. So when the re-fetched source set is *materially equivalent* to the
    existing one — the whole source model minus fetch bookkeeping
    (`attempts`/`error`); see `_source_signature` — we keep the prior
    `fetched_at`; it advances only when the content actually changed (a failure
    that becomes a success, new/edited text, a changed title, a changed failure
    reason/status). `now` is injectable for deterministic tests.
    """
    new_sources = _build_external_sources(item, extractor)
    kept = [s for s in item.content.sources if s.kind != "external_article"] if item.content else []
    sources = kept + new_sources
    # Preserve the prior timestamp on a no-material-change re-fetch (see docstring);
    # only stamp a fresh `now()` when the content set actually changed.
    if item.content is not None and _sources_materially_equal(item.content.sources, sources):
        return Content(fetched_at=item.content.fetched_at, sources=sources)
    return Content(fetched_at=now(), sources=sources)


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

# The two reasons `extract_article` escalates to the Firecrawl fallback. They are NOT in
# `_TRANSIENT_FAILURES`, so `_should_refetch` treats them as terminal and never retries them.
# That is right only while trafilatura is the whole pipeline — the moment a JS-capable
# extractor is available they become the most recoverable bucket there is.
_FALLBACK_ELIGIBLE: frozenset[FailureReason] = frozenset(
    {"js_required", "empty_content", "blocked_interstitial"}
)

# trafilatura = 1 attempt, + Firecrawl = 2 (see `ContentSourceFailure.attempts`).
_BOTH_EXTRACTORS_TRIED = 2


def _retryable_now(src: ContentSourceFailure, *, firecrawl_configured: bool) -> bool:
    """Could re-fetching this failed source plausibly produce a DIFFERENT outcome today?

    Two ways it can, and one way it cannot:

    - a TRANSIENT failure (timeout, dns, and the `unknown_error` bucket that catches HTTP 429)
      may simply succeed on a retry — same extractor, better day;
    - a FALLBACK-ELIGIBLE failure (`js_required`, `empty_content`) recorded at `attempts=1`
      never actually got the Firecrawl pass: `_firecrawl_extract` returns None when
      `FIRECRAWL_API_KEY` is unset, and `extract_article` then keeps the original failure. With
      the key configured, the retry brings a genuinely different extractor;
    - anything else (404, 403, paywall) is not an extraction problem, and a fallback-eligible
      failure already at `attempts=2` has had both extractors. Retrying either one only
      reproduces the recorded failure — that is not a repair, it is load on someone's server.
    """
    if src.failure_reason in _TRANSIENT_FAILURES:
        return True
    return (
        firecrawl_configured
        and src.failure_reason in _FALLBACK_ELIGIBLE
        and src.attempts < _BOTH_EXTRACTORS_TRIED
    )


def firecrawl_available() -> bool:
    """Is the JS-capable fallback extractor configured? (Mirrors `_firecrawl_extract`'s check —
    one source of truth for "can a retry bring a different extractor".)"""
    return bool(os.environ.get("FIRECRAWL_API_KEY"))


def should_retry_failed(content: Content | None, *, firecrawl_configured: bool) -> bool:
    """Select an item for `fetch --retry-failed`: does it carry a link failure a retry could
    actually repair?

    Unlike `_should_refetch` this is ANY, not ALL: a PARTIAL fetch (one link fetched, one
    still failing) is still missing evidence for the failed link, and the fetched article is
    no evidence for it. `content is None` is excluded — a never-fetched item is the plain
    `fetch` path's job, and `--retry-failed` must not silently widen into a general backfill.
    """
    if content is None:
        return False
    return any(
        isinstance(src, ContentSourceFailure)
        and src.kind == "external_article"
        and _retryable_now(src, firecrawl_configured=firecrawl_configured)
        for src in content.sources
    )


def _should_refetch(item: Item, force: bool) -> bool:
    """Return True if `fetch_pending` should (re)fetch this item.

    - `item.content is None` (never fetched) → True.
    - `force=True` → True regardless of recorded state.
    - No `external_article` sources, but the item HAS a non-x link → True: the link
      fetch never ran (see below).
    - Otherwise, True only if every `external_article` source is a
      `ContentSourceFailure` whose `failure_reason` is in `_TRANSIENT_FAILURES`. A
      single successful source (any `ContentSourceSuccess`) or a categorised terminal
      failure (e.g. `not_found`, `paywall`) skips.

    **Why "content exists" is NOT "the links were fetched".** Other stages attach
    sources of their own kind — the `extract` stage now stamps the quoted post, and
    `digest-video` / `expand_threads` attach transcripts and threads. Such an item
    arrives here with `content` set and NO `external_article` source. Reading that as
    "already fetched" would silently never download its article — for the 35% of the
    corpus that quotes, plus every threaded or video item. So the question is asked of
    the LINKS, not of the mere presence of content: `_build_external_sources` emits one
    source per non-x link, so "has a non-x link but no `external_article` source" can
    only mean the fetch has not run yet. Once it does, the sources exist and the
    transient-failure rule below governs — no re-fetch loop.

    The pre-#20 helper read `src.ok` / `src.failure_reason` as Optionals; after the
    tagged-union refactor the variant `isinstance` check is the single switch and
    `failure_reason` is required on the failure variant, so no `is None` special-case
    is needed. Pre-#20 records that lacked a categorised `failure_reason` are migrated
    to `timeout` by `_normalise_legacy_content_source` (see `xbrain.models`) — they
    therefore get one automatic retry under #19, matching the prior behaviour without
    an extra branch here.
    """
    if item.content is None:
        return True
    if force:
        return True
    external = [s for s in item.content.sources if s.kind == "external_article"]
    if not external:
        # No link body on the item: fetch iff there is a non-x link still to fetch.
        # (x.com links are `fetch_x`'s job and must not drag the item in here.)
        return any(not is_x_url(link.url) for link in item.links)
    return all(
        isinstance(src, ContentSourceFailure) and src.failure_reason in _TRANSIENT_FAILURES
        for src in external
    )


class RetryPlan(BaseModel):
    """What `fetch --retry-failed` would do, and — just as important — what it would NOT.

    `blocked_on_firecrawl` is the whole point of the dry run: those items are recoverable in
    principle (a JS-capable extractor would very likely get them) but re-fetching them WITHOUT
    the key just replays the identical failure. Reporting them as blocked, rather than silently
    dropping them, is the difference between "nothing to do" and "set the key and run again".
    """

    retryable: list[str] = []  # item ids a retry could actually repair
    reasons: dict[str, int] = {}  # failure_reason → count, over the retryable set
    blocked_on_firecrawl: list[str] = []  # fallback-eligible, but no key configured
    terminal: list[str] = []  # dead/blocked — no extractor will ever fix these


def _link_failures(content: Content | None) -> list[ContentSourceFailure]:
    """The recorded `external_article` failures on this item, if any."""
    if content is None:
        return []
    return [
        src
        for src in content.sources
        if isinstance(src, ContentSourceFailure) and src.kind == "external_article"
    ]


def plan_retry_failed(store: dict[str, Item], *, firecrawl_configured: bool) -> RetryPlan:
    """Classify every item with a recorded link failure into: retryable now · blocked on the
    Firecrawl key · terminal. Pure — it reads the store and mutates nothing."""
    plan = RetryPlan()
    for item in store.values():
        failures = _link_failures(item.content) if item.links else []
        if not failures:
            continue
        if should_retry_failed(item.content, firecrawl_configured=firecrawl_configured):
            plan.retryable.append(item.id)
            for src in failures:
                if _retryable_now(src, firecrawl_configured=firecrawl_configured):
                    plan.reasons[src.failure_reason] = plan.reasons.get(src.failure_reason, 0) + 1
        # Would the SAME item be retryable if the key were set? Then the key is what blocks it.
        elif should_retry_failed(item.content, firecrawl_configured=True):
            plan.blocked_on_firecrawl.append(item.id)
        else:
            plan.terminal.append(item.id)
    return plan


class RevalidateResult(BaseModel):
    """What a revalidation sweep found. `urls` are the DEMOTED SOURCES themselves — reporting the
    item's link domains instead would blame a good link that happens to sit on the same item."""

    items: list[str] = []  # item ids whose evidence changed
    urls: list[str] = []  # the source URLs whose body is not an article


def revalidate_stored_bodies(store: dict[str, Item]) -> RevalidateResult:
    """Re-run `validate_body` over the ALREADY-PERSISTED article bodies; demote the walls.

    `--retry-failed` cannot reach these: it selects recorded FAILURES, and a wall that was
    accepted is recorded as a success. Measured on the real corpus: 28 of the 189 fetched
    "articles" are walls or page chrome — 17 YouTube footer blocks, `Loading...`, `Sign in`,
    bot checks, "You need to enable JavaScript to run this app." Every one is persisted as a
    SUCCESS, so `links_content_unfetched` is False, the `[Links — content NOT fetched]` block
    never renders, and the generator was handed the footer under a `[Linked article]` label.
    That is C1, already in the store, not hypothetical.

    Purely local — no network, no extractor. It only re-judges bytes we already hold, so a
    demotion cannot lose anything: the body was never evidence in the first place.

    Mutates `store`; the caller snapshots and saves.
    """
    result = RevalidateResult()
    for item in store.values():
        content = item.content
        if content is None:
            continue
        sources: list[ContentSource] = []
        changed = False
        for src in content.sources:
            if not (isinstance(src, ContentSourceSuccess) and src.kind == "external_article"):
                sources.append(src)
                continue
            verdict = validate_body(
                src.url, FetchSuccess(title=src.title, text=src.text, http_status=200)
            )
            if isinstance(verdict, FetchSuccess):
                sources.append(src)  # a good body is left byte-identical
                continue
            changed = True
            result.urls.append(src.url)
            sources.append(_content_source_from(src.url, verdict))
        if changed:
            item.content = Content(fetched_at=content.fetched_at, sources=sources)
            result.items.append(item.id)
    return result


def _refetch_failed_sources(
    item: Item,
    extractor: ArticleExtractor,
    *,
    firecrawl_configured: bool,
    now: Callable[[], datetime],
) -> Content:
    """Re-extract ONLY the retryable failed link sources; carry everything else across untouched.

    Deliberately NOT `fetch_item`, which re-extracts every non-x link on the item. Because
    `should_retry_failed` uses ANY semantics, a PARTIAL item (one link fetched, one still
    failing) is selected — and `fetch_item` would re-fetch the good article too. A 429 during
    the retry would then replace a perfectly good body with a failure, destroying evidence that
    exists nowhere else in the store. Repairing one link must never cost another one.

    Non-link sources (`x_video` transcripts, frames, threads) are likewise carried across:
    they are not this stage's business.
    """
    content = item.content
    assert content is not None  # `plan_retry_failed` only selects items that have content
    retry_urls = {
        src.url
        for src in content.sources
        if isinstance(src, ContentSourceFailure)
        and src.kind == "external_article"
        and _retryable_now(src, firecrawl_configured=firecrawl_configured)
    }
    sources: list[ContentSource] = [
        _content_source_from(src.url, _safe_extract(src.url, extractor))
        if src.url in retry_urls and isinstance(src, ContentSourceFailure)
        else src
        for src in content.sources
    ]
    if _sources_materially_equal(content.sources, sources):
        return content  # nothing changed — do not advance fetched_at (cf. `fetch_item`)
    return Content(fetched_at=now(), sources=sources)


def retry_failed(
    store: dict[str, Item],
    plan: RetryPlan,
    extractor: ArticleExtractor = extract_article,
    *,
    firecrawl_configured: bool = False,
    now: Callable[[], datetime] = _utcnow,
    sleep: Callable[[float], None] = time.sleep,
    delay: float = _RETRY_DELAY_SECONDS,
) -> int:
    """Re-fetch the failed link sources of exactly the items `plan` marked retryable.

    Mutates `store`; the caller snapshots and saves. Paced by `delay` between items: these are
    other people's servers, and the one `unknown_error` in the real corpus is a recorded HTTP
    429 — hammering a host that already rate-limited us is the one place backoff is not
    optional.
    """
    for index, item_id in enumerate(plan.retryable):
        if index:
            sleep(delay)
        item = store[item_id]
        item.content = _refetch_failed_sources(
            item, extractor, firecrawl_configured=firecrawl_configured, now=now
        )
    return len(plan.retryable)


def fetch_pending(
    store: dict[str, Item],
    since: datetime | None = None,
    until: datetime | None = None,
    force: bool = False,
    extractor: ArticleExtractor = extract_article,
    *,
    now: Callable[[], datetime] = _utcnow,
) -> int:
    """Fetch external content for items that have non-x links and no content yet.

    A previous fetch whose only failures were transient (timeout, dns_error)
    is automatically retried — `--force` is only needed to retry terminal
    failures (404, paywall, …) or to re-hit already-successful items. A retry
    that reproduces the same content does NOT advance `content.fetched_at` (see
    `fetch_item`), so it cannot spuriously re-trigger enrichment. `now` is
    injectable for deterministic tests.
    """
    fetched = 0
    for item in store.values():
        # all links are x.com — handled by fetch_x
        if all(is_x_url(link.url) for link in item.links):
            continue
        if not _should_refetch(item, force):
            continue
        if since and item.created_at < since:
            continue
        if until and item.created_at > until:
            continue
        item.content = fetch_item(item, extractor, now=now)
        fetched += 1
    return fetched
