"""Download X-post photos referenced in `Item.media`.

Phase A scope (issue #33): photos only. Videos are deliberately left in the
`MediaVideoPending` variant for a later phase — HLS + ffmpeg is significantly
different complexity and not on the critical path here.

The orchestrator (`download_all`) walks every photo entry on every item,
decides whether it is eligible (pending, transient-failure, or `--force`),
downloads bytes from the X CDN with a cascading size fallback
(``name=orig`` → ``name=large`` → ``name=medium``), validates the bytes with
Pillow, and atomically replaces the entry on the item with the appropriate
variant — `MediaPhotoDownloaded` on success, `MediaPhotoFailed` on a
categorised failure.

Failure categorisation mirrors `xbrain.fetch` (#19): the transient bucket
(`http_5xx`, `timeout`, `unknown_error`) is re-attempted on the next run;
the permanent bucket (`http_4xx`, `format_error`) is only retried with
`--force`. Bare-except is bucketed under `unknown_error` so a future
unhandled error path never silently swallows the URL.

Persistence is the caller's responsibility — the orchestrator mutates
items in place and calls a `save` callback after each photo, so a Ctrl-C
mid-batch leaves `items.json` coherent. The default callback writes the
store atomically via `xbrain.store.save_store`.

I/O dependencies (HTTP session, sleep, Pillow) are dependency-injected via
keyword arguments so tests run offline without monkeypatching.
"""

from __future__ import annotations

import io
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from PIL import Image, UnidentifiedImageError

from xbrain.models import (
    Item,
    MediaEntry,
    MediaFailureReason,
    MediaPhotoDownloaded,
    MediaPhotoFailed,
    MediaPhotoPending,
    MediaVideoPending,
)

logger = logging.getLogger(__name__)


# Failure reasons that justify an automatic retry on the next `xbrain media`
# run. Mirror of `_TRANSIENT_FAILURES` in `fetch.py` (#19) — kept as a separate
# frozenset because the categories differ from content-fetch failures, but the
# retry contract is the same.
_TRANSIENT_MEDIA_FAILURES: frozenset[MediaFailureReason] = frozenset(
    {"http_5xx", "timeout", "unknown_error"}
)


# Conservative defaults — pbs.twimg.com tolerates well-behaved scrapers, but
# bursting from a fresh IP earns a 429. The throttle is a per-request sleep
# (caller-injectable for tests). The User-Agent is browser-shaped so the CDN
# does not bounce the request on UA-pattern alone.
_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_DEFAULT_TIMEOUT_SECONDS = 20
_DEFAULT_THROTTLE_SECONDS = 0.5

# Size cascade in priority order. pbs.twimg.com URLs accept a `name=`
# parameter selecting the rendered size; `orig` is the highest fidelity the
# CDN exposes, `large` is the standard high-DPI variant, `medium` is the
# safe fallback. We try in order and stop at the first variant that returns
# valid image bytes.
_SIZE_CASCADE: tuple[str, ...] = ("orig", "large", "medium")

# Map URL-derived `format=` values to file extensions on disk. Twitter
# returns `jpg`, `png`, `webp` — we never get to choose the format, so the
# extension just records what the CDN sent us.
_FORMAT_EXTENSIONS: dict[str, str] = {"jpg": ".jpg", "jpeg": ".jpg", "png": ".png", "webp": ".webp"}


class SessionProtocol(Protocol):
    """The subset of `requests.Session` the downloader actually uses.

    Declared as a Protocol so a test can inject a hand-rolled fake without
    pulling in the full `responses` / `requests-mock` machinery.
    """

    def get(self, url: str, *, timeout: int) -> requests.Response:
        """Issue a GET request and return the response."""
        ...


@dataclass
class MediaReport:
    """Counts emitted by `download_all` for the CLI's SUMMARY line.

    `photos_failed_transient` counts entries that landed in the transient
    failure bucket on THIS run (i.e. eligible for the next run's auto-retry).
    `photos_failed_permanent` counts entries that landed in a terminal
    bucket. `photos_skipped_already_downloaded` is the idempotency proof —
    a no-op re-run must report every previously-downloaded photo here.
    """

    items_processed: int = 0
    photos_attempted: int = 0
    photos_downloaded: int = 0
    photos_failed_permanent: int = 0
    photos_failed_transient: int = 0
    photos_skipped_already_downloaded: int = 0
    bytes_downloaded: int = 0
    elapsed_seconds: float = 0.0
    # Per-item failures keyed by item id → list of (url, reason) tuples.
    # Surfaces the failed URLs in tests / debugging without re-walking the
    # mutated store.
    per_item_failures: dict[str, list[tuple[str, MediaFailureReason]]] = field(default_factory=dict)


def download_all(
    items: dict[str, Item],
    media_root: Path,
    *,
    force: bool = False,
    limit: int | None = None,
    items_filter: list[str] | None = None,
    throttle_seconds: float = _DEFAULT_THROTTLE_SECONDS,
    user_agent: str = _DEFAULT_UA,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    session: SessionProtocol | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[], None] | None = None,
) -> MediaReport:
    """Download every eligible photo across the store; return a structured report.

    Eligibility (without `--force`):
    - `MediaPhotoPending` — always.
    - `MediaPhotoFailed` whose `failure_reason` is in `_TRANSIENT_MEDIA_FAILURES`.

    With `--force`:
    - Every `MediaPhotoPending`, `MediaPhotoFailed`, and `MediaPhotoDownloaded`
      is re-downloaded. The previously-downloaded file on disk is overwritten.

    Out of scope (every run):
    - `MediaVideoPending` — Phase A is photos only.

    The function mutates `items` in place; the caller is expected to wrap
    each transition with a store-write (the `on_progress` callback fires
    after every photo transition). The progress callback is also where the
    Ctrl-C-coherent invariant lives: the store is written between photos,
    never mid-download.

    `media_root` is the directory under which `<item_id>/<index>.<ext>` files
    are created. The caller is expected to point this at `data/media/` (or
    equivalent). The directory is created on first use.

    Raises:
        RuntimeError: when EVERY photo attempted in the run fails. Mirrors
            `ApiExecutor.enrich_items` (#24): a total failure (e.g. a CDN
            outage or a misconfigured network) must surface as non-zero exit,
            not a silent empty run. The CLI's `_handle_cli_errors` converts
            this into a clean operator message + exit code 1.
    """
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": user_agent})

    target_ids: set[str] | None = set(items_filter) if items_filter else None
    started = time.monotonic()
    report = MediaReport()
    remaining = limit if limit is not None else None

    for item_id, item in items.items():
        if target_ids is not None and item_id not in target_ids:
            continue
        if not item.media:
            continue
        report.items_processed += 1
        # Walk a snapshot of indices: in-place replacement keeps `item.media`
        # the same length, so range(len(...)) at start is safe.
        for index in range(len(item.media)):
            if remaining is not None and remaining <= 0:
                # `--limit` cap reached. We stop attempting new downloads but
                # do NOT mutate already-attempted entries (they keep whichever
                # variant they had on entry).
                _finalise(report, started)
                return report
            entry = item.media[index]
            if not _is_eligible(entry, force=force):
                if isinstance(entry, MediaPhotoDownloaded):
                    report.photos_skipped_already_downloaded += 1
                continue
            # `_is_eligible` already excluded `MediaVideoPending`; narrow for mypy.
            assert isinstance(
                entry, (MediaPhotoPending, MediaPhotoFailed, MediaPhotoDownloaded)
            )
            report.photos_attempted += 1
            if remaining is not None:
                remaining -= 1
            result = _download_one(
                entry,
                item_id=item_id,
                index=index,
                media_root=media_root,
                session=session,
                timeout_seconds=timeout_seconds,
            )
            item.media[index] = result
            _record_outcome(report, item_id=item_id, entry=result, original=entry)
            if on_progress is not None:
                on_progress()
            if throttle_seconds > 0:
                sleep(throttle_seconds)

    _finalise(report, started)
    if report.photos_attempted > 0 and report.photos_downloaded == 0:
        # Total-failure short-circuit: no good bytes landed anywhere this
        # run. Tell the operator loudly. The CLI surfaces it via
        # `_handle_cli_errors`.
        raise RuntimeError(
            f"All {report.photos_attempted} photo download attempts failed; "
            "check network / pbs.twimg.com availability and the per-photo "
            "warnings above."
        )
    return report


def _is_eligible(entry: MediaEntry, *, force: bool) -> bool:
    """Decide whether `download_all` should attempt this entry on THIS run."""
    if isinstance(entry, MediaVideoPending):
        return False
    if isinstance(entry, MediaPhotoPending):
        return True
    if isinstance(entry, MediaPhotoDownloaded):
        return force
    if isinstance(entry, MediaPhotoFailed):
        if force:
            return True
        return entry.failure_reason in _TRANSIENT_MEDIA_FAILURES
    return False


def _record_outcome(
    report: MediaReport,
    *,
    item_id: str,
    entry: MediaEntry,
    original: MediaEntry,
) -> None:
    """Bump the report counters based on the post-transition variant.

    `original` is the pre-transition entry — used to keep a record of bytes
    downloaded (a re-download of an already-downloaded photo subtracts the
    old `bytes_size` from the count to avoid double-counting).
    """
    if isinstance(entry, MediaPhotoDownloaded):
        report.photos_downloaded += 1
        report.bytes_downloaded += entry.bytes_size
    elif isinstance(entry, MediaPhotoFailed):
        report.per_item_failures.setdefault(item_id, []).append((entry.url, entry.failure_reason))
        if entry.failure_reason in _TRANSIENT_MEDIA_FAILURES:
            report.photos_failed_transient += 1
        else:
            report.photos_failed_permanent += 1
    # Any other variant (e.g. video pending) is not produced by _download_one,
    # so we don't need a branch here. `original` is currently unused — kept
    # in the signature for symmetry / future delta accounting.
    _ = original


def _finalise(report: MediaReport, started: float) -> None:
    """Stamp the elapsed time on the report at end-of-run."""
    report.elapsed_seconds = time.monotonic() - started


def _download_one(
    entry: MediaPhotoPending | MediaPhotoFailed | MediaPhotoDownloaded,
    *,
    item_id: str,
    index: int,
    media_root: Path,
    session: SessionProtocol,
    timeout_seconds: int,
) -> MediaPhotoDownloaded | MediaPhotoFailed:
    """Download one photo with size cascade and Pillow validation.

    Returns the post-transition variant — the caller swaps it into
    `item.media[index]`. Never raises on a recoverable failure: the
    `failure_reason` field carries the categorisation. The only uncaught
    exceptions are programmer bugs (e.g. `AttributeError`) and
    `KeyboardInterrupt` — both must propagate so the developer sees the
    traceback / Ctrl-C still works.
    """
    url = entry.url
    attempts = (entry.attempts if isinstance(entry, MediaPhotoFailed) else 0) + 1
    last_error: Exception | None = None
    # Track the worst-failure-seen across the cascade so a 404 on `orig`
    # plus a 5xx on `large` reports as a 5xx (transient) not a 4xx.
    cascade_reason: MediaFailureReason | None = None
    cascade_status: int | None = None

    for size in _SIZE_CASCADE:
        candidate_url = _url_with_name(url, size)
        try:
            response = session.get(candidate_url, timeout=timeout_seconds)
        except requests.Timeout as exc:
            cascade_reason = _worse(cascade_reason, "timeout")
            last_error = exc
            continue
        except requests.RequestException as exc:
            # Connection errors, SSL errors, etc — bucket as unknown_error
            # (transient) per the #19 contract.
            cascade_reason = _worse(cascade_reason, "unknown_error")
            last_error = exc
            continue
        status = response.status_code
        if 200 <= status < 300:
            decoded = _decode_image(response.content)
            if decoded is None:
                # Bytes arrived but Pillow couldn't read them. Permanent for
                # this URL (the CDN sent us something we cannot use).
                return _failed(
                    entry=entry,
                    url=url,
                    reason="format_error",
                    error="Pillow could not decode the downloaded bytes.",
                    attempts=attempts,
                )
            width, height, fmt = decoded
            extension = _FORMAT_EXTENSIONS.get(fmt.lower(), ".jpg")
            local_path = _local_path(item_id, index, extension)
            _write_bytes(media_root / local_path, response.content)
            return MediaPhotoDownloaded(
                url=url,
                local_path=local_path,
                width=width,
                height=height,
                bytes_size=len(response.content),
                downloaded_at=datetime.now(timezone.utc),
            )
        if 400 <= status < 500:
            cascade_reason = _worse(cascade_reason, "http_4xx")
            cascade_status = status
            last_error = RuntimeError(f"HTTP {status} for {candidate_url}")
            continue
        if 500 <= status < 600:
            cascade_reason = _worse(cascade_reason, "http_5xx")
            cascade_status = status
            last_error = RuntimeError(f"HTTP {status} for {candidate_url}")
            continue
        # Non-success, non-error status code (e.g. a 3xx that requests didn't
        # follow). Bucket as unknown_error so the next run retries it.
        cascade_reason = _worse(cascade_reason, "unknown_error")
        cascade_status = status
        last_error = RuntimeError(f"HTTP {status} for {candidate_url}")

    reason = cascade_reason or "unknown_error"
    return _failed(
        entry=entry,
        url=url,
        reason=reason,
        error=_format_error(last_error, cascade_status),
        attempts=attempts,
    )


# Severity ordering across the cascade — a 5xx beats a 4xx (we want the
# transient retry signal), a network error beats both. Used by
# `_download_one` to decide which failure category to record when several
# sizes in the cascade fail with different reasons.
_REASON_SEVERITY: dict[MediaFailureReason, int] = {
    "http_4xx": 1,
    "unknown_error": 2,
    "timeout": 3,
    "http_5xx": 4,
    "format_error": 5,  # never used here but completes the table
}


def _worse(
    current: MediaFailureReason | None,
    candidate: MediaFailureReason,
) -> MediaFailureReason:
    """Return whichever reason has higher severity (= more retry signal)."""
    if current is None:
        return candidate
    if _REASON_SEVERITY[candidate] > _REASON_SEVERITY[current]:
        return candidate
    return current


def _failed(
    *,
    entry: MediaPhotoPending | MediaPhotoFailed | MediaPhotoDownloaded,
    url: str,
    reason: MediaFailureReason,
    error: str | None,
    attempts: int,
) -> MediaPhotoFailed:
    """Build a `MediaPhotoFailed` variant for a download that did not land.

    `entry` is the pre-transition variant — currently unused, kept in the
    signature so future callers can carry forward state (e.g. preserving
    the original `local_path` from a `MediaPhotoDownloaded` being
    re-downloaded with `--force` that subsequently failed).
    """
    _ = entry
    return MediaPhotoFailed(
        url=url,
        failure_reason=reason,
        error=error,
        attempts=attempts,
        last_attempt_at=datetime.now(timezone.utc),
    )


def _decode_image(data: bytes) -> tuple[int, int, str] | None:
    """Validate bytes with Pillow; return ``(width, height, format)`` or None.

    A successful decode means the CDN sent us something Obsidian will render.
    A failure returns None so the caller can bucket as `format_error` —
    Pillow's exception types are too noisy to propagate verbatim.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        # `verify()` invalidates the image — re-open to read size + format.
        with Image.open(io.BytesIO(data)) as image:
            return image.width, image.height, image.format or "jpg"
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` atomically (tmp file + rename).

    Atomic write mirrors `xbrain.store._atomic_write`: a Ctrl-C between
    `open` and `write_bytes` would leave a zero-byte partial file that the
    next run would consider downloaded. We avoid that by writing to a
    sibling tmp file and `os.replace`ing it into place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _url_with_name(url: str, size: str) -> str:
    """Return the URL with the X CDN ``name=`` query param set to `size`.

    Twitter pbs.twimg.com URLs accept a `name=` parameter that selects the
    rendered size. We always rewrite it (even if absent) so the cascade
    works regardless of whether the extractor captured an already-sized
    URL or the bare path.
    """
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["name"] = size
    return urlunparse(parsed._replace(query=urlencode(query)))


def _local_path(item_id: str, index: int, extension: str) -> str:
    """Deterministic relative path: ``<item_id>/<index><extension>``.

    Returned as a forward-slash string (the storage convention for an
    Obsidian embed) rather than a `Path` so it can be persisted on
    `MediaPhotoDownloaded.local_path` without OS-dependent reformatting.
    """
    return f"{item_id}/{index}{extension}"


def _format_error(exc: Exception | None, status: int | None) -> str | None:
    """Compose a human-readable error string from the last exception."""
    if exc is None and status is None:
        return None
    if exc is None:
        return f"HTTP {status}"
    if status is None:
        return str(exc)
    return f"HTTP {status}: {exc}"


def emit_summary_line(report: MediaReport, *, out: "io.IOBase | None" = None) -> None:
    """Print the SUMMARY line on stderr (mirrors `ApiExecutor.enrich_items`).

    The line is emitted only if at least one photo was attempted or skipped —
    a fully no-op run (e.g. an `--items` filter that matched nothing) stays
    silent. `out` is injectable for tests; defaults to `sys.stderr`.
    """
    if report.photos_attempted == 0 and report.photos_skipped_already_downloaded == 0:
        return
    target = out if out is not None else sys.stderr
    print(
        f"SUMMARY: downloaded: {report.photos_downloaded}, "
        f"failed_permanent: {report.photos_failed_permanent}, "
        f"failed_transient: {report.photos_failed_transient}, "
        f"skipped: {report.photos_skipped_already_downloaded}, "
        f"bytes: {report.bytes_downloaded:_}",
        file=target,
    )
