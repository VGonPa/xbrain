"""Download X-post photos referenced in `Item.media` (and inline Article images).

Photos only — videos remain in `MediaVideoPending`. The orchestrator
`download_all` walks every photo entry, downloads from the X CDN with a
cascading size fallback (`name=orig` → `large` → `medium`), validates with
Pillow, and atomically replaces the entry with `MediaPhotoDownloaded` or
`MediaPhotoFailed`. Failure categorisation mirrors `xbrain.fetch`: a
transient bucket (`http_5xx`, `timeout`, `unknown_error`) is auto-retried
on the next run; permanent failures (`http_4xx`, `format_error`) need
`--force`.

Beyond `Item.media`, the same walk advances the **inline images of an X
long-form Article** (#39 PR4): each `ArticleImageBlock.media` on an
`x_article` `ContentSourceSuccess.blocks` starts as a `MediaPhotoPending`
(emitted by `fetch`) and is downloaded through the SAME `_download_one`
engine, only written to a namespaced `data/media/<id>/article/<n>.<ext>`
path so it never collides with the item's own `<id>/<n>` photos. The
in-place swap sets `block.media`; article-image counts are reported
separately (`MediaReport.article_images_*`) and the total-failure guard
keys on the COMBINED (photos + article images) totals.

Persistence is the caller's responsibility — `download_all` mutates items
in place and calls an `on_progress` hook after each photo/image, so a Ctrl-C
mid-batch leaves `items.json` coherent. I/O dependencies (HTTP session,
sleep) are keyword-injectable so tests run offline.
"""

from __future__ import annotations

import io
import logging
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import assert_never
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from PIL import Image, UnidentifiedImageError

from xbrain.models import (
    ArticleImageBlock,
    ContentSourceSuccess,
    Item,
    MediaEntry,
    MediaFailureReason,
    MediaPhotoDescribed,
    MediaPhotoDownloaded,
    MediaPhotoFailed,
    MediaPhotoPending,
    MediaVideoDownloaded,
    MediaVideoFailed,
    MediaVideoPending,
)

logger = logging.getLogger(__name__)


# Failure reasons that justify an automatic retry on the next `xbrain media`
# run. Mirror of `_TRANSIENT_FAILURES` in `fetch.py` — kept as a separate
# frozenset because the categories differ from content-fetch failures, but
# the retry contract is the same.
_TRANSIENT_MEDIA_FAILURES: frozenset[MediaFailureReason] = frozenset(
    {"http_5xx", "timeout", "unknown_error"}
)

# Severity ordering across the cascade — a 5xx beats a 4xx (we want the
# transient retry signal), a network error beats both. Used by
# `_download_one` to decide which failure category to record when several
# sizes in the cascade fail with different reasons. `format_error` is not
# listed: it's an early-return path inside the 2xx branch, never compared
# across the cascade.
_REASON_SEVERITY: dict[MediaFailureReason, int] = {
    "http_4xx": 1,
    "unknown_error": 2,
    "timeout": 3,
    "http_5xx": 4,
}


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

# Cap on the length of the `error` string stored on `MediaPhotoFailed`.
# A hostile or chatty CDN can return a multi-KB HTML body in the response
# reason field; persisting that bloats items.json and offers no
# diagnostic value beyond the first few hundred chars.
_MAX_ERROR_LEN = 500


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
    # Inline X-Article images (#39 PR4) — counted SEPARATELY from photos so
    # article-image downloads stay visible in the SUMMARY, never folded into
    # the photo totals. `article_images_skipped` is the article analogue of
    # `photos_skipped_already_downloaded` (already-downloaded/described images
    # passed over without `--force`), kept distinct so photo and article skip
    # counts never contaminate each other. `bytes_downloaded` is the combined
    # byte total.
    article_images_attempted: int = 0
    article_images_downloaded: int = 0
    article_images_failed_permanent: int = 0
    article_images_failed_transient: int = 0
    article_images_skipped: int = 0
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
    session: requests.Session | None = None,
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
    - `MediaVideoPending` — photos only; video download is a future
      iteration.

    The function mutates `items` in place; the caller is expected to wrap
    each transition with a store-write (the `on_progress` callback fires
    after every photo transition). The progress callback is also where the
    Ctrl-C-coherent invariant lives: the store is written between photos,
    never mid-download.

    `media_root` is the directory under which `<item_id>/<index>.<ext>` files
    are created. The caller is expected to point this at `data/media/` (or
    equivalent). The directory is created on first use.

    Raises:
        RuntimeError: when EVERY photo attempted in the run fails. A total
            failure (e.g. a CDN outage or a misconfigured network) must
            surface as non-zero exit, not a silent empty run. The CLI's
            `_handle_cli_errors` converts this into a clean operator
            message + exit code 1.
    """
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": user_agent})
    _sweep_part_orphans(media_root)
    started = time.monotonic()
    report = MediaReport()
    candidate_items = _filter_by_ids(items, items_filter)

    for item_id, item, index, entry in _iter_eligible_attempts(
        candidate_items,
        limit=limit,
        force=force,
        report=report,
    ):
        report.photos_attempted += 1
        result = _download_one(
            entry,
            item_id=item_id,
            index=index,
            media_root=media_root,
            session=session,
            timeout_seconds=timeout_seconds,
        )
        item.media[index] = result
        _record_outcome(report, item_id=item_id, entry=result)
        if on_progress is not None:
            on_progress()
        if throttle_seconds > 0:
            sleep(throttle_seconds)

    # Article inline images (#39 PR4) share the SAME per-run download budget:
    # whatever `--limit` slots the photo loop left over. `None` = unbounded.
    remaining_limit = None if limit is None else max(limit - report.photos_attempted, 0)
    _download_article_images(
        candidate_items,
        media_root,
        force=force,
        limit=remaining_limit,
        session=session,
        timeout_seconds=timeout_seconds,
        throttle_seconds=throttle_seconds,
        sleep=sleep,
        on_progress=on_progress,
        report=report,
    )

    report.elapsed_seconds = time.monotonic() - started
    # Total-failure short-circuit keys on the COMBINED (photos + article images)
    # totals: a run that downloads 0 photos but N article images (or vice-versa)
    # is a partial success, not a total failure. Only when NOTHING landed do we
    # raise. Tell the operator loudly — the CLI surfaces it via
    # `_handle_cli_errors`.
    total_attempted = report.photos_attempted + report.article_images_attempted
    total_downloaded = report.photos_downloaded + report.article_images_downloaded
    if total_attempted > 0 and total_downloaded == 0:
        raise RuntimeError(
            f"All {total_attempted} media download attempts failed; "
            "check network / pbs.twimg.com availability and the per-item "
            "warnings above."
        )
    return report


def _download_article_images(
    items: dict[str, Item],
    media_root: Path,
    *,
    force: bool,
    limit: int | None,
    session: requests.Session,
    timeout_seconds: int,
    throttle_seconds: float,
    sleep: Callable[[float], None],
    on_progress: Callable[[], None] | None,
    report: MediaReport,
) -> None:
    """Download every eligible inline Article image, swapping `block.media` in place.

    Mirrors the photo loop in `download_all` — the SAME `_download_one` engine,
    size cascade, throttle and `on_progress` (Ctrl-C-coherent) seam — but writes
    to the namespaced `<id>/article/<n>.<ext>` path (via `subdir="article"`) so
    an article image never collides with the item's own `<id>/<n>` photos, and
    updates the dedicated `article_images_*` counters.

    `limit` is the COMBINED per-run download budget left after the photo loop
    (`None` = unbounded). It is threaded INTO `_iter_eligible_article_images`,
    which — exactly like the photo generator `_iter_eligible_attempts` — checks
    the budget at the top of each iteration and stops yielding (and stops
    counting skips) once it is exhausted. So `--limit` bounds photos + article
    images together, and an exhausted budget never miscounts a skip for an image
    it never reached.
    """
    for item_id, block, index, entry in _iter_eligible_article_images(
        items, limit=limit, force=force, report=report
    ):
        report.article_images_attempted += 1
        result = _download_one(
            entry,
            item_id=item_id,
            index=index,
            media_root=media_root,
            session=session,
            timeout_seconds=timeout_seconds,
            subdir="article",
        )
        block.media = result
        _record_outcome(report, item_id=item_id, entry=result, article=True)
        if on_progress is not None:
            on_progress()
        if throttle_seconds > 0:
            sleep(throttle_seconds)


def _is_eligible(entry: MediaEntry, *, force: bool) -> bool:
    """Decide whether `download_all` should attempt this entry on THIS run.

    The described variant inherits the downloaded contract: a re-download
    only happens under `--force`. A `--force` re-download drops the
    description (the entry collapses back to `MediaPhotoDownloaded`) —
    `xbrain describe` is the path that re-adds it. Forcing the bytes
    without re-describing is the rare case (e.g. the X CDN replaced the
    asset); the operator is expected to follow with `xbrain describe
    --force` if the new bytes warrant it.
    """
    if isinstance(entry, (MediaVideoPending, MediaVideoDownloaded, MediaVideoFailed)):
        # Every video state is out of scope for the PHOTO downloader. Videos
        # are advanced by `xbrain download-videos` (see `xbrain.video_media`).
        return False
    if isinstance(entry, MediaPhotoPending):
        return True
    if isinstance(entry, (MediaPhotoDownloaded, MediaPhotoDescribed)):
        return force
    if isinstance(entry, MediaPhotoFailed):
        if force:
            return True
        return entry.failure_reason in _TRANSIENT_MEDIA_FAILURES
    assert_never(entry)


def _filter_by_ids(items: dict[str, Item], items_filter: list[str] | None) -> dict[str, Item]:
    """Restrict the store to the IDs in `items_filter`, or return it whole.

    Pulled out of `download_all` so the orchestrator does not interleave
    the filter check inside the per-photo loop. An empty / missing
    `items_filter` is a no-op (returns the same dict).
    """
    if not items_filter:
        return items
    wanted = set(items_filter)
    return {item_id: item for item_id, item in items.items() if item_id in wanted}


def _iter_eligible_attempts(
    items: dict[str, Item],
    *,
    limit: int | None,
    force: bool,
    report: MediaReport,
) -> Iterator[
    tuple[
        str,
        Item,
        int,
        MediaPhotoPending | MediaPhotoFailed | MediaPhotoDownloaded | MediaPhotoDescribed,
    ]
]:
    """Yield each (item_id, item, index, entry) pair eligible for download.

    Encapsulates the empty-media skip + per-entry eligibility cascade +
    global limit countdown that `download_all` would otherwise interleave
    with the download orchestration. Side effects on `report`: bumps
    `items_processed` once per item that has media, and
    `photos_skipped_already_downloaded` once per Downloaded / Described
    entry passed over (without `--force`) — both share the
    "bytes-already-on-disk" semantics from the downloader's perspective.
    Stops yielding once `limit` is exhausted.
    """
    remaining = limit
    for item_id, item in items.items():
        if not item.media:
            continue
        report.items_processed += 1
        for index, entry in enumerate(item.media):
            if remaining is not None and remaining <= 0:
                return
            if not _is_eligible(entry, force=force):
                if isinstance(entry, (MediaPhotoDownloaded, MediaPhotoDescribed)):
                    report.photos_skipped_already_downloaded += 1
                continue
            # `_is_eligible` already excluded `MediaVideoPending`; narrow for mypy.
            assert isinstance(
                entry,
                (
                    MediaPhotoPending,
                    MediaPhotoFailed,
                    MediaPhotoDownloaded,
                    MediaPhotoDescribed,
                ),
            )
            if remaining is not None:
                remaining -= 1
            yield item_id, item, index, entry


def _article_image_blocks(item: Item) -> Iterator[tuple[int, ArticleImageBlock]]:
    """Yield `(image_index, block)` for every inline image of an item's Articles.

    Walks each `x_article` `ContentSourceSuccess` on `item.content` in order and
    yields its `ArticleImageBlock`s with a per-item running `image_index`. The
    index counts EVERY image block (text blocks and non-image content excluded),
    so it is stable regardless of a block's download state — the guarantee the
    `<id>/article/<n>` namespace relies on (an already-downloaded block 0 must
    not shift a pending block 1 down to `article/0`). An item with no content or
    no article source yields nothing.
    """
    if item.content is None:
        return
    image_index = 0
    for source in item.content.sources:
        if not (isinstance(source, ContentSourceSuccess) and source.kind == "x_article"):
            continue
        for block in source.blocks:
            if isinstance(block, ArticleImageBlock):
                yield image_index, block
                image_index += 1


def _iter_eligible_article_images(
    items: dict[str, Item],
    *,
    limit: int | None,
    force: bool,
    report: MediaReport,
) -> Iterator[
    tuple[
        str,
        ArticleImageBlock,
        int,
        MediaPhotoPending | MediaPhotoFailed | MediaPhotoDownloaded | MediaPhotoDescribed,
    ]
]:
    """Yield each (item_id, block, image_index, entry) article image to download.

    Mirrors `_iter_eligible_attempts` for the inline images living OUTSIDE
    `item.media` — on the `x_article` `ContentSourceSuccess.blocks`. Applies the
    SAME `_is_eligible` cascade (pending always; transient-failed on retry;
    downloaded/described only under `--force`) and the SAME `limit` discipline:
    the budget is checked at the TOP of each image iteration and decremented only
    for an actually-yielded attempt, so once it hits zero the walk stops — and
    stops counting skips — exactly like the photo generator (no scanning-past a
    spent budget and miscounting the images it never reached).

    Side effects on `report`: bumps `items_processed` once per article-only item
    (an item with `item.media` was already counted by the photo walk), and the
    DEDICATED `article_images_skipped` once per already-downloaded/described image
    passed over without `--force` (kept distinct from the photo skip counter so
    the two never contaminate each other). The caller swaps the yielded `entry`
    back onto `block.media`.
    """
    remaining = limit
    for item_id, item in items.items():
        # Items with `item.media` were already counted by the photo walk; count
        # an article-only item the first time we touch one of its images.
        counted = bool(item.media)
        for image_index, block in _article_image_blocks(item):
            if remaining is not None and remaining <= 0:
                return
            if not counted:
                report.items_processed += 1
                counted = True
            entry = block.media
            if not _is_eligible(entry, force=force):
                if isinstance(entry, (MediaPhotoDownloaded, MediaPhotoDescribed)):
                    report.article_images_skipped += 1
                continue
            # `_is_eligible` already excluded every video variant; narrow for mypy.
            assert isinstance(
                entry,
                (
                    MediaPhotoPending,
                    MediaPhotoFailed,
                    MediaPhotoDownloaded,
                    MediaPhotoDescribed,
                ),
            )
            if remaining is not None:
                remaining -= 1
            yield item_id, block, image_index, entry


def _record_outcome(
    report: MediaReport,
    *,
    item_id: str,
    entry: MediaPhotoDownloaded | MediaPhotoFailed,
    article: bool = False,
) -> None:
    """Bump the report counters based on the post-transition variant.

    A successful download contributes to `photos_downloaded` /
    `article_images_downloaded` and the shared `bytes_downloaded`; a failed
    one to the appropriate transient / permanent bucket and to
    `per_item_failures` (keyed by item id). `article=True` routes the
    counters to the `article_images_*` fields so inline-Article-image
    downloads stay visible, never folded into the photo counts (the failure
    list and byte total stay shared — an item's failures are one list). Every
    failure also emits a structured `logger.warning` so the total-failure
    RuntimeError's "see warnings above" message has actual breadcrumbs.
    """
    if isinstance(entry, MediaPhotoDownloaded):
        if article:
            report.article_images_downloaded += 1
        else:
            report.photos_downloaded += 1
        report.bytes_downloaded += entry.bytes_size
        return
    report.per_item_failures.setdefault(item_id, []).append((entry.url, entry.failure_reason))
    transient = entry.failure_reason in _TRANSIENT_MEDIA_FAILURES
    if article:
        _bump_article_failure(report, transient=transient)
    elif transient:
        report.photos_failed_transient += 1
    else:
        report.photos_failed_permanent += 1
    logger.warning(
        "media: download failed item=%s url=%s reason=%s error=%s",
        item_id,
        entry.url,
        entry.failure_reason,
        entry.error,
    )


def _bump_article_failure(report: MediaReport, *, transient: bool) -> None:
    """Route an article-image failure to the transient/permanent article bucket."""
    if transient:
        report.article_images_failed_transient += 1
    else:
        report.article_images_failed_permanent += 1


def _download_one(
    entry: MediaPhotoPending | MediaPhotoFailed | MediaPhotoDownloaded | MediaPhotoDescribed,
    *,
    item_id: str,
    index: int,
    media_root: Path,
    session: requests.Session,
    timeout_seconds: int,
    subdir: str | None = None,
) -> MediaPhotoDownloaded | MediaPhotoFailed:
    """Download one photo with size cascade and Pillow validation.

    Returns the post-transition variant — the caller swaps it into
    `item.media[index]` (or, for an inline Article image, onto
    `block.media`). `subdir` namespaces the on-disk + `local_path`
    location: photos pass `None` → `<id>/<n>.<ext>`; an inline Article image
    passes `subdir="article"` → `<id>/article/<n>.<ext>` (#39 PR4). Never
    raises on a recoverable failure: the `failure_reason` field carries the
    categorisation. The only uncaught exceptions are programmer bugs (e.g.
    `AttributeError`) and `KeyboardInterrupt` — both must propagate so the
    developer sees the traceback / Ctrl-C still works.
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
            # (transient), matching `xbrain.fetch`'s retry contract.
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
                    url=url,
                    reason="format_error",
                    error="Pillow could not decode the downloaded bytes.",
                    attempts=attempts,
                )
            width, height, fmt = decoded
            extension = _FORMAT_EXTENSIONS.get(fmt.lower(), ".jpg")
            local_path = _local_path(item_id, index, extension, subdir=subdir)
            try:
                _write_bytes(media_root / local_path, response.content)
            except OSError as exc:
                # Disk full, permission denied, read-only filesystem —
                # bucket as `unknown_error` (transient) so the next run
                # picks it up once the operator clears the underlying
                # condition. Without this guard, the OSError escapes
                # per-item bucketing and aborts the whole batch.
                return _failed(
                    url=url,
                    reason="unknown_error",
                    error=f"local write failed: {exc}",
                    attempts=attempts,
                )
            return MediaPhotoDownloaded(
                url=url,
                local_path=local_path,
                width=width,
                height=height,
                bytes_size=len(response.content),
                downloaded_at=datetime.now(timezone.utc),
            )
        # Non-2xx — classify the status, bucket the failure, try next size.
        cascade_reason = _worse(cascade_reason, _classify_status(status))
        cascade_status = status
        last_error = RuntimeError(f"HTTP {status} for {candidate_url}")

    reason = cascade_reason or "unknown_error"
    return _failed(
        url=url,
        reason=reason,
        error=_format_error(last_error, cascade_status),
        attempts=attempts,
    )


def _classify_status(status: int) -> MediaFailureReason:
    """Map a non-2xx HTTP status to its failure-reason bucket.

    4xx is permanent for this URL (dead asset, bad cascade size); 5xx is
    transient (CDN hiccup). Anything else — a 3xx that `requests` did not
    follow, or a non-standard code — is bucketed as `unknown_error` so the
    next run retries it.
    """
    if 400 <= status < 500:
        return "http_4xx"
    if 500 <= status < 600:
        return "http_5xx"
    return "unknown_error"


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
    url: str,
    reason: MediaFailureReason,
    error: str | None,
    attempts: int,
) -> MediaPhotoFailed:
    """Build a `MediaPhotoFailed` variant for a download that did not land."""
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
    A failure returns None so the caller can bucket as `format_error`.

    The exception set is narrow-by-design: Pillow-specific signals only.
    Catching the parent `OSError` would swallow `FileNotFoundError`,
    `PermissionError` and other real I/O bugs — we want those to propagate
    as tracebacks, not be silently bucketed as `format_error`.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        # `verify()` invalidates the image — re-open to read size + format.
        with Image.open(io.BytesIO(data)) as image:
            return image.width, image.height, image.format or "jpg"
    except (UnidentifiedImageError, Image.DecompressionBombError, SyntaxError):
        return None


def _write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` atomically (tmp file + rename).

    Atomic write mirrors `xbrain.store._atomic_write`: a Ctrl-C between
    `open` and `write_bytes` would leave a zero-byte partial file that the
    next run would consider downloaded. We avoid that by writing to a
    sibling tmp file and `os.replace`ing it into place.

    Orphan `.part` files left behind by a hard interruption (SIGKILL, OOM)
    that bypassed our `except BaseException` cleanup are swept on the next
    `download_all` entry — see `_sweep_part_orphans`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _sweep_part_orphans(media_root: Path) -> None:
    """Remove stale ``*.part`` files left by hard-killed previous runs.

    A SIGKILL or OOM kill between `tmp.write_bytes` and `tmp.replace` would
    leave a `<n>.<ext>.part` next to the final file location with no entry
    in `items.json` referencing it. We do not block on the sweep — best
    effort: any path we cannot unlink is logged and skipped, on the theory
    that "the operator can clean up later" beats "the next photo refuses
    to download because the tree has an unwritable file in it".
    """
    if not media_root.exists():
        return
    for orphan in media_root.rglob("*.part"):
        try:
            orphan.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("media: could not remove stale .part file %s: %s", orphan, exc)


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


def _local_path(item_id: str, index: int, extension: str, *, subdir: str | None = None) -> str:
    """Deterministic relative path: ``<item_id>[/<subdir>]/<index><extension>``.

    Returned as a forward-slash string (the storage convention for an
    Obsidian embed) rather than a `Path` so it can be persisted on
    `MediaPhotoDownloaded.local_path` without OS-dependent reformatting.

    `subdir` namespaces a distinct media class under the item so it never
    collides with the item's own photos: an inline Article image (#39 PR4)
    passes ``subdir="article"`` → ``<id>/article/<n>.<ext>``, disjoint from
    the ``<id>/<n>.<ext>`` photo path. The result stays relative (no leading
    slash, no ``..``), so `_reject_local_path_traversal` still passes.
    """
    if subdir:
        return f"{item_id}/{subdir}/{index}{extension}"
    return f"{item_id}/{index}{extension}"


def _format_error(exc: Exception | None, status: int | None) -> str | None:
    """Compose a human-readable error string from the last exception.

    Capped at `_MAX_ERROR_LEN` characters: a misbehaving CDN can return a
    multi-KB HTML body in `RequestException.__str__`, and persisting that
    on every `MediaPhotoFailed` bloats `items.json` for zero diagnostic
    value beyond the first chunk.
    """
    if exc is None and status is None:
        return None
    if exc is None:
        text = f"HTTP {status}"
    elif status is None:
        text = str(exc)
    else:
        text = f"HTTP {status}: {exc}"
    if len(text) > _MAX_ERROR_LEN:
        return text[: _MAX_ERROR_LEN - 1] + "…"
    return text


def emit_summary_line(report: MediaReport, *, out: "io.IOBase | None" = None) -> None:
    """Print the SUMMARY line on stderr (mirrors `ApiExecutor.enrich_items`).

    The line is emitted only if at least one photo OR article image was
    attempted, or something was skipped — a fully no-op run (e.g. an `--items`
    filter that matched nothing) stays silent. The `article_*` counters keep
    inline-Article-image downloads (#39 PR4) visible instead of folding them
    silently into the photo counts. `out` is injectable for tests; defaults to
    `sys.stderr`.
    """
    if (
        report.photos_attempted == 0
        and report.article_images_attempted == 0
        and report.photos_skipped_already_downloaded == 0
        and report.article_images_skipped == 0
    ):
        return
    target = out if out is not None else sys.stderr
    print(
        f"SUMMARY: downloaded: {report.photos_downloaded}, "
        f"failed_permanent: {report.photos_failed_permanent}, "
        f"failed_transient: {report.photos_failed_transient}, "
        f"skipped: {report.photos_skipped_already_downloaded}, "
        f"article_downloaded: {report.article_images_downloaded}, "
        f"article_failed_permanent: {report.article_images_failed_permanent}, "
        f"article_failed_transient: {report.article_images_failed_transient}, "
        f"article_skipped: {report.article_images_skipped}, "
        f"bytes: {report.bytes_downloaded:_}",
        file=target,
    )
