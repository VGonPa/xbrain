"""Download the actual mp4 bytes for `MediaVideoPending` entries to disk.

The file-download counterpart to `xbrain.media` (photos). The orchestrator
`download_videos` walks every video entry, downloads ONLY real progressive
**mp4** streams from the X CDN, and atomically replaces the entry with
`MediaVideoDownloaded` (bytes on disk) or `MediaVideoFailed` (categorised).
Failure categorisation, the transient/permanent retry contract, the
browser-shaped User-Agent, the per-request throttle, the atomic write and the
`.part`-orphan sweep are all **reused** from `xbrain.media` rather than
re-implemented — this module imports those shared primitives so the photo and
video downloaders stay byte-for-byte consistent (and the photo path is not
refactored).

Scope (this PR): mp4 only. HLS (`.m3u8`) manifests need ffmpeg to mux into a
playable file and are a SEPARATE follow-up — they are skipped and counted, never
downloaded here. Poster-era entries (un-backfilled: `url == thumbnail_url`, or a
legacy record whose URL is neither an mp4 nor an HLS manifest) are skipped
silently and counted; `xbrain refresh-media` is the path that backfills them
into a real stream first.

Persistence is the caller's responsibility — `download_videos` mutates items in
place and calls an `on_progress` hook after each transition, so a Ctrl-C
mid-batch leaves `items.json` coherent. I/O dependencies (HTTP session, sleep)
are keyword-injectable so tests run offline.
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
from typing import Literal
from urllib.parse import urlparse

import requests

# Shared download primitives — defined once for the photo pipeline in
# `xbrain.media`, reused here so the two downloaders share one implementation
# of the retry classification, atomic write, orphan sweep, error formatting and
# the network defaults. Importing the module-level helpers (rather than copying
# them) is the "reuse, don't duplicate" contract; the photo path is untouched.
from xbrain.media import (
    _DEFAULT_THROTTLE_SECONDS,
    _DEFAULT_TIMEOUT_SECONDS,
    _DEFAULT_UA,
    _TRANSIENT_MEDIA_FAILURES,
    _classify_status,
    _filter_by_ids,
    _format_error,
    _local_path,
    _sweep_part_orphans,
    _write_bytes,
)
from xbrain.models import (
    Item,
    MediaEntry,
    MediaFailureReason,
    MediaVideoDownloaded,
    MediaVideoFailed,
    MediaVideoPending,
)

# bits/s × milliseconds ÷ this = bytes (same physics as `refresh.estimate_
# download_size`). Reused so the pre-flight estimate here and the whole-store
# estimate printed by `refresh-media` stay on one formula.
from xbrain.refresh import _BITS_PER_BYTE_TIMES_MILLIS_PER_SECOND

logger = logging.getLogger(__name__)

# The three media states the video downloader can act on. Photo variants are
# silently ignored (they are `xbrain media`'s job).
_VideoEntry = MediaVideoPending | MediaVideoDownloaded | MediaVideoFailed

# How a video URL is classified for THIS run's mp4-only scope.
VideoStreamKind = Literal["mp4", "hls", "poster"]


@dataclass
class VideoReport:
    """Counts emitted by `download_videos` for the CLI's SUMMARY line.

    `videos_failed_transient` counts entries that landed in the transient
    bucket on THIS run (eligible for the next run's auto-retry);
    `videos_failed_permanent` the terminal bucket. The three `skipped_*`
    counters are the mp4-only scope made visible: `skipped_hls` are deferred to
    the ffmpeg follow-up, `skipped_poster_era` are un-backfilled (run
    `refresh-media` first), and `skipped_already_downloaded` is the idempotency
    proof — a no-op re-run reports every previously-downloaded video here.
    """

    items_processed: int = 0
    videos_attempted: int = 0
    videos_downloaded: int = 0
    videos_failed_permanent: int = 0
    videos_failed_transient: int = 0
    videos_skipped_hls: int = 0
    videos_skipped_poster_era: int = 0
    videos_skipped_already_downloaded: int = 0
    bytes_downloaded: int = 0
    elapsed_seconds: float = 0.0
    # Per-item failures keyed by item id → list of (url, reason) tuples.
    per_item_failures: dict[str, list[tuple[str, MediaFailureReason]]] = field(default_factory=dict)


@dataclass
class VideoDownloadPlan:
    """The pre-flight preview that drives the size-gate confirmation.

    Computed WITHOUT touching the network by replaying the exact same
    eligibility walk `download_videos` will perform (same `force` / `limit` /
    `items_filter`), so the gate's promise matches what the run does.
    `estimated_bytes` sums `bitrate × duration` over the eligible mp4 set only;
    `n_unknown` counts eligible mp4s with no bitrate/duration (their size is
    unknown until the bytes land — never assumed 0).
    """

    n_to_download: int = 0
    estimated_bytes: int = 0
    n_estimable: int = 0
    n_unknown: int = 0
    n_hls_skipped: int = 0
    n_poster_skipped: int = 0
    n_already_downloaded: int = 0


def _video_class(entry: _VideoEntry) -> VideoStreamKind:
    """Classify a video entry's URL for the mp4-only download scope.

    Order matters: the poster check comes first (an un-backfilled entry's URL
    is the poster image), then `.m3u8` (HLS lives on video.twimg.com too, so the
    suffix check must win over the host check), then the mp4 test (real X
    streams are hosted on video.twimg.com, or carry an `.mp4` path on any host).
    Anything else — a legacy pbs.jpg record that was never backfilled — is
    treated as poster-era: not a downloadable stream this run.
    """
    if entry.url == entry.thumbnail_url:
        return "poster"
    parsed = urlparse(entry.url)
    path = parsed.path.lower()
    if path.endswith(".m3u8"):
        return "hls"
    if parsed.netloc.lower() == "video.twimg.com" or path.endswith(".mp4"):
        return "mp4"
    return "poster"


def _is_video_download_eligible(entry: MediaEntry, *, force: bool) -> bool:
    """Decide whether `download_videos` should download this entry THIS run.

    Only real mp4 streams are ever downloaded. A pending mp4 is always eligible;
    a downloaded mp4 only under `--force`; a failed mp4 retries when its reason
    is transient (or always under `--force`). HLS and poster-era entries are
    never eligible (they are counted as skips elsewhere), and every photo
    variant is ignored — that is `xbrain media`'s job.
    """
    if not isinstance(entry, (MediaVideoPending, MediaVideoDownloaded, MediaVideoFailed)):
        return False
    if _video_class(entry) != "mp4":
        return False
    if isinstance(entry, MediaVideoPending):
        return True
    if isinstance(entry, MediaVideoDownloaded):
        return force
    if force:
        return True
    return entry.failure_reason in _TRANSIENT_MEDIA_FAILURES


def _estimated_bytes(entry: _VideoEntry) -> int | None:
    """Estimated download size for one mp4, or None when it cannot be known.

    `bitrate × duration_millis / 1000 / 8`. A missing duration or a
    `bitrate ∈ {None, 0}` (animated GIFs always report 0) is UNKNOWN — never 0
    bytes. Mirrors `refresh.estimate_download_size`, scoped to a single entry.
    """
    bitrate = entry.bitrate
    duration = entry.duration_millis
    if not bitrate or duration is None:
        return None
    return bitrate * duration // _BITS_PER_BYTE_TIMES_MILLIS_PER_SECOND


def _record_skip(report: VideoReport, entry: _VideoEntry) -> None:
    """Tally a non-eligible video entry into the right skip bucket.

    A downloaded entry passed over (no `--force`) is `already_downloaded`; a
    pending HLS is `skipped_hls`; a pending poster-era (or legacy non-stream) is
    `skipped_poster_era`. A permanently-failed entry not retried this run is NOT
    a skip bucket — it stays a failure, awaiting `--force`.
    """
    if isinstance(entry, MediaVideoDownloaded):
        report.videos_skipped_already_downloaded += 1
        return
    if isinstance(entry, MediaVideoPending):
        if _video_class(entry) == "hls":
            report.videos_skipped_hls += 1
        else:
            report.videos_skipped_poster_era += 1


def _item_video_entries(item: Item) -> list[tuple[int, _VideoEntry]]:
    """The (index, entry) pairs for every video state on `item` (photos elided)."""
    return [
        (index, entry)
        for index, entry in enumerate(item.media)
        if isinstance(entry, (MediaVideoPending, MediaVideoDownloaded, MediaVideoFailed))
    ]


def _iter_eligible_video_attempts(
    items: dict[str, Item],
    *,
    limit: int | None,
    force: bool,
    report: VideoReport,
) -> Iterator[tuple[str, Item, int, _VideoEntry]]:
    """Yield each (item_id, item, index, entry) eligible for video download.

    Bumps `report.items_processed` once per item that carries any video entry,
    and records every non-eligible video entry into its skip bucket via
    `_record_skip`. Photo entries are passed over silently. Stops yielding once
    `limit` eligible entries have been emitted.
    """
    remaining = limit
    for item_id, item in items.items():
        video_entries = _item_video_entries(item)
        if not video_entries:
            continue
        report.items_processed += 1
        for index, entry in video_entries:
            if remaining is not None and remaining <= 0:
                return
            if not _is_video_download_eligible(entry, force=force):
                _record_skip(report, entry)
                continue
            if remaining is not None:
                remaining -= 1
            yield item_id, item, index, entry


def plan_video_downloads(
    items: dict[str, Item],
    *,
    force: bool = False,
    limit: int | None = None,
    items_filter: list[str] | None = None,
) -> VideoDownloadPlan:
    """Preview the download set + size WITHOUT any network — drives the gate.

    Replays the exact eligibility walk `download_videos` performs against a
    throwaway report, then sums `_estimated_bytes` over the eligible mp4 set so
    the operator sees how much is about to be fetched (and how many HLS / poster
    / already-downloaded entries are being passed over) before confirming.
    """
    candidate = _filter_by_ids(items, items_filter)
    probe = VideoReport()
    eligible = list(
        _iter_eligible_video_attempts(candidate, limit=limit, force=force, report=probe)
    )
    estimated_bytes = 0
    n_estimable = 0
    n_unknown = 0
    for _item_id, _item, _index, entry in eligible:
        size = _estimated_bytes(entry)
        if size is None:
            n_unknown += 1
        else:
            estimated_bytes += size
            n_estimable += 1
    return VideoDownloadPlan(
        n_to_download=len(eligible),
        estimated_bytes=estimated_bytes,
        n_estimable=n_estimable,
        n_unknown=n_unknown,
        n_hls_skipped=probe.videos_skipped_hls,
        n_poster_skipped=probe.videos_skipped_poster_era,
        n_already_downloaded=probe.videos_skipped_already_downloaded,
    )


def format_size_gate(plan: VideoDownloadPlan) -> str:
    """The human size-gate line shown before download (the "~X GB" warning).

    Reports the estimated total GB across the to-be-downloaded videos, plus the
    counts of HLS and already-downloaded entries being skipped. When some
    eligible mp4s carry no bitrate/duration their bytes are unknown, surfaced as
    a `+N of unknown size` rider rather than silently understating the total.
    """
    if plan.n_estimable:
        gigabytes = plan.estimated_bytes / 1_000_000_000
        size = f"~{gigabytes:.1f} GB"
        if plan.n_unknown:
            size += f" (+{plan.n_unknown} of unknown size)"
    else:
        size = "an unknown total size"
    return (
        f"About to download {size} across {plan.n_to_download} videos "
        f"({plan.n_hls_skipped} HLS skipped, {plan.n_already_downloaded} already downloaded)."
    )


def download_videos(
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
) -> VideoReport:
    """Download every eligible mp4 across the store; return a structured report.

    Eligibility (without `--force`): a `MediaVideoPending` whose URL is a real
    mp4 stream, or a `MediaVideoFailed` mp4 whose reason is transient. With
    `--force`, downloaded and permanently-failed mp4s are re-attempted too. HLS
    and poster-era entries are skipped and counted (never downloaded). Photo
    variants are ignored.

    Mutates `items` in place — the caller wraps each transition with a
    store-write (the `on_progress` callback fires after every transition, so a
    Ctrl-C between videos leaves `items.json` coherent, never mid-download).
    Files are written under `<media_root>/<item_id>/<index>.mp4`.

    Raises:
        RuntimeError: when EVERY video attempted in the run fails — a total
            failure (CDN outage, expired stream URLs) must surface as a non-zero
            exit, not a silent empty run. The CLI's `_handle_cli_errors` turns
            it into a clean operator message + exit code 1.
    """
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": user_agent})
    _sweep_part_orphans(media_root)
    started = time.monotonic()
    report = VideoReport()
    candidate_items = _filter_by_ids(items, items_filter)

    for item_id, item, index, entry in _iter_eligible_video_attempts(
        candidate_items,
        limit=limit,
        force=force,
        report=report,
    ):
        report.videos_attempted += 1
        result = _download_one_video(
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

    report.elapsed_seconds = time.monotonic() - started
    if report.videos_skipped_hls:
        # mp4-only scope: HLS needs ffmpeg to mux a playable file. The deferred
        # follow-up downloads these — see the module docstring.
        logger.info(
            "download-videos: %d HLS videos skipped (need ffmpeg, deferred follow-up).",
            report.videos_skipped_hls,
        )
    if report.videos_attempted > 0 and report.videos_downloaded == 0:
        raise RuntimeError(
            f"All {report.videos_attempted} video download attempts failed; "
            "check network / video.twimg.com availability and the per-video "
            "warnings above."
        )
    return report


def _record_outcome(
    report: VideoReport,
    *,
    item_id: str,
    entry: MediaVideoDownloaded | MediaVideoFailed,
) -> None:
    """Bump the report counters based on the post-transition variant."""
    if isinstance(entry, MediaVideoDownloaded):
        report.videos_downloaded += 1
        report.bytes_downloaded += entry.bytes_size
        return
    report.per_item_failures.setdefault(item_id, []).append((entry.url, entry.failure_reason))
    if entry.failure_reason in _TRANSIENT_MEDIA_FAILURES:
        report.videos_failed_transient += 1
    else:
        report.videos_failed_permanent += 1
    logger.warning(
        "download-videos: failed item=%s url=%s reason=%s error=%s",
        item_id,
        entry.url,
        entry.failure_reason,
        entry.error,
    )


def _download_one_video(
    entry: _VideoEntry,
    *,
    item_id: str,
    index: int,
    media_root: Path,
    session: requests.Session,
    timeout_seconds: int,
) -> MediaVideoDownloaded | MediaVideoFailed:
    """Download one mp4 (single GET, no Pillow decode) — return the new variant.

    Never raises on a recoverable failure: the `failure_reason` field carries
    the categorisation (mirroring `media._download_one`). Only programmer bugs
    and `KeyboardInterrupt` propagate. A 2xx with an empty body is bucketed as a
    transient `unknown_error` so the next run retries rather than persisting a
    zero-byte "download".
    """
    attempts = (entry.attempts if isinstance(entry, MediaVideoFailed) else 0) + 1
    try:
        response = session.get(entry.url, timeout=timeout_seconds)
    except requests.Timeout as exc:
        return _failed(entry, reason="timeout", error=_format_error(exc, None), attempts=attempts)
    except requests.RequestException as exc:
        # Connection / SSL errors — transient, mirroring `xbrain.fetch`.
        return _failed(
            entry, reason="unknown_error", error=_format_error(exc, None), attempts=attempts
        )

    status = response.status_code
    if not 200 <= status < 300:
        return _failed(
            entry,
            reason=_classify_status(status),
            error=_format_error(None, status),
            attempts=attempts,
        )

    content = response.content
    if not content:
        # 2xx but empty body — not a real download. Transient so we retry.
        return _failed(
            entry, reason="unknown_error", error="empty response body", attempts=attempts
        )
    local_path = _local_path(item_id, index, ".mp4")
    try:
        _write_bytes(media_root / local_path, content)
    except OSError as exc:
        # Disk full / permission denied — transient: the next run retries once
        # the operator clears the condition. Without this guard the OSError
        # escapes per-item bucketing and aborts the whole batch.
        return _failed(
            entry, reason="unknown_error", error=f"local write failed: {exc}", attempts=attempts
        )
    return MediaVideoDownloaded(
        url=entry.url,
        thumbnail_url=entry.thumbnail_url,
        bitrate=entry.bitrate,
        duration_millis=entry.duration_millis,
        local_path=local_path,
        bytes_size=len(content),
        downloaded_at=datetime.now(timezone.utc),
    )


def _failed(
    entry: _VideoEntry,
    *,
    reason: MediaFailureReason,
    error: str | None,
    attempts: int,
) -> MediaVideoFailed:
    """Build a `MediaVideoFailed`, carrying the source url + metadata forward."""
    return MediaVideoFailed(
        url=entry.url,
        thumbnail_url=entry.thumbnail_url,
        bitrate=entry.bitrate,
        duration_millis=entry.duration_millis,
        failure_reason=reason,
        error=error,
        attempts=attempts,
        last_attempt_at=datetime.now(timezone.utc),
    )


def emit_video_summary_line(report: VideoReport, *, out: "io.IOBase | None" = None) -> None:
    """Print the SUMMARY line on stderr (mirrors `media.emit_summary_line`).

    Emitted only when the run did something — at least one attempt, or at least
    one skip (HLS / poster-era / already-downloaded). A fully no-op run (e.g. an
    `--items` filter that matched nothing) stays silent. `out` is injectable for
    tests; defaults to `sys.stderr`.
    """
    did_something = (
        report.videos_attempted
        or report.videos_skipped_hls
        or report.videos_skipped_poster_era
        or report.videos_skipped_already_downloaded
    )
    if not did_something:
        return
    target = out if out is not None else sys.stderr
    print(
        f"SUMMARY: downloaded: {report.videos_downloaded}, "
        f"failed_permanent: {report.videos_failed_permanent}, "
        f"failed_transient: {report.videos_failed_transient}, "
        f"skipped_hls: {report.videos_skipped_hls}, "
        f"skipped_poster_era: {report.videos_skipped_poster_era}, "
        f"already_downloaded: {report.videos_skipped_already_downloaded}, "
        f"bytes: {report.bytes_downloaded:_}",
        file=target,
    )
