"""Ephemeral mp4 fetch for `xbrain fetch-video`.

Downloads a selected item's real progressive **mp4** to `<dest>/<id>.mp4` for
agent-side processing (transcribe / analyse), then leaves it to the caller to
discard. It is deliberately **store-non-mutating**: it reads the resolved stream
URL off each item's video entry and NEVER writes `items.json`, never takes a
snapshot, and never touches `data/media/` — the only bytes it writes live under
the caller-supplied `dest_dir`.

The download, content-validation and failure-classification are **reused** from
`xbrain.video_media` / `xbrain.media` rather than re-implemented: the mp4/HLS/
poster discriminator (`_video_class`), the size estimator (`_estimated_bytes`),
the 2xx body validation (`_read_validated_body` → the mp4-container /
`video/*` / interstitial-rejection logic), the HTTP-status classification
(`_classify_status`), the atomic write (`_write_bytes`) and the stale-`.part`
sweep (`_sweep_part_orphans`) all come from the shared primitives. Only the
destination-path policy (`<id>.mp4` in an arbitrary dir) and the lightweight,
non-persisting report are new here.

I/O dependencies (HTTP session, sleep) are keyword-injectable so tests run
offline against a fake session.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import requests

from xbrain.media import (
    _DEFAULT_THROTTLE_SECONDS,
    _DEFAULT_TIMEOUT_SECONDS,
    _DEFAULT_UA,
    _classify_status,
    _format_error,
    _write_bytes,
)
from xbrain.models import (
    Item,
    MediaVideoDownloaded,
    MediaVideoFailed,
    MediaVideoPending,
)
from xbrain.video_media import _estimated_bytes, _read_validated_body, _video_class
from xbrain.video_select import _is_video_entry

_VideoEntry = MediaVideoPending | MediaVideoDownloaded | MediaVideoFailed

FetchOutcome = Literal["fetched", "skipped", "failed"]


@dataclass(frozen=True)
class FetchResult:
    """The per-id outcome of a fetch attempt.

    `outcome="fetched"` carries the local `path` + `size_bytes`; `"skipped"`
    carries a `reason` (`unknown_item` / `no_video` / `hls` / `poster_era` /
    `too_large` / `size_unknown` / `invalid_id`); `"failed"` carries a `reason`
    (the reused `MediaFailureReason` classification) + a human `error` detail.
    """

    id: str
    outcome: FetchOutcome
    path: str | None = None
    reason: str | None = None
    error: str | None = None
    size_bytes: int | None = None


@dataclass
class FetchReport:
    """Structured, non-persisting result of a `fetch_videos` run."""

    results: list[FetchResult] = field(default_factory=list)

    @property
    def fetched(self) -> int:
        return sum(1 for r in self.results if r.outcome == "fetched")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.outcome == "skipped")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.outcome == "failed")


def _is_unsafe_id(item_id: str) -> bool:
    """True when using `item_id` as a filename would escape the dest dir.

    A hand-edited `items.json` is untrusted input: an id like `../escaped`, one
    carrying a path separator, or a bare `.`/`..` would make `dest/<id>.mp4`
    write outside `--to`. Real X ids are opaque digit strings, so any separator
    or dot-component is rejected outright (recorded as an `invalid_id` skip).
    """
    if not item_id or item_id in (".", ".."):
        return True
    return any(sep in item_id for sep in ("/", "\\", "\x00"))


def _select_entry(item: Item) -> tuple[_VideoEntry | None, str | None]:
    """Pick the item's first real-mp4 video entry, or a skip reason.

    Returns `(entry, None)` for the first entry whose stream class is `mp4`
    (regardless of pending/downloaded/failed — fetch is independent of store
    state). When the item has no downloadable mp4, returns `(None, reason)`:
    `no_video` (no video entry at all), else `hls` / `poster_era` for the
    non-mp4 stream that is present.
    """
    video_entries = [entry for entry in item.media if _is_video_entry(entry)]
    if not video_entries:
        return None, "no_video"
    for entry in video_entries:
        if _video_class(entry) == "mp4":
            return entry, None
    classes = {_video_class(entry) for entry in video_entries}
    return None, ("hls" if "hls" in classes else "poster_era")


def _cap_skip_reason(entry: _VideoEntry, max_size_bytes: int | None) -> str | None:
    """The `--max-size` skip reason for `entry`, or None when it may be fetched.

    No cap → None. With a cap, an UNKNOWN estimate is `size_unknown` (we cannot
    prove it fits) and an over-cap estimate is `too_large` — mirroring the
    conservative `download-videos` rule so list/fetch select the same set.
    """
    if max_size_bytes is None:
        return None
    estimate = _estimated_bytes(entry)
    if estimate is None:
        return "size_unknown"
    if estimate > max_size_bytes:
        return "too_large"
    return None


def _fetch_one(
    entry: _VideoEntry,
    *,
    item_id: str,
    dest_dir: Path,
    session: requests.Session,
    timeout_seconds: int,
) -> FetchResult:
    """Download one mp4 to `<dest_dir>/<item_id>.mp4` — never raises on failure.

    Reuses the shared body-validation (`_read_validated_body`) and status
    classification (`_classify_status`); a recoverable failure is returned as a
    `FetchResult(outcome="failed", ...)` so the batch continues.
    """
    try:
        response = session.get(entry.url, timeout=timeout_seconds)
    except requests.Timeout as exc:
        return FetchResult(item_id, "failed", reason="timeout", error=_format_error(exc, None))
    except requests.RequestException as exc:
        return FetchResult(
            item_id, "failed", reason="unknown_error", error=_format_error(exc, None)
        )

    status = response.status_code
    if not 200 <= status < 300:
        return FetchResult(
            item_id,
            "failed",
            reason=_classify_status(status),
            error=_format_error(None, status),
        )

    body = _read_validated_body(entry, response, attempts=1)
    if isinstance(body, MediaVideoFailed):
        return FetchResult(item_id, "failed", reason=body.failure_reason, error=body.error)

    path = dest_dir / f"{item_id}.mp4"
    try:
        _write_bytes(path, body)
    except OSError as exc:
        return FetchResult(
            item_id, "failed", reason="unknown_error", error=f"local write failed: {exc}"
        )
    return FetchResult(item_id, "fetched", path=str(path), size_bytes=len(body))


def _classify_id(
    store: dict[str, Item], item_id: str, max_size_bytes: int | None
) -> FetchResult | _VideoEntry:
    """Decide one id: a skip `FetchResult`, or the `_VideoEntry` to fetch.

    Rejects (each as a skip) an unsafe id (path traversal), an unknown id, an
    item with no downloadable mp4 (`no_video` / `hls` / `poster_era`), and an
    over-`--max-size` / unknown-size entry. Otherwise returns the mp4 entry.
    Pulled out of `fetch_videos` so the per-id branching does not inflate the
    orchestration loop's complexity.
    """
    if _is_unsafe_id(item_id):
        return FetchResult(item_id, "skipped", reason="invalid_id")
    item = store.get(item_id)
    if item is None:
        return FetchResult(item_id, "skipped", reason="unknown_item")
    entry, skip_reason = _select_entry(item)
    if entry is None:
        return FetchResult(item_id, "skipped", reason=skip_reason)
    cap_reason = _cap_skip_reason(entry, max_size_bytes)
    if cap_reason is not None:
        return FetchResult(item_id, "skipped", reason=cap_reason)
    return entry


def fetch_videos(
    store: dict[str, Item],
    ids: list[str],
    dest_dir: Path | str,
    *,
    max_size_bytes: int | None = None,
    limit: int | None = None,
    throttle_seconds: float = _DEFAULT_THROTTLE_SECONDS,
    user_agent: str = _DEFAULT_UA,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchReport:
    """Fetch each selected item's real mp4 into `dest_dir` as `<id>.mp4`.

    Ephemeral + store-non-mutating: writes ONLY under `dest_dir`, never
    `items.json` / `data/media/`, and takes no snapshot. `ids` are processed in
    order, de-duplicated (the same video is fetched once). A missing item, an
    HLS/poster-only item, or an over-`--max-size`/unknown-size item is recorded
    as a skip (never fatal); a failed download is recorded and the batch
    continues. `limit` caps the number of real fetch ATTEMPTS (skips do not
    count against it). The HTTP session/UA/throttle and every download/validation
    primitive are reused from `xbrain.video_media` / `xbrain.media`.
    """
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": user_agent})
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    # Deliberately NO `.part`-orphan sweep here: `dest` is the operator's own
    # `--to` directory (possibly ~/Downloads), and recursively unlinking every
    # `*.part` would silently destroy OTHER programs' in-progress downloads. The
    # atomic `_write_bytes` already cleans up its own `.part` on failure, so an
    # ephemeral fetch leaves no orphan of ours to sweep.

    report = FetchReport()
    attempted = 0
    seen: set[str] = set()
    for item_id in ids:
        if item_id in seen:
            continue
        seen.add(item_id)
        decision = _classify_id(store, item_id, max_size_bytes)
        if isinstance(decision, FetchResult):
            report.results.append(decision)
            continue
        if limit is not None and attempted >= limit:
            break
        attempted += 1
        report.results.append(
            _fetch_one(
                decision,
                item_id=item_id,
                dest_dir=dest,
                session=session,
                timeout_seconds=timeout_seconds,
            )
        )
        if throttle_seconds > 0:
            sleep(throttle_seconds)
    return report


def format_fetch_summary(report: FetchReport) -> str:
    """One-line human SUMMARY of a fetch run (mirrors the download summaries)."""
    return (
        f"Vídeos: descargados {report.fetched}, saltados {report.skipped}, fallidos {report.failed}"
    )


def fetch_result_to_json(result: FetchResult) -> dict[str, object]:
    """Serialise a `FetchResult` to a stable machine dict for `--json`."""
    return {
        "id": result.id,
        "outcome": result.outcome,
        "path": result.path,
        "reason": result.reason,
        "error": result.error,
        "size_bytes": result.size_bytes,
    }
