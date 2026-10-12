"""Ephemeral streaming mp4 fetch for internal ASR/agent-side processing.

Streams a selected item's real progressive **mp4** to `<dest>/<id>.mp4` for
agent-side processing (transcribe / analyse), then leaves it to the caller to
discard. It is deliberately **store-non-mutating**: it reads the resolved stream
URL off each item's video entry and NEVER writes `items.json`, never takes a
snapshot, and never touches `data/media/` — the only bytes it writes live under
the caller-supplied `dest_dir`.

The download, content-validation and failure-classification are **reused** from
`xbrain.video_media` / `xbrain.media` rather than re-implemented: the mp4/HLS/
poster discriminator (`_video_class`), the size estimator (`_estimated_bytes`),
the mp4-container / `video/*` / interstitial-rejection logic, and the HTTP-status
classification (`_classify_status`) come from the shared primitives. Only the
streaming destination-path policy (`<id>.mp4` in an arbitrary dir), hard cap, and
lightweight non-persisting report are new here.

I/O dependencies (HTTP session, sleep) are keyword-injectable so tests run
offline against a fake session.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
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
)
from xbrain.models import (
    Item,
    MediaVideoDownloaded,
    MediaVideoFailed,
    MediaVideoPending,
)
from xbrain.video_media import _content_type_of, _is_video_response, _video_class
from xbrain.video_select import _is_video_entry

_VideoEntry = MediaVideoPending | MediaVideoDownloaded | MediaVideoFailed
_CHUNK_SIZE = 1024 * 1024

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


def _fetch_one(
    entry: _VideoEntry,
    *,
    item_id: str,
    dest_dir: Path,
    session: requests.Session,
    timeout_seconds: int,
    max_size_bytes: int | None,
) -> FetchResult:
    """Stream one mp4 to `<dest_dir>/<item_id>.mp4` — never raises on failure.

    The normal ASR fallback may fetch long videos. This path must therefore not
    call `response.content`: bytes are streamed to a local `.part`, the first
    bytes are validated against obvious HTML/JSON interstitials, and
    `max_size_bytes` is enforced while streaming. The caller owns the temporary
    parent directory and removes it after transcription.
    """
    try:
        response = session.get(entry.url, timeout=timeout_seconds, stream=True)
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

    path = dest_dir / f"{item_id}.mp4"
    return _stream_response_to_path(
        item_id,
        response,
        path,
        max_size_bytes=max_size_bytes,
    )


def _response_content_length(response: object) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return None
    raw = headers.get("Content-Length") or headers.get("content-length")
    if raw is None:
        return None
    try:
        value = int(str(raw))
    except ValueError:
        return None
    return value if value >= 0 else None


def _response_chunks(response: object) -> Iterable[bytes]:
    iter_content = getattr(response, "iter_content", None)
    if callable(iter_content):
        yield from iter_content(chunk_size=_CHUNK_SIZE)
        return
    yield getattr(response, "content", b"")


def _stream_response_to_path(
    item_id: str,
    response: object,
    path: Path,
    *,
    max_size_bytes: int | None,
) -> FetchResult:
    content_length = _response_content_length(response)
    if (
        max_size_bytes is not None
        and content_length is not None
        and content_length > max_size_bytes
    ):
        return FetchResult(item_id, "skipped", reason="too_large")

    content_type = _content_type_of(response)
    part = path.with_name(f"{path.name}.part")
    total = 0
    head = b""
    validated = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with part.open("wb") as handle:
            for chunk in _response_chunks(response):
                if not chunk:
                    continue
                if not validated:
                    head += chunk
                    if len(head) < 12 and not content_type.lower().startswith("video/"):
                        continue
                    if not _is_video_response(content_type, head):
                        return _stream_failure(
                            item_id,
                            part,
                            f"non-video response (interstitial?): content-type={content_type!r}",
                        )
                    handle.write(head)
                    total += len(head)
                    head = b""
                    validated = True
                else:
                    handle.write(chunk)
                    total += len(chunk)
                if max_size_bytes is not None and total > max_size_bytes:
                    part.unlink(missing_ok=True)
                    return FetchResult(item_id, "skipped", reason="too_large")
            if not validated:
                if not head:
                    return _stream_failure(item_id, part, "empty response body")
                if not _is_video_response(content_type, head):
                    return _stream_failure(
                        item_id,
                        part,
                        f"non-video response (interstitial?): content-type={content_type!r}",
                    )
                handle.write(head)
                total += len(head)
    except requests.Timeout as exc:
        part.unlink(missing_ok=True)
        return FetchResult(item_id, "failed", reason="timeout", error=_format_error(exc, None))
    except requests.RequestException as exc:
        part.unlink(missing_ok=True)
        return FetchResult(
            item_id, "failed", reason="unknown_error", error=_format_error(exc, None)
        )
    except OSError as exc:
        part.unlink(missing_ok=True)
        return FetchResult(
            item_id, "failed", reason="unknown_error", error=f"local write failed: {exc}"
        )

    if total <= 0:
        part.unlink(missing_ok=True)
        return FetchResult(item_id, "failed", reason="unknown_error", error="empty response body")
    try:
        part.replace(path)
    except OSError as exc:
        part.unlink(missing_ok=True)
        return FetchResult(
            item_id, "failed", reason="unknown_error", error=f"local rename failed: {exc}"
        )
    return FetchResult(item_id, "fetched", path=str(path), size_bytes=total)


def _stream_failure(item_id: str, part: Path, error: str) -> FetchResult:
    part.unlink(missing_ok=True)
    return FetchResult(item_id, "failed", reason="unknown_error", error=error)


def _classify_id(store: dict[str, Item], item_id: str) -> FetchResult | _VideoEntry:
    """Decide one id: a skip `FetchResult`, or the `_VideoEntry` to fetch.

    Rejects (each as a skip) an unsafe id (path traversal), an unknown id, an
    item with no downloadable mp4 (`no_video` / `hls` / `poster_era`). It does
    not reject by `bitrate × duration` estimate: X often overstates the real CDN
    payload size. The hard `--max-size` cap is enforced against `Content-Length`
    and while streaming in `_fetch_one`.
    """
    if _is_unsafe_id(item_id):
        return FetchResult(item_id, "skipped", reason="invalid_id")
    item = store.get(item_id)
    if item is None:
        return FetchResult(item_id, "skipped", reason="unknown_item")
    entry, skip_reason = _select_entry(item)
    if entry is None:
        return FetchResult(item_id, "skipped", reason=skip_reason)
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
    HLS/poster-only item is recorded as a skip (never fatal); a failed download
    is recorded and the batch continues. `max_size_bytes` is a hard streaming
    cap: `Content-Length` can skip before reading the body, and a response
    without length is aborted/deleted as soon as streamed bytes exceed the cap.
    `limit` caps the number of real fetch ATTEMPTS (skips do not count against
    it). The HTTP session/UA/throttle and every download/validation primitive are
    reused from `xbrain.video_media` / `xbrain.media`.
    """
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": user_agent})
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    # Deliberately NO `.part`-orphan sweep here: `dest` is the operator's own
    # `--to` directory (possibly ~/Downloads), and recursively unlinking every
    # `*.part` would silently destroy OTHER programs' in-progress downloads. The
    # our streaming writer cleans up its own `<id>.mp4.part` on failure, so an
    # ephemeral fetch leaves no orphan of ours to sweep.

    report = FetchReport()
    attempted = 0
    seen: set[str] = set()
    for item_id in ids:
        if item_id in seen:
            continue
        seen.add(item_id)
        decision = _classify_id(store, item_id)
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
                max_size_bytes=max_size_bytes,
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
