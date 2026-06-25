"""Tests for `xbrain.video_media` — the mp4 video download orchestrator.

The file-download counterpart to `tests/test_media.py` (photos). HTTP is
mocked via a hand-rolled `FakeSession` (no real network). Videos carry no
Pillow decode — the body is opaque bytes, so the fakes return arbitrary
non-empty payloads. mp4-only this PR: HLS (`.m3u8`) entries are skipped and
counted (the ffmpeg follow-up handles them); poster-era entries (un-backfilled,
`url == thumbnail_url`) are skipped silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.models import (
    Author,
    Item,
    MediaPhotoDownloaded,
    MediaVideoDownloaded,
    MediaVideoFailed,
    MediaVideoPending,
)
from xbrain.video_media import (
    _TRANSIENT_MEDIA_FAILURES,
    VideoDownloadPlan,
    VideoReport,
    _is_video_download_eligible,
    _video_class,
    download_videos,
    emit_video_summary_line,
    format_size_gate,
    plan_video_downloads,
)

# --------------------------------------------------------------------- fakes


@dataclass
class FakeResponse:
    """A minimal stand-in for `requests.Response` — `status_code` + `content`."""

    status_code: int
    content: bytes = b""


@dataclass
class FakeSession:
    """Fake `requests.Session.get` returning a canned response per URL.

    `responses` is keyed by a URL substring; each call pops the first queued
    entry off the matching list (so a retry can queue a second response).
    `raise_for` maps a URL substring to an exception raised the first time
    that URL is hit (then cleared).
    """

    responses: dict[str, list[FakeResponse]] = field(default_factory=dict)
    raise_for: dict[str, Exception] = field(default_factory=dict)
    calls: list[tuple[str, int]] = field(default_factory=list)

    def get(self, url: str, *, timeout: int) -> FakeResponse:
        self.calls.append((url, timeout))
        for key, exc in list(self.raise_for.items()):
            if key in url:
                del self.raise_for[key]
                raise exc
        for matcher, queue in self.responses.items():
            if matcher in url and queue:
                return queue.pop(0)
        return FakeResponse(status_code=404, content=b"")


def _mp4_bytes(size: int = 2048) -> bytes:
    """Opaque non-empty mp4-ish payload — videos are not decoded, only written."""
    return b"\x00\x00\x00\x18ftypmp42" + (b"\x00" * size)


_MP4_URL = "https://video.twimg.com/ext_tw_video/1/vid/1280x720/A.mp4?tag=12"
_HLS_URL = "https://video.twimg.com/ext_tw_video/1/pl/B.m3u8?container=fmp4"
_POSTER = "https://pbs.twimg.com/ext_tw_video_thumb/1/img/P.jpg"


def _video_pending(
    url: str = _MP4_URL,
    *,
    thumbnail_url: str | None = _POSTER,
    bitrate: int | None = 2_176_000,
    duration_millis: int | None = 30_000,
) -> MediaVideoPending:
    return MediaVideoPending(
        url=url,
        thumbnail_url=thumbnail_url,
        bitrate=bitrate,
        duration_millis=duration_millis,
    )


def _item_with_media(media_entries: list, item_id: str = "123") -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        media=media_entries,
    )


# --------------------------------------------------------------------- classification


def test_video_class_recognises_real_mp4_by_host():
    """A video.twimg.com URL is an mp4 stream even without an `.mp4` suffix
    visible before the query string is involved."""
    assert _video_class(_video_pending(url=_MP4_URL)) == "mp4"


def test_video_class_recognises_mp4_by_path_suffix_other_host():
    """A `.mp4` path on any host (before the query) is an mp4 stream."""
    entry = _video_pending(url="https://cdn.example.com/v/clip.mp4?x=1", thumbnail_url=_POSTER)
    assert _video_class(entry) == "mp4"


def test_video_class_recognises_hls_even_on_video_host():
    """An `.m3u8` manifest is HLS — the `.m3u8` check must win over the
    video.twimg.com host check, or HLS would be misread as a downloadable mp4."""
    assert _video_class(_video_pending(url=_HLS_URL)) == "hls"


def test_video_class_poster_era_when_url_equals_thumbnail():
    """`url == thumbnail_url` is the poster-fallback (un-backfilled) marker."""
    entry = _video_pending(url=_POSTER, thumbnail_url=_POSTER)
    assert _video_class(entry) == "poster"


def test_video_class_legacy_record_without_thumbnail_is_poster():
    """A legacy pbs.jpg record (no thumbnail, not an mp4/HLS URL) is poster-era —
    it has not been backfilled with a playable stream yet, so never download it."""
    entry = _video_pending(url=_POSTER, thumbnail_url=None, bitrate=None, duration_millis=None)
    assert _video_class(entry) == "poster"


# --------------------------------------------------------------------- eligibility


def test_eligible_pending_mp4_true_hls_and_poster_false():
    assert _is_video_download_eligible(_video_pending(url=_MP4_URL), force=False) is True
    assert _is_video_download_eligible(_video_pending(url=_HLS_URL), force=False) is False
    poster = _video_pending(url=_POSTER, thumbnail_url=_POSTER)
    assert _is_video_download_eligible(poster, force=False) is False


def test_eligible_downloaded_only_with_force():
    downloaded = MediaVideoDownloaded(
        url=_MP4_URL,
        thumbnail_url=_POSTER,
        bitrate=2_176_000,
        duration_millis=30_000,
        local_path="123/0.mp4",
        bytes_size=100,
        downloaded_at=datetime.now(timezone.utc),
    )
    assert _is_video_download_eligible(downloaded, force=False) is False
    assert _is_video_download_eligible(downloaded, force=True) is True


def test_eligible_failed_transient_vs_permanent():
    transient = MediaVideoFailed(
        url=_MP4_URL,
        failure_reason="http_5xx",
        attempts=1,
        last_attempt_at=datetime.now(timezone.utc),
    )
    permanent = MediaVideoFailed(
        url=_MP4_URL,
        failure_reason="http_4xx",
        attempts=1,
        last_attempt_at=datetime.now(timezone.utc),
    )
    assert _is_video_download_eligible(transient, force=False) is True
    assert _is_video_download_eligible(permanent, force=False) is False
    assert _is_video_download_eligible(permanent, force=True) is True


def test_eligible_ignores_photo_entries():
    photo = MediaPhotoDownloaded(
        url="https://pbs.twimg.com/media/X.png",
        local_path="1/0.png",
        width=4,
        height=3,
        bytes_size=10,
        downloaded_at=datetime.now(timezone.utc),
    )
    assert _is_video_download_eligible(photo, force=True) is False


# --------------------------------------------------------------------- plan / size gate


def test_plan_estimates_only_the_eligible_mp4_set():
    """The estimate is scoped to the videos this run will actually download —
    HLS, poster-era, and already-downloaded entries are counted, not summed."""
    items = {
        "1": _item_with_media([_video_pending(url=_MP4_URL)], "1"),  # 8_160_000 bytes
        "2": _item_with_media([_video_pending(url=_HLS_URL)], "2"),  # HLS skip
        "3": _item_with_media(
            [_video_pending(url=_POSTER, thumbnail_url=_POSTER)], "3"
        ),  # poster skip
    }
    plan = plan_video_downloads(items, force=False)
    assert plan.n_to_download == 1
    # 2_176_000 b/s * 30 s / 8 = 8_160_000 bytes.
    assert plan.estimated_bytes == 8_160_000
    assert plan.n_estimable == 1
    assert plan.n_unknown == 0
    assert plan.n_hls_skipped == 1
    assert plan.n_poster_skipped == 1
    assert plan.n_already_downloaded == 0


def test_plan_counts_already_downloaded_without_force():
    downloaded = MediaVideoDownloaded(
        url=_MP4_URL,
        local_path="1/0.mp4",
        bytes_size=100,
        downloaded_at=datetime.now(timezone.utc),
    )
    items = {"1": _item_with_media([downloaded], "1")}
    plan = plan_video_downloads(items, force=False)
    assert plan.n_to_download == 0
    assert plan.n_already_downloaded == 1


def test_plan_counts_unknown_size_eligible_mp4():
    """An eligible mp4 with no bitrate/duration is unknown, never summed as 0."""
    entry = _video_pending(url=_MP4_URL, bitrate=None, duration_millis=None)
    items = {"1": _item_with_media([entry], "1")}
    plan = plan_video_downloads(items, force=False)
    assert plan.n_to_download == 1
    assert plan.estimated_bytes == 0
    assert plan.n_estimable == 0
    assert plan.n_unknown == 1


def test_plan_respects_limit_and_items_filter():
    items = {
        "1": _item_with_media([_video_pending(url=_MP4_URL)], "1"),
        "2": _item_with_media([_video_pending(url=_MP4_URL)], "2"),
    }
    plan = plan_video_downloads(items, force=False, items_filter=["2"])
    assert plan.n_to_download == 1
    plan_limited = plan_video_downloads(items, force=False, limit=1)
    assert plan_limited.n_to_download == 1


def test_format_size_gate_reports_gb_and_context():
    plan = VideoDownloadPlan(
        n_to_download=3,
        estimated_bytes=8_160_000,
        n_estimable=3,
        n_unknown=0,
        n_hls_skipped=2,
        n_poster_skipped=4,
        n_already_downloaded=1,
    )
    line = format_size_gate(plan)
    assert "~0.0 GB" in line
    assert "3 videos" in line
    assert "2 HLS skipped" in line
    assert "1 already downloaded" in line


def test_format_size_gate_unknown_total_when_nothing_estimable():
    plan = VideoDownloadPlan(n_to_download=2, estimated_bytes=0, n_estimable=0, n_unknown=2)
    line = format_size_gate(plan)
    assert "unknown" in line.lower()
    assert "2 videos" in line


# --------------------------------------------------------------------- download_videos


def test_download_videos_downloads_pending_mp4(tmp_path: Path):
    """A pending real mp4 downloads cleanly and lands as MediaVideoDownloaded."""
    data = _mp4_bytes()
    item = _item_with_media([_video_pending(url=_MP4_URL)])
    session = FakeSession(responses={".mp4": [FakeResponse(200, data)]})
    report = download_videos(
        {"123": item}, media_root=tmp_path, session=session, throttle_seconds=0
    )
    entry = item.media[0]
    assert isinstance(entry, MediaVideoDownloaded)
    assert entry.local_path == "123/0.mp4"
    assert entry.bytes_size == len(data)
    assert entry.bitrate == 2_176_000
    assert entry.thumbnail_url == _POSTER
    assert (tmp_path / "123" / "0.mp4").exists()
    assert report.videos_downloaded == 1
    assert report.bytes_downloaded == len(data)


def test_download_videos_skips_hls_without_calling_http(tmp_path: Path):
    """HLS (`.m3u8`) is counted and skipped — the ffmpeg follow-up handles it."""
    item = _item_with_media([_video_pending(url=_HLS_URL)])
    session = FakeSession()
    report = download_videos(
        {"123": item}, media_root=tmp_path, session=session, throttle_seconds=0
    )
    assert report.videos_skipped_hls == 1
    assert report.videos_attempted == 0
    assert isinstance(item.media[0], MediaVideoPending)
    assert session.calls == []


def test_download_videos_skips_poster_era_silently(tmp_path: Path):
    """A poster-era entry (`url == thumbnail_url`) is counted, never downloaded."""
    item = _item_with_media([_video_pending(url=_POSTER, thumbnail_url=_POSTER)])
    session = FakeSession()
    report = download_videos(
        {"123": item}, media_root=tmp_path, session=session, throttle_seconds=0
    )
    assert report.videos_skipped_poster_era == 1
    assert report.videos_attempted == 0
    assert session.calls == []


def test_download_videos_idempotent_for_already_downloaded(tmp_path: Path):
    downloaded = MediaVideoDownloaded(
        url=_MP4_URL,
        local_path="123/0.mp4",
        bytes_size=100,
        downloaded_at=datetime.now(timezone.utc),
    )
    item = _item_with_media([downloaded])
    session = FakeSession()
    report = download_videos(
        {"123": item}, media_root=tmp_path, session=session, throttle_seconds=0
    )
    assert report.videos_attempted == 0
    assert report.videos_skipped_already_downloaded == 1
    assert session.calls == []


def test_download_videos_force_redownloads(tmp_path: Path):
    downloaded = MediaVideoDownloaded(
        url=_MP4_URL,
        local_path="123/0.mp4",
        bytes_size=5,
        downloaded_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
    )
    item = _item_with_media([downloaded])
    new = _mp4_bytes(4096)
    session = FakeSession(responses={".mp4": [FakeResponse(200, new)]})
    report = download_videos(
        {"123": item}, media_root=tmp_path, session=session, throttle_seconds=0, force=True
    )
    entry = item.media[0]
    assert isinstance(entry, MediaVideoDownloaded)
    assert entry.bytes_size == len(new)
    assert report.videos_downloaded == 1


def test_download_videos_records_http_4xx_permanent(tmp_path: Path):
    """A 404 lands the video in the permanent bucket (partial success → no raise)."""
    items = {
        "1": _item_with_media([_video_pending(url="https://video.twimg.com/d/dead.mp4")], "1"),
        "2": _item_with_media([_video_pending(url="https://video.twimg.com/d/ok.mp4")], "2"),
    }
    session = FakeSession(
        responses={
            "dead.mp4": [FakeResponse(404, b"")],
            "ok.mp4": [FakeResponse(200, _mp4_bytes())],
        }
    )
    report = download_videos(items, media_root=tmp_path, session=session, throttle_seconds=0)
    failed = items["1"].media[0]
    assert isinstance(failed, MediaVideoFailed)
    assert failed.failure_reason == "http_4xx"
    assert failed.attempts == 1
    assert report.videos_failed_permanent == 1
    assert report.videos_downloaded == 1


def test_download_videos_records_http_5xx_transient(tmp_path: Path):
    items = {
        "1": _item_with_media([_video_pending(url="https://video.twimg.com/d/boom.mp4")], "1"),
        "2": _item_with_media([_video_pending(url="https://video.twimg.com/d/ok.mp4")], "2"),
    }
    session = FakeSession(
        responses={
            "boom.mp4": [FakeResponse(503, b"")],
            "ok.mp4": [FakeResponse(200, _mp4_bytes())],
        }
    )
    report = download_videos(items, media_root=tmp_path, session=session, throttle_seconds=0)
    failed = items["1"].media[0]
    assert isinstance(failed, MediaVideoFailed)
    assert failed.failure_reason == "http_5xx"
    assert failed.failure_reason in _TRANSIENT_MEDIA_FAILURES
    assert report.videos_failed_transient == 1


def test_download_videos_timeout_is_transient(tmp_path: Path):
    import requests

    item = _item_with_media([_video_pending(url=_MP4_URL)])

    class _AlwaysTimeout:
        def get(self, url, *, timeout):
            raise requests.Timeout("connect timeout")

    with pytest.raises(RuntimeError):
        download_videos(
            {"123": item}, media_root=tmp_path, session=_AlwaysTimeout(), throttle_seconds=0
        )
    entry = item.media[0]
    assert isinstance(entry, MediaVideoFailed)
    assert entry.failure_reason == "timeout"


def test_download_videos_connection_error_is_unknown(tmp_path: Path):
    import requests

    item = _item_with_media([_video_pending(url=_MP4_URL)])

    class _AlwaysConnError:
        def get(self, url, *, timeout):
            raise requests.ConnectionError("ECONNREFUSED")

    with pytest.raises(RuntimeError):
        download_videos(
            {"123": item}, media_root=tmp_path, session=_AlwaysConnError(), throttle_seconds=0
        )
    entry = item.media[0]
    assert isinstance(entry, MediaVideoFailed)
    assert entry.failure_reason == "unknown_error"
    assert entry.failure_reason in _TRANSIENT_MEDIA_FAILURES


def test_download_videos_empty_body_is_transient_unknown(tmp_path: Path):
    """A 200 with no body is not a real download — bucket as transient so the
    next run retries rather than persisting a zero-byte 'downloaded' record."""
    items = {
        "1": _item_with_media([_video_pending(url="https://video.twimg.com/d/empty.mp4")], "1"),
        "2": _item_with_media([_video_pending(url="https://video.twimg.com/d/ok.mp4")], "2"),
    }
    session = FakeSession(
        responses={
            "empty.mp4": [FakeResponse(200, b"")],
            "ok.mp4": [FakeResponse(200, _mp4_bytes())],
        }
    )
    report = download_videos(items, media_root=tmp_path, session=session, throttle_seconds=0)
    failed = items["1"].media[0]
    assert isinstance(failed, MediaVideoFailed)
    assert failed.failure_reason == "unknown_error"
    assert report.videos_failed_transient == 1


def test_download_videos_raises_on_total_failure(tmp_path: Path):
    item = _item_with_media([_video_pending(url="https://video.twimg.com/d/x.mp4")])
    session = FakeSession(responses={"x.mp4": [FakeResponse(404, b"")]})
    with pytest.raises(RuntimeError, match="All 1 video download attempts failed"):
        download_videos({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    assert isinstance(item.media[0], MediaVideoFailed)


def test_download_videos_retries_transient_failed_next_run(tmp_path: Path):
    item = _item_with_media(
        [
            MediaVideoFailed(
                url=_MP4_URL,
                thumbnail_url=_POSTER,
                bitrate=2_176_000,
                duration_millis=30_000,
                failure_reason="http_5xx",
                attempts=1,
                last_attempt_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            )
        ]
    )
    session = FakeSession(responses={".mp4": [FakeResponse(200, _mp4_bytes())]})
    report = download_videos(
        {"123": item}, media_root=tmp_path, session=session, throttle_seconds=0
    )
    entry = item.media[0]
    assert isinstance(entry, MediaVideoDownloaded)
    assert report.videos_downloaded == 1


def test_download_videos_failed_retry_bumps_attempts(tmp_path: Path):
    items = {
        "1": _item_with_media(
            [
                MediaVideoFailed(
                    url="https://video.twimg.com/d/still.mp4",
                    failure_reason="http_5xx",
                    attempts=2,
                    last_attempt_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
                )
            ],
            "1",
        ),
        "2": _item_with_media([_video_pending(url="https://video.twimg.com/d/ok.mp4")], "2"),
    }
    session = FakeSession(
        responses={
            "still.mp4": [FakeResponse(503, b"")],
            "ok.mp4": [FakeResponse(200, _mp4_bytes())],
        }
    )
    download_videos(items, media_root=tmp_path, session=session, throttle_seconds=0)
    failed = items["1"].media[0]
    assert isinstance(failed, MediaVideoFailed)
    assert failed.attempts == 3


def test_download_videos_skips_permanent_failed_without_force(tmp_path: Path):
    item = _item_with_media(
        [
            MediaVideoFailed(
                url=_MP4_URL,
                failure_reason="http_4xx",
                attempts=1,
                last_attempt_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            )
        ]
    )
    session = FakeSession()
    report = download_videos(
        {"123": item}, media_root=tmp_path, session=session, throttle_seconds=0
    )
    assert report.videos_attempted == 0
    assert session.calls == []


def test_download_videos_respects_limit(tmp_path: Path):
    item = _item_with_media(
        [
            _video_pending(url="https://video.twimg.com/d/v1.mp4"),
            _video_pending(url="https://video.twimg.com/d/v2.mp4"),
        ]
    )
    session = FakeSession(
        responses={
            "v1.mp4": [FakeResponse(200, _mp4_bytes())],
            "v2.mp4": [FakeResponse(200, _mp4_bytes())],
        }
    )
    report = download_videos(
        {"123": item}, media_root=tmp_path, session=session, throttle_seconds=0, limit=1
    )
    assert report.videos_attempted == 1
    assert isinstance(item.media[1], MediaVideoPending)


def test_download_videos_filters_by_items(tmp_path: Path):
    items = {
        "1": _item_with_media([_video_pending(url="https://video.twimg.com/d/m1.mp4")], "1"),
        "2": _item_with_media([_video_pending(url="https://video.twimg.com/d/m2.mp4")], "2"),
    }
    session = FakeSession(
        responses={
            "m1.mp4": [FakeResponse(200, _mp4_bytes())],
            "m2.mp4": [FakeResponse(200, _mp4_bytes())],
        }
    )
    report = download_videos(
        items, media_root=tmp_path, session=session, throttle_seconds=0, items_filter=["2"]
    )
    assert report.videos_downloaded == 1
    assert isinstance(items["1"].media[0], MediaVideoPending)
    assert isinstance(items["2"].media[0], MediaVideoDownloaded)


def test_download_videos_throttles_between_requests(tmp_path: Path):
    sleep_calls: list[float] = []
    item = _item_with_media(
        [
            _video_pending(url="https://video.twimg.com/d/k1.mp4"),
            _video_pending(url="https://video.twimg.com/d/k2.mp4"),
        ]
    )
    session = FakeSession(
        responses={
            "k1.mp4": [FakeResponse(200, _mp4_bytes())],
            "k2.mp4": [FakeResponse(200, _mp4_bytes())],
        }
    )
    download_videos(
        {"123": item},
        media_root=tmp_path,
        session=session,
        throttle_seconds=0.25,
        sleep=sleep_calls.append,
    )
    assert sleep_calls == [0.25, 0.25]


def test_download_videos_invokes_progress_callback_per_transition(tmp_path: Path):
    progress: list[int] = []
    item = _item_with_media(
        [
            _video_pending(url="https://video.twimg.com/d/l1.mp4"),
            _video_pending(url="https://video.twimg.com/d/l2.mp4"),
        ]
    )
    session = FakeSession(
        responses={
            "l1.mp4": [FakeResponse(200, _mp4_bytes())],
            "l2.mp4": [FakeResponse(200, _mp4_bytes())],
        }
    )
    download_videos(
        {"123": item},
        media_root=tmp_path,
        session=session,
        throttle_seconds=0,
        on_progress=lambda: progress.append(1),
    )
    assert len(progress) == 2


def test_download_videos_never_touches_photo_entries(tmp_path: Path):
    photo = MediaPhotoDownloaded(
        url="https://pbs.twimg.com/media/X.png",
        local_path="123/0.png",
        width=4,
        height=3,
        bytes_size=10,
        downloaded_at=datetime.now(timezone.utc),
    )
    item = _item_with_media([photo])
    session = FakeSession()
    report = download_videos(
        {"123": item}, media_root=tmp_path, session=session, throttle_seconds=0
    )
    assert report.videos_attempted == 0
    assert report.items_processed == 0
    assert isinstance(item.media[0], MediaPhotoDownloaded)
    assert session.calls == []


def test_download_videos_propagates_keyboard_interrupt(tmp_path: Path):
    class _CtrlC:
        def get(self, url, *, timeout):
            raise KeyboardInterrupt

    item = _item_with_media([_video_pending(url=_MP4_URL)])
    with pytest.raises(KeyboardInterrupt):
        download_videos({"123": item}, media_root=tmp_path, session=_CtrlC(), throttle_seconds=0)


def test_download_videos_writes_bytes_atomically(tmp_path: Path):
    item = _item_with_media([_video_pending(url=_MP4_URL)])
    session = FakeSession(responses={".mp4": [FakeResponse(200, _mp4_bytes())]})
    download_videos({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    assert sorted(p.name for p in (tmp_path / "123").iterdir()) == ["0.mp4"]


def test_download_videos_sweeps_part_orphans_on_entry(tmp_path: Path):
    orphan_dir = tmp_path / "orphan"
    orphan_dir.mkdir(parents=True)
    orphan = orphan_dir / "0.mp4.part"
    orphan.write_bytes(b"stale junk")
    item = _item_with_media([_video_pending(url=_MP4_URL)])
    session = FakeSession(responses={".mp4": [FakeResponse(200, _mp4_bytes())]})
    download_videos({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    assert not orphan.exists()


def test_download_videos_local_write_failure_is_transient(tmp_path: Path, monkeypatch):
    """A disk-full between write and rename buckets as transient (retryable) and
    leaves no half-written file."""
    items = {
        "1": _item_with_media([_video_pending(url="https://video.twimg.com/d/fail.mp4")], "1"),
        "2": _item_with_media([_video_pending(url="https://video.twimg.com/d/ok.mp4")], "2"),
    }
    session = FakeSession(
        responses={
            "fail.mp4": [FakeResponse(200, _mp4_bytes())],
            "ok.mp4": [FakeResponse(200, _mp4_bytes())],
        }
    )
    original_replace = Path.replace

    def _conditional_fail(self, target):
        if str(target).endswith("/1/0.mp4"):
            raise OSError("simulated disk full")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", _conditional_fail)
    report = download_videos(items, media_root=tmp_path, session=session, throttle_seconds=0)
    failed = items["1"].media[0]
    assert isinstance(failed, MediaVideoFailed)
    assert failed.failure_reason == "unknown_error"
    item_dir = tmp_path / "1"
    if item_dir.exists():
        names = sorted(p.name for p in item_dir.iterdir())
        assert "0.mp4.part" not in names
        assert "0.mp4" not in names
    assert report.videos_failed_transient == 1


# --------------------------------------------------------------------- summary line


def test_emit_video_summary_line_silent_when_nothing_done(capsys):
    emit_video_summary_line(VideoReport())
    assert capsys.readouterr().err == ""


def test_emit_video_summary_line_includes_all_counters(capsys):
    report = VideoReport(
        videos_attempted=10,
        videos_downloaded=7,
        videos_failed_permanent=1,
        videos_failed_transient=2,
        videos_skipped_hls=3,
        videos_skipped_poster_era=4,
        videos_skipped_already_downloaded=5,
        bytes_downloaded=2_048_000,
    )
    emit_video_summary_line(report)
    err = capsys.readouterr().err
    assert "SUMMARY: " in err
    assert "downloaded: 7" in err
    assert "failed_permanent: 1" in err
    assert "failed_transient: 2" in err
    assert "skipped_hls: 3" in err
    assert "skipped_poster_era: 4" in err
    assert "already_downloaded: 5" in err
    assert "2_048_000" in err


def test_emit_video_summary_line_emits_when_only_hls_skipped(capsys):
    """A run that only skipped HLS (zero attempts) still reports — ops needs to
    see that N videos are deferred to the ffmpeg follow-up."""
    emit_video_summary_line(VideoReport(videos_skipped_hls=2))
    assert "skipped_hls: 2" in capsys.readouterr().err
