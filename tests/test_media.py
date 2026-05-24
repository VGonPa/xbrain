"""Tests for `xbrain.media` — the photo download orchestrator.

HTTP is mocked via a hand-rolled `FakeSession` (no `requests-mock` dependency
on the unit path — keeps the test surface dependency-free and obvious).
Pillow is exercised against real PNG/JPEG bytes generated inline.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from xbrain.media import (
    _TRANSIENT_MEDIA_FAILURES,
    MediaReport,
    _is_eligible,
    _local_path,
    _url_with_name,
    download_all,
    emit_summary_line,
)
from xbrain.models import (
    Author,
    Item,
    MediaPhotoDownloaded,
    MediaPhotoFailed,
    MediaPhotoPending,
    MediaVideoPending,
)

# --------------------------------------------------------------------- fakes


@dataclass
class FakeResponse:
    """A minimal stand-in for `requests.Response` — only `status_code` + `content`."""

    status_code: int
    content: bytes = b""


@dataclass
class FakeSession:
    """Sequenced fake `requests.Session.get` returning canned responses.

    `responses` is keyed by URL (after `name=` rewriting); each call pops
    the first entry off the list, so a test can simulate a cascade
    (orig 404 → large 200) by queueing two entries on the same URL prefix.
    `raise_for` maps a URL substring to an exception class that should be
    raised the first time that URL is hit (then cleared).
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
        # Match by base-URL (everything before the `?`) so the cascade's
        # rewritten `name=` parameter does not get in the way.
        base = url.split("?", 1)[0]
        for matcher, queue in self.responses.items():
            if matcher in base or matcher in url:
                if not queue:
                    continue
                return queue.pop(0)
        # Default to a 404 so an under-specified test fails noisily.
        return FakeResponse(status_code=404, content=b"")


def _png_bytes(width: int = 4, height: int = 3) -> bytes:
    """Return a valid PNG with the requested dimensions for Pillow validation."""
    image = Image.new("RGB", (width, height), color=(123, 45, 67))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _item_with_media(media_entries: list) -> Item:
    """Build an Item populated with the given media entries (no real text)."""
    return Item(
        id="123",
        source="bookmark",
        url="https://x.com/a/status/123",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        media=media_entries,
    )


# --------------------------------------------------------------------- pure helpers


def test_url_with_name_sets_size_parameter():
    """`name=` is added when absent and overwritten when present."""
    rewritten = _url_with_name("https://pbs.twimg.com/media/X.jpg", "orig")
    assert "name=orig" in rewritten
    rewritten_existing = _url_with_name("https://pbs.twimg.com/media/X.jpg?name=small", "large")
    assert "name=large" in rewritten_existing
    assert "name=small" not in rewritten_existing


def test_local_path_is_deterministic_relative_string():
    """`<id>/<index><ext>` with forward slashes regardless of OS."""
    assert _local_path("123", 0, ".jpg") == "123/0.jpg"
    assert _local_path("abc", 2, ".png") == "abc/2.png"


def test_is_eligible_pending_always_true():
    pending = MediaPhotoPending(url="u")
    assert _is_eligible(pending, force=False) is True
    assert _is_eligible(pending, force=True) is True


def test_is_eligible_video_pending_never_attempted():
    """Phase A is photos only; videos stay in their pending variant always."""
    video = MediaVideoPending(url="u")
    assert _is_eligible(video, force=False) is False
    assert _is_eligible(video, force=True) is False


def test_is_eligible_downloaded_only_with_force():
    """Idempotency: a downloaded photo is skipped unless `--force` is passed."""
    downloaded = MediaPhotoDownloaded(
        url="u",
        local_path="123/0.jpg",
        width=4,
        height=3,
        bytes_size=100,
        downloaded_at=datetime.now(timezone.utc),
    )
    assert _is_eligible(downloaded, force=False) is False
    assert _is_eligible(downloaded, force=True) is True


def test_is_eligible_failed_distinguishes_transient_from_permanent():
    """Transient retries on the next run; permanent failures only with `--force`."""
    transient = MediaPhotoFailed(
        url="u",
        failure_reason="http_5xx",
        attempts=1,
        last_attempt_at=datetime.now(timezone.utc),
    )
    permanent = MediaPhotoFailed(
        url="u",
        failure_reason="http_4xx",
        attempts=1,
        last_attempt_at=datetime.now(timezone.utc),
    )
    assert _is_eligible(transient, force=False) is True
    assert _is_eligible(permanent, force=False) is False
    assert _is_eligible(permanent, force=True) is True


# --------------------------------------------------------------------- download_all


def test_download_all_records_local_path_and_dims_on_success(tmp_path: Path):
    """A pending photo downloads cleanly and lands as MediaPhotoDownloaded."""
    bytes_data = _png_bytes(width=10, height=7)
    item = _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/A.png")])
    session = FakeSession(responses={"pbs.twimg.com/media/A.png": [FakeResponse(200, bytes_data)]})
    report = download_all(
        {"123": item},
        media_root=tmp_path,
        session=session,
        throttle_seconds=0,
    )
    entry = item.media[0]
    assert isinstance(entry, MediaPhotoDownloaded)
    assert entry.width == 10
    assert entry.height == 7
    assert entry.bytes_size == len(bytes_data)
    assert entry.local_path == "123/0.png"
    assert (tmp_path / "123" / "0.png").exists()
    assert report.photos_downloaded == 1
    assert report.bytes_downloaded == len(bytes_data)


def test_download_all_idempotent_for_already_downloaded(tmp_path: Path):
    """A re-run on a downloaded photo bumps `skipped` not `downloaded`."""
    downloaded = MediaPhotoDownloaded(
        url="u",
        local_path="123/0.jpg",
        width=4,
        height=3,
        bytes_size=100,
        downloaded_at=datetime.now(timezone.utc),
    )
    item = _item_with_media([downloaded])
    session = FakeSession()
    report = download_all({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    assert report.photos_attempted == 0
    assert report.photos_skipped_already_downloaded == 1
    assert session.calls == []  # no HTTP call


def test_download_all_falls_back_to_large_when_orig_404s(tmp_path: Path):
    """The size cascade tries `orig` then `large` then `medium`."""
    bytes_data = _png_bytes()
    item = _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/B.png")])
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/B.png": [
                FakeResponse(404, b""),  # orig
                FakeResponse(200, bytes_data),  # large
            ]
        }
    )
    report = download_all({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    assert isinstance(item.media[0], MediaPhotoDownloaded)
    assert report.photos_downloaded == 1
    # Two HTTP calls (orig then large), each with the corresponding name= param.
    assert len(session.calls) == 2
    assert "name=orig" in session.calls[0][0]
    assert "name=large" in session.calls[1][0]


def test_download_all_records_http_4xx_failure_when_cascade_exhausted(tmp_path: Path):
    """Three 404s in a row → MediaPhotoFailed with `http_4xx` (permanent)."""
    item = _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/C.png")])
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/C.png": [
                FakeResponse(404, b""),
                FakeResponse(404, b""),
                FakeResponse(404, b""),
            ]
        }
    )
    with pytest.raises(RuntimeError):
        # A single attempted photo + zero downloads triggers the total-failure raise.
        download_all({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    entry = item.media[0]
    assert isinstance(entry, MediaPhotoFailed)
    assert entry.failure_reason == "http_4xx"
    assert entry.attempts == 1


def test_download_all_records_http_5xx_as_transient(tmp_path: Path):
    """A 5xx falls into the transient bucket — eligible for next-run retry."""
    item = _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/D.png")])
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/D.png": [
                FakeResponse(503, b""),
                FakeResponse(503, b""),
                FakeResponse(503, b""),
            ]
        }
    )
    with pytest.raises(RuntimeError):
        download_all({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    entry = item.media[0]
    assert isinstance(entry, MediaPhotoFailed)
    assert entry.failure_reason == "http_5xx"
    assert entry.failure_reason in _TRANSIENT_MEDIA_FAILURES


def test_download_all_records_timeout_as_transient(tmp_path: Path):
    """`requests.Timeout` from the session is bucketed under `timeout`."""
    import requests

    item = _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/E.png")])

    class _AlwaysTimeout:
        calls: list[tuple[str, int]] = []

        def get(self, url, *, timeout):
            self.calls.append((url, timeout))
            raise requests.Timeout("connect timeout")

    timeout_session = _AlwaysTimeout()
    with pytest.raises(RuntimeError):
        download_all(
            {"123": item}, media_root=tmp_path, session=timeout_session, throttle_seconds=0
        )
    entry = item.media[0]
    assert isinstance(entry, MediaPhotoFailed)
    assert entry.failure_reason == "timeout"


def test_download_all_records_format_error_when_pillow_rejects(tmp_path: Path):
    """A 200 with non-image bytes → `format_error` (permanent)."""
    item = _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/F.png")])
    session = FakeSession(
        responses={"pbs.twimg.com/media/F.png": [FakeResponse(200, b"not an image at all")]}
    )
    with pytest.raises(RuntimeError):
        download_all({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    entry = item.media[0]
    assert isinstance(entry, MediaPhotoFailed)
    assert entry.failure_reason == "format_error"


def test_download_all_buckets_unknown_exception_as_unknown_error(tmp_path: Path):
    """Connection errors land in the unknown_error bucket (transient).

    Mirrors `fetch.py`'s contract: a bare-except path stays retry-worthy.
    """
    import requests

    class _AlwaysConnError:
        calls: list[tuple[str, int]] = []

        def get(self, url, *, timeout):
            raise requests.ConnectionError("ECONNREFUSED")

    item = _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/G.png")])
    with pytest.raises(RuntimeError):
        download_all(
            {"123": item},
            media_root=tmp_path,
            session=_AlwaysConnError(),
            throttle_seconds=0,
        )
    entry = item.media[0]
    assert isinstance(entry, MediaPhotoFailed)
    assert entry.failure_reason == "unknown_error"
    assert entry.failure_reason in _TRANSIENT_MEDIA_FAILURES


def test_download_all_retries_transient_failures_on_next_run(tmp_path: Path):
    """A MediaPhotoFailed(http_5xx) is re-attempted automatically next run."""
    item = _item_with_media(
        [
            MediaPhotoFailed(
                url="https://pbs.twimg.com/media/H.png",
                failure_reason="http_5xx",
                attempts=1,
                last_attempt_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            )
        ]
    )
    session = FakeSession(
        responses={"pbs.twimg.com/media/H.png": [FakeResponse(200, _png_bytes())]}
    )
    report = download_all({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    assert isinstance(item.media[0], MediaPhotoDownloaded)
    assert report.photos_attempted == 1
    assert report.photos_downloaded == 1


def test_download_all_skips_permanent_failures_without_force(tmp_path: Path):
    """A MediaPhotoFailed(http_4xx) is not retried by default."""
    item = _item_with_media(
        [
            MediaPhotoFailed(
                url="u",
                failure_reason="http_4xx",
                attempts=1,
                last_attempt_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            )
        ]
    )
    session = FakeSession()
    report = download_all({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    assert report.photos_attempted == 0
    assert session.calls == []


def test_download_all_force_redownloads_already_downloaded(tmp_path: Path):
    """`--force` re-attempts a MediaPhotoDownloaded."""
    item = _item_with_media(
        [
            MediaPhotoDownloaded(
                url="https://pbs.twimg.com/media/I.png",
                local_path="123/0.jpg",
                width=1,
                height=1,
                bytes_size=10,
                downloaded_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            )
        ]
    )
    new_bytes = _png_bytes(width=20, height=15)
    session = FakeSession(responses={"pbs.twimg.com/media/I.png": [FakeResponse(200, new_bytes)]})
    report = download_all(
        {"123": item}, media_root=tmp_path, session=session, throttle_seconds=0, force=True
    )
    entry = item.media[0]
    assert isinstance(entry, MediaPhotoDownloaded)
    assert entry.width == 20  # new dimensions, not the stale "1" placeholder
    assert report.photos_downloaded == 1


def test_download_all_respects_limit(tmp_path: Path):
    """`limit=1` caps the number of photo download attempts."""
    item = _item_with_media(
        [
            MediaPhotoPending(url="https://pbs.twimg.com/media/J1.png"),
            MediaPhotoPending(url="https://pbs.twimg.com/media/J2.png"),
        ]
    )
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/J1.png": [FakeResponse(200, _png_bytes())],
            "pbs.twimg.com/media/J2.png": [FakeResponse(200, _png_bytes())],
        }
    )
    report = download_all(
        {"123": item}, media_root=tmp_path, session=session, throttle_seconds=0, limit=1
    )
    assert report.photos_attempted == 1
    assert report.photos_downloaded == 1
    # Second entry stays pending — limit hit before we got to it.
    assert isinstance(item.media[1], MediaPhotoPending)


def test_download_all_throttles_between_requests(tmp_path: Path):
    """`sleep` is called once per successful download."""
    sleep_calls: list[float] = []
    item = _item_with_media(
        [
            MediaPhotoPending(url="https://pbs.twimg.com/media/K1.png"),
            MediaPhotoPending(url="https://pbs.twimg.com/media/K2.png"),
        ]
    )
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/K1.png": [FakeResponse(200, _png_bytes())],
            "pbs.twimg.com/media/K2.png": [FakeResponse(200, _png_bytes())],
        }
    )
    download_all(
        {"123": item},
        media_root=tmp_path,
        session=session,
        throttle_seconds=0.25,
        sleep=sleep_calls.append,
    )
    assert sleep_calls == [0.25, 0.25]


def test_download_all_invokes_progress_callback_per_transition(tmp_path: Path):
    """The on_progress hook fires after each photo transition — the seam
    where the CLI persists `items.json` so Ctrl-C leaves a coherent store."""
    progress_calls: list[int] = []
    item = _item_with_media(
        [
            MediaPhotoPending(url="https://pbs.twimg.com/media/L1.png"),
            MediaPhotoPending(url="https://pbs.twimg.com/media/L2.png"),
        ]
    )
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/L1.png": [FakeResponse(200, _png_bytes())],
            "pbs.twimg.com/media/L2.png": [FakeResponse(200, _png_bytes())],
        }
    )
    download_all(
        {"123": item},
        media_root=tmp_path,
        session=session,
        throttle_seconds=0,
        on_progress=lambda: progress_calls.append(1),
    )
    assert len(progress_calls) == 2


def test_download_all_never_touches_video_pending(tmp_path: Path):
    """MediaVideoPending entries are not even iterated past."""
    video = MediaVideoPending(url="https://video.twimg.com/x.mp4")
    item = _item_with_media([video])
    session = FakeSession()
    report = download_all({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    assert report.photos_attempted == 0
    assert isinstance(item.media[0], MediaVideoPending)
    assert session.calls == []


def test_download_all_filters_by_items(tmp_path: Path):
    """`items_filter` restricts the run to the named item ids."""
    items_dict = {
        "1": _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/M1.png")]),
        "2": _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/M2.png")]),
    }
    # Patch the second item's id (it shares the constructor id="123" otherwise).
    items_dict["1"].id = "1"
    items_dict["2"].id = "2"
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/M1.png": [FakeResponse(200, _png_bytes())],
            "pbs.twimg.com/media/M2.png": [FakeResponse(200, _png_bytes())],
        }
    )
    report = download_all(
        items_dict,
        media_root=tmp_path,
        session=session,
        throttle_seconds=0,
        items_filter=["2"],
    )
    assert report.items_processed == 1
    assert report.photos_downloaded == 1
    # Item 1 was skipped entirely — its media stays pending.
    assert isinstance(items_dict["1"].media[0], MediaPhotoPending)


def test_download_all_propagates_keyboard_interrupt(tmp_path: Path):
    """Ctrl-C must NOT be swallowed — the narrow except clauses let it through."""

    class _CtrlC:
        def get(self, url, *, timeout):
            raise KeyboardInterrupt

    item = _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/N.png")])
    with pytest.raises(KeyboardInterrupt):
        download_all({"123": item}, media_root=tmp_path, session=_CtrlC(), throttle_seconds=0)


def test_download_all_raises_on_total_failure(tmp_path: Path):
    """All-failed batches surface as RuntimeError (mirror of #24)."""
    item = _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/O.png")])
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/O.png": [
                FakeResponse(404, b""),
                FakeResponse(404, b""),
                FakeResponse(404, b""),
            ]
        }
    )
    with pytest.raises(RuntimeError, match="All 1 photo download attempts failed"):
        download_all({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)


def test_download_all_partial_failure_does_not_raise(tmp_path: Path):
    """A run with some downloads + some failures is partial success, not total failure."""
    items_dict = {
        "1": _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/P1.png")]),
        "2": _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/P2.png")]),
    }
    items_dict["1"].id = "1"
    items_dict["2"].id = "2"
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/P1.png": [FakeResponse(200, _png_bytes())],
            "pbs.twimg.com/media/P2.png": [
                FakeResponse(404, b""),
                FakeResponse(404, b""),
                FakeResponse(404, b""),
            ],
        }
    )
    report = download_all(items_dict, media_root=tmp_path, session=session, throttle_seconds=0)
    assert report.photos_downloaded == 1
    assert report.photos_failed_permanent == 1


def test_download_all_writes_bytes_atomically(tmp_path: Path):
    """Local file is created via a tmp+rename so partial writes are impossible."""
    bytes_data = _png_bytes()
    item = _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/Q.png")])
    session = FakeSession(responses={"pbs.twimg.com/media/Q.png": [FakeResponse(200, bytes_data)]})
    download_all({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    # No leftover .part files; only the final file exists.
    assert sorted(p.name for p in (tmp_path / "123").iterdir()) == ["0.png"]


def test_emit_summary_line_silent_when_nothing_done(capsys):
    """A no-op run (zero attempts, zero skips) stays silent."""
    emit_summary_line(MediaReport())
    assert capsys.readouterr().err == ""


def test_emit_summary_line_includes_all_counters(capsys):
    """The SUMMARY line carries every relevant counter for ops visibility."""
    report = MediaReport(
        photos_attempted=10,
        photos_downloaded=8,
        photos_failed_permanent=1,
        photos_failed_transient=1,
        photos_skipped_already_downloaded=5,
        bytes_downloaded=1024000,
    )
    emit_summary_line(report)
    err = capsys.readouterr().err
    assert "SUMMARY: " in err
    assert "downloaded: 8" in err
    assert "failed_permanent: 1" in err
    assert "failed_transient: 1" in err
    assert "skipped: 5" in err
    assert "1_024_000" in err
