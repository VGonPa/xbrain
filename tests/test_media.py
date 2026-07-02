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
    _iter_eligible_article_images,
    _local_path,
    _url_with_name,
    download_all,
    emit_summary_line,
)
from xbrain.models import (
    ArticleImageBlock,
    ArticleTextBlock,
    Author,
    Content,
    ContentSourceSuccess,
    Item,
    MediaPhotoDownloaded,
    MediaPhotoFailed,
    MediaPhotoPending,
    MediaVideoDownloaded,
    MediaVideoFailed,
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


def _jpeg_bytes(width: int = 4, height: int = 3) -> bytes:
    """Return a valid JPEG with the requested dimensions for Pillow validation."""
    image = Image.new("RGB", (width, height), color=(11, 22, 33))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _webp_bytes(width: int = 4, height: int = 3) -> bytes:
    """Return a valid WebP with the requested dimensions for Pillow validation."""
    image = Image.new("RGB", (width, height), color=(44, 55, 66))
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP")
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
    """Photos only; videos stay in their pending variant always."""
    video = MediaVideoPending(url="u")
    assert _is_eligible(video, force=False) is False
    assert _is_eligible(video, force=True) is False


def test_is_eligible_video_downloaded_and_failed_never_attempted():
    """The PHOTO downloader ignores every video state — even with `--force`.

    Video download lives in `xbrain.video_media` (`xbrain download-videos`);
    `xbrain media` must never touch a `MediaVideoDownloaded` / `MediaVideoFailed`.
    """
    downloaded = MediaVideoDownloaded(
        url="https://video.twimg.com/x.mp4",
        local_path="1/0.mp4",
        bytes_size=10,
        downloaded_at=datetime.now(timezone.utc),
    )
    failed = MediaVideoFailed(
        url="https://video.twimg.com/x.mp4",
        failure_reason="http_5xx",
        attempts=1,
        last_attempt_at=datetime.now(timezone.utc),
    )
    assert _is_eligible(downloaded, force=False) is False
    assert _is_eligible(downloaded, force=True) is False
    assert _is_eligible(failed, force=False) is False
    assert _is_eligible(failed, force=True) is False


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


def test_download_all_records_http_4xx_when_cascade_exhausts(tmp_path: Path):
    """Three 404s in a row land the photo in the `http_4xx` (permanent) bucket.

    Setup is partial-success (a second item downloads cleanly) so the
    total-failure RuntimeError does NOT fire — the assertion is on the
    bucket alone, not on the raise.
    """
    items_dict = {
        "1": _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/C.png")]),
        "2": _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/C2.png")]),
    }
    items_dict["1"].id = "1"
    items_dict["2"].id = "2"
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/C.png": [
                FakeResponse(404, b""),
                FakeResponse(404, b""),
                FakeResponse(404, b""),
            ],
            "pbs.twimg.com/media/C2.png": [FakeResponse(200, _png_bytes())],
        }
    )
    report = download_all(items_dict, media_root=tmp_path, session=session, throttle_seconds=0)
    entry = items_dict["1"].media[0]
    assert isinstance(entry, MediaPhotoFailed)
    assert entry.failure_reason == "http_4xx"
    assert entry.attempts == 1
    assert report.photos_failed_permanent == 1
    assert report.photos_downloaded == 1


def test_download_all_raises_runtime_error_when_lone_4xx_attempt_fails(tmp_path: Path):
    """A single 4xx-only batch (no downloads at all) surfaces as RuntimeError."""
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
        download_all({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)


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
    """Total-failure batches surface as RuntimeError so callers see exit-1."""
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
    with pytest.raises(RuntimeError, match="All 1 media download attempts failed"):
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


def test_download_all_falls_back_through_full_cascade_to_medium(tmp_path: Path):
    """orig 404 + large 404 + medium 200 — the cascade exhausts both higher
    sizes before landing on `medium`. Three calls in cascade order."""
    bytes_data = _png_bytes(width=8, height=6)
    item = _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/CASC.png")])
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/CASC.png": [
                FakeResponse(404, b""),  # orig
                FakeResponse(404, b""),  # large
                FakeResponse(200, bytes_data),  # medium
            ]
        }
    )
    report = download_all({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    assert isinstance(item.media[0], MediaPhotoDownloaded)
    assert report.photos_downloaded == 1
    assert len(session.calls) == 3
    assert "name=orig" in session.calls[0][0]
    assert "name=large" in session.calls[1][0]
    assert "name=medium" in session.calls[2][0]


def test_download_all_records_webp_extension(tmp_path: Path):
    """A `.webp` URL with WebP bytes is stored with `.webp` on disk."""
    bytes_data = _webp_bytes(width=6, height=5)
    item = _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/W.webp")])
    session = FakeSession(responses={"pbs.twimg.com/media/W.webp": [FakeResponse(200, bytes_data)]})
    download_all({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    entry = item.media[0]
    assert isinstance(entry, MediaPhotoDownloaded)
    assert entry.local_path.endswith(".webp")
    assert (tmp_path / entry.local_path).exists()


def test_download_all_records_jpeg_extension(tmp_path: Path):
    """A JPEG URL with JPEG bytes is stored with `.jpg` on disk."""
    bytes_data = _jpeg_bytes(width=6, height=5)
    item = _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/J.jpg")])
    session = FakeSession(responses={"pbs.twimg.com/media/J.jpg": [FakeResponse(200, bytes_data)]})
    download_all({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    entry = item.media[0]
    assert isinstance(entry, MediaPhotoDownloaded)
    assert entry.local_path.endswith(".jpg")
    assert (tmp_path / entry.local_path).exists()


def test_download_all_buckets_3xx_status_as_unknown_error(tmp_path: Path):
    """A 304 (or any non-2xx non-4xx non-5xx) lands in `unknown_error` (transient).

    Three 304s in a row (full cascade exhausted) trigger the total-failure
    path; the bucket assertion is what we care about.
    """
    item = _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/R.png")])
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/R.png": [
                FakeResponse(304, b""),
                FakeResponse(304, b""),
                FakeResponse(304, b""),
            ]
        }
    )
    with pytest.raises(RuntimeError):
        download_all({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    entry = item.media[0]
    assert isinstance(entry, MediaPhotoFailed)
    assert entry.failure_reason == "unknown_error"


def test_download_all_atomic_write_rollback_cleans_part_file(tmp_path: Path, monkeypatch):
    """When `path.replace` raises, the `.part` file is removed and no final
    file exists. Simulates a disk-full condition between write and rename.

    Setup is partial-success (a second item downloads cleanly under a
    different media_root path that does not collide with the monkeypatch)
    so the total-failure RuntimeError does not fire and we can assert on
    the bucket + filesystem state of the failed item.
    """
    bytes_data = _png_bytes(width=10, height=8)
    fail_url = "https://pbs.twimg.com/media/ZFAIL.png"
    ok_url = "https://pbs.twimg.com/media/ZOK.png"
    items_dict = {
        "1": _item_with_media([MediaPhotoPending(url=fail_url)]),
        "2": _item_with_media([MediaPhotoPending(url=ok_url)]),
    }
    items_dict["1"].id = "1"
    items_dict["2"].id = "2"
    session = FakeSession(
        responses={
            fail_url.replace("https://", "").split("?")[0]: [FakeResponse(200, bytes_data)],
            ok_url.replace("https://", "").split("?")[0]: [FakeResponse(200, bytes_data)],
        }
    )

    original_replace = Path.replace

    def _conditional_fail(self, target):
        # Only fail the replace for the path containing item id "1".
        if "/1/" in str(target) or str(target).endswith("/1/0.png"):
            raise OSError("simulated disk full")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", _conditional_fail)
    report = download_all(items_dict, media_root=tmp_path, session=session, throttle_seconds=0)

    # Item 1 bucketed as unknown_error (transient): disk full is retryable.
    failed_entry = items_dict["1"].media[0]
    assert isinstance(failed_entry, MediaPhotoFailed)
    assert failed_entry.failure_reason == "unknown_error"
    # Item 2 downloaded cleanly.
    assert isinstance(items_dict["2"].media[0], MediaPhotoDownloaded)
    # No `.part` orphan or half-written final file under item 1.
    item_dir = tmp_path / "1"
    if item_dir.exists():
        names = sorted(p.name for p in item_dir.iterdir())
        assert "0.png.part" not in names
        assert "0.png" not in names
    assert report.photos_downloaded == 1
    assert report.photos_failed_transient == 1


def test_download_all_sweeps_part_orphans_on_entry(tmp_path: Path):
    """A stale `*.part` file left by a SIGKILL is removed before any download."""
    orphan_dir = tmp_path / "orphan-item"
    orphan_dir.mkdir(parents=True)
    orphan = orphan_dir / "0.png.part"
    orphan.write_bytes(b"stale junk from a previous run")
    item = _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/S.png")])
    session = FakeSession(
        responses={"pbs.twimg.com/media/S.png": [FakeResponse(200, _png_bytes())]}
    )
    download_all({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    assert not orphan.exists()


def test_download_all_partial_failure_summary_emits_line(capsys):
    """A run with zero downloads + N failures still emits the SUMMARY line.

    The silence rule only applies to "zero attempts AND zero skips" —
    a failed-only run is not silent: ops needs to see the failure count.
    """
    report = MediaReport(
        photos_attempted=3,
        photos_downloaded=0,
        photos_failed_permanent=2,
        photos_failed_transient=1,
    )
    emit_summary_line(report)
    err = capsys.readouterr().err
    assert "SUMMARY: " in err
    assert "downloaded: 0" in err
    assert "failed_permanent: 2" in err
    assert "failed_transient: 1" in err


def test_download_all_truncated_image_buckets_as_format_error(tmp_path: Path):
    """Pillow rejecting partial PNG bytes → `format_error` (permanent)."""
    truncated = _png_bytes()[:32]
    item = _item_with_media([MediaPhotoPending(url="https://pbs.twimg.com/media/T.png")])
    session = FakeSession(responses={"pbs.twimg.com/media/T.png": [FakeResponse(200, truncated)]})
    with pytest.raises(RuntimeError):
        download_all({"123": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    entry = item.media[0]
    assert isinstance(entry, MediaPhotoFailed)
    assert entry.failure_reason == "format_error"


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


# ------------------------------------------------------------ article images (#39 PR4)


def _article_item(
    item_id: str,
    blocks: list,
    *,
    text: str,
    media_entries: list | None = None,
) -> Item:
    """Build an Item whose content carries one `x_article` source with `blocks`.

    `text` must equal the concatenation of the `ArticleTextBlock` texts (the
    #39 PR1 invariant, enforced by `ContentSourceSuccess._text_matches_blocks`).
    `media_entries` optionally populates `item.media` (the item's own photos) so
    a test can prove the article/ namespace never collides with `<id>/<n>`.
    """
    source = ContentSourceSuccess(
        kind="x_article",
        url=f"https://x.com/i/article/{item_id}",
        title="An Article",
        text=text,
        blocks=blocks,
        http_status=200,
        attempts=1,
    )
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        media=media_entries or [],
        content=Content(
            fetched_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            sources=[source],
        ),
    )


def test_local_path_with_subdir_namespaces_article_images():
    """`subdir="article"` yields `<id>/article/<index><ext>`; default is unchanged."""
    assert _local_path("123", 0, ".jpg") == "123/0.jpg"
    assert _local_path("123", 0, ".jpg", subdir="article") == "123/article/0.jpg"
    assert _local_path("abc", 2, ".png", subdir="article") == "abc/article/2.png"


def test_iter_eligible_article_images_index_is_stable_across_ineligible_blocks():
    """The per-item image index counts EVERY image block, not just eligible ones.

    A downloaded block 0 (skipped without --force) must not shift the pending
    block 1 to index 0 — else a re-fetch would overwrite `<id>/article/0`.
    """
    downloaded = MediaPhotoDownloaded(
        url="https://pbs.twimg.com/media/AD.png",
        local_path="1/article/0.png",
        width=4,
        height=3,
        bytes_size=10,
        downloaded_at=datetime.now(timezone.utc),
    )
    pending = MediaPhotoPending(url="https://pbs.twimg.com/media/AP.png")
    item = _article_item(
        "1",
        [
            ArticleTextBlock(text="intro"),
            ArticleImageBlock(media=downloaded),
            ArticleImageBlock(media=pending),
        ],
        text="intro",
    )
    report = MediaReport()
    yielded = list(_iter_eligible_article_images({"1": item}, force=False, report=report))
    assert len(yielded) == 1
    item_id, block, index, entry = yielded[0]
    assert item_id == "1"
    assert index == 1  # the pending image is the 2nd image block → index 1
    assert entry is pending
    assert report.photos_skipped_already_downloaded == 1  # downloaded block skipped


def test_download_all_downloads_article_images(tmp_path: Path):
    """Two pending article-image blocks download to `<id>/article/{0,1}` in place."""
    png0 = _png_bytes(width=10, height=7)
    png1 = _png_bytes(width=6, height=5)
    item = _article_item(
        "500",
        [
            ArticleTextBlock(text="A"),
            ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/AA.png")),
            ArticleTextBlock(text="B"),
            ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/AB.png")),
        ],
        text="AB",
    )
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/AA.png": [FakeResponse(200, png0)],
            "pbs.twimg.com/media/AB.png": [FakeResponse(200, png1)],
        }
    )
    report = download_all({"500": item}, media_root=tmp_path, session=session, throttle_seconds=0)

    source = item.content.sources[0]
    img0 = source.blocks[1]
    img1 = source.blocks[3]
    assert isinstance(img0.media, MediaPhotoDownloaded)
    assert isinstance(img1.media, MediaPhotoDownloaded)
    assert img0.media.local_path == "500/article/0.png"
    assert img1.media.local_path == "500/article/1.png"
    assert (tmp_path / "500" / "article" / "0.png").exists()
    assert (tmp_path / "500" / "article" / "1.png").exists()
    assert report.article_images_attempted == 2
    assert report.article_images_downloaded == 2
    assert report.photos_attempted == 0  # no item.media photos here
    assert report.bytes_downloaded == len(png0) + len(png1)


def test_download_all_article_images_idempotent(tmp_path: Path):
    """A re-run over an already-downloaded article image skips it, no HTTP."""
    downloaded = MediaPhotoDownloaded(
        url="https://pbs.twimg.com/media/AC.png",
        local_path="501/article/0.png",
        width=4,
        height=3,
        bytes_size=100,
        downloaded_at=datetime.now(timezone.utc),
    )
    item = _article_item(
        "501",
        [ArticleTextBlock(text="x"), ArticleImageBlock(media=downloaded)],
        text="x",
    )
    session = FakeSession()
    report = download_all({"501": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    assert report.article_images_attempted == 0
    assert report.article_images_downloaded == 0
    assert report.photos_skipped_already_downloaded == 1
    assert session.calls == []  # no HTTP call
    assert isinstance(item.content.sources[0].blocks[1].media, MediaPhotoDownloaded)


def test_download_all_article_image_failure_not_fatal_rest_proceed(tmp_path: Path):
    """A failed article image is recorded (not fatal); the rest still download."""
    good = _png_bytes()
    item = _article_item(
        "502",
        [
            ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/AF.png")),
            ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/AG.png")),
        ],
        text="",
    )
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/AF.png": [
                FakeResponse(404, b""),
                FakeResponse(404, b""),
                FakeResponse(404, b""),
            ],
            "pbs.twimg.com/media/AG.png": [FakeResponse(200, good)],
        }
    )
    report = download_all({"502": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    blocks = item.content.sources[0].blocks
    assert isinstance(blocks[0].media, MediaPhotoFailed)
    assert blocks[0].media.failure_reason == "http_4xx"
    assert isinstance(blocks[1].media, MediaPhotoDownloaded)
    assert report.article_images_failed_permanent == 1
    assert report.article_images_downloaded == 1
    assert report.per_item_failures["502"] == [("https://pbs.twimg.com/media/AF.png", "http_4xx")]


def test_download_all_article_only_success_does_not_raise(tmp_path: Path):
    """A run that downloads 0 photos but N article images must NOT raise."""
    item = _article_item(
        "503",
        [ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/AH.png"))],
        text="",
    )
    session = FakeSession(
        responses={"pbs.twimg.com/media/AH.png": [FakeResponse(200, _png_bytes())]}
    )
    report = download_all({"503": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    assert report.photos_attempted == 0
    assert report.article_images_downloaded == 1  # no RuntimeError raised


def test_download_all_photo_failure_saved_by_article_success(tmp_path: Path):
    """A failed photo + a successful article image is partial success, not total."""
    item = _article_item(
        "504",
        [ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/AI.png"))],
        text="",
        media_entries=[MediaPhotoPending(url="https://pbs.twimg.com/media/AJ.png")],
    )
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/AI.png": [FakeResponse(200, _png_bytes())],
            "pbs.twimg.com/media/AJ.png": [
                FakeResponse(404, b""),
                FakeResponse(404, b""),
                FakeResponse(404, b""),
            ],
        }
    )
    report = download_all({"504": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    assert report.photos_failed_permanent == 1
    assert report.article_images_downloaded == 1  # combined guard: no raise


def test_download_all_combined_total_failure_raises(tmp_path: Path):
    """When BOTH the photo and the article image fail, the run raises."""
    item = _article_item(
        "505",
        [ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/AK.png"))],
        text="",
        media_entries=[MediaPhotoPending(url="https://pbs.twimg.com/media/AL.png")],
    )
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/AK.png": [FakeResponse(404, b"")] * 3,
            "pbs.twimg.com/media/AL.png": [FakeResponse(404, b"")] * 3,
        }
    )
    with pytest.raises(RuntimeError, match="All 2 media download attempts failed"):
        download_all({"505": item}, media_root=tmp_path, session=session, throttle_seconds=0)


def test_download_all_article_and_photo_no_path_collision(tmp_path: Path):
    """`item.media` photo index 0 and article image index 0 write to distinct files."""
    photo_bytes = _png_bytes(width=8, height=8)
    article_bytes = _jpeg_bytes(width=5, height=5)
    item = _article_item(
        "506",
        [ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/AN.jpg"))],
        text="",
        media_entries=[MediaPhotoPending(url="https://pbs.twimg.com/media/AM.png")],
    )
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/AM.png": [FakeResponse(200, photo_bytes)],
            "pbs.twimg.com/media/AN.jpg": [FakeResponse(200, article_bytes)],
        }
    )
    download_all({"506": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    photo_entry = item.media[0]
    article_entry = item.content.sources[0].blocks[0].media
    assert isinstance(photo_entry, MediaPhotoDownloaded)
    assert isinstance(article_entry, MediaPhotoDownloaded)
    assert photo_entry.local_path == "506/0.png"
    assert article_entry.local_path == "506/article/0.jpg"
    assert (tmp_path / "506" / "0.png").exists()
    assert (tmp_path / "506" / "article" / "0.jpg").exists()


def test_download_all_article_swap_preserves_text_validator(tmp_path: Path):
    """After the in-place media swap, the item still round-trips (text==concat holds)."""
    item = _article_item(
        "507",
        [
            ArticleTextBlock(text="Para one. "),
            ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/AO.png")),
            ArticleTextBlock(text="Para two."),
        ],
        text="Para one. Para two.",
    )
    session = FakeSession(
        responses={"pbs.twimg.com/media/AO.png": [FakeResponse(200, _png_bytes())]}
    )
    download_all({"507": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    # Round-trip through validation: the model_validator must not reject the
    # swapped body (images do not contribute to `text`).
    reloaded = Item.model_validate(item.model_dump(mode="python"))
    source = reloaded.content.sources[0]
    assert source.text == "Para one. Para two."
    assert isinstance(source.blocks[1].media, MediaPhotoDownloaded)


def test_download_all_article_image_force_redownloads(tmp_path: Path):
    """A downloaded article image is skipped without --force, re-fetched with it."""
    downloaded = MediaPhotoDownloaded(
        url="https://pbs.twimg.com/media/AP.png",
        local_path="508/article/0.png",
        width=1,
        height=1,
        bytes_size=10,
        downloaded_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
    )
    item = _article_item("508", [ArticleImageBlock(media=downloaded)], text="")
    new_bytes = _png_bytes(width=20, height=15)
    session = FakeSession(responses={"pbs.twimg.com/media/AP.png": [FakeResponse(200, new_bytes)]})
    report = download_all(
        {"508": item}, media_root=tmp_path, session=session, throttle_seconds=0, force=True
    )
    entry = item.content.sources[0].blocks[0].media
    assert isinstance(entry, MediaPhotoDownloaded)
    assert entry.width == 20  # fresh dimensions, not the stale placeholder
    assert report.article_images_downloaded == 1


def test_download_all_article_image_transient_retry(tmp_path: Path):
    """A MediaPhotoFailed(http_5xx) article image is auto-retried next run."""
    failed = MediaPhotoFailed(
        url="https://pbs.twimg.com/media/AQ.png",
        failure_reason="http_5xx",
        attempts=1,
        last_attempt_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
    )
    item = _article_item("509", [ArticleImageBlock(media=failed)], text="")
    session = FakeSession(
        responses={"pbs.twimg.com/media/AQ.png": [FakeResponse(200, _png_bytes())]}
    )
    report = download_all({"509": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    assert isinstance(item.content.sources[0].blocks[0].media, MediaPhotoDownloaded)
    assert report.article_images_attempted == 1
    assert report.article_images_downloaded == 1


def test_download_all_article_image_transient_failure_bucketed(tmp_path: Path):
    """A 5xx article-image failure lands in the transient bucket (retried next run)."""
    item = _article_item(
        "513",
        [
            ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/BA.png")),
            ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/BB.png")),
        ],
        text="",
    )
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/BA.png": [FakeResponse(503, b"")] * 3,
            "pbs.twimg.com/media/BB.png": [FakeResponse(200, _png_bytes())],
        }
    )
    report = download_all({"513": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    failed = item.content.sources[0].blocks[0].media
    assert isinstance(failed, MediaPhotoFailed)
    assert failed.failure_reason == "http_5xx"
    assert report.article_images_failed_transient == 1
    assert report.article_images_downloaded == 1  # the sibling still landed


def test_iter_eligible_article_images_skips_non_article_sources(tmp_path: Path):
    """The walk skips a non-`x_article` source and only advances article images."""
    external = ContentSourceSuccess(
        kind="external_article",
        url="https://example.com/p",
        text="external body",
        http_status=200,
        attempts=1,
    )
    article = ContentSourceSuccess(
        kind="x_article",
        url="https://x.com/i/article/514",
        title="An Article",
        text="",
        blocks=[
            ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/BC.png"))
        ],
        http_status=200,
        attempts=1,
    )
    item = Item(
        id="514",
        source="bookmark",
        url="https://x.com/a/status/514",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        content=Content(
            fetched_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            sources=[external, article],
        ),
    )
    report = MediaReport()
    yielded = list(_iter_eligible_article_images({"514": item}, force=False, report=report))
    assert len(yielded) == 1  # only the x_article image; the external source is skipped
    assert yielded[0][2] == 0  # image index restarts at 0 within the article source
    session = FakeSession(
        responses={"pbs.twimg.com/media/BC.png": [FakeResponse(200, _png_bytes())]}
    )
    download_all({"514": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    assert isinstance(item.content.sources[1].blocks[0].media, MediaPhotoDownloaded)


def test_download_all_article_image_progress_callback_fires(tmp_path: Path):
    """`on_progress` fires after each article-image transition (Ctrl-C seam)."""
    progress_calls: list[int] = []
    item = _article_item(
        "510",
        [
            ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/AR.png")),
            ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/AS.png")),
        ],
        text="",
    )
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/AR.png": [FakeResponse(200, _png_bytes())],
            "pbs.twimg.com/media/AS.png": [FakeResponse(200, _png_bytes())],
        }
    )
    download_all(
        {"510": item},
        media_root=tmp_path,
        session=session,
        throttle_seconds=0,
        on_progress=lambda: progress_calls.append(1),
    )
    assert len(progress_calls) == 2


def test_download_all_article_images_respect_combined_limit(tmp_path: Path):
    """`--limit` is a COMBINED budget: photos consume it before article images."""
    item = _article_item(
        "511",
        [
            ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/AT.png")),
            ArticleImageBlock(media=MediaPhotoPending(url="https://pbs.twimg.com/media/AU.png")),
        ],
        text="",
        media_entries=[MediaPhotoPending(url="https://pbs.twimg.com/media/AV.png")],
    )
    session = FakeSession(
        responses={
            "pbs.twimg.com/media/AV.png": [FakeResponse(200, _png_bytes())],
            "pbs.twimg.com/media/AT.png": [FakeResponse(200, _png_bytes())],
            "pbs.twimg.com/media/AU.png": [FakeResponse(200, _png_bytes())],
        }
    )
    report = download_all(
        {"511": item}, media_root=tmp_path, session=session, throttle_seconds=0, limit=2
    )
    # limit=2: the photo consumes 1, the first article image consumes the 2nd —
    # the second article image is left pending.
    assert report.photos_attempted == 1
    assert report.article_images_attempted == 1
    blocks = item.content.sources[0].blocks
    assert isinstance(blocks[0].media, MediaPhotoDownloaded)
    assert isinstance(blocks[1].media, MediaPhotoPending)


def test_download_all_never_touches_video_in_article_block(tmp_path: Path):
    """A (degenerate) video-state media inside an article block is never attempted."""
    video = MediaVideoPending(url="https://video.twimg.com/x.mp4")
    item = _article_item("512", [ArticleImageBlock(media=video)], text="")
    session = FakeSession()
    report = download_all({"512": item}, media_root=tmp_path, session=session, throttle_seconds=0)
    assert report.article_images_attempted == 0
    assert isinstance(item.content.sources[0].blocks[0].media, MediaVideoPending)
    assert session.calls == []


def test_emit_summary_line_includes_article_counters(capsys):
    """The SUMMARY line surfaces article-image counters — never folded into photos."""
    report = MediaReport(
        photos_attempted=2,
        photos_downloaded=2,
        article_images_attempted=3,
        article_images_downloaded=2,
        article_images_failed_permanent=1,
    )
    emit_summary_line(report)
    err = capsys.readouterr().err
    assert "SUMMARY: " in err
    assert "article_downloaded: 2" in err
    assert "article_failed_permanent: 1" in err
    assert "article_failed_transient: 0" in err


def test_emit_summary_line_non_silent_when_only_article_attempts(capsys):
    """A run with only article-image attempts still emits the SUMMARY line."""
    report = MediaReport(article_images_attempted=1, article_images_downloaded=1)
    emit_summary_line(report)
    assert "SUMMARY: " in capsys.readouterr().err
