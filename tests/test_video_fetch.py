"""Tests for `xbrain.video_fetch` — the ephemeral mp4 fetch for `fetch-video`.

`fetch_videos` downloads a selected item's real mp4 to `<dest>/<id>.mp4`,
REUSING `video_media`/`media` primitives (content-validation, retry
classification, atomic write, `.part` sweep, the mp4/HLS/poster discriminator).
It is deliberately store-non-mutating: it reads the resolved stream URL off the
item's video entry and never writes `items.json` nor `data/media/`. HTTP is
mocked with a hand-rolled `FakeSession` (no real network).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

from xbrain.models import (
    Author,
    Item,
    MediaVideoDownloaded,
    MediaVideoFailed,
    MediaVideoPending,
)
from xbrain.video_fetch import FetchReport, FetchResult, fetch_videos

_MP4_URL = "https://video.twimg.com/ext_tw_video/1/vid/720/A.mp4?tag=12"
_MP4_URL_B = "https://video.twimg.com/ext_tw_video/2/vid/720/B.mp4?tag=9"
_HLS_URL = "https://video.twimg.com/ext_tw_video/1/pl/B.m3u8?c=fmp4"
_POSTER = "https://pbs.twimg.com/ext_tw_video_thumb/1/img/P.jpg"


def _mp4_bytes(size: int = 2048) -> bytes:
    """Opaque non-empty mp4-ish payload (videos are not decoded, only written)."""
    return b"\x00\x00\x00\x18ftypmp42" + (b"\x00" * size)


@dataclass
class FakeResponse:
    status_code: int
    content: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class FakeSession:
    """Canned-response fake keyed by URL substring (mirrors test_video_media)."""

    responses: dict[str, list[FakeResponse]] = field(default_factory=dict)
    raise_for: dict[str, Exception] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)

    def get(self, url: str, *, timeout: int) -> FakeResponse:
        self.calls.append(url)
        for key, exc in list(self.raise_for.items()):
            if key in url:
                del self.raise_for[key]
                raise exc
        for matcher, queue in self.responses.items():
            if matcher in url and queue:
                return queue.pop(0)
        return FakeResponse(status_code=404, content=b"")


def _ok_session() -> FakeSession:
    return FakeSession(responses={".mp4": [FakeResponse(200, _mp4_bytes()) for _ in range(8)]})


def _item(item_id: str, *, media: list | None = None, source: str = "bookmark") -> Item:
    item = Item(
        id=item_id,
        source=source,  # type: ignore[arg-type]
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=f"note {item_id}",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )
    if media is not None:
        item.media = media
    return item


def _pending(url: str = _MP4_URL):
    return MediaVideoPending(
        url=url, thumbnail_url=_POSTER, bitrate=2_176_000, duration_millis=30_000
    )


def _no_throttle(_seconds: float) -> None:
    return None


# ------------------------------------------------------------ happy path


def test_fetches_mp4_to_dest_dir(tmp_path: Path):
    store = {"42": _item("42", media=[_pending()])}
    session = _ok_session()
    report = fetch_videos(
        store, ["42"], tmp_path, session=session, sleep=_no_throttle, throttle_seconds=0
    )
    out = tmp_path / "42.mp4"
    assert out.exists()
    assert out.read_bytes() == _mp4_bytes()
    assert isinstance(report, FetchReport)
    assert report.fetched == 1
    result = report.results[0]
    assert isinstance(result, FetchResult)
    assert result.outcome == "fetched"
    assert result.path == str(out)
    assert result.size_bytes == len(_mp4_bytes())


def test_fetch_does_not_mutate_store_bytes(tmp_path: Path):
    """The store must be byte-identical before/after a fetch (no mutation)."""
    from xbrain.store import load_store, save_store

    items_path = tmp_path / "items.json"
    save_store({"42": _item("42", media=[_pending()])}, items_path)
    before = items_path.read_bytes()

    store = load_store(items_path)
    dest = tmp_path / "out"
    fetch_videos(store, ["42"], dest, session=_ok_session(), sleep=_no_throttle, throttle_seconds=0)

    # fetch_videos never persists; the on-disk store is untouched, and the
    # in-memory item's pending video entry is unchanged (no transition).
    assert items_path.read_bytes() == before
    assert isinstance(store["42"].media[0], MediaVideoPending)
    assert not (tmp_path / "data").exists()


def test_dedup_of_repeated_ids(tmp_path: Path):
    store = {"42": _item("42", media=[_pending()])}
    session = _ok_session()
    report = fetch_videos(
        store, ["42", "42"], tmp_path, session=session, sleep=_no_throttle, throttle_seconds=0
    )
    assert report.fetched == 1
    assert len(session.calls) == 1


# ------------------------------------------------------------ skips


def test_hls_is_skipped_not_downloaded(tmp_path: Path):
    store = {"7": _item("7", media=[_pending(url=_HLS_URL)])}
    session = _ok_session()
    report = fetch_videos(
        store, ["7"], tmp_path, session=session, sleep=_no_throttle, throttle_seconds=0
    )
    assert report.fetched == 0
    assert report.skipped == 1
    assert report.results[0].reason == "hls"
    assert session.calls == []  # never hit the network
    assert not (tmp_path / "7.mp4").exists()


def test_poster_era_is_skipped(tmp_path: Path):
    store = {"9": _item("9", media=[MediaVideoPending(url=_POSTER, thumbnail_url=_POSTER)])}
    report = fetch_videos(
        store, ["9"], tmp_path, session=_ok_session(), sleep=_no_throttle, throttle_seconds=0
    )
    assert report.skipped == 1
    assert report.results[0].reason == "poster_era"


def test_unknown_item_is_skipped(tmp_path: Path):
    report = fetch_videos(
        {}, ["nope"], tmp_path, session=_ok_session(), sleep=_no_throttle, throttle_seconds=0
    )
    assert report.skipped == 1
    assert report.results[0].reason == "unknown_item"


def test_item_without_video_is_skipped(tmp_path: Path):
    store = {"1": _item("1", media=[])}
    report = fetch_videos(
        store, ["1"], tmp_path, session=_ok_session(), sleep=_no_throttle, throttle_seconds=0
    )
    assert report.results[0].reason == "no_video"


def test_max_size_skips_too_large(tmp_path: Path):
    store = {"42": _item("42", media=[_pending()])}  # est ~8 MB
    report = fetch_videos(
        store,
        ["42"],
        tmp_path,
        max_size_bytes=1_000_000,
        session=_ok_session(),
        sleep=_no_throttle,
        throttle_seconds=0,
    )
    assert report.skipped == 1
    assert report.results[0].reason == "too_large"
    assert not (tmp_path / "42.mp4").exists()


def test_max_size_skips_unknown_size(tmp_path: Path):
    entry = MediaVideoPending(url=_MP4_URL, thumbnail_url=_POSTER)  # no bitrate/duration
    store = {"42": _item("42", media=[entry])}
    report = fetch_videos(
        store,
        ["42"],
        tmp_path,
        max_size_bytes=1_000_000,
        session=_ok_session(),
        sleep=_no_throttle,
        throttle_seconds=0,
    )
    assert report.results[0].reason == "size_unknown"


def test_limit_caps_fetch_attempts(tmp_path: Path):
    store = {
        "a": _item("a", media=[_pending()]),
        "b": _item("b", media=[_pending(url=_MP4_URL_B)]),
    }
    report = fetch_videos(
        store,
        ["a", "b"],
        tmp_path,
        limit=1,
        session=_ok_session(),
        sleep=_no_throttle,
        throttle_seconds=0,
    )
    assert report.fetched == 1
    assert (tmp_path / "a.mp4").exists()
    assert not (tmp_path / "b.mp4").exists()


# ------------------------------------------------------------ failures


def test_failed_download_is_reported_not_fatal(tmp_path: Path):
    store = {
        "a": _item("a", media=[_pending()]),
        "b": _item("b", media=[_pending(url=_MP4_URL_B)]),
    }
    # 'a' returns a 500 (transient); 'b' succeeds — the batch continues.
    session = FakeSession(
        responses={
            "/1/vid": [FakeResponse(500, b"err")],
            "/2/vid": [FakeResponse(200, _mp4_bytes())],
        }
    )
    report = fetch_videos(
        store, ["a", "b"], tmp_path, session=session, sleep=_no_throttle, throttle_seconds=0
    )
    assert report.failed == 1
    assert report.fetched == 1
    failed = next(r for r in report.results if r.outcome == "failed")
    assert failed.id == "a"
    assert failed.reason == "http_5xx"
    assert (tmp_path / "b.mp4").exists()
    assert not (tmp_path / "a.mp4").exists()


def test_non_video_body_is_rejected(tmp_path: Path):
    """A 200 with an HTML interstitial body is rejected (content-validation reuse)."""
    store = {"42": _item("42", media=[_pending()])}
    session = FakeSession(
        responses={".mp4": [FakeResponse(200, b"<!DOCTYPE html><html>nope</html>")]}
    )
    report = fetch_videos(
        store, ["42"], tmp_path, session=session, sleep=_no_throttle, throttle_seconds=0
    )
    assert report.failed == 1
    assert not (tmp_path / "42.mp4").exists()


def test_timeout_is_classified(tmp_path: Path):
    store = {"42": _item("42", media=[_pending()])}
    session = FakeSession(raise_for={".mp4": requests.Timeout("slow")})
    report = fetch_videos(
        store, ["42"], tmp_path, session=session, sleep=_no_throttle, throttle_seconds=0
    )
    assert report.results[0].reason == "timeout"
    assert report.failed == 1


def test_generic_request_exception_is_classified_as_unknown_error(tmp_path: Path):
    """A non-timeout `RequestException` (connection reset, DNS failure, TLS error…)
    is caught as a per-video `unknown_error` failure — the batch continues, never a
    raw traceback that aborts the whole run."""
    store = {"42": _item("42", media=[_pending()])}
    session = FakeSession(raise_for={".mp4": requests.ConnectionError("reset by peer")})
    report = fetch_videos(
        store, ["42"], tmp_path, session=session, sleep=_no_throttle, throttle_seconds=0
    )
    assert report.results[0].reason == "unknown_error"
    assert report.failed == 1


def test_downloaded_entry_is_refetched_ephemerally(tmp_path: Path):
    """A store entry already `MediaVideoDownloaded` still fetches to --to (fetch is
    independent of store state; it re-resolves the stream URL and re-downloads)."""
    entry = MediaVideoDownloaded(
        url=_MP4_URL,
        thumbnail_url=_POSTER,
        bitrate=2_176_000,
        duration_millis=30_000,
        local_path="42/0.mp4",
        bytes_size=10,
        downloaded_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )
    store = {"42": _item("42", media=[entry])}
    report = fetch_videos(
        store, ["42"], tmp_path, session=_ok_session(), sleep=_no_throttle, throttle_seconds=0
    )
    assert report.fetched == 1
    assert (tmp_path / "42.mp4").exists()


def test_foreign_part_files_are_not_swept(tmp_path: Path):
    """fetch-video must NOT sweep the operator's --to dir: a foreign in-progress
    `.part` from another program survives a fetch run (and a skip-only run)."""
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "other.part").write_bytes(b"someone-elses-download")
    nested = dest / "sub"
    nested.mkdir()
    (nested / "deep.part").write_bytes(b"nested-download")

    store = {"42": _item("42", media=[_pending()])}
    fetch_videos(store, ["42"], dest, session=_ok_session(), sleep=_no_throttle, throttle_seconds=0)
    assert (dest / "other.part").read_bytes() == b"someone-elses-download"
    assert (nested / "deep.part").exists()

    # A skip-only run (HLS) must also leave the foreign .part untouched.
    fetch_videos(
        {"9": _item("9", media=[_pending(url=_HLS_URL)])},
        ["9"],
        dest,
        session=_ok_session(),
        sleep=_no_throttle,
        throttle_seconds=0,
    )
    assert (dest / "other.part").exists()
    assert (nested / "deep.part").exists()


def test_limit_not_consumed_by_leading_skip(tmp_path: Path):
    """A leading skip (HLS) must NOT eat the --limit budget; the fetchable item
    after it still downloads under limit=1."""
    store = {
        "skip": _item("skip", media=[_pending(url=_HLS_URL)]),
        "good": _item("good", media=[_pending()]),
    }
    report = fetch_videos(
        store,
        ["skip", "good"],
        tmp_path,
        limit=1,
        session=_ok_session(),
        sleep=_no_throttle,
        throttle_seconds=0,
    )
    assert report.fetched == 1
    assert (tmp_path / "good.mp4").exists()


def test_unsafe_item_id_is_rejected(tmp_path: Path):
    """A poisoned item id with path components must be rejected, never written
    outside --to (a hand-edited items.json is untrusted input)."""
    dest = tmp_path / "out"
    store = {"../escaped": _item("../escaped", media=[_pending()])}
    report = fetch_videos(
        store, ["../escaped"], dest, session=_ok_session(), sleep=_no_throttle, throttle_seconds=0
    )
    assert report.results[0].outcome == "skipped"
    assert report.results[0].reason == "invalid_id"
    assert not (tmp_path / "escaped.mp4").exists()
    assert list(dest.glob("*.mp4")) == []


def test_failed_entry_is_refetched(tmp_path: Path):
    entry = MediaVideoFailed(
        url=_MP4_URL,
        thumbnail_url=_POSTER,
        bitrate=2_176_000,
        duration_millis=30_000,
        failure_reason="http_5xx",
        error="boom",
        attempts=1,
        last_attempt_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )
    store = {"42": _item("42", media=[entry])}
    report = fetch_videos(
        store, ["42"], tmp_path, session=_ok_session(), sleep=_no_throttle, throttle_seconds=0
    )
    assert report.fetched == 1
