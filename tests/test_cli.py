# tests/test_cli.py
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from xbrain.cli import app
from xbrain.models import Author, Item, Link
from xbrain.store import save_store

runner = CliRunner()


def _setup_repo(tmp_path: Path, monkeypatch) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        f'vault = "{vault}"\n'
        'output_subdir = "x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n',
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    return vault


def _linked_item(item_id: str = "1") -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=f"Note {item_id}",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url="https://example.com/p", domain="example.com")],
    )


def test_status_runs_on_empty_store(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Items: 0" in result.stdout


def test_generate_creates_output_dir(tmp_path: Path, monkeypatch):
    vault = _setup_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["generate"])
    assert result.exit_code == 0
    assert (vault / "x-knowledge" / "_index.md").exists()


def test_status_reports_non_empty_store(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Items: 1" in result.stdout
    assert "con enlace: 1" in result.stdout


def test_generate_writes_item_note_for_linked_item(tmp_path: Path, monkeypatch):
    vault = _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["generate"])
    assert result.exit_code == 0
    notes = list((vault / "x-knowledge" / "items").glob("*.md"))
    assert notes


def test_cli_reports_missing_config_cleanly(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    result = runner.invoke(app, ["status"])
    assert result.exit_code != 0
    assert "Error:" in result.output


def test_media_command_runs_on_empty_store(tmp_path: Path, monkeypatch):
    """`xbrain media` with no media is a no-op (exit 0, items.json unchanged)."""
    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"1": _linked_item("1")}, items_path)
    before_bytes = items_path.read_bytes()

    result = runner.invoke(app, ["media"])

    assert result.exit_code == 0
    # The pre-media snapshot must have been created — the destructive-op
    # contract: every command that writes items.json snapshots first.
    snapshots = list((tmp_path / "data" / "snapshots").iterdir())
    assert any("pre-media" in p.name for p in snapshots)
    # items.json content is byte-identical: no media to advance, no
    # transitions, no spurious rewrites that would dirty timestamps.
    assert items_path.read_bytes() == before_bytes


def test_media_command_creates_media_dir_and_downloads_pending_photos(tmp_path: Path, monkeypatch):
    """End-to-end through the CLI with a fake session.

    The test patches `xbrain.media.requests.Session` to inject canned
    responses so the CLI hits the real `download_all` orchestrator
    without any real network call.
    """
    import io as _io

    from PIL import Image

    from xbrain.models import MediaPhotoDownloaded, MediaPhotoPending

    _setup_repo(tmp_path, monkeypatch)
    item = _linked_item("42")
    item.media = [MediaPhotoPending(url="https://pbs.twimg.com/media/Z.png")]
    save_store({"42": item}, tmp_path / "data" / "items.json")

    # Build a valid PNG byte payload Pillow can decode.
    buffer = _io.BytesIO()
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(buffer, format="PNG")
    bytes_data = buffer.getvalue()

    class _FakeSession:
        def __init__(self):
            self.headers: dict[str, str] = {}

        def get(self, _url, *, timeout):
            class _Resp:
                status_code = 200
                content = bytes_data

            return _Resp()

    monkeypatch.setattr("xbrain.media.requests.Session", _FakeSession)
    result = runner.invoke(app, ["media"])
    assert result.exit_code == 0, result.output

    # The downloaded photo landed in the variant + the file is on disk.
    from xbrain.store import load_store

    reloaded = load_store(tmp_path / "data" / "items.json")
    entry = reloaded["42"].media[0]
    assert isinstance(entry, MediaPhotoDownloaded)
    assert entry.local_path == "42/0.png"
    assert (tmp_path / "data" / "media" / "42" / "0.png").exists()


def test_media_command_resume_after_interrupt_completes_remaining(tmp_path: Path, monkeypatch):
    """Ctrl-C mid-batch leaves items.json valid; the next run completes the rest.

    Setup: an item with 3 pending photos. Run 1 raises KeyboardInterrupt
    after the 2nd photo's GET, so:
      - photo 0 was downloaded and persisted by `on_progress`,
      - photo 1's session.get raises, propagating out of `download_all`
        (mid-photo: no transition persisted for it, stays Pending),
      - photo 2 never starts.
    Run 2 uses a fresh session that succeeds for every photo and verifies
    that all three end up Downloaded.
    """
    import io as _io

    from PIL import Image

    from xbrain.models import MediaPhotoDownloaded, MediaPhotoPending

    _setup_repo(tmp_path, monkeypatch)
    item = _linked_item("99")
    item.media = [
        MediaPhotoPending(url="https://pbs.twimg.com/media/R0.png"),
        MediaPhotoPending(url="https://pbs.twimg.com/media/R1.png"),
        MediaPhotoPending(url="https://pbs.twimg.com/media/R2.png"),
    ]
    save_store({"99": item}, tmp_path / "data" / "items.json")

    buffer = _io.BytesIO()
    Image.new("RGB", (4, 3), color=(50, 60, 70)).save(buffer, format="PNG")
    png = buffer.getvalue()

    class _InterruptingSession:
        """Returns 200 for the 1st photo, raises KeyboardInterrupt on the 2nd."""

        def __init__(self):
            self.headers: dict[str, str] = {}
            self.call_count = 0

        def get(self, _url, *, timeout):
            self.call_count += 1
            if self.call_count == 2:
                raise KeyboardInterrupt

            class _Resp:
                status_code = 200
                content = png

            return _Resp()

    monkeypatch.setattr("xbrain.media.requests.Session", _InterruptingSession)
    result1 = runner.invoke(app, ["media"])
    # Typer surfaces KeyboardInterrupt as a non-zero exit code.
    assert result1.exit_code != 0

    from xbrain.store import load_store

    reloaded = load_store(tmp_path / "data" / "items.json")
    media = reloaded["99"].media
    assert isinstance(media[0], MediaPhotoDownloaded)
    # Photo 1 stays Pending because the KeyboardInterrupt fired BEFORE
    # the transition was recorded (the seam: `on_progress` is what
    # persists per-photo state; we never reached it).
    assert isinstance(media[1], MediaPhotoPending)
    assert isinstance(media[2], MediaPhotoPending)

    # Run 2: fresh session that succeeds for every GET.
    class _OkSession:
        def __init__(self):
            self.headers: dict[str, str] = {}

        def get(self, _url, *, timeout):
            class _Resp:
                status_code = 200
                content = png

            return _Resp()

    monkeypatch.setattr("xbrain.media.requests.Session", _OkSession)
    result2 = runner.invoke(app, ["media"])
    assert result2.exit_code == 0

    final = load_store(tmp_path / "data" / "items.json")
    for entry in final["99"].media:
        assert isinstance(entry, MediaPhotoDownloaded)


def test_media_command_propagates_total_failure_as_exit_1(tmp_path: Path, monkeypatch):
    """A run where every download fails surfaces as exit-1 at the CLI boundary."""
    from xbrain.models import MediaPhotoPending

    _setup_repo(tmp_path, monkeypatch)
    item = _linked_item("99")
    item.media = [MediaPhotoPending(url="https://pbs.twimg.com/media/X.png")]
    save_store({"99": item}, tmp_path / "data" / "items.json")

    class _FakeSession:
        def __init__(self):
            self.headers: dict[str, str] = {}

        def get(self, _url, *, timeout):
            class _Resp:
                status_code = 404
                content = b""

            return _Resp()

    monkeypatch.setattr("xbrain.media.requests.Session", _FakeSession)
    result = runner.invoke(app, ["media"])
    assert result.exit_code == 1
    # Even on total failure, the failure record is persisted: the CLI's
    # finally block writes the store before the RuntimeError propagates.
    from xbrain.store import load_store

    from xbrain.models import MediaPhotoFailed as _MPF

    reloaded = load_store(tmp_path / "data" / "items.json")
    assert isinstance(reloaded["99"].media[0], _MPF)


def _article_item_with_pending_image(item_id: str, image_url: str) -> Item:
    """An item whose `x_article` source carries one pending inline image (#39 PR4)."""
    from xbrain.models import (
        ArticleImageBlock,
        ArticleTextBlock,
        Content,
        ContentSourceSuccess,
        MediaPhotoPending,
    )

    source = ContentSourceSuccess(
        kind="x_article",
        url=f"https://x.com/i/article/{item_id}",
        title="An Article",
        text="Body text.",
        blocks=[
            ArticleTextBlock(text="Body text."),
            ArticleImageBlock(media=MediaPhotoPending(url=image_url)),
        ],
        http_status=200,
        attempts=1,
    )
    item = _linked_item(item_id)
    item.content = Content(
        fetched_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        sources=[source],
    )
    return item


def test_media_command_downloads_article_inline_images(tmp_path: Path, monkeypatch):
    """End-to-end: `xbrain media` downloads an Article's inline image to `<id>/article/`."""
    import io as _io

    from PIL import Image

    from xbrain.models import MediaPhotoDownloaded

    _setup_repo(tmp_path, monkeypatch)
    item = _article_item_with_pending_image("77", "https://pbs.twimg.com/media/ART.png")
    save_store({"77": item}, tmp_path / "data" / "items.json")

    buffer = _io.BytesIO()
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(buffer, format="PNG")
    bytes_data = buffer.getvalue()

    class _FakeSession:
        def __init__(self):
            self.headers: dict[str, str] = {}

        def get(self, _url, *, timeout):
            class _Resp:
                status_code = 200
                content = bytes_data

            return _Resp()

    monkeypatch.setattr("xbrain.media.requests.Session", _FakeSession)
    result = runner.invoke(app, ["media"])
    assert result.exit_code == 0, result.output

    from xbrain.store import load_store

    reloaded = load_store(tmp_path / "data" / "items.json")
    block = reloaded["77"].content.sources[0].blocks[1]
    assert isinstance(block.media, MediaPhotoDownloaded)
    assert block.media.local_path == "77/article/0.png"
    assert (tmp_path / "data" / "media" / "77" / "article" / "0.png").exists()
    # The pre-media snapshot fired (article images ride the SAME snapshot boundary).
    snapshots = list((tmp_path / "data" / "snapshots").iterdir())
    assert any("pre-media" in p.name for p in snapshots)


def test_media_command_snapshot_failure_aborts_before_any_write(tmp_path: Path, monkeypatch):
    """A snapshot failure propagates and aborts — the Article image stays pending."""
    from xbrain.models import MediaPhotoPending

    _setup_repo(tmp_path, monkeypatch)
    item = _article_item_with_pending_image("88", "https://pbs.twimg.com/media/ARTX.png")
    save_store({"88": item}, tmp_path / "data" / "items.json")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("snapshot disk full")

    # The download session must never be reached: if the snapshot aborts first,
    # no HTTP call and no store write happen.
    monkeypatch.setattr("xbrain.cli.snapshot.snapshot_create", _boom)
    result = runner.invoke(app, ["media"])
    assert result.exit_code != 0

    from xbrain.store import load_store

    reloaded = load_store(tmp_path / "data" / "items.json")
    block = reloaded["88"].content.sources[0].blocks[1]
    assert isinstance(block.media, MediaPhotoPending)  # untouched
    assert not (tmp_path / "data" / "media" / "88").exists()


def test_media_command_warns_when_items_filter_matches_nothing(tmp_path: Path, monkeypatch):
    """`--items` with IDs absent from the store prints an AVISO to stderr."""
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["media", "--items", "ghost-id-1,ghost-id-2"])
    assert result.exit_code == 0
    # The CLI mixes stdout and stderr in CliRunner output; the AVISO is on stderr
    # — combined output is fine for the substring check.
    assert "AVISO" in result.output
    assert "ghost-id-1" in result.output


def test_media_command_verbose_lists_failed_urls(tmp_path: Path, monkeypatch):
    """`--verbose` prints `<item_id> <reason> <url>` per failure on stderr."""
    from xbrain.models import MediaPhotoPending

    _setup_repo(tmp_path, monkeypatch)
    items = {
        "1": _linked_item("1"),
        "2": _linked_item("2"),
    }
    items["1"].media = [MediaPhotoPending(url="https://pbs.twimg.com/media/V1.png")]
    items["2"].media = [MediaPhotoPending(url="https://pbs.twimg.com/media/V2.png")]
    save_store(items, tmp_path / "data" / "items.json")

    import io as _io
    from PIL import Image

    buffer = _io.BytesIO()
    Image.new("RGB", (4, 3)).save(buffer, format="PNG")
    png = buffer.getvalue()

    class _MixedSession:
        """Photo for item 1 succeeds; photo for item 2 returns 404 three times."""

        def __init__(self):
            self.headers: dict[str, str] = {}

        def get(self, url, *, timeout):
            if "V1" in url:

                class _Ok:
                    status_code = 200
                    content = png

                return _Ok()

            class _Fail:
                status_code = 404
                content = b""

            return _Fail()

    monkeypatch.setattr("xbrain.media.requests.Session", _MixedSession)
    result = runner.invoke(app, ["media", "--verbose"])
    assert result.exit_code == 0
    assert "Failed media" in result.output
    assert "http_4xx" in result.output
    assert "V2" in result.output


def test_media_command_respects_items_filter(tmp_path: Path, monkeypatch):
    """`--items` limits the run to specific item IDs."""
    import io as _io

    from PIL import Image

    from xbrain.models import MediaPhotoPending

    _setup_repo(tmp_path, monkeypatch)
    item_a = _linked_item("a")
    item_b = _linked_item("b")
    item_a.media = [MediaPhotoPending(url="https://pbs.twimg.com/media/A.png")]
    item_b.media = [MediaPhotoPending(url="https://pbs.twimg.com/media/B.png")]
    save_store({"a": item_a, "b": item_b}, tmp_path / "data" / "items.json")

    buffer = _io.BytesIO()
    Image.new("RGB", (4, 3)).save(buffer, format="PNG")
    bytes_data = buffer.getvalue()

    class _FakeSession:
        def __init__(self):
            self.headers: dict[str, str] = {}

        def get(self, _url, *, timeout):
            class _Resp:
                status_code = 200
                content = bytes_data

            return _Resp()

    monkeypatch.setattr("xbrain.media.requests.Session", _FakeSession)
    result = runner.invoke(app, ["media", "--items", "b"])
    assert result.exit_code == 0

    from xbrain.store import load_store

    from xbrain.models import MediaPhotoDownloaded, MediaPhotoPending as _MPP

    reloaded = load_store(tmp_path / "data" / "items.json")
    # `a` stayed pending; only `b` was downloaded.
    assert isinstance(reloaded["a"].media[0], _MPP)
    assert isinstance(reloaded["b"].media[0], MediaPhotoDownloaded)


# ------------------------------------------------------------ download-videos


_MP4_URL = "https://video.twimg.com/ext_tw_video/1/vid/720/A.mp4?tag=12"
_HLS_URL = "https://video.twimg.com/ext_tw_video/1/pl/B.m3u8?c=fmp4"
_POSTER = "https://pbs.twimg.com/ext_tw_video_thumb/1/img/P.jpg"


def _video_item(item_id: str, url: str = _MP4_URL, source: str = "bookmark"):
    from xbrain.models import MediaVideoPending

    item = Item(
        id=item_id,
        source=source,
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=f"Note {item_id}",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )
    item.media = [
        MediaVideoPending(url=url, thumbnail_url=_POSTER, bitrate=2_176_000, duration_millis=30_000)
    ]
    return item


class _FakeVideoSession:
    """Fake session that returns mp4 bytes for any GET (videos are not decoded)."""

    payload = b"\x00\x00\x00\x18ftypmp42" + (b"\x00" * 1024)

    def __init__(self):
        self.headers: dict[str, str] = {}

    def get(self, _url, *, timeout):
        class _Resp:
            status_code = 200
            content = _FakeVideoSession.payload

        return _Resp()


class _FailVideoSession:
    """Fake session that returns a 500 for any GET (every download fails)."""

    def __init__(self):
        self.headers: dict[str, str] = {}

    def get(self, _url, *, timeout):
        class _Resp:
            status_code = 500
            content = b"err"

        return _Resp()


def test_download_videos_command_noop_when_no_videos(tmp_path: Path, monkeypatch):
    """A store with no downloadable mp4 is a no-op: exit 0, items.json unchanged,
    no confirmation needed, no snapshot taken (nothing is written)."""
    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"1": _linked_item("1")}, items_path)
    before = items_path.read_bytes()

    result = runner.invoke(app, ["download-videos"])

    assert result.exit_code == 0, result.output
    assert items_path.read_bytes() == before
    snapshots = tmp_path / "data" / "snapshots"
    assert not snapshots.exists() or not any(
        "pre-download-videos" in p.name for p in snapshots.iterdir()
    )


def test_download_videos_command_downloads_mp4_with_yes(tmp_path: Path, monkeypatch):
    """`--yes` bypasses the size gate; the mp4 lands as MediaVideoDownloaded with
    the bytes on disk, and the destructive-op snapshot fires first."""
    from xbrain.models import MediaVideoDownloaded
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")
    monkeypatch.setattr("xbrain.video_media.requests.Session", _FakeVideoSession)

    result = runner.invoke(app, ["download-videos", "--yes"])
    assert result.exit_code == 0, result.output
    assert "About to download" in result.output

    reloaded = load_store(tmp_path / "data" / "items.json")
    entry = reloaded["42"].media[0]
    assert isinstance(entry, MediaVideoDownloaded)
    assert entry.local_path == "42/0.mp4"
    assert (tmp_path / "data" / "media" / "42" / "0.mp4").exists()
    snapshots = list((tmp_path / "data" / "snapshots").iterdir())
    assert any("pre-download-videos" in p.name for p in snapshots)


def test_download_videos_command_aborts_when_declined(tmp_path: Path, monkeypatch):
    """Without `--yes`, declining the gate aborts: no download, and (item 6a) NO
    snapshot is left on disk — the snapshot is taken only after confirmation."""
    from xbrain.models import MediaVideoPending
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")
    monkeypatch.setattr("xbrain.video_media.requests.Session", _FakeVideoSession)

    result = runner.invoke(app, ["download-videos"], input="n\n")
    assert result.exit_code != 0
    reloaded = load_store(tmp_path / "data" / "items.json")
    assert isinstance(reloaded["42"].media[0], MediaVideoPending)
    assert not (tmp_path / "data" / "media" / "42" / "0.mp4").exists()
    # Item 6a: a declined gate leaves no snapshot behind.
    snapshots = tmp_path / "data" / "snapshots"
    assert not snapshots.exists() or not any(
        "pre-download-videos" in p.name for p in snapshots.iterdir()
    )


def test_download_videos_command_proceeds_when_confirmed(tmp_path: Path, monkeypatch):
    """Confirming the gate with `y` proceeds to download."""
    from xbrain.models import MediaVideoDownloaded
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")
    monkeypatch.setattr("xbrain.video_media.requests.Session", _FakeVideoSession)

    result = runner.invoke(app, ["download-videos"], input="y\n")
    assert result.exit_code == 0, result.output
    reloaded = load_store(tmp_path / "data" / "items.json")
    assert isinstance(reloaded["42"].media[0], MediaVideoDownloaded)


def test_download_videos_command_skips_hls(tmp_path: Path, monkeypatch):
    """An HLS-only store is a no-op for download but reports the deferred count —
    and still emits the SUMMARY line (item 5: monitor parity with `media`)."""
    from xbrain.models import MediaVideoPending
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    save_store({"7": _video_item("7", url=_HLS_URL)}, tmp_path / "data" / "items.json")

    result = runner.invoke(app, ["download-videos"])
    assert result.exit_code == 0, result.output
    assert "HLS" in result.output
    # Item 5: a skip-only run emits SUMMARY, just like the photo command.
    assert "SUMMARY:" in result.output
    assert "skipped_hls: 1" in result.output
    reloaded = load_store(tmp_path / "data" / "items.json")
    assert isinstance(reloaded["7"].media[0], MediaVideoPending)


def test_download_videos_command_source_filter(tmp_path: Path, monkeypatch):
    """`--source bookmarks` only touches bookmark items; own_tweets are left alone."""
    from xbrain.models import MediaVideoDownloaded, MediaVideoPending
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    save_store(
        {
            "b": _video_item("b", source="bookmark"),
            "t": _video_item("t", source="own_tweet"),
        },
        tmp_path / "data" / "items.json",
    )
    monkeypatch.setattr("xbrain.video_media.requests.Session", _FakeVideoSession)

    result = runner.invoke(app, ["download-videos", "--source", "bookmarks", "--yes"])
    assert result.exit_code == 0, result.output
    reloaded = load_store(tmp_path / "data" / "items.json")
    assert isinstance(reloaded["b"].media[0], MediaVideoDownloaded)
    assert isinstance(reloaded["t"].media[0], MediaVideoPending)


def test_download_videos_command_items_filter(tmp_path: Path, monkeypatch):
    """`--items` restricts the run to specific ids."""
    from xbrain.models import MediaVideoDownloaded, MediaVideoPending
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    save_store({"a": _video_item("a"), "b": _video_item("b")}, tmp_path / "data" / "items.json")
    monkeypatch.setattr("xbrain.video_media.requests.Session", _FakeVideoSession)

    result = runner.invoke(app, ["download-videos", "--items", "b", "--yes"])
    assert result.exit_code == 0, result.output
    reloaded = load_store(tmp_path / "data" / "items.json")
    assert isinstance(reloaded["a"].media[0], MediaVideoPending)
    assert isinstance(reloaded["b"].media[0], MediaVideoDownloaded)


def test_download_videos_command_persists_failed_on_total_failure(tmp_path: Path, monkeypatch):
    """Item 6b: a total-failure RuntimeError exits 1, but the CLI try/finally
    still persists the MediaVideoFailed record (no in-memory work lost)."""
    from xbrain.models import MediaVideoFailed
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")

    class _Fake404Session:
        def __init__(self):
            self.headers: dict[str, str] = {}

        def get(self, _url, *, timeout):
            class _Resp:
                status_code = 404
                content = b""
                headers: dict[str, str] = {}

            return _Resp()

    monkeypatch.setattr("xbrain.video_media.requests.Session", _Fake404Session)
    result = runner.invoke(app, ["download-videos", "--yes"])
    assert result.exit_code == 1  # total failure surfaces as clean exit-1
    reloaded = load_store(tmp_path / "data" / "items.json")
    entry = reloaded["42"].media[0]
    assert isinstance(entry, MediaVideoFailed)
    assert entry.failure_reason == "http_4xx"


def test_download_videos_command_max_size_skips_big_video(tmp_path: Path, monkeypatch):
    """Item 7 (CLI): `--max-size` skips an over-cap video; nothing is downloaded."""
    from xbrain.models import MediaVideoPending
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    # _video_item estimates 2_176_000 b/s × 30 s / 8 = 8.16 MB → over a 1MB cap.
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")
    monkeypatch.setattr("xbrain.video_media.requests.Session", _FakeVideoSession)

    result = runner.invoke(app, ["download-videos", "--max-size", "1MB", "--yes"])
    assert result.exit_code == 0, result.output
    assert "max-size" in result.output
    reloaded = load_store(tmp_path / "data" / "items.json")
    assert isinstance(reloaded["42"].media[0], MediaVideoPending)  # skipped, untouched
    assert not (tmp_path / "data" / "media" / "42" / "0.mp4").exists()


def test_download_videos_command_rejects_bad_max_size(tmp_path: Path, monkeypatch):
    """A garbage `--max-size` value is a clean operator error (exit 1)."""
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["download-videos", "--max-size", "banana", "--yes"])
    assert result.exit_code == 1
    assert "Error:" in result.output


def test_parse_date_returns_utc_aware():
    from xbrain.cli import _parse_date

    parsed = _parse_date("2025-01-01")
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert _parse_date(None) is None


def test_parse_date_since_keeps_midnight():
    """A date-only `since` stays at the day's start (00:00:00) — no snapping."""
    from xbrain.cli import _parse_date

    parsed = _parse_date("2025-12-31")
    assert parsed == datetime(2025, 12, 31, 0, 0, 0, 0, tzinfo=timezone.utc)


def test_parse_date_until_snaps_date_only_to_end_of_day():
    """A date-only `until` snaps to the last microsecond so the whole day is included."""
    from xbrain.cli import _parse_date

    parsed = _parse_date("2025-12-31", end_of_day=True)
    assert parsed == datetime(2025, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)


def test_parse_date_until_respects_explicit_time():
    """An explicit time on `until` is respected as-is — never snapped to end of day."""
    from xbrain.cli import _parse_date

    parsed = _parse_date("2025-12-31T09:00:00", end_of_day=True)
    assert parsed == datetime(2025, 12, 31, 9, 0, 0, 0, tzinfo=timezone.utc)


def test_parse_date_until_space_basic_time_is_respected():
    """A space-separated basic-format time (no colon) is a time — must NOT snap."""
    from xbrain.cli import _parse_date

    parsed = _parse_date("2025-12-31 120000", end_of_day=True)
    assert parsed == datetime(2025, 12, 31, 12, 0, 0, 0, tzinfo=timezone.utc)


def test_parse_date_until_bare_date_with_offset_snaps():
    """A bare date carrying only a tz offset (no time-of-day) still snaps to end of day."""
    from xbrain.cli import _parse_date

    parsed = _parse_date("2025-12-31+00:00", end_of_day=True)
    assert parsed == datetime(2025, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)


def test_cli_generate_until_date_only_includes_whole_final_day(tmp_path: Path, monkeypatch):
    """`--until <date>` (date-only) includes items up to 23:59:59.999999 of that day.

    Pins the microsecond boundary: an item at the last microsecond of the day is
    INCLUDED, the next-day midnight item is EXCLUDED. Catches a `>`→`>=` flip or any
    coarsening of the snap value.
    """
    vault = _setup_repo(tmp_path, monkeypatch)
    midday = _linked_item("1")
    midday.created_at = datetime(2026, 1, 15, 15, 0, 0, tzinfo=timezone.utc)
    last_us = _linked_item("2")
    last_us.created_at = datetime(2026, 1, 15, 23, 59, 59, 999999, tzinfo=timezone.utc)
    next_day = _linked_item("3")
    next_day.created_at = datetime(2026, 1, 16, 0, 0, 0, tzinfo=timezone.utc)
    save_store({"1": midday, "2": last_us, "3": next_day}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["generate", "--until", "2026-01-15"])
    assert result.exit_code == 0
    notes = list((vault / "x-knowledge" / "items").glob("*.md"))
    names = sorted(n.name for n in notes)
    assert len(names) == 2
    assert names[0].endswith("-1.md") and names[1].endswith("-2.md")
    assert not any(n.endswith("-3.md") for n in names)


def test_cli_generate_until_explicit_time_is_respected(tmp_path: Path, monkeypatch):
    """`--until <date>T09:00:00` excludes an item at 10:00 and includes one at 08:00."""
    vault = _setup_repo(tmp_path, monkeypatch)
    after = _linked_item("1")
    after.created_at = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    before = _linked_item("2")
    before.created_at = datetime(2026, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
    save_store({"1": after, "2": before}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["generate", "--until", "2026-01-15T09:00:00"])
    assert result.exit_code == 0
    notes = list((vault / "x-knowledge" / "items").glob("*.md"))
    assert len(notes) == 1
    assert notes[0].name.endswith("-2.md")


def test_cli_generate_since_date_only_includes_midnight_item(tmp_path: Path, monkeypatch):
    """`--since <date>` (date-only) still includes an item created at that day's midnight."""
    vault = _setup_repo(tmp_path, monkeypatch)
    item = _linked_item("1")
    item.created_at = datetime(2026, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
    save_store({"1": item}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["generate", "--since", "2026-01-15"])
    assert result.exit_code == 0
    notes = list((vault / "x-knowledge" / "items").glob("*.md"))
    assert len(notes) == 1


def test_cli_enrich_until_date_only_includes_whole_final_day(tmp_path, monkeypatch):
    """Pins the `end_of_day=True` wiring on a SECOND command (`enrich`), not just `generate`.

    An item created on the `--until` day at 15:00 must count as pending — a mutation
    dropping `end_of_day=True` from the enrich site would exclude it (0 pending).
    """
    _setup_repo(tmp_path, monkeypatch)
    from xbrain.models import Topic
    from xbrain.rubrics import save_vocab

    item = _linked_item("1")
    item.created_at = datetime(2026, 1, 15, 15, 0, 0, tzinfo=timezone.utc)
    save_store({"1": item}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    result = runner.invoke(app, ["enrich", "--executor", "manual", "--until", "2026-01-15"])
    assert result.exit_code == 0
    assert "1 items exportados" in result.output
    assert (tmp_path / "data" / "enrich-worksheet.json").exists()


def test_cli_fetch_with_since_does_not_crash(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    old_item = _linked_item("1")
    old_item.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    save_store({"1": old_item}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["fetch", "--since", "2025-01-01"])
    assert result.exit_code == 0


def test_cli_generate_with_since_filters_notes(tmp_path: Path, monkeypatch):
    vault = _setup_repo(tmp_path, monkeypatch)
    old_item = _linked_item("1")
    old_item.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    new_item = _linked_item("2")
    new_item.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    save_store({"1": old_item, "2": new_item}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["generate", "--since", "2025-01-01"])
    assert result.exit_code == 0
    assert (vault / "x-knowledge" / "_index.md").exists()
    notes = list((vault / "x-knowledge" / "items").glob("*.md"))
    assert len(notes) == 1


def test_vocab_command_persists_induced_topics(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    from xbrain.store import save_store

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    from xbrain.models import Topic
    import xbrain.cli as cli

    monkeypatch.setattr(
        cli, "induce_vocab", lambda *a, **k: [Topic(slug="misc", description="Noise.")]
    )
    result = runner.invoke(app, ["vocab", "--executor", "api"])
    assert result.exit_code == 0
    from xbrain.rubrics import load_vocab

    assert [t.slug for t in load_vocab(tmp_path / "data" / "vocab.yaml")] == ["misc"]


def test_enrich_manual_exports_a_worksheet(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    from xbrain.store import save_store
    from xbrain.rubrics import save_vocab
    from xbrain.models import Topic

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    result = runner.invoke(app, ["enrich", "--executor", "manual"])
    assert result.exit_code == 0
    assert (tmp_path / "data" / "enrich-worksheet.json").exists()


def test_enrich_apply_imports_a_filled_worksheet(tmp_path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    from xbrain.store import save_store
    from xbrain.rubrics import save_vocab
    from xbrain.models import Topic

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    ws = tmp_path / "ws.json"
    ws.write_text(
        json.dumps(
            {
                "judgments": [
                    {"item_id": "1", "summary": "s", "primary_topic": "misc", "topics": ["misc"]}
                ]
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["enrich", "--apply", str(ws)])
    assert result.exit_code == 0
    from xbrain.store import load_store

    store = load_store(tmp_path / "data" / "items.json")
    assert store["1"].enriched is not None


def test_enrich_api_executor_enriches_the_store(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    from xbrain.store import load_store, save_store
    from xbrain.rubrics import save_vocab
    from xbrain.models import Topic
    from xbrain.executors.base import EnrichmentJudgment
    import xbrain.cli as cli

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")

    class _FakeApiExecutor:
        def __init__(self, *a, **k):
            pass

        def enrich_items(self, items, vocab):
            return [
                EnrichmentJudgment(
                    item_id=i.id, summary="resumen", primary_topic="misc", topics=["misc"]
                )
                for i in items
            ]

    monkeypatch.setattr(cli, "ApiExecutor", _FakeApiExecutor)
    result = runner.invoke(app, ["enrich", "--executor", "api"])
    assert result.exit_code == 0
    store = load_store(tmp_path / "data" / "items.json")
    assert store["1"].enriched is not None


def test_enrich_rejects_unknown_executor(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    from xbrain.store import save_store
    from xbrain.rubrics import save_vocab
    from xbrain.models import Topic

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    result = runner.invoke(app, ["enrich", "--executor", "bogus"])
    assert result.exit_code == 1
    assert "bogus" in result.output
    assert "desconocido" in result.output


def _photo_item(item_id: str = "1") -> Item:
    from xbrain.models import MediaPhotoDownloaded

    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=f"Note {item_id}",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        media=[
            MediaPhotoDownloaded(
                url="https://p/0.png",
                local_path=f"{item_id}/0.png",
                width=4,
                height=4,
                bytes_size=9,
                downloaded_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
            )
        ],
    )


def test_describe_rejects_unknown_executor(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["describe", "--executor", "bogus"])
    assert result.exit_code == 1
    assert "bogus" in result.output
    assert "desconocido" in result.output


def test_describe_claude_code_exports_a_worksheet(tmp_path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _photo_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["describe", "--executor", "claude-code"])
    assert result.exit_code == 0
    ws = tmp_path / "data" / "describe-worksheet.json"
    assert ws.exists()
    payload = json.loads(ws.read_text(encoding="utf-8"))
    assert len(payload["photos"]) == 1
    assert payload["judgments"] == []


def test_describe_apply_dispatches_and_reports_unmatched(tmp_path, monkeypatch):
    import json

    from xbrain.models import MediaPhotoDescribed
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _photo_item("1")}, tmp_path / "data" / "items.json")
    ws = tmp_path / "ws.json"
    ws.write_text(
        json.dumps(
            {
                "version": "v1",
                "language": "English",
                "judgments": [
                    {"item_id": "1", "index": 0, "is_decorative": False, "description": "A chart."},
                    {"item_id": "ghost", "index": 0, "is_decorative": False, "description": "x"},
                ],
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["describe", "--apply", str(ws)])
    assert result.exit_code == 0
    assert "1 fotos descritas" in result.output
    assert "ghost#0" in result.output  # unmatched judgment surfaced, not dropped silently
    store = load_store(tmp_path / "data" / "items.json")
    assert isinstance(store["1"].media[0], MediaPhotoDescribed)


def test_enrich_manual_without_vocab_fails(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    from xbrain.store import save_store

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["enrich", "--executor", "manual"])
    assert result.exit_code == 1
    assert "vocabulario" in result.output


def test_enrich_apply_without_vocab_fails(tmp_path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    from xbrain.store import save_store

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    ws = tmp_path / "ws.json"
    ws.write_text(json.dumps({"judgments": []}), encoding="utf-8")
    result = runner.invoke(app, ["enrich", "--apply", str(ws)])
    assert result.exit_code == 1
    assert "vocabulario" in result.output


def test_enrich_manual_with_no_pending_items_writes_no_worksheet(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    _setup_repo(tmp_path, monkeypatch)
    from xbrain.store import save_store
    from xbrain.rubrics import save_vocab
    from xbrain.models import Enrichment, Topic

    item = _linked_item("1")
    item.enriched = Enrichment(
        enriched_at=datetime.now(timezone.utc),
        executor="manual",
        summary="s",
        primary_topic="misc",
        topics=["misc"],
    )
    save_store({"1": item}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    result = runner.invoke(app, ["enrich", "--executor", "manual"])
    assert result.exit_code == 0
    assert "No hay items pendientes" in result.output
    assert not (tmp_path / "data" / "enrich-worksheet.json").exists()


def test_cli_fetch_reports_x_articles(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    import xbrain.cli as cli

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    monkeypatch.setattr(cli, "fetch_pending", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "fetch_x_articles", lambda *a, **k: 3)
    monkeypatch.setattr(cli, "expand_threads", lambda *a, **k: 0)
    result = runner.invoke(app, ["fetch"])
    assert result.exit_code == 0
    assert "3 de X" in result.output


def test_cli_fetch_persists_partial_work_when_a_stage_raises(tmp_path, monkeypatch):
    # `_run_fetch` wraps the three fetch stages in `try:` and `save_store` in
    # `finally:` — a stage error must not discard the in-memory work the
    # earlier stages already produced. This proves that `finally` contract.
    _setup_repo(tmp_path, monkeypatch)
    import xbrain.cli as cli
    from xbrain.models import Content, ContentSourceSuccess
    from xbrain.store import load_store

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")

    def _fake_fetch_pending(store, *a, **k):
        # Mutate the SAME store object _run_fetch passed in — that is what
        # the `finally` block later persists to disk.
        store["1"].content = Content(
            fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
            sources=[
                ContentSourceSuccess(
                    kind="external_article",
                    url="https://example.com/p",
                    text="cuerpo",
                )
            ],
        )
        return 1

    def _fake_fetch_x_articles(*a, **k):
        raise RuntimeError("Sesión de X caducada")

    monkeypatch.setattr(cli, "fetch_pending", _fake_fetch_pending)
    monkeypatch.setattr(cli, "fetch_x_articles", _fake_fetch_x_articles)

    result = runner.invoke(app, ["fetch"])
    # The stage error still surfaces via _handle_cli_errors.
    assert result.exit_code == 1

    # ...but the partial work from fetch_pending survived: save_store ran in
    # the `finally` despite the raise.
    store = load_store(tmp_path / "data" / "items.json")
    assert store["1"].content is not None
    assert store["1"].content.sources[0].text == "cuerpo"


def _enriched_item(item_id: str = "1"):
    from xbrain.models import Enrichment

    item = _linked_item(item_id)
    item.enriched = Enrichment(
        enriched_at=datetime.now(timezone.utc),
        executor="api",
        summary="resumen",
        primary_topic="misc",
        topics=["misc"],
    )
    return item


def test_topics_claude_code_exports_a_worksheet(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    from xbrain.models import Topic
    from xbrain.rubrics import save_vocab

    save_store({"1": _enriched_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    result = runner.invoke(app, ["topics", "--executor", "claude-code"])
    assert result.exit_code == 0
    assert (tmp_path / "data" / "topic-worksheet.json").exists()
    assert (tmp_path / "vault" / "x-knowledge" / "topics" / "misc.md").exists()


def test_topics_apply_writes_the_overview(tmp_path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    from xbrain.models import Topic
    from xbrain.rubrics import save_vocab

    save_store({"1": _enriched_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    ws = tmp_path / "ws.json"
    ws.write_text(
        json.dumps(
            {"judgments": [{"slug": "misc", "overview": "Resumen del cajón.", "notes": []}]}
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["topics", "--apply", str(ws)])
    assert result.exit_code == 0
    page = (tmp_path / "vault" / "x-knowledge" / "topics" / "misc.md").read_text(encoding="utf-8")
    assert "Resumen del cajón." in page


def test_topics_api_executor_synthesizes(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    import xbrain.cli as cli
    from xbrain.models import Topic
    from xbrain.rubrics import save_vocab
    from xbrain.topic_synth import OverviewJudgment

    save_store({"1": _enriched_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")

    def _fake_synth(inputs, model, output_language="English", **kwargs):
        return [OverviewJudgment(slug="misc", overview="Sintetizado por API.", notes=[])]

    monkeypatch.setattr(cli, "synthesize_overviews_api", _fake_synth)
    result = runner.invoke(app, ["topics", "--executor", "api"])
    assert result.exit_code == 0
    page = (tmp_path / "vault" / "x-knowledge" / "topics" / "misc.md").read_text(encoding="utf-8")
    assert "Sintetizado por API." in page


def test_topics_without_vocab_fails(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _enriched_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["topics"])
    assert result.exit_code == 1
    assert "vocabulario" in result.output


def test_topics_run_with_no_stale_overviews(tmp_path, monkeypatch):
    # Store enriched + an up-to-date TopicPage (count matches the live posts):
    # nothing is stale, so the run only refreshes the lists.
    _setup_repo(tmp_path, monkeypatch)
    from xbrain.models import Topic, TopicPage
    from xbrain.rubrics import save_vocab
    from xbrain.store import save_topic_pages

    save_store({"1": _enriched_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    save_topic_pages(
        {
            "misc": TopicPage(
                slug="misc",
                overview="Overview ya sintetizado.",
                notes=[],
                synthesized_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
                post_count_at_synth=1,
            )
        },
        tmp_path / "data" / "topics.json",
    )
    result = runner.invoke(app, ["topics"])
    assert result.exit_code == 0
    assert "sin overviews pendientes" in result.output


def test_topics_resynth_succeeds(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    import xbrain.cli as cli
    from xbrain.models import Topic, TopicPage
    from xbrain.rubrics import save_vocab
    from xbrain.store import save_topic_pages
    from xbrain.topic_synth import OverviewJudgment

    save_store({"1": _enriched_item("1")}, tmp_path / "data" / "items.json")
    save_vocab([Topic(slug="misc", description="d")], tmp_path / "data" / "vocab.yaml")
    # Page synthesized when the topic had 0 posts — a resynth picks up the change.
    save_topic_pages(
        {
            "misc": TopicPage(
                slug="misc",
                overview="Overview viejo.",
                notes=[],
                synthesized_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
                post_count_at_synth=0,
            )
        },
        tmp_path / "data" / "topics.json",
    )

    def _fake_synth(inputs, model, output_language="English", **kwargs):
        return [OverviewJudgment(slug="misc", overview="Re-sintetizado.", notes=[])]

    monkeypatch.setattr(cli, "synthesize_overviews_api", _fake_synth)
    result = runner.invoke(app, ["topics", "--resynth", "--executor", "api"])
    assert result.exit_code == 0
    page = (tmp_path / "vault" / "x-knowledge" / "topics" / "misc.md").read_text(encoding="utf-8")
    assert "Re-sintetizado." in page


def test_vocab_claude_code_exports_a_worksheet(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["vocab", "--executor", "claude-code"])
    assert result.exit_code == 0
    assert (tmp_path / "data" / "vocab-worksheet.json").exists()


def test_vocab_apply_writes_vocab_yaml(tmp_path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    ws = tmp_path / "ws.json"
    ws.write_text(
        json.dumps({"topics": [{"slug": "misc", "description": "Ruido."}]}), encoding="utf-8"
    )
    result = runner.invoke(app, ["vocab", "--apply", str(ws)])
    assert result.exit_code == 0
    from xbrain.rubrics import load_vocab

    assert [t.slug for t in load_vocab(tmp_path / "data" / "vocab.yaml")] == ["misc"]


def test_vocab_apply_regenerate_marks_items(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timezone

    from xbrain.models import Enrichment

    _setup_repo(tmp_path, monkeypatch)
    item = _linked_item("1")
    item.enriched = Enrichment(
        enriched_at=datetime.now(timezone.utc),
        executor="manual",
        summary="s",
        primary_topic="misc",
        topics=["misc"],
    )
    save_store({"1": item}, tmp_path / "data" / "items.json")
    ws = tmp_path / "ws.json"
    ws.write_text(
        json.dumps({"topics": [{"slug": "misc", "description": "Ruido."}]}), encoding="utf-8"
    )
    result = runner.invoke(app, ["vocab", "--apply", str(ws), "--regenerate"])
    assert result.exit_code == 0
    from xbrain.store import load_store as _ls

    assert _ls(tmp_path / "data" / "items.json")["1"].enriched is None


def test_vocab_apply_with_no_valid_topics_fails(tmp_path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    ws = tmp_path / "ws.json"
    ws.write_text(json.dumps({"topics": [{"slug": "BAD SLUG"}]}), encoding="utf-8")
    result = runner.invoke(app, ["vocab", "--apply", str(ws)])
    assert result.exit_code == 1
    # `_report_invalid` must run BEFORE the raise — the user needs to see WHY
    # the topic was rejected, so the bad slug shows up in the output.
    assert "BAD SLUG" in result.output


def test_vocab_api_executor_still_works(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    import xbrain.cli as cli
    from xbrain.models import Topic

    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    monkeypatch.setattr(cli, "induce_vocab", lambda *a, **k: [Topic(slug="misc", description="d")])
    result = runner.invoke(app, ["vocab", "--executor", "api"])
    assert result.exit_code == 0
    from xbrain.rubrics import load_vocab

    assert [t.slug for t in load_vocab(tmp_path / "data" / "vocab.yaml")] == ["misc"]


def test_vocab_rejects_unknown_executor(tmp_path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["vocab", "--executor", "bogus"])
    assert result.exit_code == 1
    assert "bogus" in result.output


# ----------------------------------------------------------------------- describe


def _setup_describe_repo(tmp_path: Path, monkeypatch) -> Path:
    """Like `_setup_repo` but with a pre-populated photo on disk + Downloaded variant."""
    import io as _io
    from datetime import datetime, timezone

    from PIL import Image

    from xbrain.models import Author, Item, MediaPhotoDownloaded

    _setup_repo(tmp_path, monkeypatch)
    media_root = tmp_path / "data" / "media"
    (media_root / "42").mkdir(parents=True, exist_ok=True)
    buffer = _io.BytesIO()
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(buffer, format="JPEG")
    (media_root / "42" / "0.jpg").write_bytes(buffer.getvalue())

    item = Item(
        id="42",
        source="bookmark",
        url="https://x.com/a/status/42",
        author=Author(handle="a", name="A"),
        text="text",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        media=[
            MediaPhotoDownloaded(
                url="https://pbs.twimg.com/media/42-0.jpg",
                local_path="42/0.jpg",
                width=8,
                height=6,
                bytes_size=200,
                downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            )
        ],
    )
    save_store({"42": item}, tmp_path / "data" / "items.json")
    return tmp_path


def test_describe_command_transitions_downloaded_to_described(tmp_path: Path, monkeypatch):
    """End-to-end: a Downloaded photo becomes Described via the CLI command."""
    import json as _json

    from xbrain.models import MediaPhotoDescribed
    from xbrain.store import load_store

    _setup_describe_repo(tmp_path, monkeypatch)

    class _FakeBlock:
        type = "text"

        def __init__(self, text: str):
            self.text = text

    class _FakeResp:
        def __init__(self, text: str):
            self.content = [_FakeBlock(text)]

    class _FakeMessages:
        def __init__(self):
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            payload = _json.dumps([{"index": 0, "is_decorative": False, "description": "A chart."}])
            return _FakeResp(payload)

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    import xbrain.cli as cli

    fake_client = _FakeClient()

    def _patched_run(store, media_root, **kwargs):
        kwargs["client"] = fake_client
        return _orig(store, media_root, **kwargs)

    _orig = cli.run_describe_all
    monkeypatch.setattr(cli, "run_describe_all", _patched_run)

    result = runner.invoke(app, ["describe"])
    assert result.exit_code == 0, result.output
    reloaded = load_store(tmp_path / "data" / "items.json")
    entry = reloaded["42"].media[0]
    assert isinstance(entry, MediaPhotoDescribed)
    assert entry.description == "A chart."
    assert entry.description_lang == "English"
    assert entry.description_version == "v1"


def test_describe_command_runs_on_empty_store(tmp_path: Path, monkeypatch):
    """No items → describe is a no-op exit-0, with a snapshot still taken."""
    _setup_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["describe"])
    assert result.exit_code == 0
    # A snapshot was created — recovery boundary mirrors media.
    snapshots = list((tmp_path / "data" / "snapshots").glob("*-describe"))
    assert snapshots, "describe must auto-snapshot data/ before running"


def test_describe_command_warns_when_items_filter_matches_nothing(tmp_path, monkeypatch):
    """`--items` with no matches surfaces an AVISO line — same pattern as media."""
    _setup_describe_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["describe", "--items", "no-such-id"])
    assert result.exit_code == 0
    assert "AVISO" in result.output or "AVISO" in (result.stderr or "")


def test_describe_command_propagates_total_failure_as_exit_1(tmp_path, monkeypatch):
    """A total-failure RuntimeError surfaces as exit code 1 via `_handle_cli_errors`."""
    from anthropic import APIError

    _setup_describe_repo(tmp_path, monkeypatch)

    class _AlwaysFailingMessages:
        def __init__(self):
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            raise APIError("401 unauthorized", request=None, body=None)

    class _FakeClient:
        def __init__(self):
            self.messages = _AlwaysFailingMessages()

    import xbrain.cli as cli

    def _patched(store, media_root, **kwargs):
        kwargs["client"] = _FakeClient()
        return _orig(store, media_root, **kwargs)

    _orig = cli.run_describe_all
    monkeypatch.setattr(cli, "run_describe_all", _patched)

    result = runner.invoke(app, ["describe"])
    assert result.exit_code == 1
    assert "Error" in result.output


def test_describe_command_emits_exactly_one_summary_on_partial_failure(tmp_path, monkeypatch):
    """The CLI is the single source of truth for the SUMMARY line.

    A partial-failure run must emit EXACTLY one `SUMMARY:` on stderr —
    if the orchestrator regrows a second emitter the count goes to 2
    and this pins it. Mirrors the dedup test on the orchestrator
    side (`test_describe_all_does_not_emit_summary_on_partial_failure`).
    """
    import io as _io
    import json as _json
    from datetime import datetime, timezone

    from anthropic import APIError
    from PIL import Image

    from xbrain.models import Author, Item, MediaPhotoDownloaded
    from xbrain.store import save_store as _save_store

    _setup_repo(tmp_path, monkeypatch)
    media_root = tmp_path / "data" / "media"
    (media_root / "1").mkdir(parents=True, exist_ok=True)
    (media_root / "2").mkdir(parents=True, exist_ok=True)
    buf = _io.BytesIO()
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(buf, format="JPEG")
    (media_root / "1" / "0.jpg").write_bytes(buf.getvalue())
    (media_root / "2" / "0.jpg").write_bytes(buf.getvalue())

    def _build_item(item_id: str) -> Item:
        return Item(
            id=item_id,
            source="bookmark",
            url=f"https://x.com/a/status/{item_id}",
            author=Author(handle="a", name="A"),
            text="text",
            created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
            captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            media=[
                MediaPhotoDownloaded(
                    url=f"https://pbs.twimg.com/media/{item_id}-0.jpg",
                    local_path=f"{item_id}/0.jpg",
                    width=8,
                    height=6,
                    bytes_size=200,
                    downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
                )
            ],
        )

    _save_store(
        {"1": _build_item("1"), "2": _build_item("2")},
        tmp_path / "data" / "items.json",
    )

    class _PartialFailMessages:
        def __init__(self):
            self.calls: list[dict] = []
            self._counter = 0

        def create(self, **kwargs):
            self.calls.append(kwargs)
            self._counter += 1
            if self._counter == 1:
                # First batch fails.
                raise APIError("503", request=None, body=None)

            # Second batch returns one judgment.
            class _Block:
                type = "text"

                def __init__(self, text: str):
                    self.text = text

            class _Resp:
                def __init__(self, text: str):
                    self.content = [_Block(text)]
                    self.stop_reason = "end_turn"

            return _Resp(_json.dumps([{"index": 0, "is_decorative": False, "description": "ok"}]))

    class _FakeClient:
        def __init__(self):
            self.messages = _PartialFailMessages()

    import xbrain.cli as cli

    fake = _FakeClient()

    def _patched(store, media_root, **kwargs):
        kwargs["client"] = fake
        kwargs["batch_size"] = 1
        return _orig(store, media_root, **kwargs)

    _orig = cli.run_describe_all
    monkeypatch.setattr(cli, "run_describe_all", _patched)

    result = runner.invoke(app, ["describe"])
    assert result.exit_code == 0
    # Exactly one SUMMARY emission — the CLI's `emit_summary_line` is
    # the single source of truth; the orchestrator stays silent.
    assert result.output.count("SUMMARY:") == 1


def test_describe_command_verbose_lists_failed_photos(tmp_path, monkeypatch):
    """`--verbose` prints `Failed photos:` plus per-failure rows on partial failure.

    Pins the diagnostic branch in `_run_describe` so a future refactor
    that drops the verbose output is caught.
    """
    import io as _io
    import json as _json
    from datetime import datetime, timezone

    from anthropic import APIError
    from PIL import Image

    from xbrain.models import Author, Item, MediaPhotoDownloaded
    from xbrain.store import save_store as _save_store

    _setup_repo(tmp_path, monkeypatch)
    media_root = tmp_path / "data" / "media"
    (media_root / "1").mkdir(parents=True, exist_ok=True)
    (media_root / "2").mkdir(parents=True, exist_ok=True)
    buf = _io.BytesIO()
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(buf, format="JPEG")
    (media_root / "1" / "0.jpg").write_bytes(buf.getvalue())
    (media_root / "2" / "0.jpg").write_bytes(buf.getvalue())

    def _build_item(item_id: str) -> Item:
        return Item(
            id=item_id,
            source="bookmark",
            url=f"https://x.com/a/status/{item_id}",
            author=Author(handle="a", name="A"),
            text="text",
            created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
            captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            media=[
                MediaPhotoDownloaded(
                    url=f"https://pbs.twimg.com/media/{item_id}-0.jpg",
                    local_path=f"{item_id}/0.jpg",
                    width=8,
                    height=6,
                    bytes_size=200,
                    downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
                )
            ],
        )

    _save_store(
        {"1": _build_item("1"), "2": _build_item("2")},
        tmp_path / "data" / "items.json",
    )

    class _PartialFailMessages:
        def __init__(self):
            self.calls: list[dict] = []
            self._counter = 0

        def create(self, **kwargs):
            self.calls.append(kwargs)
            self._counter += 1
            if self._counter == 1:
                raise APIError("503", request=None, body=None)

            class _Block:
                type = "text"

                def __init__(self, text: str):
                    self.text = text

            class _Resp:
                def __init__(self, text: str):
                    self.content = [_Block(text)]
                    self.stop_reason = "end_turn"

            return _Resp(_json.dumps([{"index": 0, "is_decorative": False, "description": "ok"}]))

    class _FakeClient:
        def __init__(self):
            self.messages = _PartialFailMessages()

    import xbrain.cli as cli

    fake = _FakeClient()

    def _patched(store, media_root, **kwargs):
        kwargs["client"] = fake
        kwargs["batch_size"] = 1
        return _orig(store, media_root, **kwargs)

    _orig = cli.run_describe_all
    monkeypatch.setattr(cli, "run_describe_all", _patched)

    result = runner.invoke(app, ["describe", "--verbose"])
    assert result.exit_code == 0
    assert "Failed photos:" in result.output
    # The failed item id (1 or 2 — order is filesystem-dependent) and the URL
    # both appear on the verbose row.
    assert "pbs.twimg.com" in result.output


# ------------------------------------------------------------------ refresh-media


def _poster_era_item(item_id: str = "42"):
    """A store item with a downloaded photo + a poster-era video (url = poster)."""
    from xbrain.models import Author, Item, MediaPhotoDownloaded, MediaVideoPending

    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        media=[
            MediaPhotoDownloaded(
                url="https://pbs.twimg.com/media/42-0.jpg",
                local_path=f"{item_id}/0.jpg",
                width=8,
                height=6,
                bytes_size=200,
                downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            ),
            MediaVideoPending(url="https://pbs.twimg.com/poster.jpg"),
        ],
    )


def _playable_fresh_item(item_id: str = "42"):
    """A fresh capture of the same id whose video carries the playable stream."""
    from xbrain.models import Author, Item, MediaVideoPending

    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        media=[
            MediaVideoPending(
                url="https://v/high.mp4",
                thumbnail_url="https://pbs.twimg.com/poster.jpg",
                bitrate=2_176_000,
                duration_millis=30_000,
            )
        ],
    )


def _mock_browser(monkeypatch, extract_impl):
    """Patch `cli.x_context` (a no-op CM) + `cli.extract_source` (the impl)."""
    import contextlib

    import xbrain.cli as cli

    @contextlib.contextmanager
    def _ctx(_path, *, headless=False):
        yield object()

    monkeypatch.setattr(cli, "x_context", _ctx)
    monkeypatch.setattr(cli, "extract_source", extract_impl)


def test_refresh_media_backfills_video_and_preserves_photo(tmp_path: Path, monkeypatch):
    """End-to-end: the poster-era video gains the playable URL; the photo survives."""
    from xbrain.models import MediaPhotoDownloaded, MediaVideoPending
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _poster_era_item("42")}, tmp_path / "data" / "items.json")
    _mock_browser(monkeypatch, lambda *a, **k: [_playable_fresh_item("42")])

    result = runner.invoke(app, ["refresh-media"])
    assert result.exit_code == 0, result.output

    reloaded = load_store(tmp_path / "data" / "items.json")
    media = reloaded["42"].media
    # The downloaded photo is untouched (still Downloaded, same local_path).
    assert isinstance(media[0], MediaPhotoDownloaded)
    assert media[0].local_path == "42/0.jpg"
    # The video now carries the playable stream + bitrate + duration.
    video = media[1]
    assert isinstance(video, MediaVideoPending)
    assert video.url == "https://v/high.mp4"
    assert video.bitrate == 2_176_000
    assert video.duration_millis == 30_000
    # The END summary prints the report counts + a size estimate (with its
    # full "...; N with unknown size." tail — the refreshed mp4 is estimable).
    assert "1 refreshed" in result.output
    assert "Estimated video download" in result.output
    assert "0 with unknown size." in result.output


def test_refresh_media_creates_pre_snapshot(tmp_path: Path, monkeypatch):
    """The destructive op snapshots `data/` before writing (label pre-refresh-media)."""
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _poster_era_item("42")}, tmp_path / "data" / "items.json")
    _mock_browser(monkeypatch, lambda *a, **k: [_playable_fresh_item("42")])

    result = runner.invoke(app, ["refresh-media", "--source", "bookmarks"])
    assert result.exit_code == 0, result.output
    snapshots = list((tmp_path / "data" / "snapshots").glob("*-pre-refresh-media"))
    assert snapshots, "refresh-media must auto-snapshot data/ before writing"


def test_refresh_media_disables_skip_known_with_empty_known_ids(tmp_path: Path, monkeypatch):
    """Backfill must pass an EMPTY known_ids set so the FULL history is scrolled."""
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _poster_era_item("42")}, tmp_path / "data" / "items.json")
    captured: dict[str, object] = {}

    def _spy(context, src, url, known_ids, *a, **k):
        captured["known_ids"] = known_ids
        # Return a re-seen item so the run takes the non-failure path (≥1 known
        # item seen) — the empty-capture guard is exercised separately below.
        return [_playable_fresh_item("42")]

    _mock_browser(monkeypatch, _spy)

    result = runner.invoke(app, ["refresh-media", "--source", "bookmarks"])
    assert result.exit_code == 0, result.output
    assert captured["known_ids"] == set()


def test_refresh_media_blocks_on_empty_capture_against_nonempty_store(tmp_path: Path, monkeypatch):
    """A logged-in-but-empty capture (GraphQL drift / interrupted scroll) must NOT
    report success: 0 known items re-seen on a non-empty store exits non-zero,
    warns loudly, and saves nothing."""
    from xbrain.models import MediaVideoPending
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _poster_era_item("42")}, tmp_path / "data" / "items.json")
    _mock_browser(monkeypatch, lambda *a, **k: [])

    result = runner.invoke(app, ["refresh-media", "--source", "bookmarks"])

    assert result.exit_code != 0
    assert "no se actualizó nada" in result.output
    # No success summary was printed.
    assert "videos updated" not in result.output
    # The store is intact and still poster-era (nothing was overwritten).
    reloaded = load_store(tmp_path / "data" / "items.json")
    video = reloaded["42"].media[1]
    assert isinstance(video, MediaVideoPending)
    assert video.url == "https://pbs.twimg.com/poster.jpg"


def test_refresh_media_force_proceeds_on_empty_capture(tmp_path: Path, monkeypatch):
    """`--force` downgrades the empty-capture guard to a warning and proceeds."""
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _poster_era_item("42")}, tmp_path / "data" / "items.json")
    _mock_browser(monkeypatch, lambda *a, **k: [])

    result = runner.invoke(app, ["refresh-media", "--source", "bookmarks", "--force"])

    assert result.exit_code == 0, result.output
    assert "AVISO" in result.output
    # It proceeds: the END summary is printed (0 re-seen).
    assert "0 known items re-seen" in result.output


def test_refresh_media_empty_store_does_not_warn(tmp_path: Path, monkeypatch):
    """A fresh project (empty store) with an empty capture is a clean no-op."""
    _setup_repo(tmp_path, monkeypatch)  # no items.json saved → empty store
    _mock_browser(monkeypatch, lambda *a, **k: [])

    result = runner.invoke(app, ["refresh-media", "--source", "bookmarks"])

    assert result.exit_code == 0, result.output
    assert "AVISO" not in result.output
    assert "no se actualizó nada" not in result.output


def test_refresh_media_reports_unknown_size_without_zero_gb(tmp_path: Path, monkeypatch):
    """When no video is estimable, the summary must NOT misread as '~0.0 GB'."""
    from xbrain.models import Author, Item, MediaVideoPending

    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _poster_era_item("42")}, tmp_path / "data" / "items.json")

    # A fresh HLS stream is a real upgrade (url != thumbnail) but carries no
    # bitrate/duration, so the size estimate has 0 estimable, 1 unknown.
    hls = Item(
        id="42",
        source="bookmark",
        url="https://x.com/a/status/42",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        media=[
            MediaVideoPending(
                url="https://v/play.m3u8",
                thumbnail_url="https://pbs.twimg.com/poster.jpg",
                bitrate=None,
                duration_millis=None,
            )
        ],
    )
    _mock_browser(monkeypatch, lambda *a, **k: [hls])

    result = runner.invoke(app, ["refresh-media", "--source", "bookmarks"])

    assert result.exit_code == 0, result.output
    assert "size unknown for 1 videos" in result.output
    assert "GB" not in result.output


def test_refresh_media_surfaces_logged_out_runtimeerror(tmp_path: Path, monkeypatch):
    """A logged-out session (extract_source raises RuntimeError) exits 1 cleanly."""
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _poster_era_item("42")}, tmp_path / "data" / "items.json")

    def _raise(*a, **k):
        raise RuntimeError("Sesión de X caducada. Ejecuta `xbrain login`.")

    _mock_browser(monkeypatch, _raise)

    result = runner.invoke(app, ["refresh-media", "--source", "bookmarks"])
    assert result.exit_code == 1
    assert "Sesión de X caducada" in result.output


def test_extract_advances_cursor_to_integer_max_id(tmp_path: Path, monkeypatch):
    """The cursor must track the integer-max id, not list order or string compare."""
    from xbrain.store import load_state

    _setup_repo(tmp_path, monkeypatch)
    # Out-of-order, and "98" > "100" lexicographically — only an int max gives 100.
    _mock_browser(monkeypatch, lambda *a, **k: [_linked_item("98"), _linked_item("100")])

    result = runner.invoke(app, ["extract", "--source", "bookmarks"])

    assert result.exit_code == 0
    assert load_state(tmp_path / "data" / "state.json").bookmarks.last_seen_id == "100"


def test_extract_truncation_persists_nothing_and_exits_nonzero(tmp_path: Path, monkeypatch):
    """A RateLimitTruncated source must not merge items nor advance the cursor."""
    from xbrain.extract.extractor import RateLimitTruncated
    from xbrain.store import load_state, load_store

    _setup_repo(tmp_path, monkeypatch)

    def _truncate(*_a, **_k):
        raise RateLimitTruncated("bookmark: truncado a media timeline")

    _mock_browser(monkeypatch, _truncate)

    result = runner.invoke(app, ["extract", "--source", "bookmarks"])

    assert result.exit_code == 1
    assert load_store(tmp_path / "data" / "items.json") == {}
    assert load_state(tmp_path / "data" / "state.json").bookmarks.last_seen_id is None


# ------------------------------------------------------ list-videos / fetch-video


def _enriched_video_item(item_id: str, topic: str, url: str = _MP4_URL, source: str = "bookmark"):
    from xbrain.models import Enrichment

    item = _video_item(item_id, url=url, source=source)
    item.enriched = Enrichment(
        enriched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        executor="manual",
        primary_topic=topic,
    )
    return item


def test_list_videos_json_is_parseable_array(tmp_path: Path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _enriched_video_item("42", "ai")}, tmp_path / "data" / "items.json")

    result = runner.invoke(app, ["list-videos", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert isinstance(payload, list) and len(payload) == 1
    row = payload[0]
    assert set(row) == {"id", "url", "state", "topic", "size_bytes", "mp4_url", "text"}
    assert row["id"] == "42"
    assert row["state"] == "pending"
    assert row["topic"] == "ai"
    assert row["mp4_url"] == _MP4_URL


def test_list_videos_human_table(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _enriched_video_item("42", "ai")}, tmp_path / "data" / "items.json")

    result = runner.invoke(app, ["list-videos"])
    assert result.exit_code == 0, result.output
    assert "STATE" in result.stdout
    assert "pending" in result.stdout
    assert "ai" in result.stdout


def test_list_videos_is_read_only(tmp_path: Path, monkeypatch):
    """list-videos performs zero writes and takes no snapshot."""
    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _enriched_video_item("42", "ai")}, items_path)
    before = items_path.read_bytes()

    result = runner.invoke(app, ["list-videos", "--json"])
    assert result.exit_code == 0
    assert items_path.read_bytes() == before
    snapshots = tmp_path / "data" / "snapshots"
    assert not snapshots.exists()


def test_list_videos_topic_filter(tmp_path: Path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    save_store(
        {
            "1": _enriched_video_item("1", "ai"),
            "2": _enriched_video_item("2", "climate"),
        },
        tmp_path / "data" / "items.json",
    )

    result = runner.invoke(app, ["list-videos", "--topic", "climate", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert [row["id"] for row in payload] == ["2"]


def test_list_videos_empty_store(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["list-videos"])
    assert result.exit_code == 0
    assert "No hay" in result.stdout


def test_list_videos_rejects_bad_status(tmp_path: Path, monkeypatch):
    """`--status` is a typer Enum: a bad value is a usage error (exit 2), like
    `--source`."""
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["list-videos", "--status", "bogus"])
    assert result.exit_code == 2


def test_list_videos_json_null_topic(tmp_path: Path, monkeypatch):
    """An un-enriched video's topic is JSON null in --json (human table shows —)."""
    import json

    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["list-videos", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload[0]["topic"] is None


def test_list_videos_status_poster_era_enum(tmp_path: Path, monkeypatch):
    """The `poster-era` status value (hyphenated enum) is accepted and filters."""
    import json

    _setup_repo(tmp_path, monkeypatch)
    save_store({"9": _video_item("9", url=_POSTER)}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["list-videos", "--status", "poster-era", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert [row["id"] for row in payload] == ["9"]


def test_resolve_fetch_ids_union_dedup_and_source_asymmetry(tmp_path: Path, monkeypatch):
    """--ids are verbatim (NOT source-scoped); --topic expansion IS source-scoped;
    the union is order-preserving and de-duplicated."""
    from xbrain.cli import _resolve_fetch_ids

    store = {
        "b": _enriched_video_item("b", "ai", source="bookmark"),
        "t": _enriched_video_item("t", "ai", source="own_tweet"),
    }
    # "t" is kept verbatim though it is an own_tweet and source=bookmarks;
    # the topic expansion under bookmarks yields only "b".
    assert _resolve_fetch_ids(store, "t", "ai", "bookmarks") == ["t", "b"]
    # order-preserving dedup across --ids + --topic.
    assert _resolve_fetch_ids(store, "b,b", "ai", "all") == ["b", "t"]


def test_resolve_fetch_ids_requires_selection():
    import pytest

    from xbrain.cli import _resolve_fetch_ids

    with pytest.raises(ValueError, match="ids"):
        _resolve_fetch_ids({}, None, None, "all")


def test_fetch_video_total_failure_exits_nonzero(tmp_path: Path, monkeypatch):
    """A run where every attempted download failed exits 1 (parity with
    download-videos); a pure all-skips run stays exit 0."""
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")
    monkeypatch.setattr("xbrain.video_fetch.requests.Session", _FailVideoSession)
    dest = tmp_path / "out"

    result = runner.invoke(app, ["fetch-video", "--ids", "42", "--to", str(dest)])
    assert result.exit_code == 1
    assert not (dest / "42.mp4").exists()


def test_fetch_video_all_skips_exits_zero(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"7": _video_item("7", url=_HLS_URL)}, tmp_path / "data" / "items.json")
    monkeypatch.setattr("xbrain.video_fetch.requests.Session", _FakeVideoSession)
    dest = tmp_path / "out"

    result = runner.invoke(app, ["fetch-video", "--ids", "7", "--to", str(dest)])
    assert result.exit_code == 0, result.output


def test_fetch_video_downloads_to_dir(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")
    monkeypatch.setattr("xbrain.video_fetch.requests.Session", _FakeVideoSession)
    dest = tmp_path / "out"

    result = runner.invoke(app, ["fetch-video", "--ids", "42", "--to", str(dest)])
    assert result.exit_code == 0, result.output
    assert (dest / "42.mp4").exists()
    assert str(dest / "42.mp4") in result.stdout


def test_fetch_video_does_not_mutate_store(tmp_path: Path, monkeypatch):
    """fetch-video writes only under --to: items.json byte-identical, no snapshot,
    no data/media/ writes."""
    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _video_item("42")}, items_path)
    before = items_path.read_bytes()
    monkeypatch.setattr("xbrain.video_fetch.requests.Session", _FakeVideoSession)
    dest = tmp_path / "out"

    result = runner.invoke(app, ["fetch-video", "--ids", "42", "--to", str(dest)])
    assert result.exit_code == 0, result.output
    assert items_path.read_bytes() == before
    assert not (tmp_path / "data" / "media").exists()
    snapshots = tmp_path / "data" / "snapshots"
    assert not snapshots.exists() or not any("fetch-video" in p.name for p in snapshots.iterdir())


def test_fetch_video_json_output(tmp_path: Path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")
    monkeypatch.setattr("xbrain.video_fetch.requests.Session", _FakeVideoSession)
    dest = tmp_path / "out"

    result = runner.invoke(app, ["fetch-video", "--ids", "42", "--to", str(dest), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload[0]["id"] == "42"
    assert payload[0]["outcome"] == "fetched"
    assert payload[0]["path"].endswith("42.mp4")


def test_fetch_video_topic_selection(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store(
        {
            "1": _enriched_video_item("1", "ai"),
            "2": _enriched_video_item("2", "climate"),
        },
        tmp_path / "data" / "items.json",
    )
    monkeypatch.setattr("xbrain.video_fetch.requests.Session", _FakeVideoSession)
    dest = tmp_path / "out"

    result = runner.invoke(app, ["fetch-video", "--topic", "ai", "--to", str(dest)])
    assert result.exit_code == 0, result.output
    assert (dest / "1.mp4").exists()
    assert not (dest / "2.mp4").exists()


def test_fetch_video_requires_ids_or_topic(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")
    dest = tmp_path / "out"

    result = runner.invoke(app, ["fetch-video", "--to", str(dest)])
    assert result.exit_code == 1
    assert "ids" in result.output or "topic" in result.output


def test_fetch_video_requires_to(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["fetch-video", "--ids", "42"])
    assert result.exit_code != 0


def test_fetch_video_skips_hls(tmp_path: Path, monkeypatch):
    import json

    _setup_repo(tmp_path, monkeypatch)
    save_store({"7": _video_item("7", url=_HLS_URL)}, tmp_path / "data" / "items.json")
    monkeypatch.setattr("xbrain.video_fetch.requests.Session", _FakeVideoSession)
    dest = tmp_path / "out"

    result = runner.invoke(app, ["fetch-video", "--ids", "7", "--to", str(dest), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload[0]["outcome"] == "skipped"
    assert payload[0]["reason"] == "hls"
    assert not (dest / "7.mp4").exists()


# ------------------------------------------------------ digest-video

# A distinct signed URL for the SAME amplify_video id as _AMPLIFY_URL_1 — proves
# dedup keys on the stable path id, not the rotating URL.
_AMPLIFY_URL_1 = "https://video.twimg.com/amplify_video/900/vid/720/a.mp4?tag=16"
_AMPLIFY_URL_2 = "https://video.twimg.com/amplify_video/900/vid/1080/b.mp4?tag=21"


def _wire_digest(monkeypatch, transcript, *, calls: list | None = None):
    """Mock the network (fetch) + subprocess (transcribe) + throttle for digest-video.

    `fetch` returns mp4 bytes (via the shared `_FakeVideoSession`); the external
    transcriber is replaced with a fake returning `transcript` (or, if callable,
    `transcript(path)`). No real network, subprocess, or download runs.
    """
    from xbrain.transcribe import Transcript

    monkeypatch.setattr("xbrain.video_fetch.requests.Session", _FakeVideoSession)
    monkeypatch.setattr("xbrain.video_fetch.time.sleep", lambda *_a, **_k: None)

    def _fake_transcribe(path, **_kwargs):
        if calls is not None:
            calls.append(str(path))
        return transcript(path) if callable(transcript) else transcript

    monkeypatch.setattr("xbrain.cli.transcribe_media", _fake_transcribe)
    return Transcript


def _speech_transcript(text="a talk transcript"):
    from xbrain.transcribe import Segment, Transcript

    return Transcript(text=text, segments=[Segment(0.0, 1.0, text)], language="en", has_speech=True)


def _silent_transcript():
    from xbrain.transcribe import Transcript

    return Transcript(text="", segments=[], language=None, has_speech=False)


def test_digest_video_transcribes_and_attaches(tmp_path: Path, monkeypatch):
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _video_item("42", url=_AMPLIFY_URL_1)}, items_path)
    _wire_digest(monkeypatch, _speech_transcript("hello from the talk"))

    result = runner.invoke(app, ["digest-video", "--ids", "42"])
    assert result.exit_code == 0, result.output
    store = load_store(items_path)
    sources = store["42"].content.sources
    assert sources[0].kind == "x_video"
    assert sources[0].text == "hello from the talk"
    assert sources[0].has_speech is True


def test_digest_video_dedups_same_video(tmp_path: Path, monkeypatch):
    """Two items bookmarking the same video → transcribed ONCE, both carry it."""
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store(
        {
            "a": _video_item("a", url=_AMPLIFY_URL_1),
            "b": _video_item("b", url=_AMPLIFY_URL_2),
        },
        items_path,
    )
    calls: list = []
    _wire_digest(monkeypatch, _speech_transcript("shared"), calls=calls)

    result = runner.invoke(app, ["digest-video", "--ids", "a,b"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1  # transcribed once
    store = load_store(items_path)
    assert store["a"].content.sources[0].text == "shared"
    assert store["b"].content.sources[0].text == "shared"
    assert "1 items ← 1 vídeos" not in result.stdout  # 2 items, 1 video
    assert "2 items ← 1 vídeos" in result.stdout


def test_digest_video_no_speech_summary(tmp_path: Path, monkeypatch):
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _video_item("42", url=_AMPLIFY_URL_1)}, items_path)
    _wire_digest(monkeypatch, _silent_transcript())

    result = runner.invoke(app, ["digest-video", "--ids", "42"])
    assert result.exit_code == 0, result.output
    assert "sin voz 1" in result.stdout
    store = load_store(items_path)
    assert store["42"].content.sources[0].has_speech is False


def test_digest_video_idempotent(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _video_item("42", url=_AMPLIFY_URL_1)}, items_path)
    calls: list = []
    _wire_digest(monkeypatch, _speech_transcript(), calls=calls)

    first = runner.invoke(app, ["digest-video", "--ids", "42"])
    assert first.exit_code == 0, first.output
    second = runner.invoke(app, ["digest-video", "--ids", "42"])
    assert second.exit_code == 0, second.output
    assert "ya digeridos 1" in second.stdout
    assert len(calls) == 1  # the idempotent re-run did NOT re-transcribe


def test_digest_video_force_redigests(tmp_path: Path, monkeypatch):
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _video_item("42", url=_AMPLIFY_URL_1)}, items_path)

    _wire_digest(monkeypatch, _speech_transcript("v1"))
    runner.invoke(app, ["digest-video", "--ids", "42"])
    _wire_digest(monkeypatch, _speech_transcript("v2"))
    result = runner.invoke(app, ["digest-video", "--ids", "42", "--force"])
    assert result.exit_code == 0, result.output
    store = load_store(items_path)
    assert store["42"].content.sources[0].text == "v2"


def test_digest_video_all_pending_selects_pending(tmp_path: Path, monkeypatch):
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _video_item("42", url=_AMPLIFY_URL_1)}, items_path)
    _wire_digest(monkeypatch, _speech_transcript())

    result = runner.invoke(app, ["digest-video", "--all-pending"])
    assert result.exit_code == 0, result.output
    store = load_store(items_path)
    assert store["42"].content.sources[0].kind == "x_video"


def test_digest_video_requires_selection(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"42": _video_item("42")}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["digest-video"])
    assert result.exit_code == 1
    assert "ids" in result.output or "all-pending" in result.output


def test_digest_video_missing_transcriber_exits_nonzero(tmp_path: Path, monkeypatch):
    """A missing external transcriber is a clean operator error (exit 1), not a
    crash — and nothing is written to items.json."""
    from xbrain.store import load_store
    from xbrain.transcribe import TranscriberNotFound

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _video_item("42", url=_AMPLIFY_URL_1)}, items_path)
    monkeypatch.setattr("xbrain.video_fetch.requests.Session", _FakeVideoSession)
    monkeypatch.setattr("xbrain.video_fetch.time.sleep", lambda *_a, **_k: None)

    def _boom(_path, **_kwargs):
        raise TranscriberNotFound("parakeet-mlx not found — set [transcribe].command")

    monkeypatch.setattr("xbrain.cli.transcribe_media", _boom)
    result = runner.invoke(app, ["digest-video", "--ids", "42"])
    assert result.exit_code == 1
    assert "parakeet-mlx" in result.output
    assert load_store(items_path)["42"].content is None  # nothing persisted


# A SECOND, DISTINCT video (different ext_tw_video id) for limit/selection tests.
_DISTINCT_VIDEO_URL = "https://video.twimg.com/ext_tw_video/7700/vid/720/z.mp4?tag=3"


def test_digest_video_limit_caps_items(tmp_path: Path, monkeypatch):
    """`--limit 1` over a 2-distinct-video store transcribes exactly one item."""
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store(
        {
            "a": _video_item("a", url=_AMPLIFY_URL_1),
            "b": _video_item("b", url=_DISTINCT_VIDEO_URL),
        },
        items_path,
    )
    _wire_digest(monkeypatch, _speech_transcript())

    result = runner.invoke(app, ["digest-video", "--ids", "a,b", "--limit", "1"])
    assert result.exit_code == 0, result.output
    assert "transcritos 1" in result.stdout
    store = load_store(items_path)
    digested = [i for i in ("a", "b") if store[i].content is not None]
    assert len(digested) == 1  # exactly one, not both


def test_digest_video_topic_selects_only_matching(tmp_path: Path, monkeypatch):
    """`--topic ai` digests only the ai-topic video, honoring the catalog filter."""
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store(
        {
            "1": _enriched_video_item("1", "ai", url=_AMPLIFY_URL_1),
            "2": _enriched_video_item("2", "climate", url=_DISTINCT_VIDEO_URL),
        },
        items_path,
    )
    _wire_digest(monkeypatch, _speech_transcript())

    result = runner.invoke(app, ["digest-video", "--topic", "ai"])
    assert result.exit_code == 0, result.output
    store = load_store(items_path)
    assert store["1"].content is not None  # ai video digested
    assert store["2"].content is None  # climate video untouched


# ------------------------------------------------------ digest-video --frames (PR4)


def _setup_repo_with_vision(tmp_path: Path, monkeypatch, command: str = "vlm-describe") -> Path:
    """`_setup_repo` + a configured external `[vision].command` so `--frames` runs."""
    vault = _setup_repo(tmp_path, monkeypatch)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        cfg.read_text(encoding="utf-8") + f'[vision]\ncommand = "{command}"\n', encoding="utf-8"
    )
    return vault


def _write_slide_png(path: Path) -> None:
    """A high-edge (text/line) image so the REAL `classify_visual` reads 'slides'."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(img)
    for y in range(40, 320, 24):
        draw.line([(40, y), (600, y)], fill="black", width=3)
    draw.rectangle([300, 200, 560, 330], outline="black", width=4)
    img.save(path)


def _write_photo_png(path: Path) -> None:
    """A low-edge (smooth gradient) image so the REAL `classify_visual` reads
    'talking_head' — a camera frame / bokeh, not a slide."""
    from PIL import Image

    img = Image.new("L", (640, 360))
    for x in range(640):
        for y in range(360):
            img.putpixel((x, y), int((x / 640) * 200 + (y / 360) * 40))
    img.convert("RGB").save(path)


def _wire_frames(monkeypatch, *, describe_calls: list | None = None, writer=_write_slide_png):
    """Mock ffmpeg extraction (real PNGs) + the external vision subprocess.

    `writer` paints each fake frame — `_write_slide_png` (default, high-edge →
    'slides') or `_write_photo_png` (low-edge → 'talking_head'). The REAL
    `classify_visual` runs on the produced images, so the CLI's slide-vs-
    talking-head decision is exercised end-to-end."""
    from xbrain.video_frames import KeyFrame

    def _fake_extract(path, **_kwargs):
        frames_dir = Path(path).parent / "xbrain-frames-fake"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for index in range(2):
            frame_path = frames_dir / f"frame-{index:05d}.png"
            writer(frame_path)
            frames.append(KeyFrame(timestamp=float(index * 10), path=frame_path))
        return frames

    def _fake_describe(path, **_kwargs):
        if describe_calls is not None:
            describe_calls.append(str(path))
        return f"slide {Path(path).stem}"

    monkeypatch.setattr("xbrain.cli.extract_key_frames", _fake_extract)
    monkeypatch.setattr("xbrain.cli.describe_image", _fake_describe)


def test_digest_video_frames_requires_vision_command(tmp_path: Path, monkeypatch):
    """`--frames` with no `[vision].command` configured is a clear operator error
    (exit 1) — there is no bundled default — and nothing is persisted."""
    from xbrain.store import load_store

    _setup_repo(tmp_path, monkeypatch)  # no [vision] section
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _video_item("42", url=_AMPLIFY_URL_1)}, items_path)
    _wire_digest(monkeypatch, _speech_transcript())

    result = runner.invoke(app, ["digest-video", "--ids", "42", "--frames"])
    assert result.exit_code == 1
    assert "vision" in result.output.lower()
    assert load_store(items_path)["42"].content is None  # nothing persisted


def test_digest_video_frames_describes_and_persists_slides(tmp_path: Path, monkeypatch):
    """`--frames` on a slide video: the frames are described (external vision) and
    recorded on the `x_video` source, and the slide images are persisted under
    `data/media/<id>/frames/` for the generator — the summary reports the layer."""
    from xbrain.store import load_store

    _setup_repo_with_vision(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _video_item("42", url=_AMPLIFY_URL_1)}, items_path)
    _wire_digest(monkeypatch, _speech_transcript("the talk"))
    describe_calls: list = []
    _wire_frames(monkeypatch, describe_calls=describe_calls)

    result = runner.invoke(app, ["digest-video", "--ids", "42", "--frames"])
    assert result.exit_code == 0, result.output
    store = load_store(items_path)
    frames = store["42"].content.sources[0].frames
    assert len(frames) == 2
    assert frames[0].local_path == "42/frames/0.png"
    assert frames[0].description == "slide frame-00000"
    assert len(describe_calls) == 2  # vision ran on both slides
    assert (tmp_path / "data" / "media" / "42" / "frames" / "0.png").exists()
    assert "con slides" in result.stdout  # the visual segment of the summary


def test_digest_video_vision_model_override(tmp_path: Path, monkeypatch):
    """`--vision-model NAME` overrides `[vision].model` for the run — NAME is what
    reaches the vision command as `--model`, so a multi-backend wrapper can route
    it (e.g. `opus` → cloud, `qwen-7b` → local)."""
    _setup_repo_with_vision(tmp_path, monkeypatch)  # [vision].command set, model unset
    save_store({"42": _video_item("42", url=_AMPLIFY_URL_1)}, tmp_path / "data" / "items.json")
    _wire_digest(monkeypatch, _speech_transcript("the talk"))
    _wire_frames(monkeypatch)  # extract → slide frames; real classify_visual runs
    seen: list = []

    def _capture(path, *, command, model):
        seen.append(model)
        return "slide"

    monkeypatch.setattr("xbrain.cli.describe_image", _capture)
    result = runner.invoke(
        app, ["digest-video", "--ids", "42", "--frames", "--vision-model", "opus"]
    )
    assert result.exit_code == 0, result.output
    assert seen and set(seen) == {"opus"}  # override reached the vision command


def test_digest_video_vision_model_without_frames_errors(tmp_path: Path, monkeypatch):
    """`--vision-model` without `--frames` is a hard error, not a silent no-op."""
    import re

    _setup_repo_with_vision(tmp_path, monkeypatch)
    save_store({"42": _video_item("42", url=_AMPLIFY_URL_1)}, tmp_path / "data" / "items.json")
    result = runner.invoke(app, ["digest-video", "--ids", "42", "--vision-model", "opus"])
    assert result.exit_code == 2  # click usage error (BadParameter), robust across widths
    # Strip ANSI + collapse Rich's width-dependent wrapping before matching the message.
    plain = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", result.output).split())
    assert "--frames" in plain


def test_digest_video_vision_model_defaults_to_config(tmp_path: Path, monkeypatch):
    """With no `--vision-model`, the run uses `[vision].model` from config."""
    _setup_repo(tmp_path, monkeypatch)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        cfg.read_text(encoding="utf-8") + '[vision]\ncommand = "vlm"\nmodel = "qwen-7b"\n',
        encoding="utf-8",
    )
    save_store({"42": _video_item("42", url=_AMPLIFY_URL_1)}, tmp_path / "data" / "items.json")
    _wire_digest(monkeypatch, _speech_transcript("the talk"))
    _wire_frames(monkeypatch)
    seen: list = []

    def _capture(path, *, command, model):
        seen.append(model)
        return "slide"

    monkeypatch.setattr("xbrain.cli.describe_image", _capture)
    result = runner.invoke(app, ["digest-video", "--ids", "42", "--frames"])
    assert result.exit_code == 0, result.output
    assert seen and set(seen) == {"qwen-7b"}  # config model, no override


def test_digest_video_frames_then_generate_embeds_slides(tmp_path: Path, monkeypatch):
    """End-to-end #44 PR4 success criterion: a slide talk digested with `--frames`
    then generated embeds its key slides into the note (mirrored into `_media/`)."""
    vault = _setup_repo_with_vision(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _enriched_video_item("42", "ai", url=_AMPLIFY_URL_1)}, items_path)
    _wire_digest(monkeypatch, _speech_transcript("the talk body"))
    _wire_frames(monkeypatch)

    assert runner.invoke(app, ["digest-video", "--ids", "42", "--frames"]).exit_code == 0
    assert runner.invoke(app, ["generate"]).exit_code == 0

    note = next((vault / "x-knowledge" / "items").glob("*42*.md")).read_text(encoding="utf-8")
    assert "## Video digest" in note
    assert "![[_media/42/frames/0.png]]" in note
    assert "slide frame-00000" in note  # the vision caption
    assert (vault / "x-knowledge" / "_media" / "42" / "frames" / "0.png").exists()


def test_digest_video_without_frames_attaches_no_slides(tmp_path: Path, monkeypatch):
    """Opt-in: a normal `digest-video` (no `--frames`) never invokes ffmpeg/vision
    and attaches no frames — even with a `[vision].command` configured."""
    from xbrain.store import load_store

    _setup_repo_with_vision(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _video_item("42", url=_AMPLIFY_URL_1)}, items_path)
    _wire_digest(monkeypatch, _speech_transcript())
    describe_calls: list = []
    _wire_frames(monkeypatch, describe_calls=describe_calls)

    result = runner.invoke(app, ["digest-video", "--ids", "42"])  # no --frames
    assert result.exit_code == 0, result.output
    assert describe_calls == []  # vision NOT invoked
    assert load_store(items_path)["42"].content.sources[0].frames == []
    assert "Visual:" not in result.stdout  # summary unchanged


def test_digest_video_frames_talking_head_skips_and_embeds_nothing(tmp_path: Path, monkeypatch):
    """End-to-end #44 PR4 success criterion (the Ng-interview direction): a
    talking-head video digested with `--frames` embeds NO slides, wastes NO vision
    call, and the summary reports the talking-head skip — the mirror image of the
    slide-talk test. The REAL `classify_visual` runs on genuine low-edge frames."""
    from xbrain.store import load_store

    vault = _setup_repo_with_vision(tmp_path, monkeypatch)
    items_path = tmp_path / "data" / "items.json"
    save_store({"42": _enriched_video_item("42", "ai", url=_AMPLIFY_URL_1)}, items_path)
    _wire_digest(monkeypatch, _speech_transcript("interview body"))
    describe_calls: list = []
    _wire_frames(monkeypatch, describe_calls=describe_calls, writer=_write_photo_png)

    result = runner.invoke(app, ["digest-video", "--ids", "42", "--frames"])
    assert result.exit_code == 0, result.output
    assert describe_calls == []  # NO vision call wasted on a talking-head
    assert load_store(items_path)["42"].content.sources[0].frames == []  # nothing attached
    assert "talking-head (saltados)" in result.stdout  # the skip is surfaced

    assert runner.invoke(app, ["generate"]).exit_code == 0
    note = next((vault / "x-knowledge" / "items").glob("*42*.md")).read_text(encoding="utf-8")
    assert "## Video digest" in note  # the transcript digest still renders
    assert "_media/42/frames" not in note  # but NO slide embed
    assert not (vault / "x-knowledge" / "_media" / "42" / "frames").exists()
