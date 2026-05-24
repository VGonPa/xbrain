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
    """A run where every download fails surfaces as exit-1 (mirrors #24)."""
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
    assert "Failed photos" in result.output
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


def test_parse_date_returns_utc_aware():
    from xbrain.cli import _parse_date

    parsed = _parse_date("2025-01-01")
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert _parse_date(None) is None


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
