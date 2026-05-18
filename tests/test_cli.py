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
    result = runner.invoke(app, ["vocab"])
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

    def _fake_synth(inputs, model, **kwargs):
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

    def _fake_synth(inputs, model, **kwargs):
        return [OverviewJudgment(slug="misc", overview="Re-sintetizado.", notes=[])]

    monkeypatch.setattr(cli, "synthesize_overviews_api", _fake_synth)
    result = runner.invoke(app, ["topics", "--resynth", "--executor", "api"])
    assert result.exit_code == 0
    page = (tmp_path / "vault" / "x-knowledge" / "topics" / "misc.md").read_text(encoding="utf-8")
    assert "Re-sintetizado." in page
