# tests/test_jev_run.py
"""`jev.run.run_topics`: ONE paid pass, callable by the CLI today and by a local server later.

It prints nothing (a server has no terminal); it reports through hooks and its return value.
The CLI's own behaviour on top of it is covered in `tests/test_jev_cli.py`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.jev_fakes import FakeJevClient
from xbrain.config import Config, load_config
from xbrain.jev.assess import RunResult, select_items
from xbrain.jev.client import JevError
from xbrain.jev.models import JevRun, TopicAssessment
from xbrain.jev.run import RunOutcome, run_topics
from xbrain.jev.store import load_assessments, load_runs
from xbrain.models import Author, Enrichment, Item, Topic
from xbrain.rubrics import save_vocab
from xbrain.store import save_store

DT = datetime(2026, 9, 22, tzinfo=timezone.utc)
VOCAB = [
    Topic(slug="ai-coding", description="Construir software con IA."),
    Topic(slug="startups", description="Fundar empresas."),
]


def _item(item_id: str, text: str) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=text,
        created_at=DT,
        captured_at=DT,
        enriched=Enrichment(
            enriched_at=DT,
            executor="claude-code",
            summary="s",
            primary_topic="ai-coding",
            topics=["ai-coding"],
        ),
    )


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch) -> Config:
    vault = tmp_path / "vault"
    vault.mkdir()
    (tmp_path / "config.toml").write_text(
        f'[paths]\nvault = "{vault}"\noutput_subdir = "x"\ndata_dir = "data"\n'
        '[x]\nhandle = "v"\n[jev]\nconcurrency = 1\n',
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    save_store(
        {"1": _item("1", "Claude Code hooks"), "2": _item("2", "Seed round tips")},
        tmp_path / "data" / "items.json",
    )
    save_vocab(VOCAB, tmp_path / "data" / "vocab.yaml")
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    return load_config(tmp_path)


def _pass(cfg: Config, client: FakeJevClient, *, force: bool = False, **hooks) -> RunOutcome:
    from xbrain.store import load_store

    assessments = load_assessments(cfg.jev_topics_path)
    selection = select_items(
        load_store(cfg.items_path),
        assessments,
        VOCAB,
        ids=[],
        limit=None,
        force=force,
        fallback=cfg.jev_fallback_option,
        char_limit=cfg.jev_state_char_limit,
    )
    return run_topics(cfg, selection, assessments, VOCAB, lambda: client, **hooks)


def test_a_pass_saves_logs_and_returns_what_it_did(cfg: Config):
    client = FakeJevClient()

    outcome = _pass(cfg, client)

    assert [a.item_id for a in outcome.assessed] == ["1", "2"]
    assert outcome.failed == () and outcome.interrupted is False
    assert outcome.stored == 2
    assert set(load_assessments(cfg.jev_topics_path)) == {"1", "2"}
    assert outcome.logged == load_runs(cfg.jev_runs_path)[0]
    assert client.closed is True


def test_the_summary_hook_runs_before_the_save_and_the_log_hook_after(cfg: Config, monkeypatch):
    """The counters must reach the operator even when the save then fails."""
    events: list[str] = []
    from xbrain.jev import run as jev_run

    real_save = jev_run.save_assessments

    def _save(assessments, path):
        events.append("save")
        real_save(assessments, path)

    monkeypatch.setattr(jev_run, "save_assessments", _save)

    def _summary(result: RunResult) -> None:
        events.append("summary")

    def _logged(run: JevRun, path: Path, error: OSError | None) -> None:
        events.append("logged")

    _pass(cfg, FakeJevClient(), on_summary=_summary, on_logged=_logged)

    assert events == ["summary", "save", "logged"]


def test_an_interrupt_is_returned_not_raised_after_the_banked_records_are_saved(cfg: Config):
    """A server has no shell to exit 130 to: the pass says it was interrupted and the caller
    decides. What was answered is on disk and in the log either way."""
    banked: list[tuple[int, int]] = []

    outcome = _pass(
        cfg,
        FakeJevClient(interrupt_after=1),
        on_interrupted=lambda records, stored: banked.append((len(records), stored)),
    )

    assert outcome.interrupted is True
    assert [a.item_id for a in outcome.assessed] == ["1"]
    assert banked == [(1, 1)]
    assert list(load_assessments(cfg.jev_topics_path)) == ["1"]
    assert load_runs(cfg.jev_runs_path)[0].interrupted is True


def test_a_pass_where_every_call_failed_raises_after_logging(cfg: Config):
    with pytest.raises(JevError, match="ninguna de las 2"):
        _pass(cfg, FakeJevClient(fail_when=lambda state: True))

    [run] = load_runs(cfg.jev_runs_path)
    assert (run.requests, run.failed) == (2, 2)


def test_force_backs_up_the_side_car_before_the_client_is_built(cfg: Config):
    _pass(cfg, FakeJevClient())
    backups: list[Path] = []
    built: list[bool] = []

    def _client() -> FakeJevClient:
        built.append(bool(backups))  # was the copy already made when the client was built?
        return FakeJevClient()

    from xbrain.store import load_store

    assessments = load_assessments(cfg.jev_topics_path)
    selection = select_items(
        load_store(cfg.items_path),
        assessments,
        VOCAB,
        ids=[],
        limit=None,
        force=True,
        fallback=cfg.jev_fallback_option,
        char_limit=cfg.jev_state_char_limit,
    )
    run_topics(cfg, selection, assessments, VOCAB, _client, on_backup=backups.append)

    assert len(backups) == 1 and backups[0].exists()
    assert built == [True]


def test_run_topics_prints_nothing(cfg: Config, capsys):
    """A server calls this; stdout is not its channel."""
    _pass(cfg, FakeJevClient())

    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_the_assessments_map_handed_in_is_updated_in_place(cfg: Config):
    """The caller's map is the side-car in memory: after the pass it holds the new records,
    so a caller that keeps running (a server) does not have to reload the file."""
    from xbrain.store import load_store

    assessments: dict[str, TopicAssessment] = {}
    selection = select_items(
        load_store(cfg.items_path),
        assessments,
        VOCAB,
        ids=[],
        limit=None,
        force=False,
        fallback=cfg.jev_fallback_option,
        char_limit=cfg.jev_state_char_limit,
    )

    run_topics(cfg, selection, assessments, VOCAB, FakeJevClient)

    assert set(assessments) == {"1", "2"}
