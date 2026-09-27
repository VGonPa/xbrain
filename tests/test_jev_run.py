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
from xbrain.jev.lock import pass_lock
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
    with pass_lock(cfg.jev_lock_path, "test") as lock:
        return run_topics(cfg, selection, assessments, VOCAB, lambda: client, lock=lock, **hooks)


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

    def _logged(path: Path, line: str, error: BaseException | None) -> None:
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
    with pass_lock(cfg.jev_lock_path, "test") as lock:
        run_topics(cfg, selection, assessments, VOCAB, _client, lock=lock, on_backup=backups.append)

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

    with pass_lock(cfg.jev_lock_path, "test") as lock:
        run_topics(cfg, selection, assessments, VOCAB, FakeJevClient, lock=lock)

    assert set(assessments) == {"1", "2"}


# --------------------------------------------------------------- every exit path, at the loop
#
# The run log's exit paths are asserted HERE, on `run_topics`, where the logic lives; the CLI
# keeps one smoke test per path (`tests/test_jev_cli.py`, "the run log" section).


def _only_run(cfg: Config) -> JevRun:
    [run] = load_runs(cfg.jev_runs_path)
    return run


def test_success_logs_every_answer_at_the_seam(cfg: Config):
    _pass(cfg, FakeJevClient(input_tokens=1_500))

    run = _only_run(cfg)
    assert (run.requests, run.ok, run.failed, run.unsaved, run.interrupted) == (2, 2, 0, 0, False)
    assert run.input_tokens_by_provider == {"fake": 3_000} and run.models == ["jev-1.13.0"]
    assert run.kind == "topics"


def test_a_partial_failure_logs_the_failed_call(cfg: Config):
    _pass(cfg, FakeJevClient(fail_when=lambda state: "Seed" in state["post"]))

    assert (_only_run(cfg).ok, _only_run(cfg).failed) == (1, 1)


def test_refused_answers_are_failures_whose_tokens_are_still_logged(cfg: Config):
    """Jev answered — and billed — but with a primary outside the options, so xbrain refused
    every answer. The pass fails as a whole, and the log still carries the tokens."""
    with pytest.raises(JevError, match="ninguna de las 2"):
        _pass(cfg, FakeJevClient(primary="banana"))

    run = _only_run(cfg)
    assert (run.requests, run.ok, run.failed) == (2, 0, 2)
    assert run.input_tokens_by_provider == {"fake": 200}
    assert run.models == ["jev-1.13.0"]


def test_an_interrupt_logs_answers_it_never_saved_as_unsaved_not_failed(cfg: Config, monkeypatch):
    """Two calls answered, one drained into the side-car, then Ctrl-C: the undrained answer
    was billed and not kept. It is `unsaved`, never `failed`, and its tokens are logged."""
    from xbrain.jev import run as jev_run

    def _answer_twice_keep_one(items, vocab, client, *, on_result, **kwargs):
        from xbrain.jev.assess import assess_topics
        from xbrain.jev.questions import build_topic_questions

        questions = build_topic_questions(vocab, "otro")
        kept = assess_topics(items[0], questions, client, char_limit=100_000)
        assess_topics(items[1], questions, client, char_limit=100_000)  # answered, not drained
        on_result(kept)
        raise KeyboardInterrupt

    monkeypatch.setattr(jev_run, "run_assessments", _answer_twice_keep_one)

    outcome = _pass(cfg, FakeJevClient())

    assert outcome.interrupted is True
    run = _only_run(cfg)
    assert (run.requests, run.ok, run.failed, run.unsaved) == (2, 1, 0, 1)
    assert run.input_tokens == 200


def test_an_interrupt_before_any_call_logs_nothing(cfg: Config, monkeypatch):
    from xbrain.jev import run as jev_run

    def _interrupted_at_once(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(jev_run, "run_assessments", _interrupted_at_once)

    outcome = _pass(cfg, FakeJevClient())

    assert outcome.interrupted is True and outcome.logged is None
    assert not cfg.jev_runs_path.exists()


def test_a_side_car_that_cannot_be_written_still_logs_the_pass(cfg: Config, monkeypatch):
    from xbrain.jev import run as jev_run

    def _boom(assessments, path):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(jev_run, "save_assessments", _boom)

    with pytest.raises(JevError, match="2 evaluaciones pagadas sin guardar"):
        _pass(cfg, FakeJevClient())

    run = _only_run(cfg)
    assert (run.requests, run.ok, run.unsaved) == (2, 0, 2)  # billed, never on disk


# --------------------------------------------------------------- the log step is never the verdict


def test_a_clock_that_steps_back_keeps_the_interrupt_and_the_teardown(cfg: Config, monkeypatch):
    """`finished_at < started_at` would fail validation inside `finally` and replace the
    interrupt. The finish is clamped to the start instead."""
    from xbrain.jev import run as jev_run

    ticks = iter(
        [
            datetime(2026, 9, 26, 12, tzinfo=timezone.utc),
            datetime(2026, 9, 26, 11, tzinfo=timezone.utc),
        ]
    )
    monkeypatch.setattr(jev_run, "_now", lambda: next(ticks))
    client = FakeJevClient(interrupt_after=1)

    outcome = _pass(cfg, client)

    assert outcome.interrupted is True and client.closed is True
    run = _only_run(cfg)
    assert run.finished_at == run.started_at


def test_a_log_that_cannot_be_appended_keeps_the_interrupt(cfg: Config, monkeypatch):
    from xbrain.jev import run as jev_run

    def _boom(run, path):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(jev_run, "append_run", _boom)
    seen: list[BaseException | None] = []
    client = FakeJevClient(interrupt_after=1)

    outcome = _pass(cfg, client, on_logged=lambda path, line, error: seen.append(error))

    assert outcome.interrupted is True and client.closed is True
    assert isinstance(seen[0], OSError)
    assert list(load_assessments(cfg.jev_topics_path)) == ["1"]


def test_a_log_that_cannot_be_appended_keeps_the_all_failed_error(cfg: Config, monkeypatch):
    from xbrain.jev import run as jev_run

    def _boom(run, path):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(jev_run, "append_run", _boom)

    with pytest.raises(JevError, match="ninguna de las 2"):
        _pass(cfg, FakeJevClient(fail_when=lambda state: True))


def test_a_log_hook_that_raises_never_replaces_the_verdict(cfg: Config, caplog):
    """`xbrain jev topics | head` closes the pipe under the echo in the hook."""

    def _broken_pipe(path, line, error):
        raise BrokenPipeError(32, "Broken pipe")

    client = FakeJevClient()
    outcome = _pass(cfg, client, on_logged=_broken_pipe)

    assert outcome.interrupted is False and client.closed is True
    assert len(load_runs(cfg.jev_runs_path)) == 1
    assert "registro de pasadas" in caplog.text

    with pytest.raises(JevError, match="ninguna de las 2"):
        _pass(cfg, FakeJevClient(fail_when=lambda state: True), force=True, on_logged=_broken_pipe)


def test_the_logged_line_is_handed_to_the_hook_as_json(cfg: Config):
    lines: list[str] = []

    _pass(cfg, FakeJevClient(), on_logged=lambda path, line, error: lines.append(line))

    assert JevRun.model_validate_json(lines[0]) == _only_run(cfg)


# --------------------------------------------------------------- hooks are never the verdict
#
# PR 8 re-review Minor 2: `on_summary` and `on_interrupted` ran unguarded BEFORE the save, so
# `xbrain jev topics | head` closing the pipe under their echo threw away up to 24 paid
# records that had not reached a checkpoint. They are display, like `on_progress`.


def test_a_summary_hook_that_raises_still_saves_every_paid_record(cfg: Config, caplog):
    def _broken_pipe(result: RunResult) -> None:
        raise BrokenPipeError(32, "Broken pipe")

    outcome = _pass(cfg, FakeJevClient(), on_summary=_broken_pipe)

    assert set(load_assessments(cfg.jev_topics_path)) == {"1", "2"}
    assert [a.item_id for a in outcome.assessed] == ["1", "2"]
    assert "on_summary" in caplog.text and "Broken pipe" in caplog.text


def test_an_interrupt_hook_that_raises_still_saves_the_banked_records(cfg: Config, caplog):
    def _broken_pipe(banked, stored) -> None:
        raise BrokenPipeError(32, "Broken pipe")

    outcome = _pass(cfg, FakeJevClient(interrupt_after=1), on_interrupted=_broken_pipe)

    assert outcome.interrupted is True
    assert list(load_assessments(cfg.jev_topics_path)) == ["1"]
    assert "on_interrupted" in caplog.text


def test_a_backup_hook_that_raises_does_not_stop_the_pass(cfg: Config, caplog):
    _pass(cfg, FakeJevClient())

    def _broken_pipe(path: Path) -> None:
        raise BrokenPipeError(32, "Broken pipe")

    outcome = _pass(cfg, FakeJevClient(), force=True, on_backup=_broken_pipe)

    assert [a.item_id for a in outcome.assessed] == ["1", "2"]
    assert len(list(cfg.jev_dir.glob("topics.*.bak"))) == 1
    assert "on_backup" in caplog.text


# --------------------------------------------------------------- the pass lock
#
# PR 8 arch M7: two passes at once (a terminal and the local server) each load the side-car,
# each save their own map, and the second save drops the first pass's paid records. The lock
# covers LOAD → SAVE, so it is the caller's to take; the loop refuses to run without it.


def _selection(cfg: Config, assessments: dict[str, TopicAssessment], *, force: bool = False):
    from xbrain.store import load_store

    return select_items(
        load_store(cfg.items_path),
        assessments,
        VOCAB,
        ids=[],
        limit=None,
        force=force,
        fallback=cfg.jev_fallback_option,
        char_limit=cfg.jev_state_char_limit,
    )


def test_a_pass_without_the_lock_is_refused_before_any_client_is_built(cfg: Config):
    built: list[bool] = []

    def _client() -> FakeJevClient:
        built.append(True)
        return FakeJevClient()

    with pass_lock(cfg.jev_lock_path, "test") as released:
        pass
    with pass_lock(cfg.jev_lock_path.with_name("other.lock"), "test") as elsewhere:
        with pytest.raises(JevError, match=r"\.lock"):
            run_topics(cfg, _selection(cfg, {}), {}, VOCAB, _client, lock=elsewhere)
    with pytest.raises(JevError, match=r"\.lock"):
        run_topics(cfg, _selection(cfg, {}), {}, VOCAB, _client, lock=released)

    assert built == [] and not cfg.jev_topics_path.exists()


def test_the_pass_lock_refuses_a_second_holder_and_names_the_first(cfg: Config):
    from xbrain.jev.lock import PassLockBusy

    with pass_lock(cfg.jev_lock_path, "jev topics"):
        with pytest.raises(PassLockBusy, match=r"otra pasada .*jev topics.*pid"):
            with pass_lock(cfg.jev_lock_path, "jev serve"):
                pass  # pragma: no cover - never reached

    with pass_lock(cfg.jev_lock_path, "jev serve"):
        pass  # released: a later pass takes it


def test_the_pass_lock_refuses_another_process(cfg: Config):
    """flock is per open file, so the refusal is real across processes, not a Python flag."""
    import subprocess
    import sys

    probe = (
        "import fcntl, os, sys\n"
        "fd = os.open(sys.argv[1], os.O_RDWR)\n"
        "try:\n"
        "    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "except BlockingIOError:\n"
        "    sys.exit(3)\n"
    )
    with pass_lock(cfg.jev_lock_path, "jev topics"):
        held = subprocess.run([sys.executable, "-c", probe, str(cfg.jev_lock_path)], check=False)
    free = subprocess.run([sys.executable, "-c", probe, str(cfg.jev_lock_path)], check=False)

    assert (held.returncode, free.returncode) == (3, 0)


# --------------------------------------------------------------- cancelling a pass from outside


def test_a_cancelled_pass_stops_asking_and_keeps_what_was_answered(cfg: Config):
    """A server has no Ctrl-C of its own: `cancel` is how it interrupts a job. Nothing more is
    SENT once it is set, and the pass ends on the interrupt path — banked, saved, logged."""
    import threading

    cancel = threading.Event()

    class _CancelAfterFirst(FakeJevClient):
        def ask(self, state, questions):
            result = super().ask(state, questions)
            cancel.set()
            return result

    client = _CancelAfterFirst()

    with pass_lock(cfg.jev_lock_path, "test") as lock:
        outcome = run_topics(
            cfg, _selection(cfg, {}), {}, VOCAB, lambda: client, lock=lock, cancel=cancel
        )

    assert outcome.interrupted is True
    assert len(client.calls) == 1
    assert [a.item_id for a in outcome.assessed] == ["1"]
    assert list(load_assessments(cfg.jev_topics_path)) == ["1"]
    run = _only_run(cfg)
    assert (run.requests, run.ok, run.interrupted) == (1, 1, True)


def test_a_lock_that_cannot_be_taken_names_its_path(cfg: Config):
    blocker = cfg.jev_dir
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text("a file where the directory should be")

    with pytest.raises(JevError, match=r"candado .*\.lock"):
        with pass_lock(cfg.jev_lock_path, "test"):
            pass  # pragma: no cover - never reached


def test_a_failed_release_never_replaces_the_error_in_flight(cfg: Config, monkeypatch, caplog):
    from xbrain.jev import lock as lock_module

    def _boom(fd: int, length: int) -> None:
        raise OSError(5, "Input/output error")

    with pytest.raises(ValueError, match="la de verdad"):
        with pass_lock(cfg.jev_lock_path, "test"):
            monkeypatch.setattr(lock_module.os, "ftruncate", _boom)
            raise ValueError("la de verdad")

    assert "Input/output error" in caplog.text
    monkeypatch.undo()
    with pass_lock(cfg.jev_lock_path, "after"):
        pass  # the flock was still released


def test_a_cancelled_pass_books_refused_answers_as_failed_not_unsaved(cfg: Config):
    """After a soft cancel every answer was drained: a refused one is a failure."""
    import threading

    cancel = threading.Event()

    class _Refused(FakeJevClient):
        def ask(self, state, questions):
            result = super().ask(state, questions)
            cancel.set()
            return result

    with pass_lock(cfg.jev_lock_path, "test") as lock:
        outcome = run_topics(
            cfg,
            _selection(cfg, {}),
            {},
            VOCAB,
            lambda: _Refused(primary="banana"),
            lock=lock,
            cancel=cancel,
        )

    run = _only_run(cfg)
    assert outcome.interrupted is True
    assert (run.requests, run.ok, run.failed, run.unsaved) == (1, 0, 1, 0)


def test_the_failures_listed_are_capped_by_one_constant():
    """The terminal and a served job's outcome list the same number of failures: one
    constant (`run.FAILURES_SHOWN`), not a copy per caller."""
    from xbrain import cli
    from xbrain.jev import run, service

    assert run.FAILURES_SHOWN == 10
    assert not hasattr(cli, "_JEV_FAILURES_SHOWN") and not hasattr(service, "_FAILURES_SHOWN")


def test_a_final_save_that_fails_books_only_what_reached_the_disk(cfg: Config, monkeypatch):
    """The disk fills at the end: the run log must not book the answers held only in memory as
    `ok`. `ok` is what a save persisted (here the first checkpoint's one record); every other
    answer kept is `unsaved` — billed, not on disk."""
    from xbrain.jev import run as jev_run

    real = jev_run.save_assessments
    saves: list[int] = []

    def _fills_after_first(assessments, path):
        saves.append(len(assessments))
        if len(saves) > 1:
            raise OSError(28, "No space left on device")
        real(assessments, path)

    monkeypatch.setattr(jev_run, "CHECKPOINT_EVERY", 1)
    monkeypatch.setattr(jev_run, "save_assessments", _fills_after_first)

    with pytest.raises(JevError, match="sin guardar"):
        _pass(cfg, FakeJevClient())

    run = _only_run(cfg)
    assert (run.requests, run.ok, run.unsaved, run.failed) == (2, 1, 1, 0)
    assert list(load_assessments(cfg.jev_topics_path)) == ["1"]


def test_a_final_save_that_fails_with_nothing_checkpointed_books_everything_unsaved(
    cfg: Config, monkeypatch
):
    from xbrain.jev import run as jev_run

    def _full(assessments, path):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(jev_run, "save_assessments", _full)

    with pytest.raises(JevError, match="sin guardar"):
        _pass(cfg, FakeJevClient())

    run = _only_run(cfg)
    assert (run.requests, run.ok, run.unsaved, run.failed) == (2, 0, 2, 0)
