# tests/test_jev_runs.py
"""`data/jev/runs.jsonl`: one line per `xbrain jev topics` pass that sent a request.

The side-car keeps only the LATEST assessment per item, so it cannot say what the operator
has been billed over time. The run log can, and only if it never loses a paid line: a corrupt
line is refused by number, never skipped.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from xbrain.jev.client import JevError
from xbrain.jev.defaults import INPUT_USD_PER_MTOK, input_cost_usd, tokens_cost_usd
from xbrain.jev.models import JevRun, PrimaryChoice, TopicAssessment
from xbrain.jev.store import append_run, load_runs

T0 = datetime(2026, 9, 26, 10, 0, tzinfo=timezone.utc)


def _run(**overrides) -> JevRun:
    fields = {
        "started_at": T0,
        "finished_at": T0 + timedelta(seconds=40),
        "models": ["jev-1.13.0"],
        "requests": 20,
        "ok": 18,
        "failed": 2,
        "input_tokens_by_provider": {"typesafe": 36_000},
        "input_tokens": 36_000,
        "input_tokens_unknown": 0,
        "interrupted": False,
        "unsaved": 0,
    }
    fields.update(overrides)
    return JevRun(**fields)


# --------------------------------------------------------------------------- the record


def test_a_run_is_frozen_and_refuses_unknown_fields():
    run = _run()

    with pytest.raises(ValidationError):
        run.requests = 3  # type: ignore[misc]
    with pytest.raises(ValidationError, match="extra"):
        JevRun(**{**run.model_dump(), "cost_usd": 1.0})


def test_a_run_refuses_a_naive_or_non_utc_instant():
    with pytest.raises(ValidationError, match="started_at"):
        _run(started_at=datetime(2026, 9, 26, 10, 0))
    with pytest.raises(ValidationError, match="UTC"):
        _run(finished_at=datetime(2026, 9, 26, 12, 0, tzinfo=timezone(timedelta(hours=2))))


def test_a_run_cannot_finish_before_it_started():
    with pytest.raises(ValidationError, match="finished_at"):
        _run(finished_at=T0 - timedelta(seconds=1))


def test_a_completed_run_accounts_for_every_request():
    """Not interrupted: every request sent came back, as an answer or as a failure."""
    with pytest.raises(ValidationError, match="requests"):
        _run(requests=20, ok=18, failed=1)


def test_an_interrupted_run_may_leave_requests_in_flight_but_never_more_answers_than_asks():
    in_flight = _run(requests=20, ok=10, failed=2, interrupted=True)
    assert in_flight.requests - in_flight.ok - in_flight.failed == 8

    with pytest.raises(ValidationError, match="requests"):
        _run(requests=5, ok=10, failed=0, interrupted=True)


def test_the_token_sum_is_the_sum_of_its_providers():
    """Stored twice so a plain reader gets the total; the two must never disagree."""
    with pytest.raises(ValidationError, match="input_tokens"):
        _run(input_tokens_by_provider={"typesafe": 10, "otro": 5}, input_tokens=14)


def test_a_run_where_nothing_answered_has_no_provider_and_no_tokens():
    """A 402 on every call: 20 sent, 20 failed. That IS history, and it prices at zero."""
    run = _run(ok=0, failed=20, models=[], input_tokens_by_provider={}, input_tokens=0)

    assert run.requests == 20 and run.failed == 20


def test_models_are_stored_sorted_and_distinct():
    with pytest.raises(ValidationError, match="models"):
        _run(models=["jev-1.14.0", "jev-1.13.0"])
    with pytest.raises(ValidationError, match="models"):
        _run(models=["jev-1.13.0", "jev-1.13.0"])


# --------------------------------------------------------------------------- the price


def test_tokens_are_priced_by_the_same_table_as_a_stored_assessment():
    """ONE price table: the run log and the side-car must quote the same dollars for the
    same tokens, or a price correction reprices one history and not the other."""
    assessment = TopicAssessment(
        item_id="1",
        provider="typesafe",
        model="jev-1.13.0",
        asked_at=T0,
        contract="a" * 64,
        state_chars=3,
        membership={"ai-coding": 0.9},
        primary=PrimaryChoice(choice="ai-coding", confidence=0.8, probabilities={"ai-coding": 1}),
        input_tokens=2_000,
    )

    assert tokens_cost_usd(2_000, "typesafe") == 2_000 / 1e6 * INPUT_USD_PER_MTOK["typesafe"]
    assert tokens_cost_usd(2_000, "typesafe") == input_cost_usd([assessment])
    # An unpriced provider contributes zero rather than borrowing another vendor's rate.
    assert tokens_cost_usd(2_000, "otro-juez") == 0.0


# --------------------------------------------------------------------------- the file


def test_a_missing_log_is_an_empty_history(tmp_path: Path):
    assert load_runs(tmp_path / "jev" / "runs.jsonl") == []


def test_append_then_load_round_trips_in_order(tmp_path: Path):
    path = tmp_path / "jev" / "runs.jsonl"
    first = _run()
    second = _run(started_at=T0 + timedelta(hours=1), finished_at=T0 + timedelta(hours=2))

    append_run(first, path)
    append_run(second, path)

    assert load_runs(path) == [first, second]
    # One JSON object per line, nothing else: the file is readable with `jq -c` by hand.
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["requests"] == 20


def test_append_never_rewrites_earlier_lines(tmp_path: Path):
    """Append-only: the bytes a previous run wrote are still the file's prefix."""
    path = tmp_path / "runs.jsonl"
    append_run(_run(), path)
    before = path.read_bytes()

    append_run(_run(requests=3, ok=3, failed=0), path)

    assert path.read_bytes().startswith(before)


def test_a_corrupt_line_is_refused_by_path_and_line_number(tmp_path: Path):
    """Never silently drop paid history: the operator repairs the line, by number."""
    path = tmp_path / "runs.jsonl"
    append_run(_run(), path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")

    with pytest.raises(JevError, match=r"runs\.jsonl.*línea 2"):
        load_runs(path)


def test_a_line_the_validator_refuses_is_refused_by_line_number_too(tmp_path: Path):
    path = tmp_path / "runs.jsonl"
    record = _run().model_dump(mode="json")
    record["requests"] = -1
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(JevError, match="línea 1"):
        load_runs(path)


def test_a_blank_trailing_line_is_not_a_record(tmp_path: Path):
    """An editor that adds a final newline must not turn a good log into a corrupt one."""
    path = tmp_path / "runs.jsonl"
    append_run(_run(), path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    assert len(load_runs(path)) == 1


# --------------------------------------------------------------------------- fix wave (PR 8)


def test_a_run_is_a_topics_run_by_default_and_old_lines_still_load(tmp_path: Path):
    """`kind` exists now so a second kind of pass (`jev ask`) can share the file; a line
    written before the field existed reads as a topics pass."""
    path = tmp_path / "runs.jsonl"
    old = _run().model_dump(mode="json")
    old.pop("kind")
    old.pop("unsaved")
    path.write_text(json.dumps(old) + "\n", encoding="utf-8")

    [run] = load_runs(path)

    assert run.kind == "topics" and run.unsaved == 0


def test_answers_that_came_back_but_were_not_saved_are_their_own_bucket():
    """Ctrl-C: an answer a worker received but the main loop never drained was billed and
    not kept. It is not a failure, and it is not in flight."""
    run = _run(requests=10, ok=4, failed=1, unsaved=2, interrupted=True)

    assert run.requests - run.ok - run.failed - run.unsaved == 3  # in flight

    with pytest.raises(ValidationError, match="unsaved"):
        _run(requests=20, ok=18, failed=0, unsaved=2)  # only an interrupt leaves any
    with pytest.raises(ValidationError, match="requests"):
        _run(requests=5, ok=4, failed=1, unsaved=1, interrupted=True)


def test_tokens_by_provider_refuse_a_blank_provider_or_a_negative_count():
    with pytest.raises(ValidationError):
        _run(input_tokens_by_provider={"": 10}, input_tokens=10)
    with pytest.raises(ValidationError):
        _run(input_tokens_by_provider={"typesafe": -1}, input_tokens=-1)


def test_an_append_after_a_torn_last_line_starts_on_a_new_line(tmp_path: Path):
    """A crash mid-write leaves a fragment without its newline. Gluing the next record onto
    it would hide a paid pass inside a line the loader refuses."""
    path = tmp_path / "runs.jsonl"
    path.write_text('{"started_at": "2026-09-2', encoding="utf-8")  # torn, no newline
    run = _run()

    append_run(run, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert JevRun.model_validate_json(lines[1]) == run
    with pytest.raises(JevError, match="línea 1"):  # only the fragment is refused
        load_runs(path)


def test_a_write_that_fails_leaves_the_log_byte_identical(tmp_path: Path, monkeypatch):
    """The bytes are written, then the flush to disk fails: the partial line is rolled back
    so the next append does not land on a fragment."""
    from xbrain.jev import store as jev_store

    path = tmp_path / "runs.jsonl"
    append_run(_run(), path)
    before = path.read_bytes()

    def _fsync_fails(fd: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(jev_store.os, "fsync", _fsync_fails)
    with pytest.raises(OSError):
        append_run(_run(requests=3, ok=3, failed=0), path)

    assert path.read_bytes() == before
