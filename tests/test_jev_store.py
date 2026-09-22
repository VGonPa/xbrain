# tests/test_jev_store.py
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.jev.client import JevError
from xbrain.jev.models import PrimaryChoice, TopicAssessment
from xbrain.jev.store import load_assessments, save_assessments

DT = datetime(2026, 9, 22, tzinfo=timezone.utc)


def _assessment(item_id: str) -> TopicAssessment:
    return TopicAssessment(
        item_id=item_id,
        provider="typesafe",
        # Non-ASCII on purpose: it is what makes the `ensure_ascii=False` assertion bite.
        model="jev-1.13.0-piloto-españa",
        asked_at=DT,
        contract="a" * 64,
        state_chars=3,
        membership={"ai-coding": 0.9},
        primary=PrimaryChoice(
            choice="ai-coding", confidence=0.8, probabilities={"ai-coding": 0.8, "otro": 0.2}
        ),
    )


def test_round_trip_is_sorted_pretty_utf8_json(tmp_path: Path):
    path = tmp_path / "data" / "jev" / "topics.json"
    save_assessments({"2": _assessment("2"), "1": _assessment("1")}, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert list(raw) == ["1", "2"]
    assert raw["1"]["primary"]["choice"] == "ai-coding"
    assert load_assessments(path) == {"1": _assessment("1"), "2": _assessment("2")}

    text = path.read_text(encoding="utf-8")
    # PRETTY: the whole file is rewritten every run, so without `indent=2` a run that changed
    # two records would diff as one rewritten line and be unreadable by hand.
    assert '\n  "1": {' in text
    # UTF-8, not `\uXXXX`: the side-car is read by people, and by `diff`.
    assert "españa" in text
    assert "\\u" not in text


def test_missing_file_loads_empty(tmp_path: Path):
    assert load_assessments(tmp_path / "nope.json") == {}


def test_save_leaves_no_tmp_file_behind(tmp_path: Path):
    """The cleanup arm of `_atomic_write` — no stray `.tmp` survives a successful save.

    Renamed from "…is_atomic": a plain `write_text` satisfies this too, so it does NOT pin
    atomicity. `test_a_failed_rename_leaves_the_previous_sidecar_intact` does.
    """
    path = tmp_path / "topics.json"
    save_assessments({"1": _assessment("1")}, path)
    assert [p.name for p in tmp_path.iterdir()] == ["topics.json"]


def test_a_failed_rename_leaves_the_previous_sidecar_intact(tmp_path: Path, monkeypatch):
    """The point of the atomic write is not "no .tmp left over" — it is that a save which
    dies mid-way costs nothing: the file the LAST run paid for is still there, whole."""
    path = tmp_path / "topics.json"
    save_assessments({"1": _assessment("1")}, path)
    before = path.read_bytes()

    def _boom(src, dst):
        raise OSError("disco lleno")

    monkeypatch.setattr("xbrain.store.os.replace", _boom)
    with pytest.raises(OSError):
        save_assessments({"1": _assessment("1"), "2": _assessment("2")}, path)
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["topics.json"]


def test_a_corrupt_sidecar_raises_instead_of_reasking_the_corpus(tmp_path: Path):
    """Returning `{}` here would report "0 evaluaciones guardadas" over a full side-car and
    re-pay for every item in the corpus. The path is in the message: three files are loaded
    back-to-back by `jev topics`, and a bare parser error names none of them."""
    path = tmp_path / "topics.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(JevError) as excinfo:
        load_assessments(path)
    assert str(path) in str(excinfo.value)
    assert "side-car ilegible" in str(excinfo.value)


def test_a_record_the_validator_refuses_is_not_silently_dropped(tmp_path: Path):
    """An unknown field means the file was written by something this build does not know.
    Loading it as `{}` — or skipping just that record — would re-bill it."""
    path = tmp_path / "topics.json"
    payload = {"1": {**_assessment("1").model_dump(mode="json"), "campo_desconocido": 1}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(JevError) as excinfo:
        load_assessments(path)
    assert str(path) in str(excinfo.value)


def test_a_sidecar_that_is_not_a_json_object_is_an_operator_error(tmp_path: Path):
    """A hand-edited side-car is the one most likely to end up as `[]`. `raw.items()` on a
    list raises `AttributeError`, which `_OPERATOR_ERRORS` does not list — so it would reach
    the operator as a traceback rather than as a clean `Error:` line."""
    path = tmp_path / "topics.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(JevError) as excinfo:
        load_assessments(path)
    assert str(path) in str(excinfo.value)
