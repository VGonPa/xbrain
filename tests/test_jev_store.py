# tests/test_jev_store.py
import json
from datetime import datetime, timezone
from pathlib import Path

from xbrain.jev.models import PrimaryChoice, TopicAssessment
from xbrain.jev.store import load_assessments, save_assessments

DT = datetime(2026, 9, 22, tzinfo=timezone.utc)


def _assessment(item_id: str) -> TopicAssessment:
    return TopicAssessment(
        item_id=item_id,
        provider="typesafe",
        model="jev-1.13.0",
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


def test_missing_file_loads_empty(tmp_path: Path):
    assert load_assessments(tmp_path / "nope.json") == {}


def test_save_is_atomic_leaves_no_tmp(tmp_path: Path):
    path = tmp_path / "topics.json"
    save_assessments({"1": _assessment("1")}, path)
    assert [p.name for p in tmp_path.iterdir()] == ["topics.json"]
