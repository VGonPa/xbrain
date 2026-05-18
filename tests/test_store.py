# tests/test_store.py
from datetime import datetime, timezone
from pathlib import Path

from xbrain.models import Author, Item, State
from xbrain.store import (
    load_state,
    load_store,
    merge_items,
    save_state,
    save_store,
)


def _item(item_id: str) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/x/status/{item_id}",
        author=Author(handle="x", name="X"),
        text="t",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )


def test_save_then_load_round_trips(tmp_path: Path):
    store = {"1": _item("1"), "2": _item("2")}
    path = tmp_path / "items.json"
    save_store(store, path)
    assert load_store(path) == store


def test_load_missing_file_returns_empty(tmp_path: Path):
    assert load_store(tmp_path / "nope.json") == {}


def test_merge_adds_only_new_items():
    store = {"1": _item("1")}
    added = merge_items(store, [_item("1"), _item("2")])
    assert added == 1
    assert set(store) == {"1", "2"}


def test_state_round_trips(tmp_path: Path):
    state = State()
    state.bookmarks.last_seen_id = "999"
    path = tmp_path / "state.json"
    save_state(state, path)
    assert load_state(path).bookmarks.last_seen_id == "999"


def test_topic_pages_round_trip(tmp_path):
    from datetime import datetime, timezone

    from xbrain.models import TopicPage
    from xbrain.store import load_topic_pages, save_topic_pages

    pages = {
        "ai-coding": TopicPage(
            slug="ai-coding",
            overview="o",
            notes=["n"],
            synthesized_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
            post_count_at_synth=10,
        )
    }
    path = tmp_path / "topics.json"
    save_topic_pages(pages, path)
    restored = load_topic_pages(path)
    assert restored["ai-coding"].post_count_at_synth == 10


def test_load_topic_pages_returns_empty_when_absent(tmp_path):
    from xbrain.store import load_topic_pages

    assert load_topic_pages(tmp_path / "missing.json") == {}
