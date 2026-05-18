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
