# tests/test_models.py
from datetime import datetime, timezone

from xkb.models import Author, Item, Link, State


def test_item_round_trips_through_json():
    item = Item(
        id="123",
        source="bookmark",
        url="https://x.com/foo/status/123",
        author=Author(handle="foo", name="Foo Bar"),
        text="hello world",
        created_at=datetime(2026, 5, 10, 14, 23, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, 9, 0, tzinfo=timezone.utc),
        links=[Link(url="https://example.com/a", domain="example.com")],
    )
    restored = Item.model_validate_json(item.model_dump_json())
    assert restored == item
    assert restored.content is None
    assert restored.enriched is None


def test_state_defaults_are_empty_cursors():
    state = State()
    assert state.bookmarks.last_seen_id is None
    assert state.own_tweets.last_seen_id is None
    assert state.archive_imported is None
