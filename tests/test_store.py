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


def test_load_store_migrates_legacy_content_source_records_in_place(tmp_path: Path):
    """Pre-#20 `data/items.json` files use `ok: bool` on each ContentSource.

    `load_store` must read those records into the new tagged-union variants,
    and the next `save_store` must persist them with the `outcome`
    discriminator (no `ok` field). This is the load-bearing test for the
    upgrade story: existing users do not need to run any migration command —
    a single read/write cycle is enough.
    """
    import json

    from xbrain.models import ContentSourceFailure, ContentSourceSuccess

    legacy_items = {
        "1": {
            "id": "1",
            "source": "bookmark",
            "url": "https://x.com/a/status/1",
            "author": {"handle": "a", "name": "A"},
            "text": "t",
            "created_at": "2026-05-10T00:00:00+00:00",
            "captured_at": "2026-05-16T00:00:00+00:00",
            "media": [],
            "links": [],
            "content": {
                "fetched_at": "2026-05-17T00:00:00+00:00",
                "sources": [
                    {
                        "kind": "external_article",
                        "url": "https://e.com/good",
                        "ok": True,
                        "title": "T",
                        "text": "body",
                        "http_status": 200,
                        "failure_reason": None,
                        "error": None,
                        "attempts": 1,
                    },
                    {
                        "kind": "external_article",
                        "url": "https://e.com/dead",
                        "ok": False,
                        "title": None,
                        "text": None,
                        "http_status": 404,
                        "failure_reason": "not_found",
                        "error": "HTTP 404",
                        "attempts": 2,
                    },
                ],
            },
        }
    }
    path = tmp_path / "items.json"
    path.write_text(json.dumps(legacy_items), encoding="utf-8")

    store = load_store(path)
    sources = store["1"].content.sources
    assert isinstance(sources[0], ContentSourceSuccess)
    assert sources[0].text == "body"
    assert sources[0].title == "T"
    assert isinstance(sources[1], ContentSourceFailure)
    assert sources[1].failure_reason == "not_found"
    assert sources[1].error == "HTTP 404"

    save_store(store, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    persisted = raw["1"]["content"]["sources"]
    assert persisted[0]["outcome"] == "success"
    assert "ok" not in persisted[0]
    assert "failure_reason" not in persisted[0]
    assert persisted[1]["outcome"] == "failure"
    assert "ok" not in persisted[1]
    assert "title" not in persisted[1]

    # The post-migration file is itself idempotent under reload.
    assert load_store(path) == store


def test_load_store_rejects_content_source_without_any_discriminator(tmp_path: Path):
    """A record with neither `outcome` nor `ok` must fail loudly, not default."""
    import json

    import pytest

    legacy_items = {
        "1": {
            "id": "1",
            "source": "bookmark",
            "url": "https://x.com/a/status/1",
            "author": {"handle": "a", "name": "A"},
            "text": "t",
            "created_at": "2026-05-10T00:00:00+00:00",
            "captured_at": "2026-05-16T00:00:00+00:00",
            "media": [],
            "links": [],
            "content": {
                "fetched_at": "2026-05-17T00:00:00+00:00",
                "sources": [
                    {"kind": "external_article", "url": "https://e.com/x"},
                ],
            },
        }
    }
    path = tmp_path / "items.json"
    path.write_text(json.dumps(legacy_items), encoding="utf-8")
    with pytest.raises(Exception):  # noqa: BLE001 - pydantic ValidationError wraps the ValueError
        load_store(path)
