# tests/test_enrich.py
from datetime import datetime, timezone

import pytest

from xkb.enrich import apply_enrichment, enrich, items_pending_enrichment
from xkb.models import Author, Enrichment, Item


def _item(item_id: str) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )


def test_items_pending_returns_unenriched_items():
    store = {"1": _item("1"), "2": _item("2")}
    assert {i.id for i in items_pending_enrichment(store)} == {"1", "2"}


def test_apply_enrichment_attaches_result():
    item = _item("1")
    apply_enrichment(item, Enrichment(
        enriched_at=datetime.now(timezone.utc), executor="manual", summary="s"))
    assert item.enriched is not None
    assert item.enriched.summary == "s"
    assert items_pending_enrichment({"1": item}) == []


def test_enrich_manual_returns_pending_items():
    store = {"1": _item("1")}
    assert len(enrich(store, "manual")) == 1


def test_enrich_api_executor_is_paused():
    with pytest.raises(NotImplementedError, match="pausa"):
        enrich({"1": _item("1")}, "api")


def test_items_pending_respects_date_range():
    old_item = _item("1")
    old_item.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    new_item = _item("2")
    new_item.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = {"1": old_item, "2": new_item}
    pending = items_pending_enrichment(
        store, since=datetime(2023, 1, 1, tzinfo=timezone.utc)
    )
    assert {i.id for i in pending} == {"2"}
