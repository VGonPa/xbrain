# tests/test_fetch.py
from datetime import datetime, timezone

from xkb.fetch import fetch_item, fetch_pending
from xkb.models import Author, Item, Link


def _item(item_id: str, urls: list[str]) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url=u, domain="d") for u in urls],
    )


def _fake_extractor(url: str) -> tuple[str | None, str | None]:
    return "Título", f"cuerpo de {url}"


def test_fetch_item_extracts_external_articles():
    content = fetch_item(_item("1", ["https://example.com/p"]), _fake_extractor)
    assert content.sources[0].kind == "external_article"
    assert content.sources[0].ok is True
    assert content.sources[0].text == "cuerpo de https://example.com/p"


def test_fetch_item_marks_x_urls_as_deferred():
    content = fetch_item(_item("1", ["https://x.com/foo/status/9"]), _fake_extractor)
    assert content.sources[0].kind == "x_article"
    assert content.sources[0].ok is False
    assert "v1" in content.sources[0].error


def test_fetch_item_marks_failed_extraction():
    content = fetch_item(_item("1", ["https://example.com/p"]), lambda url: (None, None))
    assert content.sources[0].ok is False


def test_fetch_pending_skips_already_fetched_items():
    store = {"1": _item("1", ["https://example.com/p"])}
    assert fetch_pending(store, extractor=_fake_extractor) == 1
    assert fetch_pending(store, extractor=_fake_extractor) == 0  # already has content


def test_fetch_pending_skips_items_without_links():
    store = {"1": _item("1", [])}
    assert fetch_pending(store, extractor=_fake_extractor) == 0


def _raising(url: str) -> tuple[str | None, str | None]:
    raise RuntimeError("boom")


def test_fetch_item_isolates_extractor_exception():
    content = fetch_item(_item("1", ["https://example.com/p"]), _raising)
    assert len(content.sources) == 1
    assert content.sources[0].ok is False
    assert "boom" in content.sources[0].error


def test_fetch_pending_respects_since_until():
    early = _item("1", ["https://example.com/p"])
    early.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    late = _item("2", ["https://example.com/q"])
    late.created_at = datetime(2026, 6, 1, tzinfo=timezone.utc)
    store = {"1": early, "2": late}
    count = fetch_pending(
        store,
        since=datetime(2026, 5, 1, tzinfo=timezone.utc),
        until=datetime(2026, 7, 1, tzinfo=timezone.utc),
        extractor=_fake_extractor,
    )
    assert count == 1
    assert store["1"].content is None
    assert store["2"].content is not None


def test_fetch_pending_force_refetches():
    store = {"1": _item("1", ["https://example.com/p"])}
    assert fetch_pending(store, extractor=_fake_extractor) == 1
    assert fetch_pending(store, extractor=_fake_extractor) == 0
    assert fetch_pending(store, force=True, extractor=_fake_extractor) == 1


def test_fetch_item_handles_mixed_links():
    item = _item("1", ["https://x.com/foo/status/9", "https://example.com/p"])
    content = fetch_item(item, _fake_extractor)
    assert len(content.sources) == 2
    assert content.sources[0].kind == "x_article"
    assert content.sources[0].ok is False
    assert content.sources[1].kind == "external_article"
    assert content.sources[1].ok is True
