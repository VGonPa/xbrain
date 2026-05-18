# tests/test_fetch_x.py
from datetime import datetime, timezone

from xbrain.fetch_x import _classify_x_url, _x_status_id, assemble_linked_thread, fetch_x_articles
from xbrain.models import Author, Content, ContentSource, Item, Link


def _item(item_id: str, urls: list[str]) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url=u, domain="x.com") for u in urls],
    )


def test_classify_x_url():
    assert _classify_x_url("https://x.com/jack/status/123") == "status"
    assert _classify_x_url("https://x.com/i/article/456") == "article"
    assert _classify_x_url("https://x.com/some_profile") == "other"


def test_x_status_id_extracts_the_tweet_id():
    assert _x_status_id("https://x.com/jack/status/123") == "123"
    assert _x_status_id("https://x.com/i/article/9") is None


def test_assemble_linked_thread_concatenates_the_anchor_author(monkeypatch):
    # parse_tweets is injected via the module so the test stays offline.
    import xbrain.fetch_x as fx

    a1 = Item(
        id="100",
        source="bookmark",
        url="u",
        author=Author(handle="jack", name="J"),
        text="primer tweet",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    a2 = Item(
        id="101",
        source="bookmark",
        url="u",
        author=Author(handle="jack", name="J"),
        text="segundo tweet",
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        captured_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    other = Item(
        id="200",
        source="bookmark",
        url="u",
        author=Author(handle="someone", name="S"),
        text="ruido de otra persona",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(fx, "parse_tweets", lambda response, source: response)
    handle, text = assemble_linked_thread([[a1, other, a2]], anchor_id="100")
    assert handle == "jack"
    assert text == "primer tweet\n\nsegundo tweet"


def test_assemble_linked_thread_returns_empty_when_anchor_absent():
    handle, text = assemble_linked_thread([], anchor_id="999")
    assert handle is None
    assert text == ""


def test_fetch_x_articles_attaches_sources_via_injected_fetcher():
    store = {"1": _item("1", ["https://x.com/jack/status/9"])}
    fetched = fetch_x_articles(
        store,
        storage_state_path=None,
        link_fetcher=lambda url: ContentSource(kind="x_article", url=url, text="hilo", ok=True),
    )
    assert fetched == 1
    assert store["1"].content is not None
    assert store["1"].content.sources[0].kind == "x_article"
    assert store["1"].content.sources[0].text == "hilo"


def test_fetch_x_articles_skips_items_without_x_links():
    from xbrain.models import Link

    item = _item("1", [])
    item.links = [Link(url="https://example.com/p", domain="example.com")]
    store = {"1": item}
    assert fetch_x_articles(store, None, link_fetcher=lambda url: None) == 0


def test_fetch_x_articles_skips_already_fetched_unless_forced():
    store = {"1": _item("1", ["https://x.com/jack/status/9"])}

    def fetcher(url):
        return ContentSource(kind="x_article", url=url, text="hilo", ok=True)

    assert fetch_x_articles(store, None, link_fetcher=fetcher) == 1
    assert fetch_x_articles(store, None, link_fetcher=fetcher) == 0
    assert fetch_x_articles(store, None, force=True, link_fetcher=fetcher) == 1


def test_fetch_x_articles_preserves_external_sources():
    item = _item("1", ["https://x.com/jack/status/9"])
    item.content = Content(
        fetched_at=datetime.now(timezone.utc),
        sources=[ContentSource(kind="external_article", url="https://e.com", text="art", ok=True)],
    )
    store = {"1": item}
    fetch_x_articles(
        store,
        None,
        link_fetcher=lambda url: ContentSource(kind="x_article", url=url, text="x", ok=True),
    )
    kinds = {s.kind for s in store["1"].content.sources}
    assert kinds == {"external_article", "x_article"}
