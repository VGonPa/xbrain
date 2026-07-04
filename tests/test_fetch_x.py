# tests/test_fetch_x.py
import json
from datetime import datetime, timezone
from pathlib import Path

import xbrain.fetch_x as fx
from xbrain.fetch_x import (
    _attach_x_sources,
    _classify_x_url,
    _fetch_rendered,
    _x_status_id,
    assemble_linked_thread,
    fetch_x_articles,
)
from xbrain.models import (
    ArticleImageBlock,
    ArticleTextBlock,
    Author,
    Content,
    ContentSourceFailure,
    ContentSourceSuccess,
    Item,
    Link,
    MediaPhotoPending,
)


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


def test_classify_x_url_does_not_misroute_article_containing_status():
    # An X article URL that itself contains a /status/ segment stays an article.
    assert _classify_x_url("https://x.com/i/article/foo/status/1") == "article"


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
        link_fetcher=lambda url: ContentSourceSuccess(kind="x_article", url=url, text="hilo"),
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
        return ContentSourceSuccess(kind="x_article", url=url, text="hilo")

    assert fetch_x_articles(store, None, link_fetcher=fetcher) == 1
    assert fetch_x_articles(store, None, link_fetcher=fetcher) == 0
    assert fetch_x_articles(store, None, force=True, link_fetcher=fetcher) == 1


def test_fetch_x_articles_requires_storage_state_without_link_fetcher():
    import pytest

    store = {"1": _item("1", ["https://x.com/jack/status/9"])}
    with pytest.raises(ValueError, match="storage_state_path"):
        fetch_x_articles(store, storage_state_path=None)


def test_fetch_x_articles_respects_since_until():
    early = _item("1", ["https://x.com/jack/status/9"])
    early.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    late = _item("2", ["https://x.com/jack/status/10"])
    late.created_at = datetime(2026, 6, 1, tzinfo=timezone.utc)
    store = {"1": early, "2": late}
    count = fetch_x_articles(
        store,
        storage_state_path=None,
        since=datetime(2026, 5, 1, tzinfo=timezone.utc),
        until=datetime(2026, 7, 1, tzinfo=timezone.utc),
        link_fetcher=lambda url: ContentSourceSuccess(kind="x_article", url=url, text="x"),
    )
    assert count == 1
    assert store["1"].content is None
    assert store["2"].content is not None


def test_needs_x_fetch_skips_item_created_after_until():
    # The `until` UPPER bound: an item created strictly after `until` is out of
    # window and must not be fetched, even with an x.com link and no content yet.
    item = _item("1", ["https://x.com/jack/status/9"])
    item.created_at = datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert (
        fx._needs_x_fetch(item, force=False, until=datetime(2026, 7, 1, tzinfo=timezone.utc))
        is False
    )
    # Sanity: within the window (created_at <= until) it IS fetched.
    assert (
        fx._needs_x_fetch(item, force=False, until=datetime(2026, 9, 1, tzinfo=timezone.utc))
        is True
    )


def test_fetch_x_articles_dedups_repeated_x_urls():
    store = {"1": _item("1", ["https://x.com/jack/status/9", "https://x.com/jack/status/9"])}
    fetch_x_articles(
        store,
        storage_state_path=None,
        link_fetcher=lambda url: ContentSourceSuccess(kind="x_article", url=url, text="hilo"),
    )
    sources = store["1"].content.sources
    assert len(sources) == 1


def test_fetch_x_articles_preserves_external_sources():
    item = _item("1", ["https://x.com/jack/status/9"])
    item.content = Content(
        fetched_at=datetime.now(timezone.utc),
        sources=[ContentSourceSuccess(kind="external_article", url="https://e.com", text="art")],
    )
    store = {"1": item}
    fetch_x_articles(
        store,
        None,
        link_fetcher=lambda url: ContentSourceSuccess(kind="x_article", url=url, text="x"),
    )
    kinds = {s.kind for s in store["1"].content.sources}
    assert kinds == {"external_article", "x_article"}


# --- fetch: structured article body via GraphQL interception (#39 PR3) ---
#
# FIXTURE PROVENANCE: the captured GraphQL payload below is CONSTRUCTED from the
# documented Draft.js content_state shape (see tests/test_article.py), not a
# recorded live X response — validate against a real payload before production
# reliance (RFC #39 open-Q #4). The Playwright page/context/response are faked
# so the test stays fully offline.

_ARTICLE_URL = "https://x.com/i/article/1900000000000000000"
_STATUS_URL = "https://x.com/jack/status/1900000000000000001"
_IMG = "https://pbs.twimg.com/media/ABC123.jpg"


def _content_state() -> dict:
    return {
        "blocks": [
            {"key": "a", "text": "Para one.", "type": "unstyled", "entityRanges": []},
            {
                "key": "b",
                "text": " ",
                "type": "atomic",
                "entityRanges": [{"offset": 0, "length": 1, "key": 0}],
            },
            {"key": "c", "text": "Para two.", "type": "unstyled", "entityRanges": []},
        ],
        "entityMap": {"0": {"type": "IMAGE", "data": {"url": _IMG, "altText": "alt"}}},
    }


def _article_payload() -> dict:
    return {
        "data": {
            "article": {
                "article_results": {
                    "result": {
                        "__typename": "Article",
                        "rest_id": "1900000000000000000",
                        "title": "Structured Read",
                        "content_state": _content_state(),
                    }
                }
            }
        }
    }


class _FakeResponse:
    def __init__(self, url: str, payload=None, *, raise_json: bool = False):
        self.url = url
        self._payload = payload
        self._raise = raise_json

    def json(self):
        if self._raise:
            raise ValueError("not JSON")
        return self._payload


class _FakePage:
    """A minimal stand-in for a Playwright page: fires captured responses to the
    registered `response` handler during `wait_for_timeout`."""

    def __init__(self, *, url: str, html: str, responses: list[_FakeResponse]):
        self.url = url
        self._html = html
        self._responses = responses
        self._handlers: list = []

    def on(self, event: str, handler) -> None:
        if event == "response":
            self._handlers.append(handler)

    def goto(self, url: str, wait_until: str | None = None) -> None:
        self.url = url

    def wait_for_timeout(self, _ms: int) -> None:
        for response in self._responses:
            for handler in self._handlers:
                handler(response)

    def content(self) -> str:
        return self._html

    def close(self) -> None:
        pass


class _FakeContext:
    def __init__(self, page: _FakePage):
        self._page = page

    def new_page(self) -> _FakePage:
        return self._page


def test_fetch_rendered_structured_path_builds_ordered_blocks(monkeypatch):
    # Mock trafilatura so the parser test never imports/runs the real extractor
    # (the truncation tripwire calls trafilatura.extract even on the structured
    # path); returning None means no tripwire warning fires.
    monkeypatch.setattr(fx.trafilatura, "extract", lambda html: None)
    page = _FakePage(
        url=_ARTICLE_URL,
        html="<html>ignored on the structured path</html>",
        responses=[
            _FakeResponse("https://x.com/i/api/graphql/xyz/TweetArticleContent", _article_payload())
        ],
    )
    source = _fetch_rendered(_FakeContext(page), _ARTICLE_URL)

    assert isinstance(source, ContentSourceSuccess)
    assert source.kind == "x_article"
    assert source.title == "Structured Read"
    assert source.http_status == 200
    assert [type(b).__name__ for b in source.blocks] == [
        "ArticleTextBlock",
        "ArticleImageBlock",
        "ArticleTextBlock",
    ]
    image = source.blocks[1]
    assert isinstance(image, ArticleImageBlock)
    assert isinstance(image.media, MediaPhotoPending)
    assert image.media.url == _IMG
    # text == concat of the ArticleTextBlock texts (the PR1 invariant).
    assert source.text == "".join(b.text for b in source.blocks if isinstance(b, ArticleTextBlock))
    assert source.text == "Para one.\n\nPara two."


def test_fetch_rendered_falls_back_to_trafilatura_when_no_graphql(monkeypatch):
    monkeypatch.setattr(fx.trafilatura, "extract", lambda html: "fallback body")
    page = _FakePage(url=_ARTICLE_URL, html="<html>body</html>", responses=[])
    source = _fetch_rendered(_FakeContext(page), _ARTICLE_URL)

    assert isinstance(source, ContentSourceSuccess)
    assert source.kind == "x_article"
    assert source.text == "fallback body"
    assert source.blocks == []
    assert source.title is None


def test_fetch_rendered_falls_back_when_captured_payload_is_malformed(monkeypatch):
    monkeypatch.setattr(fx.trafilatura, "extract", lambda html: "fallback body")
    page = _FakePage(
        url=_ARTICLE_URL,
        html="<html>body</html>",
        responses=[
            # graphql response present but not parseable to blocks + one that
            # raises on .json() — neither must crash; both degrade to fallback.
            _FakeResponse("https://x.com/i/api/graphql/z/TweetArticleContent", {"data": {}}),
            _FakeResponse("https://x.com/i/api/graphql/z/ArticleFoo", raise_json=True),
        ],
    )
    source = _fetch_rendered(_FakeContext(page), _ARTICLE_URL)

    assert isinstance(source, ContentSourceSuccess)
    assert source.text == "fallback body"
    assert source.blocks == []


def test_fetch_rendered_empty_article_records_empty_content_failure(monkeypatch):
    monkeypatch.setattr(fx.trafilatura, "extract", lambda html: None)
    page = _FakePage(url=_ARTICLE_URL, html="<html></html>", responses=[])
    source = _fetch_rendered(_FakeContext(page), _ARTICLE_URL)

    assert isinstance(source, ContentSourceFailure)
    assert source.failure_reason == "empty_content"


def test_fetch_rendered_non_article_page_uses_fallback(monkeypatch):
    # An x.com page that is not an article never runs the structured parser.
    monkeypatch.setattr(fx.trafilatura, "extract", lambda html: "profile bio")
    url = "https://x.com/some_profile"
    page = _FakePage(
        url=url,
        html="<html>bio</html>",
        # even if a graphql response leaks in, a non-article URL ignores it.
        responses=[_FakeResponse("https://x.com/i/api/graphql/q/ArticleFoo", _article_payload())],
    )
    source = _fetch_rendered(_FakeContext(page), url)

    assert isinstance(source, ContentSourceSuccess)
    assert source.text == "profile bio"
    assert source.blocks == []


def test_fetch_rendered_timeout_on_navigation_failure(caplog):
    class _BoomPage(_FakePage):
        def goto(self, url: str, wait_until: str | None = None) -> None:
            raise RuntimeError("nav failed")

    page = _BoomPage(url=_ARTICLE_URL, html="", responses=[])
    with caplog.at_level("WARNING", logger="xbrain.fetch_x"):
        source = _fetch_rendered(_FakeContext(page), _ARTICLE_URL)
    assert isinstance(source, ContentSourceFailure)
    assert source.failure_reason == "timeout"
    # data-safety #1: the exception detail is captured (not discarded), and a
    # WARNING is logged for debuggability of the still-unvalidated article path.
    assert "nav failed" in source.error
    assert any("navigation failed" in r.message for r in caplog.records)


def test_fetch_tweet_timeout_on_navigation_failure_captures_detail(caplog):
    class _BoomPage(_FakePage):
        def goto(self, url: str, wait_until: str | None = None) -> None:
            raise RuntimeError("boom tweet nav")

    page = _BoomPage(url=_STATUS_URL, html="", responses=[])
    with caplog.at_level("WARNING", logger="xbrain.fetch_x"):
        source = fx._fetch_tweet(_FakeContext(page), _STATUS_URL)
    assert isinstance(source, ContentSourceFailure)
    assert source.failure_reason == "timeout"
    assert "boom tweet nav" in source.error
    assert any("navigation failed" in r.message for r in caplog.records)


def test_nav_error_caps_long_exception_detail():
    # A misbehaving page can surface a huge exception body; the persisted error
    # is capped (mirrors media._MAX_ERROR_LEN) so items.json never bloats.
    huge = RuntimeError("x" * 5000)
    error = fx._nav_error("No se pudo cargar.", huge)
    assert len(error) == fx._MAX_ERROR_LEN
    assert error.endswith("…")


# --- observability + tripwire + degrade-not-crash (#39 PR3 review) ---


def test_fetch_rendered_warns_when_response_captured_but_zero_blocks(monkeypatch, caplog):
    # The "feature silently went dead" signal: an article-GraphQL response WAS
    # captured but parsed to no blocks -> WARNING (op-name/shape drift), fallback.
    monkeypatch.setattr(fx.trafilatura, "extract", lambda html: "fallback body")
    page = _FakePage(
        url=_ARTICLE_URL,
        html="<html>body</html>",
        responses=[
            _FakeResponse("https://x.com/i/api/graphql/z/TweetArticleContent", {"data": {}})
        ],
    )
    with caplog.at_level("WARNING", logger="xbrain.fetch_x"):
        source = _fetch_rendered(_FakeContext(page), _ARTICLE_URL)
    assert isinstance(source, ContentSourceSuccess)
    assert source.blocks == []
    assert any("captured" in r.message and "0 blocks" in r.message for r in caplog.records)


def test_fetch_rendered_warns_when_structured_body_is_truncated(monkeypatch, caplog):
    # Structured body far shorter than the trafilatura text -> truncation warning
    # (but the structured body still wins as the source of truth).
    monkeypatch.setattr(fx.trafilatura, "extract", lambda html: "x" * 1000)
    page = _FakePage(
        url=_ARTICLE_URL,
        html="<html>body</html>",
        responses=[
            _FakeResponse("https://x.com/i/api/graphql/z/TweetArticleContent", _article_payload())
        ],
    )
    with caplog.at_level("WARNING", logger="xbrain.fetch_x"):
        source = _fetch_rendered(_FakeContext(page), _ARTICLE_URL)
    assert source.blocks  # structured body preserved
    assert any("truncated" in r.message or "<50%" in r.message for r in caplog.records)


def test_truncation_tripwire_does_not_crash_when_trafilatura_raises(monkeypatch):
    # The truncation diagnostic runs on the structured SUCCESS path; if
    # trafilatura raises there it must NOT discard the already-built good body
    # nor crash — the tripwire is best-effort ("degrade, not crash").
    def _boom(_html):
        raise RuntimeError("trafilatura exploded")

    monkeypatch.setattr(fx.trafilatura, "extract", _boom)
    page = _FakePage(
        url=_ARTICLE_URL,
        html="<html>body</html>",
        responses=[
            _FakeResponse("https://x.com/i/api/graphql/z/TweetArticleContent", _article_payload())
        ],
    )
    source = _fetch_rendered(_FakeContext(page), _ARTICLE_URL)
    assert isinstance(source, ContentSourceSuccess)
    assert source.blocks  # structured body intact, not discarded
    assert source.text == "Para one.\n\nPara two."


def test_fetch_rendered_parser_exception_degrades_to_fallback(monkeypatch, caplog):
    # Any parser exception (incl. RecursionError) must degrade to trafilatura,
    # never crash the fetch.
    def _boom(_payload):
        raise RecursionError("deep payload")

    monkeypatch.setattr(fx, "parse_article_content_state", _boom)
    monkeypatch.setattr(fx.trafilatura, "extract", lambda html: "fallback body")
    page = _FakePage(
        url=_ARTICLE_URL,
        html="<html>body</html>",
        responses=[
            _FakeResponse("https://x.com/i/api/graphql/z/TweetArticleContent", _article_payload())
        ],
    )
    with caplog.at_level("WARNING", logger="xbrain.fetch_x"):
        source = _fetch_rendered(_FakeContext(page), _ARTICLE_URL)
    assert isinstance(source, ContentSourceSuccess)
    assert source.text == "fallback body"
    assert source.blocks == []
    assert any("parser raised" in r.message for r in caplog.records)


# --- _attach_x_sources: fetched_at bumps only on a MATERIAL change (#39 PR3) ---


def _blocks_source() -> ContentSourceSuccess:
    return ContentSourceSuccess(
        kind="x_article",
        url=_ARTICLE_URL,
        title="t",
        text="Para one.\n\nPara two.",
        blocks=[
            ArticleTextBlock(text="Para one."),
            ArticleImageBlock(media=MediaPhotoPending(url=_IMG), alt="alt"),
            ArticleTextBlock(text="\n\nPara two."),
        ],
        http_status=200,
        attempts=1,
    )


def test_attach_x_sources_bumps_fetched_at_when_content_materially_changes():
    item = _item("1", [_ARTICLE_URL])
    old_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    item.content = Content(
        fetched_at=old_time,
        sources=[ContentSourceSuccess(kind="x_article", url=_ARTICLE_URL, text="old text")],
    )
    _attach_x_sources(item, [_blocks_source()])
    # structured body replaced the text-only source -> material change -> re-enrich.
    assert item.content.fetched_at > old_time


def test_attach_x_sources_preserves_fetched_at_on_idempotent_refetch():
    item = _item("1", [_ARTICLE_URL])
    old_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    item.content = Content(fetched_at=old_time, sources=[_blocks_source()])
    # Re-fetch reproduces the same material content (attempts is bookkeeping).
    _attach_x_sources(item, [_blocks_source()])
    assert item.content.fetched_at == old_time


def test_attach_x_sources_bumps_fetched_at_when_first_article_added():
    item = _item("1", [_ARTICLE_URL])
    old_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    item.content = Content(
        fetched_at=old_time,
        sources=[ContentSourceSuccess(kind="external_article", url="https://e.com", text="art")],
    )
    _attach_x_sources(item, [_blocks_source()])
    assert item.content.fetched_at > old_time
    kinds = {s.kind for s in item.content.sources}
    assert kinds == {"external_article", "x_article"}


def test_attach_x_sources_uses_injected_clock():
    # Mirrors fetch_item's injectable `now` (deterministic, testable).
    fixed = datetime(2027, 3, 3, tzinfo=timezone.utc)
    item = _item("1", [_ARTICLE_URL])
    _attach_x_sources(item, [_blocks_source()], now=lambda: fixed)
    assert item.content is not None
    assert item.content.fetched_at == fixed


# --- _structured_article on the REAL captured payload shape (#66) ---
#
# Unlike the constructed `_article_payload()` above, these feed a real trimmed
# `article_results.result` (from tests/fixtures/art-*.json) through the PRODUCTION
# wrapper, so the BFS-locates-container-then-reads-`media_entities`/`cover_media`
# combination is exercised end-to-end, not just the pure parser.

_FIXTURES = Path(__file__).parent / "fixtures"
_OPENWIKI_TITLE = "Introducing OpenWiki, an open source agent for repo documentation"
_OPENWIKI_COVER = "https://pbs.twimg.com/media/HMKNwxAbUAEMrOF.jpg"
_OPENWIKI_INLINE = "https://pbs.twimg.com/media/HMKNQeJbMAA9ljZ.jpg"


def _real_result(name: str) -> dict:
    return json.loads((_FIXTURES / f"art-{name}.json").read_text(encoding="utf-8"))


def _wrap_result(result: dict) -> dict:
    """Nest a real `article_results.result` under a full GraphQL response shape."""
    return {"data": {"article": {"article_results": {"result": result}}}}


def test_structured_article_on_real_openwiki_payload():
    source = fx._structured_article([_wrap_result(_real_result("OpenWiki"))], _ARTICLE_URL)
    assert isinstance(source, ContentSourceSuccess)
    assert source.kind == "x_article"
    assert source.title == _OPENWIKI_TITLE
    images = [b for b in source.blocks if isinstance(b, ArticleImageBlock)]
    # cover first, then the inline MEDIA image — resolved off the sibling arrays.
    assert [b.media.url for b in images] == [_OPENWIKI_COVER, _OPENWIKI_INLINE]
    assert isinstance(source.blocks[0], ArticleImageBlock)
    assert source.text == "".join(b.text for b in source.blocks if isinstance(b, ArticleTextBlock))


def test_structured_article_selects_richest_of_multiple_payloads(caplog):
    preview = _wrap_result(
        {
            "title": "Preview",
            "content_state": {
                "blocks": [{"key": "a", "text": "preview only", "type": "unstyled"}],
                "entityMap": [],
            },
        }
    )
    full = _wrap_result(_real_result("OpenWiki"))
    with caplog.at_level("WARNING", logger="xbrain.fetch_x"):
        source = fx._structured_article([preview, full], _ARTICLE_URL)
    assert isinstance(source, ContentSourceSuccess)
    # The rich OpenWiki body wins over the 1-block preview, and the ambiguity logs.
    assert source.title == _OPENWIKI_TITLE
    assert len(source.blocks) > 5
    assert any("selecting the" in r.message for r in caplog.records)
