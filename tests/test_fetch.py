# tests/test_fetch.py
import socket
import urllib.error
from datetime import datetime, timezone

from xbrain.fetch import (
    FetchResult,
    _categorize_url_error,
    _probe_status,
    _reason_for_status,
    fetch_item,
    fetch_pending,
    trafilatura_extract,
)
from xbrain.models import Author, Item, Link


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


def _fake_extractor(url: str) -> FetchResult:
    return FetchResult(title="Título", text=f"cuerpo de {url}", http_status=200)


def test_reason_for_status_maps_http_codes():
    assert _reason_for_status(404) == "not_found"
    assert _reason_for_status(410) == "not_found"
    assert _reason_for_status(403) == "forbidden"
    assert _reason_for_status(402) == "paywall"
    assert _reason_for_status(451) == "paywall"
    assert _reason_for_status(500) is None  # server error — keep raw error text


def test_categorize_url_error_detects_timeout_and_dns():
    assert _categorize_url_error(urllib.error.URLError(socket.timeout())) == "timeout"
    assert _categorize_url_error(urllib.error.URLError(socket.gaierror())) == "dns_error"
    assert _categorize_url_error(urllib.error.URLError("other")) is None


def test_probe_status_categorizes_http_error():
    def opener(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    status, reason, _ = _probe_status("https://e.com/x", opener=opener)
    assert status == 404
    assert reason == "not_found"


def test_trafilatura_extract_success():
    result = trafilatura_extract(
        "https://e.com/a",
        fetch=lambda url: "<html>body</html>",
        extract=lambda html: "el cuerpo",
        prober=lambda url: (None, None, ""),
    )
    assert result.text == "el cuerpo"
    assert result.http_status == 200
    assert result.failure_reason is None


def test_trafilatura_extract_empty_content_when_no_article():
    result = trafilatura_extract(
        "https://e.com/a",
        fetch=lambda url: "<html></html>",
        extract=lambda html: None,
        prober=lambda url: (None, None, ""),
    )
    assert result.text is None
    assert result.failure_reason == "empty_content"


def test_trafilatura_extract_probes_when_download_fails():
    result = trafilatura_extract(
        "https://e.com/a",
        fetch=lambda url: None,
        extract=lambda html: None,
        prober=lambda url: (404, "not_found", "HTTP 404"),
    )
    assert result.http_status == 404
    assert result.failure_reason == "not_found"


def test_fetch_item_extracts_external_articles():
    content = fetch_item(_item("1", ["https://example.com/p"]), _fake_extractor)
    assert content.sources[0].kind == "external_article"
    assert content.sources[0].ok is True
    assert content.sources[0].text == "cuerpo de https://example.com/p"


def test_fetch_item_skips_x_urls():
    # x.com links are handled by fetch_x.fetch_x_articles, not fetch_item.
    content = fetch_item(_item("1", ["https://x.com/foo/status/9"]), _fake_extractor)
    assert content.sources == []


def test_fetch_item_records_failure_evidence():
    content = fetch_item(
        _item("1", ["https://example.com/p"]),
        lambda url: FetchResult(http_status=404, failure_reason="not_found", error="HTTP 404"),
    )
    assert content.sources[0].ok is False
    assert content.sources[0].http_status == 404
    assert content.sources[0].failure_reason == "not_found"


def test_fetch_item_isolates_extractor_exception():
    def _raising(url):
        raise RuntimeError("boom")

    content = fetch_item(_item("1", ["https://example.com/p"]), _raising)
    assert len(content.sources) == 1
    assert content.sources[0].ok is False
    assert "boom" in content.sources[0].error


def test_fetch_item_preserves_non_external_sources_on_refetch():
    from xbrain.models import Content, ContentSource

    item = _item("1", ["https://example.com/p"])
    item.content = Content(
        fetched_at=datetime.now(timezone.utc),
        sources=[ContentSource(kind="thread", url="u", text="hilo", ok=True)],
    )
    content = fetch_item(item, _fake_extractor)
    kinds = {s.kind for s in content.sources}
    assert kinds == {"thread", "external_article"}


def test_fetch_pending_skips_already_fetched_items():
    store = {"1": _item("1", ["https://example.com/p"])}
    assert fetch_pending(store, extractor=_fake_extractor) == 1
    assert fetch_pending(store, extractor=_fake_extractor) == 0


def test_fetch_pending_skips_items_without_external_links():
    store = {"1": _item("1", []), "2": _item("2", ["https://x.com/a/status/9"])}
    assert fetch_pending(store, extractor=_fake_extractor) == 0


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


def test_fetch_item_dedups_repeated_link_urls():
    # An item whose links repeat the same URL must yield a single source.
    content = fetch_item(
        _item("1", ["https://example.com/p", "https://example.com/p"]), _fake_extractor
    )
    urls = [s.url for s in content.sources]
    assert urls == ["https://example.com/p"]


def test_fetch_pending_force_refetches():
    store = {"1": _item("1", ["https://example.com/p"])}
    assert fetch_pending(store, extractor=_fake_extractor) == 1
    assert fetch_pending(store, extractor=_fake_extractor) == 0
    assert fetch_pending(store, force=True, extractor=_fake_extractor) == 1


def test_firecrawl_extract_returns_none_without_api_key(monkeypatch):
    from xbrain.fetch import _firecrawl_extract

    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    assert _firecrawl_extract("https://e.com/a") is None


def test_firecrawl_extract_parses_markdown(monkeypatch):
    import io
    import json

    from xbrain.fetch import _firecrawl_extract

    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
    payload = {"data": {"markdown": "el cuerpo", "metadata": {"title": "T"}}}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=0):
        return _Resp(json.dumps(payload).encode())

    result = _firecrawl_extract("https://e.com/a", opener=opener)
    assert result is not None
    assert result.text == "el cuerpo"
    assert result.title == "T"


def test_extract_article_falls_back_to_firecrawl_on_js_required():
    from xbrain.fetch import FetchResult, extract_article

    def primary(url):
        return FetchResult(failure_reason="js_required", error="js", attempts=1)

    def firecrawl(url):
        return FetchResult(text="rescatado por firecrawl", attempts=1)

    result = extract_article("https://e.com/a", primary=primary, firecrawl=firecrawl)
    assert result.text == "rescatado por firecrawl"
    assert result.attempts == 2


def test_extract_article_keeps_evidence_when_firecrawl_unavailable():
    from xbrain.fetch import FetchResult, extract_article

    def primary(url):
        return FetchResult(failure_reason="js_required", error="js", attempts=1)

    def firecrawl(url):
        return None

    result = extract_article("https://e.com/a", primary=primary, firecrawl=firecrawl)
    assert result.text is None
    assert result.failure_reason == "js_required"
    assert result.attempts == 1


def test_extract_article_does_not_retry_hard_failures():
    from xbrain.fetch import FetchResult, extract_article

    # A 404 is definitive — Firecrawl must not even be called.
    calls = []

    def primary(url):
        return FetchResult(http_status=404, failure_reason="not_found", attempts=1)

    def firecrawl(url):
        calls.append(url)
        return FetchResult(text="x")

    result = extract_article("https://e.com/a", primary=primary, firecrawl=firecrawl)
    assert result.failure_reason == "not_found"
    assert calls == []


class _CtxResp:
    """A minimal context-manager response object for fake openers."""

    def __init__(self, body: bytes = b"", status: int = 200):
        self._body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


def test_probe_status_success_path_marks_js_required():
    # A reachable URL trafilatura could not parse -> categorised as js_required.
    def opener(req, timeout=0):
        return _CtxResp(status=200)

    status, reason, error = _probe_status("https://e.com/x", opener=opener)
    assert status == 200
    assert reason == "js_required"
    assert error


def test_probe_status_categorizes_url_error_wrapping_timeout():
    def opener(req, timeout=0):
        raise urllib.error.URLError(socket.timeout())

    status, reason, _ = _probe_status("https://e.com/x", opener=opener)
    assert status is None
    assert reason == "timeout"


def test_probe_status_categorizes_url_error_wrapping_dns_error():
    def opener(req, timeout=0):
        raise urllib.error.URLError(socket.gaierror())

    status, reason, _ = _probe_status("https://e.com/x", opener=opener)
    assert status is None
    assert reason == "dns_error"


def test_firecrawl_extract_returns_error_result_when_opener_raises(monkeypatch):
    from xbrain.fetch import _firecrawl_extract

    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")

    def opener(req, timeout=0):
        raise urllib.error.URLError("network down")

    result = _firecrawl_extract("https://e.com/a", opener=opener)
    assert result is not None
    assert result.text is None
    assert "Firecrawl falló" in (result.error or "")


def test_firecrawl_extract_detects_error_envelope(monkeypatch):
    import json

    from xbrain.fetch import _firecrawl_extract

    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
    payload = {"success": False, "error": "rate limited"}

    def opener(req, timeout=0):
        return _CtxResp(json.dumps(payload).encode())

    result = _firecrawl_extract("https://e.com/a", opener=opener)
    assert result is not None
    assert result.text is None
    assert "rate limited" in (result.error or "")


def test_extract_article_merges_firecrawl_error_when_both_fail():
    from xbrain.fetch import FetchResult, extract_article

    def primary(url):
        return FetchResult(failure_reason="js_required", error="js", attempts=1)

    def firecrawl(url):
        # Firecrawl reachable but returned no text — carries its own evidence.
        return FetchResult(error="Firecrawl no devolvió contenido.", attempts=1)

    result = extract_article("https://e.com/a", primary=primary, firecrawl=firecrawl)
    assert result.text is None
    assert result.attempts == 2
    assert "Firecrawl: Firecrawl no devolvió contenido." in (result.error or "")


# --------------------------------------------------------------------- #19: transient retry


def _content_with_source(*, ok: bool, failure_reason=None, kind="external_article", text=None):
    """Helper: build a Content with a single ContentSource of the requested shape."""
    from xbrain.models import Content, ContentSource

    return Content(
        fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
        sources=[
            ContentSource(
                kind=kind,
                url="https://example.com/p",
                ok=ok,
                failure_reason=failure_reason,
                text=text,
            )
        ],
    )


def _should_refetch(content, force):
    from xbrain.fetch import _should_refetch as impl

    return impl(content, force)


# --- Truth table on _should_refetch ---


def test_should_refetch_when_content_is_none():
    assert _should_refetch(None, force=False) is True
    assert _should_refetch(None, force=True) is True


def test_should_skip_successful_fetch_without_force():
    c = _content_with_source(ok=True, text="body")
    assert _should_refetch(c, force=False) is False


def test_should_refetch_successful_fetch_with_force():
    c = _content_with_source(ok=True, text="body")
    assert _should_refetch(c, force=True) is True


def test_should_refetch_all_transient_timeout():
    c = _content_with_source(ok=False, failure_reason="timeout")
    assert _should_refetch(c, force=False) is True


def test_should_refetch_all_transient_dns_error():
    c = _content_with_source(ok=False, failure_reason="dns_error")
    assert _should_refetch(c, force=False) is True


def test_should_skip_terminal_not_found_without_force():
    c = _content_with_source(ok=False, failure_reason="not_found")
    assert _should_refetch(c, force=False) is False


def test_should_skip_terminal_paywall_without_force():
    c = _content_with_source(ok=False, failure_reason="paywall")
    assert _should_refetch(c, force=False) is False


def test_should_skip_terminal_forbidden_without_force():
    c = _content_with_source(ok=False, failure_reason="forbidden")
    assert _should_refetch(c, force=False) is False


def test_should_skip_terminal_js_required_without_force():
    c = _content_with_source(ok=False, failure_reason="js_required")
    assert _should_refetch(c, force=False) is False


def test_should_skip_terminal_empty_content_without_force():
    c = _content_with_source(ok=False, failure_reason="empty_content")
    assert _should_refetch(c, force=False) is False


def test_should_refetch_terminal_with_force():
    c = _content_with_source(ok=False, failure_reason="not_found")
    assert _should_refetch(c, force=True) is True


def test_should_skip_mixed_transient_and_terminal():
    """Any terminal failure poisons the retry — the link is dead, retry is waste."""
    from xbrain.models import Content, ContentSource

    c = Content(
        fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
        sources=[
            ContentSource(kind="external_article", url="a", ok=False, failure_reason="timeout"),
            ContentSource(kind="external_article", url="b", ok=False, failure_reason="not_found"),
        ],
    )
    assert _should_refetch(c, force=False) is False


def test_should_skip_mixed_transient_and_success():
    """One source succeeded — there is good content here, do not re-hit."""
    from xbrain.models import Content, ContentSource

    c = Content(
        fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
        sources=[
            ContentSource(kind="external_article", url="a", ok=False, failure_reason="timeout"),
            ContentSource(kind="external_article", url="b", ok=True, text="got it"),
        ],
    )
    assert _should_refetch(c, force=False) is False


def test_should_skip_only_x_sources():
    """fetch_pending must not act on items whose only sources are x.com — that is fetch_x's job."""
    from xbrain.models import Content, ContentSource

    c = Content(
        fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
        sources=[
            ContentSource(
                kind="x_article",
                url="https://x.com/i/article/1",
                ok=False,
                failure_reason="timeout",
            ),
        ],
    )
    assert _should_refetch(c, force=False) is False


# --- Integration on fetch_pending ---


def _seed_with_failure(item_id, *, failure_reason):
    """Helper: an item that has already been fetched and failed once."""
    item = _item(item_id, ["https://example.com/p"])
    item.content = _content_with_source(ok=False, failure_reason=failure_reason)
    return item


def test_fetch_pending_retries_timeout_without_force():
    store = {"1": _seed_with_failure("1", failure_reason="timeout")}
    count = fetch_pending(store, extractor=_fake_extractor)
    assert count == 1
    # The retry overwrote the failure with a fresh, successful source
    src = store["1"].content.sources[0]
    assert src.ok and src.text is not None


def test_fetch_pending_skips_not_found_without_force():
    store = {"1": _seed_with_failure("1", failure_reason="not_found")}
    count = fetch_pending(store, extractor=_fake_extractor)
    assert count == 0
    # The recorded failure is preserved untouched
    src = store["1"].content.sources[0]
    assert (not src.ok) and src.failure_reason == "not_found"


def test_fetch_pending_force_refetches_not_found():
    store = {"1": _seed_with_failure("1", failure_reason="not_found")}
    assert fetch_pending(store, force=True, extractor=_fake_extractor) == 1


def test_fetch_pending_skips_transient_failures_outside_date_range():
    """The since/until filter applies on top of _should_refetch."""
    item = _seed_with_failure("1", failure_reason="timeout")
    item.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = {"1": item}
    count = fetch_pending(
        store,
        since=datetime(2026, 5, 1, tzinfo=timezone.utc),
        extractor=_fake_extractor,
    )
    assert count == 0
    # And the recorded failure is preserved
    assert store["1"].content.sources[0].failure_reason == "timeout"
