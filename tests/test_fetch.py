# tests/test_fetch.py
import socket
import urllib.error
from datetime import datetime, timezone

from xbrain.enrich import items_pending_enrichment
from xbrain.fetch import (
    FetchFailure,
    FetchResult,
    FetchSuccess,
    _categorize_url_error,
    _probe_status,
    _reason_for_status,
    _should_refetch,
    _sources_materially_equal,
    fetch_item,
    fetch_pending,
    trafilatura_extract,
)
from xbrain.models import (
    Author,
    Content,
    ContentSourceFailure,
    ContentSourceSuccess,
    Enrichment,
    Item,
    Link,
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
        links=[Link(url=u, domain="d") for u in urls],
    )


def _fake_extractor(url: str) -> FetchResult:
    return FetchSuccess(title="Título", text=f"cuerpo de {url}", http_status=200)


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
    assert isinstance(result, FetchSuccess)
    assert result.text == "el cuerpo"
    assert result.http_status == 200


def test_trafilatura_extract_empty_content_when_no_article():
    result = trafilatura_extract(
        "https://e.com/a",
        fetch=lambda url: "<html></html>",
        extract=lambda html: None,
        prober=lambda url: (None, None, ""),
    )
    assert isinstance(result, FetchFailure)
    assert result.failure_reason == "empty_content"


def test_trafilatura_extract_probes_when_download_fails():
    result = trafilatura_extract(
        "https://e.com/a",
        fetch=lambda url: None,
        extract=lambda html: None,
        prober=lambda url: (404, "not_found", "HTTP 404"),
    )
    assert isinstance(result, FetchFailure)
    assert result.http_status == 404
    assert result.failure_reason == "not_found"


def test_fetch_item_extracts_external_articles():
    content = fetch_item(_item("1", ["https://example.com/p"]), _fake_extractor)
    source = content.sources[0]
    assert isinstance(source, ContentSourceSuccess)
    assert source.kind == "external_article"
    assert source.text == "cuerpo de https://example.com/p"


def test_fetch_item_skips_x_urls():
    # x.com links are handled by fetch_x.fetch_x_articles, not fetch_item.
    content = fetch_item(_item("1", ["https://x.com/foo/status/9"]), _fake_extractor)
    assert content.sources == []


def test_fetch_item_records_failure_evidence():
    content = fetch_item(
        _item("1", ["https://example.com/p"]),
        lambda url: FetchFailure(http_status=404, failure_reason="not_found", error="HTTP 404"),
    )
    source = content.sources[0]
    assert isinstance(source, ContentSourceFailure)
    assert source.http_status == 404
    assert source.failure_reason == "not_found"


def test_fetch_item_isolates_extractor_exception():
    def _raising(url):
        raise RuntimeError("boom")

    content = fetch_item(_item("1", ["https://example.com/p"]), _raising)
    assert len(content.sources) == 1
    source = content.sources[0]
    assert isinstance(source, ContentSourceFailure)
    assert "boom" in (source.error or "")


def test_fetch_item_preserves_non_external_sources_on_refetch():
    item = _item("1", ["https://example.com/p"])
    item.content = Content(
        fetched_at=datetime.now(timezone.utc),
        sources=[ContentSourceSuccess(kind="thread", url="u", text="hilo")],
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
    assert isinstance(result, FetchSuccess)
    assert result.text == "el cuerpo"
    assert result.title == "T"


def test_extract_article_falls_back_to_firecrawl_on_js_required():
    from xbrain.fetch import extract_article

    def primary(url):
        return FetchFailure(failure_reason="js_required", error="js", attempts=1)

    def firecrawl(url):
        return FetchSuccess(text="rescatado por firecrawl", attempts=1)

    result = extract_article("https://e.com/a", primary=primary, firecrawl=firecrawl)
    assert isinstance(result, FetchSuccess)
    assert result.text == "rescatado por firecrawl"
    assert result.attempts == 2


def test_extract_article_keeps_evidence_when_firecrawl_unavailable():
    from xbrain.fetch import extract_article

    def primary(url):
        return FetchFailure(failure_reason="js_required", error="js", attempts=1)

    def firecrawl(url):
        return None

    result = extract_article("https://e.com/a", primary=primary, firecrawl=firecrawl)
    assert isinstance(result, FetchFailure)
    assert result.failure_reason == "js_required"
    assert result.attempts == 1


def test_extract_article_does_not_retry_hard_failures():
    from xbrain.fetch import extract_article

    # A 404 is definitive — Firecrawl must not even be called.
    calls = []

    def primary(url):
        return FetchFailure(http_status=404, failure_reason="not_found", attempts=1)

    def firecrawl(url):
        calls.append(url)
        return FetchSuccess(text="x")

    result = extract_article("https://e.com/a", primary=primary, firecrawl=firecrawl)
    assert isinstance(result, FetchFailure)
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
    assert isinstance(result, FetchFailure)
    assert "Firecrawl falló" in (result.error or "")


def test_firecrawl_extract_detects_error_envelope(monkeypatch):
    import json

    from xbrain.fetch import _firecrawl_extract

    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
    payload = {"success": False, "error": "rate limited"}

    def opener(req, timeout=0):
        return _CtxResp(json.dumps(payload).encode())

    result = _firecrawl_extract("https://e.com/a", opener=opener)
    assert isinstance(result, FetchFailure)
    assert "rate limited" in (result.error or "")


def test_extract_article_merges_firecrawl_error_when_both_fail():
    from xbrain.fetch import extract_article

    def primary(url):
        return FetchFailure(failure_reason="js_required", error="js", attempts=1)

    def firecrawl(url):
        # Firecrawl reachable but returned no text — carries its own evidence.
        return FetchFailure(error="Firecrawl no devolvió contenido.", attempts=1)

    result = extract_article("https://e.com/a", primary=primary, firecrawl=firecrawl)
    assert isinstance(result, FetchFailure)
    assert result.attempts == 2
    assert "Firecrawl: Firecrawl no devolvió contenido." in (result.error or "")


# --------------------------------------------------------------------- #19: transient retry


def _content_with_source(*, ok: bool, failure_reason="timeout", kind="external_article", text=None):
    """Helper: build a Content with a single ContentSource of the requested shape.

    `ok=True` builds a `ContentSourceSuccess` (text required); `ok=False`
    builds a `ContentSourceFailure` (failure_reason required, default
    ``"timeout"`` — the transient bucket).
    """
    if ok:
        source = ContentSourceSuccess(kind=kind, url="https://example.com/p", text=text or "body")
    else:
        source = ContentSourceFailure(
            kind=kind, url="https://example.com/p", failure_reason=failure_reason
        )
    return Content(
        fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
        sources=[source],
    )


def _refetch(content, force: bool, *, urls=("https://example.com/p",)) -> bool:
    """`_should_refetch` asks the question of the ITEM, not of a bare `Content`.

    It has to: "content exists" is not "the links were fetched" — the extract stage
    stamps the quoted post, `digest-video` a transcript. So the decision reads the
    item's LINKS. These wrappers keep the truth table below stated in terms of the
    content, with the item's links held at the default single non-x link.
    """
    item = _item("1", list(urls))
    item.content = content
    return _should_refetch(item, force=force)


# --- Truth table on _should_refetch ---


def test_should_refetch_when_content_is_none():
    assert _refetch(None, force=False) is True
    assert _refetch(None, force=True) is True


def test_should_skip_successful_fetch_without_force():
    c = _content_with_source(ok=True, text="body")
    assert _refetch(c, force=False) is False


def test_should_refetch_successful_fetch_with_force():
    c = _content_with_source(ok=True, text="body")
    assert _refetch(c, force=True) is True


def test_should_refetch_all_transient_timeout():
    c = _content_with_source(ok=False, failure_reason="timeout")
    assert _refetch(c, force=False) is True


def test_should_refetch_all_transient_dns_error():
    c = _content_with_source(ok=False, failure_reason="dns_error")
    assert _refetch(c, force=False) is True


def test_should_skip_terminal_not_found_without_force():
    c = _content_with_source(ok=False, failure_reason="not_found")
    assert _refetch(c, force=False) is False


def test_should_skip_terminal_paywall_without_force():
    c = _content_with_source(ok=False, failure_reason="paywall")
    assert _refetch(c, force=False) is False


def test_should_skip_terminal_forbidden_without_force():
    c = _content_with_source(ok=False, failure_reason="forbidden")
    assert _refetch(c, force=False) is False


def test_should_skip_terminal_js_required_without_force():
    c = _content_with_source(ok=False, failure_reason="js_required")
    assert _refetch(c, force=False) is False


def test_should_skip_terminal_empty_content_without_force():
    c = _content_with_source(ok=False, failure_reason="empty_content")
    assert _refetch(c, force=False) is False


def test_should_refetch_terminal_with_force():
    c = _content_with_source(ok=False, failure_reason="not_found")
    assert _refetch(c, force=True) is True


def test_should_skip_mixed_transient_and_terminal():
    """Any terminal failure poisons the retry — the link is dead, retry is waste."""
    c = Content(
        fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
        sources=[
            ContentSourceFailure(kind="external_article", url="a", failure_reason="timeout"),
            ContentSourceFailure(kind="external_article", url="b", failure_reason="not_found"),
        ],
    )
    assert _refetch(c, force=False) is False


def test_should_skip_mixed_transient_and_success():
    """One source succeeded — there is good content here, do not re-hit."""
    c = Content(
        fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
        sources=[
            ContentSourceFailure(kind="external_article", url="a", failure_reason="timeout"),
            ContentSourceSuccess(kind="external_article", url="b", text="got it"),
        ],
    )
    assert _refetch(c, force=False) is False


def test_should_skip_only_x_sources():
    """fetch_pending must not act on items whose only sources are x.com — that is fetch_x's job."""
    c = Content(
        fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
        sources=[
            ContentSourceFailure(
                kind="x_article",
                url="https://x.com/i/article/1",
                failure_reason="timeout",
            ),
        ],
    )
    assert _refetch(c, force=False, urls=("https://x.com/i/article/1",)) is False


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
    assert isinstance(src, ContentSourceSuccess)
    assert src.text


def test_fetch_pending_skips_not_found_without_force():
    store = {"1": _seed_with_failure("1", failure_reason="not_found")}
    count = fetch_pending(store, extractor=_fake_extractor)
    assert count == 0
    # The recorded failure is preserved untouched
    src = store["1"].content.sources[0]
    assert isinstance(src, ContentSourceFailure)
    assert src.failure_reason == "not_found"


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
    src = store["1"].content.sources[0]
    assert isinstance(src, ContentSourceFailure)
    assert src.failure_reason == "timeout"


# --- Additional fixes from PR #26 review pipeline ---


def test_should_refetch_legacy_uncategorized_failure_migrates_to_transient():
    """A pre-#20 record with `ok=False, failure_reason=None` (anomalous —
    uncategorised) is normalised on read to `failure_reason="unknown_error"`
    by the legacy-shape validator, so the next `fetch_pending` run retries it
    automatically. This preserves the #19 behaviour (uncategorised failures
    get one auto-retry) under the new tagged-union shape, even though after
    the refactor a new failure record can no longer have a `None` reason.
    """
    from xbrain.models import ContentSourceAdapter

    src = ContentSourceAdapter.validate_python(
        {
            "kind": "external_article",
            "url": "https://e.com/p",
            "ok": False,
            "failure_reason": None,
            "error": "anomalous failure",
        }
    )
    assert isinstance(src, ContentSourceFailure)
    # migrator picked the transient `unknown_error` bucket (not `timeout`,
    # which would mislabel the actual cause).
    assert src.failure_reason == "unknown_error"
    c = Content(fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc), sources=[src])
    assert _refetch(c, force=False) is True


def test_should_refetch_external_transient_alongside_xcom_source():
    """`_should_refetch` filters to external_article sources before deciding.
    An item with a transient-failed external_article PLUS an x.com source
    (any state) should retry on the external — the x.com source is fetch_x's
    job and must not block the external retry.
    """
    c = Content(
        fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
        sources=[
            ContentSourceFailure(
                kind="external_article",
                url="https://ext.com/a",
                failure_reason="timeout",
            ),
            ContentSourceFailure(
                kind="x_article",
                url="https://x.com/i/article/9",
                failure_reason="not_found",
            ),
        ],
    )
    assert _refetch(c, force=False) is True


def test_fetch_pending_replaces_sources_does_not_append():
    """PRD §5 invariant: a re-fetch *replaces* external_article sources, never
    appends. A transient-failed item with 1 source must still have 1 source
    after the retry (overwritten), not 2.
    """
    item = _item("1", ["https://example.com/p"])
    item.content = _content_with_source(ok=False, failure_reason="timeout")
    assert len(item.content.sources) == 1
    store = {"1": item}
    fetch_pending(store, extractor=_fake_extractor)
    assert len(store["1"].content.sources) == 1
    # And the lone source is now the fresh successful fetch
    src = store["1"].content.sources[0]
    assert isinstance(src, ContentSourceSuccess)
    assert src.text


def test_extractor_exception_persists_as_transient_failure():
    """An uncaught extractor exception must persist as a TRANSIENT failure
    (`unknown_error`), so `_should_refetch` retries it on the next run.

    Regression guard: pre-#20 the bare-except in fetch_item wrote
    `failure_reason=None`, and `_should_refetch` (post-#19) treated `None`
    as transient. After #20 introduced the discriminated union, the field
    became required — the temporary fix bucketed to `empty_content`, which
    is TERMINAL and silently broke #19's invariant. This test pins the
    correct behaviour: uncaught exceptions stay self-healing.
    """

    def _raising(url):
        raise RuntimeError("network blip simulated")

    content = fetch_item(_item("1", ["https://example.com/p"]), _raising)
    assert len(content.sources) == 1
    failure = content.sources[0]
    assert isinstance(failure, ContentSourceFailure)
    assert failure.failure_reason == "unknown_error"
    assert "network blip" in failure.error
    # And `_should_refetch` correctly retries on the next run
    assert _refetch(content, force=False) is True


def test_content_source_from_uncategorised_failure_is_transient():
    """`_content_source_from` is the public helper that maps an in-memory
    `FetchFailure` to a persisted `ContentSourceFailure`. A `FetchFailure`
    with `failure_reason=None` must fall back to `unknown_error` (transient),
    NOT `empty_content` (terminal).
    """
    from xbrain.fetch import FetchFailure, _content_source_from

    failure_result = FetchFailure(
        failure_reason=None,
        error="something went wrong",
        attempts=1,
    )
    src = _content_source_from("https://example.com/p", failure_result)
    assert isinstance(src, ContentSourceFailure)
    assert src.failure_reason == "unknown_error"


# --- Re-enrich hygiene: `fetched_at` advances only on a MATERIAL change (#44) ---
#
# `fetch_pending` re-fetches a persistently-failing transient link on EVERY run
# (its refetch decision keys on source STATE, not on `fetched_at`). If each
# identical re-fetch bumped `content.fetched_at`, the item would re-trip
# `enrich._needs_reenrichment` forever — one wasted, identical LLM call per stuck
# item per cycle. The fix: `fetch_item` keeps the prior `fetched_at` when the
# re-fetched source set is materially unchanged, and advances it only on a real
# content change (a failure that becomes a success, changed text, a changed
# failure reason). `attempts`/`error` churn is NOT a material change.

_T0 = datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)
_T1 = datetime(2026, 5, 10, 13, 0, tzinfo=timezone.utc)  # enrich, AFTER first fetch
_T2 = datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc)  # a later cycle's clock


def _failing_extractor(url: str) -> FetchResult:
    return FetchFailure(failure_reason="timeout", error="timed out")


def _enriched(item: Item, when: datetime) -> None:
    item.enriched = Enrichment(
        enriched_at=when, executor="api", summary="s", primary_topic="misc", topics=["misc"]
    )


def test_materially_equal_ignores_attempts_and_error_churn():
    a = ContentSourceFailure(
        kind="external_article", url="u", failure_reason="timeout", error="x", attempts=1
    )
    b = ContentSourceFailure(
        kind="external_article",
        url="u",
        failure_reason="timeout",
        error="totally other",
        attempts=2,
    )
    assert _sources_materially_equal([a], [b]) is True


def test_materially_equal_detects_failure_reason_change():
    a = ContentSourceFailure(kind="external_article", url="u", failure_reason="timeout")
    b = ContentSourceFailure(kind="external_article", url="u", failure_reason="dns_error")
    assert _sources_materially_equal([a], [b]) is False


def test_materially_equal_detects_failure_to_success():
    a = ContentSourceFailure(kind="external_article", url="u", failure_reason="timeout")
    b = ContentSourceSuccess(kind="external_article", url="u", text="body")
    assert _sources_materially_equal([a], [b]) is False


def test_materially_equal_detects_text_change():
    a = ContentSourceSuccess(kind="external_article", url="u", text="old body")
    b = ContentSourceSuccess(kind="external_article", url="u", text="new body")
    assert _sources_materially_equal([a], [b]) is False


def test_materially_equal_detects_title_change():
    """`title` is rendered into the enrich prompt (`Linked article ({title})`), so a
    changed title with identical body IS a material change — must re-enrich."""
    a = ContentSourceSuccess(kind="external_article", url="u", title="Old", text="body")
    b = ContentSourceSuccess(kind="external_article", url="u", title="New", text="body")
    assert _sources_materially_equal([a], [b]) is False


def test_materially_equal_detects_title_appearing():
    """A title going from absent (None) to a real value is a material change too."""
    a = ContentSourceSuccess(kind="external_article", url="u", title=None, text="body")
    b = ContentSourceSuccess(kind="external_article", url="u", title="Now titled", text="body")
    assert _sources_materially_equal([a], [b]) is False


def test_materially_equal_detects_http_status_change():
    """Two failures with the same `failure_reason` but different `http_status`
    (404 vs 410) are NOT equal — the broken-link render surfaces the status, so it
    is material evidence."""
    a = ContentSourceFailure(
        kind="external_article", url="u", failure_reason="not_found", http_status=404
    )
    b = ContentSourceFailure(
        kind="external_article", url="u", failure_reason="not_found", http_status=410
    )
    assert _sources_materially_equal([a], [b]) is False


def test_materially_equal_xvideo_frames_deterministic_and_material():
    """`fetch_item` passes an `x_video` source through by identity (never rebuilds
    it), so the model-derived signature must NOT spuriously flag it. Two distinct
    `x_video` instances with identical field values — transcript, `has_speech`,
    `language`, and `frames` — self-match (the JSON dump is deterministic, which is
    what makes the pass-through case safe); a changed frame description IS a real
    change."""
    from xbrain.models import VideoFrame

    def _video(desc: str) -> ContentSourceSuccess:
        return ContentSourceSuccess(
            kind="x_video",
            url="u",
            text="transcript",
            has_speech=True,
            language="en",
            frames=[VideoFrame(timestamp=1.5, local_path="1/frames/0.jpg", description=desc)],
        )

    assert _sources_materially_equal([_video("slide")], [_video("slide")]) is True
    assert _sources_materially_equal([_video("slide")], [_video("different")]) is False


def test_fetch_item_preserves_fetched_at_on_unchanged_failure_refetch():
    """A re-fetch that reproduces the same transient failure must NOT advance
    `fetched_at` — nothing about the content changed."""
    item = _item("1", ["https://example.com/p"])
    item.content = fetch_item(item, _failing_extractor, now=lambda: _T0)
    assert item.content.fetched_at == _T0
    refetched = fetch_item(item, _failing_extractor, now=lambda: _T1)
    assert refetched.fetched_at == _T0  # preserved, not bumped to _T1


def test_fetch_item_advances_fetched_at_when_failure_becomes_success():
    """A re-fetch that turns a failure into real text IS a material change."""
    item = _item("1", ["https://example.com/p"])
    item.content = fetch_item(item, _failing_extractor, now=lambda: _T0)
    refetched = fetch_item(item, _fake_extractor, now=lambda: _T1)
    assert refetched.fetched_at == _T1  # advanced


def test_dead_link_persistent_failure_not_reenriched_on_next_cycle():
    """Contract #1: fetch+enrich once, then re-fetch to the SAME transient
    failure → the item is NOT pending-for-enrichment again, and `fetched_at`
    is preserved (no per-cycle LLM churn on a stuck dead link)."""
    store = {"1": _item("1", ["https://slow.example/x"])}
    assert fetch_pending(store, extractor=_failing_extractor, now=lambda: _T0) == 1
    item = store["1"]
    _enriched(item, _T1)  # normal fetch→enrich order: enriched_at after fetched_at
    assert items_pending_enrichment(store) == []  # settled after the first enrich

    # Next cycle: the transient link still fails identically. It IS re-fetched
    # (pre-existing network retry — left alone), but nothing changed.
    assert fetch_pending(store, extractor=_failing_extractor, now=lambda: _T2) == 1
    assert item.content is not None
    assert item.content.fetched_at == _T0  # NOT advanced to _T2
    assert items_pending_enrichment(store) == []  # NOT re-flagged pending


def test_transient_failure_then_success_is_reenriched():
    """Contract #2: a transient failure that later succeeds with new text must
    re-enrich (real new content)."""
    store = {"1": _item("1", ["https://flaky.example/x"])}
    assert fetch_pending(store, extractor=_failing_extractor, now=lambda: _T0) == 1
    item = store["1"]
    _enriched(item, _T1)
    assert items_pending_enrichment(store) == []  # settled

    # The link finally works: real text = material change → `fetched_at` advances
    # past `enriched_at`, re-flagging the item.
    assert fetch_pending(store, extractor=_fake_extractor, now=lambda: _T2) == 1
    assert item.content is not None
    assert item.content.fetched_at == _T2  # advanced
    assert [i.id for i in items_pending_enrichment(store)] == ["1"]  # re-enriched


def test_force_refetch_unchanged_success_does_not_reenrich():
    """Contract (option a): a `--force` re-fetch of an already-successful item
    whose content is unchanged is NOT a material change — `fetched_at` is
    preserved and the item is not re-enriched."""
    store = {"1": _item("1", ["https://example.com/p"])}
    assert fetch_pending(store, extractor=_fake_extractor, now=lambda: _T0) == 1
    item = store["1"]
    _enriched(item, _T1)
    assert items_pending_enrichment(store) == []  # settled

    # `--force` re-fetches (state says skip, force overrides), but identical text
    # is not new content.
    assert fetch_pending(store, force=True, extractor=_fake_extractor, now=lambda: _T2) == 1
    assert item.content is not None
    assert item.content.fetched_at == _T0  # preserved despite --force
    assert items_pending_enrichment(store) == []  # not re-enriched


# --- A non-link source must never mask an unfetched link ---
#
# Since the quoted post is parsed at EXTRACT time, a quote-tweet arrives at `fetch`
# with `content` ALREADY stamped — but with no `external_article` source. Reading
# "content exists" as "nothing left to fetch" would silently never fetch the article
# of every quote-tweet that also links out. 35% of the corpus quotes.

_QUOTED = ContentSourceSuccess(
    kind="quoted_tweet",
    url="https://x.com/karpathy/status/999",
    text="I am leaving OpenAI.",
    author=Author(handle="karpathy", name="Andrej Karpathy"),
)


def _item_with(sources, *, urls=("https://example.com/a",)) -> Item:
    return Item(
        id="1",
        source="bookmark",
        url="https://x.com/a/status/1",
        author=Author(handle="a", name="A"),
        text="Read this and you'll understand better this career move",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url=u, domain="example.com") for u in urls],
        quoted_id="999",
        content=(
            Content(fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc), sources=list(sources))
            if sources is not None
            else None
        ),
    )


def test_a_quoted_source_does_not_mask_an_unfetched_link():
    """The regression guard: content stamped by the EXTRACT stage (a quoted post) is
    not evidence that the item's LINK was ever fetched."""
    assert _should_refetch(_item_with([_QUOTED]), force=False) is True


def test_fetch_pending_still_fetches_the_article_of_a_quote_tweet():
    """End to end: the article of a quote-tweet is fetched, and the quoted post
    SURVIVES the fetch (only `external_article` sources are rebuilt)."""
    item = _item_with([_QUOTED])
    store = {"1": item}

    fetched = fetch_pending(store, extractor=lambda url: FetchSuccess(title="T", text="body"))

    assert fetched == 1
    kinds = sorted(s.kind for s in store["1"].content.sources)
    assert kinds == ["external_article", "quoted_tweet"]
    assert _QUOTED in store["1"].content.sources


def test_an_item_whose_links_are_all_fetched_is_not_refetched_despite_a_quote():
    """The other arm — the fix must not turn into a re-fetch-forever loop."""
    item = _item_with(
        [
            _QUOTED,
            ContentSourceSuccess(kind="external_article", url="https://example.com/a", text="b"),
        ]
    )
    assert _should_refetch(item, force=False) is False


def test_an_item_with_only_x_links_and_a_quote_is_not_fetched():
    """x.com links are `fetch_x`'s job — a quoted source must not drag them here."""
    item = _item_with([_QUOTED], urls=("https://x.com/b/status/2",))
    assert _should_refetch(item, force=False) is False
