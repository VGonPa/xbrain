# tests/test_enrich.py
from datetime import datetime, timezone

from xbrain.enrich import apply_enrichment, items_pending_enrichment
from xbrain.models import Author, Enrichment, Item


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
    apply_enrichment(
        item, Enrichment(enriched_at=datetime.now(timezone.utc), executor="manual", summary="s")
    )
    assert item.enriched is not None
    assert item.enriched.summary == "s"
    assert items_pending_enrichment({"1": item}) == []


def test_enrich_with_executor_attaches_valid_judgments():
    from xbrain.enrich import enrich_with_executor
    from xbrain.executors.base import EnrichmentJudgment
    from xbrain.models import Topic

    store = {"1": _item("1"), "2": _item("2")}
    vocab = [Topic(slug="ai-coding", description="d"), Topic(slug="misc", description="d")]

    class _Fake:
        def enrich_items(self, items, vocab):
            return [
                EnrichmentJudgment(
                    item_id=i.id, summary="resumen", primary_topic="ai-coding", topics=["ai-coding"]
                )
                for i in items
            ]

    enriched, invalid = enrich_with_executor(store, _Fake(), vocab)
    assert enriched == 2 and invalid == []
    assert store["1"].enriched.primary_topic == "ai-coding"


def test_enrich_with_executor_rejects_invalid_judgment():
    from xbrain.enrich import enrich_with_executor
    from xbrain.executors.base import EnrichmentJudgment
    from xbrain.models import Topic

    store = {"1": _item("1")}

    class _Bad:
        def enrich_items(self, items, vocab):
            return [
                EnrichmentJudgment(
                    item_id="1", summary="r", primary_topic="not-in-vocab", topics=["not-in-vocab"]
                )
            ]

    enriched, invalid = enrich_with_executor(
        store, _Bad(), [Topic(slug="ai-coding", description="d")]
    )
    assert enriched == 0 and len(invalid) == 1
    assert store["1"].enriched is None


def test_apply_worksheet_judgments_attaches_valid_dicts():
    from xbrain.enrich import apply_worksheet_judgments
    from xbrain.models import Topic

    store = {"1": _item("1")}
    judgments = [{"item_id": "1", "summary": "s", "primary_topic": "misc", "topics": ["misc"]}]
    enriched, invalid = apply_worksheet_judgments(
        store, judgments, [Topic(slug="misc", description="d")]
    )
    assert enriched == 1 and invalid == []
    assert store["1"].enriched.executor == "claude-code"


def test_apply_worksheet_judgments_handles_null_topics():
    from xbrain.enrich import apply_worksheet_judgments
    from xbrain.models import Topic

    store = {"1": _item("1")}
    judgments = [{"item_id": "1", "summary": "s", "primary_topic": "misc", "topics": None}]
    enriched, invalid = apply_worksheet_judgments(
        store, judgments, [Topic(slug="misc", description="d")]
    )
    assert enriched == 0 and len(invalid) == 1
    assert store["1"].enriched is None


def test_apply_worksheet_judgments_rejects_invalid_executor():
    import pytest

    from xbrain.enrich import apply_worksheet_judgments
    from xbrain.models import Topic

    # A bad worksheet `executor` must be a clean up-front error, not an
    # uncaught pydantic.ValidationError raised mid-loop (BLOCKING B2).
    store = {"1": _item("1")}
    judgments = [{"item_id": "1", "summary": "s", "primary_topic": "misc", "topics": ["misc"]}]
    with pytest.raises(ValueError) as exc_info:
        apply_worksheet_judgments(
            store, judgments, [Topic(slug="misc", description="d")], executor_name="bogus-executor"
        )
    assert "invalid executor" in str(exc_info.value)
    assert store["1"].enriched is None


def test_apply_worksheet_judgments_reports_unknown_item_id():
    from xbrain.enrich import apply_worksheet_judgments
    from xbrain.models import Topic

    # A judgment that is structurally valid but names an item_id absent from
    # the store must surface in `invalid` with an "unknown item id" error —
    # the shared `_validate_and_attach` unknown-id branch.
    store = {"1": _item("1")}
    judgments = [{"item_id": "999", "summary": "s", "primary_topic": "misc", "topics": ["misc"]}]
    enriched, invalid = apply_worksheet_judgments(
        store, judgments, [Topic(slug="misc", description="d")]
    )
    assert enriched == 0 and len(invalid) == 1
    bad_id, errors = invalid[0]
    assert bad_id == "999"
    assert any("unknown item id" in e for e in errors)
    assert store["1"].enriched is None


def test_items_pending_respects_date_range():
    old_item = _item("1")
    old_item.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    new_item = _item("2")
    new_item.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = {"1": old_item, "2": new_item}
    pending = items_pending_enrichment(store, since=datetime(2023, 1, 1, tzinfo=timezone.utc))
    assert {i.id for i in pending} == {"2"}


def _enriched_at(item: Item, when: datetime) -> None:
    apply_enrichment(
        item,
        Enrichment(
            enriched_at=when, executor="api", summary="s", primary_topic="misc", topics=["misc"]
        ),
    )


def test_item_reenriched_when_content_fetched_after_enrichment():
    """A video bookmark enriched from its 2-line tweet, THEN given an `x_video`
    transcript (content re-fetched later), must re-appear as pending — otherwise
    it keeps topic "—" forever (the #44 re-enrichment trigger)."""
    from xbrain.models import Content, ContentSourceSuccess

    item = _item("1")
    _enriched_at(item, datetime(2026, 5, 16, tzinfo=timezone.utc))
    item.content = Content(
        fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),  # AFTER enrichment
        sources=[ContentSourceSuccess(kind="x_video", url="u", text="talk", has_speech=True)],
    )
    assert [i.id for i in items_pending_enrichment({"1": item})] == ["1"]


def test_item_not_reenriched_when_content_predates_enrichment():
    """The normal order (fetch → enrich) must NOT re-enrich: content already
    reflected in the enrichment stays processed."""
    from xbrain.models import Content, ContentSourceSuccess

    item = _item("1")
    item.content = Content(
        fetched_at=datetime(2026, 5, 15, tzinfo=timezone.utc),  # BEFORE enrichment
        sources=[ContentSourceSuccess(kind="external_article", url="u", text="body")],
    )
    _enriched_at(item, datetime(2026, 5, 16, tzinfo=timezone.utc))
    assert items_pending_enrichment({"1": item}) == []


def test_reenriched_item_settles_and_is_not_pending_again():
    """After re-enrichment, `enriched_at` is bumped past `content.fetched_at` so the
    item is NOT flagged pending forever (no infinite re-enrichment / cost churn).

    Mutation-sanity guard for the #44 merge gate: `_validate_and_attach` must stamp
    a fresh `enriched_at` on EVERY enrich. A mutation that preserves the prior
    `enriched_at` on re-enrichment would leave `content.fetched_at > enriched_at`
    permanently true, re-enriching the item on every run. This test fails RED
    against that mutation and GREEN against the correct code."""
    from xbrain.enrich import enrich_with_executor
    from xbrain.executors.base import EnrichmentJudgment
    from xbrain.models import Content, ContentSourceSuccess, Topic

    item = _item("1")
    _enriched_at(item, datetime(2026, 5, 16, tzinfo=timezone.utc))
    item.content = Content(
        fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),  # AFTER first enrich
        sources=[ContentSourceSuccess(kind="x_video", url="u", text="talk", has_speech=True)],
    )
    store = {"1": item}
    assert [i.id for i in items_pending_enrichment(store)] == ["1"]  # trigger fired

    class _Fake:
        def enrich_items(self, items, vocab):
            return [
                EnrichmentJudgment(
                    item_id=i.id, summary="s", primary_topic="ai-coding", topics=["ai-coding"]
                )
                for i in items
            ]

    enrich_with_executor(store, _Fake(), [Topic(slug="ai-coding", description="d")])
    settled = store["1"]
    assert items_pending_enrichment(store) == []  # settled — the load-bearing assertion
    # Convergence: the fresh enrichment timestamp is stamped strictly PAST the
    # fetch timestamp, which is *why* the item settles (not an incidental equality).
    assert settled.enriched is not None and settled.content is not None
    assert settled.enriched.enriched_at > settled.content.fetched_at
