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
    apply_enrichment(item, Enrichment(
        enriched_at=datetime.now(timezone.utc), executor="manual", summary="s"))
    assert item.enriched is not None
    assert item.enriched.summary == "s"
    assert items_pending_enrichment({"1": item}) == []


def test_enrich_with_executor_attaches_valid_judgments():
    from xbrain.enrich import enrich_with_executor
    from xbrain.executors.base import EnrichmentJudgment
    from xbrain.models import Topic

    store = {"1": _item("1"), "2": _item("2")}
    vocab = [Topic(slug="ai-coding", description="d"),
             Topic(slug="misc", description="d")]

    class _Fake:
        def enrich_items(self, items, vocab):
            return [EnrichmentJudgment(item_id=i.id, summary="resumen",
                                       primary_topic="ai-coding",
                                       topics=["ai-coding"]) for i in items]

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
            return [EnrichmentJudgment(item_id="1", summary="r",
                                       primary_topic="not-in-vocab",
                                       topics=["not-in-vocab"])]

    enriched, invalid = enrich_with_executor(
        store, _Bad(), [Topic(slug="ai-coding", description="d")])
    assert enriched == 0 and len(invalid) == 1
    assert store["1"].enriched is None


def test_apply_worksheet_judgments_attaches_valid_dicts():
    from xbrain.enrich import apply_worksheet_judgments
    from xbrain.models import Topic

    store = {"1": _item("1")}
    judgments = [{"item_id": "1", "summary": "s", "primary_topic": "misc",
                  "topics": ["misc"]}]
    enriched, invalid = apply_worksheet_judgments(
        store, judgments, [Topic(slug="misc", description="d")])
    assert enriched == 1 and invalid == []
    assert store["1"].enriched.executor == "claude-code"


def test_apply_worksheet_judgments_handles_null_topics():
    from xbrain.enrich import apply_worksheet_judgments
    from xbrain.models import Topic

    store = {"1": _item("1")}
    judgments = [{"item_id": "1", "summary": "s", "primary_topic": "misc",
                  "topics": None}]
    enriched, invalid = apply_worksheet_judgments(
        store, judgments, [Topic(slug="misc", description="d")])
    assert enriched == 0 and len(invalid) == 1
    assert store["1"].enriched is None


def test_apply_worksheet_judgments_rejects_invalid_executor():
    import pytest

    from xbrain.enrich import apply_worksheet_judgments
    from xbrain.models import Topic

    # A bad worksheet `executor` must be a clean up-front error, not an
    # uncaught pydantic.ValidationError raised mid-loop (BLOCKING B2).
    store = {"1": _item("1")}
    judgments = [{"item_id": "1", "summary": "s", "primary_topic": "misc",
                  "topics": ["misc"]}]
    with pytest.raises(ValueError) as exc_info:
        apply_worksheet_judgments(
            store, judgments, [Topic(slug="misc", description="d")],
            executor_name="bogus-executor")
    assert "invalid executor" in str(exc_info.value)
    assert store["1"].enriched is None


def test_apply_worksheet_judgments_reports_unknown_item_id():
    from xbrain.enrich import apply_worksheet_judgments
    from xbrain.models import Topic

    # A judgment that is structurally valid but names an item_id absent from
    # the store must surface in `invalid` with an "unknown item id" error —
    # the shared `_validate_and_attach` unknown-id branch.
    store = {"1": _item("1")}
    judgments = [{"item_id": "999", "summary": "s", "primary_topic": "misc",
                  "topics": ["misc"]}]
    enriched, invalid = apply_worksheet_judgments(
        store, judgments, [Topic(slug="misc", description="d")])
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
    pending = items_pending_enrichment(
        store, since=datetime(2023, 1, 1, tzinfo=timezone.utc)
    )
    assert {i.id for i in pending} == {"2"}
