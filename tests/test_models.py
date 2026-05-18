# tests/test_models.py
from datetime import datetime, timezone

from xbrain.models import Author, Item, Link, State


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


def test_enrichment_has_primary_topic_and_no_note_worthiness():
    from datetime import datetime, timezone
    from xbrain.models import Enrichment

    e = Enrichment(
        enriched_at=datetime.now(timezone.utc),
        executor="api",
        summary="resumen",
        primary_topic="ai-coding",
        topics=["ai-coding", "ai-and-work"],
    )
    assert e.primary_topic == "ai-coding"
    assert not hasattr(e, "note_worthiness")


def test_topic_model_holds_slug_and_description():
    from xbrain.models import Topic

    t = Topic(slug="ai-coding", description="Using LLMs to write software.")
    assert t.slug == "ai-coding"


def test_topic_rejects_non_kebab_case_slug():
    import pytest
    from pydantic import ValidationError

    from xbrain.models import Topic

    for bad in ["AI Coding", "ai_coding", "-ai", "ai-", "ai--coding"]:
        with pytest.raises(ValidationError):
            Topic(slug=bad, description="d")


def test_item_has_optional_bookmark_folder():
    from datetime import datetime, timezone
    from xbrain.models import Author, Item

    base = dict(
        id="1",
        source="bookmark",
        url="https://x.com/a/status/1",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )
    assert Item(**base).bookmark_folder is None
    assert Item(**base, bookmark_folder="AI papers").bookmark_folder == "AI papers"


def test_content_source_carries_failure_evidence():
    from xbrain.models import ContentSource

    src = ContentSource(
        kind="external_article",
        url="https://example.com/x",
        ok=False,
        http_status=404,
        failure_reason="not_found",
        attempts=2,
        error="HTTP 404",
    )
    assert src.http_status == 404
    assert src.failure_reason == "not_found"
    assert src.attempts == 2


def test_content_source_failure_fields_default_cleanly():
    # An old-shape dict (no failure fields) must still validate — the existing
    # items.json predates these fields.
    from xbrain.models import ContentSource

    src = ContentSource.model_validate(
        {"kind": "external_article", "url": "https://e.com", "ok": True}
    )
    assert src.http_status is None
    assert src.failure_reason is None
    assert src.attempts == 0


def test_topic_page_model_round_trips():
    from datetime import datetime, timezone

    from xbrain.models import TopicPage

    page = TopicPage(
        slug="ai-coding",
        overview="Resumen del tema.",
        notes=["Nota uno.", "Nota dos."],
        synthesized_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
        post_count_at_synth=42,
    )
    restored = TopicPage.model_validate(page.model_dump(mode="json"))
    assert restored.slug == "ai-coding"
    assert restored.post_count_at_synth == 42
    assert restored.notes == ["Nota uno.", "Nota dos."]
