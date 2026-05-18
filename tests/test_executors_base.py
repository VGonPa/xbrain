# tests/test_executors_base.py
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from xbrain.executors.base import EnrichmentExecutor, EnrichmentJudgment
from xbrain.models import Author, Item, Topic


def test_judgment_holds_summary_and_topics():
    j = EnrichmentJudgment(item_id="1", summary="s",
                           primary_topic="ai-coding", topics=["ai-coding"])
    assert j.item_id == "1"
    assert j.primary_topic == "ai-coding"


def test_judgment_rejects_empty_topics():
    # `topics` always carries the primary topic — an empty list is illegal.
    with pytest.raises(ValidationError):
        EnrichmentJudgment(item_id="1", summary="s",
                           primary_topic="ai-coding", topics=[])


def test_a_minimal_executor_satisfies_the_protocol():
    class Fake:
        def enrich_items(self, items, vocab):
            return [EnrichmentJudgment(item_id=i.id, summary="s",
                                       primary_topic="misc", topics=["misc"])
                    for i in items]

    executor: EnrichmentExecutor = Fake()
    item = Item(
        id="1", source="bookmark", url="https://x.com/a/status/1",
        author=Author(handle="a", name="A"), text="t",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )
    out = executor.enrich_items([item], [Topic(slug="misc", description="d")])
    assert [j.item_id for j in out] == ["1"]
    assert out[0].primary_topic == "misc"
