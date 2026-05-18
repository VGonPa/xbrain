# tests/test_vocab.py
from datetime import datetime, timezone

from xbrain.models import Author, Item
from xbrain.vocab import induce_vocab

from tests.conftest import FakeAnthropic


def _item(item_id: str, text: str) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text=text,
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )


def test_induce_vocab_runs_map_then_reduce():
    store = {str(i): _item(str(i), f"post {i}") for i in range(3)}
    client = FakeAnthropic(
        [
            {"candidates": [{"slug": "ai", "description": "AI."}]},
            {
                "topics": [
                    {"slug": "ai-coding", "description": "LLMs writing software."},
                    {"slug": "misc", "description": "Posts that do not fit a topic."},
                ]
            },
        ]
    )
    topics = induce_vocab(store, target_count=2, model="m", client=client, chunk_size=50)
    assert [t.slug for t in topics] == ["ai-coding", "misc"]
    assert len(client.messages.calls) == 2


def test_induce_vocab_chunks_the_corpus():
    store = {str(i): _item(str(i), f"post {i}") for i in range(5)}
    client = FakeAnthropic(
        [
            {"candidates": []},
            {"candidates": []},
            {"candidates": []},
            {"topics": [{"slug": "misc", "description": "Noise."}]},
        ]
    )
    induce_vocab(store, target_count=1, model="m", client=client, chunk_size=2)
    assert len(client.messages.calls) == 4  # 3 map chunks + 1 reduce


def test_induce_vocab_raises_when_map_response_has_no_candidates():
    import pytest

    # A truncated / malformed map response with no 'candidates' list must
    # surface as an error, not silently contribute nothing (BLOCKING B1).
    store = {str(i): _item(str(i), f"post {i}") for i in range(3)}
    client = FakeAnthropic(
        [
            {"wrong_key": "the map call failed"},
            {"topics": [{"slug": "misc", "description": "Noise."}]},
        ]
    )
    with pytest.raises(ValueError) as exc_info:
        induce_vocab(store, target_count=1, model="m", client=client, chunk_size=50)
    assert "candidates" in str(exc_info.value)
