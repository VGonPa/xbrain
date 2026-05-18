# tests/test_vocab.py
import json
from datetime import datetime, timezone

from xbrain.models import Author, Item
from xbrain.vocab import induce_vocab


def _item(item_id: str, text: str) -> Item:
    return Item(
        id=item_id, source="bookmark", url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"), text=text,
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )


class _FakeMessages:
    def __init__(self, payloads: list[dict]):
        self._payloads = list(payloads)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self._payloads.pop(0)

        class _Block:
            type = "text"
            text = json.dumps(payload)

        class _Resp:
            content = [_Block()]

        return _Resp()


class _FakeClient:
    def __init__(self, payloads):
        self.messages = _FakeMessages(payloads)


def test_induce_vocab_runs_map_then_reduce():
    store = {str(i): _item(str(i), f"post {i}") for i in range(3)}
    client = _FakeClient([
        {"candidates": [{"slug": "ai", "description": "AI."}]},
        {"topics": [
            {"slug": "ai-coding", "description": "LLMs writing software."},
            {"slug": "misc", "description": "Posts that do not fit a topic."}]},
    ])
    topics = induce_vocab(store, target_count=2, model="m", client=client,
                          chunk_size=50)
    assert [t.slug for t in topics] == ["ai-coding", "misc"]
    assert len(client.messages.calls) == 2


def test_induce_vocab_chunks_the_corpus():
    store = {str(i): _item(str(i), f"post {i}") for i in range(5)}
    client = _FakeClient([
        {"candidates": []}, {"candidates": []}, {"candidates": []},
        {"topics": [{"slug": "misc", "description": "Noise."}]},
    ])
    induce_vocab(store, target_count=1, model="m", client=client, chunk_size=2)
    assert len(client.messages.calls) == 4  # 3 map chunks + 1 reduce
