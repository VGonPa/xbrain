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
    topics = induce_vocab(
        store, target_count=2, model="m", output_language="English", client=client, chunk_size=50
    )
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
    induce_vocab(
        store, target_count=1, model="m", output_language="English", client=client, chunk_size=2
    )
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
        induce_vocab(
            store,
            target_count=1,
            model="m",
            output_language="English",
            client=client,
            chunk_size=50,
        )
    assert "candidates" in str(exc_info.value)


def test_export_vocab_worksheet_writes_corpus_and_rubric(tmp_path):
    from datetime import datetime, timezone

    from xbrain.models import Author, Item
    from xbrain.vocab import export_vocab_worksheet

    def _item(i):
        return Item(
            id=str(i),
            source="bookmark",
            url=f"https://x.com/a/status/{i}",
            author=Author(handle="a", name="A"),
            text=f"post {i}",
            created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            captured_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        )

    store = {"1": _item(1), "2": _item(2)}
    path = tmp_path / "vocab-worksheet.json"
    export_vocab_worksheet(store, target_count=45, path=path, output_language="English")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["target_count"] == 45
    assert payload["corpus"] == ["post 1", "post 2"]
    assert "rubric" in payload and payload["rubric"]
    assert payload["topics"] == []


def test_import_vocab_worksheet_round_trips(tmp_path):
    import json

    from xbrain.vocab import import_vocab_worksheet

    path = tmp_path / "ws.json"
    path.write_text(json.dumps({"topics": [{"slug": "ai", "description": "d"}]}), encoding="utf-8")
    assert import_vocab_worksheet(path) == [{"slug": "ai", "description": "d"}]


def test_import_vocab_worksheet_rejects_non_list_topics(tmp_path):
    import json

    import pytest

    from xbrain.vocab import import_vocab_worksheet

    path = tmp_path / "ws.json"
    path.write_text(json.dumps({"topics": "no soy lista"}), encoding="utf-8")
    with pytest.raises(ValueError):
        import_vocab_worksheet(path)


def test_import_vocab_worksheet_missing_file_raises(tmp_path):
    import pytest

    from xbrain.vocab import import_vocab_worksheet

    with pytest.raises(FileNotFoundError):
        import_vocab_worksheet(tmp_path / "absent.json")


def test_import_vocab_worksheet_rejects_non_dict_root(tmp_path):
    import json

    import pytest

    from xbrain.vocab import import_vocab_worksheet

    # A worksheet whose top-level JSON is a list (not an object) must surface a
    # clean ValueError, not an uncaught AttributeError from `data.get(...)`.
    path = tmp_path / "ws.json"
    path.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    with pytest.raises(ValueError):
        import_vocab_worksheet(path)


def test_apply_vocab_worksheet_validates_topics():
    from xbrain.vocab import apply_vocab_worksheet

    valid, invalid = apply_vocab_worksheet(
        [
            {"slug": "ai-coding", "description": "Desarrollo con IA."},
            {"slug": "BAD SLUG", "description": "espacios y mayúsculas"},
            {"slug": "misc"},  # missing description
        ]
    )
    assert [t.slug for t in valid] == ["ai-coding"]
    assert {row[0] for row in invalid} == {"BAD SLUG", "misc"}


def test_apply_vocab_worksheet_rejects_duplicate_slugs():
    from xbrain.vocab import apply_vocab_worksheet

    valid, invalid = apply_vocab_worksheet(
        [
            {"slug": "ai", "description": "d1"},
            {"slug": "ai", "description": "d2"},
        ]
    )
    assert [t.slug for t in valid] == ["ai"]
    assert invalid and invalid[0][0] == "ai"
    assert any("duplicate" in e for e in invalid[0][1])


def test_apply_vocab_worksheet_rejects_non_dict_entry():
    from xbrain.vocab import apply_vocab_worksheet

    valid, invalid = apply_vocab_worksheet(["oops"])
    assert valid == []
    assert invalid and "not a JSON object" in invalid[0][1][0]


def test_apply_vocab_worksheet_rejects_unexpected_keys():
    from xbrain.vocab import apply_vocab_worksheet

    # `Topic` is lenient, so an entry with an extra key would otherwise be
    # accepted with the extra key silently dropped — reject it instead.
    valid, invalid = apply_vocab_worksheet(
        [
            {"slug": "ai", "description": "d", "notes": []},
            {"slug": "ml", "description": "clean sibling"},
        ]
    )
    assert [t.slug for t in valid] == ["ml"]
    assert invalid and invalid[0][0] == "ai"
    assert any("unexpected keys" in e for e in invalid[0][1])
