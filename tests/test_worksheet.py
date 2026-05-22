# tests/test_worksheet.py
import json
from datetime import datetime, timezone

from xbrain.models import Author, Item, Link, Topic
from xbrain.worksheet import export_worksheet, import_worksheet


def _item(item_id: str) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="post text",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url="https://arxiv.org/abs/1", domain="arxiv.org")],
        bookmark_folder="AI papers",
    )


VOCAB = [Topic(slug="misc", description="Noise.")]


def test_export_worksheet_writes_items_vocab_and_rubrics(tmp_path):
    path = tmp_path / "ws.json"
    export_worksheet([_item("1"), _item("2")], VOCAB, path, "claude-code", "English")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert {it["item_id"] for it in data["items"]} == {"1", "2"}
    assert data["items"][0]["bookmark_folder"] == "AI papers"
    assert data["items"][0]["links"][0]["domain"] == "arxiv.org"
    assert "topics" in data["rubrics"]
    assert [t["slug"] for t in data["vocab"]] == ["misc"]
    assert data["judgments"] == []
    # The rubrics shipped in the worksheet must already have `{language}`
    # substituted — the Claude Code session reads them as-is.
    assert "{language}" not in data["rubrics"]["summary"]
    assert "**Language:** English" in data["rubrics"]["summary"]


def test_export_worksheet_records_executor(tmp_path):
    path = tmp_path / "ws.json"
    export_worksheet([_item("1")], VOCAB, path, "manual", "English")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["executor"] == "manual"


def test_import_worksheet_reads_filled_judgments(tmp_path):
    path = tmp_path / "ws.json"
    export_worksheet([_item("1")], VOCAB, path, "claude-code", "English")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["judgments"] = [
        {"item_id": "1", "summary": "s", "primary_topic": "misc", "topics": ["misc"]}
    ]
    path.write_text(json.dumps(data), encoding="utf-8")
    executor, judgments = import_worksheet(path)
    assert executor == "claude-code"
    assert judgments[0]["item_id"] == "1"


def test_import_worksheet_reads_executor_back(tmp_path):
    path = tmp_path / "ws.json"
    export_worksheet([_item("1")], VOCAB, path, "manual", "English")
    executor, judgments = import_worksheet(path)
    assert executor == "manual"
    assert judgments == []


def test_import_worksheet_missing_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        import_worksheet(tmp_path / "nope.json")


def test_import_worksheet_rejects_non_list_judgments(tmp_path):
    import pytest

    # A worksheet whose `judgments` is not a list (e.g. an object) is a clean
    # up-front error, not an obscure failure when the loop tries to iterate it.
    path = tmp_path / "ws.json"
    path.write_text(json.dumps({"judgments": {}}), encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        import_worksheet(path)
    assert "must be a list" in str(exc_info.value)
