# tests/test_generate.py
from datetime import datetime, timezone
from pathlib import Path

from xkb.generate import generate
from xkb.models import Author, Item, Link


def _item(item_id: str, with_link: bool) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=f"Note {item_id}",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url="https://example.com/p", domain="example.com")] if with_link else [],
    )


def test_generate_creates_index_log_and_only_link_notes(tmp_path: Path):
    store = {"1": _item("1", with_link=True), "2": _item("2", with_link=False)}
    generate(store, tmp_path)
    assert (tmp_path / "_index.md").exists()
    assert (tmp_path / "log.md").exists()
    notes = list((tmp_path / "items").glob("*.md"))
    assert len(notes) == 1


def test_regeneration_preserves_user_content_after_marker(tmp_path: Path):
    store = {"1": _item("1", with_link=True)}
    generate(store, tmp_path)
    note = next((tmp_path / "items").glob("*.md"))
    note.write_text(note.read_text(encoding="utf-8") + "MI ANOTACION", encoding="utf-8")
    generate(store, tmp_path)
    assert "MI ANOTACION" in note.read_text(encoding="utf-8")


def test_log_lists_every_item(tmp_path: Path):
    store = {"1": _item("1", with_link=True), "2": _item("2", with_link=False)}
    generate(store, tmp_path)
    log = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "Note 1" in log
    assert "Note 2" in log
