# tests/test_generate.py
from datetime import datetime, timezone
from pathlib import Path

from xbrain.generate import _slugify, generate
from xbrain.models import Author, Item, Link


def _item(item_id: str, with_link: bool, text: str | None = None) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=text if text is not None else f"Note {item_id}",
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


def test_slugify_handles_edge_cases():
    slug = _slugify("Café del Día")
    assert slug == slug.lower()
    assert slug.isascii()
    assert slug == "cafe-del-dia"
    assert _slugify("") == "item"
    assert _slugify("!!!") == "item"
    long_slug = _slugify("a" * 200)
    assert len(long_slug) <= 60


def test_regeneration_replaces_generated_block(tmp_path: Path):
    generate({"1": _item("1", with_link=True, text="Original text")}, tmp_path)
    generate({"1": _item("1", with_link=True, text="Updated text")}, tmp_path)
    notes = list((tmp_path / "items").glob("*.md"))
    assert len(notes) == 1  # the stale-title note is migrated, not orphaned
    content = notes[0].read_text(encoding="utf-8")
    assert "Updated text" in content
    assert "Original text" not in content


def test_missing_end_marker_preserves_file(tmp_path: Path):
    store = {"1": _item("1", with_link=True)}
    generate(store, tmp_path)
    note = next((tmp_path / "items").glob("*.md"))
    note.write_text("contenido del usuario sin marcadores", encoding="utf-8")
    generate(store, tmp_path)
    assert "contenido del usuario sin marcadores" in note.read_text(encoding="utf-8")


def test_note_filenames_do_not_collide(tmp_path: Path):
    store = {
        "1001": _item("1001", with_link=True, text="Mismo titulo"),
        "2002": _item("2002", with_link=True, text="Mismo titulo"),
    }
    generate(store, tmp_path)
    notes = list((tmp_path / "items").glob("*.md"))
    assert len(notes) == 2


def test_note_filenames_unique_for_similar_ids(tmp_path: Path):
    # Same date, identical slug, ids sharing their last 6 characters.
    # Keying filenames on the full id keeps both notes distinct.
    store = {
        "1000001": _item("1000001", with_link=True, text="Mismo titulo"),
        "2000001": _item("2000001", with_link=True, text="Mismo titulo"),
    }
    generate(store, tmp_path)
    notes = list((tmp_path / "items").glob("*.md"))
    assert len({note.name for note in notes}) == 2


def test_note_has_frontmatter(tmp_path: Path):
    generate({"1": _item("1", with_link=True)}, tmp_path)
    note = next((tmp_path / "items").glob("*.md"))
    content = note.read_text(encoding="utf-8")
    assert "id:" in content
    assert "source:" in content
    assert "tags: [x-knowledge" in content


def test_frontmatter_includes_topics_and_folder_as_tags(tmp_path):
    from datetime import datetime, timezone
    from xbrain.generate import generate
    from xbrain.models import Author, Enrichment, Item, Link

    item = Item(
        id="1", source="bookmark", url="https://x.com/a/status/1",
        author=Author(handle="a", name="A"), text="t",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url="https://arxiv.org/abs/1", domain="arxiv.org")],
        bookmark_folder="AI papers",
        enriched=Enrichment(
            enriched_at=datetime.now(timezone.utc), executor="api",
            summary="s", primary_topic="ai-coding",
            topics=["ai-coding", "ai-and-work"]),
    )
    generate({"1": item}, tmp_path)
    note = next((tmp_path / "items").glob("*-1.md")).read_text(encoding="utf-8")
    assert "ai-coding" in note and "ai-and-work" in note
    assert "ai-papers" in note          # folder, slugified, as a tag
    assert "bookmark_folder: AI papers" in note


def test_generate_since_until_filters_item_notes(tmp_path: Path):
    old_item = _item("1", with_link=True, text="Old note")
    old_item.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    new_item = _item("2", with_link=True, text="New note")
    new_item.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = {"1": old_item, "2": new_item}
    generate(store, tmp_path, since=datetime(2023, 1, 1, tzinfo=timezone.utc))
    notes = list((tmp_path / "items").glob("*.md"))
    assert len(notes) == 1
    assert "New note" in notes[0].read_text(encoding="utf-8")
    log = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "Old note" in log
    assert "New note" in log
