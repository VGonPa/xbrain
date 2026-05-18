# tests/test_notes_io.py
from datetime import datetime, timezone

from xbrain.models import Author, Item
from xbrain.notes_io import GEN_END, GEN_START, note_filename, slugify, title_of, user_tail, wrap


def _item(text: str) -> Item:
    return Item(
        id="123",
        source="bookmark",
        url="u",
        author=Author(handle="a", name="A"),
        text=text,
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )


def test_wrap_surrounds_body_with_markers():
    wrapped = wrap("cuerpo")
    assert wrapped.startswith(GEN_START)
    assert wrapped.endswith(GEN_END)
    assert "cuerpo" in wrapped


def test_user_tail_keeps_content_after_end_marker():
    existing = f"{GEN_START}\nviejo\n{GEN_END}\n\nMI NOTA"
    assert user_tail(existing, "DEFAULT") == "\n\nMI NOTA"


def test_user_tail_preserves_a_marker_less_file():
    assert user_tail("texto sin marcadores", "DEFAULT") == "\n\ntexto sin marcadores"


def test_user_tail_returns_default_for_empty_input():
    assert user_tail("", "DEFAULT") == "DEFAULT"


def test_slugify_handles_edge_cases():
    assert slugify("Café del Día") == "cafe-del-dia"
    assert slugify("") == "item"
    assert len(slugify("a" * 200)) <= 60


def test_note_filename_ends_with_the_item_id():
    assert note_filename(_item("Hola mundo")).endswith("-123.md")


def test_title_of_falls_back_to_item_text():
    assert title_of(_item("Un texto")) == "Un texto"
