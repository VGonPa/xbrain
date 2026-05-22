# tests/test_i18n.py
"""Unit tests for xbrain.i18n — the wiki UI strings table."""

from __future__ import annotations

import dataclasses

import pytest

from xbrain.i18n import SUPPORTED_LANGUAGES, Strings, strings_for


def test_supported_languages_has_english_and_spanish() -> None:
    assert "English" in SUPPORTED_LANGUAGES
    assert "Spanish" in SUPPORTED_LANGUAGES


def test_english_strings_returned_for_english() -> None:
    s = strings_for("English")
    assert s.topics_label == "Topics"
    assert s.content_header == "Content"
    assert s.summary_header == "Summary"
    assert s.primary_posts == "Primary posts"
    assert s.also_relevant == "Also relevant"


def test_spanish_strings_returned_for_spanish() -> None:
    s = strings_for("Spanish")
    assert s.topics_label == "Temas"
    assert s.content_header == "Contenido"
    assert s.summary_header == "Resumen"
    assert s.primary_posts == "Posts primarios"
    assert s.also_relevant == "También relevante"


def test_unknown_language_raises_with_supported_list() -> None:
    with pytest.raises(ValueError) as exc:
        strings_for("Klingon")
    msg = str(exc.value)
    assert "Klingon" in msg
    assert "English" in msg or "Spanish" in msg


def test_every_supported_language_populates_every_field() -> None:
    """A new language must not leave any field empty — the dataclass enforces presence."""
    fields = {f.name for f in dataclasses.fields(Strings)}
    for language in SUPPORTED_LANGUAGES:
        s = strings_for(language)
        for field in fields:
            value = getattr(s, field)
            assert isinstance(value, str) and value, f"{language}.{field} is empty"


def test_strings_dataclass_is_frozen() -> None:
    """Strings must be immutable — accidental mutation would leak between callers."""
    s = strings_for("English")
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.topics_label = "mutated"  # type: ignore[misc]
