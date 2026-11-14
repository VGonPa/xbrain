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


def test_strings_dataclass_has_exactly_the_expected_fields() -> None:
    """If a field is added or removed, every translation must be updated."""
    field_names = {f.name for f in dataclasses.fields(Strings)}
    assert field_names == {
        "language",
        "topics_label",
        "content_header",
        "summary_header",
        "primary_posts",
        "also_relevant",
        "video_digest_header",
        "silent_video",
        "video_evidence_header",
        "verify_badge_fail",
        "verify_badge_review",
        "quoted_post_header",
        "quoted_post_unavailable",
        "quoted_unavailable_deleted",
        "quoted_unavailable_protected",
        "quoted_unavailable_unknown",
    }


def test_verify_badge_strings_present_in_every_language() -> None:
    """The #79 verification-badge labels must exist in every supported language."""
    for language in SUPPORTED_LANGUAGES:
        s = strings_for(language)
        assert s.verify_badge_fail
        assert s.verify_badge_review


def test_video_digest_strings_present_in_every_language() -> None:
    """The #44 video-digest headers must exist in every supported language."""
    for language in SUPPORTED_LANGUAGES:
        s = strings_for(language)
        assert s.video_digest_header
        assert s.silent_video


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


def test_every_language_localises_the_quoted_post_strings() -> None:
    """The quoted post is rendered into the human's note, so its headers must speak the
    vault's language — a hardcoded Spanish `Post citado` would sit in an English vault.
    Asserts the strings DIFFER per language, not merely that they are present: a field
    copied verbatim across languages passes a presence check and still reads wrong."""
    english, spanish = strings_for("English"), strings_for("Spanish")

    assert english.quoted_post_header == "Quoted post"
    assert spanish.quoted_post_header == "Post citado"
    assert english.quoted_post_unavailable != spanish.quoted_post_unavailable
    assert english.quoted_unavailable_deleted != spanish.quoted_unavailable_deleted
    assert english.quoted_unavailable_protected != spanish.quoted_unavailable_protected
    assert english.quoted_unavailable_unknown != spanish.quoted_unavailable_unknown
