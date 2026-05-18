# tests/test_validate.py
from xbrain.validate import validate_judgment

VOCAB = {"ai-coding", "ai-and-work", "misc"}


def _ok():
    return {
        "summary": "un resumen",
        "primary_topic": "ai-coding",
        "topics": ["ai-coding", "ai-and-work"],
    }


def test_valid_judgment_has_no_errors():
    assert validate_judgment(_ok(), VOCAB) == []


def test_topic_outside_vocab_is_rejected():
    bad = _ok()
    bad["topics"] = ["ai-coding", "not-a-real-topic"]
    assert any("not-a-real-topic" in e for e in validate_judgment(bad, VOCAB))


def test_primary_topic_not_in_topics_is_rejected():
    bad = _ok()
    bad["primary_topic"] = "misc"
    assert any("primary_topic" in e for e in validate_judgment(bad, VOCAB))


def test_empty_summary_is_rejected():
    bad = _ok()
    bad["summary"] = "  "
    assert any("summary" in e for e in validate_judgment(bad, VOCAB))


def test_too_many_topics_is_rejected():
    bad = _ok()
    bad["topics"] = ["ai-coding", "ai-and-work", "misc", "ai-coding", "misc"]
    assert any("topics" in e for e in validate_judgment(bad, VOCAB))


def test_non_judgment_keys_are_rejected():
    bad = _ok()
    bad["filename"] = "ai-coding.md"
    assert any("filename" in e for e in validate_judgment(bad, VOCAB))


def test_duplicate_topics_are_rejected():
    bad = _ok()
    bad["topics"] = ["ai-coding", "ai-coding"]
    bad["primary_topic"] = "ai-coding"
    assert any("duplicate" in e for e in validate_judgment(bad, VOCAB))


def test_validate_overview_accepts_a_clean_judgment():
    from xbrain.validate import validate_overview

    assert validate_overview({"overview": "Un resumen.", "notes": ["Una nota."]}) == []


def test_validate_overview_rejects_empty_overview():
    from xbrain.validate import validate_overview

    errors = validate_overview({"overview": "", "notes": []})
    assert any("overview" in e for e in errors)


def test_validate_overview_rejects_wikilinks():
    from xbrain.validate import validate_overview

    errors = validate_overview({"overview": "Mira [[items/algo]].", "notes": ["limpio"]})
    assert any("wikilink" in e for e in errors)


def test_validate_overview_rejects_non_list_notes():
    from xbrain.validate import validate_overview

    errors = validate_overview({"overview": "ok", "notes": "no soy lista"})
    assert any("notes must be a list" in e for e in errors)


def test_validate_overview_rejects_unexpected_keys():
    from xbrain.validate import validate_overview

    errors = validate_overview({"overview": "ok", "notes": [], "slug": "ai-coding"})
    assert any("unexpected keys" in e for e in errors)


def test_validate_overview_rejects_non_string_overview():
    from xbrain.validate import validate_overview

    errors = validate_overview({"overview": {"a": 1}, "notes": ["limpio"]})
    assert any("overview" in e for e in errors)


def test_validate_overview_rejects_non_string_note_element():
    from xbrain.validate import validate_overview

    errors = validate_overview({"overview": "Un resumen.", "notes": [42]})
    assert any("notes entries must all be strings" in e for e in errors)


def test_validate_overview_rejects_too_many_notes():
    from xbrain.validate import validate_overview

    errors = validate_overview({"overview": "Un resumen.", "notes": ["n"] * 16})
    assert any("notes has 16 entries" in e for e in errors)
