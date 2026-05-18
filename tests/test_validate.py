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
