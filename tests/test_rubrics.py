# tests/test_rubrics.py
from pathlib import Path

from xbrain.models import Topic
from xbrain.rubrics import load_guardrails, load_rubric, load_vocab, save_vocab


def test_load_rubric_returns_file_text():
    assert "summary" in load_rubric("summary").lower()


def test_load_guardrails_returns_enrichment_constraints():
    g = load_guardrails()
    assert g["enrichment"]["topics_max"] == 4
    assert g["enrichment"]["summary_required"] is True


def test_save_then_load_vocab_roundtrips(tmp_path: Path):
    path = tmp_path / "vocab.yaml"
    topics = [
        Topic(slug="ai-coding", description="LLMs writing software."),
        Topic(slug="misc", description="Posts that do not fit a topic."),
    ]
    save_vocab(topics, path)
    loaded = load_vocab(path)
    assert [t.slug for t in loaded] == ["ai-coding", "misc"]
    assert loaded[0].description == "LLMs writing software."


def test_load_vocab_missing_file_returns_empty(tmp_path: Path):
    assert load_vocab(tmp_path / "nope.yaml") == []


def test_topic_page_rubric_loads():
    from xbrain.rubrics import load_rubric

    text = load_rubric("topic-page")
    assert "overview" in text
    assert "notes" in text
