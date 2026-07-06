# tests/test_rubrics.py
from pathlib import Path

from xbrain.models import Topic
from xbrain.rubrics import load_guardrails, load_rubric, load_vocab, save_vocab


def test_load_rubric_returns_file_text():
    assert "summary" in load_rubric("summary").lower()


def test_load_rubric_substitutes_language_placeholder():
    """When language is provided, {language} is replaced verbatim."""
    text = load_rubric("summary", language="English")
    assert "{language}" not in text
    assert "**Language:** English" in text


def test_load_rubric_supports_spanish_language():
    text = load_rubric("topic-page", language="Spanish")
    assert "{language}" not in text
    # The placeholder appears twice in topic-page (overview + notes)
    assert text.count("in Spanish") == 2


def test_load_rubric_preserves_placeholder_when_language_none():
    """No-language calls (tests, inspection) keep the literal `{language}`."""
    text = load_rubric("summary")
    assert "{language}" in text


def test_load_rubric_topics_has_no_placeholder():
    """rubric-topics emits only slugs; no language placeholder; passing one is a no-op."""
    a = load_rubric("topics")
    b = load_rubric("topics", language="English")
    assert a == b
    assert "{language}" not in a


def test_load_rubric_defensive_check_catches_unsubstituted_placeholder(tmp_path, monkeypatch):
    """A typo like {Language} (capital L) survives str.replace and would
    silently ship the literal placeholder to the LLM. The defensive regex
    catches it and raises a loud ValueError naming the typo.
    """
    import pytest

    from xbrain import rubrics as rubrics_mod

    typo_dir = tmp_path / "rubrics"
    typo_dir.mkdir()
    (typo_dir / "rubric-typo.md").write_text(
        "**Language:** {Language}, regardless of the post.\n",  # capital L typo
        encoding="utf-8",
    )
    monkeypatch.setattr(rubrics_mod, "_RUBRICS_DIR", typo_dir)

    with pytest.raises(ValueError, match=r"\{Language\}"):
        load_rubric("typo", language="English")


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


def test_describe_image_rubric_loads_and_substitutes_language():
    """The describe-image rubric ships a `{language}` placeholder; the
    loader must substitute it, and the defensive check must not trip on
    correctly-spelt placeholders.
    """
    text = load_rubric("describe-image", language="English")
    assert "{language}" not in text
    assert "English" in text
    # Sanity: the contract keys must appear in the prompt so the LLM
    # produces the right JSON shape.
    assert "is_decorative" in text
    assert "description" in text
    assert "index" in text


def test_describe_image_rubric_preserves_placeholder_when_language_none():
    """No-language calls (tests, manual inspection) keep the literal placeholder."""
    text = load_rubric("describe-image")
    assert "{language}" in text


def test_video_digest_rubric_loads_and_substitutes_language():
    """The video-digest rubric ships a `{language}` placeholder the loader must
    substitute; its structural section keys must reach the LLM."""
    text = load_rubric("video-digest", language="Spanish")
    assert "{language}" not in text
    assert "Spanish" in text
    assert "Key points" in text
    assert "What it is" in text


def test_video_digest_rubric_preserves_placeholder_when_language_none():
    text = load_rubric("video-digest")
    assert "{language}" in text
