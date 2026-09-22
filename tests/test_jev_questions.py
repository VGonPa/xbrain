# tests/test_jev_questions.py
import pytest

from xbrain.jev.client import ChoiceQuestion, NoulQuestion
from xbrain.jev.questions import PRIMARY_KEY, TOPIC_PREFIX, build_topic_questions
from xbrain.models import Topic


def _vocab():
    return [
        Topic(slug="ai-coding", description="Construir software con IA."),
        Topic(slug="startups", description="Fundar y financiar empresas."),
    ]


def test_one_noul_per_topic_then_one_primary_choice():
    questions = build_topic_questions(_vocab(), "otro")
    assert list(questions) == [f"{TOPIC_PREFIX}ai-coding", f"{TOPIC_PREFIX}startups", PRIMARY_KEY]
    noul = questions[f"{TOPIC_PREFIX}ai-coding"]
    assert isinstance(noul, NoulQuestion)
    assert "`post`" in noul.instructions and "'ai-coding'" in noul.instructions
    assert noul.criteria == {
        "true": "Construir software con IA.",
        "false": "The post is not about 'ai-coding' as described.",
    }
    primary = questions[PRIMARY_KEY]
    assert isinstance(primary, ChoiceQuestion)
    assert list(primary.criteria) == ["ai-coding", "startups", "otro"]
    assert primary.criteria["startups"] == "Fundar y financiar empresas."
    assert "'otro'" in primary.instructions


def test_fallback_colliding_with_a_slug_is_rejected():
    with pytest.raises(ValueError, match="choca"):
        build_topic_questions(_vocab(), "startups")


def test_empty_vocab_is_rejected():
    with pytest.raises(ValueError, match="vacío"):
        build_topic_questions([], "otro")


def test_duplicate_slugs_are_rejected():
    vocab = _vocab() + [Topic(slug="startups", description="otra vez")]
    with pytest.raises(ValueError, match="repetidos"):
        build_topic_questions(vocab, "otro")
