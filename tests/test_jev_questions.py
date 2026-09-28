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
    """Every instruction and criterion pinned in FULL.

    The prompt is the product, and every byte of it is hashed into each stored contract:
    a reword has to be a visible edit to this file, never a silent change that leaves the
    corpus holding answers produced by a prompt that no longer exists.
    """
    questions = build_topic_questions(_vocab(), "otro")
    assert list(questions) == [f"{TOPIC_PREFIX}ai-coding", f"{TOPIC_PREFIX}startups", PRIMARY_KEY]
    noul = questions[f"{TOPIC_PREFIX}ai-coding"]
    assert isinstance(noul, NoulQuestion)
    assert noul.instructions == "Is the post in `post` about the topic 'ai-coding'?"
    assert noul.criteria == {
        "true": "Construir software con IA.",
        "false": "The post is about something else.",
    }
    primary = questions[PRIMARY_KEY]
    assert isinstance(primary, ChoiceQuestion)
    assert primary.instructions == (
        "Which single topic is the post in `post` mainly about? "
        "Choose 'otro' only when no listed topic fits."
    )
    assert primary.criteria == {
        "ai-coding": "Construir software con IA.",
        "startups": "Fundar y financiar empresas.",
        "otro": "A topic that is not in this list.",
    }
    assert list(primary.criteria) == ["ai-coding", "startups", "otro"]


def test_questions_are_canonical_whatever_the_vocabulary_order():
    """A reshuffled `vocab.yaml` must ask the identical question set, in the identical order.

    Option order reaches the model and biases a pick-one answer, so "the contract is
    order-independent" is only honest if there is exactly ONE order on the wire.
    """
    vocab = _vocab()
    forward = build_topic_questions(vocab, "otro")
    backward = build_topic_questions([vocab[1], vocab[0]], "otro")
    assert forward == backward
    # Dicts compare equal whatever their order, so pin the ORDER separately — it is the
    # half that actually travels.
    assert list(forward) == list(backward)
    assert list(forward[PRIMARY_KEY].criteria) == list(backward[PRIMARY_KEY].criteria)
    assert list(backward) == [f"{TOPIC_PREFIX}ai-coding", f"{TOPIC_PREFIX}startups", PRIMARY_KEY]


def test_the_fallback_option_is_offered_last_even_when_its_slug_sorts_first():
    questions = build_topic_questions(_vocab(), "aaa-ninguno")
    assert list(questions[PRIMARY_KEY].criteria) == ["ai-coding", "startups", "aaa-ninguno"]


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


def test_a_blank_topic_description_is_rejected_by_slug():
    """A description is the Noul's `true` criterion — blank ships a criterion-free question.

    `Topic.description` has no `min_length` and `xbrain vocab` is an LLM call, so a blank
    one is reachable. Jev would then answer a confident probability judged against a slug
    alone, and the contract would stamp the blank as if it were the ask.
    """
    vocab = [
        Topic(slug="ai-coding", description="  \n "),
        Topic(slug="startups", description="Ok."),
    ]
    with pytest.raises(ValueError, match="el topic 'ai-coding' no tiene descripción"):
        build_topic_questions(vocab, "otro")
