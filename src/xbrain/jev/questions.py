"""The questions Jev is asked about one post.

One Noul per vocabulary topic (membership — a post can belong to several) and one Choice over
every topic plus the fallback (the primary, and a second ranking to compare with the sorted
Nouls). Instructions are English, Jev's primary language; topic descriptions travel verbatim
in the vocabulary's language — they are the criteria, i.e. data.
"""

from __future__ import annotations

from xbrain.jev.client import ChoiceQuestion, NoulQuestion, Question
from xbrain.models import Topic

TOPIC_PREFIX = "topic__"
PRIMARY_KEY = "primary"
STATE_KEY = "post"


def build_topic_questions(vocab: list[Topic], fallback: str) -> dict[str, Question]:
    """`{"topic__<slug>": Noul, ..., "primary": Choice}` for `vocab`.

    Validates the inputs so a bad vocabulary fails before the first API call, not inside a
    thread: a duplicate slug would silently collapse two questions into one key, and a
    fallback equal to a slug would make "none of these" and that topic the same option.
    """
    if not vocab:
        raise ValueError(
            "el vocabulario está vacío: ejecuta `xbrain vocab` antes de `xbrain jev topics`"
        )
    slugs = [topic.slug for topic in vocab]
    if len(set(slugs)) != len(slugs):
        raise ValueError("el vocabulario tiene slugs repetidos")
    if fallback in slugs:
        raise ValueError(f"[jev].fallback_option {fallback!r} choca con un slug del vocabulario")
    questions: dict[str, Question] = {}
    for topic in vocab:
        questions[TOPIC_PREFIX + topic.slug] = NoulQuestion(
            instructions=f"Is the post in `{STATE_KEY}` about the topic '{topic.slug}'?",
            criteria={
                "true": topic.description,
                "false": f"The post is not about '{topic.slug}' as described.",
            },
        )
    criteria: dict[str, str | None] = {topic.slug: topic.description for topic in vocab}
    criteria[fallback] = "A topic that is not in this list, or no single clear topic."
    questions[PRIMARY_KEY] = ChoiceQuestion(
        instructions=(
            f"Which single topic is the post in `{STATE_KEY}` mainly about? "
            f"Choose '{fallback}' when none of the listed topics fits."
        ),
        criteria=criteria,
    )
    return questions
