"""The questions Jev is asked about one post.

One Noul per vocabulary topic (membership — a post can belong to several) and one Choice over
every topic plus the fallback (the primary, and a second ranking to compare with the sorted
Nouls). Instructions are English, Jev's primary language (`intent.md` §4.1); topic descriptions
travel verbatim in the vocabulary's language — they are the criteria, i.e. data.

THE QUESTION SET IS CANONICAL: Nouls in slug-sorted order, the Choice's options in slug-sorted
order with the fallback last. One vocabulary always produces ONE wire form, whatever order
`vocab.yaml` happens to list its topics in. That is what lets `assess.questions_digest` — the
canonical JSON of every question's type, instructions and criteria, which `assess.topic_contract`
hashes alongside the state — serialise with sorted keys and still describe the ask: option order
reaches the model and biases a pick-one answer, so a digest that ignored order while the wire did
not would call an assessment current after the question had visibly changed.
"""

from __future__ import annotations

import unicodedata

from xbrain.jev.client import ChoiceQuestion, NoulQuestion, Question
from xbrain.models import Topic

TOPIC_PREFIX = "topic__"
PRIMARY_KEY = "primary"
STATE_KEY = "post"

# Self-contained on purpose. A negative side that deferred to its sibling ("not about '<slug>'
# AS DESCRIBED") would make the model resolve an anaphor to read the criterion at all, and
# `intent.md` §3 names LITERAL READING and INDIRECTION as Jev's two documented failure modes.
_FALSE_CRITERION = "The post is about something else."
# ONE rule for the escape option, worded identically here and in the Choice's instruction.
# Admitting it here on a condition the instruction does not mention (say, "or no single clear
# topic") would hand a literal reader two rules for one option across the ~87% of the corpus
# that carries 2+ topics (`intent.md` §2) — the exact population this signal is read off.
_FALLBACK_CRITERION = "A topic that is not in this list."


def build_topic_questions(vocab: list[Topic], fallback: str) -> dict[str, Question]:
    """`{"topic__<slug>": Noul, ..., "primary": Choice}` for `vocab`, in canonical order.

    Validates the inputs so a bad vocabulary fails before the first API call, not inside a
    thread: a duplicate slug would silently collapse two questions into one key, a fallback
    equal to a slug would make "none of these" and that topic the same option, and a blank
    description would ship a question with its criterion gone — Jev answers those with a
    confident probability judged against nothing but the slug.
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
    for topic in vocab:
        if not topic.description.strip():
            raise ValueError(f"el topic {topic.slug!r} no tiene descripción")
    ordered = sorted(vocab, key=lambda topic: topic.slug)
    questions: dict[str, Question] = {}
    for topic in ordered:
        questions[TOPIC_PREFIX + topic.slug] = NoulQuestion(
            instructions=f"Is the post in `{STATE_KEY}` about the topic '{topic.slug}'?",
            criteria={"true": topic.description, "false": _FALSE_CRITERION},
        )
    criteria: dict[str, str | None] = {topic.slug: topic.description for topic in ordered}
    # Appended AFTER the sorted slugs, so the escape option is always offered last — it is
    # not a topic and must not compete for position with one.
    criteria[fallback] = _FALLBACK_CRITERION
    questions[PRIMARY_KEY] = ChoiceQuestion(
        instructions=(
            f"Which single topic is the post in `{STATE_KEY}` mainly about? "
            f"Choose '{fallback}' only when no listed topic fits."
        ),
        criteria=criteria,
    )
    return questions


#: The one key of an ask (`xbrain jev ask`): does this post answer the user's request?
ASK_KEY = "answers"
_ASK_INSTRUCTIONS = f"Does the post in `{STATE_KEY}` answer or directly address the user's request?"
_ASK_FALSE_CRITERION = "The post does not address this request."


def normalize_query(query: str) -> str:
    """The query as it is asked and filed: NFC, runs of whitespace as one space, trimmed.

    Case and punctuation are KEPT — the query travels to Jev verbatim as the criterion, so they
    are part of the question; spacing and Unicode normal form are not, and must not turn one
    query into two paid ones. A query with nothing left is refused (`ValueError`).
    """
    text = " ".join(unicodedata.normalize("NFC", query).split())
    if not text:
        raise ValueError("la consulta está vacía")
    return text


def build_ask_questions(query: str) -> dict[str, Question]:
    """`{"answers": Noul}`: the user's query, verbatim in any language, as the TRUE criterion.

    A Noul and not a Choice over the posts: each post is judged ALONE against the request, so
    its probability does not depend on which other posts happened to be asked with it. The
    instructions are English (Jev's primary language, as for topics); the query is data.
    """
    return {
        ASK_KEY: NoulQuestion(
            instructions=_ASK_INSTRUCTIONS,
            criteria={"true": normalize_query(query), "false": _ASK_FALSE_CRITERION},
        )
    }
