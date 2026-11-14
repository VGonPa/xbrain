"""The global item profile — "what is this item about" as one composed string (spec §5.1.A).

TWO LEVELS OF REPRESENTATION, and this is the first. The PROFILE finds the item as a
conceptual unit; the CHUNKS find the fragment where a specific fact lives. Both resolve to
the same `item_id`, and a retrieval layer needs both — a query like *"what did I save about
agents that evaluate their own work"* is about the item, while *"what DunedinPACE figure was
in that image"* is about one fragment.

THE CONSTRAINT THAT GOVERNS THE MODULE: the profile is a RETRIEVAL representation and never
a citation. It is a string nobody wrote — a tweet, a summary and three topic descriptions
glued together — so returning it as evidence would present a machine's collage as something
someone said. That is exactly the confusion between derived and primary that provenance
exists to prevent. It therefore has NO `surface_id`, and this module imports neither the
chunker nor the response contracts, so it cannot construct a chunk or a match even by
accident. The suite asserts that structurally.

Composition is deterministic — fixed order, fixed separators — because the lexical ranking
built on it must not differ between two runs over the same store (spec §3.7.8).
"""

from __future__ import annotations

from xbrain.executors.api import iter_content_sources
from xbrain.knowledge.surfaces import CONTENT_KIND_TO_SURFACE_TYPES, item_topics
from xbrain.models import Item, Topic

# One separator, one place. The profile is fed to a tokenizer, so the separator only has to
# stop two fields running into one token — it is not a format anyone parses back.
_SEP = "\n"


def profile_text(item: Item, vocab: list[Topic]) -> str:
    """The item's retrieval profile: post · titles · summary · digest · topics · author.

    Parts the item does not have are OMITTED rather than emitted empty. An empty labelled
    section is not neutral — it tells the index a field is present and makes two structurally
    different items look alike.

    Topic descriptions are looked up in the vocabulary and simply absent when the slug is not
    there. The slug itself is kept, because the assignment is real data even when the
    vocabulary has moved on; what is never done is inventing a description for it (spec
    §3.7.5: topics come from the store, never from the query text).
    """
    descriptions = {topic.slug: topic.description for topic in vocab}
    parts: list[str] = []

    if item.text.strip():
        parts.append(item.text)
    parts += _titles(item)
    if item.enriched is not None and item.enriched.summary:
        parts.append(item.enriched.summary)
    parts += _digests(item)
    for slug in item_topics(item):
        parts.append(slug)
        description = descriptions.get(slug)
        if description:
            parts.append(description)
    parts += [value for value in (item.author.handle, item.author.name) if value]
    return _SEP.join(parts)


def _titles(item: Item) -> list[str]:
    """Every content source title, in source order.

    Spec §4: *article titles accompany their chunks and ALSO take part in the global
    profile*. A title is often the only place a work's name appears — an essay whose body
    never repeats its own title is the normal case, not the exception.
    """
    return [
        source.title
        for _index, source in iter_content_sources(item, set(CONTENT_KIND_TO_SURFACE_TYPES))
        if source.title
    ]


def _digests(item: Item) -> list[str]:
    """Every video digest on the item.

    Included because the digest is what makes a video findable as a unit: the transcript
    says what was said, the digest says what the talk WAS, and the second is what a
    conceptual query matches.
    """
    return [
        source.digest for _index, source in iter_content_sources(item, {"x_video"}) if source.digest
    ]
