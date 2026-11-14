# tests/test_knowledge_profile.py
"""The global item profile (Plan 01 §3.7, spec §5.1.A, step 18).

Two levels of representation exist for a reason. The PROFILE answers *what is this item
about* — it finds the item as a conceptual unit. The CHUNKS answer *where exactly does this
fact live*. Both link back to the same `item_id`.

THE CONSTRAINT THAT MATTERS: the profile is a RETRIEVAL representation, never a citation. It
is a composed string that no author ever wrote — a tweet, a summary and three topic
descriptions glued together. Returning it as evidence would present a machine's collage as
if someone had said it, which is the exact confusion between derived and primary that
provenance exists to prevent. So it has no `surface_id`, and nothing in the emitter can turn
it into a chunk.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from xbrain.knowledge import profile as profile_module
from xbrain.knowledge.profile import profile_text
from xbrain.models import Item, Topic

FIXTURES = Path(__file__).parent / "fixtures"


def _corpus():
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    items = {k: Item.model_validate(v) for k, v in raw["items"].items()}
    vocab = [Topic.model_validate(v) for v in raw["vocab"].values()]
    return items, vocab


def test_the_profile_is_deterministic() -> None:
    """Two calls give the same string, byte for byte.

    Not a triviality: the profile composes a summary, a set of topics and their
    descriptions, and any of those arriving from a set or a dict iteration would make the
    string — and therefore the lexical ranking built on it — differ between runs on the same
    store. Spec §3.7.8 requires stable order.
    """
    items, vocab = _corpus()
    item = items["k08"]
    assert profile_text(item, vocab) == profile_text(item, vocab)


def test_the_profile_composes_the_declared_parts_in_a_fixed_order() -> None:
    """Spec §5.1.A: post text · titles · summary · digest · topics + their descriptions · author.

    Asserted by ORDER of first appearance, not by "the substring is somewhere in there" —
    CLAUDE.md rule 1's canonical failure is an assertion satisfied by any occurrence
    anywhere. Order is what makes the composition deterministic and reviewable.
    """
    items, vocab = _corpus()
    text = profile_text(items["k08"], vocab)
    positions = [
        text.index("Full talk below."),  # the post
        text.index("A talk on evaluation"),  # the video title
        text.index("Una charla sobre bucles"),  # the summary
        text.index("What it is: a talk"),  # the digest
        text.index("agent-evaluation"),  # the topic slug
        text.index("Agents that evaluate"),  # the topic description
        text.index("vgonpa"),  # the author
    ]
    assert positions == sorted(positions), f"parts out of order: {positions}"


def test_the_profile_omits_what_the_item_does_not_have() -> None:
    """An item with no content and no video contributes no empty labelled sections.

    An empty section is not neutral: it tells a lexical index there is a field there, and it
    would make two different items look structurally identical.
    """
    items, vocab = _corpus()
    bare = profile_text(items["k01"], vocab)
    assert "Zephyrine" in bare
    assert bare.count("\n\n") == 0 or "None" not in bare


def test_the_profile_is_not_a_citable_surface() -> None:
    """It has no `surface_id`, and no code path turns it into a chunk or a match.

    Asserted structurally rather than by grepping the repo: `profile_text` returns a bare
    `str`, and the module imports neither the chunker nor the contracts — so it CANNOT
    construct a `KnowledgeChunk` or a `SearchMatch` even by accident. Seen red by importing
    `chunk_surface` into the module.
    """
    assert inspect.signature(profile_text).return_annotation == "str"
    imported = set(vars(profile_module))
    assert not {"KnowledgeChunk", "SearchMatch", "chunk_surface"} & imported


def test_the_profile_never_carries_a_topic_that_is_not_in_the_vocabulary() -> None:
    """Spec §3.7.5: topics and filters come from the store, never invented from the query.

    An item whose `primary_topic` is missing from the vocabulary keeps its slug — the
    assignment is real — but no description is invented for it.
    """
    items, vocab = _corpus()
    text = profile_text(items["k12"], vocab)  # knowledge-management is not in this fixture vocab
    assert "knowledge-management" in text
    assert "Agents that evaluate" not in text
