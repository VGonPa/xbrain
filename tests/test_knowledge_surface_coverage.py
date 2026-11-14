# tests/test_knowledge_surface_coverage.py
"""The three TOTALITY tests that bind the knowledge contract to the store's own types.

WHY NOT AN IDENTITY ASSERTION. CLAUDE.md rule 1: `assert knowledge.quoted_source is
api.quoted_source` is green forever the moment delegation exists, and binds nothing — the
tautology reappears in disguise every time you think you have killed it. What actually
binds two definitions is TOTALITY: a mapping asserted complete against a type the OTHER
side owns, so adding a member on that side and forgetting this one is red.

WHY THREE AND NOT TWO. The first test alone is correct but its NAME would overpromise:
with a one-to-one map it stays green when someone adds a `ContentKind`, maps its body and
forgets its DERIVED layers. `x_video` produces three surfaces — `video_transcript`,
`video_frame`, `video_digest` (152, 2 196 and 216 instances in the corpus on 2026-08-31) —
so a one-to-one map protects one of the three. The second test closes the other side and
turns "every surface has a producer" from a sentence into an assertion. The third reaches
across to the OTHER contract, `evidence._SURFACES`, so a surface added for the judge cannot
be silently absent from the knowledge layer.

Each is documented with the deletion that makes it red, and each was run in that state.
"""

from __future__ import annotations

from typing import get_args

from xbrain import evidence
from xbrain.knowledge.models import SurfaceType
from xbrain.knowledge.provenance import ORIGIN_TRUST
from xbrain.knowledge.surfaces import (
    CONTENT_KIND_TO_SURFACE_TYPES,
    EVIDENCE_KEY_SURFACES,
    NON_CONTENT_SURFACES,
    NOT_A_KNOWLEDGE_SURFACE,
    SURFACE_ORIGIN,
)
from xbrain.models import ContentKind


def test_every_content_kind_has_surface_emitters() -> None:
    """`get_args(ContentKind)` is the store's own truth; the map must cover all of it.

    Seen red by deleting `"x_video"` from `CONTENT_KIND_TO_SURFACE_TYPES`. Red in the other
    direction too: an entry for a kind the store no longer defines is a dead branch a
    reader would trust.
    """
    assert set(get_args(ContentKind)) == set(CONTENT_KIND_TO_SURFACE_TYPES)


def test_every_surface_type_is_produced_by_something() -> None:
    """The closure that the first test cannot provide (M3).

    Every `SurfaceType` is either produced by some `ContentKind` or declared in
    `NON_CONTENT_SURFACES` — the surfaces that come from `item.text`, `item.media`,
    `enriched` or a topic rather than from a content source. Seen red by deleting
    `"video_digest"` from `x_video`'s tuple: the first test stays GREEN (the kind is still
    mapped), and only this one names the orphaned surface.
    """
    produced: set[str] = set()
    for surface_types in CONTENT_KIND_TO_SURFACE_TYPES.values():
        produced |= set(surface_types)
    assert set(get_args(SurfaceType)) == produced | set(NON_CONTENT_SURFACES)


def test_content_and_non_content_surfaces_are_disjoint() -> None:
    """A surface declared on both sides would make the closure above unfalsifiable.

    With an overlap, deleting `"video_digest"` from `x_video` would leave the union
    unchanged (because `NON_CONTENT_SURFACES` still supplied it) and the second test would
    go green on a broken map — the failure mode is a test that cannot fail, so the
    partition is asserted rather than assumed.
    """
    produced: set[str] = set()
    for surface_types in CONTENT_KIND_TO_SURFACE_TYPES.values():
        produced |= set(surface_types)
    assert produced & set(NON_CONTENT_SURFACES) == set()


def test_every_surface_type_has_an_origin() -> None:
    """Spec §4's `origin` column, total over `SurfaceType`.

    Seen red by deleting `"topic_description"`. A surface with no declared origin would
    reach a consumer with no provenance, which is invariant 2 of spec §3.7 broken at the
    root: *every returned text includes origin, surface_type and locator*.
    """
    assert set(get_args(SurfaceType)) == set(SURFACE_ORIGIN)
    assert set(SURFACE_ORIGIN.values()) <= set(ORIGIN_TRUST)


def test_every_evidence_surface_key_is_classified() -> None:
    """The reach across to the OTHER contract (`xbrain.evidence`).

    `evidence._SURFACES` is the judge's list. Every key of it must be either mapped to the
    knowledge surface types it corresponds to, or explicitly declared NOT a knowledge
    surface. Today the second group is `author`, `video_title` and `article_title` —
    attribution and title metadata, which ride ON a surface (`KnowledgeSurface.title`,
    `.attribution`) rather than being bodies of text of their own.

    Seen red by adding a key to `evidence._SURFACES` without deciding here. That is the
    real and likely error: someone adds an evidence surface for the judge and the knowledge
    layer silently keeps indexing less than the judge reads.
    """
    classified = set(EVIDENCE_KEY_SURFACES) | set(NOT_A_KNOWLEDGE_SURFACE)
    assert set(evidence._SURFACES) == classified
    assert set(EVIDENCE_KEY_SURFACES) & set(NOT_A_KNOWLEDGE_SURFACE) == set()


def test_evidence_keys_map_onto_real_surface_types() -> None:
    """A classification pointing at a surface type that does not exist binds nothing."""
    mapped: set[str] = set()
    for surface_types in EVIDENCE_KEY_SURFACES.values():
        mapped |= set(surface_types)
    assert mapped <= set(get_args(SurfaceType))


def test_the_article_evidence_key_covers_both_article_kinds() -> None:
    """`evidence`'s single `article` key is `LINK_CONTENT_KINDS` — two knowledge surfaces.

    Pinned because the one-to-many shape is exactly what a future simplification to a
    scalar map would quietly break, leaving `x_article` unclassified while the totality
    test above still passed on the remaining key.
    """
    assert set(EVIDENCE_KEY_SURFACES["article"]) == {"external_article", "x_article"}
