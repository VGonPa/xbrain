# tests/test_knowledge_consumer_matrix.py
"""Consumer-matrix regression test: every knowledge service consumes the unified contracts.

Plan 02 §14 closure: the seams between the contract models and their consumers are TOTAL.
Adding a new surface type, content kind, or origin without wiring it into all consumers
goes red here, not silently missing in production.

THREE POPULATIONS, THREE TOTALITY ASSERTIONS:

1. `SurfaceType` — every declared member is classified by chunking (atomic, windowed, or
   paragraph-splittable), covered by a producer (`CONTENT_KIND_TO_SURFACE_TYPES` or
   `NON_CONTENT_SURFACES`), and assigned an origin (`SURFACE_ORIGIN`).

2. `ContentKind` — every declared member maps to at least one surface type, and that
   mapping is what `get_service` uses to answer "what surfaces does this failed fetch
   explain?".

3. The CONSUMERS — `search_service` and `get_service` read provenance from
   `SURFACE_ORIGIN` and `ORIGIN_TRUST`, never from a hand-written list. The totality
   assertions here guarantee that adding a new `SurfaceType` or `ContentKind` without
   wiring it into those maps raises immediately, not at query time.

WHAT IS NOT TESTED HERE: the generator/judge/checker contract (`test_evidence_contract.py`
already covers that) and the per-model text-field classification
(`test_knowledge_contracts.py` covers that). This file tests CONSUMERS, not schemas.
"""

from __future__ import annotations

from typing import get_args

from xbrain.knowledge.chunking import ATOMIC_SURFACES, WINDOWED_SURFACES
from xbrain.knowledge.models import SurfaceType
from xbrain.knowledge.provenance import ORIGIN_TRUST
from xbrain.knowledge.surfaces import (
    CONTENT_KIND_TO_SURFACE_TYPES,
    NON_CONTENT_SURFACES,
    SURFACE_ORIGIN,
)
from xbrain.models import ContentKind


# ---------------------------------------------------------------------------
# 1. Chunking classification is TOTAL over SurfaceType
# ---------------------------------------------------------------------------


def test_every_surface_type_is_classified_by_the_chunker() -> None:
    """ATOMIC + WINDOWED + PARAGRAPH-SPLITTABLE = all surfaces.

    The chunker decides how to cut a surface: atomic surfaces are emitted whole (a post,
    a quoted post, a frame caption), windowed surfaces get overlapping windows (a
    transcript), and the rest are split on paragraphs.

    Seen red by adding a surface type without deciding which category it falls into. The
    fix is to add it to one of the three sets in `chunking.py`, according to whether
    attribution or structure governs it.
    """
    all_surfaces = set(get_args(SurfaceType))
    atomic = set(ATOMIC_SURFACES)
    windowed = set(WINDOWED_SURFACES)

    # The paragraph-splittable surfaces are whatever remains — external_article, thread,
    # topic_overview, video_digest, x_article. They are not an explicit frozenset in
    # chunking.py because the logic is "neither atomic nor windowed", not a membership
    # check, and duplicating the list would be a second definition to keep in step.
    paragraph_splittable = all_surfaces - atomic - windowed

    # The three categories must be disjoint
    assert atomic & windowed == set(), "atomic and windowed overlap"
    assert atomic & paragraph_splittable == set(), "atomic and splittable overlap"
    assert windowed & paragraph_splittable == set(), "windowed and splittable overlap"

    # Together they must be complete
    assert atomic | windowed | paragraph_splittable == all_surfaces, (
        f"uncategorized surfaces: {all_surfaces - atomic - windowed - paragraph_splittable}"
    )

    # Sanity: the expected paragraph-splittable surfaces
    expected_splittable = {
        "external_article",
        "thread",
        "topic_overview",
        "video_digest",
        "x_article",
    }
    assert paragraph_splittable == expected_splittable, (
        f"paragraph-splittable set changed: now {paragraph_splittable}, expected {expected_splittable}. "
        "Update this test only after verifying the new member should indeed be paragraph-split."
    )


# ---------------------------------------------------------------------------
# 2. ContentKind → SurfaceType mapping is TOTAL
# ---------------------------------------------------------------------------


def test_content_kind_to_surface_mapping_covers_all_kinds() -> None:
    """Every ContentKind is in CONTENT_KIND_TO_SURFACE_TYPES with a non-empty value.

    `get_service._failed_surface_types` uses this mapping to answer "which surface types
    does this failed fetch explain?" — a kind missing from the map would make `get` unable
    to tell the caller that a requested surface failed to fetch rather than not existing.
    A kind mapped to an empty tuple would explain zero surfaces, which is semantically
    wrong: the content exists but produces nothing.

    Seen red by adding a ContentKind without mapping it to its surface(s), or by mapping
    it to an empty tuple.
    """
    assert set(get_args(ContentKind)) == set(CONTENT_KIND_TO_SURFACE_TYPES), (
        f"ContentKind / CONTENT_KIND_TO_SURFACE_TYPES mismatch: "
        f"missing from map: {set(get_args(ContentKind)) - set(CONTENT_KIND_TO_SURFACE_TYPES)}, "
        f"extra in map: {set(CONTENT_KIND_TO_SURFACE_TYPES) - set(get_args(ContentKind))}"
    )

    # Every kind must produce at least one surface — an empty tuple is semantically wrong
    for kind, surface_types in CONTENT_KIND_TO_SURFACE_TYPES.items():
        assert surface_types, (
            f"CONTENT_KIND_TO_SURFACE_TYPES[{kind!r}] is empty — every kind must produce "
            "at least one surface type"
        )


def test_every_mapped_surface_type_is_real() -> None:
    """The surface types in the mapping must exist in the enum.

    A stale entry pointing at a renamed or deleted surface type would silently become
    unreachable.
    """
    all_surfaces = set(get_args(SurfaceType))
    for kind, surface_types in CONTENT_KIND_TO_SURFACE_TYPES.items():
        for surface_type in surface_types:
            assert surface_type in all_surfaces, (
                f"CONTENT_KIND_TO_SURFACE_TYPES[{kind!r}] maps to {surface_type!r}, "
                "which is not a valid SurfaceType"
            )


# ---------------------------------------------------------------------------
# 3. SURFACE_ORIGIN is TOTAL over SurfaceType
# ---------------------------------------------------------------------------


def test_every_surface_type_has_a_declared_origin() -> None:
    """SURFACE_ORIGIN covers every SurfaceType.

    `search_service._verify_with` and `_hydrate` look up origins in this mapping. A
    surface type missing from it would raise `KeyError` at query time.

    Seen red by adding a surface type without declaring its origin.
    """
    assert set(get_args(SurfaceType)) == set(SURFACE_ORIGIN), (
        f"SurfaceType / SURFACE_ORIGIN mismatch: "
        f"missing: {set(get_args(SurfaceType)) - set(SURFACE_ORIGIN)}, "
        f"extra: {set(SURFACE_ORIGIN) - set(get_args(SurfaceType))}"
    )


def test_every_declared_origin_is_in_the_trust_map() -> None:
    """The origins in SURFACE_ORIGIN must all be in ORIGIN_TRUST.

    `search_service._verify_with` looks up trust classes in ORIGIN_TRUST. An origin
    missing from that map would raise `KeyError` when deciding what to verify against.
    """
    for surface_type, origin in SURFACE_ORIGIN.items():
        assert origin in ORIGIN_TRUST, (
            f"SURFACE_ORIGIN[{surface_type!r}] = {origin!r}, which is not in ORIGIN_TRUST"
        )


# ---------------------------------------------------------------------------
# 4. Producer coverage: CONTENT_KIND_TO_SURFACE_TYPES ∪ NON_CONTENT_SURFACES
# ---------------------------------------------------------------------------


def test_every_surface_is_produced_by_a_content_kind_or_declared_non_content() -> None:
    """The union of produced and non-content surfaces equals all surface types.

    This is already tested in `test_knowledge_surface_coverage.py`, but that file
    tests the totality of `surfaces.py` constants. THIS test is the inverse: it tests
    that the CONSUMERS can assume the two sets together cover everything. A surface
    added to the literal but missed by both sets would be unreachable from both the
    item surfaces (produced) and the standalone surfaces (non-content).
    """
    produced: set[str] = set()
    for surface_types in CONTENT_KIND_TO_SURFACE_TYPES.values():
        produced |= set(surface_types)

    non_content = set(NON_CONTENT_SURFACES)
    all_surfaces = set(get_args(SurfaceType))

    assert produced | non_content == all_surfaces, (
        f"uncovered surfaces: {all_surfaces - produced - non_content}"
    )
    assert produced & non_content == set(), (
        f"surfaces claimed by both CONTENT_KIND and NON_CONTENT: {produced & non_content}"
    )
