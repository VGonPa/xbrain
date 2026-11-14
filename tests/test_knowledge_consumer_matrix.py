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

3. The CONSUMERS read each map from ONE place. Sections 1-2 never open a consumer, so
   section 5 asserts this separately by AST. Measured before it existed: pointing
   `_verify_with` at a private byte-exact copy of `SURFACE_ORIGIN` left ruff and all 2694
   tests green. Two assertions: the map name is loaded, and it is not REBOUND locally (a
   rebind keeps the load, so one without the other passes a private copy). Both are about
   the required names only — that a consumer reads nothing ELSE is not tested, so "touches
   neither provenance map" is a description, not a guarantee.

WHAT IS NOT TESTED HERE: the generator/judge/checker contract (`test_evidence_contract.py`
already covers that) and the per-model text-field classification
(`test_knowledge_contracts.py` covers that). This file tests CONSUMERS, not schemas.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import ModuleType
from typing import get_args

from xbrain.knowledge import get_service, search_service
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


# ---------------------------------------------------------------------------
# 5. The consumer FUNCTIONS read the shared maps (structural, by AST)
# ---------------------------------------------------------------------------
#
# Sections 1-4 never open a consumer, so all six stay green when one reads a private copy.
# Function-scoped because ruff F401 already covers a copy that leaves an import unused, and
# goes quiet when a second reader keeps it live. A MODULE-scope shadow is left to ruff's F811;
# a FUNCTION-local one is not, because it is in a different scope and F811 never sees it — that
# is the second assertion. AST and not grep because `_select`'s docstring names
# `CONTENT_KIND_TO_SURFACE_TYPES` while `_failed_surface_types` is the reader — a text
# search is satisfied by prose (rule 1).

# (module, function, names that function must READ) and the module that DEFINES each name.
CONSUMER_MAP_READS: tuple[tuple[ModuleType, str, tuple[str, ...]], ...] = (
    (
        search_service,
        "_verify_with",
        ("SURFACE_ORIGIN", "ORIGIN_TRUST", "DEFAULT_EVIDENCE_CLASSES"),
    ),
    (search_service, "_hydrate", ("SURFACE_ORIGIN",)),
    (get_service, "_failed_surface_types", ("CONTENT_KIND_TO_SURFACE_TYPES",)),
)
CANONICAL_DEFINITION = {
    "CONTENT_KIND_TO_SURFACE_TYPES": "xbrain.knowledge.surfaces",
    "SURFACE_ORIGIN": "xbrain.knowledge.surfaces",
    "ORIGIN_TRUST": "xbrain.knowledge.provenance",
    "DEFAULT_EVIDENCE_CLASSES": "xbrain.knowledge.provenance",
}


def _locally_bound(fn: ast.FunctionDef) -> set[str]:
    """The names the function BINDS by store (assignment, walrus, loop/with target), by
    parameter, or by a function-local import.

    A local rebind KEEPS the Load the check above looks for, and the two ruff rules that
    comment defers to are blind to it BY CONSTRUCTION, not by oversight: F811 compares
    bindings within ONE scope, and a function local is a different scope from the module
    import; F401 stays quiet because a second consumer keeps that import live. Measured on
    this tree with `_verify_with` reading a private `dict(...)` copy of `SURFACE_ORIGIN`:
    ruff `All checks passed`, suite `2696 passed, 1 xfailed`.

    Only these three forms are collected, and each was seen red before this was written. The
    remaining binders — `except ... as NAME`, a nested `def`/`class` NAME — are not collected,
    so a shadow spelled that way is a known gap, not a covered case.
    """
    bound: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.alias):  # reachable only from a function-local import
            bound.add(node.asname or node.name.split(".")[0])
    return bound


def test_the_consumer_table_is_not_empty() -> None:
    """An empty table would pass by iterating nothing (rule 1)."""
    assert CONSUMER_MAP_READS and all(names for _, _, names in CONSUMER_MAP_READS)


def test_each_consumer_function_reads_the_shared_maps() -> None:
    """Each named function loads each required map, imported from the module defining it.

    A name that is loaded but locally REBOUND is a copy wearing the shared map's name, so
    the second assertion refuses any local binding of a required name.

    Seen red by a private copy of `SURFACE_ORIGIN` in `_verify_with`, an inline dict in
    `_failed_surface_types`, `_hydrate` dropping its lookup, and a consumer renamed so its
    row points at nothing. Ruff misses the first, third and fourth.
    """
    for module, function, required in CONSUMER_MAP_READS:
        tree = ast.parse(textwrap.dedent(inspect.getsource(module)))
        fn = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function),
            None,
        )
        assert fn is not None, f"{module.__name__} has no {function!r}; update the row"
        loaded = {
            n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        assert not set(required) - loaded, (
            f"{module.__name__}.{function} no longer reads "
            f"{', '.join(sorted(set(required) - loaded))} — it has grown a second "
            "definition that will drift from the shared map (rule 5)"
        )
        shadowed = set(required) & _locally_bound(fn)
        assert not shadowed, (
            f"{module.__name__}.{function} REBINDS {', '.join(sorted(shadowed))} locally — "
            "the load above then reads a private copy, not the shared map (rule 5)"
        )
        imported = {
            alias.asname or alias.name: imp.module or ""
            for imp in ast.walk(tree)
            if isinstance(imp, ast.ImportFrom)
            for alias in imp.names
        }
        for name in required:
            assert imported.get(name) == CANONICAL_DEFINITION[name], (
                f"{module.__name__} binds {name!r} from {imported.get(name)!r}, "
                f"expected {CANONICAL_DEFINITION[name]!r}"
            )
