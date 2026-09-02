# tests/test_knowledge_seams.py
"""The two seams of the knowledge layer, ENUMERATED (round 07, the end of a family).

Six rounds closed the FAIL-OPEN family one route at a time and four rounds closed the
ATTRIBUTION/LOCATOR family the same way. Round 06 gave each family ONE function — seam (a)
`index_build.describe_base` + `index_schema.require_database`, seam (b)
`chunking.fragment_locator` + `render._label` — and a sentinel test per seam that every
consumer must repeat verbatim. Both gates of round 07 confirmed the seams WORK, and found
consumers that did not pass through them: `status` tested `exists()` by itself (U-2), the
URLs of the human view were printed raw (U-3), the served provenance was bound to nothing
(U-5). A sentinel test proves the consumers it names agree; it says nothing about a consumer
it does not name. So this file names them ALL, structurally, and goes red when a new one
appears without passing through the seam:

* seam (a): every function of `xbrain.knowledge` that opens the base file is a declared
  DOOR, and every door except the one creator asks BOTH halves of the question — «is there
  a base?» (`require_database`) and «does the manifest describe it?» (`describe_base` or
  `require_consistent`);
* seam (b), the served fragment: every function that constructs a `KnowledgeChunk` or a
  `SearchMatch` narrows the locator through `fragment_locator`, and every function that
  recomputes a chunk fingerprint builds its evidence through `chunk_evidence`;
* seam (b), the human view: every `str` field of the frozen contract, forged, never reaches
  column 0 or the terminal — that one lives in `tests/test_knowledge_render.py` because it
  needs the renderer's fixtures, and it enumerates the CONTRACT's fields, which is the list
  a new field cannot join without deciding its side (`test_knowledge_contracts.py`).

The enumeration is read from the SOURCE with `ast`, never from a list kept beside it: a
door added to `index_build.py` that opens the base and never asks is red here before it is
red anywhere else. The sentinel tests (`test_knowledge_index_invalidation.py`,
`test_knowledge_search_service.py`, `test_knowledge_render.py`) then prove that each named
consumer really repeats the seam's answer.
"""

from __future__ import annotations

import ast
from pathlib import Path

import xbrain.knowledge

KNOWLEDGE = Path(xbrain.knowledge.__file__).parent


def _called_names(function: ast.FunctionDef) -> set[str]:
    """Every name this function CALLS — `f(...)` and `module.f(...)` alike."""
    names: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


def _functions() -> dict[tuple[str, str], set[str]]:
    """`{(module, function): names it calls}` over the whole knowledge package."""
    out: dict[tuple[str, str], set[str]] = {}
    for path in sorted(KNOWLEDGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                out[(path.stem, node.name)] = _called_names(node)
    return out


def _callers_of(functions: dict[tuple[str, str], set[str]], name: str) -> set[tuple[str, str]]:
    return {key for key, called in functions.items() if name in called}


# ---------------------------------------------------------------------------
# seam (a) — the doors
# ---------------------------------------------------------------------------

# Every function that opens `knowledge.db`. `open_index` is the one door function and the
# only place `sqlite3.connect` touches the file; these are its callers. `build` is the ONE
# creator (`create=True`, G-2) and is exempt from the existence half: it unlinks and creates.
DOORS: frozenset[tuple[str, str]] = frozenset(
    {
        ("index_build", "build"),
        ("index_build", "update"),
        ("index_build", "_index_contents"),  # `status`
        ("index_store", "open_for_query"),  # `search`
    }
)
CREATOR = ("index_build", "build")


def test_every_function_that_opens_the_base_is_a_declared_door() -> None:
    """A new opener of the base — a fifth door — is red here, not in a gate (U-2).

    The set is asserted EQUAL, in both directions: a door that stops opening the base is as
    much a drift as one that starts.
    """
    functions = _functions()
    assert _callers_of(functions, "open_index") == set(DOORS)
    # `sqlite3.connect` on a FILE lives in exactly one function; `:memory:` is the harness's.
    file_connectors = {
        key
        for key in _callers_of(functions, "connect")
        if key != ("index_schema", "open_memory_index")
    }
    assert file_connectors == {("index_schema", "open_index")}


def test_every_door_but_the_creator_asks_both_halves_of_the_question() -> None:
    """Seam (a) has two halves and `status` had skipped the first (U-2, round 07):
    `require_database` — «is there a base?», whose sentence names `build --force` when a
    manifest is standing — and `describe_base`/`require_consistent` — «does the manifest
    describe it?». Each door names both; a door that tests `exists()` itself, or counts the
    planes itself, is red here, and the sentinel tests in
    `test_knowledge_index_invalidation.py` prove the named calls are the ones answering.

    Seen red on `9dfa34e`: `_index_contents` called neither `require_database` nor
    `describe_base` before returning its three empties.
    """
    functions = _functions()
    for door in sorted(DOORS - {CREATOR}):
        called = functions[door]
        assert "require_database" in called, f"{door} does not ask whether the base exists"
        assert called & {"describe_base", "require_consistent"}, (
            f"{door} does not ask whether the manifest describes the base"
        )
    creator = functions[CREATOR]
    # The creator's half of the seam is the WRITER: a manifest sealed through the reader's
    # own validation (B1), so nothing `build` writes is a document every door would refuse.
    assert "write_manifest" in creator
    assert "write_manifest" in functions[("index_build", "update")]


# ---------------------------------------------------------------------------
# seam (b) — the served fragment
# ---------------------------------------------------------------------------

# Every function that constructs a served fragment. `_chunk` builds the chunk `get` and the
# writer emit; `_match` builds the `SearchMatch` `search` returns. Both narrow the surface's
# locator through `fragment_locator` — the fabricated locator of A-1 was a `_match` that did
# not — and the chunk's evidence fingerprint is built through `chunk_evidence` wherever it is
# computed OR recomputed (U-5), so what the emitter hashes is what the verifier checks.
FRAGMENT_BUILDERS: frozenset[tuple[str, str]] = frozenset(
    {("chunking", "_chunk"), ("search_service", "_match")}
)
EVIDENCE_HASHERS: frozenset[tuple[str, str]] = frozenset(
    {("chunking", "_chunk"), ("index_store", "verify_fingerprints")}
)


def test_every_function_that_builds_a_served_fragment_narrows_through_one_locator() -> None:
    """A new constructor of `KnowledgeChunk`/`SearchMatch` is red here until it narrows
    through `fragment_locator`; the sentinel test in `test_knowledge_search_service.py`
    then proves the named ones really call it."""
    functions = _functions()
    builders = _callers_of(functions, "KnowledgeChunk") | _callers_of(functions, "SearchMatch")
    assert builders == set(FRAGMENT_BUILDERS)
    for builder in sorted(FRAGMENT_BUILDERS):
        assert "fragment_locator" in functions[builder], f"{builder} narrows its own locator"
