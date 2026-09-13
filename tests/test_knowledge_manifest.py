# tests/test_knowledge_manifest.py
"""The manifest contract — the document that SEALS the four fingerprints and the cheap signal
(Plan 02 §2 «`manifest.json` (spec §5.6, campo a campo)» and §3; spec §5.6).

The argument for the two signals — what the cheap one can answer, what it cannot, and the
direction it fails in — is stated once, in `index_build.py`'s module docstring, and is not
restated here. What IS here is the half 02.6b adds: the document those values are written to,
read back from, and REFUSED by.

Nothing in this file builds an index, updates one or queries one. `build` / `update` / `status`
/ `search` / `get` are 02.7's and later; this child ships the contract those commands seal and
read, so every test below goes through `Manifest`, `write_manifest` and `load_manifest` alone.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.knowledge import index_build, index_schema
from xbrain.knowledge.chunking import DEFAULT_CHUNKER_PARAMS, ChunkerParams
from xbrain.knowledge.ids import CHUNKER_VERSION, SURFACE_VERSION

UTC = timezone.utc

SEALED_SIGNAL = index_build.StoreSignal(
    items_json_mtime_ns=1756600020123456789,
    items_json_size=17599352,
    vocab_yaml_mtime_ns=1756600018004112233,
    vocab_yaml_size=7084,
    topics_json_mtime_ns=1756600019551020304,
    topics_json_size=158798,
)

# The `graph` block Plan 04 §1.4 adds, written out by hand rather than built by `graph_block`,
# so the two sides of a comparison are not one. Values distinct from any default.
GRAPH_BLOCK = {
    "algorithm_version": "topic-cooccurrence/v1",
    "min_shared_items": 3,
    "min_weight": 0.05,
    "max_neighbors_per_node": 5,
    "edges": 7,
}


def _manifest(**overrides: object) -> index_build.Manifest:
    """A manifest every door accepts, so each test can move exactly one thing."""
    base: dict[str, object] = {
        "schema_version": index_schema.SCHEMA_VERSION,
        "built_at": datetime(2026, 8, 31, 10, 0, tzinfo=UTC),
        "store_fingerprint": "a" * 64,
        "store_signal": SEALED_SIGNAL,
        "vocab_fingerprint": "b" * 64,
        "topics_fingerprint": "c" * 64,
        "surface_version": SURFACE_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "chunker_params": dataclasses.asdict(DEFAULT_CHUNKER_PARAMS),
        "counts": dict.fromkeys(index_build.COUNT_PLANES, 0),
        "skipped": dict.fromkeys(index_build.SKIPPED_CAUSES, 0),
        "failed": [],
        "embeddings": None,
        "graph": dict(GRAPH_BLOCK),
    }
    return index_build.Manifest(**{**base, **overrides})  # type: ignore[arg-type]


def _document(**overrides: object) -> dict[str, object]:
    """The manifest as JSON, so a test can plant a value no writer would ever emit."""
    return {**_manifest().to_dict(), **overrides}


# ---------------------------------------------------------------------------
# The field set — spec §5.6 in ONE place, and the writer bound to it
# ---------------------------------------------------------------------------


def test_the_document_declares_exactly_the_fields_the_contract_names() -> None:
    """`MANIFEST_FIELDS` is the schema, and BOTH sides are bound to it (rule 5).

    Asserting the written document against `MANIFEST_FIELDS` alone would be rule 1's row 4 —
    both sides out of this module, satisfied by whatever the constant happens to say. So the
    constant is also asserted against the names spec §5.6 enumerates, written out here by hand:
    that literal is the only side of the comparison that does not come from `index_build`.
    """
    from_spec = {
        "schema_version",
        "built_at",
        "store_fingerprint",
        "store_signal",
        "vocab_fingerprint",
        "topics_fingerprint",
        "surface_version",
        "chunker_version",
        "chunker_params",
        "embeddings",
        "counts",
        "skipped",
        "failed",
        "graph",  # Plan 04 §1.4, not spec §5.6
    }
    assert index_build.MANIFEST_FIELDS == from_spec
    assert set(_manifest().to_dict()) == from_spec
    assert {f.name for f in dataclasses.fields(index_build.Manifest)} == from_spec


def test_the_nested_key_sets_are_read_off_the_code_they_describe() -> None:
    """`counts` counts PLANES the schema declares and `chunker_params` the chunker's own fields.

    A plane renamed in the DDL, or a parameter added to `ChunkerParams`, must not leave a
    manifest declaring a key nothing counts — so neither set is a hand-kept list here either.
    `skipped` is the one set that IS enumerated: spec §5.6 names its four causes and no code
    object carries them yet (the counters are 02.7's), so the literal is the contract.
    """
    assert index_build.COUNT_PLANES <= index_schema.TABLES
    assert index_build.COUNT_PLANES == {"items", "topics", "surfaces", "chunks", "profiles"}
    assert index_build.CHUNKER_PARAM_NAMES == {f.name for f in dataclasses.fields(ChunkerParams)}
    assert index_build.SKIPPED_CAUSES == {
        "empty_text",
        "decorative",
        "no_speech",
        "failed_sources",
    }


# ---------------------------------------------------------------------------
# The six signal fields, through the document
# ---------------------------------------------------------------------------


def test_the_six_signal_fields_survive_the_round_trip_by_name_and_by_value() -> None:
    """Each of the six lands under its OWN key and comes back to the SAME field.

    The values are six DISTINCT integers on purpose: a serialiser that paired
    `vocab_yaml_mtime_ns` with `topics_json_mtime_ns` round-trips perfectly whenever the
    values happen to be equal, and a manifest built from an unmoved corpus is exactly where
    they would be. `StoreSignal` is frozen and ordered (pinned in
    `test_knowledge_index_build.py`), so a REORDER here is a silent re-binding of every value.
    """
    document = SEALED_SIGNAL.to_dict()
    assert document == {
        "items_json_mtime_ns": 1756600020123456789,
        "items_json_size": 17599352,
        "vocab_yaml_mtime_ns": 1756600018004112233,
        "vocab_yaml_size": 7084,
        "topics_json_mtime_ns": 1756600019551020304,
        "topics_json_size": 158798,
    }
    assert index_build.StoreSignal.from_dict(document) == SEALED_SIGNAL
    assert index_build.StoreSignal.from_dict(json.loads(json.dumps(document))) == SEALED_SIGNAL


@pytest.mark.parametrize("omitted", sorted(index_build.SIGNAL_FIELDS))
def test_a_signal_missing_any_one_field_is_refused_and_never_rehydrated_with_zeros(
    omitted: str,
) -> None:
    """THE ROUND-05 DEFECT, REINSTALLED BY OMISSION, IS WHAT THIS REFUSES (Plan 02 §3).

    Zeros are what an ABSENT input reads as, so a reader that filled the four vocab/topics
    entries with `0` would declare them absent — and two such manifests compare EQUAL forever,
    however `vocab.yaml` moves. The implementation this replaces did exactly that: its four
    later fields carried `= 0` and `from_dict` read required-vs-optional off those defaults.

    A manifest that declares less than the schema is INCOMPATIBLE, not lenient (spec §9.3), and
    §2 already says what happens to one: the whole document is refused with the rebuild advice.
    There is no legacy record to rehydrate — this child is the first thing in the tree that
    seals a manifest at all — so tolerating a short signal would buy compatibility with nothing.

    Parametrised over all six because a reader can be strict about `items.json` and lax about
    the other two, which is precisely the shape the defect had.
    """
    short = {k: v for k, v in SEALED_SIGNAL.to_dict().items() if k != omitted}

    with pytest.raises(index_schema.IndexIncompatibleError) as caught:
        index_build.StoreSignal.from_dict(short)
    assert omitted in str(caught.value)
    assert index_schema.REBUILD_ADVICE in str(caught.value)


def test_an_undeclared_signal_key_is_refused_because_nothing_could_compare_it() -> None:
    """A seventh entry is a signal from another version, and this one cannot compare it.

    Accepting and ignoring it is the fail-open: the writer that emitted it watched a fourth
    input, and a reader that drops the key certifies itself current over an input it never
    looked at.
    """
    with pytest.raises(index_schema.IndexIncompatibleError, match="guardrails_yaml_size"):
        index_build.StoreSignal.from_dict({**SEALED_SIGNAL.to_dict(), "guardrails_yaml_size": 0})


def test_the_unstattable_sentinel_cannot_be_SEALED_by_a_hand_edited_manifest() -> None:
    """`(-1, -1)` IN A MANIFEST WOULD COMPARE EQUAL TO AN OBSTRUCTED INPUT, AND THAT IS FATAL.

    The contract (Plan 02 §3, «los dos valores reservados») rests on the sentinel being
    UNSEALABLE: `_read_bound` raises on every obstruction that produces it, so no writer can
    emit it, and a query that reads `UNSTATTABLE` therefore compares UNEQUAL and declares the
    index behind. A hand-edited `manifest.json` is the one path that bypasses the writer — and
    a stored `-1` size would meet the live `-1` and certify the index current over an input
    nobody can stat. The reader is what closes it, at the only boundary a hand edit crosses.

    Asserted on the SIZE and not on the pair: `UNSTATTABLE[1] < 0` is the half a real `st_size`
    cannot forge, which is the same reason `test_knowledge_index_build.py` asserts the size
    there rather than `== UNSTATTABLE` (rule 1, row 4 — both sides out of one module).
    """
    assert index_build.UNSTATTABLE[1] < 0
    for field in sorted(index_build.SIGNAL_FIELDS):
        if not field.endswith("_size"):
            continue
        with pytest.raises(index_schema.IndexIncompatibleError, match=field):
            index_build.StoreSignal.from_dict({**SEALED_SIGNAL.to_dict(), field: -1})


def test_a_pre_epoch_mtime_is_sealable_because_a_real_stat_produces_one() -> None:
    """THE MIRROR OF THE TEST ABOVE, AND WHY THE RULE IS NOT «NO NEGATIVES».

    A blanket non-negative check over the six — which is what the counters get — would refuse
    a signal a real `os.stat` can produce: `st_mtime_ns` IS negative for a file dated before
    1970, and the repo's own measurement of the sentinel («un fichero vacío con
    `os.utime(p, ns=(-1, -1))` stat'ea exactamente `(-1, 0)`») is that very case. Refusing it
    would turn a legitimate corpus into an unbuildable one, so the asymmetry is deliberate:
    ANY sign on the mtime, non-negative on the size.
    """
    pre_epoch = {**SEALED_SIGNAL.to_dict(), "items_json_mtime_ns": -1, "items_json_size": 0}
    assert index_build.StoreSignal.from_dict(pre_epoch).items_json_mtime_ns == -1


def test_the_size_half_is_three_fields_and_the_mtime_half_the_other_three() -> None:
    """The split above is made by NAME, so the names are pinned where the rule reads them.

    Renaming a field to something that stops ending in `_size` would silently move it into the
    permissive half and make the sentinel sealable again, with every other test still green.
    """
    sizes = {n for n in index_build.SIGNAL_FIELDS if n.endswith("_size")}
    mtimes = {n for n in index_build.SIGNAL_FIELDS if n.endswith("_mtime_ns")}
    assert len(sizes) == len(mtimes) == 3
    assert sizes | mtimes == set(index_build.SIGNAL_FIELDS)
    assert index_build.SIGNAL_SIZE_FIELDS == sizes


def test_an_absent_input_seals_zeros_and_reads_back_as_the_same_absence() -> None:
    """`(0, 0)` is a LEGITIMATE sealed reading — an input that was absent when the index was
    built — so the reader must accept it. It is the sentinel that is unsealable, not absence.
    """
    zeroed = index_build.StoreSignal(0, 0, 0, 0, 0, 0)
    assert index_build.StoreSignal.from_dict(zeroed.to_dict()) == zeroed


@pytest.mark.parametrize("planted", [True, "17599352", 17599352.0, None, [1]])
def test_a_signal_value_that_is_not_an_integer_is_refused_never_coerced(
    planted: object,
) -> None:
    """`True` IS AN `int` TO `isinstance`, AND A JSON `true` WHERE A SIZE BELONGS IS NOT A ONE.

    `int("17599352")` accepts the string too, and a float compares unequal to the integer a
    live stat returns, so a coerced manifest would declare the index behind on every query
    for a reason no operator could see. The check is `type(v) is int`.
    """
    with pytest.raises(index_schema.IndexIncompatibleError, match="items_json_size"):
        index_build.StoreSignal.from_dict({**SEALED_SIGNAL.to_dict(), "items_json_size": planted})


def test_a_store_signal_that_is_not_an_object_is_refused_before_it_is_subscripted() -> None:
    """A hand-edited document can hold a LIST where the signal belongs, and `set()` of a list
    of the six field NAMES has exactly those names as its elements — so a totality check that
    ran before the type check would pass it and the first subscript would raise `TypeError`
    out of the door, as a traceback. The type is checked first.
    """
    for planted in (sorted(index_build.SIGNAL_FIELDS), 3, None, "six"):
        with pytest.raises(index_schema.IndexIncompatibleError, match="store_signal"):
            index_build.StoreSignal.from_dict(planted)


# ---------------------------------------------------------------------------
# The document as a whole — total and closed
# ---------------------------------------------------------------------------


def test_a_sealed_manifest_round_trips_through_the_reader_unchanged(tmp_path: Path) -> None:
    """The whole document, written and read back, field for field.

    `built_at` is compared as an INSTANT and not as a string: the writer renders
    `datetime.isoformat()` and the reader parses it, and `2026-08-31T10:00:00Z` comes back as
    `...+00:00` — the same instant, a different spelling. Asserting the strings would pin the
    spelling and call an equal instant a difference.
    """
    original = _manifest()
    index_build.write_manifest(tmp_path, original)

    loaded = index_build.load_manifest(tmp_path)
    assert loaded == original
    assert loaded.built_at == original.built_at
    assert loaded.store_signal == SEALED_SIGNAL
    assert (
        json.loads(index_schema.manifest_path(tmp_path).read_text(encoding="utf-8"))["store_signal"]
        == SEALED_SIGNAL.to_dict()
    )


@pytest.mark.parametrize("omitted", sorted(index_build.MANIFEST_FIELDS))
def test_a_manifest_declaring_less_than_the_schema_is_refused_entirely(omitted: str) -> None:
    """Spec §9.3: an incompatible manifest is never queried PARTIALLY.

    The first version of this reader checked the top-level key set and cast what sat under it,
    so a manifest whose `counts` was `{}` loaded as compatible and the consistency check,
    iterating whatever `counts` offered, compared nothing. Every field is required; a document
    short of one is refused with the command that repairs it.
    """
    short = {k: v for k, v in _document().items() if k != omitted}

    with pytest.raises(index_schema.IndexIncompatibleError) as caught:
        index_build.Manifest.from_dict(short)
    assert omitted in str(caught.value)
    assert index_schema.REBUILD_ADVICE in str(caught.value)


def test_a_manifest_declaring_MORE_than_the_schema_is_refused_through_every_door() -> None:
    """THE TOP-LEVEL DOCUMENT WAS TOTAL BUT NOT CLOSED, AND THE NESTED MAPPINGS HID IT.

    `_closed_keys` refuses an undeclared key in `counts`, `skipped`, `chunker_params` and
    `store_signal`, and four tests pinned exactly that — so «total and closed» read as
    covered while the DOCUMENT ITSELF only ever checked what was MISSING. Measured on the
    untouched tree: a valid manifest plus `"future_plane": {...}` loaded through
    `Manifest.from_dict` AND through `load_compatible_manifest`, which returned it as
    compatible, with the extra key silently dropped.

    That is the fail-open this child's own contract rules out. A document carrying a key
    this code does not know was written by something that watches a plane this code cannot
    see; accepting it certifies the index current over exactly that plane, which is the
    round-05 defect one level up — and `search` would then answer over it saying nothing.

    THE PUBLIC FILE LOADER IS ASSERTED, NOT ONLY `from_dict`. The reader is reachable from
    disk through two doors and a caller only ever meets those; a guard proven on the
    in-memory classmethod alone leaves the ones operators actually use unproven.

    Seen red at `fdad115` on all three assertions below: `DID NOT RAISE`.
    """
    extra = {**_document(), "future_plane": {"fingerprint": "unknown"}}

    with pytest.raises(index_schema.IndexIncompatibleError) as caught:
        index_build.Manifest.from_dict(extra)
    assert "future_plane" in str(caught.value)
    assert index_schema.REBUILD_ADVICE in str(caught.value)


def test_an_unknown_top_level_key_is_refused_through_the_public_file_loader(
    tmp_path: Path,
) -> None:
    """The same gap as read from DISK — `load_manifest` and the compatibility door.

    `load_compatible_manifest` is the one a query would call, and it RETURNED a manifest
    from a newer writer as compatible. Refusing it is what spec §9.3 means by never
    querying an incompatible manifest partially.
    """
    index_build.write_manifest(tmp_path, _manifest())
    path = index_schema.manifest_path(tmp_path)
    path.write_text(
        json.dumps({**json.loads(path.read_text(encoding="utf-8")), "future_plane": {}}),
        encoding="utf-8",
    )

    with pytest.raises(index_schema.IndexIncompatibleError, match="future_plane"):
        index_build.load_manifest(tmp_path)
    with pytest.raises(index_schema.IndexIncompatibleError, match="future_plane"):
        index_build.load_compatible_manifest(tmp_path)


def test_a_writer_whose_to_dict_grows_a_key_cannot_seal_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE OTHER DIRECTION OF THE SAME GAP: the round-trip could not SEE schema drift.

    `write_manifest` validates through `Manifest.from_dict` before a byte lands, and that
    guard is what stops a writer whose shape drifted from the reader's. With the document
    open at the top level the round-trip accepted an ADDED key, so a drifted `to_dict`
    wrote it to disk and every later door dropped it in silence — the guard passing on the
    exact defect it exists to catch.

    Both halves of the writer's promise are asserted, because they fail differently: over a
    FRESH directory nothing may be created, and over an EXISTING manifest the previous
    bytes must survive intact. A writer that truncated first and validated second satisfies
    neither, and satisfies a bare `pytest.raises` perfectly.

    Seen red at `fdad115`: `DID NOT RAISE`, with the drifted key on disk.
    """
    index_build.write_manifest(tmp_path, _manifest())
    before = index_schema.manifest_path(tmp_path).read_bytes()

    drifted = {**_document(), "future_plane": {}}
    monkeypatch.setattr(index_build.Manifest, "to_dict", lambda self: drifted)

    with pytest.raises(index_schema.IndexIncompatibleError, match="future_plane"):
        index_build.write_manifest(tmp_path, _manifest())
    assert index_schema.manifest_path(tmp_path).read_bytes() == before, "previous bytes intact"

    with pytest.raises(index_schema.IndexIncompatibleError, match="future_plane"):
        index_build.write_manifest(tmp_path / "fresh", _manifest())
    assert not index_schema.manifest_path(tmp_path / "fresh").exists()


def test_the_closure_rule_has_one_definition_shared_by_the_document_and_its_mappings() -> None:
    """Rule 5: «exactly the declared keys» is ONE function, not two that drift.

    The top-level document and the four nested mappings refuse for the same reason and
    must not be able to disagree about what refusing means — which is how the document
    came to be total while every mapping under it was closed.
    """
    assert index_build._closure_gap({"a", "b"}, frozenset({"a", "c"})) == (["c"], ["b"])
    assert index_build._closure_gap({"a"}, frozenset({"a"})) == ([], [])


@pytest.mark.parametrize("planted", [["schema_version", "built_at"], 3, None, True, "manifest"])
def test_a_document_that_is_not_a_json_object_leaves_by_the_same_door(planted: object) -> None:
    """THE TOTALITY CHECK IS NOT A TYPE CHECK, and reading it as one is how a list got through.

    `MANIFEST_FIELDS - set(raw)` does not raise on a non-mapping — it takes the ELEMENTS — so a
    hand-edited `manifest.json` holding the JSON list of the field NAMES passed the guard with
    `missing` empty and the first subscript raised `TypeError: list indices must be integers`,
    out of every door, as a traceback. A top-level `3`, `null` or `true` was worse: `set()` of
    them raises from the guard LINE ITSELF, before one field was read.
    """
    with pytest.raises(index_schema.IndexIncompatibleError) as caught:
        index_build.Manifest.from_dict(planted)
    assert "no es un objeto JSON" in str(caught.value)
    assert index_schema.REBUILD_ADVICE in str(caught.value)


@pytest.mark.parametrize(
    "field_name, declared",
    [
        ("counts", index_build.COUNT_PLANES),
        ("skipped", index_build.SKIPPED_CAUSES),
        ("chunker_params", index_build.CHUNKER_PARAM_NAMES),
    ],
)
def test_a_nested_mapping_must_be_closed_as_well_as_total(
    field_name: str, declared: frozenset[str]
) -> None:
    """Closed as well as total: an undeclared key is refused, because nothing could compare it.

    A plane or a cause the reader does not know about is a document from another version, and
    dropping it is the fail-open — the consistency check would then compare a subset and call
    an amputated index healthy.
    """
    one = sorted(declared)[0]

    with pytest.raises(index_schema.IndexIncompatibleError, match=one):
        index_build.Manifest.from_dict(
            _document(**{field_name: {k: 0 for k in declared if k != one}})
        )
    with pytest.raises(index_schema.IndexIncompatibleError, match="no declaradas"):
        index_build.Manifest.from_dict(
            _document(**{field_name: {**dict.fromkeys(declared, 0), "vectors": 0}})
        )


@pytest.mark.parametrize("field_name", ["counts", "skipped", "chunker_params"])
@pytest.mark.parametrize("planted", [True, -1, "0", 1.0, None])
def test_a_counter_that_is_not_a_non_negative_integer_is_refused(
    field_name: str, planted: object
) -> None:
    """`True` is an `int` to `isinstance`, `int("0")` accepts the string, and a NEGATIVE count
    is not a count. Here — unlike the signal — non-negative IS the rule for every value: a
    plane holding minus one row, or a chunker whose `max_chars` is minus one, is a malformed
    document and not a measurement.
    """
    declared = getattr(
        index_build,
        {"counts": "COUNT_PLANES", "skipped": "SKIPPED_CAUSES"}.get(
            field_name, "CHUNKER_PARAM_NAMES"
        ),
    )
    key = sorted(declared)[0]

    with pytest.raises(index_schema.IndexIncompatibleError, match=key):
        index_build.Manifest.from_dict(
            _document(**{field_name: {**dict.fromkeys(declared, 0), key: planted}})
        )


# The `VectorSpec` a manifest declares, in the shape spec §5.6 asks for. Written out here
# rather than built from the dataclass, so the two sides of the comparison are not one.
SPEC_BLOCK = {
    "model": "intfloat/multilingual-e5-base",
    "dimension": 768,
    "normalized": True,
    "query_prefix": "query: ",
    "passage_prefix": "passage: ",
}


def test_the_embeddings_slot_is_null_or_a_whole_vector_spec() -> None:
    """`null` or the FIVE fields, total and closed — and 03.4 is when that became true.

    02.6 declared the slot and checked only its SHAPE, which was the honest thing for a child
    that could not produce one: a schema for a payload nothing emits is prose in the column
    where a guard belongs. This tree writes it (`index_build.VectorBuild`), so the guard is
    owed, and the previous version of this test — which asserted that `{"model", "dimension"}`
    LOADS — pinned the permissiveness rather than the contract.

    THE FIELD THAT MAKES IT MATTER IS THE PREFIX. `query_prefix` / `passage_prefix` are
    properties of the MODEL, so a block that omitted one would leave every query embedded bare
    against a corpus embedded prefixed: still well-formed, still unit-length, and answering a
    question nobody asked. Nothing downstream can see that; this reader is the only thing that
    ever could.
    """
    assert _manifest().embeddings is None
    assert index_build.Manifest.from_dict(_document()).embeddings is None
    assert index_build.Manifest.from_dict(_document(embeddings=SPEC_BLOCK)).embeddings == (
        SPEC_BLOCK
    )

    for planted in (3, "e5", [1, 2]):
        with pytest.raises(index_schema.IndexIncompatibleError, match="embeddings"):
            index_build.Manifest.from_dict(_document(embeddings=planted))

    # TOTAL: each field removed in turn is refused BY NAME, so no single omission survives.
    for field_name in SPEC_BLOCK:
        with pytest.raises(index_schema.IndexIncompatibleError, match=field_name):
            index_build.Manifest.from_dict(
                _document(embeddings={k: v for k, v in SPEC_BLOCK.items() if k != field_name})
            )

    # CLOSED: a property this code cannot honour is a plane written by something else.
    with pytest.raises(index_schema.IndexIncompatibleError, match="quantization"):
        index_build.Manifest.from_dict(_document(embeddings={**SPEC_BLOCK, "quantization": "int8"}))

    # TYPED: `"768"` and `normalized: 1` are hand edits, not specs. `True` IS an `int` in
    # Python, so a bare `int` check would take `dimension: true` as a width of one.
    for broken in (
        {**SPEC_BLOCK, "dimension": "768"},
        {**SPEC_BLOCK, "dimension": True},
        {**SPEC_BLOCK, "dimension": 0},
        {**SPEC_BLOCK, "normalized": 1},
        {**SPEC_BLOCK, "model": 5},
    ):
        with pytest.raises(index_schema.IndexIncompatibleError, match="embeddings"):
            index_build.Manifest.from_dict(_document(embeddings=broken))


def test_the_graph_block_is_total_closed_and_typed() -> None:
    """Plan 04 §1.4: the algorithm and thresholds `graph_edges` was derived under, or a refusal.

    `update` rewrites the graph when this block disagrees with its options, so a block missing a
    threshold compares less than it should and leaves a moved threshold's plane standing, and a
    `null` would let a writer that forgot the graph seal a manifest anyway. Same rules as the
    `embeddings` slot: total, closed, typed — `True` is an `int` and `"3"` is a hand edit.
    """
    assert index_build.GRAPH_FIELDS == set(GRAPH_BLOCK)
    assert index_build.Manifest.from_dict(_document()).graph == GRAPH_BLOCK

    for planted in (None, 3, [1]):
        with pytest.raises(index_schema.IndexIncompatibleError, match="graph"):
            index_build.Manifest.from_dict(_document(graph=planted))
    for field_name in GRAPH_BLOCK:
        with pytest.raises(index_schema.IndexIncompatibleError, match=field_name):
            index_build.Manifest.from_dict(
                _document(graph={k: v for k, v in GRAPH_BLOCK.items() if k != field_name})
            )
    with pytest.raises(index_schema.IndexIncompatibleError, match="max_supporting_item_ids"):
        index_build.Manifest.from_dict(
            _document(graph={**GRAPH_BLOCK, "max_supporting_item_ids": 20})
        )
    for broken in (
        {**GRAPH_BLOCK, "algorithm_version": 1},
        {**GRAPH_BLOCK, "min_shared_items": "3"},
        {**GRAPH_BLOCK, "min_shared_items": True},
        {**GRAPH_BLOCK, "min_weight": "0.05"},
        {**GRAPH_BLOCK, "min_weight": False},
        {**GRAPH_BLOCK, "max_neighbors_per_node": 5.0},
        {**GRAPH_BLOCK, "edges": -1},
    ):
        with pytest.raises(index_schema.IndexIncompatibleError, match="graph"):
            index_build.Manifest.from_dict(_document(graph=broken))


def test_the_failed_list_is_a_list_of_text_objects_or_the_document_is_refused() -> None:
    """`failed` is spec §5.6's *chunks omitidos o fallidos*, and it reaches an operator's
    screen: a nested structure where a string belongs would print as a Python repr from
    inside a command, so the shape is fixed where it is read rather than where it is shown.
    """
    assert index_build.Manifest.from_dict(_document(failed=[])).failed == []
    assert index_build.Manifest.from_dict(
        _document(failed=[{"item_id": "1", "reason": "empty_text"}])
    ).failed == [{"item_id": "1", "reason": "empty_text"}]

    for planted in ({}, "none", [3], [{"item_id": 1}], [{"item_id": ["a"]}]):
        with pytest.raises(index_schema.IndexIncompatibleError, match="failed"):
            index_build.Manifest.from_dict(_document(failed=planted))


@pytest.mark.parametrize("planted", ["not-an-instant", "", 3, None, "2026-13-01T00:00:00Z"])
def test_a_built_at_that_is_not_an_instant_is_actionable_and_not_a_value_error(
    planted: object,
) -> None:
    """`datetime.fromisoformat` raises a bare `ValueError` naming the string and no command.
    Spec §9.3 asks for an error an operator can act on, so the field is named and the rebuild
    advice rides with it.
    """
    with pytest.raises(index_schema.IndexIncompatibleError, match="built_at"):
        index_build.Manifest.from_dict(_document(built_at=planted))


# ---------------------------------------------------------------------------
# The writer, the reader and the doors they answer through
# ---------------------------------------------------------------------------


def test_the_writer_round_trips_through_the_reader_before_one_byte_lands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A BUILD CANNOT SEAL A DOCUMENT EVERY LATER DOOR WOULD REFUSE, AND THE PROOF IS THAT
    NOTHING IS ON DISK AFTERWARDS.

    Asserting only that `write_manifest` raises would be satisfied by a writer that raised
    AFTER writing — which is the state this guard exists to prevent, an index carrying a
    manifest no door accepts and `build --force` as the only way out. So the assertion is on
    the FILE: the directory is still empty.

    The malformed value is planted by making `to_dict` emit it, which is the one shape the
    dataclass cannot refuse on its own: a writer whose serialisation drifted from the reader's
    schema. Seen red by removing the `Manifest.from_dict(json.loads(document))` line.
    """
    drifted = {**_document(), "counts": {"items": 0}}
    monkeypatch.setattr(index_build.Manifest, "to_dict", lambda self: drifted)

    with pytest.raises(index_schema.IndexIncompatibleError, match="counts"):
        index_build.write_manifest(tmp_path, _manifest())
    assert not index_schema.manifest_path(tmp_path).exists()


def test_the_writer_creates_the_index_directory_it_writes_into(tmp_path: Path) -> None:
    """The manifest is written LAST, and on a first build nothing has made the directory yet."""
    target = tmp_path / "index"
    index_build.write_manifest(target, _manifest())
    assert index_schema.manifest_path(target).is_file()


def test_a_missing_manifest_names_the_command_that_builds_the_index(tmp_path: Path) -> None:
    """An index that was never built is not a corrupt one, and the advice differs: `build`,
    not `build --force`. `IndexMissingError` is the type every door already translates.
    """
    with pytest.raises(index_schema.IndexMissingError) as caught:
        index_build.load_manifest(tmp_path)
    assert "xbrain index build" in str(caught.value)
    assert str(index_schema.manifest_path(tmp_path)) in str(caught.value)


@pytest.mark.parametrize("raw, label", [(b"{not json", "decode"), (b"\xff\xfe{}", "undecodable")])
def test_a_corrupt_manifest_document_is_actionable_and_never_a_traceback(
    tmp_path: Path, raw: bytes, label: str
) -> None:
    """A `JSONDecodeError` — or a `UnicodeDecodeError`, which is not even a subclass of it —
    out of a query is a traceback naming no command. Both leave by the door that names the
    rebuild. Nothing is repaired: Plan 02 §11, a corrupt base is rebuilt, never patched.
    """
    index_schema.manifest_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    index_schema.manifest_path(tmp_path).write_bytes(raw)

    with pytest.raises(index_schema.IndexIncompatibleError) as caught:
        index_build.load_manifest(tmp_path)
    assert index_schema.REBUILD_ADVICE in str(caught.value), label


def test_the_manifest_seals_the_signal_the_loader_BOUND_and_never_a_later_stat(
    tmp_path: Path,
) -> None:
    """THE END-TO-END SHAPE THE WHOLE CHILD EXISTS FOR (Plan 02 §3, P1b).

    The signal that reaches the document is the one `load_index_inputs` took from the handles
    it parsed, not a `stat` of the paths at sealing time. Here the vocabulary is rewritten
    AFTER the load and BEFORE the seal — the window a real build has between reading the corpus
    and writing the manifest — and the sealed document must still describe the bytes that were
    parsed, so the next query compares UNEQUAL and the index declares itself behind.

    A writer that re-stat'ed the paths would seal the NEW vocabulary under the OLD rows and the
    comparison would come out EQUAL: stale evidence served with nothing declared, which is what
    spec §9.3 forbids and what `index_build`'s module docstring calls the whole design.
    """
    from xbrain.rubrics import save_vocab
    from xbrain.store import save_store, save_topic_pages
    from xbrain.models import Topic

    items, vocab, topics = (tmp_path / n for n in ("items.json", "vocab.yaml", "topics.json"))
    save_store({}, items)
    save_vocab([Topic(slug="ai", description="one")], vocab)
    save_topic_pages({}, topics)

    loaded = index_build.load_index_inputs(items, vocab, topics)
    save_vocab([Topic(slug="ai", description="a different description entirely")], vocab)

    index_build.write_manifest(tmp_path / "index", _manifest(store_signal=loaded.signal))
    sealed = index_build.load_manifest(tmp_path / "index").store_signal

    assert sealed == loaded.signal, "the sealed signal is the one bound to the parsed bytes"
    assert sealed != index_build.StoreSignal.of(items, vocab, topics)


# ---------------------------------------------------------------------------
# Compatibility — Plan 02 §2, «un manifest con … distinto … falla entera»
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name, planted",
    [
        ("schema_version", "1"),
        ("surface_version", "xbrain-knowledge-surface/v0"),
        ("chunker_version", "xbrain-knowledge-chunker/v1"),
    ],
)
def test_a_version_the_code_does_not_speak_refuses_the_whole_manifest(
    tmp_path: Path, field_name: str, planted: str
) -> None:
    """Spec §9.3: *manifest incompatible: no se consulta parcialmente.*

    Each of the three names a real failure, not a formality. A v3 base's chunk fingerprints
    were computed over a different projection, so every row would fail verification and the
    honest answer is the rebuild, never «22.286 chunks excluded». The refusal names WHICH
    version disagreed and with what, because «incompatible» alone is not actionable.
    """
    index_build.write_manifest(tmp_path, _manifest(**{field_name: planted}))

    with pytest.raises(index_schema.IndexIncompatibleError) as caught:
        index_build.load_compatible_manifest(tmp_path)
    assert field_name in str(caught.value)
    assert index_schema.REBUILD_ADVICE in str(caught.value)


def test_a_compatible_manifest_is_returned_and_not_merely_not_refused(tmp_path: Path) -> None:
    """The other direction, so the guard above cannot be satisfied by a door that refuses
    everything — which is what a `raise` with the condition inverted would look like.
    """
    index_build.write_manifest(tmp_path, _manifest())
    assert index_build.load_compatible_manifest(tmp_path) == _manifest()
    assert index_build.load_compatible_manifest(tmp_path, params=DEFAULT_CHUNKER_PARAMS)


def test_the_chunker_PARAMETERS_are_the_fourth_check_and_not_a_detail(tmp_path: Path) -> None:
    """PLAN 02 §7 SWEEPS `target x overlap`, AND THE SWEEP DOES NOT BUMP `CHUNKER_VERSION`.

    A base cut at `target=800` and queried under `target=1200` holds chunks whose ids RESOLVE
    and whose spans are not what they were — the worst shape, since nothing raises and the text
    behind a citable id is a different text. The version alone cannot see it, so the parameters
    are compared too, and only when the caller supplies them: a door that does not chunk has
    nothing to compare against and must not invent the defaults.
    """
    index_build.write_manifest(tmp_path, _manifest())
    swept = ChunkerParams(target=1200, max_chars=2000, overlap=150, min_chars=40)

    assert index_build.load_compatible_manifest(tmp_path) is not None
    with pytest.raises(index_schema.IndexIncompatibleError, match="chunker_params"):
        index_build.load_compatible_manifest(tmp_path, params=swept)


def test_a_forged_version_string_reaches_no_terminal_unquoted(tmp_path: Path) -> None:
    """The manifest is HAND-EDITABLE, and its version strings are interpolated into a sentence
    a command prints. Raw, a newline in `schema_version` stands at column 0 as a forged header
    and an ESC reaches the TTY. `!r` carries neither.
    """
    index_build.write_manifest(tmp_path, _manifest(schema_version="1\n\x1b[2JFAKE"))

    with pytest.raises(index_schema.IndexIncompatibleError) as caught:
        index_build.load_compatible_manifest(tmp_path)
    assert "\n" not in str(caught.value)
    assert "\x1b" not in str(caught.value)
    assert "\\n" in str(caught.value) and "\\x1b" in str(caught.value)


# ---------------------------------------------------------------------------
# LOW-1 — the one shape that escapes «a query always ANSWERS»
# ---------------------------------------------------------------------------


def test_a_path_with_an_embedded_nul_raises_where_the_three_doors_answer_empty(
    tmp_path: Path,
) -> None:
    """THE DEBT PLAN 02 §3 LEFT NAMED AND UNPINNED («Ningún test lo fija hoy», LOW-1).

    A NUL inside a path raises `ValueError` BEFORE any syscall, so it reaches neither
    `except FileNotFoundError` nor `except OSError` — it is the single hole in `_stat_signal`'s
    promise that a query always ANSWERS. Measured on this tree: `stat: embedded null character
    in path` out of both halves of the module, while `load_store` / `load_vocab` /
    `load_topic_pages` read it as `{}` / `[]` / `{}` because `Path.exists()` also answers False
    on a `ValueError`.

    THIS TEST PINS THE CURRENT BEHAVIOUR; IT DOES NOT CLOSE THE HOLE, and the distinction is
    the point. Broadening the guard to `except (OSError, ValueError)` would answer `(0, 0)` —
    ABSENT — for a string that is not a path at all, which is the fail-open direction this
    module is built never to take; `UNSTATTABLE` would at least warn, but the value belongs to
    an obstruction that was stat'ed, and inventing a third meaning for it is a contract change
    no consumer has asked for yet. The honest fix is a door that TRANSLATES it into an
    actionable error, and this child ships no such door — `search` is 02.8's.

    So what is asserted is exactly what is true: it RAISES, and the broadening mutation
    reddens here. Seen red under `except OSError` -> `except (OSError, ValueError)` in
    `_stat_signal`: `DID NOT RAISE`. The three doors are asserted in the same test because the
    divergence is the fact — a claim about one half is worth what the other half's assertion
    is worth.
    """
    from xbrain.rubrics import load_vocab
    from xbrain.store import load_store, load_topic_pages

    nul = tmp_path / "items\x00.json"
    ok = tmp_path / "vocab.yaml"
    ok.write_text("topics: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="null"):
        index_build._stat_signal(nul)
    with pytest.raises(ValueError, match="null"):
        index_build.StoreSignal.of(nul, ok, ok)
    with pytest.raises(ValueError, match="null"):
        index_build._read_bound(nul)

    assert load_store(nul) == {}
    assert load_vocab(nul) == []
    assert load_topic_pages(nul) == {}
