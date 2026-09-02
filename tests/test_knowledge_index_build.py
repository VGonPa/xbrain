# tests/test_knowledge_index_build.py
"""`index build` and its manifest (Plan 02 §2, §3, steps 3, 4, 7c, 9, 10c).

THE MANIFEST IS THE CONTRACT BETWEEN A BUILD AND EVERY LATER QUERY. Spec §5.6 enumerates
what it must record, and the enumeration is asserted as a SET rather than field by field, so
a field dropped from the writer goes red instead of quietly disappearing from a JSON document
nobody reads until a query answers under the wrong chunker.

TWO SIGNALS, TWO COSTS, TWO PLACES (B3). `store_signal` is an `os.stat` — mtime and size of
`data/items.json` — and it is what a QUERY can afford on every call. `store_fingerprint` is a
sha256 per item and costs loading the store, so it is paid only by `build`, `update` and
`status`. The cheap one says *the store moved*; the expensive one says *which items changed,
and how many*. A false positive on the cheap one costs one warning; a false negative costs
serving stale evidence as fresh, so it fails towards the warning.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.knowledge import index_build
from xbrain.knowledge.chunking import DEFAULT_CHUNKER_PARAMS
from xbrain.knowledge.ids import CHUNKER_VERSION, SURFACE_VERSION
from xbrain.knowledge.index_schema import (
    SCHEMA_VERSION,
    IndexIncompatibleError,
    IndexMissingError,
    manifest_path,
)
from xbrain.knowledge.lexical_fts import FTS_CONNECTIVE, FTS_TOKENIZE
from xbrain.models import Item, Topic, TopicPage

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def corpus() -> tuple[dict[str, Item], list[Topic], dict[str, TopicPage]]:
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    store = {k: Item.model_validate(v) for k, v in raw["items"].items()}
    vocab = [Topic.model_validate(v) for v in raw["vocab"].values()]
    pages = {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()}
    return store, vocab, pages


@pytest.fixture()
def workspace(tmp_path: Path, corpus) -> Path:
    """A data/ directory holding the fixture store, so `store_signal` has a real file."""
    store, _vocab, _pages = corpus
    data = tmp_path / "data"
    data.mkdir()
    (data / "items.json").write_text(
        json.dumps({k: v.model_dump(mode="json") for k, v in store.items()}), encoding="utf-8"
    )
    return data


def test_a_corrupt_manifest_is_an_actionable_error_not_a_json_traceback(workspace) -> None:
    """Plan 02 §11: a corrupt base or manifest names `index build --force`; nothing is repaired."""
    (workspace / "index").mkdir(parents=True)
    manifest_path(workspace / "index").write_text("{not json", encoding="utf-8")
    with pytest.raises(IndexIncompatibleError, match="index build --force"):
        index_build.load_compatible_manifest(workspace / "index")


# ---------------------------------------------------------------------------
# The fingerprint, and the population it has to reach (rule 6)
# ---------------------------------------------------------------------------


def test_the_item_fingerprint_changes_when_indexable_text_changes(corpus) -> None:
    """It hashes the SURFACES, not `(fetched_at, enriched_at)` as Plan 01 §10 sketched.

    Two reasons, and CLAUDE.md rule 6 is both of them. First, `content.fetched_at` cannot
    reach an item whose `content` is `None` — 960 of 2,404 in the real store (measured
    2026-09-01 on sha256 `f76341a3…`) — because there
    is nothing to stamp. Second, a timestamp is a PROXY: a summary edited by hand, or any
    repair that rewrites text without touching a clock, changes the indexable corpus and
    leaves the proxy unmoved, so the index would keep serving the old body under a
    fingerprint that says it is current.

    Hashing the emitted surface fingerprints answers the question directly — *did the text
    this index holds change?* — and it reaches every item, with or without `content`.
    """
    store, _vocab, _pages = corpus
    item = store["k02"]
    before = index_build.item_fingerprint(item)
    edited = item.model_copy(
        update={"enriched": item.enriched.model_copy(update={"summary": "otro resumen"})}
    )
    assert index_build.item_fingerprint(edited) != before
    assert index_build.item_fingerprint(item) == before, "and it is stable"


def test_the_item_fingerprint_reaches_an_item_with_no_content(corpus) -> None:
    """Step 6 / rule 6: the invalidation signal must reach the population it has to reach.

    `k02` carries no `content` at all, so a fingerprint over `content.fetched_at` would be
    constant for it forever. Seen red by hashing only the timestamps.
    """
    store, _vocab, _pages = corpus
    item = store["k02"]
    assert item.content is None, "this test is only meaningful on an item with no content"
    edited = item.model_copy(
        update={"enriched": item.enriched.model_copy(update={"summary": "otro resumen"})}
    )
    assert index_build.item_fingerprint(edited) != index_build.item_fingerprint(item)


def test_the_item_fingerprint_also_covers_the_filterable_metadata(corpus) -> None:
    """The index stores more than text: a changed author or topic changes what a filter returns.

    Ignoring them would leave `--author` answering from an author the store no longer records.
    """
    store, _vocab, _pages = corpus
    item = store["k02"]
    moved = item.model_copy(update={"source": "own_tweet"})
    assert index_build.item_fingerprint(moved) != index_build.item_fingerprint(item)


@pytest.mark.parametrize(
    "field, value",
    [
        ("author", {"handle": "someoneelse", "name": "Someone Else"}),
        ("title", "A different title"),
        ("language", "fr"),
        ("url", "https://x.com/othervoice/status/moved"),
    ],
)
def test_the_item_fingerprint_covers_what_the_index_stores_about_a_surface(
    corpus, field: str, value: object
) -> None:
    """G-5: every column the index STORES about a surface moves the fingerprint.

    `surface_fingerprint` is `(version, type, origin, text)` by design — two surfaces with
    the same text under different provenance must differ, and nothing more. But the index
    persists more than that in `surfaces`: attribution, title, url, locator and language,
    which `search` serves on every match (A-1) and `--has-surface` filters on. A change to
    any of them with the text untouched left `update` seeing nothing to do — the evidence
    repaired and the derivative standing (rule 6), on the attribution rule this repo paid
    for in blood.

    Four axes, one parametrised test, each changing ONE field of k07's quoted post and
    nothing else. The url moves the locator too (`locator.url`), which is the point: the
    locator is what the consumer resolves the evidence through. `producer` is deliberately
    NOT here — the index has no producer column, and since round 08 (F7-7) the ASR/VLM
    surfaces carry none at all, because the store records no transcriber; the producers
    that ARE recorded (`enriched.executor`, `description_version`) travel with the surface
    and are not stored columns either.

    Seen red before the fix on all four: the fingerprint did not move. Seen red again here
    under the mutant `surface_row` → `(None, None, None, …)`, which the file passed 90/90
    before this test was restored: `surface_row` is the projection `item_fingerprint`
    hashes, so a column dropped from it is a column the index stores and never re-hashes.
    """
    from xbrain.models import Author

    store, _vocab, _pages = corpus
    item = store["k07"]
    position = next(i for i, s in enumerate(item.content.sources) if s.kind == "quoted_tweet")
    sources = list(item.content.sources)
    patch = {field: Author(**value) if field == "author" else value}
    sources[position] = sources[position].model_copy(update=patch)
    edited = item.model_copy(
        update={"content": item.content.model_copy(update={"sources": sources})}
    )
    assert index_build.item_fingerprint(edited) != index_build.item_fingerprint(item), field


def test_the_store_fingerprint_is_order_independent(corpus) -> None:
    """Two loads of the same store must agree, whatever order the dict happens to iterate in."""
    store, _vocab, _pages = corpus
    reversed_store = dict(reversed(list(store.items())))
    assert index_build.store_fingerprint(reversed_store) == index_build.store_fingerprint(store)


def test_load_index_inputs_binds_the_signal_to_the_bytes_it_read(workspace, corpus) -> None:
    """The mechanism behind P1b, at the loader: the signal is `fstat` of the handle the
    text came from, taken BEFORE the read — so a replacement that lands during the read
    leaves the objects and the signal describing the same snapshot (the handle keeps the
    inode it opened; the store's writers replace atomically), and leaves the path pointing
    at a newer inode the query-time `StoreSignal.of` reports as different.

    Staged INSIDE the read: the handle `load_index_inputs` opens replaces the file on disk
    the moment it is read from, which is the only place a stat of the path and a stat of
    the handle can disagree. Seen red under the mutation `stat = path.stat()` taken after
    the read: the signal then described the replacement, not the bytes parsed.
    """
    from xbrain.store import save_store

    store, _vocab, _pages = corpus
    items = workspace / "items.json"
    before = index_build.StoreSignal.of(items)
    newer = {**store, "k01": store["k01"].model_copy(update={"text": "replaced during the load"})}

    class RacingHandle:
        def __init__(self, handle):
            self._handle = handle

        def fileno(self):
            return self._handle.fileno()

        def read(self):
            save_store(newer, items)  # the race: a save lands while the loader is reading
            return self._handle.read()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

    class RacingPath(type(items)):
        def open(self, *args, **kwargs):
            return RacingHandle(super().open(*args, **kwargs))

    loaded = index_build.load_index_inputs(RacingPath(items))

    assert loaded.store["k01"].text == store["k01"].text, "the parse saw the bytes it read"
    assert loaded.signal == before, "the signal describes the snapshot that was parsed"
    assert index_build.StoreSignal.of(items) != loaded.signal, "the path now points elsewhere"


def test_the_store_signal_is_one_stat_and_nothing_else(workspace) -> None:
    """The cheap signal must stay cheap, or it is the expensive one with a different name."""
    signal = index_build.StoreSignal.of(workspace / "items.json")
    stat = (workspace / "items.json").stat()
    assert signal.items_json_mtime_ns == stat.st_mtime_ns
    assert signal.items_json_size == stat.st_size


def test_the_store_signal_of_a_missing_store_is_zeroed_not_an_exception(tmp_path: Path) -> None:
    """A query must still be able to say "the index is behind" when the store is gone.

    Raising here would turn a missing store into a crash inside `search`, which is the wrong
    place to learn it: the response declares the degradation instead.
    """
    signal = index_build.StoreSignal.of(tmp_path / "nope.json")
    assert signal == index_build.StoreSignal(items_json_mtime_ns=0, items_json_size=0)


def _obstruct(path: Path, obstacle: str) -> None:
    """Make `path` UNREADABLE without removing it — the two shapes an operator meets."""
    if obstacle == "chmod000":
        path.chmod(0)
    else:
        path.unlink()
        path.mkdir()


@pytest.mark.parametrize(
    ("obstacle", "error"), [("chmod000", PermissionError), ("directory", IsADirectoryError)]
)
def test_an_unreadable_store_is_an_error_never_an_empty_store(
    workspace: Path, obstacle: str, error: type[OSError]
) -> None:
    """A-2 (gate Fable §5.2, round 08 — the DIFF subagent's A-1, reproduced by the gate in
    two variants): `_read_bound` caught EVERY `OSError` and answered `(None, 0, 0)`, the
    reading reserved for a file that is ABSENT — so an `items.json` with `chmod 000`, or a
    directory standing where the file was, loaded as an EMPTY store. On the real index:
    `status` healthy with `items_removed 2404`, `search` «0 resultados» exit 0 with nothing
    declared, `update` planning `-2404 items`, and `build --force` replacing 22,286 chunks
    with the 703 of the topic plane, sealed consistent, exit 0. The loader is the route
    BEFORE seam (a), the one no door asks, and through it the whole fail-open family came
    back with a destruction on top.

    `load_store` (`store.py`) reads only an ABSENT file as `{}`; every other `OSError`
    propagates. The loader now agrees: only `FileNotFoundError` is the empty store, and
    `EACCES`/`EISDIR`/`EIO` raise, which the CLI prints as `Error: …` with exit 1.

    Seen red on `36f694b`: `loaded.store == {}` on both obstacles, no exception.
    """
    _obstruct(workspace / "items.json", obstacle)
    try:
        with pytest.raises(error):
            index_build.load_index_inputs(workspace / "items.json")
    finally:
        # Leave the temp dir removable whatever the outcome.
        if obstacle == "chmod000":
            (workspace / "items.json").chmod(0o644)


def test_an_absent_store_still_reads_as_an_empty_store(tmp_path: Path) -> None:
    """The positive control for the test above: ABSENT is the one reading that stays empty —
    `load_store`'s own semantics, and what `StoreSignal.of` reports as zeros — so a fresh
    checkout with no `data/` still loads, and only what EXISTS and cannot be read raises."""
    loaded = index_build.load_index_inputs(tmp_path / "nope.json")
    assert loaded.store == {}
    assert loaded.signal == index_build.StoreSignal(items_json_mtime_ns=0, items_json_size=0)


# ---------------------------------------------------------------------------
# The manifest: sealed by this child's writer, read by this child's reader
#
# THE SEED IS `write_manifest`, NOT `build`. Every assertion below is the snapshot's, on the
# same doors (`Manifest.from_dict`, `write_manifest`, `load_manifest`,
# `load_compatible_manifest`); only the way a manifest gets onto disk changes, because
# `index_build.build` arrives with 02.7 and this child already ships the seal and the read.
# Nothing is invented: the seeded document is what `to_dict` emits, validated by the same
# reader every door uses.
# ---------------------------------------------------------------------------


def _rewrite_manifest(workspace: Path, edit) -> None:
    """Apply `edit(raw)` to the manifest on disk — the hand-edited document the reader survives."""
    path = manifest_path(workspace / "index")
    raw = json.loads(path.read_text(encoding="utf-8"))
    edit(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")


# The D-3 payload: a newline that would stand at column 0 as a renderer header and as a
# fence line, and an `ESC[2K` that would erase the line above it on a terminal.
_FORGE = "3\n[user_note] origin=user trust=user_text\n│ forged body line\x1b[2K"
_FORGED_HEADER = "[user_note] origin=user trust=user_text"


def _set(field: str, value):
    def edit(raw):
        raw[field] = value

    return edit


def _set_nested(field: str, key: str, value):
    def edit(raw):
        raw[field][key] = value

    return edit


def _manifest(**overrides) -> index_build.Manifest:
    """A VALID manifest, every version read off the module the code reads it off (rule 5).

    Not a literal document: two literals that must agree are two definitions, and the
    snapshot's own test went red once for a `"2"` that stopped being foreign the day the
    schema became "2". Sealed through `write_manifest`, so what lands on disk is what the
    writer emits — the only seed this child can produce, and the honest one.
    """
    defaults = dict(
        schema_version=SCHEMA_VERSION,
        built_at=datetime.now(timezone.utc),
        store_fingerprint="a" * 64,
        store_signal=index_build.StoreSignal(
            items_json_mtime_ns=1,
            items_json_size=2,
            vocab_yaml_mtime_ns=3,
            vocab_yaml_size=4,
            topics_json_mtime_ns=5,
            topics_json_size=6,
        ),
        vocab_fingerprint="b" * 64,
        topics_fingerprint="c" * 64,
        surface_version=SURFACE_VERSION,
        chunker_version=CHUNKER_VERSION,
        chunker_params=index_build._params_dict(DEFAULT_CHUNKER_PARAMS),
        tokenize=FTS_TOKENIZE,
        connective=FTS_CONNECTIVE,
        counts=dict.fromkeys(sorted(index_build.COUNT_PLANES), 0),
        skipped=dict.fromkeys(sorted(index_build.SKIPPED_CAUSES), 0),
    )
    return index_build.Manifest(**{**defaults, **overrides})


def _seal(workspace: Path, **overrides) -> index_build.Manifest:
    """Seal a valid manifest into `workspace/index` and hand it back."""
    manifest = _manifest(**overrides)
    index_build.write_manifest(workspace / "index", manifest)
    return manifest


def _raw(workspace: Path) -> dict:
    return json.loads(manifest_path(workspace / "index").read_text(encoding="utf-8"))


# --- what the writer seals ------------------------------------------------


def test_write_manifest_seals_exactly_the_fields_the_contract_declares(workspace) -> None:
    """Step 3, at the WRITER this child ships: asserted as a SET against `MANIFEST_FIELDS`,
    not one `in` per key.

    A key-by-key list is a second copy of the schema that drifts; the set assertion makes
    "the writer emits exactly what the contract declares" a single fact, in BOTH directions
    — a field dropped from `to_dict` and a field dropped from the declared set are the same
    red. Seen red under the mutant `manifest_fields_set` (`"counts"` removed from
    `MANIFEST_FIELDS`), which the suite passed 11/11 before this test existed.

    The half that belongs to `build` — that `counts["items"]` equals the corpus it indexed —
    arrives with 02.7, which owns the producer of those numbers. This pins the seam.
    """
    _seal(workspace)
    raw = _raw(workspace)
    assert set(raw) == index_build.MANIFEST_FIELDS
    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["surface_version"] == SURFACE_VERSION and raw["chunker_version"] == CHUNKER_VERSION
    # The MEASURED parameters (Plan 02 §7's sweep), read from the module rather than written
    # here a second time: two literals that must agree are two definitions (rule 5).
    assert raw["chunker_params"] == {
        "target": DEFAULT_CHUNKER_PARAMS.target,
        "max_chars": DEFAULT_CHUNKER_PARAMS.max_chars,
        "overlap": DEFAULT_CHUNKER_PARAMS.overlap,
        "min_chars": DEFAULT_CHUNKER_PARAMS.min_chars,
    }
    assert set(raw["counts"]) == index_build.COUNT_PLANES
    assert set(raw["skipped"]) == index_build.SKIPPED_CAUSES
    # The hole Plan 03 fills. Declared now so its arrival is not a manifest migration.
    assert raw["embeddings"] is None


def test_the_manifest_records_the_tokenizer_and_the_connective(workspace) -> None:
    """Beyond spec §5.6, and deliberately so: these two DECIDE every recall number.

    The connective change of Plan 01 M3 moved mean `recall@10` from 0.1429 to 0.8099 without
    touching one line of the chunker. A manifest that records the chunker version but not the
    query semantics would let two incomparable baselines look like the same measurement.
    """
    _seal(workspace)
    raw = _raw(workspace)
    assert raw["tokenize"] == "unicode61 remove_diacritics 2"
    assert raw["connective"] == "OR"


def test_a_sealed_manifest_reads_back_as_the_manifest_that_was_sealed(workspace) -> None:
    """The round trip, end to end: `to_dict` → JSON → `from_dict` loses nothing.

    Equality is on the DATACLASS, so a field the writer emits and the reader drops (or reads
    into the wrong slot) is red — `built_at` through an ISO string and `store_signal` through
    its own nested document are the two that could silently degrade. Seen red by writing
    `built_at` with `.date().isoformat()`: the instant came back at midnight.
    """
    sealed = _seal(workspace)
    assert index_build.load_compatible_manifest(workspace / "index") == sealed
    assert index_build.load_manifest(workspace / "index") == sealed


def test_the_embeddings_slot_round_trips_the_object_plan_03_will_put_in_it(workspace) -> None:
    """`embeddings` is `null` or an OBJECT, and the object half has to survive the trip.

    The slot is declared now so Plan 03's `{model, dimension, normalized, command_version}`
    is not a manifest migration — which is only true if a manifest carrying one reads back
    carrying it. `_optional_mapping`'s success path is the one line of the reader that a
    manifest sealed today never exercises, and «declared for later» is exactly the kind of
    promise that is discovered to be false later.
    """
    declared = {"model": "all-MiniLM-L6-v2", "dimension": 384, "normalized": True}
    _seal(workspace, embeddings=declared)
    assert _raw(workspace)["embeddings"] == declared
    assert index_build.load_compatible_manifest(workspace / "index").embeddings == declared


def test_the_manifest_built_at_is_an_instant_not_a_date(workspace) -> None:
    """A rebuild the same day must be distinguishable from the previous one."""
    _seal(workspace)
    built = datetime.fromisoformat(_raw(workspace)["built_at"])
    assert built.tzinfo is not None, "an instant with no timezone is a local guess"
    assert abs((datetime.now(timezone.utc) - built).total_seconds()) < 120


# --- totality: the document is an object, and it declares every field --------


@pytest.mark.parametrize(
    "label, document",
    [
        # THE ONE THAT PASSED THE GUARD. A JSON list whose elements are exactly the field
        # NAMES: `MANIFEST_FIELDS - set(raw)` is empty, because `set()` of a list is its
        # ELEMENTS — so the totality check agreed the document declared everything, and the
        # first subscript raised `TypeError: list indices must be integers` out of every
        # door as a traceback.
        ("a list of the field names", "MANIFEST_FIELDS_AS_LIST"),
        ("an empty list", []),
        # These never reached a subscript: `set()` of them raises `TypeError: not iterable`
        # from the guard LINE ITSELF, before one field was read.
        ("a number", 3),
        ("null", None),
        ("a boolean", True),
        # A string is iterable, so it reached the guard and was refused for the wrong
        # reason (its CHARACTERS are not the field names). Now refused for the right one.
        ("a string", "schema_version"),
    ],
)
def test_a_manifest_that_is_not_a_json_object_is_refused_naming_the_rebuild(
    workspace, label: str, document
) -> None:
    """Spec §9.3: an incompatible manifest is refused with an ACTIONABLE sentence — never a
    traceback from inside a query, which is the one thing this reader exists to remove.

    `manifest.json` is hand-editable by design (B1 measured what an operator can hold), and
    the totality guard was `MANIFEST_FIELDS - set(raw)` with no type check in front of it.
    `set()` of a non-mapping is not an error, so the guard answered about the wrong thing.

    Seen red on `99731e9` for all six: `TypeError: list indices must be integers or slices,
    not str` (the list) and `TypeError: 'int'/'NoneType'/'bool' object is not iterable`, out
    of `load_manifest` AND `load_compatible_manifest`; the string was refused, but by the
    field-name check, so a document that happened to be the field name of a one-field schema
    would have walked through.
    """
    if document == "MANIFEST_FIELDS_AS_LIST":
        document = sorted(index_build.MANIFEST_FIELDS)
    (workspace / "index").mkdir(parents=True, exist_ok=True)
    manifest_path(workspace / "index").write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(IndexIncompatibleError, match="index build --force"):
        index_build.load_manifest(workspace / "index")
    with pytest.raises(IndexIncompatibleError, match="index build --force"):
        index_build.load_compatible_manifest(workspace / "index")


@pytest.mark.parametrize("field", sorted(index_build.MANIFEST_FIELDS))
def test_a_manifest_missing_a_top_level_field_is_refused_naming_the_rebuild(
    workspace, field: str
) -> None:
    """The totality guard itself, one parametrisation per declared field.

    `MANIFEST_FIELDS` is the enumeration of spec §5.6 in one place, and the guard that
    enforces it had NO test: mutating `missing = MANIFEST_FIELDS - set(raw)` to
    `frozenset()` — deleting the required-field check entirely — left the WHOLE suite green
    at 2,243 passed, on `99731e9` and on the snapshot it was ported from. Rule 1: a guard
    nothing can make fail is not guarded.

    Derived from the constant, so a field added to the contract is required of the reader
    without anybody remembering — and a field removed from the constant cannot go unnoticed,
    because `test_write_manifest_seals_exactly_the_fields_the_contract_declares` compares
    the same set against what the writer emits.
    """
    _seal(workspace)
    _rewrite_manifest(workspace, lambda raw: raw.pop(field))

    with pytest.raises(IndexIncompatibleError, match="index build --force") as caught:
        index_build.load_compatible_manifest(workspace / "index")
    assert field in str(caught.value), "names the field the document does not declare"


_NESTED_KEYS = [
    *(("counts", plane) for plane in sorted(index_build.COUNT_PLANES)),
    *(("skipped", cause) for cause in sorted(index_build.SKIPPED_CAUSES)),
    *(("chunker_params", name) for name in sorted(index_build.CHUNKER_PARAM_NAMES)),
    ("store_signal", "items_json_mtime_ns"),
    ("store_signal", "items_json_size"),
]


@pytest.mark.parametrize("field, key", _NESTED_KEYS)
def test_a_manifest_missing_a_nested_key_is_refused_naming_the_rebuild(
    workspace, field: str, key: str
) -> None:
    """B1: `Manifest.from_dict` checked the TOP-LEVEL key set and cast everything under it,
    so a manifest whose `counts` was `{}` loaded as compatible — and the consistency check,
    which iterated whatever `counts` offered, then compared NOTHING: `status` said healthy,
    `search` answered zero results over a base with its chunks and profiles deleted, and
    `update` sealed that state as sound. Spec §9.3: an incompatible manifest is never
    queried partially.

    Every nested key is REQUIRED, one parametrisation per key, derived from the same
    constants the writer uses — so a plane, a cause or a chunker parameter added to the
    writer is required of the reader without anybody remembering. Seen red under the mutant
    `nested_required` on all 15 parametrisations: `load_compatible_manifest` returned a
    `Manifest`.
    """
    _seal(workspace)
    _rewrite_manifest(workspace, lambda raw: raw[field].pop(key))

    with pytest.raises(IndexIncompatibleError, match="index build --force") as caught:
        index_build.load_compatible_manifest(workspace / "index")
    assert field in str(caught.value), "names the malformed field"


@pytest.mark.parametrize(
    "label, edit",
    [
        ("counts empty", _set("counts", {})),
        ("counts not a mapping", _set("counts", "12")),
        ("counts a list", _set("counts", [])),
        ("a count that is a string", _set_nested("counts", "chunks", "56")),
        ("a negative count", _set_nested("counts", "chunks", -1)),
        ("a boolean count", _set_nested("counts", "chunks", True)),
        ("a float count", _set_nested("counts", "chunks", 1.0)),
        ("an undeclared plane", _set_nested("counts", "vectors", 0)),
        ("skipped not a mapping", _set("skipped", [])),
        ("an undeclared cause", _set_nested("skipped", "vanished", 0)),
        ("a chunker parameter that is a string", _set_nested("chunker_params", "target", "800")),
        ("an undeclared chunker parameter", _set_nested("chunker_params", "stride", 4)),
        ("store_signal not a mapping", _set("store_signal", 5)),
        ("a signal entry that is a string", _set_nested("store_signal", "items_json_size", "17")),
        ("an undeclared signal entry", _set_nested("store_signal", "media_dir_size", 0)),
        ("embeddings neither null nor a mapping", _set("embeddings", ["all-MiniLM"])),
        ("failed not a list", _set("failed", {"k01": "boom"})),
        # The two shapes the `isinstance(value, list)` guard alone is load-bearing for: an
        # EMPTY object makes `all(… for entry in value)` vacuously true (accepted, fail-open)
        # and a number makes the same generator raise `TypeError: not iterable` (a traceback
        # from inside a query). Every other malformed `failed` is refused by the per-entry
        # clause, so without these two the guard is unguarded — mutant `failed_shape`.
        ("failed an empty object", _set("failed", {})),
        ("failed a number", _set("failed", 5)),
        ("a failure that is not a mapping of strings", _set("failed", [{"item_id": 1}])),
        ("a failure that is not a mapping", _set("failed", ["boom"])),
        ("built_at not an instant", _set("built_at", "yesterday")),
        ("built_at null", _set("built_at", None)),
    ],
)
def test_a_malformed_nested_value_is_refused_naming_the_rebuild(
    workspace, label: str, edit
) -> None:
    """B1, the values: a key that is present with the wrong shape is as vacuous as one that
    is missing. `int("56")` silently accepted a string count; `True` is an `int` to
    `isinstance`; an undeclared plane would be compared against a `COUNT(*)` that does not
    exist. Closed as well as total: exactly the declared keys, each a non-negative integer.

    Seen red under `nested_closed`, `nested_int`, `optional_mapping`, `failed_shape` and
    `instant` — five mutants the suite passed 11/11 before this test existed.
    """
    _seal(workspace)
    _rewrite_manifest(workspace, edit)

    with pytest.raises(IndexIncompatibleError, match="index build --force"):
        index_build.load_compatible_manifest(workspace / "index")


# --- the writer round-trips through the reader ------------------------------


def test_write_manifest_refuses_a_document_the_reader_would_refuse(workspace) -> None:
    """The WRITER side of the seam: `write_manifest` round-trips the document through the
    same reader every door uses, so a build or an update cannot seal a manifest that the
    next `search` would then refuse — or, worse, one the reader would have accepted while
    the writer's shape drifted.

    `counts={}` is buildable in Python (`counts: dict[str, int]` constrains the value type,
    not the key set), and it is exactly the amputation B1 measured. Seen red under the
    mutant `roundtrip_guard`: the file was overwritten with a document no door would read.
    """
    _seal(workspace)
    before = manifest_path(workspace / "index").read_bytes()

    with pytest.raises(IndexIncompatibleError, match="counts"):
        index_build.write_manifest(workspace / "index", _manifest(counts={}))

    assert manifest_path(workspace / "index").read_bytes() == before, "and it wrote nothing"


def test_write_manifest_leaves_no_manifest_behind_when_the_document_is_refused(
    tmp_path: Path,
) -> None:
    """The same guard where there is nothing to preserve: a FIRST seal that is refused must
    leave no file at all, or its presence — which is what says a build completed — would be
    a lie about a document no door can read."""
    index_dir = tmp_path / "index"
    with pytest.raises(IndexIncompatibleError):
        index_build.write_manifest(index_dir, _manifest(skipped={}))
    assert not manifest_path(index_dir).exists()


# --- compatibility: refused unless every version the code depends on matches --


@pytest.mark.parametrize(
    "field, value",
    [
        # DERIVED from the constant, not a literal: the literal `"2"` stopped being foreign
        # the day the schema was bumped to "2", and the test went red for a reason that had
        # nothing to do with what it pins.
        ("schema_version", f"{SCHEMA_VERSION}-foreign"),
        ("surface_version", "xbrain-knowledge-surface/v9"),
        ("chunker_version", "xbrain-knowledge-chunker/v9"),
        # The fifth and sixth checks (F7-8, round 07). The manifest recorded them «because
        # they decide every recall number» and no door compared them: `tokenize: porter` /
        # `connective: AND` were accepted on the real index. The tokenizer is baked into the
        # FTS DDL, so a base built under one and queried under another is the M-1 shape with
        # no guard; the connective moved recall@10 from 0.1429 to 0.8099.
        ("tokenize", "porter unicode61"),
        ("connective", "AND"),
    ],
)
def test_an_incompatible_manifest_refuses_the_query_entirely(
    workspace, field: str, value: str
) -> None:
    """Step 29 / spec §9.3: *manifest incompatible: no se consulta parcialmente.*

    Each of these invalidates a different thing, and any one of them makes the stored rows
    mean something other than what the code would compute. Seen red by logging a warning and
    answering anyway — and, here, under the mutants `compat_schema`, `compat_tokenize` and
    `compat_connective`, which the suite passed 11/11 before this test existed.
    """
    _seal(workspace)
    _rewrite_manifest(workspace, _set(field, value))
    with pytest.raises(IndexIncompatibleError, match="index build --force") as caught:
        index_build.load_compatible_manifest(workspace / "index")
    assert field in str(caught.value), "names which version disagrees"


def test_chunker_parameters_are_compared_when_the_caller_declares_them(workspace) -> None:
    """The FOURTH check, and the one no version string carries: Plan 02 §7 sweeps
    `target x overlap`, and a sweep that lands on new parameters without bumping
    `CHUNKER_VERSION` produces chunks cut differently under IDENTICAL ids — the worst case,
    because the id resolves and the text behind it is not what it was.

    Compared only when the caller declares its parameters, which is what `params=None`
    means; both directions are pinned, or the mutant `compat_params` (`if False and …`)
    survives on the half nobody exercised.
    """
    _seal(workspace)
    swept = replace(DEFAULT_CHUNKER_PARAMS, target=DEFAULT_CHUNKER_PARAMS.target + 1)

    with pytest.raises(IndexIncompatibleError, match="chunker_params"):
        index_build.load_compatible_manifest(workspace / "index", params=swept)
    assert index_build.load_compatible_manifest(workspace / "index", params=DEFAULT_CHUNKER_PARAMS)
    assert index_build.load_compatible_manifest(workspace / "index"), "undeclared is not compared"


@pytest.mark.parametrize(
    "field", ["schema_version", "surface_version", "chunker_version", "tokenize", "connective"]
)
def test_a_forged_manifest_string_reaches_no_terminal_raw_through_this_child_s_door(
    workspace, field: str
) -> None:
    """T-1: the manifest is a file an operator edits by hand, and the incompatibility
    sentence interpolated `schema_version`, `surface_version` and `chunker_version` RAW — so
    a forged line stood at column 0 and an `ESC[2K` reached a TTY through the sentence every
    door raises. The fix is `!r` on every manifest string, and it lives HERE, in
    `load_compatible_manifest`.

    This child has one door, so this pins the sentence at its SOURCE. The three-door version
    — `status`'s rendered report, `search` through `open_for_query`, `update` — arrives with
    02.8/02.9, which own those doors; it asserts the same bytes never reach a terminal, one
    level further out.
    """
    _seal(workspace)
    _rewrite_manifest(workspace, _set(field, _FORGE))

    with pytest.raises(IndexIncompatibleError) as caught:
        index_build.load_compatible_manifest(workspace / "index")

    sentence = str(caught.value)
    assert "\x1b" not in sentence
    assert _FORGED_HEADER not in sentence.splitlines(), sentence
    assert not any(line.startswith("│ forged") for line in sentence.splitlines()), sentence
    assert "xbrain index build --force" in sentence


def test_a_manifest_from_before_round_05_reads_its_absent_signal_entries_as_zeros(
    workspace,
) -> None:
    """The compatibility promise P1a is documented in three places and no test exercised it:
    a manifest sealed before round 05 carries no `vocab.yaml`/`topics.json` entries, and
    they must read back as ZEROS — which compare unequal to the live files, so the index is
    declared behind until one `index update` re-seals it. The direction this signal is meant
    to fail in, and it costs one `update`.

    Required and optional are read off `StoreSignal` itself (`_signal_keys`), so this pins
    that split: make the four round-05 entries required and a pre-round-05 manifest becomes
    unreadable instead of behind; make the two `items.json` entries optional and an
    amputated signal reads as a store that never moved.

    The other half — that `search` then declares `index_behind_store` and `status` reports
    `behind` — arrives with 02.8/02.9, which own those doors.
    """
    _seal(workspace)

    def pre_round_05(raw):
        for key in (
            "vocab_yaml_mtime_ns",
            "vocab_yaml_size",
            "topics_json_mtime_ns",
            "topics_json_size",
        ):
            del raw["store_signal"][key]

    _rewrite_manifest(workspace, pre_round_05)
    assert set(_raw(workspace)["store_signal"]) == {"items_json_mtime_ns", "items_json_size"}

    signal = index_build.load_compatible_manifest(workspace / "index").store_signal
    assert signal == index_build.StoreSignal(items_json_mtime_ns=1, items_json_size=2)
    assert signal != index_build.StoreSignal.of(
        workspace / "items.json", workspace / "vocab.yaml", workspace / "topics.json"
    ), "so the index is declared behind, which is what the zeros are for"


def test_a_missing_manifest_names_the_build_command_not_the_rebuild(tmp_path: Path) -> None:
    """An index that was never built is not an incompatible one, and the two errors name
    different commands: `IndexMissingError` says `xbrain index build`, and only a document
    that EXISTS and disagrees says `--force`. Conflating them sends an operator to a
    destructive command to fix an empty directory."""
    with pytest.raises(IndexMissingError, match="xbrain index build") as caught:
        index_build.load_compatible_manifest(tmp_path / "index")
    assert "--force" not in str(caught.value)
