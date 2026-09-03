# tests/test_knowledge_index_build.py
"""The three inputs of the index, the signal bound to them, and their fingerprints
(Plan 02 §2, §3, steps 3, 4, 9).

TWO SIGNALS, TWO COSTS, TWO PLACES (B3) — three `os.stat` over the three inputs against one
sha256 per item. The argument for the split, and the direction each fails in, is stated once,
in `index_build.py`'s module docstring; it is not restated here.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.knowledge import index_build
from xbrain.knowledge.surfaces import item_surfaces, item_topics
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
def three_inputs(tmp_path: Path, corpus) -> Path:
    """A data/ directory holding ALL THREE inputs, written by the store's own writers."""
    from xbrain.rubrics import save_vocab
    from xbrain.store import save_store, save_topic_pages

    store, vocab, pages = corpus
    data = tmp_path / "data"
    save_store(store, data / "items.json")
    save_vocab(vocab, data / "vocab.yaml")
    save_topic_pages(pages, data / "topics.json")
    return data


def test_the_canonical_encoding_is_injective_where_ensure_ascii_would_not_be() -> None:
    """`ensure_ascii=True` collides a lone surrogate PAIR with the astral character it spells
    — `json.dumps` emits `"\\ud83d\\ude00"` for both — so the setting is a correctness choice
    and until now only prose held it: measured, the whole file stays GREEN under the mutant
    `ensure_ascii=False` -> `True`. Asserted on the ENCODER because the surrogate side raises
    at `_sha256`'s `.encode("utf-8")`, which is fail-closed and not a hash to compare.
    """
    assert index_build._canonical("t", ["\ud83d\ude00"]) != index_build._canonical(
        "t", ["\U0001f600"]
    )


def _rows(item: Item) -> list[index_build.SurfaceRow]:
    """The surface rows of one item — the exact call every guard below makes, named once."""
    return [index_build.surface_row(surface) for surface in item_surfaces(item)]


# ---------------------------------------------------------------------------
# The fingerprint, and the population it has to reach (rule 6)
# ---------------------------------------------------------------------------


def test_the_item_fingerprint_changes_when_indexable_text_changes(corpus) -> None:
    """It hashes the SURFACES, not `(fetched_at, enriched_at)` as Plan 01 §10 sketched.

    CLAUDE.md rule 6 twice over — a timestamp cannot reach the 960 of 2,404 items with no
    `content` at all, and it is a PROXY that a hand edit leaves unmoved. The measurement and
    the argument live in `index_build.py`'s module docstring; what this test adds is the
    executable half. The `content is None` precondition is the POPULATION half of rule 6 and
    not decoration: on this item a fingerprint over `content.fetched_at` would be constant
    forever, so it is what makes the edit below observable at all. (A second test asserting
    only that was a strict subset of this one — same item, same patch, same assertion — and
    no mutation could redden one without the other, so its precondition moved here.)
    """
    store, _vocab, _pages = corpus
    item = store["k02"]
    assert item.content is None, "a fingerprint over `content.fetched_at` could not reach it"
    before = index_build.item_fingerprint(item)
    edited = item.model_copy(
        update={"enriched": item.enriched.model_copy(update={"summary": "otro resumen"})}
    )
    assert index_build.item_fingerprint(edited) != before
    assert index_build.item_fingerprint(item) == before, "and it is stable"


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

    `surface_fingerprint` is `(version, type, origin, text)` by design, and the index persists
    more than that; why the ROW and not the fingerprint alone is `item_fingerprint`'s
    docstring, not this one. Four axes, one parametrised test, each changing ONE field of
    k07's quoted post and nothing else. The `url` case moves the locator columns but does NOT
    pin them: it is satisfied through `surface_id`, which hashes the source's kind and url,
    and stays green with both locator columns dropped from `surface_row` (round 09, B1). What
    pins the locator is the test below, which moves a field `surface_id` cannot reach.

    Seen red before the fix on all four: the fingerprint did not move. Seen red again under
    the mutant `surface_row` → `(None, None, None, …)`, which the file passed 90/90 before
    this test was restored — on three of the four, the `url` case being the exception named
    above: `surface_row` is the projection `item_fingerprint` hashes, so a column dropped
    from it is a column the index stores and never re-hashes.
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


def test_the_item_fingerprint_covers_the_locator_not_only_the_url_the_id_hashes(
    corpus,
) -> None:
    """The LOCATOR column, which the `[url]` case above does not reach (round 09, B1).

    That case moves the fingerprint through `surface_id`, so it is satisfied with BOTH locator
    columns dropped from `surface_row` — rule 1's first row, satisfied for the wrong reason.

    `Locator` carries eleven fields and ten reach the persisted `locator_json` column
    (`index_schema`) through no other hashed path; they are how a reader resolves a claim back
    to the bytes it came from, the mechanism CLAUDE.md rule 7 rests on. So this moves exactly
    one: k08's first video frame keeps its description, `frame_index`, `source_key` and
    source, and only its `frame_timestamp` changes. The two guards make it bite for the reason
    it names — exactly one row moves, in one column, deserialising to the new timestamp.

    The sibling column `surface.locator.url` is a DUPLICATE of a field `model_dump_json()`
    also serialises, so dropping it ALONE is unobservable in the hash by construction: no
    test can pin it, and one claiming to would be pinning the json.

    Seen red under the mutant that drops the two locator columns from `surface_row`.
    """
    store, _vocab, _pages = corpus
    item = store["k08"]
    position = next(i for i, s in enumerate(item.content.sources) if s.kind == "x_video")
    video = item.content.sources[position]
    assert video.frames, "this test needs a video the index locates frames inside"

    frames = list(video.frames)
    moved_to = frames[0].timestamp + 37.0
    frames[0] = frames[0].model_copy(update={"timestamp": moved_to})
    sources = list(item.content.sources)
    sources[position] = video.model_copy(update={"frames": frames})
    edited = item.model_copy(
        update={"content": item.content.model_copy(update={"sources": sources})}
    )

    before = _rows(item)
    after = _rows(edited)
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert len(changed) == 1, "exactly one surface row moves: the one whose timestamp changed"
    row_before, row_after = before[changed[0]], after[changed[0]]
    columns = [i for i, (a, b) in enumerate(zip(row_before, row_after, strict=True)) if a != b]
    assert len(columns) == 1, "and exactly one column of that row"
    assert json.loads(row_after[columns[0]])["frame_timestamp"] == moved_to, (
        "the column that moved is the serialised locator, carrying the new timestamp"
    )
    assert json.loads(row_before[columns[0]])["frame_timestamp"] == video.frames[0].timestamp

    assert index_build.item_fingerprint(edited) != index_build.item_fingerprint(item)


# The positions of three `surfaces` columns inside `SurfaceRow`, written out BY HAND — exactly
# what `surface_row`'s docstring says the correspondence with the persisted DDL still is in this
# child. 02.7 owes the writer that binds the tuple to the `INSERT` and the readback that makes
# it structural; naming the positions only lets an assertion say WHICH column it pins, and adds
# no guard `surface_row` does not have.
URL_COLUMN = 10
LOCATOR_JSON_COLUMN = 11
FINGERPRINT_COLUMN = 13
CHAR_LENGTH_COLUMN = 14


def test_the_row_carries_the_url_column_the_schema_persists_beside_the_locator_json(
    corpus,
) -> None:
    """`surfaces.url` is a persisted column of its own, and no test in this file pinned it.

    Two independent reviews read the same green mutant OPPOSITELY, and the disagreement is why
    this test exists: dropping `surface.locator.url` from `surface_row` leaves the whole file
    green, because the same url is serialised a second time inside `locator_json` and the hash
    cannot see the first copy disappear. The column is unobservable in the HASH and perfectly
    observable in the PROJECTION, so it is pinned here, on the row, and never through
    `item_fingerprint` — a test claiming the hash sees it would be pinning the json (rule 1,
    backwards). What is asserted is that redundancy turned into an invariant: the two persisted
    columns must carry the SAME url, on every surface of the corpus, because `search` serves
    `url` on every match (A-1) without deserialising the locator.

    The two preconditions stop it passing for the wrong reason: a corpus whose locators all
    carried `None` would pass with the column deleted, and one where none did would never
    exercise the `NULL` the DDL allows. The fixture holds both — a `post` locates by url, a
    `summary` by nothing — asserted before anything else is.

    Seen red under the mutant `surface.locator.url` -> `None`, on every surface that has one.
    """
    store, _vocab, _pages = corpus
    rows = [
        index_build.surface_row(surface)
        for item in store.values()
        for surface in item_surfaces(item)
    ]
    located = [row for row in rows if json.loads(row[LOCATOR_JSON_COLUMN])["url"] is not None]
    unlocated = [row for row in rows if json.loads(row[LOCATOR_JSON_COLUMN])["url"] is None]
    assert located, "some surface must locate by url, or the column is asserted against nothing"
    assert unlocated, "and some by none, or the NULL the schema allows is never exercised"

    for row in rows:
        assert row[URL_COLUMN] == json.loads(row[LOCATOR_JSON_COLUMN])["url"], row[0]


def test_the_item_fingerprint_covers_the_surface_fingerprint_column_not_its_length(
    corpus,
) -> None:
    """The column that carries the BODY, isolated from the column that carries its length.

    `surface_row` has no text column — the last one is `len(surface.text)` (spec §10.8) — so
    `surface.fingerprint` is the only thing standing between *the body changed* and an index
    reporting nothing to do. Nothing pinned it: `surface.fingerprint` -> `None` left this
    whole file green, because the flagship text test replaces a 48-character summary with a
    12-character one and so passes through the LENGTH column when the fingerprint column is
    gone — rule 1's first row. The edit here KEEPS THE LENGTH: 48 characters become 48
    different ones. Measured on the shipped code that it moves exactly one row and one column
    of it; with the mutant it moves none, and a hand-corrected summary would leave `update`
    reporting the item current while the index serves the old body (rule 6).

    Seen red under the mutant `surface.fingerprint` -> `None`: no column moves at all, so the
    "exactly one column" assertion fails before the fingerprint assertion is reached.
    """
    store, _vocab, _pages = corpus
    item = store["k02"]
    summary = item.enriched.summary
    rewritten = summary.replace("recall", "sesgos")
    assert len(rewritten) == len(summary) and rewritten != summary, (
        "a body edit that leaves the length untouched is the whole construction"
    )

    edited = item.model_copy(
        update={"enriched": item.enriched.model_copy(update={"summary": rewritten})}
    )
    before = _rows(item)
    after = _rows(edited)
    moved = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert len(moved) == 1, "exactly one surface row moves: the summary whose body was rewritten"
    row_before, row_after = before[moved[0]], after[moved[0]]
    columns = [i for i, (a, b) in enumerate(zip(row_before, row_after, strict=True)) if a != b]
    assert columns == [FINGERPRINT_COLUMN], "and exactly one column of it: the fingerprint"
    assert row_after[CHAR_LENGTH_COLUMN] == row_before[CHAR_LENGTH_COLUMN], (
        "the length column did not move, so it cannot be what carries this edit"
    )

    assert index_build.item_fingerprint(edited) != index_build.item_fingerprint(item)


@pytest.mark.parametrize(
    "axis, value",
    [
        ("author", {"handle": "someoneelse", "name": "Someone Else"}),
        ("created_at", datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)),
        ("captured_at", datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)),
        ("bookmark_folder", "to-read"),
    ],
)
def test_the_item_fingerprint_covers_the_item_plane_no_surface_carries(
    corpus, axis: str, value: object
) -> None:
    """The other half of the hash: the `items` columns, pinned where no surface pins them.

    `items` persists `author_handle`, `author_name`, `created_at`, `captured_at` and
    `bookmark_folder` (`index_schema`), and `items_author` is the index `--author` answers
    from. The test above pins `source` and nothing else: measured under the mutant that
    deletes the rest from `item_fingerprint`, this file stayed green, so `item.author`
    could change and `update` report nothing to do — the evidence repaired and the
    derivative standing (rule 6), one plane up from G-5 and on the same attribution axis.

    THE BASE ITEM CARRIES NO POST SURFACE, AND THAT IS THE WHOLE CONSTRUCTION (rule 1).
    `item.author` is ALSO the attribution of the `post` surface and of a `user_note`, so on
    an ordinary item this assertion moves through `surface_row` and passes with the metadata
    half deleted — satisfied for the wrong reason, which is what the round-09 mutants found.
    k02's text is blanked so no `post` surface is emitted and its `enriched` carries no
    `user_notes`: one `summary` surface remains, attributed to nobody. So every case asserts
    twice — that not one surface row moved, and that the fingerprint moved anyway; the first
    proves the second could not have passed through the surfaces instead.
    """
    from xbrain.models import Author

    store, _vocab, _pages = corpus
    item = store["k02"].model_copy(update={"text": ""})
    surfaces = item_surfaces(item)
    assert [s.surface_type for s in surfaces] == ["summary"], (
        "the base item must carry a surface, and not one that repeats the item's author"
    )
    rows = [index_build.surface_row(s) for s in surfaces]

    edited = item.model_copy(update={axis: Author(**value) if axis == "author" else value})
    assert _rows(edited) == rows, (
        "no surface row may move, or the fingerprint could differ for another reason"
    )
    assert index_build.item_fingerprint(edited) != index_build.item_fingerprint(item), axis


@pytest.mark.parametrize("half, value", [("handle", "otravoz"), ("name", "Otra Voz")])
def test_the_item_fingerprint_covers_the_author_handle_and_the_name_apart(
    corpus, half: str, value: str
) -> None:
    """`items.author_handle` and `items.author_name` are TWO columns, so they need two cases.

    The parametrised test above changes both halves of the author at once, so either half
    could vanish from `item_fingerprint` and its assertion would still be satisfied by the
    other — rule 1's first row again, hiding inside a case that does look isolated. Measured:
    deleting `item.author.name` alone leaves this file green, and so does deleting
    `item.author.handle` alone; only deleting the pair goes red.

    Each case here moves exactly ONE of them and leaves the other byte-identical, so a case
    can only pass through the half it names. The construction is the one the test above
    documents: k02's text blanked, no `user_notes`, one `summary` surface attributed to
    nobody. It is observable, and on the repair this repo has already paid for:
    `refresh-quoted` corrects a display name without touching the handle, and with the name
    unhashed `update` would report `0 cambiados` while `search` keeps serving the old
    attribution (rule 6).

    Seen red under two mutants applied separately: `item.author.name` deleted from
    `item_fingerprint` (the `name` case, the `handle` case still green) and
    `item.author.handle` deleted (the `handle` case, the `name` case still green).
    """
    store, _vocab, _pages = corpus
    item = store["k02"].model_copy(update={"text": ""})
    surfaces = item_surfaces(item)
    assert [s.surface_type for s in surfaces] == ["summary"], (
        "the base item must carry a surface, and not one that repeats the item's author"
    )
    rows = [index_build.surface_row(s) for s in surfaces]

    author = item.author.model_copy(update={half: value})
    assert getattr(author, half) != getattr(item.author, half), "the half under test moved"
    other = "name" if half == "handle" else "handle"
    assert getattr(author, other) == getattr(item.author, other), "and the other did not"

    edited = item.model_copy(update={"author": author})
    assert _rows(edited) == rows, (
        "no surface row may move, or the fingerprint could differ for another reason"
    )
    assert index_build.item_fingerprint(edited) != index_build.item_fingerprint(item), half


@pytest.mark.parametrize(
    "axis, patch, topics_after",
    [
        ("topics", {"topics": ["agent-evaluation"]}, ("agent-evaluation",)),
        ("primary", {"primary_topic": "observability"}, ("observability", "agent-evaluation")),
    ],
)
def test_the_item_fingerprint_covers_the_topics_the_item_plane_persists(
    corpus, axis: str, patch: dict[str, object], topics_after: tuple[str, ...]
) -> None:
    """`topics` is the sixth member of the list `item_fingerprint`'s own docstring enumerates.

    That docstring names the filterable metadata as «source, author, date, topics, content
    kinds». Five of the six have a test; `topics` had none — the topics region could be
    deleted with this whole file green, because every topic in the fixture belongs to an item
    whose surfaces move for some other reason. It is observable, and a repair the pipeline
    performs routinely: re-run `xbrain topics`, reassign an item, and with the region deleted
    `update` reports `0 cambiados` while `--topic` serves the old assignment (rule 6).

    TWO AXES, BECAUSE THE TABLE PERSISTS TWO THINGS. `item_topics` (`index_schema`) stores
    `slug` and `is_primary`, and `item_topics()` encodes the second as ORDER — primary first,
    then the rest, deduplicated. One case changes the membership with the primary fixed,
    the other changes the primary with the membership fixed, and each names the tuple it
    expects: a hash over the set alone would pass the first and fail the second.

    Guarded like the item-plane cases above: not one surface row may move, so the fingerprint
    cannot have moved through the surfaces instead. Seen red under the mutant that deletes
    `*item_topics(item)` from `parts` — both cases, and nothing else in the file.
    """
    store, _vocab, _pages = corpus
    item = store["k02"]
    rows = _rows(item)

    edited = item.model_copy(update={"enriched": item.enriched.model_copy(update=patch)})
    assert item_topics(edited) == topics_after, "the case moved the axis it names"
    assert item_topics(item) != topics_after, "and the base item did not already read that way"
    assert _rows(edited) == rows, (
        "no surface row may move, or the fingerprint could differ for another reason"
    )
    assert index_build.item_fingerprint(edited) != index_build.item_fingerprint(item), axis


def test_the_item_fingerprint_covers_a_content_kind_that_emits_no_surface(corpus) -> None:
    """`item_content_kinds` is a persisted plane, and a kind can arrive with no text at all.

    A no-speech `x_video` (108 of 260, re-derived 2026-09-03 on store `f76341a3…`; the 107 of
    259 this replaces was undated, and the corpus moved) fetched without `--frames` has an
    empty transcript, no frames and no digest, so it emits NOT ONE surface — and it is still a
    video the index records in `item_content_kinds` and a kind filter answers from. Measured
    the mutant that deletes the sorted-kinds region: green, because every kind in the fixture
    arrives attached to a surface that moves the hash on its own.

    The hash is over the sorted MULTISET while the table's primary key is `(item_id, kind)` —
    a set — so a duplicated kind moves the fingerprint without changing the stored plane: the
    accepted false positive. This case adds a kind the item did not have, so it moves both.

    Guarded like the item-plane cases above, and here the guard is also the evidence: the
    surface rows are identical precisely BECAUSE the silent video emitted nothing. The only
    other field the patch touches is `content.fetched_at`, hashed nowhere — deliberately, and
    that is the subject of this module's docstring.
    """
    from xbrain.models import Content, ContentSourceSuccess

    store, _vocab, _pages = corpus
    item = store["k02"]
    assert item.content is None, "the base item must start with no content source at all"
    rows = _rows(item)

    silent = ContentSourceSuccess(
        kind="x_video", url="https://video.example/silent.mp4", text="", has_speech=False
    )
    edited = item.model_copy(
        update={"content": Content(fetched_at=item.captured_at, sources=[silent])}
    )
    assert _rows(edited) == rows, "a silent video with no frames and no digest emits no surface"
    assert index_build.item_fingerprint(edited) != index_build.item_fingerprint(item)


def test_two_different_items_cannot_share_one_serialisation_across_the_variadic_regions(
    corpus,
) -> None:
    """The topics region and the kinds region are ADJACENT, and nothing said where one ends.

    An earlier encoding spliced three variable-length regions — `item_topics`, the sorted
    content kinds and the surface rows — into one flat `\0`-joined list, which is NOT
    injective: two different item states serialise to the very same bytes. Neither side is
    exotic; both are states the store can hold: `primary_topic="thread"` with
    `topics=["thread"]` and no content, against no topics and one
    `ContentSourceSuccess(kind="thread", text="")`. `thread` is both a `ContentKind` and a
    legal topic string (`Enrichment.topics` has no pattern), and a non-video source whose text
    is blank emits NO surface, so the surface region is byte-identical on both sides.

    IT FAILS OPEN, the one direction this module is built not to fail in: the two states hash
    the same, `update` reports the item unchanged, and the persisted `item_topics` /
    `item_content_kinds` planes go on serving a state the store no longer holds (rule 6). The
    reading this replaces argued reachability from `Topic.slug`'s pattern — not the hashed
    field, and "hard to reach" is not "impossible to express".

    WHAT IT PINS IS THE NESTING, not any one marker: each region is its own JSON array inside
    `_canonical`, so the boundary is structural and no value can move it. Seen RED at
    `bfbaf87`, both sides hashing `bafd6ed7…`; red again on this tree under the mutant that
    splices the three regions back into the parent array — 1 failed, 43 passed, diagonal.

    The last assertion is the guard; the three before it prove the two states really do differ
    on the axes they name, or an equal-hash assertion could pass on two identical items.
    """
    from xbrain.models import Content, ContentSourceSuccess

    store, _vocab, _pages = corpus
    item = store["k02"]
    assert item.content is None, "the base item must start with no content source at all"

    topic_side = item.model_copy(
        update={
            "enriched": item.enriched.model_copy(
                update={"primary_topic": "thread", "topics": ["thread"]}
            )
        }
    )
    kind_side = item.model_copy(
        update={
            "enriched": item.enriched.model_copy(update={"primary_topic": None, "topics": []}),
            "content": Content(
                fetched_at=item.captured_at,
                sources=[
                    ContentSourceSuccess(kind="thread", url="https://x.com/i/status/2", text="")
                ],
            ),
        }
    )

    assert item_topics(topic_side) == ("thread",), "one state carries `thread` as a TOPIC"
    assert item_topics(kind_side) == (), "and the other carries no topic at all"
    assert _rows(topic_side) == [index_build.surface_row(s) for s in item_surfaces(kind_side)], (
        "the blank thread source emits no surface, so the surface region cannot break the tie"
    )

    assert index_build.item_fingerprint(topic_side) != index_build.item_fingerprint(kind_side), (
        "a topic named like a content kind must not serialise as that content kind"
    )


def test_a_nul_inside_a_hashed_value_cannot_forge_the_boundary_between_two_atoms(
    corpus, tmp_path: Path
) -> None:
    """The length tags frame the REGIONS. Nothing framed the ATOMS inside them.

    The count tags the previous round added pinned how many topics there ARE, never where
    each one ENDS, and they lived inside the same joined stream — so one U+0000 in any hashed
    value re-split it and the tag stopped meaning what it said. `("a\\0b", "c")` and
    `("a", "b\\0c")` are two topics on both sides, so `topics:2` was byte-identical, and both
    flatten to the same three tokens. Same class and direction as the region collision above:
    it fails OPEN, `update` calls the item unchanged, and the `item_topics` plane goes on
    serving a state the store no longer holds (rule 6).

    NOT A BYTE ONLY MEMORY CAN HOLD. `Enrichment.topics` has no pattern and JSON escapes the
    NUL, so the round trip through the REAL writer and reader is asserted BEFORE any hash:
    `save_store` / `load_store` hand back two items that still differ. What the store cannot
    express cannot collide; this it expresses.

    THE SECOND GUARD MAKES THIS A TEST OF THE ENCODING and not of two objects being different:
    both sides carry the same number of topics AND their `\\0`-joined flattening is
    byte-identical, so the serialisation this replaces could not tell them apart by
    construction. Seen RED at `1c32cf9`, both hashes `f970afed…`.
    """
    from xbrain.store import load_store, save_store

    store, _vocab, _pages = corpus
    plain = store["k02"].enriched.model_copy(update={"primary_topic": None})
    path = tmp_path / "items.json"
    sides = []
    for topics in (["a\0b", "c"], ["a", "b\0c"]):
        item = store["k02"].model_copy(
            update={"enriched": plain.model_copy(update={"topics": topics})}
        )
        save_store({item.id: item}, path)
        sides.append(load_store(path))
    left, right = sides

    assert left != right, "both states survive the store's own writer and reader"
    flat = ["\0".join(item_topics(side[store["k02"].id])) for side in sides]
    assert len(item_topics(left["k02"])) == len(item_topics(right["k02"])) == 2, (
        "`topics:2` on both"
    )
    assert flat[0] == flat[1], "and the join this replaces cannot tell the two apart at all"

    assert index_build.item_fingerprint(left["k02"]) != index_build.item_fingerprint(
        right["k02"]
    ), "a NUL inside a topic must not move where the next atom begins"
    assert index_build.store_fingerprint(left) != index_build.store_fingerprint(right), (
        "and the deep store signal must not inherit the collision"
    )


def test_the_index_options_are_carried_inert_and_02_7_is_what_must_redden_this(corpus) -> None:
    """`options` is accepted and DISCARDED, and until now only prose said so.

    `item_fingerprint` binds `options = options or IndexOptions()` and then reads neither
    `params` nor `vault_dir`, so `item_fingerprint(i, options=A) == item_fingerprint(i)` for
    every A. Two docstrings said it and nothing executable saw it — `vulture` does not flag a
    parameter that is bound.

    THIS IS A CHARACTERIZATION TEST: green before and after, repairing no defect. What it buys
    is the other direction. 02.7's builder is the first consumer that will genuinely read
    `params` and `vault_dir`, and the commit that makes the fingerprint depend on them MUST
    turn this red; reading them while it stayed green would mean the dependency never entered
    the hash — rule 6 armed, a re-chunk under new parameters reported as `0 cambiados`.

    So in 02.7 DELETE it, never weaken it.
    """
    from xbrain.knowledge.chunking import ChunkerParams

    store, _vocab, _pages = corpus
    item = store["k02"]
    bare = index_build.item_fingerprint(item)

    swept = index_build.IndexOptions(params=ChunkerParams(target=1, max_chars=2, min_chars=1))
    vaulted = index_build.IndexOptions(vault_dir=Path("/nowhere"))
    assert index_build.item_fingerprint(item, options=swept) == bare, "params are not read"
    assert index_build.item_fingerprint(item, options=vaulted) == bare, "nor is the vault"
    bare_store = index_build.store_fingerprint(store)
    assert index_build.store_fingerprint(store, options=swept) == bare_store, "and so does"
    assert index_build.store_fingerprint(store, options=vaulted) == bare_store, "the store one"


def test_the_leading_surface_version_is_the_belt_for_an_item_with_no_surfaces(
    corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An emitter-version bump has to reach an item that emits nothing.

    `SURFACE_VERSION` is hashed twice over: once at the head of the hashed structure and once
    inside every `surface.fingerprint`. For any item that HAS a surface the second copy
    carries it, which is why deleting the first left this file green — no test held an item
    with zero surfaces. Such items exist: k01 has no content, no media and no `enriched`, so
    blanking its text leaves the emitter with nothing to emit, and the leading copy is the
    only place the version can still reach it. Without it a bump would leave that item's
    fingerprint unchanged and `update` would report it current under an emitter version that
    no longer describes how it would be emitted. Patching this module's binding alone is an
    EXACT simulation here rather than a partial one: a real bump also travels through
    `ids.surface_fingerprint`, and this item has no surface for it to travel through.
    """
    store, _vocab, _pages = corpus
    item = store["k01"].model_copy(update={"text": ""})
    assert item_surfaces(item) == (), "the belt is load-bearing only with no surfaces"

    before = index_build.item_fingerprint(item)
    monkeypatch.setattr(index_build, "SURFACE_VERSION", index_build.SURFACE_VERSION + "-next")
    assert index_build.item_fingerprint(item) != before


def test_the_store_fingerprint_is_order_independent(corpus) -> None:
    """Two loads of the same store must agree, whatever order the dict happens to iterate in."""
    store, _vocab, _pages = corpus
    reversed_store = dict(reversed(list(store.items())))
    assert index_build.store_fingerprint(reversed_store) == index_build.store_fingerprint(store)


def test_load_index_inputs_binds_the_signal_to_the_bytes_it_read(three_inputs, corpus) -> None:
    """The mechanism behind P1b, at the loader: the signal is `fstat` of the handle the text
    came from, taken BEFORE the read — so a replacement landing during the read leaves the
    objects and the signal describing the same snapshot (the handle keeps the inode it opened;
    the store's writers replace atomically) and leaves the path on a newer inode the
    query-time `StoreSignal.of` reports as different. Staged INSIDE the read, which is the
    only place a stat of the path and a stat of the handle can disagree. Seen red under the
    mutation `stat = path.stat()` taken after the read: the signal then described the
    replacement, not the bytes parsed.
    """
    from xbrain.store import save_store

    store, _vocab, _pages = corpus
    items = three_inputs / "items.json"
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


def test_the_store_signal_is_one_stat_per_named_input_and_nothing_else(
    three_inputs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cheap signal must stay cheap, or it is the expensive one with a different name.

    THE NAME THIS REPLACES SAID «one stat» WHILE THE CODE DOES THREE. Round 10 corrected both
    docstrings that carried the stale claim and left the test's own name, which is the most
    widely read prose in a test file because it is what pytest prints. And the body asserted
    nothing the name claimed — it compared the `items` entry against `path.stat()` and stayed
    green under five stats, or none.

    What is pinned instead is the contract as it actually is, and it is NOT «three»
    unconditionally: `_stat_signal` answers `(0, 0)` for an unnamed path WITHOUT touching the
    filesystem, so the cost is one stat PER NAMED INPUT. Both halves are asserted because a
    signal that grew a second stat per file and a signal that stat'ed an unnamed path are
    different regressions, and each moves a different number here.
    """
    real_stat = Path.stat
    calls: list[Path] = []

    def counted(self, *args, **kwargs):
        calls.append(self)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", counted)
    items = three_inputs / "items.json"

    signal = index_build.StoreSignal.of(
        items, three_inputs / "vocab.yaml", three_inputs / "topics.json"
    )
    assert len(set(calls)) == len(calls) == 3, f"one stat per NAMED input, all distinct: {calls}"

    calls.clear()
    partial = index_build.StoreSignal.of(items)
    assert len(calls) == 1, "an unnamed input costs no stat at all"
    assert (partial.vocab_yaml_size, partial.topics_json_size) == (0, 0)

    assert signal.items_json_mtime_ns == real_stat(items).st_mtime_ns
    assert signal.items_json_size == real_stat(items).st_size


def test_the_store_signal_of_a_missing_store_is_zeroed_not_an_exception(tmp_path: Path) -> None:
    """A query must still be able to say "the index is behind" when the store is gone.

    Raising here would turn a missing store into a crash inside `search`, which is the wrong
    place to learn it: the response declares the degradation instead.
    """
    signal = index_build.StoreSignal.of(tmp_path / "nope.json")
    assert signal == index_build.StoreSignal(items_json_mtime_ns=0, items_json_size=0)


def test_an_absent_store_still_reads_as_an_empty_store(tmp_path: Path) -> None:
    """The positive control for the test above: ABSENT is the one reading that stays empty —
    `load_store`'s own semantics, and what `StoreSignal.of` reports as zeros — so a fresh
    checkout with no `data/` still loads, and only what EXISTS and cannot be read raises."""
    loaded = index_build.load_index_inputs(tmp_path / "nope.json")
    assert loaded.store == {}
    assert loaded.signal == index_build.StoreSignal(items_json_mtime_ns=0, items_json_size=0)


# ---------------------------------------------------------------------------
# THREE INPUTS, NOT ONE — and the two fingerprints over the other two
#
# The snapshot's suite reached `items.json` on every door and `vocab.yaml` / `topics.json` on
# none: neither fingerprint had a direct test anywhere in it. Measured before writing these:
# green at 14/14 with `load_index_inputs` reduced to `vocab=[], topic_pages={}` and the four
# vocab/topics signal entries left at zero — the pre-round-05 defect P1a records, back with
# nothing to catch it. This child owns the three-input read and the three fingerprints over
# it (rule 1: each was seen red under the mutation named in its docstring).
# ---------------------------------------------------------------------------


def test_load_index_inputs_reads_the_vocabulary_and_the_topic_pages_too(
    three_inputs: Path, corpus
) -> None:
    """P1a at the loader: the index derives from THREE files, so the loader reads three.

    A topic description enters every assigned item's PROFILE (spec §5.1.A) and the overviews
    and notes are chunks the index serves, so a loader that reads `items.json` and defaults
    the other two to empty builds an index missing the whole topic plane while every signal it
    seals says the inputs were read. All six signal entries are the stat of the file that was
    actually parsed. Seen red under the mutation `vocab=[], topic_pages={}` with the four
    vocab/topics signal entries left at zero — the pre-round-05 shape: `AssertionError` on the
    vocabulary, and on `vocab_yaml_mtime_ns` once the parse was restored.
    """
    _store, vocab, pages = corpus
    loaded = index_build.load_index_inputs(
        three_inputs / "items.json", three_inputs / "vocab.yaml", three_inputs / "topics.json"
    )

    assert [t.slug for t in loaded.vocab] == [t.slug for t in vocab], "the vocabulary was read"
    assert {t.slug: t.description for t in loaded.vocab} == {
        t.slug: t.description for t in vocab
    }, "descriptions included: they enter every assigned item's profile"
    assert set(loaded.topic_pages) == set(pages), "the topic pages were read"
    assert loaded.topic_pages["agent-evaluation"].overview == pages["agent-evaluation"].overview

    for name, mtime_field, size_field in [
        ("items.json", "items_json_mtime_ns", "items_json_size"),
        ("vocab.yaml", "vocab_yaml_mtime_ns", "vocab_yaml_size"),
        ("topics.json", "topics_json_mtime_ns", "topics_json_size"),
    ]:
        stat = (three_inputs / name).stat()
        assert getattr(loaded.signal, mtime_field) == stat.st_mtime_ns, name
        assert getattr(loaded.signal, size_field) == stat.st_size, name


def test_an_unnamed_vocab_or_topics_path_reads_as_empty_with_a_zero_signal(
    three_inputs: Path,
) -> None:
    """The two optional inputs default to absent, and absent is empty — never an error.

    `load_index_inputs(items_path)` is the one-argument call every pre-round-05 caller made
    and the CLI still makes for a corpus with no vocabulary yet. It must load, not raise.

    Seen red under the mutation `_read_bound` without its `if path is None` guard:
    `AttributeError: 'NoneType' object has no attribute 'open'`.
    """
    loaded = index_build.load_index_inputs(three_inputs / "items.json")
    assert loaded.vocab == []
    assert loaded.topic_pages == {}
    assert (loaded.signal.vocab_yaml_mtime_ns, loaded.signal.vocab_yaml_size) == (0, 0)
    assert (loaded.signal.topics_json_mtime_ns, loaded.signal.topics_json_size) == (0, 0)


@pytest.mark.parametrize(
    "filename, mtime_field, size_field",
    [
        ("items.json", "items_json_mtime_ns", "items_json_size"),
        ("vocab.yaml", "vocab_yaml_mtime_ns", "vocab_yaml_size"),
        ("topics.json", "topics_json_mtime_ns", "topics_json_size"),
    ],
)
def test_the_query_time_signal_watches_each_of_the_three_inputs(
    three_inputs: Path, filename: str, mtime_field: str, size_field: str
) -> None:
    """P1a's OTHER half: `StoreSignal.of` is the side a query pays, and it had one file tested.

    The loader half is covered — a `load_index_inputs` that defaulted the vocabulary and the
    topic pages to empty goes red above. `StoreSignal.of` is the comparison every `search`
    makes against what the manifest sealed, and only `items.json` was ever asserted on it, so
    `of` could be returned to watching one file — the pre-round-05 defect P1a records, where
    `xbrain topics` wrote `topics.json`, never touched `items.json`, and every later query
    answered over the old topic plane with nothing declared (spec §9.3 forbids that silence).
    Measured: zeroing the vocab and topics stats inside `of` left this file green. One case
    per input, each asserting only its own file's two entries, so zeroing ONE stat reddens
    exactly one case and names which input lost its watch.

    THE THREE SIZES MUST DIFFER, asserted first (rule 1): compared against its own file's
    `stat`, an entry filled from a DIFFERENT input's stat would still pass wherever the two
    files happened to agree. A zero cannot be right either — the non-empty precondition.

    Seen red under three mutants applied separately, one per line of `of`: `_stat_signal(…)`
    -> `(0, 0)` for the items, the vocabulary and the topic pages, each reddening its own case
    and leaving the other two green.
    """
    sizes = [
        (three_inputs / name).stat().st_size for name in ("items.json", "vocab.yaml", "topics.json")
    ]
    assert len(set(sizes)) == 3, "distinct sizes, or one file's stat could satisfy another's entry"
    assert all(sizes), "and none empty, or a zeroed entry would be indistinguishable from the truth"

    signal = index_build.StoreSignal.of(
        three_inputs / "items.json", three_inputs / "vocab.yaml", three_inputs / "topics.json"
    )
    stat = (three_inputs / filename).stat()
    assert getattr(signal, mtime_field) == stat.st_mtime_ns, filename
    assert getattr(signal, size_field) == stat.st_size, filename


def _obstruct(path: Path, obstacle: str) -> None:
    """Make `path` UNREADABLE without removing it — the two shapes an operator meets."""
    if obstacle == "chmod000":
        path.chmod(0)
    else:
        path.unlink()
        path.mkdir()


@pytest.mark.parametrize(
    "obstacle, error", [("chmod000", PermissionError), ("directory", IsADirectoryError)]
)
@pytest.mark.parametrize("filename", ["items.json", "vocab.yaml", "topics.json"])
def test_an_unreadable_input_is_an_error_never_an_empty_one(
    three_inputs: Path, filename: str, obstacle: str, error: type[OSError]
) -> None:
    """A-2 (gate Fable §5.2, round 08), on ALL THREE inputs and not only the one it probed.

    `_read_bound` caught EVERY `OSError` and answered `(None, 0, 0)`, the reading reserved for
    a file that is ABSENT, so an unreadable input loaded as an EMPTY one — the whole fail-open
    family with a destruction on top, enumerated in `_read_bound`'s own docstring and not
    restated here. The defect is a property of that function and all three inputs route
    through it, which is why this is parametrised over the three and not over the one the gate
    happened to probe: on `vocab.yaml` it is quieter and no less destructive, loading as «no
    topics» so every profile is built without its topic descriptions and `update` plans the
    deletion of the whole topic plane, exit 0.

    Seen red on `36f694b` (`loaded.store == {}` on both `items.json` obstacles, no exception)
    and under the mutation `except OSError` in `_read_bound`: `DID NOT RAISE` on all six.
    """
    _obstruct(three_inputs / filename, obstacle)
    try:
        with pytest.raises(error):
            index_build.load_index_inputs(
                three_inputs / "items.json",
                three_inputs / "vocab.yaml",
                three_inputs / "topics.json",
            )
    finally:
        if obstacle == "chmod000":
            (three_inputs / filename).chmod(0o644)


def test_the_vocab_fingerprint_covers_the_descriptions_not_only_the_slugs(corpus) -> None:
    """The descriptions are hashed because they are INDEXED TEXT, not metadata (spec §5.1.A).

    A topic description enters the profile of every item assigned to it, so editing one
    changes indexable text on the ITEM plane, not only on the topic plane. A fingerprint
    over the slugs alone would leave `update` reporting nothing to do while every profile
    the index serves quotes the old description — rule 6, on the plane the vocabulary owns.

    And it is order-independent for the same reason `store_fingerprint` is: the order a
    vocabulary happens to be loaded in is a property of the load, not of what it contains.

    Seen red under three mutations: the description dropped from each pair on the first
    assertion, the slug dropped on the second, and `sorted(...)` → `vocab` on the third. The
    rename PRESERVES SORT POSITION (`agent-evaluationz` sorts where `agent-evaluation` did) so
    the second assertion cannot pass merely by reordering — the way its first version did on
    the topics fingerprint below.
    """
    _store, vocab, _pages = corpus
    before = index_build.vocab_fingerprint(vocab)

    edited = [
        t.model_copy(update={"description": "otra descripción"}) if t.slug == vocab[0].slug else t
        for t in vocab
    ]
    assert index_build.vocab_fingerprint(edited) != before, "a description edit moves it"

    renamed = [t.model_copy(update={"slug": f"{t.slug}z"}) if t is vocab[0] else t for t in vocab]
    assert sorted(t.slug for t in renamed) == [t.slug for t in renamed], "sort position kept"
    assert index_build.vocab_fingerprint(renamed) != before, "and so does a slug"

    assert index_build.vocab_fingerprint(list(reversed(vocab))) == before, "order-independent"


def test_the_topics_fingerprint_covers_the_overview_every_note_and_the_slug(corpus) -> None:
    """The synthesised text of the topic plane, hashed through the SAME emitter the index uses.

    `topic_overview` and `topic_note` are surface types the index stores and `search`
    serves, so each is hashed through `surface_fingerprint` — the one definition — rather
    than through a second concatenation that could drift from it (rule 5). Three axes, and
    each is a chunk a consumer can be served: the overview, ANY note (not just the first),
    and the slug those chunks are filed under.

    THE RENAME PRESERVES SORT POSITION ON PURPOSE. The first version refiled the page as
    `renombrado`, which sorts AFTER `ai-policy` — so the two pages' text swapped places and
    the hash moved whether or not the slug was hashed at all. Measured: with the slug dropped,
    that version stayed GREEN and only `…_agrees_with_the_emitter` went red. Rule 1, in the
    test written to pin the slug. `agent-evaluationz` sorts where `agent-evaluation` did.

    Seen red under two mutations: dropping the notes comprehension (the second assertion) and
    dropping the slug from each page's entry (the third). Hashing `page.overview` directly
    instead of through `surface_fingerprint` stays GREEN here and is caught by
    `test_the_topics_fingerprint_agrees_with_the_emitter` below — why that test exists.
    """
    _store, _vocab, pages = corpus
    before = index_build.topics_fingerprint(pages)

    page = pages["agent-evaluation"]
    assert len(page.notes) >= 2, "this test is only meaningful on a page with several notes"

    edited = {**pages, "agent-evaluation": page.model_copy(update={"overview": "otro resumen"})}
    assert index_build.topics_fingerprint(edited) != before, "the overview is hashed"

    last_note_changed = page.model_copy(update={"notes": [*page.notes[:-1], "otra nota"]})
    edited = {**pages, "agent-evaluation": last_note_changed}
    assert index_build.topics_fingerprint(edited) != before, "and EVERY note, not just the first"

    refiled = {("agent-evaluationz" if s == "agent-evaluation" else s): p for s, p in pages.items()}
    assert sorted(refiled) == ["agent-evaluationz", "ai-policy"], "the sort position is unchanged"
    assert index_build.topics_fingerprint(refiled) != before, "and the slug they are filed under"


def test_the_topics_fingerprint_agrees_with_the_emitter(corpus) -> None:
    """One definition of "what a topic-page surface is", shared with `knowledge.ids` (rule 5).

    The fingerprint is asserted against `surface_fingerprint` computed HERE from the page's
    own text, so a change to the emitter's `(version, type, origin, text)` shape moves this
    fingerprint too and the index re-derives; hashing the raw text instead would leave a
    `SURFACE_VERSION` bump invisible to the topic plane while every item plane re-indexed.
    This is NOT the tautology `topics_fingerprint is topics_fingerprint`: the expected value
    is rebuilt from the page objects through the public emitter, so a version bump inside
    `surface_fingerprint` propagates to both sides but a change to the *composition* — the
    surface types, the order, the slug, the SHAPE the plane is encoded in — moves only one.
    Seen red under the mutation `surface_fingerprint("topic_note", ...)` →
    `…("topic_overview", ...)` for the notes, which every other test in this file passes.
    """
    import hashlib

    from xbrain.knowledge.ids import surface_fingerprint

    _store, _vocab, pages = corpus
    plane = [
        [
            slug,
            surface_fingerprint("topic_overview", "llm", pages[slug].overview),
            [surface_fingerprint("topic_note", "llm", n) for n in pages[slug].notes],
        ]
        for slug in sorted(pages)
    ]
    blob = json.dumps(["topics", plane], ensure_ascii=False)
    expected = hashlib.sha256(blob.encode("utf-8")).hexdigest()

    assert index_build.topics_fingerprint(pages) == expected


def test_a_nul_cannot_forge_an_atom_boundary_on_the_vocabulary_or_the_topic_plane(
    tmp_path: Path,
) -> None:
    """The same delimiter injection, on the two planes that never carried a length tag at all.

    `vocab_fingerprint` joined `f"{slug}={description}"` and `topics_fingerprint` joined each
    page's `[overview_fp, *note_fps, slug]`, both with `\\0` — so a NUL in a description or in
    a page KEY re-splits the stream exactly as on the item plane. One vocabulary entry can
    impersonate two, and one topic page whose key smuggles a NUL plus another page's OVERVIEW
    FINGERPRINT can impersonate two pages. The round 10 brute force that found no
    `topics_fingerprint` collision drew slugs from an alphabet WITHOUT a NUL, so its 3,608
    stores could not have reached this one, and `parse_topic_pages` takes the key verbatim.
    `Topic.slug` IS pattern-constrained, so the vocabulary side injects through `description`.

    Both survive their own writer and reader — PyYAML emits the NUL in a double-quoted scalar,
    JSON as an escape — asserted before any hash. Each half then guards that the join this
    replaces produced the SAME bytes for the two states, so neither assertion can pass merely
    because two objects differ. Seen RED at `1c32cf9`, vocabulary `755851876951afa8…` and
    topic plane `9e19e27318ea9436…` on both sides.
    """
    from xbrain.knowledge.ids import surface_fingerprint
    from xbrain.rubrics import load_vocab, save_vocab
    from xbrain.store import load_topic_pages, save_topic_pages

    vocab_path = tmp_path / "vocab.yaml"
    vocabs = []
    for entries in ([("a", "x\0b=y")], [("a", "x"), ("b", "y")]):
        save_vocab([Topic(slug=s, description=d) for s, d in entries], vocab_path)
        vocabs.append(load_vocab(vocab_path))
    assert vocabs[0] != vocabs[1], "one entry and two entries, both back through PyYAML"
    joined = ["\0".join(f"{t.slug}={t.description}" for t in v) for v in vocabs]
    assert joined[0] == joined[1], "the join this replaces serialised both to the same bytes"
    assert index_build.vocab_fingerprint(vocabs[0]) != index_build.vocab_fingerprint(vocabs[1]), (
        "a NUL in a description must not split one vocabulary entry into two"
    )

    page = TopicPage(
        slug="b",
        overview="segunda",
        notes=[],
        synthesized_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        post_count_at_synth=1,
    )
    first = page.model_copy(update={"slug": "a", "overview": "primera"})
    smuggled = f"a\0{surface_fingerprint('topic_overview', 'llm', page.overview)}\0b"
    topics_path = tmp_path / "topics.json"
    planes = []
    for pages in ({smuggled: first}, {"a": first, "b": page}):
        save_topic_pages(pages, topics_path)
        planes.append(load_topic_pages(topics_path))
    assert list(planes[0]) != list(planes[1]), "one page filed under a forged key, against two"
    streams = [
        "\0".join(
            f"{surface_fingerprint('topic_overview', 'llm', p.overview)}\0{s}"
            for s, p in sorted(plane.items())
        )
        for plane in planes
    ]
    assert streams[0] == streams[1], "and the join this replaces serialised both to the same bytes"
    assert index_build.topics_fingerprint(planes[0]) != index_build.topics_fingerprint(planes[1]), (
        "a page key carrying another page's overview fingerprint must not impersonate it"
    )


def test_the_index_loader_and_the_store_doors_parse_through_one_parser_each(
    three_inputs: Path, corpus
) -> None:
    """The seam `parse_store` / `parse_vocab` / `parse_topic_pages` exists FOR THIS (rule 5).

    `load_index_inputs` must bind what it parsed to the exact bytes it read, so it reads
    through its own handle — and the only honest way to do that without a second copy of every
    parse is to split the parse out of `load_store` / `load_vocab` / `load_topic_pages` and
    share it. The property that matters is not "the seam exists" (a tautology the moment it
    does) but that the TWO READERS OF THE SAME BYTES agree: the path-based door every other
    stage uses, and the handle-based one the index uses. Seen red under the mutation
    `parse_vocab` → `data.get("topics")` without `Topic(**entry)` (raw dicts out of one door,
    `Topic` out of the other) and under a loader that parsed the store with a bare
    `json.loads`.
    """
    from xbrain.rubrics import load_vocab
    from xbrain.store import load_store, load_topic_pages

    loaded = index_build.load_index_inputs(
        three_inputs / "items.json", three_inputs / "vocab.yaml", three_inputs / "topics.json"
    )

    assert loaded.store == load_store(three_inputs / "items.json")
    assert loaded.vocab == load_vocab(three_inputs / "vocab.yaml")
    assert loaded.topic_pages == load_topic_pages(three_inputs / "topics.json")
