# tests/test_knowledge_ids.py
"""Identity must be reproducible and must NOT depend on list order (Plan 01 §3.3).

Spec §3.3 requires ids that are stable *while owner, surface and chunker do not change*,
free of collisions between a quoted tweet, a thread, an article and a video, reversible
from chunk back to surface and owner, and explicitly migrated when the chunker changes.

THE DECISION THIS FILE DEFENDS. A content source's key is `sha1(kind\\0url)[:12]`, not its
index in `content.sources` — because `fetch` REWRITES that list on every re-capture. An
index-keyed id would silently repoint every stored chunk at a different body the first time
a re-fetch reordered two sources, and nothing downstream would notice: the id would still
resolve, just to the wrong text.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from xbrain.knowledge.ids import (
    CHUNKER_VERSION,
    SURFACE_VERSION,
    IdError,
    chunk_fingerprint,
    chunk_id,
    content_source_keys,
    surface_fingerprint,
    surface_id,
    topic_id,
)
from xbrain.models import Author, Content, ContentSourceFailure, ContentSourceSuccess, Item

UTC = timezone.utc


def _item(sources: list) -> Item:
    return Item(
        id="42",
        source="bookmark",
        url="https://x.com/a/status/42",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        captured_at=datetime(2026, 1, 2, tzinfo=UTC),
        content=Content(fetched_at=datetime(2026, 1, 3, tzinfo=UTC), sources=sources),
    )


_ARTICLE_A = ContentSourceSuccess(kind="external_article", url="https://a.example", text="A body")
_ARTICLE_B = ContentSourceSuccess(kind="external_article", url="https://b.example", text="B body")


# ---------------------------------------------------------------------------
# Stability under reordering — the reason the key is a URL hash
# ---------------------------------------------------------------------------


def test_source_keys_survive_reordering_of_content_sources() -> None:
    """`fetch` rewrites the order of `content.sources`; the ids must not move with it.

    Seen red by keying on the list index: with `[A, B]` the article at
    `https://a.example` gets key 0, and with `[B, A]` it gets key 1 — the same body under
    two ids, and every previously stored chunk now points at B's text.
    """
    forward = content_source_keys(_item([_ARTICLE_A, _ARTICLE_B]))
    reversed_ = content_source_keys(_item([_ARTICLE_B, _ARTICLE_A]))
    assert forward[0] == reversed_[1], "A's key moved when the list was reordered"
    assert forward[1] == reversed_[0], "B's key moved when the list was reordered"


def test_source_keys_ignore_a_failure_entry_but_keep_real_indices() -> None:
    """Failures carry no text and get no surface, but they still occupy a list position.

    The returned mapping is keyed by the REAL index in `content.sources` so a
    `locator.source_index` built from it points at the right entry.
    """
    keys = content_source_keys(
        _item(
            [
                ContentSourceFailure(
                    kind="external_article", url="https://dead.example", failure_reason="not_found"
                ),
                _ARTICLE_A,
            ]
        )
    )
    assert set(keys) == {1}


def test_different_kinds_at_the_same_url_do_not_collide() -> None:
    """Spec §3.3: no collisions between a quoted tweet, a thread, an article and a video.

    The kind is hashed alongside the URL precisely because an `x_article` and an
    `external_article` can legitimately point at the same canonical URL.
    """
    keys = content_source_keys(
        _item(
            [
                ContentSourceSuccess(kind="x_article", url="https://same.example", text="one"),
                ContentSourceSuccess(
                    kind="external_article", url="https://same.example", text="two"
                ),
            ]
        )
    )
    assert keys[0] != keys[1]


# ---------------------------------------------------------------------------
# The duplicate `(kind, url)` path — 0 items today, implemented anyway
# ---------------------------------------------------------------------------


def test_duplicate_kind_and_url_get_distinct_deterministic_keys() -> None:
    """Two sources sharing `(kind, url)` must not share an id.

    Measured on the corpus: 0 items do this today, so this is a safe failure path rather
    than the normal case. It is implemented and tested anyway because 119 items already
    carry more than one source and nothing prevents two of them from sharing a URL — and
    the failure mode (two surfaces collapsing onto one id) is silent.
    """
    duplicate = ContentSourceSuccess(
        kind="external_article", url="https://a.example", text="a different body"
    )
    keys = content_source_keys(_item([_ARTICLE_A, duplicate]))
    assert keys[0] != keys[1]
    assert keys[1].endswith("#1"), "the suffix must be the occurrence ordinal, not a hash"
    assert content_source_keys(_item([_ARTICLE_A, duplicate])) == keys, "not deterministic"


def test_the_first_occurrence_carries_no_suffix() -> None:
    """A duplicate must not change the key of the source that was already there.

    Otherwise re-capturing a page that X happened to serve twice would renumber the
    original, invalidating every chunk built from it.
    """
    duplicate = ContentSourceSuccess(kind="external_article", url="https://a.example", text="dup")
    alone = content_source_keys(_item([_ARTICLE_A]))
    with_dup = content_source_keys(_item([_ARTICLE_A, duplicate]))
    assert with_dup[0] == alone[0]


# ---------------------------------------------------------------------------
# The id shapes
# ---------------------------------------------------------------------------


def test_surface_id_shape_and_reverse_resolution() -> None:
    """`<owner_type>:<owner_id>:<surface_type>:<source_key>` — spec §3.3."""
    assert surface_id("item", "42", "post", "0") == "item:42:post:0"
    assert surface_id("topic", "ai-coding", "topic_note", "3") == "topic:ai-coding:topic_note:3"


def test_topic_id_shape() -> None:
    assert topic_id("agent-evaluation") == "topic:agent-evaluation"


def test_chunk_id_carries_the_chunker_version() -> None:
    """Spec §3.3: *migración explícita cuando cambie la versión del chunker*.

    The version is IN the id, so chunks produced by two chunkers cannot be mistaken for
    each other — a rebuild after a bump writes new ids rather than overwriting old ones
    with differently-cut text.
    """
    sid = surface_id("item", "42", "external_article", "abc123")
    assert chunk_id(sid, 7) == f"{sid}:7:{CHUNKER_VERSION}"
    assert chunk_id(sid, 7, chunker_version="other/v9") == f"{sid}:7:other/v9"


@pytest.mark.parametrize("bad", ["", "a:b", "with space", "UPPER"])
def test_topic_id_rejects_a_non_slug(bad: str) -> None:
    """A slug with a colon would make `topic_id` ambiguous to parse back.

    Ids are meant to be reversible (spec §3.3); accepting arbitrary text here would make
    `topic:a:b` unparseable and push the ambiguity into every consumer.
    """
    with pytest.raises(IdError):
        topic_id(bad)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def test_surface_fingerprint_changes_with_every_arm() -> None:
    """Text, surface type, origin and the emitter version each move the hash.

    The type and the origin are hashed alongside the text because the SAME string can
    legitimately appear under two surfaces (a summary quoting the post verbatim), and they
    are not interchangeable evidence.
    """
    base = surface_fingerprint("post", "source", "hello")
    assert surface_fingerprint("post", "source", "hello!") != base
    assert surface_fingerprint("summary", "source", "hello") != base
    assert surface_fingerprint("post", "llm", "hello") != base
    assert surface_fingerprint("post", "source", "hello", surface_version="other/v2") != base
    assert len(base) == 64 and base == base.lower()


def test_chunk_fingerprint_changes_when_the_chunker_version_is_bumped() -> None:
    """Spec §5.2: *la versión del algoritmo forma parte del fingerprint*.

    Bumping the version must invalidate every chunk, because the same text cut differently
    is not the same chunk. The version is an ARGUMENT with the module constant as its
    default, so this can be asserted without monkeypatching a module global.
    """
    sid = surface_id("item", "42", "post", "0")
    base = chunk_fingerprint(sid, 0, "hello")
    assert chunk_fingerprint(sid, 0, "hello", chunker_version="xbrain-knowledge-chunker/v2") != base
    assert chunk_fingerprint(sid, 1, "hello") != base
    assert chunk_fingerprint(surface_id("item", "43", "post", "0"), 0, "hello") != base


def test_fingerprint_defaults_are_the_module_versions() -> None:
    """The default argument IS the constant — otherwise the version could silently drift.

    Without this, `SURFACE_VERSION` could be bumped for the manifest while the fingerprints
    kept hashing the old value, and the index would believe it was current.
    """
    assert surface_fingerprint("post", "source", "x") == surface_fingerprint(
        "post", "source", "x", surface_version=SURFACE_VERSION
    )
    sid = surface_id("item", "42", "post", "0")
    assert chunk_fingerprint(sid, 0, "x") == chunk_fingerprint(
        sid, 0, "x", chunker_version=CHUNKER_VERSION
    )


def test_the_nul_separator_stops_two_arms_concatenating_into_a_third_value() -> None:
    """`("ab", "c")` and `("a", "bc")` must not hash alike.

    The same reason `verification.contract_fingerprint` uses NUL. This is a content hash,
    not a security primitive — it only has to make two different inputs land on two
    different values, and naive concatenation does not.
    """
    assert surface_fingerprint("post", "source", "ab") != surface_fingerprint(
        "post", "sourcea", "b"
    )
