# tests/test_knowledge_models.py
"""The read models are frozen, closed, and refuse to carry a lie (Plan 01 §3.4).

These models are the vocabulary Plans 02–04 consume without renegotiating, so the point of
this file is the CONSTRAINTS, not the field list: `frozen=True` (a surface that can be
mutated after emission is a fingerprint that stops meaning anything), `extra="forbid"` (a
typo'd field name silently accepted is a contract that drifts), and the invariants that a
hand-edited record must not be able to violate.

WHY `verification` IS NOT HERE. It was a field on `KnowledgeSurface` in an earlier draft.
It cannot be: `surface_fingerprint` hashes (version, type, origin, text) and does NOT depend
on the verdict, so a FAIL revoked by `verify --audit` would keep being served as the PASS it
used to be — CLAUDE.md rule 6 backwards. The verdict is hydrated from the LIVE store at
response time instead; `test_knowledge_surfaces.py` owns that behaviour.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from xbrain.knowledge.models import (
    UNFETCHED_REASON_BY_FAILURE,
    DerivedText,
    KnowledgeChunk,
    KnowledgeItem,
    KnowledgeSurface,
    Locator,
    SourceFailure,
    TopicRecord,
    UnfetchedLink,
    UnfetchedReason,
)
from xbrain.models import Author, FailureReason

UTC = timezone.utc
HEX64 = "a" * 64


def _surface(**overrides) -> KnowledgeSurface:
    defaults = dict(
        surface_id="item:42:post:0",
        owner_type="item",
        owner_id="42",
        surface_type="post",
        text="the post body",
        origin="source",
        trust_class="primary_source",
        derived=False,
        locator=Locator(kind="item_text"),
        fingerprint=HEX64,
    )
    return KnowledgeSurface(**{**defaults, **overrides})


def _chunk(**overrides) -> KnowledgeChunk:
    defaults = dict(
        chunk_id="item:42:post:0:0:xbrain-knowledge-chunker/v1",
        surface_id="item:42:post:0",
        owner_type="item",
        owner_id="42",
        surface_type="post",
        text="the post body",
        chunk_index=0,
        char_start=0,
        char_end=13,
        origin="source",
        trust_class="primary_source",
        derived=False,
        fingerprint=HEX64,
    )
    return KnowledgeChunk(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# Frozen and closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_surface, _chunk])
def test_models_are_frozen(factory) -> None:
    """A surface whose text can be mutated after emission has a fingerprint that lies.

    The whole verifiability claim (spec §3.8) is *the retrieved text is verbatim with
    respect to that surface, and the fingerprint proves the index matches the current
    data*. Mutability breaks both halves at once.
    """
    instance = factory()
    with pytest.raises(ValidationError):
        instance.text = "something else"


@pytest.mark.parametrize("factory", [_surface, _chunk])
def test_models_forbid_unknown_fields(factory) -> None:
    """`extra="forbid"` — a misspelled field must be an error, not a silent no-op.

    These shapes carry no `schema_version` of their own: the number lives on the envelopes
    that transport them, and `contracts.py` states that policy once, versioning each envelope
    independently. So a model that swallows unknown keys lets a producer "add" a field that no
    consumer ever sees AND that no envelope number could ever announce — the drift spec §7.1
    exists to prevent, CLI and MCP implementing two formats, arriving unversioned.
    """
    with pytest.raises(ValidationError):
        factory(definitely_not_a_field=1)


def test_fingerprints_must_be_lowercase_sha256_hex() -> None:
    """A garbage fingerprint is rejected at construction, not discovered at query time.

    Same defence `VerificationVerdict.output_fingerprint` already carries: a hand-edited
    store cannot ship a hash that no comparison will ever match while looking plausible.
    """
    with pytest.raises(ValidationError):
        _surface(fingerprint="not-a-hash")
    with pytest.raises(ValidationError):
        _surface(fingerprint=("A" * 64))


def test_surface_has_no_verification_field() -> None:
    """M5, asserted as an absence.

    `surface_fingerprint` does not depend on the verdict, so a persisted verdict on the
    surface could never be invalidated when the verdict changed — a revoked FAIL would keep
    being served as the old PASS. Written as a test because "we decided not to add a field"
    is otherwise invisible to the next person, who will helpfully add it.
    """
    assert "verification" not in KnowledgeSurface.model_fields


# ---------------------------------------------------------------------------
# Locator
# ---------------------------------------------------------------------------


def test_locator_requires_only_its_kind() -> None:
    """Every positional field is optional: an `item_text` surface has no source index."""
    assert Locator(kind="item_text").source_index is None


def test_locator_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        Locator(kind="somewhere")


def test_locator_reserves_the_article_block_slot() -> None:
    """The X-Article inline-image hole is DECLARED, not forgotten (m18).

    Measured 2026-08-31: 250 `ArticleImageBlock`s in the corpus, 0 with a description — so
    there is no text to index today. When `describe` starts describing them they arrive as
    `image_description` with `locator.kind = "article_block"` and a `block_index`. Keeping
    the slot pinned is what stops that from looking like an oversight later.
    """
    locator = Locator(kind="article_block", block_index=4, source_index=1)
    assert (locator.block_index, locator.source_index) == (4, 1)


# ---------------------------------------------------------------------------
# The two ways a link ends up with no body
# ---------------------------------------------------------------------------


def test_every_store_failure_reason_maps_to_an_unfetched_reason() -> None:
    """Totality: a new `FailureReason` in the store cannot fall into a wrong bucket.

    Seen red by deleting `"paywall"` from the map. Without this, adding a failure reason to
    `models.FailureReason` would quietly land in whatever branch a `.get(..., default)`
    chose, and a DNS failure would be reported to the consumer as an HTTP error.
    """
    from typing import get_args

    assert set(get_args(FailureReason)) == set(UNFETCHED_REASON_BY_FAILURE)
    assert set(UNFETCHED_REASON_BY_FAILURE.values()) <= set(get_args(UnfetchedReason))


def test_a_dns_failure_is_not_reported_as_an_http_error() -> None:
    """The bucket has to be true, not merely available.

    `timeout` / `dns_error` / `unknown_error` are not HTTP refusals and not extraction
    failures — they are "the fetch failed", which is exactly the wording the repo already
    uses in `executors.api._FAILURE_CLAUSE`. Squeezing them into `http_error` to keep a
    four-value enum would state a fact about the page that nobody measured.
    """
    assert UNFETCHED_REASON_BY_FAILURE["dns_error"] == "fetch_failed"
    assert UNFETCHED_REASON_BY_FAILURE["not_found"] == "http_error"
    assert UNFETCHED_REASON_BY_FAILURE["js_required"] == "not_extractable"
    assert UNFETCHED_REASON_BY_FAILURE["blocked_interstitial"] == "blocked_interstitial"


def test_unfetched_link_carries_the_reason_and_never_a_body() -> None:
    """Spec §4: an unfetched link is METADATA the consumer must see, with its cause.

    Naming the cause never licenses describing the content, so the model has no text field
    at all — the absence is the guardrail.
    """
    link = UnfetchedLink(url="https://dead.example/a", reason="http_error", detail="HTTP 404")
    assert "text" not in UnfetchedLink.model_fields
    assert link.detail == "HTTP 404"


def test_source_failure_keeps_kind_url_and_reason() -> None:
    failure = SourceFailure(
        kind="external_article",
        url="https://dead.example/a",
        failure_reason="not_found",
        error="404 Not Found",
    )
    assert (failure.kind, failure.failure_reason) == ("external_article", "not_found")


# ---------------------------------------------------------------------------
# KnowledgeItem / TopicRecord
# ---------------------------------------------------------------------------


def _knowledge_item(**overrides) -> KnowledgeItem:
    defaults = dict(
        item_id="42",
        source="bookmark",
        url="https://x.com/a/status/42",
        author=Author(handle="a", name="A"),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        captured_at=datetime(2026, 1, 2, tzinfo=UTC),
        primary_topic=None,
        topics=(),
        available_surfaces=(),
        content_kinds=(),
    )
    return KnowledgeItem(**{**defaults, **overrides})


def test_knowledge_item_separates_failed_sources_from_unfetched_links() -> None:
    """Two distinct facts, two distinct fields (m7).

    "We tried and it failed" and "there is no body for this URL" are not the same claim,
    and collapsing them would make a link that was never attempted indistinguishable from
    one that returned a 404.
    """
    assert "failed_sources" in KnowledgeItem.model_fields
    assert "unfetched_links" in KnowledgeItem.model_fields
    item = _knowledge_item()
    assert item.failed_sources == () and item.unfetched_links == ()


def test_topic_record_staleness_is_derived_not_stored_twice() -> None:
    """`stale` is a field, but it is computed by the emitter from the live post count.

    `TopicPage` deliberately stores `post_count_at_synth` rather than a `stale` flag,
    because a stored flag desyncs. The read model carries the derived answer so consumers
    do not each re-derive it — one computation, in the emitter.
    """
    record = TopicRecord(
        topic_id="topic:ai-coding",
        slug="ai-coding",
        description=DerivedText(text="d", origin="unknown"),
        overview=None,
        notes=(),
        primary_item_ids=(),
        secondary_item_ids=(),
        synthesized_at=None,
        post_count_at_synth=None,
        stale=True,
        vocab_fingerprint=HEX64,
        synthesis_fingerprint=None,
    )
    assert record.stale is True and record.synthesis_fingerprint is None


def test_topic_layers_carry_their_own_provenance() -> None:
    """Spec §3.6: the description, the overview and the notes are *derived layers WITH THEIR
    OWN provenance*.

    They are `DerivedText`, not bare strings, for the reason m-ii gives for
    `SearchResult.summary`: nesting the text together with its `origin` is how invariant 2
    is satisfied structurally. A single `origin` on `TopicRecord` would have to lie about one
    of them — the vocabulary does not record whether a description was written or generated
    (`unknown`), while the overview and the notes are known LLM output.
    """
    assert TopicRecord.model_fields["description"].annotation is DerivedText
    assert "origin" in DerivedText.model_fields
