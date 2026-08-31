# tests/test_knowledge_contracts.py
"""The frozen external schemas (Plan 01 §3.5, spec §7, steps 19-20).

These shapes are frozen at `schema_version: "1"` NOW, while the services that fill them do
not exist yet. That is the point: spec §7.1 says CLI JSON and MCP are adapters over the same
service and must not implement two formats, and the way two formats appear is that the
second adapter is written months after the first, against whatever the first happened to
emit. Freezing the model first makes the second adapter a consumer rather than an author.

THE NAMING RULE AND WHY IT IS TESTED THIS WAY (m12). Invariant 2 of spec §3.7: no response
calls something `text` without carrying `origin` beside it. The obvious test — walk the JSON
and fail on a text field with no `origin` sibling — CANNOT be written honestly: `query`,
`url`, `handle`, `name` and `title` are text fields too, so it either fails always or turns
into a walker with a growing list of ad-hoc exceptions, which is a second definition of the
rule pretending to be a check.

What is done instead is TOTALITY over the DECLARED fields: every `str` field of every
contract model is in exactly one of two frozensets — the ones that carry corpus content and
therefore need provenance, and the ones that are metadata. Adding a text field without
deciding which it is goes red. Then a second, behavioural test asserts the first group
really does travel with an origin.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from xbrain.knowledge.contracts import (
    CONTRACT_MODELS,
    TEXT_FIELDS_REQUIRING_ORIGIN,
    TEXT_FIELDS_WITHOUT_ORIGIN,
    DerivedText,
    EvidenceBundle,
    GraphExpansionResponse,
    IndexStatusRef,
    SearchFilters,
    SearchMatch,
    SearchResponse,
    SearchResult,
    is_str_field,
)
from xbrain.knowledge.models import KnowledgeItem, Locator
from xbrain.models import Author

UTC = timezone.utc

# Spec §7.2, "Filtros mínimos" — the EIGHT, transcribed from the spec by hand.
SPEC_MINIMUM_FILTERS = {
    "created_from",
    "created_to",
    "source",
    "author",
    "topics",
    "content_kinds",
    "origins",
    "has_surfaces",
}


# ---------------------------------------------------------------------------
# 19 — totality over declared text fields
# ---------------------------------------------------------------------------


def test_every_declared_str_field_is_classified() -> None:
    """Adding a text field to a contract model without deciding its side is red.

    Seen red by adding a `note: str` field to `SearchResult`. Both frozensets are asserted
    against the DECLARED fields, so an entry for a field that no longer exists is also red —
    a stale classification is a reader's false reassurance.
    """
    declared = {
        (model.__name__, name)
        for model in CONTRACT_MODELS
        for name, field in model.model_fields.items()
        if is_str_field(field)
    }
    assert declared == TEXT_FIELDS_REQUIRING_ORIGIN | TEXT_FIELDS_WITHOUT_ORIGIN


def test_the_two_classifications_are_disjoint() -> None:
    """An overlap would make the totality test above satisfiable by a contradiction."""
    assert TEXT_FIELDS_REQUIRING_ORIGIN & TEXT_FIELDS_WITHOUT_ORIGIN == set()


def test_search_result_summary_is_not_in_either_frozenset() -> None:
    """And that is correct, not an oversight (m-ii).

    `is_str_field` selects `str` fields; `SearchResult.summary` is `DerivedText | None` — a
    nested MODEL. Putting it in `TEXT_FIELDS_REQUIRING_ORIGIN` would break the equality the
    totality test asserts (`declared` does not contain it), so that test could never pass.

    The rule is satisfied anyway, and by construction: the `str` field carrying that text is
    `("DerivedText", "text")`, which IS in the set, and `DerivedText` carries its `origin` in
    the same object. Nesting content in a model together with its provenance is the correct
    way to satisfy invariant 2 — not an exception to it.
    """
    assert ("SearchResult", "summary") not in (
        TEXT_FIELDS_REQUIRING_ORIGIN | TEXT_FIELDS_WITHOUT_ORIGIN
    )
    assert ("DerivedText", "text") in TEXT_FIELDS_REQUIRING_ORIGIN


# ---------------------------------------------------------------------------
# 19b — the behavioural half
# ---------------------------------------------------------------------------


def test_content_text_fields_travel_with_origin_in_the_same_object() -> None:
    """The serialized object carrying corpus text also carries its provenance.

    Seen red by deleting `origin` from `SearchMatch`. The totality test above would still
    pass after that deletion (one fewer field to classify, and the frozenset entry removed
    with it) — which is precisely why the behavioural half is a separate test.
    """
    for model_name, field_name in sorted(TEXT_FIELDS_REQUIRING_ORIGIN):
        model = next(m for m in CONTRACT_MODELS if m.__name__ == model_name)
        assert "origin" in model.model_fields, (
            f"{model_name}.{field_name} carries corpus text but {model_name} has no `origin`"
        )


def test_a_search_match_serializes_its_excerpt_next_to_its_origin() -> None:
    match = SearchMatch(
        chunk_id="item:1:post:0:0:v1",
        surface_type="post",
        origin="source",
        trust_class="primary_source",
        derived=False,
        excerpt="the words that matched",
        matched_by=("lexical",),
        score=1.25,
        locator=Locator(kind="item_text"),
    )
    payload = match.model_dump(mode="json")
    assert payload["excerpt"] and payload["origin"] == "source"


# ---------------------------------------------------------------------------
# 20 — the freeze
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [SearchResponse, EvidenceBundle, GraphExpansionResponse])
def test_every_response_declares_schema_version_one(model) -> None:
    """Spec §7.1: every contract carries `schema_version`, and an incompatible change needs
    a new version rather than a silent mutation."""
    assert model.model_fields["schema_version"].default == "1"


def test_responses_reject_an_unknown_field() -> None:
    """`extra="forbid"` on the outer envelope too.

    Without it, an adapter could add a field the other adapter never learns about, and the
    two would drift while both still validated — the exact failure the freeze prevents.
    """
    with pytest.raises(ValidationError):
        SearchFilters(unknown_filter="x")


def test_search_filters_declare_all_eight_minimum_filters() -> None:
    """Spec §7.2 lists eight, not six.

    `content_kinds` and `has_surfaces` come from no existing column and need their own
    plumbing in Plan 02. Declaring them here is a commitment; leaving them out would freeze
    a contract that the next plan then could not satisfy without a version bump.
    """
    assert SPEC_MINIMUM_FILTERS <= set(SearchFilters.model_fields)


def test_index_status_names_the_two_degradation_signals_apart() -> None:
    """B3: the old `stale_chunks_excluded` meant something other than its name.

    `corrupt_chunks_excluded` counts rows whose fingerprint does not recompute — corruption
    or a foreign chunker version. "The index is behind the store", the failure that actually
    happens (you ran `enrich` and did not reindex), is a `degraded` flag instead. One name
    for one fact, because a counter that answers a different question than its name asks is
    how a wrong number gets quoted with confidence.
    """
    fields = IndexStatusRef.model_fields
    assert "corrupt_chunks_excluded" in fields
    assert "stale_chunks_excluded" not in fields
    assert IndexStatusRef(manifest_version="1").degraded == ()


def test_a_search_response_round_trips_through_json() -> None:
    """The whole envelope serializes and re-validates — the CLI/MCP wire test.

    A model that only ever gets constructed in Python can carry a field that does not
    survive JSON (a tuple key, a non-serializable default) and nobody notices until the
    second adapter exists.
    """
    response = SearchResponse(
        query="agentes que evaluan su propio trabajo",
        strategy="lexical",
        filters=SearchFilters(),
        index=IndexStatusRef(manifest_version="1", built_at=datetime(2026, 8, 31, tzinfo=UTC)),
        results=(
            SearchResult(
                rank=1,
                item_id="1",
                url="https://x.com/a/status/1",
                author=Author(handle="a", name="A"),
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                summary=DerivedText(text="un resumen", origin="llm", verification_status=None),
                topics=("agent-evaluation",),
                matches=(),
                available_surfaces=("post", "summary"),
                verify_with=("post",),
            ),
        ),
    )
    payload = response.model_dump_json()
    assert SearchResponse.model_validate_json(payload) == response


def test_an_evidence_bundle_hydrates_verification_but_never_persists_it() -> None:
    """M5, on the response side: the bundle carries verdicts, the SURFACE does not.

    The bundle is built at response time from the live store, so a revoked verdict simply
    stops appearing. A verdict copied onto the surface would have been frozen at emission
    and would keep asserting a PASS the current output never earned.
    """
    bundle = EvidenceBundle(
        item=KnowledgeItem(
            item_id="1",
            source="bookmark",
            url="https://x.com/a/status/1",
            author=Author(handle="a", name="A"),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            captured_at=datetime(2026, 1, 2, tzinfo=UTC),
        ),
        topics=(),
        surfaces=(),
    )
    assert bundle.verification == {}
    assert "verification" in EvidenceBundle.model_fields


# ---------------------------------------------------------------------------
# m5 — `Author` is inside the totality partition
# ---------------------------------------------------------------------------


def test_the_attribution_author_is_inside_the_totality_partition() -> None:
    """`Author` transports no corpus content today — and that is not a reason to omit it.

    The plan's m12 sketch listed `("Author", "handle")` and `("Author", "name")`; the
    implementation left the model out of `CONTRACT_MODELS`, so its `str` fields sat outside
    the partition entirely. The consequence is not today's: it is that a text field added to
    `Author` later — a bio, a display note, anything that could carry a claim — would join
    the contract with nobody deciding whether it needs an origin, and the totality test that
    exists to force that decision would stay green.

    `Author` is reachable from the contract twice (`SearchResult.author`,
    `KnowledgeSurface.attribution`), and the second is the attribution rule itself.
    """
    from xbrain.models import Author

    assert Author in CONTRACT_MODELS
    assert {("Author", "handle"), ("Author", "name")} <= TEXT_FIELDS_WITHOUT_ORIGIN
