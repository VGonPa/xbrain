# tests/test_knowledge_contracts.py
"""The frozen external schemas (Plan 01 §3.5, spec §7, steps 19-20).

These shapes are frozen per envelope NOW (`SearchResponse` "1", `EvidenceBundle` "2" since
U-1, the graph envelope "1"), while the services that fill them do
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
    SEARCH_SCHEMA_VERSION,
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


# The version each envelope carries TODAY, and why (U-1, round 07). `SearchResponse` and
# `GraphExpansionResponse` is still the Plan 01 freeze: nothing in its shape moved.
# `EvidenceBundle` is at "2" because round 06
# made `KnowledgeChunk.locator` REQUIRED, and under `extra="forbid"` that is the incompatible
# change the freeze exists to name: the Pydantic consumer of version 1 refuses a bundle with
# the key (`chunks.0.locator: Extra inputs are not permitted`, measured against the exact
# `origin/develop` model) and the new consumer refuses a version-1 document (`Field
# required`). Two producers announcing one version that do not interoperate is the silent
# mutation spec §7.1 forbids; the number is what makes the refusal honest.
# `SearchResponse` is at "2" for the same rule read forwards (B2): `SearchMatch` gained the
# `title` spec §4 makes accompany a chunk, and a DEFAULTED key is additive only for the new
# consumer — the version-1 one forbids it, so the number moves for its refusal to name the
# version. A bump nobody needs makes two adapters disagree; a bump nobody made makes them
# disagree silently, which is worse.
ENVELOPE_VERSIONS: dict[type, str] = {
    SearchResponse: "2",
    EvidenceBundle: "2",
    GraphExpansionResponse: "1",
}


@pytest.mark.parametrize("model", [SearchResponse, EvidenceBundle, GraphExpansionResponse])
def test_every_response_declares_its_schema_version(model) -> None:
    """Spec §7.1: every contract carries `schema_version`, and an incompatible change needs
    a new version rather than a silent mutation. Seen red on `9dfa34e`: `EvidenceBundle`
    still said "1" over a shape its own version-1 model refuses."""
    assert model.model_fields["schema_version"].default == ENVELOPE_VERSIONS[model]
    assert (
        getattr(__import__("xbrain.knowledge.contracts", fromlist=["x"]), "EVIDENCE_SCHEMA_VERSION")
        == ENVELOPE_VERSIONS[EvidenceBundle]
    )


def _bundle_document() -> dict:
    """A bundle as `get --query` emits it: one chunk — serialised.

    FRONTIER ADAPTATION, declared (child PR 02.1). The snapshot builds this chunk with
    `locator=Locator(kind="item_text", char_start=0, char_end=6)`, because
    `KnowledgeChunk.locator` is REQUIRED there. That field arrives with
    `chunking.fragment_locator`, which computes it, in child PR 02.5 — porting it here would
    drag `chunking.py` into a PR whose one subject is the version of the envelopes. The
    version half of the transition is what 02.1 owns and it is ported byte for byte; the
    locator half of `test_a_version_one_bundle_is_refused_by_its_version_not_by_a_field`
    is deferred to 02.5 with it.
    """
    from xbrain.knowledge.models import KnowledgeChunk

    chunk = KnowledgeChunk(
        chunk_id="item:1:post:0:0:v2",
        surface_id="item:1:post:0",
        owner_type="item",
        owner_id="1",
        surface_type="post",
        text="cuerpo",
        chunk_index=0,
        char_start=0,
        char_end=6,
        origin="source",
        trust_class="primary_source",
        derived=False,
        fingerprint="a" * 64,
    )
    bundle = EvidenceBundle(
        item=KnowledgeItem(
            item_id="1",
            source="bookmark",
            url="https://x.com/a/status/1",
            author=Author(handle="a", name="A"),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            captured_at=datetime(2026, 1, 2, tzinfo=UTC),
        ),
        chunks=(chunk,),
    )
    return bundle.model_dump(mode="json")


def test_a_version_one_bundle_is_refused_by_its_version_not_by_a_field() -> None:
    """The transition is EXPLICIT (U-1): a document that announces `schema_version: "1"`
    — the shape whose chunks had no locator — is refused because it is version 1, and the
    refusal names the version. A document at "2" whose chunk lacks the locator is refused
    too, because the locator is what "2" means (spec §3.7 invariant 2). Neither is a
    silent acceptance and neither is a silent mutation.

    Seen red on `9dfa34e`: the version-1 document validated (same literal), and only the
    missing key was refused — a version-1 consumer and a version-2 producer under one number.

    FRONTIER ADAPTATION, declared (child PR 02.1): the second half of the snapshot's
    assertion — *a document at "2" whose chunk lacks the locator is refused too* — needs
    `KnowledgeChunk.locator` to be required, which lands in 02.5 with the
    `chunking.fragment_locator` that fills it. Ported here: the half that is about the
    VERSION, which is this PR's whole subject. The bundle at "2" therefore announces, for
    the 02.1..02.4 window, a shape whose chunk locator is not yet required — declared in
    the PR description, closed by 02.5.
    """
    document = _bundle_document()
    assert document["schema_version"] == "2"

    legacy = {**document, "schema_version": "1"}
    with pytest.raises(ValidationError) as refused:
        EvidenceBundle.model_validate(legacy)
    assert {error["loc"] for error in refused.value.errors()} == {("schema_version",)}


def test_the_search_envelope_moved_when_a_key_was_added_to_the_shape_it_transports() -> None:
    """The version policy applied to `SearchMatch.title` (B2, gate Codex on `b61e04b`).

    Round 06's bundle bump did NOT move this envelope, and the reason was stated as a fact
    about the shape: *`SearchMatch` carried `locator` from the Plan 01 freeze, so the search
    envelope's shape did not change and its version must not.* Spec §4's title changes the
    shape, so the same rule now points the other way — *a key added to a frozen shape BUMPS
    the version of every envelope that transports it* — and `SearchResponse` is the one
    envelope that transports a `SearchMatch`.

    Optional-with-a-default does not exempt it. Every model here is `extra="forbid"`, so it
    is the version-1 CONSUMER that breaks: it refuses a document carrying `title` outright,
    and the refusal has to name the version rather than a field nobody told it about. That
    is exactly the U-1 case, running in the other direction.
    """
    assert "locator" in SearchMatch.model_fields
    assert SearchMatch.model_fields["locator"].is_required()
    assert SearchResponse.model_fields["schema_version"].default == "2"
    assert SEARCH_SCHEMA_VERSION == "2"

    document = SearchResponse(
        query="q",
        strategy="lexical",
        filters=SearchFilters(),
        index=IndexStatusRef(manifest_version="1", built_at=datetime(2026, 1, 1, tzinfo=UTC)),
        results=(
            SearchResult(
                rank=1,
                item_id="1",
                url="https://x.com/a/status/1",
                author=Author(handle="a", name="A"),
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                matches=(
                    SearchMatch(
                        chunk_id="c",
                        surface_type="external_article",
                        origin="source",
                        trust_class="primary_source",
                        derived=False,
                        excerpt="x",
                        title="On Controls and Thresholds",
                        locator=Locator(kind="content_source", char_start=0, char_end=1),
                    ),
                ),
            ),
        ),
    ).model_dump(mode="json")
    assert document["schema_version"] == "2"
    assert document["results"][0]["matches"][0]["title"] == "On Controls and Thresholds"

    legacy = {**document, "schema_version": "1"}
    with pytest.raises(ValidationError) as refused:
        SearchResponse.model_validate(legacy)
    assert {error["loc"] for error in refused.value.errors()} == {("schema_version",)}


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
