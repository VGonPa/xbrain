# tests/test_knowledge_render.py
"""The human rendering (Plan 02 §6, step 27, spec §7.6).

SPEC §7.6 ENUMERATES WHAT THE HUMAN VIEW MUST ALWAYS SHOW, so the tests enumerate it too:
item, author, date and URL; surface and origin; excerpt; the channels that produced the
match; staleness and truncation warnings; and how to obtain the source with `xbrain get`.

The last one is CLAUDE.md rule 7 at its cheapest. Before building an instrument to detect a
defect, ask whether SHOWING the evidence makes it self-evident — and a line that names the
exact `xbrain get` command turns "this summary claims X" into "here is how to read what it
summarised", in one copy-paste.

EVERY ASSERTION IS ON A LABEL OR A COMMAND, never on a substring that could be satisfied by
a heading. CLAUDE.md rule 1's first row is exactly that mistake: `assert "NOT fetched" in
source` passed on the section header while the rule sentence it claimed to pin was
unprotected.
"""

from __future__ import annotations

from datetime import datetime, timezone

from xbrain.knowledge.contracts import (
    EvidenceBundle,
    IndexStatusRef,
    SearchFilters,
    SearchMatch,
    SearchResponse,
    SearchResult,
)
from xbrain.knowledge.models import (
    DerivedText,
    KnowledgeItem,
    Locator,
    SourceFailure,
    UnfetchedLink,
)
from xbrain.knowledge.render import render_get, render_search
from xbrain.models import Author

AUTHOR = Author(handle="karpathy", name="Andrej Karpathy")
WHEN = datetime(2026, 3, 4, 12, 0, tzinfo=timezone.utc)


def _match(surface_type="external_article", origin="source", trust="primary_source", derived=False):
    return SearchMatch(
        chunk_id="item:1:external_article:abc:0:v1",
        surface_type=surface_type,
        origin=origin,
        trust_class=trust,
        derived=derived,
        excerpt="El párrafo que coincidió con la consulta.",
        matched_by=("lexical",),
        lexical_rank=1,
        score=-3.2,
        locator=Locator(kind="content_source", char_start=0, char_end=40),
    )


def _result(**kwargs) -> SearchResult:
    fields = {
        "rank": 1,
        "item_id": "1884",
        "url": "https://x.com/karpathy/status/1884",
        "author": AUTHOR,
        "created_at": WHEN,
        "summary": DerivedText(text="Un resumen generado.", origin="llm"),
        "topics": ("agent-evaluation",),
        "matches": (_match(),),
        "available_surfaces": ("post", "external_article", "summary"),
        "verify_with": ("external_article",),
    }
    fields.update(kwargs)
    return SearchResult(**fields)


def _response(**kwargs) -> SearchResponse:
    fields = {
        "query": "evaluación de agentes",
        "strategy": "lexical",
        "filters": SearchFilters(),
        "index": IndexStatusRef(manifest_version="1", built_at=WHEN, degraded=("no_embeddings",)),
        "results": (_result(),),
    }
    fields.update(kwargs)
    return SearchResponse(**fields)


# ---------------------------------------------------------------------------
# 27 — everything spec §7.6 requires
# ---------------------------------------------------------------------------


def test_the_search_rendering_shows_every_field_the_spec_requires() -> None:
    """Spec §7.6, item by item. Seen red by deleting any one of the lines it names."""
    text = render_search(_response())
    assert "1884" in text
    assert "@karpathy (Andrej Karpathy)" in text
    assert "2026-03-04" in text
    assert "https://x.com/karpathy/status/1884" in text
    assert "[external_article]" in text, "the surface must be labelled, not implied"
    assert "origin=source" in text
    assert "El párrafo que coincidió" in text
    assert "via lexical" in text, "the channel that produced the match"
    assert "agent-evaluation" in text


def test_the_summary_is_shown_WITH_its_origin() -> None:
    """Invariant 2 of spec §3.7: no text is presented without the provenance qualifying it.

    Asserted on the pairing — the word `resumen` and `(llm)` on the SAME line — because a
    test that only checked both appear somewhere would pass with the origin printed at the
    bottom of the screen, which is not the same claim.
    """
    line = next(line for line in render_search(_response()).splitlines() if "resumen" in line)
    assert "(llm)" in line
    assert "Un resumen generado." in line


def test_a_verified_summary_shows_its_verdict() -> None:
    """M5: the verdict is hydrated from the live store, and the reader sees it beside the text."""
    result = _result(
        summary=DerivedText(text="Un resumen.", origin="llm", verification_status="FAIL")
    )
    line = next(
        line
        for line in render_search(_response(results=(result,))).splitlines()
        if "resumen" in line
    )
    assert "[FAIL]" in line


def test_the_rendering_names_the_exact_get_command() -> None:
    """Step 27 / rule 7: the cheapest verification layer is showing the reader the evidence.

    A COMMAND, not advice: the line is copy-pasteable, with the item id and the surface
    already filled in. Seen red by printing "usa xbrain get" without the arguments.
    """
    text = render_search(_response())
    assert "xbrain get 1884 --surface external_article" in text


def test_a_derived_match_with_no_source_prints_the_warning() -> None:
    """Step 19 in the human view: `no_underlying_source` becomes a WORD.

    The frozen `SearchResult` has no `warnings` field, so the JSON says it structurally
    (`verify_with: []`, reachable in exactly one case) and the human view says it here.
    Seen red by omitting the branch: an item with no source renders identically to one whose
    source simply was not needed.
    """
    result = _result(
        matches=(
            _match(surface_type="summary", origin="llm", trust="llm_synthesis", derived=True),
        ),
        verify_with=(),
        available_surfaces=("summary",),
    )
    text = render_search(_response(results=(result,)))
    assert "no_underlying_source" in text
    assert "ninguna fuente primaria" in text


def test_a_behind_index_warns_and_names_the_command() -> None:
    """Step 10b in the human view (B3): the degradation is a sentence, not a flag.

    A flag a reader has to look up is a flag they ignore, so the rendering says what happened
    AND what fixes it. Asserted on the command, because that is the part that cannot be
    satisfied by a generic warning banner.
    """
    response = _response(
        index=IndexStatusRef(
            manifest_version="1",
            built_at=WHEN,
            degraded=("index_behind_store", "no_embeddings"),
        )
    )
    text = render_search(response)
    assert "xbrain index update" in text
    assert "obsoleta" in text


def test_the_degradation_warning_comes_before_the_results() -> None:
    """A warning under the fold is a warning nobody read.

    A reader who stops at the first result must already have seen that the evidence may be
    stale, so the position is part of the requirement and is asserted as such.
    """
    response = _response(
        index=IndexStatusRef(manifest_version="1", built_at=WHEN, degraded=("index_behind_store",))
    )
    lines = render_search(response).splitlines()
    warning = next(i for i, line in enumerate(lines) if "index update" in line)
    first_result = next(i for i, line in enumerate(lines) if "1884" in line)
    assert warning < first_result


def test_excluded_chunks_are_reported_with_the_repair() -> None:
    """Step 10 in the human view: an exclusion nobody sees is a corpus that shrank in silence."""
    response = _response(
        index=IndexStatusRef(manifest_version="1", built_at=WHEN, corrupt_chunks_excluded=3)
    )
    text = render_search(response)
    assert "3 chunk(s) excluido(s)" in text
    assert "index build --force" in text


def test_a_profile_only_candidate_says_it_has_no_citable_fragment() -> None:
    """Spec §5.1.A: the profile is a retrieval representation, NEVER returned as a citation.

    So a result with no matches must say why there is no excerpt rather than render an empty
    block — and it must certainly not invent one from the profile, which is a string nobody
    wrote.
    """
    text = render_search(_response(results=(_result(matches=(), verify_with=("post",)),)))
    assert "perfil del item" in text
    assert "no hay fragmento citable" in text


def test_an_empty_result_set_says_so_without_claiming_anything() -> None:
    """Zero results is a legitimate answer; it just must not look like an error."""
    text = render_search(_response(results=()))
    assert "Sin resultados" in text


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


def _bundle(**kwargs) -> EvidenceBundle:
    item = KnowledgeItem(
        item_id="1884",
        source="bookmark",
        url="https://x.com/karpathy/status/1884",
        author=AUTHOR,
        created_at=WHEN,
        captured_at=WHEN,
        topics=("agent-evaluation",),
        available_surfaces=("post", "external_article"),
        failed_sources=(
            SourceFailure(
                kind="external_article", url="https://dead.example", failure_reason="not_found"
            ),
        ),
        unfetched_links=(
            UnfetchedLink(url="https://nope.example", reason="http_error", detail="404"),
        ),
    )
    # BOTH levels, because the contract declares both and `get_service` fills both from the
    # same projection. `EvidenceBundle.failures` and `KnowledgeItem.failed_sources` are a
    # redundancy Plan 01 froze; keeping the fixture faithful to what the service emits is
    # what stops this test passing against a shape the service never produces.
    fields = {
        "item": item,
        "failures": item.failed_sources,
        "unfetched_links": item.unfetched_links,
    }
    fields.update(kwargs)
    return EvidenceBundle(**fields)


def test_the_get_rendering_shows_failures_and_unfetched_links_apart() -> None:
    """Step 25 / 25b (m7): two facts, two lines, two vocabularies.

    "We tried and it failed" and "there is no body for this URL" are different, and collapsing
    them makes a link nobody attempted indistinguishable from one that returned a 404.
    """
    text = render_get(_bundle())
    assert "fetch falló: external_article https://dead.example (not_found)" in text
    assert "sin cuerpo: https://nope.example (http_error) — 404" in text


def test_the_get_rendering_offers_the_cursor_when_truncated() -> None:
    """Spec §9.3: a truncated answer says how to continue, never cuts in silence."""
    text = render_get(_bundle(truncated=True, cursor="0:3"))
    assert "xbrain get 1884 --cursor 0:3" in text


def test_the_get_rendering_lists_the_surfaces_you_can_ask_for() -> None:
    """The body is withheld by default, so the NAMES are what make it reachable."""
    text = render_get(_bundle())
    assert "superficies disponibles: post, external_article" in text
