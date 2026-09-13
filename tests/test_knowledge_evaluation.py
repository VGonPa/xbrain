# tests/test_knowledge_evaluation.py
"""The evaluation harness (Plan 01 §5, steps 26-28).

WHAT THIS MEASURES, AND THE THREE RULES IT ENFORCES.

1. **Never one global figure.** Spec §8.4 requires metrics by strategy x stratum x
   provenance. A single corpus-wide recall averages a stratum with 9 cases against one with
   2 and reports a number no decision can be made from — and, worse, hides the case where
   the semantic layer helps exactly one stratum.

2. **A stratum with no cases is reported WITHOUT COVERAGE, never as 0.0.** Spec §8.6.8:
   *failures and skips are published; zeros are never fabricated by mixing in unmeasured
   cases*. `expansion` has no mechanism until Plan 04 and `thread`/`user_note` have no data
   at all; showing them at 0.0 would say the retriever failed where nothing was asked.

3. **Report-only.** The harness never writes to `items.json` and never snapshots, exactly
   like `verify` by default and like `cv-guardrail`. Asserted by hashing the file.

AND IT MUST BE ABLE TO FAIL (acceptance 10). A harness that cannot go red is a decoration,
so `test_the_evaluation_can_fail` drives it to a failing verdict over the fixture corpus.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from xbrain.knowledge import contracts, evaluation
from xbrain.knowledge.contracts import SearchFilters
from xbrain.knowledge.evaluation import (
    EvaluationReport,
    NO_COVERAGE,
    build_index,
    evaluate,
    load_corpus,
    render_markdown,
)
from xbrain.knowledge.goldenset import load_cases, load_scenarios, resolve_cases

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_GOLDEN = FIXTURES / "knowledge_goldenset.yaml"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(FIXTURES / "knowledge_corpus.json")


@pytest.fixture(scope="module")
def report(corpus) -> EvaluationReport:
    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    return evaluate(cases, corpus, strategy="lexical", scenarios=load_scenarios(FIXTURE_GOLDEN))


# ---------------------------------------------------------------------------
# 26 — never a global figure
# ---------------------------------------------------------------------------


def test_metrics_are_reported_per_stratum_and_provenance(report) -> None:
    """Seen red by asking for a flat `report["recall@10"]`.

    Spec §8.4 asks for strategy x stratum x provenance. The shape of the report is what
    enforces it: there is no top-level metric key to reach for.
    """
    payload = report.to_dict()
    assert "recall@10" not in payload
    assert set(payload["by_stratum"]) and set(payload["by_provenance"])
    for bucket in payload["by_stratum"].values():
        if bucket != NO_COVERAGE:
            assert {"recall@1", "recall@10", "precision@10", "mrr", "cases"} <= set(bucket)


def test_every_k_the_spec_asks_for_is_present(report) -> None:
    """k in {1, 5, 10, 20} (Plan 01 §5.2).

    Reporting only recall@10 hides the two failure shapes that matter: a retriever that
    finds everything but ranks it 9th, and one that nails the top hit and misses the tail.
    """
    bucket = next(b for b in report.to_dict()["by_stratum"].values() if b != NO_COVERAGE)
    assert {f"recall@{k}" for k in (1, 5, 10, 20)} <= set(bucket)


def test_surface_recall_is_reported_beside_item_recall(report) -> None:
    """Spec §8.4: *surface recall, in addition to items*.

    Returning the right item through the wrong surface is a different and usually worse
    answer: the evidence a consumer would open is not the evidence the fact is in. A report
    with item recall only would score that as a success.
    """
    payload = report.to_dict()
    assert any("surface_recall@10" in b for b in payload["by_stratum"].values() if b != NO_COVERAGE)


def test_latency_percentiles_are_recorded(report) -> None:
    """p50/p95 (spec §8.4), because a retriever that is right and slow is a different tool."""
    assert {"p50_ms", "p95_ms"} <= set(report.to_dict()["latency"])


# ---------------------------------------------------------------------------
# 27 — no coverage is not zero
# ---------------------------------------------------------------------------


def test_a_stratum_with_no_cases_is_reported_without_coverage(report) -> None:
    """Spec §8.6.8, the rule this whole harness is judged by.

    `expansion` has no mechanism until Plan 04 exists. Reporting it as recall 0.0 would say
    the retriever failed at something nobody asked it to do, and the figure would sit in a
    table looking exactly like a measurement. Seen red by defaulting the bucket to zeros.
    """
    payload = report.to_dict()
    assert payload["by_stratum"]["expansion"] == NO_COVERAGE
    assert "expansion" in payload["without_coverage"]["strata"]


def test_surfaces_with_no_data_are_declared_not_scored(report) -> None:
    """`thread` and `user_note` have ZERO instances in the real corpus (measured 2026-08-31).

    The emitter supports them; the evaluation cannot have cases for them and does not invent
    any. Declaring them is what stops their absence reading as an oversight later.
    """
    declared = report.to_dict()["without_coverage"]["surfaces"]
    assert {"thread", "user_note"} <= set(declared)


def test_archived_scenarios_are_listed_with_their_reason_and_never_scored(report) -> None:
    """A scenario is not a case scoring zero — it is a case that does not score.

    Listing it with its reason is what spec §8.1 means by "not omitted in silence", and it
    is what lets someone later see what enumerating it would take.
    """
    payload = report.to_dict()
    assert payload["scenarios"], "the fixture golden set has archived scenarios"
    assert all(entry["reason"] for entry in payload["scenarios"])
    scored_ids = {case["id"] for case in payload["cases"]}
    assert scored_ids.isdisjoint({entry["id"] for entry in payload["scenarios"]})


# ---------------------------------------------------------------------------
# 28 — report only
# ---------------------------------------------------------------------------


def test_the_harness_never_writes_to_the_store(tmp_path: Path, corpus) -> None:
    """Hash before, hash after. `verify` and `cv-guardrail` set the same precedent.

    An evaluation that could mutate the corpus it measures is an evaluation whose next run
    measures its own side effects.
    """
    store_path = tmp_path / "items.json"
    store_path.write_text(
        (FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    before = hashlib.sha256(store_path.read_bytes()).hexdigest()
    before_mtime = store_path.stat().st_mtime_ns
    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    evaluate(cases, corpus, strategy="lexical")
    assert hashlib.sha256(store_path.read_bytes()).hexdigest() == before
    assert store_path.stat().st_mtime_ns == before_mtime


# ---------------------------------------------------------------------------
# Acceptance 10 — the evaluation MUST be able to fail
# ---------------------------------------------------------------------------


def test_the_evaluation_can_fail(corpus) -> None:
    """The gate B1 bought: this runs in CI, over the FIXTURE corpus.

    A harness nobody has seen go red is an assurance, not a gate. Driving `k` down to 1 on a
    stratum whose relevant set has more than one item makes recall@1 fall below 1.0 by
    construction — and the verdict follows the numbers rather than being asserted.
    """
    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    generous = evaluate(cases, corpus, strategy="lexical", ks=(20,), threshold=0.5)
    assert generous.passed, "the baseline must clear an easy bar, or the fixture is broken"

    strict = evaluate(cases, corpus, strategy="lexical", ks=(1,), threshold=1.0)
    assert not strict.passed
    assert strict.failures, "a failing report must name WHICH buckets failed"


def test_a_failing_report_names_the_bucket_that_failed(corpus) -> None:
    """ "It failed" is not actionable; "recall@1 in `semantico` was 0.5" is."""
    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    strict = evaluate(cases, corpus, strategy="lexical", ks=(1,), threshold=1.0)
    assert any(":" in failure for failure in strict.failures)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_the_report_serialises_to_json_and_markdown(report) -> None:
    payload = json.dumps(report.to_dict(), ensure_ascii=False)
    assert json.loads(payload)["strategy"] == "lexical"
    markdown = render_markdown(report)
    assert "sin cobertura" in markdown
    assert "expansion" in markdown


def test_the_report_records_the_population_it_measured(report) -> None:
    """CLAUDE.md rule 2: state the population you measured ON.

    A recall figure with no item count beside it cannot be compared with the next run, and
    is exactly the kind of number that gets quoted after the corpus has moved underneath it.
    """
    payload = report.to_dict()
    assert payload["corpus"]["items"] == 12
    assert payload["corpus"]["chunks"] > 0
    assert payload["corpus"]["source"].endswith("knowledge_corpus.json")


def test_building_the_index_reports_what_it_skipped(corpus) -> None:
    """Coverage of the indexed corpus (spec §8.4, last bullet).

    An index that quietly dropped every article would still score well on the post-only
    cases, and only the coverage line would say why the rest went missing.
    """
    index, stats = build_index(corpus)
    assert stats.items == 12
    assert stats.surfaces >= stats.items
    assert stats.chunks == len(index) + stats.chunks_not_indexed


# ---------------------------------------------------------------------------
# A filter the strategy cannot apply is NOT a case the retriever failed
# ---------------------------------------------------------------------------


STUB_BACKEND = "stub_backend_that_pushes_no_filter"


@pytest.fixture()
def stub_backend(monkeypatch):
    """A HYPOTHETICAL retrieval backend that exists and can push no filter at all.

    Injected rather than borrowed from the frozen `Strategy` literal (F-2). The previous
    version of the guardrail below drove the branch with `strategy="lexical"`, which made the
    test depend on the baseline staying unable to push six of the eight filters: once the
    harness builds through `index_build`'s writer, `lexical` pushes all eight and the
    guardrail would go green because nothing is unsupported any more — a test of nothing
    (rule 1). Borrowing `vector` instead is the same mistake one level up: it would make this
    guardrail's survival depend on Plan 03 not happening.

    Two injections because two facts are being supposed, and they are genuinely different
    facts: that the backend EXISTS (`IMPLEMENTED_STRATEGIES`, or `resolve_strategy` would
    degrade it to `lexical` and the filters would all be pushed after all), and that it can
    push NO filter (`SUPPORTED_FILTERS`).
    """
    monkeypatch.setattr(contracts, "IMPLEMENTED_STRATEGIES", frozenset({"lexical", STUB_BACKEND}))
    monkeypatch.setitem(evaluation.SUPPORTED_FILTERS, STUB_BACKEND, frozenset())
    return STUB_BACKEND


def test_a_case_whose_filters_the_strategy_cannot_apply_is_not_scored(
    corpus, stub_backend: str
) -> None:
    """The fabricated zero this harness exists to prevent — the MECHANISM, still guarded.

    Scoring a case whose filter nobody applied produced `filtros: recall@10 = 0.0` in this
    harness's first real-corpus run, which reads as "the retriever failed at filtering" when
    the truth was that the instrument did not exist yet. Spec §8.6.8: *failures and skips are
    published; zeros are never fabricated by mixing in unmeasured cases.*

    THE STRATEGY IS INJECTED, and that is the point. Plan 02 gave the lexical baseline all
    eight filters, so `lexical` can no longer demonstrate this branch — driving it with
    `lexical` would leave a test that passes because nothing is unsupported, which is a test
    of nothing (rule 1).

    Seen red by giving the stub every filter (`frozenset(SearchFilters.model_fields)`): FX7
    is scored, `unmeasured` is empty and `filtros` publishes a number.
    """
    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    payload = evaluate(cases, corpus, strategy=stub_backend).to_dict()

    unmeasured = {entry["id"]: entry for entry in payload["unmeasured"]}
    assert "FX7" in unmeasured, "FX7 declares `source: own_tweet`, which the stub cannot apply"
    assert unmeasured["FX7"]["unsupported_filters"] == ["source"]
    assert payload["by_stratum"]["filtros"] == NO_COVERAGE
    assert "FX7" not in {case["id"] for case in payload["cases"]}
    assert payload["strategy"] == stub_backend, "the stub RAN; nothing was degraded"


def test_the_harness_scores_the_index_the_writer_produces_not_a_second_walk(corpus) -> None:
    """Rule 5, in the one place it decides whether six of the eight filters exist at all.

    `build_index` used to walk the corpus its own way — `corpus_chunks(...)` then
    `index.add(chunks)` — which fills `chunks` and NOTHING else. The `items` table stayed
    empty, so every item-scoped clause (`source`, `author`, `created_from/to`,
    `content_kinds`) matched no row, and a case filtered on one of them could only ever score
    zero. Two walks that "should" emit the same corpus, and the half that went missing is
    exactly the half `SUPPORTED_FILTERS` now promises.

    Asserted through BEHAVIOUR rather than by `assert build_index is write_item`'s caller:
    the question is whether the metadata is queryable, not whether a name was imported.

    Seen red by restoring the two-walk body: `items` holds 0 rows and both filters below
    return 0 owners, so the two `source` values become indistinguishable.
    """
    index, stats = build_index(corpus)
    try:
        rows = index.connection.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        by_source = {
            source: {
                (hit.owner_type, hit.owner_id)
                for hit in index.search("the", 50, filters=SearchFilters(source=source))
            }
            for source in ("bookmark", "own_tweet")
        }
    finally:
        index.connection.close()

    assert rows == stats.items, "the writer's item metadata reached the harness's index"
    assert by_source["bookmark"] and by_source["own_tweet"], "both sides of the filter answer"
    assert not (by_source["bookmark"] & by_source["own_tweet"]), "and they are disjoint"


def test_a_real_vector_backend_reports_the_filter_cases_unmeasured_never_scored(
    tmp_path: Path, corpus
) -> None:
    """The day F-2's guardrail was waiting for, and what it turns out to mean (Plan 03.7).

    The test this replaces SIMULATED a vector backend that pushed all eight filters, and
    asserted the filter cases would then be scored. The real backend is the opposite: the
    vector plane has no filter columns, and a filter applied after scoring is not a filter
    (`search_service.VECTOR_FILTERS_UNSUPPORTED`). So a filtered case under `vector` is the
    exact population CLAUDE.md names — *a case whose filters a strategy cannot apply is
    UNMEASURED, never 0.0* — and it must reach `unmeasured`, never the `filtros` mean.

    Seen red by declaring every filter for `vector` in `SUPPORTED_FILTERS`: FX7 is scored and
    `filtros` publishes a number the vector channel never filtered for.
    """
    data = _vector_workspace(tmp_path, corpus)
    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    payload = evaluate(
        cases, corpus, strategy="vector", vectors=_vectors(data, tmp_path / "eval-index")
    ).to_dict()

    assert payload["strategy"] == "vector" and payload["degraded"] == []
    unmeasured = {entry["id"]: entry for entry in payload["unmeasured"]}
    assert unmeasured["FX7"]["unsupported_filters"] == ["source"]
    assert payload["by_stratum"]["filtros"] == NO_COVERAGE
    assert "FX7" not in {case["id"] for case in payload["cases"]}


def test_the_guardrail_no_longer_depends_on_vector_being_unimplemented(
    tmp_path: Path, corpus, monkeypatch
) -> None:
    """Now that `vector` RUNS, its filter cases are unmeasured because of what it DECLARES it
    can push — never because it is missing from `IMPLEMENTED_STRATEGIES`.

    WHY THIS EXACT NAME. It is an acceptance node of criterion 11 in
    `eval/plan02-acceptance.yaml`, and `tests/test_plan02_acceptance.py` fails closed on a node
    id that stops resolving. Plan 03.7 replaced the test of this name with the real-backend
    test above (a simulated vector pushing eight filters became a real one pushing none) and
    the node went with it. That test stays as it is; this one is not an alias of it.

    WHAT IT PINS THAT NOTHING ELSE DID. «Unimplemented» is still literally true in production:
    `vector` runs only through `vectors=` and is NOT in `contracts.IMPLEMENTED_STRATEGIES`. An
    `unsupported_filters` that read the gap off that membership instead of `SUPPORTED_FILTERS`
    gives byte-identical reports today — the vector pair declares no filter either way — and
    every test of the knowledge files that reach the harness stayed green under it (257,
    measured). The coupling shows only when the DECLARATION moves while the STATUS stays put,
    so that is what this does, over a real vector backend that ran (hash embedder, no model):
    as shipped the guardrail fires on a strategy that was not degraded; with `vector` declared
    able to push every filter, the same case is scored. That score is meaningless — the plane
    filtered nothing, which is why production declares nothing — and is not asserted; what is
    asserted is WHERE the decision came from. The status is PINNED, not inherited, so the day
    `vector` joins `IMPLEMENTED_STRATEGIES` this test does not quietly stop discriminating.

    Seen red both ways, by mutating `unsupported_filters` and restoring it: returning `()` for
    any runnable strategy (the guardrail lives only while vector cannot run) fails the first
    half; `supported = frozenset()` unless the strategy is in `IMPLEMENTED_STRATEGIES` fails
    the second.
    """
    monkeypatch.setattr(contracts, "IMPLEMENTED_STRATEGIES", frozenset({"lexical"}))
    data = _vector_workspace(tmp_path, corpus)
    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    index_dir = tmp_path / "eval-index"

    shipped = evaluate(
        cases, corpus, strategy="vector", vectors=_vectors(data, index_dir)
    ).to_dict()
    assert shipped["strategy"] == "vector" and shipped["degraded"] == [], "vector RAN"
    unmeasured = {entry["id"]: entry for entry in shipped["unmeasured"]}
    assert unmeasured["FX7"]["unsupported_filters"] == ["source"]

    monkeypatch.setitem(
        evaluation.SUPPORTED_FILTERS, "vector", frozenset(SearchFilters.model_fields)
    )
    declared = evaluate(
        cases, corpus, strategy="vector", vectors=_vectors(data, index_dir)
    ).to_dict()
    assert declared["strategy"] == "vector" and declared["degraded"] == [], "same status"
    assert declared["unmeasured"] == [], "the declaration decided, not the status"
    assert "FX7" in {case["id"] for case in declared["cases"]}


def test_an_unimplemented_strategy_publishes_the_strategy_that_actually_ran(
    corpus, monkeypatch
) -> None:
    """F-2 at the harness: `xbrain eval --strategy vector` published `vector`, scored by bm25.

    THE PREMISE IS PINNED, NOT INHERITED (M-2, round 02): `vector` is the example of a
    declared-but-unimplemented strategy, and borrowing that fact from production made the
    test expire the day Plan 03 lands — the same coupling F-2 removed from the guardrail.
    Simulated with `vector` added to `IMPLEMENTED_STRATEGIES`: red before the pin
    (`payload["strategy"] == "vector"`), green with it.

    21 cases, `recall@10 = 0.8099`, under a heading that named a retriever which does not
    exist. That is the metric whose label does not describe its instrument — rule 2, and spec
    §8.6.8's fabricated number wearing a different costume.

    The report now names the strategy that RAN and declares the one that could not, in the
    JSON and in the markdown heading, so no reader can take the numbers for vector's.

    Seen red before the fix: `payload["strategy"]` came back `"vector"` and the markdown
    heading named `vector` as though a vector retriever had produced the numbers.
    """
    monkeypatch.setattr(contracts, "IMPLEMENTED_STRATEGIES", frozenset({"lexical"}))
    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    report = evaluate(cases, corpus, strategy="vector")
    payload = report.to_dict()

    assert payload["strategy"] == "lexical", "what ran"
    assert payload["requested_strategy"] == "vector", "what was asked for"
    assert payload["degraded"] == ["vector_not_implemented"]
    assert payload["cases"], "spec §9.3: lexical stays operational"

    markdown = render_markdown(report)
    assert "`lexical`" in markdown.splitlines()[0]
    assert "vector" in markdown.splitlines()[0], "the request is not hidden either"


def test_every_implemented_strategy_declares_which_filters_it_can_push() -> None:
    """Rule 5: the two tables that must agree are asserted to agree, not hoped to.

    `IMPLEMENTED_STRATEGIES` says which retrievers run without vectors, and the search
    service's own vector set says which run with them (Plan 03.5); `SUPPORTED_FILTERS` says
    what each can push into `WHERE`. A strategy runnable without an entry here would fall to
    `SUPPORTED_FILTERS.get(strategy, frozenset())` — right today for the vector pair by
    accident, and wrong the day one of them gains filter columns. Read off `search_service`
    and not off `evaluation`, so the binding is between two modules rather than one module
    and itself (rule 1, row 4).
    """
    from xbrain.knowledge import search_service

    runnable = set(contracts.IMPLEMENTED_STRATEGIES) | set(search_service._VECTOR_STRATEGIES)
    assert set(evaluation.SUPPORTED_FILTERS) == runnable


def test_the_lexical_strategy_now_scores_the_filter_stratum(corpus) -> None:
    """WHAT PLAN 02 CHANGED, asserted rather than described.

    Under Plan 01 the baseline had no date, source or content-kind column, so every case in
    the `filtros` stratum was reported UNMEASURED and the stratum carried `NO_COVERAGE`. The
    persisted schema has all eight columns and the harness builds through the SAME writer as
    `index build`, so those cases are measurable — and the stratum publishes a number for the
    first time.

    Seen red by reverting `SUPPORTED_FILTERS` to `{has_surfaces, origins}`: `filtros` goes
    back to `NO_COVERAGE` and FX7 back to the unmeasured list.
    """
    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    payload = evaluate(cases, corpus, strategy="lexical").to_dict()

    assert payload["unmeasured"] == [], "no case is unmeasurable for lexical any more"
    assert "FX7" in {case["id"] for case in payload["cases"]}
    assert payload["by_stratum"]["filtros"] != NO_COVERAGE


# One DECLARED value per field of the frozen contract, each valid for its own type. Written
# as a total map and not as a `dict.get(..., None)` with a `continue`: the loop below has to
# exercise every field, and a default that means "skip" turns an eight-field assertion into a
# three-field one silently. Extending `SearchFilters` without extending this map is caught by
# the coverage assertion inside the test, not by a reader remembering.
_DECLARED_FILTER_VALUES: dict[str, object] = {
    "created_from": datetime(2025, 10, 1),
    "created_to": datetime(2025, 10, 31),
    "source": "own_tweet",
    "author": "x",
    "topics": ("agentes",),
    "content_kinds": ("x_video",),
    "origins": ("source",),
    "has_surfaces": ("post",),
}


def test_the_supported_filter_set_is_DERIVED_from_the_frozen_contract() -> None:
    """All eight, and taken from `SearchFilters` rather than written out a second time.

    A hand-written list here would be a second copy of the contract, and the day a ninth
    filter is added to `SearchFilters` the copy would silently keep declaring eight — the
    case would be scored with a filter nobody applied, which is the fabricated zero coming
    back through the door marked "supported".

    IT ALSO SUBSUMES `test_supported_filters_are_still_applied_not_skipped`, which this
    replaces: that test asserted the two filters of the hand-written set were pushed and
    `source` was not. The loop below asserts the same property for EVERY field of the
    contract, so "unsupported" cannot become a way to quietly stop measuring anything
    awkward — and it says so over the derived set rather than over a copy of it.

    «EVERY FIELD» IS NOW TRUE. It was not: the sample value came from a two-entry `dict.get`
    that returned `None` for the other six, and `None` meant `continue`, so the loop asserted
    on THREE of the eight while its own docstring said eight — and since the assertion above
    already equates the two sets, the loop could not have failed independently of it anyway.
    A test that cannot fail on its own is rule 1's first row. The map is total, the
    `asserted` tally is checked against `model_fields`, and `unsupported_filters` is exercised
    on the value each field would really carry.
    """
    from xbrain.knowledge.evaluation import SUPPORTED_FILTERS, unsupported_filters

    assert SUPPORTED_FILTERS["lexical"] == frozenset(SearchFilters.model_fields)
    assert len(SUPPORTED_FILTERS["lexical"]) == 8

    asserted: set[str] = set()
    for name in SearchFilters.model_fields:
        value = _DECLARED_FILTER_VALUES[name]
        filters = SearchFilters(**{name: value})
        # The value must actually READ as declared, or the next assertion is vacuous: a filter
        # left at its default is never "declared", so a bad sample would make
        # `unsupported_filters` return `()` for the uninteresting reason.
        assert unsupported_filters(filters, "nonexistent-strategy") == (name,), (
            f"the sample for {name!r} does not read as a DECLARED filter"
        )
        assert unsupported_filters(filters, "lexical") == ()
        asserted.add(name)

    assert asserted == set(SearchFilters.model_fields), (
        "a field of the frozen contract was skipped instead of asserted: "
        f"{sorted(set(SearchFilters.model_fields) - asserted)}"
    )


def test_the_case_count_reconciles_including_the_unmeasured(corpus) -> None:
    """Scored + unmeasured + archived == every entry in the file.

    The first version of this reconciliation counted only scored + archived, and would have
    gone green while two cases vanished from the report entirely.
    """
    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    payload = evaluate(
        cases, corpus, strategy="lexical", scenarios=load_scenarios(FIXTURE_GOLDEN)
    ).to_dict()
    total = len(load_cases(FIXTURE_GOLDEN)) + len(load_scenarios(FIXTURE_GOLDEN))
    assert len(payload["cases"]) + len(payload["unmeasured"]) + len(payload["scenarios"]) == total


def test_the_retrieval_depth_never_falls_below_the_largest_reported_k(corpus) -> None:
    """`--limit 1` while reporting recall@20 would measure the LIMIT, not the retriever.

    A knob that can silently invalidate the metric beside it is worse than no knob: the
    report would show `recall@20` computed over at most one result, and the number would look
    exactly like a retrieval failure. So the depth is `max(limit, max(ks))` — raising it is
    allowed (it answers "is this a ranking problem or an absence?"), lowering it below the
    reported k is not.
    """
    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    clamped = evaluate(cases, corpus, ks=(20,), limit=1)
    natural = evaluate(cases, corpus, ks=(20,))
    assert clamped.to_dict()["by_stratum"] == natural.to_dict()["by_stratum"]


# ---------------------------------------------------------------------------
# B1 — a metric nobody could measure is NOT a zero (spec §8.6.8)
# ---------------------------------------------------------------------------
#
# The guardrail against the fabricated zero existed at BUCKET level (`_aggregate` returns
# `NO_COVERAGE` for a bucket with no cases) and nowhere at METRIC level. A case that names no
# surface is not a case that failed `surface_recall`: it is a case that did not measure it,
# and the 0.0 it contributed entered the stratum mean looking exactly like a measurement.
#
# The tests below were seen RED against the pre-fix tree (CLAUDE.md rule 1):
#   - the surface test:  0.5 != 1.0   (the unmeasured case halved the stratum mean)
#   - the rank-1 test:   0.0 is not None, and `assert recall is None` failed on 0.0
# They are written against the PUBLIC API (`evaluate`), not against `_score`, so they keep
# holding if the internals move.


def _case(**kwargs):
    """One golden case, with only the ground truth the test is about."""
    from xbrain.knowledge.contracts import SearchFilters
    from xbrain.knowledge.goldenset import GoldenCase

    kwargs.setdefault("provenance", "construido")
    kwargs.setdefault("filters", SearchFilters())
    return GoldenCase(**kwargs)


def _some_item(corpus) -> tuple[str, str]:
    """An `(item_id, query)` pair whose query retrieves that item at rank 1.

    Taken from the fixture corpus rather than invented, so the test exercises the real
    index rather than a mock that can agree with a broken scorer.
    """
    from xbrain.knowledge.surfaces import item_surfaces

    for item_id, item in corpus.items.items():
        surfaces = item_surfaces(item)
        if surfaces and len(surfaces[0].text.split()) > 4:
            return item_id, surfaces[0].text
    raise AssertionError("the fixture corpus has no usable item")


def test_a_case_that_names_no_surface_does_not_lower_the_stratum_surface_recall(corpus) -> None:
    """A case with `relevant_surfaces: ()` is UNMEASURED for surface recall, not failed.

    Seen red: the unmeasured case contributed a hard 0.0 and the stratum mean came out
    0.5 where the only case that measured anything scored 1.0. Measured on the real corpus
    this depressed `enterrado`'s published `surface_recall@10` from 0.1667 to 0.125.
    """
    from xbrain.knowledge.goldenset import RelevantSurface

    item_id, query = _some_item(corpus)
    surface = next(
        s
        for s in __import__("xbrain.knowledge.surfaces", fromlist=["item_surfaces"]).item_surfaces(
            corpus.items[item_id]
        )
    )
    measured = _case(
        id="WITH-SURFACE",
        query=query,
        strata=("exacto",),
        relevant_items=(item_id,),
        relevant_surfaces=(
            RelevantSurface(owner_type="item", owner_id=item_id, surface_type=surface.surface_type),
        ),
    )
    silent = _case(id="NO-SURFACE", query=query, strata=("exacto",), relevant_items=(item_id,))

    alone = evaluate([measured], corpus).by_stratum["exacto"]["surface_recall@10"]
    together = evaluate([measured, silent], corpus).by_stratum["exacto"]["surface_recall@10"]
    assert alone == together, (
        "a case that names no surface must not enter the surface_recall mean; "
        f"it moved {alone} -> {together}"
    )


def test_a_case_with_no_relevant_owner_reports_recall_as_unmeasured_not_zero(corpus) -> None:
    """The 0/0 case, scoring a RANK-1 hit — B1.b, the latent half.

    `load_cases` accepts a case whose ground truth is `relevant_surfaces` only, and for it
    `recall@k` is 0/0. Returned as a hard 0.0 that is the fabricated zero one level below
    the one plan §4.4 exists to kill: the retriever put the requested surface FIRST and the
    report said it recalled nothing.

    Seen red: `recall@1 == 0.0` and `mrr == 0.0` on a perfect rank-1 answer.
    """
    from xbrain.knowledge.goldenset import RelevantSurface
    from xbrain.knowledge.surfaces import item_surfaces

    item_id, query = _some_item(corpus)
    surface = item_surfaces(corpus.items[item_id])[0]
    case = _case(
        id="SURFACE-ONLY",
        query=query,
        strata=("exacto",),
        relevant_surfaces=(
            RelevantSurface(owner_type="item", owner_id=item_id, surface_type=surface.surface_type),
        ),
    )
    metrics = evaluate([case], corpus, ks=(1,)).cases[0].metrics

    assert metrics["surface_recall@1"] == 1.0, "the retriever DID return the requested surface"
    assert metrics["recall@1"] is None, f"0/0 must be unmeasured, got {metrics['recall@1']!r}"
    assert metrics["precision@1"] is None, f"0/0 must be unmeasured, got {metrics['precision@1']!r}"
    assert metrics["mrr"] is None, f"0/0 must be unmeasured, got {metrics['mrr']!r}"


def test_a_bucket_whose_cases_all_skip_a_metric_reports_it_without_coverage(corpus) -> None:
    """The bucket exists and has cases, but nobody measured THAT metric.

    The same sentinel the empty bucket already uses, one level down — so a reader of
    `eval-report.json` cannot mistake "nobody measured this" for "it scored zero".
    """
    from xbrain.knowledge.goldenset import RelevantSurface
    from xbrain.knowledge.surfaces import item_surfaces

    item_id, query = _some_item(corpus)
    surface = item_surfaces(corpus.items[item_id])[0]
    case = _case(
        id="SURFACE-ONLY",
        query=query,
        strata=("exacto",),
        relevant_surfaces=(
            RelevantSurface(owner_type="item", owner_id=item_id, surface_type=surface.surface_type),
        ),
    )
    bucket = evaluate([case], corpus, ks=(10,)).by_stratum["exacto"]
    assert bucket["cases"] == 1
    assert bucket["recall@10"] == NO_COVERAGE
    assert bucket["surface_recall@10"] == 1.0
    assert bucket["measured"]["recall@10"] == 0, "the population per metric must be stated"
    assert bucket["measured"]["surface_recall@10"] == 1


def test_an_unmeasured_metric_can_never_be_reported_as_a_threshold_failure(corpus) -> None:
    """A gate that fails a BUCKET on a metric nobody measured is the fabricated zero again.

    `_failures` read `values.get(metric, 0.0)`; with the metric carrying the sentinel that
    default would compare a dict against a float, or (with a plain 0.0) name a failure that
    measured nothing.

    REWRITTEN for M2. The first version asserted `failures == ()`, which cemented the
    fail-open: with the only case unmeasurable, "no bucket was named" and "the gate passed"
    are two different claims and it only made the first. Both have to hold at once — no
    bucket is slandered, AND a threshold that reached nothing does not report PASS.
    """
    from xbrain.knowledge.goldenset import RelevantSurface
    from xbrain.knowledge.surfaces import item_surfaces

    item_id, query = _some_item(corpus)
    surface = item_surfaces(corpus.items[item_id])[0]
    case = _case(
        id="SURFACE-ONLY",
        query=query,
        strata=("exacto",),
        relevant_surfaces=(
            RelevantSurface(owner_type="item", owner_id=item_id, surface_type=surface.surface_type),
        ),
    )
    report = evaluate([case], corpus, ks=(10,), threshold=1.0)
    assert not any("exacto" in failure for failure in report.failures), (
        f"named a bucket as failing a metric nobody measured: {report.failures}"
    )
    assert not report.passed, "a threshold compared against nothing must not report PASS (M2)"


def test_a_threshold_that_reached_no_bucket_fails_closed_naming_the_count(corpus) -> None:
    """M2: `passed` must mean "the threshold was met", not "no failure was recorded".

    Every guard below this one is correct — an empty bucket carries the sentinel, an
    unmeasured metric is skipped, and neither may be named as a failure. What nobody checked
    is the case where the skipping leaves ZERO comparisons: `passed = not failures` then
    reads the absence of a comparison as a pass, and `--min-recall 1.0`, the strictest gate
    that exists, comes out green. That is the FAIL-OPEN cell of CLAUDE.md rule 11.

    The failure names the COUNT rather than a bucket, because no bucket failed: inventing a
    bucket here would be B1 again, in the opposite direction.
    """
    from xbrain.knowledge.goldenset import RelevantSurface
    from xbrain.knowledge.surfaces import item_surfaces

    item_id, query = _some_item(corpus)
    surface = item_surfaces(corpus.items[item_id])[0]
    case = _case(
        id="SURFACE-ONLY",
        query=query,
        strata=("exacto",),
        relevant_surfaces=(
            RelevantSurface(owner_type="item", owner_id=item_id, surface_type=surface.surface_type),
        ),
    )
    report = evaluate([case], corpus, ks=(10,), threshold=1.0)
    assert len(report.failures) == 1, report.failures
    assert "0" in report.failures[0] and "medid" in report.failures[0], report.failures[0]
    assert "sin cobertura" not in report.failures[0], "the sentinel is not a failing value"
    assert "## Fallos" in render_markdown(report), "the published report must carry it too"


def test_a_threshold_that_reached_one_bucket_is_not_reported_as_unapplied(corpus) -> None:
    """The other side of M2: one real comparison is enough, and it must NOT fail closed.

    Without this, the fix could satisfy the test above by failing every run with a
    threshold — a gate that always fails is as useless as one that never does, and it would
    make `test_the_evaluation_can_fail` pass for the wrong reason.
    """
    item_id, query = _some_item(corpus)
    case = _case(id="OWNER", query=query, strata=("exacto",), relevant_items=(item_id,))
    report = evaluate([case], corpus, ks=(10,), threshold=0.0)
    assert report.passed, report.failures
    assert report.failures == ()


def test_the_markdown_renders_an_unmeasured_metric_as_words_not_a_number(corpus) -> None:
    """The human report is where a fabricated zero does its damage — it gets quoted."""
    from xbrain.knowledge.goldenset import RelevantSurface
    from xbrain.knowledge.surfaces import item_surfaces

    item_id, query = _some_item(corpus)
    surface = item_surfaces(corpus.items[item_id])[0]
    case = _case(
        id="SURFACE-ONLY",
        query=query,
        strata=("exacto",),
        relevant_surfaces=(
            RelevantSurface(owner_type="item", owner_id=item_id, surface_type=surface.surface_type),
        ),
    )
    rendered = render_markdown(evaluate([case], corpus, ks=(1, 10)))
    row = next(line for line in rendered.splitlines() if line.startswith("| exacto |"))
    assert "0.0" not in row, f"a fabricated zero reached the published table: {row}"
    assert "sin cobertura" in row


# ---------------------------------------------------------------------------
# m3 — the coverage field measures what its name says
# ---------------------------------------------------------------------------


def test_the_stat_for_refused_chunks_is_named_for_what_it_counts(corpus) -> None:
    """`empty_surfaces` counted neither empty things nor surfaces.

    In `corpus_chunks` it was initialised to 0 and never incremented, so it was hardcoded;
    in `build_index` it was recomputed as `chunks - indexed`, which counts CHUNKS the index
    refused. Two different wrong answers under one name, and no test touched it — the same
    name/value discordance that made B3 rename `stale_chunks_excluded`.

    Renamed rather than "counted for real", because counting empty SURFACES honestly would
    be a constant: `item_surfaces` and `topic_surfaces` already drop a blank surface at the
    emitter (`_blank`), so the number could never come out any other way (rule 2). What the
    index actually refuses is a real quantity, so that is what it is called.
    """
    from xbrain.knowledge.evaluation import build_index, corpus_chunks

    chunks, surfaces = corpus_chunks(corpus)
    index, stats = build_index(corpus)

    assert not hasattr(stats, "empty_surfaces"), "the misnamed field is still there"
    assert stats.surfaces == surfaces
    assert stats.chunks == len(chunks), "`chunks` is what the chunker EMITTED"
    assert stats.chunks_not_indexed == len(chunks) - len(index)


def test_a_chunk_the_index_refuses_is_counted_not_silently_dropped() -> None:
    """And the count can be non-zero — otherwise it is a constant wearing a metric's name.

    The index refuses a blank body and a `chunk_id` it already holds. Driven here through
    `add`'s public return value, because the emitter cannot produce either shape today: that
    is exactly why the number must be reported rather than assumed to be zero.
    """
    from xbrain.knowledge.index_schema import open_memory_index
    from xbrain.knowledge.lexical import LexicalIndex

    from tests.test_knowledge_lexical import _corpus_chunks

    chunks = _corpus_chunks()
    index = LexicalIndex(open_memory_index())
    assert index.add(chunks) == len(chunks)
    assert index.add(chunks) == 0, "a duplicate chunk_id must be refused, and countable"


# ---------------------------------------------------------------------------
# M3 — an empty result set is not a precision of 0.0, and it is COUNTED
# ---------------------------------------------------------------------------


def test_precision_over_an_empty_result_set_is_unmeasured_not_zero(corpus) -> None:
    """The last instance of B1, and it is the one that reached the published table.

    `precision@k = found / len(top)` with `top` empty is a hard 0/0. Its numerator is 0 by
    construction, so the number restates the empty set instead of measuring the retriever —
    the same argument 3006c0c used two lines above for `recall` and `mrr`.

    `recall@k` is DIFFERENT and must stay 0.0: its denominator is the case's known relevant
    set, so "we retrieved none of the two items that exist" is a real measurement of a real
    failure. Asserting both here is what stops the fix from over-reaching.
    """
    item_id, _ = _some_item(corpus)
    case = _case(
        id="NOTHING-MATCHES",
        query="Zzyzxquorumbleflange",
        strata=("exacto",),
        relevant_items=(item_id,),
    )
    result = evaluate([case], corpus, ks=(10,)).cases[0]
    assert result.retrieved == (), "the fixture query must match nothing for this to discriminate"
    assert result.metrics["precision@10"] is None, "0/0 reached the report as a measurement"
    assert result.metrics["recall@10"] == 0.0, "recall over a known relevant set IS measured"


def test_a_case_that_retrieved_nothing_is_counted_as_such(corpus) -> None:
    """A 0.0 recall has two causes and the report must tell them apart.

    "The right item ranked below k" is a ranking failure; "the query matched no chunk at
    all" is a failure of the query, and on the real corpus it was the cause in 18 of 21
    cases while the report read all of them as the first. One number cannot say which, so
    the count of empty result sets ships beside `measured`.
    """
    item_id, query = _some_item(corpus)
    empty = _case(
        id="NOTHING-MATCHES",
        query="Zzyzxquorumbleflange",
        strata=("exacto",),
        relevant_items=(item_id,),
    )
    found = _case(id="MATCHES", query=query, strata=("exacto",), relevant_items=(item_id,))

    report = evaluate([empty, found], corpus, ks=(10,))

    assert report.cases[0].no_results is True
    assert report.cases[1].no_results is False
    assert report.by_stratum["exacto"]["no_results"] == 1
    assert report.by_stratum["exacto"]["cases"] == 2
    assert report.to_dict()["cases"][0]["no_results"] is True


def test_the_markdown_publishes_how_many_cases_retrieved_nothing(corpus) -> None:
    """The markdown is the surface that gets quoted, so the count has to reach it.

    Without it, a stratum reading 0.0 across the board is indistinguishable from a stratum
    whose retriever was never given a chance to rank anything.
    """
    item_id, _ = _some_item(corpus)
    case = _case(
        id="NOTHING-MATCHES",
        query="Zzyzxquorumbleflange",
        strata=("exacto",),
        relevant_items=(item_id,),
    )
    rendered = render_markdown(evaluate([case], corpus, ks=(1, 10)))
    row = next(line for line in rendered.splitlines() if line.startswith("| exacto |"))
    assert "vacíos" in rendered, "the column must be labelled where it is read"
    assert row.split("|")[3].strip() == "1", f"the empty-result count is missing from: {row}"


# ---------------------------------------------------------------------------
# m10 — one list, in code, not in a docstring
# ---------------------------------------------------------------------------


def test_the_aggregate_carries_every_metric_the_scorer_produced() -> None:
    """m10: `_metric_names` claimed a binding with `_score` that did not exist.

    Its docstring said "One list, so the aggregate and the per-case scoring cannot disagree
    about what exists" — but `_score` never called it. It built `recall@{k}` and friends from
    its own f-strings, so there were two lists that happened to match. One direction was
    guarded by accident (dropping a name from `_metric_names` turns
    `test_metrics_are_reported_per_stratum_and_provenance` red); the other was not, and a
    metric added to `_score` would vanish from the aggregate, from `measured` and from the
    published table without a word. That is rule 5 — bind them in code, never in prose — in
    the module whose docstring cites it.

    The binding is asserted the only way that discriminates: hand the aggregate a case
    carrying a metric no list mentions, and require it to come through.
    """
    from xbrain.knowledge.evaluation import CaseResult, _bucket_means

    member = CaseResult(
        id="NOVEL",
        provenance="construido",
        strata=("exacto",),
        retrieved=(),
        metrics={"recall@10": 1.0, "ndcg@10": 0.5},
        no_results=False,
    )

    means = _bucket_means([member])

    assert means["ndcg@10"] == 0.5, "a metric the scorer produced never reached the aggregate"
    assert means["measured"]["ndcg@10"] == 1, "and its denominator went missing with it"
    assert means["recall@10"] == 1.0


def test_the_aggregate_names_come_from_the_cases_not_from_a_second_list(report) -> None:
    """The same binding, through the public API: the two key sets are IDENTICAL.

    Not "the aggregate contains the metrics" — that is satisfied by a superset, which is
    exactly what a stale second list would produce.
    """
    bucket = report.by_stratum["exacto"]
    case = next(c for c in report.cases if "exacto" in c.strata)
    assert set(bucket["measured"]) == set(case.metrics)


# ---------------------------------------------------------------------------
# 02.13 — lexical_memory retirement guard
# ---------------------------------------------------------------------------


def test_lexical_memory_is_retired_and_not_imported() -> None:
    """The in-memory baseline wrapper is DELETED by Plan 02.13.

    This guard ensures no active import of `lexical_memory` remains anywhere in the knowledge
    package. The module was retired when the evaluation harness moved to the persisted index
    writer (`LexicalIndex` via `open_memory_index`), which uses the SAME schema and scorer
    as the persisted index — the difference is only where the database lives.

    R2 of the delivery plan: `evaluation.py` was the sole remaining consumer.
    """
    import ast
    from pathlib import Path

    knowledge_dir = Path(__file__).parent.parent / "src" / "xbrain" / "knowledge"
    violations: list[str] = []

    for py_file in knowledge_dir.glob("*.py"):
        if py_file.name == "lexical_memory.py":
            violations.append(f"{py_file.name}: module file still exists")
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "lexical_memory" in alias.name:
                        violations.append(f"{py_file.name}:{node.lineno}: imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and "lexical_memory" in node.module:
                    violations.append(f"{py_file.name}:{node.lineno}: imports from {node.module}")

    assert not violations, (
        "lexical_memory.py was retired in 02.13 but active imports remain:\n  "
        + "\n  ".join(violations)
    )


# ---------------------------------------------------------------------------
# U-6 (round 07) — the depth is counted in OWNERS, and `recall@k` depends on nothing else
# ---------------------------------------------------------------------------
#
# Gate Codex F5: `evaluate` asked the index for `max(limit, max(ks))` CHUNKS and deduplicated
# owners afterwards, while every metric is defined over owners. Seven windows of one
# transcript at the top of the ranking meant ten chunks held two owners; so `recall@10` for
# the real case F2 was 2/3 when `--k 10` was requested alone and 1.0 when `k=20` was requested
# beside it (the default), and the `filtros` stratum moved from 0.8333 to 1.0. A published
# figure that depends on which neighbouring figure was asked for is a figure that cannot come
# out any other way (rule 2).

OWNER_DOMINATED_QUERY = "Retrieval quality depends"


def _dominated_case():
    return _case(
        id="U6",
        query=OWNER_DOMINATED_QUERY,
        strata=("enterrado",),
        relevant_items=("k08", "k04"),
    )


def test_recall_at_k_is_independent_of_the_other_ks_requested(corpus) -> None:
    """The precondition first (rule 1): two chunks of this query hold ONE owner — the
    transcript of k08 fills the top of the ranking — so a depth counted in chunks would
    give `recall@2` a different value depending on whether a deeper k was also asked for.
    Then the property: `recall@2` is the same number whether `ks=(2,)` or `ks=(2, 10)`,
    because the ranking is materialised to at least two OWNERS either way.

    Seen red on the umbrella head `b798ad7`, whose `_search` called `index.search(q, limit)`:
    0.5 alone, 1.0 beside k=10.
    """
    index, _stats = build_index(corpus)
    try:
        two_chunks = index.search(OWNER_DOMINATED_QUERY, 2)
    finally:
        index.connection.close()
    assert len({(h.owner_type, h.owner_id) for h in two_chunks}) == 1, "the precondition moved"

    alone = evaluate([_dominated_case()], corpus, ks=(2,))
    beside = evaluate([_dominated_case()], corpus, ks=(2, 10))
    recall_alone = alone.cases[0].metrics["recall@2"]
    recall_beside = beside.cases[0].metrics["recall@2"]
    assert recall_alone == recall_beside == 1.0
    assert alone.cases[0].retrieved[:2] == beside.cases[0].retrieved[:2]


def test_the_depth_is_counted_in_owners_and_published(corpus) -> None:
    """`limit` is a number of OWNERS: `evaluate(..., ks=(3,))` materialises three distinct
    owners for a query that has them, however many chunks of the first owner sit on top —
    and the report says which depth it ran at, so a figure travels with the depth that
    produced it (rule 2)."""
    report = evaluate([_dominated_case()], corpus, ks=(3,))
    retrieved = report.cases[0].retrieved
    assert len(retrieved) == 3 and len(set(retrieved)) == 3, retrieved
    assert report.limit == 3 and report.to_dict()["limit"] == 3
    deeper = evaluate([_dominated_case()], corpus, ks=(3,), limit=5)
    assert deeper.limit == 5 and len(deeper.cases[0].retrieved) <= 5
    assert deeper.cases[0].retrieved[:3] == retrieved


def test_a_ranking_that_ran_out_of_depth_says_so_rather_than_reporting_a_short_list(
    corpus, monkeypatch
) -> None:
    """`depth_exhausted` is the declaration U-6 bought, and it has to be READ somewhere.

    `MAX_CHUNK_DEPTH` bounds the work; reaching it with fewer owners than asked means the
    owner list is short for a reason that is NOT «the owner was not there». Without the flag
    a reader cannot tell the two apart, and the missing owner reads as a retrieval failure.

    Driven by lowering the bound to 1 rather than by building a corpus big enough to hit
    10.000 chunks: the property is that the bound, WHEREVER it is, is declared when reached.

    Seen red by returning `False` unconditionally from `search_owners`: the case still
    reports two owners short and nothing on the report says why.
    """
    # The control runs FIRST, under the real bound: `monkeypatch` undoes itself at teardown,
    # so a second call after the patch would still be reading the lowered value and the
    # "unexhausted" half would be asserting nothing (rule 1).
    unbounded = evaluate([_dominated_case()], corpus, ks=(3,))
    assert unbounded.cases[0].depth_exhausted is False
    assert len(unbounded.cases[0].retrieved) == 3

    monkeypatch.setattr("xbrain.knowledge.lexical.MAX_CHUNK_DEPTH", 1)
    report = evaluate([_dominated_case()], corpus, ks=(3,))

    assert report.cases[0].depth_exhausted is True
    assert report.to_dict()["cases"][0]["depth_exhausted"] is True
    assert len(report.cases[0].retrieved) < 3, "the bound really did cut the list short"


# ---------------------------------------------------------------------------
# The chunker sweep (Plan 02 §7 · §15.12 signed-measurement half; delivery row 02.13)
# ---------------------------------------------------------------------------


def test_parse_sweep_accepts_both_syntaxes_and_refuses_a_typo() -> None:
    """A typo that silently swept nothing would publish the DEFAULT's numbers as a sweep.

    So an unknown axis is refused rather than ignored. Both spellings work: one flag per axis,
    and the plan's own quoted `target=... overlap=...`.
    """
    from xbrain.knowledge.evaluation import parse_sweep

    assert parse_sweep(["target=800,1200", "overlap=0,150"]) == {
        "target": [800, 1200],
        "overlap": [0, 150],
    }
    assert parse_sweep(["target=800,1200 overlap=0,150"]) == {
        "target": [800, 1200],
        "overlap": [0, 150],
    }
    with pytest.raises(ValueError, match="desconocido"):
        parse_sweep(["targt=800"])
    with pytest.raises(ValueError, match="inválido"):
        parse_sweep(["target"])


def test_the_sweep_scores_every_combination_and_ranks_them(corpus) -> None:
    """§7: the cartesian product, best `recall@k` first.

    The row count is the product of the axes, asserted so a sweep that silently dropped a
    combination — the failure that would make a "winner" the winner of a smaller contest —
    goes red.
    """
    from xbrain.knowledge.evaluation import sweep_chunker

    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    report = sweep_chunker(cases, corpus, {"target": [800, 1600], "overlap": [0, 150]})

    assert len(report.rows) == 4
    assert report.winner is report.rows[0]
    recalls = [row.recall for row in report.rows if row.recall is not None]
    assert recalls == sorted(recalls, reverse=True)


def test_the_sweep_reports_the_chunk_count_so_a_tie_can_be_broken(corpus) -> None:
    """Spec §13.15: a flat result is DOCUMENTED, and the tie-break is fewer chunks.

    That only works if the count is in the table, so it is asserted to be there and to differ
    between combinations — a column that were constant could not break anything.
    """
    from xbrain.knowledge.evaluation import sweep_chunker

    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    report = sweep_chunker(cases, corpus, {"target": [400, 2400]})

    counts = {row.params.target: row.chunks for row in report.rows}
    assert all(count > 0 for count in counts.values())
    assert counts[400] > counts[2400], "a smaller target must produce more chunks"


def test_the_sweep_publishes_recall_at_1_and_names_the_criterion_that_decided(corpus) -> None:
    """S-1 (gate Fable round 08): the published winner of the real sweep — 800/0 over
    1200/0 — was decided by MRR after a tie on `recall@10`, while Plan 02 §7 and the README
    said the tie-break is FEWER CHUNKS (which would have chosen 1200/0). The rule the code
    applies is `recall@k`, then MRR, then fewer chunks; it is now written where the plan and
    the README can be checked against it, and the report says WHICH criterion decided, so a
    reader never has to infer it from the table. `recall@1` is published on every row: it is
    the depth-independent figure the decision rests on and was not re-derivable from the
    sweep's own output.
    """
    from xbrain.knowledge.evaluation import render_sweep_markdown, sweep_chunker

    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    report = sweep_chunker(cases, corpus, {"target": [400, 2400]}, k=10)
    rows = report.to_dict()["rows"]
    assert all("recall@1" in row and "recall@10" in row for row in rows), rows
    assert all(row["recall@1"] is None or 0.0 <= row["recall@1"] <= 1.0 for row in rows)
    text = render_sweep_markdown(report)
    assert "recall@1" in text.splitlines()[2]
    assert "decidió" in text or "PLANO" in text, text
    assert "recall@10" in text and "MRR" in text and "menos chunks" in text


def test_a_flat_sweep_says_it_is_flat(corpus) -> None:
    """The negative result, published as one (spec §13.15).

    Two combinations that score identically must not be presented as a winner and a loser:
    the rendering says PLANO and states that the tie-break was the chunk count.
    """
    from xbrain.knowledge.evaluation import render_sweep_markdown, sweep_chunker

    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    # `min_chars` at two values that cannot change the ranking on this corpus: the fixture has
    # no fragment near the floor, so the two runs are identical by construction.
    report = sweep_chunker(cases, corpus, {"min_chars": [40, 41]})
    text = render_sweep_markdown(report)

    assert "PLANO" in text
    assert "menos chunks" in text


def test_the_sweep_publishes_the_retriever_that_ranked_the_cells(corpus, monkeypatch) -> None:
    """F-2's sixth site: the sweep branch of the command whose other branch was repaired.

    `xbrain eval --strategy vector` published a report headed `vector` produced entirely by
    bm25, and the fix named the retriever in five places. `--sweep-chunker` is the SAME
    command, it calls the same `evaluate` once per cell, `resolve_strategy` runs and the
    degradation is computed — and then it was discarded with the rest of the per-cell report.
    `data/eval-sweep.{json,md}` is the artefact Plan 03 has to beat, and it named no retriever
    at all: measured on the fixture corpus at `6b368e9`, `SweepReport.to_dict()` returned the
    keys `['k', 'limit', 'measured', 'rows', 'verdict', 'winner']` and not one of them says
    which instrument produced the ranking.

    THE PREMISE IS PINNED, NOT INHERITED, exactly as in the non-sweep twin above: `vector` is
    the example of a declared-but-unimplemented strategy, and reading that from production
    would expire the day Plan 03 lands.

    Seen red before the fix: `KeyError: 'strategy'` on the payload, and the rendered table
    contained the word `vector` nowhere.
    """
    from xbrain.knowledge.evaluation import render_sweep_markdown, sweep_chunker

    monkeypatch.setattr(contracts, "IMPLEMENTED_STRATEGIES", frozenset({"lexical"}))
    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    report = sweep_chunker(cases, corpus, {"target": [800, 1600]}, strategy="vector")
    payload = report.to_dict()

    assert payload["strategy"] == "lexical", "what ran"
    assert payload["requested_strategy"] == "vector", "what was asked for"
    assert payload["degraded"] == ["vector_not_implemented"]
    assert payload["rows"], "spec §9.3: lexical stays operational, the table is still produced"

    first = render_sweep_markdown(report).splitlines()[0]
    assert "`lexical`" in first, "the table names the retriever that ranked it"
    assert "vector" in first, "the request is not hidden either"


def test_the_sweep_and_the_report_name_the_retriever_with_the_SAME_sentence(
    corpus, monkeypatch
) -> None:
    """Rule 5, through the two PUBLIC renderers rather than through one shared symbol.

    The defect was never that the sweep lacked a field: it was that `xbrain eval` publishes a
    number down two branches and only one of them named its instrument. Asserting that both
    call `retriever_label` would be the tautology of rule 1 row 6 — once they delegate, the
    attribute IS the same object and the assertion cannot fail. So this asserts on what the
    two renderers OUTPUT over the same three values, which is what a reader actually gets.

    IT IS RUN ON THE DEGRADED CASE, and that is the whole of the test. Written first on a
    plain `lexical` run, it passed against a sweep line hardcoded back to
    `f"Recuperador: `{report.strategy}`"` — with nothing degraded the two forms produce the
    same bytes, so the assertion could not fail for the reason it exists. Only a request that
    could not run separates «names the retriever» from «echoes the strategy field».

    Seen red under exactly that mutation: the ordinary heading carries `solicitada `vector`,
    sin backend (vector_not_implemented)` and the sweep's line carries nothing.
    """
    from xbrain.knowledge.evaluation import render_sweep_markdown, sweep_chunker

    monkeypatch.setattr(contracts, "IMPLEMENTED_STRATEGIES", frozenset({"lexical"}))
    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    plain = evaluate(cases, corpus, strategy="vector")
    swept = sweep_chunker(cases, corpus, {"target": [800]}, strategy="vector")

    heading = render_markdown(plain).splitlines()[0]
    sweep_line = render_sweep_markdown(swept).splitlines()[0]

    clause = heading.removeprefix("# Evaluación de recuperación — ")
    assert clause != "`lexical`", "the degraded case is what makes the two forms separable"
    assert "vector_not_implemented" in clause
    # The SAME sentence, not two that happen to agree on the undegraded input.
    assert sweep_line == f"Recuperador: {clause}"


def test_an_unknown_sweep_strategy_raises_before_any_cell_is_scored(corpus) -> None:
    """A typo is not a degradation, and the sweep must refuse it where the report does.

    `resolve_strategy` raises for a strategy in no contract at all — answering a typo with
    lexical results would turn it into a measurement. Resolving ONCE at the top of the sweep,
    rather than inheriting the raise from the first `evaluate`, is what makes that refusal
    independent of whether the grid produced any combination to walk: an empty grid used to
    return a `SweepReport` for a strategy that does not exist.
    """
    from xbrain.knowledge.evaluation import sweep_chunker

    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    with pytest.raises(ValueError):
        sweep_chunker(cases, corpus, {"target": []}, strategy="lexcial")


def test_the_sweep_cannot_move_the_characterization_fixture(corpus) -> None:
    """Step 17b (M7), asserted where the sweep lives.

    The sweep passes `ChunkerParams` as an ARGUMENT and never assigns
    `DEFAULT_CHUNKER_PARAMS`, so the pinned ranking — which passes its own parameters — is
    untouchable by it. Checked by running the sweep and then re-running the pinned assertion
    in the same process: if the sweep mutated the module constant, the fixture would move.
    """
    from xbrain.knowledge.chunking import DEFAULT_CHUNKER_PARAMS
    from xbrain.knowledge.evaluation import sweep_chunker

    from tests.test_knowledge_lexical import test_ranking_matches_the_characterization_fixture

    before = DEFAULT_CHUNKER_PARAMS
    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    sweep_chunker(cases, corpus, {"target": [400, 2400], "overlap": [0, 300]})

    assert DEFAULT_CHUNKER_PARAMS is before, "the sweep mutated the module default"
    test_ranking_matches_the_characterization_fixture()


def test_the_sweep_honours_and_publishes_the_limit(corpus, monkeypatch) -> None:
    """The sweep runs `evaluate` once per combination; the depth it runs at is the caller's
    `limit` (the CLI's `--limit`), never silently the k. Asserted at the seam — the limit
    each `evaluate` call received — and on the published report, which carries it.

    Seen red on the snapshot's `9dfa34e`: `sweep_chunker` took no `limit`, `evaluate` received
    none, and `--limit 10` / `--limit 150` produced byte-identical sweep reports.
    """
    from xbrain.knowledge import evaluation
    from xbrain.knowledge.evaluation import render_sweep_markdown, sweep_chunker

    received: list[int | None] = []
    real = evaluation.evaluate

    def recording(*args, **kwargs):
        received.append(kwargs.get("limit"))
        return real(*args, **kwargs)

    monkeypatch.setattr(evaluation, "evaluate", recording)
    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    report = sweep_chunker(cases, corpus, {"target": [800, 1600]}, k=10, limit=150)

    assert received == [150, 150]
    assert report.limit == 150 and report.to_dict()["limit"] == 150
    assert "150" in render_sweep_markdown(report)
    # And with no limit given the sweep runs at its k — declared, not implicit.
    received.clear()
    default = sweep_chunker(cases, corpus, {"target": [800]}, k=10)
    assert received == [10] and default.limit == 10


def test_the_published_depth_is_the_depth_the_cells_ran_at_never_the_one_asked_for(
    corpus, monkeypatch
) -> None:
    """A DEVIATION FROM THE SNAPSHOT, and the reason for it. `evaluate` clamps its own depth
    to `max(limit, max(ks))`, so a `limit` BELOW `k` never reaches the index — the snapshot's
    `sweep_chunker` nonetheless recorded the unclamped value, and `report.limit` would then
    name a depth no cell ran at, which is exactly the figure CLAUDE.md rule 2 forbids. The
    sweep clamps too, so the published number and the executed one are one number.

    Seen red against the ported-verbatim version: `report.limit` was 3 while every `evaluate`
    call received 3 and ran at 10.
    """
    from xbrain.knowledge import evaluation
    from xbrain.knowledge.evaluation import sweep_chunker

    seen: list[int] = []
    real = evaluation.evaluate

    def recording(*args, **kwargs):
        report = real(*args, **kwargs)
        # What the retriever was actually asked for, read off `evaluate`'s own arithmetic.
        seen.append(max(kwargs.get("limit") or 0, max(kwargs["ks"])))
        return report

    monkeypatch.setattr(evaluation, "evaluate", recording)
    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    report = sweep_chunker(cases, corpus, {"target": [800]}, k=10, limit=3)

    assert seen == [10], seen
    assert report.limit == 10, "the report names a depth no cell ran at"


def test_evaluate_closes_the_index_it_built(corpus) -> None:
    """`evaluate` builds a `:memory:` index through `build_index` and used to return with the
    connection still open — and `sweep_chunker` calls `evaluate` once per combination, so a
    twelve-cell sweep opened twelve handles and closed none of them explicitly.

    ASSERTED ON THE CONNECTION, not on a `ResourceWarning`. The snapshot's note records one
    `unclosed database` warning per `evaluate`; that warning did NOT reproduce on this tree,
    so the surface read here is the one that answers the question directly (rule 9). Seen red
    before the `finally`: `SELECT 1` on the captured connection succeeded.
    """
    import sqlite3

    from xbrain.knowledge import evaluation

    built = []
    real = evaluation.build_index

    def capture(*args, **kwargs):
        index, stats = real(*args, **kwargs)
        built.append(index)
        return index, stats

    evaluation.build_index = capture
    try:
        evaluate(resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items), corpus)
    finally:
        evaluation.build_index = real

    assert len(built) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        built[0].connection.execute("SELECT 1")


def test_a_one_cell_sweep_does_not_report_a_tie_it_could_not_have_measured(corpus) -> None:
    """A SECOND DEVIATION FROM THE SNAPSHOT. `_sweep_verdict` tested `len(distinct) == 1`,
    which is also true of a ONE-ROW table, so `xbrain eval --sweep-chunker target=800` printed
    «PLANO: todas las combinaciones puntúan igual» — a verdict about a tie over a table with
    nothing to tie against, and one that no input could have made say anything else. That is
    the shape CLAUDE.md rule 2 rejects, inside the instrument that exists so a published
    number can be re-derived honestly.

    Seen red against the ported-verbatim version, on the real corpus and on this fixture: one
    row, `PLANO` printed.
    """
    from xbrain.knowledge.evaluation import render_sweep_markdown, sweep_chunker

    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    one = render_sweep_markdown(sweep_chunker(cases, corpus, {"target": [800]}))

    assert "PLANO" not in one, one
    assert "UNA COMBINACIÓN" in one and "target=800" in one, one
    # And a real tie still says PLANO, so the deviation narrowed the claim and did not delete it.
    flat = render_sweep_markdown(sweep_chunker(cases, corpus, {"min_chars": [40, 41]}))
    assert "PLANO" in flat, flat


def test_a_sweep_that_measured_nothing_has_no_winner_and_does_not_report_a_tie(corpus) -> None:
    """F2-3 of the final gate on #177. `_sweep_verdict` built `distinct` out of `_number(...)`
    STRINGS, and `_number(None)` is the constant `"sin cobertura"` — so N rows that could not
    be scored at all collapsed to ONE distinct value and took the flat branch. Executed there
    with `cases=[]` over two combinations, the report read:

        |  800 | 0 | 56 | sin cobertura | sin cobertura | sin cobertura |
        | 1600 | 0 | 47 | sin cobertura | sin cobertura | sin cobertura |
        PLANO: todas las combinaciones puntúan igual; gana la que produce menos chunks.

    The table is honest and the verdict is not: it makes a positive claim about a ranking that
    never happened, over rows where NOTHING was compared. It is declared deviation 3 (the
    one-row fake tie) one input class over — same predicate, same false verdict — and the
    verdict is the one line a reader takes away.

    A sweep with nothing measured therefore has NO WINNER and says so, and the contrast is
    asserted in the same test: a GENUINE tie — rows that all scored the same REAL number —
    still reports PLANO and still has a winner, so the repair narrowed the claim instead of
    deleting it.
    """
    from xbrain.knowledge.evaluation import render_sweep_markdown, sweep_chunker

    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)

    nothing = sweep_chunker([], corpus, {"target": [800, 1600]})
    text = render_sweep_markdown(nothing)
    assert nothing.winner is None, "an unscored row was published as the winner"
    assert nothing.measured is False
    assert "PLANO" not in text, text
    assert "SIN MEDICIÓN" in text, text
    payload = nothing.to_dict()
    assert payload["winner"] is None and payload["measured"] is False
    assert "SIN MEDICIÓN" in payload["verdict"]

    # A grid that resolves to no combination at all has no winner either, and says a DIFFERENT
    # thing: nothing was swept, as against swept and unscorable.
    empty = sweep_chunker(cases, corpus, {"target": []})
    assert empty.winner is None and empty.rows == ()
    assert "SIN COMBINACIONES" in render_sweep_markdown(empty)

    # THE MEASURED FLAT TIE IS PRESERVED. `min_chars` at 40 and 41 cannot change the ranking
    # on this corpus (no fragment sits near the floor), so both rows score the same REAL
    # value — which is a result about the chunker, not an absence of one.
    flat = sweep_chunker(cases, corpus, {"min_chars": [40, 41]})
    assert flat.measured is True and flat.winner is not None
    assert flat.winner.recall is not None
    assert "PLANO" in render_sweep_markdown(flat)


def test_the_top_row_is_the_winner_only_when_it_actually_scored() -> None:
    """The guard read off the ROW's own state, so a future change to the sort order cannot
    quietly publish an unscored combination as the winner.

    Asserted on a report BUILT HERE rather than on one `sweep_chunker` produced: measurability
    depends on the cases (a filter the strategy cannot push), never on the chunk size, so every
    combination of a real sweep is scorable or none is, and the mixed table below is not
    reachable through the public entry point. It is exactly the state a reordering would
    create, which is why it is pinned here instead of being argued.
    """
    from xbrain.knowledge.chunking import DEFAULT_CHUNKER_PARAMS
    from xbrain.knowledge.evaluation import SweepReport, SweepRow

    unscored = SweepRow(
        params=DEFAULT_CHUNKER_PARAMS, chunks=10, recall=None, mrr=None, by_stratum={}
    )
    scored = SweepRow(params=DEFAULT_CHUNKER_PARAMS, chunks=99, recall=0.5, mrr=0.5, by_stratum={})

    assert SweepReport(k=10, rows=(unscored, scored)).winner is None
    assert SweepReport(k=10, rows=(scored, unscored)).winner is scored


# ---------------------------------------------------------------------------
# Plan 03.7 — the harness per strategy, and the bake-off's instruments (TDD 22, 23, 28)
# ---------------------------------------------------------------------------
#
# NO TEST HERE TOUCHES A MODEL (criterion §13.11). The embedder is a hash onto the unit
# circle, as in `test_knowledge_search_hybrid.py`: a query embedded AS a chunk's text finds
# THAT chunk at cosine 1, which is what lets a test choose, deterministically, a relevant item
# the vector channel reaches and bm25 cannot. What is asserted is where a number came FROM —
# the channel that ran, the model that built the plane, the cases a verdict rests on — and
# never a value of `RRF_K` or of the weights, which this child is the one allowed to move.

NO_OVERLAP_QUERY = "Zzyzxquorumbleflange"


def _circle(text: str) -> tuple[float, float]:
    import math

    angle = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    angle *= 2 * math.pi
    return (math.cos(angle), math.sin(angle))


class _PassageEmbedder:
    """The build side: one circle point per text, counting how many texts it was paid for."""

    def __init__(self) -> None:
        self.texts = 0

    def __call__(self, texts):
        self.texts += len(texts)
        return [_circle(text) for text in texts]


def _vector_workspace(tmp_path: Path, corpus) -> Path:
    """The fixture corpus as the three files a persisted index is built from and sealed against."""
    from xbrain.rubrics import save_vocab
    from xbrain.store import save_store, save_topic_pages

    data = tmp_path / "data"
    save_store(corpus.items, data / "items.json")
    save_vocab(corpus.vocab, data / "vocab.yaml")
    save_topic_pages(corpus.topic_pages, data / "topics.json")
    return data


def _vectors(
    data: Path,
    index_dir: Path,
    *,
    requested: str = "fake/model-a",
    served: str | None = None,
    queries: dict[str, str] | None = None,
    passages: _PassageEmbedder | None = None,
    query_calls: list[str] | None = None,
    query_delay: float = 0.0,
):
    """A `VectorEvaluation` over `data`, with a backend that serves `served` (default: `requested`).

    `queries` maps a query onto the TEXT its vector should land on; an unmapped query lands on
    its own hash, which matches no chunk.
    """
    import time as _time

    from xbrain.knowledge.evaluation import VectorEvaluation
    from xbrain.knowledge.index_build import VectorBuild
    from xbrain.knowledge.vector_index import VectorSpec

    spec = VectorSpec(
        model=served or requested,
        dimension=2,
        normalized=True,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )
    targets = queries or {}

    def embed_query(query: str) -> tuple[float, float]:
        if query_calls is not None:
            query_calls.append(query)
        if query_delay:
            _time.sleep(query_delay)
        return _circle(targets.get(query, query))

    return VectorEvaluation(
        requested_model=requested,
        build=VectorBuild(spec=spec, embed=passages or _PassageEmbedder()),
        embed_query=embed_query,
        index_dir=index_dir,
        items_path=data / "items.json",
        vocab_path=data / "vocab.yaml",
        topics_path=data / "topics.json",
        command="fake-embedder --offline",
    )


def _chunk_text(corpus, item_id: str) -> str:
    """The first chunk of `item_id` as the writer stores it — the text the plane embeds."""
    index, _stats = build_index(corpus)
    try:
        row = index.connection.execute(
            "SELECT text FROM chunks WHERE owner_type = 'item' AND owner_id = ? "
            "ORDER BY chunk_id LIMIT 1",
            (item_id,),
        ).fetchone()
    finally:
        index.connection.close()
    assert row is not None, f"{item_id} has no chunk in the fixture corpus"
    return str(row[0])


def _other_item(corpus, not_this: str) -> str:
    """Another item that owns at least one chunk."""
    index, _stats = build_index(corpus)
    try:
        row = index.connection.execute(
            "SELECT owner_id FROM chunks WHERE owner_type = 'item' AND owner_id != ? "
            "ORDER BY owner_id LIMIT 1",
            (not_this,),
        ).fetchone()
    finally:
        index.connection.close()
    assert row is not None
    return str(row[0])


def test_a_vector_evaluation_is_reported_per_stratum_and_provenance_by_the_vector_channel(
    tmp_path: Path, corpus
) -> None:
    """TDD 22 (Plan 03 §7): `eval --strategy vector` reports by stratum and provenance — and
    the numbers are the VECTOR channel's.

    The second half is what makes the first worth asserting: a report of the right shape
    produced by bm25 is F-2 again. So the case is one bm25 CANNOT answer — a query with no
    word in the corpus, embedded onto the relevant item's chunk. Lexically it retrieves
    nothing (asserted: the precondition); through the plane it is rank 1.

    Seen red with the vector path scoring the lexical index: `recall@1` came back 0.0.
    """
    item_id, _query = _some_item(corpus)
    case = _case(
        id="NO-OVERLAP", query=NO_OVERLAP_QUERY, strata=("semantico",), relevant_items=(item_id,)
    )
    data = _vector_workspace(tmp_path, corpus)
    vectors = _vectors(
        data, tmp_path / "eval-index", queries={NO_OVERLAP_QUERY: _chunk_text(corpus, item_id)}
    )

    lexical = evaluate([case], corpus, ks=(1, 10))
    assert lexical.cases[0].metrics["recall@1"] == 0.0, "precondition: bm25 cannot reach it"

    payload = evaluate([case], corpus, strategy="vector", ks=(1, 10), vectors=vectors).to_dict()

    assert payload["strategy"] == "vector" and payload["requested_strategy"] == "vector"
    assert payload["degraded"] == []
    assert "recall@1" not in payload, "never one global figure"
    assert payload["by_stratum"]["semantico"]["recall@1"] == 1.0
    assert payload["by_provenance"]["construido"]["recall@1"] == 1.0
    assert payload["by_stratum"]["exacto"] == NO_COVERAGE


def _unique_token(corpus, item_id: str) -> str:
    """A word of `item_id`'s first chunk that bm25 finds in NO other owner."""
    import re as _re

    index, _stats = build_index(corpus)
    try:
        for token in _re.findall(r"[A-Za-z]{6,}", _chunk_text(corpus, item_id)):
            owners = {(hit.owner_type, hit.owner_id) for hit in index.search(token, 200)}
            if owners == {("item", item_id)}:
                return token
    finally:
        index.connection.close()
    raise AssertionError(f"{item_id} has no word unique to it in the fixture corpus")


def test_hybrid_fuses_both_channels_and_names_itself(tmp_path: Path, corpus, monkeypatch) -> None:
    """TDD 22 for `hybrid`: the page is the FUSION of both channels, not either one alone.

    Built so the order is decided by arithmetic, not by the fixture: the query is a word only
    `lexical_item` holds, and its vector lands on `vector_item`'s chunk, which shares no word
    with it. So `vector_item` scores one channel's rank 1, `1/(K+1)`; `lexical_item` scores
    that PLUS a vector rank of its own; every other owner scores less than `1/(K+1)`. Under
    equal weights the fused order is `lexical_item`, `vector_item` for ANY `RRF_K` — while the
    plane alone puts `vector_item` first, and bm25 alone never reaches it.

    THE PREMISE IS PINNED HERE, not inherited: `RRF_K` and the weights are set by the test,
    because 03.7 is the child allowed to move the ones in `fusion.py`.

    Seen red by fusing the vector channel alone (`lexical=False`): `vector_item` came first.
    """
    from xbrain.knowledge import fusion

    monkeypatch.setattr(fusion, "RRF_K", 60)
    monkeypatch.setattr(fusion, "CHANNEL_WEIGHTS", {"lexical": 1.0, "vector": 1.0})
    lexical_item, _query = _some_item(corpus)
    vector_item = _other_item(corpus, lexical_item)
    # The precondition — no other owner, `vector_item` included, holds the word — is what
    # `_unique_token` returns by construction.
    token = _unique_token(corpus, lexical_item)
    target = _chunk_text(corpus, vector_item)
    case = _case(id="FUSED", query=token, strata=("semantico",), relevant_items=(vector_item,))
    data = _vector_workspace(tmp_path, corpus)
    vectors = _vectors(data, tmp_path / "eval-index", queries={token: target})

    report = evaluate([case], corpus, strategy="hybrid", ks=(1, 10), vectors=vectors)

    assert report.strategy == "hybrid" and report.degraded == ()
    assert report.cases[0].retrieved[:2] == (f"item:{lexical_item}", f"item:{vector_item}")


def test_a_vector_strategy_names_no_model_it_did_not_measure(corpus) -> None:
    """A model passed with `lexical` would publish a lexical report beside a model name that
    produced none of its numbers — refused, like every flag a path cannot honour."""
    item_id, query = _some_item(corpus)
    case = _case(id="ONE", query=query, strata=("exacto",), relevant_items=(item_id,))
    with pytest.raises(ValueError, match="lexical"):
        evaluate([case], corpus, strategy="lexical", vectors=_vectors(Path("."), Path(".")))


def test_the_evaluation_refuses_an_index_whose_manifest_holds_another_model(
    tmp_path: Path, corpus
) -> None:
    """TDD 23 (Plan 03 §7): eval detects and FAILS when the manifest's model is not the one
    asked for.

    Two models' vectors never share a matrix, and a report headed with model B computed over
    a plane model A wrote is a number whose label does not describe its instrument (rule 2).
    Rebuilding silently over it is the other wrong answer: that index is someone's measurement
    of model A in a directory the operator named. So it refuses, naming BOTH models, and the
    manifest is left as it was found.

    Seen red with reuse decided on the dimension alone: both fake models are two wide, the
    plane was reused, and model A's numbers were published as model B's.
    """
    from xbrain.knowledge.evaluation import EmbeddingModelMismatch
    from xbrain.knowledge.index_build import load_manifest, manifest_spec

    item_id, query = _some_item(corpus)
    case = _case(id="ONE", query=query, strata=("exacto",), relevant_items=(item_id,))
    data = _vector_workspace(tmp_path, corpus)
    index_dir = tmp_path / "eval-index"
    evaluate([case], corpus, strategy="vector", vectors=_vectors(data, index_dir))

    with pytest.raises(EmbeddingModelMismatch, match="fake/model-a") as refused:
        evaluate(
            [case],
            corpus,
            strategy="vector",
            vectors=_vectors(data, index_dir, requested="fake/model-b"),
        )

    assert "fake/model-b" in str(refused.value)
    spec = manifest_spec(load_manifest(index_dir))
    assert spec is not None and spec.model == "fake/model-a", "nothing was rebuilt over it"


def test_the_evaluation_refuses_a_backend_serving_another_model_before_writing_anything(
    tmp_path: Path, corpus
) -> None:
    """TDD 23, the other door: `--embeddings-model B` answered by a backend that serves A.

    The backend declares its model on every batch and a build seals what it DECLARED, so
    without this the manifest would say one model and the report another, each internally
    consistent. Refused before a byte of the index exists.
    """
    from xbrain.knowledge.evaluation import EmbeddingModelMismatch

    item_id, query = _some_item(corpus)
    case = _case(id="ONE", query=query, strata=("exacto",), relevant_items=(item_id,))
    data = _vector_workspace(tmp_path, corpus)
    index_dir = tmp_path / "eval-index"

    with pytest.raises(EmbeddingModelMismatch, match="fake/served"):
        evaluate(
            [case],
            corpus,
            strategy="vector",
            vectors=_vectors(data, index_dir, requested="fake/asked", served="fake/served"),
        )
    assert not index_dir.exists()


def test_the_vector_index_is_built_once_timed_sized_and_then_reused(tmp_path: Path, corpus) -> None:
    """Plan 03 §3.2: full indexing time and the plane's size on disk, per candidate — and
    `vector` then `hybrid` over the SAME plane, never embedded twice.

    Without the reuse the protocol costs two corpus embeddings per candidate. The counter
    proves the embedder was not paid again; `built` says which run paid; the size is read off
    the two files on disk, not off a count someone multiplied.
    """
    from xbrain.knowledge.vector_index import VECTORS_FILENAME, VECTORS_META_FILENAME

    item_id, query = _some_item(corpus)
    case = _case(id="ONE", query=query, strata=("exacto",), relevant_items=(item_id,))
    data = _vector_workspace(tmp_path, corpus)
    index_dir = tmp_path / "eval-index"
    passages = _PassageEmbedder()

    first = evaluate(
        [case], corpus, strategy="vector", vectors=_vectors(data, index_dir, passages=passages)
    ).to_dict()
    paid = passages.texts
    second = evaluate(
        [case], corpus, strategy="hybrid", vectors=_vectors(data, index_dir, passages=passages)
    ).to_dict()

    assert paid > 0 and passages.texts == paid, "the second run re-embedded the corpus"
    assert first["indexing"]["built"] is True and second["indexing"]["built"] is False
    assert first["indexing"]["seconds"] > 0 and second["indexing"]["seconds"] is None
    on_disk = sum(
        (index_dir / name).stat().st_size for name in (VECTORS_FILENAME, VECTORS_META_FILENAME)
    )
    assert first["indexing"]["vector_bytes"] == on_disk == second["indexing"]["vector_bytes"]
    assert set(first["embeddings"]) == {
        "model",
        "dimension",
        "normalized",
        "query_prefix",
        "passage_prefix",
        "command_version",
    }
    assert first["embeddings"]["model"] == "fake/model-a"
    assert first["embeddings"]["passage_prefix"] == "passage: "
    assert "fake-embedder --offline" in first["embeddings"]["command_version"]


def test_the_vector_index_is_rebuilt_when_the_store_moved_under_it(tmp_path: Path, corpus) -> None:
    """Reuse is legal only over the store it was built from. The cheap signal moving is the
    trigger `search` already declares as `index_behind_store`; here it forces a rebuild,
    because a measurement over a stale plane is a measurement of a corpus that is gone."""
    import os

    item_id, query = _some_item(corpus)
    case = _case(id="ONE", query=query, strata=("exacto",), relevant_items=(item_id,))
    data = _vector_workspace(tmp_path, corpus)
    index_dir = tmp_path / "eval-index"
    passages = _PassageEmbedder()
    evaluate(
        [case], corpus, strategy="vector", vectors=_vectors(data, index_dir, passages=passages)
    )
    paid = passages.texts

    items = data / "items.json"
    stat = items.stat()
    os.utime(items, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    again = evaluate(
        [case], corpus, strategy="vector", vectors=_vectors(data, index_dir, passages=passages)
    ).to_dict()

    assert again["indexing"]["built"] is True
    assert passages.texts == 2 * paid


def test_a_vector_evaluation_times_the_query_embedding_apart_from_retrieval(
    tmp_path: Path, corpus
) -> None:
    """Spec §8.4's p50/p95 of a QUERY. In production `search` pays the embedder on every query,
    so that cost belongs in the latency — and is published beside retrieval, never folded in
    silently, because the two move for different reasons (a model against an index)."""
    item_id, query = _some_item(corpus)
    case = _case(id="ONE", query=query, strata=("exacto",), relevant_items=(item_id,))
    data = _vector_workspace(tmp_path, corpus)
    latency = evaluate(
        [case],
        corpus,
        strategy="vector",
        vectors=_vectors(data, tmp_path / "eval-index", query_delay=0.02),
    ).to_dict()["latency"]

    assert latency["embedding_p50_ms"] >= 20.0
    assert latency["p50_ms"] >= latency["embedding_p50_ms"]
    assert {"retrieval_p50_ms", "retrieval_p95_ms", "embedding_p95_ms", "p95_ms"} <= set(latency)


def test_ndcg_is_binary_and_published_beside_surface_recall(corpus) -> None:
    """Spec §8.4 asks for nDCG «when relevance grades exist». The golden set has no grades, so
    this is BINARY nDCG — a relevant owner gains 1, anything else 0 — which adds what MRR
    cannot see: the position of EVERY relevant owner, not only the first.

    Checked against an independent computation, with the relevant owner deliberately SECOND,
    so a scorer that returned 1.0 for «found» goes red.
    """
    import math

    retrieved = evaluate([_dominated_case()], corpus, ks=(10,)).cases[0].retrieved
    owner_type, owner_id = retrieved[1].split(":", 1)
    relevant = (
        {"relevant_items": (owner_id,)}
        if owner_type == "item"
        else {"relevant_topics": (owner_id,)}
    )
    case = _case(id="SECOND", query=OWNER_DOMINATED_QUERY, strata=("enterrado",), **relevant)

    result = evaluate([case], corpus, ks=(10,)).cases[0]
    assert result.retrieved[1] == retrieved[1], "precondition: the relevant owner is second"
    assert result.metrics["ndcg@10"] == pytest.approx(1 / math.log2(3))

    header = next(
        line
        for line in render_markdown(evaluate([case], corpus, ks=(1, 10))).splitlines()
        if line.startswith("| bucket |")
    )
    assert "nDCG@10" in header and "superficies@10" in header


def _synthetic(strategy: str, cases: dict[str, tuple[str, float]], unmeasured=()) -> dict:
    """A report payload with one metric pair per case — the shape `to_dict()` publishes."""
    return {
        "strategy": strategy,
        "cases": [
            {
                "id": case_id,
                "provenance": "construido",
                "strata": [stratum],
                "metrics": {"recall@10": value, "mrr@10": value, "mrr": value},
            }
            for case_id, (stratum, value) in cases.items()
        ],
        "unmeasured": [{"id": case_id} for case_id in unmeasured],
    }


def test_the_bakeoff_refuses_to_decide_a_stratum_with_no_enumerated_ground_truth(corpus) -> None:
    """TDD 28 (Plan 03 §7, B2): a stratum whose cases carry no enumerated relevant set is
    REFUSED by the bake-off — never read as flat, never as «does not degrade».

    `recall@k` over no enumerated owner is 0/0, and the harness already reports it unmeasured
    per case. What nothing stopped was the COMPARISON reading two unmeasured sides as equal:
    a decisive stratum with nothing in it would pass «hybrid mejora semántico» vacuously, or
    block it for no measured reason. So that stratum is `sin cobertura`, listed with its
    reason, and a gate that needs it cannot pass. The contrast strata are decided with their
    cases NAMED, so this is not a comparison that refuses everything.

    Seen red with the stratum means compared directly: `semantico` read `empata`.
    """
    from xbrain.knowledge.evaluation import compare_reports
    from xbrain.knowledge.goldenset import RelevantSurface
    from xbrain.knowledge.surfaces import item_surfaces

    item_id, query = _some_item(corpus)
    surface = item_surfaces(corpus.items[item_id])[0]
    only_surfaces = RelevantSurface(
        owner_type="item", owner_id=item_id, surface_type=surface.surface_type
    )
    cases = [
        _case(
            id="SEM-SURFACES-ONLY",
            query=query,
            strata=("semantico",),
            relevant_surfaces=(only_surfaces,),
        ),
        _case(
            id="CROSS-ENUMERATED",
            query=query,
            strata=("cruzado_idioma",),
            relevant_items=(item_id,),
        ),
        _case(id="EXACT-ENUMERATED", query=query, strata=("exacto",), relevant_items=(item_id,)),
    ]
    report = evaluate(cases, corpus, ks=(10,)).to_dict()

    verdict = compare_reports(report, report, k=10)

    assert verdict["strata"]["semantico"]["verdict"] == "sin cobertura"
    assert "enumerad" in verdict["rejected_strata"]["semantico"]
    assert verdict["strata"]["exacto"]["cases"] == ["EXACT-ENUMERATED"]
    assert verdict["strata"]["exacto"]["verdict"] == "empata"
    assert verdict["strata"]["cruzado_idioma"]["cases"] == ["CROSS-ENUMERATED"]
    assert verdict["passes_gates"] is False
    assert any("semantico" in reason for reason in verdict["reasons"])


def test_the_gates_pass_only_when_exact_holds_and_both_decisive_strata_improve() -> None:
    """Spec §8.6.3 and §8.6.4 as one decision, asserted in BOTH directions: a comparison that
    never passes is as useless as one that always does."""
    from xbrain.knowledge.evaluation import compare_reports

    lexical = _synthetic(
        "lexical", {"X": ("exacto", 1.0), "S": ("semantico", 0.0), "C": ("cruzado_idioma", 0.0)}
    )
    better = _synthetic(
        "hybrid", {"X": ("exacto", 1.0), "S": ("semantico", 1.0), "C": ("cruzado_idioma", 1.0)}
    )
    breaks_exact = _synthetic(
        "hybrid", {"X": ("exacto", 0.5), "S": ("semantico", 1.0), "C": ("cruzado_idioma", 1.0)}
    )
    flat_cross = _synthetic(
        "hybrid", {"X": ("exacto", 1.0), "S": ("semantico", 1.0), "C": ("cruzado_idioma", 0.0)}
    )

    assert compare_reports(lexical, better, k=10)["passes_gates"] is True
    refused = compare_reports(lexical, breaks_exact, k=10)
    assert refused["passes_gates"] is False
    assert refused["strata"]["exacto"]["verdict"] == "empeora"
    assert any("exacto" in reason for reason in refused["reasons"])
    flat = compare_reports(lexical, flat_cross, k=10)
    assert flat["passes_gates"] is False
    assert any("cruzado_idioma" in reason for reason in flat["reasons"])


def test_the_bakeoff_compares_only_cases_measured_on_both_sides_and_names_the_rest() -> None:
    """A case the candidate could not measure (a filter the plane cannot apply) is not paired:
    folding the baseline's value in would compare two populations under one stratum name.

    Both shapes of «not measured on one side»: `F` is absent from the candidate's cases, and
    `N` is present with no `recall@10`. The first version held only `F`, and a comparison that
    paired every id present on both sides passed it (seen green under that mutation)."""
    from xbrain.knowledge.evaluation import compare_reports

    lexical = _synthetic(
        "lexical", {"F": ("semantico", 1.0), "S": ("semantico", 0.0), "N": ("semantico", 1.0)}
    )
    candidate = _synthetic("vector", {"S": ("semantico", 1.0)}, unmeasured=("F",))
    candidate["cases"].append(
        {
            "id": "N",
            "provenance": "construido",
            "strata": ["semantico"],
            "metrics": {"recall@10": None, "mrr": None},
        }
    )

    stratum = compare_reports(lexical, candidate, k=10)["strata"]["semantico"]

    assert stratum["cases"] == ["S"]
    assert stratum["unpaired"] == ["F", "N"]
    assert stratum["baseline"]["recall@10"] == 0.0, "F's or N's 1.0 diluted the baseline"
    assert stratum["verdict"] == "mejora"


def test_the_bakeoff_excludes_a_named_case_with_its_reason_and_never_pairs_it() -> None:
    """Plan 03 §3.3-3.4: a case whose ground truth no longer verifies on THIS corpus does not
    decide — and is published by name with the reason, never dropped in silence.

    Measured on the live store the day the bake-off ran, the golden set had drifted under
    four cases (a fact moved off a re-synthesized topic note, a leak to newer articles, a
    population that grew). Filtering them out of the report files by hand would make the
    published verdict impossible to re-derive; the instrument takes the exclusions as an
    argument and carries them into its output.
    """
    from xbrain.knowledge.evaluation import compare_reports

    lexical = _synthetic("lexical", {"S": ("semantico", 0.0), "D": ("semantico", 1.0)})
    candidate = _synthetic("hybrid", {"S": ("semantico", 1.0), "D": ("semantico", 0.0)})

    verdict = compare_reports(
        lexical, candidate, k=10, exclude={"D": "la verdad de terreno se movió"}
    )
    stratum = verdict["strata"]["semantico"]

    assert stratum["cases"] == ["S"] and stratum["verdict"] == "mejora"
    assert stratum["excluded"] == {"D": "la verdad de terreno se movió"}
    assert verdict["excluded"] == {"D": "la verdad de terreno se movió"}


def test_mrr_at_k_reads_the_same_owner_prefix_as_recall_at_k(corpus) -> None:
    """PR #186, Codex F1: the bare `mrr` walks the WHOLE ranking the retriever returned, so a
    relevant owner past `k` still earns a reciprocal rank that `recall@k` cannot see. `mrr@k`
    is cut at the same prefix, and a prefix does not move with the depth (U-6).

    The relevant owner is deliberately SECOND: `mrr@1` must be 0.0 while the window MRR is 0.5.
    """
    retrieved = evaluate([_dominated_case()], corpus, ks=(10,)).cases[0].retrieved
    owner_type, owner_id = retrieved[1].split(":", 1)
    relevant = (
        {"relevant_items": (owner_id,)}
        if owner_type == "item"
        else {"relevant_topics": (owner_id,)}
    )
    case = _case(id="SECOND", query=OWNER_DOMINATED_QUERY, strata=("enterrado",), **relevant)

    shallow = evaluate([case], corpus, ks=(1, 10)).cases[0]
    deep = evaluate([case], corpus, ks=(1, 10), limit=20).cases[0]

    assert shallow.retrieved[1] == retrieved[1], "precondition: the relevant owner is second"
    assert shallow.metrics["recall@1"] == 0.0
    assert shallow.metrics["mrr@1"] == 0.0, "a rank past k leaked into the cut MRR"
    assert shallow.metrics["mrr@10"] == pytest.approx(0.5)
    assert shallow.metrics["mrr"] == pytest.approx(0.5)
    assert deep.metrics["mrr@1"] == shallow.metrics["mrr@1"]
    assert deep.metrics["mrr@10"] == shallow.metrics["mrr@10"]


def test_the_bakeoff_ranks_by_mrr_at_k_never_by_ranks_past_the_published_depth() -> None:
    """PR #186, Codex F1 — the shape of V1 in the published bake-off: found by neither strategy
    in the top 20, lexical window rank 51, hybrid window rank 56. Compared on the bare `mrr`
    that read 0.0196 → 0.0179 and published «hybrid empeora exacto»; those ranks are past the
    depth both reports publish and are cut by windows of different sizes. At `mrr@10` both
    sides measured 0.0 and the guardrail holds — seen red on the window MRR: `empeora`."""
    from xbrain.knowledge.evaluation import compare_reports

    def report(strategy: str, window_mrr: float) -> dict:
        return {
            "strategy": strategy,
            "limit": 20,
            "cases": [
                {
                    "id": "V1",
                    "provenance": "construido",
                    "strata": ["exacto"],
                    "metrics": {"recall@10": 0.0, "mrr@10": 0.0, "mrr": window_mrr},
                }
            ],
            "unmeasured": [],
        }

    verdict = compare_reports(report("lexical", 1 / 51), report("hybrid", 1 / 56), k=10)
    exacto = verdict["strata"]["exacto"]

    assert exacto["verdict"] == "empata", "a rank no report publishes decided the guardrail"
    assert verdict["rank_metric"] == "mrr@10"
    assert set(exacto["baseline"]) == {"recall@10", "mrr@10"}, "the window MRR is still read"
    assert not any(reason.startswith("exacto") for reason in verdict["reasons"])


def test_a_case_without_mrr_at_k_on_one_side_is_unpaired_never_read_as_zero() -> None:
    """The old comparison read a missing MRR as `or 0.0`: a report written before `mrr@k`
    existed would have paired on `recall@10` and lost its rank term as a measured zero."""
    from xbrain.knowledge.evaluation import compare_reports

    lexical = _synthetic("lexical", {"S": ("semantico", 0.0), "OLD": ("semantico", 1.0)})
    candidate = _synthetic("hybrid", {"S": ("semantico", 1.0), "OLD": ("semantico", 1.0)})
    del lexical["cases"][1]["metrics"]["mrr@10"]

    stratum = compare_reports(lexical, candidate, k=10)["strata"]["semantico"]

    assert stratum["cases"] == ["S"]
    assert stratum["unpaired"] == ["OLD"]
    assert "mrr@10" in stratum["unpaired_reasons"]["OLD"]
    assert "baseline `lexical`" in stratum["unpaired_reasons"]["OLD"]


def test_the_comparison_publishes_each_denominator_its_unit_and_why_a_member_stayed_out() -> None:
    """PR #186, Codex F2: every mean ships with the population it was divided by, the unit `k`
    counts, the depth of each report and, per member left out, WHY — unmeasured by one
    strategy, unmeasured by both, or excluded by argument. A case both strategies skipped used
    to vanish from the comparison instead of being named outside it."""
    from xbrain.knowledge.evaluation import compare_reports

    lexical = _synthetic(
        "lexical", {"S": ("semantico", 0.0), "T": ("semantico", 1.0), "D": ("semantico", 1.0)}
    )
    candidate = _synthetic("vector", {"S": ("semantico", 1.0), "D": ("semantico", 0.0)})
    lexical["limit"] = candidate["limit"] = 20
    candidate["unmeasured"] = [
        {"id": "T", "strata": ["semantico"], "reason": "la estrategia `vector` no puede aplicar"}
    ]
    for side in (lexical, candidate):
        side["unmeasured"].append({"id": "B", "strata": ["semantico"], "reason": "filtro"})

    verdict = compare_reports(lexical, candidate, k=10, exclude={"D": "fuga"})
    stratum = verdict["strata"]["semantico"]

    assert stratum["cases"] == ["S"]
    assert stratum["denominators"] == {"recall@10": 1, "mrr@10": 1}
    assert verdict["units"] == {"recall@10": "owners", "mrr@10": "owners"}
    assert verdict["depth"] == {"baseline": 20, "candidate": 20}
    assert stratum["unpaired"] == ["B", "T"]
    assert stratum["unpaired_reasons"]["T"].startswith("no medido en candidate `vector`")
    assert stratum["unpaired_reasons"]["B"].startswith("no medido en baseline `lexical`")
    assert stratum["excluded"] == {"D": "fuga"}


def test_the_markdown_publishes_each_denominator_the_unit_of_k_and_the_unmeasured_count(
    corpus,
) -> None:
    """PR #186, Codex F2, on the surface that gets quoted: `recall@10` over two cases and
    `superficies@10` over the one that names a surface sit in the same row, so each cell
    carries its own denominator, each column its unit, and the row the cases it left out."""
    from dataclasses import replace

    from xbrain.knowledge.goldenset import RelevantSurface
    from xbrain.knowledge.surfaces import item_surfaces

    item_id, query = _some_item(corpus)
    surface = RelevantSurface(
        owner_type="item",
        owner_id=item_id,
        surface_type=item_surfaces(corpus.items[item_id])[0].surface_type,
    )
    named = _case(
        id="NAMED",
        query=query,
        strata=("exacto",),
        relevant_items=(item_id,),
        relevant_surfaces=(surface,),
    )
    unnamed = _case(id="UNNAMED", query=query, strata=("exacto",), relevant_items=(item_id,))
    skipped = {
        "id": "FX",
        "strata": ["exacto"],
        "provenance": "construido",
        "unsupported_filters": ["source"],
        "reason": "no aplicable",
    }
    report = replace(evaluate([named, unnamed], corpus, ks=(1, 10)), unmeasured=(skipped,))

    lines = render_markdown(report).splitlines()
    columns = [
        cell.strip() for cell in next(ln for ln in lines if ln.startswith("| bucket |")).split("|")
    ]
    row = [
        cell.strip() for cell in next(ln for ln in lines if ln.startswith("| exacto |")).split("|")
    ]

    assert row[columns.index("no medidos")] == "1"
    assert row[columns.index("recall@10 [owners]")].endswith("(2)")
    assert row[columns.index("MRR@10 [owners]")].endswith("(2)")
    assert row[columns.index("superficies@10 [chunks]")].endswith("(1)")


def test_parse_fusion_sweep_reads_both_syntaxes_and_refuses_what_fuse_cannot_take() -> None:
    """Same syntax as `--sweep-chunker`, and the same refusal of a typo — plus the two values
    `fuse` cannot honour: `RRF_K < 1` divides by zero at rank one, a negative weight PENALISES
    being found."""
    from xbrain.knowledge.evaluation import parse_fusion_sweep

    assert parse_fusion_sweep(["rrf_k=10,60 w_vector=0.5,1"]) == {
        "rrf_k": [10, 60],
        "w_vector": [0.5, 1.0],
    }
    with pytest.raises(ValueError, match="desconocido"):
        parse_fusion_sweep(["rrf=10"])
    with pytest.raises(ValueError, match="rrf_k"):
        parse_fusion_sweep(["rrf_k=0"])
    with pytest.raises(ValueError, match="w_vector"):
        parse_fusion_sweep(["w_vector=-1"])


def test_the_fusion_sweep_scores_every_cell_over_one_plane_and_one_embedding_per_query(
    tmp_path: Path, corpus
) -> None:
    """Plan 03 §4.1 puts the sweep of `RRF_K` and the weights INSIDE this evaluation, and the
    delivery cut gives 03.7 the hunk that applies the winner — so the instrument ships, or the
    published winner cannot be re-derived (rule 2). It must cost ONE plane and ONE query
    embedding per scored case whatever the grid; otherwise a 12-cell sweep is 12 embeddings.
    """
    from xbrain.knowledge.evaluation import sweep_fusion, unsupported_filters

    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    scored = [case for case in cases if not unsupported_filters(case.filters, "hybrid")]
    data = _vector_workspace(tmp_path, corpus)
    passages, calls = _PassageEmbedder(), []
    report = sweep_fusion(
        cases,
        corpus,
        _vectors(data, tmp_path / "eval-index", passages=passages, query_calls=calls),
        {"rrf_k": [10, 60], "w_vector": [0.5, 1.0]},
    )

    assert len(report.rows) == 4
    assert sorted(calls) == sorted(case.query for case in scored), "one embedding per case"
    assert passages.texts > 0
    assert report.to_dict()["rows"][0].keys() >= {
        "rrf_k",
        "w_lexical",
        "w_vector",
        "recall@10",
        "mrr",
    }


def test_the_fusion_sweep_reaches_fuse_and_restores_the_constants_even_when_a_cell_fails(
    tmp_path: Path, corpus, monkeypatch
) -> None:
    """Two properties failing in opposite directions. `fusion` reads `RRF_K` and the weights
    at CALL time, which is what lets a sweep move them — so a sweep that never set them would
    score every cell alike, and one that forgot to restore them would leave every later fusion
    in the process running on the last cell's constants.

    Silencing bm25 against silencing the plane must change a ranking (the relevant item is the
    plane's answer and not bm25's), and after a cell RAISES the module holds what it held.
    """
    from xbrain.knowledge import fusion, search_service
    from xbrain.knowledge.evaluation import sweep_fusion

    before = (fusion.RRF_K, dict(fusion.CHANNEL_WEIGHTS))
    lexical_item, lexical_query = _some_item(corpus)
    vector_item = _other_item(corpus, lexical_item)
    case = _case(
        id="SPLIT", query=lexical_query, strata=("semantico",), relevant_items=(vector_item,)
    )
    data = _vector_workspace(tmp_path, corpus)
    vectors = _vectors(
        data, tmp_path / "eval-index", queries={lexical_query: _chunk_text(corpus, vector_item)}
    )

    report = sweep_fusion(
        [case], corpus, vectors, {"w_lexical": [0.0, 1.0], "w_vector": [0.0, 1.0]}
    )
    cells = {(row.w_lexical, row.w_vector): row.recall_at_1 for row in report.rows}
    assert cells[(0.0, 1.0)] == 1.0, "only the plane: its answer is first"
    assert cells[(1.0, 0.0)] == 0.0, "only bm25: the plane's answer is not first"
    assert (fusion.RRF_K, dict(fusion.CHANNEL_WEIGHTS)) == before

    real = search_service.fuse
    seen: list[int] = []

    def failing(rankings):
        seen.append(fusion.RRF_K)
        if len(seen) > 1:
            raise RuntimeError("cell failed")
        return real(rankings)

    monkeypatch.setattr(search_service, "fuse", failing)
    with pytest.raises(RuntimeError, match="cell failed"):
        sweep_fusion([case], corpus, vectors, {"rrf_k": [7, 9]})
    assert seen == [7, 9], "the constant reached `fuse` in each cell"
    assert (fusion.RRF_K, dict(fusion.CHANNEL_WEIGHTS)) == before


def test_a_flat_fusion_sweep_keeps_the_constants_in_force(tmp_path: Path, corpus) -> None:
    """Spec §13.15: a flat result is a RESULT. When every cell scores alike, the winner is the
    cell already in `fusion.py`, and the verdict says the sweep does not move it — so the
    hunk this child may apply to `fusion.py` is never applied on a tie.

    Flat BY CONSTRUCTION, not by luck of the fixture: the query has no word in the corpus, so
    bm25 contributes nothing and a one-channel RRF is monotone in that channel's rank — every
    `RRF_K` produces the same order."""
    from xbrain.knowledge import fusion
    from xbrain.knowledge.evaluation import render_fusion_sweep_markdown, sweep_fusion

    item_id, _query = _some_item(corpus)
    case = _case(
        id="FLAT", query=NO_OVERLAP_QUERY, strata=("semantico",), relevant_items=(item_id,)
    )
    data = _vector_workspace(tmp_path, corpus)
    vectors = _vectors(
        data, tmp_path / "eval-index", queries={NO_OVERLAP_QUERY: _chunk_text(corpus, item_id)}
    )

    report = sweep_fusion(
        [case], corpus, vectors, {"rrf_k": [fusion.RRF_K + 1, fusion.RRF_K, fusion.RRF_K - 1]}
    )

    assert report.winner is not None and report.winner.rrf_k == fusion.RRF_K
    assert report.moves is False
    assert "no mueve" in render_fusion_sweep_markdown(report)
