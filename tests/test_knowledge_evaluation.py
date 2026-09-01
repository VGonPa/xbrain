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
    version of the guardrail below drove the branch with `strategy="vector"`, which made the
    test depend on `vector` staying UNIMPLEMENTED: the day Plan 03 lands the vector backend,
    a guardrail about fabricated zeros would go red for a reason that has nothing to do with
    fabricated zeros, and the comfortable fix would be to delete it.

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
    of nothing (rule 1). Driving it with `vector` was the same mistake one level up: it
    borrowed a REAL entry of the frozen literal and made this guardrail's survival depend on
    that entry staying unimplemented, i.e. on Plan 03 not happening.

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


def test_the_guardrail_no_longer_depends_on_vector_being_unimplemented(corpus) -> None:
    """The coupling F-2 named, asserted so it cannot come back silently.

    `vector` IS declared in the frozen `Strategy` literal, so the day it has a backend it is
    added to `IMPLEMENTED_STRATEGIES` and this simulates that day. The guardrail above must
    still hold — it uses an injected stub — and `evaluate(strategy="vector")` must stop
    reporting the filter cases as unmeasurable, because a real vector backend that declares
    all eight filters can apply them.
    """
    monkeypatch_free = frozenset({"lexical", "vector"})
    original_implemented = contracts.IMPLEMENTED_STRATEGIES
    original_filters = dict(evaluation.SUPPORTED_FILTERS)
    contracts.IMPLEMENTED_STRATEGIES = monkeypatch_free  # type: ignore[misc]
    evaluation.SUPPORTED_FILTERS["vector"] = frozenset(SearchFilters.model_fields)
    try:
        cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
        payload = evaluate(cases, corpus, strategy="vector").to_dict()
        assert payload["strategy"] == "vector"
        assert payload["unmeasured"] == []
        assert payload["degraded"] == []
    finally:
        contracts.IMPLEMENTED_STRATEGIES = original_implemented  # type: ignore[misc]
        evaluation.SUPPORTED_FILTERS.clear()
        evaluation.SUPPORTED_FILTERS.update(original_filters)


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


def test_every_implemented_strategy_declares_which_filters_it_can_push(corpus) -> None:
    """Rule 5: the two tables that must agree are asserted to agree, not hoped to.

    `IMPLEMENTED_STRATEGIES` says which retrievers run; `SUPPORTED_FILTERS` says what each
    can push into `WHERE`. A strategy implemented without an entry here would fall to
    `SUPPORTED_FILTERS.get(strategy, frozenset())` and report every filtered case as
    UNMEASURED — the gap silently reopening under a strategy that works.
    """
    assert set(evaluation.SUPPORTED_FILTERS) == set(contracts.IMPLEMENTED_STRATEGIES)


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


def test_the_supported_filter_set_is_DERIVED_from_the_frozen_contract(corpus) -> None:
    """All eight, and taken from `SearchFilters` rather than written out a second time.

    A hand-written list here would be a second copy of the contract, and the day a ninth
    filter is added to `SearchFilters` the copy would silently keep declaring eight — the
    case would be scored with a filter nobody applied, which is the fabricated zero coming
    back through the door marked "supported".
    """
    from xbrain.knowledge.contracts import SearchFilters
    from xbrain.knowledge.evaluation import SUPPORTED_FILTERS, unsupported_filters

    assert SUPPORTED_FILTERS["lexical"] == frozenset(SearchFilters.model_fields)
    assert len(SUPPORTED_FILTERS["lexical"]) == 8
    for name in SearchFilters.model_fields:
        value = {"source": "own_tweet", "author": "x"}.get(
            name, ("post",) if name in {"has_surfaces"} else None
        )
        if value is None:
            continue
        assert unsupported_filters(SearchFilters(**{name: value}), "lexical") == ()


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
# The chunker sweep (Plan 02 §7, steps 17b)
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


def test_evaluate_closes_the_index_it_built(corpus) -> None:
    """M-1 (round 02, Codex F-08): `evaluate` built its `:memory:` index through
    `build_index` and returned the report with the connection still open — and
    `sweep_chunker` calls `evaluate` once per combination, so a twelve-cell sweep leaked
    twelve handles and the gate printed `ResourceWarning: unclosed database` for each.

    Seen red before the fix: one `unclosed database` warning per `evaluate`.
    """
    import gc
    import warnings

    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    # Reap what EARLIER tests left behind first, so the measured window holds only this
    # call's connections — in the full suite the first version caught a neighbour's leak.
    gc.collect()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        evaluate(cases, corpus)
        gc.collect()
    leaks = [
        w
        for w in caught
        if issubclass(w.category, ResourceWarning) and "unclosed database" in str(w.message)
    ]
    assert not leaks, [str(w.message) for w in leaks]
