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


def test_a_case_whose_filters_the_strategy_cannot_apply_is_not_scored(corpus) -> None:
    """The fabricated zero this harness exists to prevent, caught on the real corpus.

    The lexical baseline pushes only `has_surfaces` and `origins` into `WHERE`. It has no
    date, source or content-kind filtering — those need columns Plan 02 has to build (spec
    §7.2 says so of `content_kinds` and `has_surfaces` explicitly). Scoring F1 and F2 anyway
    produced `filtros: recall@10 = 0.0` in the first real-corpus run, which reads as "the
    retriever failed at filtering" when the truth is that the instrument does not exist yet.

    Spec §8.6.8: *failures and skips are published; zeros are never fabricated by mixing in
    unmeasured cases.* So an unsupported filter makes the case UNMEASURED — listed with the
    filters that caused it — and its stratum reports no coverage rather than a zero.
    """
    cases = resolve_cases(load_cases(FIXTURE_GOLDEN), corpus.items)
    payload = evaluate(cases, corpus, strategy="lexical").to_dict()

    unmeasured = {entry["id"]: entry for entry in payload["unmeasured"]}
    assert "FX7" in unmeasured, "FX7 declares `source: own_tweet`, which lexical cannot apply"
    assert unmeasured["FX7"]["unsupported_filters"] == ["source"]
    assert payload["by_stratum"]["filtros"] == NO_COVERAGE
    assert "FX7" not in {case["id"] for case in payload["cases"]}


def test_supported_filters_are_still_applied_not_skipped(corpus) -> None:
    """`has_surfaces` and `origins` ARE pushed down, so a case using them still scores.

    Without this, "unsupported" would be a way to quietly stop measuring anything awkward.
    """
    from xbrain.knowledge.contracts import SearchFilters
    from xbrain.knowledge.evaluation import unsupported_filters

    assert unsupported_filters(SearchFilters(has_surfaces=("post",)), "lexical") == ()
    assert unsupported_filters(SearchFilters(origins=("vlm",)), "lexical") == ()
    assert unsupported_filters(SearchFilters(source="own_tweet"), "lexical") == ("source",)


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
    """A gate that fails a bucket on a metric nobody measured is the fabricated zero again.

    `_failures` read `values.get(metric, 0.0)`; with the metric carrying the sentinel that
    default would compare a dict against a float, or (with a plain 0.0) name a failure that
    measured nothing.
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
    assert report.failures == (), f"named a failure over an unmeasured metric: {report.failures}"


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
    from xbrain.knowledge.lexical_memory import InMemoryLexicalIndex

    from tests.test_knowledge_lexical_memory import _corpus_chunks

    chunks = _corpus_chunks()
    index = InMemoryLexicalIndex()
    assert index.add(chunks) == len(chunks)
    assert index.add(chunks) == 0, "a duplicate chunk_id must be refused, and countable"
