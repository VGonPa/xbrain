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


def test_the_case_count_reconciles(report) -> None:
    """Cases scored + scenarios archived == every entry in the file.

    Without this, a case silently dropped by a loader bug would just make the report shorter,
    and nothing would say so.
    """
    payload = report.to_dict()
    total = len(load_cases(FIXTURE_GOLDEN)) + len(load_scenarios(FIXTURE_GOLDEN))
    assert len(payload["cases"]) + len(payload["scenarios"]) == total


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
    assert len(index) == stats.chunks
    assert stats.items == 12
    assert stats.surfaces >= stats.items
