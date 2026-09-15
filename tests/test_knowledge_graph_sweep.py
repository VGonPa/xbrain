"""Plan 04.5 — the graph threshold sweep (Plan 04 §1.3, spec §14).

*«umbral de coocurrencia del grafo mínimo: se mide contra expansión útil/ruido»*. The sweep runs
`min_shared_items × min_weight`, and for every cell publishes the co-occurrence edges, the mean
degree and the recall delta of `hybrid_graph` against `hybrid`, plus the two NOISE measures of
spec §8.4 — the precision of the candidates only the graph brought into the page, and how many
places the direct results lost. Then it fixes a winner and publishes every cell, including the
ones that did not contribute.

THE RULE IS FIXED BEFORE THE NUMBERS, and the tests below pin it on constructed rows so a result
cannot move it: a cell that loses MORE than 3 pp of `precision@k` in any stratum is rejected
(Plan 04 §3 defines «materialmente» before measuring); then the recall delta decides; then less
noise; then less degradation; then the sparser graph; and between identical graphs, the least
restrictive thresholds. When no cell survives the guard, the least damaging one is still applied
— the index always builds a graph — and the verdict says, in words, that none contributes.

THE GRAPH IS MEASURED THROUGH `search`, the one door `hybrid_graph` exists in (rule 5): a second
implementation of the strategy inside the harness would be a second definition of what it serves.
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.test_knowledge_search_service import _persist
from xbrain.knowledge import evaluation, graph_build, graph_service, graph_strategy, index_build
from xbrain.knowledge.contracts import GraphEdge, SearchFilters
from xbrain.knowledge.evaluation import GraphSweepReport, GraphSweepRow
from xbrain.knowledge.goldenset import GoldenCase
from xbrain.knowledge.search_service import QueryContext, search
from xbrain.models import Author, Content, Enrichment, Item, Topic

REPO = Path(__file__).resolve().parents[1]
_T = datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def test_parse_graph_sweep_reads_both_axes_and_refuses_what_graph_build_cannot_take() -> None:
    grid = evaluation.parse_graph_sweep(["min_shared_items=2,3 min_weight=0.0,0.05"])
    assert grid == {"min_shared_items": [2, 3], "min_weight": [0.0, 0.05]}
    assert evaluation.parse_graph_sweep(["min_shared_items=5", "min_weight=0.1"]) == {
        "min_shared_items": [5],
        "min_weight": [0.1],
    }

    # A typo that swept nothing would publish the thresholds in force as a winner.
    with pytest.raises(ValueError, match="min_weigth"):
        evaluation.parse_graph_sweep(["min_weigth=0.1"])
    # `build_graph_edges` treats < 1 as 1: sweeping 0 would measure the cell 1 under another name.
    with pytest.raises(ValueError, match="min_shared_items=0"):
        evaluation.parse_graph_sweep(["min_shared_items=0,2"])
    # A Jaccard index lives in [0, 1]: outside it a cell keeps everything or nothing.
    with pytest.raises(ValueError, match="min_weight=1.5"):
        evaluation.parse_graph_sweep(["min_weight=1.5"])
    with pytest.raises(ValueError, match="min_weight=-0.1"):
        evaluation.parse_graph_sweep(["min_weight=-0.1"])


# ---------------------------------------------------------------------------
# The winner rule, on constructed rows — so no measurement can move it
# ---------------------------------------------------------------------------


def _row(**overrides: object) -> GraphSweepRow:
    values: dict[str, object] = {
        "min_shared_items": 2,
        "min_weight": 0.0,
        "edges": 40,
        "mean_degree": 4.0,
        "recall": 0.5,
        "recall_delta": 0.0,
        "entrants": 0,
        "useful": 0,
        "degradation": 0.0,
        "precision_drops": {},
        "graph_ran": True,
    }
    values.update(overrides)
    return GraphSweepRow(**values)  # type: ignore[arg-type]


def _cells(rows: tuple[GraphSweepRow, ...]) -> list[tuple[int, float]]:
    return [(row.min_shared_items, row.min_weight) for row in rows]


def test_a_cell_losing_more_than_3pp_of_precision_in_any_stratum_is_rejected_before_recall() -> (
    None
):
    """Plan 04 §3: a fall of `precision@10` GREATER than 3 pp in any stratum is material.

    The rejected cell has the best recall of the two, so a rule that read recall first would
    pick it. And the boundary is strict: exactly 3 pp is not «more than 3».
    """
    noisy = _row(min_shared_items=2, recall_delta=0.2, precision_drops={"exacto": 3.5})
    clean = _row(min_shared_items=5, recall_delta=0.0, precision_drops={"exacto": 3.0})

    ranked = evaluation.rank_graph_rows((noisy, clean))

    assert _cells(ranked) == [(5, 0.0), (2, 0.0)]
    assert ranked[1].rejected_for and "exacto" in ranked[1].rejected_for[0]
    assert ranked[0].rejected_for == ()


@pytest.mark.parametrize(
    ("better", "worse", "criterion"),
    [
        ({"recall_delta": 0.1}, {"recall_delta": 0.0, "min_shared_items": 1}, "recall delta"),
        (
            {"entrants": 3, "useful": 2},
            {"entrants": 3, "useful": 1, "min_shared_items": 1},
            "noise",
        ),
        ({"degradation": 0.5}, {"degradation": 1.5, "min_shared_items": 1}, "degradation"),
        ({"edges": 20}, {"edges": 40, "min_shared_items": 1}, "sparser graph"),
    ],
)
def test_among_eligible_cells_each_criterion_decides_in_its_order(
    better: dict, worse: dict, criterion: str
) -> None:
    """Each pair differs in ONE criterion, and the worse row carries the lower thresholds, so a
    rule that skipped the criterion would fall through to the threshold tie-break and pick it."""
    good = _row(min_shared_items=8, **better)
    bad = _row(**{"min_shared_items": 1, **worse})

    ranked = evaluation.rank_graph_rows((bad, good))

    assert ranked[0] is good, criterion


def test_recall_outranks_noise_and_noise_outranks_degradation() -> None:
    """The ORDER of the criteria, not just their presence: each winner loses every later one."""
    useful_but_noisy = _row(min_shared_items=8, recall_delta=0.1, entrants=9, useful=1)
    quiet = _row(min_shared_items=2, recall_delta=0.0, entrants=0, useful=0)
    assert evaluation.rank_graph_rows((quiet, useful_but_noisy))[0] is useful_but_noisy

    quiet_but_displacing = _row(min_shared_items=8, entrants=1, useful=1, degradation=3.0)
    noisy = _row(min_shared_items=2, entrants=2, useful=1, degradation=0.0)
    assert evaluation.rank_graph_rows((noisy, quiet_but_displacing))[0] is quiet_but_displacing


def test_identical_graphs_keep_the_least_restrictive_thresholds() -> None:
    """Two cells that persisted the same graph measured the same thing: the constraint that
    changed nothing is not applied (Plan 04 §1.3 fixes the threshold, not a superfluous one)."""
    strict = _row(min_shared_items=8, min_weight=0.10, edges=52)
    loose = _row(min_shared_items=2, min_weight=0.10, edges=52)
    looser_weight = _row(min_shared_items=2, min_weight=0.05, edges=52)

    ranked = evaluation.rank_graph_rows((strict, loose, looser_weight))

    assert _cells(ranked) == [(2, 0.05), (2, 0.10), (8, 0.10)]


def test_when_every_cell_degrades_the_least_damaging_is_applied_and_said_to_contribute_nothing() -> (
    None
):
    """The index always builds a graph, so a threshold is always applied — but a cell that hurts
    less is not a cell that helps, and the verdict must not read as one (spec §13.15)."""
    rows = (
        _row(min_shared_items=2, recall_delta=-0.17, entrants=62, precision_drops={"semantico": 8}),
        _row(min_shared_items=5, recall_delta=-0.12, entrants=94, precision_drops={"semantico": 6}),
    )
    report = GraphSweepReport(k=10, limit=10, rows=evaluation.rank_graph_rows(rows))

    assert report.winner is not None and report.winner.min_shared_items == 5
    assert report.useful is False
    verdict = report.to_dict()["verdict"]
    assert verdict.startswith("NINGUNA COMBINACIÓN APORTA")
    assert "min_shared_items=5" in verdict
    assert "NO se promueve" in verdict
    assert evaluation.render_graph_sweep_markdown(report).splitlines()[-1] == verdict


def test_a_useful_winner_is_named_with_its_delta_and_is_the_only_promotable_verdict() -> None:
    rows = (_row(min_shared_items=3, recall_delta=0.25, entrants=2, useful=2), _row())
    report = GraphSweepReport(k=10, limit=10, rows=evaluation.rank_graph_rows(rows))

    assert report.useful is True
    verdict = report.to_dict()["verdict"]
    assert verdict.startswith("Gana min_shared_items=3, min_weight=0.0")
    assert "+0.2500" in verdict


def test_a_cell_where_the_graph_did_not_run_never_wins_and_a_sweep_of_them_has_no_winner() -> None:
    """A row whose `hybrid_graph` degraded to another strategy measured that strategy, not the
    threshold: it ranks last whatever its numbers, and a table of them is not a ranking."""
    ghost = _row(min_shared_items=2, recall_delta=0.5, graph_ran=False)
    real = _row(min_shared_items=5, recall_delta=0.0)
    ranked = evaluation.rank_graph_rows((ghost, real))
    assert ranked[0] is real
    assert ranked[1].rejected_for and "no corrió" in ranked[1].rejected_for[0]

    empty = GraphSweepReport(k=10, limit=10, rows=evaluation.rank_graph_rows((ghost,)))
    assert empty.winner is None
    assert empty.to_dict()["verdict"].startswith("SIN MEDICIÓN")
    assert empty.to_dict()["winner"] is None


# ---------------------------------------------------------------------------
# The sweep, end to end, over a REAL index and `search` — no doubles (rule 3)
# ---------------------------------------------------------------------------


def _item(item_id: str, *, text: str, primary: str | None, topics: tuple[str, ...] = ()) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/u/status/{item_id}",
        author=Author(handle="u", name="U"),
        text=f"tweet {item_id}",
        created_at=_T,
        captured_at=_T,
        content=Content.model_validate(
            {
                "fetched_at": _T.isoformat(),
                "sources": [
                    {
                        "outcome": "success",
                        "kind": "x_article",
                        "url": f"https://x.com/i/article/{item_id}",
                        "text": text,
                        "attempts": 1,
                        "title": f"Article {item_id}",
                    }
                ],
            }
        ),
        enriched=Enrichment(
            enriched_at=_T,
            executor="manual",
            summary="s",
            primary_topic=primary,
            topics=list(topics),
        ),
    )


def _zeta(filler: int) -> str:
    """Matches `zeta` once; more `filler` = a longer body = a lower bm25."""
    return "zeta " + " ".join(["relleno"] * (60 + filler))


def _threshold_sensitive_store() -> tuple[dict[str, Item], list[Topic]]:
    """A store where the co-occurrence threshold decides whether the graph reaches `r15`.

    `s01` (lexical 1st) and `r15` (lexical 15th, the relevant item) share the topic `hub`. Ten
    topics `t0..t9` each share TWO carrier items with `hub` (Jaccard 2/22), and the carriers never
    match the query. `graph_expand` serves a topic's co-occurrence edges before its item
    assignments, inside one budget of `max_neighbors_per_node` (10): with those ten edges kept the
    budget is spent on topics and `r15` is never reached; pruned, `r15` is the first item reached
    and the graph lifts it into the top 10.
    """
    store = {"s01": _item("s01", text=_zeta(1), primary="hub")}
    for n in range(2, 15):
        store[f"f{n:02d}"] = _item(f"f{n:02d}", text=_zeta(n), primary=None)
    store["r15"] = _item("r15", text=_zeta(15), primary="hub")
    for i in range(10):
        for j in range(2):
            carrier = f"c{i}{j}"
            store[carrier] = _item(carrier, text="omega", primary=f"t{i}", topics=("hub",))
    slugs = ["hub", *(f"t{i}" for i in range(10))]
    return store, [Topic(slug=slug, description=f"topic {slug}") for slug in slugs]


_CASE = GoldenCase(
    id="G1",
    query="zeta",
    provenance="construido",
    strata=("expansion",),
    filters=SearchFilters(),
    relevant_items=("r15",),
)


def _workspace(tmp_path: Path) -> Path:
    store, vocab = _threshold_sensitive_store()
    data = tmp_path / "data"
    _persist(data, store, vocab, {})
    return data


def _sweep(data: Path, grid: dict, cases=(_CASE,), **kwargs) -> GraphSweepReport:
    return evaluation.sweep_graph(
        cases,
        grid,
        items_path=data / "items.json",
        vocab_path=data / "vocab.yaml",
        topics_path=data / "topics.json",
        index_dir=data / "eval-index" / "graph-sweep",
        **kwargs,
    )


def test_the_fixture_is_what_it_says_r15_ranks_15th_lexically(tmp_path: Path) -> None:
    """Rule 1: if `r15` were already in the lexical top 10, a delta of 0 would prove nothing."""
    data = _workspace(tmp_path)
    index_build.build(data / "index", index_build.load_index_inputs(*(data / n for n in _INPUTS)))
    store, vocab = _threshold_sensitive_store()
    context = QueryContext(
        store=store,
        vocab=vocab,
        index_dir=data / "index",
        items_path=data / "items.json",
        vocab_path=data / "vocab.yaml",
        topics_path=data / "topics.json",
    )
    ranking = [r.item_id for r in search("zeta", context, limit=20).results]
    assert ranking.index("r15") + 1 == 15


_INPUTS = ("items.json", "vocab.yaml", "topics.json")


def test_the_graph_sweep_scores_every_cell_through_search_and_the_threshold_decides(
    tmp_path: Path,
) -> None:
    data = _workspace(tmp_path)

    report = _sweep(data, {"min_shared_items": [2, 3], "min_weight": [0.0]})

    rows = {(row.min_shared_items, row.min_weight): row for row in report.rows}
    assert set(rows) >= {(2, 0.0), (3, 0.0)}
    kept, pruned = rows[(2, 0.0)], rows[(3, 0.0)]
    # The EDGES the cell persisted: ten hub↔tᵢ pairs, both directions; none past 2 shared items.
    assert (kept.edges, pruned.edges) == (20, 0)
    assert kept.recall_delta == pytest.approx(0.0)
    assert pruned.recall_delta == pytest.approx(1.0)
    assert (pruned.entrants, pruned.useful, pruned.noise) == (1, 1, 0)
    assert pruned.entrant_precision == pytest.approx(1.0)
    assert pruned.degradation is not None and pruned.degradation > 0
    assert all(row.graph_ran for row in report.rows)

    assert report.winner is pruned
    assert report.useful is True
    # The instrument that ranked the cells, named on the report (F-2): with no embedder the
    # base is `hybrid` answered lexically, and the graph re-ranks that same ranking.
    assert report.base["requested_strategy"] == "hybrid"
    assert report.base["strategy"] == "lexical"
    assert "embeddings_not_configured" in report.base["degraded"]
    assert report.graph["strategy"] == "hybrid_graph"
    assert report.base["recall"] == pytest.approx(0.0)


def test_the_graph_sweep_never_writes_the_store_nor_the_search_index(tmp_path: Path) -> None:
    data = _workspace(tmp_path)
    before = {name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in _INPUTS}

    _sweep(data, {"min_shared_items": [2], "min_weight": [0.0]})

    after = {name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in _INPUTS}
    assert after == before
    assert not (data / "index").exists()
    assert (data / "eval-index" / "graph-sweep").is_dir()


def test_the_graph_sweep_declares_the_cases_it_cannot_measure_instead_of_scoring_them(
    tmp_path: Path,
) -> None:
    """`search` serves ITEMS, so a case whose truth is a topic has a 0/0 recall; and the vector
    plane has no filter columns, so a filtered case is unmeasured under `hybrid_graph` (spec
    §8.6.8). Both are named with their reason, and neither enters a mean."""
    data = _workspace(tmp_path)
    topic_only = GoldenCase(
        id="T1",
        query="zeta",
        provenance="construido",
        strata=("topic",),
        filters=SearchFilters(),
        relevant_topics=("hub",),
    )
    filtered = GoldenCase(
        id="F1",
        query="zeta",
        provenance="construido",
        strata=("filtros",),
        filters=SearchFilters(source="bookmark"),
        relevant_items=("r15",),
    )

    report = _sweep(
        data, {"min_shared_items": [3], "min_weight": [0.0]}, cases=(_CASE, topic_only, filtered)
    )

    unmeasured = {entry["id"]: entry["reason"] for entry in report.unmeasured}
    assert set(unmeasured) == {"T1", "F1"}
    assert "topic" in unmeasured["T1"]
    assert "source" in unmeasured["F1"]
    assert report.to_dict()["measured_cases"] == ["G1"]


def test_each_cell_is_measured_on_the_graph_its_own_manifest_seals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep that re-derived nothing between cells would publish a flat table of ONE graph
    under sixteen names. So every cell reads its thresholds back off the manifest it measured."""
    data = _workspace(tmp_path)

    def stuck(index_dir, inputs, *, options=None, dry_run=False):
        return None  # the plane stays the one the first build wrote

    monkeypatch.setattr(index_build, "update", stuck)

    with pytest.raises(ValueError, match="min_shared_items=3"):
        _sweep(data, {"min_shared_items": [2, 3], "min_weight": [0.0]})


def test_the_cell_in_force_is_always_measured_and_marked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """«The sweep moves the default» is only a statement against a MEASURED current: the cell in
    force joins the grid when the grid omits it (the fusion sweep's rule)."""
    monkeypatch.setattr(graph_build, "DEFAULT_GRAPH_MIN_SHARED_ITEMS", 3)
    monkeypatch.setattr(graph_build, "DEFAULT_GRAPH_MIN_WEIGHT", 0.0)
    data = _workspace(tmp_path)

    report = _sweep(data, {"min_shared_items": [2], "min_weight": [0.0]})

    in_force = [row for row in report.rows if row.in_force]
    assert [(row.min_shared_items, row.min_weight) for row in in_force] == [(3, 0.0)]
    assert report.winner is in_force[0]
    assert report.moves is False


def test_the_graph_sweep_artefacts_publish_every_cell_the_rule_and_the_retriever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cell in force is pinned INSIDE the grid, so the table holds exactly the two cells named
    # below whatever default the sweep last applied (the in-force append has its own test).
    monkeypatch.setattr(graph_build, "DEFAULT_GRAPH_MIN_SHARED_ITEMS", 2)
    monkeypatch.setattr(graph_build, "DEFAULT_GRAPH_MIN_WEIGHT", 0.0)
    data = _workspace(tmp_path)
    report = _sweep(data, {"min_shared_items": [2, 3], "min_weight": [0.0]})

    payload = report.to_dict()
    assert payload["winner"] == {"min_shared_items": 3, "min_weight": 0.0}
    assert payload["material_precision_drop_pp"] == 3.0
    assert {(row["min_shared_items"], row["min_weight"]) for row in payload["rows"]} >= {
        (2, 0.0),
        (3, 0.0),
    }
    assert payload["rows"][0].keys() >= {
        "edges",
        "mean_degree",
        "recall@10",
        "recall_delta",
        "entrants",
        "useful",
        "noise",
        "entrant_precision",
        "degradation",
        "precision_drops",
        "rejected_for",
        "graph_ran",
        "in_force",
    }

    markdown = evaluation.render_graph_sweep_markdown(report)
    lines = markdown.splitlines()
    assert lines[0].startswith("Recuperador: `hybrid_graph` sobre `lexical`")
    # EVERY cell is a table row, the losing one included (Plan 04 §1.3).
    table = [line for line in lines if re.match(r"^\| \d+ \| ", line)]
    assert [line.split("|")[1:3] for line in table] == [[" 3 ", " 0.0 "], [" 2 ", " 0.0 "]]
    assert lines[-1] == payload["verdict"]


# ---------------------------------------------------------------------------
# The expansion stratum (Plan 04 §11.9): the pairs only the graph COULD bring in
# ---------------------------------------------------------------------------


def _context(data: Path) -> QueryContext:
    store, vocab = _threshold_sensitive_store()
    return QueryContext(
        store=store,
        vocab=vocab,
        index_dir=data / "index",
        items_path=data / "items.json",
        vocab_path=data / "vocab.yaml",
        topics_path=data / "topics.json",
    )


def _build(data: Path, min_shared_items: int, *, force: bool = False) -> None:
    inputs = index_build.load_index_inputs(*(data / n for n in _INPUTS))
    options = index_build.IndexOptions(
        graph_min_shared_items=min_shared_items, graph_min_weight=0.0
    )
    if force:
        index_build.build(data / "index", inputs, options=options, force=True)
    else:
        index_build.update(data / "index", inputs, options=options)


def test_every_relevant_pair_is_classified_by_the_route_that_can_reach_it(tmp_path: Path) -> None:
    """Spec §8.2: `expansión` = «relevante accesible mediante el grafo mínimo» — and, Plan 04 §3,
    NOT directly. So each relevant item of a measured case is one of four things, relative to the
    ranking `hybrid_graph` re-ranks and the seed it expands from (`s01`, the top hit):

    - `s01`, `f02`: already in the top 10 — the graph cannot add what is there (`direct`);
    - `r15`: 15th, and shares `hub` with the seed — only the graph can lift it (`graph_reachable`);
    - `c00`: shares `hub` too, but no channel scored it, and the graph never admits the unscored
      (§3.4) — reachable, never liftable (`graph_unscored`);
    - `f14`: 14th, with no topic at all — no route but the direct one, and it missed (`unreachable`).

    THE NEIGHBOUR BUDGET IS NOT PART OF IT: at `2 / 0.0` the ten topic edges spend `hub`'s budget
    and the walk never reaches `r15`, and the pair is still reachable — the budget is what the
    sweep measures, so a classification that read it would call a failure of the graph «no
    coverage». Proven by classifying under both thresholds.
    """
    data = _workspace(tmp_path)
    case = GoldenCase(
        id="E1",
        query="zeta",
        provenance="construido",
        strata=("expansion",),
        filters=SearchFilters(),
        relevant_items=("s01", "f02", "r15", "c00", "f14"),
    )
    expected = {
        ("s01", "direct", 1),
        ("f02", "direct", 2),
        ("r15", "graph_reachable", 15),
        ("c00", "graph_unscored", None),
        ("f14", "unreachable", 14),
    }

    _build(data, 2, force=True)
    kept = evaluation.classify_expansion((case,), _context(data), k=10)
    _build(data, 3)
    pruned = evaluation.classify_expansion((case,), _context(data), k=10)

    assert {(pair.item_id, pair.kind, pair.rank) for pair in kept} == expected
    assert pruned == kept
    assert {pair.case_id for pair in kept} == {"E1"}


_E1 = GoldenCase(
    id="E1",
    query="zeta",
    provenance="construido",
    strata=("expansion",),
    filters=SearchFilters(),
    relevant_items=("s01", "f02", "r15", "c00", "f14"),
)


def _drop_reached(node_id: str, edges: list[GraphEdge]) -> list[GraphEdge]:
    return [edge for edge in edges if not {edge.source, edge.target} & {"item:r15", "item:c00"}]


def _add_unreached(node_id: str, edges: list[GraphEdge]) -> list[GraphEdge]:
    # `f14` has no topic: an assignment edge arriving at a topic, re-pointed, hangs it off `hub`.
    arriving = [edge for edge in edges if edge.target == node_id]
    return [*edges, *(edge.model_copy(update={"source": "item:f14"}) for edge in arriving[:1])]


def _reached_relevant(context: QueryContext) -> set[str]:
    """What `graph_expand` itself reaches of E1's relevant items from its seed (`s01`, the top hit)."""
    walk = graph_service.graph_expand(
        ("item:s01",),
        context,
        max_hops=graph_strategy.GRAPH_MAX_HOPS,
        max_neighbors_per_node=len(context.store) + len(context.vocab),
    )
    return {node.node_id.removeprefix("item:") for node in walk.nodes} & set(_E1.relevant_items)


@pytest.mark.parametrize(
    "rewrite",
    [_drop_reached, _add_unreached, lambda node_id, edges: []],
    ids=["drop-reached-r15-c00", "add-unreached-f14", "reach-nothing"],
)
def test_the_expansion_stratum_never_reads_the_graph_service_whose_lift_it_measures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rewrite
) -> None:
    """PR #198, round 2, HIGH-1. The population the `useful` column counts out of must not move
    with the answer it is compared against. If `graph_expand` decides which pairs are
    `graph_reachable`, a walk that loses a node shrinks the denominator in the same run that
    fails to lift it, and «0 of N» measures the service against itself.

    So the service is falsified three ways — a reached node dropped, an unreached one added,
    nothing reached — and not one pair may move. Membership is read off the persisted
    `graph_edges`, the ground truth and the `hybrid` ranking the graph re-ranks; never off the
    graph channel. The honest classification is asserted FIRST, so a stratum that ignored the
    graph altogether (every pair `unreachable`, trivially stable) cannot pass.

    THE FUNCTION IS FALSIFIED, NEVER A NAME (PR #198, round 3, F1). This test used to patch the
    attribute `graph_service.graph_expand`, and a `from … import graph_expand` at a module's
    HEADER binds the original object at import time: the circular classification, with its
    import moved there, passed all three cases. The walk reads its edges through
    `graph_service._incident`, which `graph_expand` looks up in its module's globals on every
    call, so falsifying it there reaches the walk under any name that bound it. The service is
    then asked directly, so a walk that stopped going through `_incident` fails here instead of
    turning the invariance below into a tautology.
    """
    data = _workspace(tmp_path)
    _build(data, 3, force=True)
    context = _context(data)
    honest = evaluation.classify_expansion((_E1,), context, k=10)
    assert {(pair.item_id, pair.kind) for pair in honest} >= {
        ("r15", "graph_reachable"),
        ("c00", "graph_unscored"),
        ("f14", "unreachable"),
    }
    honest_reach = _reached_relevant(context)

    real = graph_service._incident
    monkeypatch.setattr(
        graph_service,
        "_incident",
        lambda connection, node_id: rewrite(node_id, real(connection, node_id)),
    )

    assert _reached_relevant(context) != honest_reach
    assert evaluation.classify_expansion((_E1,), context, k=10) == honest


def test_the_expansion_stratum_reads_reachability_off_the_persisted_graph_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """…and the corpus property it reads is the PERSISTED graph, the table `hybrid_graph` walks,
    not a second derivation from the store's enrichments (rule 5): an index whose `graph_edges`
    never held `r15 → hub` cannot call `r15` reachable, though the store still assigns it `hub`."""
    data = _workspace(tmp_path)
    derive = index_build.build_graph_edges

    def without_r15(store, **kwargs):
        return [edge for edge in derive(store, **kwargs) if edge.source != "item:r15"]

    monkeypatch.setattr(index_build, "build_graph_edges", without_r15)
    _build(data, 3, force=True)

    pairs = evaluation.classify_expansion((_E1,), _context(data), k=10)

    assert {(pair.item_id, pair.kind, pair.rank) for pair in pairs} == {
        ("s01", "direct", 1),
        ("f02", "direct", 2),
        ("r15", "unreachable", 15),
        ("c00", "graph_unscored", None),
        ("f14", "unreachable", 14),
    }


def test_the_expansion_stratum_refuses_a_graph_behind_the_store(tmp_path: Path) -> None:
    """Reading the table directly must not shed the refusal `graph_expand` gave for free: a graph
    derived before `items.json` moved would classify pairs against assignments the corpus may no
    longer hold, and nothing in the report would say so."""
    data = _workspace(tmp_path)
    _build(data, 3, force=True)
    items = data / "items.json"
    stat = items.stat()
    os.utime(items, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    with pytest.raises(ValueError, match="index_behind_store"):
        evaluation.classify_expansion((_E1,), _context(data), k=10)


def test_the_graph_sweep_publishes_the_expansion_population_its_useful_column_counts_from(
    tmp_path: Path,
) -> None:
    """`útiles = 0` is a finding only against a population that could have been lifted: over no
    such pair it is 0 by construction (rule 2). So the report carries the population, and the
    `useful` of every cell is a count OUT OF it."""
    data = _workspace(tmp_path)

    report = _sweep(data, {"min_shared_items": [2, 3], "min_weight": [0.0]})

    expansion = report.to_dict()["expansion"]
    assert expansion["population"] == 1
    assert expansion["cases"] == ["G1"]
    assert expansion["by_kind"] == {
        "direct": 0,
        "graph_reachable": 1,
        "graph_unscored": 0,
        "unreachable": 0,
    }
    assert expansion["pairs"] == [
        {"case": "G1", "item": "r15", "kind": "graph_reachable", "rank": 15}
    ]
    assert expansion["untagged_with_population"] == []
    assert expansion["tagged_without_population"] == []
    rows = {(row.min_shared_items, row.min_weight): row for row in report.rows}
    assert (rows[(2, 0.0)].useful, rows[(3, 0.0)].useful) == (0, 1)

    lines = evaluation.render_graph_sweep_markdown(report).splitlines()
    assert (
        "Estrato `expansion` (Plan 04 §11.9): 1 pares relevantes alcanzables sólo por el grafo "
        "en 1 casos (G1) — 0 directos, 0 alcanzables sin puntuar, 0 inalcanzables. La columna "
        "«útiles» cuenta cuántos de esos 1 entraron en el top 10."
    ) in lines


def test_a_sweep_with_no_pair_only_the_graph_reaches_says_useful_cannot_be_anything_but_zero(
    tmp_path: Path,
) -> None:
    """And the golden set's `expansion` tag is checked against the measurement, both ways: a
    tagged case with no reachable pair, and a reachable pair in an untagged case, are named."""
    data = _workspace(tmp_path)
    tagged_direct = GoldenCase(
        id="D1",
        query="zeta",
        provenance="construido",
        strata=("expansion",),
        filters=SearchFilters(),
        relevant_items=("s01",),
    )
    untagged_direct = GoldenCase(
        id="D2",
        query="zeta",
        provenance="construido",
        strata=("semantico",),
        filters=SearchFilters(),
        relevant_items=("f02",),
    )

    report = _sweep(
        data, {"min_shared_items": [3], "min_weight": [0.0]}, cases=(tagged_direct, untagged_direct)
    )

    expansion = report.to_dict()["expansion"]
    assert expansion["population"] == 0
    assert expansion["tagged_without_population"] == ["D1"]
    lines = evaluation.render_graph_sweep_markdown(report).splitlines()
    assert (
        "Estrato `expansion` (Plan 04 §11.9): SIN COBERTURA — ninguno de los 2 pares relevantes "
        "medidos es alcanzable sólo por el grafo (2 directos, 0 alcanzables sin puntuar, 0 "
        "inalcanzables), así que «útiles» = 0 no puede salir de otra manera y no mide al grafo."
    ) in lines
    assert "Casos etiquetados `expansion` sin ningún par alcanzable sólo por el grafo: D1." in lines

    reachable_untagged = GoldenCase(
        id="G2",
        query="zeta",
        provenance="construido",
        strata=("semantico",),
        filters=SearchFilters(),
        relevant_items=("r15",),
    )
    report = _sweep(
        data, {"min_shared_items": [3], "min_weight": [0.0]}, cases=(reachable_untagged,)
    )
    assert report.to_dict()["expansion"]["untagged_with_population"] == ["G2"]
    assert (
        "Casos con pares alcanzables sólo por el grafo sin la etiqueta `expansion`: G2."
        in evaluation.render_graph_sweep_markdown(report).splitlines()
    )


# ---------------------------------------------------------------------------
# The threshold the signed sweep chose is the one the code applies
# ---------------------------------------------------------------------------

SWEEP_DOC = REPO / "docs" / "graph-threshold-sweep.md"
# Plan 04 §1.3's grid, typed here rather than imported, so the doc cannot satisfy it by agreeing
# with a constant of its own (rule 1).
PLAN_GRID = [(m, w) for m in (2, 3, 5, 8) for w in (0.0, 0.02, 0.05, 0.10)]


def _applied_in_doc() -> tuple[int, float]:
    match = re.search(
        r"^\*\*Umbral aplicado:\*\* `min_shared_items = (\d+)` · `min_weight = ([0-9.]+)`$",
        SWEEP_DOC.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert match, f"{SWEEP_DOC} does not declare the applied threshold"
    return int(match.group(1)), float(match.group(2))


def _published_rows() -> list[GraphSweepRow]:
    """The signed table's cells as the rows the sweep ranked, in the order the document prints.

    Rebuilt from the published NUMBERS, and each one proven to re-render byte for byte into the
    line it was read from — so what the tests below rank is the table a reader sees, not a
    parse that dropped a column. The sub-threshold precision losses are not in the table; they
    cannot move the rule either, which reads only whether a loss exceeds the ceiling.
    """
    rows = []
    for line in SWEEP_DOC.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\| \d+ \| ", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        shared, weight, edges, degree, recall, delta, entrants, useful = cells[:8]
        degradation, rejected, in_force = cells[10:]
        row = GraphSweepRow(
            min_shared_items=int(shared),
            min_weight=float(weight),
            edges=int(edges),
            mean_degree=float(degree),
            recall=float(recall),
            recall_delta=float(delta),
            entrants=int(entrants),
            useful=int(useful),
            degradation=float(degradation),
            precision_drops={
                stratum: float(pp)
                for pp, stratum in re.findall(r"cae ([0-9.]+) pp en `([^`]+)`", rejected)
            },
            graph_ran="no corrió" not in rejected,
            in_force=in_force == "sí",
        )
        assert evaluation._graph_table_row(row) == line, f"unparseable table row: {line}"
        rows.append(row)
    return rows


def _published_verdict() -> str:
    match = re.search(
        r"^\*\*Veredicto del instrumento, literal:\*\* (.+?)\n\n",
        SWEEP_DOC.read_text(encoding="utf-8"),
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match, f"{SWEEP_DOC} does not publish the instrument's verdict"
    return " ".join(match.group(1).split())


def test_the_signed_sweep_publishes_every_cell_of_the_plan_grid() -> None:
    """Plan 04 §1.3: «se publica la tabla, incluidos los ajustes que no aportaron»."""
    cells = re.findall(
        r"^\| (\d+) \| ([0-9.]+) \| ", SWEEP_DOC.read_text(encoding="utf-8"), re.MULTILINE
    )
    assert sorted((int(m), float(w)) for m, w in cells) == sorted(PLAN_GRID)


def test_the_signed_table_is_in_the_order_the_rule_gives_its_own_numbers() -> None:
    """The published order is not the author's: `rank_graph_rows` over the published data must
    reproduce it, so a row moved to the top by hand — or a winner renamed — is red."""
    rows = _published_rows()
    assert len(rows) == len(PLAN_GRID)
    assert list(evaluation.rank_graph_rows(rows)) == rows


def test_the_applied_graph_threshold_is_the_winner_the_signed_sweep_published(
    tmp_path: Path,
) -> None:
    """Delivery row 04.5: «el umbral queda aplicado donde se decidió». Bound in code, at every
    place a build reads it: the module default, the config default, `IndexOptions`, and the
    `graph` block a real build seals.

    THE WINNER IS DERIVED, NOT READ: the published rows are re-ranked by the rule and handed to
    `GraphSweepReport`, whose `winner` and verdict are the instrument's own. The «Umbral
    aplicado» line, the verdict paragraph and every default must all name THAT cell — so moving
    the line, the defaults and `config.toml.example` together to another cell is still red,
    because the numbers the sweep measured did not move with them."""
    from xbrain.config import load_config

    report = GraphSweepReport(k=10, limit=10, rows=evaluation.rank_graph_rows(_published_rows()))
    assert report.winner is not None, "the signed table has no winner to apply"
    min_shared, min_weight = report.winner.min_shared_items, report.winner.min_weight
    assert _applied_in_doc() == (min_shared, min_weight)
    assert _published_verdict() == report.to_dict()["verdict"]
    assert (min_shared, min_weight) in PLAN_GRID
    assert graph_build.DEFAULT_GRAPH_MIN_SHARED_ITEMS == min_shared
    assert graph_build.DEFAULT_GRAPH_MIN_WEIGHT == pytest.approx(min_weight)

    (tmp_path / "config.toml").write_text(
        '[paths]\nvault = "vault"\noutput_subdir = "x-knowledge"\ndata_dir = "data"\n'
        '[x]\nhandle = "u"\n',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert (cfg.index_graph_min_shared_items, cfg.index_graph_min_weight) == (
        min_shared,
        min_weight,
    )

    data = _workspace(tmp_path)
    index_build.build(data / "index", index_build.load_index_inputs(*(data / n for n in _INPUTS)))
    sealed = index_build.load_manifest(data / "index").graph
    assert (sealed["min_shared_items"], sealed["min_weight"]) == (min_shared, min_weight)


def test_config_example_documents_the_applied_threshold() -> None:
    example = (REPO / "config.toml.example").read_text(encoding="utf-8")
    min_shared, min_weight = _applied_in_doc()
    assert f"# graph_min_shared_items = {min_shared}\n" in example
    assert f"# graph_min_weight = {min_weight}\n" in example
    assert "the sweep that fixes them has not run" not in example
