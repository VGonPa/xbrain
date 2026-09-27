# tests/test_jev_ask_rank.py
"""Ask results RANK, never cut at `[jev].threshold`; topics are multi-select (task 14).

The first real ask (Víctor, 2026-09-27, `jev_ask_fixtures`) answered 808 posts with a max of
0.80 and showed «0 de 808 llegan a 0,85»: the topic-membership bar was reused as a results
cut. Now the results are every current answer ranked by probability, the first
`[jev].ask_top` shown; an optional minimum (`--min`, «Relevancia mínima») replaces the cut.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.jev_ask_fixtures import VICTOR_QUERY, victor_probabilities, victor_shaped_repo
from tests.test_jev_ask import (
    DT,
    QUERY,
    _ByText,
    _jev_with_topics,
    _refuse_client,
    _run,
    _sent,
    _use,
    make_cfg,
)
from xbrain import cli
from xbrain.cli import app
from xbrain.config import Config
from xbrain.jev.ask import (
    AskFilters,
    AskQuery,
    filter_posts,
    finish_ask,
    load_history,
    plan_ask,
    post_topics,
    refine_results,
    saved_results,
    topic_counts,
)
from xbrain.jev.assess import build_topic_state
from xbrain.jev.dashboard import ask_page_data
from xbrain.jev.load import load_jev_pairs
from xbrain.jev.models import AskHistoryEntry
from xbrain.jev.store import ASK_INDEX, load_ask_index, load_asks, load_runs
from xbrain.store import load_store

runner = CliRunner()


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch) -> Config:
    return make_cfg(tmp_path, monkeypatch)


# --------------------------------------------------------------------------- ranking


def test_results_are_every_current_answer_ranked_never_cut_at_the_topic_bar(cfg: Config):
    """_ByText answers 0.95 for posts about hooks and 0.1 for the rest: under the old cut at
    `[jev].threshold` (0.85) post 2 was not a result; now it is, last."""
    plan, _, results = _run(cfg, _ByText())

    assert [(item.id, record.probability) for item, record in results.ranked] == [
        ("1", 0.95),
        ("3", 0.95),
        ("2", 0.1),
    ]
    assert results.answered == 3


def test_the_configured_threshold_never_reaches_the_results(tmp_path: Path, monkeypatch):
    """`[jev].threshold` is the topic-membership bar; at 0.99 the results are unchanged."""
    cfg = victor_shaped_repo(tmp_path, jev="threshold = 0.99\n")
    [row] = ask_page_data(cfg, load_jev_pairs(cfg), [])["history"]
    assert len(row["answers"]["ids"]) == 808 and row["answered"] == 808


def test_ties_rank_by_post_id_whatever_the_candidates_order(cfg: Config):
    plan, _, _ = _run(cfg, _ByText())
    store = load_store(cfg.items_path)
    records = load_asks(cfg.jev_asks_dir / f"{plan.query.sha}.json", plan.query)
    records["2"] = records["2"].model_copy(update={"probability": 0.95})

    found = saved_results(
        dict(reversed(store.items())),
        None,
        plan.query,
        AskFilters(),
        records,
        topic_threshold=0.85,
        minimum=0.0,
        state_text=_sent,
    )

    assert [item.id for item, _ in found.ranked] == ["1", "2", "3"]


def test_a_minimum_keeps_answers_at_or_above_it_and_counts_every_answer(cfg: Config):
    plan, _, _ = _run(cfg, _ByText())

    at = finish_ask(cfg, plan, None, minimum=0.95)
    above = finish_ask(cfg, plan, None, minimum=0.96)

    assert [item.id for item, _ in at.ranked] == ["1", "3"] and at.answered == 3
    assert above.ranked == () and above.answered == 3


def test_the_history_records_the_minimum_and_how_many_reach_it(cfg: Config):
    plan, _, _ = _run(cfg, _ByText())
    finish_ask(cfg, plan, None, minimum=0.5)

    [entry] = load_ask_index(cfg.jev_asks_dir / ASK_INDEX).queries.values()

    assert (entry.last_evaluated, entry.last_results, entry.last_min) == (3, 2, 0.5)
    assert entry.last_threshold is None  # the old results cut is never written again


def test_a_minimum_outside_zero_to_one_is_refused(cfg: Config):
    plan, _, _ = _run(cfg, _ByText())
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="mínima"):
            finish_ask(cfg, plan, None, minimum=bad)


# --------------------------------------------------------------------------- Víctor's ask


def test_victor_shaped_fixture_has_the_real_landmarks():
    values = victor_probabilities()
    assert max(values) == 0.80 and statistics.median(values) == 0.15
    assert sum(v >= 0.7 for v in values) == 35 and sum(v >= 0.5 for v in values) == 156


def test_victors_saved_ask_reopens_ranked_at_no_cost(tmp_path: Path):
    """The entry his use left (`last_threshold` 0.85, `last_results` 0) reopens with every
    answer ranked: the 0.80 first, the page showing the first 20."""
    cfg = victor_shaped_repo(tmp_path)

    [row] = ask_page_data(cfg, load_jev_pairs(cfg), [])["history"]

    ps = row["answers"]["p"]
    assert row["answered"] == 808 and len(ps) == 808
    assert ps == sorted(ps, reverse=True) and ps[0] == 0.80 and ps[19] == 0.72
    assert row["min"] == 0.0 and row["filters"] == {"since": "2026-05-07"}
    assert "threshold" not in row
    assert not load_runs(cfg.jev_runs_path)  # nothing was paid to reopen it


def test_victors_old_entry_reads_as_no_minimum(tmp_path: Path):
    cfg = victor_shaped_repo(tmp_path)
    [entry] = load_history(cfg).queries.values()
    assert entry.last_threshold == 0.85 and entry.last_min == 0.0


def test_cli_victors_query_again_prints_the_first_20_ranked_for_free(tmp_path: Path, monkeypatch):
    cfg = victor_shaped_repo(tmp_path)
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)

    result = runner.invoke(app, ["jev", "ask", VICTOR_QUERY, "--since", "2026-05-07"])

    out = result.output
    assert result.exit_code == 0, out
    lines = out.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("Resultados"))
    assert lines[header] == (
        "Resultados: los 20 primeros de 808 leídos por Jev, de mayor a menor probabilidad"
    )
    rows = lines[header + 1 : header + 21]
    assert [row.split()[0] for row in rows][:3] == ["0.80", "0.79", "0.79"]
    assert "0.72" == rows[19].split()[0]
    assert lines[header + 21] == "  … y 788 más (--top N o --all para verlos)"
    assert not load_runs(cfg.jev_runs_path)
    [entry] = load_ask_index(cfg.jev_asks_dir / ASK_INDEX).queries.values()
    assert (entry.times, entry.last_results, entry.last_min) == (2, 808, 0.0)


def test_cli_each_result_shows_its_probability_as_a_bar_and_a_number(cfg: Config, monkeypatch):
    _use(monkeypatch, _ByText())

    out = runner.invoke(app, ["jev", "ask", QUERY]).output

    lines = out.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("Resultados"))
    first, last = lines[header + 1], lines[header + 3]
    # Ten cells, one per 0.1, rounded: the bar is the number, drawn.
    assert first.split()[:2] == ["0.95", "██████████"]
    assert last.split()[:2] == ["0.10", "█·········"]


def test_cli_top_all_and_min(cfg: Config, monkeypatch):
    _use(monkeypatch, _ByText())
    runner.invoke(app, ["jev", "ask", QUERY])
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)

    top1 = runner.invoke(app, ["jev", "ask", QUERY, "--top", "1"]).output
    every = runner.invoke(app, ["jev", "ask", QUERY, "--top", "1", "--all"]).output
    at_min = runner.invoke(app, ["jev", "ask", QUERY, "--min", "0.5"]).output

    assert "los 1 primeros de 3 leídos por Jev" in top1 and "… y 2 más" in top1
    assert "Resultados: los 3 de 3 leídos por Jev" in every and "más (" not in every
    assert (
        "Resultados: los 2 de 2 con relevancia ≥ 0.5 · 3 leídos por Jev, "
        "de mayor a menor probabilidad" in at_min
    )


def test_cli_refuses_a_bad_minimum_or_top_and_has_no_threshold_flag(cfg: Config, monkeypatch):
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)
    for args, word in (
        (["--min", "1.5"], "--min"),
        (["--min", "-0.1"], "--min"),
        (["--top", "0"], "--top"),
    ):
        result = runner.invoke(app, ["jev", "ask", QUERY, *args])
        assert result.exit_code == 1 and word in result.output, result.output
    # The results cut is gone; an old habit gets told, in Spanish, what replaces it.
    result = runner.invoke(app, ["jev", "ask", QUERY, "--threshold", "0.5"])
    assert result.exit_code == 1, result.output
    assert "--threshold ya no existe en jev ask" in result.output
    assert "--min" in result.output and "relevancia mínima" in result.output
    assert "--threshold" not in runner.invoke(app, ["jev", "ask", "--help"]).output


def test_cli_the_top_defaults_to_the_configured_ask_top(tmp_path: Path, monkeypatch):
    cfg = victor_shaped_repo(tmp_path, jev="ask_top = 3\n")
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)

    out = runner.invoke(app, ["jev", "ask", VICTOR_QUERY, "--since", "2026-05-07"]).output

    assert "los 3 primeros de 808" in out and "… y 805 más" in out
    assert cfg.jev_ask_top == 3


def test_cli_asks_says_the_minimum_only_when_one_was_set(cfg: Config, monkeypatch):
    _use(monkeypatch, _ByText())
    runner.invoke(app, ["jev", "ask", QUERY])
    runner.invoke(app, ["jev", "ask", "otra consulta", "--min", "0.96"])

    out = runner.invoke(app, ["jev", "asks"]).output

    assert "último uso: 3 con respuesta\n" in out
    assert "último uso: 3 con respuesta, 0 ≥ 0.96 (relevancia mínima)" in out


# --------------------------------------------------------------------------- multi-topic


def test_topics_are_or_within_and_and_with_the_other_filters(cfg: Config):
    _jev_with_topics(cfg, {"startups": 0.9})
    jev = load_jev_pairs(cfg)
    both = AskFilters(topics=("startups", "ai-coding"))

    kept, dropped = filter_posts(jev.store, both, jev=jev, threshold=0.85)
    assert sorted(item.id for item in kept) == ["1", "2", "3", "4"] and dropped == 0

    with_author = replace(both, author="bob")
    kept, _ = filter_posts(jev.store, with_author, jev=jev, threshold=0.85)
    assert [item.id for item in kept] == ["2"]
    older = replace(both, until=DT.date() - timedelta(days=1))
    assert [i.id for i in filter_posts(jev.store, older, jev=jev, threshold=0.85)[0]] == ["3"]


def test_every_topic_is_checked_and_an_unknown_one_refused(cfg: Config):
    jev = load_jev_pairs(cfg)
    with pytest.raises(Exception, match="nutricion"):
        filter_posts(
            jev.store, AskFilters(topics=("startups", "nutricion")), jev=jev, threshold=0.85
        )


def test_topics_are_kept_sorted_and_once_so_equal_picks_compare_equal():
    assert AskFilters(topics=("b", "a", "b")).topics == ("a", "b")
    assert AskFilters(topics=("b", "a")) == AskFilters(topics=("a", "b"))
    with pytest.raises(ValueError, match="--topic"):
        AskFilters(topics=("a", " "))


def test_topics_round_trip_as_a_list_and_an_old_single_topic_still_reads():
    filters = AskFilters(topics=("startups", "ai-coding"), author="bob")
    assert filters.as_json() == {"topics": ["ai-coding", "startups"], "author": "bob"}
    assert AskFilters.from_json(filters.as_json()) == filters
    assert AskFilters.from_json({"topic": "startups"}) == AskFilters(topics=("startups",))
    assert AskFilters.from_json({"topics": []}) == AskFilters()
    for bad, word in (
        ({"topics": "startups"}, "lista"),
        ({"topics": ["a", 3]}, "lista"),
        ({"topic": "a", "topics": ["b"]}, "topic"),
        ({"topic": 3}, "topic"),
    ):
        with pytest.raises(ValueError, match=word):
            AskFilters.from_json(bad)


def test_an_old_history_entry_with_one_topic_reopens(cfg: Config):
    _run(cfg, _ByText(), filters=AskFilters(topics=("ai-coding",)))
    path = cfg.jev_asks_dir / ASK_INDEX
    data = json.loads(path.read_text(encoding="utf-8"))
    [entry] = data["queries"].values()
    entry["last_filters"] = {"topic": "ai-coding"}
    entry["last_threshold"] = 0.85
    del entry["last_min"]
    path.write_text(json.dumps(data), encoding="utf-8")

    [row] = ask_page_data(cfg, load_jev_pairs(cfg), [])["history"]

    assert row["filters"] == {"topics": ["ai-coding"]}
    assert row["answers"]["ids"] == ["1", "3"] and "error" not in row


def test_the_plan_asks_the_union_of_the_topics(cfg: Config):
    one = plan_ask(cfg, AskQuery.of(QUERY), AskFilters(topics=("startups",)), None)
    two = plan_ask(cfg, AskQuery.of(QUERY), AskFilters(topics=("startups", "ai-coding")), None)
    assert sorted(i.id for i in one.candidates) == ["2", "4"]
    assert sorted(i.id for i in two.candidates) == ["1", "2", "3", "4"]
    assert two.estimate.posts > one.estimate.posts


# --------------------------------------------------------------------------- per-topic counts


def _brute(store, filters, jev, threshold) -> dict[str, int]:
    out = {}
    for topic in sorted({t.slug for t in jev.vocab} | {"ai-coding", "startups"}):
        kept, _ = filter_posts(
            store, replace(filters, topics=(topic,)), jev=jev, threshold=threshold
        )
        out[topic] = len(kept)
    return out


@pytest.mark.parametrize("threshold", [0.85, 0.5])
def test_topic_counts_equal_filtering_by_each_topic_alone(cfg: Config, threshold: float):
    _jev_with_topics(cfg, {"startups": 0.7, "ai-coding": 0.9})
    jev = load_jev_pairs(cfg)
    day = DT.date()
    combos = [
        AskFilters(),
        AskFilters(author="alice"),
        AskFilters(since=day),
        AskFilters(until=day - timedelta(days=1)),
        AskFilters(only_evaluated=True),
        AskFilters(topics=("startups",), author="bob"),  # its own topics are ignored
    ]
    for filters in combos:
        counts = topic_counts(jev.store, filters, jev=jev, threshold=threshold)
        assert counts == _brute(jev.store, filters, jev, threshold), filters


def test_topic_counts_without_jev_use_enrich_alone(cfg: Config):
    store = load_store(cfg.items_path)
    assert topic_counts(store, AskFilters(), jev=None, threshold=0.85) == {
        "ai-coding": 2,
        "startups": 2,
    }


def test_the_page_ships_the_per_topic_counts_of_the_whole_corpus(cfg: Config):
    _jev_with_topics(cfg, {"startups": 0.9})
    jev = load_jev_pairs(cfg)
    view = ask_page_data(cfg, jev, [])
    assert view["topic_counts"] == topic_counts(
        jev.store, AskFilters(), jev=jev, threshold=cfg.jev_threshold
    )
    assert view["topic_counts"]["startups"] == 3


def test_a_history_entry_model_accepts_the_old_shape_and_the_new():
    base = {
        "query_sha": "a" * 64,
        "query": "q",
        "first_asked_at": "2026-09-27T07:16:58Z",
        "last_asked_at": "2026-09-27T07:16:58Z",
        "times": 1,
        "last_evaluated": 3,
        "last_results": 0,
        "last_filters": {"topics": ["a", "b"]},
    }
    old = AskHistoryEntry.model_validate({**base, "last_threshold": 0.85})
    new = AskHistoryEntry.model_validate({**base, "last_min": 0.4})
    assert (old.last_min, old.last_threshold) == (0.0, 0.85)
    assert (new.last_min, new.last_threshold) == (0.4, None)


# --------------------------------------------------------------------------- refine, free


def _victor(tmp_path: Path):
    cfg = victor_shaped_repo(tmp_path)
    jev = load_jev_pairs(cfg)
    query = AskQuery.of(VICTOR_QUERY)
    records = load_asks(cfg.jev_asks_dir / f"{query.sha}.json", query)
    found = saved_results(
        jev.store,
        jev,
        query,
        AskFilters(since=datetime(2026, 5, 7).date()),
        records,
        topic_threshold=cfg.jev_threshold,
        minimum=0.0,
        state_text=_sent,
    )
    return cfg, jev, found


def _expected(found, jev, *, minimum=0.0, topics=(), since=None, until=None, author=None):
    """Brute force, straight from the rules: day in UTC, handle casefold, `post_topics`."""
    current = {i.id: a for i, a in jev.pairs}
    out = []
    for item, record in found.ranked:
        day = item.created_at.astimezone(timezone.utc).date()
        if record.probability < minimum:
            continue
        if since and day < since or until and day > until:
            continue
        if author and item.author.handle.casefold() != author.lstrip("@").casefold():
            continue
        if topics and not set(topics) & post_topics(item, current.get(item.id), 0.85):
            continue
        out.append(item.id)
    return out


@pytest.mark.parametrize(
    "refine",
    [
        {},
        {"minimum": 0.5},
        {"topics": ("agentic-engineering",)},
        {"topics": ("startups", "agentic-engineering"), "minimum": 0.2},
        {"since": datetime(2026, 5, 20).date(), "until": datetime(2026, 5, 30).date()},
        {"author": "@SOMEONE", "minimum": 0.7},
        {"author": "nadie"},
    ],
)
def test_refining_a_saved_query_filters_its_ranked_answers_by_the_filter_rules(
    tmp_path: Path, refine: dict
):
    cfg, jev, found = _victor(tmp_path)
    minimum = refine.pop("minimum", 0.0)

    refined = refine_results(
        found,
        AskFilters(**refine),
        minimum,
        store=jev.store,
        jev=jev,
        threshold=cfg.jev_threshold,
    )

    assert [item.id for item, _ in refined.ranked] == _expected(
        found, jev, minimum=minimum, **refine
    )
    assert refined.answered == found.answered == 808
    # Order is the ranking's, untouched: refining never re-ranks.
    kept = {item.id for item, _ in refined.ranked}
    assert [i.id for i, _ in found.ranked if i.id in kept] == [i.id for i, _ in refined.ranked]


def test_the_page_ships_every_answer_ranked_and_the_python_refine_keys(tmp_path: Path):
    cfg = victor_shaped_repo(tmp_path)
    path = cfg.jev_asks_dir / ASK_INDEX
    data = json.loads(path.read_text(encoding="utf-8"))
    [entry] = data["queries"].values()
    entry["last_min"] = 0.5
    path.write_text(json.dumps(data), encoding="utf-8")
    jev = load_jev_pairs(cfg)

    view = ask_page_data(cfg, jev, [])
    [row] = view["history"]

    # The use's minimum is the refine DEFAULT, never a cut of what ships.
    assert (len(row["answers"]["ids"]), row["min"], row["answered"]) == (808, 0.5, 808)
    item = jev.store[row["answers"]["ids"][0]]
    assert view["keys"][item.id] == {
        "d": item.created_at.astimezone(timezone.utc).date().isoformat(),
        "a": item.author.handle.casefold(),
        "t": sorted(post_topics(item, None, cfg.jev_threshold)),
        "n": build_topic_state(item, cfg.jev_state_char_limit)[1],
    }
    assert set(view["keys"]) == set(row["answers"]["ids"])


def test_cli_asks_reprints_a_saved_query_refined_without_a_client(tmp_path: Path, monkeypatch):
    cfg = victor_shaped_repo(tmp_path)
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)
    _, jev, found = _victor(tmp_path / "again")
    expected = _expected(found, jev, minimum=0.5, topics=("agentic-engineering",))

    result = runner.invoke(
        app, ["jev", "asks", "1", "--min", "0.5", "--topic", "agentic-engineering", "--top", "3"]
    )

    out = result.output
    assert result.exit_code == 0, out
    assert f"«{VICTOR_QUERY}»" in out
    assert (
        f"Resultados: los 3 primeros de {len(expected)} que pasan el refinado (topic "
        "agentic-engineering) con relevancia ≥ 0.5 · 808 leídos por Jev, de mayor a menor "
        "probabilidad" in out
    )
    assert f"  … y {len(expected) - 3} más (--top N o --all para verlos)" in out
    lines = [line for line in out.splitlines() if "https://x.com/" in line]
    assert [line.split()[2] for line in lines] == expected[:3]
    assert not load_runs(cfg.jev_runs_path)
    # Reprinting is reading: the history is not touched.
    [entry] = load_ask_index(cfg.jev_asks_dir / ASK_INDEX).queries.values()
    assert (entry.times, entry.last_min) == (1, 0.0)


def test_cli_asks_picks_a_query_by_sha_prefix_and_refuses_an_unknown_one(
    tmp_path: Path, monkeypatch
):
    victor_shaped_repo(tmp_path)
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)
    sha = AskQuery.of(VICTOR_QUERY).sha

    by_sha = runner.invoke(app, ["jev", "asks", sha[:8], "--since", "2026-06-01", "--all"])
    unknown = runner.invoke(app, ["jev", "asks", "ffffffff"])
    beyond = runner.invoke(app, ["jev", "asks", "2"])

    assert by_sha.exit_code == 0, by_sha.output
    assert (
        "Resultados: los 220 de 220 que pasan el refinado (desde 2026-06-01) · 808 leídos por "
        "Jev" in by_sha.output
    )
    for bad in (unknown, beyond):
        assert bad.exit_code == 1 and "no está en el historial" in bad.output


def test_cli_asks_a_saved_query_defaults_to_its_own_minimum(cfg: Config, monkeypatch):
    _use(monkeypatch, _ByText())
    runner.invoke(app, ["jev", "ask", QUERY, "--min", "0.5"])
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)

    default = runner.invoke(app, ["jev", "asks", "1"]).output
    none = runner.invoke(app, ["jev", "asks", "1", "--min", "0"]).output

    assert "Resultados: los 2 de 2 con relevancia ≥ 0.5 · 3 leídos por Jev" in default
    assert "Resultados: los 3 de 3 leídos por Jev" in none


def test_refine_flags_without_a_query_are_refused(cfg: Config):
    result = runner.invoke(app, ["jev", "asks", "--min", "0.5"])
    assert result.exit_code == 1 and "consulta" in result.output


def test_cli_a_refined_reprint_counts_what_passes_the_refine(tmp_path: Path, monkeypatch):
    """The first real refine: «agentic-engineering, ≥ 0.5» keeps 156 − those enrich puts
    elsewhere; the header counts THAT (never «105 de 808 llegan a 0.5»), and the tail adds up:
    shown + «y N más» == what passes."""
    victor_shaped_repo(tmp_path)
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)
    _, jev, found = _victor(tmp_path / "again")
    kept = _expected(found, jev, topics=("startups",), author="@someone")

    out = runner.invoke(
        app, ["jev", "asks", "1", "--topic", "startups", "--author", "@someone"]
    ).output

    assert (
        f"Resultados: los 20 primeros de {len(kept)} que pasan el refinado (topic startups, "
        "autor @someone) · 808 leídos por Jev, de mayor a menor probabilidad" in out
    )
    assert f"  … y {len(kept) - 20} más (--top N o --all para verlos)" in out


def test_cli_asks_numbers_each_query_and_the_number_or_sha_reopens_it(cfg: Config, monkeypatch):
    """What `jev asks` prints is what `jev asks N` / `jev asks <sha>` takes."""
    _use(monkeypatch, _ByText())
    runner.invoke(app, ["jev", "ask", QUERY])
    runner.invoke(app, ["jev", "ask", "otra consulta", "--min", "0.96"])
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)

    listed = runner.invoke(app, ["jev", "asks"]).output

    heads = [line for line in listed.splitlines() if line[:1].isdigit()]
    assert len(heads) == 2
    for n, head in enumerate(heads, start=1):
        number, sha, rest = head.split(" ", 2)
        assert number == f"{n}." and len(sha) == 8 and rest.startswith("«")
        query = rest[1 : rest.index("»")]
        assert sha == AskQuery.of(query).sha[:8]
        for which in (str(n), sha):
            again = runner.invoke(app, ["jev", "asks", which, "--top", "1"])
            assert again.exit_code == 0 and f"«{query}»" in again.output, which
