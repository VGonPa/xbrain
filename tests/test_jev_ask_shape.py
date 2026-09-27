# tests/test_jev_ask_shape.py
"""The shape a saved query's answers travel in (PR 14 fix wave): what PR 16 streams into and
PR 17 sorts and groups, so it is fixed here.

* ONE answer view (`ask.answer_view`): id, probability, model, minute asked, refine keys —
  the blob builds every answer through it (and PR 16's stream will).
* COMPACT rows: per query `{"ids": […], "p": […]}` (full floats), the model and the minute
  asked once, with an `exceptions` map for the answers that differ — under a byte budget.
* Per post, once: the refine keys and the state's size (`asks.keys`).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.jev_ask_fixtures import JEV_AGENTIC, VICTOR_QUERY, victor_shaped_repo
from xbrain.jev.ask import (
    AnswerView,
    AskFilters,
    AskQuery,
    answer_view,
    post_topics,
    refine_keys,
    refine_results,
)
from xbrain.jev.assess import build_topic_state
from xbrain.jev.dashboard import answer_columns, ask_page_data
from xbrain.jev.load import load_jev_pairs
from xbrain.jev.store import load_asks

#: Bytes per answer a query's columns may take: a 19-digit post id and a full float (17
#: significant digits) are ~43 B in JSON. The old rows took ~130 B.
BYTES_PER_ANSWER = 45


def _victor(tmp_path: Path, **kw):
    cfg = victor_shaped_repo(tmp_path, **kw)
    jev = load_jev_pairs(cfg)
    view = ask_page_data(cfg, jev, [])
    [row] = view["history"]
    return cfg, jev, view, row


def test_a_row_ships_its_answers_as_columns_in_the_ranked_order(tmp_path: Path):
    cfg, jev, view, row = _victor(tmp_path)
    query = AskQuery.of(VICTOR_QUERY)
    records = load_asks(cfg.jev_asks_dir / f"{query.sha}.json", query)

    answers = row["answers"]

    assert "results" not in row
    assert len(answers["ids"]) == len(answers["p"]) == 808 == row["answered"]
    expected = sorted(records.values(), key=lambda r: (-r.probability, r.item_id))
    assert answers["ids"] == [r.item_id for r in expected]
    assert answers["p"] == [r.probability for r in expected]
    # One pass, one model, one minute: said once, no exception.
    assert (answers["model"], answers["asked_at"], answers["exceptions"]) == (
        "jev-1.13.0",
        "2026-09-27T07:16Z",
        {},
    )


def test_answers_that_differ_from_the_querys_model_or_minute_are_exceptions():
    moment = datetime(2026, 9, 27, 7, 16, 32, tzinfo=timezone.utc)
    later = datetime(2026, 9, 28, 9, 0, 1, tzinfo=timezone.utc)
    keys = {"d": "2026-05-07", "a": "x", "t": []}
    views = [
        AnswerView(id="1", p=0.9, model="jev-1", asked_at=moment, keys=keys),
        AnswerView(id="2", p=0.8, model="jev-1", asked_at=moment.replace(second=59), keys=keys),
        AnswerView(id="3", p=0.7, model="jev-2", asked_at=moment, keys=keys),
        AnswerView(id="4", p=0.6, model="jev-1", asked_at=later, keys=keys),
    ]

    columns = answer_columns(views)

    assert columns == {
        "ids": ["1", "2", "3", "4"],
        "p": [0.9, 0.8, 0.7, 0.6],
        "model": "jev-1",
        "asked_at": "2026-09-27T07:16Z",
        "exceptions": {"3": {"model": "jev-2"}, "4": {"asked_at": "2026-09-28T09:00Z"}},
    }


def test_no_answers_ship_as_empty_columns():
    assert answer_columns([]) == {
        "ids": [],
        "p": [],
        "model": None,
        "asked_at": None,
        "exceptions": {},
    }


@pytest.mark.parametrize("answers", [1, 808, 5000])
def test_a_querys_columns_stay_under_the_byte_budget_per_answer(answers: int):
    moment = datetime(2026, 9, 27, 7, 16, 32, tzinfo=timezone.utc)
    keys = {"d": "2026-05-07", "a": "x", "t": []}
    views = [
        # Worst realistic case: X's 19-digit ids and a float with every digit.
        AnswerView(
            id=str(2052333620652847425 + n),
            p=0.12345678901234568 + n * 1e-9,
            model="jev-1.13.0",
            asked_at=moment,
            keys=keys,
        )
        for n in range(answers)
    ]

    size = len(json.dumps(answer_columns(views), separators=(",", ":")))

    assert size <= BYTES_PER_ANSWER * answers + 120


def test_the_answer_view_is_what_the_blob_ships_for_each_answer(tmp_path: Path):
    cfg, jev, view, row = _victor(tmp_path, topics=True)
    query = AskQuery.of(VICTOR_QUERY)
    records = load_asks(cfg.jev_asks_dir / f"{query.sha}.json", query)
    current = jev.current_by_id()
    item = jev.store[row["answers"]["ids"][0]]

    one = answer_view(item, records[item.id], current.get(item.id), cfg.jev_threshold)

    assert (one.id, one.p, one.model, one.asked_at) == (
        item.id,
        records[item.id].probability,
        records[item.id].model,
        records[item.id].asked_at,
    )
    assert one.keys == refine_keys(item, current.get(item.id), cfg.jev_threshold)
    shipped = dict(view["keys"][item.id])
    assert shipped.pop("n") == build_topic_state(item, cfg.jev_state_char_limit)[1]
    assert shipped == one.keys


def test_every_answer_has_keys_and_each_post_ships_them_once(tmp_path: Path):
    _, _, view, row = _victor(tmp_path)
    assert set(view["keys"]) == set(row["answers"]["ids"])


def test_the_jev_side_of_the_topic_keys_is_judged_at_the_topic_bar(tmp_path: Path):
    """Jev puts enrich-`startups` posts in `agentic-engineering` at 0.85 (in) and 0.84 (out):
    their refine keys, and a refine by that topic, follow the bar exactly."""
    cfg, jev, view, row = _victor(tmp_path, topics=True)
    current = jev.current_by_id()
    by_noul: dict[float, list[str]] = {0.85: [], 0.84: []}
    for n, item in enumerate(jev.store.values()):
        assessment = current.get(item.id)
        startups = item.enriched.primary_topic == "startups"
        if assessment is not None and startups and n % 7 in JEV_AGENTIC and item.id in view["keys"]:
            by_noul[JEV_AGENTIC[n % 7]].append(item.id)
    assert by_noul[0.85] and by_noul[0.84]
    for item_id in by_noul[0.85]:
        assert view["keys"][item_id]["t"] == ["agentic-engineering", "startups"]
    for item_id in by_noul[0.84]:
        assert view["keys"][item_id]["t"] == ["startups"]
    # The refine agrees: the 0.85 posts are kept by the topic, the 0.84 ones are not.
    from xbrain.jev.ask import reopen_results
    from xbrain.jev.ask import load_history

    found = reopen_results(cfg, load_history(cfg).queries[row["sha"]], jev)
    refined = refine_results(
        found,
        AskFilters(topics=("agentic-engineering",)),
        0.0,
        store=jev.store,
        jev=jev,
        threshold=cfg.jev_threshold,
    )
    kept = {item.id for item, _ in refined.ranked}
    assert set(by_noul[0.85]) <= kept and not set(by_noul[0.84]) & kept
    assert all(
        "agentic-engineering" in post_topics(jev.store[i], current.get(i), cfg.jev_threshold)
        for i in kept
    )


def test_the_current_answers_by_id_are_built_once(tmp_path: Path):
    cfg = victor_shaped_repo(tmp_path, topics=True)
    jev = load_jev_pairs(cfg)

    first = jev.current_by_id()

    assert first is jev.current_by_id()
    assert first == {item.id: assessment for item, assessment in jev.pairs}


# --------------------------------------------------------------------------- one author rule


@pytest.mark.parametrize("typed", ["someone", "@SomeOne", " @someone ", "@ SOMEONE", "@@someone"])
def test_an_author_is_read_by_one_rule_on_both_sides(tmp_path: Path, typed: str):
    """`strip().lstrip("@").strip().casefold()` — what the page applies to what is typed and
    Python to the filter, and to the handle the refine keys carry."""
    from xbrain.jev.ask import filter_posts, normalise_author

    cfg, jev, view, row = _victor(tmp_path)
    kept, _ = filter_posts(
        jev.store, AskFilters(author=typed), jev=jev, threshold=cfg.jev_threshold
    )

    assert normalise_author(typed) == "someone"
    assert len(kept) == len(jev.store)
    assert {keys["a"] for keys in view["keys"].values()} == {"someone"}


def test_the_history_leaves_out_only_an_absent_legacy_threshold(tmp_path: Path):
    """`last_threshold` (legacy) is left out when absent; nothing else is dropped for being
    empty — a field that is None is written, never silently lost."""
    from xbrain.jev.models import AskHistoryEntry, AskIndex
    from xbrain.jev.store import save_ask_index

    base = {
        "query_sha": "a" * 64,
        "query": "q",
        "first_asked_at": "2026-09-27T07:16:58Z",
        "last_asked_at": "2026-09-27T07:16:58Z",
        "times": 1,
        "last_evaluated": 3,
        "last_results": 0,
        "last_filters": {},
    }
    new = AskHistoryEntry.model_validate(base)
    old = AskHistoryEntry.model_validate({**base, "query_sha": "b" * 64, "last_threshold": 0.85})
    path = tmp_path / "index.json"

    save_ask_index(AskIndex(queries={new.query_sha: new, old.query_sha: old}), path)

    written = json.loads(path.read_text(encoding="utf-8"))["queries"]
    fields = set(AskHistoryEntry.model_fields)
    assert set(written["a" * 64]) == fields - {"last_threshold"}
    assert set(written["b" * 64]) == fields and written["b" * 64]["last_threshold"] == 0.85
    assert "calibration" in json.loads(path.read_text(encoding="utf-8"))
