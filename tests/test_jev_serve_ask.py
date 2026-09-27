# tests/test_jev_serve_ask.py
"""`xbrain jev serve`, kind `ask`: the «Preguntar» tab's endpoints, over HTTP, with the fake.

The same money path as topics (estimate → single-use confirmation → the ONE job slot → the
pass lock → `run.run_ask` through `_Metered`), and the read-only views of what was asked
(`/api/asks`, `/api/ask/<sha>`), which are slices of the blob the static page carries.
Every client is a `FakeJevClient`: no test reaches TypeSafe.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.jev_fakes import FakeJevClient
from tests.test_jev_serve import ITEMS, _locked, _repo, _Served
from xbrain.config import Config
from xbrain.jev.ask import AskFilters, AskQuery, estimate_ask, filter_posts, select_ask_items
from xbrain.jev.client import JevResult, NoulAnswer
from xbrain.jev.defaults import DEFAULT_CHARS_PER_TOKEN, tokens_cost_usd
from xbrain.jev.load import load_jev_pairs
from xbrain.jev.lock import pass_lock
from xbrain.jev.questions import ASK_KEY
from xbrain.jev.store import load_ask_index, load_asks, load_runs
from xbrain.store import load_store, save_store

QUERY = "¿Cómo configuro hooks en Claude Code?"
#: What the fake answers per post: 1 and 3 answer the query, 5 is at 0.5, the rest do not.
PROBS = {"1": 0.97, "2": 0.1, "3": 0.9, "4": 0.2, "5": 0.5}


class _Asker(FakeJevClient):
    """A priced fake that answers the ask's Noul per post (`PROBS`) and records the posts."""

    def __init__(self, *, delay: float = 0.0, gate: threading.Event | None = None, **kw: Any):
        super().__init__(provider="typesafe", **kw)
        self.delay = delay
        self.gate = gate
        self.asked: list[str] = []
        self._lock = threading.Lock()

    def ask(self, state, questions):
        text = state["post"]
        post = next(
            i for i, item in ITEMS.items() if item.text.strip() and text.startswith(item.text)
        )
        with self._lock:
            self.asked.append(post)
        if self.gate is not None:
            assert self.gate.wait(10)
        if self.delay:
            time.sleep(self.delay)
        result = super().ask(state, questions)
        assert set(questions) == {ASK_KEY}
        return JevResult(
            provider=result.provider,
            model=result.model,
            answers={ASK_KEY: NoulAnswer(noul=PROBS[post])},
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )


class _AskServed(_Served):
    def ask_estimate(self, body: dict[str, Any]) -> dict[str, Any]:
        status, data, _ = self.request("POST", "/api/ask/estimate", body)
        assert status == 200, data
        return data

    def ask_evaluate(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        status, data, _ = self.request("POST", "/api/ask/evaluate", body)
        return status, data

    def ask_run(self, body: dict[str, Any]) -> dict[str, Any]:
        estimate = self.ask_estimate(body)
        assert estimate["allowed"], estimate
        status, job = self.ask_evaluate({**body, "confirm_token": estimate["confirm_token"]})
        assert status == 202, job
        return self.wait_job()


def _served(tmp_path: Path, monkeypatch, client: FakeJevClient, jev: str = "") -> _AskServed:
    return _AskServed(_repo(tmp_path, monkeypatch, jev=jev), client)


@pytest.fixture
def served(tmp_path: Path, monkeypatch) -> Iterator[_AskServed]:
    s = _served(tmp_path, monkeypatch, _Asker())
    yield s
    s.close()


def _expected(cfg: Config, query: str = QUERY, **filters: Any):
    """What the CLI's own functions select and estimate for `query` today."""
    ask_query = AskQuery.of(query)
    store = load_store(cfg.items_path)
    wanted = AskFilters(**filters)
    jev = load_jev_pairs(cfg) if wanted.needs_jev else None
    candidates, dropped = filter_posts(store, wanted, jev=jev, threshold=cfg.jev_threshold)
    records = load_asks(cfg.jev_asks_dir / f"{ask_query.sha}.json", ask_query)
    selection = select_ask_items(
        candidates, records, ask_query, char_limit=cfg.jev_state_char_limit, limit=None
    )
    return ask_query, selection, dropped


# --------------------------------------------------------------------------- the estimate


def test_the_ask_estimate_is_the_commands_selection_and_price(served: _AskServed):
    query, selection, _ = _expected(served.cfg)
    priced = estimate_ask(
        selection,
        query,
        char_limit=served.cfg.jev_state_char_limit,
        chars_per_token=DEFAULT_CHARS_PER_TOKEN,
    )

    estimate = served.ask_estimate({"query": QUERY})

    assert estimate["kind"] == "ask"
    assert estimate["query_sha"] == query.sha
    assert estimate["ids"] == ["1", "2", "3", "4", "5"]
    assert (estimate["posts"], estimate["skipped_current"], estimate["skipped_no_evidence"]) == (
        5,
        0,
        1,
    )
    assert (estimate["tokens"], estimate["usd"]) == (priced.tokens, priced.usd)
    assert estimate["chars_per_token"] == {"value": DEFAULT_CHARS_PER_TOKEN, "measured": 0}
    assert estimate["allowed"] is True and estimate["confirm_token"]
    assert estimate["max_usd"] == served.cfg.jev_serve_max_usd
    assert served.built == 0
    assert not served.cfg.jev_asks_dir.exists()


def test_the_ask_estimate_normalises_the_query_like_the_command(served: _AskServed):
    spaced = served.ask_estimate({"query": "  ¿Cómo configuro   hooks en Claude Code? "})

    assert spaced["query_sha"] == AskQuery.of(QUERY).sha
    assert spaced["pick"]["query"] == AskQuery.of(QUERY).text


def test_the_ask_filters_narrow_what_is_paid_for(served: _AskServed):
    by_topic = served.ask_estimate({"query": QUERY, "topic": "startups"})
    limited = served.ask_estimate({"query": QUERY, "limit": 2})
    other_author = served.ask_estimate({"query": QUERY, "author": "@nadie"})
    evaluated = served.ask_estimate({"query": QUERY, "only_evaluated": True})
    days = served.ask_estimate({"query": QUERY, "since": "2026-09-22", "until": "2026-09-22"})

    assert (by_topic["ids"], by_topic["dropped"]) == (["2", "4"], 4)
    assert (limited["ids"], limited["remaining"]) == (["1", "2"], 3)
    assert other_author["ids"] == [] and other_author["dropped"] == 6
    assert evaluated["ids"] == ["1", "2"]
    assert days["ids"] == ["1", "2", "3", "4", "5"]
    assert by_topic["pick"] == {"query": AskQuery.of(QUERY).text, "topic": "startups"}


@pytest.mark.parametrize(
    "body,words",
    [
        ({}, "query"),
        ({"query": "   "}, "vacía"),
        ({"query": 3}, "query"),
        ({"query": QUERY, "force": True}, "desconocidos"),
        ({"query": QUERY, "ids": ["1"]}, "desconocidos"),
        ({"query": QUERY, "topic": ""}, "topic"),
        ({"query": QUERY, "topic": "inventado"}, "topic desconocido"),
        ({"query": QUERY, "since": "22/09/2026"}, "since"),
        ({"query": QUERY, "until": 20260922}, "until"),
        ({"query": QUERY, "since": "2026-09-23", "until": "2026-09-22"}, "posterior"),
        ({"query": QUERY, "limit": 0}, "limit"),
        ({"query": QUERY, "limit": True}, "limit"),
        ({"query": QUERY, "only_evaluated": "sí"}, "only_evaluated"),
        ({"query": QUERY, "author": ""}, "author"),
        ({"query": "x" * 2001}, "2000"),
        ([QUERY], "objeto"),
    ],
)
def test_a_malformed_ask_is_refused_before_anything_is_selected(
    served: _AskServed, body: Any, words: str
):
    status, error, _ = served.request("POST", "/api/ask/estimate", body)

    assert status == 400, error
    assert words in error["error"]


def test_filters_that_leave_nothing_to_ask_say_so_and_mint_nothing(served: _AskServed):
    estimate = served.ask_estimate({"query": QUERY, "author": "nadie"})

    assert estimate["allowed"] is False and estimate["confirm_token"] is None
    assert "ningún post" in estimate["refusal"]


def test_an_ask_over_the_cap_is_refused_and_says_so(tmp_path: Path, monkeypatch):
    s = _served(tmp_path, monkeypatch, _Asker(), jev="serve_max_usd = 0.000001\n")
    try:
        estimate = s.ask_estimate({"query": QUERY})
    finally:
        s.close()

    assert estimate["allowed"] is False and estimate["confirm_token"] is None
    assert "serve_max_usd" in estimate["refusal"]


def test_the_ask_estimate_measures_chars_per_token_once_answers_exist(served: _AskServed):
    served.ask_run({"query": QUERY, "limit": 2})
    records = load_asks(
        served.cfg.jev_asks_dir / f"{AskQuery.of(QUERY).sha}.json", AskQuery.of(QUERY)
    )
    ratio = sum(r.prompt_chars for r in records.values()) / sum(
        r.input_tokens or 0 for r in records.values()
    )

    estimate = served.ask_estimate({"query": "posts sobre rondas seed"})

    assert estimate["chars_per_token"] == {"value": ratio, "measured": 2}


# --------------------------------------------------------------------------- the job


def test_an_ask_job_asks_what_was_estimated_saves_logs_and_keeps_the_history(
    served: _AskServed,
):
    query = AskQuery.of(QUERY)

    job = served.ask_run({"query": QUERY})

    assert job["state"] == "done" and job["kind"] == "ask"
    assert job["query_sha"] == query.sha
    assert sorted(served.client.asked) == ["1", "2", "3", "4", "5"]
    assert job["outcome"]["ok"] == 5 and job["outcome"]["logged"] is True
    assert job["outcome"]["results"] == 2
    assert job["usd"] == pytest.approx(5 * tokens_cost_usd(100, "typesafe"))
    records = load_asks(served.cfg.jev_asks_dir / f"{query.sha}.json", query)
    assert {post: r.probability for post, r in records.items()} == PROBS
    (run,) = [r for r in load_runs(served.cfg.jev_runs_path) if r.kind == "ask"]
    assert (run.query_sha, run.ok, run.requests) == (query.sha, 5, 5)
    entry = load_ask_index(served.cfg.jev_asks_dir / "index.json")[query.sha]
    assert (entry.query, entry.times, entry.evaluated, entry.results) == (query.text, 1, 5, 2)
    assert entry.threshold == served.cfg.jev_threshold
    assert not _locked(served.cfg.jev_lock_path)


def test_the_history_keeps_the_filters_the_job_ran_with(served: _AskServed):
    served.ask_run({"query": QUERY, "topic": "ai-coding", "limit": 2})

    entry = load_ask_index(served.cfg.jev_asks_dir / "index.json")[AskQuery.of(QUERY).sha]
    assert entry.filters == {"topic": "ai-coding"}
    assert (entry.evaluated, entry.results) == (2, 1)


def test_asking_again_costs_nothing_builds_no_client_and_still_counts_in_the_history(
    served: _AskServed,
):
    served.ask_run({"query": QUERY})
    built = served.built

    again = served.ask_estimate({"query": QUERY})
    job = served.ask_run({"query": QUERY})

    assert (again["posts"], again["skipped_current"], again["usd"]) == (0, 5, 0.0)
    assert again["allowed"] is True
    assert job["state"] == "done" and job["total"] == 0 and job["usd"] == 0
    assert served.built == built
    assert len([r for r in load_runs(served.cfg.jev_runs_path) if r.kind == "ask"]) == 1
    entry = load_ask_index(served.cfg.jev_asks_dir / "index.json")[AskQuery.of(QUERY).sha]
    assert entry.times == 2


def test_new_evidence_between_estimate_and_confirm_refuses_the_ask_job(served: _AskServed):
    served.ask_run({"query": QUERY, "limit": 1})
    estimate = served.ask_estimate({"query": QUERY, "limit": 1})
    items = load_store(served.cfg.items_path)
    items["1"].text = "Claude Code hooks, revisado"
    save_store(items, served.cfg.items_path)

    status, error = served.ask_evaluate(
        {"query": QUERY, "limit": 1, "confirm_token": estimate["confirm_token"]}
    )

    assert status == 409 and "cambió" in error["error"]
    assert served.request("GET", "/api/job")[1]["state"] == "done"


def test_an_ask_confirmation_is_bound_to_its_query_filters_and_kind(served: _AskServed):
    estimate = served.ask_estimate({"query": QUERY, "limit": 2})
    token = estimate["confirm_token"]

    other_query = served.ask_evaluate({"query": "otra", "limit": 2, "confirm_token": token})
    other_limit = served.ask_evaluate({"query": QUERY, "limit": 3, "confirm_token": token})
    as_topics = served.request(
        "POST", "/api/topics/evaluate", {"ids": ["3"], "confirm_token": token}
    )

    assert other_query[0] == other_limit[0] == as_topics[0] == 409
    assert served.ask_evaluate({"query": QUERY, "limit": 2, "confirm_token": token})[0] == 202


def test_a_topics_job_and_an_ask_job_share_the_one_slot(tmp_path: Path, monkeypatch):
    gate = threading.Event()
    s = _served(tmp_path, monkeypatch, _Asker(gate=gate))
    try:
        topics = s.estimate({"ids": ["3"]})
        ask = s.ask_estimate({"query": QUERY})
        assert s.evaluate({"ids": ["3"], "confirm_token": topics["confirm_token"]})[0] == 202

        status, error = s.ask_evaluate({"query": QUERY, "confirm_token": ask["confirm_token"]})
        gate.set()
        s.wait_job()
        # The confirmation was kept: the ask runs once the slot is free.
        after = s.ask_evaluate({"query": QUERY, "confirm_token": ask["confirm_token"]})
        s.wait_job()
    finally:
        gate.set()
        s.close()

    assert status == 409 and "en curso" in error["error"]
    assert after[0] == 202


def test_a_terminal_pass_holding_the_lock_refuses_the_ask_job(served: _AskServed):
    estimate = served.ask_estimate({"query": QUERY})

    with pass_lock(served.cfg.jev_lock_path, "xbrain jev topics"):
        status, error = served.ask_evaluate(
            {"query": QUERY, "confirm_token": estimate["confirm_token"]}
        )

    assert status == 409 and "xbrain jev topics" in error["error"]
    assert served.client.asked == []
    assert served.request("GET", "/api/job")[1] == {"state": "idle"}


def test_an_ask_job_stops_at_the_cap_keeps_what_it_paid_and_keeps_the_history(
    tmp_path: Path, monkeypatch
):
    # The estimate at 3 chars/token is well above what the fake bills (100 tokens a post), so
    # a cap between the estimate of two posts and the real cost of three lets two through.
    s = _served(tmp_path, monkeypatch, _Asker(input_tokens=5000))
    try:
        cap = s.ask_estimate({"query": QUERY, "limit": 2})["usd"]
        s.close()
        s = _served(
            tmp_path / "again",
            monkeypatch,
            _Asker(input_tokens=5000),
            jev=f"serve_max_usd = {cap}\n",
        )
        job = s.ask_run({"query": QUERY, "limit": 2})
        query = AskQuery.of(QUERY)
        records = load_asks(s.cfg.jev_asks_dir / f"{query.sha}.json", query)
        entry = load_ask_index(s.cfg.jev_asks_dir / "index.json")[query.sha]
    finally:
        s.close()

    assert job["state"] == "interrupted" and job["reason"] == "tope"
    assert job["outcome"]["ok"] == len(records) >= 1
    assert job["usd"] <= cap * (1 + 1e-9) + 5000 / 1e6 * 0.042
    assert entry.evaluated == len(records)


def test_a_page_stop_ends_an_ask_job_softly_and_the_history_has_it(tmp_path: Path, monkeypatch):
    gate = threading.Event()
    s = _served(tmp_path, monkeypatch, _Asker(gate=gate), jev="concurrency = 1\n")
    try:
        estimate = s.ask_estimate({"query": QUERY})
        assert (
            s.ask_evaluate({"query": QUERY, "confirm_token": estimate["confirm_token"]})[0] == 202
        )
        s.wait_job(lambda job: len(s.client.asked) >= 1)
        status, _, _ = s.request("POST", "/api/job/cancel", {})
        gate.set()
        job = s.wait_job()
        entry = load_ask_index(s.cfg.jev_asks_dir / "index.json")[AskQuery.of(QUERY).sha]
    finally:
        gate.set()
        s.close()

    assert status == 200
    assert (job["state"], job["reason"]) == ("interrupted", "cancelado")
    assert job["outcome"]["ok"] == 1 and job["outcome"]["logged"] is True
    assert entry.evaluated == 1


def test_an_ask_job_that_fails_says_why_and_records_no_history(tmp_path: Path, monkeypatch):
    s = _served(tmp_path, monkeypatch, _Asker(fail_when=lambda state: True))
    try:
        job = s.ask_run({"query": QUERY, "limit": 2})
    finally:
        s.close()

    assert job["state"] == "error" and job["error"]
    assert not (s.cfg.jev_asks_dir / "index.json").exists()
    (run,) = [r for r in load_runs(s.cfg.jev_runs_path) if r.kind == "ask"]
    assert run.failed == 2


# --------------------------------------------------------------------------- what was asked


def test_the_history_and_one_querys_results_are_the_blob_the_page_carries(served: _AskServed):
    served.ask_run({"query": QUERY})
    query = AskQuery.of(QUERY)

    _, history, _ = served.request("GET", "/api/asks")
    status, one, _ = served.request("GET", f"/api/ask/{query.sha}")
    _, blob, _ = served.request("GET", "/api/data")

    assert status == 200
    assert history == blob["asks"]
    assert one == blob["asks"]["history"][0]
    assert one["sha"] == query.sha and one["query"] == query.text
    assert [(r["id"], r["p"]) for r in one["results"]] == [("1", 0.97), ("3", 0.9)]
    assert one["answered"] == 5
    assert one["cost"]["cost_usd"] == pytest.approx(5 * tokens_cost_usd(100, "typesafe"))
    assert one["cost"]["requests"] == 5


@pytest.mark.parametrize("sha", ["0" * 64, "nope", "A" * 64])
def test_an_unknown_query_is_a_404(served: _AskServed, sha: str):
    status, error, _ = served.request("GET", f"/api/ask/{sha}")

    assert status == 404 and error["error"]


def test_reopening_a_query_costs_nothing(served: _AskServed):
    served.ask_run({"query": QUERY})
    built, asked = served.built, list(served.client.asked)

    for _ in range(3):
        served.request("GET", f"/api/ask/{AskQuery.of(QUERY).sha}")
        served.request("GET", "/api/asks")

    assert (served.built, served.client.asked) == (built, asked)
    assert len(load_runs(served.cfg.jev_runs_path)) == 2  # the seed pass and the one ask


def test_the_data_follows_an_ask_a_terminal_made(served: _AskServed):
    """`xbrain jev ask` in a terminal while the page is open: the next GET shows it."""
    from xbrain.jev.ask import record_ask
    from xbrain.jev.run import run_ask

    query, selection, _ = _expected(served.cfg)
    with pass_lock(served.cfg.jev_lock_path, "xbrain jev ask") as lock:
        records: dict[str, Any] = {}
        run_ask(served.cfg, selection, query, records, lambda: _Asker(), lock=lock)
        record_ask(
            served.cfg,
            query,
            filters=AskFilters(),
            evaluated=5,
            results=2,
            threshold=served.cfg.jev_threshold,
        )

    _, history, _ = served.request("GET", "/api/asks")

    assert [h["sha"] for h in history["history"]] == [query.sha]


def test_the_results_follow_a_querys_answers_file_even_without_the_history(served: _AskServed):
    """A terminal `jev ask` stopped by Ctrl-C writes the answers and not the history: the
    served results must still come from the file as it is now."""
    from xbrain.jev.store import save_asks

    served.ask_run({"query": QUERY})
    query = AskQuery.of(QUERY)
    path = served.cfg.jev_asks_dir / f"{query.sha}.json"
    records = load_asks(path, query)
    records["3"] = records["3"].model_copy(update={"probability": 0.1})
    save_asks(query, records, path)

    _, one, _ = served.request("GET", f"/api/ask/{query.sha}")

    assert [r["id"] for r in one["results"]] == ["1"]
