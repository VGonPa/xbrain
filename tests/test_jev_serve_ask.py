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
from xbrain.jev.ask import AskFilters, AskQuery, plan_ask
from xbrain.jev.client import JevResult, NoulAnswer
from xbrain.jev.defaults import tokens_cost_usd
from xbrain.jev.lock import pass_lock
from xbrain.jev.questions import ASK_KEY
from xbrain.jev.store import load_ask_index, load_asks, load_runs
from xbrain.store import load_store, save_store

QUERY = "¿Cómo configuro hooks en Claude Code?"
#: What the fake answers per post. Probability order is NOT id order (3 before 1), and 1 and
#: 5 tie, so a ranking by id or without its tie-break shows.
PROBS = {"1": 0.9, "2": 0.1, "3": 0.97, "4": 0.2, "5": 0.9}


def ranked(threshold: float = 0.85, posts: str = "12345") -> list[tuple[str, float]]:
    """THE expected results: every surface's order is compared with this one source."""
    kept = [(post, PROBS[post]) for post in posts if PROBS[post] >= threshold]
    return sorted(kept, key=lambda pair: (-pair[1], pair[0]))


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


def _plan(cfg: Config, query: str = QUERY, **filters: Any):
    """What the command's own planner selects and estimates for `query` today."""
    return plan_ask(cfg, AskQuery.of(query), AskFilters(**filters), None)


def _model(plan) -> dict[str, Any]:
    model = plan.estimate.model
    return {
        "per_call": model.per_call,
        "chars_per_token": model.chars_per_token,
        "measured": model.measured,
        "answers": model.answers,
    }


def _entry(cfg: Config, query: str = QUERY):
    return load_ask_index(cfg.jev_asks_dir / "index.json").queries[AskQuery.of(query).sha]


# --------------------------------------------------------------------------- the estimate


def test_the_ask_estimate_is_the_commands_selection_and_price(served: _AskServed):
    plan = _plan(served.cfg)
    query, priced = plan.query, plan.estimate

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
    assert estimate["cost_model"] == _model(plan) and plan.estimate.model.measured is False
    assert estimate["chars"] == priced.chars and estimate["similar"] == []
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


def test_the_ask_estimate_uses_the_model_fitted_on_paid_answers(served: _AskServed):
    served.ask_run({"query": QUERY})

    estimate = served.ask_estimate({"query": "posts sobre rondas seed"})

    plan = _plan(served.cfg, "posts sobre rondas seed")
    # Five paid answers are in the calibration; the fake bills a flat 100 tokens, so the fit
    # may keep the prior — whichever it is, the server says exactly what the planner used.
    assert load_ask_index(served.cfg.jev_asks_dir / "index.json").calibration.answers == 5
    assert estimate["cost_model"] == _model(plan)
    assert (estimate["tokens"], estimate["usd"]) == (plan.estimate.tokens, plan.estimate.usd)


def test_the_ask_estimate_names_a_query_asked_before_in_other_words(served: _AskServed):
    served.ask_run({"query": QUERY, "limit": 1})

    estimate = served.ask_estimate({"query": QUERY.lower().rstrip("?")})

    assert estimate["similar"] == [AskQuery.of(QUERY).text]


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
    assert job["outcome"]["results"] == len(ranked()) == 3
    assert job["outcome"]["recorded"] is True
    assert job["usd"] == pytest.approx(5 * tokens_cost_usd(100, "typesafe"))
    records = load_asks(served.cfg.jev_asks_dir / f"{query.sha}.json", query)
    assert {post: r.probability for post, r in records.items()} == PROBS
    (run,) = [r for r in load_runs(served.cfg.jev_runs_path) if r.kind == "ask"]
    assert (run.query_sha, run.ok, run.requests) == (query.sha, 5, 5)
    entry = _entry(served.cfg)
    assert (entry.query, entry.times, entry.last_evaluated, entry.last_results) == (
        query.text,
        1,
        5,
        3,
    )
    assert entry.last_threshold == served.cfg.jev_threshold
    assert not _locked(served.cfg.jev_lock_path)


def test_the_history_keeps_the_filters_the_job_ran_with(served: _AskServed):
    served.ask_run({"query": QUERY, "topic": "ai-coding", "limit": 2})

    entry = _entry(served.cfg)
    assert entry.last_filters == {"topic": "ai-coding"}
    assert (entry.last_evaluated, entry.last_results) == (2, 1)


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
    # Nothing was sent, so nothing was logged — and that is a clean end, not a missing line.
    assert (job["outcome"]["sent"], job["outcome"]["logged"]) == (0, False)
    assert job["outcome"]["recorded"] is True
    assert served.built == built
    assert len([r for r in load_runs(served.cfg.jev_runs_path) if r.kind == "ask"]) == 1
    entry = _entry(served.cfg)
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
        entry = _entry(s.cfg)
    finally:
        s.close()

    assert job["state"] == "interrupted" and job["reason"] == "tope"
    assert job["outcome"]["ok"] == len(records) >= 1
    assert job["usd"] <= cap * (1 + 1e-9) + 5000 / 1e6 * 0.042
    assert entry.last_evaluated == len(records)


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
        entry = _entry(s.cfg)
    finally:
        gate.set()
        s.close()

    assert status == 200
    assert (job["state"], job["reason"]) == ("interrupted", "cancelado")
    assert job["outcome"]["ok"] == 1 and job["outcome"]["logged"] is True
    assert entry.last_evaluated == 1


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
    assert {k: v for k, v in one.items() if k != "surfaces"} == blob["asks"]["history"][0]
    assert one["sha"] == query.sha and one["query"] == query.text
    assert (
        [(r["id"], r["p"]) for r in one["results"]]
        == ranked()
        == [
            ("3", 0.97),
            ("1", 0.9),
            ("5", 0.9),
        ]
    )
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
    from xbrain.jev.ask import finish_ask
    from xbrain.jev.run import run_ask

    with pass_lock(served.cfg.jev_lock_path, "xbrain jev ask") as lock:
        plan = _plan(served.cfg)
        outcome = run_ask(served.cfg, plan, lambda: _Asker(), lock=lock)
        finish_ask(served.cfg, plan, outcome, threshold=served.cfg.jev_threshold)
    query = plan.query

    _, history, _ = served.request("GET", "/api/asks")

    assert [h["sha"] for h in history["history"]] == [query.sha]


def test_the_results_follow_a_querys_answers_file_even_without_the_history(served: _AskServed):
    """A terminal `jev ask` stopped by Ctrl-C writes the answers and not the history: the
    served results must still come from the file as it is now."""
    from xbrain.jev.store import save_asks

    served.ask_run({"query": QUERY})
    query = AskQuery.of(QUERY)
    # Read once, so the server holds this data; only the answers file changes after.
    _, before, _ = served.request("GET", f"/api/ask/{query.sha}")
    assert [r["id"] for r in before["results"]] == ["3", "1", "5"]
    path = served.cfg.jev_asks_dir / f"{query.sha}.json"
    records = load_asks(path, query)
    records["3"] = records["3"].model_copy(update={"probability": 0.1})
    save_asks(query, records, path)

    _, one, _ = served.request("GET", f"/api/ask/{query.sha}")

    assert [r["id"] for r in one["results"]] == ["1", "5"]


# --------------------------------------------------------------------------- the fix wave


def test_a_history_that_cannot_be_written_keeps_the_paid_answers_in_view(
    served: _AskServed, monkeypatch
):
    """The answers are paid, saved and logged when `finish_ask` runs: its failure is said,
    beside the outcome, and never turns the job into «El trabajo falló» with nothing shown."""
    from xbrain.jev import service as service_module

    def _full_disk(*args: Any, **kwargs: Any) -> Any:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(service_module, "finish_ask", _full_disk)

    job = served.ask_run({"query": QUERY})

    query = AskQuery.of(QUERY)
    assert job["state"] == "done"
    outcome = job["outcome"]
    assert (outcome["ok"], outcome["logged"], outcome["recorded"]) == (5, True, False)
    assert "No space left on device" in outcome["history_error"]
    assert outcome["file"] == str(served.cfg.jev_asks_dir / f"{query.sha}.json")
    assert len(load_asks(served.cfg.jev_asks_dir / f"{query.sha}.json", query)) == 5


def test_a_query_the_history_lost_is_rebuilt_on_the_tab(served: _AskServed):
    served.ask_run({"query": QUERY})
    (served.cfg.jev_asks_dir / "index.json").unlink()

    _, asks, _ = served.request("GET", "/api/asks")

    [row] = asks["history"]
    assert row["rebuilt"] is True and row["sha"] == AskQuery.of(QUERY).sha
    assert [(r["id"], r["p"]) for r in row["results"]] == ranked()


class _FailsAfterStop(_Asker):
    """Its one call waits for «Parar», then fails: the use ends interrupted with nothing kept."""

    def ask(self, state, questions):
        from xbrain.jev.client import JevError

        self.asked.append("?")
        assert self.gate is not None and self.gate.wait(10)
        raise JevError("respuesta ilegible")


def test_an_ask_stopped_before_keeping_anything_is_not_recorded(tmp_path: Path, monkeypatch):
    gate = threading.Event()
    s = _served(tmp_path, monkeypatch, _FailsAfterStop(gate=gate), jev="concurrency = 1\n")
    try:
        estimate = s.ask_estimate({"query": QUERY})
        assert (
            s.ask_evaluate({"query": QUERY, "confirm_token": estimate["confirm_token"]})[0] == 202
        )
        s.wait_job(lambda job: len(s.client.asked) >= 1)
        s.request("POST", "/api/job/cancel", {})
        gate.set()
        job = s.wait_job()
    finally:
        gate.set()
        s.close()

    assert (job["state"], job["reason"]) == ("interrupted", "cancelado")
    assert (job["outcome"]["ok"], job["outcome"]["recorded"]) == (0, False)
    assert not (s.cfg.jev_asks_dir / "index.json").exists()


def test_an_ask_without_a_price_is_refused():
    from xbrain.jev.assess import Selection
    from xbrain.jev.service import _AskKind, _Priced

    priced = _Priced(
        Selection(items=(), skipped_current=0, skipped_no_evidence=0),
        ("1",),
        None,
        None,
        {"candidates": 1},
        None,
    )

    refusal = _AskKind().refusal(None, priced)  # type: ignore[arg-type]

    assert refusal is not None and "precio" in refusal


class _PerChar(_Asker):
    """Bills what the prior cost model says: 1,000 tokens a call plus a token per 4
    characters sent — so the plan's per-post prices are the real ones."""

    def ask(self, state, questions):
        from dataclasses import replace

        from xbrain.jev.ask import question_chars

        result = super().ask(state, questions)
        chars = len(state["post"]) + question_chars(questions)
        return replace(result, input_tokens=round(1000 + chars / 4))


def test_a_long_post_first_does_not_stop_an_ask_its_estimate_fits(tmp_path: Path, monkeypatch):
    """The reservation is each post's own planned price: a long post first no longer makes
    every later post look as dear as it (the job's mean), which stopped at «tope» a job whose
    estimate fitted the cap."""
    from xbrain.config import load_config

    cfg = _repo(tmp_path, monkeypatch)
    items = load_store(cfg.items_path)
    items["1"].text = "Claude Code hooks " + "x" * 60_000
    save_store(items, cfg.items_path)
    cap = _plan(cfg).estimate.usd * 1.01
    config = cfg.repo_root / "config.toml"
    config.write_text(config.read_text() + f"serve_max_usd = {cap!r}\n", encoding="utf-8")
    s = _AskServed(load_config(cfg.repo_root), _PerChar())
    try:
        job = s.ask_run({"query": QUERY})
    finally:
        s.close()

    assert s.client.asked[0] == "1"
    assert job["state"] == "done" and "reason" not in job, job
    assert job["outcome"]["ok"] == 5 and job["usd"] <= cap


def test_a_confirmation_keeps_what_the_recheck_compares_not_the_whole_plan(served: _AskServed):
    """Up to 32 confirmations live as long as the server: each keeps the selection, never the
    query's answers, every candidate's state or the history."""
    served.ask_run({"query": QUERY, "limit": 1})
    estimate = served.ask_estimate({"query": QUERY})

    confirm = served.service._confirms[estimate["confirm_token"]]
    plan = confirm.priced.context
    assert [item.id for item in plan.selection.items] == estimate["ids"]
    assert (plan.records, plan.states, plan.candidates, plan.history.queries) == ({}, {}, (), {})


def test_one_query_carries_what_jev_read_for_each_of_its_results(served: _AskServed):
    served.ask_run({"query": QUERY})

    _, one, _ = served.request("GET", f"/api/ask/{AskQuery.of(QUERY).sha}")
    _, blob, _ = served.request("GET", "/api/data")

    cards = {card["id"]: card for card in blob["posts"]}
    assert sorted(one["surfaces"]) == sorted(r["id"] for r in one["results"])
    for post, surfaces in one["surfaces"].items():
        card = cards[post]
        expected = card["jev"]["surfaces"] if card["jev"] else blob["asks"]["surfaces"][post]
        assert surfaces == expected
    # Sent once: a post whose card has a Jev block is not repeated in the tab's surfaces.
    assert not set(blob["asks"]["surfaces"]) & {p for p in cards if cards[p]["jev"]}
    assert any(cards[p]["jev"] for p in one["surfaces"]) and any(
        not cards[p]["jev"] for p in one["surfaces"]
    )


@pytest.mark.parametrize("path", ["/api/ask/estimate", "/api/ask/evaluate"])
@pytest.mark.parametrize(
    "refusal",
    [
        {"token": False},
        {"origin": "http://evil.example"},
        {"origin": None},
        {"headers": {"Host": "evil.example:1"}},
    ],
)
def test_the_ask_posts_have_the_same_guards_as_topics(
    served: _AskServed, path: str, refusal: dict[str, Any]
):
    status, error, _ = served.request("POST", path, {"query": QUERY}, **refusal)

    assert status == 403 and error["error"]
    assert served.built == 0


@pytest.mark.parametrize("path", ["/api/ask/estimate", "/api/ask/evaluate"])
def test_an_oversized_ask_body_is_refused_unread(served: _AskServed, path: str):
    status, _, _ = served.request("POST", path, raw=b"{" + b" " * 70_000 + b"}")

    assert status == 413


@pytest.mark.parametrize("path", ["/api/asks", "/api/ask/" + "a" * 64])
@pytest.mark.parametrize(
    "refusal", [{"headers": {"Host": "evil.example:1"}}, {"origin": "http://evil.example"}]
)
def test_what_was_asked_is_only_read_by_this_page(
    served: _AskServed, path: str, refusal: dict[str, Any]
):
    status, _, _ = served.request("GET", path, **refusal)

    assert status == 403


class _Triple(_PerChar):
    """Bills three times what the plan said each post would cost."""

    def ask(self, state, questions):
        from dataclasses import replace

        result = super().ask(state, questions)
        return replace(result, input_tokens=3 * (result.input_tokens or 0))


def test_answers_dearer_than_planned_raise_the_next_reservations(tmp_path: Path, monkeypatch):
    """Planned p a post, billed 3p, cap 1.6 × the estimate (8p for five posts): reserving only
    p would send the third post at 6p + p ≤ 8p and end at 9p, over the cap. Scaled by the
    job's real/planned ratio, the third reserves 3p and is not sent."""
    from xbrain.config import load_config

    cfg = _repo(tmp_path, monkeypatch)
    cap = _plan(cfg).estimate.usd * 1.6
    config = cfg.repo_root / "config.toml"
    config.write_text(config.read_text() + f"serve_max_usd = {cap!r}\n", encoding="utf-8")
    s = _AskServed(load_config(cfg.repo_root), _Triple())
    try:
        job = s.ask_run({"query": QUERY})
    finally:
        s.close()

    assert (job["state"], job["reason"]) == ("interrupted", "tope")
    assert job["outcome"]["ok"] == 2
    assert job["usd"] <= cap
