# tests/test_jev_serve_stream.py
"""`GET /api/job?since=<cursor>`: a running ask hands the page its answers as they are banked.

The stream is the job's own memory, fed by the pass's `on_answer` hook (`run.run_pass`, after a
record is banked): no file is read per poll. What these tests pin:

* CURSOR SEMANTICS — every answer banked is handed over exactly once across polls, whatever
  the concurrency, and each reply starts where the cursor it was given ends;
* ORDER (PR 14 arch I4) — the stream is ARRIVAL order (the answers already current first,
  ranked; then each new answer as the pool lands it); ranking is the reader's, and the stream's
  answers ranked by `(-p, id)` are exactly `finish_ask`'s results;
* ONE PATH — each answer is `ask.answer_view` as the blob carries it (probability, model,
  minute, refine keys, what Jev read), so a streamed card and the final card agree;
* BOUNDS AND GUARDS — at most `STREAM_PAGE` answers per reply, a cursor that is not one is a
  400, and the same Host/Origin guards as every GET.

Every client is a fake: no test reaches TypeSafe.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tests.test_jev_serve import _item, _repo
from tests.test_jev_serve_ask import PROBS, QUERY, _Asker, _AskServed, ranked
from xbrain.jev import service as service_module
from xbrain.jev.store import load_ask_index
from xbrain.store import load_store, save_store

#: The posts an ask over `_repo` asks (6 has no evidence).
ASKED = ["1", "2", "3", "4", "5"]


class _Metered(_Asker):
    """`_Asker` whose calls wait for a permit each: the test decides how many answers exist."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.permits = threading.Semaphore(0)

    def ask(self, state, questions):
        assert self.permits.acquire(timeout=10), "the test never let this answer through"
        return super().ask(state, questions)


def _served(tmp_path: Path, monkeypatch, client: _Asker, concurrency: int = 2) -> _AskServed:
    return _AskServed(_repo(tmp_path, monkeypatch, jev=f"concurrency = {concurrency}\n"), client)


def _poll(served: _AskServed, cursor: str, **headers: str) -> dict[str, Any]:
    status, job, _ = served.request("GET", f"/api/job?since={cursor}", headers=headers)
    assert status == 200, job
    return job


def _start(served: _AskServed, body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {"query": QUERY}
    estimate = served.ask_estimate(body)
    status, job = served.ask_evaluate({**body, "confirm_token": estimate["confirm_token"]})
    assert status == 202, job
    return job


def _drain(served: _AskServed, cursor: str = "0") -> tuple[list[dict[str, Any]], str, dict]:
    """Every answer from `cursor` on, poll after poll, until the job ended and nothing is left:
    each reply must start where the last one ended."""
    got: list[dict[str, Any]] = []
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = _poll(served, cursor)
        stream = job["stream"]
        if cursor != "0":
            assert stream["from"] == cursor
        got += stream["answers"]
        cursor = stream["next"]
        if job["state"] != "running" and not stream["more"]:
            return got, cursor, job
    raise AssertionError("the job never ended")


# --------------------------------------------------------------------------- cursor semantics


def test_each_answer_is_handed_over_once_across_polls_as_it_is_banked(tmp_path: Path, monkeypatch):
    client = _Metered()
    served = _served(tmp_path, monkeypatch, client)
    try:
        job = _start(served)
        first = _poll(served, "0")
        # Nothing answered yet: an empty page of the stream, and a cursor into THIS job.
        assert first["state"] == "running" and first["stream"]["answers"] == []
        assert first["stream"]["next"] == f"{job['number']}-0"
        assert first["stream"]["expected"] == len(ASKED)
        cursor, seen, sizes = first["stream"]["next"], [], []
        for released in (1, 2, 2):
            for _ in range(released):
                client.permits.release()
            # Poll until the answers let through are there — never more than were released.
            deadline = time.monotonic() + 10
            got: list[dict[str, Any]] = []
            while len(got) < released and time.monotonic() < deadline:
                reply = _poll(served, cursor)["stream"]
                assert reply["from"] == cursor
                got += reply["answers"]
                cursor = reply["next"]
                assert len(got) <= released
            sizes.append(len(got))
            seen += got
        rest, cursor, end = _drain(served, cursor)
        # A cursor at the end hands over nothing more, however often it is asked.
        again = [_poll(served, cursor)["stream"] for _ in range(2)]
    finally:
        served.close()

    assert sizes == [1, 2, 2] and rest == []
    assert sorted(a["id"] for a in seen) == ASKED  # every answer, none twice
    assert all(a["cached"] is False for a in seen)
    assert end["state"] == "done" and cursor == f"{job['number']}-{len(ASKED)}"
    assert [(s["answers"], s["next"], s["more"]) for s in again] == [([], cursor, False)] * 2


@pytest.mark.parametrize("pollers", [2, 3])
def test_concurrent_pollers_each_get_every_answer_exactly_once(
    tmp_path: Path, monkeypatch, pollers: int
):
    """Answers land from 4 workers at once, while several pages poll with their own cursors."""
    client = _Asker(delay=0.02)
    served = _served(tmp_path, monkeypatch, client, concurrency=4)
    results: list[list[str]] = []
    errors: list[BaseException] = []

    def _page() -> None:
        try:
            got, _, _ = _drain(served)
            results.append([a["id"] for a in got])
        except BaseException as exc:  # pragma: no cover — reported below
            errors.append(exc)

    try:
        _start(served)
        threads = [threading.Thread(target=_page) for _ in range(pollers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(15)
    finally:
        served.close()

    assert not errors, errors
    assert len(results) == pollers
    for ids in results:
        assert sorted(ids) == ASKED and len(ids) == len(set(ids))
    # One stream: every page saw the same arrival order.
    assert all(ids == results[0] for ids in results)


def test_a_cursor_into_another_job_starts_at_the_beginning_of_this_one(tmp_path: Path, monkeypatch):
    served = _served(tmp_path, monkeypatch, _Asker())
    try:
        first = _start(served, {"query": QUERY, "limit": 2})
        _, old_cursor, _ = _drain(served)
        second = _start(served, {"query": "otra pregunta"})
        served.wait_job()
        restarted = _poll(served, old_cursor)["stream"]
        rest, _, _ = _drain(served, restarted["next"])
    finally:
        served.close()

    assert old_cursor == f"{first['number']}-2"
    assert second["number"] == first["number"] + 1
    # The old job's cursor names no place in this one: its stream is handed over from 0.
    assert restarted["from"] == f"{second['number']}-0"
    assert sorted(a["id"] for a in restarted["answers"] + rest) == ASKED


def test_a_cursor_past_the_end_of_its_job_is_refused(tmp_path: Path, monkeypatch):
    served = _served(tmp_path, monkeypatch, _Asker())
    try:
        job = _start(served)
        _drain(served)
        status, error, _ = served.request("GET", f"/api/job?since={job['number']}-{len(ASKED) + 1}")
    finally:
        served.close()

    assert status == 400 and "cursor" in error["error"]


# --------------------------------------------------------------------------- order (arch I4)


def test_the_stream_is_arrival_order_and_ranked_it_is_finish_asks_results(
    tmp_path: Path, monkeypatch
):
    """The pool lands answers in its own order; the stream keeps it (the page ranks). Ranked by
    `(-p, id)` — the page's `askOrder` — they are exactly the results `finish_ask` returned."""
    finished: list[Any] = []
    real = service_module.finish_ask

    def _spy(*args: Any, **kw: Any):
        found = real(*args, **kw)
        finished.append(found)
        return found

    monkeypatch.setattr(service_module, "finish_ask", _spy)
    client = _Asker()
    served = _served(tmp_path, monkeypatch, client, concurrency=1)
    try:
        _start(served)
        got, _, _ = _drain(served)
    finally:
        served.close()

    # concurrency 1: the pool asks, and lands, in selection order — which is not rank order.
    assert [a["id"] for a in got] == client.asked == ASKED
    streamed = sorted(((a["id"], a["p"]) for a in got), key=lambda pair: (-pair[1], pair[0]))
    assert streamed == ranked()
    assert [(item.id, record.probability) for item, record in finished[0].ranked] == ranked()


def test_answers_already_current_open_the_stream_ranked_then_new_ones_arrive(
    tmp_path: Path, monkeypatch
):
    served = _served(tmp_path, monkeypatch, _Asker(), concurrency=1)
    try:
        _start(served, {"query": QUERY, "limit": 2})  # 1 and 2 answered and saved
        _drain(served)
        job = _start(served)
        got, _, end = _drain(served)
    finally:
        served.close()

    cached = [a for a in got if a["cached"]]
    fresh = [a for a in got if not a["cached"]]
    # The two answers the query already had come first, ranked; the three new ones after.
    assert [a["id"] for a in got[:2]] == ["1", "2"] and [a["id"] for a in cached] == ["1", "2"]
    assert [a["id"] for a in fresh] == ["3", "4", "5"]
    assert end["stream"]["expected"] == 5 and job["total"] == 3


def test_the_stream_says_the_minimum_the_results_open_at(tmp_path: Path, monkeypatch):
    """A page ask sends no minimum and keeps the query's last one (`finish_ask`): the live list
    opens at that same default, so it never re-cuts at the end."""
    served = _served(tmp_path, monkeypatch, _Asker())
    try:
        _start(served, {"query": QUERY, "min": 0.5, "limit": 1})
        _drain(served)
        _start(served)
        _, _, end = _drain(served)
        entry = load_ask_index(served.cfg.jev_asks_dir / "index.json").queries
    finally:
        served.close()

    assert end["stream"]["min"] == 0.5
    assert next(iter(entry.values())).last_min == 0.5


# --------------------------------------------------------------------------- one path


def test_each_streamed_answer_is_the_blobs_answer_for_that_post(tmp_path: Path, monkeypatch):
    served = _served(tmp_path, monkeypatch, _Asker())
    try:
        _start(served)
        got, _, _ = _drain(served)
        _, blob, _ = served.request("GET", "/api/data")
    finally:
        served.close()

    asks = blob["asks"]
    row = asks["history"][0]
    columns = row["answers"]
    by_id = {a["id"]: a for a in got}
    for index, post in enumerate(columns["ids"]):
        streamed = by_id[post]
        assert streamed["p"] == columns["p"][index] == PROBS[post]
        model = columns["exceptions"].get(post, {}).get("model", columns["model"])
        minute = columns["exceptions"].get(post, {}).get("asked_at", columns["asked_at"])
        assert (streamed["model"], streamed["asked_at"]) == (model, minute)
        assert streamed["keys"] == asks["keys"][post]
        # What Jev read: sent for a post whose card has no Jev block (3–5), never for one
        # whose card carries it already (1–2, evaluated by the seed pass).
        if post in asks["surfaces"]:
            assert streamed["surfaces"] == asks["surfaces"][post]
        else:
            assert "surfaces" not in streamed
    assert sorted(by_id) == sorted(columns["ids"])
    assert "surfaces" in by_id["3"] and "surfaces" not in by_id["1"]


def test_a_display_hook_that_fails_costs_no_answer(tmp_path: Path, monkeypatch):
    """The stream is display: an answer it cannot show is still banked, saved and ranked."""

    def _broken(*args: Any, **kw: Any):
        raise RuntimeError("roto")

    monkeypatch.setattr(service_module, "streamed_answer", _broken)
    served = _served(tmp_path, monkeypatch, _Asker())
    try:
        _start(served)
        got, _, end = _drain(served)
    finally:
        served.close()

    assert got == []
    assert end["state"] == "done" and end["outcome"]["ok"] == len(ASKED)
    assert end["outcome"]["results"] == len(ASKED)


# --------------------------------------------------------------------------- bounds and guards


def test_one_reply_hands_over_at_most_a_page_of_answers(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(service_module, "STREAM_PAGE", 2)
    served = _served(tmp_path, monkeypatch, _Asker())
    try:
        _start(served)
        served.wait_job()
        cursor, pages = "0", []
        while True:
            stream = _poll(served, cursor)["stream"]
            pages.append((len(stream["answers"]), stream["more"]))
            cursor = stream["next"]
            if not stream["more"]:
                break
    finally:
        served.close()

    assert pages == [(2, True), (2, True), (1, False)]


def test_a_streamed_answer_is_bounded_whatever_the_post_says(tmp_path: Path, monkeypatch):
    """What Jev read ships cut to the page's own limit per surface: a huge post does not make a
    huge reply."""
    from xbrain.jev.dashboard import PAGE_SURFACE_CHARS

    monkeypatch.setattr(service_module, "STREAM_PAGE", 1)
    cfg = _repo(tmp_path, monkeypatch, jev="concurrency = 1\n")
    store = load_store(cfg.items_path)
    store["3"] = _item("3", "Cursor rules " + "x" * 200_000)
    save_store(store, cfg.items_path)
    served = _AskServed(cfg, _Asker())
    try:
        _start(served)
        served.wait_job()
        status, _, response = served.request("GET", "/api/job?since=0")
        size = int(response.getheader("Content-Length"))
        got, _, _ = _drain(served)
    finally:
        served.close()

    assert status == 200
    big = next(a for a in got if a["id"] == "3")
    assert max(len(s["text"] or "") for s in big["surfaces"]) <= PAGE_SURFACE_CHARS
    assert size < 4 * PAGE_SURFACE_CHARS + 16_000


def test_without_since_the_job_view_is_unchanged(tmp_path: Path, monkeypatch):
    served = _served(tmp_path, monkeypatch, _Asker())
    try:
        _start(served)
        served.wait_job()
        _, job, _ = served.request("GET", "/api/job")
        _, idle_like, _ = served.request("GET", "/api/job?since=0")
    finally:
        served.close()

    assert "stream" not in job
    assert idle_like["stream"]["answers"]


def test_no_job_and_a_topics_job_have_no_stream(tmp_path: Path, monkeypatch):
    served = _served(tmp_path, monkeypatch, _Asker())
    try:
        idle = _poll(served, "0")
        estimate = served.estimate({"ids": ["3"]})
        status, _ = served.evaluate({"ids": ["3"], "confirm_token": estimate["confirm_token"]})
        assert status == 202
        served.wait_job()
        topics = _poll(served, "0")
    finally:
        served.close()

    assert idle == {"state": "idle", "stream": None}
    assert topics["kind"] == "topics" and topics["stream"] is None


@pytest.mark.parametrize(
    "query",
    [
        "since=abc",
        "since=-1",
        "since=1-",
        "since=01-2",
        "since=1-2-3",
        "since=",
        "since=0&since=0",
        "since=0&job=1",
        "after=0",
    ],
)
def test_what_is_not_a_cursor_is_refused(tmp_path: Path, monkeypatch, query: str):
    served = _served(tmp_path, monkeypatch, _Asker())
    try:
        status, error, _ = served.request("GET", "/api/job?" + query)
    finally:
        served.close()

    assert status == 400 and error["error"]


@pytest.mark.parametrize("refusal", [{"Host": "evil.example:1"}, {"Origin": "http://evil.example"}])
def test_the_stream_is_only_read_by_this_page(tmp_path: Path, monkeypatch, refusal):
    served = _served(tmp_path, monkeypatch, _Asker())
    try:
        _start(served)
        served.wait_job()
        status, body, _ = served.request("GET", "/api/job?since=0", headers=refusal)
        mine = _poll(served, "0")
    finally:
        served.close()

    assert status == 403 and "stream" not in body
    assert mine["stream"]["answers"]  # refused by the guard, not for want of a stream


def test_the_stream_is_a_get_that_changes_nothing(tmp_path: Path, monkeypatch):
    served = _served(tmp_path, monkeypatch, _Asker())
    try:
        _start(served)
        served.wait_job()
        before = sorted(p.name for p in served.cfg.jev_dir.rglob("*"))
        for _ in range(3):
            assert _poll(served, "0")["stream"]["answers"]
        status, _, _ = served.request("POST", "/api/job?since=0", {})
        after = sorted(p.name for p in served.cfg.jev_dir.rglob("*"))
    finally:
        served.close()

    assert before == after
    assert status == 404  # no POST route reads a cursor
