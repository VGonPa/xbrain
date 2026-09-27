# tests/test_jev_serve.py
"""`xbrain jev serve`: the local server that lets the page ask Jev — over HTTP, with the fake.

Every test here talks to a real `ThreadingHTTPServer` on 127.0.0.1 through `http.client`, and
every client the server builds is `tests/jev_fakes.FakeJevClient`: no test reaches TypeSafe.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from tests.jev_fakes import FakeJevClient
from xbrain.config import Config, load_config
from xbrain.jev.assess import select_items
from xbrain.jev.lock import PassLockBusy, pass_lock
from xbrain.jev.run import run_topics
from xbrain.jev.serve import TOKEN_HEADER, make_server, serve_until_interrupted
from xbrain.jev.service import JevService
from xbrain.jev.store import load_assessments, load_runs
from xbrain.models import Author, Enrichment, Item, Topic
from xbrain.rubrics import save_vocab
from xbrain.store import save_store

DT = datetime(2026, 9, 22, tzinfo=timezone.utc)
VOCAB = [
    Topic(slug="ai-coding", description="Construir software con IA."),
    Topic(slug="startups", description="Fundar empresas."),
]
#: 100 input tokens per call (the fake's default) at 0.042 $/MTok.
PER_POST_USD = 100 / 1e6 * 0.042


def _item(item_id: str, text: str, topics: tuple[str, ...] = ("ai-coding",)) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=text,
        created_at=DT,
        captured_at=DT,
        enriched=Enrichment(
            enriched_at=DT,
            executor="claude-code",
            summary="s",
            primary_topic=topics[0],
            topics=list(topics),
        ),
    )


#: Six posts: 1–2 evaluated by the seed pass, 3–5 not yet, 6 has no evidence at all.
ITEMS = {
    "1": _item("1", "Claude Code hooks"),
    "2": _item("2", "Seed round tips", ("startups",)),
    "3": _item("3", "Cursor rules"),
    "4": _item("4", "Series A metrics", ("startups",)),
    "5": _item("5", "Agents that write tests"),
    "6": _item("6", " "),
}
# No text and no author: nothing to ask about (`select_items` counts it, never asks it).
ITEMS["6"].author = Author(handle="", name="")


def _locked(path: Path) -> bool:
    """Whether a pass holds the lock at `path` — asked by trying to take it, as a pass would."""
    try:
        with pass_lock(path, "probe"):
            return False
    except PassLockBusy:
        return True


def _repo(tmp_path: Path, monkeypatch, jev: str = "", seed_tokens: int = 100) -> Config:
    vault = tmp_path / "vault"
    (vault / "x" / "_media").mkdir(parents=True)
    (tmp_path / "config.toml").write_text(
        f'[paths]\nvault = "{vault}"\noutput_subdir = "x"\ndata_dir = "data"\n'
        f'[x]\nhandle = "v"\n[jev]\n' + ("" if "concurrency" in jev else "concurrency = 1\n") + jev,
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    save_store(ITEMS, tmp_path / "data" / "items.json")
    save_vocab(VOCAB, tmp_path / "data" / "vocab.yaml")
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    cfg = load_config(tmp_path)
    _seed_pass(cfg, ["1", "2"], tokens=seed_tokens)
    return cfg


def _seed_pass(cfg: Config, ids: list[str], tokens: int = 100) -> None:
    """A priced pass over `ids` (`tokens` per answer), so the page has a mean cost per post to
    estimate from."""
    with pass_lock(cfg.jev_lock_path, "seed") as lock:
        assessments = load_assessments(cfg.jev_topics_path)
        selection = select_items(
            ITEMS,
            assessments,
            VOCAB,
            ids=ids,
            limit=None,
            force=False,
            fallback=cfg.jev_fallback_option,
            char_limit=cfg.jev_state_char_limit,
        )
        run_topics(
            cfg,
            selection,
            assessments,
            VOCAB,
            lambda: FakeJevClient(
                provider="typesafe", nouls={"ai-coding": 0.95}, input_tokens=tokens
            ),
            lock=lock,
        )


class _Recorder(FakeJevClient):
    """A priced fake that remembers which POST each call was about (by its text)."""

    def __init__(self, *, delay: float = 0.0, **kwargs: Any) -> None:
        super().__init__(provider="typesafe", **kwargs)
        self.delay = delay
        self.asked: list[str] = []

    def ask(self, state, questions):
        text = state["post"]
        post = next(
            i for i, item in ITEMS.items() if item.text.strip() and text.startswith(item.text)
        )
        self.asked.append(post)
        if self.delay:
            time.sleep(self.delay)
        return super().ask(state, questions)


class _Served:
    """A running server over one repo, and the fake every job of it asks."""

    def __init__(self, cfg: Config, client: FakeJevClient) -> None:
        self.cfg = cfg
        self.client = client
        self.built = 0

        def _make() -> FakeJevClient:
            self.built += 1
            return client

        self.service = JevService(cfg, _make)
        self.server = make_server(self.service, 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self.thread.start()

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        headers: dict[str, str] | None = None,
        token: bool = True,
        origin: str | None = "default",
        raw: bytes | None = None,
    ) -> tuple[int, Any, http.client.HTTPResponse]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        sent: dict[str, str] = {}
        if method == "POST":
            sent["Content-Type"] = "application/json"
            if token:
                sent[TOKEN_HEADER] = self.service.token
            if origin == "default":
                sent["Origin"] = f"http://127.0.0.1:{self.port}"
        if origin not in (None, "default"):
            sent["Origin"] = origin
        sent.update(headers or {})
        payload = raw if raw is not None else (None if body is None else json.dumps(body))
        conn.request(method, path, body=payload, headers=sent)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        kind = response.getheader("Content-Type") or ""
        return response.status, (json.loads(data) if "json" in kind else data), response

    def estimate(self, body: dict[str, Any]) -> dict[str, Any]:
        status, data, _ = self.request("POST", "/api/topics/estimate", body)
        assert status == 200, data
        return data

    def evaluate(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        status, data, _ = self.request("POST", "/api/topics/evaluate", body)
        return status, data

    def wait_job(self, until: Callable[[dict[str, Any]], bool] | None = None) -> dict[str, Any]:
        """The job view once `until` holds (by default: once it is no longer running)."""
        until = until or (lambda job: job["state"] != "running")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            _, job, _ = self.request("GET", "/api/job")
            if until(job):
                return job
            time.sleep(0.01)
        raise AssertionError(f"the job never got there: {job}")

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.service.stop()


@pytest.fixture
def served(tmp_path: Path, monkeypatch) -> Iterator[_Served]:
    s = _Served(_repo(tmp_path, monkeypatch), _Recorder())
    yield s
    s.close()


# --------------------------------------------------------------------------- where it listens


def test_the_server_binds_to_loopback_only(served: _Served):
    assert served.server.server_address[0] == "127.0.0.1"


def test_make_server_has_no_way_to_bind_another_host():
    import inspect

    assert "host" not in inspect.signature(make_server).parameters


# --------------------------------------------------------------------------- the page and data


def test_the_page_is_the_live_page_with_its_token_and_the_served_flag(served: _Served):
    status, html, response = served.request("GET", "/")
    blob = json.loads(html.decode().split("let DATA = ", 1)[1].split(";\n", 1)[0])

    assert status == 200
    assert response.getheader("Content-Type") == "text/html; charset=utf-8"
    assert blob["serve"] == {"token": served.service.token, "max_usd": 1.0, "finished_at": None}
    page_row = next(row for row in blob["config"]["files"] if row["key"] == "page")
    assert page_row["served"] is True
    assert sorted(p["id"] for p in blob["posts"] if p["status"] == "compared") == ["1", "2"]


def test_api_data_is_the_same_blob_the_page_carries(served: _Served):
    _, html, _ = served.request("GET", "/")
    page_blob = json.loads(html.decode().split("let DATA = ", 1)[1].split(";\n", 1)[0])

    status, blob, response = served.request("GET", "/api/data")

    assert status == 200 and response.getheader("Cache-Control") == "no-store"
    assert blob == page_blob


def test_api_cards_returns_card_bodies_by_id_in_the_order_asked(served: _Served):
    _, blob, _ = served.request("GET", "/api/data")
    by_id = {p["id"]: p for p in blob["posts"]}

    status, cards, _ = served.request("GET", "/api/cards?ids=3,1")

    assert status == 200
    assert cards == {"cards": [by_id["3"], by_id["1"]]}
    status, error, _ = served.request("GET", "/api/cards?ids=1,nope")
    assert status == 404 and "nope" in error["error"]


def test_the_data_follows_the_files_a_terminal_pass_writes(served: _Served):
    """A pass from the terminal while the page is open: the next GET shows it."""
    _seed_pass(served.cfg, ["3"])

    _, blob, _ = served.request("GET", "/api/data")

    assert next(p for p in blob["posts"] if p["id"] == "3")["status"] == "compared"


# --------------------------------------------------------------------------- static files


def test_media_is_served_from_the_output_dirs_media_folder(served: _Served):
    photo = served.cfg.output_dir / "_media" / "photos" / "a.jpg"
    photo.parent.mkdir(parents=True)
    photo.write_bytes(b"\xff\xd8jpeg")

    status, data, response = served.request("GET", "/_media/photos/a.jpg")

    assert (status, data) == (200, b"\xff\xd8jpeg")
    assert response.getheader("Content-Type") == "image/jpeg"


@pytest.mark.parametrize(
    "path",
    [
        "/_media/../../../data/items.json",
        "/_media/%2e%2e/%2e%2e/%2e%2e/data/items.json",
        "/_media/..%2f..%2f..%2fdata%2fitems.json",
        "/../data/items.json",
        "/config.toml",
        "/items/some-note.md",
        "/_media/link-out/items.json",
        "/_media/",
    ],
)
def test_no_path_reaches_outside_the_media_folder(served: _Served, path: str):
    (served.cfg.output_dir / "items").mkdir()
    (served.cfg.output_dir / "items" / "some-note.md").write_text("nota")
    # A symlink inside `_media/` that points out of it is outside, however it is spelled.
    (served.cfg.output_dir / "_media" / "link-out").symlink_to(served.cfg.data_dir)

    status, data, _ = served.request("GET", path)

    assert status == 404
    assert b"Claude Code hooks" not in (data if isinstance(data, bytes) else b"")


# --------------------------------------------------------------------------- who may ask


def _estimate_body() -> dict[str, Any]:
    return {"ids": ["3"]}


def test_a_post_without_the_token_is_refused(served: _Served):
    status, error, _ = served.request("POST", "/api/topics/estimate", _estimate_body(), token=False)

    assert status == 403 and "token" in error["error"]


def test_a_post_with_a_wrong_token_is_refused(served: _Served):
    status, _, _ = served.request(
        "POST",
        "/api/topics/estimate",
        _estimate_body(),
        token=False,
        headers={TOKEN_HEADER: "x" * 43},
    )

    assert status == 403


@pytest.mark.parametrize(
    "origin", [None, "http://evil.example", "http://127.0.0.1:1", "null", "https://127.0.0.1"]
)
def test_a_post_from_another_origin_or_none_is_refused(served: _Served, origin: str | None):
    status, error, _ = served.request(
        "POST", "/api/topics/estimate", _estimate_body(), origin=origin
    )

    assert status == 403 and "Origin" in error["error"]


def test_localhost_is_the_same_origin_as_127_0_0_1(served: _Served):
    status, _, _ = served.request(
        "POST",
        "/api/topics/estimate",
        _estimate_body(),
        origin=f"http://localhost:{served.port}",
        headers={"Host": f"localhost:{served.port}"},
    )

    assert status == 200


@pytest.mark.parametrize(
    "method,path", [("GET", "/api/data"), ("GET", "/"), ("POST", "/api/topics/estimate")]
)
def test_a_foreign_host_header_is_refused_on_every_route(served: _Served, method: str, path: str):
    """DNS rebinding: a page on evil.example resolved to 127.0.0.1 sends `Host: evil.example`.
    The page and /api/data carry the token, so GET is guarded too."""
    body = _estimate_body() if method == "POST" else None

    status, data, _ = served.request(method, path, body, headers={"Host": "evil.example:8765"})

    assert status == 403
    assert served.service.token.encode() not in (data if isinstance(data, bytes) else b"")


def test_a_post_that_is_not_json_is_refused(served: _Served):
    status, _, _ = served.request(
        "POST", "/api/topics/estimate", raw=b"ids=3", headers={"Content-Type": "text/plain"}
    )

    assert status == 415


def test_a_body_too_large_is_refused_unread(served: _Served):
    status, _, _ = served.request("POST", "/api/topics/estimate", raw=b"{" + b" " * 70_000 + b"}")

    assert status == 413


# --------------------------------------------------------------------------- the estimate


def test_the_estimate_counts_what_the_command_would_ask_and_prices_it(served: _Served):
    estimate = served.estimate({"unevaluated": 10})

    assert estimate["ids"] == ["3", "4", "5"]
    assert (estimate["posts"], estimate["skipped_current"], estimate["skipped_no_evidence"]) == (
        3,
        2,
        1,
    )
    assert estimate["usd"] == pytest.approx(3 * PER_POST_USD)
    assert estimate["tokens"] == 300
    assert estimate["estimate"] is True
    assert estimate["per_post"] == {"n": 2, "of": 2}
    assert estimate["allowed"] is True and estimate["confirm_token"]


def test_the_estimate_of_n_unevaluated_takes_the_first_n(served: _Served):
    estimate = served.estimate({"unevaluated": 2})

    assert (estimate["ids"], estimate["remaining"]) == (["3", "4"], 1)


def test_a_topic_selects_its_posts_and_skips_the_current_ones(served: _Served):
    estimate = served.estimate({"topic": "startups"})

    assert estimate["ids"] == ["4"]
    assert estimate["skipped_current"] == 1


def test_a_topic_with_no_posts_selects_nothing_never_the_whole_corpus(served: _Served, tmp_path):
    """`select_items(ids=[])` means EVERY post: an empty topic must not become that."""
    save_vocab([*VOCAB, Topic(slug="quantum", description="Qubits.")], served.cfg.vocab_path)

    estimate = served.estimate({"topic": "quantum"})

    assert estimate["posts"] == 0 and estimate["ids"] == []
    assert estimate["allowed"] is False and estimate["confirm_token"] is None


def test_an_unknown_topic_is_refused(served: _Served):
    status, error, _ = served.request("POST", "/api/topics/estimate", {"topic": "nope"})

    assert status == 400 and "nope" in error["error"]


def test_force_re_asks_current_posts_and_says_so(served: _Served):
    estimate = served.estimate({"ids": ["1", "3"], "force": True})

    assert estimate["ids"] == ["1", "3"] and estimate["forced"] == 1


def test_a_pair_selects_the_posts_behind_it(served: _Served):
    _, blob, _ = served.request("GET", "/api/data")
    kind, sets = next((k, v) for k, v in blob["post_sets"].items() if k != "bands" and v)
    key, ids = next(iter(sets.items()))

    estimate = served.estimate({"pair": {"kind": kind, "key": key}, "force": True})

    assert estimate["ids"] == ids


def test_a_band_selects_the_posts_in_it(served: _Served):
    _, blob, _ = served.request("GET", "/api/data")
    key, ids = next((k, v) for k, v in blob["post_sets"]["bands"].items() if v)

    estimate = served.estimate({"band": key, "force": True})

    assert estimate["ids"] == ids


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"ids": ["3"], "topic": "startups"},
        {"ids": []},
        {"ids": "3"},
        {"unevaluated": 0},
        {"unevaluated": True},
        {"unevaluated": 3, "force": True},
        {"ids": ["3"], "force": "yes"},
        {"pair": {"kind": "zz", "key": "a~b"}},
        {"pair": {"kind": "cx", "key": "no~such"}},
        {"band": "nope"},
        {"ids": ["3"], "surprise": 1},
        [1, 2],
    ],
)
def test_a_malformed_selection_is_refused(served: _Served, body: Any):
    status, error, _ = served.request("POST", "/api/topics/estimate", body)

    assert status == 400 and error["error"]


def test_an_unknown_id_is_refused_by_name(served: _Served):
    status, error, _ = served.request("POST", "/api/topics/estimate", {"ids": ["3", "nope"]})

    assert status == 400 and "nope" in error["error"]


def test_an_estimate_over_the_cap_is_not_allowed_and_says_why(tmp_path: Path, monkeypatch):
    cfg = _repo(tmp_path, monkeypatch, jev="serve_max_usd = 0.000005\n")
    s = _Served(cfg, _Recorder())
    try:
        over = s.estimate({"ids": ["3", "4"]})
        under = s.estimate({"ids": ["3"]})
    finally:
        s.close()

    assert over["allowed"] is False and over["confirm_token"] is None
    assert "serve_max_usd" in over["refusal"]
    assert under["allowed"] is True


def test_with_no_priced_answer_there_is_no_estimate_and_no_job(tmp_path: Path, monkeypatch):
    """Fail-closed: without a mean cost the cap cannot be checked, so nothing is allowed."""
    cfg = _repo(tmp_path, monkeypatch)
    cfg.jev_topics_path.unlink()
    s = _Served(cfg, _Recorder())
    try:
        estimate = s.estimate({"ids": ["3"]})
    finally:
        s.close()

    assert estimate["usd"] is None and estimate["allowed"] is False
    assert "xbrain jev topics --limit" in estimate["refusal"]


# --------------------------------------------------------------------------- the job


def test_evaluate_runs_exactly_what_was_estimated_and_logs_the_pass(served: _Served):
    estimate = served.estimate({"unevaluated": 10})

    status, job = served.evaluate({"unevaluated": 10, "confirm_token": estimate["confirm_token"]})
    done = served.wait_job()

    assert status == 202 and job["state"] == "running"
    assert served.client.asked == estimate["ids"]
    assert done["state"] == "done"
    assert (done["done"], done["total"], done["outcome"]["ok"]) == (3, 3, 3)
    assert done["outcome"]["ids"] == ["3", "4", "5"]
    assert done["usd"] == pytest.approx(3 * PER_POST_USD)
    assert set(load_assessments(served.cfg.jev_topics_path)) == {"1", "2", "3", "4", "5"}
    runs = load_runs(served.cfg.jev_runs_path)
    assert len(runs) == 2 and (runs[-1].requests, runs[-1].ok) == (3, 3)
    assert done["outcome"]["logged"] is True
    assert served.client.closed is True


def test_the_data_after_a_job_shows_the_new_answers(served: _Served):
    estimate = served.estimate({"ids": ["3"]})
    served.evaluate({"ids": ["3"], "confirm_token": estimate["confirm_token"]})
    served.wait_job()

    _, blob, _ = served.request("GET", "/api/data")

    assert next(p for p in blob["posts"] if p["id"] == "3")["status"] == "compared"


def test_evaluate_without_a_confirmation_is_refused_and_costs_nothing(served: _Served):
    status, error, _ = served.request("POST", "/api/topics/evaluate", {"ids": ["3"]})

    assert status == 400 and "confirm_token" in error["error"]
    assert served.built == 0


def test_a_confirmation_is_bound_to_its_selection(served: _Served):
    """Bound to the selection AS ASKED, not only to the posts it resolved to: «post 3» and
    «the next unevaluated one» are both post 3 today, and a confirmation of one is still not
    a confirmation of the other."""
    estimate = served.estimate({"ids": ["3"]})

    status, error = served.evaluate({"unevaluated": 1, "confirm_token": estimate["confirm_token"]})

    assert status == 409 and "no corresponde" in error["error"]
    assert served.built == 0


def test_a_confirmation_is_used_once(served: _Served):
    """Forced, so the same selection is still exactly the same posts afterwards: only the
    spent confirmation can refuse the second call."""
    estimate = served.estimate({"ids": ["3"], "force": True})
    served.evaluate({"ids": ["3"], "force": True, "confirm_token": estimate["confirm_token"]})
    served.wait_job()

    status, error = served.evaluate(
        {"ids": ["3"], "force": True, "confirm_token": estimate["confirm_token"]}
    )

    assert status == 409 and "ya se usó" in error["error"]
    assert served.built == 1


def test_a_selection_that_changed_since_the_estimate_is_refused(served: _Served):
    estimate = served.estimate({"unevaluated": 10})
    _seed_pass(served.cfg, ["3"])  # a terminal pass answers one of them meanwhile

    status, error = served.evaluate({"unevaluated": 10, "confirm_token": estimate["confirm_token"]})

    assert status == 409 and "vuelve a estimar" in error["error"]
    assert served.built == 0


def test_the_cap_is_refused_server_side_even_with_a_token(tmp_path: Path, monkeypatch):
    """A confirm token is only minted under the cap; a forged one is not a token."""
    cfg = _repo(tmp_path, monkeypatch, jev="serve_max_usd = 0.000005\n")
    s = _Served(cfg, _Recorder())
    try:
        s.estimate({"ids": ["3", "4"]})
        status, error = s.evaluate({"ids": ["3", "4"], "confirm_token": "forged"})
    finally:
        s.close()

    assert status == 409 and s.built == 0


def test_a_job_stops_when_what_it_really_spent_reaches_the_cap(tmp_path: Path, monkeypatch):
    """The estimate is a mean; the real bill can be larger. The job is stopped at the cap."""
    cfg = _repo(tmp_path, monkeypatch, jev="serve_max_usd = 0.000015\n")
    client = _Recorder(input_tokens=250)  # 2.5× the mean the estimate used
    s = _Served(cfg, client)
    try:
        estimate = s.estimate({"unevaluated": 3})
        assert estimate["allowed"] is True  # 3 × 4.2e-6 = 1.26e-5 under 1.5e-5
        s.evaluate({"unevaluated": 3, "confirm_token": estimate["confirm_token"]})
        job = s.wait_job()
    finally:
        s.close()

    assert job["state"] == "interrupted" and job["reason"] == "tope"
    # The first answer cost 1.05e-5 (2.5× the mean): the next post is reserved at THAT, and
    # 1.05e-5 + 1.05e-5 would pass 1.5e-5 — so it is never sent.
    assert client.asked == ["3"]
    assert load_runs(cfg.jev_runs_path)[-1].interrupted is True


def test_only_one_job_at_a_time(tmp_path: Path, monkeypatch):
    s = _Served(_repo(tmp_path, monkeypatch), _Recorder(delay=0.2))
    try:
        first = s.estimate({"ids": ["3"]})
        second = s.estimate({"ids": ["4"]})
        s.evaluate({"ids": ["3"], "confirm_token": first["confirm_token"]})
        status, error = s.evaluate({"ids": ["4"], "confirm_token": second["confirm_token"]})
        s.wait_job()
        # Refused BEFORE the confirmation is spent: once the first job ends, it still runs.
        later, _ = s.evaluate({"ids": ["4"], "confirm_token": second["confirm_token"]})
        s.wait_job()
    finally:
        s.close()

    assert status == 409 and error["error"].startswith("ya hay un trabajo en curso")
    assert later == 202


def test_the_job_holds_the_pass_lock_and_releases_it(tmp_path: Path, monkeypatch):
    cfg = _repo(tmp_path, monkeypatch)
    seen: list[bool] = []

    class _Watching(_Recorder):
        def ask(self, state, questions):
            seen.append(_locked(cfg.jev_lock_path))
            return super().ask(state, questions)

    s = _Served(cfg, _Watching())
    try:
        estimate = s.estimate({"ids": ["3"]})
        s.evaluate({"ids": ["3"], "confirm_token": estimate["confirm_token"]})
        s.wait_job()
    finally:
        s.close()

    assert seen == [True]
    assert _locked(cfg.jev_lock_path) is False
    with pass_lock(cfg.jev_lock_path, "after"):
        pass


def test_a_terminal_pass_holding_the_lock_refuses_the_job(served: _Served):
    estimate = served.estimate({"ids": ["3"]})

    with pass_lock(served.cfg.jev_lock_path, "xbrain jev topics"):
        status, error = served.evaluate({"ids": ["3"], "confirm_token": estimate["confirm_token"]})

    assert status == 409 and "xbrain jev topics" in error["error"]
    assert served.built == 0


def test_a_job_that_fails_reports_the_error_and_still_logs(tmp_path: Path, monkeypatch):
    cfg = _repo(tmp_path, monkeypatch)
    s = _Served(cfg, _Recorder(fail_when=lambda state: True))
    try:
        estimate = s.estimate({"ids": ["3"]})
        s.evaluate({"ids": ["3"], "confirm_token": estimate["confirm_token"]})
        job = s.wait_job()
    finally:
        s.close()

    assert job["state"] == "error" and "ninguna de las 1" in job["error"]
    assert load_runs(cfg.jev_runs_path)[-1].failed == 1


def test_a_job_whose_key_is_missing_reports_it(tmp_path: Path, monkeypatch):
    from xbrain.jev.client import JevError

    cfg = _repo(tmp_path, monkeypatch)

    def _no_key() -> FakeJevClient:
        raise JevError("TYPESAFE_API_KEY no encontrada")

    service = JevService(cfg, _no_key)
    server = make_server(service, 0)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        estimate = service.estimate("topics", {"ids": ["3"]})
        service.evaluate("topics", {"ids": ["3"], "confirm_token": estimate["confirm_token"]})
        service.wait()
        job = service.job_view()
    finally:
        server.shutdown()
        server.server_close()

    assert job["state"] == "error" and "TYPESAFE_API_KEY" in job["error"]


def test_the_job_view_before_any_job_is_idle(served: _Served):
    status, job, _ = served.request("GET", "/api/job")

    assert status == 200 and job == {"state": "idle"}


# --------------------------------------------------------------------------- Ctrl-C


def test_ctrl_c_stops_accepting_lets_the_job_checkpoint_and_log_and_exits_130(
    tmp_path: Path, monkeypatch
):
    cfg = _repo(tmp_path, monkeypatch)
    client = _Recorder(delay=0.05)
    service = JevService(cfg, lambda: client)
    server = make_server(service, 0)
    port = server.server_address[1]
    estimate = service.estimate("topics", {"unevaluated": 10})
    service.evaluate("topics", {"unevaluated": 10, "confirm_token": estimate["confirm_token"]})

    def _interrupted_after_one_answer() -> None:
        deadline = time.monotonic() + 10
        while service.job_view().get("done", 0) < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        raise KeyboardInterrupt

    monkeypatch.setattr(server, "serve_forever", _interrupted_after_one_answer)

    code, last = serve_until_interrupted(server, service)

    assert code == 130
    job = service.job_view()
    assert last == job
    assert job["state"] == "interrupted" and job["reason"] == "servidor parado"
    assert len(client.asked) < 3
    kept = set(load_assessments(cfg.jev_topics_path)) - {"1", "2"}
    assert kept == set(client.asked)
    assert load_runs(cfg.jev_runs_path)[-1].interrupted is True
    assert _locked(cfg.jev_lock_path) is False
    with pytest.raises(OSError):
        http.client.HTTPConnection("127.0.0.1", port, timeout=1).request("GET", "/api/job")


# =========================================================================== PR 10a fix wave
#
# Every test above runs at concurrency 1, which hides what posts in flight do to a cap. These
# run the default shape: several workers, more posts queued than workers.

MEAN_USD = PER_POST_USD  # the seed pass: 100 tokens per answer, priced


def _big_repo(
    tmp_path: Path, monkeypatch, *, posts: int, jev: str, seed_tokens: int = 100
) -> Config:
    """`_repo` with `posts` more unevaluated posts (`p00`, `p01`, …) and `jev` settings."""
    cfg = _repo(tmp_path, monkeypatch, jev=jev, seed_tokens=seed_tokens)
    extra = {
        f"p{n:02d}": _item(f"p{n:02d}", f"Post número {n:02d} sobre agentes") for n in range(posts)
    }
    ITEMS.update(extra)
    try:
        save_store(ITEMS, cfg.items_path)
    finally:
        for key in extra:
            ITEMS.pop(key)
    return cfg


class _Priced(FakeJevClient):
    """A fake that sleeps a little (so workers overlap) and answers `tokens` per call."""

    def __init__(self, *, delay: float = 0.02, **kwargs: Any) -> None:
        kwargs.setdefault("provider", "typesafe")
        super().__init__(**kwargs)
        self.delay = delay
        self.lock = threading.Lock()
        self.sent = 0

    def ask(self, state, questions):
        with self.lock:
            self.sent += 1
        time.sleep(self.delay)
        return super().ask(state, questions)


def _run_job(s: _Served, body: dict[str, Any]) -> dict[str, Any]:
    estimate = s.estimate(body)
    assert estimate["allowed"], estimate
    status, job = s.evaluate({**body, "confirm_token": estimate["confirm_token"]})
    assert status == 202, job
    return s.wait_job()


def test_the_cap_is_a_hard_bound_at_concurrency_with_posts_queued(tmp_path: Path, monkeypatch):
    """4 workers, 10 posts estimated exactly at the cap, each really 1.5× the mean: the bill
    passes the cap by at most what the posts in flight cost above their reservation — and
    every answer that came back is stored, and logged with its tokens."""
    cap = 10 * MEAN_USD
    cfg = _big_repo(
        tmp_path, monkeypatch, posts=12, jev=f"concurrency = 4\nserve_max_usd = {cap!r}\n"
    )
    client = _Priced(input_tokens=150)
    s = _Served(cfg, client)
    try:
        job = _run_job(s, {"unevaluated": 10})
    finally:
        s.close()

    per_post = 150 / 1e6 * 0.042
    run = load_runs(cfg.jev_runs_path)[-1]
    stored = set(load_assessments(cfg.jev_topics_path)) - {"1", "2"}
    assert job["state"] == "interrupted" and job["reason"] == "tope"
    assert client.sent < 10
    assert len(stored) == client.sent == job["answered"] == run.ok == run.requests
    assert run.input_tokens == 150 * client.sent == job["tokens"]
    assert run.unsaved == 0 and run.interrupted is True
    assert job["usd"] == pytest.approx(client.sent * per_post)
    # Every post in flight may cost more than its reservation: at most `concurrency` of them.
    assert job["usd"] <= cap + 4 * (per_post - MEAN_USD) + 1e-12


def test_a_job_that_finishes_every_post_at_the_cap_is_done_without_a_reason(
    tmp_path: Path, monkeypatch
):
    cap = 3 * MEAN_USD
    cfg = _big_repo(
        tmp_path, monkeypatch, posts=0, jev=f"concurrency = 3\nserve_max_usd = {cap!r}\n"
    )
    s = _Served(cfg, _Priced(input_tokens=100))
    try:
        estimate = s.estimate({"unevaluated": 3})
        job = _run_job(s, {"unevaluated": 3})
    finally:
        s.close()

    assert estimate["usd"] == pytest.approx(cap) and estimate["allowed"] is True  # == is allowed
    assert job["state"] == "done" and "reason" not in job
    assert job["outcome"]["ok"] == 3


@pytest.mark.parametrize(
    "kwargs", [{"provider": "fake"}, {"input_tokens": None}], ids=["unpriced", "no-tokens"]
)
def test_answers_that_cannot_be_priced_are_charged_the_reservation_never_zero(
    tmp_path: Path, monkeypatch, kwargs: dict[str, Any]
):
    """A provider with no price (`jev-latest` answering as a new one) or no usage reported
    would otherwise spend at $0 under the cap forever."""
    cfg = _big_repo(tmp_path, monkeypatch, posts=6, jev="concurrency = 2\n")
    s = _Served(cfg, _Priced(**kwargs))
    try:
        job = _run_job(s, {"unevaluated": 5})
    finally:
        s.close()

    assert job["usd"] == pytest.approx(5 * MEAN_USD)
    assert job["charged_at_estimate"] == 5
    if "provider" in kwargs:
        assert job["unpriced_providers"] == ["fake"] and job["tokens"] == 500
    else:
        assert job["tokens_unknown"] == 5 and job["tokens"] == 0


def test_unpriced_answers_count_against_the_cap(tmp_path: Path, monkeypatch):
    """Charged at the reservation, unpriced answers fill the cap like priced ones: a job
    whose own priced mean then rises stops instead of spending at $0."""
    cap = 4 * MEAN_USD
    cfg = _big_repo(
        tmp_path, monkeypatch, posts=6, jev=f"concurrency = 1\nserve_max_usd = {cap!r}\n"
    )

    class _HalfUnpriced(_Priced):
        def ask(self, state, questions):
            result = super().ask(state, questions)
            from dataclasses import replace

            # The first answer is expensive and priced; the rest are unpriced.
            if self.sent == 1:
                return replace(result, input_tokens=300)
            return replace(result, provider="fake")

    client = _HalfUnpriced()
    s = _Served(cfg, client)
    try:
        job = _run_job(s, {"unevaluated": 4})
    finally:
        s.close()

    # 1 × 1.26e-5 (priced, 3× the mean) then each unpriced post reserved at that mean: the
    # second would already pass 4 × 4.2e-6.
    assert client.sent == 1 and job["reason"] == "tope"


def test_a_server_stop_waits_for_the_calls_in_flight_and_keeps_every_one(
    tmp_path: Path, monkeypatch
):
    cfg = _big_repo(tmp_path, monkeypatch, posts=12, jev="concurrency = 3\n")
    client = _Priced(delay=0.1)
    s = _Served(cfg, client)
    try:
        estimate = s.estimate({"unevaluated": 12})
        s.evaluate({"unevaluated": 12, "confirm_token": estimate["confirm_token"]})
        s.wait_job(lambda job: job.get("answered", 0) >= 1)
        s.service.stop()
        job = s.service.job_view()
    finally:
        s.close()

    run = load_runs(cfg.jev_runs_path)[-1]
    stored = set(load_assessments(cfg.jev_topics_path)) - {"1", "2"}
    assert job["state"] == "interrupted" and job["reason"] == "servidor parado"
    assert client.sent < 12
    assert len(stored) == client.sent == run.ok == run.requests and run.unsaved == 0
    assert run.input_tokens == 100 * client.sent


def test_the_job_view_is_frozen_once_the_job_ends(served: _Served):
    job = _run_job(served, {"ids": ["3"]})

    time.sleep(0.05)
    assert served.service.job_view() == job


# --------------------------------------------------------------------------- stopping


def test_a_stopping_server_refuses_new_jobs_with_503(served: _Served):
    estimate = served.estimate({"ids": ["3"]})
    served.service.stop()

    status, error = served.evaluate({"ids": ["3"], "confirm_token": estimate["confirm_token"]})

    assert status == 503 and error["error"] == "el servidor se está parando"
    assert served.built == 0


def test_a_stop_during_the_start_refuses_the_job_before_any_backup(served: _Served, monkeypatch):
    """A forced job cancelled between taking the lock and asking must not have copied the
    side-car (a `.bak` for a pass that never ran) or built a client."""
    from contextlib import contextmanager

    from xbrain.jev import service as service_module

    real = service_module.pass_lock

    @contextmanager
    def _stopped_meanwhile(path, holder):
        with real(path, holder) as lock:
            served.service._starting.cancel.set()
            yield lock

    monkeypatch.setattr(service_module, "pass_lock", _stopped_meanwhile)
    estimate = served.estimate({"ids": ["1"], "force": True})

    status, error = served.evaluate(
        {"ids": ["1"], "force": True, "confirm_token": estimate["confirm_token"]}
    )

    assert status == 503 and error["error"] == "el servidor se está parando"
    assert list(served.cfg.jev_dir.glob("topics.*.bak")) == [] and served.built == 0


# --------------------------------------------------------------------------- the job slot


def test_a_refused_job_leaves_the_slot_idle_and_the_next_one_runs(served: _Served):
    estimate = served.estimate({"ids": ["3"]})
    with pass_lock(served.cfg.jev_lock_path, "xbrain jev topics"):
        status, _ = served.evaluate({"ids": ["3"], "confirm_token": estimate["confirm_token"]})
        _, job, _ = served.request("GET", "/api/job")

    assert status == 409 and job == {"state": "idle"}
    again = served.estimate({"ids": ["3"]})
    status, _ = served.evaluate({"ids": ["3"], "confirm_token": again["confirm_token"]})
    assert status == 202
    assert served.wait_job()["state"] == "done"


def test_a_thread_that_cannot_start_gives_the_confirmation_back(served: _Served, monkeypatch):
    """Called on the service, not over HTTP: patching `Thread.start` would stop the HTTP
    server's own request threads too."""
    from xbrain.jev.picks import ServeError

    service = served.service
    estimate = service.estimate("topics", {"ids": ["3"]})
    body = {"ids": ["3"], "confirm_token": estimate["confirm_token"]}

    def _no_threads(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", _no_threads)
    with pytest.raises(ServeError) as refused:
        service.evaluate("topics", body)
    monkeypatch.undo()

    assert refused.value.status == 503 and "no se pudo arrancar" in refused.value.message
    assert service.job_view() == {"state": "idle"}
    assert service.evaluate("topics", body)["state"] in ("running", "done")
    served.wait_job()


def test_two_evaluates_at_once_start_exactly_one_job(tmp_path: Path, monkeypatch):
    s = _Served(_repo(tmp_path, monkeypatch), _Recorder(delay=0.1))
    try:
        first, second = s.estimate({"ids": ["3"]}), s.estimate({"ids": ["4"]})
        barrier = threading.Barrier(2)
        statuses: list[tuple[int, str]] = []

        def _go(body: dict[str, Any]) -> None:
            barrier.wait()
            status, answer = s.evaluate(body)
            statuses.append((status, answer.get("error", "")))

        threads = [
            threading.Thread(
                target=_go, args=({"ids": ["3"], "confirm_token": first["confirm_token"]},)
            ),
            threading.Thread(
                target=_go, args=({"ids": ["4"], "confirm_token": second["confirm_token"]},)
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        s.wait_job()
    finally:
        s.close()

    # The slot refuses the second — not the pass lock, whose refusal would spend its
    # confirmation (a job being STARTED holds the slot before it holds the lock).
    assert sorted(status for status, _ in statuses) == [202, 409]
    assert [e for status, e in statuses if status == 409] == [
        "ya hay un trabajo en curso: espera a que termine"
    ]


def test_the_lock_is_released_after_a_job_that_failed(tmp_path: Path, monkeypatch):
    cfg = _repo(tmp_path, monkeypatch)
    s = _Served(cfg, _Recorder(fail_when=lambda state: True))
    try:
        _run_job(s, {"ids": ["3"]})
    finally:
        s.close()

    assert _locked(cfg.jev_lock_path) is False


# --------------------------------------------------------------------------- confirmations


def test_a_confirmation_expires(served: _Served, monkeypatch):
    from xbrain.jev import service as service_module

    estimate = served.estimate({"ids": ["3"]})
    later = service_module._monotonic() + service_module.CONFIRM_TTL_S + 1
    monkeypatch.setattr(service_module, "_monotonic", lambda: later)

    status, error = served.evaluate({"ids": ["3"], "confirm_token": estimate["confirm_token"]})

    assert status == 409 and "caducó" in error["error"] and served.built == 0


def test_a_price_that_rose_over_the_cap_since_the_estimate_is_refused(tmp_path: Path, monkeypatch):
    cfg = _repo(tmp_path, monkeypatch, jev="serve_max_usd = 0.000005\n")
    s = _Served(cfg, _Recorder())
    try:
        estimate = s.estimate({"ids": ["3"]})
        # A terminal pass meanwhile: one very expensive answer lifts the mean past the cap.
        with pass_lock(cfg.jev_lock_path, "seed") as lock:
            assessments = load_assessments(cfg.jev_topics_path)
            selection = select_items(
                ITEMS,
                assessments,
                VOCAB,
                ids=["5"],
                limit=None,
                force=False,
                fallback=cfg.jev_fallback_option,
                char_limit=cfg.jev_state_char_limit,
            )
            run_topics(
                cfg,
                selection,
                assessments,
                VOCAB,
                lambda: FakeJevClient(provider="typesafe", input_tokens=10_000),
                lock=lock,
            )
        status, error = s.evaluate({"ids": ["3"], "confirm_token": estimate["confirm_token"]})
    finally:
        s.close()

    assert status == 409 and "precio estimado cambió" in error["error"] and s.built == 0


@pytest.mark.parametrize(
    "other",
    [{"ids": ["3"], "force": True}, {"topic": "ai-coding"}],
    ids=["force", "topic"],
)
def test_a_confirmation_is_for_its_force_and_its_kind_of_pick(served: _Served, other):
    estimate = served.estimate({"ids": ["3"]})

    status, _ = served.evaluate({**other, "confirm_token": estimate["confirm_token"]})

    assert status == 409 and served.built == 0


def test_the_estimate_and_the_job_say_which_kind_of_pass(served: _Served):
    estimate = served.estimate({"ids": ["3"]})
    job = _run_job(served, {"ids": ["3"]})

    assert estimate["kind"] == job["kind"] == "topics"
    assert estimate["pick"] == {"ids": ["3"], "force": False}


def test_an_unknown_kind_of_pass_is_404(served: _Served):
    status, error, _ = served.request("POST", "/api/verify/estimate", {"ids": ["3"]})

    assert status == 404 and "verify" in error["error"]


# --------------------------------------------------------------------------- the terminal log


def test_a_failed_job_is_logged_as_an_error(tmp_path: Path, monkeypatch, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    cfg = _repo(tmp_path, monkeypatch)
    s = _Served(cfg, _Recorder(fail_when=lambda state: True))
    try:
        _run_job(s, {"ids": ["3"]})
    finally:
        s.close()

    assert any(
        r.levelno == logging.ERROR and "ninguna de las 1" in r.getMessage() for r in caplog.records
    )


def test_a_run_log_line_that_could_not_be_written_reaches_the_terminal_whole(
    served: _Served, monkeypatch, caplog
):
    import logging

    from xbrain.jev import run as run_module

    def _full_disk(run, path):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(run_module, "append_run", _full_disk)
    job = _run_job(served, {"ids": ["3"]})

    assert "No space left" in job["log_error"]
    [record] = [r for r in caplog.records if "añádela a mano" in r.getMessage()]
    assert record.levelno == logging.ERROR and '"requests":1' in record.getMessage()


def test_an_interrupted_job_is_logged_as_a_warning(tmp_path: Path, monkeypatch, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    cfg = _repo(tmp_path, monkeypatch, jev="serve_max_usd = 0.000015\n")
    s = _Served(cfg, _Recorder(input_tokens=250))
    try:
        _run_job(s, {"unevaluated": 3})
    finally:
        s.close()

    assert any(
        r.levelno == logging.WARNING and "interrumpido (tope)" in r.getMessage()
        for r in caplog.records
    )


# --------------------------------------------------------------------------- HTTP minors


@pytest.mark.parametrize("cut", ["drop-last", "first-char", "extra-char"])
def test_a_token_that_is_almost_right_is_refused(served: _Served, cut: str):
    token = served.service.token
    sent = {"drop-last": token[:-1], "first-char": token[:1], "extra-char": token + "x"}[cut]

    status, _, _ = served.request(
        "POST", "/api/topics/estimate", {"ids": ["3"]}, token=False, headers={TOKEN_HEADER: sent}
    )

    assert status == 403


def test_a_host_on_another_port_is_refused(served: _Served):
    status, _, _ = served.request("GET", "/api/data", headers={"Host": "127.0.0.1:1"})

    assert status == 403


def test_a_get_from_a_foreign_origin_is_refused(served: _Served):
    status, _, _ = served.request("GET", "/api/data", origin="http://evil.example")

    assert status == 403


def test_a_negative_content_length_is_refused_not_waited_on(served: _Served):
    status, _, _ = served.request(
        "POST", "/api/topics/estimate", raw=b"", headers={"Content-Length": "-1"}
    )

    assert status == 400


@pytest.mark.parametrize(
    "name,body",
    [("x.svg", b"<svg onload='alert(1)'/>"), ("x.html", b"<script>1</script>"), ("x.txt", b"hi")],
)
def test_media_serves_only_images_and_videos(served: _Served, name: str, body: bytes):
    (served.cfg.output_dir / "_media" / name).write_bytes(body)

    status, _, _ = served.request("GET", f"/_media/{name}")

    assert status == 404


def test_media_answers_with_a_sandboxing_policy(served: _Served):
    (served.cfg.output_dir / "_media" / "a.png").write_bytes(b"\x89PNG")

    status, _, response = served.request("GET", "/_media/a.png")

    assert status == 200 and response.getheader("Content-Security-Policy") == "sandbox"


def test_a_nul_byte_in_a_media_path_is_a_404(served: _Served):
    status, _, _ = served.request("GET", "/_media/a%00b.png")

    assert status == 404


def test_an_internal_error_says_so_without_its_detail(served: _Served, monkeypatch):
    def _boom():
        raise RuntimeError("secreto: /Users/alguien/.env")

    monkeypatch.setattr(served.service, "blob", _boom)

    status, error, _ = served.request("GET", "/api/data")

    assert status == 500 and "secreto" not in error["error"]


def test_the_job_view_counts_calls_that_failed(tmp_path: Path, monkeypatch):
    cfg = _repo(tmp_path, monkeypatch)
    s = _Served(cfg, _Recorder(fail_when=lambda state: "Cursor" in state["post"]))
    try:
        job = _run_job(s, {"ids": ["3", "4"]})
    finally:
        s.close()

    assert (job["failed_calls"], job["answered"], job["outcome"]["failed"]) == (1, 1, 1)


def test_refused_answers_under_a_cap_stop_are_failures_not_unsaved(tmp_path: Path, monkeypatch):
    """A soft stop DRAINS: every answer that came back was seen. One xbrain refuses is a
    failure — never `unsaved`, which means "answered, not kept, because of Ctrl-C"."""
    cap = 2 * MEAN_USD
    cfg = _big_repo(
        tmp_path, monkeypatch, posts=6, jev=f"concurrency = 1\nserve_max_usd = {cap!r}\n"
    )
    # Refused by the parser, and 1.5× the mean: the first answer lifts the reservation and the
    # cap cuts the second post off.
    s = _Served(cfg, _Priced(input_tokens=150, primary="banana"))
    try:
        estimate = s.estimate({"unevaluated": 2})
        s.evaluate({"unevaluated": 2, "confirm_token": estimate["confirm_token"]})
        job = s.wait_job()
    finally:
        s.close()

    run = load_runs(cfg.jev_runs_path)[-1]
    assert job["state"] == "interrupted" and job["reason"] == "tope"
    assert (run.ok, run.unsaved, run.failed) == (0, 0, run.requests)
    assert job["outcome"]["failed"] == run.requests and job["outcome"]["unsaved"] == 0


def test_a_call_that_raised_is_charged_its_reservation(tmp_path: Path, monkeypatch):
    """Fail closed: a call can raise AFTER the vendor answered and billed (an answer type the
    adapter does not model), so a raised call counts against the cap at its reservation."""
    cfg = _big_repo(tmp_path, monkeypatch, posts=4, jev="concurrency = 1\n")
    s = _Served(cfg, _Priced(fail_when=lambda state: True))
    try:
        job = _run_job(s, {"unevaluated": 3})
    finally:
        s.close()

    assert job["failed_calls"] == 3
    assert job["usd"] == pytest.approx(3 * MEAN_USD)


def test_a_stopped_server_mints_no_confirmation(served: _Served):
    served.service.stop()

    status, error, _ = served.request("POST", "/api/topics/estimate", {"ids": ["3"]})

    assert status == 503 and error["error"] == "el servidor se está parando"


# --------------------------------------------------------------------------- the page's stop


def test_the_page_can_stop_a_job_softly_and_what_was_paid_is_kept(tmp_path: Path, monkeypatch):
    """`POST /api/job/cancel`: nothing queued is sent, every call in flight is waited for,
    saved and logged, and the job ends as «cancelado» — not as a stopped server."""
    cfg = _big_repo(tmp_path, monkeypatch, posts=12, jev="concurrency = 3\n")
    client = _Priced(delay=0.1)
    s = _Served(cfg, client)
    try:
        estimate = s.estimate({"unevaluated": 12})
        s.evaluate({"unevaluated": 12, "confirm_token": estimate["confirm_token"]})
        s.wait_job(lambda job: job.get("answered", 0) >= 1)
        status, view, _ = s.request("POST", "/api/job/cancel", {})
        job = s.wait_job()
    finally:
        s.close()

    run = load_runs(cfg.jev_runs_path)[-1]
    stored = set(load_assessments(cfg.jev_topics_path)) - {"1", "2"}
    assert status == 200 and view["state"] == "running" and view["kind"] == "topics"
    assert job["state"] == "interrupted" and job["reason"] == "cancelado"
    assert client.sent < 12
    assert len(stored) == client.sent == job["answered"] == run.ok == run.requests
    assert run.unsaved == 0 and run.interrupted is True
    assert run.input_tokens == 100 * client.sent == job["tokens"]


def test_a_cancel_with_no_job_running_is_refused(served: _Served):
    status, error, _ = served.request("POST", "/api/job/cancel", {})
    _run_job(served, {"ids": ["3"]})
    after, error_after, _ = served.request("POST", "/api/job/cancel", {})

    assert status == 409 and error["error"] == "no hay ningún trabajo en curso"
    assert after == 409 and error_after == error
    # The finished job is not touched by the refused cancel.
    assert served.service.job_view()["state"] == "done"


@pytest.mark.parametrize(
    "kwargs",
    [{"token": False}, {"origin": None}, {"origin": "http://evil.example"}],
    ids=["no-token", "no-origin", "foreign-origin"],
)
def test_a_cancel_needs_the_token_and_this_servers_origin(
    tmp_path: Path, monkeypatch, kwargs: dict[str, Any]
):
    s = _Served(_repo(tmp_path, monkeypatch), _Recorder(delay=0.3))
    try:
        estimate = s.estimate({"ids": ["3", "4"]})
        s.evaluate({"ids": ["3", "4"], "confirm_token": estimate["confirm_token"]})
        status, _, _ = s.request("POST", "/api/job/cancel", {}, **kwargs)
        job = s.wait_job()
    finally:
        s.close()

    assert status == 403
    assert job["state"] == "done" and job["outcome"]["ok"] == 2


@pytest.mark.parametrize("body", [[1], "x", 3], ids=["list", "string", "number"])
def test_a_cancel_body_other_than_an_object_is_refused(served: _Served, body: Any):
    status, error, _ = served.request("POST", "/api/job/cancel", body)

    assert status == 400 and "objeto" in error["error"]


def test_a_cancel_needs_json(served: _Served):
    status, _, _ = served.request(
        "POST", "/api/job/cancel", raw=b"{}", headers={"Content-Type": "text/plain"}
    )

    assert status == 415


def test_the_blob_names_the_last_finished_job_its_data_already_includes(served: _Served):
    """An idle tab compares this with `/api/job`'s `finished_at` to know that a job another
    tab ran has ended since its data was built."""
    _, before, _ = served.request("GET", "/api/data")
    job = _run_job(served, {"ids": ["3"]})
    _, after, _ = served.request("GET", "/api/data")

    assert before["serve"]["finished_at"] is None
    assert job["finished_at"] is not None
    assert after["serve"]["finished_at"] == job["finished_at"]
    assert next(p for p in after["posts"] if p["id"] == "3")["status"] == "compared"
