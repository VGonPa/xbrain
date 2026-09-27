# tests/test_jev_ask.py
"""`xbrain jev ask`'s core: one Noul per post against the user's query, cached per contract.

The CLI on top of it is covered at the end of this file; the shared pool and pass are covered
by `test_jev_run.py` and `test_jev_assess.py`.
"""

from __future__ import annotations

import json
from typing import Any
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from tests.jev_fakes import FakeJevClient
from xbrain import cli
from xbrain.cli import app
from xbrain.config import Config, load_config
from xbrain.jev import ask as jev_ask
from xbrain.jev.ask import (
    AskFilters,
    AskQuery,
    assess_post,
    cost_model,
    filter_posts,
    finish_ask,
    load_history,
    plan_ask,
    same_selection,
    saved_results,
    select_ask_items,
)
from xbrain.jev.assess import assess_topics, build_topic_state, topic_contract
from xbrain.jev.dashboard import NO_ASKS, ask_page_data
from xbrain.jev.client import JevError, JevResult, NoulAnswer, NoulQuestion, Question
from xbrain.jev.defaults import (
    DEFAULT_ASK_TOKENS_PER_CALL,
    DEFAULT_CHARS_PER_TOKEN,
    tokens_cost_usd,
)
from xbrain.jev.load import load_jev_pairs
from xbrain.jev.lock import PassLockBusy, pass_lock
from xbrain.jev.models import AskAssessment, AskCalibration, JevRun
from xbrain.jev.questions import ASK_KEY, build_ask_questions, normalize_query
from xbrain.jev.report import ask_cost, ask_cost_by_query, run_history
from xbrain.jev.run import run_ask
from xbrain.jev.store import (
    ASK_INDEX,
    append_run,
    load_ask_index,
    load_asks,
    load_assessments,
    load_runs,
    save_asks,
    save_assessments,
)
from xbrain.models import Author, Enrichment, Item, Topic
from xbrain.rubrics import save_vocab
from xbrain.store import load_store, save_store

DT = datetime(2026, 9, 22, tzinfo=timezone.utc)
VOCAB = [
    Topic(slug="ai-coding", description="Construir software con IA."),
    Topic(slug="startups", description="Fundar empresas."),
]
QUERY = "¿Cómo configuro hooks en Claude Code?"
runner = CliRunner()


def _item(
    item_id: str,
    text: str,
    *,
    topics: tuple[str, ...] = ("ai-coding",),
    handle: str = "alice",
    created: datetime = DT,
) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/{handle}/status/{item_id}",
        author=Author(handle=handle, name=handle.title()),
        text=text,
        created_at=created,
        captured_at=created,
        enriched=Enrichment(
            enriched_at=DT,
            executor="claude-code",
            summary="s",
            primary_topic=topics[0] if topics else None,
            topics=list(topics),
        ),
    )


def _corpus() -> dict[str, Item]:
    return {
        "1": _item("1", "Claude Code hooks: PreToolUse and PostToolUse explained"),
        "2": _item("2", "Seed round tips", topics=("startups",), handle="bob"),
        "3": _item("3", "Hooks in Claude Code, a thread", created=DT - timedelta(days=30)),
        # No text and no author: nothing to ask about.
        "4": _item("4", " ", topics=("startups",), handle=""),
    }


class _ByText(FakeJevClient):
    """Answers the ask's Noul by what the post says: `hooks` → 0.95, else 0.1. Optional
    `delay` and a high-water mark of calls in flight, to prove the pool really overlaps."""

    def __init__(
        self, *, delay: float = 0.0, bill: tuple[int, float] | None = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.delay = delay
        #: `(per_call, chars_per_token)`: bill each call a constant plus its characters, the
        #: shape a real provider bills in (a fixed prompt around the state and question).
        self.bill = bill
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def ask(self, state: dict[str, str], questions: dict[str, Question]) -> JevResult:
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay)
            result = super().ask(state, questions)
        finally:
            with self._lock:
                self.in_flight -= 1
        noul = 0.95 if "hooks" in state["post"].lower() else 0.1
        tokens = result.input_tokens
        if self.bill is not None:
            chars = len(state["post"]) + jev_ask.question_chars(questions)
            tokens = round(self.bill[0] + chars / self.bill[1])
        return JevResult(
            provider=result.provider,
            model=result.model,
            answers={ASK_KEY: NoulAnswer(noul=noul)},
            input_tokens=tokens,
            output_tokens=result.output_tokens,
        )


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch) -> Config:
    return make_cfg(tmp_path, monkeypatch)


def make_cfg(tmp_path: Path, monkeypatch) -> Config:
    """The four-post repo (`_corpus`, `VOCAB`) every ask test starts from."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (tmp_path / "config.toml").write_text(
        f'[paths]\nvault = "{vault}"\noutput_subdir = "x"\ndata_dir = "data"\n'
        '[x]\nhandle = "v"\n[jev]\nconcurrency = 2\n',
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    save_store(_corpus(), tmp_path / "data" / "items.json")
    save_vocab(VOCAB, tmp_path / "data" / "vocab.yaml")
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    return load_config(tmp_path)


def _ask_path(cfg: Config, query: AskQuery) -> Path:
    return cfg.jev_asks_dir / f"{query.sha}.json"


def _run(
    cfg: Config,
    client: FakeJevClient,
    query_text: str = QUERY,
    filters: AskFilters | None = None,
    **hooks,
):
    """plan → run → finish, the flow the CLI and the server share, under the lock."""
    query = AskQuery.of(query_text)
    with pass_lock(cfg.jev_lock_path, "test") as lock:
        plan = plan_ask(cfg, query, filters or AskFilters(), None)
        outcome = run_ask(cfg, plan, lambda: client, lock=lock, **hooks)
        results = finish_ask(cfg, plan, outcome)
    return plan, outcome, results


# --------------------------------------------------------------------------- the question


def test_the_question_carries_the_query_verbatim_as_its_true_side():
    questions = build_ask_questions(QUERY)

    assert list(questions) == [ASK_KEY]
    question = questions[ASK_KEY]
    assert isinstance(question, NoulQuestion)
    assert question.instructions == (
        "Does the post in `post` answer or directly address the user's request?"
    )
    assert question.criteria == {
        "true": QUERY,
        "false": "The post does not address this request.",
    }


def test_a_query_is_normalised_so_spacing_does_not_make_a_new_one():
    assert normalize_query("  ¿Cómo  configuro\n hooks? ") == "¿Cómo configuro hooks?"
    # NFD and NFC of the same visible text are one query.
    assert normalize_query("Café") == normalize_query("Café")
    assert AskQuery.of(" hooks  en   Claude ").sha == AskQuery.of("hooks en Claude").sha
    # Case is kept: it reaches Jev verbatim, so it is part of the question.
    assert AskQuery.of("Hooks").sha != AskQuery.of("hooks").sha


def test_a_blank_query_is_refused_before_anything_is_asked():
    with pytest.raises(ValueError, match="vacía"):
        build_ask_questions("   \n ")


# --------------------------------------------------------------------------- one post


def test_one_post_gets_one_call_with_the_topics_state_and_one_probability():
    item = _corpus()["1"]
    query = AskQuery.of(QUERY)
    client = _ByText()

    record = assess_post(item, query, client, char_limit=100_000)

    [(state, questions)] = client.calls
    assert state == build_topic_state(item, 100_000)[0]  # the SAME evidence as topics
    assert questions == build_ask_questions(QUERY)
    assert record.item_id == "1" and record.probability == 0.95
    assert (record.provider, record.model, record.input_tokens) == ("fake", "jev-1.13.0", 100)
    assert record.prompt_chars == len(state["post"]) + jev_ask.question_chars(query.questions)


def test_an_answer_without_the_ask_noul_is_a_jev_error_never_a_record():
    class _Wrong(FakeJevClient):
        def ask(self, state, questions):
            return JevResult(provider="fake", model="m", answers={})

    with pytest.raises(JevError, match="no contestó"):
        assess_post(_corpus()["1"], AskQuery.of(QUERY), _Wrong(), char_limit=100_000)


def test_an_ask_record_is_frozen_and_bounds_its_probability():
    record = assess_post(_corpus()["1"], AskQuery.of(QUERY), _ByText(), char_limit=100_000)
    with pytest.raises(ValidationError):
        record.probability = 0.2  # type: ignore[misc]
    with pytest.raises(ValidationError, match="probability"):
        AskAssessment(**{**record.model_dump(), "probability": 1.5})


# --------------------------------------------------------------------------- the cache


def test_a_current_answer_is_not_asked_again_and_new_evidence_is():
    store = _corpus()
    query = AskQuery.of(QUERY)
    records = {
        "1": assess_post(store["1"], query, _ByText(), char_limit=100_000),
        "2": assess_post(store["2"], query, _ByText(), char_limit=100_000),
    }
    store["2"] = store["2"].model_copy(update={"text": "Seed round tips, now with numbers"})

    selection = select_ask_items(
        list(store.values()), records, query, char_limit=100_000, limit=None
    )

    # 1 is current; 2's evidence changed; 3 was never asked; 4 has no evidence.
    assert [item.id for item in selection.items] == ["2", "3"]
    assert (selection.skipped_current, selection.skipped_no_evidence) == (1, 1)


def test_an_answer_to_another_query_or_under_another_cut_is_not_current():
    store = _corpus()
    other = AskQuery.of("posts sobre creatina")
    records = {"1": assess_post(store["1"], other, _ByText(), char_limit=100_000)}

    selection = select_ask_items(
        [store["1"]], records, AskQuery.of(QUERY), char_limit=100_000, limit=None
    )
    assert [item.id for item in selection.items] == ["1"]
    selection = select_ask_items([store["1"]], records, other, char_limit=10, limit=None)
    assert [item.id for item in selection.items] == ["1"]


def test_a_topics_contract_never_passes_for_an_ask_contract():
    """Two contracts over the SAME state and digest: the version string keeps them apart."""
    query = AskQuery.of(QUERY)
    state = build_topic_state(_corpus()["1"], 100_000)[0]["post"]

    assert topic_contract(state, query.digest) != jev_ask.ask_contract(state, query.digest)


def test_limit_cuts_what_is_paid_for_and_counts_the_rest():
    query = AskQuery.of(QUERY)

    selection = select_ask_items(list(_corpus().values()), {}, query, char_limit=100_000, limit=1)

    assert [item.id for item in selection.items] == ["1"]
    assert selection.remaining == 2


# --------------------------------------------------------------------------- the store


def test_the_ask_file_round_trips_and_names_its_query(tmp_path: Path):
    query = AskQuery.of(QUERY)
    path = tmp_path / "asks" / f"{query.sha}.json"
    record = assess_post(_corpus()["1"], query, _ByText(), char_limit=100_000)

    save_asks(query, {"1": record}, path)

    assert load_asks(path, query) == {"1": record}
    assert json.loads(path.read_text(encoding="utf-8"))["query"] == QUERY
    assert load_asks(tmp_path / "missing.json", query) == {}


def test_a_corrupt_ask_file_is_refused_never_read_as_empty(tmp_path: Path):
    query = AskQuery.of(QUERY)
    path = tmp_path / f"{query.sha}.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(JevError, match=str(path)):
        load_asks(path, query)

    path.write_text(json.dumps({"query": QUERY, "assessments": {"1": {"x": 1}}}), "utf-8")
    with pytest.raises(JevError, match="ilegible"):
        load_asks(path, query)


def test_an_ask_file_for_another_query_is_refused(tmp_path: Path):
    query = AskQuery.of(QUERY)
    path = tmp_path / f"{query.sha}.json"
    save_asks(AskQuery.of("otra"), {}, path)

    with pytest.raises(JevError, match="otra"):
        load_asks(path, query)


def test_a_record_filed_under_another_post_id_is_refused(tmp_path: Path):
    query = AskQuery.of(QUERY)
    path = tmp_path / f"{query.sha}.json"
    record = assess_post(_corpus()["1"], query, _ByText(), char_limit=100_000)
    path.write_text(
        json.dumps({"query": QUERY, "assessments": {"2": record.model_dump(mode="json")}}),
        encoding="utf-8",
    )
    with pytest.raises(JevError, match="ilegible"):
        load_asks(path, query)


# --------------------------------------------------------------------------- the pass


def test_a_pass_asks_the_selection_at_concurrency_saves_and_logs_an_ask_line(cfg: Config):
    client = _ByText(delay=0.05)

    plan, outcome, _ = _run(cfg, client)

    assert client.max_in_flight == 2  # `[jev].concurrency = 2`, and the pool really overlaps
    assert sorted(state["post"] for state, _ in client.calls) == sorted(
        build_topic_state(item, cfg.jev_state_char_limit)[0]["post"]
        for item in plan.selection.items
    )
    assert [a.item_id for a in outcome.assessed] == ["1", "2", "3"]
    assert set(load_asks(_ask_path(cfg, plan.query), plan.query)) == {"1", "2", "3"}
    [line] = load_runs(cfg.jev_runs_path)
    assert (line.kind, line.query_sha, line.requests, line.ok) == ("ask", plan.query.sha, 3, 3)
    assert outcome.logged == line
    assert client.closed is True
    # An ask never writes the topics side-car.
    assert not cfg.jev_topics_path.exists()


def test_stored_counts_the_answers_already_there_plus_the_new_ones(cfg: Config):
    query = AskQuery.of(QUERY)
    prior = assess_post(_corpus()["1"], query, _ByText(), char_limit=cfg.jev_state_char_limit)
    save_asks(query, {"1": prior}, _ask_path(cfg, query))

    _, outcome, _ = _run(cfg, _ByText())

    assert [a.item_id for a in outcome.assessed] == ["2", "3"]
    assert outcome.stored == 3


def test_asking_the_same_query_again_costs_nothing(cfg: Config):
    _run(cfg, _ByText())
    again = _ByText()

    plan, outcome, results = _run(cfg, again)

    assert again.calls == [] and plan.selection.items == ()
    assert plan.selection.skipped_current == 3
    assert outcome.logged is None
    assert len(load_runs(cfg.jev_runs_path)) == 1
    assert [item.id for item, _ in results.ranked] == ["1", "3", "2"]


def test_changed_evidence_re_asks_only_that_post(cfg: Config):
    _run(cfg, _ByText())
    store = load_store(cfg.items_path)
    store["3"] = store["3"].model_copy(update={"text": "Hooks in Claude Code, updated"})
    save_store(store, cfg.items_path)
    again = _ByText()

    plan, _, _ = _run(cfg, again)

    assert [item.id for item in plan.selection.items] == ["3"]
    assert len(again.calls) == 1


def test_an_ask_pass_needs_the_pass_lock(cfg: Config):
    plan = plan_ask(cfg, AskQuery.of(QUERY), AskFilters(), None)
    with pass_lock(cfg.jev_lock_path, "test") as lock:
        pass
    client = _ByText()
    with pytest.raises(JevError, match="candado"):
        run_ask(cfg, plan, lambda: client, lock=lock)
    assert client.calls == []


def test_ask_lines_stay_out_of_the_topics_cost_views(cfg: Config):
    _run(cfg, _ByText())

    history = run_history(load_runs(cfg.jev_runs_path), load_assessments(cfg.jev_topics_path))

    assert history["runs"] == [] and history["total"]["requests"] == 0


def test_a_soft_cancel_stops_an_ask_pass_like_a_topics_one(cfg: Config):
    cancel = threading.Event()
    cancel.set()

    _, outcome, _ = _run(cfg, _ByText(), cancel=cancel)

    assert outcome.interrupted is True and outcome.assessed == ()


# --------------------------------------------------------------------------- plan / finish


def test_the_plan_builds_each_posts_evidence_once(cfg: Config, monkeypatch):
    built: list[str] = []
    real = jev_ask.build_topic_state

    def _counting(item, char_limit):
        built.append(item.id)
        return real(item, char_limit)

    monkeypatch.setattr(jev_ask, "build_topic_state", _counting)

    plan = plan_ask(cfg, AskQuery.of(QUERY), AskFilters(), None)

    assert sorted(built) == ["1", "2", "3", "4"]
    assert set(plan.states) == {"1", "2", "3"}  # the evidence-free post has no state to send


def test_finish_records_the_history_when_an_interrupted_pass_banked_something(cfg: Config):
    config = cfg.repo_root / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8").replace("concurrency = 2", "concurrency = 1"),
        encoding="utf-8",
    )
    cfg = load_config(cfg.repo_root)
    # One worker: the race this test holds shut is about the one worker running ahead.
    assert cfg.jev_concurrency == 1
    cancel = threading.Event()

    def _stop_after_first(done: int, total: int) -> None:
        cancel.set()

    class _SecondWaits(_ByText):
        """A second call that starts before the cancel waits for it, so the one worker can
        never run ahead and answer every post before the cancel lands (the race under load)."""

        def ask(self, state, questions):
            if len(self.calls) == 1:
                assert cancel.wait(10)
            return super().ask(state, questions)

    plan, outcome, results = _run(cfg, _SecondWaits(), cancel=cancel, on_progress=_stop_after_first)

    # Post 2 is skipped if the cancel came first, or drained if it was already in flight; post 3
    # is never sent either way (post 2's call holds the one worker until the cancel lands).
    assert outcome.interrupted and 1 <= len(outcome.assessed) < 3
    assert results.recorded is True
    [entry] = load_history(cfg).queries.values()
    assert entry.last_evaluated == len(outcome.assessed)


def test_finish_records_nothing_when_an_interrupted_pass_banked_nothing(cfg: Config):
    cancel = threading.Event()
    cancel.set()

    _, _, results = _run(cfg, _ByText(), cancel=cancel)

    assert results.recorded is False
    assert not (cfg.jev_asks_dir / ASK_INDEX).exists()


def test_a_use_stopped_before_any_answer_writes_no_answers_file(cfg: Config):
    """Nothing banked, nothing saved: an empty file would come back from `load_history` as a
    «rebuilt» query nobody asked for, first in the Preguntar tab."""
    cancel = threading.Event()

    class _FailsOnceCancelled(_ByText):
        def ask(self, state, questions):
            cancel.set()
            raise JevError("respuesta ilegible")

    plan, outcome, results = _run(cfg, _FailsOnceCancelled(), cancel=cancel)

    assert outcome.interrupted and outcome.assessed == () and results.recorded is False
    assert not _ask_path(cfg, plan.query).exists()
    assert load_history(cfg).queries == {}


def test_a_query_whose_history_entry_was_lost_is_rebuilt_from_its_file(cfg: Config):
    plan, _, _ = _run(cfg, _ByText())
    (cfg.jev_asks_dir / ASK_INDEX).unlink()

    history = load_history(cfg)

    entry = history.queries[plan.query.sha]
    assert entry.query == QUERY and entry.rebuilt is True
    # Rebuilt with no minimum: every answer in the file is a result.
    assert (entry.last_evaluated, entry.last_results, entry.last_min) == (3, 3, 0.0)
    # The calibration is rebuilt from every file, too.
    assert history.calibration.answers == 3


def test_the_selection_is_compared_by_the_posts_it_would_pay_for(cfg: Config):
    query = AskQuery.of(QUERY)
    before = plan_ask(cfg, query, AskFilters(), None)
    store = load_store(cfg.items_path)
    store["5"] = _item("5", "More hooks")
    save_store(store, cfg.items_path)

    assert same_selection(before, plan_ask(cfg, query, AskFilters(), None)) is False
    assert same_selection(before, before) is True


def test_a_near_duplicate_query_is_pointed_out(cfg: Config):
    _run(cfg, _ByText())

    plan = plan_ask(cfg, AskQuery.of("como configuro HOOKS en claude code"), AskFilters(), None)

    assert plan.similar == ()  # accents differ: not the same after casefold
    plan = plan_ask(cfg, AskQuery.of("¿cómo configuro hooks en Claude Code"), AskFilters(), None)
    assert plan.similar == (QUERY,)
    assert plan_ask(cfg, AskQuery.of(QUERY), AskFilters(), None).similar == ()


# --------------------------------------------------------------------------- filters


def _jev_with_topics(cfg: Config, nouls: dict[str, float]) -> None:
    """Assess every post but 3 for topics, so the Jev side of `--topic` has something."""
    store = load_store(cfg.items_path)
    client = FakeJevClient(nouls=nouls)
    from xbrain.jev.questions import build_topic_questions

    questions = build_topic_questions(VOCAB, cfg.jev_fallback_option)
    records = {
        item.id: assess_topics(item, questions, client, char_limit=cfg.jev_state_char_limit)
        for item in store.values()
        if item.id != "3"
    }
    save_assessments(records, cfg.jev_topics_path)


def _ids(items) -> list[str]:
    return sorted(item.id for item in items)


def test_topic_filter_takes_enrich_or_current_jev_membership(cfg: Config):
    _jev_with_topics(cfg, {"startups": 0.9})
    jev = load_jev_pairs(cfg)
    store = jev.store

    kept, dropped = filter_posts(store, AskFilters(topics=("startups",)), jev=jev, threshold=0.85)

    # 2 and 4 by enrich; 1 by Jev's 0.9; 3 was never assessed and enrich says ai-coding.
    assert _ids(kept) == ["1", "2", "4"] and dropped == 1
    kept, _ = filter_posts(store, AskFilters(topics=("startups",)), jev=jev, threshold=0.95)
    assert _ids(kept) == ["2", "4"]


def test_a_stale_topics_answer_does_not_put_a_post_in_a_topic(cfg: Config):
    _jev_with_topics(cfg, {"startups": 0.9})
    store = load_store(cfg.items_path)
    store["1"] = store["1"].model_copy(update={"text": "Claude Code hooks, rewritten"})
    save_store(store, cfg.items_path)
    jev = load_jev_pairs(cfg)

    kept, _ = filter_posts(jev.store, AskFilters(topics=("startups",)), jev=jev, threshold=0.85)
    assert _ids(kept) == ["2", "4"]
    kept, _ = filter_posts(jev.store, AskFilters(only_evaluated=True), jev=jev, threshold=0.85)
    assert _ids(kept) == ["2", "4"]


def test_the_plan_filters_topics_at_the_configured_threshold_not_the_results_one(cfg: Config):
    """`--threshold` is for results; the Jev side of `--topic` is `[jev].threshold` — so a low
    results bar never widens (and re-bills) the posts a query is asked about."""
    _jev_with_topics(cfg, {"startups": 0.5})
    plan = plan_ask(cfg, AskQuery.of(QUERY), AskFilters(topics=("startups",)), None)

    assert _ids(plan.candidates) == ["2", "4"]  # 1's 0.5 is under [jev].threshold = 0.85


def test_topic_filter_refuses_a_topic_nobody_uses(cfg: Config):
    jev = load_jev_pairs(cfg)
    with pytest.raises(JevError, match="nutricion"):
        filter_posts(jev.store, AskFilters(topics=("nutricion",)), jev=jev, threshold=0.85)


def test_only_evaluated_keeps_posts_with_a_current_topics_answer(cfg: Config):
    _jev_with_topics(cfg, {})
    jev = load_jev_pairs(cfg)

    kept, dropped = filter_posts(
        jev.store, AskFilters(only_evaluated=True), jev=jev, threshold=0.85
    )

    assert _ids(kept) == ["1", "2", "4"] and dropped == 1


def test_date_and_author_filters(cfg: Config):
    store = load_store(cfg.items_path)
    day = DT.date()

    # Both ends are whole days, included: `since` = `until` = the day of 1, 2 and 4.
    kept, _ = filter_posts(store, AskFilters(since=day, until=day), jev=None, threshold=0.85)
    assert _ids(kept) == ["1", "2", "4"]
    before = AskFilters(until=day - timedelta(days=1))
    assert _ids(filter_posts(store, before, jev=None, threshold=0.85)[0]) == ["3"]
    after = AskFilters(since=day + timedelta(days=1))
    assert filter_posts(store, after, jev=None, threshold=0.85)[0] == []
    kept, dropped = filter_posts(store, AskFilters(author="@BOB"), jev=None, threshold=0.85)
    assert _ids(kept) == ["2"] and dropped == 3


def test_no_filter_keeps_every_post(cfg: Config):
    store = load_store(cfg.items_path)
    kept, dropped = filter_posts(store, AskFilters(), jev=None, threshold=0.85)
    assert _ids(kept) == ["1", "2", "3", "4"] and dropped == 0


def test_filters_that_cannot_select_anything_are_refused():
    day = DT.date()
    with pytest.raises(ValueError, match="--since"):
        AskFilters(since=day, until=day - timedelta(days=1))
    for blank in ("", "  ", "@"):
        with pytest.raises(ValueError, match="--author"):
            AskFilters(author=blank)
    with pytest.raises(ValueError, match="--topic"):
        AskFilters(topics=(" ",))


def test_filters_round_trip_through_json_and_refuse_what_they_do_not_know():
    filters = AskFilters(
        topics=("startups",),
        since=DT.date(),
        until=DT.date(),
        author="bob",
        only_evaluated=True,
    )
    assert AskFilters.from_json(filters.as_json()) == filters
    assert AskFilters.from_json({}) == AskFilters()
    with pytest.raises(ValueError, match="colour"):
        AskFilters.from_json({"colour": "red"})
    with pytest.raises(ValueError, match="since"):
        AskFilters.from_json({"since": "ayer"})
    with pytest.raises(ValueError, match="only_evaluated"):
        AskFilters.from_json({"only_evaluated": "yes"})
    with pytest.raises(ValueError, match="author"):
        AskFilters.from_json({"author": 3})


def test_a_limit_below_one_is_refused_never_read_as_a_smaller_bill(cfg: Config):
    for limit in (0, -1):
        with pytest.raises(JevError, match="--limit"):
            plan_ask(cfg, AskQuery.of(QUERY), AskFilters(), limit)


# --------------------------------------------------------------------------- estimate


def test_the_estimate_counts_the_posts_the_pass_asks_and_the_chars_it_sends(cfg: Config):
    plan = plan_ask(cfg, AskQuery.of(QUERY), AskFilters(), None)
    prior = cost_model(AskCalibration())

    assert (prior.per_call, prior.chars_per_token, prior.measured) == (
        DEFAULT_ASK_TOKENS_PER_CALL,
        DEFAULT_CHARS_PER_TOKEN,
        False,
    )
    per_question = jev_ask.question_chars(plan.query.questions)
    chars = sum(
        len(build_topic_state(item, 100_000)[0]["post"]) + per_question
        for item in plan.selection.items
    )
    estimate = plan.estimate
    assert (estimate.posts, estimate.chars) == (3, chars)
    assert estimate.tokens == round(3 * DEFAULT_ASK_TOKENS_PER_CALL + chars / 4.0)
    assert estimate.usd == pytest.approx(tokens_cost_usd(estimate.tokens, "typesafe"))


def test_the_estimate_is_built_from_the_same_cut_as_the_call(cfg: Config):
    config = cfg.repo_root / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8") + "state_char_limit = 12\n", encoding="utf-8"
    )
    cfg = load_config(cfg.repo_root)

    plan, outcome, _ = _run(cfg, _ByText())

    assert plan.estimate.chars == sum(record.prompt_chars for record in outcome.assessed)
    assert all(record.truncated for record in outcome.assessed)


def test_the_cost_model_is_fitted_from_paid_answers_once_two_sizes_exist():
    calibration = AskCalibration()
    calibration = calibration.add(1_000, 500 + 1_000 // 4)
    assert cost_model(calibration).measured is False  # one size cannot separate the two terms
    calibration = calibration.add(3_000, 500 + 3_000 // 4)

    model = cost_model(calibration)

    assert model.measured is True and model.answers == 2
    assert model.per_call == pytest.approx(500)
    assert model.chars_per_token == pytest.approx(4.0)


def test_a_fit_that_makes_no_sense_falls_back_to_the_prior():
    decreasing = AskCalibration().add(1_000, 900).add(3_000, 100)
    assert cost_model(decreasing) == cost_model(AskCalibration())


def test_a_negative_intercept_is_refitted_through_zero():
    calibration = AskCalibration().add(1_000, 100).add(3_000, 900)

    model = cost_model(calibration)

    assert model.per_call == 0.0
    assert model.chars_per_token == pytest.approx((1_000**2 + 3_000**2) / (100_000 + 2_700_000))


def test_answers_without_usage_do_not_calibrate(cfg: Config):
    _run(cfg, _ByText(input_tokens=None))

    assert load_history(cfg).calibration.answers == 0


def test_after_one_paid_query_the_next_estimate_is_the_real_bill(cfg: Config):
    """A provider that bills a constant per call plus the characters: the first pass teaches
    the model both terms, and the next query's estimate is what it is then billed."""
    bill = (700, 3.5)
    _run(cfg, _ByText(bill=bill))

    plan = plan_ask(cfg, AskQuery.of("posts sobre seed rounds"), AskFilters(), None)
    assert plan.estimate.model.measured is True
    assert plan.estimate.model.per_call == pytest.approx(700, abs=2)
    _, outcome, _ = _run(cfg, _ByText(bill=bill), "posts sobre seed rounds")

    billed = sum(record.input_tokens for record in outcome.assessed)
    assert plan.estimate.tokens == pytest.approx(billed, abs=3)


# --------------------------------------------------------------------------- results


def _sent(item: Item) -> str:
    """A post's state as a call sends it at the default cut."""
    return build_topic_state(item, 100_000)[0]["post"]


def test_results_are_the_current_answers_at_or_over_the_minimum_best_first(cfg: Config):
    plan, _, _ = _run(cfg, _ByText())
    store = load_store(cfg.items_path)
    records = load_asks(_ask_path(cfg, plan.query), plan.query)
    # 3's answer is rewritten above the bar; 1's evidence moves, so its answer is stale.
    records["3"] = records["3"].model_copy(update={"probability": 0.97})
    store["1"] = store["1"].model_copy(update={"text": "Claude Code hooks, v2"})

    results = saved_results(
        store,
        None,
        plan.query,
        AskFilters(),
        records,
        topic_threshold=0.85,
        minimum=0.9,
        state_text=_sent,
    )

    assert [(item.id, record.probability) for item, record in results.ranked] == [("3", 0.97)]
    assert results.answered == 2  # 2 and 3 are current; 1 is stale; 4 never asked


def test_results_rank_ties_by_post_id(cfg: Config):
    plan, _, _ = _run(cfg, _ByText())
    store = load_store(cfg.items_path)
    records = load_asks(_ask_path(cfg, plan.query), plan.query)

    # Candidates in REVERSE id order: the tie is broken by id, not by the order asked.
    candidates = dict(reversed(store.items()))
    results = saved_results(
        candidates,
        None,
        plan.query,
        AskFilters(),
        records,
        topic_threshold=0.85,
        minimum=0.0,
        state_text=_sent,
    )

    assert [item.id for item, _ in results.ranked] == ["1", "3", "2"]


def test_a_probability_exactly_at_the_minimum_is_a_result(cfg: Config):
    plan, _, _ = _run(cfg, _ByText())

    results = finish_ask(cfg, plan, None, minimum=0.95)

    assert [item.id for item, _ in results.ranked] == ["1", "3"]


# --------------------------------------------------------------------------- history and cost


def test_each_query_keeps_its_last_use_in_the_history(cfg: Config):
    _run(cfg, _ByText(), filters=AskFilters(topics=("ai-coding",)))
    plan, _, _ = _run(cfg, _ByText())

    [entry] = load_ask_index(cfg.jev_asks_dir / ASK_INDEX).queries.values()

    assert entry.query == QUERY and entry.query_sha == plan.query.sha
    assert entry.first_asked_at <= entry.last_asked_at
    assert (entry.times, entry.last_evaluated, entry.last_results) == (2, 3, 3)
    assert (entry.last_min, entry.last_filters, entry.rebuilt) == (0.0, {}, False)


def test_a_corrupt_history_is_refused(cfg: Config):
    path = cfg.jev_asks_dir / ASK_INDEX
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(JevError, match="ilegible"):
        load_ask_index(path)


def test_a_history_entry_filed_under_another_sha_is_refused(cfg: Config):
    plan, _, _ = _run(cfg, _ByText())
    path = cfg.jev_asks_dir / ASK_INDEX
    data = json.loads(path.read_text(encoding="utf-8"))
    data["queries"] = {"b" * 64: data["queries"][plan.query.sha]}
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(JevError, match="ilegible"):
        load_ask_index(path)


def test_an_unreadable_file_of_another_query_is_named_as_such(cfg: Config):
    cfg.jev_asks_dir.mkdir(parents=True)
    other = cfg.jev_asks_dir / f"{'c' * 64}.json"
    other.write_text("{", encoding="utf-8")

    with pytest.raises(JevError, match="otra consulta; sácalo de"):
        plan_ask(cfg, AskQuery.of(QUERY), AskFilters(), None)


def test_what_each_query_has_cost_comes_from_the_run_log(cfg: Config):
    first, _, _ = _run(cfg, _ByText(input_tokens=1_000))
    second, _, _ = _run(cfg, _ByText(input_tokens=2_000), "otra consulta")
    runs = load_runs(cfg.jev_runs_path)

    cost = ask_cost(runs, first.query.sha)

    assert (cost["runs"], cost["requests"], cost["input_tokens"]) == (1, 3, 3_000)
    assert cost["cost_usd"] == pytest.approx(0.0)  # the fake provider has no price
    assert cost["unpriced_providers"] == ["fake"]
    by_query = ask_cost_by_query(runs)
    assert set(by_query) == {first.query.sha, second.query.sha}
    assert by_query[second.query.sha]["input_tokens"] == 6_000
    assert ask_cost(runs, "d" * 64)["runs"] == 0


# --------------------------------------------------------------------------- the CLI


def _refuse_client(cfg: Config):
    raise AssertionError("the CLI must not build a client here")


def _use(monkeypatch, client: FakeJevClient) -> None:
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: client)


def test_cli_dry_run_prints_the_estimate_and_never_builds_a_client(cfg: Config, monkeypatch):
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)

    result = runner.invoke(app, ["jev", "ask", QUERY, "--dry-run"])

    out = result.output
    assert result.exit_code == 0, out
    assert "3 posts por preguntar · 1 sin evidencia" in out
    estimate = plan_ask(cfg, AskQuery.of(QUERY), AskFilters(), None).estimate
    assert f"estimación: ~{estimate.tokens} tokens de entrada (~{estimate.usd:.4f} $)" in out
    assert "a priori: 1000 tokens por petición + 4.00 caracteres por token" in out
    assert "--dry-run: no se llama a Jev" in out
    assert not cfg.jev_asks_dir.exists() and not cfg.jev_runs_path.exists()


def test_cli_asks_ranks_the_results_and_keeps_the_history(cfg: Config, monkeypatch):
    client = _ByText()
    _use(monkeypatch, client)

    result = runner.invoke(app, ["jev", "ask", QUERY])

    out = result.output
    assert result.exit_code == 0, out
    assert len(client.calls) == 3 and client.closed
    lines = out.splitlines()
    header = lines.index(
        "Resultados: los 3 de 3 posts con respuesta vigente, de mayor a menor probabilidad"
    )
    assert lines[header + 1].split()[:4] == ["0.95", "██████████", "1", "@alice"]
    assert lines[header + 2].split()[:4] == ["0.95", "██████████", "3", "@alice"]
    assert lines[header + 3].split()[:4] == ["0.10", "█·········", "2", "@bob"]
    assert "https://x.com/alice/status/1" in lines[header + 1]
    [entry] = load_ask_index(cfg.jev_asks_dir / ASK_INDEX).queries.values()
    assert (entry.query, entry.last_evaluated, entry.last_results) == (QUERY, 3, 3)
    [line] = load_runs(cfg.jev_runs_path)
    assert line.kind == "ask"
    assert "3 respuestas · 0 fallidas" in out
    assert "Esta consulta ha costado: 1 pasada · 3 peticiones · 300 tokens de entrada" in out


def test_cli_a_repeated_query_costs_nothing_and_still_answers(cfg: Config, monkeypatch):
    _use(monkeypatch, _ByText())
    runner.invoke(app, ["jev", "ask", QUERY])
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)

    result = runner.invoke(app, ["jev", "ask", QUERY, "--min", "0.5"])

    out = result.output
    assert result.exit_code == 0, out
    assert "0 posts por preguntar · 3 ya respondidos" in out
    assert "Resultados: 2 de 3 posts con respuesta vigente llegan a la relevancia mínima 0.5" in out
    assert len(load_runs(cfg.jev_runs_path)) == 1
    [entry] = load_ask_index(cfg.jev_asks_dir / ASK_INDEX).queries.values()
    assert (entry.times, entry.last_min, entry.last_results) == (2, 0.5, 2)


def _cap(tmp_path: Path, monkeypatch, usd: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(path.read_text(encoding="utf-8") + f"ask_max_usd = {usd}\n", encoding="utf-8")


def test_cli_above_the_cap_asks_first_and_a_no_spends_nothing(cfg: Config, monkeypatch):
    _cap(cfg.repo_root, monkeypatch, "0.0000001")
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)

    result = runner.invoke(app, ["jev", "ask", QUERY], input="n\n")

    out = result.output
    assert result.exit_code == 1, out
    assert "pasa de [jev].ask_max_usd = 1e-07 $" in out
    assert "no se ha preguntado nada" in out
    assert not cfg.jev_runs_path.exists() and not cfg.jev_asks_dir.exists()


def test_cli_above_the_cap_a_yes_at_the_prompt_or_the_flag_spends(cfg: Config, monkeypatch):
    _cap(cfg.repo_root, monkeypatch, "0.0000001")
    first = _ByText()
    _use(monkeypatch, first)

    result = runner.invoke(app, ["jev", "ask", QUERY], input="y\n")
    assert result.exit_code == 0, result.output
    assert len(first.calls) == 3

    second = _ByText()
    _use(monkeypatch, second)
    result = runner.invoke(app, ["jev", "ask", "otra consulta", "--yes"])
    assert result.exit_code == 0, result.output
    assert len(second.calls) == 3
    assert "¿Preguntar" not in result.output


def test_cli_an_estimate_exactly_at_the_cap_does_not_ask(monkeypatch):
    """The cap is the most that runs without a question: equal is allowed."""

    def _no_prompt(*args, **kwargs):
        raise AssertionError("asked for confirmation at the cap")

    monkeypatch.setattr(cli.typer, "confirm", _no_prompt)
    cli._confirm_ask_cost(0.25, 0.25)
    with pytest.raises(AssertionError, match="confirmation"):
        cli._confirm_ask_cost(0.2500001, 0.25)


def test_cli_under_the_cap_does_not_ask(cfg: Config, monkeypatch):
    client = _ByText()
    _use(monkeypatch, client)

    result = runner.invoke(app, ["jev", "ask", QUERY])  # no input: a prompt would abort

    assert result.exit_code == 0 and len(client.calls) == 3


def test_cli_filters_narrow_what_is_paid_for(cfg: Config, monkeypatch):
    client = _ByText()
    _use(monkeypatch, client)

    result = runner.invoke(app, ["jev", "ask", QUERY, "--author", "bob", "--since", "2026-09-01"])

    out = result.output
    assert result.exit_code == 0, out
    assert [state["post"].splitlines()[0] for state, _ in client.calls] == ["Seed round tips"]
    assert "3 descartados por los filtros" in out
    [entry] = load_ask_index(cfg.jev_asks_dir / ASK_INDEX).queries.values()
    assert entry.last_filters == {"author": "bob", "since": "2026-09-01"}


def test_cli_limit_and_topic(cfg: Config, monkeypatch):
    client = _ByText()
    _use(monkeypatch, client)

    result = runner.invoke(app, ["jev", "ask", QUERY, "--topic", "ai-coding", "--limit", "1"])

    out = result.output
    assert result.exit_code == 0, out
    assert len(client.calls) == 1
    assert "1 fuera del límite" in out and "2 descartados por los filtros" in out


def test_cli_refuses_a_bad_minimum_and_an_unknown_topic(cfg: Config, monkeypatch):
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)

    result = runner.invoke(app, ["jev", "ask", QUERY, "--min", "1.5"])
    assert result.exit_code == 1 and "--min" in result.output
    result = runner.invoke(app, ["jev", "ask", QUERY, "--topic", "nutricion"])
    assert result.exit_code == 1 and "topic desconocido" in result.output
    result = runner.invoke(app, ["jev", "ask", "   "])
    assert result.exit_code == 1 and "vacía" in result.output


def test_cli_refuses_a_corrupt_ask_file_before_building_a_client(cfg: Config, monkeypatch):
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)
    path = _ask_path(cfg, AskQuery.of(QUERY))
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")

    result = runner.invoke(app, ["jev", "ask", QUERY])

    assert result.exit_code == 1 and "ilegible" in result.output
    assert path.read_text(encoding="utf-8") == "{"


def test_cli_waits_for_no_one_when_another_pass_holds_the_lock(cfg: Config, monkeypatch):
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)

    with pass_lock(cfg.jev_lock_path, "xbrain jev serve"):
        result = runner.invoke(app, ["jev", "ask", QUERY])

    assert result.exit_code == 75
    assert "xbrain jev serve" in result.output


def test_cli_ctrl_c_keeps_what_was_paid_and_exits_130(cfg: Config, monkeypatch):
    monkeypatch.setattr(cli, "_jev_client", lambda c: _ByText(interrupt_after=1))
    config = cfg.repo_root / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8").replace("concurrency = 2", "concurrency = 1"),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["jev", "ask", QUERY])

    assert result.exit_code == 130, result.output
    query = AskQuery.of(QUERY)
    assert list(load_asks(_ask_path(cfg, query), query)) == ["1"]
    [line] = load_runs(cfg.jev_runs_path)
    assert (line.kind, line.interrupted, line.ok) == ("ask", True, 1)
    # Something was banked, so the history knows the query (and what it holds).
    [entry] = load_history(cfg).queries.values()
    assert entry.last_evaluated == 1


def test_cli_ctrl_c_before_any_answer_leaves_no_history(cfg: Config, monkeypatch):
    monkeypatch.setattr(cli, "_jev_client", lambda c: _ByText(interrupt_after=0))
    config = cfg.repo_root / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8").replace("concurrency = 2", "concurrency = 1"),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["jev", "ask", QUERY])

    assert result.exit_code == 130, result.output
    assert not (cfg.jev_asks_dir / ASK_INDEX).exists()


def test_cli_dry_run_uses_the_cost_model_measured_on_paid_answers(cfg: Config, monkeypatch):
    _use(monkeypatch, _ByText(bill=(700, 3.5)))
    runner.invoke(app, ["jev", "ask", QUERY])
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)

    result = runner.invoke(app, ["jev", "ask", "posts sobre seed rounds", "--dry-run"])

    out = result.output
    model = load_history(cfg)
    fitted = cost_model(model.calibration)
    assert fitted.measured
    assert (
        f"medido en 3 respuestas: {fitted.per_call:.0f} tokens por petición + "
        f"{fitted.chars_per_token:.2f} caracteres por token" in out
    )
    estimate = plan_ask(cfg, AskQuery.of("posts sobre seed rounds"), AskFilters(), None).estimate
    assert f"~{estimate.tokens} tokens de entrada" in out


def test_cli_a_low_minimum_does_not_widen_the_topic(cfg: Config, monkeypatch):
    _jev_with_topics(cfg, {"startups": 0.5})
    client = _ByText()
    _use(monkeypatch, client)

    result = runner.invoke(app, ["jev", "ask", QUERY, "--topic", "startups", "--min", "0.3"])

    assert result.exit_code == 0, result.output
    # 1's Jev answer (0.5) is under [jev].threshold: only 2 carries startups with evidence.
    assert [state["post"].splitlines()[0] for state, _ in client.calls] == ["Seed round tips"]


def test_cli_confirms_without_the_lock_and_refuses_if_the_posts_moved(cfg: Config, monkeypatch):
    """The prompt never holds the lock (a server job may run meanwhile), so the selection is
    re-made under it — and a different one is refused, not paid."""
    _cap(cfg.repo_root, monkeypatch, "0.0000001")
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)
    seen: list[bool] = []

    def _confirm(*args, **kwargs) -> bool:
        try:
            with pass_lock(cfg.jev_lock_path, "probe"):
                seen.append(False)
        except PassLockBusy:
            seen.append(True)
        store = load_store(cfg.items_path)
        store["5"] = _item("5", "More hooks")
        save_store(store, cfg.items_path)
        return True

    monkeypatch.setattr(cli.typer, "confirm", _confirm)

    result = runner.invoke(app, ["jev", "ask", QUERY])

    assert seen == [False]  # the lock was free while the prompt waited
    assert result.exit_code == 1
    assert "cambió desde la estimación" in result.output
    assert not cfg.jev_runs_path.exists()


def test_cli_refuses_a_limit_below_one(cfg: Config, monkeypatch):
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)

    result = runner.invoke(app, ["jev", "ask", QUERY, "--limit", "0"])

    assert result.exit_code == 1 and "--limit" in result.output


def test_cli_refuses_since_after_until_and_a_blank_author(cfg: Config, monkeypatch):
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)

    result = runner.invoke(
        app, ["jev", "ask", QUERY, "--since", "2026-09-02", "--until", "2026-09-01"]
    )
    assert result.exit_code == 1 and "--since" in result.output
    result = runner.invoke(app, ["jev", "ask", QUERY, "--author", "@"])
    assert result.exit_code == 1 and "--author" in result.output


def test_cli_a_corrupt_history_stops_before_the_prompt_and_the_client(cfg: Config, monkeypatch):
    _cap(cfg.repo_root, monkeypatch, "0.0000001")
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)
    path = cfg.jev_asks_dir / ASK_INDEX
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")

    result = runner.invoke(app, ["jev", "ask", QUERY])  # no input: a prompt would abort

    assert result.exit_code == 1 and "historial de consultas ilegible" in result.output
    assert "¿Preguntar" not in result.output


def test_cli_an_unreadable_file_of_another_query_stops_before_the_client(cfg: Config, monkeypatch):
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)
    cfg.jev_asks_dir.mkdir(parents=True)
    (cfg.jev_asks_dir / f"{'c' * 64}.json").write_text("{", encoding="utf-8")

    result = runner.invoke(app, ["jev", "ask", QUERY])

    assert result.exit_code == 1 and "otra consulta; sácalo de" in result.output


def test_cli_points_out_a_near_duplicate_before_paying(cfg: Config, monkeypatch):
    _use(monkeypatch, _ByText())
    runner.invoke(app, ["jev", "ask", QUERY])

    result = runner.invoke(app, ["jev", "ask", "¿cómo configuro hooks en claude code", "--dry-run"])

    assert f"consulta parecida ya hecha: «{QUERY}»" in result.output


def test_cli_asks_lists_the_history_with_what_each_query_cost(cfg: Config, monkeypatch):
    _use(monkeypatch, _ByText(provider="typesafe", input_tokens=1_000))
    runner.invoke(app, ["jev", "ask", QUERY])
    runner.invoke(app, ["jev", "ask", QUERY, "--min", "0.5"])
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)

    result = runner.invoke(app, ["jev", "asks"])

    out = result.output
    assert result.exit_code == 0, out
    assert f"«{QUERY}»" in out
    assert "2 veces" in out and "último uso: 3 con respuesta, 2 ≥ 0.5 (relevancia mínima)" in out
    assert "1 pasada · 3 peticiones · 3000 tokens de entrada (~0.0001 $)" in out


def test_cli_asks_with_no_history_says_so(cfg: Config):
    result = runner.invoke(app, ["jev", "asks"])
    assert result.exit_code == 0 and "sin consultas" in result.output


def test_jev_report_adds_what_queries_cost_and_leaves_the_topics_numbers(cfg: Config, monkeypatch):
    _jev_with_topics(cfg, {"ai-coding": 0.9})

    def _line(output: str, prefix: str) -> str:
        return next(line for line in output.splitlines() if line.startswith(prefix))

    before = runner.invoke(app, ["jev", "report"]).output
    _use(monkeypatch, _ByText(provider="typesafe", input_tokens=1_000))
    runner.invoke(app, ["jev", "ask", QUERY])

    after = runner.invoke(app, ["jev", "report"]).output

    assert _line(after, "Histórico:") == _line(before, "Histórico:")  # asks never in it
    queries = "Consultas (jev ask): 1 pasada · 3 peticiones · 3000 tokens de entrada (~0.0001 $)"
    assert _line(after, "Consultas") == queries
    # A topics pass in the same log stays out of the queries' line.
    append_run(
        JevRun(
            started_at=DT,
            finished_at=DT,
            models=["jev-1.13.0"],
            requests=7,
            ok=7,
            failed=0,
            input_tokens_by_provider={"typesafe": 70},
            input_tokens=70,
            input_tokens_unknown=0,
            interrupted=False,
        ),
        cfg.jev_runs_path,
    )
    assert _line(runner.invoke(app, ["jev", "report"]).output, "Consultas") == queries


def test_cli_refuses_when_the_price_rose_over_the_cap_while_nobody_confirmed(
    cfg: Config, monkeypatch
):
    """Under the cap there is no prompt, so nobody agreed to a price: if the re-made plan
    under the lock costs more than the cap, it is refused rather than paid."""
    from dataclasses import replace

    monkeypatch.setattr(cli, "_jev_client", _refuse_client)
    plans: list[object] = []
    real = cli.plan_ask

    def _dearer_second_time(*args, **kwargs):
        plan = real(*args, **kwargs)
        plans.append(plan)
        if len(plans) == 2:
            return replace(plan, estimate=replace(plan.estimate, usd=cfg.jev_ask_max_usd * 2))
        return plan

    monkeypatch.setattr(cli, "plan_ask", _dearer_second_time)

    result = runner.invoke(app, ["jev", "ask", QUERY])

    assert result.exit_code == 1, result.output
    assert "la estimación subió" in result.output
    assert not cfg.jev_runs_path.exists()


def _touch(path: Path, seconds: float) -> None:
    """Move `path`'s mtime `seconds` from now — which of two files was written last."""
    import os

    moment = time.time() + seconds
    os.utime(path, (moment, moment))


def test_answers_saved_after_the_history_are_folded_into_the_calibration(cfg: Config):
    """A crash between saving a query's file and writing the history leaves paid answers the
    calibration never saw. The file is then newer than `index.json`, and only its answers newer
    than the entry's last use are folded in — the older ones were counted already."""
    plan, _, _ = _run(cfg, _ByText())
    path = _ask_path(cfg, plan.query)
    records = load_asks(path, plan.query)
    entry = load_history(cfg).queries[plan.query.sha]
    records["2"] = records["2"].model_copy(
        update={"asked_at": entry.last_asked_at + timedelta(minutes=5), "input_tokens": 250}
    )
    save_asks(plan.query, records, path)
    _touch(cfg.jev_asks_dir / ASK_INDEX, -60)

    history = load_history(cfg)

    assert history.calibration.answers == 4
    assert history.calibration.tokens == 3 * 100 + 250
    assert history.queries[plan.query.sha] == entry


def test_a_file_older_than_the_history_is_not_counted_twice(cfg: Config):
    """An entry removed by hand from a newer `index.json` is rebuilt, but its answers were
    folded when they were paid: the calibration does not count them again."""
    plan, _, _ = _run(cfg, _ByText())
    index_path = cfg.jev_asks_dir / ASK_INDEX
    data = json.loads(index_path.read_text(encoding="utf-8"))
    data["queries"] = {}
    index_path.write_text(json.dumps(data), encoding="utf-8")
    _touch(_ask_path(cfg, plan.query), -60)

    history = load_history(cfg)

    assert history.queries[plan.query.sha].rebuilt is True
    assert history.calibration.answers == 3


# --------------------------------------------------------------------------- the page's view
# `dashboard.asks_view`: what the static `jev.html` and the server's `/api/asks` show of every
# query asked — results recomputed now over each query's own filters and threshold, its cost
# from the run log, what Jev read once per result post, an unreadable file costing its row.


def _asks(cfg: Config) -> dict[str, Any]:
    return ask_page_data(cfg, load_jev_pairs(cfg), load_runs(cfg.jev_runs_path))


def test_the_page_filters_by_topic_at_the_jev_threshold_like_the_command(cfg: Config):
    """`--topic` judges Jev at `[jev].threshold`; the results bar (here 0.5) only ranks. Post
    1 is enrich's ai-coding with Jev's «startups» at 0.7: not a startups post at 0.85."""
    _jev_with_topics(cfg, {"startups": 0.7})
    _run(cfg, _ByText())  # every post answered, post 1 (0.95) included
    query = AskQuery.of(QUERY)
    with pass_lock(cfg.jev_lock_path, "test") as lock:
        plan = plan_ask(cfg, query, AskFilters(topics=("startups",)), None)
        outcome = run_ask(cfg, plan, lambda: _ByText(), lock=lock)
        found = finish_ask(cfg, plan, outcome, minimum=0.05)

    [row] = _asks(cfg)["history"]

    cli = [(item.id, record.probability) for item, record in found.ranked]
    assert "1" not in [item.id for item in plan.candidates]
    assert "1" in plan.records and plan.records["1"].probability == 0.95
    assert [(r["id"], r["p"]) for r in row["results"]] == cli
    assert row["answered"] == found.answered and row["min"] == 0.05


def test_the_page_lists_each_query_with_its_current_results_best_first(cfg: Config):
    plan, _, _ = _run(cfg, _ByText(provider="typesafe"))

    [row] = _asks(cfg)["history"]

    assert (row["sha"], row["query"], row["times"], row["answered"]) == (
        plan.query.sha,
        QUERY,
        1,
        3,
    )
    assert [(r["id"], r["p"]) for r in row["results"]] == [("1", 0.95), ("3", 0.95), ("2", 0.1)]
    assert row["min"] == 0.0 and row["filters"] == {}
    assert row["cost"]["cost_usd"] == tokens_cost_usd(300, "typesafe")
    assert row["cost"]["requests"] == 3
    assert "error" not in row


def test_a_query_no_logged_pass_paid_for_costs_zero_on_the_page(cfg: Config):
    _run(cfg, _ByText(provider="typesafe"))
    cfg.jev_runs_path.unlink()

    [row] = _asks(cfg)["history"]

    assert (row["cost"]["runs"], row["cost"]["cost_usd"]) == (0, 0.0)


def test_the_page_keeps_what_jev_read_once_per_result_post(cfg: Config):
    _run(cfg, _ByText())
    _run(cfg, _ByText(), query_text="hooks otra vez")

    view = _asks(cfg)

    # Every result post, once — post 2's 0.1 is a result too, ranked last.
    assert sorted(view["surfaces"]) == ["1", "2", "3"]
    assert view["surfaces"]["1"][0]["chars"] == len(load_store(cfg.items_path)["1"].text)


def test_a_changed_post_is_not_a_result_on_the_page(cfg: Config):
    _run(cfg, _ByText())
    store = load_store(cfg.items_path)
    store["3"].text = "Hooks in Claude Code, a thread (edited)"
    save_store(store, cfg.items_path)

    [row] = _asks(cfg)["history"]

    assert [r["id"] for r in row["results"]] == ["1", "2"] and row["answered"] == 2


def test_the_page_applies_the_filters_the_query_was_asked_with(cfg: Config):
    _run(cfg, _ByText(), filters=AskFilters(author="bob"))

    [row] = _asks(cfg)["history"]

    assert [(r["id"], r["p"]) for r in row["results"]] == [("2", 0.1)]
    assert (row["answered"], row["filters"]) == (1, {"author": "bob"})


def test_the_last_query_asked_is_listed_first(cfg: Config):
    _run(cfg, _ByText(), query_text="primera")
    _run(cfg, _ByText(), query_text="segunda")

    assert [row["query"] for row in _asks(cfg)["history"]] == ["segunda", "primera"]


def test_an_unreadable_query_file_costs_its_row_never_the_page(cfg: Config):
    plan, _, _ = _run(cfg, _ByText())
    _run(cfg, _ByText(), query_text="otra")
    _ask_path(cfg, plan.query).write_text("{", encoding="utf-8")

    view = _asks(cfg)
    rows = {row["query"]: row for row in view["history"]}

    assert "ilegible" in rows[QUERY]["error"] and rows[QUERY]["results"] == []
    assert "error" not in rows["otra"]


def test_a_history_entry_whose_topic_left_the_vocabulary_says_so(cfg: Config):
    _run(cfg, _ByText(), filters=AskFilters(topics=("ai-coding",)))
    store = load_store(cfg.items_path)
    for item in store.values():
        assert item.enriched is not None
        item.enriched.topics, item.enriched.primary_topic = ["startups"], "startups"
    save_store(store, cfg.items_path)
    save_vocab([VOCAB[1]], cfg.vocab_path)

    [row] = _asks(cfg)["history"]

    assert "topic desconocido" in row["error"]


def test_an_unreadable_history_is_the_tabs_error(cfg: Config):
    path = cfg.jev_asks_dir / "index.json"
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")

    view = _asks(cfg)

    assert view["history"] == [] and "ilegible" in view["error"]


def test_a_page_built_without_asks_has_an_empty_tab():
    assert NO_ASKS == {"history": [], "surfaces": {}, "topic_counts": {}, "error": None}


def test_a_run_log_that_cannot_be_read_gives_no_cost_rather_than_zero(cfg: Config):
    from xbrain.jev.dashboard import build_page_data

    _run(cfg, _ByText(provider="typesafe"))
    with cfg.jev_runs_path.open("a", encoding="utf-8") as log:
        log.write("{roto\n")

    [row] = build_page_data(cfg, now=DT)["asks"]["history"]

    assert set(row["cost"]) == {"error"} and "runs.jsonl" in row["cost"]["error"]


def test_the_page_cost_of_a_query_is_the_run_logs(cfg: Config):
    plan, _, _ = _run(cfg, _ByText(provider="typesafe", input_tokens=200_000))

    [row] = _asks(cfg)["history"]

    assert row["cost"] == ask_cost_by_query(load_runs(cfg.jev_runs_path))[plan.query.sha]
    assert row["cost"]["cost_usd"] > 0.01


def test_an_unreadable_file_costs_only_its_own_query_when_the_history_is_rebuilt(cfg: Config):
    """The index is gone and one answer file is broken: every other lost query still comes
    back (rebuilt) and the broken file is named — it never blanks the tab."""
    good, _, _ = _run(cfg, _ByText())
    bad, _, _ = _run(cfg, _ByText(), query_text="otra")
    (cfg.jev_asks_dir / ASK_INDEX).unlink()
    _ask_path(cfg, bad.query).write_text("{", encoding="utf-8")

    view = _asks(cfg)

    assert [(row["query"], row["rebuilt"]) for row in view["history"]] == [(QUERY, True)]
    assert f"{bad.query.sha}.json" in view["error"]
    assert [r["id"] for r in view["history"][0]["results"]] == ["1", "3", "2"]


def test_a_results_failure_that_is_not_a_filter_refusal_says_so_plainly(cfg: Config, monkeypatch):
    from xbrain.jev import dashboard

    _run(cfg, _ByText())

    def _broken(*args, **kwargs):
        raise JevError("algo distinto")

    monkeypatch.setattr(dashboard, "saved_results", _broken)

    [row] = _asks(cfg)["history"]

    assert row["error"] == "No se pudieron calcular los resultados de esta consulta: algo distinto"


# --------------------------------------------------------------------------- final review fixes


def test_load_history_skipping_unreadable_rebuilds_the_rest_and_folds_their_calibration(
    cfg: Config,
):
    """The page's tolerant read is `load_history`'s own rebuild: an unreadable file is named
    in `skipped`, every readable lost query is rebuilt AND its paid answers are folded into the
    calibration, exactly as the strict read would."""
    good, _, _ = _run(cfg, _ByText())
    bad, _, _ = _run(cfg, _ByText(), query_text="otra")
    (cfg.jev_asks_dir / ASK_INDEX).unlink()
    _ask_path(cfg, bad.query).write_text("{", encoding="utf-8")

    index, skipped = load_history(cfg, skip_unreadable=True)

    assert list(index.queries) == [good.query.sha] and index.queries[good.query.sha].rebuilt
    assert index.calibration.answers == 3
    assert len(skipped) == 1 and f"{bad.query.sha}.json" in skipped[0]
    with pytest.raises(JevError, match="ilegible"):
        load_history(cfg)


def test_an_unknown_topic_is_refused_by_its_own_type(cfg: Config):
    from xbrain.jev.ask import JevFilterRefused

    store = load_store(cfg.items_path)

    with pytest.raises(JevFilterRefused) as refused:
        filter_posts(store, AskFilters(topics=("nadie",)), jev=load_jev_pairs(cfg), threshold=0.85)

    assert isinstance(refused.value, JevError)


def test_a_failure_whose_text_looks_like_a_filter_refusal_is_not_one(cfg: Config, monkeypatch):
    """The page tells a refused filter by its TYPE, never by the error's wording."""
    from xbrain.jev import dashboard

    _run(cfg, _ByText())

    def _broken(*args, **kwargs):
        raise JevError("topic desconocido: parece un filtro, pero no lo es")

    monkeypatch.setattr(dashboard, "saved_results", _broken)

    [row] = _asks(cfg)["history"]

    assert row["error"].startswith("No se pudieron calcular los resultados de esta consulta:")


def test_the_cost_model_counts_tokens_once_for_every_caller():
    """`tokens = posts × per_call + chars / chars_per_token`, in one method; the estimate
    rounds that same number."""
    from xbrain.jev.ask import CostModel, estimate_ask

    model = CostModel(per_call=1000.0, chars_per_token=4.0, measured=True, answers=5)

    assert model.tokens(400) == 1100.0
    assert model.tokens(400, posts=3) == 3100.0
    assert estimate_ask([150, 250, 1], model).tokens == round(model.tokens(401, posts=3))


def test_cli_refuses_a_confirmed_ask_whose_price_rose_under_the_lock(cfg: Config, monkeypatch):
    """Over the cap the user agreed to ONE price. If the plan made again under the lock costs
    more than that, nothing is asked: the yes was for the estimate shown, not for any price."""
    from dataclasses import replace

    monkeypatch.setattr(cli, "_jev_client", _refuse_client)
    plans: list[object] = []
    real = cli.plan_ask

    def _dearer_under_the_lock(*args, **kwargs):
        plan = real(*args, **kwargs)
        plans.append(plan)
        usd = cfg.jev_ask_max_usd * (2 if len(plans) == 1 else 3)
        return replace(plan, estimate=replace(plan.estimate, usd=usd))

    monkeypatch.setattr(cli, "plan_ask", _dearer_under_the_lock)

    result = runner.invoke(app, ["jev", "ask", QUERY], input="y\n")

    assert result.exit_code == 1, result.output
    assert "¿Preguntar igualmente?" in result.output
    assert "la estimación subió" in result.output and "confirmad" in result.output
    assert not cfg.jev_runs_path.exists()
