# tests/test_jev_ask.py
"""`xbrain jev ask`'s core: one Noul per post against the user's query, cached per contract.

The CLI on top of it is covered at the end of this file; the shared pool and pass are covered
by `test_jev_run.py` and `test_jev_assess.py`.
"""

from __future__ import annotations

import json
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
    ask_results,
    assess_post,
    chars_per_token,
    estimate_ask,
    filter_posts,
    record_ask,
    select_ask_items,
)
from xbrain.jev.assess import assess_topics, build_topic_state, topic_contract
from xbrain.jev.client import JevError, JevResult, NoulAnswer, NoulQuestion, Question
from xbrain.jev.defaults import DEFAULT_CHARS_PER_TOKEN, tokens_cost_usd
from xbrain.jev.load import load_jev_pairs
from xbrain.jev.lock import pass_lock
from xbrain.jev.models import AskAssessment
from xbrain.jev.questions import ASK_KEY, build_ask_questions, normalize_query
from xbrain.jev.report import run_history
from xbrain.jev.run import run_ask
from xbrain.jev.store import (
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

    def __init__(self, *, delay: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.delay = delay
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
        return JevResult(
            provider=result.provider,
            model=result.model,
            answers={ASK_KEY: NoulAnswer(noul=noul)},
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch) -> Config:
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


def _run(cfg: Config, client: FakeJevClient, query_text: str = QUERY, **hooks):
    query = AskQuery.of(query_text)
    store = load_store(cfg.items_path)
    records = load_asks(_ask_path(cfg, query), query)
    selection = select_ask_items(
        list(store.values()), records, query, char_limit=cfg.jev_state_char_limit, limit=None
    )
    with pass_lock(cfg.jev_lock_path, "test") as lock:
        outcome = run_ask(cfg, selection, query, records, lambda: client, lock=lock, **hooks)
    return query, selection, outcome


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

    query, selection, outcome = _run(cfg, client)

    assert client.max_in_flight == 2  # `[jev].concurrency = 2`, and the pool really overlaps
    assert sorted(state["post"] for state, _ in client.calls) == sorted(
        build_topic_state(item, cfg.jev_state_char_limit)[0]["post"] for item in selection.items
    )
    assert [a.item_id for a in outcome.assessed] == ["1", "2", "3"]
    assert set(load_asks(_ask_path(cfg, query), query)) == {"1", "2", "3"}
    [line] = load_runs(cfg.jev_runs_path)
    assert (line.kind, line.query_sha, line.requests, line.ok) == ("ask", query.sha, 3, 3)
    assert outcome.logged == line
    assert client.closed is True
    # An ask never writes the topics side-car.
    assert not cfg.jev_topics_path.exists()


def test_asking_the_same_query_again_costs_nothing(cfg: Config):
    _run(cfg, _ByText())
    again = _ByText()

    _, selection, outcome = _run(cfg, again)

    assert again.calls == [] and selection.items == () and selection.skipped_current == 3
    assert outcome.logged is None
    assert len(load_runs(cfg.jev_runs_path)) == 1


def test_changed_evidence_re_asks_only_that_post(cfg: Config):
    _run(cfg, _ByText())
    store = load_store(cfg.items_path)
    store["3"] = store["3"].model_copy(update={"text": "Hooks in Claude Code, updated"})
    save_store(store, cfg.items_path)
    again = _ByText()

    _, selection, _ = _run(cfg, again)

    assert [item.id for item in selection.items] == ["3"]
    assert len(again.calls) == 1


def test_an_ask_pass_needs_the_pass_lock(cfg: Config):
    query = AskQuery.of(QUERY)
    selection = select_ask_items(
        list(_corpus().values()), {}, query, char_limit=100_000, limit=None
    )
    with pass_lock(cfg.jev_lock_path, "test") as lock:
        pass
    client = _ByText()
    with pytest.raises(JevError, match="candado"):
        run_ask(cfg, selection, query, {}, lambda: client, lock=lock)
    assert client.calls == []


def test_ask_lines_stay_out_of_the_topics_cost_views(cfg: Config):
    _run(cfg, _ByText())

    history = run_history(load_runs(cfg.jev_runs_path), load_assessments(cfg.jev_topics_path))

    assert history["runs"] == [] and history["total"]["requests"] == 0


def test_a_soft_cancel_stops_an_ask_pass_like_a_topics_one(cfg: Config):
    cancel = threading.Event()
    cancel.set()

    _, _, outcome = _run(cfg, _ByText(), cancel=cancel)

    assert outcome.interrupted is True and outcome.assessed == ()


# --------------------------------------------------------------------------- filters


def _jev_with_topics(cfg: Config, nouls: dict[str, float]) -> None:
    """Assess every post for topics so the Jev side of `--topic` has something to read."""
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


def _ids(items: list[Item]) -> list[str]:
    return sorted(item.id for item in items)


def test_topic_filter_takes_enrich_or_current_jev_membership(cfg: Config):
    _jev_with_topics(cfg, {"startups": 0.9})
    jev = load_jev_pairs(cfg)
    store = jev.store

    kept, dropped = filter_posts(store, AskFilters(topic="startups"), jev=jev, threshold=0.85)

    # 2 and 4 by enrich; 1 by Jev's 0.9; 3 was never assessed and enrich says ai-coding.
    assert _ids(kept) == ["1", "2", "4"] and dropped == 1
    kept, _ = filter_posts(store, AskFilters(topic="startups"), jev=jev, threshold=0.95)
    assert _ids(kept) == ["2", "4"]


def test_topic_filter_refuses_a_topic_nobody_uses(cfg: Config):
    jev = load_jev_pairs(cfg)
    with pytest.raises(JevError, match="nutricion"):
        filter_posts(jev.store, AskFilters(topic="nutricion"), jev=jev, threshold=0.85)


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


# --------------------------------------------------------------------------- estimate


def test_the_estimate_covers_exactly_the_posts_the_pass_asks(cfg: Config):
    query = AskQuery.of(QUERY)
    store = load_store(cfg.items_path)
    selection = select_ask_items(list(store.values()), {}, query, char_limit=100_000, limit=None)

    estimate = estimate_ask(selection, query, char_limit=100_000, chars_per_token=4.0)

    chars = sum(
        len(build_topic_state(item, 100_000)[0]["post"]) + jev_ask.question_chars(query.questions)
        for item in selection.items
    )
    assert estimate.posts == len(selection.items) == 3
    assert estimate.tokens == round(chars / 4.0)
    assert estimate.usd == pytest.approx(tokens_cost_usd(estimate.tokens, "typesafe"))
    # The pass then asks exactly those posts.
    client = _ByText()
    with pass_lock(cfg.jev_lock_path, "t") as lock:
        run_ask(cfg, selection, query, {}, lambda: client, lock=lock)
    assert len(client.calls) == estimate.posts


def test_chars_per_token_is_measured_from_paid_answers_and_falls_back_when_none():
    query = AskQuery.of(QUERY)
    item = _corpus()["1"]
    measured = assess_post(item, query, FakeJevClient(input_tokens=None), char_limit=100_000)
    assert chars_per_token([measured]) == DEFAULT_CHARS_PER_TOKEN  # no usage: not a measure
    priced = [
        assess_post(i, query, _ByText(input_tokens=50), char_limit=100_000)
        for i in _corpus().values()
        if i.text
    ]
    expected = sum(r.prompt_chars for r in priced) / (50 * len(priced))
    assert chars_per_token(priced) == pytest.approx(expected)
    assert chars_per_token([]) == DEFAULT_CHARS_PER_TOKEN


# --------------------------------------------------------------------------- results


def test_results_are_the_current_answers_over_the_threshold_best_first(cfg: Config):
    query, _, _ = _run(cfg, _ByText())
    store = load_store(cfg.items_path)
    records = load_asks(_ask_path(cfg, query), query)
    # 3's answer is rewritten above the bar; 1's evidence moves, so its answer is stale.
    records["3"] = records["3"].model_copy(update={"probability": 0.97})
    store["1"] = store["1"].model_copy(update={"text": "Claude Code hooks, v2"})

    results = ask_results(
        list(store.values()), records, query, char_limit=cfg.jev_state_char_limit, threshold=0.9
    )

    assert [(item.id, record.probability) for item, record in results.ranked] == [("3", 0.97)]
    assert results.answered == 2  # 2 and 3 are current; 1 is stale; 4 never asked


def test_results_rank_ties_by_post_id(cfg: Config):
    query, _, _ = _run(cfg, _ByText())
    store = load_store(cfg.items_path)
    records = load_asks(_ask_path(cfg, query), query)

    # Candidates in REVERSE id order: the tie is broken by id, not by the order asked.
    candidates = list(reversed(store.values()))
    results = ask_results(candidates, records, query, char_limit=100_000, threshold=0.5)

    assert [item.id for item, _ in results.ranked] == ["1", "3"]


def test_a_probability_exactly_at_the_threshold_is_a_result(cfg: Config):
    query, _, _ = _run(cfg, _ByText())
    store = load_store(cfg.items_path)
    records = load_asks(_ask_path(cfg, query), query)

    results = ask_results(list(store.values()), records, query, char_limit=100_000, threshold=0.95)

    assert [item.id for item, _ in results.ranked] == ["1", "3"]


# --------------------------------------------------------------------------- history


def test_each_query_made_is_kept_in_the_history(cfg: Config):
    query = AskQuery.of(QUERY)
    first = DT + timedelta(days=1)
    record_ask(
        cfg,
        query,
        filters=AskFilters(topic="ai-coding"),
        evaluated=3,
        results=2,
        threshold=0.85,
        now=first,
    )
    record_ask(
        cfg,
        query,
        filters=AskFilters(),
        evaluated=3,
        results=1,
        threshold=0.9,
        now=first + timedelta(hours=1),
    )

    [entry] = load_ask_index(cfg.jev_asks_dir / "index.json").values()

    assert entry.query == QUERY and entry.query_sha == query.sha
    assert (entry.first_asked_at, entry.last_asked_at) == (first, first + timedelta(hours=1))
    assert (entry.times, entry.evaluated, entry.results, entry.threshold) == (2, 3, 1, 0.9)
    assert entry.filters == {}


def test_a_corrupt_history_is_refused(cfg: Config):
    path = cfg.jev_asks_dir / "index.json"
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(JevError, match="ilegible"):
        load_ask_index(path)


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
    query = AskQuery.of(QUERY)
    selection = select_ask_items(
        list(load_store(cfg.items_path).values()), {}, query, char_limit=100_000, limit=None
    )
    estimate = estimate_ask(
        selection, query, char_limit=100_000, chars_per_token=DEFAULT_CHARS_PER_TOKEN
    )
    assert f"estimación: ~{estimate.tokens} tokens de entrada (~{estimate.usd:.4f} $)" in out
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
    header = lines.index("Resultados (≥ 0.85): 2 de 3 posts con respuesta vigente")
    assert lines[header + 1].split()[:3] == ["0.95", "1", "@alice"]
    assert lines[header + 2].split()[:3] == ["0.95", "3", "@alice"]
    assert "https://x.com/alice/status/1" in lines[header + 1]
    [entry] = load_ask_index(cfg.jev_asks_dir / "index.json").values()
    assert (entry.query, entry.evaluated, entry.results) == (QUERY, 3, 2)
    [line] = load_runs(cfg.jev_runs_path)
    assert line.kind == "ask"


def test_cli_a_repeated_query_costs_nothing_and_still_answers(cfg: Config, monkeypatch):
    _use(monkeypatch, _ByText())
    runner.invoke(app, ["jev", "ask", QUERY])
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)

    result = runner.invoke(app, ["jev", "ask", QUERY, "--threshold", "0.5"])

    out = result.output
    assert result.exit_code == 0, out
    assert "0 posts por preguntar · 3 ya respondidos" in out
    assert "Resultados (≥ 0.5): 2 de 3" in out
    assert len(load_runs(cfg.jev_runs_path)) == 1
    [entry] = load_ask_index(cfg.jev_asks_dir / "index.json").values()
    assert (entry.times, entry.threshold) == (2, 0.5)


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
    [entry] = load_ask_index(cfg.jev_asks_dir / "index.json").values()
    assert entry.filters == {"author": "bob", "since": "2026-09-01"}


def test_cli_limit_and_topic(cfg: Config, monkeypatch):
    client = _ByText()
    _use(monkeypatch, client)

    result = runner.invoke(app, ["jev", "ask", QUERY, "--topic", "ai-coding", "--limit", "1"])

    out = result.output
    assert result.exit_code == 0, out
    assert len(client.calls) == 1
    assert "1 fuera del límite" in out and "2 descartados por los filtros" in out


def test_cli_refuses_a_bad_threshold_and_an_unknown_topic(cfg: Config, monkeypatch):
    monkeypatch.setattr(cli, "_jev_client", _refuse_client)

    result = runner.invoke(app, ["jev", "ask", QUERY, "--threshold", "1.5"])
    assert result.exit_code == 1 and "--threshold" in result.output
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


# --------------------------------------------------------------------------- the page's view


def test_filters_round_trip_through_the_history_json():
    from datetime import date

    every = AskFilters(
        topic="ai-coding",
        since=date(2026, 9, 1),
        until=date(2026, 9, 22),
        author="@alice",
        only_evaluated=True,
    )

    assert AskFilters.from_json(every.as_json()) == every
    assert AskFilters.from_json({}) == AskFilters()


def test_the_token_ratio_says_how_many_paid_answers_it_rests_on(cfg: Config):
    from xbrain.jev.ask import token_ratio

    assert token_ratio([]) == jev_ask.TokenRatio(DEFAULT_CHARS_PER_TOKEN, 0)
    query, _, outcome = _run(cfg, _ByText(input_tokens=50))
    unknown = outcome.assessed[0].model_copy(update={"input_tokens": None})

    ratio = token_ratio([*outcome.assessed, unknown])

    assert ratio.measured == 3
    assert ratio.value == sum(a.prompt_chars for a in outcome.assessed) / 150


def test_ask_bill_sums_only_this_querys_ask_passes(cfg: Config):
    from xbrain.jev.report import ask_bill

    query, _, _ = _run(cfg, _ByText(provider="typesafe", input_tokens=100))
    _run(cfg, _ByText(provider="typesafe", input_tokens=7), query_text="otra pregunta")
    runs = load_runs(cfg.jev_runs_path)

    bill = ask_bill(runs, query.sha)

    assert (bill["runs"], bill["requests"], bill["input_tokens"]) == (1, 3, 300)
    assert bill["cost_usd"] == tokens_cost_usd(300, "typesafe")
    assert ask_bill(runs, "0" * 64)["runs"] == 0


def _history(cfg: Config, query: AskQuery, **filters) -> None:
    record_ask(cfg, query, filters=AskFilters(**filters), evaluated=0, results=0, threshold=0.85)


def _asks(cfg: Config) -> dict:
    from xbrain.jev.dashboard import asks_view, load_saved_asks

    saved, error = load_saved_asks(cfg)
    return asks_view(
        saved, load_jev_pairs(cfg), load_runs(cfg.jev_runs_path), char_limit=100_000, error=error
    )


def test_the_page_lists_each_query_with_its_current_results_best_first(cfg: Config):
    query, _, _ = _run(cfg, _ByText(provider="typesafe"))
    _history(cfg, query)

    [row] = _asks(cfg)["history"]

    assert (row["sha"], row["query"], row["times"], row["answered"]) == (query.sha, QUERY, 1, 3)
    assert [(r["id"], r["p"]) for r in row["results"]] == [("1", 0.95), ("3", 0.95)]
    assert row["cost"]["cost_usd"] == tokens_cost_usd(300, "typesafe")
    assert "error" not in row


def test_the_page_keeps_what_jev_read_once_per_result_post(cfg: Config):
    query, _, _ = _run(cfg, _ByText())
    _history(cfg, query)
    other, _, _ = _run(cfg, _ByText(), query_text="hooks otra vez")
    _history(cfg, other)

    view = _asks(cfg)

    assert sorted(view["surfaces"]) == ["1", "3"]
    assert view["surfaces"]["1"][0]["chars"] == len(load_store(cfg.items_path)["1"].text)


def test_a_changed_post_is_not_a_result_on_the_page(cfg: Config):
    query, _, _ = _run(cfg, _ByText())
    _history(cfg, query)
    store = load_store(cfg.items_path)
    store["3"].text = "Hooks in Claude Code, a thread (edited)"
    save_store(store, cfg.items_path)

    [row] = _asks(cfg)["history"]

    assert [r["id"] for r in row["results"]] == ["1"] and row["answered"] == 2


def test_the_page_applies_the_filters_the_query_was_asked_with(cfg: Config):
    query, _, _ = _run(cfg, _ByText())
    _history(cfg, query, author="bob")

    [row] = _asks(cfg)["history"]

    assert (row["results"], row["answered"], row["filters"]) == ([], 1, {"author": "bob"})


def test_the_last_query_asked_is_listed_first(cfg: Config):
    first, second = AskQuery.of("primera"), AskQuery.of("segunda")
    record_ask(cfg, second, filters=AskFilters(), evaluated=0, results=0, threshold=0.85, now=DT)
    record_ask(
        cfg,
        first,
        filters=AskFilters(),
        evaluated=0,
        results=0,
        threshold=0.85,
        now=DT + timedelta(hours=1),
    )

    assert [row["query"] for row in _asks(cfg)["history"]] == ["primera", "segunda"]


def test_an_unreadable_query_file_costs_its_row_never_the_page(cfg: Config):
    query, _, _ = _run(cfg, _ByText())
    _history(cfg, query)
    fine = AskQuery.of("otra")
    _history(cfg, fine)
    _ask_path(cfg, query).write_text("{", encoding="utf-8")

    rows = {row["query"]: row for row in _asks(cfg)["history"]}

    assert "ilegible" in rows[QUERY]["error"] and rows[QUERY]["results"] == []
    assert "error" not in rows["otra"]


def test_a_history_entry_whose_topic_left_the_vocabulary_says_so(cfg: Config):
    query, _, _ = _run(cfg, _ByText())
    _history(cfg, query, topic="ai-coding")
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
    from xbrain.jev.dashboard import NO_ASKS

    assert NO_ASKS == {"history": [], "surfaces": {}, "error": None}
