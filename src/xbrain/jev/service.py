"""What `xbrain jev serve` does, without HTTP: the page's data, estimates, and the ONE job.

THE MONEY PATH is estimate → confirm → one job, per kind: `topics` (`jev topics`) and `ask`
(`jev ask`, the «Preguntar» tab). ONE JOB SLOT for the whole server, whatever the kind, and
every paid pass takes the pass lock (`jev.lock`), so the page and a terminal never run two
passes over one side-car at once. A kind (`_TopicsKind`, `_AskKind`) says only how its body
is parsed, what it selects and costs, and which pass it runs; everything below is shared.

* `estimate(kind, body)`: the pick (`jev.picks`) resolved to posts and priced.
  - topics: resolved from the blob the page is showing, `assess.select_items` over them (the
    `--dry-run` answer), priced by `report.topics_pass_estimate` over the blob's
    `cost.per_post` — the one mean the cost strip and the Configuración tab show;
  - ask: `ask.plan_ask`, the call `jev ask` makes: its pre-filters, funnel and estimate
    (posts × tokens per call + characters ÷ characters per token, fitted on paid answers or
    the prior until they exist). No price, no confirmation. A query every candidate already
    answers selects nothing and costs nothing: it may still run, to be counted in the
    history — the job asks nobody and builds no client.
  Under `[jev].serve_max_usd` (at most equal) it mints a
  single-use confirmation, bound to the kind and the pick AS ASKED, that expires after
  `CONFIRM_TTL_S`. With no priced mean there is nothing to check the cap against, so no
  confirmation: fail-closed.
* `evaluate(kind, body)`: refused with 503 while the server stops, 409 while a job runs (the
  confirmation is kept), 409 for a confirmation that is unknown, spent, expired or for another
  pick. Otherwise the job thread takes the pass lock, re-reads, re-selects and RE-PRICES, and
  refuses (409, before any client exists and before any backup) if the posts moved or the
  price went over the cap. Only a job that passed all that is published in the slot.
* The job runs `run.run_topics` or `run.run_ask` — the terminal's pass — through `_Metered`,
  which makes the cap a HARD bound by reservation: before each call it reserves that post's
  expected cost and does not send when spent + reserved + that reservation would pass the cap.
  For topics that is the estimate's mean (or this job's own priced mean when higher); for an
  ask it is the post's OWN planned price (its characters by the plan's cost model, scaled to
  the confirmed estimate, and raised by the job's real/planned ratio once answers are dearer
  than planned), so a long post first does not make every later post look as dear. An answer replaces its reservation with
  its real cost; an answer with no token count or from a provider with no price is charged
  the reservation, never $0. So the bill passes the cap only by what the posts in flight
  cost above their reservation.
* Every stop from here is SOFT (`run_topics(cancel=…)`) — the cap, the server stopping, and
  the page's «Parar» (`cancel_job`, reason `cancelado`): nothing queued is sent, every call
  in flight is waited for, banked, saved and logged. Counters freeze when the job ends. An
  ask then runs `ask.finish_ask` under the lock, whose one rule decides the history: a use
  that kept answers is recorded (a soft stop included); one stopped before keeping any, or
  whose every call failed, is not. A history that cannot be written after the answers were
  paid leaves the job `done` with its outcome and `history_error` — never «failed».
* An ask job STREAMS its answers (`job_view(since=…)`, `/api/job?since=<cursor>`): the
  answers the query already had (ranked) and then each new one the moment the pass banks it
  (`run.run_pass`' `on_answer`), each as `dashboard.streamed_answer` — the blob's own
  `ask.answer_view`. The stream is the job's memory, never a file read per poll. Its order is
  ARRIVAL order; ranking is the reader's (the page's `askOrder` is `ask._rank`'s key), and at
  the end the page takes `finish_ask`'s results from the reloaded blob. A cursor
  (`<job number>-<index>`, or `0` for "from the start") names a place in ONE job's stream, so
  across polls no answer is missed or handed over twice; one reply carries at most
  `STREAM_PAGE` answers and says whether more are waiting. A topics job has no stream.
"""

from __future__ import annotations

import logging
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

from xbrain.config import Config
from xbrain.jev.ask import (
    AskPlan,
    AskQuery,
    ask_path,
    current_answers,
    filter_posts,
    finish_ask,
    plan_ask,
    question_chars,
    same_selection,
    topic_counts,
    use_minimum,
)
from xbrain.jev.assess import Selection, select_items
from xbrain.jev.client import CallSkipped, JevClient, JevError, JevResult, Question
from xbrain.jev.dashboard import (
    MediaFiles,
    build_page_data,
    collect_jev_media,
    render_jev_dashboard_html,
    streamed_answer,
)
from xbrain.jev.defaults import DEFAULT_PROVIDER, tokens_cost_usd, unpriced
from xbrain.jev.load import JevPairs, load_jev_pairs
from xbrain.jev.lock import PassLock, pass_lock
from xbrain.jev.models import AskAssessment, AskIndex
from xbrain.jev.questions import STATE_KEY
from xbrain.jev.errors import ServeError, refuse
from xbrain.jev.picks import (
    AskPick,
    TopicsPick,
    parse_ask,
    parse_filters,
    parse_pick,
    pick_ids,
)
from xbrain.jev.report import report_paths, topics_pass_estimate
from xbrain.jev.run import FAILURES_SHOWN, RunOutcome, run_ask, run_topics
from xbrain.jev.store import ASK_INDEX

logger = logging.getLogger(__name__)

#: The kinds of pass this server runs. Routes are `/api/<kind>/estimate|evaluate`.
KIND_TOPICS = "topics"
KIND_ASK = "ask"
#: What the server calls itself in the pass lock, for the message another pass gets.
LOCK_HOLDER = "xbrain jev serve"
#: Confirmations kept at once; the oldest is dropped past this (each is single-use anyway).
_CONFIRMS_KEPT = 32
#: A confirmation older than this is refused: the price it was minted at may have moved.
CONFIRM_TTL_S = 600
#: Float room at the cap: `n × mean` summed call by call must not refuse the n-th post the
#: estimate allowed at exactly the cap.
_CAP_ROOM = 1e-9
#: Answers one `/api/job?since=` reply hands over at most; the rest wait for the next poll,
#: which the page makes at once when a reply says `more`. What Jev read ships cut per surface
#: (`dashboard.PAGE_SURFACE_CHARS`), so this bounds the reply's size too.
STREAM_PAGE = 100
#: Seconds an ended job keeps its stream in memory. Every page takes the blob at a job's end
#: (`finish_ask`'s ranking) and never needs the tail, but a page may still be behind when the
#: job ends (a hidden tab looks about once a minute): past this, the entries are let go and a
#: late cursor gets no answers and a `next` at the end. Without it a whole-corpus stream
#: (~3,000 entries, 5-10 MB) would live until the next job.
STREAM_KEEP_S = 120.0
#: A stream cursor: `0` (the start of whatever job is current) or `<job number>-<index>`.
_CURSOR = re.compile(r"^(?:0|([1-9][0-9]*)-(0|[1-9][0-9]*))$")


def _monotonic() -> float:
    """The clock confirmations expire by — one seam, so a test can move it."""
    return time.monotonic()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def over_cap(usd: float, cap: float) -> bool:
    """`usd` passes `cap` (equal is allowed), with room for float sums."""
    return usd > cap * (1 + _CAP_ROOM)


@dataclass(frozen=True)
class _Confirm:
    """What one estimate priced, for the one `evaluate` that may spend it."""

    kind: str
    pick: TopicsPick | AskPick
    ids: tuple[str, ...]
    usd: float
    minted_at: float
    priced: _Priced[Any] | None = None


@dataclass
class _Job:
    """One background pass and everything the page polls about it. Guarded by `lock`."""

    kind: str
    pick: TopicsPick | AskPick
    ids: tuple[str, ...]
    max_usd: float
    #: The reservation per post: the confirmed estimate's mean (re-priced under the lock).
    per_post_usd: float
    #: What a kind adds to the view (the ask's `query_sha`).
    extra: dict[str, Any] = field(default_factory=dict)
    #: What the confirmed estimate selected, for the re-check under the lock (`_PassKind.same`).
    confirmed: Any = None
    #: A kind that prices each post on its own (the ask: its planned characters) reserves that
    #: price for the post instead of the job's mean. `None` for topics.
    post_price: Callable[[dict[str, str]], float] | None = None
    #: What the priced answers were planned to cost, to scale the next reservations when the
    #: answers turn out dearer than planned.
    planned_usd: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancel: threading.Event = field(default_factory=threading.Event)
    ready: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    state: str = "starting"
    terminal: bool = False
    refusal: str | None = None
    #: 409 for a job the data or the price no longer allow; 503 when the server is stopping.
    refusal_status: int = 409
    total: int = 0
    done: int = 0
    answered: int = 0
    failed_calls: int = 0
    tokens: int = 0
    tokens_unknown: int = 0
    #: What the job has spent: real costs, plus the reservation for every answer that could
    #: not be priced (no token count, or no price for its provider).
    usd: float = 0.0
    in_flight_usd: float = 0.0
    priced_usd: float = 0.0
    priced_answers: int = 0
    charged_at_estimate: int = 0
    unpriced_providers: set[str] = field(default_factory=set)
    cap_hit: bool = False
    stopped: bool = False
    reason: str | None = None
    error: str | None = None
    backup: str | None = None
    log_error: str | None = None
    outcome: dict[str, Any] | None = None
    started_at: str | None = None
    finished_at: str | None = None
    #: Which job of this server it is (1, 2, …): what tells one job's end from another's.
    #: `finished_at` is a time to the second, and two jobs can end in the same one (a free ask
    #: right after another): an idle tab that compared times missed the second end.
    number: int = 0
    #: What the page's live results receive (`dashboard.streamed_answer` each), in arrival
    #: order: the answers already current first, then each one the pass banks. `None` for a
    #: kind that does not stream (topics). Append-only: a cursor is an index into it. Held
    #: until `STREAM_KEEP_S` after the job ended (`_let_go`), then emptied; `stream_size`
    #: keeps its length, so cursors keep their meaning.
    stream: list[dict[str, Any]] | None = None
    stream_size: int = 0
    #: When the job ended, on `_monotonic`'s clock (for `STREAM_KEEP_S`).
    ended_mono: float | None = None
    #: How many answers the stream will hold if every post answers: current + to ask.
    stream_expected: int = 0
    #: The minimum the results open at (`ask.use_minimum`): the live list's refine default.
    stream_min: float = 0.0

    def planned(self, state: dict[str, str]) -> float:
        """What this post was planned to cost: its own price when the kind knows it, else the
        estimate's mean."""
        return self.post_price(state) if self.post_price is not None else self.per_post_usd

    def reservation(self, planned: float) -> float:
        """What the next post is expected to cost: its planned price, raised once this job's
        answers turn out dearer than planned (the corpus may be pricier than its past) — by
        their mean for a flat estimate, by their ratio to plan for a per-post one."""
        if not self.priced_answers:
            return planned
        if self.post_price is None:
            return max(planned, self.priced_usd / self.priced_answers)
        return planned * max(1.0, self.priced_usd / self.planned_usd if self.planned_usd else 1.0)

    def view(self, since: tuple[int, int] | None = None) -> dict[str, Any]:
        """What `/api/job` sends; with `since` (a parsed cursor, `_read_cursor`), the stream's
        answers from there too — read under the same lock as the counters beside them."""
        with self.lock:
            self._let_go()
            view = self.view_unlocked()
            if since is not None:
                view["stream"] = self._stream_from(since)
            return view

    def _let_go(self) -> None:
        """An ended job's stream, `STREAM_KEEP_S` after its end: emptied (its size kept)."""
        if (
            self.stream
            and self.ended_mono is not None
            and _monotonic() - self.ended_mono >= STREAM_KEEP_S
        ):
            self.stream = []

    def _stream_from(self, since: tuple[int, int]) -> dict[str, Any] | None:
        """At most `STREAM_PAGE` answers from the cursor `since`, and the cursor after them.
        A cursor into another job (or `0`) starts at this job's first answer; one past this
        job's end is not a cursor this server gave. Once the ended job's entries were let go
        (`_let_go`), a cursor before its end gets none and the cursor at the end."""
        if self.stream is None:
            return None
        size = self.stream_size
        number, index = since
        if number != self.number:
            index = 0
        elif index > size:
            raise refuse(f"cursor fuera de rango: el trabajo {self.number} tiene {size} respuestas")
        answers = self.stream[index : index + STREAM_PAGE]
        after = index + len(answers) if len(self.stream) == size else size
        return {
            "from": f"{self.number}-{index}",
            "next": f"{self.number}-{after}",
            "answers": answers,
            "more": after < size,
            "expected": self.stream_expected,
            "min": self.stream_min,
        }

    def view_unlocked(self) -> dict[str, Any]:
        """`view` for a caller that already holds `lock`."""
        view: dict[str, Any] = {
            "kind": self.kind,
            "state": self.state,
            "pick": self.pick.as_json(),
            "ids": list(self.ids),
            "total": self.total,
            "done": self.done,
            "answered": self.answered,
            "failed_calls": self.failed_calls,
            "tokens": self.tokens,
            "tokens_unknown": self.tokens_unknown,
            "usd": self.usd,
            "charged_at_estimate": self.charged_at_estimate,
            "unpriced_providers": sorted(self.unpriced_providers),
            "max_usd": self.max_usd,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "number": self.number,
            **self.extra,
        }
        for key in ("reason", "error", "backup", "log_error", "outcome"):
            value = getattr(self, key)
            if value is not None:
                view[key] = value
        return view


class _Metered:
    """The job's client: the cap as a hard bound by reservation, and the live counters.

    Before a call: refuse (`CallSkipped`, and cancel the rest) when spent + reserved in flight
    + this call's reservation would pass the cap. After it: the reservation becomes the real
    cost (`defaults.tokens_cost_usd`, THE price formula), or stays as the charge when the
    answer cannot be priced. Once the job is terminal nothing here moves a counter.
    """

    def __init__(self, inner: JevClient, job: _Job) -> None:
        self._inner = inner
        self._job = job

    def _reserve(self, planned: float) -> float:
        job = self._job
        with job.lock:
            if job.terminal or job.cancel.is_set():
                raise CallSkipped("el trabajo se está parando")
            reserve = job.reservation(planned)
            if over_cap(job.usd + job.in_flight_usd + reserve, job.max_usd):
                job.cap_hit = True
                job.cancel.set()
                raise CallSkipped("tope por trabajo")
            job.in_flight_usd += reserve
            return reserve

    def _settle(self, reserve: float, planned: float, result: JevResult) -> None:
        job = self._job
        with job.lock:
            job.in_flight_usd -= reserve
            if job.terminal:
                return
            job.answered += 1
            tokens = result.input_tokens
            if tokens is None:
                job.tokens_unknown += 1
            else:
                job.tokens += tokens
            if unpriced([result.provider]):
                job.unpriced_providers.add(result.provider)
            if tokens is None or unpriced([result.provider]):
                job.charged_at_estimate += 1
                job.usd += reserve
                return
            cost = tokens_cost_usd(tokens, result.provider)
            job.usd += cost
            job.priced_usd += cost
            job.priced_answers += 1
            job.planned_usd += planned

    def ask(self, state: dict[str, str], questions: dict[str, Question]) -> JevResult:
        planned = self._job.planned(state)
        reserve = self._reserve(planned)
        try:
            result = self._inner.ask(state, questions)
        except Exception:
            # Fail closed: a call can raise AFTER the vendor answered and billed (an answer
            # the adapter cannot model), and nothing here can tell which, so a raised call is
            # charged its reservation against the cap.
            with self._job.lock:
                self._job.in_flight_usd -= reserve
                if not self._job.terminal:
                    self._job.failed_calls += 1
                    self._job.usd += reserve
            raise
        except BaseException:
            with self._job.lock:
                self._job.in_flight_usd -= reserve
            raise
        self._settle(reserve, planned, result)
        return result

    def close(self) -> None:
        self._inner.close()


def _set(job: _Job, **fields: Any) -> None:
    with job.lock:
        if job.terminal:
            return
        for key, value in fields.items():
            setattr(job, key, value)


def _reason(job: _Job) -> str:
    if job.cap_hit:
        return "tope"
    return "servidor parado" if job.stopped else "cancelado"


def _finish(job: _Job, outcome: RunOutcome, extra: dict[str, Any]) -> None:
    """Freeze the job with its outcome (plus what its kind adds: the ask's results). A reason
    only when something was actually cut off."""
    logged = outcome.logged
    with job.lock:
        job.state = "interrupted" if outcome.interrupted else "done"
        job.reason = _reason(job) if outcome.interrupted else None
        job.finished_at = _now_iso()
        job.outcome = {
            "ok": len(outcome.assessed),
            "ids": [a.item_id for a in outcome.assessed],
            "failed": len(outcome.failed),
            "failures": [list(failure) for failure in outcome.failed[:FAILURES_SHOWN]],
            "unsaved": logged.unsaved if logged is not None else 0,
            "stored": outcome.stored,
            "logged": logged is not None,
            # Calls that reached the vendor: 0 for a use every answer of which was cached,
            # which has nothing to log and is not a failure to log.
            "sent": job.answered + job.failed_calls,
            **extra,
        }
        job.terminal = True
        job.ended_mono = _monotonic()
    if outcome.interrupted:
        logger.warning(
            "trabajo de Jev interrumpido (%s): %d guardadas de %d",
            job.reason,
            len(outcome.assessed),
            job.total,
        )


def _fail(job: _Job, message: str) -> None:
    with job.lock:
        job.state, job.error, job.finished_at = "error", message, _now_iso()
        job.terminal = True
        job.ended_mono = _monotonic()
    logger.error("el trabajo de Jev falló: %s", message)


#: A kind's pick (`TopicsPick`, `AskPick`) and what its pass carries from estimate to run.
Pk = TypeVar("Pk")
C = TypeVar("C")


@dataclass(frozen=True)
class _Priced(Generic[C]):
    """What one pick selects today and what it would cost — the estimate's answer and the job's
    re-check under the lock alike. `reply` is what the kind adds to the estimate's reply;
    `context` is what its pass needs (`None` for topics; the ask's `AskPlan`)."""

    selection: Selection
    ids: tuple[str, ...]
    tokens: int | None
    usd: float | None
    reply: dict[str, Any]
    context: C


def _cap_refusal(usd: float, cap: float, command: str) -> str | None:
    if not over_cap(usd, cap):
        return None
    return (
        f"la estimación (~{usd:.4f} $) pasa del tope por trabajo "
        f"([jev].serve_max_usd = {cap} $): elige menos posts, o sube el tope en "
        f"config.toml, o lánzalo desde la terminal con `{command}`"
    )


class _PassKind(Protocol[Pk, C]):
    """One kind of paid pass the server runs: how its body is parsed, what it selects and
    costs, why it may not run, and the pass itself — nothing else differs between kinds."""

    def parse(self, body: Any) -> Pk: ...

    def price(self, cfg: Config, pick: Pk, blob: dict[str, Any], jev: JevPairs) -> _Priced[C]: ...

    def refusal(self, cfg: Config, priced: _Priced[C]) -> str | None: ...

    def view(self, pick: Pk) -> dict[str, Any]: ...

    def same(self, confirmed: _Priced[C], now: _Priced[C]) -> bool: ...

    def slim(self, priced: _Priced[C]) -> _Priced[C]: ...

    def run(
        self,
        cfg: Config,
        job: _Job,
        priced: _Priced[C],
        jev: JevPairs,
        make_client: Callable[[], JevClient],
        lock: PassLock,
    ) -> tuple[RunOutcome, dict[str, Any]]: ...


def _hooks(job: _Job) -> dict[str, Any]:
    """The pass's progress and run-log hooks, into the job view."""
    return {
        "on_progress": lambda done, total: _set(job, done=done),
        "on_logged": lambda path, line, error: _log_line(job, path, line, error),
        "cancel": job.cancel,
    }


class _TopicsKind:
    """`xbrain jev topics` from the page: a `TopicsPick` over the blob's posts."""

    def parse(self, body: Any) -> TopicsPick:
        return parse_pick(body)

    def price(
        self, cfg: Config, pick: TopicsPick, blob: dict[str, Any], jev: JevPairs
    ) -> _Priced[None]:
        selection = _select_topics(cfg, pick, blob, jev)
        per_post = blob["cost"]["per_post"]
        ids = tuple(item.id for item in selection.items)
        priced = topics_pass_estimate(per_post, len(ids))
        return _Priced(
            selection,
            ids,
            priced["tokens"],
            priced["usd"],
            {"forced": selection.forced, "per_post": {"n": per_post["n"], "of": per_post["of"]}},
            None,
        )

    def refusal(self, cfg: Config, priced: _Priced[None]) -> str | None:
        selection = priced.selection
        if not priced.ids:
            return (
                f"nada que evaluar: {selection.skipped_current} vigentes, "
                f"{selection.skipped_no_evidence} sin evidencia (volver a evaluar las vigentes pide "
                "`force`: la casilla «Volver a evaluar también…» de la página)"
            )
        if priced.usd is None:
            return (
                "no hay coste medio con el que estimar (ninguna evaluación vigente con tokens y "
                "tarifa), así que el tope no se puede comprobar: haz una primera pasada pequeña "
                "desde la terminal, `xbrain jev topics --limit 5`"
            )
        return _cap_refusal(priced.usd, cfg.jev_serve_max_usd, "xbrain jev topics")

    def view(self, pick: TopicsPick) -> dict[str, Any]:
        return {}

    def same(self, confirmed: _Priced[None], now: _Priced[None]) -> bool:
        return confirmed.ids == now.ids

    def slim(self, priced: _Priced[None]) -> _Priced[None]:
        return priced

    def run(
        self,
        cfg: Config,
        job: _Job,
        priced: _Priced[None],
        jev: JevPairs,
        make_client: Callable[[], JevClient],
        lock: PassLock,
    ) -> tuple[RunOutcome, dict[str, Any]]:
        outcome = run_topics(
            cfg,
            priced.selection,
            dict(jev.assessments),
            jev.vocab,
            lambda: _Metered(make_client(), job),
            lock=lock,
            on_backup=lambda path: _set(job, backup=str(path)),
            **_hooks(job),
        )
        return outcome, {}


def _select_topics(cfg: Config, pick: TopicsPick, blob: dict[str, Any], jev: JevPairs) -> Selection:
    """`assess.select_items` over the posts `pick` names — the `--dry-run` answer.

    A named set that turns out EMPTY selects nothing: handing `ids=[]` down would mean
    "every post" (`select_items`' own convention), a whole-corpus pass out of an empty topic.
    """
    ids = pick_ids(pick, blob, jev.vocab)
    if ids is not None and not ids:
        return Selection(items=(), skipped_current=0, skipped_no_evidence=0)
    try:
        return select_items(
            jev.store,
            jev.assessments,
            jev.vocab,
            ids=ids,
            limit=pick.value if pick.kind == "unevaluated" else None,
            force=pick.force,
            fallback=cfg.jev_fallback_option,
            char_limit=cfg.jev_state_char_limit,
        )
    except JevError as exc:
        raise refuse(str(exc)) from exc


class _AskKind:
    """`xbrain jev ask` from the page: its query, pre-filters and limit (`AskPick`), planned,
    re-checked, run and finished by the command's own functions (`ask.plan_ask`,
    `ask.same_selection`, `run.run_ask`, `ask.finish_ask`) — the same sequence, never a copy.
    Results are RANKED, cut only by the pick's own minimum; the history rule is
    `finish_ask`'s."""

    def parse(self, body: Any) -> AskPick:
        return parse_ask(body)

    def price(
        self, cfg: Config, pick: AskPick, blob: dict[str, Any], jev: JevPairs
    ) -> _Priced[AskPlan]:
        plan = plan_ask(cfg, AskQuery.of(pick.query), pick.filters, pick.limit, jev=jev)
        estimate, model = plan.estimate, plan.estimate.model
        reply = {
            "query_sha": plan.query.sha,
            "dropped": plan.dropped,
            "candidates": len(plan.candidates),
            "chars": estimate.chars,
            "cost_model": {
                "per_call": model.per_call,
                "chars_per_token": model.chars_per_token,
                "measured": model.measured,
                "answers": model.answers,
            },
            "similar": list(plan.similar),
        }
        ids = tuple(item.id for item in plan.selection.items)
        return _Priced(plan.selection, ids, estimate.tokens, estimate.usd, reply, plan)

    def refusal(self, cfg: Config, priced: _Priced[AskPlan]) -> str | None:
        selection = priced.selection
        if not priced.ids and not selection.skipped_current:
            return (
                f"ningún post que preguntar: los filtros dejan {priced.reply['candidates']} "
                f"posts y {selection.skipped_no_evidence} no tienen evidencia"
            )
        if priced.usd is None:
            return "no hay precio con el que estimar esta pregunta, así que el tope no se puede comprobar"
        return _cap_refusal(priced.usd, cfg.jev_serve_max_usd, "xbrain jev ask")

    def view(self, pick: AskPick) -> dict[str, Any]:
        return {"query_sha": AskQuery.of(pick.query).sha}

    def same(self, confirmed: _Priced[AskPlan], now: _Priced[AskPlan]) -> bool:
        return same_selection(confirmed.context, now.context)

    def slim(self, priced: _Priced[AskPlan]) -> _Priced[AskPlan]:
        """What a confirmation keeps: the plan's selection and estimate, for `same` — not
        the query's answers, every candidate's state or the history (32 of them live as long
        as the server)."""
        plan: AskPlan = priced.context
        light = replace(plan, candidates=(), records={}, states={}, history=AskIndex(), similar=())
        return replace(priced, context=light)

    def run(
        self,
        cfg: Config,
        job: _Job,
        priced: _Priced[AskPlan],
        jev: JevPairs,
        make_client: Callable[[], JevClient],
        lock: PassLock,
    ) -> tuple[RunOutcome, dict[str, Any]]:
        """The ask's pass, then — still under the lock — `finish_ask`: its results and, by
        that function's one rule, its line in the history. With every answer current no
        client is built and nothing is logged."""
        plan: AskPlan = priced.context
        pick = job.pick
        if not isinstance(pick, AskPick):  # pragma: no cover — the slot runs its own kind
            raise TypeError("an ask job carries an AskPick")
        job.post_price = _post_price(plan)
        on_answer: Callable[[AskAssessment], None] | None = None
        try:
            answer = _stream_entry(cfg, plan, jev)
        except Exception as exc:
            # Display only: the job runs with no stream (the page shows its results at the
            # end), never refused over what the page would show.
            logger.warning(
                "sin resultados en vivo: no se pudo preparar el stream (%s: %s)",
                type(exc).__name__,
                exc,
            )
        else:
            _open_stream(
                job,
                lambda: [answer(record, True) for _, record in current_answers(plan).ranked],
                asking=len(plan.selection.items),
                minimum=use_minimum(plan, pick.minimum),
            )
            on_answer = lambda record: _stream(job, answer(record, False))  # noqa: E731
        outcome = run_ask(
            cfg,
            plan,
            lambda: _Metered(make_client(), job),
            lock=lock,
            on_answer=on_answer,
            **_hooks(job),
        )
        try:
            found = finish_ask(cfg, plan, outcome, minimum=pick.minimum)
        except Exception as exc:
            # The answers are paid, saved and logged by now: a history that cannot be written
            # must not turn them into «El trabajo falló» with nothing to show.
            logger.error("no se pudo escribir el historial de consultas: %s", exc)
            return outcome, {
                "recorded": False,
                "file": str(ask_path(cfg, plan.query)),
                "history_error": str(exc),
            }
        return outcome, {
            "results": len(found.ranked),
            # The minimum `results` is cut at (`finish_ask`'s: this use's, else the query's
            # last): the page names it beside the count.
            "min": use_minimum(plan, pick.minimum),
            "answered": found.answered,
            "recorded": found.recorded,
        }


#: The filters `/api/ask/counts` reads from its query string (topics are what it counts).
_COUNT_PARAMS = ("since", "until", "author", "only_evaluated")


def _filters_from_query(params: dict[str, list[str]]) -> dict[str, Any]:
    """A query string as `AskFilters.from_json` data: one value per key, `only_evaluated` as
    `true`/`false`; anything else is a 400 naming it."""
    unknown = sorted(set(params) - set(_COUNT_PARAMS))
    if unknown:
        raise refuse(f"parámetro desconocido: {unknown[0]}")
    data: dict[str, Any] = {}
    for key, values in params.items():
        if len(values) != 1:
            raise refuse(f"el parámetro {key} va una sola vez")
        data[key] = values[0]
    if "only_evaluated" in data:
        flag = {"true": True, "false": False}.get(data["only_evaluated"])
        if flag is None:
            raise refuse("el parámetro only_evaluated debe ser true o false")
        data["only_evaluated"] = flag
    return data


def _stream_entry(
    cfg: Config, plan: AskPlan, jev: JevPairs
) -> Callable[[AskAssessment, bool], dict[str, Any]]:
    """How one answer of `plan` goes on the job's stream: `dashboard.streamed_answer` over its
    post, with the post's CURRENT topics answer at `[jev].threshold` — the blob's inputs."""
    posts = {item.id: item for item in plan.candidates}
    current = jev.current_by_id()

    def entry(record: AskAssessment, cached: bool) -> dict[str, Any]:
        return streamed_answer(
            posts[record.item_id],
            record,
            current.get(record.item_id),
            threshold=cfg.jev_threshold,
            char_limit=cfg.jev_state_char_limit,
            cached=cached,
        )

    return entry


def _open_stream(
    job: _Job, current: Callable[[], list[dict[str, Any]]], *, asking: int, minimum: float
) -> None:
    """The job's stream, opened with the answers the query already has (`current()`). Display
    only: one that cannot be built is logged and the stream opens without them — the pass,
    which pays, is never stopped by what the page shows."""
    try:
        opened = current()
    except Exception as exc:
        logger.warning(
            "no se pudieron mostrar las respuestas ya guardadas (%s: %s)", type(exc).__name__, exc
        )
        opened = []
    with job.lock:
        job.stream = opened
        job.stream_size = len(opened)
        job.stream_expected = len(opened) + asking
        job.stream_min = minimum


def _stream(job: _Job, entry: dict[str, Any]) -> None:
    """`run_ask`'s `on_answer`: one banked answer onto the stream (append-only; a cursor is an
    index). A display hook: `run._call_hook` logs what it raises and the answer is kept."""
    with job.lock:
        if job.stream is not None and not job.terminal:
            job.stream.append(entry)
            job.stream_size += 1


def _read_cursor(params: dict[str, list[str]]) -> tuple[int, int]:
    """`/api/job`'s query string: exactly one `since`, a cursor this server hands out (`0`, or
    `<job number>-<index>` from a reply's `next`) as `(number, index)`; `0` is `(0, 0)`, which
    names no job (they are numbered from 1). Anything else is a 400 naming it."""
    unknown = sorted(set(params) - {"since"})
    if unknown:
        raise refuse(f"parámetro desconocido: {unknown[0]}")
    values = params.get("since", [])
    if len(values) != 1:
        raise refuse("el parámetro since va una sola vez")
    found = _CURSOR.match(values[0])
    if found is None:
        raise refuse(f"since no es un cursor: {values[0]!r} (0, o el `next` de una respuesta)")
    if found.group(1) is None:
        return 0, 0
    return int(found.group(1)), int(found.group(2))


def _post_price(plan: AskPlan) -> Callable[[dict[str, str]], float]:
    """Each post's own planned price, from the characters its call sends (the state as sent
    plus the question) by the plan's cost model — scaled so the posts selected add up to the
    estimate confirmed, so a job at exactly the cap is never refused its last post."""
    model = plan.estimate.model
    question = question_chars(plan.query.questions)

    def raw(chars: int) -> float:
        return tokens_cost_usd(model.tokens(chars), DEFAULT_PROVIDER)

    total = sum(raw(len(plan.states[item.id]) + question) for item in plan.selection.items)
    scale = plan.estimate.usd / total if total else 1.0
    return lambda state: raw(len(state[STATE_KEY]) + question) * scale


def _result_surfaces(row: dict[str, Any], blob: dict[str, Any]) -> dict[str, Any]:
    """What Jev read for each result of `row`: the card's own Jev block when it has one, else
    the tab's `asks.surfaces` (the blob sends each only once)."""
    cards = {card["id"]: card for card in blob["posts"]}
    out: dict[str, Any] = {}
    for post_id in row["answers"]["ids"]:
        card = cards.get(post_id)
        jev = card.get("jev") if card else None
        out[post_id] = jev["surfaces"] if jev else blob["asks"]["surfaces"].get(post_id, [])
    return out


def _signature(paths: list[Path]) -> tuple[tuple[int, int] | None, ...]:
    """(mtime, size) of each input, `None` for a missing one: what the cached page depends on."""
    stamps: list[tuple[int, int] | None] = []
    for path in paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            stamps.append(None)
        else:
            stamps.append((stat.st_mtime_ns, stat.st_size))
    return tuple(stamps)


def _refuse_nothing_to_ask(jev: JevPairs, cfg: Config) -> None:
    """No vocabulary or no posts: nothing to ask, or nothing to ask it with. An EMPTY SIDE-CAR
    is served — unlike `jev dashboard`, which refuses it — because a first pass can start here.
    """
    if not jev.vocab:
        raise JevError(
            f"el vocabulario está vacío o falta {cfg.vocab_path}: ejecuta `xbrain vocab`"
        )
    if not jev.store:
        raise JevError(f"no hay items en {cfg.items_path}: ejecuta `xbrain extract`")


class JevService:
    """Everything the server does, without HTTP: the page's data, estimates, and the one job.

    `make_client` builds the client a job asks through — `cli._jev_client` in production, a
    fake in tests. It is called only inside the pass (`run_topics` / `run_ask`), under the lock,
    so serving the page never needs a key.
    """

    def __init__(
        self,
        cfg: Config,
        make_client: Callable[[], JevClient],
        *,
        token: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.token = token or secrets.token_urlsafe(32)
        self._make_client = make_client
        self._build_lock = threading.Lock()
        self._state = threading.Lock()
        self._media: MediaFiles | None = None
        self._cache: tuple[Any, dict[str, Any], JevPairs] | None = None
        self._pairs_lock = threading.Lock()
        self._pairs_cache: tuple[Any, JevPairs] | None = None
        self._html: tuple[Any, str] | None = None
        self._confirms: dict[str, _Confirm] = {}
        #: THE job slot: the job being started (under the lock, not yet published) and the
        #: current or last one. One for the whole server, whatever the kind.
        self._starting: _Job | None = None
        self._job: _Job | None = None
        #: How many jobs this server has numbered (`_Job.number`).
        self._numbered = 0
        self._closing = False
        #: The per-kind part of the money path; the slot, the lock and the cap are shared.
        self._kinds: dict[str, _PassKind[Any, Any]] = {
            KIND_TOPICS: _TopicsKind(),
            KIND_ASK: _AskKind(),
        }

    # ------------------------------------------------------------------ the page's data

    def _inputs(self) -> list[Path]:
        cfg = self.cfg
        return [
            cfg.items_path,
            cfg.vocab_path,
            cfg.jev_topics_path,
            cfg.jev_runs_path,
            *report_paths(cfg.jev_dir),
            cfg.jev_page_path,
            # The query history and every query's answers (a terminal's Ctrl-C writes the
            # answers without the history).
            cfg.jev_asks_dir / ASK_INDEX,
            *sorted(cfg.jev_asks_dir.glob("*.json")),
        ]

    def _built(self) -> tuple[Any, dict[str, Any], JevPairs]:
        """The blob and the loader's result, rebuilt only when an input file changed.

        The media look-up (thousands of `stat`s on an iCloud vault) is made once per server:
        photos `xbrain generate` mirrors later show after a restart.
        """
        with self._build_lock:
            signature = _signature(self._inputs())
            if self._cache is not None and self._cache[0] == signature:
                return self._cache
            # Read BEFORE the files: a job that had finished by now wrote them before this.
            finished_at, finished_job = self._last_finished()
            jev = self._pairs()
            if self._media is None:
                self._media = collect_jev_media(
                    list(jev.store.values()), self.cfg.output_dir, self.cfg.media_dir
                )
            blob = build_page_data(
                self.cfg, now=datetime.now(timezone.utc), jev=jev, served=True, media=self._media
            )
            blob["serve"] = {
                "token": self.token,
                "max_usd": self.cfg.jev_serve_max_usd,
                # The last job whose files this data already includes: an idle tab whose
                # `/api/job` reports a finished job of another `number` has older data and
                # reloads it (`finished_at`, to the second, cannot tell two ends apart).
                "finished_at": finished_at,
                "finished_job": finished_job,
            }
            self._cache = (signature, blob, jev)
            return self._cache

    def _pairs(self) -> JevPairs:
        """The posts and their current topics answers (`load_jev_pairs`), reloaded only when
        the items, the vocabulary or the topics side-car changed — never for a query's answers
        or the run log, which an ask job writes at every checkpoint. What the blob is built
        from, and all `/api/ask/counts` reads: a counts GET never rebuilds the page."""
        cfg = self.cfg
        with self._pairs_lock:
            signature = _signature([cfg.items_path, cfg.vocab_path, cfg.jev_topics_path])
            if self._pairs_cache is not None and self._pairs_cache[0] == signature:
                return self._pairs_cache[1]
            jev = load_jev_pairs(cfg)
            _refuse_nothing_to_ask(jev, cfg)
            self._pairs_cache = (signature, jev)
            return jev

    def _last_finished(self) -> tuple[str | None, int | None]:
        """The last job's `finished_at` and `number`; `(None, None)` while none has ended."""
        with self._state:
            job = self._job
        if job is None:
            return None, None
        with job.lock:
            if job.finished_at is None:
                return None, None
            return job.finished_at, job.number

    def blob(self) -> dict[str, Any]:
        """The page's data, as `/api/data` sends it and the page embeds it."""
        return self._built()[1]

    def page_html(self) -> str:
        signature, blob, _ = self._built()
        with self._build_lock:
            if self._html is None or self._html[0] != signature:
                self._html = (signature, render_jev_dashboard_html(blob))
            return self._html[1]

    # ------------------------------------------------------------------ picks and prices

    def asks(self) -> dict[str, Any]:
        """The «Preguntar» history with each query's results, as `/api/asks` sends it: the
        blob's `asks`, so the static page and the server never disagree. Costs nothing."""
        return self.blob()["asks"]

    def ask(self, sha: str) -> dict[str, Any]:
        """One query of the history, with its results; 404 for a query never asked."""
        blob = self.blob()
        for row in blob["asks"]["history"]:
            if row["sha"] == sha:
                return {**row, "surfaces": _result_surfaces(row, blob)}
        raise ServeError(404, "esa consulta no está en el historial")

    def ask_counts(self, params: dict[str, list[str]]) -> dict[str, Any]:
        """How many posts each topic keeps under the filters in `params` (a query string:
        `since`, `until`, `author`, `only_evaluated=true|false`; `ask.topic_counts`, the
        filter's own rule) and how many those filters keep: what the Preguntar tab writes
        beside each topic. A GET: reads only, costs nothing, and reads the cached posts and
        topics answers (`_pairs`), never the page's data — so a cross-site GET cannot force a
        rebuild, and typing while a job checkpoints does not either."""
        filters = parse_filters(_filters_from_query(params))
        jev = self._pairs()
        threshold = self.cfg.jev_threshold
        try:
            kept, _ = filter_posts(jev.store, filters, jev=jev, threshold=threshold)
            counts = topic_counts(jev.store, filters, jev=jev, threshold=threshold)
        except JevError as exc:
            raise refuse(str(exc)) from exc
        return {"posts": len(kept), "topic_counts": counts}

    # ------------------------------------------------------------------ picks and prices

    def _kind(self, kind: str) -> _PassKind[Any, Any]:
        found = self._kinds.get(kind)
        if found is None:
            raise ServeError(404, f"no hay pasadas de tipo «{kind}»")
        return found

    def estimate(self, kind: str, body: Any) -> dict[str, Any]:
        """How many posts `body`'s pick would ask, what that would cost, and — when it may
        run — the single-use confirmation `evaluate` needs."""
        pass_kind = self._kind(kind)
        pick = pass_kind.parse(body)
        with self._state:
            if self._closing:
                raise ServeError(503, "el servidor se está parando")
        _, blob, jev = self._built()
        priced = pass_kind.price(self.cfg, pick, blob, jev)
        refusal = pass_kind.refusal(self.cfg, priced)
        confirm_token = None
        if refusal is None:
            confirm_token = secrets.token_urlsafe(16)
            confirm = _Confirm(
                kind, pick, priced.ids, priced.usd or 0.0, _monotonic(), pass_kind.slim(priced)
            )
            with self._state:
                self._confirms[confirm_token] = confirm
                while len(self._confirms) > _CONFIRMS_KEPT:
                    self._confirms.pop(next(iter(self._confirms)))
        selection = priced.selection
        return {
            "kind": kind,
            "pick": pick.as_json(),
            "ids": list(priced.ids),
            "posts": len(priced.ids),
            "skipped_current": selection.skipped_current,
            "skipped_no_evidence": selection.skipped_no_evidence,
            "remaining": selection.remaining,
            **priced.reply,
            "tokens": priced.tokens,
            "usd": priced.usd,
            "estimate": True,
            "max_usd": self.cfg.jev_serve_max_usd,
            "allowed": refusal is None,
            "refusal": refusal,
            "confirm_token": confirm_token,
            "expires_in_s": CONFIRM_TTL_S if confirm_token else None,
        }

    # ------------------------------------------------------------------ the job slot

    def _claim(self, kind: str, token: str, pick: TopicsPick | AskPick) -> tuple[_Job, _Confirm]:
        """Under the state lock: the slot is free, the server is not stopping, and `token` is
        a live confirmation of exactly this pick — then the confirmation is spent and the job
        is the one being started. A refusal here spends nothing."""
        with self._state:
            if self._closing:
                raise ServeError(503, "el servidor se está parando")
            if self._starting is not None or (self._job is not None and not self._job.terminal):
                raise ServeError(409, "ya hay un trabajo en curso: espera a que termine")
            confirm = self._confirms.get(token)
            if confirm is None or confirm.kind != kind or confirm.pick != pick:
                raise ServeError(
                    409,
                    "esta confirmación no corresponde a esta selección, o ya se usó: "
                    "vuelve a estimar",
                )
            del self._confirms[token]
            if _monotonic() - confirm.minted_at > CONFIRM_TTL_S:
                raise ServeError(409, "la confirmación caducó: vuelve a estimar")
            # A query every candidate already answers asks nothing: no reservation to make.
            per_post = confirm.usd / len(confirm.ids) if confirm.ids else 0.0
            job = _Job(kind, pick, confirm.ids, self.cfg.jev_serve_max_usd, per_post)
            self._numbered += 1
            job.number = self._numbered
            job.extra = self._kinds[kind].view(pick)
            job.confirmed = confirm.priced
            self._starting = job
            return job, confirm

    def evaluate(self, kind: str, body: Any) -> dict[str, Any]:
        """Start the job an estimate confirmed. Returns once it holds the lock and has checked
        the pick is still the posts and the price confirmed; 409/503 otherwise."""
        pass_kind = self._kind(kind)
        if not isinstance(body, dict) or not isinstance(body.get("confirm_token"), str):
            raise refuse("falta confirm_token: pide antes una estimación")
        body = dict(body)
        token = body.pop("confirm_token")
        job, confirm = self._claim(kind, token, pass_kind.parse(body))
        job.thread = threading.Thread(target=self._run, args=(job,), name=f"jev-serve-{kind}")
        try:
            job.thread.start()
        except BaseException as exc:
            with self._state:
                self._starting = None
                self._confirms[token] = confirm
            raise ServeError(503, f"no se pudo arrancar el trabajo: {exc}") from exc
        job.ready.wait()
        with self._state:
            self._starting = None
            if job.refusal is None:
                self._job = job
        if job.refusal is not None:
            raise ServeError(job.refusal_status, job.refusal)
        return job.view()

    def _recheck(self, job: _Job, blob: dict[str, Any], jev: JevPairs) -> _Priced[Any] | None:
        """Under the lock, before anything is written: still running, still the same posts,
        still under the cap at today's price. `None` (and `job.refusal`) when not."""
        if job.cancel.is_set():
            job.refusal, job.refusal_status = "el servidor se está parando", 503
            return None
        kind = self._kinds[job.kind]
        priced = kind.price(self.cfg, job.pick, blob, jev)
        if not kind.same(job.confirmed, priced):
            job.refusal = (
                "la selección cambió desde la estimación (otra pasada o un cambio en los "
                "datos): vuelve a estimar"
            )
            return None
        usd = priced.usd
        if usd is None or over_cap(usd, job.max_usd):
            job.refusal = "el precio estimado cambió y ya no cabe en el tope: vuelve a estimar"
            return None
        if job.ids:
            job.per_post_usd = max(job.per_post_usd, usd / len(job.ids))
        return priced

    def _run(self, job: _Job) -> None:
        """The job's thread, for every kind: lock, reload, re-check, run the kind's ONE loop,
        record the end."""
        try:
            with pass_lock(self.cfg.jev_lock_path, LOCK_HOLDER) as lock:
                _, blob, jev = self._built()
                priced = self._recheck(job, blob, jev)
                if priced is None:
                    return
                _set(job, state="running", total=len(job.ids), started_at=_now_iso())
                job.ready.set()
                outcome, extra = self._kinds[job.kind].run(
                    self.cfg, job, priced, jev, self._make_client, lock
                )
            _finish(job, outcome, extra)
        except Exception as exc:
            message = str(exc) if isinstance(exc, (JevError, ServeError)) else repr(exc)
            if not job.ready.is_set():
                job.refusal = message
            else:
                if not isinstance(exc, JevError):
                    logger.exception("el trabajo de Jev falló")
                _fail(job, message)
        finally:
            job.ready.set()

    def cancel_job(self) -> dict[str, Any]:
        """The page's «Parar»: the running job stops SOFTLY — nothing queued is sent, the calls
        in flight are waited for, saved and logged — and ends as «cancelado». Returns its view
        (still running: the page follows it to the end); 409 when no job is running."""
        with self._state:
            job = self._job
        if job is not None:
            with job.lock:
                if not job.terminal:
                    job.cancel.set()
                    return job.view_unlocked()
        raise ServeError(409, "no hay ningún trabajo en curso")

    def job_view(self, params: dict[str, list[str]] | None = None) -> dict[str, Any]:
        """The current (or last) job, as `/api/job` sends it; `{"state": "idle"}` before any.
        With a query string (`params`: `since=<cursor>`), its stream from that cursor too
        (`stream`, `None` for a job with none — topics — or no job at all)."""
        since = _read_cursor(params) if params else None
        with self._state:
            job = self._job
        if job is None:
            return {"state": "idle"} if since is None else {"state": "idle", "stream": None}
        return job.view(since)

    def wait(self, timeout: float | None = None) -> None:
        """Block until the jobs this server started have ended (for tests and `stop`)."""
        with self._state:
            jobs = [self._starting, self._job]
        for job in jobs:
            if job is not None and job.thread is not None and job.thread.is_alive():
                job.thread.join(timeout)

    def stop(self) -> None:
        """Refuse new jobs (503), cancel the one running or starting at its next call, and
        wait for it to save and log what it was paid for."""
        with self._state:
            self._closing = True
            jobs = [self._starting, self._job]
        for job in jobs:
            if job is None:
                continue
            with job.lock:
                if not job.terminal and not job.cancel.is_set():
                    job.stopped = True
                    job.cancel.set()
        self.wait()


def _log_line(job: _Job, path: Path, line: str, error: BaseException | None) -> None:
    """`run_topics`' `on_logged`: a line that could not be appended reaches the terminal log
    whole, so the operator can append it by hand, and the job view."""
    if error is None:
        return
    _set(job, log_error=f"{error}")
    logger.error("no se pudo registrar la pasada en %s (%s); añádela a mano: %s", path, error, line)
