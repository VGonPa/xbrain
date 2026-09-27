"""What `xbrain jev serve` does, without HTTP: the page's data, estimates, and the ONE job.

THE MONEY PATH is estimate → confirm → one job, per kind (`topics` today; the «preguntar» ask
of PRs 11–13 adds its own kind). ONE JOB SLOT for the whole server, whatever the kind, and
every paid pass takes the pass lock (`jev.lock`), so the page and a terminal never run two
passes over one side-car at once.

* `estimate(kind, body)`: the pick (`jev.picks`) resolved to posts from the blob the page is
  showing, `assess.select_items` over them (the `--dry-run` answer), priced by
  `report.topics_pass_estimate` over the blob's `cost.per_post` — the one mean the cost strip
  and the Configuración tab show. Under `[jev].serve_max_usd` (at most equal) it mints a
  single-use confirmation, bound to the kind and the pick AS ASKED, that expires after
  `CONFIRM_TTL_S`. With no priced mean there is nothing to check the cap against, so no
  confirmation: fail-closed.
* `evaluate(kind, body)`: refused with 503 while the server stops, 409 while a job runs (the
  confirmation is kept), 409 for a confirmation that is unknown, spent, expired or for another
  pick. Otherwise the job thread takes the pass lock, re-reads, re-selects and RE-PRICES, and
  refuses (409, before any client exists and before any backup) if the posts moved or the
  price went over the cap. Only a job that passed all that is published in the slot.
* The job runs `run.run_topics` — the terminal's pass — through `_Metered`, which makes the
  cap a HARD bound by reservation: before each call it reserves that post's expected cost (the
  estimate's mean, or this job's own priced mean when higher) and does not send when spent +
  reserved + the next reservation would pass the cap. An answer replaces its reservation with
  its real cost; an answer with no token count or from a provider with no price is charged
  the reservation, never $0. So the bill passes the cap only by what the posts in flight
  cost above their reservation.
* Every stop from here is SOFT (`run_topics(cancel=…)`): nothing queued is sent, every call
  in flight is waited for, banked, saved and logged. Counters freeze when the job ends.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xbrain.config import Config
from xbrain.jev.assess import Selection, select_items
from xbrain.jev.client import CallSkipped, JevClient, JevError, JevResult, Question
from xbrain.jev.dashboard import (
    MediaFiles,
    build_page_data,
    collect_jev_media,
    render_jev_dashboard_html,
)
from xbrain.jev.defaults import tokens_cost_usd, unpriced
from xbrain.jev.load import JevPairs, load_jev_pairs
from xbrain.jev.lock import pass_lock
from xbrain.jev.picks import ServeError, TopicsPick, parse_pick, pick_ids, refuse
from xbrain.jev.report import report_paths, topics_pass_estimate
from xbrain.jev.run import RunOutcome, run_topics

logger = logging.getLogger(__name__)

#: The kind of pass this server can run today. Routes are `/api/<kind>/estimate|evaluate`.
KIND_TOPICS = "topics"
#: What the server calls itself in the pass lock, for the message another pass gets.
LOCK_HOLDER = "xbrain jev serve"
#: Confirmations kept at once; the oldest is dropped past this (each is single-use anyway).
_CONFIRMS_KEPT = 32
#: A confirmation older than this is refused: the price it was minted at may have moved.
CONFIRM_TTL_S = 600
#: Failures listed in a finished job's outcome, like the CLI's `_JEV_FAILURES_SHOWN`.
_FAILURES_SHOWN = 10
#: Float room at the cap: `n × mean` summed call by call must not refuse the n-th post the
#: estimate allowed at exactly the cap.
_CAP_ROOM = 1e-9


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
    pick: TopicsPick
    ids: tuple[str, ...]
    usd: float
    minted_at: float


@dataclass
class _Job:
    """One background pass and everything the page polls about it. Guarded by `lock`."""

    kind: str
    pick: TopicsPick
    ids: tuple[str, ...]
    max_usd: float
    #: The reservation per post: the confirmed estimate's mean (re-priced under the lock).
    per_post_usd: float
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

    def reservation(self) -> float:
        """What the next post is expected to cost: the estimate's mean, or this job's own
        priced mean once it is higher (the corpus may be pricier than its past)."""
        own = self.priced_usd / self.priced_answers if self.priced_answers else 0.0
        return max(self.per_post_usd, own)

    def view(self) -> dict[str, Any]:
        with self.lock:
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

    def _reserve(self) -> float:
        job = self._job
        with job.lock:
            if job.terminal or job.cancel.is_set():
                raise CallSkipped("el trabajo se está parando")
            reserve = job.reservation()
            if over_cap(job.usd + job.in_flight_usd + reserve, job.max_usd):
                job.cap_hit = True
                job.cancel.set()
                raise CallSkipped("tope por trabajo")
            job.in_flight_usd += reserve
            return reserve

    def _settle(self, reserve: float, result: JevResult) -> None:
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

    def ask(self, state: dict[str, str], questions: dict[str, Question]) -> JevResult:
        reserve = self._reserve()
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
        self._settle(reserve, result)
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


def _finish(job: _Job, outcome: RunOutcome) -> None:
    """Freeze the job with its outcome. A reason only when something was actually cut off."""
    logged = outcome.logged
    with job.lock:
        job.state = "interrupted" if outcome.interrupted else "done"
        job.reason = _reason(job) if outcome.interrupted else None
        job.finished_at = _now_iso()
        job.outcome = {
            "ok": len(outcome.assessed),
            "ids": [a.item_id for a in outcome.assessed],
            "failed": len(outcome.failed),
            "failures": [list(failure) for failure in outcome.failed[:_FAILURES_SHOWN]],
            "unsaved": logged.unsaved if logged is not None else 0,
            "stored": outcome.stored,
            "logged": logged is not None,
        }
        job.terminal = True
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
    logger.error("el trabajo de Jev falló: %s", message)


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
    fake in tests. It is called only inside `run_topics`, after the backup and under the lock,
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
        self._html: tuple[Any, str] | None = None
        self._confirms: dict[str, _Confirm] = {}
        #: THE job slot: the job being started (under the lock, not yet published) and the
        #: current or last one. One for the whole server, whatever the kind.
        self._starting: _Job | None = None
        self._job: _Job | None = None
        self._closing = False
        self._runners: dict[str, Callable[[_Job], None]] = {KIND_TOPICS: self._run_topics}

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
            jev = load_jev_pairs(self.cfg)
            _refuse_nothing_to_ask(jev, self.cfg)
            if self._media is None:
                self._media = collect_jev_media(
                    list(jev.store.values()), self.cfg.output_dir, self.cfg.media_dir
                )
            blob = build_page_data(
                self.cfg, now=datetime.now(timezone.utc), jev=jev, served=True, media=self._media
            )
            blob["serve"] = {"token": self.token, "max_usd": self.cfg.jev_serve_max_usd}
            self._cache = (signature, blob, jev)
            return self._cache

    def blob(self) -> dict[str, Any]:
        """The page's data, as `/api/data` sends it and the page embeds it."""
        return self._built()[1]

    def page_html(self) -> str:
        signature, blob, _ = self._built()
        with self._build_lock:
            if self._html is None or self._html[0] != signature:
                self._html = (signature, render_jev_dashboard_html(blob))
            return self._html[1]

    def cards(self, ids: list[str]) -> list[dict[str, Any]]:
        """Card bodies by id, in the order asked; a 404 naming the ids the corpus lacks. The
        by-id refresh for result lists (the «preguntar» results of PR 12)."""
        by_id = {card["id"]: card for card in self.blob()["posts"]}
        missing = [item_id for item_id in ids if item_id not in by_id]
        if missing:
            raise ServeError(404, f"posts desconocidos: {', '.join(missing)}")
        return [by_id[item_id] for item_id in ids]

    # ------------------------------------------------------------------ picks and prices

    def _select(self, pick: TopicsPick, blob: dict[str, Any], jev: JevPairs) -> Selection:
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
                fallback=self.cfg.jev_fallback_option,
                char_limit=self.cfg.jev_state_char_limit,
            )
        except JevError as exc:
            raise refuse(str(exc)) from exc

    def _kind(self, kind: str) -> Callable[[_Job], None]:
        runner = self._runners.get(kind)
        if runner is None:
            raise ServeError(404, f"no hay pasadas de tipo «{kind}»")
        return runner

    def estimate(self, kind: str, body: Any) -> dict[str, Any]:
        """How many posts `body`'s pick would ask, what that would cost by the page's mean,
        and — when it may run — the single-use confirmation `evaluate` needs."""
        self._kind(kind)
        pick = parse_pick(body)
        with self._state:
            if self._closing:
                raise ServeError(503, "el servidor se está parando")
        _, blob, jev = self._built()
        selection = self._select(pick, blob, jev)
        per_post = blob["cost"]["per_post"]
        ids = tuple(item.id for item in selection.items)
        priced = topics_pass_estimate(per_post, len(ids))
        refusal = self._refusal(len(ids), priced["usd"], selection)
        confirm_token = None
        if refusal is None:
            confirm_token = secrets.token_urlsafe(16)
            confirm = _Confirm(kind, pick, ids, priced["usd"], _monotonic())
            with self._state:
                self._confirms[confirm_token] = confirm
                while len(self._confirms) > _CONFIRMS_KEPT:
                    self._confirms.pop(next(iter(self._confirms)))
        return {
            "kind": kind,
            "pick": pick.as_json(),
            "ids": list(ids),
            "posts": len(ids),
            "skipped_current": selection.skipped_current,
            "skipped_no_evidence": selection.skipped_no_evidence,
            "forced": selection.forced,
            "remaining": selection.remaining,
            "tokens": priced["tokens"],
            "usd": priced["usd"],
            "estimate": True,
            "per_post": {"n": per_post["n"], "of": per_post["of"]},
            "max_usd": self.cfg.jev_serve_max_usd,
            "allowed": refusal is None,
            "refusal": refusal,
            "confirm_token": confirm_token,
            "expires_in_s": CONFIRM_TTL_S if confirm_token else None,
        }

    def _refusal(self, posts: int, usd: float | None, selection: Selection) -> str | None:
        """Why this pick may not run from the server, or `None` when it may."""
        if not posts:
            return (
                f"nada que evaluar: {selection.skipped_current} vigentes, "
                f"{selection.skipped_no_evidence} sin evidencia (re-evaluar pide «forzar»)"
            )
        if usd is None:
            return (
                "no hay coste medio con el que estimar (ninguna evaluación vigente con tokens y "
                "tarifa), así que el tope no se puede comprobar: haz una primera pasada pequeña "
                "desde la terminal, `xbrain jev topics --limit 5`"
            )
        cap = self.cfg.jev_serve_max_usd
        if over_cap(usd, cap):
            return (
                f"la estimación (~{usd:.4f} $) pasa del tope por trabajo "
                f"([jev].serve_max_usd = {cap} $): elige menos posts, o sube el tope en "
                "config.toml, o lánzalo desde la terminal con `xbrain jev topics`"
            )
        return None

    # ------------------------------------------------------------------ the job slot

    def _claim(self, kind: str, token: str, pick: TopicsPick) -> tuple[_Job, _Confirm]:
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
            per_post = confirm.usd / len(confirm.ids)
            job = _Job(kind, pick, confirm.ids, self.cfg.jev_serve_max_usd, per_post)
            self._starting = job
            return job, confirm

    def evaluate(self, kind: str, body: Any) -> dict[str, Any]:
        """Start the job an estimate confirmed. Returns once it holds the lock and has checked
        the pick is still the posts and the price confirmed; 409/503 otherwise."""
        runner = self._kind(kind)
        if not isinstance(body, dict) or not isinstance(body.get("confirm_token"), str):
            raise refuse("falta confirm_token: pide antes una estimación")
        body = dict(body)
        token = body.pop("confirm_token")
        job, confirm = self._claim(kind, token, parse_pick(body))
        job.thread = threading.Thread(target=runner, args=(job,), name=f"jev-serve-{kind}")
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

    def _recheck(self, job: _Job, blob: dict[str, Any], jev: JevPairs) -> Selection | None:
        """Under the lock, before anything is written: still running, still the same posts,
        still under the cap at today's mean. `None` (and `job.refusal`) when not."""
        if job.cancel.is_set():
            job.refusal, job.refusal_status = "el servidor se está parando", 503
            return None
        selection = self._select(job.pick, blob, jev)
        if tuple(item.id for item in selection.items) != job.ids:
            job.refusal = (
                "la selección cambió desde la estimación (otra pasada o un cambio en los "
                "datos): vuelve a estimar"
            )
            return None
        usd = topics_pass_estimate(blob["cost"]["per_post"], len(job.ids))["usd"]
        if usd is None or over_cap(usd, job.max_usd):
            job.refusal = "el precio estimado cambió y ya no cabe en el tope: vuelve a estimar"
            return None
        job.per_post_usd = max(job.per_post_usd, usd / len(job.ids))
        return selection

    def _run_topics(self, job: _Job) -> None:
        """The topics job's thread: lock, reload, re-check, run the ONE loop, record the end."""
        try:
            with pass_lock(self.cfg.jev_lock_path, LOCK_HOLDER) as lock:
                _, blob, jev = self._built()
                selection = self._recheck(job, blob, jev)
                if selection is None:
                    return
                _set(job, state="running", total=len(job.ids), started_at=_now_iso())
                job.ready.set()
                outcome = run_topics(
                    self.cfg,
                    selection,
                    dict(jev.assessments),
                    jev.vocab,
                    lambda: _Metered(self._make_client(), job),
                    lock=lock,
                    on_backup=lambda path: _set(job, backup=str(path)),
                    on_progress=lambda done, total: _set(job, done=done),
                    on_logged=lambda path, line, error: _log_line(job, path, line, error),
                    cancel=job.cancel,
                )
            _finish(job, outcome)
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

    def job_view(self) -> dict[str, Any]:
        """The current (or last) job, as `/api/job` sends it; `{"state": "idle"}` before any."""
        with self._state:
            job = self._job
        return {"state": "idle"} if job is None else job.view()

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
