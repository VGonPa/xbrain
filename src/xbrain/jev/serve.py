"""`xbrain jev serve`: the Jev page served live on 127.0.0.1, with a JSON API to evaluate from it.

WHAT IT ADDS TO THE STATIC PAGE. `jev.html` can only hand over a command to copy. Served, the
page can ask for an ESTIMATE of a selection (a post, a topic, the next N unevaluated posts, the
posts behind a pair or a confidence band), confirm it, and start ONE background job that runs
`jev.run.run_topics` — the same loop `xbrain jev topics` runs, with the same backup rule,
checkpoints, run-log line and teardown. Nothing here asks Jev any other way.

NOTHING HERE COMPUTES A NUMBER EITHER. The page is `dashboard.build_page_data` (with
`served=True`); the selection is `assess.select_items`, the `--dry-run` count; the estimate is
`report.topics_pass_estimate` over the page's own `cost.per_post` (the mean cost of the answers
already paid for) — the figure the Configuración tab shows.

THE MONEY RULES, all server-side:

* a POST needs the per-process token embedded in the page (header `X-Xbrain-Token`), an
  `Origin` that is this server, and JSON; every request needs a `Host` that is this server
  (a page elsewhere that rebinds its DNS to 127.0.0.1 still sends its own name);
* `/api/evaluate` runs only a selection that `/api/estimate` priced under
  `[jev].serve_max_usd` and minted a single-use `confirm_token` for — and only if the same
  selection, recomputed under the pass lock, is still exactly those posts;
* the job is stopped when what it has REALLY spent (the answers' own tokens) reaches the cap:
  the estimate is a mean and a real bill can be larger;
* the pass lock (`jev.lock`) is held from loading the side-car to the end of the job, so a
  terminal `xbrain jev topics` and the page never run at once;
* Ctrl-C stops accepting requests, cancels the job at its next call (what was answered is
  saved and logged) and exits 130.

Bound to 127.0.0.1 only; there is no option to bind anything else.
"""

from __future__ import annotations

import hmac
import json
import logging
import mimetypes
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from xbrain.config import Config
from xbrain.generate import VAULT_MEDIA_SUBDIR
from xbrain.jev.assess import Selection, select_items
from xbrain.jev.client import JevClient, JevError, JevResult, Question
from xbrain.jev.dashboard import (
    MediaFiles,
    build_page_data,
    collect_jev_media,
    render_jev_dashboard_html,
)
from xbrain.jev.defaults import tokens_cost_usd
from xbrain.jev.load import JevPairs, load_jev_pairs
from xbrain.jev.lock import pass_lock
from xbrain.jev.models import JevRun
from xbrain.jev.report import report_paths, topics_pass_estimate
from xbrain.jev.run import RunOutcome, run_topics

logger = logging.getLogger(__name__)

#: The only address the server binds. Not an option: the API spends money.
HOST = "127.0.0.1"
#: The header a POST carries the page's token in.
TOKEN_HEADER = "X-Xbrain-Token"
#: The largest request body read, in bytes. A selection is a few ids.
MAX_BODY = 64 * 1024
#: Posts one selection may name by id. The page never needs more.
MAX_IDS = 5000
#: Confirmations kept at once; the oldest is dropped past this (each is single-use anyway).
_CONFIRMS_KEPT = 32
#: Failures listed in a finished job's outcome, like the CLI's `_JEV_FAILURES_SHOWN`.
_FAILURES_SHOWN = 10
#: What the server calls itself in the pass lock, for the message another pass gets.
LOCK_HOLDER = "xbrain jev serve"
#: The selection kinds, in the order a body is checked for them (`_VALUES` checks each value).
_KINDS = ("ids", "topic", "unevaluated", "pair", "band")
#: `post_sets` keys a pair may name: topic confusion, primary confusion, primary agreement.
_PAIR_KINDS = ("cx", "px", "pd")


class ServeError(Exception):
    """A request refused with an HTTP status and a Spanish message for the page."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class Ask:
    """One selection as the page names it: which posts, and whether current ones are re-asked.

    `value` is the ids (a tuple), the topic slug, N, the pair's `(kind, key)` or the band key.
    Frozen and comparable, so a confirmation is bound to exactly the selection it priced.
    """

    kind: str
    value: Any
    force: bool

    def as_json(self) -> dict[str, Any]:
        value = self.value
        if self.kind == "ids":
            value = list(value)
        elif self.kind == "pair":
            value = {"kind": value[0], "key": value[1]}
        return {self.kind: value, "force": self.force}


def _refuse(message: str) -> ServeError:
    return ServeError(400, message)


def _ids_value(value: Any, force: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or len(value) > MAX_IDS:
        raise _refuse(f"`ids` debe ser una lista de 1 a {MAX_IDS} ids")
    if not all(isinstance(item_id, str) and item_id for item_id in value):
        raise _refuse("`ids` debe contener ids de post (texto)")
    return tuple(dict.fromkeys(value))


def _unevaluated_value(value: Any, force: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _refuse("`unevaluated` debe ser un entero >= 1")
    if force:
        raise _refuse(
            "«los N siguientes sin evaluar» no se fuerza: re-evaluar se pide por post, topic o cruce"
        )
    return value


def _pair_value(value: Any, force: bool) -> tuple[str, str]:
    if (
        not isinstance(value, dict)
        or set(value) != {"kind", "key"}
        or value["kind"] not in _PAIR_KINDS
        or not isinstance(value["key"], str)
    ):
        raise _refuse(f"`pair` debe ser {{kind: {'|'.join(_PAIR_KINDS)}, key: texto}}")
    return (value["kind"], value["key"])


def _text_value(value: Any, force: bool) -> str:
    if not isinstance(value, str) or not value:
        raise _refuse("`topic` y `band` deben ser texto")
    return value


#: Each selection kind's value check: the value normalised (hashable, so an `Ask` compares), or
#: a 400 naming what is wrong.
_VALUES: dict[str, Callable[[Any, bool], Any]] = {
    "ids": _ids_value,
    "topic": _text_value,
    "unevaluated": _unevaluated_value,
    "pair": _pair_value,
    "band": _text_value,
}


def parse_ask(body: Any) -> Ask:
    """The selection a request body names — exactly one kind, plus `force` — or a 400."""
    if not isinstance(body, dict):
        raise _refuse("el cuerpo debe ser un objeto JSON")
    unknown = sorted(set(body) - set(_KINDS) - {"force"})
    if unknown:
        raise _refuse(f"campos desconocidos: {', '.join(unknown)}")
    kinds = [kind for kind in _KINDS if kind in body]
    if len(kinds) != 1:
        raise _refuse(f"indica exactamente uno de: {', '.join(_KINDS)}")
    force = body.get("force", False)
    if not isinstance(force, bool):
        raise _refuse("`force` debe ser true o false")
    kind = kinds[0]
    return Ask(kind=kind, value=_VALUES[kind](body[kind], force), force=force)


@dataclass(frozen=True)
class _Confirm:
    """What one estimate priced, for the one `/api/evaluate` that may spend it."""

    ask: Ask
    ids: tuple[str, ...]


@dataclass
class _Job:
    """One background pass and everything the page polls about it. Guarded by `lock`."""

    ask: Ask
    ids: tuple[str, ...]
    max_usd: float
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancel: threading.Event = field(default_factory=threading.Event)
    ready: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    state: str = "starting"
    refusal: str | None = None
    total: int = 0
    done: int = 0
    answered: int = 0
    failed_calls: int = 0
    tokens: int = 0
    tokens_unknown: int = 0
    usd: float = 0.0
    reason: str | None = None
    error: str | None = None
    backup: str | None = None
    log_error: str | None = None
    outcome: dict[str, Any] | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def view(self) -> dict[str, Any]:
        with self.lock:
            view: dict[str, Any] = {
                "state": self.state,
                "ask": self.ask.as_json(),
                "ids": list(self.ids),
                "total": self.total,
                "done": self.done,
                "answered": self.answered,
                "failed_calls": self.failed_calls,
                "tokens": self.tokens,
                "tokens_unknown": self.tokens_unknown,
                "usd": self.usd,
                "max_usd": self.max_usd,
                "estimate": False,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }
            for key in ("reason", "error", "backup", "log_error", "outcome"):
                value = getattr(self, key)
                if value is not None:
                    view[key] = value
            return view


class _Metered:
    """The job's client: counts every answer as it lands (the page's live tokens and $) and
    stops the job once what it has REALLY spent reaches the cap.

    The count is priced by `defaults.tokens_cost_usd`, THE price formula. An answer with no
    token count adds nothing it cannot prove (`tokens_unknown` says how many), so the stop is
    at least as late as the real bill — the cap is enforced on what is known.
    """

    def __init__(self, inner: JevClient, job: _Job) -> None:
        self._inner = inner
        self._job = job

    def ask(self, state: dict[str, str], questions: dict[str, Question]) -> JevResult:
        try:
            result = self._inner.ask(state, questions)
        except Exception:
            with self._job.lock:
                self._job.failed_calls += 1
            raise
        job = self._job
        with job.lock:
            job.answered += 1
            if result.input_tokens is None:
                job.tokens_unknown += 1
            else:
                job.tokens += result.input_tokens
                job.usd += tokens_cost_usd(result.input_tokens, result.provider)
            if job.usd >= job.max_usd and not job.cancel.is_set():
                job.reason = "tope"
                job.cancel.set()
        return result

    def close(self) -> None:
        self._inner.close()


def _topic_posts(slug: str, blob: dict[str, Any], jev: JevPairs) -> list[str]:
    """The posts a topic filter shows on the page: every card whose `slugs` carry it (enrich's
    topics and Jev's rows alike), in the page's order."""
    if slug not in {topic.slug for topic in jev.vocab}:
        raise _refuse(f"topic desconocido: {slug}")
    return [card["id"] for card in blob["posts"] if slug in card["slugs"]]


def _set_posts(sets: dict[str, dict[str, list[str]]], kind: str, key: str, what: str) -> list[str]:
    """The posts behind a pair or band, from the page's `post_sets` (`report.post_sets`)."""
    if key not in sets.get(kind, {}):
        raise _refuse(f"{what} desconocido: {key}")
    return list(sets[kind][key])


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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
        self._job: _Job | None = None

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
        `xbrain generate` mirroring new photos needs a restart to show them.
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
        """Card bodies by id, in the order asked; a 404 naming the ids the corpus lacks."""
        by_id = {card["id"]: card for card in self.blob()["posts"]}
        missing = [item_id for item_id in ids if item_id not in by_id]
        if missing:
            raise ServeError(404, f"posts desconocidos: {', '.join(missing)}")
        return [by_id[item_id] for item_id in ids]

    # ------------------------------------------------------------------ selections

    def _ids(self, ask: Ask, blob: dict[str, Any], jev: JevPairs) -> list[str] | None:
        """The posts `ask` names, in order; `None` for «the next N unevaluated» (a limit)."""
        if ask.kind == "ids":
            return list(ask.value)
        if ask.kind == "unevaluated":
            return None
        if ask.kind == "topic":
            return _topic_posts(ask.value, blob, jev)
        kind, key = ask.value if ask.kind == "pair" else ("bands", ask.value)
        return _set_posts(blob["post_sets"], kind, key, "cruce" if ask.kind == "pair" else "banda")

    def _select(self, ask: Ask, blob: dict[str, Any], jev: JevPairs) -> Selection:
        """`assess.select_items` over the posts `ask` names — the `--dry-run` answer.

        A named set that turns out EMPTY selects nothing: handing `ids=[]` down would mean
        "every post" (`select_items`' own convention), a whole-corpus pass out of an empty topic.
        """
        ids = self._ids(ask, blob, jev)
        if ids is not None and not ids:
            return Selection(items=(), skipped_current=0, skipped_no_evidence=0)
        return select_items(
            jev.store,
            jev.assessments,
            jev.vocab,
            ids=ids,
            limit=ask.value if ask.kind == "unevaluated" else None,
            force=ask.force,
            fallback=self.cfg.jev_fallback_option,
            char_limit=self.cfg.jev_state_char_limit,
        )

    def estimate(self, body: Any) -> dict[str, Any]:
        """How many posts `body`'s selection would ask, what that would cost by the page's mean,
        and — when it may run — the single-use token `/api/evaluate` needs."""
        ask = parse_ask(body)
        _, blob, jev = self._built()
        try:
            selection = self._select(ask, blob, jev)
        except JevError as exc:
            raise _refuse(str(exc)) from exc
        per_post = blob["cost"]["per_post"]
        ids = tuple(item.id for item in selection.items)
        priced = topics_pass_estimate(per_post, len(ids))
        refusal = self._refusal(len(ids), priced["usd"], selection)
        confirm_token = None
        if refusal is None:
            confirm_token = secrets.token_urlsafe(16)
            with self._state:
                self._confirms[confirm_token] = _Confirm(ask=ask, ids=ids)
                while len(self._confirms) > _CONFIRMS_KEPT:
                    self._confirms.pop(next(iter(self._confirms)))
        return {
            "ask": ask.as_json(),
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
        }

    def _refusal(self, posts: int, usd: float | None, selection: Selection) -> str | None:
        """Why this selection may not run from the page, or `None` when it may."""
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
        if usd > cap:
            return (
                f"la estimación (~{usd:.4f} $) pasa del tope por trabajo "
                f"([jev].serve_max_usd = {cap} $): elige menos posts, o sube el tope en "
                "config.toml, o lánzalo desde la terminal con `xbrain jev topics`"
            )
        return None

    # ------------------------------------------------------------------ the job

    def evaluate(self, body: Any) -> dict[str, Any]:
        """Start the job an estimate confirmed. Returns once it has the lock and has checked
        the selection is still the one priced; refuses with 409 otherwise."""
        if not isinstance(body, dict) or not isinstance(body.get("confirm_token"), str):
            raise _refuse("falta confirm_token: pide antes una estimación (/api/estimate)")
        body = dict(body)
        token = body.pop("confirm_token")
        ask = parse_ask(body)
        with self._state:
            if self._job is not None and self._job.state in ("starting", "running"):
                raise ServeError(409, "ya hay un trabajo en curso: espera a que termine")
            confirm = self._confirms.get(token)
            if confirm is None or confirm.ask != ask:
                raise ServeError(
                    409,
                    "esta confirmación no corresponde a esta selección, o ya se usó: "
                    "vuelve a estimar",
                )
            del self._confirms[token]
            previous = self._job
            job = _Job(ask=ask, ids=confirm.ids, max_usd=self.cfg.jev_serve_max_usd)
            job.thread = threading.Thread(target=self._run, args=(job,), name="jev-serve-job")
            self._job = job
        job.thread.start()
        job.ready.wait()
        if job.refusal is not None:
            with self._state:
                self._job = previous
            raise ServeError(409, job.refusal)
        return job.view()

    def _run(self, job: _Job) -> None:
        """The job's thread: lock, reload, re-check, run the ONE loop, record the outcome."""
        try:
            with pass_lock(self.cfg.jev_lock_path, LOCK_HOLDER):
                _, blob, jev = self._built()
                selection = self._select(job.ask, blob, jev)
                if tuple(item.id for item in selection.items) != job.ids:
                    job.refusal = (
                        "la selección cambió desde la estimación (otra pasada o un cambio en "
                        "los datos): vuelve a estimar"
                    )
                    return
                with job.lock:
                    job.state, job.total, job.started_at = "running", len(job.ids), _now_iso()
                job.ready.set()
                outcome = run_topics(
                    self.cfg,
                    selection,
                    dict(jev.assessments),
                    jev.vocab,
                    lambda: _Metered(self._make_client(), job),
                    on_backup=lambda path: _set(job, backup=str(path)),
                    on_progress=lambda done, total: _set(job, done=done),
                    on_logged=lambda path, line, error: _set(
                        job, log_error=None if error is None else f"{error}; la línea: {line}"
                    ),
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
                with job.lock:
                    job.state, job.error, job.finished_at = "error", message, _now_iso()
        finally:
            job.ready.set()

    def job_view(self) -> dict[str, Any]:
        """The current (or last) job, as `/api/job` sends it; `{"state": "idle"}` before any."""
        with self._state:
            job = self._job
        return {"state": "idle"} if job is None else job.view()

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current job's thread has ended (for tests and `stop`)."""
        with self._state:
            job = self._job
        if job is not None and job.thread is not None:
            job.thread.join(timeout)

    def stop(self) -> None:
        """Cancel the running job at its next call and wait for it to save and log."""
        with self._state:
            job = self._job
        if job is not None:
            with job.lock:
                if job.state in ("starting", "running") and not job.cancel.is_set():
                    job.reason = "servidor parado"
                    job.cancel.set()
        self.wait()


def _set(job: _Job, **fields: Any) -> None:
    with job.lock:
        for key, value in fields.items():
            setattr(job, key, value)


def _finish(job: _Job, outcome: RunOutcome) -> None:
    logged: JevRun | None = outcome.logged
    with job.lock:
        job.state = "interrupted" if outcome.interrupted else "done"
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


# ---------------------------------------------------------------------- HTTP


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    service: JevService


class _Handler(BaseHTTPRequestHandler):
    """Routing and the request guards; every decision is `JevService`'s."""

    server: _Server
    server_version = "xbrain-jev"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("jev serve: " + format, *args)

    # -------------------------------------------------------------- guards

    def _allowed_hosts(self) -> set[str]:
        port = self.server.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _guard(self, *, post: bool) -> None:
        """Refuse a foreign Host (every route), a foreign or missing Origin (POST; GET only
        when it sends one) and a missing or wrong token (POST)."""
        hosts = self._allowed_hosts()
        if self.headers.get("Host") not in hosts:
            raise ServeError(403, "Host no permitido: este servidor solo atiende a 127.0.0.1")
        origin = self.headers.get("Origin")
        if (post or origin is not None) and origin not in {f"http://{h}" for h in hosts}:
            raise ServeError(403, "Origin no permitido: solo la página de este servidor")
        if post:
            sent = self.headers.get(TOKEN_HEADER, "")
            if not hmac.compare_digest(sent.encode(), self.server.service.token.encode()):
                raise ServeError(403, "falta el token de este servidor, o no es el suyo")

    def _body(self) -> Any:
        kind = self.headers.get("Content-Type", "")
        if kind.split(";")[0].strip() != "application/json":
            raise ServeError(415, "el cuerpo debe ser JSON (Content-Type: application/json)")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise ServeError(411, "falta Content-Length") from None
        if length > MAX_BODY:
            if length <= 16 * MAX_BODY:
                self.rfile.read(length)  # drained, so the refusal reaches the client
            self.close_connection = True
            raise ServeError(413, f"cuerpo demasiado grande (máximo {MAX_BODY} bytes)")
        try:
            return json.loads(self.rfile.read(length) or b"null")
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise _refuse(f"JSON ilegible: {exc}") from exc

    # -------------------------------------------------------------- responses

    def _send(self, status: int, body: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _handle(self, route: Callable[[], None]) -> None:
        try:
            route()
        except ServeError as exc:
            self._json(exc.status, {"error": exc.message})
        except JevError as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            logger.exception("jev serve: %s %s falló", self.command, self.path)
            self._json(500, {"error": f"error interno: {type(exc).__name__}: {exc}"})

    # -------------------------------------------------------------- routes

    def do_GET(self) -> None:  # noqa: N802 — the stdlib's name
        self._handle(self._get)

    def do_POST(self) -> None:  # noqa: N802 — the stdlib's name
        self._handle(self._post)

    def _get(self) -> None:
        self._guard(post=False)
        url = urlsplit(self.path)
        service = self.server.service
        if url.path in ("/", "/jev.html"):
            self._send(200, service.page_html().encode("utf-8"), "text/html; charset=utf-8")
        elif url.path == "/api/data":
            self._json(200, service.blob())
        elif url.path == "/api/cards":
            ids = [i for raw in parse_qs(url.query).get("ids", []) for i in raw.split(",") if i]
            self._json(200, {"cards": service.cards(ids)})
        elif url.path == "/api/job":
            self._json(200, service.job_view())
        elif url.path.startswith(f"/{VAULT_MEDIA_SUBDIR}/"):
            self._media(url.path)
        else:
            raise ServeError(404, "no existe")

    def _post(self) -> None:
        self._guard(post=True)
        body = self._body()
        path = urlsplit(self.path).path
        service = self.server.service
        if path == "/api/estimate":
            self._json(200, service.estimate(body))
        elif path == "/api/evaluate":
            self._json(202, service.evaluate(body))
        else:
            raise ServeError(404, "no existe")

    def _media(self, raw_path: str) -> None:
        """A file under `<output_dir>/_media/`, and nothing outside it however it is spelled:
        the path is decoded, resolved (symlinks too) and must stay inside the folder."""
        output_dir = self.server.service.cfg.output_dir
        root = (output_dir / VAULT_MEDIA_SUBDIR).resolve()
        target = (output_dir / unquote(raw_path).lstrip("/")).resolve()
        if not target.is_relative_to(root) or target == root or not target.is_file():
            raise ServeError(404, "no existe")
        kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), kind)


def make_server(service: JevService, port: int) -> ThreadingHTTPServer:
    """The HTTP server for `service` on 127.0.0.1:`port` (0 = any free port). Not started."""
    server = _Server((HOST, port), _Handler)
    server.service = service
    return server


def serve_until_interrupted(server: ThreadingHTTPServer, service: JevService) -> int:
    """Serve until Ctrl-C; then stop accepting, let the job save and log, and return 130.

    The job runs in a NON-daemon thread, so even a second Ctrl-C while `stop` waits leaves the
    interpreter waiting for the job's interrupt path to write what was paid for.
    """
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
        service.stop()
    return 0
