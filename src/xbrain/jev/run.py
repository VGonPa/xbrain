"""One paid `jev topics` pass: back up, ask, checkpoint, save, log, release — and nothing else.

THE ONE RUN LOOP. `xbrain jev topics` calls it today; a local server that lets the page
request evaluations calls the SAME function, so both get the same backup rule, the same
checkpointing, the same run-log line and the same teardown. A second loop would be a second
place where a paid record can be lost.

IT PRINTS NOTHING. A server has no terminal, so the pass reports through hooks (called at the
moments the CLI needs to speak — the summary BEFORE the save, the log line AFTER it) and
through its return value. An interrupt is RETURNED (`RunOutcome.interrupted`), not raised: the
shell's exit 130 is the CLI's business, not the loop's.

What it raises: the all-failed `JevError` from `run_assessments` (after logging the pass), a
side-car that cannot be written (`JevError` naming the paid count), a backup that cannot be
made (before any client exists, so before any cost), and whatever `make_client` raises (a
missing key). A run-log line that cannot be written is NEVER raised: it goes to `on_logged`
with its error, because an exception from a `finally` would replace the pass's real verdict.
"""

from __future__ import annotations

import logging
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from xbrain.config import Config
from xbrain.jev.assess import RunResult, Selection, run_assessments
from xbrain.jev.client import CountingJevClient, JevClient, JevError, SeamCounts
from xbrain.jev.defaults import plural
from xbrain.jev.models import JevRun, TopicAssessment
from xbrain.jev.store import append_run, save_assessments
from xbrain.models import Topic

logger = logging.getLogger(__name__)

#: Records banked between durable writes of the side-car. The dict in memory is not
#: durability: a SIGTERM, a closed terminal, an OOM kill or any exception that is not
#: `KeyboardInterrupt` skips the checkpoint handler entirely and every paid record in it is
#: gone. Flushing every N bounds that loss to at most N-1 records, at the price of one
#: rewrite of the side-car per N items — the repo's standard bargain for paid batch work
#: (`media`, `describe` and `refetch` all write between units rather than at the end).
CHECKPOINT_EVERY = 25


@dataclass(frozen=True)
class RunOutcome:
    """What one pass did. `assessed` is what THIS pass banked, never the whole side-car.

    `failed` is empty on an interrupted pass: `run_assessments` discards its own collection
    when interrupted, and the run log's `requests - ok - failed` says how many were in flight.
    `logged` is the run-log line written, or `None` when nothing was sent or the write failed.
    """

    assessed: tuple[TopicAssessment, ...]
    failed: tuple[tuple[str, str], ...]
    stored: int
    interrupted: bool
    logged: JevRun | None


def _save_side_car(assessments: dict[str, TopicAssessment], path: Path, *, paid: int) -> None:
    """Write the side-car, and NAME THE BILL when it cannot be written.

    `Error: [Errno 28] No space left on device` is true and useless: it is about a
    filesystem, and nothing connects it to "you were just billed for 2,998 assessments and
    none of them was written". The paid count is the part the operator has to act on, and it
    is the reason this is not simply `save_assessments`.
    """
    try:
        save_assessments(assessments, path)
    except OSError as exc:
        lost = plural(paid, "evaluación pagada", "evaluaciones pagadas")
        raise JevError(f"no se pudo guardar {path} ({lost} sin guardar): {exc}") from exc


def back_up_before_forced_overwrite(path: Path, forced: int) -> Path | None:
    """Copy the side-car beside itself under a UTC stamp and return the copy. Or do nothing.

    THE SIDE-CAR'S OWN REVERSIBILITY. ARCHITECTURE invariant 8 says every command that
    overwrites a `data/` artifact leaves a way back, and every other one gets that from
    `snapshot create`. This file does not: `data/jev/topics.json` is deliberately OUTSIDE
    `snapshot._ARTIFACTS` (a snapshot covers the corpus, not a second opinion about it), and
    `data/` is gitignored in full. So before this, `--force` over thousands of paid records
    had no undo anywhere — not git, not `snapshot restore`, not a copy.

    Stamped rather than fixed (`topics.bak`) because the second forced run would otherwise
    overwrite the first one's only copy: the loss the backup exists to prevent, one run later.
    The stamp is `snapshot_create`'s, to the millisecond and in UTC, so the two reversibility
    mechanisms sort and read alike and scripted runs in the same second do not collide.

    NEVER PRUNED — by anything, ever. A cleanup rule would have to decide which paid copy is
    expendable, and the one an operator wants is the one from before the run they regret,
    which is not knowable here. `docs/jev.md` tells them to delete them by hand.

    `forced`, not the `--force` FLAG: a forced run that re-asks nothing current overwrites
    nothing, and a `.bak` per ordinary run is how an operator learns to ignore them.
    """
    if not forced or not path.exists():
        return None
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H-%M-%S-") + f"{now.microsecond // 1000:03d}Z"
    backup = path.with_name(f"{path.stem}.{stamp}.bak")
    try:
        shutil.copy2(path, backup)
    except OSError as exc:
        # Same argument as `_save_side_car`: the errno is about a filesystem and says nothing
        # about what is at stake. Raised BEFORE the client is built, so a run that cannot be
        # made reversible has not yet cost anything.
        raise JevError(
            f"no se pudo copiar {path} a {backup} ({exc}); "
            f"--force sobrescribiría evaluaciones pagadas sin copia"
        ) from exc
    return backup


def _release(client: JevClient) -> None:
    """Release the client's pool without ever becoming the pass's verdict.

    `close()` reaches the vendor's HTTP stack, which is the layer `typesafe.py` wraps
    everywhere else. A teardown fault is NOISE: the answers are already paid for, already in
    memory and — by the time this runs — already written. Letting it out would (a) replace an
    all-failed `JevError` with a socket errno, which is Python's default `finally` behaviour,
    and (b) on the success path, turn a completed pass into an error.

    `original_error` is read BEFORE the `try`: inside an `except` block `sys.exc_info()`
    reports the exception being handled here, not the one that was already travelling.

    On the interrupt path this can close the transport while a worker is still in flight —
    `run_assessments` shuts its pool down with `wait=False`, so a call already issued is
    cancelled rather than awaited. That worker's exception lands in a future nobody reads,
    which makes it noise and not a lost record: every record that completed was handed to
    the checkpoint and written before this runs.
    """
    original_error = sys.exc_info()[1]
    try:
        client.close()
    except Exception as exc:
        note = " (el error anterior sigue propagándose)" if original_error is not None else ""
        logger.warning(
            "cerrar el cliente Jev falló (%s: %s); el run no se ve afectado%s",
            type(exc).__name__,
            exc,
            note,
        )


def _now() -> datetime:
    """The wall clock, in UTC — one seam, so a test can make it step backwards."""
    return datetime.now(timezone.utc)


def run_record(
    counts: SeamCounts,
    *,
    kept: int,
    started_at: datetime,
    finished_at: datetime,
    interrupted: bool,
) -> JevRun:
    """The `runs.jsonl` line for one pass, from what the seam saw and what was kept.

    `kept` is what was BANKED into the side-car. Everything else that came back is either a
    failure (`raised`, plus answers xbrain refused) or — on an interrupted pass only —
    `unsaved`: answered but never drained. An interrupt cannot tell a refused answer that
    was not yet drained from a good one, and neither was kept, so both land in `unsaved`.

    `finished_at` is clamped to `started_at`: a wall clock that steps back (NTP, a laptop
    waking) must not make the record invalid — this is built inside a `finally`.
    """
    answered_not_kept = counts.answered - kept
    return JevRun(
        started_at=started_at,
        finished_at=max(finished_at, started_at),
        models=list(counts.models),
        requests=counts.sent,
        ok=kept,
        failed=counts.raised + (0 if interrupted else answered_not_kept),
        unsaved=answered_not_kept if interrupted else 0,
        input_tokens_by_provider=counts.input_tokens_by_provider,
        input_tokens=sum(counts.input_tokens_by_provider.values()),
        input_tokens_unknown=counts.input_tokens_unknown,
        interrupted=interrupted,
    )


#: What `on_logged` receives: the log path, the line (the JSON record, or the raw seam counts
#: when no valid record could be built), and the error that kept it out of the file, if any.
LoggedHook = Callable[[Path, str, BaseException | None], None]


def _log_pass(
    path: Path,
    client: CountingJevClient,
    kept: int,
    started_at: datetime,
    *,
    interrupted: bool,
    on_logged: LoggedHook | None,
) -> JevRun | None:
    """Append the pass to the run log. NEVER raises — it runs inside `run_topics`' `finally`.

    Nothing is written when nothing was SENT (a Ctrl-C before the first call): the log is a
    history of requests made, not of invocations.

    An exception out of a `finally` REPLACES the one in flight, so anything here — a record
    that fails validation, a full disk, a hook whose echo hits a closed pipe — would turn an
    interrupt or the all-failed `JevError` into a different error, or a paid, saved pass into
    a failure, and skip the client's release. So every step is caught. The line is still
    handed to `on_logged` with the error (the CLI prints it for the operator to append by
    hand), and if the hook itself fails the line goes to the log as a warning.
    """
    counts = client.snapshot()
    if not counts.sent:
        return None
    line = repr(counts)
    error: BaseException | None = None
    run: JevRun | None = None
    try:
        run = run_record(
            counts, kept=kept, started_at=started_at, finished_at=_now(), interrupted=interrupted
        )
        line = run.model_dump_json()
        append_run(run, path)
    except Exception as exc:
        error = exc
    if on_logged is not None:
        try:
            on_logged(path, line, error)
        except Exception as exc:
            logger.warning(
                "no se pudo informar del registro de pasadas (%s: %s); la línea: %s",
                type(exc).__name__,
                exc,
                line,
            )
    elif error is not None:
        logger.warning("no se pudo escribir el registro de pasadas (%s); la línea: %s", error, line)
    return None if error is not None else run


def run_topics(
    cfg: Config,
    selection: Selection,
    assessments: dict[str, TopicAssessment],
    vocab: list[Topic],
    make_client: Callable[[], JevClient],
    *,
    on_backup: Callable[[Path], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_summary: Callable[[RunResult], None] | None = None,
    on_interrupted: Callable[[tuple[TopicAssessment, ...], int], None] | None = None,
    on_logged: LoggedHook | None = None,
) -> RunOutcome:
    """Ask Jev about `selection.items`, keeping everything that was paid for.

    `assessments` is the side-car in memory — the map `selection` was computed against — and
    it is UPDATED IN PLACE as records land, so a caller that keeps running holds the new
    state without reloading the file. `make_client` is called only after the backup, so a
    pass that cannot be made reversible never builds a client (and never needs a key).

    Hooks, each optional and each called from the calling thread:

    * `on_backup(path)` — the side-car was copied before a forced overwrite.
    * `on_progress(done, total)` — after each item comes back.
    * `on_summary(result)` — the pass finished; called BEFORE the side-car is written, so the
      counters reach the operator even when the save then fails.
    * `on_interrupted(banked, stored)` — Ctrl-C landed; called before the banked records are
      saved. `banked` may be empty (nothing to save, and nothing is written).
    * `on_logged(path, line, error)` — the run-log line, after every other step, on EVERY
      exit path of a pass that sent a request; `error` is set when it could not be appended
      (then `line` is what to append by hand). A hook that raises is logged, never raised.

    There is no `force` argument: which items are re-asked, current ones included, is
    `selection`'s decision (`assess.select_items(force=...)`), already made.
    """
    if not selection.items:
        return RunOutcome(
            assessed=(), failed=(), stored=len(assessments), interrupted=False, logged=None
        )
    # BEFORE the client, so a backup that cannot be written stops a pass that has not yet
    # been billed — and before the first checkpoint, so the copy is the file as it was.
    backup = back_up_before_forced_overwrite(cfg.jev_topics_path, selection.forced)
    if backup is not None and on_backup is not None:
        on_backup(backup)

    # The records THIS pass paid for, in order of arrival. `assessments` also holds every
    # earlier pass's work, so reporting its length as the rescue would tell an operator who
    # banked 3 records into a side-car of 500 that they rescued 503.
    banked: dict[str, TopicAssessment] = {}

    def _checkpoint(assessment: TopicAssessment) -> None:
        """Bank each record as it lands, and flush to disk every `CHECKPOINT_EVERY`.

        Through `_save_side_car`, not `save_assessments`: ONE failure must not produce two
        messages depending on when the disk filled. `assess._deliver_result` swallows whatever
        this raises (a broken checkpoint must never discard the record it was called to save),
        so the log line is the only surface there is.
        """
        assessments[assessment.item_id] = assessment
        banked[assessment.item_id] = assessment
        if len(banked) % CHECKPOINT_EVERY == 0:
            _save_side_car(assessments, cfg.jev_topics_path, paid=len(banked))

    # Wrapped so the run log can say how many calls were SENT, which nothing else observes.
    client = CountingJevClient(make_client())
    started_at = _now()
    interrupted = False
    logged: JevRun | None = None
    try:
        result = run_assessments(
            list(selection.items),
            vocab,
            client,
            fallback=cfg.jev_fallback_option,
            char_limit=cfg.jev_state_char_limit,
            concurrency=cfg.jev_concurrency,
            on_progress=on_progress,
            on_result=_checkpoint,
        )
    except KeyboardInterrupt:
        interrupted = True
        # Summary first, then persist — the tally can never fail, so a save that does never
        # suppresses it.
        if on_interrupted is not None:
            on_interrupted(tuple(banked.values()), len(assessments))
        if banked:
            # With nothing banked, saving would write `assessments` UNCHANGED — for a first
            # run that is `{}` over the side-car.
            _save_side_car(assessments, cfg.jev_topics_path, paid=len(banked))
        failed: tuple[tuple[str, str], ...] = ()
    else:
        # The checkpoint has already put every one of these in `assessments`; the merge stays
        # here so the NORMAL path does not depend on a hook whose failures are swallowed.
        for assessment in result.assessed:
            assessments[assessment.item_id] = assessment
        if on_summary is not None:
            on_summary(result)
        _save_side_car(assessments, cfg.jev_topics_path, paid=len(result.assessed))
        failed = result.failed
    finally:
        # EVERY exit path of a pass that sent something lands here: success, a partial
        # failure, the all-failed `JevError` (a 402 on every call is still requests made),
        # a side-car that could not be written, and Ctrl-C.
        logged = _log_pass(
            cfg.jev_runs_path,
            client,
            len(banked),
            started_at,
            interrupted=interrupted,
            on_logged=on_logged,
        )
        # LAST, and guarded: releasing the pool is cleanup, never the pass's verdict.
        _release(client)
    return RunOutcome(
        assessed=tuple(sorted(banked.values(), key=lambda a: a.item_id)),
        failed=failed,
        stored=len(assessments),
        interrupted=interrupted,
        logged=logged,
    )
