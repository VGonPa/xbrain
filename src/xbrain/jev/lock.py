"""`data/jev/.lock`: one paid pass at a time over the side-car, across processes.

WHY A LOCK. A pass loads `topics.json`, asks, and saves its whole map back (the file is
rewritten wholesale — `store.save_assessments`). Two passes at once — `xbrain jev topics` in a
terminal and a job of the local server (`xbrain jev serve`), or two terminals — each load the
same file, each save their own map, and the second save drops every record the first one paid
for. The lock covers LOAD → SAVE, so it is taken by the CALLER, before it reads the side-car;
`run.run_topics` requires the `PassLock` handle it yields and refuses a released one or one
for another file, so a new caller cannot forget it.

`flock`, not a pid file: the kernel releases it when the process dies, however it dies, so
there is no stale lock to clean by hand after a crash or a `kill -9`. It is per open file, so
a second holder in the SAME process is refused too (the server's job and a stray second job).
The file's text — who holds it, which pid, since when — is only for the refusal message; the
lock is the `flock`, never the text. A process that died holding it leaves its line behind;
the next holder overwrites it, and a line with nothing holding the lock means nothing.
"""

from __future__ import annotations

import fcntl
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from xbrain.jev.client import JevError

logger = logging.getLogger(__name__)


class PassLockBusy(JevError):
    """Another pass holds the lock. Transient, and the CLI exits 75 (EX_TEMPFAIL) on it:
    nothing is wrong except the timing, and running again later is the whole remedy."""


@dataclass
class PassLock:
    """Proof of holding the pass lock at `path`: what `run_topics` asks for. `held` turns
    False when the `with` block that yielded it ends."""

    path: Path
    holder: str
    held: bool = True

    def covers(self, path: Path) -> bool:
        """Held, and for `path` — the check `run_topics` makes before any cost."""
        return self.held and self.path.resolve() == path.resolve()


def _holder_text(fd: int) -> str:
    """What the current holder wrote into the file, or a plain fallback when it is unreadable."""
    try:
        text = os.pread(fd, 512, 0).decode("utf-8", "replace").strip()
    except OSError:
        text = ""
    return text or "otro proceso"


def _open(path: Path) -> int:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        raise JevError(f"no se pudo abrir el candado de pasadas {path}: {exc}") from exc


def _release(fd: int, path: Path) -> None:
    """Empty the file, unlock, close — each step guarded: a release that raised would replace
    the error already travelling out of the pass (and the kernel frees the lock at close or
    exit anyway)."""
    for step, action in (
        ("vaciar", lambda: os.ftruncate(fd, 0)),
        ("soltar", lambda: fcntl.flock(fd, fcntl.LOCK_UN)),
        ("cerrar", lambda: os.close(fd)),
    ):
        try:
            action()
        except OSError as exc:
            logger.warning("no se pudo %s el candado %s: %s", step, path, exc)


@contextmanager
def pass_lock(path: Path, holder: str) -> Iterator[PassLock]:
    """Hold the pass lock at `path` for the block and yield its handle, or raise:
    `PassLockBusy` naming who holds it, `JevError` naming the path when it cannot be taken.

    `holder` says which command holds it (`xbrain jev topics`, `xbrain jev serve`), for the
    message another pass gets. Never waits: a pass that would queue behind another would be
    computed against a side-car that is about to change, so it is refused and re-run instead.
    """
    fd = _open(path)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        who = _holder_text(fd)
        os.close(fd)
        raise PassLockBusy(
            f"otra pasada de Jev está en curso ({who}): espera a que termine "
            f"y vuelve a lanzarla. Candado: {path}"
        ) from None
    except OSError as exc:
        os.close(fd)
        raise JevError(f"no se pudo tomar el candado de pasadas {path}: {exc}") from exc
    handle = PassLock(path=path, holder=holder)
    try:
        since = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            os.ftruncate(fd, 0)
            os.pwrite(fd, f"{holder} · pid {os.getpid()} · desde {since}\n".encode(), 0)
        except OSError as exc:
            raise JevError(f"no se pudo escribir el candado de pasadas {path}: {exc}") from exc
        yield handle
    finally:
        handle.held = False
        _release(fd, path)
