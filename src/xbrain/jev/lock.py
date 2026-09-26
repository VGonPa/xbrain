"""`data/jev/.lock`: one paid pass at a time over the side-car, across processes.

WHY A LOCK. A pass loads `topics.json`, asks, and saves its whole map back (the file is
rewritten wholesale — `store.save_assessments`). Two passes at once — `xbrain jev topics` in a
terminal and the local server's job (`xbrain jev serve`), or two terminals — each load the
same file, each save their own map, and the second save drops every record the first one paid
for. The lock covers LOAD → SAVE, so it is taken by the CALLER, before it reads the side-car;
`run.run_topics` refuses to start without it (`held`), so a new caller cannot forget it.

`flock`, not a pid file: the kernel releases it when the process dies, however it dies, so
there is no stale lock to clean by hand after a crash or a `kill -9`. It is per open file, so
a second holder in the SAME process is refused too (the server's job and a stray second job).
The file's text — who holds it, which pid, since when — is only for the refusal message; the
lock is the `flock`, never the text.
"""

from __future__ import annotations

import fcntl
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from xbrain.jev.client import JevError

#: The lock files this process holds, by resolved path. `held` reads it; the kernel's lock is
#: the real exclusion, this only lets `run_topics` check its caller took it.
_HELD: set[Path] = set()
_GUARD = threading.Lock()


def held(path: Path) -> bool:
    """Whether THIS process holds the pass lock at `path` right now."""
    with _GUARD:
        return path.resolve() in _HELD


def _holder_text(fd: int) -> str:
    """What the current holder wrote into the file, or a plain fallback when it is unreadable."""
    try:
        text = os.pread(fd, 512, 0).decode("utf-8", "replace").strip()
    except OSError:
        text = ""
    return text or "otro proceso"


@contextmanager
def pass_lock(path: Path, holder: str) -> Iterator[None]:
    """Hold the pass lock at `path` for the block, or raise `JevError` naming who holds it.

    `holder` says which command holds it (`jev topics`, `jev serve`), for the message another
    pass gets. Never waits: a pass that would queue behind another would be computed against
    a side-car that is about to change, so it is refused and re-run instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise JevError(
                f"otra pasada de Jev está en curso ({_holder_text(fd)}): espera a que termine "
                f"y vuelve a lanzarla. Candado: {path}"
            ) from None
        since = datetime.now(timezone.utc).isoformat(timespec="seconds")
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"{holder} · pid {os.getpid()} · desde {since}\n".encode(), 0)
        key = path.resolve()
        with _GUARD:
            _HELD.add(key)
        try:
            yield
        finally:
            with _GUARD:
                _HELD.discard(key)
            # Emptied before the release, so a reader never quotes a holder that has gone.
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
