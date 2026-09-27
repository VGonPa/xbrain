"""`data/jev/topics.json`: one `TopicAssessment` per item id.

A SIDE-CAR. `items.json` is never opened for writing by anything under `xbrain.jev`: the
topic assessment is a second opinion about an item, not a fact about it, and writing it
onto the item would make the enrich assignment and the judge's answer indistinguishable
the moment a report wanted to compare them. Keeping them in separate files is what lets
`xbrain jev` be re-run, thrown away or ignored without touching the corpus.

The file is rewritten WHOLESALE on every save, which is why `TopicAssessment` forbids
unknown fields: a record a newer writer added a field to would otherwise be silently
narrowed by an older one on the next run.

ONE ASSESSMENT PER ITEM. The key is the `item_id` and nothing else, so a run by a SECOND
judge — another provider, another model — does not sit beside the first, it OVERWRITES it,
paid record for paid record. `TopicAssessment` carries `provider`/`model` so a stored record
says who answered it, not so two answers can be held at once; and because the contract
excludes the judge, re-pointing `[jev].model` leaves every record current, so only `--force`
re-asks — which is the path that destroys the old work. Holding a panel means re-keying this
file by `(item_id, provider, model)`, and that is a follow-up, not something the current
shape supports.

THIS FILE COSTS MONEY TO REGENERATE and the usual undo does not reach it. It is NOT
snapshotted with the rest of `data/` (`snapshot create` covers the corpus, not the
side-car) and `data/` is gitignored in full, so there is no `git checkout` and no
`snapshot restore` back to a good copy. Two consequences worth knowing before touching it:

* `--force` overwrites paid records, so the CLI copies this file to
  `topics.<UTC stamp>.bak` beside it before a run that re-asks a current one — the
  side-car's OWN reversibility, standing in for the snapshot it does not get. Those copies
  are never pruned, and nothing in THIS module writes them: the backup belongs to the
  pass that decided to overwrite (`run.back_up_before_forced_overwrite`), not to the writer.
* A corrupt file is repaired by hand or paid for again — which is why `load_assessments`
  refuses one instead of quietly starting from `{}`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from xbrain.jev.client import JevError
from xbrain.jev.models import (
    AskAssessment,
    AskFile,
    AskIndex,
    JevRun,
    TopicAssessment,
)
from xbrain.store import _atomic_write

if TYPE_CHECKING:
    from xbrain.jev.ask import AskQuery

#: The history file inside the asks directory; every other `*.json` there is one query.
ASK_INDEX = "index.json"


def load_assessments(path: Path) -> dict[str, TopicAssessment]:
    """The assessments keyed by item id; an empty dict if the file does not exist.

    A missing file is the FIRST RUN, not a fault — the side-car is created by the first
    save. A file that exists but is unreadable is NEVER swallowed: returning `{}` would
    report "0 evaluaciones guardadas" over a full side-car and re-ask, and re-pay for, the
    whole corpus, then overwrite whatever was still readable with only the new records.

    Every way the file can be unusable — unparseable JSON, a top level that is not an
    object, a record this build's validator refuses — raises `JevError` naming the PATH.
    `jev topics` loads three files back to back (`items.json`, `vocab.yaml`, the side-car),
    and a bare `Expecting value: line 1 column 1` sends the operator to none of them.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise JevError(f"{path}: side-car ilegible (el nivel superior no es un objeto JSON)")
        return {item_id: TopicAssessment.model_validate(data) for item_id, data in raw.items()}
    except (json.JSONDecodeError, ValidationError) as exc:
        raise JevError(f"{path}: side-car ilegible ({exc})") from exc


def save_assessments(assessments: dict[str, TopicAssessment], path: Path) -> None:
    """Persist as pretty, sorted, UTF-8 JSON (atomic write, like `store.save_store`).

    Sorted and pretty for DIFFABILITY BY HAND, which is not about git: `data/` is gitignored
    in full (`.gitignore`: `data/*`), so neither this file nor `items.json` is ever in
    version control. The whole file is rewritten on every run, so without a stable key order
    an unchanged corpus would not re-dump byte-identically and a run that changed two records
    would show up as a reordering of everything — making `diff` between two runs, or against
    a manual copy, useless. That copy is also the only backup there is.

    Atomic because a run is paid for — a partial file from an interrupted write would lose
    every assessment, including the ones already billed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        item_id: assessment.model_dump(mode="json")
        for item_id, assessment in sorted(assessments.items())
    }
    _atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def load_runs(path: Path) -> list[JevRun]:
    """Every logged pass, in file (= chronological) order; `[]` when the log does not exist.

    Same stance as `load_assessments`: a line that does not parse or validate is REFUSED with
    the path and its line number, never skipped. Each line is paid history, and a report that
    silently dropped one would under-quote the bill. Blank lines are not records.
    """
    if not path.exists():
        return []
    runs: list[JevRun] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            runs.append(JevRun.model_validate_json(line))
        except ValidationError as exc:
            raise JevError(
                f"{path}: registro de pasadas ilegible en la línea {number} ({exc})"
            ) from exc
    return runs


def append_run(run: JevRun, path: Path) -> None:
    """Append one line to the run log: never rewrite, never glue, never leave a fragment.

    APPEND, never rewrite: earlier lines are paid history and this call must not be able to
    touch them. Two guards make a single writer's append safe to trust:

    * If the file does not end in a newline — a crash or a full disk tore the last write —
      a newline goes first, so this record starts on its own line. Glued onto the fragment
      it would be refused with it, and a paid pass would be hidden inside a broken line.
    * If the write fails, is short, or the flush to disk fails, the file is truncated back to its size
      before this call, then the error is raised: the log stays exactly as it was.

    Like the side-car, the log is not snapshotted and lives under the gitignored `data/`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (run.model_dump_json() + "\n").encode("utf-8")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        size = os.fstat(fd).st_size
        if size and os.pread(fd, 1, size - 1) != b"\n":
            line = b"\n" + line
        try:
            written = os.write(fd, line)
            if written != len(line):
                # A short write raises nothing on its own; a record cut short IS a torn line.
                raise OSError(f"short write: {written} of {len(line)} bytes to {path}")
            os.fsync(fd)
        except OSError:
            os.ftruncate(fd, size)
            raise
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- `jev ask`


def _dump(payload: object, path: Path) -> None:
    """Pretty, sorted, atomic — the side-car's own reasons (`save_assessments`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def load_asks(path: Path, query: AskQuery) -> dict[str, AskAssessment]:
    """One query's answers keyed by post id; `{}` when the query was never asked.

    Same stance as `load_assessments`: a file that exists but cannot be read is REFUSED with
    its path, never read as empty — that would re-pay every answer in it and then overwrite
    them. A file whose `query` is another query is refused too: its answers were paid for a
    different question, and reading them here would present them as answers to this one.
    """
    if not path.exists():
        return {}
    try:
        stored = AskFile.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise JevError(f"{path}: consulta guardada ilegible ({exc})") from exc
    if stored.query != query.text:
        raise JevError(
            f"{path} guarda la consulta {stored.query!r}, no {query.text!r}: "
            "el fichero no corresponde a su nombre; muévelo aparte y vuelve a lanzar"
        )
    return dict(stored.assessments)


def save_asks(query: AskQuery, records: dict[str, AskAssessment], path: Path) -> None:
    """Write one query's file whole (atomic): its query and every record, sorted by post id."""
    stored = AskFile(query=query.text, assessments=dict(sorted(records.items())))
    _dump(stored.model_dump(mode="json"), path)


def ask_files(directory: Path) -> list[Path]:
    """Every query's answer file under `directory` (not the history), by name."""
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.json") if path.name != ASK_INDEX)


def load_ask_file(path: Path) -> AskFile:
    """A query's file read on its own — for a reader that does not know its query yet (the
    history rebuilding a lost entry). A file that cannot be read is refused and named as
    ANOTHER query's, with where to move it: the operator is asking something else, and the
    fix is to take this file out of the way, not to repair the query they typed."""
    try:
        return AskFile.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise JevError(
            f"{path}: consulta guardada ilegible (otra consulta; sácalo de {path.parent}/ "
            f"y vuelve a lanzar): {exc}"
        ) from exc


def load_ask_index(path: Path) -> AskIndex:
    """The query history and the cost calibration; empty before the first query.

    Refused when corrupt — the history is the only list of what was asked, and an empty one
    saved over it loses it — including an entry filed under another query's sha."""
    if not path.exists():
        return AskIndex()
    try:
        return AskIndex.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise JevError(f"{path}: historial de consultas ilegible ({exc})") from exc


def save_ask_index(index: AskIndex, path: Path) -> None:
    """Write the history whole (atomic), queries sorted by sha."""
    _dump(index.model_dump(mode="json"), path)
