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
  command that decided to overwrite (`cli._backup_jev_sidecar`), not to the writer.
* A corrupt file is repaired by hand or paid for again — which is why `load_assessments`
  refuses one instead of quietly starting from `{}`.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from xbrain.jev.client import JevError
from xbrain.jev.models import TopicAssessment
from xbrain.store import _atomic_write


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
