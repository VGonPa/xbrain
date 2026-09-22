"""`data/jev/topics.json`: one `TopicAssessment` per item id.

A SIDE-CAR. `items.json` is never opened for writing by anything under `xbrain.jev`: the
topic assessment is a second opinion about an item, not a fact about it, and writing it
onto the item would make the enrich assignment and the judge's answer indistinguishable
the moment a report wanted to compare them. Keeping them in separate files is what lets
`xbrain jev` be re-run, thrown away or ignored without touching the corpus.

The file is rewritten WHOLESALE on every save, which is why `TopicAssessment` forbids
unknown fields: a record a newer writer added a field to would otherwise be silently
narrowed by an older one on the next run.
"""

from __future__ import annotations

import json
from pathlib import Path

from xbrain.jev.models import TopicAssessment
from xbrain.store import _atomic_write


def load_assessments(path: Path) -> dict[str, TopicAssessment]:
    """The assessments keyed by item id; an empty dict if the file does not exist.

    A missing file is the FIRST RUN, not a fault — the side-car is created by the first
    save. A file that exists but is malformed is not swallowed: it raises, because
    silently returning `{}` would re-ask (and re-pay for) the whole corpus.
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {item_id: TopicAssessment.model_validate(data) for item_id, data in raw.items()}


def save_assessments(assessments: dict[str, TopicAssessment], path: Path) -> None:
    """Persist as pretty, sorted, UTF-8 JSON (atomic write, like `store.save_store`).

    Sorted and pretty for the same reason the item store is: this file is committed, so a
    run that changed two records must show two changed records in the diff and not a
    reordering of the whole corpus. Atomic because a run is paid for — a partial file from
    an interrupted write would lose every assessment, including the ones already billed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        item_id: assessment.model_dump(mode="json")
        for item_id, assessment in sorted(assessments.items())
    }
    _atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
