"""The `[jev]` defaults and the per-provider input price — defined ONCE, here.

`config.py` imports these rather than retyping them (rule 5): a literal in the loader would
be a second definition that drifts the day this one moves, and nothing would go red because
each file stays internally consistent. Consumers read `cfg.jev_*` and never re-declare one.
"""

from __future__ import annotations

#: A moving alias on purpose: TypeSafe's updates are allowed to reach us, and every stored
#: assessment records the concrete `model` the API answered with.
DEFAULT_MODEL = "jev-latest"
#: A topic counts as "backed by Jev" at or above this probability.
DEFAULT_THRESHOLD = 0.85
#: The escape option of the primary-topic Choice: "a topic not in the vocabulary".
DEFAULT_FALLBACK_OPTION = "otro"
#: Requests in flight.
DEFAULT_CONCURRENCY = 8
#: Evidence text is cut here; assessments record the pre-cut length and `truncated`.
DEFAULT_STATE_CHAR_LIMIT = 100_000

#: USD per million INPUT tokens, per provider. TypeSafe's list price for `jev-1.13.0` on
#: 2026-09-22 (docs.typesafe.ai/models): charged per input token, output tokens are free.
#: This is a PER-VERSION price while `[jev].model` defaults to the moving `jev-latest`
#: alias, so any figure derived from it is an ESTIMATE, not a bill — re-check it when the
#: alias advances. Assessments record the concrete model, so a report can say what it priced.
INPUT_USD_PER_MTOK: dict[str, float] = {"typesafe": 0.042}
