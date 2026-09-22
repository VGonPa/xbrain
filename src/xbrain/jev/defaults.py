"""The `[jev]` defaults, the per-provider input price, and the ONE way to price a run.

`config.py` imports the defaults rather than retyping them (rule 5): a literal in the loader
would be a second definition that drifts the day this one moves, and nothing would go red
because each file stays internally consistent. Consumers read `cfg.jev_*` and never
re-declare one.

The pricing helpers live here for the same reason. `xbrain jev topics` reports what a run
cost and `xbrain jev summarize` reports it again; a formula inlined at each call site is two
definitions of the bill, and the one that drifts is the one nobody re-derives. Every consumer
calls these — none re-implements `tokens / 1e6 * rate`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Annotations only — `from __future__ import annotations` keeps them strings, so this
    # module stays importable without dragging in pydantic. `config.py` imports it during
    # `load_config`, which every command pays for.
    from xbrain.jev.models import TopicAssessment

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


def input_tokens_total(assessments: Iterable[TopicAssessment]) -> tuple[int, int]:
    """`(counted_tokens, records_without_count)` over `assessments`.

    The second number is not decoration. `input_tokens` is `None` whenever the provider
    reported no usage, which is a documented real behaviour — and folding that into 0 turns
    a run that was paid for into a run that reports itself as free. The caller shows the
    count so the first number is read as "at least this much", never as the whole bill.
    """
    counted = 0
    unknown = 0
    for assessment in assessments:
        if assessment.input_tokens is None:
            unknown += 1
        else:
            counted += assessment.input_tokens
    return counted, unknown


def input_cost_usd(assessments: Iterable[TopicAssessment]) -> float:
    """Estimated USD for the INPUT tokens of `assessments`, priced PER RECORD.

    Per record, by the provider that ANSWERED it, never one rate for the batch: `assessed`
    may mix providers, and each record carries its own. A provider absent from
    `INPUT_USD_PER_MTOK` contributes 0.0 rather than borrowing another vendor's rate — an
    invented rate reads as a bill, and that is the worse error. Because 0.0 renders
    identically to a genuinely free run, callers pair this with `unpriced_providers` and
    NAME them; the number alone cannot say "unknown".

    An ESTIMATE, not a bill: `INPUT_USD_PER_MTOK` is a per-version list price while
    `[jev].model` defaults to a moving alias, and records without a token count contribute
    nothing they cannot prove (see `input_tokens_total`).
    """
    return sum(
        (assessment.input_tokens or 0) / 1e6 * INPUT_USD_PER_MTOK.get(assessment.provider, 0.0)
        for assessment in assessments
    )


def unpriced_providers(assessments: Iterable[TopicAssessment]) -> tuple[str, ...]:
    """The distinct providers in `assessments` that `INPUT_USD_PER_MTOK` cannot price.

    Sorted and de-duplicated so the operator-facing line is stable across runs: this is
    what turns a bare `~0.000 $` into "0.000 because nobody prices this judge".
    """
    return tuple(sorted({a.provider for a in assessments if a.provider not in INPUT_USD_PER_MTOK}))
