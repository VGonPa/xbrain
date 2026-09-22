"""The `[jev]` defaults, the per-provider input price, and the ONE way to price a run.

`config.py` imports the defaults rather than retyping them (rule 5): a literal in the loader
would be a second definition that drifts the day this one moves, and nothing would go red
because each file stays internally consistent. Consumers read `cfg.jev_*` and never
re-declare one.

The pricing helpers live here for the same reason. `xbrain jev topics` reports what a run
cost and `xbrain jev report` reports it again; a formula inlined at each call site is two
definitions of the bill, and the one that drifts is the one nobody re-derives. Every consumer
calls these — none re-implements `tokens / 1e6 * rate`.

The RENDERING lives here too, for the third time the same argument: `jev_cost_fragment` is the
one Spanish sentence that quotes a bill. Three call sites used to format it themselves and
printed the same side-car as `~0.000 $` and `~0.0001 $`, with the unpriced marker worded two
ways — a recap that disagrees with the bill it recaps.
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
    # `float(...)`: `sum()` over an empty iterable returns `int 0`, and a function annotated
    # `-> float` that sometimes returns an int makes every caller remember the cast — or ship a
    # JSON key whose type changes with the contents of the side-car.
    return float(
        sum(
            (assessment.input_tokens or 0) / 1e6 * INPUT_USD_PER_MTOK.get(assessment.provider, 0.0)
            for assessment in assessments
        )
    )


def unpriced_providers(assessments: Iterable[TopicAssessment]) -> tuple[str, ...]:
    """The distinct providers in `assessments` that `INPUT_USD_PER_MTOK` cannot price.

    Sorted and de-duplicated so the operator-facing line is stable across runs: this is
    what turns a bare `~0.0000 $` into "0.0000 because nobody prices this judge".
    """
    return tuple(sorted({a.provider for a in assessments if a.provider not in INPUT_USD_PER_MTOK}))


def plural(count: int, singular: str, plural: str) -> str:
    """`count` with its noun agreed. Spanish agrees at 1 ONLY — "0 evaluaciones" is plural.

    One helper rather than a conditional per string: the count-bearing strings in this package
    are operator-facing Spanish, and "1 evaluaciones guardadas" in the line that reports what a
    run cost reads as a bug in the counting, not in the grammar.

    It lives HERE rather than in `cli.py` because the markdown report needs the same rule and
    must not import the CLI; a second copy is a second rule, and the one that drifts is the one
    nobody re-reads.
    """
    return f"{count} {singular if count == 1 else plural}"


def jev_cost_fragment(tokens: int, unknown: int, cost_usd: float, unpriced: Iterable[str]) -> str:
    """`N tokens de entrada (+K sin recuento) (~X $ · proveedor sin tarifa: a, b)`.

    THE ONE SENTENCE THAT QUOTES A BILL. `xbrain jev topics` reports what a run cost, and
    `xbrain jev report` and `topics-report.md` recap the same side-car; a recap that prints a
    different figure from the bill it recaps is the one thing a recap must not do. Formatting
    it at each call site produced exactly that — `~0.000 $` against `~0.0001 $` for one run.

    FOUR decimals, never three. At `0.042 $/MTok` a whole corpus costs under `0.50 $`, so three
    decimals round most real runs to `~0.000 $`: a bill that reports itself as free.

    The two parenthetical markers exist because a bare `~0.0000 $` cannot say which zero it is.
    A record whose provider reported no usage (`input_tokens is None`, a documented real
    behaviour) contributes nothing it can prove, so without `(+K sin recuento)` a fully paid run
    reports itself as free. A provider absent from the price table contributes 0.0 rather than
    borrowing another vendor's rate — an invented rate reads as a bill, and that is the worse
    error — so it is NAMED, not counted.
    """
    line = plural(tokens, "token de entrada", "tokens de entrada")
    if unknown:
        line += f" (+{unknown} sin recuento)"
    cost = f"~{cost_usd:.4f} $"
    names = list(unpriced)
    if names:
        noun = "proveedor sin tarifa" if len(names) == 1 else "proveedores sin tarifa"
        cost += f" · {noun}: {', '.join(names)}"
    return f"{line} ({cost})"
