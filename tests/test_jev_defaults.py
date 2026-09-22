# tests/test_jev_defaults.py — the ONE place a Jev run's input cost is computed.
from datetime import datetime, timezone

from xbrain.jev.defaults import (
    INPUT_USD_PER_MTOK,
    input_cost_usd,
    input_tokens_total,
    jev_cost_fragment,
    plural,
    unpriced_providers,
)
from xbrain.jev.models import PrimaryChoice, TopicAssessment

DT = datetime(2026, 9, 22, tzinfo=timezone.utc)


def _assessment(item_id: str, provider: str, input_tokens: int | None) -> TopicAssessment:
    return TopicAssessment(
        item_id=item_id,
        provider=provider,
        model="jev-1.13.0",
        asked_at=DT,
        contract="a" * 64,
        state_chars=3,
        membership={"ai-coding": 0.9},
        primary=PrimaryChoice(choice="ai-coding", confidence=0.8, probabilities={"ai-coding": 0.8}),
        input_tokens=input_tokens,
    )


def test_the_rate_is_read_from_the_table_not_retyped():
    """If this drifts, every figure below is arithmetic about a number nobody ships."""
    assert INPUT_USD_PER_MTOK["typesafe"] == 0.042


def test_cost_is_summed_per_record_by_the_provider_that_answered():
    """`assessed` may mix providers — a run spanning a re-point of `[jev].model`, or a
    future panel. One rate for the batch would bill the unpriced half at the priced rate."""
    records = [
        _assessment("1", "typesafe", 6_000_000),
        _assessment("2", "otro-juez", 6_000_000),
    ]
    assert input_cost_usd(records) == 6_000_000 / 1e6 * 0.042


def test_an_unpriced_provider_contributes_zero_and_is_named():
    """Zero rather than a borrowed rate — an invented rate reads as a bill. But `~0.000 $`
    is indistinguishable from a genuinely free run, so the providers are NAMED instead of
    the number being left to speak for itself."""
    records = [_assessment("1", "fake", 6_000_000), _assessment("2", "fake", 1)]
    assert input_cost_usd(records) == 0.0
    assert unpriced_providers(records) == ("fake",)
    assert unpriced_providers([_assessment("1", "typesafe", 1)]) == ()


def test_unpriced_providers_are_deduplicated_and_sorted():
    records = [
        _assessment("1", "zeta", 1),
        _assessment("2", "alfa", 1),
        _assessment("3", "zeta", 1),
        _assessment("4", "typesafe", 1),
    ]
    assert unpriced_providers(records) == ("alfa", "zeta")


def test_tokens_report_the_count_and_how_many_records_had_none():
    """`input_tokens is None` is a real provider behaviour. Folding it into 0 reports a
    paid run as free; the second number is what stops the first from lying."""
    records = [
        _assessment("1", "typesafe", 1_500),
        _assessment("2", "typesafe", None),
        _assessment("3", "typesafe", None),
    ]
    assert input_tokens_total(records) == (1_500, 2)


def test_a_record_without_a_token_count_costs_nothing_it_cannot_prove():
    """An unknown count must not be invented as an average or as zero-with-confidence: the
    cost is what the counted records prove, and the caller says how much is unaccounted."""
    records = [_assessment("1", "typesafe", 1_000_000), _assessment("2", "typesafe", None)]
    assert input_cost_usd(records) == 0.042
    assert input_tokens_total(records) == (1_000_000, 1)


def test_an_empty_run_is_zero_of_everything():
    assert input_cost_usd([]) == 0.0
    assert input_tokens_total([]) == (0, 0)
    assert unpriced_providers([]) == ()


# ------------------------------------------------------- the one Spanish cost fragment


def test_plural_agrees_at_one_only():
    """Spanish agrees at 1 ONLY. "1 evaluaciones" reads as a bug in the counting."""
    assert plural(0, "evaluación", "evaluaciones") == "0 evaluaciones"
    assert plural(1, "evaluación", "evaluaciones") == "1 evaluación"
    assert plural(2, "evaluación", "evaluaciones") == "2 evaluaciones"


def test_the_cost_fragment_prints_four_decimals_so_a_real_bill_is_never_rounded_away():
    """At 0.042 $/MTok the numbers are small, so three decimals round a partial run away.

    The two cases are the two ends of a real corpus, measured (`docs/jev.md`): a handful of
    items, whose bill THREE decimals would render `~0.000 $` — free, for work that was paid
    for — and a whole ~2,600-item pass at ~15.5 M input tokens.
    """
    assert jev_cost_fragment(1_500, 0, 0.0001, ()) == "1500 tokens de entrada (~0.0001 $)"
    assert jev_cost_fragment(15_546_000, 0, 0.6529, ()) == "15546000 tokens de entrada (~0.6529 $)"


def test_the_cost_fragment_names_both_kinds_of_zero():
    """`~0.0000 $` alone cannot say WHICH zero it is: nothing was reported, or nobody prices
    the judge. Each marker appears only when it has something to say."""
    assert jev_cost_fragment(0, 2, 0.0, ()) == "0 tokens de entrada (+2 sin recuento) (~0.0000 $)"
    assert (
        jev_cost_fragment(10, 0, 0.0, ("fake",))
        == "10 tokens de entrada (~0.0000 $ · proveedor sin tarifa: fake)"
    )
    assert (
        jev_cost_fragment(10, 0, 0.0, ("a", "b"))
        == "10 tokens de entrada (~0.0000 $ · proveedores sin tarifa: a, b)"
    )
    # One token is one token.
    assert jev_cost_fragment(1, 0, 0.0, ()) == "1 token de entrada (~0.0000 $)"


def test_input_cost_is_a_float_even_when_there_is_nothing_to_price():
    """`sum()` over an empty iterable returns `int 0`, so the annotation would be a lie and
    every caller would have to remember the cast."""
    assert isinstance(input_cost_usd([]), float)
    assert input_cost_usd([]) == 0.0
