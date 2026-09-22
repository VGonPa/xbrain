# tests/test_jev_defaults.py — the ONE place a Jev run's input cost is computed.
from datetime import datetime, timezone

from xbrain.jev.defaults import (
    INPUT_USD_PER_MTOK,
    input_cost_usd,
    input_tokens_total,
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
