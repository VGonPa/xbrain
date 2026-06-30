# tests/test_extractor.py
from datetime import datetime, timezone

from test_graphql import SAMPLE_RESPONSE
from xbrain.extract.extractor import _filter_in_range, collect_new_items, rate_limit_decision


def _sample_item():
    """The single parsed item from SAMPLE_RESPONSE (created 2026-05-10 14:23 UTC)."""
    items, _ = collect_new_items([SAMPLE_RESPONSE], "bookmark", set())
    return items[0]


def test_collect_new_items_returns_parsed_items():
    items, hit_known = collect_new_items([SAMPLE_RESPONSE], "bookmark", set())
    assert [i.id for i in items] == ["111"]
    assert hit_known is False


def test_collect_new_items_flags_known_id_and_skips_it():
    items, hit_known = collect_new_items([SAMPLE_RESPONSE], "bookmark", {"111"})
    assert items == []
    assert hit_known is True


def test_collect_new_items_handles_empty_responses():
    items, hit_known = collect_new_items([], "bookmark", set())
    assert items == []
    assert hit_known is False


def test_rate_limit_decision_scrolls_when_no_new_429():
    assert rate_limit_decision(new_hits=False, backoffs_done=0, max_backoffs=3) == "scroll"
    # No fresh 429 even after prior backoffs → keep scrolling.
    assert rate_limit_decision(new_hits=False, backoffs_done=3, max_backoffs=3) == "scroll"


def test_rate_limit_decision_backs_off_on_fresh_429_within_budget():
    assert rate_limit_decision(new_hits=True, backoffs_done=0, max_backoffs=3) == "backoff"
    assert rate_limit_decision(new_hits=True, backoffs_done=2, max_backoffs=3) == "backoff"


def test_rate_limit_decision_aborts_once_backoff_budget_is_spent():
    # Stop rather than hammer X and risk a ban once we've backed off enough times.
    assert rate_limit_decision(new_hits=True, backoffs_done=3, max_backoffs=3) == "abort"
    assert rate_limit_decision(new_hits=True, backoffs_done=9, max_backoffs=3) == "abort"


def test_filter_in_range_keeps_item_with_open_bounds():
    item = _sample_item()
    assert _filter_in_range([item], None, None) == [item]


def test_filter_in_range_drops_item_before_since():
    since = datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert _filter_in_range([_sample_item()], since, None) == []


def test_filter_in_range_drops_item_after_until():
    until = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert _filter_in_range([_sample_item()], None, until) == []


def test_filter_in_range_keeps_item_within_both_bounds_and_dedups_by_id():
    item = _sample_item()
    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    until = datetime(2026, 12, 31, tzinfo=timezone.utc)
    assert _filter_in_range([item, item], since, until) == [item]
