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


def test_filter_in_range_bounds_are_inclusive():
    # An item sitting exactly on `since` or `until` is kept ([since, until]).
    item = _sample_item()
    assert _filter_in_range([item], item.created_at, None) == [item]
    assert _filter_in_range([item], None, item.created_at) == [item]


def test_filter_in_range_dedup_keeps_first_seen():
    # Two distinct objects sharing an id → the FIRST survives (setdefault).
    first = _sample_item()
    second = first.model_copy(update={"text": "variante distinta, mismo id"})
    out = _filter_in_range([first, second], None, None)
    assert len(out) == 1
    assert out[0].text == first.text


def test_collect_new_items_persists_the_raw_payload_before_parsing(tmp_path):
    """Persistence happens at INGEST, before any parsing decision — so a field we do not
    yet read (the next `note_tweet`) is on disk anyway. That is the entire point: the
    payload is kept for the parser we have not written yet."""
    from xbrain.extract.extractor import collect_new_items
    from xbrain.payloads import load_payload

    response = {
        "data": {
            "x": {
                "tweet_results": {
                    "result": {
                        "__typename": "Tweet",
                        "rest_id": "42",
                        "core": {
                            "user_results": {
                                "result": {"legacy": {"screen_name": "a", "name": "A"}}
                            }
                        },
                        "legacy": {
                            "full_text": "hello",
                            "created_at": "Wed Jan 01 10:00:00 +0000 2025",
                        },
                        "a_field_no_parser_reads_yet": "kept anyway",
                    }
                }
            }
        }
    }
    items, _ = collect_new_items([response], "bookmark", set(), tmp_path)

    assert [i.id for i in items] == ["42"]
    stored = load_payload(tmp_path, "42")
    assert stored["a_field_no_parser_reads_yet"] == "kept anyway"


def test_collect_new_items_without_a_payload_dir_persists_nothing(tmp_path):
    """Backwards compatible: the pure path stays pure (tests, archive imports)."""
    from xbrain.extract.extractor import collect_new_items
    from xbrain.payloads import stored_ids

    collect_new_items([], "bookmark", set(), None)
    assert stored_ids(tmp_path) == set()
