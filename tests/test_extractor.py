# tests/test_extractor.py
from test_graphql import SAMPLE_RESPONSE
from xbrain.extract.extractor import collect_new_items


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
