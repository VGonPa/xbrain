# tests/test_llm_json.py
from xbrain.llm_json import extract_json


def test_extract_json_handles_a_fenced_block():
    fenced = ('Here:\n```json\n{"summary":"r","primary_topic":"misc",'
              '"topics":["misc"]}\n```')
    assert extract_json(fenced)["primary_topic"] == "misc"


def test_extract_json_raises_value_error_on_garbage():
    import pytest

    with pytest.raises(ValueError):
        extract_json("totally not json at all")


def test_extract_json_raises_value_error_on_malformed_json():
    import pytest

    # A bracketed blob that the regex matches but json.loads cannot parse.
    with pytest.raises(ValueError) as exc_info:
        extract_json('{"summary": "r", "primary_topic": missing-quotes}')
    assert "malformed JSON" in str(exc_info.value)
