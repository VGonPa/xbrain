# tests/test_llm_json.py
from xbrain.llm_json import extract_json, json_from_response


def test_extract_json_handles_a_fenced_block():
    fenced = 'Here:\n```json\n{"summary":"r","primary_topic":"misc","topics":["misc"]}\n```'
    assert extract_json(fenced)["primary_topic"] == "misc"


def test_extract_json_raises_value_error_on_garbage():
    import pytest

    with pytest.raises(ValueError):
        extract_json("totally not json at all")


def test_extract_json_raises_value_error_on_malformed_json():
    import pytest

    # A bracketed blob that json.loads cannot parse.
    with pytest.raises(ValueError) as exc_info:
        extract_json('{"summary": "r", "primary_topic": missing-quotes}')
    assert "malformed JSON" in str(exc_info.value)


def test_extract_json_finds_object_amid_prose_with_stray_braces():
    # Prose with a lone `{` before the real object — a greedy `{.*}` regex
    # would start at the stray brace and fail; raw_decode scanning recovers.
    text = (
        "The model said something like { and then later gave us "
        '{"summary": "r", "primary_topic": "misc", "topics": ["misc"]} '
        "as the answer."
    )
    result = extract_json(text)
    assert result["primary_topic"] == "misc"
    assert result["topics"] == ["misc"]


def test_extract_json_returns_first_parseable_object():
    text = '{"primary_topic": "first"} then {"primary_topic": "second"}'
    assert extract_json(text)["primary_topic"] == "first"


class _TextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _NonTextBlock:
    type = "tool_use"


class _Response:
    def __init__(self, content):
        self.content = content


def test_json_from_response_joins_text_blocks():
    response = _Response(
        [
            _TextBlock('{"summary": "r", '),
            _TextBlock('"primary_topic": "misc", "topics": ["misc"]}'),
        ]
    )
    assert json_from_response(response)["primary_topic"] == "misc"


def test_json_from_response_raises_when_no_text_block():
    import pytest

    response = _Response([_NonTextBlock()])
    with pytest.raises(ValueError) as exc_info:
        json_from_response(response, context="item 42")
    assert "no text block" in str(exc_info.value)
    assert "item 42" in str(exc_info.value)
