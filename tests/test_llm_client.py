import json

import pytest

from xbrain.llm_client import (
    DEFAULT_NANOGPT_BASE_URL,
    DEFAULT_NANOGPT_MODEL,
    NanoGPTAPIError,
    NanoGPTClient,
    normalize_llm_provider,
)
from xbrain.llm_json import json_from_response


class _HTTPResponse:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Session:
    def __init__(self, response: _HTTPResponse):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.response


def test_nanogpt_client_posts_non_streaming_json_completion():
    session = _Session(
        _HTTPResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"summary": "ok", "primary_topic": "misc", "topics": ["misc"]}
                            )
                        }
                    }
                ]
            },
        )
    )
    client = NanoGPTClient(api_key="ng_key", session=session)

    response = client.messages.create(
        model=DEFAULT_NANOGPT_MODEL,
        max_tokens=600,
        system="system prompt",
        messages=[{"role": "user", "content": "user prompt"}],
    )

    assert json_from_response(response, context="test") == {
        "summary": "ok",
        "primary_topic": "misc",
        "topics": ["misc"],
    }
    call = session.calls[0]
    assert call["url"] == f"{DEFAULT_NANOGPT_BASE_URL}/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer ng_key"
    payload = call["json"]
    assert payload["model"] == DEFAULT_NANOGPT_MODEL
    assert payload["max_tokens"] == 600
    assert payload["stream"] is False
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["reasoning"] == {"exclude": True}
    assert payload["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]


def test_nanogpt_client_converts_image_blocks_to_openai_shape():
    session = _Session(
        _HTTPResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": '[{"index": 0, "is_decorative": false, "description": "chart"}]'
                        }
                    }
                ]
            },
        )
    )
    client = NanoGPTClient(api_key="ng_key", session=session)

    response = client.messages.create(
        model=DEFAULT_NANOGPT_MODEL,
        max_tokens=1200,
        system="describe images",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc123",
                        },
                    },
                    {"type": "text", "text": "Describe image 0."},
                ],
            }
        ],
    )

    assert response.content[0].text.startswith("[")
    payload = session.calls[0]["json"]
    assert "response_format" not in payload
    assert payload["messages"][1]["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc123"},
        },
        {"type": "text", "text": "Describe image 0."},
    ]


def test_nanogpt_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("NANOGPT_API_KEY", raising=False)
    session = _Session(_HTTPResponse(200, {}))
    client = NanoGPTClient(session=session)

    with pytest.raises(NanoGPTAPIError, match="NANOGPT_API_KEY"):
        client.messages.create(model=DEFAULT_NANOGPT_MODEL, max_tokens=1, messages=[])

    assert session.calls == []


def test_nanogpt_client_raises_on_http_error():
    session = _Session(_HTTPResponse(401, text="unauthorized"))
    client = NanoGPTClient(api_key="bad", session=session)

    with pytest.raises(NanoGPTAPIError, match="HTTP 401"):
        client.messages.create(model=DEFAULT_NANOGPT_MODEL, max_tokens=1, messages=[])


def test_nanogpt_client_raises_on_malformed_response():
    session = _Session(_HTTPResponse(200, {"data": []}))
    client = NanoGPTClient(api_key="ng_key", session=session)

    with pytest.raises(NanoGPTAPIError, match="no first choice"):
        client.messages.create(model=DEFAULT_NANOGPT_MODEL, max_tokens=1, messages=[])


def test_normalize_llm_provider_accepts_nanogpt_aliases():
    assert normalize_llm_provider("nanogpt") == "nanogpt"
    assert normalize_llm_provider("nano-gpt") == "nanogpt"
    assert normalize_llm_provider("anthropic") == "anthropic"

    with pytest.raises(ValueError, match="provider"):
        normalize_llm_provider("openrouter")
