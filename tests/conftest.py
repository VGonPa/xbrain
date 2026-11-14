# tests/conftest.py
"""Shared test doubles for the Anthropic client.

The real code (`xbrain.llm_json.json_from_response`) filters response blocks
with ``getattr(b, "type", None) == "text"`` and joins their ``.text``, so the
fake's blocks must carry BOTH a ``type`` and a ``text`` attribute.

`FakeAnthropic` takes a list of JSON-serialisable payload dicts and returns
them in order from ``.messages.create(...)``, recording every call. A payload
that is an ``Exception`` instance is raised instead of returned, so a single
fake can simulate a transient API failure mid-batch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_firecrawl_credentials(monkeypatch, tmp_path):
    """Point the Firecrawl credentials lookup at an empty temp dir, for EVERY test.

    `fetch.firecrawl_key` falls back to the `firecrawl` CLI's stored credentials
    when `FIRECRAWL_API_KEY` is unset. Without this fixture, "no key configured"
    would mean "no key on the machine RUNNING the tests" — the suite would pass
    in CI and fail on a developer laptop that has run `firecrawl login` (or the
    reverse), and a `monkeypatch.delenv("FIRECRAWL_API_KEY")` would quietly stop
    meaning what it says. Autouse because the risk is exactly in the tests that
    do not think about Firecrawl at all.
    """
    monkeypatch.setattr(
        "xbrain.fetch.FIRECRAWL_CREDENTIAL_PATHS",
        (Path(tmp_path) / "no-such-credentials.json",),
    )


class FakeBlock:
    """One Anthropic content block — a text block holding a JSON payload."""

    type = "text"

    def __init__(self, payload: dict):
        self.text = json.dumps(payload)


class FakeResponse:
    """An Anthropic API response — a `.content` list of blocks."""

    def __init__(self, payload: dict):
        self.content = [FakeBlock(payload)]


class FakeMessages:
    """A fake `client.messages` that pops one payload per `create` call."""

    def __init__(self, payloads: list):
        self._payloads = list(payloads)
        self.calls: list[dict] = []

    def create(self, **kwargs) -> FakeResponse:
        self.calls.append(kwargs)
        payload = self._payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return FakeResponse(payload)


class FakeAnthropic:
    """Drop-in fake for `anthropic.Anthropic`.

    Pass a list of JSON-serialisable payload dicts (or `Exception` instances);
    each `.messages.create(...)` call returns/raises the next one in order.
    Recorded calls are available on `.messages.calls`.
    """

    def __init__(self, payloads: list):
        self.messages = FakeMessages(payloads)
