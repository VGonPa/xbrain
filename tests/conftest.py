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
import re
from pathlib import Path

import pytest

#: Rich's ANSI colour spans, and its panel/box-drawing chrome (U+2500–U+257F).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_BOX_RE = re.compile("[\u2500-\u257f]")


def plain_output(output: str) -> str:
    """Normalise CliRunner output for substring assertions on Rich/Typer panels.

    With colour on (CI, but not under pytest capture) Rich styles each flag with the
    leading dash in its own ANSI span, so `--apply` never appears contiguously; the panel
    also wraps onto several lines with `│` borders landing between words. Stripping the
    escapes rejoins the flag, dropping the chrome and collapsing whitespace rejoins wrapped
    words — which makes an assertion terminal-width AND colour independent.

    It lives here rather than in a test module so any of them can assert on rendered output
    without a second copy; `tests/test_cli.py`'s help battery is today's only caller. It
    replaced a `COLUMNS` env crutch that was width-lucky rather than width-proof and did
    nothing about ANSI.
    """
    return " ".join(_BOX_RE.sub(" ", _ANSI_RE.sub("", output)).split())


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


@pytest.fixture(autouse=True)
def _isolate_typesafe_credentials(monkeypatch, tmp_path):
    """No test may see a real TYPESAFE_API_KEY or a real `.env`, for EVERY test.

    Same reasoning as the Firecrawl fixture above, with money attached: `xbrain.jev` calls
    the paid TypeSafe API, and the key it uses comes from the environment or from
    `<repo>/.env`. Without this, a suite run on a machine that has a key configured could
    reach the live API — and "no key configured" would mean "no key on the machine RUNNING
    the tests", so the same test would pass in CI and fail on a developer laptop.
    Autouse because the risk is exactly in the tests that do not think about Jev at all.

    `tests/test_jev_env.py` restores the real lookup, since it tests the lookup itself.
    """
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    empty = Path(tmp_path) / "no-such-dotenv" / ".env"
    monkeypatch.setattr("xbrain.jev.env.dotenv_path", lambda repo_root: empty)


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
