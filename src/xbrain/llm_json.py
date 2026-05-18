"""Shared helper: extract a JSON object from an LLM text response."""
from __future__ import annotations

import json
import re

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response (fenced or bare).

    Raises ValueError (with context) when no JSON object is present or the
    matched text is not valid JSON — so one malformed response is diagnosable.
    """
    match = _JSON_OBJECT.search(text)
    if not match:
        raise ValueError(f"no JSON object in model response: {text[:200]!r}")
    snippet = match.group(0)
    try:
        return json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"malformed JSON in model response: {exc} -- snippet: {snippet[:200]!r}"
        ) from exc
