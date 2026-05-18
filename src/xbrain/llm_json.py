"""Shared helper: extract a JSON object from an LLM text response."""

from __future__ import annotations

import json


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response (fenced or bare).

    Scans each ``{`` in order and tries ``raw_decode`` starting there, so prose
    with stray braces does not derail the parse (a greedy ``{.*}`` regex would
    span the first ``{`` to the last ``}`` and break). Returns the first object
    that parses.

    Raises ValueError (with context) when no JSON object is present or none of
    the candidates is valid JSON — so one malformed response is diagnosable.
    """
    decoder = json.JSONDecoder()
    last_error: json.JSONDecodeError | None = None
    found_brace = False
    idx = text.find("{")
    while idx != -1:
        found_brace = True
        try:
            obj, _ = decoder.raw_decode(text, idx)
        except json.JSONDecodeError as exc:
            last_error = exc
        else:
            if isinstance(obj, dict):
                return obj
        idx = text.find("{", idx + 1)

    if not found_brace:
        raise ValueError(f"no JSON object in model response: {text[:200]!r}")
    raise ValueError(f"malformed JSON in model response: {last_error} -- snippet: {text[:200]!r}")


def json_from_response(response, context: str = "") -> dict:
    """Extract a JSON object from an Anthropic API response.

    Joins every text block in the response and runs :func:`extract_json` on the
    result. Raises ValueError (with optional ``context``) when the response
    carries no text block at all.
    """
    blocks = [b for b in response.content if getattr(b, "type", None) == "text"]
    if not blocks:
        suffix = f" for {context}" if context else ""
        raise ValueError(f"no text block in model response{suffix}")
    text = "".join(b.text for b in blocks)
    return extract_json(text)
