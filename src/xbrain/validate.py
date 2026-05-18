"""Mechanical validation of executor output against guardrails + vocabulary.

The LLM emits only judgment (summary + topics). This module proves, with code,
that the judgment is structurally sound — it never trusts the LLM for that.
"""

from __future__ import annotations

from collections.abc import Iterable

from xbrain.rubrics import load_guardrails

# The only keys an enrichment judgment may contain.
_ALLOWED_KEYS = {"summary", "primary_topic", "topics"}


def validate_judgment(judgment: dict, vocab_slugs: Iterable[str]) -> list[str]:
    """Return a list of human-readable errors; an empty list means valid."""
    rules = load_guardrails().get("enrichment", {})
    vocab = set(vocab_slugs)
    errors: list[str] = []

    extra = set(judgment) - _ALLOWED_KEYS
    if extra:
        errors.append(f"unexpected keys (LLM must emit only judgment): {sorted(extra)}")

    summary = judgment.get("summary")
    if rules.get("summary_required", True) and not (summary and str(summary).strip()):
        errors.append("summary is missing or empty")

    topics = judgment.get("topics")
    if not isinstance(topics, list):
        errors.append("topics must be a list")
        return errors

    lo, hi = rules.get("topics_min", 1), rules.get("topics_max", 4)
    if not (lo <= len(topics) <= hi):
        errors.append(f"topics has {len(topics)} entries, must be {lo}-{hi}")

    if len(set(topics)) != len(topics):
        errors.append("topics has duplicate entries")

    if rules.get("topics_must_be_in_vocab", True):
        for slug in topics:
            if slug not in vocab:
                errors.append(f"topic '{slug}' is not in the vocabulary")

    primary = judgment.get("primary_topic")
    if not primary:
        errors.append("primary_topic is missing")
    else:
        if rules.get("topics_must_be_in_vocab", True) and primary not in vocab:
            errors.append(f"primary_topic '{primary}' is not in the vocabulary")
        if rules.get("primary_topic_must_be_in_topics", True) and primary not in topics:
            errors.append(f"primary_topic '{primary}' is not inside topics")

    return errors
