"""Mechanical validation of executor output against guardrails + vocabulary.

The LLM emits only judgment (summary + topics). This module proves, with code,
that the judgment is structurally sound — it never trusts the LLM for that.
"""

from __future__ import annotations

from collections.abc import Iterable

from xbrain.rubrics import load_guardrails

# The only keys an enrichment judgment may contain.
_ALLOWED_KEYS = {"summary", "primary_topic", "topics"}


def _validate_judgment_keys(judgment: dict) -> list[str]:
    """Reject any key outside the allowed enrichment schema."""
    extra = set(judgment) - _ALLOWED_KEYS
    if extra:
        return [f"unexpected keys (LLM must emit only judgment): {sorted(extra)}"]
    return []


def _validate_summary(judgment: dict, rules: dict) -> list[str]:
    """Require a non-empty summary when guardrails demand it."""
    summary = judgment.get("summary")
    if rules.get("summary_required", True) and not (summary and str(summary).strip()):
        return ["summary is missing or empty"]
    return []


def _validate_topics_list(judgment: dict, rules: dict, vocab: set[str]) -> list[str]:
    """Validate the topics list itself: count bounds, duplicates, vocabulary membership.

    Returns an empty list when `topics` is not a list — the caller (`validate_judgment`)
    is responsible for emitting the type error and aborting further topic-related checks.
    """
    topics = judgment.get("topics")
    if not isinstance(topics, list):
        return []
    errors: list[str] = []
    lo, hi = rules.get("topics_min", 1), rules.get("topics_max", 4)
    if not (lo <= len(topics) <= hi):
        errors.append(f"topics has {len(topics)} entries, must be {lo}-{hi}")
    if len(set(topics)) != len(topics):
        errors.append("topics has duplicate entries")
    if rules.get("topics_must_be_in_vocab", True):
        for slug in topics:
            if slug not in vocab:
                errors.append(f"topic '{slug}' is not in the vocabulary")
    return errors


def _validate_primary_topic(
    judgment: dict, topics: list, rules: dict, vocab: set[str]
) -> list[str]:
    """Validate `primary_topic`: presence, vocabulary membership, and inclusion in topics."""
    primary = judgment.get("primary_topic")
    if not primary:
        return ["primary_topic is missing"]
    errors: list[str] = []
    if rules.get("topics_must_be_in_vocab", True) and primary not in vocab:
        errors.append(f"primary_topic '{primary}' is not in the vocabulary")
    if rules.get("primary_topic_must_be_in_topics", True) and primary not in topics:
        errors.append(f"primary_topic '{primary}' is not inside topics")
    return errors


def validate_judgment(judgment: dict, vocab_slugs: Iterable[str]) -> list[str]:
    """Return a list of human-readable errors; an empty list means valid."""
    rules = load_guardrails().get("enrichment", {})
    vocab = set(vocab_slugs)

    errors: list[str] = []
    errors += _validate_judgment_keys(judgment)
    errors += _validate_summary(judgment, rules)

    topics = judgment.get("topics")
    if not isinstance(topics, list):
        errors.append("topics must be a list")
        return errors

    errors += _validate_topics_list(judgment, rules, vocab)
    errors += _validate_primary_topic(judgment, topics, rules, vocab)
    return errors


# The only keys a topic-overview judgment may contain.
_ALLOWED_OVERVIEW_KEYS = {"overview", "notes"}


def _validate_overview_notes(notes: list, rules: dict) -> list[str]:
    """Validate the `notes` list — count bounds and per-entry string typing."""
    errors: list[str] = []
    lo, hi = rules.get("notes_min", 0), rules.get("notes_max", 15)
    if not (lo <= len(notes) <= hi):
        errors.append(f"notes has {len(notes)} entries, must be {lo}-{hi}")
    if any(not isinstance(n, str) for n in notes):
        errors.append("notes entries must all be strings")
    return errors


def validate_overview(judgment: dict) -> list[str]:
    """Return a list of human-readable errors; an empty list means valid.

    Enforces the hard rule mechanically: a topic overview is plain prose — any
    wikilink (`[[`) is an identifier the LLM must never emit.
    """
    rules = load_guardrails().get("topic_overview", {})
    errors: list[str] = []

    extra = set(judgment) - _ALLOWED_OVERVIEW_KEYS
    if extra:
        errors.append(f"unexpected keys (LLM must emit only judgment): {sorted(extra)}")

    overview = judgment.get("overview")
    if rules.get("overview_required", True) and not (
        isinstance(overview, str) and overview.strip()
    ):
        errors.append("overview must be a non-empty string")

    notes = judgment.get("notes")
    if not isinstance(notes, list):
        errors.append("notes must be a list")
        return errors

    errors += _validate_overview_notes(notes, rules)

    blob = str(overview or "") + " ".join(str(note) for note in notes)
    if "[[" in blob:
        errors.append("overview/notes must not contain a wikilink ('[[')")

    return errors
