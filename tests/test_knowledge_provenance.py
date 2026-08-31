# tests/test_knowledge_provenance.py
"""Provenance is a TYPE with a total mapping, not a decorative label (Plan 01 §3.1).

Spec §3.4 is explicit that provenance *determines what a model may assert about the
fragment*: `origin=asr` does not mean the speaker wrote those words, `origin=vlm` does not
mean the text appeared in the image, and `origin=llm` is not a primary source. A label
nobody can enumerate cannot carry that weight, so `Origin` is a `Literal` and `ORIGIN_TRUST`
is asserted TOTAL over it — adding an origin without deciding its trust class is red.

The mapping table below is transcribed from spec §3.4/§3.5 independently of the module. It
is not a restatement of `ORIGIN_TRUST`: it is the other half of the pair, and the test's
job is to catch the day the two disagree.
"""

from __future__ import annotations

from typing import get_args

import pytest

from xbrain.knowledge.provenance import (
    DEFAULT_EVIDENCE_CLASSES,
    ORIGIN_TRUST,
    Origin,
    TrustClass,
    is_derived,
)

# Spec §3.4 (the `origin` table) crossed with §3.5 (the "class" column). Transcribed by
# hand from the spec, deliberately NOT imported from the module under test.
SPEC_ORIGIN_TRUST = {
    "source": "primary_source",  # texto capturado de la fuente
    "user": "user_text",  # texto escrito por el usuario
    "asr": "machine_extracted",  # transcripción automática de audio
    "vlm": "machine_extracted",  # descripción automática de contenido visual
    "llm": "llm_synthesis",  # síntesis o clasificación generada
    "unknown": "llm_synthesis",  # falla cerrado — ver el test dedicado
}

# Spec §3.5, third column ("Evidencia final por defecto" == sí).
SPEC_DEFAULT_EVIDENCE_CLASSES = {"primary_source", "user_text", "machine_extracted"}


def test_origin_trust_is_total_over_origin() -> None:
    """Every declared `Origin` has a trust class, and no dict key is a typo.

    Seen red by deleting `"unknown"` from `ORIGIN_TRUST`: the set comparison names the
    missing key. Both directions matter — an entry for an origin that no longer exists is
    dead weight that a reader would trust.
    """
    assert set(get_args(Origin)) == set(ORIGIN_TRUST)


def test_origin_trust_matches_the_spec_table_row_by_row() -> None:
    """The module's table equals the spec's, row by row.

    EXCLUDED, on purpose: the spec §3.5 row *"topic asignado"*. That row is not a surface —
    it is the `Enrichment.primary_topic` / `topics` ASSIGNMENT, which travels as
    `KnowledgeItem.topics` and `KnowledgeChunk.topics` and is never citable text with a
    trust class of its own. Inventing a surface for it to make the table look complete
    would put a classification signal where a quotable body belongs.
    """
    assert ORIGIN_TRUST == SPEC_ORIGIN_TRUST


def test_unknown_fails_closed_to_llm_synthesis() -> None:
    """`unknown` is never promoted to a source.

    `Topic.description` does not record whether it was generated or hand-edited (spec §4),
    so its origin is genuinely unknown. The safe direction is down: treat it as synthesis.
    Mapping it to `primary_source` would let a hand-editable field be cited as captured
    evidence, which is the one thing provenance exists to prevent.
    """
    assert ORIGIN_TRUST["unknown"] == "llm_synthesis"
    assert is_derived("unknown") is True


def test_default_evidence_classes_match_the_spec_column() -> None:
    """Seen red by adding `llm_synthesis` to the frozenset.

    Spec §3.5 says a summary/digest is NOT final evidence by default when a recoverable
    underlying source exists, and a topic overview is not until it keeps per-claim support.
    """
    assert set(DEFAULT_EVIDENCE_CLASSES) == SPEC_DEFAULT_EVIDENCE_CLASSES


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ("source", False),
        ("user", False),
        ("asr", True),
        ("vlm", True),
        ("llm", True),
        ("unknown", True),
    ],
)
def test_is_derived_is_true_exactly_for_machine_produced_text(origin: str, expected: bool) -> None:
    """ "Derived" == a machine produced this text from something else.

    `source` and `user` are the only two origins where a human wrote the words that are
    being indexed. ASR and VLM are machine transformations of real material — evidence,
    but automatic — and LLM output is synthesis.
    """
    assert is_derived(origin) is expected


def test_is_derived_agrees_with_the_trust_table() -> None:
    """The two derived-ness signals cannot drift apart.

    `is_derived` is a predicate over `Origin`; `ORIGIN_TRUST` maps to a class. They encode
    the same fact from two angles, so a change to one that is not made in the other is a
    silent contradiction — which is CLAUDE.md rule 5 in miniature.
    """
    for origin, trust in ORIGIN_TRUST.items():
        assert is_derived(origin) == (trust not in ("primary_source", "user_text"))


def test_classification_is_reserved_and_unreachable() -> None:
    """`classification` is declared but no `Origin` maps to it — and that is deliberate.

    It exists for the future advanced graph (spec §12), where a topic ASSIGNMENT will carry
    its own confidence. Today no surface can be classified as one, because a topic
    assignment is not a text surface. Pinned from both sides so it is neither quietly
    reachable (which would let a classification be cited as evidence) nor quietly deleted
    (which would make the future migration invisible).
    """
    assert "classification" not in set(ORIGIN_TRUST.values())
    assert "classification" in get_args(TrustClass)
