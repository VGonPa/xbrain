"""Provenance as a TYPE — who or what produced a piece of text (spec §3.4, §3.5).

Provenance is not decoration. Spec §3.4: it *determines what a model may assert about the
fragment*. `origin=asr` does not mean the speaker wrote those words; `origin=vlm` does not
mean the text appeared literally in the image; `origin=llm` is not a primary source; and
`origin=unknown` is not filled in by intuition — it stays unknown until the data records
its producer.

TWO VOCABULARIES, ONE TABLE. `Origin` answers *what produced this text*; `TrustClass`
answers *what role it may play in an answer*. `ORIGIN_TRUST` is the single mapping between
them, asserted TOTAL over `Origin` by the suite, so adding an origin without deciding its
class is a red test rather than a silent default.
"""

from __future__ import annotations

from typing import Literal

# What produced the text. Spec §3.4, verbatim.
Origin = Literal["source", "asr", "vlm", "llm", "user", "unknown"]

# What role the text may play in an answer. Spec §3.5, one class per row of its table.
TrustClass = Literal[
    "primary_source",  # captured from the source itself
    "user_text",  # written by the user
    "machine_extracted",  # ASR / VLM: automatic evidence, carries an origin warning
    "llm_synthesis",  # summary / digest / overview / topic note
    "classification",  # RESERVED — see the note below; no Origin maps here today
]

# THE table. One mapping, not five call sites deciding for themselves.
#
# `unknown` maps to `llm_synthesis` ON PURPOSE, and this is the fail-closed decision of the
# whole module. `Topic.description` does not record whether it was generated or hand-edited
# (spec §4), so its producer is genuinely unknown. The safe direction is DOWN: an unknown
# producer is never promoted to the status of a captured source, because the cost of the
# two errors is not symmetric — treating a source as synthesis loses a citation, treating
# synthesis as a source manufactures one.
ORIGIN_TRUST: dict[Origin, TrustClass] = {
    "source": "primary_source",
    "user": "user_text",
    "asr": "machine_extracted",
    "vlm": "machine_extracted",
    "llm": "llm_synthesis",
    "unknown": "llm_synthesis",
}

# The classes that may stand as final evidence by default — spec §3.5, third column.
#
# `llm_synthesis` is absent by design: a summary or a topic overview is a discovery signal,
# and when a recoverable underlying source exists the answer should cite that instead. It
# is not a prohibition on SHOWING derived text (spec §3.5 is explicit that the system "does
# not forbid a model from seeing derivatives"), only on presenting it as a primary fact.
DEFAULT_EVIDENCE_CLASSES: frozenset[TrustClass] = frozenset(
    {"primary_source", "user_text", "machine_extracted"}
)

# `classification` is RESERVED and deliberately outside the codomain of `ORIGIN_TRUST`.
#
# The spec §3.5 row *"topic asignado"* is not a surface: it is the
# `Enrichment.primary_topic` / `topics` assignment, which travels as `KnowledgeItem.topics`
# and `KnowledgeChunk.topics` — never as citable text with a trust class of its own. The
# class stays declared because the advanced graph (spec §12) will need it the day an
# assignment carries its own confidence. Pinned from both sides by the suite so it is
# neither quietly reachable nor quietly deleted.


def is_derived(origin: Origin) -> bool:
    """True when a machine produced this text FROM something else.

    `source` and `user` are the only origins where a human wrote the indexed words. ASR and
    VLM are machine transformations of real material — evidence, but automatic, and the
    response must say so. LLM output is synthesis, and `unknown` is treated as synthesis by
    the same fail-closed rule that governs `ORIGIN_TRUST`.
    """
    return origin not in ("source", "user")
