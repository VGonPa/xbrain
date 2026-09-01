"""The knowledge layer — xbrain's read contract for external models (spec §3).

A logical read view over `Item` + `Content` + `Enrichment` + `Topic` + `TopicPage`, so a
consumer never has to know the shape of `items.json`, hunt for a markdown heading, or guess
which account wrote a quoted tweet. Nothing here mutates the store: every module is
read-only by construction.

The layer is built in four PRs (spec §11). This one — the contract — ships the read models,
provenance, identity, the surface emitter, the chunker, the frozen external schemas, the
golden set and the evaluation harness. The index, embeddings, the graph and MCP arrive
later and consume these names without renegotiating them.
"""

from xbrain.knowledge.provenance import (
    DEFAULT_EVIDENCE_CLASSES,
    ORIGIN_TRUST,
    Origin,
    TrustClass,
    is_derived,
)

__all__ = [
    "DEFAULT_EVIDENCE_CLASSES",
    "ORIGIN_TRUST",
    "Origin",
    "TrustClass",
    "is_derived",
]
