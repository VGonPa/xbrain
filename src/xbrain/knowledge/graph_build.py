"""The minimal graph (Plan 04.2, spec §6.2): assignment edges and topic co-occurrence.

An edge here records that xbrain ASSIGNED topics to items — never that the concepts are related
in the world (spec §6.4). The shape is `contracts.GraphEdge`; this module only derives edges
from the store and never writes it.

Two edge families:

- `HAS_PRIMARY_TOPIC` / `HAS_TOPIC`: one edge per (item, topic), read off `Item.enriched`.
- `CO_OCCURS_WITH`: one edge per ordered pair of topics that share at least one item, in BOTH
  directions with the same weight, so an expansion seeded at either end reads it the same way.
  The weight is the Jaccard index of the two topics' item sets, `|A ∩ B| / |A ∪ B|`, so a topic
  assigned to half the corpus does not outrank a tight pair merely by overlapping everything.
  `shared_items` keeps the raw `|A ∩ B|` beside it.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping

from xbrain.knowledge.contracts import GraphEdge
from xbrain.knowledge.ids import topic_id
from xbrain.models import Enrichment, Item

ASSIGNMENT_METHOD = "enrichment_assignment"
CO_OCCURRENCE_METHOD = "jaccard_topic_co_occurrence"

# The (source node type, target node type) of every relation `GraphEdge` admits. There is NO
# item–item relation: two items are related only THROUGH a topic both carry, so a consumer
# always sees which assignment connects them (spec §6.2). A relation added to the contract
# without an entry here turns `test_no_item_to_item_edge_exists_in_the_schema` red.
RELATION_ENDPOINTS: dict[str, tuple[str, str]] = {
    "HAS_PRIMARY_TOPIC": ("item", "topic"),
    "HAS_TOPIC": ("item", "topic"),
    "CO_OCCURS_WITH": ("topic", "topic"),
}


def _assigned_topics(enriched: Enrichment) -> tuple[str, ...]:
    """The item's topics, primary first, each slug once."""
    head = (enriched.primary_topic,) if enriched.primary_topic else ()
    return tuple(dict.fromkeys((*head, *enriched.topics)))


def _support_fingerprint(store: Mapping[str, Item], item_ids: tuple[str, ...]) -> str:
    """sha256 over the topic assignments of EVERY supporting item, in id order.

    Hashes the assignment (primary + topics), not the id list: re-assigning a supporting item's
    topics must move the fingerprint even when the pair's support set is unchanged.
    """
    atoms = []
    for item_id in item_ids:
        enriched = store[item_id].enriched
        assert enriched is not None  # only enriched items can support an edge
        atoms.append([item_id, enriched.primary_topic, list(enriched.topics)])
    blob = json.dumps(atoms, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_graph_edges(
    store: Mapping[str, Item],
    *,
    min_shared_items: int = 1,
    max_supporting_item_ids: int | None = None,
) -> list[GraphEdge]:
    """Every assignment edge and every `CO_OCCURS_WITH` edge the store's enrichments imply.

    `min_shared_items` prunes co-occurrence pairs that share fewer items; assignment edges are
    never pruned. A value below 1 behaves as 1, since a pair sharing nothing is not an edge.

    `max_supporting_item_ids` caps the ids an edge LISTS, never what it DECLARES: `shared_items`
    stays the full `|A ∩ B|`, and the weight and the fingerprint are computed over the whole
    support, so a truncated edge is still distinguishable from a thin one.
    """
    edges: list[GraphEdge] = []
    members: dict[str, set[str]] = defaultdict(set)
    for item_id in sorted(store):
        enriched = store[item_id].enriched
        if enriched is None:
            continue
        source = f"item:{item_id}"
        for slug in _assigned_topics(enriched):
            members[slug].add(item_id)
            relation = "HAS_PRIMARY_TOPIC" if slug == enriched.primary_topic else "HAS_TOPIC"
            edges.append(
                GraphEdge(
                    source=source,
                    target=topic_id(slug),
                    relation=relation,
                    method=ASSIGNMENT_METHOD,
                )
            )

    slugs = sorted(members)
    for i, a in enumerate(slugs):
        for b in slugs[i + 1 :]:
            shared = members[a] & members[b]
            if len(shared) < max(min_shared_items, 1):
                continue
            support = tuple(sorted(shared))
            weight = len(shared) / len(members[a] | members[b])
            fingerprint = _support_fingerprint(store, support)
            for left, right in ((a, b), (b, a)):
                edges.append(
                    GraphEdge(
                        source=topic_id(left),
                        target=topic_id(right),
                        relation="CO_OCCURS_WITH",
                        method=CO_OCCURRENCE_METHOD,
                        weight=weight,
                        shared_items=len(shared),
                        supporting_item_ids=support[:max_supporting_item_ids],
                        input_fingerprints=(fingerprint,),
                    )
                )
    return edges
