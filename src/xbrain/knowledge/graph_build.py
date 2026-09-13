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

# The index's defaults for the three `[index].graph_*` settings and the support cap: ONE
# definition, imported by `config.py` and `index_build.IndexOptions` (rule 5). UNSWEPT starting
# points: Plan 04 §1.3 fixes the threshold by a sweep of `min_shared_items ∈ {2, 3, 5, 8}` ×
# `min_weight ∈ {0.0, 0.02, 0.05, 0.10}` that has not run, so these are that sweep's most
# permissive corner, not its winner. 10 and 20 are spec §6.3's bounds.
DEFAULT_GRAPH_MIN_SHARED_ITEMS = 2
DEFAULT_GRAPH_MIN_WEIGHT = 0.0
DEFAULT_GRAPH_MAX_NEIGHBORS_PER_NODE = 10
MAX_SUPPORTING_ITEM_IDS = 20

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
    min_weight: float = 0.0,
    max_neighbors_per_node: int | None = None,
    max_supporting_item_ids: int | None = None,
    input_fingerprints: tuple[str, ...] = (),
) -> list[GraphEdge]:
    """Every assignment edge and every `CO_OCCURS_WITH` edge the store's enrichments imply.

    `min_shared_items` prunes co-occurrence pairs that share fewer items; assignment edges are
    never pruned. A value below 1 behaves as 1, since a pair sharing nothing is not an edge.
    `min_weight` prunes pairs whose Jaccard falls below it. `max_neighbors_per_node` keeps, per
    SOURCE topic, only its strongest co-occurrence edges (weight, then shared items, then target
    id), so a topic's outgoing list is bounded even when it overlaps everything; because it
    ranks per source, `a → b` can survive while `b → a` is cut from a busier `b`.

    `max_supporting_item_ids` caps the ids an edge LISTS, never what it DECLARES: `shared_items`
    stays the full `|A ∩ B|`, and the weight and the fingerprint are computed over the whole
    support, so a truncated edge is still distinguishable from a thin one.

    `input_fingerprints` are the caller's fingerprints of the OTHER planes the graph is read
    beside (the index passes vocabulary and topic pages), appended after the support's own on
    every `CO_OCCURS_WITH` edge. They are passed in rather than computed here because they are
    `index_build`'s definitions, and `index_build` imports this module.
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

    outgoing: dict[str, list[GraphEdge]] = defaultdict(list)
    slugs = sorted(members)
    for i, a in enumerate(slugs):
        for b in slugs[i + 1 :]:
            shared = members[a] & members[b]
            if len(shared) < max(min_shared_items, 1):
                continue
            weight = len(shared) / len(members[a] | members[b])
            if weight < min_weight:
                continue
            support = tuple(sorted(shared))
            fingerprint = _support_fingerprint(store, support)
            for left, right in ((a, b), (b, a)):
                outgoing[left].append(
                    GraphEdge(
                        source=topic_id(left),
                        target=topic_id(right),
                        relation="CO_OCCURS_WITH",
                        method=CO_OCCURRENCE_METHOD,
                        weight=weight,
                        shared_items=len(shared),
                        supporting_item_ids=support[:max_supporting_item_ids],
                        input_fingerprints=(fingerprint, *input_fingerprints),
                    )
                )
    for slug in slugs:
        ranked = sorted(outgoing[slug], key=lambda e: (-e.weight, -e.shared_items, e.target))
        edges.extend(ranked[:max_neighbors_per_node])
    return edges
