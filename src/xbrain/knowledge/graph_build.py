"""The minimal graph (Plan 04.2, spec §6.2): item→topic assignment edges.

An edge here records that xbrain ASSIGNED a topic to an item — never that the concepts are
related in the world (spec §6.4). The shape is `contracts.GraphEdge`; this module only derives
edges from the store and never writes it.
"""

from __future__ import annotations

from collections.abc import Mapping

from xbrain.knowledge.contracts import GraphEdge
from xbrain.knowledge.ids import topic_id
from xbrain.models import Item

ASSIGNMENT_METHOD = "enrichment_assignment"


def build_graph_edges(store: Mapping[str, Item]) -> list[GraphEdge]:
    """Every `HAS_PRIMARY_TOPIC` and `HAS_TOPIC` edge the store's enrichments assign."""
    edges: list[GraphEdge] = []
    for item_id in sorted(store):
        enriched = store[item_id].enriched
        if enriched is None:
            continue
        source = f"item:{item_id}"
        if enriched.primary_topic:
            edges.append(
                GraphEdge(
                    source=source,
                    target=topic_id(enriched.primary_topic),
                    relation="HAS_PRIMARY_TOPIC",
                    method=ASSIGNMENT_METHOD,
                )
            )
        for slug in enriched.topics:
            edges.append(
                GraphEdge(
                    source=source,
                    target=topic_id(slug),
                    relation="HAS_TOPIC",
                    method=ASSIGNMENT_METHOD,
                )
            )
    return edges
