"""`graph_expand` as a SERVICE (Plan 04.3, spec §6.3, §7.4) — the first consumer of the graph.

It reads `graph_edges` through the same query door `search` uses (`open_for_query`), so an
index the code cannot answer honestly is refused here too, and it never writes the store or
the index.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from xbrain.knowledge.contracts import GraphEdge, GraphExpansionResponse
from xbrain.knowledge.index_store import open_for_query
from xbrain.knowledge.search_service import QueryContext

_EDGE_COLUMNS = (
    "source, target, relation, method, weight, shared_items, "
    "supporting_item_ids_json, input_fingerprints_json"
)


def _edge(row: Sequence[object]) -> GraphEdge:
    source, target, relation, method, weight, shared, support, fingerprints = row
    return GraphEdge(
        source=str(source),
        target=str(target),
        relation=relation,  # type: ignore[arg-type]  # the CHECKed column GraphEdge re-validates
        method=str(method),
        weight=float(weight),  # type: ignore[arg-type]
        shared_items=int(shared),  # type: ignore[call-overload]
        supporting_item_ids=tuple(json.loads(str(support))),
        input_fingerprints=tuple(json.loads(str(fingerprints))),
    )


def graph_expand(
    seeds: Sequence[str],
    context: QueryContext,
    *,
    max_hops: int = 1,
) -> GraphExpansionResponse:
    """Expand `seeds` over the persisted graph, keeping every edge's relation as stored."""
    index = open_for_query(
        context.index_dir,
        context.items_path,
        context.vocab_path,
        context.topics_path,
        params=context.params,
    )
    try:
        connection = index.lexical.connection
        edges: list[GraphEdge] = []
        for seed in seeds:
            rows = connection.execute(
                f"SELECT {_EDGE_COLUMNS} FROM graph_edges WHERE source = ? "
                "ORDER BY target, relation",
                (seed,),
            )
            edges.extend(_edge(row) for row in rows)
    finally:
        index.close()
    return GraphExpansionResponse(seeds=tuple(seeds), edges=tuple(edges))
