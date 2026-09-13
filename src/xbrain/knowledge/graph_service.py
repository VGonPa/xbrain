"""`graph_expand` as a SERVICE (Plan 04.3, spec §6.3, §7.4) — the first consumer of the graph.

It reads `graph_edges` through the same query door `search` uses (`open_for_query`), so an
index the code cannot answer honestly is refused here too, and it never writes the store or
the index.

EDGES ARE RETURNED AS STORED, never re-oriented. An assignment edge always reads
`item → topic` and an expansion seeded at a topic reaches the item by walking it backwards; a
`CO_OCCURS_WITH` edge is stored in both directions, so walking it forwards is enough. A path
therefore names its nodes in the order they were REACHED, while each edge keeps the direction
`graph_build` gave it — the relation is what the consumer reads, never the arrow.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Literal

from xbrain.knowledge.contracts import GraphEdge, GraphExpansionResponse, GraphNode, GraphPath
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


def _incident(connection: sqlite3.Connection, node_id: str) -> list[GraphEdge]:
    """Every edge leaving `node_id`, plus the assignment edges arriving at it."""
    rows = connection.execute(
        f"SELECT {_EDGE_COLUMNS} FROM graph_edges WHERE source = ? "
        "UNION ALL "
        f"SELECT {_EDGE_COLUMNS} FROM graph_edges WHERE target = ? "
        "AND relation != 'CO_OCCURS_WITH' "
        "ORDER BY source, target, relation",
        (node_id, node_id),
    )
    return [_edge(row) for row in rows]


def _node_type(node_id: str) -> Literal["item", "topic"]:
    return "item" if node_id.startswith("item:") else "topic"


def _support_ids(edge: GraphEdge) -> tuple[str, ...]:
    """The store ids an edge rests on: its listed support, or the item it assigns."""
    if edge.relation == "CO_OCCURS_WITH":
        return edge.supporting_item_ids
    return (edge.source.removeprefix("item:"),)


def _require_resolvable(edge: GraphEdge, store: Mapping[str, object]) -> None:
    """Refuse an edge whose support is no longer in the LIVE store.

    The graph is derived and the store is the truth: an id the index lists but the store no
    longer holds is one `get` cannot open, so serving it would cite evidence that does not
    exist. The contract has no field to declare an excluded edge, so the refusal is WHOLE and
    names the repair — dropping it quietly would be the silent cut spec §9.3 forbids.
    """
    missing = [item_id for item_id in _support_ids(edge) if item_id not in store]
    if missing:
        raise ValueError(
            f"La arista {edge.source} → {edge.target} ({edge.relation}) se apoya en items que "
            f"ya no están en el store: {missing!r}. El grafo del índice va por detrás del "
            "store: ejecuta `xbrain index update`."
        )


def graph_expand(
    seeds: Sequence[str],
    context: QueryContext,
    *,
    max_hops: int = 1,
) -> GraphExpansionResponse:
    """Expand `seeds` over the persisted graph, one explicit path per reached node.

    Every served edge's support resolves in `context.store` (`_require_resolvable`).
    """
    index = open_for_query(
        context.index_dir,
        context.items_path,
        context.vocab_path,
        context.topics_path,
        params=context.params,
    )
    try:
        connection = index.lexical.connection
        reached: dict[str, None] = dict.fromkeys(seeds)
        edges: dict[tuple[str, str, str], GraphEdge] = {}
        paths: list[GraphPath] = []
        for seed in seeds:
            for edge in _incident(connection, seed):
                _require_resolvable(edge, context.store)
                edges[(edge.source, edge.target, edge.relation)] = edge
                other = edge.target if edge.source == seed else edge.source
                if other in reached:
                    continue
                reached[other] = None
                paths.append(GraphPath(nodes=(seed, other), edges=(edge,)))
    finally:
        index.close()
    return GraphExpansionResponse(
        seeds=tuple(seeds),
        nodes=tuple(GraphNode(node_id=n, node_type=_node_type(n)) for n in reached),
        edges=tuple(edges.values()),
        paths=tuple(paths),
    )
