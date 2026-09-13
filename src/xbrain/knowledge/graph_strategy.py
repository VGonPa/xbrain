"""The `hybrid_graph` strategy (Plan 04.4): the fused ranking, widened by graph neighbourhood.

OPT-IN, and it stays that way. `search` keeps `lexical` as its default, and the graph channel
runs only when the caller asks for `hybrid_graph` AND the switch is on: a strategy that cannot be
turned off cannot be measured against the ranking it claims to improve.

**The graph re-orders, it never admits.** A neighbour of a seed that no channel scored is not a
result: co-occurrence in the corpus says two items share topics, not that the second answers the
query (spec §6.4). So the graph can lift a candidate a channel found, and cannot conjure one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from xbrain.knowledge import fusion
from xbrain.knowledge.contracts import Channel, GraphExpansionResponse, Strategy
from xbrain.knowledge.fusion import FusedChunk
from xbrain.knowledge.graph_service import graph_expand
from xbrain.knowledge.search_service import QueryContext

GRAPH_STRATEGY: Final[Strategy] = "hybrid_graph"

# The switch is born OFF. Promoting the graph is a decision the golden set makes, not a default.
GRAPH_ENABLED_BY_DEFAULT: Final[bool] = False

# The graph channel's RRF weight — a starting point like `fusion.CHANNEL_WEIGHTS`, read at call
# time so a measured value changes the ranking.
GRAPH_WEIGHT: float = 1.0

# item → topic → item: one hop only reaches the seed's topics, never another item.
GRAPH_MAX_HOPS: Final[int] = 2

_ITEM_NODE: Final[str] = "item:"


@dataclass(frozen=True)
class GraphRankedItem:
    """One item of the `hybrid_graph` ranking: its id, who found it, its (uncalibrated) signal."""

    item_id: str
    matched_by: tuple[Channel, ...]
    score: float


def graph_channel_runs(requested: str, *, enabled: bool) -> bool:
    """Whether the graph channel runs for this request: asked for by name, and switched on."""
    return enabled and requested == GRAPH_STRATEGY


def rank_with_graph(
    rankings: Mapping[Channel, Sequence[str]],
    context: QueryContext,
    *,
    seeds: int,
    limit: int,
    expand: Callable[..., GraphExpansionResponse] = graph_expand,
) -> tuple[GraphRankedItem, ...]:
    """Fuse the channel rankings of item ids, then lift the seeds' graph neighbours, best first.

    The seeds are the `seeds` best fused items; `expand` is `graph_expand`, injectable only so a
    test can hand it a fixture envelope instead of a built index.
    """
    scored = fusion.fuse(rankings)[:limit]
    response = _expand_seeds(scored[:seeds], context, expand)
    graph_ranks = _collect_candidates(response)
    return _merge(scored, graph_ranks, _rescore(scored, graph_ranks))[:limit]


def _expand_seeds(
    heads: Sequence[FusedChunk],
    context: QueryContext,
    expand: Callable[..., GraphExpansionResponse],
) -> GraphExpansionResponse:
    """`graph_expand` from the head of the fusion, as item nodes."""
    seeds = [f"{_ITEM_NODE}{head.chunk_id}" for head in heads]
    return expand(seeds, context, max_hops=GRAPH_MAX_HOPS)


def _collect_candidates(response: GraphExpansionResponse) -> dict[str, int]:
    """`{item_id: graph_rank}` for every reached item that is not a seed, in reach order."""
    seeds = set(response.seeds)
    reached = [
        node.node_id.removeprefix(_ITEM_NODE)
        for node in response.nodes
        if node.node_type == "item" and node.node_id not in seeds
    ]
    return {item_id: rank for rank, item_id in enumerate(reached, start=1)}


def _rescore(scored: Sequence[FusedChunk], graph_ranks: Mapping[str, int]) -> dict[str, float]:
    """The fused score plus the graph's RRF term — for channel-scored items and nothing else."""
    k, weight = fusion.RRF_K, GRAPH_WEIGHT
    scores = {chunk.chunk_id: chunk.score for chunk in scored}
    for item_id, rank in graph_ranks.items():
        if item_id in scores:
            scores[item_id] += weight / (k + rank)
    return scores


def _merge(
    scored: Sequence[FusedChunk], graph_ranks: Mapping[str, int], scores: Mapping[str, float]
) -> tuple[GraphRankedItem, ...]:
    """The ranked items, best first, ties broken by `item_id`.

    `graph` is ADDED to the channels that found an item, never put in their place, and named in
    fusion's contract order — the graph explains a lift, it does not erase who found the item.
    """
    fused_by = {chunk.chunk_id: chunk.matched_by for chunk in scored}
    merged = []
    for item_id, score in scores.items():
        channels = {*fused_by.get(item_id, ()), *(("graph",) if item_id in graph_ranks else ())}
        matched_by = tuple(ch for ch in fusion._CHANNEL_ORDER if ch in channels)
        merged.append(GraphRankedItem(item_id, matched_by, score))  # type: ignore[arg-type]
    merged.sort(key=lambda item: (-item.score, item.item_id))
    return tuple(merged)
