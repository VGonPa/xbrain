"""The `hybrid_graph` strategy (Plan 04.4): the fused ranking, widened by graph neighbourhood.

OPT-IN, and it stays that way. `search` keeps `lexical` as its default, and the graph channel
runs only when the caller asks for `hybrid_graph` AND the switch is on: a strategy that cannot be
turned off cannot be measured against the ranking it claims to improve.
"""

from __future__ import annotations

from typing import Final

from xbrain.knowledge.contracts import Strategy

GRAPH_STRATEGY: Final[Strategy] = "hybrid_graph"

# The switch is born OFF. Promoting the graph is a decision the golden set makes, not a default.
GRAPH_ENABLED_BY_DEFAULT: Final[bool] = False


def graph_channel_runs(requested: str, *, enabled: bool) -> bool:
    """Whether the graph channel runs for this request: asked for by name, and switched on."""
    return enabled and requested == GRAPH_STRATEGY
