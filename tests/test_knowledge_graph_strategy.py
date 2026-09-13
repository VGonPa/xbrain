"""Plan 04.4 — la estrategia ``hybrid_graph``.

Paso 20: existe, es desactivable, y el default NO cambia.
"""

from __future__ import annotations

import inspect
from typing import get_args

from xbrain.knowledge import graph_strategy, search_service
from xbrain.knowledge.contracts import Strategy


def test_hybrid_graph_existe_es_desactivable_y_el_default_no_cambia() -> None:
    # EXISTE: el nombre que declara el módulo es el del contrato, no una copia.
    assert graph_strategy.GRAPH_STRATEGY == "hybrid_graph"
    assert graph_strategy.GRAPH_STRATEGY in get_args(Strategy)

    # DESACTIVABLE: con el grafo apagado, pedir hybrid_graph NO corre el grafo.
    assert graph_strategy.graph_channel_runs("hybrid_graph", enabled=True) is True
    assert graph_strategy.graph_channel_runs("hybrid_graph", enabled=False) is False
    # y ninguna otra estrategia lo enciende, esté o no habilitado.
    for other in ("lexical", "vector", "hybrid"):
        assert graph_strategy.graph_channel_runs(other, enabled=True) is False

    # EL DEFAULT NO CAMBIA: el grafo nace apagado, y `search` sin estrategia sigue en lexical.
    assert graph_strategy.GRAPH_ENABLED_BY_DEFAULT is False
    default = inspect.signature(search_service.search).parameters["strategy"].default
    assert default == "lexical"
