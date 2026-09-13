"""Plan 04.4 — la estrategia ``hybrid_graph``.

Paso 20: existe, es desactivable, y el default NO cambia.
Paso 18: la vecindad del grafo sola no admite a un candidato.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import cast, get_args

from xbrain.knowledge import graph_strategy, search_service
from xbrain.knowledge.contracts import GraphExpansionResponse, GraphNode, Strategy
from xbrain.knowledge.search_service import QueryContext


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


# The context is never opened: the fake `expand` below stands in for `graph_expand`, which is
# the only consumer of it. A sentinel makes any other use of it fail loudly.
_CONTEXT = cast(QueryContext, object())


def _expansion(
    reached: Sequence[str], calls: list[tuple[str, ...]]
) -> Callable[..., GraphExpansionResponse]:
    """A `graph_expand` stand-in: the seeds, then `reached` in reach order, as the real envelope."""

    def expand(seeds: Sequence[str], context: QueryContext, **_: object) -> GraphExpansionResponse:
        assert context is _CONTEXT
        calls.append(tuple(seeds))
        node_ids = (*seeds, *reached)
        return GraphExpansionResponse(
            seeds=tuple(seeds),
            nodes=tuple(
                GraphNode(node_id=node, node_type="topic" if node.startswith("topic:") else "item")
                for node in node_ids
            ),
        )

    return expand


def test_un_candidato_solo_grafo_que_no_puntua_en_ningun_canal_no_entra() -> None:
    # Paso 18. `fantasma` es vecino de la semilla `a` pero NINGÚN canal lo rankea. Hay sitio
    # de sobra en la página (limit=10, tres candidatos), así que si la vecindad bastara para
    # entrar, entraría: su ausencia la decide la regla, no la falta de hueco.
    calls: list[tuple[str, ...]] = []
    ranked = graph_strategy.rank_with_graph(
        {"lexical": ["a", "b", "c"]},
        _CONTEXT,
        seeds=1,
        limit=10,
        expand=_expansion(["topic:t", "item:fantasma", "item:c"], calls),
    )

    # La semilla es la cabeza de la fusión, y el grafo se expande DESDE ella.
    assert calls == [("item:a",)]
    # El grafo SÍ corre — `c`, tercero en léxico y vecino de la semilla, sube por encima de
    # ella — así que la ausencia de `fantasma` no es la de un canal que no se ejecutó.
    assert [r.item_id for r in ranked] == ["c", "a", "b"]
    assert "fantasma" not in {r.item_id for r in ranked}
