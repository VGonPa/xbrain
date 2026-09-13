"""Plan 04.4 — la estrategia ``hybrid_graph``.

Paso 20: existe, es desactivable, y el default NO cambia.
Paso 18: la vecindad del grafo sola no admite a un candidato.
Conexión: `search(strategy="hybrid_graph")` sirve el orden de `rank_with_graph`.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast, get_args

from tests.test_knowledge_search_service import FIXTURES, _build, _context, _persist
from xbrain.knowledge import fusion, graph_strategy, search_service
from xbrain.knowledge.contracts import GraphExpansionResponse, GraphNode, Strategy
from xbrain.knowledge.graph_service import graph_expand
from xbrain.knowledge.search_service import QueryContext
from xbrain.models import Item, Topic, TopicPage


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


def test_matched_by_anade_graph_a_los_canales_que_ya_lo_encontraron() -> None:
    # Paso 19. `b` llega por léxico y `c` por léxico y vector; los dos son además vecinos de la
    # semilla `a`. El mapping se construye con `vector` PRIMERO, así que si el orden saliera del
    # caller y no del contrato (`Channel`, el que fija fusion), `c` diría vector antes que léxico.
    calls: list[tuple[str, ...]] = []
    ranked = graph_strategy.rank_with_graph(
        {"vector": ["a", "c"], "lexical": ["a", "b", "c"]},
        _CONTEXT,
        seeds=1,
        limit=10,
        expand=_expansion(["item:b", "item:c"], calls),
    )

    matched_by = {r.item_id: r.matched_by for r in ranked}
    assert calls == [("item:a",)]
    assert matched_by["b"] == ("lexical", "graph")
    assert matched_by["c"] == ("lexical", "vector", "graph")
    # la semilla no es vecina de sí misma: conserva sus canales, sin `graph`.
    assert matched_by["a"] == ("lexical", "vector")


def test_un_item_en_el_puesto_40_del_lexico_que_llega_por_vecindad_entra_en_el_top_10() -> None:
    # Paso 18b. Los canales se puntúan sobre el conjunto COMPLETO de candidatos: puntuar sobre
    # el top-k ya cortado deja fuera a `i40` antes de que el grafo pueda levantarlo, y el delta
    # del grafo sale 0 POR CONSTRUCCIÓN.
    lexical = [f"i{n:02d}" for n in range(1, 51)]
    calls: list[tuple[str, ...]] = []

    # La fixture es lo que dice ser: `i40` está REALMENTE en el puesto 40, y la fusión sin grafo
    # lo deja fuera del top-10. Si ya estuviera dentro, entrar no probaría nada (regla 1).
    assert lexical.index("i40") + 1 == 40
    assert "i40" not in {chunk.chunk_id for chunk in fusion.fuse({"lexical": lexical})[:10]}

    ranked = graph_strategy.rank_with_graph(
        {"lexical": lexical},
        _CONTEXT,
        seeds=1,
        limit=10,
        expand=_expansion(["topic:t", "item:i40"], calls),
    )

    assert calls == [("item:i01",)]
    assert len(ranked) == 10
    lifted = {r.item_id: r for r in ranked}
    assert "i40" in lifted
    assert lifted["i40"].matched_by == ("lexical", "graph")


def test_search_hybrid_graph_sirve_un_item_traido_por_vecindad_fuera_del_top_k_lexico(
    tmp_path: Path,
) -> None:
    # Conexión. Un índice REAL (plano del grafo incluido) y `graph_expand` REAL, sin dobles
    # (regla 3): `search` con el grafo encendido sirve lo que `rank_with_graph` ordena.
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    store = {k: Item.model_validate(v) for k, v in raw["items"].items()}
    vocab = [Topic.model_validate(v) for v in raw["vocab"].values()]
    pages = {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()}
    data = tmp_path / "data"
    _persist(data, store, vocab, pages)
    _build(data)
    context = _context(data, store, vocab, pages)

    # La fixture es lo que dice ser: `k07` casa en léxico, pero FUERA del top-2, y es vecino de la
    # cabeza. Si ya estuviera en el top-2, verlo en la página no probaría nada (regla 1).
    ranking = [r.item_id for r in search_service.search("export", context, limit=50).results]
    top_k = [r.item_id for r in search_service.search("export", context, limit=2).results]
    assert "k07" in ranking
    assert "k07" not in top_k
    reached = graph_expand([f"item:{top_k[0]}"], context, max_hops=graph_strategy.GRAPH_MAX_HOPS)
    assert "item:k07" in {node.node_id for node in reached.nodes}

    graph = search_service.search(
        "export", context, limit=2, strategy="hybrid_graph", graph_enabled=True
    )

    assert graph.strategy == "hybrid_graph"
    assert "hybrid_graph_not_implemented" not in graph.index.degraded
    assert "k07" in [r.item_id for r in graph.results]


def test_search_hybrid_graph_sirve_graph_en_matched_by_del_item_elevado(tmp_path: Path) -> None:
    # Paso 19 A TRAVÉS DE `search`, no de `rank_with_graph`: la función ya añadía `graph`, y
    # `_graph_order` lo tiraba al volver — el item elevado salía con `("lexical",)`.
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    store = {k: Item.model_validate(v) for k, v in raw["items"].items()}
    vocab = [Topic.model_validate(v) for v in raw["vocab"].values()]
    pages = {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()}
    data = tmp_path / "data"
    _persist(data, store, vocab, pages)
    _build(data)
    context = _context(data, store, vocab, pages)

    # La fixture es lo que dice ser: `k07` casa en léxico FUERA del top-2 y la estrategia lo sube.
    top_k = [r.item_id for r in search_service.search("export", context, limit=2).results]
    assert "k07" not in top_k

    graph = search_service.search(
        "export", context, limit=2, strategy="hybrid_graph", graph_enabled=True
    )

    lifted = {r.item_id: r for r in graph.results}["k07"]
    assert lifted.matches
    # `graph` AÑADIDO al canal que ya lo traía, en el orden del contrato que fija fusion.
    assert fusion._CHANNEL_ORDER.index("lexical") < fusion._CHANNEL_ORDER.index("graph")
    assert {m.matched_by for m in lifted.matches} == {("lexical", "graph")}


def test_search_hybrid_graph_admite_un_vecino_que_cae_fuera_de_la_ventana_de_la_pagina(
    tmp_path: Path,
) -> None:
    # Puerta 11.13. `search` materializaba `offset + limit + 1` dueños y SOLO ENTONCES llamaba al
    # grafo: un vecino fuera de esa ventana no podía entrar y el delta del grafo salía 0 por
    # construcción. El test anterior no lo ve — `k07` está en el puesto 3 con `limit=2`, DENTRO
    # de la ventana `limit + 1`.
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    store = {k: Item.model_validate(v) for k, v in raw["items"].items()}
    vocab = [Topic.model_validate(v) for v in raw["vocab"].values()]
    pages = {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()}
    data = tmp_path / "data"
    _persist(data, store, vocab, pages)
    _build(data)
    context = _context(data, store, vocab, pages)

    # La fixture es lo que dice ser: con `limit=1` la ventana es de 2 dueños, `k03` es el 3º del
    # léxico — fuera — y es vecino de la cabeza, desde la que el grafo se expande.
    ranking = [r.item_id for r in search_service.search("policy", context, limit=50).results]
    assert ranking.index("k03") + 1 == 3
    reached = graph_expand([f"item:{ranking[0]}"], context, max_hops=graph_strategy.GRAPH_MAX_HOPS)
    assert "item:k03" in {node.node_id for node in reached.nodes}

    graph = search_service.search(
        "policy", context, limit=1, strategy="hybrid_graph", graph_enabled=True
    )

    assert graph.strategy == "hybrid_graph"
    assert [r.item_id for r in graph.results] == ["k03"]
