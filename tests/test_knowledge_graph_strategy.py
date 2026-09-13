"""Plan 04.4 — la estrategia ``hybrid_graph``.

Paso 20: existe, es desactivable, y el default NO cambia.
Paso 18: la vecindad del grafo sola no admite a un candidato.
Conexión: `search(strategy="hybrid_graph")` sirve el orden de `rank_with_graph`.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast, get_args

import pytest

from tests import test_knowledge_search_hybrid as vector_fixture
from tests.test_knowledge_search_service import FIXTURES, _build, _context, _persist
from xbrain.knowledge import fusion, graph_strategy, search_service
from xbrain.knowledge.contracts import GraphExpansionResponse, GraphNode, Strategy
from xbrain.knowledge.graph_service import graph_expand
from xbrain.knowledge.search_service import QueryContext
from xbrain.models import Author, Content, Enrichment, Item, Topic, TopicPage

_T = datetime(2026, 1, 1, tzinfo=UTC)


# Chunks per item. `search_owners` opens `owners * OWNER_CHUNK_MULTIPLIER` (4) ROWS, so with one
# matching chunk per item a window sized for 11 owners already held 44 of them and the 40th was
# inside it under every too-small horizon. Six chunks each put 39 × 6 = 234 rows in front of the
# 40th item — more than the widest mutated window (22 owners × 4 = 88 rows) can reach.
_CHUNKS_PER_ITEM = 6


def _ranked_item(item_id: str, filler: int, topic: str | None) -> Item:
    """An item whose article matches `zeta` in every chunk; more `filler` = a lower bm25."""
    paragraph = "zeta " + " ".join(["relleno"] * (80 + filler))
    text = "\n\n".join([paragraph] * _CHUNKS_PER_ITEM)
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/u/status/{item_id}",
        author=Author(handle="u", name="U"),
        text=f"tweet {item_id}",
        created_at=_T,
        captured_at=_T,
        content=Content.model_validate(
            {
                "fetched_at": _T.isoformat(),
                "sources": [
                    {
                        "outcome": "success",
                        "kind": "x_article",
                        "url": f"https://x.com/i/article/{item_id}",
                        "text": text,
                        "attempts": 1,
                        "title": f"Article {item_id}",
                    }
                ],
            }
        ),
        enriched=Enrichment(
            enriched_at=_T,
            executor="manual",
            summary="s",
            primary_topic=topic,
            topics=[],
        ),
    )


def _ranked_context(tmp_path: Path, store: dict[str, Item], slugs: Sequence[str]) -> QueryContext:
    """A REAL index (graph plane included) over `store`, and the context `search` reads."""
    vocab = [Topic(slug=slug, description=f"topic {slug}") for slug in slugs]
    data = tmp_path / "data"
    _persist(data, store, vocab, {})
    _build(data)
    return _context(data, store, vocab, {})


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


def test_search_hybrid_graph_con_limit_10_sirve_al_vecino_del_puesto_40_del_lexico(
    tmp_path: Path,
) -> None:
    # Puerta 11.13 atada al CONTRATO, no a un mínimo. El test de `k03` solo exigía un horizonte
    # >= limit + 2, así que `needed = beyond + 1`, `needed = 2 * beyond` y un horizonte de 3
    # pasaban con el vecino del puesto 40 FUERA. Aquí el vecino está en el 40 y la página es 10.
    store = {
        f"u{n:02d}": _ranked_item(f"u{n:02d}", filler=n, topic="hub" if n in (1, 40) else None)
        for n in range(1, 51)
    }
    context = _ranked_context(tmp_path, store, ["hub"])

    # La fixture es lo que dice ser: `u40` es el 40º del léxico, fuera del top-10, y es vecino
    # de la cabeza desde la que el grafo se expande (regla 1).
    ranking = [r.item_id for r in search_service.search("zeta", context, limit=50).results]
    assert len(ranking) == 50
    assert ranking.index("u40") + 1 == 40
    reached = graph_expand([f"item:{ranking[0]}"], context, max_hops=graph_strategy.GRAPH_MAX_HOPS)
    assert "item:u40" in {node.node_id for node in reached.nodes}

    graph = search_service.search(
        "zeta", context, limit=10, strategy="hybrid_graph", graph_enabled=True
    )

    assert graph.strategy == "hybrid_graph"
    assert "u40" in [r.item_id for r in graph.results]


def test_search_hybrid_graph_pagina_por_encima_del_horizonte_sin_duplicar_ni_perder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Por encima del horizonte el grafo reordenaba lo que la PÁGINA había materializado, así que
    # cada página cortaba su propio ranking: con 4,102 items y las constantes reales, `limit=500`
    # duplicaba un id y perdía otro al cruzar el dueño 1,000. El mecanismo no depende de cuánto
    # vale el horizonte, así que se baja a 4 para cruzarlo con 50 items (se lee en cada llamada).
    monkeypatch.setattr(search_service, "GRAPH_CANDIDATE_HORIZON", 4)
    store = {
        f"u{n:02d}": _ranked_item(f"u{n:02d}", filler=n, topic="hub" if n in (1, 3, 40) else None)
        for n in range(1, 51)
    }
    context = _ranked_context(tmp_path, store, ["hub"])

    # La fixture es lo que dice ser: `u03` (dentro del horizonte) y `u40` (fuera) son vecinos de
    # la cabeza, así que el grafo SÍ reordena y hay un vecino que solo una página honda vería.
    lexical = [r.item_id for r in search_service.search("zeta", context, limit=50).results]
    assert lexical.index("u03") + 1 == 3
    assert lexical.index("u40") + 1 == 40
    reached = graph_expand([f"item:{lexical[0]}"], context, max_hops=graph_strategy.GRAPH_MAX_HOPS)
    assert {"item:u03", "item:u40"} <= {node.node_id for node in reached.nodes}

    def page(limit: int, cursor: str | None = None) -> tuple[list[str], str | None]:
        response = search_service.search(
            "zeta", context, limit=limit, strategy="hybrid_graph", graph_enabled=True, cursor=cursor
        )
        return [r.item_id for r in response.results], response.cursor

    whole, _ = page(50)
    assert whole[0] == "u03"  # el grafo actuó dentro del horizonte

    # EL ORDEN NO DEPENDE DEL LIMIT: cada página corta es un prefijo del ranking entero.
    for limit in (1, 3, 10, 39, 41):
        assert page(limit)[0] == whole[:limit], limit

    # LA UNIÓN DE LAS PÁGINAS ES EL RANKING: ni un id dos veces, ni uno perdido.
    walked: list[str] = []
    ids, cursor = page(7)
    walked += ids
    while cursor is not None:
        ids, cursor = page(7, cursor)
        walked += ids
    assert len(walked) == len(set(walked))
    assert set(walked) == set(store)
    assert walked == whole


def test_search_hybrid_graph_export_limit_1_reproduce_donde_queda_k07_y_por_que(
    tmp_path: Path,
) -> None:
    # La revisión esperaba `k07` en la página 1 de `export` con `limit=1`. Este test no lo
    # decide por un vecino "mejor": RECALCULA la puntuación RRF de cada candidato con la fórmula
    # de spec/Plan 03 §4 a partir de dos medidas independientes de `rank_with_graph` — el puesto
    # léxico que sirve `search` y el orden de alcance que devuelve `graph_expand` REAL — y exige
    # que `search`, página a página, sirva exactamente ese orden.
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    store = {k: Item.model_validate(v) for k, v in raw["items"].items()}
    vocab = [Topic.model_validate(v) for v in raw["vocab"].values()]
    pages = {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()}
    data = tmp_path / "data"
    _persist(data, store, vocab, pages)
    _build(data)
    context = _context(data, store, vocab, pages)

    # La fixture es lo que dice ser: `k07` es el 3º del léxico, FUERA de la ventana de 2 dueños
    # que `limit=1` materializaba, y es vecino de la cabeza.
    lexical = [r.item_id for r in search_service.search("export", context, limit=50).results]
    assert lexical.index("k07") + 1 == 3
    reached = graph_expand([f"item:{lexical[0]}"], context, max_hops=graph_strategy.GRAPH_MAX_HOPS)
    reach = [
        n.node_id.removeprefix("item:")
        for n in reached.nodes
        if n.node_type == "item" and n.node_id not in reached.seeds
    ]
    assert "k07" in reach

    k, weight = fusion.RRF_K, graph_strategy.GRAPH_WEIGHT
    expected_score = {
        item: 1 / (k + lexical.index(item) + 1)
        + (weight / (k + reach.index(item) + 1) if item in reach else 0.0)
        for item in lexical
    }
    expected = sorted(lexical, key=lambda item: (-expected_score[item], item))

    walked: list[str] = []
    cursor: str | None = None
    while True:
        response = search_service.search(
            "export", context, limit=1, strategy="hybrid_graph", graph_enabled=True, cursor=cursor
        )
        walked += [r.item_id for r in response.results]
        cursor = response.cursor
        if cursor is None:
            break

    # `search` sirve el orden de la fórmula, página a página.
    assert walked == expected
    # `k07` SÍ se eleva: del 3º léxico al puesto que la fórmula le da, delante de su puesto léxico.
    assert walked.index("k07") < lexical.index("k07")
    # Y NO es el 1º porque la fórmula pone a otro por encima, con la diferencia a la vista:
    # `k07` pierde en léxico Y en orden de alcance frente a quien encabeza.
    head = expected[0]
    assert head != "k07"
    assert lexical.index(head) < lexical.index("k07")
    assert reach.index(head) < reach.index("k07")
    assert expected_score[head] > expected_score["k07"]


def test_search_hybrid_graph_con_indice_por_detras_del_store_degrada_por_la_puerta_unica(
    tmp_path: Path,
) -> None:
    # Punto 4b. `graph_expand` REFUSA un índice por detrás del store (no tiene campo `degraded`),
    # y `search` le pasaba ese índice: pedir `hybrid_graph` lanzaba `ValueError` mientras
    # `lexical` y `hybrid` degradan declarándolo. Plan 04 §«Degradaciones»: `hybrid_graph`
    # responde declarando `index_behind_store`, nunca en silencio — y nunca con un grafo stale.
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    store = {k: Item.model_validate(v) for k, v in raw["items"].items()}
    vocab = [Topic.model_validate(v) for v in raw["vocab"].values()]
    pages = {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()}
    data = tmp_path / "data"
    _persist(data, store, vocab, pages)
    _build(data)
    context = _context(data, store, vocab, pages)
    items = data / "items.json"
    items.write_text(items.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    # La fixture es lo que dice ser: la señal barata YA declara el índice por detrás, y la
    # puerta del grafo lo rechaza (regla 1).
    lexical = search_service.search("export", context, limit=2)
    assert "index_behind_store" in lexical.index.degraded
    with pytest.raises(ValueError, match="index_behind_store"):
        graph_expand(["item:k03"], context, max_hops=graph_strategy.GRAPH_MAX_HOPS)

    graph = search_service.search(
        "export", context, limit=2, strategy="hybrid_graph", graph_enabled=True
    )

    # Responde lo que corrió — léxico — con la causa ya declarada por el índice, y sin `graph`.
    assert graph.strategy == "lexical"
    assert graph.index.degraded == lexical.index.degraded
    assert graph.results == lexical.results


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


def _vector_corpus() -> tuple[dict[str, Item], list[Topic], dict[str, TopicPage]]:
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    return (
        {k: Item.model_validate(v) for k, v in raw["items"].items()},
        [Topic.model_validate(v) for v in raw["vocab"].values()],
        {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()},
    )


def test_search_hybrid_graph_con_plano_vectorial_ejecuta_hybrid_y_el_grafo_sobre_ambos_canales(
    tmp_path: Path,
) -> None:
    # Plan 04 §3.1-§3.3: `hybrid_graph` EJECUTA `hybrid` y el grafo reordena candidatos puntuados por
    # léxico Y vector. Con el plano y el embedder DISPONIBLES, `search` corría solo léxico, no llamaba
    # al embedder y dejaba `degraded` vacío: una respuesta que se llama hybrid_graph sin su vector.
    corpus = _vector_corpus()
    data = vector_fixture._data(tmp_path, corpus, with_plane=True)
    target, text = vector_fixture._pick(data, contains_query=False)
    embedder = vector_fixture.QueryEmbedder(text)
    context = vector_fixture._context(data, corpus, embedder)
    query = vector_fixture.QUERY

    # La fixture es lo que dice ser (regla 1): `hybrid` corre el vector, `target` es un chunk que
    # SOLO el vector encuentra, y hay vecinos de la cabeza de `hybrid` que el vector trajo.
    hybrid = search_service.search(query, context, strategy="hybrid", limit=50)
    assert hybrid.strategy == "hybrid"
    assert embedder.calls == [query]
    hybrid_matches = {m.chunk_id: m for r in hybrid.results for m in r.matches}
    assert hybrid_matches[target].matched_by == ("vector",)
    reached = graph_expand(
        [f"item:{hybrid.results[0].item_id}"], context, max_hops=graph_strategy.GRAPH_MAX_HOPS
    )
    neighbours = {
        node.node_id.removeprefix("item:")
        for node in reached.nodes
        if node.node_type == "item" and node.node_id not in reached.seeds
    }
    vector_found = {
        r.item_id for r in hybrid.results if any("vector" in m.matched_by for m in r.matches)
    }
    assert neighbours & vector_found

    embedder.calls.clear()
    graph = search_service.search(
        query, context, limit=50, strategy="hybrid_graph", graph_enabled=True
    )

    # 1. el canal vectorial CORRIÓ: el embedder se llamó, y nada se declara degradado por su causa.
    assert embedder.calls == [query]
    assert graph.strategy == "hybrid_graph"
    assert not {
        search_service.EMBEDDINGS_NOT_CONFIGURED,
        search_service.EMBEDDER_UNAVAILABLE,
        "no_embeddings",
    } & set(graph.index.degraded)
    served = {m.chunk_id: m for r in graph.results for m in r.matches}
    assert served[target].matched_by == ("vector",)
    assert served[target].vector_rank is not None
    # 2. un vecino que el vector trajo lleva `vector` Y `graph`, junto a `lexical` si también lo
    # trajo, en el orden del contrato que fija fusion — medido contra el chunk que sirvió `hybrid`.
    lifted = [
        (match, hybrid_matches[match.chunk_id])
        for r in graph.results
        if r.item_id in neighbours & vector_found
        for match in r.matches
    ]
    assert lifted
    for match, before in lifted:
        channels = {*before.matched_by, "graph"}
        assert match.matched_by == tuple(c for c in fusion._CHANNEL_ORDER if c in channels)
    assert any({"vector", "graph"} <= set(match.matched_by) for match, _ in lifted)


@pytest.mark.parametrize(
    ("with_plane", "with_embedder", "cause"),
    [
        (True, False, search_service.EMBEDDINGS_NOT_CONFIGURED),
        (False, True, "no_embeddings"),
    ],
)
def test_search_hybrid_graph_sin_canal_vectorial_lo_declara_y_no_finge_vector(
    tmp_path: Path, with_plane: bool, with_embedder: bool, cause: str
) -> None:
    # Si el vector NO puede correr, `hybrid_graph` no calla: `degraded` NOMBRA la causa, por la
    # misma puerta que `hybrid`, y ningún match dice `vector`.
    corpus = _vector_corpus()
    data = vector_fixture._data(tmp_path, corpus, with_plane=with_plane)
    passage = vector_fixture._pick(data, contains_query=False)[1]
    embedder = vector_fixture.QueryEmbedder(passage) if with_embedder else None
    context = vector_fixture._context(data, corpus, embedder)

    graph = search_service.search(
        vector_fixture.QUERY, context, limit=50, strategy="hybrid_graph", graph_enabled=True
    )

    assert cause in graph.index.degraded
    assert graph.results
    assert not [m for r in graph.results for m in r.matches if "vector" in m.matched_by]
    if embedder is not None:
        assert embedder.calls == []
