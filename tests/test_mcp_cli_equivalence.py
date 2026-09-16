# tests/test_mcp_cli_equivalence.py
"""Los dos tests de atadura del Plan 04 §4.2: CLI `--json` y MCP no pueden divergir.

Que CLI y MCP «deberían» coincidir es prosa. Esto es lo que los ata, y por eso vive en su
propio fichero: una herramienta que añade un campo, que aplica un filtro de más o que
recorta con otro límite sale ROJA aquí, aunque las dos mitades sigan siendo verdes por
separado (regla 5 — dos definiciones de la misma cosa divergen en silencio).

**EL NOMBRE NO PROMETE MÁS DE LO QUE PRUEBA (m13).** Esto es igualdad ESTRUCTURAL:
`json.loads(...) == json.loads(...)`. No es igualdad de bytes y no podría serlo — el CLI
imprime con `indent=2` y el payload MCP llega envuelto en un `CallToolResult` del que hay
que desempaquetar el bloque de texto. Estructural es además lo correcto: lo que no puede
divergir es el modelo, no el orden de las claves.

**NINGÚN CASO PUEDE PASAR POR VACÍO.** Dos respuestas sin resultados son idénticas y no
prueban nada, así que cada caso declara de qué colección tiene que servir algo y la
igualdad se comprueba DESPUÉS de haberlo verificado sobre el lado del CLI.

**ENTRA POR LA SUPERFICIE PÚBLICA DE LAS DOS PUERTAS.** El arnés —el workspace de fixture,
`run_cli`, el cliente MCP en proceso y el desempaquetado del envelope— vive en
`tests/test_mcp_server.py` y se importa: tres copias serían tres cosas que divergen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.test_mcp_server import (
    call_mcp_tool,
    make_workspace,
    run_cli,
    unwrap_mcp_content,
)
from xbrain.knowledge.contracts import SearchFilters
from xbrain.mcp_server import MCP_TOOLS

# Se fija pequeño A PROPÓSITO: con el presupuesto por defecto (40.000) ningún `get` de esta
# fixture pagina, y la paginación es justo uno de los requisitos del §4.3 que las dos puertas
# tienen que cumplir igual. Con 400, tres casos truncan y entregan cursor por los dos lados —
# lo que prueba que ambos leen `[index].get_char_budget` y no cada uno el suyo.
GET_CHAR_BUDGET = 400


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch) -> Path:
    """El repo de mentira con el corpus de fixture, el presupuesto corto y el índice hecho."""
    from tests.test_mcp_server import build_index

    root = make_workspace(tmp_path, monkeypatch, get_char_budget=GET_CHAR_BUDGET)
    build_index()
    return root


# ---------------------------------------------------------------------------
# Los casos: queries, filtros, ids y semillas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """Una petición, dicha en los dos idiomas, y de qué tiene que servir algo."""

    id: str
    argv: tuple[str, ...]
    tool: str
    arguments: dict[str, Any]
    # Las colecciones del payload cuya suma de longitudes tiene que ser > 0 para que el caso
    # cuente. Es lo que impide que dos respuestas vacías se den por idénticas.
    evidence: tuple[str, ...] = field(default=("results",))


SEARCH_EVIDENCE = ("results",)
GET_EVIDENCE = ("surfaces", "chunks")
GRAPH_EVIDENCE = ("nodes", "edges", "paths")

EQUIVALENCE_CASES: tuple[Case, ...] = (
    # --- queries -----------------------------------------------------------
    Case(
        id="search-plain",
        argv=("search", "retrieval"),
        tool="xbrain.search",
        arguments={"query": "retrieval"},
    ),
    Case(
        id="search-explicit-strategy",
        argv=("search", "agent", "--strategy", "lexical"),
        tool="xbrain.search",
        arguments={"query": "agent", "strategy": "lexical"},
    ),
    # --- los ocho filtros del spec §7.2 ------------------------------------
    Case(
        id="filter-topics",
        argv=("search", "agent", "--topic", "agent-evaluation"),
        tool="xbrain.search",
        arguments={"query": "agent", "filters": {"topics": ["agent-evaluation"]}},
    ),
    Case(
        id="filter-content-kinds",
        argv=("search", "export", "--kind", "external_article"),
        tool="xbrain.search",
        arguments={"query": "export", "filters": {"content_kinds": ["external_article"]}},
    ),
    Case(
        id="filter-has-surfaces",
        argv=("search", "the", "--has-surface", "video_transcript"),
        tool="xbrain.search",
        arguments={"query": "the", "filters": {"has_surfaces": ["video_transcript"]}},
    ),
    Case(
        id="filter-origins",
        argv=("search", "the", "--origin", "asr"),
        tool="xbrain.search",
        arguments={"query": "the", "filters": {"origins": ["asr"]}},
    ),
    Case(
        # `--mine` es el ATAJO del CLI para `source=own_tweet`. Que las dos puertas coincidan
        # aquí es lo que prueba que el atajo no es una segunda semántica.
        id="filter-source-via-mine",
        argv=("search", "the", "--mine"),
        tool="xbrain.search",
        arguments={"query": "the", "filters": {"source": "own_tweet"}},
    ),
    Case(
        # LAS DOS FECHAS SE DICEN DISTINTO A PROPÓSITO, y el caso es MÁS fuerte por eso.
        # `--from`/`--to` son azúcar del CLI para un humano que teclea un día suelto:
        # `_parse_date` lo normaliza a un instante UTC, y `--to` al FINAL del día. El campo
        # del contrato es un `datetime`, así que por MCP viaja el instante. Al exigir que los
        # dos `filters` echoed coincidan, este caso deja CLAVADA esa correspondencia: si el
        # azúcar del CLI cambiase de instante, saldría rojo aquí.
        id="filter-author-and-dates",
        argv=("search", "the", "--author", "vgonpa", "--from", "2020-01-01", "--to", "2030-01-01"),
        tool="xbrain.search",
        arguments={
            "query": "the",
            "filters": {
                "author": "vgonpa",
                "created_from": "2020-01-01T00:00:00Z",
                "created_to": "2030-01-01T23:59:59.999999Z",
            },
        },
    ),
    # --- límites y paginación ---------------------------------------------
    Case(
        id="search-truncated-page",
        argv=("search", "the", "--limit", "1"),
        tool="xbrain.search",
        arguments={"query": "the", "limit": 1},
    ),
    Case(
        id="search-cursor-continuation",
        argv=("search", "the", "--limit", "1", "--cursor", "s:1"),
        tool="xbrain.search",
        arguments={"query": "the", "limit": 1, "cursor": "s:1"},
    ),
    # --- ids ---------------------------------------------------------------
    Case(
        id="get-default-surfaces",
        argv=("get", "k03"),
        tool="xbrain.get",
        arguments={"item_id": "k03"},
        evidence=GET_EVIDENCE,
    ),
    Case(
        id="get-named-surface",
        argv=("get", "k05", "--surface", "x_article"),
        tool="xbrain.get",
        arguments={"item_id": "k05", "surfaces": ["x_article"]},
        evidence=GET_EVIDENCE,
    ),
    Case(
        id="get-thread-surface",
        argv=("get", "k06", "--surface", "thread"),
        tool="xbrain.get",
        arguments={"item_id": "k06", "surfaces": ["thread"]},
        evidence=GET_EVIDENCE,
    ),
    Case(
        id="get-truncated-by-budget",
        argv=("get", "k03", "--surface", "external_article"),
        tool="xbrain.get",
        arguments={"item_id": "k03", "surfaces": ["external_article"]},
        evidence=GET_EVIDENCE,
    ),
    Case(
        id="get-cursor-continuation",
        argv=("get", "k03", "--surface", "external_article", "--cursor", "0:1"),
        tool="xbrain.get",
        arguments={"item_id": "k03", "surfaces": ["external_article"], "cursor": "0:1"},
        evidence=GET_EVIDENCE,
    ),
    Case(
        id="get-ranked-by-query",
        argv=("get", "k08", "--surface", "video_transcript", "--query", "evaluation"),
        tool="xbrain.get",
        arguments={
            "item_id": "k08",
            "surfaces": ["video_transcript"],
            "query": "evaluation",
        },
        evidence=GET_EVIDENCE,
    ),
    # --- semillas ----------------------------------------------------------
    Case(
        id="graph-one-hop",
        argv=("graph-expand", "--item", "k03"),
        tool="xbrain.graph_expand",
        arguments={"item_id": "k03"},
        evidence=GRAPH_EVIDENCE,
    ),
    Case(
        id="graph-two-hops",
        argv=("graph-expand", "--item", "k03", "--max-hops", "2"),
        tool="xbrain.graph_expand",
        arguments={"item_id": "k03", "max_hops": 2},
        evidence=GRAPH_EVIDENCE,
    ),
    Case(
        id="graph-capped-neighbors",
        argv=("graph-expand", "--item", "k08", "--max-neighbors", "1"),
        tool="xbrain.graph_expand",
        arguments={"item_id": "k08", "max_neighbors": 1},
        evidence=GRAPH_EVIDENCE,
    ),
)


def _served(payload: dict[str, Any], case: Case) -> int:
    """Cuántas cosas sirvió este caso. Cero ⇒ el caso no prueba nada y el test lo dice."""
    return sum(len(payload[key]) for key in case.evidence)


# ---------------------------------------------------------------------------
# a) La equivalencia
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", EQUIVALENCE_CASES, ids=lambda case: case.id)
def test_mcp_and_cli_json_are_structurally_identical(case: Case, workspace: Path) -> None:
    """Plan 04 §4.2 / §11.10: el MISMO documento por las dos puertas, en todos los casos.

    Un campo que un adaptador añade y el otro no ⇒ rojo. Un filtro de más, un límite
    distinto o un formato propio ⇒ rojo.
    """
    cli = json.loads(run_cli(case.argv))
    mcp = json.loads(unwrap_mcp_content(call_mcp_tool(case.tool, case.arguments)))
    assert _served(cli, case) > 0, f"{case.id} no sirvió nada: la igualdad sería vacía"
    assert cli == mcp


# ---------------------------------------------------------------------------
# b) La exposición
# ---------------------------------------------------------------------------


def test_every_service_is_exposed_and_nothing_else() -> None:
    """Plan 04 §4.2: tres servicios, tres herramientas, y ninguna cuarta puerta."""
    assert set(MCP_TOOLS) == {"xbrain.search", "xbrain.get", "xbrain.graph_expand"}


# ---------------------------------------------------------------------------
# Que el conjunto de casos no se quede corto sin que nadie se entere
# ---------------------------------------------------------------------------


def test_the_cases_exercise_every_tool_and_every_declared_filter() -> None:
    """Los casos cubren las tres herramientas y los OCHO filtros del contrato.

    El conjunto de filtros se DERIVA de `SearchFilters.model_fields`, no se teclea: un
    noveno filtro añadido al contrato deja este test rojo hasta que alguien escriba el caso
    que lo ejerce por las dos puertas. Una lista escrita a mano aquí envejecería sola y la
    equivalencia dejaría de cubrirlo en silencio (regla 5).
    """
    assert {case.tool for case in EQUIVALENCE_CASES} == set(MCP_TOOLS)
    exercised = {
        name
        for case in EQUIVALENCE_CASES
        for name in case.arguments.get("filters", {})
        if case.arguments["filters"][name]
    }
    assert exercised == set(SearchFilters.model_fields)


def test_the_cases_exercise_truncation_and_continuation() -> None:
    """Los límites y la paginación del §4.3 se recorren por las dos puertas, no se suponen.

    Sin esto, `--limit`, `--cursor` y `get_char_budget` podrían divergir sin que un solo
    caso lo tocara: la igualdad estructural sólo prueba lo que se le pide.
    """
    cursors = {case.id for case in EQUIVALENCE_CASES if case.arguments.get("cursor")}
    assert {"search-cursor-continuation", "get-cursor-continuation"} <= cursors
    assert any(case.arguments.get("limit") for case in EQUIVALENCE_CASES)
