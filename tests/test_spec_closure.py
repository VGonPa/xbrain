"""El cierre del spec «conocimiento verificable» §13, ejecutable (Plan 04 §11, fila 04.8).

LA REGLA, LITERAL DEL PLAN 04 §11: *«un criterio sin prueba nombrada es un criterio no
cumplido»*. Este fichero es la tabla que esa frase pide: los QUINCE criterios del spec §13,
cada uno con los tests (`fichero::test`) o las secciones de documento (fichero, cabecera
exacta y una frase que tiene que seguir dentro) que lo demuestran, y su estado REAL.

QUÉ LO HACE EJECUTABLE, y no una lista de afirmaciones:

  · borrar o renombrar un test nombrado aquí pone la suite ROJA (`ast`, sin importar nada);
  · renombrar una cabecera citada, o quitar de su sección la frase citada, también;
  · cada criterio NO cumplido lleva un TESTIGO que deja de pasar el día que alguien lo
    arregle, así que su estado no puede quedarse rancio en ninguna de las dos direcciones;
  · dos criterios no tenían prueba ninguna en el árbol —§13.12 y la mitad de §13.13— y la
    tienen AQUÍ, pasando por las puertas públicas, no por las funciones.

LA TRAMPA DE NUMERACIÓN, escrita para que nadie vuelva a caer. El repositorio cita «§13.N» de
TRES documentos distintos: el spec, el Plan 03 y el Plan 04. `docs/embeddings-bakeoff.md` §10
dice «§13.8 — NO CUMPLE: 1 de 3», y ese §13.8 es el del PLAN 03 (bake-off con ≥ 3
candidatos). El §13.8 del SPEC es la paginación de `get`, y ése se cumple. Lo que el bake-off
incompleto deja sin cumplir en el spec es el §13.5, porque el Plan 04 §11 asigna al Plan 03 su
parte de «textual y vectorial se evalúan por separado y juntas». Igual con §13.12: los tests
que lo citan (`test_knowledge_cli.py`, `test_knowledge_degradation.py`) hablan del criterio 12
del Plan 03 —el extra `[embeddings]`—, no de «sin llamada a un LLM generativo».

LO QUE NO PUEDE VER. Ve nombres y frases, no comportamiento: un test renombrado que además
afloje su aserción pasa por aquí sin ruido (el mismo límite que `test_plan02_acceptance.py`
declara). Lo concluyente es la dirección negativa: un node id que ya no resuelve no está
guardando nada. La COPIA de los quince textos es a mano porque el spec vive en
`zz-support-files/`, que no está versionado; `SPEC_13_COUNT` es lo que impide encogerla sin
que un número se mueva en el diff.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.test_mcp_server import (
    build_index,
    call_mcp_tool,
    make_workspace,
    run_cli,
    unwrap_mcp_content,
)
from xbrain.knowledge.lexical_fts import match_expression

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF = "tests/test_spec_closure.py"

SPEC_13_COUNT = 15

# Los que NO se cumplen hoy. Bajar esta lista exige editar esta línea, y el testigo de cada uno
# se pone rojo el día que deja de ser verdad.
UNMET: frozenset[int] = frozenset({1, 5, 14})

# Criterios que el spec define como documentación o como publicación, no como comportamiento.
DOCUMENTARY: frozenset[int] = frozenset({14, 15})


@dataclass(frozen=True)
class Node:
    """Un test que existe en este árbol: `tests/fichero.py::nombre` (o `::Clase::nombre`)."""

    node: str


@dataclass(frozen=True)
class Section:
    """Una cabecera EXACTA de un documento versionado y una frase que vive dentro de ella."""

    doc: str
    heading: str
    quote: str


@dataclass(frozen=True)
class Criterion:
    number: int
    text: str
    met: bool
    proofs: tuple[Node | Section, ...]
    note: str = ""


BAKEOFF = "docs/embeddings-bakeoff.md"
SWEEP = "docs/graph-threshold-sweep.md"
SEARCH = "tests/test_knowledge_search_service.py"
GET = "tests/test_knowledge_get_service.py"
GRAPH = "tests/test_knowledge_graph_service.py"
LEXICAL = "tests/test_knowledge_lexical.py"
HYBRID = "tests/test_knowledge_search_hybrid.py"
PROVENANCE = "tests/test_knowledge_provenance.py"
EVALUATION = "tests/test_knowledge_evaluation.py"
INVALIDATION = "tests/test_knowledge_index_invalidation.py"
STRATEGY = "tests/test_knowledge_graph_strategy.py"
GRAPH_SWEEP = "tests/test_knowledge_graph_sweep.py"
GRAPH_BUILD = "tests/test_knowledge_graph_build.py"
MCP = "tests/test_mcp_server.py"
EQUIVALENCE = "tests/test_mcp_cli_equivalence.py"
INJECTION = "tests/test_mcp_prompt_injection.py"
SCHEMA = "tests/test_knowledge_index_schema.py"
GOLDEN = "tests/test_knowledge_goldenset.py"

CLOSURE: tuple[Criterion, ...] = (
    Criterion(
        1,
        "un agente puede buscar por concepto, frase exacta, topic y filtros estructurados",
        met=False,
        note=(
            "Topic y los ocho filtros, sí. Concepto, sólo con el plano vectorial opt-in, y su "
            "calidad es la pregunta abierta de §13.5. FRASE EXACTA, NO: `match_expression` parte "
            "la consulta en términos y los une con OR, así que no hay operador de frase; lo que "
            "existe es el literal de UN término (`@simonw`, `11.37%`)."
        ),
        proofs=(
            Node(f"{HYBRID}::test_vector_serves_only_what_the_vector_channel_found"),
            Node(f"{LEXICAL}::test_the_query_is_escaped_not_interpolated"),
            Node(f"{SEARCH}::test_a_topic_note_match_returns_the_topics_supporting_items"),
            Node(f"{LEXICAL}::test_every_declared_filter_is_actually_pushed_to_sql"),
            Node(f"{LEXICAL}::test_the_filter_is_applied_before_scoring_not_after"),
            Node(f"{SELF}::test_criterion_1_stays_unmet_while_a_quoted_query_is_a_disjunction"),
        ),
    ),
    Criterion(
        2,
        "puede recuperar la fuente real sin depender del summary",
        met=True,
        proofs=(
            Node(f"{GET}::test_get_returns_the_whole_article_body_untruncated"),
            Node(f"{GET}::test_get_works_with_the_index_directory_deleted"),
            Node(f"{SEARCH}::test_a_summary_match_points_at_the_underlying_article"),
            Node(f"{MCP}::test_mcp_get_answers_with_the_index_deleted"),
        ),
    ),
    Criterion(
        3,
        "cada fragmento expone procedencia, autoría y localizador",
        met=True,
        proofs=(
            Node(f"{SEARCH}::test_a_quoted_post_match_carries_the_quoted_author_not_the_poster"),
            Node(f"{SEARCH}::test_a_match_locator_is_the_surface_locator_plus_the_character_range"),
            Node(
                f"{GET}::test_a_chunk_from_get_carries_the_locator_of_the_source_whose_text_it_delivers"
            ),
            Node(f"{INJECTION}::test_every_served_fragment_carries_its_three_labels"),
        ),
    ),
    Criterion(
        4,
        "summaries, digests y topic syntheses están disponibles pero etiquetados como derivados",
        met=True,
        proofs=(
            Node(f"{PROVENANCE}::test_is_derived_is_true_exactly_for_machine_produced_text"),
            Node(f"{PROVENANCE}::test_unknown_fails_closed_to_llm_synthesis"),
            Node(f"{SEARCH}::test_a_derived_match_with_no_primary_source_says_so"),
            Node(
                f"{GET}::test_get_without_surfaces_delivers_the_summary_and_withholds_every_long_body"
            ),
        ),
    ),
    Criterion(
        5,
        "la búsqueda textual y la vectorial se evalúan por separado y juntas",
        met=False,
        note=(
            "El instrumento existe y mide `lexical`, `vector` y `hybrid` por separado y fusionadas. "
            "La evaluación que el Plan 03 debía entregar para este criterio es su §13.8 —un "
            "bake-off con ≥ 3 candidatos, ganador y perdedores— y NO CUMPLE: 1 de 3. Sólo "
            "MiniLM se midió (y perdió); e5-small se interrumpió a 859 s por presión de memoria "
            "y disco; e5-base, bge-m3 y jina-v3 no se corrieron."
        ),
        proofs=(
            Node(f"{EVALUATION}::test_metrics_are_reported_per_stratum_and_provenance"),
            Node(
                f"{EVALUATION}::test_a_vector_evaluation_is_reported_per_stratum_and_provenance_by_the_vector_channel"
            ),
            Node(f"{EVALUATION}::test_hybrid_fuses_both_channels_and_names_itself"),
            Section(BAKEOFF, "## 0. Resultado", "el bake-off está INCOMPLETO"),
            Section(
                BAKEOFF,
                "## 10. Puertas del spec §8.6 y criterios del Plan 03 §13",
                "NO CUMPLE: 1 de 3",
            ),
        ),
    ),
    Criterion(
        6,
        "el índice incremental detecta cambios y nunca sirve chunks stale",
        met=True,
        note=(
            "Con las definiciones del propio spec (§5.6, §9.3): un chunk stale es uno cuyo "
            "fingerprint no recomputa, y se excluye y se cuenta; un índice por detrás del store "
            "se DECLARA (`index_behind_store`). La señal barata tiene un punto ciego declarado."
        ),
        proofs=(
            Node(f"{INVALIDATION}::test_update_touches_only_the_changed_item"),
            Node(
                f"{INVALIDATION}::test_update_detects_a_summary_change_on_an_item_with_no_content"
            ),
            Node(f"{SEARCH}::test_a_chunk_with_a_manipulated_fingerprint_is_excluded_and_counted"),
            Node(f"{SEARCH}::test_editing_the_store_without_reindexing_declares_the_index_behind"),
            Node(
                f"{HYBRID}::test_a_vector_left_stale_by_update_is_not_served_and_the_response_says_so"
            ),
            Node(f"{GRAPH}::test_an_index_behind_the_store_is_refused_not_expanded"),
            Section(
                "docs/knowledge-index.md",
                "## Known limits of the lexical baseline",
                "invisible to it",
            ),
        ),
    ),
    Criterion(
        7,
        "`search` agrupa matches sin ocultar la superficie que produjo cada uno",
        met=True,
        proofs=(
            Node(f"{SEARCH}::test_a_long_transcript_yields_one_result_with_at_most_three_matches"),
            Node(
                f"{SEARCH}::test_a_primary_match_names_ITSELF_not_every_primary_surface_the_item_has"
            ),
            Node(f"{HYBRID}::test_a_chunk_both_channels_found_is_explained_by_both"),
        ),
    ),
    Criterion(
        8,
        "`get` puede entregar fuentes largas de manera selectiva/paginada",
        met=True,
        proofs=(
            Node(f"{GET}::test_a_body_over_the_budget_is_paginated_not_cut"),
            Node(f"{GET}::test_the_cursor_continues_where_the_previous_call_stopped"),
            Node(f"{GET}::test_asking_for_a_surface_the_item_does_not_have_lists_what_it_does"),
            Node(f"{EQUIVALENCE}::test_the_cases_exercise_truncation_and_continuation"),
        ),
    ),
    Criterion(
        9,
        "el grafo mínimo explica paths y conserva support ids",
        met=True,
        proofs=(
            Node(f"{GRAPH}::test_every_path_carries_node_types_relation_method_weight_and_support"),
            Node(
                f"{GRAPH}::test_every_served_path_rests_on_item_ids_that_resolve_in_the_live_store"
            ),
            Node(f"{GRAPH}::test_a_co_occurrence_whose_listed_support_left_the_store_is_refused"),
            Node(f"{GRAPH_BUILD}::test_no_item_to_item_edge_exists_in_the_schema"),
        ),
    ),
    Criterion(
        10,
        "la expansión por grafo puede activarse o desactivarse y tiene una métrica incremental",
        met=True,
        note=(
            "El interruptor es `search(..., graph_enabled=True)` y lo usa `xbrain eval --strategy "
            "hybrid_graph`; `xbrain search` y MCP no pueden encenderlo (backlog). La métrica es el "
            "Δ recall@10 frente a `hybrid` y el estrato `expansion`, y su resultado es NEGATIVO."
        ),
        proofs=(
            Node(f"{STRATEGY}::test_hybrid_graph_existe_es_desactivable_y_el_default_no_cambia"),
            Node(
                f"{STRATEGY}::test_un_candidato_solo_grafo_que_no_puntua_en_ningun_canal_no_entra"
            ),
            Node(
                f"{GRAPH_SWEEP}::test_the_applied_graph_threshold_is_the_winner_the_signed_sweep_published"
            ),
            Node(
                f"{GRAPH_SWEEP}::test_the_graph_sweep_publishes_the_expansion_population_its_useful_column_counts_from"
            ),
            Section(SWEEP, "## 0. Resultado", "Ninguna de las 16 combinaciones aporta"),
        ),
    ),
    Criterion(
        11,
        "CLI JSON y MCP usan los mismos modelos y producen semántica equivalente",
        met=True,
        proofs=(
            Node(f"{EQUIVALENCE}::test_mcp_and_cli_json_are_structurally_identical"),
            Node(f"{EQUIVALENCE}::test_every_service_is_exposed_and_nothing_else"),
            Node(f"{MCP}::test_the_output_schema_is_the_plan01_model_itself"),
            Node(f"{MCP}::test_mcp_refuses_with_the_same_message_as_the_cli"),
            Section("docs/mcp.md", "## The three tools", "same response model"),
        ),
    ),
    Criterion(
        12,
        "query y retrieval funcionan sin llamada a un LLM generativo",
        met=True,
        proofs=(
            Node(f"{SELF}::test_query_and_retrieval_never_construct_a_generative_client"),
            Node(f"{MCP}::test_the_mcp_server_imports_nothing_that_speaks_to_the_network"),
        ),
    ),
    Criterion(
        13,
        "ningún artefacto personal o índice entra en Git",
        met=True,
        proofs=(
            Node(f"{SELF}::test_no_personal_artifact_or_index_can_enter_git"),
            Node(f"{SCHEMA}::test_the_index_directory_is_git_ignored"),
            Node(f"{GOLDEN}::test_the_golden_set_is_tracked_and_the_reports_are_not"),
        ),
    ),
    Criterion(
        14,
        "README, tutorial, arquitectura y troubleshooting se actualizan con el código de cada plan",
        met=False,
        note=(
            "README, ARCHITECTURE y troubleshooting cubren los cuatro planes. `docs/tutorial.md` "
            "se actualizó en el Plan 02 (02.15) y en ningún hijo del 03 ni del 04, y la fila "
            "04.8 no lo incluye en su alcance: no enseña `vector`/`hybrid`, `graph-expand` ni MCP."
        ),
        proofs=(
            Section("README.md", "## Search & retrieval", "xbrain mcp-serve"),
            Section(
                "ARCHITECTURE.md", "### The minimal graph", "never a relationship in the world"
            ),
            Section("ARCHITECTURE.md", "### The MCP server", "thin adapter"),
            Section("docs/troubleshooting.md", "## The knowledge index", "hybrid_graph"),
            Node(f"{SELF}::test_criterion_14_stays_unmet_while_the_tutorial_skips_plans_03_and_04"),
        ),
    ),
    Criterion(
        15,
        "los resultados negativos de evaluación se documentan en vez de ocultarse",
        met=True,
        proofs=(
            Section(BAKEOFF, "## 0. Resultado", "Pierde en las dos"),
            Section(SWEEP, "## 0. Resultado", "0 de esos 33"),
            Section("CLAUDE.md", "## Architecture", "reports `800/150` tied"),
            Section(
                "docs/knowledge-index.md", "## The graph — opt-in, and measured negative", "0 of 33"
            ),
        ),
    ),
)


def _test_names(path: Path) -> set[str]:
    """`test_*` de un módulo, de nivel superior y dentro de clases, por `ast`: sin importar."""
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names |= {
                f"{node.name}::{child.name}"
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name.startswith("test_")
            }
    return names


def _section_body(text: str, heading: str) -> str | None:
    """El cuerpo bajo `heading` (línea exacta) hasta la siguiente cabecera de igual o mayor rango."""
    lines = text.splitlines()
    if heading not in lines:
        return None
    start = lines.index(heading) + 1
    level = len(heading) - len(heading.lstrip("#"))
    body = []
    fenced = False
    for line in lines[start:]:
        # Un `# →` dentro de un bloque de código es un comentario de shell, no una cabecera.
        if line.lstrip().startswith("```"):
            fenced = not fenced
        match = None if fenced else re.match(r"^(#+) ", line)
        if match and len(match.group(1)) <= level:
            break
        body.append(line)
    return " ".join(" ".join(body).split())


def _ids(criterion: Criterion) -> str:
    return f"13.{criterion.number}"


def test_the_closure_is_the_fifteen_criteria_of_spec_13_in_order() -> None:
    """Totalidad: los quince, una vez cada uno, en el orden del spec.

    Visto en rojo quitando la fila 8: `[1..7, 9..15] != [1..15]`.
    """
    assert len(CLOSURE) == SPEC_13_COUNT
    assert [c.number for c in CLOSURE] == list(range(1, SPEC_13_COUNT + 1))


@pytest.mark.parametrize("criterion", CLOSURE, ids=_ids)
def test_every_named_test_exists_in_this_tree(criterion: Criterion) -> None:
    """Un node id que no resuelve es una prueba que se fue sin que nadie lo declarara.

    Visto en rojo renombrando `test_every_path_carries_node_types_relation_method_weight_and_support`
    en `tests/test_knowledge_graph_service.py`: falla 13.9 nombrando el node id.
    """
    missing = []
    for proof in criterion.proofs:
        if not isinstance(proof, Node):
            continue
        module, _, name = proof.node.partition("::")
        path = REPO_ROOT / module
        if not path.is_file() or name not in _test_names(path):
            missing.append(proof.node)
    assert not missing, f"§13.{criterion.number} nombra tests que no existen: {missing}"


@pytest.mark.parametrize("criterion", CLOSURE, ids=_ids)
def test_every_cited_section_exists_and_still_says_it(criterion: Criterion) -> None:
    """Una cabecera renombrada, o una frase que ya no está bajo ella, deja al criterio sin prueba.

    La frase se busca con los espacios colapsados, para que re-envolver un párrafo no rompa
    nada y reescribirlo sí. Visto en rojo cambiando `## 0. Resultado` por `## 0. Veredicto` en
    `docs/graph-threshold-sweep.md`: fallan 13.10 y 13.15.
    """
    problems = []
    for proof in criterion.proofs:
        if not isinstance(proof, Section):
            continue
        path = REPO_ROOT / proof.doc
        if not path.is_file():
            problems.append(f"{proof.doc}: no existe")
            continue
        body = _section_body(path.read_text(encoding="utf-8"), proof.heading)
        if body is None:
            problems.append(f"{proof.doc}: no hay cabecera {proof.heading!r}")
        elif proof.quote not in body:
            problems.append(f"{proof.doc} {proof.heading!r}: ya no dice {proof.quote!r}")
    assert not problems, f"§13.{criterion.number}: " + "; ".join(problems)


@pytest.mark.parametrize("criterion", CLOSURE, ids=_ids)
def test_a_criterion_of_behaviour_names_at_least_one_test(criterion: Criterion) -> None:
    """Un criterio de comportamiento no se demuestra con prosa: necesita al menos un test.

    Sólo §13.14 y §13.15 son, por su enunciado, criterios de documentación.
    """
    assert criterion.proofs, f"§13.{criterion.number} no nombra ninguna prueba"
    if criterion.number not in DOCUMENTARY:
        assert any(isinstance(p, Node) for p in criterion.proofs), (
            f"§13.{criterion.number} es de comportamiento y sólo cita documentos"
        )


def test_the_unmet_criteria_are_declared_and_say_why() -> None:
    """Un criterio no cumplido se DICE, con su razón; cambiar el conjunto exige editar `UNMET`."""
    assert {c.number for c in CLOSURE if not c.met} == UNMET
    silent = [c.number for c in CLOSURE if not c.met and not c.note]
    assert not silent, f"criterios no cumplidos sin razón escrita: {silent}"


def test_criterion_1_stays_unmet_while_a_quoted_query_is_a_disjunction() -> None:
    """El testigo de §13.1: una consulta entre comillas sigue siendo una disyunción de términos.

    El día que exista un operador de frase, esto se pone rojo y §13.1 tiene que revisarse.
    """
    assert match_expression('"harness engineering"') == '"harness" OR "engineering"'


def test_criterion_14_stays_unmet_while_the_tutorial_skips_plans_03_and_04() -> None:
    """El testigo de §13.14: el tutorial no enseña ni el grafo, ni MCP, ni la estrategia vectorial.

    Actualizar el tutorial pone esto rojo, y eso es lo que obliga a mover §13.14 a cumplido.
    """
    tutorial = (REPO_ROOT / "docs" / "tutorial.md").read_text(encoding="utf-8")
    assert "graph-expand" not in tutorial
    assert "mcp-serve" not in tutorial
    assert "--strategy hybrid" not in tutorial


def test_query_and_retrieval_never_construct_a_generative_client(tmp_path, monkeypatch) -> None:
    """§13.12 por las dos puertas públicas: CLI y MCP, con el cliente de Anthropic inconstruible.

    Toda llamada a un LLM generativo de xbrain construye `anthropic.Anthropic()` con un import
    PEREZOSO (`executors/api.py`, `describe.py`, `vocab.py`, `topic_synth.py`), que resuelve el
    atributo del módulo en el momento de llamar: sustituirlo aquí intercepta cualquiera de ellas.
    No es un cerrojo de red (retirado por decisión de alcance en 04.7): es la pregunta exacta del
    criterio, y el registro atrapa también un intento cuya excepción alguien se tragara.

    Visto en rojo añadiendo `from anthropic import Anthropic; Anthropic()` al comando `search`
    de `cli.py`: la llamada CLI sale con código 1 y el registro no queda vacío.
    """
    import anthropic

    attempts: list[str] = []

    class _Refused:
        def __init__(self, *args: object, **kwargs: object) -> None:
            attempts.append("constructed")
            raise AssertionError("query/retrieval construyó un cliente de LLM generativo")

    monkeypatch.setattr(anthropic, "Anthropic", _Refused)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", _Refused)
    make_workspace(tmp_path, monkeypatch)
    build_index()

    found = json.loads(run_cli(["search", "attention"]))
    item_id = found["results"][0]["item_id"]
    json.loads(run_cli(["get", item_id, "--query", "attention"]))
    json.loads(run_cli(["graph-expand", "--item", item_id]))
    for tool, arguments in (
        ("xbrain.search", {"query": "attention"}),
        ("xbrain.get", {"item_id": item_id}),
        ("xbrain.graph_expand", {"item_id": item_id}),
    ):
        json.loads(unwrap_mcp_content(call_mcp_tool(tool, arguments)))
    assert attempts == []


# Lo personal y lo derivado, por los caminos que el código escribe de verdad.
PERSONAL_PATHS = (
    "config.toml",
    "auth/storage_state.json",
    "data/items.json",
    "data/vocab.yaml",
    "data/topics.json",
    "data/payloads/12/2063609922667815012.json.gz",
    "data/media/2063609922667815012/0.jpg",
    "data/eval-report.json",
    "data/index/knowledge.db",
    "data/index/manifest.json",
    "data/index/vectors.f32",
    "data/index/vectors.meta.json",
    "data/eval-index/graph-sweep/knowledge.db",
)


def test_no_personal_artifact_or_index_can_enter_git() -> None:
    """§13.13 preguntado a GIT, no a `.gitignore`, y desde los dos lados.

    Ignorado: cada ruta personal o derivada. Versionado bajo `data/` y `auth/`: sólo los
    `.gitkeep`. Visto en rojo añadiendo `!config.toml` a `.gitignore`.
    """
    result = subprocess.run(  # noqa: S603
        ["git", "check-ignore", "--no-index", *PERSONAL_PATHS],  # noqa: S607
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    ignored = set(result.stdout.split())
    assert set(PERSONAL_PATHS) - ignored == set()
    tracked = subprocess.run(  # noqa: S603
        ["git", "ls-files", "data", "auth"],  # noqa: S607
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert set(tracked) == {"data/.gitkeep", "auth/.gitkeep"}
