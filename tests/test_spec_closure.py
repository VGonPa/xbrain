"""El cierre del spec «conocimiento verificable» §13, ejecutable (Plan 04 §11, fila 04.8).

LA REGLA, LITERAL DEL PLAN 04 §11: *«un criterio sin prueba nombrada es un criterio no
cumplido»*. Este fichero es la tabla que esa frase pide: los QUINCE criterios del spec §13,
cada uno con los tests (`fichero::test`) o las secciones de documento (fichero, cabecera
exacta y una frase que tiene que seguir dentro) que lo demuestran, y su estado REAL.

QUÉ LO HACE EJECUTABLE, y no una lista de afirmaciones:

  · borrar o renombrar un test nombrado aquí pone la suite ROJA (`ast`, sin importar nada);
  · renombrar una cabecera citada, o quitar de su sección la frase citada, también;
  · cada criterio NO cumplido lleva un TESTIGO (`Witness`) que deja de pasar el día que alguien
    lo arregle, y ningún criterio cumplido lleva uno: cambiar el veredicto sin mover el testigo,
    o quitar el testigo sin cambiar el veredicto, pone rojo;
  · dos criterios no tenían prueba ninguna en el árbol —§13.12 y la mitad de §13.13— y la
    tienen AQUÍ, pasando por las puertas públicas, no por las funciones.

QUE UNA PRUEBA EXISTA NO QUIERE DECIR QUE PRUEBE ALGO. Borrar SÓLO la aserción del testigo de
§13.1 dejaba este fichero en 51 de 51 verde: el nombre seguía resolviendo, así que el «no
cumple» podía quedarse rancio sin ruido, que es justo el defecto que este fichero existe para
impedir. Por eso hay dos mecanismos más, y así se justifican:

  · EL TESTIGO SE EJECUTA EN EL MUNDO ARREGLADO. Cada `Witness` trae `fixes`, un mundo por
    cláusula en el que el criterio SÍ se cumple: un `match_expression` con operador de frase,
    un tutorial que enseña `graph-expand`… Se simulan con `monkeypatch` sobre los nombres que
    lee el testigo. El testigo tiene que pasar hoy y dar `AssertionError` en cada uno de esos
    mundos, y entre todos tienen que disparar CADA `assert` suyo, por número de línea. Es la
    regla 1 («míralo en rojo primero») ejecutada en cada corrida en vez de recordada una vez.
    Cierra la clase entera para los testigos: vaciarlo, debilitarlo, cortarlo con un
    `return`, saltarlo o reescribirlo para que lea otra cosa lo deja verde en algún mundo
    arreglado, y eso es rojo. Se puede porque el cierre sabe qué significa ARREGLAR cada
    criterio suyo; de un test de comportamiento ajeno no lo sabe.
  · SUELO ESTÁTICO para cada test nombrado y cada test de este fichero (`_hollow_reasons`,
    por `ast`). Es rojo en cinco casos: si no queda ninguna aserción viva (un `assert` que lee
    algo, o un `pytest.raises`), si lleva skip/xfail, si tiene un `return` propio, si un `try`
    se traga la aserción o si una condición constante la deja muerta. Medido al escribirlo:
    los 62 tests distintos que cubre (53 nombrados por la tabla, 48 de ellos en otros ficheros,
    más los 14 de éste) lo cumplen, sin excepciones.
  · Las dos guardas se vigilan entre sí
    (`test_the_guards_of_this_file_see_the_defect_they_exist_for`). Cada una tiene que ver el
    defecto que existe para ver cuando se le da un espécimen de él, y el suelo se aplica a su
    propio test desde fuera, porque un test vaciado no puede denunciarse a sí mismo.

LO QUE NO PUEDE VER, declarado en vez de prometido:

  · un test de comportamiento DEBILITADO (una tautología, una aserción sobre otra cosa);
  · un test de comportamiento vaciado EN EJECUCIÓN, como un `for` sobre una colección que ha
    quedado vacía. Los dos pasan el suelo.

Cerrarlos exigiría saber qué defecto persigue cada uno de los 50 tests de comportamiento
nombrados, para mutarlo test a test, o contar las aserciones ejecutadas con un plugin de pytest
en un subproceso. Es un aparato desproporcionado para lo que guarda, así que se DECLARA: es la
misma decisión que convirtió la garantía de red de 04.7 en una declaración.

Tampoco ve dos ediciones coordinadas: vaciar las dos guardas en el mismo cambio, o borrar un
mundo arreglado junto con su cláusula. Eso es editar la tabla, y lo que lo para es que alguien
lea el diff (regla 13).

La COPIA de los quince textos es a mano porque el spec vive en `zz-support-files/`, que no está
versionado; `SPEC_13_COUNT` es lo que impide encogerla sin que un número se mueva en el diff.

LA TRAMPA DE NUMERACIÓN, escrita para que nadie vuelva a caer. «§13.N» es ambiguo:

  · el spec, el Plan 01, el Plan 02 y el Plan 03 tienen cada uno su §13;
  · sólo los §13 del spec y del Plan 03 son criterios de aceptación (el del Plan 01 son sus
    quality gates y el del Plan 02, su documentación);
  · el Plan 04 NO tiene §13: sus criterios son su §11.

`docs/embeddings-bakeoff.md` §10 dice «§13.8 — NO CUMPLE: 1 de 3», y ese §13.8 es el del
PLAN 03 (bake-off con ≥ 3 candidatos). El §13.8 del SPEC es la paginación de `get`, y ése se
cumple. Lo que el bake-off incompleto deja sin cumplir en el spec es el §13.5, porque el Plan 04
§11 asigna al Plan 03 su parte de «textual y vectorial se evalúan por separado y juntas».

Igual con §13.12: los tests que lo citan (`test_knowledge_cli.py`,
`test_knowledge_degradation.py`) hablan del criterio 12 del Plan 03 —el extra
`[embeddings]`—, no de «sin llamada a un LLM generativo».
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
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


Fix = Callable[[pytest.MonkeyPatch, Path], None]


@dataclass(frozen=True)
class Witness(Node):
    """El test que mantiene honesto un «NO cumple»: pasa mientras el criterio no se cumple.

    `fixes` son los mundos en que SÍ se cumple, uno por cláusula del testigo. Cada uno recibe un
    `MonkeyPatch` de vida acotada y un directorio vacío, y simula el arreglo sobre los nombres
    que el testigo lee. Un testigo vive en este fichero: es donde se sabe qué es arreglarlo.
    """

    fixes: tuple[Fix, ...] = ()


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

BAKEOFF_RESULT = "## 0. Resultado"
BAKEOFF_GATES = "## 10. Puertas del spec §8.6 y criterios del Plan 03 §13"
TUTORIAL = "docs/tutorial.md"


def _serve(
    patched: pytest.MonkeyPatch, root: Path, relpath: str, edit: Callable[[str], str]
) -> None:
    """Simula un arreglo documental: `REPO_ROOT` pasa a un árbol con `relpath` editado."""
    real = (REPO_ROOT / relpath).read_text(encoding="utf-8")
    target = root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(edit(real), encoding="utf-8")
    patched.setattr(sys.modules[__name__], "REPO_ROOT", root)


def _a_phrase_operator_exists(patched: pytest.MonkeyPatch, root: Path) -> None:
    """§13.1 arreglado: una consulta entre comillas llega a FTS5 como una frase, no como un OR."""
    patched.setattr(sys.modules[__name__], "match_expression", lambda query: query)


def _the_bakeoff_result_is_complete(patched: pytest.MonkeyPatch, root: Path) -> None:
    """§13.5 arreglado, primera cláusula: el resultado firmado deja de llamarse incompleto."""
    _serve(patched, root, BAKEOFF, lambda text: text.replace("está INCOMPLETO", "está COMPLETO"))


def _the_bakeoff_gate_is_met(patched: pytest.MonkeyPatch, root: Path) -> None:
    """§13.5 arreglado, segunda cláusula: la puerta del Plan 03 §13.8 pasa a cumplirse."""
    _serve(patched, root, BAKEOFF, lambda text: text.replace("NO CUMPLE: 1 de 3", "CUMPLE: 3 de 3"))


def _the_tutorial_teaches(term: str) -> Fix:
    """§13.14 arreglado para UNA cláusula: el tutorial enseña `term` y nada más cambia."""

    def fix(patched: pytest.MonkeyPatch, root: Path) -> None:
        _serve(patched, root, TUTORIAL, lambda text: f"{text}\n`xbrain {term}`\n")

    fix.__name__ = f"_the_tutorial_teaches({term!r})"
    return fix


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
            Witness(
                f"{SELF}::test_criterion_1_stays_unmet_while_a_quoted_query_is_a_disjunction",
                fixes=(_a_phrase_operator_exists,),
            ),
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
            Witness(
                f"{SELF}::test_criterion_5_stays_unmet_while_the_bakeoff_says_it_is_incomplete",
                fixes=(_the_bakeoff_result_is_complete, _the_bakeoff_gate_is_met),
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
            "El interruptor es `search(..., graph_enabled=True)`, y el único comando que lo usa "
            "es el barrido `xbrain eval --strategy hybrid_graph --sweep-graph …`. Sin "
            "`--sweep-graph`, `xbrain eval --strategy hybrid_graph` responde `lexical` declarando "
            "`hybrid_graph_not_implemented`, igual que `xbrain search` y MCP, que no pueden "
            "encenderlo (backlog). La métrica es el Δ recall@10 frente a `hybrid` y el estrato "
            "`expansion`, y su resultado es NEGATIVO."
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
            Witness(
                f"{SELF}::test_criterion_14_stays_unmet_while_the_tutorial_skips_plans_03_and_04",
                fixes=(
                    _the_tutorial_teaches("graph-expand"),
                    _the_tutorial_teaches("mcp-serve"),
                    _the_tutorial_teaches("search --strategy hybrid"),
                ),
            ),
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


def test_criterion_5_stays_unmet_while_the_bakeoff_says_it_is_incomplete() -> None:
    """El testigo de §13.5: el bake-off firmado sigue diciendo que está incompleto.

    Completarlo —≥ 3 candidatos, con ganador y perdedores— obliga a re-firmar
    `docs/embeddings-bakeoff.md`, y ese documento deja de decir estas dos frases: esto se pone
    rojo y §13.5 tiene que revisarse.
    """
    bakeoff = (REPO_ROOT / BAKEOFF).read_text(encoding="utf-8")
    assert "el bake-off está INCOMPLETO" in (_section_body(bakeoff, BAKEOFF_RESULT) or "")
    assert "NO CUMPLE: 1 de 3" in (_section_body(bakeoff, BAKEOFF_GATES) or "")


# ---------------------------------------------------------------------------
# Que una prueba exista no quiere decir que pruebe algo (ver el docstring del módulo)
# ---------------------------------------------------------------------------

_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_SKIPS = frozenset({"skip", "skipif", "xfail", "importorskip"})
_SWALLOWS = frozenset({"AssertionError", "Exception", "BaseException"})


def _function_def(node_id: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """La definición de `fichero::nombre` (o `::Clase::nombre`), por `ast`; `None` si no está."""
    module, _, name = node_id.partition("::")
    path = REPO_ROOT / module
    if not path.is_file():
        return None
    scope: list[ast.stmt] = ast.parse(path.read_text(encoding="utf-8")).body
    owner, _, name = name.rpartition("::")
    if owner:
        classes = [n for n in scope if isinstance(n, ast.ClassDef) and n.name == owner]
        scope = classes[0].body if classes else []
    found = [
        n
        for n in scope
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    ]
    return found[0] if found else None


def _own_nodes(function: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.AST]:
    """Lo que el test ejecuta él mismo: su cuerpo, sin entrar en funciones ni clases anidadas."""
    stack: list[ast.AST] = [n for n in function.body if not isinstance(n, _NESTED)]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(c for c in ast.iter_child_nodes(node) if not isinstance(c, _NESTED))


def _tail_name(node: ast.AST) -> str:
    """`pytest.mark.skipif(…)` → `skipif`; `pytest.raises` → `raises`; `skip` → `skip`."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def _reads_something(expr: ast.AST) -> bool:
    """Falso para `True`, `1 == 1` o `"x"`: una aserción que no lee nada no puede fallar."""
    return any(
        isinstance(n, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)) for n in ast.walk(expr)
    )


def _never_true(test: ast.expr) -> bool:
    """`if False:` / `while 0:` / `if not True:` — el bloque que guardan es código muerto."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return isinstance(test.operand, ast.Constant) and bool(test.operand.value)
    return isinstance(test, ast.Constant) and not test.value


def _hollow_reasons(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """El suelo estático: por qué este test, tal como está escrito, no puede ponerse rojo."""
    own = list(_own_nodes(function))
    reasons = []
    if not any(
        (isinstance(n, ast.Assert) and _reads_something(n.test))
        or (isinstance(n, ast.withitem) and _tail_name(n.context_expr) == "raises")
        for n in own
    ):
        reasons.append("no queda ninguna aserción viva")
    reasons += [
        f"`{_tail_name(d)}` en sus decoradores"
        for d in function.decorator_list
        if _tail_name(d) in _SKIPS
    ]
    for node in own:
        if isinstance(node, ast.Call) and _tail_name(node) in _SKIPS:
            reasons.append(f"llama a `{_tail_name(node)}`")
        elif isinstance(node, ast.Return):
            reasons.append(f"`return` propio en la línea {node.lineno}")
        elif isinstance(node, (ast.If, ast.While)) and _never_true(node.test):
            reasons.append(f"condición que nunca se cumple en la línea {node.lineno}")
        elif isinstance(node, ast.ExceptHandler) and (
            node.type is None or {_tail_name(n) for n in ast.walk(node.type)} & _SWALLOWS
        ):
            reasons.append(f"`except` que se traga la aserción en la línea {node.lineno}")
    return reasons


def _outcome(run: Callable[[], None]) -> tuple[str, int | None]:
    """`("red", línea)` si `run` falla por una aserción suya; si no, qué pasó y `None`."""
    try:
        run()
    except AssertionError as error:
        traceback, line = error.__traceback__, None
        while traceback is not None:
            if traceback.tb_frame.f_code is run.__code__:
                line = traceback.tb_lineno
            traceback = traceback.tb_next
        return "red", line
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:  # noqa: BLE001 - un skip o un fallo ajeno NO es «rojo»
        return type(error).__name__, None
    return "green", None


def _witness_problems(witness: Witness, scratch: Path) -> list[str]:
    """Todo lo que impide creer a este testigo: pasa hoy, falla arreglado, cláusula a cláusula."""
    module, _, name = witness.node.partition("::")
    run = globals().get(name) if module == SELF else None
    if not callable(run) or inspect.signature(run).parameters:
        return [f"{witness.node}: un testigo es una función sin argumentos de {SELF}"]
    problems = []
    today, _ = _outcome(run)
    if today != "green":
        problems.append(f"{name} no pasa hoy ({today})")
    if not witness.fixes:
        problems.append(f"{name} no trae ningún mundo arreglado: nada demuestra que pueda fallar")
    fired: set[int | None] = set()
    for index, fix in enumerate(witness.fixes):
        with pytest.MonkeyPatch.context() as patched:
            fix(patched, scratch / f"{name}-{index}")
            verdict, line = _outcome(run)
        if verdict == "red":
            fired.add(line)
        else:
            problems.append(f"{name} sigue {verdict} en el mundo `{fix.__name__}`: está hueco")
    function = _function_def(witness.node)
    own = _own_nodes(function) if function is not None else iter(())
    clauses = {n.lineno for n in own if isinstance(n, ast.Assert)}
    if witness.fixes and fired != clauses:
        problems.append(
            f"{name}: sus mundos arreglados disparan {sorted(fired, key=str)} y sus cláusulas "
            f"están en {sorted(clauses)}: cada `assert` necesita el mundo que lo pone rojo"
        )
    return problems


WITNESSES: tuple[Witness, ...] = tuple(
    p for c in CLOSURE for p in c.proofs if isinstance(p, Witness)
)
FLOOR = f"{SELF}::test_no_named_proof_and_no_test_of_this_file_is_hollow"


def test_a_verdict_and_its_witness_travel_together() -> None:
    """Cumplido ⇒ ningún testigo; NO cumplido ⇒ al menos uno.

    Sin esto, pasar §13.1 a cumplido y sacarlo de `UNMET` —dos líneas del mismo fichero—
    quedaba verde con el testigo del «no cumple» todavía listado como prueba del «cumple»; y
    quitar el testigo de un criterio no cumplido, también.
    """
    wrong = [c.number for c in CLOSURE if c.met == any(isinstance(p, Witness) for p in c.proofs)]
    assert not wrong, f"el veredicto y el testigo no van juntos en §13.{wrong}"


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda w: w.node.rpartition("::")[2])
def test_every_witness_passes_today_and_fails_in_every_world_that_fixes_it(
    witness: Witness, tmp_path: Path
) -> None:
    """La regla 1, ejecutada en cada corrida: el testigo se ve rojo en cada mundo arreglado.

    Visto en rojo borrando SÓLO la aserción de
    `test_criterion_1_stays_unmet_while_a_quoted_query_is_a_disjunction`, la mutación que
    antes dejaba este fichero en 51 de 51 verde.
    """
    problems = _witness_problems(witness, tmp_path)
    assert not problems, "; ".join(problems)


PROOF_NODES: tuple[str, ...] = tuple(
    dict.fromkeys(
        [p.node for c in CLOSURE for p in c.proofs if isinstance(p, Node)]
        + [f"{SELF}::{name}" for name in sorted(_test_names(REPO_ROOT / SELF))]
    )
)


@pytest.mark.parametrize("node", PROOF_NODES, ids=lambda node: node.rpartition("::")[2])
def test_no_named_proof_and_no_test_of_this_file_is_hollow(node: str) -> None:
    """El suelo: un test nombrado —o una guarda de aquí— que ya no puede ponerse rojo."""
    function = _function_def(node)
    reasons = ["no existe"] if function is None else _hollow_reasons(function)
    assert not reasons, f"{node} está hueco: {reasons}"


_SPECIMEN = ""


def _specimen_hollow_witness() -> None:
    """Un testigo al que le han borrado la aserción: el espécimen de la mutación reproducida."""


def _specimen_two_clause_witness() -> None:
    assert "a" not in _SPECIMEN
    assert "b" not in _SPECIMEN


def _the_specimen_gains(text: str) -> Fix:
    def fix(patched: pytest.MonkeyPatch, root: Path) -> None:
        patched.setattr(sys.modules[__name__], "_SPECIMEN", text)

    fix.__name__ = f"_the_specimen_gains({text!r})"
    return fix


HOLLOW_SHAPES: dict[str, str] = {
    "sin aserción": "def test_x():\n    '''doc'''\n",
    "sólo `pass`": "def test_x():\n    pass\n",
    "assert constante": "def test_x():\n    assert 1 == 1\n",
    "skip decorado": "@pytest.mark.skip\ndef test_x():\n    assert f()\n",
    "skipif decorado": "@pytest.mark.skipif(True, reason='')\ndef test_x():\n    assert f()\n",
    "pytest.skip()": "def test_x():\n    pytest.skip('')\n    assert f()\n",
    "return delante": "def test_x():\n    return\n    assert f()\n",
    "if False": "def test_x():\n    if False:\n        assert f()\n",
    "except que traga": (
        "def test_x():\n    try:\n        assert f()\n    except AssertionError:\n        pass\n"
    ),
    "aserción sólo en una función anidada": "def test_x():\n    def g():\n        assert f()\n",
}


def test_the_guards_of_this_file_see_the_defect_they_exist_for(tmp_path: Path) -> None:
    """Las dos guardas se vigilan entre sí, porque ninguna puede vigilarse sola.

    El suelo tiene que ver cada forma de test hueco; el ejecutor de testigos, un testigo vaciado
    y un testigo con una cláusula sin mundo. Y el suelo se aplica AQUÍ al test del propio suelo:
    un test vaciado no se denuncia a sí mismo, y el ejecutor ya lo cubre el suelo.
    """
    blind = [
        shape for shape, source in HOLLOW_SHAPES.items() if not _hollow_reasons(_parse(source))
    ]
    specimens = {
        "testigo vaciado": Witness(
            f"{SELF}::_specimen_hollow_witness", fixes=(_the_specimen_gains("a"),)
        ),
        "cláusula sin mundo": Witness(
            f"{SELF}::_specimen_two_clause_witness", fixes=(_the_specimen_gains("a"),)
        ),
        "testigo sin mundos": Witness(f"{SELF}::_specimen_two_clause_witness"),
    }
    blind += [kind for kind, w in specimens.items() if not _witness_problems(w, tmp_path)]
    floor = _function_def(FLOOR)
    if floor is None or _hollow_reasons(floor):
        blind.append(f"{FLOOR} no existe o está hueco")
    assert not blind, f"guardas que ya no ven su defecto: {blind}"


def _parse(source: str) -> ast.FunctionDef:
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    return function


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
