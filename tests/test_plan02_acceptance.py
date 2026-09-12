"""La batería dirigida del §15 del Plan 02: la mitad fail-closed de su listón.

QUÉ PROBLEMA CIERRA. El §5.3 del documento de entrega atómica lleva una fila
«batería dirigida sobre los 15 criterios del Plan 02 §15 | 106 passed» que manda re-derivar
sobre `SNAPSHOT_FUENTE`. Nunca se re-derivó, y el informe del §5 midió por qué: la selección
que producía el 106 no está en el repositorio. Vivía en la prosa de un informe de revisión, en
catorce bloques `pytest -q` sueltos, y las cuatro lecturas defendibles de esas listas dan 81,
83, 104 y 112 **sobre el propio árbol donde se midió el 106**. Un número cuya selección no
existe no es un listón: es una cita.

QUÉ GUARDA ESTE FICHERO, y qué NO. Guarda que la selección versionada en
`eval/plan02-acceptance.yaml` sigue apuntando a tests que existen, que ningún criterio se queda
vacío, y que los criterios del §15 siguen todos representados o declarados. Es la mitad
fail-closed: borrar una entrada rompe la suite, que es la propiedad de la regla 11 —lo que se
quita ya se guarda solo—. NO ejecuta la batería ni cuenta sus tests: eso lo hace
`scripts/plan02-acceptance.sh`, porque una cifra va con el comando que la re-deriva y un
`pytest` anidado dentro de la suite mediría la suite, no la batería.

EL PUNTO DÉBIL, DECLARADO. La totalidad se comprueba contra `PLAN_02_CRITERIA`, que es una
COPIA a mano del §15 porque el plan vive en `zz-support-files/` y está gitignored. Una copia
sólo falla cerrado contra la omisión que NO comparte con lo copiado: nació sin `4c`, `4d` ni
`4e` —los tres estaban en el §15 y en ninguna de las dos listas del YAML— y el test de
totalidad pasó igual, porque la copia y la selección omitían exactamente lo mismo. Eso es la
regla 5 dentro del instrumento escrito para sustituir una cita. Lo que lo cierra es
`PLAN_02_COUNT`: el tamaño de la copia es también un listón, así que encogerla exige editar una
cifra, y una cifra que baja es lo único que un revisor ve en un diff.

LO QUE NO PUEDE VER. Ve NOMBRES, no comportamiento — el mismo límite que el barrido de símbolos
del informe del §5 declara antes de citar ninguna cifra. Un test renombrado que además afloje su
aserción pasa por aquí sin ruido. Lo concluyente es la dirección negativa: un node id que ya no
resuelve no está guardando nada.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = REPO_ROOT / "eval" / "plan02-acceptance.yaml"

# Los criterios del Plan 02 §15 — los quince numerados MÁS los cinco desdoblamientos que el
# documento numera él mismo (`4b`, `4c`, `4d`, `4e`, `8b`) — y la sección de seguridad del §12
# que la ronda 08 cuenta dentro de la batería. Escritos aquí para que la totalidad sea una
# ASERCIÓN y no una lectura: un criterio que desaparezca del fichero de selección tiene que
# hacer fallar algo.
#
# POR QUÉ ESTÁ ESCRITO A MANO, y qué lo vigila. El plan está en `zz-support-files/`, que es
# gitignored, así que esta lista NO se puede derivar: es una copia, y una copia es «las dos
# listas que deberían coincidir» de la regla 5 metida dentro del instrumento que existe para
# sustituir una cita. Nació con tres omisiones —`4c`, `4d` y `4e` estaban en el §15 y en
# ninguna de las dos listas del fichero de selección— y el test de totalidad pasaba igual,
# porque comparaba la copia contra la selección y las dos omitían lo mismo. Una copia sólo
# falla cerrado si su TAMAÑO es también un listón: `PLAN_02_COUNT` abajo obliga a que quitar
# un criterio de aquí aparezca en el diff como una cifra que baja, que es el mismo mecanismo
# que `listed_nodes` en el YAML y la única forma de que un revisor lo vea.
#
# Los cuatro desdoblamientos de `4` no son un invento de esta lista: `4b`, `4c`, `4d` y `4e`
# están numerados así en el §15, igual que `8b`. `4e` es un criterio de DOCUMENTACIÓN («si
# aparece la promesa, el criterio falla»), así que vive en `not_a_test_selection`, como el 15.
PLAN_02_CRITERIA: frozenset[str] = frozenset(
    {
        "1",
        "2",
        "3",
        "4",
        "4b",
        "4c",
        "4d",
        "4e",
        "5",
        "6",
        "7",
        "8",
        "8b",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
    }
) | {"sec"}

# Los 20 numerados del §15 más `sec`. Es el listón de la COPIA, y baja sólo editando esta
# línea: sin él, borrar un criterio de `PLAN_02_CRITERIA` y del YAML a la vez deja el test de
# totalidad verde midiendo menos, que es el fail-open que esta pasada encontró abierto.
PLAN_02_COUNT: int = 21


@pytest.fixture(scope="module")
def selection() -> dict:
    return yaml.safe_load(SELECTION_PATH.read_text(encoding="utf-8"))


def _nodes(selection: dict) -> list[tuple[str, str]]:
    """`(criterion id, node id)` for every entry, in file order."""
    return [
        (criterion["id"], entry["node"])
        for criterion in selection["criteria"]
        for entry in criterion["tests"]
    ]


def _test_functions(path: Path) -> set[str]:
    """Top-level `test_*` names of one test module, by `ast` — no import, no collection."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def test_every_node_of_the_acceptance_selection_exists_in_this_tree(selection: dict) -> None:
    """Un node id que ya no resuelve es un guardián que se fue sin que nadie lo declarara.

    Ésta es la aserción que hace fail-closed a la fila del §5.3: renombrar o borrar un test de
    la batería pone la suite en rojo, en vez de dejar que el recuento baje en silencio la
    próxima vez que alguien ejecute el guion.

    Visto en rojo apuntando una entrada a `::test_que_no_existe`: falla nombrando el criterio
    y el node id, que es lo que un lector necesita para decidir si fue renombrado o perdido.
    """
    missing: list[str] = []
    for criterion_id, node in _nodes(selection):
        module, _, name = node.partition("::")
        path = REPO_ROOT / module
        if not path.is_file():
            missing.append(f"[{criterion_id}] fichero ausente: {node}")
            continue
        functions = _test_functions(path)
        if not name:
            if not functions:
                missing.append(f"[{criterion_id}] fichero sin tests: {node}")
        elif name not in functions:
            missing.append(f"[{criterion_id}] test ausente: {node}")
    assert not missing, "la batería apunta a tests que este árbol no tiene:\n  " + "\n  ".join(
        missing
    )


def test_every_criterion_of_the_selection_names_at_least_one_test(selection: dict) -> None:
    """Un criterio vaciado deja la fila del §5.3 verde midiendo menos: el fail-open de la regla 11.

    Se comprueba sobre el fichero y no sobre el recuento total, porque un criterio vaciado y
    otro engordado se compensan en la suma y no en esta aserción.
    """
    empty = [criterion["id"] for criterion in selection["criteria"] if not criterion["tests"]]
    assert not empty, f"criterios sin ningún test: {empty}"


def test_the_fifteen_criteria_are_represented_or_declared(selection: dict) -> None:
    """Totalidad: cada criterio del §15 está en la selección o declarado como no-seleccionable.

    El criterio 15 es `scripts/check.sh` sobre el árbol del merge, no una selección de
    `pytest`; omitirlo en silencio y declararlo se leen igual en el fichero y distinto aquí.

    Visto en rojo borrando el bloque `not_a_test_selection`: el criterio 15 queda sin cubrir y
    la aserción lo nombra.
    """
    covered = {criterion["id"] for criterion in selection["criteria"]}
    declared = {entry["id"] for entry in selection.get("not_a_test_selection", ())}
    overlap = covered & declared
    assert not overlap, f"un criterio no puede estar seleccionado Y declarado sin tests: {overlap}"
    assert PLAN_02_CRITERIA == covered | declared, (
        "los criterios del §15 y los del fichero de selección no coinciden: "
        f"faltan {sorted(PLAN_02_CRITERIA - (covered | declared))}, "
        f"sobran {sorted((covered | declared) - PLAN_02_CRITERIA)}"
    )


def test_the_criteria_copy_cannot_shrink_without_the_number_moving() -> None:
    """La mitad fail-closed de la COPIA, que es la que faltaba.

    `test_the_fifteen_criteria_are_represented_or_declared` compara `PLAN_02_CRITERIA` con el
    YAML, así que una omisión PRESENTE EN LAS DOS es invisible para él: es exactamente como
    `4c`, `4d` y `4e` sobrevivieron a la selección que se escribió para acabar con las citas.
    Con el tamaño fijado, borrar un criterio de la copia obliga a bajar esta línea, y bajarla
    es la edición que se quiere ver.

    Los 20 numerados del §15 (`1`–`14`, `15`, con `4b`/`4c`/`4d`/`4e` y `8b`) más `sec`.

    Visto en rojo quitando `"4c"` del conjunto: `20 != 21`, sin necesidad de tocar el YAML.
    """
    assert len(PLAN_02_CRITERIA) == PLAN_02_COUNT, (
        "la copia a mano del §15 cambió de tamaño sin que `PLAN_02_COUNT` se moviera con ella: "
        f"{sorted(PLAN_02_CRITERIA)}"
    )
    # Los desdoblamientos son la parte que se pierde en silencio: un criterio numerado `4c` se
    # lee como parte del `4` y desaparece sin dejar hueco en la secuencia 1..15.
    assert {"4b", "4c", "4d", "4e", "8b"} <= PLAN_02_CRITERIA, (
        "falta alguno de los cinco desdoblamientos que el §15 numera él mismo"
    )


def test_the_selection_size_matches_the_number_it_publishes(selection: dict) -> None:
    """`listed_nodes` es el listón, y bajarlo exige editar la línea que lo dice.

    Sin esta aserción, quitar una entrada y no tocar nada más deja el fichero coherente
    consigo mismo; con ella, la rebaja tiene que aparecer en el diff como una cifra que baja,
    que es la única forma de que un revisor la vea.
    """
    nodes = [node for _criterion, node in _nodes(selection)]
    assert len(nodes) == selection["listed_nodes"]
    assert len(set(nodes)) == len(nodes), (
        "un node id repetido infla el listón sin añadir cobertura: "
        f"{sorted({n for n in nodes if nodes.count(n) > 1})}"
    )
