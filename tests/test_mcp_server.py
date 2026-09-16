# tests/test_mcp_server.py
"""El servidor MCP (Plan 04 §4): tres herramientas, y nada propio detrás de ellas.

TODO LO DE AQUÍ ENTRA POR LA SUPERFICIE PÚBLICA DE LAS DOS PUERTAS. Por MCP, un `mcp.Client`
conectado en proceso al servidor — el cliente de verdad, con su handshake y su envelope, no
la función del handler llamada a mano. Por el CLI, `CliRunner` sobre la app real. En 04.4
probar la función y dejar el camino real descubierto costó seis rondas: el defecto vivía
justo en el tramo que el test saltaba.

Este módulo es además el que sirve el ARNÉS a los otros dos (`test_mcp_cli_equivalence.py`,
`test_mcp_prompt_injection.py`): una sola definición de «workspace con el corpus de
fixture», de «llamar a una tool» y de «desempaquetar el envelope». Tres copias de eso serían
tres cosas que divergen en silencio, que es la regla 5 en la carpeta de tests.

Las corrutinas se conducen con `asyncio.run` de la stdlib a propósito: el árbol no tiene
`pytest-asyncio` y añadir un plugin es tocar `pyproject.toml` y el lock, que es el hijo 04.6
y ya está integrado.
"""

from __future__ import annotations

# `_socket` es el piso de abajo de `socket`, y el cerrojo del paso 27 se echa AHÍ a propósito:
# `socket.getaddrinfo` es un envoltorio Python que baja a `_socket.getaddrinfo` por atributo en
# cada llamada, así que cortar abajo alcanza también a una referencia capturada arriba antes de
# que el cerrojo existiera. El razonamiento completo está en la cabecera de la sección.
import _io
import _socket
import asyncio
import contextlib
import builtins
import gc
import io
import json
import os
import socket
import ssl
import stat
import sys
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager as ContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from typer.testing import CliRunner

from xbrain.cli import app
from xbrain.mcp_server import MCP_TOOLS, build_server

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS = FIXTURES / "knowledge_corpus.json"
runner = CliRunner()

# Las tres, literales. El conjunto viene del Plan 04 §4.1: `xbrain.search`, `xbrain.get` y
# `xbrain.graph_expand` son las tres puertas de los tres servicios.
EXPECTED_TOOLS = {"xbrain.search", "xbrain.get", "xbrain.graph_expand"}


# ---------------------------------------------------------------------------
# El arnés: un repo de mentira con el corpus de fixture, y las dos puertas
# ---------------------------------------------------------------------------


def make_workspace(
    tmp_path: Path,
    monkeypatch,
    *,
    get_char_budget: int | None = None,
    items: Mapping[str, Any] | None = None,
) -> Path:
    """Un directorio con forma de repo, con `data/` construido desde la fixture del corpus.

    El corpus es una FIXTURE de población conocida (12 items, 2 topics del vocabulario),
    nunca `data/`: en CI no hay store, y un test que fuese a buscarlo sería un test que deja
    de correr allí sin decirlo.

    `items` permite sustituir el diccionario de items —lo usa el test de inyección, que
    necesita un item con un texto concreto— sin duplicar el resto del montaje.
    """
    raw = json.loads(CORPUS.read_text(encoding="utf-8"))
    data = tmp_path / "data"
    data.mkdir()
    payload = raw["items"] if items is None else items
    (data / "items.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (data / "topics.json").write_text(json.dumps(raw["topics"], indent=2), encoding="utf-8")
    (data / "vocab.yaml").write_text(
        yaml.safe_dump({"topics": list(raw["vocab"].values())}, allow_unicode=True),
        encoding="utf-8",
    )
    index = "" if get_char_budget is None else f"[index]\nget_char_budget = {get_char_budget}\n"
    (tmp_path / "config.toml").write_text(
        '[paths]\nvault = "vault"\noutput_subdir = "x-knowledge"\ndata_dir = "data"\n'
        '[x]\nhandle = "vgonpa"\n' + index,
        encoding="utf-8",
    )
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def build_index() -> None:
    """`xbrain index build` en el workspace actual."""
    result = runner.invoke(app, ["index", "build"])
    assert result.exit_code == 0, result.output


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch) -> Path:
    """El repo de mentira, SIN índice: es el estado del que habla el paso 25."""
    return make_workspace(tmp_path, monkeypatch)


@pytest.fixture()
def indexed_workspace(workspace: Path) -> Path:
    """El repo de mentira con el índice ya construido."""
    build_index()
    return workspace


def run_cli(argv: Sequence[str]) -> str:
    """stdout de `xbrain <cmd> --json`, entero. Una línea de log suelta rompe el `json.loads`."""
    result = runner.invoke(app, [*argv, "--json"])
    assert result.exit_code == 0, result.output
    return result.stdout


def cli_error(argv: Sequence[str]) -> str:
    """El mensaje del CLI cuando se niega: código 1 y la PRIMERA línea `Error: …`.

    La primera, y no el stderr entero, porque en esta rama `_handle_cli_errors` envuelve a
    `_handle_index_errors` y `typer.Exit` hereda de `RuntimeError`, así que el de fuera
    vuelve a atrapar la salida del de dentro y añade un `Error:` vacío detrás. Es un defecto
    cosmético anterior a este PR y ajeno a MCP; aquí sólo se esquiva, no se toca.
    """
    result = runner.invoke(app, [*argv, "--json"])
    assert result.exit_code == 1, result.output
    first = result.stderr.strip().splitlines()[0]
    assert first.startswith("Error: "), result.stderr
    return first.removeprefix("Error: ")


def cli_stderr(argv: Sequence[str]) -> str:
    """El stderr ENTERO del CLI cuando se niega. Para mensajes de varias líneas.

    `cli_error` se queda con la primera porque compara el mensaje contra el de MCP; un error
    de validación de Pydantic ocupa cuatro líneas y la enumeración de valores válidos está en
    la tercera, así que quedarse con la primera sería mirar sólo el encabezado.
    """
    result = runner.invoke(app, [*argv, "--json"])
    assert result.exit_code == 1, result.output
    return result.stderr


def call_mcp_tool(tool: str, arguments: dict[str, Any]) -> Any:
    """La herramienta, llamada por un cliente MCP real conectado en proceso al servidor."""

    async def _call() -> Any:
        from mcp import Client

        async with Client(build_server()) as client:
            return await client.call_tool(tool, arguments)

    return asyncio.run(_call())


def unwrap_mcp_content(result: Any) -> str:
    """El payload de la tool, DESEMPAQUETADO de su envelope MCP.

    Comprueba de paso que la llamada no fue un error y que el contenido es UN bloque de
    texto: un segundo bloque, o un `is_error` silencioso, convertiría el resto del test en
    una comparación contra lo que se le ocurriese al SDK.
    """
    assert result.is_error is False, result.content
    blocks = [block for block in result.content if block.type == "text"]
    assert len(blocks) == 1, [block.type for block in result.content]
    return blocks[0].text


def mcp_error(result: Any) -> str:
    """El texto del error estructurado que ve el agente."""
    assert result.is_error is True, result.content
    blocks = [block for block in result.content if block.type == "text"]
    assert len(blocks) == 1, [block.type for block in result.content]
    return blocks[0].text


# ---------------------------------------------------------------------------
# Paso 23: las tres tools
# ---------------------------------------------------------------------------


def test_the_server_serves_these_three_tools_and_nothing_else() -> None:
    """Paso 23: las tres tools existen — y NINGUNA MÁS.

    Se lee lo que el servidor SIRVE (`list_tools()`, la respuesta del protocolo), no lo que
    el módulo declara: una constante y un registro pueden divergir, y el agente sólo ve el
    segundo. `MCP_TOOLS` se comprueba después contra la misma referencia, para que una
    herramienta registrada a mano y ausente de la constante también salga roja.
    """
    served = {tool.name for tool in asyncio.run(build_server().list_tools())}
    assert served == EXPECTED_TOOLS
    assert set(MCP_TOOLS) == EXPECTED_TOOLS


# ---------------------------------------------------------------------------
# Paso 25: los errores estructurados (§4.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorCase:
    """Una negativa, dicha en los dos idiomas, y la frase que la hace reconocible."""

    id: str
    argv: tuple[str, ...]
    tool: str
    arguments: dict[str, Any]
    # Un fragmento que el mensaje TIENE que llevar. Sin esto la igualdad podría cumplirse
    # entre dos mensajes vacíos, o entre dos genéricos que no dicen qué pasó.
    names: str
    indexed: bool = True


ERROR_CASES: tuple[ErrorCase, ...] = (
    ErrorCase(
        id="missing-index-search",
        argv=("search", "retrieval"),
        tool="xbrain.search",
        arguments={"query": "retrieval"},
        names="xbrain index build",
        indexed=False,
    ),
    ErrorCase(
        id="missing-index-graph",
        argv=("graph-expand", "--item", "k03"),
        tool="xbrain.graph_expand",
        arguments={"item_id": "k03"},
        names="xbrain index build",
        indexed=False,
    ),
    ErrorCase(
        id="empty-query",
        argv=("search", ""),
        tool="xbrain.search",
        arguments={"query": ""},
        names="vacía",
    ),
    ErrorCase(
        id="unknown-topic",
        argv=("search", "agent", "--topic", "no-existe"),
        tool="xbrain.search",
        arguments={"query": "agent", "filters": {"topics": ["no-existe"]}},
        names="agent-evaluation",
    ),
    ErrorCase(
        id="unknown-item",
        argv=("get", "no-existe"),
        tool="xbrain.get",
        arguments={"item_id": "no-existe"},
        names="no-existe",
    ),
    ErrorCase(
        id="unknown-surface",
        argv=("get", "k03", "--surface", "no-existe"),
        tool="xbrain.get",
        arguments={"item_id": "k03", "surfaces": ["no-existe"]},
        names="Superficies disponibles",
    ),
    ErrorCase(
        # `strategy` se declara `str` y no `Literal` justo para que el rechazo lo dé el
        # servicio, enumerando las estrategias, y no el esquema con un error de validación.
        id="unknown-strategy",
        argv=("search", "agent", "--strategy", "no-existe"),
        tool="xbrain.search",
        arguments={"query": "agent", "strategy": "no-existe"},
        names="implementadas hoy",
    ),
)


# El envoltorio que el SDK pone ALREDEDOR del mensaje, y lo ÚNICO que se le concede.
#
# Lo impone `mcp/server/mcpserver/tools/base.py`, que re-lanza todo fallo de una tool como
# `ToolError(f"Error executing tool {self.name}: {exc}")`. No es negociable desde aquí: es el
# texto que el agente lee, venga de donde venga. Por eso se fija ENTERO y se compara EXACTO.
#
# La comparación floja de antes —`case.tool in text` más `text.endswith(message)`— dejaba sin
# atar todo lo que hubiera EN MEDIO, que es justo donde viviría un formato propio: medido,
# cambiar `ToolError(str(exc))` por `ToolError(f"mutated error: {exc}")` dejaba los siete
# casos en verde. Y eso es lo que el §4.1 («ni un formato propio») y el §4.3 («el MISMO error
# estructurado que el CLI») niegan.
#
# Si un día el SDK cambia este envoltorio, este test se pone rojo, y debe: lo que cambia es
# el texto que ve el agente.
MCP_ERROR_FRAMING = "Error executing tool {tool}: {message}"


@pytest.mark.parametrize("case", ERROR_CASES, ids=lambda case: case.id)
def test_mcp_refuses_with_the_same_message_as_the_cli(case: ErrorCase, workspace: Path) -> None:
    """Paso 25 / §4.3: MCP da EL MISMO error estructurado que el CLI, no uno genérico.

    Sin traducción explícita el SDK convierte cualquier excepción que no sea un `ToolError`
    en un `UnexpectedToolError` cuyo texto para el agente es literalmente `Error executing
    tool xbrain.search`: el operador recibe «constrúyelo con `xbrain index build`» y el
    agente, nada. Esa asimetría es lo que este test caza.

    Se exige IGUALDAD EXACTA contra el envoltorio obligatorio del SDK con el mensaje del CLI
    dentro. «Mismo error» significa el mismo, no uno que lo contenga: un prefijo propio, una
    coletilla, un código inventado o un reformateo caben enteros dentro de un `in` y de un
    `endswith`, y cualquiera de ellos es la puerta MCP hablando un idioma que el operador no
    oye. La igualdad absorbe además el viejo `"Traceback" not in text`: un traceback en el
    texto es, por construcción, un texto distinto del esperado.
    """
    if case.indexed:
        build_index()
    message = cli_error(case.argv)
    assert case.names in message, message

    text = mcp_error(call_mcp_tool(case.tool, case.arguments))
    assert text == MCP_ERROR_FRAMING.format(tool=case.tool, message=message), text


# ---------------------------------------------------------------------------
# Paso 26: read-only (§4.3)
# ---------------------------------------------------------------------------

# La escritura de prueba. Un `CREATE TABLE` es una escritura válida sea cual sea el esquema,
# así que un fallo sólo puede venir de que la base esté abierta en sólo lectura — y no de un
# `NOT NULL` o una columna que no existe, que es lo que un `INSERT` arriesgaría.
WRITE_PROBE = "CREATE TABLE _mcp_write_probe (x)"


def spy_on_index_writes(monkeypatch) -> list[BaseException | None]:
    """Cada apertura del índice intenta escribir en ÉL, y anota qué pasó.

    Se parchea el nombre en los módulos que lo USAN, no en `index_store`: los dos servicios
    hacen `from … import open_for_query`, así que parchear el origen no los alcanzaría y el
    espía se quedaría mirando una puerta por la que no pasa nadie.

    La escritura se intenta DENTRO del espía a propósito: los servicios cierran la conexión
    en un `finally`, así que intentarlo después daría «cannot operate on a closed database» —
    un error que también se cumple con la base abierta en lectura y escritura, y que por
    tanto no probaría nada.
    """
    from xbrain.knowledge import graph_service, index_store, search_service

    attempts: list[BaseException | None] = []
    real = index_store.open_for_query

    def spy(*args: Any, **kwargs: Any) -> Any:
        index = real(*args, **kwargs)
        try:
            index.lexical.connection.execute(WRITE_PROBE)
        except Exception as exc:  # noqa: BLE001 - se anota, se comprueba abajo
            attempts.append(exc)
        else:
            attempts.append(None)
        return index

    for module in (search_service, graph_service):
        monkeypatch.setattr(module, "open_for_query", spy)
    return attempts


def test_the_index_the_mcp_tools_open_refuses_a_write(indexed_workspace: Path, monkeypatch) -> None:
    """Paso 26 / §4.3: la base que abren las tools RECHAZA una escritura.

    «Comprobamos que no escribe» es una afirmación sobre el código; no poder escribir es una
    propiedad del objeto. Lo que se interroga aquí es la conexión EXACTA que sirvió la
    llamada MCP, no una que el test abra por su cuenta.
    """
    import sqlite3

    attempts = spy_on_index_writes(monkeypatch)
    call_mcp_tool("xbrain.search", {"query": "retrieval"})
    call_mcp_tool("xbrain.graph_expand", {"item_id": "k03"})

    assert len(attempts) == 2, "alguna tool no llegó a abrir el índice"
    for attempt in attempts:
        assert isinstance(attempt, sqlite3.OperationalError), attempt
        assert "readonly" in str(attempt), attempt


def _digest(root: Path) -> dict[str, str]:
    """sha256 de las tres entradas del store Y de cada fichero del índice."""
    import hashlib

    files = [root / "data" / name for name in ("items.json", "vocab.yaml", "topics.json")]
    files += sorted(p for p in (root / "data" / "index").rglob("*") if p.is_file())
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }


def test_no_mcp_tool_writes_to_the_store_or_the_index(indexed_workspace: Path) -> None:
    """Paso 26 / §4.3: después de las tres herramientas, ni un byte ha cambiado.

    Las tres entradas y TODOS los ficheros del índice, no sólo `items.json`: una herramienta
    que reescribiese `vocab.yaml` o compactase la base pasaría una comprobación de un fichero.
    """
    before = _digest(indexed_workspace)
    assert len(before) > 3, "el índice no se construyó: la comprobación sería vacía"
    for tool, arguments in (
        ("xbrain.search", {"query": "retrieval"}),
        ("xbrain.get", {"item_id": "k03"}),
        ("xbrain.graph_expand", {"item_id": "k03"}),
    ):
        unwrap_mcp_content(call_mcp_tool(tool, arguments))
    assert _digest(indexed_workspace) == before


def test_mcp_get_answers_with_the_index_deleted(indexed_workspace: Path) -> None:
    """Invariante 7 del spec §3.7, heredado: `get` lee el STORE, nunca el índice.

    Vale la pena por MCP y no sólo por el servicio: si la herramienta hubiese añadido una
    consulta al índice «para enriquecer», el invariante se rompería por la puerta nueva y el
    test del servicio seguiría verde.
    """
    import shutil

    shutil.rmtree(indexed_workspace / "data" / "index")
    payload = json.loads(unwrap_mcp_content(call_mcp_tool("xbrain.get", {"item_id": "k03"})))
    assert payload["item"]["item_id"] == "k03"
    assert payload["surfaces"], "sin superficies la comprobación pasaría por vacío"


# ---------------------------------------------------------------------------
# Paso 27: sin red (§4.3, §10.5)
# ---------------------------------------------------------------------------
#
# LA COSTURA, Y POR QUÉ ESTA. Van cuatro rondas y el hueco ha aparecido siempre en el mismo
# sitio por un eje nuevo —datagramas, preconectados, descriptor desnudo, clase base en C,
# enlace temprano— porque el cerrojo se echaba sobre una LISTA DE NOMBRES ESCRITA A MANO, y
# de una lista escrita a mano siempre falta un nombre. La prueba de que ése era el problema y
# no los tres nombres es que, escribiendo esto, la auditoría encontró CUATRO más por el mismo
# mecanismo —un alias del tipo (`_socket.SocketType`), un método ligado precapturado
# (`bound = sock.send`), el descriptor no ligado (`DESC = _socket.socket.send`) y regalar el
# descriptor a un objeto fichero—, todos medidos entregando bytes. Así que el cambio no es
# añadir siete filas: es dejar de escribir la lista.
#
# Un byte sale a la red desde Python por UN SOCKET o por SU DESCRIPTOR, y no hay un tercero
# porque el kernel no ofrece un tercero. Lo que cambia por piso es dónde vive el nombre, y por
# eso el corte va en sitios distintos: el socket se corta como OBJETO (no nace, y si ya
# existía tiene los métodos puestos), y el descriptor se corta al ESCRIBIRSE y al REGALARSE,
# preguntándole la familia al kernel. La resolución de nombres es una tercera clase que no
# mueve bytes del corpus pero sí saca una consulta. Y en los tres, la lista de lo que se
# corta se DERIVA del intérprete en vez de escribirse:
#
#   1. EL OBJETO NO NACE. Ningún socket INET puede construirse mientras el cerrojo está
#      echado. Los nombres por los que se construye NO se enumeran: se buscan por IDENTIDAD
#      —todo atributo de `socket`/`_socket` que ES el tipo real (`_SOCKET_TYPE_ALIASES`)—, que
#      es lo único que encontró `_socket.SocketType`, un alias del MISMO tipo inmutable que
#      `_socket.socket`, y por tanto una cuarta entrada idéntica a la del `_socket.socket`
#      que reportó la revisión.
#   2. EL OBJETO QUE YA EXISTÍA tiene sus métodos de salida parcheados, y SE COMPRUEBA que no
#      quede nada fuera: `_egress_outside_the_lock` recorre los objetos VIVOS y exige dos
#      cosas — que todo socket INET resuelva sus métodos de salida al parche, y que no haya
#      por ahí una REFERENCIA A UN MÉTODO capturada antes del cerrojo (`bound = sock.send`
#      entrega bytes con la clase ya parcheada: medido). Si queda cualquiera de las dos,
#      `block_network` SE NIEGA a echarse en vez de mentir. Esto es lo que hace el cierre
#      DEMOSTRABLE en vez de argumentado: lo que no se puede parchear —un tipo C inmutable,
#      una referencia ya capturada— se puede al menos demostrar ausente.
#   3. EL DESCRIPTOR se corta en DOS sitios, y en ninguno de los dos decide un nombre: decide
#      el KERNEL. `_fd_family` pregunta por el descriptor (`fstat` → ¿es socket? → `dup` →
#      `family`), así que el fichero temporal, el store y la salida de pytest pasan y un
#      socket INET no. Eso es lo que hacía imposible la versión anterior de este docstring,
#      que declaraba la salida como no cubierta: no hacía falta «reconstruir un socket en cada
#      escritura», sólo un `fstat` (0,96 µs, medido) que descarta de un golpe todo lo que no
#      es un socket. Los dos sitios son (a) ESCRIBIR por el descriptor —`os.write`,
#      `os.writev`, `os.sendfile`— y (b) REGALARLO: copiarlo (`os.dup`, `os.dup2`,
#      `socket.dup`, `_socket.dup`) o envolverlo en un objeto fichero (`io.FileIO`, `io.open`
#      y sus alias, `os.fdopen`). (b) es la SÉPTIMA salida de esta serie y salió de la
#      auditoría propia de esta ronda: cortar (a) no cierra (b), porque el `write` de un
#      objeto fichero es C y no pasa por `os.write` — medido, `io.FileIO(sock.fileno(),
#      "wb").write(b"s")` entregó el byte con `os.write` ya parcheado.
#   4. LA RESOLUCIÓN se corta en el PISO DE ABAJO, el módulo `_socket`. `socket.getaddrinfo`
#      es un envoltorio Python que llama a `_socket.getaddrinfo` POR ATRIBUTO en cada llamada
#      (`socket.py:977`), así que parchear abajo alcanza también a una referencia capturada
#      arriba antes de que el cerrojo existiera — que es exactamente la fuga por enlace
#      temprano que reportó la revisión.
#   5. Y LO QUE NINGÚN PARCHE ALCANZA —una referencia capturada antes del cerrojo, cuando lo
#      capturado ES el objeto de C y no un envoltorio: `from socket import gethostbyname`, o
#      `DESC = _socket.socket.send`— no se persigue con más parches, porque no se puede: se
#      PROHÍBE en el código bajo prueba, y lo prohíbe un test que recorre el grafo de imports
#      de xbrain y se pone rojo si alguno la guarda en un global. La otra forma de captura, el
#      método LIGADO, sí está cubierta del todo, y la cubre el punto 2: los objetos
#      `builtin_function_or_method` están rastreados por el gc (medido), así que el barrido
#      los encuentra esté donde estén guardados.
#
# Los cuatro tests de totalidad del final cierran el bucle: la lista de lo que se parchea se
# compara contra `dir(socket.socket)`, `dir(socket)`, `dir(_socket)` y
# `ssl.SSLSocket.__dict__`, y todo nombre que no esté parcheado tiene que estar CLASIFICADO a
# mano con su razón. Una versión de Python que añada un método nuevo deja ese nombre sin
# clasificar y el test se pone ROJO — y en la otra dirección, borrar una fila deja huérfana su
# clasificación y también se pone rojo (`_CUT_ONE_FLOOR_BELOW`). Ahí es donde termina el
# regreso de la regla 3: la ausencia del guardián falla cerrado.


class NoNetworkAllowed(AssertionError):
    """Alguien intentó salir a la red durante una consulta."""


class NetworkLockIncomplete(AssertionError):
    """El cerrojo no puede echarse: queda vivo un socket de red al que no alcanza.

    Es una clase DISTINTA de `NoNetworkAllowed` a propósito. Si el barrido levantase
    `NoNetworkAllowed`, cualquier fila de `_EGRESS` podría ponerse verde por el barrido en
    vez de por su propio intento — un `pytest.raises` satisfecho por el motivo equivocado,
    que es la regla 1 exacta.
    """


# El destino de todos los intentos: el puerto DISCARD (9) en loopback. Loopback para que un
# control mal echado no mande nada fuera de la máquina, y el puerto 9 porque nadie escucha
# ahí — un intento que atraviese el cerrojo se pierde, no se entrega.
BLACKHOLE = ("127.0.0.1", 9)

# Las dos familias que llegan a la red. El cerrojo se decide por AQUÍ —por la familia del
# socket—, nunca por el nombre del método: `send` es a la vez la salida a internet de
# cualquier cliente HTTP y el despertador del bucle de eventos de asyncio, y sólo la familia
# distingue una cosa de la otra.
_INET_FAMILIES = frozenset({socket.AF_INET, socket.AF_INET6})

# El tipo C de verdad, capturado al importar y antes de que nadie lo sustituya. Hace falta
# guardarlo porque sus MÉTODOS no se pueden parchear: `_socket.socket` es un tipo INMUTABLE
# —medido: `TypeError: cannot set 'send' attribute of immutable type '_socket.socket'`—, así
# que «parchear la clase base» no es una opción disponible y lo único que se puede mover es
# el ATRIBUTO de módulo que apunta a ella.
_REAL_SOCKET_TYPE = _socket.socket

# Y el `os.dup` de verdad, por la misma razón: la sonda de familia duplica el descriptor, y
# desde que `os.dup` lleva parche la sonda se rechazaría a sí misma en la primera llamada.
_REAL_DUP = os.dup

# Los dos tipos de método LIGADO: el de una función Python (nuestro parche, una vez puesto)
# y el de un descriptor de C (el original, y lo que lleva una referencia capturada antes del
# cerrojo). El barrido filtra por ESTOS y no por `getattr(obj, "__self__")`, porque preguntar
# un atributo a un objeto arbitrario ejecuta código ajeno — ver `_egress_outside_the_lock`.
_BOUND_METHOD_TYPES: tuple[type, ...] = (types.MethodType, types.BuiltinMethodType)

# La marca que llevan los envoltorios del cerrojo. El barrido de sockets vivos pregunta por
# ELLA y no por el módulo en que está definida la función: una comprobación por `__module__`
# adivina, y ésta no.
_LOCK_TAG = "_xbrain_network_lock"


def _fd_family(fd: int) -> int | None:
    """La familia del socket que hay detrás de un descriptor, PREGUNTADA AL KERNEL.

    Es la función que hace posible cortar el descriptor desnudo sin cortar el resto del
    proceso. `os.write` es por donde salen también el fichero temporal, el store y la salida
    de pytest, así que la decisión no puede ser «es un fd, lo rechazo»: tiene que ser «el
    kernel dice que este fd es un socket INET».

    El coste es el argumento de que esto es viable: un `fstat` (0,96 µs medidos) descarta de
    un golpe todo lo que no es un socket —que es casi todo—, y sólo para un socket de verdad
    se paga el `dup` + construir la sonda + `close` (1,62 µs medidos).

    El `dup` es lo que deja el descriptor original INTACTO: la sonda se queda con la copia y
    al cerrarse cierra la copia. Construirla con `_REAL_SOCKET_TYPE` y no con `socket.socket`
    es deliberado: el tipo real no pasa por el guardián de nacimiento, así que la sonda
    funciona igual con el cerrojo echado.

    Si el fd es un socket pero no se puede averiguar su familia, devuelve AF_INET: falla
    CERRADO. Un falso positivo cuesta un rechazo en una escritura a un socket; un falso
    negativo entrega bytes a la red.
    """
    try:
        if not stat.S_ISSOCK(os.fstat(fd).st_mode):
            return None
    except OSError:
        return None
    try:
        duplicate = _REAL_DUP(fd)
    except OSError:
        return socket.AF_INET
    try:
        probe = _REAL_SOCKET_TYPE(fileno=duplicate)
    except OSError:
        os.close(duplicate)
        return socket.AF_INET
    try:
        return int(probe.family)
    finally:
        probe.close()


def _tag(refuse: Callable[..., Any]) -> Callable[..., Any]:
    """Marca un envoltorio como parte del cerrojo, para que el barrido lo reconozca."""
    setattr(refuse, _LOCK_TAG, True)
    return refuse


def _refuse_always(original: Callable[..., Any]) -> Callable[..., Any]:
    """Rechaza la llamada entera. Para las funciones de MÓDULO.

    `create_connection`, `getaddrinfo` y las demás entradas de resolución no son métodos de
    un socket: no hay un `self.family` que mirar, y tampoco hace falta. Existen para alcanzar
    la red y nada más —no hay uso local de ninguna—, así que cortarlas de raíz no puede
    romper IPC que el arnés necesite.
    """

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise NoNetworkAllowed(f"salida a la red durante una consulta: {args[:2]}")

    return _tag(refuse)


def _refuse_on_inet(original: Callable[..., Any]) -> Callable[..., Any]:
    """Rechaza SÓLO si el socket alcanza la red; delega si es local. Para los MÉTODOS.

    Los métodos de un socket tienen dos vidas con el mismo nombre. `send` sobre AF_INET es
    salir a internet; `send` sobre el AF_UNIX del self-pipe de asyncio es cómo el bucle de
    eventos se despierta a sí mismo (`csock.send(b"\0")` en `asyncio.selector_events`), y
    ese bucle es quien conduce al cliente MCP de estos tests. Un rechazo a ciegas mataría el
    servidor y el verde de abajo pasaría a significar «no llegó a ejecutarse nada», que es
    la forma más barata de fabricar un verde.

    Filtrar por familia parte esas dos vidas por donde de verdad se separan. Se mide, además,
    y no se supone: `asyncio.new_event_loop()._csock.family` es `AF_UNIX` en esta máquina.
    """

    def refuse(self: Any, *args: Any, **kwargs: Any) -> Any:
        if self.family in _INET_FAMILIES:
            raise NoNetworkAllowed(
                f"salida a la red durante una consulta: familia {self.family!r}, {args[:1]}"
            )
        return original(self, *args, **kwargs)

    return _tag(refuse)


def _refuse_on_inet_fd(original: Callable[..., Any]) -> Callable[..., Any]:
    """Rechaza una escritura por DESCRIPTOR si el descriptor es un socket INET.

    El piso de abajo de `_refuse_on_inet`: aquí no hay objeto socket del que leer `family`,
    sólo un entero, así que la familia se le pregunta al kernel (`_fd_family`). Las tres
    funciones que lleva —`os.write`, `os.writev`, `os.sendfile`— toman el descriptor de
    DESTINO como primer argumento, que es lo que hace que un solo envoltorio sirva para las
    tres.
    """

    def refuse(fd: int, *args: Any, **kwargs: Any) -> Any:
        if _fd_family(fd) in _INET_FAMILIES:
            raise NoNetworkAllowed(f"salida a la red por el descriptor desnudo {fd}")
        return original(fd, *args, **kwargs)

    return _tag(refuse)


def _refuse_on_inet_source_fd(original: Callable[..., Any]) -> Callable[..., Any]:
    """Rechaza COPIAR el descriptor de un socket INET. Para `dup`/`dup2`.

    Una copia del descriptor es una salida a la red con otro número: el `write` de un objeto
    fichero construido sobre ella es C y no pasa por `os.write`. Se corta en la copia, que sí
    tiene nombre en Python, y no en la escritura, que no lo tiene.
    """

    def refuse(fd: int, *args: Any, **kwargs: Any) -> Any:
        if _fd_family(fd) in _INET_FAMILIES:
            raise NoNetworkAllowed(f"copia del descriptor de un socket de red: {fd}")
        return original(fd, *args, **kwargs)

    return _tag(refuse)


def _refuse_wrapping_an_inet_fd(original: Callable[..., Any]) -> Callable[..., Any]:
    """Rechaza ENVOLVER el descriptor de un socket INET en un objeto fichero de Python.

    Sólo mira el caso del descriptor: con una ruta se delega, que es lo que deja pasar el
    `open(__file__, "rb")` de esta misma sección, el store y todo lo que escribe pytest.
    `bool` se excluye a mano porque es un `int` y `open(True)` no es abrir un descriptor.
    """

    def refuse(file: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            isinstance(file, int)
            and not isinstance(file, bool)
            and _fd_family(file) in _INET_FAMILIES
        ):
            raise NoNetworkAllowed(
                f"objeto fichero sobre el descriptor de un socket de red: {file}"
            )
        return original(file, *args, **kwargs)

    return _tag(refuse)


def _resolution_attempt_on_fd(owner: Any, name: str) -> Callable[[Any], Any]:
    """Llamar `<owner>.<name>(sock.fileno())`. El `getattr` va en la llamada, por el parche."""

    def attempt(sock: socket.socket) -> Any:
        return getattr(owner, name)(sock.fileno())

    return attempt


class _GuardedSocketType(_REAL_SOCKET_TYPE):  # type: ignore[misc,valid-type]
    """Ocupa el sitio del tipo real mientras el cerrojo está echado: un socket INET NO NACE.

    Es el arma contra la clase entera de fugas «un método que no enumeré»: si el objeto no
    existe, no hay método al que llamar. Cubre por construcción `_socket.socket(AF_INET, …)`,
    `_socket.SocketType(…)`, `socket.SocketType(…)`, `socket.socket()` (que resuelve su
    familia por defecto a AF_INET y llama aquí por `socket.py:233`), `socket.fromfd`,
    `socket.create_server` y cualquier nombre futuro que construya por uno de esos alias.

    Delega en `_REAL_SOCKET_TYPE.__init__` EXPLÍCITAMENTE y no por `super()`. `socket.py:233`
    hace `_socket.socket.__init__(self, …)` con un `self` que es un `socket.socket` —
    hermano de esta clase, no instancia de ella—, y el `super()` de argumento cero exigiría
    `isinstance(self, _GuardedSocketType)` y levantaría `TypeError`. Medido: con la
    delegación explícita, `socket.socketpair()` y `asyncio.new_event_loop()` siguen
    funcionando con el cerrojo puesto.

    Con `fileno` la familia llega como -1 y se delega: es lo que necesita la sonda de
    `_fd_family`, y reconstruir un socket desde un descriptor que ya existe no abre una
    salida nueva — sus métodos son los de la subclase parcheada.
    """

    def __init__(
        self, family: int = -1, type: int = -1, proto: int = -1, fileno: Any = None
    ) -> None:
        if family in _INET_FAMILIES:
            raise NoNetworkAllowed(f"nacimiento de un socket de red: familia {family!r}")
        _REAL_SOCKET_TYPE.__init__(self, family, type, proto, fileno)


def _install_the_guarded_type(original: Callable[..., Any]) -> Callable[..., Any]:
    """El «parche» de una fila de nacimiento: sustituir el alias por el tipo guardián."""
    return cast(Callable[..., Any], _GuardedSocketType)


# Todo atributo de `socket`/`_socket` que ES el tipo real, buscado por IDENTIDAD y no por
# nombre. Esto es la costura, no una lista: `_socket.SocketType` y `socket.SocketType` son
# el MISMO objeto que `_socket.socket` (medido), así que parchear sólo `_socket.socket`
# deja dos constructores sin guardar — la cuarta entrada, idéntica a la que reportó la
# revisión y que ninguna lista escrita a mano iba a contener. `socket.socket` NO sale aquí
# (es la subclase, no el tipo real) y no hace falta: hereda el guardián por `socket.py:233`.
_SOCKET_TYPE_ALIASES: tuple[tuple[Any, str], ...] = tuple(
    (owner, name)
    for owner in (_socket, socket)
    for name in sorted(dir(owner))
    if not name.startswith("__") and getattr(owner, name, None) is _REAL_SOCKET_TYPE
)


def _udp_of(family: int) -> socket.socket:
    """Un socket UDP SIN conectar, de la familia que toque."""
    return socket.socket(family, socket.SOCK_DGRAM)


def _blackhole(family: int) -> tuple[str, int]:
    """El mismo puerto DISCARD, en el loopback de la familia que toque."""
    return ("127.0.0.1" if family == socket.AF_INET else "::1", 9)


def _skip_without_ipv6(family: int) -> None:
    """Salta la fila si esta máquina no tiene loopback IPv6: no se puede ejercer."""
    if family == socket.AF_INET6 and not HAS_IPV6_LOOPBACK:
        pytest.skip("esta máquina no tiene loopback IPv6: el control v6 no puede ejercerse")


@contextlib.contextmanager
def _fresh_udp(family: int = socket.AF_INET) -> Iterator[socket.socket]:
    """Un socket UDP creado AQUÍ, antes del cerrojo, y sin conectar.

    Existe por la regla 1. Desde que el cerrojo prohíbe el NACIMIENTO de un socket INET, un
    control que llamase a `socket.socket(AF_INET, …)` dentro de `attempt` se pondría verde
    por el guardián de nacimiento y no por el parche de su propio método: la fila pasaría con
    su parche quitado. Creando el socket antes, el `sendto`/`connect` de abajo sólo puede
    fallar por su propia fila.
    """
    _skip_without_ipv6(family)
    with _udp_of(family) as sock:
        yield sock


@contextlib.contextmanager
def _preconnected_udp(family: int) -> Iterator[socket.socket]:
    """Un socket UDP YA CONECTADO — y conectado AQUÍ, antes de que el cerrojo exista.

    Esta función es el control entero. Un `send` no lleva destino: lo saca de un `connect`
    ANTERIOR, y ese connect puede haber ocurrido mucho antes de que nadie echase el cerrojo
    — al importar un módulo, al construir un cliente HTTP en el arranque. Cortar `connect`
    no alcanza a lo que ya pasó por él: el destino vive en el kernel, donde no llega ningún
    monkeypatch. Por eso el control conecta ANTES, y por eso `Egress.setup` abre antes que
    `block_network` en el test. Un control que conectase después no reproduciría nada: se
    estrellaría contra el `connect` parcheado y se pondría verde por el motivo equivocado.
    """
    _skip_without_ipv6(family)
    with _udp_of(family) as sock:
        sock.connect(_blackhole(family))
        yield sock


@contextlib.contextmanager
def _preconnected_tcp() -> Iterator[socket.socket]:
    """Lo mismo para TCP, que es lo que `sendfile` exige: un flujo ya establecido.

    El par entero vive en loopback —un `listen` efímero y su `accept`— porque `sendfile`
    necesita un socket de verdad conectado a alguien de verdad, y un destino externo haría
    que este control dependiese de la red que el cerrojo existe para prohibir.
    """
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        with socket.socket() as client:
            client.connect(listener.getsockname())
            with listener.accept()[0]:
                yield client


@contextlib.contextmanager
def _handshakeless_tls() -> Iterator[Any]:
    """Un `SSLSocket` con `_sslobj` VIVO, y sin handshake, que es lo que hace útil la fila.

    La versión anterior de este módulo declaraba `ssl.SSLSocket` NO CUBIERTO porque «montar
    un `SSLSocket` en loopback sin servidor TLS se colgó». Se colgaba por una razón
    concreta y evitable: sobre un socket BLOQUEANTE, el primer `write` intenta completar el
    handshake y se queda esperando al servidor que no existe. Con `setblocking(False)` antes
    de envolver, `wrap_socket(do_handshake_on_connect=False)` devuelve en el acto y cada
    método de salida levanta `SSLWantReadError` — medido, sin cuelgue.

    `_sslobj` VIVO es la condición que hace honesta a la fila, y por eso se asegura: con
    `_sslobj` a `None`, `SSLSocket.send` delega en `socket.socket.send` y la fila se pondría
    verde por el parche de al lado, con el suyo quitado (regla 1). Con `_sslobj` vivo los
    bytes van por `self._sslobj.write`, que no pasa por ninguna otra fila.
    """
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        plain = socket.socket()
        plain.connect(listener.getsockname())
        with listener.accept()[0]:
            plain.setblocking(False)
            wrapped = context.wrap_socket(
                plain, do_handshake_on_connect=False, server_hostname="localhost"
            )
            assert wrapped._sslobj is not None, (
                "sin `_sslobj` este control cae en `socket.socket.send` y probaría el parche "
                "de al lado en vez del suyo (regla 1)"
            )
            try:
                yield wrapped
            finally:
                wrapped.close()


def _sendfile_of_a_real_file(sock: socket.socket) -> int:
    """`sendfile` de un fichero REGULAR, y el adjetivo es el control.

    Con un `BytesIO` esta fila no probaría nada: `socket.sendfile` sólo baja a `os.sendfile`
    cuando el objeto tiene un `fileno()` de fichero regular, y con cualquier otra cosa cae en
    `_sendfile_use_send`, que ya está cortado por la fila de `send`. La fila se pondría verde
    con su propio parche quitado, que es la regla 1 exacta. Con un fichero de disco no: sin
    el parche de `sendfile`, los bytes salen por `os.sendfile` sin tocar `send`.
    """
    with open(__file__, "rb") as handle:
        return sock.sendfile(handle)


def _os_sendfile_of_a_real_file(sock: socket.socket) -> int:
    """`os.sendfile` directo: el piso de abajo del anterior, sin pasar por el objeto socket.

    Medido en este árbol sobre un TCP preconectado: 16 bytes de `/etc/hosts` entregados al
    receptor sin tocar ningún método del socket. Es la fila del descriptor desnudo para el
    caso en que los bytes vienen de otro descriptor y no de un `bytes` de Python.
    """
    with open(__file__, "rb") as handle:
        return os.sendfile(sock.fileno(), handle.fileno(), 0, 16)


def _ipv6_loopback_works() -> bool:
    """¿Hay pila IPv6 local? Se MIDE, no se supone.

    UDP `connect` sobre loopback no habla con nadie —sólo fija el destino—, así que esto
    no manda un byte: falla si y sólo si la familia no existe en esta máquina.
    """
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as probe:
            probe.connect(_blackhole(socket.AF_INET6))
    except OSError:
        return False
    return True


HAS_IPV6_LOOPBACK = _ipv6_loopback_works()


@dataclass(frozen=True)
class Egress:
    """Una salida a la red: dónde se corta, CÓMO se corta, y la llamada EXACTA que lo prueba.

    El parche y su control salen de la MISMA fila a propósito. Una lista de cosas parcheadas
    y otra de cosas comprobadas son dos definiciones de «qué es salir a la red» (regla 5), y
    el día que divergen el cerrojo se queda con un agujero y su control verde al lado — que
    es exactamente cómo `sendto` sobrevivió a la primera versión de este módulo, `send` a la
    segunda, y `os.write` + `_socket.socket` + el enlace temprano a la tercera. Aquí no se
    puede añadir un parche sin añadir la llamada que lo ejerce, ni al revés.

    `setup` se abre ANTES de echar el cerrojo y `attempt` recibe lo que entregue. Ese orden no
    es una comodidad: es lo único que reproduce el socket preconectado, la salida que
    sobrevive a que cortes `connect`, y desde que el cerrojo prohíbe el nacimiento de un
    socket INET es también lo único que deja a cada fila ejercer SU parche y no el del
    guardián de nacimiento.
    """

    id: str
    owner: Any
    name: str
    attempt: Callable[[Any], Any]
    setup: Callable[[], ContextManager[Any]] = contextlib.nullcontext
    patch: Callable[[Callable[..., Any]], Callable[..., Any]] = _refuse_always


# --- 1. EL NACIMIENTO. Derivado por identidad, no escrito: una fila por alias del tipo real.
def _birth_attempt(owner: Any, name: str) -> Callable[[Any], Any]:
    """Construir un socket INET por ESTE alias. `getattr` en la llamada, para ver el parche."""

    def attempt(_: Any) -> Any:
        return getattr(owner, name)(socket.AF_INET, socket.SOCK_DGRAM)

    return attempt


_BIRTH_EGRESS: tuple[Egress, ...] = tuple(
    Egress(
        f"birth:{owner.__name__}.{name}",
        owner,
        name,
        _birth_attempt(owner, name),
        patch=_install_the_guarded_type,
    )
    for owner, name in _SOCKET_TYPE_ALIASES
)


# --- 2. LOS MÉTODOS del socket, sobre un objeto que nació ANTES del cerrojo.
_SOCKET_METHOD_EGRESS: tuple[Egress, ...] = (
    # CON conexión: el destino se fija antes, así que basta con cortar el `connect`. Es lo
    # que hace cualquier cliente HTTP, y por tanto cualquier `requests`/`httpx` del futuro.
    Egress(
        "connect",
        socket.socket,
        "connect",
        lambda sock: sock.connect(BLACKHOLE),
        setup=_fresh_udp,
        patch=_refuse_on_inet,
    ),
    Egress(
        "connect_ex",
        socket.socket,
        "connect_ex",
        lambda sock: sock.connect_ex(BLACKHOLE),
        setup=_fresh_udp,
        patch=_refuse_on_inet,
    ),
    # SIN conexión: el datagrama lleva su destino DENTRO de la llamada, así que no pasa por
    # `connect` y el cerrojo de arriba no lo ve. Este es el agujero que encontró la primera
    # revisión: con sólo `connect`/`getaddrinfo` parcheados, un `sendto` real dentro de
    # `_search` dejaba los dos tests del paso 27 en verde.
    Egress(
        "sendto",
        socket.socket,
        "sendto",
        lambda sock: sock.sendto(b"x", BLACKHOLE),
        setup=_fresh_udp,
        patch=_refuse_on_inet,
    ),
    Egress(
        "sendmsg",
        socket.socket,
        "sendmsg",
        lambda sock: sock.sendmsg([b"x"], [], 0, BLACKHOLE),
        setup=_fresh_udp,
        patch=_refuse_on_inet,
    ),
    # PRECONECTADAS: el agujero que encontró la SEGUNDA revisión, y el que se justificó mal.
    # `send`/`sendall` no llevan destino, y de ahí se dedujo —aquí, por escrito— que
    # cortar `connect` ya las alcanzaba. No las alcanza: el `connect` y el `send` están
    # separados EN EL TIEMPO, y un socket conectado antes de echar el cerrojo se lo lleva
    # puesto. Medido: con estas dos filas fuera, un `send` y un `sendall` dentro de `_search`
    # entregaron sus datagramas a un receptor bindeado mientras los dos tests del paso 27
    # seguían verdes.
    Egress(
        "send",
        socket.socket,
        "send",
        lambda sock: sock.send(b"x"),
        setup=lambda: _preconnected_udp(socket.AF_INET),
        patch=_refuse_on_inet,
    ),
    Egress(
        "sendall",
        socket.socket,
        "sendall",
        lambda sock: sock.sendall(b"x"),
        setup=lambda: _preconnected_udp(socket.AF_INET),
        patch=_refuse_on_inet,
    ),
    # Y las mismas dos en IPv6, porque el predicado nombra DOS familias y una sola de ellas
    # ejercida deja la otra sostenida por la lectura del código. Reducir `_INET_FAMILIES` a
    # AF_INET —el «simplificado» más natural del mundo— tiene que ponerse rojo aquí.
    Egress(
        "send_ipv6",
        socket.socket,
        "send",
        lambda sock: sock.send(b"x"),
        setup=lambda: _preconnected_udp(socket.AF_INET6),
        patch=_refuse_on_inet,
    ),
    Egress(
        "sendall_ipv6",
        socket.socket,
        "sendall",
        lambda sock: sock.sendall(b"x"),
        setup=lambda: _preconnected_udp(socket.AF_INET6),
        patch=_refuse_on_inet,
    ),
    # `sendfile` NO pasa por `send`: baja a `os.sendfile`, en el kernel, con el fichero y el
    # socket y sin intermediario en Python. Medido en este árbol sobre un TCP preconectado:
    # 17 bytes entregados al receptor y `send` sin llamarse ni una vez. Es el mismo agujero
    # que `send`, un piso más abajo, y por eso lleva su propia fila.
    Egress(
        "sendfile",
        socket.socket,
        "sendfile",
        _sendfile_of_a_real_file,
        setup=_preconnected_tcp,
        patch=_refuse_on_inet,
    ),
)


# --- 2b. LOS MÉTODOS QUE `ssl.SSLSocket` REDEFINE. Estaban declarados NO CUBIERTOS.
# `SSLSocket` los redefine (están en su `__dict__`), así que parchear `socket.socket` no los
# alcanza: una conexión TLS establecida antes del cerrojo escribiría por `self._sslobj` sin
# tocar ninguna fila de arriba. No llevaban fila porque su control «se colgaba» — y se
# colgaba por ser un socket bloqueante, no por ser imposible (`_handshakeless_tls`).
# `send`/`sendall`/`write` son los que entregan datos, y `do_handshake`/`unwrap` también
# escriben en el flujo (un ClientHello, un close_notify), aunque sólo al par ya establecido.
# `sendto`/`sendmsg` los rechaza `SSLSocket` por diseño y `connect`/`connect_ex` exigen un
# socket sin conectar (medido: `ValueError`/`NotImplementedError`), así que ésos cuatro no
# pueden entregar nada — y llevan fila igual, porque cuestan un envoltorio y así la frase
# «todo método de salida que `SSLSocket` redefine está parcheado» es DERIVABLE del
# `__dict__` de la clase y no una lectura de esta lista.
_TLS_METHOD_EGRESS: tuple[Egress, ...] = tuple(
    Egress(
        f"tls:{name}",
        ssl.SSLSocket,
        name,
        attempt,
        setup=_handshakeless_tls,
        patch=_refuse_on_inet,
    )
    for name, attempt in (
        ("send", lambda sock: sock.send(b"x")),
        ("sendall", lambda sock: sock.sendall(b"x")),
        ("write", lambda sock: sock.write(b"x")),
        ("sendto", lambda sock: sock.sendto(b"x", BLACKHOLE)),
        ("sendmsg", lambda sock: sock.sendmsg([b"x"])),
        ("sendfile", lambda sock: sock.sendfile(open(__file__, "rb"))),  # noqa: SIM115
        ("do_handshake", lambda sock: sock.do_handshake()),
        ("unwrap", lambda sock: sock.unwrap()),
        ("connect", lambda sock: sock.connect(BLACKHOLE)),
        ("connect_ex", lambda sock: sock.connect_ex(BLACKHOLE)),
    )
)


# --- 3. EL DESCRIPTOR DESNUDO. La fuga que este módulo declaraba NO CUBIERTA.
_FD_EGRESS: tuple[Egress, ...] = (
    Egress(
        "os.write",
        os,
        "write",
        lambda sock: os.write(sock.fileno(), b"x"),
        setup=lambda: _preconnected_udp(socket.AF_INET),
        patch=_refuse_on_inet_fd,
    ),
    Egress(
        "os.writev",
        os,
        "writev",
        lambda sock: os.writev(sock.fileno(), [b"x"]),
        setup=lambda: _preconnected_udp(socket.AF_INET),
        patch=_refuse_on_inet_fd,
    ),
    Egress(
        "os.sendfile",
        os,
        "sendfile",
        _os_sendfile_of_a_real_file,
        setup=_preconnected_tcp,
        patch=_refuse_on_inet_fd,
    ),
)


# --- 3b. REGALAR EL DESCRIPTOR. La séptima salida, y la encontró la auditoría de esta ronda.
# Cortar `os.write` cierra la escritura por descriptor DESDE PYTHON, no el descriptor. Medido:
# `io.FileIO(sock.fileno(), "wb").write(b"s")` entregó el byte, y también `os.dup` + `FileIO`,
# `os.dup2` + `FileIO`, `_socket.dup` + `FileIO` y `os.fdopen` — porque el `write` de un objeto
# fichero es C y no pasa por `os.write`. La costura no está en la escritura: está en el momento
# en que el descriptor de un socket INET se COPIA o se ENVUELVE, que sí tiene nombre en Python.
# Los nombres se buscan otra vez por IDENTIDAD, porque otra vez son los mismos objetos con
# nombres distintos: `io.FileIO is _io.FileIO`, y `io.open is _io.open is builtins.open`.
_REAL_FILEIO = io.FileIO
_REAL_OPEN = io.open


def _aliases_of(target: Any, owners: tuple[Any, ...]) -> tuple[tuple[Any, str], ...]:
    """Todo atributo de estos módulos que ES este objeto. Por identidad, no por nombre."""
    return tuple(
        (owner, name)
        for owner in owners
        for name in sorted(dir(owner))
        if not name.startswith("__") and getattr(owner, name, None) is target
    )


def _wrap_attempt(owner: Any, name: str) -> Callable[[Any], Any]:
    """Envolver el descriptor del socket en un objeto fichero, POR ESTE nombre."""

    def attempt(sock: socket.socket) -> Any:
        return getattr(owner, name)(sock.fileno(), "wb", closefd=False)

    return attempt


def _dup2_onto_devnull(sock: socket.socket) -> Any:
    """`dup2` del socket sobre un descriptor ya abierto, que es la forma que engaña al ojo.

    Un `os.write(1, …)` parece inofensivo, y lo es — hasta que alguien ha puesto el socket
    DETRÁS del 1. `os.open` no se parchea (abrir un fichero no es salir a la red), así que el
    hueco para el destino se abre aquí y se cierra en el `finally` pase lo que pase: si el
    parche no mordiese, el `dup2` habría dejado el socket colgando de ese número.
    """
    placeholder = os.open(os.devnull, os.O_WRONLY)
    try:
        return os.dup2(sock.fileno(), placeholder)
    finally:
        os.close(placeholder)


_DESCRIPTOR_HANDOUT_EGRESS: tuple[Egress, ...] = (
    Egress(
        "os.dup",
        os,
        "dup",
        lambda sock: os.dup(sock.fileno()),
        setup=lambda: _preconnected_udp(socket.AF_INET),
        patch=_refuse_on_inet_source_fd,
    ),
    Egress(
        "os.dup2",
        os,
        "dup2",
        _dup2_onto_devnull,
        setup=lambda: _preconnected_udp(socket.AF_INET),
        patch=_refuse_on_inet_source_fd,
    ),
    # `socket.dup` Y `_socket.dup` son el MISMO objeto por dos atributos, igual que las
    # entradas de resolución, así que hacen falta las dos filas. Y la tercera es el método:
    # `socket.socket.dup` resuelve en su cuerpo el global `dup` que `from _socket import *`
    # dejó enlazado temprano (leído en `socket.py`), así que lo alcanza el parche de
    # `socket.dup` y NO el de `_socket.dup` — motivo por el que el método no lleva fila propia
    # y sí lleva control propio.
    *(
        Egress(
            f"{owner.__name__}.dup",
            owner,
            "dup",
            _resolution_attempt_on_fd(owner, "dup"),
            setup=lambda: _preconnected_udp(socket.AF_INET),
            patch=_refuse_on_inet_source_fd,
        )
        for owner in (socket, _socket)
    ),
    Egress(
        "socket.socket.dup (por el global de socket.py)",
        socket,
        "dup",
        lambda sock: sock.dup(),
        setup=lambda: _preconnected_udp(socket.AF_INET),
        patch=_refuse_on_inet_source_fd,
    ),
    *(
        Egress(
            f"wrap:{owner.__name__}.{name}",
            owner,
            name,
            _wrap_attempt(owner, name),
            setup=lambda: _preconnected_udp(socket.AF_INET),
            patch=_refuse_wrapping_an_inet_fd,
        )
        for owner, name in (
            *_aliases_of(_REAL_FILEIO, (io, _io)),
            *_aliases_of(_REAL_OPEN, (io, _io, builtins)),
        )
    ),
    # `os.fdopen` no lleva parche propio: su cuerpo es `return io.open(fd, …)` por atributo
    # (leído en `os.py:1069`), así que cae con la fila de `io.open`. Lleva control igual,
    # porque «cae con» es una lectura hasta que se ejecuta.
    Egress(
        "os.fdopen (por io.open)",
        io,
        "open",
        lambda sock: os.fdopen(sock.fileno(), "wb", closefd=False),
        setup=lambda: _preconnected_udp(socket.AF_INET),
        patch=_refuse_wrapping_an_inet_fd,
    ),
)


# --- 4. LA RESOLUCIÓN, cortada en el PISO DE ABAJO (`_socket`) para alcanzar el enlace
# temprano. Un nombre resuelto es una consulta que YA salió, aunque después no se conecte
# nadie, y `gethostbyname` no pasa por `getaddrinfo`: es otra entrada de la libc.
def _early_bound(owner: Any, name: str) -> Callable[[Any], Any]:
    """Captura la referencia AHORA —al importar, antes del cerrojo— y la llama DESPUÉS.

    Es la fuga por enlace temprano, ejercida: `from socket import getaddrinfo` guarda el
    objeto función, así que mover el atributo del módulo después no alcanza a quien ya lo
    tiene. La fila se pone verde sólo si el corte está en el piso al que ese objeto baja en
    CADA llamada — `_socket.getaddrinfo` (`socket.py:977`) —, y se pone roja si el corte se
    queda en `socket.getaddrinfo`.
    """
    captured = getattr(owner, name)

    def attempt(_: Any) -> Any:
        return captured("localhost", 80)

    return attempt


# Las cuatro entradas de resolución de la libc que `socket` RE-EXPORTA con `from _socket
# import *`, y los argumentos con que se ejercen. Cada una produce DOS filas, y las dos hacen
# falta: `socket.gethostbyname` y `_socket.gethostbyname` son el MISMO objeto pero DOS
# atributos distintos (medido: `same=True`), así que mover uno no mueve el otro. Un parche
# puesto sólo abajo deja `socket.gethostbyname` intacto — y es lo que puso rojas estas cuatro
# filas la primera vez que se escribió esta sección. `getaddrinfo` NO está aquí porque es el
# único con envoltorio: se trata aparte, abajo.
_REEXPORTED_RESOLUTION: tuple[tuple[str, tuple[Any, ...]], ...] = (
    ("gethostbyname", ("example.invalid",)),
    ("gethostbyname_ex", ("example.invalid",)),
    ("gethostbyaddr", ("127.0.0.1",)),
    ("getnameinfo", (("127.0.0.1", 80), 0)),
)


def _resolution_attempt(owner: Any, name: str, args: tuple[Any, ...]) -> Callable[[Any], Any]:
    """Resolver POR ESTE módulo. El `getattr` va en la llamada, para que vea el parche."""

    def attempt(_: Any) -> Any:
        return getattr(owner, name)(*args)

    return attempt


_RESOLUTION_EGRESS: tuple[Egress, ...] = (
    # `getaddrinfo` es el caso con suerte, y por eso su corte va SÓLO abajo:
    # `socket.getaddrinfo` es un ENVOLTORIO Python que llama a `_socket.getaddrinfo` por
    # atributo en cada llamada (`socket.py:977`), así que cortar abajo alcanza incluso a quien
    # capturó el envoltorio antes de que el cerrojo existiera.
    Egress("getaddrinfo", _socket, "getaddrinfo", lambda _: socket.getaddrinfo("localhost", 80)),
    # La MISMA entrada por una referencia capturada ANTES del cerrojo: la fuga que reportó la
    # revisión, ejercida. Dos filas y no una porque son dos caminos al mismo corte, y sólo la
    # segunda se pone roja si alguien «simplifica» el parche subiéndolo a `socket.getaddrinfo`.
    Egress("getaddrinfo_early", _socket, "getaddrinfo", _early_bound(socket, "getaddrinfo")),
    *(
        Egress(f"{owner.__name__}.{name}", owner, name, _resolution_attempt(owner, name, args))
        for name, args in _REEXPORTED_RESOLUTION
        for owner in (socket, _socket)
    ),
    # `create_connection` es Python puro en `socket.py` y NO tiene piso de abajo, así que se
    # corta donde vive. Un enlace temprano suyo se escaparía — y es lo que prohíbe
    # `test_no_module_under_test_early_binds_a_network_primitive`.
    Egress(
        "create_connection",
        socket,
        "create_connection",
        lambda _: socket.create_connection(("example.invalid", 80)),
    ),
)


_EGRESS: tuple[Egress, ...] = (
    Egress("getaddrinfo", _socket, "getaddrinfo", lambda _: socket.getaddrinfo("localhost", 80)),
    # La MISMA entrada, llamada por una referencia capturada antes del cerrojo. Dos filas y
    # no una porque son dos caminos distintos al mismo corte, y sólo la segunda se pone roja
    # si alguien «simplifica» el parche subiéndolo a `socket.getaddrinfo`.
    Egress("getaddrinfo_early", _socket, "getaddrinfo", _early_bound(socket, "getaddrinfo")),
    Egress(
        "gethostbyname",
        _socket,
        "gethostbyname",
        lambda _: socket.gethostbyname("example.invalid"),
    ),
    Egress(
        "gethostbyname_ex",
        _socket,
        "gethostbyname_ex",
        lambda _: socket.gethostbyname_ex("example.invalid"),
    ),
    Egress("gethostbyaddr", _socket, "gethostbyaddr", lambda _: socket.gethostbyaddr("127.0.0.1")),
    Egress(
        "getnameinfo", _socket, "getnameinfo", lambda _: socket.getnameinfo(("127.0.0.1", 80), 0)
    ),
    # `create_connection` es Python puro en `socket.py` y NO tiene piso de abajo, así que se
    # corta donde vive. Un enlace temprano suyo se escaparía — y es lo que prohíbe
    # `test_no_module_under_test_early_binds_a_network_primitive`.
    Egress(
        "create_connection",
        socket,
        "create_connection",
        lambda _: socket.create_connection(("example.invalid", 80)),
    ),
)


_EGRESS: tuple[Egress, ...] = (
    *_BIRTH_EGRESS,
    *_SOCKET_METHOD_EGRESS,
    *_TLS_METHOD_EGRESS,
    *_FD_EGRESS,
    *_DESCRIPTOR_HANDOUT_EGRESS,
    *_RESOLUTION_EGRESS,
)

# Los métodos que el cerrojo parchea sobre un TIPO, derivados de las filas y no escritos otra
# vez: es la lista contra la que el barrido de sockets vivos comprueba cada objeto.
_EGRESS_METHOD_NAMES: frozenset[str] = frozenset(
    egress.name for egress in _EGRESS if isinstance(egress.owner, type)
)


def _egress_outside_the_lock() -> list[str]:
    """Todo lo VIVO que puede sacar un byte a la red sin pasar por el cerrojo.

    Es el arma contra la parte que NO se puede parchear, y son DOS cosas, las dos medidas:

    · UN SOCKET DE UN TIPO QUE NO SE PUEDE PARCHEAR. `_socket.socket` es un tipo inmutable
      —`TypeError: cannot set 'send' attribute of immutable type`—, así que un objeto de ese
      tipo exacto, o de una subclase de terceros que redefina un método de salida, entregaría
      bytes sin tocar ninguna fila. Lo que sí se puede hacer es DEMOSTRAR que no existe
      ninguno.

    · UNA REFERENCIA A UN MÉTODO CAPTURADA ANTES DEL CERROJO. Medido: con
      `bound = sock.send` guardado antes de parchear, `bound(b"z")` entregó el byte al
      receptor con `socket.socket.send` ya sustituido. Un método ligado resuelve su función
      al CAPTURARSE, no al llamarse, así que ningún parache posterior lo alcanza — es la
      misma clase que el enlace temprano de `getaddrinfo`, un piso más abajo. Y sólo se puede
      cerrar aquí porque `builtin_function_or_method` está RASTREADO por el gc (medido:
      `gc.is_tracked(bound) is True`), así que la referencia aparece en `gc.get_objects()`
      esté donde esté guardada — un global, una variable local, un cierre, el atributo de un
      objeto. Un escaneo de globales no habría llegado a las tres últimas.

    La pregunta se hace por la MARCA `_LOCK_TAG` que llevan los envoltorios, no por el módulo
    donde están definidos: una comprobación por `__module__` adivinaría. Y sólo se mira un
    nombre que EXISTA en la clase — `write` no está en `socket.socket` y `sendfile` sí, así
    que exigirlos todos a todas las clases marcaría como suelto a un socket perfectamente
    cerrado.

    Se recorre `type(obj).__mro__` y no `isinstance`: un objeto puede mentir sobre su
    `__class__` (cualquier `Mock` lo hace) y el MRO del tipo real no.

    Y NADA de `getattr` sobre un objeto arbitrario, por la misma razón un paso más allá:
    recorrer los objetos vivos significa tocar los de todo el proceso, y un `getattr` sobre
    uno de ellos EJECUTA código ajeno. La primera versión de este barrido preguntaba
    `getattr(obj, "__self__", None)` a todo lo vivo y murió con un
    `ModuleNotFoundError: No module named '_gdbm'`: el proxy de módulo perezoso de `six`
    responde a cualquier atributo importando el módulo que representa. Por eso el filtro es
    `type(obj) in _BOUND_METHOD_TYPES` —identidad de tipo, cero búsquedas de atributo— y los
    atributos se leen sólo después, cuando ya se sabe qué es el objeto.
    """
    offenders: set[str] = set()
    for obj in gc.get_objects():
        cls = type(obj)
        if _REAL_SOCKET_TYPE in getattr(cls, "__mro__", ()):
            if not _can_still_reach_the_network(obj):
                continue
            loose = sorted(
                name
                for name in _EGRESS_METHOD_NAMES
                if (attr := getattr(cls, name, None)) is not None
                and not getattr(attr, _LOCK_TAG, False)
            )
            if loose:
                offenders.add(f"socket {cls.__module__}.{cls.__qualname__} → {loose}")
            continue
        if type(obj) not in _BOUND_METHOD_TYPES:
            continue
        bound_to = obj.__self__
        if _REAL_SOCKET_TYPE not in getattr(type(bound_to), "__mro__", ()):
            continue
        name = obj.__name__
        if name not in _EGRESS_METHOD_NAMES or not _can_still_reach_the_network(bound_to):
            continue
        if not getattr(getattr(obj, "__func__", obj), _LOCK_TAG, False):
            offenders.add(f"método precapturado {type(bound_to).__qualname__}.{name}")
    return sorted(offenders)


def _can_still_reach_the_network(sock: Any) -> bool:
    """¿Este socket puede sacar un byte AHORA? Familia de red y descriptor todavía abierto.

    Las DOS condiciones, y la segunda no es una comodidad para que el barrido calle: un
    socket cerrado no tiene descriptor, y medido, `send`/`sendto` y hasta un método ligado
    precapturado levantan `OSError 9, Bad file descriptor`. Su `family` sigue diciendo
    `AF_INET` —el dato se queda en el objeto—, así que sin mirar el `fileno` el barrido
    marcaría como fuga algo que no puede entregar nada, y bastaría con que un test anterior
    dejase un socket cerrado en un traceback para que el cerrojo no se pudiera echar nunca
    más en esa sesión. Es exactamente lo que pasó al escribir esto.
    """
    try:
        return sock.family in _INET_FAMILIES and sock.fileno() >= 0
    except (OSError, AttributeError):  # pragma: no cover - socket a medio construir
        return False


def block_network(monkeypatch) -> None:
    """Cierra la red: nacer, conectar, resolver o MANDAR — por el objeto o por el descriptor.

    Se corta en las filas de `_EGRESS`, cada una con el rechazo que le toca, y después se
    COMPRUEBA que no quede vivo nada capaz de sacar un byte por su cuenta. Los cinco pisos y
    el por qué de cada uno están en el comentario de cabecera de esta sección. Aquí va la
    lista de lo que queda DENTRO y lo que queda FUERA, que es la parte que ya se ha escrito
    mal tres veces: nada de lo que sigue es un argumento sin ejecutar.

    LO QUE QUEDA CUBIERTO, y cada punto tiene su fila con la llamada que lo ejerce:

    1. EL NACIMIENTO de un socket INET, por TODOS los alias del tipo real — buscados por
       identidad, no escritos: `_socket.socket`, `_socket.SocketType` y `socket.SocketType`
       son el MISMO objeto (medido), y con ellos caen `socket.socket()`, `socket.fromfd`,
       `socket.create_server` y `socket.has_dualstack_ipv6`, que construyen por ahí.
    2. LOS MÉTODOS de `socket.socket`: `connect`, `connect_ex`, `sendto`, `sendmsg`, `send`,
       `sendall` (en IPv4 y en IPv6) y `sendfile`.
    3. LOS MÉTODOS QUE `ssl.SSLSocket` REDEFINE y que por tanto no alcanzaba el punto
       anterior: `send`, `sendall`, `write`, `sendto`, `sendmsg`, `sendfile`, `do_handshake`,
       `unwrap`, `connect` y `connect_ex`.
    4. EL DESCRIPTOR, en sus dos mitades y las dos con la familia decidida por el KERNEL
       (`_fd_family`) y no por un nombre: ESCRIBIR por él (`os.write`, `os.writev`,
       `os.sendfile`) y REGALARLO — copiarlo (`os.dup`, `os.dup2`, `socket.dup`,
       `_socket.dup`, y el método `socket.socket.dup`, que cae con el primero de ésos) o
       envolverlo en un objeto fichero (`io.FileIO` y `io.open` con TODOS sus alias —
       `io.FileIO is _io.FileIO`, `io.open is _io.open is builtins.open` — más `os.fdopen`,
       que cae con `io.open`). La segunda mitad es la séptima salida de la serie y salió de la
       auditoría de esta ronda: el `write` de un objeto fichero es C y no pasa por `os.write`,
       medido — `io.FileIO(sock.fileno(), "wb").write(b"s")` entregó el byte con `os.write` ya
       parcheado, igual que `os.dup`/`os.dup2`/`_socket.dup` + `FileIO` y `os.fdopen`.
    5. LA RESOLUCIÓN, cada entrada por las puertas que de verdad tiene. `getaddrinfo` tiene
       UNA que sirve para las dos, `_socket.getaddrinfo`, porque `socket.getaddrinfo` es un
       envoltorio que baja ahí en cada llamada y por tanto el corte alcanza también a una
       referencia capturada antes del cerrojo. `gethostbyname`, `gethostbyname_ex`,
       `gethostbyaddr` y `getnameinfo` tienen DOS cada una —el mismo objeto de C por dos
       atributos, `socket.*` y `_socket.*` (medido)—, así que llevan dos filas. Y
       `socket.create_connection` sólo existe arriba: es Python puro, sin piso de abajo.
    6. Y LO QUE NO SE PUEDE PARCHEAR PERO ESTÁ VIVO: `_egress_outside_the_lock` lo detecta y
       `block_network` se NIEGA a echarse — un `_socket.socket` pelado (tipo C inmutable), una
       subclase de terceros que redefina un método de salida, o un método ligado capturado
       antes del cerrojo (`bound = sock.send`, medido entregando bytes con la clase ya
       parcheada). No es un parche: es la prueba de que no hay nada que pudiera usarlo.

    Y el self-pipe AF_UNIX de asyncio sigue vivo con todo esto puesto: es lo que conduce al
    cliente MCP, y las tres tools contestan en `test_every_tool_answers_with_the_network_blocked`.

    LO QUE QUEDA FUERA, Y POR QUÉ. Los ocho primeros NO PUEDEN sacar un byte a la red, y cada
    uno con la medida que lo dice — cuando algo podía, se ha metido dentro. Los CINCO últimos
    sí podrían, y están ahí porque no hay costura de Python donde ponerse: se dicen en voz
    alta en vez de taparse, y uno de ellos está cubierto sólo a medias y lo declara.

    · Crear un socket AF_UNIX — permitido, y tiene que estarlo. `asyncio.run` monta su
      self-pipe con `socket.socketpair()`. Se filtra por FAMILIA y no por el hecho de crear:
      medido, con el guardián puesto `socket.socketpair()` y `asyncio.new_event_loop()`
      siguen funcionando.

    · `_socket.socketpair` — NO parcheado, y medido por qué no hace falta: con AF_INET y con
      AF_INET6 levanta `OSError 102, Operation not supported on socket`. No puede fabricar un
      socket de red, y es por donde pasa el self-pipe del bucle de eventos.

    · `os.pwrite` / `os.pwritev` — NO parcheados, también medido: sobre un socket
      preconectado los dos levantan `OSError 29, Illegal seek`. Escriben en un desplazamiento
      y un socket no es posicionable. No pueden sacar un byte.

    · `recv`/`recv_into`/`recvfrom`/`recvmsg` y los `read` de TLS — NO parcheados: traen
      bytes, no los sacan, y un `recv` sobre un socket ya establecido no emite una consulta
      con datos del corpus. Que lo que ENTRA pueda ser hostil es la medida 3 del §4.4 y se
      prueba en `test_mcp_prompt_injection.py`, que es otro paso.

    · `bind`/`listen`/`shutdown`/`accept` y los accesores — NO parcheados: no mueven un byte
      al otro lado, y el socket INET sobre el que se llamarían ya no puede nacer. `makefile`
      tampoco: devuelve un `SocketIO` cuyo `write` llama a `self._sock.send`, que es la fila
      del punto 2.

    · `os.open` — NO parcheado: abrir un fichero por su ruta no puede alcanzar un socket, y
      es lo que usa el control de `dup2` para tener un descriptor de destino. Lo que sí queda
      cortado es lo que se haga DESPUÉS con un descriptor de socket (puntos 4a y 4b).

    · El resto de `io` — NO enumerado, y es una acotación, no una defensa: las dos únicas
      entradas de la biblioteca que aceptan un DESCRIPTOR son `FileIO` y `open`, y las demás
      clases (`BufferedWriter`, `TextIOWrapper`…) envuelven un objeto ya construido, así que
      caen con ellas. `dir(io)` no se compara contra nada, a diferencia de `socket`/`_socket`/
      `ssl.SSLSocket`.

    · `gethostname`/`getservbyname`/`getservbyport`/`getprotobyname` — NO parcheados: ninguno
      lleva un nombre de host, así que no hay destino que resolver y ningún valor del corpus
      puede viajar en ellos. `getfqdn` sí resuelve, pero es Python puro sobre `gethostbyaddr`
      y `gethostname` y cae con la fila del primero.

    · LLEGAR AL TIPO INMUTABLE POR EL `__mro__` — NO CUBIERTO, y medido:
      `socket.socket.__mro__[1] is _socket.socket`, así que construir por ahí esquiva el
      guardián de nacimiento y los métodos del objeto resultante son los de C, que no se
      pueden sustituir. No hay costura: `__mro__` y `__bases__` no son parcheables. Lo único
      que se hace es que el barrido corra TAMBIÉN después de la consulta, así que se detecta
      si el objeto sobrevive a la llamada; si se creara, mandara y soltara, no se vería. Esto
      es la regla 13 exacta: el cerrojo caza descuidos, no ataques, y nadie llega ahí por
      descuido.

    · UNA REFERENCIA AL DESCRIPTOR NO LIGADO capturada antes del cerrojo
      (`DESC = _socket.socket.send`; `DESC(sock, b"w")`) — CUBIERTA SÓLO A MEDIAS, y medida:
      entregó el byte con la clase ya parcheada. El barrido no puede marcarla por su
      existencia, porque el descriptor vive permanentemente en el diccionario del tipo y
      estaría ahí en cualquier sesión. Lo que sí se cierra es la mitad que este PR gobierna:
      `test_no_module_under_test_early_binds_a_network_primitive` se pone rojo si un módulo de
      xbrain la guarda en un global. Guardada en una local, en un cierre o en un atributo de
      un objeto, se escapa. (El método LIGADO —`bound = sock.send`— sí está cubierto del todo,
      porque el barrido lo encuentra en `gc.get_objects()` esté donde esté guardado.)

    · Un SUBPROCESO — NO CUBIERTO, y no es cubrible desde aquí: un `curl` hijo tiene su
      propia tabla de sockets. Ninguna de las tres tools lanza uno; los comandos externos de
      xbrain (`transcribe`, `vision`, `embeddings`) viven fuera de la ruta de consulta, y el
      único que la tocaría es el embedder bajo `--strategy vector`, que estos tests no usan.

    · Una escritura hecha por CÓDIGO C sobre un descriptor de socket, sin pasar por ningún
      nombre de Python — NO CUBIERTO, y es el límite real del piso 4: lo que se corta es
      copiar o envolver el descriptor, porque eso tiene nombre; escribir en él desde C no lo
      tiene. Misma clase que el subproceso de arriba y la extensión de abajo.

    · Una extensión en C que llame a `socket(2)` o a `send(2)` por su cuenta — NO CUBIERTO, y
      de la misma clase que el subproceso: no pasa por ningún nombre de Python, así que no hay
      dónde ponerse. Lo que sí queda cubierto es cualquier extensión que construya su socket
      por la API de Python, porque el nacimiento está guardado.

    Todo lo demás que alcanza la red desde Python pasa por una fila de `_EGRESS`, y los
    cuatro tests de totalidad del final se ponen ROJOS si aparece un nombre nuevo sin
    clasificar en `socket`, `_socket` o `ssl.SSLSocket`.
    """
    for egress in _EGRESS:
        monkeypatch.setattr(
            egress.owner, egress.name, egress.patch(getattr(egress.owner, egress.name))
        )
    outside = _egress_outside_the_lock()
    if outside:
        raise NetworkLockIncomplete(
            "queda vivo algo capaz de sacar un byte a la red sin pasar por el cerrojo, así "
            f"que «sin red» no se puede prometer: {outside}"
        )


@pytest.mark.parametrize("egress", _EGRESS, ids=lambda egress: egress.id)
def test_the_network_block_actually_bites(egress: Egress, monkeypatch) -> None:
    """El control del paso 27, UNA FILA POR SALIDA: sin esto, «las tools funcionaron» no
    prueba nada.

    Un bloqueo mal puesto no rompe ningún test — deja pasar las tres herramientas
    exactamente igual que uno bien puesto. La única forma de que el verde de abajo signifique
    algo es demostrar aquí que el cerrojo está echado (regla 2).

    Se comprueba fila a fila, y no con dos llamadas de muestra, porque una muestra sólo
    habla de lo que muestrea: el cerrojo cortaba `create_connection` y `getaddrinfo`, su
    control probaba esas dos, y `sendto` pasaba por al lado con los dos tests en verde.
    `NoNetworkAllowed` es una clase de este módulo, así que sólo puede levantarla el parche:
    ninguna fila puede ponerse verde por un fallo del sistema.

    El `setup()` de la fila corre ANTES de `block_network`, y ése es el orden que hace útil
    a este test por DOS razones. Una, las filas preconectadas: un socket que pasó por
    `connect` antes de que el cerrojo existiera se lleva el destino puesto, y un control que
    conectase después chocaría con el `connect` parcheado. Y dos, desde que el cerrojo
    prohíbe el NACIMIENTO de un socket INET, crear el socket dentro de `attempt` pondría
    verde cualquier fila de método por el guardián de nacimiento, con su propio parche
    quitado — el verde por el motivo equivocado de la regla 1.
    """
    with egress.setup() as handle:
        block_network(monkeypatch)
        with pytest.raises(NoNetworkAllowed):
            egress.attempt(handle)


def test_the_lock_refuses_to_install_when_it_cannot_reach_a_live_socket(monkeypatch) -> None:
    """El control del barrido: si no detectase nada, `block_network` mentiría en silencio.

    Se fabrica exactamente lo que el cerrojo NO puede parchear —una instancia del tipo C
    inmutable `_socket.socket`, cuyos métodos no se pueden sustituir— y se exige que echar el
    cerrojo FALLE. Sin el barrido, `block_network` se instalaría tan tranquilo y ese socket
    entregaría sus bytes: es la fuga que reportó la revisión, y el barrido es lo que hace que
    su ausencia falle CERRADO en vez de abierto.

    `NetworkLockIncomplete` y no `NoNetworkAllowed` porque son dos hechos distintos: uno dice
    «alguien intentó salir», el otro «no puedo prometer que nadie pueda».
    """
    bare = _REAL_SOCKET_TYPE(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(NetworkLockIncomplete, match=r"socket _socket\.socket"):
            block_network(monkeypatch)
    finally:
        bare.close()


def test_the_lock_refuses_to_install_with_a_method_captured_before_it(monkeypatch) -> None:
    """La otra mitad del barrido, y la salida que encontró la auditoría propia de esta ronda.

    Un método ligado resuelve su función al CAPTURARSE, no al llamarse. Medido antes de
    escribir esto: con `bound = sock.send` guardado antes de parchear, `bound(b"z")` entregó
    el byte al receptor con `socket.socket.send` ya sustituido — la misma clase que el enlace
    temprano de `getaddrinfo`, un piso más abajo y fuera del alcance de cualquier parche.

    No se puede interceptar, así que se exige que no exista: si la referencia está viva
    cuando se echa el cerrojo, el cerrojo se niega. Se comprueba además que la referencia
    sigue siendo la de verdad y no una ya parcheada, porque un `sock.send` capturado DESPUÉS
    del cerrojo sí lleva la marca y no debe disparar nada — eso lo prueba el test de abajo,
    que echa el cerrojo con sockets vivos y no protesta.
    """
    with _preconnected_udp(socket.AF_INET) as sock:
        captured = sock.send
        assert not getattr(captured, _LOCK_TAG, False), (
            "la referencia se capturó ya parcheada: este control probaría lo contrario de lo "
            "que dice (regla 1)"
        )
        with pytest.raises(NetworkLockIncomplete, match="método precapturado"):
            block_network(monkeypatch)
        del captured


def test_the_lock_installs_cleanly_when_every_live_socket_is_reachable(monkeypatch) -> None:
    """La otra mitad del control: el barrido no puede estar siempre rojo.

    Un guardián que se queja siempre no distingue nada, y el test de arriba se pondría verde
    con un `raise` incondicional dentro de `block_network`. Aquí se echa el cerrojo con la
    sesión tal cual —con el socket AF_UNIX del bucle de eventos vivo y con un `socket.socket`
    INET de la subclase parcheada delante— y se exige que se instale sin protestar.
    """
    with _preconnected_udp(socket.AF_INET):
        block_network(monkeypatch)
        assert _egress_outside_the_lock() == []


@pytest.mark.parametrize(
    ("tool", "arguments", "evidence"),
    [
        ("xbrain.search", {"query": "retrieval"}, "results"),
        ("xbrain.get", {"item_id": "k03"}, "surfaces"),
        ("xbrain.graph_expand", {"item_id": "k03"}, "nodes"),
    ],
    ids=["search", "get", "graph_expand"],
)
def test_every_tool_answers_with_the_network_blocked(
    tool: str, arguments: dict[str, Any], evidence: str, indexed_workspace: Path, monkeypatch
) -> None:
    """Paso 27 / §4.3 · §4.4.3: las tres herramientas contestan sin tocar la red.

    Es también lo que hace cierta la medida 3 del §4.4: ninguna URL del corpus se
    dereferencia. Si alguna tool fuese a buscar el artículo enlazado, el corpus podría
    dirigir a dónde va la petición, y eso es una superficie de inyección, no una consulta.

    Se exige que la respuesta traiga algo: una respuesta vacía también «funciona» sin red.
    """
    block_network(monkeypatch)
    payload = json.loads(unwrap_mcp_content(call_mcp_tool(tool, arguments)))
    assert payload[evidence], f"{tool} no sirvió nada: el verde sería vacío"
    # El barrido otra vez, AHORA. Antes de la consulta prueba que el cerrojo se echó sobre un
    # proceso limpio; aquí prueba que la consulta no dejó vivo nada que se le escape — el
    # residuo del `__mro__` documentado en `block_network` es justo esto, y esta línea lo
    # convierte en «detectado después» siempre que el objeto sobreviva a la llamada. Si la
    # consulta lo creara, mandara y soltara, seguiría sin verse: está dicho, no tapado.
    assert _egress_outside_the_lock() == [], "la consulta dejó vivo algo fuera del cerrojo"


# --- Los dos tests de totalidad: la lista de lo que se corta se DERIVA del intérprete ------
#
# Aquí termina el regreso de «¿y quién vigila al vigilante?». No hay otro cazador: lo que hay
# es que la AUSENCIA de una clasificación falla cerrado. Un método nuevo en una versión futura
# de CPython, o un alias nuevo del tipo de socket, no está ni parcheado ni declarado inocuo, y
# el test se pone ROJO hasta que alguien lo mire.

# Métodos de `_socket.socket` que NO pueden sacar un byte a la red, con su razón. El
# razonamiento no es «no se me ocurre cómo»: es que entran bytes en vez de salir, o que no
# hay bytes en juego.
_HARMLESS_SOCKET_METHODS: frozenset[str] = frozenset(
    {
        # Traen bytes, no los sacan (la hostilidad de lo que ENTRA es el §4.4, otro paso).
        "recv",
        "recv_into",
        "recvfrom",
        "recvfrom_into",
        "recvmsg",
        "recvmsg_into",
        # Abren un puerto local o cierran el flujo: ni un byte al otro lado.
        "bind",
        "listen",
        "shutdown",
        "close",
        "detach",
        # Accesores y ajustes: no hay I/O.
        "family",
        "fileno",
        "getblocking",
        "getpeername",
        "getsockname",
        "getsockopt",
        "gettimeout",
        "proto",
        "setblocking",
        "setsockopt",
        "settimeout",
        "timeout",
        "type",
        # Sólo en la subclase `socket.socket`. `accept` es entrante y exige un socket
        # escuchando, que ya no puede nacer; `get_inheritable`/`set_inheritable` sólo leen y
        # escriben una bandera; y `makefile` devuelve un `SocketIO` cuyo `write` llama a
        # `self._sock.send`, que es una fila de arriba.
        "accept",
        "get_inheritable",
        "makefile",
        "set_inheritable",
        # `dup` SÍ saca el descriptor, y por eso lleva control propio. No lleva fila propia
        # porque el parche que lo alcanza es el de `socket.dup`: el cuerpo del método resuelve
        # en cada llamada el global `dup` que `from _socket import *` dejó enlazado temprano.
        "dup",
    }
)

# Lo mismo para `ssl.SSLSocket`, que redefine casi todo. `accept`/`verify_client_post_handshake`
# son del lado SERVIDOR y exigen un socket escuchando, que ya no puede nacer; `dup` levanta
# `NotImplementedError` en `SSLSocket`; el resto son accesores del handshake.
_HARMLESS_TLS_METHODS: frozenset[str] = frozenset(
    {
        "accept",
        "cipher",
        "compression",
        "context",
        "dup",
        "get_channel_binding",
        "get_unverified_chain",
        "get_verified_chain",
        "getpeercert",
        "pending",
        "read",
        "recv",
        "recv_into",
        "recvfrom",
        "recvfrom_into",
        "recvmsg",
        "recvmsg_into",
        "selected_alpn_protocol",
        "selected_npn_protocol",
        "session",
        "session_reused",
        "shared_ciphers",
        "shutdown",
        "verify_client_post_handshake",
        "version",
    }
)

# Callables públicos de `_socket` que no sacan una consulta a la red, con su razón.
_HARMLESS_SOCKET_MODULE: frozenset[str] = frozenset(
    {
        # Conversiones puras: ni syscall ni resolución.
        "CMSG_LEN",
        "CMSG_SPACE",
        "htonl",
        "htons",
        "ntohl",
        "ntohs",
        "inet_aton",
        "inet_ntoa",
        "inet_ntop",
        "inet_pton",
        # Interfaces y hostname LOCALES: no llevan un nombre que resolver fuera.
        "gethostname",
        "sethostname",
        "if_indextoname",
        "if_nameindex",
        "if_nametoindex",
        # Ficheros locales (`/etc/services`, `/etc/protocols`): no llevan destino.
        "getprotobyname",
        "getservbyname",
        "getservbyport",
        # Descriptores y ajustes: un fd duplicado sólo se puede escribir por `os.*` (cortado)
        # o envolviéndolo en un socket, cuyos métodos son los de la subclase parcheada.
        "close",
        "getdefaulttimeout",
        "setdefaulttimeout",
        # Medido: con AF_INET y AF_INET6 levanta `OSError 102`. No fabrica un socket de red.
        "socketpair",
        # Excepciones y alias de tipo, no funciones.
        "error",
        "gaierror",
        "herror",
        "timeout",
    }
)


# Y lo que `socket` añade POR SU CUENTA: todo callable público suyo que no sea el MISMO
# objeto que el de `_socket`. Derivado por identidad, porque es la única forma de separar
# `socket.getaddrinfo` (envoltorio propio) de `socket.gethostbyname` (el objeto de C
# re-exportado, ya cubierto por la superficie de `_socket`).
_HARMLESS_SOCKET_WRAPPERS: frozenset[str] = frozenset(
    {
        # Construyen un socket con una familia EXPLÍCITA, así que el guardián de nacimiento
        # los rechaza antes de que haya socket: no hace falta parchearlos uno a uno.
        "socket",
        "create_server",
        "fromfd",
        "has_dualstack_ipv6",
        # Medido: `_socket.socketpair` levanta `OSError 102` con AF_INET y AF_INET6, y el
        # AF_UNIX que sí fabrica es el self-pipe del bucle de eventos.
        "socketpair",
        # Python puro sobre `gethostbyaddr`/`gethostname`: cae con la fila del primero.
        "getfqdn",
        # Pasan por un método del socket, que es una fila de arriba: `send_fds` por `sendmsg`,
        # `SocketIO.write` por `send`, `recv_fds` por `recvmsg` (y ése es entrante).
        "send_fds",
        "recv_fds",
        "SocketIO",
        # Enumeraciones y re-exports de `enum`: no hay I/O.
        "AddressFamily",
        "AddressInfo",
        "IntEnum",
        "IntFlag",
        "MsgFlag",
        "SocketKind",
    }
)


def _patched_names(owner: Any) -> frozenset[str]:
    """Los nombres que `_EGRESS` parchea sobre este propietario. Derivado de las filas."""
    return frozenset(egress.name for egress in _EGRESS if egress.owner is owner)


# Los nombres de `socket` que se cortan en el PISO DE ABAJO y no aquí. La clasificación no es
# prosa: es una COMPROBACIÓN. `socket.getaddrinfo` no lleva fila propia a propósito —el corte
# va en `_socket.getaddrinfo` para alcanzar el enlace temprano—, y esta derivación ata las dos
# cosas: si alguien borra la fila de abajo, este conjunto pierde el nombre, el nombre se queda
# sin clasificar y el test de totalidad se pone ROJO. Dos listas que «deberían» coincidir son
# la regla 5; una derivada de la otra no puede divergir.
_CUT_ONE_FLOOR_BELOW: frozenset[str] = _patched_names(_socket)


@pytest.mark.parametrize(
    ("label", "owner", "surface", "harmless"),
    [
        (
            # `dir(socket.socket)` y no `dir(_socket.socket)`: la superficie que interesa es la
            # de la clase que el cerrojo PARCHEA, y la subclase añade seis nombres propios
            # (entre ellos `sendfile`, que sí saca bytes y lleva fila).
            "socket.socket",
            socket.socket,
            frozenset(name for name in dir(socket.socket) if not name.startswith("_")),
            _HARMLESS_SOCKET_METHODS,
        ),
        (
            "ssl.SSLSocket",
            ssl.SSLSocket,
            frozenset(name for name in vars(ssl.SSLSocket) if not name.startswith("_")),
            _HARMLESS_TLS_METHODS,
        ),
        (
            "socket (lo propio)",
            socket,
            frozenset(
                name
                for name in dir(socket)
                if not name.startswith("_")
                and callable(getattr(socket, name))
                and getattr(socket, name) is not getattr(_socket, name, None)
            ),
            _HARMLESS_SOCKET_WRAPPERS | _CUT_ONE_FLOOR_BELOW,
        ),
        (
            "_socket",
            _socket,
            frozenset(
                name
                for name in dir(_socket)
                if not name.startswith("_") and callable(getattr(_socket, name))
            ),
            _HARMLESS_SOCKET_MODULE,
        ),
    ],
    ids=["socket_methods", "tls_methods", "socket_wrappers", "socket_module"],
)
def test_every_name_on_the_network_surface_is_patched_or_classified(
    label: str, owner: Any, surface: frozenset[str], harmless: frozenset[str]
) -> None:
    """La lista del cerrojo se COMPARA contra la del intérprete, nombre a nombre.

    Es la respuesta estructural al patrón de cuatro rondas. Parchear por enumeración pierde
    porque siempre queda una entrada más; lo que no pierde es derivar la superficie de
    `dir()` y exigir que cada nombre esté en uno de dos sitios: parcheado, o declarado
    inocuo con su razón escrita arriba. Un método nuevo de CPython no está en ninguno y esto
    se pone rojo — nadie tiene que acordarse de venir a mirar.

    Se comprueba en las DOS direcciones. Que no falte nada por clasificar, y que no sobre:
    un nombre declarado inocuo que ya no existe en esta versión de Python es una razón que
    dejó de aplicarse, y dejarla ahí es cómo una lista escrita a mano empieza a mentir.
    """
    patched = _patched_names(owner)
    assert surface, f"la superficie de {label} salió vacía: la comprobación sería hueca"
    unclassified = sorted(surface - patched - harmless)
    assert not unclassified, (
        f"{label} expone nombres que el cerrojo no parchea ni declara inocuos: "
        f"{unclassified}. Mételos en una fila de `_EGRESS` o clasifícalos con su razón."
    )
    stale = sorted((patched | harmless) - surface - _CUT_ONE_FLOOR_BELOW)
    assert not stale, f"{label} ya no expone estos nombres clasificados: {stale}"


def test_the_totality_check_would_notice_an_unclassified_name() -> None:
    """El control del test de arriba: que sepa ponerse rojo.

    Un test de totalidad que no pueda fallar es una lista de nombres con forma de aserción.
    Aquí se le mete un nombre inventado en la superficie y se exige que lo señale.
    """
    surface = frozenset(name for name in dir(socket.socket) if not name.startswith("_"))
    with pytest.raises(AssertionError, match="send_by_carrier_pigeon"):
        test_every_name_on_the_network_surface_is_patched_or_classified(
            "socket.socket",
            socket.socket,
            surface | {"send_by_carrier_pigeon"},
            _HARMLESS_SOCKET_METHODS,
        )


def _owner_label(owner: Any) -> str:
    """Cómo se nombra el propietario de una fila: un módulo por su nombre, un tipo por el suyo."""
    return owner.__name__ if not isinstance(owner, type) else owner.__qualname__


# TODA primitiva que el cerrojo mueve, con la referencia capturada al importar este fichero —
# antes de que nadie parchee nada. Las funciones de módulo (`socket.getaddrinfo`,
# `socket.create_connection`) y TAMBIÉN los descriptores no ligados de los métodos
# (`_socket.socket.send`), porque las dos formas sobreviven al parche y las dos están medidas:
# `DESC = _socket.socket.send` guardado antes y llamado como `DESC(sock, b"w")` entregó el byte
# con `socket.socket.send` ya sustituido. Se guarda el OBJETO y no sólo su `id()`, para que un
# `id` reutilizado no produzca un falso positivo, y se deduplica porque varias filas comparten
# primitiva (`send` y `send_ipv6` son el mismo método).
_EARLY_BINDABLE: tuple[tuple[Any, str], ...] = tuple(
    {
        id(primitive): (primitive, label)
        for primitive, label in (
            (
                getattr(egress.owner, egress.name),
                f"{_owner_label(egress.owner)}.{egress.name}",
            )
            for egress in _EGRESS
        )
    }.values()
)


def _early_bindings_in(module: Any) -> list[str]:
    """Globales de este módulo que SON una de las funciones de red que el cerrojo mueve."""
    return sorted(
        f"{module.__name__}.{attribute} → {label}"
        for attribute, value in vars(module).items()
        for primitive, label in _EARLY_BINDABLE
        if value is primitive
    )


def test_the_early_binding_check_finds_a_real_one() -> None:
    """El control: el predicado se estrena contra un módulo que SÍ enlaza temprano.

    `socket.py` hace `from _socket import *`, así que `socket.gethostbyname` ES el objeto de
    C de `_socket.gethostbyname` (medido: `same=True`). Si el predicado no encontrase eso, el
    test de abajo estaría contando cero por no saber contar.
    """
    found = _early_bindings_in(socket)
    assert found, "el predicado no detecta ni el enlace temprano de la propia stdlib"


def test_no_module_under_test_early_binds_a_network_primitive(indexed_workspace: Path) -> None:
    """La salida que NINGÚN parche alcanza, cerrada donde sí se puede: en el código propio.

    Una referencia al objeto función capturada antes del cerrojo no se puede interceptar. Con
    `socket.getaddrinfo` hay suerte —es un envoltorio Python que baja a `_socket.getaddrinfo`
    en cada llamada, y ahí está la fila—, pero `gethostbyname` y sus hermanas son el objeto
    de C RE-EXPORTADO: `socket.gethostbyname is _socket.gethostbyname`, así que quien lo
    guarde en un global se lo lleva puesto y no hay atributo que mover.

    Lo mismo, y medido igual, con el DESCRIPTOR no ligado de un método:
    `DESC = _socket.socket.send` guardado antes del cerrojo y llamado como `DESC(sock, b"w")`
    entregó el byte con la clase ya parcheada. Ése el barrido no lo puede marcar por existir
    —el descriptor vive siempre en el diccionario del tipo—, así que la única mitad que se
    puede cerrar es ésta.

    Como no hay costura, se prohíbe la captura en lo único que este PR gobierna: el código
    bajo prueba. Si mañana un módulo de xbrain escribe `from socket import gethostbyname` o
    guarda `_socket.socket.send` en un global, esto se pone rojo y la promesa del paso 27
    sigue siendo cierta. Una referencia guardada en una local, en un cierre o en un atributo
    de un objeto NO se escanea y está declarada como residuo en `block_network`; el método
    LIGADO sí está cubierto del todo, y lo cubre el barrido, no este test.

    Las tres tools se llaman primero A PROPÓSITO: la ruta de consulta importa sus módulos de
    forma perezosa (`from xbrain.knowledge import search_service` dentro de `_search`), así
    que escanear antes sería escanear un grafo a medio construir. Y se exige un suelo de
    módulos escaneados, porque «cero infractores sobre cero módulos» es un verde vacío.
    """
    for tool, arguments in (
        ("xbrain.search", {"query": "retrieval"}),
        ("xbrain.get", {"item_id": "k03"}),
        ("xbrain.graph_expand", {"item_id": "k03"}),
    ):
        unwrap_mcp_content(call_mcp_tool(tool, arguments))
    scanned = [
        module
        for name, module in sorted(sys.modules.items())
        if (name == "xbrain" or name.startswith("xbrain.")) and module is not None
    ]
    assert len(scanned) >= 10, f"sólo {len(scanned)} módulos de xbrain importados: verde vacío"
    offenders = [binding for module in scanned for binding in _early_bindings_in(module)]
    assert not offenders, (
        "un módulo de xbrain guarda una primitiva de red en un global, y una referencia "
        f"capturada antes del cerrojo no se puede interceptar: {offenders}"
    )


# ---------------------------------------------------------------------------
# Paso 29: `xbrain mcp-serve` sin el extra (§4.5)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def without_the_mcp_extra() -> Iterator[None]:
    """Simula que el extra `[mcp]` NO está instalado, de verdad.

    No basta con borrar `mcp` de `sys.modules`: el import volvería a encontrarlo en el disco.
    Se instala un buscador al frente de `sys.meta_path` que levanta `ModuleNotFoundError`
    para `mcp` y todo lo que cuelgue de él, que es exactamente lo que ve una máquina que
    instaló `xbrain` sin el extra.

    Los módulos ya importados se apartan y se devuelven al salir: dejarlos fuera obligaría a
    reimportarlos y el resto de la sesión acabaría con DOS copias de `mcp.types`, cuyas
    clases no son la misma y cuyos `isinstance` empiezan a fallar sin explicación.
    """
    import sys

    class Blocker:
        """Un buscador que se niega a encontrar `mcp`."""

        def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
            if name == "mcp" or name.startswith("mcp."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    hidden = {name: mod for name, mod in sys.modules.items() if name == "mcp" or name[:4] == "mcp."}
    for name in hidden:
        del sys.modules[name]
    blocker = Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        for name, mod in hidden.items():
            sys.modules[name] = mod


def test_the_blocker_actually_hides_the_extra() -> None:
    """El control del paso 29: si el bloqueo no bloquea, el test de abajo no prueba nada."""
    with without_the_mcp_extra():
        with pytest.raises(ModuleNotFoundError):
            import mcp  # noqa: F401
    import mcp  # noqa: F401  - y vuelve a estar, para el resto de la sesión


def test_mcp_serve_without_the_extra_is_actionable_not_a_traceback(workspace: Path) -> None:
    """Paso 29 / §4.5: un mensaje que dice cómo arreglarlo, no un `ImportError` crudo.

    Se comprueba sobre `result.exception`, no sobre la ausencia de la palabra «Traceback» en
    stderr: `CliRunner` ATRAPA la excepción y la guarda ahí en vez de imprimirla, así que
    «no hay traceback en stderr» se cumple igual cuando la excepción se ha escapado — la
    aserción que parece probarlo y no prueba nada (regla 1).
    """
    with without_the_mcp_extra():
        result = runner.invoke(app, ["mcp-serve"])
    assert not isinstance(result.exception, ImportError), result.exception
    assert result.exit_code == 1, result.output
    assert "xbrain[mcp]" in result.stderr, result.stderr


def test_mcp_serve_starts_the_stdio_server(workspace: Path, monkeypatch) -> None:
    """El camino feliz: el subcomando existe y llega a `serve`.

    Sin esto, el paso 29 podría estar verde con un comando que no sirve nada — el mensaje
    accionable es lo único que se habría probado.
    """
    started: list[bool] = []
    monkeypatch.setattr("xbrain.mcp_server.serve", lambda: started.append(True))
    result = runner.invoke(app, ["mcp-serve"])
    assert result.exit_code == 0, result.output
    assert started == [True]


# ---------------------------------------------------------------------------
# §4.3: los esquemas se DERIVAN de los modelos del Plan 01, no se teclean
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "model_name"),
    [
        ("xbrain.search", "SearchResponse"),
        ("xbrain.get", "EvidenceBundle"),
        ("xbrain.graph_expand", "GraphExpansionResponse"),
    ],
)
def test_the_output_schema_is_the_plan01_model_itself(tool: str, model_name: str) -> None:
    """§4.3: el esquema de salida ES el del modelo del contrato, no una copia suya.

    Los dos lados vienen de sitios distintos: el izquierdo lo derivó el SDK del tipo de
    retorno del handler, el derecho lo genera Pydantic del modelo del Plan 01. Un esquema
    escrito a mano, o un handler que devolviera un dict «equivalente», rompen la igualdad.
    """
    from xbrain.knowledge import contracts

    served = {tool.name: tool for tool in asyncio.run(build_server().list_tools())}
    model = getattr(contracts, model_name)
    assert served[tool].output_schema == model.model_json_schema()


def test_the_search_input_embeds_the_whole_filter_contract() -> None:
    """§4.3: los OCHO filtros llegan al esquema por el modelo, no re-tecleados.

    `SearchFilters` entra entera como `$ref`, así que un noveno filtro añadido al contrato
    aparece en la herramienta sin que nadie toque este módulo — y, si alguien los copiase a
    mano, este test diría en cuál se quedó corta la copia.
    """
    from xbrain.knowledge.contracts import SearchFilters

    served = {tool.name: tool for tool in asyncio.run(build_server().list_tools())}
    schema = served["xbrain.search"].input_schema
    assert set(schema["properties"]) == {"query", "filters", "limit", "strategy", "cursor"}
    assert schema["required"] == ["query"]
    embedded = schema["$defs"]["SearchFilters"]["properties"]
    assert set(embedded) == set(SearchFilters.model_fields)


# Cada handler con su servicio, y los parámetros que el handler nombra y el servicio NO.
#
# `search` y `get` no tienen ninguno: sus firmas son la del servicio menos `context`, que lo
# pone el adaptador. `graph_expand` tiene dos, y las dos con razón escrita:
#   · `item_id` — el servicio toma `seeds`; el handler toma un id y arma `item:<id>`,
#     exactamente como `xbrain graph-expand --item`;
#   · `max_neighbors` — se llama así en el CLI, y su `None` significa «el valor por defecto
#     del grafo», igual que allí.
# Lo que NO aparece en ninguna fila es `limits`: el handler de `get` no lo expone porque el
# presupuesto sale de `[index].get_char_budget`, como en el CLI.
_DEFAULT_BINDINGS = (
    ("_search", "search_service", "search", set()),
    ("_get", "get_service", "get", set()),
    ("_graph_expand", "graph_service", "graph_expand", {"item_id", "max_neighbors"}),
)


@pytest.mark.parametrize(
    ("handler_name", "module_name", "service_name", "extra"),
    _DEFAULT_BINDINGS,
    ids=[row[0] for row in _DEFAULT_BINDINGS],
)
def test_the_tool_defaults_are_the_service_defaults(
    handler_name: str, module_name: str, service_name: str, extra: set[str]
) -> None:
    """§4.1: «ni un límite distinto». Cada default compartido sale del servicio.

    Este test mira la FIRMA, no una respuesta, y por eso no depende de cuántos resultados
    tenga el corpus.

    La equivalencia CLI↔MCP sí depende, y hoy da la casualidad de que lo cubre: un
    `limit=5` en el handler donde el servicio trae 10 también la pone roja —medido: UN
    fallo, en `filter-author-and-dates`—. Pero eso es una propiedad de la FIXTURE, no del
    test. Medidos los diez casos de `search` de su tabla por la puerta del CLI:
    `filter-author-and-dates` sirve 10 items y es el ÚNICO por encima de cinco; los otros
    siete con el límite por defecto sirven entre 1 y 3, y los dos restantes pasan un
    `--limit 1` explícito, con lo que el default ni les llega. Es decir: la equivalencia
    caza el recorte por UN caso de diez. El día que ese caso devuelva menos —un ítem que
    cambia de autor, una fecha que se sale del rango— las dos puertas darán el MISMO
    documento con defaults distintos y el recorte no aparecerá en ninguna parte. Un test
    cuya cobertura depende de cuántas filas trae la fixture no es la defensa de un
    contrato; éste lo es.
    """
    import inspect

    from xbrain import mcp_server
    from xbrain.knowledge import get_service, graph_service, search_service

    modules = {
        "search_service": search_service,
        "get_service": get_service,
        "graph_service": graph_service,
    }
    handler = inspect.signature(getattr(mcp_server, handler_name)).parameters
    service = inspect.signature(getattr(modules[module_name], service_name)).parameters

    shared = set(handler) & set(service)
    assert shared, "ningún parámetro en común: la comparación sería vacía"
    for name in sorted(shared):
        assert handler[name].default == service[name].default, name
    assert set(handler) - set(service) == extra
