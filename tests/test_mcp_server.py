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

import asyncio
import contextlib
import json
import socket
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


class NoNetworkAllowed(AssertionError):
    """Alguien intentó salir a la red durante una consulta."""


# El destino de todos los intentos: el puerto DISCARD (9) en loopback. Loopback para que un
# control mal echado no mande nada fuera de la máquina, y el puerto 9 porque nadie escucha
# ahí — un intento que atraviese el cerrojo se pierde, no se entrega.
BLACKHOLE = ("127.0.0.1", 9)


def _udp() -> socket.socket:
    """Un socket UDP SIN conectar: el que puede mandar sin haber pasado por `connect`."""
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


@dataclass(frozen=True)
class Egress:
    """Una salida a la red: dónde se corta, y la llamada EXACTA que lo demuestra.

    El parche y su control salen de la MISMA fila a propósito. Una lista de cosas parcheadas
    y otra de cosas comprobadas son dos definiciones de «qué es salir a la red» (regla 5), y
    el día que divergen el cerrojo se queda con un agujero y su control verde al lado — que
    es exactamente cómo `sendto` sobrevivió a la primera versión de este módulo. Aquí no se
    puede añadir un parche sin añadir la llamada que lo ejerce, ni al revés.
    """

    id: str
    owner: Any
    name: str
    attempt: Callable[[], Any]


_EGRESS: tuple[Egress, ...] = (
    # CON conexión: el destino se fija antes, así que basta con cortar el `connect`. Es lo
    # que hace cualquier cliente HTTP, y por tanto cualquier `requests`/`httpx` del futuro.
    Egress("connect", socket.socket, "connect", lambda: _udp().connect(BLACKHOLE)),
    Egress("connect_ex", socket.socket, "connect_ex", lambda: _udp().connect_ex(BLACKHOLE)),
    Egress(
        "create_connection",
        socket,
        "create_connection",
        lambda: socket.create_connection(("example.invalid", 80)),
    ),
    # RESOLUCIÓN: un nombre resuelto es una consulta que YA salió, aunque después no se
    # conecte nadie. `gethostbyname` no pasa por `getaddrinfo`: es otra entrada de la libc.
    Egress("getaddrinfo", socket, "getaddrinfo", lambda: socket.getaddrinfo("example.invalid", 80)),
    Egress(
        "gethostbyname", socket, "gethostbyname", lambda: socket.gethostbyname("example.invalid")
    ),
    # SIN conexión: el datagrama lleva su destino DENTRO de la llamada, así que no pasa por
    # `connect` y el cerrojo de arriba no lo ve. Este es el agujero que encontró la revisión:
    # con sólo `connect`/`getaddrinfo` parcheados, un `sendto` real dentro de `_search` dejaba
    # los dos tests del paso 27 en verde.
    Egress("sendto", socket.socket, "sendto", lambda: _udp().sendto(b"x", BLACKHOLE)),
    Egress("sendmsg", socket.socket, "sendmsg", lambda: _udp().sendmsg([b"x"], [], 0, BLACKHOLE)),
)


def block_network(monkeypatch) -> None:
    """Cierra la red: conectar, resolver un nombre o MANDAR UN DATAGRAMA levanta.

    Se corta en las filas de `_EGRESS`, y hay dos cosas que deliberadamente NO están ahí:

    · `socket.socket` — crear un socket no es salir a la red, y `asyncio.run` —quien conduce
      el cliente MCP en proceso— monta su self-pipe con `socket.socketpair()`, que por dentro
      construye uno. Cortar ahí mataría el bucle de eventos y el test «pasaría» por no haber
      llegado a ejecutar nada, que es la forma más barata de fabricar un verde.

    · `send`/`sendall` — por lo mismo y por una razón más fuerte. Lo mismo: el self-pipe de
      asyncio despierta el bucle con `csock.send(b"\\0")` (comprobado en
      `asyncio.selector_events`), así que parchearlo rompe el servidor. Y la razón fuerte:
      `send` NO lleva destino, lo saca del `connect` previo — que sí está cortado—, así que
      un `send` a la red es inalcanzable con el cerrojo echado. `sendto` y `sendmsg` son las
      dos que llevan el destino en la propia llamada, y por eso son las dos que hay que
      cortar aparte.
    """

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise NoNetworkAllowed(f"salida a la red durante una consulta: {args[:2]}")

    for egress in _EGRESS:
        monkeypatch.setattr(egress.owner, egress.name, refuse)


@pytest.mark.parametrize("egress", _EGRESS, ids=lambda egress: egress.id)
def test_the_network_block_actually_bites(egress: Egress, monkeypatch) -> None:
    """El control del paso 27, UNA FILA POR SALIDA: sin esto, «las tools funcionaron» no
    prueba nada.

    Un bloqueo mal puesto no rompe ningún test — deja pasar las tres herramientas
    exactamente igual que uno bien puesto. La única forma de que el verde de abajo signifique
    algo es demostrar aquí que el cerrojo está echado (regla 2).

    Se comprueba fila a fila, y no con dos llamadas de muestra, porque una muestra sólo
    habla de lo que muestrea: el cerrojo anterior cortaba `create_connection` y
    `getaddrinfo`, su control probaba esas dos, y `sendto` pasaba por al lado con los dos
    tests en verde. `NoNetworkAllowed` es una clase de este módulo, así que sólo puede
    levantarla el parche: ninguna fila puede ponerse verde por un fallo del sistema.
    """
    block_network(monkeypatch)
    with pytest.raises(NoNetworkAllowed):
        egress.attempt()


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
