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

LA FRONTERA DE CONFIANZA (§4.3, §10.5), DECLARADA
=================================================
xbrain no llama a servicios externos. Si algún día lo hiciera, se declara a cuáles. Hoy la
declaración completa cabe en tres viñetas, y esto es lo que necesita saber quien conecte un
agente a este servidor:

  · EL SERVIDOR NO HACE LLAMADAS DE RED PROPIAS. Ningún handler abre un socket ni habla HTTP.
    `test_the_mcp_server_imports_nothing_that_speaks_to_the_network`, más abajo, lo ata sobre
    los imports del fuente del adaptador.
  · DURANTE UNA CONSULTA XBRAIN PUEDE INVOCAR UN PROCESO EXTERNO, y sólo uno: el embebedor
    configurado en `[embeddings].command`, al que se le entrega EL TEXTO DE LA CONSULTA. No
    ocurre de fábrica, y conviene no leerlo de más: `_search` sirve `strategy="lexical"` por
    defecto (`mcp_server.py`) y `[embeddings].command` nace vacío (`config.py`), así que el
    embebedor sólo entra si quien llama pide `vector`/`hybrid` Y alguien configuró el comando.
    `scripts/xbrain-embed` es el backend de REFERENCIA y corre en local; NO es un valor por
    defecto, porque no hay ninguno.
  · LO QUE ESE BINARIO HAGA CON EL TEXTO QUEDA FUERA DEL ALCANCE DE XBRAIN. Por eso esto se
    DECLARA en vez de prometerse: xbrain gobierna a quién llama, no lo que el llamado hace.
    Quien apunte `[embeddings].command` a un servicio remoto está eligiendo eso, y la
    elección es suya y visible en su `config.toml`.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from typer.testing import CliRunner

import xbrain.mcp_server
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
# Paso 27: la frontera de confianza (§4.3, §10.5)
# ---------------------------------------------------------------------------
#
# LA DECLARACIÓN ESTÁ EN EL DOCSTRING DE ESTE MÓDULO, arriba. Aquí va lo único que hace falta
# para que no se quede en prosa.
#
# Aquí vivía un CERROJO: ~1.900 líneas que parcheaban la superficie de red del intérprete
# entera —el tipo socket y sus alias, los métodos de un objeto ya nacido, el descriptor
# desnudo, la resolución en `_socket`, el enlace temprano— y luego la censaban contra
# `dir(socket.socket)` para demostrar que no faltaba ninguna. Se retira por DECISIÓN DE
# ALCANCE del owner, no porque aquellos tests fuesen incorrectos: el aparato pesaba más que
# lo probado —un adaptador de 282 líneas que delega en tres servicios— y su premisa de
# entorno se rompía al cambiar de máquina.
#
# Lo que queda es el corte barato, y muerde donde empieza el defecto: una llamada de red
# propia tiene que IMPORTAR algo que hable red, y eso se ve en el fuente sin ejecutar nada.
# Estático a propósito —`ast` sobre el fichero, cero monkeypatching— y sobre el árbol ENTERO,
# porque la ruta de consulta importa sus módulos de forma perezosa dentro de los handlers
# (`from xbrain.knowledge import search_service` dentro de `_search`) y un escaneo de sólo el
# nivel superior los perdería. El control de abajo se estrena justo contra ese caso.
#
# LO QUE ESTO NO ALCANZA, dicho y no tapado: no mira lo que importen los módulos que el
# adaptador llama, y no mira el proceso externo del embebedor. Eso segundo es precisamente la
# tercera viñeta de la declaración, y se declara porque no se puede comprobar desde aquí.


# Los paquetes que hablan red. Decide el PRIMER componente del nombre punteado, así que
# `urllib.request` cae por `urllib` y `http.client` por `http`.
_NETWORK_PACKAGES: frozenset[str] = frozenset(
    {
        "_socket",
        "aiohttp",
        "ftplib",
        "http",
        "httpx",
        "requests",
        "smtplib",
        "socket",
        "socketserver",
        "ssl",
        "urllib",
        "urllib3",
        "websockets",
        "xmlrpc",
    }
)


def _imported_modules(source: Path) -> list[str]:
    """Todo módulo que el FUENTE de `source` importa, a cualquier profundidad del árbol.

    `ast.walk` y no sólo los nodos del nivel superior: los imports de la ruta de consulta son
    perezosos y viven DENTRO de los handlers, así que leer sólo la cabecera dejaría fuera
    justo la parte que ejecuta una consulta.

    Los imports relativos (`node.level`) se saltan: nombran módulos del propio paquete, nunca
    la stdlib de red.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            imported.append(node.module)
    return imported


def _network_imports_in(source: Path) -> list[str]:
    """Los imports de `source` que hablan red, por el primer componente del nombre."""
    return sorted(
        {name for name in _imported_modules(source) if name.split(".")[0] in _NETWORK_PACKAGES}
    )


def test_the_import_check_finds_a_network_import(tmp_path: Path) -> None:
    """El control: el predicado se estrena contra un fuente que SÍ importa red.

    Y contra el caso DIFÍCIL, no contra el fácil: el `import socket` va dentro de una
    función, que es la forma que un escaneo de sólo el nivel superior no vería. Sin este
    control, el test de abajo podría estar contando cero por no saber contar.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import functools\n"
        "\n"
        "\n"
        "def handler():\n"
        "    import socket\n"
        "    from urllib.request import urlopen\n"
        "    return socket, urlopen\n",
        encoding="utf-8",
    )
    assert _network_imports_in(probe) == ["socket", "urllib.request"]


def test_the_mcp_server_imports_nothing_that_speaks_to_the_network() -> None:
    """La primera viñeta de la frontera de confianza, atada al fuente del adaptador.

    Si mañana alguien añade `import socket`, `import httpx` o `from urllib.request import
    urlopen` al adaptador —el primer paso inevitable de cualquier llamada de red propia—,
    esto se pone rojo y hay que elegir: retirar el import, o corregir la declaración.

    Se exige un SUELO de imports leídos porque «cero infractores sobre cero imports» es un
    verde vacío: un fichero que no fuese éste, o un `ast.parse` sobre nada, pasaría igual.
    """
    source = Path(cast(str, xbrain.mcp_server.__file__))
    imported = _imported_modules(source)
    assert len(imported) >= 5, f"sólo {len(imported)} imports leídos en {source}: verde vacío"
    offenders = _network_imports_in(source)
    assert not offenders, (
        f"{source.name} importa módulos que hablan red: {offenders}. La frontera de confianza "
        "de este módulo declara que el servidor no hace llamadas de red propias: o se retira "
        "el import, o se corrige la declaración."
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
    assert "uv pip install -e '.[mcp]'" in result.stderr, result.stderr
    assert "xbrain[" not in result.stderr, "names a package this repo does not publish"


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
