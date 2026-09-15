# tests/test_mcp_server.py
"""El servidor MCP (Plan 04 §4): tres herramientas, y nada propio detrás de ellas.

TODO LO DE AQUÍ ENTRA POR LA SUPERFICIE PÚBLICA DEL SERVIDOR — `build_server()` y los
métodos `list_tools()` / `call_tool()` del objeto que devuelve, que son exactamente por
donde entra el agente externo. Un test que llamara a la función interna que atiende una
herramienta dejaría descubierto el camino real (el registro de la tool, el esquema, el
envelope), que es donde vive el defecto que este fichero existe para cazar.

Las corrutinas se conducen con `asyncio.run` de la stdlib a propósito: el árbol no tiene
`pytest-asyncio` y añadir un plugin es tocar `pyproject.toml` y el lock, que es el hijo
04.6 y ya está integrado.
"""

from __future__ import annotations

import asyncio

from xbrain.mcp_server import MCP_TOOLS, build_server

# Las tres, literales. El conjunto viene del Plan 04 §4.1: `xbrain.search`,
# `xbrain.get` y `xbrain.graph_expand` son las tres puertas de los tres servicios.
EXPECTED_TOOLS = {"xbrain.search", "xbrain.get", "xbrain.graph_expand"}


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
