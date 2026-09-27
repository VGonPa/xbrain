"""`xbrain jev serve`: the Jev page served live on 127.0.0.1, and a JSON API to run passes.

HTTP ONLY. Everything the server decides is `jev.service.JevService`'s (the page's data, the
estimate, the confirmation, the one job and its cap); this module routes and guards:

* every request needs a `Host` that is this server (`127.0.0.1:<port>` / `localhost:<port>`):
  a page elsewhere that rebinds its DNS to 127.0.0.1 still sends its own name, and the page
  and `/api/data` carry the token;
* every POST needs an `Origin` that is this server, the per-start token in `X-Xbrain-Token`
  (compared in constant time), `application/json` and a body of at most `MAX_BODY` bytes; a
  GET that sends an `Origin` needs it to be this server's too;
* files are served only from `<output_dir>/_media/`, only images and videos, and nothing
  outside that folder however the path is spelled.

Routes: `/` (the page), `/api/data` (its blob), `/api/job`
(the one job), `POST /api/job/cancel` (the page's soft stop of that job), `/_media/…`, per
kind of pass (`topics`, `ask`) `POST /api/<kind>/estimate` and `POST /api/<kind>/evaluate`,
and what was asked: `/api/asks` (the query history with each query's results) and
`/api/ask/<sha>` (one query) — both read-only, both slices of the blob.

Bound to 127.0.0.1 only; there is no option to bind anything else. Ctrl-C stops accepting
requests, stops the job softly (what is in flight is waited for, saved and logged) and exits
130.
"""

from __future__ import annotations

import hmac
import json
import logging
import mimetypes
import re
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit

from xbrain.generate import VAULT_MEDIA_SUBDIR
from xbrain.jev.client import JevError
from xbrain.jev.errors import ServeError, refuse
from xbrain.jev.service import JevService

logger = logging.getLogger(__name__)

#: The only address the server binds. Not an option: the API spends money.
HOST = "127.0.0.1"
#: The header a POST carries the page's token in.
TOKEN_HEADER = "X-Xbrain-Token"
#: The largest request body read, in bytes. A pick is a few ids (at most `picks.MAX_IDS`).
MAX_BODY = 64 * 1024
#: `/api/<kind>/<action>`.
_PASS_ROUTE = re.compile(r"^/api/([a-z]+)/(estimate|evaluate)$")
#: `/api/ask/<sha>`: one query of the history, by the sha256 of its normalised text.
_ASK_ROUTE = re.compile(r"^/api/ask/([0-9a-f]{64})$")
#: What `/_media/` serves: what the page shows. An `.svg` or `.html` there would run as this
#: origin and could read the token, so anything else is not served.
_MEDIA_TYPES = ("image/", "video/")
#: An image type that is a document: SVG runs script.
_MEDIA_REFUSED = frozenset({"image/svg+xml"})


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    service: JevService


class _Handler(BaseHTTPRequestHandler):
    """Routing and the request guards; every decision is `JevService`'s."""

    server: _Server
    server_version = "xbrain-jev"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("jev serve: " + format, *args)

    # -------------------------------------------------------------- guards

    def _allowed_hosts(self) -> set[str]:
        port = self.server.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _guard(self, *, post: bool) -> None:
        """Refuse a foreign Host (every route), a foreign or missing Origin (POST; GET only
        when it sends one) and a missing or wrong token (POST)."""
        hosts = self._allowed_hosts()
        if self.headers.get("Host") not in hosts:
            raise ServeError(403, "Host no permitido: este servidor solo atiende a 127.0.0.1")
        origin = self.headers.get("Origin")
        if (post or origin is not None) and origin not in {f"http://{h}" for h in hosts}:
            raise ServeError(403, "Origin no permitido: solo la página de este servidor")
        if post:
            sent = self.headers.get(TOKEN_HEADER, "")
            if not hmac.compare_digest(sent.encode(), self.server.service.token.encode()):
                raise ServeError(403, "falta el token de este servidor, o no es el suyo")

    def _body(self) -> Any:
        kind = self.headers.get("Content-Type", "")
        if kind.split(";")[0].strip() != "application/json":
            raise ServeError(415, "el cuerpo debe ser JSON (Content-Type: application/json)")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise ServeError(411, "falta Content-Length") from None
        if length < 0:
            self.close_connection = True
            raise refuse("Content-Length negativo")
        if length > MAX_BODY:
            if length <= 16 * MAX_BODY:
                self.rfile.read(length)  # drained, so the refusal reaches the client
            self.close_connection = True
            raise ServeError(413, f"cuerpo demasiado grande (máximo {MAX_BODY} bytes)")
        try:
            return json.loads(self.rfile.read(length) or b"null")
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise refuse(f"JSON ilegible: {exc}") from exc

    # -------------------------------------------------------------- responses

    def _send(self, status: int, body: bytes, kind: str, *extra: tuple[str, str]) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for name, value in extra:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _handle(self, route: Callable[[], None]) -> None:
        try:
            route()
        except ServeError as exc:
            self._json(exc.status, {"error": exc.message})
        except JevError as exc:
            self._json(400, {"error": str(exc)})
        except Exception:
            # The detail goes to the log, never to the page: an exception's text can carry a
            # path, a key's name or a vendor message nobody meant to publish.
            logger.exception("jev serve: %s %s falló", self.command, self.path)
            self._json(
                500, {"error": "error interno del servidor (el detalle está en su registro)"}
            )

    # -------------------------------------------------------------- routes

    def do_GET(self) -> None:  # noqa: N802 — the stdlib's name
        self._handle(self._get)

    def do_POST(self) -> None:  # noqa: N802 — the stdlib's name
        self._handle(self._post)

    def _get(self) -> None:
        self._guard(post=False)
        url = urlsplit(self.path)
        service = self.server.service
        if url.path in ("/", "/jev.html"):
            self._send(200, service.page_html().encode("utf-8"), "text/html; charset=utf-8")
        elif url.path.startswith(f"/{VAULT_MEDIA_SUBDIR}/"):
            self._media(url.path)
        else:
            self._json(200, self._get_json(url.path))

    def _get_json(self, path: str) -> Any:
        """The read-only JSON routes: the blob, the job, and what was asked."""
        service = self.server.service
        if path == "/api/data":
            return service.blob()
        if path == "/api/job":
            return service.job_view()
        if path == "/api/asks":
            return service.asks()
        asked = _ASK_ROUTE.match(path)
        if asked is not None:
            return service.ask(asked.group(1))
        raise ServeError(404, "no existe")

    def _post(self) -> None:
        self._guard(post=True)
        path = urlsplit(self.path).path
        if path == "/api/job/cancel":
            if not isinstance(self._body(), (dict, type(None))):
                raise refuse("el cuerpo debe ser un objeto JSON ({})")
            self._json(200, self.server.service.cancel_job())
            return
        route = _PASS_ROUTE.match(path)
        if route is None:
            raise ServeError(404, "no existe")
        body = self._body()
        kind, action = route.groups()
        service = self.server.service
        if action == "estimate":
            self._json(200, service.estimate(kind, body))
        else:
            self._json(202, service.evaluate(kind, body))

    def _media(self, raw_path: str) -> None:
        """An image or video under `<output_dir>/_media/`, and nothing else: the path is
        decoded, resolved (symlinks too) and must stay inside the folder; a path the OS
        refuses (a NUL byte) is simply not there."""
        output_dir = self.server.service.cfg.output_dir
        try:
            root = (output_dir / VAULT_MEDIA_SUBDIR).resolve()
            target = (output_dir / unquote(raw_path).lstrip("/")).resolve()
            inside = target.is_relative_to(root) and target != root and target.is_file()
            kind = mimetypes.guess_type(target.name)[0] or ""
            if not inside or not kind.startswith(_MEDIA_TYPES) or kind in _MEDIA_REFUSED:
                raise ServeError(404, "no existe")
            body = target.read_bytes()
        except (OSError, ValueError):
            raise ServeError(404, "no existe") from None
        self._send(200, body, kind, ("Content-Security-Policy", "sandbox"))


def make_server(service: JevService, port: int) -> ThreadingHTTPServer:
    """The HTTP server for `service` on 127.0.0.1:`port` (0 = any free port). Not started."""
    server = _Server((HOST, port), _Handler)
    server.service = service
    return server


def serve_until_interrupted(
    server: ThreadingHTTPServer, service: JevService
) -> tuple[int, dict[str, Any]]:
    """Serve until Ctrl-C; then stop accepting, stop the job softly and let it save and log.
    Returns the exit code (130 after Ctrl-C) and the last job's view, for the terminal.

    The job runs in a NON-daemon thread, so even a second Ctrl-C while `stop` waits leaves the
    interpreter waiting for the job to write what was paid for.
    """
    code = 0
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        code = 130
    finally:
        server.server_close()
        service.stop()
    return code, service.job_view()
