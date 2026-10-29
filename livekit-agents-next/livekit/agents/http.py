"""``AgentServer.http``: the ASGI app behind an agent's HTTP endpoints, served on the endpoint data
plane.

Routes are served through the LiveKit server at ``/agents/{agent_name}/{deployment}/{path}``; no
local listener exposes them. The Rust worker (``_core.Worker``) accepts one stream per request,
parses the node's preamble and hands the stream here; ``_core.EndpointServer`` runs one ASGI call
over it: an HTTP/1.1 exchange, or a WebSocket after an upgrade.

The handler sees the preamble through ``request.state.lk_request`` (:class:`EndpointRequest`):
the request id, whether the caller holds the call permission, the manifest route that matched,
and whether the route is public (``configure_endpoint(path, public=True)``).
The project credential never reaches the worker: the node strips it and asserts ``authorized``
here; a thread grant reaches a public route as its bearer, for the handler to verify.
"""

from __future__ import annotations

import traceback
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Protocol, TypeVar

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Mount, Route, WebSocketRoute
from starlette.types import ASGIApp, Message, Scope
from starlette.websockets import WebSocket

from . import _core
from ._generated import worker_endpoint as asgi
from ._generated.worker_endpoint import (
    EndpointOptions as EndpointOptions,
    EndpointRequest as EndpointRequest,
)
from ._generated.worker_reset_codes import (
    HSR_ABORT as HSR_ABORT,
    HSR_INTERNAL as HSR_INTERNAL,
    HSR_PROTOCOL as HSR_PROTOCOL,
    HSR_REFUSED as HSR_REFUSED,
    HSR_TIMEOUT as HSR_TIMEOUT,
)

_HttpEndpoint = Callable[[Request], Response | Awaitable[Response]]
_HttpEndpointT = TypeVar("_HttpEndpointT", bound=_HttpEndpoint)
_WebSocketEndpoint = Callable[[WebSocket], Awaitable[None]]
_WebSocketEndpointT = TypeVar("_WebSocketEndpointT", bound=_WebSocketEndpoint)

_ALL_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"]


class AgentHttp:
    """The routes an agent serves, as a Starlette application: its text doors, mounted by
    ``AgentServer.add``, served through the LiveKit server at
    ``/agents/{agent_name}/{deployment}/{path}``; a handler reads the request's preamble as
    ``request.state.lk_request``.

    Handlers are Starlette endpoints: a request in, a response out. Any ASGI application mounts.
    """

    def __init__(self) -> None:
        self._routes: list[BaseRoute] = []
        self._app: ASGIApp | None = None
        self._options: dict[str, EndpointOptions] = {}
        self._server = _core.EndpointServer()

    # --- registration -------------------------------------------------------------------------

    def route(
        self, path: str, *, methods: Iterable[str], name: str | None = None
    ) -> Callable[[_HttpEndpointT], _HttpEndpointT]:
        def decorator(func: _HttpEndpointT) -> _HttpEndpointT:
            self.add_route(path, func, methods=list(methods), name=name)
            return func

        return decorator

    def get(
        self, path: str, *, name: str | None = None
    ) -> Callable[[_HttpEndpointT], _HttpEndpointT]:
        return self.route(path, methods=["GET"], name=name)

    def post(
        self, path: str, *, name: str | None = None
    ) -> Callable[[_HttpEndpointT], _HttpEndpointT]:
        return self.route(path, methods=["POST"], name=name)

    def put(
        self, path: str, *, name: str | None = None
    ) -> Callable[[_HttpEndpointT], _HttpEndpointT]:
        return self.route(path, methods=["PUT"], name=name)

    def patch(
        self, path: str, *, name: str | None = None
    ) -> Callable[[_HttpEndpointT], _HttpEndpointT]:
        return self.route(path, methods=["PATCH"], name=name)

    def delete(
        self, path: str, *, name: str | None = None
    ) -> Callable[[_HttpEndpointT], _HttpEndpointT]:
        return self.route(path, methods=["DELETE"], name=name)

    def websocket(
        self, path: str, *, name: str | None = None
    ) -> Callable[[_WebSocketEndpointT], _WebSocketEndpointT]:
        def decorator(func: _WebSocketEndpointT) -> _WebSocketEndpointT:
            self.add_websocket_route(path, func, name=name)
            return func

        return decorator

    def add_route(
        self,
        path: str,
        endpoint: _HttpEndpoint | ASGIApp,
        *,
        methods: list[str],
        name: str | None = None,
    ) -> None:
        self._add(Route(path, endpoint, methods=methods, name=name))

    def add_websocket_route(
        self, path: str, endpoint: _WebSocketEndpoint | ASGIApp, *, name: str | None = None
    ) -> None:
        self._add(WebSocketRoute(path, endpoint, name=name))

    def mount(self, path: str, app: ASGIApp, *, name: str | None = None) -> None:
        """Serve an ASGI application (FastAPI, another Starlette app, ...) under ``path``."""
        self._add(Mount(path, app=app, name=name))

    def configure_endpoint(self, path: str, *, public: bool = False) -> None:
        """Set routing policy for one route, by its registered path."""
        self._options[path] = EndpointOptions(public=public)

    def _add(self, route: BaseRoute) -> None:
        if self._app is not None:
            raise RuntimeError("routes cannot be added once the agent server is running")
        self._routes.append(route)

    @property
    def routes(self) -> list[BaseRoute]:
        return list(self._routes)

    # --- what the worker needs --------------------------------------------------------------

    @property
    def app(self) -> ASGIApp:
        """The ASGI application, built on first use; routes are fixed from then on."""
        if self._app is None:
            self._app = Starlette(routes=self._routes)
        return self._app

    def manifest(self) -> list[dict[str, Any]]:
        """The endpoint manifest the worker registers, in route order: ``{path, methods, kind,
        public}`` per route, as ``_core.Worker`` takes it.

        Methods travel exactly as Starlette routes them (a GET route also answers HEAD); a
        WebSocket route is a GET; a mount declares every method under ``{path:path}``.
        """
        manifest: list[dict[str, Any]] = []
        for route in self._routes:
            if isinstance(route, Route):
                path, methods = route.path, sorted(route.methods or ())
            elif isinstance(route, WebSocketRoute):
                path, methods = route.path, ["GET"]
            elif isinstance(route, Mount):
                path, methods = route.path.rstrip("/") + "/{path:path}", list(_ALL_METHODS)
            else:
                continue
            opts = self._options.get(route.path, EndpointOptions())
            manifest.append(
                {
                    "path": path,
                    "methods": methods,
                    "kind": "http",
                    "public": opts.public,
                }
            )
        return manifest

    async def close_websockets(self, code: int = 1001, reason: str = "worker draining") -> None:
        """Send every open WebSocket a close frame, so clients reconnect to another worker."""
        await self._server.close_websockets(code, reason)


async def serve_endpoint(
    stream: ExchangeStream,
    request: EndpointRequest,
    app: ASGIApp,
    registry: AgentHttp | None = None,
) -> None:
    """Serve one exchange stream (an ``EndpointStream`` after its preamble, or any object with
    its five methods) with ``app``; ``registry`` keeps its WebSocket for draining."""
    server = registry._server if registry is not None else _core.EndpointServer()
    await server.serve(stream, AsgiApplication(app), request)


class ExchangeStream(Protocol):
    """One exchange after its preamble: ``_core.EndpointStream``, or a test's node."""

    def read(self, max_bytes: int) -> Awaitable[bytes]: ...
    def write(self, data: bytes) -> Awaitable[None]: ...
    def finish(self) -> None: ...
    def reset(self, code: int) -> None: ...
    def stop(self, code: int) -> None: ...


class AsgiApplication:
    """An ASGI application as the core calls it: the declared scope and messages shaped as
    ASGI dicts, and the application's failure answered with its traceback."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def _core_call(
        self,
        scope: asgi.AsgiScope,
        request: EndpointRequest,
        receive: _core._core_AsgiReceive,
        send: _core._core_AsgiSend,
    ) -> asgi.AsgiFailure | None:
        async def asgi_receive() -> Message:
            return _received(await receive())

        async def asgi_send(message: Message) -> None:
            await send(_sent(message))

        try:
            await self._app(_scope(scope, request), asgi_receive, asgi_send)
        except Exception as e:
            return asgi.AsgiFailure(
                summary=f"{type(e).__qualname__}: {e}",
                traceback="".join(traceback.format_tb(e.__traceback__)),
            )
        return None


def _scope(scope: asgi.AsgiScope, request: EndpointRequest) -> Scope:
    d: Scope = {
        "type": scope.kind,
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "scheme": scope.scheme,
        "path": scope.path,
        "raw_path": scope.raw_path,
        "query_string": scope.query_string,
        "root_path": "",
        "headers": list(scope.headers),
        "client": scope.client,
        "server": None,
        "state": {"lk_request": request},
    }
    if scope.kind == "http":
        d["method"] = scope.method
    else:
        d["subprotocols"] = list(scope.subprotocols)
    return d


def _received(message: asgi.AsgiReceived) -> Message:
    match message:
        case asgi.AsgiReceivedRequest(body=body, more_body=more_body):
            return {"type": "http.request", "body": body, "more_body": more_body}
        case asgi.AsgiReceivedDisconnect():
            return {"type": "http.disconnect"}
        case asgi.AsgiReceivedWsConnect():
            return {"type": "websocket.connect"}
        case asgi.AsgiReceivedWsText(text=text):
            return {"type": "websocket.receive", "text": text}
        case asgi.AsgiReceivedWsBytes(data=data):
            return {"type": "websocket.receive", "bytes": data}
        case asgi.AsgiReceivedWsDisconnect(code=code):
            return {"type": "websocket.disconnect", "code": code}
    raise TypeError(f"unknown ASGI message {message!r}")


def _sent(m: Message) -> asgi.AsgiSent:
    """ASGI headers and bodies may be any bytes-like object."""

    def headers() -> list[tuple[bytes, bytes]]:
        return [(bytes(k), bytes(v)) for k, v in m.get("headers") or ()]

    def body() -> bytes:
        return bytes(m.get("body") or b"")

    kind: str = m["type"]
    if kind == "http.response.start":
        return asgi.AsgiSentResponseStart(status=m["status"], headers=headers())
    if kind == "http.response.body":
        return asgi.AsgiSentResponseBody(body=body(), more_body=bool(m.get("more_body")))
    if kind == "http.response.trailers":
        return asgi.AsgiSentResponseTrailers()
    if kind == "websocket.accept":
        return asgi.AsgiSentWsAccept(subprotocol=m.get("subprotocol"), headers=headers())
    if kind == "websocket.send":
        if (data := m.get("bytes")) is not None:
            return asgi.AsgiSentWsBytes(data=bytes(data))
        return asgi.AsgiSentWsText(text=m.get("text") or "")
    if kind == "websocket.close":
        return asgi.AsgiSentWsClose(code=m.get("code") or 1000, reason=m.get("reason") or "")
    if kind == "websocket.http.response.start":
        return asgi.AsgiSentWsHttpResponseStart(status=m["status"], headers=headers())
    if kind == "websocket.http.response.body":
        return asgi.AsgiSentWsHttpResponseBody(body=body(), more_body=bool(m.get("more_body")))
    return asgi.AsgiSentUnknown(kind=kind)
