"""``AgentServer.http``: the agent's HTTP routes, served as an ASGI application.

Routes are served at ``/agents/{agent_name}/{deployment}/{path}``: through LiveKit Cloud, or the
worker's public listener where nothing routes them (a self-hosted server, the console). The Rust
worker owns each connection: it parses the request, enforces limits and deadlines, frames the
response and closes WebSockets. Python only adapts each parsed exchange to one ASGI call.

The handler sees the preamble through ``request.state.lk_request`` (:class:`EndpointRequest`):
the request id, whether the caller holds the call permission, the manifest route that matched,
and whether the route is public (``configure_endpoint(path, public=True)``).
The project credential never reaches the worker: the node strips it and asserts ``authorized``
here; a thread grant reaches a public route as its bearer, for the handler to verify.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from functools import cached_property
from typing import TypeVar

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Mount, Route, WebSocketRoute
from starlette.types import ASGIApp, Message, Scope
from starlette.websockets import WebSocket

from . import _core
from ._generated.worker_endpoint import (
    EndpointOptions as EndpointOptions,
    EndpointRequest as EndpointRequest,
    ExchangeEvent,
    ExchangeEventBinary,
    ExchangeEventBody,
    ExchangeEventClosed,
    ExchangeEventDisconnected,
    ExchangeEventText,
    ExchangeMessage,
    ExchangeMessageAccept,
    ExchangeMessageBinary,
    ExchangeMessageBody,
    ExchangeMessageClose,
    ExchangeMessageStart,
    ExchangeMessageText,
)
from ._generated.worker_server import EndpointRoute

logger = logging.getLogger(__name__)

_HttpEndpoint = Callable[[Request], Response | Awaitable[Response]]
_HttpEndpointT = TypeVar("_HttpEndpointT", bound=_HttpEndpoint)
_WebSocketEndpoint = Callable[[WebSocket], Awaitable[None]]
_WebSocketEndpointT = TypeVar("_WebSocketEndpointT", bound=_WebSocketEndpoint)

_ALL_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"]


class AgentRoutes:
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

    # --- what the worker needs ----------------------------------------------------------------

    @cached_property
    def _server(self) -> _AsgiServer:
        return _AsgiServer(self.app)

    @property
    def app(self) -> ASGIApp:
        """The ASGI application, built on first use; routes are fixed from then on."""
        if self._app is None:
            self._app = Starlette(routes=self._routes)
        return self._app

    def manifest(self) -> list[EndpointRoute]:
        """The endpoint manifest the worker registers, in route order as typed ``EndpointRoute``
        records, as ``_core.Worker`` takes them.

        Methods travel exactly as Starlette routes them (a GET route also answers HEAD); a
        WebSocket route is a GET; a mount declares every method under ``{path:path}``.
        """
        manifest: list[EndpointRoute] = []
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
                EndpointRoute(path=path, methods=methods, kind="http", public=opts.public)
            )
        return manifest


class _AsgiServer:
    """Run the ASGI application once per exchange the worker serves."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def _core_handle(self, exchange: _core.HttpExchange, request: EndpointRequest) -> None:
        head = exchange.head
        websocket = head.kind == "websocket"
        scope: Scope = {
            "type": head.kind,
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": head.http_version,
            "scheme": {"http": "ws", "https": "wss"}[request.scheme]
            if websocket
            else request.scheme,
            "path": head.path,
            "raw_path": head.raw_path,
            "query_string": head.query_string,
            "root_path": "",
            "headers": head.headers,
            "client": None,  # The node supplies a host only, not a client socket address.
            "server": None,
            "state": {"lk_request": request},
        }
        if websocket:
            scope["subprotocols"] = head.subprotocols
            scope["extensions"] = {"websocket.http.response": {}}
        else:
            scope["method"] = head.method
        connected = not websocket

        async def receive() -> Message:
            nonlocal connected
            if not connected:
                connected = True
                return {"type": "websocket.connect"}
            return _event(await exchange.receive())

        async def send(message: Message) -> None:
            if (outgoing := _message(message)) is not None:
                await exchange.send(outgoing)

        try:
            await self._app(scope, receive, send)
        except Exception:
            logger.exception("endpoint application failed: %s", request.request_id)
            raise


def _event(event: ExchangeEvent) -> Message:
    match event:
        case ExchangeEventBody(data=data, more=more):
            return {"type": "http.request", "body": data, "more_body": more}
        case ExchangeEventDisconnected():
            return {"type": "http.disconnect"}
        case ExchangeEventText(text=text):
            return {"type": "websocket.receive", "text": text}
        case ExchangeEventBinary(data=data):
            return {"type": "websocket.receive", "bytes": data}
        case ExchangeEventClosed(code=code):
            return {"type": "websocket.disconnect", "code": code}


def _message(message: Message) -> ExchangeMessage | None:
    """The exchange message for an ASGI message; ``None`` for one the worker has no use for."""
    match message["type"]:
        case "http.response.start" | "websocket.http.response.start":
            return ExchangeMessageStart(
                status=message["status"], headers=_headers(message.get("headers", ()))
            )
        case "http.response.body" | "websocket.http.response.body":
            return ExchangeMessageBody(
                data=bytes(message.get("body", b"")), more=message.get("more_body", False)
            )
        case "http.response.trailers":
            return None
        case "websocket.accept":
            return ExchangeMessageAccept(
                subprotocol=message.get("subprotocol"),
                headers=_headers(message.get("headers", ())),
            )
        case "websocket.send":
            if (data := message.get("bytes")) is not None:
                return ExchangeMessageBinary(data=bytes(data))
            return ExchangeMessageText(text=message.get("text") or "")
        case "websocket.close":
            return ExchangeMessageClose(
                code=message.get("code", 1000), reason=message.get("reason") or ""
            )
        case kind:
            raise RuntimeError(f"unexpected ASGI message {kind!r}")


def _headers(headers: Iterable[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    return [(bytes(name), bytes(value)) for name, value in headers]
