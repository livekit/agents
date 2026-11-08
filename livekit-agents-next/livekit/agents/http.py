from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route, WebSocketRoute
from starlette.types import Message, Scope
from starlette.websockets import WebSocket

from . import _core
from ._generated.worker_endpoint import (
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
    RequestHead,
)
from ._generated.worker_server import EndpointRoute

logger = logging.getLogger(__name__)

_HttpEndpoint = Callable[[Request], Response | Awaitable[Response]]
_HttpEndpointT = TypeVar("_HttpEndpointT", bound=_HttpEndpoint)
_WebSocketEndpoint = Callable[[WebSocket], Awaitable[None]]
_WebSocketEndpointT = TypeVar("_WebSocketEndpointT", bound=_WebSocketEndpoint)


class AgentRoutes:
    """HTTP and WebSocket routes served next to the agent, at
    ``/agents/{agent_name}/{deployment}/{path}``::

        @server.http.get("/token", public=True)
        async def token(request: Request) -> Response:
            return JSONResponse({"token": ...})

    Handlers are Starlette endpoints. Routes require a LiveKit token unless ``public=True``.
    ``request.state.lk_request`` holds the request's ``EndpointRequest``.
    """

    def __init__(self) -> None:
        self._app = Starlette()
        self._manifest: list[EndpointRoute] = []

    def route(
        self, path: str, *, methods: Iterable[str], public: bool = False
    ) -> Callable[[_HttpEndpointT], _HttpEndpointT]:
        def decorator(func: _HttpEndpointT) -> _HttpEndpointT:
            route = Route(path, func, methods=list(methods))
            # Starlette adds HEAD to a GET route
            self._add(route, sorted(route.methods or ()), public)
            return func

        return decorator

    def get(self, path: str, *, public: bool = False) -> Callable[[_HttpEndpointT], _HttpEndpointT]:
        return self.route(path, methods=["GET"], public=public)

    def post(
        self, path: str, *, public: bool = False
    ) -> Callable[[_HttpEndpointT], _HttpEndpointT]:
        return self.route(path, methods=["POST"], public=public)

    def put(self, path: str, *, public: bool = False) -> Callable[[_HttpEndpointT], _HttpEndpointT]:
        return self.route(path, methods=["PUT"], public=public)

    def patch(
        self, path: str, *, public: bool = False
    ) -> Callable[[_HttpEndpointT], _HttpEndpointT]:
        return self.route(path, methods=["PATCH"], public=public)

    def delete(
        self, path: str, *, public: bool = False
    ) -> Callable[[_HttpEndpointT], _HttpEndpointT]:
        return self.route(path, methods=["DELETE"], public=public)

    def websocket(
        self, path: str, *, public: bool = False
    ) -> Callable[[_WebSocketEndpointT], _WebSocketEndpointT]:
        def decorator(func: _WebSocketEndpointT) -> _WebSocketEndpointT:
            self._add(WebSocketRoute(path, func), ["GET"], public)
            return func

        return decorator

    def _add(self, route: Route | WebSocketRoute, methods: list[str], public: bool) -> None:
        self._app.router.routes.append(route)
        self._manifest.append(
            EndpointRoute(path=route.path, methods=methods, kind="http", public=public)
        )

    async def _core_handle(self, exchange: _core.HttpExchange, request: EndpointRequest) -> None:
        scope = _scope(exchange.head, request)
        pending: list[Message] = (
            [{"type": "websocket.connect"}] if scope["type"] == "websocket" else []
        )

        async def receive() -> Message:
            return pending.pop() if pending else _event(await exchange.receive())

        async def send(message: Message) -> None:
            if (outgoing := _message(message)) is not None:
                await exchange.send(outgoing)

        try:
            await self._app(scope, receive, send)
        except Exception:
            logger.exception("endpoint application failed: %s", request.request_id)
            raise


def _scope(head: RequestHead, request: EndpointRequest) -> Scope:
    scope: Scope = {
        "type": head.kind,
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": head.http_version,
        "scheme": request.scheme,
        "path": head.path,
        "raw_path": head.raw_path,
        "query_string": head.query_string,
        "root_path": "",
        "headers": head.headers,
        "client": None,  # the node supplies a host only, not a client socket address
        "server": None,
        "state": {"lk_request": request},
    }
    if head.kind == "websocket":
        scope["scheme"] = {"http": "ws", "https": "wss"}[request.scheme]
        scope["subprotocols"] = head.subprotocols
        scope["extensions"] = {"websocket.http.response": {}}
    else:
        scope["method"] = head.method
    return scope


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
            return None  # the worker writes its own completion trailers
        case "websocket.accept":
            return ExchangeMessageAccept(
                subprotocol=message.get("subprotocol"),
                headers=_headers(message.get("headers", ())),
            )
        case "websocket.send":
            data, text = message.get("bytes"), message.get("text")
            if data is not None and text is None:
                return ExchangeMessageBinary(data=bytes(data))
            if text is not None and data is None:
                return ExchangeMessageText(text=text)
            raise RuntimeError("websocket.send carries exactly one of bytes and text")
        case "websocket.close":
            return ExchangeMessageClose(
                code=message.get("code", 1000), reason=message.get("reason") or ""
            )
        case kind:
            raise RuntimeError(f"unexpected ASGI message {kind!r}")


def _headers(headers: Iterable[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    return [(bytes(name), bytes(value)) for name, value in headers]
