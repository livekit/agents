"""``AgentServer.http``: the ASGI app behind an agent's HTTP endpoints, served on the endpoint data
plane.

Routes are served through the LiveKit server at ``/agents/{agent_name}/{deployment}/{path}``; no
local listener exposes them. The Rust worker (``_core.Worker``) accepts one stream per request,
parses the node's preamble and hands the byte stream here. Python runs one ASGI call
over it: an HTTP/1.1 exchange, or a WebSocket after an upgrade.

The handler sees the preamble through ``request.state.lk_request`` (:class:`EndpointRequest`):
the request id, whether the caller holds the call permission, the manifest route that matched,
and whether the route is public (``configure_endpoint(path, public=True)``).
The project credential never reaches the worker: the node strips it and asserts ``authorized``
here; a thread grant reaches a public route as its bearer, for the handler to verify.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from functools import cached_property
from http import HTTPStatus
from typing import Literal, Protocol, TypeVar
from urllib.parse import unquote

import h11
import wsproto
import wsproto.events as ws
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Mount, Route, WebSocketRoute
from starlette.types import ASGIApp, Message, Scope
from starlette.websockets import WebSocket
from wsproto.frame_protocol import CloseReason
from wsproto.utilities import RemoteProtocolError as WsProtocolError

from . import _core
from ._generated.http import _HttpServerLimits as _HttpServerLimits
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
from ._generated.worker_server import EndpointRoute

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

    # --- what the worker needs --------------------------------------------------------------

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

    async def close_websockets(
        self, code: int = CloseReason.GOING_AWAY, reason: str = "worker draining"
    ) -> None:
        """Send every open WebSocket a close frame, so clients reconnect to another worker."""
        await self._server.close_websockets(code, reason)


async def serve_endpoint(
    stream: ByteTransport,
    request: EndpointRequest,
    app: ASGIApp,
    registry: AgentRoutes | None = None,
) -> None:
    """Serve one exchange stream (an ``ByteStream`` after its preamble, or any object with
    its five methods) with ``app``. When supplied, ``registry`` must own ``app``
    and keeps its WebSockets for draining."""
    if registry is not None and app is not registry.app:
        raise ValueError("registry must own the supplied ASGI app")
    server = registry._server if registry is not None else _AsgiServer(app)
    await server._core_handle(stream, request)


class ByteTransport(Protocol):
    """One exchange after its preamble: ``_core.ByteStream``, or a test's node."""

    def read(self, max_bytes: int) -> Awaitable[bytes]: ...
    def write(self, data: bytes) -> Awaitable[None]: ...
    def finish(self) -> None: ...
    def reset(self, code: int) -> None: ...
    def stop(self, code: int) -> None: ...


logger = logging.getLogger(__name__)
_READ_SIZE = 64 << 10
_COMPLETION_REASON_LIMIT = 256
_Completion = Literal["internal", "timeout", "canceled"]
_ScopeKind = Literal["http", "websocket"]


def _scope(head: h11.Request, request: EndpointRequest, kind: _ScopeKind) -> Scope:
    raw_path, _, query = head.target.partition(b"?")
    return {
        "type": kind,
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": head.http_version.decode("ascii"),
        "scheme": request.scheme,
        "path": unquote(raw_path.decode("latin-1")),
        "raw_path": raw_path,
        "query_string": query,
        "root_path": "",
        "headers": list(head.headers),
        "client": None,  # The node supplies a host only, not a client socket address.
        "server": None,
        "state": {"lk_request": request},
    }


class _BoundedInbox:
    """Own queued payloads, their capacity and notification of data or closure.

    Receiving transfers a payload and releases its reservation without suspending.
    Closing discards pending payloads; cancellation only interrupts notification waits.
    """

    def __init__(
        self,
        *,
        max_items: int,
        max_bytes: int,
        reserve: Callable[[int], bool],
        release: Callable[[int], None],
    ) -> None:
        if max_items < 0 or max_bytes < 0:
            raise ValueError("inbox limits must be nonnegative")
        self._max_items = max_items
        self._max_bytes = max_bytes
        self._reserve = reserve
        self._release = release
        self._items: deque[tuple[str | bytes, int]] = deque()
        self._bytes = 0
        self._closed = False
        self._ready = asyncio.Event()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def queued_bytes(self) -> int:
        return self._bytes

    @property
    def closed(self) -> bool:
        return self._closed

    def try_put(self, payload: str | bytes) -> bool:
        size = len(payload.encode()) if isinstance(payload, str) else len(payload)
        if (
            self._closed
            or len(self._items) >= self._max_items
            or self._bytes + size > self._max_bytes
            or not self._reserve(size)
        ):
            return False
        self._items.append((payload, size))
        self._bytes += size
        self._ready.set()
        return True

    async def receive(self) -> str | bytes | None:
        while not self._closed:
            if self._items:
                payload, size = self._items.popleft()
                self._bytes -= size
                self._release(size)
                return payload
            self._ready.clear()
            await self._ready.wait()
        return None

    async def wait_closed(self) -> None:
        while not self._closed:
            self._ready.clear()
            await self._ready.wait()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._release(self._bytes)
        self._items.clear()
        self._bytes = 0
        self._ready.set()


class _AsgiServer:
    """Serve an ASGI app over worker streams and drain its open WebSocket connections."""

    def __init__(self, app: ASGIApp, limits: _HttpServerLimits | None = None) -> None:
        self._app = app
        self.limits = limits or _HttpServerLimits()
        self._capacity = _core._core_HttpServerCapacity(
            self.limits.pending_handlers, self.limits.payload_bytes
        )
        self.websockets: set[_WebSocketConnection] = set()

    @property
    def handlers(self) -> int:
        return self._capacity.handlers

    @property
    def payload_bytes(self) -> int:
        return self._capacity.payload_bytes

    def reserve(self, size: int) -> bool:
        return self._capacity.reserve(size)

    def release(self, size: int) -> None:
        self._capacity.release(size)

    async def _core_drain(self) -> None:
        await self.close_websockets(CloseReason.GOING_AWAY, "worker draining")

    async def close_websockets(self, code: int, reason: str) -> None:
        await asyncio.gather(*(ex.close(code, reason) for ex in tuple(self.websockets)))

    async def _core_handle(self, stream: ByteTransport, request: EndpointRequest) -> None:
        reservation = self.limits.request_head_bytes + _READ_SIZE
        if not self._capacity.admit(reservation):
            stream.reset(HSR_REFUSED)
            return
        exchange: _HttpExchange | _WebSocketConnection | None = None
        try:
            # Bound even a complete head delivered in one read; h11's incomplete-event limit
            # alone does not constrain complete requests.
            raw = bytearray()
            while (end := raw.find(b"\r\n\r\n")) < 0:
                if len(raw) >= self.limits.request_head_bytes:
                    raise h11.RemoteProtocolError("request head exceeds byte limit")
                data = await stream.read(min(_READ_SIZE, self.limits.request_head_bytes - len(raw)))
                if not data:
                    raise h11.RemoteProtocolError("stream ended before request head")
                raw.extend(data)
            if end + 4 > self.limits.request_head_bytes:
                raise h11.RemoteProtocolError("request head exceeds byte limit")
            protocol = h11.Connection(
                h11.SERVER, max_incomplete_event_size=self.limits.body_metadata_bytes
            )
            protocol.receive_data(bytes(raw))
            head = protocol.next_event()
            if not isinstance(head, h11.Request):
                raise h11.RemoteProtocolError("expected an HTTP request")

            def has_token(name: bytes, token: bytes) -> bool:
                return any(
                    token == part.strip().lower()
                    for key, value in head.headers
                    if key == name
                    for part in value.split(b",")
                )

            upgrade = has_token(b"upgrade", b"websocket") and has_token(b"connection", b"upgrade")
            if upgrade and head.method == b"GET":
                parser_bytes = self.limits.websocket_message_bytes + _READ_SIZE
                if not self.reserve(parser_bytes):
                    stream.reset(HSR_REFUSED)
                    return
                reservation += parser_bytes
                exchange = _WebSocketConnection(self, stream, head, bytes(raw[end + 4 :]))
                self.websockets.add(exchange)
                scope = _scope(head, request, "websocket")
                scope["scheme"] = "wss" if request.scheme == "https" else "ws"
                scope["subprotocols"] = exchange.subprotocols
            else:
                exchange = _HttpExchange(stream, protocol, head.method == b"HEAD")
                scope = _scope(head, request, "http")
                scope["method"] = head.method.decode("ascii")
            await exchange.run(self._app, scope, request)
        except asyncio.CancelledError:
            if isinstance(exchange, _WebSocketConnection):
                await exchange.close(CloseReason.GOING_AWAY, "worker stopped")
            elif isinstance(exchange, _HttpExchange):
                await exchange.fail("canceled", "worker stopped", HSR_ABORT)
            else:
                stream.reset(HSR_ABORT)
            raise
        except (h11.RemoteProtocolError, WsProtocolError):
            stream.reset(HSR_PROTOCOL)
        except Exception:
            stream.reset(HSR_INTERNAL)
            logger.exception("endpoint stream failed: %s", request.request_id)
        finally:
            try:
                if isinstance(exchange, _WebSocketConnection):
                    self.websockets.discard(exchange)
                    await exchange.cleanup()
            finally:
                self._capacity.complete(reservation)


class _HttpExchange:
    def __init__(self, stream: ByteTransport, protocol: h11.Connection, is_head: bool) -> None:
        self.stream = stream
        self.protocol = protocol
        self.is_head = is_head
        self.read_lock = asyncio.Lock()
        self.write_lock = asyncio.Lock()
        self.started = asyncio.Event()
        self.done = asyncio.Event()
        self.request_done = False
        self.chunked = False

    async def receive(self) -> Message:
        async with self.read_lock:
            if self.request_done:
                # The node's FIN ends the request body; it does not disconnect a response.
                await self.done.wait()
                return {"type": "http.disconnect"}
            try:
                while True:
                    event = self.protocol.next_event()
                    if isinstance(event, h11.Data):
                        return {
                            "type": "http.request",
                            "body": bytes(event.data),
                            "more_body": True,
                        }
                    if isinstance(event, h11.EndOfMessage):
                        self.request_done = True
                        return {"type": "http.request", "body": b"", "more_body": False}
                    if event is h11.NEED_DATA:
                        self.protocol.receive_data(await self.stream.read(_READ_SIZE))
                    else:
                        raise h11.RemoteProtocolError("incomplete request body")
            except (h11.RemoteProtocolError, OSError, RuntimeError):
                self.request_done = True
                self.done.set()
                self.stream.reset(HSR_PROTOCOL)
                return {"type": "http.disconnect"}

    async def _write(self, event: h11.Event) -> None:
        data = self.protocol.send(event)
        if data:
            await self.stream.write(data)

    async def send(self, message: Message) -> None:
        async with self.write_lock:
            if self.done.is_set():
                raise RuntimeError("endpoint exchange is closed")
            kind = message["type"]
            if kind == "http.response.start":
                headers = [(bytes(k), bytes(v)) for k, v in message.get("headers", [])]
                self.chunked = (
                    self.protocol.their_http_version == b"1.1"
                    and not self.is_head
                    and message["status"] not in (HTTPStatus.NO_CONTENT, HTTPStatus.NOT_MODIFIED)
                    and not any(k.lower() == b"content-length" for k, _ in headers)
                )
                await self._write(h11.Response(status_code=message["status"], headers=headers))
                self.started.set()
            elif kind == "http.response.body":
                await self._write(h11.Data(data=bytes(message.get("body", b""))))
                if not message.get("more_body", False):
                    await self._write(h11.EndOfMessage())
                    self.stream.finish()
                    self.stream.stop(HSR_ABORT)
                    self.done.set()
            elif kind != "http.response.trailers":
                raise RuntimeError(f"unexpected ASGI message {kind!r} on an http scope")

    async def fail(self, completion: _Completion, reason: str, code: int) -> None:
        if self.done.is_set():
            return
        async with self.write_lock:
            if not self.started.is_set():
                self.stream.reset(code)
            elif self.chunked and self.protocol.our_state is h11.SEND_BODY:
                reason = "".join(c if ord(c) >= 32 and ord(c) != 127 else " " for c in reason)[
                    :_COMPLETION_REASON_LIMIT
                ]
                await self._write(
                    h11.EndOfMessage(
                        headers=[
                            (b"x-lk-completion", completion.encode()),
                            (b"x-lk-error", reason.encode("utf-8")),
                        ]
                    )
                )
                self.stream.finish()
            else:
                self.stream.reset(HSR_ABORT)
            self.done.set()

    async def run(self, app: ASGIApp, scope: Scope, request: EndpointRequest) -> None:
        async def call() -> None:
            try:
                await app(scope, self.receive, self.send)
            except Exception as error:
                logger.exception("endpoint application failed: %s", request.request_id)
                await self.fail("internal", f"{type(error).__qualname__}: {error}", HSR_INTERNAL)
            else:
                await self.fail(
                    "internal", "application returned with no complete response", HSR_INTERNAL
                )
            self.done.set()

        task = asyncio.create_task(call())
        head = asyncio.create_task(self.started.wait())
        try:
            if request.timeout_ms:
                ready, _ = await asyncio.wait(
                    (task, head),
                    timeout=request.timeout_ms / 1000,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not ready:
                    task.cancel()
                    await self.fail("timeout", "no response head before the deadline", HSR_TIMEOUT)
                    return
            await task
        finally:
            head.cancel()
            task.cancel()
            await asyncio.gather(task, head, return_exceptions=True)


class _WebSocketFrameLimiter:
    """Reject an oversized declared frame before buffering its payload in wsproto."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.header = bytearray()
        self.left = 0

    def feed(self, data: bytes) -> bool:
        pos = 0
        while pos < len(data):
            if self.left:
                size = min(self.left, len(data) - pos)
                self.left -= size
                pos += size
                continue
            self.header.append(data[pos])
            pos += 1
            if len(self.header) < 2:
                continue
            size = self.header[1] & 127
            extra = 2 if size == 126 else 8 if size == 127 else 0
            if len(self.header) < 2 + extra:
                continue
            length = int.from_bytes(self.header[2 : 2 + extra], "big") if extra else size
            if length > self.limit:
                return False
            masked = bool(self.header[1] & 128)
            if len(self.header) < 2 + extra + (4 if masked else 0):
                continue
            self.header.clear()
            self.left = length
        return True


class _WebSocketConnection:
    def __init__(
        self, server: _AsgiServer, stream: ByteTransport, head: h11.Request, rest: bytes
    ) -> None:
        self.server = server
        self.stream = stream
        self.protocol = wsproto.WSConnection(wsproto.ConnectionType.SERVER)
        self.protocol.initiate_upgrade_connection(list(head.headers), head.target)
        request = next(self.protocol.events())
        if not isinstance(request, ws.Request):
            raise h11.RemoteProtocolError("invalid websocket handshake")
        self.subprotocols = request.subprotocols
        self.rest = rest
        self.accepted = asyncio.Event()
        self.code: int = CloseReason.ABNORMAL_CLOSURE
        self.connect = True
        self._close_task: asyncio.Task[None] | None = None
        self.inbox = _BoundedInbox(
            max_items=server.limits.websocket_inbox,
            max_bytes=server.limits.websocket_inbox_bytes,
            reserve=server.reserve,
            release=server.release,
        )
        self.outbox_bytes = 0
        self.write_lock = asyncio.Lock()
        self.reader: asyncio.Task[None] | None = None
        self.frame_limit = _WebSocketFrameLimiter(server.limits.websocket_frame_bytes)

    @property
    def closing(self) -> bool:
        return self._close_task is not None or self.inbox.closed

    async def receive(self) -> Message:
        if self.connect:
            self.connect = False
            return {"type": "websocket.connect"}
        payload = await self.inbox.receive()
        if payload is None or self.closing:
            await self.inbox.wait_closed()
            return {"type": "websocket.disconnect", "code": self.code}
        if isinstance(payload, str):
            return {"type": "websocket.receive", "text": payload}
        return {"type": "websocket.receive", "bytes": payload}

    async def _write(self, event: ws.Event) -> None:
        async with self.write_lock:
            data = self.protocol.send(event)
            if data:
                await self.stream.write(data)

    async def send(self, message: Message) -> None:
        kind = message["type"]
        if kind == "websocket.accept":
            if self.accepted.is_set():
                raise RuntimeError("websocket already accepted")
            await self._write(
                ws.AcceptConnection(
                    subprotocol=message.get("subprotocol"),
                    extra_headers=message.get("headers", []),
                )
            )
            self.accepted.set()
        elif kind == "websocket.send":
            if not self.accepted.is_set():
                raise RuntimeError("websocket.send before accept")
            if self.closing:
                return
            event: ws.Event
            if message.get("bytes") is not None:
                data = bytes(message["bytes"])
                size = len(data)
                event = ws.BytesMessage(data=data)
            else:
                text = message.get("text", "")
                size = len(text.encode())
                event = ws.TextMessage(data=text)
            limits = self.server.limits
            if (
                size > limits.websocket_message_bytes
                or self.outbox_bytes + size > limits.websocket_outbox_bytes
            ):
                raise RuntimeError("websocket response capacity exhausted")
            if not self.server.reserve(size):
                raise RuntimeError("worker response capacity exhausted")
            self.outbox_bytes += size
            try:
                await self._write(event)
            finally:
                self.outbox_bytes -= size
                self.server.release(size)
        elif kind == "websocket.close":
            await self.close(
                message.get("code", CloseReason.NORMAL_CLOSURE), message.get("reason", "")
            )
        elif kind == "websocket.http.response.start":
            await self._write(
                ws.RejectConnection(
                    status_code=message["status"],
                    headers=message.get("headers", []),
                    has_body=True,
                )
            )
        elif kind == "websocket.http.response.body":
            await self._write(
                ws.RejectData(
                    data=bytes(message.get("body", b"")),
                    body_finished=not message.get("more_body", False),
                )
            )
            if not message.get("more_body", False):
                self.stream.finish()
                self.inbox.close()
        else:
            raise RuntimeError(f"unexpected ASGI message {kind!r} on a websocket scope")

    async def close(self, code: int, reason: str) -> None:
        if self.inbox.closed:
            return
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._finish_close(code, reason))
        # A caller owns its wait, while the connection owns the close operation.
        await asyncio.shield(self._close_task)

    async def _finish_close(self, code: int, reason: str) -> None:
        self.code = code
        try:
            async with asyncio.timeout(self.server.limits.close_flush_timeout):
                if self.accepted.is_set():
                    await self._write(ws.CloseConnection(code=code, reason=reason))
                else:
                    await self._write(ws.RejectConnection(status_code=HTTPStatus.FORBIDDEN))
                self.stream.finish()
        except asyncio.CancelledError:
            self.stream.reset(HSR_ABORT)
            raise
        except (TimeoutError, OSError, RuntimeError):
            self.stream.reset(HSR_ABORT)
        finally:
            self.inbox.close()

    async def _read(self) -> None:
        text: list[str] = []
        binary = bytearray()
        size = 0
        try:
            await self.accepted.wait()
            data, self.rest = self.rest, b""
            while not self.closing:
                if not data:
                    data = await self.stream.read(_READ_SIZE)
                    if not data:
                        break
                if not self.frame_limit.feed(data):
                    await self.close(CloseReason.MESSAGE_TOO_BIG, "websocket frame limit exceeded")
                    return
                self.protocol.receive_data(data)
                for event in self.protocol.events():
                    if isinstance(event, ws.Ping):
                        await self._write(event.response())
                    elif isinstance(event, ws.CloseConnection):
                        await self.close(event.code, event.reason or "")
                        return
                    elif isinstance(event, (ws.TextMessage, ws.BytesMessage)):
                        piece = event.data.encode() if isinstance(event.data, str) else event.data
                        size += len(piece)
                        if size > self.server.limits.websocket_message_bytes:
                            await self.close(
                                CloseReason.MESSAGE_TOO_BIG, "websocket message limit exceeded"
                            )
                            return
                        if isinstance(event, ws.TextMessage):
                            text.append(event.data)
                        else:
                            binary.extend(event.data)
                        if event.message_finished:
                            payload = (
                                "".join(text)
                                if isinstance(event, ws.TextMessage)
                                else bytes(binary)
                            )
                            if not self.inbox.try_put(payload):
                                await self.close(
                                    CloseReason.MESSAGE_TOO_BIG,
                                    "websocket inbox capacity exhausted",
                                )
                                return
                            text.clear()
                            binary.clear()
                            size = 0
                data = b""
        except Exception:
            logger.exception("websocket reader failed")
            await self.close(CloseReason.INTERNAL_ERROR, "websocket reader failed")
        finally:
            if not self.closing:
                self.stream.finish()
                self.inbox.close()

    async def run(self, app: ASGIApp, scope: Scope, request: EndpointRequest) -> None:
        self.reader = asyncio.create_task(self._read())
        try:
            await app(scope, self.receive, self.send)
        except Exception:
            logger.exception("websocket application failed: %s", request.request_id)
            await self.close(CloseReason.INTERNAL_ERROR, "application failed")
        else:
            await self.close(CloseReason.NORMAL_CLOSURE, "")

    async def cleanup(self) -> None:
        tasks = [task for task in (self.reader, self._close_task) if task is not None]
        for task in tasks:
            task.cancel()
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self.inbox.close()
