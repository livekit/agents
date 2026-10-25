"""``AgentServer.http``: the ASGI app behind an agent's HTTP endpoints, and the bridge that serves
it on the endpoint data plane.

Routes are served through the LiveKit server at ``/agents/{agent_name}/{deployment}/{path}``;
the only local listener is the loopback one (:func:`serve_loopback`), for the worker's own jobs. The Rust worker (``_core.Worker``) accepts one stream per request,
parses the node's preamble and hands the stream here: :func:`serve_endpoint` runs one ASGI call
over it, an h11 codec for an HTTP exchange, a wsproto codec after a WebSocket upgrade. Nothing is
multiplexed in Python.

The handler sees the preamble through ``request.state.lk_request`` (:class:`EndpointRequest`):
the request id, whether the caller holds the call permission, the manifest route that matched,
and whether the route is public (``configure_endpoint(path, public=True)``).
The project credential never reaches the worker: the node strips it and asserts ``authorized``
here; a thread grant reaches a public route as its bearer, for the handler to verify.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import urllib.parse
from collections.abc import Awaitable, Callable, Iterable, MutableMapping
from typing import Any, Protocol

import h11
import wsproto
import wsproto.events as ws_events
from starlette.applications import Starlette
from starlette.routing import BaseRoute, Mount, Route, WebSocketRoute
from starlette.types import ASGIApp

from .log import logger

# `AgentHttp.HttpStreamResetCode` (livekit_agent.proto): the QUIC code a stream is reset with
# before any HTTP bytes flowed. After bytes flowed an outcome travels in trailers instead.
HSR_ABORT = 0
HSR_REFUSED = 1
HSR_INTERNAL = 2
HSR_TIMEOUT = 3
HSR_PROTOCOL = 4

# `wire.ReservedHeaderPrefix`: the in-band signalling the node strips in both directions.
_COMPLETION_TRAILER = b"x-lk-completion"
_ERROR_TRAILER = b"x-lk-error"

_ALL_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"]
_READ_SIZE = 64 * 1024
# `wire.MaxRequestHeadSize`; the node enforces it, this only bounds what a bad peer makes us hold
_MAX_REQUEST_HEAD = 1 << 20


class EndpointStreamProtocol(Protocol):
    """What the bridge needs of a stream: ``_core.EndpointStream`` in production, a fake in tests."""

    def read(self, max_bytes: int = ...) -> Awaitable[bytes]: ...
    def write(self, data: bytes) -> Awaitable[None]: ...
    def finish(self) -> None: ...
    def reset(self, code: int) -> None: ...
    def stop(self, code: int) -> None: ...


@dataclasses.dataclass(frozen=True)
class EndpointRequest:
    """The node's preamble for one request, as ``request.state.lk_request``."""

    request_id: str
    authorized: bool
    """The caller holds agent-endpoint call permission for this agent and deployment. Always true
    on a route that is not public."""
    route: str
    """The manifest template that matched, e.g. ``/items/{id}``."""
    kind: str
    """``"http"``, or ``"text"`` for a session endpoint."""
    client_addr: str
    scheme: str
    timeout_ms: int


@dataclasses.dataclass
class EndpointOptions:
    """Per-route policy carried in the manifest."""

    public: bool = False
    """Reachable without a project token."""


class AgentHttp:
    """The routes an agent serves, as a Starlette application.

    ::

        @server.http.get("/token")
        async def token(request: Request) -> Response:
            return JSONResponse({"ok": True})

        @server.http.websocket("/echo")
        async def echo(ws: WebSocket) -> None:
            await ws.accept()
            ...

        server.http.mount("/api", fastapi_app)

    Handlers are Starlette endpoints: a request in, a response out. Any ASGI application mounts.
    """

    def __init__(self) -> None:
        self._routes: list[BaseRoute] = []
        self._app: ASGIApp | None = None
        self._options: dict[str, EndpointOptions] = {}
        self._websockets: set[_WebSocketBridge] = set()

    # --- registration -------------------------------------------------------------------------

    def route(
        self, path: str, *, methods: Iterable[str], name: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_route(path, func, methods=list(methods), name=name)
            return func

        return decorator

    def get(self, path: str, **kw: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.route(path, methods=["GET"], **kw)

    def post(self, path: str, **kw: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.route(path, methods=["POST"], **kw)

    def put(self, path: str, **kw: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.route(path, methods=["PUT"], **kw)

    def patch(self, path: str, **kw: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.route(path, methods=["PATCH"], **kw)

    def delete(self, path: str, **kw: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.route(path, methods=["DELETE"], **kw)

    def websocket(
        self, path: str, *, name: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_websocket_route(path, func, name=name)
            return func

        return decorator

    def add_route(
        self,
        path: str,
        endpoint: Callable[..., Any],
        *,
        methods: list[str],
        name: str | None = None,
    ) -> None:
        self._add(Route(path, endpoint, methods=methods, name=name))

    def add_websocket_route(
        self, path: str, endpoint: Callable[..., Any], *, name: str | None = None
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

    async def serve(self, args: dict[str, Any]) -> None:
        """One ``open("endpoint", args)`` call from the worker: serve the stream to completion."""
        await serve_endpoint(args.pop("stream"), _request_from_args(args), self.app, self)

    async def close_websockets(self, code: int = 1001, reason: str = "worker draining") -> None:
        """Send every open WebSocket a close frame, so clients reconnect to another worker."""
        bridges = list(self._websockets)
        await asyncio.gather(*(b.close(code, reason) for b in bridges), return_exceptions=True)


def _request_from_args(args: dict[str, Any]) -> EndpointRequest:
    return EndpointRequest(
        request_id=args.get("request_id", ""),
        authorized=bool(args.get("authorized", False)),
        route=args.get("route", ""),
        kind=args.get("kind", "http"),
        client_addr=args.get("client_addr", ""),
        scheme=args.get("scheme") or "http",
        timeout_ms=int(args.get("timeout_ms") or 0),
    )


async def serve_endpoint(
    stream: EndpointStreamProtocol,
    request: EndpointRequest,
    app: ASGIApp,
    registry: AgentHttp | None = None,
) -> None:
    """Serve one exchange stream, positioned after its preamble, with ``app``.

    The request head decides the codec: a WebSocket upgrade runs the wsproto bridge, anything
    else the h11 one. Every outcome is reported the way the data plane expects: a reset code
    before any response byte, trailers after, FIN on success.
    """
    head = _HeadReader(stream)
    try:
        upgrade = await head.is_websocket_upgrade()
    except _ProtocolError as e:
        logger.warning(
            "endpoint request head is not HTTP/1.1",
            extra={"requestId": request.request_id, "error": str(e)},
        )
        stream.reset(HSR_PROTOCOL)
        return
    except Exception:
        # the peer went away before a head arrived; nothing to answer
        stream.reset(HSR_ABORT)
        return
    if upgrade:
        bridge = _WebSocketBridge(stream, request, app, head.buffered, registry)
        await bridge.run()
    else:
        await _HttpBridge(stream, request, app, head.buffered).run()


LOCAL_ROUTES_ENV = "LIVEKIT_AGENT_LOCAL_ROUTES"
"""``<public prefix>=<loopback base>``, comma separated: the worker's own routes, which a job of
the worker reaches over the loopback listener instead of through the node."""


class _TcpStream:
    """A loopback connection as the stream ``serve_endpoint`` reads and writes."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader = reader
        self._writer = writer

    async def read(self, max_bytes: int = _READ_SIZE) -> bytes:
        return await self._reader.read(max_bytes)

    async def write(self, data: bytes) -> None:
        self._writer.write(data)
        await self._writer.drain()

    def finish(self) -> None:
        self._writer.close()

    def reset(self, code: int) -> None:
        self._writer.transport.abort()

    def stop(self, code: int) -> None:
        self._writer.transport.abort()


async def serve_loopback(app: ASGIApp) -> tuple[asyncio.Server, str]:
    """Serves ``app`` on 127.0.0.1, one exchange per connection, with the bridge the node's
    streams use: responses stream as written, an SSE answer included. Returns the server and
    its base URL. A request authenticates with its own token, as no node vouches for it."""

    async def connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        request = EndpointRequest(
            request_id=f"loopback-{id(writer):x}",
            authorized=False,
            route="",
            kind="http",
            client_addr=f"{peer[0]}:{peer[1]}" if peer else "",
            scheme="http",
            timeout_ms=0,
        )
        try:
            await serve_endpoint(_TcpStream(reader, writer), request, app)
        finally:
            writer.close()

    server = await asyncio.start_server(connection, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"http://127.0.0.1:{port}"


class _PeerGone(Exception):
    """The client reset the stream or stopped reading before the response ended."""


class _ProtocolError(Exception):
    pass


class _HeadReader:
    """Reads enough of the request to tell a WebSocket upgrade from an HTTP exchange, keeping what
    it read for the codec that follows."""

    def __init__(self, stream: EndpointStreamProtocol) -> None:
        self._stream = stream
        self.buffered = b""

    async def is_websocket_upgrade(self) -> bool:
        while b"\r\n\r\n" not in self.buffered:
            if len(self.buffered) > _MAX_REQUEST_HEAD:
                raise _ProtocolError("request head exceeds 1 MiB")
            chunk = await self._stream.read(_READ_SIZE)
            if not chunk:
                raise _ProtocolError("stream ended before the request head")
            self.buffered += chunk
        head = self.buffered[: self.buffered.index(b"\r\n\r\n")]
        lines = head.split(b"\r\n")
        if not lines or not lines[0].upper().startswith(b"GET "):
            return False
        upgrade = connection = False
        for line in lines[1:]:
            name, _, value = line.partition(b":")
            name = name.strip().lower()
            tokens = [t.strip().lower() for t in value.split(b",")]
            if name == b"upgrade" and b"websocket" in tokens:
                upgrade = True
            elif name == b"connection" and b"upgrade" in tokens:
                connection = True
        return upgrade and connection


def _scope_base(
    request: EndpointRequest, headers: list[tuple[bytes, bytes]], target: bytes
) -> dict[str, Any]:
    raw_path, _, query = target.partition(b"?")
    return {
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "path": urllib.parse.unquote(raw_path.decode("latin-1")),
        "raw_path": raw_path,
        "query_string": query,
        "root_path": "",
        "headers": headers,
        "client": (request.client_addr, 0) if request.client_addr else None,
        "server": None,
        # the documented hook: `request.state.lk_request`
        "state": {"lk_request": request},
    }


class _HttpBridge:
    """One HTTP exchange: h11 events in, ASGI ``http`` events to the app, h11 bytes out."""

    def __init__(
        self,
        stream: EndpointStreamProtocol,
        request: EndpointRequest,
        app: ASGIApp,
        buffered: bytes,
    ) -> None:
        self._stream = stream
        self._request = request
        self._app = app
        self._h11 = h11.Connection(h11.SERVER)
        if buffered:
            self._h11.receive_data(buffered)
        self._response_started = False
        self._head_sent = asyncio.Event()
        self._response_ended = False
        self._request_done = False
        self._exchange_done = asyncio.Event()
        self._chunked = False
        self._is_head = False
        self._eof = False
        self._write_lock = asyncio.Lock()

    async def run(self) -> None:
        try:
            await self._run()
        finally:
            self._exchange_done.set()

    async def _run(self) -> None:
        req = self._request
        try:
            event = await self._next_event()
        except h11.RemoteProtocolError as e:
            logger.warning(
                "endpoint request is not valid HTTP/1.1",
                extra={"requestId": req.request_id, "error": str(e)},
            )
            self._stream.reset(HSR_PROTOCOL)
            return
        if not isinstance(event, h11.Request):
            self._stream.reset(HSR_PROTOCOL)
            return
        self._is_head = event.method == b"HEAD"
        scope = _scope_base(req, [(k, v) for k, v in event.headers], event.target)
        scope.update(type="http", method=event.method.decode(), scheme=req.scheme)

        app_task = asyncio.ensure_future(self._app(scope, self._receive, self._send))
        timeout = req.timeout_ms / 1000 if req.timeout_ms > 0 else None
        try:
            if timeout is None:
                await app_task
            else:
                await self._await_with_head_deadline(app_task, timeout)
        except asyncio.CancelledError:
            # the worker is stopping or reconnecting; the stream goes with it
            app_task.cancel()
            with contextlib.suppress(BaseException):
                await app_task
            await self._fail("canceled", "worker stopped", HSR_ABORT)
            raise
        except asyncio.TimeoutError:
            logger.warning(
                "endpoint request timed out before a response head",
                extra={"requestId": req.request_id, "route": req.route},
            )
            await self._fail("timeout", "no response head before the deadline", HSR_TIMEOUT)
            return
        except _PeerGone as e:
            logger.debug(
                "endpoint client went away",
                extra={"requestId": req.request_id, "route": req.route, "reason": str(e)},
            )
            return
        except Exception as e:
            logger.exception(
                "endpoint request failed", extra={"requestId": req.request_id, "route": req.route}
            )
            await self._fail("internal", f"{type(e).__name__}: {e}", HSR_INTERNAL)
            return
        if not self._response_started:
            logger.error(
                "ASGI app returned without a response",
                extra={"requestId": req.request_id, "route": req.route},
            )
            await self._fail("internal", "no response", HSR_INTERNAL)
            return
        if not self._response_ended:
            await self._fail("internal", "response body left open", HSR_INTERNAL)

    async def _await_with_head_deadline(
        self, app_task: asyncio.Future[Any], timeout: float
    ) -> None:
        # the deadline bounds the wait for a response head only; bodies are unbounded
        head = asyncio.ensure_future(self._head_sent.wait())
        try:
            done, _ = await asyncio.wait(
                {app_task, head}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            head.cancel()
        if not done:
            app_task.cancel()
            with contextlib.suppress(BaseException):
                await app_task
            raise asyncio.TimeoutError
        await app_task

    async def _fail(self, completion: str, reason: str, code: int) -> None:
        """Report an outcome other than success in the only sound place for it."""
        if self._response_ended:
            return
        if not self._response_started:
            self._stream.reset(code)
            return
        if self._chunked and not self._is_head:
            # in band, where it cannot race the body bytes
            trailers = [
                (_COMPLETION_TRAILER, completion.encode()),
                (_ERROR_TRAILER, _sanitize(reason)),
            ]
            with contextlib.suppress(Exception):
                await self._write(self._h11.send(h11.EndOfMessage(headers=trailers)))
                self._stream.finish()
            self._response_ended = True
            return
        # a declared length cannot carry trailers: the short body is the signal
        self._stream.reset(HSR_ABORT)

    # --- the request side ---------------------------------------------------------------------

    async def _next_event(self) -> Any:
        while True:
            event = self._h11.next_event()
            if event is h11.NEED_DATA:
                if self._eof:
                    return h11.ConnectionClosed()
                chunk = await self._stream.read(_READ_SIZE)
                if not chunk:
                    self._eof = True
                    self._h11.receive_data(b"")
                else:
                    self._h11.receive_data(chunk)
                continue
            return event

    async def _receive(self) -> MutableMapping[str, Any]:
        if self._request_done:
            # The node half-closes once the request is written; that is the end of the body, not
            # a disconnect. A client that goes away shows as a failed write, so an app listening
            # for a disconnect (a streaming response does) waits here until the exchange ends.
            await self._exchange_done.wait()
            return {"type": "http.disconnect"}
        try:
            event = await self._next_event()
        except Exception:
            return {"type": "http.disconnect"}
        if isinstance(event, h11.Data):
            return {"type": "http.request", "body": bytes(event.data), "more_body": True}
        if isinstance(event, h11.EndOfMessage):
            self._request_done = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    # --- the response side --------------------------------------------------------------------

    async def _send(self, message: MutableMapping[str, Any]) -> None:
        kind = message["type"]
        if kind == "http.response.start":
            if self._response_started:
                raise RuntimeError("response already started")
            headers = [(bytes(k), bytes(v)) for k, v in message.get("headers", [])]
            names = {k.lower() for k, _ in headers}
            # h11 frames an HTTP/1.1 response without a length as chunked, the one framing that
            # can carry a completion trailer, so a dying stream reads as truncation, never success
            self._chunked = b"content-length" not in names and b"transfer-encoding" not in names
            status = int(message["status"])
            self._response_started = True
            self._head_sent.set()
            data = self._h11.send(h11.Response(status_code=status, headers=headers))
            await self._write(data)
            if status in (204, 304) or self._is_head:
                self._chunked = False
        elif kind == "http.response.body":
            if not self._response_started:
                raise RuntimeError("response body before its head")
            if self._response_ended:
                return
            body = message.get("body", b"")
            if body and not self._is_head:
                await self._write(self._h11.send(h11.Data(data=body)))
            if not message.get("more_body", False):
                data = self._h11.send(h11.EndOfMessage())
                if data:
                    await self._write(data)
                self._response_ended = True
                self._stream.finish()
                # nothing more of the request is wanted (`pipe` in the conformance worker)
                self._stream.stop(HSR_ABORT)
                self._exchange_done.set()
        elif kind == "http.response.trailers":
            pass  # the x-lk- namespace is ours; applications do not send trailers here
        else:
            raise RuntimeError(f"unexpected ASGI message {kind!r} on an http scope")

    async def _write(self, data: bytes | None) -> None:
        if not data:
            return
        async with self._write_lock:
            try:
                await self._stream.write(data)
            except RuntimeError as e:
                # the peer reset the stream or stopped reading: the client is gone
                raise _PeerGone(str(e)) from None


def _sanitize(reason: str) -> bytes:
    """`wire.sanitizeReason`: bounded, no control characters that would break trailer framing."""
    cleaned = "".join(" " if ord(c) < 0x20 or ord(c) == 0x7F else c for c in reason)
    return cleaned[:256].encode("utf-8", "replace")


class _WebSocketBridge:
    """After the upgrade request the stream is a byte pipe: wsproto frames in, ASGI ``websocket``
    events to the app, wsproto bytes out. The 101 itself is wsproto's ``AcceptConnection``."""

    def __init__(
        self,
        stream: EndpointStreamProtocol,
        request: EndpointRequest,
        app: ASGIApp,
        buffered: bytes,
        registry: AgentHttp | None,
    ) -> None:
        self._stream = stream
        self._request = request
        self._app = app
        self._ws = wsproto.WSConnection(wsproto.ConnectionType.SERVER)
        self._buffered = buffered
        self._registry = registry
        self._events: asyncio.Queue[MutableMapping[str, Any]] = asyncio.Queue()
        self._accepted = False
        self._closed = False
        self._write_lock = asyncio.Lock()
        self._text = ""
        self._bytes = b""
        self._reader: asyncio.Task[None] | None = None

    async def run(self) -> None:
        req = self._request
        self._ws.receive_data(self._buffered)
        handshake = None
        for event in self._ws.events():
            if isinstance(event, ws_events.Request):
                handshake = event
                break
        if handshake is None:
            # the head looked like an upgrade but wsproto does not take it (a missing key, ...)
            self._stream.reset(HSR_PROTOCOL)
            return
        headers = [(k.lower(), v) for k, v in handshake.extra_headers]
        headers.extend(
            [
                (b"host", handshake.host.encode()),
                (b"upgrade", b"websocket"),
                (b"connection", b"upgrade"),
            ]
        )
        scope = _scope_base(req, headers, handshake.target.encode())
        scope.update(
            type="websocket",
            scheme="wss" if req.scheme == "https" else "ws",
            subprotocols=list(handshake.subprotocols),
        )
        self._events.put_nowait({"type": "websocket.connect"})
        if self._registry is not None:
            self._registry._websockets.add(self)
        self._reader = asyncio.create_task(self._read_loop())
        try:
            await self._app(scope, self._receive, self._send)
            if not self._accepted and not self._closed:
                # the app returned without deciding: ASGI servers answer 403
                await self._reject(403)
        except asyncio.CancelledError:
            await self.close(1001, "worker stopped")
            raise
        except Exception as e:
            logger.exception(
                "websocket endpoint failed", extra={"requestId": req.request_id, "route": req.route}
            )
            if not self._accepted and not self._closed:
                await self._reject(500)
            else:
                await self.close(1011, f"{type(e).__name__}: {e}")
        finally:
            if self._registry is not None:
                self._registry._websockets.discard(self)
            self._reader.cancel()
            with contextlib.suppress(BaseException):
                await self._reader
            if not self._closed:
                await self.close(1000, "")

    # --- the client side ----------------------------------------------------------------------

    async def _read_loop(self) -> None:
        try:
            while True:
                chunk = await self._stream.read(_READ_SIZE)
                if not chunk:
                    self._ws.receive_data(None)
                else:
                    self._ws.receive_data(chunk)
                for event in self._ws.events():
                    await self._on_event(event)
                if not chunk:
                    break
        except Exception:
            pass
        # the peer is gone (FIN, reset or a protocol error): tell the app once
        if not self._closed:
            self._closed = True
            self._events.put_nowait({"type": "websocket.disconnect", "code": 1006})

    async def _on_event(self, event: ws_events.Event) -> None:
        if isinstance(event, ws_events.TextMessage):
            self._text += event.data
            if event.message_finished:
                self._events.put_nowait({"type": "websocket.receive", "text": self._text})
                self._text = ""
        elif isinstance(event, ws_events.BytesMessage):
            self._bytes += event.data
            if event.message_finished:
                self._events.put_nowait({"type": "websocket.receive", "bytes": self._bytes})
                self._bytes = b""
        elif isinstance(event, ws_events.Ping):
            await self._write(self._ws.send(event.response()))
        elif isinstance(event, ws_events.CloseConnection):
            if self._ws.state is wsproto.connection.ConnectionState.REMOTE_CLOSING:
                with contextlib.suppress(Exception):
                    await self._write(self._ws.send(event.response()))
            self._closed = True
            self._events.put_nowait({"type": "websocket.disconnect", "code": event.code})
            with contextlib.suppress(Exception):
                self._stream.finish()

    async def _receive(self) -> MutableMapping[str, Any]:
        return await self._events.get()

    # --- the app side -------------------------------------------------------------------------

    async def _send(self, message: MutableMapping[str, Any]) -> None:
        kind = message["type"]
        if kind == "websocket.accept":
            if self._accepted:
                raise RuntimeError("websocket already accepted")
            self._accepted = True
            extra = [(bytes(k), bytes(v)) for k, v in message.get("headers", [])]
            await self._write(
                self._ws.send(
                    ws_events.AcceptConnection(
                        subprotocol=message.get("subprotocol"), extra_headers=extra
                    )
                )
            )
        elif kind == "websocket.send":
            if not self._accepted:
                raise RuntimeError("websocket.send before accept")
            if self._closed:
                return
            if message.get("bytes") is not None:
                await self._write(self._ws.send(ws_events.BytesMessage(data=message["bytes"])))
            else:
                await self._write(
                    self._ws.send(ws_events.TextMessage(data=message.get("text") or ""))
                )
        elif kind == "websocket.close":
            if not self._accepted:
                await self._reject(403)
            else:
                await self.close(int(message.get("code", 1000)), message.get("reason") or "")
        elif kind == "websocket.http.response.start":
            # a rejection with a body, as the ASGI extension spells it
            await self._reject(
                int(message["status"]),
                [(bytes(k), bytes(v)) for k, v in message.get("headers", [])],
            )
        elif kind == "websocket.http.response.body":
            if message.get("body"):
                await self._write(message["body"])
            if not message.get("more_body", False):
                with contextlib.suppress(Exception):
                    self._stream.finish()
        else:
            raise RuntimeError(f"unexpected ASGI message {kind!r} on a websocket scope")

    async def _reject(self, status: int, headers: list[tuple[bytes, bytes]] | None = None) -> None:
        """A declined upgrade is an ordinary HTTP response, then FIN."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            await self._write(
                self._ws.send(
                    ws_events.RejectConnection(
                        status_code=status, headers=headers or [], has_body=False
                    )
                )
            )
            self._stream.finish()

    async def close(self, code: int, reason: str) -> None:
        """A close frame to the client, then FIN. Idempotent."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            await self._write(self._ws.send(ws_events.CloseConnection(code=code, reason=reason)))
        with contextlib.suppress(Exception):
            self._stream.finish()
        self._events.put_nowait({"type": "websocket.disconnect", "code": code})

    async def _write(self, data: bytes) -> None:
        if not data:
            return
        async with self._write_lock:
            await self._stream.write(data)
