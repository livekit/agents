"""Agent HTTP endpoints data plane, worker side.

The server exposes ``server.http`` routes at ``/agents/{deployment}/{path}``
by opening capsule streams over worker-dialed data connections. Each stream
carries one opaque HTTP/1.1 exchange, bridged here into the FastAPI app
IN-PROCESS (h11 -> ASGI): the websocket is the only way to execute a request,
no local listener serves user routes.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import urllib.parse
from collections.abc import Iterable, MutableMapping
from typing import TYPE_CHECKING, Any

import aiohttp
import h11
from starlette.routing import Mount, Route, WebSocketRoute

from livekit.protocol import agent

from .log import logger

if TYPE_CHECKING:
    from fastapi import FastAPI


@dataclasses.dataclass
class EndpointOptions:
    """Per-route policy carried in the manifest."""

    public: bool = False


# routes served by the health listener, never part of the manifest
_RESERVED_PATHS = {"/", "/worker"}

# FastAPI's auto-generated documentation routes: not exposed through the tunnel
# unless explicitly configured with configure_endpoint
_DOC_PATHS = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}


def build_manifest(
    app: FastAPI, options: dict[str, EndpointOptions]
) -> list[agent.AgentHttp.AgentEndpoint]:
    """Derive the endpoint manifest from FastAPI route introspection.

    APIRouter prefixes are already flattened on ``app.routes``; methods travel
    exactly as FastAPI routes them (FastAPI does not imply HEAD from GET); route
    order is preserved (the server picks the first full path+method match in
    manifest order, like starlette).
    """
    manifest: list[agent.AgentHttp.AgentEndpoint] = []
    for route in app.routes:
        if isinstance(route, Mount):
            raise ValueError(
                f"mounted sub-applications are not supported over the tunnel: {route.path!r}"
            )
        if isinstance(route, WebSocketRoute):
            logger.warning(
                "websocket routes are not tunneled yet, skipping", extra={"path": route.path}
            )
            continue
        if not isinstance(route, Route):
            continue
        if route.path in _RESERVED_PATHS:
            logger.warning(
                "route path is reserved for the local health listener and will not be served",
                extra={"path": route.path},
            )
            continue
        if route.path in _DOC_PATHS and route.path not in options:
            continue

        opts = options.get(route.path, EndpointOptions())
        ep = agent.AgentHttp.AgentEndpoint(
            path=route.path,
            methods=sorted(route.methods or set()),
            kind=agent.AgentHttp.AgentEndpointKind.AEK_HTTP,
            public=opts.public,
        )
        manifest.append(ep)
    return manifest


class TunnelClient:
    """One registration epoch's data plane: the fixed pool of attached data
    connections, each multiplexing capsule streams bridged into the ASGI app."""

    def __init__(
        self,
        *,
        http_session: aiohttp.ClientSession,
        agent_url: str,
        auth_headers: dict[str, str],
        worker_id: str,
        instance_id: str,
        settings: agent.AgentHttp.AgentEndpointSettings,
        app: FastAPI,
        http_proxy: str | None = None,
    ) -> None:
        self._http_session = http_session
        self._agent_url = agent_url
        self._auth_headers = auth_headers
        self._worker_id = worker_id
        self._instance_id = instance_id
        self._settings = settings
        self._app = app
        self._http_proxy = http_proxy
        self._conns: list[_DataConn] = []
        self._closed = False

    async def start(self) -> None:
        try:
            for _ in range(self._settings.data_connection_count):
                conn = _DataConn(self)
                await conn.connect()
                self._conns.append(conn)
        except BaseException:
            # a partially-built pool must not leak its connected sockets
            await self.aclose()
            raise
        logger.info(
            "agent endpoints attached",
            extra={"data_connections": len(self._conns)},
        )

    async def aclose(self) -> None:
        self._closed = True
        await asyncio.gather(*(c.aclose() for c in self._conns), return_exceptions=True)
        self._conns.clear()


_ATTACH_RETRIES = 5
_ATTACH_RETRY_DELAY = 0.25


class _DataConn:
    def __init__(self, client: TunnelClient) -> None:
        self._c = client
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._streams: dict[int, _StreamBridge] = {}
        self._write_lock = asyncio.Lock()
        self._read_task: asyncio.Task[None] | None = None
        self._redial_task: asyncio.Task[None] | None = None
        self._closed = False
        # per-wire flow-control parameters, set by the attach response
        self._params = agent.AgentHttp.AttachDataConnectionResponse()
        # shared connection-level send window (stream 0 credit refills it);
        # _send_ev wakes every sender blocked on stream or connection credit
        self._conn_send_credit = 0
        self._conn_recv_unacked = 0
        self._send_ev = asyncio.Event()
        self._bg_tasks: set[asyncio.Task[None]] = set()

    def conn_consumed(self, n: int) -> None:
        """Replenish the shared receive window once request bytes are consumed
        (or provably never will be), threshold-acked on stream 0."""
        if n <= 0:
            return
        self._conn_recv_unacked += n
        if self._conn_recv_unacked >= int(self._params.connection_window) // 2:
            credit = self._conn_recv_unacked
            self._conn_recv_unacked = 0
            f = agent.AgentHttp.Frame()
            f.stream_id = 0
            f.credit = credit
            task = asyncio.create_task(self._send_quiet(f))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)

    async def connect(self) -> None:
        # registration completes on the server just after the register response
        # reaches the worker, so an eager attach can land before it (or on a
        # node still unaware of the epoch): retry briefly before giving up
        delay = _ATTACH_RETRY_DELAY
        for attempt in range(_ATTACH_RETRIES):
            try:
                await self._connect_once()
                return
            except Exception:
                if self._closed or self._c._closed or attempt == _ATTACH_RETRIES - 1:
                    raise
                logger.warning(
                    "data connection attach failed, retrying",
                    exc_info=True,
                    extra={"attempt": attempt},
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 2.0)

    async def _connect_once(self) -> None:
        ws = await self._c._http_session.ws_connect(
            self._c._agent_url,
            headers=self._c._auth_headers,
            params={"attach": "1"},
            autoping=True,
            proxy=self._c._http_proxy or None,
            heartbeat=30.0,
        )
        try:
            req = agent.AgentHttp.Frame()
            req.attach.worker_id = self._c._worker_id
            req.attach.instance_id = self._c._instance_id
            req.attach.attach_token = self._c._settings.attach_token
            await ws.send_bytes(req.SerializeToString())

            resp_b = await ws.receive_bytes()
            resp = agent.AgentHttp.Frame()
            resp.ParseFromString(resp_b)
            if not resp.HasField("attach_response"):
                raise RuntimeError("expected attach response")
            if resp.attach_response.error:
                raise RuntimeError(f"attach rejected: {resp.attach_response.error}")
        except BaseException:
            await ws.close()
            raise

        p = resp.attach_response
        if not (
            p.credit_window and p.connection_window and p.max_frame_size and p.max_streams_per_conn
        ):
            raise RuntimeError("attach response missing wire parameters")
        # the adopting node's wire parameters govern this wire
        self._params = p
        self._conn_send_credit = int(self._params.connection_window)
        self._conn_recv_unacked = 0
        self._ws = ws
        self._read_task = asyncio.create_task(self._read_loop())

    async def _redial(self) -> None:
        # the attach token stays valid for the whole registration epoch, so a
        # dropped data connection redials until the tunnel itself closes
        delay = _ATTACH_RETRY_DELAY
        while not self._closed and not self._c._closed:
            try:
                await self._connect_once()
                logger.info("data connection re-attached")
                return
            except Exception:
                logger.warning("data connection redial failed, retrying", exc_info=True)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)

    async def aclose(self) -> None:
        self._closed = True
        if self._redial_task is not None:
            self._redial_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._redial_task
        for bridge in list(self._streams.values()):
            bridge.abort(ConnectionError("data connection closed"))
        self._streams.clear()
        if self._ws is not None:
            await self._ws.close()
        if self._read_task is not None:
            self._read_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._read_task

    async def send(self, frame: agent.AgentHttp.Frame) -> None:
        assert self._ws is not None
        async with self._write_lock:
            await self._ws.send_bytes(frame.SerializeToString())

    async def _send_quiet(self, frame: agent.AgentHttp.Frame) -> None:
        # a credit for a closing wire is moot; never surface the send failure
        with contextlib.suppress(Exception):
            await self.send(frame)

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            while True:
                wsmsg = await self._ws.receive()
                if wsmsg.type != aiohttp.WSMsgType.BINARY:
                    if wsmsg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        break
                    continue
                frame = agent.AgentHttp.Frame()
                frame.ParseFromString(wsmsg.data)
                sid = frame.stream_id
                which = frame.WhichOneof("message")
                if which == "open":
                    self._handle_open(sid, frame.open)
                elif which == "data":
                    bridge = self._streams.get(sid)
                    if bridge is not None:
                        bridge.on_data(frame.data)
                    else:
                        # unknown stream: ignored by protocol contract, but the
                        # bytes consumed the shared window and must come back
                        self.conn_consumed(len(frame.data))
                elif which == "eof":
                    bridge = self._streams.get(sid)
                    if bridge is not None:
                        bridge.on_eof()
                elif which == "reset":
                    bridge = self._streams.pop(sid, None)
                    if bridge is not None:
                        bridge.abort(ConnectionResetError(frame.reset.error))
                elif which == "credit":
                    if sid == 0:
                        self._conn_send_credit += int(frame.credit)
                        self._send_ev.set()
                    else:
                        bridge = self._streams.get(sid)
                        if bridge is not None:
                            bridge.on_credit(frame.credit)
                else:
                    logger.error(
                        "unexpected frame on data connection, closing",
                        extra={"frame": which},
                    )
                    break
        except Exception:
            logger.exception("data connection read loop failed")
        finally:
            if not self._closed:
                for bridge in list(self._streams.values()):
                    bridge.abort(ConnectionError("data connection lost"))
                self._streams.clear()
                if not self._c._closed:
                    self._redial_task = asyncio.create_task(self._redial())

    def _handle_open(self, stream_id: int, open_msg: agent.AgentHttp.HttpStreamOpen) -> None:
        if stream_id in self._streams:
            asyncio.create_task(
                self.send(
                    _reset_frame(
                        stream_id,
                        agent.AgentHttp.HttpStreamResetCode.HSR_PROTOCOL,
                        "duplicate stream id",
                    )
                )
            )
            return
        if len(self._streams) >= self._params.max_streams_per_conn:
            asyncio.create_task(
                self.send(_reset_frame(stream_id, agent.AgentHttp.HttpStreamResetCode.HSR_REFUSED))
            )
            return
        bridge = _StreamBridge(self, stream_id, open_msg.client_addr)
        self._streams[stream_id] = bridge
        bridge.start()

    def stream_done(self, stream_id: int) -> None:
        self._streams.pop(stream_id, None)


def _reset_frame(
    stream_id: int, code: agent.AgentHttp.HttpStreamResetCode, error: str = ""
) -> agent.AgentHttp.Frame:
    f = agent.AgentHttp.Frame()
    f.stream_id = stream_id
    f.reset.code = code
    f.reset.error = error
    return f


class _StreamBridge:
    """One capsule stream = one HTTP exchange, parsed with h11 and dispatched to
    the ASGI app in-process.

    Backpressure both ways: request bytes are credited back only after the app
    consumed them; response bytes wait for server credit before leaving."""

    def __init__(self, conn: _DataConn, stream_id: int, client_addr: str) -> None:
        self._conn = conn
        self._id = stream_id
        self._client_addr = client_addr
        self._h11 = h11.Connection(h11.SERVER)

        self._inbox: asyncio.Queue[bytes | None] = asyncio.Queue()  # bounded by credit window
        # the server debits its windows by raw payload bytes sent (request head
        # and body framing included), so replenishment must count the same raw
        # bytes, not the decoded body
        self._raw_fed = 0
        self._raw_credited = 0
        # everything that ever arrived for this stream; the shared connection
        # window is settled against it at teardown
        self._raw_received = 0
        self._conn_credited = 0
        self._done = False
        self._send_credit = int(conn._params.credit_window)
        self._aborted: BaseException | None = None
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    # --- capsule delivery (called from the connection read loop; never blocks,
    # the credit window bounds what can pile up here) ---

    def on_data(self, payload: bytes) -> None:
        if self._done:
            # torn down: the payload will never be consumed
            self._conn.conn_consumed(len(payload))
            return
        self._raw_received += len(payload)
        self._inbox.put_nowait(payload)

    def on_eof(self) -> None:
        self._inbox.put_nowait(None)

    def on_credit(self, increment: int) -> None:
        self._send_credit += increment
        self._conn._send_ev.set()

    def abort(self, exc: BaseException) -> None:
        self._aborted = exc
        self._conn._send_ev.set()
        self._inbox.put_nowait(None)
        # settle here as well: a task cancelled before its first step never
        # enters _run, so its finally alone cannot be trusted with the window
        self._settle()
        if self._task is not None:
            self._task.cancel()

    def _settle(self) -> None:
        """Exactly-once release of everything received but never credited, so a
        torn-down stream cannot leak the wire's shared receive window."""
        if self._done:
            return
        self._done = True
        self._conn.conn_consumed(self._raw_received - self._conn_credited)
        self._conn_credited = self._raw_received

    # --- exchange execution ---

    async def _run(self) -> None:
        try:
            await self._serve_exchange()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("endpoint exchange failed")
            with contextlib.suppress(Exception):
                await self._conn.send(
                    _reset_frame(self._id, agent.AgentHttp.HttpStreamResetCode.HSR_INTERNAL, str(e))
                )
        finally:
            self._settle()
            self._conn.stream_done(self._id)

    async def _next_h11_event(self) -> Any:
        while True:
            event = self._h11.next_event()
            if event is h11.NEED_DATA:
                chunk = await self._inbox.get()
                if self._aborted is not None:
                    raise self._aborted
                if chunk is None:
                    self._h11.receive_data(b"")  # EOF
                else:
                    self._raw_fed += len(chunk)
                    self._h11.receive_data(chunk)
                continue
            return event

    async def _serve_exchange(self) -> None:
        request = await self._next_h11_event()
        if not isinstance(request, h11.Request):
            await self._conn.send(
                _reset_frame(
                    self._id, agent.AgentHttp.HttpStreamResetCode.HSR_PROTOCOL, "expected request"
                )
            )
            return

        raw_path, _, query = request.target.partition(b"?")
        headers = [(k, v) for k, v in request.headers]
        # starlette answers HEAD with the GET body; the transport strips it
        # (h11 frames responses to HEAD as zero-length regardless of headers)
        is_head = request.method == b"HEAD"

        # the request head was consumed by the parse above: credit it now so a
        # huge head can never exhaust the window before the body flows
        await self._flush_credit()

        body_queue: asyncio.Queue[tuple[bytes | None, bool]] = asyncio.Queue(maxsize=4)

        async def feed_request_body() -> None:
            # h11 events -> ASGI receive, crediting raw bytes back as the app
            # consumes them (the put blocks while the app is behind)
            try:
                while True:
                    event = await self._next_h11_event()
                    if isinstance(event, h11.Data):
                        await body_queue.put((bytes(event.data), True))
                        if (
                            self._raw_fed - self._raw_credited
                            >= int(self._conn._params.credit_window) // 2
                        ):
                            await self._flush_credit()
                    elif isinstance(event, h11.EndOfMessage):
                        await body_queue.put((b"", False))
                        await self._flush_credit()
                        return
                    elif event is h11.PAUSED or isinstance(event, h11.ConnectionClosed):
                        await body_queue.put((b"", False))
                        await self._flush_credit()
                        return
            except BaseException as exc:
                # the app may be blocked in receive(): mark the exchange dead
                # and wake it, or a malformed request body hangs it forever
                if self._aborted is None:
                    self._aborted = exc
                with contextlib.suppress(asyncio.QueueFull):
                    body_queue.put_nowait((None, False))
                raise

        feeder = asyncio.create_task(feed_request_body())

        disconnected = False

        async def receive() -> MutableMapping[str, Any]:
            nonlocal disconnected
            if self._aborted is not None or disconnected:
                disconnected = True
                return {"type": "http.disconnect"}
            body, more = await body_queue.get()
            if body is None:
                disconnected = True
                return {"type": "http.disconnect"}
            return {"type": "http.request", "body": body, "more_body": more}

        response_started = False

        async def send(message: MutableMapping[str, Any]) -> None:
            nonlocal response_started
            if self._aborted is not None:
                raise ConnectionError("stream aborted") from self._aborted
            if message["type"] == "http.response.start":
                response_started = True
                out_headers = [(bytes(k), bytes(v)) for k, v in message.get("headers", [])]
                names = {k.lower() for k, _ in out_headers}
                # frame with chunked when the app gave no content-length so a
                # dying stream is a detectable truncation, never a clean EOF
                if b"content-length" not in names and b"transfer-encoding" not in names:
                    out_headers.append((b"transfer-encoding", b"chunked"))
                data = self._h11.send(
                    h11.Response(status_code=message["status"], headers=out_headers)
                )
                await self._send_payload(data)
            elif message["type"] == "http.response.body":
                body = b"" if is_head else message.get("body", b"")
                if body:
                    data = self._h11.send(h11.Data(data=body))
                    await self._send_payload(data)
                if not message.get("more_body", False):
                    data = self._h11.send(h11.EndOfMessage())
                    if data:
                        await self._send_payload(data)
                    eof = agent.AgentHttp.Frame()
                    eof.stream_id = self._id
                    eof.eof.SetInParent()
                    await self._conn.send(eof)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": request.http_version.decode(),
            "method": request.method.decode(),
            "scheme": "http",
            "path": urllib.parse.unquote(raw_path.decode()),
            "raw_path": raw_path,
            "query_string": query,
            "root_path": "",
            "headers": headers,
            "client": _split_addr(self._client_addr),
            "server": None,
        }

        try:
            await self._conn._c._app(scope, receive, send)
            if not response_started:
                raise RuntimeError("ASGI app returned without a response")
        finally:
            feeder.cancel()
            with contextlib.suppress(BaseException):
                await feeder

    async def _flush_credit(self) -> None:
        if self._done:
            return
        pending = self._raw_fed - self._raw_credited
        if pending <= 0:
            return
        self._raw_credited += pending
        self._conn_credited += pending
        self._conn.conn_consumed(pending)
        f = agent.AgentHttp.Frame()
        f.stream_id = self._id
        f.credit = pending
        await self._conn.send(f)

    async def _send_payload(self, data: bytes) -> None:
        conn = self._conn
        max_frame = int(conn._params.max_frame_size)
        view = memoryview(data)
        while len(view) > 0:
            while self._send_credit <= 0 or conn._conn_send_credit <= 0:
                if self._aborted is not None:
                    raise ConnectionError("stream aborted") from self._aborted
                conn._send_ev.clear()
                if self._send_credit > 0 and conn._conn_send_credit > 0:
                    break
                await conn._send_ev.wait()
            n = min(len(view), self._send_credit, conn._conn_send_credit, max_frame)
            # reserve before awaiting: send() yields the loop and a sibling
            # bridge must not double-spend the shared window
            self._send_credit -= n
            conn._conn_send_credit -= n
            f = agent.AgentHttp.Frame()
            f.stream_id = self._id
            f.data = bytes(view[:n])
            await conn.send(f)
            view = view[n:]


def _split_addr(addr: str) -> Iterable[Any] | None:
    host, _, port = addr.rpartition(":")
    if not host:
        return None
    try:
        return [host.strip("[]"), int(port)]
    except ValueError:
        return None
