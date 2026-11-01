"""The endpoint bridge (`livekit.agents.http`) over a fake stream: what the Rust worker hands it is
an `ByteStream` after the preamble, so a fake with the same five methods exercises every path
in process: head and streamed body in, head and streamed body out, trailers for a late failure,
resets, and the byte pipe after a 101."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import h11
import pytest
import wsproto
import wsproto.events as ws_events
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.types import Message
from starlette.websockets import WebSocket

from livekit.agents.http import (
    HSR_INTERNAL,
    HSR_PROTOCOL,
    HSR_TIMEOUT,
    AgentRoutes,
    EndpointRequest,
    serve_endpoint,
)

pytestmark = pytest.mark.unit


class FakeStream:
    """An `ByteStream` in memory: the test is the node."""

    def __init__(self) -> None:
        self._inbox: asyncio.Queue[bytes] = asyncio.Queue()
        self._pending = b""
        self._eof = False
        self.out = bytearray()
        self.finished = False
        self.reset_code: int | None = None
        self.stop_code: int | None = None
        self.changed = asyncio.Event()

    def feed(self, data: bytes) -> None:
        self._inbox.put_nowait(data)

    def feed_eof(self) -> None:
        self._inbox.put_nowait(b"")

    async def read(self, max_bytes: int = 65536) -> bytes:
        if not self._pending and not self._eof:
            chunk = await self._inbox.get()
            if chunk == b"":
                self._eof = True
            self._pending += chunk
        out, self._pending = self._pending[:max_bytes], self._pending[max_bytes:]
        return out

    async def write(self, data: bytes) -> None:
        self.out += data
        self.changed.set()

    def finish(self) -> None:
        self.finished = True
        self.changed.set()

    def reset(self, code: int) -> None:
        self.reset_code = code
        self.changed.set()

    def stop(self, code: int) -> None:
        self.stop_code = code

    async def wait_done(self) -> None:
        while not (self.finished or self.reset_code is not None):
            self.changed.clear()
            await asyncio.wait_for(self.changed.wait(), 5)


class GatedCloseStream(FakeStream):
    """Pause the close reply of an application that sends only a handshake."""

    def __init__(self) -> None:
        super().__init__()
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.peer_eof = asyncio.Event()

    async def write(self, data: bytes) -> None:
        if self.out:
            self.close_started.set()
            await self.allow_close.wait()
        await super().write(data)

    async def read(self, max_bytes: int = 65536) -> bytes:
        data = await super().read(max_bytes)
        if not data:
            self.peer_eof.set()
        return data


def preamble(**kw: Any) -> EndpointRequest:
    base: dict[str, Any] = {
        "request_id": "AER_test",
        "authorized": True,
        "route": "/",
        "kind": "http",
        "client_addr": "127.0.0.1",
        "scheme": "http",
        "timeout_ms": 0,
    }
    base.update(kw)
    return EndpointRequest(**base)


def request_bytes(
    method: str, target: str, body: bytes = b"", headers: list[tuple[str, str]] | None = None
) -> bytes:
    c = h11.Connection(h11.CLIENT)
    hs = [("Host", "agents.test")] + (headers or [])
    if body:
        hs.append(("Content-Length", str(len(body))))
    data = c.send(h11.Request(method=method, target=target, headers=hs))
    if body:
        data += c.send(h11.Data(data=body))
    data += c.send(h11.EndOfMessage())
    return data


def parse_response(raw: bytes) -> tuple[h11.Response, bytes, h11.EndOfMessage]:
    c = h11.Connection(h11.CLIENT)
    c.send(h11.Request(method="GET", target="/", headers=[("Host", "x")]))
    c.send(h11.EndOfMessage())
    c.receive_data(raw)
    c.receive_data(b"")
    head = None
    body = b""
    while True:
        ev = c.next_event()
        if isinstance(ev, h11.Response):
            head = ev
        elif isinstance(ev, h11.Data):
            body += bytes(ev.data)
        elif isinstance(ev, h11.EndOfMessage):
            assert head is not None
            return head, body, ev
        elif ev is h11.NEED_DATA:
            raise AssertionError("response truncated")


def app_with_routes() -> AgentRoutes:
    http = AgentRoutes()

    @http.get("/hello")
    async def hello(request: Request) -> Response:
        info = request.state.lk_request
        return JSONResponse({"hello": "world", "route": info.route})

    @http.post("/echo-size")
    async def echo_size(request: Request) -> Response:
        n = 0
        async for chunk in request.stream():
            n += len(chunk)
        return PlainTextResponse(str(n))

    @http.get("/boom")
    async def boom(request: Request) -> Response:
        raise RuntimeError("before any head")

    @http.get("/late-boom")
    async def late_boom(request: Request) -> Response:
        async def gen():
            yield b"first chunk\n"
            await asyncio.sleep(0)
            raise RuntimeError("the source died")

        return StreamingResponse(gen(), media_type="text/plain")

    async def raw_late_boom(scope, receive, send):
        # a bare ASGI app, mounted: the head and one chunk go out, then the source dies
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"first chunk\n", "more_body": True})
        raise RuntimeError("the source died")

    http.mount("/raw", raw_late_boom)

    @http.get("/stream")
    async def stream(request: Request) -> Response:
        async def gen():
            for i in range(3):
                yield f"chunk {i}\n".encode()
                await asyncio.sleep(0)

        return StreamingResponse(gen(), media_type="text/plain")

    @http.get("/slow")
    async def slow(request: Request) -> Response:
        await asyncio.sleep(5)
        return PlainTextResponse("late")

    @http.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    return
                if msg.get("text") is not None:
                    await websocket.send_text("echo:" + msg["text"])
                else:
                    await websocket.send_bytes(bytes(reversed(msg["bytes"])))
        finally:
            pass

    http.configure_endpoint("/echo-size", public=True)
    return http


async def run_exchange(
    http: AgentRoutes, raw: bytes, info: EndpointRequest | None = None
) -> FakeStream:
    stream = FakeStream()
    stream.feed(raw)
    stream.feed_eof()
    task = asyncio.create_task(http._server._core_handle(stream, info or preamble()))
    await stream.wait_done()
    await asyncio.wait_for(task, 5)
    return stream


async def test_a_small_exchange_round_trips_and_the_preamble_reaches_the_handler() -> None:
    http = app_with_routes()
    stream = await run_exchange(http, request_bytes("GET", "/hello"), preamble(route="/hello"))
    head, body, _ = parse_response(bytes(stream.out))
    assert head.status_code == 200
    assert json.loads(body) == {"hello": "world", "route": "/hello"}
    assert stream.finished and stream.reset_code is None


async def test_the_nodes_half_close_ends_the_request_without_disconnecting_a_streaming_response() -> (
    None
):
    # the node writes the request and FINs; a StreamingResponse listens for a disconnect while it
    # streams, and that FIN must not read as one
    http = app_with_routes()
    stream = await run_exchange(http, request_bytes("GET", "/stream"), preamble(route="/stream"))
    head, body, end = parse_response(bytes(stream.out))
    assert head.status_code == 200
    assert body == b"chunk 0\nchunk 1\nchunk 2\n"
    assert not list(end.headers) and stream.finished and stream.reset_code is None


async def test_a_large_body_streams_in_and_the_response_is_chunked_without_trailers() -> None:
    http = app_with_routes()
    body = bytes(range(256)) * 16 * 1024  # 4 MiB
    stream = FakeStream()
    raw = request_bytes("POST", "/echo-size", body)
    # arrives in pieces, as a QUIC stream delivers it
    for i in range(0, len(raw), 100_000):
        stream.feed(raw[i : i + 100_000])
    stream.feed_eof()
    task = asyncio.create_task(serve_endpoint(stream, preamble(route="/echo-size"), http.app, http))
    await stream.wait_done()
    await asyncio.wait_for(task, 10)
    head, out, end = parse_response(bytes(stream.out))
    assert head.status_code == 200
    assert out == str(len(body)).encode()
    assert not list(end.headers), "a successful body announces nothing in trailers"
    assert stream.finished


async def test_a_failure_before_the_head_is_a_reset_and_a_failure_after_is_a_trailer() -> None:
    http = app_with_routes()
    # Starlette answers an unhandled exception with a 500 itself, so the head goes out and the
    # exchange ends cleanly; the bridge must not then also reset.
    stream = await run_exchange(http, request_bytes("GET", "/boom"), preamble(route="/boom"))
    head, _, _ = parse_response(bytes(stream.out))
    assert head.status_code == 500 and stream.finished and stream.reset_code is None

    for target, route in [("/late-boom", "/late-boom"), ("/raw/anything", "/raw/{path:path}")]:
        stream = await run_exchange(http, request_bytes("GET", target), preamble(route=route))
        head, body, end = parse_response(bytes(stream.out))
        assert head.status_code == 200
        assert body == b"first chunk\n"
        trailers = dict(end.headers)
        assert trailers[b"x-lk-completion"] == b"internal", target
        assert trailers[b"x-lk-error"], target
        assert stream.finished and stream.reset_code is None
    assert b"the source died" in trailers[b"x-lk-error"]

    # an app that returns without a response head: entered, so internal rather than refused
    async def silent(scope, receive, send):
        return

    stream = FakeStream()
    stream.feed(request_bytes("GET", "/x"))
    stream.feed_eof()
    await asyncio.wait_for(serve_endpoint(stream, preamble(), silent), 5)
    assert stream.reset_code == HSR_INTERNAL and not stream.out


async def test_bad_bytes_are_a_protocol_reset_and_a_missed_deadline_a_timeout_reset() -> None:
    http = app_with_routes()
    stream = FakeStream()
    stream.feed(b"this is not http\r\n\r\n")
    stream.feed_eof()
    await asyncio.wait_for(serve_endpoint(stream, preamble(), http.app, http), 5)
    assert stream.reset_code == HSR_PROTOCOL and not stream.out

    stream = FakeStream()
    stream.feed(request_bytes("GET", "/slow"))
    stream.feed_eof()
    await asyncio.wait_for(
        serve_endpoint(stream, preamble(route="/slow", timeout_ms=50), http.app, http), 5
    )
    assert stream.reset_code == HSR_TIMEOUT and not stream.out


async def test_a_websocket_upgrade_turns_the_stream_into_a_frame_pipe() -> None:
    http = app_with_routes()
    client = wsproto.WSConnection(wsproto.ConnectionType.CLIENT)
    stream = FakeStream()
    # the node forwards the client's handshake with Connection/Upgrade restated
    stream.feed(client.send(ws_events.Request(host="agents.test", target="/ws")))
    task = asyncio.create_task(serve_endpoint(stream, preamble(route="/ws"), http.app, http))

    consumed = 0

    async def next_events(n: int) -> list[ws_events.Event]:
        nonlocal consumed
        got: list[ws_events.Event] = []
        while len(got) < n:
            if consumed < len(stream.out):
                client.receive_data(bytes(stream.out[consumed:]))
                consumed = len(stream.out)
                got.extend(client.events())
                continue
            stream.changed.clear()
            await asyncio.wait_for(stream.changed.wait(), 5)
        return got

    (accepted,) = await next_events(1)
    assert isinstance(accepted, ws_events.AcceptConnection)
    assert bytes(stream.out).startswith(b"HTTP/1.1 101")

    stream.feed(client.send(ws_events.TextMessage(data="hi")))
    stream.feed(client.send(ws_events.BytesMessage(data=b"abc")))
    text, data = await next_events(2)
    assert isinstance(text, ws_events.TextMessage) and text.data == "echo:hi"
    assert isinstance(data, ws_events.BytesMessage) and data.data == b"cba"

    stream.feed(client.send(ws_events.CloseConnection(code=1000, reason="bye")))
    (closed,) = await next_events(1)
    assert isinstance(closed, ws_events.CloseConnection) and closed.code == 1000
    stream.feed_eof()
    await asyncio.wait_for(task, 5)
    assert stream.finished and stream.reset_code is None


async def test_drain_sends_an_open_websocket_a_close_frame() -> None:
    http = app_with_routes()
    client = wsproto.WSConnection(wsproto.ConnectionType.CLIENT)
    stream = FakeStream()
    stream.feed(client.send(ws_events.Request(host="agents.test", target="/ws")))
    task = asyncio.create_task(serve_endpoint(stream, preamble(route="/ws"), http.app, http))
    while not stream.out:
        stream.changed.clear()
        await asyncio.wait_for(stream.changed.wait(), 5)
    await http.close_websockets()
    stream.feed_eof()
    await asyncio.wait_for(task, 5)
    client.receive_data(bytes(stream.out))
    kinds = [type(e) for e in client.events()]
    assert kinds == [ws_events.AcceptConnection, ws_events.CloseConnection]
    assert stream.finished


def test_the_manifest_carries_methods_kinds_and_public() -> None:
    from livekit.agents._generated.worker_server import EndpointRoute

    http = app_with_routes()
    by_path = {e.path: e for e in http.manifest()}
    assert by_path["/hello"] == EndpointRoute(
        path="/hello", methods=["GET", "HEAD"], kind="http", public=False
    )
    assert by_path["/echo-size"].methods == ["POST"] and by_path["/echo-size"].public
    assert by_path["/ws"] == EndpointRoute(path="/ws", methods=["GET"], kind="http", public=False)
    assert by_path["/raw/{path:path}"].methods == [
        "GET",
        "HEAD",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
        "TRACE",
    ]
    assert [e.path for e in http.manifest()][0] == "/hello", "route order is preserved"


async def test_scope_preserves_encoded_path_and_query() -> None:
    seen = {}

    async def app(scope, receive, send):
        seen.update(scope)
        await send({"type": "http.response.start", "status": 204})
        await send({"type": "http.response.body"})

    stream = FakeStream()
    stream.feed(request_bytes("GET", "/a%20b/%E2%82%AC?q=%20"))
    await serve_endpoint(stream, preamble(), app)
    assert seen["path"] == "/a b/€"
    assert seen["raw_path"] == b"/a%20b/%E2%82%AC"
    assert seen["query_string"] == b"q=%20"
    assert seen["client"] is None  # No port was supplied by the node.
    assert seen["state"]["lk_request"].client_addr == "127.0.0.1"
    assert stream.finished


async def test_chunked_request_body_is_decoded_in_python() -> None:
    stream = await run_exchange(
        app_with_routes(),
        b"POST /echo-size HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"
        b"3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n",
    )
    assert parse_response(bytes(stream.out))[1] == b"5"


@pytest.mark.parametrize("tail", [b"1;" + b"x" * (1 << 20), b"0\r\nX: " + b"x" * (1 << 20)])
async def test_oversized_chunk_metadata_is_a_protocol_reset(tail: bytes) -> None:
    stream = await run_exchange(
        app_with_routes(),
        b"POST /echo-size HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n" + tail,
    )
    assert stream.reset_code == HSR_PROTOCOL


async def test_header_limit_applies_before_the_application_runs() -> None:
    called = False

    async def app(scope, receive, send):
        nonlocal called
        called = True

    stream = FakeStream()
    stream.feed(b"GET / HTTP/1.1\r\nHost: x\r\nX: " + b"x" * (1 << 20) + b"\r\n\r\n")
    await serve_endpoint(stream, preamble(), app)
    assert stream.reset_code == HSR_PROTOCOL and not called


async def test_cancel_releases_admission_and_resets_the_stream() -> None:
    from livekit.agents.http import HSR_ABORT, HSR_REFUSED, _AsgiServer, _HttpServerLimits

    app = app_with_routes().app
    server = _AsgiServer(app, _HttpServerLimits(pending_handlers=1))
    stalled = FakeStream()
    task = asyncio.create_task(server._core_handle(stalled, preamble()))
    await asyncio.sleep(0)
    refused = FakeStream()
    await server._core_handle(refused, preamble())
    assert refused.reset_code == HSR_REFUSED
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stalled.reset_code == HSR_ABORT
    assert server.handlers == server.payload_bytes == 0

    accepted = FakeStream()
    accepted.feed(request_bytes("GET", "/hello"))
    await server._core_handle(accepted, preamble())
    assert accepted.finished and accepted.reset_code is None


async def test_cancellation_after_response_start_reports_completion_trailers() -> None:
    from livekit.agents.http import _AsgiServer

    started = asyncio.Event()

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"partial", "more_body": True})
        started.set()
        await asyncio.Future()

    server = _AsgiServer(app)
    stream = FakeStream()
    stream.feed(request_bytes("GET", "/"))
    task = asyncio.create_task(server._core_handle(stream, preamble()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    _, body, end = parse_response(bytes(stream.out))
    assert body == b"partial" and dict(end.headers)[b"x-lk-completion"] == b"canceled"
    assert server.handlers == server.payload_bytes == 0


async def test_deadline_stops_at_response_head_and_does_not_cancel_streaming_body() -> None:
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        await asyncio.sleep(0.03)
        await send({"type": "http.response.body", "body": b"late body"})

    stream = FakeStream()
    stream.feed(request_bytes("GET", "/"))
    await serve_endpoint(stream, preamble(timeout_ms=5), app)
    assert parse_response(bytes(stream.out))[1] == b"late body"
    assert stream.reset_code is None


@pytest.mark.parametrize("flood", [False, True])
async def test_websocket_limits_close_and_release_payloads(flood: bool) -> None:
    from livekit.agents.http import _AsgiServer, _HttpServerLimits

    client = wsproto.WSConnection(wsproto.ConnectionType.CLIENT)
    stream = FakeStream()
    stream.feed(client.send(ws_events.Request(host="agents.test", target="/ws")))
    if flood:
        stream.feed(b"\x82\x81\x00\x00\x00\x00a" * 2)
    else:
        # A declared 9-byte frame exceeds the limit even without sending a payload.
        stream.feed(b"\x82\x89\x00\x00\x00\x00")

    async def app(scope, receive, send):
        assert (await receive())["type"] == "websocket.connect"
        await send({"type": "websocket.accept"})
        await stream.wait_done()

    server = _AsgiServer(app, _HttpServerLimits(websocket_inbox=1, websocket_frame_bytes=8))
    await asyncio.wait_for(server._core_handle(stream, preamble()), 5)
    client.receive_data(bytes(stream.out))
    events = list(client.events())
    assert isinstance(events[-1], ws_events.CloseConnection) and events[-1].code == 1009
    assert stream.finished and server.handlers == server.payload_bytes == 0


async def test_websocket_rejection_is_an_http_response() -> None:
    client = wsproto.WSConnection(wsproto.ConnectionType.CLIENT)
    stream = FakeStream()
    stream.feed(client.send(ws_events.Request(host="agents.test", target="/ws")))

    async def app(scope, receive, send):
        await send({"type": "websocket.close"})

    await serve_endpoint(stream, preamble(), app)
    assert bytes(stream.out).startswith(b"HTTP/1.1 403")
    assert stream.finished and stream.reset_code is None


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("cancel_after_yields", range(7))
async def test_cancelled_websocket_receive_preserves_delivery_order(
    binary: bool, cancel_after_yields: int
) -> None:
    from livekit.agents.http import _AsgiServer

    client = wsproto.WSConnection(wsproto.ConnectionType.CLIENT)
    stream = FakeStream()
    stream.feed(client.send(ws_events.Request(host="agents.test", target="/ws")))
    attempts: asyncio.Queue[asyncio.Task[Message]] = asyncio.Queue()
    received: list[str | bytes] = []
    payloads = [b"first", b"second", b"sentinel"] if binary else ["héllo", "second", "sentinel"]

    async def app(scope, receive, send):
        assert (await receive())["type"] == "websocket.connect"
        await send({"type": "websocket.accept"})
        first = asyncio.create_task(receive())
        attempts.put_nowait(first)
        try:
            message = await first
        except asyncio.CancelledError:
            message = await receive()
        received.append(message["bytes" if binary else "text"])
        message = await receive()
        received.append(message["bytes" if binary else "text"])

    server = _AsgiServer(app)
    serving = asyncio.create_task(server._core_handle(stream, preamble()))
    try:
        first = await asyncio.wait_for(attempts.get(), 5)
        client.receive_data(bytes(stream.out))
        list(client.events())
        frames = [
            client.send(ws_events.BytesMessage(data=payload))
            if isinstance(payload, bytes)
            else client.send(ws_events.TextMessage(data=payload))
            for payload in payloads
        ]
        stream.feed(b"".join(frames))
        # Exercise cancellation before, during and after delivery. These yield scheduler turns,
        # not elapsed time; each outcome must preserve the same externally visible ordering.
        for _ in range(cancel_after_yields):
            await asyncio.sleep(0)
        first.cancel()
        await asyncio.wait_for(serving, 5)
        assert received == payloads[:2]
        assert stream.finished and stream.reset_code is None
        assert server.handlers == server.payload_bytes == 0
    finally:
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)


async def test_cancelled_websocket_receive_before_data_does_not_consume_the_next_message() -> None:
    from livekit.agents.http import _AsgiServer

    client = wsproto.WSConnection(wsproto.ConnectionType.CLIENT)
    stream = FakeStream()
    stream.feed(client.send(ws_events.Request(host="agents.test", target="/ws")))
    retrying = asyncio.Event()
    received = []

    async def app(scope, receive, send):
        await receive()
        await send({"type": "websocket.accept"})
        first = asyncio.create_task(receive())
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        retrying.set()
        received.append(await receive())

    server = _AsgiServer(app)
    serving = asyncio.create_task(server._core_handle(stream, preamble()))
    try:
        await asyncio.wait_for(retrying.wait(), 5)
        client.receive_data(bytes(stream.out))
        list(client.events())
        stream.feed(client.send(ws_events.TextMessage(data="still here")))
        await asyncio.wait_for(serving, 5)
        assert received == [{"type": "websocket.receive", "text": "still here"}]
        assert server.handlers == server.payload_bytes == 0
    finally:
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)


@pytest.mark.parametrize("payload", ["éé", b"\x00\x01\x02\x03"])
async def test_websocket_byte_capacity_is_reusable_after_each_delivery(
    payload: str | bytes,
) -> None:
    from livekit.agents.http import _AsgiServer, _HttpServerLimits

    client = wsproto.WSConnection(wsproto.ConnectionType.CLIENT)
    stream = FakeStream()
    stream.feed(client.send(ws_events.Request(host="agents.test", target="/ws")))
    accepted = asyncio.Event()
    delivered: asyncio.Queue[Message] = asyncio.Queue()
    release = asyncio.Event()

    async def app(scope, receive, send):
        await receive()
        await send({"type": "websocket.accept"})
        accepted.set()
        for _ in range(8):
            message = await receive()
            delivered.put_nowait(message)
            if message["type"] == "websocket.disconnect":
                return
        await release.wait()

    server = _AsgiServer(app, _HttpServerLimits(websocket_inbox=1, websocket_inbox_bytes=4))
    serving = asyncio.create_task(server._core_handle(stream, preamble()))
    try:
        await asyncio.wait_for(accepted.wait(), 5)
        client.receive_data(bytes(stream.out))
        list(client.events())
        for _ in range(8):
            event = (
                ws_events.TextMessage(data=payload)
                if isinstance(payload, str)
                else ws_events.BytesMessage(data=payload)
            )
            stream.feed(client.send(event))
            message = await asyncio.wait_for(delivered.get(), 5)
            assert message == {
                "type": "websocket.receive",
                "text" if isinstance(payload, str) else "bytes": payload,
            }
        release.set()
        await asyncio.wait_for(serving, 5)
        assert server.handlers == server.payload_bytes == 0
    finally:
        release.set()
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)


async def test_websocket_disconnect_with_queued_data_releases_capacity() -> None:
    from livekit.agents.http import _AsgiServer

    client = wsproto.WSConnection(wsproto.ConnectionType.CLIENT)
    stream = FakeStream()
    stream.feed(client.send(ws_events.Request(host="agents.test", target="/ws")))
    accepted = asyncio.Event()
    received = []

    async def app(scope, receive, send):
        await receive()
        await send({"type": "websocket.accept"})
        accepted.set()
        received.append(await receive())

    server = _AsgiServer(app)
    serving = asyncio.create_task(server._core_handle(stream, preamble()))
    try:
        await asyncio.wait_for(accepted.wait(), 5)
        client.receive_data(bytes(stream.out))
        list(client.events())
        stream.feed(
            client.send(ws_events.TextMessage(data="queued"))
            + client.send(ws_events.CloseConnection(code=1000, reason="done"))
        )
        await asyncio.wait_for(serving, 5)
        assert received == [{"type": "websocket.disconnect", "code": 1000}]
        assert server.handlers == server.payload_bytes == 0
    finally:
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)


async def test_endpoint_helper_rejects_a_registry_for_a_different_app() -> None:
    http = app_with_routes()
    called = False

    async def other_app(scope, receive, send):
        nonlocal called
        called = True

    stream = FakeStream()
    stream.feed(request_bytes("GET", "/hello"))
    with pytest.raises(ValueError, match="registry.*app"):
        await serve_endpoint(stream, preamble(), other_app, http)
    assert not called and not stream.out and stream.reset_code is None
    assert http._server.handlers == http._server.payload_bytes == 0


async def test_websocket_disconnect_waits_for_the_close_reply_to_flush() -> None:
    from livekit.agents.http import _AsgiServer, _HttpServerLimits

    client = wsproto.WSConnection(wsproto.ConnectionType.CLIENT)
    stream = GatedCloseStream()
    stream.feed(client.send(ws_events.Request(host="agents.test", target="/ws")))
    accepted = asyncio.Event()
    begin_receive = asyncio.Event()
    receive_started = asyncio.Event()
    received: list[Message] = []

    async def app(scope, receive, send):
        await receive()
        await send({"type": "websocket.accept"})
        accepted.set()
        await begin_receive.wait()
        receive_started.set()
        received.append(await receive())

    server = _AsgiServer(app, _HttpServerLimits(close_flush_timeout=5))
    serving = asyncio.create_task(server._core_handle(stream, preamble()))
    try:
        await asyncio.wait_for(accepted.wait(), 5)
        client.receive_data(bytes(stream.out))
        list(client.events())
        stream.feed(client.send(ws_events.CloseConnection(code=1000, reason="done")))
        await asyncio.wait_for(stream.close_started.wait(), 5)
        begin_receive.set()
        await asyncio.wait_for(receive_started.wait(), 5)
        assert not received and not serving.done() and not stream.finished
        stream.allow_close.set()
        await asyncio.wait_for(serving, 5)
        assert received == [{"type": "websocket.disconnect", "code": 1000}]
        assert stream.finished and stream.reset_code is None
        assert server.handlers == server.payload_bytes == 0
    finally:
        stream.allow_close.set()
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)


@pytest.mark.parametrize("cancel_caller", [False, True])
async def test_close_is_owned_by_the_connection_and_shared_with_drain(cancel_caller: bool) -> None:
    from livekit.agents.http import _AsgiServer, _HttpServerLimits

    stream = GatedCloseStream()
    client = wsproto.WSConnection(wsproto.ConnectionType.CLIENT)
    stream.feed(client.send(ws_events.Request(host="agents.test", target="/ws")))
    close_callers: asyncio.Queue[asyncio.Task[None]] = asyncio.Queue()
    caller_cancelled = asyncio.Event()
    received: list[Message] = []

    async def app(scope, receive, send):
        await receive()
        await send({"type": "websocket.accept"})
        closing = asyncio.create_task(send({"type": "websocket.close", "code": 1000}))
        close_callers.put_nowait(closing)
        try:
            await closing
        except asyncio.CancelledError:
            caller_cancelled.set()
        received.append(await receive())

    server = _AsgiServer(app, _HttpServerLimits(close_flush_timeout=5))
    initial_tasks = asyncio.all_tasks()
    serving = asyncio.create_task(server._core_handle(stream, preamble()))
    draining: asyncio.Task[None] | None = None
    try:
        closing = await asyncio.wait_for(close_callers.get(), 5)
        await asyncio.wait_for(stream.close_started.wait(), 5)
        handshake_size = len(stream.out)
        client.receive_data(bytes(stream.out))
        list(client.events())
        draining = asyncio.create_task(server._core_drain())
        if cancel_caller:
            closing.cancel()
            await asyncio.wait_for(caller_cancelled.wait(), 5)
        stream.feed_eof()
        await asyncio.wait_for(stream.peer_eof.wait(), 5)
        assert not next(iter(server.websockets)).inbox.closed
        assert not serving.done() and not draining.done()
        assert not stream.finished and stream.reset_code is None
        stream.allow_close.set()
        await asyncio.wait_for(asyncio.gather(serving, draining), 5)
        client.receive_data(bytes(stream.out[handshake_size:]))
        assert [type(event) for event in client.events()] == [ws_events.CloseConnection]
        assert received == [{"type": "websocket.disconnect", "code": 1000}]
        assert stream.finished and stream.reset_code is None
        assert server.handlers == server.payload_bytes == 0
        assert asyncio.all_tasks() == initial_tasks
    finally:
        stream.allow_close.set()
        tasks = [task for task in (serving, draining) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_cancellation_during_reader_cleanup_releases_all_capacity() -> None:
    from livekit.agents.http import _AsgiServer

    reading = asyncio.Event()
    cleaning = asyncio.Event()

    class SlowReader(FakeStream):
        async def read(self, max_bytes: int = 65536) -> bytes:
            if self.out:
                reading.set()
            try:
                return await super().read(max_bytes)
            except asyncio.CancelledError:
                cleaning.set()
                # Model an I/O boundary that must perform async cleanup on cancellation.
                await asyncio.Future()
                raise

    stream = SlowReader()
    client = wsproto.WSConnection(wsproto.ConnectionType.CLIENT)
    stream.feed(client.send(ws_events.Request(host="agents.test", target="/ws")))

    async def app(scope, receive, send):
        await receive()
        await send({"type": "websocket.accept"})
        await reading.wait()

    server = _AsgiServer(app)
    initial_tasks = asyncio.all_tasks()
    serving = asyncio.create_task(server._core_handle(stream, preamble()))
    try:
        await asyncio.wait_for(cleaning.wait(), 5)
        serving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(serving, 5)
        assert server.handlers == server.payload_bytes == 0
        assert asyncio.all_tasks() == initial_tasks
    finally:
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
