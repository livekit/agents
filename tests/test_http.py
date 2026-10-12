"""The endpoint bridge (`livekit.agents.http`) over a fake stream: what the Rust worker hands it is
an `EndpointStream` after the preamble, so a fake with the same five methods exercises every path
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
from starlette.websockets import WebSocket

from livekit.agents.http import (
    HSR_INTERNAL,
    HSR_PROTOCOL,
    HSR_TIMEOUT,
    AgentHttp,
    EndpointRequest,
    serve_endpoint,
)

pytestmark = pytest.mark.unit


class FakeStream:
    """An `EndpointStream` in memory: the test is the node."""

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


def app_with_routes() -> AgentHttp:
    http = AgentHttp()

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
    http: AgentHttp, raw: bytes, info: EndpointRequest | None = None
) -> FakeStream:
    stream = FakeStream()
    stream.feed(raw)
    stream.feed_eof()
    task = asyncio.create_task(serve_endpoint(stream, info or preamble(), http.app, http))
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
    http = app_with_routes()
    by_path = {e["path"]: e for e in http.manifest()}
    assert by_path["/hello"] == {
        "path": "/hello",
        "methods": ["GET", "HEAD"],
        "kind": "http",
        "public": False,
    }
    assert by_path["/echo-size"]["methods"] == ["POST"] and by_path["/echo-size"]["public"]
    assert by_path["/ws"] == {"path": "/ws", "methods": ["GET"], "kind": "http", "public": False}
    assert by_path["/raw/{path:path}"]["methods"] == [
        "GET",
        "HEAD",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
        "TRACE",
    ]
    assert [e["path"] for e in http.manifest()][0] == "/hello", "route order is preserved"
