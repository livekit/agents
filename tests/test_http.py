"""``livekit.agents.http`` client: limits, errors, websocket messages, the job session and trust,
against loopback peers."""

from __future__ import annotations

import asyncio
import os
import socket
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import aiohttp
import certifi
import pytest
from aiohttp import web

from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError
from livekit.agents.http import HttpSession, WebSocketMessageType, client as http
from livekit.agents.types import APIConnectOptions

pytestmark = pytest.mark.unit

FAST = 0.2
SLOW = 5.0


def _options(**limits: float | None) -> APIConnectOptions:
    every = {"pool_timeout": SLOW, "connect_timeout": SLOW, "write_timeout": SLOW}
    return APIConnectOptions(max_retry=0, **(every | {"read_timeout": SLOW} | limits))


def _session(**connector: Any) -> HttpSession:
    """An ``HttpSession`` over a connector the test shapes (a pool of one, a slow resolver)."""
    session = HttpSession()
    session._pool._client = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(**connector), trace_configs=[http._CONNECTION_TRACE]
    )
    return session


@pytest.fixture
async def server() -> AsyncIterator[str]:
    """``/slow?delay=`` answers late, ``/hang`` never answers, ``/endless`` never finishes its
    body, ``/sink`` never reads the request body, ``/fail`` answers 500, ``/ws`` never reads
    frames and ``/echo`` echoes until told to close."""
    release = asyncio.Event()

    async def slow(request: web.Request) -> web.Response:
        await asyncio.sleep(float(request.query["delay"]))
        return web.Response(text="ok")

    async def hang(request: web.Request) -> web.Response:
        await release.wait()
        return web.Response()

    async def endless(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        await release.wait()
        return response

    async def sink(request: web.Request) -> web.Response:
        while request.transport is not None and not request.transport.is_closing():
            await asyncio.sleep(0.01)  # until the client gives up
        return web.Response()

    async def fail(request: web.Request) -> web.Response:
        return web.Response(status=500, text="quota exceeded", headers={"x-trace-id": "tr-1"})

    async def redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound("/slow?delay=0")

    async def lines(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        for chunk in (b"data: a\r\nda", b"ta: " + b"b" * 100_000 + b"\n", b"tail"):
            await response.write(chunk)
        return response

    async def big(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_bytes(b"x" * 2048)
        await release.wait()
        return ws

    async def ws_idle(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        request.transport.pause_reading()  # type: ignore[union-attr]
        await release.wait()
        return ws

    async def echo(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        ws.headers["x-request-id"] = "req-1"
        await ws.prepare(request)
        async for msg in ws:
            if msg.data == "close":
                await ws.close(code=4000, message=b"bye")
            elif msg.type is aiohttp.WSMsgType.TEXT:
                await ws.send_str(msg.data)
            else:
                await ws.send_bytes(msg.data)
        return ws

    app = web.Application(client_max_size=1 << 30)
    app.router.add_route("*", "/slow", slow)
    app.router.add_get("/hang", hang)
    app.router.add_get("/endless", endless)
    app.router.add_post("/sink", sink)
    app.router.add_get("/fail", fail)
    app.router.add_get("/redirect", redirect)
    app.router.add_get("/lines", lines)
    app.router.add_get("/big", big)
    app.router.add_get("/ws", ws_idle)
    app.router.add_get("/echo", echo)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        release.set()
        await runner.cleanup()


@pytest.fixture
async def silent_peer() -> AsyncIterator[str]:
    """Accepts connections and never answers."""
    held: list[asyncio.StreamWriter] = []

    async def accept(_: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        held.append(writer)

    srv = await asyncio.start_server(accept, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        for w in held:
            w.close()
        srv.close()


@pytest.fixture
async def session() -> AsyncIterator[HttpSession]:
    session = HttpSession()
    yield session
    await session.aclose()


async def _timeout_of(call: Callable[[], Awaitable[object]]) -> str | None:
    """The limit ``call`` fails with; a limit that never fires fails the test after 3s."""
    with pytest.raises(APITimeoutError) as raised:
        async with asyncio.timeout(3):
            await call()
    return raised.value.phase


async def _get(session: HttpSession, url: str, options: APIConnectOptions) -> None:
    async with session.get(url, conn_options=options) as response:
        await response.read()


# --- limits -----------------------------------------------------------------------------------


async def test_a_slow_response_fires_the_read_limit(server: str, session: HttpSession) -> None:
    phase = await _timeout_of(lambda: _get(session, f"{server}/hang", _options(read_timeout=FAST)))
    assert phase == "read"


async def test_a_response_within_the_read_limit_ignores_the_other_limits(
    server: str, session: HttpSession
) -> None:
    """Waiting for the response is not a write, pool or connect wait."""
    options = _options(read_timeout=SLOW, write_timeout=0.05, pool_timeout=0.05, connect_timeout=1)
    async with session.post(f"{server}/slow?delay=0.3", conn_options=options, data=b"x") as resp:
        assert await resp.text() == "ok"


async def test_a_slow_dns_lookup_fires_the_connect_limit(server: str) -> None:
    class SlowResolver(aiohttp.ThreadedResolver):
        async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET):  # type: ignore[no-untyped-def,override]
            await asyncio.sleep(SLOW)
            return await super().resolve(host, port, family)

    session = _session(resolver=SlowResolver(), use_dns_cache=False)
    url = server.replace("127.0.0.1", "localhost") + "/hang"
    try:
        phase = await _timeout_of(lambda: _get(session, url, _options(connect_timeout=FAST)))
    finally:
        await session.aclose()
    assert phase == "connect"


async def test_waiting_for_a_pooled_connection_fires_the_pool_limit(server: str) -> None:
    session = _session(limit=1)
    try:
        async with session.get(f"{server}/endless", conn_options=_options()):  # the only one
            phase = await _timeout_of(
                lambda: _get(session, f"{server}/hang", _options(pool_timeout=FAST))
            )
    finally:
        await session.aclose()
    assert phase == "pool"


async def test_a_body_the_peer_does_not_read_fires_the_write_limit(
    server: str, session: HttpSession
) -> None:
    async def post() -> None:
        options = _options(write_timeout=FAST)
        async with session.post(f"{server}/sink", conn_options=options, data=b"x" * (256 << 20)):
            pass

    try:
        assert await _timeout_of(post) == "write"
    finally:
        await session.aclose()
        await asyncio.sleep(0.05)  # the server sees the disconnect


async def test_a_silent_websocket_handshake_fires_the_read_limit(
    silent_peer: str, session: HttpSession
) -> None:
    phase = await _timeout_of(
        lambda: session.ws_connect(f"{silent_peer}/ws", conn_options=_options(read_timeout=FAST))
    )
    assert phase == "read"


async def test_frames_the_peer_does_not_read_fire_the_write_limit(
    server: str, session: HttpSession
) -> None:
    ws = await session.ws_connect(f"{server}/ws", conn_options=_options(write_timeout=FAST))

    async def flood() -> None:
        while True:
            await ws.send_bytes(b"x" * (1 << 20))

    try:
        phase = await _timeout_of(flood)
    finally:
        ws._ws._writer.transport.abort()  # type: ignore[union-attr]
    assert phase == "write"


async def test_receive_timeout_fires_the_read_limit(server: str, session: HttpSession) -> None:
    ws = await session.ws_connect(f"{server}/echo")
    assert await _timeout_of(lambda: ws.receive(timeout=FAST)) == "read"
    await ws.close()


# --- errors and messages ----------------------------------------------------------------------


async def test_an_unreachable_host_is_a_connection_error(session: HttpSession) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # nothing listens once closed
    with pytest.raises(APIConnectionError) as raised:
        await _get(session, f"http://127.0.0.1:{port}/", _options())
    assert not isinstance(raised.value, APITimeoutError)


async def test_an_error_status_is_returned_and_raise_for_status_keeps_the_body(
    server: str, session: HttpSession
) -> None:
    async with session.get(f"{server}/fail") as response:
        assert (response.status, response.ok) == (500, False)
        with pytest.raises(APIStatusError) as raised:
            await response.raise_for_status()
    assert (raised.value.status_code, raised.value.body) == (500, "quota exceeded")


async def test_a_rejected_handshake_is_a_status_error(server: str, session: HttpSession) -> None:
    with pytest.raises(APIStatusError) as raised:
        await session.ws_connect(f"{server}/fail")
    assert raised.value.status_code == 500


async def test_redirects_are_returned_not_followed(server: str, session: HttpSession) -> None:
    async with session.get(f"{server}/redirect") as response:
        assert (response.status, response.headers["Location"]) == (302, "/slow?delay=0")
        assert not response.ok
        with pytest.raises(APIStatusError) as raised:
            await response.raise_for_status()
    assert raised.value.status_code == 302


async def test_with_request_id_names_the_header_status_errors_carry(
    server: str, session: HttpSession
) -> None:
    traced = session.with_request_id("x-trace-id")
    async with traced.get(f"{server}/fail") as response:
        with pytest.raises(APIStatusError) as raised:
            await response.raise_for_status()
    assert raised.value.request_id == "tr-1"
    with pytest.raises(APIStatusError) as raised:
        await traced.ws_connect(f"{server}/fail")
    assert raised.value.request_id == "tr-1"
    async with session.get(f"{server}/fail") as response:  # the session itself names none
        with pytest.raises(APIStatusError) as raised:
            await response.raise_for_status()
    assert raised.value.request_id is None


async def test_iter_lines_splits_across_chunks_without_a_line_limit(
    server: str, session: HttpSession
) -> None:
    async with session.get(f"{server}/lines") as response:
        lines = [line async for line in response.iter_lines()]
    assert lines == [b"data: a", b"data: " + b"b" * 100_000, b"tail"]


async def test_a_message_above_max_msg_size_fails_the_socket(
    server: str, session: HttpSession
) -> None:
    ws = await session.ws_connect(f"{server}/big", max_msg_size=1024)
    with pytest.raises(APIConnectionError):
        await ws.receive()


async def test_receive_timeout_zero_expires_at_once(server: str, session: HttpSession) -> None:
    ws = await session.ws_connect(f"{server}/echo")
    assert await _timeout_of(lambda: ws.receive(timeout=0)) == "read"
    await ws.close()


async def test_a_closed_session_refuses_requests(server: str) -> None:
    session = HttpSession()
    traced = session.with_request_id("x-trace-id")
    await session.aclose()
    assert session.closed and traced.closed
    with pytest.raises(RuntimeError):
        await _get(session, f"{server}/slow?delay=0", _options())


async def test_websocket_messages_and_close(server: str, session: HttpSession) -> None:
    ws = await session.ws_connect(f"{server}/echo")
    assert ws.headers["x-request-id"] == "req-1"
    await ws.send_json({"a": 1})
    await ws.send_bytes(b"\x01")
    await ws.send_str("close")
    messages = [msg async for msg in ws]
    assert [(m.type, m.data) for m in messages] == [
        (WebSocketMessageType.TEXT, '{"a": 1}'),
        (WebSocketMessageType.BINARY, b"\x01"),
    ]
    assert messages[0].json() == {"a": 1}
    assert (messages[0].text, messages[1].binary) == ('{"a": 1}', b"\x01")
    with pytest.raises(TypeError):
        _ = messages[0].binary
    closing = await ws.receive()
    assert closing.type is WebSocketMessageType.CLOSE
    assert ws.close_code == 4000


async def test_the_job_session_applies_the_limits_too(server: str) -> None:
    async with http.open() as session:
        phase = await _timeout_of(
            lambda: _get(session, f"{server}/hang", _options(read_timeout=FAST))
        )
    assert phase == "read"


# --- the job session --------------------------------------------------------------------------


async def test_open_binds_a_session_and_closes_it_on_exit(server: str) -> None:
    with pytest.raises(RuntimeError):
        http.http_session()

    async with http.open() as session:
        assert http.http_session() is session
        async with session.get(f"{server}/slow?delay=0") as response:
            assert await response.text() == "ok"

    assert session.closed
    with pytest.raises(RuntimeError):
        http.http_session()


async def test_open_is_reentrant_inner_does_not_close_outer() -> None:
    async with http.open() as outer:
        async with http.open() as inner:
            assert inner is outer
        assert http.http_session() is outer


async def test_open_isolated_per_task() -> None:
    barrier = asyncio.Event()

    async def worker() -> HttpSession:
        async with http.open() as session:
            await barrier.wait()
            assert http.http_session() is session
        return session

    task_a = asyncio.create_task(worker())
    task_b = asyncio.create_task(worker())
    await asyncio.sleep(0.01)
    barrier.set()
    sess_a, sess_b = await asyncio.gather(task_a, task_b)
    assert sess_a is not sess_b


# --- trust ------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_ssl_context_cache() -> None:
    http._ssl_context.cache_clear()


def test_ssl_context_is_built_once_from_the_exported_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every session reuses one context, which trusts what SSL_CERT_FILE names."""
    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    ctx = http._ssl_context()
    assert http._ssl_context() is ctx
    certifi_count = ssl.create_default_context(cafile=certifi.where()).cert_store_stats()["x509"]
    assert ctx.cert_store_stats()["x509"] == certifi_count


def test_export_ca_paths_falls_back_to_certifi_without_system_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    paths = ssl.get_default_verify_paths()._replace(cafile=None, capath=None)
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: paths)

    http._export_ca_paths()
    assert os.environ.get("SSL_CERT_FILE") == certifi.where()


def test_export_ca_paths_shares_system_paths_with_native_clients(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    paths = ssl.get_default_verify_paths()._replace(cafile=certifi.where(), capath=str(tmp_path))
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: paths)

    http._export_ca_paths()
    assert os.environ["SSL_CERT_FILE"] == certifi.where()
    assert os.environ["SSL_CERT_DIR"] == str(tmp_path)


def test_export_ca_paths_keeps_an_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSL_CERT_FILE", "/custom/bundle.pem")
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)

    http._export_ca_paths()
    assert os.environ["SSL_CERT_FILE"] == "/custom/bundle.pem"


# --- boundaries -------------------------------------------------------------------------------


def test_the_client_and_the_server_never_import_each_other() -> None:
    """``livekit.agents.http`` holds both directions; each stays independent of the other."""
    import ast

    import livekit.agents.http as package

    root = Path(package.__file__).parent
    for side, other in (("client", "server"), ("server", "client")):
        tree = ast.parse((root / f"{side}.py").read_text())
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 1 and not node.module
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module
        }
        assert other not in imported, f"http/{side}.py imports http/{other}.py"


def test_no_plugin_imports_aiohttp() -> None:
    """Plugins reach their providers through ``livekit.agents.http`` only; aiohttp is internal."""
    import ast

    plugins = Path(__file__).resolve().parents[1] / "livekit-plugins"
    offenders = []
    for source in plugins.glob("livekit-plugins-*/livekit/**/*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom) and node.level == 0
                else []
            )
            if any(name == "aiohttp" or name.startswith("aiohttp.") for name in names):
                offenders.append(f"{source.relative_to(plugins)}:{node.lineno}")
    assert not offenders, offenders
