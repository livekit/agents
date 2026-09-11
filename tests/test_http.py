from __future__ import annotations

import asyncio
import json
import signal
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace

import aiohttp
import pytest
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel

from livekit.agents import AgentServer, JobContext
from livekit.agents.http import _HttpRunner, _register_builtin_routes, agent_health

pytestmark = pytest.mark.unit


class RunRequest(BaseModel):
    room: str


def _make_server() -> AgentServer:
    server = AgentServer(ws_url="ws://localhost:7880", api_key="key", api_secret="secret")

    @server.rtc_session()
    async def _entrypoint(ctx: JobContext) -> None:
        pass

    # the process pool belongs to run(); the worker info route only counts its jobs
    server._proc_pool = SimpleNamespace(processes=[])  # type: ignore[assignment]
    return server


@asynccontextmanager
async def _serving(server: AgentServer) -> AsyncIterator[str]:
    """Wire up the routes the way run() does, serve on an ephemeral port, yield the url."""
    _register_builtin_routes(server)
    runner = _HttpRunner(server.http, host="127.0.0.1", port=0)
    await runner.start()
    try:
        yield f"http://127.0.0.1:{runner.port}"
    finally:
        await runner.aclose()


async def test_serves_user_routes_and_builtins() -> None:
    server = _make_server()

    @server.http.get("/token")
    async def token() -> dict:
        return {"ok": True}

    @server.http.post("/run")
    async def run(payload: RunRequest) -> dict:
        return {"room": payload.room, "active": len(server.active_jobs)}

    async with _serving(server) as base, aiohttp.ClientSession() as sess:
        async with sess.get(f"{base}/token") as resp:
            assert resp.status == 200
            assert await resp.json() == {"ok": True}

        async with sess.post(f"{base}/run", json={"room": "my-room"}) as resp:
            assert resp.status == 200
            assert await resp.json() == {"room": "my-room", "active": 0}

        # pydantic validates the body, so we do not
        async with sess.post(f"{base}/run", json={"nope": 1}) as resp:
            assert resp.status == 422

        async with sess.get(f"{base}/") as resp:
            assert resp.status == 200
            assert await resp.text() == "OK"

        async with sess.get(f"{base}/worker") as resp:
            assert resp.status == 200
            info = json.loads(await resp.text())
            assert info["worker_type"] == "JT_ROOM"
            assert info["sdk_version"]
            assert info.get("active_jobs", 0) == 0  # proto json drops zero values


async def test_health_reports_a_failed_connection() -> None:
    server = _make_server()
    assert agent_health(server) is None

    server._connection_failed = True
    assert agent_health(server) == "failed to connect to livekit"

    async with _serving(server) as base, aiohttp.ClientSession() as sess:
        async with sess.get(f"{base}/") as resp:
            assert resp.status == 503
            assert await resp.text() == "failed to connect to livekit"


async def test_a_user_health_route_replaces_the_default() -> None:
    server = _make_server()

    @server.http.get("/")
    async def health() -> dict:
        return {"ok": agent_health(server) is None, "mine": True}

    async with _serving(server) as base, aiohttp.ClientSession() as sess:
        async with sess.get(f"{base}/") as resp:
            assert resp.status == 200
            assert await resp.json() == {"ok": True, "mine": True}


async def test_worker_info_cannot_be_replaced() -> None:
    server = _make_server()

    @server.http.get("/worker")
    async def worker() -> dict:
        return {"mine": True}

    with pytest.raises(ValueError, match="GET /worker"):
        _register_builtin_routes(server)


async def test_other_methods_on_worker_reach_the_user() -> None:
    server = _make_server()

    @server.http.post("/worker")
    async def worker() -> dict:
        return {"mine": True}

    async with _serving(server) as base, aiohttp.ClientSession() as sess:
        async with sess.post(f"{base}/worker") as resp:
            assert resp.status == 200
            assert await resp.json() == {"mine": True}
        async with sess.get(f"{base}/worker") as resp:
            assert resp.status == 200
            assert json.loads(await resp.text())["worker_type"] == "JT_ROOM"


async def test_user_middleware_applies_to_every_route() -> None:
    server = _make_server()

    @server.http.get("/token")
    async def token() -> dict:
        return {"ok": True}

    @server.http.middleware("http")
    async def require_auth(request: Request, call_next: Callable) -> Response:
        if request.headers.get("x-auth") != "secret":
            return Response(status_code=401)
        return await call_next(request)  # type: ignore[no-any-return]

    async with _serving(server) as base, aiohttp.ClientSession() as sess:
        # the built-ins are ordinary routes, so the user's stack covers them too
        for path in ("/", "/worker", "/token"):
            async with sess.get(f"{base}{path}") as resp:
                assert resp.status == 401
            async with sess.get(f"{base}{path}", headers={"x-auth": "secret"}) as resp:
                assert resp.status == 200


async def test_port_is_resolved_from_the_listening_socket() -> None:
    server = _make_server()
    runner = _HttpRunner(server.http, host="127.0.0.1", port=0)
    assert runner.port == 0
    await runner.start()
    try:
        assert runner.port != 0
    finally:
        await runner.aclose()


async def test_user_app_lifespan_runs() -> None:
    server = _make_server()
    events: list[str] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        events.append("startup")
        yield
        events.append("shutdown")

    server.http.router.lifespan_context = lifespan

    async with _serving(server) as base, aiohttp.ClientSession() as sess:
        assert events == ["startup"]
        async with sess.get(f"{base}/") as resp:
            assert resp.status == 200

    assert events == ["startup", "shutdown"]


async def test_openapi_describes_the_served_routes() -> None:
    server = _make_server()

    @server.http.get("/token")
    async def token() -> dict:
        return {"ok": True}

    async with _serving(server) as base, aiohttp.ClientSession() as sess:
        async with sess.get(f"{base}/docs") as resp:
            assert resp.status == 200
        async with sess.get(f"{base}/openapi.json") as resp:
            assert resp.status == 200
            assert sorted((await resp.json())["paths"]) == ["/", "/token", "/worker"]


async def test_signal_handlers_are_left_to_the_cli() -> None:
    server = _make_server()
    before = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)

    async with _serving(server):
        assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before

    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


async def test_streaming_response() -> None:
    server = _make_server()

    @server.http.get("/stream")
    async def stream() -> Response:
        from fastapi.responses import StreamingResponse

        async def gen() -> AsyncIterator[bytes]:
            for i in range(3):
                yield f"data: {i}\n\n".encode()
                await asyncio.sleep(0.01)

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with _serving(server) as base, aiohttp.ClientSession() as sess:
        async with sess.get(f"{base}/stream") as resp:
            assert resp.status == 200
            assert resp.headers["Content-Type"].startswith("text/event-stream")
            assert await resp.text() == "data: 0\n\ndata: 1\n\ndata: 2\n\n"
