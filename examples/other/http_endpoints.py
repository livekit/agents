"""HTTP endpoints served alongside an agent.

`server.http` is a FastAPI app, so anything you can do in FastAPI you can do here.
Run it with `python http_endpoints.py dev`, then:

    curl localhost:8321/hello
    curl -X POST localhost:8321/dispatch -H 'content-type: application/json' \
         -d '{"room": "my-room", "identity": "caller"}'
    open localhost:8321/docs

The other routes cover streaming in both directions, server-sent events and a websocket:

    head -c 12582912 /dev/zero | curl -X POST --data-binary @- localhost:8321/upload
    curl -o /dev/null localhost:8321/download/12582912
    curl -N localhost:8321/sse/5
    websocat ws://localhost:8321/ws

Experimental: `server.http_tunnel` also serves them through LiveKit, so they can be
reached without the worker accepting inbound connections. It reads LIVEKIT_URL and the
API key and secret from the environment, the same as the server itself. The first path
segment names the endpoint, and the cloud forwards the rest untouched, so every command
above works against the cloud by swapping the host and adding a token:

    curl -H "Authorization: Bearer $TOKEN" $LIVEKIT_HTTP_URL/agent-http/sse/5
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from fastapi import Depends, Header, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.agents.http import agent_health
from livekit.agents.tunnel import WebSocketTunnel

logger = logging.getLogger("http-endpoints")

load_dotenv()

server = AgentServer(port=8321)
server.http_tunnel = WebSocketTunnel()

server.http.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@server.http.get("/hello")
async def hello() -> dict:
    return {"hello": "world"}


class DispatchRequest(BaseModel):
    room: str
    identity: str


@server.http.post("/dispatch")
async def dispatch(payload: DispatchRequest) -> dict:
    """Take a validated body and act on the agent server.

    A body that does not match DispatchRequest is a 422 before this ever runs.
    """
    # handlers close over `server`: the HTTP server runs in this same process
    logger.info("dispatch requested", extra={"room": payload.room})
    return {"room": payload.room, "identity": payload.identity, "active": len(server.active_jobs)}


@server.http.post("/upload")
async def upload(request: Request) -> dict:
    """Read a body of any size without holding it in memory."""
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
    return {"bytes": total}


@server.http.get("/download/{size}")
async def download(size: int) -> StreamingResponse:
    """Write a body of any size the same way."""

    async def gen() -> AsyncIterator[bytes]:
        sent = 0
        while sent < size:
            chunk = min(65536, size - sent)
            yield b"x" * chunk
            sent += chunk

    return StreamingResponse(gen(), media_type="application/octet-stream")


@server.http.get("/sse/{count}")
async def sse(count: int) -> StreamingResponse:
    """Server-sent events, which reach the client as they are produced, not at the end."""

    async def gen() -> AsyncIterator[bytes]:
        for i in range(count):
            yield f"data: {i}\n\n".encode()
            await asyncio.sleep(0.3)

    return StreamingResponse(gen(), media_type="text/event-stream")


@server.http.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    """A socket, which the tunnel carries as the bytes of an upgrade like any other."""
    await sock.accept()
    with contextlib.suppress(Exception):
        while True:
            await sock.send_text(f"echo:{await sock.receive_text()}")


async def _verify_admin(x_api_key: str = Header(default="")) -> None:
    if x_api_key != "keep-me-in-an-env-var":
        raise HTTPException(status_code=401, detail="bad api key")


@server.http.get("/admin/jobs", dependencies=[Depends(_verify_admin)])
async def admin_jobs() -> dict:
    """Guard one route with a dependency.

    Prefer this over `add_middleware` for auth: middleware also covers `GET /`, and a
    401 there makes orchestrators restart the process.
    """
    return {"jobs": [job.job.id for job in server.active_jobs]}


@server.http.get("/")
async def health() -> dict:
    """Replace the built-in health check while keeping its checks.

    Defining `GET /` is optional; without it the built-in plain-text one is served.
    """
    reason = agent_health(server)
    return {"ok": reason is None, "reason": reason}


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(llm="openai/gpt-4.1-mini")
    await session.start(
        agent=Agent(instructions="You are a helpful assistant."),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
