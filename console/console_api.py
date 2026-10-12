"""The console's debug API over the threads service and agent-db, a development tool.

What the web console beside it reads and writes, served on ``:7795`` from the environment of
``examples/README.md``: the project token it talks to the agents with, the examples
it can talk to and whether each runs, the threads of a scope started on one of them, one
thread's database as the console draws it (lanes, items, journal, inbox, tasks), a watch of
it over SSE, its history and background tasks from the threads service, the thread's
attributes, and the answer streams, relayed. Run with
``uv run python console_api.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import jwt
import tomllib
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from livekit.agents._generated.storage_threads import ThreadInfo, ThreadItem
from livekit.agents.a2a import Credentials
from livekit.agents.storage import AgentDBStorage, ThreadNotFound

SERVER_URL = re.sub(r"^ws", "http", os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880")).rstrip(
    "/"
)
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
CARD_TIMEOUT = 2.0
SUBS = ["demo-user", "demo-user-2"]
IDLE_TTL_SECONDS = 24 * 3600  # a conversation outlives the break; anonymous threads get 15 minutes
TOKEN_TTL = 3600
WATCH_INTERVAL = 1.0

storage = AgentDBStorage()
credentials = Credentials(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])

# the streams the relay reads, through the public API the storage uses
STREAMS_URL = f"{storage.url}/v0/agentstore/streams"


def storage_token() -> str:
    """The worker's credential on agent-db, ``agent.databaseAdmin``, for the stream relay."""
    key, secret = os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
    now = int(time.time())
    claims = {"iss": key, "nbf": now - 5, "exp": now + TOKEN_TTL, "agent": {"databaseAdmin": True}}
    return jwt.encode(claims, secret, algorithm="HS256")


def row(t: ThreadInfo) -> dict[str, Any]:
    """A thread as the console's session row."""
    return {
        "sessionId": t.thread_id,
        "agentName": t.attributes.get("agent", ""),
        "sub": t.scope,
        "attributes": t.attributes,
        "region": "",
        "databaseId": t.database_id,
        "createdAt": t.created_at,
        "lastActive": t.last_active,
        "expiresAt": t.expires_at,
        "idleTtlSeconds": t.idle_ttl_seconds,
    }


def dump_of(model: Any) -> Any:
    return model.model_dump(mode="json")


async def database_dump(thread_id: str) -> dict[str, Any]:
    """The thread's database as the console draws it."""
    db = await storage.open(thread_id)
    try:
        lanes = await db.lanes()
        items: list[Any] = []
        inbox: list[Any] = []
        tasks: list[Any] = []
        for lane in lanes:
            items += await db.history(lane.lane)
            inbox += await db.inbox(lane.lane)
            tasks += await db.tasks(lane.lane, limit=200)
        events = await db.events(0, 1000)
        return {
            "databaseId": db.database_id,
            "tip": db.tip,
            "tables": ["lanes", "items", "events", "inbox", "tasks"],
            "schemaVersion": 1,
            "lanes": [dump_of(lane) for lane in lanes],
            "items": [dump_of(item) for item in items],
            "events": [dump_of(event) for event in events],
            "inbox": [dump_of(entry) for entry in inbox],
            "tasks": [dump_of(task) for task in tasks],
        }
    finally:
        db.close()


async def session_dump(thread_id: str) -> dict[str, Any]:
    info = await storage.get_thread(thread_id, fresh=True)
    out: dict[str, Any] = {
        "fetchedAt": int(time.time() * 1000),
        "session": row(info),
        "streams": {"url": STREAMS_URL, "available": True},
    }
    try:
        database = await database_dump(thread_id)
    except Exception as e:
        out["databaseError"] = str(e)
    else:
        out["database"] = database
    return out


async def config(request: Request) -> Response:
    return JSONResponse(
        {
            "agentstoreUrl": storage.url,
            "streamsUrl": STREAMS_URL,
            "streams": {"url": STREAMS_URL, "available": True},
            "grant": {"subs": SUBS},
        }
    )


async def token(request: Request) -> Response:
    """The project credential: through the node a thread grant is a stripped bearer in v0, so
    the console, a tool of the project, talks as the project and scopes threads itself."""
    sub = request.query_params.get("sub") or SUBS[0]
    agent, endpoint = request.query_params["agent"], request.query_params["endpoint"]
    return JSONResponse(
        {
            "token": credentials.project_token(identity=sub, ttl_seconds=TOKEN_TTL),
            "agent": agent,
            "sub": sub,
            "endpoint": endpoint_url(agent, endpoint),
            "endpointName": endpoint,
            "expiresAt": int(time.time()) + TOKEN_TTL,
        }
    )


def endpoint_url(agent: str, endpoint: str) -> str:
    return f"{SERVER_URL}/agents/{agent}/default/{endpoint}"


def started_on(t: ThreadInfo, agent: str, endpoint: str) -> bool:
    """Whether the thread was started on ``agent``'s ``endpoint``."""
    return (t.attributes.get("agent"), t.attributes.get("endpoint")) == (agent, endpoint)


async def sessions(request: Request) -> Response:
    """The subject's threads started on one agent's endpoint, newest first."""
    sub = request.query_params.get("sub") or SUBS[0]
    agent, endpoint = request.query_params["agent"], request.query_params["endpoint"]
    limit = int(request.query_params.get("limit") or 50)
    found: list[ThreadInfo] = []
    before = request.query_params.get("before") or ""
    while len(found) < limit:
        page = await storage.threads.list_threads(sub, before=before, limit=limit)
        found += [t for t in page.threads if started_on(t, agent, endpoint)]
        if not (before := page.before):
            break
    return JSONResponse({"sessions": [row(t) for t in found[:limit]], "before": before})


async def create_session(request: Request) -> Response:
    body = await request.json()
    attributes = {
        "agent": str(body["agent"]),
        "endpoint": str(body["endpoint"]),
        **(body.get("attributes") or {}),
    }
    created = await storage.threads.create_thread(
        scope=str(body.get("sub") or SUBS[0]),
        attributes=attributes,
        idle_ttl_seconds=IDLE_TTL_SECONDS,
    )
    return JSONResponse({"session": row(created)}, status_code=201)


def snake(name: str) -> str:
    return re.sub(r"[A-Z]", lambda m: "_" + m.group().lower(), name)


def seconds(timestamp: str) -> float:
    """An RFC 3339 UTC timestamp, as protobuf JSON writes it, in unix seconds."""
    head, _, fraction = timestamp.removesuffix("Z").removesuffix("+00:00").partition(".")
    whole = datetime.fromisoformat(head).replace(tzinfo=timezone.utc).timestamp()
    return whole + float(f"0.{fraction or 0}")


def history_item(entry: ThreadItem) -> dict[str, Any] | None:
    """A ``ChatContext.ChatItem`` in protobuf JSON as the console's history item: the kind as
    ``type``, snake_case fields, ``created_at`` in seconds."""
    if not entry.item:
        return None
    kind, body = next(iter(entry.item.items()))
    out = {snake(k): v for k, v in body.items()}
    out["type"] = snake(kind)
    if isinstance(out.get("role"), str):
        out["role"] = out["role"].lower()
    if isinstance(out.get("created_at"), str):
        out["created_at"] = seconds(out["created_at"])
    return out


async def history(request: Request) -> Response:
    """The agent's items from the threads service, oldest first."""
    thread_id = request.path_params["id"]
    try:
        entries: list[ThreadItem] = []
        after = ""
        while True:
            page = await storage.threads.get_thread_items(
                thread_id, agent=request.query_params["agent"], after=after
            )
            entries += page.items
            if not (after := page.after):
                break
    except ThreadNotFound:
        return JSONResponse({"error": f"unknown thread {thread_id}"}, status_code=404)
    items = [item for entry in entries if (item := history_item(entry))]
    return JSONResponse({"items": items})


async def tasks(request: Request) -> Response:
    """The thread's background tasks from the threads service, oldest first."""
    thread_id = request.path_params["id"]
    try:
        found = await storage.threads.get_thread_tasks(
            thread_id, status=request.query_params.get("status")
        )
    except ThreadNotFound:
        return JSONResponse({"error": f"unknown thread {thread_id}"}, status_code=404)
    return JSONResponse({"tasks": [dump_of(task) for task in found]})


async def session(request: Request) -> Response:
    thread_id = request.path_params["id"]
    try:
        return JSONResponse(await session_dump(thread_id))
    except ThreadNotFound:
        return JSONResponse({"error": f"unknown thread {thread_id}"}, status_code=404)


async def attributes(request: Request) -> Response:
    thread_id = request.path_params["id"]
    body = await request.json()
    try:
        info = await storage.threads.update_thread(thread_id, attributes=body.get("attributes"))
    except ThreadNotFound:
        return JSONResponse({"error": f"unknown thread {thread_id}"}, status_code=404)
    return JSONResponse(row(info))


async def examples(request: Request) -> Response:
    """Every ``examples/*/console.toml``, live when its endpoint answers its agent card."""
    found = []
    for path in sorted(EXAMPLES_DIR.glob("*/console.toml")):
        example = tomllib.loads(path.read_text())
        example["dir"] = str(path.parent.relative_to(EXAMPLES_DIR.parent))
        example["url"] = endpoint_url(example["agent"], example["endpoint"])
        found.append(example)
    headers = {"Authorization": f"Bearer {credentials.project_token(ttl_seconds=60)}"}

    async def live(client: httpx.AsyncClient, url: str) -> bool:
        try:
            card = await client.get(f"{url}/.well-known/agent-card.json", headers=headers)
        except httpx.HTTPError:
            return False
        return card.status_code == 200

    async with httpx.AsyncClient(timeout=CARD_TIMEOUT) as client:
        states = await asyncio.gather(*(live(client, e["url"]) for e in found))
    return JSONResponse(
        {"examples": [{**e, "live": state} for e, state in zip(found, states, strict=True)]}
    )


async def watch(request: Request) -> Response:
    """The dump again every ``WATCH_INTERVAL``, as the Go tool did; the console diffs."""
    thread_id = request.path_params["id"]

    async def frames() -> AsyncIterator[bytes]:
        while not await request.is_disconnected():
            try:
                event, body = "dump", json.dumps(await session_dump(thread_id))
            except Exception as e:
                event, body = "error", json.dumps({"error": str(e)})
            yield f"event: {event}\ndata: {body}\n\n".encode()
            await asyncio.sleep(WATCH_INTERVAL)

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


async def stream(request: Request) -> Response:
    """``/v1/streams/{id}`` relayed as is, under the project token."""
    stream_id = request.path_params["id"]
    url = f"{STREAMS_URL}/{stream_id}"
    headers = {"Authorization": f"Bearer {storage_token()}"}
    if accept := request.headers.get("accept"):
        headers["Accept"] = accept
    client = httpx.AsyncClient(timeout=None)
    upstream = await client.send(
        client.build_request("GET", url, params=request.query_params, headers=headers),
        stream=True,
    )
    passed = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() in ("content-type", "cache-control", "etag") or k.lower().startswith("stream-")
    }
    passed["X-Accel-Buffering"] = "no"

    async def body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(body(), status_code=upstream.status_code, headers=passed)


app = Starlette(
    routes=[
        Route("/api/config", config),
        Route("/api/token", token),
        Route("/api/examples", examples),
        Route("/api/sessions", sessions),
        Route("/api/sessions", create_session, methods=["POST"]),
        Route("/api/sessions/{id}", session),
        Route("/api/sessions/{id}/attributes", attributes, methods=["POST"]),
        Route("/api/sessions/{id}/history", history),
        Route("/api/sessions/{id}/tasks", tasks),
        Route("/api/sessions/{id}/watch", watch),
        Route("/api/streams/{id}", stream),
    ]
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("CONSOLE_API_PORT", "7795")))
