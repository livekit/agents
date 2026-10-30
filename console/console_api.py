"""Local developer backend. Project and database credentials never leave this process."""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import sqlite3
import time
import tomllib
from contextlib import asynccontextmanager, closing
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from google.protobuf.json_format import MessageToDict, ParseDict
from models import DatabaseDump, QueryResult
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from livekit import api
from livekit.agents._generated.storage_class import ThreadInfo, ThreadItem
from livekit.agents.a2a import AgentThreadGrant, Credentials
from livekit.agents.storage import AgentDBStorage, ThreadNotFound
from livekit.agents.storage._protocol import payload_to_item

SERVER_URL = re.sub(r"^ws", "http", os.environ["LIVEKIT_URL"]).rstrip("/")
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
CARD_TIMEOUT = 2.0
SUBS = ["demo-user", "demo-user-2"]
IDLE_TTL_SECONDS = 24 * 3600
TOKEN_TTL = 3600
ADMIN_URL = os.environ.get("CONSOLE_AGENTDB_ADMIN_URL", "").rstrip("/")
API_TOKEN = os.environ["CONSOLE_API_TOKEN"]
if not API_TOKEN:
    raise ValueError("CONSOLE_API_TOKEN must be set")
MAX_DUMP_BYTES = 64 * 1024 * 1024

storage = AgentDBStorage()
credentials = Credentials(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])


def storage_token() -> str:
    """The backend-only credential for the existing AgentDB admin service."""
    return (
        api.AccessToken(credentials.api_key, credentials.api_secret)
        .with_agent_grants(api.AgentGrants(database_admin=True))
        .with_ttl(timedelta(seconds=TOKEN_TTL))
        .to_jwt()
    )


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


@asynccontextmanager
async def snapshot(thread_id: str):
    """Export through the existing AgentDBService, retaining its credential server-side."""
    from livekit.protocol.agentdb import AgentDB

    if not ADMIN_URL:
        raise RuntimeError("Set CONSOLE_AGENTDB_ADMIN_URL to the AgentDB admin service")
    info = await storage.get_thread(thread_id, fresh=True)
    request = AgentDB.DumpRequest(database_id=info.database_id)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{ADMIN_URL}/twirp/livekit.AgentDBService/DumpDatabase",
            headers={"Authorization": f"Bearer {storage_token()}"},
            json=MessageToDict(request, preserving_proto_field_name=True),
        )
        response.raise_for_status()
        export = ParseDict(response.json(), AgentDB.DumpResponse())
        if not export.download_url:
            raise ValueError("DumpDatabase returned no download URL")
        data = bytearray()
        async with client.stream("GET", export.download_url) as download:
            download.raise_for_status()
            async for chunk in download.aiter_bytes():
                data.extend(chunk)
                if len(data) > MAX_DUMP_BYTES:
                    raise ValueError("Database export exceeds the console's 64 MiB limit")
    yield bytes(data), export.tip, info


def read_database(data: bytes, database_id: str, tip: int) -> dict[str, Any]:
    with closing(sqlite3.connect(":memory:")) as db:
        db.deserialize(data)
        db.row_factory = sqlite3.Row
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]

        def rows(table: str) -> list[dict[str, Any]]:
            columns = list(db.execute(f'PRAGMA table_info("{table}")'))
            names = [r[1] for r in columns]
            json_columns = {r[1] for r in columns if r[2].upper() == "JSONB"}
            fields = ", ".join(
                f'json("{name}") AS "{name}"' if name in json_columns else f'"{name}"'
                for name in names
            )
            result = []
            for row in db.execute(f'SELECT {fields} FROM "{table}" ORDER BY rowid LIMIT 1000'):
                item = dict(row)
                for name in json_columns:
                    item[name] = json.loads(item[name]) if item[name] is not None else None
                result.append(item)
            return result

        items = rows("items")
        for item in items:
            item["payload"] = history_item(item["payload"])
            item["type"] = item["payload"]["type"]
        result = DatabaseDump.model_validate(
            {
                "databaseId": database_id,
                "tip": tip,
                "tables": tables,
                "schemaVersion": db.execute("SELECT version FROM schema_version").fetchone()[0],
                "lanes": rows("lanes"),
                "items": items,
                "events": rows("events"),
                "inbox": rows("inbox"),
                "tasks": rows("tasks"),
            }
        )
        return result.model_dump(mode="json")


async def session_dump(thread_id: str) -> dict[str, Any]:
    info = await storage.get_thread(thread_id, fresh=True)
    out: dict[str, Any] = {
        "fetchedAt": int(time.time() * 1000),
        "session": row(info),
        "streams": {"url": "", "available": True},
    }
    try:
        async with snapshot(thread_id) as (data, tip, _):
            out["database"] = await asyncio.to_thread(read_database, data, info.database_id, tip)
    except Exception:
        out["databaseError"] = "The inspector could not load this conversation."
    return out


async def config(request: Request) -> Response:
    return JSONResponse({"grant": {"subs": SUBS}, "streams": {"url": "", "available": True}})


async def token(request: Request) -> Response:
    """Only a scoped thread grant is returned to the browser."""
    sub = request.query_params.get("sub") or SUBS[0]
    agent, endpoint = request.query_params.get("agent"), request.query_params.get("endpoint")
    if sub not in SUBS or not agent or not endpoint:
        return JSONResponse(
            {"error": "Choose a configured subject, agent and endpoint"}, status_code=400
        )
    grant = AgentThreadGrant(scope=sub, create=True, write=True, read=True, list=True)
    return JSONResponse(
        {
            "token": credentials.thread_grant_token(grant, identity=sub, ttl_seconds=TOKEN_TTL),
            "agent": agent,
            "sub": sub,
            "endpoint": endpoint_url(agent, endpoint),
            "endpointName": endpoint,
            "expiresAt": int(time.time()) + TOKEN_TTL,
        },
        headers={"Cache-Control": "no-store"},
    )


def endpoint_url(agent: str, endpoint: str) -> str:
    return f"{SERVER_URL}/agents/{agent}/default/{endpoint}"


async def sessions(request: Request) -> Response:
    """A scope's conversations can be continued by any configured agent."""
    sub = request.query_params.get("sub") or SUBS[0]
    if sub not in SUBS:
        return JSONResponse({"error": "Unknown subject"}, status_code=403)
    try:
        limit = min(50, max(1, int(request.query_params.get("limit") or 50)))
    except ValueError:
        return JSONResponse({"error": "Invalid limit"}, status_code=400)
    page = await storage.threads.list_threads(
        sub, before=request.query_params.get("before") or "", limit=limit
    )
    return JSONResponse({"sessions": [row(t) for t in page.threads], "before": page.before})


async def create_session(request: Request) -> Response:
    body = await request.json()
    if not isinstance(body, dict) or not body.get("agent") or not body.get("endpoint"):
        return JSONResponse({"error": "agent and endpoint are required"}, status_code=400)
    if body.get("sub", SUBS[0]) not in SUBS:
        return JSONResponse({"error": "Unknown subject"}, status_code=403)
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


def history_item(payload: dict[str, Any]) -> dict[str, Any]:
    """A ``ChatContext.ChatItem`` in protobuf JSON as the console's history item."""
    return dump_of(payload_to_item(payload))


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
    items = [history_item(entry.item) for entry in entries if entry.item]
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


async def session_info(request: Request) -> Response:
    try:
        return JSONResponse(row(await storage.get_thread(request.path_params["id"], fresh=True)))
    except ThreadNotFound:
        return JSONResponse({"error": "Unknown conversation"}, status_code=404)


async def session(request: Request) -> Response:
    thread_id = request.path_params["id"]
    try:
        return JSONResponse(await session_dump(thread_id))
    except ThreadNotFound:
        return JSONResponse({"error": f"unknown thread {thread_id}"}, status_code=404)


def query_snapshot(data: bytes, sql: str, limit: int, tip: int) -> dict[str, Any]:
    started = time.monotonic()
    with closing(sqlite3.connect(":memory:")) as db:
        db.deserialize(data)
        db.execute("PRAGMA query_only=ON")

        # query_only enforces writes; prevent SQL from relaxing it or attaching files.
        def authorize(action, first, *_):
            if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_PRAGMA and first not in (
                "table_info",
                "table_xinfo",
                "index_info",
                "index_list",
            ):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        db.set_authorizer(authorize)
        db.set_progress_handler(lambda: int(time.monotonic() - started > 2), 1000)
        cursor = db.execute(sql)
        rows = cursor.fetchmany(limit + 1)
        result = QueryResult(
            columns=[c[0] for c in cursor.description or []],
            rows=[
                [{"blob": len(v)} if isinstance(v, bytes) else v for v in r] for r in rows[:limit]
            ],
            truncated=len(rows) > limit,
            tip=tip,
            ms=round((time.monotonic() - started) * 1000),
        )
        return result.model_dump(mode="json")


async def query(request: Request) -> Response:
    body = await request.json()
    sql = body.get("sql") if isinstance(body, dict) else None
    if not isinstance(sql, str) or not sql.strip():
        return JSONResponse({"error": "sql is required"}, status_code=400)
    try:
        limit = min(10_000, max(1, int(body.get("limit", 500))))
        async with snapshot(request.path_params["id"]) as (data, tip, _):
            result = await asyncio.to_thread(query_snapshot, data, sql, limit, tip)
        return JSONResponse(result)
    except (ValueError, sqlite3.Error):
        return JSONResponse(
            {"error": "Only a valid read query can run on the snapshot"}, status_code=400
        )


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
    headers = {
        "Authorization": f"Bearer {credentials.project_token(ttl_seconds=60)}",
        "A2A-Version": "1.0",
    }

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


class DeveloperAuthorization(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        supplied = request.headers.get("authorization", "")
        if not secrets.compare_digest(supplied, f"Bearer {API_TOKEN}"):
            return JSONResponse({"error": "Developer authorization required"}, status_code=401)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response


app = Starlette(
    routes=[
        Route("/api/config", config),
        Route("/api/token", token),
        Route("/api/examples", examples),
        Route("/api/sessions", sessions),
        Route("/api/sessions", create_session, methods=["POST"]),
        Route("/api/admin/sessions/{id}", session),
        Route("/api/sessions/{id}/info", session_info),
        Route("/api/sessions/{id}/attributes", attributes, methods=["POST"]),
        Route("/api/admin/sessions/{id}/query", query, methods=["POST"]),
        Route("/api/sessions/{id}/history", history),
        Route("/api/sessions/{id}/tasks", tasks),
    ]
)
app.add_middleware(DeveloperAuthorization)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("CONSOLE_API_PORT", "7795")))
