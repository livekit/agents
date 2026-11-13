"""auth.md through the LiveKit node: a registered worker serves an A2A endpoint and callers reach
it at the node's /agents/{agent}/{deployment}/ front with the tokens auth.md describes. The node
verifies the signature and the endpoint call grant; the worker reads the forwarded token and
authorizes the thread, the method and its flag."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
import jwt
import pytest

from livekit.agents import AgentServer, JobContext
from livekit.agents.a2a import A2ASessionContext, Credentials
from livekit.agents._generated.a2a_auth import AgentThreadGrant
from livekit.agents.storage import SQLiteStorage

from .resources import REQUEST_SECONDS, STARTUP_SECONDS


@dataclass
class Door:
    url: str  # the endpoint's JSON-RPC URL through the node
    card: str
    key: str
    secret: str
    credentials: Credentials

    def grant(self, **fields: object) -> str:
        """A thread grant with the endpoint call grant, as an application backend mints it."""
        return self.credentials.thread_grant_token(AgentThreadGrant(**fields))  # type: ignore[arg-type]

    def raw(self, claims: dict[str, object], *, secret: str | None = None) -> str:
        """A token with exactly these grants, signed by the project's key unless told otherwise."""
        now = int(time.time())
        body = {"iss": self.key, "sub": "caller", "nbf": now - 5, "exp": now + 600, **claims}
        return jwt.encode(body, secret or self.secret, algorithm="HS256")


async def _entrypoint(ctx: JobContext) -> None:
    """The voice door's entrypoint; these journeys dispatch no job."""


@pytest.fixture
async def door(live_credentials: tuple[str, str, str], tmp_path: Path) -> AsyncIterator[Door]:
    url, key, secret = live_credentials
    name = "e2e-auth-" + uuid.uuid4().hex[:8]
    parts = urlsplit(url)
    scheme = "https" if parts.scheme in ("https", "wss") else "http"
    server = AgentServer(
        agent_name=name,
        storage=SQLiteStorage(directory=tmp_path),
        listeners={"health": {"port": 0}},
        num_idle_processes=0,
    )

    @server.a2a_session(endpoint="desk", description="Echoes the caller.")
    async def desk(ctx: A2ASessionContext) -> str:
        return "echo:" + (ctx.message.text_content or "")

    server.rtc_session()(_entrypoint)
    base = f"{scheme}://{parts.netloc}/agents/{name}/_/desk"
    credentials = Credentials(key, secret)
    d = Door(f"{base}/v1", f"{base}/.well-known/agent-card.json", key, secret, credentials)
    run = asyncio.create_task(server._run(devmode=False))
    try:
        reach = {"Authorization": f"Bearer {credentials.project_token()}"}
        async with aiohttp.ClientSession() as http, asyncio.timeout(STARTUP_SECONDS):
            while True:
                if run.done():
                    run.result()
                try:
                    async with http.get(d.card, headers=reach) as response:
                        if response.status == 200:
                            break
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(0.2)
        yield d
    finally:
        await server.aclose()
        await run


@pytest.fixture
async def http() -> AsyncIterator[aiohttp.ClientSession]:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(REQUEST_SECONDS)) as session:
        yield session


def message(text: str, thread: str = "") -> dict[str, object]:
    msg: dict[str, object] = {
        "messageId": uuid.uuid4().hex,
        "role": "ROLE_USER",
        "parts": [{"text": text}],
    }
    if thread:
        msg["contextId"] = thread
    return {"message": msg}


async def call(
    http: aiohttp.ClientSession,
    door: Door,
    token: str | None,
    method: str,
    params: dict[str, object],
    *,
    in_query: bool = False,
) -> tuple[int, dict[str, object]]:
    url, headers = door.url, {}
    if token and in_query:
        url += f"?access_token={token}"
    elif token:
        headers["Authorization"] = f"Bearer {token}"
    body = {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params}
    async with http.post(url, json=body, headers=headers) as response:
        text = await response.text()
        try:
            payload = await response.json(content_type=None)
        except ValueError:
            payload = {"text": text}
        return response.status, payload


def task_of(payload: dict[str, object]) -> dict[str, object]:
    result = payload["result"]
    assert isinstance(result, dict), payload
    task = result.get("task", result)
    assert isinstance(task, dict), payload
    return task


async def new_thread(http: aiohttp.ClientSession, door: Door, scope: str) -> str:
    token = door.grant(scope=scope, create=True, write=True, read=True)
    status, payload = await call(http, door, token, "message/send", message("hello"))
    assert status == 200, payload
    thread = task_of(payload)["contextId"]
    assert isinstance(thread, str) and thread.startswith("AT_"), payload
    return thread


# -- the node: signature and endpoint call grant


async def test_the_node_refuses_a_caller_it_cannot_verify(
    door: Door, http: aiohttp.ClientSession
) -> None:
    """No token, a token signed with the wrong secret, and an unknown key are 401 at the node."""
    grant = {"agentEndpoint": {"call": True}, "agentThread": {"all": True, "write": True}}

    anonymous, _ = await call(http, door, None, "message/send", message("hi"))
    forged, _ = await call(http, door, door.raw(grant, secret="not-the-secret"), "message/send", message("hi"))
    unknown_key = jwt.encode({**grant, "iss": "APInotakey", "sub": "x", "exp": int(time.time()) + 60}, "s")
    unknown, _ = await call(http, door, unknown_key, "message/send", message("hi"))

    assert (anonymous, forged, unknown) == (401, 401, 401)


async def test_a_token_from_another_key_of_the_project_is_admitted(
    door: Door, http: aiohttp.ClientSession
) -> None:
    """The node verifies against every key of the project, not only the worker's."""
    key = os.environ.get("LIVEKIT_SECOND_API_KEY")
    secret = os.environ.get("LIVEKIT_SECOND_API_SECRET")
    if not key or not secret:
        pytest.skip("set LIVEKIT_SECOND_API_KEY/SECRET to a second key of the project")
    token = Credentials(key, secret).thread_grant_token(
        AgentThreadGrant(scope="user:k2", create=True, write=True, read=True)
    )

    status, payload = await call(http, door, token, "message/send", message("hi"))

    assert status == 200, payload


async def test_a_thread_grant_without_the_endpoint_call_grant_is_refused_at_the_node(
    door: Door, http: aiohttp.ClientSession
) -> None:
    """agentThread alone is data, not reach: the node refuses it before the worker."""
    token = door.raw({"agentThread": {"all": True, "write": True, "create": True, "scope": "u"}})

    status, _ = await call(http, door, token, "message/send", message("hi"))

    assert status == 403


# -- the worker: thread, method and flag


async def test_the_endpoint_call_grant_alone_reads_the_card_but_sends_nothing(
    door: Door, http: aiohttp.ClientSession
) -> None:
    """agentEndpoint reaches the card; a message needs a thread grant (403 from the worker)."""
    token = door.credentials.project_token()

    async with http.get(door.card, headers={"Authorization": f"Bearer {token}"}) as card:
        card_status = card.status
    status, payload = await call(http, door, token, "message/send", message("hi"))

    assert (card_status, status) == (200, 403), payload


async def test_a_scope_grant_creates_a_thread_and_continues_it(
    door: Door, http: aiohttp.ClientSession
) -> None:
    """With no contextId, create opens a thread in the scope; the same grant continues it."""
    token = door.grant(scope="user:42", create=True, write=True, read=True)

    status, payload = await call(http, door, token, "message/send", message("one"))
    thread = task_of(payload)["contextId"]
    again, payload2 = await call(http, door, token, "message/send", message("two", str(thread)))

    assert (status, again) == (200, 200), (payload, payload2)
    assert task_of(payload2)["contextId"] == thread


async def test_a_grant_pinned_to_threads_reaches_only_those(
    door: Door, http: aiohttp.ClientSession
) -> None:
    """threadIds=[A] writes A and is 403 on B."""
    a = await new_thread(http, door, "user:a")
    b = await new_thread(http, door, "user:b")
    token = door.grant(thread_ids=[a], write=True, read=True)

    on_a, payload = await call(http, door, token, "message/send", message("hi", a))
    on_b, _ = await call(http, door, token, "message/send", message("hi", b))

    assert (on_a, on_b) == (200, 403), payload


async def test_each_method_needs_its_flag(door: Door, http: aiohttp.ClientSession) -> None:
    """read alone gets a task but cannot send; create is needed for a new thread."""
    thread = await new_thread(http, door, "user:flags")
    writer = door.grant(thread_ids=[thread], write=True, read=True)
    _, sent = await call(http, door, writer, "message/send", message("hi", thread))
    task_id = task_of(sent)["id"]
    reader = door.grant(thread_ids=[thread], read=True)
    no_create = door.grant(scope="user:flags", write=True, read=True)

    got, payload = await call(http, door, reader, "tasks/get", {"id": task_id, "contextId": thread})
    send, _ = await call(http, door, reader, "message/send", message("hi", thread))
    create, _ = await call(http, door, no_create, "message/send", message("new"))

    assert (got, send, create) == (200, 403, 403), payload


async def test_another_scope_is_refused(door: Door, http: aiohttp.ClientSession) -> None:
    """A thread created under user:1 is 403 to a grant scoped to user:2."""
    thread = await new_thread(http, door, "user:1")
    other = door.grant(scope="user:2", write=True, read=True)

    status, _ = await call(http, door, other, "message/send", message("hi", thread))

    assert status == 403


async def test_a_browser_token_in_the_query_is_authorized_the_same_way(
    door: Door, http: aiohttp.ClientSession
) -> None:
    """access_token in the query is verified by the node and read by the worker."""
    token = door.grant(scope="user:q", create=True, write=True, read=True)
    pinned = door.grant(thread_ids=["AT_notyours"], write=True, read=True)

    status, payload = await call(http, door, token, "message/send", message("hi"), in_query=True)
    refused, _ = await call(http, door, pinned, "message/send", message("hi"), in_query=True)

    assert status == 200, payload
    assert refused == 403
