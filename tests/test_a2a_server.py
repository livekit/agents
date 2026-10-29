"""``a2a_session`` over the wire, in process: a stock a2a-sdk client over HTTP+SSE with and
without the extension header, two workers sharing one local agent-db (``storage_fixture.py``)
for the thin slice, the interrupt, the inbox, a resend and a resubscribe, and the session grant
at the endpoint."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import httpx
import pytest
from a2a.client import ClientConfig, ClientFactory
from a2a.client.card_resolver import A2ACardResolver
from a2a.types import a2a_pb2 as pb
from a2a.utils.constants import TransportProtocol

from livekit.agents import Agent, AgentServer, RunContext
from livekit.agents._generated.storage_class import ThreadInfo
from livekit.agents.a2a import (
    A2ASessionContext,
    AgentThreadGrant,
    Credentials,
    TextPart,
    Unauthorized,
)
from livekit.agents.a2a.endpoint import AGENT_CARD_PATH, A2AEndpoint
from livekit.agents.a2a.extension import (
    CHANNEL,
    EXTENSION_URI,
    INTERRUPTED,
    INTERRUPTION_REASON,
    KIND,
    KIND_CHAT_CTX,
    MERGED_INTO,
    REASON,
    as_dict,
)
from livekit.agents.llm import ChatItem, FunctionToolCall
from livekit.agents.storage import Storage
from livekit.agents.storage.errors import StorageUnavailable
from livekit.agents.storage.resolve import Thread

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit]

PROJECT_ENV = (
    "LIVEKIT_URL",
    "LIVEKIT_AGENT_NAME",
    "LIVEKIT_AGENT_NAME_OVERRIDE",
    "LIVEKIT_AGENT_DEPLOYMENT",
    "LIVEKIT_AGENT_VERSION",
)


@pytest.fixture(autouse=True)
def no_project_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests build their servers in memory and address them as ``agents.test``; a
    developer's exported demo environment (``LIVEKIT_URL`` shapes the card's URLs, the agent
    variables its name and version) must not reach them."""
    for name in PROJECT_ENV:
        monkeypatch.delenv(name, raising=False)


AGENT = "front-desk"
API_KEY = "APItest"
API_SECRET = "secret-secret-secret-secret-secret"
CREDENTIALS = Credentials(API_KEY, API_SECRET)
ENDPOINT = "desk"


# ---------------------------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------------------------


async def echo(ctx: A2ASessionContext) -> str:
    await ctx.update("looking")
    return "reply to: " + ctx.message.text_content


Workers = tuple[Storage, Storage]


def make_server(
    storage: Storage,
    handler: Callable[[A2ASessionContext], Awaitable[Any]] = echo,
    **kw: Any,
) -> tuple[AgentServer, A2AEndpoint]:
    server = AgentServer(storage=storage, agent_name=AGENT, api_key=API_KEY, api_secret=API_SECRET)
    kw.setdefault("description", "Answers front desk questions.")
    endpoint = server.a2a_session(endpoint=ENDPOINT, **kw)(handler)
    return server, endpoint


def http_client(server: AgentServer, **headers: str) -> httpx.AsyncClient:
    headers.setdefault(
        "Authorization",
        "Bearer "
        + CREDENTIALS.thread_grant_token(
            AgentThreadGrant(
                all=True, scope="test", create=True, write=True, read=True, list=True, delete=True
            )
        ),
    )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server._http.app),
        base_url="http://agents.test",
        headers=headers,
        timeout=30.0,
    )


async def stock_client(http: httpx.AsyncClient, *, extensions: str | None = None) -> Any:
    """A vanilla a2a-sdk client over JSON-RPC; ``extensions`` is the header it sends."""
    card = await A2ACardResolver(http, f"http://agents.test/{ENDPOINT}").get_agent_card()
    client = ClientFactory(
        ClientConfig(
            httpx_client=http,
            streaming=True,
            supported_protocol_bindings=[TransportProtocol.JSONRPC],
            accepted_output_modes=["text/plain"],
        )
    ).create(card)
    if extensions:
        from a2a.client import ClientCallContext

        client._lk_context = ClientCallContext(service_parameters={"A2A-Extensions": extensions})
    else:
        client._lk_context = None
    return client


def user_message(
    text: str, message_id: str, *, context_id: str | None = None, **metadata: Any
) -> pb.Message:
    msg = pb.Message(message_id=message_id, role=pb.Role.ROLE_USER, parts=[pb.Part(text=text)])
    if context_id:
        msg.context_id = context_id
    if metadata:
        from livekit.agents.a2a.extension import struct

        msg.metadata.CopyFrom(struct(metadata))
    return msg


async def send(client: Any, message: pb.Message) -> list[Any]:
    request = pb.SendMessageRequest(message=message)
    return [e async for e in client.send_message(request, context=client._lk_context)]


def payloads(events: list[Any]) -> list[Any]:
    return [getattr(e, e.WhichOneof("payload")) for e in events]


async def rpc(
    http: httpx.AsyncClient, method: str, params: Mapping[str, Any], **headers: str
) -> httpx.Response:
    return await http.post(
        f"/{ENDPOINT}/v1",
        json={"jsonrpc": "2.0", "id": method, "method": method, "params": dict(params)},
        headers=headers,
    )


def sse_results(response: httpx.Response) -> list[dict[str, Any]]:
    out = []
    for block in response.text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[len("data: ") :]))
    return out


def wire(
    text: str, message_id: str, *, context_id: str | None = None, **metadata: Any
) -> dict[str, Any]:
    msg: dict[str, Any] = {"messageId": message_id, "role": "ROLE_USER", "parts": [{"text": text}]}
    if context_id:
        msg["contextId"] = context_id
    if metadata:
        msg["metadata"] = metadata
    return msg


def state_of(result: dict[str, Any]) -> str | None:
    update = result.get("statusUpdate")
    return update["status"]["state"] if update else None


async def history_items(storage: Storage, thread_id: str, lane: str) -> list[ChatItem]:
    return (await Thread(storage, thread_id, agent_name=lane).chat_history()).items


async def waiter_of(storage: Storage, thread_id: str, lane: str) -> str:
    db = await storage._open(thread_id)
    try:
        row = await db.lane_info(lane)
    finally:
        db.close()
    return (row.waiter or "") if row is not None else ""


# ---------------------------------------------------------------------------------------------
# the card and the stock client
# ---------------------------------------------------------------------------------------------


async def test_the_card_lists_the_http_interface_and_offers_the_extension(
    thread_storage: Storage,
) -> None:
    server, endpoint = make_server(thread_storage)
    async with http_client(server) as http:
        response = await http.get(f"/{ENDPOINT}{AGENT_CARD_PATH}")
    assert response.status_code == 200
    card = response.json()
    assert card["name"] == ENDPOINT
    bindings = [(i["protocolBinding"], i["url"]) for i in card["supportedInterfaces"]]
    assert bindings == [("JSONRPC", f"http://agents.test/{ENDPOINT}/v1")]
    extensions = {e["uri"]: e for e in card["capabilities"]["extensions"]}
    assert set(extensions) == {EXTENSION_URI}
    # proto3 omits a false bool: not required is said by saying nothing
    assert all(e.get("required", False) is False for e in extensions.values())
    assert [s["id"] for s in card["skills"]] == ["delegate"]
    assert "livekit" in card["securitySchemes"]
    # the url is the server's when it knows its project and deployment
    endpoint.url = "https://acme.livekit.cloud/agents/front-desk/production/desk"
    assert endpoint.card()["supportedInterfaces"][0]["url"] == (
        "https://acme.livekit.cloud/agents/front-desk/production/desk/v1"
    )


async def test_a_stock_client_without_the_header_gets_a_plain_text_agent(
    thread_storage: Storage,
) -> None:
    server, _ = make_server(thread_storage)
    async with http_client(server) as http:
        client = await stock_client(http)
        events = await send(client, user_message("what is the fare", "m1"))
        await client.close()
    items = payloads(events)
    assert isinstance(items[0], pb.Task) and items[0].id.startswith("AST_")
    assert items[0].context_id.startswith("AT_")
    statuses = [p for p in items if isinstance(p, pb.TaskStatusUpdateEvent)]
    assert [s.status.state for s in statuses] == [
        pb.TaskState.TASK_STATE_WORKING,
        pb.TaskState.TASK_STATE_COMPLETED,
    ]
    assert statuses[0].status.message.parts[0].text == "looking"
    artifacts = [p for p in items if isinstance(p, pb.TaskArtifactUpdateEvent)]
    assert [a.artifact.parts[0].text for a in artifacts] == ["reply to: what is the fare"]
    assert artifacts[0].last_chunk is True
    # nothing profile-specific rides on a plain request
    for s in statuses:
        assert not as_dict(s.status.message.metadata)
        assert all(p.WhichOneof("content") == "text" for p in s.status.message.parts)


async def test_the_header_is_echoed(thread_storage: Storage) -> None:
    async def handler(ctx: A2ASessionContext) -> str:
        await ctx.update("one moment")
        return "Goodbye."

    server, _ = make_server(thread_storage, handler)
    async with http_client(server) as http:
        response = await rpc(
            http,
            "message/stream",
            {"message": wire("that is all", "m1")},
            **{"A2A-Extensions": EXTENSION_URI},
        )
        assert response.headers["a2a-extensions"] == EXTENSION_URI
        results = [r["result"] for r in sse_results(response)]
        working = next(r["statusUpdate"] for r in results if state_of(r) == "TASK_STATE_WORKING")
        assert "metadata" not in working["status"]["message"]

        # the same turn without the header: the same text, no echo
        response = await rpc(http, "message/stream", {"message": wire("that is all", "m2")})
        assert "a2a-extensions" not in response.headers
        results = [r["result"] for r in sse_results(response)]
        artifact = next(r["artifactUpdate"] for r in results if "artifactUpdate" in r)
        assert artifact["artifact"]["parts"][0]["text"] == "Goodbye."


async def test_errors_map_as_the_plan_lists_them(thread_storage: Storage) -> None:
    server, _ = make_server(thread_storage)
    async with http_client(server) as http:
        # a message with no messageId
        r = await rpc(
            http, "message/send", {"message": {"role": "ROLE_USER", "parts": [{"text": "x"}]}}
        )
        assert r.json()["error"]["code"] == -32602
        # a task id alone names no session in v0
        r = await rpc(http, "tasks/get", {"id": "AST_nope"})
        assert r.json()["error"]["code"] == -32001
        first = (await rpc(http, "message/send", {"message": wire("hello", "m1")})).json()[
            "result"
        ]["task"]
        ctx_id, task_id = first["contextId"], first["id"]
        # a message to a terminal task
        r = await rpc(
            http,
            "message/send",
            {"message": {**wire("more", "m2", context_id=ctx_id), "taskId": task_id}},
        )
        assert r.json()["error"]["code"] == -32004
        # cancel on a terminal task
        r = await rpc(http, "tasks/cancel", {"id": task_id, "contextId": ctx_id})
        assert r.json()["error"]["code"] == -32002
        # an unknown method, a malformed envelope
        r = await rpc(http, "tasks/explode", {})
        assert r.json()["error"]["code"] == -32601
        r = await http.post(f"/{ENDPOINT}/v1", content=b"{not json")
        assert r.json()["error"]["code"] == -32700
        # an unknown session fails the task thread-not-found (5.12)
        r = await rpc(http, "message/send", {"message": wire("hi", "m9", context_id="AT_unknown")})
        task = r.json()["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_FAILED"
        assert task["metadata"][REASON] == "thread-not-found"


# ---------------------------------------------------------------------------------------------
# two workers, one backend: the demo checks (plan 10.1)
# ---------------------------------------------------------------------------------------------


async def test_thin_slice_across_two_workers(thread_storages: Workers) -> None:
    a, b = thread_storages
    seen: list[list[str]] = []

    async def handler(ctx: A2ASessionContext) -> str:
        seen.append([i.text_content or "" for i in ctx.message_history.items])
        return "seen " + str(len(ctx.message_history.items))

    server_a, _ = make_server(a, handler)
    server_b, _ = make_server(b, handler)
    async with http_client(server_a) as http_a, http_client(server_b) as http_b:
        client_a = await stock_client(http_a)
        first = payloads(await send(client_a, user_message("hello", "m1")))
        ctx_id = first[0].context_id
        assert ctx_id.startswith("AT_")
        client_b = await stock_client(http_b)
        second = payloads(await send(client_b, user_message("again", "m2", context_id=ctx_id)))
        assert second[0].context_id == ctx_id
        listed = await client_b.list_tasks(pb.ListTasksRequest(context_id=ctx_id))
        await client_a.close()
        await client_b.close()
    assert seen == [[], ["hello", "seen 0"]]
    assert [t.id for t in listed.tasks] == [second[0].id, first[0].id]
    assert all(t.status.state == pb.TaskState.TASK_STATE_COMPLETED for t in listed.tasks)


def _blocking_agent(tool_entered: asyncio.Event, release: asyncio.Event) -> Agent:
    front = Agent(
        instructions="front desk",
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="first",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[FunctionToolCall(name="lookup", arguments="{}", call_id="c1")],
                ),
                FakeLLMResponse(input="found", content="never sent", ttft=0, duration=0),
                FakeLLMResponse(input="second", content="both answered", ttft=0, duration=0),
            ]
        ),
    )

    @front.tool
    async def lookup(ctx: RunContext) -> str:
        """Look the caller up."""
        tool_entered.set()
        await release.wait()
        return "found"

    return front


async def test_interrupt_two_streams_within_a_second_on_two_workers(
    thread_storages: Workers,
) -> None:
    a, b = thread_storages
    tool_entered, release = asyncio.Event(), asyncio.Event()
    front = _blocking_agent(tool_entered, release)

    async def handler(ctx: A2ASessionContext) -> str | None:
        return (await front.run(ctx.message, message_history=ctx.message_history)).output

    server_a, _ = make_server(a, handler)
    server_b, _ = make_server(b, handler)
    ctx_id = (await a.threads.create_thread()).thread_id
    async with http_client(server_a) as http_a, http_client(server_b) as http_b:
        first = asyncio.create_task(
            rpc(http_a, "message/stream", {"message": wire("first", "m1", context_id=ctx_id)})
        )
        await asyncio.wait_for(tool_entered.wait(), 5)
        second = asyncio.create_task(
            rpc(http_b, "message/stream", {"message": wire("second", "m2", context_id=ctx_id)})
        )
        for _ in range(200):
            if (await waiter_of(b, ctx_id, AGENT)).startswith("worker-b/"):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("the waiter was never set")
        release.set()
        r1, r2 = await asyncio.gather(first, second)

    results_1 = [r["result"] for r in sse_results(r1)]
    results_2 = [r["result"] for r in sse_results(r2)]
    final_1 = results_1[-1]["statusUpdate"]
    assert final_1["status"]["state"] == "TASK_STATE_CANCELED"
    meta = final_1["metadata"]
    assert meta[INTERRUPTED] is True and meta[INTERRUPTION_REASON] == "new-message"
    assert not any("artifactUpdate" in r for r in results_1)
    artifacts = [r["artifactUpdate"]["artifact"] for r in results_2 if "artifactUpdate" in r]
    # streamed in pieces as the model writes it, the last piece empty
    assert "".join(p["text"] for a in artifacts for p in a.get("parts", [])) == "both answered"
    assert results_2[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
    items = await history_items(a, ctx_id, AGENT)
    assert [i.type for i in items] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
        "message",
    ]


async def test_an_inbox_burst_on_one_worker_is_one_turn_too(
    thread_storage: Storage,
) -> None:
    """Two requests of one worker on one lane hold two leases, not one re-taken lease."""
    storage = thread_storage
    batches: list[list[str]] = []

    async def handler(ctx: A2ASessionContext) -> str:
        batches.append([m.parts[0].text for m in ctx.messages])
        await asyncio.sleep(0.5)
        return "one reply"

    from datetime import timedelta

    server, _ = make_server(storage, handler, debounce=timedelta(seconds=0.3))
    ctx_id = (await storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = asyncio.create_task(
            rpc(
                http,
                "message/send",
                {"message": wire("where", "m1", context_id=ctx_id, **{CHANNEL: "sms"})},
            )
        )
        await asyncio.sleep(0.05)
        second = asyncio.create_task(
            rpc(
                http,
                "message/send",
                {"message": wire("are you", "m2", context_id=ctx_id, **{CHANNEL: "sms"})},
            )
        )
        r1, r2 = await asyncio.gather(first, second)
    task_1, task_2 = r1.json()["result"]["task"], r2.json()["result"]["task"]
    assert batches == [["where", "are you"]]
    assert task_1["status"]["state"] == "TASK_STATE_COMPLETED" and "artifacts" not in task_1
    assert task_1["metadata"][MERGED_INTO] == task_2["id"]
    assert task_2["artifacts"][0]["parts"][0]["text"] == "one reply"


async def test_a_resent_message_id_returns_the_existing_task(
    thread_storage: Storage,
) -> None:
    server, _ = make_server(thread_storage)
    async with http_client(server) as http:
        first = (await rpc(http, "message/send", {"message": wire("hi", "m1")})).json()["result"][
            "task"
        ]
        again = (
            await rpc(
                http, "message/send", {"message": wire("hi", "m1", context_id=first["contextId"])}
            )
        ).json()["result"]["task"]
        listed = (await rpc(http, "tasks/list", {"contextId": first["contextId"]})).json()[
            "result"
        ]["tasks"]
    assert again["id"] == first["id"]
    assert again["artifacts"][0]["parts"][0]["text"] == "reply to: hi"
    assert [t["id"] for t in listed] == [first["id"]]
    storage = server.storage
    assert storage is not None
    items = await history_items(storage, first["contextId"], AGENT)
    assert [i.text_content for i in items] == ["hi", "reply to: hi"]


async def test_resubscribe_replays_after_the_first_stream_is_dropped(
    thread_storages: Workers,
) -> None:
    a, b = thread_storages
    release = asyncio.Event()

    async def handler(ctx: A2ASessionContext) -> str:
        await ctx.update("working on it")
        await release.wait()
        return "done"

    server_a, endpoint_a = make_server(a, handler)
    server_b, _ = make_server(b, handler)
    ctx_id = (await a.threads.create_thread()).thread_id

    # the first client reads the acknowledgment and drops the stream
    principal = CREDENTIALS._principal(
        {
            "Authorization": "Bearer "
            + CREDENTIALS.thread_grant_token(AgentThreadGrant(all=True, read=True, write=True))
        },
        agent_name=AGENT,
    )
    stream = await endpoint_a.service.dispatch(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "initial",
                "method": "message/stream",
                "params": {"message": wire("slow", "m1", context_id=ctx_id)},
            }
        ).encode(),
        principal,
    )
    ack = await stream.next()
    assert ack is not None
    task_id = json.loads(ack.json)["result"]["task"]["id"]
    await stream.aclose()

    async with http_client(server_b) as http_b:
        # a live follow from the other worker while the turn still runs
        following = asyncio.create_task(
            rpc(http_b, "tasks/resubscribe", {"id": task_id, "contextId": ctx_id})
        )
        await asyncio.sleep(0.1)
        release.set()
        response = await asyncio.wait_for(following, 10)
        results = [r["result"] for r in sse_results(response)]
        # the Task object first, as the spec requires, then the frames from the start
        assert list(results[0]) == ["task"] and results[0]["task"]["id"] == task_id
        kinds = [list(r)[0] for r in results[1:]]
        assert kinds == ["task", "statusUpdate", "artifactUpdate", "statusUpdate"]
        assert results[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
        # after the fact: the Task object, then the whole stream from its start, not followed
        later = await rpc(http_b, "tasks/resubscribe", {"id": task_id, "contextId": ctx_id})
        results = [r["result"] for r in sse_results(later)]
        assert [list(r)[0] for r in results] == [
            "task",
            "task",
            "statusUpdate",
            "artifactUpdate",
            "statusUpdate",
        ]
        assert results[0]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
        assert results[1]["task"]["id"] == task_id


async def test_the_stream_completes_when_the_first_client_vanishes_without_closing(
    thread_storages: Workers,
) -> None:
    """A browser reload drops the response mid-turn: the ASGI send fails and the generator
    is left suspended, never closed. The durable stream must still receive the rest of the
    turn, or a tasks/resubscribe from another worker ends at the last frame the first client
    saw."""
    a, b = thread_storages
    release = asyncio.Event()

    async def handler(ctx: A2ASessionContext) -> str:
        await ctx.update("working on it")
        await release.wait()
        return "done"

    server_a, endpoint_a = make_server(a, handler)
    server_b, _ = make_server(b, handler)
    ctx_id = (await a.threads.create_thread()).thread_id

    principal = CREDENTIALS._principal(
        {
            "Authorization": "Bearer "
            + CREDENTIALS.thread_grant_token(AgentThreadGrant(all=True, read=True, write=True))
        },
        agent_name=AGENT,
    )
    stream = await endpoint_a.service.dispatch(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "initial",
                "method": "message/stream",
                "params": {"message": wire("slow", "m1", context_id=ctx_id)},
            }
        ).encode(),
        principal,
    )
    ack = await stream.next()
    assert ack is not None
    task_id = json.loads(ack.json)["result"]["task"]["id"]
    # held, never closed: the suspended generator of a response whose client is gone

    async with http_client(server_b) as http_b:
        following = asyncio.create_task(
            rpc(http_b, "tasks/resubscribe", {"id": task_id, "contextId": ctx_id})
        )
        await asyncio.sleep(0.1)
        release.set()
        response = await asyncio.wait_for(following, 10)
        results = [r["result"] for r in sse_results(response)]
        assert results[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
    del stream


async def test_tasks_cancel_ends_a_running_turn_at_its_checkpoint(
    thread_storage: Storage,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    front = _blocking_agent(entered, release)

    async def handler(ctx: A2ASessionContext) -> str | None:
        return (await front.run(ctx.message, message_history=ctx.message_history)).output

    server, _ = make_server(thread_storage, handler)
    ctx_id = (await thread_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        running = asyncio.create_task(
            rpc(http, "message/stream", {"message": wire("first", "m1", context_id=ctx_id)})
        )
        await asyncio.wait_for(entered.wait(), 5)
        listed = (await rpc(http, "tasks/list", {"contextId": ctx_id})).json()["result"]["tasks"]
        (task,) = listed
        assert task["status"]["state"] == "TASK_STATE_WORKING"
        canceled = await rpc(
            http,
            "tasks/cancel",
            {
                "id": task["id"],
                "contextId": ctx_id,
                "metadata": {f"{EXTENSION_URI}/reason": "user_interrupted"},
            },
        )
        assert canceled.json()["result"]["status"]["state"] == "TASK_STATE_CANCELED"
        release.set()
        response = await running
    results = [r["result"] for r in sse_results(response)]
    final = results[-1]["statusUpdate"]
    assert final["status"]["state"] == "TASK_STATE_CANCELED"
    # a client's cancel, not the server's interrupt
    assert INTERRUPTED not in final.get("metadata", {})
    assert final["metadata"][REASON] == "user_interrupted"


# ---------------------------------------------------------------------------------------------
# the session grant at the endpoint (plan 4.3.2)
# ---------------------------------------------------------------------------------------------


def grant_token(scope: str, **flags: bool) -> str:
    flags = flags or {"create": True, "write": True, "read": True, "list": True}
    return CREDENTIALS.thread_grant_token(AgentThreadGrant(scope=scope, **flags))


async def test_the_grant_creates_in_its_scope_and_refuses_another_scopes_session(
    thread_storage: Storage,
) -> None:
    server, _ = make_server(thread_storage)
    storage = server.storage
    assert storage is not None
    async with http_client(server) as http:
        alice = {"Authorization": f"Bearer {grant_token('user:alice')}"}
        bob = {"Authorization": f"Bearer {grant_token('user:bob')}"}
        created = (await rpc(http, "message/send", {"message": wire("hi", "m1")}, **alice)).json()[
            "result"
        ]["task"]
        ctx_id = created["contextId"]
        info = await storage.threads.get_thread(ctx_id)
        assert info.scope == "user:alice"
        # Scope and explicit ids both authorize existing threads.
        r = await rpc(http, "message/send", {"message": wire("hi", "m3", context_id=ctx_id)}, **bob)
        assert r.status_code == 403
        r = await rpc(http, "tasks/list", {"contextId": ctx_id}, **alice)
        assert r.status_code == 200
        listed = AgentThreadGrant(thread_ids=(ctx_id,), create=True, write=True, read=True)
        by_id = {"Authorization": f"Bearer {CREDENTIALS.thread_grant_token(listed)}"}
        r = await rpc(http, "tasks/list", {"contextId": ctx_id}, **by_id)
        assert r.status_code == 200 and len(r.json()["result"]["tasks"]) == 1
        # a grant with no scope cannot create, a flag the grant lacks, a token nobody signed
        assert (
            await rpc(http, "message/send", {"message": wire("hi", "m4")}, **by_id)
        ).status_code == 403
        read_only = {"Authorization": f"Bearer {grant_token('user:alice', read=True)}"}
        assert (
            await rpc(
                http, "message/send", {"message": wire("hi", "m5", context_id=ctx_id)}, **read_only
            )
        ).status_code == 403
        forged = Credentials("APIother", API_SECRET).thread_grant_token(
            AgentThreadGrant(scope="user:alice", write=True, create=True)
        )
        assert (
            await rpc(
                http,
                "message/send",
                {"message": wire("hi", "m6")},
                Authorization=f"Bearer {forged}",
            )
        ).status_code == 401


def test_the_nodes_verdict_is_the_project_credential_and_a_bearer_is_verified_here() -> None:
    """Through a node the project credential is stripped and arrives as ``authorized``; a thread
    grant arrives as the bearer it is and the worker verifies it, as it verifies every token
    when reached directly."""
    bob = {"Authorization": f"Bearer {grant_token('user:bob')}"}
    principal = CREDENTIALS._principal(bob, agent_name=AGENT, node_authorized=True)
    assert principal.kind == "grant" and principal.grant is not None
    assert principal.grant.scope == "user:bob" and principal.grant.write
    # the node's verdict without a bearer is the project credential; nothing is anonymous
    assert CREDENTIALS._principal({}, agent_name=AGENT, node_authorized=True).kind == "project"
    with pytest.raises(Unauthorized):
        CREDENTIALS._principal({}, agent_name=AGENT)
    # a grant naming neither a scope nor a thread is refused
    bare = {"Authorization": f"Bearer {grant_token('', write=True)}"}
    with pytest.raises(Unauthorized):
        CREDENTIALS._principal(bare, agent_name=AGENT)
    # the checks, as protocol's: scope covers create and list, ids or all cover a thread,
    # update needs all
    thread = ThreadInfo(
        thread_id="AT_1",
        scope="user:bob",
        attributes={},
        created_at=0,
        last_active=0,
        expires_at=0,
        idle_ttl_seconds=0,
        database_id="",
    )
    listed = AgentThreadGrant(thread_ids=("AT_1",), read=True)
    assert listed.can_read(thread) and not listed.can_write(thread)
    assert not listed.can_create("user:bob")
    scoped = AgentThreadGrant(scope="user:bob", create=True, write=True)
    assert scoped.can_create("user:bob") and not scoped.can_create("user:x")
    assert scoped.can_write(thread)
    everything = AgentThreadGrant(all=True, create=True, write=True)
    assert not everything.can_create("user:x") and everything.can_write(thread)


async def test_endpoint_auth_and_thread_authority_are_checked_on_every_request(
    thread_storage: Storage,
) -> None:
    seen: list[str] = []

    def auth(connection: Any) -> bool:
        seen.append(connection.headers.get("x-api-key", ""))
        return connection.headers.get("x-api-key") == "letmein"

    server, _ = make_server(thread_storage, auth=auth)
    async with http_client(server, Authorization="") as http:
        # The developer check and thread authority are independent.
        r = await rpc(
            http, "message/send", {"message": wire("hi", "m1")}, **{"X-Api-Key": "letmein"}
        )
        assert r.status_code == 401
        token = CREDENTIALS.project_token()
        r = await rpc(
            http, "message/send", {"message": wire("hi", "m2")}, Authorization=f"Bearer {token}"
        )
        assert r.status_code == 401
        endpoint_only = await rpc(
            http,
            "message/send",
            {"message": wire("hi", "endpoint-only")},
            Authorization=f"Bearer {token}",
            **{"X-Api-Key": "letmein"},
        )
        assert endpoint_only.status_code == 403
        r = await rpc(
            http,
            "message/send",
            {"message": wire("hi", "m3")},
            Authorization="Bearer " + grant_token("user:alice"),
            **{"X-Api-Key": "letmein"},
        )
        assert (
            r.status_code == 200
            and r.json()["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
        )
        assert (await http.get(f"/{ENDPOINT}{AGENT_CARD_PATH}")).status_code == 401
        response = await http.get(
            f"/{ENDPOINT}{AGENT_CARD_PATH}",
            headers={
                "Authorization": "Bearer " + grant_token("user:alice"),
                "X-Api-Key": "letmein",
            },
        )
        assert response.status_code == 200 and response.json()["securityRequirements"]
    assert seen == ["letmein", "", "letmein", "letmein", "", "letmein"]


async def test_a_delegation_part_and_kind_reach_the_handler(
    thread_storage: Storage,
) -> None:
    seen: dict[str, Any] = {}

    async def handler(ctx: A2ASessionContext) -> str:
        seen["caller_history"] = ctx.caller_history
        seen["channel"] = ctx.messages[0].channel
        seen["parts"] = ctx.messages[0].parts
        return "ok"

    server, _ = make_server(thread_storage, handler)
    from google.protobuf import json_format

    # a2a.md#conversation-data-part: the narrower schema, not the stored chat context's
    history = {
        "items": [
            {
                "id": "i1",
                "type": "message",
                "role": "user",
                "content": ["earlier"],
                "created_at": 1.0,
            }
        ]
    }
    msg = {
        "messageId": "m1",
        "role": "ROLE_USER",
        "parts": [
            {"text": "what is the fare"},
            {"data": history, "metadata": {KIND: KIND_CHAT_CTX}},
        ],
        "metadata": {KIND: "delegation"},
    }
    async with http_client(server) as http:
        client = await stock_client(http, extensions=EXTENSION_URI)
        events = await send(client, json_format.ParseDict(msg, pb.Message()))
        await client.close()
    assert payloads(events)[-1].status.state == pb.TaskState.TASK_STATE_COMPLETED
    assert seen["channel"] == "delegation"
    assert seen["caller_history"] is not None
    assert [i.text_content for i in seen["caller_history"].items] == ["earlier"]
    # the conversation part is not the turn's text
    assert seen["parts"] == [TextPart(text="what is the fare")]


async def test_a_storage_error_mid_turn_fails_the_task_storage_unavailable(
    thread_storages: Workers,
) -> None:
    """docs/protocol/a2a.md: the turn's save could not be written; the task fails with the reason, and
    the client resends."""
    a, _ = thread_storages

    async def handler(ctx: A2ASessionContext) -> str:
        raise StorageUnavailable("the database went away")

    server, _ = make_server(a, handler)
    ctx_id = (await a.threads.create_thread()).thread_id
    async with http_client(server) as http:
        r = await rpc(http, "message/send", {"message": wire("hi", "m1", context_id=ctx_id)})
    task = r.json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_FAILED"
    reason = task["status"]["message"]["metadata"][REASON]
    assert reason == "storage-unavailable"


async def test_each_request_supplies_its_own_application_context(thread_storage: Storage) -> None:
    import contextvars

    label = contextvars.ContextVar("request-label", default="missing")
    seen: list[str] = []

    async def handler(ctx: A2ASessionContext) -> str:
        seen.append(label.get())
        return label.get()

    server, _ = make_server(thread_storage, handler)
    async with http_client(server) as http:
        for value in ("first", "second"):
            token = label.set(value)
            try:
                result = await rpc(http, "message/send", {"message": wire("hello", value)})
                assert result.json()["result"]["task"]["artifacts"][0]["parts"][0]["text"] == value
            finally:
                label.reset(token)
    assert seen == ["first", "second"]


async def test_application_typed_output_crosses_the_native_door(thread_storage: Storage) -> None:
    from pydantic import BaseModel

    class Answer(BaseModel):
        reservation: str

    async def handler(ctx: A2ASessionContext) -> Answer:
        return Answer(reservation="R42")

    server, _ = make_server(thread_storage, handler)
    async with http_client(server) as http:
        response = await rpc(http, "message/send", {"message": wire("book", "typed")})
    task = response.json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["parts"][0]["data"] == {"reservation": "R42"}


async def test_handler_transitions_answer_the_task(thread_storage: Storage) -> None:
    from livekit.agents import Completed, Failed

    async def handler(ctx: A2ASessionContext) -> Completed[str] | Failed:
        if ctx.message.text_content == "fail":
            return Failed("no room on that date")
        return Completed("booked")

    server, _ = make_server(thread_storage, handler)
    async with http_client(server) as http:
        done = (await rpc(http, "message/send", {"message": wire("book", "ok")})).json()
        failed = (await rpc(http, "message/send", {"message": wire("fail", "no")})).json()
    done_task, failed_task = done["result"]["task"], failed["result"]["task"]
    assert done_task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert done_task["artifacts"][0]["parts"][0]["text"] == "booked"
    assert failed_task["status"]["state"] == "TASK_STATE_FAILED"


async def test_payload_budget_refuses_before_durable_receipt(thread_storage: Storage) -> None:
    calls = 0

    async def handler(ctx: A2ASessionContext) -> str:
        nonlocal calls
        calls += 1
        return "unexpected"

    server, endpoint = make_server(thread_storage, handler)
    native = await endpoint.service.native()
    async with http_client(server) as http:
        response = await http.post(f"/{ENDPOINT}/v1", content=b" " * (native.max_payload_bytes + 1))
    assert response.json()["error"]["code"] == -32602
    assert calls == 0
    assert (await thread_storage.threads.list_threads("test")).threads == []
