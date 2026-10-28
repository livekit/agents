"""``a2a_session`` over the wire, in process: a stock a2a-sdk client over HTTP+SSE with and
without the extension header, two workers sharing one local agent-db (``agentdb_fixture.py``)
for the thin slice, the interrupt, the inbox, a resend and a resubscribe, and the session grant
at the endpoint."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from a2a.types import a2a_pb2 as pb

from livekit.agents import Agent, RunContext
from livekit.agents._generated.storage_threads import ThreadInfo
from livekit.agents.a2a import (
    AGENT_CARD_PATH,
    EXTENSION_URI,
    A2ASessionContext,
    AgentThreadGrant,
    Credentials,
    Extensions,
    Unauthorized,
)
from livekit.agents.a2a.codec import TaskUpdate, metadata_from_wire, updates_from_events
from livekit.agents.a2a.extension import (
    INTERRUPTED,
    INTERRUPTION_REASON,
    KIND,
    MERGED_INTO,
    REASON,
    as_dict,
)
from livekit.agents.llm import ChatItem, FunctionToolCall
from livekit.agents.storage import AgentDBStorage, StorageUnavailable
from livekit.agents.storage.resolve import Thread

from .a2a_driver import (
    AGENT,
    API_SECRET,
    CREDENTIALS,
    ENDPOINT,
    Door,
    Reply,
    ScriptedLLM,
    call,
    http_client,
    make_server,
    no_project_env,  # noqa: F401  (autouse: the demo environment must not shape the cards)
    output_of,
    payloads,
    rpc,
    send,
    sse_results,
    state_of,
    stock_client,
    user_message,
    wire,
)
from .agentdb_fixture import past_the_fresh_acquire
from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit]

Workers = tuple[AgentDBStorage, AgentDBStorage]


async def history_items(storage: AgentDBStorage, thread_id: str, lane: str) -> list[ChatItem]:
    return (await Thread(storage, thread_id, agent_name=lane).chat_history()).items


async def waiter_of(storage: AgentDBStorage, thread_id: str, lane: str) -> str:
    db = await storage.open(thread_id)
    try:
        row = await db.lane(lane)
    finally:
        db.close()
    return (row.waiter or "") if row is not None else ""


# ---------------------------------------------------------------------------------------------
# the card and the stock client
# ---------------------------------------------------------------------------------------------


async def test_the_card_lists_the_http_interface_and_offers_the_extension(
    agentdb_storage: AgentDBStorage,
) -> None:
    server, endpoint = make_server(agentdb_storage)
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
    agentdb_storage: AgentDBStorage,
) -> None:
    server, _ = make_server(agentdb_storage)
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


async def test_the_header_is_echoed(agentdb_storage: AgentDBStorage) -> None:
    async def handler(ctx: A2ASessionContext[Any]) -> str:
        await ctx.update("one moment")
        return "Goodbye."

    server, _ = make_server(agentdb_storage, handler)
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


async def _updates(results: list[dict[str, Any]]) -> list[TaskUpdate]:
    from google.protobuf import json_format

    async def events() -> AsyncIterator[Any]:
        for r in results:
            if "result" in r:
                yield json_format.ParseDict(
                    r["result"], pb.StreamResponse(), ignore_unknown_fields=True
                )

    return [u async for u in updates_from_events(events())]


async def test_errors_map_as_the_plan_lists_them(agentdb_storage: AgentDBStorage) -> None:
    server, _ = make_server(agentdb_storage)
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
        assert metadata_from_wire(task["metadata"])["livekit/reason"] == "thread-not-found"


# ---------------------------------------------------------------------------------------------
# two workers, one backend: the demo checks (plan 10.1)
# ---------------------------------------------------------------------------------------------


async def test_thin_slice_across_two_workers(agentdb_workers: Workers) -> None:
    a, b = agentdb_workers
    seen: list[list[str]] = []

    async def handler(ctx: A2ASessionContext[Any]) -> str:
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
    assert [t.id for t in listed.tasks] == [first[0].id, second[0].id]
    assert all(t.status.state == pb.TaskState.TASK_STATE_COMPLETED for t in listed.tasks)


def _blocking_agent(tool_entered: asyncio.Event, release: asyncio.Event) -> Agent:
    front = Agent(instructions="front desk")

    @front.tool
    async def lookup(ctx: RunContext) -> str:
        """Look the caller up."""
        tool_entered.set()
        await release.wait()
        return "found"

    front.update_options(
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
        )
    )
    return front


async def test_interrupt_two_streams_within_a_second_on_two_workers(
    agentdb_workers: Workers,
) -> None:
    a, b = agentdb_workers
    tool_entered, release = asyncio.Event(), asyncio.Event()
    front = _blocking_agent(tool_entered, release)

    async def handler(ctx: A2ASessionContext[Any]) -> str | None:
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
        await past_the_fresh_acquire()
        release.set()
        r1, r2 = await asyncio.gather(first, second)

    results_1 = [r["result"] for r in sse_results(r1)]
    results_2 = [r["result"] for r in sse_results(r2)]
    final_1 = results_1[-1]["statusUpdate"]
    assert final_1["status"]["state"] == "TASK_STATE_CANCELED"
    meta = metadata_from_wire(final_1["metadata"])
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


async def test_inbox_sms_burst_is_one_reply_and_a_merge(agentdb_workers: Workers) -> None:
    a, b = agentdb_workers
    batches: list[list[str]] = []

    async def handler(ctx: A2ASessionContext[Any]) -> str:
        batches.append([m.parts[0]["text"] for m in ctx.messages])
        return "one reply"

    from datetime import timedelta

    server_a, _ = make_server(a, handler, debounce=timedelta(seconds=0.3))
    server_b, _ = make_server(b, handler, debounce=timedelta(seconds=0.3))
    ctx_id = (await a.threads.create_thread()).thread_id
    async with http_client(server_a) as http_a, http_client(server_b) as http_b:
        first = asyncio.create_task(
            rpc(
                http_a,
                "message/send",
                {"message": wire("where", "m1", context_id=ctx_id, **{"livekit/channel": "sms"})},
            )
        )
        await asyncio.sleep(0.05)
        second = asyncio.create_task(
            rpc(
                http_b,
                "message/send",
                {"message": wire("are you", "m2", context_id=ctx_id, **{"livekit/channel": "sms"})},
            )
        )
        r1, r2 = await asyncio.gather(first, second)
    task_1, task_2 = r1.json()["result"]["task"], r2.json()["result"]["task"]
    assert batches == [["where", "are you"]]
    assert task_1["status"]["state"] == "TASK_STATE_COMPLETED" and "artifacts" not in task_1
    assert metadata_from_wire(task_1["metadata"])[MERGED_INTO] == task_2["id"]
    assert task_2["artifacts"][0]["parts"][0]["text"] == "one reply"


@pytest.mark.rule("R10")
async def test_an_inbox_burst_on_one_worker_is_one_turn_too(door: Callable[..., Door]) -> None:
    """Two requests of one worker on one lane hold two leases, not one re-taken lease."""
    batches: list[list[str]] = []

    async def handler(ctx: A2ASessionContext[Any]) -> str:
        batches.append([m.parts[0]["text"] for m in ctx.messages])
        await asyncio.sleep(0.5)
        return "one reply"

    from datetime import timedelta

    desk = door(handler, debounce=timedelta(seconds=0.3))
    ctx_id = await desk.thread()
    sms = {"livekit/channel": "sms"}
    first = asyncio.create_task(desk.send("where", context_id=ctx_id, **sms))
    await asyncio.sleep(0.05)
    second = asyncio.create_task(desk.send("are you", context_id=ctx_id, **sms))
    r1, r2 = await asyncio.gather(first, second)
    assert batches == [["where", "are you"]]
    assert r1.state == "completed" and r1.answer is None
    assert r1.metadata[MERGED_INTO] == r2.task["id"]
    assert r2.answer == "one reply"


@pytest.mark.rule("R3")
async def test_a_resent_message_id_returns_the_existing_task(door: Callable[..., Door]) -> None:
    runs: list[str] = []

    async def handler(ctx: A2ASessionContext[Any]) -> str:
        runs.append(ctx.message.text_content or "")
        return "reply to: " + (ctx.message.text_content or "")

    desk = door(handler)
    first = await desk.send("hi", message_id="m1")
    ctx_id = first.task["contextId"]
    again = await desk.send("hi", message_id="m1", context_id=ctx_id)
    assert again.task["id"] == first.task["id"]
    assert again.answer == "reply to: hi"
    assert [t["id"] for t in await desk.list_tasks(ctx_id)] == [first.task["id"]]
    assert runs == ["hi"], "nothing runs twice"


async def test_resubscribe_replays_after_the_first_stream_is_dropped(
    agentdb_workers: Workers,
) -> None:
    a, b = agentdb_workers
    release = asyncio.Event()

    async def handler(ctx: A2ASessionContext[Any]) -> str:
        await ctx.update("working on it")
        await release.wait()
        return "done"

    server_a, endpoint_a = make_server(a, handler)
    server_b, _ = make_server(b, handler)
    ctx_id = (await a.threads.create_thread()).thread_id

    # the first client reads the acknowledgment and drops the stream
    from livekit.agents.a2a.methods import CallContext

    stream = endpoint_a.service.stream(
        {"message": wire("slow", "m1", context_id=ctx_id)},
        CallContext(extensions=Extensions(livekit=True)),
    )
    ack = await stream.__anext__()
    task_id = ack.id
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
    agentdb_workers: Workers,
) -> None:
    """A browser reload drops the response mid-turn: the ASGI send fails and the generator
    is left suspended, never closed. The durable stream must still receive the rest of the
    turn, or a tasks/resubscribe from another worker ends at the last frame the first client
    saw."""
    a, b = agentdb_workers
    release = asyncio.Event()

    async def handler(ctx: A2ASessionContext[Any]) -> str:
        await ctx.update("working on it")
        await release.wait()
        return "done"

    server_a, endpoint_a = make_server(a, handler)
    server_b, _ = make_server(b, handler)
    ctx_id = (await a.threads.create_thread()).thread_id

    from livekit.agents.a2a.methods import CallContext

    stream = endpoint_a.service.stream(
        {"message": wire("slow", "m1", context_id=ctx_id)},
        CallContext(extensions=Extensions(livekit=True)),
    )
    ack = await stream.__anext__()
    task_id = ack.id
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


@pytest.mark.rule("R12")
async def test_tasks_cancel_ends_a_running_turn_at_its_boundary(door: Callable[..., Door]) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    model = ScriptedLLM(
        Reply(tool_calls=[call("lookup", "c1")]), Reply("never sent", when=output_of("c1"))
    )
    front = Agent(instructions="front desk", llm=model)

    @front.tool
    async def lookup(ctx: RunContext) -> str:
        """Look the caller up."""
        entered.set()
        await release.wait()
        return "found"

    async def handler(ctx: A2ASessionContext[Any]) -> str | None:
        return (await front.run(ctx.message, message_history=ctx.message_history)).output

    desk = door(handler)
    ctx_id = await desk.thread()
    running = asyncio.create_task(desk.send("first", context_id=ctx_id))
    await asyncio.wait_for(entered.wait(), 5)
    (task,) = await desk.list_tasks(ctx_id)
    assert task["status"]["state"] == "TASK_STATE_WORKING"
    canceled = await desk.cancel(task["id"], ctx_id, **{"livekit/reason": "user_interrupted"})
    assert canceled["result"]["status"]["state"] == "TASK_STATE_CANCELED"
    await past_the_fresh_acquire()
    release.set()
    sent = await running
    final = sent.frames[-1]["statusUpdate"]
    assert final["status"]["state"] == "TASK_STATE_CANCELED" and sent.state == "canceled"
    # a client's cancel, not the server's interrupt
    assert INTERRUPTED not in metadata_from_wire(final.get("metadata", {}))
    assert metadata_from_wire(final["metadata"])["livekit/reason"] == "user_interrupted"
    assert len(model.requests) == 1, "the tool's output never reached the model"


# ---------------------------------------------------------------------------------------------
# the session grant at the endpoint (plan 4.3.2)
# ---------------------------------------------------------------------------------------------


def grant_token(scope: str, **flags: bool) -> str:
    flags = flags or {"create": True, "write": True, "read": True, "list": True}
    return CREDENTIALS.thread_grant_token(AgentThreadGrant(scope=scope, **flags))


async def test_the_grant_creates_in_its_scope_and_refuses_another_scopes_session(
    agentdb_storage: AgentDBStorage,
) -> None:
    server, _ = make_server(agentdb_storage)
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
        # a scope covers create and list only: a thread is read and written by id
        r = await rpc(http, "message/send", {"message": wire("hi", "m3", context_id=ctx_id)}, **bob)
        assert r.status_code == 403
        r = await rpc(http, "tasks/list", {"contextId": ctx_id}, **alice)
        assert r.status_code == 403
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
    principal = CREDENTIALS.principal(bob, agent_name=AGENT, node_authorized=True)
    assert principal.kind == "grant" and principal.grant is not None
    assert principal.grant.scope == "user:bob" and principal.grant.write
    # the node's verdict without a bearer is the project credential; nothing is anonymous
    assert CREDENTIALS.principal({}, agent_name=AGENT, node_authorized=True).unscoped
    assert CREDENTIALS.principal({}, agent_name=AGENT).anonymous
    # a grant naming neither a scope nor a thread is refused
    bare = {"Authorization": f"Bearer {grant_token('', write=True)}"}
    with pytest.raises(Unauthorized):
        CREDENTIALS.principal(bare, agent_name=AGENT)
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
    assert not scoped.can_write(thread) and not scoped.can_update()
    everything = AgentThreadGrant(all=True, create=True, write=True)
    assert everything.can_create("user:x") and everything.can_write(thread)
    assert everything.can_update() and not AgentThreadGrant(all=True, read=True).can_update()


async def test_a_private_endpoint_takes_a_project_token_and_the_developers_auth(
    agentdb_storage: AgentDBStorage,
) -> None:
    seen: list[str] = []

    def auth(connection: Any) -> bool:
        seen.append(connection.headers.get("x-api-key", ""))
        return connection.headers.get("x-api-key") == "letmein"

    server, _ = make_server(agentdb_storage, public=False, auth=auth)
    async with http_client(server) as http:
        # nobody, the developer's check failing, then a project token with the right key
        r = await rpc(
            http, "message/send", {"message": wire("hi", "m1")}, **{"X-Api-Key": "letmein"}
        )
        assert r.status_code == 401
        token = CREDENTIALS.project_token()
        r = await rpc(
            http, "message/send", {"message": wire("hi", "m2")}, Authorization=f"Bearer {token}"
        )
        assert r.status_code == 401
        r = await rpc(
            http,
            "message/send",
            {"message": wire("hi", "m3")},
            Authorization=f"Bearer {token}",
            **{"X-Api-Key": "letmein"},
        )
        assert (
            r.status_code == 200
            and r.json()["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
        )
        # the card stays reachable, and says the endpoint is not public
        card = (await http.get(f"/{ENDPOINT}{AGENT_CARD_PATH}")).json()
        assert card["securityRequirements"]
    assert seen == ["letmein", "", "letmein"]


async def test_a_delegation_part_and_kind_reach_the_handler(
    agentdb_storage: AgentDBStorage,
) -> None:
    from livekit.agents.llm import ChatContext, ChatMessage

    seen: dict[str, Any] = {}

    async def handler(ctx: A2ASessionContext[Any]) -> str:
        seen["caller_history"] = ctx.caller_history
        seen["channel"] = ctx.messages[0].channel
        seen["parts"] = ctx.messages[0].parts
        return "ok"

    server, _ = make_server(agentdb_storage, handler)
    from livekit.agents.a2a.codec import chat_ctx_part, message_to_pb

    history = ChatContext([ChatMessage(role="user", content=["earlier"])])
    msg = {
        "messageId": "m1",
        "role": "user",
        "parts": [{"kind": "text", "text": "what is the fare"}, chat_ctx_part(history)],
        "metadata": {KIND: "delegation"},
    }
    async with http_client(server) as http:
        client = await stock_client(http, extensions=EXTENSION_URI)
        events = await send(client, message_to_pb(msg))
        await client.close()
    assert payloads(events)[-1].status.state == pb.TaskState.TASK_STATE_COMPLETED
    assert seen["channel"] == "delegation"
    assert seen["caller_history"] is not None
    assert [i.text_content for i in seen["caller_history"].items] == ["earlier"]
    # the conversation part is not the turn's text
    assert seen["parts"] == [{"kind": "text", "text": "what is the fare"}]


async def test_a_storage_error_mid_turn_fails_the_task_storage_unavailable(
    agentdb_workers: Workers,
) -> None:
    """docs/a2a.md: the turn's save could not be written; the task fails with the reason, and
    the client resends."""
    a, _ = agentdb_workers

    async def handler(ctx: A2ASessionContext[Any]) -> str:
        raise StorageUnavailable("the database went away")

    server, _ = make_server(a, handler)
    ctx_id = (await a.threads.create_thread()).thread_id
    async with http_client(server) as http:
        r = await rpc(http, "message/send", {"message": wire("hi", "m1", context_id=ctx_id)})
    task = r.json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_FAILED"
    reason = metadata_from_wire(task["status"]["message"]["metadata"])[REASON]
    assert reason == "storage-unavailable"
