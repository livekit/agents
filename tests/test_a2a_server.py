"""a2a_session over the real wire: our client and a stock one, against a loopback port."""

from __future__ import annotations

import asyncio
import contextlib
import pathlib
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.agents.a2a import TaskInput, TaskUpdate
from livekit.agents.a2a.extension import EXTENSION_URI, KIND, as_dict
from livekit.agents.a2a.server import AGENT_CARD_PATH, A2ASessionContext, mount
from livekit.agents.llm import ToolFlag

from .fake_llm import FakeLLM
from .test_a2a_runner import _AnsweringLLM, _says, _tool_call

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@function_tool
async def check_fares(ctx: RunContext) -> str:
    """Check the fare rules, slowly, reporting as it goes."""
    await ctx.update("checking the fare rules")
    await asyncio.sleep(0.05)
    return "fare is 240 USD"


@function_tool(flags={ToolFlag.CANCELLABLE})
async def hold_seat(ctx: RunContext) -> str:
    """Hold a seat, slowly enough that a cancel can land on it."""
    await ctx.update("holding the seat")
    await asyncio.sleep(60)
    return "held"


@function_tool
async def say_goodbye(ctx: RunContext) -> str:
    """Called when the caller is done."""
    request = ctx.request
    assert request is not None
    request.set_directive("end_session", reason="user_request")
    return "wrapped up"


def _fare_desk_llm() -> FakeLLM:
    return _AnsweringLLM(
        fake_responses=[
            _says("what is the change fee", "The change fee is $75."),
            _says("what is the fare", "", calls=[_tool_call("check_fares", "cf1")]),
            _says("that is all", "", calls=[_tool_call("say_goodbye", "sg1")]),
            _says("hold it", "", calls=[_tool_call("hold_seat", "hs1")]),
        ],
        fallbacks=["It is 240 USD.", "Thanks for calling."],
    )


class _Served:
    """An A2A session endpoint on a real loopback port."""

    def __init__(self, base_url: str, executor: Any) -> None:
        self.base_url = base_url
        self.executor = executor
        self.sessions: list[AgentSession] = []


@contextlib.asynccontextmanager
async def _serving(
    endpoint: str = "fare-desk",
    *,
    idle_timeout: float | None = None,
    handler: Callable[[A2ASessionContext, _Served], Awaitable[None]] | None = None,
) -> AsyncIterator[_Served]:
    app = FastAPI()
    served: _Served = _Served("", None)  # filled once the port is known

    async def fare_desk(ctx: A2ASessionContext) -> None:
        if handler is not None:
            await handler(ctx, served)
            return
        session = AgentSession(llm=_fare_desk_llm())
        await session.start(
            agent=Agent(instructions="fare desk", tools=[check_fares, say_goodbye, hold_seat])
        )
        served.sessions.append(session)
        ctx.attach(session)

    executor = mount(
        app,
        endpoint=endpoint,
        handler=fare_desk,
        description="Answers fare questions.",
        idle_timeout=idle_timeout,
    )
    served.executor = executor

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_config=None, access_log=False)
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            if serve_task.done():
                serve_task.result()
                raise RuntimeError("the test server stopped before it started")
            await asyncio.sleep(0.01)
        port = next(s.getsockname()[1] for srv in server.servers for s in srv.sockets)
        served.base_url = f"http://127.0.0.1:{port}"
        yield served
    finally:
        with contextlib.suppress(Exception):
            await executor.aclose()
        server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(serve_task, timeout=10.0)
        await _drain_sse_watcher()


async def _drain_sse_watcher() -> None:
    """sse-starlette watches for shutdown on a poll, and outlives the response it served."""
    from sse_starlette.sse import AppStatus

    watchers = [
        task
        for task in asyncio.all_tasks()
        if "_shutdown_watcher" in getattr(task.get_coro(), "__qualname__", "")
    ]
    if not watchers:
        return
    AppStatus.should_exit = True
    try:
        await asyncio.wait(watchers, timeout=10.0)
    finally:
        # the next test serves its own stream, and a latched flag would stop it dead
        AppStatus.should_exit = False


async def _collect(client: Any, task_input: TaskInput) -> list[TaskUpdate]:
    async with client.send(task_input) as stream:
        return [update async for update in stream]


def test_the_shipped_example_still_wires_up() -> None:
    """The example is the only thing a reader copies, so it must not rot silently."""
    import importlib.util

    path = (
        pathlib.Path(__file__).parent.parent
        / "examples"
        / "voice_agents"
        / "delegation"
        / "expert.py"
    )
    spec = importlib.util.spec_from_file_location("_delegation_example_expert", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    paths = {getattr(route, "path", "") for route in module.server.http.routes}
    assert f"/fare-desk{AGENT_CARD_PATH}" in paths
    assert "/fare-desk/v1/message:stream" in paths
    # the card route is registered before the binding's catch-all mount, which would shadow it
    ordered = [getattr(route, "path", "") for route in module.server.http.routes]
    assert ordered.index(f"/fare-desk{AGENT_CARD_PATH}") < ordered.index("/{tenant}")


async def test_the_card_names_the_endpoint_and_offers_the_extension() -> None:
    async with _serving() as served, httpx.AsyncClient(timeout=10.0) as http:
        response = await http.get(f"{served.base_url}/fare-desk{AGENT_CARD_PATH}")

    assert response.status_code == 200
    card = response.json()
    assert card["name"] == "fare-desk"
    # the card is not shadowed by the binding's own catch-all mount
    assert card["supportedInterfaces"][0]["url"].endswith("/fare-desk/v1")
    assert [e["uri"] for e in card["capabilities"]["extensions"]] == [EXTENSION_URI]
    # proto3 omits a false bool, so the offer is "not required" by saying nothing
    assert card["capabilities"]["extensions"][0].get("required", False) is False


async def test_our_client_gets_progress_then_the_answer() -> None:
    from livekit.agents.a2a import A2AClient

    async with _serving() as served:
        client = A2AClient(f"{served.base_url}/fare-desk")
        try:
            updates = await _collect(client, TaskInput(instruction="what is the fare"))
        finally:
            await client.aclose()

    assert client.extension_active is True
    assert [(u.state, u.text) for u in updates if u.text] == [
        ("working", "checking the fare rules"),
        ("completed", "It is 240 USD."),
    ]


async def test_a_directive_reaches_the_caller_with_the_answer() -> None:
    from livekit.agents.a2a import A2AClient

    async with _serving() as served:
        client = A2AClient(f"{served.base_url}/fare-desk")
        try:
            updates = await _collect(client, TaskInput(instruction="that is all"))
        finally:
            await client.aclose()

    (answer,) = [u for u in updates if u.state == "completed"]
    assert answer.directive is not None
    assert (answer.directive.kind, answer.directive.reason) == ("end_session", "user_request")


async def test_two_requests_share_one_context() -> None:
    from livekit.agents.a2a import A2AClient

    async with _serving() as served:
        client = A2AClient(f"{served.base_url}/fare-desk")
        try:
            first = await _collect(client, TaskInput(instruction="what is the change fee"))
            second = await _collect(client, TaskInput(instruction="what is the fare"))
        finally:
            await client.aclose()

    assert [u.text for u in first if u.state == "completed"] == ["The change fee is $75."]
    assert [u.text for u in second if u.state == "completed"] == ["It is 240 USD."]
    # one context, one handler run, one session
    assert len(served.sessions) == 1


async def test_closing_drops_the_context() -> None:
    from livekit.agents.a2a import A2AClient

    async with _serving() as served:
        client = A2AClient(f"{served.base_url}/fare-desk")
        await _collect(client, TaskInput(instruction="what is the change fee"))
        assert len(served.executor._contexts) == 1

        # aclose says goodbye, so the endpoint drops the context rather than waiting it out
        await client.aclose()
        assert served.executor._contexts == {}

    # the expert's session went with it
    assert not served.sessions[0]._started


async def test_cancelling_a_task_ends_it_canceled_over_the_wire() -> None:
    """A cancel is a state the caller can act on, not a stream that stops mid-sentence."""
    from livekit.agents.a2a import A2AClient

    async with _serving() as served:
        client = A2AClient(f"{served.base_url}/fare-desk")
        updates: list[TaskUpdate] = []
        try:
            async with client.send(TaskInput(instruction="hold it")) as stream:
                async for update in stream:
                    updates.append(update)
                    if update.text == "holding the seat":
                        await stream.cancel("user_interrupted")
        finally:
            await client.aclose()

    assert updates[-1].state == "canceled"


async def test_a_context_nobody_comes_back_to_is_dropped() -> None:
    """The backstop behind lk/kind = close: a caller that crashes says goodbye to nobody."""
    from livekit.agents.a2a import A2AClient

    async with _serving(idle_timeout=0.05) as served:
        client = A2AClient(f"{served.base_url}/fare-desk")
        try:
            await _collect(client, TaskInput(instruction="what is the change fee"))
            assert len(served.executor._contexts) == 1

            await asyncio.sleep(0.1)
            await served.executor._drop_idle()
            # the goodbye below would drop it too, so the claim is made before saying one
            assert served.executor._contexts == {}
            assert not served.sessions[0]._started
        finally:
            await client.aclose()


async def test_a_context_is_kept_unless_an_endpoint_asks_for_idle() -> None:
    """Dropping loses what the context held, so it is off until something wants it."""
    from livekit.agents.a2a import A2AClient

    async with _serving() as served:
        client = A2AClient(f"{served.base_url}/fare-desk")
        try:
            await _collect(client, TaskInput(instruction="what is the change fee"))
            assert served.executor._sweeper is None
            assert len(served.executor._contexts) == 1
        finally:
            await client.aclose()


async def test_a_stock_client_reads_the_same_endpoint() -> None:
    """A vanilla a2a-sdk client, with no extension header and no LiveKit code."""
    from a2a.client import ClientConfig, ClientFactory
    from a2a.client.card_resolver import A2ACardResolver
    from a2a.types import a2a_pb2 as pb
    from a2a.utils.constants import TransportProtocol

    async with _serving() as served, httpx.AsyncClient(timeout=30.0) as http:
        card = await A2ACardResolver(http, f"{served.base_url}/fare-desk").get_agent_card()
        client = ClientFactory(
            ClientConfig(
                httpx_client=http,
                streaming=True,
                supported_protocol_bindings=[TransportProtocol.HTTP_JSON],
                accepted_output_modes=["text/plain"],
            )
        ).create(card)

        request = pb.SendMessageRequest(
            message=pb.Message(
                message_id="m1",
                role=pb.Role.ROLE_USER,
                parts=[pb.Part(text="what is the fare")],
            )
        )
        events = [event async for event in client.send_message(request)]
        await client.close()

    payloads = [getattr(e, e.WhichOneof("payload")) for e in events]
    # the acknowledgment carries the server's task id, before any status event
    assert isinstance(payloads[0], pb.Task)
    assert payloads[0].id

    statuses = [p for p in payloads if isinstance(p, pb.TaskStatusUpdateEvent)]
    assert statuses[-1].status.state == pb.TaskState.TASK_STATE_COMPLETED

    # a person's turn is answered by the desk's own model, which writes a line about the
    # report on the way, so the fallbacks land in that order and the answer is the second
    artifacts = [p for p in payloads if isinstance(p, pb.TaskArtifactUpdateEvent)]
    assert [a.artifact.parts[0].text for a in artifacts] == ["Thanks for calling."]

    # what goes out is still plain text a client that ignores our parts reads; the tool's
    # own words stay in the chat item beside it rather than being said twice
    working = [s for s in statuses if s.status.state == pb.TaskState.TASK_STATE_WORKING]
    said = [part.text for s in working for part in s.status.message.parts if part.text]
    assert said == ["It is 240 USD."]
    assert "checking the fare rules" not in said


async def test_the_typed_item_rides_beside_the_relayed_text() -> None:
    """What the extension adds on top of what a vanilla client already sees."""
    from a2a.client import ClientConfig, ClientFactory
    from a2a.client.card_resolver import A2ACardResolver
    from a2a.types import a2a_pb2 as pb
    from a2a.utils.constants import TransportProtocol

    async with _serving() as served, httpx.AsyncClient(timeout=30.0) as http:
        card = await A2ACardResolver(http, f"{served.base_url}/fare-desk").get_agent_card()
        client = ClientFactory(
            ClientConfig(
                httpx_client=http,
                streaming=True,
                supported_protocol_bindings=[TransportProtocol.HTTP_JSON],
                accepted_output_modes=["text/plain"],
            )
        ).create(card)
        request = pb.SendMessageRequest(
            message=pb.Message(
                message_id="m1", role=pb.Role.ROLE_USER, parts=[pb.Part(text="what is the fare")]
            )
        )
        events = [event async for event in client.send_message(request)]
        await client.close()

    data_parts = [
        as_dict(part.data)
        for e in events
        if (p := getattr(e, e.WhichOneof("payload"))) and isinstance(p, pb.TaskStatusUpdateEvent)
        for part in p.status.message.parts
        if part.WhichOneof("content") == "data"
    ]
    kinds = [
        as_dict(part.metadata).get(KIND)
        for e in events
        if (p := getattr(e, e.WhichOneof("payload"))) and isinstance(p, pb.TaskStatusUpdateEvent)
        for part in p.status.message.parts
        if part.WhichOneof("content") == "data"
    ]
    assert set(kinds) == {"chat_item"}
    # the report names the call it reports for, and travels whether or not it is said
    assert any(d.get("update_of") == "cf1" for d in data_parts)


async def test_a_stock_client_hands_the_handler_where_to_persist() -> None:
    """The keys are plain message metadata, so any A2A client can send them."""
    from a2a.client import ClientConfig, ClientFactory
    from a2a.client.card_resolver import A2ACardResolver
    from a2a.types import a2a_pb2 as pb
    from a2a.utils.constants import TransportProtocol

    from livekit.agents.a2a import CALLER, CONVERSATION
    from livekit.agents.a2a.extension import struct

    seen: list[tuple[str, str | None, str | None]] = []

    async def recording(ctx: A2ASessionContext, served: _Served) -> None:
        seen.append((ctx.context_id, ctx.conversation_id, ctx.caller_session_id))
        session = AgentSession(llm=_fare_desk_llm())
        await session.start(agent=Agent(instructions="fare desk"))
        served.sessions.append(session)
        ctx.attach(session)

    async with _serving(handler=recording) as served, httpx.AsyncClient(timeout=30.0) as http:
        card = await A2ACardResolver(http, f"{served.base_url}/fare-desk").get_agent_card()
        client = ClientFactory(
            ClientConfig(
                httpx_client=http,
                streaming=True,
                supported_protocol_bindings=[TransportProtocol.HTTP_JSON],
                accepted_output_modes=["text/plain"],
            )
        ).create(card)
        request = pb.SendMessageRequest(
            message=pb.Message(
                message_id="m1",
                context_id="ctx-stock",
                role=pb.Role.ROLE_USER,
                parts=[pb.Part(text="what is the change fee")],
                metadata=struct({CONVERSATION: "DB_stock", CALLER: "voice"}),
            )
        )
        _ = [event async for event in client.send_message(request)]
        await client.close()

    assert seen == [("ctx-stock", "DB_stock", "voice")]


async def test_the_keys_stay_home_when_the_card_does_not_offer_the_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A conversation id is ours to share only with an endpoint that joins it."""
    from livekit.agents.a2a import A2AClient, client as client_module

    seen: list[tuple[str | None, str | None]] = []

    async def recording(ctx: A2ASessionContext, served: _Served) -> None:
        seen.append((ctx.conversation_id, ctx.caller_session_id))
        session = AgentSession(llm=_fare_desk_llm())
        await session.start(agent=Agent(instructions="fare desk"))
        served.sessions.append(session)
        ctx.attach(session)

    monkeypatch.setattr(client_module, "offers_extension", lambda card: False)
    async with _serving(handler=recording) as served:
        client = A2AClient(f"{served.base_url}/fare-desk")
        try:
            await _collect(
                client,
                TaskInput(
                    instruction="what is the change fee",
                    conversation_id="DB_secret",
                    caller_session_id="voice",
                ),
            )
        finally:
            await client.aclose()

    assert client.extension_active is False
    assert seen == [(None, None)]


async def test_a_dropped_context_rehydrates_on_the_next_request(
    tmp_path: pathlib.Path,
) -> None:
    from livekit.agents import store
    from livekit.agents.a2a import A2AClient
    from livekit.agents.store.executor import SQLiteExecutor

    local = store.LocalStore(tmp_path)
    conversation_id = await local.create_database()

    async def persisted(ctx: A2ASessionContext, served: _Served) -> None:
        assert ctx.conversation_id is not None
        session = AgentSession(llm=_fare_desk_llm())
        await session.start(
            agent=Agent(instructions="fare desk", tools=[check_fares]),
            persist=local.session(
                ctx.conversation_id, ctx.context_id, parent=ctx.caller_session_id
            ),
        )
        served.sessions.append(session)
        ctx.attach(session)

    delegation = {"conversation_id": conversation_id, "caller_session_id": "voice"}
    async with _serving(handler=persisted) as served:
        first = A2AClient(f"{served.base_url}/fare-desk", context_id="ctx-1")
        await _collect(first, TaskInput(instruction="what is the change fee", **delegation))
        # the goodbye closes the session, which releases it and so the connection
        await first.aclose()
        assert served.executor._contexts == {}
        with pytest.raises(store.StoreError):
            _ = local._databases[conversation_id].executor

        second = A2AClient(f"{served.base_url}/fare-desk", context_id="ctx-1")
        try:
            await _collect(second, TaskInput(instruction="what is the fare", **delegation))
            resumed = served.sessions[-1]
            texts = [m.text_content for m in resumed.history.messages()]
        finally:
            await second.aclose()
    await local.aclose()

    assert len(served.sessions) == 2
    # the second handler run starts where the first one stopped
    assert "what is the change fee" in texts and "The change fee is $75." in texts
    assert texts[-1] == "It is 240 USD."

    reopened = SQLiteExecutor(str(tmp_path / f"{conversation_id}.sqlite"))
    rows = [
        row
        async for row in reopened.query(
            "SELECT session_id, parent_session_id, closed_at IS NOT NULL AS closed FROM sessions"
        )
    ]
    assert rows == [{"session_id": "ctx-1", "parent_session_id": "voice", "closed": 1}]
    await reopened.aclose()
