"""Delegation (plan 7.2, 7.5): ``A2ADelegate`` against ``a2a_session`` in process over HTTP+SSE,
and against a stock a2a-sdk server that speaks plain A2A and nothing else; the delegate as a
toolset, its tool named from the card."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import a2a_pb2 as pb
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol
from starlette.applications import Starlette
from starlette.routing import Mount

from livekit.agents import Agent, AgentServer, InputRequired, RunContext, _core, llm
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.a2a.codec import metadata_from_wire
from livekit.agents.a2a.extension import KIND
from livekit.agents.delegation import A2ADelegate, DelegationFailed
from livekit.agents.llm import ChatContext, ChatMessage, FunctionToolCall
from livekit.agents.llm.tool_context import ASK_IN_OWN_WORDS
from livekit.agents.storage import AgentDBStorage
from livekit.agents.storage.resolve import Thread
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from .a2a_driver import (
    AGENT,
    ENDPOINT,
    Reply,
    ScriptedLLM,
    call,
    http_client,
    make_server,
    no_project_env,  # noqa: F401  (autouse: the demo environment must not shape the cards)
    output_of,
    rpc,
    sse_results,
    wire,
)
from .fake_llm import FakeLLM, FakeLLMResponse
from .test_a2a_server import history_items


async def lanes_of(storage: AgentDBStorage, thread_id: str) -> set[str]:
    db = await storage.open(thread_id)
    core = _core.Thread(db, storage.holder)
    try:
        return {lane.name for lane in await core.lanes()}
    finally:
        core.close()


pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------------------------
# against a2a_session, in process
# ---------------------------------------------------------------------------------------------


async def fare_expert(ctx: A2ASessionContext[Any]) -> str:
    await ctx.update("checking the fare rules")
    if ctx.message.text_content == "say goodbye":
        return "Thanks for calling, goodbye."
    caller = ctx.caller_history
    seen = f" (caller history: {len(caller.items)})" if caller is not None else ""
    return f"the fare is 240 USD{seen}"


async def test_a2a_delegate_against_a_livekit_expert(agentdb_storage: AgentDBStorage) -> None:
    server, _ = make_server(agentdb_storage, fare_expert)
    storage = server.storage
    assert storage is not None
    caller_session = (await storage.threads.create_thread()).thread_id
    progress: list[str] = []
    async with http_client(server) as http:
        delegate = A2ADelegate(f"http://agents.test/{ENDPOINT}", httpx_client=http)
        result = await delegate.run(
            "what is the fare", progress=progress.append, context_id=caller_session
        )
        assert delegate.transport == "JSONRPC"
        assert delegate.extension_active is True
        goodbye = await delegate.run(
            "say goodbye", progress=progress.append, context_id=caller_session
        )
        await delegate.aclose()

    assert progress == ["checking the fare rules", "checking the fare rules"]
    assert result.output == "the fare is 240 USD" and result.state == "completed"
    # inside the project the expert writes its own lane of the caller's session (plan 4.4)
    assert result.context_id == caller_session
    assert AGENT in await lanes_of(storage, caller_session)
    items = await history_items(storage, caller_session, AGENT)
    assert [i.text_content for i in items] == [
        "what is the fare",
        "the fare is 240 USD",
        "say goodbye",
        "Thanks for calling, goodbye.",
    ]
    tasks = await Thread(storage, caller_session, agent_name=AGENT).tasks()
    assert {t.id for t in tasks} == {result.task_id, goodbye.task_id}
    # the message went as a delegation
    assert all(t.data.history[0]["metadata"][KIND] == "delegation" for t in tasks)
    assert goodbye.output == "Thanks for calling, goodbye."


async def test_a2a_delegate_outside_the_session_sends_the_conversation(
    expert: Callable[..., A2ADelegate],
) -> None:
    history = ChatContext(
        [ChatMessage(role="user", content=["hi"]), ChatMessage(role="assistant", content=["hello"])]
    )
    from livekit.agents.delegation import record_delegation
    from livekit.agents.llm import FunctionCall

    delegate = expert(fare_expert)
    first = await delegate.run("what is the fare", message_history=history)
    asked = FunctionCall(name="ask", call_id="c1", arguments="{}")
    record_delegation(asked, first)
    second = await delegate.run("what is the fare", message_history=[*history.items, asked])
    assert first.output == "the fare is 240 USD (caller history: 2)"
    # outside a turn there is no session to name: a session of its own, found again from the
    # call that recorded it
    assert first.context_id.startswith("AT_") and second.context_id == first.context_id


# ---------------------------------------------------------------------------------------------
# against a stock a2a-sdk server: plain A2A, nothing else
# ---------------------------------------------------------------------------------------------


class _Reasoner(AgentExecutor):
    """A stock a2a-sdk server: a plain text agent, no extension, no LiveKit code."""

    def __init__(self) -> None:
        self.seen: list[pb.Message] = []

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        assert context.message is not None
        self.seen.append(context.message)
        task_id = context.task_id or "task-1"
        context_id = context.context_id or "ctx-1"
        await event_queue.enqueue_event(
            pb.Task(
                id=task_id,
                context_id=context_id,
                status=pb.TaskStatus(state=pb.TaskState.TASK_STATE_SUBMITTED),
            )
        )
        updater = TaskUpdater(event_queue, task_id, context_id)
        progress = pb.Message(
            message_id="p1",
            role=pb.Role.ROLE_AGENT,
            task_id=task_id,
            context_id=context_id,
            parts=[pb.Part(text="looking that up")],
        )
        await updater.update_status(pb.TaskState.TASK_STATE_WORKING, message=progress)
        text = context.get_user_input()
        if text == "fail":
            await updater.update_status(
                pb.TaskState.TASK_STATE_FAILED,
                message=pb.Message(
                    message_id="f1", role=pb.Role.ROLE_AGENT, parts=[pb.Part(text="no can do")]
                ),
            )
            return
        await updater.add_artifact(
            [pb.Part(text="Your balance is ")], artifact_id="a1", append=False, last_chunk=False
        )
        await updater.add_artifact(
            [pb.Part(text="forty dollars.")], artifact_id="a1", append=True, last_chunk=True
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


@contextlib.asynccontextmanager
async def stock_server(executor: _Reasoner) -> AsyncIterator[Starlette]:
    """The stock server as an ASGI app, with the SDK's background tasks and sse-starlette's
    shutdown watcher cleaned up after."""
    card = pb.AgentCard(
        name="reasoner",
        description="A stock reasoner.",
        version="0.1.0",
        capabilities=pb.AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[pb.AgentSkill(id="balance", name="balance", description="Balances.")],
        supported_interfaces=[
            pb.AgentInterface(
                url="http://reasoner.test/a2a/v1",
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_1_0,
            )
        ],
    )
    handler = DefaultRequestHandler(
        agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card
    )
    routes = [*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/v1")]
    try:
        yield Starlette(routes=[Mount("/a2a", routes=routes)])
    finally:
        with contextlib.suppress(Exception):
            await handler.aclose()
        from sse_starlette.sse import AppStatus

        watchers = [
            t
            for t in asyncio.all_tasks()
            if "_shutdown_watcher" in getattr(t.get_coro(), "__qualname__", "")
        ]
        AppStatus.should_exit = True
        try:
            if watchers:
                await asyncio.wait(watchers, timeout=5)
        finally:
            AppStatus.should_exit = False


def stock_http(app: Starlette) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://reasoner.test", timeout=30.0
    )


async def test_a2a_delegate_speaks_plain_a2a_to_a_stock_server() -> None:
    """A stock server gets a stock request, and its stock answer is enough: the working
    statuses are the progress, the artifact is the answer, failed is a failure. Nothing of the
    extension is sent: no header, no LiveKit session header, no metadata, no chat_ctx part."""
    plain = _Reasoner()
    seen_headers: list[dict[str, str]] = []

    async with stock_server(plain) as app:

        async def recording(scope: Any, receive: Any, send: Any) -> None:
            if scope["type"] == "http":
                seen_headers.append({k.decode(): v.decode() for k, v in scope["headers"]})
            await app(scope, receive, send)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=recording),
            base_url="http://reasoner.test",
            timeout=30.0,
        ) as http:
            progress: list[str] = []
            delegate = A2ADelegate("http://reasoner.test/a2a", httpx_client=http)
            result = await delegate.run(
                "balance please",
                progress=progress.append,
                context_id="AT_caller",
                message_history=ChatContext([ChatMessage(role="user", content=["earlier"])]),
            )
            with pytest.raises(DelegationFailed, match="no can do"):
                await delegate.run("fail", context_id="AT_caller")
            assert delegate.extension_active is False
            await delegate.aclose()
    assert progress == ["looking that up"]
    assert result.output == "Your balance is forty dollars."
    assert result.state == "completed"
    assert result.profile == "a2a"
    # a stock server never sees the caller's session id (the sdk mints a context of its own)
    assert plain.seen[0].context_id != "AT_caller"
    message = plain.seen[0]
    assert [p.WhichOneof("content") for p in message.parts] == ["text"]
    assert KIND not in dict(message.metadata) and not dict(message.metadata)
    for headers in seen_headers:
        assert "a2a-extensions" not in headers and "x-livekit-session-id" not in headers


async def test_a_minted_project_token_is_fresh_on_every_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delegate on the project's host mints its token per request, so a worker that outlives
    one token's TTL still reaches the expert."""
    from livekit.agents.a2a.auth import Credentials

    monkeypatch.setenv("LIVEKIT_URL", "http://reasoner.test")
    monkeypatch.setenv("LIVEKIT_API_KEY", "APItest")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret-secret-secret-secret-secret")
    minted: list[str] = []

    def project_token(self: Credentials, *, identity: str = "worker", **_: Any) -> str:
        minted.append(f"token-{len(minted)}")
        return minted[-1]

    monkeypatch.setattr(Credentials, "project_token", project_token)
    bearers: list[str] = []

    async with stock_server(_Reasoner()) as app:

        async def recording(scope: Any, receive: Any, send: Any) -> None:
            if scope["type"] == "http":
                headers = {k.decode(): v.decode() for k, v in scope["headers"]}
                bearers.append(headers.get("authorization", ""))
            await app(scope, receive, send)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=recording),
            base_url="http://reasoner.test",
            timeout=30.0,
        ) as http:
            delegate = A2ADelegate("http://reasoner.test/a2a", httpx_client=http)
            await delegate.run("balance please", context_id="AT_caller")
            await delegate.run("balance again", context_id="AT_caller")
            await delegate.aclose()

    # the card, then one send per run, each with a token of its own
    assert len(minted) == 3
    assert bearers == [f"Bearer {token}" for token in minted]


async def test_a_delegate_is_a_toolset_named_from_the_card(
    agentdb_storage: AgentDBStorage,
) -> None:
    expert_server = AgentServer(storage=agentdb_storage, agent_name="eligibility")
    expert_server.a2a_session(endpoint=ENDPOINT, description="Payer eligibility checks.")(
        fare_expert
    )
    async with http_client(expert_server) as http:
        back_office = A2ADelegate(f"http://agents.test/{ENDPOINT}", httpx_client=http)
        assert back_office.tools == [], "nothing before the card is read"
        await back_office.setup()
        (tool,) = back_office.tools
        assert tool.info.name == "ask"
        assert tool.info.description.startswith("Payer eligibility checks.")
        assert back_office.id == ENDPOINT
        front = Agent(instructions="front desk", tools=[back_office])
        assert [t.info.name for t in llm.ToolContext(front.tools).flatten()] == ["ask"]
        await back_office.aclose()


# ---------------------------------------------------------------------------------------------
# a tool's progress and the session, from inside agent.run()
# ---------------------------------------------------------------------------------------------


def _front_desk(expert: A2ADelegate) -> Agent:
    front = Agent(instructions="front desk")

    @front.tool
    async def coverage(ctx: RunContext) -> str:
        """Ask the eligibility expert."""
        await ctx.update("asking the expert")
        # no context_id: the delegate finds the turn's session through the run hooks
        return (await expert.run("is it covered", progress=ctx.update)).output

    front.update_options(
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="covered?",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[FunctionToolCall(name="coverage", arguments="{}", call_id="c1")],
                ),
                FakeLLMResponse(
                    input="the fare is 240 USD", content="Yes, 240 USD.", ttft=0, duration=0
                ),
            ]
        )
    )
    return front


async def test_a_tools_update_inside_agent_run_reaches_the_door_and_the_delegate_the_session(
    agentdb_workers: tuple[AgentDBStorage, AgentDBStorage],
) -> None:
    # one project: the clinic and the expert share the threads service and the databases, and the
    # expert is its own agent, so its lane is its own
    clinic_storage, expert_storage = agentdb_workers
    expert_server = AgentServer(storage=expert_storage, agent_name="eligibility")
    expert_server.a2a_session(endpoint=ENDPOINT, description="x")(fare_expert)
    async with http_client(expert_server) as http:
        delegate = A2ADelegate(f"http://agents.test/{ENDPOINT}", httpx_client=http)
        front = _front_desk(delegate)

        async def handler(ctx: A2ASessionContext[Any]) -> str:
            return (await front.run(ctx.message, message_history=ctx.message_history)).output

        clinic, _ = make_server(clinic_storage, handler)
        async with http_client(clinic) as clinic_http:
            response = await rpc(clinic_http, "message/stream", {"message": wire("covered?", "m1")})
            results = [r["result"] for r in sse_results(response)]
        await delegate.aclose()
    thread_id = results[0]["task"]["contextId"]
    working = [
        r["statusUpdate"]["status"]["message"]["parts"][0]["text"]
        for r in results
        if "statusUpdate" in r and r["statusUpdate"]["status"]["state"] == "TASK_STATE_WORKING"
    ]
    assert working == ["asking the expert", "checking the fare rules"]
    assert results[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
    # the expert wrote its own lane of the clinic's session, with no context_id passed
    storage = expert_server.storage
    assert storage is not None
    assert await lanes_of(storage, thread_id) == {AGENT, "eligibility"}


# ---------------------------------------------------------------------------------------------
# the expert asks, the person answers: the answer goes to the expert as they gave it
# ---------------------------------------------------------------------------------------------


def _expert_that_asks(*, verbatim: bool = False) -> Agent:
    expert = Agent(
        instructions="eligibility",
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="check coverage",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[FunctionToolCall(name="which_payer", arguments="{}", call_id="e1")],
                ),
                FakeLLMResponse(
                    input="Aetna, member 12-345",
                    content="covered by Aetna 12-345",
                    ttft=0,
                    duration=0,
                ),
            ]
        ),
    )

    @expert.tool
    async def which_payer(ctx: RunContext) -> InputRequired:
        """Ask for the payer."""
        return InputRequired("Which payer, and the member id?", verbatim=verbatim)

    return expert


@pytest.mark.rule("R23")
@pytest.mark.xfail(
    strict=True,
    reason="R32: Rust ask, continuation by taskId: the routed answer ends in a store error",
)
@pytest.mark.parametrize("verbatim", [False, True])
async def test_the_experts_question_is_answered_through_the_delegates_tool(
    agentdb_workers: tuple[AgentDBStorage, AgentDBStorage], verbatim: bool
) -> None:
    """The expert's question reaches the person in the front desk's words, or as the expert
    wrote it when verbatim; the answer goes back to the expert either way."""
    clinic_storage, expert_storage = agentdb_workers
    expert = _expert_that_asks(verbatim=verbatim)
    question = "Which payer, and the member id?"

    async def expert_door(ctx: A2ASessionContext[Any]) -> str:
        result = await expert.run(ctx.message, message_history=ctx.message_history)
        return result.input_required or str(result.output)

    expert_server = AgentServer(storage=expert_storage, agent_name="eligibility")
    expert_server.a2a_session(endpoint=ENDPOINT, description="Payer eligibility checks.")(
        expert_door
    )
    async with http_client(expert_server) as http:
        delegate = A2ADelegate(f"http://agents.test/{ENDPOINT}", httpx_client=http)
        front = Agent(
            instructions="front desk",
            tools=[delegate],
            llm=FakeLLM(
                fake_responses=[
                    FakeLLMResponse(
                        input="am I covered?",
                        content="",
                        ttft=0,
                        duration=0,
                        tool_calls=[
                            FunctionToolCall(
                                name="ask",
                                arguments='{"request": "check coverage"}',
                                call_id="c1",
                            )
                        ],
                    ),
                    FakeLLMResponse(
                        input=ASK_IN_OWN_WORDS.format(question=question),
                        content="Who is your insurer, and your member number?",
                        ttft=0,
                        duration=0,
                    ),
                    # the answer never reaches this model: its next input is the expert's reply
                    FakeLLMResponse(
                        input="covered by Aetna 12-345",
                        content="You are covered.",
                        ttft=0,
                        duration=0,
                    ),
                ]
            ),
        )

        async def desk(ctx: A2ASessionContext[Any]) -> str:
            result = await front.run(ctx.message, message_history=ctx.message_history)
            return result.input_required or str(result.output)

        clinic, _ = make_server(clinic_storage, desk)
        async with http_client(clinic) as clinic_http:
            first = [
                r["result"]
                for r in sse_results(
                    await rpc(
                        clinic_http, "message/stream", {"message": wire("am I covered?", "m1")}
                    )
                )
            ]
            thread_id = first[0]["task"]["contextId"]
            asked = first[-1]["statusUpdate"]["status"]
            assert asked["state"] == "TASK_STATE_INPUT_REQUIRED"
            (part,) = asked["message"]["parts"]
            assert part["text"] == (
                question if verbatim else "Who is your insurer, and your member number?"
            )
            # as written only when its part is flagged; the status metadata keeps no copy
            assert metadata_from_wire(part.get("metadata")).get("livekit/verbatim") is (
                True if verbatim else None
            )
            assert "livekit/verbatim" not in metadata_from_wire(
                first[-1]["statusUpdate"].get("metadata")
            )

            second = [
                r["result"]
                for r in sse_results(
                    await rpc(
                        clinic_http,
                        "message/stream",
                        {"message": wire("Aetna, member 12-345", "m2", context_id=thread_id)},
                    )
                )
            ]
        await delegate.aclose()
    assert second[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
    artifacts = [r["artifactUpdate"]["artifact"] for r in second if "artifactUpdate" in r]
    # streamed in pieces as the model writes it, the last piece empty
    assert "".join(p["text"] for a in artifacts for p in a.get("parts", [])) == "You are covered."

    db = await clinic_storage.open(thread_id)
    try:
        history = [row.payload for row in await db.history(AGENT)]
    finally:
        db.close()
    calls = [i for i in history if i["type"] == "function_call" and i["name"] == "ask"]
    assert [json.loads(c["arguments"])["request"] for c in calls] == [
        "check coverage",
        "Aetna, member 12-345",  # the person's words, not the model's
    ]


# ---------------------------------------------------------------------------------------------
# how the delegate reads what the expert sends (Flow B)
# ---------------------------------------------------------------------------------------------


class _Scripted(_Reasoner):
    """A stock server whose events after the acknowledgment are the test's own."""

    def __init__(self, script: Callable[[TaskUpdater], Any]) -> None:
        super().__init__()
        self._script = script

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id, context_id = context.task_id or "task-1", context.context_id or "ctx-1"
        await event_queue.enqueue_event(
            pb.Task(
                id=task_id,
                context_id=context_id,
                status=pb.TaskStatus(state=pb.TaskState.TASK_STATE_SUBMITTED),
            )
        )
        await self._script(TaskUpdater(event_queue, task_id, context_id))


@pytest.mark.rule("R24", "B23")
async def test_the_answer_is_the_artifact_that_ended_last() -> None:
    """A partial streamed under one id, then the whole answer under another: the answer is the
    whole one, read once."""

    async def script(updater: TaskUpdater) -> None:
        await updater.add_artifact(
            [pb.Part(text="Your bal")], artifact_id="partial", name="answer", last_chunk=False
        )
        await updater.add_artifact(
            [pb.Part(text="Your balance is forty dollars.")],
            artifact_id="whole",
            name="answer",
            last_chunk=True,
        )
        await updater.complete()

    async with stock_server(_Scripted(script)) as app, stock_http(app) as http:
        delegate = A2ADelegate("http://reasoner.test/a2a", httpx_client=http)
        result = await delegate.run("balance please", context_id="AT_caller")
        await delegate.aclose()
    assert result.output == "Your balance is forty dollars."


@pytest.mark.rule("R25")
async def test_an_expert_never_answers_with_its_progress(
    expert: Callable[..., A2ADelegate],
) -> None:
    """An expert task that ends with no text answers with no text, not its last progress."""

    async def silent(ctx: A2ASessionContext[Any]) -> str:
        await ctx.update("checking the schedule")
        return ""

    progress: list[str] = []
    result = await expert(silent).run("book a cleaning", progress=progress.append)
    assert progress == ["checking the schedule"]
    assert result.state == "completed"
    assert result.output == ""


STREAMED = ("SendStreamingMessage", "message/stream", "SubscribeToTask", "tasks/resubscribe")


class _Dropping(httpx.AsyncBaseTransport):
    """An expert's door whose first ``drops`` answer streams, the first one or a resubscribe,
    break after their first frame."""

    def __init__(self, app: Any, drops: int) -> None:
        self._inner = httpx.ASGITransport(app=app)
        self.drops = drops

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        method = json.loads(request.content).get("method", "") if request.content else ""
        if method not in STREAMED or self.drops <= 0:
            return response
        self.drops -= 1
        body = await response.aread()
        head = body[: body.index(b"\n\n") + 2]
        return httpx.Response(200, headers=response.headers, stream=_Broken(head), request=request)


class _Broken(httpx.AsyncByteStream):
    """The acknowledgment, then a read error."""

    def __init__(self, head: bytes) -> None:
        self._head = head

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._head
        raise httpx.ReadError("the stream dropped")


@pytest.mark.rule("R27", "B6")
@pytest.mark.parametrize(
    ("max_retry", "drops", "answered"),
    [(DEFAULT_API_CONNECT_OPTIONS.max_retry, 3, True), (1, 2, False)],
)
async def test_a_dropped_answer_stream_is_read_again_up_to_the_delegates_retries(
    agentdb_storage: AgentDBStorage, max_retry: int, drops: int, answered: bool
) -> None:
    """The expert's answer stream breaks after the acknowledgment: the delegate reads the task
    again up to ``conn_options.max_retry`` times, and the answer arrives with its progress
    reported once, or the delegation fails."""
    server = AgentServer(storage=agentdb_storage, agent_name="expert")
    server.a2a_session(endpoint="expert", description="Answers fares.")(fare_expert)
    transport = _Dropping(server.http.app, drops=drops)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://experts.test", timeout=30.0
    ) as http:
        delegate = A2ADelegate(
            "http://experts.test/expert",
            httpx_client=http,
            conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0),
        )
        progress: list[str] = []
        thread_id = (await agentdb_storage.threads.create_thread()).thread_id
        try:
            result = await delegate.run(
                "what is the fare", progress=progress.append, context_id=thread_id
            )
        except DelegationFailed:
            result = None
        await delegate.aclose()

    assert transport.drops == 0
    if answered:
        assert result is not None and result.output == "the fare is 240 USD"
        assert progress == ["checking the fare rules"]
    else:
        assert result is None


@pytest.mark.rule("R20", "B7")
async def test_a_stock_experts_context_is_read_from_the_conversation() -> None:
    """One delegate shared by two conversations: each keeps the stock expert's context of its
    own, recorded on the delegating call, and its next delegation goes back to it."""
    from livekit.agents.llm import FunctionCall

    plain = _Reasoner()
    async with stock_server(plain) as app, stock_http(app) as http:
        delegate = A2ADelegate("http://reasoner.test/a2a", httpx_client=http)

        def front() -> Agent:
            model = ScriptedLLM(
                Reply(tool_calls=[call("ask", "c1", request="balance please")]),
                Reply("Forty dollars.", when=output_of("c1")),
            )
            return Agent(instructions="front desk", tools=[delegate], llm=model)

        first = await front().run("my balance?")
        await front().run("my balance?")
        await front().run("and again?", message_history=first.new_messages())
        await delegate.aclose()

    one, two, again = (m.context_id for m in plain.seen)
    assert two != one
    assert again == one
    (delegated,) = [i for i in first.new_messages() if isinstance(i, FunctionCall)]
    assert delegated.extra["lk.delegation.context_id"] == one


@pytest.mark.rule("B32")
async def test_the_shared_client_names_no_thread_it_was_not_given(
    agentdb_storage: AgentDBStorage,
) -> None:
    """A client sends for many conversations: after a send in one thread, a cancel or a
    resubscribe of another names no thread unless it is told one."""
    from livekit.agents.a2a.client import AgentClient

    server = AgentServer(storage=agentdb_storage, agent_name="expert")
    server.a2a_session(endpoint="expert", description="Answers fares.")(fare_expert)
    bodies: list[dict[str, Any]] = []

    async def seen(request: httpx.Request) -> None:
        if request.content:
            bodies.append(json.loads(request.content))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.http.app),
        base_url="http://experts.test",
        timeout=30.0,
        event_hooks={"request": [seen]},
    ) as http:
        client = AgentClient("http://experts.test/expert", None, httpx_client=http)
        first = (await agentdb_storage.threads.create_thread()).thread_id
        second = (await agentdb_storage.threads.create_thread()).thread_id
        reply = await client.send("what is the fare", thread_id=first)
        async for _ in reply:
            pass
        async for _ in await client.send("what is the fare", thread_id=second):
            pass
        with contextlib.suppress(Exception):
            await client.cancel(reply.task_id)
        async for _ in client.resubscribe(reply.task_id):
            pass
        await client.aclose()

    later = [b for b in bodies if b.get("method") in ("CancelTask", "SubscribeToTask")]
    assert [b["method"] for b in later] == ["CancelTask", "SubscribeToTask"]
    assert all("contextId" not in b["params"] for b in later)
