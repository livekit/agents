from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import aiohttp
import pytest

from livekit.agents import Agent, AgentSession, ClientDelegation, DelegationContext, RunContext, llm
from livekit.agents.voice.tool_executor import _RunningTasks
from livekit.plugins.openai.realtime import GPTLiveModel, GPTLiveSession

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


async def reached(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), 3)


class Socket:
    """In-memory transport; the production plugin send/receive loops still run."""

    def __init__(self) -> None:
        self.inbound: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.changed = asyncio.Event()
        self.started = asyncio.Event()

    def receive_event(self, event: dict[str, Any]) -> None:
        self.inbound.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(event), ""))

    async def receive(self) -> aiohttp.WSMessage:
        return await self.inbound.get()

    async def send_str(self, data: str) -> None:
        event = json.loads(data)
        self.sent.append(event)
        self.changed.set()
        if event["type"] == "session.start":
            self.receive_event({"type": "session.started", "session": {"id": "test"}})
            self.started.set()
        elif event["type"] == "session.close":
            self.receive_event({"type": "session.closed", "reason": "close_requested"})

    async def close(self) -> None:
        self.inbound.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, ""))

    async def wait_for(self, predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        async def wait() -> dict[str, Any]:
            while True:
                self.changed.clear()
                if match := next((event for event in self.sent if predicate(event)), None):
                    return match
                await self.changed.wait()

        return await asyncio.wait_for(wait(), 3)

    @property
    def commentary(self) -> list[dict[str, Any]]:
        return [event for event in self.sent if event["type"] == "session.commentary.append"]


class Harness:
    def __init__(
        self, sockets: list[Socket], session: AgentSession, backend: ClientDelegation
    ) -> None:
        self.sockets, self.session, self.backend = sockets, session, backend

    @property
    def socket(self) -> Socket:
        return self.sockets[-1]

    @property
    def plugin(self) -> GPTLiveSession:
        plugin = self.session.current_agent.duplex_session
        assert isinstance(plugin, GPTLiveSession)
        return plugin

    def delegate(self, id: str, text: str) -> None:
        # Finish the preceding transcript to model separate utterances. A separate test
        # covers updates to the same still-open transcript item.
        self.plugin._end_speech("user")
        self.plugin._handle_event({"type": "session.input_transcript.delta", "delta": text})
        self.plugin._handle_event(
            {"type": "session.delegation.created", "delegation": {"id": id, "target": "client"}}
        )


@pytest.fixture
async def start(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Callable[..., Awaitable[Harness]]]:
    sessions: list[AgentSession] = []

    async def create(
        handler: Callable[[DelegationContext], Awaitable[str | None]] | None = None,
        *,
        shared: bool = True,
        tools: list[llm.FunctionTool] | None = None,
        model: llm.LLM | None = None,
    ) -> Harness:
        sockets: list[Socket] = []

        async def connect(self: GPTLiveSession) -> Any:
            socket = Socket()
            sockets.append(socket)
            return socket

        monkeypatch.setattr(GPTLiveSession, "_create_ws_conn", connect)
        backend = ClientDelegation(
            handler=handler,
            model=model,
            tools=tools or [],
            select_task=(lambda request: "desk") if shared else None,
        )
        session: AgentSession = AgentSession(
            llm=GPTLiveModel(api_key="test", delegation="client"), tools=[backend]
        )
        sessions.append(session)
        await session.start(Agent(instructions="Delegate requests and corrections."))
        await reached(sockets[0].started)
        return Harness(sockets, session, backend)

    yield create
    for session in sessions:
        await session.aclose()


async def test_real_session_dispatch_progress_result_and_lifecycle(
    start: Callable[..., Awaitable[Harness]],
) -> None:
    contexts: list[DelegationContext] = []

    async def handler(ctx: DelegationContext) -> str:
        contexts.append(ctx)
        await ctx.update("Checking availability", silent=True)
        await ctx.update("I found a match.")
        return "Tuesday is available."

    h = await start(handler)
    events: list[Any] = []
    h.session.on("tool_execution_updated", events.append)
    h.delegate("d1", "Find Tuesday")
    await h.socket.wait_for(lambda event: event.get("content") == "Tuesday is available.")
    assert [event["content"] for event in h.socket.commentary] == [
        "I found a match.",
        "Tuesday is available.",
    ]
    assert all(event["delegation_id"] == "d1" for event in h.socket.commentary)
    assert any(
        event["type"] == "session.thinking.append"
        and event.get("content") == "Checking availability"
        for event in h.socket.sent
    )
    assert [event.update.type for event in events] == [
        "tool_call_started",
        "tool_call_updated",
        "tool_call_updated",
        "tool_call_ended",
    ]
    assert events[0].update.function_call.extra["delegation_id"] == "d1"
    assert events[-1].update.status == "done"
    with pytest.raises(RuntimeError, match="not running"):
        await contexts[0].update("after completion")


async def test_correction_drops_old_result_and_preserves_input(
    start: Callable[..., Awaitable[Harness]],
) -> None:
    old_started, old_release, old_finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    histories: list[str] = []

    async def handler(ctx: DelegationContext) -> str:
        if ctx.request.id == "d1":
            old_started.set()
            await old_release.wait()
            old_finished.set()
            return "Monday is available."
        histories.extend(
            item.text_content or ""
            for item in ctx.chat_ctx.items
            if isinstance(item, llm.ChatMessage)
        )
        return "Tuesday is available."

    h = await start(handler)
    h.delegate("d1", "Find Monday")
    await reached(old_started)
    h.delegate("d2", "Actually Tuesday")
    await h.socket.wait_for(lambda event: event.get("content") == "Tuesday is available.")
    old_release.set()
    await reached(old_finished)
    await h.backend._executor.drain()
    assert [event["content"] for event in h.socket.commentary] == ["Tuesday is available."]
    assert "Find Monday" in histories and "Actually Tuesday" in histories


async def test_independent_requests_keep_their_results(
    start: Callable[..., Awaitable[Harness]],
) -> None:
    first_started, release = asyncio.Event(), asyncio.Event()

    async def handler(ctx: DelegationContext) -> str:
        if ctx.request.id == "d1":
            first_started.set()
            await release.wait()
        return ctx.request.pending_transcript

    h = await start(handler, shared=False)
    h.delegate("d1", "order")
    await reached(first_started)
    h.delegate("d2", "weather")
    await h.socket.wait_for(lambda event: event.get("content") == "weather")
    release.set()
    await h.socket.wait_for(lambda event: event.get("content") == "order")
    assert {event["delegation_id"] for event in h.socket.commentary} == {"d1", "d2"}


async def test_completion_queued_before_correction_is_checked_at_wire(
    start: Callable[..., Awaitable[Harness]],
) -> None:
    first_started, release = asyncio.Event(), asyncio.Event()

    async def handler(ctx: DelegationContext) -> str:
        if ctx.request.id == "d1":
            first_started.set()
            await release.wait()
        return ctx.request.id

    h = await start(handler)
    h.delegate("d1", "Monday")
    await reached(first_started)
    # Queue an old chunk synchronously, then advance the revision before the sender runs.
    request = next(task.ctx for task in _RunningTasks[h.session].values())
    delivered = request._reply_handler
    assert delivered is not None
    await delivered("old queued chunk", False, False)
    h.delegate("d2", "Tuesday")
    release.set()
    await h.socket.wait_for(lambda event: event.get("content") == "d2")
    assert [event["content"] for event in h.socket.commentary] == ["d2"]


async def test_reconnect_rejects_old_connection_and_accepts_reused_transport_id(
    start: Callable[..., Awaitable[Harness]],
) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    count = 0

    async def handler(ctx: DelegationContext) -> str:
        nonlocal count
        count += 1
        if count == 1:
            started.set()
            await release.wait()
            return "old connection"
        return "new connection"

    h = await start(handler)
    h.delegate("same-id", "Monday")
    await reached(started)
    h.plugin._reset_for_reconnect()
    h.plugin._session_started_fut.set_result(None)
    h.delegate("same-id", "Tuesday")
    release.set()
    await h.socket.wait_for(lambda event: event.get("content") == "new connection")
    assert [event["content"] for event in h.socket.commentary] == ["new connection"]


async def test_cancellation_resistant_shutdown_does_not_speak(
    start: Callable[..., Awaitable[Harness]],
) -> None:
    started, finished = asyncio.Event(), asyncio.Event()

    async def handler(ctx: DelegationContext) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            finished.set()
            return "late result"
        return "unreachable"

    h = await start(handler)
    h.delegate("d1", "Monday")
    await reached(started)
    await asyncio.wait_for(h.session.aclose(), 3)
    await reached(finished)
    assert not _RunningTasks.get(h.session)
    assert not h.socket.commentary


async def test_duplicate_event_does_not_repeat_work(
    start: Callable[..., Awaitable[Harness]],
) -> None:
    count = 0

    async def handler(ctx: DelegationContext) -> str:
        nonlocal count
        count += 1
        return "done"

    h = await start(handler)
    h.delegate("d1", "Monday")
    await h.socket.wait_for(lambda event: event.get("content") == "done")
    h.delegate("d1", "Monday")
    await asyncio.sleep(0)
    assert count == 1


async def test_failure_uses_correlated_delivery_and_error_lifecycle(
    start: Callable[..., Awaitable[Harness]],
) -> None:
    async def handler(ctx: DelegationContext) -> str:
        raise llm.ToolError("Lookup unavailable")

    h = await start(handler)
    events: list[Any] = []
    h.session.on("tool_execution_updated", events.append)
    h.delegate("d1", "Monday")
    await h.socket.wait_for(lambda event: event.get("content") == "Lookup unavailable")
    assert h.socket.commentary[0]["delegation_id"] == "d1"
    assert events[-1].update.status == "error"


async def test_actual_tool_outcome_survives_parent_cancellation_and_correction(
    start: Callable[..., Awaitable[Harness]],
) -> None:
    tool_started, release_tool, outcome_recorded = asyncio.Event(), asyncio.Event(), asyncio.Event()
    contexts: list[DelegationContext] = []
    executions = 0

    @llm.function_tool
    async def book() -> str:
        nonlocal executions
        executions += 1
        tool_started.set()
        await release_tool.wait()
        return "Booked: operation-42"

    async def handler(ctx: DelegationContext) -> str:
        contexts.append(ctx)
        if ctx.request.id == "d1":
            return str(await ctx.execute_tool("book", {}, call_id="backend-call-1"))
        await outcome_recorded.wait()
        assert any(
            isinstance(item, llm.FunctionCallOutput) and item.output == "Booked: operation-42"
            for item in ctx.chat_ctx.items
        )
        return "The original booking completed; a change needs reconciliation."

    h = await start(handler, tools=[book])
    events: list[Any] = []
    h.session.on("tool_execution_updated", events.append)
    h.delegate("d1", "Book Monday")
    await reached(tool_started)
    first = next(
        task
        for task in _RunningTasks[h.session].values()
        if task.ctx.function_call.name == "lk_agents_delegate"
    )
    h.delegate("d2", "Actually Tuesday")
    assert await first.executor.cancel(first.ctx.function_call.call_id)
    release_tool.set()
    # Waiting for the child executor task proves its final outcome callback completed.
    child = _RunningTasks[h.session]["backend-call-1"]
    await child.exe_task
    outcome_recorded.set()
    await h.socket.wait_for(
        lambda event: (
            event.get("delegation_id") == "d2" and event["type"] == "session.commentary.append"
        )
    )
    assert executions == 1
    assert all(event["delegation_id"] == "d2" for event in h.socket.commentary)
    assert any(
        event.update.type == "tool_call_ended"
        and event.update.call_id == "backend-call-1"
        and event.update.status == "done"
        for event in events
    )
    assert contexts[1].revision == 2


async def test_open_transcript_is_updated_without_mutating_older_input(
    start: Callable[..., Awaitable[Harness]],
) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    contexts: list[DelegationContext] = []

    async def handler(ctx: DelegationContext) -> str:
        contexts.append(ctx)
        if ctx.request.id == "d1":
            started.set()
            await release.wait()
        return ctx.request.id

    h = await start(handler)
    h.delegate("d1", "Monday")
    await reached(started)
    h.plugin._handle_event(
        {"type": "session.input_transcript.delta", "delta": ", actually Tuesday"}
    )
    h.plugin._handle_event(
        {"type": "session.delegation.created", "delegation": {"id": "d2", "target": "client"}}
    )
    await h.socket.wait_for(lambda event: event.get("content") == "d2")
    old = [
        item.text_content
        for item in contexts[0].request.chat_ctx.items
        if isinstance(item, llm.ChatMessage) and item.role == "user"
    ]
    new = [
        item.text_content
        for item in contexts[1].chat_ctx.items
        if isinstance(item, llm.ChatMessage) and item.role == "user"
    ]
    assert old == ["Monday"]
    assert new == ["Monday, actually Tuesday"]
    release.set()


async def test_streamed_chunks_are_rechecked_after_a_correction(
    start: Callable[..., Awaitable[Harness]],
) -> None:
    first_chunk, continue_stream = asyncio.Event(), asyncio.Event()

    async def handler(ctx: DelegationContext) -> str:
        if ctx.request.id == "d1":
            await ctx.update("First valid chunk")
            first_chunk.set()
            await continue_stream.wait()
            await ctx.update("obsolete quiet update", silent=True)
            await ctx.update("obsolete spoken update")
            return "obsolete completion"
        return "corrected result"

    h = await start(handler)
    h.delegate("d1", "Monday")
    await reached(first_chunk)
    await h.socket.wait_for(lambda event: event.get("content") == "First valid chunk")
    h.delegate("d2", "Tuesday")
    continue_stream.set()
    await h.socket.wait_for(lambda event: event.get("content") == "corrected result")
    assert [event["content"] for event in h.socket.commentary] == [
        "First valid chunk",
        "corrected result",
    ]
    assert not any(str(event.get("content", "")).startswith("obsolete") for event in h.socket.sent)


async def test_oversized_chunk_fails_visibly_without_truncating(
    start: Callable[..., Awaitable[Harness]],
) -> None:
    async def handler(ctx: DelegationContext) -> str:
        return "a" * 501

    h = await start(handler)
    ended = asyncio.Event()
    terminal: list[Any] = []

    def on_event(event: Any) -> None:
        if event.update.type == "tool_call_ended":
            terminal.append(event.update)
            ended.set()

    h.session.on("tool_execution_updated", on_event)
    h.delegate("d1", "Monday")
    await reached(ended)
    assert terminal[0].status == "error"
    assert "500 UTF-8 bytes" in terminal[0].message
    assert not h.socket.commentary


async def test_queued_output_does_not_cross_a_reconnect(
    start: Callable[..., Awaitable[Harness]],
) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    async def handler(ctx: DelegationContext) -> str:
        started.set()
        await release.wait()
        return "old final"

    h = await start(handler)
    h.delegate("d1", "Monday")
    await reached(started)
    run = next(task.ctx for task in _RunningTasks[h.session].values())
    assert run._reply_handler is not None
    await run._reply_handler("old queued", False, False)
    h.plugin._reset_for_reconnect()
    h.plugin._session_started_fut.set_result(None)
    release.set()
    await h.backend._executor.drain()
    await asyncio.sleep(0)
    assert not h.socket.commentary


async def test_default_backend_uses_sdk_tool_execution_and_retains_context(
    start: Callable[..., Awaitable[Harness]],
) -> None:
    calls = 0

    async def lookup_order(ctx: RunContext) -> str:
        nonlocal calls
        calls += 1
        await ctx.update("Checking order", silent=True)
        return "Order A1042 shipped"

    tool = llm.function_tool(lookup_order)
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="Where is A1042?",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    llm.FunctionToolCall(name="lookup_order", arguments="{}", call_id="lookup-1")
                ],
            ),
            FakeLLMResponse(
                input="Order A1042 shipped", content="Your order shipped.", ttft=0, duration=0
            ),
            FakeLLMResponse(
                input="Repeat that", content="Your order already shipped.", ttft=0, duration=0
            ),
        ]
    )
    h = await start(model=model, tools=[tool])
    events: list[Any] = []
    h.session.on("tool_execution_updated", events.append)
    h.delegate("d1", "Where is A1042?")
    first_result = await h.socket.wait_for(
        lambda event: event["type"] == "session.commentary.append"
    )
    assert first_result["content"] == "Your order shipped.", events
    h.delegate("d2", "Repeat that")
    await h.socket.wait_for(lambda event: event.get("content") == "Your order already shipped.")
    assert calls == 1
    history = h.backend._states["desk"].history.items
    assert any(
        isinstance(item, llm.FunctionCallOutput) and item.call_id == "lookup-1" for item in history
    )
    assert any(
        event.get("content") == "Checking order" and event["type"] == "session.thinking.append"
        for event in h.socket.sent
    )


async def test_session_scoped_backend_survives_handoff_but_output_keeps_connection_identity(
    start: Callable[..., Awaitable[Harness]],
) -> None:
    started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def handler(ctx: DelegationContext) -> str:
        if ctx.request.id == "d1":
            started.set()
            await release.wait()
            finished.set()
            return "old connection result"
        return "current connection result"

    h = await start(handler)
    old_socket = h.socket
    h.delegate("d1", "Monday")
    await reached(started)
    h.session.update_agent(Agent(instructions="Delegate requests and corrections."))
    assert h.session._update_activity_atask is not None
    await h.session._update_activity_atask
    assert not h.backend._closed
    assert _RunningTasks.get(h.session)
    release.set()
    await reached(finished)
    h.delegate("d2", "Tuesday")
    await h.socket.wait_for(lambda event: event.get("content") == "current connection result")
    assert not old_socket.commentary
    assert [event["content"] for event in h.socket.commentary] == ["current connection result"]
    assert h.backend._states["desk"].revision == 2


async def test_original_detached_example_reproduction() -> None:
    # Preserve the failing scenario from 34a4e8f: each delegation answered independently.
    release = {day: asyncio.Event() for day in ("Monday", "Tuesday")}
    answers: list[str] = []

    async def answer(day: str) -> None:
        await release[day].wait()
        answers.append(day)

    workers = [asyncio.create_task(answer(day)) for day in release]
    release["Tuesday"].set()
    await workers[1]
    release["Monday"].set()
    await workers[0]
    assert answers == ["Tuesday", "Monday"]


async def test_superseded_tool_waiting_for_admission_cannot_execute(
    start: Callable[..., Awaitable[Harness]],
) -> None:
    started = asyncio.Event()
    executions = 0

    @llm.function_tool(on_duplicate="reject")
    async def lookup() -> str:
        nonlocal executions
        executions += 1
        return "stale lookup"

    async def handler(ctx: DelegationContext) -> str:
        if ctx.request.id == "d1":
            started.set()
            return str(await ctx.execute_tool("lookup", {}, call_id="lookup-1"))
        return "corrected"

    h = await start(handler, tools=[lookup])
    await h.backend._executor._duplicate_check_lock.acquire()
    h.delegate("d1", "Monday")
    await reached(started)
    h.delegate("d2", "Tuesday")
    h.backend._executor._duplicate_check_lock.release()
    await h.socket.wait_for(lambda event: event.get("content") == "corrected")
    await h.backend._executor.drain()
    assert executions == 0
    assert [event["content"] for event in h.socket.commentary] == ["corrected"]


async def test_manual_client_delegation_remains_available(
    start: Callable[..., Awaitable[Harness]],
) -> None:
    async def handler(ctx: DelegationContext) -> str:
        raise AssertionError("managed handler must not run")

    h = await start(handler)
    await h.plugin._update_tools([])
    received: list[Any] = []
    h.plugin.on("delegation_created", received.append)
    h.delegate("manual-1", "Monday")
    assert len(received) == 1
    assert received[0].id == "manual-1"
    assert received[0].pending_transcript == "Monday"
    h.plugin.append_commentary("Manual answer", delegation_id="manual-1")
    await h.socket.wait_for(lambda event: event.get("content") == "Manual answer")


async def test_backend_context_cannot_be_shared_between_sessions() -> None:
    async def handler(ctx: DelegationContext) -> None:
        return None

    backend = ClientDelegation(handler=handler)
    first: AgentSession = AgentSession(llm=FakeLLM())
    second: AgentSession = AgentSession(llm=FakeLLM())
    backend._attach_activity(activity=None, session=first)
    with pytest.raises(ValueError, match="between sessions"):
        backend._attach_activity(activity=None, session=second)
    await backend.aclose()
