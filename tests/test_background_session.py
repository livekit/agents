import asyncio
from dataclasses import FrozenInstanceError
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

import livekit.agents as agents
import livekit.agents.voice as voice
from livekit.agents import llm
from livekit.agents.llm import ChatContext
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.background_session import (
    _BACKGROUND_SEND_TOOL_NAME,
    _BACKGROUND_STATE_TOOL_NAME,
    BackgroundContext,
    BackgroundDefinition,
    BackgroundHandlingOptions,
    BackgroundReplyPromptArgs,
    BackgroundUpdatePromptArgs,
    _BackgroundRuntimeManager,
    background,
)

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


def test_background_maps_public_name_to_internal_id() -> None:
    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        del ctx

    assert worker.id == "worker"
    assert worker.description == "Handles work."


def test_background_uses_entrypoint_docstring_as_description() -> None:
    @background(name="research")
    async def research(ctx: BackgroundContext) -> None:
        """Investigates requests."""
        del ctx

    assert research.id == "research"
    assert research.description == "Investigates requests."


def test_background_decorator_description_overrides_docstring() -> None:
    @background(name="research", description="Decorator description.")
    async def research(ctx: BackgroundContext) -> None:
        """Docstring description."""
        del ctx

    assert research.description == "Decorator description."


def test_background_requires_decorator_or_docstring_description() -> None:
    with pytest.raises(ValueError, match="background session description must be non-empty"):

        @background(name="worker")
        async def worker(ctx: BackgroundContext) -> None:
            del ctx


def test_background_symbols_are_exported_from_public_packages() -> None:
    expected = {
        "background": background,
        "BackgroundDefinition": BackgroundDefinition,
        "BackgroundContext": BackgroundContext,
        "BackgroundUpdatePromptArgs": BackgroundUpdatePromptArgs,
        "BackgroundReplyPromptArgs": BackgroundReplyPromptArgs,
        "BackgroundHandlingOptions": BackgroundHandlingOptions,
        "BackgroundMessageReceived": voice.BackgroundMessageReceived,
        "BackgroundReplyUpdated": voice.BackgroundReplyUpdated,
        "BackgroundMessageUpdatedEvent": voice.BackgroundMessageUpdatedEvent,
    }

    for name, symbol in expected.items():
        assert getattr(voice, name) is symbol
        assert getattr(agents, name) is symbol
        assert name in voice.__all__
        assert name in agents.__all__


def test_background_cleanup_exports_context_session_and_tool_name() -> None:
    assert voice.background is agents.background
    assert voice.BackgroundDefinition is agents.BackgroundDefinition
    assert voice.BackgroundContext is agents.BackgroundContext
    assert "background" in voice.__all__
    assert "BackgroundDefinition" in voice.__all__
    assert "BackgroundContext" in voice.__all__

    @voice.background(name="worker", description="Handles work.")
    async def worker(ctx: voice.BackgroundContext) -> None:
        assert ctx.session is session

    session = AgentSession(vad=None, background=[worker])
    assert session._background_manager is not None
    assert session._background_manager._runtimes["worker"].context.session is session
    tool = llm.ToolContext(session.tools).get_function_tool(_BACKGROUND_SEND_TOOL_NAME)
    assert tool is not None
    assert tool.info.name == _BACKGROUND_SEND_TOOL_NAME


@pytest.mark.asyncio
async def test_background_send_emits_raw_message_once_after_acceptance() -> None:
    @background(name="research", description="Investigates requests.")
    async def research(ctx: BackgroundContext) -> None:
        del ctx

    session = _make_session()
    manager = _BackgroundRuntimeManager([research], session=session)
    context = manager._runtimes["research"].context

    await context.send("raw answer")

    background_calls = [
        call for call in session.emit.call_args_list if call.args[0] == "background_message_updated"
    ]
    assert len(background_calls) == 1
    event = background_calls[0].args[1]
    assert isinstance(event, voice.BackgroundMessageUpdatedEvent)
    assert event.update == voice.BackgroundMessageReceived(
        background_id="research",
        message_id=event.update.message_id,
        content="raw answer",
    )
    [message] = session.history.insert.call_args.args[0]
    assert event.update.message_id == message.extra["background_message_id"]
    await manager.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interrupted", "chat_items", "expected_status"),
    [
        (False, [MagicMock()], "completed"),
        (True, [], "interrupted"),
        (False, [], "skipped"),
    ],
)
async def test_background_reply_events_link_message_and_speech_ids(
    interrupted: bool,
    chat_items: list[object],
    expected_status: Literal["completed", "interrupted", "skipped"],
) -> None:
    @background(name="research", description="Investigates requests.")
    async def research(ctx: BackgroundContext) -> None:
        del ctx

    callbacks: list[object] = []
    speech = MagicMock()
    speech.id = "speech_1"
    speech.interrupted = interrupted
    speech.chat_items = chat_items
    speech.add_done_callback.side_effect = callbacks.append
    session = _make_session()
    session.generate_reply.return_value = speech
    manager = _BackgroundRuntimeManager([research], session=session)

    await manager._runtimes["research"].context.send("answer")
    reply_task = manager._runtimes["research"]._reply_scheduler.reply_task
    assert reply_task is not None
    await reply_task
    callbacks[0](speech)  # type: ignore[operator]

    events = [
        call.args[1]
        for call in session.emit.call_args_list
        if call.args[0] == "background_message_updated"
    ]
    assert len(events) == 3
    received = events[0].update
    assert events[1].update == voice.BackgroundReplyUpdated(
        background_id="research",
        message_ids=[received.message_id],
        status="scheduled",
        speech_id="speech_1",
    )
    assert events[2].update == voice.BackgroundReplyUpdated(
        background_id="research",
        message_ids=[received.message_id],
        status=expected_status,
        speech_id="speech_1",
    )
    await manager.aclose()


def test_agent_session_generates_stable_routing_tool() -> None:
    @background(name="codex", description="Reviews code.")
    async def codex(ctx: BackgroundContext) -> None:
        del ctx

    @background(name="claude", description="Handles repository tasks.")
    async def claude(ctx: BackgroundContext) -> None:
        del ctx

    tools: list[llm.Tool | llm.Toolset] = []
    session = AgentSession(vad=None, tools=tools, background=[codex, claude])

    assert len(session.tools) == 2
    assert session.tools is not tools
    assert tools == []
    schema = llm.ToolContext(session.tools).parse_function_tools("openai")
    assert schema == [
        {
            "type": "function",
            "function": {
                "name": _BACKGROUND_SEND_TOOL_NAME,
                "description": (
                    "Send a message to a background session. Delivery is asynchronous and does "
                    "not wait for a response.\n\n"
                    "Available background sessions:\n"
                    "- claude: Handles repository tasks.\n"
                    "- codex: Reviews code."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "background_session_id": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["background_session_id", "content"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": _BACKGROUND_STATE_TOOL_NAME,
                "description": (
                    "Get the real-time state reported by a background session — what it is "
                    "working on right now. Use this when the user asks about the progress or "
                    "status of background work.\n\n"
                    "Available background sessions:\n"
                    "- claude: Handles repository tasks.\n"
                    "- codex: Reviews code."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"background_session_id": {"type": "string"}},
                    "required": ["background_session_id"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
    ]


@pytest.mark.asyncio
async def test_background_entrypoint_starts_after_current_agent_exists() -> None:
    started = asyncio.Event()
    observed_agents: list[Agent] = []

    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        del ctx
        observed_agents.append(session.current_agent)
        started.set()
        await asyncio.Event().wait()

    session = AgentSession(vad=None, background=[worker])
    agent = Agent(instructions="Test agent.")

    await session.start(agent)
    try:
        assert started.is_set()
        assert observed_agents == [agent]
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_cancelled_start_closes_started_background_entrypoints() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    wait_started_called = asyncio.Event()
    block_wait_started = asyncio.Event()

    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        del ctx
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    session = AgentSession(vad=None, background=[worker])
    assert session._background_manager is not None
    manager = session._background_manager

    async def wait_started() -> None:
        wait_started_called.set()
        await block_wait_started.wait()

    manager.wait_started = wait_started  # type: ignore[method-assign]
    start_task = asyncio.create_task(session.start(Agent(instructions="Test agent.")))

    await wait_started_called.wait()
    await entered.wait()
    start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start_task

    try:
        await asyncio.wait_for(cancelled.wait(), timeout=0.05)
    finally:
        await manager.aclose()
        if session._activity is not None:
            await session._activity.aclose()

    task = manager._runtimes["worker"]._task
    assert task is not None and task.done()


@pytest.mark.asyncio
async def test_generated_tool_acknowledges_immediately_and_routes_only_selected_queue() -> None:
    @background(name="first", description="First worker.")
    async def first(ctx: BackgroundContext) -> None:
        del ctx

    @background(name="second", description="Second worker.")
    async def second(ctx: BackgroundContext) -> None:
        del ctx

    session = AgentSession(vad=None, background=[first, second])
    tool = llm.ToolContext(session.tools).get_function_tool(_BACKGROUND_SEND_TOOL_NAME)
    assert tool is not None

    result = await tool(background_session_id="second", content="do work")

    assert result == "Message has been delivered."
    assert session._background_manager is not None
    assert session._background_manager._runtimes["first"]._incoming.empty()
    assert session._background_manager._runtimes["second"]._incoming.get_nowait() == "do work"


@pytest.mark.asyncio
async def test_agent_background_send_conflict_fails_before_activity_starts() -> None:
    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        del ctx

    entered = False

    class ConflictingAgent(Agent):
        @llm.function_tool(name=_BACKGROUND_SEND_TOOL_NAME)
        async def conflicting_send(self, content: str) -> str:
            return content

        async def on_enter(self) -> None:
            nonlocal entered
            entered = True

    session = AgentSession(vad=None, background=[worker])
    agent = ConflictingAgent(instructions="Test agent.")

    with pytest.raises(ValueError, match=rf"duplicate function name: {_BACKGROUND_SEND_TOOL_NAME}"):
        await session.start(agent)

    assert entered is False
    assert session._activity is None
    assert agent._activity is None


@pytest.mark.asyncio
async def test_agent_rejects_generated_background_send_tool_instance() -> None:
    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        del ctx

    session = AgentSession(vad=None, background=[worker])
    generated_tool = session.tools[0]
    agent = Agent(instructions="Test agent.", tools=[generated_tool])

    try:
        with pytest.raises(ValueError, match=rf"{_BACKGROUND_SEND_TOOL_NAME} is reserved"):
            await session.start(agent)
    finally:
        if session._started:
            await session.aclose()

    assert session._activity is None
    assert agent._activity is None


@pytest.mark.asyncio
@pytest.mark.parametrize("realtime", [False, True])
async def test_dynamic_agent_tools_reject_reserved_name_without_mutation(
    realtime: bool,
) -> None:
    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        del ctx
        await asyncio.Event().wait()

    @llm.function_tool(name=_BACKGROUND_SEND_TOOL_NAME)
    async def conflicting_send(content: str) -> str:
        return content

    session = AgentSession(vad=None, background=[worker])
    agent = Agent(instructions="Test agent.")
    await session.start(agent)
    assert session._activity is not None
    activity = session._activity
    rt_session = AsyncMock() if realtime else None
    activity._rt_session = rt_session

    original_tools = agent.tools
    original_agent_items = list(agent.chat_ctx.items)
    original_history_items = list(session.history.items)

    try:
        with pytest.raises(ValueError, match=rf"{_BACKGROUND_SEND_TOOL_NAME} is reserved"):
            await agent.update_tools([conflicting_send])

        assert agent.tools == original_tools
        assert list(agent.chat_ctx.items) == original_agent_items
        assert list(session.history.items) == original_history_items
        if rt_session is not None:
            rt_session.update_tools.assert_not_awaited()
            rt_session.update_chat_ctx.assert_not_awaited()
    finally:
        activity._rt_session = None
        await session.aclose()


def test_session_background_send_conflict_fails_during_construction() -> None:
    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        del ctx

    @llm.function_tool(name=_BACKGROUND_SEND_TOOL_NAME)
    async def conflicting_send(content: str) -> str:
        return content

    with pytest.raises(ValueError, match="reserved for background sessions"):
        AgentSession(vad=None, tools=[conflicting_send], background=[worker])


def test_empty_background_configuration_preserves_session_tools() -> None:
    @llm.function_tool
    async def existing_tool() -> str:
        return "ok"

    tools: list[llm.Tool | llm.Toolset] = [existing_tool]
    session = AgentSession(vad=None, tools=tools, background=[])

    assert session.tools is tools
    assert session._background_manager is None


def test_agent_session_validates_direct_background_definitions() -> None:
    async def worker(ctx: BackgroundContext) -> None:
        del ctx

    invalid = BackgroundDefinition(
        id="",
        description="Handles work.",
        background_handling=None,
        entrypoint=worker,
    )

    with pytest.raises(ValueError, match="background name must be non-empty"):
        AgentSession(vad=None, background=[invalid])


@pytest.mark.asyncio
async def test_generated_tool_rejects_unknown_id_with_valid_ids() -> None:
    @background(name="alpha", description="First worker.")
    async def alpha(ctx: BackgroundContext) -> None:
        del ctx

    @background(name="beta", description="Second worker.")
    async def beta(ctx: BackgroundContext) -> None:
        del ctx

    session = AgentSession(vad=None, background=[beta, alpha])
    tool = llm.ToolContext(session.tools).get_function_tool(_BACKGROUND_SEND_TOOL_NAME)
    assert tool is not None

    with pytest.raises(llm.ToolError, match=r"Valid IDs: alpha, beta"):
        await tool(background_session_id="missing", content="ignored")

    assert session._background_manager is not None
    assert all(
        runtime._incoming.empty() for runtime in session._background_manager._runtimes.values()
    )


@pytest.mark.asyncio
async def test_background_runtime_and_tool_survive_agent_handoff_without_restart() -> None:
    starts = 0
    received: list[str] = []
    received_two = asyncio.Event()

    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        nonlocal starts
        starts += 1
        async for message in ctx.message_stream():
            received.append(message)
            if len(received) == 2:
                received_two.set()

    session = AgentSession(vad=None, background=[worker])
    first_agent = Agent(instructions="First.")
    second_agent = Agent(instructions="Second.")
    tool = llm.ToolContext(session.tools).get_function_tool(_BACKGROUND_SEND_TOOL_NAME)
    assert tool is not None

    await session.start(first_agent)
    assert session._background_manager is not None
    manager = session._background_manager
    runtime_task = manager._runtimes["worker"]._task
    try:
        await tool(background_session_id="worker", content="before")
        await session._update_activity(second_agent)
        handoff_tool = llm.ToolContext(session.tools).get_function_tool(_BACKGROUND_SEND_TOOL_NAME)
        assert handoff_tool is tool
        await handoff_tool(background_session_id="worker", content="after")
        await asyncio.wait_for(received_two.wait(), timeout=1)

        assert received == ["before", "after"]
        assert starts == 1
        assert session._background_manager is manager
        assert manager._runtimes["worker"]._task is runtime_task
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("drain", [False, True])
async def test_session_shutdown_owns_background_tasks_for_all_drain_modes(
    drain: bool,
) -> None:
    contexts: list[BackgroundContext] = []
    cancelled = asyncio.Event()

    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        contexts.append(ctx)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    session = AgentSession(vad=None, background=[worker])
    await session.start(Agent(instructions="Test agent."))
    assert session._background_manager is not None
    task = session._background_manager._runtimes["worker"]._task
    assert task is not None

    session.shutdown(drain=drain)
    closing_task = session._closing_task
    assert closing_task is not None
    await closing_task

    assert cancelled.is_set()
    assert task.done()
    with pytest.raises(RuntimeError, match="closed"):
        await contexts[0].send("too late")


@pytest.mark.asyncio
async def test_session_restart_creates_fresh_background_runtime_state() -> None:
    contexts: list[BackgroundContext] = []
    cancelled: list[BackgroundContext] = []

    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        contexts.append(ctx)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(ctx)

    session = AgentSession(vad=None, background=[worker])
    agent = Agent(instructions="Test agent.")
    tool = llm.ToolContext(session.tools).get_function_tool(_BACKGROUND_SEND_TOOL_NAME)
    assert tool is not None

    await session.start(agent)
    assert session._background_manager is not None
    manager = session._background_manager
    first_runtime = manager._runtimes["worker"]
    first_task = first_runtime._task
    await session.aclose()

    assert first_task is not None and first_task.done()
    assert cancelled == [contexts[0]]

    try:
        await session.start(agent)
        second_runtime = manager._runtimes["worker"]
        second_task = second_runtime._task
        restarted_tool = llm.ToolContext(session.tools).get_function_tool(
            _BACKGROUND_SEND_TOOL_NAME
        )

        assert session._background_manager is manager
        assert restarted_tool is tool
        assert second_runtime is not first_runtime
        assert second_runtime.context is not first_runtime.context
        assert second_task is not None and second_task is not first_task
        assert second_runtime._incoming.empty()

        result = await restarted_tool(background_session_id="worker", content="fresh")
        assert result == "Message has been delivered."
        assert second_runtime._incoming.get_nowait() == "fresh"
    finally:
        if session._started:
            await session.aclose()

    assert second_task.done()
    assert cancelled == contexts


def test_decorator_builds_immutable_reusable_definition() -> None:
    @background(name="research", description="Investigates requests.")
    async def research(ctx: BackgroundContext) -> None:
        del ctx

    assert isinstance(research, BackgroundDefinition)
    assert research.id == "research"
    assert research.description == "Investigates requests."
    assert research.background_handling is None

    with pytest.raises(FrozenInstanceError):
        research.id = "changed"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_definition_snapshots_mutable_background_handling() -> None:
    options = {"update_template": "original: {message}"}

    @background(
        name="research",
        description="Investigates requests.",
        background_handling=options,
    )
    async def research(ctx: BackgroundContext) -> None:
        del ctx

    options["update_template"] = "caller mutation: {message}"
    exposed_options = research.background_handling
    assert exposed_options is not None
    exposed_options["update_template"] = "attribute mutation: {message}"

    session = _make_session()
    manager = _BackgroundRuntimeManager([research], session=session)
    await manager._runtimes["research"].context.send("answer")

    [message] = session.history.insert.call_args.args[0]
    assert message.text_content == "original: answer"
    await manager.aclose()


@pytest.mark.parametrize(
    ("name", "description", "expected_error"),
    [
        ("", "Valid", "background name must be non-empty"),
        ("valid", "", "background session description must be non-empty"),
        ("   ", "Valid", "background name must be non-empty"),
        ("valid", "   ", "background session description must be non-empty"),
    ],
)
def test_decorator_rejects_empty_metadata(
    name: str,
    description: str,
    expected_error: str,
) -> None:
    async def worker(ctx: BackgroundContext) -> None:
        del ctx

    with pytest.raises(ValueError, match=expected_error):
        background(name=name, description=description)(worker)


def test_decorator_rejects_non_async_or_wrong_signature() -> None:
    def sync_entrypoint(ctx: BackgroundContext) -> None:
        del ctx

    async def no_context() -> None:
        pass

    async def two_contexts(ctx: BackgroundContext, other: object) -> None:
        del ctx, other

    decorate = background(name="valid", description="Valid")
    with pytest.raises(TypeError, match="async"):
        decorate(sync_entrypoint)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exactly one"):
        decorate(no_context)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exactly one"):
        decorate(two_contexts)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_reused_definition_has_isolated_fifo_streams() -> None:
    received: dict[str, list[str]] = {"one": [], "two": []}
    ready = asyncio.Event()
    contexts: list[BackgroundContext] = []

    @background(name="worker", description="Processes messages.")
    async def worker(ctx: BackgroundContext) -> None:
        contexts.append(ctx)
        if len(contexts) == 2:
            ready.set()
        destination = "one" if len(contexts) == 1 else "two"
        async for message in ctx.message_stream():
            received[destination].append(message)

    first = _BackgroundRuntimeManager([worker], session=MagicMock())
    second = _BackgroundRuntimeManager([worker], session=MagicMock())
    first.start()
    second.start()
    await ready.wait()

    first.enqueue("worker", "first-a")
    second.enqueue("worker", "second-a")
    first.enqueue("worker", "first-b")
    await asyncio.sleep(0)

    assert received == {"one": ["first-a", "first-b"], "two": ["second-a"]}
    assert contexts[0] is not contexts[1]
    assert all(ctx.id == "worker" for ctx in contexts)
    assert all(ctx.description == "Processes messages." for ctx in contexts)
    await first.aclose()
    await second.aclose()


def _make_session() -> MagicMock:
    session = MagicMock()
    agent = MagicMock()
    agent.chat_ctx = ChatContext.empty()

    async def update_chat_ctx(chat_ctx: ChatContext) -> None:
        agent.chat_ctx = chat_ctx

    agent.update_chat_ctx = AsyncMock(side_effect=update_chat_ctx)
    session.current_agent = agent
    session.history = MagicMock()
    session._global_run_state = None
    activity = MagicMock()
    activity.agent = agent
    session.wait_for_idle = AsyncMock(return_value=activity)
    speech = MagicMock()
    speech.id = "speech"
    session.generate_reply = MagicMock(return_value=speech)
    return session


@pytest.mark.asyncio
async def test_outbound_message_needs_no_inbound_and_has_origin_metadata() -> None:
    sent = asyncio.Event()

    @background(name="research", description="Investigates requests.")
    async def research(ctx: BackgroundContext) -> None:
        await ctx.send("found an answer")
        sent.set()
        await asyncio.Event().wait()

    session = _make_session()
    manager = _BackgroundRuntimeManager([research], session=session)
    manager.start()
    await sent.wait()

    [message] = session.history.insert.call_args.args[0]
    assert message.role == "user"
    assert message.text_content == (
        "Background session `research` sent an update:\nfound an answer"
    )
    assert message.extra["background_session_id"] == "research"
    assert message.extra["background_message_id"]
    assert message.extra["background_message_id"] != message.id
    await manager.aclose()


@pytest.mark.asyncio
async def test_outbound_updates_eagerly_enter_active_context_and_history_once() -> None:
    @background(name="research", description="Investigates requests.")
    async def research(ctx: BackgroundContext) -> None:
        del ctx

    session = _make_session()

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    session.wait_for_idle = AsyncMock(side_effect=wait_forever)
    manager = _BackgroundRuntimeManager([research], session=session)
    context = manager._runtimes["research"].context

    await context.send("first")
    await context.send("second")

    messages = session.current_agent.chat_ctx.messages()
    assert [message.role for message in messages] == ["user", "user"]
    assert [message.text_content for message in messages] == [
        "Background session `research` sent an update:\nfirst",
        "Background session `research` sent an update:\nsecond",
    ]
    assert session.history.insert.call_count == 2
    inserted = [call.args[0][0] for call in session.history.insert.call_args_list]
    assert inserted == messages
    message_ids = [message.extra["background_message_id"] for message in messages]
    assert len(set(message_ids)) == 2
    assert all(message.extra["background_session_id"] == "research" for message in messages)

    await manager.aclose()


@pytest.mark.asyncio
async def test_one_background_coalesces_pending_updates_after_idle() -> None:
    @background(name="research", description="Investigates requests.")
    async def research(ctx: BackgroundContext) -> None:
        del ctx

    idle = asyncio.Event()
    session = _make_session()
    activity = MagicMock()
    activity.agent = session.current_agent

    async def wait_for_idle() -> object:
        await idle.wait()
        return activity

    session.wait_for_idle = AsyncMock(side_effect=wait_for_idle)
    manager = _BackgroundRuntimeManager([research], session=session)
    context = manager._runtimes["research"].context

    await context.send("first")
    await context.send("second")
    await asyncio.sleep(0)
    assert session.generate_reply.call_count == 0

    idle.set()
    reply_task = manager._runtimes["research"]._reply_scheduler.reply_task
    assert reply_task is not None
    await reply_task

    session.wait_for_idle.assert_awaited_once()
    session.generate_reply.assert_called_once()
    call = session.generate_reply.call_args
    assert call.kwargs["tool_choice"] == "none"
    assert call.kwargs["instructions"] == (
        "New updates arrived from background session `research`.\n"
        "Summarize the updates naturally. Do not repeat information you already told the user."
    )
    await manager.aclose()


@pytest.mark.asyncio
async def test_different_backgrounds_schedule_separate_replies() -> None:
    async def run(ctx: BackgroundContext) -> None:
        del ctx

    first = background(name="first", description="First worker.")(run)
    second = background(name="second", description="Second worker.")(run)
    idle = asyncio.Event()
    session = _make_session()
    activity = MagicMock()
    activity.agent = session.current_agent

    async def wait_for_idle() -> object:
        await idle.wait()
        return activity

    session.wait_for_idle = AsyncMock(side_effect=wait_for_idle)
    manager = _BackgroundRuntimeManager([first, second], session=session)

    await manager._runtimes["first"].context.send("one")
    await manager._runtimes["second"].context.send("two")
    assert (
        manager._runtimes["first"]._reply_scheduler
        is not manager._runtimes["second"]._reply_scheduler
    )

    idle.set()
    reply_tasks = [runtime._reply_scheduler.reply_task for runtime in manager._runtimes.values()]
    assert all(task is not None for task in reply_tasks)
    await asyncio.gather(*(task for task in reply_tasks if task is not None))

    assert session.wait_for_idle.await_count == 2
    assert session.generate_reply.call_count == 2
    instructions = {call.kwargs["instructions"] for call in session.generate_reply.call_args_list}
    assert instructions == {
        "New updates arrived from background session `first`.\n"
        "You may have already mentioned them in a recent reply. Respond only if there is new "
        "information\n"
        "the user has not heard.",
        "New updates arrived from background session `second`.\n"
        "Summarize the updates naturally. Do not repeat information you already told the user.",
    }
    assert all(
        call.kwargs["tool_choice"] == "none" for call in session.generate_reply.call_args_list
    )
    await manager.aclose()


@pytest.mark.asyncio
async def test_handoff_before_delivery_retargets_current_agent_with_missing_context() -> None:
    @background(name="research", description="Investigates requests.")
    async def research(ctx: BackgroundContext) -> None:
        del ctx

    idle = asyncio.Event()
    session = _make_session()
    first_agent = session.current_agent
    second_agent = MagicMock()
    second_agent.chat_ctx = ChatContext.empty()

    async def update_second_chat_ctx(chat_ctx: ChatContext) -> None:
        second_agent.chat_ctx = chat_ctx

    second_agent.update_chat_ctx = AsyncMock(side_effect=update_second_chat_ctx)
    activity = MagicMock()
    activity.agent = second_agent

    async def wait_for_idle() -> object:
        await idle.wait()
        return activity

    session.wait_for_idle = AsyncMock(side_effect=wait_for_idle)
    manager = _BackgroundRuntimeManager([research], session=session)
    await manager._runtimes["research"].context.send("handoff update")
    [message] = first_agent.chat_ctx.messages()

    session.current_agent = second_agent
    idle.set()
    reply_task = manager._runtimes["research"]._reply_scheduler.reply_task
    assert reply_task is not None
    await reply_task

    session.generate_reply.assert_called_once()
    call = session.generate_reply.call_args
    assert call.kwargs["tool_choice"] == "none"
    assert "chat_ctx" not in call.kwargs
    second_agent.update_chat_ctx.assert_awaited_once()
    assert second_agent.chat_ctx.messages() == [message]
    session.history.insert.assert_called_once_with([message])
    await manager.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted", [False, True])
async def test_reply_completion_does_not_corrupt_later_delivery(
    interrupted: bool,
) -> None:
    @background(name="research", description="Investigates requests.")
    async def research(ctx: BackgroundContext) -> None:
        del ctx

    callbacks: list[object] = []

    def make_speech(id: str) -> MagicMock:
        speech = MagicMock()
        speech.id = id
        speech.interrupted = id == "first" and interrupted
        speech.chat_items = [] if speech.interrupted else [MagicMock()]
        speech.add_done_callback.side_effect = callbacks.append
        return speech

    first_speech = make_speech("first")
    second_speech = make_speech("second")
    session = _make_session()
    session.generate_reply.side_effect = [first_speech, second_speech]
    manager = _BackgroundRuntimeManager([research], session=session)
    runtime = manager._runtimes["research"]

    await runtime.context.send("first update")
    first_reply_task = runtime._reply_scheduler.reply_task
    assert first_reply_task is not None
    await first_reply_task
    assert len(callbacks) == 1
    callbacks[0](first_speech)  # type: ignore[operator]

    await runtime.context.send("second update")
    second_reply_task = runtime._reply_scheduler.reply_task
    assert second_reply_task is not None and second_reply_task is not first_reply_task
    await second_reply_task

    assert session.generate_reply.call_count == 2
    assert len(callbacks) == 2
    assert all(
        call.kwargs["tool_choice"] == "none" for call in session.generate_reply.call_args_list
    )
    await manager.aclose()


@pytest.mark.asyncio
async def test_reused_definition_has_independent_outbound_schedulers() -> None:
    @background(name="research", description="Investigates requests.")
    async def research(ctx: BackgroundContext) -> None:
        del ctx

    first_session = _make_session()
    second_session = _make_session()
    first = _BackgroundRuntimeManager([research], session=first_session)
    second = _BackgroundRuntimeManager([research], session=second_session)

    await first._runtimes["research"].context.send("first")
    await second._runtimes["research"].context.send("second")
    first_scheduler = first._runtimes["research"]._reply_scheduler
    second_scheduler = second._runtimes["research"]._reply_scheduler
    assert first_scheduler is not second_scheduler
    assert first_scheduler.reply_task is not None
    assert second_scheduler.reply_task is not None
    await asyncio.gather(first_scheduler.reply_task, second_scheduler.reply_task)

    [first_message] = first_session.history.insert.call_args.args[0]
    [second_message] = second_session.history.insert.call_args.args[0]
    assert first_message.text_content.endswith("\nfirst")
    assert second_message.text_content.endswith("\nsecond")
    assert (
        first_message.extra["background_message_id"]
        != second_message.extra["background_message_id"]
    )
    await first.aclose()
    await second.aclose()


@pytest.mark.asyncio
async def test_decorator_templates_win_without_merging_session_options() -> None:
    sent = asyncio.Event()

    @background(
        name="research",
        description="Investigates requests.",
        background_handling={"update_template": "decorator: {message}"},
    )
    async def research(ctx: BackgroundContext) -> None:
        await ctx.send("answer")
        sent.set()
        await asyncio.Event().wait()

    session = _make_session()
    manager = _BackgroundRuntimeManager(
        [research],
        session=session,
        background_handling={
            "update_template": "session: {message}",
            "reply_at_tail_template": "session reply",
        },
    )
    manager.start()
    await sent.wait()
    runtime = manager._runtimes["research"]
    assert runtime._reply_scheduler.reply_task is not None
    await runtime._reply_scheduler.reply_task

    [message] = session.history.insert.call_args.args[0]
    assert message.text_content == "decorator: answer"
    instructions = session.generate_reply.call_args.kwargs["instructions"]
    assert instructions == (
        "New updates arrived from background session `research`.\n"
        "Summarize the updates naturally. Do not repeat information you already told the user."
    )
    await manager.aclose()


@pytest.mark.asyncio
async def test_session_templates_receive_exact_prompt_args_and_message_ids() -> None:
    update_args: list[dict[str, object]] = []
    reply_args: list[dict[str, object]] = []
    sent = asyncio.Event()

    def update_template(args: dict[str, object]) -> str:
        update_args.append(args)
        return f"rendered: {args['message']}"

    def reply_template(args: dict[str, object]) -> str:
        reply_args.append(args)
        return "reply"

    @background(name="research", description="Investigates requests.")
    async def research(ctx: BackgroundContext) -> None:
        await ctx.send("answer")
        sent.set()
        await asyncio.Event().wait()

    session = _make_session()
    manager = _BackgroundRuntimeManager(
        [research],
        session=session,
        background_handling={
            "update_template": update_template,  # type: ignore[typeddict-item]
            "reply_at_tail_template": reply_template,  # type: ignore[typeddict-item]
        },
    )
    manager.start()
    await sent.wait()
    runtime = manager._runtimes["research"]
    assert runtime._reply_scheduler.reply_task is not None
    await runtime._reply_scheduler.reply_task

    assert update_args == [
        {
            "background_session_id": "research",
            "background_session_description": "Investigates requests.",
            "message": "answer",
        }
    ]
    [message] = session.history.insert.call_args.args[0]
    assert reply_args == [
        {
            "background_session_id": "research",
            "message_ids": [message.extra["background_message_id"]],
        }
    ]
    await manager.aclose()


def test_manager_rejects_duplicate_ids() -> None:
    @background(name="duplicate", description="First.")
    async def first(ctx: BackgroundContext) -> None:
        del ctx

    @background(name="duplicate", description="Second.")
    async def second(ctx: BackgroundContext) -> None:
        del ctx

    with pytest.raises(ValueError, match="duplicate"):
        _BackgroundRuntimeManager([first, second], session=_make_session())


@pytest.mark.asyncio
async def test_close_rejects_outbound_and_cancels_all_background_work() -> None:
    contexts: list[BackgroundContext] = []
    started = asyncio.Event()
    closed_in_finally: list[str] = []

    async def run(ctx: BackgroundContext) -> None:
        contexts.append(ctx)
        await ctx.send("pending")
        if len(contexts) == 2:
            started.set()
        try:
            await asyncio.Event().wait()
        finally:
            with pytest.raises(RuntimeError, match="closed"):
                await ctx.send("too late")
            closed_in_finally.append(ctx.id)

    first = background(name="first", description="First.")(run)
    second = background(name="second", description="Second.")(run)
    session = _make_session()

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    session.wait_for_idle = AsyncMock(side_effect=wait_forever)
    manager = _BackgroundRuntimeManager([first, second], session=session)
    assert (
        manager._runtimes["first"]._reply_scheduler
        is not manager._runtimes["second"]._reply_scheduler
    )
    manager.start()
    await started.wait()
    tasks = [runtime._task for runtime in manager._runtimes.values()]
    reply_tasks = [runtime._reply_scheduler.reply_task for runtime in manager._runtimes.values()]

    await manager.aclose()

    assert sorted(closed_in_finally) == ["first", "second"]
    assert all(task is not None and task.done() for task in tasks)
    assert all(task is not None and task.done() for task in reply_tasks)
    with pytest.raises(RuntimeError, match="closed"):
        await contexts[0].send("after close")


@pytest.mark.asyncio
async def test_context_send_rejects_close_during_blocked_enqueue() -> None:
    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        del ctx

    update_started = asyncio.Event()
    release_update = asyncio.Event()
    session = _make_session()

    async def blocked_update(chat_ctx: ChatContext) -> None:
        update_started.set()
        await release_update.wait()
        session.current_agent.chat_ctx = chat_ctx

    session.current_agent.update_chat_ctx = AsyncMock(side_effect=blocked_update)
    manager = _BackgroundRuntimeManager([worker], session=session)
    context = manager._runtimes["worker"].context

    send_task = asyncio.create_task(context.send("racing update"))
    await update_started.wait()
    close_task = asyncio.create_task(manager.aclose())
    await asyncio.sleep(0)
    release_update.set()

    send_result, close_result = await asyncio.gather(send_task, close_task, return_exceptions=True)

    assert isinstance(send_result, RuntimeError)
    assert "closed" in str(send_result)
    assert close_result is None
    assert session.history.insert.call_count == 0
    assert manager._runtimes["worker"]._reply_scheduler.reply_task is None
    assert not any(
        call.args[0] == "background_message_updated" for call in session.emit.call_args_list
    )


@pytest.mark.asyncio
async def test_manager_close_cancels_entrypoint_blocked_in_send() -> None:
    update_started = asyncio.Event()
    release_update = asyncio.Event()

    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        await ctx.send("blocked")

    session = _make_session()

    async def blocked_update(chat_ctx: ChatContext) -> None:
        update_started.set()
        await release_update.wait()
        session.current_agent.chat_ctx = chat_ctx

    session.current_agent.update_chat_ctx = AsyncMock(side_effect=blocked_update)
    manager = _BackgroundRuntimeManager([worker], session=session)
    manager.start()
    await update_started.wait()

    try:
        await asyncio.wait_for(manager.aclose(), timeout=1)
    finally:
        release_update.set()
        await manager.aclose()

    runtime = manager._runtimes["worker"]
    assert runtime._task is not None and runtime._task.done()
    assert runtime._reply_scheduler.reply_task is None
    assert runtime._reply_scheduler._pending == []
    session.generate_reply.assert_not_called()


@pytest.mark.asyncio
async def test_entrypoint_failure_is_logged_and_not_restarted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = 0

    @background(name="broken", description="Always fails.")
    async def broken(ctx: BackgroundContext) -> None:
        nonlocal calls
        del ctx
        calls += 1
        raise RuntimeError("boom")

    session = _make_session()
    manager = _BackgroundRuntimeManager([broken], session=session)
    manager.start()
    task = manager._runtimes["broken"]._task
    assert task is not None
    await task

    assert calls == 1
    assert session.generate_reply.call_count == 0
    assert any(
        "background session entrypoint failed" in record.message
        and getattr(record, "background_session_id", None) == "broken"
        for record in caplog.records
    )
    await manager.aclose()


# ---------------------------------------------------------------------------
# Real-time state (lk_background_state) and silent context-only insertion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_tool_returns_reported_state_or_default_message() -> None:
    @background(name="research", description="Investigates requests.")
    async def research(ctx: BackgroundContext) -> None:
        del ctx

    session = AgentSession(vad=None, background=[research])
    tool = llm.ToolContext(session.tools).get_function_tool(_BACKGROUND_STATE_TOOL_NAME)
    assert tool is not None

    result = await tool(background_session_id="research")
    assert result == "The background session has not reported any state yet."

    assert session._background_manager is not None
    context = session._background_manager._runtimes["research"].context
    state = {"phase": "searching", "topic": "EV chargers", "sources_read": 4}
    context.set_state(state)

    assert await tool(background_session_id="research") == state

    context.set_state("verifying claim 3 of 10")
    assert await tool(background_session_id="research") == "verifying claim 3 of 10"


@pytest.mark.asyncio
async def test_state_tool_rejects_unknown_id_with_valid_ids() -> None:
    @background(name="alpha", description="First worker.")
    async def alpha(ctx: BackgroundContext) -> None:
        del ctx

    session = AgentSession(vad=None, background=[alpha])
    tool = llm.ToolContext(session.tools).get_function_tool(_BACKGROUND_STATE_TOOL_NAME)
    assert tool is not None

    with pytest.raises(llm.ToolError, match=r"Valid IDs: alpha"):
        await tool(background_session_id="missing")


@pytest.mark.asyncio
async def test_set_state_never_inserts_context_or_schedules_reply() -> None:
    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        del ctx

    session = _make_session()
    manager = _BackgroundRuntimeManager([worker], session=session)
    runtime = manager._runtimes["worker"]

    runtime.context.set_state({"status": "busy"})

    assert runtime.state == {"status": "busy"}
    assert list(session.current_agent.chat_ctx.items) == []
    session.history.insert.assert_not_called()
    session.generate_reply.assert_not_called()
    session.emit.assert_not_called()
    assert runtime._reply_scheduler.reply_task is None
    await manager.aclose()


@pytest.mark.asyncio
async def test_set_state_validates_tool_return_shape_and_close() -> None:
    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        del ctx

    session = _make_session()
    manager = _BackgroundRuntimeManager([worker], session=session)
    context = manager._runtimes["worker"].context

    with pytest.raises(ValueError, match="background state must be a valid function-tool"):
        context.set_state(object())

    context.set_state(["ok", 1, {"nested": True}])

    await manager.aclose()
    with pytest.raises(RuntimeError, match="background session 'worker' is closed"):
        context.set_state("late")


@pytest.mark.asyncio
async def test_silent_send_inserts_context_only_without_reply() -> None:
    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        del ctx

    session = _make_session()
    manager = _BackgroundRuntimeManager([worker], session=session)
    runtime = manager._runtimes["worker"]

    await runtime.context.send("edited src/app.py", silent=True)

    items = list(session.current_agent.chat_ctx.items)
    assert len(items) == 1
    assert items[0].role == "user"
    assert "edited src/app.py" in items[0].text_content
    assert items[0].extra["background_session_id"] == "worker"
    session.history.insert.assert_called_once()

    assert runtime._reply_scheduler.reply_task is None
    session.generate_reply.assert_not_called()

    background_calls = [
        call for call in session.emit.call_args_list if call.args[0] == "background_message_updated"
    ]
    assert len(background_calls) == 1
    event = background_calls[0].args[1]
    assert event.update.type == "background_message_received"
    assert event.update.silent is True
    assert event.update.content == "edited src/app.py"
    await manager.aclose()


@pytest.mark.asyncio
async def test_silent_send_rejected_after_close() -> None:
    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        del ctx

    session = _make_session()
    manager = _BackgroundRuntimeManager([worker], session=session)
    context = manager._runtimes["worker"].context
    await manager.aclose()

    with pytest.raises(RuntimeError, match="background session 'worker' is closed"):
        await context.send("late note", silent=True)


def test_session_background_state_conflict_fails_during_construction() -> None:
    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        del ctx

    @llm.function_tool(name=_BACKGROUND_STATE_TOOL_NAME)
    async def conflicting_state(background_session_id: str) -> str:
        return background_session_id

    with pytest.raises(ValueError, match=rf"{_BACKGROUND_STATE_TOOL_NAME} is reserved"):
        AgentSession(vad=None, tools=[conflicting_state], background=[worker])


@pytest.mark.asyncio
async def test_silent_send_retargets_when_handoff_races_insert() -> None:
    @background(name="worker", description="Handles work.")
    async def worker(ctx: BackgroundContext) -> None:
        del ctx

    session = _make_session()
    agent_a = session.current_agent
    agent_b = MagicMock()
    agent_b.chat_ctx = ChatContext.empty()

    async def update_b(chat_ctx: ChatContext) -> None:
        agent_b.chat_ctx = chat_ctx

    agent_b.update_chat_ctx = AsyncMock(side_effect=update_b)

    async def update_a(chat_ctx: ChatContext) -> None:
        agent_a.chat_ctx = chat_ctx
        session.current_agent = agent_b  # handoff during the awaited insert

    agent_a.update_chat_ctx = AsyncMock(side_effect=update_a)

    manager = _BackgroundRuntimeManager([worker], session=session)
    runtime = manager._runtimes["worker"]

    await runtime.context.send("edited src/app.py", silent=True)

    # the silent note followed the handoff onto the now-current agent
    [item_b] = agent_b.chat_ctx.items
    assert "edited src/app.py" in item_b.text_content
    [item_a] = agent_a.chat_ctx.items
    assert item_a.id == item_b.id
    session.history.insert.assert_called_once()
    assert runtime._reply_scheduler.reply_task is None
    session.generate_reply.assert_not_called()
    await manager.aclose()
