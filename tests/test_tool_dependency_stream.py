from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.agents.llm import FunctionCall, ToolContext, ToolFlag
from livekit.agents.voice.events import ToolCallEnded, ToolExecutionUpdatedEvent
from livekit.agents.voice.generation import perform_tool_executions
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_llm import FakeLLM

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


async def _new_session(agent: Agent) -> AgentSession:
    session = AgentSession(
        llm=FakeLLM(),
        stt=None,
        vad=None,
        tts=None,
        turn_handling={"turn_detection": None},
    )
    await session.start(agent)
    return session


async def _wait(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), timeout=2)


def _tool_tasks() -> list[asyncio.Task[object]]:
    return [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and (
            task.get_name().startswith(("tool_dependency_", "tool_exec_", "func_exec_"))
            or task.get_name() in {"execute_tools_task", "tool_dependency_ready"}
        )
    ]


async def _finish(
    session: AgentSession,
    execution_task: asyncio.Task[None],
    background_task: asyncio.Task[None],
) -> None:
    for task in (execution_task, background_task):
        if not task.done():
            task.cancel()
    await asyncio.wait_for(
        asyncio.gather(execution_task, background_task, return_exceptions=True), timeout=2
    )
    await asyncio.wait_for(session.aclose(), timeout=2)
    assert not _tool_tasks(), "tool or dependency tasks survived session close"


def _calls() -> tuple[FunctionCall, FunctionCall]:
    return (
        FunctionCall(name="root", call_id="root-call", arguments="{}"),
        FunctionCall(name="dependent", call_id="dependent-call", arguments="{}"),
    )


@pytest.mark.asyncio
async def test_cancelled_function_stream_cancels_readiness_and_settles_queued_dependency() -> None:
    root_started = asyncio.Event()
    dependent_started = asyncio.Event()
    terminals: defaultdict[str, list[ToolCallEnded]] = defaultdict(list)

    @function_tool(name="root")
    async def root(ctx: RunContext) -> str:
        root_started.set()
        return "root done"

    @function_tool(name="dependent", after=("root",))
    async def dependent(ctx: RunContext) -> str:
        dependent_started.set()
        return "must not run"

    async def cancelled_stream() -> AsyncIterator[FunctionCall]:
        root_call, dependent_call = _calls()
        yield root_call
        yield dependent_call
        raise asyncio.CancelledError

    session = await _new_session(Agent(instructions="test", tools=[root, dependent]))
    session.on(
        "tool_execution_updated",
        lambda event: (
            terminals[event.update.call_id].append(event.update)
            if isinstance(event, ToolExecutionUpdatedEvent)
            and isinstance(event.update, ToolCallEnded)
            else None
        ),
    )
    execution_task, tool_output = perform_tool_executions(
        session=session,
        speech_handle=SpeechHandle.create(),
        tool_ctx=ToolContext([root, dependent]),
        tool_choice="auto",
        function_stream=cancelled_stream(),
        tool_execution_started_cb=lambda _: None,
        tool_execution_completed_cb=lambda _: None,
    )
    assert tool_output.background_task is not None
    try:
        await asyncio.wait_for(tool_output.background_task, timeout=2)
        await _wait(root_started)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(execution_task), timeout=0.5)
        assert not dependent_started.is_set()
        assert len(terminals["dependent-call"]) == 1
        assert terminals["dependent-call"][0].status in {"error", "cancelled"}
        assert tool_output.ready_fut is not None and tool_output.ready_fut.cancelled()
        assert session._activity is not None
        assert not session._activity._dependency_schedulers
    finally:
        await _finish(session, execution_task, tool_output.background_task)


@pytest.mark.asyncio
async def test_function_stream_exception_fails_readiness_and_settles_queued_dependency() -> None:
    dependent_started = asyncio.Event()
    terminals: defaultdict[str, list[ToolCallEnded]] = defaultdict(list)

    @function_tool(name="root")
    async def root(ctx: RunContext) -> str:
        return "root done"

    @function_tool(name="dependent", after=("root",))
    async def dependent(ctx: RunContext) -> str:
        dependent_started.set()
        return "must not run"

    async def failing_stream() -> AsyncIterator[FunctionCall]:
        root_call, dependent_call = _calls()
        yield root_call
        yield dependent_call
        raise RuntimeError("stream failed")

    session = await _new_session(Agent(instructions="test", tools=[root, dependent]))
    session.on(
        "tool_execution_updated",
        lambda event: (
            terminals[event.update.call_id].append(event.update)
            if isinstance(event, ToolExecutionUpdatedEvent)
            and isinstance(event.update, ToolCallEnded)
            else None
        ),
    )
    execution_task, tool_output = perform_tool_executions(
        session=session,
        speech_handle=SpeechHandle.create(),
        tool_ctx=ToolContext([root, dependent]),
        tool_choice="auto",
        function_stream=failing_stream(),
        tool_execution_started_cb=lambda _: None,
        tool_execution_completed_cb=lambda _: None,
    )
    assert tool_output.background_task is not None
    try:
        with pytest.raises(RuntimeError, match="stream failed"):
            await asyncio.wait_for(asyncio.shield(execution_task), timeout=0.5)
        assert not dependent_started.is_set()
        assert len(terminals["dependent-call"]) == 1
        assert terminals["dependent-call"][0].status == "error"
        assert tool_output.ready_fut is not None
        assert isinstance(tool_output.ready_fut.exception(), RuntimeError)
    finally:
        await _finish(session, execution_task, tool_output.background_task)


@pytest.mark.asyncio
async def test_external_readiness_cancellation_abandons_dispatcher_without_orphans() -> None:
    root_started = asyncio.Event()
    release_root = asyncio.Event()
    dependent_started = asyncio.Event()
    terminals: defaultdict[str, list[ToolCallEnded]] = defaultdict(list)

    @function_tool(name="root", flags=ToolFlag.CANCELLABLE)
    async def root(ctx: RunContext) -> str:
        root_started.set()
        await release_root.wait()
        return "root done"

    @function_tool(name="dependent", after=("root",))
    async def dependent(ctx: RunContext) -> str:
        dependent_started.set()
        return "must not run"

    async def open_stream() -> AsyncIterator[FunctionCall]:
        root_call, dependent_call = _calls()
        yield root_call
        yield dependent_call
        await asyncio.Event().wait()

    session = await _new_session(Agent(instructions="test", tools=[root, dependent]))
    session.on(
        "tool_execution_updated",
        lambda event: (
            terminals[event.update.call_id].append(event.update)
            if isinstance(event, ToolExecutionUpdatedEvent)
            and isinstance(event.update, ToolCallEnded)
            else None
        ),
    )
    execution_task, tool_output = perform_tool_executions(
        session=session,
        speech_handle=SpeechHandle.create(),
        tool_ctx=ToolContext([root, dependent]),
        tool_choice="auto",
        function_stream=open_stream(),
        tool_execution_started_cb=lambda _: None,
        tool_execution_completed_cb=lambda _: None,
    )
    assert tool_output.background_task is not None
    try:
        await _wait(root_started)
        execution_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(execution_task, timeout=2)
        await asyncio.wait_for(tool_output.background_task, timeout=2)
        assert not dependent_started.is_set()
        assert len(terminals["dependent-call"]) == 1
        assert terminals["dependent-call"][0].status in {"error", "cancelled"}
    finally:
        release_root.set()
        await _finish(session, execution_task, tool_output.background_task)
