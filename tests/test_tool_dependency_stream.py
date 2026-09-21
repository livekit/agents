from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.agents.llm import FunctionCall, ToolFlag

from .fake_llm import FakeLLM
from .tool_dependency_helpers import collect_terminals, dispatch_tool_stream, wait

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
async def test_malformed_default_dispatch_emits_one_error_without_starting_tool() -> None:
    body_started = asyncio.Event()
    started_calls: list[FunctionCall] = []

    @function_tool(name="prepare")
    async def prepare(ctx: RunContext) -> str:
        body_started.set()
        return "must not run"

    async def malformed_stream() -> AsyncIterator[FunctionCall]:
        yield FunctionCall(name="prepare", call_id="prepare-call", arguments="not-json")

    session = await _new_session(Agent(instructions="test", tools=[prepare]))
    execution_task, tool_output = dispatch_tool_stream(
        session,
        [prepare],
        malformed_stream(),
        tool_execution_started_cb=started_calls.append,
    )
    try:
        await asyncio.wait_for(execution_task, timeout=2)
        assert len(tool_output.output) == 1
        assert len([out for out in tool_output.output if out.fnc_call_out.is_error]) == 1
        assert "Error parsing arguments for `prepare`" in tool_output.output[0].fnc_call_out.output
        assert not started_calls
        assert tool_output.first_tool_started_fut is not None
        assert not tool_output.first_tool_started_fut.done()
        assert not body_started.is_set()
    finally:
        await asyncio.wait_for(session.aclose(), timeout=2)
        assert not _tool_tasks(), "tool or dependency tasks survived session close"


@pytest.mark.asyncio
async def test_cancelled_function_stream_cancels_readiness_and_settles_queued_dependency() -> None:
    root_started = asyncio.Event()
    dependent_started = asyncio.Event()

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
    terminals = collect_terminals(session)
    execution_task, tool_output = dispatch_tool_stream(
        session, [root, dependent], cancelled_stream()
    )
    assert tool_output.background_task is not None
    try:
        await asyncio.wait_for(tool_output.background_task, timeout=2)
        await wait(root_started, timeout=2)
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
    terminals = collect_terminals(session)
    execution_task, tool_output = dispatch_tool_stream(session, [root, dependent], failing_stream())
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
    terminals = collect_terminals(session)
    execution_task, tool_output = dispatch_tool_stream(session, [root, dependent], open_stream())
    assert tool_output.background_task is not None
    try:
        await wait(root_started, timeout=2)
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
