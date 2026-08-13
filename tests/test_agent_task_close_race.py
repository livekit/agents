from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, AgentTask
from livekit.agents.llm import ToolError

from .fake_llm import FakeLLM

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


class _SimpleTask(AgentTask):
    def __init__(self) -> None:
        super().__init__(instructions="simple task")


class _ParentAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="parent agent")
        self.entered = asyncio.Event()
        self.exit_started = asyncio.Event()
        self.task_error: BaseException | None = None

    async def on_enter(self) -> None:
        self.entered.set()
        # hold on_enter until the session close has started draining this activity
        # (on_exit runs inside drain() while it holds the activity lock), reproducing
        # a participant disconnect racing `await AgentTask()`
        await self.exit_started.wait()
        try:
            await _SimpleTask()
        except ToolError as e:
            self.task_error = e

    async def on_exit(self) -> None:
        self.exit_started.set()


class _CompletingTask(AgentTask[None]):
    def __init__(self) -> None:
        super().__init__(instructions="completing task")
        self.entered = asyncio.Event()
        self.complete_task = asyncio.Event()

    async def on_enter(self) -> None:
        self.entered.set()
        await self.complete_task.wait()
        self.complete(None)


class _TaskAwaitingParent(Agent):
    def __init__(self, task: _CompletingTask) -> None:
        super().__init__(instructions="task awaiting parent")
        self.task = task
        self.task_finished = asyncio.Event()
        self.task_error: BaseException | None = None

    async def on_enter(self) -> None:
        try:
            await self.task
        except RuntimeError as e:
            self.task_error = e
        finally:
            self.task_finished.set()


class _GatedTaskAwaitingParent(Agent):
    def __init__(self, task: AgentTask[Any]) -> None:
        super().__init__(instructions="gated task awaiting parent")
        self.task = task
        self.start_task = asyncio.Event()
        self.task_finished = asyncio.Event()

    async def on_enter(self) -> None:
        await self.start_task.wait()
        try:
            await self.task
        finally:
            self.task_finished.set()


@pytest.mark.asyncio
async def test_aclose_while_on_enter_awaits_agent_task() -> None:
    """Closing the session while on_enter awaits an AgentTask must not deadlock:
    drain() waits for the on_enter task, which waits for the activity handoff,
    which waits for the activity lock held by drain()."""
    session = AgentSession(llm=FakeLLM())
    agent = _ParentAgent()
    await session.start(agent)
    await asyncio.wait_for(agent.entered.wait(), timeout=5.0)

    await asyncio.wait_for(session.aclose(), timeout=10.0)

    assert isinstance(agent.task_error, ToolError)


@pytest.mark.asyncio
async def test_aclose_finishes_when_agent_task_activity_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed AgentTask handoff must not leave the session close waiting forever."""
    task = _CompletingTask()
    parent = _TaskAwaitingParent(task)
    session = AgentSession(llm=FakeLLM())
    await session.start(parent)
    await asyncio.wait_for(task.entered.wait(), timeout=5.0)

    activity = task._activity
    assert activity is not None
    original_close_session = activity._close_session
    close_attempts = 0

    async def fail_close_session() -> None:
        nonlocal close_attempts
        close_attempts += 1
        await original_close_session()
        if close_attempts == 1:
            raise RuntimeError("simulated AgentActivity._close_session failure")

    monkeypatch.setattr(activity, "_close_session", fail_close_session)
    task.complete_task.set()
    await asyncio.wait_for(parent.task_finished.wait(), timeout=5.0)

    assert isinstance(parent.task_error, RuntimeError)
    await asyncio.wait_for(session.aclose(), timeout=1.0)

    assert close_attempts == 2
    assert activity._closed
    assert task._activity is None
    assert not session._started


@pytest.mark.asyncio
async def test_aclose_finishes_when_agent_task_setup_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation during AgentTask setup must not leave session close waiting forever."""
    task = _SimpleTask()
    parent = _GatedTaskAwaitingParent(task)
    session = AgentSession(llm=FakeLLM())
    await session.start(parent)

    original_update_activity = session._update_activity
    setup_started = asyncio.Event()

    async def pause_after_task_activity_starts(agent: Agent, **kwargs: Any) -> None:
        await original_update_activity(agent, **kwargs)
        if agent is task:
            setup_started.set()
            await asyncio.Future()

    monkeypatch.setattr(session, "_update_activity", pause_after_task_activity_starts)
    parent.start_task.set()
    await asyncio.wait_for(setup_started.wait(), timeout=5.0)

    task_activity = task._activity
    assert task_activity is not None
    parent_activity = parent._activity
    assert parent_activity is not None
    parent_on_enter = parent_activity._on_enter_task
    assert parent_on_enter is not None
    parent_on_enter.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await parent_on_enter
    await asyncio.wait_for(parent.task_finished.wait(), timeout=5.0)

    await asyncio.wait_for(session.aclose(), timeout=1.0)

    assert task_activity._closed
    assert not session._started
