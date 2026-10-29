from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, CloseEvent, RunContext, function_tool
from livekit.agents.llm import ChatMessage, FunctionToolCall, ToolError
from livekit.agents.stt import STTCapabilities
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_llm import FakeLLM, FakeLLMResponse, FakeLLMStream
from .fake_stt import FakeSTT

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


class _KeytermSTT(FakeSTT):
    """FakeSTT that accepts keyterms so the session-level detector binds to it."""

    def __init__(self) -> None:
        super().__init__()
        self._capabilities = STTCapabilities(streaming=True, interim_results=False, keyterms=True)

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        pass


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


@pytest.mark.asyncio
@pytest.mark.virtual_time
async def test_cancelled_drain_allows_inline_task() -> None:
    drain_started = asyncio.Event()
    task_completed = asyncio.Event()

    class InlineTask(AgentTask[str]):
        def __init__(self) -> None:
            super().__init__(instructions="Run an inline task")

        async def on_enter(self) -> None:
            self.complete("completed")

    class Source(Agent):
        @function_tool
        async def start_task(self) -> str:
            """Run an inline task."""
            result = await InlineTask()
            task_completed.set()
            return result

    session = AgentSession(
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="start_task",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(name="start_task", arguments="{}", call_id="task")
                    ],
                )
            ]
        )
    )

    async def drain() -> None:
        drain_started.set()
        await session.drain()

    async with session:
        await session.start(Source(instructions="Source agent"))
        assert session._activity is not None
        async with session._activity._lock:
            drain_task = asyncio.create_task(drain())
            await asyncio.wait_for(drain_started.wait(), timeout=5.0)
            drain_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await drain_task

        await asyncio.wait_for(session.run(user_input="start_task"), timeout=5.0)
        await asyncio.wait_for(task_completed.wait(), timeout=5.0)


@pytest.mark.asyncio
@pytest.mark.virtual_time
@pytest.mark.parametrize("generation_pending", [False, True])
async def test_public_drain_while_tool_awaits_agent_task(
    generation_pending: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    release_generation = asyncio.Event()
    task_attempted = asyncio.Event()
    task_completed = asyncio.Event()
    tool_task: asyncio.Task | None = None
    drain_task: asyncio.Task | None = None

    async def stream_response(stream: FakeLLMStream) -> None:
        if stream._get_index_text() == "start_task":
            stream._send_chunk(
                tool_calls=[FunctionToolCall(name="start_task", arguments="{}", call_id="task")]
            )
            # Emit the tool call before ending generation to exercise both scheduler waits.
            await release_generation.wait()

    monkeypatch.setattr(FakeLLMStream, "_run", stream_response)

    class InlineTask(AgentTask[str]):
        def __init__(self) -> None:
            super().__init__(instructions="Run an inline task")

        async def on_enter(self) -> None:
            assert drain_task is not None and drain_task.done()
            assert scheduler.done()
            self.complete("completed")

    class Source(Agent):
        @function_tool
        async def start_task(self, ctx: RunContext) -> str:
            """Run an inline task."""
            nonlocal tool_task
            tool_task = asyncio.current_task()
            ctx.speech_handle.allow_interruptions = False
            tool_started.set()
            await release_tool.wait()
            assert activity._lock.locked()
            assert activity.scheduling_paused
            assert not activity._new_turns_blocked
            assert not scheduler.done()
            assert speech._generations[-1].done() == (not generation_pending)
            task_attempted.set()
            result = await InlineTask()
            assert result == "completed"
            task_completed.set()
            return result

    session = AgentSession(llm=FakeLLM())
    try:
        await session.start(Source(instructions="Source agent"))
        activity = session._activity
        assert activity is not None
        scheduler = activity._scheduling_atask
        assert scheduler is not None
        speech = session.generate_reply(user_input="start_task")
        await asyncio.wait_for(tool_started.wait(), timeout=5.0)
        if not generation_pending:
            release_generation.set()
            await asyncio.wait_for(speech._wait_for_generation(), timeout=5.0)

        async def drain() -> None:
            await session.drain()
            assert scheduler.done()
            assert not activity._lock.locked()

        async def wait_paused() -> None:
            while not activity.scheduling_paused:
                await asyncio.sleep(0.01)

        drain_task = asyncio.create_task(drain())
        await asyncio.wait_for(wait_paused(), timeout=5.0)
        release_tool.set()
        await asyncio.wait_for(task_attempted.wait(), timeout=5.0)
        if generation_pending:
            assert activity.current_speech is speech
            assert session._activity_lock.locked()
            assert activity._lock.locked()
            assert not drain_task.done()
            release_generation.set()

        await asyncio.wait_for(asyncio.shield(drain_task), timeout=5.0)
        await asyncio.wait_for(task_completed.wait(), timeout=5.0)
        assert session._activity is activity
        assert not activity.scheduling_paused
    finally:
        release_tool.set()
        release_generation.set()
        if tool_task is not None:
            tool_task.cancel()
            await asyncio.gather(tool_task, return_exceptions=True)
        if drain_task is not None:
            await asyncio.wait_for(drain_task, timeout=5.0)
        await asyncio.wait_for(session.aclose(), timeout=5.0)


@pytest.mark.asyncio
async def test_aclose_after_update_agent_while_tool_awaits_agent_task() -> None:
    """Closing after update_agent() replaces the agent of an in-flight AgentTask must
    not deadlock: the task's unwind from inside the tool task awaits the old activity,
    whose drain waits for that very tool task."""
    tool_started = asyncio.Event()
    task_entered = asyncio.Event()
    close_events: list[CloseEvent] = []

    class InlineTask(AgentTask[str]):
        def __init__(self) -> None:
            super().__init__(instructions="Run an inline task")

        async def on_enter(self) -> None:
            task_entered.set()

    class Target(Agent):
        pass

    class Source(Agent):
        @function_tool
        async def start_task(self, ctx: RunContext) -> str:
            """Run an inline task."""
            ctx.speech_handle.allow_interruptions = False
            tool_started.set()
            return await InlineTask()

    session = AgentSession(
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="start_task",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(name="start_task", arguments="{}", call_id="task")
                    ],
                )
            ]
        )
    )
    session.on("close", close_events.append)

    await session.start(Source(instructions="Source agent"))
    session.generate_reply(user_input="start_task")
    await asyncio.wait_for(tool_started.wait(), timeout=5.0)
    await asyncio.wait_for(task_entered.wait(), timeout=5.0)

    session.update_agent(Target(instructions="Target agent"))
    await asyncio.wait_for(session.aclose(), timeout=5.0)

    assert len(close_events) == 1


@pytest.mark.asyncio
async def test_agent_task_completion_after_update_agent_returns_to_tool() -> None:
    """A tool blocked in await AgentTask() must return when the task completes after
    update_agent(); the task's unwind must not await the activity that owns the tool."""
    tool_started = asyncio.Event()
    task_entered = asyncio.Event()
    target_entered = asyncio.Event()
    tool_returned = asyncio.Event()

    class InlineTask(AgentTask[str]):
        def __init__(self) -> None:
            super().__init__(instructions="Run an inline task")

        async def on_enter(self) -> None:
            task_entered.set()

    class Target(Agent):
        async def on_enter(self) -> None:
            target_entered.set()

    inline_task = InlineTask()

    class Source(Agent):
        @function_tool
        async def start_task(self, ctx: RunContext) -> str:
            """Run an inline task."""
            ctx.speech_handle.allow_interruptions = False
            tool_started.set()
            result = await inline_task
            tool_returned.set()
            return result

    session = AgentSession(
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="start_task",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(name="start_task", arguments="{}", call_id="task")
                    ],
                )
            ]
        )
    )

    await session.start(Source(instructions="Source agent"))
    session.generate_reply(user_input="start_task")
    await asyncio.wait_for(tool_started.wait(), timeout=5.0)
    await asyncio.wait_for(task_entered.wait(), timeout=5.0)

    session.update_agent(Target(instructions="Target agent"))
    await asyncio.wait_for(target_entered.wait(), timeout=5.0)

    inline_task.complete("done")
    await asyncio.wait_for(tool_returned.wait(), timeout=5.0)

    await asyncio.wait_for(session.aclose(), timeout=5.0)


@pytest.mark.asyncio
async def test_detached_close_keeps_keyterm_detection_on_replacement() -> None:
    """The detached close of a superseded AgentTask's old activity must not unbind the
    session's keyterm detector from the replacement agent's activity."""
    tool_started = asyncio.Event()
    task_entered = asyncio.Event()
    target_entered = asyncio.Event()
    tool_returned = asyncio.Event()

    class InlineTask(AgentTask[str]):
        def __init__(self) -> None:
            super().__init__(instructions="Run an inline task")

        async def on_enter(self) -> None:
            task_entered.set()

    class Target(Agent):
        async def on_enter(self) -> None:
            target_entered.set()

    inline_task = InlineTask()

    class Source(Agent):
        @function_tool
        async def start_task(self, ctx: RunContext) -> str:
            """Run an inline task."""
            ctx.speech_handle.allow_interruptions = False
            tool_started.set()
            result = await inline_task
            tool_returned.set()
            return result

    session = AgentSession(
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="start_task",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(name="start_task", arguments="{}", call_id="task")
                    ],
                )
            ]
        ),
        stt_context_options={"keyterm_detection": {"enabled": True, "llm": FakeLLM()}},
    )
    detector = session._keyterm_detector

    try:
        await session.start(Source(instructions="Source agent", stt=_KeytermSTT()))
        session.generate_reply(user_input="start_task")
        await asyncio.wait_for(tool_started.wait(), timeout=5.0)
        await asyncio.wait_for(task_entered.wait(), timeout=5.0)

        session.update_agent(Target(instructions="Target agent", stt=_KeytermSTT()))
        await asyncio.wait_for(target_entered.wait(), timeout=5.0)
        assert detector._session is session  # bound to the replacement activity

        inline_task.complete("done")
        await asyncio.wait_for(tool_returned.wait(), timeout=5.0)
        while session._detached_activity_close_tasks:
            await asyncio.gather(*session._detached_activity_close_tasks, return_exceptions=True)

        # the late close of the superseded activity must leave the detector bound
        assert detector._session is session
        session.emit(
            "conversation_item_added",
            ConversationItemAddedEvent(
                item=ChatMessage(role="user", content=["My company is Acme"])
            ),
        )
        assert detector._detect_task is not None
        await asyncio.gather(detector._detect_task, return_exceptions=True)
    finally:
        await asyncio.wait_for(session.aclose(), timeout=5.0)


@pytest.mark.asyncio
async def test_second_agent_task_in_tool_after_update_agent_does_not_deadlock() -> None:
    """A tool that awaits a second AgentTask after update_agent() superseded the first
    must not resume the old activity while its detached close holds the lock that close
    waits on the tool task for: the second await fails instead of deadlocking."""
    tool_started = asyncio.Event()
    first_entered = asyncio.Event()
    target_entered = asyncio.Event()
    after_first = asyncio.Event()
    second_failed = asyncio.Event()
    tool_returned = asyncio.Event()
    close_events: list[CloseEvent] = []

    class InlineTask(AgentTask[str]):
        def __init__(self, entered: asyncio.Event) -> None:
            super().__init__(instructions="Run an inline task")
            self._entered = entered

        async def on_enter(self) -> None:
            self._entered.set()

    class Target(Agent):
        async def on_enter(self) -> None:
            target_entered.set()

    first = InlineTask(first_entered)
    second = InlineTask(asyncio.Event())

    class Source(Agent):
        @function_tool
        async def start_task(self, ctx: RunContext) -> str:
            """Run two inline tasks."""
            ctx.speech_handle.allow_interruptions = False
            tool_started.set()
            first_result = await first
            after_first.set()
            try:
                await second
            except ToolError:
                second_failed.set()
            tool_returned.set()
            return first_result

    session = AgentSession(
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="start_task",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(name="start_task", arguments="{}", call_id="task")
                    ],
                )
            ]
        )
    )
    session.on("close", close_events.append)
    target = Target(instructions="Target agent")

    await session.start(Source(instructions="Source agent"))
    session.generate_reply(user_input="start_task")
    await asyncio.wait_for(tool_started.wait(), timeout=5.0)
    await asyncio.wait_for(first_entered.wait(), timeout=5.0)

    session.update_agent(target)
    await asyncio.wait_for(target_entered.wait(), timeout=5.0)

    first.complete("one")
    await asyncio.wait_for(after_first.wait(), timeout=5.0)
    # `first` handed the tool's activity to a detached close; resuming it for `second`
    # would wait on the close's lock while that close waits for this very tool task
    await asyncio.wait_for(second_failed.wait(), timeout=5.0)
    await asyncio.wait_for(tool_returned.wait(), timeout=5.0)

    # the stale tool did not take the session over: the replacement agent is still current
    assert session.current_agent is target

    await asyncio.wait_for(session.aclose(), timeout=5.0)
    assert len(close_events) == 1


@pytest.mark.asyncio
async def test_late_agent_task_unwind_after_aclose_closes_superseded_activity() -> None:
    """A superseded AgentTask that completes after session.aclose() must still close the
    activity that owns the tool: the close path already finished and never reaches it."""
    tool_started = asyncio.Event()
    task_entered = asyncio.Event()
    target_entered = asyncio.Event()
    tool_returned = asyncio.Event()
    close_events: list[CloseEvent] = []

    class InlineTask(AgentTask[str]):
        def __init__(self) -> None:
            super().__init__(instructions="Run an inline task")

        async def on_enter(self) -> None:
            task_entered.set()

    class Target(Agent):
        async def on_enter(self) -> None:
            target_entered.set()

    inline_task = InlineTask()
    source_activity: list[AgentActivity] = []
    source_speech: list[SpeechHandle] = []

    class Source(Agent):
        @function_tool
        async def start_task(self, ctx: RunContext) -> str:
            """Run an inline task."""
            source_activity.append(ctx.session._activity)
            source_speech.append(ctx.speech_handle)
            ctx.speech_handle.allow_interruptions = False
            tool_started.set()
            result = await inline_task
            tool_returned.set()
            return result

    session = AgentSession(
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="start_task",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(name="start_task", arguments="{}", call_id="task")
                    ],
                )
            ]
        )
    )
    session.on("close", close_events.append)

    await session.start(Source(instructions="Source agent"))
    session.generate_reply(user_input="start_task")
    await asyncio.wait_for(tool_started.wait(), timeout=5.0)
    await asyncio.wait_for(task_entered.wait(), timeout=5.0)

    session.update_agent(Target(instructions="Target agent"))
    await asyncio.wait_for(target_entered.wait(), timeout=5.0)
    await asyncio.wait_for(session.aclose(), timeout=5.0)
    assert len(close_events) == 1

    # the superseded task is not cancelled by the close; completing it unwinds afterwards
    old_activity = source_activity[0]
    assert not old_activity._closed
    inline_task.complete("done")
    await asyncio.wait_for(tool_returned.wait(), timeout=5.0)
    while session._detached_activity_close_tasks:
        await asyncio.gather(*session._detached_activity_close_tasks, return_exceptions=True)

    assert old_activity._closed
    # the tool's result lands on a closed activity: let its speech wind down
    await asyncio.wait_for(source_speech[0], timeout=5.0)


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
@pytest.mark.virtual_time
@pytest.mark.parametrize(
    "phase",
    [
        "handoff_requested",
        "direct_handoff",
        "on_exit",
        "scheduling_paused",
        "tool_starts_during_drain",
    ],
)
async def test_handoff_while_tool_awaits_agent_task(phase: str) -> None:
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    exit_started = asyncio.Event()
    release_exit = asyncio.Event()
    task_attempted = asyncio.Event()
    target_entered = asyncio.Event()
    task_entered = asyncio.Event()
    switch_started = asyncio.Event()
    release_switch = asyncio.Event()
    tool_task: asyncio.Task | None = None
    handoff_task: asyncio.Task | None = None
    task_error: ToolError | None = None

    async def wait_until(predicate: Callable[[], bool]) -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    class InlineTask(AgentTask[str]):
        def __init__(self) -> None:
            super().__init__(instructions="Run an inline task")

        async def on_enter(self) -> None:
            task_entered.set()
            self.complete("completed")

    class Target(Agent):
        async def on_enter(self) -> None:
            target_entered.set()

    target = Target(instructions="Target agent")

    class Source(Agent):
        async def on_exit(self) -> None:
            exit_started.set()
            if phase in ("on_exit", "direct_handoff"):
                await release_exit.wait()

        @function_tool
        async def start_task(self, ctx: RunContext) -> str:
            """Run an inline task."""
            nonlocal tool_task, task_error
            tool_task = asyncio.current_task()
            ctx.speech_handle.allow_interruptions = False
            tool_started.set()
            await release_tool.wait()
            task_attempted.set()
            try:
                return await InlineTask()
            except ToolError as e:
                task_error = e
                raise

        @function_tool
        async def switch(self) -> Agent:
            """Switch to the target agent."""
            switch_started.set()
            if phase == "tool_starts_during_drain":
                await release_switch.wait()
            return target

    source = Source(instructions="Source agent")
    tool_call_delay = 1.0 if phase == "tool_starts_during_drain" else 0.0
    session = AgentSession(
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input=name,
                    content="",
                    ttft=tool_call_delay if name == "start_task" else 0,
                    duration=tool_call_delay if name == "start_task" else 0,
                    tool_calls=[FunctionToolCall(name=name, arguments="{}", call_id=name)],
                )
                for name in ("start_task", "switch")
            ]
        ),
    )
    try:
        await session.start(source)
        activity = session._activity
        assert activity is not None
        if phase == "tool_starts_during_drain":
            session.generate_reply(user_input="switch")
            await asyncio.wait_for(switch_started.wait(), timeout=5.0)
            await asyncio.wait_for(wait_until(lambda: activity.current_speech is None), timeout=5.0)
            # The next LLM response is in flight when the handoff tool returns.
            session.generate_reply(user_input="start_task", allow_interruptions=False)
            release_switch.set()
            await asyncio.wait_for(exit_started.wait(), timeout=5.0)
            await asyncio.wait_for(wait_until(lambda: activity.scheduling_paused), timeout=5.0)
            assert not tool_started.is_set()
            await asyncio.wait_for(tool_started.wait(), timeout=5.0)
            release_tool.set()
            await asyncio.wait_for(task_attempted.wait(), timeout=5.0)
        else:
            session.generate_reply(user_input="start_task")
            await asyncio.wait_for(tool_started.wait(), timeout=5.0)
            await asyncio.wait_for(wait_until(lambda: activity.current_speech is None), timeout=5.0)

        if phase == "handoff_requested":
            # Hold the transition before drain starts to test synchronous admission blocking.
            async with session._activity_lock:
                session.update_agent(target)
                release_tool.set()
                await asyncio.wait_for(task_attempted.wait(), timeout=5.0)
                assert not exit_started.is_set()
                assert task_error is not None
        elif phase in ("direct_handoff", "on_exit", "scheduling_paused"):
            if phase == "direct_handoff":
                handoff_task = asyncio.create_task(session._update_activity(target))
            else:
                session.generate_reply(user_input="switch")
            await asyncio.wait_for(exit_started.wait(), timeout=5.0)
            if phase == "scheduling_paused":
                await asyncio.wait_for(wait_until(lambda: activity.scheduling_paused), timeout=5.0)
            release_tool.set()
            await asyncio.wait_for(task_attempted.wait(), timeout=5.0)
            release_exit.set()

        await asyncio.wait_for(target_entered.wait(), timeout=5.0)
        assert task_error is not None
        assert task_error.message == (
            "An agent transition is in progress, so this tool call cannot continue. "
            "Wait until the transition is complete before retrying, if the tool is "
            "available to the new agent."
        )
        assert not task_entered.is_set()
        assert tool_task is not None and tool_task.done()
        outputs = [
            item
            for item in source.chat_ctx.items
            if item.type == "function_call_output" and item.call_id == "start_task"
        ]
        assert len(outputs) == 1
        assert outputs[0].is_error
        assert outputs[0].output == task_error.message
        await asyncio.wait_for(session.aclose(), timeout=5.0)
    finally:
        release_switch.set()
        release_exit.set()
        if tool_task is not None:
            tool_task.cancel()
            await asyncio.gather(tool_task, return_exceptions=True)
        if session._update_activity_atask is not None:
            await asyncio.wait_for(session._update_activity_atask, timeout=5.0)
        if handoff_task is not None:
            await asyncio.wait_for(handoff_task, timeout=5.0)
        await asyncio.wait_for(session.aclose(), timeout=5.0)
