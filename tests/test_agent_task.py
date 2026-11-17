from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool
from livekit.agents.beta.workflows import TaskGroup
from livekit.agents.llm import ChatMessage, FunctionToolCall, ToolError
from livekit.agents.voice.events import CloseReason, UserStateChangedEvent

from .fake_llm import FakeLLM, FakeLLMResponse, FakeLLMStream
from .fake_realtime import FakeRealtimeModel
from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


def _tool_call(name: str, call_id: str, arguments: str = "{}") -> FunctionToolCall:
    return FunctionToolCall(name=name, arguments=arguments, call_id=call_id)


def _reply(input: str, content: str = "", *tool_calls: FunctionToolCall) -> FakeLLMResponse:
    return FakeLLMResponse(
        input=input, content=content, ttft=0, duration=0, tool_calls=list(tool_calls)
    )


class FinishTask(AgentTask):
    """Greets on entry and completes on its finish tool, so it needs a user turn."""

    def __init__(self, greeting: str = "task_greeting") -> None:
        super().__init__(instructions="finish task")
        self._greeting = greeting

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions=self._greeting)

    @function_tool
    async def finish(self, ctx: RunContext) -> str:
        """Called to complete the task."""
        self.complete(None)
        return "done"


# -- task chains and nesting --------------------------------------------------------------


class GreetingTask(AgentTask[None]):
    def __init__(self) -> None:
        super().__init__(instructions="greeting")

    async def on_enter(self) -> None:
        await self.session.say("am I speaking with the account holder?")

    @function_tool
    async def holder(self, ctx: RunContext) -> str:
        """Called when the user confirms they are the account holder."""
        self.complete(None)
        return "ok"


class IdTask(AgentTask[None]):
    def __init__(self) -> None:
        super().__init__(instructions="id")

    async def on_enter(self) -> None:
        await self.session.say("could you confirm your ID number?")

    @function_tool
    async def confirmed(self, ctx: RunContext) -> str:
        """Called when the user confirms the ID number."""
        self.complete(None)
        return "ok"


class Closing(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="closing")

    async def on_enter(self) -> None:
        await self.session.say("thank you, transferring you now.")


class ChainedEntry(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="entry")

    async def on_enter(self) -> None:
        await GreetingTask()
        await IdTask()
        self.session.update_agent(Closing())


class GroupedEntry(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="entry")

    async def on_enter(self) -> None:
        await (
            TaskGroup(summarize_chat_ctx=False)
            .add(GreetingTask, id="greeting", description="greet the user")
            .add(IdTask, id="id", description="confirm the id number")
        )
        self.session.update_agent(Closing())


@pytest.mark.parametrize("entry", [ChainedEntry, GroupedEntry])
async def test_next_task_speech_is_recorded_in_the_run(entry: type[Agent]) -> None:
    llm = FakeLLM(
        fake_responses=[
            _reply("yes, speaking", "", _tool_call("holder", "c1")),
            _reply("yes, that is right", "", _tool_call("confirmed", "c2")),
            _reply("ok"),
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(entry())
        # the start-time handoff into the first task is still in flight when start() returns
        greeting = "am I speaking with the account holder?"
        for _ in range(100):
            if any(i.type == "message" and i.text_content == greeting for i in sess.history.items):
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("the first task never spoke")

        result = await asyncio.wait_for(sess.run(user_input="yes, speaking"), timeout=5.0)
        result.expect.next_event().is_function_call(name="holder")
        result.expect.next_event().is_function_call_output()
        assert [e.item.text_content for e in result.events if e.type == "message"] == [
            "could you confirm your ID number?"
        ]

        result = await asyncio.wait_for(sess.run(user_input="yes, that is right"), timeout=5.0)
        result.expect.next_event().is_function_call(name="confirmed")
        result.expect.next_event().is_function_call_output()
        result.expect.contains_agent_handoff(new_agent_type=Closing)
        assert [e.item.text_content for e in result.events if e.type == "message"] == [
            "thank you, transferring you now."
        ]


async def test_tool_resumes_in_the_run_after_task_completes_in_on_enter() -> None:
    class InstantTask(AgentTask[str]):
        async def on_enter(self) -> None:
            self.complete("found")

    class LookupEntry(Agent):
        @function_tool
        async def lookup(self, ctx: RunContext) -> str:
            """Look the account up."""
            result = await InstantTask(instructions="instant")
            self.session.say("one moment while I pull that up.")
            return result

    llm = FakeLLM(fake_responses=[_reply("look it up", "", _tool_call("lookup", "c1"))])
    async with AgentSession(llm=llm) as sess:
        await sess.start(LookupEntry(instructions="entry"))

        result = await asyncio.wait_for(sess.run(user_input="look it up"), timeout=5.0)
        result.expect.contains_function_call_output(output="found")
        assert [e.item.text_content for e in result.events if e.type == "message"] == [
            "one moment while I pull that up."
        ]


async def test_caught_task_startup_failure_does_not_block_later_run() -> None:
    recovered = asyncio.Event()
    release_parent = asyncio.Event()
    failed_task = AgentTask[None](
        instructions="Invalid realtime configuration.",
        llm=FakeRealtimeModel(),
        turn_handling={"interruption": {"enabled": False}},
    )

    class Parent(Agent):
        async def on_enter(self) -> None:
            with pytest.raises(ValueError, match="allow_interruptions cannot be False"):
                await failed_task
            recovered.set()
            await release_parent.wait()

    async with AgentSession(llm=FakeLLM(fake_responses=[_reply("hello", "Hello.")])) as session:
        try:
            await session.start(
                Parent(instructions="Recover from startup failure."), session_host=False
            )
            await asyncio.wait_for(recovered.wait(), timeout=5)

            result = await asyncio.wait_for(session.run(user_input="hello"), timeout=5)
            assert [
                event.item.text_content
                for event in result.events
                if event.type == "message" and event.item.role == "assistant"
            ] == ["Hello."]
            assert not session._run_suspensions
            assert not failed_task.done()
        finally:
            failed_task.cancel()
            release_parent.set()


class OuterTask(AgentTask):
    def __init__(self) -> None:
        super().__init__(instructions="outer task")

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="outer_greeting")

    @function_tool
    async def start_inner(self, ctx: RunContext) -> str:
        """Transitions into the inner task."""
        await FinishTask("inner_greeting")
        self.complete(None)
        return "inner completed"


class RootAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="root agent")

    @function_tool
    async def start_outer(self, ctx: RunContext) -> str:
        """Transitions into OuterTask."""
        await OuterTask()
        return "outer completed"


async def test_nested_agent_task_needing_user_input_returns_the_run() -> None:
    llm = FakeLLM(
        fake_responses=[
            _reply("go", "", _tool_call("start_outer", "call_1")),
            _reply("outer_greeting", "", _tool_call("start_inner", "call_2")),
            _reply("inner_greeting", "hello from inner"),
            _reply("done", "", _tool_call("finish", "call_3")),
            _reply("inner completed"),
            _reply("outer completed", "all done"),
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(RootAgent())

        await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)
        assert isinstance(sess.current_agent, FinishTask)
        await asyncio.wait_for(sess.run(user_input="done"), timeout=5.0)


async def test_nested_agent_task_from_a_later_turn() -> None:
    """A tool of a later turn awaiting a nested task is let past the outer task's floor hold."""
    llm = FakeLLM(
        fake_responses=[
            _reply("go", "", _tool_call("start_outer", "call_1")),
            _reply("outer_greeting", "what is your name?"),
            _reply("Dana", "", _tool_call("start_inner", "call_2")),
            _reply("inner_greeting", "and your date of birth?"),
        ]
    )
    sess = AgentSession(llm=llm)
    try:
        await sess.start(RootAgent())

        await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)
        assert isinstance(sess.current_agent, OuterTask)

        await asyncio.wait_for(sess.run(user_input="Dana"), timeout=5.0)
        assert isinstance(sess.current_agent, FinishTask)
    finally:
        # bounded so a regression reports the assertion instead of a hung close
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(sess.aclose(), timeout=30.0)


async def test_handoff_from_pre_run_speech() -> None:
    """A handoff from on_enter speech keeps the overlapping run alive until the task starts."""

    class EnterHandoffAgent(Agent):
        async def on_enter(self) -> None:
            await self.session.generate_reply(instructions="enter_greeting")

        @function_tool
        async def start_task(self, ctx: RunContext) -> str:
            """Transitions into the task."""
            await FinishTask()
            return "task completed"

    llm = FakeLLM(
        fake_responses=[
            # slow enough that run(user_input="hi") starts before the tool call lands
            FakeLLMResponse(
                input="enter_greeting",
                content="",
                ttft=1.0,
                duration=1.0,
                tool_calls=[_tool_call("start_task", "call_1")],
            ),
            FakeLLMResponse(input="hi", content="hello!", ttft=1.0, duration=2.0),
            _reply("task_greeting", "hello from task"),
            _reply("bye", "", _tool_call("finish", "call_2")),
            _reply("task completed", "all done"),
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(EnterHandoffAgent(instructions="root agent"))

        await asyncio.wait_for(sess.run(user_input="hi"), timeout=5.0)
        assert isinstance(sess.current_agent, FinishTask)

        await asyncio.wait_for(sess.run(user_input="bye"), timeout=5.0)
        assert isinstance(sess.current_agent, EnterHandoffAgent)


async def test_parallel_agent_tasks_run_in_turn() -> None:
    """Tasks awaited from one turn's parallel tool calls queue and run one after the other."""

    class ParallelDialogAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="root agent")
            self.outcomes: list[str] = []

        @function_tool
        async def open_name_dialog(self, ctx: RunContext) -> str:
            """Collects the name."""
            return await self._open("name")

        @function_tool
        async def open_email_dialog(self, ctx: RunContext) -> str:
            """Collects the email."""
            return await self._open("email")

        async def _open(self, which: str) -> str:
            # report a refusal as the tool output; re-raising would fail the awaiting run()
            try:
                await FinishTask()
            except ToolError as e:
                self.outcomes.append(f"{which}:refused")
                return f"{which} refused: {e}"
            self.outcomes.append(f"{which}:ran")
            return f"{which} captured"

    llm = FakeLLM(
        fake_responses=[
            _reply(
                "go",
                "",
                _tool_call("open_name_dialog", "call_1"),
                _tool_call("open_email_dialog", "call_2"),
            ),
            _reply("task_greeting", "what is it?"),
            _reply("done", "", _tool_call("finish", "call_3")),
        ]
    )
    agent = ParallelDialogAgent()
    sess = AgentSession(llm=llm)
    try:
        await sess.start(agent)

        await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)
        first = sess.current_agent
        assert isinstance(first, FinishTask)

        await asyncio.wait_for(sess.run(user_input="done"), timeout=5.0)
        second = sess.current_agent
        assert isinstance(second, FinishTask) and second is not first

        await asyncio.wait_for(sess.run(user_input="done"), timeout=5.0)
        assert isinstance(sess.current_agent, ParallelDialogAgent)
    finally:
        # bounded so a hung queued task reports the assertions instead of a hung close
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(sess.aclose(), timeout=30.0)

    assert sorted(agent.outcomes) == ["email:ran", "name:ran"]
    outputs = {
        item.call_id: item.output
        for item in agent.chat_ctx.items
        if item.type == "function_call_output" and item.call_id in ("call_1", "call_2")
    }
    assert set(outputs) == {"call_1", "call_2"}
    assert not any("refused" in out for out in outputs.values())


# -- update_agent and handoffs ------------------------------------------------------------


class AskNameTask(AgentTask):
    def __init__(self) -> None:
        super().__init__(instructions="ask name task")

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="ask_name")

    @function_tool
    async def record_name(self, ctx: RunContext, name: str) -> str:
        """Called when the user provides their name."""
        self.complete(None)
        return "recorded"


async def test_update_agent_on_enter_output_captured() -> None:
    """run() waits for the new agent's delayed on_enter greeting."""

    class GreetingAgent(Agent):
        async def on_enter(self) -> None:
            await asyncio.sleep(0.5)
            await self.session.generate_reply(instructions="delayed_greeting")

    class Greeter(Agent):
        @function_tool
        async def start(self, ctx: RunContext) -> None:
            """Called when the user asks to proceed."""
            self.session.update_agent(GreetingAgent(instructions="greeting agent"))

    llm = FakeLLM(
        fake_responses=[
            _reply("go", "", _tool_call("start", "call_1")),
            _reply("delayed_greeting", "hello!"),
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(Greeter(instructions="greeter agent"))

        result = await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)
        result.expect.contains_message(role="assistant")


async def test_update_agent_long_on_enter_no_deadlock() -> None:
    """run() returns when the new agent's on_enter awaits a task needing another user turn."""

    class SurveyAgent(Agent):
        async def on_enter(self) -> None:
            await AskNameTask()

    class Greeter(Agent):
        @function_tool
        async def start_survey(self, ctx: RunContext) -> None:
            """Called when the user is ready to start the survey."""
            self.session.update_agent(SurveyAgent(instructions="survey agent"))

    llm = FakeLLM(
        fake_responses=[
            _reply("ready", "", _tool_call("start_survey", "call_1")),
            _reply("ask_name", "what is your name?"),
            _reply("Bob", "", _tool_call("record_name", "call_2", '{"name": "Bob"}')),
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(Greeter(instructions="greeter agent"))

        await asyncio.wait_for(sess.run(user_input="ready"), timeout=5.0)
        second_result = await asyncio.wait_for(sess.run(user_input="Bob"), timeout=5.0)
        second_result.expect.contains_function_call(name="record_name")


async def test_update_agent_rejects_agent_running_elsewhere() -> None:
    current, occupied = Agent(instructions="current"), Agent(instructions="occupied")
    async with AgentSession() as sess, AgentSession() as other:
        await sess.start(current)
        await other.start(occupied)

        with pytest.raises(RuntimeError):
            sess.update_agent(occupied)

        assert sess.current_agent is current
        assert not sess._activity._new_turns_blocked


@pytest.mark.parametrize("save_delay", [0.0, 0.5], ids=["sibling_immediate", "sibling_delayed"])
async def test_handoff_survives_a_parallel_tool(save_delay: float) -> None:
    """A handoff batched with an ordinary tool still switches agents, whichever ends last."""

    class Greeter(Agent):
        async def on_enter(self) -> None:
            self.session.generate_reply(instructions="greet")

    class Router(Agent):
        @function_tool
        async def handoff(self, ctx: RunContext) -> Agent:
            return Greeter(instructions="greeter")

        @function_tool
        async def save_note(self, ctx: RunContext) -> str:
            await asyncio.sleep(save_delay)
            return "saved"

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm(
        "", tool_calls=[_tool_call("handoff", "call_handoff"), _tool_call("save_note", "call_save")]
    )
    actions.add_llm("hello from the greeter", input="greet")
    actions.add_tts(1.0)
    session = create_session(actions)
    messages: list[ChatMessage] = []
    session.on(
        "conversation_item_added",
        lambda ev: messages.append(ev.item) if ev.item.type == "message" else None,
    )

    await run_session(session, Router(instructions="router"), drain_delay=3.0)

    assert isinstance(session.current_agent, Greeter)
    assert "hello from the greeter" in [m.text_content for m in messages if m.role == "assistant"]


# -- drain, close and handoff races -------------------------------------------------------


async def test_cancelled_drain_allows_inline_task() -> None:
    drain_started = asyncio.Event()
    task_completed = asyncio.Event()

    class InlineTask(AgentTask[str]):
        async def on_enter(self) -> None:
            self.complete("completed")

    class Source(Agent):
        @function_tool
        async def start_task(self) -> str:
            """Run an inline task."""
            result = await InlineTask(instructions="Run an inline task")
            task_completed.set()
            return result

    session = AgentSession(
        llm=FakeLLM(fake_responses=[_reply("start_task", "", _tool_call("start_task", "task"))])
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
            stream._send_chunk(tool_calls=[_tool_call("start_task", "task")])
            # emit the tool call before ending generation to exercise both scheduler waits
            await release_generation.wait()

    monkeypatch.setattr(FakeLLMStream, "_run", stream_response)

    class InlineTask(AgentTask[str]):
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
            result = await InlineTask(instructions="Run an inline task")
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

        drain_task = asyncio.create_task(drain())
        while not activity.scheduling_paused:
            await asyncio.sleep(0.01)
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


async def test_aclose_while_on_enter_awaits_agent_task() -> None:
    """Closing the session while on_enter awaits an AgentTask does not deadlock on the lock."""

    class ParentAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="parent agent")
            self.entered = asyncio.Event()
            self.exit_started = asyncio.Event()
            self.task_error: BaseException | None = None

        async def on_enter(self) -> None:
            self.entered.set()
            # await the task only once the close is draining this activity under its lock
            await self.exit_started.wait()
            try:
                await AgentTask(instructions="simple task")
            except ToolError as e:
                self.task_error = e

        async def on_exit(self) -> None:
            self.exit_started.set()

    session = AgentSession(llm=FakeLLM())
    agent = ParentAgent()
    await session.start(agent)
    await asyncio.wait_for(agent.entered.wait(), timeout=5.0)

    await asyncio.wait_for(session.aclose(), timeout=10.0)

    assert isinstance(agent.task_error, ToolError)


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
                return await InlineTask(instructions="Run an inline task")
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
                    tool_calls=[_tool_call(name, name)],
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
            # the next LLM response is in flight when the handoff tool returns
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
            # hold the transition before drain starts to test synchronous admission blocking
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


# -- shutdown while a task speaks ---------------------------------------------------------

SESSION_TIMEOUT = 60.0
# run_session closes the session itself after this delay, the close under test comes first
DRAIN_DELAY = 20.0
QUESTION = "what is your experience"
GOODBYE = "thank you for your time, goodbye"


@function_tool
async def disqualify(context: RunContext) -> None:
    """End the interview and hang up."""
    context.session.generate_reply(instructions="say_goodbye")
    context.session.shutdown()


class InterviewTask(AgentTask[str]):
    def __init__(self) -> None:
        super().__init__(instructions="interview", tools=[disqualify])

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="ask_question")

    @function_tool
    async def finish(self, context: RunContext) -> None:
        """Thank the candidate, end the interview and hang up."""
        context.session.generate_reply(instructions="say_goodbye")
        self.complete("finished")
        context.session.shutdown()


class ParentOnEnter(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="parent")

    async def on_enter(self) -> None:
        # let run_session's audio reach the STT stream before the handoff moves it to the task
        await asyncio.sleep(0.2)
        with contextlib.suppress(ToolError):
            await InterviewTask()


class ParentWithTaskGroup(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="parent")

    async def on_enter(self) -> None:
        await asyncio.sleep(0.2)  # see ParentOnEnter
        with contextlib.suppress(ToolError):
            await TaskGroup(summarize_chat_ctx=False).add(
                InterviewTask, id="interview", description="the interview"
            )


class ParentWithTool(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="parent")

    @function_tool
    async def start_interview(self, context: RunContext) -> str:
        """Start the interview."""
        return await InterviewTask()


class ParentWithToolTaskGroup(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="parent")

    @function_tool
    async def start_interview(self, context: RunContext) -> str:
        """Start the interview."""
        result = await TaskGroup(summarize_chat_ctx=False).add(
            InterviewTask, id="interview", description="the interview"
        )
        return str(result.task_results["interview"])


async def _said_during(session: AgentSession, agent: Agent) -> list[tuple[str | None, bool]]:
    loop = asyncio.get_running_loop()
    messages: list[ChatMessage] = []
    closes: list[tuple[float, CloseReason]] = []
    session.on(
        "conversation_item_added",
        lambda ev: messages.append(ev.item) if ev.item.type == "message" else None,
    )
    session.on("close", lambda ev: closes.append((loop.time(), ev.reason)))
    started_at = loop.time()
    await asyncio.wait_for(
        run_session(session, agent, drain_delay=DRAIN_DELAY), timeout=SESSION_TIMEOUT
    )
    # closed once, by the shutdown() under test and not by run_session
    [(closed_at, reason)] = closes
    assert reason == CloseReason.USER_INITIATED and closed_at - started_at < DRAIN_DELAY
    return [(m.text_content, m.interrupted) for m in messages if m.role == "assistant"]


def _interview_actions(*, from_tool: bool, tool: str) -> FakeActions:
    actions = FakeActions()
    if from_tool:
        actions.add_user_speech(0.5, 1.0, "start")
        actions.add_llm("", tool_calls=[_tool_call("start_interview", "s")])
    actions.add_llm(QUESTION, input="ask_question")
    actions.add_tts(1.0)
    actions.add_user_speech(4.0, 5.0, "i will not answer that")
    actions.add_llm("", tool_calls=[_tool_call(tool, "t")])
    actions.add_llm(GOODBYE, input="say_goodbye")
    actions.add_tts(3.0)
    return actions


@pytest.mark.parametrize("drain", [True, False])
async def test_shutdown_while_an_agent_task_speaks(drain: bool) -> None:
    """A drained shutdown() lets the task finish its sentence; drain=False cuts it off."""
    actions = FakeActions()
    actions.add_llm(QUESTION, input="ask_question")
    actions.add_tts(5.0)
    session = create_session(actions, with_stt=False)

    asyncio.get_running_loop().call_later(2.0, lambda: session.shutdown(drain=drain))
    [(text, interrupted)] = await _said_during(session, ParentOnEnter())
    if drain:
        assert (text, interrupted) == (QUESTION, False)
    else:
        assert interrupted and text != QUESTION


@pytest.mark.parametrize(
    "parent", [ParentOnEnter, ParentWithTaskGroup, ParentWithTool, ParentWithToolTaskGroup]
)
async def test_goodbye_queued_before_shutdown_plays_out(parent: type[Agent]) -> None:
    """A goodbye queued before shutdown() plays in full, and the parent ignores the cancel."""
    from_tool = parent in (ParentWithTool, ParentWithToolTaskGroup)
    actions = _interview_actions(from_tool=from_tool, tool="disqualify")
    # what the parent would say to the task the close cancelled
    actions.add_llm("sorry, something went wrong", input="AgentTask interview_task is cancelled")
    actions.add_tts(2.0)
    session = create_session(actions)

    said = await _said_during(session, parent())
    assert said == [(QUESTION, False), (GOODBYE, False)]


async def test_task_completed_before_shutdown_keeps_goodbye_and_parent_reply() -> None:
    actions = _interview_actions(from_tool=True, tool="finish")
    actions.add_llm("the interview is finished", input="finished")
    actions.add_tts(2.0)
    session = create_session(actions)

    said = await _said_during(session, ParentWithTool())
    assert said == [(QUESTION, False), (GOODBYE, False), ("the interview is finished", False)]


# -- user away while a task waits on the user ---------------------------------------------

USER_AWAY_TIMEOUT = 2.0


class AskEmail(AgentTask[str]):
    def __init__(self) -> None:
        super().__init__(instructions="Ask for the user's e-mail address.")

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="ask_email")

    @function_tool
    async def save(self, ctx: RunContext, email: str) -> None:
        """Saves the user's e-mail address."""
        self.complete(email)


class AskName(AgentTask[str]):
    def __init__(self) -> None:
        super().__init__(instructions="Ask for the user's name.")

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="ask_name")


class CollectingAgent(Agent):
    def __init__(self, *, work_after_task: float = 0.0) -> None:
        super().__init__(instructions="root agent")
        self._work_after_task = work_after_task
        self.task_done = asyncio.Event()

    @function_tool
    async def collect_email(self, ctx: RunContext) -> str:
        """Collects the user's e-mail address."""
        email = await AskEmail()
        self.task_done.set()
        # backend work after the task, such as saving the address
        await asyncio.sleep(self._work_after_task)
        return f"saved {email}"


class TwoDialogsAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="root agent")

    @function_tool
    async def collect_name(self, ctx: RunContext) -> str:
        """Collects the user's name."""
        return await AskName()

    @function_tool
    async def collect_email(self, ctx: RunContext) -> str:
        """Collects the user's e-mail address."""
        return await AskEmail()


class LookupAndCollectAgent(Agent):
    def __init__(self, *, lookup_duration: float = USER_AWAY_TIMEOUT * 10) -> None:
        super().__init__(instructions="root agent")
        self._lookup_duration = lookup_duration

    @function_tool
    async def lookup_order(self, ctx: RunContext) -> str:
        """Looks up the user's order."""
        await asyncio.sleep(self._lookup_duration)
        return "order 42"

    @function_tool
    async def collect_email(self, ctx: RunContext) -> str:
        """Collects the user's e-mail address."""
        return await AskEmail()


def _away_llm() -> FakeLLM:
    return FakeLLM(
        fake_responses=[
            _reply("go", "", _tool_call("collect_email", "call_1")),
            _reply(
                "both",
                "",
                _tool_call("collect_name", "call_3"),
                _tool_call("collect_email", "call_4"),
            ),
            _reply(
                "lookup",
                "",
                _tool_call("lookup_order", "call_5"),
                _tool_call("collect_email", "call_6"),
            ),
            _reply("ask_name", "What is your name?"),
            _reply("ask_email", "What is your e-mail address?"),
            _reply("saved me@example.com", "Thanks!"),
        ]
    )


@pytest.mark.parametrize(
    ("agent_factory", "user_input", "waiting_task", "silence"),
    [
        # the tool awaiting the task must not hold the away timer off
        (CollectingAgent, "go", AskEmail, USER_AWAY_TIMEOUT * 3),
        # a dialog queued behind another waits on the user as much as the first
        (TwoDialogsAgent, "both", (AskName, AskEmail), USER_AWAY_TIMEOUT * 3),
        # a backend tool of the paused agent can't speak until the task ends
        (LookupAndCollectAgent, "lookup", AskEmail, USER_AWAY_TIMEOUT * 3),
        # that tool landing mid-task doesn't restart the window the task started
        (
            lambda: LookupAndCollectAgent(lookup_duration=USER_AWAY_TIMEOUT * 0.75),
            "lookup",
            AskEmail,
            USER_AWAY_TIMEOUT * 1.25,
        ),
    ],
    ids=["awaited_from_tool", "queued_dialog", "paused_backend_tool", "paused_tool_lands"],
)
async def test_user_goes_away_while_a_task_waits_on_them(
    agent_factory: Callable[[], Agent],
    user_input: str,
    waiting_task: type | tuple[type, ...],
    silence: float,
) -> None:
    async with AgentSession(llm=_away_llm(), user_away_timeout=USER_AWAY_TIMEOUT) as session:
        away: list[UserStateChangedEvent] = []
        session.on("user_state_changed", lambda ev: ev.new_state == "away" and away.append(ev))

        await session.start(agent_factory())
        await asyncio.wait_for(session.run(user_input=user_input), timeout=5.0)

        await asyncio.sleep(silence)
        assert isinstance(session.current_agent, waiting_task)
        assert len(away) == 1


async def test_the_tool_holds_the_away_timer_again_once_its_task_is_done() -> None:
    """After its task completes, a tool doing backend work holds the away timer off again."""
    agent = CollectingAgent(work_after_task=USER_AWAY_TIMEOUT * 3)

    async with AgentSession(llm=_away_llm(), user_away_timeout=USER_AWAY_TIMEOUT) as session:
        away: list[UserStateChangedEvent] = []
        session.on("user_state_changed", lambda ev: ev.new_state == "away" and away.append(ev))

        await session.start(agent)
        await asyncio.wait_for(session.run(user_input="go"), timeout=5.0)

        # the task completes without a user turn, while the timer runs
        task = session.current_agent
        assert isinstance(task, AskEmail)
        await asyncio.sleep(USER_AWAY_TIMEOUT / 2)
        task.complete("me@example.com")
        await asyncio.wait_for(agent.task_done.wait(), timeout=5.0)

        await asyncio.sleep(USER_AWAY_TIMEOUT * 2)
        assert away == []
