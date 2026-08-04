from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class Forwarded(Agent):
    """The agent handed to, speaking from on_enter after the handoff."""

    def __init__(self) -> None:
        super().__init__(instructions="forwarded agent")

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="forwarded_greeting")


class Router(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="router agent")

    @function_tool
    async def forward(self, ctx: RunContext) -> None:
        """Hand off to the forwarded agent while holding the floor."""
        # the update goes non-blocking, so the handoff below runs after the turn ended
        await ctx.update("")
        forward_agent = Forwarded()
        async with ctx.foreground():
            ctx.session.update_agent(forward_agent)


class BackgroundWorker(Agent):
    """Answers early, then keeps working for far longer than a run should last."""

    def __init__(self) -> None:
        super().__init__(instructions="background worker")
        self.cancelled = False

    @function_tool
    async def slow_job(self, ctx: RunContext) -> None:
        """Report progress, then keep working in the background."""
        await ctx.update("working on it")
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class LateFloorWorker(Agent):
    """Answers early, then asks for the floor long after its own run ended."""

    HOLD = 5.0

    def __init__(self) -> None:
        super().__init__(instructions="late floor worker")
        self.resume = asyncio.Event()

    @function_tool
    async def slow_job(self, ctx: RunContext) -> None:
        """Report progress, then take the floor much later."""
        await ctx.update("working on it")
        await self.resume.wait()
        async with ctx.foreground():
            await asyncio.sleep(self.HOLD)


class AskNameTask(AgentTask[None]):
    """A task that needs a future user turn to complete."""

    def __init__(self) -> None:
        super().__init__(instructions="ask name task")

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="ask_name")

    @function_tool
    async def record_name(self, ctx: RunContext, name: str) -> str:
        """Called when the user provides their name."""
        self.complete(None)
        return "recorded"


class Surveyor(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="surveyor agent")

    @function_tool
    async def start_survey(self, ctx: RunContext) -> None:
        """Called when the user is ready to start the survey."""
        async with ctx.foreground():
            await AskNameTask()

    @function_tool
    async def ask_over_speech(self, ctx: RunContext) -> None:
        """Called when the user wants the survey read out first."""
        async with ctx.foreground():
            await self.session.generate_reply(instructions="ask_via_tool")

    @function_tool
    async def collect_name(self, ctx: RunContext) -> str:
        """Called when the name is needed."""
        await AskNameTask()
        return "collected"


def _router_llm() -> FakeLLM:
    return FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go",
                content="",
                ttft=0.1,
                duration=0.1,
                tool_calls=[FunctionToolCall(name="forward", arguments="{}", call_id="call_1")],
            ),
            FakeLLMResponse(
                input="forwarded_greeting", content="here is a joke", ttft=0.1, duration=0.1
            ),
        ]
    )


async def test_handoff_inside_foreground_after_going_non_blocking() -> None:
    """A tool that hands off inside ``ctx.foreground()`` must still have the new agent's
    on_enter reply recorded on the same RunResult.

    ``ctx.foreground()`` waits for the activity to go idle, and going idle is what
    completes the run. Without a guard, the handoff and everything it triggers land
    after ``run()`` returned, and under the test framework the session closes first.
    """
    async with AgentSession(llm=_router_llm()) as sess:
        await sess.start(Router())

        result = await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)

        kinds = [type(e).__name__ for e in result.events]
        assert "ChatMessageEvent" in kinds, (
            f"the forwarded agent's reply never reached the run; got {kinds}"
        )


async def test_handoff_inside_foreground_event_sequence() -> None:
    """The same case asserted positionally, the way the testing guide shows."""
    async with AgentSession(llm=_router_llm()) as sess:
        await sess.start(Router())

        result = await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)

        result.expect.next_event().is_function_call(name="forward")
        result.expect.next_event().is_function_call_output()
        result.expect.next_event().is_agent_handoff(new_agent_type=Forwarded)
        result.expect.next_event().is_message(role="assistant")


async def test_background_tool_does_not_hold_the_run_open() -> None:
    """``ctx.update()`` is also how a tool answers early and keeps working. Those tools
    never take the floor, so they must not delay the run at all."""
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go",
                content="",
                ttft=0.1,
                duration=0.1,
                tool_calls=[FunctionToolCall(name="slow_job", arguments="{}", call_id="call_1")],
            ),
            FakeLLMResponse(input="", content="ok, working on it", ttft=0.1, duration=0.1),
        ]
    )
    agent = BackgroundWorker()
    async with AgentSession(llm=llm) as sess:
        await sess.start(agent)

        loop = asyncio.get_running_loop()
        started_at = loop.time()
        result = await asyncio.wait_for(sess.run(user_input="go"), timeout=60.0)
        elapsed = loop.time() - started_at

        assert "FunctionCallEvent" in [type(e).__name__ for e in result.events]
        # the tool sleeps for an hour; the run comes back on the turn's own timing
        assert elapsed < 1.0, f"the background tool held run() open for {elapsed:.2f}s"
        assert not agent.cancelled


async def test_guard_released_when_the_floor_never_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``wait_for_idle`` raises when the activity closes under a handoff. The guard is
    registered before that wait, so releasing it has to cover the failure path too."""
    from livekit.agents.voice.agent_activity import ActivityClosedError

    async def _raise(self: AgentSession) -> None:
        raise ActivityClosedError("activity closed")

    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go",
                content="",
                ttft=0.1,
                duration=0.1,
                tool_calls=[FunctionToolCall(name="forward", arguments="{}", call_id="call_1")],
            ),
            FakeLLMResponse(input="", content="ok", ttft=0.1, duration=0.1),
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(Router())
        monkeypatch.setattr(AgentSession, "wait_for_idle", _raise)

        # the tool raises out of ctx.foreground(); the run still has to come back
        await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)


async def test_stale_foreground_scope_does_not_hold_a_later_run() -> None:
    """A tool that takes the floor after its own run ended must not gate the next run.

    The guard belongs to the run that owns the turn of the tool. A later run is a
    different run, so it gets no guard and keeps its own timing.
    """
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go",
                content="",
                ttft=0.1,
                duration=0.1,
                tool_calls=[FunctionToolCall(name="slow_job", arguments="{}", call_id="call_1")],
            ),
            FakeLLMResponse(input="", content="ok, working on it", ttft=0.1, duration=0.1),
            FakeLLMResponse(input="second", content="hello again", ttft=0.1, duration=0.1),
        ]
    )
    agent = LateFloorWorker()
    async with AgentSession(llm=llm) as sess:
        await sess.start(agent)

        await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)

        second_run = sess.run(user_input="second")
        agent.resume.set()  # the tool of the finished run now asks for the floor

        loop = asyncio.get_running_loop()
        started_at = loop.time()
        await asyncio.wait_for(second_run, timeout=60.0)
        elapsed = loop.time() - started_at

        assert elapsed < agent.HOLD, f"the second run was held open for {elapsed:.2f}s"


async def test_agent_task_inside_foreground_does_not_deadlock() -> None:
    """A foregrounded AgentTask that needs another user turn must not stall the run:
    ``run()`` has to return before that turn can arrive."""
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="ready",
                content="",
                ttft=0.1,
                duration=0.1,
                tool_calls=[
                    FunctionToolCall(name="start_survey", arguments="{}", call_id="call_1")
                ],
            ),
            FakeLLMResponse(input="ask_name", content="what is your name?", ttft=0.1, duration=0.1),
            FakeLLMResponse(
                input="Bob",
                content="",
                ttft=0.1,
                duration=0.1,
                tool_calls=[
                    FunctionToolCall(
                        name="record_name", arguments='{"name": "Bob"}', call_id="call_2"
                    )
                ],
            ),
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(Surveyor())

        await asyncio.wait_for(sess.run(user_input="ready"), timeout=5.0)

        second_result = await asyncio.wait_for(sess.run(user_input="Bob"), timeout=5.0)
        second_result.expect.contains_function_call(name="record_name")


async def test_agent_task_under_a_foregrounded_reply_does_not_deadlock() -> None:
    """The same circular wait one level deeper: the foreground scope holds a
    ``generate_reply``, and a tool of that reply awaits the task.

    Only the deadlock is asserted. Whether the task survives to the next turn is a
    separate question, and it behaves the same way with or without the guard.
    """
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="ready",
                content="",
                ttft=0.1,
                duration=0.1,
                tool_calls=[
                    FunctionToolCall(name="ask_over_speech", arguments="{}", call_id="call_1")
                ],
            ),
            FakeLLMResponse(
                input="ask_via_tool",
                content="",
                ttft=0.1,
                duration=0.1,
                tool_calls=[
                    FunctionToolCall(name="collect_name", arguments="{}", call_id="call_2")
                ],
            ),
            FakeLLMResponse(input="ask_name", content="what is your name?", ttft=0.1, duration=0.1),
            FakeLLMResponse(input="collected", content="all done", ttft=0.1, duration=0.1),
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(Surveyor())

        await asyncio.wait_for(sess.run(user_input="ready"), timeout=5.0)
