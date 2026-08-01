from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class Forwarded(Agent):
    """The agent handed to. Speaks from on_enter, after the handoff."""

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
        # mirrors the reported code exactly: an empty ctx.update() first
        await ctx.update("")
        forward_agent = Forwarded()
        async with ctx.foreground():
            session = ctx.session
            session.update_agent(forward_agent)
        return None


class BackgroundWorker(Agent):
    """Answers early, then keeps working for far longer than the run should wait."""

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

    @function_tool
    async def failing_job(self, ctx: RunContext) -> None:
        """Report progress, then fail in the background."""
        await ctx.update("working on it")
        raise ValueError("background boom")


def _llm() -> FakeLLM:
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


async def test_message_after_handoff_inside_foreground_is_recorded() -> None:
    """A tool that hands off inside ``ctx.foreground()`` must still have the new
    agent's on_enter reply recorded on the same RunResult.

    RunResult stops recording once every watched handle is done. The handoff
    watches ``_on_enter_task``, which is an asyncio.Task, and _watch_handle only
    wires ``_item_added`` for SpeechHandle. So the guard keeps the run alive but
    does not capture the speech handle on_enter creates.
    """
    async with AgentSession(llm=_llm()) as sess:
        await sess.start(Router())

        result = await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)

        kinds = [type(e).__name__ for e in result.events]
        assert any(k == "FunctionCallEvent" for k in kinds), kinds
        assert any(k == "FunctionCallOutputEvent" for k in kinds), kinds
        # the reply produced by Forwarded.on_enter
        assert any(k == "ChatMessageEvent" for k in kinds), (
            f"the forwarded agent's reply never reached the run; got {kinds}"
        )


async def test_handoff_in_tool_event_sequence() -> None:
    """The same case asserted the way the docs suggest, with next_event()."""
    async with AgentSession(llm=_llm()) as sess:
        await sess.start(Router())

        result = await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)

        result.expect.next_event().is_function_call(name="forward")
        result.expect.next_event().is_function_call_output()
        result.expect.next_event().is_agent_handoff(new_agent_type=Forwarded)
        result.expect.next_event().is_message(role="assistant")


async def test_background_only_tool_does_not_gate_the_run() -> None:
    """ctx.update() is also how a tool answers early and keeps working. Those
    tools must not hold run() open until the background work finishes."""
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go",
                content="",
                ttft=0.1,
                duration=0.1,
                tool_calls=[FunctionToolCall(name="slow_job", arguments="{}", call_id="call_1")],
            )
        ]
    )
    agent = BackgroundWorker()
    async with AgentSession(llm=llm) as sess:
        await sess.start(agent)

        # the tool sleeps for an hour; the run has to come back on its own
        result = await asyncio.wait_for(sess.run(user_input="go"), timeout=60.0)

        kinds = [type(e).__name__ for e in result.events]
        assert any(k == "FunctionCallEvent" for k in kinds), kinds
        # the waiter shields exe_task, so giving up on it must not cancel the tool
        assert not agent.cancelled


async def test_tool_failing_after_going_non_blocking_does_not_break_the_run() -> None:
    """A tool that raises after ctx.update() is reported through the normal tool
    lifecycle, so the waiter must not turn it into a failed run."""
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go",
                content="",
                ttft=0.1,
                duration=0.1,
                tool_calls=[FunctionToolCall(name="failing_job", arguments="{}", call_id="call_1")],
            )
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(BackgroundWorker())

        result = await asyncio.wait_for(sess.run(user_input="go"), timeout=60.0)

        kinds = [type(e).__name__ for e in result.events]
        assert any(k == "FunctionCallEvent" for k in kinds), kinds
