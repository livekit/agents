"""Voice event payloads remain usable across the native session boundary."""

import asyncio

import pytest

from livekit.agents import Agent, RunContext, VoiceSession
from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.events import (
    ToolCallEnded,
    ToolCallStarted,
    ToolCallUpdated,
    ToolExecutionUpdatedEvent,
)

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


def _tool_model(name: str) -> FakeLLM:
    return FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="check my booking",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name=name, arguments="{}", call_id="booking-1")],
            ),
        ]
    )


async def test_tool_progress_and_completion_reach_session_listeners() -> None:
    agent = Agent(instructions="Check the booking.")
    release = asyncio.Event()
    completed = asyncio.Event()

    @agent.tool
    async def lookup(ctx: RunContext) -> str:
        """Look up the booking."""
        await ctx.update("Checking the booking system.")
        await release.wait()
        return "Confirmed."

    updates: list[ToolExecutionUpdatedEvent] = []

    def on_update(event: ToolExecutionUpdatedEvent) -> None:
        updates.append(event)
        if isinstance(event.update, ToolCallUpdated):
            release.set()
        elif isinstance(event.update, ToolCallEnded):
            completed.set()

    async with VoiceSession(
        llm=_tool_model("lookup"), vad=None, turn_detection="manual"
    ) as session:
        session.on("tool_execution_updated", on_update)
        await session.start(agent)
        await asyncio.wait_for(session._run(user_input="check my booking"), timeout=5)
        await asyncio.wait_for(completed.wait(), timeout=5)

    started = [ev.update for ev in updates if isinstance(ev.update, ToolCallStarted)]
    progress = [ev.update for ev in updates if isinstance(ev.update, ToolCallUpdated)]
    ended = [ev.update for ev in updates if isinstance(ev.update, ToolCallEnded)]
    assert [ev.function_call.call_id for ev in started] == ["booking-1"]
    assert [(ev.call_id, ev.message) for ev in progress] == [
        ("booking-1", "Checking the booking system.")
    ]
    assert [(ev.call_id, ev.message, ev.status) for ev in ended] == [
        ("booking-1", "Confirmed.", "done")
    ]
