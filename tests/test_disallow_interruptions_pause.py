from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.io import PlaybackFinishedEvent

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

TRANSFER_MESSAGE = "I'll connect you now."


class TransferAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="Transfer callers when requested.")
        self.transfer_started = asyncio.Event()
        self.transfer_completed = asyncio.Event()

    @function_tool
    async def transfer_to_live_agent(self, context: RunContext) -> None:
        """Transfer the caller to a live agent."""
        self.transfer_started.set()
        context.disallow_interruptions()
        context.session.input.set_audio_enabled(False)
        await context.session.say(TRANSFER_MESSAGE, allow_interruptions=False)
        self.transfer_completed.set()


async def test_uninterruptible_tool_speech_plays_after_paused_parent() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Connect me to an agent.")
    actions.add_llm(
        content="Happy to connect you.",
        tool_calls=[
            FunctionToolCall(
                name="transfer_to_live_agent",
                arguments="{}",
                call_id="transfer-1",
            )
        ],
        ttft=0.1,
        duration=1.0,
    )
    # VAD pauses the parent after its first frame is forwarded but before the tool starts.
    actions.add_tts(2.0, ttfb=0.1, duration=0.2)
    actions.add_user_speech(3.1, 5.0, "")
    actions.add_tts(1.0, input=TRANSFER_MESSAGE)

    session = create_session(actions, can_pause_audio=True)
    agent = TransferAgent()
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.output.audio.on("playback_finished", playback_finished_events.append)

    run_task = asyncio.create_task(run_session(session, agent))
    try:
        await asyncio.wait_for(agent.transfer_started.wait(), timeout=5.0)
        await asyncio.wait_for(agent.transfer_completed.wait(), timeout=5.0)
    finally:
        if not run_task.done():
            await session.aclose()
        await run_task

    assert [event.playback_position for event in playback_finished_events] == pytest.approx(
        [2.0, 1.0], abs=0.02
    )
