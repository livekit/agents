from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable

import pytest

from livekit.agents import Agent, FlushSentinel, ModelSettings, RunContext, function_tool
from livekit.agents.llm import ChatChunk, ChatContext, ChoiceDelta, FunctionToolCall, Tool
from livekit.agents.voice.io import PlaybackFinishedEvent

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

TRANSFER_MESSAGE = "I'll connect you now."


class TransferAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="Transfer callers when requested.")
        self.transfer_started = asyncio.Event()
        self.transfer_completed = asyncio.Event()
        self.paused_parent_state_before_disallow: str | None = None
        self.parent_state_after_disallow: str | None = None
        self.recognition_speaking_after_disallow: bool | None = None

    @function_tool
    async def transfer_to_live_agent(self, context: RunContext) -> None:
        """Transfer the caller to a live agent."""
        self.transfer_started.set()
        activity = context.session._activity
        assert activity is not None and activity._audio_recognition is not None
        paused_speech = activity._paused_speech
        self.paused_parent_state_before_disallow = (
            paused_speech.agent_state if paused_speech is not None else None
        )
        context.disallow_interruptions()
        self.parent_state_after_disallow = context.session.agent_state
        self.recognition_speaking_after_disallow = activity._audio_recognition._agent_speaking
        context.session.input.set_audio_enabled(False)
        await context.session.say(TRANSFER_MESSAGE, allow_interruptions=False)
        self.transfer_completed.set()


class DelayedTransferAgent(TransferAgent):
    async def llm_node(
        self,
        chat_ctx: ChatContext,
        tools: list[Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[ChatChunk | str | FlushSentinel]:
        del chat_ctx, tools, model_settings
        yield "Happy to connect you."
        yield FlushSentinel()
        await asyncio.sleep(3.0)
        yield ChatChunk(
            id="transfer-response",
            delta=ChoiceDelta(
                role="assistant",
                tool_calls=[
                    FunctionToolCall(
                        name="transfer_to_live_agent",
                        arguments="{}",
                        call_id="transfer-1",
                    )
                ],
            ),
        )


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


async def test_uninterruptible_tool_restores_paused_parent_speaking_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Connect me to an agent.")
    # The first LLM segment reaches TTS immediately. Its playout is active when
    # VAD pauses the parent, before DelayedTransferAgent emits the tool call.
    actions.add_tts(5.0, input="Happy to connect you.", ttfb=0.1, duration=0.2)
    # VAD pauses the parent after playback has started but before the tool runs.
    actions.add_user_speech(4.0, 8.0, "")
    actions.add_tts(1.0, input=TRANSFER_MESSAGE)

    session = create_session(actions, can_pause_audio=True)
    agent = DelayedTransferAgent()
    audio_output = session.output.audio
    assert audio_output is not None
    original_resume = audio_output.resume
    restored_state_observed_during_resume = False

    def resume_with_state_assertion() -> None:
        nonlocal restored_state_observed_during_resume
        if agent.transfer_started.is_set() and agent.parent_state_after_disallow is None:
            activity = session._activity
            assert activity is not None and activity._audio_recognition is not None
            assert session.agent_state == "speaking"
            assert activity._audio_recognition._agent_speaking
            restored_state_observed_during_resume = True
        original_resume()

    monkeypatch.setattr(audio_output, "resume", resume_with_state_assertion)

    run_task = asyncio.create_task(run_session(session, agent))
    try:
        await asyncio.wait_for(agent.transfer_started.wait(), timeout=7.0)
        await asyncio.wait_for(agent.transfer_completed.wait(), timeout=8.0)
    finally:
        if not run_task.done():
            await session.aclose()
        await run_task

    assert agent.paused_parent_state_before_disallow == "speaking"
    assert restored_state_observed_during_resume
    assert agent.parent_state_after_disallow == "speaking"
    assert agent.recognition_speaking_after_disallow is True
