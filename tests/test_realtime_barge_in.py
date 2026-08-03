"""
Tests for ``RealtimeCapabilities.server_barge_in``: a full-duplex model keeps streaming through
user speech and yields on its own, so its speech-started event must not interrupt playback. Only
the framework's own VAD / interruption detection may cut the agent off.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, TurnHandlingOptions, llm, utils

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


class _TracingAudioOutput(FakeAudioOutput):
    """Records how many times the playout buffer was cleared."""

    def __init__(self) -> None:
        super().__init__()
        self.clears = 0

    def clear_buffer(self) -> None:
        self.clears += 1
        super().clear_buffer()


async def _start_speech(
    session: AgentSession, model: FakeRealtimeModel
) -> tuple[utils.aio.Chan[rtc.AudioFrame], utils.aio.Chan[str]]:
    """Drive one assistant generation and leave it mid-playout."""
    session.generate_reply()
    while not model.active_session._reply_futs:
        await asyncio.sleep(0)

    message_ch = utils.aio.Chan[llm.MessageGeneration]()
    function_ch = utils.aio.Chan[llm.FunctionCall]()
    text_ch = utils.aio.Chan[str]()
    audio_ch = utils.aio.Chan[rtc.AudioFrame]()
    modalities = asyncio.Future[list[str]]()
    modalities.set_result(["audio", "text"])

    message_ch.send_nowait(
        llm.MessageGeneration(
            message_id="msg-1",
            text_stream=text_ch,
            audio_stream=audio_ch,
            modalities=modalities,
        )
    )
    message_ch.close()
    function_ch.close()
    text_ch.send_nowait("the weather today is")
    audio_ch.send_nowait(
        rtc.AudioFrame(
            data=b"\x00\x00" * 2400, sample_rate=24000, num_channels=1, samples_per_channel=2400
        )
    )

    model.active_session._reply_futs[0].set_result(
        llm.GenerationCreatedEvent(
            message_stream=message_ch,
            function_stream=function_ch,
            user_initiated=True,
        )
    )
    # let the speech reach playout, but leave the streams open so it stays in flight
    for _ in range(50):
        await asyncio.sleep(0)
    return audio_ch, text_ch


async def test_speech_started_does_not_interrupt_with_server_barge_in() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(server_barge_in=True))
    audio_out = _TracingAudioOutput()

    async with AgentSession(llm=model, aec_warmup_duration=None) as session:
        session.output.audio = audio_out
        await session.start(Agent(instructions="be concise"))
        audio_ch, text_ch = await _start_speech(session, model)
        speech = session.current_speech
        assert speech is not None

        model.active_session.emit("input_speech_started", llm.InputSpeechStartedEvent())
        for _ in range(20):
            await asyncio.sleep(0)

        # the model is still speaking: playback is left alone
        assert speech.interrupted is False
        assert audio_out.clears == 0
        # the user is still reported as speaking
        assert session.user_state == "speaking"

        audio_ch.close()
        text_ch.close()


async def test_speech_started_interrupts_without_server_barge_in() -> None:
    # regression guard for server-side VAD models that cancel their own response: the client
    # must still drop the playout to stay in sync
    model = FakeRealtimeModel(capabilities=fake_capabilities(server_barge_in=False))
    audio_out = _TracingAudioOutput()

    async with AgentSession(llm=model, aec_warmup_duration=None) as session:
        session.output.audio = audio_out
        await session.start(Agent(instructions="be concise"))
        audio_ch, text_ch = await _start_speech(session, model)
        speech = session.current_speech
        assert speech is not None

        model.active_session.emit("input_speech_started", llm.InputSpeechStartedEvent())
        for _ in range(20):
            await asyncio.sleep(0)

        assert speech.interrupted is True
        assert audio_out.clears > 0

        audio_ch.close()
        text_ch.close()


async def test_client_side_interruption_still_available_with_server_barge_in() -> None:
    # the model cannot be told to stop, so an app that wires its own VAD must still be able to
    # cut the agent off locally
    model = FakeRealtimeModel(capabilities=fake_capabilities(server_barge_in=True))
    audio_out = _TracingAudioOutput()

    async with AgentSession(llm=model, vad=FakeVAD(), aec_warmup_duration=None) as session:
        session.output.audio = audio_out
        await session.start(Agent(instructions="be concise"))
        audio_ch, text_ch = await _start_speech(session, model)
        speech = session.current_speech
        assert speech is not None
        assert session._activity is not None

        session._activity._interrupt_by_audio_activity()
        for _ in range(20):
            await asyncio.sleep(0)

        assert speech.interrupted is True
        assert audio_out.clears > 0

        audio_ch.close()
        text_ch.close()


async def test_audio_activity_interruption_stays_off_without_server_barge_in() -> None:
    # for these models the server's speech-started event interrupts, so the VAD path must keep
    # deferring to it
    model = FakeRealtimeModel(capabilities=fake_capabilities(server_barge_in=False))
    audio_out = _TracingAudioOutput()

    async with AgentSession(llm=model, vad=FakeVAD(), aec_warmup_duration=None) as session:
        session.output.audio = audio_out
        await session.start(Agent(instructions="be concise"))
        audio_ch, text_ch = await _start_speech(session, model)
        speech = session.current_speech
        assert speech is not None
        assert session._activity is not None

        session._activity._interrupt_by_audio_activity()
        for _ in range(20):
            await asyncio.sleep(0)

        assert speech.interrupted is False
        assert audio_out.clears == 0

        audio_ch.close()
        text_ch.close()


async def test_allow_interruptions_false_is_allowed_with_server_barge_in() -> None:
    # nothing interrupts from the speech-started event, so there is no conflict to reject
    model = FakeRealtimeModel(capabilities=fake_capabilities(server_barge_in=True))
    async with AgentSession(
        llm=model,
        turn_handling=TurnHandlingOptions(interruption={"enabled": False}),
        aec_warmup_duration=None,
    ) as session:
        await session.start(Agent(instructions="be concise"))
        assert session._activity is not None


async def test_allow_interruptions_false_still_rejected_for_server_turn_detection() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(server_barge_in=False))
    async with AgentSession(
        llm=model,
        turn_handling=TurnHandlingOptions(interruption={"enabled": False}),
        aec_warmup_duration=None,
    ) as session:
        with pytest.raises(ValueError, match="allow_interruptions cannot be False"):
            await session.start(Agent(instructions="be concise"))
