"""Tests for ``RealtimeCapabilities.supports_overlapping_speech``: the model and the caller may
speak at once, so neither the caller's speech nor the framework cuts the model's turn short.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, TurnHandlingOptions, llm, utils

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


class _TracingAudioOutput(FakeAudioOutput):
    """Counts how often the playout was cleared or paused."""

    def __init__(self, *, can_pause: bool = False) -> None:
        super().__init__(can_pause=can_pause)
        self.clears = 0
        self.pauses = 0

    def clear_buffer(self) -> None:
        self.clears += 1
        super().clear_buffer()

    def pause(self) -> None:
        self.pauses += 1
        super().pause()


async def _start_speech(
    session: AgentSession, model: FakeRealtimeModel
) -> tuple[utils.aio.Chan[rtc.AudioFrame], utils.aio.Chan[str]]:
    """Drive one assistant generation and leave it mid-playout."""
    session.generate_reply()
    for _ in range(50):
        if model.active_session._reply_futs:
            break
        await asyncio.sleep(0)
    assert model.active_session._reply_futs, "the session never asked the model for a reply"

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


async def test_speech_started_leaves_an_overlapping_model_alone() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(supports_overlapping_speech=True))
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


async def test_speech_started_clears_buffered_playout() -> None:
    # regression guard for server-side VAD models that cancel their own response: the client
    # must still drop the playout to stay in sync
    model = FakeRealtimeModel(capabilities=fake_capabilities(supports_overlapping_speech=False))
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


@pytest.mark.parametrize("overlapping", [False, True])
async def test_audio_activity_interruption_stays_off_under_server_turn_detection(
    overlapping: bool,
) -> None:
    # the model detects the user itself, so the VAD path defers to it either way
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(supports_overlapping_speech=overlapping)
    )
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


@pytest.mark.parametrize("overlapping", [False, True])
async def test_allow_interruptions_false_is_rejected_under_server_turn_detection(
    overlapping: bool,
) -> None:
    # the model decides when to yield either way, so the setting cannot be honored
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(supports_overlapping_speech=overlapping)
    )
    async with AgentSession(
        llm=model,
        turn_handling=TurnHandlingOptions(interruption={"enabled": False}),
        aec_warmup_duration=None,
    ) as session:
        with pytest.raises(ValueError, match="allow_interruptions cannot be False"):
            await session.start(Agent(instructions="be concise"))


async def test_the_model_speaks_over_a_caller_who_is_still_talking() -> None:
    # the caller's turn must not hold back a model that decides for itself when to speak, and
    # a pausable output must not park the burst either
    model = FakeRealtimeModel(capabilities=fake_capabilities(supports_overlapping_speech=True))
    audio_out = _TracingAudioOutput(can_pause=True)

    async with AgentSession(llm=model, aec_warmup_duration=None) as session:
        session.output.audio = audio_out
        await session.start(Agent(instructions="be concise"))
        assert (activity := session._activity) is not None

        # what a configured VAD reports when the caller starts talking, and keeps talking
        activity.on_start_of_speech(None, speech_start_time=time.time())
        assert not activity._user_silence_event.is_set()

        audio_ch, text_ch = await _start_speech(session, model)

        assert audio_out.pauses == 0
        assert audio_out.clears == 0
        assert session.agent_state == "speaking"

        audio_ch.close()
        text_ch.close()


async def test_a_model_that_cannot_overlap_still_waits_the_caller_out() -> None:
    # regression guard: the caller's turn keeps gating every other realtime model
    model = FakeRealtimeModel(capabilities=fake_capabilities(supports_overlapping_speech=False))

    async with AgentSession(llm=model, aec_warmup_duration=None) as session:
        session.output.audio = _TracingAudioOutput(can_pause=True)
        await session.start(Agent(instructions="be concise"))
        assert (activity := session._activity) is not None

        activity.on_start_of_speech(None, speech_start_time=time.time())
        session.generate_reply()
        for _ in range(50):
            await asyncio.sleep(0)

        # the reply is never even asked for while the caller holds the floor
        assert not model.active_session._reply_futs
        assert session.agent_state != "speaking"


async def test_client_side_turn_taking_takes_the_floor_back() -> None:
    # overlap is the model's only while it owns turn-taking; once the client drives turns the
    # framework does, and its VAD would otherwise cut a reply it never gated
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            supports_overlapping_speech=True, can_disable_turn_detection=True
        )
    )

    async with AgentSession(
        llm=model,
        vad=FakeVAD(),
        turn_handling=TurnHandlingOptions(turn_detection="vad"),
        aec_warmup_duration=None,
    ) as session:
        session.output.audio = _TracingAudioOutput(can_pause=True)
        await session.start(Agent(instructions="be concise"))
        assert (activity := session._activity) is not None
        assert activity._rt_turn_detection_enabled is False
        assert activity._rt_overlapping_speech_enabled is False

        activity.on_start_of_speech(None, speech_start_time=time.time())
        session.generate_reply()
        for _ in range(50):
            await asyncio.sleep(0)

        assert not model.active_session._reply_futs
        assert session.agent_state != "speaking"
