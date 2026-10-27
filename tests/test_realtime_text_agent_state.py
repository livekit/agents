import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, TurnHandlingOptions, inference, llm, utils, vad

from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput
from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession, fake_capabilities
from .fake_stt import FakeSTT

pytestmark = pytest.mark.unit


class _SpeechStartVAD(vad.VAD):
    """Report speech as soon as the test supplies audio, without a wall-clock schedule."""

    def __init__(self) -> None:
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.032))

    def stream(self) -> vad.VADStream:
        return _SpeechStartVADStream(self)


class _SpeechStartVADStream(vad.VADStream):
    async def _main_task(self) -> None:
        await self._input_ch.recv()
        self._event_ch.send_nowait(
            vad.VADEvent(
                type=vad.VADEventType.START_OF_SPEECH,
                samples_index=0,
                timestamp=0.0,
                speech_duration=0.0,
                silence_duration=0.0,
                frames=[],
            )
        )
        self._event_ch.send_nowait(
            vad.VADEvent(
                type=vad.VADEventType.INFERENCE_DONE,
                samples_index=512,
                timestamp=0.032,
                speech_duration=0.032,
                silence_duration=0.0,
                speaking=True,
            )
        )
        async for _ in self._input_ch:
            pass


async def _provide_text_reply(session: FakeRealtimeSession) -> None:
    """Deliver a text-only response through the fake provider's reply future."""
    while not session._reply_futs:
        await asyncio.sleep(0)

    messages = utils.aio.Chan[llm.MessageGeneration]()
    functions = utils.aio.Chan[llm.FunctionCall]()
    text = utils.aio.Chan[str]()
    audio = utils.aio.Chan[rtc.AudioFrame]()
    modalities = asyncio.Future[list[str]]()
    modalities.set_result(["text"])
    messages.send_nowait(
        llm.MessageGeneration(
            message_id="message-id",
            text_stream=text,
            audio_stream=audio,
            modalities=modalities,
        )
    )
    messages.close()
    functions.close()
    text.send_nowait("Hello")
    text.close()
    audio.close()
    session._reply_futs[0].set_result(
        llm.GenerationCreatedEvent(
            message_stream=messages,
            function_stream=functions,
            user_initiated=True,
            response_id="response-id",
        )
    )


async def test_realtime_text_reply_returns_to_listening_without_audio_output() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    async with AgentSession(llm=model) as session:
        session.output.transcription = FakeTextOutput()
        states: list[str] = []
        session.on("agent_state_changed", lambda ev: states.append(ev.new_state))
        await session.start(Agent(instructions="test"))

        reply = session.generate_reply()
        await asyncio.wait_for(_provide_text_reply(model.active_session), timeout=5)
        await asyncio.wait_for(reply.wait_for_playout(), timeout=5)

        assert reply.exception() is None
        assert states == ["listening", "speaking", "listening"]
        assert session.agent_state == "listening"


async def test_realtime_text_reply_returns_to_listening_with_audio_disabled() -> None:
    model = FakeRealtimeModel()
    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput(can_pause=True)
        session.output.set_audio_enabled(False)
        session.output.transcription = FakeTextOutput()
        await session.start(Agent(instructions="test"))

        reply = session.generate_reply()
        await asyncio.wait_for(_provide_text_reply(model.active_session), timeout=5)
        await asyncio.wait_for(reply.wait_for_playout(), timeout=5)

        assert reply.exception() is None
        assert session.agent_state == "listening"


@pytest.mark.no_concurrent
@pytest.mark.parametrize("audio_output", [None, "disabled"])
async def test_user_transcript_is_delivered_after_realtime_text_reply(
    monkeypatch: pytest.MonkeyPatch, audio_output: str | None
) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret")
    # Keep the remote classifier pending: a normal user turn must not need its verdict.
    verdicts = utils.aio.Chan[inference.OverlappingSpeechEvent]()
    stream = MagicMock()
    stream.__aiter__.side_effect = verdicts.__aiter__
    stream.aclose = AsyncMock()
    monkeypatch.setattr(inference.AdaptiveInterruptionDetector, "stream", lambda self: stream)

    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=False, user_transcription=False, audio_output=audio_output is not None
        )
    )
    stt = FakeSTT()
    async with AgentSession(
        llm=model,
        stt=stt,
        vad=_SpeechStartVAD(),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            interruption={"mode": "adaptive", "backchannel_boundary": 0.0},
        ),
    ) as session:
        audio_input = FakeAudioInput()
        session.input.audio = audio_input
        if audio_output is not None:
            session.output.audio = FakeAudioOutput(can_pause=True)
            session.output.set_audio_enabled(False)
        session.output.transcription = FakeTextOutput()
        await session.start(Agent(instructions="test"))

        reply = session.generate_reply()
        await asyncio.wait_for(_provide_text_reply(model.active_session), timeout=5)
        await asyncio.wait_for(reply.wait_for_playout(), timeout=5)

        user_states = utils.aio.Chan[str]()
        transcripts = utils.aio.Chan[str]()
        session.on("user_state_changed", lambda ev: user_states.send_nowait(ev.new_state))
        session.on("user_input_transcribed", lambda ev: transcripts.send_nowait(ev.transcript))
        audio_input.push(0.032)
        assert await asyncio.wait_for(user_states.recv(), timeout=5) == "speaking"
        stt_stream = await asyncio.wait_for(stt.stream_ch.recv(), timeout=5)
        stt_stream.send_fake_transcript("Next user turn")

        assert await asyncio.wait_for(transcripts.recv(), timeout=5) == "Next user turn"


@pytest.mark.no_concurrent
@pytest.mark.parametrize("audio_output", [None, "disabled"])
async def test_vad_interrupts_pending_reply_after_realtime_text_reply(
    monkeypatch: pytest.MonkeyPatch, audio_output: str | None
) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret")
    verdicts = utils.aio.Chan[inference.OverlappingSpeechEvent]()
    stream = MagicMock()
    stream.__aiter__.side_effect = verdicts.__aiter__
    stream.aclose = AsyncMock()
    monkeypatch.setattr(inference.AdaptiveInterruptionDetector, "stream", lambda self: stream)

    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=False, user_transcription=False, audio_output=audio_output is not None
        )
    )
    async with AgentSession(
        llm=model,
        stt=FakeSTT(),
        vad=_SpeechStartVAD(),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            interruption={"mode": "adaptive", "min_duration": 0.0, "backchannel_boundary": 0.0},
        ),
    ) as session:
        audio_input = FakeAudioInput()
        session.input.audio = audio_input
        if audio_output is not None:
            session.output.audio = FakeAudioOutput(can_pause=True)
            session.output.set_audio_enabled(False)
        session.output.transcription = FakeTextOutput()
        await session.start(Agent(instructions="test"))

        reply = session.generate_reply()
        await asyncio.wait_for(_provide_text_reply(model.active_session), timeout=5)
        await asyncio.wait_for(reply.wait_for_playout(), timeout=5)

        pending_reply = session.generate_reply()

        async def wait_for_next_request() -> None:
            while model.active_session.generate_reply_calls < 2:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_next_request(), timeout=5)
        audio_input.push(0.032)
        await asyncio.wait_for(pending_reply.wait_for_playout(), timeout=5)

        assert pending_reply.interrupted
