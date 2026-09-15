from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, Agent, AgentSession, stt, vad
from livekit.agents.types import APIConnectOptions

from .fake_llm import FakeLLM
from .fake_stt import FakeSTT
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


class _NonStreamingFakeSTT(FakeSTT):
    def __init__(self) -> None:
        super().__init__()
        self._capabilities = stt.STTCapabilities(streaming=False, interim_results=False)
        self.close_count = 0

    async def aclose(self) -> None:
        self.close_count += 1


def _metrics_listener_count(stt_impl: stt.STT) -> int:
    return len(stt_impl._events.get("metrics_collected", set()))


async def test_temporary_stream_adapter_is_closed_with_session() -> None:
    stt_impl = _NonStreamingFakeSTT()
    baseline = _metrics_listener_count(stt_impl)
    agent = Agent(
        instructions="test",
        llm=FakeLLM(),
        stt=stt_impl,
        vad=FakeVAD(),
    )
    session = AgentSession(
        vad=None,
        turn_handling={"turn_detection": None},
        session_close_transcript_timeout=0.0,
    )

    await session.start(agent)
    try:
        assert _metrics_listener_count(stt_impl) > baseline
    finally:
        await session.aclose()

    assert _metrics_listener_count(stt_impl) == baseline
    assert stt_impl.close_count == 0


class _RecordingStream(stt.RecognizeStream):
    def __init__(self, stt_impl: stt.STT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt_impl, conn_options=conn_options)
        self.inputs: asyncio.Queue[rtc.AudioFrame | stt.RecognizeStream._FlushSentinel] = (
            asyncio.Queue()
        )

    async def _run(self) -> None:
        async for data in self._input_ch:
            self.inputs.put_nowait(data)


class _FlushableSTT(FakeSTT):
    def __init__(self, *, manual_flush: bool) -> None:
        super().__init__()
        self.capabilities.manual_flush = manual_flush
        self.streams: asyncio.Queue[_RecordingStream] = asyncio.Queue()

    def stream(self, *, conn_options=DEFAULT_API_CONNECT_OPTIONS, **kwargs):
        stream = _RecordingStream(self, conn_options)
        self.streams.put_nowait(stream)
        return stream


class _TestAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="test")
        self.entered = asyncio.Event()

    async def on_enter(self) -> None:
        self.entered.set()


class _PassthroughAgent(_TestAgent):
    def stt_node(self, audio, model_settings):
        async def frames():
            async for frame in audio:
                assert isinstance(frame, rtc.AudioFrame)
                yield frame

        return Agent.default.stt_node(self, frames(), model_settings)


@pytest.mark.parametrize("agent_type", [_TestAgent, _PassthroughAgent])
@pytest.mark.parametrize("turn_detection", ["vad", None, "manual", "stt"])
@pytest.mark.parametrize("manual_flush", [True, False])
async def test_vad_eos_flushes_supported_stt_in_audio_order(
    turn_detection, manual_flush: bool, agent_type
) -> None:
    stt_impl = _FlushableSTT(manual_flush=manual_flush)
    session = AgentSession(
        stt=stt_impl,
        vad=FakeVAD(),
        turn_handling={"turn_detection": turn_detection},
        session_close_transcript_timeout=0.0,
    )
    await session.start(agent_type())
    try:
        stream = await asyncio.wait_for(stt_impl.streams.get(), 5)
        recognition = session._activity._audio_recognition
        pipeline = recognition._stt_pipeline
        frame = rtc.AudioFrame.create(16000, 1, 160)
        for segment in range(2):
            if segment == 1:
                next_agent = agent_type()
                session.update_agent(next_agent)
                await asyncio.wait_for(next_agent.entered.wait(), 5)
                recognition = session._activity._audio_recognition
                if agent_type is _TestAgent:
                    assert recognition._stt_pipeline is pipeline
                    assert stt_impl.streams.empty()
                else:
                    assert pipeline._flush_callback is None
                    assert recognition._stt_pipeline is not pipeline
                    pipeline = recognition._stt_pipeline
                    stream = await asyncio.wait_for(stt_impl.streams.get(), 5)
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.START_OF_SPEECH,
                    samples_index=0,
                    timestamp=0,
                    speech_duration=0,
                    silence_duration=0,
                )
            )
            pipeline.audio_ch.send_nowait(frame)
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.END_OF_SPEECH,
                    samples_index=160,
                    timestamp=0.01,
                    speech_duration=0.01,
                    silence_duration=0,
                )
            )
            pipeline.audio_ch.send_nowait(frame)
            assert await asyncio.wait_for(stream.inputs.get(), 5) is frame
            if manual_flush and turn_detection in ("vad", None):
                assert isinstance(
                    await asyncio.wait_for(stream.inputs.get(), 5),
                    stt.RecognizeStream._FlushSentinel,
                )
            assert await asyncio.wait_for(stream.inputs.get(), 5) is frame

        # A late or duplicate EOS must not finalize a second segment.
        await recognition._on_vad_event(
            vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=160,
                timestamp=0.01,
                speech_duration=0.01,
                silence_duration=0,
            )
        )
        pipeline.audio_ch.send_nowait(frame)
        assert await asyncio.wait_for(stream.inputs.get(), 5) is frame
    finally:
        await session.aclose()
    assert pipeline._flush_callback is None
