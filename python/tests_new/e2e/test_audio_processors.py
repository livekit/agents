"""Input audio processors (`AudioInputOptions.noise_cancellation`) on a real room: what a Python
`rtc.FrameProcessor` returns is what the STT hears, a selector picks one per participant track,
the session closes what it selected and leaves the application's own processor open, and a
failing processor costs one recoverable error, not the call."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import Agent, ModelSettings, RunContext, VoiceSession, stt
from livekit.agents.voice.events import ErrorEvent
from livekit.agents.voice.room_io import (
    AudioInputOptions,
    AudioProcessingError,
    NoiseCancellationParams,
    RoomOptions,
)
from tests_new.fakes import FakeLLM, FakeSTT, until

from .sfu import connected, tone


def loudness(frame: rtc.AudioFrame) -> float:
    samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0


class Recorder(rtc.FrameProcessor[rtc.AudioFrame]):
    """A processor that keeps what it is given and returns ``process(frame)``."""

    enabled = True

    def __init__(self, process: Callable[[rtc.AudioFrame], rtc.AudioFrame] = lambda f: f) -> None:
        self.process = process
        self.received: list[rtc.AudioFrame] = []
        self.closed = False
        self.stream_info: dict[str, str] | None = None
        self.credentials: dict[str, str] | None = None

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        self.received.append(frame)
        return self.process(frame)

    def _close(self) -> None:
        self.closed = True

    def _on_stream_info_updated(self, **info: str) -> None:
        self.stream_info = info

    def _on_credentials_updated(self, **credentials: str) -> None:
        self.credentials = credentials


def silenced(frame: rtc.AudioFrame) -> rtc.AudioFrame:
    return rtc.AudioFrame.create(frame.sample_rate, frame.num_channels, frame.samples_per_channel)


@dataclass
class Call:
    """A person speaking a 1 kHz tone into a room, and the agent's room connection."""

    person: rtc.Room
    agent: rtc.Room
    microphone: rtc.LocalTrackPublication
    source: rtc.AudioSource

    async def replace_microphone(self) -> None:
        """Unpublish the microphone and publish a new track of the same source."""
        await self.person.local_participant.unpublish_track(self.microphone.sid)
        self.microphone = await self.person.local_participant.publish_track(
            rtc.LocalAudioTrack.create_audio_track("microphone", self.source),
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )


@pytest.fixture
async def call(sfu: str) -> AsyncIterator[Call]:
    name = "processors-" + uuid.uuid4().hex
    async with connected(sfu, name, "person") as person, connected(sfu, name, "agent") as agent:
        source = rtc.AudioSource(48000, 1)
        publication = await person.local_participant.publish_track(
            rtc.LocalAudioTrack.create_audio_track("microphone", source),
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )

        async def speak() -> None:
            async for frame in tone(1000):
                await source.capture_frame(frame)

        speaking = asyncio.create_task(speak())
        try:
            yield Call(person, agent, publication, source)
        finally:
            speaking.cancel()
            await asyncio.gather(speaking, return_exceptions=True)
            await source.aclose()


def listening_session(heard: list[rtc.AudioFrame], llm: FakeLLM | None = None) -> VoiceSession:
    """A session whose STT input is kept in ``heard``."""
    session = VoiceSession(
        stt=FakeSTT(),
        llm=llm or FakeLLM(),
        vad=None,
        turn_handling={"turn_detection": "stt"},
        aec_warmup_duration=None,
    )

    @session.stt_node
    async def hear(
        ctx: RunContext, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> AsyncIterator[stt.SpeechEvent]:
        async def kept() -> AsyncIterator[rtc.AudioFrame]:
            async for frame in audio:
                heard.append(frame)
                yield frame

        async for event in session.default.stt_node(ctx, kept(), model_settings):
            yield event

    return session


NoiseCancellation = (
    rtc.FrameProcessor[rtc.AudioFrame]
    | Callable[[NoiseCancellationParams], rtc.FrameProcessor[rtc.AudioFrame] | None]
)


def processing(nc: NoiseCancellation) -> RoomOptions:
    return RoomOptions(
        audio_input=AudioInputOptions(noise_cancellation=nc),
        audio_output=False,
        transcription_output=False,
    )


AGENT = Agent(instructions="Listen.")


async def test_the_stt_hears_what_the_processor_returns(call: Call) -> None:
    """The person's frames go through the processor, and its output, not the input, reaches
    the STT."""
    processor, heard = Recorder(silenced), list[rtc.AudioFrame]()
    session = listening_session(heard)

    async with session:
        await session.start(AGENT, room=call.agent, room_options=processing(processor))
        await until(lambda: len(heard) >= 20, "the STT heard nothing")

    assert all(loudness(f) > 1000 for f in processor.received[5:20])
    assert all(loudness(f) == 0 for f in heard[:20])


async def test_a_selector_is_called_per_participant_track_and_may_choose_none(call: Call) -> None:
    """A callable gets the participant and track it processes; returning None leaves the audio
    unprocessed."""
    selected: list[NoiseCancellationParams] = []
    heard: list[rtc.AudioFrame] = []
    session = listening_session(heard)

    def select(params: NoiseCancellationParams) -> None:
        selected.append(params)

    async with session:
        await session.start(AGENT, room=call.agent, room_options=processing(select))
        await until(lambda: len(heard) >= 20, "the STT heard nothing")

    assert [(p.participant.identity, p.track.sid) for p in selected] == [
        ("person", call.microphone.sid)
    ]
    assert any(loudness(f) > 1000 for f in heard)


async def test_a_selected_processor_is_closed_when_its_track_ends(call: Call) -> None:
    """The session owns a processor its selector returned: it processes the track's frames, and
    only unpublishing the track closes it.

    https://github.com/livekit/agents/pull/5433 https://github.com/livekit/agents/issues/5448
    """
    processors: list[Recorder] = []
    heard: list[rtc.AudioFrame] = []
    session = listening_session(heard)

    def select(params: NoiseCancellationParams) -> Recorder:
        processors.append(Recorder())
        return processors[-1]

    async with session:
        await session.start(AGENT, room=call.agent, room_options=processing(select))
        await until(lambda: len(heard) >= 5, "the STT heard nothing")
        assert processors[0].received and not processors[0].closed

        await call.person.local_participant.unpublish_track(call.microphone.sid)

        await until(lambda: processors[0].closed, "the selected processor was left open")


async def test_a_processor_passed_directly_outlives_its_track_and_closes_with_the_session(
    call: Call,
) -> None:
    """A processor given directly serves every track of the call: the track ending leaves it
    open, the session ending closes it."""
    processor, heard = Recorder(), list[rtc.AudioFrame]()
    session = listening_session(heard)

    async with session:
        await session.start(AGENT, room=call.agent, room_options=processing(processor))
        await until(lambda: len(heard) >= 5, "the STT heard nothing")
        await call.replace_microphone()
        served = len(processor.received)
        await until(lambda: len(processor.received) > served + 20, "the new track went unheard")
        assert not processor.closed

    assert processor.closed


async def test_a_failing_processor_reports_one_recoverable_error_and_the_stt_still_hears(
    call: Call,
) -> None:
    """A raising processor is one recoverable `AudioProcessingError`; its frames reach the STT
    unprocessed and the call goes on."""

    def fail(frame: rtc.AudioFrame) -> rtc.AudioFrame:
        raise RuntimeError("model crashed")

    processor, heard = Recorder(fail), list[rtc.AudioFrame]()
    errors: list[ErrorEvent] = []
    session = listening_session(heard)
    session.on("error", errors.append)

    async with session:
        await session.start(AGENT, room=call.agent, room_options=processing(processor))
        await until(lambda: len(heard) >= 20, "the STT heard nothing")

    assert [(type(e.error), e.source, e.error.recoverable) for e in errors] == [
        (AudioProcessingError, processor, True)
    ]
    assert str(errors[0].error.error) == "model crashed"
    assert any(loudness(f) > 1000 for f in heard)


async def test_a_processor_learns_its_room_stream_and_credentials(call: Call, sfu: str) -> None:
    """As `rtc.AudioStream` does, the session tells a processor the room, participant and track
    it processes and the room's token and URL: cloud models (Krisp, ai-coustics) authenticate
    with them."""
    processor, heard = Recorder(), list[rtc.AudioFrame]()
    session = listening_session(heard)

    async with session:
        await session.start(AGENT, room=call.agent, room_options=processing(processor))
        await until(lambda: len(heard) >= 5, "the STT heard nothing")

    assert processor.stream_info == {
        "room_name": call.agent.name,
        "participant_identity": "person",
        "publication_sid": call.microphone.sid,
    }
    assert processor.credentials is not None and processor.credentials["url"] == sfu
