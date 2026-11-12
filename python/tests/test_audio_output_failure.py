from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents import Agent, VoiceSession
from livekit.agents.voice.events import ErrorEvent
from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities, AudioOutputError

from .fake_io import FakeAudioOutput
from .fake_tts import FakeTTS

pytestmark = pytest.mark.unit


class RaisingSink(FakeAudioOutput):
    """A sink whose `raise_at`th frame (from zero) raises after its `super()` opened the segment."""

    def __init__(self, raise_at: int) -> None:
        super().__init__(can_pause=True)
        self.raise_at = raise_at
        self.frames = 0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        nth, self.frames = self.frames, self.frames + 1
        if nth == self.raise_at:
            await AudioOutput.capture_frame(self, frame)
            raise RuntimeError("the sink failed")
        await super().capture_frame(frame)


class Stage(AudioOutput):
    """A user stage forwarding everything to the sink below it."""

    def __init__(self, sink: AudioOutput) -> None:
        super().__init__(
            label="stage", capabilities=AudioOutputCapabilities(pause=True), next_in_chain=sink
        )
        self.sink = sink

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        await self.sink.capture_frame(frame)

    def flush(self) -> None:
        super().flush()
        self.sink.flush()

    def clear_buffer(self) -> None:
        self.sink.clear_buffer()


async def _two_speeches(output: AudioOutput) -> tuple[list, list[ErrorEvent], list[str | None]]:
    session = VoiceSession(
        vad=None, tts=FakeTTS(fake_audio_duration=0.5), turn_handling={"turn_detection": None}
    )
    session.output.audio = output
    errors: list[ErrorEvent] = []
    said: list[str | None] = []
    session.on("error", errors.append)
    session.on(
        "conversation_item_added",
        lambda ev: ev.item.type == "message" and said.append(ev.item.text_content),
    )
    handles = []
    async with session:
        await session.start(Agent(instructions="desk", name="desk"))
        for text in ("The first sentence here.", "The second one."):
            handle = session.say(text)
            async with asyncio.timeout(10):
                await handle
            handles.append(handle)
    return handles, errors, said


async def test_a_sink_below_a_stage_that_raises_on_its_first_frame_plays_the_next_speech() -> None:
    sink = RaisingSink(raise_at=0)
    stage = Stage(sink)
    (first, second), errors, said = await _two_speeches(stage)
    assert first.interrupted
    assert isinstance(first.exception(), RuntimeError)
    assert second.exception() is None and not second.interrupted
    assert said[-1] == "The second one."
    assert sink.frames > 1, "the next speech reached the sink"
    [error] = errors
    assert isinstance(error.error, AudioOutputError) and error.error.recoverable
    assert error.source is stage


async def test_an_output_that_raises_mid_speech_ends_the_speech_with_its_error() -> None:
    sink = RaisingSink(raise_at=3)
    (first, second), errors, said = await _two_speeches(sink)
    assert first.interrupted
    exception = first.exception()
    assert isinstance(exception, RuntimeError) and str(exception) == "the sink failed"
    [error] = errors
    assert isinstance(error.error, AudioOutputError)
    assert error.error.error is exception and error.error.label == "FakeIO"
    assert second.exception() is None and not second.interrupted
    assert said[-1] == "The second one."
