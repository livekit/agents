import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import cast

import pytest

from livekit import rtc
from livekit.agents.voice.room_io import Room, TrackPublication

pytestmark = pytest.mark.unit


class FailingProcessor:
    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        raise RuntimeError("processor failed")


@pytest.mark.asyncio
async def test_preconnect_processor_failure_reaches_the_input_owner() -> None:
    room = object.__new__(Room)
    room._processors = {
        "TR_input": (cast(rtc.FrameProcessor[rtc.AudioFrame], FailingProcessor()), None),
    }
    publication = TrackPublication(
        room, cast(rtc.TrackPublication, SimpleNamespace(sid="TR_input"))
    )
    frame = rtc.AudioFrame(bytes(960), 24000, 1, 480)
    with pytest.raises(RuntimeError, match="processor failed"):
        await publication._core_process_audio([frame])


@pytest.mark.asyncio
async def test_preconnect_uses_the_live_stream_agc(monkeypatch: pytest.MonkeyPatch) -> None:
    processed: list[rtc.AudioFrame] = []
    tasks: list[asyncio.Task[None]] = []
    head = rtc.AudioFrame(bytes(960), 24000, 1, 480)
    live = rtc.AudioFrame(bytes(960), 24000, 1, 480)

    class Agc:
        def __init__(self, *, auto_gain_control: bool) -> None:
            assert auto_gain_control

        def process_stream(self, frame: rtc.AudioFrame) -> None:
            processed.append(frame)

    class Stream:
        def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
            return self.frames()

        async def frames(self) -> AsyncIterator[SimpleNamespace]:
            yield SimpleNamespace(frame=live)

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(rtc, "AudioProcessingModule", Agc)
    monkeypatch.setattr(rtc.AudioStream, "from_track", lambda **kwargs: Stream())
    room = object.__new__(Room)
    room._processors = {}
    monkeypatch.setattr(room, "_spawn", lambda coro: tasks.append(asyncio.create_task(coro)))
    publication = TrackPublication(
        room, cast(rtc.TrackPublication, SimpleNamespace(sid="TR_input", track=object()))
    )
    options = SimpleNamespace(
        noise_cancellation=None,
        sample_rate=24000,
        num_channels=1,
        frame_size_ms=20,
        auto_gain_control=True,
    )
    frames = SimpleNamespace(push=lambda frame: False, close=lambda: None)
    publication._core_open_audio_stream(options, frames)
    await publication._core_process_audio([head])
    await asyncio.gather(*tasks)
    assert processed == [head, live]
