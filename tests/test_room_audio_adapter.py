import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from types import SimpleNamespace
from typing import cast

import pytest

from livekit import rtc
from livekit.agents.voice.room_io import Room, TrackPublication

pytestmark = pytest.mark.unit


def record_tasks(monkeypatch: pytest.MonkeyPatch, room: Room) -> list[asyncio.Task[None]]:
    tasks: list[asyncio.Task[None]] = []

    def spawn(coro: Coroutine[object, object, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(coro)
        tasks.append(task)
        return task

    monkeypatch.setattr(room, "_spawn", spawn)
    return tasks


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
    tasks = record_tasks(monkeypatch, room)
    publication = TrackPublication(
        room, cast(rtc.TrackPublication, SimpleNamespace(sid="TR_input", track=object()))
    )
    options = SimpleNamespace(
        noise_cancellation=None,
        sample_rate=24000,
        num_channels=1,
        frame_size_ms=20,
        auto_gain_control=True,
        queue_capacity=50,
    )
    frames = SimpleNamespace(push=lambda frame: False, close=lambda: None)
    publication._core_open_audio_stream(options, frames)
    await publication._core_process_audio([head])
    await asyncio.gather(*tasks)
    assert processed == [head, live]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["byte", "text"])
@pytest.mark.parametrize("outcome", ["refused", "failed", "cancelled", "cancelled_before_read"])
async def test_inbound_stream_releases_both_sdk_reader_and_core_sink(
    monkeypatch: pytest.MonkeyPatch, kind: str, outcome: str
) -> None:
    handlers: dict[str, object] = {}
    entered = asyncio.Event()
    parked = asyncio.Event()
    reader_closed = False
    sink_closed = False

    class Reader:
        info = object()

        def __aiter__(self) -> AsyncIterator[bytes | str]:
            return self.read()

        async def read(self) -> AsyncIterator[bytes | str]:
            yield b"head" if kind == "byte" else "head"
            entered.set()
            if outcome == "failed":
                raise RuntimeError("transport failed")
            await parked.wait()

        def close(self) -> None:
            nonlocal reader_closed
            reader_closed = True

        async def aclose(self) -> None:
            self.close()

    class Sink:
        def push(self, chunk: bytes | str) -> bool:
            return True

        def close(self) -> None:
            nonlocal sink_closed
            sink_closed = True

    reader = Reader()
    room = object.__new__(Room)
    room._room = cast(
        rtc.Room,
        SimpleNamespace(
            **{f"register_{kind}_stream_handler": lambda topic, cb: handlers.update({topic: cb})}
        ),
    )
    tasks = record_tasks(monkeypatch, room)
    streams = SimpleNamespace(open=lambda info, sender: None if outcome == "refused" else Sink())
    getattr(room, f"_core_register_{kind}_stream_handler")("input", streams)
    cast(Callable[[Reader, str], None], handlers["input"])(reader, "user")
    if outcome.startswith("cancelled"):
        if outcome == "cancelled":
            await asyncio.wait_for(entered.wait(), 1)
        for task in tasks:
            task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    if outcome == "failed":
        assert any(isinstance(result, RuntimeError) for result in results)
    assert reader_closed, "refused, failed and cancelled streams must release their SDK reader"
    if outcome != "refused":
        assert sink_closed, "a failed SDK read must settle the core's chunk receiver"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,capacity", [("audio", 50), ("video", 4)])
async def test_stalled_sdk_media_retains_only_its_latest_bounded_window(
    monkeypatch: pytest.MonkeyPatch, kind: str, capacity: int
) -> None:
    from livekit.rtc._utils import RingQueue

    delivered: list[int] = []
    source_closed = False

    class Stream:
        def __init__(self, *, capacity: int = 0, **kwargs: object) -> None:
            self.queue: RingQueue[SimpleNamespace] = RingQueue(capacity)
            for index in range(1000):
                frame = (
                    rtc.AudioFrame(index.to_bytes(2, "little"), 24000, 1, 1)
                    if kind == "audio"
                    else SimpleNamespace(tag=index, width=1, height=1)
                )
                self.queue.put(SimpleNamespace(frame=frame, timestamp_us=index))

        def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
            return self.read()

        async def read(self) -> AsyncIterator[SimpleNamespace]:
            yield await self.queue.get()

        async def aclose(self) -> None:
            nonlocal source_closed
            source_closed = True

    monkeypatch.setattr(rtc.AudioStream, "from_track", Stream)
    monkeypatch.setattr(rtc.VideoStream, "from_track", Stream)
    room = object.__new__(Room)
    room._processors = {}
    tasks = record_tasks(monkeypatch, room)
    publication = TrackPublication(
        room, cast(rtc.TrackPublication, SimpleNamespace(sid="TR_input", track=object()))
    )

    def capture(frame: rtc.AudioFrame | SimpleNamespace, *metadata: int) -> bool:
        delivered.append(frame.data[0] if isinstance(frame, rtc.AudioFrame) else frame.tag)
        return False

    frames = SimpleNamespace(push=capture, close=lambda: None)
    if kind == "audio":
        options = SimpleNamespace(
            noise_cancellation=None,
            sample_rate=24000,
            num_channels=1,
            frame_size_ms=20,
            auto_gain_control=False,
            queue_capacity=capacity,
        )
        publication._core_open_audio_stream(options, frames)
    else:
        publication._core_open_video_stream(frames)
    await asyncio.gather(*tasks)
    assert delivered == [1000 - capacity], "a stalled SDK leg must evict its oldest media"
    assert source_closed
