import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from types import SimpleNamespace
from typing import cast

import pytest

from livekit import rtc
from livekit.agents.voice.room_io import LocalParticipant, Room, TrackPublication

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


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["track", "publish", "cancelled"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_failed_audio_publication_releases_its_unadopted_source(
    monkeypatch: pytest.MonkeyPatch, phase: str, cleanup_fails: bool
) -> None:
    closed = 0
    started = asyncio.Event()
    failure = RuntimeError("publication failed")

    class Source:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def aclose(self) -> None:
            nonlocal closed
            closed += 1
            if cleanup_fails:
                raise RuntimeError("source cleanup failed")

    def create_track(name: str, source: Source) -> object:
        if phase == "track":
            raise failure
        return object()

    async def publish_track(track: object, options: object) -> object:
        started.set()
        if phase == "publish":
            raise failure
        await asyncio.Event().wait()
        raise AssertionError("cancelled publication cannot return")

    monkeypatch.setattr(rtc, "AudioSource", Source)
    monkeypatch.setattr(rtc.LocalAudioTrack, "create_audio_track", create_track)
    room = object.__new__(Room)
    room._room = cast(
        rtc.Room, SimpleNamespace(local_participant=SimpleNamespace(publish_track=publish_track))
    )
    participant = LocalParticipant(room)
    options = SimpleNamespace(
        sample_rate=24000, num_channels=1, queue_size_ms=200, name="speech", source="microphone"
    )
    task = asyncio.create_task(participant._core_publish_audio_track(options))
    if phase == "cancelled":
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await task
        error: BaseException = cancelled.value
    else:
        with pytest.raises(RuntimeError) as failed:
            await task
        assert failed.value is failure, "cleanup must preserve the publication failure"
        error = failed.value
    assert closed == 1, "a source rejected before adoption must be closed exactly once"
    if cleanup_fails:
        assert any("source cleanup failed" in note for note in error.__notes__)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["audio", "video"])
@pytest.mark.parametrize(
    "outcome",
    [
        "cancelled_before_read",
        "cancelled",
        "failed",
        "failed_cleanup",
        "cancelled_before_read_cleanup",
    ],
)
async def test_media_pump_releases_sdk_source_and_core_sink(
    monkeypatch: pytest.MonkeyPatch, kind: str, outcome: str
) -> None:
    from livekit.agents.voice import room_io as room_adapter

    logged: list[BaseException] = []

    def report(message: str, *, exc_info: BaseException) -> None:
        logged.append(exc_info)

    monkeypatch.setattr(room_adapter.logger, "error", report)
    entered = asyncio.Event()
    closed_sinks = 0
    failure = RuntimeError("media read failed")

    class Stream:
        def __init__(self, **kwargs: object) -> None:
            self.closed = 0
            streams.append(self)

        def __aiter__(self) -> AsyncIterator[object]:
            return self.read()

        async def read(self) -> AsyncIterator[object]:
            entered.set()
            if outcome.startswith("failed"):
                raise failure
            await asyncio.Event().wait()
            yield object()

        async def aclose(self) -> None:
            self.closed += 1
            await asyncio.sleep(0)
            if outcome.endswith("cleanup"):
                raise RuntimeError("media cleanup failed")

    streams: list[Stream] = []

    def close_sink() -> None:
        nonlocal closed_sinks
        closed_sinks += 1

    monkeypatch.setattr(rtc.AudioStream, "from_track", Stream)
    monkeypatch.setattr(rtc.VideoStream, "from_track", Stream)
    room = object.__new__(Room)
    room._processors = {}
    tasks = record_tasks(monkeypatch, room)
    publication = TrackPublication(
        room, cast(rtc.TrackPublication, SimpleNamespace(sid="TR_cancelled", track=object()))
    )
    frames = SimpleNamespace(push=lambda *args: True, close=close_sink)
    if kind == "audio":
        options = SimpleNamespace(
            noise_cancellation=None,
            sample_rate=24000,
            num_channels=1,
            frame_size_ms=20,
            auto_gain_control=False,
            queue_capacity=50,
        )
        publication._core_open_audio_stream(options, frames)
    else:
        publication._core_open_video_stream(frames)
    if outcome == "cancelled":
        await asyncio.wait_for(entered.wait(), 1)
    if outcome.startswith("cancelled"):
        tasks[0].cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # A stream owns one cleanup task even when its reader never reached the first poll.
    while any(not task.done() for task in tasks):
        await asyncio.gather(*tasks, return_exceptions=True)
    if outcome.startswith("failed"):
        assert any(result is failure for result in results), "read errors must retain their cause"
    if outcome == "failed_cleanup":
        assert any("media cleanup failed" in note for note in failure.__notes__)
    if outcome == "cancelled_before_read_cleanup":
        assert len(logged) == 1, "cleanup without a waiting reader must report failure once"
        assert str(logged[0]) == "media cleanup failed"
    else:
        assert not logged, "failures observed by the reader must not be reported twice"
    assert closed_sinks == 1, "cancelling a reader must settle the core frame receiver"
    assert all(stream.closed == 1 for stream in streams), "every acquired SDK stream must close"
    assert not room._processors, "cancelled streams must release their processor registration"


@pytest.mark.asyncio
async def test_cancelling_media_cleanup_retains_its_sdk_close_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reading = asyncio.Event()
    closing = asyncio.Event()
    release = asyncio.Event()
    closed = False

    class Stream:
        def __init__(self, **kwargs: object) -> None:
            pass

        def __aiter__(self) -> AsyncIterator[object]:
            return self.read()

        async def read(self) -> AsyncIterator[object]:
            reading.set()
            await asyncio.Event().wait()
            yield object()

        async def aclose(self) -> None:
            nonlocal closed
            closing.set()
            await release.wait()
            closed = True

    monkeypatch.setattr(rtc.AudioStream, "from_track", Stream)
    room = object.__new__(Room)
    room._processors = {}
    tasks = record_tasks(monkeypatch, room)
    publication = TrackPublication(
        room, cast(rtc.TrackPublication, SimpleNamespace(sid="TR_cleanup", track=object()))
    )
    options = SimpleNamespace(
        noise_cancellation=None,
        sample_rate=24000,
        num_channels=1,
        frame_size_ms=20,
        auto_gain_control=False,
        queue_capacity=50,
    )
    publication._core_open_audio_stream(
        options, SimpleNamespace(push=lambda frame: True, close=lambda: None)
    )
    await asyncio.wait_for(reading.wait(), 1)
    reader = tasks[0]
    reader.cancel()
    await asyncio.wait_for(closing.wait(), 1)
    reader.cancel()
    await asyncio.gather(reader, return_exceptions=True)
    release.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert closed, "cancelling the reader's cleanup wait must not abandon SDK close"


@pytest.mark.parametrize("kind", ["audio", "video"])
def test_missing_track_refuses_media_before_allocating_sdk_resources(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    allocated = 0
    closed = 0

    def open_stream(**kwargs: object) -> object:
        nonlocal allocated
        allocated += 1
        raise RuntimeError("SDK received a missing track")

    def close_sink() -> None:
        nonlocal closed
        closed += 1

    monkeypatch.setattr(rtc.AudioStream, "from_track", open_stream)
    monkeypatch.setattr(rtc.VideoStream, "from_track", open_stream)
    room = object.__new__(Room)
    room._processors = {}
    tasks = record_tasks(monkeypatch, room)
    publication = TrackPublication(
        room, cast(rtc.TrackPublication, SimpleNamespace(sid="TR_gone", track=None))
    )
    frames = SimpleNamespace(push=lambda *args: True, close=close_sink)
    with pytest.raises(RuntimeError, match="no longer subscribed"):
        if kind == "audio":
            publication._core_open_audio_stream(
                SimpleNamespace(noise_cancellation=None, auto_gain_control=False), frames
            )
        else:
            publication._core_open_video_stream(frames)
    assert allocated == 0, "gone tracks must be refused before the SDK acquires a stream"
    assert not tasks
    assert closed == 1, "refusal must settle the core's frame receiver"
