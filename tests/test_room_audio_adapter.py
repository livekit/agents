import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest

from livekit import rtc
from livekit.agents._generated import rtc_wire
from livekit.agents.voice.room_io import (
    AudioProcessingError,
    LocalParticipant,
    Room,
    TrackPublication,
    _AudioStream,
)

pytestmark = pytest.mark.unit


class FailingProcessor(rtc.FrameProcessor[rtc.AudioFrame]):
    @property
    def enabled(self) -> bool:
        return True

    @enabled.setter
    def enabled(self, value: bool) -> None:
        pass

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        raise RuntimeError("processor failed")

    def _close(self) -> None:
        pass


def _frames(items: list[Any]) -> list[rtc.AudioFrame]:
    return [item.frame for item in items if isinstance(item, rtc_wire.AudioStreamItemFrame)]


def _failed(items: list[Any]) -> list[Any]:
    return [i for i in items if isinstance(i, rtc_wire.AudioStreamItemProcessorFailed)]


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("stage", ["processor", "agc"])
async def test_processing_failure_is_reported_ahead_of_its_frame_which_keeps_original_audio(
    replay: bool, stage: str
) -> None:
    original = bytes([1, 0]) * 480
    frame = rtc.AudioFrame(bytearray(original), 24000, 1, 480)
    next_frame = rtc.AudioFrame(bytes([2, 0]) * 480, 24000, 1, 480)

    class FailsOnce(FailingProcessor):
        def __init__(self) -> None:
            self.calls = 0

        def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
            self.calls += 1
            if self.calls == 1:
                frame.data[0] = 300
                raise RuntimeError("processor failed after mutation")
            return frame

    class Agc:
        def __init__(self) -> None:
            self.calls = 0

        def process_stream(self, frame: rtc.AudioFrame) -> None:
            self.calls += 1
            if self.calls == 1:
                frame.data[0] = 300
                raise RuntimeError("AGC failed after mutation")

    class Stream:
        def __init__(self) -> None:
            self.frames = iter([frame, next_frame])

        async def __anext__(self) -> rtc.AudioFrameEvent:
            return rtc.AudioFrameEvent(next(self.frames))

    processor = FailsOnce() if stage == "processor" else None
    agc = Agc() if stage == "agc" else None
    stream = cast(rtc.AudioStream, Stream())
    source = _AudioStream(lambda: stream, processor, cast(rtc.AudioProcessingModule, agc))
    if replay:
        items = await source._core_process_audio([frame, next_frame])
    else:
        items = [await anext(source) for _ in range(3)]
    failed, recovered, following = items
    assert isinstance(failed, rtc_wire.AudioStreamItemProcessorFailed)
    assert bytes(recovered.frame.data) == original
    assert bytes(following.frame.data) == bytes([2, 0]) * 480
    assert failed.failure.processor is (processor if stage == "processor" else agc)
    assert isinstance(failed.failure.error, RuntimeError)


@pytest.mark.asyncio
async def test_each_processor_reports_its_own_failure_and_the_other_still_runs() -> None:
    frame = rtc.AudioFrame(bytes(960), 24000, 1, 480)
    processed = []

    class Agc:
        def process_stream(self, frame: rtc.AudioFrame) -> None:
            processed.append(frame)

    class Stream:
        async def __anext__(self) -> rtc.AudioFrameEvent:
            return rtc.AudioFrameEvent(frame)

    nc = FailingProcessor()
    stream = cast(rtc.AudioStream, Stream())
    source = _AudioStream(lambda: stream, nc, cast(rtc.AudioProcessingModule, Agc()))
    items = [await anext(source) for _ in range(2)]
    [failed] = _failed(items)
    assert failed.failure.processor is nc
    assert len(_frames(items)) == 1 and len(processed) == 1, "AGC skipped after NC failed"


def test_a_processor_failure_is_an_error_event_whose_source_is_the_processor(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from livekit.agents import VoiceSession
    from livekit.agents._generated.session_wire import SessionErrorWire
    from livekit.agents.voice.events import ErrorEvent

    processor = FailingProcessor()
    sessions: list[VoiceSession[Any]] = [VoiceSession(), VoiceSession()]
    errors: list[list[ErrorEvent]] = [[], []]
    for session, received in zip(sessions, errors, strict=True):
        session.on("error", received.append)
    cause = RuntimeError("processor failed")
    failure = rtc_wire.ProcessorFailure(
        label="plugin.Denoiser", processor=processor, error=cause, message=str(cause)
    )
    sessions[0]._core_event(
        SessionErrorWire(
            source="audio_processing_error",
            recoverable=True,
            label="engine",
            error=RuntimeError(failure.message),
            speech_id=None,
            processor_failure=failure,
        )
    )
    [event] = errors[0]
    [logged] = [r for r in caplog.records if r.getMessage() == "session error"]
    assert logged.levelno == logging.WARNING, "a recoverable failure is logged as an error"
    assert logged.exc_info is not None and logged.exc_info[1] is cause, "traceback dropped"
    assert isinstance(event.error, AudioProcessingError)
    assert (event.error.label, event.error.error, event.source) == (
        "plugin.Denoiser",
        cause,
        processor,
    )
    assert errors[1] == []


@pytest.mark.asyncio
async def test_preconnect_uses_its_captured_live_stream_agc_after_another_stream_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed: list[tuple[int, rtc.AudioFrame]] = []
    head = rtc.AudioFrame(bytes(960), 24000, 1, 480)
    live = rtc.AudioFrame(bytes(960), 24000, 1, 480)
    apms = []

    class Agc:
        def __init__(self, *, auto_gain_control: bool) -> None:
            assert auto_gain_control
            self.identity = len(apms)
            apms.append(self)

        def process_stream(self, frame: rtc.AudioFrame) -> None:
            processed.append((self.identity, frame))

    class Stream:
        def __aiter__(self) -> AsyncIterator[rtc.AudioFrameEvent]:
            return self

        async def __anext__(self) -> rtc.AudioFrameEvent:
            return rtc.AudioFrameEvent(live)

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(rtc, "AudioProcessingModule", Agc)
    monkeypatch.setattr(rtc.AudioStream, "from_track", lambda **kwargs: Stream())
    publication = TrackPublication(
        object.__new__(Room),
        cast(rtc.TrackPublication, SimpleNamespace(sid="TR_input", track=object())),
    )
    options = SimpleNamespace(
        noise_cancellation=None,
        sample_rate=24000,
        num_channels=1,
        frame_size_ms=20,
        auto_gain_control=True,
        queue_size_ms=1000,
    )
    original = publication._core_open_audio_stream(options)
    replacement = publication._core_open_audio_stream(options)
    try:
        assert _frames(await original._core_process_audio([head])) == [head]
        assert (await anext(original._core_frames())).frame is live
        assert (await anext(replacement._core_frames())).frame is live
        assert processed == [(0, head), (0, live), (1, live)]
    finally:
        await original.aclose()
        await replacement.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("owned", [False, True])
async def test_custom_processor_runs_once_and_only_its_owner_closes_it(
    monkeypatch: pytest.MonkeyPatch, owned: bool
) -> None:
    processed = 0
    closed = 0
    frame = rtc.AudioFrame(bytes(960), 24000, 1, 480)

    class Processor(FailingProcessor):
        def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
            nonlocal processed
            processed += 1
            return frame

        def _close(self) -> None:
            nonlocal closed
            closed += 1

    class Stream:
        async def __anext__(self) -> rtc.AudioFrameEvent:
            return rtc.AudioFrameEvent(frame)

        async def aclose(self) -> None:
            pass

    def from_track(**kwargs: object) -> Stream:
        assert kwargs["noise_cancellation"] is None
        return Stream()

    monkeypatch.setattr(rtc.AudioStream, "from_track", from_track)
    processor = Processor()
    room = object.__new__(Room)
    room.rtc = cast(rtc.Room, SimpleNamespace(remote_participants={"user": object()}))
    publication = TrackPublication(
        room, cast(rtc.TrackPublication, SimpleNamespace(sid="TR_input", track=object()))
    )
    source = publication._core_open_audio_stream(
        SimpleNamespace(
            participant_identity="user",
            noise_cancellation=(lambda params: processor) if owned else processor,
            sample_rate=24000,
            num_channels=1,
            frame_size_ms=20,
            auto_gain_control=False,
            queue_size_ms=1000,
        )
    )
    assert _frames(await source._core_process_audio([frame])) == [frame]
    assert (await anext(source)).frame is frame
    assert processed == 2
    await source.aclose()
    await source.aclose()
    assert closed == int(owned)


@pytest.mark.parametrize("owned,cleanup_fails", [(True, False), (True, True), (False, True)])
def test_failed_input_setup_releases_only_its_owned_processor_and_keeps_the_cause(
    monkeypatch: pytest.MonkeyPatch, owned: bool, cleanup_fails: bool
) -> None:
    closed = 0
    failure = RuntimeError("apm setup failed")
    original_cause = ValueError("SDK cause")
    failure.__cause__ = original_cause

    class Processor(FailingProcessor):
        def _close(self) -> None:
            nonlocal closed
            closed += 1
            if cleanup_fails:
                raise RuntimeError("processor cleanup failed")

    def make_apm(**kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(rtc, "AudioProcessingModule", make_apm)
    processor = Processor()
    room = object.__new__(Room)
    room.rtc = cast(rtc.Room, SimpleNamespace(remote_participants={"user": object()}))
    publication = TrackPublication(
        room, cast(rtc.TrackPublication, SimpleNamespace(sid="TR_input", track=object()))
    )
    with pytest.raises(RuntimeError) as failed:
        publication._core_open_audio_stream(
            SimpleNamespace(
                participant_identity="user",
                noise_cancellation=(lambda params: processor) if owned else processor,
                sample_rate=24000,
                num_channels=1,
                frame_size_ms=20,
                auto_gain_control=True,
                queue_size_ms=1000,
            )
        )
    assert failed.value is failure
    assert failed.value.__cause__ is original_cause
    assert closed == int(owned)
    if owned and cleanup_fails:
        assert any("processor cleanup failed" in note for note in failed.value.__notes__)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["audio", "video"])
async def test_sdk_stream_opens_on_the_loop_not_on_the_cores_thread(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Guards a real room's input going dead (no user turns) and the agent's speech with it: the
    core opens the stream from its own thread, and an SDK stream binds the loop it is created on.
    """
    opened_on: list[bool] = []
    frame = SimpleNamespace(width=2, height=2, sample_rate=24000)

    class Stream:
        def __init__(self, **kwargs: object) -> None:
            opened_on.append(asyncio.get_running_loop() is loop)

        async def __anext__(self) -> SimpleNamespace:
            return SimpleNamespace(frame=frame, timestamp_us=0)

        async def aclose(self) -> None:
            pass

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(rtc.AudioStream, "from_track", Stream)
    monkeypatch.setattr(rtc.VideoStream, "from_track", Stream)
    publication = TrackPublication(
        object.__new__(Room),
        cast(rtc.TrackPublication, SimpleNamespace(sid="TR_input", track=object())),
    )
    options = SimpleNamespace(
        noise_cancellation=None,
        sample_rate=24000,
        num_channels=1,
        frame_size_ms=20,
        auto_gain_control=False,
        queue_size_ms=1000,
    )

    def open_off_loop() -> Any:
        if kind == "audio":
            return publication._core_open_audio_stream(cast(Any, options))._core_frames()
        return publication._core_open_video_stream()

    source = await asyncio.to_thread(open_off_loop)
    assert opened_on == []
    await anext(source)
    assert opened_on == [True]
    await source.aclose()


@pytest.mark.parametrize("kind", ["byte", "text"])
@pytest.mark.parametrize("accepted", [False, True])
def test_sdk_reader_transfers_before_any_pull_and_refusal_closes_it(
    kind: str,
    accepted: bool,
) -> None:
    handlers: dict[str, Callable[..., None]] = {}
    received = []

    class Reader:
        info = object()
        closed = False

        def __aiter__(self) -> AsyncIterator[bytes | str]:
            raise AssertionError("header admission must not pull the SDK reader")

        def close(self) -> None:
            self.closed = True

    reader = Reader()
    room = object.__new__(Room)
    room.rtc = cast(
        rtc.Room,
        SimpleNamespace(
            **{
                f"register_{kind}_stream_handler": lambda topic, cb: handlers.update({topic: cb}),
            }
        ),
    )

    def opened(info: object, sender: str, source: Reader) -> bool:
        received.append((info, sender, source))
        return accepted

    getattr(room, f"_core_register_{kind}_stream_handler")("input", SimpleNamespace(open=opened))
    handlers["input"](reader, "user")
    assert received == [(reader.info, "user", reader)]
    assert reader.closed is not accepted


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,capacity", [("audio", 50), ("video", 4)])
async def test_stalled_sdk_media_retains_only_its_latest_bounded_window(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    capacity: int,
) -> None:
    from livekit.agents.voice import room_io as adapter
    from livekit.rtc._utils import RingQueue

    closed = []

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
            return self

        async def __anext__(self) -> SimpleNamespace:
            return await self.queue.get()

        async def aclose(self) -> None:
            closed.append(self)

    monkeypatch.setattr(rtc.AudioStream, "from_track", Stream)
    monkeypatch.setattr(rtc.VideoStream, "from_track", Stream)
    monkeypatch.setattr(adapter.rtc_wire, "VideoFrameRef", lambda *, frame, **_: frame)
    publication = TrackPublication(
        object.__new__(Room),
        cast(rtc.TrackPublication, SimpleNamespace(sid="TR_input", track=object())),
    )
    if kind == "audio":
        source = publication._core_open_audio_stream(
            SimpleNamespace(
                noise_cancellation=None,
                sample_rate=24000,
                num_channels=1,
                frame_size_ms=20,
                auto_gain_control=False,
                queue_size_ms=capacity * 20,
            )
        )
        frame = (await anext(source._core_frames())).frame
        assert int.from_bytes(frame.data, "little") == 1000 - capacity
    else:
        source = publication._core_open_video_stream()
        frame = await anext(source)
        assert frame.tag == 1000 - capacity
    await source.aclose()
    assert len(closed) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["track", "publish", "cancelled"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_failed_audio_publication_releases_its_unadopted_source(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    cleanup_fails: bool,
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
    room.rtc = cast(
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
        assert failed.value is failure
        error = failed.value
    assert closed == 1
    if cleanup_fails:
        assert any("source cleanup failed" in note for note in error.__notes__)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["audio", "video"])
async def test_sdk_source_read_and_cleanup_failures_keep_their_causes(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    failure = RuntimeError("media read failed")
    cleanup = RuntimeError("media cleanup failed")

    class Stream:
        def __init__(self, **kwargs: object) -> None:
            pass

        def __aiter__(self) -> AsyncIterator[object]:
            return self

        async def __anext__(self) -> object:
            raise failure

        async def aclose(self) -> None:
            raise cleanup

    monkeypatch.setattr(rtc.AudioStream, "from_track", Stream)
    monkeypatch.setattr(rtc.VideoStream, "from_track", Stream)
    publication = TrackPublication(
        object.__new__(Room),
        cast(rtc.TrackPublication, SimpleNamespace(sid="TR_cancelled", track=object())),
    )
    source = (
        publication._core_open_audio_stream(
            SimpleNamespace(
                noise_cancellation=None,
                sample_rate=24000,
                num_channels=1,
                frame_size_ms=20,
                auto_gain_control=False,
                queue_size_ms=1000,
            )
        )
        if kind == "audio"
        else publication._core_open_video_stream()
    )
    with pytest.raises(RuntimeError) as failed:
        await anext(source)
    assert failed.value is failure
    with pytest.raises(RuntimeError) as failed_cleanup:
        await source.aclose()
    assert failed_cleanup.value is cleanup


@pytest.mark.parametrize("kind", ["audio", "video"])
def test_missing_track_refuses_media_before_allocating_sdk_resources(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    def open_stream(**kwargs: object) -> object:
        raise AssertionError("gone tracks must be refused before SDK allocation")

    monkeypatch.setattr(rtc.AudioStream, "from_track", open_stream)
    monkeypatch.setattr(rtc.VideoStream, "from_track", open_stream)
    publication = TrackPublication(
        object.__new__(Room), cast(rtc.TrackPublication, SimpleNamespace(sid="TR_gone", track=None))
    )
    with pytest.raises(RuntimeError, match="no longer subscribed"):
        if kind == "audio":
            publication._core_open_audio_stream(
                SimpleNamespace(noise_cancellation=None, auto_gain_control=False)
            )
        else:
            publication._core_open_video_stream()
