from __future__ import annotations

import asyncio
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import utils
from livekit.agents.types import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.agents.voice.io import PlaybackFinishedEvent
from livekit.agents.voice.room_io._input import (
    _ActiveSpeakerAudioInput,
    _MixedParticipantAudioInput,
    _ParticipantAudioInputStream,
    _ParticipantInputStream,
)
from livekit.agents.voice.room_io._output import (
    _ParticipantAudioOutput,
    _ParticipantStreamTranscriptionOutput,
    _ParticipantTranscriptionOutput,
)
from livekit.agents.voice.room_io.room_io import RoomIO
from livekit.agents.voice.room_io.types import (
    AudioInputOptions,
    NoiseCancellationParams,
    RoomOptions,
)

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

# -- helpers ------------------------------------------------------------------


class _FakeRoom:
    def __init__(self) -> None:
        self._events: dict[str, list[object]] = defaultdict(list)
        self.remote_participants: dict[str, object] = {}
        self.local_participant = SimpleNamespace(identity="local")
        self.name = "test-room"
        self._token = "test-token"
        self._server_url = "wss://test.livekit.cloud"

    def on(self, event: str, callback: object) -> None:
        self._events[event].append(callback)

    def off(self, event: str, callback: object) -> None:
        callbacks = self._events[event]
        callbacks.remove(callback)
        if not callbacks:
            self._events.pop(event, None)

    def listener_count(self, event: str) -> int:
        return len(self._events.get(event, []))

    def isconnected(self) -> bool:
        return True

    def register_text_stream_handler(self, topic: str, callback: object) -> None:
        self.on(f"text:{topic}", callback)

    def unregister_text_stream_handler(self, topic: str) -> None:
        self._events.pop(f"text:{topic}", None)

    def register_byte_stream_handler(self, topic: str, callback: object) -> None:
        self.on(f"bytes:{topic}", callback)

    def unregister_byte_stream_handler(self, topic: str) -> None:
        self._events.pop(f"bytes:{topic}", None)


class _MockAudioStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def aclose(self) -> None:
        pass


class _MockFrameProcessor(rtc.FrameProcessor[rtc.AudioFrame]):
    def __init__(self) -> None:
        self._enabled = True
        self.stream_info_calls: list[dict[str, str]] = []
        self.credentials_calls: list[dict[str, str]] = []
        self.close_calls: int = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def _on_stream_info_updated(
        self, *, room_name: str, participant_identity: str, publication_sid: str
    ) -> None:
        self.stream_info_calls.append(
            {
                "room_name": room_name,
                "participant_identity": participant_identity,
                "publication_sid": publication_sid,
            }
        )

    def _on_credentials_updated(self, *, token: str, url: str) -> None:
        self.credentials_calls.append({"token": token, "url": url})

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        return frame

    def _close(self) -> None:
        self.close_calls += 1


class _NoopAudioInputStream(_ParticipantInputStream[rtc.AudioFrame]):
    def __init__(self, room: _FakeRoom) -> None:
        super().__init__(room, track_source=rtc.TrackSource.SOURCE_MICROPHONE)

    def _create_stream(
        self, track: rtc.RemoteTrack, participant: rtc.Participant
    ) -> rtc.AudioStream:
        raise AssertionError("_create_stream should not be called in teardown tests")


class _FakeWriter:
    def __init__(self) -> None:
        self.close_calls = 0
        self.chunks: list[str] = []

    async def write(self, text: str) -> None:
        self.chunks.append(text)

    async def aclose(self, attributes: dict[str, str] | None = None) -> None:
        self.close_calls += 1


def _make_track_available_args(
    identity: str = "test-user", sid: str = "TR_123"
) -> tuple[MagicMock, MagicMock, MagicMock]:
    track = MagicMock()
    publication = MagicMock()
    publication.source = rtc.TrackSource.SOURCE_MICROPHONE
    publication.sid = sid
    participant = MagicMock()
    participant.identity = identity
    return track, publication, participant


def _make_audio_input_stream(
    room: _FakeRoom,
    noise_cancellation,
) -> _ParticipantAudioInputStream:
    return _ParticipantAudioInputStream(
        room,
        sample_rate=24000,
        num_channels=1,
        noise_cancellation=noise_cancellation,
        auto_gain_control=False,
        pre_connect_audio_handler=None,
    )


# -- teardown tests -----------------------------------------------------------


@pytest.mark.asyncio
async def test_participant_input_stream_aclose_unregisters_track_unpublished() -> None:
    room = _FakeRoom()
    stream = _NoopAudioInputStream(room)

    assert room.listener_count("track_subscribed") == 1
    assert room.listener_count("track_unpublished") == 1

    await stream.aclose()

    assert room.listener_count("track_subscribed") == 0
    assert room.listener_count("track_unpublished") == 0


@pytest.mark.asyncio
async def test_transcription_output_aclose_unregisters_and_closes_resources() -> None:
    room = _FakeRoom()
    output = _ParticipantTranscriptionOutput(room=room, participant=None)
    legacy_output, stream_output = output._ParticipantTranscriptionOutput__outputs

    legacy_output._flush_task = asyncio.create_task(asyncio.sleep(0))
    writer = _FakeWriter()
    stream_output._writer = writer

    assert room.listener_count("track_published") == 2
    assert room.listener_count("local_track_published") == 2

    await output.aclose()
    await output.aclose()

    assert room.listener_count("track_published") == 0
    assert room.listener_count("local_track_published") == 0
    assert legacy_output._flush_task is not None and legacy_output._flush_task.done()
    assert writer.close_calls == 1


@pytest.mark.asyncio
async def test_transcription_output_strips_markup_but_keeps_links() -> None:
    room = _FakeRoom()
    writer = _FakeWriter()
    room.local_participant.stream_text = AsyncMock(return_value=writer)

    output = _ParticipantStreamTranscriptionOutput(room=room, participant="agent")
    await output.capture_text(
        '<expr type="expression" label="happy"/>See [the docs](https://docs.livekit.io)'
    )
    output.flush()
    assert output._flush_atask is not None
    await output._flush_atask

    published = "".join(writer.chunks)
    # markup is removed; a markdown link is prose and must reach the user intact
    assert "<expr" not in published
    assert "[the docs](https://docs.livekit.io)" in published


@pytest.mark.asyncio
async def test_roomio_aclose_unregisters_disconnect_and_closes_transcription_outputs() -> None:
    room = _FakeRoom()
    agent_session = SimpleNamespace(
        off=MagicMock(),
        input=SimpleNamespace(audio=None, video=None),
        output=SimpleNamespace(audio=None, transcription=None),
    )
    room_io = RoomIO(agent_session, room)

    room.on("participant_connected", room_io._on_participant_connected)
    room.on("connection_state_changed", room_io._on_connection_state_changed)
    room.on("participant_disconnected", room_io._on_participant_disconnected)

    order: list[str] = []

    async def _mark(name: str) -> None:
        order.append(name)

    async def _close_sync() -> None:
        await _mark("sync")

    async def _close_user() -> None:
        await _mark("user")

    async def _close_agent() -> None:
        await _mark("agent")

    room_io._tr_synchronizer = SimpleNamespace(aclose=AsyncMock(side_effect=_close_sync))
    room_io._user_tr_output = SimpleNamespace(aclose=AsyncMock(side_effect=_close_user))
    room_io._agent_tr_output = SimpleNamespace(aclose=AsyncMock(side_effect=_close_agent))

    assert room.listener_count("participant_disconnected") == 1

    await room_io.aclose()

    assert room.listener_count("participant_connected") == 0
    assert room.listener_count("connection_state_changed") == 0
    assert room.listener_count("participant_disconnected") == 0
    assert order == ["sync", "user", "agent"]
    room_io._tr_synchronizer.aclose.assert_awaited_once()
    room_io._user_tr_output.aclose.assert_awaited_once()
    room_io._agent_tr_output.aclose.assert_awaited_once()


# -- frame processor lifecycle tests ------------------------------------------


@pytest.mark.asyncio
async def test_direct_processor_lifecycle() -> None:
    """Direct FrameProcessor survives track transitions and is only closed on aclose()."""
    room = _FakeRoom()
    processor = _MockFrameProcessor()
    stream = _make_audio_input_stream(room, noise_cancellation=processor)
    stream.set_participant("test-user")

    track1, pub1, participant = _make_track_available_args(sid="TR_001")
    track2, pub2, _ = _make_track_available_args(sid="TR_002")

    with patch("livekit.rtc.AudioStream.from_track", side_effect=lambda **kw: _MockAudioStream()):
        # first track subscription
        stream._on_track_available(track1, pub1, participant)

        assert stream._processor is processor
        assert processor.close_calls == 0

        # track switch — processor must survive
        stream._on_track_available(track2, pub2, participant)

        assert stream._processor is processor
        assert processor.close_calls == 0

    # final teardown closes the processor exactly once
    await stream.aclose()
    assert processor.close_calls == 1
    assert stream._processor is None


@pytest.mark.asyncio
async def test_selector_processor_lifecycle() -> None:
    """Selector-created processors are closed on track switch; the replacement
    receives lifecycle calls and is closed on aclose()."""
    room = _FakeRoom()
    processors: list[_MockFrameProcessor] = []

    def selector(_params: NoiseCancellationParams) -> _MockFrameProcessor:
        p = _MockFrameProcessor()
        processors.append(p)
        return p

    stream = _make_audio_input_stream(room, noise_cancellation=selector)
    stream.set_participant("test-user")

    track1, pub1, participant = _make_track_available_args(sid="TR_001")
    track2, pub2, _ = _make_track_available_args(sid="TR_002")

    with patch("livekit.rtc.AudioStream.from_track", side_effect=lambda **kw: _MockAudioStream()):
        # first track
        stream._on_track_available(track1, pub1, participant)

        assert len(processors) == 1
        assert stream._processor is processors[0]

        # track switch — old processor closed, new one receives lifecycle calls
        stream._on_track_available(track2, pub2, participant)

    assert len(processors) == 2
    assert processors[0].close_calls == 1
    assert stream._processor is processors[1]

    # final teardown closes the active processor
    await stream.aclose()
    assert processors[1].close_calls == 1


@pytest.mark.asyncio
async def test_selector_processor_track_disappears() -> None:
    """When a track vanishes with no replacement, the selector-created processor is closed."""
    room = _FakeRoom()
    processor = _MockFrameProcessor()
    stream = _make_audio_input_stream(room, noise_cancellation=lambda _params: processor)
    stream.set_participant("test-user")

    track, publication, participant = _make_track_available_args()

    with patch("livekit.rtc.AudioStream.from_track", side_effect=lambda **kw: _MockAudioStream()):
        stream._on_track_available(track, publication, participant)

    assert stream._processor is processor

    # track unpublished with no replacement
    stream._on_track_unavailable(publication, participant)

    assert processor.close_calls == 1
    assert stream._processor is None

    await stream.aclose()


@pytest.mark.asyncio
async def test_selector_returns_noise_cancellation_options() -> None:
    """When a selector returns NoiseCancellationOptions instead of a FrameProcessor,
    no processor is tracked."""
    room = _FakeRoom()
    nc_options = rtc.NoiseCancellationOptions(module_id="bvc", options={})
    stream = _make_audio_input_stream(room, noise_cancellation=lambda _params: nc_options)
    stream.set_participant("test-user")

    track, publication, participant = _make_track_available_args()

    with patch("livekit.rtc.AudioStream.from_track", side_effect=lambda **kw: _MockAudioStream()):
        stream._on_track_available(track, publication, participant)

    assert stream._processor is None

    await stream.aclose()


# -- audio output tests -------------------------------------------------------


class _FakeAudioSource:
    def __init__(self, *args, **kwargs) -> None:
        self.captured: list[rtc.AudioFrame] = []
        self.queued_duration = 0.0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self.captured.append(frame)

    async def wait_for_playout(self) -> None:
        pass

    def clear_queue(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


class _QueuedAudioSource(_FakeAudioSource):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clear_count = 0
        self.played_duration = 0.0
        self.frame_queued = asyncio.Event()

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self.queued_duration += frame.duration
        self.frame_queued.set()

    async def wait_for_playout(self) -> None:
        await asyncio.sleep(0)
        self.played_duration += self.queued_duration
        self.queued_duration = 0.0

    def clear_queue(self) -> None:
        self.clear_count += 1
        self.queued_duration = 0.0


class _WaitObservedEvent(asyncio.Event):
    def __init__(self) -> None:
        super().__init__()
        self.wait_started = asyncio.Event()

    async def wait(self) -> bool:
        self.wait_started.set()
        return await super().wait()


class _BlockingAudioSource(_QueuedAudioSource):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.capture_started = asyncio.Event()
        self.capture_allowed = asyncio.Event()
        self.playout_started = asyncio.Event()
        self.playout_allowed = asyncio.Event()

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        if not self.capture_started.is_set():
            self.capture_started.set()
            await self.capture_allowed.wait()

    async def wait_for_playout(self) -> None:
        self.playout_started.set()
        await self.playout_allowed.wait()
        await super().wait_for_playout()


@pytest.mark.asyncio
async def test_audio_output_playback_started_fires_once_across_pause_resume() -> None:
    """A mid-segment pause/resume (false interruption) must not re-announce playback_started.

    The synchronizer anchors its transcript clock on the first playback_started of a
    segment and accounts for the pause gap itself, so a second one would be rejected.
    """
    frame = rtc.AudioFrame(bytes(2400 * 2), 24000, 1, 2400)  # 100ms

    with patch("livekit.rtc.AudioSource", _FakeAudioSource):
        output = _ParticipantAudioOutput(
            _FakeRoom(),
            sample_rate=24000,
            num_channels=1,
            track_publish_options=rtc.TrackPublishOptions(),
        )
    output._subscribed_fut.set_result(None)  # skip track publish/subscription
    forward_task = asyncio.create_task(output._forward_audio())

    started: list[float] = []
    output.on("playback_started", lambda ev: started.append(ev.created_at))

    output.resume()  # every generation resumes the output before forwarding audio
    for _ in range(3):
        await output.capture_frame(frame)
    await asyncio.sleep(0)
    assert len(started) == 1

    output.pause()
    await asyncio.sleep(0)
    output.resume()
    for _ in range(3):
        await output.capture_frame(frame)
    await asyncio.sleep(0)

    assert len(started) == 1

    await utils.aio.cancel_and_wait(forward_task)


@pytest.mark.asyncio
async def test_audio_output_does_not_report_discarded_audio_as_played() -> None:
    frame = rtc.AudioFrame(bytes(24000 * 2), 48000, 1, 24000)  # 500ms
    next_frame = rtc.AudioFrame(bytes(960 * 2), 48000, 1, 960)  # 20ms

    with patch("livekit.rtc.AudioSource", _QueuedAudioSource):
        output = _ParticipantAudioOutput(
            _FakeRoom(),
            sample_rate=48000,
            num_channels=1,
            track_publish_options=rtc.TrackPublishOptions(),
        )
    output._subscribed_fut.set_result(None)  # skip track publish/subscription
    forward_task = asyncio.create_task(output._forward_audio())

    finished: list[PlaybackFinishedEvent] = []
    output.on("playback_finished", finished.append)

    try:
        await output.capture_frame(frame)
        await asyncio.sleep(0)
        assert output._audio_source.queued_duration > 0

        output.pause()
        await output.capture_frame(next_frame)
        output.flush()
        await asyncio.sleep(0)
        assert output._audio_source.clear_count == 1

        output.clear_buffer()
        await output.wait_for_playout()
    finally:
        await utils.aio.cancel_and_wait(forward_task)

    assert len(finished) == 1
    assert finished[0].interrupted
    assert finished[0].playback_position == 0


@pytest.mark.asyncio
async def test_audio_output_excludes_discarded_audio_after_resume() -> None:
    frame = rtc.AudioFrame(bytes(24000 * 2), 48000, 1, 24000)  # 500ms
    resumed_frame = rtc.AudioFrame(bytes(9600 * 2), 48000, 1, 9600)  # 200ms

    with patch("livekit.rtc.AudioSource", _QueuedAudioSource):
        output = _ParticipantAudioOutput(
            _FakeRoom(),
            sample_rate=48000,
            num_channels=1,
            track_publish_options=rtc.TrackPublishOptions(),
        )
    output._subscribed_fut.set_result(None)  # skip track publish/subscription
    forward_task = asyncio.create_task(output._forward_audio())

    try:
        await output.capture_frame(frame)
        await asyncio.sleep(0)

        output.pause()
        await output.capture_frame(resumed_frame)
        output.flush()
        await asyncio.sleep(0)
        assert output._audio_source.clear_count == 1

        output.resume()
        finished = await output.wait_for_playout()
    finally:
        await utils.aio.cancel_and_wait(forward_task)

    assert not finished.interrupted
    assert finished.playback_position == pytest.approx(output._audio_source.played_duration)


@pytest.mark.asyncio
async def test_audio_output_finishes_playout_when_paused_after_forwarding_drains() -> None:
    frame = rtc.AudioFrame(bytes(960 * 2), 48000, 1, 960)  # 20ms

    with patch("livekit.rtc.AudioSource", _QueuedAudioSource):
        output = _ParticipantAudioOutput(
            _FakeRoom(),
            sample_rate=48000,
            num_channels=1,
            track_publish_options=rtc.TrackPublishOptions(),
        )
    output._subscribed_fut.set_result(None)  # skip track publish/subscription
    forward_task = asyncio.create_task(output._forward_audio())

    try:
        await output.capture_frame(frame)
        await asyncio.wait_for(output._audio_source.frame_queued.wait(), timeout=1.0)

        output.flush()
        output.pause()
        finished = await asyncio.wait_for(output.wait_for_playout(), timeout=1.0)
    finally:
        output.resume()
        if output._flush_task is not None and not output._flush_task.done():
            await output._flush_task
        await utils.aio.cancel_and_wait(forward_task)

    assert not finished.interrupted
    assert finished.playback_position == pytest.approx(frame.duration)


@pytest.mark.asyncio
async def test_audio_output_drops_a_paused_frame_from_an_interrupted_segment() -> None:
    old_frame = rtc.AudioFrame(b"\x01\x00" * 960, 48000, 1, 960)  # 20ms
    new_frame = rtc.AudioFrame(b"\x02\x00" * 1920, 48000, 1, 1920)  # 40ms

    with patch("livekit.rtc.AudioSource", _QueuedAudioSource):
        output = _ParticipantAudioOutput(
            _FakeRoom(),
            sample_rate=48000,
            num_channels=1,
            track_publish_options=rtc.TrackPublishOptions(),
        )
    playback_enabled = _WaitObservedEvent()
    playback_enabled.set()
    output._playback_enabled = playback_enabled
    output._subscribed_fut.set_result(None)  # skip track publish/subscription
    forward_task = asyncio.create_task(output._forward_audio())

    try:
        output.pause()
        await output.capture_frame(old_frame)
        await asyncio.wait_for(playback_enabled.wait_started.wait(), timeout=1.0)
        assert output._audio_buf.empty()
        assert not output._forwarding_idle.is_set()

        output.flush()
        output.clear_buffer()
        interrupted = await output.wait_for_playout()

        output.resume()
        await output.capture_frame(new_frame)
        output.flush()
        finished = await output.wait_for_playout()
    finally:
        output.resume()
        if output._flush_task is not None and not output._flush_task.done():
            await output._flush_task
        await utils.aio.cancel_and_wait(forward_task)

    assert interrupted.interrupted
    assert interrupted.playback_position == 0
    assert not finished.interrupted
    assert finished.playback_position == pytest.approx(new_frame.duration)
    assert b"".join(bytes(f.data) for f in output._audio_source.captured) == bytes(new_frame.data)


@pytest.mark.asyncio
async def test_audio_output_waits_for_active_submission_and_source_playout() -> None:
    # One progressive chunk leaves no buffered remainder after the forwarder dequeues it.
    frame = rtc.AudioFrame(bytes(960 * 2), 48000, 1, 960)  # 20ms

    with patch("livekit.rtc.AudioSource", _BlockingAudioSource):
        output = _ParticipantAudioOutput(
            _FakeRoom(),
            sample_rate=48000,
            num_channels=1,
            track_publish_options=rtc.TrackPublishOptions(),
        )
    forwarding_idle = _WaitObservedEvent()
    forwarding_idle.set()
    output._forwarding_idle = forwarding_idle
    output._subscribed_fut.set_result(None)  # skip track publish/subscription
    forward_task = asyncio.create_task(output._forward_audio())
    playout_task: asyncio.Task[PlaybackFinishedEvent] | None = None

    try:
        await output.capture_frame(frame)
        await asyncio.wait_for(output._audio_source.capture_started.wait(), timeout=1.0)

        output.flush()
        playout_task = asyncio.create_task(output.wait_for_playout())
        await asyncio.wait_for(forwarding_idle.wait_started.wait(), timeout=1.0)

        assert not playout_task.done()
        assert not output._audio_source.playout_started.is_set()

        output._audio_source.capture_allowed.set()
        await asyncio.wait_for(output._audio_source.playout_started.wait(), timeout=1.0)
        assert not playout_task.done()

        output._audio_source.playout_allowed.set()
        finished = await playout_task
    finally:
        output._audio_source.capture_allowed.set()
        output._audio_source.playout_allowed.set()
        if output._flush_task is not None and not output._flush_task.done():
            await output._flush_task
        if playout_task is not None and not playout_task.done():
            await utils.aio.cancel_and_wait(playout_task)
        await utils.aio.cancel_and_wait(forward_task)

    assert not finished.interrupted
    assert finished.playback_position == pytest.approx(frame.duration)


# -- multi participant inputs -------------------------------------------------


class _ConstantAudioStream:
    """AudioStream stand-in delivering frames of a constant amplitude, paced like a track."""

    def __init__(self, value: int, *, samples: int = 1200, count: int = 200) -> None:
        self._frame = rtc.AudioFrame(
            data=np.full(samples, value, dtype=np.int16).tobytes(),
            sample_rate=24000,
            num_channels=1,
            samples_per_channel=samples,
        )
        self._left = count

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._left <= 0:
            raise StopAsyncIteration
        self._left -= 1
        await asyncio.sleep(self._frame.duration)
        return SimpleNamespace(frame=self._frame)

    async def aclose(self) -> None:
        pass


def _make_room_audio_input(room: _FakeRoom, cls):
    return cls(
        room,
        sample_rate=24000,
        num_channels=1,
        noise_cancellation=None,
        auto_gain_control=False,
        pre_connect_audio_handler=None,
    )


def _start_track(
    stream, identity: str, value: int, sid: str | None = None, muted: bool = False
) -> None:
    """Publish a microphone track of a constant amplitude for this participant."""
    track, publication, participant = _make_track_available_args(
        identity, sid=sid or f"TR_{identity}"
    )
    publication.muted = muted
    with patch(
        "livekit.rtc.AudioStream.from_track",
        side_effect=lambda **kw: _ConstantAudioStream(value),
    ):
        stream._on_track_available(track, publication, participant)


def _publish(audio_input, identity: str, value: int) -> None:
    """Add a participant to a group input and start their microphone."""
    audio_input.add_participant(identity)
    _start_track(audio_input._streams[identity], identity, value)


def _mute_track(room: _FakeRoom, audio_input, identity: str, muted: bool) -> None:
    """Mute or unmute a participant's published track and deliver the room event."""
    audio_input._streams[identity]._publication.muted = muted
    event = "track_muted" if muted else "track_unmuted"
    publication = MagicMock()
    publication.source = rtc.TrackSource.SOURCE_MICROPHONE
    for callback in list(room._events[event]):
        callback(SimpleNamespace(identity=identity), publication)


def _report_speakers(room: _FakeRoom, *identities: str) -> None:
    """Deliver an `active_speakers_changed` update, loudest first."""
    for callback in list(room._events["active_speakers_changed"]):
        callback([SimpleNamespace(identity=identity) for identity in identities])


async def _amplitudes(audio_input, count: int) -> list[int]:
    """Amplitude of the next `count` frames the session would read."""
    frames = [await asyncio.wait_for(audio_input.__anext__(), timeout=5.0) for _ in range(count)]
    return [int(np.frombuffer(frame.data, dtype=np.int16)[0]) for frame in frames]


@pytest.mark.asyncio
async def test_linked_audio_input_only_hears_the_linked_participant() -> None:
    """The default input follows `set_participant` and ignores the rest of the room."""
    room = _FakeRoom()
    audio_input = _make_room_audio_input(room, _ParticipantAudioInputStream)

    try:
        audio_input.set_participant("alice")
        audio_input.add_participant("bob")  # RoomIO calls this for everyone, it is a no-op here
        _start_track(audio_input, "alice", 100)
        _start_track(audio_input, "bob", 200)

        assert set(await _amplitudes(audio_input, 5)) == {100}

        audio_input.remove_participant("bob")
        assert set(await _amplitudes(audio_input, 3)) == {100}
    finally:
        await audio_input.aclose()


@pytest.mark.asyncio
async def test_mixed_audio_input_sums_every_participant() -> None:
    """Participants are summed while they stream, and dropped from the mix once removed."""
    room = _FakeRoom()
    audio_input = _make_room_audio_input(room, _MixedParticipantAudioInput)

    try:
        _publish(audio_input, "alice", 100)
        _publish(audio_input, "bob", 200)
        assert audio_input._mixing == {"alice", "bob"}
        assert 300 in await _amplitudes(audio_input, 3)

        audio_input.remove_participant("bob")
        assert audio_input._mixing == {"alice"}
        assert 100 in await _amplitudes(audio_input, 5)
    finally:
        await audio_input.aclose()


@pytest.mark.asyncio
async def test_active_speaker_audio_input_forwards_one_participant_at_a_time() -> None:
    """Only the reported speaker is forwarded, preceded by their pre-roll and a turn gap."""
    room = _FakeRoom()
    audio_input = _make_room_audio_input(room, _ActiveSpeakerAudioInput)

    try:
        _publish(audio_input, "alice", 100)
        _publish(audio_input, "bob", 200)

        # a second of alice speaking, which is also a second of bob held as pre-roll
        _report_speakers(room, "alice")
        assert set(await _amplitudes(audio_input, 20)) == {100}

        _report_speakers(room, "bob")
        started = asyncio.get_running_loop().time()
        heard = await _amplitudes(audio_input, 15)
        gap = heard.index(0)  # the turn boundary, so the stt doesn't merge both speakers
        assert set(heard[gap + 1 :]) == {200}
        # bob's pre-roll arrives at once: read live, those 15 frames would take 15 * 50ms
        assert asyncio.get_running_loop().time() - started < 0.5
    finally:
        await audio_input.aclose()


# -- participant linking ------------------------------------------------------


class _RecordingAudioInput:
    """Stands in for the audio input to record the lifecycle RoomIO drives."""

    def __init__(self) -> None:
        self.added: list[str] = []
        self.removed: list[str] = []

    def add_participant(self, participant) -> None:
        self.added.append(participant.identity)

    def remove_participant(self, identity: str) -> None:
        self.removed.append(identity)


def _remote_participant(
    identity: str,
    *,
    kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
    attributes: dict[str, str] | None = None,
):
    participant = MagicMock()
    participant.identity = identity
    participant.kind = kind
    participant.attributes = attributes or {}
    return participant


def _make_room_io(room: _FakeRoom, **options) -> RoomIO:
    agent_session = SimpleNamespace(
        _on_room_io_participant_linked=MagicMock(),
        on=MagicMock(),
        off=MagicMock(),
        input=SimpleNamespace(audio=None, video=None),
        output=SimpleNamespace(audio=None, transcription=None),
    )
    return RoomIO(agent_session, room, options=RoomOptions(**options))


@pytest.mark.asyncio
async def test_roomio_links_only_the_configured_participant() -> None:
    """`participant_identity` still decides who is linked, whatever else joins."""
    room = _FakeRoom()
    room_io = _make_room_io(room, participant_identity="alice")

    room_io._on_participant_connected(_remote_participant("bob"))
    assert room_io.linked_participant is None

    alice = _remote_participant("alice")
    room_io._on_participant_connected(alice)
    assert room_io.linked_participant is alice


@pytest.mark.asyncio
async def test_roomio_skips_agent_publishers_and_unaccepted_kinds() -> None:
    room = _FakeRoom()
    room_io = _make_room_io(room)

    # an avatar worker publishing on the agent's behalf is not the user
    room_io._on_participant_connected(
        _remote_participant("avatar", attributes={ATTRIBUTE_PUBLISH_ON_BEHALF: "local"})
    )
    room_io._on_participant_connected(
        _remote_participant("egress", kind=rtc.ParticipantKind.PARTICIPANT_KIND_EGRESS)
    )
    assert room_io.linked_participant is None

    user = _remote_participant("user")
    room_io._on_participant_connected(user)
    assert room_io.linked_participant is user


@pytest.mark.asyncio
async def test_roomio_routes_every_participant_to_the_audio_input() -> None:
    """Listening and linking are separate: `participant_identity` only picks the linked one."""
    room = _FakeRoom()
    room_io = _make_room_io(room, participant_identity="alice")
    audio_input = _RecordingAudioInput()
    room_io._audio_input = audio_input  # type: ignore[assignment]

    bob = _remote_participant("bob")
    alice = _remote_participant("alice")
    room_io._on_participant_connected(bob)
    room_io._on_participant_connected(alice)
    room_io._on_participant_connected(
        _remote_participant("avatar", attributes={ATTRIBUTE_PUBLISH_ON_BEHALF: "local"})
    )

    assert audio_input.added == ["bob", "alice"]
    assert room_io.linked_participant is alice

    room_io._on_participant_disconnected(bob)
    assert audio_input.removed == ["bob"]


@pytest.mark.asyncio
async def test_mixed_audio_input_republishing_does_not_close_the_turn() -> None:
    """Swapping a track empties the mix for an instant, which is not the end of a turn."""
    room = _FakeRoom()
    audio_input = _make_room_audio_input(room, _MixedParticipantAudioInput)

    try:
        _publish(audio_input, "alice", 100)
        assert 100 in await _amplitudes(audio_input, 2)

        _start_track(audio_input._streams["alice"], "alice", 100, sid="TR_alice_2")
        assert audio_input._mixing == {"alice"}
        assert 0 not in await _amplitudes(audio_input, 5), "no silence gap on a republish"
    finally:
        await audio_input.aclose()


@pytest.mark.asyncio
async def test_active_speaker_audio_input_reselects_when_the_speaker_leaves() -> None:
    """The floor goes to whoever else the server reported, not to nobody."""
    room = _FakeRoom()
    audio_input = _make_room_audio_input(room, _ActiveSpeakerAudioInput)

    try:
        _publish(audio_input, "alice", 100)
        _publish(audio_input, "bob", 200)

        _report_speakers(room, "alice", "bob")
        assert set(await _amplitudes(audio_input, 3)) == {100}

        audio_input.remove_participant("alice")
        assert 200 in await _amplitudes(audio_input, 10)
    finally:
        await audio_input.aclose()


@pytest.mark.asyncio
async def test_active_speaker_audio_input_releases_a_muted_speaker() -> None:
    """A muted speaker the server still reports must not hold the floor."""
    room = _FakeRoom()
    audio_input = _make_room_audio_input(room, _ActiveSpeakerAudioInput)

    try:
        _publish(audio_input, "alice", 100)
        _publish(audio_input, "bob", 200)

        _report_speakers(room, "alice", "bob")
        assert audio_input._speaking == "alice"

        _mute_track(room, audio_input, "alice", True)
        await asyncio.sleep(0)  # the floor settles on the next tick
        assert audio_input._speaking == "bob"
        assert 200 in await _amplitudes(audio_input, 10)
    finally:
        await audio_input.aclose()


@pytest.mark.asyncio
async def test_roomio_skips_pre_connect_audio_for_group_inputs() -> None:
    """The buffer predates the participant, so it has nowhere to go in a combined stream."""
    room = _FakeRoom()

    outputs_off = {"audio_output": False, "text_output": False, "text_input": False}
    linked = _make_room_io(room, audio_input=AudioInputOptions(), **outputs_off)
    mixed = _make_room_io(room, audio_input=AudioInputOptions(participants="mix"), **outputs_off)
    for room_io in (linked, mixed):
        await room_io.start()

    try:
        assert linked._pre_connect_audio_handler is not None
        assert mixed._pre_connect_audio_handler is None
    finally:
        for room_io in (linked, mixed):
            await room_io.aclose()


@pytest.mark.asyncio
async def test_active_speaker_audio_input_republishing_keeps_the_floor() -> None:
    """Swapping a track is not the speaker stopping, they keep talking through it."""
    room = _FakeRoom()
    audio_input = _make_room_audio_input(room, _ActiveSpeakerAudioInput)

    try:
        _publish(audio_input, "alice", 100)
        _publish(audio_input, "bob", 200)

        _report_speakers(room, "alice", "bob")
        assert set(await _amplitudes(audio_input, 3)) == {100}

        _start_track(audio_input._streams["alice"], "alice", 100, sid="TR_alice_2")
        await asyncio.sleep(0)
        assert audio_input._speaking == "alice"
        assert set(await _amplitudes(audio_input, 5)) == {100}, "no gap, no handover to bob"
    finally:
        await audio_input.aclose()


@pytest.mark.asyncio
async def test_mixed_audio_input_does_not_buffer_an_unmixed_participant() -> None:
    """The mixer is a child's only reader, so a child it dropped must stop producing."""
    room = _FakeRoom()
    audio_input = _make_room_audio_input(room, _MixedParticipantAudioInput)

    try:
        _publish(audio_input, "alice", 100)
        _publish(audio_input, "bob", 200)
        assert 300 in await _amplitudes(audio_input, 2)

        # a muted track that keeps delivering is exactly the case that used to pile up
        _mute_track(room, audio_input, "alice", True)
        assert audio_input._mixing == {"bob"}

        await asyncio.sleep(1.0)  # 20 frames' worth of audio nobody is reading
        assert audio_input._streams["alice"]._data_ch.qsize() == 0
    finally:
        await audio_input.aclose()


@pytest.mark.asyncio
async def test_active_speaker_audio_input_selects_a_participant_reported_before_joining() -> None:
    """Someone already talking when the agent arrives must not wait for the next update."""
    room = _FakeRoom()
    audio_input = _make_room_audio_input(room, _ActiveSpeakerAudioInput)

    try:
        _report_speakers(room, "alice")  # dispatched before the stream exists
        _publish(audio_input, "alice", 100)

        await asyncio.sleep(0)
        assert audio_input._speaking == "alice"
        assert 100 in await _amplitudes(audio_input, 3)
    finally:
        await audio_input.aclose()


@pytest.mark.asyncio
async def test_mixed_audio_input_does_not_buffer_a_participant_muted_on_arrival() -> None:
    """A track subscribed while already muted never joins the mix, so it must not produce."""
    room = _FakeRoom()
    audio_input = _make_room_audio_input(room, _MixedParticipantAudioInput)

    try:
        audio_input.add_participant("alice")
        _start_track(audio_input._streams["alice"], "alice", 100, muted=True)
        assert audio_input._mixing == set()

        await asyncio.sleep(1.0)  # 20 frames' worth of audio nobody is reading
        assert audio_input._streams["alice"]._data_ch.qsize() == 0

        _mute_track(room, audio_input, "alice", False)
        assert audio_input._mixing == {"alice"}
        assert 100 in await _amplitudes(audio_input, 3)
    finally:
        await audio_input.aclose()
