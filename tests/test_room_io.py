from __future__ import annotations

import asyncio
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.io import PlaybackFinishedEvent
from livekit.agents.voice.room_io._input import (
    _ParticipantAudioInputStream,
    _ParticipantInputStream,
)
from livekit.agents.voice.room_io._output import (
    _ParticipantAudioOutput,
    _ParticipantStreamTranscriptionOutput,
    _ParticipantTranscriptionOutput,
)
from livekit.agents.voice.room_io.room_io import RoomIO
from livekit.agents.voice.room_io.types import NoiseCancellationParams

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

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self.queued_duration += frame.duration

    async def wait_for_playout(self) -> None:
        await asyncio.sleep(0)
        self.played_duration += self.queued_duration
        self.queued_duration = 0.0

    def clear_queue(self) -> None:
        self.clear_count += 1
        self.queued_duration = 0.0


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
