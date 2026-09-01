from __future__ import annotations

import asyncio
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents import NOT_GIVEN, utils
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
from livekit.agents.voice.room_io.types import (
    AudioInputOptions,
    NoiseCancellationParams,
    RoomOptions,
)
from livekit.rtc._proto.track_pb2 import AudioTrackFeature

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
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.ended = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.started.set()
        await self.ended.wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.end()

    def end(self) -> None:
        if self.ended.is_set():
            return
        self.ended.set()


class _NonClosingMockAudioStream(_MockAudioStream):
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
    track.sid = sid
    publication = MagicMock()
    publication.source = rtc.TrackSource.SOURCE_MICROPHONE
    publication.sid = sid
    publication.track = track
    publication.subscribed = True
    publication.audio_features = []
    participant = MagicMock()
    participant.identity = identity
    participant.track_publications = {sid: publication}
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
async def test_participant_input_stream_aclose_unregisters_track_events() -> None:
    room = _FakeRoom()
    stream = _NoopAudioInputStream(room)

    assert room.listener_count("track_subscribed") == 1
    assert room.listener_count("track_unsubscribed") == 1
    assert room.listener_count("track_unpublished") == 1

    await stream.aclose()

    assert room.listener_count("track_subscribed") == 0
    assert room.listener_count("track_unsubscribed") == 0
    assert room.listener_count("track_unpublished") == 0


@pytest.mark.asyncio
async def test_audio_input_aclose_cancels_superseded_forward_task() -> None:
    room = _FakeRoom()
    audio_input = _make_audio_input_stream(room, noise_cancellation=None)
    audio_input.set_participant("test-user")
    old_track, publication, participant = _make_track_available_args()
    new_track = MagicMock()
    old_stream = _NonClosingMockAudioStream()
    new_stream = _MockAudioStream()

    with patch(
        "livekit.rtc.AudioStream.from_track",
        side_effect=[old_stream, new_stream],
    ):
        assert audio_input._on_track_available(old_track, publication, participant)
        await asyncio.wait_for(old_stream.started.wait(), timeout=1)
        old_forward_task = audio_input._forward_atask
        assert old_forward_task is not None

        publication.track = new_track
        assert audio_input._on_track_available(new_track, publication, participant)
        await audio_input.aclose()

    old_forward_task_done = old_forward_task.done()
    if not old_forward_task_done:
        await utils.aio.cancel_and_wait(old_forward_task)
    assert old_forward_task_done


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


@pytest.mark.parametrize(
    ("noise_cancellation", "auto_gain_control", "expected_auto_gain_control"),
    [
        (None, NOT_GIVEN, True),
        (rtc.NoiseCancellationOptions(module_id="bvc", options={}), NOT_GIVEN, False),
        (lambda _params: None, NOT_GIVEN, True),
        (lambda _params: None, False, False),
        (rtc.NoiseCancellationOptions(module_id="bvc", options={}), True, True),
        (None, False, False),
    ],
)
@pytest.mark.asyncio
async def test_roomio_resolves_auto_gain_control(
    noise_cancellation,
    auto_gain_control,
    expected_auto_gain_control: bool,
) -> None:
    room = _FakeRoom()
    agent_session = SimpleNamespace(
        on=MagicMock(),
        off=MagicMock(),
        input=SimpleNamespace(audio=None, video=None),
        output=SimpleNamespace(audio=None, transcription=None),
    )
    room_io = RoomIO(
        agent_session,
        room,
        options=RoomOptions(
            audio_input=AudioInputOptions(
                noise_cancellation=noise_cancellation,
                auto_gain_control=auto_gain_control,
                pre_connect_audio=False,
            ),
            video_input=False,
            audio_output=False,
            text_output=False,
        ),
    )
    audio_input = SimpleNamespace(aclose=AsyncMock())

    with patch(
        "livekit.agents.voice.room_io.room_io._ParticipantAudioInputStream",
        return_value=audio_input,
    ) as create_audio_input:
        await room_io.start()

    assert create_audio_input.call_args.kwargs["auto_gain_control"] is expected_auto_gain_control
    await room_io.aclose()


# -- frame processor lifecycle tests ------------------------------------------


@pytest.mark.asyncio
async def test_audio_input_replaces_concrete_track_for_same_publication() -> None:
    room = _FakeRoom()
    audio_input = _make_audio_input_stream(room, noise_cancellation=None)
    audio_input.set_participant("test-user")
    old_track, publication, participant = _make_track_available_args()
    new_track = MagicMock()
    old_stream = _MockAudioStream()
    new_stream = _MockAudioStream()

    with patch(
        "livekit.rtc.AudioStream.from_track", side_effect=[old_stream, new_stream]
    ) as create_stream:
        assert audio_input._on_track_available(old_track, publication, participant)
        await asyncio.wait_for(old_stream.started.wait(), timeout=1)

        publication.track = None
        publication.subscribed = False
        audio_input._on_track_unsubscribed(old_track, publication, participant)
        await asyncio.wait_for(old_stream.ended.wait(), timeout=1)

        publication.track = new_track
        publication.subscribed = True
        assert audio_input._on_track_available(new_track, publication, participant)
        await asyncio.wait_for(new_stream.started.wait(), timeout=1)

    assert create_stream.call_count == 2
    assert audio_input._stream is new_stream
    assert audio_input._track is new_track

    await audio_input.aclose()


@pytest.mark.asyncio
async def test_audio_input_ignores_duplicate_event_for_active_track() -> None:
    room = _FakeRoom()
    audio_input = _make_audio_input_stream(room, noise_cancellation=None)
    audio_input.set_participant("test-user")
    track, publication, participant = _make_track_available_args()
    rtc_stream = _MockAudioStream()

    with patch("livekit.rtc.AudioStream.from_track", return_value=rtc_stream) as create_stream:
        assert audio_input._on_track_available(track, publication, participant)
        await asyncio.wait_for(rtc_stream.started.wait(), timeout=1)

        assert not audio_input._on_track_available(track, publication, participant)

    create_stream.assert_called_once()
    assert audio_input._stream is rtc_stream
    assert audio_input._track is track

    await audio_input.aclose()


@pytest.mark.asyncio
async def test_stale_track_unsubscribe_does_not_close_replacement() -> None:
    room = _FakeRoom()
    audio_input = _make_audio_input_stream(room, noise_cancellation=None)
    audio_input.set_participant("test-user")
    old_track, publication, participant = _make_track_available_args()
    new_track = MagicMock()
    old_stream = _MockAudioStream()
    new_stream = _MockAudioStream()

    with patch("livekit.rtc.AudioStream.from_track", side_effect=[old_stream, new_stream]):
        assert audio_input._on_track_available(old_track, publication, participant)
        await asyncio.wait_for(old_stream.started.wait(), timeout=1)

        publication.track = new_track
        assert audio_input._on_track_available(new_track, publication, participant)
        await asyncio.wait_for(new_stream.started.wait(), timeout=1)

        audio_input._on_track_unsubscribed(old_track, publication, participant)
        await asyncio.sleep(0)

    assert audio_input._stream is new_stream
    assert audio_input._track is new_track
    assert not new_stream.ended.is_set()

    await audio_input.aclose()


@pytest.mark.asyncio
async def test_audio_input_closes_active_track_on_unsubscribe() -> None:
    room = _FakeRoom()
    audio_input = _make_audio_input_stream(room, noise_cancellation=None)
    audio_input.set_participant("test-user")
    track, publication, participant = _make_track_available_args()
    rtc_stream = _MockAudioStream()

    with patch("livekit.rtc.AudioStream.from_track", return_value=rtc_stream) as create_stream:
        assert audio_input._on_track_available(track, publication, participant)
        await asyncio.wait_for(rtc_stream.started.wait(), timeout=1)

        publication.subscribed = False
        publication.track = None
        audio_input._on_track_unsubscribed(track, publication, participant)
        await asyncio.wait_for(rtc_stream.ended.wait(), timeout=1)

    create_stream.assert_called_once()
    assert audio_input._stream is None
    assert audio_input._track is None

    await audio_input.aclose()


@pytest.mark.asyncio
async def test_pre_connect_audio_runs_once_across_concrete_track_replacement() -> None:
    room = _FakeRoom()
    pre_connect_audio_handler = SimpleNamespace(wait_for_data=AsyncMock(return_value=[]))
    audio_input = _ParticipantAudioInputStream(
        room,
        sample_rate=24000,
        num_channels=1,
        noise_cancellation=None,
        auto_gain_control=False,
        pre_connect_audio_handler=pre_connect_audio_handler,
    )
    audio_input.set_participant("test-user")
    old_track, publication, participant = _make_track_available_args()
    publication.audio_features = [AudioTrackFeature.TF_PRECONNECT_BUFFER]
    new_track = MagicMock()
    new_track.sid = publication.sid
    initial_stream = _MockAudioStream()
    replacement_stream = _MockAudioStream()

    with patch(
        "livekit.rtc.AudioStream.from_track",
        side_effect=[initial_stream, replacement_stream],
    ):
        assert audio_input._on_track_available(old_track, publication, participant)
        await asyncio.wait_for(initial_stream.started.wait(), timeout=1)

        publication.track = new_track
        assert audio_input._on_track_available(new_track, publication, participant)
        await asyncio.wait_for(replacement_stream.started.wait(), timeout=1)

    pre_connect_audio_handler.wait_for_data.assert_awaited_once_with(publication.sid)

    await audio_input.aclose()


@pytest.mark.asyncio
async def test_pre_connect_audio_retries_after_track_switch_cancels_fetch() -> None:
    room = _FakeRoom()
    first_fetch_started = asyncio.Event()
    fetch_count = 0

    async def wait_for_data(_track_id: str) -> list[rtc.AudioFrame]:
        nonlocal fetch_count
        fetch_count += 1
        if fetch_count == 1:
            first_fetch_started.set()
            await asyncio.Event().wait()
        return []

    pre_connect_audio_handler = SimpleNamespace(wait_for_data=AsyncMock(side_effect=wait_for_data))
    audio_input = _ParticipantAudioInputStream(
        room,
        sample_rate=24000,
        num_channels=1,
        noise_cancellation=None,
        auto_gain_control=False,
        pre_connect_audio_handler=pre_connect_audio_handler,
    )
    audio_input.set_participant("test-user")
    old_track, publication, participant = _make_track_available_args()
    publication.audio_features = [AudioTrackFeature.TF_PRECONNECT_BUFFER]
    new_track = MagicMock()
    new_track.sid = publication.sid
    initial_stream = _MockAudioStream()
    replacement_stream = _MockAudioStream()

    with patch(
        "livekit.rtc.AudioStream.from_track",
        side_effect=[initial_stream, replacement_stream],
    ):
        assert audio_input._on_track_available(old_track, publication, participant)
        await asyncio.wait_for(first_fetch_started.wait(), timeout=1)

        publication.track = new_track
        assert audio_input._on_track_available(new_track, publication, participant)
        await asyncio.wait_for(replacement_stream.started.wait(), timeout=1)

    await audio_input.aclose()
    assert pre_connect_audio_handler.wait_for_data.await_count == 2


@pytest.mark.asyncio
async def test_pre_connect_audio_does_not_retry_after_timeout() -> None:
    room = _FakeRoom()
    pre_connect_audio_handler = SimpleNamespace(
        wait_for_data=AsyncMock(side_effect=asyncio.TimeoutError)
    )
    audio_input = _ParticipantAudioInputStream(
        room,
        sample_rate=24000,
        num_channels=1,
        noise_cancellation=None,
        auto_gain_control=False,
        pre_connect_audio_handler=pre_connect_audio_handler,
    )
    audio_input.set_participant("test-user")
    old_track, publication, participant = _make_track_available_args()
    publication.audio_features = [AudioTrackFeature.TF_PRECONNECT_BUFFER]
    new_track = MagicMock()
    new_track.sid = publication.sid
    initial_stream = _MockAudioStream()
    replacement_stream = _MockAudioStream()

    with patch(
        "livekit.rtc.AudioStream.from_track",
        side_effect=[initial_stream, replacement_stream],
    ):
        assert audio_input._on_track_available(old_track, publication, participant)
        await asyncio.wait_for(initial_stream.started.wait(), timeout=1)

        publication.track = new_track
        assert audio_input._on_track_available(new_track, publication, participant)
        await asyncio.wait_for(replacement_stream.started.wait(), timeout=1)

    await audio_input.aclose()
    assert pre_connect_audio_handler.wait_for_data.await_count == 1


@pytest.mark.asyncio
async def test_audio_input_does_not_flush_silence_when_detached() -> None:
    room = _FakeRoom()
    audio_input = _make_audio_input_stream(room, noise_cancellation=None)
    audio_input.set_participant("test-user")
    track, publication, participant = _make_track_available_args()
    rtc_stream = _MockAudioStream()

    with patch("livekit.rtc.AudioStream.from_track", return_value=rtc_stream):
        assert audio_input._on_track_available(track, publication, participant)
        await asyncio.wait_for(rtc_stream.started.wait(), timeout=1)
        audio_input.on_detached()
        rtc_stream.end()
        assert audio_input._forward_atask is not None
        await audio_input._forward_atask

    queued_frames = audio_input._data_ch.qsize()
    await audio_input.aclose()
    assert queued_frames == 0


@pytest.mark.asyncio
async def test_selector_processor_lifecycle_across_concrete_track_replacement() -> None:
    room = _FakeRoom()
    processors: list[_MockFrameProcessor] = []

    def selector(_params: NoiseCancellationParams) -> _MockFrameProcessor:
        processor = _MockFrameProcessor()
        processors.append(processor)
        return processor

    audio_input = _make_audio_input_stream(room, noise_cancellation=selector)
    audio_input.set_participant("test-user")
    old_track, publication, participant = _make_track_available_args()
    new_track = MagicMock()
    initial_stream = _MockAudioStream()
    replacement_stream = _MockAudioStream()

    with patch(
        "livekit.rtc.AudioStream.from_track",
        side_effect=[initial_stream, replacement_stream],
    ):
        assert audio_input._on_track_available(old_track, publication, participant)
        await asyncio.wait_for(initial_stream.started.wait(), timeout=1)

        publication.track = new_track
        assert audio_input._on_track_available(new_track, publication, participant)
        await asyncio.wait_for(replacement_stream.started.wait(), timeout=1)
        assert processors[0].close_calls == 1

    assert len(processors) == 2
    assert [processor.close_calls for processor in processors] == [1, 0]

    await audio_input.aclose()
    assert [processor.close_calls for processor in processors] == [1, 1]


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
    participant.track_publications.clear()
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
