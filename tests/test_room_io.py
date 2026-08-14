from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents import utils
from livekit.agents.types import ATTRIBUTE_PUBLISH_ON_BEHALF
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
        self.processed_frames: list[rtc.AudioFrame] = []
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
        self.processed_frames.append(frame)
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
    *,
    mix_participants: bool = False,
    frame_size_ms: int = 50,
) -> _ParticipantAudioInputStream:
    return _ParticipantAudioInputStream(
        room,
        sample_rate=24000,
        num_channels=1,
        noise_cancellation=noise_cancellation,
        auto_gain_control=False,
        pre_connect_audio_handler=None,
        frame_size_ms=frame_size_ms,
        mix_participants=mix_participants,
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


# -- multi-participant mixing tests -------------------------------------------


class _ConstantAudioStream:
    """Emits one frame of a constant sample value, then stays open without producing."""

    def __init__(self, value: int, samples: int, sample_rate: int) -> None:
        self._frame = rtc.AudioFrame(
            value.to_bytes(2, "little", signed=True) * samples, sample_rate, 1, samples
        )
        self._sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._sent:
            await asyncio.Event().wait()  # keep the stream alive, but silent
        self._sent = True
        return SimpleNamespace(frame=self._frame)

    async def aclose(self) -> None:
        pass


def _drain(chan) -> list[rtc.AudioFrame]:
    """Everything currently queued for the session."""
    frames = []
    while not chan.empty():
        frames.append(chan.recv_nowait())
    return frames


def _mixing_identities(stream: _ParticipantAudioInputStream) -> set[str]:
    """Which participants are actually feeding the mixer (it is keyed on channels)."""
    return {i for i, src in stream._mix_sources.items() if src.chan in stream._mixing}


def _make_mix_participant(
    identity: str,
    sid: str,
    *,
    kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
    attributes=None,
    publishes: bool = True,
    muted: bool = False,
) -> MagicMock:
    publication = MagicMock()
    publication.source = rtc.TrackSource.SOURCE_MICROPHONE
    publication.sid = sid
    publication.track = MagicMock()
    publication.audio_features = []
    publication.muted = muted

    participant = MagicMock()
    participant.identity = identity
    participant.kind = kind
    participant.attributes = attributes or {}
    participant.track_publications = {sid: publication} if publishes else {}
    return participant


@pytest.mark.asyncio
async def test_mix_participants_sums_every_participant_audio() -> None:
    """Both participants are heard at once: the input yields their mixed samples."""
    room = _FakeRoom()
    stream = _make_audio_input_stream(room, None, mix_participants=True, frame_size_ms=10)
    samples = 240  # 10ms @ 24kHz

    streams = [_ConstantAudioStream(v, samples, 24000) for v in (100, 200)]
    with patch("livekit.rtc.AudioStream.from_track", side_effect=lambda **kw: streams.pop(0)):
        stream.add_participant(_make_mix_participant("candidate", "TR_1"))
        stream.add_participant(_make_mix_participant("interviewer", "TR_2"))

        assert set(stream._mix_sources) == {"candidate", "interviewer"}

        frame = await asyncio.wait_for(stream.__anext__(), timeout=5)

    assert frame.samples_per_channel == samples
    assert bytes(frame.data) == (300).to_bytes(2, "little", signed=True) * samples

    await stream.aclose()


@pytest.mark.asyncio
async def test_only_producing_sources_are_registered_with_the_mixer() -> None:
    """A registered stream that delivers nothing paces the whole mixer below real time,
    so the live speakers fall behind. Only sources with a live track may be registered."""
    room = _FakeRoom()
    stream = _make_audio_input_stream(room, None, mix_participants=True, frame_size_ms=10)

    speaker = _make_mix_participant("speaker", "TR_1")
    muted = _make_mix_participant("muted", "TR_2", muted=True)
    listener = _make_mix_participant("listener", "TR_3", publishes=False)

    with patch(
        "livekit.rtc.AudioStream.from_track",
        side_effect=lambda **kw: _TickingAudioStream(240, 24000),
    ):
        for participant in (speaker, muted, listener):
            stream.add_participant(participant)
        await asyncio.sleep(0.05)  # let the forward tasks reach their live loop

        # all three are mixed participants, only the one actually sending feeds the mixer
        assert set(stream._mix_sources) == {"speaker", "muted", "listener"}
        assert _mixing_identities(stream) == {"speaker"}
        assert stream._mixer is not None
        assert stream._mix_sources["speaker"].chan in stream._mixer._streams
        # a muted track isn't even read, so nothing accumulates behind the mixer
        assert stream._mix_sources["muted"].task is None

        # unmuting brings a source back, muting drops it again
        # (rtc flips publication.muted before it emits the event)
        muted.track_publications["TR_2"].muted = False
        stream._on_track_unmuted(muted, muted.track_publications["TR_2"])
        await asyncio.sleep(0.05)
        assert _mixing_identities(stream) == {"speaker", "muted"}

        speaker.track_publications["TR_1"].muted = True
        stream._on_track_muted(speaker, speaker.track_publications["TR_1"])
        assert _mixing_identities(stream) == {"muted"}

        # and leaving takes the source with it
        stream.remove_participant("muted")
        assert not stream._mixing
        assert len(stream._mixer._streams) == 0

    await stream.aclose()


@pytest.mark.asyncio
async def test_a_resubscribed_source_gets_a_fresh_channel() -> None:
    """The old forward task is still winding down; frames it writes must not reach the new mix."""
    room = _FakeRoom()
    stream = _make_audio_input_stream(room, None, mix_participants=True, frame_size_ms=10)
    participant = _make_mix_participant("speaker", "TR_1")

    with patch(
        "livekit.rtc.AudioStream.from_track",
        side_effect=lambda **kw: _TickingAudioStream(240, 24000),
    ):
        stream.add_participant(participant)
        await asyncio.sleep(0.05)
        first_chan = stream._mix_sources["speaker"].chan

        participant.track_publications["TR_1"].muted = True
        stream._on_track_muted(participant, participant.track_publications["TR_1"])
        assert first_chan.closed  # closed before its writer is cancelled, so nothing sneaks in

        participant.track_publications["TR_1"].muted = False
        stream._on_track_unmuted(participant, participant.track_publications["TR_1"])
        await asyncio.sleep(0.05)

        second_chan = stream._mix_sources["speaker"].chan
        assert second_chan is not first_chan
        assert not second_chan.closed
        assert second_chan in stream._mixer._streams
        assert first_chan not in stream._mixer._streams

    await stream.aclose()


@pytest.mark.asyncio
async def test_mixed_pre_connect_buffer_runs_through_the_participants_processor() -> None:
    """The live track is filtered by its stream; the pre-connect buffer must be too."""
    room = _FakeRoom()
    processors: list[_MockFrameProcessor] = []

    def selector(_params: NoiseCancellationParams) -> _MockFrameProcessor:
        processors.append(_MockFrameProcessor())
        return processors[-1]

    buffered = rtc.AudioFrame(b"\x07\x00" * 240, 24000, 1, 240)
    loaded = asyncio.Event()
    handler = MagicMock()

    async def _wait_for_data(_sid: str) -> list[rtc.AudioFrame]:
        await loaded.wait()
        return [buffered]

    handler.wait_for_data = _wait_for_data

    stream = _ParticipantAudioInputStream(
        room,
        sample_rate=24000,
        num_channels=1,
        noise_cancellation=selector,
        auto_gain_control=False,
        pre_connect_audio_handler=handler,
        frame_size_ms=10,
        mix_participants=True,
    )

    participant = _make_mix_participant("speaker", "TR_1")
    participant.track_publications["TR_1"].audio_features = [AudioTrackFeature.TF_PRECONNECT_BUFFER]

    with patch(
        "livekit.rtc.AudioStream.from_track",
        side_effect=lambda **kw: _TickingAudioStream(240, 24000),
    ):
        stream.add_participant(participant)
        await asyncio.sleep(0.05)

        source = stream._mix_sources["speaker"]
        assert source.processor is processors[0]  # kept on the source, not on the shared stream
        # a source waiting on its buffer would stall every other speaker
        assert _mixing_identities(stream) == set()

        loaded.set()
        await asyncio.sleep(0.05)
        assert _mixing_identities(stream) == {"speaker"}

    assert len(processors[0].processed_frames) == 1
    # pre-join audio is not concurrent with anyone: it reaches the session directly, so it
    # cannot leave this source permanently behind the mix
    first = await asyncio.wait_for(stream.__anext__(), timeout=5)
    assert bytes(first.data) == bytes(buffered.data)

    await stream.aclose()


@pytest.mark.asyncio
async def test_resubscribe_before_the_forwarder_starts_leaves_the_live_source_mixed() -> None:
    """Several room events can be dispatched in one loop iteration, so a forward task can find
    its source already resubscribed by the time it first runs. It must not touch the successor's
    channel: writing to it leaks the old track, and unregistering it drops a live speaker."""
    room = _FakeRoom()
    stream = _make_audio_input_stream(room, None, mix_participants=True, frame_size_ms=10)
    participant = _make_mix_participant("speaker", "TR_1")
    publication = participant.track_publications["TR_1"]

    with patch(
        "livekit.rtc.AudioStream.from_track",
        side_effect=lambda **kw: _TickingAudioStream(240, 24000),
    ):
        # all synchronous: the first forward task never gets a turn before it is replaced
        stream.add_participant(participant)
        first_chan = stream._mix_sources["speaker"].chan
        publication.muted = True
        stream._on_track_muted(participant, publication)
        publication.muted = False
        stream._on_track_unmuted(participant, publication)

        live_chan = stream._mix_sources["speaker"].chan
        assert live_chan is not first_chan

        await asyncio.sleep(0.1)  # let the stale task run and then be cancelled

        assert _mixing_identities(stream) == {"speaker"}
        assert live_chan in stream._mixer._streams

    await stream.aclose()


@pytest.mark.asyncio
async def test_unmuting_does_not_wait_for_the_pre_connect_buffer_again() -> None:
    """The handler drops the buffer after one read, so a second wait just burns the timeout
    while the freshly subscribed stream backs up."""
    room = _FakeRoom()
    reads: list[str] = []
    handler = MagicMock()

    async def _wait_for_data(sid: str) -> list[rtc.AudioFrame]:
        reads.append(sid)
        if len(reads) > 1:
            await asyncio.sleep(3)  # what the un-resolved future would do
            raise asyncio.TimeoutError
        return [rtc.AudioFrame(b"\x07\x00" * 240, 24000, 1, 240)]

    handler.wait_for_data = _wait_for_data

    stream = _ParticipantAudioInputStream(
        room,
        sample_rate=24000,
        num_channels=1,
        noise_cancellation=None,
        auto_gain_control=False,
        pre_connect_audio_handler=handler,
        frame_size_ms=10,
        mix_participants=True,
    )

    participant = _make_mix_participant("speaker", "TR_1")
    publication = participant.track_publications["TR_1"]
    publication.audio_features = [AudioTrackFeature.TF_PRECONNECT_BUFFER]
    publication.track.sid = "TR_1"

    with patch(
        "livekit.rtc.AudioStream.from_track",
        side_effect=lambda **kw: _TickingAudioStream(240, 24000),
    ):
        stream.add_participant(participant)
        await asyncio.sleep(0.05)
        assert reads == ["TR_1"]

        publication.muted = True
        stream._on_track_muted(participant, publication)
        publication.muted = False
        stream._on_track_unmuted(participant, publication)
        await asyncio.sleep(0.05)

        # the buffer is not read a second time, so the speaker is mixed in straight away
        assert reads == ["TR_1"]
        assert _mixing_identities(stream) == {"speaker"}

    await stream.aclose()


@pytest.mark.asyncio
async def test_pre_connect_audio_is_not_spliced_into_an_ongoing_conversation() -> None:
    """It bypasses the mixer, so writing it while others are live would interleave two
    speakers frame by frame in the channel the session reads."""
    room = _FakeRoom()
    buffered = rtc.AudioFrame(b"\x07\x00" * 240, 24000, 1, 240)
    handler = MagicMock()
    handler.wait_for_data = AsyncMock(return_value=[buffered])

    stream = _ParticipantAudioInputStream(
        room,
        sample_rate=24000,
        num_channels=1,
        noise_cancellation=None,
        auto_gain_control=False,
        pre_connect_audio_handler=handler,
        frame_size_ms=10,
        mix_participants=True,
    )

    incumbent = _make_mix_participant("incumbent", "TR_1")
    joiner = _make_mix_participant("joiner", "TR_2")
    joiner.track_publications["TR_2"].audio_features = [AudioTrackFeature.TF_PRECONNECT_BUFFER]
    joiner.track_publications["TR_2"].track.sid = "TR_2"

    with patch(
        "livekit.rtc.AudioStream.from_track",
        side_effect=lambda **kw: _TickingAudioStream(240, 24000),
    ):
        stream.add_participant(incumbent)
        await asyncio.sleep(0.05)
        assert _mixing_identities(stream) == {"incumbent"}  # conversation already under way

        stream.add_participant(joiner)
        await asyncio.sleep(0.05)
        assert _mixing_identities(stream) == {"incumbent", "joiner"}

        # the joiner is mixed in live, but their pre-join words were not spliced in
        mixed = b"".join(bytes(f.data) for f in _drain(stream._data_ch))
        assert bytes(buffered.data) not in mixed

    await stream.aclose()


@pytest.mark.asyncio
async def test_departing_mid_flush_is_not_logged_as_an_error(caplog) -> None:
    """ChanClosed derives from Exception; the broad handler must not report it as a failure."""
    room = _FakeRoom()
    handler = MagicMock()
    handler.wait_for_data = AsyncMock(
        return_value=[rtc.AudioFrame(b"\x07\x00" * 240, 24000, 1, 240)]
    )

    stream = _ParticipantAudioInputStream(
        room,
        sample_rate=24000,
        num_channels=1,
        noise_cancellation=None,
        auto_gain_control=False,
        pre_connect_audio_handler=handler,
        frame_size_ms=10,
        mix_participants=True,
    )

    participant = _make_mix_participant("speaker", "TR_1")
    participant.track_publications["TR_1"].audio_features = [AudioTrackFeature.TF_PRECONNECT_BUFFER]
    participant.track_publications["TR_1"].track.sid = "TR_1"

    with caplog.at_level(logging.ERROR, logger="livekit.agents"):
        with patch(
            "livekit.rtc.AudioStream.from_track",
            side_effect=lambda **kw: _TickingAudioStream(240, 24000),
        ):
            stream.add_participant(participant)
            stream._data_ch.close()  # the session shuts down while the buffer is in flight
            await asyncio.sleep(0.05)

    assert [record.message for record in caplog.records if record.levelno >= logging.ERROR] == []

    await stream.aclose()


@pytest.mark.asyncio
async def test_detached_input_drops_pre_connect_audio() -> None:
    """Pre-connect frames bypass the mixer, so they need their own detached check."""
    room = _FakeRoom()
    handler = MagicMock()
    handler.wait_for_data = AsyncMock(
        return_value=[rtc.AudioFrame(b"\x07\x00" * 240, 24000, 1, 240)]
    )

    stream = _ParticipantAudioInputStream(
        room,
        sample_rate=24000,
        num_channels=1,
        noise_cancellation=None,
        auto_gain_control=False,
        pre_connect_audio_handler=handler,
        frame_size_ms=10,
        mix_participants=True,
    )
    stream.on_detached()  # session.input.set_audio_enabled(False)

    participant = _make_mix_participant("speaker", "TR_1")
    participant.track_publications["TR_1"].audio_features = [AudioTrackFeature.TF_PRECONNECT_BUFFER]

    with patch(
        "livekit.rtc.AudioStream.from_track",
        side_effect=lambda **kw: _TickingAudioStream(240, 24000),
    ):
        stream.add_participant(participant)
        await asyncio.sleep(0.05)

        # the source still paces the mixer, but nothing reaches the session
        assert _mixing_identities(stream) == {"speaker"}
        assert stream._data_ch.empty()

    await stream.aclose()


class _TickingAudioStream:
    """A live mic: one frame per frame-interval, forever.

    Paced with a real sleep so virtual time can advance between frames; spinning on sleep(0)
    would keep the loop busy and freeze the clock.
    """

    def __init__(self, samples: int, sample_rate: int) -> None:
        self._frame = rtc.AudioFrame(b"\x01\x00" * samples, sample_rate, 1, samples)
        self._interval = samples / sample_rate

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(self._interval)
        return SimpleNamespace(frame=self._frame)

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_mix_participant_departure_is_not_logged_as_an_error(caplog) -> None:
    """A participant leaving closes their sink mid-forward: an expected end, not a crash."""
    room = _FakeRoom()
    stream = _make_audio_input_stream(room, None, mix_participants=True, frame_size_ms=10)

    with patch(
        "livekit.rtc.AudioStream.from_track",
        side_effect=lambda **kw: _TickingAudioStream(240, 24000),
    ):
        stream.add_participant(_make_mix_participant("candidate", "TR_1"))

    source = stream._mix_sources["candidate"]
    assert source.task is not None

    with caplog.at_level(logging.ERROR, logger="livekit.agents"):
        source.chan.close()  # what remove_participant does while frames are in flight
        await asyncio.wait_for(source.task, timeout=5)  # returns, never raises ChanClosed

    assert [record.message for record in caplog.records if record.levelno >= logging.ERROR] == []

    await stream.aclose()


@pytest.mark.asyncio
async def test_mix_participants_warns_on_a_shared_frame_processor(caplog) -> None:
    """One stateful processor cannot serve every speaker at once — say so up front."""
    room = _FakeRoom()

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        stream = _make_audio_input_stream(room, _MockFrameProcessor(), mix_participants=True)

    assert any("noise cancellation processor is shared" in r.message for r in caplog.records)

    await stream.aclose()


@pytest.mark.asyncio
async def test_mix_participants_tracks_room_membership() -> None:
    """RoomIO mixes in every accepted participant, but never its own avatar worker."""
    room = _FakeRoom()
    agent_session = SimpleNamespace(
        off=MagicMock(),
        _on_room_io_participant_linked=MagicMock(),
        input=SimpleNamespace(audio=None, video=None),
        output=SimpleNamespace(audio=None, transcription=None),
    )
    room_io = RoomIO(agent_session, room)
    room_io._audio_input = _make_audio_input_stream(room, None, mix_participants=True)

    candidate = _make_mix_participant("candidate", "TR_1")
    interviewer = _make_mix_participant("interviewer", "TR_2")
    avatar = _make_mix_participant(
        "avatar", "TR_3", attributes={ATTRIBUTE_PUBLISH_ON_BEHALF: "local"}
    )

    with patch("livekit.rtc.AudioStream.from_track", side_effect=lambda **kw: _MockAudioStream()):
        for participant in (candidate, interviewer, avatar):
            room_io._on_participant_connected(participant)

        assert set(room_io._audio_input._mix_sources) == {"candidate", "interviewer"}
        # only the first participant is linked, the rest are input-only
        assert room_io.linked_participant is candidate

        room_io._on_participant_disconnected(interviewer)
        assert set(room_io._audio_input._mix_sources) == {"candidate"}

    await room_io._audio_input.aclose()


@pytest.mark.asyncio
async def test_mix_does_not_relink_to_a_participant_who_cannot_speak() -> None:
    """An observer publishes no microphone, so it must not keep the session alive on its own.
    A muted participant can unmute, so it must."""
    for publishes_mic, expect_closed in ((False, True), (True, False)):
        room = _FakeRoom()
        agent_session = SimpleNamespace(
            off=MagicMock(),
            _on_room_io_participant_linked=MagicMock(),
            _close_soon=MagicMock(),
            input=SimpleNamespace(audio=None, video=None),
            output=SimpleNamespace(audio=None, transcription=None),
        )
        room_io = RoomIO(agent_session, room)
        room_io._audio_input = _make_audio_input_stream(room, None, mix_participants=True)

        speaker = _make_mix_participant("speaker", "TR_1")
        # either a muted microphone, or no microphone at all
        other = _make_mix_participant("other", "TR_2", publishes=publishes_mic, muted=True)
        room.remote_participants = {"speaker": speaker, "other": other}

        with patch(
            "livekit.rtc.AudioStream.from_track", side_effect=lambda **kw: _MockAudioStream()
        ):
            room_io._on_participant_connected(speaker)
            room_io._on_participant_connected(other)
            assert room_io.linked_participant is speaker

            speaker.disconnect_reason = rtc.DisconnectReason.CLIENT_INITIATED
            room.remote_participants.pop("speaker")
            room_io._on_participant_disconnected(speaker)

        assert agent_session._close_soon.called is expect_closed
        assert (room_io.linked_participant is other) is not expect_closed

        await room_io._audio_input.aclose()


@pytest.mark.asyncio
async def test_mix_relinks_instead_of_closing_when_the_linked_participant_leaves() -> None:
    """The linked participant is one of N contributors: their exit must not hang up on the rest."""
    room = _FakeRoom()
    agent_session = SimpleNamespace(
        off=MagicMock(),
        _on_room_io_participant_linked=MagicMock(),
        _close_soon=MagicMock(),
        input=SimpleNamespace(audio=None, video=None),
        output=SimpleNamespace(audio=None, transcription=None),
    )
    room_io = RoomIO(agent_session, room)
    room_io._audio_input = _make_audio_input_stream(room, None, mix_participants=True)

    candidate = _make_mix_participant("candidate", "TR_1")
    interviewer = _make_mix_participant("interviewer", "TR_2")
    room.remote_participants = {"candidate": candidate, "interviewer": interviewer}

    with patch("livekit.rtc.AudioStream.from_track", side_effect=lambda **kw: _MockAudioStream()):
        room_io._on_participant_connected(candidate)
        room_io._on_participant_connected(interviewer)
        assert room_io.linked_participant is candidate

        candidate.disconnect_reason = rtc.DisconnectReason.CLIENT_INITIATED
        room.remote_participants.pop("candidate")
        room_io._on_participant_disconnected(candidate)

        # outputs follow the remaining speaker, the session stays up
        assert room_io.linked_participant is interviewer
        assert room_io._participant_identity == "interviewer"
        agent_session._close_soon.assert_not_called()
        # a relink is not a new call: re-notifying would re-arm the AEC warmup and swallow
        # everyone's audio through the agent's next reply
        agent_session._on_room_io_participant_linked.assert_called_once()

        # last one out does close it
        interviewer.disconnect_reason = rtc.DisconnectReason.CLIENT_INITIATED
        room.remote_participants.pop("interviewer")
        room_io._on_participant_disconnected(interviewer)
        agent_session._close_soon.assert_called_once()

    await room_io._audio_input.aclose()


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
