from __future__ import annotations

from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents.voice.room_io._input import _ParticipantAudioInputStream
from livekit.agents.voice.room_io.types import NoiseCancellationParams


class _MockAudioStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def aclose(self) -> None:
        pass


class _FakeRoom:
    def __init__(self) -> None:
        self._events: dict[str, list[object]] = defaultdict(list)
        self.remote_participants: dict[str, object] = {}
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


@pytest.mark.asyncio
async def test_selector_frame_processor_receives_lifecycle_calls() -> None:
    """When a NoiseCancellationSelector returns a FrameProcessor, it should
    receive _on_stream_info_updated and _on_credentials_updated calls."""
    room = _FakeRoom()
    processor = _MockFrameProcessor()
    stream = _make_audio_input_stream(room, noise_cancellation=lambda _params: processor)
    stream.set_participant("test-user")

    track, publication, participant = _make_track_available_args()

    with patch("livekit.rtc.AudioStream.from_track", side_effect=lambda **kw: _MockAudioStream()):
        stream._on_track_available(track, publication, participant)

    assert stream._processor is processor
    assert len(processor.stream_info_calls) == 1
    assert processor.stream_info_calls[0] == {
        "room_name": "test-room",
        "participant_identity": "test-user",
        "publication_sid": "TR_123",
    }
    assert len(processor.credentials_calls) == 1
    assert processor.credentials_calls[0] == {
        "token": "test-token",
        "url": "wss://test.livekit.cloud",
    }

    await stream.aclose()


@pytest.mark.asyncio
async def test_selector_noise_cancellation_options_does_not_set_processor() -> None:
    """When a NoiseCancellationSelector returns NoiseCancellationOptions,
    self._processor should remain None (no lifecycle management needed)."""
    room = _FakeRoom()
    nc_options = rtc.NoiseCancellationOptions(module_id="bvc", options={})
    stream = _make_audio_input_stream(room, noise_cancellation=lambda _params: nc_options)
    stream.set_participant("test-user")

    track, publication, participant = _make_track_available_args()

    with patch("livekit.rtc.AudioStream.from_track", side_effect=lambda **kw: _MockAudioStream()):
        stream._on_track_available(track, publication, participant)

    assert stream._processor is None

    await stream.aclose()


@pytest.mark.asyncio
async def test_selector_closes_previous_processor_on_track_switch() -> None:
    """When a new track triggers _on_track_available, the previous
    FrameProcessor should be closed and the new one should receive
    lifecycle calls."""
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
        stream._on_track_available(track1, pub1, participant)
        stream._on_track_available(track2, pub2, participant)

    assert len(processors) == 2

    assert processors[0].close_calls == 1

    assert stream._processor is processors[1]
    assert len(processors[1].stream_info_calls) == 1
    assert len(processors[1].credentials_calls) == 1

    await stream.aclose()


@pytest.mark.asyncio
async def test_selector_clears_processor_when_returning_options_after_processor() -> None:
    """When a selector returns a FrameProcessor for one track but
    NoiseCancellationOptions for the next, self._processor must be
    cleared to prevent stale lifecycle calls on the closed processor."""
    room = _FakeRoom()
    processor = _MockFrameProcessor()
    nc_options = rtc.NoiseCancellationOptions(module_id="bvc", options={})
    results = iter([processor, nc_options])

    stream = _make_audio_input_stream(room, noise_cancellation=lambda _params: next(results))
    stream.set_participant("test-user")

    track1, pub1, participant = _make_track_available_args(sid="TR_001")
    track2, pub2, _ = _make_track_available_args(sid="TR_002")

    with patch("livekit.rtc.AudioStream.from_track", side_effect=lambda **kw: _MockAudioStream()):
        stream._on_track_available(track1, pub1, participant)

        assert stream._processor is processor
        assert len(processor.stream_info_calls) == 1

        stream._on_track_available(track2, pub2, participant)

    assert processor.close_calls == 1
    assert stream._processor is None

    await stream.aclose()


@pytest.mark.asyncio
async def test_token_refresh_does_not_hit_closed_processor_after_track_unpublish() -> None:
    """After a track is unpublished with no replacement, _on_token_refreshed
    must not call _on_credentials_updated on the closed processor."""
    room = _FakeRoom()
    processor = _MockFrameProcessor()
    stream = _make_audio_input_stream(room, noise_cancellation=lambda _params: processor)
    stream.set_participant("test-user")

    track, publication, participant = _make_track_available_args()

    with patch("livekit.rtc.AudioStream.from_track", side_effect=lambda **kw: _MockAudioStream()):
        stream._on_track_available(track, publication, participant)

    assert stream._processor is processor
    assert len(processor.credentials_calls) == 1

    stream._on_track_unavailable(publication, participant)

    assert processor.close_calls == 1
    assert stream._processor is None

    room._token = "refreshed-token"
    room._server_url = "wss://refreshed.livekit.cloud"
    stream._on_token_refreshed()

    assert len(processor.credentials_calls) == 1

    await stream.aclose()


@pytest.mark.asyncio
async def test_aclose_closes_active_processor() -> None:
    """aclose() must deterministically close an active FrameProcessor
    rather than relying on garbage collection."""
    room = _FakeRoom()
    processor = _MockFrameProcessor()
    stream = _make_audio_input_stream(room, noise_cancellation=lambda _params: processor)
    stream.set_participant("test-user")

    track, publication, participant = _make_track_available_args()

    with patch("livekit.rtc.AudioStream.from_track", side_effect=lambda **kw: _MockAudioStream()):
        stream._on_track_available(track, publication, participant)

    assert stream._processor is processor

    await stream.aclose()

    assert processor.close_calls == 1
    assert stream._processor is None
