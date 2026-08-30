from __future__ import annotations

from types import SimpleNamespace

import pytest

from livekit.agents.voice.avatar._datastream_io import DataStreamAudioReceiver


class _Reader:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.mark.unit
def test_new_audio_stream_closes_reader_with_lost_trailer() -> None:
    receiver = DataStreamAudioReceiver(SimpleNamespace(), sender_identity="sender")
    receiver._remote_participant = SimpleNamespace(identity="sender")
    current = _Reader()
    next_reader = _Reader()
    receiver._current_reader = current

    receiver._handle_stream_received(next_reader, "sender")

    assert current.closed
    assert receiver._stream_readers == [next_reader]
    assert receiver._stream_reader_changed.is_set()


@pytest.mark.unit
def test_new_audio_stream_closes_last_queued_reader_with_lost_trailer() -> None:
    receiver = DataStreamAudioReceiver(SimpleNamespace(), sender_identity="sender")
    receiver._remote_participant = SimpleNamespace(identity="sender")
    current = _Reader()
    queued = _Reader()
    next_reader = _Reader()
    receiver._current_reader = current
    receiver._stream_readers.append(queued)

    receiver._handle_stream_received(next_reader, "sender")

    assert not current.closed
    assert queued.closed
    assert receiver._stream_readers == [queued, next_reader]
    assert receiver._stream_reader_changed.is_set()
