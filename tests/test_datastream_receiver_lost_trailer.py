"""DataStreamAudioReceiver must survive a lost stream trailer.

If a segment's trailer is lost, the receiver used to block on it forever, wedging
all later segments. A new stream header (or an idle timeout) must end the segment.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, _datastream_io as ds

pytestmark = pytest.mark.unit

SAMPLE_RATE = 24000
FRAME_MS = 10
CHUNK = b"\x00\x00" * int(SAMPLE_RATE * FRAME_MS / 1000)  # exactly one 10 ms frame
SENDER = "agent"


class _FakeReader:
    """Stands in for rtc.ByteStreamReader: chunks are pushed in, the trailer is
    an explicit close() — which the lost-trailer tests never call."""

    def __init__(self, stream_id: str) -> None:
        self.info = SimpleNamespace(
            stream_id=stream_id,
            attributes={"sample_rate": str(SAMPLE_RATE), "num_channels": "1"},
        )
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    def push(self, data: bytes) -> None:
        self._queue.put_nowait(data)

    def close(self) -> None:
        self._queue.put_nowait(None)


async def _start_receiver():
    room = MagicMock()
    handlers: dict = {}
    room.register_byte_stream_handler.side_effect = lambda topic, h: handlers.__setitem__(topic, h)
    room.local_participant._rpc_handlers = {}
    rpc_methods: dict = {}
    room.local_participant.register_rpc_method.side_effect = lambda name, h: (
        rpc_methods.__setitem__(name, h)
    )
    sender = SimpleNamespace(identity=SENDER)
    with patch.object(ds.utils, "wait_for_participant", AsyncMock(return_value=sender)):
        receiver = ds.DataStreamAudioReceiver(room, sender_identity=SENDER, frame_size_ms=FRAME_MS)
        await receiver.start()
    return receiver, handlers[ds.AUDIO_STREAM_TOPIC], rpc_methods[ds.RPC_CLEAR_BUFFER]


async def _next(receiver, timeout: float = 1.0):
    return await asyncio.wait_for(receiver.__anext__(), timeout)


async def test_normal_segments_are_unaffected():
    receiver, on_stream, _clear = await _start_receiver()
    try:
        first = _FakeReader("s1")
        on_stream(first, SENDER)
        first.push(CHUNK)
        assert isinstance(await _next(receiver), rtc.AudioFrame)
        first.close()
        assert isinstance(await _next(receiver), AudioSegmentEnd)

        second = _FakeReader("s2")
        on_stream(second, SENDER)
        second.push(CHUNK)
        assert isinstance(await _next(receiver), rtc.AudioFrame)
        second.close()
        assert isinstance(await _next(receiver), AudioSegmentEnd)
    finally:
        await receiver.aclose()


async def test_lost_trailer_is_recovered_by_the_next_stream():
    receiver, on_stream, _clear = await _start_receiver()
    try:
        first = _FakeReader("s1")
        on_stream(first, SENDER)
        first.push(CHUNK)
        assert isinstance(await _next(receiver), rtc.AudioFrame)
        # first.close() never happens: the trailer was lost.

        second = _FakeReader("s2")
        on_stream(second, SENDER)
        # The first segment is ended so the second one can be read.
        assert isinstance(await _next(receiver), AudioSegmentEnd)

        second.push(CHUNK)
        assert isinstance(await _next(receiver), rtc.AudioFrame)
        second.close()
        assert isinstance(await _next(receiver), AudioSegmentEnd)
    finally:
        await receiver.aclose()


async def test_clear_buffer_ends_the_segment_without_waiting_for_the_trailer():
    """The cleared segment's data is discarded anyway, so AudioSegmentEnd must
    arrive right away — no next stream header and no idle timeout needed, even
    if the trailer was lost."""
    receiver, on_stream, clear = await _start_receiver()
    try:
        first = _FakeReader("s1")
        on_stream(first, SENDER)
        first.push(CHUNK)
        assert isinstance(await _next(receiver), rtc.AudioFrame)

        # interruption: clear_buffer lands, then the stream's trailer is lost
        assert clear(SimpleNamespace(caller_identity=SENDER)) == "ok"
        assert isinstance(await _next(receiver), AudioSegmentEnd)

        # the receiver is ready for the next utterance
        second = _FakeReader("s2")
        on_stream(second, SENDER)
        second.push(CHUNK)
        assert isinstance(await _next(receiver), rtc.AudioFrame)
        second.close()
        assert isinstance(await _next(receiver), AudioSegmentEnd)
    finally:
        await receiver.aclose()


async def test_clear_buffer_with_a_healthy_trailer_still_ends_one_segment():
    """A trailer that does arrive after clear_buffer must not produce a second
    AudioSegmentEnd or disturb the next segment."""
    receiver, on_stream, clear = await _start_receiver()
    try:
        first = _FakeReader("s1")
        on_stream(first, SENDER)
        first.push(CHUNK)
        assert isinstance(await _next(receiver), rtc.AudioFrame)

        assert clear(SimpleNamespace(caller_identity=SENDER)) == "ok"
        first.close()  # healthy interruption: the trailer still arrives
        assert isinstance(await _next(receiver), AudioSegmentEnd)

        second = _FakeReader("s2")
        on_stream(second, SENDER)
        second.push(CHUNK)
        assert isinstance(await _next(receiver), rtc.AudioFrame)
        second.close()
        assert isinstance(await _next(receiver), AudioSegmentEnd)
    finally:
        await receiver.aclose()


async def test_last_stream_lost_trailer_is_recovered_by_idle_timeout(monkeypatch):
    """No following stream to supersede it: an open reader that goes silent for
    STREAM_IDLE_TIMEOUT while a segment is owed is ended anyway, so the receiver
    is ready for the next utterance instead of wedged on the lost trailer."""
    monkeypatch.setattr(ds, "STREAM_IDLE_TIMEOUT", 0.2)
    receiver, on_stream, _clear = await _start_receiver()
    try:
        first = _FakeReader("s1")
        on_stream(first, SENDER)
        first.push(CHUNK)
        assert isinstance(await _next(receiver), rtc.AudioFrame)
        # trailer lost, and no next stream arrives.
        assert isinstance(await _next(receiver, timeout=1.0), AudioSegmentEnd)

        # the receiver is not wedged: a later utterance is read normally.
        second = _FakeReader("s2")
        on_stream(second, SENDER)
        second.push(CHUNK)
        assert isinstance(await _next(receiver), rtc.AudioFrame)
        second.close()
        assert isinstance(await _next(receiver), AudioSegmentEnd)
    finally:
        await receiver.aclose()


async def test_queued_trailer_still_wins_over_the_superseded_signal():
    """Ordered channel: a header can only follow its predecessor's trailer, so
    a trailer that is already queued must end the segment normally (frames
    flushed) — the superseded signal is a fallback, never a preemption."""
    receiver, on_stream, _clear = await _start_receiver()
    try:
        first = _FakeReader("s1")
        on_stream(first, SENDER)
        # a partial (half) frame that only a normal flush would emit
        first.push(CHUNK[: len(CHUNK) // 2])
        first.close()
        second = _FakeReader("s2")
        on_stream(second, SENDER)

        frame = await _next(receiver)
        assert isinstance(frame, rtc.AudioFrame)
        assert isinstance(await _next(receiver), AudioSegmentEnd)

        second.push(CHUNK)
        assert isinstance(await _next(receiver), rtc.AudioFrame)
        second.close()
        assert isinstance(await _next(receiver), AudioSegmentEnd)
    finally:
        await receiver.aclose()
