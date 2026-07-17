"""Unit tests for Soniox TTS stream idle / FlushSentinel rotation.

Covers the #6225 / #6425 fix: ending an open ``stream_id`` with ``text_end`` when
input stalls or a FlushSentinel arrives, then using a new ``stream_id`` when text
resumes (idle path). Flush ends the Soniox stream so the segment completes cleanly.

Idle-timeout tests use ``virtual_time`` so ``asyncio.sleep`` / ``wait_for`` timers
advance deterministically (no wall-clock flake).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from livekit.agents import APIConnectOptions, APIStatusError
from livekit.agents.tts import AudioEmitter
from livekit.plugins.soniox import tts as soniox_tts

pytestmark = [
    pytest.mark.plugin("soniox"),
    pytest.mark.virtual_time,
    pytest.mark.no_concurrent,
]

# 10 ms of mono s16le silence at 24 kHz — enough for AudioEmitter to emit frames.
_SILENCE_PCM = b"\x00\x00" * 240


@dataclass
class _StreamSlot:
    emitter: AudioEmitter
    waiter: asyncio.Future[None]


class _FakeConnection:
    """Minimal stand-in for ``_Connection`` that records ``send_text`` and pushes audio."""

    def __init__(self) -> None:
        self.is_current = True
        self.closed = False
        self.send_calls: list[tuple[str, str, bool]] = []
        self._streams: dict[str, _StreamSlot] = {}
        self.registered_ids: list[str] = []
        self.unregistered_ids: list[str] = []

    def register_stream(
        self,
        stream_id: str,
        emitter: AudioEmitter,
        waiter: asyncio.Future[None],
        *,
        opts: object,
    ) -> None:
        self._streams[stream_id] = _StreamSlot(emitter=emitter, waiter=waiter)
        self.registered_ids.append(stream_id)

    def unregister_stream(self, stream_id: str) -> None:
        self._streams.pop(stream_id, None)
        self.unregistered_ids.append(stream_id)

    def send_text(self, stream_id: str, text: str, *, text_end: bool = False) -> None:
        self.send_calls.append((stream_id, text, text_end))
        slot = self._streams.get(stream_id)
        if slot is None:
            return
        if text and not text_end:
            slot.emitter.push(_SILENCE_PCM)
        if text_end:
            if not slot.waiter.done():
                slot.waiter.set_result(None)

    def cancel_stream(self, stream_id: str) -> None:
        slot = self._streams.pop(stream_id, None)
        if slot is not None and not slot.waiter.done():
            slot.waiter.set_result(None)


def _text_end_calls(conn: _FakeConnection) -> list[tuple[str, str, bool]]:
    return [c for c in conn.send_calls if c[2]]


def _text_calls(conn: _FakeConnection) -> list[tuple[str, str, bool]]:
    return [c for c in conn.send_calls if c[1] and not c[2]]


def _patch_connection(tts: soniox_tts.TTS, fake_conn: _FakeConnection) -> None:
    async def _current_connection(*, timeout: float) -> tuple[_FakeConnection, float, bool]:
        return fake_conn, 0.0, False

    tts._current_connection = _current_connection  # type: ignore[method-assign]


async def _consume_stream(stream: soniox_tts.SynthesizeStream) -> None:
    async for _ev in stream:
        pass


@pytest.mark.asyncio
async def test_flush_sentinel_sends_text_end() -> None:
    """FlushSentinel must send text_end (previously a no-op on Soniox)."""
    fake_conn = _FakeConnection()
    tts = soniox_tts.TTS(api_key="test-key")
    _patch_connection(tts, fake_conn)
    stream = soniox_tts.SynthesizeStream(
        tts=tts, conn_options=APIConnectOptions(max_retry=0, timeout=1.0)
    )

    consumer = asyncio.create_task(_consume_stream(stream))
    await asyncio.sleep(0)
    stream.push_text("hello")
    stream.flush()
    stream.end_input()
    await consumer
    await stream.aclose()

    texts = _text_calls(fake_conn)
    text_ends = _text_end_calls(fake_conn)
    assert len(texts) == 1
    assert texts[0][1] == "hello"
    assert len(text_ends) == 1
    assert text_ends[0][0] == texts[0][0]


@pytest.mark.asyncio
async def test_flush_before_any_text_does_not_send_text_end() -> None:
    fake_conn = _FakeConnection()
    tts = soniox_tts.TTS(api_key="test-key")
    _patch_connection(tts, fake_conn)
    stream = soniox_tts.SynthesizeStream(
        tts=tts, conn_options=APIConnectOptions(max_retry=0, timeout=1.0)
    )

    consumer = asyncio.create_task(_consume_stream(stream))
    await asyncio.sleep(0)
    stream.flush()
    stream.push_text("only")
    stream.end_input()
    await consumer
    await stream.aclose()

    texts = _text_calls(fake_conn)
    text_ends = _text_end_calls(fake_conn)
    assert len(texts) == 1
    assert texts[0][1] == "only"
    assert len(text_ends) == 1


@pytest.mark.asyncio
async def test_idle_timeout_sends_text_end_and_rotates_stream_id() -> None:
    fake_conn = _FakeConnection()
    tts = soniox_tts.TTS(api_key="test-key", stream_idle_timeout=1.0)
    _patch_connection(tts, fake_conn)
    stream = soniox_tts.SynthesizeStream(
        tts=tts, conn_options=APIConnectOptions(max_retry=0, timeout=1.0)
    )

    consumer = asyncio.create_task(_consume_stream(stream))
    await asyncio.sleep(0)
    stream.push_text("before idle")
    # Virtual time autojumps past the 1s idle timeout.
    await asyncio.sleep(2.0)
    stream.push_text(" after idle")
    stream.end_input()
    await consumer
    await stream.aclose()

    texts = _text_calls(fake_conn)
    text_ends = _text_end_calls(fake_conn)
    assert len(texts) == 2
    assert texts[0][1] == "before idle"
    assert texts[1][1] == " after idle"
    assert texts[0][0] != texts[1][0]
    assert len(text_ends) == 2
    assert texts[0][0] == text_ends[0][0]
    assert texts[1][0] == text_ends[1][0]


@pytest.mark.asyncio
async def test_stream_idle_timeout_zero_disables_idle_rotation() -> None:
    fake_conn = _FakeConnection()
    tts = soniox_tts.TTS(api_key="test-key", stream_idle_timeout=0)
    _patch_connection(tts, fake_conn)
    stream = soniox_tts.SynthesizeStream(
        tts=tts, conn_options=APIConnectOptions(max_retry=0, timeout=1.0)
    )

    consumer = asyncio.create_task(_consume_stream(stream))
    await asyncio.sleep(0)
    stream.push_text("chunk1")
    # With idle disabled, a long virtual pause must not rotate stream_id.
    await asyncio.sleep(5.0)
    stream.push_text(" chunk2")
    stream.end_input()
    await consumer
    await stream.aclose()

    texts = _text_calls(fake_conn)
    text_ends = _text_end_calls(fake_conn)
    assert len(texts) == 2
    assert texts[0][0] == texts[1][0]
    assert len(text_ends) == 1


def test_stream_idle_timeout_default_is_disabled() -> None:
    tts = soniox_tts.TTS(api_key="test-key")
    assert tts._opts.stream_idle_timeout == 0.0  # noqa: SLF001
    assert soniox_tts.DEFAULT_STREAM_IDLE_TIMEOUT == 0.0


def test_stream_idle_timeout_rejects_negative() -> None:
    with pytest.raises(ValueError, match="stream_idle_timeout"):
        soniox_tts.TTS(api_key="test-key", stream_idle_timeout=-1)


def test_update_options_stream_idle_timeout() -> None:
    tts = soniox_tts.TTS(api_key="test-key", stream_idle_timeout=5.0)
    tts.update_options(stream_idle_timeout=2.5)
    assert tts._opts.stream_idle_timeout == 2.5  # noqa: SLF001


@pytest.mark.asyncio
async def test_update_options_applies_to_in_flight_stream_idle() -> None:
    """update_options must affect the active stream's next idle wait, not only TTS._opts."""
    fake_conn = _FakeConnection()
    # Start with idle disabled so the first pause does not rotate.
    tts = soniox_tts.TTS(api_key="test-key", stream_idle_timeout=0)
    _patch_connection(tts, fake_conn)
    # Must go through TTS.stream() so update_options can find the live stream.
    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))

    consumer = asyncio.create_task(_consume_stream(stream))
    await asyncio.sleep(0)
    stream.push_text("before")
    await asyncio.sleep(2.0)
    # Still one stream_id — idle was disabled.
    assert len(_text_calls(fake_conn)) == 1
    assert len(_text_end_calls(fake_conn)) == 0

    tts.update_options(stream_idle_timeout=1.0)
    # Wake the blocked unlimited recv so the next wait re-reads the new timeout.
    stream.push_text(" mid")
    await asyncio.sleep(2.0)
    stream.push_text(" after")
    stream.end_input()
    await consumer
    await stream.aclose()

    texts = _text_calls(fake_conn)
    text_ends = _text_end_calls(fake_conn)
    assert len(texts) == 3
    # "before" and " mid" share a stream_id; idle then rotates before " after".
    assert texts[0][0] == texts[1][0]
    assert texts[1][0] != texts[2][0]
    assert len(text_ends) == 2
    assert texts[0][0] == text_ends[0][0]
    assert texts[2][0] == text_ends[1][0]


@pytest.mark.asyncio
async def test_server_error_surfaces_while_input_still_open() -> None:
    """Server errors on the waiter must fail the stream before end_input (no prolonged silence)."""
    fake_conn = _FakeConnection()
    tts = soniox_tts.TTS(api_key="test-key", stream_idle_timeout=0)
    _patch_connection(tts, fake_conn)
    stream = soniox_tts.SynthesizeStream(
        tts=tts, conn_options=APIConnectOptions(max_retry=0, timeout=1.0)
    )

    consumer = asyncio.create_task(_consume_stream(stream))
    await asyncio.sleep(0)
    stream.push_text("hello")
    await asyncio.sleep(0)

    assert len(fake_conn.registered_ids) == 1
    slot = fake_conn._streams[fake_conn.registered_ids[0]]  # noqa: SLF001
    slot.waiter.set_exception(APIStatusError("synthetic server error", status_code=500))

    # Do not call end_input — error must surface from the concurrent waiter monitor.
    with pytest.raises(APIStatusError, match="synthetic server error"):
        await asyncio.wait_for(consumer, timeout=2.0)
    await stream.aclose()
