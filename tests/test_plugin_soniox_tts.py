"""Unit tests for Soniox TTS sentence buffering and stream rotation.

The plugin buffers LLM chunks into complete sentences, feeds them into one
shared Soniox stream, and rotates to a fresh ``stream_id`` when input goes
idle for ``stream_idle_timeout`` or after a transient failure (batch replay).
These tests drive the real ``SynthesizeStream`` against a fake ``_Connection``.

Idle-timeout tests use ``virtual_time`` so timers advance deterministically.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from livekit.agents import APIStatusError
from livekit.agents.tts import AudioEmitter
from livekit.plugins import soniox
from livekit.plugins.soniox.tts import _Connection

pytestmark = [
    pytest.mark.plugin("soniox"),
    pytest.mark.virtual_time,
    pytest.mark.no_concurrent,
]

# 10 ms of mono s16le silence at 24 kHz
_SILENCE_PCM = b"\x00\x00" * 240

SENTENCES = [
    "Hello there, this is the first sentence. ",
    "And here comes a second sentence! ",
    "Finally a third one?",
]


async def test_websocket_authenticates_once_for_multiple_streams() -> None:
    sent: asyncio.Queue[str] = asyncio.Queue()
    received: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
    ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    ws.closed = False
    ws.send_str = AsyncMock(side_effect=sent.put)
    ws.receive = AsyncMock(side_effect=received.get)
    session = MagicMock(spec=aiohttp.ClientSession)
    session.ws_connect = AsyncMock(return_value=ws)
    tts = soniox.TTS(api_key="test-key", http_session=session)

    try:
        for stream_id in ("first", "second"):
            connection, _, _ = await tts._current_connection(timeout=1.0)
            waiter = asyncio.get_running_loop().create_future()
            connection.register_stream(stream_id, MagicMock(), waiter, opts=tts._opts)
            try:
                connection.send_text(stream_id, "Hello!", text_end=True)
                config = json.loads(await asyncio.wait_for(sent.get(), timeout=1.0))
                text = json.loads(await asyncio.wait_for(sent.get(), timeout=1.0))

                assert "api_key" not in config
                assert config["model"] == tts.model
                assert config["stream_id"] == stream_id
                assert text == {"stream_id": stream_id, "text": "Hello!", "text_end": True}
            finally:
                connection.unregister_stream(stream_id)
                waiter.cancel()

        session.ws_connect.assert_awaited_once_with(
            tts._opts.websocket_url, headers={"Authorization": "Bearer test-key"}
        )
    finally:
        await tts.aclose()


@pytest.mark.parametrize("stream_id", [None, ""])
@pytest.mark.parametrize("status_code", [401, 429])
@pytest.mark.parametrize("register_before_error", [False, True])
async def test_connection_error_reaches_all_streams(
    stream_id: str | None, status_code: int, register_before_error: bool
) -> None:
    tts = soniox.TTS(api_key="test-key")
    connection = _Connection(tts._opts, MagicMock())
    payload: dict[str, Any] = {
        "error_code": status_code,
        "error_message": "Connection rejected",
        "request_id": "test-request",
    }
    if stream_id is not None:
        payload["stream_id"] = stream_id

    ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    ws.closed = False
    ws.close_code = 1008
    ws.receive = AsyncMock(
        side_effect=[
            aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(payload), ""),
            aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, 1008, ""),
        ]
    )
    connection._ws = ws
    waiters: list[asyncio.Future[None]] = []

    def register_stream(name: str) -> None:
        waiter = asyncio.get_running_loop().create_future()
        waiters.append(waiter)
        connection.register_stream(name, MagicMock(), waiter, opts=tts._opts)

    try:
        if register_before_error:
            register_stream("first")
            register_stream("second")

        await connection._recv_loop()
        register_stream("closing")
        assert connection._close_task is not None
        await connection._close_task
        register_stream("closed")

        for waiter in waiters:
            with pytest.raises(APIStatusError) as exc_info:
                await asyncio.wait_for(waiter, timeout=1.0)
            assert exc_info.value.status_code == status_code
            assert exc_info.value.message == "Connection rejected"
            assert exc_info.value.retryable is (status_code == 429)
            assert exc_info.value.request_id == "test-request"

        assert not connection.is_current
        assert connection.closed
        assert connection.num_active_streams == 0
        ws.receive.assert_awaited_once()
    finally:
        await connection.aclose()
        if connection._close_task is not None:
            await connection._close_task
        for waiter in waiters:
            if waiter.done() and not waiter.cancelled():
                waiter.exception()
            else:
                waiter.cancel()
        await tts.aclose()


@dataclass
class _StreamSlot:
    emitter: AudioEmitter
    waiter: asyncio.Future[None]


class _FakeConnection:
    """Stand-in for ``_Connection``: text produces audio, ``text_end`` terminates."""

    def __init__(self, *, fail_on_send: int | None = None) -> None:
        self.is_current = True
        self.closed = False
        self.registered_ids: list[str] = []
        self.send_calls: list[tuple[str, str, bool]] = []
        self.max_open_streams = 0
        self._streams: dict[str, _StreamSlot] = {}
        self._sends = 0
        self._fail_on_send = fail_on_send

    def register_stream(
        self,
        stream_id: str,
        emitter: AudioEmitter,
        waiter: asyncio.Future[None],
        *,
        opts: Any,
    ) -> None:
        self.registered_ids.append(stream_id)
        self._streams[stream_id] = _StreamSlot(emitter, waiter)
        self.max_open_streams = max(self.max_open_streams, len(self._streams))

    def unregister_stream(self, stream_id: str) -> None:
        self._streams.pop(stream_id, None)

    def send_text(self, stream_id: str, text: str, *, text_end: bool = False) -> None:
        self.send_calls.append((stream_id, text, text_end))
        slot = self._streams.get(stream_id)
        if slot is None:
            return
        if text_end and not text:
            # server: final audio + audio_end + terminated
            if not slot.waiter.done():
                slot.waiter.set_result(None)
            return
        self._sends += 1
        if self._sends == self._fail_on_send:
            self._fail_on_send = None
            if not slot.waiter.done():
                slot.waiter.set_exception(
                    APIStatusError("transient", status_code=429, retryable=True)
                )
            return
        slot.emitter.push(_SILENCE_PCM)

    def cancel_stream(self, stream_id: str) -> None:
        slot = self._streams.get(stream_id)
        if slot is not None and not slot.waiter.done():
            slot.waiter.set_result(None)


async def _synthesize(
    fake: _FakeConnection,
    *,
    chunk_delays: list[float] | None = None,
    **tts_kwargs: Any,
) -> int:
    tts = soniox.TTS(api_key="fake-key", **tts_kwargs)

    async def _fake_current_connection(*, timeout: float) -> tuple[Any, float, bool]:
        return fake, 0.0, True

    tts._current_connection = _fake_current_connection  # type: ignore[method-assign]

    stream = tts.stream()
    delays = chunk_delays or [0.01] * len(SENTENCES)

    async def _push() -> None:
        for chunk, delay in zip(SENTENCES, delays, strict=True):
            stream.push_text(chunk)
            await asyncio.sleep(delay)
        stream.end_input()

    push_t = asyncio.create_task(_push())
    frames = 0
    async for _ in stream:
        frames += 1
    await push_t
    await stream.aclose()
    return frames


async def test_steady_flow_uses_single_stream() -> None:
    fake = _FakeConnection()
    frames = await _synthesize(fake)

    assert frames > 0
    assert len(fake.registered_ids) == 1
    text_sends = [c for c in fake.send_calls if not c[2]]
    end_sends = [c for c in fake.send_calls if c[2]]
    assert len(text_sends) == len(SENTENCES)
    assert len(end_sends) == 1


async def test_idle_stall_rotates_stream() -> None:
    fake = _FakeConnection()
    frames = await _synthesize(
        fake,
        chunk_delays=[3.0, 0.01, 0.01],
        stream_idle_timeout=1.0,
    )

    assert frames > 0
    # sentence 1 on the first stream, finalized during the stall; 2 and 3 on the next
    assert len(fake.registered_ids) == 2
    assert sum(1 for c in fake.send_calls if c[2]) == 2
    assert fake.max_open_streams == 1


async def test_transient_failure_replays_batch() -> None:
    fake = _FakeConnection(fail_on_send=1)
    frames = await _synthesize(fake)

    assert frames > 0
    assert len(fake.registered_ids) == 2
    failed_id, replacement_id = fake.registered_ids
    replayed = [text for sid, text, end in fake.send_calls if sid == replacement_id and not end]
    assert replayed[0].strip() == SENTENCES[0].strip()
    assert fake.max_open_streams == 1


async def test_invalid_stream_idle_timeout_rejected() -> None:
    with pytest.raises(ValueError):
        soniox.TTS(api_key="fake-key", stream_idle_timeout=0)
    with pytest.raises(ValueError):
        soniox.TTS(api_key="fake-key", stream_idle_timeout=-1.0)
