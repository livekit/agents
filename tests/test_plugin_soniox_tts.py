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
from unittest.mock import MagicMock

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


async def test_connection_sends_api_key_on_the_handshake() -> None:
    tts = soniox.TTS(api_key="fake-key")
    sent: list[dict[str, Any]] = []
    config_sent = asyncio.Event()

    class FakeWebSocket:
        closed = False
        close_code = None

        async def send_str(self, message: str) -> None:
            msg = json.loads(message)
            sent.append(msg)
            if "model" in msg:
                config_sent.set()

        async def receive(self) -> Any:
            await asyncio.Future()  # parked until aclose cancels the recv loop

        async def close(self) -> None:
            self.closed = True

    class FakeSession:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        async def ws_connect(self, url: str, **kwargs: Any) -> FakeWebSocket:
            self.kwargs = kwargs
            return FakeWebSocket()

    session = FakeSession()
    conn = _Connection(tts._opts, session)  # type: ignore[arg-type]
    await conn.connect()
    conn.register_stream(
        "s1", MagicMock(), asyncio.get_running_loop().create_future(), opts=tts._opts
    )
    conn.send_text("s1", "Hello.")
    await asyncio.wait_for(config_sent.wait(), timeout=1.0)
    conn.unregister_stream("s1")
    await conn.aclose()

    assert session.kwargs["headers"] == {"Authorization": "Bearer fake-key"}
    config = next(m for m in sent if "model" in m)
    # The handshake is the only carrier; the config message no longer repeats the key.
    assert "api_key" not in config
