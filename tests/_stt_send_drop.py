"""Shared harness for the send-side WebSocket-drop acceptance matrix (issue #6473).

Every STT plugin builds its ``SpeechStream`` differently, so each provider supplies a
``build_stream(session)`` callable and this harness injects a mid-send socket drop.
It encodes the release invariant from the issue:

    one unexpected send-side drop  -> one retryable APIError (APIConnectionError, or
                                      APIStatusError for slng) that SpeechStream._main_task
                                      turns into one bounded reconnect
    a drop during shutdown         -> a quiet return, no reconnect loop
    the in-flight audio frame      -> attempted at most once (no tight replay), loss observable

This module is intentionally un-marked: it is imported by the per-plugin
``test_plugin_<name>_stt.py`` files (each owns its ``pytest.mark.plugin(...)`` marker so CI's
``--plugin <name>`` isolation keeps working). It is not collected as a test module — the
category scanner only globs ``test_*.py``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import aiohttp
import pytest

from livekit import rtc

# The three raw failures aiohttp surfaces when send_bytes()/send_str() hits a peer-closed socket.
SEND_DROP_ERRORS = [
    pytest.param(
        lambda: aiohttp.ClientConnectionResetError("Cannot write to closing transport"),
        id="ClientConnectionResetError",
    ),
    pytest.param(lambda: aiohttp.ClientOSError(32, "Broken pipe"), id="ClientOSError"),
    pytest.param(lambda: ConnectionError("connection reset by peer"), id="ConnectionError"),
]


class DropWebSocket:
    """Lets the first ``setup_sends`` messages (handshake) succeed, then raises
    ``error_factory()`` on every later send — i.e. the socket drops mid-stream.

    ``audio_sends`` counts sends *after* the handshake: it must be exactly 1, proving the
    plugin surfaces the error after a single attempt instead of retrying the send in a loop.
    """

    def __init__(
        self, error_factory, *, setup_sends: int = 0, ready_messages=(), drop_on_bytes: bool = False
    ) -> None:
        self._error_factory = error_factory
        self._setup_sends = setup_sends
        # drop_on_bytes: audio (send_bytes) always drops, control frames (send_str/send_json) always
        # pass. Needed for plugins that reconnect inside _run (e.g. slng's endpoint failover), where
        # every reconnection re-sends a handshake that must succeed regardless of the global count.
        self._drop_on_bytes = drop_on_bytes
        self._ready = list(ready_messages)  # TEXT frames recv_task must see (e.g. a server "ready")
        self._sends = 0
        self.close_code: int | None = None
        self.sent: list[object] = []
        self.audio_sends = 0

    async def _maybe_drop(self, payload: object, *, is_bytes: bool) -> None:
        if self._drop_on_bytes:
            if not is_bytes:
                self.sent.append(payload)
                return
            self.audio_sends += 1
            raise self._error_factory()
        self._sends += 1
        if self._sends <= self._setup_sends:
            self.sent.append(payload)
            return
        self.audio_sends += 1
        raise self._error_factory()

    async def send_str(self, message: str) -> None:
        await self._maybe_drop(message, is_bytes=False)

    async def send_bytes(self, data: bytes) -> None:
        await self._maybe_drop(data, is_bytes=True)

    async def send_json(self, data: object) -> None:
        await self._maybe_drop(data, is_bytes=False)

    async def receive(self):
        # deliver any handshake frames first (unblocks send paths gated on a server "ready"),
        # then stay blocked so the *send* drop is what completes _run
        if self._ready:
            return aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, self._ready.pop(0), "")
        await asyncio.Event().wait()

    async def close(self) -> None:
        pass


class DropSession:
    """Minimal aiohttp.ClientSession stand-in returning a single DropWebSocket."""

    def __init__(self, ws: DropWebSocket, *, closed: bool = False) -> None:
        self._ws = ws
        self.closed = closed

    async def ws_connect(self, *args, **kwargs) -> DropWebSocket:
        return self._ws


def _no_background_task():
    """Neutralize the STT base class's background create_task so nothing runs off-test."""

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    return patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task)


def _frame(sample_rate: int, samples: int = 1920) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


async def run_send_drop(
    build_stream,
    error_factory,
    *,
    sample_rate: int = 16000,
    setup_sends: int = 0,
    session_closed: bool = False,
    ready_messages=(),
    drop_on_bytes: bool = False,
    n_frames: int = 25,
    timeout: float = 5.0,
):
    """Build a stream over a dropping socket, feed audio, run it once.

    Enough frames are pushed to guarantee at least one audio send regardless of how the
    provider buffers. Returns ``(exception_or_None, ws)`` — the exception raised by ``_run()``
    (``None`` if it returned quietly) and the DropWebSocket for inspecting ``audio_sends``.
    """
    ws = DropWebSocket(
        error_factory,
        setup_sends=setup_sends,
        ready_messages=ready_messages,
        drop_on_bytes=drop_on_bytes,
    )
    session = DropSession(ws, closed=session_closed)
    with _no_background_task():
        stream = build_stream(session)
    for _ in range(n_frames):
        stream.push_frame(_frame(sample_rate))
    stream.end_input()
    try:
        await asyncio.wait_for(stream._run(), timeout=timeout)
        return None, ws
    except BaseException as exc:  # noqa: BLE001 — the caller asserts on the exact type
        return exc, ws
