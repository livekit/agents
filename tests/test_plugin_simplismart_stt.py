from __future__ import annotations

import asyncio
import gc
from typing import Any

import aiohttp
import pytest

from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.plugins.simplismart import stt as simplismart_stt

from .stt_reconnect import ReconnectClock, wait_until

pytestmark = pytest.mark.unit


async def _idle_run(self: object) -> None:
    del self
    await asyncio.Event().wait()  # cancelled by aclose()


async def test_aclose_closes_every_stream_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(simplismart_stt.SpeechStream, "_run", _idle_run)

    stt = simplismart_stt.STT(api_key="sk_test")
    closed_first = stt.stream()
    sessions = [closed_first._session, stt.stream()._session]
    await closed_first.aclose()

    await stt.aclose()

    assert all(s.closed for s in sessions)
    assert len(stt._streams) == 0


async def test_aclose_closes_session_of_dropped_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """A finished stream the caller dropped must not take its session to the GC."""

    async def _immediate_run(self: object) -> None:
        del self  # return immediately: the stream task finishes on its own

    monkeypatch.setattr(simplismart_stt.SpeechStream, "_run", _immediate_run)

    stt = simplismart_stt.STT(api_key="sk_test")
    stream = stt.stream()
    session = stream._session

    del stream
    gc.collect()

    await stt.aclose()

    assert session.closed


class _ParkedSocket:
    """A half-open socket: it accepts writes and never delivers anything."""

    def __init__(self) -> None:
        self.closed = False
        self._closed = asyncio.Event()

    async def send_json(self, data: Any) -> None:
        pass

    async def send_bytes(self, data: bytes) -> None:
        pass

    async def receive(self) -> aiohttp.WSMessage:
        await self._closed.wait()
        raise AssertionError("the test should never let recv_task resume")

    async def close(self) -> None:
        self.closed = True
        self._closed.set()


class _HeartbeatTimeoutSocket:
    """A socket after an unanswered ping: ERROR frames, then CLOSED, as aiohttp reports it."""

    def __init__(self) -> None:
        self.closed = False
        self.receives = 0

    async def send_json(self, data: Any) -> None:
        pass

    async def send_bytes(self, data: bytes) -> None:
        pass

    def exception(self) -> BaseException:
        return aiohttp.ServerTimeoutError("No PONG received after 15.0 seconds")

    async def receive(self) -> aiohttp.WSMessage:
        self.receives += 1
        # yield, so a recv loop that steps over the error instead of ending fails the
        # receive-count assertion rather than starving the event loop
        await asyncio.sleep(0)
        if self.receives > 3:
            return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None)
        return aiohttp.WSMessage(aiohttp.WSMsgType.ERROR, self.exception(), None)

    async def close(self) -> None:
        self.closed = True


class _FakeSession:
    """The per-stream aiohttp session: records connect kwargs and hands out sockets."""

    def __init__(self, make_socket) -> None:
        self.closed = False
        self.kwargs: list[dict[str, Any]] = []
        self.sockets: list[Any] = []
        self._make_socket = make_socket

    async def ws_connect(self, url: str, **kwargs: Any) -> Any:
        self.kwargs.append(kwargs)
        ws = self._make_socket()
        self.sockets.append(ws)
        return ws

    async def close(self) -> None:
        self.closed = True


def _stream(session: _FakeSession) -> simplismart_stt.SpeechStream:
    stt = simplismart_stt.STT(api_key="sk_test")
    return simplismart_stt.SpeechStream(
        stt=stt,
        opts=simplismart_stt.SimplismartSTTOptions(language="en"),
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        api_key="sk_test",
        http_session=session,  # type: ignore[arg-type]
    )


async def test_socket_exposes_control_frames_to_monitor() -> None:
    """Control frames must reach the shared monitor to reset its silence deadline."""
    session = _FakeSession(_ParkedSocket)
    stream = _stream(session)
    try:
        await wait_until(lambda: len(session.kwargs) == 1)
        assert session.kwargs[0].get("autoping") is False
    finally:
        await stream.aclose()


async def test_heartbeat_timeout_reconnects_without_spinning() -> None:
    """The ERROR frame itself must end the recv loop and request a reconnect."""
    session = _FakeSession(_HeartbeatTimeoutSocket)
    stream = _stream(session)
    try:
        await wait_until(lambda: len(session.sockets) > 1)
        assert session.sockets[0].receives == 1
    finally:
        await stream.aclose()


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_audio_clock(monkeypatch):
    clock = ReconnectClock(monkeypatch)
    provider = simplismart_stt.STT(api_key="test-key", streaming=True, http_session=clock)
    stream = simplismart_stt.SpeechStream(
        stt=provider,
        opts=provider._opts,
        api_key="test-key",
        http_session=clock,
        conn_options=APIConnectOptions(max_retry=0),
    )
    try:
        await clock.check(stream, lambda: stream._reconnect_event.set())
    finally:
        await stream.aclose()
        await provider.aclose()
