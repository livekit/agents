"""Simplismart STT: stream sessions are closed and a half-open socket is noticed.

Streaming always dials ``wss://<host>/ws/audio``, so these tests stand in for the per-stream
``http.HttpSession`` the plugin opens rather than for a server.
"""

from __future__ import annotations

import asyncio
import gc

import pytest

from livekit.agents import APIConnectionError, http
from livekit.plugins.simplismart import STT
from tests.fakes import until


class _Socket:
    """A socket that either stays silent (half-open) or fails each read like a missed PONG."""

    def __init__(self, *, pong_missed: bool) -> None:
        self.pong_missed = pong_missed
        self.receives = 0
        self.close_requested = asyncio.Event()

    async def send_json(self, data: object) -> None:
        pass

    async def send_bytes(self, data: bytes) -> None:
        pass

    async def receive(self) -> http.WebSocketMessage:
        self.receives += 1
        await asyncio.sleep(0)
        if self.pong_missed and self.receives <= 3:
            raise APIConnectionError("No PONG received after 15.0 seconds")
        await self.close_requested.wait()
        return http.WebSocketMessage(http.WebSocketMessageType.CLOSE, "")

    async def close(self) -> None:
        self.close_requested.set()


class _Session:
    """The per-stream session: records each connect and hands out sockets."""

    pong_missed = False

    def __init__(self) -> None:
        self.closed = False
        self.connects: list[dict[str, object]] = []
        self.sockets: list[_Socket] = []

    async def ws_connect(self, url: str, **kwargs: object) -> _Socket:
        self.connects.append(kwargs)
        self.sockets.append(_Socket(pong_missed=self.pong_missed))
        return self.sockets[-1]

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> list[_Session]:
    created: list[_Session] = []

    def make() -> _Session:
        created.append(_Session())
        return created[-1]

    monkeypatch.setattr(http, "HttpSession", make)
    return created


async def test_stt_aclose_closes_every_stream_session(sessions: list[_Session]) -> None:
    """``STT.aclose()`` closes the session of each open stream. https://github.com/livekit/agents/pull/7012"""
    stt = STT(api_key="sk_test")
    stt.stream()
    stt.stream()

    await stt.aclose()

    assert len(sessions) == 2
    assert all(session.closed for session in sessions)


async def test_stt_aclose_closes_the_session_of_a_dropped_stream(sessions: list[_Session]) -> None:
    """A stream the caller dropped still has its session closed. https://github.com/livekit/agents/pull/7012"""
    stt = STT(api_key="sk_test")
    stt.stream()
    gc.collect()

    await stt.aclose()

    [session] = sessions
    assert session.closed


async def test_socket_is_opened_with_a_heartbeat(sessions: list[_Session]) -> None:
    """The socket pings, so a half-open connection is noticed. https://github.com/livekit/agents/issues/7368"""
    stt = STT(api_key="sk_test")
    stt.stream()

    await until(lambda: sessions and sessions[0].connects, "the stream to connect")
    await stt.aclose()

    assert sessions[0].connects[0]["heartbeat"] == 30.0


async def test_missed_heartbeat_reconnects_without_spinning(
    sessions: list[_Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missed PONG ends the read loop and reconnects. https://github.com/livekit/agents/issues/7368"""
    monkeypatch.setattr(_Session, "pong_missed", True)
    stt = STT(api_key="sk_test")
    stt.stream()

    await until(lambda: sessions and len(sessions[0].sockets) > 1, "a reconnect")
    await stt.aclose()

    assert sessions[0].sockets[0].receives == 1
