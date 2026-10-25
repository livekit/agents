"""Unit tests for the Inworld STT plugin."""

from __future__ import annotations

import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins.inworld import stt as inworld_stt

from .fake_ws import FakeWSSession, HeartbeatTimeoutSocket, ParkedSocket, wait_until

pytestmark = pytest.mark.unit


def _halfopen_stream(session: FakeWSSession) -> inworld_stt.SpeechStream:
    stt_ = inworld_stt.STT(
        api_key="sk_test",
        base_url="http://localhost:8000/",
        http_session=session,  # type: ignore[arg-type]
    )
    return stt_.stream(conn_options=DEFAULT_API_CONNECT_OPTIONS)


# -- half-open socket detection (#7900) --


@pytest.mark.asyncio
async def test_socket_is_opened_with_a_heartbeat() -> None:
    """aiohttp defaults `heartbeat` to None, so without it the read side of a
    half-open socket parks forever and the reconnect, which only fires when
    `recv_task` ends, never gets a turn."""
    session = FakeWSSession(ParkedSocket)
    stream = _halfopen_stream(session)
    try:
        await wait_until(lambda: len(session.kwargs) == 1)
        assert session.kwargs[0].get("heartbeat") == 30.0
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_heartbeat_timeout_reconnects_without_spinning() -> None:
    """The ERROR frame a missed heartbeat produces has to end the recv loop and
    request a reconnect, instead of being stepped over until aiohttp reports CLOSED."""
    session = FakeWSSession(HeartbeatTimeoutSocket)
    stream = _halfopen_stream(session)
    try:
        await wait_until(lambda: len(session.sockets) > 1)
        assert session.sockets[0].receives == 1
    finally:
        await stream.aclose()
