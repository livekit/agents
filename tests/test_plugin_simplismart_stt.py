from __future__ import annotations

import asyncio
import gc
import time
from typing import Any

import pytest

from livekit.agents import APIConnectionError, http
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins.simplismart import stt as simplismart_stt

pytestmark = pytest.mark.unit


async def _idle_run(self: object) -> None:
    del self
    await asyncio.Event().wait()  # cancelled by aclose()


class _RecordingSession(http.HttpSession):
    closed = False

    async def aclose(self) -> None:
        self.closed = True
        await super().aclose()


@pytest.fixture(autouse=True)
def _recording_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(simplismart_stt.http, "HttpSession", _RecordingSession)


@pytest.mark.asyncio
async def test_simplismart_stt_aclose_closes_tracked_stream_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simplismart_stt.SpeechStream, "_run", _idle_run)

    stt = simplismart_stt.STT(api_key="sk_test")
    stream_a = stt.stream()
    stream_b = stt.stream()
    sessions = [stream_a._session, stream_b._session]

    assert all(not s.closed for s in sessions)
    assert len(stt._streams) == 2

    await stt.aclose()

    assert all(s.closed for s in sessions), "STT.aclose() must close every per-stream HttpSession"
    assert len(stt._streams) == 0


@pytest.mark.asyncio
async def test_simplismart_stt_async_context_closes_stream_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simplismart_stt.SpeechStream, "_run", _idle_run)

    async with simplismart_stt.STT(api_key="sk_test") as stt:
        stream = stt.stream()
        session = stream._session
        assert not session.closed

    assert session.closed, "exiting `async with` must close per-stream sessions"


@pytest.mark.asyncio
async def test_simplismart_stt_aclose_tolerates_already_closed_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simplismart_stt.SpeechStream, "_run", _idle_run)

    stt = simplismart_stt.STT(api_key="sk_test")
    stream = stt.stream()
    session = stream._session

    await stream.aclose()
    await stt.aclose()  # must not raise on an already-closed stream

    assert session.closed


@pytest.mark.asyncio
async def test_simplismart_stt_aclose_closes_session_of_dropped_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished stream the caller dropped must not take its session to the GC.

    With weak tracking, a completed SpeechStream can be collected before
    STT.aclose() snapshots the set, leaving its per-stream HttpSession
    unreachable and unclosed. Strong ownership keeps it reachable until
    aclose closes it.
    """

    async def _immediate_run(self: object) -> None:
        del self  # return immediately: the stream task finishes on its own

    monkeypatch.setattr(simplismart_stt.SpeechStream, "_run", _immediate_run)

    stt = simplismart_stt.STT(api_key="sk_test")
    stream = stt.stream()
    session = stream._session

    del stream
    gc.collect()

    assert len(stt._streams) == 1, "a dropped stream must stay tracked until aclose"

    await stt.aclose()

    assert session.closed, "aclose() must close the session of a dropped stream"
    assert len(stt._streams) == 0


# ---------------------------------------------------------------------------
# Half-open socket detection (same fix as #7206 for Deepgram v1)
# ---------------------------------------------------------------------------


async def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for the stream"
        await asyncio.sleep(0.01)


class _ParkedSocket:
    """A socket that accepts writes and never delivers anything: the half-open case
    from the client's point of view. `receive()` only returns once the socket is
    closed, and the test never lets that happen."""

    def __init__(self) -> None:
        self.closed = False
        self._closed = asyncio.Event()

    async def send_json(self, data: Any) -> None:
        pass

    async def send_bytes(self, data: bytes) -> None:
        pass

    async def receive(self) -> http.WebSocketMessage:
        await self._closed.wait()
        raise AssertionError("the test should never let recv_task resume")

    async def close(self) -> None:
        self.closed = True
        self._closed.set()


class _HeartbeatTimeoutSocket:
    """A socket whose heartbeat went unanswered.

    The heartbeat closes the connection itself, so `receive()` raises
    APIConnectionError rather than returning a close frame. After a few failures it
    reports CLOSE; a recv loop that steps over the error only recovers because of that.
    """

    def __init__(self) -> None:
        self.closed = False
        self.receives = 0

    async def send_json(self, data: Any) -> None:
        pass

    async def send_bytes(self, data: bytes) -> None:
        pass

    async def receive(self) -> http.WebSocketMessage:
        self.receives += 1
        # yield, so a recv loop that steps over the error instead of ending fails the
        # receive-count assertion rather than starving the event loop
        await asyncio.sleep(0)
        if self.receives > 3:
            return http.WebSocketMessage(http.WebSocketMessageType.CLOSE, "")
        raise APIConnectionError("No PONG received after 15.0 seconds")

    async def close(self) -> None:
        self.closed = True


class _FakeSession:
    """Stands in for the per-stream HttpSession: records connect kwargs, hands
    out sockets, and can be closed by `SpeechStream.aclose`."""

    def __init__(self, make_socket) -> None:
        self.closed = False
        self.kwargs: list[dict[str, Any]] = []
        self.sockets: list[Any] = []
        self._make_socket = make_socket

    async def ws_connect(self, url: str, *, conn_options: Any, **kwargs: Any) -> Any:
        self.kwargs.append(kwargs)
        ws = self._make_socket()
        self.sockets.append(ws)
        return ws

    async def aclose(self) -> None:
        self.closed = True


def _stream(session: _FakeSession) -> simplismart_stt.SpeechStream:
    stt = simplismart_stt.STT(api_key="sk_test")
    return simplismart_stt.SpeechStream(
        stt=stt,
        opts=simplismart_stt.SimplismartSTTOptions(language="en"),
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        api_key="sk_test",
        http_session=session,  # type: ignore[arg-type]
        base_url=stt._base_url,
        on_closed=stt._streams.discard,
    )


@pytest.mark.asyncio
async def test_socket_is_opened_with_a_heartbeat() -> None:
    """`heartbeat` defaults to None, so without it the read side of a
    half-open socket parks forever and the reconnect in `_run`, which only fires
    when `recv_task` ends, never gets a turn."""
    session = _FakeSession(_ParkedSocket)
    stream = _stream(session)
    try:
        await _wait_until(lambda: len(session.kwargs) == 1)
        assert session.kwargs[0].get("heartbeat") == 30.0
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_heartbeat_timeout_reconnects_without_spinning() -> None:
    """The receive failure has to end the recv loop and request a reconnect.
    Continuing past it only reconnects because the socket happens to report CLOSE
    next."""
    session = _FakeSession(_HeartbeatTimeoutSocket)
    stream = _stream(session)
    try:
        await _wait_until(lambda: len(session.sockets) > 1)
        assert session.sockets[0].receives == 1
    finally:
        await stream.aclose()
