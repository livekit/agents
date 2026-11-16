"""Tests for the xAI STT plugin: half-open socket detection.

Runs a real `SpeechStream` (its real `_run` loop, driven by `_main_task`) against
fake sockets handed out by a fake aiohttp session, so the connect kwargs and the
reconnect behaviour are assertable without a network. Same shape as the Deepgram
tests added in #7206 and the Telnyx tests in #7359.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp
import pytest

from livekit.agents import APIConnectOptions

pytestmark = pytest.mark.plugin("xai")

# Retry immediately so a reconnect shows up within the test timeout.
_FAST_RETRY = APIConnectOptions(max_retry=3, retry_interval=0.01, timeout=1.0)


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

    async def send_bytes(self, data: bytes) -> None:
        pass

    async def send_str(self, data: str) -> None:
        pass

    async def receive(self) -> aiohttp.WSMessage:
        await self._closed.wait()
        raise AssertionError("the test should never let recv_task resume")

    async def close(self) -> None:
        self.closed = True
        self._closed.set()


class _HeartbeatTimeoutSocket:
    """A socket in the state aiohttp leaves it in when a ping goes unanswered.

    The heartbeat closes the connection itself, so this arrives as WSMsgType.ERROR
    rather than as a close frame, and the reason lives only on `exception()`. After a
    few ERROR frames it reports CLOSED, which is what aiohttp does on the next pass;
    a recv loop that steps over the ERROR only recovers because of that.
    """

    def __init__(self) -> None:
        self.closed = False
        self.receives = 0

    async def send_bytes(self, data: bytes) -> None:
        pass

    async def send_str(self, data: str) -> None:
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
    """Stands in for the aiohttp session: records connect kwargs, hands out sockets."""

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


def _stream(session: _FakeSession):
    from livekit.plugins.xai import STT

    instance = STT(api_key="test-key", http_session=session)  # type: ignore[arg-type]
    return instance.stream(conn_options=_FAST_RETRY)


async def test_socket_exposes_control_frames_to_monitor():
    """Control frames must reach the shared monitor to reset its silence deadline."""
    session = _FakeSession(_ParkedSocket)
    stream = _stream(session)
    try:
        await _wait_until(lambda: len(session.kwargs) == 1)
        assert session.kwargs[0].get("autoping") is False
    finally:
        await stream.aclose()


async def test_heartbeat_timeout_reconnects_without_spinning():
    """The ERROR frame has to end the recv loop and surface as a retryable error.
    Logging it and continuing only reconnected because aiohttp happened to report
    CLOSED next, and it threw away the one value that says why the socket went."""
    session = _FakeSession(_HeartbeatTimeoutSocket)
    stream = _stream(session)
    try:
        await _wait_until(lambda: len(session.sockets) > 1)
        assert session.sockets[0].receives == 1
    finally:
        await stream.aclose()


@pytest.mark.parametrize(
    "endpoint_type",
    ["speech_final", "done", "empty_done", "wordless_done", "wordless_summary_done"],
)
@pytest.mark.parametrize("end_time", [0.4, 0.0])
@pytest.mark.parametrize("prior_timed_chunk", [False, True])
def test_endpoint_preserves_word_timing(endpoint_type, end_time, prior_timed_chunk):
    from types import SimpleNamespace

    from livekit.agents import stt
    from livekit.plugins.xai import stt as xai_stt

    stream = object.__new__(xai_stt.SpeechStream)
    stream._opts = SimpleNamespace(enable_diarization=False, enable_interim_results=True)
    stream._speaking = False
    stream._emitted_chunk_final = False
    stream._final_alternatives = []
    stream._input_ch = SimpleNamespace(start_time_offset=7.0)
    stream._request_id = "test"
    events = []
    stream._event_ch = SimpleNamespace(send_nowait=events.append)
    final = {
        "type": "transcript.partial",
        "text": "hello",
        "is_final": True,
        "words": [{"text": "hello", "start": 0.1, "end": end_time}],
    }
    if prior_timed_chunk:
        stream._process_stream_event(
            {**final, "text": "earlier", "words": [{"text": "earlier", "start": 0.0, "end": 0.2}]}
        )
    stream._process_stream_event(final)
    endpoint = dict(final)
    if endpoint_type == "speech_final":
        endpoint["speech_final"] = True
    else:
        endpoint["type"] = "transcript.done"
        if endpoint_type == "empty_done":
            endpoint["text"] = ""
            endpoint["words"] = []
        elif endpoint_type in ("wordless_done", "wordless_summary_done"):
            endpoint["words"] = []
            if endpoint_type == "wordless_summary_done":
                endpoint["text"] = "earlier hello"
    stream._process_stream_event(endpoint)
    assert events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    if end_time:
        assert events[-1].alternatives[0].end_time == pytest.approx(7.4)
    else:
        assert events[-1].alternatives == []
    stream._process_stream_event({"type": "transcript.partial", "text": "next"})
    stream._process_stream_event({"type": "transcript.done"})
    assert events[-1].alternatives == []


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_audio_clock(monkeypatch):
    from livekit.agents import APIConnectOptions
    from livekit.plugins.xai.stt import STT

    from .stt_reconnect import ReconnectClock

    clock = ReconnectClock(monkeypatch, ready={"type": "transcript.created"})
    provider = STT(api_key="test-key", http_session=clock)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        await clock.check(stream, lambda: stream.update_options(language="fr"))
    finally:
        await stream.aclose()
        await provider.aclose()
