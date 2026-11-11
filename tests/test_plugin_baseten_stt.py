from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Literal

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, APITimeoutError, stt
from livekit.plugins.baseten import STT

from .stt_reconnect import FakeWebSocket, ReconnectClock

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_transcript_timing(monkeypatch):
    clock = ReconnectClock(monkeypatch)
    provider = STT(api_key="test-key", model_endpoint="wss://test.invalid/stt", http_session=clock)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        await clock.check(stream, lambda: stream.update_options(language="fr"))
        clock.sockets[-1].respond(
            {
                "is_final": True,
                "segments": [{"text": "hello", "start_time": 0.1, "end_time": 0.4}],
            }
        )
        async with asyncio.timeout(2):
            async for event in stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    assert event.alternatives[0].end_time == pytest.approx(17.4)
                    break
            else:
                pytest.fail("No FINAL_TRANSCRIPT from the restarted stream")
    finally:
        await stream.aclose()
        await provider.aclose()


_Mode = Literal["reply", "slow", "ignore", "drop", "stall"]


class _WhisperSocket(FakeWebSocket):
    """Answers end_audio the way a Baseten streaming Whisper runtime does in each mode."""

    def __init__(self, mode: _Mode) -> None:
        super().__init__()
        self.mode = mode

    async def ping(self) -> None:
        self.incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.PONG, b"", ""))

    async def send_str(self, data: str) -> None:
        self.sent.append(data)
        if json.loads(data).get("type") == "end_audio" and self.mode != "ignore":
            asyncio.get_running_loop().create_task(self._finish_flush())

    async def _finish_flush(self) -> None:
        self.respond({"type": "end_audio", "body": {"status": "acknowledged"}})
        if self.mode == "drop":
            await self.close()
            return
        if self.mode == "stall":
            return
        if self.mode == "slow":
            await asyncio.sleep(5.5)
        self.respond(
            {
                "type": "transcription",
                "is_final": True,
                "segments": [{"text": "trailing words", "start_time": 0.0, "end_time": 1.0}],
            }
        )
        self.respond({"type": "end_audio", "body": {"status": "finished"}})


async def _run_stream(
    sockets: list[_WhisperSocket],
    mode: _Mode,
    samples: int = 512,
    timeout: float = 10.0,
    max_retry: int = 0,
) -> list[stt.SpeechEvent]:
    def ws_connect(*args: Any, **kwargs: Any) -> _WhisperSocket:
        sockets.append(_WhisperSocket(mode))
        return sockets[-1]

    session = SimpleNamespace(closed=False, ws_connect=ws_connect)
    provider = STT(api_key="test-key", model_endpoint="wss://test.invalid/ws", http_session=session)
    stream = provider.stream(
        conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0.01, timeout=timeout)
    )
    stream.push_frame(rtc.AudioFrame.create(16000, 1, samples))
    stream.end_input()
    try:
        async with asyncio.timeout(15):
            return [ev async for ev in stream]
    finally:
        await stream.aclose()


def _final_texts(events: list[stt.SpeechEvent]) -> list[str]:
    return [
        ev.alternatives[0].text for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ]


async def test_end_input_flushes_buffered_audio() -> None:
    sockets: list[_WhisperSocket] = []
    # 1.5 buffers: the trailing 256 samples only go out when the input ends
    events = await _run_stream(sockets, "reply", samples=768)

    (ws,) = sockets
    assert sum(len(m) for m in ws.sent if isinstance(m, bytes)) == 768 * 2
    # after the metadata, end_audio is the only text message
    assert [json.loads(m) for m in ws.sent if isinstance(m, str)][1:] == [{"type": "end_audio"}]
    assert _final_texts(events) == ["trailing words"]


async def test_slow_flush_still_delivers_the_final_transcript() -> None:
    events = await _run_stream([], "slow")
    assert _final_texts(events) == ["trailing words"]


async def test_end_input_completes_when_server_ignores_end_audio() -> None:
    # runtimes without end_audio support never answer; the stream still has to finish
    events = await _run_stream([], "ignore")
    assert _final_texts(events) == []


async def test_close_before_finished_is_an_error_and_not_retried() -> None:
    # a retry would reconnect with the input already drained and flush nothing
    sockets: list[_WhisperSocket] = []
    with pytest.raises(APIStatusError):
        await _run_stream(sockets, "drop", max_retry=3)
    assert len(sockets) == 1


async def test_acknowledged_flush_that_never_finishes_times_out() -> None:
    sockets: list[_WhisperSocket] = []
    with pytest.raises(APITimeoutError):
        await _run_stream(sockets, "stall", timeout=1.0, max_retry=3)
    assert len(sockets) == 1
