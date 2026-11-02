from __future__ import annotations

import asyncio
import json
import struct
import time
from unittest.mock import AsyncMock

import aiohttp
import numpy as np
import pytest

from livekit import rtc
from livekit.agents.inference import OverlappingSpeechEvent
from livekit.agents.inference.interruption import (
    AdaptiveInterruptionDetector,
    InterruptionWebSocketStream,
    _AgentSpeechEndedSentinel,
    _AgentSpeechStartedSentinel,
    _OverlapSpeechEndedSentinel,
    _OverlapSpeechStartedSentinel,
)
from livekit.agents.types import APIConnectOptions

pytestmark = pytest.mark.unit

CONN_OPTIONS = APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=1.0)


class _FakeWebSocket:
    def __init__(self) -> None:
        self.closed = False
        self.close_code: int | None = None
        self.sent_audio: asyncio.Queue[bytes] = asyncio.Queue()
        self.release_audio_send = asyncio.Event()
        self.release_audio_send.set()
        self.session_close_sent = asyncio.Event()
        self._received: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()

    async def send_str(self, data: str) -> None:
        message_type = json.loads(data)["type"]
        if message_type == "session.create":
            self.send_json({"type": "session.created", "default_threshold": 0.5})
        elif message_type == "session.close":
            self.session_close_sent.set()

    async def send_bytes(self, data: bytes) -> None:
        self.sent_audio.put_nowait(data)
        await self.release_audio_send.wait()

    async def receive(self) -> aiohttp.WSMessage:
        return await self._received.get()

    async def close(self) -> bool:
        self.closed = True
        return True

    def send_json(self, data: dict[str, object]) -> None:
        self._received.put_nowait(
            aiohttp.WSMessage(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(data),
                extra=None,
            )
        )


def _make_audio_frame() -> rtc.AudioFrame:
    samples = np.zeros(1600, dtype=np.int16)
    return rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=len(samples),
    )


async def _start_agent_overlap(stream: InterruptionWebSocketStream, ws: _FakeWebSocket) -> int:
    stream.push_frame(_AgentSpeechStartedSentinel())
    stream.push_frame(_OverlapSpeechStartedSentinel(speech_duration=0.0, started_at=time.time()))
    stream.push_frame(_make_audio_frame())

    payload = await asyncio.wait_for(ws.sent_audio.get(), timeout=1.0)
    created_at = struct.unpack_from("<Q", payload)[0]

    async def _request_is_cached() -> None:
        while created_at not in stream._cache:
            await asyncio.sleep(0)

    await asyncio.wait_for(_request_is_cached(), timeout=1.0)
    return created_at


@pytest.mark.asyncio
async def test_late_verdict_from_previous_agent_speech_is_ignored() -> None:
    ws = _FakeWebSocket()
    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.closed = False
    mock_session.ws_connect = AsyncMock(return_value=ws)
    detector = AdaptiveInterruptionDetector(
        base_url="http://localhost:9999",
        api_key="test-key",
        api_secret="test-secret",
        http_session=mock_session,
    )
    events: list[OverlappingSpeechEvent] = []
    verdict_received = asyncio.Event()

    def _on_overlap_speech(event: OverlappingSpeechEvent) -> None:
        events.append(event)
        verdict_received.set()

    detector.on("overlapping_speech", _on_overlap_speech)
    stream = detector.stream(conn_options=CONN_OPTIONS)

    try:
        old_request_id = await _start_agent_overlap(stream, ws)
        stream.push_frame(_OverlapSpeechEndedSentinel(ended_at=time.time(), agent_ended=True))
        stream.push_frame(_AgentSpeechEndedSentinel())

        new_request_id = await _start_agent_overlap(stream, ws)
        events.clear()
        verdict_received.clear()

        ws.send_json(
            {
                "type": "bargein_detected",
                "created_at": old_request_id,
                "probabilities": [0.99, 0.99],
            }
        )
        ws.send_json(
            {
                "type": "bargein_detected",
                "created_at": new_request_id,
                "probabilities": [0.75, 0.75],
            }
        )

        await asyncio.wait_for(verdict_received.wait(), timeout=1.0)

        assert len(events) == 1
        assert events[0].probability == pytest.approx(0.75)
        assert events[0].speech_input is not None
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_agent_speech_end_invalidates_a_blocked_send() -> None:
    ws = _FakeWebSocket()
    ws.release_audio_send.clear()
    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.closed = False
    mock_session.ws_connect = AsyncMock(return_value=ws)
    detector = AdaptiveInterruptionDetector(
        base_url="http://localhost:9999",
        api_key="test-key",
        api_secret="test-secret",
        http_session=mock_session,
    )
    stream = detector.stream(conn_options=CONN_OPTIONS)

    try:
        stream.push_frame(_AgentSpeechStartedSentinel())
        stream.push_frame(
            _OverlapSpeechStartedSentinel(speech_duration=0.0, started_at=time.time())
        )
        stream.push_frame(_make_audio_frame())
        payload = await asyncio.wait_for(ws.sent_audio.get(), timeout=1.0)
        request_id = struct.unpack_from("<Q", payload)[0]

        stream.push_frame(_OverlapSpeechEndedSentinel(ended_at=time.time(), agent_ended=True))
        stream.push_frame(_AgentSpeechEndedSentinel())
        stream.end_input()

        async def _agent_speech_ended() -> None:
            while stream._agent_speech_started:
                await asyncio.sleep(0)

        await asyncio.wait_for(_agent_speech_ended(), timeout=1.0)
        ws.release_audio_send.set()
        await asyncio.wait_for(ws.session_close_sent.wait(), timeout=1.0)

        assert request_id not in stream._cache
    finally:
        await stream.aclose()
