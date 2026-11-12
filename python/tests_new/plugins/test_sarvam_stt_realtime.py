"""Sarvam realtime STT (saaras:v3-realtime) against a local peer speaking its event protocol.

https://github.com/livekit/agents/pull/6562
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TypedDict

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from typing_extensions import Unpack

from livekit import rtc
from livekit.agents import APIStatusError, http, stt
from livekit.plugins import sarvam

Peer = Callable[[web.Request, web.WebSocketResponse], Awaitable[None]]


class RealtimeOptions(TypedDict, total=False):
    stream_type: str
    vad_sot_threshold: float
    vad_min_speech_ms: int
    vad_min_silence_ms: int


@asynccontextmanager
async def _realtime_stt(
    peer: Peer, **options: Unpack[RealtimeOptions]
) -> AsyncIterator[sarvam.STTRealtime]:
    async def handle(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await peer(request, ws)
        return ws

    app = web.Application()
    app.router.add_get("/ws", handle)
    async with TestServer(app) as server, http.open():
        url = str(server.make_url("/ws")).replace("http", "ws", 1)
        model = sarvam.STTRealtime(api_key="sk_test", language="hi-IN", base_url=url, **options)
        try:
            yield model
        finally:
            await model.aclose()


def _audio(seconds: float) -> rtc.AudioFrame:
    samples = int(16000 * seconds)
    return rtc.AudioFrame(b"\x01\x00" * samples, 16000, 1, samples)


async def test_the_connection_url_carries_the_session_and_vad_options() -> None:
    queries: asyncio.Queue[dict[str, str]] = asyncio.Queue()

    async def peer(request: web.Request, ws: web.WebSocketResponse) -> None:
        queries.put_nowait(dict(request.query))
        async for _ in ws:
            pass

    async with _realtime_stt(
        peer,
        stream_type="fast",
        vad_sot_threshold=0.4,
        vad_min_speech_ms=300,
        vad_min_silence_ms=800,
    ) as model:
        async with model.stream():
            query = await queries.get()

    assert query == {
        "language_code": "hi-IN",
        "stream_type": "fast",
        "endpointing": "vad",
        "encoding": "linear16",
        "sample_rate": "16000",
        "model": "saaras:v3-realtime",
        "mode": "transcribe",
        "return_timestamps": "false",
        "threshold": "0.4",
        "min_speech_duration_ms": "300",
        "silence_duration_ms": "800",
    }


async def test_the_final_is_held_until_speech_ends_and_lands_at_the_speech_end() -> None:
    """START, INTERIM, then END_OF_SPEECH and the FINAL ending where the audio stopped."""

    async def peer(request: web.Request, ws: web.WebSocketResponse) -> None:
        request_id = "20260608_31c9dc1d-3435-4e76-ae51-05de31025a68"
        await ws.send_json({"event": "session.begin", "session_id": "s", "request_id": request_id})
        await ws.send_json({"event": "vad.speech_start", "utterance_idx": 0})
        await ws.send_json(
            {"event": "transcript.partial", "utterance_idx": 0, "text": "नमस्ते", "confidence": 0.9}
        )
        await ws.send_json(
            {
                "event": "transcript.final",
                "utterance_idx": 0,
                "text": "नमस्ते आप कैसे हैं",
                "language": "hi-IN",
                "language_confidence": 0.99,
            }
        )
        audio_bytes = 0
        async for message in ws:
            if message.type == web.WSMsgType.BINARY:
                audio_bytes += len(message.data)
            if audio_bytes >= 2 * 16000:  # one second heard
                await ws.send_json({"event": "vad.speech_end", "utterance_idx": 0})
                audio_bytes = -1_000_000

    async with _realtime_stt(peer) as model:
        async with model.stream() as stream:
            stream.push_frame(_audio(1.0))
            events = []
            async for event in stream:
                if event.type != stt.SpeechEventType.RECOGNITION_USAGE:
                    events.append(event)
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    break

    assert [e.type for e in events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
    ]
    assert events[2].alternatives == []
    final = events[3]
    assert final.request_id == "20260608_31c9dc1d-3435-4e76-ae51-05de31025a68"
    data = final.alternatives[0]
    assert (data.text, data.language) == ("नमस्ते आप कैसे हैं", "hi-IN")
    # no recognition confidence on the wire: 1.0, the language score stays in metadata
    assert data.confidence == 1.0
    assert data.metadata is not None
    assert data.metadata["language_confidence"] == 0.99
    speech_end = data.metadata["speech_end_wall_time"]
    assert isinstance(speech_end, float) and speech_end > 0
    assert data.end_time == pytest.approx(1.0)


async def test_a_non_fatal_error_is_survived_and_a_fatal_one_raises_its_status() -> None:
    async def peer(request: web.Request, ws: web.WebSocketResponse) -> None:
        await ws.send_json(
            {
                "event": "error",
                "code": "chunk_too_large",
                "message": "chunk too large",
                "is_fatal": False,
            }
        )
        await ws.send_json({"event": "vad.speech_start", "utterance_idx": 0})
        await ws.send_json(
            {
                "event": "error",
                "code": "model_unavailable",
                "message": "backend saturated",
                "is_fatal": True,
                "status_code": 503,
            }
        )
        async for _ in ws:
            pass

    async with _realtime_stt(peer) as model:
        events = []
        with pytest.raises(APIStatusError) as caught:
            async with model.stream() as stream:
                async for event in stream:
                    events.append(event.type)

    assert stt.SpeechEventType.START_OF_SPEECH in events
    assert caught.value.status_code == 503
    assert isinstance(caught.value.body, dict)
    assert caught.value.body["code"] == "model_unavailable"
    assert caught.value.retryable is True
