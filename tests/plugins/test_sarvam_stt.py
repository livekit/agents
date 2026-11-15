"""Sarvam STT against a local peer: realtime streaming (saaras:v3-realtime, its event protocol)
and batch recognition over HTTP.

https://github.com/livekit/agents/pull/6562
"""

from __future__ import annotations

import asyncio
import json
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
from tests.fakes import until

Peer = Callable[[web.Request, web.WebSocketResponse], Awaitable[None]]


class RealtimeOptions(TypedDict, total=False):
    stream_type: str
    endpointing: str
    vad_sot_threshold: float
    vad_min_speech_ms: int
    vad_min_silence_ms: int


@asynccontextmanager
async def _realtime_stt(
    peer: Peer, **options: Unpack[RealtimeOptions]
) -> AsyncIterator[sarvam.STT]:
    async def handle(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await peer(request, ws)
        return ws

    app = web.Application()
    app.router.add_get("/ws", handle)
    async with TestServer(app) as server, http.open():
        url = str(server.make_url("/ws")).replace("http", "ws", 1)
        model = sarvam.STT(api_key="sk_test", language="hi-IN", base_url=url, **options)
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


async def test_sarvam_vad_bounds_the_speech_and_its_final_keeps_the_request_and_scores() -> None:
    """START on vad.speech_start, the final before END on vad.speech_end, whichever came first.

    https://github.com/livekit/agents/issues/5829
    """

    async def peer(request: web.Request, ws: web.WebSocketResponse) -> None:
        request_id = "20260608_31c9dc1d-3435-4e76-ae51-05de31025a68"
        await ws.send_json({"event": "session.begin", "session_id": "s", "request_id": request_id})
        await ws.send_json({"event": "vad.speech_start", "utterance_idx": 0})
        await ws.send_json(
            {"event": "transcript.partial", "utterance_idx": 0, "text": "नमस्ते", "confidence": 0.9}
        )
        await ws.send_json({"event": "vad.speech_end", "utterance_idx": 0})
        await ws.send_json(
            {
                "event": "transcript.final",
                "utterance_idx": 0,
                "text": "नमस्ते आप कैसे हैं",
                "language": "hi-IN",
                "language_confidence": 0.99,
            }
        )
        async for _ in ws:
            pass

    async with _realtime_stt(peer) as model:
        async with model.stream() as stream:
            stream.push_frame(_audio(0.1))
            events = []
            async for event in stream:
                if event.type != stt.SpeechEventType.RECOGNITION_USAGE:
                    events.append(event)
                if event.type == stt.SpeechEventType.END_OF_SPEECH:
                    break

    assert [e.type for e in events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH_PENDING,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    final = events[3]
    assert final.request_id == "20260608_31c9dc1d-3435-4e76-ae51-05de31025a68"
    data = final.alternatives[0]
    assert (data.text, data.language) == ("नमस्ते आप कैसे हैं", "hi-IN")
    # no recognition confidence on the wire: 1.0, the language score stays in metadata
    assert data.confidence == 1.0
    assert data.metadata == {"utterance_idx": 0, "language_confidence": 0.99}


async def test_a_vad_speech_sarvam_sends_no_words_for_ends_at_its_speech_end() -> None:
    """Sarvam documents no final for a speech its VAD ended without words (noise, or a
    ``simulated`` stream before its final): END_OF_SPEECH comes on ``vad.speech_end`` instead of
    waiting for a final that may never come."""

    async def peer(request: web.Request, ws: web.WebSocketResponse) -> None:
        await ws.send_json({"event": "session.begin", "session_id": "s", "request_id": "r"})
        await ws.send_json({"event": "vad.speech_start", "utterance_idx": 0})
        await ws.send_json({"event": "vad.speech_end", "utterance_idx": 0})
        async for _ in ws:
            pass

    async with _realtime_stt(peer) as model:
        async with model.stream() as stream:
            stream.push_frame(_audio(0.1))
            events: list[stt.SpeechEventType] = []

            async def collect() -> None:
                async for event in stream:
                    if event.type != stt.SpeechEventType.RECOGNITION_USAGE:
                        events.append(event.type)
                    if event.type == stt.SpeechEventType.END_OF_SPEECH:
                        return

            collecting = asyncio.create_task(collect())
            try:
                await until(collecting.done, "END_OF_SPEECH on vad.speech_end")
            finally:
                collecting.cancel()

    assert events == [stt.SpeechEventType.START_OF_SPEECH, stt.SpeechEventType.END_OF_SPEECH]


async def test_manual_endpointing_brackets_each_flushed_segment() -> None:
    """With manual endpointing each segment's audio goes between speech_start and speech_end,
    and the audio after a flush opens the next one on the same connection."""
    # each message's event, consecutive audio frames as one "audio"
    seen: list[str] = []

    async def peer(request: web.Request, ws: web.WebSocketResponse) -> None:
        async for message in ws:
            kind = (
                json.loads(message.data)["event"] if message.type == web.WSMsgType.TEXT else "audio"
            )
            if not seen or seen[-1] != kind:
                seen.append(kind)

    async with _realtime_stt(peer, endpointing="manual") as model:
        async with model.stream() as stream:
            stream.push_frame(_audio(0.05))
            stream.flush()
            stream.push_frame(_audio(0.05))
            stream.end_input()
            await until(lambda: "end" in seen, "the end of the audio")

    # end_input() flushes the last segment before the end
    assert seen == [
        "speech_start",
        "audio",
        "speech_end",
        "speech_start",
        "audio",
        "speech_end",
        "end",
    ]


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
                stream.push_frame(_audio(0.05))
                async for event in stream:
                    events.append(event.type)

    assert stt.SpeechEventType.START_OF_SPEECH in events
    assert caught.value.status_code == 503
    assert isinstance(caught.value.body, dict)
    assert caught.value.body["code"] == "model_unavailable"
    assert caught.value.retryable is True


async def test_closing_the_stt_closes_every_open_stream() -> None:
    """https://github.com/livekit/agents/pull/7058"""
    heard: asyncio.Queue[None] = asyncio.Queue()
    closed: asyncio.Queue[None] = asyncio.Queue()

    async def peer(request: web.Request, ws: web.WebSocketResponse) -> None:
        async for _ in ws:
            heard.put_nowait(None)
        closed.put_nowait(None)

    async with _realtime_stt(peer) as model:
        for language in ("en-IN", "hi-IN"):
            model.stream(language=language).push_frame(_audio(0.05))
            await heard.get()  # connected and streaming

        await model.aclose()

        await closed.get()
        await closed.get()


def test_a_realtime_option_with_a_batch_model_is_refused() -> None:
    with pytest.raises(ValueError, match="stream_type"):
        sarvam.STT(api_key="sk_test", model="saaras:v4", stream_type="fast")


@pytest.mark.parametrize("as_list", [False, True])
async def test_batch_recognition_posts_to_base_url_and_ends_at_the_audio_duration(
    as_list: bool,
) -> None:
    """A batch recognition goes to the ``base_url`` the STT was built with; a transcript
    without timestamps ends where the audio does."""
    paths: list[str] = []

    async def recognize(request: web.Request) -> web.Response:
        paths.append(request.path)
        await request.read()
        return web.json_response({"transcript": "hello", "language_code": "en-IN"})

    app = web.Application()
    app.router.add_post("/custom/speech-to-text", recognize)
    async with TestServer(app) as server, http.open():
        model = sarvam.STT(
            api_key="test",
            model="saaras:v3",
            base_url=str(server.make_url("/custom/speech-to-text")),
        )
        frame = rtc.AudioFrame.create(sample_rate=16000, num_channels=1, samples_per_channel=1600)
        result = await model.recognize([frame] if as_list else frame)
        await model.aclose()

    assert paths == ["/custom/speech-to-text"]
    assert result.alternatives[0].end_time == pytest.approx(0.1)
