"""Smallest AI STT plugin: speech events at the public streaming boundary."""

import asyncio
import struct

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, http, stt
from livekit.plugins.smallestai import STT, TTS


async def test_malformed_frame_does_not_drop_following_transcript_and_speech_boundaries() -> None:
    """A malformed frame is skipped; the transcript and speech boundaries after it arrive."""

    async def recognize(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.send_str("{")
        await socket.send_json({"type": "speech_started"})
        await socket.send_json(
            {
                "session_id": "session",
                "transcript": "Hello",
                "is_final": True,
                "words": [{"word": "Hello", "start": 0.1, "end": 0.5, "confidence": 0.9}],
                "utterances": [{"start": 0.1, "end": 0.5, "text": "Hello"}],
            }
        )
        await socket.send_json({"type": "speech_ended"})
        await socket.send_json({"is_last": True})
        async for _ in socket:
            pass
        return socket

    app = web.Application()
    app.router.add_get("/stt/live", recognize)
    async with TestServer(app) as server, http.open() as session:
        provider = STT(api_key="key", base_url=str(server.make_url("")), http_session=session)
        try:
            async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                stream.end_input()
                async with asyncio.timeout(1):
                    events = [event async for event in stream]
        finally:
            await provider.aclose()
    speech_events = [
        event for event in events if event.type != stt.SpeechEventType.RECOGNITION_USAGE
    ]
    assert [event.type for event in speech_events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    transcript = speech_events[1].alternatives[0]
    assert transcript.text == "Hello"
    assert transcript.words is not None and [word.text for word in transcript.words] == ["Hello"]
    assert transcript.metadata == {
        "utterances": [
            {
                "start": transcript.words[0].start_time,
                "end": transcript.words[0].end_time,
                "text": "Hello",
            }
        ]
    }


async def test_finalize_and_close_stream_deliver_last_transcript() -> None:
    """Finalization preserves speech boundaries and closes with the final transcript. https://github.com/livekit/agents/pull/5562 https://github.com/livekit/agents/pull/7379 https://github.com/livekit/agents/pull/5953"""
    import json

    from livekit import rtc

    commands = []
    query = []

    async def recognize(request: web.Request) -> web.StreamResponse:
        query.append(dict(request.query))
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        async for msg in socket:
            if msg.type != web.WSMsgType.TEXT:
                continue
            command = json.loads(msg.data)["type"]
            commands.append(command)
            if command == "finalize" and commands.count("finalize") == 1:
                await socket.send_json({"type": "speech_started"})
                await socket.send_json({"transcript": "Hello", "is_final": True})
                await socket.send_json({"transcript": "...", "is_final": True})
                await socket.send_json({"transcript": "", "is_final": True, "from_finalize": True})
            elif command == "close_stream":
                await socket.send_json({"transcript": "world", "is_final": True, "is_last": True})
        return socket

    app = web.Application()
    app.router.add_get("/stt/live", recognize)
    async with TestServer(app) as server, http.open() as session:
        provider = STT(
            api_key="key",
            language="es",
            sample_rate=24000,
            base_url=str(server.make_url("")),
            http_session=session,
            vad_events=True,
            diarize=True,
            keywords=[("LiveKit", 2)],
        )
        try:
            async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                stream.push_frame(
                    rtc.AudioFrame(
                        data=bytes(2400),
                        sample_rate=24000,
                        num_channels=1,
                        samples_per_channel=1200,
                    )
                )
                stream.flush()
                stream.end_input()
                async with asyncio.timeout(2):
                    events = [ev async for ev in stream]
        finally:
            await provider.aclose()
    assert commands[-1] == "close_stream"
    assert set(commands[:-1]) == {"finalize"}
    finals = [
        ev.alternatives[0].text for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ]
    assert finals == ["Hello", "world"]
    assert sum(ev.type == stt.SpeechEventType.START_OF_SPEECH for ev in events) == 2
    assert sum(ev.type == stt.SpeechEventType.END_OF_SPEECH for ev in events) == 2
    assert query[0]["language"] == "es" and query[0]["sample_rate"] == "24000"
    assert query[0]["diarize"] == "true" and query[0]["keywords"] == "LiveKit:2"


@pytest.mark.parametrize(
    ("output_format", "encoded", "sample"), [("ulaw", 0x00, -32124), ("alaw", 0x55, -8)]
)
async def test_g711_synthesis_decodes_to_pcm(output_format: str, encoded: int, sample: int) -> None:
    """The ulaw/alaw output formats play as PCM instead of failing the decoder."""

    async def synthesize(request: web.Request) -> web.Response:
        return web.Response(body=bytes([encoded]) * 800, content_type="audio/basic")

    app = web.Application()
    app.router.add_post("/tts", synthesize)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(
            api_key="key",
            sample_rate=8000,
            output_format=output_format,
            base_url=str(server.make_url("")).rstrip("/"),
            http_session=session,
        )
        async with provider.synthesize("Hello.", conn_options=APIConnectOptions(max_retry=0)) as s:
            pcm = b"".join([event.frame.data.tobytes() async for event in s])
        await provider.aclose()

    assert pcm[: 800 * 2] == struct.pack("<h", sample) * 800
