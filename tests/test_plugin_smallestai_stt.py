"""Smallest AI speech events at the public streaming boundary."""

import asyncio

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, stt
from livekit.plugins.smallestai import STT

pytestmark = pytest.mark.unit


async def test_malformed_frame_does_not_drop_following_transcript_and_speech_boundaries():
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
    async with TestServer(app) as server, aiohttp.ClientSession() as http:
        provider = STT(api_key="key", base_url=str(server.make_url("")), http_session=http)
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
    assert transcript.words is not None and list(transcript.words) == ["Hello"]
    assert transcript.metadata == {
        "utterances": [
            {
                "start": transcript.words[0].start_time,
                "end": transcript.words[0].end_time,
                "text": "Hello",
            }
        ]
    }
