"""Fireworks AI plugin: streaming STT connection options."""

from __future__ import annotations

import asyncio

from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, http
from livekit.agents.stt import SpeechEventType
from livekit.agents.types import NOT_GIVEN
from livekit.plugins.fireworksai import STT


async def test_stream_language_applies_to_that_stream_only() -> None:
    """A per-stream language reaches only that socket; https://github.com/livekit/agents/issues/8368."""
    languages: list[str | None] = []

    async def transcribe(request: web.Request) -> web.WebSocketResponse:
        languages.append(request.query.get("language"))
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                break
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/{path:.*}", transcribe)
    async with TestServer(app) as server, http.open() as session:
        model = STT(
            api_key="test",
            language="en",
            base_url=str(server.make_url("/v1").with_scheme("ws")),
            http_session=session,
        )
        options = APIConnectOptions(max_retry=0)
        try:
            for language in ("fr", NOT_GIVEN):
                async with model.stream(language=language, conn_options=options) as stream:
                    stream.end_input()
                    async for _ in stream:
                        pass
        finally:
            await model.aclose()

    assert languages == ["fr", "en"]


async def test_final_transcript_ends_the_speech() -> None:
    """A final closes the speech, so turn_detection="stt" can commit the turn."""

    async def transcribe(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        words = [{"word": "hello", "is_final": True}]
        await ws.send_json({"segments": [{"id": 0, "text": "hello", "words": words}]})
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                break
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/{path:.*}", transcribe)
    async with TestServer(app) as server, http.open() as session:
        model = STT(
            api_key="test",
            base_url=str(server.make_url("/v1").with_scheme("ws")),
            http_session=session,
        )
        try:
            types: list[SpeechEventType] = []
            async with (
                asyncio.timeout(2),
                model.stream(conn_options=APIConnectOptions(max_retry=0)) as stream,
            ):
                async for event in stream:
                    types.append(event.type)
                    if event.type == SpeechEventType.END_OF_SPEECH:
                        stream.end_input()  # only the final's own END_OF_SPEECH gets here
        finally:
            await model.aclose()

    speech = [t for t in types if t != SpeechEventType.RECOGNITION_USAGE]
    assert speech == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]


async def test_a_reconnected_socket_transcribes_from_its_first_segment() -> None:
    """Segment ids restart with each connection: after a reconnect, the new socket's segment 0
    is a new transcript, not one older than the last final."""
    connections = 0

    async def transcribe(request: web.Request) -> web.WebSocketResponse:
        nonlocal connections
        connections += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        segment_id, text = (3, "hello") if connections == 1 else (0, "again")
        words = [{"word": text, "is_final": True}]
        await ws.send_json({"segments": [{"id": segment_id, "text": text, "words": words}]})
        if connections == 1:
            await ws.close(code=1011)
            return ws
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/{path:.*}", transcribe)
    async with TestServer(app) as server, http.open() as session:
        model = STT(
            api_key="test",
            base_url=str(server.make_url("/v1").with_scheme("ws")),
            http_session=session,
        )
        finals: list[str] = []
        try:
            async with (
                asyncio.timeout(2),
                model.stream(conn_options=APIConnectOptions(max_retry=1, retry_interval=0)) as s,
            ):
                async for event in s:
                    if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                        finals.append(event.alternatives[0].text)
                        if len(finals) == 2:
                            break
        finally:
            await model.aclose()

    assert finals == ["hello", "again"]
