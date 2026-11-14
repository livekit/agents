"""Simplismart STT: the streaming wire and batch recognition of a streaming instance."""

from __future__ import annotations

import asyncio
import json

from aiohttp import web

from livekit import rtc
from livekit.agents import APIConnectOptions, stt
from livekit.plugins.simplismart import STT
from tests.fakes import until
from tests.plugins.conftest import WsPeer


def _audio() -> rtc.AudioFrame:
    return rtc.AudioFrame(b"\0\0" * 1600, 16000, 1, 1600)


async def test_each_binary_message_is_a_final_transcript_after_the_language(
    ws_peer: WsPeer,
) -> None:
    """The socket opens with the language; each binary message the server sends is a final."""
    model = STT(api_key="key", base_url=ws_peer.url, streaming=True, language="fr")
    stream = model.stream(conn_options=APIConnectOptions(max_retry=0))
    events: list[stt.SpeechEvent] = []

    async def read() -> None:
        async for event in stream:
            events.append(event)

    reader = asyncio.create_task(read())
    try:
        stream.push_frame(_audio())
        connection = await ws_peer.connected()
        await ws_peer.send(b"bonjour")
        await until(
            lambda: stt.SpeechEventType.FINAL_TRANSCRIPT in [e.type for e in events],
            "the final transcript",
        )
    finally:
        await model.aclose()
        await reader

    assert connection.request.path == "/ws/audio"
    assert json.loads(connection.texts[0]) == {"language": "fr"}
    [final] = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert final.alternatives[0].text == "bonjour"


async def test_recognize_on_a_streaming_instance_posts_to_the_base_url(ws_peer: WsPeer) -> None:
    """streaming=True used to replace the batch URL with the socket's, so recognize() dialed
    wss://<host>/ws/audio."""
    paths: list[str] = []

    async def predict(request: web.Request) -> web.StreamResponse:
        paths.append(request.path)
        return web.json_response({"transcription": ["hello"], "timestamps": [[0.0, 0.5]]})

    ws_peer.http = predict
    model = STT(api_key="key", base_url=ws_peer.url + "/predict", streaming=True)

    event = await model.recognize(_audio(), conn_options=APIConnectOptions(max_retry=0))

    assert paths == ["/predict"]
    assert event.alternatives[0].text == "hello"
