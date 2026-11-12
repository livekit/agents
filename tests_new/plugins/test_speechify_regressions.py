"""Speechify client ownership, route selection and pagination."""

import base64
import json
from typing import Any

import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from speechify import AsyncSpeechify

from livekit.agents import APIConnectOptions
from livekit.plugins.speechify import TTS
from livekit.plugins.speechify.models import TTSModels

# Any: decoded wire JSON, indexed by the provider's documented shape
Wire = dict[str, Any]


async def test_list_voices_follows_async_pagination() -> None:
    """Voice listing returns every async page and normalizes unspecified gender. https://github.com/livekit/agents/pull/8034"""
    cursors: list[str | None] = []

    async def voices(request: web.Request) -> web.StreamResponse:
        cursor = request.query.get("cursor")
        cursors.append(cursor)
        return web.json_response(
            {
                "voices": [
                    {
                        "id": "two" if cursor else "one",
                        "display_name": "Voice",
                        "gender": "notSpecified",
                        "locale": "en-US",
                        "type": "shared",
                        "models": [{"name": "simba-3.2", "languages": []}],
                    }
                ],
                "has_more": not bool(cursor),
                "next_cursor": None if cursor else "next",
            }
        )

    app = web.Application()
    app.router.add_get("/v1/voices", voices)
    async with TestServer(app) as server:
        provider = TTS(api_key="key", base_url=str(server.make_url("")))
        try:
            found = await provider.list_voices()
            assert [voice.id for voice in found] == ["one", "two"]
            assert [voice.gender for voice in found] == ["neutral", "neutral"]
            assert cursors == [None, "next"]
        finally:
            await provider.aclose()


@pytest.mark.parametrize("model", ["simba-3.2", "simba-3.0", "simba-english", "simba-multilingual"])
async def test_model_routes_and_client_lifetime(
    model: TTSModels, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Models use the matching route, drain responses and close only owned clients. https://github.com/livekit/agents/pull/8029 https://github.com/livekit/agents/pull/6542 https://github.com/livekit/agents/pull/7543"""
    requests: list[tuple[str, dict[str, str], Wire]] = []
    closed_clients: list[httpx.AsyncClient] = []
    pcm = b"\x03\x00" * 4800
    original_close = httpx.AsyncClient.aclose

    async def record_close(client: httpx.AsyncClient) -> None:
        closed_clients.append(client)
        await original_close(client)

    monkeypatch.setattr(httpx.AsyncClient, "aclose", record_close)

    async def speak(request: web.Request) -> web.StreamResponse:
        requests.append((request.path, dict(request.headers), await request.json()))
        if "/with-timestamps" in request.path:
            events: list[dict[str, object]] = [
                {
                    "type": "speech.chunk",
                    "audio": base64.b64encode(pcm).decode(),
                    "speech_marks": [],
                },
                {"type": "speech.done", "billable_characters_count": 6, "audio_duration_ms": 200},
            ]
            return web.Response(
                text="".join(
                    f"event: {event['type']}\ndata: " + json.dumps(event) + "\n\n"
                    for event in events
                ),
                content_type="text/event-stream",
            )
        return web.json_response(
            {
                "audio_data": base64.b64encode(pcm).decode(),
                "audio_format": "pcm",
                "billable_characters_count": 6,
                "speech_marks": {
                    "chunks": [],
                    "type": "sentence",
                    "start": 0,
                    "end": 6,
                    "start_time": 0,
                    "end_time": 200,
                },
            }
        )

    app = web.Application()
    app.router.add_post("/{path:.*}", speak)
    async with TestServer(app) as server:
        url = str(server.make_url(""))
        provider = TTS(api_key="key", base_url=url, model=model, voice_id="voice")
        try:
            async with provider.synthesize(
                "Hello.", conn_options=APIConnectOptions(max_retry=0)
            ) as stream:
                frames = [ev async for ev in stream]
            audio = b"".join(bytes(ev.frame.data) for ev in frames)
            assert audio[: len(pcm)] == pcm and not any(audio[len(pcm) :])
            assert len(requests) == 1
            path, headers, body = requests[0]
            assert path == (
                "/v1/audio/stream/with-timestamps"
                if model.startswith("simba-3.")
                else "/v1/audio/speech"
            )
            assert body["model"] == model
            assert headers["Speechify-Caller"] == "livekit"
        finally:
            await provider.aclose()
        assert len(closed_clients) == 1 and closed_clients[0].is_closed
        async with httpx.AsyncClient() as injected_http:
            client = AsyncSpeechify(token="key", base_url=url, httpx_client=injected_http)
            injected = TTS(client=client)
            await injected.aclose()
            assert not injected_http.is_closed
            response = await injected_http.post(url + "/v1/audio/speech", json={})
            assert response.status_code == 200
