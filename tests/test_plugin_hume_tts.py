from __future__ import annotations

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectionError, APIConnectOptions, APIError, APIStatusError
from livekit.plugins.hume import TTS, VoiceByName

pytestmark = pytest.mark.unit


def test_update_options_rejects_instant_mode_without_a_voice():
    tts = TTS(api_key="test-key", voice=None)
    with pytest.raises(ValueError, match="instant_mode cannot be enabled without"):
        tts.update_options(instant_mode=True)
    assert tts._opts.instant_mode is False

    tts = TTS(api_key="test-key")
    voice = tts._opts.voice
    with pytest.raises(ValueError, match="instant_mode cannot be enabled without"):
        tts.update_options(voice=None, speed=1.5)
    assert tts._opts.voice == voice
    assert tts._opts.speed is None


def test_update_options_accepts_a_consistent_voice_and_instant_mode():
    tts = TTS(api_key="test-key")
    tts.update_options(voice=None, instant_mode=False)
    assert tts._opts.voice is None
    assert tts._opts.instant_mode is False

    voice = VoiceByName(name="Ava Song")
    tts.update_options(voice=voice, instant_mode=True)
    assert tts._opts.voice == voice
    assert tts._opts.instant_mode is True


async def test_synthesize_does_not_retry_a_client_error() -> None:
    requests = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal requests
        requests += 1
        return web.Response(status=401)

    app = web.Application()
    app.router.add_post("/v0/tts/stream/json", handler)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        tts = TTS(api_key="test-key", base_url=str(server.make_url("")), http_session=session)
        with pytest.raises(APIStatusError) as exc_info:
            async with tts.synthesize("Hello there.") as stream:
                await stream.collect()

    assert exc_info.value.status_code == 401
    assert requests == 1


async def test_synthesize_keeps_a_stream_error_from_hume() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(text='{"type": "error", "message": "voice not found"}\n')

    app = web.Application()
    app.router.add_post("/v0/tts/stream/json", handler)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        tts = TTS(api_key="test-key", base_url=str(server.make_url("")), http_session=session)
        with pytest.raises(APIError) as exc_info:
            async with tts.synthesize(
                "Hello there.", conn_options=APIConnectOptions(max_retry=0)
            ) as stream:
                await stream.collect()

    assert not isinstance(exc_info.value, APIConnectionError)
    assert "voice not found" in exc_info.value.message
