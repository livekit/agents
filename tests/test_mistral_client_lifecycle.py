"""Mistral provider adapters close their owned SDK transports."""

from __future__ import annotations

import importlib

import pytest
from mistralai.client import Mistral

from livekit.plugins.mistralai import LLM, STT, TTS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("provider_type", [LLM, STT, TTS])
@pytest.mark.parametrize("borrowed", [False, True])
async def test_only_owned_mistral_sdk_transports_close(
    provider_type: type[LLM] | type[STT] | type[TTS],
    borrowed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = Mistral(api_key="test")
    sync_transport = sdk.sdk_configuration.client
    async_transport = sdk.sdk_configuration.async_client
    assert sync_transport is not None and async_transport is not None
    monkeypatch.setattr(
        importlib.import_module(provider_type.__module__), "Mistral", lambda **_: sdk
    )
    provider = provider_type(client=sdk) if borrowed else provider_type(api_key="test")
    try:
        await provider.aclose()
        assert sync_transport.is_closed is not borrowed
        assert async_transport.is_closed is not borrowed
    finally:
        await sdk.__aexit__(None, None, None)
        sdk.__exit__(None, None, None)


async def test_realtime_sdk_connection_closes_when_recognition_is_cancelled() -> None:
    import asyncio

    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from livekit.agents import APIConnectOptions, stt
    from tests.fake_vad import FakeVAD

    closed = asyncio.Event()

    async def transcribe(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        try:
            await ws.send_json(
                {
                    "type": "session.created",
                    "session": {
                        "request_id": "test",
                        "model": "voxtral-realtime",
                        "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
                    },
                }
            )
            await ws.send_json({"type": "transcription.text.delta", "text": "hello"})
            async for _ in ws:
                pass
        finally:
            closed.set()
        return ws

    app = web.Application()
    app.router.add_get("/v1/audio/transcriptions/realtime", transcribe)
    async with TestServer(app) as server:
        sdk = Mistral(api_key="test", server_url=str(server.make_url("")))
        provider = STT(client=sdk, model="voxtral-realtime", vad=FakeVAD())
        try:
            async with asyncio.timeout(2):
                async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                    event = await anext(stream)
                    assert event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT
                    assert event.alternatives[0].text == "hello"
                await closed.wait()
        finally:
            await provider.aclose()
            await sdk.__aexit__(None, None, None)
            sdk.__exit__(None, None, None)
