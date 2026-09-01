"""Tests for Rime TTS plugin defaults and request controls."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any
from urllib.parse import parse_qs, urlparse

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions

pytestmark = pytest.mark.unit


def test_model_and_speaker_defaults() -> None:
    from livekit.plugins.rime import TTS

    default_tts = TTS(api_key="test-key")
    explicit_coda_tts = TTS(api_key="test-key", model="coda")

    assert default_tts.model == "coda"
    assert default_tts._opts.speaker == "astra"
    assert default_tts.sample_rate == 24000
    assert explicit_coda_tts.model == "coda"
    assert explicit_coda_tts._opts.speaker == "lyra"
    assert explicit_coda_tts.sample_rate == 24000


def test_arcana_model_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    from livekit.plugins.rime import TTS

    with caplog.at_level(logging.WARNING, logger="livekit.plugins.rime"):
        tts = TTS(api_key="test-key", model="arcana")

    assert 'Rime Arcana is no longer supported. Use model="coda" instead.' in caplog.messages

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.rime"):
        tts.update_options(model="arcana")

    assert 'Rime Arcana is no longer supported. Use model="coda" instead.' in caplog.messages


def test_coda_request_controls() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(
        api_key="test-key",
        model="coda",
        repetition_penalty=1.1,
        temperature=0.5,
        top_p=0.9,
        max_tokens=200,
        time_scale_factor=1.2,
    )

    params = parse_qs(urlparse(tts._ws_url()).query)

    assert params["repetition_penalty"] == ["1.1"]
    assert params["temperature"] == ["0.5"]
    assert params["top_p"] == ["0.9"]
    assert params["max_tokens"] == ["200"]
    assert params["timeScaleFactor"] == ["1.2"]

    tts.update_options(
        repetition_penalty=1.2,
        temperature=0.6,
        top_p=0.8,
        max_tokens=300,
        time_scale_factor=1.1,
    )
    updated_params = parse_qs(urlparse(tts._ws_url()).query)

    assert updated_params["repetition_penalty"] == ["1.2"]
    assert updated_params["temperature"] == ["0.6"]
    assert updated_params["top_p"] == ["0.8"]
    assert updated_params["max_tokens"] == ["300"]
    assert updated_params["timeScaleFactor"] == ["1.1"]


@pytest.mark.parametrize(
    ("model", "expected_sample_rate"),
    [("coda", 24000), ("mistv3", 24000), ("mistv2", 22050)],
)
def test_sample_rate_uses_model_service_default(model: str, expected_sample_rate: int) -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key", model=model, use_websocket=True)

    assert tts.sample_rate == expected_sample_rate
    assert "samplingRate" not in parse_qs(urlparse(tts._ws_url()).query)


def test_explicit_sample_rate_is_sent_and_preserved_across_model_updates() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key", model="coda", sample_rate=16000, use_websocket=True)

    assert tts.sample_rate == 16000
    assert parse_qs(urlparse(tts._ws_url()).query)["samplingRate"] == ["16000"]

    tts.update_options(model="mistv2")

    assert tts.sample_rate == 16000
    assert parse_qs(urlparse(tts._ws_url()).query)["samplingRate"] == ["16000"]


def test_sample_rate_tracks_service_default_across_model_updates() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key", model="coda", use_websocket=True)

    tts.update_options(model="mistv2")
    assert tts.sample_rate == 22050
    assert "samplingRate" not in parse_qs(urlparse(tts._ws_url()).query)

    tts.update_options(model="mistv3")
    assert tts.sample_rate == 24000
    assert "samplingRate" not in parse_qs(urlparse(tts._ws_url()).query)


async def test_chunked_stream_keeps_sample_rate_after_parent_update() -> None:
    from livekit.plugins.rime import TTS

    request_received = asyncio.Event()
    release_response = asyncio.Event()
    payload: dict[str, object] = {}

    async def synthesize(request: web.Request) -> web.Response:
        payload.update(await request.json())
        request_received.set()
        await release_response.wait()
        return web.Response(body=b"\x01\x00" * 2400, content_type="audio/pcm")

    app = web.Application()
    app.router.add_post("/tts", synthesize)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]

    try:
        async with aiohttp.ClientSession() as session:
            tts = TTS(
                api_key="test-key",
                model="coda",
                base_url=f"http://127.0.0.1:{port}/tts",
                http_session=session,
            )
            stream = tts.synthesize("hello", conn_options=APIConnectOptions(max_retry=0, timeout=2))
            try:
                await asyncio.wait_for(request_received.wait(), timeout=2)
                tts.update_options(model="mistv2")
                release_response.set()
                events = [event async for event in stream]
            finally:
                release_response.set()
                await stream.aclose()
                await tts.aclose()
    finally:
        await runner.cleanup()

    assert payload["modelId"] == "coda"
    assert "samplingRate" not in payload
    assert events[0].frame.sample_rate == 24000


async def test_chunked_stream_copies_nested_sample_rate_options() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key", model="coda", sample_rate=22050)
    stream = tts.synthesize("hello")
    try:
        tts.update_options(sample_rate=16000)

        assert stream._opts.coda_options is not None
        assert stream._opts.coda_options.sample_rate == 22050
        assert stream._sample_rate == 22050
    finally:
        await stream.aclose()
        await tts.aclose()


def test_websocket_url_selects_coda_v1() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(
        api_key="test-key",
        websocket_url="wss://api.rimetts.com/coda/v1/coda/ws",
    )

    assert tts.model == "coda"
    assert tts.capabilities.streaming is True
    assert tts.capabilities.aligned_transcript is False
    assert "websocket_url" in inspect.signature(TTS).parameters
    assert "tokenizer" in inspect.signature(TTS).parameters
    assert "websocket_protocol" not in inspect.signature(TTS).parameters
    assert "sentence_tokenization" not in inspect.signature(TTS).parameters


@pytest.mark.parametrize(
    "websocket_url",
    [
        "https://api.rimetts.com/coda/v1/coda/ws",
        "/coda/v1/coda/ws",
    ],
)
def test_tts_rejects_non_websocket_urls(websocket_url: str) -> None:
    from livekit.plugins.rime import TTS

    with pytest.raises(ValueError, match="absolute ws or wss URL"):
        TTS(api_key="test-key", websocket_url=websocket_url)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "websocket_url": "wss://example.com/coda/ws",
                "base_url": "https://example.com/coda",
            },
            "cannot be used with base_url",
        ),
        (
            {
                "websocket_url": "wss://example.com/coda/ws",
                "model": "mistv2",
            },
            'selects model="coda"',
        ),
        (
            {
                "websocket_url": "wss://example.com/coda/ws",
                "use_websocket": True,
            },
            "omit use_websocket",
        ),
        (
            {
                "websocket_url": "wss://example.com/coda/ws",
                "speed_alpha": 1.1,
            },
            "speed_alpha",
        ),
        (
            {
                "websocket_url": "wss://example.com/coda/ws",
                "temperature": 0.7,
            },
            "generation controls",
        ),
    ],
)
def test_v1_rejects_invalid_configuration(kwargs: dict[str, Any], message: str) -> None:
    from livekit.plugins.rime import TTS

    with pytest.raises(ValueError, match=message):
        TTS(api_key="test-key", **kwargs)


def test_v1_rejects_generation_controls_on_update() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key", websocket_url="wss://example.com/coda/ws")

    with pytest.raises(ValueError, match="generation controls"):
        tts.update_options(top_p=0.8)
