"""Tests for Rime TTS plugin defaults and request controls."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import traceback
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions, APIError

pytestmark = pytest.mark.unit

_SECRET = "customer-secret-marker"


def _assert_exception_is_safe(exc: BaseException) -> None:
    rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    assert _SECRET not in str(exc)
    assert _SECRET not in repr(exc)
    assert _SECRET not in rendered


class _FailingWS3:
    def __init__(self, failure: str) -> None:
        self._failure = failure

    async def send_str(self, data: str) -> None:
        payload = json.loads(data)
        if self._failure == "send" and "text" in payload:
            raise RuntimeError(_SECRET)

    async def receive(self, *, timeout: float | None = None) -> aiohttp.WSMessage:
        if self._failure == "provider":
            return aiohttp.WSMessage(
                aiohttp.WSMsgType.TEXT,
                json.dumps({"type": "error", "message": _SECRET}),
                None,
            )
        if self._failure == "transport":
            return aiohttp.WSMessage(aiohttp.WSMsgType.ERROR, None, None)
        return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None)

    def exception(self) -> BaseException:
        return RuntimeError(_SECRET)

    async def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("model", "expected_is_mist", "expected_time_scale", "expected_reduce_latency"),
    [
        ("coda", False, True, False),
        ("mistv2", True, False, True),
        ("mistv3", True, True, False),
        ("mistv4", True, True, False),
        ("custom-mist", False, True, False),
    ],
)
def test_model_capabilities(
    model: str,
    expected_is_mist: bool,
    expected_time_scale: bool,
    expected_reduce_latency: bool,
) -> None:
    from livekit.plugins.rime.models import (
        is_mist_model,
        supports_reduce_latency,
        supports_time_scale_factor,
    )

    assert is_mist_model(model) is expected_is_mist
    assert supports_time_scale_factor(model) is expected_time_scale
    assert supports_reduce_latency(model) is expected_reduce_latency


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


async def test_chunked_stream_sends_default_mistv2_sample_rate() -> None:
    from livekit.plugins.rime import TTS

    payload: dict[str, object] = {}

    async def synthesize(request: web.Request) -> web.Response:
        payload.update(await request.json())
        return web.Response(body=b"\x01\x00" * 2205, content_type="audio/pcm")

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
                model="mistv2",
                base_url=f"http://127.0.0.1:{port}/tts",
                http_session=session,
            )
            stream = tts.synthesize("hello", conn_options=APIConnectOptions(max_retry=0, timeout=2))
            try:
                events = [event async for event in stream]
            finally:
                await stream.aclose()
                await tts.aclose()
    finally:
        await runner.cleanup()

    assert payload["samplingRate"] == 22050
    assert {event.frame.sample_rate for event in events} == {22050}


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
    assert payload["samplingRate"] == 24000
    assert events[0].frame.sample_rate == 24000


async def test_chunked_stream_copies_sample_rate_options() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key", model="coda", sample_rate=22050)
    stream = tts.synthesize("hello")
    try:
        tts.update_options(sample_rate=16000)

        assert stream._opts.sample_rate == 22050
        assert stream._sample_rate == 22050
    finally:
        await stream.aclose()
        await tts.aclose()


def test_websocket_url_selects_v1_with_binary_and_coda_defaults() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(
        api_key="test-key",
        websocket_url="wss://api.rime.ai/coda/ws",
    )

    assert tts.model == "coda"
    assert tts.capabilities.streaming is True
    assert tts.capabilities.aligned_transcript is False
    assert "websocket_url" in inspect.signature(TTS).parameters
    assert "tokenizer" in inspect.signature(TTS).parameters
    assert "audio_format" in inspect.signature(TTS).parameters
    assert inspect.signature(TTS).parameters["websocket_protocol"].default == "binary"
    assert tts._opts.audio_format == "audio/pcm"
    assert "sentence_tokenization" not in inspect.signature(TTS).parameters


def test_websocket_url_derives_mist_model_and_accepts_options() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(
        api_key="test-key",
        websocket_url="wss://api.rime.ai/mist/ws",
        pause_between_brackets=True,
    )

    assert tts.model == "mistv3"
    assert tts._opts.speaker == "cove"
    assert tts._opts.mist_options is not None
    assert tts._opts.mist_options.pause_between_brackets is True


def test_dedicated_websocket_url_uses_explicit_model() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(
        api_key="test-key",
        websocket_url=("wss://tigerstripe-dialpad.aws-us-east-1.whiteglove.rime.ai/ws"),
        model="coda",
    )

    assert tts.model == "coda"


def test_dedicated_websocket_url_requires_model() -> None:
    from livekit.plugins.rime import TTS

    with pytest.raises(ValueError, match="model is required"):
        TTS(
            api_key="test-key",
            websocket_url=("wss://tigerstripe-dialpad.aws-us-east-1.whiteglove.rime.ai/ws"),
        )


def test_custom_websocket_endpoint_requires_explicit_opt_in() -> None:
    from livekit.plugins.rime import TTS

    with pytest.raises(ValueError, match="trusted Rime host"):
        TTS(
            api_key="test-key",
            websocket_url="wss://voice.customer.example/coda/ws",
        )

    tts = TTS(
        api_key="test-key",
        websocket_url="wss://voice.customer.example/coda/ws",
        allow_custom_endpoint=True,
    )

    assert tts.model == "coda"


def test_custom_base_url_requires_explicit_opt_in() -> None:
    from livekit.plugins.rime import TTS

    with pytest.raises(ValueError, match="trusted Rime host"):
        TTS(api_key="test-key", base_url="https://voice.customer.example/tts")

    tts = TTS(
        api_key="test-key",
        base_url="https://voice.customer.example/tts",
        allow_custom_endpoint=True,
    )

    assert tts._base_url == "https://voice.customer.example/tts"


@pytest.mark.parametrize(
    "websocket_url",
    [
        "https://api.rime.ai/coda/ws",
        "/coda/ws",
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
                "websocket_url": "wss://api.rime.ai/coda/ws",
                "base_url": "https://users.rime.ai/coda",
            },
            "cannot be used with base_url",
        ),
        (
            {
                "websocket_url": "wss://api.rime.ai/coda/ws",
                "use_websocket": True,
            },
            "omit use_websocket",
        ),
        (
            {
                "websocket_url": "wss://api.rime.ai/coda/ws",
                "speed_alpha": 1.1,
            },
            "speed_alpha",
        ),
        (
            {
                "websocket_url": "wss://api.rime.ai/coda/ws",
                "temperature": 0.7,
            },
            "generation controls",
        ),
        (
            {
                "websocket_url": "wss://api.rime.ai/coda/ws",
                "model": "coda",
            },
            "model is derived",
        ),
        (
            {
                "websocket_url": "wss://api.rime.ai/coda/ws",
                "websocket_protocol": "auto",
            },
            "binary.*json",
        ),
        (
            {
                "websocket_url": "wss://api.rime.ai/coda/ws",
                "audio_format": "audio/aac",
            },
            "unsupported Rime audio_format",
        ),
        (
            {
                "websocket_url": "wss://api.rime.ai/coda/ws",
                "pause_between_brackets": True,
            },
            "Mist options",
        ),
    ],
)
def test_v1_rejects_invalid_configuration(kwargs: dict[str, Any], message: str) -> None:
    from livekit.plugins.rime import TTS

    with pytest.raises(ValueError, match=message):
        TTS(api_key="test-key", **kwargs)


def test_v1_rejects_generation_controls_on_update() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key", websocket_url="wss://api.rime.ai/coda/ws")

    with pytest.raises(ValueError, match="generation controls"):
        tts.update_options(top_p=0.8)


def test_audio_format_requires_v1_websocket() -> None:
    from livekit.plugins.rime import TTS

    with pytest.raises(ValueError, match="only supported with the Rime v1"):
        TTS(api_key="test-key", audio_format="audio/wav")


def test_v1_updates_audio_format() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key", websocket_url="wss://api.rime.ai/coda/ws")
    tts.update_options(audio_format="audio/mpeg")

    assert tts._opts.audio_format == "audio/mpeg"

    with pytest.raises(ValueError, match="unsupported Rime audio_format"):
        tts.update_options(audio_format="audio/aac")

    assert tts._opts.audio_format == "audio/mpeg"


def test_v1_derives_model_when_endpoint_is_updated() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key", websocket_url="wss://api.rime.ai/coda/ws")
    tts.update_options(
        websocket_url="wss://api.rime.ai/mist/ws",
        pause_between_brackets=True,
    )

    assert tts.model == "mistv3"
    assert tts._opts.mist_options is not None
    assert tts._opts.mist_options.pause_between_brackets is True


def test_v1_rejects_model_update() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key", websocket_url="wss://api.rime.ai/coda/ws")

    with pytest.raises(ValueError, match="only be updated together"):
        tts.update_options(model="mistv3")


def test_v1_dedicated_endpoint_update_keeps_current_model() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key", websocket_url="wss://api.rime.ai/coda/ws")
    tts.update_options(websocket_url="wss://tigerstripe.aws-us-east-1.whiteglove.rime.ai/ws")

    assert tts.model == "coda"


def test_v1_dedicated_endpoint_rejects_model_change_for_same_url() -> None:
    from livekit.plugins.rime import TTS

    websocket_url = "wss://tigerstripe.aws-us-east-1.whiteglove.rime.ai/ws"
    tts = TTS(api_key="test-key", websocket_url=websocket_url, model="coda")

    with pytest.raises(ValueError, match="model cannot change without changing websocket_url"):
        tts.update_options(websocket_url=websocket_url, model="mistv3")

    assert tts.model == "coda"


@pytest.mark.parametrize(
    "updated_websocket_url",
    [
        "wss://tigerstripe.aws-us-east-1.whiteglove.rime.ai/ws/",
        "wss://tigerstripe.aws-us-east-1.whiteglove.rime.ai/ws?token=rotated",
        "wss://TIGERSTRIPE.aws-us-east-1.whiteglove.rime.ai/ws",
        "wss://tigerstripe.aws-us-east-1.whiteglove.rime.ai:443/ws",
    ],
)
def test_v1_dedicated_endpoint_rejects_model_change_for_equivalent_url(
    updated_websocket_url: str,
) -> None:
    from livekit.plugins.rime import TTS

    websocket_url = "wss://tigerstripe.aws-us-east-1.whiteglove.rime.ai/ws"
    tts = TTS(
        api_key="test-key",
        websocket_url=websocket_url,
        model="coda",
    )

    with pytest.raises(ValueError, match="model cannot change without changing websocket_url"):
        tts.update_options(websocket_url=updated_websocket_url, model="mistv3")

    assert tts.model == "coda"
    assert tts._websocket_v1_adapter is not None
    assert tts._websocket_v1_adapter._websocket_v1_url == websocket_url


def test_v1_dedicated_endpoint_allows_same_model_for_same_url() -> None:
    from livekit.plugins.rime import TTS

    websocket_url = "wss://tigerstripe.aws-us-east-1.whiteglove.rime.ai/ws"
    tts = TTS(api_key="test-key", websocket_url=websocket_url, model="coda")

    tts.update_options(websocket_url=websocket_url, model="coda")

    assert tts.model == "coda"


def test_v1_dedicated_endpoint_updates_connection_url_for_same_model() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(
        api_key="test-key",
        websocket_url="wss://tigerstripe.aws-us-east-1.whiteglove.rime.ai/ws",
        model="coda",
    )
    adapter = tts._websocket_v1_adapter
    assert adapter is not None
    previous_pool = adapter._pool
    updated_websocket_url = "wss://tigerstripe.aws-us-east-1.whiteglove.rime.ai/ws?token=rotated"

    tts.update_options(websocket_url=updated_websocket_url, model="coda")

    assert tts.model == "coda"
    assert adapter._websocket_v1_url == updated_websocket_url
    assert adapter._pool is not previous_pool


def test_v1_dedicated_endpoint_allows_model_change_with_new_url() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(
        api_key="test-key",
        websocket_url="wss://coda.aws-us-east-1.whiteglove.rime.ai/ws",
        model="coda",
    )

    tts.update_options(
        websocket_url="wss://mist.aws-us-east-1.whiteglove.rime.ai/ws",
        model="mistv3",
    )

    assert tts.model == "mistv3"

    with pytest.raises(ValueError, match="model cannot change without changing websocket_url"):
        tts.update_options(
            websocket_url="wss://mist.aws-us-east-1.whiteglove.rime.ai/ws",
            model="coda",
        )

    assert tts.model == "mistv3"


@pytest.mark.parametrize("failure", ["transport", "provider", "send"])
async def test_ws3_errors_do_not_expose_provider_or_transport_data(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from livekit.plugins.rime import TTS
    from livekit.plugins.rime._legacy_websocket_adapter import LegacyWebSocketAdapter

    websocket = _FailingWS3(failure)

    async def connect_ws(
        self: LegacyWebSocketAdapter, *, websocket_url: str, timeout: float
    ) -> aiohttp.ClientWebSocketResponse:
        return cast(aiohttp.ClientWebSocketResponse, websocket)

    monkeypatch.setattr(LegacyWebSocketAdapter, "_connect", connect_ws)
    tts = TTS(api_key="test-key", model="coda", use_websocket=True)
    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
    stream.push_text("hello")
    stream.end_input()

    try:
        with pytest.raises(APIError) as exc_info:
            _ = [event async for event in stream]
    finally:
        await stream.aclose()
        await tts.aclose()

    _assert_exception_is_safe(exc_info.value)
    assert exc_info.value.__cause__ is None


async def test_ws3_stream_keeps_options_after_parent_update() -> None:
    from livekit.plugins.rime import TTS

    request_models: list[str] = []

    async def websocket(request: web.Request) -> web.WebSocketResponse:
        request_models.append(request.query["modelId"])
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            payload = json.loads(message.data)
            if "text" in payload:
                await ws.send_json(
                    {
                        "type": "chunk",
                        "data": base64.b64encode(b"\x01\x00" * 2400).decode(),
                    }
                )
            elif payload.get("operation") == "flush":
                await ws.send_json({"type": "done"})
            elif payload.get("operation") == "eos":
                await ws.send_json({"type": "done"})
                break
        return ws

    app = web.Application()
    app.router.add_get("/ws3", websocket)
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
                base_url=f"ws://127.0.0.1:{port}",
                http_session=session,
                allow_custom_endpoint=True,
            )
            metrics = []
            tts.on("metrics_collected", metrics.append)
            coda_stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
            tts.update_options(model="mistv2")
            mist_stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))

            async def collect(stream: Any) -> list[Any]:
                stream.push_text("hello")
                stream.end_input()
                try:
                    return [event async for event in stream]
                finally:
                    await stream.aclose()

            coda_events = await collect(coda_stream)
            mist_events = await collect(mist_stream)
            await tts.aclose()
    finally:
        await runner.cleanup()

    assert request_models == ["coda", "mistv2"]
    assert {event.frame.sample_rate for event in coda_events} == {24000}
    assert {event.frame.sample_rate for event in mist_events} == {22050}
    assert [metric.metadata.model_name for metric in metrics] == ["coda", "mistv2"]
