"""Tests for Rime TTS requests, streaming, protocol compatibility, and error handling."""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import json
import logging
import traceback
from typing import Any, cast
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

import aiohttp
import av
import numpy as np
import pytest
from aiohttp import web
from google.protobuf import json_format
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rime_api import text_to_speech_pb2 as proto

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
)
from livekit.agents.telemetry import trace_types
from livekit.agents.telemetry.traces import _DynamicTracer
from livekit.agents.tts import tts as tts_module
from livekit.plugins.rime import TTS
from livekit.plugins.rime._websocket_v1 import _codec_for_protocol

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


@pytest.mark.parametrize(
    ("options", "expected_model", "expected_speaker"),
    [
        pytest.param({}, "coda", "lyra", id="http-default"),
        pytest.param({"model": "coda"}, "coda", "lyra", id="http-coda"),
        pytest.param({"use_websocket": True}, "coda", "lyra", id="ws3-default"),
        pytest.param({"use_websocket": True, "model": "coda"}, "coda", "lyra", id="ws3-coda"),
        pytest.param(
            {"websocket_url": "wss://api.rime.ai/coda/ws"}, "coda", "lyra", id="v1-default"
        ),
        pytest.param(
            {"websocket_url": "wss://api.rime.ai/ws", "model": "coda"},
            "coda",
            "lyra",
            id="v1-dedicated",
        ),
        pytest.param({"model": "mistv3"}, "mistv3", "cove", id="http-mist"),
        pytest.param({"use_websocket": True, "model": "mistv3"}, "mistv3", "cove", id="ws3-mist"),
        pytest.param(
            {"websocket_url": "wss://api.rime.ai/mist/ws"}, "mistv3", "cove", id="v1-mist"
        ),
    ],
)
@pytest.mark.parametrize("speaker", [None, "custom-voice"], ids=["default", "explicit"])
def test_model_and_speaker_defaults(
    options: dict[str, Any], expected_model: str, expected_speaker: str, speaker: str | None
) -> None:
    if speaker is not None:
        options = {**options, "speaker": speaker}
    tts = TTS(api_key="test-key", **options)

    assert tts.model == expected_model
    assert tts._opts.speaker == (speaker if speaker is not None else expected_speaker)
    assert tts.sample_rate == 24000


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
    assert parse_qs(urlparse(tts._ws_url()).query)["samplingRate"] == [str(expected_sample_rate)]


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
    assert parse_qs(urlparse(tts._ws_url()).query)["samplingRate"] == ["22050"]

    tts.update_options(model="mistv3")
    assert tts.sample_rate == 24000
    assert parse_qs(urlparse(tts._ws_url()).query)["samplingRate"] == ["24000"]


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


@pytest.mark.parametrize(
    ("update_endpoint", "update_model"),
    [(True, False), (False, True), (True, True)],
    ids=["endpoint", "model", "endpoint-and-model"],
)
async def test_chunked_stream_keeps_endpoint_and_timeout_after_parent_update(
    update_endpoint: bool, update_model: bool
) -> None:
    requests: list[tuple[str, str]] = []

    async def synthesize(request: web.Request) -> web.Response:
        payload = await request.json()
        requests.append((request.path, payload["modelId"]))
        return web.Response(body=b"\x01\x00" * 2400, content_type="audio/pcm")

    app = web.Application()
    app.router.add_post("/{endpoint}", synthesize)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    base_url = f"http://127.0.0.1:{runner.addresses[0][1]}"
    updated_endpoint = "/new" if update_endpoint else "/old"
    updated_model = "mistv3" if update_model else "coda"
    updated_timeout = 30 if update_model else 240
    conn_options = APIConnectOptions(max_retry=0, timeout=2)

    try:
        async with aiohttp.ClientSession() as session:
            tts = TTS(
                api_key="test-key",
                model="coda",
                base_url=f"{base_url}/old",
                http_session=session,
            )
            try:
                with patch.object(session, "post", wraps=session.post) as post:
                    old_stream = tts.synthesize("hello", conn_options=conn_options)
                    # Update before the scheduled request can read the parent's settings.
                    updates: dict[str, Any] = {}
                    if update_endpoint:
                        updates["base_url"] = f"{base_url}{updated_endpoint}"
                    if update_model:
                        updates["model"] = updated_model
                    async with old_stream:
                        tts.update_options(**updates)
                        assert [event async for event in old_stream]
                    async with tts.synthesize("hello", conn_options=conn_options) as new_stream:
                        assert [event async for event in new_stream]

                assert requests == [("/old", "coda"), (updated_endpoint, updated_model)]
                assert [call.kwargs["timeout"].total for call in post.call_args_list] == [
                    240,
                    updated_timeout,
                ]
                assert all(call.kwargs["timeout"].sock_connect == 2 for call in post.call_args_list)
            finally:
                await tts.aclose()
    finally:
        await runner.cleanup()


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
    previous_pool = adapter._pools.current
    updated_websocket_url = "wss://tigerstripe.aws-us-east-1.whiteglove.rime.ai/ws?token=rotated"

    tts.update_options(websocket_url=updated_websocket_url, model="coda")

    assert tts.model == "coda"
    assert adapter._websocket_v1_url == updated_websocket_url
    assert adapter._pools.current is not previous_pool


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
    errors: list[tts_module.TTSError] = []
    tts.on("error", errors.append)
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
    assert len(errors) == 1
    assert errors[0].error is exc_info.value
    assert errors[0].label == tts.label


@pytest.mark.parametrize("stream_count", [1, 2])
async def test_ws3_stream_keeps_options_after_parent_update(stream_count: int) -> None:
    from livekit.plugins.rime import TTS

    request_options: list[tuple[str, str | None]] = []
    coda_closed = asyncio.Event()

    async def websocket(request: web.Request) -> web.WebSocketResponse:
        request_options.append((request.query["modelId"], request.query.get("samplingRate")))
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
        await ws.close()
        if request.query["modelId"] == "coda":
            coda_closed.set()
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
            coda_streams = [
                tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
                for _ in range(stream_count)
            ]
            tts.update_options(model="mistv2")
            mist_stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))

            async def collect(stream: Any) -> list[Any]:
                stream.push_text("hello")
                stream.end_input()
                try:
                    return [event async for event in stream]
                finally:
                    await stream.aclose()

            coda_events = []
            for stream in coda_streams:
                assert not coda_closed.is_set()
                coda_events.extend(await collect(stream))
            await asyncio.wait_for(coda_closed.wait(), timeout=1)
            mist_events = await collect(mist_stream)
            await tts.aclose()
    finally:
        await runner.cleanup()

    assert sorted(request_options) == [("coda", "24000")] * stream_count + [("mistv2", "22050")]
    assert {event.frame.sample_rate for event in coda_events} == {24000}
    assert {event.frame.sample_rate for event in mist_events} == {22050}
    assert [metric.metadata.model_name for metric in metrics] == ["coda"] * stream_count + [
        "mistv2"
    ]


_PCM = b"\x01\x00" * 2205


def _encode_container_audio(container_format: str, codec: str) -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=container_format) as container:
        stream = container.add_stream(codec, rate=24000)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(
            np.zeros((1, 2400), dtype=np.int16), format="s16", layout="mono"
        )
        frame.sample_rate = 24000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def encoded_audio_by_format() -> dict[str, bytes]:
    return {
        "audio/pcm": _PCM,
        "audio/wav": _encode_container_audio("wav", "pcm_s16le"),
        "audio/mpeg": _encode_container_audio("mp3", "mp3"),
        "audio/ogg;codecs=opus": _encode_container_audio("ogg", "libopus"),
        "audio/webm;codecs=opus": _encode_container_audio("webm", "libopus"),
        "audio/pcmu": b"\xff" * 2400,
    }


class _RimeV1Server:
    def __init__(
        self,
        *,
        response_mode: str = "normal",
        ready: dict[str, Any] | None = None,
        cancel_reply: bool = True,
        fail_before_audio: int = 0,
        error_kind: str = "invalid_input",
        audio: bytes = _PCM,
    ) -> None:
        self.response_mode = response_mode
        self.ready = ready or {"protocol": 1, "languages": ["eng"]}
        self.cancel_reply = cancel_reply
        self.fail_before_audio = fail_before_audio
        self.error_kind = error_kind
        self.audio = audio
        self.connections = 0
        self.ready_events = 0
        self.requests: list[dict[str, Any]] = []
        self.request_connections: list[int] = []
        self.headers: list[dict[str, str]] = []
        self.protocols: list[str | None] = []
        self.paths: list[str] = []
        self.text_messages_received = 0
        self._text_received_condition = asyncio.Condition()
        self.request_received = asyncio.Event()
        self.unexpected_requests: list[dict[str, Any]] = []
        self.connection_opened = asyncio.Event()
        self.connection_closed = asyncio.Event()
        self.closed_connections = 0

    async def __aenter__(self) -> _RimeV1Server:
        app = web.Application()
        app.router.add_get("/{path:.*}", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await self._site.start()
        port = self._runner.addresses[0][1]
        self.websocket_url = f"ws://127.0.0.1:{port}/coda/ws"
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.session.close()
        await self._runner.cleanup()
        assert not self.unexpected_requests, (
            f"fake Rime v1 server received unexpected requests: {self.unexpected_requests!r}"
        )

    async def wait_for_text_messages(self, count: int) -> None:
        async with self._text_received_condition:
            await self._text_received_condition.wait_for(
                lambda: self.text_messages_received >= count
            )

    async def _handle(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(protocols=("rime.v1.binary", "rime.v1.json"))
        await ws.prepare(request)
        connection = self.connections
        self.connections += 1
        self.headers.append(dict(request.headers))
        self.protocols.append(ws.ws_protocol)
        self.paths.append(request.path)
        await self._send(ws, {"ready": self.ready})
        self.ready_events += 1
        self.connection_opened.set()

        try:
            async for message in ws:
                envelope = self._decode_request(message)
                self.requests.append(envelope)
                self.request_connections.append(connection)
                self.request_received.set()
                context_id = envelope.get("contextId", "")
                if "start" in envelope:
                    started = (
                        {}
                        if self.response_mode == "missing_started_request_id"
                        else {"requestId": f"request-{connection}"}
                    )
                    await self._send(
                        ws,
                        {
                            "contextId": context_id,
                            "started": started,
                        },
                    )
                elif "text" in envelope:
                    async with self._text_received_condition:
                        self.text_messages_received += 1
                        self._text_received_condition.notify_all()
                    if self.fail_before_audio > 0:
                        self.fail_before_audio -= 1
                        await self._send(
                            ws,
                            {
                                "contextId": context_id,
                                "error": {
                                    "kind": "unavailable",
                                    "message": "retry later",
                                    "requestId": f"request-{connection}",
                                },
                            },
                        )
                    else:
                        await self._respond_to_text(ws, context_id)
                elif "end" in envelope:
                    if self.response_mode == "malformed_done":
                        await self._send_missing_payload(ws, context_id)
                    elif self.response_mode == "invalid_done_type":
                        await ws.send_json({"contextId": context_id, "done": "bad"})
                    elif self.response_mode not in ("error", "partial_error", "no_done"):
                        await self._send(ws, {"contextId": context_id, "done": {}})
                elif "cancel" in envelope and self.cancel_reply:
                    if self.response_mode == "malformed_cancelled":
                        await self._send_missing_payload(ws, context_id)
                    else:
                        terminal = "done" if self.response_mode == "done_on_cancel" else "cancelled"
                        await self._send(ws, {"contextId": context_id, terminal: {}})
                elif "flush" in envelope:
                    self.unexpected_requests.append(envelope)
                    await self._send(
                        ws,
                        {
                            "contextId": context_id,
                            "error": {
                                "kind": "invalid_input",
                                "message": "flush is not part of the Rime v1 protocol",
                            },
                        },
                    )
        finally:
            self.closed_connections += 1
            self.connection_closed.set()
        return ws

    def _decode_request(self, message: aiohttp.WSMessage) -> dict[str, Any]:
        request = proto.WebSocketRequest()
        if message.type == aiohttp.WSMsgType.BINARY:
            request.ParseFromString(message.data)
        elif message.type == aiohttp.WSMsgType.TEXT:
            json_format.Parse(message.data, request, ignore_unknown_fields=True)
        else:
            raise AssertionError(f"unexpected request frame type: {message.type}")
        return json_format.MessageToDict(
            request,
            preserving_proto_field_name=False,
            always_print_fields_with_no_presence=True,
        )

    async def _send(self, ws: web.WebSocketResponse, payload: dict[str, Any]) -> None:
        response = proto.WebSocketResponse()
        json_format.ParseDict(payload, response, ignore_unknown_fields=True)
        if ws.ws_protocol == "rime.v1.binary":
            await ws.send_bytes(response.SerializeToString())
        else:
            await ws.send_json(payload)

    async def _send_missing_payload(self, ws: web.WebSocketResponse, context_id: str) -> None:
        await self._send(ws, {"contextId": context_id})

    async def _respond_to_text(self, ws: web.WebSocketResponse, context_id: str) -> None:
        if self.response_mode == "normal":
            await self._send(
                ws, {"contextId": context_id, "audio": base64.b64encode(self.audio).decode()}
            )
        elif self.response_mode == "wrong_context":
            await self._send(ws, {"contextId": "wrong", "done": {}})
        elif self.response_mode == "early_done":
            await self._send(ws, {"contextId": context_id, "done": {}})
        elif self.response_mode == "invalid_envelope":
            if ws.ws_protocol == "rime.v1.binary":
                await ws.send_bytes(b"\xff")
            else:
                await ws.send_str("{")
        elif self.response_mode == "wrong_frame":
            if ws.ws_protocol == "rime.v1.binary":
                await ws.send_str("{}")
            else:
                await ws.send_bytes(b"")
        elif self.response_mode == "invalid_base64":
            await ws.send_json({"contextId": context_id, "audio": "AQI=%%%"})
        elif self.response_mode == "error":
            await self._send(
                ws,
                {
                    "contextId": context_id,
                    "error": {"kind": self.error_kind, "message": "bad input"},
                },
            )
        elif self.response_mode == "connection_error":
            await self._send(
                ws, {"error": {"kind": self.error_kind, "message": "connection failed"}}
            )
        elif self.response_mode == "partial_error":
            await self._send(
                ws, {"contextId": context_id, "audio": base64.b64encode(_PCM).decode()}
            )
            await asyncio.sleep(0.05)
            await self._send(
                ws,
                {
                    "contextId": context_id,
                    "error": {"kind": "unavailable", "message": "failed late"},
                },
            )


def _v1_tts(server: _RimeV1Server, *, endpoint_model: str = "coda", **kwargs: Any):
    from livekit.plugins.rime import TTS

    websocket_url = server.websocket_url.replace("/coda/ws", f"/{endpoint_model}/ws")
    return TTS(
        api_key="test-key",
        websocket_url=websocket_url,
        http_session=server.session,
        **kwargs,
    )


async def _collect(stream) -> list:
    events = []
    async for event in stream:
        events.append(event)
    return events


def _payloads(server: _RimeV1Server) -> list[str]:
    assert all("flush" not in request for request in server.requests)
    return [next(key for key in request if key != "contextId") for request in server.requests]


def test_v1_binary_envelope_goldens_match_rime_field_numbers() -> None:
    request = proto.WebSocketRequest(context_id="turn-42", text="hello")
    response = proto.WebSocketResponse(context_id="turn-42", audio=b"\x01\x02")

    assert request.SerializeToString() == b"\x0a\x07turn-42\x22\x05hello"
    assert response.SerializeToString() == b"\x0a\x07turn-42\x22\x02\x01\x02"


def test_v1_decodes_pcmu_to_little_endian_pcm16() -> None:
    from livekit.plugins.rime import _websocket_v1

    decoded = _websocket_v1._decode_audio("audio/pcmu", bytes([0xFF, 0x7F, 0x80, 0x00]))

    assert np.frombuffer(decoded, dtype="<i2").tolist() == [0, 0, 32124, -32124]


@pytest.mark.parametrize("websocket_protocol", ["binary", "json"])
async def test_v1_streams_audio_before_end_and_maps_supported_start_options(
    websocket_protocol: str,
) -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(
            server,
            websocket_protocol=websocket_protocol,
            lang="eng",
            sample_rate=22050,
            time_scale_factor=1.2,
        )
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        try:
            first_event = asyncio.create_task(anext(stream))
            stream.push_text("Hello from ")
            stream.push_text("LiveKit today. Next")
            first = await asyncio.wait_for(first_event, timeout=2)
            assert first.frame.data
            assert first.frame.sample_rate == 22050

            stream.push_text(" sentence.")
            stream.end_input()
            remaining = await _collect(stream)
        finally:
            await stream.aclose()
            await tts.aclose()

    assert remaining[-1].is_final
    assert server.paths == ["/coda/ws"]
    assert server.headers[0]["Authorization"] == "Bearer test-key"
    assert server.headers[0]["Sec-WebSocket-Protocol"] == f"rime.v1.{websocket_protocol}"
    assert server.protocols == [f"rime.v1.{websocket_protocol}"]
    assert _payloads(server) == ["start", "text", "text", "end"]
    assert [request["text"] for request in server.requests if "text" in request] == [
        "Hello from LiveKit today. ",
        "Next sentence. ",
    ]
    start = server.requests[0]["start"]
    assert start == {
        "speaker": "lyra",
        "language": "eng",
        "text": "",
        "audioParameters": {
            "audioFormat": "audio/pcm",
            "samplingRate": 22050,
            "timeScaleFactor": 1.2,
        },
    }


async def test_v1_defaults_to_binary_protocol() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert server.protocols == ["rime.v1.binary"]


@pytest.mark.parametrize("websocket_protocol", ["binary", "json"])
@pytest.mark.parametrize(
    "audio_format",
    [
        "audio/pcm",
        "audio/wav",
        "audio/mpeg",
        "audio/ogg;codecs=opus",
        "audio/webm;codecs=opus",
        "audio/pcmu",
    ],
)
async def test_v1_supports_each_rime_audio_format_with_each_protocol(
    websocket_protocol: str,
    audio_format: str,
    encoded_audio_by_format: dict[str, bytes],
) -> None:
    async with _RimeV1Server(audio=encoded_audio_by_format[audio_format]) as server:
        tts = _v1_tts(
            server,
            websocket_protocol=websocket_protocol,
            audio_format=audio_format,
        )
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        try:
            stream.push_text("hello")
            stream.end_input()
            events = await _collect(stream)
        finally:
            await stream.aclose()
            await tts.aclose()

    assert any(event.frame.data for event in events)
    assert events[-1].is_final
    assert server.requests[0]["start"]["audioParameters"]["audioFormat"] == audio_format
    assert server.protocols == [f"rime.v1.{websocket_protocol}"]


async def test_v1_maps_mist_options_without_a_second_stream_implementation() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(
            server,
            endpoint_model="mist",
            pause_between_brackets=True,
            phonemize_between_brackets=False,
        )
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    start = server.requests[0]["start"]
    assert start["mistParameters"] == {
        "pauseBetweenBrackets": True,
        "phonemizeBetweenBrackets": False,
        "inlineTimeScaleFactors": [],
    }


@pytest.mark.parametrize(
    ("endpoint_model", "expected_sample_rate"),
    [("coda", 24000), ("mist", 24000), ("mistv2", 22050), ("future-model", 22050)],
)
async def test_v1_sends_resolved_sample_rate(
    endpoint_model: str, expected_sample_rate: int
) -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server, endpoint_model=endpoint_model)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        try:
            stream.push_text("hello")
            stream.end_input()
            events = await _collect(stream)
        finally:
            await stream.aclose()
            await tts.aclose()

    assert tts.sample_rate == expected_sample_rate
    assert events[0].frame.sample_rate == expected_sample_rate
    assert server.requests[0]["start"]["audioParameters"] == {
        "audioFormat": "audio/pcm",
        "samplingRate": expected_sample_rate,
    }


async def test_v1_stream_keeps_sample_rate_after_parent_update() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server, sample_rate=22050)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        tts.update_options(sample_rate=16000)
        try:
            stream.push_text("hello")
            stream.end_input()
            events = await _collect(stream)
        finally:
            await stream.aclose()
            await tts.aclose()

    assert server.requests[0]["start"]["audioParameters"]["samplingRate"] == 22050
    assert events[0].frame.sample_rate == 22050


async def test_v1_rejects_wrong_ready_protocol() -> None:
    async with _RimeV1Server(ready={"protocol": 2}) as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIConnectionError, match="unsupported protocol"):
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()


async def test_v1_buffers_fragments_into_complete_sentences() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("This is the first sentence. ")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(server.wait_for_text_messages(1), timeout=0.05)

        stream.push_text("Second")
        await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert [request["text"] for request in server.requests if "text" in request] == [
        "This is the first sentence. ",
        "Second ",
    ]


async def test_v1_default_tokenizer_preserves_fragment_boundaries() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        try:
            stream.push_text("The price is 1.")
            stream.push_text("7 dollars. Hel")
            stream.push_text("lo world. Next")
            stream.push_text(" sentence.")
            stream.end_input()
            await _collect(stream)
        finally:
            await stream.aclose()
            await tts.aclose()

    assert [request["text"] for request in server.requests if "text" in request] == [
        "The price is 1.7 dollars. ",
        "Hello world. ",
        "Next sentence. ",
    ]


async def test_v1_uses_custom_tokenizer_behavior() -> None:
    from livekit.agents import tokenize

    async with _RimeV1Server() as server:
        tokenizer = tokenize.basic.SentenceTokenizer(
            min_sentence_len=1000,
            stream_context_len=1,
        )
        tts = _v1_tts(server, tokenizer=tokenizer)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        try:
            stream.push_text("First sentence. Second sentence.")
            stream.end_input()
            await _collect(stream)
        finally:
            await stream.aclose()
            await tts.aclose()

    assert [request["text"] for request in server.requests if "text" in request] == [
        "First sentence. Second sentence. "
    ]


@pytest.mark.parametrize("websocket_protocol", ["binary", "json"])
@pytest.mark.parametrize("failure_stage", ["stream", "push_text", "output", "end_input"])
async def test_v1_tokenizer_failure_closes_stream(
    websocket_protocol: str, failure_stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from livekit.agents import APIError, tokenize

    tokenizer = tokenize.basic.SentenceTokenizer()
    tokens = tokenizer.stream()

    def fail(*args: Any, **kwargs: Any) -> None:
        raise ValueError(_SECRET)

    async def fail_output(*args: Any, **kwargs: Any) -> None:
        fail()

    monkeypatch.setattr(tokenizer, "stream", lambda **kwargs: tokens)
    if failure_stage == "stream":
        monkeypatch.setattr(tokenizer, "stream", fail)
    elif failure_stage == "output":
        monkeypatch.setattr(type(tokens), "__anext__", fail_output)
    else:
        monkeypatch.setattr(tokens, failure_stage, fail)

    async with _RimeV1Server() as server:
        tts = _v1_tts(server, websocket_protocol=websocket_protocol, tokenizer=tokenizer)
        stream = tts.stream(
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0, timeout=2)
        )
        try:
            stream.push_text("Hello.")
            if failure_stage == "end_input":
                stream.end_input()
            # Other failures must propagate even while input remains open.
            with pytest.raises(APIError, match="Rime sentence tokenization failed") as exc_info:
                await asyncio.wait_for(_collect(stream), timeout=1)
            assert exc_info.value.retryable is False
            _assert_exception_is_safe(exc_info.value)
            if failure_stage != "stream":
                assert tokens.closed
            assert server.connections == 1
        finally:
            await stream.aclose()
            await tts.aclose()
            if not tokens.closed:
                await tokens.aclose()


async def test_v1_local_flush_drains_text_before_end() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("first")
        stream.flush()
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert _payloads(server) == ["start", "text", "end"]


async def test_v1_local_flush_sends_text_without_control_message() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("first")
        stream.flush()
        try:
            await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)
        finally:
            stream.end_input()
            await _collect(stream)
            await stream.aclose()
            await tts.aclose()

    assert _payloads(server) == ["start", "text", "end"]


async def test_v1_resumes_context_after_local_flush() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        metrics = []
        tts.on("metrics_collected", metrics.append)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("first")
        stream.flush()
        await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)
        stream.push_text("second")
        stream.end_input()
        events = await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert _payloads(server) == ["start", "text", "text", "end"]
    assert [request["text"] for request in server.requests if "text" in request] == [
        "first ",
        "second ",
    ]
    assert len({request["contextId"] for request in server.requests}) == 1
    assert sum(event.is_final for event in events) == 1
    assert len(metrics) == 1
    assert metrics[0].characters_count == len("firstsecond")


async def test_v1_input_pause_can_exceed_api_timeout() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.1))
        stream.push_text("First sentence.")
        stream.flush()
        await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)

        await asyncio.sleep(0.2)

        stream.push_text("Second sentence.")
        stream.end_input()
        events = await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert _payloads(server) == ["start", "text", "text", "end"]
    assert len({request["contextId"] for request in server.requests}) == 1
    assert events[-1].is_final


async def test_v1_flush_before_first_text_sends_no_rime_message() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.flush()
        await asyncio.wait_for(server.connection_opened.wait(), timeout=2)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(server.request_received.wait(), timeout=0.05)

        stream.push_text("first")
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert _payloads(server) == ["start", "text", "end"]


async def test_v1_end_input_drains_buffered_final_text_before_end() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("buffered final fragment")
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert _payloads(server) == ["start", "text", "end"]
    assert server.requests[1]["text"] == "buffered final fragment "


async def test_v1_reuses_socket_and_does_not_start_empty_context() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        for text in ("one", "two"):
            stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
            stream.push_text(text)
            stream.end_input()
            await _collect(stream)
            await stream.aclose()

        empty = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        empty.end_input()
        assert await _collect(empty) == []
        await empty.aclose()
        await tts.aclose()

    assert server.connections == 1
    assert server.ready_events == 1
    assert _payloads(server).count("start") == 2


async def test_v1_overlapping_streams_use_separate_connections() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        first = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        first.push_text("First stream stays active. Pending")
        await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)

        second = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        second.push_text("Second stream completes.")
        second.end_input()
        events = await _collect(second)

        await first.aclose()
        await second.aclose()
        await tts.aclose()

    contexts_by_connection: dict[int, set[str]] = {}
    for connection, request in zip(server.request_connections, server.requests, strict=True):
        contexts_by_connection.setdefault(connection, set()).add(request["contextId"])

    assert server.connections == 2
    assert len(contexts_by_connection) == 2
    assert all(len(contexts) == 1 for contexts in contexts_by_connection.values())
    assert events[-1].is_final


async def test_v1_stream_snapshots_options() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        tts.update_options(speaker="changed")
        stream.push_text("hello")
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert server.requests[0]["start"]["speaker"] == "lyra"


async def test_v1_stream_snapshots_websocket_url() -> None:
    async with _RimeV1Server() as first_server, _RimeV1Server() as second_server:
        tts = _v1_tts(first_server)
        first_stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        tts.update_options(websocket_url=second_server.websocket_url)
        second_stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))

        first_stream.push_text("first")
        first_stream.end_input()
        second_stream.push_text("second")
        second_stream.end_input()
        await _collect(first_stream)
        await _collect(second_stream)
        await first_stream.aclose()
        await second_stream.aclose()
        await tts.aclose()

    assert [request["text"] for request in first_server.requests if "text" in request] == ["first "]
    assert [request["text"] for request in second_server.requests if "text" in request] == [
        "second "
    ]


@pytest.mark.parametrize("new_stream_first", [False, True], ids=["old-first", "new-first"])
async def test_v1_stream_metrics_keep_model_after_endpoint_update(
    new_stream_first: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = _DynamicTracer("test-rime-stream-metrics")
    tracer.set_provider(provider)
    monkeypatch.setattr(tts_module, "tracer", tracer)

    async with _RimeV1Server() as first_server, _RimeV1Server() as second_server:
        tts = _v1_tts(first_server)
        metrics = []
        tts.on("metrics_collected", metrics.append)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("first")
        stream.flush()
        new_stream = None
        try:
            await asyncio.wait_for(first_server.wait_for_text_messages(1), timeout=2)
            mist_url = second_server.websocket_url.replace("/coda/ws", "/mist/ws")
            tts.update_options(websocket_url=mist_url)
            new_stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
            new_stream.push_text("second")
            streams = [(stream, "coda"), (new_stream, "mistv3")]
            if new_stream_first:
                streams.reverse()

            for count, (active_stream, model) in enumerate(streams, start=1):
                active_stream.end_input()
                events = await _collect(active_stream)
                await active_stream.aclose()
                assert len(metrics) == count
                assert metrics[-1].request_id == events[0].request_id
                assert metrics[-1].metadata.model_name == model
                assert metrics[-1].metadata.model_provider == tts.provider
                assert metrics[-1].label == tts.label
                assert tts.model == "mistv3"
        finally:
            await stream.aclose()
            if new_stream is not None:
                await new_stream.aclose()
            await tts.aclose()

    spans = [span for span in exporter.get_finished_spans() if span.name == "tts_request"]
    assert len(spans) == 2
    for span, metric in zip(spans, metrics, strict=True):
        assert span.attributes is not None
        assert span.attributes[trace_types.ATTR_TTS_LABEL] == tts.label
        assert json.loads(span.attributes[trace_types.ATTR_TTS_METRICS]) == metric.model_dump()


async def test_v1_omits_retained_time_scale_factor_after_switch_to_mistv2() -> None:
    async with _RimeV1Server() as first_server, _RimeV1Server() as second_server:
        tts = _v1_tts(first_server, time_scale_factor=1.2)
        mistv2_url = second_server.websocket_url.replace("/coda/ws", "/mistv2/ws")
        tts.update_options(websocket_url=mistv2_url)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))

        stream.push_text("hello")
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert second_server.paths == ["/mistv2/ws"]
    assert "timeScaleFactor" not in second_server.requests[0]["start"]["audioParameters"]


@pytest.mark.parametrize("response_mode", ["no_audio", "done_on_cancel"])
@pytest.mark.parametrize("websocket_protocol", ["binary", "json"])
async def test_v1_clean_interruption_cancels_and_reuses_socket(
    response_mode: str, websocket_protocol: str
) -> None:
    async with _RimeV1Server(response_mode=response_mode) as server:
        tts = _v1_tts(server, websocket_protocol=websocket_protocol)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("Please stop this synthesis now. Pending")
        await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)
        await stream.aclose()

        second = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        second.end_input()
        await _collect(second)
        await second.aclose()
        await tts.aclose()

    assert "cancel" in _payloads(server)
    assert server.connections == 1


async def test_v1_interruption_stays_cancelled_without_retry() -> None:
    async with _RimeV1Server(response_mode="no_audio") as server:
        tts = _v1_tts(server)
        stream = tts.stream(
            conn_options=APIConnectOptions(max_retry=1, timeout=0.1, retry_interval=0)
        )
        stream.push_text("Please stop this synthesis now. Pending")
        await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)
        await asyncio.wait_for(stream.aclose(), timeout=0.5)
        await tts.aclose()

    assert _payloads(server).count("start") == 1


@pytest.mark.parametrize("payload", ["start", "text", "end", "cancel"])
@pytest.mark.parametrize("websocket_protocol", ["binary", "json"])
async def test_v1_blocked_write_times_out(
    payload: str, websocket_protocol: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_started = asyncio.Event()
    release_write = asyncio.Event()
    send_method = "send_bytes" if websocket_protocol == "binary" else "send_str"
    original_send = getattr(aiohttp.ClientWebSocketResponse, send_method)

    async def blocked_send(
        websocket: aiohttp.ClientWebSocketResponse, data: bytes | str, **kwargs: Any
    ) -> None:
        request = proto.WebSocketRequest()
        if isinstance(data, bytes):
            request.ParseFromString(data)
        else:
            json_format.Parse(data, request)
        if request.WhichOneof("payload") == payload:
            write_started.set()
            await release_write.wait()
        await original_send(websocket, data, **kwargs)

    monkeypatch.setattr(aiohttp.ClientWebSocketResponse, send_method, blocked_send)
    async with _RimeV1Server(response_mode="no_audio") as server:
        tts = _v1_tts(server, websocket_protocol=websocket_protocol)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.03))
        stream.push_text("Please synthesize this sentence. Pending")
        if payload == "cancel":
            await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)
            task = asyncio.create_task(stream.aclose())
        else:
            stream.end_input()
            task = asyncio.create_task(_collect(stream))

        try:
            await asyncio.wait_for(write_started.wait(), timeout=2)
            done, _ = await asyncio.wait({task}, timeout=0.5)
            assert task in done, f"blocked {payload} write exceeded the configured timeout"
            if payload == "cancel":
                await task
            else:
                with pytest.raises(APITimeoutError):
                    await task
            await asyncio.wait_for(server.connection_closed.wait(), timeout=1)

            # A timed-out write leaves an uncertain context. Do not reuse its socket.
            empty = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
            try:
                empty.end_input()
                assert await _collect(empty) == []
            finally:
                await empty.aclose()
            assert server.connections == 2
        finally:
            release_write.set()
            await stream.aclose()
            await asyncio.gather(task, return_exceptions=True)
            await tts.aclose()


async def test_v1_cancels_when_start_write_is_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.rime import _websocket_v1

    async with _RimeV1Server() as server:
        original_send = _websocket_v1._send_envelope
        start_written = asyncio.Event()
        block_first_start = True

        async def _send_envelope(
            connection: _websocket_v1.Connection,
            context_id: str,
            payload: str,
            value: object,
        ) -> None:
            nonlocal block_first_start
            await original_send(connection, context_id, payload, value)
            if payload == "start" and block_first_start:
                block_first_start = False
                start_written.set()
                await asyncio.Future()

        monkeypatch.setattr(_websocket_v1, "_send_envelope", _send_envelope)
        tts = _v1_tts(server)
        interrupted = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        interrupted.push_text("Please stop this synthesis now. Pending")
        await asyncio.wait_for(start_written.wait(), timeout=2)
        await interrupted.aclose()

        next_stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        next_stream.push_text("next")
        next_stream.end_input()
        events = await _collect(next_stream)
        await next_stream.aclose()
        await tts.aclose()

    assert events[-1].is_final
    assert "cancel" in _payloads(server)
    assert server.connections == 1


async def test_v1_closes_socket_when_cancel_has_no_reply() -> None:
    async with _RimeV1Server(response_mode="no_audio", cancel_reply=False) as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.1))
        stream.push_text("Please stop this synthesis now. Pending")
        await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)
        await stream.aclose()

        second = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        second.end_input()
        await _collect(second)
        await second.aclose()
        await tts.aclose()

    assert server.connections == 2


async def test_v1_end_times_out_when_terminal_event_never_arrives() -> None:
    async with _RimeV1Server(response_mode="no_done") as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.1))
        stream.push_text("The server will not finish this context.")
        stream.end_input()
        with pytest.raises(APITimeoutError, match="after end"):
            await _collect(stream)
        await stream.aclose()

        empty = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        empty.end_input()
        assert await _collect(empty) == []
        await empty.aclose()
        await tts.aclose()

    assert server.connections == 2


@pytest.mark.parametrize("websocket_protocol", ["binary", "json"])
async def test_v1_rejects_done_before_input_ends(websocket_protocol: str) -> None:
    async with _RimeV1Server(response_mode="early_done") as server:
        tts = _v1_tts(server, websocket_protocol=websocket_protocol)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        try:
            stream.push_text("First sentence. Pending")
            with pytest.raises(APIConnectionError, match="done before input ended"):
                await asyncio.wait_for(_collect(stream), timeout=2)
            await asyncio.wait_for(server.connection_closed.wait(), timeout=2)
            assert _payloads(server) == ["start", "text"]

            server.response_mode = "normal"
            second = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
            try:
                second.push_text("Next turn.")
                second.end_input()
                assert await asyncio.wait_for(_collect(second), timeout=2)
            finally:
                await second.aclose()
            assert server.connections == 2
        finally:
            await stream.aclose()
            await tts.aclose()


@pytest.mark.parametrize("websocket_protocol", ["binary", "json"])
async def test_v1_accepts_done_before_end_write_returns(
    websocket_protocol: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from livekit.plugins.rime import _websocket_v1

    done_received = asyncio.Event()
    end_write_pending = False
    original_send = _websocket_v1._send_envelope
    codec_class = type(_websocket_v1._codec_for_protocol(websocket_protocol))
    original_decode = codec_class.decode_response

    async def send_envelope(
        connection: _websocket_v1.Connection, context_id: str, payload: str, value: object
    ) -> None:
        nonlocal end_write_pending
        if payload == "end":
            end_write_pending = True
        await original_send(connection, context_id, payload, value)
        if payload == "end":
            await done_received.wait()
            end_write_pending = False

    def decode_response(self: Any, message: aiohttp.WSMessage) -> proto.WebSocketResponse:
        response = original_decode(self, message)
        if response.WhichOneof("payload") == "done":
            assert end_write_pending
            done_received.set()
        return response

    monkeypatch.setattr(_websocket_v1, "_send_envelope", send_envelope)
    monkeypatch.setattr(codec_class, "decode_response", decode_response)
    async with _RimeV1Server() as server:
        tts = _v1_tts(server, websocket_protocol=websocket_protocol)
        try:
            for _ in range(2):
                done_received.clear()
                stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
                try:
                    stream.push_text("Hello.")
                    stream.end_input()
                    assert await asyncio.wait_for(_collect(stream), timeout=2)
                    assert done_received.is_set()
                finally:
                    await stream.aclose()
            assert server.connections == 1
        finally:
            await tts.aclose()


async def test_v1_rejects_malformed_done_before_socket_reuse() -> None:
    async with _RimeV1Server(response_mode="malformed_done") as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIConnectionError, match="exactly one payload"):
            await _collect(stream)
        await stream.aclose()

        empty = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        empty.end_input()
        await _collect(empty)
        await empty.aclose()
        await tts.aclose()

    assert server.connections == 2


async def test_v1_rejects_malformed_cancelled_before_socket_reuse() -> None:
    async with _RimeV1Server(response_mode="malformed_cancelled") as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("Please stop this synthesis now. Pending")
        await asyncio.wait_for(server.wait_for_text_messages(1), timeout=2)
        await stream.aclose()

        empty = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        empty.end_input()
        await _collect(empty)
        await empty.aclose()
        await tts.aclose()

    assert server.connections == 2


@pytest.mark.parametrize(
    "response_mode",
    ["wrong_context", "invalid_envelope", "wrong_frame"],
)
async def test_v1_rejects_contaminated_responses(response_mode: str) -> None:
    async with _RimeV1Server(response_mode=response_mode) as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIConnectionError):
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()


async def test_v1_json_rejects_invalid_base64_audio() -> None:
    async with _RimeV1Server(response_mode="invalid_base64") as server:
        tts = _v1_tts(server, websocket_protocol="json")
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIConnectionError, match="invalid Base64 audio"):
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()


async def test_v1_json_rejects_invalid_done_type() -> None:
    async with _RimeV1Server(response_mode="invalid_done_type") as server:
        tts = _v1_tts(server, websocket_protocol="json")
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIConnectionError, match="malformed done event"):
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()


@pytest.mark.parametrize("websocket_protocol", ["binary", "json"])
async def test_v1_rejects_started_without_request_id(websocket_protocol: str) -> None:
    async with _RimeV1Server(response_mode="missing_started_request_id") as server:
        tts = _v1_tts(server, websocket_protocol=websocket_protocol)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIConnectionError, match="malformed started event"):
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert server.closed_connections == 1


@pytest.mark.parametrize(
    ("kind", "status_code", "retryable"),
    [
        ("invalid_input", 400, False),
        ("unauthenticated", 401, False),
        ("permission_denied", 403, False),
        ("not_found", 404, False),
        ("resource_exhausted", 429, True),
        ("timeout", 504, True),
        ("unavailable", 503, True),
        ("unimplemented", 501, False),
        ("internal", 500, True),
    ],
)
async def test_v1_maps_context_error(kind: str, status_code: int, retryable: bool) -> None:
    async with _RimeV1Server(response_mode="error", error_kind=kind) as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIStatusError) as exc_info:
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert exc_info.value.status_code == status_code
    assert exc_info.value.retryable is retryable


async def test_v1_maps_connection_scoped_error() -> None:
    async with _RimeV1Server(response_mode="connection_error", error_kind="unavailable") as server:
        tts = _v1_tts(server)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIStatusError) as exc_info:
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert exc_info.value.status_code == 503


@pytest.mark.parametrize("websocket_protocol", ["binary", "json"])
@pytest.mark.parametrize("message", [None, ""])
@pytest.mark.parametrize(
    "kind, status_code, retryable", [("unavailable", 503, True), ("unimplemented", 501, False)]
)
@pytest.mark.parametrize("explicit_request_id", [False, True])
async def test_v1_maps_error_without_message(
    websocket_protocol: str,
    message: str | None,
    kind: str,
    status_code: int,
    retryable: bool,
    explicit_request_id: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _RimeV1Server(response_mode="error") as server:

        async def respond(ws: web.WebSocketResponse, context_id: str) -> None:
            error = {"kind": kind}
            if message is not None:
                error["message"] = message
            if explicit_request_id:
                error["requestId"] = "provider-request"
            await server._send(ws, {"contextId": context_id, "error": error})

        monkeypatch.setattr(server, "_respond_to_text", respond)
        tts = _v1_tts(server, websocket_protocol=websocket_protocol)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        try:
            stream.push_text("Hello.")
            stream.end_input()
            with pytest.raises(APIStatusError) as exc_info:
                await _collect(stream)
            assert exc_info.value.status_code == status_code
            assert exc_info.value.retryable is retryable
            assert exc_info.value.request_id == (
                "provider-request" if explicit_request_id else "request-0"
            )
            assert exc_info.value.message == "Rime v1 request failed"
        finally:
            await stream.aclose()
            await tts.aclose()


async def test_v1_does_not_retry_unimplemented_error() -> None:
    async with _RimeV1Server(response_mode="error", error_kind="unimplemented") as server:
        tts = _v1_tts(server)
        stream = tts.stream(
            conn_options=APIConnectOptions(max_retry=1, timeout=2, retry_interval=0)
        )
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIStatusError) as exc_info:
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert exc_info.value.status_code == 501
    assert exc_info.value.retryable is False
    assert server.connections == 1


async def test_v1_retries_before_audio() -> None:
    async with _RimeV1Server(fail_before_audio=1) as server:
        tts = _v1_tts(server)
        stream = tts.stream(
            conn_options=APIConnectOptions(max_retry=1, timeout=2, retry_interval=0)
        )
        stream.push_text("hello")
        stream.end_input()
        events = await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert events[-1].is_final
    assert server.connections == 2


async def test_v1_does_not_retry_after_partial_audio() -> None:
    async with _RimeV1Server(response_mode="partial_error") as server:
        tts = _v1_tts(server)
        stream = tts.stream(
            conn_options=APIConnectOptions(max_retry=1, timeout=2, retry_interval=0)
        )
        stream.push_text("hello")
        stream.end_input()
        with pytest.raises(APIStatusError):
            await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert server.connections == 1


async def test_v1_ready_error_does_not_expose_provider_event() -> None:
    async with _RimeV1Server(ready={"protocol": _SECRET, "providerData": _SECRET}) as server:
        tts = _v1_tts(server, websocket_protocol="json")
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("hello")
        stream.end_input()
        try:
            with pytest.raises(APIConnectionError) as exc_info:
                await _collect(stream)
            _assert_exception_is_safe(exc_info.value)
        finally:
            await stream.aclose()
            await tts.aclose()


def test_v1_context_mismatch_does_not_expose_provider_value() -> None:
    from livekit.plugins.rime import _websocket_v1

    response = proto.WebSocketResponse(context_id=_SECRET)
    with pytest.raises(APIConnectionError, match="unexpected contextId") as exc_info:
        _websocket_v1._check_context(response, "expected-context")

    _assert_exception_is_safe(exc_info.value)


@pytest.mark.parametrize(
    "websocket_url",
    [
        "wss://api.rime.ai/coda/ws",
        "ws://127.0.0.1:8080/coda/ws",
        "ws://[::1]:8080/coda/ws",
    ],
)
def test_v1_accepts_secure_or_loopback_websocket_url(websocket_url: str) -> None:
    from livekit.plugins.rime import _websocket_v1

    _websocket_v1.validate_websocket_url(websocket_url)


def test_v1_rejects_untrusted_secure_host_without_opt_in() -> None:
    from livekit.plugins.rime import _websocket_v1

    with pytest.raises(ValueError, match="trusted Rime host"):
        _websocket_v1.validate_websocket_url("wss://attacker.example/coda/ws")


def test_v1_accepts_custom_secure_host_with_opt_in() -> None:
    from livekit.plugins.rime import _websocket_v1

    _websocket_v1.validate_websocket_url(
        "wss://voice.customer.example/coda/ws", allow_custom_endpoint=True
    )


def test_v1_accepts_dedicated_rime_subdomain() -> None:
    from livekit.plugins.rime import _websocket_v1

    _websocket_v1.validate_websocket_url(
        "wss://tigerstripe-dialpad.aws-us-east-1.whiteglove.rime.ai/ws"
    )


def test_v1_rejects_lookalike_rime_host() -> None:
    from livekit.plugins.rime import _websocket_v1

    with pytest.raises(ValueError, match="trusted Rime host"):
        _websocket_v1.validate_websocket_url("wss://whiteglove.rime.ai.attacker.example/coda/ws")


@pytest.mark.parametrize(
    ("websocket_url", "model"),
    [
        ("wss://api.rime.ai/coda/ws", "coda"),
        ("wss://api.rime.ai/mist/ws", "mist"),
        ("wss://api.rime.ai/future-model/ws/?token=value", "future-model"),
    ],
)
def test_v1_reads_model_from_websocket_url(websocket_url: str, model: str) -> None:
    from livekit.plugins.rime import _websocket_v1

    assert _websocket_v1.model_from_websocket_url(websocket_url) == model


def test_v1_dedicated_websocket_url_has_no_embedded_model() -> None:
    from livekit.plugins.rime import _websocket_v1

    assert (
        _websocket_v1.model_from_websocket_url(
            "wss://tigerstripe-dialpad.aws-us-east-1.whiteglove.rime.ai/ws"
        )
        is None
    )


@pytest.mark.parametrize(
    "websocket_url",
    [
        "wss://api.rime.ai/coda",
        "wss://api.rime.ai/coda/stream",
    ],
)
def test_v1_rejects_url_not_ending_in_ws(websocket_url: str) -> None:
    from livekit.plugins.rime import _websocket_v1

    with pytest.raises(ValueError, match="end with /ws"):
        _websocket_v1.model_from_websocket_url(websocket_url)


@pytest.mark.parametrize(
    "websocket_url",
    [
        "ws://api.rime.ai/coda/ws",
        "ws://192.168.1.20/coda/ws",
        "ws://localhost:8080/coda/ws",
        "http://api.rime.ai/coda/ws",
    ],
)
def test_v1_rejects_insecure_remote_websocket_url(websocket_url: str) -> None:
    from livekit.plugins.rime import _websocket_v1

    with pytest.raises(ValueError):
        _websocket_v1.validate_websocket_url(websocket_url)


async def test_v1_connection_error_does_not_expose_transport_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.rime import _websocket_v1

    async def _fail_connect(*args: Any, **kwargs: Any) -> aiohttp.ClientWebSocketResponse:
        raise RuntimeError(f"credential-bearing transport error: {_SECRET}")

    monkeypatch.setattr(aiohttp.ClientSession, "ws_connect", _fail_connect)
    async with aiohttp.ClientSession() as session:
        with pytest.raises(APIConnectionError) as exc_info:
            await _websocket_v1.connect(
                session,
                websocket_url="wss://api.rime.ai/coda/ws",
                api_key="test-key",
                protocol="binary",
                timeout=1,
            )

    _assert_exception_is_safe(exc_info.value)
    assert exc_info.value.__cause__ is None


async def test_v1_connect_rejects_untrusted_host_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.rime import _websocket_v1

    transport_called = False

    async def _connect(*args: Any, **kwargs: Any) -> aiohttp.ClientWebSocketResponse:
        nonlocal transport_called
        transport_called = True
        raise AssertionError("transport must not receive credentials")

    monkeypatch.setattr(aiohttp.ClientSession, "ws_connect", _connect)
    async with aiohttp.ClientSession() as session:
        with pytest.raises(ValueError, match="trusted Rime host"):
            await _websocket_v1.connect(
                session,
                websocket_url="wss://attacker.example/coda/ws",
                api_key="test-key",
                protocol="binary",
                timeout=1,
            )

    assert transport_called is False


@pytest.mark.parametrize(
    ("payload", "value"),
    [
        ("start", proto.SynthesisRequest(text="")),
        ("text", "hello"),
        ("end", None),
    ],
)
async def test_v1_wraps_write_failures_as_safe_api_errors(
    payload: str,
    value: object,
) -> None:
    from livekit.plugins.rime import _websocket_v1

    class _FailingWebSocket:
        async def send_str(self, data: str) -> None:
            raise ConnectionResetError(f"write failed with {_SECRET}")

    connection = _websocket_v1.Connection(
        websocket=cast(aiohttp.ClientWebSocketResponse, _FailingWebSocket()),
        codec=_websocket_v1._JsonEnvelopeCodec(),
    )
    with pytest.raises(APIConnectionError) as exc_info:
        await _websocket_v1._send_envelope(connection, "context", payload, value)

    _assert_exception_is_safe(exc_info.value)
    assert exc_info.value.retryable is True
    assert exc_info.value.__cause__ is None


async def test_v1_retries_after_write_failure_before_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_send_str = aiohttp.ClientWebSocketResponse.send_str
    fail_first_text = True

    async def _send_str(
        self: aiohttp.ClientWebSocketResponse,
        data: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        nonlocal fail_first_text
        if fail_first_text and "text" in json.loads(data):
            fail_first_text = False
            raise ConnectionResetError(f"write failed with {_SECRET}")
        await original_send_str(self, data, *args, **kwargs)

    monkeypatch.setattr(aiohttp.ClientWebSocketResponse, "send_str", _send_str)

    async with _RimeV1Server() as server:
        tts = _v1_tts(server, websocket_protocol="json")
        errors: list[Any] = []
        tts.on("error", errors.append)
        stream = tts.stream(
            conn_options=APIConnectOptions(max_retry=1, timeout=2, retry_interval=0)
        )
        stream.push_text("hello")
        stream.end_input()
        try:
            events = await _collect(stream)
        finally:
            await stream.aclose()
            await tts.aclose()

    assert events[-1].is_final
    assert server.connections == 2
    assert len(errors) == 1
    assert errors[0].label == tts.label
    assert isinstance(errors[0].error, APIConnectionError)
    assert errors[0].recoverable is True
    _assert_exception_is_safe(errors[0].error)


async def test_v1_websocket_error_does_not_expose_transport_data() -> None:
    from livekit.plugins.rime import _websocket_v1

    class _ErrorWebSocket:
        async def receive(self, *, timeout: float) -> aiohttp.WSMessage:
            return aiohttp.WSMessage(aiohttp.WSMsgType.ERROR, None, None)

        def exception(self) -> BaseException:
            return RuntimeError(f"socket failed with {_SECRET}")

    connection = _websocket_v1.Connection(
        websocket=cast(aiohttp.ClientWebSocketResponse, _ErrorWebSocket()),
        codec=_websocket_v1._JsonEnvelopeCodec(),
    )
    with pytest.raises(APIConnectionError) as exc_info:
        await _websocket_v1._receive_envelope(connection, timeout=1)

    _assert_exception_is_safe(exc_info.value)


async def test_v1_invalid_json_does_not_retain_provider_frame() -> None:
    from livekit.plugins.rime import _websocket_v1

    class _InvalidJsonWebSocket:
        async def receive(self, *, timeout: float) -> aiohttp.WSMessage:
            return aiohttp.WSMessage(
                aiohttp.WSMsgType.TEXT,
                f'{{"providerData":"{_SECRET}"',
                None,
            )

    connection = _websocket_v1.Connection(
        websocket=cast(aiohttp.ClientWebSocketResponse, _InvalidJsonWebSocket()),
        codec=_websocket_v1._JsonEnvelopeCodec(),
    )
    with pytest.raises(APIConnectionError) as exc_info:
        await _websocket_v1._receive_envelope(connection, timeout=1)

    _assert_exception_is_safe(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    "error",
    [
        proto.WebSocketError(kind="invalid_input", message=_SECRET),
        proto.WebSocketError(kind=_SECRET, message="provider failure"),
        proto.WebSocketError(kind="invalid_input", message=""),
    ],
)
def test_v1_error_mapping_does_not_expose_provider_payload(
    error: proto.WebSocketError,
) -> None:
    from livekit.plugins.rime import _websocket_v1

    exc = _websocket_v1._rime_error(error, fallback_request_id="request-id")

    _assert_exception_is_safe(exc)
    assert _SECRET not in repr(exc.body)


async def test_v1_closes_idle_retired_pools_after_url_changes() -> None:
    async with (
        _RimeV1Server() as first_server,
        _RimeV1Server() as second_server,
        _RimeV1Server() as third_server,
    ):
        tts = _v1_tts(first_server)
        try:
            tts.prewarm()
            await asyncio.wait_for(first_server.connection_opened.wait(), timeout=2)

            tts.update_options(websocket_url=second_server.websocket_url)
            await asyncio.wait_for(first_server.connection_closed.wait(), timeout=1)

            tts.prewarm()
            await asyncio.wait_for(second_server.connection_opened.wait(), timeout=2)

            tts.update_options(websocket_url=third_server.websocket_url)
            await asyncio.wait_for(second_server.connection_closed.wait(), timeout=1)

            assert first_server.closed_connections == 1
            assert second_server.closed_connections == 1
        finally:
            await tts.aclose()


@pytest.mark.parametrize("stream_count", [1, 2])
async def test_v1_closes_retired_pool_after_its_last_stream_finishes(stream_count: int) -> None:
    async with _RimeV1Server() as first_server, _RimeV1Server() as second_server:
        tts = _v1_tts(first_server)
        streams = [
            tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
            for _ in range(stream_count)
        ]
        tts.update_options(websocket_url=second_server.websocket_url)

        try:
            for stream in streams:
                assert not first_server.connection_closed.is_set()
                stream.push_text("first")
                stream.end_input()
                events = await _collect(stream)
                assert events[-1].is_final
            await asyncio.wait_for(first_server.connection_closed.wait(), timeout=1)
        finally:
            for stream in streams:
                await stream.aclose()
            await tts.aclose()

    assert first_server.closed_connections == stream_count
    assert second_server.connections == 0


# Fixed wire messages detect protocol changes independently of the installed schema.
_FIXTURES = [
    {
        "message": "WebSocketRequest",
        "binary": "0a036374781a2f1203656e671a046c756e61220648656c6c6f2e2a140a09617564696f2f70636d10c0bb011d0000a03f300242024000",
        "json": {
            "contextId": "ctx",
            "start": {
                "language": "eng",
                "speaker": "luna",
                "text": "Hello.",
                "audioParameters": {
                    "audioFormat": "audio/pcm",
                    "samplingRate": 24000,
                    "timeScaleFactor": 1.25,
                },
                "splitStrategy": "SPLIT_STRATEGY_NONE",
                "codaParameters": {"textLookaheadTokens": 0},
            },
        },
    },
    {
        "message": "WebSocketRequest",
        "binary": "0a036374781a2c12001a002a120a09617564696f2f70636d10001d0000000030004a10080010011a080000003f0000803f2000",
        "json": {
            "contextId": "ctx",
            "start": {
                "language": "",
                "speaker": "",
                "audioParameters": {
                    "audioFormat": "audio/pcm",
                    "samplingRate": 0,
                    "timeScaleFactor": 0.0,
                },
                "splitStrategy": "SPLIT_STRATEGY_UNSPECIFIED",
                "mistParameters": {
                    "pauseBetweenBrackets": False,
                    "phonemizeBetweenBrackets": True,
                    "inlineTimeScaleFactors": [0.5, 1.0],
                    "saveOovs": False,
                },
            },
        },
    },
    {
        "message": "WebSocketRequest",
        "binary": "0a036374781a00",
        "json": {"contextId": "ctx", "start": {}},
    },
    {
        "message": "WebSocketRequest",
        "binary": "12190a0c4170692d4b6579207465737412027b7d1a051203656e67",
        "json": {
            "config": {
                "authorization": "Api-Key test",
                "license": "{}",
                "defaults": {"language": "eng"},
            }
        },
    },
    {
        "message": "WebSocketRequest",
        "binary": "0a03637478220648656c6c6f2e",
        "json": {"contextId": "ctx", "text": "Hello."},
    },
    {
        "message": "WebSocketRequest",
        "binary": "0a036374782200",
        "json": {"contextId": "ctx", "text": ""},
    },
    {
        "message": "WebSocketRequest",
        "binary": "0a036374782a00",
        "json": {"contextId": "ctx", "end": {}},
    },
    {
        "message": "WebSocketRequest",
        "binary": "0a036374783200",
        "json": {"contextId": "ctx", "cancel": {}},
    },
    {
        "message": "WebSocketResponse",
        "binary": "121108011203656e6712037370611a03656e67",
        "json": {"ready": {"protocol": 1, "languages": ["eng", "spa"], "defaultLanguage": "eng"}},
    },
    {
        "message": "WebSocketResponse",
        "binary": "120408011a00",
        "json": {"ready": {"protocol": 1, "defaultLanguage": ""}},
    },
    {"message": "WebSocketResponse", "binary": "12020801", "json": {"ready": {"protocol": 1}}},
    {
        "message": "WebSocketResponse",
        "binary": "0a036374781a050a03726571",
        "json": {"contextId": "ctx", "started": {"requestId": "req"}},
    },
    {
        "message": "WebSocketResponse",
        "binary": "0a0363747822040001feff",
        "json": {"contextId": "ctx", "audio": "AAH+/w=="},
    },
    {
        "message": "WebSocketResponse",
        "binary": "0a036374782200",
        "json": {"contextId": "ctx", "audio": ""},
    },
    {
        "message": "WebSocketResponse",
        "binary": "0a036374782a00",
        "json": {"contextId": "ctx", "done": {}},
    },
    {
        "message": "WebSocketResponse",
        "binary": "0a036374783200",
        "json": {"contextId": "ctx", "cancelled": {}},
    },
    {
        "message": "WebSocketResponse",
        "binary": "0a036374783a200a0d696e76616c69645f696e707574120a74657374206572726f721a03726571",
        "json": {
            "contextId": "ctx",
            "error": {"kind": "invalid_input", "message": "test error", "requestId": "req"},
        },
    },
    {
        "message": "WebSocketResponse",
        "binary": "0a036374783a0c0a08696e7465726e616c1a00",
        "json": {"contextId": "ctx", "error": {"kind": "internal", "requestId": ""}},
    },
]


@pytest.mark.parametrize("fixture", _FIXTURES)
@pytest.mark.parametrize("protocol", ["binary", "json"])
async def test_fixed_wire_envelope(fixture: dict[str, Any], protocol: str) -> None:
    codec = _codec_for_protocol(protocol)
    binary = bytes.fromhex(fixture["binary"])
    expected = fixture["json"]
    if fixture["message"] == "WebSocketRequest":
        request = json_format.ParseDict(expected, proto.WebSocketRequest())
        websocket = AsyncMock(spec=aiohttp.ClientWebSocketResponse)
        await codec.send_request(websocket, request)
        if protocol == "binary":
            websocket.send_bytes.assert_awaited_once_with(binary)
        else:
            websocket.send_str.assert_awaited_once()
            assert json.loads(websocket.send_str.call_args.args[0]) == expected
        assert json_format.MessageToDict(proto.WebSocketRequest.FromString(binary)) == expected
        if request.HasField("start"):
            assert not request.start.HasField("arcana_parameters")
    else:
        message = aiohttp.WSMessage(
            aiohttp.WSMsgType.BINARY if protocol == "binary" else aiohttp.WSMsgType.TEXT,
            binary if protocol == "binary" else json.dumps(expected),
            "",
        )
        response = codec.decode_response(message)
        assert json_format.MessageToDict(response) == expected
        assert response.SerializeToString() == binary
        assert response.WhichOneof("payload") == next(key for key in expected if key != "contextId")


@pytest.mark.parametrize(
    ("mode", "status", "raise_for_status"),
    [
        ("http", 400, False),
        ("http", 500, False),
        ("handshake", 400, False),
        ("handshake", 500, False),
        ("handshake", 400, True),
        ("handshake", 500, True),
        ("transport", 302, False),
    ],
)
async def test_rime_errors_do_not_expose_provider_or_transport_details(
    mode: str,
    status: int,
    raise_for_status: bool,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = 0

    async def fail_request(request: web.Request) -> web.Response:
        nonlocal requests
        requests += 1
        if mode == "transport":
            # aiohttp rejects this redirect and includes the URL in its exception.
            return web.Response(
                status=302, headers={"Location": f"ftp://localhost/file?token={_SECRET}"}
            )
        return web.Response(status=status, reason=_SECRET, text=_SECRET)

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = _DynamicTracer("test-rime-error-safety")
    tracer.set_provider(provider)
    monkeypatch.setattr(tts_module, "tracer", tracer)
    # The plugin must produce safe exceptions even when telemetry redaction is off.
    monkeypatch.setattr(tts_module.telemetry_utils, "redaction_enabled", lambda *_: False)
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    app = web.Application()
    app.router.add_route("*", "/coda/ws", fail_request)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{runner.addresses[0][1]}/coda/ws"
    conn_options = APIConnectOptions(max_retry=1, retry_interval=0, timeout=1)

    try:
        async with aiohttp.ClientSession(raise_for_status=raise_for_status) as session:
            if mode == "handshake":
                tts = TTS(
                    api_key="test-key",
                    websocket_url=url.replace("http:", "ws:"),
                    http_session=session,
                )
                stream = tts.stream(conn_options=conn_options)
                stream.push_text("hello")
                stream.end_input()
            else:
                tts = TTS(api_key="test-key", base_url=url, http_session=session)
                stream = tts.synthesize("hello", conn_options=conn_options)
            try:
                with pytest.raises(APIError) as exc_info:
                    _ = [event async for event in stream]
            finally:
                await stream.aclose()
                await tts.aclose()
    finally:
        await runner.cleanup()
        provider.shutdown()

    error = exc_info.value
    expected_retryable = mode == "transport" or status == 500
    if mode == "transport":
        assert isinstance(error, APIConnectionError)
    else:
        assert isinstance(error, APIStatusError)
        assert error.status_code == status
        assert error.request_id is None
    assert error.body is None
    assert error.retryable is expected_retryable
    assert requests == (2 if expected_retryable else 1)
    assert _SECRET not in str(error)
    assert _SECRET not in repr(error)
    assert _SECRET not in "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    assert error.__cause__ is None
    assert _SECRET not in caplog.text
    retry_logs = [record for record in caplog.records if "retrying in" in record.getMessage()]
    assert len(retry_logs) == (1 if expected_retryable else 0)
    spans = exporter.get_finished_spans()
    attempts = [span for span in spans if span.name == "tts_request_run"]
    assert len(attempts) == requests
    assert all(span.events for span in attempts)
    assert all(_SECRET not in span.to_json() for span in spans)
