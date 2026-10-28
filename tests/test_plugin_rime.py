"""Rime TTS, optional imports, and Prism realtime regression tests."""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import json
import logging
import subprocess
import sys
import textwrap
import traceback
from contextlib import asynccontextmanager
from importlib.metadata import version
from importlib.util import find_spec
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
from packaging.version import Version
from rime_api import text_to_speech_pb2 as proto

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AgentTask,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    RunContext,
    function_tool,
    llm,
)
from livekit.agents.llm import RealtimeModelFallbackAdapter
from livekit.agents.telemetry import trace_types, utils as telemetry_utils
from livekit.agents.telemetry.traces import _DynamicTracer
from livekit.agents.tts import tts as tts_module
from livekit.plugins import rime
from livekit.plugins.rime import TTS
from livekit.plugins.rime._websocket_v1 import _codec_for_protocol

from .fake_io import FakeAudioOutput

pytestmark = pytest.mark.unit

HAS_REALTIME_SDK = find_spec("rimelabs_sdk") is not None
if HAS_REALTIME_SDK:
    from rimelabs_sdk import Rime, RimeInputError, realtime as r
    from websockets.asyncio.server import serve

    from livekit.plugins.rime.realtime import (
        PrismError,
        PrismGenerationCreatedEvent,
        PrismModelError,
        PrismSpeechStoppedEvent,
        PrismUnsupported,
    )


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
            {"websocket_url": "wss://coda.api.rime.ai/ws", "model": "coda"},
            "coda",
            "lyra",
            id="v1-public-coda",
        ),
        pytest.param(
            {"websocket_url": "wss://mist.api.rime.ai/ws", "model": "mistv3"},
            "mistv3",
            "cove",
            id="v1-public-mist",
        ),
        pytest.param({"model": "mistv3"}, "mistv3", "cove", id="http-mist"),
        pytest.param({"use_websocket": True, "model": "mistv3"}, "mistv3", "cove", id="ws3-mist"),
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


_LEXICON = [
    {"spelling": "expedite", "pronunciation": '" E k . s p @ . % d aI t'},
    {"spelling": "kubectl", "pronunciation": '" k u b . k @ n . % t r\\ oU l'},
]


async def test_chunked_stream_sends_and_clears_coda_custom_lexicon() -> None:
    from livekit.plugins.rime import TTS

    payloads: list[dict[str, Any]] = []

    async def synthesize(request: web.Request) -> web.Response:
        payloads.append(await request.json())
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
                custom_lexicon=_LEXICON,
            )
            conn_options = APIConnectOptions(max_retry=0, timeout=2)
            async with tts.synthesize("expedite it", conn_options=conn_options) as stream:
                [event async for event in stream]
            tts.update_options(custom_lexicon=[])
            async with tts.synthesize("expedite it", conn_options=conn_options) as stream:
                [event async for event in stream]
            await tts.aclose()
    finally:
        await runner.cleanup()

    assert payloads[0]["customLexicon"] == _LEXICON
    assert "customLexicon" not in payloads[1]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": "mistv3"}, "custom_lexicon requires the coda model"),
        ({"model": "coda", "use_websocket": True}, "legacy Rime WebSocket"),
        (
            {"model": "mistv3", "websocket_url": "wss://mistv3.api.rime.ai/ws"},
            "custom_lexicon requires the coda model",
        ),
    ],
)
def test_custom_lexicon_rejects_unsupported_configuration(
    kwargs: dict[str, Any], message: str
) -> None:
    from livekit.plugins.rime import TTS

    with pytest.raises(ValueError, match=message):
        TTS(api_key="test-key", custom_lexicon=_LEXICON, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": "mistv3"}, "custom_lexicon requires the coda model"),
        ({"model": "coda", "use_websocket": True}, "legacy Rime WebSocket"),
    ],
)
def test_custom_lexicon_update_rejects_unsupported_configuration(
    kwargs: dict[str, Any], message: str
) -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key", **kwargs)

    with pytest.raises(ValueError, match=message):
        tts.update_options(custom_lexicon=_LEXICON)


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
        model="coda",
        websocket_url="wss://coda.api.rime.ai/ws",
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


def test_websocket_url_uses_explicit_mist_model_and_accepts_options() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(
        api_key="test-key",
        websocket_url="wss://mist.api.rime.ai/ws",
        model="mistv3",
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
        websocket_url="wss://dedicated.rime.ai/ws",
        model="coda",
    )

    assert tts.model == "coda"


@pytest.mark.parametrize(
    "websocket_url",
    [
        "wss://coda.api.rime.ai/ws",
        "wss://mist.api.rime.ai:443/ws",
        "wss://dedicated.rime.ai/ws",
        "wss://api.rime.ai/coda/ws",
        "wss://api.rime.ai/mist/ws",
        "wss://voice.rime.ai/custom/stream",
    ],
)
def test_websocket_url_requires_model(websocket_url: str) -> None:
    from livekit.plugins.rime import TTS

    with pytest.raises(ValueError, match="model is required"):
        TTS(
            api_key="test-key",
            websocket_url=websocket_url,
        )


def test_custom_websocket_endpoint_requires_explicit_opt_in() -> None:
    from livekit.plugins.rime import TTS

    with pytest.raises(ValueError, match="trusted Rime host"):
        TTS(
            api_key="test-key",
            model="coda",
            websocket_url="wss://voice.customer.example/coda/ws",
        )

    tts = TTS(
        api_key="test-key",
        model="coda",
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
        TTS(api_key="test-key", model="coda", websocket_url=websocket_url)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "websocket_url": "wss://coda.api.rime.ai/ws",
                "base_url": "https://users.rime.ai/coda",
            },
            "cannot be used with base_url",
        ),
        (
            {
                "websocket_url": "wss://coda.api.rime.ai/ws",
                "use_websocket": True,
            },
            "omit use_websocket",
        ),
        (
            {
                "websocket_url": "wss://coda.api.rime.ai/ws",
                "speed_alpha": 1.1,
            },
            "speed_alpha",
        ),
        (
            {
                "websocket_url": "wss://coda.api.rime.ai/ws",
                "temperature": 0.7,
            },
            "generation controls",
        ),
        (
            {
                "websocket_url": "wss://coda.api.rime.ai/ws",
                "websocket_protocol": "auto",
            },
            "binary.*json",
        ),
        (
            {
                "websocket_url": "wss://coda.api.rime.ai/ws",
                "audio_format": "audio/aac",
            },
            "unsupported Rime audio_format",
        ),
        (
            {
                "websocket_url": "wss://coda.api.rime.ai/ws",
                "pause_between_brackets": True,
            },
            "Mist options",
        ),
    ],
)
def test_v1_rejects_invalid_configuration(kwargs: dict[str, Any], message: str) -> None:
    from livekit.plugins.rime import TTS

    with pytest.raises(ValueError, match=message):
        TTS(api_key="test-key", model="coda", **kwargs)


def test_v1_rejects_generation_controls_on_update() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key", model="coda", websocket_url="wss://coda.api.rime.ai/ws")

    with pytest.raises(ValueError, match="generation controls"):
        tts.update_options(top_p=0.8)


def test_audio_format_requires_v1_websocket() -> None:
    from livekit.plugins.rime import TTS

    with pytest.raises(ValueError, match="only supported with the Rime v1"):
        TTS(api_key="test-key", audio_format="audio/wav")


def test_v1_updates_audio_format() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key", model="coda", websocket_url="wss://coda.api.rime.ai/ws")
    tts.update_options(audio_format="audio/mpeg")

    assert tts._opts.audio_format == "audio/mpeg"

    with pytest.raises(ValueError, match="unsupported Rime audio_format"):
        tts.update_options(audio_format="audio/aac")

    assert tts._opts.audio_format == "audio/mpeg"


def test_v1_updates_model_with_endpoint() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key", model="coda", websocket_url="wss://coda.api.rime.ai/ws")
    tts.update_options(
        websocket_url="wss://mist.api.rime.ai/ws",
        model="mistv3",
        pause_between_brackets=True,
    )

    assert tts.model == "mistv3"
    assert tts._opts.mist_options is not None
    assert tts._opts.mist_options.pause_between_brackets is True


def test_v1_rejects_model_update() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key", model="coda", websocket_url="wss://coda.api.rime.ai/ws")

    with pytest.raises(ValueError, match="only be updated together"):
        tts.update_options(model="mistv3")


def test_v1_dedicated_endpoint_update_keeps_current_model() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key", model="coda", websocket_url="wss://coda.api.rime.ai/ws")
    tts.update_options(websocket_url="wss://dedicated.rime.ai/ws")

    assert tts.model == "coda"


def test_v1_dedicated_endpoint_rejects_model_change_for_same_url() -> None:
    from livekit.plugins.rime import TTS

    websocket_url = "wss://dedicated.rime.ai/ws"
    tts = TTS(api_key="test-key", websocket_url=websocket_url, model="coda")

    with pytest.raises(ValueError, match="model cannot change without changing websocket_url"):
        tts.update_options(websocket_url=websocket_url, model="mistv3")

    assert tts.model == "coda"


@pytest.mark.parametrize(
    "updated_websocket_url",
    [
        "wss://dedicated.rime.ai/ws/",
        "wss://dedicated.rime.ai/ws?token=rotated",
        "wss://DEDICATED.rime.ai/ws",
        "wss://dedicated.rime.ai:443/ws",
    ],
)
def test_v1_dedicated_endpoint_rejects_model_change_for_equivalent_url(
    updated_websocket_url: str,
) -> None:
    from livekit.plugins.rime import TTS

    websocket_url = "wss://dedicated.rime.ai/ws"
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

    websocket_url = "wss://dedicated.rime.ai/ws"
    tts = TTS(api_key="test-key", websocket_url=websocket_url, model="coda")

    tts.update_options(websocket_url=websocket_url, model="coda")

    assert tts.model == "coda"


def test_v1_dedicated_endpoint_updates_connection_url_for_same_model() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(
        api_key="test-key",
        websocket_url="wss://dedicated.rime.ai/ws",
        model="coda",
    )
    adapter = tts._websocket_v1_adapter
    assert adapter is not None
    previous_pool = adapter._pools.current
    updated_websocket_url = "wss://dedicated.rime.ai/ws?token=rotated"

    tts.update_options(websocket_url=updated_websocket_url, model="coda")

    assert tts.model == "coda"
    assert adapter._websocket_v1_url == updated_websocket_url
    assert adapter._pools.current is not previous_pool


@pytest.mark.parametrize(
    ("coda_url", "mist_url"),
    [
        ("wss://coda.api.rime.ai/ws", "wss://mist.api.rime.ai/ws"),
        (
            "wss://coda-test.rime.ai/ws",
            "wss://mist-test.rime.ai/ws",
        ),
    ],
)
def test_v1_dedicated_endpoint_allows_model_change_with_new_url(
    coda_url: str, mist_url: str
) -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(
        api_key="test-key",
        websocket_url=coda_url,
        model="coda",
    )

    tts.update_options(
        websocket_url=mist_url,
        model="mistv3",
        pause_between_brackets=True,
    )

    assert tts.model == "mistv3"
    assert tts._opts.mist_options is not None
    assert tts._opts.mist_options.pause_between_brackets is True
    assert tts._websocket_v1_adapter is not None
    assert tts._websocket_v1_adapter._websocket_v1_url == mist_url

    with pytest.raises(ValueError, match="model cannot change without changing websocket_url"):
        tts.update_options(
            websocket_url=mist_url,
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
        self.websocket_url = f"ws://127.0.0.1:{port}/custom/stream"
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


def _v1_tts(server: _RimeV1Server, *, model: str = "coda", **kwargs: Any):
    from livekit.plugins.rime import TTS

    return TTS(
        api_key="test-key",
        websocket_url=server.websocket_url,
        model=model,
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
    assert server.paths == ["/custom/stream"]
    assert server.headers[0]["Authorization"] == "Bearer test-key"
    assert server.headers[0]["Sec-WebSocket-Protocol"] == f"rime.v1.{websocket_protocol}"
    assert server.protocols == [f"rime.v1.{websocket_protocol}"]
    assert _payloads(server) == ["start", "text", "text", "end"]
    assert [request["text"] for request in server.requests if "text" in request] == [
        "Hello from LiveKit today. ",
        "Next sentence. ",
    ]
    start = dict(server.requests[0]["start"])
    # rime-api 0.4 adds this repeated field. The test server prints empty repeated
    # fields, even though the plugin does not send them on the wire.
    assert start.pop("customLexicon", []) == []
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


@pytest.mark.parametrize("websocket_protocol", ["binary", "json"])
@pytest.mark.parametrize(
    ("sample_rate", "expected_sample_rate"),
    [(None, 24000), (8000, 8000), (24000, 24000)],
    ids=["model-default", "8khz", "24khz"],
)
async def test_v1_pcmu_preserves_sample_rate_and_samples_after_parent_update(
    websocket_protocol: str, sample_rate: int | None, expected_sample_rate: int
) -> None:
    encoded_audio = bytes([0xFF, 0x7F, 0x80, 0x00]) * 600
    expected_pcm = np.tile(np.array([0, 0, 32124, -32124], dtype="<i2"), 600).tobytes()
    options = {"sample_rate": sample_rate} if sample_rate is not None else {}
    updated_sample_rate = 8000 if expected_sample_rate == 24000 else 24000

    async with _RimeV1Server(audio=encoded_audio) as server:
        tts = _v1_tts(
            server, websocket_protocol=websocket_protocol, audio_format="audio/pcmu", **options
        )
        assert tts.sample_rate == expected_sample_rate
        old_stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        try:
            tts.update_options(audio_format="audio/pcmu", sample_rate=updated_sample_rate)
            new_stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
            try:
                for stream, rate in [
                    (old_stream, expected_sample_rate),
                    (new_stream, updated_sample_rate),
                ]:
                    stream.push_text("hello")
                    stream.end_input()
                    events = await _collect(stream)
                    assert events[-1].is_final
                    assert {event.frame.sample_rate for event in events} == {rate}
                    assert {event.frame.num_channels for event in events} == {1}
                    assert b"".join(bytes(event.frame.data) for event in events) == expected_pcm
                    assert sum(event.frame.duration for event in events) == pytest.approx(
                        len(encoded_audio) / rate
                    )
            finally:
                await new_stream.aclose()
        finally:
            await old_stream.aclose()
            await tts.aclose()

    starts = [request["start"] for request in server.requests if "start" in request]
    assert [start["audioParameters"] for start in starts] == [
        {"audioFormat": "audio/pcmu", "samplingRate": rate}
        for rate in [expected_sample_rate, updated_sample_rate]
    ]


async def test_v1_maps_mist_options_without_a_second_stream_implementation() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(
            server,
            model="mistv3",
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


async def test_v1_sends_coda_custom_lexicon_in_start() -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server, custom_lexicon=_LEXICON)
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        stream.push_text("expedite it")
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert server.requests[0]["start"]["customLexicon"] == _LEXICON


async def test_v1_stream_keeps_lexicon_after_caller_mutates_entry() -> None:
    entry = {"spelling": "Acme", "pronunciation": '" { k . m i'}
    async with _RimeV1Server() as server:
        tts = _v1_tts(server, custom_lexicon=[entry])
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        entry["pronunciation"] = '" A k . m i'
        stream.push_text("Acme")
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert server.requests[0]["start"]["customLexicon"] == [
        {"spelling": "Acme", "pronunciation": '" { k . m i'}
    ]


@pytest.mark.parametrize(
    ("model", "expected_sample_rate"),
    [("coda", 24000), ("mistv3", 24000), ("mistv2", 22050), ("future-model", 22050)],
)
async def test_v1_sends_resolved_sample_rate(model: str, expected_sample_rate: int) -> None:
    async with _RimeV1Server() as server:
        tts = _v1_tts(server, model=model)
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
            mist_url = second_server.websocket_url
            tts.update_options(websocket_url=mist_url, model="mistv3")
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
        mistv2_url = second_server.websocket_url
        tts.update_options(websocket_url=mistv2_url, model="mistv2")
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))

        stream.push_text("hello")
        stream.end_input()
        await _collect(stream)
        await stream.aclose()
        await tts.aclose()

    assert second_server.paths == ["/custom/stream"]
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
        "wss://coda.api.rime.ai/ws",
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

    _websocket_v1.validate_websocket_url("wss://dedicated.rime.ai/ws")


def test_v1_rejects_lookalike_rime_host() -> None:
    from livekit.plugins.rime import _websocket_v1

    with pytest.raises(ValueError, match="trusted Rime host"):
        _websocket_v1.validate_websocket_url("wss://rime.ai.attacker.example/coda/ws")


@pytest.mark.parametrize(
    "websocket_url",
    [
        "wss://api.rime.ai/coda/ws",
        "wss://api.rime.ai/mist/ws",
        "wss://voice.customer.example/custom/stream?token=value",
        "wss://voice.customer.example/tts/%ZZ/stream",
    ],
)
def test_v1_custom_paths_do_not_select_model(websocket_url: str) -> None:
    tts = TTS(
        api_key="test-key",
        websocket_url=websocket_url,
        model="mistv3",
        allow_custom_endpoint=True,
    )
    assert tts.model == "mistv3"
    assert tts._websocket_v1_adapter is not None
    assert tts._websocket_v1_adapter._websocket_v1_url == websocket_url


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
                websocket_url="wss://coda.api.rime.ai/ws",
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
        # rime-api 0.4 removed the legacy Arcana field; older schemas retain it.
        if (
            request.HasField("start")
            and "arcana_parameters" in request.start.DESCRIPTOR.fields_by_name
        ):
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
    if hasattr(telemetry_utils, "redaction_enabled"):
        monkeypatch.setattr(telemetry_utils, "redaction_enabled", lambda *_: False)
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
                    model="coda",
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


def _check_imports(source: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("import_style", ["attribute", "from"])
def test_realtime_namespace_loads_on_demand(import_style: str) -> None:
    pytest.importorskip("rimelabs_sdk.realtime")
    _check_imports(
        f"""
        import importlib
        import sys
        from livekit.plugins import rime

        assert "livekit.plugins.rime.realtime" not in sys.modules
        assert "rimelabs_sdk" not in sys.modules
        assert "realtime" in rime.__all__
        assert "RealtimeModel" not in rime.__all__
        assert not hasattr(rime, "RealtimeModel")

        if {import_style!r} == "from":
            from livekit.plugins.rime import realtime
        else:
            realtime = rime.realtime

        assert realtime is importlib.import_module("livekit.plugins.rime.realtime")
        assert rime.realtime is realtime
        assert callable(realtime.RealtimeModel)
        assert not hasattr(rime, "RealtimeModel")
        """
    )


def test_tts_import_and_missing_realtime_sdk_error() -> None:
    _check_imports(
        """
        import sys
        sys.modules["rimelabs_sdk"] = None

        from livekit.plugins import rime
        from livekit.plugins.rime import TTS

        assert rime.TTS is TTS
        assert "livekit.plugins.rime.realtime" not in sys.modules
        assert not hasattr(rime, "RealtimeModel")
        try:
            rime.realtime
        except ImportError as error:
            assert "livekit-plugins-rime[realtime]" in str(error)
            assert "Python 3.11+" in str(error)
        else:
            raise AssertionError("Realtime imported without its SDK")

        assert "livekit.plugins.rime.realtime" not in sys.modules
        assert "realtime" not in vars(rime)
        assert rime.TTS is TTS
        """
    )


class Peer:
    def __init__(self):
        self.requests = asyncio.Queue()
        self.socket = None
        self.serial = 0
        self.settings = None
        self.updates = []
        self.revision = 0
        self.request_id = None
        self.update_started = asyncio.Event()
        self.update_release = asyncio.Event()
        self.update_release.set()
        self.connections = 0
        self.receipts = asyncio.Queue()
        self.truncations = asyncio.Queue()
        self.items = []
        self.server_items = []
        self.history_operations = []
        self.history_started = asyncio.Event()
        self.history_release = asyncio.Event()
        self.history_release.set()
        self.fail_history_at = None
        self.block_history_at = None
        self.running = {}
        self.history_serial = 0
        self.refusals = {}
        self.cancel_received = asyncio.Event()
        self.cancel_release = asyncio.Event()
        self.cancel_release.set()

    async def emit(self, kind, **body):
        if kind == "response.created":
            response = body["response"]
            self.running[response["id"]] = set(response.get("prism_input_item_ids", []))
        elif kind == "response.function_call_arguments.done":
            self.server_items.append(
                {
                    "id": body["item_id"],
                    "type": "function_call",
                    "call_id": body["call_id"],
                    "name": body["name"],
                    "arguments": body["arguments"],
                }
            )
            self.running.setdefault(body["response_id"], set()).add(body["item_id"])
        elif kind == "response.text.done":
            self.server_items.append(
                {
                    "id": body["item_id"],
                    "type": "message",
                    "role": "assistant",
                    "text": body["text"],
                }
            )
            self.running.setdefault(body["response_id"], set()).add(body["item_id"])
        elif kind == "response.done":
            self.running.pop(body["response"]["id"], None)
        await self.socket.send(json.dumps({"type": kind, "event_id": "server", **body}))

    async def handler(self, socket):
        self.socket = socket
        self.connections += 1
        self.settings = None
        self.server_items = []
        self.running = {}
        self.request_id = socket.request.headers.get("x-request-id", "peer-request")
        await self.emit("session.created", session={"id": "session-1"})
        async for raw in socket:
            event = json.loads(raw)
            kind = event["type"]
            if kind in self.refusals:
                code, scope = self.refusals.pop(kind)
                await self.emit(
                    "error",
                    error={
                        "code": code,
                        "message": "Request refused",
                        "scope": scope,
                        "owner": {"kind": scope, "event_id": event["event_id"]},
                    },
                )
                if kind == "response.cancel" and scope == "event":
                    await self.end(event["response_id"])
                continue
            if kind == "session.update":
                self.update_started.set()
                await self.update_release.wait()
                self.updates.append(event["session"])
                self.settings = {**(self.settings or {}), **event["session"]}
                self.revision += 1
                await self.emit(
                    "session.updated",
                    prism_request_event_id=event["event_id"],
                    session={
                        **self.settings,
                        "id": "session-1",
                        "voice": self.settings.get("voice", "test"),
                        "request_id": self.request_id,
                        "prism_settings_revision": self.revision,
                        "prism_effective_interrupt_response": self.settings.get(
                            "turn_detection", {}
                        ).get("interrupt_response", True),
                    },
                )
                await self.emit("prism.typed_input.ready")
            elif kind in ("conversation.item.create", "conversation.item.delete"):
                self.history_operations.append(event)
                self.history_started.set()
                if self.block_history_at is None or self.block_history_at == len(
                    self.history_operations
                ):
                    await self.history_release.wait()
                if self.fail_history_at == len(self.history_operations):
                    await self.emit(
                        "error",
                        error={
                            "code": "invalid_request",
                            "message": "History refused",
                            "scope": "event",
                            "owner": {"kind": "event", "event_id": event["event_id"]},
                        },
                    )
                    continue
                if kind == "conversation.item.create":
                    self.history_serial += 1
                    item = {"id": f"history-{self.history_serial}", **event["item"]}
                    self.items.append(event["item"])
                    self.server_items.append(item)
                    await self.emit(
                        "conversation.item.created",
                        prism_request_event_id=event["event_id"],
                        item=item,
                    )
                else:
                    item = next((i for i in self.server_items if i["id"] == event["item_id"]), None)
                    if item is not None:
                        call_id = item.get("call_id")
                        self.server_items[:] = [
                            i
                            for i in self.server_items
                            if i["id"] != item["id"]
                            and (not call_id or i.get("call_id") != call_id)
                        ]
                    await self.emit(
                        "conversation.item.deleted",
                        prism_request_event_id=event["event_id"],
                        item_id=event["item_id"],
                    )
            elif kind == "prism.conversation.get":
                active_ids = {i for ids in self.running.values() for i in ids}
                await self.emit(
                    "prism.conversation",
                    prism_request_event_id=event["event_id"],
                    items=[i for i in self.server_items if i["id"] not in active_ids],
                )
            elif kind == "response.cancel":
                self.cancel_received.set()
                await self.cancel_release.wait()
                await self.end(event["response_id"], status="cancelled")
            elif kind == "prism.playback.drained":
                self.receipts.put_nowait(event)
                await self.emit("prism.typed_input.ready")
            elif kind == "conversation.item.truncate":
                self.truncations.put_nowait(event)
                await self.emit(
                    "conversation.item.truncated",
                    item_id=event["item_id"],
                    content_index=event["content_index"],
                    audio_end_ms=event["audio_end_ms"],
                )
            elif kind == "input_audio_buffer.clear":
                await self.emit("input_audio_buffer.cleared")
            else:
                self.requests.put_nowait(event)

    async def next(self, kind="response.create"):
        event = await asyncio.wait_for(self.requests.get(), 3)
        assert event["type"] == kind
        return event

    async def start(self, request=None, response_id=None):
        self.serial += 1
        response_id = response_id or f"response-{self.serial}"
        body = {
            "response": {
                "id": response_id,
                "prism_settings_revision": self.revision,
                "metadata": request["response"]["metadata"]
                if request
                else {"prism_cause": "user_audio"},
            }
        }
        if request and "prism_input_text" in request["response"]:
            input_id = f"typed-{self.serial}"
            body["response"]["prism_input_item_ids"] = [input_id]
            self.server_items.append(
                {
                    "id": input_id,
                    "type": "message",
                    "role": "user",
                    "text": request["response"]["prism_input_text"],
                }
            )
        if request:
            body["prism_request_event_id"] = request["event_id"]
        await self.emit("response.created", **body)
        return response_id

    async def message(self, response_id, *, seconds=0.02, text="Hello"):
        item_id = f"message-{response_id}"
        await self.emit(
            "response.output_item.added",
            response_id=response_id,
            output_index=0,
            item={"id": item_id, "type": "message"},
        )
        common = {
            "response_id": response_id,
            "output_index": 0,
            "item_id": item_id,
            "content_index": 0,
        }
        await self.emit("response.text.delta", **common, delta=text)
        await self.emit("response.text.done", **common, text=text)
        await self.emit(
            "response.audio.delta",
            **common,
            delta=base64.b64encode(b"\0\0" * int(24000 * seconds)).decode(),
        )
        await self.emit("response.audio.done", **common)
        return item_id

    async def end(self, response_id, *, status="completed"):
        await self.emit(
            "response.done",
            response={
                "id": response_id,
                "status": status,
                "metadata": {"prism_drain_token": f"drain-{response_id}"},
            },
        )

        await self.emit("prism.typed_input.ready")


@asynccontextmanager
async def peer_model(**options):
    peer = Peer()
    async with serve(peer.handler, "127.0.0.1", 0) as server:
        endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime"
        async with Rime(api_key="test") as client:
            model = rime.realtime.RealtimeModel(endpoint=endpoint, client=client, **options)
            try:
                yield peer, model, client
            finally:
                await model.aclose()


async def wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), 3)


@function_tool
async def lookup() -> str:
    """Look up an order."""
    return "shipped"


def messages(*texts):
    ctx = llm.ChatContext.empty()
    for text in texts:
        ctx.add_message(role="user", content=text)
    return ctx


def tool_history():
    ctx = messages("Earlier question")
    ctx.add_message(role="system", content="Transferred to billing")
    ctx.items.extend(
        [
            llm.FunctionCall(call_id="past-call", name="lookup", arguments='{"order": "1"}'),
            llm.FunctionCallOutput(
                call_id="past-call", name="lookup", output="shipped", is_error=False
            ),
        ]
    )
    ctx.add_message(role="assistant", content="Your order has shipped")
    return ctx


async def completed_message(peer, session):
    response_id = await peer.start()
    item_id = await peer.message(response_id, text="The full generated answer")
    await peer.end(response_id)
    await wait_until(lambda: session.chat_ctx.get_by_id(item_id) is not None)
    return item_id


async def acknowledge(peer, item_id, kept_text, *, content_index=0):
    await peer.emit(
        "conversation.item.truncated",
        item_id=item_id,
        content_index=content_index,
        audio_end_ms=120,
        **({} if kept_text is None else {"prism_kept_text": kept_text}),
    )


@pytest.mark.skipif(not HAS_REALTIME_SDK, reason="Rime realtime SDK is not installed")
class TestRealtime:
    async def test_startup_seeds_history_before_first_request(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            ctx = llm.ChatContext.empty()
            ctx.items.append(llm.AgentConfigUpdate(instructions="Old configuration"))
            ctx.add_message(role="user", content="Earlier question")
            ctx.add_message(role="assistant", content="Earlier answer")
            await session._update_session(instructions="Be clear", chat_ctx=ctx, tools=[])
            assert peer.settings["instructions"] == "Be clear"
            assert [i["text"] for i in peer.items] == ["Earlier question", "Earlier answer"]
            assert all(isinstance(item, llm.ChatMessage) for item in session.chat_ctx.items)
            await session.update_instructions("Be clear")
            await session.update_instructions("Different")
            assert peer.settings["instructions"] == "Different"

    @pytest.mark.parametrize("invalid_item", ["developer", "image", "tool", "long"])
    async def test_history_update_validates_all_new_items_before_sending(self, invalid_item):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="test")
            ctx = session.chat_ctx
            ctx.add_message(role="user", content="Valid new history")
            if invalid_item == "developer":
                ctx.add_message(role="developer", content="Unsupported role")
            elif invalid_item == "image":
                ctx.add_message(role="user", content=[llm.ImageContent(image="https://test/image")])
            elif invalid_item == "tool":
                ctx.items.append(
                    llm.FunctionCallOutput(call_id="unknown", output="Result", is_error=False)
                )
            else:
                ctx.add_message(
                    role="user", content=" \t" if invalid_item == "blank" else "x" * 4001
                )
            with pytest.raises((llm.RealtimeError, RimeInputError)):
                await session.update_chat_ctx(ctx)
            assert peer.items == []
            assert session.chat_ctx.items == []

    async def test_agent_session_caller_interrupt_reports_truncate(self):
        async with peer_model() as (peer, model, _):
            output = FakeAudioOutput(can_pause=True)
            async with AgentSession(llm=model) as session:
                session.output.audio = output
                await session.start(Agent(instructions="test"))
                handle = session.generate_reply()
                request = await peer.next()
                response_id = await peer.start(request)
                item_id = await peer.message(response_id, seconds=1)
                while output._started_at is None:
                    await asyncio.sleep(0.001)
                await peer.emit(
                    "input_audio_buffer.speech_started", item_id="user-1", audio_start_ms=0
                )
                await peer.end(response_id, status="cancelled")
                await asyncio.wait_for(handle.wait_for_playout(), 5)
                event = await asyncio.wait_for(peer.truncations.get(), 3)
                assert event["item_id"] == item_id
                assert event["content_index"] == 0
                assert peer.receipts.empty()

    async def test_late_transcript_keeps_user_before_assistant(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="test")
            await peer.emit("input_audio_buffer.speech_started", item_id="u", audio_start_ms=0)
            await peer.emit("input_audio_buffer.committed", item_id="u")
            response_id = await peer.start()
            await peer.message(response_id)
            await peer.end(response_id)
            transcript = asyncio.Future()
            session.on("input_audio_transcription_completed", transcript.set_result)
            await peer.emit(
                "conversation.item.input_audio_transcription.completed",
                item_id="u",
                transcript="Question",
            )
            event = await asyncio.wait_for(transcript, 3)
            assert event.turn_started_at is not None
            assert [i.role for i in session.chat_ctx.items] == ["user", "assistant"]

    async def test_agent_session_executes_tool_then_continues(self):
        ran = []

        @function_tool
        async def lookup() -> str:
            """Look up a value."""
            ran.append(True)
            return "42"

        async with peer_model() as (peer, model, _):
            async with AgentSession(llm=model) as session:
                await session.start(Agent(instructions="test", tools=[lookup]))
                handle = session.generate_reply()
                request = await peer.next()
                parent = await peer.start(request)
                await peer.emit(
                    "response.function_call_arguments.done",
                    response_id=parent,
                    item_id="tool-item",
                    call_id="call-1",
                    name="lookup",
                    arguments="{}",
                )
                await peer.end(parent)
                continuation = await peer.next()
                assert ran == [True]
                assert peer.items[-1]["call_id"] == "call-1"
                assert continuation["response"]["metadata"]["prism_parent_response_id"] == parent
                response_id = await peer.start(continuation)
                await peer.end(response_id)
                await asyncio.wait_for(handle.wait_for_playout(), 5)

    async def test_forced_tool_choice_emits_fatal_error(self):
        async with peer_model() as (_, model, _):
            session = model.session()
            errors = []
            session.on("error", errors.append)
            await session._update_session(instructions="test")
            with pytest.raises(PrismUnsupported):
                session.update_options(tool_choice="required")
            assert errors[-1].recoverable is False

    async def test_injected_client_stays_open(self):
        async with peer_model() as (peer, model, client):
            await model.session()._update_session(instructions="test")
            await model.aclose()
            async with client.realtime.connect(endpoint=model._endpoint) as session:
                assert session.info.session_id == "session-1"

    async def test_abandoned_reply_does_not_generate_after_ready(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="test")
            # Keep the first response active so the second request waits for readiness.
            first = session.generate_reply()
            request = await peer.next()
            response_id = await peer.start(request)
            await first
            abandoned = session.generate_reply()
            await asyncio.sleep(0)
            abandoned.cancel()
            await asyncio.gather(abandoned, return_exceptions=True)
            await peer.end(response_id)
            await asyncio.sleep(0.02)
            assert peer.requests.empty()

    async def test_admission_timeout_leaves_conversation_usable(self, monkeypatch):
        async with peer_model() as (peer, model, client):
            connect = client.realtime.connect
            monkeypatch.setattr(
                client.realtime,
                "connect",
                lambda **kwargs: connect(
                    **{**kwargs, "timeouts": r.RealtimeTimeouts(ready_s=0.02, request_s=0.5)}
                ),
            )
            session = model.session()
            errors = []
            session.on("error", errors.append)
            await session._update_session(instructions="test")
            first = session.generate_reply()
            response_id = await peer.start(await peer.next())
            await first
            with pytest.raises(llm.RealtimeError) as raised:
                await session.generate_reply()
            assert raised.value.code == "admission_timeout"
            assert peer.requests.empty()
            await peer.end(response_id)
            await peer.emit("prism.typed_input.ready")
            next_reply = session.generate_reply()
            next_id = await peer.start(await peer.next())
            await peer.end(next_id)
            assert (await next_reply).response_id == next_id
            assert not errors

    @pytest.mark.parametrize(
        "code", ["tool_continuation_unavailable", "tool_continuation_not_ready"]
    )
    async def test_expired_tool_continuation_reports_result_without_reexecution(self, code):
        ran = []

        @function_tool
        async def lookup() -> str:
            """Look up a value."""
            ran.append(True)
            return "42"

        async with peer_model() as (peer, model, _):
            async with AgentSession(llm=model) as session:
                await session.start(Agent(instructions="test", tools=[lookup]))
                speech = session.generate_reply()
                parent = await peer.start(await peer.next())
                peer.refusals["response.create"] = (code, "event")
                await peer.emit(
                    "response.function_call_arguments.done",
                    response_id=parent,
                    item_id="tool",
                    call_id="call",
                    name="lookup",
                    arguments="{}",
                )
                await peer.end(parent)
                await peer.emit("prism.typed_input.ready")
                if code == "tool_continuation_unavailable":
                    report = await peer.next()
                    assert report["response"]["metadata"]["prism_cause"] == "proactive"
                    assert report["response"]["prism_call_id"] == "call"
                    response_id = await peer.start(report)
                    await peer.message(response_id, text="The result is 42.")
                    await peer.end(response_id)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert ran == [True]
                assert len(peer.items) == 1
                assert peer.items[0]["call_id"] == "call"
                if code == "tool_continuation_unavailable":
                    assert speech.exception() is None
                    assert any(
                        item.text_content == "The result is 42."
                        for item in session.history.items
                        if isinstance(item, llm.ChatMessage)
                    )
                else:
                    assert speech.exception().code == code
                    assert peer.requests.empty()

    @pytest.mark.parametrize("tool_choice", ["auto", "none"])
    async def test_connection_failure_cancels_reply_waiting_for_adapter_observation(
        self, monkeypatch, tool_choice
    ):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="test")
            observed = session._on_event

            def delay_response_started(event):
                # The SDK has accepted the response, but the adapter has not yet
                # registered its generation when the connection fails.
                if not isinstance(event.payload, r.ResponseStarted):
                    observed(event)

            monkeypatch.setattr(session, "_on_event", delay_response_started)
            waiting = asyncio.Event()
            wait = session._generation_ready.wait

            async def mark_waiting():
                waiting.set()
                return await wait()

            monkeypatch.setattr(session._generation_ready, "wait", mark_waiting)
            reply = session.generate_reply(tool_choice=tool_choice)
            await peer.start(await peer.next())
            await asyncio.wait_for(waiting.wait(), 3)
            assert not reply.done()
            await peer.socket.close()
            done, _ = await asyncio.wait([reply], timeout=3)
            assert reply in done
            assert reply.cancelled()
            assert session._failure is not None
            await asyncio.wait_for(session.aclose(), 3)

    @pytest.mark.parametrize("wrapped", [False, True])
    async def test_standard_session_text_and_empty_replies_need_no_playback_receipts(self, wrapped):
        async with peer_model() as (peer, model, _):
            provider = RealtimeModelFallbackAdapter([model]) if wrapped else model
            async with AgentSession(llm=provider) as session:
                session.output.audio = FakeAudioOutput()
                await session.start(Agent(instructions="Help"))
                for text in ("First question", "Next question"):
                    speech = session.generate_reply(user_input=text)
                    request = await peer.next()
                    assert "prism_input_text" not in request["response"]
                    assert request["response"]["metadata"]["prism_cause"] == "proactive"
                    assert [i["text"] for i in peer.items if i.get("text") == text] == [text]
                    response = await peer.start(request)
                    if text == "Next question":
                        await peer.message(response)
                    await peer.end(response)
                    await asyncio.wait_for(speech.wait_for_playout(), 3)
                    assert speech.exception() is None
                assert peer.receipts.empty()

    @pytest.mark.parametrize("max_tool_steps", [0, 1, 2])
    async def test_standard_session_tool_limit_produces_final_reply_and_restores_tools(
        self, max_tool_steps
    ):
        ran = []

        @function_tool
        async def lookup() -> str:
            """Look up a value."""
            ran.append(True)
            return "42"

        async with peer_model() as (peer, model, _):
            async with AgentSession(llm=model, max_tool_steps=max_tool_steps) as session:
                await session.start(Agent(instructions="Help", tools=[lookup]))
                speech = session.generate_reply()
                for step in range(max_tool_steps + 1):
                    request = await peer.next()
                    assert peer.settings["tools"]
                    parent = await peer.start(request)
                    await peer.emit(
                        "response.function_call_arguments.done",
                        response_id=parent,
                        item_id=f"tool-{step}",
                        call_id=f"call-{step}",
                        name="lookup",
                        arguments="{}",
                    )
                    await peer.end(parent)
                final = await peer.next()
                tools_disabled_at_limit = peer.settings["tools"] == []
                assert final["response"]["metadata"]["prism_cause"] == "tool_continuation"
                response = await peer.start(final)
                await peer.message(response, text="The result is 42.")
                await peer.end(response)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert speech.exception() is None
                assert ran == [True] * (max_tool_steps + 1)
                assert peer.settings["tools"][0]["function"]["name"] == "lookup"
                assert model._session._tool_choice == "auto"
                assert peer.receipts.empty()
                # Check the full reply before accepting the older core's tool-limit failure.
                if Version(version("livekit-agents")) < Version("1.8.6"):
                    if not tools_disabled_at_limit:
                        pytest.xfail(
                            "LiveKit <1.8.6 does not enforce realtime max_tool_steps (#7555)"
                        )
                assert tools_disabled_at_limit

    async def test_tool_disabled_response_never_executes_unexpected_call(self):
        ran = []

        @function_tool
        async def lookup() -> str:
            """Look up a value."""
            ran.append(True)
            return "42"

        async with peer_model() as (peer, model, _):
            async with AgentSession(llm=model) as session:
                await session.start(Agent(instructions="Help", tools=[lookup]))
                errors = asyncio.Queue()
                model._session.on("error", errors.put_nowait)
                session.generate_reply(tool_choice="none")
                request = await peer.next()
                assert peer.settings["tools"] == []
                response = await peer.start(request)
                # Even if the model calls a tool from history, it must not reach execution.
                await peer.emit(
                    "response.function_call_arguments.done",
                    response_id=response,
                    item_id="unexpected",
                    call_id="unexpected",
                    name="lookup",
                    arguments="{}",
                )
                error = await asyncio.wait_for(errors.get(), 3)
                assert isinstance(error.error, PrismUnsupported)
                assert not error.recoverable
                assert ran == []

    async def test_cancelled_tool_disabled_reply_restores_after_late_acceptance(self):
        @function_tool
        async def lookup() -> str:
            """Look up a value."""
            return "42"

        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_tools([lookup])
            pending = session.generate_reply(tool_choice="none")
            request = await peer.next()
            pending.cancel()
            await asyncio.sleep(0)
            assert peer.settings["tools"] == []
            response = await peer.start(request)
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 3)
            assert peer.cancel_received.is_set()
            assert peer.settings["tools"][0]["function"]["name"] == "lookup"
            assert peer.requests.empty()
            assert session._generations[response].abandoned

    async def test_refused_tool_disabled_reply_restores_tools_and_keeps_session(self):
        @function_tool
        async def lookup() -> str:
            """Look up a value."""
            return "42"

        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_tools([lookup])
            errors = []
            session.on("error", errors.append)
            peer.refusals["response.create"] = ("invalid_request", "event")
            with pytest.raises(llm.RealtimeError, match="invalid_request"):
                await session.generate_reply(tool_choice="none")
            assert peer.settings["tools"][0]["function"]["name"] == "lookup"
            assert not errors
            pending = session.generate_reply(tool_choice="auto")
            response = await peer.start(await peer.next())
            await peer.end(response)
            assert (await pending).response_id == response

    async def test_failed_tool_restoration_fails_session(self):
        @function_tool
        async def lookup() -> str:
            """Look up a value."""
            return "42"

        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_tools([lookup])
            errors = asyncio.Queue()
            session.on("error", errors.put_nowait)
            pending = session.generate_reply(tool_choice="none")
            request = await peer.next()
            peer.refusals["session.update"] = ("invalid_session", "event")
            await peer.start(request)
            await asyncio.gather(pending, return_exceptions=True)
            error = await asyncio.wait_for(errors.get(), 3)
            assert not error.recoverable
            assert "restoration" in str(error.error)
            assert session._failure is not None

    @pytest.mark.parametrize("entrypoint", ["speech_handle", "session"])
    async def test_ordinary_session_application_stop_truncates_without_completion_receipt(
        self, entrypoint
    ):
        async with peer_model() as (peer, model, _):
            output = FakeAudioOutput(can_pause=True)
            async with AgentSession(llm=model) as session:
                session.output.audio = output
                await session.start(Agent(instructions="Help"))
                speech = session.generate_reply()
                response = await peer.start(await peer.next())
                item = await peer.message(response, seconds=1)
                async with asyncio.timeout(3):
                    while output._started_at is None:
                        await asyncio.sleep(0)
                if entrypoint == "session":
                    session.interrupt(force=True)
                else:
                    speech.interrupt(force=True)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                truncate = await asyncio.wait_for(peer.truncations.get(), 3)
                assert truncate["item_id"] == item
                await asyncio.wait_for(peer.cancel_received.wait(), 3)
                assert peer.receipts.empty()

    @pytest.mark.parametrize("revision", ["older", "suppressed", "missing"])
    async def test_automatic_reply_uses_its_own_tool_settings_during_suppression(self, revision):
        @function_tool
        async def lookup() -> str:
            """Look up a value."""
            return "42"

        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_tools([lookup])
            old_revision = peer.revision
            generations = asyncio.Queue()
            errors = asyncio.Queue()
            session.on("generation_created", generations.put_nowait)
            session.on("error", errors.put_nowait)
            pending = session.generate_reply(tool_choice="none")
            try:
                request = await peer.next()
                response = {"id": "early-speech", "metadata": {"prism_cause": "user_audio"}}
                if revision != "missing":
                    response["prism_settings_revision"] = (
                        old_revision if revision == "older" else peer.revision
                    )
                await peer.emit("response.created", response=response)
                generation = await asyncio.wait_for(generations.get(), 3)
                await peer.emit(
                    "response.function_call_arguments.done",
                    response_id="early-speech",
                    item_id="early-tool",
                    call_id="early-call",
                    name="lookup",
                    arguments="{}",
                )
                if revision == "older":
                    call = await asyncio.wait_for(anext(generation.function_stream), 3)
                    assert call.name == "lookup"
                    assert errors.empty()
                    await peer.end("early-speech")
                    response_id = await peer.start(request)
                    await peer.end(response_id)
                    await asyncio.wait_for(pending, 3)
                    assert peer.settings["tools"][0]["function"]["name"] == "lookup"
                else:
                    error = await asyncio.wait_for(errors.get(), 3)
                    assert isinstance(error.error, PrismUnsupported)
                    assert not error.recoverable
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    async def test_connection_settings_fillers_and_session_information(self):
        async with peer_model(
            voice="test",
            turn_detection=r.TurnDetection(eagerness="low", end_of_turn_timeout_ms=3000),
            request_id="livekit-validation",
            response_metadata={"workflow": "billing"},
            tool_fillers={"lookup": r.PrismFiller(type="text", text="Checking your order.")},
        ) as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="Help", tools=[lookup])
            assert peer.request_id == "livekit-validation"
            assert peer.settings["turn_detection"]["eagerness"] == "low"
            assert peer.settings["turn_detection"]["prism_end_of_turn_timeout_ms"] == 3000
            assert peer.settings["tools"][0]["prism_filler"] == {
                "type": "text",
                "text": "Checking your order.",
            }
            assert model.session_info.request_id == "livekit-validation"
            assert model.session_info.response_metadata == {"workflow": "billing"}

    async def test_settings_updates_and_rejection_preserve_session(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="Original")
            await session._update_session(instructions="Billing", tools=[lookup])
            assert peer.settings["instructions"] == "Billing"
            assert peer.updates[-1]["tools"][0]["function"]["name"] == "lookup"
            await model.update_session(voice="new-voice")
            assert model.session_info.voice == "new-voice"
            assert model.session_info.interrupt_on_speech is True
            await model.update_session(response_metadata={"step": "2"})
            assert model.session_info.response_metadata == {"step": "2"}
            old_revision = model.session_info.settings_revision
            peer.refusals["session.update"] = ("invalid_session", "event")
            with pytest.raises(PrismError) as raised:
                await session.update_instructions("Rejected")
            assert raised.value.code == "invalid_session"
            assert raised.value.retryable is False
            assert peer.settings["instructions"] == "Billing"
            assert model.session_info.settings_revision == old_revision
            await session._update_session(instructions="", tools=[])
            assert peer.settings["instructions"] == ""
            assert peer.settings["tools"] == []
            assert peer.connections == 1

    @pytest.mark.parametrize("cancellations", [1, 2])
    async def test_cancelled_update_waits_for_acknowledged_settings(self, cancellations):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="Original")
            peer.update_started.clear()
            peer.update_release.clear()
            change = asyncio.create_task(session.update_tools([lookup]))
            await asyncio.wait_for(peer.update_started.wait(), 3)
            for _ in range(cancellations):
                change.cancel()
                await asyncio.sleep(0)
                assert not change.done()
            peer.update_release.set()
            with pytest.raises(asyncio.CancelledError):
                await change
            assert "lookup" in session.tools.function_tools
            assert peer.settings["tools"][0]["function"]["name"] == "lookup"

    @pytest.mark.parametrize("operation", ["settings", "history"])
    async def test_session_shutdown_settles_an_update_waiting_for_acknowledgment(self, operation):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("Original")
            if operation == "settings":
                started, release = peer.update_started, peer.update_release
                update = session.update_tools([lookup])
            else:
                started, release = peer.history_started, peer.history_release
                target = session.chat_ctx
                target.add_message(role="user", content="pending")
                update = session.update_chat_ctx(target)
            started.clear()
            release.clear()
            change = asyncio.create_task(update)
            closing = None
            try:
                await asyncio.wait_for(started.wait(), 3)
                closing = asyncio.create_task(session.aclose())
                await wait_until(lambda: session._closed)
                # The SDK settles an already submitted request before propagating
                # cancellation, including during shutdown.
                release.set()
                await asyncio.wait_for(closing, 3)
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(change, 3)
            finally:
                release.set()
                change.cancel()
                await asyncio.gather(change, return_exceptions=True)
                if closing is not None:
                    await asyncio.gather(closing, return_exceptions=True)

    async def test_metadata_refusal_preserves_acknowledged_voice(self):
        async with peer_model(response_metadata={"step": "1"}) as (peer, model, _):
            session = model.session()
            await session.update_instructions("Help")

            def refuse_next_update(info):
                if info.voice == "new-voice":
                    peer.refusals["session.update"] = ("invalid_session", "event")

            model.on("session_updated", refuse_next_update)
            with pytest.raises(PrismError, match="invalid_session"):
                await model.update_session(voice="new-voice", response_metadata={"step": "2"})

            assert model.session_info.voice == model._voice == peer.settings["voice"] == "new-voice"
            assert model.session_info.response_metadata == model._response_metadata == {"step": "1"}
            assert session._failure is None

    async def test_agent_handoff_reuses_conversation_and_updates_tools(self):
        async with peer_model() as (peer, model, _):
            async with AgentSession(llm=model) as session:
                first = Agent(instructions="Triage")
                await session.start(first)
                speech = session.generate_reply(user_input="Hello")
                request = await peer.next()
                response_id = await peer.start(request)
                await peer.message(response_id)
                await peer.end(response_id)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                second = Agent(instructions="Billing", tools=[lookup], chat_ctx=first.chat_ctx)
                session.update_agent(second)
                await wait_until(
                    lambda: (
                        peer.settings.get("instructions") == "Billing"
                        and bool(peer.settings.get("tools"))
                    )
                )
                assert peer.connections == 1
                assert peer.settings["tools"][0]["function"]["name"] == "lookup"
                assert any(
                    item.text_content == "Hello"
                    for item in second.chat_ctx.items
                    if item.type == "message"
                )

    async def test_agent_task_returns_to_parent_on_the_same_connection(self):
        entered = []
        result = asyncio.Future()

        class Task(AgentTask[str]):
            def __init__(self, chat_ctx):
                super().__init__(instructions="Task", chat_ctx=chat_ctx)

            async def on_enter(self):
                entered.append(self.session.current_agent.instructions)
                self.complete("finished")

        class Parent(Agent):
            def __init__(self):
                super().__init__(instructions="Parent")

            async def on_enter(self):
                if not result.done():
                    result.set_result(await Task(self.chat_ctx))

        async with peer_model() as (peer, model, _):
            async with AgentSession(llm=model) as session:
                await session.start(Parent())
                assert await asyncio.wait_for(result, 3) == "finished"
                await wait_until(lambda: peer.settings["instructions"] == "Parent")
                assert entered == ["Task"]
                assert peer.connections == 1

    async def test_tool_handoff_preserves_the_pending_tool_result(self):
        ran = []

        @function_tool
        async def transfer(context: RunContext) -> Agent:
            """Transfer to billing."""
            ran.append("transfer")
            return Agent(instructions="Billing", chat_ctx=context.session.current_agent.chat_ctx)

        async with peer_model() as (peer, model, _):
            async with AgentSession(llm=model) as session:
                await session.start(Agent(instructions="Triage", tools=[transfer]))
                speech = session.generate_reply(user_input="Billing please")
                response_id = await peer.start(await peer.next())
                await peer.emit(
                    "response.function_call_arguments.done",
                    response_id=response_id,
                    item_id="transfer-item",
                    call_id="transfer-call",
                    name="transfer",
                    arguments="{}",
                )
                await peer.end(response_id)
                await wait_until(
                    lambda: (
                        peer.settings is not None and peer.settings.get("instructions") == "Billing"
                    )
                )
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert ran == ["transfer"]
                assert peer.connections == 1
                assert any(item.get("call_id") == "transfer-call" for item in peer.items)

    async def test_generation_diagnostics_and_transcript_deltas(self):
        async with peer_model(request_id="diagnostics") as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="Help")
            generations, stopped, transcripts = [], [], []
            model.on("generation_created", generations.append)
            session.on("input_speech_stopped", stopped.append)
            session.on("input_audio_transcription_completed", transcripts.append)
            await peer.emit(
                "input_audio_buffer.speech_started", item_id="input-1", audio_start_ms=0
            )
            await peer.emit(
                "conversation.item.input_audio_transcription.delta", item_id="input-1", delta="Hel"
            )
            await peer.emit(
                "conversation.item.input_audio_transcription.delta", item_id="input-1", delta="lo"
            )
            await peer.emit(
                "input_audio_buffer.speech_stopped",
                item_id="input-1",
                audio_end_ms=200,
                prism_end_reason="semantic",
            )
            await peer.emit(
                "response.created",
                response={
                    "id": "diagnostic-response",
                    "metadata": {"prism_cause": "user_audio", "workflow": "billing"},
                    "prism_input_item_ids": ["input-1"],
                    "prism_settings_revision": peer.revision,
                },
            )
            await wait_until(lambda: generations and stopped and len(transcripts) == 2)
            assert [event.transcript for event in transcripts] == ["Hel", "Hello"]
            assert all(not event.is_final for event in transcripts)
            assert isinstance(stopped[0], PrismSpeechStoppedEvent)
            assert stopped[0].end_reason == "semantic"
            assert isinstance(generations[0], PrismGenerationCreatedEvent)
            assert generations[0].metadata == {"workflow": "billing"}
            assert generations[0].input_item_ids == ("input-1",)
            assert generations[0].provider_request_id == "diagnostics"
            assert generations[0].settings_revision == peer.revision

    async def test_reply_labels_and_saved_instruction_deletion(self):
        async with peer_model(response_metadata={"call": "shared"}) as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="Help")
            labels = {"call": "this-reply", "step": "2"}
            pending = session.generate_reply(instructions="Check in", metadata=labels)
            labels["step"] = "changed"
            request = await peer.next()
            assert request["response"]["metadata"] == {
                "prism_cause": "proactive",
                "call": "this-reply",
                "step": "2",
            }
            request["response"]["metadata"]["prism_instruction_item_id"] = "saved-instruction"
            peer.server_items.append(
                {"id": "saved-instruction", "type": "message", "role": "user", "text": "Check in"}
            )
            response_id = await peer.start(request)
            created = await pending
            assert created.instruction_item_id == "saved-instruction"
            with pytest.raises(RimeInputError, match="before its reply ends"):
                await model.delete_instruction_item(created.instruction_item_id)
            await peer.message(response_id)
            await peer.end(response_id)
            await wait_until(lambda: session._generations[response_id].ended)
            session._generations.pop(response_id)
            await model.delete_instruction_item(created.instruction_item_id)
            assert peer.history_operations[-1]["item_id"] == "saved-instruction"
            assert all(item["id"] != "saved-instruction" for item in peer.server_items)
            with pytest.raises(RimeInputError, match="Unknown saved"):
                await model.delete_instruction_item("saved-instruction")

    @pytest.mark.parametrize(
        "code,retryable",
        [
            ("capacity_exceeded", True),
            ("response_timeout", True),
            ("context_length_exceeded", False),
        ],
    )
    @pytest.mark.parametrize("via_fallback", [False, True])
    async def test_fault_retryability_does_not_replay_requests(self, code, retryable, via_fallback):
        async with peer_model() as (peer, model, _):
            realtime_model = RealtimeModelFallbackAdapter([model]) if via_fallback else model
            async with AgentSession(llm=realtime_model) as session:
                errors = []
                session.on("error", errors.append)
                await session.start(Agent(instructions="Help"))
                await peer.emit(
                    "error",
                    error={
                        "code": code,
                        "message": "Provider refusal",
                        "scope": "response",
                        "owner": {"kind": "response", "response_id": "response-1"},
                    },
                )
                await wait_until(lambda: errors)
                model_error = errors[0].error
                assert isinstance(model_error, PrismModelError)
                assert model_error.retryable is retryable
                assert isinstance(model_error.error, PrismError)
                assert model_error.error.retryable is retryable
                assert model_error.error.code == code
                assert model_error.error.scope == "response"
                assert model_error.error.response_id == "response-1"
                assert model_error.error.fault.code == code
                assert errors[0].model_dump()["error"]["retryable"] is retryable
                assert peer.requests.empty()

    @pytest.mark.parametrize("clear", [False, True])
    async def test_audio_before_connection_is_buffered_and_can_be_cleared(self, clear):
        async with peer_model() as (peer, model, _):
            session = model.session()

            def frame(sample):
                return rtc.AudioFrame(
                    data=sample.to_bytes(2, "little") * 640,
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=640,
                )

            first, fresh = frame(1), frame(2)
            session.push_audio(first)
            if clear:
                session.clear_audio()
                session.push_audio(fresh)
            assert session._sdk is None
            assert session._failure is None
            event = await peer.next("input_audio_buffer.append")
            expected = fresh if clear else first
            assert base64.b64decode(event["audio"]) == bytes(expected.data)
            await session._update_session(instructions="test")
            reply = session.generate_reply()
            response_id = await peer.start(await peer.next())
            await peer.end(response_id)
            assert (await asyncio.wait_for(reply, 3)).response_id == response_id
            assert peer.requests.empty()

    async def test_audio_overflow_during_handshake_closes_the_late_connection(self):
        async with peer_model() as (peer, model, _):
            peer.update_release.clear()
            session = model.session()
            await asyncio.wait_for(peer.update_started.wait(), 3)
            frame = rtc.AudioFrame.create(16000, 1, 640)
            for _ in range(101):
                session.push_audio(frame)
            assert isinstance(session._failure, asyncio.QueueFull)
            peer.update_release.set()
            await asyncio.wait_for(peer.socket.wait_closed(), 3)
            assert session._shutdown_task is not None
            await asyncio.wait_for(session.aclose(), 3)

    @pytest.mark.parametrize(
        "sample_rate,channels", [(44100, 1), (44100, 2), (44100, 4), (48000, 4)]
    )
    async def test_audio_converts_formats_outside_sdk_support(self, sample_rate, channels):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="test")
            frame = rtc.AudioFrame(
                data=np.full((sample_rate // 5, channels), 1000, dtype=np.int16).tobytes(),
                sample_rate=sample_rate,
                num_channels=channels,
                samples_per_channel=sample_rate // 5,
            )
            session.push_audio(frame)
            assert session._failure is None
            event = await peer.next("input_audio_buffer.append")
            samples = np.frombuffer(base64.b64decode(event["audio"]), dtype=np.int16)
            assert len(samples) == 640
            assert np.allclose(samples[100:], 1000, atol=5)
            await session.aclose()

    async def test_audio_resampling_keeps_state_between_short_frames(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="test")
            frame = rtc.AudioFrame(
                data=np.full(441, 1000, dtype=np.int16).tobytes(),
                sample_rate=44100,
                num_channels=1,
                samples_per_channel=441,
            )
            # Each frame is too short for a fresh resampler to produce output.
            for _ in range(20):
                session.push_audio(frame)
            event = await peer.next("input_audio_buffer.append")
            samples = np.frombuffer(base64.b64decode(event["audio"]), dtype=np.int16)
            assert np.allclose(samples[100:], 1000, atol=5)
            await session.aclose()

    @pytest.mark.parametrize("reset", ["clear", "format_change"])
    async def test_audio_conversion_discards_buffered_samples_on_reset(self, reset):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="test")
            session.push_audio(
                rtc.AudioFrame(
                    data=np.full(100, 30000, dtype=np.int16).tobytes(),
                    sample_rate=44100,
                    num_channels=1,
                    samples_per_channel=100,
                )
            )
            if reset == "clear":
                session.clear_audio()
            else:
                session.push_audio(rtc.AudioFrame.create(16000, 1, 640))
                await peer.next("input_audio_buffer.append")
            session.push_audio(rtc.AudioFrame.create(44100, 1, 8820))
            event = await peer.next("input_audio_buffer.append")
            samples = np.frombuffer(base64.b64decode(event["audio"]), dtype=np.int16)
            assert np.max(np.abs(samples)) <= 1
            await session.aclose()

    @pytest.mark.parametrize("sample_rate", [8000, 16000, 24000, 48000])
    @pytest.mark.parametrize("channels", [1, 2])
    async def test_sdk_supported_audio_reaches_sdk_unchanged(
        self, sample_rate, channels, monkeypatch
    ):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="test")
            sdk = session._require()
            send_audio = sdk.send_audio
            chunks = []

            async def observe(chunk):
                chunks.append(chunk)
                await send_audio(chunk)

            monkeypatch.setattr(sdk, "send_audio", observe)
            frame = rtc.AudioFrame.create(sample_rate, channels, sample_rate // 25)
            session.push_audio(frame)
            await peer.next("input_audio_buffer.append")
            assert len(chunks) == 1
            assert chunks[0].format.sample_rate == sample_rate
            assert chunks[0].format.channels == channels
            assert chunks[0].data == bytes(frame.data)
            await session.aclose()

    async def test_clear_audio_follows_inflight_audio_and_discards_queued_audio(self, monkeypatch):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="test")
            sdk = session._require()
            send = sdk._send
            clear = sdk.clear_audio
            append_started = asyncio.Event()
            append_release = asyncio.Event()
            clear_started = asyncio.Event()
            sent = []

            async def blocked_send(event, *args, **kwargs):
                if event["type"] == "input_audio_buffer.append" and not append_started.is_set():
                    # The actual SDK already owns its audio lock at this point.
                    append_started.set()
                    await append_release.wait()
                sent.append(event["type"])
                await send(event, *args, **kwargs)

            async def observed_clear():
                clear_started.set()
                await clear()

            monkeypatch.setattr(sdk, "_send", blocked_send)
            monkeypatch.setattr(sdk, "clear_audio", observed_clear)

            def frame(sample):
                return rtc.AudioFrame(
                    data=sample.to_bytes(2, "little") * 640,
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=640,
                )

            first, queued, fresh = frame(1), frame(2), frame(3)
            session.push_audio(first)
            await asyncio.wait_for(append_started.wait(), 3)
            session.push_audio(queued)
            session.clear_audio()
            await asyncio.wait_for(clear_started.wait(), 3)
            session.push_audio(fresh)
            append_release.set()

            before_clear = await peer.next("input_audio_buffer.append")
            after_clear = await peer.next("input_audio_buffer.append")
            assert base64.b64decode(before_clear["audio"]) == bytes(first.data)
            assert base64.b64decode(after_clear["audio"]) == bytes(fresh.data)
            assert sent == [
                "input_audio_buffer.append",
                "input_audio_buffer.clear",
                "input_audio_buffer.append",
            ]
            await session.aclose()

    async def test_background_dependency_error_stops_the_session(self, monkeypatch):
        """Inject an SDK failure while exercising the adapter's public clear method."""
        async with peer_model() as (peer, model, client):
            failure = llm.RealtimeError("background clear failed")
            connect = client.realtime.connect

            async def fail_clear():
                raise failure

            @asynccontextmanager
            async def connect_with_failure(**options):
                async with connect(**options) as sdk:
                    monkeypatch.setattr(sdk, "clear_audio", fail_clear)
                    yield sdk

            monkeypatch.setattr(client.realtime, "connect", connect_with_failure)
            session = model.session()
            errors = asyncio.Queue()
            session.on("error", errors.put_nowait)
            await session._update_session(instructions="Help")
            session.clear_audio()
            error = await asyncio.wait_for(errors.get(), 3)
            assert error.error is failure
            assert not error.recoverable
            with pytest.raises(llm.RealtimeError, match="background clear failed"):
                await session.update_instructions("new turn")
            assert peer.requests.empty()

    async def test_reply_refusal_reaches_caller_without_stopping_session(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            errors = asyncio.Queue()
            session.on("error", errors.put_nowait)
            await session._update_session(instructions="Help")
            peer.refusals["response.create"] = ("invalid_request", "event")
            with pytest.raises(PrismError) as raised:
                await asyncio.wait_for(session.generate_reply(), 3)
            assert raised.value.code == "invalid_request"
            assert errors.empty()
            # A named refusal did not create a response. The next explicit reply
            # request can still use this conversation and must not replay the first.
            await peer.emit("prism.typed_input.ready")
            pending = session.generate_reply()
            response_id = await peer.start(await peer.next())
            await peer.end(response_id)
            assert (await asyncio.wait_for(pending, 3)).response_id == response_id
            assert errors.empty()
            assert peer.requests.empty()

    async def test_session_fault_reports_one_fatal_error(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            errors = asyncio.Queue()
            session.on("error", errors.put_nowait)
            await session._update_session(instructions="Help")
            await peer.emit(
                "error",
                error={
                    "code": "unavailable",
                    "message": "Session cannot continue",
                    "scope": "session",
                    "owner": {"kind": "session"},
                },
            )
            error = await asyncio.wait_for(errors.get(), 3)
            assert not error.recoverable
            assert error.retryable is True
            assert error.error.scope == "session"
            await session.aclose()
            assert errors.empty()

    async def test_prism_fallback_typed_input_playback_and_explicit_restart(self) -> None:
        async with peer_model() as (peer, model, _):
            adapter = RealtimeModelFallbackAdapter([model])
            async with AgentSession(llm=adapter) as session:
                session.output.audio = FakeAudioOutput()
                await session.start(Agent(instructions="test"))
                for text in ("First turn", "After explicit restart"):
                    speech = session.generate_reply(user_input=text)
                    request = await peer.next()
                    assert "prism_input_text" not in request["response"]
                    assert peer.items[-1]["text"] == text
                    response_id = await peer.start(request)
                    await peer.message(response_id)
                    await peer.end(response_id)
                    await asyncio.wait_for(speech.wait_for_playout(), 3)
                    assert peer.receipts.empty()
                    if text == "First turn":
                        await adapter.restart_session()
                        assert [item["text"] for item in peer.server_items] == [
                            "First turn",
                            "Hello",
                        ]

    @pytest.mark.parametrize("policy", [None, "auto", "disabled", pytest.param(0.9, id="policy3")])
    async def test_truncation_at_connect_update_and_restart(self, policy):
        if isinstance(policy, float):
            policy = r.RetentionRatio(policy)
        async with peer_model(truncation=policy) as (peer, model, _):
            fallback = RealtimeModelFallbackAdapter([model])
            session = fallback.session()
            try:
                await session._update_session(instructions="Help")
                if policy is None:
                    assert "truncation" not in peer.settings
                    assert model.session_info.truncation is None
                else:
                    assert model.session_info.truncation == policy
                await model.update_session(truncation=r.RetentionRatio(0.7))
                assert model.session_info.truncation == r.RetentionRatio(0.7)
                await fallback.restart_session()
                assert model.session_info.truncation == r.RetentionRatio(0.7)
            finally:
                await session.aclose()

    async def test_rejected_truncation_leaves_acknowledged_policy_unchanged(self):
        async with peer_model(truncation="auto") as (peer, model, _):
            session = model.session()
            await session.update_instructions("test")
            peer.refusals["session.update"] = ("invalid_session", "event")
            with pytest.raises(PrismError):
                await model.update_session(truncation="disabled")
            assert model.session_info.truncation == "auto"
            await model.update_session(truncation="disabled")
            assert model.session_info.truncation == "disabled"

    async def test_initial_history_accepts_all_supported_types_without_executing_tools(self):
        ran = []

        @function_tool
        async def lookup(order: str) -> str:
            """Look up an order."""
            ran.append(order)
            return "unexpected"

        async with peer_model() as (peer, model, _):
            async with AgentSession(llm=model) as session:
                ctx = tool_history()
                await session.start(Agent(instructions="Help", tools=[lookup], chat_ctx=ctx))
                history = await model.get_history()
                assert [type(i) for i in history] == [
                    r.HistoryMessage,
                    r.HistoryMessage,
                    r.HistoryFunctionCall,
                    r.HistoryFunctionResult,
                    r.HistoryMessage,
                ]
                assert history[1].role == "system"
                assert history[2].arguments == '{"order": "1"}'
                assert history[3].output == "shipped"
                assert [i.id for i in model._session.chat_ctx.items] == [i.id for i in ctx.items]
                assert not ran
                assert peer.requests.empty()

    @pytest.mark.parametrize(
        "edit", ["append", "prepend", "middle", "delete", "replace", "reorder", "noop"]
    )
    async def test_context_edits_preserve_local_identity_and_server_order(self, edit):
        async with peer_model() as (peer, model, _):
            session = model.session()
            original = messages("one", "two", "three")
            await session._update_session(chat_ctx=original)
            target = original.copy()
            if edit == "append":
                target.add_message(role="system", content="four")
            elif edit in ("prepend", "middle"):
                target.items.insert(
                    0 if edit == "prepend" else 1,
                    llm.ChatMessage(role="system", content=["inserted"]),
                )
            elif edit == "delete":
                del target.items[1]
            elif edit == "replace":
                target.items[1] = target.items[1].model_copy(update={"content": ["replacement"]})
            elif edit == "reorder":
                target.items.reverse()
            before = len(peer.history_operations)
            await session.update_chat_ctx(target)
            history = await model.get_history()
            assert [i.text for i in history] == [i.text_content for i in target.items]
            assert [i.id for i in session.chat_ctx.items] == [i.id for i in target.items]
            assert peer.requests.empty()
            accepted_operations = len(peer.history_operations)
            if edit == "noop":
                assert accepted_operations == before
            await session.update_chat_ctx(target)
            assert len(peer.history_operations) == accepted_operations
            assert [i.text for i in await model.get_history()] == [
                i.text_content for i in target.items
            ]

    async def test_completed_tool_pair_deletion_and_restart_do_not_execute_tools(self):
        async with peer_model() as (peer, model, _):
            fallback = RealtimeModelFallbackAdapter([model])
            session = fallback.session()
            try:
                await session._update_session(chat_ctx=tool_history())
                await fallback.restart_session()
                assert len(await model.get_history()) == 5
                target = session.chat_ctx.copy(exclude_function_call=True)
                await session.update_chat_ctx(target)
                history = await model.get_history()
                assert all(isinstance(i, r.HistoryMessage) for i in history)
                assert len(history) == 3
                assert peer.requests.empty()
            finally:
                await session.aclose()

    @pytest.mark.parametrize("invalid", ["orphan", "duplicate", "split_pair", "image", "encoding"])
    async def test_invalid_batch_sends_nothing_and_keeps_conversation_usable(self, invalid):
        async with peer_model() as (peer, model, _):
            session = model.session()
            original = messages("keep")
            await session._update_session(chat_ctx=original)
            target = original.copy()
            target.add_message(role="system", content="valid addition")
            if invalid == "orphan":
                target.items.append(
                    llm.FunctionCall(call_id="missing", name="lookup", arguments="{}")
                )
            elif invalid == "duplicate":
                target.items.append(original.items[0])
            elif invalid == "split_pair":
                target.items.extend(tool_history().items[2:4])
                target.items.insert(
                    -1, llm.ChatMessage(role="user", content=["between call and result"])
                )
            elif invalid == "image":
                target.add_message(
                    role="user", content=[llm.ImageContent(image="https://test/image")]
                )
            else:
                target.add_message(role="system", content=chr(0xD800))
            before = len(peer.history_operations)
            with pytest.raises(llm.RealtimeError):
                await session.update_chat_ctx(target)
            assert len(peer.history_operations) == before
            assert session.chat_ctx.is_equivalent(original)
            assert len(await model.get_history()) == 1

    async def test_rejected_first_operation_leaves_history_unchanged(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            original = messages("keep")
            await session._update_session(chat_ctx=original)
            peer.refusals["conversation.item.create"] = ("invalid_request", "event")
            target = original.copy()
            target.add_message(role="system", content="rejected")
            with pytest.raises(PrismError):
                await session.update_chat_ctx(target)
            assert session.chat_ctx.is_equivalent(original)
            assert len(await model.get_history()) == 1
            await session.update_chat_ctx(target)
            assert len(await model.get_history()) == 2

    async def test_partial_history_failure_closes_uncertain_conversation(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session._update_session(chat_ctx=messages("old"))
            peer.fail_history_at = len(peer.history_operations) + 2
            errors = asyncio.Queue()
            session.on("error", errors.put_nowait)
            with pytest.raises(llm.RealtimeError, match="partially applied"):
                await session.update_chat_ctx(messages("new"))
            error = await asyncio.wait_for(errors.get(), 3)
            assert not error.recoverable
            with pytest.raises(llm.RealtimeError):
                await session.update_instructions("must not use uncertain history")

    async def test_cancelled_history_update_settles_and_commits_accepted_change(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session._update_session(chat_ctx=messages("old"))
            peer.history_started.clear()
            peer.history_release.clear()
            target = session.chat_ctx
            target.add_message(role="system", content="accepted")
            change = asyncio.create_task(session.update_chat_ctx(target))
            try:
                await asyncio.wait_for(peer.history_started.wait(), 3)
                change.cancel()
                await asyncio.sleep(0)
                change.cancel()
                await asyncio.sleep(0)
                assert not change.done()
            finally:
                peer.history_release.set()
            with pytest.raises(asyncio.CancelledError):
                await change
            assert session.chat_ctx.is_equivalent(target)
            assert len(await model.get_history()) == 2

    async def test_running_reply_snapshot_does_not_remove_live_local_items(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("test")
            ctx = session.chat_ctx
            message = ctx.add_message(role="user", content="question")
            await session.update_chat_ctx(ctx)
            pending = session.generate_reply()
            response_id = await peer.start(await peer.next())
            await pending
            await peer.message(response_id)
            assert [item.text for item in await model.get_history()] == ["question"]
            assert session.chat_ctx.get_by_id(message.id) is not None
            with pytest.raises(llm.RealtimeError, match="active replies"):
                await session.update_chat_ctx(messages("replacement"))
            await peer.end(response_id)
            await wait_until(lambda: len(session.chat_ctx.items) == 2)
            await session.update_chat_ctx(llm.ChatContext.empty())
            assert await model.get_history() == ()

    @pytest.mark.parametrize("removed_type", [llm.FunctionCall, llm.FunctionCallOutput])
    async def test_deleting_half_a_tool_pair_is_rejected_before_mutation(self, removed_type):
        async with peer_model() as (peer, model, _):
            session = model.session()
            ctx = tool_history()
            await session._update_session(chat_ctx=ctx)
            before = len(peer.history_operations)
            target = llm.ChatContext([i for i in ctx.items if not isinstance(i, removed_type)])
            with pytest.raises(llm.RealtimeError):
                await session.update_chat_ctx(target)
            assert len(peer.history_operations) == before
            assert len(await model.get_history()) == 5

    async def test_history_read_refusal_uses_the_adapter_error_and_keeps_session(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("test")
            peer.refusals["prism.conversation.get"] = ("invalid_request", "event")
            with pytest.raises(PrismError) as raised:
                await model.get_history()
            assert raised.value.code == "invalid_request"
            assert await model.get_history() == ()

    async def test_reply_waiting_for_readiness_does_not_block_live_tool_results(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("test")
            first = session.generate_reply()
            parent = await peer.start(await peer.next())
            generation = await first
            await peer.emit(
                "response.function_call_arguments.done",
                response_id=parent,
                item_id="live-call",
                call_id="live-call",
                name="lookup",
                arguments="{}",
            )
            call = await generation.function_stream.__anext__()
            await peer.end(parent)
            await wait_until(lambda: session.chat_ctx.get_by_id(call.id) is not None)
            waiting = session.generate_reply()
            try:
                await asyncio.sleep(0)
                target = session.chat_ctx
                target.items.append(
                    llm.FunctionCallOutput(call_id=call.call_id, output="done", is_error=False)
                )
                await asyncio.wait_for(session.update_chat_ctx(target), 1)
                assert peer.items[-1]["output"] == "done"
                await peer.emit("prism.typed_input.ready")
                response = await peer.start(await peer.next())
                await peer.end(response)
                await asyncio.wait_for(waiting, 3)
            finally:
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)

    @pytest.mark.parametrize("ended", [False, True])
    async def test_reply_started_during_final_rebuild_acknowledgment_requires_recovery(self, ended):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session._update_session(chat_ctx=messages("old"))
            peer.block_history_at = len(peer.history_operations) + 2
            peer.history_release.clear()
            change = asyncio.create_task(session.update_chat_ctx(messages("replacement")))
            errors = asyncio.Queue()
            session.on("error", errors.put_nowait)
            try:
                await wait_until(lambda: len(peer.history_operations) == peer.block_history_at)
                await peer.emit(
                    "response.created",
                    response={"id": "racing-reply", "metadata": {"prism_cause": "user_audio"}},
                )
                if ended:
                    await peer.end("racing-reply")
                    await wait_until(
                        lambda: (
                            session._generations.get("racing-reply")
                            and session._generations["racing-reply"].ended
                        )
                    )
                else:
                    await wait_until(lambda: "racing-reply" in session._generations)
            finally:
                peer.history_release.set()
            with pytest.raises(llm.RealtimeError, match="partially applied"):
                await change
            event = await asyncio.wait_for(errors.get(), 3)
            assert not event.recoverable

    async def test_omitted_new_options_preserve_existing_text_history_requests(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            original = messages("seed")
            await session._update_session(chat_ctx=original)
            target = session.chat_ctx
            target.add_message(role="assistant", content="appended history")
            await session.update_chat_ctx(target)
            assert "truncation" not in peer.settings
            assert [event["item"] for event in peer.history_operations] == [
                {"type": "message", "role": "user", "text": "seed"},
                {"type": "message", "role": "assistant", "text": "appended history"},
            ]

    async def test_rebuild_validates_completed_live_tool_pair_before_mutation(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("Help")
            pending = session.generate_reply()
            response = await peer.start(await peer.next())
            generation = await pending
            await peer.emit(
                "response.function_call_arguments.done",
                response_id=response,
                item_id="live-tool",
                call_id="live-call",
                name="lookup",
                arguments="{}",
            )
            call = await asyncio.wait_for(anext(generation.function_stream), 3)
            await peer.end(response)
            assert [item async for item in generation.function_stream] == []
            completed = session.chat_ctx
            completed.items.append(
                llm.FunctionCallOutput(
                    call_id=call.call_id, name=call.name, output="done", is_error=False
                )
            )
            await session.update_chat_ctx(completed)
            before = len(peer.history_operations)
            invalid = session.chat_ctx
            invalid.items.insert(1, llm.ChatMessage(role="system", content=["between"]))
            with pytest.raises(llm.RealtimeError, match="immediately follow"):
                await session.update_chat_ctx(invalid)
            assert len(peer.history_operations) == before
            assert session.chat_ctx.is_equivalent(completed)
            valid = session.chat_ctx
            valid.items.insert(0, llm.ChatMessage(role="system", content=["before"]))
            await session.update_chat_ctx(valid)
            assert session.chat_ctx.is_equivalent(valid)
            assert [type(item) for item in await model.get_history()] == [
                r.HistoryMessage,
                r.HistoryFunctionCall,
                r.HistoryFunctionResult,
            ]

    async def test_chat_snapshot_changes_do_not_mutate_acknowledged_history(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_chat_ctx(messages("original"))
            snapshot = session.chat_ctx
            snapshot.items[0].content[:] = ["edited"]
            snapshot.items[0].extra["local"] = True
            assert session.chat_ctx.items[0].text_content == "original"
            assert session.chat_ctx.items[0].extra == {}
            before = len(peer.history_operations)
            await session.update_chat_ctx(snapshot)
            assert len(peer.history_operations) == before + 2
            assert (await model.get_history())[0].text == "edited"
            snapshot.items[0].content[:] = ["changed again"]
            assert session.chat_ctx.items[0].text_content == "edited"

    async def test_nonblocking_tool_annotation_is_read_from_execution_context(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("Help")
            pending = session.generate_reply()
            response = await peer.start(await peer.next())
            generation = await pending
            await peer.emit(
                "response.function_call_arguments.done",
                response_id=response,
                item_id="live-tool",
                call_id="live-call",
                name="lookup",
                arguments="{}",
            )
            call = await asyncio.wait_for(anext(generation.function_stream), 3)
            await peer.end(response)
            assert [item async for item in generation.function_stream] == []
            call.extra["__livekit_agents_tool_non_blocking"] = True
            target = llm.ChatContext(
                [
                    call,
                    llm.FunctionCallOutput(
                        call_id=call.call_id, name=call.name, output="working", is_error=False
                    ),
                ]
            )
            before = len(peer.history_operations)
            with pytest.raises(llm.RealtimeError, match="tool progress"):
                await session.update_chat_ctx(target)
            assert len(peer.history_operations) == before
            assert session.chat_ctx.items[0].extra == {}

    async def test_partial_history_failure_does_not_submit_a_queued_edit(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            original = messages("old")
            await session.update_chat_ctx(original)
            before = len(peer.history_operations)
            peer.block_history_at = peer.fail_history_at = before + 2
            peer.history_release.clear()
            first = asyncio.create_task(session.update_chat_ctx(messages("replacement")))
            queued = None
            try:
                await wait_until(lambda: len(peer.history_operations) == before + 2)
                queued = asyncio.create_task(session.update_chat_ctx(messages("queued")))
                await asyncio.sleep(0)
                peer.history_release.set()
                with pytest.raises(llm.RealtimeError, match="partially applied"):
                    await first
                results = await asyncio.gather(queued, return_exceptions=True)
                assert isinstance(results[0], (llm.RealtimeError, asyncio.CancelledError))
                await session.aclose()
                assert len(peer.history_operations) == before + 2
            finally:
                peer.history_release.set()
                first.cancel()
                tasks = [first]
                if queued is not None:
                    queued.cancel()
                    tasks.append(queued)
                await asyncio.gather(*tasks, return_exceptions=True)

    @pytest.mark.parametrize("kept_text", [None, "", "The full", "The full generated answer"])
    async def test_retained_text_updates_history_without_changing_existing_snapshots(
        self, kept_text
    ):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("test")
            events = []
            model.on("item_truncated", events.append)
            item_id = await completed_message(peer, session)
            before = session.chat_ctx.get_by_id(item_id)
            await acknowledge(peer, item_id, kept_text)
            await wait_until(lambda: len(events) == 1)
            after = session.chat_ctx.get_by_id(item_id)
            assert after.content == [before.text_content if kept_text is None else kept_text]
            assert after.created_at == before.created_at
            assert before.text_content == "The full generated answer"
            assert events[0] == r.ItemTruncated(
                item_id=item_id, content_index=0, audio_end_ms=120, kept_text=kept_text
            )
            # An acknowledgment from an older server must not erase known retained text.
            await acknowledge(peer, item_id, None)
            await wait_until(lambda: len(events) == 2)
            assert session.chat_ctx.get_by_id(item_id) == after

    async def test_unavailable_retained_text_during_generation_keeps_completion_fallback(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("test")
            events = []
            model.on("item_truncated", events.append)
            response_id = await peer.start()
            item_id = await peer.message(response_id, text="Generated words")
            await acknowledge(peer, item_id, None)
            await wait_until(lambda: events)
            await peer.end(response_id)
            await wait_until(lambda: session.chat_ctx.get_by_id(item_id) is not None)
            assert session.chat_ctx.get_by_id(item_id).text_content == "Generated words"

    @pytest.mark.parametrize("kept_text", ["The full", ""])
    @pytest.mark.parametrize("played_text", [None, "The"])
    async def test_early_retained_text_survives_response_completion(self, kept_text, played_text):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("test")
            events, ended = [], []
            model.on("item_truncated", events.append)
            model.on("response_ended", ended.append)
            response_id = await peer.start()
            item_id = await peer.message(response_id, text="The full generated answer")
            await wait_until(
                lambda: (
                    response_id in session._generations
                    and item_id in session._generations[response_id].outputs
                )
            )
            if played_text is not None:
                session.truncate(
                    message_id=item_id,
                    modalities=["audio"],
                    audio_end_ms=120,
                    audio_transcript=played_text,
                )
                await wait_until(lambda: events)
                events.clear()
            await acknowledge(peer, item_id, kept_text)
            await wait_until(lambda: events)
            # Missing text in a later acknowledgment must not erase a known value.
            await acknowledge(peer, item_id, None)
            await wait_until(lambda: len(events) == 2)
            await peer.end(response_id)
            await wait_until(lambda: ended)
            assert session.chat_ctx.get_by_id(item_id).content == [kept_text]

    @pytest.mark.parametrize("edit", ["delete", "replace"])
    async def test_late_acknowledgment_cannot_restore_deleted_or_replaced_history(self, edit):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("test")
            events = []
            model.on("item_truncated", events.append)
            item_id = await completed_message(peer, session)
            target = session.chat_ctx
            if edit == "delete":
                target.items.clear()
            else:
                target.get_by_id(item_id).content = ["Replacement"]
            await session.update_chat_ctx(target)
            await acknowledge(peer, item_id, "Stale words")
            await wait_until(lambda: events)
            assert session.chat_ctx.items == target.items

    @pytest.mark.parametrize("edit", ["append", "delete", "replace"])
    async def test_acknowledgment_during_history_edit_preserves_the_committed_result(self, edit):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("test")
            events = []
            model.on("item_truncated", events.append)
            item_id = await completed_message(peer, session)
            target = session.chat_ctx
            if edit == "append":
                target.add_message(role="user", content="Next question")
            elif edit == "delete":
                target.items.clear()
            else:
                target.get_by_id(item_id).content = ["Replacement"]
            peer.history_started.clear()
            peer.history_release.clear()
            update = asyncio.create_task(session.update_chat_ctx(target))
            try:
                await asyncio.wait_for(peer.history_started.wait(), 3)
                await acknowledge(peer, item_id, "Retained words")
                await wait_until(lambda: events)
            finally:
                peer.history_release.set()
                await asyncio.wait_for(update, 3)
            if edit == "append":
                assert session.chat_ctx.get_by_id(item_id).text_content == "Retained words"
                assert session.chat_ctx.items[-1].text_content == "Next question"
            else:
                assert session.chat_ctx.items == target.items

    async def test_unknown_item_and_other_content_part_are_diagnostics_only(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("test")
            events = []
            model.on("item_truncated", events.append)
            item_id = await completed_message(peer, session)
            before = session.chat_ctx
            await acknowledge(peer, "unknown", "Unrelated")
            await acknowledge(peer, item_id, "Other part", content_index=1)
            await wait_until(lambda: len(events) == 2)
            assert session.chat_ctx.items == before.items

    @pytest.mark.parametrize(
        "summary",
        [None, {"items_left_out": 3}, {"items_left_out": 3, "first_kept_item_id": "kept"}],
    )
    @pytest.mark.parametrize("status", ["completed", "cancelled", "failed"])
    async def test_response_summary_is_exposed_without_deleting_history(self, summary, status):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("test")
            item_id = await completed_message(peer, session)
            before = session.chat_ctx
            events = []
            model.on("response_ended", events.append)
            response_id = await peer.start()
            await peer.emit(
                "response.done",
                response={
                    "id": response_id,
                    "status": status,
                    **({} if summary is None else {"prism_truncation": summary}),
                },
            )
            await wait_until(lambda: events)
            assert events[0].response.response_id == response_id
            assert events[0].status == status
            assert events[0].truncation == (
                None if summary is None else r.ResponseTruncation(**summary)
            )
            assert session.chat_ctx.items == before.items
            # A completed response reference is not required to process a later acknowledgment.
            session._generations.clear()
            truncated = []
            model.on("item_truncated", truncated.append)
            await acknowledge(peer, item_id, "Retained")
            await wait_until(lambda: truncated)
            assert session.chat_ctx.get_by_id(item_id).text_content == "Retained"

    async def test_retained_server_text_does_not_replace_agent_playback_transcript(self):
        async with peer_model() as (peer, model, _):
            output = FakeAudioOutput(can_pause=True)
            async with AgentSession(llm=model) as session:
                session.output.audio = output
                await session.start(Agent(instructions="test"))
                handle = session.generate_reply()
                request = await peer.next()
                response_id = await peer.start(request)
                item_id = await peer.message(response_id, seconds=1)
                await wait_until(lambda: output._started_at is not None)
                await peer.emit(
                    "input_audio_buffer.speech_started", item_id="user-1", audio_start_ms=0
                )
                await peer.end(response_id, status="cancelled")
                await asyncio.wait_for(handle.wait_for_playout(), 5)
                await asyncio.wait_for(peer.truncations.get(), 3)
                before = session.history.to_dict()
                events = []
                model.on("item_truncated", events.append)
                await acknowledge(peer, item_id, "All words retained by the server")
                await wait_until(
                    lambda: events and events[-1].kept_text == "All words retained by the server"
                )
                assert model._session.chat_ctx.get_by_id(item_id).text_content == (
                    "All words retained by the server"
                )
                assert session.history.to_dict() == before
