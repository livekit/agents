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
from fractions import Fraction
from importlib.metadata import version
from importlib.util import find_spec
from types import SimpleNamespace
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
    StopResponse,
    ToolError,
    function_tool,
    llm,
)
from livekit.agents.llm import RealtimeModelFallbackAdapter
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.telemetry import trace_types, utils as telemetry_utils
from livekit.agents.telemetry.traces import _DynamicTracer
from livekit.agents.tts import tts as tts_module
from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities
from livekit.plugins import rime
from livekit.plugins.rime import TTS
from livekit.plugins.rime._websocket_v1 import _codec_for_protocol

from .fake_io import FakeAudioOutput
from .fake_tts import FakeTTS

pytestmark = pytest.mark.unit

HAS_REALTIME_SDK = find_spec("rimelabs_sdk") is not None
if HAS_REALTIME_SDK:
    from rimelabs_sdk import (
        Rime,
        RimeInputError,
        RimeResourceLimitError,
        RimeStreamError,
        RimeTimeoutError,
        realtime as r,
    )
    from websockets.asyncio.server import serve

    from livekit.plugins.rime.realtime import (
        PrismError,
        PrismHistoryUpdateError,
        PrismModelError,
        realtime_model,
    )
    from livekit.plugins.rime.realtime._playback import Playback


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
    payloads: list[dict[str, object]] = []

    async def synthesize(request: web.Request) -> web.Response:
        payloads.append(await request.json())
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
                sample_rate=22050,
                base_url=f"http://127.0.0.1:{port}/tts",
                http_session=session,
            )
            stream = tts.synthesize("hello", conn_options=APIConnectOptions(max_retry=0, timeout=2))
            tts.update_options(sample_rate=16000)
            try:
                await asyncio.wait_for(request_received.wait(), timeout=2)
                tts.update_options(model="mistv2")
                release_response.set()
                events = [event async for event in stream]
                next_stream = tts.synthesize(
                    "next", conn_options=APIConnectOptions(max_retry=0, timeout=2)
                )
                try:
                    next_events = [event async for event in next_stream]
                finally:
                    await next_stream.aclose()
            finally:
                release_response.set()
                await stream.aclose()
                await tts.aclose()
    finally:
        await runner.cleanup()

    assert [(p["modelId"], p["samplingRate"]) for p in payloads] == [
        ("coda", 22050),
        ("mistv2", 16000),
    ]
    assert events[0].frame.sample_rate == 22050
    assert next_events[0].frame.sample_rate == 16000


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
            (12000 * np.sin(2 * np.pi * 440 * np.arange(2400) / 24000)).astype(np.int16)[None, :],
            format="s16",
            layout="mono",
        )
        frame.sample_rate = 24000
        frame.pts = 0
        frame.time_base = Fraction(1, 24000)
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
        "audio/pcmu": b"\x80\x00" * 1200,
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

    samples = np.frombuffer(b"".join(bytes(event.frame.data) for event in events), dtype="<i2")
    assert np.max(np.abs(samples.astype(np.int32))) > 0
    assert all(
        event.frame.sample_rate == 24000 and event.frame.num_channels == 1 for event in events
    )
    expected = len(_PCM) // 2 if audio_format == "audio/pcm" else 2400
    # The shared low-latency decoder can consume one 20 ms Opus packet while probing.
    # MP3 can retain encoder padding. Other formats must preserve every sample.
    missing = 480 if "opus" in audio_format else 0
    padding = 1152 if audio_format == "audio/mpeg" else 0
    assert expected - missing <= len(samples) <= expected + padding
    if audio_format == "audio/pcm":
        assert samples.tobytes() == _PCM
    elif audio_format == "audio/pcmu":
        np.testing.assert_array_equal(samples, np.tile([32124, -32124], 1200))
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


async def test_v1_stream_snapshots_endpoint_speaker_and_sample_rate() -> None:
    async with _RimeV1Server() as first_server, _RimeV1Server() as second_server:
        tts = _v1_tts(first_server, sample_rate=22050)
        first_stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        tts.update_options(
            websocket_url=second_server.websocket_url, speaker="changed", sample_rate=16000
        )
        second_stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        try:
            first_stream.push_text("first")
            first_stream.end_input()
            second_stream.push_text("second")
            second_stream.end_input()
            first_events, second_events = await asyncio.gather(
                _collect(first_stream), _collect(second_stream)
            )
        finally:
            await first_stream.aclose()
            await second_stream.aclose()
            await tts.aclose()

    for server, text, speaker, rate, events in [
        (first_server, "first ", "lyra", 22050, first_events),
        (second_server, "second ", "changed", 16000, second_events),
    ]:
        assert [request["text"] for request in server.requests if "text" in request] == [text]
        start = server.requests[0]["start"]
        assert start["speaker"] == speaker
        assert start["audioParameters"]["samplingRate"] == rate
        assert events and all(event.frame.sample_rate == rate for event in events)


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


@pytest.mark.parametrize(
    "websocket_protocol,message,kind,status_code,retryable,explicit_request_id",
    [
        ("binary", None, "unavailable", 503, True, False),
        ("binary", "", "unimplemented", 501, False, True),
        ("json", None, "unimplemented", 501, False, True),
        ("json", "", "unavailable", 503, True, False),
    ],
)
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
        self.imported_calls = set()
        self.history_operations = []
        self.history_reads = 0
        self.history_started = asyncio.Event()
        self.history_release = asyncio.Event()
        self.history_release.set()
        self.fail_history_at = None
        self.fail_history_scope = "event"
        self.close_history_at = None
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
        elif kind == "conversation.item.truncated" and "prism_kept_text" in body:
            for item in self.server_items:
                if item["id"] == body["item_id"]:
                    item["text"] = body["prism_kept_text"]
        await self.socket.send(json.dumps({"type": kind, "event_id": "server", **body}))

    async def handler(self, socket):
        self.socket = socket
        self.connections += 1
        self.settings = None
        self.server_items = []
        self.imported_calls = set()
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
                if self.close_history_at == len(self.history_operations):
                    await socket.close()
                    continue
                if self.fail_history_at == len(self.history_operations):
                    await self.emit(
                        "error",
                        error={
                            "code": "invalid_request",
                            "message": "History refused",
                            "scope": self.fail_history_scope,
                            "owner": {
                                "kind": self.fail_history_scope,
                                "event_id": event["event_id"],
                            },
                        },
                    )
                    continue
                if kind == "conversation.item.create":
                    self.history_serial += 1
                    item = {"id": f"history-{self.history_serial}", **event["item"]}
                    self.items.append(event["item"])
                    # Imported results have server-defined placement after their call.
                    if item["type"] == "function_call":
                        self.imported_calls.add(item["call_id"])
                    if (
                        item["type"] == "function_call_output"
                        and item["call_id"] in self.imported_calls
                    ):
                        call_index = next(
                            index
                            for index, call in enumerate(self.server_items)
                            if call.get("type") == "function_call"
                            and call["call_id"] == item["call_id"]
                        )
                        self.server_items.insert(call_index + 1, item)
                    else:
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
                self.history_reads += 1
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
async def peer_model(*, process_request=None, **options):
    peer = Peer()
    async with serve(peer.handler, "127.0.0.1", 0, process_request=process_request) as server:
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


def collect_payloads(session, payload_type, received):
    @session.on("rime_server_event_received")
    def on_event(event):
        if isinstance(event.payload, payload_type):
            received.append(event.payload)


async def acknowledge(peer, item_id, kept_text, *, content_index=0):
    await peer.emit(
        "conversation.item.truncated",
        item_id=item_id,
        content_index=content_index,
        audio_end_ms=120,
        **({} if kept_text is None else {"prism_kept_text": kept_text}),
    )


USAGE = {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20, "context_window": 100}


async def finish_metric_response(peer, response_id, *, status="completed", usage=USAGE):
    response = {"id": response_id, "status": status}
    if usage is not None:
        response["usage"] = usage
    await peer.emit("response.done", response=response)


class ControlledOutput(AudioOutput):
    def __init__(self, *, sample_rate=None, synchronous=False):
        super().__init__(
            label="controlled",
            capabilities=AudioOutputCapabilities(pause=True),
            sample_rate=sample_rate,
        )
        self.frames = []
        self.segments = []
        self.duration = 0
        self.synchronous = synchronous
        self.paused = False

    async def capture_frame(self, frame):
        await super().capture_frame(frame)
        self.frames.append(frame)
        self.duration += frame.duration

    def flush(self):
        super().flush()
        if self.duration:
            self.segments.append(self.duration)
            self.duration = 0
            if self.synchronous:
                self.finish()

    def finish(self, *, interrupted=False):
        duration = self.segments.pop(0)
        self.on_playback_finished(playback_position=duration, interrupted=interrupted)

    def clear_buffer(self):
        if self.duration:
            self.flush()
        if self.segments:
            self.finish(interrupted=True)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


class CountingTTS(FakeTTS):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def stream(self, **kwargs):
        self.calls += 1
        return super().stream(**kwargs)

    def synthesize(self, text, **kwargs):
        self.calls += 1
        return super().synthesize(text, **kwargs)


async def emit_message(peer, response, name, *, seconds=0.02):
    await peer.emit(
        "response.output_item.added",
        response_id=response,
        output_index=0,
        item={"id": name, "type": "message"},
    )
    fields = {"response_id": response, "output_index": 0, "item_id": name, "content_index": 0}
    await peer.emit("response.text.delta", **fields, delta=name)
    await peer.emit("response.text.done", **fields, text=name)
    await peer.emit(
        "response.audio.delta",
        **fields,
        delta=base64.b64encode(b"\0\0" * int(24000 * seconds)).decode(),
    )
    await peer.emit("response.audio.done", **fields)


async def emit_caller(peer, item_id, text):
    await peer.emit("input_audio_buffer.speech_started", item_id=item_id, audio_start_ms=0)
    await peer.emit("input_audio_buffer.speech_stopped", item_id=item_id, audio_end_ms=20)
    await peer.emit("input_audio_buffer.committed", item_id=item_id)
    await peer.emit(
        "conversation.item.input_audio_transcription.completed", item_id=item_id, transcript=text
    )


async def emit_note_call(peer, response):
    await peer.emit(
        "response.function_call_arguments.done",
        response_id=response,
        item_id="note-tool",
        call_id="note-call",
        name="save_note",
        arguments="{}",
    )
    return await peer.message(response, text="Got it, I'm making a note of that.")


async def retain_note_prefix(peer, provider, item_id):
    await peer.emit(
        "conversation.item.truncated",
        item_id=item_id,
        content_index=0,
        audio_end_ms=10,
        prism_kept_text="Got it",
    )
    await wait_until(lambda: provider.chat_ctx.get_by_id(item_id).text_content == "Got it")


async def end_note_response(peer, provider, response):
    ended = asyncio.Event()

    @provider.on("rime_server_event_received")
    def on_event(event):
        if isinstance(event.payload, r.ResponseEnded):
            if event.payload.response.response_id == response:
                ended.set()

    await peer.end(response)
    await asyncio.wait_for(ended.wait(), 3)


async def pending_note_snapshot(peer, provider):
    await provider.update_instructions("test")
    await emit_caller(peer, "caller-1", "Next week.")
    response = await peer.start()
    item_id = await emit_note_call(peer, response)
    await end_note_response(peer, provider, response)
    await wait_until(lambda: provider.chat_ctx.get_by_id(item_id) is not None)
    desired = provider.chat_ctx.copy()
    desired.items.append(
        llm.FunctionCallOutput(
            call_id="note-call",
            name="save_note",
            output="The note was saved.",
            is_error=False,
            reply_required=False,
        )
    )
    await retain_note_prefix(peer, provider, item_id)
    return desired, item_id


async def finish_discard_response(peer, response):
    await peer.end(response)
    receipt = await asyncio.wait_for(peer.receipts.get(), 3)
    assert receipt["response_id"] == response


async def tool_turn(peer, *calls):
    response = await peer.start()
    for call in calls:
        await peer.emit(
            "response.function_call_arguments.done",
            response_id=response,
            item_id=call,
            call_id=call,
            name="lookup",
            arguments="{}",
        )
    await finish_discard_response(peer, response)
    return response


async def submit(provider, call, *, reply_required=True):
    await provider.update_chat_ctx(
        llm.ChatContext(
            [
                llm.FunctionCallOutput(
                    call_id=call, output="done", is_error=False, reply_required=reply_required
                )
            ]
        )
    )


async def reply(peer, provider):
    pending = provider.generate_reply()
    request = await peer.next()
    response = await peer.start(request)
    await finish_discard_response(peer, response)
    await pending
    return request


@pytest.mark.skipif(not HAS_REALTIME_SDK, reason="Rime realtime SDK is not installed")
class TestRealtime:
    @pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
    async def test_startup_retries_refusal_and_preserves_waiting_reply(self, status):
        attempts = 0

        async def refuse_first(connection, request):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return connection.respond(status, "busy")

        async with peer_model(
            process_request=refuse_first,
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
        ) as (peer, model, _):
            session = model.session()
            errors = []
            session.on("error", errors.append)
            reply = session.generate_reply(instructions="Say hello")
            request = await peer.next()
            response_id = await peer.start(request)
            await peer.end(response_id)
            assert (await reply).response_id == response_id
            assert attempts == 2
            assert peer.connections == 1
            assert [e.recoverable for e in errors] == [True]
            assert [e.retryable for e in errors] == [True]

    @pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 501])
    async def test_startup_does_not_retry_permanent_refusal(self, status):
        attempts = 0

        async def refuse(connection, request):
            nonlocal attempts
            attempts += 1
            return connection.respond(status, "refused")

        async with peer_model(process_request=refuse) as (_, model, _):
            session = model.session()
            errors = []
            session.on("error", errors.append)
            with pytest.raises(Exception, match=f"HTTP {status}"):
                await session.update_instructions("test")
            assert attempts == 1
            assert [e.recoverable for e in errors] == [False]

    @pytest.mark.parametrize("max_retry", [0, 2])
    async def test_startup_retry_budget(self, max_retry):
        attempts = 0

        async def refuse(connection, request):
            nonlocal attempts
            attempts += 1
            return connection.respond(503, "busy")

        async with peer_model(
            process_request=refuse,
            conn_options=APIConnectOptions(
                max_retry=max_retry,
                retry_interval=0,
            ),
        ) as (_, model, _):
            session = model.session()
            errors = []
            session.on("error", errors.append)
            with pytest.raises(Exception, match="HTTP 503"):
                await session.update_instructions("test")
            assert attempts == max_retry + 1
            assert [e.recoverable for e in errors] == [True] * max_retry + [False]

    async def test_startup_close_cancels_retry_delay(self):
        attempts = 0
        refused = asyncio.Event()

        async def refuse(connection, request):
            nonlocal attempts
            attempts += 1
            response = connection.respond(503, "busy")
            response.headers["Retry-After"] = "2"
            return response

        async with peer_model(process_request=refuse) as (_, model, _):
            session = model.session()
            session.on("error", lambda _: refused.set())
            waiting = asyncio.create_task(session.update_instructions("test"))
            await asyncio.wait_for(refused.wait(), 2)
            await session.aclose()
            await asyncio.gather(waiting, return_exceptions=True)
            await asyncio.sleep(0.15)
            assert attempts == 1

    async def test_startup_retry_after_cannot_exceed_deadline(self):
        attempts = 0

        async def refuse(connection, request):
            nonlocal attempts
            attempts += 1
            response = connection.respond(503, "busy")
            response.headers["Retry-After"] = "60"
            return response

        async with peer_model(process_request=refuse, startup_timeout=0.2) as (_, model, _):
            session = model.session()
            with pytest.raises(Exception, match="HTTP 503"):
                await asyncio.wait_for(session.update_instructions("test"), 1)
            assert attempts == 1

    @pytest.mark.parametrize(
        "options",
        [
            {"startup_timeout": 0.1},
            {"conn_options": APIConnectOptions(timeout=0.1)},
        ],
    )
    async def test_startup_deadline_closes_unfinished_initialization(self, options):
        async with peer_model(**options) as (peer, model, client):
            peer.update_release.clear()
            session = model.session()
            try:
                with pytest.raises(Exception, match="timed out|no acknowledgment"):
                    await asyncio.wait_for(session.update_instructions("test"), 1)
            finally:
                # The SDK settles the submitted settings request before its
                # cancellation cleanup closes the partially opened session.
                peer.update_release.set()
            await asyncio.wait_for(peer.socket.wait_closed(), 3)
            assert session._shutdown_task is not None
            await asyncio.wait_for(asyncio.shield(session._shutdown_task), 3)
            assert not client.realtime._opening
            assert not client.realtime._sessions
            assert peer.connections == 1

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

    async def test_forced_tool_choice_preserves_session(self):
        async with peer_model() as (_, model, _):
            session = model.session()
            errors = []
            session.on("error", errors.append)
            await session._update_session(instructions="test")
            with pytest.raises(llm.RealtimeError):
                session.update_options(tool_choice="required")
            assert errors == []
            await session.update_instructions("still usable")

    async def test_unsupported_inputs_warn_and_keep_session_and_tools(self, caplog):
        async with peer_model() as (peer, model, _):
            session = model.session()
            errors = []
            session.on("error", errors.append)
            await session.update_tools([lookup])
            frame = rtc.VideoFrame(2, 2, rtc.VideoBufferType.RGBA, bytes(16))
            session.push_video(frame)
            session.commit_audio()
            reply = session.generate_reply(tools=[])
            response = await peer.start(await peer.next())
            await reply
            await peer.end(response)
            await session.update_instructions("still usable")
            assert len(peer.settings["tools"]) == 1
            assert "lookup" in session.tools.function_tools
            assert "ignoring frame" in caplog.text
            assert "ignoring commit_audio" in caplog.text
            assert "ignoring tools override" in caplog.text
            assert not errors and peer.connections == 1

    async def test_invalid_session_request_does_not_start_or_stop_a_connection(self):
        async with peer_model() as (peer, model, _):
            with pytest.raises(llm.RealtimeError, match="server turn detection"):
                model.session(turn_detection_disabled=True)
            assert model._session is None and peer.connections == 0
            session = model.session()
            await session.update_instructions("original")
            with pytest.raises(llm.RealtimeError, match="previous Prism session"):
                model.session()
            assert model._session is session
            await session.update_instructions("still usable")
            assert peer.settings["instructions"] == "still usable"
            assert peer.connections == 1

    @pytest.mark.parametrize("refused_at", [1, 2])
    async def test_rejected_imported_result_can_complete_its_accepted_call(self, refused_at):
        async with peer_model() as (peer, model, _):
            async with AgentSession(llm=model) as owner:
                agent = Agent(instructions="test")
                await owner.start(agent)
                session = model._session
                await session._update_session(instructions="test")
                target = llm.ChatContext(tool_history().items[2:4])
                peer.fail_history_at = refused_at
                with pytest.raises(PrismHistoryUpdateError):
                    await session.update_chat_ctx(target)
                await model._reconcile_chat_ctx(agent)
                assert len(session.chat_ctx.items) == refused_at - 1
                await session.update_chat_ctx(session.chat_ctx)
                pending = session.generate_reply()
                response = await peer.start(await peer.next())
                await peer.end(response)
                await pending
                await session.update_chat_ctx(target)
                assert len(await model._get_history()) == 2
                assert len(peer.history_operations) == 3
                pending = session.generate_reply()
                response = await peer.start(await peer.next())
                await peer.end(response)
                await pending

    @pytest.mark.parametrize("active_reply", [False, True])
    async def test_imported_result_retry_preserves_new_speech_and_acknowledged_ids(
        self, active_reply
    ):
        async with peer_model(context_mode="strict") as (peer, model, _):
            async with AgentSession(llm=model) as owner:
                agent = Agent(instructions="test")
                await owner.start(agent)
                session = model._session
                await session._update_session(instructions="test")
                errors = []
                session.on("error", errors.append)
                target = llm.ChatContext(tool_history().items[2:4])
                peer.fail_history_at = 2
                with pytest.raises(PrismHistoryUpdateError):
                    await session.update_chat_ctx(target)
                await model._reconcile_chat_ctx(agent)
                call_id = (await model._get_history())[0].id
                peer.server_items.append(
                    {"id": "speech", "type": "message", "role": "user", "text": "Keep this"}
                )
                await peer.emit(
                    "conversation.item.input_audio_transcription.completed",
                    item_id="speech",
                    transcript="Keep this",
                )
                await wait_until(lambda: len(session.chat_ctx.items) == 2)
                target.items.append(session.chat_ctx.items[-1])
                response = await peer.start() if active_reply else None
                if response:
                    await wait_until(lambda: response in session._history._active)
                peer.fail_history_at = 3
                with pytest.raises(PrismHistoryUpdateError):
                    await session.update_chat_ctx(target)
                await model._reconcile_chat_ctx(agent)
                assert [i.id for i in session.chat_ctx.items] == [target.items[0].id, "speech"]

                peer.history_started.clear()
                peer.history_release.clear()
                change = asyncio.create_task(session.update_chat_ctx(target))
                try:
                    await asyncio.wait_for(peer.history_started.wait(), 3)
                    # Do not record the result before Prism acknowledges it.
                    assert len(session.chat_ctx.items) == 2
                    peer.server_items.append(
                        {"id": "later", "type": "message", "role": "user", "text": "Keep this too"}
                    )
                    await peer.emit(
                        "conversation.item.input_audio_transcription.completed",
                        item_id="later",
                        transcript="Keep this too",
                    )
                    await wait_until(lambda: len(session.chat_ctx.items) == 3)
                finally:
                    peer.history_release.set()
                await change
                assert [i.id for i in session.chat_ctx.items] == [
                    *[i.id for i in target.items],
                    "later",
                ]
                history = await model._get_history()
                assert [i.id for i in history] == [call_id, history[1].id, "speech", "later"]
                assert history[1].call_id == target.items[0].call_id
                assert [i["type"] for i in peer.history_operations] == [
                    "conversation.item.create"
                ] * 4
                assert all(
                    i["item"]["type"] == "function_call_output" for i in peer.history_operations[1:]
                )
                await session.update_chat_ctx(session.chat_ctx)
                assert len(peer.history_operations) == 4
                if response:
                    await peer.end(response)
                pending = session.generate_reply()
                response = await peer.start(await peer.next())
                await peer.end(response)
                await pending
                assert not any(not error.recoverable for error in errors)

    @pytest.mark.parametrize("failure", ["event", "session", "disconnect"])
    async def test_result_insertions_keep_accepted_prefix_and_reject_uncertain_outcomes(
        self, failure
    ):
        async with peer_model(context_mode="strict") as (peer, model, _):
            async with AgentSession(llm=model) as owner:
                agent = Agent(instructions="test")
                await owner.start(agent)
                session = model._session
                await session._update_session(instructions="test")
                errors = []
                session.on("error", errors.append)
                pairs = []
                for index in range(2):
                    call = llm.FunctionCall(
                        call_id=f"imported-{index}", name="lookup", arguments="{}"
                    )
                    result = llm.FunctionCallOutput(
                        call_id=call.call_id, output="42", is_error=False
                    )
                    pairs.extend([call, result])
                    target = session.chat_ctx
                    target.items.extend([call, result])
                    peer.fail_history_at = len(peer.history_operations) + 2
                    with pytest.raises(PrismHistoryUpdateError):
                        await session.update_chat_ctx(target)
                    await model._reconcile_chat_ctx(agent)
                target = session.chat_ctx
                target.add_message(role="user", content="Newer speech")
                await session.update_chat_ctx(target)
                before = await model._get_history()
                target = llm.ChatContext([*pairs, target.items[-1]])
                if failure == "disconnect":
                    peer.close_history_at = len(peer.history_operations) + 2
                else:
                    peer.fail_history_at = len(peer.history_operations) + 2
                    peer.fail_history_scope = failure
                if failure == "event":
                    with pytest.raises(PrismHistoryUpdateError):
                        await session.update_chat_ctx(target)
                    await model._reconcile_chat_ctx(agent)
                else:
                    with pytest.raises(
                        (PrismHistoryUpdateError, llm.RealtimeError, RimeStreamError)
                    ):
                        await session.update_chat_ctx(target)
                assert [i.id for i in session.chat_ctx.items] == [
                    pairs[0].id,
                    pairs[1].id,
                    pairs[2].id,
                    target.items[-1].id,
                ]
                if failure != "event":
                    await wait_until(lambda: any(not error.recoverable for error in errors))
                    with pytest.raises(
                        (PrismHistoryUpdateError, llm.RealtimeError, RimeStreamError)
                    ):
                        await session.update_chat_ctx(target)
                    return
                accepted_id = (await model._get_history())[1].id
                await session.update_chat_ctx(target)
                after = await model._get_history()
                assert [i.id for i in after] == [
                    before[0].id,
                    accepted_id,
                    before[1].id,
                    after[3].id,
                    before[2].id,
                ]
                assert [i.id for i in session.chat_ctx.items] == [i.id for i in target.items]
                assert len(peer.history_operations) == 8
                assert not any(not error.recoverable for error in errors)
                pending = session.generate_reply()
                response = await peer.start(await peer.next())
                await peer.end(response)
                await pending

    @pytest.mark.parametrize("context_mode", ["strict", "shared"])
    @pytest.mark.parametrize("edit", ["ordinary", "no_reply", "mixed", "uncertain"])
    async def test_partial_tool_result_reply_policy(self, context_mode, edit):
        async with peer_model(context_mode=context_mode, playback="discard") as (peer, model, _):
            provider = model.session()
            errors = []
            provider.on("error", errors.append)
            await provider.update_instructions("test")
            await tool_turn(peer, "first", "second")
            target = provider.chat_ctx
            for index, call in enumerate(("first", "second")):
                target.items.append(
                    llm.FunctionCallOutput(
                        call_id=call,
                        output="42",
                        is_error=False,
                        reply_required=index == (1 if edit == "no_reply" else 0),
                    )
                )
            peer.fail_history_at = 2
            if edit == "mixed":
                target.add_message(role="user", content="not accepted")
            if edit == "uncertain":
                peer.fail_history_scope = "session"
            with pytest.raises((PrismHistoryUpdateError, asyncio.CancelledError)) as caught:
                await provider.update_chat_ctx(target)
            error = caught.value
            if edit == "uncertain":
                # Terminal SDK failure can close the session before the pending
                # update returns its structured error. Neither path may reply.
                await wait_until(lambda: any(not error.recoverable for error in errors))
                assert peer.requests.empty()
                assert not isinstance(error, llm.RealtimeError)
                if isinstance(error, PrismHistoryUpdateError):
                    assert error.status == "uncertain"
                    assert error.confirmed_chat_ctx is None
                return
            assert isinstance(error, llm.RealtimeError) == (edit == "ordinary")
            assert error.status == "partial"
            assert error.code == f"history_update_{error.status}"
            assert peer.requests.empty()
            assert error.recoverable
            assert [
                i.call_id
                for i in error.confirmed_chat_ctx.items
                if isinstance(i, llm.FunctionCallOutput)
            ] == ["first"]
            if edit == "no_reply":
                return
            request = await reply(peer, provider)
            assert request["response"]["prism_call_id"] == "first"
            assert len(peer.history_operations) == 2
            await provider.update_chat_ctx(target)
            assert len(peer.history_operations) == (4 if edit == "mixed" else 3)

    @pytest.mark.parametrize("context_mode", ["strict", "shared"])
    async def test_explicit_partial_tool_result_edit_still_stops(self, context_mode):
        ran = []
        hold = asyncio.Event()

        @function_tool
        async def lookup() -> str:
            """Look up a value."""
            ran.append(True)
            await asyncio.wait_for(hold.wait(), 3)
            raise StopResponse()

        async with peer_model(context_mode=context_mode) as (
            peer,
            model,
            _,
        ):
            async with AgentSession(llm=model) as session:
                agent = Agent(instructions="test", tools=[lookup])
                await session.start(agent)
                session.generate_reply()
                parent = await peer.start(await peer.next())
                for call in ("first", "second"):
                    await peer.emit(
                        "response.function_call_arguments.done",
                        response_id=parent,
                        item_id=call,
                        call_id=call,
                        name="lookup",
                        arguments="{}",
                    )
                await peer.end(parent)
                await wait_until(lambda: len(ran) == 2)
                target = model._session.chat_ctx
                for call in ("first", "second"):
                    target.items.append(
                        llm.FunctionCallOutput(
                            call_id=call,
                            output="42",
                            is_error=False,
                            reply_required=True,
                        )
                    )
                peer.fail_history_at = 2
                with pytest.raises(PrismHistoryUpdateError) as caught:
                    await model._replace_chat_ctx(agent, target)
                assert caught.value.status == "partial"
                assert not isinstance(caught.value, llm.RealtimeError)
                assert not caught.value.needs_reconciliation
                assert peer.requests.empty()
                hold.set()

    @pytest.mark.parametrize("failure", ["session", "disconnect"])
    async def test_partial_append_with_uncertain_or_session_failure_remains_fatal(self, failure):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="test")
            if failure == "session":
                peer.fail_history_at = 2
                peer.fail_history_scope = "session"
            else:
                peer.close_history_at = 2
            errors = []
            session.on("error", errors.append)
            with pytest.raises((PrismHistoryUpdateError, llm.RealtimeError, RimeStreamError)):
                await session.update_chat_ctx(messages("accepted", "unknown"))
            await wait_until(lambda: any(not error.recoverable for error in errors))
            with pytest.raises((llm.RealtimeError, RimeStreamError)):
                await session.update_instructions("must not continue")

    @pytest.mark.parametrize("context_mode", ["strict", "shared"])
    @pytest.mark.parametrize("refused_at", [1, 2])
    async def test_agent_session_continues_after_refused_tool_result_without_reexecuting_tools(
        self, context_mode, refused_at, caplog
    ):
        ran = []

        @function_tool
        async def lookup() -> str:
            """Look up a value."""
            ran.append(True)
            return "42"

        async with peer_model(context_mode=context_mode) as (
            peer,
            model,
            _,
        ):
            async with AgentSession(llm=model) as session:
                agent = Agent(instructions="test", tools=[lookup])
                await session.start(agent)
                realtime = model._session
                errors = []
                realtime.on("error", errors.append)
                failed_handle = session.generate_reply()
                parent = await peer.start(await peer.next())
                for index in range(2):
                    await peer.emit(
                        "response.function_call_arguments.done",
                        response_id=parent,
                        item_id=f"tool-{index}",
                        call_id=f"call-{index}",
                        name="lookup",
                        arguments="{}",
                    )
                peer.fail_history_at = refused_at
                await peer.end(parent)
                await wait_until(lambda: len(peer.history_operations) == refused_at)
                if refused_at == 2:
                    # The same speech handle continues automatically with the
                    # accepted result; no application generate_reply() is needed.
                    resumed = await peer.next()
                    assert resumed["response"]["prism_call_id"] == "call-0"
                    response = await peer.start(resumed)
                    await peer.end(response)
                await asyncio.wait_for(failed_handle.wait_for_playout(), 3)
                assert peer.requests.empty()
                rejected_target = agent.chat_ctx.copy()
                assert not any(not error.recoverable for error in errors)
                assert "Prism history update failed" in caplog.text
                assert len(ran) == 2
                assert (
                    sum(isinstance(i, llm.FunctionCallOutput) for i in realtime.chat_ctx.items)
                    == refused_at - 1
                )
                assert peer.requests.empty()
                await agent.update_chat_ctx(rejected_target)
                reply = session.generate_reply()
                continuation = await peer.next()
                if refused_at == 1:
                    assert (
                        continuation["response"]["metadata"]["prism_parent_response_id"] == parent
                    )
                else:
                    assert continuation["response"]["prism_call_id"] == "call-1"
                response = await peer.start(continuation)
                await peer.end(response)
                await asyncio.wait_for(reply.wait_for_playout(), 3)
                assert len(ran) == 2
                assert len(peer.history_operations) == 3

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
                    assert report["response"]["prism_instruction"] == (
                        "Continue according to the agent's instructions, using this tool result. "
                        "Reply or call a tool as appropriate."
                    )
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
                assert isinstance(error.error, llm.RealtimeError)
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
                    assert isinstance(error.error, llm.RealtimeError)
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
            assert model._session._sdk.info.request_id == "livekit-validation"
            assert model._session._sdk.info.response_metadata == {"workflow": "billing"}

    async def test_tool_fillers_survive_agent_handoff_and_return(self):
        filler = r.PrismFiller(type="text", text="Checking your order.")
        async with peer_model(tool_fillers={"lookup": filler}) as (peer, model, _):
            async with AgentSession(llm=model) as session:
                first = Agent(instructions="Lookup", tools=[lookup])
                await session.start(first)
                await wait_until(lambda: bool(peer.settings.get("tools")))
                expected = {"type": "text", "text": "Checking your order."}
                assert peer.settings["tools"][0]["prism_filler"] == expected

                session.update_agent(Agent(instructions="No tools"))
                await wait_until(
                    lambda: (
                        peer.settings.get("instructions") == "No tools"
                        and peer.settings.get("tools") == []
                    )
                )
                session.update_agent(first)
                await wait_until(
                    lambda: (
                        peer.settings.get("instructions") == "Lookup"
                        and bool(peer.settings.get("tools"))
                    )
                )
                assert peer.settings["tools"][0]["prism_filler"] == expected
                assert peer.connections == 1

    async def test_tool_fillers_update_while_absent_and_clear_explicitly(self):
        filler = r.PrismFiller(type="text", text="Checking your order.")
        replacement = r.PrismFiller(type="text", text="Looking it up.")
        async with peer_model(tool_fillers={"lookup": filler}) as (peer, model, _):
            session = model.session()
            await session.update_tools([])
            model.update_options(tool_fillers={"lookup": replacement})
            await model._session._wait_for_options()
            assert peer.settings["tools"] == []
            await session.update_tools([lookup])
            assert peer.settings["tools"][0]["prism_filler"]["text"] == "Looking it up."
            peer.refusals["session.update"] = ("invalid_session", "event")
            errors = []
            session.on("error", errors.append)
            model.update_options(tool_fillers={})
            await session._wait_for_options()
            assert len(errors) == 1 and errors[0].recoverable
            await session.update_tools([])
            await session.update_tools([lookup])
            assert peer.settings["tools"][0]["prism_filler"]["text"] == "Looking it up."
            model.update_options(tool_fillers={})
            await model._session._wait_for_options()
            await session.update_tools([])
            await session.update_tools([lookup])
            assert peer.settings["tools"][0].get("prism_filler") is None
            assert peer.connections == 1

    async def test_settings_updates_and_rejection_preserve_session(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="Original")
            await session._update_session(instructions="Billing", tools=[lookup])
            assert peer.settings["instructions"] == "Billing"
            assert peer.updates[-1]["tools"][0]["function"]["name"] == "lookup"
            model.update_options(voice="new-voice")
            await model._session._wait_for_options()
            assert model._session._sdk.info.voice == "new-voice"
            assert model._session._sdk.info.interrupt_on_speech is True
            model.update_options(response_metadata={"step": "2"})
            await model._session._wait_for_options()
            assert model._session._sdk.info.response_metadata == {"step": "2"}
            old_revision = model._session._sdk.info.settings_revision
            peer.refusals["session.update"] = ("invalid_session", "event")
            with pytest.raises(PrismError) as raised:
                await session.update_instructions("Rejected")
            assert raised.value.code == "invalid_session"
            assert raised.value.retryable is False
            assert peer.settings["instructions"] == "Billing"
            assert model._session._sdk.info.settings_revision == old_revision
            await session._update_session(instructions="", tools=[])
            assert peer.settings["instructions"] == ""
            assert peer.settings["tools"] == []
            assert peer.connections == 1

    @pytest.mark.parametrize(
        "options", [{"voice": "new-voice"}, {"response_metadata": {"step": "2"}}]
    )
    async def test_model_update_options_returns_before_acknowledgment(self, options):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("Help")
            before = session._sdk.info
            peer.update_started.clear()
            peer.update_release.clear()
            assert model.update_options(**options) is None
            change = session._options_task
            try:
                await asyncio.wait_for(peer.update_started.wait(), 3)
                assert not change.done()
                assert session._sdk.info == before
                peer.update_release.set()
                await asyncio.wait_for(change, 3)
                for name, value in options.items():
                    assert getattr(session._sdk.info, name) == value
                    assert getattr(model, f"_{name}") == value
            finally:
                peer.update_release.set()
                change.cancel()
                await asyncio.gather(change, return_exceptions=True)

    async def test_cancelled_update_waits_for_acknowledged_settings(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="Original")
            peer.update_started.clear()
            peer.update_release.clear()
            change = asyncio.create_task(session.update_tools([lookup]))
            await asyncio.wait_for(peer.update_started.wait(), 3)
            for _ in range(2):
                change.cancel()
                await asyncio.sleep(0)
                assert not change.done()
            peer.update_release.set()
            with pytest.raises(asyncio.CancelledError):
                await change
            assert "lookup" in session.tools.function_tools
            assert peer.settings["tools"][0]["function"]["name"] == "lookup"

    async def test_queued_options_copy_inputs_and_settle_before_tools_and_reply(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("Help")
            peer.update_started.clear()
            peer.update_release.clear()
            labels = {"step": "queued"}
            fillers = {"lookup": r.PrismFiller(type="text", text="Checking.")}
            model.update_options(voice="first")
            await asyncio.wait_for(peer.update_started.wait(), 3)
            session.update_options(voice="second", response_metadata=labels, tool_fillers=fillers)
            labels["step"] = "mutated"
            fillers.clear()
            tools = asyncio.create_task(session.update_tools([lookup]))
            reply = session.generate_reply()
            try:
                await asyncio.sleep(0.01)
                assert not tools.done() and peer.requests.empty()
                peer.update_release.set()
                await asyncio.wait_for(tools, 3)
                request = await peer.next()
                assert session._sdk.info.voice == "second"
                assert session._sdk.info.response_metadata == {"step": "queued"}
                assert peer.settings["tools"][0]["prism_filler"]["text"] == "Checking."
                voices = [u["voice"] for u in peer.updates if "voice" in u]
                assert voices[-2:] == ["first", "second"]
                response = await peer.start(request)
                await reply
                await peer.end(response)
            finally:
                peer.update_release.set()
                tools.cancel()
                reply.cancel()
                await asyncio.gather(tools, reply, return_exceptions=True)

    async def test_options_configure_initial_session_without_an_open_connection(self):
        async with peer_model() as (peer, model, _):
            labels = {"step": "initial"}
            assert model.update_options(voice="initial", response_metadata=labels) is None
            labels.clear()
            session = model.session()
            await session.update_instructions("Help")
            assert session._sdk.info.voice == "initial"
            assert session._sdk.info.response_metadata == {"step": "initial"}
            assert peer.connections == 1

    async def test_options_after_close_raise_without_changing_defaults(self):
        async with peer_model(voice="original") as (peer, model, _):
            session = model.session()
            await session.update_instructions("Help")
            await session.aclose()
            with pytest.raises(llm.RealtimeError, match="not open"):
                model.update_options(voice="ignored")
            assert model._voice == "original" and peer.connections == 1

    async def test_queued_options_continue_after_recoverable_refusal(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("Help")
            errors = []
            session.on("error", errors.append)
            peer.refusals["session.update"] = ("invalid_session", "event")
            model.update_options(voice="rejected")
            model.update_options(voice="accepted")
            await session._wait_for_options()
            assert len(errors) == 1 and errors[0].recoverable
            assert session._sdk.info.voice == model._voice == "accepted"
            assert peer.connections == 1 and session._failure is None

    async def test_queued_options_recover_after_local_sdk_resource_refusal(self, monkeypatch):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("Help")
            errors = []
            session.on("error", errors.append)
            before = len(peer.updates)
            with monkeypatch.context() as limit:
                limit.setattr("rimelabs_sdk.realtime._session._LIMIT", 0)
                model.update_options(voice="not submitted")
                await session._wait_for_options()
            assert len(peer.updates) == before
            assert len(errors) == 1 and errors[0].recoverable
            assert isinstance(errors[0].error, RimeResourceLimitError)
            assert session._failure is None
            model.update_options(voice="accepted")
            await session._wait_for_options()
            assert session._sdk.info.voice == model._voice == "accepted"
            assert peer.connections == 1

    async def test_queued_options_sdk_resource_failure_remains_fatal(self, monkeypatch):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("Help")
            errors = []
            session.on("error", errors.append)
            peer.update_started.clear()
            peer.update_release.clear()
            model.update_options(voice="pending")
            try:
                await asyncio.wait_for(peer.update_started.wait(), 3)
                # The update is already submitted. A provider response that exceeds
                # the SDK's response limit terminates the SDK and its pending request.
                with monkeypatch.context() as limit:
                    limit.setattr("rimelabs_sdk.realtime._session._LIMIT", 0)
                    await peer.start(response_id="resource-limit")
                    await wait_until(lambda: any(not error.recoverable for error in errors))
                    await wait_until(lambda: session._closed)
                assert isinstance(session._failure, RimeResourceLimitError)
                assert model._voice != "pending"
                with pytest.raises(RimeResourceLimitError):
                    model.update_options(voice="must not run")
            finally:
                peer.update_release.set()

    async def test_queued_options_session_failure_remains_fatal(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("Help")
            errors = []
            session.on("error", errors.append)
            peer.refusals["session.update"] = ("invalid_session", "session")
            model.update_options(voice="rejected")
            await wait_until(lambda: any(not error.recoverable for error in errors))
            await wait_until(lambda: session._closed)
            assert model._voice != "rejected"

    @pytest.mark.parametrize("operation", ["settings", "history", "options"])
    async def test_session_shutdown_settles_an_update_waiting_for_acknowledgment(self, operation):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("Original")
            if operation == "settings":
                started, release = peer.update_started, peer.update_release
                update = session.update_tools([lookup])
            elif operation == "options":
                started, release = peer.update_started, peer.update_release
                model.update_options(voice="pending")
                update = session._options_task
            else:
                started, release = peer.history_started, peer.history_release
                target = session.chat_ctx
                target.add_message(role="user", content="pending")
                update = session.update_chat_ctx(target)
            started.clear()
            release.clear()
            change = asyncio.ensure_future(update)
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

            emit = peer.emit

            async def refuse_next_update(kind, **body):
                await emit(kind, **body)
                if kind == "session.updated" and body["session"]["voice"] == "new-voice":
                    peer.refusals["session.update"] = ("invalid_session", "event")

            errors = []
            session.on("error", errors.append)
            with patch.object(peer, "emit", side_effect=refuse_next_update):
                model.update_options(voice="new-voice", response_metadata={"step": "2"})
                await session._wait_for_options()
            assert len(errors) == 1 and errors[0].recoverable
            assert errors[0].error.code == "invalid_session"

            assert (
                model._session._sdk.info.voice
                == model._voice
                == peer.settings["voice"]
                == "new-voice"
            )
            assert (
                model._session._sdk.info.response_metadata
                == model._response_metadata
                == {"step": "1"}
            )
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
            events, generations, stopped, transcripts = [], [], [], []
            session.on("rime_server_event_received", events.append)
            session.on("generation_created", generations.append)
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
                event_id="diagnostic-event",
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
            assert len(stopped) == len(generations) == 1
            assert type(stopped[0]) is llm.InputSpeechStoppedEvent
            assert stopped[0].user_transcription_enabled
            assert type(generations[0]) is llm.GenerationCreatedEvent
            assert generations[0].response_id == "diagnostic-response"
            assert not generations[0].user_initiated
            speech_events = [e for e in events if isinstance(e.payload, r.SpeechStopped)]
            assert len(speech_events) == 1
            assert speech_events[0].payload.end_reason == "semantic"
            response_events = [e for e in events if isinstance(e.payload, r.ResponseStarted)]
            assert len(response_events) == 1
            event = response_events[0]
            assert isinstance(event, r.SessionEvent)
            assert event.session_id == "session-1"
            assert event.event_id == "diagnostic-event"
            assert event.request_id is None
            assert event.payload.response.response_id == generations[0].response_id
            assert event.payload.metadata == {"workflow": "billing"}
            assert event.payload.input_item_ids == ("input-1",)
            assert event.payload.settings_revision == peer.revision

    async def test_reply_labels_and_saved_instruction_deletion(self):
        async with peer_model(response_metadata={"call": "shared"}) as (peer, model, _):
            session = model.session()
            await session._update_session(instructions="Help")
            events, generations = [], []
            session.on("rime_server_event_received", events.append)
            session.on("generation_created", generations.append)
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
            assert type(created) is llm.GenerationCreatedEvent
            assert created.user_initiated
            assert created.response_id == response_id
            assert generations == []
            response_events = [e for e in events if isinstance(e.payload, r.ResponseStarted)]
            assert len(response_events) == 1
            event = response_events[0]
            assert event.session_id == "session-1"
            assert event.event_id == "server"
            assert event.request_id == request["event_id"]
            assert event.payload.metadata == {"call": "this-reply", "step": "2"}
            instruction_item_id = event.payload.instruction_item_id
            assert instruction_item_id == "saved-instruction"
            with pytest.raises(RimeInputError, match="before its reply ends"):
                await model._delete_instruction_item(instruction_item_id)
            await peer.message(response_id)
            await peer.end(response_id)
            await wait_until(lambda: session._generations[response_id].ended)
            session._generations.pop(response_id)
            await model._delete_instruction_item(instruction_item_id)
            assert peer.history_operations[-1]["item_id"] == "saved-instruction"
            assert all(item["id"] != "saved-instruction" for item in peer.server_items)
            with pytest.raises(RimeInputError, match="Unknown saved"):
                await model._delete_instruction_item("saved-instruction")

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

    @pytest.mark.parametrize("sample_rate,channels", [(16000, 1), (48000, 2)])
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
            # Distinct samples and stereo channels expose silence, reordering, and downmixing.
            samples = np.arange(sample_rate // 25 * channels, dtype=np.int16) * 7 - 12000
            expected_audio = samples.tobytes()
            frame = rtc.AudioFrame(expected_audio, sample_rate, channels, sample_rate // 25)
            session.push_audio(frame)
            await peer.next("input_audio_buffer.append")
            assert len(chunks) == 1
            assert chunks[0].format.sample_rate == sample_rate
            assert chunks[0].format.channels == channels
            assert chunks[0].data == expected_audio
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
                    assert model._session._sdk.info.truncation is None
                else:
                    assert model._session._sdk.info.truncation == policy
                model.update_options(truncation=r.RetentionRatio(0.7))
                await model._session._wait_for_options()
                assert model._session._sdk.info.truncation == r.RetentionRatio(0.7)
                await fallback.restart_session()
                assert model._session._sdk.info.truncation == r.RetentionRatio(0.7)
            finally:
                await session.aclose()

    async def test_rejected_truncation_leaves_acknowledged_policy_unchanged(self):
        async with peer_model(truncation="auto") as (peer, model, _):
            session = model.session()
            await session.update_instructions("test")
            peer.refusals["session.update"] = ("invalid_session", "event")
            errors = []
            session.on("error", errors.append)
            model.update_options(truncation="disabled")
            await session._wait_for_options()
            assert len(errors) == 1 and errors[0].recoverable
            assert model._session._sdk.info.truncation == "auto"
            model.update_options(truncation="disabled")
            await model._session._wait_for_options()
            assert model._session._sdk.info.truncation == "disabled"

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
                history = await model._get_history()
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
        "edit", ["append", "prepend", "middle", "delete", "replace", "reorder"]
    )
    async def test_context_edits_preserve_local_identity_and_server_order(self, edit):
        async with peer_model(context_mode="strict") as (peer, model, _):
            session = model.session()
            original = messages("one", "two", "three")
            await session._update_session(chat_ctx=original)
            target = original.copy()
            if edit == "append":
                target.add_message(role="assistant", content="four")
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
            await session.update_chat_ctx(target)
            history = await model._get_history()
            assert [i.text for i in history] == [i.text_content for i in target.items]
            assert [i.id for i in session.chat_ctx.items] == [i.id for i in target.items]
            assert peer.requests.empty()
            accepted_operations = len(peer.history_operations)
            if edit == "append":
                assert "truncation" not in peer.settings
                assert [event["item"] for event in peer.history_operations] == [
                    {"type": "message", "role": "user", "text": text}
                    for text in ("one", "two", "three")
                ] + [{"type": "message", "role": "assistant", "text": "four"}]
            await session.update_chat_ctx(target)
            assert len(peer.history_operations) == accepted_operations
            assert [i.text for i in await model._get_history()] == [
                i.text_content for i in target.items
            ]

    async def test_completed_tool_pair_deletion_and_restart_do_not_execute_tools(self):
        async with peer_model(context_mode="strict") as (peer, model, _):
            fallback = RealtimeModelFallbackAdapter([model])
            session = fallback.session()
            try:
                await session._update_session(chat_ctx=tool_history())
                await fallback.restart_session()
                assert len(await model._get_history()) == 5
                target = session.chat_ctx.copy(exclude_function_call=True)
                await session.update_chat_ctx(target)
                history = await model._get_history()
                assert all(isinstance(i, r.HistoryMessage) for i in history)
                assert len(history) == 3
                assert peer.requests.empty()
            finally:
                await session.aclose()

    @pytest.mark.parametrize(
        "invalid",
        [
            "orphan",
            "orphan_result",
            "duplicate",
            "split_pair",
            "image",
            "encoding",
            "developer",
            "long",
        ],
    )
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
            elif invalid == "orphan_result":
                target.items.append(
                    llm.FunctionCallOutput(call_id="unknown", output="Result", is_error=False)
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
            elif invalid == "developer":
                target.add_message(role="developer", content="Unsupported role")
            elif invalid == "long":
                target.add_message(role="user", content="x" * 4001)
            else:
                target.add_message(role="system", content=chr(0xD800))
            before = len(peer.history_operations)
            with pytest.raises(PrismHistoryUpdateError):
                await session.update_chat_ctx(target)
            assert len(peer.history_operations) == before
            assert session.chat_ctx.is_equivalent(original)
            assert len(await model._get_history()) == 1

    async def test_rejected_first_operation_leaves_history_unchanged(self):
        async with peer_model() as (peer, model, _):
            async with AgentSession(llm=model) as owner:
                agent = Agent(instructions="test")
                await owner.start(agent)
                session = model._session
                original = messages("keep")
                await session._update_session(chat_ctx=original)
                peer.refusals["conversation.item.create"] = ("invalid_request", "event")
                target = original.copy()
                target.add_message(role="system", content="rejected")
                with pytest.raises(PrismHistoryUpdateError):
                    await session.update_chat_ctx(target)
                assert session.chat_ctx.is_equivalent(original)
                assert len(await model._get_history()) == 1
                await session.update_chat_ctx(target)
                assert len(await model._get_history()) == 2

    @pytest.mark.parametrize("replacement", [False, True])
    async def test_refused_first_deletion_warns_and_preserves_history(self, replacement, caplog):
        async with peer_model(context_mode="strict") as (peer, model, _):
            async with AgentSession(llm=model) as owner:
                agent = Agent(instructions="test")
                await owner.start(agent)
                session = model._session
                await session._update_session(chat_ctx=messages("keep"))
                original = session.chat_ctx
                server_id = (await model._get_history())[0].id
                peer.fail_history_at = len(peer.history_operations) + 1
                target = messages("replacement") if replacement else llm.ChatContext.empty()
                with pytest.raises(PrismHistoryUpdateError):
                    await session.update_chat_ctx(target)
                assert "Prism history update failed" in caplog.text
                assert session.chat_ctx.is_equivalent(original)
                assert (await model._get_history())[0].id == server_id
                pending = session.generate_reply()
                response = await peer.start(await peer.next())
                await peer.end(response)
                await pending
                await session.update_chat_ctx(target)
                assert session.chat_ctx.is_equivalent(target)

    async def test_partial_history_failure_keeps_confirmed_connection(self):
        async with peer_model(context_mode="strict") as (peer, model, _):
            session = model.session()
            await session._update_session(chat_ctx=messages("old"))
            peer.fail_history_at = len(peer.history_operations) + 2
            errors = []
            session.on("error", errors.append)
            with pytest.raises(PrismHistoryUpdateError, match="partially applied"):
                await session.update_chat_ctx(messages("new"))
            assert not errors
            assert not session.chat_ctx.items
            await session.update_instructions("connection remains usable")
            pending = session.generate_reply()
            response = await peer.start(await peer.next())
            await peer.end(response)
            await pending

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
            assert len(await model._get_history()) == 2

    async def test_running_reply_snapshot_does_not_remove_live_local_items(self):
        async with peer_model(context_mode="strict") as (peer, model, _):
            async with AgentSession(llm=model) as owner:
                agent = Agent(instructions="test")
                await owner.start(agent)
                session = model._session
                await session.update_instructions("test")
                ctx = session.chat_ctx
                message = ctx.add_message(role="user", content="question")
                await session.update_chat_ctx(ctx)
                pending = session.generate_reply()
                response_id = await peer.start(await peer.next())
                await pending
                await peer.message(response_id)
                assert [item.text for item in await model._get_history()] == ["question"]
                assert session.chat_ctx.get_by_id(message.id) is not None
                with pytest.raises(PrismHistoryUpdateError, match="active replies"):
                    await session.update_chat_ctx(messages("replacement"))
                await peer.end(response_id)
                await wait_until(lambda: len(session.chat_ctx.items) == 2)
                await model._reconcile_chat_ctx(agent)
                await session.update_chat_ctx(llm.ChatContext.empty())
                assert await model._get_history() == ()

    @pytest.mark.parametrize("removed_type", [llm.FunctionCall, llm.FunctionCallOutput])
    async def test_deleting_half_a_tool_pair_is_rejected_before_mutation(self, removed_type):
        async with peer_model(context_mode="strict") as (peer, model, _):
            session = model.session()
            ctx = tool_history()
            await session._update_session(chat_ctx=ctx)
            before = len(peer.history_operations)
            target = llm.ChatContext([i for i in ctx.items if not isinstance(i, removed_type)])
            with pytest.raises(PrismHistoryUpdateError):
                await session.update_chat_ctx(target)
            assert len(peer.history_operations) == before
            assert len(await model._get_history()) == 5

    async def test_history_read_refusal_uses_the_adapter_error_and_keeps_session(self):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("test")
            peer.refusals["prism.conversation.get"] = ("invalid_request", "event")
            with pytest.raises(PrismError) as raised:
                await model._get_history()
            assert raised.value.code == "invalid_request"
            assert await model._get_history() == ()

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

    async def test_rebuild_validates_completed_live_tool_pair_before_mutation(self):
        async with peer_model(context_mode="strict") as (peer, model, _):
            async with AgentSession(llm=model) as owner:
                agent = Agent(instructions="test")
                await owner.start(agent)
                session = model._session
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
                with pytest.raises(PrismHistoryUpdateError, match="immediately follow"):
                    await session.update_chat_ctx(invalid)
                assert len(peer.history_operations) == before
                assert session.chat_ctx.is_equivalent(completed)
                valid = session.chat_ctx
                valid.items.insert(0, llm.ChatMessage(role="system", content=["before"]))
                await model._reconcile_chat_ctx(agent)
                await session.update_chat_ctx(valid)
                assert session.chat_ctx.is_equivalent(valid)
                assert [type(item) for item in await model._get_history()] == [
                    r.HistoryMessage,
                    r.HistoryFunctionCall,
                    r.HistoryFunctionResult,
                ]

    async def test_chat_snapshot_changes_do_not_mutate_acknowledged_history(self):
        async with peer_model(context_mode="strict") as (peer, model, _):
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
            assert (await model._get_history())[0].text == "edited"
            snapshot.items[0].content[:] = ["changed again"]
            assert session.chat_ctx.items[0].text_content == "edited"

    async def test_nonblocking_tool_result_is_submitted_once(self):
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
            await session.update_chat_ctx(target)
            await session.update_chat_ctx(target)
            assert len(peer.history_operations) == before + 1
            assert session.chat_ctx.items[-1].output == "working"
            assert session.chat_ctx.items[0].extra == {}

    @pytest.mark.parametrize("context_mode", ["shared", "strict"])
    @pytest.mark.parametrize("silent", [False, True])
    @pytest.mark.parametrize("fails", [False, True])
    async def test_agent_session_tool_progress(self, context_mode, silent, fails):
        next_update, finish = asyncio.Event(), asyncio.Event()
        executions = 0

        @function_tool
        async def lookup(ctx: RunContext) -> str:
            """Look up an order."""
            nonlocal executions
            executions += 1
            await ctx.update("Searching", silent=silent)
            await next_update.wait()
            await ctx.update("Found the order", silent=silent)
            await finish.wait()
            if fails:
                raise ToolError("Order service unavailable")
            return "The order shipped"

        async with peer_model(playback="discard", context_mode=context_mode) as (peer, model, _):
            async with AgentSession(llm=model) as owner:
                await owner.start(Agent(instructions="test", tools=[lookup]))
                speech = owner.generate_reply()
                response = await peer.start(await peer.next())
                await peer.emit(
                    "response.function_call_arguments.done",
                    response_id=response,
                    item_id="lookup-item",
                    call_id="lookup-call",
                    name="lookup",
                    arguments="{}",
                )
                await wait_until(lambda: executions == 1)
                # Prism can finish its filler after the first progress output was
                # created. LiveKit inserts that output before the filler by time.
                await peer.message(response, text="Checking your order")
                await peer.end(response)

                async def reply(text):
                    request = await peer.next()
                    response = await peer.start(request)
                    await peer.message(response, text=text)
                    await peer.end(response)
                    return request

                async def result(call_id):
                    await wait_until(
                        lambda: any(
                            isinstance(i, llm.FunctionCallOutput) and i.call_id == call_id
                            for i in model._session.chat_ctx.items
                        )
                    )

                await result("lookup-call")
                if not silent:
                    await reply("Searching")
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                # Completed references are bounded; their history remains after
                # eviction and must still accept later progress/final results.
                model._session._history._calls.pop("lookup-call")
                next_update.set()
                await result("lookup-call_update_1")
                if not silent:
                    await reply("Found the order")
                await asyncio.wait_for(owner.wait_for_idle(), 3)
                if silent:
                    assert peer.requests.empty()
                finish.set()
                await result("lookup-call_final")
                request = await reply("Please try later" if fails else "The order shipped")
                assert "lookup-call_final" in request["response"]["prism_instruction"]
                await asyncio.wait_for(owner.wait_for_idle(), 3)
                outputs = [
                    i
                    for i in model._session.chat_ctx.items
                    if isinstance(i, llm.FunctionCallOutput)
                ]
                assert [i.call_id for i in outputs] == [
                    "lookup-call",
                    "lookup-call_update_1",
                    "lookup-call_final",
                ]
                assert "Searching" in outputs[0].output
                assert "Found the order" in outputs[1].output
                assert outputs[-1].is_error is fails
                assert outputs[-1].output == (
                    "Order service unavailable" if fails else "The order shipped"
                )
                assert executions == 1
                assert peer.connections == 1
                if context_mode == "shared":
                    before = len(peer.history_operations)
                    for edit in ("output", "insert"):
                        target = owner.current_agent.chat_ctx.copy()
                        first = next(
                            i for i in target.items if isinstance(i, llm.FunctionCallOutput)
                        )
                        if edit == "output":
                            first.output = "Changed progress"
                        else:
                            target.items.insert(
                                target.items.index(first) + 1,
                                llm.ChatMessage(role="user", content=["Inserted before filler"]),
                            )
                        with pytest.raises(PrismHistoryUpdateError):
                            await owner.current_agent.update_chat_ctx(target)
                    assert len(peer.history_operations) == before

    @pytest.mark.parametrize("context_mode", ["shared", "strict"])
    async def test_tool_progress_without_waiting_for_a_reply(self, context_mode):
        executions = 0

        @function_tool
        async def lookup(ctx: RunContext) -> str:
            """Look up an order."""
            nonlocal executions
            executions += 1
            await ctx.update("Searching")
            await ctx.update("Found the order")
            return "The order shipped"

        async with peer_model(playback="discard", context_mode=context_mode) as (peer, model, _):
            async with AgentSession(llm=model) as owner:
                await owner.start(Agent(instructions="test", tools=[lookup]))
                owner.generate_reply()
                response = await peer.start(await peer.next())
                await peer.emit(
                    "response.function_call_arguments.done",
                    response_id=response,
                    item_id="lookup-item",
                    call_id="lookup-call",
                    name="lookup",
                    arguments="{}",
                )
                await peer.end(response)
                replies = []

                async def respond():
                    while True:
                        request = await peer.next()
                        replies.append(request)
                        response = await peer.start(request)
                        await peer.message(response)
                        await peer.end(response)

                responder = asyncio.create_task(respond())
                try:
                    await wait_until(
                        lambda: any(
                            isinstance(i, llm.FunctionCallOutput)
                            and i.call_id == "lookup-call_final"
                            for i in model._session.chat_ctx.items
                        )
                    )
                    await wait_until(
                        lambda: any(
                            "lookup-call_final" in req["response"].get("prism_instruction", "")
                            for req in replies
                        )
                    )
                    await asyncio.wait_for(owner.wait_for_idle(), 3)
                    outputs = [
                        i.call_id
                        for i in model._session.chat_ctx.items
                        if isinstance(i, llm.FunctionCallOutput)
                    ]
                    assert sorted(outputs) == [
                        "lookup-call",
                        "lookup-call_final",
                        "lookup-call_update_1",
                    ]
                    assert executions == 1
                finally:
                    responder.cancel()
                    await asyncio.gather(responder, return_exceptions=True)

    @pytest.mark.parametrize("context_mode", ["shared", "strict"])
    @pytest.mark.parametrize("rapid", [False, True])
    @pytest.mark.parametrize("silent", [False, True])
    @pytest.mark.parametrize("finish_during_reply", [False, True])
    async def test_interleaved_tool_progress_with_late_filler(
        self, context_mode, rapid, silent, finish_during_reply
    ):
        ready: set[str] = set()
        executions: list[str] = []
        later, finish = asyncio.Event(), asyncio.Event()

        async def work(ctx: RunContext, name: str) -> str:
            executions.append(name)
            await ctx.update(f"Searching {name}", silent=silent)
            if rapid:
                await ctx.update(f"Found {name}", silent=silent)
            ready.add(name)
            await later.wait()
            await ctx.update(f"Finishing {name}", silent=silent)
            await finish.wait()
            return f"{name} shipped"

        @function_tool
        async def lookup_a(ctx: RunContext) -> str:
            """Look up the first order."""
            return await work(ctx, "a")

        @function_tool
        async def lookup_b(ctx: RunContext) -> str:
            """Look up the second order."""
            return await work(ctx, "b")

        names = ["a"] if rapid else ["a", "b"]
        async with peer_model(playback="discard", context_mode=context_mode) as (peer, model, _):
            async with AgentSession(llm=model) as owner:
                await owner.start(Agent(instructions="test", tools=[lookup_a, lookup_b]))
                speech = owner.generate_reply()
                response = await peer.start(await peer.next())
                for name in names:
                    await peer.emit(
                        "response.function_call_arguments.done",
                        response_id=response,
                        item_id=f"lookup-{name}-item",
                        call_id=f"lookup-{name}-call",
                        name=f"lookup_{name}",
                        arguments="{}",
                    )
                await wait_until(lambda: ready == set(names))
                await peer.message(response, text="Checking your orders")
                await peer.end(response)
                replies = []

                async def respond():
                    while True:
                        request = await peer.next()
                        replies.append(request)
                        response = await peer.start(request)
                        await peer.message(response)
                        await peer.end(response)

                def output_ids():
                    return [
                        i.call_id
                        for i in model._session.chat_ctx.items
                        if isinstance(i, llm.FunctionCallOutput)
                    ]

                responder = asyncio.create_task(respond())
                try:
                    await asyncio.wait_for(speech.wait_for_playout(), 3)
                    await asyncio.wait_for(owner.wait_for_idle(), 3)
                    later.set()
                    update = "_update_2" if rapid else "_update_1"
                    updates = {f"lookup-{name}-call{update}" for name in names}
                    await wait_until(lambda: updates <= set(output_ids()))
                    if not silent and not finish_during_reply:
                        await wait_until(
                            lambda: all(
                                any(
                                    call in req["response"].get("prism_instruction", "")
                                    for req in replies
                                )
                                for call in updates
                            )
                        )
                    if not finish_during_reply:
                        await asyncio.wait_for(owner.wait_for_idle(), 3)
                    finish.set()
                    finals = {f"lookup-{name}-call_final" for name in names}
                    await wait_until(lambda: finals <= set(output_ids()))
                    await wait_until(
                        lambda: all(
                            any(
                                call in req["response"].get("prism_instruction", "")
                                for req in replies
                            )
                            for call in finals
                        )
                    )
                    await asyncio.wait_for(owner.wait_for_idle(), 3)
                    assert sorted(executions) == names
                    assert len(output_ids()) == (4 if rapid else 6)
                    assert len(set(output_ids())) == len(output_ids())
                    assert all(
                        op["type"] != "conversation.item.delete" for op in peer.history_operations
                    )
                    assert peer.connections == 1
                    if context_mode == "shared" and not rapid:
                        # Correcting result positions must not hide a call reorder.
                        target = owner.current_agent.chat_ctx.copy()
                        indices = [target.index_by_id(f"lookup-{name}-item") for name in names]
                        a, b = indices
                        target.items[a], target.items[b] = target.items[b], target.items[a]
                        before = len(peer.history_operations)
                        with pytest.raises(PrismHistoryUpdateError):
                            await owner.current_agent.update_chat_ctx(target)
                        assert len(peer.history_operations) == before
                finally:
                    responder.cancel()
                    await asyncio.gather(responder, return_exceptions=True)

    @pytest.mark.parametrize("late_message", [False, True])
    @pytest.mark.parametrize("explicit", [False, True])
    async def test_progress_snapshot_preserves_only_later_observations(
        self, late_message, explicit
    ):
        def progress():
            return llm.ChatContext(
                [
                    llm.FunctionCall(
                        call_id="past-update",
                        name="lookup",
                        arguments="{}",
                        extra={"__livekit_agents_tool_non_blocking": True},
                    ),
                    llm.FunctionCallOutput(call_id="past-update", output="Working", is_error=False),
                ]
            )

        async with peer_model(context_mode="strict", playback="discard") as (peer, model, _):
            async with AgentSession(llm=model) as owner:
                agent = Agent(instructions="test", tools=[lookup])
                await owner.start(agent)
                provider = model._session
                if late_message:
                    target = progress()
                item_id = await completed_message(peer, provider)
                await asyncio.wait_for(owner.wait_for_idle(), 3)
                if not late_message:
                    target = progress()
                if explicit:
                    await model._replace_chat_ctx(agent, target)
                else:
                    await provider.update_chat_ctx(target)
                kept = late_message and not explicit
                assert (provider.chat_ctx.get_by_id(item_id) is not None) is kept
                assert any(i["id"] == item_id for i in peer.server_items) is kept
                assert provider.chat_ctx.items[-1].call_id == "past-update"

    async def test_partial_history_failure_allows_a_queued_independent_edit(self):
        async with peer_model(context_mode="strict") as (peer, model, _):
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
                with pytest.raises(PrismHistoryUpdateError, match="partially applied"):
                    await first
                await queued
                assert [i.text_content for i in session.chat_ctx.items] == ["queued"]
                assert [i["text"] for i in peer.server_items] == ["queued"]
                await session.aclose()
                assert len(peer.history_operations) == before + 3
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
            collect_payloads(model._session, r.ItemTruncated, events)
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
            collect_payloads(model._session, r.ItemTruncated, events)
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
            collect_payloads(model._session, r.ItemTruncated, events)
            collect_payloads(model._session, r.ResponseEnded, ended)
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
        async with peer_model(context_mode="strict") as (peer, model, _):
            session = model.session()
            await session.update_instructions("test")
            events = []
            collect_payloads(model._session, r.ItemTruncated, events)
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
        async with peer_model(context_mode="strict") as (peer, model, _):
            session = model.session()
            await session.update_instructions("test")
            events = []
            collect_payloads(model._session, r.ItemTruncated, events)
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
            collect_payloads(model._session, r.ItemTruncated, events)
            item_id = await completed_message(peer, session)
            before = session.chat_ctx
            await acknowledge(peer, "unknown", "Unrelated")
            await acknowledge(peer, item_id, "Other part", content_index=1)
            await wait_until(lambda: len(events) == 2)
            assert session.chat_ctx.items == before.items

    @pytest.mark.parametrize(
        "summary,status",
        [
            (None, "completed"),
            ({"items_left_out": 3}, "cancelled"),
            ({"items_left_out": 3, "first_kept_item_id": "kept"}, "failed"),
        ],
    )
    async def test_response_summary_is_exposed_without_deleting_history(self, summary, status):
        async with peer_model() as (peer, model, _):
            session = model.session()
            await session.update_instructions("test")
            item_id = await completed_message(peer, session)
            before = session.chat_ctx
            events = []
            collect_payloads(model._session, r.ResponseEnded, events)
            response_id = await peer.start()
            usage = {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "context_window": 100,
            }
            await peer.emit(
                "response.done",
                response={
                    "id": response_id,
                    "status": status,
                    "usage": usage,
                    **({} if summary is None else {"prism_truncation": summary}),
                },
            )
            await wait_until(lambda: events)
            assert events[0].response.response_id == response_id
            assert events[0].status == status
            assert events[0].usage == r.ResponseUsage(**usage)
            assert events[0].truncation == (
                None if summary is None else r.ResponseTruncation(**summary)
            )
            assert session.chat_ctx.items == before.items
            # A completed response reference is not required to process a later acknowledgment.
            session._generations.clear()
            truncated = []
            collect_payloads(model._session, r.ItemTruncated, truncated)
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
                event = await asyncio.wait_for(peer.truncations.get(), 3)
                assert event["item_id"] == item_id
                assert event["content_index"] == 0
                assert peer.receipts.empty()
                before = session.history.to_dict()
                events = []
                collect_payloads(model._session, r.ItemTruncated, events)
                await acknowledge(peer, item_id, "All words retained by the server")
                await wait_until(
                    lambda: events and events[-1].kept_text == "All words retained by the server"
                )
                assert model._session.chat_ctx.get_by_id(item_id).text_content == (
                    "All words retained by the server"
                )
                assert session.history.to_dict() == before

    @pytest.mark.parametrize("context_mode", [None, "shared", "strict"])
    async def test_context_mode_controls_omitted_history(self, context_mode):
        options = {} if context_mode is None else {"context_mode": context_mode}
        async with peer_model(**options) as (_, model, _):
            provider = model.session()
            original = messages("first", "second")
            await provider.update_chat_ctx(original)
            reduced = original.copy()
            reduced.items.pop(0)
            await provider.update_chat_ctx(reduced)
            expected = ["second"] if context_mode == "strict" else ["first", "second"]
            assert [item.text for item in await model._get_history()] == expected

            reduced.add_message(role="user", content="third")
            await provider.update_chat_ctx(reduced)
            assert [item.text for item in await model._get_history()] == [*expected, "third"]

    async def test_context_helpers_preserve_bound_tools(self):
        class Assistant(Agent):
            def __init__(self):
                super().__init__(instructions="Help")

            @function_tool
            async def lookup(self) -> str:
                """Look up a value."""
                return "found"

        async with peer_model() as (_, model, _):
            async with AgentSession(llm=model) as session:
                agent = Assistant()
                await session.start(agent)
                config = next(
                    item for item in agent.chat_ctx.items if item.type == "agent_config_update"
                )
                initial = agent.chat_ctx.copy()
                initial.items.extend(messages("first", "second").items)
                await agent.update_chat_ctx(initial)
                await model._replace_chat_ctx(agent, messages("replacement"))
                assert [item.text for item in await model._get_history()] == ["replacement"]
                await model._reconcile_chat_ctx(agent)
                await model._clear_chat_ctx(agent)
                assert not await model._get_history()
                saved = next(item for item in agent.chat_ctx.items if item.id == config.id)
                assert saved.model_dump() == config.model_dump()
                assert saved._tools == config._tools

    @pytest.mark.parametrize(
        "entrypoint,empty_context,nested_task,task_reply,tool_error",
        [
            pytest.param("on_enter", False, False, False, False, id="on-enter"),
            pytest.param("tool", False, False, False, False, id="tool"),
            pytest.param("tool", False, False, False, True, id="tool-error"),
            pytest.param("tool", False, False, True, False, id="speaking-task"),
            pytest.param("tool", False, False, True, True, id="speaking-task-error"),
            pytest.param("tool", True, False, False, False, id="empty-task-context"),
            pytest.param("tool", True, True, False, False, id="nested-empty-task-context"),
        ],
    )
    async def test_task_restores_parent_tools_and_continues(
        self, entrypoint, empty_context, nested_task, task_reply, tool_error, caplog
    ):
        completed, recorded = [], []
        entered = asyncio.Future()
        error_message = "Input incomplete."

        class Child(AgentTask[str]):
            def __init__(self):
                super().__init__(instructions="Child", chat_ctx=llm.ChatContext.empty())

            async def on_enter(self):
                self.complete("finished")

        class Task(AgentTask[str]):
            def __init__(self):
                options = {"chat_ctx": llm.ChatContext.empty()} if empty_context else {}
                super().__init__(instructions="Task", **options)

            async def on_enter(self):
                if task_reply:
                    await self.session.generate_reply(
                        instructions="Acknowledge the detail"
                    ).wait_for_playout()
                self.complete(await Child() if nested_task else "finished")

        @function_tool
        async def collect_detail() -> str:
            """Collect a detail in a sub-task."""
            result = await Task()
            completed.append(result)
            if tool_error:
                raise ToolError(error_message)
            return result

        @function_tool
        async def record_detail(detail: str) -> str:
            """Record the collected detail."""
            recorded.append(detail)
            return "saved"

        class Parent(Agent):
            def __init__(self):
                super().__init__(instructions="Parent", tools=[collect_detail, record_detail])

            async def on_enter(self):
                if entrypoint == "on_enter" and not entered.done():
                    entered.set_result(await Task())

        async with peer_model(playback="discard") as (peer, model, _):
            async with AgentSession(llm=model) as session:
                await session.start(Parent())
                if entrypoint == "on_enter":
                    assert await asyncio.wait_for(entered, 3) == "finished"
                    await wait_until(lambda: peer.settings["instructions"] == "Parent")
                    speech = session.generate_reply()
                    continuation = await peer.next()
                else:
                    speech = session.generate_reply()
                    parent = await peer.start(await peer.next())
                    await peer.emit(
                        "response.function_call_arguments.done",
                        response_id=parent,
                        item_id="parent-item",
                        call_id="parent-call",
                        name="collect_detail",
                        arguments="{}",
                    )
                    await peer.end(parent)
                    if task_reply:
                        child = await peer.start(await peer.next())
                        await peer.message(child, text="Thank you")
                        await peer.end(child)
                    continuation = await peer.next()
                    if task_reply:
                        assert continuation["response"]["prism_call_id"] == "parent-call"
                    else:
                        assert (
                            continuation["response"]["metadata"]["prism_parent_response_id"]
                            == parent
                        )
                    results = [item for item in peer.items if item.get("call_id") == "parent-call"]
                    assert len(results) == 1
                    if tool_error:
                        error = json.loads(results[0]["output"])
                        assert error["status"] == "failed"
                        assert error["message"] == error_message
                        assert "retryable" not in error
                    else:
                        assert results[0]["output"] == "finished"
                    assert completed == ["finished"]

                assert peer.settings["instructions"] == "Parent"
                assert {tool["function"]["name"] for tool in peer.settings["tools"]} == {
                    "collect_detail",
                    "record_detail",
                }
                response = await peer.start(continuation)
                await peer.emit(
                    "response.function_call_arguments.done",
                    response_id=response,
                    item_id="record-item",
                    call_id="record-call",
                    name="record_detail",
                    arguments=json.dumps({"detail": "received"}),
                )
                await peer.end(response)
                response = await peer.start(await peer.next())
                await peer.message(response, text="Recorded")
                await peer.end(response)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert recorded == ["received"]
                results = [item for item in peer.items if item.get("call_id") == "record-call"]
                assert len(results) == 1 and results[0]["output"] == "saved"

                followup = session.generate_reply(user_input="Continue")
                response = await peer.start(await peer.next())
                await peer.message(response, text="Still here")
                await peer.end(response)
                await asyncio.wait_for(followup.wait_for_playout(), 3)
                assert any(
                    item.type == "message" and item.text_content == "Still here"
                    for item in session.history.items
                )
                if entrypoint == "tool":
                    assert completed == ["finished"]
                    assert len([i for i in peer.items if i.get("call_id") == "parent-call"]) == 1
                assert "active tool call" not in caplog.text
                assert peer.connections == 1

    @pytest.mark.parametrize("playback", ["auto", "discard"])
    async def test_response_metrics_measure_first_audio_and_monotonic_duration(
        self, monkeypatch, playback
    ):
        clock = SimpleNamespace(now=10.0, wall=1000.0)
        monkeypatch.setattr(
            realtime_model,
            "time",
            SimpleNamespace(time=lambda: clock.wall, monotonic=lambda: clock.now),
        )
        async with peer_model(playback=playback) as (peer, model, _):
            session = model.session()
            metrics, events = [], []
            session.on("metrics_collected", metrics.append)
            session.on("rime_server_event_received", events.append)
            await session.update_instructions("test")
            response = await peer.start()
            await wait_until(lambda: any(e.payload.kind == "response.started" for e in events))
            await peer.emit(
                "response.output_item.added",
                response_id=response,
                output_index=0,
                item={"id": "message", "type": "message"},
            )
            fields = {
                "response_id": response,
                "output_index": 0,
                "item_id": "message",
                "content_index": 0,
            }
            clock.now = 11.0
            await peer.emit("response.text.delta", **fields, delta="Hello")
            await peer.emit("response.audio.delta", **fields, delta="")
            await wait_until(lambda: any(e.payload.kind == "audio.delta" for e in events))
            clock.now = 12.0
            await peer.emit(
                "response.audio.delta", **fields, delta=base64.b64encode(b"\0\0").decode()
            )
            await wait_until(lambda: sum(e.payload.kind == "audio.delta" for e in events) == 2)
            clock.now = 13.0
            await peer.emit(
                "response.audio.delta", **fields, delta=base64.b64encode(b"\0\0").decode()
            )
            await wait_until(lambda: sum(e.payload.kind == "audio.delta" for e in events) == 3)
            clock.now, clock.wall = 14.0, 900.0
            await finish_metric_response(peer, response)
            await wait_until(lambda: len(metrics) == 1)
            metric = metrics[0]
            assert isinstance(metric, RealtimeModelMetrics)
            assert metric.request_id == response
            assert metric.timestamp == 1000.0
            assert metric.duration == 4.0
            assert metric.ttft == 2.0
            assert (metric.input_tokens, metric.output_tokens, metric.total_tokens) == (12, 8, 20)
            assert metric.tokens_per_second == 2.0
            assert not metric.cancelled
            assert metric.label == model.label
            assert metric.metadata.model_name == "prism"
            assert metric.metadata.model_provider == "rime"
            assert metric.input_token_details == RealtimeModelMetrics.InputTokenDetails()
            assert metric.output_token_details == RealtimeModelMetrics.OutputTokenDetails()
            await finish_metric_response(peer, response)
            # A later response is a processing barrier for the repeated terminal event.
            later = await peer.start()
            await finish_metric_response(peer, later, usage=None)
            await wait_until(lambda: len(metrics) == 2)
            assert [m.request_id for m in metrics] == [response, later]
            assert metrics[1].tokens_per_second == 0

    @pytest.mark.parametrize("status", ["completed", "cancelled", "failed"])
    async def test_no_audio_or_usage_has_defined_metrics(self, status):
        async with peer_model() as (peer, model, _):
            session = model.session()
            metrics = []
            session.on("metrics_collected", metrics.append)
            await session.update_instructions("test")
            response = await peer.start()
            await finish_metric_response(peer, response, status=status, usage=None)
            await wait_until(lambda: len(metrics) == 1)
            metric = metrics[0]
            assert metric.ttft == -1
            assert metric.duration >= 0
            assert metric.cancelled == (status == "cancelled")
            assert (metric.input_tokens, metric.output_tokens, metric.total_tokens) == (0, 0, 0)
            assert metric.tokens_per_second == 0

    async def test_agent_session_forwards_metrics_and_accumulates_usage(self):
        async with peer_model(playback="discard") as (peer, model, _):
            async with AgentSession(llm=model) as session:
                events = []
                session.on("metrics_collected", events.append)
                await session.start(Agent(instructions="test"))
                speech = session.generate_reply(user_input="Hello")
                response = await peer.start(await peer.next())
                await peer.message(response)
                await finish_metric_response(peer, response)
                await wait_until(lambda: len(events) == 1)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert events[0].metrics.request_id == response
                assert events[0].metrics.total_tokens == 20
                usage = session.usage.model_usage[0]
                assert usage.input_tokens == 12
                assert usage.output_tokens == 8

    async def test_interrupted_generation_retains_late_terminal_metrics(self, monkeypatch):
        # Exercise adapter eviction within the older SDK's larger reference window.
        monkeypatch.setattr(realtime_model, "_GENERATION_LIMIT", 3)
        async with peer_model() as (peer, model, _):
            session = model.session()
            metrics, events = [], []
            session.on("metrics_collected", metrics.append)
            session.on("rime_server_event_received", events.append)
            await session.update_instructions("test")
            first = await peer.start()
            await wait_until(lambda: any(e.payload.kind == "response.started" for e in events))
            peer.cancel_release.clear()
            session.interrupt()
            await asyncio.wait_for(peer.cancel_received.wait(), 3)
            assert not metrics
            try:
                responses = []
                for number in range(4):
                    response = await peer.start()
                    responses.append(response)
                    await finish_metric_response(peer, response)
                    await wait_until(lambda number=number: len(metrics) == number + 1)
                await finish_metric_response(peer, first, status="cancelled")
                await wait_until(lambda: len(metrics) == 5)
                assert metrics[-1].request_id == first
                assert metrics[-1].cancelled
                assert metrics[-1].total_tokens == 20
            finally:
                peer.cancel_release.set()
            # Process the repeated cancellation terminal event before checking metrics.
            later = await peer.start()
            await finish_metric_response(peer, later)
            await wait_until(lambda: len(metrics) == 6)
            assert [m.request_id for m in metrics] == [*responses, first, later]

    async def test_generation_limit_refuses_to_evict_unfinished_responses(self, monkeypatch):
        monkeypatch.setattr(realtime_model, "_GENERATION_LIMIT", 3)
        async with peer_model() as (peer, model, _):
            session = model.session()
            errors, metrics = [], []
            session.on("error", errors.append)
            session.on("metrics_collected", metrics.append)
            await session.update_instructions("test")
            for _ in range(4):
                await peer.start()
            await wait_until(lambda: errors)
            with pytest.raises(RimeResourceLimitError, match="generation limit"):
                await session.update_instructions("test")
            assert not metrics

    async def test_terminal_duplicate_after_generation_eviction_is_ignored(self, monkeypatch):
        monkeypatch.setattr(realtime_model, "_GENERATION_LIMIT", 3)
        async with peer_model() as (peer, model, _):
            session = model.session()
            metrics = []
            session.on("metrics_collected", metrics.append)
            await session.update_instructions("test")
            responses = []
            for number in range(4):
                responses.append(await peer.start())
                await finish_metric_response(peer, responses[-1])
                await wait_until(lambda number=number: len(metrics) == number + 1)
            await finish_metric_response(peer, responses[0])
            responses.append(await peer.start())
            await finish_metric_response(peer, responses[-1])
            await wait_until(lambda: len(metrics) == 5)
            assert [metric.request_id for metric in metrics] == responses

    async def test_queued_interruptions_keep_holds_past_cleanup_deadline(self):
        count = 140
        peer = Peer()
        first_truncate = asyncio.Event()
        release_truncate = asyncio.Event()
        emit = peer.emit

        async def delayed_emit(kind, **body):
            if kind == "conversation.item.truncated" and not first_truncate.is_set():
                first_truncate.set()
                await release_truncate.wait()
            await emit(kind, **body)

        peer.emit = delayed_emit
        async with serve(peer.handler, "127.0.0.1", 0) as server:
            endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime"
            async with (
                Rime(api_key="test") as client,
                client.realtime.connect(
                    endpoint=endpoint, timeouts=r.RealtimeTimeouts(request_s=1.0)
                ) as sdk,
            ):
                tasks = []

                def spawn(operation):
                    task = asyncio.create_task(operation)
                    tasks.append(task)
                    return task

                playback = Playback(sdk, "auto", 1.0, sdk.report_playback, spawn, lambda: True)
                outputs = {}
                tokens = {}
                ended = set()

                async def consume():
                    async for event in sdk.events:
                        payload = event.payload
                        if isinstance(payload, r.ResponseStarted):
                            if payload.response.response_id in {"first", "second"}:
                                playback.start(payload.response)
                        elif isinstance(payload, r.MessageStarted):
                            outputs[payload.output.response.response_id] = payload.output
                            tokens[payload.output.item_id] = playback.output(payload.output)
                        elif isinstance(payload, r.AudioDelta):
                            frame = rtc.AudioFrame(
                                payload.audio.data, 24000, 1, len(payload.audio.data) // 2
                            )
                            playback.tag(tokens[payload.output.item_id], frame)
                        elif isinstance(payload, r.ResponseEnded):
                            playback.end(payload.response)
                            ended.add(payload.response.response_id)

                reader = asyncio.create_task(consume())
                try:
                    for response_id in ("first", "second"):
                        await peer.start(response_id=response_id)
                        await peer.message(response_id)
                        await peer.end(response_id)
                    await wait_until(lambda: "second" in ended)
                    # Age both responses, then block the first report while the second
                    # waits in the playback queue. Their cleanup deadlines now expire
                    # before either acknowledgment, but before the SDK request timeout.
                    await asyncio.sleep(0.8)
                    playback.interrupt(
                        r.PlaybackInterrupted(output=outputs["first"], audio_end_ms=10)
                    )
                    await asyncio.wait_for(first_truncate.wait(), 1)
                    playback.interrupt(
                        r.PlaybackInterrupted(output=outputs["second"], audio_end_ms=10)
                    )
                    await asyncio.sleep(0.25)
                    for number in range(count):
                        response_id = f"ordinary-{number}"
                        await peer.start(response_id=response_id)
                        await peer.end(response_id)
                        await wait_until(lambda response_id=response_id: response_id in ended)
                    release_truncate.set()
                    await asyncio.wait_for(playback.settle(), 1)
                    reported = [await asyncio.wait_for(peer.truncations.get(), 1) for _ in range(2)]
                    assert [event["item_id"] for event in reported] == [
                        outputs["first"].item_id,
                        outputs["second"].item_id,
                    ]
                    assert peer.receipts.empty()
                finally:
                    release_truncate.set()
                    playback.close()
                    reader.cancel()
                    await asyncio.gather(reader, *tasks, return_exceptions=True)

    @pytest.mark.parametrize("application_segments", [129, 257])
    async def test_segment_overflow_recovers_only_after_old_callbacks_drain(
        self, application_segments
    ):

        async with peer_model() as (peer, model, _):
            sink = ControlledOutput()
            wrapper = model._wrap_audio_output(sink)
            forwarded = []
            wrapper.on("playback_finished", forwarded.append)
            provider = model.session()
            await provider.update_instructions("test")
            for _ in range(application_segments):
                await wrapper.capture_frame(rtc.AudioFrame(b"\0\0" * 480, 24000, 1, 480))
                wrapper.flush()

            async def capture_rime_response():
                pending = provider.generate_reply()
                response = await peer.start(await peer.next())
                generation = await pending
                await peer.message(response)
                await peer.end(response)
                message = await anext(generation.message_stream)
                async for frame in message.audio_stream:
                    await wrapper.capture_frame(frame)
                wrapper.flush()
                return response

            await capture_rime_response()
            # Old callbacks must consume the dropped count, not the newly queued
            # Rime output. A response captured during ambiguity keeps estimates.
            for _ in range(application_segments):
                sink.finish()
            await asyncio.sleep(0.01)
            assert peer.receipts.empty()
            sink.finish()
            await asyncio.sleep(0.01)
            assert peer.receipts.empty()

            # All earlier callbacks have drained. A fresh identified segment can
            # now prove completion, but only when its own callback arrives.
            fresh = await capture_rime_response()
            await asyncio.sleep(0.01)
            assert peer.receipts.empty()
            sink.finish()
            receipt = await asyncio.wait_for(peer.receipts.get(), 3)
            assert receipt["response_id"] == fresh
            assert receipt["played_ms"] == pytest.approx(20)
            assert len(forwarded) == application_segments + 2
            assert peer.receipts.empty()

    @pytest.mark.parametrize("sink,tts", [(False, False), (True, False), (True, True)])
    async def test_discard_keeps_text_without_audio_or_automatic_tts(self, sink, tts):
        async with peer_model(playback="discard") as (peer, model, _):
            speech_tts = CountingTTS()
            async with AgentSession(llm=model, tts=speech_tts if tts else None) as session:
                output = ControlledOutput(synchronous=True)
                if sink:
                    session.output.audio = output
                await session.start(Agent(instructions="test"))
                speech = session.generate_reply(user_input="hello")
                response = await peer.start(await peer.next())
                await peer.message(response, text="Visible answer", seconds=3)
                assert peer.receipts.empty()
                await peer.end(response)
                receipt = await asyncio.wait_for(peer.receipts.get(), 3)
                assert receipt["response_id"] == response
                assert receipt["played_ms"] == 0
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert not output.frames
                assert speech_tts.calls == 0
                assert any(
                    getattr(i, "text_content", None) == "Visible answer"
                    for i in session.history.items
                )
                await peer.end(response)
                await asyncio.sleep(0.01)
                assert peer.receipts.empty()

    @pytest.mark.parametrize("sample_rate", [None, 48000])
    @pytest.mark.parametrize("synchronous", [False, True])
    async def test_measured_playback_through_real_pipeline(self, sample_rate, synchronous):
        async with peer_model() as (peer, model, _):
            async with AgentSession(llm=model) as session:
                sink = ControlledOutput(sample_rate=sample_rate, synchronous=synchronous)
                wrapper = model._wrap_audio_output(sink)
                forwarded = []
                wrapper.on("playback_finished", forwarded.append)
                session.output.audio = wrapper
                assert wrapper.sample_rate is None
                wrapper.pause()
                assert sink.paused
                wrapper.resume()
                await session.start(Agent(instructions="test"))
                speech = session.generate_reply()
                response = await peer.start(await peer.next())
                await emit_message(peer, response, "first")
                await emit_message(peer, response, "second")
                await peer.end(response)
                if not synchronous:
                    await wait_until(lambda: bool(sink.segments))
                    assert peer.receipts.empty()
                    sink.finish()
                    await wait_until(lambda: bool(sink.segments))
                    assert peer.receipts.empty()
                    sink.finish()
                receipt = await asyncio.wait_for(peer.receipts.get(), 3)
                assert receipt["response_id"] == response
                assert receipt["played_ms"] == pytest.approx(40)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert len(forwarded) == 2
                if sample_rate:
                    assert {frame.sample_rate for frame in sink.frames} == {sample_rate}
                assert peer.receipts.empty()

    @pytest.mark.parametrize("mode", ["estimated", "auto"])
    async def test_unwrapped_output_keeps_estimate(self, mode):
        async with peer_model(playback=mode) as (peer, model, _):
            async with AgentSession(llm=model) as session:
                session.output.audio = ControlledOutput(synchronous=True)
                await session.start(Agent(instructions="test"))
                speech = session.generate_reply()
                response = await peer.start(await peer.next())
                await peer.message(response)
                await peer.end(response)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert peer.receipts.empty()

    @pytest.mark.parametrize("failure_at", [1, 2, 3, 4])
    async def test_exact_edit_failure_repairs_local_projection_and_keeps_connection(
        self, failure_at
    ):
        async with peer_model() as (peer, model, _):
            async with AgentSession(llm=model) as session:
                agent = Agent(instructions="test")
                await session.start(agent)
                await model._replace_chat_ctx(agent, messages("old-a", "old-b"))
                peer.fail_history_at = len(peer.history_operations) + failure_at
                with pytest.raises(PrismHistoryUpdateError) as caught:
                    await model._replace_chat_ctx(agent, messages("new-a", "new-b"))
                error = caught.value
                assert error.status == ("rejected" if failure_at == 1 else "partial")
                assert error.recoverable and not error.needs_reconciliation
                assert [
                    i.text_content for i in agent.chat_ctx.items if isinstance(i, llm.ChatMessage)
                ] == [i["text"] for i in peer.server_items]
                assert "old-a" not in str(error)
                assert peer.connections == 1
                speech = session.generate_reply()
                response = await peer.start(await peer.next())
                await peer.end(response)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                await model._clear_chat_ctx(agent)
                assert not peer.server_items

    @pytest.mark.parametrize("context_mode", ["strict", "shared"])
    async def test_ordinary_partial_allows_independent_reply_and_update(self, context_mode, caplog):
        async with peer_model(context_mode=context_mode) as (
            peer,
            model,
            _,
        ):
            async with AgentSession(llm=model) as session:
                agent = Agent(instructions="test")
                await session.start(agent)
                errors = []
                model._session.on("error", errors.append)
                original = messages("old")
                await agent.update_chat_ctx(original)
                target = original.copy()
                target.add_message(role="user", content="accepted")
                target.add_message(role="user", content="rejected")
                target.add_message(role="user", content="not sent yet")
                peer.fail_history_at = len(peer.history_operations) + 2
                with pytest.raises(PrismHistoryUpdateError) as caught:
                    await agent.update_chat_ctx(target)
                assert caught.value.status == "partial"
                assert caught.value.needs_reconciliation
                assert "Prism history update failed" in caplog.text
                assert [i.text_content for i in caught.value.confirmed_chat_ctx.items] == [
                    "old",
                    "accepted",
                ]
                assert [i.text_content for i in model._session.chat_ctx.items] == [
                    "old",
                    "accepted",
                ]
                before = len(peer.history_operations)
                assert before == 3
                assert not any(not error.recoverable for error in errors)
                speech = session.generate_reply()
                response = await peer.start(await peer.next())
                await peer.end(response)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert len(peer.history_operations) == before
                # The caller can retry the requested view. Accepted items are not replayed.
                await agent.update_chat_ctx(target)
                assert len(peer.history_operations) == before + 2
                assert [i["text"] for i in peer.server_items] == [
                    "old",
                    "accepted",
                    "rejected",
                    "not sent yet",
                ]

    @pytest.mark.parametrize("context_mode", ["strict", "shared"])
    async def test_invalid_ordinary_update_allows_reply_and_valid_update(self, context_mode):
        async with peer_model(context_mode=context_mode) as (peer, model, _):
            async with AgentSession(llm=model) as session:
                agent = Agent(instructions="test")
                await session.start(agent)
                await agent.update_chat_ctx(messages("keep"))
                before = len(peer.history_operations)
                with pytest.raises(PrismHistoryUpdateError) as caught:
                    await agent.update_chat_ctx(messages("x" * 4001))
                assert caught.value.recoverable
                assert caught.value.code == "history_update_rejected"
                assert len(peer.history_operations) == before
                speech = session.generate_reply()
                response = await peer.start(await peer.next())
                await peer.end(response)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                await agent.update_chat_ctx(messages("valid"))
                assert peer.server_items[-1]["text"] == "valid"
                assert peer.connections == 1

    async def test_shared_views_preserve_omissions_and_explicit_clear_is_exact(self):
        async with peer_model(context_mode="shared") as (peer, model, _):
            async with AgentSession(llm=model) as session:
                agent = Agent(instructions="test")
                await session.start(agent)
                original = messages("one", "two")
                await agent.update_chat_ctx(original)
                count = len(peer.history_operations)
                await agent.update_chat_ctx(llm.ChatContext.empty())
                await agent.update_chat_ctx(llm.ChatContext([original.items[-1]]))
                assert len(peer.history_operations) == count
                assert [i["text"] for i in peer.server_items] == ["one", "two"]
                target = original.copy()
                target.items[0].content = ["one!"]
                with pytest.raises(PrismHistoryUpdateError):
                    await agent.update_chat_ctx(target)
                await model._reconcile_chat_ctx(agent)
                await model._replace_chat_ctx(agent, target)
                assert [i["text"] for i in peer.server_items] == ["one!", "two"]
                await model._clear_chat_ctx(agent)
                assert not peer.server_items

    @pytest.mark.parametrize("operation", ["replace", "clear", "reconcile"])
    async def test_inactive_agent_cannot_claim_helper_success(self, operation):
        async with peer_model() as (_, model, _):
            agent = Agent(instructions="inactive", llm=model)
            with pytest.raises(PrismHistoryUpdateError, match="active"):
                if operation == "replace":
                    await model._replace_chat_ctx(agent, messages("new"))
                elif operation == "clear":
                    await model._clear_chat_ctx(agent)
                else:
                    await model._reconcile_chat_ctx(agent)

    @pytest.mark.parametrize("failure", ["cancel", "error"])
    async def test_failing_task_inside_tool_submits_one_result(self, failure, caplog):
        class Task(AgentTask[str]):
            def __init__(self):
                super().__init__(instructions="Task")

            async def on_enter(self):
                if failure == "cancel":
                    self.cancel()
                else:
                    self.complete(ValueError("task failed"))

        @function_tool
        async def task_tool() -> str:
            """Run a task that cannot finish."""
            return await Task()

        async with peer_model(context_mode="shared", playback="discard") as (peer, model, _):
            async with AgentSession(llm=model) as session:
                await session.start(Agent(instructions="Parent", tools=[task_tool]))
                speech = session.generate_reply()
                parent = await peer.start(await peer.next())
                await peer.emit(
                    "response.function_call_arguments.done",
                    response_id=parent,
                    item_id="parent-item",
                    call_id="parent-call",
                    name="task_tool",
                    arguments="{}",
                )
                await peer.end(parent)
                request = await peer.next()
                response = await peer.start(request)
                await peer.end(response)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert len([i for i in peer.items if i.get("call_id") == "parent-call"]) == 1
                assert "active tool call" not in caplog.text

    async def test_long_task_receives_two_turns_and_child_tool_before_parent_result(self, caplog):
        ran = []
        entered = asyncio.Event()

        class Task(AgentTask[str]):
            def __init__(self):
                super().__init__(instructions="Task", chat_ctx=llm.ChatContext.empty())

            async def on_enter(self):
                entered.set()

            @function_tool
            async def child(self) -> str:
                """Read a detail."""
                ran.append("child")
                return "detail"

            @function_tool
            async def finish(self) -> None:
                """Finish the task."""
                self.complete("finished")

        @function_tool
        async def parent_tool() -> str:
            """Run the task."""
            ran.append("parent")
            return await Task()

        async with peer_model(context_mode="shared", playback="discard") as (peer, model, _):
            async with AgentSession(llm=model) as session:
                await session.start(Agent(instructions="Parent", tools=[parent_tool]))
                speech = session.generate_reply()
                parent = await peer.start(await peer.next())
                await peer.emit(
                    "response.function_call_arguments.done",
                    response_id=parent,
                    item_id="parent-item",
                    call_id="parent-call",
                    name="parent_tool",
                    arguments="{}",
                )
                await peer.end(parent)
                await asyncio.wait_for(entered.wait(), 3)
                for number, tool in enumerate(("child", "finish")):
                    item = f"user-{number}"
                    await peer.emit(
                        "input_audio_buffer.speech_started", item_id=item, audio_start_ms=0
                    )
                    await peer.emit(
                        "input_audio_buffer.speech_stopped", item_id=item, audio_end_ms=20
                    )
                    await peer.emit("input_audio_buffer.committed", item_id=item)
                    await peer.emit(
                        "conversation.item.input_audio_transcription.completed",
                        item_id=item,
                        transcript=f"turn {number}",
                    )
                    response = await peer.start()
                    await peer.emit(
                        "response.function_call_arguments.done",
                        response_id=response,
                        item_id=f"{tool}-item",
                        call_id=f"{tool}-call",
                        name=tool,
                        arguments="{}",
                    )
                    await peer.end(response)
                    if tool == "child":
                        request = await peer.next()
                        reply = await peer.start(request)
                        await peer.message(reply)
                        await peer.end(reply)
                        await wait_until(
                            lambda: any(
                                getattr(i, "text_content", None) == "Hello"
                                for i in session.history.items
                            )
                        )
                await wait_until(lambda: any(i.get("call_id") == "parent-call" for i in peer.items))
                request = await peer.next()
                assert request["response"]["prism_call_id"] == "parent-call"
                reply = await peer.start(request)
                await peer.end(reply)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert ran == ["parent", "child"]
                assert len([i for i in peer.items if i.get("call_id") == "parent-call"]) == 1
                assert "active tool call" not in caplog.text

    async def test_observed_interruption_variant_appends_without_rebuild(self):
        async with peer_model(context_mode="shared") as (peer, model, _):
            provider = model.session()
            await provider.update_instructions("test")
            response = await peer.start()
            item_id = await peer.message(response, text="Full answer.")
            await peer.end(response)
            await wait_until(lambda: provider.chat_ctx.get_by_id(item_id) is not None)
            provider.truncate(
                message_id=item_id,
                modalities=["text", "audio"],
                audio_end_ms=10,
                audio_transcript="Full",
            )
            await asyncio.wait_for(peer.truncations.get(), 3)
            await peer.emit(
                "conversation.item.truncated",
                item_id=item_id,
                content_index=0,
                audio_end_ms=10,
                prism_kept_text="Full ",
            )
            await wait_until(lambda: provider.chat_ctx.get_by_id(item_id).text_content == "Full ")
            observed = provider.chat_ctx
            observed.get_by_id(item_id).content = ["Full"]
            observed.add_message(role="user", content="Next")
            before = len(peer.history_operations)
            await provider.update_chat_ctx(observed)
            assert [op["type"] for op in peer.history_operations[before:]] == [
                "conversation.item.create"
            ]
            assert provider.chat_ctx.get_by_id(item_id).text_content == "Full "

    async def test_strict_tool_result_rejects_concurrent_playback_history_change(self):
        async with peer_model(context_mode="strict") as (peer, model, _):
            provider = model.session()
            desired, item_id = await pending_note_snapshot(peer, provider)
            before = len(peer.history_operations)
            with pytest.raises(
                PrismHistoryUpdateError,
                match="Finish active replies and tool calls before replacing history",
            ):
                await provider.update_chat_ctx(desired)
            assert len(peer.history_operations) == before
            assert provider.chat_ctx.get_by_id(item_id).text_content == "Got it"
            # The original call is still live. A fresh, exact snapshot can settle it.
            fresh = provider.chat_ctx.copy()
            fresh.items.append(desired.items[-1])
            await provider.update_chat_ctx(fresh)
            assert len(peer.history_operations) == before + 1
            assert not peer.imported_calls

    async def test_shared_pending_tool_result_rejects_unknown_text_change(self):
        async with peer_model(context_mode="shared") as (peer, model, _):
            provider = model.session()
            desired, item_id = await pending_note_snapshot(peer, provider)
            desired.get_by_id(item_id).content = ["An unrelated intentional edit."]
            before = len(peer.history_operations)
            with pytest.raises(PrismHistoryUpdateError, match="conflicts with confirmed content"):
                await provider.update_chat_ctx(desired)
            assert len(peer.history_operations) == before
            assert provider.chat_ctx.get_by_id(item_id).text_content == "Got it"

    async def test_agent_session_interrupted_tool_sync_survives_playback_history_change(
        self, monkeypatch, caplog
    ):
        started = asyncio.Event()
        finish = asyncio.Event()
        copied = asyncio.Event()
        apply = asyncio.Event()
        executions = []
        snapshots = []

        @function_tool
        async def save_note() -> str:
            """Save the caller's note."""
            executions.append("save")
            started.set()
            try:
                await finish.wait()
            except asyncio.CancelledError:
                # The external write finishes despite speech interruption.
                await finish.wait()
            return "The note was saved."

        async with peer_model(context_mode="shared") as (peer, model, _):
            async with AgentSession(llm=model) as session:
                sink = ControlledOutput()
                session.output.audio = sink
                agent = Agent(instructions="test", tools=[save_note])
                await session.start(agent)
                provider = model._session
                errors = []
                provider.on("error", errors.append)
                real_update = provider.update_chat_ctx

                async def schedule_update(chat_ctx):
                    if any(isinstance(i, llm.FunctionCallOutput) for i in chat_ctx.items):
                        snapshots.append(chat_ctx.copy())
                        copied.set()
                        await apply.wait()
                    # Only schedule the race at the public boundary. All history
                    # decisions and SDK writes use the real plugin implementation.
                    await real_update(chat_ctx)

                monkeypatch.setattr(provider, "update_chat_ctx", schedule_update)
                speech = session.generate_reply(user_input="Next week.")
                response = await peer.start(await peer.next())
                item_id = await emit_note_call(peer, response)
                await end_note_response(peer, provider, response)
                await asyncio.wait_for(started.wait(), 3)
                await wait_until(lambda: bool(sink.segments))
                before = len(peer.history_operations)
                speech.interrupt()
                finish.set()
                try:
                    await asyncio.wait_for(copied.wait(), 3)
                    assert snapshots[0].get_by_id(item_id).text_content == (
                        "Got it, I'm making a note of that."
                    )
                    assert snapshots[0].items[-1].output == "The note was saved."
                    assert snapshots[0].items[-1].reply_required is False
                    await retain_note_prefix(peer, provider, item_id)
                finally:
                    apply.set()
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                await wait_until(lambda: any(i.get("call_id") == "note-call" for i in peer.items))
                assert executions == ["save"]
                assert speech.interrupted
                assert provider.chat_ctx.get_by_id(item_id).text_content == "Got it"
                operations = peer.history_operations[before:]
                assert len(operations) == 1
                assert operations[0]["type"] == "conversation.item.create"
                assert operations[0]["item"]["type"] == "function_call_output"
                assert operations[0]["item"]["call_id"] == "note-call"
                assert operations[0]["item"]["output"] == "The note was saved."
                assert not peer.imported_calls
                await provider.update_chat_ctx(snapshots[0])
                await provider.update_chat_ctx(provider.chat_ctx.copy())
                assert len(peer.history_operations) == before + 1
                assert provider.chat_ctx.get_by_id(item_id).text_content == "Got it"
                assert next(i for i in peer.server_items if i["id"] == item_id)["text"] == "Got it"
                sink.synchronous = True
                next_speech = session.generate_reply(user_input="Thank you.")
                next_response = await peer.start(await peer.next())
                await peer.message(next_response, text="You're welcome.")
                await peer.end(next_response)
                await asyncio.wait_for(next_speech.wait_for_playout(), 3)
                assert executions == ["save"]
                assert len([i for i in peer.items if i.get("call_id") == "note-call"]) == 1
                assert peer.connections == 1
                assert not errors
                assert (
                    "failed to sync the tool results of an interrupted generation"
                    not in caplog.text
                )
                assert "Prism history update failed" not in caplog.text
                assert (
                    "Finish active replies and tool calls before replacing history"
                    not in caplog.text
                )

    @pytest.mark.parametrize(
        "mutation", ["missing", "duplicate", "unconsumed", "unrelated", "detach"]
    )
    async def test_unproven_playback_never_reports_finished(self, mutation):
        async with peer_model(timeouts=r.RealtimeTimeouts(request_s=0.05)) as (peer, model, _):
            provider = model.session()
            sink = ControlledOutput(synchronous=True)
            wrapper = model._wrap_audio_output(sink)
            await provider.update_instructions("test")
            pending = provider.generate_reply()
            response = await peer.start(await peer.next())
            generation = await pending
            await peer.message(response)
            await peer.end(response)
            message = await anext(generation.message_stream)
            frames = [frame async for frame in message.audio_stream]
            if mutation == "missing":
                for frame in frames:
                    frame.userdata.clear()
            if mutation == "duplicate":
                frames *= 2
            if mutation == "unrelated":
                frames = [rtc.AudioFrame(b"\0\0" * 480, 24000, 1, 480)]
            if mutation != "unconsumed":
                for frame in frames:
                    await wrapper.capture_frame(frame)
                if mutation == "detach":
                    wrapper.on_detached()
                wrapper.flush()
                await wrapper.wait_for_playout()
            await asyncio.sleep(0.08)
            assert peer.receipts.empty()
            assert peer.connections == 1

    async def test_cancelled_explicit_edit_settles_before_cancellation(self):
        async with peer_model() as (peer, model, _):
            async with AgentSession(llm=model) as session:
                agent = Agent(instructions="test")
                await session.start(agent)
                peer.history_release.clear()
                edit = asyncio.create_task(model._replace_chat_ctx(agent, messages("accepted")))
                await asyncio.wait_for(peer.history_started.wait(), 3)
                edit.cancel()
                await asyncio.sleep(0)
                assert not edit.done()
                peer.history_release.set()
                with pytest.raises(asyncio.CancelledError):
                    await edit
                assert peer.server_items[0]["text"] == "accepted"
                await model._reconcile_chat_ctx(agent)
                assert agent.chat_ctx.items[0].text_content == "accepted"

    @pytest.mark.parametrize("context_mode", ["strict", "shared"])
    async def test_cancelled_refused_update_settles_and_allows_independent_reply(
        self, context_mode
    ):
        async with peer_model(context_mode=context_mode) as (peer, model, _):
            async with AgentSession(llm=model) as session:
                agent = Agent(instructions="test")
                await session.start(agent)
                peer.history_release.clear()
                peer.fail_history_at = 2
                edit = asyncio.create_task(agent.update_chat_ctx(messages("accepted", "refused")))
                await asyncio.wait_for(peer.history_started.wait(), 3)
                edit.cancel()
                await asyncio.sleep(0)
                assert not edit.done()
                peer.history_release.set()
                with pytest.raises(PrismHistoryUpdateError) as caught:
                    await edit
                assert caught.value.status == "partial"
                assert [i["text"] for i in peer.server_items] == ["accepted"]
                speech = session.generate_reply()
                response = await peer.start(await peer.next())
                await peer.end(response)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert len(peer.history_operations) == 2
                await agent.update_chat_ctx(caught.value.confirmed_chat_ctx)
                assert len(peer.history_operations) == 2

    async def test_reconciliation_rejects_revision_race(self):
        class DelayedAgent(Agent):
            delayed = False
            entered = asyncio.Event()
            release = asyncio.Event()

            async def update_chat_ctx(self, chat_ctx, **kwargs):
                if self.delayed:
                    self.entered.set()
                    await self.release.wait()
                await super().update_chat_ctx(chat_ctx, **kwargs)

        async with peer_model(context_mode="shared") as (peer, model, _):
            async with AgentSession(llm=model) as session:
                agent = DelayedAgent(instructions="test")
                await session.start(agent)
                await model._replace_chat_ctx(agent, messages("one"))
                agent.delayed = True
                edit = asyncio.create_task(model._reconcile_chat_ctx(agent))
                await asyncio.wait_for(agent.entered.wait(), 3)
                await peer.emit(
                    "conversation.item.input_audio_transcription.completed",
                    item_id="new-input",
                    transcript="new",
                )
                await wait_until(
                    lambda: any(
                        getattr(i, "text_content", None) == "new" for i in session.history.items
                    )
                )
                agent.release.set()
                with pytest.raises(PrismHistoryUpdateError, match="changed"):
                    await edit
                agent.delayed = False
                await model._reconcile_chat_ctx(agent)
                assert any(getattr(i, "text_content", None) == "new" for i in agent.chat_ctx.items)

    async def test_pending_parent_result_survives_installed_sdk_reference_lifetime(self):
        count = 140
        async with peer_model(playback="discard", context_mode="shared") as (peer, model, _):
            provider = model.session()
            await provider.update_instructions("test")
            parent = await peer.start()
            await peer.emit(
                "response.function_call_arguments.done",
                response_id=parent,
                item_id="parent",
                call_id="parent-call",
                name="lookup",
                arguments="{}",
            )
            await peer.end(parent)
            await asyncio.wait_for(peer.receipts.get(), 3)
            for _ in range(count):
                response = await peer.start()
                await peer.end(response)
                await asyncio.wait_for(peer.receipts.get(), 3)
            target = llm.ChatContext(
                [llm.FunctionCallOutput(call_id="parent-call", output="done", is_error=False)]
            )
            await provider.update_chat_ctx(target)
            reply = provider.generate_reply()
            request = await peer.next()
            assert request["response"]["prism_call_id"] == "parent-call"
            response = await peer.start(request)
            await peer.end(response)
            await reply
            assert len([i for i in peer.items if i.get("call_id") == "parent-call"]) == 1

    @pytest.mark.parametrize("acknowledgment", ["missing", "malformed"])
    async def test_unknown_acknowledgment_is_terminal_without_replay(
        self, acknowledgment, monkeypatch
    ):
        async with peer_model(timeouts=r.RealtimeTimeouts(request_s=0.05)) as (peer, model, _):
            async with AgentSession(llm=model) as session:
                agent = Agent(instructions="test")
                await session.start(agent)
                if acknowledgment == "missing":
                    peer.history_release.clear()
                else:
                    emit = peer.emit

                    async def malformed_emit(kind, **fields):
                        if kind == "conversation.item.created":
                            fields["item"] = {**fields["item"], "id": ""}
                        await emit(kind, **fields)

                    monkeypatch.setattr(peer, "emit", malformed_emit)
                try:
                    with pytest.raises(PrismHistoryUpdateError) as caught:
                        await model._replace_chat_ctx(agent, messages("unknown outcome"))
                    assert caught.value.status == "uncertain"
                    assert not caught.value.recoverable
                    assert caught.value.confirmed_chat_ctx is None
                    assert len(peer.history_operations) == 1
                    with pytest.raises((llm.RealtimeError, RimeTimeoutError, RimeStreamError)):
                        await model._reconcile_chat_ctx(agent)
                    assert len(peer.history_operations) == 1
                    assert peer.requests.empty()
                finally:
                    peer.history_release.set()

    async def test_headless_empty_response_and_wrong_model_helper(self):
        async with peer_model(playback="discard") as (peer, model, _):
            async with AgentSession(llm=model) as session:
                agent = Agent(instructions="test")
                await session.start(agent)
                speech = session.generate_reply()
                response = await peer.start(await peer.next())
                await peer.end(response)
                receipt = await asyncio.wait_for(peer.receipts.get(), 3)
                assert receipt["played_ms"] == 0
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                async with peer_model() as (_, other, _):
                    with pytest.raises(PrismHistoryUpdateError, match="directly"):
                        await other._clear_chat_ctx(agent)

    async def test_roomio_uses_estimates_without_a_wrapper(self):
        """Real RoomIO and RTC audio source; only room publication/subscription is local."""
        from types import SimpleNamespace

        from livekit.agents.voice.room_io import RoomIO, RoomOptions

        class Room(rtc.EventEmitter):
            name = "local-validation"

            def __init__(self):
                super().__init__()
                self.local_participant = SimpleNamespace(
                    identity="agent",
                    publish_track=self.publish_track,
                    set_attributes=self.set_attributes,
                )
                self.remote_participants = {
                    "user": SimpleNamespace(
                        identity="user",
                        sid="PA_user",
                        name="user",
                        metadata="",
                        attributes={},
                        kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
                    )
                }

            def isconnected(self):
                return True

            async def publish_track(self, track, options):
                return SimpleNamespace(sid="TR_audio", wait_for_subscription=self.subscribed)

            async def subscribed(self):
                pass

            async def set_attributes(self, attributes):
                pass

        async with peer_model() as (peer, model, _):
            async with AgentSession(llm=model) as session:
                room_io = RoomIO(
                    session,
                    Room(),
                    options=RoomOptions(audio_input=False, text_input=False, text_output=False),
                )
                await room_io.start()
                try:

                    class OpeningAgent(Agent):
                        async def on_enter(self):
                            await self.session.generate_reply(instructions="Say hello")

                    await session.start(OpeningAgent(instructions="test"))
                    response = await peer.start(await peer.next())
                    await peer.message(response, seconds=0.02)
                    await peer.end(response)
                    await wait_until(lambda: session.agent_state == "listening")
                    assert peer.receipts.empty()
                    assert not hasattr(model, "wrap_audio_output")
                finally:
                    await room_io.aclose()

    @pytest.mark.parametrize(
        "operation,acknowledgment,ended",
        [pytest.param("helper", index, False, id=f"helper-ack-{index}") for index in range(1, 5)]
        + [
            pytest.param("direct", 4, False, id="direct-final-ack-active"),
            pytest.param("direct", 4, True, id="direct-final-ack-ended"),
        ],
    )
    async def test_rebuild_race_allows_append_and_completed_reply(
        self, operation, acknowledgment, ended
    ):
        async with peer_model(context_mode="strict", playback="discard") as (peer, model, _):
            async with AgentSession(llm=model) as session:
                agent = Agent(instructions="test")
                await session.start(agent)
                provider = model._session
                errors, events = [], []
                provider.on("error", errors.append)
                provider.on("rime_server_event_received", events.append)
                await model._replace_chat_ctx(agent, messages("old-a", "old-b"))
                before = len(peer.history_operations)
                peer.block_history_at = before + acknowledgment
                peer.history_release.clear()
                desired = messages("new-a", "new-b")
                edit = asyncio.create_task(
                    model._replace_chat_ctx(agent, desired)
                    if operation == "helper"
                    else provider.update_chat_ctx(desired)
                )
                try:
                    await wait_until(lambda: len(peer.history_operations) == peer.block_history_at)
                    response = await peer.start()
                    await wait_until(
                        lambda: any(e.payload.kind == "response.started" for e in events)
                    )
                    if ended:
                        await peer.end(response)
                        await wait_until(
                            lambda: any(e.payload.kind == "response.ended" for e in events)
                        )
                finally:
                    peer.history_release.set()
                with pytest.raises(PrismHistoryUpdateError) as caught:
                    await edit
                assert caught.value.status == "partial" and caught.value.recoverable
                assert caught.value.needs_reconciliation is (operation == "direct")
                assert len(peer.history_operations) == before + acknowledgment
                expected = [["old-b"], [], ["new-a"], ["new-a", "new-b"]][acknowledgment - 1]
                assert [item.text for item in await model._get_history()] == expected
                if not ended:
                    await peer.end(response)
                    await wait_until(
                        lambda: any(e.payload.kind == "response.ended" for e in events)
                    )
                if operation == "direct":
                    await model._reconcile_chat_ctx(agent)
                assert [
                    i.text_content for i in agent.chat_ctx.items if isinstance(i, llm.ChatMessage)
                ] == [i["text"] for i in peer.server_items]

                accepted = list(peer.history_operations)
                retained_ids = [item["id"] for item in peer.server_items]
                followup = agent.chat_ctx.copy()
                followup.add_message(role="user", content="after the race")
                await agent.update_chat_ctx(followup)
                assert peer.history_operations[:-1] == accepted
                assert len(peer.history_operations) == len(accepted) + 1
                assert [item["id"] for item in peer.server_items[:-1]] == retained_ids
                assert peer.server_items[-1]["text"] == "after the race"
                speech = session.generate_reply()
                response = await peer.start(await peer.next())
                await peer.message(response, text="Recovered")
                await peer.end(response)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert any(
                    getattr(item, "text_content", None) == "Recovered"
                    for item in session.history.items
                )
                assert not errors
                assert peer.connections == 1

    @pytest.mark.parametrize("context_mode", ["strict", "shared"])
    @pytest.mark.parametrize("failure", ["invalid", "rejected"])
    async def test_refused_typed_input_cannot_start_reply_and_allows_recovery(
        self, context_mode, failure, caplog
    ):
        async with peer_model(context_mode=context_mode) as (
            peer,
            model,
            _,
        ):
            async with AgentSession(llm=model) as session:
                agent = Agent(instructions="test")
                await session.start(agent)
                if failure == "rejected":
                    peer.fail_history_at = 1
                speech = session.generate_reply(
                    user_input="refused input" if failure == "rejected" else "x" * 4001
                )
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert peer.requests.empty() and not peer.server_items
                assert "Prism history update failed" in caplog.text
                assert len(peer.history_operations) == (failure == "rejected")
                assert not agent.chat_ctx.copy(
                    exclude_config_update=True, exclude_handoff=True
                ).items
                for user_input in (None, "independent input"):
                    speech = (
                        session.generate_reply()
                        if user_input is None
                        else session.generate_reply(user_input=user_input)
                    )
                    response = await peer.start(await peer.next())
                    await peer.end(response)
                    await asyncio.wait_for(speech.wait_for_playout(), 3)
                assert [i["text"] for i in peer.server_items] == ["independent input"]
                assert peer.connections == 1

    async def test_late_playback_cannot_release_fresh_provider_session(self):
        async with peer_model() as (peer, model, _):
            sink = ControlledOutput()
            wrapper = model._wrap_audio_output(sink)
            provider = model.session()
            await provider.update_instructions("test")
            pending = provider.generate_reply()
            response = await peer.start(await peer.next())
            generation = await pending
            await peer.message(response)
            await peer.end(response)
            message = await anext(generation.message_stream)
            async for frame in message.audio_stream:
                await wrapper.capture_frame(frame)
            wrapper.flush()
            await provider.aclose()
            fresh = model.session()
            await fresh.update_instructions("fresh")
            sink.finish()
            await asyncio.sleep(0.01)
            assert peer.receipts.empty()
            assert peer.connections == 2

    async def test_detached_task_cannot_consume_explicit_edit_token(self):
        class DetachedAgent(Agent):
            async def update_chat_ctx(self, chat_ctx, **kwargs):
                await asyncio.create_task(super().update_chat_ctx(chat_ctx, **kwargs))

        async with peer_model() as (peer, model, _):
            async with AgentSession(llm=model) as session:
                agent = DetachedAgent(instructions="test")
                await session.start(agent)
                with pytest.raises(PrismHistoryUpdateError, match="mismatched"):
                    await model._replace_chat_ctx(agent, messages("not accepted"))
                assert not peer.history_operations

    @pytest.mark.parametrize("outcome", ["accepted", "refused", "active-omission"])
    async def test_acknowledged_append_merges_concurrent_speech_in_server_order(self, outcome):
        async with peer_model() as (peer, model, _):
            provider = model.session()
            await provider.update_chat_ctx(messages("old"))
            target = provider.chat_ctx
            target.add_message(role="user", content="A")
            target.add_message(role="user", content="B")
            peer.history_started.clear()
            peer.history_release.clear()
            if outcome == "refused":
                peer.fail_history_at = len(peer.history_operations) + 2
            update = asyncio.create_task(provider.update_chat_ctx(target))
            try:
                await asyncio.wait_for(peer.history_started.wait(), 3)
                peer.server_items.append(
                    {"id": "speech", "type": "message", "role": "user", "text": "incoming"}
                )
                await peer.emit(
                    "conversation.item.input_audio_transcription.completed",
                    item_id="speech",
                    transcript="incoming",
                )
                if outcome == "active-omission":
                    await peer.emit(
                        "response.created",
                        response={"id": "active", "prism_input_item_ids": ["speech"]},
                    )
                await wait_until(lambda: provider.chat_ctx.get_by_id("speech") is not None)
            finally:
                peer.history_release.set()
            if outcome == "accepted":
                await update
            else:
                with pytest.raises(PrismHistoryUpdateError) as caught:
                    await update
                assert caught.value.status == ("partial" if outcome == "refused" else "uncertain")
                if outcome == "active-omission":
                    assert caught.value.confirmed_chat_ctx is None
                    assert not caught.value.recoverable
                    assert peer.history_reads == 1
                    return
            assert peer.history_reads == 1
            assert [i.text_content for i in provider.chat_ctx.items] == [
                i["text"] for i in peer.server_items
            ]
            assert [i.text_content for i in provider.chat_ctx.items] == ["old", "incoming", "A"] + (
                ["B"] if outcome == "accepted" else []
            )

    @pytest.mark.parametrize("superseded_by", ["response", "speech", "report"])
    async def test_superseded_tool_replies_do_not_exhaust_response_holds(self, superseded_by):
        async with peer_model(playback="discard", context_mode="shared") as (peer, model, _):
            provider = model.session()
            await provider.update_instructions("test")
            errors = []
            provider.on("error", errors.append)
            for number in range(140):
                call = f"call-{number}"
                await tool_turn(peer, call)
                if superseded_by == "report":
                    await finish_discard_response(peer, await peer.start())
                await submit(provider, call)
                if superseded_by == "speech":
                    started = asyncio.Event()
                    provider.once("input_speech_started", lambda _, ready=started: ready.set())
                    await peer.emit(
                        "input_audio_buffer.speech_started",
                        item_id=f"input-{number}",
                        audio_start_ms=0,
                    )
                    await asyncio.wait_for(started.wait(), 3)
                    await peer.emit("prism.typed_input.ready")
                else:
                    await finish_discard_response(peer, await peer.start())
                request = await reply(peer, provider)
                assert "prism_call_id" not in request["response"]
                assert request["response"]["metadata"]["prism_cause"] == "proactive"
            assert not errors
            assert len(peer.items) == 140
            assert peer.connections == 1

    async def test_replaced_tool_reports_release_only_the_old_response(self):
        async with peer_model(playback="discard", context_mode="shared") as (peer, model, _):
            provider = model.session()
            await provider.update_instructions("test")
            for number in range(140):
                first, second = f"first-{number}", f"second-{number}"
                await tool_turn(peer, first)
                await tool_turn(peer, second)
                await finish_discard_response(peer, await peer.start())
                await submit(provider, first)
                await submit(provider, second)
                request = await reply(peer, provider)
                assert request["response"]["prism_call_id"] == second
            assert len(peer.items) == 280
            assert peer.connections == 1

    async def test_superseded_partial_result_keeps_unresolved_tool_reference(self):
        async with peer_model(playback="discard", context_mode="shared") as (peer, model, _):
            provider = model.session()
            await provider.update_instructions("test")
            await tool_turn(peer, "first", "pending")
            await submit(provider, "first")
            for _ in range(140):
                await finish_discard_response(peer, await peer.start())
            await submit(provider, "pending")
            request = await reply(peer, provider)
            assert request["response"]["prism_call_id"] == "pending"
            assert [item["call_id"] for item in peer.items] == ["first", "pending"]

    @pytest.mark.parametrize("ending", ["refused", "cancelled_before_start"])
    async def test_failed_tool_reply_releases_its_response_hold(self, ending):
        async with peer_model(playback="discard", context_mode="shared") as (peer, model, _):
            provider = model.session()
            await provider.update_instructions("test")
            for number in range(140):
                call = f"call-{number}"
                await tool_turn(peer, call)
                await submit(provider, call)
                if ending == "refused":
                    peer.refusals["response.create"] = ("tool_continuation_not_ready", "event")
                    with pytest.raises(PrismError, match="tool_continuation_not_ready"):
                        await provider.generate_reply()
                else:
                    pending = provider.generate_reply()
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                assert peer.requests.empty()
            await reply(peer, provider)
            assert len(peer.items) == 140
            assert peer.connections == 1

    async def test_in_flight_continuation_keeps_reference_until_fallback_settles(self):
        async with peer_model(playback="discard", context_mode="shared") as (peer, model, _):
            provider = model.session()
            await provider.update_instructions("test")
            parent = await tool_turn(peer, "call")
            await submit(provider, "call")
            pending = provider.generate_reply()
            request = await peer.next()
            try:
                for _ in range(140):
                    await finish_discard_response(peer, await peer.start())
                await peer.emit(
                    "error",
                    error={
                        "code": "tool_continuation_unavailable",
                        "message": "Superseded",
                        "scope": "event",
                        "owner": {"kind": "event", "event_id": request["event_id"]},
                    },
                )
                report = await peer.next()
                assert report["response"]["prism_call_id"] == "call"
                await finish_discard_response(peer, await peer.start(report))
                await pending
                for _ in range(140):
                    await finish_discard_response(peer, await peer.start())
                # The public SDK rejects this completed reference once ownership ends.
                assert provider._sdk is not None
                with pytest.raises(RimeInputError):
                    provider._sdk.hold_response(
                        r.ResponseRef(session_id=provider._sdk.info.session_id, response_id=parent)
                    )
                assert len(peer.items) == 1
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
