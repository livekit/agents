"""Tests for Gnani Vachana TTS plugin configuration and behavior."""

from __future__ import annotations

import base64
import json
import struct
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions, APIStatusError

pytestmark = pytest.mark.plugin("gnani")


def test_tts_requires_api_key():
    """TTS constructor raises when no API key is provided."""
    from livekit.plugins.gnani import TTS

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            TTS(api_key=None)


def test_tts_accepts_api_key_directly():
    """TTS constructor accepts api_key argument."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.api_key == "test-key"


def test_tts_accepts_api_key_from_env():
    """TTS constructor reads GNANI_API_KEY from environment."""
    from livekit.plugins.gnani import TTS

    with patch.dict("os.environ", {"GNANI_API_KEY": "env-key"}):
        tts = TTS()
        assert tts._opts.api_key == "env-key"


def test_tts_default_voice():
    """TTS defaults to 'Pranav' voice."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.voice == "Pranav"


def test_tts_custom_voice():
    """TTS accepts custom voice."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", voice="Raju")
    assert tts._opts.voice == "Raju"


def test_tts_all_voices_accepted():
    """TTS accepts all documented voices."""
    from livekit.plugins.gnani import TTS

    voices = ["Karan", "Simran", "Nara", "Riya", "Viraj", "Raju"]
    for voice in voices:
        tts = TTS(api_key="test-key", voice=voice)
        assert tts._opts.voice == voice


def test_tts_accepts_arbitrary_voice():
    """TTS accepts arbitrary voice strings (validated server-side)."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", voice="some-new-voice")
    assert tts._opts.voice == "some-new-voice"


def test_tts_default_model():
    """TTS defaults to timbre-v2.5 (DEFAULT_MODEL); timbre-v2.0 is retired."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.model == "timbre-v2.5"


def test_tts_model_independent_of_voice():
    """The default model is used regardless of the selected voice."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", voice="Simran")
    assert tts._opts.model == "timbre-v2.5"
    assert tts.model == "timbre-v2.5"


def test_tts_model_explicit_override():
    """Explicit model parameter overrides default."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", voice="Karan", model="custom-model")
    assert tts._opts.model == "custom-model"


def test_tts_model_property():
    """TTS model property returns the configured model."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts.model == "timbre-v2.5"


def test_tts_provider_property():
    """TTS provider property returns 'Gnani'."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts.provider == "Gnani"


def test_tts_capabilities():
    """TTS reports streaming=True."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts.capabilities.streaming is True


def test_tts_default_sample_rate():
    """TTS defaults to 16000 Hz sample rate."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts.sample_rate == 16000


def test_tts_custom_sample_rate():
    """TTS accepts custom sample rate."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", sample_rate=44100)
    assert tts.sample_rate == 44100


def test_tts_default_encoding():
    """TTS defaults to linear_pcm encoding."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.encoding == "linear_pcm"


def test_tts_default_container():
    """TTS defaults to wav container."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.container == "wav"


def test_tts_custom_audio_config():
    """TTS accepts custom encoding and container."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", encoding="oggopus", container="ogg")
    assert tts._opts.encoding == "oggopus"
    assert tts._opts.container == "ogg"


def test_tts_update_options_voice():
    """update_options can change voice."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", voice="Karan")
    tts.update_options(voice="Simran")
    assert tts._opts.voice == "Simran"


def test_tts_update_options_voice_and_model():
    """update_options with both voice and model uses the explicit model."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", voice="Karan")
    tts.update_options(voice="Riya", model="custom-model")
    assert tts._opts.voice == "Riya"
    assert tts._opts.model == "custom-model"


def test_tts_update_options_accepts_arbitrary_voice():
    """update_options accepts arbitrary voice strings (validated server-side)."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    tts.update_options(voice="some-new-voice")
    assert tts._opts.voice == "some-new-voice"


def test_tts_update_options_model():
    """update_options can change model."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    tts.update_options(model="custom-model")
    assert tts._opts.model == "custom-model"


def test_tts_default_synthesize_method():
    """TTS defaults to 'rest' synthesize method."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.synthesize_method == "rest"


def test_tts_sse_synthesize_method():
    """TTS with synthesize_method='sse' stores the option correctly."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", synthesize_method="sse")
    assert tts._opts.synthesize_method == "sse"


def test_tts_rest_synthesize_method():
    """TTS with synthesize_method='rest' stores the option correctly."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", synthesize_method="rest")
    assert tts._opts.synthesize_method == "rest"


def test_tts_websocket_synthesize_method():
    """TTS with synthesize_method='websocket' stores the option correctly."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", synthesize_method="websocket")
    assert tts._opts.synthesize_method == "websocket"


def test_synthesize_routes_rest_by_default():
    """synthesize() returns RESTChunkedStream when method is 'rest'."""
    from unittest.mock import MagicMock, patch

    from livekit.plugins.gnani import TTS
    from livekit.plugins.gnani.tts import RESTChunkedStream

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    tts = TTS(api_key="test-key", synthesize_method="rest")
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.synthesize("hello")
    assert isinstance(stream, RESTChunkedStream)


def test_synthesize_routes_sse():
    """synthesize() returns SSEChunkedStream when method is 'sse'."""
    from unittest.mock import MagicMock, patch

    from livekit.plugins.gnani import TTS
    from livekit.plugins.gnani.tts import SSEChunkedStream

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    tts = TTS(api_key="test-key", synthesize_method="sse")
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.synthesize("hello")
    assert isinstance(stream, SSEChunkedStream)


def test_synthesize_routes_websocket():
    """synthesize() returns WebSocketChunkedStream when method is 'websocket'."""
    from unittest.mock import MagicMock, patch

    from livekit.plugins.gnani import TTS
    from livekit.plugins.gnani.tts import WebSocketChunkedStream

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    tts = TTS(api_key="test-key", synthesize_method="websocket")
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.synthesize("hello")
    assert isinstance(stream, WebSocketChunkedStream)


def test_tts_base_url_default():
    """TTS defaults to Vachana API base URL."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.base_url == "https://api.vachana.ai"


def test_tts_ws_url_from_https():
    """SynthesizeStream builds wss:// from https:// base."""
    from unittest.mock import MagicMock, patch

    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.gnani.tts import TTS, SynthesizeStream

    tts = TTS(api_key="test-key")

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = SynthesizeStream(tts=tts, conn_options=DEFAULT_API_CONNECT_OPTIONS)

    assert stream._build_ws_url() == "wss://api.vachana.ai/api/v1/tts"


def test_tts_ws_url_from_http():
    """SynthesizeStream builds ws:// from http:// base."""
    from unittest.mock import MagicMock, patch

    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.gnani.tts import TTS, SynthesizeStream

    tts = TTS(api_key="test-key", base_url="http://localhost:9090")

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = SynthesizeStream(tts=tts, conn_options=DEFAULT_API_CONNECT_OPTIONS)

    assert stream._build_ws_url() == "ws://localhost:9090/api/v1/tts"


def test_tts_default_num_channels():
    """TTS defaults to 1 channel."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.num_channels == 1
    assert tts.num_channels == 1


def test_tts_custom_num_channels():
    """TTS accepts custom num_channels."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", num_channels=2)
    assert tts._opts.num_channels == 2


def test_tts_default_bitrate():
    """TTS defaults bitrate to None."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.bitrate is None


def test_tts_custom_bitrate():
    """TTS accepts custom bitrate."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", bitrate="128k")
    assert tts._opts.bitrate == "128k"


def test_tts_rejects_invalid_sample_rate():
    """TTS rejects unsupported sample rates."""
    from livekit.plugins.gnani import TTS

    with pytest.raises(ValueError, match="sample_rate"):
        TTS(api_key="test-key", sample_rate=12345)


def test_tts_all_sample_rates_accepted():
    """TTS accepts all documented sample rates."""
    from livekit.plugins.gnani import TTS

    for rate in (8000, 16000, 22050, 24000, 44100, 48000):
        tts = TTS(api_key="test-key", sample_rate=rate)
        assert tts.sample_rate == rate


def test_tts_stream_returns_synthesize_stream():
    """TTS.stream() returns a SynthesizeStream instance."""
    from unittest.mock import MagicMock, patch

    from livekit.plugins.gnani import TTS
    from livekit.plugins.gnani.tts import SynthesizeStream

    tts = TTS(api_key="test-key")

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.stream()
    assert isinstance(stream, SynthesizeStream)


def test_tts_update_options_preserves_other_fields():
    """update_options only changes specified fields."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", voice="Karan")
    tts.update_options(voice="Raju")
    assert tts._opts.voice == "Raju"
    assert tts._opts.model == "timbre-v2.5"
    assert tts._opts.encoding == "linear_pcm"


def test_tts_websocket_chunked_stream_ws_url():
    """WebSocketChunkedStream builds correct WS URL."""
    from unittest.mock import MagicMock, patch

    from livekit.plugins.gnani import TTS
    from livekit.plugins.gnani.tts import WebSocketChunkedStream

    tts = TTS(api_key="test-key", synthesize_method="websocket")

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.synthesize("hello")
    assert isinstance(stream, WebSocketChunkedStream)
    assert stream._build_ws_url() == "wss://api.vachana.ai/api/v1/tts"


# ---------------------------------------------------------------------------
# timbre-v3.0, retired timbre-v2.0, model-aware voices, speed and language
# ---------------------------------------------------------------------------


def test_tts_default_voice_depends_on_model():
    """The default voice is the model's own: voices are not shared across models."""
    from livekit.plugins.gnani import TTS

    assert TTS(api_key="k", model="timbre-v2.5")._opts.voice == "Pranav"
    assert TTS(api_key="k", model="timbre-v3.0")._opts.voice == "Avinash"


def test_tts_v3_explicit_voice():
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="k", model="timbre-v3.0", voice="Shivani")
    assert tts._opts.model == "timbre-v3.0"
    assert tts._opts.voice == "Shivani"


@pytest.mark.parametrize("model", ["timbre-v2.0"])
def test_tts_retired_model_is_rejected(model):
    """timbre-v2.0 is rejected by the API (HTTP 400), so fail fast with a clear message."""
    from livekit.plugins.gnani import TTS

    with pytest.raises(ValueError, match="retired") as exc:
        TTS(api_key="k", model=model)
    assert "timbre-v2.5" in str(exc.value) and "timbre-v3.0" in str(exc.value)


def test_tts_update_options_rejects_retired_model_and_changes_nothing():
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="k", model="timbre-v3.0")
    with pytest.raises(ValueError, match="retired"):
        tts.update_options(model="timbre-v2.0", voice="Pranav", speed=1.1)
    assert tts._opts.model == "timbre-v3.0"
    assert tts._opts.voice == "Avinash"
    assert tts._opts.speed is None


@pytest.mark.parametrize(
    ("model", "voice"),
    [
        ("timbre-v3.0", "Pranav"),  # a timbre-v2.5 voice
        ("timbre-v2.5", "Avinash"),  # a timbre-v3.0 voice
    ],
)
def test_tts_known_cross_model_voice_is_rejected(model, voice):
    """The API answers 400 for these; surface it at construction instead of first synthesis."""
    from livekit.plugins.gnani import TTS

    with pytest.raises(ValueError, match="not supported by model"):
        TTS(api_key="k", model=model, voice=voice)


def test_tts_unknown_voice_is_passed_through_for_every_model():
    """Voices Gnani adds later must keep working without a plugin release."""
    from livekit.plugins.gnani import TTS

    for model in ("timbre-v2.5", "timbre-v3.0"):
        assert TTS(api_key="k", model=model, voice="BrandNewVoice")._opts.voice == "BrandNewVoice"


def test_tts_update_options_model_switch_moves_default_voice():
    """A voice the user never chose follows the model; otherwise the API would 400."""
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="k")
    assert tts._opts.voice == "Pranav"
    tts.update_options(model="timbre-v3.0")
    assert (tts._opts.model, tts._opts.voice) == ("timbre-v3.0", "Avinash")
    tts.update_options(model="timbre-v2.5")
    assert (tts._opts.model, tts._opts.voice) == ("timbre-v2.5", "Pranav")


def test_tts_update_options_model_switch_keeps_explicit_voice_when_valid():
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="k", model="timbre-v3.0", voice="Shivani")
    tts.update_options(model="timbre-v3.0")
    assert tts._opts.voice == "Shivani"


def test_tts_update_options_explicit_voice_mismatching_new_model_is_rejected_atomically():
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="k", model="timbre-v2.5", voice="Nalini")
    with pytest.raises(ValueError, match="not supported by model"):
        tts.update_options(model="timbre-v3.0")  # Nalini was chosen explicitly
    assert (tts._opts.model, tts._opts.voice) == ("timbre-v2.5", "Nalini")

    tts.update_options(model="timbre-v3.0", voice="Shivani")
    assert (tts._opts.model, tts._opts.voice) == ("timbre-v3.0", "Shivani")


def test_tts_voice_catalogues_are_disjoint():
    """Voices are bound to one model; the catalogue check relies on that."""
    from typing import get_args

    from livekit.plugins.gnani.models import (
        GnaniTTSV25Voices,
        GnaniTTSV30Voices,
        GnaniTTSVoices,
    )

    v25, v30 = set(get_args(GnaniTTSV25Voices)), set(get_args(GnaniTTSV30Voices))
    assert len(v30) == 13
    assert not v25 & v30
    assert "Pranav" in v25 and "Avinash" in v30
    assert set(get_args(GnaniTTSVoices)) == v25 | v30


def test_tts_speed_validation():
    from livekit.plugins.gnani import TTS

    assert TTS(api_key="k", speed=0.85)._opts.speed == 0.85
    assert TTS(api_key="k", speed=1.15)._opts.speed == 1.15
    for bad in (0.5, 0.849, 1.151, 2.0):
        with pytest.raises(ValueError, match="speed"):
            TTS(api_key="k", speed=bad)


def test_tts_update_options_speed_and_language():
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="k", model="timbre-v3.0")
    tts.update_options(speed=1.1, language="hi-IN")
    assert tts._opts.speed == 1.1
    assert tts._opts.language == "hi-IN"
    with pytest.raises(ValueError, match="speed"):
        tts.update_options(speed=3.0)
    assert tts._opts.speed == 1.1


def _payload(**kwargs):
    from livekit.plugins.gnani import TTS
    from livekit.plugins.gnani.tts import _build_payload

    tts = TTS(api_key="k", **kwargs)
    return _build_payload(tts._opts, "namaste")


def test_payload_v3_sends_language_speed_and_model():
    """timbre-v3.0 honors language (the API accepts it); it used to be dropped."""
    body = _payload(model="timbre-v3.0", language="hi-IN", speed=1.1)
    assert body["model"] == "timbre-v3.0"
    assert body["voice"] == "Avinash"
    assert body["language"] == "hi-IN"
    assert body["speed"] == 1.1
    assert body["text"] == "namaste"


def test_payload_v25_sends_language_and_speed():
    body = _payload(model="timbre-v2.5", language="hi-en", speed=0.9)
    assert body["voice"] == "Pranav"
    assert body["language"] == "hi-en"
    assert body["speed"] == 0.9


def test_payload_omits_optional_fields_when_unset():
    """Unset language/speed are not sent, so the API's own defaults apply."""
    for model in ("timbre-v2.5", "timbre-v3.0"):
        body = _payload(model=model)
        assert "language" not in body
        assert "speed" not in body


def test_payload_passes_audio_tags_through_untouched():
    """Audio tags (timbre-v3.0) are inline text; the plugin must not alter them."""
    from livekit.plugins.gnani import TTS
    from livekit.plugins.gnani.tts import _build_payload

    tts = TTS(api_key="k", model="timbre-v3.0")
    text = "Haha <laugh> sach mein? <whispers> dheere bolo <pauses> theek hai."
    assert _build_payload(tts._opts, text)["text"] == text


# ================================================================================================
# TTS behaviour against a local fake of the Vachana API
#
# Gnani TTS request/response behaviour against a local fake of the Vachana API.
#
# The fake reproduces wire behaviour observed on the live API (2026-10-30):
#
# * REST returns a complete WAV file.
# * SSE / WebSocket return one WAV header with ``0xFFFFFFFF`` placeholder sizes
#   (the length is unknown while streaming) followed by PCM, then raw PCM chunks.
# * A WebSocket request ends with *two* ``complete`` messages: one carrying the final
#   audio chunk, then a bare ``"Streaming completed"``.
# * Unsupported models get ``400 {"success": false, "message": ..., "status_code": 400}``.
# ================================================================================================


SAMPLE_RATE = 16000
# 0.5 s of a quiet ramp, so we can tell real audio from padding/silence
PCM = struct.pack("<8000h", *(((i * 37) % 600) + 200 for i in range(8000)))


def _wav_header(data_len: int | None, sample_rate: int = SAMPLE_RATE) -> bytes:
    """44-byte PCM16 mono WAV header; ``data_len=None`` mimics the streaming placeholder."""
    unknown = 0xFFFFFFFF
    riff_size = unknown if data_len is None else 36 + data_len
    size = unknown if data_len is None else data_len
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", size)
    )


def _chunks(pcm: bytes, n: int = 4) -> list[bytes]:
    """Split a streamed response: the first chunk carries the WAV header, the rest are raw PCM."""
    step = len(pcm) // n
    parts = [pcm[i * step : (i + 1) * step] for i in range(n - 1)] + [pcm[(n - 1) * step :]]
    parts[0] = _wav_header(None) + parts[0]
    return parts


class FakeGnani:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []

    def _record(self, request: web.Request, body: dict[str, Any]) -> None:
        self.requests.append(body)
        self.headers.append(dict(request.headers))

    @staticmethod
    def _error(body: dict[str, Any]) -> web.Response | None:
        if body.get("model") not in ("timbre-v2.5", "timbre-v3.0"):
            return web.json_response(
                {
                    "success": False,
                    "message": f"Model '{body.get('model')}' is deprecated and is no longer "
                    'available. The supported models are "timbre-v2.5", "timbre-v3.0".',
                    "status_code": 400,
                },
                status=400,
            )
        return None

    async def rest(self, request: web.Request) -> web.Response:
        body = await request.json()
        self._record(request, body)
        if (err := self._error(body)) is not None:
            return err
        return web.Response(body=_wav_header(len(PCM)) + PCM, content_type="audio/wav")

    async def sse(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self._record(request, body)
        if (err := self._error(body)) is not None:
            return err
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b'event: start\ndata: {"status": "streaming_started", "text": "hi"}\n\n')
        parts = _chunks(PCM)
        for i, part in enumerate(parts, start=1):
            last = i == len(parts)
            payload = {"chunk_index": i, "audio": base64.b64encode(part).decode()}
            if last:
                payload["is_final"] = True
            await resp.write(f"event: chunk\ndata: {json.dumps(payload)}\n\n".encode())
        await resp.write_eof()
        return resp

    async def ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            body = json.loads(msg.data)
            self._record(request, body)
            if (err := self._error(body)) is not None:
                await ws.send_json({"type": "error", "message": json.loads(err.text)["message"]})
                continue
            await ws.send_json({"type": "start", "message": "Streaming started"})
            parts = _chunks(PCM)
            for i, part in enumerate(parts[:-1], start=1):
                await ws.send_json(
                    {
                        "type": "audio",
                        "data": {"chunk_index": i, "audio": base64.b64encode(part).decode()},
                    }
                )
            await ws.send_json(
                {
                    "type": "complete",
                    "data": {
                        "chunk_index": len(parts),
                        "audio": base64.b64encode(parts[-1]).decode(),
                        "is_final": True,
                    },
                }
            )
            await ws.send_json({"type": "complete", "message": "Streaming completed"})
        return ws


@pytest.fixture
async def server():
    fake = FakeGnani()
    app = web.Application()
    app.router.add_post("/api/v1/tts/inference", fake.rest)
    app.router.add_post("/api/v1/tts/sse", fake.sse)
    app.router.add_get("/api/v1/tts", fake.ws)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    fake.base_url = f"http://127.0.0.1:{port}"  # type: ignore[attr-defined]
    fake.session = aiohttp.ClientSession()  # type: ignore[attr-defined]
    try:
        yield fake
    finally:
        await fake.session.close()  # type: ignore[attr-defined]
        await runner.cleanup()


def _make_tts(server: Any, **kwargs: Any):
    from livekit.plugins.gnani import TTS

    tts = TTS(api_key="test-key", base_url=server.base_url, sample_rate=SAMPLE_RATE, **kwargs)
    tts._session = server.session
    return tts


OPTS = APIConnectOptions(max_retry=0, timeout=10)


async def _synthesize(tts, text: str = "namaste") -> bytes:
    audio = bytearray()
    async with tts.synthesize(text, conn_options=OPTS) as stream:
        async for ev in stream:
            audio += bytes(ev.frame.data)
    return bytes(audio)


def _assert_is_the_speech(audio: bytes) -> None:
    """Every PCM sample must come back unchanged (no clipped header, no dropped chunk);
    the emitter may only zero-pad the final frame."""
    assert audio[: len(PCM)] == PCM
    assert set(audio[len(PCM) :]) <= {0}


@pytest.mark.parametrize("method", ["rest", "sse", "websocket"])
@pytest.mark.parametrize("model", ["timbre-v2.5", "timbre-v3.0"])
async def test_synthesize_sends_model_aware_request_and_returns_clean_audio(server, method, model):
    tts = _make_tts(server, model=model, synthesize_method=method, language="hi-IN", speed=1.1)

    audio = await _synthesize(tts)

    sent = server.requests[-1]
    assert sent["model"] == model
    assert sent["voice"] == ("Avinash" if model == "timbre-v3.0" else "Pranav")
    assert sent["language"] == "hi-IN"
    assert sent["speed"] == 1.1
    assert sent["text"] == "namaste"
    assert server.headers[-1]["X-API-Key-ID"] == "test-key"
    _assert_is_the_speech(audio)


@pytest.mark.parametrize("method", ["rest", "sse", "websocket"])
async def test_synthesize_omits_unset_optional_fields(server, method):
    tts = _make_tts(server, model="timbre-v3.0", synthesize_method=method)
    await _synthesize(tts)
    assert "language" not in server.requests[-1]
    assert "speed" not in server.requests[-1]


async def test_stream_synthesizes_text_pushed_incrementally(server):
    """stream() (always WebSocket) sends one request with the joined text."""
    tts = _make_tts(server, model="timbre-v3.0", voice="Shivani")

    audio = bytearray()
    async with tts.stream(conn_options=OPTS) as stream:
        stream.push_text("Haha <laugh> ")
        stream.push_text("sach mein?")
        stream.end_input()
        async for ev in stream:
            audio += bytes(ev.frame.data)

    assert len(server.requests) == 1
    assert server.requests[0]["text"] == "Haha <laugh> sach mein?"
    assert server.requests[0]["voice"] == "Shivani"
    assert server.requests[0]["model"] == "timbre-v3.0"
    _assert_is_the_speech(bytes(audio))


@pytest.mark.parametrize("method", ["rest", "sse"])
async def test_api_error_is_surfaced_with_status_and_body(server, method):
    """A model the API rejects (e.g. a retired or unknown one) raises APIStatusError(400)."""
    tts = _make_tts(server, model="timbre-v9.9", synthesize_method=method)

    with pytest.raises(APIStatusError) as exc:
        await _synthesize(tts)

    assert exc.value.status_code == 400
    assert "no longer available" in str(exc.value) or "no longer available" in str(exc.value.body)


async def test_websocket_api_error_is_surfaced(server):
    tts = _make_tts(server, model="timbre-v9.9", synthesize_method="websocket")

    with pytest.raises(APIStatusError, match="no longer available"):
        await _synthesize(tts)
