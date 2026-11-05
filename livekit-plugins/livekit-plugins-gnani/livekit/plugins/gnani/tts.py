# Copyright 2025 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text-to-Speech implementation for Gnani Vachana

This module provides a TTS implementation that uses the Gnani Vachana API,
supporting three synthesis modes via ``synthesize()``:
  - REST  (RESTChunkedStream) — single-request batch synthesis
  - SSE   (SSEChunkedStream)  — streaming via Server-Sent Events
  - WebSocket (WebSocketChunkedStream) — lowest-latency via WebSocket

Additionally, ``stream()`` uses WebSocket (SynthesizeStream) for real-time
token-by-token streaming input.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass, replace
from typing import Any, Literal

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils.misc import is_given

from .log import logger
from .models import (
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    DEFAULT_VOICES,
    MAX_SPEED,
    MIN_SPEED,
    RETIRED_MODELS,
    TIMBRE_V25_VOICES,
    TIMBRE_V30_VOICES,
    GnaniTTSBitrates,
    GnaniTTSContainers,
    GnaniTTSEncodings,
    GnaniTTSLanguages,
    GnaniTTSModels,
    GnaniTTSVoices,
)

GNANI_TTS_BASE_URL = "https://api.vachana.ai"

GnaniTTSSynthesizeMethod = Literal["rest", "sse", "websocket"]

SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)


_DEPRECATED_TTS_KWARGS = frozenset(("http_session",))


def _check_deprecated_tts_args(kwargs: dict[str, Any], *, caller: str = "TTS.__init__") -> None:
    """Warn about deprecated kwargs and raise on truly unknown ones."""
    for name in _DEPRECATED_TTS_KWARGS:
        if name in kwargs:
            logger.warning(f"`{name}` is deprecated and no longer used")

    unknown = set(kwargs) - _DEPRECATED_TTS_KWARGS
    if unknown:
        raise TypeError(
            f"{caller}() got unexpected keyword argument(s): {', '.join(sorted(unknown))}"
        )


# Voices known to belong to exactly one model. Used only to fail fast on a
# cross-model mismatch (which the API rejects with a 400); voices outside these
# sets are passed through untouched so newly added voices keep working.
_MODEL_VOICES: dict[str, frozenset[str]] = {
    "timbre-v2.5": frozenset(TIMBRE_V25_VOICES),
    "timbre-v3.0": frozenset(TIMBRE_V30_VOICES),
}


def _validate_model(model: str) -> None:
    if model in RETIRED_MODELS:
        raise ValueError(
            f"Gnani TTS model {model!r} has been retired by the API and is no longer "
            f"available. Use model='timbre-v2.5' or model='timbre-v3.0' instead "
            "(voices are model-specific, see https://docs.gnani.ai/api/TTS/available-voices)."
        )


def _validate_voice(model: str, voice: str) -> None:
    """Raise if ``voice`` is known to belong to a *different* model than ``model``."""
    own = _MODEL_VOICES.get(model)
    if own is None or voice in own:
        return
    for other_model, other_voices in _MODEL_VOICES.items():
        if other_model != model and voice in other_voices:
            raise ValueError(
                f"Voice {voice!r} belongs to model {other_model!r} and is not supported by "
                f"model {model!r}. Pass a {model} voice, e.g. voice={DEFAULT_VOICES[model]!r}."
            )


def _validate_speed(speed: float | None) -> None:
    if speed is not None and not (MIN_SPEED <= speed <= MAX_SPEED):
        raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}, got {speed}")


def _default_voice(model: str) -> str:
    return DEFAULT_VOICES.get(model, DEFAULT_VOICE)


@dataclass
class GnaniTTSOptions:
    api_key: str
    voice: str = DEFAULT_VOICE
    model: str = DEFAULT_MODEL
    language: str | None = None
    speed: float | None = None
    sample_rate: int = 16000
    encoding: str = "linear_pcm"
    container: str = "wav"
    num_channels: int = 1
    sample_width: int = 2
    bitrate: str | None = None
    base_url: str = GNANI_TTS_BASE_URL
    synthesize_method: str = "rest"


class TTS(tts.TTS):
    """Gnani Vachana Text-to-Speech implementation.

    Provides text-to-speech functionality using Gnani's Vachana platform.
    Supports REST, SSE, and WebSocket synthesis modes.

    Args:
        voice: Voice to use for synthesis. Voices belong to a model; when omitted
            the model's default is used (Pranav for timbre-v2.5, Avinash for
            timbre-v3.0). See https://docs.gnani.ai/api/TTS/available-voices.
        model: TTS model name: "timbre-v2.5" (default) or "timbre-v3.0".
            "timbre-v2.0" has been retired by the API and raises ``ValueError``.
        language: Language code (e.g. "hi-IN", "auto") for timbre-v2.5 and
            timbre-v3.0. timbre-v3.0 voices speak every supported language.
        speed: Speaking rate multiplier between 0.85 and 1.15. Omitted from the
            request when not set (the API default is 1.0).
        sample_rate: Audio output sample rate (8000, 16000, 22050, 24000, 44100, 48000).
        encoding: Audio encoding (linear_pcm or oggopus).
        container: Audio container format (raw, mp3, wav, ogg).
        api_key: Gnani API key (falls back to GNANI_API_KEY env var).
        base_url: Vachana API base URL.
        synthesize_method: Synthesis mode — "rest", "sse", or "websocket".
    """

    def __init__(
        self,
        *,
        voice: NotGivenOr[GnaniTTSVoices | str] = NOT_GIVEN,
        model: GnaniTTSModels | str = DEFAULT_MODEL,
        language: GnaniTTSLanguages | str | None = None,
        speed: float | None = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        encoding: GnaniTTSEncodings | str = "linear_pcm",
        container: GnaniTTSContainers | str = "wav",
        bitrate: GnaniTTSBitrates | str | None = None,
        api_key: str | None = None,
        base_url: str = GNANI_TTS_BASE_URL,
        synthesize_method: GnaniTTSSynthesizeMethod = "rest",
        **kwargs: Any,
    ) -> None:
        _check_deprecated_tts_args(kwargs)

        _validate_model(model)
        _validate_speed(speed)
        voice_is_default = not is_given(voice)
        resolved_voice = _default_voice(model) if not is_given(voice) else voice
        _validate_voice(model, resolved_voice)

        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {SUPPORTED_SAMPLE_RATES}, got {sample_rate}"
            )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._api_key = api_key or os.environ.get("GNANI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gnani API key is required. "
                "Provide it directly or set GNANI_API_KEY environment variable."
            )

        self._opts = GnaniTTSOptions(
            api_key=self._api_key,
            voice=resolved_voice,
            model=model,
            language=language,
            speed=speed,
            sample_rate=sample_rate,
            encoding=encoding,
            container=container,
            num_channels=num_channels,
            bitrate=bitrate,
            base_url=base_url,
            synthesize_method=synthesize_method,
        )
        self._voice_is_default = voice_is_default
        self._session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Gnani"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        if self._opts.synthesize_method == "sse":
            return SSEChunkedStream(tts=self, input_text=text, conn_options=conn_options)
        if self._opts.synthesize_method == "websocket":
            return WebSocketChunkedStream(tts=self, input_text=text, conn_options=conn_options)
        return RESTChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)

    def update_options(
        self,
        *,
        voice: GnaniTTSVoices | str | None = None,
        model: GnaniTTSModels | str | None = None,
        language: GnaniTTSLanguages | str | None = None,
        speed: float | None = None,
        **kwargs: Any,
    ) -> None:
        _check_deprecated_tts_args(kwargs, caller="TTS.update_options")

        new_model = model if model is not None else self._opts.model
        _validate_model(new_model)
        _validate_speed(speed)

        # Voices are model-specific: when only the model changes and the voice was
        # never chosen explicitly, move to the new model's default voice.
        new_voice = self._opts.voice
        voice_is_default = self._voice_is_default
        if voice is not None:
            new_voice = voice
            voice_is_default = False
        elif voice_is_default and model is not None:
            new_voice = _default_voice(new_model)
        _validate_voice(new_model, new_voice)

        # Apply only after everything validated, so a rejected call changes nothing.
        self._opts.model = new_model
        self._opts.voice = new_voice
        self._voice_is_default = voice_is_default
        if language is not None:
            self._opts.language = language
        if speed is not None:
            self._opts.speed = speed

    async def aclose(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_payload(opts: GnaniTTSOptions, text: str) -> dict:
    audio_config: dict = {
        "sample_rate": opts.sample_rate,
        "encoding": opts.encoding,
        "num_channels": opts.num_channels,
        "sample_width": opts.sample_width,
        "container": opts.container,
    }
    if opts.bitrate is not None:
        audio_config["bitrate"] = opts.bitrate
    payload: dict[str, Any] = {
        "text": text,
        "voice": opts.voice,
        "model": opts.model,
        "audio_config": audio_config,
    }
    if opts.language is not None:
        payload["language"] = opts.language
    if opts.speed is not None:
        payload["speed"] = opts.speed
    return payload


def _build_headers(opts: GnaniTTSOptions) -> dict[str, str]:
    return {
        "X-API-Key-ID": opts.api_key,
        "Content-Type": "application/json",
    }


def _mime_type(opts: GnaniTTSOptions) -> str:
    if opts.container == "raw":
        return "audio/pcm"
    return f"audio/{opts.container}"


# ---------------------------------------------------------------------------
# REST ChunkedStream
# ---------------------------------------------------------------------------


class RESTChunkedStream(tts.ChunkedStream):
    """REST-based chunked TTS — POST /api/v1/tts/inference."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            async with self._tts._ensure_session().post(
                url=f"{self._opts.base_url}/api/v1/tts/inference",
                json=_build_payload(self._opts, self._input_text),
                headers=_build_headers(self._opts),
                timeout=aiohttp.ClientTimeout(
                    total=self._conn_options.timeout,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as res:
                if res.status != 200:
                    error_text = await res.text()
                    logger.error("Gnani TTS REST error: %s - %s", res.status, error_text)
                    raise APIStatusError(
                        message=f"Gnani TTS API Error ({res.status}): {error_text}",
                        status_code=res.status,
                        body=error_text,
                    )

                audio_bytes = await res.read()

                request_id = utils.shortuuid()
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type=_mime_type(self._opts),
                )
                output_emitter.push(audio_bytes)
                output_emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError("Gnani TTS REST request timed out") from e
        except (APIStatusError, APIConnectionError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError(f"Gnani TTS REST error: {e}") from e


# ---------------------------------------------------------------------------
# SSE ChunkedStream
# ---------------------------------------------------------------------------


class SSEChunkedStream(tts.ChunkedStream):
    """SSE-based chunked TTS — POST /api/v1/tts/sse.

    Each SSE ``audio`` payload carries container-encoded audio (WAV by
    default). Bytes are pushed straight to the ``AudioEmitter``, whose built-in
    ``AudioStreamDecoder`` decodes and re-frames them into a single contiguous
    PCM stream.
    """

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        try:
            async with self._tts._ensure_session().post(
                url=f"{self._opts.base_url}/api/v1/tts/sse",
                json=_build_payload(self._opts, self._input_text),
                headers=_build_headers(self._opts),
                timeout=aiohttp.ClientTimeout(
                    total=self._conn_options.timeout,
                    sock_connect=self._conn_options.timeout,
                ),
                read_bufsize=10 * 1024 * 1024,
            ) as res:
                if res.status != 200:
                    error_text = await res.text()
                    logger.error("Gnani TTS SSE error: %s - %s", res.status, error_text)
                    raise APIStatusError(
                        message=f"Gnani TTS SSE Error ({res.status}): {error_text}",
                        status_code=res.status,
                        body=error_text,
                    )

                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type=_mime_type(self._opts),
                )

                buf = ""
                async for raw_bytes in res.content:
                    raw_line = raw_bytes.decode("utf-8").strip()
                    if not raw_line:
                        continue
                    # Ignore SSE keep-alive comments (":") and non-data metadata
                    # fields; appending them would corrupt the JSON accumulator
                    # and drop the remaining audio for this request.
                    if raw_line.startswith((":", "event:", "id:", "retry:")):
                        continue
                    if raw_line.startswith("data:"):
                        raw_line = raw_line[5:].strip()
                        if not raw_line:
                            continue

                    buf += raw_line
                    try:
                        payload = json.loads(buf)
                    except json.JSONDecodeError:
                        continue
                    buf = ""

                    if payload.get("status") == "error" or "error" in payload:
                        raise APIStatusError(
                            message=payload.get("message", json.dumps(payload)),
                            status_code=500,
                            body=payload,
                        )
                    if payload.get("status") == "streaming_started":
                        continue
                    if payload.get("is_final", False):
                        audio_b64 = payload.get("audio", "")
                        if audio_b64:
                            output_emitter.push(base64.b64decode(audio_b64))
                        break

                    audio_b64 = payload.get("audio", "")
                    if audio_b64:
                        output_emitter.push(base64.b64decode(audio_b64))

                output_emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError("Gnani TTS SSE request timed out") from e
        except (APIStatusError, APIConnectionError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError(f"Gnani TTS SSE error: {e}") from e


# ---------------------------------------------------------------------------
# WebSocket ChunkedStream (for synthesize_method="websocket")
# ---------------------------------------------------------------------------


class WebSocketChunkedStream(tts.ChunkedStream):
    """WebSocket-based chunked TTS — wss://api.vachana.ai/api/v1/tts.

    Wraps the WebSocket endpoint into the ChunkedStream interface so that
    ``synthesize()`` can use it when ``synthesize_method="websocket"``. Audio
    bytes are pushed straight to the ``AudioEmitter``, whose built-in
    ``AudioStreamDecoder`` decodes and re-frames them for LiveKit playback.
    """

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    def _build_ws_url(self) -> str:
        base = self._opts.base_url
        if base.startswith("https://"):
            ws_base = "wss://" + base[len("https://") :]
        elif base.startswith("http://"):
            ws_base = "ws://" + base[len("http://") :]
        else:
            ws_base = "wss://" + base
        return f"{ws_base}/api/v1/tts"

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        import websockets

        request_id = utils.shortuuid()
        try:
            ws_url = self._build_ws_url()
            async with websockets.connect(
                ws_url,
                additional_headers=_build_headers(self._opts),
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
            ) as ws:
                request_body = _build_payload(self._opts, self._input_text)
                await ws.send(json.dumps(request_body))

                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type=_mime_type(self._opts),
                )

                async for msg in ws:
                    if isinstance(msg, bytes):
                        output_emitter.push(msg)
                        continue

                    payload = json.loads(msg)
                    msg_type = payload.get("type", "")

                    if msg_type == "audio":
                        inner = payload.get("data", {})
                        audio_b64 = inner.get("audio", "")
                        if audio_b64:
                            output_emitter.push(base64.b64decode(audio_b64))

                    elif msg_type == "complete":
                        inner = payload.get("data")
                        if inner is not None:
                            audio_b64 = inner.get("audio", "")
                            if audio_b64:
                                output_emitter.push(base64.b64decode(audio_b64))
                        break

                    elif msg_type == "error":
                        error_msg = payload.get("message", "Unknown error")
                        logger.error("Gnani TTS WS error: %s", error_msg)
                        raise APIStatusError(
                            message=f"Gnani TTS stream error: {error_msg}",
                            status_code=500,
                            body=error_msg,
                        )

                output_emitter.flush()

        except websockets.exceptions.ConnectionClosed as e:
            raise APIConnectionError(f"Gnani TTS WebSocket closed: {e}") from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Gnani TTS WebSocket timed out") from e
        except (APIStatusError, APIConnectionError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError(f"Gnani TTS WebSocket error: {e}") from e


# ---------------------------------------------------------------------------
# WebSocket SynthesizeStream (for stream())
# ---------------------------------------------------------------------------


class SynthesizeStream(tts.SynthesizeStream):
    """WebSocket-based streaming TTS — wss://api.vachana.ai/api/v1/tts."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    def _build_ws_url(self) -> str:
        base = self._opts.base_url
        if base.startswith("https://"):
            ws_base = "wss://" + base[len("https://") :]
        elif base.startswith("http://"):
            ws_base = "ws://" + base[len("http://") :]
        else:
            ws_base = "wss://" + base
        return f"{ws_base}/api/v1/tts"

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        import websockets

        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type=_mime_type(self._opts),
            stream=True,
        )

        text_parts: list[str] = []
        async for data in self._input_ch:
            if isinstance(data, str):
                text_parts.append(data)
            elif isinstance(data, self._FlushSentinel):
                break

        full_text = "".join(text_parts).strip()
        if not full_text:
            return

        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        try:
            ws_url = self._build_ws_url()
            async with websockets.connect(
                ws_url,
                additional_headers=_build_headers(self._opts),
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
            ) as ws:
                request_body = _build_payload(self._opts, full_text)
                await ws.send(json.dumps(request_body))

                self._mark_started()

                async for msg in ws:
                    if isinstance(msg, bytes):
                        output_emitter.push(msg)
                        continue

                    payload = json.loads(msg)
                    msg_type = payload.get("type", "")

                    if msg_type == "audio":
                        inner = payload.get("data", {})
                        audio_b64 = inner.get("audio", "")
                        if audio_b64:
                            output_emitter.push(base64.b64decode(audio_b64))

                    elif msg_type == "complete":
                        inner = payload.get("data")
                        if inner is not None:
                            audio_b64 = inner.get("audio", "")
                            if audio_b64:
                                output_emitter.push(base64.b64decode(audio_b64))
                        break

                    elif msg_type == "error":
                        error_msg = payload.get("message", "Unknown error")
                        logger.error("Gnani TTS WS error: %s", error_msg)
                        raise APIStatusError(
                            message=f"Gnani TTS stream error: {error_msg}",
                            status_code=500,
                            body=error_msg,
                        )

        except websockets.exceptions.ConnectionClosed as e:
            raise APIConnectionError(f"Gnani TTS WebSocket closed: {e}") from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Gnani TTS WebSocket timed out") from e
        except (APIStatusError, APIConnectionError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError(f"Gnani TTS WebSocket error: {e}") from e

        # Only flush/close the segment on the success path. On error the
        # per-attempt output_emitter is discarded before retry, so flushing or
        # ending the segment here would only add an unnecessary final frame.
        output_emitter.flush()
        output_emitter.end_segment()


__all__ = ["TTS", "SynthesizeStream"]
