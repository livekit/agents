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

from .log import logger

GNANI_TTS_BASE_URL = "https://api.vachana.ai"

GnaniTTSVoices = Literal[
    "Karan",
    "Simran",
    "Nara",
    "Riya",
    "Viraj",
    "Raju",
]

SUPPORTED_VOICES: set[str] = {"Karan", "Simran", "Nara", "Riya", "Viraj", "Raju"}

GnaniTTSEncodings = Literal["linear_pcm", "oggopus"]
GnaniTTSContainers = Literal["raw", "mp3", "wav", "mulaw", "ogg"]
GnaniTTSBitrates = Literal["96k", "128k", "192k"]
GnaniTTSSynthesizeMethod = Literal["rest", "sse", "websocket"]

SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 44100)

_WAV_HEADER_SIZE = 44


@dataclass
class GnaniTTSOptions:
    api_key: str
    voice: str = "Karan"
    model: str = "vachana-voice-v3"
    sample_rate: int = 16000
    encoding: str = "linear_pcm"
    container: str = "wav"
    num_channels: int = 1
    sample_width: int = 2
    bitrate: str | None = None
    base_url: str = GNANI_TTS_BASE_URL
    synthesize_method: str = "rest"


_DEPRECATED_TTS_KWARGS = frozenset(("language", "http_session"))


def _check_deprecated_tts_args(kwargs: dict[str, Any]) -> None:
    """Warn about deprecated kwargs and raise on truly unknown ones."""
    for name in _DEPRECATED_TTS_KWARGS:
        if name in kwargs:
            logger.warning(f"`{name}` is deprecated and no longer used")

    unknown = set(kwargs) - _DEPRECATED_TTS_KWARGS
    if unknown:
        raise TypeError(
            f"TTS.__init__() got unexpected keyword argument(s): {', '.join(sorted(unknown))}"
        )


class TTS(tts.TTS):
    """Gnani Vachana Text-to-Speech implementation.

    Provides text-to-speech functionality using Gnani's Vachana platform.
    Supports REST, SSE, and WebSocket synthesis modes.

    Args:
        voice: Voice to use for synthesis (Karan, Simran, Riya, etc.).
        model: TTS model name (default: vachana-voice-v3).
        sample_rate: Audio output sample rate (8000-44100).
        encoding: Audio encoding (linear_pcm or oggopus).
        container: Audio container format (raw, mp3, wav, mulaw, ogg).
        api_key: Gnani API key (falls back to GNANI_API_KEY env var).
        base_url: Vachana API base URL.
        synthesize_method: Synthesis mode — "rest", "sse", or "websocket".
    """

    def __init__(
        self,
        *,
        voice: GnaniTTSVoices | str = "Karan",
        model: str = "vachana-voice-v3",
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

        if voice not in SUPPORTED_VOICES:
            raise ValueError(
                f"Voice '{voice}' not supported. "
                f"Supported voices: {', '.join(sorted(SUPPORTED_VOICES))}"
            )

        self._opts = GnaniTTSOptions(
            api_key=self._api_key,
            voice=voice,
            model=model,
            sample_rate=sample_rate,
            encoding=encoding,
            container=container,
            num_channels=num_channels,
            bitrate=bitrate,
            base_url=base_url,
            synthesize_method=synthesize_method,
        )
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
        voice: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        _check_deprecated_tts_args(kwargs)

        if voice is not None:
            if voice not in SUPPORTED_VOICES:
                raise ValueError(
                    f"Voice '{voice}' not supported. "
                    f"Supported voices: {', '.join(sorted(SUPPORTED_VOICES))}"
                )
            self._opts.voice = voice
        if model is not None:
            self._opts.model = model

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
    return {
        "text": text,
        "voice": opts.voice,
        "model": opts.model,
        "audio_config": audio_config,
    }


def _build_headers(opts: GnaniTTSOptions) -> dict[str, str]:
    return {
        "X-API-Key-ID": opts.api_key,
        "Content-Type": "application/json",
    }


def _mime_type(opts: GnaniTTSOptions) -> str:
    if opts.container == "raw":
        return "audio/pcm"
    return f"audio/{opts.container}"


def _strip_wav_header(data: bytes) -> bytes:
    """Strip the RIFF/WAV header if present, returning only PCM samples."""
    if len(data) > _WAV_HEADER_SIZE and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return data[_WAV_HEADER_SIZE:]
    return data


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

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
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

    Each SSE chunk decodes to a complete WAV file. This class strips per-chunk
    WAV headers and emits only raw PCM so the LiveKit pipeline receives a
    single contiguous audio stream.
    """

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
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
                    request_id=utils.shortuuid(),
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/pcm",
                )

                data_buf = ""
                while True:
                    line_bytes = await res.content.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode("utf-8").rstrip("\r\n")

                    if not line:
                        if not data_buf:
                            continue
                        try:
                            payload = json.loads(data_buf)
                        except json.JSONDecodeError:
                            data_buf = ""
                            continue
                        data_buf = ""

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
                                output_emitter.push(_strip_wav_header(base64.b64decode(audio_b64)))
                            break

                        audio_b64 = payload.get("audio", "")
                        if audio_b64:
                            output_emitter.push(_strip_wav_header(base64.b64decode(audio_b64)))
                        continue

                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        continue
                    if line.startswith("id:") or line.startswith("retry:"):
                        continue
                    if line.startswith("data:"):
                        data_buf += line[5:].strip()
                    else:
                        data_buf += line.strip()

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
    ``synthesize()`` can use it when ``synthesize_method="websocket"``.
    Each received chunk's WAV header is stripped; only raw PCM is emitted.
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
                    request_id=utils.shortuuid(),
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/pcm",
                )

                async for msg in ws:
                    if isinstance(msg, bytes):
                        output_emitter.push(_strip_wav_header(msg))
                        continue

                    payload = json.loads(msg)
                    msg_type = payload.get("type", "")

                    if msg_type == "audio":
                        inner = payload.get("data", {})
                        audio_b64 = inner.get("audio", "")
                        if audio_b64:
                            output_emitter.push(_strip_wav_header(base64.b64decode(audio_b64)))

                    elif msg_type == "complete":
                        inner = payload.get("data")
                        if inner is not None:
                            audio_b64 = inner.get("audio", "")
                            if audio_b64:
                                output_emitter.push(_strip_wav_header(base64.b64decode(audio_b64)))
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
            mime_type="audio/pcm",
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
                        output_emitter.push(_strip_wav_header(msg))
                        continue

                    payload = json.loads(msg)
                    msg_type = payload.get("type", "")

                    if msg_type == "audio":
                        inner = payload.get("data", {})
                        audio_b64 = inner.get("audio", "")
                        if audio_b64:
                            output_emitter.push(_strip_wav_header(base64.b64decode(audio_b64)))

                    elif msg_type == "complete":
                        inner = payload.get("data")
                        if inner is not None:
                            audio_b64 = inner.get("audio", "")
                            if audio_b64:
                                output_emitter.push(_strip_wav_header(base64.b64decode(audio_b64)))
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

        output_emitter.end_segment()
