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
import struct
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
from .models import (
    DEFAULT_MODEL,
    GnaniTTSBitrates,
    GnaniTTSContainers,
    GnaniTTSEncodings,
    GnaniTTSLanguages,
    GnaniTTSModels,
    GnaniTTSVoices,
)
from .utils import ws_header_kwargs as _ws_header_kwargs

GNANI_TTS_BASE_URL = "https://api.vachana.ai"

GnaniTTSSynthesizeMethod = Literal["rest", "sse", "websocket"]

SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 44100)

_WAV_HEADER_SIZE = 44
# Match RoomIO _ParticipantAudioOutput target frame size (sample_rate // 20).
_STREAM_FRAME_MS = 50


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


def _strip_wav_header(data: bytes) -> bytes:
    """Strip a RIFF/WAV container if present, returning only PCM samples.

    Gnani streaming sends the WAV header as the first chunk (often with zero
    PCM bytes) and then raw PCM continuations without per-chunk headers. A
    header-only first chunk must not be emitted as audio.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data

    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        if chunk_id == b"data":
            data_start = offset + 8
            return data[data_start : data_start + chunk_size]
        offset += 8 + chunk_size

    # No `data` sub-chunk could be located (truncated or non-standard header).
    # Fall back to stripping a standard fixed-size WAV header so the PCM that
    # follows it is preserved instead of dropped.
    if len(data) <= _WAV_HEADER_SIZE:
        return b""
    return data[_WAV_HEADER_SIZE:]


class _Pcm16Aligner:
    """Ensure emitted PCM chunks contain whole 16-bit samples."""

    def __init__(self) -> None:
        self._remainder = b""

    def reset(self) -> None:
        self._remainder = b""

    def align(self, audio: bytes) -> bytes:
        audio = self._remainder + audio
        aligned_len = len(audio) - (len(audio) % 2)
        self._remainder = audio[aligned_len:]
        return audio[:aligned_len]


class _TtsPcmProcessor:
    """Strip WAV containers and align PCM for one streaming utterance."""

    def __init__(self) -> None:
        self._aligner = _Pcm16Aligner()
        self._wav_pending = b""

    def reset(self) -> None:
        self._aligner.reset()
        self._wav_pending = b""

    def process(self, audio: bytes) -> bytes:
        if self._wav_pending or (len(audio) >= 4 and audio[:4] == b"RIFF"):
            buf = self._wav_pending + audio
            self._wav_pending = b""

            if buf[:4] == b"RIFF" and len(buf) < _WAV_HEADER_SIZE:
                self._wav_pending = buf
                return b""

            pcm = _strip_wav_header(buf)
            return self._aligner.align(pcm)

        return self._aligner.align(audio)


class _PcmCoalescer:
    """Buffer PCM and push in stable frame-sized blocks matching AudioEmitter."""

    def __init__(
        self,
        *,
        sample_rate: int,
        num_channels: int,
        sample_width: int = 2,
        frame_ms: int = _STREAM_FRAME_MS,
    ) -> None:
        self._target_bytes = sample_rate * num_channels * sample_width * frame_ms // 1000
        self._buffer = bytearray()

    def push(self, pcm: bytes, output_emitter: tts.AudioEmitter) -> None:
        if not pcm:
            return
        self._buffer.extend(pcm)
        while len(self._buffer) >= self._target_bytes:
            output_emitter.push(bytes(self._buffer[: self._target_bytes]))
            del self._buffer[: self._target_bytes]

    def flush(self, output_emitter: tts.AudioEmitter) -> None:
        if self._buffer:
            output_emitter.push(bytes(self._buffer))
            self._buffer.clear()


class _StreamingAudioPusher:
    """Strip/align API audio, coalesce to steady frames, then push to AudioEmitter.

    Other LiveKit TTS plugins (Deepgram, Cartesia) push raw PCM directly in
    larger chunks. Gnani's API delivers many small bursts; coalescing here
    avoids AudioEmitter flush_if_delayed resets that cause live volume pumping.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        num_channels: int,
    ) -> None:
        self._processor = _TtsPcmProcessor()
        self._coalescer = _PcmCoalescer(sample_rate=sample_rate, num_channels=num_channels)

    def push(self, output_emitter: tts.AudioEmitter, audio: bytes) -> None:
        pcm = self._processor.process(audio)
        if pcm:
            self._coalescer.push(pcm, output_emitter)

    def finalize(self, output_emitter: tts.AudioEmitter) -> None:
        self._coalescer.flush(output_emitter)


def _process_and_push(
    pusher: _StreamingAudioPusher,
    output_emitter: tts.AudioEmitter,
    audio: bytes,
) -> None:
    pusher.push(output_emitter, audio)


@dataclass
class GnaniTTSOptions:
    api_key: str
    voice: str = "Pranav"
    model: str = DEFAULT_MODEL
    language: str | None = None
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
        voice: Voice to use for synthesis
            (see https://docs.gnani.ai/api/TTS/tts-sse#available-voices).
        model: TTS model name (default: timbre-v2.0; also: timbre-v2.5).
        language: BCP-47 language code for timbre-v2.5 only (e.g. "hi-IN").
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
        voice: GnaniTTSVoices | str = "Pranav",
        model: GnaniTTSModels | str = DEFAULT_MODEL,
        language: GnaniTTSLanguages | str | None = None,
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

        self._opts = GnaniTTSOptions(
            api_key=self._api_key,
            voice=voice,
            model=model,
            language=language,
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
        voice: GnaniTTSVoices | str | None = None,
        model: GnaniTTSModels | str | None = None,
        language: GnaniTTSLanguages | str | None = None,
        **kwargs: Any,
    ) -> None:
        _check_deprecated_tts_args(kwargs, caller="TTS.update_options")

        if voice is not None:
            self._opts.voice = voice
        if model is not None:
            self._opts.model = model
        if language is not None:
            self._opts.language = language

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
    if opts.model == "timbre-v2.5" and opts.language is not None:
        payload["language"] = opts.language
    return payload


def _streaming_payload_opts(opts: GnaniTTSOptions) -> GnaniTTSOptions:
    """Streaming transports request raw PCM — no per-chunk WAV containers."""
    if opts.container == "raw":
        return opts
    return replace(opts, container="raw")


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

    Each SSE chunk decodes to a complete WAV file. This class strips per-chunk
    WAV headers and emits only raw PCM so the LiveKit pipeline receives a
    single contiguous audio stream.
    """

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        pcm_processor = _StreamingAudioPusher(
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
        )
        stream_opts = _streaming_payload_opts(self._opts)
        try:
            async with self._tts._ensure_session().post(
                url=f"{self._opts.base_url}/api/v1/tts/sse",
                json=_build_payload(stream_opts, self._input_text),
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
                    mime_type="audio/pcm",
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
                            _process_and_push(
                                pcm_processor,
                                output_emitter,
                                base64.b64decode(audio_b64),
                            )
                        break

                    audio_b64 = payload.get("audio", "")
                    if audio_b64:
                        _process_and_push(
                            pcm_processor,
                            output_emitter,
                            base64.b64decode(audio_b64),
                        )

                pcm_processor.finalize(output_emitter)
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
    Requests raw PCM from the API and coalesces frames for LiveKit playback.
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
        pcm_processor = _StreamingAudioPusher(
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
        )
        stream_opts = _streaming_payload_opts(self._opts)
        try:
            ws_url = self._build_ws_url()
            async with websockets.connect(
                ws_url,
                **_ws_header_kwargs(_build_headers(self._opts)),
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
            ) as ws:
                request_body = _build_payload(stream_opts, self._input_text)
                await ws.send(json.dumps(request_body))

                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/pcm",
                )

                async for msg in ws:
                    if isinstance(msg, bytes):
                        _process_and_push(pcm_processor, output_emitter, msg)
                        continue

                    payload = json.loads(msg)
                    msg_type = payload.get("type", "")

                    if msg_type == "audio":
                        inner = payload.get("data", {})
                        audio_b64 = inner.get("audio", "")
                        if audio_b64:
                            _process_and_push(
                                pcm_processor,
                                output_emitter,
                                base64.b64decode(audio_b64),
                            )

                    elif msg_type == "complete":
                        inner = payload.get("data")
                        if inner is not None:
                            audio_b64 = inner.get("audio", "")
                            if audio_b64:
                                _process_and_push(
                                    pcm_processor,
                                    output_emitter,
                                    base64.b64decode(audio_b64),
                                )
                        break

                    elif msg_type == "error":
                        error_msg = payload.get("message", "Unknown error")
                        logger.error("Gnani TTS WS error: %s", error_msg)
                        raise APIStatusError(
                            message=f"Gnani TTS stream error: {error_msg}",
                            status_code=500,
                            body=error_msg,
                        )

                pcm_processor.finalize(output_emitter)
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

        pcm_processor = _StreamingAudioPusher(
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
        )
        stream_opts = _streaming_payload_opts(self._opts)
        try:
            ws_url = self._build_ws_url()
            async with websockets.connect(
                ws_url,
                **_ws_header_kwargs(_build_headers(self._opts)),
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
            ) as ws:
                request_body = _build_payload(stream_opts, full_text)
                await ws.send(json.dumps(request_body))

                self._mark_started()

                async for msg in ws:
                    if isinstance(msg, bytes):
                        _process_and_push(pcm_processor, output_emitter, msg)
                        continue

                    payload = json.loads(msg)
                    msg_type = payload.get("type", "")

                    if msg_type == "audio":
                        inner = payload.get("data", {})
                        audio_b64 = inner.get("audio", "")
                        if audio_b64:
                            _process_and_push(
                                pcm_processor,
                                output_emitter,
                                base64.b64decode(audio_b64),
                            )

                    elif msg_type == "complete":
                        inner = payload.get("data")
                        if inner is not None:
                            audio_b64 = inner.get("audio", "")
                            if audio_b64:
                                _process_and_push(
                                    pcm_processor,
                                    output_emitter,
                                    base64.b64decode(audio_b64),
                                )
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

        # Only flush/close the segment on the success path. On error, flushing
        # the coalescer's sub-frame remainder would push partial audio and set
        # pushed_duration > 0.0, which makes the base retry logic skip an
        # otherwise-retryable transient failure. The per-attempt output_emitter
        # is discarded on retry, so end_segment() is unnecessary on error.
        pcm_processor.finalize(output_emitter)
        output_emitter.flush()
        output_emitter.end_segment()


__all__ = ["TTS", "SynthesizeStream"]
