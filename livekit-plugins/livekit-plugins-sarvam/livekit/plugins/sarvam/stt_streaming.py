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

from __future__ import annotations

import asyncio
import json
import os
import platform
import time
import weakref
from dataclasses import dataclass, replace
from typing import Any, Literal, cast
from urllib.parse import urlencode

import aiohttp
import numpy as np

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    __version__ as livekit_version,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit.agents.utils.misc import is_given

from ._utils import PeriodicCollector
from .log import logger
from .stt import _looks_like_error_text

USER_AGENT = f"Livekit/{livekit_version} Python/{platform.python_version()}"

SARVAM_STT_REALTIME_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
REALTIME_MODEL = "saaras:v3-realtime"

RealtimeStreamType = Literal["fast", "balanced", "simulated"]
RealtimeEndpointing = Literal["vad", "manual"]
RealtimeEncoding = Literal["linear16", "linear32", "mulaw", "alaw"]
RealtimeMode = Literal["transcribe", "translate", "verbatim", "translit", "codemix"]

SUPPORTED_SAMPLE_RATES = {8000, 16000}
SUPPORTED_STREAM_TYPES = {"fast", "balanced", "simulated"}
SUPPORTED_ENDPOINTING = {"vad", "manual"}
SUPPORTED_ENCODINGS = {"linear16", "linear32", "mulaw", "alaw"}
SUPPORTED_MODES = {"transcribe", "translate", "verbatim", "translit", "codemix"}
# How much audio the client buffers before writing a frame. `stream_type` is a
# server-side latency profile (how often the server flushes to produce a partial),
# not a send cadence, and the contract sets no client chunk size. Matches the
# non-realtime Sarvam websocket plugin so audio reaches server VAD promptly.
AUDIO_CHUNK_MS = 50


def _encode_pcm_for_wire(encoding: RealtimeEncoding | str, pcm: bytes) -> bytes:
    """Encode little-endian signed 16-bit PCM for Sarvam's realtime wire format."""
    if len(pcm) % 2:
        raise ValueError("PCM data must contain whole 16-bit samples")
    if encoding == "linear16":
        return pcm

    samples = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
    if encoding == "linear32":
        return (samples.astype(np.int64) * (1 << 16)).astype("<i4").tobytes()
    if encoding == "mulaw":
        return _encode_mulaw(samples)
    if encoding == "alaw":
        return _encode_alaw(samples)
    raise ValueError(f"Unsupported realtime encoding: {encoding}")


def _encode_mulaw(samples: np.ndarray) -> bytes:
    """Encode signed linear PCM samples as 8-bit ITU-T G.711 mu-law."""
    magnitude = np.minimum(np.abs(samples), 32635) + 0x84
    exponent = np.clip(np.floor(np.log2(magnitude)).astype(np.int32) - 7, 0, 7)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    sign = np.where(samples < 0, 0x80, 0)
    return cast(bytes, (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes())


def _encode_alaw(samples: np.ndarray) -> bytes:
    """Encode signed linear PCM samples as 8-bit ITU-T G.711 A-law."""
    magnitude = np.minimum(np.abs(samples), 32767)
    exponent = np.clip(np.floor(np.log2(np.maximum(magnitude, 1))).astype(np.int32) - 7, 0, 7)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    encoded = np.where(magnitude < 256, magnitude >> 4, (exponent << 4) | mantissa)
    xor_mask = np.where(samples >= 0, 0xD5, 0x55)
    return bytes((encoded ^ xor_mask).astype(np.uint8).tobytes())


SUPPORTED_LANGUAGES = {
    "en-IN",
    "hi-IN",
    "bn-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "or-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
    "gu-IN",
    "as-IN",
    "ur-IN",
    "ne-IN",
    "kok-IN",
    "ks-IN",
    "sd-IN",
    "sa-IN",
    "sat-IN",
    "mni-IN",
    "brx-IN",
    "mai-IN",
    "doi-IN",
    "auto",
}


@dataclass
class RealtimeSTTOptions:
    """Resolved options for a single Sarvam realtime STT connection.

    Values are validated against the realtime contract in ``__post_init__``, which
    raises ``ValueError`` for anything the endpoint would reject. Some fields are
    negotiated at connection time and cannot change on a live stream; see
    :meth:`RealtimeSpeechStream.update_options`.
    """

    language: str
    api_key: str
    stream_type: RealtimeStreamType | str = "balanced"
    mode: RealtimeMode | str = "transcribe"
    endpointing: RealtimeEndpointing | str = "vad"
    encoding: RealtimeEncoding | str = "linear16"
    sample_rate: int = 16000
    model: str = REALTIME_MODEL
    base_url: str = SARVAM_STT_REALTIME_URL
    prompt: str | None = None
    return_timestamps: bool = False
    vad_sot_threshold: float | None = None
    vad_min_speech_ms: int | None = None
    vad_min_silence_ms: int | None = None
    vad_prefix_padding_ms: int | None = None

    def __post_init__(self) -> None:
        if self.model != REALTIME_MODEL:
            raise ValueError(f"model must be {REALTIME_MODEL}")
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"language {self.language} is not supported")
        if self.stream_type not in SUPPORTED_STREAM_TYPES:
            raise ValueError(
                f"stream_type must be one of {', '.join(sorted(SUPPORTED_STREAM_TYPES))}"
            )
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {', '.join(sorted(SUPPORTED_MODES))}")
        if self.endpointing not in SUPPORTED_ENDPOINTING:
            raise ValueError(
                f"endpointing must be one of {', '.join(sorted(SUPPORTED_ENDPOINTING))}"
            )
        if self.encoding not in SUPPORTED_ENCODINGS:
            raise ValueError(f"encoding must be one of {', '.join(sorted(SUPPORTED_ENCODINGS))}")
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {', '.join(str(r) for r in SUPPORTED_SAMPLE_RATES)}"
            )
        if self.vad_sot_threshold is not None and not 0.0 <= self.vad_sot_threshold <= 1.0:
            raise ValueError("vad_sot_threshold must be between 0.0 and 1.0")
        if self.vad_min_speech_ms is not None and self.vad_min_speech_ms < 0:
            raise ValueError("vad_min_speech_ms must be greater than or equal to 0")
        if self.vad_min_silence_ms is not None and self.vad_min_silence_ms < 0:
            raise ValueError("vad_min_silence_ms must be greater than or equal to 0")
        if self.vad_prefix_padding_ms is not None and self.vad_prefix_padding_ms < 0:
            raise ValueError("vad_prefix_padding_ms must be greater than or equal to 0")


def _build_realtime_ws_url(base_url: str, opts: RealtimeSTTOptions) -> str:
    params: dict[str, str] = {
        "language_code": opts.language,
        "stream_type": opts.stream_type,
        "endpointing": opts.endpointing,
        "encoding": opts.encoding,
        "sample_rate": str(opts.sample_rate),
        "model": opts.model,
    }

    params["mode"] = opts.mode
    params["return_timestamps"] = str(opts.return_timestamps).lower()
    if opts.prompt is not None:
        params["prompt"] = opts.prompt

    if opts.endpointing == "vad":
        if opts.vad_sot_threshold is not None:
            params["threshold"] = str(opts.vad_sot_threshold)
        if opts.vad_min_speech_ms is not None:
            params["min_speech_duration_ms"] = str(opts.vad_min_speech_ms)
        if opts.vad_min_silence_ms is not None:
            params["silence_duration_ms"] = str(opts.vad_min_silence_ms)
        if opts.vad_prefix_padding_ms is not None:
            params["prefix_padding_ms"] = str(opts.vad_prefix_padding_ms)

    return f"{base_url}?{urlencode(params)}"


class STTRealtime(stt.STT):
    """Speech-to-text using Sarvam's realtime WebSocket endpoint (``saaras:v3-realtime``).

    This endpoint streams interim and final transcripts over a single
    WebSocket connection and supports either server-side VAD or
    client-driven (manual) turn boundaries.
    """

    def __init__(
        self,
        *,
        language: str = "en-IN",
        stream_type: RealtimeStreamType | str = "balanced",
        mode: RealtimeMode | str = "transcribe",
        endpointing: RealtimeEndpointing | str = "vad",
        encoding: RealtimeEncoding | str = "linear16",
        sample_rate: int = 16000,
        prompt: str | None = None,
        return_timestamps: bool = False,
        api_key: str | None = None,
        base_url: str = SARVAM_STT_REALTIME_URL,
        http_session: aiohttp.ClientSession | None = None,
        vad_sot_threshold: float | None = None,
        vad_min_speech_ms: int | None = None,
        vad_min_silence_ms: int | None = None,
        vad_prefix_padding_ms: int | None = None,
    ) -> None:
        """Create a Sarvam realtime STT instance.

        Args:
            language: BCP-47 language code, or ``auto`` for adaptive language identification.
            stream_type: Latency profile: ``fast``, ``balanced``, or ``simulated``.
            mode: Task applied to finals: ``transcribe``, ``translate``, ``verbatim``,
                ``translit``, or ``codemix``.
            endpointing: ``vad`` for server-side turn detection, or ``manual`` when the
                caller delimits turns by flushing the stream.
            encoding: Wire encoding: ``linear16``, ``linear32``, ``mulaw``, or ``alaw``.
            sample_rate: Audio sample rate in Hz; ``8000`` or ``16000``.
            prompt: Optional context or terminology hint used to bias decoding.
            return_timestamps: Whether finals should carry segment-level start and end times.
            api_key: Sarvam API key. Falls back to the ``SARVAM_API_KEY`` environment variable.
            base_url: WebSocket URL of the realtime endpoint.
            http_session: Optional aiohttp session to reuse for the connection.
            vad_sot_threshold: VAD activation threshold (``vad`` endpointing only).
            vad_min_speech_ms: Minimum speech duration in ms (``vad`` endpointing only).
            vad_min_silence_ms: End-of-turn silence in ms (``vad`` endpointing only).
            vad_prefix_padding_ms: Audio retained before speech onset in ms
                (``vad`` endpointing only).

        Raises:
            ValueError: If no API key is provided or found in the environment, or if an
                option falls outside the values the endpoint accepts.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript=False,
                offline_recognize=False,
            )
        )

        api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not api_key:
            raise ValueError(
                "Sarvam API key is required. "
                "Provide it directly or set SARVAM_API_KEY environment variable."
            )

        self._opts = RealtimeSTTOptions(
            language=language,
            api_key=api_key,
            stream_type=stream_type,
            mode=mode,
            endpointing=endpointing,
            encoding=encoding,
            sample_rate=sample_rate,
            base_url=base_url,
            prompt=prompt,
            return_timestamps=return_timestamps,
            vad_sot_threshold=vad_sot_threshold,
            vad_min_speech_ms=vad_min_speech_ms,
            vad_min_silence_ms=vad_min_silence_ms,
            vad_prefix_padding_ms=vad_prefix_padding_ms,
        )
        self._session = http_session
        self._owns_session = http_session is None
        self._streams = weakref.WeakSet[RealtimeSpeechStream]()

    @property
    def model(self) -> str:
        """Name of the Sarvam realtime model backing this instance."""
        return REALTIME_MODEL

    @property
    def provider(self) -> str:
        """Name of the speech-to-text provider."""
        return "Sarvam"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            try:
                self._session = utils.http_context.http_session()
                self._owns_session = False
            except RuntimeError:
                self._session = aiohttp.ClientSession()
                self._owns_session = True
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        del buffer, language, conn_options
        raise NotImplementedError("Sarvam realtime STT only supports streaming")

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        stream_type: NotGivenOr[RealtimeStreamType | str] = NOT_GIVEN,
        mode: NotGivenOr[RealtimeMode | str] = NOT_GIVEN,
        endpointing: NotGivenOr[RealtimeEndpointing | str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        prompt: NotGivenOr[str | None] = NOT_GIVEN,
        return_timestamps: NotGivenOr[bool] = NOT_GIVEN,
        vad_sot_threshold: NotGivenOr[float | None] = NOT_GIVEN,
        vad_min_speech_ms: NotGivenOr[int | None] = NOT_GIVEN,
        vad_min_silence_ms: NotGivenOr[int | None] = NOT_GIVEN,
        vad_prefix_padding_ms: NotGivenOr[int | None] = NOT_GIVEN,
    ) -> None:
        """Update options for this instance and every stream it created.

        Options that Sarvam only accepts at connection time (``sample_rate``,
        ``return_timestamps``, and ``vad_prefix_padding_ms``) take effect on
        newly created streams only.
        The remaining options are sent to active streams as an in-band
        ``config.update``, and the boundary-gated ones apply from the next
        utterance boundary.

        Args:
            language: BCP-47 language code, or ``auto`` for adaptive language identification.
            stream_type: Latency profile: ``fast``, ``balanced``, or ``simulated``.
            mode: Task applied to finals.
            endpointing: ``vad`` for server-side turn detection, or ``manual``.
            sample_rate: Audio sample rate in Hz; applies to new streams only.
            prompt: Context or terminology hint; ``None`` clears it.
            return_timestamps: Segment-level timestamps; applies to new streams only.
            vad_sot_threshold: VAD activation threshold (``vad`` endpointing only).
            vad_min_speech_ms: Minimum speech duration in ms (``vad`` endpointing only).
            vad_min_silence_ms: End-of-turn silence in ms (``vad`` endpointing only).
            vad_prefix_padding_ms: Audio retained before speech onset; new streams only.

        Raises:
            ValueError: If an option falls outside the values the endpoint accepts.
        """
        opts = RealtimeSTTOptions(
            language=language if is_given(language) else self._opts.language,
            api_key=self._opts.api_key,
            stream_type=stream_type if is_given(stream_type) else self._opts.stream_type,
            mode=mode if is_given(mode) else self._opts.mode,
            endpointing=endpointing if is_given(endpointing) else self._opts.endpointing,
            encoding=self._opts.encoding,
            sample_rate=sample_rate if is_given(sample_rate) else self._opts.sample_rate,
            base_url=self._opts.base_url,
            prompt=prompt if is_given(prompt) else self._opts.prompt,
            return_timestamps=return_timestamps
            if is_given(return_timestamps)
            else self._opts.return_timestamps,
            vad_sot_threshold=vad_sot_threshold
            if is_given(vad_sot_threshold)
            else self._opts.vad_sot_threshold,
            vad_min_speech_ms=vad_min_speech_ms
            if is_given(vad_min_speech_ms)
            else self._opts.vad_min_speech_ms,
            vad_min_silence_ms=vad_min_silence_ms
            if is_given(vad_min_silence_ms)
            else self._opts.vad_min_silence_ms,
            vad_prefix_padding_ms=vad_prefix_padding_ms
            if is_given(vad_prefix_padding_ms)
            else self._opts.vad_prefix_padding_ms,
        )
        self._opts = opts
        # Forward the given fields only, so a stream created with a per-stream
        # override (e.g. `stream(language=...)`) keeps it through unrelated updates.
        for stream in self._streams:
            stream.update_options(
                language=language,
                stream_type=stream_type,
                mode=mode,
                endpointing=endpointing,
                sample_rate=sample_rate,
                prompt=prompt,
                return_timestamps=return_timestamps,
                vad_sot_threshold=vad_sot_threshold,
                vad_min_speech_ms=vad_min_speech_ms,
                vad_min_silence_ms=vad_min_silence_ms,
                vad_prefix_padding_ms=vad_prefix_padding_ms,
            )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RealtimeSpeechStream:
        """Create a new realtime speech stream.

        Args:
            language: Overrides the instance language for this stream only.
            conn_options: Connection options. ``max_retry`` is forced to ``0`` because this
                endpoint bills per connection and must not silently reconnect.

        Returns:
            A stream that accepts audio frames and yields speech events.
        """
        conn_options = replace(conn_options, max_retry=0)
        opts = RealtimeSTTOptions(
            language=language if is_given(language) else self._opts.language,
            api_key=self._opts.api_key,
            stream_type=self._opts.stream_type,
            mode=self._opts.mode,
            endpointing=self._opts.endpointing,
            encoding=self._opts.encoding,
            sample_rate=self._opts.sample_rate,
            base_url=self._opts.base_url,
            prompt=self._opts.prompt,
            return_timestamps=self._opts.return_timestamps,
            vad_sot_threshold=self._opts.vad_sot_threshold,
            vad_min_speech_ms=self._opts.vad_min_speech_ms,
            vad_min_silence_ms=self._opts.vad_min_silence_ms,
            vad_prefix_padding_ms=self._opts.vad_prefix_padding_ms,
        )
        stream = RealtimeSpeechStream(
            stt=self,
            opts=opts,
            conn_options=conn_options,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close every stream created by this instance and any owned HTTP session."""
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()


class RealtimeSpeechStream(stt.SpeechStream):
    """A single WebSocket session against Sarvam's realtime STT endpoint.

    Audio pushed into the stream is forwarded in the configured wire encoding, and
    the events Sarvam returns are translated into LiveKit speech events. Audio
    duration is reported incrementally while the session runs and reconciled
    against the server's authoritative total when the session ends.
    """

    def __init__(
        self,
        *,
        stt: STTRealtime,
        opts: RealtimeSTTOptions,
        conn_options: APIConnectOptions,
        http_session: aiohttp.ClientSession,
    ) -> None:
        """Create a realtime speech stream.

        Args:
            stt: The parent instance that created this stream.
            opts: Resolved options for this connection.
            conn_options: Connection options for this stream.
            http_session: aiohttp session used to open the WebSocket.
        """
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._session = http_session
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._request_id = ""
        self._session_id = ""
        self._resolved_config: dict[str, Any] | None = None
        self._session_ended = False
        self._utterance_idx: int | None = None
        self._utterance_in_progress = False
        self._active_endpointing = opts.endpointing
        self._pending_endpointing: RealtimeEndpointing | str | None = None
        self._endpointing_update_acknowledged = False
        self._endpointing_update_sent = False
        self._pending_config_update: dict[str, Any] | None = None
        self._manual_speech_started = False
        self._flush_observed = False
        self._pending_final_data: dict[str, Any] | None = None
        self._utterance_start_audio_pos = 0.0
        self._utterance_speech_end_audio_pos: float | None = None
        self._utterance_speech_end_wall: float | None = None
        self._final_received_for_utterance = False
        self._eos_emitted_for_utterance = False
        self._stream_started_at = time.time()
        self._audio_position = 0.0
        self._local_audio_duration = 0.0
        self._total_reported_audio_duration = 0.0
        self._server_audio_duration_reported = False
        self._audio_duration_collector = PeriodicCollector(
            callback=self._on_audio_duration_report,
            duration=5.0,
        )
        self._logger = logger

    @property
    def resolved_config(self) -> dict[str, Any] | None:
        """Return the configuration resolved by Sarvam for this connection."""
        return dict(self._resolved_config) if self._resolved_config is not None else None

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        stream_type: NotGivenOr[RealtimeStreamType | str] = NOT_GIVEN,
        mode: NotGivenOr[RealtimeMode | str] = NOT_GIVEN,
        endpointing: NotGivenOr[RealtimeEndpointing | str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        prompt: NotGivenOr[str | None] = NOT_GIVEN,
        return_timestamps: NotGivenOr[bool] = NOT_GIVEN,
        vad_sot_threshold: NotGivenOr[float | None] = NOT_GIVEN,
        vad_min_speech_ms: NotGivenOr[int | None] = NOT_GIVEN,
        vad_min_silence_ms: NotGivenOr[int | None] = NOT_GIVEN,
        vad_prefix_padding_ms: NotGivenOr[int | None] = NOT_GIVEN,
    ) -> None:
        """Apply an option change to this live connection.

        Only the options explicitly passed here are changed, so per-stream overrides
        such as a ``language`` given to :meth:`STTRealtime.stream` survive an unrelated
        update. Connection-time options are retained at their current values and a
        warning is logged, since changing them would desynchronize the
        already-negotiated session. Every other change is queued as an in-band
        ``config.update`` sent before the next audio frame.

        Args:
            language: BCP-47 language code, or ``auto`` for adaptive identification.
            stream_type: Latency profile: ``fast``, ``balanced``, or ``simulated``.
            mode: Task applied to finals.
            endpointing: ``vad`` for server-side turn detection, or ``manual``.
            sample_rate: Audio sample rate in Hz; retained on a live stream.
            prompt: Context or terminology hint; ``None`` clears it.
            return_timestamps: Segment-level timestamps; retained on a live stream.
            vad_sot_threshold: VAD activation threshold (``vad`` endpointing only).
            vad_min_speech_ms: Minimum speech duration in ms (``vad`` endpointing only).
            vad_min_silence_ms: End-of-turn silence in ms (``vad`` endpointing only).
            vad_prefix_padding_ms: Speech-onset padding; retained on a live stream.

        Raises:
            ValueError: If an option falls outside the values the endpoint accepts.
        """
        previous_opts = self._opts
        requested: dict[str, Any] = {}
        if is_given(language):
            requested["language"] = language
        if is_given(stream_type):
            requested["stream_type"] = stream_type
        if is_given(mode):
            requested["mode"] = mode
        if is_given(endpointing):
            requested["endpointing"] = endpointing
        if is_given(sample_rate):
            requested["sample_rate"] = sample_rate
        if is_given(prompt):
            requested["prompt"] = prompt
        if is_given(return_timestamps):
            requested["return_timestamps"] = return_timestamps
        if is_given(vad_sot_threshold):
            requested["vad_sot_threshold"] = vad_sot_threshold
        if is_given(vad_min_speech_ms):
            requested["vad_min_speech_ms"] = vad_min_speech_ms
        if is_given(vad_min_silence_ms):
            requested["vad_min_silence_ms"] = vad_min_silence_ms
        if is_given(vad_prefix_padding_ms):
            requested["vad_prefix_padding_ms"] = vad_prefix_padding_ms

        if not requested:
            return

        opts = replace(previous_opts, **requested)
        connection_only_options: list[str] = []
        if opts.sample_rate != previous_opts.sample_rate:
            connection_only_options.append("sample_rate")
            opts = replace(opts, sample_rate=previous_opts.sample_rate)
        if opts.return_timestamps != previous_opts.return_timestamps:
            connection_only_options.append("return_timestamps")
            opts = replace(opts, return_timestamps=previous_opts.return_timestamps)
        if opts.vad_prefix_padding_ms != previous_opts.vad_prefix_padding_ms:
            connection_only_options.append("vad_prefix_padding_ms")
            opts = replace(opts, vad_prefix_padding_ms=previous_opts.vad_prefix_padding_ms)
        if connection_only_options:
            self._logger.warning(
                "Sarvam realtime STT connection-only option updates only apply to new streams",
                extra={
                    **self._build_log_context(),
                    "options": connection_only_options,
                },
            )

        self._opts = opts
        if opts.endpointing != previous_opts.endpointing:
            if opts.endpointing == "manual" and not self._flush_observed:
                self._logger.warning(
                    "Sarvam realtime STT switched to manual endpointing without an external VAD; "
                    "turns will not be delimited unless the agent framework flushes the stream. "
                    "Configure a VAD on the AgentSession to receive end-of-turn boundaries.",
                    extra=self._build_log_context(),
                )
            self._pending_endpointing = opts.endpointing
            self._endpointing_update_acknowledged = False
            self._endpointing_update_sent = False

        update = self._config_update_payload(previous_opts, opts)
        if update is not None:
            if self._pending_config_update is None:
                self._pending_config_update = update
            else:
                self._pending_config_update.update(update)

    @staticmethod
    def _config_update_payload(
        previous: RealtimeSTTOptions,
        current: RealtimeSTTOptions,
    ) -> dict[str, Any] | None:
        payload: dict[str, Any] = {"event": "config.update"}
        values = (
            ("language_code", previous.language, current.language),
            ("stream_type", previous.stream_type, current.stream_type),
            ("mode", previous.mode, current.mode),
            ("prompt", previous.prompt, current.prompt),
            ("endpointing", previous.endpointing, current.endpointing),
            ("threshold", previous.vad_sot_threshold, current.vad_sot_threshold),
            (
                "min_speech_duration_ms",
                previous.vad_min_speech_ms,
                current.vad_min_speech_ms,
            ),
            (
                "silence_duration_ms",
                previous.vad_min_silence_ms,
                current.vad_min_silence_ms,
            ),
        )
        for key, old_value, new_value in values:
            if old_value != new_value:
                if key == "prompt" and new_value is None:
                    new_value = ""
                payload[key] = new_value
        return payload if len(payload) > 1 else None

    @staticmethod
    def _ack_lists_endpointing(data: dict[str, Any]) -> bool | None:
        """Whether a ``config.updated`` acknowledges an endpointing change.

        The server echoes each applied key as ``"<key>=<value>"``, optionally
        suffixed when the change is deferred to the next utterance boundary.
        Returns ``None`` when ``applied`` is missing or not a list of strings, so
        the caller can fall back instead of stalling on an unexpected shape.
        """
        applied = data.get("applied")
        if not isinstance(applied, list) or not all(isinstance(e, str) for e in applied):
            return None
        return any(e.split("=", 1)[0].strip() == "endpointing" for e in applied)

    def _handle_config_updated(self, data: dict[str, Any]) -> None:
        if self._pending_endpointing is None:
            return
        if not self._endpointing_update_sent:
            # This acknowledges an earlier update; ours is still queued locally and
            # the server is still in the old mode.
            return
        if self._ack_lists_endpointing(data) is False:
            return

        self._endpointing_update_acknowledged = True
        self._apply_pending_endpointing()

    def _apply_pending_endpointing(self) -> None:
        if (
            self._pending_endpointing is not None
            and self._endpointing_update_acknowledged
            and not self._utterance_in_progress
        ):
            self._active_endpointing = self._pending_endpointing
            self._pending_endpointing = None
            self._endpointing_update_acknowledged = False
            self._endpointing_update_sent = False

    def _complete_utterance(self) -> None:
        self._utterance_in_progress = False
        self._apply_pending_endpointing()

    def _build_log_context(self) -> dict[str, Any]:
        return {
            "request_id": self._request_id,
            "session_id": self._session_id,
            "model": self._opts.model,
            "language": self._opts.language,
            "stream_type": self._opts.stream_type,
            "endpointing": self._opts.endpointing,
            "utterance_idx": self._utterance_idx,
        }

    @staticmethod
    def _extract_request_id(data: dict[str, Any]) -> str | None:
        request_id = data.get("request_id")
        if request_id is None:
            nested = data.get("data")
            if isinstance(nested, dict):
                request_id = nested.get("request_id")
            metadata = data.get("metadata")
            if request_id is None and isinstance(metadata, dict):
                request_id = metadata.get("request_id")
        if isinstance(request_id, str) and request_id:
            return request_id
        return None

    @staticmethod
    def _extract_session_id(data: dict[str, Any]) -> str | None:
        session_id = data.get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
        return None

    def _capture_server_ids(self, data: dict[str, Any]) -> None:
        session_id = self._extract_session_id(data)
        if session_id is not None:
            self._session_id = session_id

        if not self._request_id:
            request_id = self._extract_request_id(data)
            if request_id is not None:
                self._request_id = request_id

    async def aclose(self) -> None:
        """Close the connection, reporting any audio duration not yet billed.

        Agents normally end a stream here rather than waiting for ``session.end``, so
        the pending duration is flushed before the base class cancels the tasks that
        deliver usage metrics.
        """
        try:
            if not self._event_ch.closed:
                self._emit_local_usage_fallback()
                # Give the metrics monitor a chance to consume the usage event before
                # super().aclose() cancels it.
                await asyncio.sleep(0)
            if self._ws and not self._ws.closed:
                await self._ws.close()
        finally:
            self._ws = None
            await super().aclose()

    async def _run(self) -> None:
        # A single connection attempt: this endpoint bills per connection, so the
        # stream never reconnects on its own (`stream()` also forces max_retry=0).
        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            ws = await self._connect_ws()
            self._ws = ws
            tasks = [
                asyncio.create_task(self._process_audio(ws)),
                asyncio.create_task(self._process_messages(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Timed out connecting to Sarvam realtime STT") from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=self._request_id or None,
                body=e.message,
            ) from e
        except aiohttp.ClientConnectorError as e:
            raise APIConnectionError("failed to connect to Sarvam realtime STT") from e
        finally:
            if ws is not None:
                await ws.close()
            self._ws = None

    def _reset_utterance_state(self) -> None:
        self._utterance_idx = None
        self._pending_final_data = None
        self._utterance_start_audio_pos = self._audio_position
        self._utterance_speech_end_audio_pos = None
        self._utterance_speech_end_wall = None
        self._final_received_for_utterance = False
        self._eos_emitted_for_utterance = False

    def _begin_manual_utterance(self) -> None:
        """Open a client-delimited turn.

        Sarvam emits no ``vad.speech_start`` under manual endpointing, so the client
        boundary is what starts an utterance. Resetting here keeps the per-utterance
        flags and timings from leaking across turns, including after an ``endpointing``
        switch from ``vad`` to ``manual``.
        """
        self._reset_utterance_state()
        self._utterance_in_progress = True
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.START_OF_SPEECH,
                request_id=self._request_id,
            )
        )

    def _end_manual_utterance(self) -> None:
        """Close a client-delimited turn and anchor its speech-end position."""
        self._utterance_speech_end_audio_pos = self._audio_position
        self._utterance_speech_end_wall = time.time()
        self._emit_end_of_speech()
        self._complete_utterance()

    async def _safe_send_str(
        self,
        ws: Any,
        payload: dict[str, Any],
    ) -> bool:
        """Send a JSON control message, tolerating a peer that already closed.

        Returns:
            Whether the payload reached the socket.
        """
        if ws.closed:
            return False

        try:
            await ws.send_str(json.dumps(payload))
        except (aiohttp.ClientConnectionResetError, ConnectionError):
            self._logger.debug(
                "Sarvam realtime STT WebSocket closed before send completed",
                extra={**self._build_log_context(), "payload": payload},
            )
            return False
        return True

    async def _safe_send_bytes(self, ws: Any, payload: bytes) -> None:
        if ws.closed:
            return

        try:
            await ws.send_bytes(payload)
        except (aiohttp.ClientConnectionResetError, ConnectionError):
            self._logger.debug(
                "Sarvam realtime STT WebSocket closed before audio send completed",
                extra={**self._build_log_context(), "payload_bytes": len(payload)},
            )

    async def _send_pending_config_update(self, ws: Any) -> None:
        payload = self._pending_config_update
        self._pending_config_update = None
        if payload is None:
            return
        if await self._safe_send_str(ws, payload) and "endpointing" in payload:
            # Only now can a config.updated acknowledgement refer to our change.
            self._endpointing_update_sent = True

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        ws_url = _build_realtime_ws_url(self._opts.base_url, self._opts)
        headers = {
            "API-SUBSCRIPTION-KEY": self._opts.api_key,
            "User-Agent": USER_AGENT,
        }
        self._logger.debug(
            "Connecting to Sarvam realtime STT WebSocket", extra=self._build_log_context()
        )
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(ws_url, headers=headers, heartbeat=30.0),
                self._conn_options.timeout,
            )
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            self._logger.error(
                "Failed to connect to Sarvam realtime STT WebSocket",
                extra={**self._build_log_context(), "error": str(e), "url": ws_url},
                exc_info=True,
            )
            raise
        except aiohttp.ClientResponseError as e:
            self._logger.error(
                "Sarvam realtime STT WebSocket handshake failed",
                extra={
                    **self._build_log_context(),
                    "error": e.message,
                    "status_code": e.status,
                    "url": ws_url,
                },
                exc_info=True,
            )
            raise
        except Exception as e:
            self._logger.error(
                "Unexpected Sarvam realtime STT WebSocket connection error",
                extra={**self._build_log_context(), "error": str(e), "url": ws_url},
                exc_info=True,
            )
            raise APIConnectionError("failed to connect to Sarvam realtime STT") from e
        self._logger.debug(
            "Sarvam realtime STT WebSocket connected", extra=self._build_log_context()
        )
        return ws

    @utils.log_exceptions(logger=logger)
    async def _process_audio(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        samples_per_channel = max(int(self._opts.sample_rate * AUDIO_CHUNK_MS / 1000), 1)
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            samples_per_channel=samples_per_channel,
        )

        async for data in self._input_ch:
            # The server is done reading, so stop the pump instead of writing into a
            # socket whose reset would fail the whole stream (max_retry is forced to 0).
            if self._session_ended or ws.closed:
                break

            await self._send_pending_config_update(ws)
            frames: list[rtc.AudioFrame] = []
            if isinstance(data, rtc.AudioFrame):
                frames.extend(audio_bstream.write(data.data.tobytes()))
            if isinstance(data, self._FlushSentinel):
                self._flush_observed = True
                frames.extend(audio_bstream.flush())

            for frame in frames:
                if self._active_endpointing == "manual" and not self._manual_speech_started:
                    await self._safe_send_str(ws, {"event": "speech_start"})
                    self._manual_speech_started = True
                    self._begin_manual_utterance()

                self._audio_duration_collector.push(frame.duration)
                self._audio_position += frame.duration
                await self._safe_send_bytes(
                    ws, _encode_pcm_for_wire(self._opts.encoding, frame.data.tobytes())
                )

            if isinstance(data, self._FlushSentinel):
                self._audio_duration_collector.flush()
                if self._active_endpointing == "manual" and self._manual_speech_started:
                    await self._safe_send_str(ws, {"event": "speech_end"})
                    self._manual_speech_started = False
                    self._end_manual_utterance()

        self._emit_local_usage_fallback()
        if not self._session_ended:
            await self._safe_send_str(ws, {"event": "end"})

    @utils.log_exceptions(logger=logger)
    async def _process_messages(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            msg = await ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    await self._handle_message(json.loads(msg.data))
                except json.JSONDecodeError as e:
                    if _looks_like_error_text(msg.data):
                        self._logger.error(
                            "Sarvam realtime STT non-JSON error message",
                            extra={**self._build_log_context(), "raw_message": msg.data},
                        )
                        raise APIStatusError(
                            message=f"Sarvam realtime STT non-JSON error message: {msg.data}",
                            request_id=self._request_id or None,
                            body={"raw_message": msg.data},
                        ) from e
                    self._logger.warning(
                        "Invalid JSON received from Sarvam realtime STT",
                        extra={**self._build_log_context(), "raw_data": msg.data},
                    )
                    continue
                if self._session_ended:
                    break
            elif msg.type == aiohttp.WSMsgType.ERROR:
                self._logger.error(
                    "Sarvam realtime STT WebSocket error",
                    extra={**self._build_log_context(), "raw_message": msg.data},
                )
                raise APIConnectionError(f"Sarvam realtime STT WebSocket error: {msg.data}")
            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                close_code = ws.close_code if ws.close_code is not None else msg.data
                close_reason = msg.extra
                if self._session_ended and close_code in (1000, 1001, None):
                    self._flush_terminal_utterance()
                    self._emit_local_usage_fallback()
                    break
                if close_code in (1000, 1001, None) and not _looks_like_error_text(close_reason):
                    self._flush_terminal_utterance()
                    self._emit_local_usage_fallback()
                    break
                self._logger.error(
                    "Sarvam realtime STT WebSocket closed unexpectedly",
                    extra={
                        **self._build_log_context(),
                        "close_code": close_code,
                        "close_reason": close_reason,
                    },
                )
                raise self._status_error_from_close(close_code, close_reason)
            else:
                self._logger.debug(
                    "Unknown Sarvam realtime STT WebSocket message type",
                    extra={**self._build_log_context(), "message_type": str(msg.type)},
                )

    def _status_error_from_close(self, close_code: object, close_reason: object) -> APIStatusError:
        status_code = int(close_code) if isinstance(close_code, int) else -1
        retryable = close_code == 1013
        message = f"Sarvam realtime STT WebSocket closed unexpectedly: {close_reason}"
        if close_code == 1003:
            message = "Sarvam realtime STT authentication, quota, or rate limit error"
        elif close_code == 1008:
            message = "Sarvam realtime STT session timed out or exceeded the maximum duration"
        elif close_code == 1013:
            message = "Sarvam realtime STT backend temporarily unavailable"
        elif close_code == 4000:
            message = f"Sarvam realtime STT rejected the session: {close_reason}"

        return APIStatusError(
            message=message,
            status_code=status_code,
            request_id=self._request_id or None,
            body={
                "close_code": close_code,
                "close_reason": close_reason,
            },
            retryable=retryable,
        )

    async def _handle_message(self, data: dict[str, Any]) -> None:
        event = data.get("event")
        self._capture_server_ids(data)
        if event == "session.begin":
            config = data.get("config")
            self._resolved_config = dict(config) if isinstance(config, dict) else None
        self._log_stt_event(event, data)
        if event == "session.begin":
            return
        elif event == "vad.speech_start":
            self._reset_utterance_state()
            self._utterance_in_progress = True
            utterance_idx = data.get("utterance_idx")
            self._utterance_idx = utterance_idx if isinstance(utterance_idx, int) else None
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.START_OF_SPEECH,
                    request_id=self._request_id,
                )
            )
        elif event == "vad.speech_end":
            self._handle_speech_end()
        elif event == "transcript.partial":
            self._send_transcript_event(stt.SpeechEventType.INTERIM_TRANSCRIPT, data)
        elif event == "transcript.final":
            if self._active_endpointing == "vad":
                if self._is_valid_transcript(data):
                    self._pending_final_data = data
                    self._final_received_for_utterance = True
                    self._try_commit_utterance()
            elif self._send_transcript_event(stt.SpeechEventType.FINAL_TRANSCRIPT, data):
                self._final_received_for_utterance = True
                self._complete_utterance()
        elif event == "session.end":
            self._handle_session_end(data)
        elif event == "config.updated":
            self._handle_config_updated(data)
            return
        elif event == "error":
            self._handle_error_event(data)
        elif event == "pong":
            return
        else:
            self._logger.debug(
                "Unknown Sarvam realtime STT event",
                extra={**self._build_log_context(), "event": event, "data": data},
            )

    def _log_stt_event(self, event: object, data: dict[str, Any]) -> None:
        if event == "pong":
            return

        extra: dict[str, Any] = {
            **self._build_log_context(),
            "event": event,
            "utterance_idx": data.get("utterance_idx"),
        }
        if event in {"transcript.partial", "transcript.final"}:
            # Recognized speech is personal data, so only its length is safe for the
            # INFO record; the text itself stays in the opt-in DEBUG raw payload.
            text = data.get("text")
            if isinstance(text, str):
                extra["text_length"] = len(text)
            extra["language"] = data.get("language") or self._opts.language
            extra["confidence"] = data.get("language_confidence", data.get("confidence"))
        elif event == "vad.speech_start":
            extra["audio_position"] = self._audio_position
        elif event == "vad.speech_end":
            extra["audio_position"] = self._audio_position
        elif event == "session.begin":
            pass
        elif event == "session.end":
            extra["audio_duration_s"] = data.get("audio_duration_s")
        elif event == "config.updated":
            extra["applied"] = data.get("applied")
        elif event == "error":
            extra["error_code"] = data.get("code")
            extra["error_message"] = data.get("message")
            extra["status_code"] = data.get("status_code")
        else:
            return

        if event == "transcript.partial":
            self._logger.debug(
                "Sarvam realtime STT transcript.partial",
                extra={**extra, "raw_data": data},
            )
            return

        self._logger.info(f"Sarvam realtime STT {event}", extra=extra)
        self._logger.debug(
            "Sarvam realtime STT raw event",
            extra={**extra, "raw_data": data},
        )

    def _is_valid_transcript(self, data: dict[str, Any]) -> bool:
        text = data.get("text")
        # Whitespace carries no content, and emitting it would commit a user turn
        # with no words.
        return isinstance(text, str) and bool(text.strip())

    def _handle_speech_end(self) -> None:
        self._utterance_speech_end_audio_pos = self._audio_position
        self._utterance_speech_end_wall = time.time()

        if self._active_endpointing != "vad":
            self._emit_end_of_speech()
        elif not self._eos_emitted_for_utterance:
            self._emit_end_of_speech()
            if self._final_received_for_utterance:
                self._try_commit_utterance()

        # The server's speech end is the utterance boundary, so the turn is over even
        # when the final is empty or never arrives. Completing unconditionally is what
        # lets a boundary-gated endpointing change promote; leaving the utterance open
        # would strand the stream in the old mode with the server in the new one.
        self._complete_utterance()

    def _try_commit_utterance(self) -> None:
        if self._pending_final_data is None or self._utterance_speech_end_audio_pos is None:
            return

        committed_data = self._pending_final_data
        if self._send_transcript_event(
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            committed_data,
        ):
            self._logger.debug(
                "Sarvam realtime STT utterance committed",
                extra={
                    **self._build_log_context(),
                    "end_time": self._utterance_speech_end_audio_pos,
                    "speech_end_wall_time": self._utterance_speech_end_wall,
                },
            )
            if not self._eos_emitted_for_utterance:
                self._emit_end_of_speech()
            self._pending_final_data = None
            self._complete_utterance()

    def _flush_terminal_utterance(self) -> None:
        """Commit a buffered final transcript when the session ends mid-utterance.

        In VAD endpointing a ``transcript.final`` is held until ``vad.speech_end``
        supplies the speech-end position. When the input audio ends mid-utterance
        the server finalizes and closes without that event, so the speech-end
        position is anchored to the audio consumed so far instead of dropping the
        transcript. Safe to call more than once per session.
        """
        if self._pending_final_data is not None and self._utterance_speech_end_audio_pos is None:
            self._utterance_speech_end_audio_pos = self._audio_position
            if self._utterance_speech_end_wall is None:
                self._utterance_speech_end_wall = time.time()

        if not self._eos_emitted_for_utterance and self._pending_final_data is not None:
            self._emit_end_of_speech()

        self._try_commit_utterance()

    def _emit_end_of_speech(self) -> None:
        if self._eos_emitted_for_utterance:
            return

        # Emitted without alternatives so the agent pipeline treats it as a sentinel
        # it can hold and release with a concrete transcript. The speech-end timing
        # travels on the FINAL_TRANSCRIPT event instead.
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.END_OF_SPEECH,
                request_id=self._request_id,
            )
        )
        self._eos_emitted_for_utterance = True

    def _send_transcript_event(self, event_type: stt.SpeechEventType, data: dict[str, Any]) -> bool:
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            return False

        language = data.get("language") or self._opts.language
        # Recognition confidence only: `language_confidence` is a language-identification
        # score and stays in metadata. The endpoint sends no per-segment confidence
        # today, and an absent value falls back to 1.0 (as `_extract_confidence` in
        # stt.py does) so it isn't averaged downstream as "no confidence".
        # bool is a subclass of int, so exclude it explicitly.
        confidence = data.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            confidence = 1.0

        metadata: dict[str, Any] = {
            key: data[key]
            for key in ("utterance_idx", "language_confidence")
            if key in data and data[key] is not None
        }
        if (
            event_type == stt.SpeechEventType.FINAL_TRANSCRIPT
            and self._utterance_speech_end_wall is not None
        ):
            metadata["speech_end_wall_time"] = self._utterance_speech_end_wall
        end_time = 0.0
        start_time = 0.0
        if event_type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            start_s = data.get("start_s")
            end_s = data.get("end_s")
            if isinstance(start_s, (int, float)) and not isinstance(start_s, bool):
                start_time = max(float(start_s), 0.0)
            if isinstance(end_s, (int, float)) and not isinstance(end_s, bool):
                end_time = max(float(end_s), 0.0)
        if (
            event_type == stt.SpeechEventType.FINAL_TRANSCRIPT
            and self._utterance_speech_end_audio_pos is not None
            and end_time == 0.0
        ):
            end_time = self._utterance_speech_end_audio_pos
        elif (
            event_type == stt.SpeechEventType.FINAL_TRANSCRIPT
            and self._audio_position > 0
            and end_time == 0.0
        ):
            end_time = self._audio_position

        speech_data = stt.SpeechData(
            language=LanguageCode(language),
            text=text,
            start_time=start_time,
            end_time=end_time,
            confidence=float(confidence),
            metadata=metadata or None,
        )
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=event_type,
                request_id=self._request_id,
                alternatives=[speech_data],
            )
        )
        return True

    def _handle_session_end(self, data: dict[str, Any]) -> None:
        self._capture_server_ids(data)
        self._flush_terminal_utterance()
        audio_duration = data.get("audio_duration_s")
        if (
            isinstance(audio_duration, (int, float))
            and not isinstance(audio_duration, bool)
            and not self._server_audio_duration_reported
        ):
            # Report whatever audio is still buffered locally, then top up to Sarvam's
            # authoritative total so the session bills exactly once for it.
            self._audio_duration_collector.flush()
            server_audio_duration = max(float(audio_duration), 0.0)
            delta = max(server_audio_duration - self._total_reported_audio_duration, 0.0)
            if delta:
                self._emit_usage(delta)
            self._server_audio_duration_reported = True
        else:
            self._emit_local_usage_fallback()
        self._session_ended = True

    def _handle_error_event(self, data: dict[str, Any]) -> None:
        if not data.get("is_fatal", False):
            self._logger.warning(
                "Non-fatal Sarvam realtime STT error",
                extra={
                    **self._build_log_context(),
                    "error_code": data.get("code"),
                    "error_message": data.get("message"),
                    "status_code": data.get("status_code"),
                    "raw_message": data,
                },
            )
            return

        code = data.get("code", "unknown")
        status_code = data.get("status_code", -1)
        if not isinstance(status_code, int):
            status_code = -1
        self._logger.error(
            "Fatal Sarvam realtime STT error",
            extra={
                **self._build_log_context(),
                "error_code": code,
                "error_message": data.get("message", code),
                "status_code": status_code,
                "raw_message": data,
            },
        )
        raise APIStatusError(
            message=f"Sarvam realtime STT error: {data.get('message', code)}",
            status_code=status_code,
            request_id=self._request_id or None,
            body=data,
            retryable=code == "model_unavailable",
        )

    def _on_audio_duration_report(self, duration: float) -> None:
        self._local_audio_duration += duration
        self._emit_usage(duration)

    def _emit_local_usage_fallback(self) -> None:
        if self._server_audio_duration_reported:
            return
        self._audio_duration_collector.flush()

    def _emit_usage(self, duration: float) -> None:
        self._total_reported_audio_duration += duration
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                recognition_usage=stt.RecognitionUsage(audio_duration=duration),
            )
        )


# Deprecated compatibility aliases. Prefer the endpoint-specific realtime names above.
STTStreaming = STTRealtime
StreamingSpeechStream = RealtimeSpeechStream
StreamingSTTOptions = RealtimeSTTOptions
