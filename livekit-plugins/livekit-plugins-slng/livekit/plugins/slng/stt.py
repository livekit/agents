# Copyright 2023 LiveKit, Inc.
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
import base64
import contextlib
import dataclasses
import json
import os
import weakref
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.stt import SpeechEvent
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .gateway_adapter import build_stt_init_payload, normalize_region_override
from .log import logger

STTEncoding = Literal["pcm_s16le", "pcm_mulaw"]
DEFAULT_BUFFER_SIZE_SECONDS = 0.064
MAX_BUFFER_SECONDS = 600
MAX_IMMEDIATE_RETRIES = 1

# Define bytes per frame for different encoding types
bytes_per_frame = {
    "pcm_s16le": 2,
    "pcm_mulaw": 1,
}


def _safe_error_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _extract_model_from_endpoint(model_endpoint: str) -> str | None:
    parsed = urlparse(model_endpoint)
    path = parsed.path.rstrip("/")
    marker = "/v1/stt/"
    marker_index = path.find(marker)
    if marker_index == -1:
        return None

    model_part = path[marker_index + len(marker) :]
    return model_part or None


def _default_stt_endpoint(*, slng_base_url: str, model: str) -> str:
    protocol = "ws" if "localhost" in slng_base_url or "127.0.0.1" in slng_base_url else "wss"
    return f"{protocol}://{slng_base_url}/v1/stt/{model}"


@dataclass
class STTOptions:
    # Audio format options
    sample_rate: int = 16000
    buffer_size_seconds: float = DEFAULT_BUFFER_SIZE_SECONDS
    encoding: str = "pcm_s16le"

    # Common SLNG streaming options (work across all models)
    enable_partial_transcripts: bool = True

    # Common VAD options (work across all models)
    vad_threshold: float = 0.5
    vad_min_silence_duration_ms: int = 300
    vad_speech_pad_ms: int = 30

    # Common diarization options (work across all models)
    enable_diarization: bool = False
    min_speakers: int | None = None
    max_speakers: int | None = None

    language: str = "en"


class STT(stt.STT):
    def __bool__(self) -> bool:
        # LiveKit Agents code may use truthiness checks like `stt or None`.
        # Some EventEmitter-style bases can be falsy when they have no listeners,
        # which would unintentionally disable STT.
        return True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_token: str | None = None,
        model: str = "deepgram/nova:3",
        model_endpoint: str | None = None,
        model_endpoints: list[str] | None = None,
        slng_base_url: str = "api.slng.ai",
        region_override: str | list[str] | None = None,
        sample_rate: int = 16000,
        encoding: NotGivenOr[STTEncoding] = NOT_GIVEN,
        buffer_size_seconds: float = DEFAULT_BUFFER_SIZE_SECONDS,
        # Common SLNG options
        enable_partial_transcripts: bool = True,
        vad_threshold: float = 0.5,
        vad_min_silence_duration_ms: int = 300,
        vad_speech_pad_ms: int = 30,
        enable_diarization: bool = False,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        language: str = "en",
        http_session: aiohttp.ClientSession | None = None,
        **model_options: Any,
    ) -> None:
        """
        Initialize SLNG STT.

        Args:
            api_key: SLNG API key for authentication.
            api_token: Deprecated alias for api_key. Use api_key instead.
            model: SLNG model identifier (for example "deepgram/nova:3")
            model_endpoint: Optional full SLNG WebSocket endpoint URL
            model_endpoints: Optional fallback STT endpoints
            slng_base_url: SLNG gateway host. Defaults to "api.slng.ai"
            region_override: Optional gateway region override sent as X-Region-Override.
            sample_rate: Audio sample rate (default: 16000)
            encoding: Audio encoding format
            buffer_size_seconds: Buffer size in seconds
            enable_partial_transcripts: Enable interim results
            vad_threshold: Voice activity detection threshold
            vad_min_silence_duration_ms: Min silence duration for VAD
            vad_speech_pad_ms: Speech padding for VAD
            enable_diarization: Enable speaker identification
            min_speakers: Minimum speakers for diarization
            max_speakers: Maximum speakers for diarization
            language: Language code (default: "en")
            http_session: Optional HTTP session
            **model_options: Model-specific options (e.g., whisper_params={"task": "translate"})
        """
        # Resolve api_key from parameter, legacy api_token, or SLNG_API_KEY env var
        resolved_key = api_key or api_token or os.environ.get("SLNG_API_KEY")
        if not resolved_key:
            raise ValueError("api_key is required, or set SLNG_API_KEY environment variable")
        if api_token and not api_key:
            import warnings

            warnings.warn(
                "api_token is deprecated, use api_key instead",
                DeprecationWarning,
                stacklevel=2,
            )

        # Detect if endpoint supports streaming (WebSocket endpoints do)
        # - streaming=True: Supports real-time streaming (WebSocket only)
        # - streaming=False: HTTP batch recognition only
        resolved_model_endpoint = model_endpoint or _default_stt_endpoint(
            slng_base_url=slng_base_url,
            model=model,
        )
        endpoints = list(
            model_endpoints
            or [
                resolved_model_endpoint,
            ]
        )
        if not endpoints:
            endpoints = [resolved_model_endpoint]
        primary_endpoint = endpoints[0]

        is_streaming = primary_endpoint.startswith("ws://") or primary_endpoint.startswith("wss://")

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=is_streaming,
                interim_results=is_streaming,
            ),
        )

        self._api_key = resolved_key
        self._region_override_header = normalize_region_override(region_override)
        self._model_endpoints = endpoints
        self._active_endpoint_index = 0
        self._model_endpoint = endpoints[0]
        self._models = [_extract_model_from_endpoint(e) for e in endpoints]
        self._model = (
            self._models[0] if self._models else _extract_model_from_endpoint(primary_endpoint)
        )

        self._opts = STTOptions(
            sample_rate=sample_rate,
            buffer_size_seconds=buffer_size_seconds,
            enable_partial_transcripts=enable_partial_transcripts,
            vad_threshold=vad_threshold,
            vad_min_silence_duration_ms=vad_min_silence_duration_ms,
            vad_speech_pad_ms=vad_speech_pad_ms,
            enable_diarization=enable_diarization,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            language=language,
        )

        if is_given(encoding):
            self._opts.encoding = encoding

        # Store any extra model-specific options
        self._model_options = model_options

        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    def _set_active_endpoint_index(self, index: int) -> None:
        """Update the active endpoint index (called by SpeechStream after successful failover)."""
        self._active_endpoint_index = index

    @property
    def model(self) -> str:
        return "slng"

    @property
    def provider(self) -> str:
        return "SLNG"

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """
        HTTP batch recognition for non-streaming STT.

        Converts audio buffer to base64 and sends to SLNG HTTP endpoint.
        """
        # Use language from parameter or fall back to instance default
        lang = language if is_given(language) else self._opts.language

        # Convert AudioBuffer to bytes
        audio_data = buffer.data.tobytes()  # type: ignore

        # Encode as base64
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        # Prepare request payload
        payload = {
            "audio_b64": audio_b64,
            "language": lang,
        }

        # Add any model-specific options
        if self._model_options:
            payload.update(self._model_options)

        try:
            async with self.session.post(
                self._model_endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    **(
                        {"X-Region-Override": self._region_override_header}
                        if self._region_override_header
                        else {}
                    ),
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=conn_options.timeout, sock_connect=conn_options.timeout
                ),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"[SLNG STT] HTTP error {resp.status}: {error_text}")
                    raise APIStatusError(
                        f"SLNG STT HTTP error {resp.status}: {error_text}",
                        status_code=resp.status,
                    )

                data = await resp.json()

                # Extract transcription from response
                # Expected format: {"text": "...", "language": "en", "segments": [...]}
                text = data.get("text", "")
                detected_language = data.get("language", lang)
                segments = data.get("segments", [])

                # Calculate start and end times from segments
                start_time = segments[0].get("start", 0.0) if segments else 0.0
                end_time = segments[-1].get("end", 0.0) if segments else 0.0

                # Build SpeechEvent
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            language=detected_language,
                            text=text,
                            confidence=1.0,  # SLNG doesn't provide confidence in HTTP mode
                            start_time=start_time,
                            end_time=end_time,
                        )
                    ],
                )

        except aiohttp.ClientError as e:
            logger.error(f"[SLNG STT] HTTP connection error: {e}")
            raise APIConnectionError(f"SLNG STT HTTP connection error: {e}") from e
        except TimeoutError:
            logger.error("[SLNG STT] HTTP request timed out")
            raise APITimeoutError("SLNG STT HTTP request timed out") from None
        except APIStatusError:
            raise
        except Exception as e:
            logger.error(f"[SLNG STT] HTTP unexpected error: {e}", exc_info=True)
            raise APIStatusError(f"SLNG STT HTTP error: {e}") from e

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = dataclasses.replace(self._opts)
        if is_given(language):
            config.language = language
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            api_key=self._api_key,
            region_override_header=self._region_override_header,
            model_endpoints=self._model_endpoints,
            models=self._models,
            active_endpoint_index=self._active_endpoint_index,
            model_options=self._model_options,
            http_session=self.session,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        enable_partial_transcripts: NotGivenOr[bool] = NOT_GIVEN,
        enable_diarization: NotGivenOr[bool] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        vad_min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_speech_pad_ms: NotGivenOr[int] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(enable_partial_transcripts):
            self._opts.enable_partial_transcripts = enable_partial_transcripts
        if is_given(enable_diarization):
            self._opts.enable_diarization = enable_diarization
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(vad_min_silence_duration_ms):
            self._opts.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        if is_given(vad_speech_pad_ms):
            self._opts.vad_speech_pad_ms = vad_speech_pad_ms
        if is_given(language):
            self._opts.language = language
        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds

        for stream in self._streams:
            stream.update_options(
                enable_partial_transcripts=enable_partial_transcripts,
                enable_diarization=enable_diarization,
                vad_threshold=vad_threshold,
                vad_min_silence_duration_ms=vad_min_silence_duration_ms,
                vad_speech_pad_ms=vad_speech_pad_ms,
                language=language,
                buffer_size_seconds=buffer_size_seconds,
            )


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        region_override_header: str | None,
        model_endpoints: list[str],
        models: list[str | None],
        active_endpoint_index: int,
        model_options: dict,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)

        self._stt_parent: STT = stt
        self._opts = opts
        self._api_key = api_key
        self._region_override_header = region_override_header
        self._model_endpoints = list(model_endpoints)
        self._models = list(models)
        self._active_endpoint_index = active_endpoint_index
        self._model_options = model_options
        self._session = http_session
        self._speech_duration: float = 0

        # keep a list of final transcripts to combine them inside the END_OF_SPEECH event
        self._final_events: list[SpeechEvent] = []
        self._reconnect_event = asyncio.Event()
        self._pending_replay: bytes | None = None

    def update_options(
        self,
        *,
        enable_partial_transcripts: NotGivenOr[bool] = NOT_GIVEN,
        enable_diarization: NotGivenOr[bool] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        vad_min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_speech_pad_ms: NotGivenOr[int] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(enable_partial_transcripts):
            self._opts.enable_partial_transcripts = enable_partial_transcripts
        if is_given(enable_diarization):
            self._opts.enable_diarization = enable_diarization
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(vad_min_silence_duration_ms):
            self._opts.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        if is_given(vad_speech_pad_ms):
            self._opts.vad_speech_pad_ms = vad_speech_pad_ms
        if is_given(language):
            self._opts.language = language
        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds

        self._reconnect_event.set()

    def _samples_per_buffer(self) -> int:
        try:
            buffer_size_seconds = float(self._opts.buffer_size_seconds)
        except (TypeError, ValueError):
            buffer_size_seconds = DEFAULT_BUFFER_SIZE_SECONDS

        if buffer_size_seconds <= 0:
            buffer_size_seconds = DEFAULT_BUFFER_SIZE_SECONDS

        return max(1, round(self._opts.sample_rate * buffer_size_seconds))

    async def _run(self) -> None:
        bps = bytes_per_frame.get(self._opts.encoding, 2)
        max_buffer_bytes = int(MAX_BUFFER_SECONDS * self._opts.sample_rate * bps)

        buffered_audio = bytearray()
        pending_switch_succeeded: tuple[str | None, str | None] | None = None
        send: asyncio.Task[None] | None = None
        recv: asyncio.Task[None] | None = None
        wait_reconnect: asyncio.Task[bool] | None = None
        immediate_reconnect_attempts: dict[int, int] = {}

        def current_model() -> str | None:
            try:
                return self._models[self._active_endpoint_index]
            except Exception:
                return None

        def next_model() -> str | None:
            idx = self._active_endpoint_index + 1
            if idx < len(self._models):
                return self._models[idx]
            return None

        async def failover(*, exc: BaseException | None) -> bool:
            nonlocal pending_switch_succeeded
            from_model = current_model()
            exc_info = (
                (type(exc), exc, exc.__traceback__)
                if exc is not None and exc.__traceback__ is not None
                else None
            )
            if self._active_endpoint_index + 1 >= len(self._model_endpoints):
                logger.error(
                    "STT fallback exhausted: from=%s",
                    from_model,
                    exc_info=exc_info,
                )
                return False

            to_model = next_model()
            logger.warning(
                "STT attempt failed: switching %s -> %s",
                from_model,
                to_model,
                exc_info=exc_info,
            )

            pending_switch_succeeded = (from_model, to_model)
            self._active_endpoint_index += 1
            self._pending_replay = bytes(buffered_audio)
            return True

        async def next_audio_frame() -> Any | None:
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    continue
                return item
            return None

        async def send_task(
            ws: aiohttp.ClientWebSocketResponse, *, pending_frames: list[Any]
        ) -> None:
            samples_per_buffer = self._samples_per_buffer()
            bytes_per_sample = bytes_per_frame.get(self._opts.encoding, 2)
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_buffer,
            )

            if self._pending_replay:
                data = self._pending_replay
                self._pending_replay = None
                chunk_bytes = samples_per_buffer * bytes_per_sample
                for i in range(0, len(data), chunk_bytes):
                    await ws.send_bytes(data[i : i + chunk_bytes])

            for frame in pending_frames:
                frames = audio_bstream.write(frame.data.tobytes())
                for out in frames:
                    if len(out.data) % bytes_per_sample != 0:
                        continue
                    payload = bytes(out.data)
                    await ws.send_bytes(payload)
                    buffered_audio.extend(payload)
                    if len(buffered_audio) > max_buffer_bytes:
                        excess = len(buffered_audio) - max_buffer_bytes
                        del buffered_audio[:excess]

            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    frames = audio_bstream.flush()
                else:
                    frames = audio_bstream.write(item.data.tobytes())

                for frame in frames:
                    if len(frame.data) % bytes_per_sample != 0:
                        continue

                    payload = bytes(frame.data)
                    await ws.send_bytes(payload)

                    buffered_audio.extend(payload)
                    if len(buffered_audio) > max_buffer_bytes:
                        excess = len(buffered_audio) - max_buffer_bytes
                        del buffered_audio[:excess]

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            speech_started = False

            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError("SLNG connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.debug("[SLNG STT] ignoring non-JSON text frame: %s", msg.data[:200])
                    continue
                if not isinstance(data, dict):
                    continue

                msg_type = data.get("type")
                if msg_type in ("Metadata", "SpeechStarted", "UtteranceEnd"):
                    continue

                if msg_type == "Results":
                    is_final_value = data.get("is_final")
                    if isinstance(is_final_value, str):
                        is_final = is_final_value.strip().lower() in ("true", "1")
                    else:
                        is_final = bool(is_final_value)
                    raw_channel = data.get("channel")
                    channel = raw_channel if isinstance(raw_channel, dict) else {}
                    raw_alts = channel.get("alternatives")
                    alternatives = raw_alts if isinstance(raw_alts, list) else []
                    alt0 = (
                        alternatives[0]
                        if alternatives and isinstance(alternatives[0], dict)
                        else {}
                    )
                    data = {
                        "type": "final_transcript" if is_final else "partial_transcript",
                        "transcript": alt0.get("transcript", ""),
                        "confidence": alt0.get("confidence", 0.0),
                        "words": alt0.get("words", []),
                        "language": data.get("language", alt0.get("language")),
                    }
                    msg_type = data["type"]

                if msg_type == "Error":
                    raise APIStatusError(
                        f"SLNG STT error: {data.get('description') or data.get('message')}"
                    )

                if msg_type in ("partial_transcript", "final_transcript"):
                    if (
                        msg_type == "partial_transcript"
                        and not self._opts.enable_partial_transcripts
                    ):
                        continue
                    text = data.get("transcript", "")
                    if not text:
                        continue

                    is_final = msg_type == "final_transcript"
                    confidence = data.get("confidence", 0.0)
                    language = data.get("language", self._opts.language)
                    words = data.get("words", [])

                    # Emit START_OF_SPEECH on first transcript (interim or final)
                    if not speech_started:
                        speech_started = True
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                        )

                    if is_final:
                        buffered_audio.clear()

                        start_time = words[0].get("start", 0.0) if words else 0.0
                        end_time = words[-1].get("end", 0.0) if words else 0.0
                    else:
                        start_time = 0.0
                        end_time = 0.0

                    event = stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT
                        if is_final
                        else stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        alternatives=[
                            stt.SpeechData(
                                language=language,
                                text=text,
                                confidence=confidence,
                                start_time=start_time,
                                end_time=end_time,
                            )
                        ],
                    )
                    self._event_ch.send_nowait(event)

                    # Emit END_OF_SPEECH after each final transcript.
                    # Note: the gateway may send multiple final_transcript messages
                    # per utterance (e.g., sentence-by-sentence). Each final is
                    # treated as a completed segment, so START/END bracket each one.
                    if is_final:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                        )
                        speech_started = False

        while True:
            send = None
            recv = None
            wait_reconnect = None

            pending_frames: list[Any] = []
            if self._pending_replay is None:
                first = await next_audio_frame()
                if first is None:
                    return
                pending_frames.append(first)

            endpoint = self._model_endpoints[self._active_endpoint_index]
            model = current_model()

            ws: aiohttp.ClientWebSocketResponse | None = None
            try:
                ws = await self._connect_ws(model_endpoint=endpoint, model=model)
                if pending_switch_succeeded:
                    from_model, to_model = pending_switch_succeeded
                    logger.info("STT switched to fallback: %s -> %s", from_model, to_model)
                    pending_switch_succeeded = None
                    # Propagate successful failover to parent so new streams
                    # start from the working endpoint
                    self._stt_parent._set_active_endpoint_index(self._active_endpoint_index)

                send = asyncio.create_task(send_task(ws, pending_frames=pending_frames))
                recv = asyncio.create_task(recv_task(ws))
                wait_reconnect = asyncio.create_task(self._reconnect_event.wait())

                tasks_group = asyncio.gather(send, recv)
                done, _ = await asyncio.wait(
                    (tasks_group, wait_reconnect),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if wait_reconnect in done:
                    self._reconnect_event.clear()
                    tasks_group.cancel()
                    await utils.aio.gracefully_cancel(send, recv, wait_reconnect)
                    continue

                for task in done:
                    task.result()

                await utils.aio.gracefully_cancel(wait_reconnect)
                return
            except Exception as exc:
                tasks = [t for t in (send, recv, wait_reconnect) if t is not None]
                if tasks:
                    with contextlib.suppress(Exception):
                        await utils.aio.gracefully_cancel(*tasks)
                if (
                    isinstance(exc, APIStatusError)
                    and not buffered_audio
                    and self._pending_replay is None
                ):
                    status_code = _safe_error_code(exc)
                    is_permanent_client_error = (
                        status_code is not None and 400 <= status_code < 500 and status_code != 429
                    )
                    endpoint_index = self._active_endpoint_index
                    attempts = immediate_reconnect_attempts.get(endpoint_index, 0)

                    if not is_permanent_client_error and attempts < MAX_IMMEDIATE_RETRIES:
                        immediate_reconnect_attempts[endpoint_index] = attempts + 1
                        replay_bytes = bytearray()
                        for frame in pending_frames:
                            data = getattr(frame, "data", None)
                            if data is None:
                                continue
                            replay_bytes.extend(data.tobytes())
                        if replay_bytes:
                            self._pending_replay = bytes(replay_bytes)
                        continue
                if not await failover(exc=exc):
                    return
                immediate_reconnect_attempts[self._active_endpoint_index] = 0
                continue
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(
        self, *, model_endpoint: str, model: str | None
    ) -> aiohttp.ClientWebSocketResponse:
        # Match e2e test headers exactly - send both Authorization and X-API-Key
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-API-Key": self._api_key,
        }
        if self._region_override_header:
            headers["X-Region-Override"] = self._region_override_header

        # Don't enable compression - e2e tests work without it and compress=15
        # was causing handshake errors with Deepgram Nova endpoint
        ws = await self._session.ws_connect(
            model_endpoint,
            headers=headers,
        )

        init_message = build_stt_init_payload(
            model=model,
            language=self._opts.language,
            sample_rate=self._opts.sample_rate,
            encoding=self._opts.encoding,
            vad_threshold=self._opts.vad_threshold,
            vad_min_silence_duration_ms=self._opts.vad_min_silence_duration_ms,
            vad_speech_pad_ms=self._opts.vad_speech_pad_ms,
            enable_diarization=self._opts.enable_diarization,
            enable_partial_transcripts=self._opts.enable_partial_transcripts,
            min_speakers=self._opts.min_speakers,
            max_speakers=self._opts.max_speakers,
            model_options=self._model_options,
        )

        try:
            await ws.send_str(json.dumps(init_message))
        except Exception:
            await ws.close()
            raise
        return ws
