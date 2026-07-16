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
import contextlib
import dataclasses
import json
import os
import warnings
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .connection import (
    CandidateState,
    PluginEvent,
    STTConnectionConfig,
    bridge_endpoint,
    bridge_model,
)
from .gateway_adapter import (
    build_external_tracking_headers,
    build_stt_init_payload,
    error_details,
    extract_error_status,
    is_non_retryable_client_error,
    is_payload_too_large,
    normalize_region_override,
    normalize_world_part_override,
)
from .log import logger

if TYPE_CHECKING:
    from livekit.agents import AgentSession, UserStateChangedEvent

# Only pcm_s16le is supported: LiveKit AudioFrames carry 16-bit PCM and the
# plugin sends those bytes verbatim (no transcoding is performed).
STTEncoding = Literal["pcm_s16le"]
_BYTES_PER_SAMPLE = 2
DEFAULT_BUFFER_SIZE_SECONDS = 0.064
_FINALIZE_MSG = json.dumps({"type": "finalize"})
_KEEPALIVE_MSG = json.dumps({"type": "keepalive"})
_KEEPALIVE_INTERVAL_S = 5.0


def _safe_error_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _extract_error_text(data: dict[str, object]) -> str:
    nested = data.get("data")
    if isinstance(nested, dict):
        for key in ("message", "description", "error"):
            value = nested.get(key)
            if isinstance(value, str) and value:
                return value

    for key in ("message", "description", "error"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            nested_message = value.get("message") or value.get("description") or value.get("error")
            if isinstance(nested_message, str) and nested_message:
                return nested_message

    return "Unknown error"


def _extract_error_code(data: dict[str, object]) -> int | None:
    return extract_error_status(data)


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
        model: str | None = None,
        connections: Sequence[str | STTConnectionConfig] | None = None,
        model_endpoint: str | None = None,
        model_endpoints: Sequence[str] | None = None,
        provider_api_key: str | None = None,
        slng_base_url: str = "api.slng.ai",
        region_override: str | list[str] | None = None,
        world_part_override: str | None = None,
        external_agent_id: str | None = None,
        external_session_id: str | None = None,
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
        extra_headers: dict[str, str] | None = None,
        final_timeout_s: float | None = None,
        fallback_recovery_cooldown_s: float = 60.0,
        **model_options: Any,
    ) -> None:
        """
        Initialize SLNG STT.

        Args:
            api_key: SLNG API key. Falls back to the ``SLNG_API_KEY`` env var.
            api_token: Deprecated alias for ``api_key``.
            model: Unmute Bridge model identifier, e.g. "deepgram/nova:3".
            connections: Ordered model, endpoint, or typed connection candidates.
            provider_api_key: Optional BYOK provider credential, sent as the
                ``X-Slng-Provider-Key`` header (external providers only).
            slng_base_url: Gateway host. Defaults to "api.slng.ai".
            region_override: Optional gateway region override, sent as the
                ``X-Region-Override`` header. Accepts a single region or a list
                of preferred regions in priority order.
            world_part_override: Optional gateway world-part override, sent as
                the ``X-World-Part-Override`` header. Constrains routing to a
                broad geographic zone (for example "eu", "na", "ap") when an
                exact region is not required. ``region_override`` takes
                precedence when both are set.
            external_agent_id: Optional tracking ID attached to usage events as
                the ``X-SLNG-Agent-Id`` header (max 128 chars).
            external_session_id: Optional tracking ID attached to usage events as
                the ``X-SLNG-Session-Id`` header (max 128 chars).
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
        if model_endpoint is not None or model_endpoints is not None:
            raise ValueError(
                "model_endpoint/model_endpoints were removed in 2.0: the plugin "
                "always connects through the Unmute Bridge. Pass "
                "model='provider/model:variant' or connections=[...] instead."
            )
        if api_token is not None:
            warnings.warn(
                "api_token is deprecated, use api_key instead",
                DeprecationWarning,
                stacklevel=2,
            )
        resolved_key = api_key or api_token or os.environ.get("SLNG_API_KEY")
        if not resolved_key:
            raise ValueError("api_key is required, or set the SLNG_API_KEY environment variable")

        if model is not None and connections:
            raise ValueError("use model or connections, not both")
        raw_connections: Sequence[str | STTConnectionConfig]
        if connections:
            raw_connections = connections
        elif model is not None:
            raw_connections = [model]
        else:
            raise ValueError("model or connections is required")

        configs: list[STTConnectionConfig] = []
        for candidate in raw_connections:
            if isinstance(candidate, STTConnectionConfig):
                endpoint_model = bridge_model(candidate.endpoint, "stt")
                if candidate.model is not None and candidate.model != endpoint_model:
                    raise ValueError("STT connection model must match its endpoint")
                configs.append(
                    STTConnectionConfig(
                        endpoint=candidate.endpoint,
                        model=endpoint_model,
                        headers=dict(candidate.headers),
                        init=dict(candidate.init) if candidate.init is not None else None,
                    )
                )
                continue
            endpoint = (
                candidate
                if "://" in candidate
                else bridge_endpoint(slng_base_url, "stt", candidate)
            )
            configs.append(
                STTConnectionConfig(
                    endpoint=endpoint,
                    model=bridge_model(endpoint, "stt"),
                )
            )

        resolved_model_endpoint = configs[0].endpoint

        # Detect if endpoint supports streaming (WebSocket endpoints do)
        # - streaming=True: Supports real-time streaming (WebSocket only)
        # - streaming=False: HTTP batch recognition only
        is_streaming = resolved_model_endpoint.startswith(
            "ws://"
        ) or resolved_model_endpoint.startswith("wss://")

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=is_streaming,
                interim_results=is_streaming,
                aligned_transcript="chunk" if is_streaming else False,
                offline_recognize=False,
            ),
        )

        self._api_token = resolved_key
        self._connections = configs
        self._candidate_state = CandidateState(len(configs), fallback_recovery_cooldown_s)
        self._model_endpoint = configs[0].endpoint
        self._model = configs[0].model

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
            if encoding != "pcm_s16le":
                raise ValueError(
                    "only pcm_s16le encoding is supported: LiveKit audio frames "
                    "are 16-bit PCM and the plugin does not transcode"
                )
            self._opts.encoding = encoding

        # Store any extra model-specific options
        self._model_options = model_options

        self._final_timeout_s = final_timeout_s
        self._session = http_session
        self._extra_headers = dict(extra_headers or {})
        # Route toward preferred gateway regions via the X-Region-Override header.
        region_override_header = normalize_region_override(region_override)
        if region_override_header:
            self._extra_headers.setdefault("X-Region-Override", region_override_header)
        world_part_header = normalize_world_part_override(world_part_override)
        if world_part_header:
            self._extra_headers.setdefault("X-World-Part-Override", world_part_header)
        self._extra_headers.update(
            build_external_tracking_headers(
                external_agent_id=external_agent_id,
                external_session_id=external_session_id,
            )
        )
        if provider_api_key is not None:
            byok_key = provider_api_key.strip()
            if not byok_key:
                raise ValueError("provider_api_key must not be empty")
            self._extra_headers["X-Slng-Provider-Key"] = byok_key
        self._streams = weakref.WeakSet[SpeechStream]()

    def _emit_plugin_event(
        self,
        name: str,
        level: Literal["info", "warning", "error"] = "info",
        **data: Any,
    ) -> None:
        self.emit(
            "slng_event",
            PluginEvent(name=name, component="stt", level=level, data=data),
        )

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
        del buffer, language, conn_options
        raise NotImplementedError(
            "SLNG STT recognize() is not supported: the Unmute Bridge is "
            "WebSocket-only. Use stream() instead."
        )

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
            api_token=self._api_token,
            connections=self._connections,
            active_endpoint_index=self._candidate_state.start(),
            candidate_state=self._candidate_state,
            model_options=self._model_options,
            http_session=self.session,
            extra_headers=self._extra_headers,
            final_timeout_s=self._final_timeout_s,
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

    def notify_user_state(self, new_state: str) -> None:
        normalized = str(new_state or "").strip().lower()
        if normalized not in ("speaking", "listening"):
            return
        for stream in list(self._streams):
            try:
                stream.notify_user_state(normalized)
            except Exception:
                logger.debug("Failed to notify STT stream of user_state change", exc_info=True)

    def attach_to_session(self, session: AgentSession) -> None:
        """Forward the session's user state changes to this STT instance.

        Registers a ``user_state_changed`` handler on the given ``AgentSession`` so
        end-of-turn finalization works without manual wiring. Equivalent to calling
        ``notify_user_state(ev.new_state)`` from your own handler.
        """

        @session.on("user_state_changed")
        def _on_user_state_changed(ev: UserStateChangedEvent) -> None:
            self.notify_user_state(ev.new_state)

    def set_final_timeout_allowed(self, allowed: bool) -> None:
        for stream in list(self._streams):
            try:
                stream.set_final_timeout_allowed(allowed)
            except Exception:
                logger.debug("Failed to update STT final-timeout permission", exc_info=True)


class SpeechStream(stt.SpeechStream):
    # Used to close websocket
    _CLOSE_MSG: str = json.dumps({"type": "close"})

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_token: str,
        connections: Sequence[STTConnectionConfig],
        active_endpoint_index: int,
        candidate_state: CandidateState,
        model_options: dict[str, Any],
        http_session: aiohttp.ClientSession,
        extra_headers: dict[str, str],
        final_timeout_s: float | None,
    ) -> None:
        self._candidate_max_retry = conn_options.max_retry
        super().__init__(
            stt=stt,
            conn_options=dataclasses.replace(conn_options, max_retry=0),
            sample_rate=opts.sample_rate,
        )

        self._opts = opts
        self._slng_stt = stt
        self._api_token = api_token
        self._connections = list(connections)
        self._active_endpoint_index = active_endpoint_index
        self._candidate_state = candidate_state
        self._model_options = model_options
        self._session = http_session
        self._extra_headers = dict(extra_headers)
        self._speech_duration: float = 0

        self._reconnect_event = asyncio.Event()
        self._final_timeout_s = final_timeout_s
        self._user_state_event = asyncio.Event()
        self._timeout_permission_event = asyncio.Event()
        self._user_state: Literal["speaking", "listening"] | None = None
        self._has_spoken_since_last_response = False
        self._final_timeout_allowed = True

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

    def notify_user_state(self, new_state: str) -> None:
        normalized = str(new_state or "").strip().lower()
        if normalized not in ("speaking", "listening"):
            return
        self._user_state = normalized  # type: ignore[assignment]
        if normalized == "speaking":
            self._has_spoken_since_last_response = True
        self._user_state_event.set()

    def set_final_timeout_allowed(self, allowed: bool) -> None:
        self._final_timeout_allowed = bool(allowed)
        self._timeout_permission_event.set()

    def _samples_per_buffer(self) -> int:
        try:
            buffer_size_seconds = float(self._opts.buffer_size_seconds)
        except (TypeError, ValueError):
            buffer_size_seconds = DEFAULT_BUFFER_SIZE_SECONDS

        if buffer_size_seconds <= 0:
            buffer_size_seconds = DEFAULT_BUFFER_SIZE_SECONDS

        return max(1, round(self._opts.sample_rate * buffer_size_seconds))

    async def _run(self) -> None:
        max_buffer_seconds = 600
        max_buffer_bytes = int(max_buffer_seconds * self._opts.sample_rate * 2)

        buffered_audio = bytearray()
        awaiting_final = False
        final_timeout_task: asyncio.Task[None] | None = None
        switch_event = asyncio.Event()
        pending_failover: (
            tuple[Literal["hard_fail", "timeout"], BaseException | None, float | None] | None
        ) = None
        pending_replay: bytes | None = None
        finalize_requested_for_buffer = False
        pending_switch_succeeded: (
            tuple[str | None, str | None, Literal["hard_fail", "timeout"], float | None] | None
        ) = None
        sent_audio_since_finalize = False
        pending_non_empty_transcript = False
        # Hoisted to _run scope so a mid-utterance failover/reconnect does not
        # emit a duplicate START_OF_SPEECH for the same utterance.
        speech_started = False
        input_finished = False
        closing = False
        protocol_close_sent = False
        pending_user_state_finalize = False
        send_lock = asyncio.Lock()
        last_client_send_at = asyncio.get_running_loop().time()
        send: asyncio.Task[None] | None = None
        recv: asyncio.Task[bool] | None = None
        wait_switch: asyncio.Task[bool] | None = None
        wait_reconnect: asyncio.Task[bool] | None = None
        candidate_attempts = 0
        recover_primary_after_final = False

        def current_model() -> str | None:
            try:
                return self._connections[self._active_endpoint_index].model
            except Exception:
                return None

        def next_model() -> str | None:
            idx = self._active_endpoint_index + 1
            if idx < len(self._connections):
                return self._connections[idx].model
            return None

        def start_final_timeout(timeout_s: float) -> None:
            nonlocal final_timeout_task

            if timeout_s <= 0:
                return
            if final_timeout_task and not final_timeout_task.done():
                final_timeout_task.cancel()

            async def _monitor() -> None:
                nonlocal pending_failover
                try:
                    await asyncio.sleep(timeout_s)
                except asyncio.CancelledError:
                    return
                if awaiting_final:
                    pending_failover = ("timeout", TimeoutError(), timeout_s)
                    switch_event.set()

            final_timeout_task = asyncio.create_task(_monitor())

        def cancel_final_timeout() -> None:
            nonlocal final_timeout_task
            if final_timeout_task and not final_timeout_task.done():
                final_timeout_task.cancel()
            final_timeout_task = None

        def audio_duration_from_bytes(payload: bytes | bytearray) -> float:
            if self._opts.sample_rate <= 0:
                return 0.0
            return len(payload) / (self._opts.sample_rate * _BYTES_PER_SAMPLE)

        def emit_recognition_usage() -> None:
            if self._speech_duration <= 0:
                return
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    alternatives=[],
                    recognition_usage=stt.RecognitionUsage(audio_duration=self._speech_duration),
                )
            )
            self._speech_duration = 0

        def retrieve_task_exception(task: asyncio.Task[Any]) -> None:
            if task.cancelled():
                return
            with contextlib.suppress(Exception):
                task.exception()

        async def send_finalize_if_needed(
            ws: aiohttp.ClientWebSocketResponse,
            *,
            reason: str,
        ) -> bool:
            nonlocal last_client_send_at, sent_audio_since_finalize
            nonlocal finalize_requested_for_buffer
            async with send_lock:
                if not sent_audio_since_finalize:
                    return False
                finalize_requested_for_buffer = True
                await ws.send_str(_FINALIZE_MSG)
                last_client_send_at = asyncio.get_running_loop().time()
                sent_audio_since_finalize = False
            logger.debug("Sent STT finalize (model=%s reason=%s)", current_model(), reason)
            return True

        async def send_close(ws: aiohttp.ClientWebSocketResponse, *, reason: str) -> None:
            nonlocal last_client_send_at, closing, protocol_close_sent
            closing = True
            async with send_lock:
                if protocol_close_sent:
                    return
                await ws.send_str(self._CLOSE_MSG)
                last_client_send_at = asyncio.get_running_loop().time()
                protocol_close_sent = True
            logger.debug("Sent STT close (model=%s reason=%s)", current_model(), reason)

        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal last_client_send_at
            while True:
                await asyncio.sleep(_KEEPALIVE_INTERVAL_S)
                if input_finished or closing or protocol_close_sent:
                    return
                idle_s = asyncio.get_running_loop().time() - last_client_send_at
                if idle_s < _KEEPALIVE_INTERVAL_S:
                    continue
                async with send_lock:
                    if input_finished or closing or protocol_close_sent:
                        return
                    await ws.send_str(_KEEPALIVE_MSG)
                    last_client_send_at = asyncio.get_running_loop().time()
                logger.debug(
                    "Sent STT keepalive (model=%s idle_s=%.3f)",
                    current_model(),
                    idle_s,
                )

        async def send_audio_payload(
            ws: aiohttp.ClientWebSocketResponse,
            payload: bytes,
            *,
            count_usage: bool,
        ) -> None:
            nonlocal last_client_send_at, sent_audio_since_finalize
            nonlocal finalize_requested_for_buffer
            async with send_lock:
                await ws.send_bytes(payload)
                last_client_send_at = asyncio.get_running_loop().time()
                sent_audio_since_finalize = True
                if count_usage:
                    finalize_requested_for_buffer = False
            if count_usage:
                self._speech_duration += audio_duration_from_bytes(payload)

        async def handle_user_state_change() -> None:
            nonlocal awaiting_final, pending_user_state_finalize

            state = self._user_state
            if state == "speaking":
                if awaiting_final:
                    awaiting_final = False
                    cancel_final_timeout()
            elif state == "listening":
                if not awaiting_final:
                    try:
                        self._input_ch.send_nowait(self._FlushSentinel())
                    except RuntimeError:
                        pending_user_state_finalize = False
                    else:
                        pending_user_state_finalize = True
                elif not self._has_spoken_since_last_response:
                    awaiting_final = False
                    cancel_final_timeout()

        async def failover(
            *,
            reason: Literal["hard_fail", "timeout"],
            exc: BaseException | None,
            timeout_s: float | None,
        ) -> bool:
            nonlocal input_finished, closing, protocol_close_sent
            nonlocal pending_non_empty_transcript, pending_replay
            nonlocal pending_switch_succeeded, sent_audio_since_finalize
            nonlocal candidate_attempts
            from_model = current_model()
            exc_info = (
                (type(exc), exc, exc.__traceback__)
                if exc is not None and exc.__traceback__ is not None
                else None
            )
            details = error_details(exc)
            next_index = self._candidate_state.advance(self._active_endpoint_index)
            if next_index is None:
                logger.error(
                    "STT fallback exhausted (reason=%s timeout_s=%s error=%s): from=%s",
                    reason,
                    timeout_s,
                    details["error_message"],
                    from_model,
                    exc_info=exc_info,
                )
                self._slng_stt._emit_plugin_event(
                    "fallback.exhausted",
                    "error",
                    from_model=from_model,
                    reason=reason,
                    timeout_s=timeout_s,
                    **details,
                )
                if exc is not None:
                    raise exc
                raise APIConnectionError("SLNG STT fallback exhausted")

            to_model = next_model()
            logger.warning(
                "STT attempt failed (reason=%s timeout_s=%s error=%s): switching %s -> %s",
                reason,
                timeout_s,
                details["error_message"],
                from_model,
                to_model,
                exc_info=exc_info,
            )
            self._slng_stt._emit_plugin_event(
                "fallback.attempt_failed",
                "warning",
                from_model=from_model,
                to_model=to_model,
                reason=reason,
                timeout_s=timeout_s,
                **details,
            )

            pending_switch_succeeded = (from_model, to_model, reason, timeout_s)
            self._active_endpoint_index = next_index
            candidate_attempts = 0
            input_finished = False
            closing = False
            protocol_close_sent = False
            pending_non_empty_transcript = False
            sent_audio_since_finalize = False
            with contextlib.suppress(Exception):
                self._candidate_state.select(self._active_endpoint_index)
            pending_replay = bytes(buffered_audio)
            return True

        async def next_audio_frame() -> Any | None:
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    continue
                return item
            return None

        async def send_task(
            ws: aiohttp.ClientWebSocketResponse,
            *,
            pending_frames: list[Any],
            close_on_input_end: bool,
        ) -> None:
            nonlocal awaiting_final, input_finished, pending_replay
            nonlocal pending_user_state_finalize, recover_primary_after_final
            samples_per_buffer = self._samples_per_buffer()
            bytes_per_sample = _BYTES_PER_SAMPLE
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_buffer,
            )

            if pending_replay:
                replay_data = pending_replay
                pending_replay = None
                chunk_bytes = samples_per_buffer * bytes_per_sample
                for i in range(0, len(replay_data), chunk_bytes):
                    await send_audio_payload(
                        ws, replay_data[i : i + chunk_bytes], count_usage=False
                    )
                if finalize_requested_for_buffer:
                    await send_finalize_if_needed(ws, reason="failover_replay")
                if awaiting_final and self._final_timeout_s is not None:
                    start_final_timeout(self._final_timeout_s)

            for frame in pending_frames:
                frames = audio_bstream.write(frame.data.tobytes())
                for out in frames:
                    if len(out.data) % bytes_per_sample != 0:
                        continue
                    payload = bytes(out.data)
                    await send_audio_payload(ws, payload, count_usage=True)
                    buffered_audio.extend(payload)
                    if len(buffered_audio) > max_buffer_bytes:
                        excess = len(buffered_audio) - max_buffer_bytes
                        del buffered_audio[:excess]

            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    frames = audio_bstream.flush()
                else:
                    frames = audio_bstream.write(data.data.tobytes())

                for frame in frames:
                    if len(frame.data) % bytes_per_sample != 0:
                        continue

                    payload = bytes(frame.data)
                    await send_audio_payload(ws, payload, count_usage=True)

                    buffered_audio.extend(payload)
                    if len(buffered_audio) > max_buffer_bytes:
                        excess = len(buffered_audio) - max_buffer_bytes
                        del buffered_audio[:excess]

                if isinstance(data, self._FlushSentinel):
                    reason = "user_state_listening" if pending_user_state_finalize else "flush"
                    pending_user_state_finalize = False
                    finalized = await send_finalize_if_needed(ws, reason=reason)
                    if (
                        finalized
                        and self._active_endpoint_index != 0
                        and self._candidate_state.start() == 0
                    ):
                        recover_primary_after_final = True
                    if (
                        finalized
                        and reason == "user_state_listening"
                        and self._final_timeout_allowed
                    ):
                        awaiting_final = True
                        if self._final_timeout_s is not None:
                            start_final_timeout(self._final_timeout_s)

            frames = audio_bstream.flush()
            for frame in frames:
                if len(frame.data) % bytes_per_sample != 0:
                    continue

                payload = bytes(frame.data)
                await send_audio_payload(ws, payload, count_usage=True)
                buffered_audio.extend(payload)
                if len(buffered_audio) > max_buffer_bytes:
                    excess = len(buffered_audio) - max_buffer_bytes
                    del buffered_audio[:excess]

            await send_finalize_if_needed(ws, reason="input_end")
            if close_on_input_end:
                await send_close(ws, reason="input_end")
            input_finished = True

        def recover_primary_if_ready() -> None:
            nonlocal recover_primary_after_final, candidate_attempts
            if not recover_primary_after_final:
                return
            self._active_endpoint_index = 0
            candidate_attempts = 0
            recover_primary_after_final = False
            self._slng_stt._emit_plugin_event("fallback.primary_recovered")
            self._reconnect_event.set()

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> bool:
            nonlocal awaiting_final, pending_non_empty_transcript, speech_started
            nonlocal recover_primary_after_final, candidate_attempts
            nonlocal finalize_requested_for_buffer
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if pending_non_empty_transcript:
                        raise APIStatusError("SLNG STT closed before final transcript")
                    if awaiting_final:
                        raise APIStatusError("SLNG STT closed while awaiting final transcript")
                    if input_finished or protocol_close_sent:
                        return True
                    if not buffered_audio:
                        return False
                    raise APIStatusError("SLNG connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(msg.data)
                if not isinstance(data, dict):
                    continue

                msg_type = data.get("type")
                if msg_type == "ready":
                    self._emit_session_event(data)
                    continue
                if msg_type in (
                    "Metadata",
                    "SpeechStarted",
                    "UtteranceEnd",
                    "utterance_end",
                ):
                    continue

                if msg_type == "Results":
                    is_final_value = data.get("is_final")
                    if isinstance(is_final_value, str):
                        is_final = is_final_value.strip().lower() in ("true", "1")
                    else:
                        is_final = bool(is_final_value)
                    raw_channel = data.get("channel")
                    channel = raw_channel if isinstance(raw_channel, dict) else {}
                    raw_alternatives = channel.get("alternatives")
                    alternatives = raw_alternatives if isinstance(raw_alternatives, list) else []
                    alt0 = (
                        alternatives[0]
                        if alternatives and isinstance(alternatives[0], dict)
                        else {}
                    )
                    detected_language = data.get("language") or alt0.get("language")
                    data = {
                        "type": "final_transcript" if is_final else "partial_transcript",
                        "transcript": alt0.get("transcript", ""),
                        "confidence": alt0.get("confidence", 0.0),
                        "words": alt0.get("words", []),
                    }
                    if detected_language:
                        data["language"] = detected_language
                    msg_type = data["type"]

                if msg_type in ("Error", "error"):
                    message = _extract_error_text(data)
                    status_code = _extract_error_code(data)
                    if status_code is not None:
                        raise APIStatusError(f"SLNG STT error: {message}", status_code=status_code)
                    raise APIStatusError(f"SLNG STT error: {message}")

                if msg_type in ("partial_transcript", "final_transcript"):
                    text = data.get("transcript", "")
                    text = text.strip() if isinstance(text, str) else ""
                    is_final = msg_type == "final_transcript"
                    if not is_final and text:
                        pending_non_empty_transcript = True
                    if (
                        msg_type == "partial_transcript"
                        and not self._opts.enable_partial_transcripts
                    ):
                        continue
                    if not text:
                        if msg_type == "final_transcript" and not pending_non_empty_transcript:
                            if awaiting_final:
                                awaiting_final = False
                                cancel_final_timeout()
                            self._has_spoken_since_last_response = False
                            buffered_audio.clear()
                            finalize_requested_for_buffer = False
                            if speech_started:
                                # Close the bracket opened by an earlier interim so
                                # clients never see a dangling START_OF_SPEECH.
                                speech_started = False
                                self._event_ch.send_nowait(
                                    stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                                )
                            emit_recognition_usage()
                            recover_primary_if_ready()
                        continue

                    confidence = data.get("confidence", 0.0)
                    language = data.get("language", self._opts.language)
                    words = data.get("words", [])

                    # Only a FINAL transcript satisfies an outstanding finalize;
                    # an interim must not disarm the final-result watchdog, or a
                    # provider that never sends the final would hang the stream.
                    if is_final:
                        if awaiting_final:
                            awaiting_final = False
                            cancel_final_timeout()
                        self._has_spoken_since_last_response = False
                        pending_non_empty_transcript = False
                        buffered_audio.clear()
                        finalize_requested_for_buffer = False

                        start_time = words[0].get("start", 0.0) if words else 0.0
                        end_time = words[-1].get("end", 0.0) if words else 0.0
                    else:
                        start_time = 0.0
                        end_time = 0.0

                    if not speech_started:
                        speech_started = True
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                        )

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
                    if is_final:
                        speech_started = False
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                        )
                        emit_recognition_usage()
                        recover_primary_if_ready()

        while True:
            switch_event.clear()
            pending_failover = None
            send = None
            recv = None
            keepalive: asyncio.Task[None] | None = None
            wait_switch = None
            wait_reconnect = None
            wait_user_state: asyncio.Task[bool] | None = None
            wait_timeout_permission: asyncio.Task[bool] | None = None

            pending_frames: list[Any] = []
            if pending_replay is None:
                first = await next_audio_frame()
                if first is None:
                    return
                pending_frames.append(first)

            endpoint = self._connections[self._active_endpoint_index].endpoint
            model = current_model()

            ws: aiohttp.ClientWebSocketResponse | None = None
            try:
                input_finished = False
                closing = False
                protocol_close_sent = False
                ws = await self._connect_ws(model_endpoint=endpoint, model=model)
                last_client_send_at = asyncio.get_running_loop().time()
                if pending_switch_succeeded:
                    from_model, to_model, reason, timeout_s = pending_switch_succeeded
                    logger.info(
                        "STT switched to fallback (reason=%s timeout_s=%s): %s -> %s",
                        reason,
                        timeout_s,
                        from_model,
                        to_model,
                    )
                    self._slng_stt._emit_plugin_event(
                        "fallback.switch_succeeded",
                        from_model=from_model,
                        to_model=to_model,
                        reason=reason,
                        timeout_s=timeout_s,
                    )
                    pending_switch_succeeded = None

                send = asyncio.create_task(
                    send_task(
                        ws,
                        pending_frames=pending_frames,
                        close_on_input_end=True,
                    )
                )
                recv = asyncio.create_task(recv_task(ws))
                keepalive = asyncio.create_task(keepalive_task(ws))
                send.add_done_callback(retrieve_task_exception)
                recv.add_done_callback(retrieve_task_exception)
                if keepalive is not None:
                    keepalive.add_done_callback(retrieve_task_exception)
                wait_switch = asyncio.create_task(switch_event.wait())
                wait_reconnect = asyncio.create_task(self._reconnect_event.wait())
                wait_user_state = asyncio.create_task(self._user_state_event.wait())
                wait_timeout_permission = asyncio.create_task(self._timeout_permission_event.wait())

                retry_connection = False
                while True:
                    active_tasks = [
                        task
                        for task in (
                            send,
                            recv,
                            keepalive,
                            wait_switch,
                            wait_reconnect,
                            wait_user_state,
                            wait_timeout_permission,
                        )
                        if task is not None
                    ]
                    done, _ = await asyncio.wait(
                        active_tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if wait_user_state in done:
                        self._user_state_event.clear()
                        await handle_user_state_change()
                        wait_user_state = asyncio.create_task(self._user_state_event.wait())
                        continue

                    if wait_timeout_permission in done:
                        self._timeout_permission_event.clear()
                        if not self._final_timeout_allowed:
                            awaiting_final = False
                            cancel_final_timeout()
                        wait_timeout_permission = asyncio.create_task(
                            self._timeout_permission_event.wait()
                        )
                        continue

                    if wait_reconnect in done:
                        self._reconnect_event.clear()
                        # An options-change reconnect abandons any in-flight
                        # finalize; disarm the watchdog so the stale timer does
                        # not trigger a spurious failover on the new connection.
                        awaiting_final = False
                        cancel_final_timeout()
                        pending_non_empty_transcript = False
                        # Close the speech bracket for the abandoned utterance
                        # so clients never see a dangling START_OF_SPEECH and
                        # the next utterance opens a fresh one.
                        if speech_started:
                            speech_started = False
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                            )
                        await utils.aio.gracefully_cancel(
                            *[
                                task
                                for task in (
                                    send,
                                    recv,
                                    keepalive,
                                    wait_switch,
                                    wait_reconnect,
                                    wait_user_state,
                                    wait_timeout_permission,
                                )
                                if task is not None
                            ]
                        )
                        retry_connection = True
                        break

                    if wait_switch in done and pending_failover is not None:
                        reason, exc, timeout_s = pending_failover
                        await utils.aio.gracefully_cancel(
                            *[
                                task
                                for task in (
                                    send,
                                    recv,
                                    keepalive,
                                    wait_switch,
                                    wait_reconnect,
                                    wait_user_state,
                                    wait_timeout_permission,
                                )
                                if task is not None
                            ]
                        )
                        if not await failover(reason=reason, exc=exc, timeout_s=timeout_s):
                            return
                        retry_connection = True
                        break

                    if keepalive is not None and keepalive in done:
                        keepalive.result()
                        keepalive = None
                        continue

                    if send is not None and send in done:
                        send.result()
                        send = None
                        continue

                    if recv is not None and recv in done:
                        input_exhausted = recv.result()
                        await utils.aio.gracefully_cancel(
                            *[
                                task
                                for task in (
                                    send,
                                    keepalive,
                                    wait_switch,
                                    wait_reconnect,
                                    wait_user_state,
                                    wait_timeout_permission,
                                )
                                if task is not None
                            ]
                        )
                        if input_exhausted:
                            return
                        retry_connection = True
                        break

                    await utils.aio.gracefully_cancel(
                        *[
                            task
                            for task in (
                                send,
                                recv,
                                keepalive,
                                wait_switch,
                                wait_reconnect,
                                wait_user_state,
                                wait_timeout_permission,
                            )
                            if task is not None
                        ]
                    )
                    return

                if retry_connection:
                    continue
            except asyncio.CancelledError:
                cancel_final_timeout()
                tasks = [
                    t
                    for t in (
                        send,
                        recv,
                        keepalive,
                        wait_switch,
                        wait_reconnect,
                        wait_user_state,
                        wait_timeout_permission,
                    )
                    if t is not None
                ]
                if tasks:
                    with contextlib.suppress(Exception, asyncio.CancelledError):
                        await utils.aio.gracefully_cancel(*tasks)
                raise
            except Exception as exc:
                cancel_final_timeout()
                tasks = [
                    t
                    for t in (
                        send,
                        recv,
                        keepalive,
                        wait_switch,
                        wait_reconnect,
                        wait_user_state,
                        wait_timeout_permission,
                    )
                    if t is not None
                ]
                if tasks:
                    with contextlib.suppress(Exception):
                        await utils.aio.gracefully_cancel(*tasks)
                if (
                    not is_non_retryable_client_error(exc)
                    and candidate_attempts < self._candidate_max_retry
                    and not awaiting_final
                    and not buffered_audio
                    and pending_replay is None
                ):
                    candidate_attempts += 1
                    continue
                if is_payload_too_large(exc):
                    # Every candidate receives the same oversized body, so
                    # failing over cannot help; surface immediately.
                    self._slng_stt._emit_plugin_event(
                        "fallback.exhausted",
                        "error",
                        from_model=current_model(),
                        reason="hard_fail",
                        terminal_status=413,
                        **error_details(exc),
                    )
                    raise exc
                if not await failover(reason="hard_fail", exc=exc, timeout_s=None):
                    return
                continue
            finally:
                if ws is not None:
                    if not protocol_close_sent:
                        with contextlib.suppress(Exception):
                            await send_close(ws, reason="shutdown")
                    await ws.close()

    async def _connect_ws(
        self, *, model_endpoint: str, model: str | None
    ) -> aiohttp.ClientWebSocketResponse:
        # Match e2e test headers exactly - send both Authorization and X-API-Key
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "X-API-Key": self._api_token,
        }
        headers.update(self._extra_headers)
        connection = self._connections[self._active_endpoint_index]
        headers.update(connection.headers)

        # Don't enable compression - e2e tests work without it and compress=15
        # was causing handshake errors with Deepgram Nova endpoint
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    model_endpoint,
                    headers=headers,
                ),
                self._conn_options.timeout,
            )
        except (TimeoutError, aiohttp.ClientConnectorError) as e:
            raise APIConnectionError("failed to connect to SLNG STT") from e

        if connection.init is not None:
            init_message = dict(connection.init)
        else:
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
            await self._wait_for_bridge_ready(ws)
        except Exception:
            await ws.close()
            raise
        return ws

    async def _wait_for_bridge_ready(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            try:
                msg = await asyncio.wait_for(
                    ws.receive(),
                    timeout=self._conn_options.timeout,
                )
            except TimeoutError as exc:
                raise APIConnectionError("timed out waiting for SLNG STT ready") from exc

            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                raise APIConnectionError("SLNG STT closed before ready")

            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue

            msg_type = data.get("type")
            if msg_type == "ready":
                self._emit_session_event(data)
                return
            if msg_type in ("Error", "error"):
                message = _extract_error_text(data)
                status_code = _extract_error_code(data)
                if status_code is not None:
                    raise APIStatusError(
                        f"SLNG STT error: {message}",
                        status_code=status_code,
                    )
                raise APIStatusError(f"SLNG STT error: {message}")

    def _emit_session_event(self, data: dict[str, Any]) -> None:
        self._slng_stt._emit_plugin_event(
            "gateway.session",
            gateway_request_id=(
                data.get("slng_request_id")
                if isinstance(data.get("slng_request_id"), str)
                else None
            ),
            gateway_session_id=(
                data.get("session_id") if isinstance(data.get("session_id"), str) else None
            ),
        )
