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
import base64
import binascii
import contextlib
import json
import os
import time
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import JsonValue, TypeAdapter, ValidationError

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.metrics import TTSMetrics
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .connection import (
    PluginEvent,
    TTSConnectionConfig,
    bridge_endpoint,
    bridge_model,
)
from .gateway_adapter import (
    build_external_tracking_headers,
    build_tts_init_payload,
    extract_error_status,
    normalize_region_override,
    normalize_world_part_override,
)
from .log import logger

NUM_CHANNELS = 1
WS_CLOSE_TIMEOUT_S = 1.0
_FLUSH_MSG = json.dumps({"type": "flush"})
_MESSAGE = TypeAdapter(dict[str, JsonValue])

# ElevenLabs generates one audio fragment per incoming text frame when
# auto_mode is enabled, so single-word frames produce fragmented (choppy,
# slow-to-complete) audio. Batch words into phrases before sending: flush on
# clause/sentence punctuation or once the buffer reaches this many characters.
_PHRASE_FLUSH_SUFFIXES = (".", "!", "?", ",", ";", ":")


async def _close_ws(ws: http.WebSocket, *, context: str) -> None:
    try:
        await asyncio.wait_for(ws.close(), timeout=WS_CLOSE_TIMEOUT_S)
    except TimeoutError:
        logger.warning(
            "[TTS] websocket close timed out",
            extra={"context": context, "timeout_s": WS_CLOSE_TIMEOUT_S},
        )
    except Exception:
        logger.warning("[TTS] websocket close failed", extra={"context": context}, exc_info=True)


def _extract_audio_b64(resp: dict[str, JsonValue]) -> str | None:
    data = resp.get("data")
    if isinstance(data, str) and data:
        return data
    if isinstance(data, dict):
        nested_audio = data.get("audio")
        if isinstance(nested_audio, str) and nested_audio:
            return nested_audio

    audio = resp.get("audio")
    if isinstance(audio, str) and audio:
        return audio

    return None


def _extract_error_message(resp: dict[str, JsonValue]) -> str:
    data = resp.get("data")
    if isinstance(data, dict):
        nested = data.get("message") or data.get("description") or data.get("error")
        if isinstance(nested, str) and nested:
            return nested

    top_level = resp.get("message") or resp.get("description") or resp.get("error")
    if isinstance(top_level, str) and top_level:
        return top_level

    return "Unknown error"


def _extract_error_status(resp: dict[str, JsonValue]) -> int | None:
    """Extract an HTTP status from an error frame's int/numeric/string code."""
    return extract_error_status(resp)


def _contains_letter(text: str) -> bool:
    """True if the text has at least one alphabetic character in any script.

    ``str.isalpha`` covers every Unicode letter — Latin, Devanagari, Bengali,
    Tamil, etc. — so this is script-agnostic. Tokens with only punctuation,
    digits, whitespace, or symbols (e.g. "—", "4.5", "5,000") return False.
    Some providers (notably Sarvam Bulbul) reject a per-token WebSocket frame
    that carries no allowed-language character, so such tokens must be merged
    into a neighbouring word rather than sent on their own.
    """
    return any(ch.isalpha() for ch in text)


@dataclass(frozen=True)
class _ReceivedWsEvent:
    kind: Literal["audio_chunk", "audio_end", "error", "ignore", "unknown"]
    audio: bytes | None = None
    error: str | None = None
    error_status: int | None = None


def _tts_error(status: int | None, detail: str | None) -> APIError:
    """An error frame: its status's kind, or a refused request when it has none."""
    if status is None:
        return APIError("SLNG TTS error", kind="invalid_request", detail=detail)
    return APIStatusError("SLNG TTS error", status_code=status, detail=detail)


def _decode_audio(audio_b64: str) -> bytes:
    try:
        return base64.b64decode(audio_b64, validate=True)
    except binascii.Error as e:
        raise APIError("SLNG sent malformed audio", kind="protocol") from e


def _decode_audio_payload(resp: dict[str, JsonValue]) -> bytes | None:
    audio_b64 = _extract_audio_b64(resp)
    return _decode_audio(audio_b64) if audio_b64 else None


def _normalize_ws_message_type(resp: dict[str, JsonValue]) -> str | None:
    mtype = resp.get("type")
    if not isinstance(mtype, str):
        return None

    if mtype in ("Metadata", "Open", "control_ack", "ready"):
        return None
    if mtype == "Flushed":
        return "audio_end"
    if mtype in ("audio_chunk", "Audio", "audio", "chunk"):
        return "audio_chunk"
    if mtype == "done":
        return "audio_end"
    if mtype in ("Error", "error"):
        return "error"
    if mtype == "event":
        data = resp.get("data")
        if isinstance(data, dict):
            event_name = data.get("event") or data.get("event_type")
            if isinstance(event_name, str):
                normalized = event_name.strip().lower()
                if normalized in {"complete", "completed", "done", "end", "final"}:
                    return "audio_end"
                if normalized in {"error", "failed"}:
                    return "error"
    return mtype


def _parse_ws_event(resp: dict[str, JsonValue]) -> _ReceivedWsEvent:
    mtype = _normalize_ws_message_type(resp)
    if mtype is None:
        return _ReceivedWsEvent(kind="ignore")
    if mtype == "audio_chunk":
        return _ReceivedWsEvent(
            kind="audio_chunk",
            audio=_decode_audio_payload(resp),
        )
    if mtype in ("audio_end", "end", "flushed"):
        return _ReceivedWsEvent(
            kind="audio_end",
            audio=_decode_audio_payload(resp),
        )
    if mtype == "error":
        return _ReceivedWsEvent(
            kind="error",
            error=_extract_error_message(resp),
            error_status=_extract_error_status(resp),
        )
    return _ReceivedWsEvent(kind="unknown")


@dataclass
class _TTSOptions:
    model_endpoint: str
    model: str
    voice: str
    language: str
    sample_rate: int
    encoding: Literal["linear16"]
    speed: float
    api_key: str
    model_options: dict[str, JsonValue]
    extra_headers: dict[str, str]
    runtime_init: dict[str, JsonValue] | None
    warm_standby_enabled: bool
    text_chunking: Literal["auto", "word", "phrase"]
    phrase_max_chars: int


@dataclass
class _WsConnectionTiming:
    ws_connect_ms: float
    init_send_ms: float
    connect_total_ms: float


@dataclass
class _WarmStandbyConnection:
    ws: http.WebSocket
    standby_ready_ms: float
    # Settings epoch the socket was opened under; a checkout whose epoch no
    # longer matches the current one is stale (voice/language/speed changed).
    epoch: int


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000


class _Connection(tts.TTS[Literal["slng_event"]]):
    """One SLNG bridge connection; ``TTS`` chains them."""

    def __bool__(self) -> bool:
        # LiveKit Agents code may use truthiness checks like `tts or None`.
        # Some EventEmitter-style bases can be falsy when they have no listeners,
        # which would unintentionally disable TTS.
        return True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        connections: list[str | TTSConnectionConfig] | None = None,
        provider_api_key: str | None = None,
        voice: str,
        slng_base_url: str = "api.slng.ai",
        region_override: str | list[str] | None = None,
        world_part_override: str | None = None,
        external_agent_id: str | None = None,
        external_session_id: str | None = None,
        language: str = "en",
        sample_rate: int = 24000,
        speed: float = 1.0,
        http_session: http.HttpSession | None = None,
        extra_headers: dict[str, str] | None = None,
        # Advanced / optional. Used by integrations that drive the session
        # themselves; a typical client can ignore these.
        runtime_init: dict[str, JsonValue] | None = None,
        warm_standby_enabled: bool = False,
        text_chunking: Literal["auto", "word", "phrase"] = "auto",
        phrase_max_chars: int = 60,
        **model_options: JsonValue,
    ) -> None:
        """
        Create a new instance of SLNG TTS.

        Args:
            api_key (str): SLNG API key. Falls back to the ``SLNG_API_KEY`` env var.
            model: Unmute Bridge model identifier, e.g. "deepgram/aura:2".
            connections: Ordered model, endpoint, typed config, or SLNG TTS candidates.
            provider_api_key: Optional BYOK provider credential, sent as the
                ``X-Slng-Provider-Key`` header (external providers only).
            slng_base_url (str): Gateway host. Defaults to "api.slng.ai".
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
            voice: Required voice identifier.
            language (str): Language code. Defaults to "en".
            sample_rate (int): Sample rate of audio. Defaults to 24000.
            http_session (http.HttpSession): Optional HttpSession to use for requests.
        """
        if {"first_audio_timeout_s", "fallback_recovery_cooldown_s"} & model_options.keys():
            raise TypeError(
                "Configure timeouts with APIConnectOptions and fallback with tts.FallbackAdapter"
            )
        resolved_key = api_key or os.environ.get("SLNG_API_KEY")
        if not resolved_key:
            raise ValueError("api_key is required, or set the SLNG_API_KEY environment variable")

        if not voice.strip():
            raise ValueError("voice is required")
        if text_chunking not in {"auto", "word", "phrase"}:
            raise ValueError("text_chunking must be 'auto', 'word', or 'phrase'")
        if phrase_max_chars <= 0:
            raise ValueError("phrase_max_chars must be positive")
        if model is not None and connections:
            raise ValueError("use model or connections, not both")

        raw_connections: list[str | TTSConnectionConfig]
        if connections:
            raw_connections = connections
        elif model is not None:
            raw_connections = [model]
        else:
            raise ValueError("model or connections is required")

        primary = raw_connections[0]
        if isinstance(primary, TTSConnectionConfig):
            resolved_model_endpoint = primary.endpoint
            endpoint_model = bridge_model(primary.endpoint, "tts")
            if primary.model is not None and primary.model != endpoint_model:
                raise ValueError("TTS connection model must match its endpoint")
            model = endpoint_model
            voice = primary.voice or voice
        else:
            resolved_model_endpoint = (
                primary if "://" in primary else bridge_endpoint(slng_base_url, "tts", primary)
            )
            model = bridge_model(resolved_model_endpoint, "tts")
            primary = TTSConnectionConfig(
                endpoint=resolved_model_endpoint,
                model=model,
                voice=voice,
            )

        headers = dict(extra_headers or {})
        region_override_header = normalize_region_override(region_override)
        if region_override_header:
            headers.setdefault("X-Region-Override", region_override_header)
        world_part_header = normalize_world_part_override(world_part_override)
        if world_part_header:
            headers.setdefault("X-World-Part-Override", world_part_header)
        headers.update(primary.headers)
        headers.update(
            build_external_tracking_headers(
                external_agent_id=external_agent_id,
                external_session_id=external_session_id,
            )
        )
        if provider_api_key is not None:
            byok_key = provider_api_key.strip()
            if not byok_key:
                raise ValueError("provider_api_key must not be empty")
            headers["X-Slng-Provider-Key"] = byok_key

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, text_unit="words"),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        self._opts = _TTSOptions(
            model_endpoint=resolved_model_endpoint,
            model=model,
            voice=voice,
            language=language,
            sample_rate=sample_rate,
            # LiveKit expects raw PCM. Some SLNG models default to MP3 unless explicitly requested.
            encoding="linear16",
            speed=speed,
            api_key=resolved_key,
            model_options=dict(model_options),
            extra_headers=headers,
            runtime_init=(
                dict(primary.init)
                if primary.init is not None
                else dict(runtime_init)
                if runtime_init is not None
                else None
            ),
            warm_standby_enabled=warm_standby_enabled,
            text_chunking=text_chunking,
            phrase_max_chars=phrase_max_chars,
        )
        self._session = http_session
        # Set by the chain: a connection's voice is only updated at runtime when it was inherited
        # from the chain-level default rather than set explicitly per connection.
        self._inherits_voice = False
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._ws_connection_timings: dict[int, _WsConnectionTiming] = {}
        self._standby_lock = asyncio.Lock()
        self._standby: _WarmStandbyConnection | None = None
        # Bumped on every option change; invalidates any standby opened earlier.
        self._standby_epoch = 0
        self._standby_task: asyncio.Task[None] | None = None
        self._standby_close_tasks: set[asyncio.Task[None]] = set()
        self._tasks = utils.aio.TaskSet()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "SLNG"

    @property
    def warm_standby_enabled(self) -> bool:
        return self._opts.warm_standby_enabled

    def _is_ws_usable(self, ws: http.WebSocket) -> bool:
        return not bool(getattr(ws, "closed", False))

    async def _connect_ws(self, conn_options: APIConnectOptions) -> http.WebSocket:
        session = self._ensure_session()

        # Connect to WebSocket
        model_endpoint = self._opts.model_endpoint
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "X-API-Key": self._opts.api_key,
        }
        headers.update(self._opts.extra_headers)
        connect_started_at = time.perf_counter()
        ws = await session.ws_connect(model_endpoint, headers=headers, conn_options=conn_options)
        ws_connect_ms = _elapsed_ms(connect_started_at)

        # SLNG-specific: Send init and wait for ready
        init_payload = self._opts.runtime_init
        if init_payload is None:
            init_payload = build_tts_init_payload(
                model=self._opts.model,
                voice=self._opts.voice,
                language=self._opts.language,
                sample_rate=self._opts.sample_rate,
                encoding=self._opts.encoding,
                speed=self._opts.speed,
                model_options=self._opts.model_options,
            )

        try:
            init_started_at = time.perf_counter()
            await ws.send_str(json.dumps(init_payload))
            init_send_ms = _elapsed_ms(init_started_at)
        except Exception:
            await _close_ws(ws, context="init_failure")
            raise

        timing = _WsConnectionTiming(
            ws_connect_ms=ws_connect_ms,
            init_send_ms=init_send_ms,
            connect_total_ms=_elapsed_ms(connect_started_at),
        )
        self._ws_connection_timings[id(ws)] = timing
        logger.info(
            "[TTS] websocket connected",
            extra={
                "tts_model": self._opts.model,
                "tts_provider": self.provider,
                "tts_endpoint": model_endpoint,
                "ws_connect_ms": timing.ws_connect_ms,
                "init_send_ms": timing.init_send_ms,
                "connect_total_ms": timing.connect_total_ms,
            },
        )

        return ws

    def _start_standby_replenish(self, conn_options: APIConnectOptions) -> bool:
        if not self.warm_standby_enabled or self._tasks.closed:
            return False
        if self._standby is not None and self._is_ws_usable(self._standby.ws):
            return False
        if self._standby_task is not None and not self._standby_task.done():
            return False

        async def _open_standby() -> None:
            async with self._standby_lock:
                if self._standby is not None and self._is_ws_usable(self._standby.ws):
                    return
                epoch = self._standby_epoch
                standby_started_at = time.perf_counter()
                ws: http.WebSocket | None = None
                try:
                    ws = await self._connect_ws(conn_options)
                    if epoch != self._standby_epoch:
                        # Options changed while connecting; this socket carries
                        # the old init payload, so discard it rather than
                        # installing a stale standby.
                        return
                    self._standby = _WarmStandbyConnection(
                        ws=ws,
                        standby_ready_ms=_elapsed_ms(standby_started_at),
                        epoch=epoch,
                    )
                    ws = None
                finally:
                    if ws is not None:
                        self._ws_connection_timings.pop(id(ws), None)
                        await _close_ws(ws, context="warm_standby_replenish_cancelled")

        task = self._tasks.spawn(_open_standby())
        self._standby_task = task

        def _log_standby_failure(done: asyncio.Task[None]) -> None:
            if done.cancelled():
                return
            exc = done.exception()
            if exc is None:
                return
            logger.warning(
                "[TTS] warm standby replenish failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

        task.add_done_callback(_log_standby_failure)
        return True

    async def _checkout_standby(
        self,
    ) -> tuple[http.WebSocket | None, float | None, str | None]:
        if not self.warm_standby_enabled:
            return None, None, "disabled"

        if self._standby_task is not None and self._standby_task.done():
            with contextlib.suppress(Exception):
                self._standby_task.result()
            self._standby_task = None

        async with self._standby_lock:
            standby = self._standby
            if standby is None:
                if self._standby_task is not None and not self._standby_task.done():
                    return None, None, "standby_pending"
                return None, None, "standby_empty"
            if standby.epoch != self._standby_epoch:
                # Opened before an option change: never hand it out. Leave it
                # for the scheduled _close_standby and open a fresh connection.
                return None, None, "standby_stale"
            self._standby = None
            if not self._is_ws_usable(standby.ws):
                return None, None, "standby_closed"
            return standby.ws, standby.standby_ready_ms, None

    async def _close_standby(self) -> None:
        # Cancel the opener first (outside the lock, since it holds the lock
        # while connecting) so it cannot install a socket after we clear state.
        task = self._standby_task
        self._standby_task = None
        if task is not None:
            await utils.aio.cancel_and_wait(task)
        async with self._standby_lock:
            standby = self._standby
            self._standby = None
        if standby is not None:
            self._ws_connection_timings.pop(id(standby.ws), None)
            await _close_ws(standby.ws, context="warm_standby_close")

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            voice (str): Voice to use.
            language (str): Language code.
            speed (float): Playback speed multiplier.
            sentence_tokenizer (SentenceTokenizer): What splits the text into the chunks the TTS
                is sent.
            pacing (SentenceStreamPacer | None): A ``SentenceStreamPacer``, or ``None`` to send each
                chunk as soon as it is split.
        """
        super().update_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing)
        invalidate_pool = False
        if is_given(voice):
            if not voice.strip():
                raise ValueError("voice is required")
            invalidate_pool = invalidate_pool or self._opts.voice != voice
            self._opts.voice = voice
        if is_given(language):
            invalidate_pool = invalidate_pool or self._opts.language != language
            self._opts.language = language
        if is_given(speed):
            invalidate_pool = invalidate_pool or self._opts.speed != speed
            self._opts.speed = speed

        # Warm-standby sockets were initialized with the old voice/language.
        # Bump the epoch synchronously so any already-open standby is treated as
        # stale on checkout immediately (no await here, so no checkout can
        # interleave), then drop the socket in the background.
        if invalidate_pool:
            self._standby_epoch += 1
        if invalidate_pool and (self._standby is not None or self._standby_task is not None):
            with contextlib.suppress(RuntimeError):
                close_task = self._tasks.spawn(self._close_standby())
                self._standby_close_tasks.add(close_task)
                close_task.add_done_callback(self._standby_close_tasks.discard)

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=chunk.text,
            conn_options=conn_options,
            options=self._opts,
            connect=self._connect_ws,
            connection_timings=self._ws_connection_timings,
        )

    def _stream(self, *, conn_options: APIConnectOptions) -> tts.SynthesizeStream:
        stream = SynthesizeStream(
            tts=self,
            conn_options=conn_options,
            options=self._opts,
            connect=self._connect_ws,
            connection_timings=self._ws_connection_timings,
            checkout_standby=self._checkout_standby,
            replenish_standby=self._start_standby_replenish,
        )
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        if self.warm_standby_enabled:
            self._start_standby_replenish(conn_options=DEFAULT_API_CONNECT_OPTIONS)

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        # A detached standby remains owned until its bounded close settles.
        for close_task in list(self._standby_close_tasks):
            with contextlib.suppress(Exception):
                await asyncio.shield(close_task)
        self._standby_close_tasks.clear()
        await self._close_standby()
        await self._tasks.aclose()


class TTS(tts.FallbackAdapter):
    """SLNG TTS over one or more bridge connections, in order of preference: a failed one is
    skipped until it recovers. Its events (``error``, ``tts_availability_changed``,
    ``metrics_collected``, ``slng_event``) come from this object."""

    def __bool__(self) -> bool:
        # LiveKit Agents code may use truthiness checks like `tts or None`.
        return True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        connections: list[str | TTSConnectionConfig] | None = None,
        provider_api_key: str | None = None,
        voice: str,
        slng_base_url: str = "api.slng.ai",
        region_override: str | list[str] | None = None,
        world_part_override: str | None = None,
        external_agent_id: str | None = None,
        external_session_id: str | None = None,
        language: str = "en",
        sample_rate: int = 24000,
        speed: float = 1.0,
        http_session: http.HttpSession | None = None,
        extra_headers: dict[str, str] | None = None,
        runtime_init: dict[str, JsonValue] | None = None,
        warm_standby_enabled: bool = False,
        text_chunking: Literal["auto", "word", "phrase"] = "auto",
        phrase_max_chars: int = 60,
        **model_options: JsonValue,
    ) -> None:
        """
        Create a new instance of SLNG TTS.

        Args:
            api_key (str): SLNG API key. Falls back to the ``SLNG_API_KEY`` env var.
            model: Unmute Bridge model identifier, e.g. "deepgram/aura:2".
            connections: Ordered model, endpoint, typed config, or SLNG TTS candidates.
            provider_api_key: Optional BYOK provider credential, sent as the
                ``X-Slng-Provider-Key`` header (external providers only).
            slng_base_url (str): Gateway host. Defaults to "api.slng.ai".
            region_override: Optional gateway region override, sent as the
                ``X-Region-Override`` header. Accepts a single region or a list
                of preferred regions in priority order.
            world_part_override: Optional gateway world-part override, sent as
                the ``X-World-Part-Override`` header.
            external_agent_id: Optional tracking ID attached to usage events as
                the ``X-SLNG-Agent-Id`` header (max 128 chars).
            external_session_id: Optional tracking ID attached to usage events as
                the ``X-SLNG-Session-Id`` header (max 128 chars).
            voice: Required voice identifier.
            language (str): Language code. Defaults to "en".
            sample_rate (int): Sample rate of audio. Defaults to 24000.
            http_session (http.HttpSession): Optional HttpSession to use for requests.
        """
        if model is not None and connections:
            raise ValueError("use model or connections, not both")
        raw_connections: list[str | TTSConnectionConfig]
        if connections:
            raw_connections = connections
        elif model is not None:
            raw_connections = [model]
        else:
            raise ValueError("model or connections is required")

        self._connections: list[_Connection] = []
        for connection in raw_connections:
            config = (
                connection
                if isinstance(connection, TTSConnectionConfig)
                else TTSConnectionConfig(
                    endpoint=(
                        connection
                        if "://" in connection
                        else bridge_endpoint(slng_base_url, "tts", connection)
                    )
                )
            )
            candidate = _Connection(
                api_key=api_key,
                model=None,
                connections=[config],
                provider_api_key=provider_api_key,
                voice=config.voice or voice,
                slng_base_url=slng_base_url,
                region_override=region_override,
                world_part_override=world_part_override,
                external_agent_id=external_agent_id,
                external_session_id=external_session_id,
                language=language,
                sample_rate=sample_rate,
                speed=speed,
                http_session=http_session,
                extra_headers=extra_headers,
                runtime_init=runtime_init,
                warm_standby_enabled=warm_standby_enabled,
                text_chunking=text_chunking,
                phrase_max_chars=phrase_max_chars,
                **model_options,
            )
            candidate._inherits_voice = config.voice is None
            self._connections.append(candidate)
        super().__init__(list(self._connections), sample_rate=sample_rate)
        for candidate in self._connections:
            candidate.on("metrics_collected", self._forward_metrics)
            candidate.on("slng_event", self._forward_plugin_event)

    def _forward_metrics(self, metrics: TTSMetrics) -> None:
        self.emit("metrics_collected", metrics)

    def _forward_plugin_event(self, event: PluginEvent) -> None:
        self.emit("slng_event", event)  # type: ignore[arg-type]  # the plugin's own event

    @property
    def provider(self) -> str:
        return "SLNG"

    @property
    def warm_standby_enabled(self) -> bool:
        return self._connections[0].warm_standby_enabled

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """Update every connection, so a later failover does not synthesize with
        construction-time settings; a connection with its own voice keeps it.

        Args:
            voice (str): Voice to use.
            language (str): Language code.
            speed (float): Playback speed multiplier.
            sentence_tokenizer (SentenceTokenizer): What splits the text into the chunks the TTS
                is sent.
            pacing (SentenceStreamPacer | None): A ``SentenceStreamPacer``, or ``None`` to send each
                chunk as soon as it is split.
        """
        super().update_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing)
        for candidate in self._connections:
            candidate.update_options(
                voice=voice if is_given(voice) and candidate._inherits_voice else NOT_GIVEN,
                language=language,
                speed=speed,
            )

    async def aclose(self) -> None:
        await super().aclose()
        for candidate in self._connections:
            await candidate.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Non-streaming synthesis: send the full text once over the SLNG WebSocket."""

    def __init__(
        self,
        *,
        tts: _Connection,
        input_text: str,
        conn_options: APIConnectOptions,
        options: _TTSOptions,
        connect: Callable[[APIConnectOptions], Awaitable[http.WebSocket]],
        connection_timings: dict[int, _WsConnectionTiming],
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._connect = connect
        self._connection_timings = connection_timings
        self._opts = replace(options)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        ws: http.WebSocket | None = None
        audio_received = False

        try:
            # Chunked synthesis always uses a dedicated connection: terminal
            # protocols close the socket after the final audio.
            ws = await self._connect(self._conn_options)
            if self._input_text:
                await ws.send_str(json.dumps({"type": "text", "text": self._input_text}))
            await ws.send_str(_FLUSH_MSG)

            while True:
                msg = await ws.receive(timeout=self._conn_options.read_timeout)
                if msg.type is http.WebSocketMessageType.CLOSE:
                    # Several bridge providers (Rime, Cartesia) terminate a
                    # segment by closing the socket. Mirror the streaming path:
                    # a close after audio is a normal end-of-segment, not an
                    # error. Only a close with no audio at all is a failure.
                    if audio_received:
                        output_emitter.flush()
                        break
                    raise ws.close_error("SLNG websocket closed unexpectedly")

                if msg.type is http.WebSocketMessageType.BINARY:
                    output_emitter.push(msg.binary)
                    audio_received = True
                    continue

                try:
                    resp = _MESSAGE.validate_json(msg.data)
                except ValidationError:
                    logger.debug(
                        "[SLNG TTS] ignoring non-JSON text frame",
                        extra={"lk.pii.data": msg.data},
                    )
                    continue

                if "type" not in resp:
                    # Raw provider passthrough (e.g. ElevenLabs): {"audio", "isFinal"}.
                    is_final_value = resp.get("isFinal")
                    is_final = (
                        is_final_value is True
                        or is_final_value == 1
                        or (
                            isinstance(is_final_value, str)
                            and is_final_value.strip().lower() in ("true", "1")
                        )
                    )
                    audio_b64 = resp.get("audio")
                    if isinstance(audio_b64, str) and audio_b64:
                        output_emitter.push(_decode_audio(audio_b64))
                        audio_received = True

                    if is_final:
                        output_emitter.flush()
                        break

                    if resp.get("error") is not None:
                        raise _tts_error(_extract_error_status(resp), str(resp.get("error")))
                    continue

                event = _parse_ws_event(resp)
                if event.kind == "ignore":
                    continue
                if event.kind == "audio_chunk":
                    if event.audio:
                        output_emitter.push(event.audio)
                        audio_received = True
                elif event.kind == "audio_end":
                    if event.audio:
                        output_emitter.push(event.audio)
                        audio_received = True
                    output_emitter.flush()
                    break
                elif event.kind == "error":
                    raise _tts_error(event.error_status, event.error)
                else:
                    logger.debug("[TTS] ignoring unknown message", extra={"lk.pii.data": resp})
        except TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if ws is not None:
                self._connection_timings.pop(id(ws), None)
                await _close_ws(ws, context="chunked_synthesis")


class SynthesizeStream(tts.SynthesizeStream):
    # SLNG protocol messages (different from Deepgram)
    _CANCEL_MSG: str = json.dumps({"type": "cancel"})
    _CLOSE_MSG: str = json.dumps({"type": "close"})

    def __init__(
        self,
        *,
        tts: _Connection,
        conn_options: APIConnectOptions,
        options: _TTSOptions,
        connect: Callable[[APIConnectOptions], Awaitable[http.WebSocket]],
        connection_timings: dict[int, _WsConnectionTiming],
        checkout_standby: Callable[
            [], Awaitable[tuple[http.WebSocket | None, float | None, str | None]]
        ],
        replenish_standby: Callable[[APIConnectOptions], bool],
    ) -> None:
        logger.debug("[TTS] SynthesizeStream.__init__ STARTING")
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: _Connection = tts
        self._connect = connect
        self._connection_timings = connection_timings
        self._checkout_standby = checkout_standby
        self._replenish_standby = replenish_standby
        self._opts = replace(options)
        logger.debug("[TTS] SynthesizeStream.__init__ DONE")

    def _emit_plugin_event(
        self,
        name: str,
        level: Literal["info", "warning", "error"] = "info",
        **data: object,
    ) -> None:
        self._tts.emit(
            "slng_event",
            PluginEvent(name=name, component="tts", level=level, data=data),
        )

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # Create segments_ch per run so base-class retries after an error get a
        # fresh channel (matching the Deepgram plugin pattern).
        self._segments_ch = utils.aio.Chan[utils.aio.Chan[str]]()
        request_id = utils.shortuuid()
        logger.debug(f"[TTS] _run starting: request_id={request_id}")
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _segment_input() -> None:
            segment: utils.aio.Chan[str] | None = None
            text_count = 0
            async for the_input in self.input:
                if isinstance(the_input, tts.SpeechChunk):
                    text_count += 1
                    if text_count == 1:
                        logger.debug(
                            "[TTS] First text received", extra={"lk.pii.text": the_input.text}
                        )
                    if segment is None:
                        segment = utils.aio.Chan[str]()
                        self._segments_ch.send_nowait(segment)
                    segment.send_nowait(the_input.text)
                else:
                    logger.debug(f"[TTS] Flush sentinel received after {text_count} texts")
                    if segment is not None:
                        segment.close()
                    segment = None
            if segment is not None:
                segment.close()
            logger.debug(f"[TTS] _segment_input done: {text_count} total texts")
            self._segments_ch.close()

        async def _run_segments() -> None:
            segment_count = 0
            async for segment in self._segments_ch:
                segment_count += 1
                logger.debug(f"[TTS] Processing segment {segment_count}")
                await self._run_ws(segment, output_emitter)
            logger.debug(f"[TTS] _run_segments done: {segment_count} segments")

        try:
            async with utils.aio.TaskSet() as tasks:
                await asyncio.gather(tasks.spawn(_segment_input()), tasks.spawn(_run_segments()))
        except TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

    async def _run_ws(self, segment: utils.aio.Chan[str], output_emitter: tts.AudioEmitter) -> None:
        segment_id = utils.shortuuid()
        logger.debug(f"[TTS] _run_ws starting: segment_id={segment_id}")
        segment_started_at = time.perf_counter()
        output_emitter.start_segment(segment_id=segment_id)
        input_sent_event = asyncio.Event()
        phrase_batching = self._opts.text_chunking in {"auto", "phrase"}
        outcome = "completed"
        ws_connect_ms: float | None = None
        init_send_ms: float | None = None
        connect_total_ms: float | None = None
        ready_ms: float | None = None
        first_text_send_ms: float | None = None
        first_audio_ms: float | None = None
        audio_end_ms: float | None = None
        close_ms: float | None = None
        close_timed_out = False
        gateway_request_id: str | None = None
        gateway_session_id: str | None = None
        audio_chunks_seen = 0
        standby_enabled = self._opts.warm_standby_enabled
        standby_used = False
        standby_ready_ms: float | None = None
        standby_miss_reason: str | None = None
        standby_replenish_started = False
        standby_replenish_requested = False

        def capture_ws_timing(ws: http.WebSocket) -> None:
            nonlocal ws_connect_ms, init_send_ms, connect_total_ms
            timing = self._connection_timings.pop(id(ws), None)
            if timing is None:
                return
            ws_connect_ms = timing.ws_connect_ms
            init_send_ms = timing.init_send_ms
            connect_total_ms = timing.connect_total_ms

        def mark_first_text_sent() -> None:
            nonlocal first_text_send_ms
            if first_text_send_ms is None:
                first_text_send_ms = _elapsed_ms(segment_started_at)

        def mark_first_audio_seen() -> None:
            nonlocal first_audio_ms
            self._core_attempts.audio_received()
            if first_audio_ms is None:
                first_audio_ms = _elapsed_ms(segment_started_at)

        def request_standby_replenish() -> None:
            nonlocal standby_replenish_requested, standby_replenish_started
            if not standby_enabled or standby_replenish_requested:
                return
            standby_replenish_requested = True
            standby_replenish_started = self._replenish_standby(self._conn_options)

        def log_segment_timing() -> None:
            logger.info(
                "[TTS] segment timing",
                extra={
                    "tts_model": self._opts.model,
                    "tts_provider": self._tts.provider,
                    "tts_endpoint": self._opts.model_endpoint,
                    "segment_id": segment_id,
                    "outcome": outcome,
                    "ws_connect_ms": ws_connect_ms,
                    "init_send_ms": init_send_ms,
                    "connect_total_ms": connect_total_ms,
                    "ready_ms": ready_ms,
                    "first_text_send_ms": first_text_send_ms,
                    "first_audio_ms": first_audio_ms,
                    "audio_end_ms": audio_end_ms,
                    "segment_total_ms": _elapsed_ms(segment_started_at),
                    "close_ms": close_ms,
                    "close_timed_out": close_timed_out,
                    "audio_chunks": audio_chunks_seen,
                    "gateway_request_id": gateway_request_id,
                    "gateway_session_id": gateway_session_id,
                    "standby_enabled": standby_enabled,
                    "standby_used": standby_used,
                    "standby_ready_ms": standby_ready_ms,
                    "standby_miss_reason": standby_miss_reason,
                    "standby_replenish_started": standby_replenish_started,
                },
            )

        async def send_task(ws: http.WebSocket) -> None:
            word_count = 0

            async def _emit_text(text: str) -> None:
                # SLNG: Use "text" type instead of "Speak"
                self.mark_started()
                await ws.send_str(json.dumps({"type": "text", "text": text}))
                mark_first_text_sent()
                request_standby_replenish()
                input_sent_event.set()

            # Buffer one spoken word so that letterless tokens (dashes, bare
            # numbers, stray punctuation) attach to a neighbouring word instead
            # of being sent as their own frame. A provider like Sarvam Bulbul
            # rejects a text frame with no allowed-language character, which
            # otherwise hard-fails the whole segment.
            text_buffer = ""
            buffer_has_letter = False

            async for chunk in segment:
                word_count += 1
                if word_count == 1:
                    logger.debug("[TTS] send_task: first word", extra={"lk.pii.text": chunk})
                piece = f"{chunk} "
                if phrase_batching:
                    text_buffer += piece
                    buffer_has_letter = buffer_has_letter or _contains_letter(chunk)
                    stripped = text_buffer.rstrip()
                    at_boundary = stripped and (
                        stripped.endswith(_PHRASE_FLUSH_SUFFIXES)
                        or len(stripped) >= self._opts.phrase_max_chars
                    )
                    if at_boundary and buffer_has_letter:
                        await _emit_text(text_buffer)
                        text_buffer = ""
                        buffer_has_letter = False
                    # A letterless buffer (bare numbers, punctuation) is kept at
                    # a boundary so it joins the next phrase instead of being
                    # dropped.
                    continue

                if not text_buffer:
                    text_buffer = piece
                    buffer_has_letter = _contains_letter(chunk)
                elif _contains_letter(chunk):
                    # A new spoken word arrived. Flush the buffered word (with any
                    # letterless tokens that trailed it); if the buffer so far is
                    # only leading punctuation, prepend it to this word instead.
                    if buffer_has_letter:
                        await _emit_text(text_buffer)
                        text_buffer = piece
                    else:
                        text_buffer += piece
                    buffer_has_letter = True
                else:
                    # Letterless token: keep it with the buffered word.
                    text_buffer += piece

            # Flush whatever remains. A buffer with no letter at all (a segment of
            # pure punctuation, or trailing punctuation with no following word) is
            # dropped — sending it alone would be rejected and carries no audio.
            if text_buffer and buffer_has_letter:
                await _emit_text(text_buffer)

            logger.debug(f"[TTS] send_task: sent {word_count} words, flushing")
            await ws.send_str(_FLUSH_MSG)
            input_sent_event.set()
            # the read deadline runs only from here: before, a pause in the text owes nothing
            self._core_attempts.expect_response(final=True)

        async def recv_task(ws: http.WebSocket) -> None:
            nonlocal ready_ms, audio_end_ms, gateway_request_id, gateway_session_id
            nonlocal audio_chunks_seen
            await input_sent_event.wait()
            while True:
                msg = await ws.receive()
                if msg.type is http.WebSocketMessageType.CLOSE:
                    if audio_chunks_seen > 0:
                        logger.info(
                            "[TTS] recv_task: websocket closed after terminal "
                            "model output (chunks=%s)",
                            audio_chunks_seen,
                        )
                        audio_end_ms = audio_end_ms or _elapsed_ms(segment_started_at)
                        output_emitter.end_segment()
                        break
                    raise ws.close_error("SLNG websocket closed unexpectedly")

                # SLNG: Handle both binary (legacy) and JSON audio_chunk messages
                if msg.type is http.WebSocketMessageType.BINARY:
                    audio_chunks_seen += 1
                    mark_first_audio_seen()
                    output_emitter.push(msg.binary)
                else:
                    try:
                        resp = _MESSAGE.validate_json(msg.data)
                    except ValidationError:
                        logger.debug(
                            "[SLNG TTS] ignoring non-JSON text frame",
                            extra={"lk.pii.data": msg.data},
                        )
                        continue

                    if resp.get("type") == "ready":
                        if ready_ms is None:
                            ready_ms = _elapsed_ms(segment_started_at)
                        raw_request_id = resp.get("slng_request_id")
                        if isinstance(raw_request_id, str):
                            gateway_request_id = raw_request_id
                        raw_session_id = resp.get("session_id")
                        if isinstance(raw_session_id, str):
                            gateway_session_id = raw_session_id
                        self._emit_plugin_event(
                            "gateway.session",
                            gateway_request_id=gateway_request_id,
                            gateway_session_id=gateway_session_id,
                        )
                        continue

                    if "type" not in resp:
                        is_final_value = resp.get("isFinal")
                        is_final = (
                            is_final_value is True
                            or is_final_value == 1
                            or (
                                isinstance(is_final_value, str)
                                and is_final_value.strip().lower() in ("true", "1")
                            )
                        )
                        audio_b64 = resp.get("audio")
                        if isinstance(audio_b64, str) and audio_b64:
                            audio = _decode_audio(audio_b64)
                            mark_first_audio_seen()
                            output_emitter.push(audio)
                            audio_chunks_seen += 1

                        if is_final:
                            audio_end_ms = audio_end_ms or _elapsed_ms(segment_started_at)
                            output_emitter.end_segment()
                            break

                        if resp.get("error") is not None:
                            raise _tts_error(_extract_error_status(resp), str(resp.get("error")))
                        continue

                    event = _parse_ws_event(resp)
                    if event.kind == "ignore":
                        continue

                    if event.kind == "audio_chunk":
                        if event.audio:
                            audio_chunks_seen += 1
                            mark_first_audio_seen()
                            output_emitter.push(event.audio)

                    # SLNG: "audio_end" or "end" instead of "Flushed"
                    elif event.kind == "audio_end":
                        if event.audio:
                            audio_chunks_seen += 1
                            mark_first_audio_seen()
                            output_emitter.push(event.audio)
                        logger.debug(f"[TTS] recv_task: audio_end after {audio_chunks_seen} chunks")
                        audio_end_ms = audio_end_ms or _elapsed_ms(segment_started_at)
                        output_emitter.end_segment()
                        break

                    elif event.kind == "error":
                        raise _tts_error(event.error_status, event.error)

                    else:
                        logger.debug("[TTS] ignoring unknown message", extra={"lk.pii.data": resp})

        try:
            ws: http.WebSocket | None = None
            if standby_enabled:
                (
                    ws,
                    standby_ready_ms,
                    standby_miss_reason,
                ) = await self._checkout_standby()
                standby_used = ws is not None
            if ws is None:
                ws = await self._connect(self._conn_options)
            capture_ws_timing(ws)
            try:
                async with utils.aio.TaskSet() as tasks:
                    await asyncio.gather(tasks.spawn(send_task(ws)), tasks.spawn(recv_task(ws)))
            finally:
                input_sent_event.set()
                close_started_at = time.perf_counter()
                try:
                    await _close_ws(ws, context="terminal_model_segment")
                finally:
                    close_ms = _elapsed_ms(close_started_at)
                    close_timed_out = close_ms >= (WS_CLOSE_TIMEOUT_S * 1000)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception:
            outcome = "error"
            raise
        finally:
            log_segment_timing()
