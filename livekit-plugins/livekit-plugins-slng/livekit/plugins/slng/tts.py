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
import contextlib
import json
import os
import time
import weakref
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Any, Literal

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .connection import (
    CandidateState,
    PluginEvent,
    TTSConnectionConfig,
    bridge_endpoint,
    bridge_model,
)
from .gateway_adapter import (
    build_external_tracking_headers,
    build_tts_init_payload,
    extract_error_status,
    is_non_retryable_client_error,
    is_payload_too_large,
    normalize_region_override,
    normalize_world_part_override,
)
from .log import logger

NUM_CHANNELS = 1
WS_CLOSE_TIMEOUT_S = 1.0

# Text frames sent to the gateway are cut by `text_chunking`. The default,
# "sentence", relies on a sentence tokenizer and sends one whole sentence per
# frame, which sounds right whether the provider voices each frame as it
# arrives (Rime segment="immediate", ElevenLabs auto_mode) or buffers to the
# sentence itself. "phrase" is the previous behaviour: words re-batched at the
# punctuation below or once the buffer reaches `phrase_max_chars`. "word"
# sends every word.
_PHRASE_FLUSH_SUFFIXES = (".", "!", "?", ",", ";", ":")


async def _close_ws(ws: aiohttp.ClientWebSocketResponse, *, context: str) -> None:
    try:
        await asyncio.wait_for(ws.close(), timeout=WS_CLOSE_TIMEOUT_S)
    except (TimeoutError, asyncio.TimeoutError):
        logger.warning(
            "[TTS] websocket close timed out",
            extra={"context": context, "timeout_s": WS_CLOSE_TIMEOUT_S},
        )
    except Exception:
        logger.warning("[TTS] websocket close failed", extra={"context": context}, exc_info=True)


def _extract_audio_b64(resp: dict[str, object]) -> str | None:
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


def _extract_error_message(resp: dict[str, object]) -> str:
    data = resp.get("data")
    if isinstance(data, dict):
        nested = data.get("message") or data.get("description") or data.get("error")
        if isinstance(nested, str) and nested:
            return nested

    top_level = resp.get("message") or resp.get("description") or resp.get("error")
    if isinstance(top_level, str) and top_level:
        return top_level

    return "Unknown error"


def _extract_error_status(resp: dict[str, object]) -> int | None:
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


@dataclass
class _SegmentInput:
    """One reply's tokenizer stream plus the flag that lets the sender spot the last frame."""

    stream: tokenize.WordStream | tokenize.SentenceStream
    # Set by _tokenize_input immediately before it calls stream.end_input().
    # The tokenizer retains its last token until end_input(), so a token pulled
    # while this is False is never the reply's last; once True, every remaining
    # token is already queued in the closed channel and can be drained at once.
    input_ended: bool = False


class _FrameBatcher:
    """Groups tokenizer tokens into the text frames sent to the gateway.

    In phrase/sentence mode a frame is cut when the buffer ends with one of
    ``_PHRASE_FLUSH_SUFFIXES`` or reaches ``max_chars``; with a sentence
    tokenizer every token is already a whole sentence, so this yields one frame
    per sentence. In word mode every spoken word is a frame. A frame is never
    cut without a letter in it (see ``_contains_letter``): letterless tokens
    stay attached to a neighbouring word, because some providers reject a text
    frame that carries no allowed-language character.
    """

    def __init__(self, *, batch_phrases: bool, max_chars: int) -> None:
        self._batch_phrases = batch_phrases
        self._max_chars = max_chars
        self._buf = ""
        self._has_letter = False

    def push(self, token: str) -> str | None:
        """Add a token; return a completed frame when this token closes one."""
        piece = f"{token} "
        if self._batch_phrases:
            self._buf += piece
            self._has_letter = self._has_letter or _contains_letter(token)
            stripped = self._buf.rstrip()
            at_boundary = bool(stripped) and (
                stripped.endswith(_PHRASE_FLUSH_SUFFIXES) or len(stripped) >= self._max_chars
            )
            return self._take() if at_boundary and self._has_letter else None
        if not self._buf:
            self._buf = piece
            self._has_letter = _contains_letter(token)
            return None
        if _contains_letter(token):
            # A new spoken word arrived. Emit the buffered word (with any
            # letterless tokens that trailed it); if the buffer so far is only
            # leading punctuation, prepend it to this word instead.
            if self._has_letter:
                frame = self._take()
                self._buf = piece
                self._has_letter = True
                return frame
            self._buf += piece
            self._has_letter = True
            return None
        self._buf += piece
        return None

    def finish(self) -> str | None:
        """Return the trailing buffer as a frame, or None if it is empty or letterless."""
        if self._buf and self._has_letter:
            return self._take()
        self._buf, self._has_letter = "", False
        return None

    def _take(self) -> str:
        frame, self._buf, self._has_letter = self._buf, "", False
        return frame


@dataclass(frozen=True)
class _ReceivedWsEvent:
    kind: Literal["audio_chunk", "audio_end", "error", "ignore", "unknown"]
    audio: bytes | None = None
    error: str | None = None
    error_status: int | None = None


def _decode_audio_payload(resp: dict[str, object], *, context: str) -> bytes | None:
    audio_b64 = _extract_audio_b64(resp)
    if not audio_b64:
        return None
    try:
        return base64.b64decode(audio_b64)
    except Exception:
        logger.warning("[TTS] invalid base64 audio (%s)", context, exc_info=True)
        return None


def _normalize_ws_message_type(resp: dict[str, object]) -> str | None:
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


def _parse_ws_event(resp: dict[str, object]) -> _ReceivedWsEvent:
    mtype = _normalize_ws_message_type(resp)
    if mtype is None:
        return _ReceivedWsEvent(kind="ignore")
    if mtype == "audio_chunk":
        return _ReceivedWsEvent(
            kind="audio_chunk",
            audio=_decode_audio_payload(resp, context="audio_chunk"),
        )
    if mtype in ("audio_end", "end", "flushed"):
        return _ReceivedWsEvent(
            kind="audio_end",
            audio=_decode_audio_payload(resp, context="audio_end"),
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
    language: NotGivenOr[str]
    sample_rate: int
    encoding: Literal["linear16"]
    speed: NotGivenOr[float]
    word_tokenizer: tokenize.WordTokenizer | tokenize.SentenceTokenizer
    api_key: str
    model_options: dict[str, object]
    extra_headers: dict[str, str]
    runtime_init: dict[str, Any] | None
    warm_standby_enabled: bool
    text_chunking: Literal["sentence", "word", "phrase"]
    phrase_max_chars: int


@dataclass
class _WsConnectionTiming:
    ws_connect_ms: float
    init_send_ms: float
    connect_total_ms: float


@dataclass
class _WarmStandbyConnection:
    ws: aiohttp.ClientWebSocketResponse
    standby_ready_ms: float
    # Settings epoch the socket was opened under; a checkout whose epoch no
    # longer matches the current one is stale (voice/language/speed changed).
    epoch: int


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000


class TTS(tts.TTS):
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
        model_endpoint: str | None = None,
        provider_api_key: str | None = None,
        voice: str,
        slng_base_url: str = "api.slng.ai",
        region_override: str | list[str] | None = None,
        world_part_override: str | None = None,
        external_agent_id: str | None = None,
        external_session_id: str | None = None,
        language: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = 24000,
        speed: NotGivenOr[float] = NOT_GIVEN,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer | tokenize.SentenceTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        extra_headers: dict[str, str] | None = None,
        # Advanced / optional. Used by integrations that drive the session
        # themselves; a typical client can ignore these.
        runtime_init: dict[str, Any] | None = None,
        warm_standby_enabled: bool = False,
        text_chunking: Literal["auto", "sentence", "word", "phrase"] = "auto",
        phrase_max_chars: int = 60,
        first_audio_timeout_s: float | None = None,
        fallback_recovery_cooldown_s: float = 60.0,
        _candidate: bool = False,
        **model_options: Any,
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
            language: Optional language code. When omitted, the model's catalog
                default applies.
            speed: Optional speed multiplier. When omitted, the model's catalog
                default applies.
            sample_rate (int): Sample rate of audio. Defaults to 24000.
            word_tokenizer: Optional tokenizer for processing text. Defaults to
                ``tokenize.blingfire.SentenceTokenizer()`` in sentence mode and
                ``tokenize.basic.WordTokenizer(ignore_punctuation=False)`` otherwise.
            text_chunking: How LLM text is cut into gateway frames. ``"sentence"``
                (the default; ``"auto"`` resolves to it) sends one frame per
                sentence. ``"phrase"`` re-batches words at clause punctuation or
                every ``phrase_max_chars``. ``"word"`` sends one frame per word.
            http_session (aiohttp.ClientSession): Optional aiohttp session to use for requests.
        """
        if model_endpoint is not None:
            raise ValueError(
                "model_endpoint was removed in 2.0: the plugin always connects "
                "through the Unmute Bridge. Pass model='provider/model:variant' "
                "or connections=[...] instead."
            )
        resolved_key = api_key or os.environ.get("SLNG_API_KEY")
        if not resolved_key:
            raise ValueError("api_key is required, or set the SLNG_API_KEY environment variable")

        if not voice.strip():
            raise ValueError("voice is required")
        if text_chunking not in {"auto", "sentence", "word", "phrase"}:
            raise ValueError("text_chunking must be 'auto', 'sentence', 'word', or 'phrase'")
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
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        resolved_chunking: Literal["sentence", "word", "phrase"] = (
            "sentence" if text_chunking == "auto" else text_chunking
        )
        if not is_given(word_tokenizer):
            word_tokenizer = (
                tokenize.blingfire.SentenceTokenizer()
                if resolved_chunking == "sentence"
                else tokenize.basic.WordTokenizer(ignore_punctuation=False)
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
            word_tokenizer=word_tokenizer,
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
            text_chunking=resolved_chunking,
            phrase_max_chars=phrase_max_chars,
        )
        self._session = http_session
        self._first_audio_timeout_s = first_audio_timeout_s
        self._candidate_state = CandidateState(len(raw_connections), fallback_recovery_cooldown_s)
        self._active_candidate_index = 0
        self._candidate_tts: list[TTS] = [self]
        self._is_candidate = _candidate
        # Set by the parent chain on the candidates it constructs: a
        # candidate's voice is only updated at runtime when it was inherited
        # from the chain-level default rather than set explicitly per
        # candidate.
        self._inherits_voice = False
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._ws_connection_timings: dict[int, _WsConnectionTiming] = {}
        self._standby_lock = asyncio.Lock()
        self._standby: _WarmStandbyConnection | None = None
        # Bumped on every option change; invalidates any standby opened earlier.
        self._standby_epoch = 0
        self._standby_task: asyncio.Task[None] | None = None
        # Retain fire-and-forget standby-close tasks so the event loop cannot
        # garbage-collect them mid-run before the socket is actually closed.
        self._standby_close_tasks: set[asyncio.Task[None]] = set()

        if not _candidate:
            for fallback in raw_connections[1:]:
                config = (
                    fallback
                    if isinstance(fallback, TTSConnectionConfig)
                    else TTSConnectionConfig(
                        endpoint=(
                            fallback
                            if "://" in fallback
                            else bridge_endpoint(slng_base_url, "tts", fallback)
                        )
                    )
                )
                candidate = TTS(
                    api_key=resolved_key,
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
                    word_tokenizer=word_tokenizer,
                    http_session=http_session,
                    extra_headers=extra_headers,
                    runtime_init=runtime_init,
                    warm_standby_enabled=warm_standby_enabled,
                    text_chunking=resolved_chunking,
                    phrase_max_chars=phrase_max_chars,
                    first_audio_timeout_s=first_audio_timeout_s,
                    fallback_recovery_cooldown_s=fallback_recovery_cooldown_s,
                    _candidate=True,
                    **model_options,
                )
                candidate._inherits_voice = config.voice is None
                if (
                    candidate.sample_rate != self.sample_rate
                    or candidate.num_channels != self.num_channels
                ):
                    raise ValueError("all TTS candidates must use the same audio format")
                self._candidate_tts.append(candidate)
                candidate.on("metrics_collected", self._forward_metrics)
                candidate.on("slng_event", self._forward_plugin_event)

    def _forward_metrics(self, metrics: Any) -> None:
        self.emit("metrics_collected", metrics)

    def _forward_plugin_event(self, event: PluginEvent) -> None:
        self.emit("slng_event", event)

    def _emit_plugin_event(
        self,
        name: str,
        level: Literal["info", "warning", "error"] = "info",
        **data: Any,
    ) -> None:
        self.emit(
            "slng_event",
            PluginEvent(name=name, component="tts", level=level, data=data),
        )

    @property
    def model(self) -> str:
        return self._candidate_tts[self._active_candidate_index]._opts.model

    @property
    def provider(self) -> str:
        return "SLNG"

    @property
    def warm_standby_enabled(self) -> bool:
        return self._opts.warm_standby_enabled

    def _is_ws_usable(self, ws: aiohttp.ClientWebSocketResponse) -> bool:
        return not bool(getattr(ws, "closed", False))

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()

        # Connect to WebSocket
        model_endpoint = self._opts.model_endpoint
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "X-API-Key": self._opts.api_key,
        }
        headers.update(self._opts.extra_headers)
        connect_started_at = time.perf_counter()
        ws = await asyncio.wait_for(
            session.ws_connect(
                model_endpoint,
                headers=headers,
            ),
            timeout,
        )
        ws_connect_ms = _elapsed_ms(connect_started_at)

        # SLNG-specific: Send init and wait for ready
        init_payload = self._opts.runtime_init
        if init_payload is None:
            init_payload = build_tts_init_payload(
                model=self._opts.model,
                voice=self._opts.voice,
                sample_rate=self._opts.sample_rate,
                encoding=self._opts.encoding,
                language=self._opts.language if is_given(self._opts.language) else None,
                speed=self._opts.speed if is_given(self._opts.speed) else None,
                model_options=self._opts.model_options,
            )

        try:
            init_started_at = time.perf_counter()
            await ws.send_str(json.dumps(init_payload))
            init_send_ms = _elapsed_ms(init_started_at)
        except BaseException:
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

    def _start_standby_replenish(self, *, timeout: float) -> bool:
        if not self.warm_standby_enabled:
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
                ws: aiohttp.ClientWebSocketResponse | None = None
                try:
                    ws = await self._connect_ws(timeout=timeout)
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

        task = asyncio.create_task(_open_standby())
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
    ) -> tuple[aiohttp.ClientWebSocketResponse | None, float | None, str | None]:
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
            await utils.aio.gracefully_cancel(task)
        async with self._standby_lock:
            standby = self._standby
            self._standby = None
        if standby is not None:
            self._ws_connection_timings.pop(id(standby.ws), None)
            await _close_ws(standby.ws, context="warm_standby_close")

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            voice (str): Voice to use.
            language (str): Language code.
            speed (float): Playback speed multiplier.
        """
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
                close_task = asyncio.get_running_loop().create_task(self._close_standby())
                self._standby_close_tasks.add(close_task)
                close_task.add_done_callback(self._standby_close_tasks.discard)

        # Keep fallback candidates consistent with the primary so a later
        # failover does not synthesize with construction-time settings. A
        # candidate with an explicit per-candidate voice keeps it.
        if not self._is_candidate:
            for candidate in self._candidate_tts[1:]:
                candidate.update_options(
                    voice=voice if is_given(voice) and candidate._inherits_voice else NOT_GIVEN,
                    language=language,
                    speed=speed,
                )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        if not self._is_candidate:
            return _FallbackChunkedStream(
                parent=self,
                text=text,
                conn_options=conn_options,
            )  # type: ignore[return-value]
        return self._synthesize_candidate(text, conn_options=conn_options)

    def _synthesize_candidate(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        if not self._is_candidate:
            return _FallbackSynthesizeStream(
                parent=self,
                conn_options=conn_options,
            )  # type: ignore[return-value]
        return self._stream_candidate(conn_options=conn_options)

    def _stream_candidate(
        self,
        *,
        conn_options: APIConnectOptions,
    ) -> SynthesizeStream:
        logger.debug("[TTS] TTS.stream() called, creating SynthesizeStream")
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        logger.debug("[TTS] TTS.stream() returning stream")
        return stream

    def prewarm(self) -> None:
        if len(self._candidate_tts) > 1 and not self._is_candidate:
            active = self._candidate_tts[self._candidate_state.start()]
            if active is not self:
                active.prewarm()
                return
        if self.warm_standby_enabled:
            self._start_standby_replenish(timeout=10.0)
        # Without warm standby there is nothing to prewarm: every segment uses
        # a dedicated connection because terminal protocols close the socket
        # after the final audio.

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        # Let any in-flight standby cleanup finish: _close_standby detaches
        # self._standby before awaiting the socket close, so interrupting it
        # mid-close would strand that already-detached socket open. shield keeps
        # the close running even if aclose() itself is cancelled (a bare
        # `await close_task` would propagate that cancel into the close). The
        # close is bounded by WS_CLOSE_TIMEOUT_S.
        for close_task in list(self._standby_close_tasks):
            with contextlib.suppress(Exception):
                await asyncio.shield(close_task)
        self._standby_close_tasks.clear()
        await self._close_standby()
        if not self._is_candidate:
            for candidate in self._candidate_tts[1:]:
                await candidate.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Non-streaming synthesis: send the full text once over the SLNG WebSocket."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        if not self._input_text:
            output_emitter.flush()
            return

        ws: aiohttp.ClientWebSocketResponse | None = None
        audio_received = False

        try:
            # Chunked synthesis always uses a dedicated connection: it is a
            # one-shot call, not part of a call's reply stream.
            ws = await self._tts._connect_ws(timeout=self._conn_options.timeout)
            await ws.send_str(json.dumps({"type": "text", "text": self._input_text, "flush": True}))

            while True:
                msg = await ws.receive(timeout=self._conn_options.timeout)
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    # Several bridge providers (Rime, Cartesia) terminate a
                    # segment by closing the socket. Mirror the streaming path:
                    # a close after audio is a normal end-of-segment, not an
                    # error. Only a close with no audio at all is a failure.
                    if audio_received:
                        output_emitter.flush()
                        break
                    raise APIStatusError("SLNG websocket connection closed unexpectedly")

                if msg.type == aiohttp.WSMsgType.BINARY:
                    output_emitter.push(msg.data)
                    audio_received = True
                    continue

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    resp = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.debug(
                        "[SLNG TTS] ignoring non-JSON text frame",
                        extra={"lk.pii.data": msg.data},
                    )
                    continue

                if not isinstance(resp, dict):
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
                        try:
                            output_emitter.push(base64.b64decode(audio_b64))
                            audio_received = True
                        except Exception:
                            logger.warning(
                                "[TTS] invalid base64 audio in chunked synthesis",
                                exc_info=True,
                            )

                    if is_final:
                        output_emitter.flush()
                        break

                    if resp.get("error") is not None:
                        raise APIStatusError(
                            f"SLNG TTS error: {resp.get('error')}",
                            status_code=_extract_error_status(resp) or -1,
                        )
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
                    raise APIStatusError(
                        f"SLNG TTS error: {event.error}",
                        status_code=event.error_status or -1,
                    )
                else:
                    logger.debug("[TTS] ignoring unknown message", extra={"lk.pii.data": resp})
        except (TimeoutError, asyncio.TimeoutError):
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from None
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if ws is not None:
                self._tts._ws_connection_timings.pop(id(ws), None)
                await _close_ws(ws, context="chunked_synthesis")


class SynthesizeStream(tts.SynthesizeStream):
    # SLNG protocol messages (different from Deepgram)
    _FLUSH_MSG: str = json.dumps({"type": "flush"})
    _CANCEL_MSG: str = json.dumps({"type": "cancel"})
    _CLOSE_MSG: str = json.dumps({"type": "close"})

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        logger.debug("[TTS] SynthesizeStream.__init__ STARTING")
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        logger.debug("[TTS] SynthesizeStream.__init__ DONE")

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # Create segments_ch per run so base-class retries after an error get a
        # fresh channel (matching the Deepgram plugin pattern).
        self._segments_ch = utils.aio.Chan[_SegmentInput]()
        request_id = utils.shortuuid()
        logger.debug(f"[TTS] _run starting: request_id={request_id}")
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _tokenize_input() -> None:
            # Converts incoming text into token streams and sends them into _segments_ch
            logger.debug("[TTS] _tokenize_input starting, waiting for input...")
            segment: _SegmentInput | None = None
            text_count = 0
            async for the_input in self._input_ch:
                if isinstance(the_input, str):
                    text_count += 1
                    if text_count == 1:
                        logger.debug(f"[TTS] First text received: '{the_input[:50]}...'")
                    if segment is None:
                        segment = _SegmentInput(stream=self._opts.word_tokenizer.stream())
                        self._segments_ch.send_nowait(segment)
                        logger.debug("[TTS] New token stream created")
                    segment.stream.push_text(the_input)
                elif isinstance(the_input, self._FlushSentinel):
                    logger.debug(f"[TTS] Flush sentinel received after {text_count} texts")
                    if segment is not None:
                        # Mark before ending the input: the sender reads this to
                        # know the next tokens it pulls are the reply's last.
                        segment.input_ended = True
                        segment.stream.end_input()
                    segment = None

            logger.debug(f"[TTS] _tokenize_input done: {text_count} total texts")
            self._segments_ch.close()

        async def _run_segments() -> None:
            logger.debug("[TTS] _run_segments starting, waiting for word_streams...")
            segment_count = 0
            async for segment in self._segments_ch:
                segment_count += 1
                logger.debug(f"[TTS] Processing segment {segment_count}")
                await self._run_ws(segment, output_emitter)
            logger.debug(f"[TTS] _run_segments done: {segment_count} segments")

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except (TimeoutError, asyncio.TimeoutError):
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from None
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(
        self,
        segment: _SegmentInput,
        output_emitter: tts.AudioEmitter,
    ) -> None:
        segment_id = utils.shortuuid()
        logger.debug(f"[TTS] _run_ws starting: segment_id={segment_id}")
        segment_started_at = time.perf_counter()
        output_emitter.start_segment(segment_id=segment_id)
        input_sent_event = asyncio.Event()
        phrase_batching = self._opts.text_chunking in {"sentence", "phrase"}
        # (text, flush) for every frame this reply put on the wire, in order.
        sent_frames: list[tuple[str, bool]] = []
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
        standby_enabled = self._tts.warm_standby_enabled
        standby_used = False
        standby_ready_ms: float | None = None
        standby_miss_reason: str | None = None
        standby_replenish_started = False
        standby_replenish_requested = False

        def capture_ws_timing(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal ws_connect_ms, init_send_ms, connect_total_ms
            timing = self._tts._ws_connection_timings.pop(id(ws), None)
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
            if first_audio_ms is None:
                first_audio_ms = _elapsed_ms(segment_started_at)

        def request_standby_replenish() -> None:
            nonlocal standby_replenish_requested, standby_replenish_started
            if not standby_enabled or standby_replenish_requested:
                return
            standby_replenish_requested = True
            standby_replenish_started = self._tts._start_standby_replenish(
                timeout=self._conn_options.timeout
            )

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

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            batcher = _FrameBatcher(
                batch_phrases=phrase_batching, max_chars=self._opts.phrase_max_chars
            )

            async def _send(frame: str, *, flush: bool) -> None:
                # SLNG: "text" type, with the reply's terminating flush inline on
                # the last frame (the canonical Unmute form: one message, not two).
                self._mark_started()
                payload: dict[str, object] = {"type": "text", "text": frame}
                if flush:
                    payload["flush"] = True
                await ws.send_str(json.dumps(payload))
                sent_frames.append((frame, flush))
                mark_first_text_sent()
                input_sent_event.set()

            stream = segment.stream
            token_count = 0
            flushed = False
            while True:
                try:
                    token = (await stream.__anext__()).token
                except StopAsyncIteration:
                    break
                token_count += 1
                if token_count == 1:
                    logger.debug(f"[TTS] send_task: first token '{token}'")
                if not segment.input_ended:
                    # More text is still coming, so this token cannot close the
                    # reply: send its frame straight away without the flush.
                    frame = batcher.push(token)
                    if frame is not None:
                        await _send(frame, flush=False)
                    continue

                # The reply's text is complete: every remaining token is already
                # queued behind a closed channel, so draining never suspends.
                # Batch the whole tail first so the flush lands on its last frame.
                tail = [token]
                while True:
                    try:
                        tail.append((await stream.__anext__()).token)
                    except StopAsyncIteration:
                        break
                token_count += len(tail) - 1
                frames = [frame for frame in (batcher.push(t) for t in tail) if frame is not None]
                last = batcher.finish()
                if last is not None:
                    frames.append(last)
                for frame in frames[:-1]:
                    await _send(frame, flush=False)
                if frames:
                    await _send(frames[-1], flush=True)
                    flushed = True
                break

            if not flushed:
                # Only reachable when the stream ended without input_ended being
                # observed (a tokenizer that emits eagerly), or when the tail
                # carried no letter and was dropped.
                last = batcher.finish()
                if last is not None:
                    await _send(last, flush=True)
                elif sent_frames:
                    logger.debug("[TTS] tail produced no frame; sending standalone flush")
                    await ws.send_str(self._FLUSH_MSG)
                    sent_frames[-1] = (sent_frames[-1][0], True)

            logger.debug(f"[TTS] send_task: sent {len(sent_frames)} frames ({token_count} tokens)")
            input_sent_event.set()

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal ready_ms, audio_end_ms, gateway_request_id, gateway_session_id
            nonlocal audio_chunks_seen
            await input_sent_event.wait()
            if not sent_frames:
                # A reply whose text carried no letters at all: nothing was sent,
                # so no audio is coming. End the segment rather than waiting.
                logger.debug("[TTS] recv_task: no frames were sent, ending segment")
                output_emitter.end_segment()
                return
            while True:
                msg = await ws.receive(timeout=self._conn_options.timeout)
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if audio_chunks_seen > 0:
                        logger.info(
                            "[TTS] recv_task: websocket closed after terminal "
                            "model output (chunks=%s)",
                            audio_chunks_seen,
                        )
                        audio_end_ms = audio_end_ms or _elapsed_ms(segment_started_at)
                        output_emitter.end_segment()
                        break
                    raise APIStatusError("SLNG websocket connection closed unexpectedly")

                # SLNG: Handle both binary (legacy) and JSON audio_chunk messages
                if msg.type == aiohttp.WSMsgType.BINARY:
                    audio_chunks_seen += 1
                    mark_first_audio_seen()
                    output_emitter.push(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        resp = json.loads(msg.data)
                    except json.JSONDecodeError:
                        logger.debug(
                            "[SLNG TTS] ignoring non-JSON text frame",
                            extra={"lk.pii.data": msg.data},
                        )
                        continue

                    if not isinstance(resp, dict):
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
                        self._tts._emit_plugin_event(
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
                            try:
                                mark_first_audio_seen()
                                output_emitter.push(base64.b64decode(audio_b64))
                            except Exception:
                                if is_final:
                                    logger.warning(
                                        "[TTS] invalid base64 audio (isFinal frame)",
                                        exc_info=True,
                                    )
                                else:
                                    logger.warning(
                                        "[TTS] invalid base64 audio (audio frame)",
                                        exc_info=True,
                                    )
                            else:
                                audio_chunks_seen += 1

                        if is_final:
                            audio_end_ms = audio_end_ms or _elapsed_ms(segment_started_at)
                            output_emitter.end_segment()
                            break

                        if resp.get("error") is not None:
                            raise APIStatusError(
                                f"SLNG TTS error: {resp.get('error')}",
                                status_code=_extract_error_status(resp) or -1,
                            )
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
                        raise APIStatusError(
                            f"SLNG TTS error: {event.error}",
                            status_code=event.error_status or -1,
                        )

                    else:
                        logger.debug("[TTS] ignoring unknown message", extra={"lk.pii.data": resp})

        try:
            ws: aiohttp.ClientWebSocketResponse | None = None
            if standby_enabled:
                (
                    ws,
                    standby_ready_ms,
                    standby_miss_reason,
                ) = await self._tts._checkout_standby()
                standby_used = ws is not None
            if ws is None:
                ws = await self._tts._connect_ws(timeout=self._conn_options.timeout)
            capture_ws_timing(ws)
            tasks = [
                asyncio.create_task(send_task(ws)),
                asyncio.create_task(recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                input_sent_event.set()
                await utils.aio.gracefully_cancel(*tasks)
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


class _FallbackStreamBase:
    def __init__(self, *, parent: TTS, conn_options: APIConnectOptions) -> None:
        self._parent = parent
        self._conn_options = conn_options
        self._index = parent._candidate_state.start()
        self._parent._active_candidate_index = self._index
        self._stream: Any = None
        self._deadline: float | None = None
        self._started = False
        self._closed = False
        self._attempts = 0

    async def _start_stream(self) -> None:
        raise NotImplementedError

    def __aiter__(self) -> _FallbackStreamBase:
        return self

    async def __aenter__(self) -> _FallbackStreamBase:
        if self._stream is None:
            await self._start_stream()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        if self._stream is None:
            # Lazily start so plain `async for` (without `async with` or an
            # explicit __aenter__) works exactly like the non-fallback streams.
            await self._start_stream()
        while True:
            try:
                if not self._started and self._deadline is not None:
                    timeout = max(0.0, self._deadline - time.monotonic())
                    item = await asyncio.wait_for(self._stream.__anext__(), timeout)
                else:
                    item = await self._stream.__anext__()
            except StopAsyncIteration:
                if self._started:
                    raise
                await self._handle_failure(APIConnectionError("TTS produced no audio"))
                continue
            except (TimeoutError, asyncio.TimeoutError) as exc:
                if self._started or self._deadline is None:
                    raise
                await self._handle_failure(exc, allow_retry=False)
                continue
            except Exception as exc:
                if self._started:
                    raise
                await self._handle_failure(exc)
                continue
            self._started = True
            self._deadline = None
            return item

    async def aclose(self) -> None:
        self._closed = True
        stream, self._stream = self._stream, None
        if stream is not None:
            with contextlib.suppress(Exception):
                await stream.aclose()

    async def _handle_failure(self, exc: BaseException, *, allow_retry: bool = True) -> None:
        candidate = self._parent._candidate_tts[self._index]
        if is_payload_too_large(exc):
            # Every candidate receives the same oversized request body, so
            # walking the chain cannot help; surface immediately.
            self._parent._emit_plugin_event(
                "fallback.exhausted",
                "error",
                from_model=candidate._opts.model,
                terminal_status=413,
                error=str(exc),
            )
            raise exc
        if (
            allow_retry
            and not is_non_retryable_client_error(exc)
            and self._attempts < self._conn_options.max_retry
        ):
            self._attempts += 1
            self._parent._emit_plugin_event(
                "fallback.attempt_failed",
                "warning",
                from_model=candidate._opts.model,
                to_model=candidate._opts.model,
                same_model_retry=True,
                error=str(exc),
            )
            await self._restart()
            return

        next_index = self._parent._candidate_state.advance(self._index)
        if next_index is None:
            self._parent._emit_plugin_event(
                "fallback.exhausted",
                "error",
                from_model=candidate._opts.model,
                error=str(exc),
            )
            raise exc

        next_candidate = self._parent._candidate_tts[next_index]
        self._parent._emit_plugin_event(
            "fallback.attempt_failed",
            "warning",
            from_model=candidate._opts.model,
            to_model=next_candidate._opts.model,
            error=str(exc),
        )
        self._index = next_index
        self._parent._active_candidate_index = next_index
        self._attempts = 0
        await self._restart()
        self._parent._emit_plugin_event(
            "fallback.switch_succeeded",
            from_model=candidate._opts.model,
            to_model=next_candidate._opts.model,
        )

    async def _restart(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            with contextlib.suppress(Exception):
                await stream.aclose()
        self._closed = False
        self._started = False
        await self._start_stream()


class _FallbackSynthesizeStream(_FallbackStreamBase):
    # Marker stored in the replay buffer so a segment flush is replayed at the
    # right position when a fallback candidate restarts the stream.
    _FLUSH_MARK = object()

    def __init__(self, *, parent: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(parent=parent, conn_options=conn_options)
        self._texts: list[Any] = []
        self._input_ended = False

    def push_text(self, text: str) -> None:
        self._texts.append(text)
        if self._stream is not None:
            self._arm_timeout()
            self._stream.push_text(text)

    def flush(self) -> None:
        self._texts.append(self._FLUSH_MARK)
        if self._stream is not None:
            self._arm_timeout()
            self._stream.flush()

    def end_input(self) -> None:
        self._input_ended = True
        if self._stream is not None:
            self._arm_timeout()
            self._stream.end_input()

    async def _start_stream(self) -> None:
        candidate = self._parent._candidate_tts[self._index]
        options = replace(self._conn_options, max_retry=0)
        self._stream = await candidate._stream_candidate(conn_options=options).__aenter__()
        for item in self._texts:
            if item is self._FLUSH_MARK:
                self._stream.flush()
            else:
                self._stream.push_text(item)
        if self._input_ended:
            self._stream.end_input()
        if self._texts:
            timeout = self._parent._first_audio_timeout_s
            self._deadline = time.monotonic() + timeout if timeout is not None else None

    def _arm_timeout(self) -> None:
        timeout = self._parent._first_audio_timeout_s
        if self._deadline is None and timeout is not None:
            self._deadline = time.monotonic() + timeout


class _FallbackChunkedStream(_FallbackStreamBase):
    def __init__(
        self,
        *,
        parent: TTS,
        text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(parent=parent, conn_options=conn_options)
        self._text = text
        timeout = parent._first_audio_timeout_s
        self._deadline = time.monotonic() + timeout if timeout is not None else None

    async def _start_stream(self) -> None:
        candidate = self._parent._candidate_tts[self._index]
        options = replace(self._conn_options, max_retry=0)
        self._stream = await candidate._synthesize_candidate(
            self._text,
            conn_options=options,
        ).__aenter__()
        timeout = self._parent._first_audio_timeout_s
        self._deadline = time.monotonic() + timeout if timeout is not None else None

    async def collect(self) -> rtc.AudioFrame:
        """Utility method to collect every frame in a single call"""
        frames = []
        async for ev in self:
            frames.append(ev.frame)

        return rtc.combine_audio_frames(frames)
