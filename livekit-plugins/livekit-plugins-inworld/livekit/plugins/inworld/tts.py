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
import uuid
import weakref
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Literal, cast, get_args
from urllib.parse import urljoin

import aiohttp

from livekit.agents import tokenize, tts, utils
from livekit.agents._exceptions import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import aio, is_given
from livekit.agents.voice.io import TimedString

from .log import logger
from .version import __version__

USER_AGENT = f"livekit-agents-py/{__version__}"

DEFAULT_BIT_RATE = 64000
DEFAULT_ENCODING: Encoding = "PCM"
DEFAULT_MODEL = "inworld-tts-1.5-max"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_URL = "https://api.inworld.ai/"
DEFAULT_WS_URL = "wss://api.inworld.ai/"
DEFAULT_VOICE = "Ashley"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SPEAKING_RATE = 1.0
DEFAULT_BUFFER_CHAR_THRESHOLD = 250
DEFAULT_MAX_BUFFER_DELAY_MS = 3000
NUM_CHANNELS = 1

TTSModels = Literal["inworld-tts-2", "inworld-tts-1.5-max"]
Encoding = Literal["LINEAR16", "PCM", "MP3", "OGG_OPUS", "FLAC"]
TimestampType = Literal["TIMESTAMP_TYPE_UNSPECIFIED", "WORD", "CHARACTER"]
_TextNormalizationStr = Literal["APPLY_TEXT_NORMALIZATION_UNSPECIFIED", "ON", "OFF"]
TextNormalization = _TextNormalizationStr | bool
TimestampTransportStrategy = Literal["TIMESTAMP_TRANSPORT_STRATEGY_UNSPECIFIED", "SYNC", "ASYNC"]
DeliveryMode = Literal["DELIVERY_MODE_UNSPECIFIED", "STABLE", "BALANCED", "CREATIVE"]
AutoModeStrategy = Literal["CLIENT_SEGMENTED", "SENTENCE_BOUNDARY"]

DEFAULT_TIMESTAMP_TRANSPORT_STRATEGY: TimestampTransportStrategy = "ASYNC"


def _validate_str_param(value: object, name: str, literal_type: Any) -> None:
    valid = get_args(literal_type)
    if not isinstance(value, str) or value not in valid:
        raise ValueError(f"Invalid {name}: {value!r}. Must be one of {sorted(valid)}")


def _supports_sentence_boundary(model: str) -> bool:
    # the SENTENCE_BOUNDARY auto-mode strategy is only served by inworld-tts-2 models
    return model.startswith("inworld-tts-2")


def _resolve_text_normalization(value: Any) -> _TextNormalizationStr:
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    _validate_str_param(value, "text_normalization", _TextNormalizationStr)
    return cast(_TextNormalizationStr, value)


@dataclass
class _TTSOptions:
    model: TTSModels | str
    encoding: Encoding
    voice: str
    sample_rate: int
    bit_rate: int
    speaking_rate: float
    temperature: float
    language: NotGivenOr[str] = NOT_GIVEN
    timestamp_type: NotGivenOr[TimestampType] = NOT_GIVEN
    text_normalization: NotGivenOr[TextNormalization] = NOT_GIVEN
    delivery_mode: NotGivenOr[DeliveryMode] = NOT_GIVEN
    timestamp_transport_strategy: TimestampTransportStrategy = DEFAULT_TIMESTAMP_TRANSPORT_STRATEGY
    buffer_char_threshold: NotGivenOr[int] = NOT_GIVEN
    max_buffer_delay_ms: int = DEFAULT_MAX_BUFFER_DELAY_MS
    auto_mode_strategy: AutoModeStrategy = "CLIENT_SEGMENTED"

    @property
    def mime_type(self) -> str:
        if self.encoding == "PCM":
            return "audio/pcm"
        elif self.encoding == "MP3":
            return "audio/mpeg"
        elif self.encoding == "OGG_OPUS":
            return "audio/ogg"
        elif self.encoding == "FLAC":
            return "audio/flac"
        else:
            return "audio/wav"


# Shared WebSocket connection infrastructure
class _ContextState(Enum):
    ACTIVE = "active"
    CREATING = "creating"
    CLOSING = "closing"


@dataclass
class _ContextInfo:
    context_id: str
    state: _ContextState
    emitter: tts.AudioEmitter | None = None
    waiter: asyncio.Future[None] | None = None
    segment_started: bool = False
    flush_after_push: bool = False
    created_at: float = field(default_factory=time.time)
    close_started_at: float | None = None
    # Cumulative timestamp tracking for monotonic timestamps across generations.
    # When auto_mode is enabled or flush_context() is called, the server resets
    # timestamps to 0 after each generation. We add cumulative_time to maintain
    # monotonically increasing timestamps within an agent turn.
    cumulative_time: float = 0.0
    generation_end_time: float = 0.0


@dataclass
class _CreateContextMsg:
    context_id: str
    opts: _TTSOptions


@dataclass
class _SendTextMsg:
    context_id: str
    text: str


@dataclass
class _FlushContextMsg:
    context_id: str


@dataclass
class _CloseContextMsg:
    context_id: str
    cancel: bool = False


_OutboundMessage = _CreateContextMsg | _SendTextMsg | _FlushContextMsg | _CloseContextMsg


class _InworldConnection:
    """A WebSocket connection to Inworld TTS, multiplexing up to 5 concurrent contexts."""

    HANDSHAKE_TIMEOUT = 10.0

    def __init__(
        self,
        session: aiohttp.ClientSession,
        ws_url: str,
        authorization: str,
    ) -> None:
        self._session = session
        self._ws_url = ws_url
        self._authorization = authorization
        self._ws: aiohttp.ClientWebSocketResponse | None = None

        self._contexts: dict[str, _ContextInfo] = {}

        # Message queue for outbound messages
        self._outbound_queue: asyncio.Queue[_OutboundMessage] = asyncio.Queue()

        # Connection state
        self._closed = False

        # Background tasks
        self._connect_task: asyncio.Task[None] | None = None
        self._send_task: asyncio.Task[None] | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None

    async def connect(self, timeout: float) -> None:
        """Wait up to ``timeout`` for the WebSocket handshake, starting it on first call.

        The handshake is shared by all callers and has its own deadline, so a caller that
        times out or is cancelled doesn't affect the others.
        """
        if self._connect_task is None:
            self._connect_task = asyncio.create_task(self._connect())
        await asyncio.wait_for(asyncio.shield(self._connect_task), timeout)

    async def _connect(self) -> None:
        """Open the WebSocket and start the send, receive and cleanup loops.

        If the handshake fails or exceeds HANDSHAKE_TIMEOUT, the connection is closed and
        every registered context's waiter fails with APIConnectionError.
        """
        if self._closed:
            return

        url = urljoin(self._ws_url, "/tts/v1/voice:streamBidirectional")
        request_id = str(uuid.uuid4())
        try:
            self._ws = await asyncio.wait_for(
                self._session.ws_connect(
                    url,
                    headers={
                        "Authorization": self._authorization,
                        "X-User-Agent": USER_AGENT,
                        "X-Request-Id": request_id,
                    },
                ),
                timeout=self.HANDSHAKE_TIMEOUT,
            )
        except Exception as e:
            logger.warning(
                "Failed to connect to Inworld TTS WebSocket",
                extra={"request_id": request_id, "error": str(e) or type(e).__name__},
            )
            await self._handle_connection_error(APIConnectionError())
            return

        logger.debug(
            "Established Inworld TTS WebSocket connection",
            extra={"request_id": request_id},
        )

        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_stale_contexts())

    def create_context(
        self, emitter: tts.AudioEmitter, opts: _TTSOptions
    ) -> tuple[str, asyncio.Future[None]]:
        """Register a new context and queue its create message, sent once connected."""
        ctx_id = utils.shortuuid()
        waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        self._contexts[ctx_id] = _ContextInfo(
            context_id=ctx_id,
            state=_ContextState.CREATING,
            emitter=emitter,
            waiter=waiter,
            flush_after_push=opts.encoding == "PCM",
        )
        self._outbound_queue.put_nowait(_CreateContextMsg(context_id=ctx_id, opts=opts))
        return ctx_id, waiter

    def send_text(self, context_id: str, text: str) -> None:
        """Queue text to be sent to a context."""
        try:
            self._outbound_queue.put_nowait(_SendTextMsg(context_id=context_id, text=text))
        except asyncio.QueueFull:
            logger.warning("Outbound queue full, dropping text")

    def flush_context(self, context_id: str) -> None:
        """Queue a flush message for a context."""
        try:
            self._outbound_queue.put_nowait(_FlushContextMsg(context_id=context_id))
        except asyncio.QueueFull:
            logger.warning("Outbound queue full, dropping flush")

    def _release_context(self, context_id: str) -> None:
        self._contexts.pop(context_id, None)

    def close_context(self, context_id: str, *, cancel: bool = False) -> None:
        """Queue a close message for a context.

        With ``cancel``, the server stops generating and discards text it has not
        synthesized yet instead of finishing it.
        """
        ctx = self._contexts.get(context_id)
        if ctx:
            ctx.state = _ContextState.CLOSING
            ctx.close_started_at = time.time()
        try:
            self._outbound_queue.put_nowait(_CloseContextMsg(context_id=context_id, cancel=cancel))
        except asyncio.QueueFull:
            logger.warning("Outbound queue full, dropping close")

    async def _send_loop(self) -> None:
        """Process outbound messages to WebSocket."""
        try:
            while not self._closed and self._ws:
                try:
                    msg = await asyncio.wait_for(self._outbound_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if self._closed or not self._ws:
                    break

                if isinstance(msg, _CreateContextMsg):
                    opts = msg.opts
                    pkt: dict[str, Any] = {
                        "create": {
                            "voiceId": opts.voice,
                            "modelId": opts.model,
                            "audioConfig": {
                                "audioEncoding": opts.encoding,
                                "sampleRateHertz": opts.sample_rate,
                                "bitrate": opts.bit_rate,
                                "speakingRate": opts.speaking_rate,
                            },
                            "temperature": opts.temperature,
                            "maxBufferDelayMs": opts.max_buffer_delay_ms,
                            "timestampTransportStrategy": opts.timestamp_transport_strategy,
                        },
                        "contextId": msg.context_id,
                    }
                    if is_given(opts.language):
                        pkt["create"]["language"] = opts.language
                    if is_given(opts.timestamp_type):
                        pkt["create"]["timestampType"] = opts.timestamp_type
                    if is_given(opts.text_normalization):
                        pkt["create"]["applyTextNormalization"] = opts.text_normalization
                    if is_given(opts.delivery_mode):
                        pkt["create"]["deliveryMode"] = opts.delivery_mode
                    pkt["create"]["autoMode"] = True
                    if opts.auto_mode_strategy == "SENTENCE_BOUNDARY":
                        pkt["create"]["autoModeStrategy"] = opts.auto_mode_strategy
                    # with SENTENCE_BOUNDARY the threshold also cuts long sentences, so only a
                    # set value is sent and the server default applies otherwise
                    if is_given(opts.buffer_char_threshold):
                        pkt["create"]["bufferCharThreshold"] = opts.buffer_char_threshold
                    elif opts.auto_mode_strategy != "SENTENCE_BOUNDARY":
                        pkt["create"]["bufferCharThreshold"] = DEFAULT_BUFFER_CHAR_THRESHOLD
                    await self._ws.send_str(json.dumps(pkt))

                elif isinstance(msg, _SendTextMsg):
                    pkt = {
                        "send_text": {"text": msg.text},
                        "contextId": msg.context_id,
                    }
                    await self._ws.send_str(json.dumps(pkt))

                elif isinstance(msg, _FlushContextMsg):
                    pkt = {"flush_context": {}, "contextId": msg.context_id}
                    await self._ws.send_str(json.dumps(pkt))

                elif isinstance(msg, _CloseContextMsg):
                    close_key = "cancel_context" if msg.cancel else "close_context"
                    pkt = {close_key: {}, "contextId": msg.context_id}
                    await self._ws.send_str(json.dumps(pkt))

        except Exception as e:
            logger.error("Inworld send loop error", exc_info=e)
        finally:
            if not self._closed:
                await self._handle_connection_error(APIConnectionError())

    async def _recv_loop(self) -> None:
        """Process inbound messages and route to correct context."""
        try:
            while not self._closed and self._ws:
                try:
                    msg = await self._ws.receive(timeout=60.0)
                except asyncio.TimeoutError:
                    continue

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(msg.data)

                # Check for errors that are not associated with a context
                if err_obj := data.get("error"):
                    logger.warning(
                        "Received error from Inworld",
                        extra={
                            "error_code": err_obj.get("code", 0),
                            "error_message": err_obj.get("message", "Unknown error"),
                        },
                    )
                    continue

                result = data.get("result", {})
                context_id = result.get("contextId")

                ctx = self._contexts.get(context_id) if context_id else None

                # Check for errors in status
                status = result.get("status", {})
                if status.get("code", 0) != 0:
                    error_msg = status.get("message", "Unknown error")
                    error = APIError(f"Inworld error: {error_msg}")
                    logger.warning(
                        "Received error from Inworld",
                        extra={
                            "context_id": context_id,
                            "error_code": status.get("code"),
                            "error_message": error_msg,
                            "context_state": ctx.state.value if ctx else "unknown",
                            "context_known": ctx is not None,
                        },
                    )
                    if ctx:
                        if ctx.waiter and not ctx.waiter.done():
                            ctx.waiter.set_exception(error)
                        self._release_context(context_id)
                    continue

                if not ctx:
                    continue

                if "contextCreated" in result:
                    ctx.state = _ContextState.ACTIVE
                    logger.info(
                        "Context created confirmed by server",
                        extra={"context_id": context_id},
                    )
                    continue

                if audio_chunk := result.get("audioChunk"):
                    if ctx.emitter:
                        if not ctx.segment_started:
                            ctx.emitter.start_segment(segment_id=context_id)
                            ctx.segment_started = True

                        # Adjust timestamps for cumulative offset
                        if timestamp_info := audio_chunk.get("timestampInfo"):
                            if word_align := timestamp_info.get("wordAlignment"):
                                raw_words = word_align.get("words", [])
                                raw_starts = word_align.get("wordStartTimeSeconds", [])
                                raw_ends = word_align.get("wordEndTimeSeconds", [])
                                logger.debug(
                                    "Raw timestamps from server",
                                    extra={
                                        "context_id": context_id,
                                        "cumulative_offset": ctx.cumulative_time,
                                        "lk.pii.raw_words": raw_words,
                                        "raw_starts": raw_starts,
                                        "raw_ends": raw_ends,
                                    },
                                )

                            timed_strings = _parse_timestamp_info(
                                timestamp_info, cumulative_time=ctx.cumulative_time
                            )
                            # Track generation end time from last word for cumulative offset
                            if timed_strings:
                                last_ts = timed_strings[-1]
                                if utils.is_given(last_ts.end_time):
                                    ctx.generation_end_time = last_ts.end_time

                                logger.debug(
                                    "Adjusted timestamps (with cumulative offset)",
                                    extra={
                                        "context_id": context_id,
                                        "lk.pii.words": [str(ts) for ts in timed_strings],
                                        "adjusted_starts": [ts.start_time for ts in timed_strings],
                                        "adjusted_ends": [ts.end_time for ts in timed_strings],
                                        "generation_end_time": ctx.generation_end_time,
                                    },
                                )

                            for ts in timed_strings:
                                ctx.emitter.push_timed_transcript(ts)

                        if audio_content := audio_chunk.get("audioContent"):
                            ctx.emitter.push(base64.b64decode(audio_content))
                            if ctx.flush_after_push:
                                ctx.emitter.flush()
                    continue

                if "flushCompleted" in result:
                    # Signals the end of a generation, subsequent timestampes from the server
                    # will reset offset to 0. We need to update the cumulative time to the
                    # generation end time to maintain monotonically increasing timestamps
                    # within the agent turn.
                    if ctx.generation_end_time > 0:
                        logger.debug(
                            "flushCompleted - updating cumulative time",
                            extra={
                                "context_id": context_id,
                                "old_cumulative_time": ctx.cumulative_time,
                                "new_cumulative_time": ctx.generation_end_time,
                            },
                        )
                        ctx.cumulative_time = ctx.generation_end_time
                    continue

                if "contextClosed" in result:
                    if ctx.waiter and not ctx.waiter.done():
                        ctx.waiter.set_result(None)
                    self._release_context(context_id)
                    continue

        except Exception as e:
            logger.error("Inworld recv loop error", exc_info=e)
        finally:
            if not self._closed:
                await self._handle_connection_error(APIConnectionError())

    async def _cleanup_stale_contexts(self) -> None:
        """Periodically clean up orphaned contexts stuck in CLOSING state."""
        while not self._closed:
            await asyncio.sleep(60.0)
            now = time.time()
            for ctx in list(self._contexts.values()):
                # Use close_started_at if available, otherwise fall back to created_at
                close_time = ctx.close_started_at or ctx.created_at
                if ctx.state == _ContextState.CLOSING and now - close_time > 120.0:
                    if ctx.waiter and not ctx.waiter.done():
                        ctx.waiter.set_result(None)
                    self._release_context(ctx.context_id)

    async def _handle_connection_error(self, error: Exception) -> None:
        """Fail all active contexts, stop the background loops and close the WebSocket."""
        for ctx in list(self._contexts.values()):
            if ctx.waiter and not ctx.waiter.done():
                ctx.waiter.set_exception(error)
        self._contexts.clear()
        self._closed = True

        # called from inside the send or receive loop, which must not cancel itself
        current = asyncio.current_task()
        for task in (self._send_task, self._recv_task, self._cleanup_task):
            if task and task is not current:
                task.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug("Error closing Inworld TTS WebSocket", extra={"error": str(e)})

    async def aclose(self) -> None:
        """Close the connection and all contexts."""
        self._closed = True

        # Cancel background tasks
        for task in [self._connect_task, self._send_task, self._recv_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close all contexts and fail their waiters
        for ctx in list(self._contexts.values()):
            if ctx.waiter and not ctx.waiter.done():
                ctx.waiter.cancel()

        self._contexts.clear()

        if self._ws:
            await self._ws.close()
            self._ws = None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        encoding: NotGivenOr[Encoding] = NOT_GIVEN,
        bit_rate: NotGivenOr[int] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        speaking_rate: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        timestamp_type: NotGivenOr[TimestampType] = NOT_GIVEN,
        text_normalization: NotGivenOr[TextNormalization] = NOT_GIVEN,
        delivery_mode: NotGivenOr[DeliveryMode] = NOT_GIVEN,
        timestamp_transport_strategy: NotGivenOr[TimestampTransportStrategy] = NOT_GIVEN,
        buffer_char_threshold: NotGivenOr[int] = NOT_GIVEN,
        max_buffer_delay_ms: NotGivenOr[int] = NOT_GIVEN,
        base_url: str = DEFAULT_URL,
        ws_url: str = DEFAULT_WS_URL,
        http_session: aiohttp.ClientSession | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        retain_format: NotGivenOr[bool] = NOT_GIVEN,
        max_connections: NotGivenOr[int] = NOT_GIVEN,
        idle_connection_timeout: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Inworld TTS.

        Args:
            api_key (str, optional): The Inworld API key.
                If not provided, it will be read from the INWORLD_API_KEY environment variable.
            voice (str, optional): The voice to use. Defaults to "Ashley".
            model (str, optional): The Inworld model to use. Defaults to "inworld-tts-1.5-max".
            encoding (str, optional): The encoding to use. Defaults to "PCM".
            bit_rate (int, optional): Bits per second of the audio. Defaults to 64000.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            speaking_rate (float, optional): The speed of the voice, in the range [0.5, 1.5].
                Defaults to 1.0.
            temperature (float, optional): Determines the degree of randomness when sampling audio
                tokens to generate the response. Range (0, 2]. Defaults to 1.0.
            language (str, optional): BCP-47 language tag (e.g., "en-US", "fr-FR", "ja-JP")
                specifying the language that the given voice should speak the text in.
                If not set, the model default applies.
            timestamp_type (str, optional): Controls timestamp metadata returned with the audio.
                Use "WORD" for word-level timestamps or "CHARACTER" for character-level.
                Useful for karaoke-style captions, word highlighting, and lipsync.
            text_normalization (str, optional): Controls text normalization. When "ON", numbers,
                dates, and abbreviations are expanded (e.g., "Dr." -> "Doctor"). When "OFF",
                text is read exactly as written. Defaults to automatic.
            delivery_mode (str, optional): Controls output variation on ``inworld-tts-2`` only.
                One of "DELIVERY_MODE_UNSPECIFIED", "STABLE", "BALANCED", or "CREATIVE".
                The Inworld API ignores ``temperature`` on ``inworld-tts-2`` — use
                ``delivery_mode`` to steer output variation on that model instead.
                Defaults to the server-side default ("BALANCED").
            timestamp_transport_strategy (str, optional): Controls how timestamp info is
                transported relative to audio data. "SYNC" returns timestamps in the same
                message as audio data. "ASYNC" allows timestamps to return in trailing
                messages after the audio data. Defaults to "ASYNC".
            buffer_char_threshold (int, optional): For streaming, the minimum number of characters
                in the buffer that automatically triggers audio generation. Defaults to 250, or to
                the server default when the server splits sentences.
            max_buffer_delay_ms (int, optional): For streaming, the maximum time in ms to buffer
                before starting generation. Defaults to 3000.
            base_url (str, optional): The base URL for the Inworld TTS API.
                Defaults to "https://api.inworld.ai/".
            ws_url (str, optional): The WebSocket URL for streaming TTS.
                Defaults to "wss://api.inworld.ai/".
            http_session (aiohttp.ClientSession, optional): The HTTP session to use.
            tokenizer (tokenize.SentenceTokenizer, optional): The tokenizer to use for streaming.
                Defaults to `livekit.agents.tokenize.blingfire.SentenceTokenizer`. When neither
                this nor `retain_format=False` is set, inworld-tts-2 models stream text as it
                arrives and the server splits it at sentence boundaries instead.
            retain_format (bool, optional): Whether to retain the format of the text when tokenizing.
                Defaults to True.
            max_connections (int, optional): Deprecated, has no effect.
            idle_connection_timeout (float, optional): Deprecated, has no effect.
        """
        if not is_given(sample_rate):
            sample_rate = DEFAULT_SAMPLE_RATE
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=is_given(timestamp_type)
                and timestamp_type != "TIMESTAMP_TYPE_UNSPECIFIED",
            ),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        if is_given(max_connections):
            logger.warning("`max_connections` is deprecated and no longer used")
        if is_given(idle_connection_timeout):
            logger.warning("`idle_connection_timeout` is deprecated and no longer used")

        key = api_key if is_given(api_key) else os.getenv("INWORLD_API_KEY")
        if not key:
            raise ValueError(
                "Inworld API key is required, either as argument or set"
                " INWORLD_API_KEY environment variable"
            )

        self._authorization = f"Basic {key}"
        self._base_url = base_url
        self._ws_url = ws_url
        self._session = http_session

        if is_given(encoding):
            _validate_str_param(encoding, "encoding", Encoding)
        if is_given(timestamp_type):
            _validate_str_param(timestamp_type, "timestamp_type", TimestampType)
        if is_given(text_normalization):
            text_normalization = _resolve_text_normalization(text_normalization)
        if is_given(delivery_mode):
            _validate_str_param(delivery_mode, "delivery_mode", DeliveryMode)
        if is_given(timestamp_transport_strategy):
            _validate_str_param(
                timestamp_transport_strategy,
                "timestamp_transport_strategy",
                TimestampTransportStrategy,
            )

        self._opts = _TTSOptions(
            voice=voice if is_given(voice) else DEFAULT_VOICE,
            model=model if is_given(model) else DEFAULT_MODEL,
            encoding=encoding if is_given(encoding) else DEFAULT_ENCODING,
            bit_rate=bit_rate if is_given(bit_rate) else DEFAULT_BIT_RATE,
            sample_rate=sample_rate if is_given(sample_rate) else DEFAULT_SAMPLE_RATE,
            speaking_rate=speaking_rate if is_given(speaking_rate) else DEFAULT_SPEAKING_RATE,
            temperature=temperature if is_given(temperature) else DEFAULT_TEMPERATURE,
            language=language,
            timestamp_type=timestamp_type,
            text_normalization=text_normalization,
            delivery_mode=delivery_mode,
            timestamp_transport_strategy=timestamp_transport_strategy
            if is_given(timestamp_transport_strategy)
            else DEFAULT_TIMESTAMP_TRANSPORT_STRATEGY,
            buffer_char_threshold=buffer_char_threshold,
            max_buffer_delay_ms=max_buffer_delay_ms
            if is_given(max_buffer_delay_ms)
            else DEFAULT_MAX_BUFFER_DELAY_MS,
        )

        self._connection: _InworldConnection | None = None
        self._prewarm_task: asyncio.Task[None] | None = None
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._client_segmentation = is_given(tokenizer) or retain_format is False
        self._sentence_tokenizer = (
            tokenizer
            if is_given(tokenizer)
            else tokenize.blingfire.SentenceTokenizer(
                retain_format=retain_format if is_given(retain_format) else True
            )
        )

    class Markup(tts.TTS.Markup):
        # markup delegation lives in the base class, keyed on _provider_key()
        def _provider_key(self) -> str:
            # only inworld-tts-2 understands the markup tags; older models get no
            # markup so the tags aren't injected, converted, or stripped (matches
            # the inference gateway's behavior)
            return "inworld" if "tts-2" in self._tts.model else ""

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Inworld"

    def _get_connection(self) -> _InworldConnection:
        """Return the shared connection, replacing it with a new one once it has closed."""
        if self._connection is None or self._connection._closed:
            self._connection = _InworldConnection(
                session=self._ensure_session(),
                ws_url=self._ws_url,
                authorization=self._authorization,
            )
        return self._connection

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        encoding: NotGivenOr[Encoding] = NOT_GIVEN,
        bit_rate: NotGivenOr[int] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        speaking_rate: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        timestamp_type: NotGivenOr[TimestampType] = NOT_GIVEN,
        text_normalization: NotGivenOr[TextNormalization] = NOT_GIVEN,
        delivery_mode: NotGivenOr[DeliveryMode] = NOT_GIVEN,
        timestamp_transport_strategy: NotGivenOr[TimestampTransportStrategy] = NOT_GIVEN,
        buffer_char_threshold: NotGivenOr[int] = NOT_GIVEN,
        max_buffer_delay_ms: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS configuration options.

        Args:
            voice (str, optional): The voice to use.
            model (str, optional): The Inworld model to use.
            encoding (str, optional): The encoding to use.
            bit_rate (int, optional): Bits per second of the audio.
            sample_rate (int, optional): The audio sample rate in Hz.
            speaking_rate (float, optional): The speed of the voice.
            temperature (float, optional): Determines the degree of randomness when sampling audio
                tokens to generate the response.
            language (str, optional): BCP-47 language tag (e.g., "en-US", "fr-FR").
            timestamp_type (str, optional): Controls timestamp metadata ("WORD" or "CHARACTER").
            text_normalization (str, optional): Controls text normalization ("ON" or "OFF").
            delivery_mode (str, optional): Controls output variation on ``inworld-tts-2`` only
                ("DELIVERY_MODE_UNSPECIFIED", "STABLE", "BALANCED", or "CREATIVE"). The Inworld
                API ignores ``temperature`` on TTS-2; use this instead.
            timestamp_transport_strategy (str, optional): Controls timestamp transport strategy
                ("SYNC" or "ASYNC").
            buffer_char_threshold (int, optional): For streaming, min characters before triggering.
            max_buffer_delay_ms (int, optional): For streaming, max time to buffer.
        """
        if is_given(voice):
            self._opts.voice = voice
        if is_given(model):
            self._opts.model = model
        if is_given(encoding):
            _validate_str_param(encoding, "encoding", Encoding)
            self._opts.encoding = encoding
        if is_given(bit_rate):
            self._opts.bit_rate = bit_rate
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
            # keep the base class property in sync; the framework reads tts.sample_rate
            self._sample_rate = sample_rate
        if is_given(speaking_rate):
            self._opts.speaking_rate = speaking_rate
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(language):
            self._opts.language = language
        if is_given(timestamp_type):
            _validate_str_param(timestamp_type, "timestamp_type", TimestampType)
            self._opts.timestamp_type = timestamp_type
        if is_given(text_normalization):
            self._opts.text_normalization = _resolve_text_normalization(text_normalization)
        if is_given(delivery_mode):
            _validate_str_param(delivery_mode, "delivery_mode", DeliveryMode)
            self._opts.delivery_mode = delivery_mode
        if is_given(timestamp_transport_strategy):
            _validate_str_param(
                timestamp_transport_strategy,
                "timestamp_transport_strategy",
                TimestampTransportStrategy,
            )
            self._opts.timestamp_transport_strategy = timestamp_transport_strategy
        if is_given(buffer_char_threshold):
            self._opts.buffer_char_threshold = buffer_char_threshold
        if is_given(max_buffer_delay_ms):
            self._opts.max_buffer_delay_ms = max_buffer_delay_ms

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def prewarm(self) -> None:
        # Don't replace a prewarm still in flight: the old task would lose its only
        # reference, and aclose() cancels just the latest one, so it could open a
        # connection after shutdown. Same guard as the soniox plugin.
        if self._prewarm_task is None or self._prewarm_task.done():
            self._prewarm_task = asyncio.create_task(self._prewarm_impl())

    async def _prewarm_impl(self) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await self._get_connection().connect(DEFAULT_API_CONNECT_OPTIONS.timeout)

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        # Cancel first: a prewarm still in flight here would otherwise open a
        # connection after this one was closed, leaving one nothing owns.
        if self._prewarm_task is not None:
            await aio.cancel_and_wait(self._prewarm_task)
            self._prewarm_task = None

        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        if self._connection:
            await self._connection.aclose()
            self._connection = None

    async def list_voices(self, language: str | None = None) -> list[dict[str, Any]]:
        """
        List all available voices in the workspace associated with the API key.

        Args:
            language (str, optional): ISO 639-1 language code to filter voices (e.g., 'en', 'es', 'fr').
        """
        url = urljoin(self._base_url, "tts/v1/voices")
        params = {}
        if language:
            params["filter"] = f"language={language}"

        async with self._ensure_session().get(
            url,
            headers={
                "Authorization": self._authorization,
                "X-User-Agent": USER_AGENT,
                "X-Request-Id": str(uuid.uuid4()),
            },
            params=params,
        ) as resp:
            if not resp.ok:
                error_body = await resp.json()
                raise APIStatusError(
                    message=error_body.get("message"),
                    status_code=resp.status,
                    request_id=None,
                    body=None,
                )

            data = await resp.json()
            return cast(list[dict[str, Any]], data.get("voices", []))


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            audio_config: dict[str, Any] = {
                "audioEncoding": self._opts.encoding,
                "bitrate": self._opts.bit_rate,
                "sampleRateHertz": self._opts.sample_rate,
                "temperature": self._opts.temperature,
                "speakingRate": self._opts.speaking_rate,
            }

            body_params: dict[str, Any] = {
                "text": self._input_text,
                "voiceId": self._opts.voice,
                "modelId": self._opts.model,
                "audioConfig": audio_config,
            }
            if utils.is_given(self._opts.language):
                body_params["language"] = self._opts.language
            if utils.is_given(self._opts.timestamp_type):
                body_params["timestampType"] = self._opts.timestamp_type
            if utils.is_given(self._opts.text_normalization):
                body_params["applyTextNormalization"] = self._opts.text_normalization
            if utils.is_given(self._opts.delivery_mode):
                body_params["deliveryMode"] = self._opts.delivery_mode
            body_params["timestampTransportStrategy"] = self._opts.timestamp_transport_strategy

            x_request_id = str(uuid.uuid4())
            async with self._tts._ensure_session().post(
                urljoin(self._tts._base_url, "/tts/v1/voice:stream"),
                headers={
                    "Authorization": self._tts._authorization,
                    "X-User-Agent": USER_AGENT,
                    "X-Request-Id": x_request_id,
                },
                json=body_params,
                timeout=aiohttp.ClientTimeout(sock_connect=self._conn_options.timeout),
                # large read_bufsize to avoid `ValueError: Chunk too big`
                read_bufsize=10 * 1024 * 1024,
            ) as resp:
                resp.raise_for_status()

                request_id = utils.shortuuid()
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=self._opts.mime_type,
                )

                async for raw_line in resp.content:
                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(
                            "failed to parse Inworld response line", extra={"lk.pii.line": line}
                        )
                        continue

                    if result := data.get("result"):
                        # Handle timestamp info if present
                        if timestamp_info := result.get("timestampInfo"):
                            timed_strings = _parse_timestamp_info(timestamp_info)
                            if timed_strings:
                                output_emitter.push_timed_transcript(timed_strings)

                        if audio_content := result.get("audioContent"):
                            output_emitter.push(base64.b64decode(audio_content))
                            if self._opts.encoding == "PCM":
                                output_emitter.flush()
                    elif error := data.get("error"):
                        raise APIStatusError(
                            message=error.get("message"),
                            status_code=error.get("code"),
                            request_id=x_request_id,
                            body=None,
                        )
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=x_request_id, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        if not tts._client_segmentation and _supports_sentence_boundary(self._opts.model):
            self._opts.auto_mode_strategy = "SENTENCE_BOUNDARY"

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        sent_tokenizer_stream = self._tts._sentence_tokenizer.stream()

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type=self._opts.mime_type,
            stream=True,
        )

        connection = self._tts._get_connection()
        context_id, waiter = connection.create_context(output_emitter, self._opts)

        def _send_text(text: str) -> None:
            # Chunk to stay within Inworld's 1000 char limit
            for i in range(0, len(text), 1000):
                connection.send_text(context_id, text[i : i + 1000])
                self._mark_started()

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_tokenizer_stream.flush()
                    continue
                sent_tokenizer_stream.push_text(data)
            sent_tokenizer_stream.end_input()

        async def _send_task() -> None:
            async for ev in sent_tokenizer_stream:
                _send_text(ev.token)
            connection.flush_context(context_id)
            connection.close_context(context_id)

        async def _forward_task() -> None:
            # the server buffers partial sentences until it finds a boundary
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    connection.flush_context(context_id)
                elif data:
                    _send_text(data)
            connection.close_context(context_id)

        if self._opts.auto_mode_strategy == "SENTENCE_BOUNDARY":
            tasks = [asyncio.create_task(_forward_task())]
        else:
            tasks = [
                asyncio.create_task(_input_task()),
                asyncio.create_task(_send_task()),
            ]

        try:
            await connection.connect(self._conn_options.timeout)
            await asyncio.wait_for(waiter, timeout=self._conn_options.timeout + 60)
        except asyncio.TimeoutError:
            connection.close_context(context_id, cancel=True)
            raise APITimeoutError() from None
        except asyncio.CancelledError:
            connection.close_context(context_id, cancel=True)
            raise
        except APIError:
            raise
        except Exception as e:
            logger.error("Inworld stream error", extra={"context_id": context_id, "error": e})
            connection.close_context(context_id, cancel=True)
            raise APIConnectionError() from e
        finally:
            # nothing awaits the waiter after this, so a later connection error must not land on it
            if not waiter.done():
                waiter.cancel()
            elif not waiter.cancelled():
                waiter.exception()
            await utils.aio.gracefully_cancel(*tasks)
            await sent_tokenizer_stream.aclose()
            output_emitter.end_input()


def _parse_timestamp_info(
    timestamp_info: dict[str, Any], cumulative_time: float = 0.0
) -> list[TimedString]:
    """Parse timestamp info from API response into TimedString objects.

    Args:
        timestamp_info: The timestamp info from the API response.
        cumulative_time: Offset to add to all timestamps for monotonic ordering
            across multiple generations within a single context.
    """
    timed_strings: list[TimedString] = []

    # Handle word-level alignment
    if word_align := timestamp_info.get("wordAlignment"):
        words = word_align.get("words", [])
        starts = word_align.get("wordStartTimeSeconds", [])
        ends = word_align.get("wordEndTimeSeconds", [])

        for word, start, end in zip(words, starts, ends, strict=False):
            text = f"{word}"
            timed_strings.append(
                TimedString(
                    text,
                    start_time=cumulative_time + start,
                    end_time=cumulative_time + end,
                )
            )

    # Handle character-level alignment
    if char_align := timestamp_info.get("characterAlignment"):
        chars = char_align.get("characters", [])
        starts = char_align.get("characterStartTimeSeconds", [])
        ends = char_align.get("characterEndTimeSeconds", [])

        for char, start, end in zip(chars, starts, ends, strict=False):
            timed_strings.append(
                TimedString(
                    char,
                    start_time=cumulative_time + start,
                    end_time=cumulative_time + end,
                )
            )

    return timed_strings
