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
import json
import os
import time
import uuid
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Literal, cast
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
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from .log import logger
from .version import __version__

USER_AGENT = f"livekit-agents-py/{__version__}"

DEFAULT_BIT_RATE = 64000
DEFAULT_ENCODING = "OGG_OPUS"
DEFAULT_MODEL = "inworld-tts-1"
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_URL = "https://api.inworld.ai/"
DEFAULT_WS_URL = "wss://api.inworld.ai/"
DEFAULT_VOICE = "Ashley"
DEFAULT_TEMPERATURE = 1.1
DEFAULT_SPEAKING_RATE = 1.0
DEFAULT_BUFFER_CHAR_THRESHOLD = 120
DEFAULT_MAX_BUFFER_DELAY_MS = 3000
NUM_CHANNELS = 1

Encoding = Literal["LINEAR16", "MP3", "OGG_OPUS", "ALAW", "MULAW", "FLAC"] | str
TimestampType = Literal["TIMESTAMP_TYPE_UNSPECIFIED", "WORD", "CHARACTER"]
TextNormalization = Literal["APPLY_TEXT_NORMALIZATION_UNSPECIFIED", "ON", "OFF"]
TimestampTransportStrategy = Literal["TIMESTAMP_TRANSPORT_STRATEGY_UNSPECIFIED", "SYNC", "ASYNC"]

DEFAULT_TIMESTAMP_TRANSPORT_STRATEGY: TimestampTransportStrategy = "ASYNC"


@dataclass
class _TTSOptions:
    model: str
    encoding: Encoding
    voice: str
    sample_rate: int
    bit_rate: int
    speaking_rate: float
    temperature: float
    timestamp_type: NotGivenOr[TimestampType] = NOT_GIVEN
    text_normalization: NotGivenOr[TextNormalization] = NOT_GIVEN
    timestamp_transport_strategy: TimestampTransportStrategy = DEFAULT_TIMESTAMP_TRANSPORT_STRATEGY
    buffer_char_threshold: int = DEFAULT_BUFFER_CHAR_THRESHOLD
    max_buffer_delay_ms: int = DEFAULT_MAX_BUFFER_DELAY_MS

    @property
    def mime_type(self) -> str:
        if self.encoding == "MP3":
            return "audio/mpeg"
        elif self.encoding == "OGG_OPUS":
            return "audio/ogg"
        elif self.encoding == "FLAC":
            return "audio/flac"
        elif self.encoding in ("ALAW", "MULAW"):
            return "audio/basic"
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


_OutboundMessage = _CreateContextMsg | _SendTextMsg | _FlushContextMsg | _CloseContextMsg


class _InworldConnection:
    """Manages a single shared WebSocket connection with up to 5 concurrent contexts."""

    MAX_CONTEXTS = 5

    def __init__(
        self,
        session: aiohttp.ClientSession,
        ws_url: str,
        authorization: str,
        on_capacity_available: Callable[[], None] | None = None,
    ) -> None:
        self._session = session
        self._ws_url = ws_url
        self._authorization = authorization
        self._on_capacity_available = on_capacity_available
        self._ws: aiohttp.ClientWebSocketResponse | None = None

        # Context pool
        self._contexts: dict[str, _ContextInfo] = {}
        self._context_available = asyncio.Event()
        self._pending_acquisitions: int = 0  # Reserved slots not yet in _contexts

        # Message queue for outbound messages
        self._outbound_queue: asyncio.Queue[_OutboundMessage] = asyncio.Queue()

        # Connection state
        self._closed = False
        self._connect_lock = asyncio.Lock()
        self._acquire_lock = asyncio.Lock()  # Ensures atomic capacity check + context creation
        self._last_activity: float = time.time()

        # Background tasks
        self._send_task: asyncio.Task[None] | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None

    @property
    def context_count(self) -> int:
        """Number of contexts using server resources.

        Includes all states (CREATING, ACTIVE, CLOSING) until contextClosed is received
        from the server and the context is removed from _contexts.
        """
        return len(self._contexts)

    @property
    def has_capacity(self) -> bool:
        """Whether this connection can accept more contexts.

        Accounts for both active contexts and pending acquisitions (reserved slots
        that haven't completed context creation yet).
        """
        return (self.context_count + self._pending_acquisitions) < self.MAX_CONTEXTS

    def reserve_capacity(self) -> None:
        """Reserve a slot for a pending context acquisition.

        Call this before releasing the pool lock to prevent other callers from
        over-subscribing this connection. The reservation is released when
        acquire_context() completes (success or failure).
        """
        self._pending_acquisitions += 1

    def release_reservation(self) -> None:
        """Release a previously reserved slot.

        Called automatically by acquire_context() on success, or manually on failure.
        """
        if self._pending_acquisitions > 0:
            self._pending_acquisitions -= 1

    @property
    def is_idle(self) -> bool:
        """Whether this connection has no active contexts and no pending acquisitions."""
        return self.context_count == 0 and self._pending_acquisitions == 0

    @property
    def last_activity(self) -> float:
        """Timestamp of last context acquisition or release."""
        return self._last_activity

    async def connect(self) -> None:
        """Establish WebSocket connection and start background loops."""
        async with self._connect_lock:
            if self._ws is not None or self._closed:
                return

            url = urljoin(self._ws_url, "/tts/v1/voice:streamBidirectional")
            request_id = str(uuid.uuid4())
            self._ws = await self._session.ws_connect(
                url,
                headers={
                    "Authorization": self._authorization,
                    "X-User-Agent": USER_AGENT,
                    "X-Request-Id": request_id,
                },
            )
            logger.debug(
                "Established Inworld TTS WebSocket connection (shared)",
                extra={"request_id": request_id},
            )

            self._send_task = asyncio.create_task(self._send_loop())
            self._recv_task = asyncio.create_task(self._recv_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_stale_contexts())

    async def acquire_context(
        self,
        emitter: tts.AudioEmitter,
        opts: _TTSOptions,
        timeout: float,
    ) -> tuple[str, asyncio.Future[None]]:
        """Acquire a new context for TTS synthesis.

        Note: Caller should check has_capacity before calling this method when using
        a connection pool. This method will still wait if at capacity, but the pool
        should route to connections with available capacity first.
        """
        await self.connect()

        # Fail fast if connection is closed
        if self._closed or self._ws is None:
            raise APIConnectionError("Connection is closed")

        start_time = time.time()

        while True:
            # Check closed state at start of each iteration
            if self._closed or self._ws is None:
                raise APIConnectionError("Connection is closed")

            # Use lock to ensure atomic capacity check + context creation
            async with self._acquire_lock:
                # Check raw context_count, not has_capacity, because we ARE one of
                # the pending acquisitions - has_capacity would always be false
                if self.context_count < self.MAX_CONTEXTS:
                    self._last_activity = time.time()
                    ctx_id = utils.shortuuid()
                    waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()

                    ctx_info = _ContextInfo(
                        context_id=ctx_id,
                        state=_ContextState.CREATING,
                        emitter=emitter,
                        waiter=waiter,
                    )
                    self._contexts[ctx_id] = ctx_info
                    # Release reservation now that we have a real context
                    self.release_reservation()

                    await self._outbound_queue.put(_CreateContextMsg(context_id=ctx_id, opts=opts))
                    return ctx_id, waiter

            # No capacity - wait outside the lock
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            if remaining <= 0:
                raise APITimeoutError()

            try:
                await asyncio.wait_for(self._context_available.wait(), timeout=remaining)
                # Clear after wait returns to avoid lost-wakeup race
                self._context_available.clear()

                # Check closed state after waking
                if self._closed or self._ws is None:
                    raise APIConnectionError("Connection is closed")
            except asyncio.TimeoutError:
                raise APITimeoutError() from None

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

    def close_context(self, context_id: str) -> None:
        """Queue a close message for a context (removes from pool)."""
        ctx = self._contexts.get(context_id)
        if ctx:
            ctx.state = _ContextState.CLOSING
            ctx.close_started_at = time.time()
        try:
            self._outbound_queue.put_nowait(_CloseContextMsg(context_id=context_id))
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
                            "bufferCharThreshold": opts.buffer_char_threshold,
                            "maxBufferDelayMs": opts.max_buffer_delay_ms,
                            "timestampTransportStrategy": opts.timestamp_transport_strategy,
                        },
                        "contextId": msg.context_id,
                    }
                    if is_given(opts.timestamp_type):
                        pkt["create"]["timestampType"] = opts.timestamp_type
                    if is_given(opts.text_normalization):
                        pkt["create"]["applyTextNormalization"] = opts.text_normalization
                    # Always enable auto_mode since we always use SentenceTokenizer
                    pkt["create"]["autoMode"] = True
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
                    pkt = {"close_context": {}, "contextId": msg.context_id}
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
                result = data.get("result", {})
                context_id = result.get("contextId")

                ctx = self._contexts.get(context_id) if context_id else None

                # Check for errors in status
                status = result.get("status", {})
                if status.get("code", 0) != 0:
                    error = APIError(f"Inworld error: {status.get('message', 'Unknown error')}")
                    if ctx:
                        if ctx.waiter and not ctx.waiter.done():
                            ctx.waiter.set_exception(error)
                        # Release the stuck context and signal capacity
                        self._contexts.pop(context_id, None)
                        self._last_activity = time.time()
                        self._context_available.set()
                        if self._on_capacity_available:
                            self._on_capacity_available()
                    continue

                if not ctx:
                    continue

                if "contextCreated" in result:
                    ctx.state = _ContextState.ACTIVE
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
                                        "raw_words": raw_words,
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
                                        "words": [str(ts) for ts in timed_strings],
                                        "adjusted_starts": [ts.start_time for ts in timed_strings],
                                        "adjusted_ends": [ts.end_time for ts in timed_strings],
                                        "generation_end_time": ctx.generation_end_time,
                                    },
                                )

                            for ts in timed_strings:
                                ctx.emitter.push_timed_transcript(ts)

                        if audio_content := audio_chunk.get("audioContent"):
                            ctx.emitter.push(base64.b64decode(audio_content))
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
                    self._contexts.pop(context_id, None)
                    self._last_activity = time.time()
                    self._context_available.set()
                    if self._on_capacity_available:
                        self._on_capacity_available()
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
                    # Resolve waiter before evicting
                    if ctx.waiter and not ctx.waiter.done():
                        ctx.waiter.set_result(None)
                    self._contexts.pop(ctx.context_id, None)
                    self._last_activity = now
                    self._context_available.set()
                    if self._on_capacity_available:
                        self._on_capacity_available()

    async def _handle_connection_error(self, error: Exception) -> None:
        """Handle connection-level error by failing all active contexts."""
        for ctx in list(self._contexts.values()):
            if ctx.waiter and not ctx.waiter.done():
                ctx.waiter.set_exception(error)
        self._contexts.clear()
        self._closed = True

        # Wake local waiters so they can fail fast
        self._context_available.set()

        # Notify pool so callers blocked on capacity can create replacement connections
        if self._on_capacity_available:
            self._on_capacity_available()

    async def aclose(self) -> None:
        """Close the connection and all contexts."""
        self._closed = True

        # Cancel background tasks
        for task in [self._send_task, self._recv_task, self._cleanup_task]:
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


DEFAULT_MAX_CONNECTIONS = 20
DEFAULT_IDLE_CONNECTION_TIMEOUT = 300.0  # 5 minutes


class _ConnectionPool:
    """Manages a pool of _InworldConnection instances for high-concurrency scenarios.

    Each connection supports up to 5 concurrent contexts. The pool automatically creates
    new connections when all existing ones are at capacity, up to max_connections (default 20).
    Idle connections are automatically cleaned up after idle_timeout seconds.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        ws_url: str,
        authorization: str,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        idle_timeout: float = DEFAULT_IDLE_CONNECTION_TIMEOUT,
    ) -> None:
        self._session = session
        self._ws_url = ws_url
        self._authorization = authorization
        self._max_connections = max_connections
        self._idle_timeout = idle_timeout

        self._connections: list[_InworldConnection] = []
        self._pool_lock = asyncio.Lock()
        self._closed = False

        # Event signaled when any connection has capacity available
        self._capacity_available = asyncio.Event()
        self._capacity_available.set()  # Initially available since we can create connections

        # Cleanup task
        self._cleanup_task: asyncio.Task[None] | None = None

    async def acquire_context(
        self,
        emitter: tts.AudioEmitter,
        opts: _TTSOptions,
        timeout: float,
    ) -> tuple[str, asyncio.Future[None], _InworldConnection]:
        """Acquire a context from the pool, creating new connections as needed.

        Returns:
            tuple of (context_id, waiter future, connection) - the connection is returned
            so the caller can call send_text/flush_context/close_context on it.
        """
        if self._closed:
            raise APIConnectionError("Connection pool is closed")

        start_time = time.time()
        remaining_timeout = timeout

        while True:
            conn: _InworldConnection | None = None
            created_new = False

            async with self._pool_lock:
                # Start cleanup task if not already running
                if self._cleanup_task is None:
                    self._cleanup_task = asyncio.create_task(self._cleanup_idle_connections())

                # Prune closed connections first
                self._connections = [c for c in self._connections if not c._closed]

                # First, try to find a connection with capacity
                for existing in self._connections:
                    if not existing._closed and existing.has_capacity:
                        conn = existing
                        break

                # No available capacity - can we create a new connection?
                if conn is None and len(self._connections) < self._max_connections:
                    conn = _InworldConnection(
                        session=self._session,
                        ws_url=self._ws_url,
                        authorization=self._authorization,
                        on_capacity_available=self.notify_capacity_available,
                    )
                    created_new = True
                    # Add to pool IMMEDIATELY so other callers can see it
                    self._connections.append(conn)
                    logger.debug(
                        "Created new Inworld connection",
                        extra={"pool_size": len(self._connections)},
                    )

                # Reserve capacity BEFORE releasing lock so other callers see it
                if conn:
                    conn.reserve_capacity()

            # Acquire context OUTSIDE the lock to avoid head-of-line blocking
            if conn:
                try:
                    ctx_id, waiter = await conn.acquire_context(emitter, opts, remaining_timeout)
                except Exception:
                    # Release reservation since we didn't get a context
                    conn.release_reservation()
                    # Remove failed new connection from pool
                    if created_new:
                        async with self._pool_lock:
                            if conn in self._connections:
                                self._connections.remove(conn)
                        await conn.aclose()
                    raise

                return ctx_id, waiter, conn

            # At max connections and all at capacity - wait for one to free up
            elapsed = time.time() - start_time
            remaining_timeout = timeout - elapsed
            if remaining_timeout <= 0:
                raise APITimeoutError("Timed out waiting for available connection capacity")

            try:
                await asyncio.wait_for(
                    self._capacity_available.wait(),
                    timeout=remaining_timeout,
                )
                # Clear after wait returns to avoid lost-wakeup race
                self._capacity_available.clear()
            except asyncio.TimeoutError:
                raise APITimeoutError(
                    "Timed out waiting for available connection capacity"
                ) from None

    def notify_capacity_available(self) -> None:
        """Called when a context is released and capacity may be available."""
        self._capacity_available.set()

    async def _cleanup_idle_connections(self) -> None:
        """Periodically close idle connections that have been unused for a while."""
        while not self._closed:
            await asyncio.sleep(60.0)  # Check every minute

            async with self._pool_lock:
                now = time.time()
                # Keep at least one connection, close others if idle
                connections_to_close: list[_InworldConnection] = []

                for conn in self._connections:
                    if (
                        conn.is_idle
                        and now - conn.last_activity > self._idle_timeout
                        and len(self._connections) - len(connections_to_close) > 1
                    ):
                        connections_to_close.append(conn)

                for conn in connections_to_close:
                    self._connections.remove(conn)
                    logger.debug(
                        "Closing idle Inworld connection",
                        extra={"pool_size": len(self._connections)},
                    )

            # Close connections outside the lock
            for conn in connections_to_close:
                await conn.aclose()

    async def aclose(self) -> None:
        """Close all connections in the pool."""
        self._closed = True

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        async with self._pool_lock:
            connections = list(self._connections)
            self._connections.clear()

        for conn in connections:
            await conn.aclose()


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[Encoding] = NOT_GIVEN,
        bit_rate: NotGivenOr[int] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        speaking_rate: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        timestamp_type: NotGivenOr[TimestampType] = NOT_GIVEN,
        text_normalization: NotGivenOr[TextNormalization] = NOT_GIVEN,
        timestamp_transport_strategy: NotGivenOr[TimestampTransportStrategy] = NOT_GIVEN,
        buffer_char_threshold: NotGivenOr[int] = NOT_GIVEN,
        max_buffer_delay_ms: NotGivenOr[int] = NOT_GIVEN,
        base_url: str = DEFAULT_URL,
        ws_url: str = DEFAULT_WS_URL,
        http_session: aiohttp.ClientSession | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        retain_format: NotGivenOr[bool] = NOT_GIVEN,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        idle_connection_timeout: float = DEFAULT_IDLE_CONNECTION_TIMEOUT,
    ) -> None:
        """
        Create a new instance of Inworld TTS.

        Args:
            api_key (str, optional): The Inworld API key.
                If not provided, it will be read from the INWORLD_API_KEY environment variable.
            voice (str, optional): The voice to use. Defaults to "Ashley".
            model (str, optional): The Inworld model to use. Defaults to "inworld-tts-1".
            encoding (str, optional): The encoding to use. Defaults to "OGG_OPUS".
            bit_rate (int, optional): Bits per second of the audio. Defaults to 64000.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 48000.
            speaking_rate (float, optional): The speed of the voice, in the range [0.5, 1.5].
                Defaults to 1.0.
            temperature (float, optional): Determines the degree of randomness when sampling audio
                tokens to generate the response. Range [0, 2]. Defaults to 1.1.
            timestamp_type (str, optional): Controls timestamp metadata returned with the audio.
                Use "WORD" for word-level timestamps or "CHARACTER" for character-level.
                Useful for karaoke-style captions, word highlighting, and lipsync.
            text_normalization (str, optional): Controls text normalization. When "ON", numbers,
                dates, and abbreviations are expanded (e.g., "Dr." -> "Doctor"). When "OFF",
                text is read exactly as written. Defaults to automatic.
            timestamp_transport_strategy (str, optional): Controls how timestamp info is
                transported relative to audio data. "SYNC" returns timestamps in the same
                message as audio data. "ASYNC" allows timestamps to return in trailing
                messages after the audio data. Defaults to "ASYNC".
            buffer_char_threshold (int, optional): For streaming, the minimum number of characters
                in the buffer that automatically triggers audio generation. Defaults to 1000.
            max_buffer_delay_ms (int, optional): For streaming, the maximum time in ms to buffer
                before starting generation. Defaults to 3000.
            base_url (str, optional): The base URL for the Inworld TTS API.
                Defaults to "https://api.inworld.ai/".
            ws_url (str, optional): The WebSocket URL for streaming TTS.
                Defaults to "wss://api.inworld.ai/".
            http_session (aiohttp.ClientSession, optional): The HTTP session to use.
            tokenizer (tokenize.SentenceTokenizer, optional): The tokenizer to use for streaming.
                Defaults to `livekit.agents.tokenize.blingfire.SentenceTokenizer`.
            retain_format (bool, optional): Whether to retain the format of the text when tokenizing.
                Defaults to True.
            max_connections (int, optional): Maximum number of concurrent WebSocket connections.
                Each connection supports up to 5 concurrent synthesis streams. Defaults to 20.
            idle_connection_timeout (float, optional): Time in seconds after which idle connections
                are closed. Defaults to 300 (5 minutes).
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

        self._opts = _TTSOptions(
            voice=voice if is_given(voice) else DEFAULT_VOICE,
            model=model if is_given(model) else DEFAULT_MODEL,
            encoding=encoding if is_given(encoding) else DEFAULT_ENCODING,
            bit_rate=bit_rate if is_given(bit_rate) else DEFAULT_BIT_RATE,
            sample_rate=sample_rate if is_given(sample_rate) else DEFAULT_SAMPLE_RATE,
            speaking_rate=speaking_rate if is_given(speaking_rate) else DEFAULT_SPEAKING_RATE,
            temperature=temperature if is_given(temperature) else DEFAULT_TEMPERATURE,
            timestamp_type=timestamp_type,
            text_normalization=text_normalization,
            timestamp_transport_strategy=cast(
                TimestampTransportStrategy, timestamp_transport_strategy
            )
            if is_given(timestamp_transport_strategy)
            else DEFAULT_TIMESTAMP_TRANSPORT_STRATEGY,
            buffer_char_threshold=buffer_char_threshold
            if is_given(buffer_char_threshold)
            else DEFAULT_BUFFER_CHAR_THRESHOLD,
            max_buffer_delay_ms=max_buffer_delay_ms
            if is_given(max_buffer_delay_ms)
            else DEFAULT_MAX_BUFFER_DELAY_MS,
        )

        self._max_connections = max_connections
        self._idle_connection_timeout = idle_connection_timeout
        self._pool: _ConnectionPool | None = None
        self._pool_lock = asyncio.Lock()
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._sentence_tokenizer = (
            tokenizer
            if is_given(tokenizer)
            else tokenize.blingfire.SentenceTokenizer(
                retain_format=retain_format if is_given(retain_format) else True
            )
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Inworld"

    async def _get_pool(self) -> _ConnectionPool:
        """Get the connection pool, creating if needed."""
        async with self._pool_lock:
            if self._pool is None or self._pool._closed:
                self._pool = _ConnectionPool(
                    session=self._ensure_session(),
                    ws_url=self._ws_url,
                    authorization=self._authorization,
                    max_connections=self._max_connections,
                    idle_timeout=self._idle_connection_timeout,
                )
            return self._pool

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[Encoding] = NOT_GIVEN,
        bit_rate: NotGivenOr[int] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        speaking_rate: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        timestamp_type: NotGivenOr[TimestampType] = NOT_GIVEN,
        text_normalization: NotGivenOr[TextNormalization] = NOT_GIVEN,
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
            timestamp_type (str, optional): Controls timestamp metadata ("WORD" or "CHARACTER").
            text_normalization (str, optional): Controls text normalization ("ON" or "OFF").
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
            self._opts.encoding = encoding
        if is_given(bit_rate):
            self._opts.bit_rate = bit_rate
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(speaking_rate):
            self._opts.speaking_rate = speaking_rate
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(timestamp_type):
            self._opts.timestamp_type = cast(TimestampType, timestamp_type)
        if is_given(text_normalization):
            self._opts.text_normalization = cast(TextNormalization, text_normalization)
        if is_given(timestamp_transport_strategy):
            self._opts.timestamp_transport_strategy = cast(
                TimestampTransportStrategy, timestamp_transport_strategy
            )
        if is_given(buffer_char_threshold):
            self._opts.buffer_char_threshold = buffer_char_threshold
        if is_given(max_buffer_delay_ms):
            self._opts.max_buffer_delay_ms = max_buffer_delay_ms

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def prewarm(self) -> None:
        asyncio.create_task(self._prewarm_impl())

    async def _prewarm_impl(self) -> None:
        # Just ensure the pool is created - first acquire will establish a connection
        await self._get_pool()

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
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        if self._pool:
            await self._pool.aclose()
            self._pool = None

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
            if utils.is_given(self._opts.timestamp_type):
                body_params["timestampType"] = self._opts.timestamp_type
            if utils.is_given(self._opts.text_normalization):
                body_params["applyTextNormalization"] = self._opts.text_normalization
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
                        logger.warning("failed to parse Inworld response line: %s", line)
                        continue

                    if result := data.get("result"):
                        # Handle timestamp info if present
                        if timestamp_info := result.get("timestampInfo"):
                            timed_strings = _parse_timestamp_info(timestamp_info)
                            if timed_strings:
                                output_emitter.push_timed_transcript(timed_strings)

                        if audio_content := result.get("audioContent"):
                            output_emitter.push(base64.b64decode(audio_content))
                    elif error := data.get("error"):
                        raise APIStatusError(
                            message=error.get("message"),
                            status_code=error.get("code"),
                            request_id=x_request_id,
                            body=None,
                        )
                output_emitter.flush()
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

        pool = await self._tts._get_pool()
        context_id, waiter, connection = await pool.acquire_context(
            emitter=output_emitter,
            opts=self._opts,
            timeout=self._conn_options.timeout,
        )

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_tokenizer_stream.flush()
                    continue
                sent_tokenizer_stream.push_text(data)
            sent_tokenizer_stream.end_input()

        async def _send_task() -> None:
            async for ev in sent_tokenizer_stream:
                text = ev.token
                # Chunk to stay within Inworld's 1000 char limit
                for i in range(0, len(text), 1000):
                    connection.send_text(context_id, text[i : i + 1000])
                    self._mark_started()
            connection.flush_context(context_id)
            connection.close_context(context_id)

        tasks = [
            asyncio.create_task(_input_task()),
            asyncio.create_task(_send_task()),
        ]

        try:
            await asyncio.wait_for(waiter, timeout=self._conn_options.timeout + 60)
        except asyncio.TimeoutError:
            connection.close_context(context_id)
            raise APITimeoutError() from None
        except asyncio.CancelledError:
            connection.close_context(context_id)
            raise
        except APIError:
            raise
        except Exception as e:
            logger.error("Inworld stream error", extra={"context_id": context_id, "error": e})
            connection.close_context(context_id)
            raise APIConnectionError() from e
        finally:
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

        last_idx = len(words) - 1
        for idx, (word, start, end) in enumerate(zip(words, starts, ends, strict=False)):
            # Each word gets a trailing space so that when the synchronizer concatenates
            # them via `pushed_text += text`, the transcript reads naturally.
            text = f"{word} " if idx < last_idx else word
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
