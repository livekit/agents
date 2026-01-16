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
import weakref
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Literal, Union, cast
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

Encoding = Union[Literal["LINEAR16", "MP3", "OGG_OPUS", "ALAW", "MULAW", "FLAC"], str]
TimestampType = Literal["TIMESTAMP_TYPE_UNSPECIFIED", "WORD", "CHARACTER"]
TextNormalization = Literal["APPLY_TEXT_NORMALIZATION_UNSPECIFIED", "ON", "OFF"]


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


_OutboundMessage = Union[_CreateContextMsg, _SendTextMsg, _FlushContextMsg, _CloseContextMsg]


class _InworldConnection:
    """Manages a single shared WebSocket connection with up to 5 concurrent contexts."""

    MAX_CONTEXTS = 5

    def __init__(self, tts_instance: "TTS") -> None:
        self._tts = tts_instance
        self._ws: aiohttp.ClientWebSocketResponse | None = None

        # Context pool
        self._contexts: dict[str, _ContextInfo] = {}
        self._context_available = asyncio.Event()

        # Message queue for outbound messages
        self._outbound_queue: asyncio.Queue[_OutboundMessage] = asyncio.Queue()

        # Connection state
        self._closed = False
        self._connect_lock = asyncio.Lock()

        # Background tasks
        self._send_task: asyncio.Task[None] | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        """Establish WebSocket connection and start background loops."""
        async with self._connect_lock:
            if self._ws is not None or self._closed:
                return

            session = self._tts._ensure_session()
            url = urljoin(self._tts._ws_url, "/tts/v1/voice:streamBidirectional")
            self._ws = await session.ws_connect(
                url, headers={"Authorization": self._tts._authorization}
            )
            logger.debug("Established Inworld TTS WebSocket connection (shared)")

            self._send_task = asyncio.create_task(self._send_loop())
            self._recv_task = asyncio.create_task(self._recv_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_stale_contexts())

    async def acquire_context(
        self,
        emitter: tts.AudioEmitter,
        opts: _TTSOptions,
        timeout: float,
    ) -> tuple[str, asyncio.Future[None]]:
        """Acquire a new context for TTS synthesis."""
        await self.connect()

        # Always create a fresh context (context reuse is problematic with interruptions)
        # The main optimization is connection reuse, which we get by sharing the WebSocket

        # Wait if we're at the context limit
        while len([c for c in self._contexts.values() if c.state != _ContextState.CLOSING]) >= self.MAX_CONTEXTS:
            try:
                self._context_available.clear()
                await asyncio.wait_for(self._context_available.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                raise APITimeoutError() from None

        ctx_id = utils.shortuuid()
        waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()

        ctx_info = _ContextInfo(
            context_id=ctx_id,
            state=_ContextState.CREATING,
            emitter=emitter,
            waiter=waiter,
        )
        self._contexts[ctx_id] = ctx_info

        await self._outbound_queue.put(_CreateContextMsg(context_id=ctx_id, opts=opts))
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

    def close_context(self, context_id: str) -> None:
        """Queue a close message for a context (removes from pool)."""
        ctx = self._contexts.get(context_id)
        if ctx:
            ctx.state = _ContextState.CLOSING
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
                        },
                        "contextId": msg.context_id,
                    }
                    if is_given(opts.timestamp_type):
                        pkt["create"]["timestampType"] = opts.timestamp_type
                    if is_given(opts.text_normalization):
                        pkt["create"]["applyTextNormalization"] = opts.text_normalization
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
                    if ctx and ctx.waiter and not ctx.waiter.done():
                        ctx.waiter.set_exception(error)
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

                        if timestamp_info := audio_chunk.get("timestampInfo"):
                            timed_strings = _parse_timestamp_info(timestamp_info)
                            for ts in timed_strings:
                                ctx.emitter.push_timed_transcript(ts)

                        if audio_content := audio_chunk.get("audioContent"):
                            ctx.emitter.push(base64.b64decode(audio_content))
                    continue

                if "flushCompleted" in result:
                    continue

                if "contextClosed" in result:
                    if ctx.waiter and not ctx.waiter.done():
                        ctx.waiter.set_result(None)
                    self._contexts.pop(context_id, None)
                    self._context_available.set()
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
                if ctx.state == _ContextState.CLOSING and now - ctx.created_at > 120.0:
                    self._contexts.pop(ctx.context_id, None)
                    self._context_available.set()

    async def _handle_connection_error(self, error: Exception) -> None:
        """Handle connection-level error by failing all active contexts."""
        for ctx in list(self._contexts.values()):
            if ctx.waiter and not ctx.waiter.done():
                ctx.waiter.set_exception(error)
        self._contexts.clear()
        self._closed = True

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
        buffer_char_threshold: NotGivenOr[int] = NOT_GIVEN,
        max_buffer_delay_ms: NotGivenOr[int] = NOT_GIVEN,
        base_url: str = DEFAULT_URL,
        ws_url: str = DEFAULT_WS_URL,
        http_session: aiohttp.ClientSession | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        retain_format: NotGivenOr[bool] = NOT_GIVEN,
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
            raise ValueError("Inworld API key required. Set INWORLD_API_KEY or provide api_key.")

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
            buffer_char_threshold=buffer_char_threshold
            if is_given(buffer_char_threshold)
            else DEFAULT_BUFFER_CHAR_THRESHOLD,
            max_buffer_delay_ms=max_buffer_delay_ms
            if is_given(max_buffer_delay_ms)
            else DEFAULT_MAX_BUFFER_DELAY_MS,
        )

        self._connection: _InworldConnection | None = None
        self._connection_lock = asyncio.Lock()
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

    async def _get_connection(self) -> _InworldConnection:
        """Get the shared connection, creating if needed."""
        async with self._connection_lock:
            if self._connection is None or self._connection._closed:
                self._connection = _InworldConnection(self)
            return self._connection

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
        conn = await self._get_connection()
        await conn.connect()

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
            headers={"Authorization": self._authorization},
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

            async with self._tts._ensure_session().post(
                urljoin(self._tts._base_url, "/tts/v1/voice:stream"),
                headers={
                    "Authorization": self._tts._authorization,
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
                            output_emitter.flush()
                    elif error := data.get("error"):
                        raise APIStatusError(
                            message=error.get("message"),
                            status_code=error.get("code"),
                            request_id=request_id,
                            body=None,
                        )
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
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

        connection = await self._tts._get_connection()
        context_id, waiter = await connection.acquire_context(
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
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            await sent_tokenizer_stream.aclose()
            output_emitter.end_input()


def _parse_timestamp_info(timestamp_info: dict[str, Any]) -> list[TimedString]:
    """Parse timestamp info from API response into TimedString objects."""
    timed_strings: list[TimedString] = []

    # Handle word-level alignment
    if word_align := timestamp_info.get("wordAlignment"):
        words = word_align.get("words", [])
        starts = word_align.get("wordStartTimeSeconds", [])
        ends = word_align.get("wordEndTimeSeconds", [])

        for word, start, end in zip(words, starts, ends):
            timed_strings.append(TimedString(word, start_time=start, end_time=end))

    # Handle character-level alignment
    if char_align := timestamp_info.get("characterAlignment"):
        chars = char_align.get("characters", [])
        starts = char_align.get("characterStartTimeSeconds", [])
        ends = char_align.get("characterEndTimeSeconds", [])

        for char, start, end in zip(chars, starts, ends):
            timed_strings.append(TimedString(char, start_time=start, end_time=end))

    return timed_strings
