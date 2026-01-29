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
import json
import os
import weakref
from dataclasses import dataclass, replace
from typing import Any, Union

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import TTSEncoding, TTSLanguages, TTSModels

# Voice.AI TTS outputs at 32kHz sample rate
SAMPLE_RATE = 32000

# Default encoding for audio output
_DefaultEncoding: TTSEncoding = "mp3"

# API configuration
API_BASE_URL = "https://dev.voice.ai"
WS_INACTIVITY_TIMEOUT = 180
WS_CONNECT_TIMEOUT = 30


def _get_content_type(encoding: TTSEncoding) -> str:
    """Get MIME type for audio format"""
    return {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(encoding, "audio/mpeg")


@dataclass
class Voice:
    """Represents a Voice.AI voice.

    Attributes:
        id: Unique voice identifier
        name: Display name of the voice
        status: Current status of the voice (e.g., "ready", "training")
    """

    id: str
    name: str
    status: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        model: TTSModels | str = "voiceai-tts-v1-latest",
        encoding: NotGivenOr[TTSEncoding] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        language: TTSLanguages | str = "en",
        temperature: float = 1.0,
        top_p: float = 0.8,
        inactivity_timeout: int = WS_INACTIVITY_TIMEOUT,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer | tokenize.SentenceTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Voice.AI TTS.

        Args:
            voice_id (NotGivenOr[str]): Voice ID. If not provided, uses the default built-in voice.
            model (TTSModels | str): TTS model to use. Defaults to "voiceai-tts-v1-latest".
            encoding (NotGivenOr[TTSEncoding]): Audio output format. Defaults to "mp3".
                Options: "mp3" (compressed), "wav" (uncompressed), "pcm" (raw 16-bit).
            api_key (NotGivenOr[str]): Voice.AI API key. Can be set via argument or
                `VOICEAI_API_KEY` environment variable.
            base_url (NotGivenOr[str]): Custom base URL for the API. Defaults to Voice.AI production.
            language (TTSLanguages | str): Language code (ISO 639-1). Defaults to "en".
                Supported: en, ca, sv, es, fr, de, it, pt, pl, ru, nl.
            temperature (float): Sampling temperature (0.0-2.0). Defaults to 1.0.
            top_p (float): Nucleus sampling parameter (0.0-1.0). Defaults to 0.8.
            inactivity_timeout (int): Inactivity timeout in seconds for WebSocket. Defaults to 180.
            word_tokenizer (NotGivenOr[tokenize.WordTokenizer | tokenize.SentenceTokenizer]):
                Tokenizer for processing text. Defaults to SentenceTokenizer.
            http_session (aiohttp.ClientSession | None): Custom HTTP session. Optional.
        """

        if not is_given(encoding):
            encoding = _DefaultEncoding

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=False,  # Voice.AI doesn't provide word-level alignment yet
            ),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )

        voiceai_api_key = api_key if is_given(api_key) else os.environ.get("VOICEAI_API_KEY")
        if not voiceai_api_key:
            raise ValueError(
                "Voice.AI API key is required, either as argument or set VOICEAI_API_KEY environment variable"
            )

        if not is_given(word_tokenizer):
            word_tokenizer = tokenize.blingfire.SentenceTokenizer()

        self._opts = _TTSOptions(
            voice_id=voice_id if is_given(voice_id) else None,
            model=model,
            api_key=voiceai_api_key,
            base_url=base_url if is_given(base_url) else API_BASE_URL,
            encoding=encoding,
            language=language,
            temperature=temperature,
            top_p=top_p,
            inactivity_timeout=inactivity_timeout,
            word_tokenizer=word_tokenizer,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

        self._current_connection: _Connection | None = None
        self._connection_lock = asyncio.Lock()

    @property
    def model(self) -> str:
        """The TTS model being used."""
        return self._opts.model

    @property
    def provider(self) -> str:
        """The provider name (VoiceAI)."""
        return "VoiceAI"

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session exists, creating one if needed."""
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authorization headers for API requests"""
        return {
            "Authorization": f"Bearer {self._opts.api_key}",
            "Content-Type": "application/json",
        }

    async def list_voices(self) -> list[Voice]:
        """List voices owned by the authenticated user"""
        async with self._ensure_session().get(
            f"{self._opts.base_url}/api/v1/tts/voices",
            headers=self._get_auth_headers(),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return [
                Voice(id=v["voice_id"], name=v.get("name", ""), status=v.get("status", ""))
                for v in data
            ]

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[TTSLanguages | str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Update TTS options at runtime.

        Args:
            voice_id (NotGivenOr[str]): Voice ID to use.
            model (NotGivenOr[TTSModels | str]): TTS model to use.
            language (NotGivenOr[TTSLanguages | str]): Language code.
            temperature (NotGivenOr[float]): Sampling temperature (0.0-2.0).
            top_p (NotGivenOr[float]): Nucleus sampling parameter (0.0-1.0).
        """
        changed = False

        if is_given(model) and model != self._opts.model:
            self._opts.model = model
            changed = True

        if is_given(voice_id):
            new_voice_id = voice_id if voice_id else None
            if new_voice_id != self._opts.voice_id:
                self._opts.voice_id = new_voice_id
                changed = True

        if is_given(language) and language != self._opts.language:
            self._opts.language = language
            changed = True

        if is_given(temperature) and temperature != self._opts.temperature:
            self._opts.temperature = temperature
            changed = True

        if is_given(top_p) and top_p != self._opts.top_p:
            self._opts.top_p = top_p
            changed = True

        if changed and self._current_connection:
            self._current_connection.mark_non_current()
            self._current_connection = None

    async def current_connection(self) -> _Connection:
        """Get the current WebSocket connection, creating one if needed"""
        async with self._connection_lock:
            if (
                self._current_connection
                and self._current_connection.is_current
                and not self._current_connection._closed
            ):
                return self._current_connection

            session = self._ensure_session()
            conn = _Connection(self._opts, session)
            await conn.connect()
            self._current_connection = conn
            return conn

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        """Synthesize text to speech (non-streaming HTTP request)"""
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Stream text to speech over WebSocket (multi-stream)"""
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close all active streams and connections."""
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()

        if self._current_connection:
            await self._current_connection.aclose()
            self._current_connection = None


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the HTTP streaming endpoint"""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        """Initialize chunked stream for HTTP-based synthesis."""
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Execute HTTP streaming synthesis request."""
        request_body: dict[str, Any] = {
            "text": self._input_text,
            "audio_format": self._opts.encoding,
            "language": self._opts.language,
            "temperature": self._opts.temperature,
            "top_p": self._opts.top_p,
        }

        # Only include voice_id if specified
        if self._opts.voice_id:
            request_body["voice_id"] = self._opts.voice_id

        # Only include model if explicitly set (API auto-selects based on language)
        if self._opts.model:
            request_body["model"] = self._opts.model

        try:
            async with self._tts._ensure_session().post(
                f"{self._opts.base_url}/api/v1/tts/speech/stream",
                headers=self._tts._get_auth_headers(),
                json=request_body,
                timeout=aiohttp.ClientTimeout(
                    total=60,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    mime_type=_get_content_type(self._opts.encoding),
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streamed TTS using WebSocket multi-stream API

    Uses Voice.AI multi-stream WebSocket:
    wss://dev.voice.ai/api/v1/tts/multi-stream
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        """Initialize WebSocket-based synthesis stream."""
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._context_id = utils.shortuuid()
        self._sent_tokenizer_stream = self._opts.word_tokenizer.stream()
        self._connection: _Connection | None = None

    async def aclose(self) -> None:
        """Close the synthesis stream and clean up resources."""
        await self._sent_tokenizer_stream.aclose()
        await super().aclose()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Execute WebSocket streaming synthesis."""
        output_emitter.initialize(
            request_id=self._context_id,
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            stream=True,
            mime_type=_get_content_type(self._opts.encoding),
        )
        output_emitter.start_segment(segment_id=self._context_id)

        connection: _Connection
        try:
            connection = await asyncio.wait_for(
                self._tts.current_connection(), self._conn_options.timeout
            )
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            raise APIConnectionError("could not connect to Voice.AI") from e

        waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        connection.register_stream(self, output_emitter, waiter)

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    self._sent_tokenizer_stream.flush()
                    continue
                self._sent_tokenizer_stream.push_text(data)
            self._sent_tokenizer_stream.end_input()

        async def _sentence_stream_task() -> None:
            is_first_message = True
            async for data in self._sent_tokenizer_stream:
                text = data.token

                if is_first_message:
                    # First message includes all parameters (init message)
                    connection.send_content(
                        _SynthesizeContent(
                            context_id=self._context_id,
                            text=text,
                            flush=True,
                            is_init=True,
                            voice_id=self._opts.voice_id,
                            language=self._opts.language,
                            model=self._opts.model,
                            audio_format=self._opts.encoding,
                            temperature=self._opts.temperature,
                            top_p=self._opts.top_p,
                        )
                    )
                    is_first_message = False
                else:
                    # Subsequent messages are text-only
                    connection.send_content(
                        _SynthesizeContent(
                            context_id=self._context_id,
                            text=text,
                            flush=True,
                            is_init=False,
                        )
                    )
                self._mark_started()

            # Close the context when input stream ends - waiter will complete when context_closed is received
            connection.close_context(self._context_id)

        input_t = asyncio.create_task(_input_task())
        stream_t = asyncio.create_task(_sentence_stream_task())

        try:
            await waiter
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            if isinstance(e, APIStatusError):
                raise e
            raise APIStatusError("Could not synthesize") from e
        finally:
            output_emitter.end_segment()
            await utils.aio.gracefully_cancel(input_t, stream_t)


@dataclass
class _TTSOptions:
    """Internal TTS configuration options."""

    api_key: str
    voice_id: str | None
    model: TTSModels | str
    language: TTSLanguages | str
    base_url: str
    encoding: TTSEncoding
    temperature: float
    top_p: float
    inactivity_timeout: int
    word_tokenizer: tokenize.WordTokenizer | tokenize.SentenceTokenizer


@dataclass
class _SynthesizeContent:
    """Message sent to WebSocket to synthesize text content."""

    context_id: str
    text: str
    flush: bool = False
    is_init: bool = False
    voice_id: str | None = None
    language: str | None = None
    model: str | None = None
    audio_format: str | None = None
    temperature: float | None = None
    top_p: float | None = None


@dataclass
class _CloseContext:
    """Message to close a synthesis context."""

    context_id: str


@dataclass
class _StreamData:
    """Tracking data for an active synthesis stream."""

    emitter: tts.AudioEmitter
    stream: SynthesizeStream
    waiter: asyncio.Future[None]
    timeout_timer: asyncio.TimerHandle | None = None


class _Connection:
    """Manages a single WebSocket connection with send/recv loops for multi-context TTS"""

    def __init__(self, opts: _TTSOptions, session: aiohttp.ClientSession):
        """Initialize WebSocket connection manager.

        Args:
            opts: TTS configuration options
            session: HTTP session for WebSocket connection
        """
        self._opts = opts
        self._session = session
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._is_current = True
        self._active_contexts: set[str] = set()
        self._input_queue = utils.aio.Chan[Union[_SynthesizeContent, _CloseContext]]()

        self._context_data: dict[str, _StreamData] = {}

        self._send_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None
        self._closed = False

    @property
    def is_current(self) -> bool:
        """Whether this connection is the current active connection."""
        return self._is_current

    def mark_non_current(self) -> None:
        """Mark this connection as no longer current - it will shut down when drained"""
        self._is_current = False

    async def connect(self) -> None:
        """Establish WebSocket connection and start send/recv loops"""
        if self._ws or self._closed:
            return

        # Build WebSocket URL for multi-stream
        url = f"{self._opts.base_url}/api/v1/tts/multi-stream".replace(
            "https://", "wss://"
        ).replace("http://", "ws://")

        headers = {"Authorization": f"Bearer {self._opts.api_key}"}
        self._ws = await asyncio.wait_for(
            self._session.ws_connect(url, headers=headers), WS_CONNECT_TIMEOUT
        )

        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())

    def register_stream(
        self, stream: SynthesizeStream, emitter: tts.AudioEmitter, done_fut: asyncio.Future[None]
    ) -> None:
        """Register a new synthesis stream with this connection"""
        context_id = stream._context_id
        self._context_data[context_id] = _StreamData(
            emitter=emitter, stream=stream, waiter=done_fut
        )

    def send_content(self, content: _SynthesizeContent) -> None:
        """Send synthesis content to the connection.

        Args:
            content: Synthesis content message to send
        """
        if self._closed or not self._ws or self._ws.closed:
            raise APIConnectionError("WebSocket connection is closed")
        self._input_queue.send_nowait(content)

    def close_context(self, context_id: str) -> None:
        """Close a specific synthesis context.

        Args:
            context_id: ID of the context to close
        """
        if self._closed or not self._ws or self._ws.closed:
            raise APIConnectionError("WebSocket connection is closed")
        self._input_queue.send_nowait(_CloseContext(context_id))

    async def _send_loop(self) -> None:
        """Send loop - processes messages from input queue"""
        try:
            while not self._closed:
                try:
                    msg = await self._input_queue.recv()
                except utils.aio.ChanClosed:
                    break

                if not self._ws or self._ws.closed:
                    break

                if isinstance(msg, _SynthesizeContent):
                    is_new_context = msg.context_id not in self._active_contexts

                    # If not current and this is a new context, ignore it
                    if not self._is_current and is_new_context:
                        continue

                    if msg.is_init:
                        # Init message with full parameters
                        pkt: dict[str, Any] = {
                            "context_id": msg.context_id,
                            "text": msg.text,
                            "language": msg.language or self._opts.language,
                            "flush": msg.flush,
                        }
                        if msg.voice_id:
                            pkt["voice_id"] = msg.voice_id
                        if msg.model:
                            pkt["model"] = msg.model
                        if msg.audio_format:
                            pkt["audio_format"] = msg.audio_format
                        if msg.temperature is not None:
                            pkt["temperature"] = msg.temperature
                        if msg.top_p is not None:
                            pkt["top_p"] = msg.top_p

                        self._active_contexts.add(msg.context_id)
                    else:
                        # Text-only message for existing context
                        pkt = {
                            "context_id": msg.context_id,
                            "text": msg.text,
                            "flush": msg.flush,
                        }

                    # Start timeout timer for this context
                    self._start_timeout_timer(msg.context_id)
                    await self._ws.send_json(pkt)

                elif isinstance(msg, _CloseContext):
                    if msg.context_id in self._active_contexts:
                        close_pkt = {
                            "context_id": msg.context_id,
                            "close_context": True,
                        }
                        await self._ws.send_json(close_pkt)

        except Exception as e:
            logger.warning("send loop error", exc_info=e)
        finally:
            if not self._closed:
                await self.aclose()

    async def _recv_loop(self) -> None:
        """Receive loop - processes messages from WebSocket

        Message types:
        - `is_last`: Sent after each flush completes (indicates inference is done, context still active)
        - `context_closed`: Sent when a context is explicitly closed (confirmation, clean up context)
        """
        try:
            while not self._closed and self._ws and not self._ws.closed:
                msg = await self._ws.receive()

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if not self._closed and len(self._context_data) > 0:
                        logger.warning("websocket closed unexpectedly")
                    break

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                context_id = data.get("context_id")
                ctx = self._context_data.get(context_id) if context_id is not None else None

                # Handle errors
                if error := data.get("error"):
                    logger.error(
                        "voiceai tts returned error",
                        extra={"context_id": context_id, "error": error, "data": data},
                    )
                    if context_id is not None:
                        if ctx and not ctx.waiter.done():
                            ctx.waiter.set_exception(APIError(message=error))
                        self._cleanup_context(context_id)
                    continue

                if ctx is None:
                    # If it's a context_closed or is_last message for an already-cleaned context, just ignore it
                    if data.get("context_closed") or data.get("is_last"):
                        continue
                    logger.warning(
                        "unexpected message received from voiceai tts - context_id not found in registered contexts",
                        extra={
                            "context_id": context_id,
                            "registered": list(self._context_data.keys()),
                            "data": data,
                        },
                    )
                    continue

                emitter = ctx.emitter

                # Handle audio chunks
                if audio_data := data.get("audio"):
                    b64data = base64.b64decode(audio_data)
                    emitter.push(b64data)
                    if ctx.timeout_timer:
                        ctx.timeout_timer.cancel()
                        ctx.timeout_timer = None

                # Handle flush completion (is_last)
                # is_last indicates that inference for this flush is complete - context remains active
                if data.get("is_last"):
                    # Just a flush completion - don't complete waiter here
                    pass

                # Handle context closure (context_closed)
                # context_closed is confirmation that the context was explicitly closed - complete waiter and cleanup
                if data.get("context_closed"):
                    if not ctx.waiter.done():
                        ctx.waiter.set_result(None)
                    self._cleanup_context(context_id)

                    if not self._is_current and not self._active_contexts:
                        break

        except Exception as e:
            logger.warning("recv loop error", exc_info=e)
            for ctx in self._context_data.values():
                if not ctx.waiter.done():
                    ctx.waiter.set_exception(e)
                if ctx.timeout_timer:
                    ctx.timeout_timer.cancel()
            self._context_data.clear()
        finally:
            if not self._closed:
                await self.aclose()

    def _cleanup_context(self, context_id: str) -> None:
        """Clean up context state"""
        ctx = self._context_data.pop(context_id, None)
        if ctx and ctx.timeout_timer:
            ctx.timeout_timer.cancel()

        self._active_contexts.discard(context_id)

    def _start_timeout_timer(self, context_id: str) -> None:
        """Start a timeout timer for a context"""
        if not (ctx := self._context_data.get(context_id)) or ctx.timeout_timer:
            return

        timeout = self._opts.inactivity_timeout

        def _on_timeout() -> None:
            if not ctx.waiter.done():
                ctx.waiter.set_exception(
                    APITimeoutError(f"Voice.AI TTS timed out after {timeout} seconds")
                )
            self._cleanup_context(context_id)

        ctx.timeout_timer = asyncio.get_event_loop().call_later(timeout, _on_timeout)

    async def aclose(self) -> None:
        """Close the connection and clean up"""
        if self._closed:
            return

        self._closed = True
        self._input_queue.close()

        for ctx in self._context_data.values():
            if not ctx.waiter.done():
                ctx.waiter.set_exception(APIStatusError("connection closed"))
            if ctx.timeout_timer:
                ctx.timeout_timer.cancel()
        self._context_data.clear()

        if self._ws:
            await self._ws.close()

        if self._send_task:
            await utils.aio.gracefully_cancel(self._send_task)
        if self._recv_task:
            await utils.aio.gracefully_cancel(self._recv_task)

        self._ws = None
