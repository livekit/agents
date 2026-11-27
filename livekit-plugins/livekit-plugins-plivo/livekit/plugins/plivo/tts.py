# Copyright 2026 LiveKit, Inc.
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
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from .log import logger
from .models import TTSDefaultVoiceId, TTSLanguages, TTSModels, TTSSampleRates

SILENT_LIMIT = 20.0  # Plivo pings every 15 s: a connection silent this long takes no new stream
DEAD_LIMIT = 60.0  # Plivo drops a connection after 30 s without a pong
MAX_TEXT_LENGTH = 80_000  # at most 12 bytes a character as JSON: a message over 1 MiB closes it
CLOSE_TIMEOUT = 2.0

_T = TypeVar("_T")


async def _open_within(
    seconds: float | None, opening: Awaitable[_T], release: Callable[[_T], object]
) -> _T:
    """Await `opening` for at most `seconds` (None: no limit); release what it opens too late."""
    # not a timeout that cancels the caller: on Python 3.10 that can turn its own cancel into one
    task = asyncio.ensure_future(opening)
    done: set[asyncio.Future[_T]] = set()
    try:
        done, _ = await asyncio.wait((task,), timeout=seconds)
    finally:
        if not done:
            task.cancel()
            task.add_done_callback(
                lambda task: task.cancelled() or task.exception() or release(task.result())
            )
    if not done:
        raise asyncio.TimeoutError
    return task.result()


def _abort(ws: aiohttp.ClientWebSocketResponse[bool]) -> None:
    ws._writer.transport.abort()  # transport.close() waits on a flush that a peer can hold
    ws._response.close()  # gives its connection back to the session's pool


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: str
    language: LanguageCode | None
    speed: float | None
    volume: float | None
    pronunciation_dictionary_id: str | None
    seed: int | None
    continuation: bool | None
    text_normalization: bool | None
    use_normalized_timestamps: bool | None
    word_timestamps: bool
    sample_rate: int

    def request_fields(self, *, streaming: bool) -> dict[str, Any]:
        """Return the options as the fields of a request, without those that are None."""
        pcm = {"container": "raw", "encoding": "pcm_s16le", "sample_rate": self.sample_rate}
        language = self.language
        if language and language.language != "en":  # only English takes a region
            language = LanguageCode(language.language)
        fields = {
            "model_id": self.model,
            "voice_id": self.voice,
            "language": language,
            "pace": self.speed,
            "volume": self.volume,
            "pronunciation_dictionary_id": self.pronunciation_dictionary_id,
            "seed": self.seed,
            "continuation": self.continuation,
            "text_normalization": self.text_normalization,
            "output_format": pcm,
            "use_normalized_timestamps": self.use_normalized_timestamps if streaming else None,
            "timestamps": self.word_timestamps if streaming else None,
        }
        return {name: value for name, value in fields.items() if value is not None}


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = "koyal-v1",
        voice: str = TTSDefaultVoiceId,
        language: TTSLanguages | str | None = None,
        speed: float | None = None,
        volume: float | None = None,
        pronunciation_dictionary_id: str | None = None,
        seed: int | None = None,
        continuation: bool | None = None,
        text_normalization: bool | None = None,
        use_normalized_timestamps: bool | None = None,
        word_timestamps: bool = True,
        sample_rate: TTSSampleRates = 24000,
        auth_id: str | None = None,
        auth_token: str | None = None,
        client_ref: str | None = None,
        base_url: str = "https://api.plivo.com",
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Plivo TTS.

        See https://www.plivo.com/docs/models/text-to-speech/overview for more details on Koyal,
        Plivo's TTS model.

        Args:
            model (TTSModels | str): TTS model to use. Defaults to "koyal-v1".
            voice (str): ID of the voice. Defaults to Hailey, a US English voice.
            language (TTSLanguages | str | None): Language code of the text, such as "hi" or
                "en-IN". For English, the region ("en-IN", "en-US", "en-GB" or "en-AU") sets how
                numbers and dates are normalized; the voice keeps its accent. By default, the TTS
                model finds the language of each sentence among the voice's languages.
            speed (float | None): Speaking rate, from 0.7 to 1.2. Defaults to 1.0.
            volume (float | None): Volume of the speech, from 0.5 to 2.0. Defaults to 1.0.
            pronunciation_dictionary_id (str | None): ID of a pronunciation dictionary. Its rules
                can be used to set the pronunciation of a word in IPA, in ARPAbet or as a
                respelling.
            seed (int | None): Seed of the speech, from 0 to 2**53 - 1. Defaults to the voice's
                default seed.
            continuation (bool | None): Whether the speech of a stream continues from the
                previous stream's, for continuity between them. Realtime API only. Defaults to
                True.
            text_normalization (bool | None): Whether the TTS model normalizes the text before
                speech: it spells out numbers, dates and similar text. Defaults to True.
            use_normalized_timestamps (bool | None): Whether the word timestamps follow the
                normalized text in place of the text as sent. Realtime API only. Defaults to
                False.
            word_timestamps (bool): Whether the Realtime API sends timestamps, which give
                LiveKit's aligned transcript its word times. Defaults to True.
            sample_rate (TTSSampleRates): Sample rate of the audio in Hz. Defaults to 24000.
            auth_id (str | None): Plivo auth ID. If not given, it is read from the
                `PLIVO_AUTH_ID` environment variable.
            auth_token (str | None): Plivo auth token. If not given, it is read from the
                `PLIVO_AUTH_TOKEN` environment variable.
            client_ref (str | None): Label of 1 to 64 letters, digits, "_" or "-". The API ID of
                each request starts with it.
            base_url (str): Base URL of the API. Defaults to "https://api.plivo.com".
            http_session (aiohttp.ClientSession | None): Session for the HTTP and WebSocket
                requests. Defaults to the job's session.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=word_timestamps),
            sample_rate=sample_rate,
            num_channels=1,
        )
        auth_id = auth_id or os.environ.get("PLIVO_AUTH_ID")
        auth_token = auth_token or os.environ.get("PLIVO_AUTH_TOKEN")
        if not auth_id or not auth_token:
            raise ValueError(
                "Plivo credentials are required: pass auth_id and auth_token, or set "
                "PLIVO_AUTH_ID and PLIVO_AUTH_TOKEN"
            )

        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            language=LanguageCode(language) if language else None,
            speed=speed,
            volume=volume,
            pronunciation_dictionary_id=pronunciation_dictionary_id,
            seed=seed,
            continuation=continuation,
            text_normalization=text_normalization,
            use_normalized_timestamps=use_normalized_timestamps,
            word_timestamps=word_timestamps,
            sample_rate=sample_rate,
        )
        credentials = base64.b64encode(f"{auth_id}:{auth_token}".encode()).decode()
        self._headers = {"Authorization": f"Basic {credentials}"}
        if client_ref:
            self._headers["client-ref"] = client_ref
        self._tts_url = f"{base_url.rstrip('/')}/v1/Account/{auth_id}/TTS"
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._connection: _Connection | None = None
        self._connection_lock = asyncio.Lock()
        self._prewarm_task: asyncio.Task[None] | None = None

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Plivo"

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[TTSLanguages | str | None] = NOT_GIVEN,
        speed: NotGivenOr[float | None] = NOT_GIVEN,
        volume: NotGivenOr[float | None] = NOT_GIVEN,
        pronunciation_dictionary_id: NotGivenOr[str | None] = NOT_GIVEN,
        seed: NotGivenOr[int | None] = NOT_GIVEN,
        continuation: NotGivenOr[bool | None] = NOT_GIVEN,
        text_normalization: NotGivenOr[bool | None] = NOT_GIVEN,
        use_normalized_timestamps: NotGivenOr[bool | None] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options.

        The new options apply to the Realtime streams and the Batch requests created after this
        call. An option that is not given keeps its value, and an option set to None returns to
        its default.

        Args:
            model (NotGivenOr[TTSModels | str]): TTS model to use.
            voice (NotGivenOr[str]): ID of the voice.
            language (NotGivenOr[TTSLanguages | str | None]): Language code of the text.
            speed (NotGivenOr[float | None]): Speaking rate, from 0.7 to 1.2.
            volume (NotGivenOr[float | None]): Volume of the speech, from 0.5 to 2.0.
            pronunciation_dictionary_id (NotGivenOr[str | None]): ID of a pronunciation
                dictionary.
            seed (NotGivenOr[int | None]): Seed of the speech.
            continuation (NotGivenOr[bool | None]): Whether the speech of a stream continues
                from the previous stream's. Realtime API only.
            text_normalization (NotGivenOr[bool | None]): Whether the TTS model normalizes the
                text before speech.
            use_normalized_timestamps (NotGivenOr[bool | None]): Whether the word timestamps
                follow the normalized text in place of the text as sent. Realtime API only.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = LanguageCode(language) if language else None
        if is_given(speed):
            self._opts.speed = speed
        if is_given(volume):
            self._opts.volume = volume
        if is_given(pronunciation_dictionary_id):
            self._opts.pronunciation_dictionary_id = pronunciation_dictionary_id
        if is_given(seed):
            self._opts.seed = seed
        if is_given(continuation):
            self._opts.continuation = continuation
        if is_given(text_normalization):
            self._opts.text_normalization = text_normalization
        if is_given(use_normalized_timestamps):
            self._opts.use_normalized_timestamps = use_normalized_timestamps

    def _ensure_session(self) -> aiohttp.ClientSession:
        return self._session or utils.http_context.http_session()

    async def _current_connection(self, timeout: float) -> tuple[_Connection, float, bool]:
        """Return the connection for a new stream, opening one when there is none.

        Returns:
            The connection, the seconds it took to open, and whether it was open already.
        """
        async with self._connection_lock:
            if self._connection and self._connection.is_current:
                return self._connection, 0.0, True
            if self._connection:
                self._connection.mark_non_current()

            started = time.perf_counter()
            url = f"{self._tts_url.replace('http', 'ws', 1)}/Realtime"
            # _Connection answers pings, which count as messages, and closes within CLOSE_TIMEOUT
            opening = self._ensure_session().ws_connect(
                url, headers=self._headers, autoping=False, autoclose=False
            )
            ws = await _open_within(timeout, opening, _abort)
            api_id = ws._response.headers.get("api-id")
            logger.debug("Plivo connection opened", extra={"plivo_api_id": api_id})
            self._connection = _Connection(ws)
            return self._connection, time.perf_counter() - started, False

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        """Open the connection before the first stream."""

        async def _prewarm() -> None:
            try:
                await self._current_connection(DEFAULT_API_CONNECT_OPTIONS.timeout)
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                # only the type is logged: the error's request info holds the Authorization header
                logger.warning(
                    "failed to prewarm Plivo connection", extra={"exception_type": type(e).__name__}
                )

        if self._prewarm_task is None or self._prewarm_task.done():
            self._prewarm_task = asyncio.create_task(_prewarm())

    async def aclose(self) -> None:
        await asyncio.shield(self._aclose())  # a cancelled aclose() still closes the connection

    async def _aclose(self) -> None:
        if self._prewarm_task:
            await utils.aio.gracefully_cancel(self._prewarm_task)

        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()

        if self._connection:
            await asyncio.wait((self._connection.close(),))


@contextlib.contextmanager
def _api_errors() -> Iterator[None]:
    """Raise a failed connection or request as one of LiveKit's API errors."""
    try:
        yield
    except asyncio.TimeoutError:
        raise APITimeoutError() from None
    except aiohttp.ClientResponseError as e:
        request_id = e.headers.get("api-id") if e.headers else None
        # `from None`: the error's request info holds the Authorization header
        raise APIStatusError(e.message, status_code=e.status, request_id=request_id) from None
    except aiohttp.ClientError as e:
        raise APIConnectionError() from e


class ChunkedStream(tts.ChunkedStream):
    """Synthesize a text with one request to the Batch API."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # no total timeout: a long text streams for longer. The connect and each read are timed
        timeout = self._conn_options.timeout
        if not timeout:  # aiohttp takes 0 as no limit: here, as in a stream, 0 fails at once
            raise APITimeoutError()
        with _api_errors():
            request = self._tts._ensure_session().post(
                self._tts._tts_url,
                headers=self._tts._headers,
                json={**self._opts.request_fields(streaming=False), "text": self._input_text},
                timeout=aiohttp.ClientTimeout(total=None, connect=timeout, sock_read=timeout),
            )
            # aiohttp's timers cancel the task they run in: here a task apart from the caller's
            resp = await _open_within(None, request, aiohttp.ClientResponse.close)
            # not `async with resp` or resp.text(): on Python 3.10 their wait drops a cancel
            try:
                if resp.status != 200:
                    raise APIStatusError(
                        "Plivo error",
                        status_code=resp.status,
                        request_id=resp.headers.get("api-id"),
                        body=(await resp.content.read()).decode(errors="replace"),
                    )
                output_emitter.initialize(
                    request_id=resp.headers["api-id"],
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm",
                )
                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)
                output_emitter.flush()
            finally:
                resp.release()


class SynthesizeStream(tts.SynthesizeStream):
    """Synthesize streamed text with the Realtime API.

    Each stream is one context on a WebSocket connection that the streams of a TTS share.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        context_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=context_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=context_id)

        async def _input_task(connection: _Connection) -> None:
            async for data in self._input_ch:
                if isinstance(data, str):
                    connection.send_text(context_id, data)
                else:
                    connection.send("flush", context_id)
            connection.send("text", context_id, finish=True)

        # the first text to speak opens the context; a stream with none sends nothing
        async for first in self._input_ch:
            if isinstance(first, str) and first.strip():
                break
        else:
            return

        with _api_errors():
            acquire = self._tts._current_connection(self._conn_options.timeout)
            connection, self._acquire_time, self._connection_reused = await acquire
            fields = self._opts.request_fields(streaming=True)
            self._mark_started()
            context = connection.open_context(context_id, output_emitter, **fields, text=first)
            input_t = asyncio.create_task(_input_task(connection))
            try:
                # an interruption cancels neither, so release_context() sees `ended` open
                await asyncio.wait((input_t, context.ended), return_when=asyncio.FIRST_COMPLETED)
                # after the input ends, no audio for the timeout fails the stream
                loop = asyncio.get_running_loop()
                waited = loop.time()
                while not context.ended.done():
                    deadline = max(waited, context.audio_at) + self._conn_options.timeout
                    if deadline <= loop.time():
                        raise asyncio.TimeoutError
                    await asyncio.wait((context.ended,), timeout=deadline - loop.time())
                await context.ended
            except asyncio.TimeoutError:
                # no audio for the timeout: the next stream gets a new connection
                connection.mark_non_current()
                raise
            finally:
                connection.release_context(context_id)
                await utils.aio.gracefully_cancel(input_t)


def _status_error(event: dict[str, Any]) -> APIStatusError:
    """Return an `error` event as LiveKit's error. Its HTTP status says if a retry can help."""
    message = f"Plivo error {event.get('code')}: {event.get('message')}"
    status, api_id = event.get("status_code", -1), event.get("api_id")
    return APIStatusError(message, status_code=status, request_id=api_id, body=event)


@dataclass
class _Context:
    """What a connection keeps for one stream.

    `emitter` takes the stream's audio and word timestamps. `ended` resolves at the context's
    `done` event, and raises at its `error` event or when the connection closes. `audio_at` is
    the event loop's time of the last audio.
    `pending_word` holds the characters of a word until the word ends.
    """

    emitter: tts.AudioEmitter
    ended: asyncio.Future[None] = field(default_factory=asyncio.Future)
    audio_at: float = 0.0
    pending_word: TimedString = TimedString("")

    def __post_init__(self) -> None:
        # read the error here, so that asyncio does not report it when no stream waits for it
        self.ended.add_done_callback(lambda ended: ended.exception())

    def add_timestamps(self, event: dict[str, Any]) -> None:
        """Join an event's timed characters into words. A word keeps the whitespace that ends it."""
        timed_chars = zip(event["chars"], event["start_ms"], event["duration_ms"], strict=False)
        for char, start_ms, duration_ms in timed_chars:
            start = self.pending_word.start_time if self.pending_word else start_ms / 1000
            end = (start_ms + duration_ms) / 1000
            self.pending_word = TimedString(self.pending_word + char, start, end)
            if char.isspace():
                self.push_word()
        # the last word goes out with its audio, without a wait for the whitespace that ends it
        self.push_word()

    def push_word(self) -> None:
        if self.pending_word.strip():  # whitespace alone stays in front of the next word
            self.emitter.push_timed_transcript(self.pending_word)
            self.pending_word = TimedString("")


class _Connection:
    """A WebSocket connection to the Realtime API that the streams of a TTS share.

    The send loop writes the streams' messages in order, and the receive loop routes each
    message to its context.
    """

    def __init__(self, ws: aiohttp.ClientWebSocketResponse[bool]) -> None:
        self._ws = ws
        self._is_current = True
        self._closing = False
        self._contexts: dict[str, _Context] = {}
        # the last message, when it is an error with no context: the reason the connection
        # closes next, such as an account with no credit
        self._error: dict[str, Any] | None = None
        loop = asyncio.get_running_loop()
        self._received = loop.time()
        self._send_ch = utils.aio.Chan[str]()
        self._pongs: set[asyncio.Task[None]] = set()
        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._recv_task.add_done_callback(self._ended)  # also when cancelled before it ran
        self._watchdog = loop.call_later(DEAD_LIMIT, self._watch)

    @property
    def is_current(self) -> bool:
        silent = asyncio.get_running_loop().time() - self._received >= SILENT_LIMIT
        # aiohttp closes a dropped WebSocket inside receive(), before the receive loop ends
        return self._is_current and not self._ws.closed and not silent

    def mark_non_current(self) -> None:
        """Take no new stream. The streams on the connection keep it until they end."""
        self._is_current = False
        if not self._contexts:
            self.close()

    def close(self) -> asyncio.Task[None]:
        """End the connection. The task it returns ends when the connection has ended."""
        self._is_current = False
        if not self._closing:
            self._closing = True
            self._recv_task.cancel()
        return self._recv_task

    def _end(self) -> None:
        self._is_current = False
        message = f"Plivo connection closed: close code {self._ws.close_code or 1006}"
        for context in self._contexts.values():
            if not context.ended.done():
                closed = _status_error(self._error) if self._error else APIConnectionError(message)
                closed.__cause__ = self._ws.exception()
                context.ended.set_exception(closed)
        self._watchdog.cancel()
        self._send_task.cancel()

    def _ended(self, _: asyncio.Task[None]) -> None:
        self._end()
        _abort(self._ws)

    def _watch(self) -> None:
        # a stream is timed only after its input ends: until then, this ends a dead path
        silent = asyncio.get_running_loop().time() - self._received
        if silent < DEAD_LIMIT:
            self._watchdog = asyncio.get_running_loop().call_later(DEAD_LIMIT - silent, self._watch)
        else:
            logger.warning("Plivo connection silent for %.0f s: closing it", silent)
            self.close()

    def open_context(self, context_id: str, emitter: tts.AudioEmitter, **fields: Any) -> _Context:
        """Register a stream's context and queue its first message, which opens the context."""
        context = _Context(emitter)
        self._contexts[context_id] = context
        self.send_text(context_id, fields.pop("text"), **fields)
        return context

    def send(self, kind: str, context_id: str, **fields: Any) -> None:
        """Queue a message for a context. The send loop writes the messages in order."""
        self._send_ch.send_nowait(json.dumps({"type": kind, "context_id": context_id, **fields}))

    def send_text(self, context_id: str, text: str, **fields: Any) -> None:
        """Queue a text in messages of at most MAX_TEXT_LENGTH characters, fields in the first."""
        for start in range(0, len(text), MAX_TEXT_LENGTH):
            self.send("text", context_id, **fields, text=text[start : start + MAX_TEXT_LENGTH])
            fields = {}

    def release_context(self, context_id: str) -> None:
        """Stop routing to a context, and cancel it if it has not ended.

        A connection that is no longer current closes when its last context is released.
        """
        context = self._contexts.pop(context_id)
        if not self._is_current and not self._contexts:
            self.close()
        elif not context.ended.done():
            self.send("cancel", context_id)

    async def _send_loop(self) -> None:
        """Write the queued messages. A failed write closes the WebSocket: the receive loop ends."""
        try:
            async for message in self._send_ch:
                await self._ws.send_str(message)
        except ConnectionError:
            logger.exception("failed to send a message to Plivo")
            self.close()

    async def _recv_loop(self) -> None:
        """Route each message to its context. When the connection closes, the open contexts fail."""
        loop = asyncio.get_running_loop()
        try:
            async for msg in self._ws:
                self._received = loop.time()
                if msg.type in (aiohttp.WSMsgType.PING, aiohttp.WSMsgType.PONG):
                    if msg.type == aiohttp.WSMsgType.PING:
                        # a task of its own: this loop never waits on a write
                        pong = asyncio.create_task(self._ws.pong(msg.data))
                        self._pongs.add(pong)
                        pong.add_done_callback(self._pong_done)
                    continue
                self._error = None
                if msg.type == aiohttp.WSMsgType.BINARY:
                    # [2-byte little-endian length of the context ID][context ID][audio]
                    audio_start = 2 + int.from_bytes(msg.data[:2], "little")
                    if context := self._contexts.get(msg.data[2:audio_start].decode()):
                        context.emitter.push(msg.data[audio_start:])
                        context.audio_at = loop.time()
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    self._on_event(msg.json())
        except Exception:
            logger.exception("failed to read a message from Plivo")
        finally:
            self._closing = True  # a close() from here on must not cut this close
            self._end()
            abort = loop.call_later(CLOSE_TIMEOUT, _abort, self._ws)
            await self._ws.close()
            abort.cancel()

    def _pong_done(self, pong: asyncio.Task[None]) -> None:
        self._pongs.discard(pong)
        if not pong.cancelled():
            pong.exception()  # read it, so that asyncio does not report it

    def _on_event(self, event: dict[str, Any]) -> None:
        """Route an event to its context.

        A warning, or an error with no context, is logged. Such an error is also kept as the reason
        when the connection closes next.
        """
        if event["type"] == "warning" or "context_id" not in event:
            message = f"Plivo {event['type']} {event.get('code')}: {event.get('message')}"
            logger.warning(message, extra={"lk.pii.event": event})
            if event["type"] == "error":
                self._error = event
        elif (context := self._contexts.get(event["context_id"])) is None or context.ended.done():
            return  # its stream or its context has ended
        elif event["type"] == "timestamps":
            context.add_timestamps(event)
        elif event["type"] == "done":
            context.ended.set_result(None)
        elif event["type"] == "error":
            context.ended.set_exception(_status_error(event))
