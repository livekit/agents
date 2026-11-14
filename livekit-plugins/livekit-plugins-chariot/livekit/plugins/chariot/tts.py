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
import time
import weakref
from collections.abc import Coroutine
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

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

from . import _utils
from .log import logger
from .models import TTSModels

DEFAULT_VOICE = "bac7d666-094d-4698-91fa-741d60fce662"  # Matt, a Chariot stock voice
DEFAULT_MODEL: TTSModels = "chariot-multilingual-v1"
# The API always streams 16-bit mono PCM at this rate
SAMPLE_RATE = 44100
NUM_CHANNELS = 1
# The server closes a connection after this many quiet seconds (its maximum)
IDLE_TIMEOUT = 300
# Pooled connections retire before the server's idle timeout closes them
MAX_IDLE_SECONDS = 240
# The server ends a reply that has heard nothing for 20 seconds, which a slow LLM can exceed
# between sentences; an empty input.text resets that timer
KEEPALIVE_INTERVAL = 10.0
# The longest input.text the smallest model accepts; longer sentences are sent in pieces
MAX_FRAGMENT_CHARS = 500
CLOSE_TIMEOUT = 1.0

# errors the server sends without a context id just before it ends the whole connection
_SESSION_ENDING_ERRORS = ("insufficient_credits", "idle_timeout")

if TYPE_CHECKING:
    # aiohttp narrows this type on Python 3.11+ only; `bool` holds on every supported version
    _WebSocket = aiohttp.ClientWebSocketResponse[bool]
else:
    # aiohttp before 3.14 does not make the class subscriptable at runtime
    _WebSocket = aiohttp.ClientWebSocketResponse


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: str
    speed: NotGivenOr[float]
    optimize_streaming_latency: NotGivenOr[int]
    enable_logging: NotGivenOr[bool]
    api_key: str
    base_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        model: TTSModels | str = DEFAULT_MODEL,
        speed: NotGivenOr[float] = NOT_GIVEN,
        optimize_streaming_latency: NotGivenOr[int] = NOT_GIVEN,
        enable_logging: NotGivenOr[bool] = NOT_GIVEN,
        api_key: str | None = None,
        base_url: str = _utils.BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Chariot TTS.

        Args:
            voice (str): Voice id, from `GET /v1/voices`. Defaults to Matt. Any voice speaks every
                supported language; the language of the audio follows the text.
            model (TTSModels | str): TTS model. Defaults to "chariot-multilingual-v1".
            speed (float): Speaking rate. Rates are prepared per voice and per model, so this is
                one of a fixed set, listed under `models[].speeds` in `GET /v1/voices`; another
                rate is refused. Defaults to the voice as recorded.
            optimize_streaming_latency (int): 0 to 4. Higher values trade quality passes for
                latency. Defaults to the server's setting.
            enable_logging (bool): False keeps the generated audio out of storage. Defaults to
                the workspace's data retention setting. Applies to `stream()`.
            api_key (str): Chariot API key. Defaults to the CHARIOT_API_KEY environment variable.
            base_url (str): API base URL. Defaults to "https://api.chariot.in".
            http_session (aiohttp.ClientSession): Optional aiohttp session to use for requests.
            sentence_tokenizer (tokenize.SentenceTokenizer): Splits a reply into the sentences
                that are generated one at a time. Defaults to blingfire.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            speed=speed,
            optimize_streaming_latency=optimize_streaming_latency,
            enable_logging=enable_logging,
            api_key=_utils.resolve_api_key(api_key),
            base_url=base_url.rstrip("/"),
        )
        self._session = http_session
        self._sentence_tokenizer = (
            sentence_tokenizer
            if is_given(sentence_tokenizer)
            else tokenize.blingfire.SentenceTokenizer()
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._tasks: set[asyncio.Task[Any]] = set()
        # a reader for each connection that no reply holds; see _park()
        self._idle_readers: dict[_WebSocket, asyncio.Task[None]] = {}
        self._pool = utils.ConnectionPool[_WebSocket](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=MAX_IDLE_SECONDS,
            mark_refreshed_on_get=True,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Chariot"

    async def _connect_ws(self, timeout: float) -> _WebSocket:
        params: dict[str, Any] = {"model_type": self._opts.model, "idle_timeout": IDLE_TIMEOUT}
        if is_given(self._opts.optimize_streaming_latency):
            params["optimize_streaming_latency"] = self._opts.optimize_streaming_latency
        if is_given(self._opts.enable_logging):
            params["enable_logging"] = str(self._opts.enable_logging).lower()
        url = f"{_utils.ws_base_url(self._opts.base_url)}/v1/tts/ws-multi?{urlencode(params)}"
        ws = await asyncio.wait_for(
            self._ensure_session().ws_connect(url, headers=_utils.headers(self._opts.api_key)),
            timeout,
        )
        try:
            # a refused connection (an invalid key or model, no credits) answers with an error
            event = await _receive_event(ws, timeout)
            if event.get("type") == "error":
                raise _utils.error_event(event)
            if event.get("type") != "session.ready":
                raise APIError(f"unexpected Chariot message before session.ready: {event['type']}")
        except BaseException:
            await self._close_ws(ws)
            raise
        self._park(ws)  # until a reply claims it, e.g. after prewarm()
        return ws

    async def _close_ws(self, ws: _WebSocket) -> None:
        reader = self._idle_readers.pop(ws, None)
        if reader is not None and reader is not asyncio.current_task():
            await utils.aio.cancel_and_wait(reader)
        # closing ends every reply still open on the connection, which stops its generation;
        # a half-open peer never answers the closing handshake, so it is not waited on
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(ws.close(), CLOSE_TIMEOUT)

    def _park(self, ws: _WebSocket) -> None:
        """Read a connection while no reply holds it.

        The server pings every connection and closes one that does not answer, and aiohttp
        answers pings only while something reads. Reading also drains audio a cancelled reply
        had in flight, and notices a connection the server closed, which then leaves the pool.
        """

        async def _read() -> None:
            with contextlib.suppress(Exception):
                while True:
                    msg = await ws.receive()
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        break
            self._idle_readers.pop(ws, None)
            self._pool.remove(ws)

        if ws not in self._idle_readers:
            self._idle_readers[ws] = asyncio.create_task(_read())

    async def _acquire(self, timeout: float) -> _WebSocket:
        """Take a connection from the pool for one reply, skipping any the server closed."""
        while True:
            ws = await self._pool.get(timeout=timeout)
            reader = self._idle_readers.pop(ws, None)
            if reader is not None:
                await utils.aio.cancel_and_wait(reader)
            if not ws.closed:
                return ws
            self._pool.remove(ws)

    def _release(self, ws: _WebSocket) -> None:
        """Return a connection to the pool and read it until a reply claims it again."""
        self._pool.put(ws)
        self._park(ws)

    def _recycle(self, ws: _WebSocket, context_id: str) -> None:
        """Cancel an unfinished reply and return its connection to the pool.

        The server stops generating at once. Audio for the reply that is already on the way
        carries its context id and is skipped by the next reply on this connection.
        """

        async def _cancel() -> None:
            try:
                await asyncio.wait_for(
                    ws.send_str(json.dumps({"type": "context.cancel", "context_id": context_id})),
                    CLOSE_TIMEOUT,
                )
            except Exception:
                self._drop(ws)
            else:
                self._release(ws)

        self._spawn(_cancel())

    def _drop(self, ws: _WebSocket) -> None:
        """Take a connection out of the pool and close it."""
        self._pool.remove(ws)
        self._spawn(self._close_ws(ws))

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        optimize_streaming_latency: NotGivenOr[int] = NOT_GIVEN,
        enable_logging: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options. Replies already speaking finish with the old ones.

        Voice and speed are chosen per reply, so changing them keeps the open connections.
        The model, latency level and logging are fixed for a connection, so changing any of
        them replaces the idle ones.

        Args:
            voice (str): Voice id, from `GET /v1/voices`.
            model (TTSModels | str): TTS model.
            speed (float): Speaking rate, one the voice offers under the model.
            optimize_streaming_latency (int): 0 to 4.
            enable_logging (bool): False keeps the generated audio out of storage.
        """
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speed):
            self._opts.speed = speed
        if is_given(model):
            self._opts.model = model
        if is_given(optimize_streaming_latency):
            self._opts.optimize_streaming_latency = optimize_streaming_latency
        if is_given(enable_logging):
            self._opts.enable_logging = enable_logging
        if is_given(model) or is_given(optimize_streaming_latency) or is_given(enable_logging):
            self._pool.invalidate()

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
        self._pool.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await utils.aio.cancel_and_wait(*self._tasks)
        await self._pool.aclose()
        await utils.aio.cancel_and_wait(*self._idle_readers.values())
        self._idle_readers.clear()


class ChunkedStream(tts.ChunkedStream):
    """Synthesizes the whole text with one `POST /v1/tts/stream`, which returns raw PCM as it
    is generated."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        if not self._input_text.strip():
            return  # the API rejects blank text, and there is nothing to say

        payload: dict[str, Any] = {
            "voice_id": self._opts.voice,
            "text": self._input_text,
            "model_type": self._opts.model,
        }
        if is_given(self._opts.speed):
            payload["speed"] = self._opts.speed
        if is_given(self._opts.optimize_streaming_latency):
            payload["optimize_streaming_latency"] = self._opts.optimize_streaming_latency

        try:
            async with self._tts._ensure_session().post(
                f"{self._opts.base_url}/v1/tts/stream",
                headers=_utils.headers(self._opts.api_key),
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    sock_connect=self._conn_options.timeout, sock_read=self._conn_options.timeout
                ),
            ) as resp:
                request_id = resp.headers.get("X-inference-id")
                if resp.status != 200:
                    raise await _utils.http_error(resp, request_id)
                if request_id:
                    output_emitter._note_provider_request_id(request_id)
                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)
                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streams one reply over a pooled `WSS /v1/tts/ws-multi` connection.

    The reply is one context on the connection. Each sentence is generated as soon as it is
    complete, so audio starts while the LLM is still writing. An interrupted reply cancels its
    context and leaves the connection open for the next one.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        context_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=context_id,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=context_id)

        try:
            await self._run_ws(output_emitter, context_id)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(e.message, status_code=e.status) from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

        output_emitter.end_segment()

    async def _run_ws(self, output_emitter: tts.AudioEmitter, context_id: str) -> None:
        pool = self._tts._pool
        ws: _WebSocket | None = None
        opened = False  # the server holds a context for this reply
        done = False  # the server ended the reply's context
        refused = False  # the server refused the reply but kept the connection
        session_ended = False  # the server is ending the whole connection
        # once the reply's text is complete, the server must answer within the timeout of
        # context.close and of every message after it; until then, audio may legitimately
        # pause while the LLM is still writing
        deadline: float | None = None
        send_lock = asyncio.Lock()
        sent_stream = self._tts._sentence_tokenizer.stream()

        async def _send(ws: _WebSocket, message: dict[str, Any]) -> None:
            async with send_lock:
                await ws.send_str(json.dumps(message))

        async def _send_keepalive(ws: _WebSocket) -> None:
            async with send_lock:
                # checked under the lock: once context.close is sent, the server refuses
                # any further input for the context
                if deadline is None:
                    message = {"type": "input.text", "context_id": context_id, "text": ""}
                    await ws.send_str(json.dumps(message))

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_stream.flush()
                    continue
                sent_stream.push_text(data)
            sent_stream.end_input()

        async def _send_task(ws: _WebSocket) -> None:
            nonlocal opened, deadline
            # the first message opens the reply's context with its voice and speed
            open_message: dict[str, Any] = {
                "type": "input.text",
                "context_id": context_id,
                "text": "",
                "voice_id": self._opts.voice,
            }
            if is_given(self._opts.speed):
                open_message["speed"] = self._opts.speed
            await _send(ws, open_message)
            opened = True

            async for ev in sent_stream:
                if not ev.token.strip():
                    continue
                self._mark_started()
                for piece in _split_text(ev.token, MAX_FRAGMENT_CHARS):
                    await _send(ws, {"type": "input.text", "context_id": context_id, "text": piece})
                await _send(ws, {"type": "input.flush", "context_id": context_id})

            # generate anything still buffered, then end the context
            await _send(ws, {"type": "context.close", "context_id": context_id})
            deadline = time.monotonic() + self._conn_options.timeout

        async def _keepalive_task(ws: _WebSocket) -> None:
            while deadline is None:
                await asyncio.sleep(KEEPALIVE_INTERVAL)
                if opened:
                    await _send_keepalive(ws)

        async def _recv_task(ws: _WebSocket) -> None:
            nonlocal done, refused, session_ended, deadline
            session_error: APIError | None = None
            while True:
                timeout = self._conn_options.timeout
                if deadline is not None:
                    timeout = max(deadline - time.monotonic(), 0.001)
                try:
                    event = await _receive_event(ws, timeout)
                except asyncio.TimeoutError:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise
                    continue

                event_context = event.get("context_id")
                if event_context is not None and event_context != context_id:
                    continue  # an earlier reply's messages, still arriving on a reused connection
                if deadline is not None:
                    deadline = time.monotonic() + self._conn_options.timeout

                event_type = event.get("type")
                if event_type == "audio.chunk":
                    output_emitter.push(base64.b64decode(event["audio"]))
                elif event_type == "audio.done":
                    if event.get("error"):
                        logger.warning(
                            "Chariot failed to generate a segment", extra={"context_id": context_id}
                        )
                elif event_type == "context.done":
                    done = True
                    reason = event.get("reason")
                    if reason == "completed":
                        return
                    if reason == "error":
                        refused = True
                        raise _utils.error_event(event)
                    if reason == "session_closed":
                        session_ended = True
                        raise session_error or APIConnectionError("Chariot closed the connection")
                    raise APIError(f"Chariot ended the reply early: {reason}")
                elif event_type == "error":
                    error = _utils.error_event(event)
                    code = event.get("code")
                    if event_context is None and code in _SESSION_ENDING_ERRORS:
                        # the server is about to end the connection and every reply on it
                        session_ended = True
                        session_error = error
                        raise error
                    if event_context is None and isinstance(code, str):
                        # a refused message that is not this reply's; the connection stays open
                        logger.warning("Chariot refused a message: %s", error.message)
                        continue
                    refused = event_context is not None
                    raise error

        try:
            ws = await self._tts._acquire(self._conn_options.timeout)
            self._acquire_time = pool.last_acquire_time
            self._connection_reused = pool.last_connection_reused
            tasks = [
                asyncio.create_task(_input_task()),
                asyncio.create_task(_send_task(ws)),
                asyncio.create_task(_recv_task(ws)),
                asyncio.create_task(_keepalive_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks[:3])
            finally:
                await sent_stream.aclose()
                await utils.aio.gracefully_cancel(*tasks)
        except BaseException as e:
            if ws is not None:
                # an interrupted or refused reply leaves the connection usable; anything else
                # (a timeout, a closed or broken connection) replaces it
                reusable = isinstance(e, asyncio.CancelledError) or done or refused
                if reusable and not session_ended and not ws.closed:
                    if opened and not done:
                        self._tts._recycle(ws, context_id)
                    else:
                        self._tts._release(ws)
                else:
                    self._tts._drop(ws)
            raise

        self._tts._release(ws)


async def _receive_event(ws: _WebSocket, timeout: float | None) -> Any:
    msg = await ws.receive(timeout=timeout)
    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
        raise _utils.close_error(ws.close_code, str(msg.extra or ""))
    if msg.type != aiohttp.WSMsgType.TEXT:
        raise APIConnectionError(f"unexpected Chariot message type {msg.type}")
    return json.loads(msg.data)


def _split_text(text: str, limit: int) -> list[str]:
    """Split text into pieces of at most `limit` characters, at the last space when possible.

    Concatenating the pieces reproduces the input exactly.
    """
    pieces = []
    while len(text) > limit:
        cut = text.rfind(" ", 1, limit + 1)
        if cut <= 0:
            cut = limit
        pieces.append(text[:cut])
        text = text[cut:]
    if text:
        pieces.append(text)
    return pieces
