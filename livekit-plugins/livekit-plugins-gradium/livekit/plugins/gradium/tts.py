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
import time
import weakref
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any

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

from .api import ws_headers
from .log import logger

SUPPORTED_SAMPLE_RATE = 48000
# Catalog voice "Everly" (en-us), used when neither `voice` nor `voice_id` is given.
DEFAULT_VOICE_ID = "vjFyp2BcCyTlUmYb"

# Client-side ping interval. The server answers with pongs; a missing pong (aiohttp
# allows half the interval) closes the socket, which is how a dead transport is
# detected while sessions wait for the server, silent itself while it waits for text
# (a slow LLM, a tool call mid-reply). The server closes sessions that produce nothing
# for 120s on its own. A false alarm fails every session on the shared socket, so the
# 5s pong window leaves room for an event loop stalled by inference or GC.
WS_HEARTBEAT_S = 10.0

# Websocket close codes the server uses for requests it would reject again verbatim:
# protocol error and unsupported data (unknown model, setup validation). Everything
# else is worth a retry. 1008 (policy violation) covers the org's concurrency cap,
# which frees up when another session ends, next to credits and API-key failures
# that will not; the frame does not tell them apart, so the transient reading wins.
# 1011 covers worker failures and "no worker available" (but also unknown voice ids,
# which the server does not distinguish for clients either). The server's idle and
# duration limits close with 1008 after an error frame; a 1000/1001 before
# end_of_stream means the server shut the socket mid-session, also worth a retry.
_NON_RETRYABLE_CLOSE_CODES = frozenset({1002, 1003})


def _close_error(
    message: str, *, code: int | None, body: Any, reason: Any = None
) -> APIStatusError:
    code = code if isinstance(code, int) else -1
    if isinstance(reason, str) and reason:
        # the close frame's reason is the server's own explanation of the failure
        message = f"{message}: {reason}"
    return APIStatusError(
        message,
        status_code=code,
        body=body,
        retryable=code not in _NON_RETRYABLE_CLOSE_CODES,
    )


def _api_error(msg_data: dict[str, Any]) -> APIStatusError:
    """Build the error for a `{"type": "error"}` frame.

    The server sends it right before closing the socket, so the message is the only
    place the actual failure reason (invalid voice, no credits, worker failure, ...)
    shows up. Its `code` is the websocket close code that follows.
    """
    message = msg_data.get("message") or "unknown error"
    return _close_error(
        f"Gradium returned error: {message}", code=msg_data.get("code"), body=msg_data
    )


@dataclass
class _TTSOptions:
    voice: str | None
    voice_id: str | None
    pronunciation_id: str | None
    word_tokenizer: tokenize.WordTokenizer
    json_config: dict[str, Any] | None = None


def _setup_msg(opts: _TTSOptions, model_name: str, client_req_id: str) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "type": "setup",
        "model_name": model_name,
        "output_format": "pcm",
        # Keep the socket open after this session's end_of_stream: the other replies
        # of this TTS run as further sessions on it, told apart by client_req_id.
        "close_ws_on_eos": False,
        "client_req_id": client_req_id,
    }
    if opts.voice is not None:
        msg["voice"] = opts.voice
    if opts.voice_id is not None:
        msg["voice_id"] = opts.voice_id
    if opts.pronunciation_id is not None:
        msg["pronunciation_id"] = opts.pronunciation_id
    if opts.json_config is not None:
        msg["json_config"] = json.dumps(opts.json_config)
    return msg


_Frame = dict[str, Any]


class _Connection:
    """The websocket of a TTS instance, shared by all of its sessions.

    A reader task routes every frame to the session it is tagged with, so sessions
    run in parallel on the socket. An interrupted reply is cancelled server-side and
    whatever still arrives for it is dropped. When the socket dies, every open session
    is handed the error and the connection reports itself closed, so the next session
    opens a new one.
    """

    def __init__(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        self._ws = ws
        self._sessions: dict[str, asyncio.Queue[_Frame | APIError]] = {}
        self._error: APIError | None = None
        self._reader = asyncio.create_task(self._read(), name="gradium.tts.reader")

    @property
    def closed(self) -> bool:
        return self._error is not None or self._ws.closed

    @asynccontextmanager
    async def session(
        self, client_req_id: str, setup: _Frame
    ) -> AsyncIterator[asyncio.Queue[_Frame | APIError]]:
        """Run one session on the socket: send its setup, yield the queue its frames
        arrive on, and stop listening on exit; whatever the server still sends for it
        is dropped.

        A session left before its last frame arrived (the agent interrupted it, it
        timed out waiting for the server, or it failed) is cancelled server-side, so a
        retry on the same socket does not run next to an orphan holding a worker and a
        concurrency slot. Only a session whose setup went out is cancelled: a message
        for an id the server has never seen is a protocol error that closes the socket
        under every other session. A cancel landing after the session ended on its own
        changes nothing on the server.
        """
        if self._error is not None:
            raise self._error
        queue: asyncio.Queue[_Frame | APIError] = asyncio.Queue()
        self._sessions[client_req_id] = queue  # before the setup: its `ready` follows at once
        setup_sent = False
        try:
            await self.send(setup, client_req_id)
            setup_sent = True
            yield queue
        except BaseException:
            if setup_sent:
                await self._cancel(client_req_id)
            raise
        finally:
            self._sessions.pop(client_req_id, None)

    async def _cancel(self, client_req_id: str) -> None:
        """Cancel a session: the server stops synthesizing it, releases the worker and
        answers with `cancelled` as the session's last frame; the socket and the other
        sessions on it are unaffected."""
        self._sessions.pop(client_req_id, None)
        if self.closed:
            return
        try:
            await asyncio.wait_for(self.send({"type": "cancel"}, client_req_id), 1.0)
        except Exception:
            pass  # the socket is going down; the reader reports it to the live sessions

    async def send(self, msg: _Frame, client_req_id: str) -> None:
        msg["client_req_id"] = client_req_id
        data = json.dumps(msg)
        try:
            await self._ws.send_str(data)
        except ConnectionError as e:
            # the transport is gone, but the reader may not have noticed yet: mark the
            # connection dead so a retry opens a new socket instead of reusing this one
            error = APIConnectionError("Gradium websocket send failed")
            error.__cause__ = e
            self._fail(error)
            raise

    async def _read(self) -> None:
        error: APIError = APIConnectionError("Gradium websocket closed")
        try:
            while True:
                msg = await self._ws.receive()
                if msg.type == aiohttp.WSMsgType.ERROR:
                    error = APIConnectionError("Gradium websocket error")
                    error.__cause__ = self._ws.exception()
                    break
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    error = _close_error(
                        "Gradium connection closed unexpectedly",
                        code=self._ws.close_code,
                        body=f"{msg.data=} {msg.extra=}",
                        reason=msg.extra,
                    )
                    break
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = json.loads(msg.data)
                except ValueError:
                    data = None
                if not isinstance(data, dict):
                    # one bad frame must not take every session down with it
                    logger.warning(
                        "ignoring a malformed Gradium frame", extra={"lk.pii.data": msg.data}
                    )
                    continue
                client_req_id = data.get("client_req_id")
                if not isinstance(client_req_id, str):
                    client_req_id = None
                queue = self._sessions.get(client_req_id) if client_req_id else None
                if queue is not None:
                    queue.put_nowait(data)
                elif data.get("type") == "error":
                    if client_req_id is None:
                        # a connection-level rejection; the server closes the socket next
                        error = _api_error(data)
                        break
                    # a session nobody listens to any more (an interrupted reply, say)
                    # failing; worth a line if the socket goes down after it
                    logger.warning(
                        "Gradium error for an earlier session on this socket",
                        extra={"client_req_id": client_req_id, "error": data.get("message")},
                    )
        except Exception as e:
            error = APIConnectionError(f"Gradium websocket reader failed: {type(e).__name__}")
            error.__cause__ = e
        finally:
            self._fail(error)  # also when aclose() cancels the reader

    def _fail(self, error: APIError) -> None:
        """Hand the error to every open session, once; after it the socket is unusable."""
        if self._error is not None:
            return
        self._error = error
        for queue in self._sessions.values():
            queue.put_nowait(error)

    async def aclose(self) -> None:
        if self._ws.closed and self._error is None:
            await self._let_reader_record_close()
        self._fail(APIConnectionError("Gradium websocket closed"))
        await utils.aio.gracefully_cancel(self._reader)
        await self._ws.close()

    async def _let_reader_record_close(self) -> None:
        """aiohttp marks the socket closed before it hands the reader the CLOSE frame;
        give the reader a moment to record it, so the sessions get the server's close
        code rather than a generic error."""
        try:
            await asyncio.wait_for(asyncio.shield(self._reader), 1.0)
        except Exception:
            pass


async def _consume_session(
    queue: asyncio.Queue[_Frame | APIError],
    *,
    ready_timeout: float,
    on_audio: Callable[[bytes], None],
) -> None:
    """Consume one session's frames until its end_of_stream.

    `ready_timeout` bounds the wait for the server's `ready` only. After that the
    session waits as long as the socket lives: the server is silent while it waits for
    text (a slow LLM), and a dead socket is reported by the connection's reader.
    """
    ready = False
    while True:
        item = await (queue.get() if ready else asyncio.wait_for(queue.get(), ready_timeout))
        if isinstance(item, APIError):
            raise item
        type_ = item.get("type")
        if type_ == "audio":
            on_audio(base64.b64decode(item["audio"]))
        elif type_ == "end_of_stream":
            return
        elif type_ == "error":
            raise _api_error(item)
        elif type_ == "ready":
            ready = True
        elif type_ == "cancelled":
            # only sent in answer to a client cancel, after which nobody listens here;
            # if the server ever cancels a session on its own, it ended without audio
            raise APIStatusError(
                "Gradium cancelled the session", status_code=1000, body=item, retryable=True
            )
        elif type_ in ("text", "need_more_text", "flushed"):
            # TODO(laurent): handle text timestamps?
            pass
        else:
            logger.warning(f"unknown message type: {type_}")


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_endpoint: str | None = None,
        model_name: str = "default",
        voice: str | None = None,
        voice_id: str | None = None,
        pronunciation_id: str | None = None,
        json_config: dict[str, Any] | None = None,
        http_session: aiohttp.ClientSession | None = None,
        word_tokenizer: tokenize.WordTokenizer | None = None,
    ) -> None:
        """
        Initialize the Gradium TTS.

        Args:
            api_key (str): Gradium API key, or `GRADIUM_API_KEY` env var.
            model_endpoint (str): Gradium model endpoint, or `GRADIUM_MODEL_ENDPOINT` env var.
            model_name (str): Model name.
            voice (str): Speaker voice name. Mutually exclusive with `voice_id`.
            voice_id (str): Speaker voice ID. Mutually exclusive with `voice`; used by
                default (`DEFAULT_VOICE_ID`) when neither is given.
            pronunciation_id (str): Optional pronunciation ID for controlling TTS pronunciation.
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text. Defaults to basic WordTokenizer.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SUPPORTED_SAMPLE_RATE,
            num_channels=1,
        )

        api_key = api_key or os.environ.get("GRADIUM_API_KEY")

        if not api_key:
            raise ValueError(
                "Gradium API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `GRADIUM_API_KEY` environment variable"
            )

        model_endpoint = (
            model_endpoint
            or os.environ.get("GRADIUM_MODEL_ENDPOINT")
            or "wss://api.gradium.ai/api/speech/tts"
        )

        self._api_key = api_key
        self._model_endpoint = model_endpoint
        self._model_name = model_name

        # The server resolves `voice_id` to an embedding and then ignores `voice`, so
        # sending both would silently discard the voice name.
        if voice is not None and voice_id is not None:
            raise ValueError("`voice` and `voice_id` are mutually exclusive")
        if voice is None and voice_id is None:
            voice_id = DEFAULT_VOICE_ID

        if not word_tokenizer:
            word_tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)
        self._opts = _TTSOptions(
            voice=voice,
            voice_id=voice_id,
            pronunciation_id=pronunciation_id,
            word_tokenizer=word_tokenizer,
            json_config=json_config,
        )
        self._session = http_session
        # One socket per TTS instance, shared by all of its replies as multiplexed
        # sessions, so a reply does not pay a handshake; see _Connection.
        self._conn: _Connection | None = None
        self._conn_lock = asyncio.Lock()
        self._streams = weakref.WeakSet[ChunkedStream | SynthesizeStream]()
        self._prewarm_task: asyncio.Task[None] | None = None

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "Gradium"

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        """Open the websocket, bounding the handshake by `timeout`.

        The shared aiohttp session has no connect timeout of its own, so a hung
        handshake would otherwise block for aiohttp's default of five minutes.
        """
        return await asyncio.wait_for(
            self._ensure_session().ws_connect(
                self._model_endpoint,
                headers=ws_headers(self._api_key),
                heartbeat=WS_HEARTBEAT_S,
                timeout=aiohttp.ClientWSTimeout(ws_close=10),
            ),
            timeout,
        )

    async def _connection(self, timeout: float) -> tuple[_Connection, float, bool]:
        """The shared socket, opened on first use or after the server closed it.

        Returns the connection, the time spent opening it, and whether it was reused.
        """
        async with self._conn_lock:
            if self._conn is not None and not self._conn.closed:
                return self._conn, 0.0, True
            if self._conn is not None:
                await self._conn.aclose()
                self._conn = None
            t0 = time.perf_counter()
            self._conn = _Connection(await self._connect_ws(timeout))
            return self._conn, time.perf_counter() - t0, False

    async def _connection_for(self, stream: ChunkedStream | SynthesizeStream) -> _Connection:
        """The shared socket for one reply, recording on the stream how it was acquired."""
        conn, stream._acquire_time, stream._connection_reused = await self._connection(
            stream._conn_options.timeout
        )
        return conn

    def prewarm(self) -> None:
        if self._prewarm_task is not None and not self._prewarm_task.done():
            return
        self._prewarm_task = asyncio.create_task(self._prewarm())

    async def _prewarm(self) -> None:
        try:
            await self._connection(DEFAULT_API_CONNECT_OPTIONS.timeout)
        except Exception as e:
            logger.warning(
                "failed to prewarm the Gradium TTS socket",
                extra={"exception_type": type(e).__name__},
            )

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        if self._prewarm_task is not None:
            await utils.aio.gracefully_cancel(self._prewarm_task)
        # under the lock: a connect still in flight would otherwise store its socket
        # after this, and nothing would ever close it
        async with self._conn_lock:
            if self._conn is not None:
                await self._conn.aclose()
                self._conn = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        json_config: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> None:
        if is_given(voice) and is_given(voice_id):
            raise ValueError("`voice` and `voice_id` are mutually exclusive")
        if is_given(voice):
            self._opts.voice = voice
            self._opts.voice_id = None
        if is_given(voice_id):
            self._opts.voice_id = voice_id
            self._opts.voice = None
        if is_given(json_config):
            self._opts.json_config = json_config

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        stream = ChunkedStream(tts=self, input_text=text, conn_options=conn_options)
        self._streams.add(stream)
        return stream


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)

        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # TODO(laurent): once we support the POST requests, we should use it here rather than the websocket API.
        client_req_id = utils.shortuuid()
        try:
            conn = await self._tts._connection_for(self)
            output_emitter.initialize(
                request_id=client_req_id,
                sample_rate=SUPPORTED_SAMPLE_RATE,
                num_channels=1,
                mime_type="audio/pcm",
            )

            setup = _setup_msg(self._opts, self._tts._model_name, client_req_id)
            async with conn.session(client_req_id, setup) as queue:
                await conn.send({"type": "text", "text": self._input_text}, client_req_id)
                await conn.send({"type": "end_of_stream"}, client_req_id)
                await _consume_session(
                    queue, ready_timeout=self._conn_options.timeout, on_audio=output_emitter.push
                )
            output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Stream-based text-to-speech synthesis using Gradium WebSocket API."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        segments_ch = utils.aio.Chan[tokenize.WordStream]()
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SUPPORTED_SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _tokenize_input() -> None:
            # Converts incoming text into WordStreams and sends them into _segments_ch
            word_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if word_stream is None:
                        word_stream = self._opts.word_tokenizer.stream()
                        segments_ch.send_nowait(word_stream)
                    word_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if word_stream:
                        word_stream.end_input()
                    word_stream = None

            segments_ch.close()

        async def _run_segments() -> None:
            async for word_stream in segments_ch:
                await self._run_session(word_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_session(
        self, word_stream: tokenize.WordStream, output_emitter: tts.AudioEmitter
    ) -> None:
        client_req_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=client_req_id)
        conn = await self._tts._connection_for(self)
        setup = _setup_msg(self._opts, self._tts._model_name, client_req_id)
        async with conn.session(client_req_id, setup) as queue:

            async def send_task() -> None:
                async for word in word_stream:
                    self._mark_started()
                    await conn.send({"type": "text", "text": f"{word.token} "}, client_req_id)

                await conn.send({"type": "end_of_stream"}, client_req_id)

            async def recv_task() -> None:
                await _consume_session(
                    queue, ready_timeout=self._conn_options.timeout, on_audio=output_emitter.push
                )

            tasks = [asyncio.create_task(send_task()), asyncio.create_task(recv_task())]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
        output_emitter.end_segment()
