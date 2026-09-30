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
import json
import time
import weakref
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ._utils import ATTRIBUTION_HEADERS, SPRAG_BASE_URL, resolve_api_key
from .log import logger
from .models import TTSModels

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

# the server closes a socket that leaves its pings unanswered for 60s
IDLE_SOCKET_MAX_AGE = 45
# the server ends a session after an hour
SOCKET_MAX_LIFETIME = 3000
_CLOSE_TIMEOUT = 2.0

# error codes that describe a transient server fault rather than a bad request
_RETRYABLE_ERROR_CODES = frozenset({"backend_error"})
_RETRYABLE_CLOSE_CODES = frozenset({1000, 1001, 1006, 1011, 1012, 1013})


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: str
    instructions: NotGivenOr[str]
    tokenizer: tokenize.SentenceTokenizer


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = "chorus-clone",
        voice: str = "wade",
        instructions: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = SPRAG_BASE_URL,
        tokenizer: tokenize.SentenceTokenizer | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Sprag TTS.

        Speech streams over a realtime WebSocket that stays open across turns.

        Args:
            model: The Sprag speech model, ``chorus-voices`` or ``chorus-clone``.
            voice: A preset voice id for the chosen model.
            instructions: Style guidance, for speech models that accept it.
            api_key: Your Sprag API key. If not provided, will use the SPRAG_API_KEY
                environment variable.
            base_url: The Sprag API base URL.
            tokenizer: Splits streamed text into the sentences synthesized one at a time.
                Defaults to ``tokenize.blingfire.SentenceTokenizer()``.
            http_session: Optional aiohttp ClientSession to use for the WebSocket.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._api_key = resolve_api_key(api_key)
        self._ws_url = (
            base_url.rstrip("/").replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        )
        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            instructions=instructions,
            tokenizer=tokenizer or tokenize.blingfire.SentenceTokenizer(),
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._connected_at: dict[aiohttp.ClientWebSocketResponse, float] = {}
        self._closing: set[asyncio.Task[None]] = set()
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            # a pooled socket is only read while synthesizing, so it cannot answer pings
            max_session_duration=IDLE_SOCKET_MAX_AGE,
            mark_refreshed_on_get=True,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Sprag"

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        url = f"{self._ws_url}/realtime?{urlencode({'model': self._opts.model})}"
        # the failure is raised outside the except blocks: aiohttp's RequestInfo carries the
        # request headers, so an exception left in the chain would expose the API key
        rejected: tuple[str, int] | None = None
        failure: str | None = None
        try:
            ws = await asyncio.wait_for(
                self._ensure_session().ws_connect(
                    url,
                    headers={"Authorization": f"Bearer {self._api_key}", **ATTRIBUTION_HEADERS},
                    timeout=aiohttp.ClientWSTimeout(ws_close=_CLOSE_TIMEOUT),
                ),
                timeout,
            )
        except asyncio.TimeoutError:
            failure = "timed out"
        except aiohttp.ClientResponseError as e:
            rejected = (e.message, e.status)
        except Exception as e:
            failure = type(e).__name__
        if rejected is not None:
            message, status = rejected
            raise APIStatusError(message=message, status_code=status, request_id=None, body=None)
        if failure is not None:
            raise APIConnectionError(f"failed to connect to Sprag ({failure})")

        try:
            await asyncio.wait_for(self._configure(ws), timeout)
        except BaseException:
            await ws.close()
            raise
        self._connected_at[ws] = time.monotonic()
        return ws

    async def _configure(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        session: dict[str, Any] = {
            "type": "realtime",
            "audio": {"output": {"voice": self._opts.voice}},
        }
        if is_given(self._opts.instructions):
            session["instructions"] = self._opts.instructions
        await ws.send_str(json.dumps({"type": "session.update", "session": session}))
        while True:
            data = await _receive_event(ws, timeout=None)
            if data["type"] == "session.updated":
                return
            if data["type"] == "error":
                raise _event_error(data)

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        self._connected_at.pop(ws, None)
        await ws.close()

    def _close_soon(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Close a socket the pool dropped now, so the server stops serving it."""
        task = asyncio.create_task(self._close_ws(ws))
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    def _expired(self, ws: aiohttp.ClientWebSocketResponse) -> bool:
        connected_at = self._connected_at.get(ws)
        return connected_at is not None and time.monotonic() - connected_at > SOCKET_MAX_LIFETIME

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        instructions: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options. Open connections are replaced on their next use.

        Args:
            model: The Sprag speech model.
            voice: A preset voice id for the chosen model.
            instructions: Style guidance, for speech models that accept it.
        """
        before = (self._opts.model, self._opts.voice, self._opts.instructions)
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
        if is_given(instructions):
            self._opts.instructions = instructions
        if (self._opts.model, self._opts.voice, self._opts.instructions) != before:
            self._pool.invalidate()

    def prewarm(self) -> None:
        self._pool.prewarm()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

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
        await self._pool.aclose()
        await asyncio.gather(*self._closing, return_exceptions=True)


class SynthesizeStream(tts.SynthesizeStream):
    """Streams synthesis over a Sprag realtime speech session, one sentence per response."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._tokenizer = tts._opts.tokenizer

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            stream=True,
            mime_type="audio/pcm",
        )

        segments_ch = utils.aio.Chan[tokenize.SentenceStream]()

        async def _tokenize_input() -> None:
            input_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if input_stream is None:
                        input_stream = self._tokenizer.stream()
                        segments_ch.send_nowait(input_stream)
                    input_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if input_stream:
                        input_stream.end_input()
                    input_stream = None

            if input_stream:
                input_stream.end_input()
            segments_ch.close()

        async def _run_segments() -> None:
            async for input_stream in segments_ch:
                await self._run_segment(input_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except (APIStatusError, APIConnectionError):
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _acquire(self) -> aiohttp.ClientWebSocketResponse:
        pool = self._tts._pool
        while True:
            ws = await pool.get(timeout=self._conn_options.timeout)
            if not self._tts._expired(ws):
                break
            pool.remove(ws)
            self._tts._close_soon(ws)
        self._acquire_time = pool.last_acquire_time
        self._connection_reused = pool.last_connection_reused
        return ws

    async def _run_segment(
        self, input_stream: tokenize.SentenceStream, output_emitter: tts.AudioEmitter
    ) -> None:
        output_emitter.start_segment(segment_id=utils.shortuuid())
        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            async for sentence in input_stream:
                text = sentence.token.strip()
                if not text:
                    continue
                if ws is None:
                    ws = await self._acquire()
                self._mark_started()
                await self._speak(ws, text, output_emitter)
        except BaseException:
            if ws is not None:
                # an interrupted socket may still be mid-response, so it is never reused
                self._tts._pool.remove(ws)
                self._tts._close_soon(ws)
            raise
        if ws is not None:
            self._tts._pool.put(ws)
        output_emitter.end_segment()

    async def _speak(
        self, ws: aiohttp.ClientWebSocketResponse, text: str, output_emitter: tts.AudioEmitter
    ) -> None:
        item = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }
        await ws.send_str(json.dumps({"type": "conversation.item.create", "item": item}))
        await ws.send_str(json.dumps({"type": "response.create"}))
        while True:
            data = await _receive_event(ws, timeout=self._conn_options.timeout)
            event_type = data["type"]
            if event_type == "response.output_audio.delta":
                output_emitter.push(base64.b64decode(data["delta"]))
            elif event_type == "response.done":
                response = data.get("response") or {}
                status = response.get("status")
                if status != "completed":
                    error = (response.get("status_details") or {}).get("error") or {}
                    raise APIStatusError(
                        f"Sprag speech response ended as {status}",
                        status_code=-1,
                        body=str(data),
                        retryable=error.get("code") in _RETRYABLE_ERROR_CODES,
                    )
                return
            elif event_type == "error":
                raise _event_error(data)


async def _receive_event(
    ws: aiohttp.ClientWebSocketResponse, *, timeout: float | None
) -> dict[str, Any]:
    while True:
        msg = await ws.receive(timeout=timeout)
        if msg.type == aiohttp.WSMsgType.TEXT:
            data: dict[str, Any] = json.loads(msg.data)
            return data
        if msg.type in (
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
        ):
            code = ws.close_code
            raise APIStatusError(
                "Sprag connection closed unexpectedly",
                status_code=code or -1,
                body=f"{msg.data=} {msg.extra=}",
                retryable=code is None or code in _RETRYABLE_CLOSE_CODES,
            )
        if msg.type == aiohttp.WSMsgType.ERROR:
            raise APIConnectionError(f"Sprag connection failed ({type(msg.data).__name__})")
        logger.debug("ignoring Sprag message of type %s", msg.type)


def _event_error(data: dict[str, Any]) -> APIStatusError:
    error = data.get("error") or {}
    return APIStatusError(
        error.get("message", "unknown Sprag error"),
        status_code=-1,
        body=str(data),
        retryable=error.get("code") in _RETRYABLE_ERROR_CODES,
    )
