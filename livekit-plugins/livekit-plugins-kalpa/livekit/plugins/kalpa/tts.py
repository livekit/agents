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
from collections.abc import Coroutine
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .models import TTSModels

BASE_URL = "https://api.kalpalabs.ai"
DEFAULT_VOICE = "18c26f49-9f9c-400c-bec9-7467f28e1db5"  # Arjun
# LiveKit resamples to the pipeline's rate, so the plugin always requests this one
SAMPLE_RATE = 24000
NUM_CHANNELS = 1
# Pooled sockets retire before the API's idle timeout closes them.
MAX_IDLE_SECONDS = 240
# How long an interrupted response may take to acknowledge its cancel before its socket is dropped.
CANCEL_ACK_TIMEOUT = 1.0
CLOSE_TIMEOUT = 1.0

# HTTP status for each error type, as documented at https://docs.kalpalabs.ai/rate-limits-and-errors
_ERROR_STATUS = {
    "invalid_request": 400,
    "context_length_exceeded": 400,
    "authentication_error": 401,
    "allowance_exhausted": 402,
    "account_suspended": 403,
    "not_found": 404,
    "rate_limit_exceeded": 429,
    "internal_error": 500,
    "inference_error": 502,
    "overloaded": 503,
    "service_unavailable": 503,
}


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: str
    api_key: str
    base_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        model: TTSModels | str = "kalpa-tts-1",
        api_key: str | None = None,
        base_url: str = BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Kalpa TTS.

        Args:
            voice (str): Voice id, from `GET /v1/voices`. Defaults to Arjun.
            model (TTSModels | str): TTS model. Defaults to "kalpa-tts-1".
            api_key (str): Kalpa API key. Defaults to the KALPA_API_KEY environment variable.
            base_url (str): API base URL. Defaults to "https://api.kalpalabs.ai".
            http_session (aiohttp.ClientSession): Optional aiohttp session to use for requests.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("KALPA_API_KEY")
        if not api_key:
            raise ValueError("Kalpa API key required. Set KALPA_API_KEY or provide api_key.")

        self._opts = _TTSOptions(
            model=model, voice=voice, api_key=api_key, base_url=base_url.rstrip("/")
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
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
        return "Kalpa"

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        url = f"{self._opts.base_url.replace('http', 'ws', 1)}/v1/tts/{self._opts.voice}/stream"
        ws = await asyncio.wait_for(self._ensure_session().ws_connect(url), timeout)
        try:
            await ws.send_str(
                json.dumps(
                    {
                        "type": "initializeConnection",
                        "api_key": self._opts.api_key,
                        "model": self._opts.model,
                        "sample_rate": SAMPLE_RATE,
                        # audio starts at the first complete sentences, before the reply's flush
                        "generation_config": {},
                    }
                )
            )
            event = await _receive_event(ws, timeout)
            if event["type"] != "sessionCreated":
                raise APIError(f"unexpected Kalpa message before sessionCreated: {event['type']}")
        except BaseException:
            await self._close_ws(ws)
            raise
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        # a half-open peer never answers the closing handshake; cancelling drops the connection
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(ws.close(), CLOSE_TIMEOUT)

    def _recycle(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Cancel an interrupted response in the background and return its socket to the pool once
        the server acknowledges the cancel; without a timely acknowledgement, replace the socket."""

        async def _cancel() -> None:
            try:
                await asyncio.wait_for(_cancel_response(ws), CANCEL_ACK_TIMEOUT)
            except Exception:
                self._drop(ws)
                self._pool.prewarm()
            else:
                self._pool.put(ws)

        self._spawn(_cancel())

    def _drop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Take a socket out of the pool and close it now, so the server stops any work for it."""
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
    ) -> None:
        """
        Update the voice or model. Replies in progress finish with the old ones.

        Args:
            voice (str): Voice id, from `GET /v1/voices`.
            model (TTSModels | str): TTS model.
        """
        if is_given(voice):
            self._opts.voice = voice
        if is_given(model):
            self._opts.model = model
        if is_given(voice) or is_given(model):
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


class ChunkedStream(tts.ChunkedStream):
    """Synthesizes the whole text in one `POST /v1/tts/{voice_id}/stream`, which returns raw PCM
    as it is generated."""

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

        try:
            async with self._tts._ensure_session().post(
                f"{self._opts.base_url}/v1/tts/{self._opts.voice}/stream",
                headers={"Authorization": f"Bearer {self._opts.api_key}"},
                json={
                    "text": self._input_text,
                    "model": self._opts.model,
                    "sample_rate": SAMPLE_RATE,
                },
                timeout=aiohttp.ClientTimeout(
                    sock_connect=self._conn_options.timeout, sock_read=self._conn_options.timeout
                ),
            ) as resp:
                if resp.status != 200:
                    try:
                        error = (await resp.json(content_type=None))["error"]
                    except Exception:
                        error = {}
                    raise _status_error(error, status=resp.status)

                output_emitter._note_provider_request_id(resp.headers.get("X-Request-ID", ""))
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
    """Streams one reply's text over a pooled `WSS /v1/tts/{voice_id}/stream` connection and
    receives its audio while the text is still arriving."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=request_id)

        try:
            await self._run_ws(output_emitter)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(e.message, status_code=e.status) from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

        output_emitter.end_segment()

    async def _run_ws(self, output_emitter: tts.AudioEmitter) -> None:
        pool = self._tts._pool
        ws: aiohttp.ClientWebSocketResponse | None = None
        sent = False  # the server holds text or a response from this reply
        done = False  # the reply's response finished
        # the server must answer within the timeout of the flush, and of every event after it;
        # before the flush, audio may legitimately pause while the reply's text is still arriving
        deadline: float | None = None

        async def _send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal sent, deadline
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    break
                if data.strip():
                    self._mark_started()
                sent = True
                await ws.send_str(json.dumps({"type": "sendText", "text": data}))

            sent = True
            await ws.send_str(json.dumps({"type": "sendText", "flush": True}))
            deadline = time.monotonic() + self._conn_options.timeout

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal done, deadline
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
                if deadline is not None:
                    deadline = time.monotonic() + self._conn_options.timeout

                if event["type"] == "responseCreated":
                    output_emitter._note_provider_request_id(event["response_id"])
                elif event["type"] == "responseAudio":
                    output_emitter.push(base64.b64decode(event["pcm_b64"]))
                elif event["type"] == "responseDone":
                    done = True
                    if event["status"] != "completed":
                        raise APIError(f"Kalpa response ended with status {event['status']}")
                    return

        try:
            ws = await pool.get(timeout=self._conn_options.timeout)
            self._acquire_time = pool.last_acquire_time
            self._connection_reused = pool.last_connection_reused
            tasks = [
                asyncio.create_task(_send_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
        except asyncio.CancelledError:
            # the reply was interrupted: keep the socket for the next reply if it can be cleared
            if ws is not None:
                if sent and not done:
                    self._tts._recycle(ws)
                else:
                    pool.put(ws)
            raise
        except BaseException:
            if ws is not None:
                self._tts._drop(ws)
            raise

        pool.put(ws)


async def _receive_event(ws: aiohttp.ClientWebSocketResponse, timeout: float | None) -> Any:
    msg = await ws.receive(timeout=timeout)
    if msg.type != aiohttp.WSMsgType.TEXT:
        raise APIConnectionError(f"Kalpa connection closed (code {ws.close_code})")
    event = json.loads(msg.data)
    if event["type"] == "error":
        raise _status_error(event["error"])
    return event


async def _cancel_response(ws: aiohttp.ClientWebSocketResponse) -> None:
    await ws.send_str(json.dumps({"type": "cancelResponse"}))
    while (await _receive_event(ws, None))["type"] != "cancelled":
        pass


def _status_error(error: dict[str, Any], *, status: int | None = None) -> APIStatusError:
    error_type = error.get("type")
    return APIStatusError(
        error.get("message") or "Kalpa API error",
        status_code=status or _ERROR_STATUS.get(error_type or "", 500),
        request_id=error.get("request_id"),
        body=error_type,
    )
