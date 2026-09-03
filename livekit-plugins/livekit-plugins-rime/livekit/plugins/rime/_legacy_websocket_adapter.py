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
from collections.abc import Callable
from dataclasses import dataclass

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
from livekit.agents.voice.io import TimedString

from .log import logger

NUM_CHANNELS = 1

_Pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse]


@dataclass(frozen=True)
class LegacySynthesisOptions:
    model: str
    websocket_url: str
    sample_rate: int


class LegacyWebSocketAdapter:
    """Own endpoint-bound pools and immutable options for legacy WS3 streams."""

    def __init__(
        self,
        *,
        websocket_url: str,
        api_key: str,
        ensure_session: Callable[[], aiohttp.ClientSession],
        sentence_tokenizer: tokenize.SentenceTokenizer,
    ) -> None:
        self._websocket_url = websocket_url
        self._api_key = api_key
        self._ensure_session = ensure_session
        self._sentence_tokenizer = sentence_tokenizer
        self._retired_pools: set[_Pool] = set()
        self._pool_stream_counts: dict[_Pool, int] = {}
        self._pool_close_tasks: set[asyncio.Task[None]] = set()
        self._pool = self._new_pool()

    def _new_pool(self) -> _Pool:
        websocket_url = self._websocket_url

        async def _connect(timeout: float) -> aiohttp.ClientWebSocketResponse:
            return await self._connect(websocket_url=websocket_url, timeout=timeout)

        return utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=_connect,
            close_cb=self._close,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )

    async def _connect(
        self, *, websocket_url: str, timeout: float
    ) -> aiohttp.ClientWebSocketResponse:
        return await asyncio.wait_for(
            self._ensure_session().ws_connect(
                websocket_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
            ),
            timeout,
        )

    async def _close(self, websocket: aiohttp.ClientWebSocketResponse) -> None:
        try:
            await websocket.send_str(json.dumps({"operation": "eos"}))
            try:
                await asyncio.wait_for(websocket.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
        except Exception as error:
            logger.warning(
                "error during Rime WebSocket close sequence",
                extra={"exception_type": type(error).__name__},
            )
        finally:
            await websocket.close()

    def stream(
        self,
        *,
        tts_instance: tts.TTS,
        options: LegacySynthesisOptions,
        conn_options: APIConnectOptions,
    ) -> _LegacyWebSocketSynthesizeStream:
        if options.websocket_url != self._websocket_url:
            raise RuntimeError("Rime WS3 stream options do not match the active endpoint")

        pool = self._pool
        stream = _LegacyWebSocketSynthesizeStream(
            tts_instance=tts_instance,
            pool=pool,
            options=options,
            conn_options=conn_options,
            sentence_tokenizer=self._sentence_tokenizer,
        )
        self._retain_pool(pool)

        def _release_pool(_: asyncio.Task[None]) -> None:
            self._release_pool(pool)

        stream._task.add_done_callback(_release_pool)
        return stream

    def prewarm(self) -> None:
        self._pool.prewarm()

    def update_endpoint(self, websocket_url: str) -> None:
        if websocket_url == self._websocket_url:
            return

        old_pool = self._pool
        self._websocket_url = websocket_url
        self._pool = self._new_pool()
        self._retire_pool(old_pool)

    async def aclose(self) -> None:
        await self._pool.aclose()
        for pool in list(self._retired_pools):
            await pool.aclose()
        self._retired_pools.clear()
        if self._pool_close_tasks:
            await asyncio.gather(*list(self._pool_close_tasks), return_exceptions=True)

    def _retain_pool(self, pool: _Pool) -> None:
        self._pool_stream_counts[pool] = self._pool_stream_counts.get(pool, 0) + 1

    def _release_pool(self, pool: _Pool) -> None:
        stream_count = self._pool_stream_counts.get(pool, 0)
        if stream_count > 1:
            self._pool_stream_counts[pool] = stream_count - 1
            return

        self._pool_stream_counts.pop(pool, None)
        self._schedule_retired_pool_close(pool)

    def _retire_pool(self, pool: _Pool) -> None:
        self._retired_pools.add(pool)
        if self._pool_stream_counts.get(pool, 0) == 0:
            self._schedule_retired_pool_close(pool)

    def _schedule_retired_pool_close(self, pool: _Pool) -> None:
        if pool not in self._retired_pools:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        self._retired_pools.remove(pool)
        task = loop.create_task(pool.aclose())
        self._pool_close_tasks.add(task)
        task.add_done_callback(self._on_retired_pool_closed)

    def _on_retired_pool_closed(self, task: asyncio.Task[None]) -> None:
        self._pool_close_tasks.discard(task)
        if task.cancelled():
            return
        if error := task.exception():
            logger.warning(
                "failed to close a retired Rime WebSocket pool",
                extra={"exception_type": type(error).__name__},
            )


class _LegacyWebSocketSynthesizeStream(tts.SynthesizeStream):
    """Run one LiveKit stream as one legacy Rime WS3 synthesis request."""

    def __init__(
        self,
        *,
        tts_instance: tts.TTS,
        pool: _Pool,
        options: LegacySynthesisOptions,
        conn_options: APIConnectOptions,
        sentence_tokenizer: tokenize.SentenceTokenizer,
    ) -> None:
        super().__init__(tts=tts_instance, conn_options=conn_options)
        self._metrics_model = options.model
        self._pool = pool
        self._options = options
        self._sentence_tokenizer = sentence_tokenizer

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        context_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._options.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=context_id)

        sent_stream = self._sentence_tokenizer.stream()
        input_sent_event = asyncio.Event()
        empty_input = False

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_stream.flush()
                    continue
                sent_stream.push_text(data)
            sent_stream.end_input()

        async def _send_task(websocket: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal empty_input
            sent_count = 0
            async for event in sent_stream:
                packet = {"text": event.token + " ", "contextId": context_id}
                self._mark_started()
                await websocket.send_str(json.dumps(packet))
                input_sent_event.set()
                sent_count += 1
            if sent_count == 0:
                empty_input = True
                input_sent_event.set()
                output_emitter.end_input()
                return
            await websocket.send_str(json.dumps({"operation": "flush", "contextId": context_id}))

        async def _recv_task(websocket: aiohttp.ClientWebSocketResponse) -> None:
            await input_sent_event.wait()
            if empty_input:
                return
            while True:
                message = await websocket.receive(timeout=self._conn_options.timeout)
                if message.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Rime ws closed unexpectedly",
                        request_id=request_id,
                    )
                if message.type == aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError("Rime WebSocket transport error")
                if message.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Rime ws message type %s", message.type)
                    continue
                data = json.loads(message.data)
                message_type = data.get("type")
                if message_type == "chunk":
                    output_emitter.push(base64.b64decode(data["data"]))
                elif message_type == "timestamps":
                    word_timestamps = data.get("word_timestamps") or {}
                    words = word_timestamps.get("words") or []
                    starts = word_timestamps.get("start") or []
                    ends = word_timestamps.get("end") or []
                    for word, start, end in zip(words, starts, ends, strict=False):
                        output_emitter.push_timed_transcript(
                            TimedString(text=word + " ", start_time=start, end_time=end)
                        )
                elif message_type == "done":
                    output_emitter.end_input()
                    break
                elif message_type == "error":
                    raise APIError("Rime WebSocket request failed")

        try:
            async with self._pool.connection(timeout=self._conn_options.timeout) as websocket:
                tasks = [
                    asyncio.create_task(_input_task()),
                    asyncio.create_task(_send_task(websocket)),
                    asyncio.create_task(_recv_task(websocket)),
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    input_sent_event.set()
                    await sent_stream.aclose()
                    await utils.aio.gracefully_cancel(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as error:
            raise APIStatusError(
                message="Rime WebSocket request failed",
                status_code=error.status,
                request_id=None,
                body=None,
            ) from None
        except APIError:
            raise
        except Exception:
            raise APIConnectionError("Rime WebSocket request failed") from None
