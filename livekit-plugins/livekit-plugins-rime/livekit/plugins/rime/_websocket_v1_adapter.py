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
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass

import aiohttp

from livekit.agents import APIConnectOptions, APIError, tokenize, tts, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from . import _websocket_v1
from .log import logger

_Pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse]


class _TokenizerFlush:
    """Drain the current local sentence tokenizer without ending Coda input."""


@dataclass(frozen=True)
class CodaV1SynthesisOptions:
    """Coda settings before conversion to the WebSocket v1 wire format."""

    speaker: str
    language: NotGivenOr[str] = NOT_GIVEN
    sampling_rate: NotGivenOr[int] = NOT_GIVEN
    time_scale_factor: NotGivenOr[float] = NOT_GIVEN

    def _to_protocol(self) -> _websocket_v1.SynthesisOptions:
        if not is_given(self.language) or not is_given(self.sampling_rate):
            raise APIError("Rime v1 requires Coda language and sample_rate", retryable=False)

        return _websocket_v1.SynthesisOptions(
            speaker=self.speaker,
            language=self.language,
            sampling_rate=self.sampling_rate,
            time_scale_factor=(
                self.time_scale_factor if is_given(self.time_scale_factor) else None
            ),
        )


class WebSocketV1Adapter:
    """Adapt LiveKit streaming to the Rime WebSocket v1 protocol."""

    def __init__(
        self,
        *,
        websocket_v1_url: str,
        api_key: str,
        ensure_session: Callable[[], aiohttp.ClientSession],
        sentence_tokenizer: tokenize.SentenceTokenizer | None = None,
    ) -> None:
        _websocket_v1.validate_websocket_url(websocket_v1_url)
        self._websocket_v1_url = websocket_v1_url
        self._api_key = api_key
        self._ensure_session = ensure_session
        self._sentence_tokenizer = (
            sentence_tokenizer
            if sentence_tokenizer is not None
            else tokenize.blingfire.SentenceTokenizer(min_sentence_len=1)
        )
        self._retired_pools: set[_Pool] = set()
        self._pool_stream_counts: dict[_Pool, int] = {}
        self._pool_close_tasks: set[asyncio.Task[None]] = set()
        self._pool = self._new_pool()

    def _new_pool(self) -> _Pool:
        websocket_v1_url = self._websocket_v1_url

        async def _connect(timeout: float) -> aiohttp.ClientWebSocketResponse:
            return await _websocket_v1.connect(
                self._ensure_session(),
                websocket_url=websocket_v1_url,
                api_key=self._api_key,
                timeout=timeout,
            )

        return utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=_connect,
            close_cb=_websocket_v1.close,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )

    def stream(
        self,
        *,
        tts_instance: tts.TTS,
        options: CodaV1SynthesisOptions,
        conn_options: APIConnectOptions,
    ) -> _WebSocketV1SynthesizeStream:
        pool = self._pool
        stream = _WebSocketV1SynthesizeStream(
            tts_instance=tts_instance,
            pool=pool,
            options=options._to_protocol(),
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

    def update_endpoint(self, websocket_v1_url: str) -> None:
        _websocket_v1.validate_websocket_url(websocket_v1_url)
        if websocket_v1_url == self._websocket_v1_url:
            return

        old_pool = self._pool
        self._websocket_v1_url = websocket_v1_url
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


class _WebSocketV1SynthesizeStream(tts.SynthesizeStream):
    """Run one LiveKit stream as one Rime WebSocket v1 synthesis context."""

    def __init__(
        self,
        *,
        tts_instance: tts.TTS,
        pool: _Pool,
        options: _websocket_v1.SynthesisOptions,
        conn_options: APIConnectOptions,
        sentence_tokenizer: tokenize.SentenceTokenizer,
    ) -> None:
        super().__init__(tts=tts_instance, conn_options=conn_options)
        self._pool = pool
        self._options = options
        self._sentence_tokenizer = sentence_tokenizer
        self._end_flush_sentinel: object | None = None

    def _enqueue_flush_sentinel(self) -> tts.SynthesizeStream._FlushSentinel:
        sentinel = self._FlushSentinel()
        self._input_ch.send_nowait(sentinel)
        self._input_buffer.append(sentinel)
        return sentinel

    def flush(self) -> None:
        """Drain pending text without finalizing the Rime synthesis context.

        The stream accepts more text after this call. Rime does not send ``done``
        until :meth:`end_input` finalizes the context.
        """
        if self._input_ch.closed:
            return

        # The base method records every flush as a LiveKit segment boundary. This
        # adapter keeps the Coda context, metric text, and segment state active while
        # it drains the current local sentence tokenizer.
        self._enqueue_flush_sentinel()

    def end_input(self) -> None:
        """Finalize Rime input and let the stream wait for the provider ``done`` event."""
        if self._input_ch.closed:
            return

        if self._mtc_text:
            self._mtc_pending_texts.append(self._mtc_text)
            self._mtc_text = ""

        self._end_flush_sentinel = self._enqueue_flush_sentinel()
        self._input_ch.close()
        self._input_ended = True

    async def aclose(self) -> None:
        """Cancel the active Rime context when synthesis has not finished."""
        await super().aclose()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        async def _raw_input_events() -> AsyncIterable[str | _TokenizerFlush]:
            async for event in self._input_ch:
                if isinstance(event, self._FlushSentinel):
                    if event is not self._end_flush_sentinel:
                        yield _TokenizerFlush()
                else:
                    yield event

        input_events = _sentence_tokenized_input_events(
            _raw_input_events(),
            sentence_tokenizer=self._sentence_tokenizer,
            language=self._options.language,
        )

        ws = await self._pool.get(timeout=self._conn_options.timeout)
        self._acquire_time = self._pool.last_acquire_time
        self._connection_reused = self._pool.last_connection_reused
        reusable = False
        try:
            result = await _websocket_v1.run_context(
                ws,
                context_id=utils.shortuuid(),
                options=self._options,
                input_events=input_events,
                output_emitter=output_emitter,
                timeout=self._conn_options.timeout,
                mark_started=self._mark_started,
            )
            reusable = result.reusable
        except _websocket_v1._ContextCancelled as error:
            reusable = error.reusable
            raise
        finally:
            if reusable:
                self._pool.put(ws)
            else:
                self._pool.remove(ws)
                await _websocket_v1.close(ws)


async def _sentence_tokenized_input_events(
    input_events: AsyncIterable[str | _TokenizerFlush],
    *,
    sentence_tokenizer: tokenize.SentenceTokenizer,
    language: str,
) -> AsyncIterable[str]:
    """Convert text fragments to sentence units and drain on local flushes."""
    output = utils.aio.Chan[str]()

    async def _drive_input() -> None:
        sentence_stream = sentence_tokenizer.stream(language=language)
        forward_task: asyncio.Task[None] | None = None

        async def _forward_sentences() -> None:
            async for event in sentence_stream:
                text = event.token
                if text and not text[-1].isspace():
                    text += " "
                if text:
                    output.send_nowait(text)

        def _start_forwarding() -> asyncio.Task[None]:
            return asyncio.create_task(
                _forward_sentences(), name="rime-v1-sentence-tokenizer-output"
            )

        try:
            forward_task = _start_forwarding()
            async for event in input_events:
                if isinstance(event, _TokenizerFlush):
                    sentence_stream.end_input()
                    await forward_task
                    sentence_stream = sentence_tokenizer.stream(language=language)
                    forward_task = _start_forwarding()
                else:
                    sentence_stream.push_text(event)

            sentence_stream.end_input()
            await forward_task
        finally:
            if not sentence_stream.closed:
                await sentence_stream.aclose()
            if forward_task is not None:
                await utils.aio.gracefully_cancel(forward_task)
            output.close()

    input_task = asyncio.create_task(_drive_input(), name="rime-v1-sentence-tokenizer-input")
    try:
        async for event in output:
            yield event
        await input_task
    finally:
        await utils.aio.gracefully_cancel(input_task)
