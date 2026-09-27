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
from ._pool_manager import PoolManager
from ._stream_tts import StreamTTS

_Pool = utils.ConnectionPool[_websocket_v1.Connection]


class _DrainTokenizer:
    """Release buffered text from the current local sentence tokenizer."""


@dataclass(frozen=True)
class V1SynthesisOptions:
    """Model settings before conversion to the WebSocket v1 wire format."""

    model: str
    speaker: str
    language: NotGivenOr[str] = NOT_GIVEN
    audio_format: _websocket_v1.RimeAudioFormat = _websocket_v1.DEFAULT_AUDIO_FORMAT
    sampling_rate: NotGivenOr[int] = NOT_GIVEN
    time_scale_factor: NotGivenOr[float] = NOT_GIVEN
    pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN
    phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN

    def _to_protocol(self) -> _websocket_v1.SynthesisOptions:
        if not is_given(self.language):
            raise APIError("Rime v1 requires a language", retryable=False)

        return _websocket_v1.SynthesisOptions(
            model=self.model,
            speaker=self.speaker,
            language=self.language,
            audio_format=self.audio_format,
            sampling_rate=self.sampling_rate if is_given(self.sampling_rate) else None,
            time_scale_factor=(
                self.time_scale_factor if is_given(self.time_scale_factor) else None
            ),
            pause_between_brackets=(
                self.pause_between_brackets if is_given(self.pause_between_brackets) else None
            ),
            phonemize_between_brackets=(
                self.phonemize_between_brackets
                if is_given(self.phonemize_between_brackets)
                else None
            ),
        )


class WebSocketV1Adapter:
    """Adapt LiveKit streaming to the Rime WebSocket v1 protocol."""

    def __init__(
        self,
        *,
        websocket_v1_url: str,
        websocket_protocol: _websocket_v1.WebSocketProtocol,
        api_key: str,
        ensure_session: Callable[[], aiohttp.ClientSession],
        sentence_tokenizer: tokenize.SentenceTokenizer | None = None,
        allow_custom_endpoint: bool = False,
    ) -> None:
        _websocket_v1.validate_websocket_url(
            websocket_v1_url, allow_custom_endpoint=allow_custom_endpoint
        )
        _websocket_v1._codec_for_protocol(websocket_protocol)
        self._websocket_v1_url = websocket_v1_url
        self._websocket_protocol = websocket_protocol
        self._api_key = api_key
        self._ensure_session = ensure_session
        self._allow_custom_endpoint = allow_custom_endpoint
        self._sentence_tokenizer = (
            sentence_tokenizer
            if sentence_tokenizer is not None
            else tokenize.blingfire.SentenceTokenizer(min_sentence_len=1)
        )
        self._pools = PoolManager(self._new_pool())

    def _new_pool(self) -> _Pool:
        websocket_v1_url = self._websocket_v1_url

        async def _connect(timeout: float) -> _websocket_v1.Connection:
            return await _websocket_v1.connect(
                self._ensure_session(),
                websocket_url=websocket_v1_url,
                api_key=self._api_key,
                protocol=self._websocket_protocol,
                timeout=timeout,
                allow_custom_endpoint=self._allow_custom_endpoint,
            )

        return utils.ConnectionPool[_websocket_v1.Connection](
            connect_cb=_connect,
            close_cb=_websocket_v1.close,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )

    def stream(
        self,
        *,
        tts_instance: tts.TTS,
        options: V1SynthesisOptions,
        conn_options: APIConnectOptions,
    ) -> _WebSocketV1SynthesizeStream:
        pool = self._pools.current
        stream = _WebSocketV1SynthesizeStream(
            tts_instance=tts_instance,
            pool=pool,
            options=options._to_protocol(),
            conn_options=conn_options,
            sentence_tokenizer=self._sentence_tokenizer,
        )
        self._pools.track_stream(pool, stream._task)
        return stream

    def prewarm(self) -> None:
        self._pools.current.prewarm()

    def update_endpoint(self, websocket_v1_url: str, *, model_changed: bool) -> None:
        """Update the connection URL after validating its model binding."""
        _websocket_v1.validate_websocket_url(
            websocket_v1_url, allow_custom_endpoint=self._allow_custom_endpoint
        )
        model_endpoint_changed = _websocket_v1._model_endpoint_identity(
            websocket_v1_url
        ) != _websocket_v1._model_endpoint_identity(self._websocket_v1_url)
        if model_changed and not model_endpoint_changed:
            raise ValueError("model cannot change without changing websocket_url")
        if websocket_v1_url == self._websocket_v1_url:
            return

        self._websocket_v1_url = websocket_v1_url
        self._pools.replace(self._new_pool())

    async def aclose(self) -> None:
        await self._pools.aclose()


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
        self._sample_rate = tts_instance.sample_rate
        super().__init__(
            tts=StreamTTS(tts_instance, model=options.model), conn_options=conn_options
        )
        self._pool = pool
        self._options = options
        self._sentence_tokenizer = sentence_tokenizer
        self._end_input_sentinel: object | None = None

    def _enqueue_tokenizer_drain(self) -> tts.SynthesizeStream._FlushSentinel:
        sentinel = self._FlushSentinel()
        self._input_ch.send_nowait(sentinel)
        self._input_buffer.append(sentinel)
        return sentinel

    def flush(self) -> None:
        """Drain the local tokenizer without ending Rime input.

        This implements the LiveKit stream interface. It does not send a Rime
        protocol message because the Rime v1 protocol has no ``flush`` operation.
        Any released text is sent through the current context as a normal ``text``
        message. The stream accepts more text after this call.
        """
        if self._input_ch.closed:
            return

        # Do not call super().flush(), which records a LiveKit segment boundary. Keep
        # the Rime context, metric text, and segment state active while the local
        # sentence tokenizer drains.
        self._enqueue_tokenizer_drain()

    def end_input(self) -> None:
        """Finalize Rime input and let the stream wait for the provider ``done`` event."""
        if self._input_ch.closed:
            return

        if self._mtc_text:
            self._mtc_pending_texts.append(self._mtc_text)
            self._mtc_text = ""

        self._end_input_sentinel = self._enqueue_tokenizer_drain()
        self._input_ch.close()
        self._input_ended = True

    async def aclose(self) -> None:
        """Cancel the active Rime context when synthesis has not finished."""
        await super().aclose()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        async def _raw_input_events() -> AsyncIterable[str | _DrainTokenizer]:
            async for event in self._input_ch:
                if isinstance(event, self._FlushSentinel):
                    if event is not self._end_input_sentinel:
                        yield _DrainTokenizer()
                else:
                    yield event

        input_events = _sentence_tokenized_input_events(
            _raw_input_events(),
            sentence_tokenizer=self._sentence_tokenizer,
            language=self._options.language,
        )

        connection = await self._pool.get(timeout=self._conn_options.timeout)
        self._acquire_time = self._pool.last_acquire_time
        self._connection_reused = self._pool.last_connection_reused
        reusable = False
        try:
            result = await _websocket_v1.run_context(
                connection,
                context_id=utils.shortuuid(),
                options=self._options,
                sample_rate=self._sample_rate,
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
                self._pool.put(connection)
            else:
                self._pool.remove(connection)
                await _websocket_v1.close(connection)


async def _sentence_tokenized_input_events(
    input_events: AsyncIterable[str | _DrainTokenizer],
    *,
    sentence_tokenizer: tokenize.SentenceTokenizer,
    language: str,
) -> AsyncIterable[str]:
    """Convert text fragments to sentence units and handle local drain requests."""
    output = utils.aio.Chan[str]()

    async def _drive_input() -> None:
        sentence_stream: tokenize.SentenceStream | None = None
        forward_task: asyncio.Task[None] | None = None
        next_input_task: asyncio.Task[str | _DrainTokenizer] | None = None
        input_iterator = aiter(input_events)

        async def _next_input() -> str | _DrainTokenizer:
            return await anext(input_iterator)

        async def _forward_sentences(stream: tokenize.SentenceStream) -> None:
            async for event in stream:
                text = event.token
                if text and not text[-1].isspace():
                    text += " "
                if text:
                    output.send_nowait(text)

        def _start_forwarding(stream: tokenize.SentenceStream) -> asyncio.Task[None]:
            return asyncio.create_task(
                _forward_sentences(stream), name="rime-v1-sentence-tokenizer-output"
            )

        try:
            sentence_stream = sentence_tokenizer.stream(language=language)
            forward_task = _start_forwarding(sentence_stream)
            while True:
                next_input_task = asyncio.create_task(_next_input(), name="rime-v1-next-input")
                # A tokenizer failure must interrupt an open, idle input stream.
                await asyncio.wait(
                    (next_input_task, forward_task), return_when=asyncio.FIRST_COMPLETED
                )
                if forward_task.done():
                    await forward_task
                try:
                    event = await next_input_task
                except StopAsyncIteration:
                    break
                if isinstance(event, _DrainTokenizer):
                    sentence_stream.end_input()
                    await forward_task
                    sentence_stream = sentence_tokenizer.stream(language=language)
                    forward_task = _start_forwarding(sentence_stream)
                else:
                    sentence_stream.push_text(event)

            sentence_stream.end_input()
            await forward_task
        except Exception:
            raise APIError("Rime sentence tokenization failed", retryable=False) from None
        finally:
            try:
                if next_input_task is not None:
                    await utils.aio.gracefully_cancel(next_input_task)
                if forward_task is not None:
                    await utils.aio.gracefully_cancel(forward_task)
                if sentence_stream is not None and not sentence_stream.closed:
                    await sentence_stream.aclose()
            finally:
                output.close()

    input_task = asyncio.create_task(_drive_input(), name="rime-v1-sentence-tokenizer-input")
    try:
        async for event in output:
            yield event
        await input_task
    finally:
        await utils.aio.gracefully_cancel(input_task)
