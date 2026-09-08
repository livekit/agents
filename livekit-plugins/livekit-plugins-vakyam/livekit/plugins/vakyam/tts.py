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

"""Text-to-Speech implementation for Vakyam AI (Raaga 1).

Streaming uses the realtime WebSocket API with sentence tokenization because
Vakyam expects one complete utterance per ``text`` message. One-shot
``synthesize()`` uses HTTP ``POST /v1/tts/stream`` for PCM bytes.
"""

from __future__ import annotations

import asyncio
import os
import weakref
from contextlib import suppress
from dataclasses import dataclass, replace

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

from ._utils import (
    http_stream_url,
    normalize_base_url,
    raise_http_error,
    speech_payload,
    split_text,
    validate_tts_options,
)
from ._websocket import AsyncStreamingTTSSession, TTSSessionConfig
from .log import logger
from .models import (
    DEFAULT_BASE_URL,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    KEEPALIVE_INTERVAL_SECONDS,
    TTSLanguages,
    TTSModels,
    TTSSampleRates,
)
from .version import __version__

NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: str
    language: TTSLanguages | str
    sample_rate: TTSSampleRates | int
    speed: float
    api_key: str
    base_url: str
    allow_insecure_base_url: bool


class TTS(tts.TTS):
    """Vakyam AI text-to-speech for LiveKit Agents.

    Uses the realtime WebSocket API (``WS /v1/tts/websocket``) for low-latency
    streaming. Text is sentence-tokenized because Vakyam expects one complete
    utterance per synthesis request.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        language: TTSLanguages | str = DEFAULT_LANGUAGE,
        sample_rate: TTSSampleRates | int = DEFAULT_SAMPLE_RATE,
        speed: float = DEFAULT_SPEED,
        base_url: str | None = None,
        allow_insecure_base_url: bool = False,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Vakyam TTS instance.

        Args:
            api_key: Vakyam API key. Defaults to ``VAKYAM_API_KEY``.
            model: TTS model id. Currently ``raaga-v1``.
            voice: Voice selector — a preset name from ``GET /v1/voices``, or a
                custom voice ID beginning with ``vc_``.
            language: BCP-47 language code (for example ``ta-IN``).
            sample_rate: Output PCM sample rate in Hz (8000, 16000, 24000, 48000).
            speed: Speech rate multiplier (``0.5``–``2.0``).
            base_url: API base URL. Defaults to ``https://api.vakyam.ai``.
            allow_insecure_base_url: Allow non-localhost ``http://`` base URLs.
            tokenizer: Sentence tokenizer used for streaming synthesis.
            http_session: Optional aiohttp session for HTTP ``synthesize()``.
        """
        resolved_key = api_key or os.environ.get("VAKYAM_API_KEY")
        if not resolved_key:
            raise ValueError("Vakyam API key is required, either as api_key= or VAKYAM_API_KEY")

        validate_tts_options(
            model=str(model),
            language=str(language),
            sample_rate=int(sample_rate),
            speed=speed,
            voice=voice,
        )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=int(sample_rate),
            num_channels=NUM_CHANNELS,
        )

        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            language=language,
            sample_rate=sample_rate,
            speed=speed,
            api_key=resolved_key,
            base_url=normalize_base_url(
                base_url or DEFAULT_BASE_URL, allow_insecure_base_url=allow_insecure_base_url
            ),
            allow_insecure_base_url=allow_insecure_base_url,
        )
        if is_given(tokenizer):
            self._sentence_tokenizer = tokenizer
        else:
            try:
                self._sentence_tokenizer = tokenize.blingfire.SentenceTokenizer()
            except Exception:
                self._sentence_tokenizer = tokenize.basic.SentenceTokenizer()
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._pools: dict[TTSSessionConfig, utils.ConnectionPool[AsyncStreamingTTSSession]] = {}
        self._ws_keepalive_tasks: dict[AsyncStreamingTTSSession, asyncio.Task[None]] = {}

    @property
    def model(self) -> str:
        return str(self._opts.model)

    @property
    def provider(self) -> str:
        return "Vakyam"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        model: TTSModels | str | None = None,
        voice: str | None = None,
        language: TTSLanguages | str | None = None,
        sample_rate: TTSSampleRates | int | None = None,
        speed: float | None = None,
    ) -> None:
        """Update synthesis options for streams created after this call."""
        next_opts = replace(self._opts)
        next_sample_rate = self._sample_rate
        if model is not None:
            next_opts.model = model
        if voice is not None:
            next_opts.voice = voice
        if language is not None:
            next_opts.language = language
        if sample_rate is not None:
            next_opts.sample_rate = sample_rate
            next_sample_rate = int(sample_rate)
        if speed is not None:
            next_opts.speed = speed

        validate_tts_options(
            model=str(next_opts.model),
            language=str(next_opts.language),
            sample_rate=int(next_opts.sample_rate),
            speed=next_opts.speed,
            voice=next_opts.voice,
        )
        self._opts = next_opts
        self._sample_rate = next_sample_rate

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
        self._pool_for(self._opts).prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        for pool in self._pools.values():
            await pool.aclose()
        self._pools.clear()

    def _session_config(self, opts: _TTSOptions) -> TTSSessionConfig:
        return TTSSessionConfig(
            model=str(opts.model),
            voice=opts.voice,
            language=str(opts.language),
            sample_rate=int(opts.sample_rate),
            speed=opts.speed,
            output_format="pcm",
        )

    def _pool_for(self, opts: _TTSOptions) -> utils.ConnectionPool[AsyncStreamingTTSSession]:
        config = self._session_config(opts)
        existing_pool = self._pools.get(config)
        if existing_pool is not None:
            return existing_pool

        pool_ref: utils.ConnectionPool[AsyncStreamingTTSSession] | None = None

        async def _connect(timeout: float) -> AsyncStreamingTTSSession:
            session = AsyncStreamingTTSSession(
                api_key=opts.api_key,
                base_url=opts.base_url,
                allow_insecure_base_url=opts.allow_insecure_base_url,
                config=config,
            )
            await session.connect(timeout=timeout)
            assert pool_ref is not None
            self._start_keepalive(session, pool_ref)
            return session

        pool = utils.ConnectionPool(
            connect_cb=_connect,
            close_cb=self._close_ws,
            max_session_duration=3600,
            mark_refreshed_on_get=False,
        )
        pool_ref = pool
        self._pools[config] = pool
        return pool

    async def _close_ws(self, session: AsyncStreamingTTSSession) -> None:
        await self._stop_keepalive(session)
        await session.close()

    def _start_keepalive(
        self,
        session: AsyncStreamingTTSSession,
        pool: utils.ConnectionPool[AsyncStreamingTTSSession],
    ) -> None:
        task = self._ws_keepalive_tasks.get(session)
        if task is not None and not task.done():
            return
        self._ws_keepalive_tasks[session] = asyncio.create_task(
            self._keepalive_loop(session, pool), name="vakyam-tts-ws-keepalive"
        )

    async def _stop_keepalive(self, session: AsyncStreamingTTSSession) -> None:
        task = self._ws_keepalive_tasks.pop(session, None)
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _keepalive_loop(
        self,
        session: AsyncStreamingTTSSession,
        pool: utils.ConnectionPool[AsyncStreamingTTSSession],
    ) -> None:
        try:
            while True:
                await asyncio.sleep(KEEPALIVE_INTERVAL_SECONDS)
                if not session.connected:
                    return
                try:
                    await session.ping()
                except Exception as exc:
                    logger.debug(
                        "Vakyam TTS keepalive failed (%s); evicting session",
                        type(exc).__name__,
                    )
                    pool.remove(session)
                    return
        except asyncio.CancelledError:
            return
        finally:
            current = asyncio.current_task()
            if self._ws_keepalive_tasks.get(session) is current:
                self._ws_keepalive_tasks.pop(session, None)


class ChunkedStream(tts.ChunkedStream):
    """One-shot synthesis over HTTP ``POST /v1/tts/stream``."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        payload = speech_payload(
            text=self._input_text,
            model=str(self._opts.model),
            voice=self._opts.voice,
            language=str(self._opts.language),
            sample_rate=int(self._opts.sample_rate),
            speed=self._opts.speed,
            output_format="pcm",
        )
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"LiveKit-Agents-Vakyam/{__version__}",
        }
        try:
            async with self._tts._ensure_session().post(
                http_stream_url(self._opts.base_url),
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    sock_connect=self._conn_options.timeout,
                    sock_read=self._conn_options.timeout,
                ),
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise_http_error(resp.status, body)

                content_type = resp.headers.get("Content-Type", "").lower()
                if content_type and not (
                    content_type.startswith("audio/")
                    or content_type.startswith("application/octet-stream")
                ):
                    body = await resp.text()
                    raise APIStatusError(
                        "Vakyam TTS returned a non-audio response",
                        status_code=502,
                        body=body,
                    )

                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=int(self._opts.sample_rate),
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )
                async for chunk, _ in resp.content.iter_chunks():
                    if chunk:
                        output_emitter.push(chunk)
                output_emitter.flush()
        except (APIStatusError, APIConnectionError, APITimeoutError):
            raise
        except asyncio.TimeoutError as exc:
            raise APITimeoutError("Vakyam TTS HTTP stream timed out") from exc
        except aiohttp.ClientError as exc:
            raise APIConnectionError(f"Vakyam TTS HTTP stream connection error: {exc}") from exc


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming synthesis with sentence tokenization over a persistent WebSocket."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        segments_ch = utils.aio.Chan[tokenize.SentenceStream]()
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=int(self._opts.sample_rate),
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
            frame_size_ms=50,
        )

        async def _tokenize_input() -> None:
            sentence_stream: tokenize.SentenceStream | None = None
            async for data in self._input_ch:
                if isinstance(data, str):
                    if sentence_stream is None:
                        sentence_stream = self._tts._sentence_tokenizer.stream()
                        segments_ch.send_nowait(sentence_stream)
                    sentence_stream.push_text(data)
                elif isinstance(data, self._FlushSentinel):
                    if sentence_stream is not None:
                        sentence_stream.end_input()
                        sentence_stream = None
            if sentence_stream is not None:
                sentence_stream.end_input()
            segments_ch.close()

        async def _process_segments() -> None:
            async for sentence_stream in segments_ch:
                await self._run_segment(sentence_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_process_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except (APIStatusError, APIConnectionError, APITimeoutError):
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise APIConnectionError(f"Vakyam TTS stream failed: {exc}") from exc
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            output_emitter.end_input()

    async def _run_segment(
        self, sentence_stream: tokenize.SentenceStream, output_emitter: tts.AudioEmitter
    ) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)
        pool = self._tts._pool_for(self._opts)
        deferred_error: APIStatusError | None = None
        cancelled = False

        async with pool.connection(timeout=self._conn_options.timeout) as session:
            self._acquire_time = pool.last_acquire_time
            self._connection_reused = pool.last_connection_reused
            await self._tts._stop_keepalive(session)
            reusable = False
            try:
                started = False
                async for sentence in sentence_stream:
                    for text in split_text(sentence.token):
                        if not started:
                            self._mark_started()
                            started = True

                        async for chunk in session.synthesize_stream(
                            text, timeout=self._conn_options.timeout
                        ):
                            output_emitter.push(chunk)

                        if session.last_result and session.last_result.cancelled:
                            logger.debug("Vakyam TTS utterance cancelled")
                            break
                        if session.last_result and session.last_result.truncated:
                            deferred_error = APIStatusError(
                                "Vakyam TTS utterance was truncated",
                                status_code=500,
                                retryable=False,
                            )
                            break
                    if deferred_error is not None or (
                        session.last_result and session.last_result.cancelled
                    ):
                        break
                reusable = True
            except asyncio.CancelledError:
                # Reuse only after Vakyam acknowledges cancel and trailing audio is drained.
                reusable = bool(session.last_result and session.last_result.cancelled)
                if not reusable:
                    raise
                cancelled = True
            except APIStatusError as exc:
                # Only JSON per-message errors leave the WebSocket open. Close-code
                # status errors represent a dead transport and must be evicted.
                if isinstance(exc.body, dict) and exc.body.get("type") == "error":
                    reusable = True
                    deferred_error = exc
                else:
                    raise
            finally:
                if reusable:
                    self._tts._start_keepalive(session, pool)

        if cancelled:
            raise asyncio.CancelledError
        if deferred_error is not None:
            raise deferred_error
        output_emitter.end_segment()
