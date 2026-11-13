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

Streaming uses the realtime WebSocket API, one ``text`` message per prepared
chunk because Vakyam expects one complete utterance per message. One-shot
``synthesize()`` uses HTTP ``POST /v1/tts/stream`` for PCM bytes.
"""

from __future__ import annotations

import asyncio
import os
import weakref
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, replace

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from ._utils import (
    http_stream_url,
    normalize_base_url,
    raise_http_error,
    speech_payload,
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
    MAX_TEXT_CHARACTERS,
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
    streaming. Each prepared text chunk is one synthesis request because Vakyam
    expects one complete utterance per request.
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
        http_session: http.HttpSession | None = None,
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
            http_session: Optional HttpSession for HTTP ``synthesize()``.
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
            capabilities=tts.TTSCapabilities(
                streaming=True, aligned_transcript=False, max_text_len=MAX_TEXT_CHARACTERS
            ),
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
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._pools: dict[TTSSessionConfig, utils.ConnectionPool[AsyncStreamingTTSSession]] = {}
        self._ws_keepalive_tasks: dict[AsyncStreamingTTSSession, asyncio.Task[None]] = {}
        self._tasks = utils.aio.TaskSet()

    @property
    def model(self) -> str:
        return str(self._opts.model)

    @property
    def provider(self) -> str:
        return "Vakyam"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    def update_options(
        self,
        *,
        model: TTSModels | str | None = None,
        voice: str | None = None,
        language: TTSLanguages | str | None = None,
        sample_rate: TTSSampleRates | int | None = None,
        speed: float | None = None,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """Update synthesis options for streams created after this call."""
        super().update_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing)
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

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            opts=replace(self._opts),
            http_session=self._ensure_session,
            input_text=chunk.text,
            conn_options=conn_options,
        )

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        opts = replace(self._opts)
        stream = SynthesizeStream(
            tts=self,
            opts=opts,
            pool=self._pool_for(opts),
            start_keepalive=self._start_keepalive,
            stop_keepalive=self._stop_keepalive,
            reuse_after_cancel=self._reuse_after_cancel,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        self._pool_for(self._opts).prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        # keepalives and cancel drains first: each hands its socket back to its pool
        await self._tasks.aclose()
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

        async def _connect(conn_options: APIConnectOptions) -> AsyncStreamingTTSSession:
            session = AsyncStreamingTTSSession(
                api_key=opts.api_key,
                base_url=opts.base_url,
                allow_insecure_base_url=opts.allow_insecure_base_url,
                config=config,
                session=self._ensure_session(),
            )
            await session.connect(conn_options)
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
        if (task is not None and not task.done()) or self._tasks.closed:
            return
        self._ws_keepalive_tasks[session] = self._tasks.spawn(
            self._keepalive_loop(session, pool), name="vakyam-tts-ws-keepalive"
        )

    def _reuse_after_cancel(
        self,
        session: AsyncStreamingTTSSession,
        pool: utils.ConnectionPool[AsyncStreamingTTSSession],
        timeout: float | None,
    ) -> None:
        """Returns a socket whose utterance was cancelled to its pool once Vakyam acknowledges
        the cancel. The model owns the drain, so closing the stream never waits on it."""
        if self._tasks.closed:
            pool.remove(session)
            return

        async def _drain() -> None:
            try:
                await session.abort(timeout=timeout)
            except BaseException:
                pool.remove(session)
                raise
            if session.last_result is not None and session.last_result.cancelled:
                pool.put(session)
                self._start_keepalive(session, pool)
            else:
                pool.remove(session)

        self._tasks.spawn(_drain(), name="vakyam-tts-cancel-drain")

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

    def __init__(
        self,
        *,
        tts: TTS,
        opts: _TTSOptions,
        http_session: Callable[[], http.HttpSession],
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._http_session = http_session

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
        async with self._http_session().post(
            http_stream_url(self._opts.base_url),
            json=payload,
            headers=headers,
            conn_options=self._conn_options,
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
            async for chunk in resp.iter_bytes():
                if chunk:
                    output_emitter.push(chunk)
            output_emitter.flush()


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming synthesis over a persistent WebSocket, one request per chunk."""

    def __init__(
        self,
        *,
        tts: TTS,
        opts: _TTSOptions,
        pool: utils.ConnectionPool[AsyncStreamingTTSSession],
        start_keepalive: Callable[
            [AsyncStreamingTTSSession, utils.ConnectionPool[AsyncStreamingTTSSession]], None
        ],
        stop_keepalive: Callable[[AsyncStreamingTTSSession], Awaitable[None]],
        reuse_after_cancel: Callable[
            [
                AsyncStreamingTTSSession,
                utils.ConnectionPool[AsyncStreamingTTSSession],
                float | None,
            ],
            None,
        ],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self._pool = pool
        self._start_keepalive = start_keepalive
        self._stop_keepalive = stop_keepalive
        self._reuse_after_cancel = reuse_after_cancel

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        segments_ch = utils.aio.Chan[utils.aio.Chan[str]]()
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=int(self._opts.sample_rate),
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
            frame_size_ms=50,
        )

        async def _split_segments() -> None:
            segment: utils.aio.Chan[str] | None = None
            async for data in self.input:
                if isinstance(data, tts.SpeechChunk):
                    if segment is None:
                        segment = utils.aio.Chan[str]()
                        segments_ch.send_nowait(segment)
                    segment.send_nowait(data.text)
                elif segment is not None:
                    segment.close()
                    segment = None
            if segment is not None:
                segment.close()
            segments_ch.close()

        async def _process_segments() -> None:
            async for segment in segments_ch:
                await self._run_segment(segment, output_emitter)

        try:
            async with utils.aio.TaskSet() as tasks:
                await asyncio.gather(
                    tasks.spawn(_split_segments()), tasks.spawn(_process_segments())
                )
        except APIError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise APIConnectionError(f"Vakyam TTS stream failed: {exc}") from exc
        finally:
            output_emitter.end_input()

    async def _run_segment(
        self, segment: utils.aio.Chan[str], output_emitter: tts.AudioEmitter
    ) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)
        pool = self._pool
        deferred_error: APIStatusError | None = None

        session = await pool.get(conn_options=self._conn_options)
        speaking = False
        try:
            await self._stop_keepalive(session)
            started = False
            async for text in segment:
                if not started:
                    self.mark_started()
                    started = True

                speaking = True
                async for chunk in session.synthesize_stream(
                    text, timeout=self._conn_options.read_timeout
                ):
                    output_emitter.push(chunk)
                speaking = False

                if session.last_result:
                    output_emitter.report_usage(
                        characters_count=session.last_result.characters_used
                    )
                if session.last_result and session.last_result.cancelled:
                    logger.debug("Vakyam TTS utterance cancelled")
                    break
                if session.last_result and session.last_result.truncated:
                    deferred_error = APIStatusError(
                        "Vakyam TTS utterance was truncated",
                        status_code=500,
                    )
                    break
        except asyncio.CancelledError:
            if speaking:
                self._reuse_after_cancel(session, pool, self._conn_options.read_timeout)
            else:
                pool.put(session)
                self._start_keepalive(session, pool)
            raise
        except APIStatusError as exc:
            # Only JSON per-message errors leave the WebSocket open. Close-code
            # status errors represent a dead transport and must be evicted.
            match exc.body:
                case {"type": "error"}:
                    deferred_error = exc
                case _:
                    pool.remove(session)
                    raise
        except BaseException:
            pool.remove(session)
            raise
        pool.put(session)
        self._start_keepalive(session, pool)

        if deferred_error is not None:
            raise deferred_error
        output_emitter.end_segment()
