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
        self._ws_session: AsyncStreamingTTSSession | None = None
        self._session_lock = asyncio.Lock()
        self._keepalive_task: asyncio.Task[None] | None = None
        self._needs_reconnect = False

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
        """Update synthesis options. Takes effect on the next WebSocket session."""
        next_opts = replace(self._opts)
        if model is not None:
            next_opts.model = model
        if voice is not None:
            next_opts.voice = voice
        if language is not None:
            next_opts.language = language
        if sample_rate is not None:
            next_opts.sample_rate = sample_rate
            self._sample_rate = int(sample_rate)
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
        self._needs_reconnect = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._invalidate_ws_session())

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
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._ensure_ws_session())

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._invalidate_ws_session()

    def _session_config(self) -> TTSSessionConfig:
        return TTSSessionConfig(
            model=str(self._opts.model),
            voice=self._opts.voice,
            language=str(self._opts.language),
            sample_rate=int(self._opts.sample_rate),
            speed=self._opts.speed,
            output_format="pcm",
        )

    async def _ensure_ws_session(self) -> AsyncStreamingTTSSession:
        async with self._session_lock:
            if self._needs_reconnect:
                await self._stop_keepalive()
                if self._ws_session is not None:
                    await self._ws_session.close()
                    self._ws_session = None
                self._needs_reconnect = False

            if self._ws_session is not None and self._ws_session.connected:
                return self._ws_session

            session = AsyncStreamingTTSSession(
                api_key=self._opts.api_key,
                base_url=self._opts.base_url,
                allow_insecure_base_url=self._opts.allow_insecure_base_url,
                config=self._session_config(),
            )
            await session.connect()
            self._ws_session = session
            self._start_keepalive()
            return session

    async def _invalidate_ws_session(self) -> None:
        async with self._session_lock:
            self._needs_reconnect = False
            await self._stop_keepalive()
            if self._ws_session is not None:
                await self._ws_session.close()
                self._ws_session = None

    def _start_keepalive(self) -> None:
        if self._keepalive_task and not self._keepalive_task.done():
            return
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def _stop_keepalive(self) -> None:
        task = self._keepalive_task
        self._keepalive_task = None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _keepalive_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(KEEPALIVE_INTERVAL_SECONDS)
                session = self._ws_session
                if session is None or not session.connected:
                    return
                # Avoid racing with synthesize_stream / cancel drain on recv.
                if session.utterance_active:
                    continue
                try:
                    await session.ping()
                except Exception:
                    logger.debug("Vakyam TTS keepalive failed; resetting session", exc_info=True)
                    await self._invalidate_ws_session()
                    return
        except asyncio.CancelledError:
            return


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
                    total=self._conn_options.timeout,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise_http_error(resp.status, body)

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
        self._segments_ch = utils.aio.Chan[tokenize.SentenceStream]()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
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
                        self._segments_ch.send_nowait(sentence_stream)
                    sentence_stream.push_text(data)
                elif isinstance(data, self._FlushSentinel):
                    if sentence_stream is not None:
                        sentence_stream.end_input()
                        sentence_stream = None
            if sentence_stream is not None:
                sentence_stream.end_input()
            self._segments_ch.close()

        async def _process_segments() -> None:
            async for sentence_stream in self._segments_ch:
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
        session = await self._tts._ensure_ws_session()

        started = False
        async for sentence in sentence_stream:
            text = sentence.token.strip()
            if not text:
                continue
            if not started:
                self._mark_started()
                started = True
            try:
                async for chunk in session.synthesize_stream(text):
                    output_emitter.push(chunk)
            except asyncio.CancelledError:
                # synthesize_stream already sent cancel + drained; keep WS open.
                raise
            except Exception:
                # Stale or broken socket — reconnect once and retry the utterance.
                await self._tts._invalidate_ws_session()
                session = await self._tts._ensure_ws_session()
                async for chunk in session.synthesize_stream(text):
                    output_emitter.push(chunk)

            if session.last_result and session.last_result.cancelled:
                logger.debug("Vakyam TTS utterance cancelled")
                break
            if session.last_result and session.last_result.truncated:
                logger.warning(
                    "Vakyam TTS utterance truncated",
                    extra={"reason": session.last_result.truncation_reason},
                )

        output_emitter.end_segment()

    async def aclose(self) -> None:
        self._segments_ch.close()
        await super().aclose()
