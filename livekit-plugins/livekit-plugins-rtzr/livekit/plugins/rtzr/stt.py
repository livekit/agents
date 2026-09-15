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
import json
import time
import weakref
from dataclasses import dataclass, replace

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.voice.io import TimedString

from .log import logger
from .rtzrapi import DEFAULT_SAMPLE_RATE, RTZRConnectionError, RTZROpenAPIClient, RTZRStatusError

_STREAMING_CHUNK_MS = 200
_IDLE_TIMEOUT_SECONDS = 25.0
_RECV_COMPLETION_TIMEOUT = 5.0
_IDLE_CHECK_INTERVAL = 1.0
_BYTES_PER_SAMPLE = 2
_FINALIZE_MESSAGE = '{"type":"Finalize"}'
_SUPPORTED_PLUGIN_WS_ENCODINGS = {"LINEAR16"}
_SUPPORTED_DOMAINS = {"CALL", "MEETING"}


@dataclass
class _STTOptions:
    model_name: str = "sommers_ko"  # sommers_ko: "ko", sommers_ja: "ja"
    language: LanguageCode = LanguageCode("ko")  # ko, ja, en
    sample_rate: int = DEFAULT_SAMPLE_RATE
    encoding: str = "LINEAR16"  # or "OGG_OPUS" in future
    domain: str = "CALL"  # CALL, MEETING
    epd_time: float = 0.8  # endpoint detection time in seconds
    noise_threshold: float = 0.60
    active_threshold: float = 0.80
    use_itn: bool = True
    use_disfluency_filter: bool = False
    use_profanity_filter: bool = False
    use_punctuation: bool = False
    keywords: list[str] | list[tuple[str, float]] | None = None


class STT(stt.STT):
    """RTZR Streaming STT over WebSocket."""

    def __init__(
        self,
        *,
        model: str = "sommers_ko",
        language: str = "ko",
        sample_rate: int = 8000,
        encoding: str = "LINEAR16",
        domain: str = "CALL",
        epd_time: float = 0.8,
        noise_threshold: float = 0.60,
        active_threshold: float = 0.80,
        use_itn: bool = True,
        use_disfluency_filter: bool = False,
        use_profanity_filter: bool = False,
        use_punctuation: bool = False,
        keywords: list[str] | list[tuple[str, float]] | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                # word timestamps don't seem to work despite the docs saying they do
                aligned_transcript="chunk",
                offline_recognize=False,
            )
        )

        normalized_encoding = encoding.upper()
        normalized_domain = domain.upper()
        if not 8000 <= sample_rate <= 48000:
            raise ValueError("RTZR sample_rate must be between 8000 and 48000 Hz")
        if normalized_encoding not in _SUPPORTED_PLUGIN_WS_ENCODINGS:
            raise ValueError("RTZR encoding must be LINEAR16")
        if normalized_domain not in _SUPPORTED_DOMAINS:
            raise ValueError("RTZR domain must be CALL or MEETING")

        self._params = _STTOptions(
            model_name=model,
            language=LanguageCode({"sommers_ko": "ko", "sommers_ja": "ja"}.get(model, language)),
            sample_rate=sample_rate,
            encoding=normalized_encoding,
            domain=normalized_domain,
            epd_time=epd_time,
            noise_threshold=noise_threshold,
            active_threshold=active_threshold,
            use_itn=use_itn,
            use_disfluency_filter=use_disfluency_filter,
            use_profanity_filter=use_profanity_filter,
            use_punctuation=use_punctuation,
            keywords=keywords,
        )
        if keywords and not (model == "sommers_ko" or (model == "whisper" and language == "ko")):
            logger.warning("RTZR keyword boosting requires sommers_ko or Korean whisper")
        self._client = RTZROpenAPIClient(http_session=http_session)
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return self._params.model_name

    @property
    def provider(self) -> str:
        return "RTZR"

    async def aclose(self) -> None:
        """Close the RTZR client and cleanup resources."""
        await asyncio.gather(*(stream.aclose() for stream in self._streams))
        await self._client.close()

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Single-shot recognition is not supported; use stream().")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Create a stream, optionally overriding the Whisper language for this stream."""
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            language=language,
        )
        self._streams.add(stream)
        return stream


class SpeechStream(stt.RecognizeStream):
    _pending_input: rtc.AudioFrame | stt.RecognizeStream._FlushSentinel | None

    def __init__(
        self, *, stt: STT, conn_options: APIConnectOptions, language: NotGivenOr[str] = NOT_GIVEN
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._params.sample_rate)
        self._rtzr_stt = stt
        self._opts = replace(stt._params)
        if utils.is_given(language) and self._opts.model_name == "whisper":
            self._opts.language = LanguageCode(language)
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._connection_lock = asyncio.Lock()
        self._audio_chunker = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            samples_per_channel=self._opts.sample_rate * _STREAMING_CHUNK_MS // 1000,
            progressive=True,
        )
        self._pending_input = None
        self._pending_usage_audio_duration = 0.0
        self._idle_timeout = _IDLE_TIMEOUT_SECONDS
        self._last_audio_at = 0.0
        self._closing = False
        self._connection_offset = 0.0
        self._run_started_at = 0.0
        self._failure: asyncio.Future[None] | None = None

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        config = self._rtzr_stt._client.build_config(
            model_name=self._opts.model_name,
            domain=self._opts.domain,
            sample_rate=self._opts.sample_rate,
            encoding=self._opts.encoding,
            epd_time=self._opts.epd_time,
            noise_threshold=self._opts.noise_threshold,
            active_threshold=self._opts.active_threshold,
            use_itn=self._opts.use_itn,
            use_disfluency_filter=self._opts.use_disfluency_filter,
            use_profanity_filter=self._opts.use_profanity_filter,
            use_punctuation=self._opts.use_punctuation,
            keywords=self._opts.keywords,
            language=self._opts.language,
        )

        try:
            ws = await asyncio.wait_for(
                self._rtzr_stt._client.connect_websocket(config),
                timeout=self._conn_options.timeout,
            )
            logger.debug(
                "RTZR STT WS connected (model=%s, sr=%s, enc=%s, domain=%s, epd=%.2fs, "
                "noise=%.2f, active=%.2f, itn=%s, disfluency=%s, profanity=%s, punct=%s)",
                self._opts.model_name,
                self._opts.sample_rate,
                self._opts.encoding,
                self._opts.domain,
                self._opts.epd_time,
                self._opts.noise_threshold,
                self._opts.active_threshold,
                self._opts.use_itn,
                self._opts.use_disfluency_filter,
                self._opts.use_profanity_filter,
                self._opts.use_punctuation,
            )
            return ws
        except asyncio.TimeoutError:
            raise APITimeoutError("WebSocket connection timeout") from None
        except RTZRStatusError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status_code or 500,
                request_id=None,
                body=None,
            ) from None
        except RTZRConnectionError:
            raise APIConnectionError("RTZR API connection failed") from None

    async def _run(self) -> None:
        self._run_started_at = time.monotonic()
        self._failure = asyncio.get_running_loop().create_future()
        send_task = asyncio.create_task(self._send_audio_task(), name="RTZR.send_audio")
        idle_task = asyncio.create_task(self._idle_watchdog(), name="RTZR.idle_watchdog")
        graceful = False
        try:
            try:
                done, _ = await asyncio.wait(
                    [send_task, idle_task, self._failure], return_when=asyncio.FIRST_COMPLETED
                )
                if self._failure in done:
                    await self._failure
                if idle_task in done:
                    await idle_task
                await send_task
                graceful = True
            finally:
                await utils.aio.cancel_and_wait(send_task, idle_task)
                for task in (send_task, idle_task):
                    if not task.cancelled():
                        task.exception()
                try:
                    async with self._connection_lock:
                        await self._close_connection(graceful=graceful)
                finally:
                    if self._failure.done() and not self._failure.cancelled():
                        self._failure.exception()
                    self._failure.cancel()
        except APIError as error:
            # Retrying an exhausted input cannot recover a missing final response.
            if self._input_ch.closed and self._input_ch.empty() and self._pending_input is None:
                error.retryable = False
            raise

    def _recv_done(self, task: asyncio.Task[None]) -> None:
        # Retrieve every receiver failure, including errors during shutdown.
        if not task.cancelled() and (error := task.exception()) is not None:
            if self._failure is not None and not self._failure.done():
                self._failure.set_exception(error)

    async def _ensure_connected(self) -> None:
        # All connection and send operations run under _connection_lock.
        if self._ws is None:
            self._connection_offset = (
                self.start_time_offset + time.monotonic() - self._run_started_at
            )
            self._ws = await self._connect_ws()
            self._closing = False
            self._recv_task = asyncio.create_task(self._recv_loop(self._ws), name="RTZR.recv_loop")
            self._recv_task.add_done_callback(self._recv_done)
            self._last_audio_at = time.monotonic()

    async def _send_audio(self, audio: bytes) -> None:
        await self._ensure_connected()
        assert self._ws is not None
        try:
            await self._ws.send_bytes(audio)
        except (aiohttp.ClientError, OSError):
            raise APIConnectionError("RTZR audio send failed") from None
        self._record_sent_audio(audio)
        self._last_audio_at = time.monotonic()

    async def _finalize_segment(self) -> None:
        for frame in self._audio_chunker.flush():
            await self._send_audio(frame.data.tobytes())
        self._audio_chunker.clear()
        if self._pending_usage_audio_duration <= 0.0:
            return
        assert self._ws is not None
        try:
            await self._ws.send_str(_FINALIZE_MESSAGE)
        except (aiohttp.ClientError, OSError):
            raise APIConnectionError("RTZR finalize send failed") from None
        self._emit_usage_event_if_needed()

    async def _close_connection(self, *, graceful: bool) -> None:
        ws, recv_task = self._ws, self._recv_task
        self._closing = True
        try:
            frames = self._audio_chunker.flush()
            self._audio_chunker.clear()
            if ws is None:
                return
            if graceful:
                for frame in frames:
                    await self._send_audio(frame.data.tobytes())
                try:
                    await ws.send_str("EOS")
                except (aiohttp.ClientError, OSError):
                    raise APIConnectionError("RTZR EOS send failed") from None
                if recv_task is not None:
                    try:
                        await asyncio.wait_for(recv_task, timeout=_RECV_COMPLETION_TIMEOUT)
                    except asyncio.TimeoutError:
                        raise APITimeoutError("RTZR final response timed out") from None
        finally:
            if recv_task is not None:
                await utils.aio.cancel_and_wait(recv_task)
            try:
                if ws is not None:
                    await asyncio.wait_for(ws.close(), timeout=_RECV_COMPLETION_TIMEOUT)
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as error:
                logger.warning(
                    "RTZR WebSocket cleanup failed", extra={"error_type": type(error).__name__}
                )
            finally:
                self._ws = None
                self._recv_task = None
                self._emit_usage_event_if_needed()

    async def _idle_watchdog(self) -> None:
        while True:
            await asyncio.sleep(_IDLE_CHECK_INTERVAL)
            async with self._connection_lock:
                if (
                    self._ws is not None
                    and time.monotonic() - self._last_audio_at >= self._idle_timeout
                ):
                    await self._close_connection(graceful=True)

    def _record_sent_audio(self, audio: bytes) -> None:
        sample_rate = self._opts.sample_rate
        self._pending_usage_audio_duration += len(audio) / (sample_rate * _BYTES_PER_SAMPLE)

    def _emit_usage_event_if_needed(self) -> None:
        if self._pending_usage_audio_duration <= 0.0:
            return

        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                alternatives=[],
                recognition_usage=stt.RecognitionUsage(
                    audio_duration=self._pending_usage_audio_duration
                ),
            )
        )
        self._pending_usage_audio_duration = 0.0

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Validate channels before the base stream resamples input audio."""
        if frame.num_channels != 1:
            raise ValueError("RTZR streaming requires mono audio")
        super().push_frame(frame)

    async def _send_audio_task(self) -> None:
        while True:
            if self._pending_input is None:
                try:
                    self._pending_input = await self._input_ch.__anext__()
                except StopAsyncIteration:
                    return
            async with self._connection_lock:
                if self._failure is not None and self._failure.done():
                    await self._failure
                data = self._pending_input
                if isinstance(data, rtc.AudioFrame):
                    # Keep the first frame across connection retries before consuming PCM.
                    await self._ensure_connected()
                    self._pending_input = None
                    for frame in self._audio_chunker.write(data.data.tobytes()):
                        await self._send_audio(frame.data.tobytes())
                    self._last_audio_at = time.monotonic()
                else:
                    self._pending_input = None
                    await self._finalize_segment()

    def _parse_words(self, words: list[dict], *, utterance_start: float) -> list[TimedString]:
        """Parse word timing data from RTZR response."""
        return [
            TimedString(
                text=w.get("text", ""),
                start_time=utterance_start
                + w.get("start_at", 0) / 1000.0
                + self._connection_offset,
                end_time=utterance_start
                + (w.get("start_at", 0) + w.get("duration", 0)) / 1000.0
                + self._connection_offset,
            )
            for w in words
        ]

    def _check_error_response(self, data: dict) -> None:
        """Reject server errors without exposing response contents in logs."""
        if "error" in data or data.get("type") == "error":
            raise APIStatusError(message="RTZR server returned an error", status_code=500)

    def _process_transcript_event(
        self,
        data: dict,
        in_speech: bool,
    ) -> tuple[list[stt.SpeechEvent], bool]:
        """Parse RTZR response into SpeechEvents.

        Returns: (events, in_speech)
        """
        start_time = data.get("start_at", 0) / 1000.0
        duration = data.get("duration", 0) / 1000.0

        if "alternatives" not in data or not data["alternatives"]:
            return [], in_speech

        alternative = data["alternatives"][0]
        words = alternative.get("words", [])
        text = alternative.get("text", "")
        is_final = bool(data.get("final", False))

        if not text:
            return [], in_speech

        events: list[stt.SpeechEvent] = []

        if not in_speech:
            in_speech = True
            events.append(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

        event_type = (
            stt.SpeechEventType.FINAL_TRANSCRIPT
            if is_final
            else stt.SpeechEventType.INTERIM_TRANSCRIPT
        )

        events.append(
            stt.SpeechEvent(
                type=event_type,
                alternatives=[
                    stt.SpeechData(
                        text=text,
                        language=self._opts.language,
                        start_time=start_time + self._connection_offset,
                        end_time=start_time + duration + self._connection_offset,
                        words=self._parse_words(words, utterance_start=start_time)
                        if words
                        else None,
                        confidence=alternative.get("confidence", 0.0),
                    )
                ],
            )
        )

        if is_final:
            events.append(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
            in_speech = False

        return events, in_speech

    async def _recv_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        in_speech = False
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        raise APIConnectionError("Invalid RTZR JSON response") from None
                    if not isinstance(data, dict):
                        raise APIConnectionError("Invalid RTZR response object")
                    self._check_error_response(data)
                    events, in_speech = self._process_transcript_event(data, in_speech)
                    for event in events:
                        self._event_ch.send_nowait(event)
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError("RTZR WebSocket receive failed")
        except (aiohttp.ClientError, OSError):
            raise APIConnectionError("RTZR WebSocket receive failed") from None
        if not self._closing:
            raise APIConnectionError("RTZR WebSocket closed before EOS")
