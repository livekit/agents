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
from dataclasses import dataclass
from enum import Enum, auto

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    Language,
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

_INITIAL_CHUNK_MS = 50
_STREAMING_CHUNK_MS = 200
_IDLE_TIMEOUT_SECONDS = 25.0
_RECV_COMPLETION_TIMEOUT = 5.0
_IDLE_CHECK_INTERVAL = 1.0
# originally supported encodings from RTZR: https://developers.rtzr.ai/docs/en/stt-streaming/#supported-encodings
# rtc.AudioFrame uses raw PCM audio frame. You should use LINEAR16 encoding.
_SUPPORTED_PLUGIN_WS_ENCODINGS = {"LINEAR16"}
_SUPPORTED_DOMAINS = {"CALL", "MEETING"}


@dataclass
class _STTOptions:
    model_name: str = "sommers_ko"  # sommers_ko: "ko", sommers_ja: "ja", "sommers_en": "en"
    language: Language = Language("ko")  # ko, ja, en
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


class _StreamState(Enum):
    """State machine for SpeechStream lifecycle."""

    IDLE = auto()  # No active speech, no WS connection
    ACTIVE = auto()  # Speech active, WS connected
    CLOSING = auto()  # Sending EOS, waiting for recv completion
    CLOSED = auto()  # Stream fully closed


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

        if sample_rate < 8000 or sample_rate > 48000:
            raise ValueError("RTZR sample_rate must be between 8000 and 48000 Hz")
        if normalized_encoding not in _SUPPORTED_PLUGIN_WS_ENCODINGS:
            raise ValueError(
                f"Requested encoding {normalized_encoding} is not supported. Use LINEAR16."
            )
        if normalized_domain not in _SUPPORTED_DOMAINS:
            raise ValueError(f"RTZR domain must be one of {', '.join(sorted(_SUPPORTED_DOMAINS))}")

        self._params = _STTOptions(
            model_name=model,
            language=Language(language),
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
        if keywords and model != "sommers_ko":
            logger.warning("RTZR keyword boosting is only supported with sommers_ko model")
        self._client = RTZROpenAPIClient(http_session=http_session)

    @property
    def model(self) -> str:
        return self._params.model_name

    @property
    def provider(self) -> str:
        return "RTZR"

    async def aclose(self) -> None:
        """Close the RTZR client and cleanup resources."""
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
        return SpeechStream(
            stt=self,
            conn_options=conn_options,
        )


class SpeechStream(stt.SpeechStream):
    def __init__(self, *, stt: STT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._params.sample_rate)
        self._rtzr_stt: STT = stt
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._state = _StreamState.IDLE
        self._connection_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._pending_usage_audio_duration = 0.0
        self._idle_timeout = _IDLE_TIMEOUT_SECONDS
        self._last_audio_at: float | None = None
        self._idle_task: asyncio.Task[None] | None = None

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        config = self._rtzr_stt._client.build_config(
            model_name=self._rtzr_stt._params.model_name,
            domain=self._rtzr_stt._params.domain,
            sample_rate=self._rtzr_stt._params.sample_rate,
            encoding=self._rtzr_stt._params.encoding,
            epd_time=self._rtzr_stt._params.epd_time,
            noise_threshold=self._rtzr_stt._params.noise_threshold,
            active_threshold=self._rtzr_stt._params.active_threshold,
            use_itn=self._rtzr_stt._params.use_itn,
            use_disfluency_filter=self._rtzr_stt._params.use_disfluency_filter,
            use_profanity_filter=self._rtzr_stt._params.use_profanity_filter,
            use_punctuation=self._rtzr_stt._params.use_punctuation,
            keywords=self._rtzr_stt._params.keywords,
            language=self._rtzr_stt._params.language,
        )

        try:
            ws = await asyncio.wait_for(
                self._rtzr_stt._client.connect_websocket(config),
                timeout=self._conn_options.timeout,
            )
            logger.debug(
                "RTZR STT WS connected (model=%s, sr=%s, enc=%s, domain=%s, epd=%.2fs, "
                "noise=%.2f, active=%.2f, itn=%s, disfluency=%s, profanity=%s, punct=%s)",
                self._rtzr_stt._params.model_name,
                self._rtzr_stt._params.sample_rate,
                self._rtzr_stt._params.encoding,
                self._rtzr_stt._params.domain,
                self._rtzr_stt._params.epd_time,
                self._rtzr_stt._params.noise_threshold,
                self._rtzr_stt._params.active_threshold,
                self._rtzr_stt._params.use_itn,
                self._rtzr_stt._params.use_disfluency_filter,
                self._rtzr_stt._params.use_profanity_filter,
                self._rtzr_stt._params.use_punctuation,
            )
            return ws
        except asyncio.TimeoutError as e:
            raise APITimeoutError("WebSocket connection timeout") from e
        except RTZRStatusError as e:
            logger.error("RTZR API status error: %s", e)
            raise APIStatusError(
                message=e.message,
                status_code=e.status_code or 500,
                request_id=None,
                body=None,
            ) from e
        except RTZRConnectionError as e:
            logger.error("RTZR API connection error: %s", e)
            raise APIConnectionError("RTZR API connection failed") from e

    async def _run(self) -> None:
        # Retry runs should always start from a clean baseline.
        self._state = _StreamState.IDLE
        self._last_audio_at = None
        self._pending_usage_audio_duration = 0.0

        send_task = asyncio.create_task(self._send_audio_task(), name="RTZR.send_audio")
        self._idle_task = asyncio.create_task(self._idle_watchdog(), name="RTZR.idle_watchdog")

        try:
            await send_task
        except asyncio.CancelledError:
            pass
        finally:
            if not send_task.done():
                await utils.aio.gracefully_cancel(send_task)

            idle_task = self._idle_task
            self._idle_task = None
            if idle_task and not idle_task.done():
                await utils.aio.gracefully_cancel(idle_task)

            # Prefer graceful RTZR-side session close with EOS when a connection exists.
            if self._ws:
                await self._end_segment()

            await self._await_recv_completion()
            await self._cleanup_connection()
            self._emit_usage_event_if_needed()
            self._state = _StreamState.CLOSED
            self._last_audio_at = None

    async def _ensure_connected(self) -> None:
        """Lazy connect on first audio."""
        async with self._connection_lock:
            if self._ws is not None:
                return
            try:
                ws = await self._connect_ws()
                self._ws = ws
                self._state = _StreamState.ACTIVE
                self._recv_task = asyncio.create_task(self._recv_loop(ws), name="RTZR.recv_loop")
                self._last_audio_at = time.monotonic()
            except Exception:
                self._state = _StreamState.IDLE
                raise

    async def _end_segment(self) -> None:
        """Close current segment, prepare for next."""
        ws_to_close = None
        recv_to_wait = None

        async with self._connection_lock:
            if not self._ws or self._state == _StreamState.CLOSING:
                return
            self._state = _StreamState.CLOSING
            ws_to_close = self._ws
            recv_to_wait = self._recv_task
            self._ws = None
            self._recv_task = None

        try:
            if ws_to_close:
                try:
                    async with self._send_lock:
                        await ws_to_close.send_str("EOS")
                    logger.debug("Sent EOS to close audio segment")
                except Exception:
                    logger.exception("Failed to send EOS")

            if recv_to_wait:
                try:
                    await asyncio.wait_for(recv_to_wait, timeout=_RECV_COMPLETION_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning("recv completion timed out; cancelling")
                    await utils.aio.gracefully_cancel(recv_to_wait)
                except Exception:
                    logger.exception("recv task raised unexpected error")
                    await utils.aio.gracefully_cancel(recv_to_wait)
        finally:
            if ws_to_close:
                try:
                    await ws_to_close.close()
                except Exception:
                    pass

                self._emit_usage_event_if_needed()

            async with self._connection_lock:
                # Only reset to IDLE if we haven't started a new active connection
                # while waiting for the old one to close
                if self._state == _StreamState.CLOSING:
                    self._state = _StreamState.IDLE
                    self._last_audio_at = None

    async def _idle_watchdog(self) -> None:
        try:
            while self._state != _StreamState.CLOSED:
                await asyncio.sleep(_IDLE_CHECK_INTERVAL)
                if self._state != _StreamState.ACTIVE or not self._ws:
                    continue
                if self._last_audio_at is None:
                    continue
                if time.monotonic() - self._last_audio_at >= self._idle_timeout:
                    logger.info(
                        "RTZR STT idle timeout reached (%.0fs); closing segment",
                        self._idle_timeout,
                    )
                    await self._end_segment()
        except asyncio.CancelledError:
            pass

    async def _cleanup_connection(self) -> None:
        if self._ws:
            ws_to_close = self._ws
            try:
                await ws_to_close.close()
            except Exception:
                logger.exception("Failed to close RTZR websocket during cleanup")
            finally:
                self._ws = None

            self._emit_usage_event_if_needed()

    async def _await_recv_completion(self) -> None:
        recv_task = self._recv_task
        self._recv_task = None
        if not recv_task:
            return

        try:
            await asyncio.wait_for(recv_task, timeout=_RECV_COMPLETION_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("recv completion timed out; cancelling")
            await utils.aio.gracefully_cancel(recv_task)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("recv task raised unexpected error")
            await utils.aio.gracefully_cancel(recv_task)

    def _accumulate_usage(self, duration: float) -> None:
        self._pending_usage_audio_duration += duration

    def _emit_usage_event_if_needed(self) -> None:
        if self._pending_usage_audio_duration <= 0.0:
            return

        usage_event = stt.SpeechEvent(
            type=stt.SpeechEventType.RECOGNITION_USAGE,
            alternatives=[],
            recognition_usage=stt.RecognitionUsage(
                audio_duration=self._pending_usage_audio_duration
            ),
        )
        self._event_ch.send_nowait(usage_event)
        self._pending_usage_audio_duration = 0.0

    @utils.log_exceptions(logger=logger)
    async def _send_audio_task(self) -> None:
        sample_rate = self._rtzr_stt._params.sample_rate

        # Start with _INITIAL_CHUNK_MS chunks to meet RTZR minimum initial payload requirement
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=sample_rate * _INITIAL_CHUNK_MS // 1000,
        )
        switched_to_streaming = False

        has_ended = False
        async for data in self._input_ch:
            frames: list[rtc.AudioFrame] = []

            if isinstance(data, rtc.AudioFrame):
                frames.extend(audio_bstream.write(data.data.tobytes()))
            elif isinstance(data, self._FlushSentinel):
                frames.extend(audio_bstream.flush())
                has_ended = True

            if frames and not self._ws:
                await self._ensure_connected()
                # After initial _INITIAL_CHUNK_MS chunk sent, switch to _STREAMING_CHUNK_MS for low-latency streaming
                if not switched_to_streaming:
                    audio_bstream = utils.audio.AudioByteStream(
                        sample_rate=sample_rate,
                        num_channels=1,
                        samples_per_channel=sample_rate * _STREAMING_CHUNK_MS // 1000,
                    )
                    switched_to_streaming = True

            if frames:
                try:
                    async with self._send_lock:
                        for frame in frames:
                            if self._ws and self._state == _StreamState.ACTIVE:
                                ws = self._ws
                                await ws.send_bytes(frame.data.tobytes())
                                self._accumulate_usage(frame.duration)
                                self._last_audio_at = time.monotonic()
                except Exception:
                    logger.warning("WS send failed; breaking segment to attempt recovery")
                    has_ended = True

            if has_ended:
                if self._ws:
                    await self._end_segment()
                has_ended = False
                # Reset to _INITIAL_CHUNK_MS for the next segment's first frame
                audio_bstream = utils.audio.AudioByteStream(
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=sample_rate * _INITIAL_CHUNK_MS // 1000,
                )
                switched_to_streaming = False

        # Final shutdown — reuse _end_segment for lock consistency
        await self._end_segment()

    def _parse_words(self, words: list[dict]) -> list[TimedString]:
        """Parse word timing data from RTZR response."""
        return [
            TimedString(
                text=w.get("text", ""),
                start_time=w.get("start_at", 0) / 1000.0 + self.start_time_offset,
                end_time=(w.get("start_at", 0) + w.get("duration", 0)) / 1000.0
                + self.start_time_offset,
            )
            for w in words
        ]

    def _check_error_response(self, data: dict) -> None:
        """Check for error in RTZR response and raise if found."""
        if "error" in data:
            raise APIStatusError(
                message=f"Server error: {data['error']}",
                status_code=500,
                request_id=None,
                body=None,
            )
        if data.get("type") == "error" and "message" in data:
            raise APIStatusError(
                message=f"Server error: {data['message']}",
                status_code=500,
                request_id=None,
                body=None,
            )

    def _process_transcript_event(
        self,
        data: dict,
        in_speech: bool,
        speech_started_at: float | None,
    ) -> tuple[list[stt.SpeechEvent], bool, float | None]:
        """Parse RTZR response into SpeechEvents.

        Returns: (events, in_speech, speech_started_at)
        """
        start_time = data.get("start_at", 0) / 1000.0
        duration = data.get("duration", 0) / 1000.0
        words = data.get("words", [])

        if "alternatives" not in data or not data["alternatives"]:
            return [], in_speech, speech_started_at

        text = data["alternatives"][0].get("text", "")
        is_final = bool(data.get("final", False))

        if not text:
            return [], in_speech, speech_started_at

        events: list[stt.SpeechEvent] = []

        if not in_speech:
            in_speech = True
            speech_started_at = time.monotonic()
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
                        language=self._rtzr_stt._params.language,
                        start_time=start_time + self.start_time_offset,
                        end_time=start_time + duration + self.start_time_offset,
                        words=self._parse_words(words) if words else None,
                    )
                ],
            )
        )

        if is_final:
            speech_duration = (
                time.monotonic() - speech_started_at if speech_started_at is not None else 0.0
            )
            logger.debug("RTZR final transcript received (speech_duration=%.2fs)", speech_duration)
            events.append(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
            in_speech = False
            speech_started_at = None

        return events, in_speech, speech_started_at

    @utils.log_exceptions(logger=logger)
    async def _recv_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        in_speech = False
        speech_started_at: float | None = None

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        logger.warning("Non-JSON text from RTZR STT: %s", msg.data)
                        continue

                    self._check_error_response(data)

                    events, in_speech, speech_started_at = self._process_transcript_event(
                        data, in_speech, speech_started_at
                    )
                    for event in events:
                        self._event_ch.send_nowait(event)

                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("WebSocket error: %s", ws.exception())
                    raise APIConnectionError("WebSocket error occurred")
                else:
                    logger.debug("Ignored WebSocket message type: %s", msg.type)
        finally:
            # Defense against premature / server-initiated connection close
            # Ensure the stream isn't left stuck in ACTIVE with a dead socket
            async with self._connection_lock:
                if self._ws is ws:
                    logger.debug("RTZR recv loop closed unexpectedly, resetting connection")
                    self._ws = None
                    self._recv_task = None
                    self._state = _StreamState.IDLE
                    self._last_audio_at = None
