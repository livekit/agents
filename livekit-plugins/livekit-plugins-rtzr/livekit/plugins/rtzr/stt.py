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
from collections import deque
from dataclasses import dataclass

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
    vad as agents_vad,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.voice.io import TimedString

from .log import logger
from .rtzrapi import DEFAULT_SAMPLE_RATE, RTZRConnectionError, RTZROpenAPIClient, RTZRStatusError

_DEFAULT_CHUNK_MS = 100
_IDLE_TIMEOUT_SECONDS = 25.0


@dataclass
class _STTOptions:
    model_name: str = "sommers_ko"  # sommers_ko: "ko", sommers_ja: "ja"
    language: str = "ko"  # ko, ja, en
    sample_rate: int = DEFAULT_SAMPLE_RATE
    encoding: str = "LINEAR16"  # or "OGG_OPUS" in future
    domain: str = "CALL"  # CALL, MEETING
    epd_time: float = 0.3
    noise_threshold: float = 0.60
    active_threshold: float = 0.80
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
        domain: str = "CALL",
        epd_time: float = 1.5,
        noise_threshold: float = 0.60,
        active_threshold: float = 0.80,
        use_punctuation: bool = False,
        use_vad_event: bool = True,
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

        self._params = _STTOptions(
            model_name=model,
            language=language,
            sample_rate=sample_rate,
            domain=domain,
            epd_time=epd_time,
            noise_threshold=noise_threshold,
            active_threshold=active_threshold,
            use_punctuation=use_punctuation,
            keywords=keywords,
        )
        if keywords and model != "sommers_ko":
            logger.warning("RTZR keyword boosting is only supported with sommers_ko model")
        self._client = RTZROpenAPIClient(http_session=http_session)
        self._use_vad_event = use_vad_event
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()
        self._last_vad_speaking: bool | None = None
        self._vad_speech_total = 0.0
        self._vad_silence_total = 0.0
        self._use_vad_endpointing = False

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

    def on_vad_event(self, ev: agents_vad.VADEvent) -> None:  # pragma: no cover - logging only
        if not self._use_vad_event:
            return
        try:
            current: bool | None
            if ev.type == agents_vad.VADEventType.START_OF_SPEECH:
                current = True
            elif ev.type == agents_vad.VADEventType.END_OF_SPEECH:
                current = False
            else:
                current = ev.speaking

            if current is None:
                return

            speech_total = self._vad_speech_total
            silence_total = self._vad_silence_total
            state_initialized = self._last_vad_speaking is not None

            if current != self._last_vad_speaking:
                if state_initialized or current:
                    logger.info(
                        "LK VAD speaking=%s (speech=%.3fs silence=%.3fs)",
                        current,
                        speech_total,
                        silence_total,
                    )
                self._last_vad_speaking = current
                if current:
                    self._vad_silence_total = 0.0
                else:
                    self._vad_speech_total = 0.0

            if ev.speaking is True and ev.speech_duration > self._vad_speech_total:
                self._vad_speech_total = ev.speech_duration
            elif ev.speaking is False and ev.silence_duration > self._vad_silence_total:
                self._vad_silence_total = ev.silence_duration

            self._use_vad_endpointing = True

            for stream in list(self._streams):
                stream._handle_vad_event(ev)
        except Exception:
            logger.exception("Failed to process LiveKit VAD event")

    def _register_stream(self, stream: SpeechStream) -> None:
        self._streams.add(stream)

    def _unregister_stream(self, stream: SpeechStream) -> None:
        self._streams.discard(stream)


class SpeechStream(stt.SpeechStream):
    def __init__(self, *, stt: STT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._params.sample_rate)
        self._rtzr_stt: STT = stt
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._vad_event_queue: asyncio.Queue[agents_vad.VADEvent] = asyncio.Queue(maxsize=32)
        self._speech_active = False
        self._pending_speech_frames: deque[bytes] = deque()
        self._connection_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._closed = False
        self._idle_timeout = _IDLE_TIMEOUT_SECONDS
        self._last_audio_at: float | None = None
        self._idle_task: asyncio.Task[None] | None = None
        self._fallback_mode = not self._rtzr_stt._use_vad_endpointing

        self._rtzr_stt._register_stream(self)

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        config = self._rtzr_stt._client.build_config(
            model_name=self._rtzr_stt._params.model_name,
            domain=self._rtzr_stt._params.domain,
            sample_rate=self._rtzr_stt._params.sample_rate,
            encoding=self._rtzr_stt._params.encoding,
            epd_time=self._rtzr_stt._params.epd_time,
            noise_threshold=self._rtzr_stt._params.noise_threshold,
            active_threshold=self._rtzr_stt._params.active_threshold,
            use_punctuation=self._rtzr_stt._params.use_punctuation,
            keywords=self._rtzr_stt._params.keywords,
        )

        try:
            ws = await asyncio.wait_for(
                self._rtzr_stt._client.connect_websocket(config),
                timeout=self._conn_options.timeout,
            )
            logger.debug(
                "RTZR STT WS connected (model=%s, sr=%s, epd=%.2fs, "
                "noise=%.2f, active=%.2f, punct=%s)",
                self._rtzr_stt._params.model_name,
                self._rtzr_stt._params.sample_rate,
                self._rtzr_stt._params.epd_time,
                self._rtzr_stt._params.noise_threshold,
                self._rtzr_stt._params.active_threshold,
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
        send_task = asyncio.create_task(self._send_audio_task(), name="RTZR.send_audio")
        vad_task = asyncio.create_task(self._process_vad_events(), name="RTZR.vad_events")
        self._idle_task = asyncio.create_task(self._idle_watchdog(), name="RTZR.idle_watchdog")

        try:
            await asyncio.gather(send_task, vad_task)
        finally:
            self._closed = True
            if not send_task.done():
                await utils.aio.gracefully_cancel(send_task)
            if not vad_task.done():
                await utils.aio.gracefully_cancel(vad_task)
            if self._idle_task is not None and not self._idle_task.done():
                await utils.aio.gracefully_cancel(self._idle_task)
            await self._await_recv_completion()
            await self._cleanup_connection()
            self._rtzr_stt._unregister_stream(self)

    def _handle_vad_event(self, ev: agents_vad.VADEvent) -> None:
        if self._closed:
            return
        if not self._rtzr_stt._use_vad_event:
            return

        if ev.type == agents_vad.VADEventType.INFERENCE_DONE:
            if ev.speaking:
                self._last_audio_at = time.monotonic()
            return

        self._fallback_mode = False
        try:
            self._vad_event_queue.put_nowait(ev)
        except asyncio.QueueFull:
            logger.warning("Dropping VAD event due to backpressure")

    async def _process_vad_events(self) -> None:
        try:
            while True:
                ev = await self._vad_event_queue.get()
                if ev.type == agents_vad.VADEventType.START_OF_SPEECH:
                    await self._handle_vad_start(ev)
                elif ev.type == agents_vad.VADEventType.END_OF_SPEECH:
                    await self._handle_vad_end()
        except asyncio.CancelledError:  # pragma: no cover - task cancellation
            pass

    async def _handle_vad_start(self, ev: agents_vad.VADEvent) -> None:
        async with self._connection_lock:
            if self._closed:
                return

            self._speech_active = True
            self._last_audio_at = time.monotonic()
            self._pending_speech_frames.clear()

            await self._cleanup_connection()
            ws = await self._connect_ws()
            self._ws = ws
            self._recv_task = asyncio.create_task(self._recv_loop(ws), name="RTZR.recv_loop")

            for frame in ev.frames:
                payload = frame.data.tobytes()
                async with self._send_lock:
                    await ws.send_bytes(payload)
                self._last_audio_at = time.monotonic()

    async def _handle_vad_end(self) -> None:
        async with self._connection_lock:
            self._speech_active = False

            if not self._ws:
                self._last_audio_at = None
                self._pending_speech_frames.clear()
                self._fallback_mode = not self._rtzr_stt._use_vad_endpointing
                return

            await self._flush_pending_frames()

            try:
                await self._ws.send_str("EOS")
                logger.info("Sent EOS to close audio stream")
            except Exception as e:  # pragma: no cover - defensive logging
                logger.error("Failed to send EOS: %s", e)

            await self._await_recv_completion()
            await self._cleanup_connection()
            self._pending_speech_frames.clear()
            self._last_audio_at = None
            self._fallback_mode = not self._rtzr_stt._use_vad_endpointing

    async def _handle_idle_timeout(self) -> None:
        async with self._connection_lock:
            if not self._ws or not self._speech_active:
                return

            logger.info(
                "RTZR STT idle timeout reached (%.0fs); closing websocket",
                self._idle_timeout,
            )

            await self._flush_pending_frames()
            try:
                await self._ws.send_str("EOS")
            except Exception:
                logger.exception("Failed to send EOS during idle shutdown")

            await self._await_recv_completion()
            await self._cleanup_connection()

            self._speech_active = False
            self._pending_speech_frames.clear()
            self._last_audio_at = None
            self._fallback_mode = True

    async def _idle_watchdog(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(1.0)
                if not self._speech_active or not self._ws:
                    continue
                if self._last_audio_at is None:
                    continue
                if time.monotonic() - self._last_audio_at >= self._idle_timeout:
                    await self._handle_idle_timeout()
        except asyncio.CancelledError:  # pragma: no cover - task cancellation
            pass

    async def _cleanup_connection(self) -> None:
        if self._ws:
            try:
                await self._ws.close()
            finally:
                self._ws = None

    async def _await_recv_completion(self) -> None:
        if self._recv_task:
            try:
                await asyncio.wait_for(self._recv_task, timeout=5.0)
            except asyncio.TimeoutError:
                await utils.aio.gracefully_cancel(self._recv_task)
            finally:
                self._recv_task = None

    async def _flush_pending_frames(self) -> None:
        if not self._ws:
            return

        while self._pending_speech_frames:
            payload = self._pending_speech_frames.popleft()
            async with self._send_lock:
                await self._ws.send_bytes(payload)
            self._last_audio_at = time.monotonic()

    async def _emit_audio(self, payload: bytes) -> None:
        if (
            not self._rtzr_stt._use_vad_endpointing
            and not self._speech_active
            and not self._closed
        ):
            synthetic_ev = agents_vad.VADEvent(
                type=agents_vad.VADEventType.START_OF_SPEECH,
                samples_index=0,
                timestamp=time.time(),
                speech_duration=0.0,
                silence_duration=0.0,
            )
            await self._handle_vad_start(synthetic_ev)

        self._last_audio_at = time.monotonic()

        if self._ws:
            async with self._send_lock:
                await self._ws.send_bytes(payload)
        else:
            self._pending_speech_frames.append(payload)

    @utils.log_exceptions(logger=logger)
    async def _send_audio_task(self) -> None:
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._rtzr_stt._params.sample_rate,
            num_channels=1,
            samples_per_channel=self._rtzr_stt._params.sample_rate
            // (1000 // _DEFAULT_CHUNK_MS),
        )

        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                frames = audio_bstream.write(data.data.tobytes())
            elif isinstance(data, self._FlushSentinel):
                frames = audio_bstream.flush()
            else:
                frames = []

            for frame in frames:
                await self._emit_audio(frame.data.tobytes())

        await self._flush_pending_frames()

        if self._ws:
            try:
                await self._ws.send_str("EOS")
                logger.info("Sent EOS to close audio stream")
            except Exception:
                logger.exception("Failed to send EOS during shutdown")
            await self._await_recv_completion()

        self._speech_active = False
        self._last_audio_at = None
        self._pending_speech_frames.clear()

    @utils.log_exceptions(logger=logger)
    async def _recv_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        in_speech = False
        speech_started_at: float | None = None

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON text from RTZR STT: %s", msg.data)
                    continue

                # msec -> sec
                start_time = data.get("start_at", 0) / 1000.0
                duration = data.get("duration", 0) / 1000.0
                words = data.get("words", [])

                if "alternatives" in data and data["alternatives"]:
                    text = data["alternatives"][0].get("text", "")
                    is_final = bool(data.get("final", False))
                    if text:
                        if not in_speech:
                            in_speech = True
                            speech_started_at = time.monotonic()
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                            )

                        event_type = (
                            stt.SpeechEventType.FINAL_TRANSCRIPT
                            if is_final
                            else stt.SpeechEventType.INTERIM_TRANSCRIPT
                        )
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=event_type,
                                alternatives=[
                                    stt.SpeechData(
                                        text=text,
                                        language=self._rtzr_stt._params.language,
                                        start_time=start_time + self.start_time_offset,
                                        end_time=start_time + duration + self.start_time_offset,
                                        words=[
                                            TimedString(
                                                text=word.get("text", ""),
                                                start_time=word.get("start_at", 0) / 1000.0
                                                + self.start_time_offset,
                                                end_time=(
                                                    word.get("start_at", 0)
                                                    + word.get("duration", 0)
                                                )
                                                / 1000.0
                                                + self.start_time_offset,
                                            )
                                            for word in words
                                        ]
                                        if words
                                        else None,
                                    )
                                ],
                            )
                        )

                        if is_final:
                            duration = (
                                time.monotonic() - speech_started_at
                                if speech_started_at is not None
                                else 0.0
                            )
                            logger.info("VAD END_OF_SPEECH (speech_duration=%.2fs)", duration)
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                            )
                            in_speech = False
                            speech_started_at = None

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
