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

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
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

_DEFAULT_CHUNK_MS = 100
_IDLE_TIMEOUT_SECONDS = 25.0
_RECV_COMPLETION_TIMEOUT = 5.0


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
        epd_time: float = 0.8,
        noise_threshold: float = 0.60,
        active_threshold: float = 0.80,
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

        self._params = _STTOptions(
            model_name=model,
            language=LanguageCode(language),
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
        self._idle_timeout = _IDLE_TIMEOUT_SECONDS

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
        try:
            await self._send_audio_task()
        finally:
            await self._await_recv_completion()
            await self._cleanup_connection()

    async def _ensure_connected(self) -> None:
        """Lazy connect on first audio."""
        if self._ws is None:
            self._ws = await self._connect_ws()
            self._recv_task = asyncio.create_task(self._recv_loop(self._ws), name="RTZR.recv_loop")

    async def _end_segment(self) -> None:
        """Drain the current connection before reading audio for the next one."""
        if self._ws is None:
            return
        try:
            await self._ws.send_str("EOS")
        except Exception:
            logger.exception("Failed to send EOS")
        await self._await_recv_completion()
        await self._cleanup_connection()

    async def _cleanup_connection(self) -> None:
        if self._ws:
            try:
                await self._ws.close()
            finally:
                self._ws = None

    async def _await_recv_completion(self) -> None:
        if self._recv_task:
            try:
                await asyncio.wait_for(self._recv_task, timeout=_RECV_COMPLETION_TIMEOUT)
            except asyncio.TimeoutError:
                await utils.aio.gracefully_cancel(self._recv_task)
            finally:
                self._recv_task = None

    @utils.log_exceptions(logger=logger)
    async def _send_audio_task(self) -> None:
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._rtzr_stt._params.sample_rate,
            num_channels=1,
            samples_per_channel=self._rtzr_stt._params.sample_rate // (1000 // _DEFAULT_CHUNK_MS),
        )

        self._input_ch.restart_provider_clock()
        while True:
            try:
                data = await asyncio.wait_for(
                    self._input_ch.recv(),
                    timeout=self._idle_timeout if self._ws is not None else None,
                )
            except asyncio.TimeoutError:
                data = self._FlushSentinel()
            except utils.aio.ChanClosed:
                break

            if isinstance(data, rtc.AudioFrame):
                frames = audio_bstream.write(data.data.tobytes())
            else:
                frames = audio_bstream.flush()

            if frames:
                await self._ensure_connected()
                assert self._ws is not None
                for frame in frames:
                    await self._ws.send_bytes(frame.data.tobytes())

            if isinstance(data, self._FlushSentinel):
                await self._end_segment()
                self._input_ch.restart_provider_clock()

        frames = audio_bstream.flush()
        if frames:
            await self._ensure_connected()
            assert self._ws is not None
            for frame in frames:
                await self._ws.send_bytes(frame.data.tobytes())
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
            events.append(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH,
                    alternatives=events[-1].alternatives if duration > 0 else [],
                )
            )
            in_speech = False
            speech_started_at = None

        return events, in_speech, speech_started_at

    @utils.log_exceptions(logger=logger)
    async def _recv_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        in_speech = False
        speech_started_at: float | None = None

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON text from RTZR STT", extra={"lk.pii.data": msg.data})
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
