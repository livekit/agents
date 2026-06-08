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

from __future__ import annotations

import asyncio
import json
import os
import platform
import time
import weakref
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlencode

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    __version__ as livekit_version,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit.agents.utils.misc import is_given

from ._utils import PeriodicCollector
from .log import logger
from .stt import _looks_like_error_text

USER_AGENT = f"Livekit/{livekit_version} Python/{platform.python_version()}"

SARVAM_STT_REALTIME_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
REALTIME_MODEL = "saaras:v3-realtime"

RealtimeStreamType = Literal["fast", "balanced", "simulated"]
RealtimeEndpointing = Literal["vad", "manual"]
RealtimeEncoding = Literal["linear16"]
RealtimeMode = Literal["transcribe", "translate", "indic-en", "verbatim", "translit", "codemix"]

SUPPORTED_SAMPLE_RATES = {8000, 16000}
SUPPORTED_STREAM_TYPES = {"fast", "balanced", "simulated"}
SUPPORTED_ENDPOINTING = {"vad", "manual"}
SUPPORTED_ENCODINGS = {"linear16"}
SUPPORTED_MODES = {"transcribe", "translate", "indic-en", "verbatim", "translit", "codemix"}
STREAM_TYPE_CHUNK_MS = {"fast": 500, "balanced": 1000, "simulated": 1000}
EOS_FALLBACK_TIMEOUT = 2.0

SUPPORTED_LANGUAGES = {
    "en-IN",
    "hi-IN",
    "bn-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "or-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
    "gu-IN",
    "as-IN",
    "ur-IN",
    "ne-IN",
    "kok-IN",
    "ks-IN",
    "sd-IN",
    "sa-IN",
    "sat-IN",
    "mni-IN",
    "brx-IN",
    "mai-IN",
    "doi-IN",
    "auto",
}


@dataclass
class StreamingSTTOptions:
    language: str
    api_key: str
    stream_type: RealtimeStreamType | str = "balanced"
    mode: RealtimeMode | str = "transcribe"
    endpointing: RealtimeEndpointing | str = "vad"
    encoding: RealtimeEncoding | str = "linear16"
    sample_rate: int = 16000
    model: str = REALTIME_MODEL
    base_url: str = SARVAM_STT_REALTIME_URL
    vad_sot_threshold: float | None = None
    vad_min_speech_ms: int | None = None
    vad_min_silence_ms: int | None = None
    vad_smoothing_alpha: float | None = None

    def __post_init__(self) -> None:
        if self.model != REALTIME_MODEL:
            raise ValueError(f"model must be {REALTIME_MODEL}")
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"language {self.language} is not supported")
        if self.stream_type not in SUPPORTED_STREAM_TYPES:
            raise ValueError(
                f"stream_type must be one of {', '.join(sorted(SUPPORTED_STREAM_TYPES))}"
            )
        if self.language == "auto" and self.stream_type != "simulated":
            raise ValueError("language auto is only supported when stream_type is simulated")
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {', '.join(sorted(SUPPORTED_MODES))}")
        if self.endpointing not in SUPPORTED_ENDPOINTING:
            raise ValueError(
                f"endpointing must be one of {', '.join(sorted(SUPPORTED_ENDPOINTING))}"
            )
        if self.encoding not in SUPPORTED_ENCODINGS:
            raise ValueError(f"encoding must be one of {', '.join(sorted(SUPPORTED_ENCODINGS))}")
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {', '.join(str(r) for r in SUPPORTED_SAMPLE_RATES)}"
            )
        if self.vad_sot_threshold is not None and not 0.0 <= self.vad_sot_threshold <= 1.0:
            raise ValueError("vad_sot_threshold must be between 0.0 and 1.0")
        if self.vad_smoothing_alpha is not None and not 0.0 < self.vad_smoothing_alpha <= 1.0:
            raise ValueError("vad_smoothing_alpha must be greater than 0.0 and at most 1.0")
        if self.vad_min_speech_ms is not None and self.vad_min_speech_ms < 0:
            raise ValueError("vad_min_speech_ms must be greater than or equal to 0")
        if self.vad_min_silence_ms is not None and self.vad_min_silence_ms < 0:
            raise ValueError("vad_min_silence_ms must be greater than or equal to 0")


def _build_realtime_ws_url(base_url: str, opts: StreamingSTTOptions) -> str:
    params: dict[str, str] = {
        "language-code": opts.language,
        "stream-type": opts.stream_type,
        "endpointing": opts.endpointing,
        "encoding": opts.encoding,
        "sample_rate": str(opts.sample_rate),
        "model": opts.model,
    }

    if opts.stream_type == "simulated":
        params["mode"] = opts.mode

    if opts.endpointing == "vad":
        if opts.vad_sot_threshold is not None:
            params["vad_sot_threshold"] = str(opts.vad_sot_threshold)
        if opts.vad_min_speech_ms is not None:
            params["vad_min_speech_ms"] = str(opts.vad_min_speech_ms)
        if opts.vad_min_silence_ms is not None:
            params["vad_min_silence_ms"] = str(opts.vad_min_silence_ms)
        if opts.vad_smoothing_alpha is not None:
            params["vad_smoothing_alpha"] = str(opts.vad_smoothing_alpha)

    return f"{base_url}?{urlencode(params)}"


class STTStreaming(stt.STT):
    def __init__(
        self,
        *,
        language: str = "en-IN",
        stream_type: RealtimeStreamType | str = "balanced",
        mode: RealtimeMode | str = "transcribe",
        endpointing: RealtimeEndpointing | str = "vad",
        encoding: RealtimeEncoding | str = "linear16",
        sample_rate: int = 16000,
        api_key: str | None = None,
        base_url: str = SARVAM_STT_REALTIME_URL,
        http_session: aiohttp.ClientSession | None = None,
        vad_sot_threshold: float | None = None,
        vad_min_speech_ms: int | None = None,
        vad_min_silence_ms: int | None = None,
        vad_smoothing_alpha: float | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript=False,
                offline_recognize=False,
            )
        )

        api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not api_key:
            raise ValueError(
                "Sarvam API key is required. "
                "Provide it directly or set SARVAM_API_KEY environment variable."
            )

        self._opts = StreamingSTTOptions(
            language=language,
            api_key=api_key,
            stream_type=stream_type,
            mode=mode,
            endpointing=endpointing,
            encoding=encoding,
            sample_rate=sample_rate,
            base_url=base_url,
            vad_sot_threshold=vad_sot_threshold,
            vad_min_speech_ms=vad_min_speech_ms,
            vad_min_silence_ms=vad_min_silence_ms,
            vad_smoothing_alpha=vad_smoothing_alpha,
        )
        self._session = http_session
        self._owns_session = http_session is None
        self._streams = weakref.WeakSet[StreamingSpeechStream]()

    @property
    def model(self) -> str:
        return REALTIME_MODEL

    @property
    def provider(self) -> str:
        return "Sarvam"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            try:
                self._session = utils.http_context.http_session()
                self._owns_session = False
            except RuntimeError:
                self._session = aiohttp.ClientSession()
                self._owns_session = True
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        del buffer, language, conn_options
        raise NotImplementedError("Sarvam realtime STT only supports streaming")

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        stream_type: NotGivenOr[RealtimeStreamType | str] = NOT_GIVEN,
        mode: NotGivenOr[RealtimeMode | str] = NOT_GIVEN,
        endpointing: NotGivenOr[RealtimeEndpointing | str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        vad_sot_threshold: NotGivenOr[float | None] = NOT_GIVEN,
        vad_min_speech_ms: NotGivenOr[int | None] = NOT_GIVEN,
        vad_min_silence_ms: NotGivenOr[int | None] = NOT_GIVEN,
        vad_smoothing_alpha: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> None:
        opts = StreamingSTTOptions(
            language=language if is_given(language) else self._opts.language,
            api_key=self._opts.api_key,
            stream_type=stream_type if is_given(stream_type) else self._opts.stream_type,
            mode=mode if is_given(mode) else self._opts.mode,
            endpointing=endpointing if is_given(endpointing) else self._opts.endpointing,
            encoding=self._opts.encoding,
            sample_rate=sample_rate if is_given(sample_rate) else self._opts.sample_rate,
            base_url=self._opts.base_url,
            vad_sot_threshold=vad_sot_threshold
            if is_given(vad_sot_threshold)
            else self._opts.vad_sot_threshold,
            vad_min_speech_ms=vad_min_speech_ms
            if is_given(vad_min_speech_ms)
            else self._opts.vad_min_speech_ms,
            vad_min_silence_ms=vad_min_silence_ms
            if is_given(vad_min_silence_ms)
            else self._opts.vad_min_silence_ms,
            vad_smoothing_alpha=vad_smoothing_alpha
            if is_given(vad_smoothing_alpha)
            else self._opts.vad_smoothing_alpha,
        )
        self._opts = opts
        for stream in self._streams:
            stream.update_options(opts)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> StreamingSpeechStream:
        opts = StreamingSTTOptions(
            language=language if is_given(language) else self._opts.language,
            api_key=self._opts.api_key,
            stream_type=self._opts.stream_type,
            mode=self._opts.mode,
            endpointing=self._opts.endpointing,
            encoding=self._opts.encoding,
            sample_rate=self._opts.sample_rate,
            base_url=self._opts.base_url,
            vad_sot_threshold=self._opts.vad_sot_threshold,
            vad_min_speech_ms=self._opts.vad_min_speech_ms,
            vad_min_silence_ms=self._opts.vad_min_silence_ms,
            vad_smoothing_alpha=self._opts.vad_smoothing_alpha,
        )
        stream = StreamingSpeechStream(
            stt=self,
            opts=opts,
            conn_options=conn_options,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()


class StreamingSpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STTStreaming,
        opts: StreamingSTTOptions,
        conn_options: APIConnectOptions,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._session = http_session
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._request_id = ""
        self._session_id = ""
        self._session_ended = False
        self._reconnect_event = asyncio.Event()
        self._manual_speech_started = False
        self._pending_eos = False
        self._pending_eos_time: float | None = None
        self._pending_final_data: dict[str, Any] | None = None
        self._utterance_start_audio_pos = 0.0
        self._utterance_speech_end_audio_pos: float | None = None
        self._utterance_speech_end_wall: float | None = None
        self._final_received_for_utterance = False
        self._eos_emitted_for_utterance = False
        self._eos_fallback_task: asyncio.Task[None] | None = None
        self._stream_started_at = time.time()
        self._audio_position = 0.0
        self._total_reported_audio_duration = 0.0
        self._audio_duration_collector = PeriodicCollector(
            callback=self._on_audio_duration_report,
            duration=5.0,
        )
        self._logger = logger

    def update_options(self, opts: StreamingSTTOptions) -> None:
        self._opts = opts
        self._reconnect_event.set()

    def _build_log_context(self) -> dict[str, Any]:
        return {
            "request_id": self._request_id,
            "session_id": self._session_id,
            "model": self._opts.model,
            "language": self._opts.language,
            "stream_type": self._opts.stream_type,
            "endpointing": self._opts.endpointing,
        }

    @staticmethod
    def _extract_request_id(data: dict[str, Any]) -> str | None:
        request_id = data.get("request_id")
        if request_id is None:
            nested = data.get("data")
            if isinstance(nested, dict):
                request_id = nested.get("request_id")
            metadata = data.get("metadata")
            if request_id is None and isinstance(metadata, dict):
                request_id = metadata.get("request_id")
        if isinstance(request_id, str) and request_id:
            return request_id
        return None

    @staticmethod
    def _extract_session_id(data: dict[str, Any]) -> str | None:
        session_id = data.get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
        return None

    def _capture_server_ids(self, data: dict[str, Any]) -> None:
        session_id = self._extract_session_id(data)
        if session_id is not None:
            self._session_id = session_id

        if not self._request_id:
            request_id = self._extract_request_id(data)
            if request_id is not None:
                self._request_id = request_id

    async def aclose(self) -> None:
        try:
            self._cancel_eos_fallback()
            if self._ws and not self._ws.closed:
                await self._ws.close()
        finally:
            self._ws = None
            await super().aclose()

    async def _run(self) -> None:
        ws: aiohttp.ClientWebSocketResponse | None = None
        while True:
            try:
                ws = await self._connect_ws()
                self._ws = ws
                tasks = [
                    asyncio.create_task(self._process_audio(ws)),
                    asyncio.create_task(self._process_messages(ws)),
                ]
                reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                tasks_group = asyncio.gather(*tasks)

                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        if task is not reconnect_task:
                            task.result()

                    if reconnect_task not in done:
                        break
                    self._reconnect_event.clear()
                    self._reset_connection_state()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, reconnect_task)
                    tasks_group.cancel()
                    tasks_group.exception()
            except asyncio.TimeoutError as e:
                raise APITimeoutError("Timed out connecting to Sarvam realtime STT") from e
            except aiohttp.ClientResponseError as e:
                raise APIStatusError(
                    message=e.message,
                    status_code=e.status,
                    request_id=self._request_id or None,
                    body=e.message,
                ) from e
            except aiohttp.ClientConnectorError as e:
                raise APIConnectionError("failed to connect to Sarvam realtime STT") from e
            finally:
                if ws is not None:
                    await ws.close()
                self._ws = None

    def _cancel_eos_fallback(self) -> None:
        if self._eos_fallback_task and not self._eos_fallback_task.done():
            self._eos_fallback_task.cancel()
        self._eos_fallback_task = None

    def _reset_connection_state(self) -> None:
        self._cancel_eos_fallback()
        self._request_id = ""
        self._session_id = ""
        self._session_ended = False
        self._manual_speech_started = False
        self._pending_eos = False
        self._pending_eos_time = None
        self._pending_final_data = None
        self._utterance_start_audio_pos = 0.0
        self._utterance_speech_end_audio_pos = None
        self._utterance_speech_end_wall = None
        self._final_received_for_utterance = False
        self._eos_emitted_for_utterance = False
        self._audio_duration_collector.flush()
        self._total_reported_audio_duration = 0.0

    def _reset_utterance_state(self) -> None:
        self._cancel_eos_fallback()
        self._pending_eos = False
        self._pending_eos_time = None
        self._pending_final_data = None
        self._utterance_start_audio_pos = self._audio_position
        self._utterance_speech_end_audio_pos = None
        self._utterance_speech_end_wall = None
        self._final_received_for_utterance = False
        self._eos_emitted_for_utterance = False

    async def _safe_send_str(
        self,
        ws: Any,
        payload: dict[str, Any],
    ) -> None:
        if ws.closed:
            return

        try:
            await ws.send_str(json.dumps(payload))
        except (aiohttp.ClientConnectionResetError, ConnectionError):
            self._logger.debug(
                "Sarvam realtime STT WebSocket closed before send completed",
                extra={**self._build_log_context(), "payload": payload},
            )

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        ws_url = _build_realtime_ws_url(self._opts.base_url, self._opts)
        headers = {
            "API-SUBSCRIPTION-KEY": self._opts.api_key,
            "User-Agent": USER_AGENT,
        }
        self._logger.info(
            "Connecting to Sarvam realtime STT WebSocket", extra=self._build_log_context()
        )
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(ws_url, headers=headers, heartbeat=30.0),
                self._conn_options.timeout,
            )
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError):
            raise
        except aiohttp.ClientResponseError:
            raise
        except Exception as e:
            raise APIConnectionError("failed to connect to Sarvam realtime STT") from e
        self._logger.info(
            "Sarvam realtime STT WebSocket connected", extra=self._build_log_context()
        )
        return ws

    @utils.log_exceptions(logger=logger)
    async def _process_audio(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        cap_ms = STREAM_TYPE_CHUNK_MS[self._opts.stream_type]
        samples_per_channel = max(int(self._opts.sample_rate * cap_ms / 1000), 1)
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            samples_per_channel=samples_per_channel,
        )

        async for data in self._input_ch:
            frames: list[rtc.AudioFrame] = []
            if isinstance(data, rtc.AudioFrame):
                frames.extend(audio_bstream.write(data.data.tobytes()))
            elif isinstance(data, self._FlushSentinel):
                frames.extend(audio_bstream.flush())

            for frame in frames:
                if self._opts.endpointing == "manual" and not self._manual_speech_started:
                    await self._safe_send_str(ws, {"event": "speech_start"})
                    self._manual_speech_started = True

                self._audio_duration_collector.push(frame.duration)
                self._audio_position += frame.duration
                await ws.send_bytes(frame.data.tobytes())

            if isinstance(data, self._FlushSentinel):
                self._audio_duration_collector.flush()
                if self._opts.endpointing == "manual" and self._manual_speech_started:
                    await self._safe_send_str(ws, {"event": "speech_end"})
                    self._manual_speech_started = False

        self._audio_duration_collector.flush()
        await self._safe_send_str(ws, {"event": "end"})

    @utils.log_exceptions(logger=logger)
    async def _process_messages(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            msg = await ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    await self._handle_message(json.loads(msg.data))
                except json.JSONDecodeError as e:
                    if _looks_like_error_text(msg.data):
                        raise APIStatusError(
                            message=f"Sarvam realtime STT non-JSON error message: {msg.data}",
                            request_id=self._request_id or None,
                            body={"raw_message": msg.data},
                        ) from e
                    self._logger.warning(
                        "Invalid JSON received from Sarvam realtime STT",
                        extra={**self._build_log_context(), "raw_data": msg.data},
                    )
                    continue
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError(f"Sarvam realtime STT WebSocket error: {msg.data}")
            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                close_code = ws.close_code if ws.close_code is not None else msg.data
                close_reason = msg.extra
                if self._session_ended and close_code in (1000, 1001, None):
                    break
                if close_code in (1000, 1001, None) and not _looks_like_error_text(close_reason):
                    break
                raise self._status_error_from_close(close_code, close_reason)
            else:
                self._logger.debug(
                    "Unknown Sarvam realtime STT WebSocket message type",
                    extra={**self._build_log_context(), "message_type": str(msg.type)},
                )

    def _status_error_from_close(self, close_code: object, close_reason: object) -> APIStatusError:
        status_code = int(close_code) if isinstance(close_code, int) else -1
        retryable = close_code == 1013
        message = f"Sarvam realtime STT WebSocket closed unexpectedly: {close_reason}"
        if close_code == 1003:
            message = "Sarvam realtime STT authentication, quota, or rate limit error"
        elif close_code == 1008:
            message = "Sarvam realtime STT session timed out or exceeded the maximum duration"
        elif close_code == 1013:
            message = "Sarvam realtime STT backend temporarily unavailable"
        elif close_code == 4000:
            message = f"Sarvam realtime STT rejected the session: {close_reason}"

        return APIStatusError(
            message=message,
            status_code=status_code,
            request_id=self._request_id or None,
            body={
                "close_code": close_code,
                "close_reason": close_reason,
            },
            retryable=retryable,
        )

    async def _handle_message(self, data: dict[str, Any]) -> None:
        event = data.get("event")
        self._capture_server_ids(data)
        self._log_stt_event(event, data)
        if event == "session.begin":
            return
        elif event == "vad.speech_start":
            self._reset_utterance_state()
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.START_OF_SPEECH,
                    request_id=self._request_id,
                )
            )
        elif event == "vad.speech_end":
            self._handle_speech_end()
        elif event == "transcript.partial":
            self._send_transcript_event(stt.SpeechEventType.INTERIM_TRANSCRIPT, data)
        elif event == "transcript.final":
            if self._opts.endpointing == "vad":
                if self._is_valid_transcript(data):
                    self._pending_final_data = data
                    self._final_received_for_utterance = True
                    self._try_commit_utterance()
            elif self._send_transcript_event(stt.SpeechEventType.FINAL_TRANSCRIPT, data):
                self._final_received_for_utterance = True
        elif event == "session.end":
            self._handle_session_end(data)
        elif event == "error":
            self._handle_error_event(data)
        elif event == "pong":
            return
        else:
            self._logger.debug(
                "Unknown Sarvam realtime STT event",
                extra={**self._build_log_context(), "event": event, "data": data},
            )

    def _log_stt_event(self, event: object, data: dict[str, Any]) -> None:
        if event == "pong":
            return

        extra: dict[str, Any] = {
            **self._build_log_context(),
            "event": event,
            "utterance_idx": data.get("utterance_idx"),
            "raw_data": data,
        }
        if event in {"transcript.partial", "transcript.final"}:
            text = data.get("text")
            if isinstance(text, str):
                extra["text"] = text[:200]
                extra["text_length"] = len(text)
            extra["language"] = data.get("language") or self._opts.language
            extra["confidence"] = data.get("language_confidence", data.get("confidence"))
        elif event == "vad.speech_start":
            extra["audio_position"] = self._audio_position
        elif event == "vad.speech_end":
            extra["audio_position"] = self._audio_position
        elif event == "session.begin":
            pass
        elif event == "session.end":
            extra["audio_duration_s"] = data.get("audio_duration_s")
        else:
            return

        self._logger.info(f"Sarvam realtime STT {event}", extra=extra)

    def _is_valid_transcript(self, data: dict[str, Any]) -> bool:
        text = data.get("text")
        return isinstance(text, str) and bool(text)

    def _handle_speech_end(self) -> None:
        self._utterance_speech_end_audio_pos = self._audio_position
        self._utterance_speech_end_wall = time.time()

        if self._opts.endpointing != "vad":
            self._emit_end_of_speech()
            return

        if self._eos_emitted_for_utterance:
            return

        if self._final_received_for_utterance:
            self._try_commit_utterance()
            return

        self._pending_eos = True
        self._pending_eos_time = self._utterance_speech_end_wall
        if self._eos_fallback_task is None or self._eos_fallback_task.done():
            self._eos_fallback_task = asyncio.create_task(self._emit_pending_eos_after_timeout())

    async def _emit_pending_eos_after_timeout(
        self,
        timeout: float = EOS_FALLBACK_TIMEOUT,
    ) -> None:
        try:
            if timeout > 0:
                await asyncio.sleep(timeout)
            if self._pending_eos and not self._eos_emitted_for_utterance:
                self._emit_end_of_speech()
        except asyncio.CancelledError:
            raise

    def _try_commit_utterance(self) -> None:
        if (
            self._pending_final_data is None
            or self._utterance_speech_end_audio_pos is None
            or self._eos_emitted_for_utterance
        ):
            return

        committed_data = self._pending_final_data
        if self._send_transcript_event(
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            committed_data,
        ):
            self._logger.info(
                "Sarvam realtime STT utterance committed",
                extra={
                    **self._build_log_context(),
                    "end_time": self._utterance_speech_end_audio_pos,
                    "speech_end_wall_time": self._utterance_speech_end_wall,
                },
            )
            self._emit_end_of_speech()
            self._pending_final_data = None

    def _emit_end_of_speech(self) -> None:
        current_task = asyncio.current_task()
        fallback_task = self._eos_fallback_task
        self._eos_fallback_task = None
        if fallback_task and fallback_task is not current_task and not fallback_task.done():
            fallback_task.cancel()

        if self._eos_emitted_for_utterance:
            return

        alternatives: list[stt.SpeechData] = []
        if self._utterance_speech_end_audio_pos is not None:
            metadata: dict[str, Any] = {}
            if self._utterance_speech_end_wall is not None:
                metadata["speech_end_wall_time"] = self._utterance_speech_end_wall
            if self._pending_final_data is not None:
                utterance_idx = self._pending_final_data.get("utterance_idx")
                if utterance_idx is not None:
                    metadata["utterance_idx"] = utterance_idx
            alternatives.append(
                stt.SpeechData(
                    language=LanguageCode(self._opts.language),
                    text="",
                    end_time=self._utterance_speech_end_audio_pos,
                    metadata=metadata or None,
                )
            )

        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.END_OF_SPEECH,
                request_id=self._request_id,
                alternatives=alternatives,
            )
        )
        self._pending_eos = False
        self._pending_eos_time = None
        self._eos_emitted_for_utterance = True

    def _send_transcript_event(self, event_type: stt.SpeechEventType, data: dict[str, Any]) -> bool:
        text = data.get("text")
        if not isinstance(text, str) or not text:
            return False

        language = data.get("language") or self._opts.language
        confidence = data.get("language_confidence")
        if confidence is None:
            confidence = data.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            confidence = 0.0

        metadata = {
            key: data[key]
            for key in ("utterance_idx", "language_confidence")
            if key in data and data[key] is not None
        }
        if (
            event_type == stt.SpeechEventType.FINAL_TRANSCRIPT
            and self._utterance_speech_end_wall is not None
        ):
            metadata["speech_end_wall_time"] = self._utterance_speech_end_wall
        end_time = 0.0
        if (
            event_type == stt.SpeechEventType.FINAL_TRANSCRIPT
            and self._utterance_speech_end_audio_pos is not None
        ):
            end_time = self._utterance_speech_end_audio_pos
        elif event_type == stt.SpeechEventType.FINAL_TRANSCRIPT and self._audio_position > 0:
            end_time = self._audio_position

        speech_data = stt.SpeechData(
            language=LanguageCode(language),
            text=text,
            end_time=end_time,
            confidence=float(confidence),
            metadata=metadata or None,
        )
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=event_type,
                request_id=self._request_id,
                alternatives=[speech_data],
            )
        )
        return True

    def _handle_session_end(self, data: dict[str, Any]) -> None:
        self._capture_server_ids(data)
        audio_duration = data.get("audio_duration_s")
        if isinstance(audio_duration, (int, float)) and not isinstance(audio_duration, bool):
            delta = max(float(audio_duration) - self._total_reported_audio_duration, 0.0)
            if delta:
                self._emit_usage(delta)
        self._session_ended = True

    def _handle_error_event(self, data: dict[str, Any]) -> None:
        if not data.get("is_fatal", False):
            self._logger.warning(
                "Non-fatal Sarvam realtime STT error",
                extra={**self._build_log_context(), "error": data},
            )
            return

        code = data.get("code", "unknown")
        status_code = data.get("status_code", -1)
        if not isinstance(status_code, int):
            status_code = -1
        raise APIStatusError(
            message=f"Sarvam realtime STT error: {data.get('message', code)}",
            status_code=status_code,
            request_id=self._request_id or None,
            body=data,
            retryable=code == "model_unavailable",
        )

    def _on_audio_duration_report(self, duration: float) -> None:
        self._emit_usage(duration)

    def _emit_usage(self, duration: float) -> None:
        self._total_reported_audio_duration += duration
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                recognition_usage=stt.RecognitionUsage(audio_duration=duration),
            )
        )
