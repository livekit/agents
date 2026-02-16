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
import dataclasses
import json
import os
import time
import weakref
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

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
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TimedString
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger
from .models import STTEncoding, STTModels, STTSampleRates

NUM_CHANNELS = 1
SMALLEST_STT_BASE_URL = "https://waves-api.smallest.ai/api/v1"
SMALLEST_STT_WS_BASE_URL = "wss://waves-api.smallest.ai/api/v1"


class _PeriodicDurationCollector:
    def __init__(self, *, duration: float, callback: Any) -> None:
        self._duration = duration
        self._callback = callback
        self._last_flush_time = time.monotonic()
        self._total = 0.0

    def push(self, duration: float) -> None:
        self._total += duration
        if time.monotonic() - self._last_flush_time >= self._duration:
            self.flush()

    def flush(self) -> None:
        if self._total > 0:
            self._callback(self._total)
            self._total = 0.0
        self._last_flush_time = time.monotonic()


@dataclass
class _STTOptions:
    model: STTModels | str
    language: str
    base_url: str
    ws_base_url: str
    sample_rate: STTSampleRates | int
    encoding: STTEncoding | str
    word_timestamps: bool
    diarize: bool


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: STTModels | str = "pulse",
        language: str = "en",
        base_url: str = SMALLEST_STT_BASE_URL,
        ws_base_url: str = SMALLEST_STT_WS_BASE_URL,
        sample_rate: STTSampleRates | int = 16000,
        encoding: STTEncoding | str = "linear16",
        word_timestamps: bool = True,
        diarize: bool = False,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Smallest AI Pulse STT.
        Args:
            api_key: Smallest AI API key. Can be set via argument or
                ``SMALLEST_API_KEY`` environment variable.
            model: The STT model to use. Currently ``"pulse"`` is supported.
            language: Language code for transcription (e.g. ``"en"``, ``"hi"``).
                Use ``"multi"`` for automatic language detection.
            base_url: Base URL for the Smallest AI HTTP API.
            ws_base_url: Base URL for the Smallest AI WebSocket API.
            sample_rate: Audio sample rate in Hz. Default is 16000.
            encoding: Audio encoding format for streaming. Default is ``"linear16"``.
            word_timestamps: Whether to include word-level timestamps in responses.
            diarize: Whether to enable speaker diarization.
            http_session: An existing aiohttp ClientSession to use. Optional.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=diarize,
                aligned_transcript="word" if word_timestamps else False,
            )
        )

        smallest_api_key = api_key if is_given(api_key) else os.environ.get("SMALLEST_API_KEY")
        if not smallest_api_key:
            raise ValueError(
                "Smallest.ai API key is required, either as argument or set"
                " SMALLEST_API_KEY environment variable"
            )

        self._api_key = smallest_api_key
        self._opts = _STTOptions(
            model=model,
            language=language,
            base_url=base_url,
            ws_base_url=ws_base_url,
            sample_rate=sample_rate,
            encoding=encoding,
            word_timestamps=word_timestamps,
            diarize=diarize,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "SmallestAI"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        config = self._sanitize_options(language=language)

        query = urlencode(
            {
                "language": config.language,
                "encoding": config.encoding,
                "sample_rate": config.sample_rate,
                "word_timestamps": str(config.word_timestamps).lower(),
                "diarize": str(config.diarize).lower(),
            }
        )
        url = f"{config.base_url}/{config.model}/get_text?{query}"

        try:
            async with self._ensure_session().post(
                url=url,
                data=rtc.combine_audio_frames(buffer).to_wav_bytes(),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                    "Content-Type": "audio/wav",
                },
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=conn_options.timeout,
                ),
            ) as response:
                response_json = await response.json(content_type=None)
                if response.status != 200:
                    raise APIStatusError(
                        message=response_json.get("message", "SmallestAI STT request failed"),
                        status_code=response.status,
                        request_id=response.headers.get("x-request-id"),
                        body=response_json,
                    )
                return _transcription_to_speech_event(
                    response_json,
                    default_language=config.language,
                    word_timestamps=config.word_timestamps,
                )
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except APIStatusError:
            raise
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = self._sanitize_options(language=language)
        _validate_stream_encoding(config.encoding)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            api_key=self._api_key,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[STTSampleRates | int] = NOT_GIVEN,
        encoding: NotGivenOr[STTEncoding | str] = NOT_GIVEN,
        word_timestamps: NotGivenOr[bool] = NOT_GIVEN,
        diarize: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._opts.language = language
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(encoding):
            self._opts.encoding = encoding
        if is_given(word_timestamps):
            self._opts.word_timestamps = word_timestamps
        if is_given(diarize):
            self._opts.diarize = diarize

        for stream in self._streams:
            stream.update_options(
                language=language,
                sample_rate=sample_rate,
                encoding=encoding,
                word_timestamps=word_timestamps,
                diarize=diarize,
            )

    def _sanitize_options(self, *, language: NotGivenOr[str] = NOT_GIVEN) -> _STTOptions:
        config = dataclasses.replace(self._opts)
        if is_given(language):
            config.language = language
        return config


class SpeechStream(stt.SpeechStream):
    """Streaming speech recognition using the Smallest AI Pulse WebSocket API"""

    _END_MSG = json.dumps({"type": "end"})

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        opts: _STTOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        _validate_stream_encoding(opts.encoding)
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._speaking = False
        self._request_id = ""
        self._reconnect_event = asyncio.Event()
        self._is_last_event = asyncio.Event()
        self._audio_duration_collector = _PeriodicDurationCollector(
            callback=self._on_audio_duration_report,
            duration=5.0,
        )

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[STTSampleRates | int] = NOT_GIVEN,
        encoding: NotGivenOr[STTEncoding | str] = NOT_GIVEN,
        word_timestamps: NotGivenOr[bool] = NOT_GIVEN,
        diarize: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._opts.language = language
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
            self._needed_sr = sample_rate
            self._resampler = None
        if is_given(encoding):
            _validate_stream_encoding(encoding)
            self._opts.encoding = encoding
        if is_given(word_timestamps):
            self._opts.word_timestamps = word_timestamps
        if is_given(diarize):
            self._opts.diarize = diarize
        self._reconnect_event.set()

    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            try:
                while True:
                    await ws.ping()
                    await asyncio.sleep(30)
            except Exception:
                return

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            # Pulse accepts binary audio chunks. We chunk close to ~4096 bytes.
            samples_per_channel = max(1, 4096 // (2 * NUM_CHANNELS))
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=NUM_CHANNELS,
                samples_per_channel=samples_per_channel,
            )

            has_flushed = False
            async for data in self._input_ch:
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(audio_bstream.write(data.data.tobytes()))
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())
                    has_flushed = True

                for frame in frames:
                    self._audio_duration_collector.push(frame.duration)
                    await ws.send_bytes(frame.data.tobytes())

                if has_flushed:
                    self._audio_duration_collector.flush()
                    await ws.send_str(SpeechStream._END_MSG)
                    has_flushed = False

            self._audio_duration_collector.flush()
            closing_ws = True
            await ws.send_str(SpeechStream._END_MSG)

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._session.closed:
                        return
                    raise APIStatusError(
                        message="SmallestAI STT connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected SmallestAI STT message type %s", msg.type)
                    continue

                try:
                    self._process_stream_event(json.loads(msg.data))
                except APIConnectionError:
                    raise
                except Exception:
                    logger.exception("failed to process SmallestAI STT message")

                if self._is_last_event.is_set():
                    closing_ws = True
                    return

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                self._is_last_event.clear()
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                    asyncio.create_task(keepalive_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    try:
                        tasks_group.exception()
                    except asyncio.CancelledError:
                        pass
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        query = urlencode(
            {
                "language": self._opts.language,
                "encoding": self._opts.encoding,
                "sample_rate": self._opts.sample_rate,
                "word_timestamps": str(self._opts.word_timestamps).lower(),
                "diarize": str(self._opts.diarize).lower(),
            }
        )
        ws_url = f"{self._opts.ws_base_url}/{self._opts.model}/get_text?{query}"
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    ws_url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                ),
                self._conn_options.timeout,
            )
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to smallestai") from e
        return ws

    def _on_audio_duration_report(self, duration: float) -> None:
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                alternatives=[],
                recognition_usage=stt.RecognitionUsage(audio_duration=duration),
            )
        )

    def _process_stream_event(self, data: dict[str, Any]) -> None:
        if data.get("type") == "error":
            raise APIConnectionError(data.get("message", "SmallestAI STT error"))

        if data.get("type") != "transcription":
            logger.debug("ignored unexpected SmallestAI STT event: %s", data)
            return

        transcript = str(data.get("transcription") or data.get("transcript") or data.get("text") or "")
        is_final = bool(data.get("is_final", False))
        is_last = bool(data.get("is_last", False))
        request_id = str(data.get("request_id", self._request_id))
        if request_id:
            self._request_id = request_id

        speech_data = _build_speech_data(
            transcript=transcript,
            language=str(data.get("language") or self._opts.language),
            words=data.get("words"),
            start_time_offset=self.start_time_offset,
            word_timestamps=self._opts.word_timestamps,
        )

        if transcript:
            if not self._speaking:
                self._speaking = True
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )

            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=(
                        stt.SpeechEventType.FINAL_TRANSCRIPT
                        if is_final
                        else stt.SpeechEventType.INTERIM_TRANSCRIPT
                    ),
                    request_id=self._request_id,
                    alternatives=[speech_data],
                )
            )

        if is_final and self._speaking:
            self._speaking = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

        if is_last:
            self._is_last_event.set()


def _build_speech_data(
    *,
    transcript: str,
    language: str,
    words: Any,
    start_time_offset: float,
    word_timestamps: bool,
) -> stt.SpeechData:
    typed_words: list[TimedString] | None = None
    start_time = start_time_offset
    end_time = start_time_offset
    first_word_start: float | None = None
    last_word_end: float | None = None

    if word_timestamps and isinstance(words, list) and words:
        typed_words = []
        for word in words:
            if not isinstance(word, dict):
                continue

            w_text = str(word.get("text") or word.get("word") or "")
            raw_start = _safe_float(word.get("start", word.get("start_time")), 0.0)
            raw_end = _safe_float(word.get("end", word.get("end_time")), raw_start)
            w_start = raw_start + start_time_offset
            w_end = raw_end + start_time_offset
            typed_words.append(
                TimedString(
                    text=w_text,
                    start_time=w_start,
                    end_time=w_end,
                    start_time_offset=start_time_offset,
                )
            )
            if first_word_start is None:
                first_word_start = w_start
            last_word_end = w_end

        if first_word_start is not None and last_word_end is not None:
            start_time = first_word_start
            end_time = last_word_end

    return stt.SpeechData(
        language=language,
        text=transcript,
        start_time=start_time,
        end_time=end_time,
        words=typed_words,
    )


def _safe_float(value: object, default: float) -> float:
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _validate_stream_encoding(encoding: STTEncoding | str) -> None:
    if encoding != "linear16":
        raise ValueError(
            "SmallestAI streaming STT currently supports only 'linear16' encoding. "
            "Transcoding for other encodings is not implemented."
        )


def _transcription_to_speech_event(
    data: dict[str, Any],
    *,
    default_language: str,
    word_timestamps: bool,
) -> stt.SpeechEvent:
    transcript = str(data.get("transcription") or "")
    request_id = str(data.get("request_id", ""))
    speech_data = _build_speech_data(
        transcript=transcript,
        language=str(data.get("language") or default_language),
        words=data.get("words"),
        start_time_offset=0.0,
        word_timestamps=word_timestamps,
    )
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        request_id=request_id,
        alternatives=[speech_data],
    )
