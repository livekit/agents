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
import time
import weakref
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Generic, TypeVar
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
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TimedString
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger
from .models import STTEncoding, STTModels
from .version import __version__

NUM_CHANNELS = 1
# Base URL for the Smallest AI API.
# Streaming: wss://api.smallest.ai/waves/v1/{model}/get_text
# Batch:     https://api.smallest.ai/waves/v1/{model}/get_text
SMALLEST_STT_BASE_URL = "https://api.smallest.ai/waves/v1"

# ---------------------------------------------------------------------------
# Minimal PeriodicCollector — same logic as livekit-plugins-deepgram/_utils.py
# ---------------------------------------------------------------------------

T = TypeVar("T")


class _PeriodicCollector(Generic[T]):
    def __init__(self, callback: Callable[[T], None], *, duration: float) -> None:
        self._duration = duration
        self._callback = callback
        self._last_flush_time = time.monotonic()
        self._total: T | None = None

    def push(self, value: T) -> None:
        if self._total is None:
            self._total = value
        else:
            self._total += value  # type: ignore[operator]
        if time.monotonic() - self._last_flush_time >= self._duration:
            self.flush()

    def flush(self) -> None:
        if self._total is not None:
            self._callback(self._total)
            self._total = None
        self._last_flush_time = time.monotonic()


# ---------------------------------------------------------------------------


@dataclass
class _STTOptions:
    model: STTModels | str
    api_key: str
    language: str  # BCP-47 code, e.g. "en", "hi"; use "multi" for auto-detection
    sample_rate: int
    encoding: STTEncoding | str
    word_timestamps: bool
    diarize: bool
    eou_timeout_ms: int  # end-of-utterance silence timeout (100–10 000 ms)
    base_url: str


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: STTModels | str = "pulse",
        language: str = "en",
        sample_rate: int = 16000,
        encoding: STTEncoding | str = "linear16",
        word_timestamps: bool = True,
        diarize: bool = False,
        eou_timeout_ms: int = 0,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = SMALLEST_STT_BASE_URL,
    ) -> None:
        """Create a new instance of Smallest AI Pulse STT.

        Args:
            model: STT model to use. Currently only "pulse" is available.
            language: BCP-47 language code (e.g. "en", "hi", "fr"). Use "multi"
                for automatic language detection across 39 supported languages.
            sample_rate: Audio sample rate in Hz. Supported: 8000, 16000, 22050,
                24000, 44100, 48000. Defaults to 16000.
            encoding: PCM encoding of the audio stream. Use "linear16" for raw
                16-bit PCM (the default and most compatible choice for streaming).
            word_timestamps: Include per-word start/end timestamps and confidence
                scores in transcripts. Defaults to True.
            diarize: Enable speaker diarization. When True, each word includes a
                speaker ID (integer during streaming). Defaults to False.
            eou_timeout_ms: Milliseconds of silence before the server considers an
                utterance complete and emits a final transcript. Set to 0 to disable
                server-side end-of-utterance detection, which is recommended when using
                LiveKit's built-in turn detection to minimise latency. Defaults to 0.
            api_key: Smallest AI API key. Falls back to the SMALLEST_API_KEY
                environment variable if not provided.
            http_session: An existing aiohttp ClientSession to reuse.
            base_url: Override the default API base URL.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=diarize,
                aligned_transcript="word" if word_timestamps else False,
            )
        )

        api_key = api_key or os.environ.get("SMALLEST_API_KEY")
        if not api_key:
            raise ValueError(
                "Smallest AI API key is required, either as argument or set "
                "SMALLEST_API_KEY environment variable"
            )

        self._opts = _STTOptions(
            model=model,
            api_key=api_key,
            language=language,
            sample_rate=sample_rate,
            encoding=encoding,
            word_timestamps=word_timestamps,
            diarize=diarize,
            eou_timeout_ms=eou_timeout_ms,
            base_url=base_url,
        )
        self._session = http_session
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()

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
        params: dict[str, Any] = {
            "language": config.language,
            "encoding": config.encoding,
            "sample_rate": config.sample_rate,
            "word_timestamps": str(config.word_timestamps).lower(),
            "diarize": str(config.diarize).lower(),
        }

        try:
            async with self._ensure_session().post(
                url=f"{config.base_url}/{config.model}/get_text",
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/octet-stream",
                    "X-Source": "livekit",
                    "X-LiveKit-Version": __version__,
                },
                params=params,
                # to_wav_bytes() produces a valid WAV file; the server auto-detects format.
                data=rtc.combine_audio_frames(buffer).to_wav_bytes(),
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=conn_options.timeout,
                ),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return _batch_transcription_to_speech_event(config.language, data)

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
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
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        encoding: NotGivenOr[STTEncoding | str] = NOT_GIVEN,
        eou_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """Update STT options; propagates to all active streams (triggers reconnect)."""
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(encoding):
            self._opts.encoding = encoding
        if is_given(eou_timeout_ms):
            self._opts.eou_timeout_ms = eou_timeout_ms

        for stream in self._streams:
            stream.update_options(
                model=model,
                language=language,
                sample_rate=sample_rate,
                encoding=encoding,
                eou_timeout_ms=eou_timeout_ms,
            )

    def _sanitize_options(self, *, language: NotGivenOr[str] = NOT_GIVEN) -> _STTOptions:
        config = replace(self._opts)
        if is_given(language):
            config.language = language
        return config


class SpeechStream(stt.SpeechStream):
    # Signals end of stream: server flushes remaining audio, emits final transcripts,
    # and responds with is_last=True before closing the session.
    # Use {"type": "finalize"} mid-session to force is_final without closing.
    _CLOSE_STREAM_MSG: str = json.dumps({"type": "close_stream"})

    def __init__(
        self,
        *,
        stt: STT,
        opts: _STTOptions,
        conn_options: APIConnectOptions,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._session = http_session
        self._speaking = False
        self._session_id = ""
        self._reconnect_event = asyncio.Event()
        self._audio_duration_collector = _PeriodicCollector(
            callback=self._on_audio_duration_report,
            duration=5.0,
        )

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        encoding: NotGivenOr[STTEncoding | str] = NOT_GIVEN,
        eou_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(encoding):
            self._opts.encoding = encoding
        if is_given(eou_timeout_ms):
            self._opts.eou_timeout_ms = eou_timeout_ms
        self._reconnect_event.set()

    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            # Send audio in 50ms chunks; matches the 50–100ms guidance from Smallest AI docs.
            samples_per_chunk = self._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=NUM_CHANNELS,
                samples_per_channel=samples_per_chunk,
            )

            async for data in self._input_ch:
                if isinstance(data, rtc.AudioFrame):
                    for frame in audio_bstream.write(data.data.tobytes()):
                        self._audio_duration_collector.push(frame.duration)
                        await ws.send_bytes(frame.data.tobytes())
                elif isinstance(data, self._FlushSentinel):
                    # User paused: drain the accumulator so the server gets all buffered
                    # audio. The server's eou_timeout_ms will then detect the silence and
                    # emit a final transcript — no explicit flush message is needed.
                    for frame in audio_bstream.flush():
                        self._audio_duration_collector.push(frame.duration)
                        await ws.send_bytes(frame.data.tobytes())
                    self._audio_duration_collector.flush()

            # Input channel closed: close the stream so the server flushes remaining
            # audio, emits final transcripts, and sends is_last=True.
            closing_ws = True
            await ws.send_str(SpeechStream._CLOSE_STREAM_MSG)

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
                        message="Smallest AI STT connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Smallest AI STT message type: %s", msg.type)
                    continue

                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning("failed to parse Smallest AI STT message: %s", msg.data)
                    continue

                self._process_stream_event(data)

                # Server confirms the session is fully flushed; recv loop can exit.
                if data.get("is_last"):
                    return

        ws: aiohttp.ClientWebSocketResponse | None = None
        while True:
            try:
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
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
                    tasks_group.exception()
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        params: dict[str, Any] = {
            "language": self._opts.language,
            "encoding": self._opts.encoding,
            "sample_rate": self._opts.sample_rate,
            "word_timestamps": str(self._opts.word_timestamps).lower(),
            "diarize": str(self._opts.diarize).lower(),
        }
        # Only send eou_timeout_ms when explicitly set (non-zero).
        # When 0, omit the parameter and let the server use its default,
        # which avoids adding server-side silence latency on top of LiveKit's
        # own end-of-turn detection.
        if self._opts.eou_timeout_ms > 0:
            params["eou_timeout_ms"] = self._opts.eou_timeout_ms
        ws_url = (
            self._opts.base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
            + f"/{self._opts.model}/get_text"
            + f"?{urlencode(params)}"
        )

        t0 = time.perf_counter()
        try:
            # heartbeat sends standard WebSocket ping frames every 5s, which is sufficient
            # to keep the Smallest AI connection alive without a custom JSON message.
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    ws_url,
                    headers={
                        "Authorization": f"Bearer {self._opts.api_key}",
                        "X-Source": "livekit",
                        "X-LiveKit-Version": __version__,
                    },
                    heartbeat=5.0,
                ),
                self._conn_options.timeout,
            )
            self._report_connection_acquired(time.perf_counter() - t0, False)
            logger.debug("established Smallest AI STT WebSocket connection")
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to Smallest AI STT") from e
        return ws

    def _on_audio_duration_report(self, duration: float) -> None:
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._session_id,
                alternatives=[],
                recognition_usage=stt.RecognitionUsage(audio_duration=duration),
            )
        )

    def _process_stream_event(self, data: dict[str, Any]) -> None:
        # Streaming WebSocket response schema (Smallest AI Pulse API):
        # {
        #   "session_id":   str,
        #   "transcript":   str,        # partial or final text for this utterance
        #   "is_final":     bool,       # True when the utterance is complete
        #   "is_last":      bool,       # True when the session itself is done (after close_stream)
        #   "language":     str,        # present when is_final=True (detected or echoed)
        #   "words":        [           # present when word_timestamps=True
        #     {"word": str, "start": float, "end": float,
        #      "confidence": float, "speaker": int}  # speaker only when diarize=True
        #   ]
        # }
        session_id = data.get("session_id", "")
        if session_id:
            self._session_id = session_id

        transcript = data.get("transcript", "")
        is_final = data.get("is_final", False)

        if not transcript:
            return

        # Infer START_OF_SPEECH — the Pulse API does not emit a dedicated speech-start event.
        if not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

        alts = _transcript_to_speech_data(
            language=self._opts.language,
            data=data,
            start_time_offset=self.start_time_offset,
            diarize=self._opts.diarize,
        )

        if is_final:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=self._session_id,
                    alternatives=alts,
                )
            )
            if self._speaking:
                self._speaking = False
                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
        else:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._session_id,
                    alternatives=alts,
                )
            )


def _transcript_to_speech_data(
    language: str,
    data: dict[str, Any],
    *,
    start_time_offset: float,
    diarize: bool,
) -> list[stt.SpeechData]:
    transcript = data.get("transcript", "")
    raw_words: list[dict[str, Any]] = data.get("words") or []

    words: list[TimedString] | None = (
        [
            TimedString(
                text=w.get("word", ""),
                start_time=w.get("start", 0.0) + start_time_offset,
                end_time=w.get("end", 0.0) + start_time_offset,
            )
            for w in raw_words
        ]
        if raw_words
        else None
    )

    start_time = raw_words[0].get("start", 0.0) + start_time_offset if raw_words else 0.0
    end_time = raw_words[-1].get("end", 0.0) + start_time_offset if raw_words else 0.0

    # Streaming diarization: per-word speaker IDs are integers (0, 1, …).
    # Pick the most frequent speaker across the utterance for top-level speaker_id.
    speaker_id: str | None = None
    if diarize and raw_words:
        speaker_counts = Counter(w["speaker"] for w in raw_words if "speaker" in w)
        if speaker_counts:
            speaker_id = f"S{speaker_counts.most_common(1)[0][0]}"

    # When language="multi", the server echoes the detected language in is_final responses.
    detected_language = data.get("language", language) or language

    return [
        stt.SpeechData(
            language=LanguageCode(detected_language),
            text=transcript,
            start_time=start_time,
            end_time=end_time,
            confidence=raw_words[0].get("confidence", 0.0) if raw_words else 0.0,
            words=words,
            speaker_id=speaker_id,
        )
    ]


def _batch_transcription_to_speech_event(
    language: str,
    data: dict[str, Any],
) -> stt.SpeechEvent:
    # Batch HTTP response schema (Smallest AI Pulse API):
    # {
    #   "status":       str,
    #   "transcription": str,
    #   "audio_length": str,   # duration in seconds as a string
    #   "words":        [{"word": str, "start": float, "end": float,
    #                     "confidence": float, "speaker": str}],
    #   "language":     str,
    #   "metadata":     {"filename": str, "duration": float, "fileSize": int}
    # }
    transcript = data.get("transcription", "")
    raw_words: list[dict[str, Any]] = data.get("words") or []
    detected_language = data.get("language", language) or language

    words: list[TimedString] | None = (
        [
            TimedString(
                text=w.get("word", ""),
                start_time=w.get("start", 0.0),
                end_time=w.get("end", 0.0),
            )
            for w in raw_words
        ]
        if raw_words
        else None
    )

    start_time = raw_words[0].get("start", 0.0) if raw_words else 0.0
    end_time = raw_words[-1].get("end", 0.0) if raw_words else 0.0

    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        request_id=utils.shortuuid(),
        alternatives=[
            stt.SpeechData(
                language=LanguageCode(detected_language),
                text=transcript,
                start_time=start_time,
                end_time=end_time,
                confidence=raw_words[0].get("confidence", 0.0) if raw_words else 0.0,
                words=words,
            )
        ],
    )
