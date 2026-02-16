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
import os
import time
from dataclasses import dataclass, replace
from typing import Any

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString
from mistralai import Mistral
from mistralai.extra.realtime.connection import RealtimeConnection, UnknownRealtimeEvent
from mistralai.extra.realtime.transcription import RealtimeTranscription
from mistralai.models import AudioFormat
from mistralai.models.realtimetranscriptionerror import RealtimeTranscriptionError
from mistralai.models.realtimetranscriptionsessioncreated import (
    RealtimeTranscriptionSessionCreated,
)
from mistralai.models.realtimetranscriptionsessionupdated import (
    RealtimeTranscriptionSessionUpdated,
)
from mistralai.models.sdkerror import SDKError
from mistralai.models.transcriptionstreamdone import TranscriptionStreamDone
from mistralai.models.transcriptionstreamlanguage import TranscriptionStreamLanguage
from mistralai.models.transcriptionstreamsegmentdelta import (
    TranscriptionStreamSegmentDelta,
)
from mistralai.models.transcriptionstreamtextdelta import TranscriptionStreamTextDelta

from .log import logger
from .models import STTModels

# Voxtral recommended config (docs.mistral.ai/capabilities/audio_transcription):
#   encoding: pcm_s16le, 16kHz mono, 100ms chunks
#   model processes in 80ms token boundaries; 1 text token = 80ms audio
#   server-side transcription_delay_ms: 480ms recommended (range: 80-2400, multiples of 80)
#   not configurable from this plugin — uses Mistral API default
DEFAULT_SAMPLE_RATE = 16000
NUM_CHANNELS = 1
CHUNK_DURATION_MS = 100
DEFAULT_FINALIZE_DELAY_MS = (
    100  # client-side finalize delay, not the model's transcription_delay_ms
)
MIN_IDLE_FINALIZE_MS = 650


@dataclass
class _STTOptions:
    model: STTModels | str
    language: str | None
    sample_rate: int
    interim_results: bool
    finalize_delay_ms: int


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: str | None = "en",
        model: STTModels | str = "voxtral-mini-latest",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        client: Mistral | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        interim_results: bool = True,
        finalize_delay_ms: int = DEFAULT_FINALIZE_DELAY_MS,
    ):
        """
        Create a new instance of MistralAI STT.

        Args:
            language: The language code to use for transcription (e.g., "en" for English).
                Segment timestamps will only be available if set to None.
            model: The MistralAI model to use for transcription, default is voxtral-mini-latest.
            api_key: Your MistralAI API key. If not provided, will use the MISTRAL_API_KEY
                environment variable.
            client: Optional pre-configured MistralAI client instance.
            sample_rate: Audio sample rate in Hz (default 16000).
            interim_results: Whether to emit interim transcripts (default True).
            finalize_delay_ms: Delay after VAD end-of-speech before emitting FINAL_TRANSCRIPT.
                Lower values reduce latency but may truncate; higher values ensure completeness.
        """

        super().__init__(
            capabilities=self._build_capabilities(model=model, interim_results=interim_results)
        )
        self._opts = _STTOptions(
            language=language,
            model=model,
            sample_rate=sample_rate,
            interim_results=interim_results,
            finalize_delay_ms=finalize_delay_ms,
        )

        mistral_api_key = api_key if is_given(api_key) else os.environ.get("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise ValueError("MistralAI API key is required. Set MISTRAL_API_KEY or pass api_key")
        self._client = client or Mistral(api_key=mistral_api_key)

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "MistralAI"

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """
        Update the options for the STT.

        Args:
            model: The model to use for transcription.
            language: The language to transcribe in.
            sample_rate: Audio sample rate in Hz.
            interim_results: Whether to emit interim transcripts.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(interim_results):
            self._opts.interim_results = interim_results

        if is_given(model) or is_given(interim_results):
            self._capabilities = self._build_capabilities(
                model=self._opts.model,
                interim_results=self._opts.interim_results,
            )

    @staticmethod
    def _build_capabilities(
        *, model: STTModels | str, interim_results: bool
    ) -> stt.STTCapabilities:
        is_realtime = STT._is_realtime_model(model)
        return stt.STTCapabilities(
            streaming=is_realtime,
            interim_results=interim_results,
            offline_recognize=not is_realtime,
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        opts = replace(self._opts)
        if is_given(language):
            opts.language = language

        return SpeechStream(
            stt=self,
            client=self._client,
            opts=opts,
            conn_options=conn_options,
        )

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            if self._is_realtime_model(self._opts.model):
                raise APIStatusError(
                    message="Realtime model requires streaming; use stream()",
                    status_code=400,
                    request_id=None,
                    body=None,
                )
            if is_given(language):
                self._opts.language = language
            data = rtc.combine_audio_frames(buffer).to_wav_bytes()

            resp = await self._client.audio.transcriptions.complete_async(
                model=self._opts.model,
                file={"content": data, "file_name": "audio.wav"},
                language=self._opts.language or None,
                timestamp_granularities=None if self._opts.language else ["segment"],
            )

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=resp.text,
                        language=self._opts.language or "",
                        start_time=resp.segments[0].start if resp.segments else 0.0,
                        end_time=resp.segments[-1].end if resp.segments else 0.0,
                        words=[
                            TimedString(
                                text=segment.text,
                                start_time=segment.start,
                                end_time=segment.end,
                            )
                            for segment in resp.segments
                        ]
                        if resp.segments
                        else None,
                    ),
                ],
            )

        except SDKError as e:
            if e.status_code in (408, 504):
                raise APITimeoutError() from e
            raise APIStatusError(e.message, status_code=e.status_code, body=e.body) from e
        except (APIStatusError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError() from e

    @staticmethod
    def _is_realtime_model(model: str) -> bool:
        return "realtime" in model


class SpeechStream(stt.RecognizeStream):
    """Realtime speech recognition stream for MistralAI Voxtral.

    Uses debounced finalization: after VAD flush, waits for transcription
    to stabilize before emitting FINAL_TRANSCRIPT. Also supports idle-based
    finalization when FlushSentinel is not used (LiveKit 1.x STTNode path).
    """

    def __init__(
        self,
        *,
        stt: STT,
        client: Mistral,
        opts: _STTOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=opts.sample_rate,
        )
        self._client = client
        self._opts = opts

        self._speaking = False
        self._partial_text = ""

        self._pending_finalize = False
        self._last_text_time = 0.0
        self._flush_time = 0.0

        self._audio_duration = 0.0
        self._data_sent = False
        self._detected_language = self._opts.language or ""
        self._request_id = ""
        self._last_final_text = ""
        self._last_final_time = 0.0
        self._has_sent_final_for_turn = False
        self._usage_emitted = False

    @utils.log_exceptions(logger=logger)
    async def _run(self) -> None:
        audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=self._opts.sample_rate)

        try:
            realtime = RealtimeTranscription(self._client.sdk_configuration)
            finalize_task = asyncio.create_task(self._finalization_checker())

            try:
                async with await realtime.connect(
                    model=self._opts.model,
                    audio_format=audio_format,
                ) as connection:
                    self._request_id = connection.request_id or self._request_id
                    send_task = asyncio.create_task(self._send_audio(connection))
                    try:
                        async for event in connection.events():
                            if self._process_event(event):
                                break
                    finally:
                        await utils.aio.gracefully_cancel(send_task)
            finally:
                await utils.aio.gracefully_cancel(finalize_task)

            self._finalize_utterance(reason="stream_end")
            self._emit_usage_metrics()

        except SDKError as e:
            if e.status_code in (408, 504):
                raise APITimeoutError(retryable=not self._data_sent) from e
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                body=e.body,
                retryable=not self._data_sent,
            ) from e
        except (APIStatusError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError(retryable=not self._data_sent) from e

    async def _send_audio(self, connection: RealtimeConnection) -> None:
        samples_per_chunk = self._opts.sample_rate * CHUNK_DURATION_MS // 1000
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            samples_per_channel=samples_per_chunk,
        )

        async for data in self._input_ch:
            if connection.is_closed:
                break

            if isinstance(data, rtc.AudioFrame):
                data_bytes = data.data.tobytes()
                bytes_per_second = self._opts.sample_rate * NUM_CHANNELS * 2
                self._audio_duration += len(data_bytes) / bytes_per_second
                for frame in audio_bstream.write(data_bytes):
                    await connection.send_audio(frame.data.tobytes())
            elif isinstance(data, self._FlushSentinel):
                for frame in audio_bstream.flush():
                    await connection.send_audio(frame.data.tobytes())
                self._pending_finalize = True
                self._flush_time = time.monotonic()

        for frame in audio_bstream.flush():
            if connection.is_closed:
                break
            await connection.send_audio(frame.data.tobytes())

        if not connection.is_closed:
            await connection.end_audio()

    async def _finalization_checker(self) -> None:
        check_interval = 0.05
        finalize_delay_s = max(self._opts.finalize_delay_ms / 1000.0, 0.05)
        idle_finalize_delay_s = max(finalize_delay_s, MIN_IDLE_FINALIZE_MS / 1000.0)

        while True:
            await asyncio.sleep(check_interval)

            now = time.monotonic()
            time_since_text = now - self._last_text_time if self._last_text_time > 0 else 0.0

            if self._pending_finalize:
                time_since_flush = now - self._flush_time
                should_finalize = time_since_flush >= finalize_delay_s and (
                    self._last_text_time == 0 or time_since_text >= finalize_delay_s
                )
                if should_finalize:
                    self._finalize_utterance(reason="flush_timeout")
                    continue

            idle_ready = (
                self._speaking
                and self._last_text_time > 0
                and time_since_text >= idle_finalize_delay_s
            )
            if idle_ready:
                self._finalize_utterance(reason="idle_timeout")

    def _finalize_utterance(
        self,
        *,
        reason: str,
        start_time: float = 0.0,
        end_time: float = 0.0,
    ) -> None:
        text = self._partial_text.strip()
        if text:
            self._emit_final_text(text, start_time=start_time, end_time=end_time)

        if self._speaking:
            self._speaking = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

        self._partial_text = ""
        self._last_text_time = 0.0
        self._pending_finalize = False

    def _emit_usage_metrics(self, audio_duration: float | None = None) -> None:
        if self._usage_emitted:
            return

        duration = audio_duration if audio_duration is not None else self._audio_duration
        if duration > 0:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    request_id=self._request_id,
                    recognition_usage=stt.RecognitionUsage(audio_duration=duration),
                )
            )
            self._usage_emitted = True

    @staticmethod
    def _provider_audio_duration(
        event: TranscriptionStreamDone,
    ) -> float | None:
        usage = getattr(event, "usage", None)
        value = getattr(usage, "prompt_audio_seconds", None)
        if value is None:
            return None
        try:
            duration = float(value)
        except (TypeError, ValueError):
            return None
        return duration if duration > 0 else None

    def _process_event(self, event: Any) -> bool:
        if isinstance(
            event,
            RealtimeTranscriptionSessionCreated | RealtimeTranscriptionSessionUpdated,
        ):
            session = event.session
            if session.request_id:
                self._request_id = session.request_id
            return False

        if isinstance(event, TranscriptionStreamLanguage):
            if event.audio_language:
                self._detected_language = event.audio_language
            return False

        if isinstance(event, TranscriptionStreamTextDelta):
            self._handle_interim_text(event.text)
            return False

        if isinstance(event, TranscriptionStreamSegmentDelta):
            self._handle_final_segment(
                event.text,
                start_time=event.start,
                end_time=event.end,
            )
            return False

        if isinstance(event, TranscriptionStreamDone):
            if event.language:
                self._detected_language = event.language

            done_text = event.text.strip()
            if done_text:
                if self._partial_text:
                    self._partial_text = self._merge_partial_text(self._partial_text, done_text)
                elif not self._is_redundant_done_text(done_text):
                    self._partial_text = done_text

            self._finalize_utterance(reason="transcription_done")
            self._emit_usage_metrics(self._provider_audio_duration(event))
            return True

        if isinstance(event, RealtimeTranscriptionError):
            error = event.error
            message = error.message if isinstance(error.message, str) else str(error.message)
            raise APIStatusError(
                message=message,
                status_code=error.code,
                request_id=None,
                body=error,
                retryable=not self._data_sent,
            )

        if isinstance(event, UnknownRealtimeEvent):
            logger.warning(
                "unhandled unknown realtime event | type=%s | error=%s",
                event.type or "unknown",
                event.error or "n/a",
            )

        return False

    def _ensure_speaking(self) -> None:
        if not self._speaking:
            self._speaking = True
            self._has_sent_final_for_turn = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
            self._data_sent = True

    def _handle_interim_text(self, text: str) -> None:
        clean_text = text.strip()
        if not clean_text:
            return

        self._last_text_time = time.monotonic()
        self._ensure_speaking()
        self._partial_text = self._merge_partial_text(self._partial_text, clean_text)

        if self._opts.interim_results:
            language = self._detected_language or self._opts.language or ""
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[stt.SpeechData(text=self._partial_text, language=language)],
                )
            )
            self._data_sent = True

    def _handle_final_segment(
        self,
        text: str,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> None:
        clean_text = text.strip()
        if not clean_text:
            return

        self._last_text_time = time.monotonic()
        self._ensure_speaking()
        self._partial_text = ""
        start = (start_time or 0.0) + self.start_time_offset
        end = (end_time or 0.0) + self.start_time_offset
        self._emit_final_text(clean_text, start_time=start, end_time=end)

    def _emit_final_text(
        self,
        text: str,
        *,
        start_time: float = 0.0,
        end_time: float = 0.0,
    ) -> None:
        if not text or self._is_duplicate_final(text):
            return

        language = self._detected_language or self._opts.language or ""
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=[
                    stt.SpeechData(
                        text=text,
                        language=language,
                        start_time=start_time,
                        end_time=end_time,
                    )
                ],
            )
        )
        self._last_final_text = text
        self._last_final_time = time.monotonic()
        self._has_sent_final_for_turn = True
        self._data_sent = True

    def _is_redundant_done_text(self, text: str) -> bool:
        if not self._has_sent_final_for_turn or not self._last_final_text:
            return False

        return (
            text == self._last_final_text
            or text.endswith(self._last_final_text)
            or self._last_final_text.endswith(text)
        )

    def _is_duplicate_final(self, text: str) -> bool:
        if not self._last_final_text:
            return False
        return text == self._last_final_text and (time.monotonic() - self._last_final_time) < 1.0

    @staticmethod
    def _merge_partial_text(current: str, incoming: str) -> str:
        if not current:
            return incoming
        if not incoming:
            return current
        if incoming.startswith(current):
            return incoming
        if current.startswith(incoming):
            return current

        max_overlap = min(len(current), len(incoming))
        for overlap in range(max_overlap, 0, -1):
            if current.endswith(incoming[:overlap]):
                return current + incoming[overlap:]

        separator = (
            ""
            if current.endswith((" ", "\n", "\t")) or incoming.startswith((" ", "\n", "\t"))
            else " "
        )
        return f"{current}{separator}{incoming}"
