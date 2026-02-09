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
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Any, cast

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
from mistralai.extra.realtime.transcription import RealtimeTranscription

try:
    from mistralai.models import AudioFormat
except Exception:  # pragma: no cover - fallback for older SDK versions

    @dataclass
    class AudioFormat:  # type: ignore[no-redef]
        encoding: str
        sample_rate: int


from mistralai.models.sdkerror import SDKError

from .log import logger
from .models import STTModels

DEFAULT_SAMPLE_RATE = 16000
NUM_CHANNELS = 1
CHUNK_DURATION_MS = 50
DEFAULT_FINALIZE_DELAY_MS = 100
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
            capabilities=stt.STTCapabilities(
                streaming=self._is_realtime_model(model),
                interim_results=interim_results,
            )
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
            self._capabilities = stt.STTCapabilities(
                streaming=self._is_realtime_model(self._opts.model),
                interim_results=self._opts.interim_results,
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

            # MistralAI transcription API call
            resp = await self._client.audio.transcriptions.complete_async(
                model=self._opts.model,
                file={"content": data, "file_name": "audio.wav"},
                language=self._opts.language if self._opts.language else None,
                timestamp_granularities=None if self._opts.language else ["segment"],
            )

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=resp.text,
                        language=self._opts.language if self._opts.language else "",
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
            if e.status_code in (408, 504):  # Request Timeout, Gateway Timeout
                raise APITimeoutError() from e
            else:
                raise APIStatusError(e.message, status_code=e.status_code, body=e.body) from e
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
        self._last_final_text = ""
        self._last_final_time = 0.0

    @utils.log_exceptions(logger=logger)
    async def _run(self) -> None:
        audio_format = cast(Any, AudioFormat)(
            encoding="pcm_s16le", sample_rate=self._opts.sample_rate
        )

        async def audio_generator() -> AsyncIterator[bytes]:
            samples_per_chunk = self._opts.sample_rate * CHUNK_DURATION_MS // 1000
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=NUM_CHANNELS,
                samples_per_channel=samples_per_chunk,
            )

            async for data in self._input_ch:
                if isinstance(data, rtc.AudioFrame):
                    data_bytes = data.data.tobytes()
                    self._audio_duration += len(data_bytes) / (
                        self._opts.sample_rate * NUM_CHANNELS * 2
                    )
                    for frame in audio_bstream.write(data_bytes):
                        yield frame.data.tobytes()
                elif isinstance(data, self._FlushSentinel):
                    for frame in audio_bstream.flush():
                        yield frame.data.tobytes()
                    self._pending_finalize = True
                    self._flush_time = time.monotonic()

            for frame in audio_bstream.flush():
                yield frame.data.tobytes()

        try:
            # Directly instantiate RealtimeTranscription to avoid intermittent
            # AttributeError on client.audio.realtime property in forked processes
            realtime = RealtimeTranscription(self._client.sdk_configuration)

            finalize_task = asyncio.create_task(self._finalization_checker())

            try:
                async for event in realtime.transcribe_stream(
                    audio_stream=audio_generator(),
                    model=self._opts.model,
                    audio_format=audio_format,
                ):
                    self._process_event(event)
            finally:
                finalize_task.cancel()
                try:
                    await finalize_task
                except asyncio.CancelledError:
                    pass

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
        except Exception as e:
            raise APIConnectionError(retryable=not self._data_sent) from e

    async def _finalization_checker(self) -> None:
        """Background task that finalizes utterances after debounce delay or idle timeout."""
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

            if (
                self._partial_text
                and self._last_text_time > 0
                and time_since_text >= idle_finalize_delay_s
            ):
                self._finalize_utterance(reason="idle_timeout")

    def _finalize_utterance(
        self,
        *,
        reason: str,
        start_time: float = 0.0,
        end_time: float = 0.0,
    ) -> None:
        text = self._partial_text.strip()
        if text and not self._is_duplicate_final(text):
            language = self._detected_language or self._opts.language or ""
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
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
            self._data_sent = True

        if self._speaking:
            self._speaking = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

        self._partial_text = ""
        self._last_text_time = 0.0
        self._pending_finalize = False

    def _emit_usage_metrics(self) -> None:
        if self._audio_duration > 0:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    recognition_usage=stt.RecognitionUsage(audio_duration=self._audio_duration),
                )
            )

    def _process_event(self, event: Any) -> None:
        event_name = type(event).__name__
        event_type = self._event_type(event)

        if event_name == "RealtimeTranscriptionSessionCreated" or event_type == "session.created":
            return

        if event_name == "RealtimeTranscriptionSessionUpdated" or event_type == "session.updated":
            return

        if event_name == "TranscriptionStreamLanguage" or event_type == "transcription.language":
            language = getattr(event, "audio_language", "") or ""
            if language:
                self._detected_language = language
            return

        if event_name == "TranscriptionStreamTextDelta" or event_type == "transcription.text.delta":
            self._handle_interim_text(self._extract_text(event))
            return

        if event_name == "TranscriptionStreamSegmentDelta" or event_type == "transcription.segment":
            self._handle_final_text(
                self._extract_text(event),
                start_time=float(getattr(event, "start", 0.0) or 0.0),
                end_time=float(getattr(event, "end", 0.0) or 0.0),
            )
            return

        if event_name == "TranscriptionStreamDone" or event_type == "transcription.done":
            done_text = self._extract_text(event)
            if done_text:
                self._handle_final_text(done_text)
            else:
                self._finalize_utterance(reason="transcription_done")
            return

        if event_name == "RealtimeTranscriptionError" or event_type == "error":
            error = getattr(event, "error", None)
            raise APIStatusError(
                message=getattr(error, "message", "Unknown realtime transcription error"),
                status_code=getattr(error, "status_code", 500),
                request_id=None,
                body=error,
                retryable=not self._data_sent,
            )

        if event_name == "UnknownRealtimeEvent":
            self._handle_unknown_event(event)
            return

    def _handle_interim_text(self, text: str) -> None:
        clean_text = text.strip()
        if not clean_text:
            return

        self._last_text_time = time.monotonic()

        if not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
            self._data_sent = True

        self._partial_text = self._merge_partial_text(self._partial_text, clean_text)
        language = self._detected_language or self._opts.language or ""

        if self._opts.interim_results:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=[stt.SpeechData(text=self._partial_text, language=language)],
                )
            )
            self._data_sent = True

    def _handle_final_text(
        self,
        text: str,
        *,
        start_time: float = 0.0,
        end_time: float = 0.0,
    ) -> None:
        clean_text = text.strip()
        if not clean_text:
            self._finalize_utterance(reason="final_empty", start_time=start_time, end_time=end_time)
            return

        self._last_text_time = time.monotonic()
        if not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
            self._data_sent = True

        self._partial_text = self._merge_partial_text(self._partial_text, clean_text)
        self._finalize_utterance(reason="segment", start_time=start_time, end_time=end_time)

    def _handle_unknown_event(self, event: Any) -> None:
        payload = getattr(event, "content", None)
        event_type = getattr(event, "type", None)
        event_type_str = event_type if isinstance(event_type, str) else ""

        if isinstance(payload, dict):
            payload_type = payload.get("type")
            if not event_type_str and isinstance(payload_type, str):
                event_type_str = payload_type

            text = self._extract_text_from_mapping(payload)
            if text:
                if self._is_final_payload(payload, event_type_str):
                    self._handle_final_text(text)
                else:
                    self._handle_interim_text(text)
                return

        logger.warning(
            "unhandled unknown realtime event | type=%s | error=%s",
            event_type_str or "unknown",
            getattr(event, "error", "n/a"),
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

    @staticmethod
    def _event_type(event: Any) -> str:
        event_type = getattr(event, "type", None)
        if isinstance(event_type, str):
            return event_type
        alias_type = getattr(event, "TYPE", None)
        if isinstance(alias_type, str):
            return alias_type
        return ""

    def _extract_text(self, payload: Any) -> str:
        if isinstance(payload, dict):
            return self._extract_text_from_mapping(payload)

        for key in ("text", "transcript", "full_transcript", "delta", "utterance"):
            value = getattr(payload, key, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _extract_text_from_mapping(payload: dict[str, Any]) -> str:
        for key in ("text", "transcript", "full_transcript", "delta", "utterance"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _is_final_payload(payload: dict[str, Any], event_type: str) -> bool:
        if any(token in event_type.lower() for token in ("done", "final", "segment")):
            return True

        for key in ("is_final", "final", "is_last", "done", "completed"):
            value = payload.get(key)
            if isinstance(value, bool) and value:
                return True
        return False
