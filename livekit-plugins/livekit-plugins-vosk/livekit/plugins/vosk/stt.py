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
import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Any, cast

import vosk  # type: ignore[import-untyped]
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    stt,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString

from .log import logger
from .utils import resample_audio

_DEFAULT_SAMPLE_RATE = 16000


@dataclass
class _STTOptions:
    model_path: str
    sample_rate: int
    language: str
    log_level: int | None


class STT(stt.STT):
    def __init__(
        self,
        *,
        model_path: str | None = None,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        language: str = "en",
        log_level: int | None = -1,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=False,
                aligned_transcript="word",
                offline_recognize=True,
            )
        )

        resolved_model_path = model_path or os.getenv("VOSK_MODEL_PATH")
        if not resolved_model_path:
            raise ValueError("Vosk model path is required. Set model_path or VOSK_MODEL_PATH.")
        if not os.path.isdir(resolved_model_path):
            raise ValueError(f"Vosk model path does not exist: {resolved_model_path}")

        self._opts = _STTOptions(
            model_path=resolved_model_path,
            sample_rate=sample_rate,
            language=language,
            log_level=log_level,
        )

        if self._opts.log_level is not None:
            vosk.SetLogLevel(self._opts.log_level)

        logger.info("Loading Vosk model from %s", resolved_model_path)
        self._model = vosk.Model(resolved_model_path)

    @property
    def model(self) -> str:
        return os.path.basename(self._opts.model_path)

    @property
    def provider(self) -> str:
        return "Vosk"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        del conn_options
        lang = language if is_given(language) else self._opts.language
        try:
            return await asyncio.to_thread(self._recognize_sync, buffer, lang)
        except ValueError:
            raise
        except Exception as e:
            raise APIConnectionError("Vosk recognition failed") from e

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        lang = language if is_given(language) else self._opts.language
        return SpeechStream(stt=self, conn_options=conn_options, language=lang)

    def _recognize_sync(self, buffer: AudioBuffer, language: str) -> stt.SpeechEvent:
        frames = resample_audio(buffer, self._opts.sample_rate)
        if not frames:
            return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT)

        recognizer = self._create_recognizer()
        segments: list[tuple[dict[str, Any], float]] = []
        segment_start = 0.0
        elapsed = 0.0
        for frame in frames:
            pcm = frame.data.tobytes()
            if pcm:
                elapsed += frame.duration
                if recognizer.AcceptWaveform(pcm):
                    segments.append((json.loads(recognizer.Result()), segment_start))
                    segment_start = elapsed

        segments.append((json.loads(recognizer.FinalResult()), segment_start))
        speech_data = self._merge_results(segments, language)
        if speech_data is None:
            return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[speech_data],
        )

    def _create_recognizer(self) -> vosk.KaldiRecognizer:
        recognizer = vosk.KaldiRecognizer(self._model, self._opts.sample_rate)
        recognizer.SetWords(True)
        return recognizer

    def _result_to_speech_data(
        self,
        result: dict[str, Any],
        language: str,
        *,
        start_time_offset: float,
    ) -> stt.SpeechData | None:
        text = (result.get("text") or "").strip()
        if not text:
            return None
        words, confidence = self._parse_words(result.get("result") or [], start_time_offset)
        start_time = cast(float, words[0].start_time) if words else 0.0
        end_time = cast(float, words[-1].end_time) if words else 0.0

        return stt.SpeechData(
            language=language,
            text=text,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            words=words if words else None,
        )

    def _merge_results(
        self,
        segments: list[tuple[dict[str, Any], float]],
        language: str,
    ) -> stt.SpeechData | None:
        texts: list[str] = []
        words: list[TimedString] = []
        conf_sum = 0.0
        conf_count = 0

        for result, offset in segments:
            text = (result.get("text") or "").strip()
            if text:
                texts.append(text)

            parsed_words, parsed_conf = self._parse_words(result.get("result") or [], offset)
            words.extend(parsed_words)
            if parsed_words:
                conf_sum += parsed_conf * len(parsed_words)
                conf_count += len(parsed_words)

        full_text = " ".join(texts).strip()
        if not full_text:
            return None

        start_time = cast(float, words[0].start_time) if words else 0.0
        end_time = cast(float, words[-1].end_time) if words else 0.0
        confidence = conf_sum / conf_count if conf_count else 0.0

        return stt.SpeechData(
            language=language,
            text=full_text,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            words=words if words else None,
        )

    def _parse_words(
        self,
        word_entries: list[dict[str, Any]],
        start_time_offset: float,
    ) -> tuple[list[TimedString], float]:
        words: list[TimedString] = []
        conf_sum = 0.0
        conf_count = 0

        for word in word_entries:
            start = float(word.get("start", 0.0)) + start_time_offset
            end = float(word.get("end", 0.0)) + start_time_offset
            words.append(
                TimedString(
                    text=str(word.get("word", "")),
                    start_time=start,
                    end_time=end,
                )
            )
            if "conf" in word:
                conf_sum += float(word["conf"])
                conf_count += 1

        confidence = conf_sum / conf_count if conf_count else 0.0
        return words, confidence

    async def aclose(self) -> None:
        return None


_THREAD_FLUSH = object()
_THREAD_STOP = object()


class SpeechStream(stt.SpeechStream):
    def __init__(self, *, stt: STT, conn_options: APIConnectOptions, language: str) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._opts.sample_rate)
        self._vosk_stt = stt
        self._language = language
        self._audio_queue: queue.Queue[object] = queue.Queue()
        self._recognition_thread: threading.Thread | None = None
        self._speaking = False
        self._last_partial = ""
        self._request_id = ""
        self._segment_index = 0

        self._event_loop = asyncio.get_running_loop()
        self._done_fut: asyncio.Future[None] = self._event_loop.create_future()

    async def _run(self) -> None:
        self._recognition_thread = threading.Thread(
            target=self._recognition_worker,
            name="vosk-recognition",
            daemon=True,
        )
        self._recognition_thread.start()

        try:
            await self._collect_audio()
        finally:
            self._audio_queue.put(_THREAD_STOP)
            await self._done_fut

    async def _collect_audio(self) -> None:
        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                pcm = data.data.tobytes()
                if pcm:
                    self._audio_queue.put(pcm)
            elif isinstance(data, self._FlushSentinel):
                self._audio_queue.put(_THREAD_FLUSH)

    def _recognition_worker(self) -> None:
        try:
            recognizer = self._vosk_stt._create_recognizer()
            while True:
                item = self._audio_queue.get()
                if item is _THREAD_STOP:
                    self._finalize_segment(recognizer)
                    break
                if item is _THREAD_FLUSH:
                    self._finalize_segment(recognizer)
                    recognizer = self._vosk_stt._create_recognizer()
                    continue
                if not isinstance(item, (bytes, bytearray, memoryview)):
                    continue

                if recognizer.AcceptWaveform(item):
                    result = json.loads(recognizer.Result())
                    self._emit_final(result)
                else:
                    partial = json.loads(recognizer.PartialResult())
                    self._emit_partial(partial)
        except Exception as e:
            logger.exception("Vosk recognition worker failed")
            if not self._done_fut.done():
                exc = APIConnectionError("Vosk streaming failed")
                exc.__cause__ = e
                self._event_loop.call_soon_threadsafe(self._done_fut.set_exception, exc)
        else:
            if not self._done_fut.done():
                self._event_loop.call_soon_threadsafe(self._done_fut.set_result, None)

    def _finalize_segment(self, recognizer: vosk.KaldiRecognizer) -> None:
        result = json.loads(recognizer.FinalResult())
        self._emit_final(result)

        if self._speaking:
            self._event_loop.call_soon_threadsafe(
                self._event_ch.send_nowait,
                stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH),
            )
            self._speaking = False
            self._last_partial = ""

    def _ensure_speaking(self) -> None:
        if self._speaking:
            return
        self._speaking = True
        self._segment_index += 1
        self._request_id = f"vosk-{uuid.uuid4().hex}-{self._segment_index}"
        self._event_loop.call_soon_threadsafe(
            self._event_ch.send_nowait,
            stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH),
        )

    def _emit_partial(self, data: dict[str, Any]) -> None:
        text = (data.get("partial") or "").strip()
        if not text or text == self._last_partial:
            return

        self._last_partial = text
        self._ensure_speaking()

        speech_data = stt.SpeechData(language=self._language, text=text)
        self._event_loop.call_soon_threadsafe(
            self._event_ch.send_nowait,
            stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=[speech_data],
            ),
        )

    def _emit_final(self, data: dict[str, Any]) -> None:
        speech_data = self._vosk_stt._result_to_speech_data(
            data,
            self._language,
            start_time_offset=self.start_time_offset,
        )
        if speech_data is None:
            return

        self._ensure_speaking()
        self._last_partial = ""

        self._event_loop.call_soon_threadsafe(
            self._event_ch.send_nowait,
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=[speech_data],
            ),
        )

        if self._speaking:
            self._event_loop.call_soon_threadsafe(
                self._event_ch.send_nowait,
                stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH),
            )
            self._speaking = False
