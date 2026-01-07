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
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    TimedString,
)
from livekit.agents.utils import AudioBuffer

from .log import logger
from .models import validate_model_path

if TYPE_CHECKING:
    import vosk  # type: ignore

# Global cache for loaded models
_model_cache: dict[str, vosk.Model] = {}
_spk_model_cache: dict[str, vosk.SpkModel] = {}
_cache_lock = threading.Lock()


def _get_model(path: str) -> vosk.Model:
    import vosk

    # Suppress verbose Vosk logs
    vosk.SetLogLevel(-1)

    with _cache_lock:
        if path not in _model_cache:
            logger.info(f"Loading Vosk model: {path}")
            _model_cache[path] = vosk.Model(path)
        return _model_cache[path]


def _get_spk_model(path: str) -> vosk.SpkModel:
    import vosk

    with _cache_lock:
        if path not in _spk_model_cache:
            logger.info(f"Loading Vosk speaker model: {path}")
            _spk_model_cache[path] = vosk.SpkModel(path)
        return _spk_model_cache[path]


@dataclass
class STTOptions:
    model_path: str
    sample_rate: int = 16000
    language: str = "en"
    enable_words: bool = True
    max_alternatives: int = 0
    speaker_model_path: str | None = None


class STT(stt.STT):
    def __init__(
        self,
        *,
        model_path: str,
        language: str = "en",
        sample_rate: int = 16000,
        enable_words: bool = True,
        max_alternatives: int = 0,
        speaker_model_path: str | None = None,
    ):
        """
        Create a new instance of Vosk STT.

        Args:
            model_path: Path to the Vosk model directory. Download models from
                https://alphacephei.com/vosk/models
            language: Language code for metadata (e.g., "en", "es", "fr")
            sample_rate: Audio sample rate in Hz. Vosk typically uses 16000.
            enable_words: Whether to include word-level timestamps in results
            max_alternatives: Number of alternative transcriptions to return (0 = disabled)
            speaker_model_path: Optional path to speaker identification model for diarization
        """
        # Validate model path exists
        self._model_path = validate_model_path(model_path)
        self._speaker_model_path = (
            validate_model_path(speaker_model_path) if speaker_model_path else None
        )

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=bool(speaker_model_path),
                aligned_transcript="word" if enable_words else False,
                offline_recognize=True,
            )
        )

        self._opts = STTOptions(
            model_path=str(self._model_path),
            sample_rate=sample_rate,
            language=language,
            enable_words=enable_words,
            max_alternatives=max_alternatives,
            speaker_model_path=str(self._speaker_model_path) if self._speaker_model_path else None,
        )

        self._label = f"vosk-{language}"

    def prewarm(self) -> None:
        """
        Preload models into memory to reduce latency for the first request.
        """
        try:
            _get_model(self._opts.model_path)
            if self._opts.speaker_model_path:
                _get_spk_model(self._opts.speaker_model_path)
            logger.info("Vosk models prewarmed")
        except Exception as e:
            logger.error(f"Failed to prewarm Vosk models: {e}")

    @property
    def model(self) -> str:
        return self._label

    @property
    def provider(self) -> str:
        return "vosk"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """
        Perform batch recognition on an audio buffer.
        """
        try:
            import vosk  # noqa: F401
        except ImportError as e:
            raise ImportError("Vosk is not installed. Install it with: pip install vosk") from e

        loop = asyncio.get_event_loop()

        # Create recognizer in thread pool
        def _create_recognizer() -> vosk.KaldiRecognizer:
            # Use cached models
            model = _get_model(self._opts.model_path)

            if self._opts.speaker_model_path:
                spk_model = _get_spk_model(self._opts.speaker_model_path)
                recognizer = vosk.KaldiRecognizer(model, self._opts.sample_rate, spk_model)
            else:
                recognizer = vosk.KaldiRecognizer(model, self._opts.sample_rate)

            recognizer.SetWords(self._opts.enable_words)
            if self._opts.max_alternatives > 0:
                recognizer.SetMaxAlternatives(self._opts.max_alternatives)

            return recognizer

        recognizer = await loop.run_in_executor(None, _create_recognizer)

        # Convert audio buffer to PCM16
        pcm16_data = _convert_audio_buffer_to_pcm16(buffer, self._opts.sample_rate)

        # Process audio in thread pool
        def _process_audio() -> str:
            recognizer.AcceptWaveform(pcm16_data)
            return str(recognizer.FinalResult())

        result_json = await loop.run_in_executor(None, _process_audio)

        # Parse result
        return _parse_vosk_result(
            result_json,
            is_final=True,
            language=self._opts.language,
            start_time_offset=0.0,
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        return SpeechStream(
            stt=self,
            opts=self._opts,
            conn_options=conn_options,
        )


class SpeechStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
    ):
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=opts.sample_rate,
        )
        self._opts = opts
        self._recognizer: vosk.KaldiRecognizer | None = None
        self._resampler: rtc.AudioResampler | None = None

    async def _run(self) -> None:
        try:
            import vosk  # noqa: F401
        except ImportError as e:
            raise ImportError("Vosk is not installed. Install it with: pip install vosk") from e

        loop = asyncio.get_event_loop()

        # Create recognizer in thread pool
        def _create_recognizer() -> vosk.KaldiRecognizer:
            # Use cached models
            model = _get_model(self._opts.model_path)

            if self._opts.speaker_model_path:
                spk_model = _get_spk_model(self._opts.speaker_model_path)
                recognizer = vosk.KaldiRecognizer(model, self._opts.sample_rate, spk_model)
            else:
                recognizer = vosk.KaldiRecognizer(model, self._opts.sample_rate)

            recognizer.SetWords(self._opts.enable_words)
            if self._opts.max_alternatives > 0:
                recognizer.SetMaxAlternatives(self._opts.max_alternatives)

            return recognizer

        self._recognizer = await loop.run_in_executor(None, _create_recognizer)
        logger.info("Vosk recognizer initialized", extra={"model": self._opts.model_path})

        # Main processing loop
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                # Finalize current segment
                def _finalize() -> str:
                    try:
                        if self._recognizer:
                            return self._recognizer.FinalResult() or ""
                        return ""
                    except Exception:
                        logger.exception("Vosk FinalResult failed")
                        return ""

                result_json = await loop.run_in_executor(None, _finalize)

                if result_json:
                    event = _parse_vosk_result(
                        result_json,
                        is_final=True,
                        language=self._opts.language,
                        start_time_offset=self.start_time_offset,
                    )

                    if event.alternatives and event.alternatives[0].text:
                        self._event_ch.send_nowait(event)

                # Reset recognizer for next segment (reuse same instance)
                def _reset() -> vosk.KaldiRecognizer | None:
                    if self._recognizer and hasattr(self._recognizer, "Reset"):
                        self._recognizer.Reset()
                        return None
                    else:
                        # Fallback if Reset not available (older versions)
                        logger.warning("Vosk recognizer does not support Reset(), recreating")
                        return _create_recognizer()

                # If result of _reset is a recognizer, update it
                res = await loop.run_in_executor(None, _reset)
                if res:
                    self._recognizer = res

                continue

            # Process audio frame
            frame: rtc.AudioFrame = data

            # Resample if needed
            if frame.sample_rate != self._opts.sample_rate:
                if not self._resampler or self._resampler.input_rate != frame.sample_rate:  # type: ignore
                    self._resampler = rtc.AudioResampler(
                        frame.sample_rate,
                        self._opts.sample_rate,
                        quality=rtc.AudioResamplerQuality.HIGH,
                    )
                    # Helper attribute to track input rate
                    self._resampler.input_rate = frame.sample_rate  # type: ignore

                resampled_frames = self._resampler.push(frame)
                for resampled_frame in resampled_frames:
                    pcm16_data = _convert_frame_to_pcm16(resampled_frame)

                    # Process in thread pool
                    def _accept_waveform_resampled(data: bytes) -> tuple[bool, str]:
                        try:
                            if self._recognizer:
                                is_final = self._recognizer.AcceptWaveform(data)
                                if is_final:
                                    return True, self._recognizer.Result() or ""
                                else:
                                    return False, self._recognizer.PartialResult() or ""
                            return False, "{}"
                        except Exception as e:
                            logger.error(f"Vosk processing error: {e}")
                            return False, "{}"

                    is_final, result_json = await loop.run_in_executor(
                        None, _accept_waveform_resampled, pcm16_data
                    )

                    if not result_json or result_json == "{}":
                        continue

                    event = _parse_vosk_result(
                        result_json,
                        is_final=is_final,
                        language=self._opts.language,
                        start_time_offset=self.start_time_offset,
                    )
                    if event.alternatives and event.alternatives[0].text:
                        self._event_ch.send_nowait(event)
                continue

            # Convert frame to PCM16
            pcm16_data = _convert_frame_to_pcm16(frame)

            # Process in thread pool
            def _accept_waveform(data: bytes) -> tuple[bool, str]:
                try:
                    if self._recognizer:
                        is_final = self._recognizer.AcceptWaveform(data)
                        if is_final:
                            return True, str(self._recognizer.Result() or "")
                        else:
                            return False, str(self._recognizer.PartialResult() or "")
                    return False, "{}"
                except Exception as e:
                    logger.error(f"Vosk processing error: {e}")
                    return False, "{}"

            is_final, result_json = await loop.run_in_executor(None, _accept_waveform, pcm16_data)

            if not result_json or result_json == "{}":
                continue

            # Parse and emit event
            event = _parse_vosk_result(
                result_json,
                is_final=is_final,
                language=self._opts.language,
                start_time_offset=self.start_time_offset,
            )

            if event.alternatives and event.alternatives[0].text:
                self._event_ch.send_nowait(event)


def _convert_frame_to_pcm16(frame: rtc.AudioFrame) -> bytes:
    """
    Convert AudioFrame to PCM16 format for Vosk.

    Vosk expects:
    - Format: PCM16 (16-bit signed integer)
    - Channels: Mono (1 channel)
    """
    # AudioFrame data is int16
    samples = np.frombuffer(frame.data, dtype=np.int16)

    # Reshape if multi-channel
    if frame.num_channels > 1:
        samples = samples.reshape(-1, frame.num_channels)
        # Convert to mono by averaging channels
        # Use float32 for averaging to avoid overflow/precision loss
        samples = samples.astype(np.float32).mean(axis=1).astype(np.int16)

    return samples.tobytes()


def _convert_audio_buffer_to_pcm16(buffer: AudioBuffer, target_sample_rate: int) -> bytes:
    """
    Convert AudioBuffer to PCM16 format for Vosk.
    """
    # Merge all frames in the buffer
    merged_frame = buffer.merge()  # type: ignore

    # Resample if needed
    if merged_frame.sample_rate != target_sample_rate:
        resampler = rtc.AudioResampler(
            merged_frame.sample_rate,
            target_sample_rate,
            quality=rtc.AudioResamplerQuality.HIGH,
        )
        frames = resampler.push(merged_frame)
        if frames:
            merged_frame = frames[0]

    return _convert_frame_to_pcm16(merged_frame)


def _parse_vosk_result(
    result_json: str,
    is_final: bool,
    language: str,
    start_time_offset: float,
) -> stt.SpeechEvent:
    """
    Parse Vosk JSON result into SpeechEvent.

    Vosk partial result format:
    {
        "partial": "hello world"
    }

    Vosk final result format:
    {
        "text": "hello world",
        "result": [
            {"conf": 1.0, "end": 0.5, "start": 0.0, "word": "hello"},
            {"conf": 0.98, "end": 1.2, "start": 0.5, "word": "world"}
        ]
    }

    With speaker diarization:
    {
        "text": "hello world",
        "spk": [0.1, 0.2, ...],  // Speaker vector
        "result": [...]
    }
    """
    result = json.loads(result_json)

    if is_final:
        text = result.get("text", "")
        words = []

        if "result" in result and result["result"]:
            for word_data in result["result"]:
                words.append(
                    TimedString(
                        text=word_data["word"],
                        start_time=word_data["start"] + start_time_offset,
                        end_time=word_data["end"] + start_time_offset,
                        confidence=word_data.get("conf", 1.0),
                    )
                )

        # Calculate overall confidence (average of word confidences)
        confidence = 1.0
        if words:
            confidence = sum(float(w.confidence) for w in words) / len(words)  # type: ignore

        # Calculate start and end times
        start_time = float(words[0].start_time) if words else float(start_time_offset)  # type: ignore
        end_time = float(words[-1].end_time) if words else float(start_time_offset)  # type: ignore

        alternatives = [
            stt.SpeechData(
                language=language,
                text=text,
                start_time=start_time,
                end_time=end_time,
                confidence=confidence,
                words=words if words else None,
            )
        ]

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=alternatives,
        )
    else:
        # Interim result
        text = result.get("partial", "")

        alternatives = [
            stt.SpeechData(
                language=language,
                text=text,
                start_time=start_time_offset,
                end_time=start_time_offset,
                confidence=0.0,  # Partial results don't have confidence
            )
        ]

        return stt.SpeechEvent(
            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
            alternatives=alternatives,
        )
