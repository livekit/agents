# Copyright 2026 LiveKit, Inc.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import sherpa_onnx

from livekit import rtc
from livekit.agents import (
    NOT_GIVEN,
    APIConnectOptions,
    LanguageCode,
    NotGivenOr,
    stt,
    utils,
)
from livekit.agents.utils import AudioBuffer, is_given

from .models import (
    BATCH_STT_MODELS,
    BatchSTTModels,
    MoonshineModel,
    WhisperModel,
    local_model_dir,
)
from .stt import SAMPLE_RATE, _to_float32


class OfflineResult(Protocol):
    @property
    def text(self) -> str: ...


class OfflineStream(Protocol):
    def accept_waveform(self, sample_rate: int, waveform: npt.NDArray[np.float32]) -> None: ...
    @property
    def result(self) -> OfflineResult: ...


class OfflineRecognizer(Protocol):
    """The subset of `sherpa_onnx.OfflineRecognizer` this plugin uses."""

    def create_stream(self) -> OfflineStream: ...
    def decode_stream(self, s: OfflineStream) -> None: ...


@dataclass
class _BatchSTTOptions:
    language: LanguageCode


class BatchSTT(stt.STT):
    def __init__(
        self,
        *,
        model: BatchSTTModels = "moonshine-tiny-en-int8",
        num_threads: int = 2,
        recognizer: OfflineRecognizer | None = None,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Non-streaming, fully offline STT backed by a sherpa-onnx offline recognizer.

        Transcribes a whole utterance at once, which is how Whisper, Moonshine, SenseVoice,
        Parakeet and the other sherpa-onnx offline models work. `AgentSession` makes it live
        on its own: it wraps a non-streaming STT in `stt.StreamAdapter` with the session's VAD
        (a local Silero VAD by default) and transcribes each detected utterance. Outside a
        session, wrap it yourself: `stt.StreamAdapter(stt=sherpa.BatchSTT(), vad=...)`.

        Args:
            model: Built-in model to load. Fetch it first with `download-files`.
            num_threads: ONNX Runtime threads used by the recognizer.
            recognizer: Use an already built `sherpa_onnx.OfflineRecognizer` instead of
                `model`, e.g. from `OfflineRecognizer.from_sense_voice(...)`. Any offline model
                sherpa-onnx supports works.
            language: Language reported on transcripts. Defaults to the model's language.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False, interim_results=False, offline_recognize=True
            )
        )
        spec = BATCH_STT_MODELS[model]
        self._model_name = model if recognizer is None else "custom"
        self._recognizer = recognizer or _load_offline(spec, num_threads=num_threads)
        self._opts = _BatchSTTOptions(
            language=LanguageCode(language if is_given(language) else spec.language)
        )

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "sherpa-onnx"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer)
        if frame.sample_rate != SAMPLE_RATE:
            resampler = rtc.AudioResampler(
                frame.sample_rate, SAMPLE_RATE, num_channels=frame.num_channels
            )
            frame = rtc.combine_audio_frames([*resampler.push(frame), *resampler.flush()])

        text = await asyncio.to_thread(self._decode_blocking, _to_float32(frame))
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=utils.shortuuid(),
            alternatives=[
                stt.SpeechData(
                    language=LanguageCode(language) if is_given(language) else self._opts.language,
                    text=text,
                    end_time=frame.duration,
                )
            ],
        )

    def _decode_blocking(self, samples: npt.NDArray[np.float32]) -> str:
        s = self._recognizer.create_stream()
        s.accept_waveform(SAMPLE_RATE, samples)
        self._recognizer.decode_stream(s)
        return s.result.text.strip()


def _load_offline(spec: MoonshineModel | WhisperModel, *, num_threads: int) -> OfflineRecognizer:
    root = Path(local_model_dir(spec))
    recognizer: OfflineRecognizer
    if isinstance(spec, MoonshineModel):
        recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=str(root / spec.preprocessor),
            encoder=str(root / spec.encoder),
            uncached_decoder=str(root / spec.uncached_decoder),
            cached_decoder=str(root / spec.cached_decoder),
            tokens=str(root / spec.tokens),
            num_threads=num_threads,
        )
    else:
        recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(root / spec.encoder),
            decoder=str(root / spec.decoder),
            tokens=str(root / spec.tokens),
            language=spec.language,
            num_threads=num_threads,
        )
    return recognizer
