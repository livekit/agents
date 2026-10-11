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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import sherpa_onnx

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    LanguageCode,
    NotGivenOr,
    stt,
    utils,
)
from livekit.agents.types import TimedString
from livekit.agents.utils import AudioBuffer, is_given

from .models import STT_MODELS, STTModels, TransducerModel, local_model_dir

SAMPLE_RATE = 16000
# Silence appended when an utterance is forced to end, so the model can emit its last tokens.
TAIL_PADDING_S = 0.5


class OnlineResult(Protocol):
    text: str
    tokens: Sequence[str]
    timestamps: Sequence[float]
    start_time: float


class OnlineStream(Protocol):
    def accept_waveform(self, sample_rate: int, waveform: npt.NDArray[np.float32]) -> None: ...
    def input_finished(self) -> None: ...


class OnlineRecognizer(Protocol):
    """The subset of `sherpa_onnx.OnlineRecognizer` this plugin uses."""

    def create_stream(self) -> OnlineStream: ...
    def is_ready(self, s: OnlineStream) -> bool: ...
    def decode_stream(self, s: OnlineStream) -> None: ...
    def get_result_all(self, s: OnlineStream) -> OnlineResult: ...
    def is_endpoint(self, s: OnlineStream) -> bool: ...
    def reset(self, s: OnlineStream) -> None: ...


@dataclass
class _Snapshot:
    """A copy of the recognizer state, taken on the worker thread."""

    text: str
    tokens: list[str]
    timestamps: list[float]
    segment_start: float
    endpoint: bool


@dataclass
class _STTOptions:
    language: LanguageCode


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: STTModels = "zipformer-en-2023-06-26",
        num_threads: int = 2,
        endpoint_silence: float = 0.5,
        recognizer: OnlineRecognizer | None = None,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Streaming, fully offline STT backed by a sherpa-onnx online recognizer.

        Args:
            model: Built-in model to load. Fetch it first with `download-files`.
            num_threads: ONNX Runtime threads used by the recognizer.
            endpoint_silence: Trailing silence, in seconds, after which an utterance ends
                and a final transcript is emitted.
            recognizer: Use an already built `sherpa_onnx.OnlineRecognizer` instead of `model`
                (any online model sherpa-onnx supports). Endpoint detection must be enabled.
            language: Language reported on transcripts. Defaults to the model's language.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript="word",
                offline_recognize=True,
            )
        )
        spec = STT_MODELS[model]
        self._model_name = model if recognizer is None else "custom"
        self._recognizer = recognizer or _load_transducer(
            spec, num_threads=num_threads, endpoint_silence=endpoint_silence
        )
        self._opts = _STTOptions(
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

        snap = await asyncio.to_thread(self._recognize_blocking, _to_float32(frame))
        lang = LanguageCode(language) if is_given(language) else self._opts.language
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[_speech_data(snap, lang, start_time_offset=0.0)],
        )

    def _recognize_blocking(self, samples: npt.NDArray[np.float32]) -> _Snapshot:
        s = self._recognizer.create_stream()
        s.accept_waveform(SAMPLE_RATE, samples)
        s.accept_waveform(SAMPLE_RATE, _silence(TAIL_PADDING_S))
        s.input_finished()
        while self._recognizer.is_ready(s):
            self._recognizer.decode_stream(s)
        return _snapshot(self._recognizer.get_result_all(s), endpoint=True, clock_shift=0.0)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        lang = LanguageCode(language) if is_given(language) else self._opts.language
        return SpeechStream(
            stt=self, recognizer=self._recognizer, language=lang, conn_options=conn_options
        )


class SpeechStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: STT,
        recognizer: OnlineRecognizer,
        language: LanguageCode,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._recognizer = recognizer
        self._language = language
        self._stream = recognizer.create_stream()
        # sherpa's clock counts the padding we inject on flush; the session clock does not.
        self._padding_s = 0.0
        self._speaking = False
        self._last_text = ""
        self._unreported_audio_s = 0.0
        self._request_id = utils.shortuuid()

    async def _run(self) -> None:
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                snap = await asyncio.to_thread(self._finish_blocking)
            else:
                self._unreported_audio_s += data.duration
                snap = await asyncio.to_thread(self._step_blocking, _to_float32(data))
            self._handle(snap)

        self._handle(await asyncio.to_thread(self._finish_blocking))

    def _step_blocking(self, samples: npt.NDArray[np.float32]) -> _Snapshot:
        self._stream.accept_waveform(SAMPLE_RATE, samples)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        endpoint = self._recognizer.is_endpoint(self._stream)
        snap = _snapshot(
            self._recognizer.get_result_all(self._stream),
            endpoint=endpoint,
            clock_shift=self._padding_s,
        )
        if endpoint:
            self._recognizer.reset(self._stream)
        return snap

    def _finish_blocking(self) -> _Snapshot:
        """Force the current utterance to end, without waiting for trailing silence."""
        self._stream.accept_waveform(SAMPLE_RATE, _silence(TAIL_PADDING_S))
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        snap = _snapshot(
            self._recognizer.get_result_all(self._stream),
            endpoint=True,
            clock_shift=self._padding_s,
        )
        self._recognizer.reset(self._stream)
        self._padding_s += TAIL_PADDING_S
        return snap

    def _handle(self, snap: _Snapshot) -> None:
        if snap.text and not self._speaking:
            self._speaking = True
            self._emit(stt.SpeechEventType.START_OF_SPEECH)

        if not snap.endpoint:
            if snap.text and snap.text != self._last_text:
                self._last_text = snap.text
                self._emit(stt.SpeechEventType.INTERIM_TRANSCRIPT, snap)
            return

        # The final transcript must come before END_OF_SPEECH for the same utterance.
        if snap.text:
            self._emit(stt.SpeechEventType.FINAL_TRANSCRIPT, snap)
        if self._speaking:
            self._emit(stt.SpeechEventType.END_OF_SPEECH)
            self._report_usage()
        self._speaking = False
        self._last_text = ""
        self._request_id = utils.shortuuid()

    def _emit(self, type: stt.SpeechEventType, snap: _Snapshot | None = None) -> None:
        alternatives = (
            [_speech_data(snap, self._language, start_time_offset=self.start_time_offset)]
            if snap is not None
            else []
        )
        self._event_ch.send_nowait(
            stt.SpeechEvent(type=type, request_id=self._request_id, alternatives=alternatives)
        )

    def _report_usage(self) -> None:
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                recognition_usage=stt.RecognitionUsage(audio_duration=self._unreported_audio_s),
            )
        )
        self._unreported_audio_s = 0.0


def _load_transducer(
    spec: TransducerModel, *, num_threads: int, endpoint_silence: float
) -> OnlineRecognizer:
    root = Path(local_model_dir(spec))
    recognizer: OnlineRecognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(root / spec.tokens),
        encoder=str(root / spec.encoder),
        decoder=str(root / spec.decoder),
        joiner=str(root / spec.joiner),
        num_threads=num_threads,
        sample_rate=SAMPLE_RATE,
        enable_endpoint_detection=True,
        rule2_min_trailing_silence=endpoint_silence,
    )
    return recognizer


def _snapshot(result: OnlineResult, *, endpoint: bool, clock_shift: float) -> _Snapshot:
    """Copy a result. `clock_shift` is the injected padding to remove from sherpa's clock."""
    return _Snapshot(
        text=result.text.strip(),
        tokens=list(result.tokens),
        timestamps=list(result.timestamps),
        segment_start=result.start_time - clock_shift,
        endpoint=endpoint,
    )


def _speech_data(
    snap: _Snapshot, language: LanguageCode, *, start_time_offset: float
) -> stt.SpeechData:
    base = snap.segment_start + start_time_offset
    spans = [(text, base + start, base + end) for text, start, end in _word_spans(snap)]
    return stt.SpeechData(
        language=language,
        text=snap.text,
        start_time=spans[0][1] if spans else start_time_offset,
        end_time=spans[-1][2] if spans else start_time_offset,
        words=[
            TimedString(text, start_time=start, end_time=end, start_time_offset=start_time_offset)
            for text, start, end in spans
        ]
        or None,
    )


def _word_spans(snap: _Snapshot) -> list[tuple[str, float, float]]:
    """Group BPE tokens into (word, start, end), relative to the segment start.

    A token starting with a space opens a new word. sherpa-onnx gives one onset per token,
    so a word ends where the next one starts and the last word ends at its last token.
    """
    words: list[tuple[str, float]] = []
    for token, onset in zip(snap.tokens, snap.timestamps, strict=False):
        if token.startswith(" ") or not words:
            words.append((token.strip(), onset))
        else:
            text, start = words[-1]
            words[-1] = (text + token, start)
    if not words:
        return []
    ends = [start for _, start in words[1:]] + [snap.timestamps[len(snap.tokens) - 1]]
    return [(text, start, end) for (text, start), end in zip(words, ends, strict=True) if text]


def _to_float32(frame: rtc.AudioFrame) -> npt.NDArray[np.float32]:
    pcm = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
    mono = pcm.mean(axis=1) if frame.num_channels > 1 else pcm[:, 0]
    return (mono / 32768.0).astype(np.float32)


def _silence(seconds: float) -> npt.NDArray[np.float32]:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)
