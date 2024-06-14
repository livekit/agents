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
import contextlib
import time
import numpy as np

from collections import deque
from dataclasses import dataclass
from typing import List, Optional

from livekit import agents, rtc

from .log import logger
from . import onnx_model


@dataclass
class _VADOptions:
    min_speech_duration: float
    min_silence_duration: float
    padding_duration: float
    max_buffered_speech: float
    window_size_samples: int
    activation_threshold: float
    sample_rate: int


class VAD(agents.vad.VAD):
    def __init__(
        self,
        *,
        min_speech_duration: float = 0.25,  # 250ms
        min_silence_duration: float = 0.1,  # 100ms
        padding_duration: float = 0.1,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.5,
        sample_rate: int = 16000,
        window_size_samples: int = 1024,
        force_cpu: bool = True,
    ) -> None:
        """
        Initialize the Silero VAD with the given options.
        The options are already set to strong defaults.

        Args:
            min_speech_duration: minimum duration of speech to start a new speech chunk
            min_silence_duration: In the end of each speech, wait min_silence_duration before ending the speech
            padding_duration: pad the chunks with this duration on both sides
            max_buffered_speech: maximum duration of speech to keep in the buffer (in seconds)
            activation_threshold: threshold to consider a frame as speech
            sample_rate: sample rate for the inference (only 8KHz and 16KHz are supported)
            window_size_samples: audio chunk size to use for the inference
                512, 1024, 1536 samples for 16000 sample rate and 256, 512, 768 samples for 8000 sample rate
            force_cpu: force to use CPU for inference
        """

        if sample_rate not in onnx_model.SUPPORTED_SAMPLE_RATES:
            raise ValueError("Silero VAD only supports 8KHz and 16KHz sample rates")

        if sample_rate == 8000 and window_size_samples not in [256, 512, 768]:
            raise ValueError("window_size_samples must be 256, 512, or 768 for 8KHz")

        if sample_rate == 16000 and window_size_samples not in [512, 1024, 1536]:
            raise ValueError("window_size_samples must be 512, 1024, or 1536 for 16KHz")

        self._onnx_session = onnx_model.new_inference_session(force_cpu)
        self._opts = _VADOptions(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            padding_duration=padding_duration,
            max_buffered_speech=max_buffered_speech,
            activation_threshold=activation_threshold,
            sample_rate=sample_rate,
            window_size_samples=window_size_samples,
        )

    def stream(
        self,
    ) -> "VADStream":
        return VADStream(
            self._opts,
            onnx_model.OnnxModel(
                onnx_session=self._onnx_session, sample_rate=self._opts.sample_rate
            ),
        )


@dataclass
class _WindowData:
    inference_data: np.ndarray
    # data returned to the user are the original frames (int16)
    original_data: np.ndarray


class VADStream(agents.vad.VADStream):
    def __init__(
        self,
        opts: _VADOptions,
        model: onnx_model.OnnxModel,
    ) -> None:
        super().__init__()
        self._opts, self._model = opts, model
        self._main_atask = asyncio.create_task(self._main_task())

        self._window_data: _WindowData | None = None
        self._window_index = 0
        self._pushed_sample_rate: int | None = None

        self._input_q = asyncio.Queue[Optional[_WindowData]]()

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """
        Push frame to the VAD stream for processing.
        The frames are split into chunks of the given window size and processed.
        (Buffered if the window size is not reached yet)
        """
        super().push_frame(frame)

        if frame.sample_rate != 8000 and frame.sample_rate % 16000 != 0:
            raise ValueError("only 8KHz and 16KHz*X sample rates are supported")

        if (
            self._pushed_sample_rate is not None
            and self._pushed_sample_rate != frame.sample_rate
        ):
            raise ValueError("a frame with another sample rate was already pushed")

        self._pushed_sample_rate = frame.sample_rate
        step = frame.sample_rate // 16000

        if self._window_data is None:
            self._window_data = _WindowData(
                inference_data=np.zeros(
                    self._opts.window_size_samples, dtype=np.float32
                ),
                original_data=np.zeros(
                    self._opts.window_size_samples * step, dtype=np.int16
                ),
            )

        if frame.num_channels != 1
            raise ValueError("vad currently only supports mono audio frames")

        ndata = np.frombuffer(frame.data, dtype=np.int16)

        rem_samples = len(ndata)
        while rem_samples > 0:
            window_index = self._window_index
            window_size = self._opts.window_size_samples

            to_copy = min(rem_samples, window_size - window_index)
            self._window_data.original_frame[window_index : window_index + to_copy] = (
                ndata[:to_copy]
            )
            self._window_data.inference_frame[window_index : window_index + to_copy] = (
                ndata[:to_copy].astype(np.float32) / np.iinfo(np.int16).max
            )

            rem_samples -= to_copy
            self._window_index += to_copy

            if self._window_index == window_size:
                self._input_q.put_nowait(self._window_data)
                self._window_data = _WindowData(
                    inference_frame=np.zeros(window_size, dtype=np.float32),
                    original_frame=np.zeros(window_size, dtype=np.int16),
                )
                self._window_index = 0

    async def aclose(self) -> None:
        super().aclose()
        await self._main_atask

    async def _main_task(self):
        while True:
            data = await self._input_q.get()
            if data is None:
                break

            window = data.inference_frame
            raw_prob = await asyncio.to_thread(lambda: self._model(window))

            print(raw_prob)

    async def _run_inference(self) -> None:
        # run inference
        start_time = time.time()
        raw_prob = await asyncio.to_thread(
            lambda: self._model(tensor, self._sample_rate).item()
        )
        probability = self._filter.apply(1.0, raw_prob)
        inference_duration = time.time() - start_time

        # inference done
        event = agents.vad.VADEvent(
            type=agents.vad.VADEventType.INFERENCE_DONE,
            samples_index=self._current_sample,
            probability=probability,
            raw_inference_prob=raw_prob,
            inference_duration=inference_duration,
        )
        self._event_queue.put_nowait(event)

        self._dispatch_event(original_frames, probability, raw_prob, inference_duration)
        self._current_sample += merged_frame.samples_per_channel

    def _dispatch_event(
        self,
        original_frames: List[rtc.AudioFrame],
        probability: float,
        raw_inference_prob: float,
        inference_duration: float,
    ):
        """
        Dispatches a VAD event based on the speech probability and the options
        Args:
            speech_prob: speech probability of the current frame
            original_frames: original frames of the current inference
        """

        self._buffered_frames.extend(original_frames)

        if len(self._buffered_frames) > max_buffer_count:
            self._buffered_frames = self._buffered_frames[-max_buffer_count:]

        if probability >= self._threshold:
            # speaking, wait for min_speaking_duration to trigger START_OF_SPEECH
            self._waiting_end = False
            if not self._waiting_start and not self._speaking:
                self._waiting_start = True
                self._start_speech = self._current_sample

            if self._waiting_start and (
                self._current_sample - self._start_speech >= self._min_speaking_samples
            ):
                self._waiting_start = False
                self._speaking = True

                # since we're waiting for the min_spaking_duration to trigger START_OF_SPEECH,
                # put the speech that were used to trigger the start here
                print("START OF SPEECH")
                event = agents.vad.VADEvent(
                    type=agents.vad.VADEventType.START_OF_SPEECH,
                    samples_index=self._start_speech,
                    frames=[],
                    speaking=True,
                )
                self._buffered_frames = []
                self._event_queue.put_nowait(event)

        # we don't check the speech_prob here
        event = agents.vad.VADEvent(
            type=agents.vad.VADEventType.INFERENCE_DONE,
            samples_index=self._current_sample,
            frames=original_frames.copy(),
            probability=probability,
            raw_inference_prob=raw_inference_prob,
            inference_duration=inference_duration,
            speaking=self._speaking,
        )
        self._event_queue.put_nowait(event)

        if probability < self._threshold:
            # stopped speaking, s for min_silence_duration to trigger END_OF_SPEECH,
            self._waiting_start = False
            if not self._waiting_end and self._speaking:
                self._waiting_end = True
                self._end_speech = self._current_sample

            if self._waiting_end and (
                self._current_sample - self._end_speech
                >= max(self._min_silence_samples, self._padding_duration_samples)
            ):
                self._waiting_end = False
                self._speaking = False
                print("END OF SPEECH")
                event = agents.vad.VADEvent(
                    type=agents.vad.VADEventType.END_OF_SPEECH,
                    samples_index=self._end_speech,
                    duration=(self._end_speech - self._start_speech)
                    / self._sample_rate,
                    frames=self._buffered_frames.copy(),
                    speaking=False,
                )
                self._event_queue.put_nowait(event)

    async def __anext__(self) -> agents.vad.VADEvent:
        evt = await self._event_queue.get()
        if evt is None:
            raise StopAsyncIteration

        return evt
