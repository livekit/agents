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
import time
import weakref
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np

from livekit import rtc
from livekit.local_inference import VAD as _NativeVAD, VAD_WINDOW_SAMPLES

from .. import utils, vad
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given

if TYPE_CHECKING:
    from concurrent.futures import Executor

SLOW_INFERENCE_THRESHOLD = 0.2  # late by 200ms
_MODEL_SAMPLE_RATE = 16000

VADModels = Literal["silero"]


@dataclass
class _VADOptions:
    min_speech_duration: float
    min_silence_duration: float
    prefix_padding_duration: float
    max_buffered_speech: float
    activation_threshold: float
    deactivation_threshold: float


class VAD(vad.VAD):
    """Voice Activity Detection backed by ``livekit-local-inference``.

    The native model singleton is loaded once at module import (via the
    pybind11 ``.so`` constructor); each stream allocates its own per-instance
    LSTM/context state. Pass ``executor`` to isolate inference from the event
    loop's default executor; the caller remains responsible for shutting it down.
    """

    def __init__(
        self,
        *,
        model: VADModels = "silero",
        min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.25,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.5,
        deactivation_threshold: NotGivenOr[float] = NOT_GIVEN,
        executor: Executor | None = None,
    ) -> None:
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.032))
        if model != "silero":
            raise ValueError(f"Unknown VAD model: {model!r}. Supported: 'silero'.")
        if is_given(deactivation_threshold) and deactivation_threshold <= 0:
            raise ValueError("deactivation_threshold must be greater than 0")
        self._model = model
        self._executor = executor
        self._opts = _VADOptions(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
            activation_threshold=activation_threshold,
            deactivation_threshold=deactivation_threshold
            if is_given(deactivation_threshold)
            else max(activation_threshold - 0.15, 0.01),
        )
        self._streams: weakref.WeakSet[_VADStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "livekit-local-inference"

    def stream(self) -> vad.VADStream:
        # Each stream owns its own _VADOptions snapshot so that
        # _VADStream.update_options() can read the prior value of
        # max_buffered_speech before mutating it. Sharing the dataclass would
        # let VAD.update_options() mutate the stream's view first, and the
        # stream would never observe an increase.
        stream = _VADStream(self, replace(self._opts), executor=self._executor)
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        min_speech_duration: NotGivenOr[float] = NOT_GIVEN,
        min_silence_duration: NotGivenOr[float] = NOT_GIVEN,
        prefix_padding_duration: NotGivenOr[float] = NOT_GIVEN,
        max_buffered_speech: NotGivenOr[float] = NOT_GIVEN,
        activation_threshold: NotGivenOr[float] = NOT_GIVEN,
        deactivation_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(min_speech_duration):
            self._opts.min_speech_duration = min_speech_duration
        if is_given(min_silence_duration):
            self._opts.min_silence_duration = min_silence_duration
        if is_given(prefix_padding_duration):
            self._opts.prefix_padding_duration = prefix_padding_duration
        if is_given(max_buffered_speech):
            self._opts.max_buffered_speech = max_buffered_speech
        if is_given(activation_threshold):
            self._opts.activation_threshold = activation_threshold
        if is_given(deactivation_threshold):
            self._opts.deactivation_threshold = deactivation_threshold

        for stream in self._streams:
            stream.update_options(
                min_speech_duration=min_speech_duration,
                min_silence_duration=min_silence_duration,
                prefix_padding_duration=prefix_padding_duration,
                max_buffered_speech=max_buffered_speech,
                activation_threshold=activation_threshold,
                deactivation_threshold=deactivation_threshold,
            )

    @property
    def min_silence_duration(self) -> float | None:
        return self._opts.min_silence_duration


class _VADStream(vad.VADStream):
    def __init__(self, parent: VAD, opts: _VADOptions, *, executor: Executor | None) -> None:
        super().__init__(parent)
        self._opts = opts
        self._executor = executor
        self._native_vad = _NativeVAD()

        self._input_sample_rate = 0
        self._speech_buffer: np.ndarray | None = None
        self._speech_buffer_max_reached = False
        self._prefix_padding_samples = 0  # (input_sample_rate)

    def update_options(
        self,
        *,
        min_speech_duration: NotGivenOr[float] = NOT_GIVEN,
        min_silence_duration: NotGivenOr[float] = NOT_GIVEN,
        prefix_padding_duration: NotGivenOr[float] = NOT_GIVEN,
        max_buffered_speech: NotGivenOr[float] = NOT_GIVEN,
        activation_threshold: NotGivenOr[float] = NOT_GIVEN,
        deactivation_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        old_max_buffered_speech = self._opts.max_buffered_speech

        if is_given(min_speech_duration):
            self._opts.min_speech_duration = min_speech_duration
        if is_given(min_silence_duration):
            self._opts.min_silence_duration = min_silence_duration
        if is_given(prefix_padding_duration):
            self._opts.prefix_padding_duration = prefix_padding_duration
        if is_given(max_buffered_speech):
            self._opts.max_buffered_speech = max_buffered_speech
        if is_given(activation_threshold):
            self._opts.activation_threshold = activation_threshold
        if is_given(deactivation_threshold):
            self._opts.deactivation_threshold = deactivation_threshold

        if self._input_sample_rate:
            assert self._speech_buffer is not None

            self._prefix_padding_samples = int(
                self._opts.prefix_padding_duration * self._input_sample_rate
            )

            self._speech_buffer.resize(
                int(self._opts.max_buffered_speech * self._input_sample_rate)
                + self._prefix_padding_samples
            )

            if self._opts.max_buffered_speech > old_max_buffered_speech:
                self._speech_buffer_max_reached = False

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        speech_buffer_index: int = 0

        # "pub_" means public, these values are exposed to the users through events
        pub_speaking = False
        pub_speech_duration = 0.0
        pub_silence_duration = 0.0
        pub_current_sample = 0
        pub_timestamp = 0.0

        speech_threshold_duration = 0.0
        silence_threshold_duration = 0.0

        input_frames: list[rtc.AudioFrame] = []
        inference_frames: list[rtc.AudioFrame] = []
        resampler: rtc.AudioResampler | None = None

        # used to avoid drift when the sample_rate ratio is not an integer
        input_copy_remaining_fract = 0.0

        extra_inference_time = 0.0

        def _reset_state() -> None:
            nonlocal speech_buffer_index
            nonlocal pub_speaking, pub_speech_duration, pub_silence_duration
            nonlocal pub_current_sample, pub_timestamp
            nonlocal speech_threshold_duration, silence_threshold_duration
            nonlocal input_frames, inference_frames, resampler
            nonlocal input_copy_remaining_fract, extra_inference_time

            self._native_vad.reset()

            speech_buffer_index = 0
            self._speech_buffer_max_reached = False
            if self._speech_buffer is not None:
                self._speech_buffer.fill(0)

            pub_speaking = False
            pub_speech_duration = 0.0
            pub_silence_duration = 0.0
            pub_current_sample = 0
            pub_timestamp = 0.0
            speech_threshold_duration = 0.0
            silence_threshold_duration = 0.0

            input_frames = []
            inference_frames = []
            input_copy_remaining_fract = 0.0
            extra_inference_time = 0.0

            if self._input_sample_rate and self._input_sample_rate != _MODEL_SAMPLE_RATE:
                resampler = rtc.AudioResampler(
                    input_rate=self._input_sample_rate,
                    output_rate=_MODEL_SAMPLE_RATE,
                    quality=rtc.AudioResamplerQuality.QUICK,
                )
            else:
                resampler = None

        async for input_frame in self._input_ch:
            if isinstance(input_frame, self._FlushSentinel):
                _reset_state()
                continue

            if not isinstance(input_frame, rtc.AudioFrame):
                continue

            if not self._input_sample_rate:
                self._input_sample_rate = input_frame.sample_rate

                # alloc the buffers now that we know the input sample rate
                self._prefix_padding_samples = int(
                    self._opts.prefix_padding_duration * self._input_sample_rate
                )

                self._speech_buffer = np.empty(
                    int(self._opts.max_buffered_speech * self._input_sample_rate)
                    + self._prefix_padding_samples,
                    dtype=np.int16,
                )

                if self._input_sample_rate != _MODEL_SAMPLE_RATE:
                    # resampling needed: the input sample rate isn't the same as the model's
                    # sample rate used for inference
                    resampler = rtc.AudioResampler(
                        input_rate=self._input_sample_rate,
                        output_rate=_MODEL_SAMPLE_RATE,
                        quality=rtc.AudioResamplerQuality.QUICK,  # VAD doesn't need high quality
                    )

            elif self._input_sample_rate != input_frame.sample_rate:
                logger.error("a frame with another sample rate was already pushed")
                continue

            assert self._speech_buffer is not None

            input_frames.append(input_frame)
            if resampler is not None:
                # the resampler may have a bit of latency, but it is OK to ignore since it should be
                # negligible
                inference_frames.extend(resampler.push(input_frame))
            else:
                inference_frames.append(input_frame)

            while True:
                start_time = time.perf_counter()

                available_inference_samples = sum(
                    [frame.samples_per_channel for frame in inference_frames]
                )
                if available_inference_samples < VAD_WINDOW_SAMPLES:
                    break  # not enough samples to run inference

                input_frame = utils.combine_frames(input_frames)
                inference_frame = utils.combine_frames(inference_frames)

                # native lib takes int16 directly — no float32 conversion
                inference_window = np.asarray(
                    inference_frame.data[:VAD_WINDOW_SAMPLES], dtype=np.int16
                )

                # run the inference
                p = await asyncio.get_running_loop().run_in_executor(
                    self._executor, self._native_vad.predict, inference_window
                )

                window_duration = VAD_WINDOW_SAMPLES / _MODEL_SAMPLE_RATE

                pub_current_sample += VAD_WINDOW_SAMPLES
                pub_timestamp += window_duration

                resampling_ratio = self._input_sample_rate / _MODEL_SAMPLE_RATE
                to_copy = VAD_WINDOW_SAMPLES * resampling_ratio + input_copy_remaining_fract
                to_copy_int = int(to_copy)
                input_copy_remaining_fract = to_copy - to_copy_int

                # copy the inference window to the speech buffer
                available_space = len(self._speech_buffer) - speech_buffer_index
                to_copy_buffer = min(to_copy_int, available_space)
                if to_copy_buffer > 0:
                    self._speech_buffer[
                        speech_buffer_index : speech_buffer_index + to_copy_buffer
                    ] = input_frame.data[:to_copy_buffer]
                    speech_buffer_index += to_copy_buffer
                elif not self._speech_buffer_max_reached:
                    # reached self._opts.max_buffered_speech (padding is included)
                    self._speech_buffer_max_reached = True
                    logger.warning(
                        "max_buffered_speech reached, ignoring further data for the current speech input"  # noqa: E501
                    )

                inference_duration = time.perf_counter() - start_time
                extra_inference_time = max(
                    0.0,
                    extra_inference_time + inference_duration - window_duration,
                )
                if inference_duration > SLOW_INFERENCE_THRESHOLD:
                    logger.warning(
                        "inference is slower than realtime",
                        extra={"delay": extra_inference_time},
                    )

                def _reset_write_cursor() -> None:
                    nonlocal speech_buffer_index
                    assert self._speech_buffer is not None

                    if speech_buffer_index <= self._prefix_padding_samples:
                        return

                    padding_data = self._speech_buffer[
                        speech_buffer_index - self._prefix_padding_samples : speech_buffer_index
                    ]

                    self._speech_buffer_max_reached = False
                    self._speech_buffer[: self._prefix_padding_samples] = padding_data
                    speech_buffer_index = self._prefix_padding_samples

                def _copy_speech_buffer() -> rtc.AudioFrame:
                    # copy the data from speech_buffer
                    assert self._speech_buffer is not None
                    speech_data = self._speech_buffer[:speech_buffer_index].tobytes()  # noqa: B023

                    return rtc.AudioFrame(
                        sample_rate=self._input_sample_rate,
                        num_channels=1,
                        samples_per_channel=speech_buffer_index,  # noqa: B023
                        data=speech_data,
                    )

                if pub_speaking:
                    pub_speech_duration += window_duration
                else:
                    pub_silence_duration += window_duration

                self._event_ch.send_nowait(
                    vad.VADEvent(
                        type=vad.VADEventType.INFERENCE_DONE,
                        samples_index=pub_current_sample,
                        timestamp=pub_timestamp,
                        silence_duration=pub_silence_duration,
                        speech_duration=pub_speech_duration,
                        probability=p,
                        inference_duration=inference_duration,
                        frames=[
                            rtc.AudioFrame(
                                data=input_frame.data[:to_copy_int].tobytes(),
                                sample_rate=self._input_sample_rate,
                                num_channels=1,
                                samples_per_channel=to_copy_int,
                            )
                        ],
                        speaking=pub_speaking,
                        raw_accumulated_silence=silence_threshold_duration,
                        raw_accumulated_speech=speech_threshold_duration,
                    )
                )

                if p >= self._opts.activation_threshold or (
                    pub_speaking and p > self._opts.deactivation_threshold
                ):
                    speech_threshold_duration += window_duration
                    silence_threshold_duration = 0.0

                    if not pub_speaking:
                        if speech_threshold_duration >= self._opts.min_speech_duration:
                            pub_speaking = True
                            pub_silence_duration = 0.0
                            pub_speech_duration = speech_threshold_duration

                            self._event_ch.send_nowait(
                                vad.VADEvent(
                                    type=vad.VADEventType.START_OF_SPEECH,
                                    samples_index=pub_current_sample,
                                    timestamp=pub_timestamp,
                                    silence_duration=pub_silence_duration,
                                    speech_duration=pub_speech_duration,
                                    frames=[_copy_speech_buffer()],
                                    speaking=True,
                                )
                            )

                else:
                    silence_threshold_duration += window_duration
                    speech_threshold_duration = 0.0

                    if not pub_speaking:
                        _reset_write_cursor()

                    if (
                        pub_speaking
                        and silence_threshold_duration >= self._opts.min_silence_duration
                    ):
                        pub_speaking = False
                        pub_silence_duration = silence_threshold_duration

                        self._event_ch.send_nowait(
                            vad.VADEvent(
                                type=vad.VADEventType.END_OF_SPEECH,
                                samples_index=pub_current_sample,
                                timestamp=pub_timestamp,
                                silence_duration=pub_silence_duration,
                                speech_duration=max(
                                    0.0, pub_speech_duration - silence_threshold_duration
                                ),
                                frames=[_copy_speech_buffer()],
                                speaking=False,
                            )
                        )

                        pub_speech_duration = 0.0

                        _reset_write_cursor()

                # remove the frames that were used for inference from the input and inference frames
                input_frames = []
                inference_frames = []

                # add the remaining data
                if len(input_frame.data) - to_copy_int > 0:
                    data = input_frame.data[to_copy_int:].tobytes()
                    input_frames.append(
                        rtc.AudioFrame(
                            data=data,
                            sample_rate=self._input_sample_rate,
                            num_channels=1,
                            samples_per_channel=len(data) // 2,
                        )
                    )

                if len(inference_frame.data) - VAD_WINDOW_SAMPLES > 0:
                    data = inference_frame.data[VAD_WINDOW_SAMPLES:].tobytes()
                    inference_frames.append(
                        rtc.AudioFrame(
                            data=data,
                            sample_rate=_MODEL_SAMPLE_RATE,
                            num_channels=1,
                            samples_per_channel=len(data) // 2,
                        )
                    )
