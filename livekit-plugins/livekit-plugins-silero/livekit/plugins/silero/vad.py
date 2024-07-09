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
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import numpy as np
from livekit import agents, rtc
from livekit.agents import utils

from . import onnx_model
from .log import logger


@dataclass
class _VADOptions:
    min_speech_duration: float
    min_silence_duration: float
    padding_duration: float
    max_buffered_speech: float
    activation_threshold: float
    sample_rate: int


class VAD(agents.vad.VAD):
    def __init__(
        self,
        *,
        min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.1,
        padding_duration: float = 0.1,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.25,
        sample_rate: int = 16000,
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
            force_cpu: force to use CPU for inference
        """

        if sample_rate not in onnx_model.SUPPORTED_SAMPLE_RATES:
            raise ValueError("Silero VAD only supports 8KHz and 16KHz sample rates")

        self._onnx_session = onnx_model.new_inference_session(force_cpu)
        self._opts = _VADOptions(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            padding_duration=padding_duration,
            max_buffered_speech=max_buffered_speech,
            activation_threshold=activation_threshold,
            sample_rate=sample_rate,
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
        self._opts, self._model = opts, model
        self._main_atask = asyncio.create_task(self._main_task())

        self._original_sample_rate: int | None = None
        self._window_data: _WindowData | None = None

        self._remaining_samples = model.window_size_samples

        self._input_q = asyncio.Queue[Optional[_WindowData]]()
        self._event_ch = utils.aio.Chan[agents.vad.VADEvent]()
        self._closed = False

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """
        Push frame to the VAD stream for processing.
        The frames are split into chunks of the given window size and processed.
        (Buffered if the window size is not reached yet)
        """
        if self._closed:
            raise ValueError("vad stream is closed")

        if frame.sample_rate != 8000 and frame.sample_rate % 16000 != 0:
            raise ValueError("only 8KHz and 16KHz*X sample rates are supported")

        if (
            self._original_sample_rate is not None
            and self._original_sample_rate != frame.sample_rate
        ):
            raise ValueError("a frame with another sample rate was already pushed")

        self._original_sample_rate = frame.sample_rate
        step = frame.sample_rate // 16000

        if self._window_data is None:
            self._window_data = _WindowData(
                inference_data=np.zeros(
                    self._model.window_size_samples, dtype=np.float32
                ),
                original_data=np.zeros(
                    self._model.window_size_samples * step, dtype=np.int16
                ),
            )

        if frame.num_channels != 1:
            raise ValueError("vad currently only supports mono audio frames")

        og_frame = np.frombuffer(frame.data, dtype=np.int16)
        if_frame = og_frame[::step].astype(np.float32) / np.iinfo(np.int16).max

        remaining_data = len(if_frame)
        while remaining_data > 0:
            i = self._model.window_size_samples - self._remaining_samples
            to_copy = min(remaining_data, self._remaining_samples)
            self._remaining_samples -= to_copy
            remaining_data -= to_copy

            self._window_data.original_data[i * step : i * step + to_copy * step] = (
                og_frame[: to_copy * step]
            )
            self._window_data.inference_data[i : i + to_copy] = if_frame[:to_copy]

            if self._remaining_samples == 0:
                self._input_q.put_nowait(self._window_data)
                self._window_data = _WindowData(
                    inference_data=np.zeros(
                        self._model.window_size_samples, dtype=np.float32
                    ),
                    original_data=np.zeros(
                        self._model.window_size_samples * step, dtype=np.int16
                    ),
                )
                self._remaining_samples = self._model.window_size_samples

    async def aclose(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._input_q.put_nowait(None)
        await self._main_atask

    @agents.utils.log_exceptions(logger=logger)
    async def _main_task(self):
        pub_speaking = False
        pub_speech_duration = 0.0
        pub_silence_duration = 0.0
        pub_speech_buf = np.array([], dtype=np.int16)

        may_start_at_sample = -1
        may_end_at_sample = -1

        min_speech_samples = self._opts.min_speech_duration * self._opts.sample_rate
        min_silence_samples = self._opts.min_silence_duration * self._opts.sample_rate

        current_sample = 0

        while True:
            window_data = await self._input_q.get()
            if window_data is None:
                break

            assert self._original_sample_rate is not None

            inference_data = window_data.inference_data
            start_time = time.time()
            raw_prob = await asyncio.to_thread(lambda: self._model(inference_data))
            inference_duration = time.time() - start_time

            window_duration = self._model.window_size_samples / self._opts.sample_rate
            if inference_duration > window_duration:
                # slower than realtime
                logger.warning(
                    "vad inference took too long - slower than realtime: %f",
                    inference_duration,
                )

            # append new data to current speech buffer
            pub_speech_buf = np.append(pub_speech_buf, window_data.original_data)
            max_data_s = self._opts.padding_duration
            if not pub_speaking:
                max_data_s += self._opts.min_speech_duration
            else:
                max_data_s += self._opts.max_buffered_speech

            cl = int(max_data_s) * self._original_sample_rate
            if len(pub_speech_buf) > cl:
                pub_speech_buf = pub_speech_buf[-cl:]

            # dispatch start/end when needed
            if raw_prob >= self._opts.activation_threshold:
                may_end_at_sample = -1

                if may_start_at_sample == -1:
                    may_start_at_sample = current_sample + min_speech_samples

                if may_start_at_sample <= current_sample and not pub_speaking:
                    pub_speaking = True
                    self._event_ch.send_nowait(
                        agents.vad.VADEvent(
                            type=agents.vad.VADEventType.START_OF_SPEECH,
                            silence_duration=pub_silence_duration,
                            speech_duration=0.0,
                            samples_index=current_sample,
                            speaking=True,
                        )
                    )

                    pub_silence_duration = 0
                    pub_speech_duration += self._opts.min_speech_duration

            if pub_speaking:
                pub_speech_duration += window_duration
            else:
                pub_silence_duration = 0

            self._event_ch.send_nowait(
                agents.vad.VADEvent(
                    type=agents.vad.VADEventType.INFERENCE_DONE,
                    samples_index=current_sample,
                    silence_duration=0.0,
                    speech_duration=pub_speech_duration,
                    probability=raw_prob,
                    inference_duration=inference_duration,
                    speaking=pub_speaking,
                )
            )

            if raw_prob < self._opts.activation_threshold:
                may_start_at_sample = -1

                if may_end_at_sample == -1:
                    may_end_at_sample = current_sample + min_silence_samples

                if may_end_at_sample <= current_sample and pub_speaking:
                    pub_speaking = False

                    frame = rtc.AudioFrame(
                        sample_rate=self._original_sample_rate,
                        num_channels=1,
                        samples_per_channel=len(pub_speech_buf),
                        data=pub_speech_buf.tobytes(),
                    )

                    self._event_ch.send_nowait(
                        agents.vad.VADEvent(
                            type=agents.vad.VADEventType.END_OF_SPEECH,
                            samples_index=current_sample,
                            silence_duration=0.0,
                            speech_duration=pub_speech_duration,
                            frames=[frame],
                            speaking=False,
                        )
                    )

                    pub_speech_buf = np.array([], dtype=np.int16)
                    pub_speech_duration = 0
                    pub_silence_duration += self._opts.min_silence_duration

            current_sample += self._model.window_size_samples

        self._event_ch.close()

    async def __anext__(self) -> agents.vad.VADEvent:
        return await self._event_ch.__anext__()

    def __aiter__(self) -> AsyncIterator[agents.vad.VADEvent]:
        return self._event_ch
