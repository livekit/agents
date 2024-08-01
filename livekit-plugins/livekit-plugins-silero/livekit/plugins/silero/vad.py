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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import onnxruntime  # type: ignore
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
    @classmethod
    def load(
        cls,
        *,
        min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.25,
        padding_duration: float = 0.1,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.25,
        sample_rate: int = 16000,
        force_cpu: bool = True,
    ) -> "VAD":
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

        session = onnx_model.new_inference_session(force_cpu)
        opts = _VADOptions(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            padding_duration=padding_duration,
            max_buffered_speech=max_buffered_speech,
            activation_threshold=activation_threshold,
            sample_rate=sample_rate,
        )
        return cls(session=session, opts=opts)

    def __init__(
        self,
        *,
        session: onnxruntime.InferenceSession,
        opts: _VADOptions,
    ) -> None:
        super().__init__(capabilities=agents.vad.VADCapabilities(update_interval=0.032))
        self._onnx_session = session
        self._opts = opts

    def stream(self) -> "VADStream":
        return VADStream(
            self._opts,
            onnx_model.OnnxModel(
                onnx_session=self._onnx_session, sample_rate=self._opts.sample_rate
            ),
        )


class VADStream(agents.vad.VADStream):
    def __init__(self, opts: _VADOptions, model: onnx_model.OnnxModel) -> None:
        super().__init__()
        self._opts, self._model = opts, model
        self._loop = asyncio.get_event_loop()

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._task.add_done_callback(lambda _: self._executor.shutdown(wait=False))
        self._exp_filter = utils.ExpFilter(alpha=0.35)

    @agents.utils.log_exceptions(logger=logger)
    async def _main_task(self):
        og_sample_rate = 0
        og_needed_samples = 0  # needed samples to complete the window data
        og_window_size_samples = 0  # size in samples of og_window_data
        og_window_data: np.ndarray | None = None

        index_step = 0
        inference_window_data = np.empty(
            self._model.window_size_samples, dtype=np.float32
        )

        # a copy is exposed to the user in END_OF_SPEECH
        speech_buffer: np.ndarray | None = None
        speech_buffer_index: int = 0

        # "pub_" means public, these values are exposed to the users through events
        pub_speaking = False
        pub_speech_duration = 0.0
        pub_silence_duration = 0.0
        pub_current_sample = 0

        speech_threshold_duration = 0.0
        silence_threshold_duration = 0.0

        async for frame in self._input_ch:
            if not isinstance(frame, rtc.AudioFrame):
                continue  # ignore flush sentinel for now

            if frame.sample_rate != 8000 and frame.sample_rate % 16000 != 0:
                logger.error("only 8KHz and 16KHz*X sample rates are supported")
                continue
            elif og_window_data is None:
                # alloc the og buffers now that we know the pushed sample rate
                og_sample_rate = frame.sample_rate
                og_window_size_samples = int(
                    (self._model.window_size_samples / self._model.sample_rate)
                    * og_sample_rate
                )
                og_window_data = np.empty(og_window_size_samples, dtype=np.int16)
                og_needed_samples = og_window_size_samples
                index_step = frame.sample_rate // 16000

                speech_buffer = np.empty(
                    int(self._opts.max_buffered_speech * og_sample_rate), dtype=np.int16
                )
            elif og_sample_rate != frame.sample_rate:
                logger.error("a frame with another sample rate was already pushed")
                continue

            frame_data = np.frombuffer(frame.data, dtype=np.int16)
            remaining_samples = len(frame_data)
            while remaining_samples > 0:
                to_copy = min(remaining_samples, og_needed_samples)

                index = len(og_window_data) - og_needed_samples
                og_window_data[index : index + to_copy] = frame_data[:to_copy]

                remaining_samples -= to_copy
                og_needed_samples -= to_copy

                if og_needed_samples != 0:
                    continue

                og_needed_samples = og_window_size_samples

                # copy the data to the inference buffer by sampling at each index_step & convert to float
                np.divide(
                    og_window_data[::index_step],
                    np.iinfo(np.int16).max,
                    out=inference_window_data,
                    dtype=np.float32,
                )

                # run the inference
                start_time = time.time()
                raw_prob = await self._loop.run_in_executor(
                    self._executor, self._model, inference_window_data
                )

                prob_change = abs(raw_prob - self._exp_filter.filtered())
                exp = 0.5 if prob_change > 0.25 else 1
                raw_prob = self._exp_filter.apply(exp=exp, sample=raw_prob)

                inference_duration = time.time() - start_time
                window_duration = (
                    self._model.window_size_samples / self._opts.sample_rate
                )
                if inference_duration > window_duration:
                    logger.warning(
                        "vad inference took too long - slower than realtime: %f",
                        inference_duration,
                    )

                pub_current_sample += og_window_size_samples

                def _copy_window():
                    nonlocal speech_buffer_index
                    to_copy = min(
                        og_window_size_samples,
                        len(speech_buffer) - speech_buffer_index,
                    )
                    if to_copy <= 0:
                        # max_buffered_speech reached
                        return

                    speech_buffer[
                        speech_buffer_index : speech_buffer_index + to_copy
                    ] = og_window_data
                    speech_buffer_index += og_window_size_samples

                if pub_speaking:
                    pub_speech_duration += window_duration
                    _copy_window()
                else:
                    pub_silence_duration += window_duration

                self._event_ch.send_nowait(
                    agents.vad.VADEvent(
                        type=agents.vad.VADEventType.INFERENCE_DONE,
                        samples_index=pub_current_sample,
                        silence_duration=pub_silence_duration,
                        speech_duration=pub_speech_duration,
                        probability=raw_prob,
                        inference_duration=inference_duration,
                        speaking=pub_speaking,
                    )
                )

                if raw_prob >= self._opts.activation_threshold:
                    speech_threshold_duration += window_duration
                    silence_threshold_duration = 0.0

                    if not pub_speaking:
                        _copy_window()

                        if speech_threshold_duration >= self._opts.min_speech_duration:
                            pub_speaking = True
                            pub_silence_duration = 0.0
                            pub_speech_duration = speech_threshold_duration

                            self._event_ch.send_nowait(
                                agents.vad.VADEvent(
                                    type=agents.vad.VADEventType.START_OF_SPEECH,
                                    samples_index=pub_current_sample,
                                    silence_duration=pub_silence_duration,
                                    speech_duration=pub_speech_duration,
                                    speaking=True,
                                )
                            )
                else:
                    silence_threshold_duration += window_duration
                    speech_threshold_duration = 0.0

                    if not pub_speaking:
                        speech_buffer_index = 0

                    if (
                        pub_speaking
                        and silence_threshold_duration
                        >= self._opts.min_silence_duration
                    ):
                        pub_speaking = False
                        pub_speech_duration = 0.0
                        pub_silence_duration = silence_threshold_duration

                        speech_data = speech_buffer[
                            :speech_buffer_index
                        ].tobytes()  # copy the data from speech_buffer

                        self._event_ch.send_nowait(
                            agents.vad.VADEvent(
                                type=agents.vad.VADEventType.END_OF_SPEECH,
                                samples_index=pub_current_sample,
                                silence_duration=pub_silence_duration,
                                speech_duration=pub_speech_duration,
                                frames=[
                                    rtc.AudioFrame(
                                        sample_rate=og_sample_rate,
                                        num_channels=1,
                                        samples_per_channel=speech_buffer_index,
                                        data=speech_data,
                                    )
                                ],
                                speaking=False,
                            )
                        )

                        speech_buffer_index = 0
