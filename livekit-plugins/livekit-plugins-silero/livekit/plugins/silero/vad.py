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

from __future__ import annotations, print_function

import asyncio
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np
import onnxruntime  # type: ignore
from livekit import agents, rtc
from livekit.agents import utils

from . import onnx_model
from .log import logger

SLOW_INFERENCE_THRESHOLD = 0.2  # late by 200ms


@dataclass
class _VADOptions:
    min_speech_duration: float
    min_silence_duration: float
    prefix_padding_duration: float
    max_buffered_speech: float
    activation_threshold: float
    sample_rate: int


class VAD(agents.vad.VAD):
    """
    Silero Voice Activity Detection (VAD) class.

    This class provides functionality to detect speech segments within audio data using the Silero VAD model.
    """

    @classmethod
    def load(
        cls,
        *,
        min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.25,
        prefix_padding_duration: float = 0.1,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.5,
        sample_rate: Literal[8000, 16000] = 16000,
        force_cpu: bool = True,
        # deprecated
        padding_duration: float | None = None,
    ) -> "VAD":
        """
        Load and initialize the Silero VAD model.

        This method loads the ONNX model and prepares it for inference. When options are not provided,
        sane defaults are used.

        **Note:**
            This method is blocking and may take time to load the model into memory.
            It is recommended to call this method inside your prewarm mechanism.

        **Example:**

            ```python
            def prewarm(proc: JobProcess):
                proc.userdata["vad"] = silero.VAD.load()


            async def entrypoint(ctx: JobContext):
                vad = (ctx.proc.userdata["vad"],)
                # your agent logic...


            if __name__ == "__main__":
                cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
            ```

        Args:
            min_speech_duration (float): Minimum duration of speech to start a new speech chunk.
            min_silence_duration (float): At the end of each speech, wait this duration before ending the speech.
            prefix_padding_duration (float): Duration of padding to add to the beginning of each speech chunk.
            max_buffered_speech (float): Maximum duration of speech to keep in the buffer (in seconds).
            activation_threshold (float): Threshold to consider a frame as speech.
            sample_rate (Literal[8000, 16000]): Sample rate for the inference (only 8KHz and 16KHz are supported).
            force_cpu (bool): Force the use of CPU for inference.
            padding_duration (float | None): **Deprecated**. Use `prefix_padding_duration` instead.

        Returns:
            VAD: An instance of the VAD class ready for streaming.

        Raises:
            ValueError: If an unsupported sample rate is provided.
        """
        if sample_rate not in onnx_model.SUPPORTED_SAMPLE_RATES:
            raise ValueError("Silero VAD only supports 8KHz and 16KHz sample rates")

        if padding_duration is not None:
            logger.warning(
                "padding_duration is deprecated and will be removed in 1.5.0, use prefix_padding_duration instead",
            )
            prefix_padding_duration = padding_duration

        session = onnx_model.new_inference_session(force_cpu)
        opts = _VADOptions(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding_duration,
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
        """
        Create a new VADStream for processing audio data.

        Returns:
            VADStream: A stream object for processing audio input and detecting speech.
        """
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

        self._extra_inference_time = 0.0

    @agents.utils.log_exceptions(logger=logger)
    async def _main_task(self):
        inference_f32_data = np.empty(self._model.window_size_samples, dtype=np.float32)

        # a copy is exposed to the user in END_OF_SPEECH
        speech_buffer: np.ndarray | None = None
        speech_buffer_max_reached = False
        speech_buffer_index: int = 0

        # "pub_" means public, these values are exposed to the users through events
        pub_speaking = False
        pub_speech_duration = 0.0
        pub_silence_duration = 0.0
        pub_current_sample = 0
        pub_timestamp = 0.0

        pub_sample_rate = 0
        pub_prefix_padding_samples = 0  # size in samples of padding data

        speech_threshold_duration = 0.0
        silence_threshold_duration = 0.0

        input_frames = []
        inference_frames = []
        resampler: rtc.AudioResampler | None = None

        # used to avoid drift when the sample_rate ratio is not an integer
        input_copy_remaining_fract = 0.0

        async for input_frame in self._input_ch:
            if not isinstance(input_frame, rtc.AudioFrame):
                continue  # ignore flush sentinel for now

            if not pub_sample_rate or speech_buffer is None:
                pub_sample_rate = input_frame.sample_rate

                # alloc the buffers now that we know the input sample rate
                pub_prefix_padding_samples = math.ceil(
                    self._opts.prefix_padding_duration * pub_sample_rate
                )

                speech_buffer = np.empty(
                    int(self._opts.max_buffered_speech * pub_sample_rate)
                    + int(self._opts.prefix_padding_duration * pub_sample_rate),
                    dtype=np.int16,
                )

                if pub_sample_rate != self._opts.sample_rate:
                    # resampling needed: the input sample rate isn't the same as the model's
                    # sample rate used for inference
                    resampler = rtc.AudioResampler(
                        input_rate=pub_sample_rate,
                        output_rate=self._opts.sample_rate,
                        quality=rtc.AudioResamplerQuality.QUICK,  # VAD doesn't need high quality
                    )

            elif pub_sample_rate != input_frame.sample_rate:
                logger.error("a frame with another sample rate was already pushed")
                continue

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
                if available_inference_samples < self._model.window_size_samples:
                    break  # not enough samples to run inference

                input_frame = utils.combine_frames(input_frames)
                inference_frame = utils.combine_frames(inference_frames)

                # convert data to f32
                np.divide(
                    inference_frame.data[: self._model.window_size_samples],
                    np.iinfo(np.int16).max,
                    out=inference_f32_data,
                    dtype=np.float32,
                )

                # run the inference
                p = await self._loop.run_in_executor(
                    self._executor, self._model, inference_f32_data
                )
                p = self._exp_filter.apply(exp=1.0, sample=p)

                window_duration = (
                    self._model.window_size_samples / self._opts.sample_rate
                )

                pub_current_sample += self._model.window_size_samples
                pub_timestamp += window_duration

                resampling_ratio = pub_sample_rate / self._model.sample_rate
                to_copy = (
                    self._model.window_size_samples * resampling_ratio
                    + input_copy_remaining_fract
                )
                to_copy_int = int(to_copy)
                input_copy_remaining_fract = to_copy - to_copy_int

                # copy the inference window to the speech buffer
                available_space = len(speech_buffer) - speech_buffer_index
                to_copy_buffer = min(self._model.window_size_samples, available_space)
                if to_copy_buffer > 0:
                    speech_buffer[
                        speech_buffer_index : speech_buffer_index + to_copy_buffer
                    ] = input_frame.data[:to_copy_buffer]
                    speech_buffer_index += to_copy_buffer
                elif not speech_buffer_max_reached:
                    # reached self._opts.max_buffered_speech (padding is included)
                    speech_buffer_max_reached = True
                    logger.warning(
                        "max_buffered_speech reached, ignoring further data for the current speech input"
                    )

                inference_duration = time.perf_counter() - start_time
                self._extra_inference_time = max(
                    0.0,
                    self._extra_inference_time + inference_duration - window_duration,
                )
                if inference_duration > SLOW_INFERENCE_THRESHOLD:
                    logger.warning(
                        "inference is slower than realtime",
                        extra={"delay": self._extra_inference_time},
                    )

                def _reset_write_cursor():
                    nonlocal speech_buffer_index, speech_buffer_max_reached
                    assert speech_buffer is not None

                    if speech_buffer_index <= pub_prefix_padding_samples:
                        return

                    padding_data = speech_buffer[
                        speech_buffer_index
                        - pub_prefix_padding_samples : speech_buffer_index
                    ]

                    speech_buffer[:pub_prefix_padding_samples] = padding_data
                    speech_buffer_index = pub_prefix_padding_samples
                    speech_buffer_max_reached = False

                def _copy_speech_buffer() -> rtc.AudioFrame:
                    # copy the data from speech_buffer
                    assert speech_buffer is not None
                    speech_data = speech_buffer[:speech_buffer_index].tobytes()

                    return rtc.AudioFrame(
                        sample_rate=pub_sample_rate,
                        num_channels=1,
                        samples_per_channel=speech_buffer_index,
                        data=speech_data,
                    )

                if pub_speaking:
                    pub_speech_duration += window_duration
                else:
                    pub_silence_duration += window_duration

                self._event_ch.send_nowait(
                    agents.vad.VADEvent(
                        type=agents.vad.VADEventType.INFERENCE_DONE,
                        samples_index=pub_current_sample,
                        timestamp=pub_timestamp,
                        silence_duration=pub_silence_duration,
                        speech_duration=pub_speech_duration,
                        probability=p,
                        inference_duration=inference_duration,
                        frames=[
                            rtc.AudioFrame(
                                data=input_frame.data[:to_copy_int].tobytes(),
                                sample_rate=pub_sample_rate,
                                num_channels=1,
                                samples_per_channel=to_copy_int,
                            )
                        ],
                        speaking=pub_speaking,
                    )
                )

                if p >= self._opts.activation_threshold:
                    speech_threshold_duration += window_duration
                    silence_threshold_duration = 0.0

                    if not pub_speaking:
                        if speech_threshold_duration >= self._opts.min_speech_duration:
                            pub_speaking = True
                            pub_silence_duration = 0.0
                            pub_speech_duration = speech_threshold_duration

                            self._event_ch.send_nowait(
                                agents.vad.VADEvent(
                                    type=agents.vad.VADEventType.START_OF_SPEECH,
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
                        and silence_threshold_duration
                        >= self._opts.min_silence_duration
                    ):
                        pub_speaking = False
                        pub_speech_duration = 0.0
                        pub_silence_duration = silence_threshold_duration

                        self._event_ch.send_nowait(
                            agents.vad.VADEvent(
                                type=agents.vad.VADEventType.END_OF_SPEECH,
                                samples_index=pub_current_sample,
                                timestamp=pub_timestamp,
                                silence_duration=pub_silence_duration,
                                speech_duration=pub_speech_duration,
                                frames=[_copy_speech_buffer()],
                                speaking=False,
                            )
                        )

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
                            sample_rate=pub_sample_rate,
                            num_channels=1,
                            samples_per_channel=len(data) // 2,
                        )
                    )

                if len(inference_frame.data) - self._model.window_size_samples > 0:
                    data = inference_frame.data[
                        self._model.window_size_samples :
                    ].tobytes()
                    inference_frames.append(
                        rtc.AudioFrame(
                            data=data,
                            sample_rate=self._opts.sample_rate,
                            num_channels=1,
                            samples_per_channel=len(data) // 2,
                        )
                    )
