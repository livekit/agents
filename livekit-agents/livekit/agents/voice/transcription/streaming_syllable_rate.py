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
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import librosa
import numpy as np

from livekit import rtc
from livekit.agents import utils
from livekit.agents.log import logger

logging.getLogger("numba").setLevel(logging.WARNING)


@dataclass
class _SyllableRateOptions:
    window_duration: float  # Window length in seconds for syllable rate calculation
    hop_duration: float  # How frequently to update the syllable rate
    sample_rate: int  # Sample rate for processing


@dataclass
class SyllableRateEvent:
    """
    Represents an event from the Syllable Rate Detector.
    """

    timestamp: float
    speaking: bool
    syllable_count: int  # Number of syllables in the window
    syllable_rate: float  # Syllables per second
    samples_index: int = 0
    inference_duration: float = 0.0


class SyllableRateDetector:
    """
    Syllable Rate Detector class.

    This class provides functionality to detect syllable rates within streaming audio data.
    """

    def __init__(
        self,
        *,
        window_size: float = 1.0,
        step_size: float = 0.2,
        sample_rate: int = 44100,
    ) -> None:
        super().__init__()
        self._opts = _SyllableRateOptions(
            window_duration=window_size,
            hop_duration=step_size,
            sample_rate=sample_rate,
        )

    def stream(self) -> SyllableRateStream:
        return SyllableRateStream(self, self._opts)


class SyllableRateStream:
    class _FlushSentinel:
        pass

    def __init__(self, detector: SyllableRateDetector, opts: _SyllableRateOptions) -> None:
        self._detector = detector
        self._opts = opts

        self._loop = asyncio.get_event_loop()
        self._executor = ThreadPoolExecutor(max_workers=1)

        self._input_ch = utils.aio.Chan[rtc.AudioFrame | SyllableRateStream._FlushSentinel]()
        self._event_ch = utils.aio.Chan[SyllableRateEvent]()

        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._executor.shutdown(wait=False))
        self._task.add_done_callback(lambda _: self._event_ch.close())

        self._input_sample_rate = 0
        self._window_size_samples = int(self._opts.window_duration * self._opts.sample_rate)
        self._hop_size_samples = int(self._opts.hop_duration * self._opts.sample_rate)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self):
        inference_f32_data = np.empty(self._window_size_samples, dtype=np.float32)

        pub_timestamp = self._opts.window_duration / 2
        inference_frames = []
        resampler = None

        async for input_frame in self._input_ch:
            if not isinstance(input_frame, rtc.AudioFrame):
                # available_samples = sum(frame.samples_per_channel for frame in inference_frames)
                # if available_samples > self._window_size_samples * 0.5:
                #     frame = utils.combine_frames(inference_frames)
                #     frame_f32_data = np.divide(frame.data, np.iinfo(np.int16).max, dtype=np.float32)
                #     pub_timestamp += frame.duration
                #     self._event_ch.send_nowait(
                #         SyllableRateEvent(
                #             timestamp=pub_timestamp,
                #             speaking=not self._detect_silence(frame_f32_data),
                #             syllable_count=0,  # skip syllable count for last frame
                #             syllable_rate=0,
                #         )
                #     )
                #     inference_frames = []
                continue

            if not self._input_sample_rate:
                self._input_sample_rate = input_frame.sample_rate
                if self._input_sample_rate != self._opts.sample_rate:
                    resampler = rtc.AudioResampler(
                        input_rate=self._input_sample_rate,
                        output_rate=self._opts.sample_rate,
                        num_channels=1,
                        quality=rtc.AudioResamplerQuality.MEDIUM,
                    )
            elif self._input_sample_rate != input_frame.sample_rate:
                logger.error(
                    "a frame with different sample rate was pushed",
                    extra={
                        "sample_rate": input_frame.sample_rate,
                        "expected_sample_rate": self._input_sample_rate,
                    },
                )
                continue

            if resampler is not None:
                inference_frames.extend(resampler.push(input_frame))
            else:
                inference_frames.append(input_frame)

            while True:
                available_samples = sum(frame.samples_per_channel for frame in inference_frames)
                if available_samples < self._window_size_samples:
                    break

                inference_frame = utils.combine_frames(inference_frames)
                np.divide(
                    inference_frame.data[: self._window_size_samples],
                    np.iinfo(np.int16).max,
                    out=inference_f32_data,
                    dtype=np.float32,
                )

                # run the inference
                # speaking, syllable_count = await self._loop.run_in_executor(
                #     self._executor,
                #     self._compute_syllable_count,
                #     inference_f32_data,
                #     self._opts.sample_rate,
                # )
                speaking = not self._detect_silence(inference_f32_data)
                if not speaking:
                    syllable_count = 0
                else:
                    syllable_count = self._compute_syllable_count(
                        inference_f32_data, self._opts.sample_rate
                    )
                self._event_ch.send_nowait(
                    SyllableRateEvent(
                        timestamp=pub_timestamp,
                        speaking=speaking,
                        syllable_count=syllable_count,
                        syllable_rate=syllable_count / self._opts.window_duration,
                    )
                )

                # move the window forward
                pub_timestamp += self._opts.hop_duration
                if len(inference_frame.data) - self._hop_size_samples > 0:
                    data = inference_frame.data[self._hop_size_samples :]
                    inference_frames = [
                        rtc.AudioFrame(
                            data=data,
                            sample_rate=inference_frame.sample_rate,
                            num_channels=1,
                            samples_per_channel=len(data) // 2,
                        )
                    ]
        print("syllable rate stream end")

    def _compute_syllable_count(self, audio: np.ndarray, sample_rate: int) -> int:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=256,
            # pre_max=20,
            # post_max=20,
            delta=0.3,
            wait=10,
        )
        return len(onsets)

    def _detect_silence(self, audio: np.ndarray) -> bool:
        overall_rms = np.sqrt(np.mean(audio**2))
        silence_threshold = 0.005

        if overall_rms < silence_threshold:
            return True

        if overall_rms > silence_threshold * 10:
            return False

        # for borderline cases, check if the right half of the window is silent
        half_window = len(audio) // 2
        if np.mean(audio[half_window:] ** 2) < silence_threshold**2:
            return True

        return False

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Push audio frame for syllable rate detection"""
        self._input_ch.send_nowait(frame)

    def flush(self) -> None:
        """Mark the end of the current segment"""
        self._input_ch.send_nowait(self._FlushSentinel())

    def end_input(self) -> None:
        """Mark the end of input, no more audio will be pushed"""
        self.flush()
        self._input_ch.close()

    async def aclose(self) -> None:
        """Close this stream immediately"""
        self._input_ch.close()
        await utils.aio.cancel_and_wait(self._task)
        self._event_ch.close()

    def __aiter__(self):
        return self._event_ch
