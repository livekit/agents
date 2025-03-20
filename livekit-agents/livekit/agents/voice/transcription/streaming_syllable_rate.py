from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import librosa
import numpy as np

from livekit import rtc
from livekit.agents import utils
from livekit.agents.log import logger

logging.getLogger("numba").setLevel(logging.WARNING)


@dataclass
class _SyllableOptions:
    window_duration: float
    "window size in seconds"
    hop_duration: float
    "hop size in seconds"
    sample_rate: int
    "inference sample rate"
    enabled: bool = True
    "enable/disable the syllable detector, if disabled, only estimate speaking"
    _hop_length: int = 256
    "hop length for onset detection"
    _delta: float = 0.3
    "delta for onset detection"
    _wait: int = 10
    "wait for onset detection"
    _silence_threshold: float = 0.005
    "silence threshold for silence detection on audio RMS"


@dataclass
class SyllableEvent:
    timestamp: float
    speaking: bool
    syllable_count: int
    syllable_rate: float


class SyllableDetector:
    def __init__(
        self,
        *,
        window_size: float = 1.0,
        step_size: float = 0.2,
        sample_rate: int = 44100,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self._opts = _SyllableOptions(
            window_duration=window_size,
            hop_duration=step_size,
            sample_rate=sample_rate,
            enabled=enabled,
        )

    def stream(self) -> SyllableStream:
        return SyllableStream(self, self._opts)


class SyllableStream:
    class _FlushSentinel:
        pass

    def __init__(self, detector: SyllableDetector, opts: _SyllableOptions) -> None:
        self._detector = detector
        self._opts = opts

        self._input_ch = utils.aio.Chan[rtc.AudioFrame | SyllableStream._FlushSentinel]()
        self._event_ch = utils.aio.Chan[SyllableEvent]()

        self._task = asyncio.create_task(self._main_task())
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
                # estimate the syllable rate for the last frame
                available_samples = sum(frame.samples_per_channel for frame in inference_frames)
                if available_samples > self._window_size_samples * 0.5:
                    frame = utils.combine_frames(inference_frames)
                    frame_f32_data = np.divide(frame.data, np.iinfo(np.int16).max, dtype=np.float32)
                    speaking = not self._detect_silence(frame_f32_data)
                    syllable_count = (
                        self._compute_syllable_count(frame_f32_data, self._opts.sample_rate)
                        if speaking
                        else 0
                    )
                    pub_timestamp += frame.duration
                    self._event_ch.send_nowait(
                        SyllableEvent(
                            timestamp=pub_timestamp,
                            speaking=speaking,
                            syllable_count=syllable_count,
                            syllable_rate=syllable_count / frame.duration,
                        )
                    )
                inference_frames = []
                continue

            # resample the input frame if necessary
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
                speaking = not self._detect_silence(inference_f32_data)
                if not speaking:
                    syllable_count = 0
                else:
                    syllable_count = self._compute_syllable_count(
                        inference_f32_data, self._opts.sample_rate
                    )
                self._event_ch.send_nowait(
                    SyllableEvent(
                        timestamp=pub_timestamp,
                        speaking=speaking,
                        syllable_count=syllable_count,
                        syllable_rate=syllable_count / self._opts.window_duration,
                    )
                )

                # move the window forward by the hop size
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

    def _compute_syllable_count(self, audio: np.ndarray, sample_rate: int) -> int:
        if not self._opts.enabled:
            return 1

        # TODO(long): this tooks 200M of memory
        onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=self._opts._hop_length,
            delta=self._opts._delta,
            wait=self._opts._wait,
        )
        return len(onsets)

    def _detect_silence(self, audio: np.ndarray) -> bool:
        overall_rms = np.sqrt(np.mean(audio**2))
        silence_threshold = self._opts._silence_threshold

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
