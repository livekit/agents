from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Union

import numpy as np

from livekit import rtc

from ...log import logger
from ...utils import aio, log_exceptions


@dataclass
class _SpeakingRateDetectionOptions:
    window_duration: float
    "window size in seconds"
    step_size: float
    "step size in seconds"
    sample_rate: int | None
    "inference sample rate, if None, use the sample rate of the input frame"
    _silence_threshold: float = 0.005
    "silence threshold for silence detection on audio RMS"


@dataclass
class SpeakingRateEvent:
    timestamp: float
    speaking: bool
    speaking_rate: float


class SpeakingRateDetector:
    def __init__(
        self,
        *,
        window_size: float = 1.0,
        step_size: float = 0.1,
        sample_rate: int | None = None,
    ) -> None:
        super().__init__()
        self._opts = _SpeakingRateDetectionOptions(
            window_duration=window_size,
            step_size=step_size,
            sample_rate=sample_rate,
        )

    def stream(self) -> SpeakingRateStream:
        return SpeakingRateStream(self, self._opts)


class SpeakingRateStream:
    class _FlushSentinel:
        pass

    def __init__(self, detector: SpeakingRateDetector, opts: _SpeakingRateDetectionOptions) -> None:
        self._detector = detector
        self._opts = opts

        self._input_ch = aio.Chan[Union[rtc.AudioFrame, SpeakingRateStream._FlushSentinel]]()
        self._event_ch = aio.Chan[SpeakingRateEvent]()

        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

        self._input_sample_rate = 0
        self._window_size_samples = 0
        self._step_size_samples = 0

    @log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        _inference_sample_rate = 0
        inference_f32_data = np.empty(0, dtype=np.float32)

        pub_timestamp = self._opts.window_duration / 2
        inference_frames: list[rtc.AudioFrame] = []
        resampler: rtc.AudioResampler | None = None

        async for input_frame in self._input_ch:
            if not isinstance(input_frame, rtc.AudioFrame):
                # estimate the speech rate for the last frame
                available_samples = sum(frame.samples_per_channel for frame in inference_frames)
                if available_samples > self._window_size_samples * 0.5:
                    frame = rtc.combine_audio_frames(inference_frames)
                    frame_f32_data = np.divide(frame.data, np.iinfo(np.int16).max, dtype=np.float32)

                    sr = self._compute_speaking_rate(frame_f32_data, _inference_sample_rate)
                    pub_timestamp += frame.duration
                    self._event_ch.send_nowait(
                        SpeakingRateEvent(
                            timestamp=pub_timestamp,
                            speaking=sr > 0,
                            speaking_rate=sr,
                        )
                    )
                inference_frames = []
                continue

            # resample the input frame if necessary
            if not self._input_sample_rate:
                self._input_sample_rate = input_frame.sample_rate
                _inference_sample_rate = self._opts.sample_rate or self._input_sample_rate

                self._window_size_samples = int(self._opts.window_duration * _inference_sample_rate)
                self._step_size_samples = int(self._opts.step_size * _inference_sample_rate)
                inference_f32_data = np.empty(self._window_size_samples, dtype=np.float32)

                if self._input_sample_rate != _inference_sample_rate:
                    resampler = rtc.AudioResampler(
                        input_rate=self._input_sample_rate,
                        output_rate=_inference_sample_rate,
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

                inference_frame = rtc.combine_audio_frames(inference_frames)
                np.divide(
                    inference_frame.data[: self._window_size_samples],
                    np.iinfo(np.int16).max,
                    out=inference_f32_data,
                    dtype=np.float32,
                )

                # run the inference
                sr = self._compute_speaking_rate(inference_f32_data, _inference_sample_rate)
                self._event_ch.send_nowait(
                    SpeakingRateEvent(
                        timestamp=pub_timestamp,
                        speaking=sr > 0,
                        speaking_rate=sr,
                    )
                )

                # move the window forward by the hop size
                pub_timestamp += self._opts.step_size
                if len(inference_frame.data) - self._step_size_samples > 0:
                    data = inference_frame.data[self._step_size_samples :]
                    inference_frames = [
                        rtc.AudioFrame(
                            data=data,
                            sample_rate=inference_frame.sample_rate,
                            num_channels=1,
                            samples_per_channel=len(data) // 2,
                        )
                    ]

    def _compute_speaking_rate(
        self, audio: np.ndarray[tuple[int], np.dtype[np.float32]], sample_rate: int
    ) -> float:
        """
        Compute the speaking rate of the audio using the selected method
        """
        silence_threshold = self._opts._silence_threshold

        audio_sq = audio**2
        # check if the audio is silent
        overall_rms = np.sqrt(np.mean(audio_sq))
        if overall_rms < silence_threshold:
            return 0.0

        # or if the tail of the audio is silent
        tail_audio_sq = audio_sq[int(len(audio_sq) * 0.7) :]
        if len(tail_audio_sq) > 0 and np.sqrt(np.mean(tail_audio_sq)) < silence_threshold * 0.5:
            return 0.0

        return self._spectral_flux(audio, sample_rate)

    def _stft(
        self,
        audio: np.ndarray[tuple[int], np.dtype[np.float32]],
        frame_length: int,
        hop_length: int,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        num_frames = (len(audio) - frame_length) // hop_length + 1
        result = np.zeros((frame_length // 2 + 1, num_frames), dtype=np.complex128)

        window = np.hanning(frame_length)
        scale_factor = 1.0 / np.sqrt(np.sum(window**2))
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = audio[start:end]
            windowed = frame * window

            # perform fft and scale
            fft_result = np.fft.rfft(windowed)
            result[:, i] = fft_result * scale_factor

        return result

    def _spectral_flux(
        self, audio: np.ndarray[tuple[int], np.dtype[np.float32]], sample_rate: int
    ) -> float:
        """
        Calculate speaking rate based on spectral flux.
        Higher spectral flux correlates with more rapid speech articulation.
        """
        # Parameters
        frame_length = int(sample_rate * 0.025)  # 25ms
        hop_length = frame_length // 2  # 50% overlap

        Zxx = self._stft(audio, frame_length, hop_length)

        # calculate spectral flux (sum of spectral magnitude changes between frames)
        spectral_magnitudes = np.abs(Zxx)
        spectral_flux_values = []

        for i in range(1, spectral_magnitudes.shape[1]):
            # l1 norm of difference between consecutive spectral frames
            flux = np.sum(np.abs(spectral_magnitudes[:, i] - spectral_magnitudes[:, i - 1]))
            spectral_flux_values.append(flux)

        if not spectral_flux_values:
            return 0.0

        avg_flux = float(np.mean(spectral_flux_values))
        return avg_flux

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
        await aio.cancel_and_wait(self._task)
        self._event_ch.close()

    def __aiter__(self) -> AsyncIterator[SpeakingRateEvent]:
        return self._event_ch
