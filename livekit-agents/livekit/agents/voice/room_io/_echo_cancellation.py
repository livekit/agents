from __future__ import annotations

import time
from collections import deque

import numpy as np
from numpy.typing import NDArray

from livekit import rtc

from ...log import logger

_ANALYSIS_RATE = 8000
_WINDOW_SAMPLES = 4800  # 600 ms of capture audio
_MAX_DELAY_SAMPLES = 20000  # 2.5 s at the analysis rate
_HISTORY_SECONDS = 3.5
_SUPPORTED_RATES = (8000, 16000, 24000, 32000, 48000)


class _DelayEstimator:
    """GCC-PHAT delay search, confirmed by three estimates within 30 ms of each other."""

    def __init__(self, sample_rate: int) -> None:
        self._capture_resampler = rtc.AudioResampler(sample_rate, _ANALYSIS_RATE)
        self._render_resampler = rtc.AudioResampler(sample_rate, _ANALYSIS_RATE)
        self._capture: deque[NDArray[np.float32]] = deque(maxlen=61)
        self._render: deque[NDArray[np.float32]] = deque(maxlen=311)
        self._pending: deque[float] = deque(maxlen=3)
        self._frames = 0

    def update(self, capture: rtc.AudioFrame, render: rtc.AudioFrame) -> int | None:
        for frame in self._capture_resampler.push(capture):
            self._capture.append(
                np.frombuffer(frame.data, np.int16).astype(np.float32) / np.float32(32768)
            )
        for frame in self._render_resampler.push(render):
            self._render.append(
                np.frombuffer(frame.data, np.int16).astype(np.float32) / np.float32(32768)
            )
        self._frames += 1
        if self._frames % 10 or not self._capture or not self._render:
            return None

        capture_pcm = np.concatenate(self._capture)[-_WINDOW_SAMPLES:]
        if len(capture_pcm) < _WINDOW_SAMPLES:
            return None
        history_samples = _WINDOW_SAMPLES + _MAX_DELAY_SAMPLES
        render_pcm = np.concatenate(self._render)[-history_samples:]
        render_pcm = np.pad(render_pcm, (history_samples - len(render_pcm), 0))
        if np.mean(capture_pcm**2) < 1e-6 or np.mean(render_pcm**2) < 3e-6:
            self._pending.clear()
            return None

        # Whitening can turn a tone's quantization noise into a false sharp peak.
        # Require energy across at least 50 Hz before trying to align speech.
        spectrum = np.abs(np.fft.rfft(capture_pcm * np.hanning(_WINDOW_SAMPLES))) ** 2
        spectrum = spectrum[150:2041]  # 250..3400 Hz in a 600 ms window
        if np.count_nonzero(spectrum > np.max(spectrum) * 0.01) < 30:
            self._pending.clear()
            return None

        fft_size = 1 << (len(render_pcm) + len(capture_pcm) - 1).bit_length()
        cross = np.fft.rfft(render_pcm, fft_size) * np.conj(np.fft.rfft(capture_pcm, fft_size))
        cross /= np.abs(cross) + 1e-10
        frequencies = np.fft.rfftfreq(fft_size, 1 / _ANALYSIS_RATE)
        cross[(frequencies < 250) | (frequencies > 3400)] = 0
        correlation = np.abs(np.fft.irfft(cross, fft_size))[: _MAX_DELAY_SAMPLES + 1]

        power = np.concatenate(([0.0], np.cumsum(render_pcm.astype(np.float64) ** 2)))
        window_power = (power[_WINDOW_SAMPLES:] - power[:-_WINDOW_SAMPLES]) / _WINDOW_SAMPLES
        candidates = np.where(window_power > 1e-6, correlation, 0)
        peak = int(np.argmax(candidates))
        quality = candidates[peak] / (np.median(correlation) + 1e-12)
        other_peaks = candidates.copy()
        radius = _ANALYSIS_RATE * 30 // 1000
        other_peaks[max(0, peak - radius) : peak + radius + 1] = 0
        ambiguity = candidates[peak] / (np.max(other_peaks) + 1e-12)
        delay_ms = (_MAX_DELAY_SAMPLES - peak) * 1000 / _ANALYSIS_RATE
        if quality < 10 or ambiguity < 1.4 or not 20 < delay_ms < 2480:
            self._pending.clear()
            return None

        if self._pending and abs(delay_ms - float(np.median(self._pending))) > 30:
            self._pending.clear()
        self._pending.append(delay_ms)
        if len(self._pending) < 3:
            return None

        # Leave a short residual delay for AEC3 to track without moving the bulk delay line.
        return max(0, int((float(np.median(self._pending)) - 80) // 10) * 10)


class _RenderHistory:
    """Bounded mono PCM on the local playout timeline; unwritten intervals are silence."""

    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._frames: deque[tuple[int, NDArray[np.int16]]] = deque()

    def push(self, pcm: NDArray[np.int16], *, started_at: float) -> None:
        start = round(started_at * self._sample_rate)
        self._frames.append((start, pcm.copy()))
        cutoff = start + len(pcm) - round(_HISTORY_SECONDS * self._sample_rate)
        while self._frames and self._frames[0][0] + len(self._frames[0][1]) <= cutoff:
            self._frames.popleft()

    def read(self, *, started_at: float, samples: int) -> NDArray[np.int16]:
        start = round(started_at * self._sample_rate)
        result = np.zeros(samples, dtype=np.int16)
        for frame_start, pcm in self._frames:
            lo, hi = max(start, frame_start), min(start + samples, frame_start + len(pcm))
            if hi > lo:
                result[lo - start : hi - start] = pcm[lo - frame_start : hi - frame_start]
        return result

    def discard_after(self, timestamp: float) -> None:
        stop = round(timestamp * self._sample_rate)
        while self._frames and self._frames[-1][0] >= stop:
            self._frames.pop()
        if self._frames:
            start, pcm = self._frames[-1]
            if start + len(pcm) > stop:
                self._frames[-1] = (start, pcm[: stop - start])


class _AdaptiveEchoCanceller:
    """Align room output with capture audio before applying WebRTC AEC.

    All timestamps use the same monotonic clock. Capture audio passes through until
    the delay estimate is confident. Render history survives capture resets and pauses
    because already-played audio can still echo back after playback stops.
    """

    def __init__(self, *, sample_rate: int) -> None:
        if sample_rate not in _SUPPORTED_RATES:
            raise ValueError(f"echo cancellation requires a sample rate in {_SUPPORTED_RATES}")
        self._sample_rate = sample_rate
        self._frame_samples = sample_rate // 100
        self._render = _RenderHistory(sample_rate)
        self._render_resampler: rtc.AudioResampler | None = None
        self._render_rate: int | None = None
        self._render_end: float | None = None
        self._render_cursor: float = 0
        self.reset_capture()

    @property
    def bulk_delay_ms(self) -> int | None:
        return self._bulk_delay_ms

    def reset_capture(self) -> None:
        self._estimator = _DelayEstimator(self._sample_rate)
        self._apm: rtc.AudioProcessingModule | None = None
        self._bulk_delay_ms: int | None = None
        self._capture_end: float | None = None

    def push_render(self, frame: rtc.AudioFrame, *, started_at: float) -> None:
        if (
            self._render_end is None
            or abs(started_at - self._render_end) > 0.01
            or frame.sample_rate != self._render_rate
        ):
            self._flush_render()
            self._render_resampler = (
                rtc.AudioResampler(frame.sample_rate, self._sample_rate)
                if frame.sample_rate != self._sample_rate
                else None
            )
            self._render_cursor = started_at
            self._render_rate = frame.sample_rate
        self._render_end = started_at + frame.duration

        if frame.num_channels != 1:
            pcm = np.frombuffer(frame.data, np.int16).reshape(-1, frame.num_channels)
            mono = np.mean(pcm, axis=1).astype(np.int16)
            frame = rtc.AudioFrame(mono.tobytes(), frame.sample_rate, 1, len(mono))
        frames = self._render_resampler.push(frame) if self._render_resampler else [frame]
        for converted in frames:
            self._render.push(
                np.frombuffer(converted.data, np.int16), started_at=self._render_cursor
            )
            self._render_cursor += converted.duration

    def discard_render_after(self, timestamp: float) -> None:
        self._flush_render()
        self._render.discard_after(timestamp)
        self._render_end = None

    def _flush_render(self) -> None:
        if self._render_resampler is not None:
            for frame in self._render_resampler.flush():
                self._render.push(
                    np.frombuffer(frame.data, np.int16), started_at=self._render_cursor
                )
                self._render_cursor += frame.duration
        self._render_resampler = None

    def process_capture(self, frame: rtc.AudioFrame, *, started_at: float) -> rtc.AudioFrame:
        if (
            frame.sample_rate != self._sample_rate
            or frame.num_channels != 1
            or frame.samples_per_channel % self._frame_samples
        ):
            # A partial final frame cannot be passed to APM's 10 ms native interface.
            self.reset_capture()
            return frame
        if self._capture_end is not None:
            if abs(started_at - self._capture_end) > 0.1:
                self.reset_capture()
            else:
                started_at = self._capture_end

        pcm = np.frombuffer(frame.data, np.int16)
        output = pcm.copy()
        for offset in range(0, len(pcm), self._frame_samples):
            timestamp = started_at + offset / self._sample_rate
            capture = rtc.AudioFrame(
                pcm[offset : offset + self._frame_samples].tobytes(),
                self._sample_rate,
                1,
                self._frame_samples,
            )
            render = self._render.read(started_at=timestamp, samples=self._frame_samples)
            reference = rtc.AudioFrame(render.tobytes(), self._sample_rate, 1, self._frame_samples)
            delay = self._estimator.update(capture, reference)
            # Shorter paths need an earlier adjustment to keep the residual delay positive.
            if delay is not None and (
                self._bulk_delay_ms is None
                or delay - self._bulk_delay_ms >= 100
                or self._bulk_delay_ms - delay >= 60
            ):
                self._bulk_delay_ms = delay
                self._apm = rtc.AudioProcessingModule(echo_cancellation=True)
                logger.debug("echo reference aligned", extra={"bulk_delay_ms": delay})

            if self._apm is not None and self._bulk_delay_ms is not None:
                render = self._render.read(
                    started_at=timestamp - self._bulk_delay_ms / 1000,
                    samples=self._frame_samples,
                )
                reference = rtc.AudioFrame(
                    render.tobytes(), self._sample_rate, 1, self._frame_samples
                )
                self._apm.process_reverse_stream(reference)
                self._apm.set_stream_delay_ms(0)
                self._apm.process_stream(capture)
                output[offset : offset + self._frame_samples] = np.frombuffer(
                    capture.data, np.int16
                )
        self._capture_end = started_at + frame.duration
        if self._apm is None:
            return frame
        return rtc.AudioFrame(
            output.tobytes(), self._sample_rate, 1, len(output), userdata=frame.userdata
        )


class _EchoCancellationProcessor(rtc.FrameProcessor[rtc.AudioFrame]):
    """AEC before NC; the input stream retains ownership of the downstream processor."""

    def __init__(
        self,
        echo_canceller: _AdaptiveEchoCanceller,
        next_processor: rtc.FrameProcessor[rtc.AudioFrame] | None,
    ) -> None:
        self._echo_canceller = echo_canceller
        self._next = next_processor
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        processed = self._echo_canceller.process_capture(
            frame, started_at=time.monotonic() - frame.duration
        )
        if self._next is not None and self._next.enabled:
            return self._next._process(processed)
        return processed

    def _on_stream_info_updated(
        self, *, room_name: str, participant_identity: str, publication_sid: str
    ) -> None:
        if self._next is not None:
            self._next._on_stream_info_updated(
                room_name=room_name,
                participant_identity=participant_identity,
                publication_sid=publication_sid,
            )

    def _on_stream_info_cleared(self) -> None:
        if self._next is not None:
            self._next._on_stream_info_cleared()

    def _on_credentials_updated(self, *, token: str, url: str) -> None:
        if self._next is not None:
            self._next._on_credentials_updated(token=token, url=url)

    def _on_credentials_cleared(self) -> None:
        if self._next is not None:
            self._next._on_credentials_cleared()

    def _close(self) -> None:
        # A replaced RTC stream can deliver queued frames and metadata until it closes.
        self._enabled = False
        self._next = None
