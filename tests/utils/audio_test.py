"""Audio test helpers: known-signal sources and simple energy monitoring.

Designed to be reused across E2E tests that need to assert "audio is still
flowing" without depending on speech recognition or transcription. The
canonical pattern is:

1. Publisher pushes a known steady tone via :class:`SineToneSource`.
2. Subscriber wraps the remote track with :class:`AudioEnergyMonitor` and
   calls :meth:`AudioEnergyMonitor.wait_for_audio` to assert the tone's
   energy is observed above the noise floor.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import numpy as np

from livekit import rtc

__all__ = [
    "SineToneSource",
    "rms",
    "AudioEnergyMonitor",
]


_DEFAULT_SAMPLE_RATE = 48000
_DEFAULT_FREQUENCY = 440.0  # A4 — easy to hear if you debug locally
_FRAME_DURATION_MS = 20  # standard Opus framing


def rms(samples: np.ndarray) -> float:
    """Root-mean-square energy for an int16 PCM buffer, scaled to [0, 1]."""
    if samples.size == 0:
        return 0.0
    s = samples.astype(np.float64) / 32768.0
    return float(np.sqrt(np.mean(s * s)))


class SineToneSource:
    """Steadily emits a sine wave into an `rtc.AudioSource` via a background task.

    Intended as a known-signal source for E2E tests. Construct, optionally
    wrap into a `LocalAudioTrack`, then call :meth:`start` to begin pushing
    frames at the natural cadence (one 20ms frame every 20ms).
    """

    def __init__(
        self,
        *,
        frequency: float = _DEFAULT_FREQUENCY,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        amplitude: float = 0.5,
        num_channels: int = 1,
    ) -> None:
        self.source = rtc.AudioSource(sample_rate, num_channels)
        self._frequency = frequency
        self._sample_rate = sample_rate
        self._amplitude = amplitude
        self._num_channels = num_channels
        self._samples_per_frame = sample_rate * _FRAME_DURATION_MS // 1000
        self._phase = 0.0
        self._task: asyncio.Task[None] | None = None

    def make_track(self, name: str = "test_tone") -> rtc.LocalAudioTrack:
        return rtc.LocalAudioTrack.create_audio_track(name, self.source)

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        await self.source.aclose()

    async def _run(self) -> None:
        # Prebuild one period worth of samples then loop the phase forward.
        omega = 2.0 * math.pi * self._frequency / self._sample_rate
        next_tick = asyncio.get_event_loop().time()
        while True:
            i = np.arange(self._samples_per_frame, dtype=np.float64)
            wave = np.sin(self._phase + omega * i) * self._amplitude
            self._phase = (self._phase + omega * self._samples_per_frame) % (2.0 * math.pi)
            samples = (wave * 32767.0).astype(np.int16)
            if self._num_channels > 1:
                samples = np.repeat(samples[:, None], self._num_channels, axis=1).reshape(-1)

            frame = rtc.AudioFrame(
                data=samples.tobytes(),
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
                samples_per_channel=self._samples_per_frame,
            )
            await self.source.capture_frame(frame)

            next_tick += _FRAME_DURATION_MS / 1000.0
            delay = next_tick - asyncio.get_event_loop().time()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                # Behind schedule (e.g. just resumed from a pause); resync.
                next_tick = asyncio.get_event_loop().time()


class AudioEnergyMonitor:
    """Subscribes to a remote audio track and reports rolling RMS levels.

    Use as an async context manager around a `RemoteAudioTrack`::

        async with AudioEnergyMonitor.watch(audio_track) as mon:
            await mon.wait_for_audio(min_rms=0.05, timeout=2.0)
            # ... do something that should keep audio flowing ...
            await mon.assert_audio_continuous(min_rms=0.05, duration=1.0)
    """

    def __init__(self, track: rtc.RemoteAudioTrack) -> None:
        self._track = track
        self._stream = rtc.AudioStream(track)
        self._latest_rms: float = 0.0
        self._frame_count: int = 0
        self._task: asyncio.Task[None] | None = None

    @classmethod
    @asynccontextmanager
    async def watch(cls, track: rtc.RemoteAudioTrack) -> AsyncIterator[AudioEnergyMonitor]:
        mon = cls(track)
        await mon.start()
        try:
            yield mon
        finally:
            await mon.aclose()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        await self._stream.aclose()

    @property
    def current_rms(self) -> float:
        return self._latest_rms

    @property
    def frame_count(self) -> int:
        return self._frame_count

    async def wait_for_audio(self, *, min_rms: float = 0.01, timeout: float = 5.0) -> None:
        """Resolve when rolling RMS exceeds `min_rms`, or raise on timeout."""
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            if self._latest_rms >= min_rms:
                return
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"audio energy never reached {min_rms} (last RMS={self._latest_rms}, "
                    f"frames seen={self._frame_count})"
                )
            await asyncio.sleep(0.05)

    async def assert_audio_continuous(
        self, *, min_rms: float = 0.01, duration: float = 1.0, sample_interval: float = 0.05
    ) -> None:
        """Assert RMS stays above `min_rms` for `duration` seconds.

        Raises `AssertionError` if RMS drops below threshold at any sample.
        """
        end = asyncio.get_event_loop().time() + duration
        seen_above_threshold = False
        while asyncio.get_event_loop().time() < end:
            if self._latest_rms >= min_rms:
                seen_above_threshold = True
            else:
                if seen_above_threshold:
                    raise AssertionError(
                        f"audio energy dropped to {self._latest_rms} below {min_rms}"
                    )
            await asyncio.sleep(sample_interval)
        if not seen_above_threshold:
            raise AssertionError(
                f"audio energy never reached {min_rms} during the {duration}s window"
            )

    async def _run(self) -> None:
        async for ev in self._stream:
            frame = ev.frame
            samples = np.frombuffer(frame.data, dtype=np.int16)
            self._latest_rms = rms(samples)
            self._frame_count += 1
