from __future__ import annotations

from collections import deque

import numpy as np

from livekit import rtc

from ...log import logger
from ...utils.audio_gate import AudioGate, frame_rms
from .activity import ActivityTimeline, TrackActivity


class _TrackInput:
    """One participant's audio, cut into the mixer's blocks."""

    def __init__(
        self,
        identity: str,
        *,
        track_sid: str | None,
        gate: AudioGate,
        sample_rate: int,
        block_size: int,
        max_blocks: int,
    ) -> None:
        self.identity = identity
        self.track_sid = track_sid
        self.gate = gate
        self.ended = False
        """No more audio is coming; the input leaves the mix once what it sent has played."""

        self._sample_rate = sample_rate
        self._block_size = block_size
        self._max_blocks = max_blocks
        self._pending = np.zeros(0, dtype=np.int16)
        self._blocks: deque[np.ndarray] = deque()
        self._resampler: rtc.AudioResampler | None = None
        self._resampler_rate = 0
        self._overrun_logged = False
        # the backlog over the current window of mixes: at its start, and at its lowest
        self._window_mixes = 0
        self._window_first = 0
        self._window_low = 0

    @property
    def ready(self) -> bool:
        return bool(self._blocks)

    @property
    def queued(self) -> int:
        """Number of complete blocks waiting to be mixed."""
        return len(self._blocks)

    def push(self, frame: rtc.AudioFrame) -> None:
        if frame.num_channels > 1:
            samples = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
            frame = rtc.AudioFrame(
                data=samples.mean(axis=1).astype(np.int16).tobytes(),
                sample_rate=frame.sample_rate,
                num_channels=1,
                samples_per_channel=frame.samples_per_channel,
            )

        if frame.sample_rate == self._sample_rate:
            self._append(np.frombuffer(frame.data, dtype=np.int16))
            return

        if self._resampler is None or self._resampler_rate != frame.sample_rate:
            self._resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate, output_rate=self._sample_rate, num_channels=1
            )
            self._resampler_rate = frame.sample_rate
        for resampled in self._resampler.push(frame):
            self._append(np.frombuffer(resampled.data, dtype=np.int16))

    def pop(self) -> np.ndarray | None:
        return self._blocks.popleft() if self._blocks else None

    def settle(self, max_backlog: int, window: int) -> None:
        """Drop the oldest blocks past ``max_backlog`` when more stayed queued for ``window``
        mixes without shrinking.

        A long frame or a burst queues a lot at once but drains; audio that came in late and is
        followed in real time never does, and is mixed that far behind everyone else's for good.
        """
        queued = len(self._blocks)
        if not self._window_mixes:
            self._window_first = self._window_low = queued
        self._window_low = min(self._window_low, queued)
        self._window_mixes += 1
        if self._window_mixes < window:
            return

        self._window_mixes = 0
        if self._window_low > max_backlog and queued >= self._window_first:
            while len(self._blocks) > max_backlog:
                self._blocks.popleft()

    def _append(self, samples: np.ndarray) -> None:
        self._pending = np.concatenate([self._pending, samples])
        while self._pending.size >= self._block_size:
            self._blocks.append(self._pending[: self._block_size])
            self._pending = self._pending[self._block_size :]

        # a participant sending faster than real time would otherwise delay itself without bound
        while len(self._blocks) > self._max_blocks:
            self._blocks.popleft()
            if not self._overrun_logged:
                self._overrun_logged = True
                logger.warning(
                    "transcription input is ahead of the mix, dropping its oldest audio",
                    extra={"participant": self.identity},
                )


class _Mixer:
    """Sums every participant into one block at a time and records who was audible in it."""

    def __init__(
        self,
        *,
        sample_rate: int,
        block_size: int,
        timeline: ActivityTimeline,
        max_backlog: int,
        backlog_window: int,
    ) -> None:
        """Create a mixer.

        Args:
            max_backlog: How many blocks a participant can keep queued without draining.
            backlog_window: How many mixes in a row a backlog must last before it is dropped.
        """
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._timeline = timeline
        self._max_backlog = max_backlog
        self._backlog_window = backlog_window
        self._inputs: dict[str, _TrackInput] = {}

    @property
    def inputs(self) -> dict[str, _TrackInput]:
        return self._inputs

    @property
    def index(self) -> int:
        return self._timeline.end_index

    def add(self, track_input: _TrackInput) -> None:
        self._inputs[track_input.identity] = track_input

    def remove(self, identity: str) -> _TrackInput | None:
        return self._inputs.pop(identity, None)

    def all_ready(self) -> bool:
        return all(inp.ready for inp in self._inputs.values())

    def has_pending(self) -> bool:
        """Whether any audio a participant sent is still waiting to be mixed."""
        return any(inp.ready for inp in self._inputs.values())

    def prune(self) -> None:
        """Drop inputs that ended and have nothing left to mix."""
        for identity, inp in list(self._inputs.items()):
            if inp.ended and not inp.ready:
                del self._inputs[identity]

    def mix(self) -> rtc.AudioFrame:
        mixed = np.zeros(self._block_size, dtype=np.float32)
        activity: dict[str, TrackActivity] = {}

        # audio that arrived after the mix went on without it stays queued, mixed behind the
        # others' for good; a backlog that never drains is dropped so speech stays in step. A
        # participant who is leaving keeps everything they sent
        for inp in self._inputs.values():
            if not inp.ended:
                inp.settle(self._max_backlog, self._backlog_window)

        for inp in self._inputs.values():
            # a participant whose audio hasn't arrived is silent in this block, and the gate
            # doesn't learn its floor from that silence
            if (samples := inp.pop()) is None:
                continue

            frame = rtc.AudioFrame(
                data=samples.tobytes(),
                sample_rate=self._sample_rate,
                num_channels=1,
                samples_per_channel=self._block_size,
            )
            mixed += samples
            activity[inp.identity] = TrackActivity(
                rms=frame_rms(frame), open=inp.gate.update(frame), track_sid=inp.track_sid
            )

        self._timeline.append(activity)
        return rtc.AudioFrame(
            data=np.clip(mixed, -32768, 32767).astype(np.int16).tobytes(),
            sample_rate=self._sample_rate,
            num_channels=1,
            samples_per_channel=self._block_size,
        )
