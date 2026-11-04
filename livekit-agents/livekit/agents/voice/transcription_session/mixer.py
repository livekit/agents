from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from livekit import rtc

from ...log import logger
from ...utils.audio_gate import AudioGate, frame_rms
from .activity import ActivityTimeline, TrackActivity


@dataclass
class _DelayedBlock:
    samples: np.ndarray
    open: bool
    rms: float
    real: bool
    """False for the silence that stands in for a block the participant never sent."""


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

        self.delay: deque[_DelayedBlock] = deque()
        self.gain = 1.0

    @property
    def ready(self) -> bool:
        return bool(self._blocks)

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
    """Sums every participant into one block at a time and records who was audible in it.

    A participant whose gate is closed is ducked rather than summed at full level, so a room of
    open microphones doesn't hand the STT the sum of their noise floors. The mix runs
    ``lookahead_blocks`` behind its inputs so the gate has opened before a word's onset is mixed.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        block_size: int,
        timeline: ActivityTimeline,
        duck_gain: float | None,
        lookahead_blocks: int,
    ) -> None:
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._timeline = timeline
        self._duck_gain = duck_gain
        self._lookahead = lookahead_blocks if duck_gain is not None else 0
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
        return any(
            inp.ready or any(block.real for block in inp.delay) for inp in self._inputs.values()
        )

    def prune(self) -> None:
        """Drop inputs that ended and have nothing left to mix."""
        for identity, inp in list(self._inputs.items()):
            if inp.ended and not inp.ready and not any(block.real for block in inp.delay):
                del self._inputs[identity]

    def mix(self) -> rtc.AudioFrame:
        mixed = np.zeros(self._block_size, dtype=np.float32)
        activity: dict[str, TrackActivity] = {}

        for inp in self._inputs.values():
            samples = inp.pop()
            if samples is None:
                # an underrun is silence, but not silence the gate should learn its floor from
                inp.delay.append(
                    _DelayedBlock(
                        np.zeros(self._block_size, dtype=np.int16), False, 0.0, real=False
                    )
                )
            else:
                frame = rtc.AudioFrame(
                    data=samples.tobytes(),
                    sample_rate=self._sample_rate,
                    num_channels=1,
                    samples_per_channel=self._block_size,
                )
                inp.delay.append(
                    _DelayedBlock(samples, inp.gate.update(frame), frame_rms(frame), real=True)
                )

            if len(inp.delay) <= self._lookahead:
                continue  # still filling the pre-roll

            block = inp.delay.popleft()
            gain = 1.0
            if self._duck_gain is not None and not (
                block.open or any(ahead.open for ahead in inp.delay)
            ):
                gain = self._duck_gain

            ramp = np.linspace(inp.gain, gain, self._block_size, dtype=np.float32)
            inp.gain = gain
            mixed += block.samples.astype(np.float32) * ramp
            activity[inp.identity] = TrackActivity(rms=block.rms, open=block.open)

        self._timeline.append(activity)
        return rtc.AudioFrame(
            data=np.clip(mixed, -32768, 32767).astype(np.int16).tobytes(),
            sample_rate=self._sample_rate,
            num_channels=1,
            samples_per_channel=self._block_size,
        )
