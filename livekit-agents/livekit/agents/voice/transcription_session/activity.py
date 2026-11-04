from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

# block edges computed from float seconds land a hair off the integer; this keeps them on it
_EPSILON = 1e-9


@dataclass(frozen=True)
class TrackActivity:
    """One participant's contribution to one block of the mix."""

    rms: float
    open: bool
    """Whether the participant's gate considered the block speech."""
    track_sid: str | None = None
    """The track the block came from; a participant can switch microphones mid-session."""


class ActivityTimeline:
    """Who was audible in each block of the mixed stream, indexed by the mixer's clock.

    The clock is the audio pushed to the STT, so a timestamp the STT reports maps onto a block
    exactly; wall clock never enters the lookup.
    """

    def __init__(self, *, block_duration: float, retention: float) -> None:
        self._block_duration = block_duration
        self._max_blocks = max(1, math.ceil(retention / block_duration))
        self._blocks: deque[dict[str, TrackActivity]] = deque()
        self._first_index = 0

    @property
    def block_duration(self) -> float:
        return self._block_duration

    @property
    def end_index(self) -> int:
        """Index of the next block to be appended."""
        return self._first_index + len(self._blocks)

    @property
    def duration(self) -> float:
        """Stream time covered so far, in seconds."""
        return self.end_index * self._block_duration

    def append(self, activity: dict[str, TrackActivity]) -> None:
        self._blocks.append(activity)
        while len(self._blocks) > self._max_blocks:
            self._blocks.popleft()
            self._first_index += 1

    def track_sid(self, identity: str, start: float, end: float) -> str | None:
        """The track a participant spoke on over ``[start, end)``: the last one heard there."""
        sid: str | None = None
        for activity in self._span(start, end):
            if (track := activity.get(identity)) is not None and track.track_sid is not None:
                sid = track.track_sid
        return sid

    def scores(self, start: float, end: float) -> dict[str, float]:
        """Energy each participant produced with its gate open over ``[start, end)``.

        A span shorter than a block still covers the block it starts in. Participants whose gate
        stayed closed throughout are left out.
        """
        scores: dict[str, float] = {}
        for block in self._span(start, end):
            for identity, activity in block.items():
                if activity.open:
                    scores[identity] = scores.get(identity, 0.0) + activity.rms
        return scores

    def _span(self, start: float, end: float) -> Iterator[dict[str, TrackActivity]]:
        first = math.floor(start / self._block_duration + _EPSILON)
        last = max(first + 1, math.ceil(end / self._block_duration - _EPSILON))
        for index in range(max(first, self._first_index), min(last, self.end_index)):
            yield self._blocks[index - self._first_index]
