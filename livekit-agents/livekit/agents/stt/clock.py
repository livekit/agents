from __future__ import annotations

import time
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..types import NOT_GIVEN, TimedString
from ..utils import is_given

if TYPE_CHECKING:
    from .stt import SpeechData, SpeechEvent

# Maximum timing difference treated as jitter, in seconds.
_INPUT_GAP_TOLERANCE = 0.1
# Retain completed runs for late results for this many seconds.
_INPUT_RUN_HISTORY = 300.0
# Allow floating-point error in summed frame durations, below one audio sample.
_INPUT_RUN_BOUNDARY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class _SessionStart:
    """Session start as monotonic and wall-clock timestamps of the same instant."""

    monotonic_timestamp: float
    wall_timestamp: float

    @classmethod
    def from_wall_timestamp(cls, wall_timestamp: float) -> _SessionStart:
        """Pair ``wall_timestamp`` with the monotonic timestamp it has right now."""
        return cls(
            monotonic_timestamp=wall_timestamp - (time.time() - time.perf_counter()),
            wall_timestamp=wall_timestamp,
        )


@dataclass
class _InputRun:
    """A contiguous audio run on the input stream.

    ``pushed_duration`` is the stream position at this run's start.
    ``started_at`` is the estimated monotonic timestamp of its first frame.
    ``ended_at`` is the monotonic timestamp where the run's clock stops. Later
    stream positions in this run share that timestamp.

    Buffered audio can move a new run's start time earlier. Once later frames
    arrive at normal playback speed, or the run is used to translate a timestamp,
    its start time is fixed. The run is then settled: later frames cannot shift
    timestamps already assigned to earlier audio.
    """

    pushed_duration: float
    started_at: float
    settled: bool = False
    ended_at: float | None = None


class _InputClock:
    """Translate between stream positions and session timestamps.

    A stream position is the audio duration from the stream's first frame, in
    seconds, excluding input gaps. A session timestamp is the elapsed time since
    the session start, including gaps. Runs estimate audio timing from frame
    enqueue times. Use this clock on the stream's event loop.

    When audio arrives faster than normal playback speed, an unsettled run can
    move its start earlier, bounded by the end of the preceding audio. A settled
    run keeps its start. If it gets ahead of real time beyond the jitter tolerance,
    its clock stops without moving already assigned timestamps. Later positions
    share that stop time until frames resume at normal playback speed.

    Run timestamps are compared with the session start's monotonic timestamp, so
    wall-clock changes do not affect gap detection. Its wall-clock timestamp is
    used only for translating speech bounds.
    """

    def __init__(self) -> None:
        self._session_start: _SessionStart | None = None
        self._start_time_offset = 0.0

        self._runs: deque[_InputRun] = deque()
        self._enqueued_duration = 0.0
        self._dequeued_duration = 0.0

    def set_session_timeline(self, offset: float) -> None:
        """Set the session start to ``offset`` seconds ago, replacing any earlier value."""
        self.session_start = _SessionStart(
            monotonic_timestamp=time.perf_counter() - offset,
            wall_timestamp=time.time() - offset,
        )

    @property
    def session_start(self) -> _SessionStart | None:
        """Monotonic and wall-clock timestamps of the session start.

        ``None`` before the session start is known. Assign another stream's value
        to continue its session on this stream.
        """
        return self._session_start

    @session_start.setter
    def session_start(self, value: _SessionStart) -> None:
        self._session_start = value
        self._start_time_offset = max(time.perf_counter() - value.monotonic_timestamp, 0.0)

    @property
    def dequeued_duration(self) -> float:
        """Cumulative audio duration dequeued from the input stream, in seconds."""
        return self._dequeued_duration

    def start_connection(self) -> _ConnectionClock:
        """Create a connection clock starting at the next frame to be dequeued.

        If the session start is unknown, use that frame's estimated timestamp.
        """
        if self._session_start is None:
            # The session start bounds this run's start without settling it.
            monotonic_start = self._stream_to_monotonic_timestamp(
                self.dequeued_duration, after_gap=True, settle=False
            )
            self._session_start = _SessionStart(
                monotonic_timestamp=monotonic_start,
                wall_timestamp=time.time() + (monotonic_start - time.perf_counter()),
            )
        return _ConnectionClock(self, self.dequeued_duration)

    def on_audio_dequeued(self, duration: float) -> None:
        """Record ``duration`` seconds of audio dequeued from the input stream."""
        self._dequeued_duration += duration

    def on_audio_enqueued(
        self, duration: float, pushed_at: float, connection_clock: _ConnectionClock
    ) -> None:
        """Record ``duration`` seconds of audio enqueued at monotonic timestamp ``pushed_at``.

        Save the current connection's first-frame timestamp before pruning its run.
        """
        position = self._enqueued_duration
        self._enqueued_duration += duration
        # Estimate the frame's start from its enqueue time and duration.
        started_at = pushed_at - duration
        settled = False
        while self._runs:
            run = self._runs[-1]
            run_time = run.started_at + (position - run.pushed_duration)
            if run.ended_at is not None:
                run_time = min(run_time, run.ended_at)
            # Positive lead means the run's clock is ahead of this frame's estimated start.
            lead = run_time - started_at
            if lead < -_INPUT_GAP_TOLERANCE:
                break

            if run.settled:
                if lead > _INPUT_GAP_TOLERANCE:
                    if run.ended_at is None:
                        # Stop the clock without moving already pushed positions.
                        run.ended_at = max(pushed_at, run_time)
                    return
                if run.ended_at is None:
                    return
                # Real time caught up with the burst: continue from its end.
                started_at = max(started_at, run.ended_at)
                settled = True
                break

            if lead <= 0:
                run.settled = True
                return

            # The opening burst of a run is buffered audio: its start is earlier.
            run_started_at = started_at - (position - run.pushed_duration)
            if len(self._runs) == 1:
                # Keep audio at or after the session start.
                session_start = self._session_start
                run.started_at = (
                    run_started_at
                    if session_start is None
                    else max(run_started_at, session_start.monotonic_timestamp)
                )
                return

            previous_ended_at = self._run_ended_at(len(self._runs) - 2)
            if run_started_at > previous_ended_at or self._runs[-2].ended_at is not None:
                # no audio precedes the end of a burst on the previous run
                run.started_at = max(run_started_at, previous_ended_at)
                return
            # Catch-up audio closes this gap; merge it into the previous run.
            self._runs.pop()

        self._runs.append(
            _InputRun(pushed_duration=position, started_at=started_at, settled=settled)
        )
        while len(self._runs) > 1 and self._runs[1].started_at < started_at - _INPUT_RUN_HISTORY:
            start_position = connection_clock._start_position
            if (
                connection_clock._first_frame_at is None
                and self._runs[1].pushed_duration - start_position > _INPUT_RUN_BOUNDARY_TOLERANCE
            ):
                # Preserve the first frame's monotonic time before pruning its run.
                connection_clock._first_frame_at = self._stream_to_monotonic_timestamp(
                    start_position, after_gap=True
                )
            self._runs.popleft()

    def session_to_stream_timestamps(self, speech_data: SpeechData) -> SpeechData | None:
        """Copy segment and word timestamps from the session timeline to stream positions.

        A timestamp within a gap maps to the stream position at the gap. While a
        run's clock is stopped, inverse timestamps clamp to its stop position. At
        a shared boundary, starts use the following run and ends use the preceding
        run. Zero-length spans use the preceding run for both bounds. Preserve unknown
        bounds and other fields. Leave the source data and words unchanged.
        Return ``None`` before audio is enqueued, and when a timestamp precedes
        the retained run history (``_INPUT_RUN_HISTORY``).

        TODO: @chenghao-mou This is solely for multi_speaker_adapter.py because
        RMS is measured on the input stream and the events are already on the session
        timeline.
        """
        session_start = self._session_start
        runs = self._runs
        if session_start is None or not runs:
            return None
        monotonic_start = session_start.monotonic_timestamp

        start_time, end_time = speech_data.start_time, speech_data.end_time
        words = speech_data.words
        if runs[0].pushed_duration > 0:
            # Skip timestamps from pruned runs instead of selecting later audio.
            # A start of 0 is known when the end is known.
            times = [start_time] if start_time or end_time else []
            times += [end_time] if end_time else []
            times += [word.start_time for word in words or [] if is_given(word.start_time)]
            times += [word.end_time for word in words or [] if is_given(word.end_time)]
            if any(monotonic_start + t < runs[0].started_at for t in times):
                return None

        def position(timeline_time: float | None, *, after_gap: bool) -> float | None:
            if timeline_time is None:
                return None
            timestamp = monotonic_start + timeline_time
            last = len(runs) - 1
            boundary = timestamp + (
                _INPUT_RUN_BOUNDARY_TOLERANCE if after_gap else -_INPUT_RUN_BOUNDARY_TOLERANCE
            )
            index = max(0, bisect_right(runs, boundary, key=lambda run: run.started_at) - 1)
            run = runs[index]
            if index == last or (
                index == last - 1
                and timestamp - self._run_ended_at(index) > _INPUT_RUN_BOUNDARY_TOLERANCE
            ):
                # This stream position depends on the newest run's start time.
                runs[last].settled = True
            run_end = runs[index + 1].pushed_duration if index < last else self._enqueued_duration
            if run.ended_at is not None:
                timestamp = min(timestamp, run.ended_at)
            return min(
                max(run.pushed_duration + (timestamp - run.started_at), run.pushed_duration),
                run_end,
            )

        def span(start: float | None, end: float | None) -> tuple[float | None, float | None]:
            has_length = start is None or end is None or end - start > _INPUT_RUN_BOUNDARY_TOLERANCE
            return position(start, after_gap=has_length), position(end, after_gap=False)

        if words is not None:
            translated_words = []
            for word in words:
                word_start, word_end = span(
                    word.start_time if is_given(word.start_time) else None,
                    word.end_time if is_given(word.end_time) else None,
                )
                translated_words.append(
                    TimedString(
                        str(word),
                        start_time=NOT_GIVEN if word_start is None else word_start,
                        end_time=NOT_GIVEN if word_end is None else word_end,
                        confidence=word.confidence,
                        start_time_offset=0.0,
                        speaker_id=word.speaker_id,
                    )
                )
            words = translated_words
        start, end = span(start_time if start_time or end_time else None, end_time or None)
        return replace(
            speech_data,
            start_time=start or 0.0,
            end_time=end or 0.0,
            words=words,
        )

    def _run_ended_at(self, index: int) -> float:
        """Monotonic timestamp of the last stream position in the run at ``index``."""
        runs = self._runs
        run = runs[index]
        end = runs[index + 1].pushed_duration if index + 1 < len(runs) else self._enqueued_duration
        ended_at = run.started_at + (end - run.pushed_duration)
        return ended_at if run.ended_at is None else min(ended_at, run.ended_at)

    def _stream_to_monotonic_timestamp(
        self, position: float, *, after_gap: bool, settle: bool = True
    ) -> float:
        """Estimated monotonic timestamp of a stream position.

        A gap has two timestamps at the same stream position; ``after_gap`` selects
        the later timestamp. The tolerance covers floating-point error in summed
        audio durations. Using the newest run fixes its start time unless
        ``settle`` is false.
        """
        runs = self._runs
        index = max(0, bisect_right(runs, position, key=lambda run: run.pushed_duration) - 1)
        if after_gap:
            if (
                index + 1 < len(runs)
                and runs[index + 1].pushed_duration - position <= _INPUT_RUN_BOUNDARY_TOLERANCE
            ):
                index += 1
                position = runs[index].pushed_duration
        elif index > 0 and position - runs[index].pushed_duration <= _INPUT_RUN_BOUNDARY_TOLERANCE:
            position = runs[index].pushed_duration
            index -= 1

        run = runs[index]
        if settle and index == len(runs) - 1:
            run.settled = True
        timestamp = run.started_at + (position - run.pushed_duration)
        return timestamp if run.ended_at is None else min(timestamp, run.ended_at)


class _ConnectionClock:
    """Translate connection timestamps to session timestamps.

    A connection timestamp counts audio seconds from the connection's first frame.
    ``start_position`` is that frame's stream position.

    Example:
        | Event                  | Wall clock | Stream position | Connection's start_position |
        |------------------------|------------|-----------------|-----------------------------|
        | RecognizeStream starts | t0         | 0               | —                           |
        | Connection 1 starts    | t0         | 0               | 0                           |
        | Gap starts             | t1         | t1 - t0         | —                           |
        | Connection 2 starts    | t2         | t1 - t0         | t1 - t0                     |

    Once audio is available, the first use caches the first frame's monotonic
    timestamp. If its run was already pruned, the estimate uses the oldest retained
    run. The current connection's timestamp is also saved before pruning, even if
    unused.
    """

    def __init__(self, input_clock: _InputClock, start_position: float) -> None:
        self._input_clock = input_clock
        self._start_position = start_position
        self._first_frame_at: float | None = None

    @property
    def start_time_offset(self) -> float:
        """Session timestamp of this connection's first audio frame, in seconds."""
        input_clock = self._input_clock
        start_position = self._start_position
        session_start = input_clock.session_start
        if session_start is None or not input_clock._runs:
            # Use the declared offset until audio can be placed.
            return input_clock._start_time_offset
        if (first_frame_at := self._first_frame_at) is None:
            first_frame_at = input_clock._stream_to_monotonic_timestamp(
                start_position, after_gap=True
            )
            if input_clock._enqueued_duration - start_position > _INPUT_RUN_BOUNDARY_TOLERANCE:
                self._first_frame_at = first_frame_at
        return first_frame_at - session_start.monotonic_timestamp

    def _connection_to_session_timestamps(
        self, start: float | None, end: float | None
    ) -> tuple[float | None, float | None]:
        """Convert connection timestamps to session timestamps.

        ``None`` means an unknown bound.

        At a gap, starts belong to the following audio and ends to the preceding
        audio. A zero-length span uses the preceding side for both bounds. A lone
        start uses the following side; a lone end uses the preceding side.

        Example:
            | Timeline   | Audio before gap | Input gap | Audio after gap |
            |------------|------------------|-----------|-----------------|
            | Stream     | [0, 2]           | —         | [2, 4]          |
            | Connection | [0, 2]           | —         | [2, 4]          |
            | Session    | [0, 2]           | [2, 7]    | [7, 9]          |

            | Case                        | Connection bounds | Session bounds |
            |-----------------------------|-------------------|----------------|
            | Span ends at gap            | [1, 2]            | [1, 2]         |
            | Span starts at gap          | [2, 3]            | [7, 8]         |
            | Span crosses gap            | [1, 3]            | [1, 8]         |
            | Zero-length span at gap     | [2, 2]            | [2, 2]         |
            | Only a start is known       | [2, None]         | [7, None]      |
            | Only an end is known        | [None, 2]         | [None, 2]      |
        """
        if start is None and end is None:
            return None, None
        input_clock = self._input_clock
        start_position = self._start_position
        session_start = input_clock.session_start
        start_time_offset = self.start_time_offset
        if session_start is None or not input_clock._runs:
            return (
                None if start is None else start_time_offset + start,
                None if end is None else start_time_offset + end,
            )

        monotonic_start = session_start.monotonic_timestamp
        has_length = start is None or end is None or end - start > _INPUT_RUN_BOUNDARY_TOLERANCE

        def place(connection_time: float | None, *, after_gap: bool) -> float | None:
            if connection_time is None:
                return None
            if connection_time <= _INPUT_RUN_BOUNDARY_TOLERANCE:
                # Connection timestamp 0 belongs to its first frame, after any input gap.
                return start_time_offset + connection_time
            timestamp = input_clock._stream_to_monotonic_timestamp(
                start_position + connection_time, after_gap=after_gap
            )
            return timestamp - monotonic_start

        return place(start, after_gap=has_length), place(end, after_gap=False)

    def connection_to_session_timestamps(self, speech_data: SpeechData) -> SpeechData:
        """Copy segment and word timestamps from the connection timeline to the session timeline.

        Preserve unknown bounds and other fields. Leave the source data and words
        unchanged. A segment end of 0 is unknown; when its start is also 0, both
        bounds are unknown. Word bounds use ``NOT_GIVEN`` for unknown times.
        """
        start_time, end_time = self._connection_to_session_timestamps(
            speech_data.start_time if speech_data.start_time or speech_data.end_time else None,
            speech_data.end_time or None,
        )
        words = speech_data.words
        if words is not None:
            start_time_offset = self.start_time_offset
            translated_words = []
            for word in words:
                word_start, word_end = self._connection_to_session_timestamps(
                    word.start_time if is_given(word.start_time) else None,
                    word.end_time if is_given(word.end_time) else None,
                )
                translated_words.append(
                    TimedString(
                        str(word),
                        start_time=NOT_GIVEN if word_start is None else word_start,
                        end_time=NOT_GIVEN if word_end is None else word_end,
                        confidence=word.confidence,
                        start_time_offset=start_time_offset,
                        speaker_id=word.speaker_id,
                    )
                )
            words = translated_words
        return replace(
            speech_data, start_time=start_time or 0.0, end_time=end_time or 0.0, words=words
        )

    def translate_event_timestamps(self, event: SpeechEvent) -> SpeechEvent:
        """Copy ``event`` with translated transcript timestamps and wall-clock speech bounds."""
        speech_start_time, speech_end_time = self._connection_to_session_timestamps(
            event.speech_start_time, event.speech_end_time
        )
        session_start = self._input_clock.session_start
        if session_start is None:
            speech_start_time = speech_end_time = None
        else:
            if speech_start_time is not None:
                speech_start_time += session_start.wall_timestamp
            if speech_end_time is not None:
                speech_end_time += session_start.wall_timestamp

        return replace(
            event,
            alternatives=[self.connection_to_session_timestamps(alt) for alt in event.alternatives],
            speech_start_time=speech_start_time,
            speech_end_time=speech_end_time,
        )
