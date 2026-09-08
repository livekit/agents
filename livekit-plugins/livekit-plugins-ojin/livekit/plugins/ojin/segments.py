from __future__ import annotations

import time


class _SegmentTracker:
    """Decides when an utterance is over.

    The runner reports playback-finished to the agent session only when the
    generator yields an ``AudioSegmentEnd``, and the session blocks on that
    report after every utterance - so getting this wrong either ends a turn early
    or wedges the conversation. Ojin has no segment concept, so the marker is
    inferred from two independent signals:

    * the **input** side, reported by the generator: the turn opened, real audio
      was pushed, the turn closed;
    * the **output** side, seen in the frame stream: real audio played, and the
      client's stop edge fired.

    A turn is over only when both agree. The stop edge alone is not enough: the
    SDK refills a draining buffer mid-utterance, so slow TTS produces a stop edge
    with the input still open. Every method that can end a segment returns
    whether the caller should emit the marker, keeping the decision here and the
    queueing in the sink.
    """

    def __init__(self, *, turn_render_timeout: float) -> None:
        self._turn_render_timeout = turn_render_timeout
        self._output_open = False  # real audio played since the last close
        self._input_closed = True  # the generator reports the input side
        self._output_drained = False  # stop edge seen while the input was open
        self._input_had_audio = False  # real audio pushed in this input segment
        self._render_deadline: float | None = None
        self._rendered_turns = 0
        self._retired = False  # a forced-closed turn may still render late
        self._owes_end = False  # a captured segment still awaiting its marker

    @property
    def output_open(self) -> bool:
        return self._output_open

    @property
    def rendered_turns(self) -> int:
        """Turns that actually produced audio; resets a run of render stalls."""
        return self._rendered_turns

    @property
    def input_closed(self) -> bool:
        return self._input_closed

    @property
    def input_had_audio(self) -> bool:
        return self._input_had_audio

    @property
    def owes_segment_end(self) -> bool:
        """The runner captured a segment and no marker has been emitted for it.

        The runner consumes the audio *and* the AudioSegmentEnd off the queue
        before this decides anything, so once that happens the queue no longer
        holds the segment: only this marker can complete it, and a teardown that
        drops it leaves the session blocked in ``wait_for_playout()`` forever.
        """
        return self._owes_end

    @property
    def retired(self) -> bool:
        """The last turn was forced closed and its late output must be ignored.

        Ojin's protocol carries no turn identity, so output arriving after a
        forced end cannot be told apart from the next turn's. Ignoring it until
        the next turn opens keeps a late render from reopening the output state
        and letting its stop edge close the following turn early.
        """
        return self._retired

    @property
    def render_deadline_armed(self) -> bool:
        return self._render_deadline is not None

    # --- input side, reported by the generator ---

    def input_opened(self) -> None:
        self._input_closed = False
        self._retired = False
        self._owes_end = False
        self._output_drained = False
        self._input_had_audio = False
        self._render_deadline = None

    def input_audio(self) -> None:
        """A real chunk was pushed; its echo has not played yet.

        Clearing the drained flag here rather than on output ticks stops a
        back-to-back final-chunk + input-close from ending the segment while the
        refilled tail is still unplayed.
        """
        self._output_drained = False
        self._input_had_audio = True

    def input_ended(self, *, had_real_audio: bool) -> bool:
        self._input_closed = True

        if not had_real_audio and not self._output_open:
            # An all-zero utterance never echoes back, so no stop edge will ever
            # close it - but the runner captured the segment and the session is
            # waiting on its report.
            self._owes_end = False
            return True

        if self._output_open and self._output_drained:
            return self._end()

        self._owes_end = True

        if self._input_had_audio and not self._output_open:
            # Fed but not rendering yet: bound the wait. A healthy transport can
            # stream idle frames forever while a turn never renders.
            self._render_deadline = time.monotonic() + self._turn_render_timeout

        return False

    # --- output side, seen in the frame stream ---

    def output_audio(self) -> None:
        self._output_open = True
        self._render_deadline = None
        self._rendered_turns += 1

    def stopped_speaking(self) -> bool:
        if not self._output_open:
            return False
        if self._input_closed:
            return self._end()
        # A mid-turn underrun: TTS is slower than playback. Not a turn end.
        self._output_drained = True
        return False

    # --- interruption and stalls ---

    def cleared(self) -> None:
        """Retire the segment on barge-in; the runner reports it as interrupted."""
        self._output_open = False
        self._retired = False
        self._output_drained = False
        self._input_closed = True
        self._owes_end = False
        self._input_had_audio = False
        self._render_deadline = None

    def render_deadline_expired(self) -> bool:
        return self._render_deadline is not None and time.monotonic() > self._render_deadline

    def force_end(self) -> bool:
        """End a fed turn the server never rendered, so the session can proceed."""
        self._end()
        self._retired = True
        return True

    def _end(self) -> bool:
        self._owes_end = False
        self._output_open = False
        self._output_drained = False
        self._render_deadline = None
        return True
