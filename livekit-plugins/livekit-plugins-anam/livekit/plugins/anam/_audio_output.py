from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

from livekit import rtc
from livekit.agents.voice.avatar import DataStreamAudioOutput

from .log import logger

# A forwarded lk.clear_buffer is answered by lk.playback_finished within a few hundred
# milliseconds, so this only fires when the avatar stops responding.
CLEAR_BUFFER_TIMEOUT = 5.0
# How long an interrupt waits for lk.playback_started before it is forwarded anyway.
PLAYBACK_STARTED_TIMEOUT = 5.0


@dataclass
class _Segment:
    # the byte stream of the segment was opened, so the avatar received it
    opened: bool = False
    started: bool = False
    clear_requested: bool = False
    clear_sent: bool = False


class AnamAudioOutput(DataStreamAudioOutput):
    """DataStreamAudioOutput that never clears the avatar buffer before playback started.

    The Anam avatar sends no lk.playback_finished for a lk.clear_buffer that reaches it
    before it rendered the first frame of a segment, while a clear that arrives just after
    gets one. The agent cannot tell these apart, and waits for that finish after every
    interrupt. So an interrupt is handled per pending segment:

    - nothing pending: no RPC is sent
    - the segment never reached the avatar: it finishes locally
    - the segment reached the avatar but did not start: the clear is held until
      lk.playback_started arrives
    - every pending segment started: the clear is forwarded now
    """

    def __init__(
        self,
        room: rtc.Room,
        *,
        destination_identity: str,
        sample_rate: int | None = None,
        wait_remote_track: rtc.TrackKind.ValueType | None = None,
        playback_started_timeout: float = PLAYBACK_STARTED_TIMEOUT,
    ) -> None:
        super().__init__(
            room,
            destination_identity=destination_identity,
            sample_rate=sample_rate,
            wait_remote_track=wait_remote_track,
            clear_buffer_timeout=CLEAR_BUFFER_TIMEOUT,
            wait_playback_start=True,
        )
        self._playback_started_timeout = playback_started_timeout
        # captured segments without a playback finished event, oldest first. The avatar
        # plays them in order and its events carry no segment id, so every event belongs
        # to the oldest segment it applies to.
        self._segments: deque[_Segment] = deque()
        self._capturing_segment: _Segment | None = None
        self._held_clear_timer: asyncio.TimerHandle | None = None

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        captured = self.captured_playout_segments
        try:
            await super().capture_frame(frame)
        finally:
            if self.captured_playout_segments > captured:
                # the first frame of a segment was counted, even if it was cancelled
                # before its byte stream was opened
                self._capturing_segment = _Segment(opened=self._stream_writer is not None)
                self._segments.append(self._capturing_segment)

    def flush(self) -> None:
        super().flush()
        self._capturing_segment = None
        self._finish_unopened_segments()

    def clear_buffer(self) -> None:
        self._finish_unopened_segments()
        if self._pending_playback_count == 0:
            return

        if not self._segments:
            # out of sync with the counts of the base class, keep its behaviour
            super().clear_buffer()
            return

        if all(segment.clear_sent for segment in self._segments):
            # an interrupt can clear the buffer more than once
            return

        for segment in self._segments:
            segment.clear_requested = True

        self._release_held_clear()
        if self._held_segments() and self._held_clear_timer is None:
            self._held_clear_timer = asyncio.get_event_loop().call_later(
                self._playback_started_timeout, self._on_held_clear_timeout
            )

    def on_playback_started(self, *, created_at: float) -> None:
        for segment in self._segments:
            if segment.opened and not segment.started:
                segment.started = True
                break

        super().on_playback_started(created_at=created_at)
        self._release_held_clear()

    def on_playback_finished(
        self,
        *,
        playback_position: float,
        interrupted: bool,
        synchronized_transcript: str | None = None,
    ) -> None:
        if self._pending_playback_count > 0 and self._segments:
            self._segments.popleft()

        super().on_playback_finished(
            playback_position=playback_position,
            interrupted=interrupted,
            synchronized_transcript=synchronized_transcript,
        )
        self._finish_unopened_segments()
        self._release_held_clear()

    def _reset_playback_count(self) -> None:
        super()._reset_playback_count()
        self._segments.clear()
        self._capturing_segment = None
        self._cancel_held_clear_timer()

    def _finish_unopened_segments(self) -> None:
        # nothing of these segments reached the avatar, so it never reports them
        while (
            self._segments
            and not self._segments[0].opened
            and self._segments[0] is not self._capturing_segment
        ):
            self._segments.popleft()
            super().on_playback_finished(playback_position=0.0, interrupted=True)

    def _held_segments(self) -> list[_Segment]:
        return [
            segment
            for segment in self._segments
            if segment.opened and segment.clear_requested and not segment.clear_sent
        ]

    def _release_held_clear(self) -> None:
        held = self._held_segments()
        if not held:
            # also when the avatar finished the segments without a playback started event
            self._cancel_held_clear_timer()
        elif all(segment.started for segment in held):
            self._forward_clear()

    def _on_held_clear_timeout(self) -> None:
        self._held_clear_timer = None
        logger.warning(
            "no lk.playback_started after an interrupt, clearing the avatar buffer anyway",
            extra={"timeout": self._playback_started_timeout},
        )
        self._forward_clear()

    def _forward_clear(self) -> None:
        self._cancel_held_clear_timer()
        # the avatar clears everything it buffered, not only the requested segments
        for segment in self._segments:
            segment.clear_sent = True
        super().clear_buffer()

    def _cancel_held_clear_timer(self) -> None:
        if self._held_clear_timer is not None:
            self._held_clear_timer.cancel()
            self._held_clear_timer = None
