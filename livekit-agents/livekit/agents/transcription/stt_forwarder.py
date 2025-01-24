from __future__ import annotations

import abc
import asyncio
import sys
from typing import AsyncIterator

from livekit import rtc

from .. import stt, utils
from ..log import logger
from ..utils import log_exceptions
from . import _utils


class STTSegmentsForwarder(abc.ABC):
    """Base class for forwarding STT segments from speech recognition."""

    def __init__(self):
        self._forward_task: asyncio.Task | None = None
        self._current_id = _utils.segment_uuid()
        self._segment_stream = utils.aio.Chan[rtc.TranscriptionSegment]()

    def update(self, ev: stt.SpeechEvent) -> None:
        """Update with new speech recognition event."""
        if self._forward_task is None:
            self._forward_task = asyncio.create_task(
                self._forward_segments(self._segment_stream)
            )

        # Common handling of speech events
        if not ev.alternatives:
            return

        text = ev.alternatives[0].text
        is_final = ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT

        segment = rtc.TranscriptionSegment(
            id=self._current_id,
            text=text,
            start_time=0,
            end_time=0,
            final=is_final,
            language="",  # TODO: Add language support
        )

        if is_final:
            self._current_id = _utils.segment_uuid()

        self._segment_stream.send_nowait(segment)

    async def aclose(self) -> None:
        """Close the forwarder."""
        self._segment_stream.close()
        if self._forward_task:
            await self._forward_task

    @abc.abstractmethod
    async def _forward_segments(
        self, segment_stream: AsyncIterator[rtc.TranscriptionSegment]
    ) -> None:
        """Forward segments to the target destination.

        Args:
            segment_stream: Stream of transcription segments from speech recognition
        """
        pass


class STTRoomForwarder(STTSegmentsForwarder):
    """Forwards STT segments to a LiveKit room."""

    def __init__(
        self,
        room: rtc.Room,
        participant: rtc.Participant | str,
        track: rtc.Track | rtc.TrackPublication | str | None = None,
    ):
        super().__init__()
        identity = participant if isinstance(participant, str) else participant.identity
        if track is None:
            track = _utils.find_micro_track_id(room, identity)
        elif isinstance(track, (rtc.TrackPublication, rtc.Track)):
            track = track.sid

        self._room = room
        self._participant_identity = identity
        self._track_id = track

    @log_exceptions(logger=logger)
    async def _forward_segments(
        self, segment_stream: AsyncIterator[rtc.TranscriptionSegment]
    ) -> None:
        """Forward segments to LiveKit room."""
        try:
            async for segment in segment_stream:
                if not self._room.isconnected():
                    continue

                transcription = rtc.Transcription(
                    participant_identity=self._participant_identity,
                    track_sid=self._track_id,
                    segments=[segment],
                )

                try:
                    await self._room.local_participant.publish_transcription(
                        transcription
                    )
                except Exception:
                    logger.exception("Failed to publish transcription")
                    continue

        except Exception:
            logger.exception("Error in forward segments task")


class STTStdoutForwarder(STTSegmentsForwarder):
    """Forwards STT segments to stdout with real-time display."""

    def __init__(self, *, show_timing: bool = False):
        super().__init__()
        self._show_timing = show_timing
        self._start_time = asyncio.get_running_loop().time()
        self._last_text = ""

    @log_exceptions(logger=logger)
    async def _forward_segments(
        self, segment_stream: AsyncIterator[rtc.TranscriptionSegment]
    ) -> None:
        """Forward segments to console with real-time display."""
        try:
            async for segment in segment_stream:
                text = segment.text
                if text == self._last_text:
                    continue

                # Clear the line and write the new text
                sys.stdout.write("\r" + " " * len(self._last_text) + "\r")

                if self._show_timing:
                    elapsed = asyncio.get_running_loop().time() - self._start_time
                    timing = f"[{elapsed:.2f}s] "
                    sys.stdout.write(timing)

                sys.stdout.write(text)
                sys.stdout.flush()
                self._last_text = text

                if segment.final:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    self._last_text = ""

        except Exception:
            logger.exception("Error in forward segments task")
