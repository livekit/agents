from __future__ import annotations

import abc
import asyncio
import sys
from typing import AsyncIterator, Optional

from livekit import rtc

from ..log import logger
from ..utils import log_exceptions
from . import _utils
from .tts_segments_sync import TTSSegmentsSync, TTSSegmentsSyncOptions


class TTSSegmentsForwarder(TTSSegmentsSync, abc.ABC):
    """Base class for forwarding TTS segments with synchronized timing."""

    def __init__(
        self,
        sync_options: TTSSegmentsSyncOptions | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__(sync_options or TTSSegmentsSyncOptions(), loop=loop)
        self._forward_task: Optional[asyncio.Task] = None

    def segment_playout_started(self) -> None:
        """Override to start forwarding when playout starts."""
        super().segment_playout_started()
        if self._forward_task is None:
            self._forward_task = asyncio.create_task(
                self._forward_segments(self.segment_stream)
            )

    async def aclose(self) -> None:
        """Close both syncer and forwarder."""
        await super().aclose()
        if self._forward_task:
            await self._forward_task

    @abc.abstractmethod
    async def _forward_segments(
        self, segment_stream: AsyncIterator[rtc.TranscriptionSegment]
    ) -> None:
        """Forward synchronized segments to the target destination.

        Args:
            segment_stream: Stream of transcription segments synchronized with audio timing
        """
        pass


class TTSRoomForwarder(TTSSegmentsForwarder):
    """Forwards synchronized TTS segments to a LiveKit room."""

    def __init__(
        self,
        room: rtc.Room,
        participant: rtc.Participant | str,
        track: rtc.Track | rtc.TrackPublication | str | None = None,
        sync_options: TTSSegmentsSyncOptions | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__(sync_options, loop=loop)
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
        """Forward transcription segments to LiveKit room."""
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


class TTSStdoutForwarder(TTSSegmentsForwarder):
    """Forwards synchronized TTS segments to stdout with real-time display."""

    def __init__(
        self,
        sync_options: TTSSegmentsSyncOptions | None = None,
        *,
        show_timing: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__(sync_options, loop=loop)
        self._show_timing = show_timing
        self._start_time: Optional[float] = None

    def segment_playout_started(self) -> None:
        """Override to capture start time for timing display."""
        if self._forward_task is None:
            self._start_time = asyncio.get_running_loop().time()
        super().segment_playout_started()

    @log_exceptions(logger=logger)
    async def _forward_segments(
        self, segment_stream: AsyncIterator[rtc.TranscriptionSegment]
    ) -> None:
        """Forward transcription segments to console with real-time display."""
        try:
            last_text = ""
            async for segment in segment_stream:
                text = segment.text
                if text == last_text:
                    continue

                # Clear the line and write the new text
                sys.stdout.write("\r" + " " * len(last_text) + "\r")

                if self._show_timing and self._start_time is not None:
                    elapsed = asyncio.get_running_loop().time() - self._start_time
                    timing = f"[{elapsed:.2f}s] "
                    sys.stdout.write(timing)

                sys.stdout.write(text)
                sys.stdout.flush()
                last_text = text

                if segment.final:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    last_text = ""

        except Exception:
            logger.exception("Error in forward segments task")
