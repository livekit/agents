import asyncio
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Optional, TextIO

from livekit import rtc

from .. import stt
from ..log import logger
from ..utils import aio, log_exceptions
from ._utils import find_micro_track_id, segment_uuid


class TranscriptionForwarder(ABC):
    """Base class for all transcription forwarders."""

    def __init__(self):
        self._event_ch = aio.Chan[rtc.TranscriptionSegment]()
        self._main_task = asyncio.create_task(self._main_task())
        self._current_id = segment_uuid()

    @abstractmethod
    async def _forward_segment(self, segment: rtc.TranscriptionSegment) -> None:
        """Forward a single segment to the target destination."""
        pass

    async def _main_task(self) -> None:
        """Process segments from the event channel."""
        try:
            async for segment in self._event_ch:
                await self._forward_segment(segment)
        except Exception:
            logger.exception("Error processing segment stream")

    def update(self, ev: rtc.TranscriptionSegment | stt.SpeechEvent) -> None:
        if isinstance(ev, rtc.TranscriptionSegment):
            self._event_ch.send_nowait(ev)
            return

        # SpeechEvent to TranscriptionSegment
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
            self._current_id = segment_uuid()
        self._event_ch.send_nowait(segment)

    async def aclose(self) -> None:
        """Close the forwarder and cleanup resources."""
        self._event_ch.close()
        await aio.cancel_and_wait(self._main_task)


class TranscriptionRoomForwarder(TranscriptionForwarder):
    """Forwards transcriptions to LiveKit rooms."""

    def __init__(
        self,
        room: rtc.Room,
        participant: rtc.Participant | str,
        track: rtc.Track | rtc.TrackPublication | str | None = None,
        *,
        forward_delta: bool = False,
    ):
        super().__init__()
        identity = participant if isinstance(participant, str) else participant.identity
        if track is None:
            track = find_micro_track_id(room, identity)
        elif isinstance(track, (rtc.TrackPublication, rtc.Track)):
            track = track.sid

        self._room = room
        self._participant_identity = identity
        self._track_id = track

        self._forward_delta = forward_delta
        self._last_segment_id: Optional[str] = None
        self._played_text = ""

    @log_exceptions(logger=logger)
    async def _forward_segment(self, segment: rtc.TranscriptionSegment) -> None:
        """Forward transcription segment to LiveKit room."""
        if not self._room.isconnected():
            return

        if self._last_segment_id != segment.id:
            self._played_text = ""
            self._last_segment_id = segment.id
        self._played_text += segment.text

        if not self._forward_delta:
            segment = rtc.TranscriptionSegment(**asdict(segment))
            segment.text = self._played_text

        transcription = rtc.Transcription(
            participant_identity=self._participant_identity,
            track_sid=self._track_id,
            segments=[segment],
        )

        try:
            await self._room.local_participant.publish_transcription(transcription)
        except Exception:
            logger.exception("Failed to publish transcription")


class TranscriptionStreamForwarder(TranscriptionForwarder):
    """Forwards transcriptions to text streams."""

    def __init__(self, stream: TextIO = sys.stdout):
        super().__init__()
        self._stream = stream

    @log_exceptions(logger=logger)
    async def _forward_segment(self, segment: rtc.TranscriptionSegment) -> None:
        """Forward transcription segment to the stream with real-time display."""
        self._stream.write(segment.text)
        self._stream.flush()

        if segment.final:
            self._stream.write("\n")
            self._stream.flush()
