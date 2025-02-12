import asyncio
import sys
from abc import ABC, abstractmethod
from typing import Optional, TextIO

from livekit import rtc
from livekit.rtc.participant import STREAM_CHUNK_SIZE, split_utf8

from ..log import logger
from ..utils import aio, log_exceptions
from ._utils import find_micro_track_id, segment_uuid
from .synchronizer import TranscriptSegment


class TranscriptionForwarder(ABC):
    """Base class for all transcription forwarders."""

    def __init__(self):
        self._event_ch = aio.Chan[TranscriptSegment]()
        self._main_task = asyncio.create_task(self._main_task())
        self._current_id = segment_uuid()

    @abstractmethod
    async def _forward_segment(self, segment: TranscriptSegment) -> None:
        """Forward a single segment to the target destination."""
        pass

    async def _main_task(self) -> None:
        """Process segments from the event channel."""
        try:
            async for segment in self._event_ch:
                await self._forward_segment(segment)
        except Exception:
            logger.exception("Error processing segment stream")

    def update(self, segment: TranscriptSegment) -> None:
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

        self._last_segment_id: Optional[str] = None
        self._played_text = ""

    @log_exceptions(logger=logger)
    async def _forward_segment(self, segment: TranscriptSegment) -> None:
        """Forward transcription segment to LiveKit room."""
        if not self._room.isconnected():
            return

        if self._last_segment_id != segment.id:
            # reset the played text for the new segment
            self._played_text = ""
            self._last_segment_id = segment.id

        if segment.is_delta:
            self._played_text += segment.text
        else:
            self._played_text = segment.text

        transcription = rtc.Transcription(
            participant_identity=self._participant_identity,
            track_sid=self._track_id,
            segments=[
                rtc.TranscriptionSegment(
                    id=segment.id,
                    text=self._played_text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    final=segment.final,
                    language=segment.language,
                )
            ],
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
    async def _forward_segment(self, segment: TranscriptSegment) -> None:
        """Forward transcription segment to the stream with real-time display."""
        self._stream.write(segment.text)
        self._stream.flush()

        if segment.final:
            self._stream.write("\n")
            self._stream.flush()


DEFAULT_TRANSCRIPTION_TOPIC = "lk.transcription"


class TranscriptionDataStreamForwarder(TranscriptionForwarder):
    """Forwards transcription data to a stream."""

    def __init__(
        self,
        room: rtc.Room,
        *,
        destination_identities: Optional[list[str]] = None,
        topic: str = DEFAULT_TRANSCRIPTION_TOPIC,
        attributes: Optional[dict[str, str]] = None,
    ):
        super().__init__()
        self._room = room
        self._destination_identities = destination_identities
        self._topic = topic
        self._attributes = attributes or {}

        self._text_writer: Optional[rtc.TextStreamWriter] = None
        self._last_segment_id: Optional[str] = None

    async def _forward_segment(self, segment: TranscriptSegment) -> None:
        """Write a segment to the text stream.

        If segment is a delta, it appends the segment to the existing stream until
        the segment is final or new segment id is received.

        If segment is not a delta, it creates a new stream for every segment.
        """
        if not self._text_writer or self._last_segment_id != segment.id:
            if self._text_writer:
                await self._text_writer.aclose()

            attributes = {
                **self._attributes,
                "segment_id": segment.id,
                "language": segment.language,
                "mode": "delta" if segment.is_delta else "full",
            }
            self._text_writer = await self._room.local_participant.stream_text(
                destination_identities=self._destination_identities,
                topic=self._topic,
                attributes=attributes,
            )
            self._last_segment_id = segment.id

        for chunk in split_utf8(segment.text, STREAM_CHUNK_SIZE):
            await self._text_writer.write(chunk)

        if segment.final or not segment.is_delta:
            # close the stream when the segment is final or is not a delta
            await self._text_writer.aclose()
            self._text_writer = None
