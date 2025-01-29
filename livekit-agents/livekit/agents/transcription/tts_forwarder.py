import asyncio
import sys
from abc import ABC, abstractmethod

from livekit import rtc

from ..utils import aio
from ._utils import find_micro_track_id


class TTSForwarder(ABC):
    def __init__(self):
        self._event_ch = aio.Chan[rtc.TranscriptionSegment]()
        self._main_task = asyncio.create_task(self._main_task())

    @abstractmethod
    async def _forward_segment(self, segment: rtc.TranscriptionSegment) -> None:
        pass

    def __call__(self, segment: rtc.TranscriptionSegment) -> None:
        self._event_ch.send_nowait(segment)

    async def _main_task(self) -> None:
        async for segment in self._event_ch:
            await self._forward_segment(segment)

    async def aclose(self) -> None:
        self._event_ch.close()
        await aio.cancel_and_wait(self._main_task)


class TTSRoomForwarder(TTSForwarder):
    """Forwards synchronized TTS segments to a LiveKit room."""

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

    async def _forward_segment(self, segment: rtc.TranscriptionSegment) -> None:
        """Forward transcription segment to LiveKit room."""
        if not self._room.isconnected():
            return

        transcription = rtc.Transcription(
            participant_identity=self._participant_identity,
            track_sid=self._track_id,
            segments=[segment],
        )
        await self._room.local_participant.publish_transcription(transcription)


class TTSStreamForwarder(TTSForwarder):
    """Forwards synchronized TTS segments to a text IO stream with real-time display."""

    def __init__(self, stream=sys.stdout):
        super().__init__()
        self._stream = stream
        self._last_text = ""

    async def _forward_segment(self, segment: rtc.TranscriptionSegment) -> None:
        """Forward transcription segment to the stream with real-time display."""
        text = segment.text
        if text == self._last_text and not segment.final:
            return

        # Find the appended text by comparing with last text
        if text.startswith(self._last_text):
            new_text = text[len(self._last_text) :]
        else:
            # If the new text doesn't start with the old text,
            # it's a completely new segment
            new_text = text
            if self._last_text:
                self._stream.write("\n")

        self._stream.write(new_text)
        self._stream.flush()
        self._last_text = text

        if segment.final:
            self._stream.write("\n")
            self._stream.flush()
            self._last_text = ""
