import asyncio
import sys
import time
from typing import Optional

from livekit import rtc

from ._utils import find_micro_track_id


class TTSRoomForwarder:
    """Forwards synchronized TTS segments to a LiveKit room."""

    def __init__(
        self,
        room: rtc.Room,
        participant: rtc.Participant | str,
        track: rtc.Track | rtc.TrackPublication | str | None = None,
    ):
        identity = participant if isinstance(participant, str) else participant.identity
        if track is None:
            track = find_micro_track_id(room, identity)
        elif isinstance(track, (rtc.TrackPublication, rtc.Track)):
            track = track.sid

        self._room = room
        self._participant_identity = identity
        self._track_id = track

    def __call__(self, segment: rtc.TranscriptionSegment) -> None:
        """Forward transcription segment to LiveKit room."""
        if not self._room.isconnected():
            return

        transcription = rtc.Transcription(
            participant_identity=self._participant_identity,
            track_sid=self._track_id,
            segments=[segment],
        )
        asyncio.create_task(
            self._room.local_participant.publish_transcription(transcription)
        )


class TTSStdoutForwarder:
    """Forwards synchronized TTS segments to stdout with real-time display."""

    def __init__(self, show_timing: bool = False):
        self._show_timing = show_timing
        self._start_time: Optional[float] = None
        self._last_text = ""

    def reset(self) -> None:
        self._start_time = None
        self._last_text = ""

    def __call__(self, segment: rtc.TranscriptionSegment) -> None:
        """Forward transcription segment to console with real-time display."""
        text = segment.text
        if text == self._last_text:
            return

        # Clear the line and write the new text
        sys.stdout.write("\r" + " " * len(self._last_text) + "\r")

        if self._show_timing and self._start_time is None:
            self._start_time = time.time()

        if self._show_timing:
            elapsed = time.time() - self._start_time
            timing = f"[{elapsed:.2f}s] "
            sys.stdout.write(timing)

        sys.stdout.write(text)
        sys.stdout.flush()
        self._last_text = text

        if segment.final:
            if self._show_timing:
                sys.stdout.write(" [END]")
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._last_text = ""
            # self._start_time = None
