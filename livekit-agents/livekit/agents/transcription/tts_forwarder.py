from __future__ import annotations

import abc
import asyncio
import sys
import time
from typing import Optional

from livekit import rtc

from ..log import logger
from ..pipeline.io import AudioSink, TextSink
from ..utils import log_exceptions
from . import _utils
from .transcription_sync import TranscriptionSyncIO, TranscriptionSyncOptions


class TTSSegmentsForwarder(TranscriptionSyncIO, abc.ABC):
    """Base class for forwarding TTS segments with synchronized timing."""

    def __init__(
        self,
        audio_sink: AudioSink,
        text_sink: Optional[TextSink] = None,
        sync_options: TranscriptionSyncOptions | None = None,
    ):
        super().__init__(
            audio_sink=audio_sink,
            text_sink=text_sink,
            sync_options=sync_options,
        )
        self.on("transcription_segment", self._forward_segments)

    @abc.abstractmethod
    def _forward_segments(self, segment: rtc.TranscriptionSegment) -> None:
        """Forward a single segment to the target destination."""
        pass


class TTSRoomForwarder(TTSSegmentsForwarder):
    """Forwards synchronized TTS segments to a LiveKit room."""

    def __init__(
        self,
        *,
        room: rtc.Room,
        participant: rtc.Participant | str,
        audio_sink: AudioSink,
        text_sink: Optional[TextSink] = None,
        sync_options: TranscriptionSyncOptions | None = None,
        track: rtc.Track | rtc.TrackPublication | str | None = None,
    ):
        super().__init__(audio_sink, text_sink, sync_options=sync_options)
        identity = participant if isinstance(participant, str) else participant.identity
        if track is None:
            track = _utils.find_micro_track_id(room, identity)
        elif isinstance(track, (rtc.TrackPublication, rtc.Track)):
            track = track.sid

        self._room = room
        self._participant_identity = identity
        self._track_id = track

    @log_exceptions(logger=logger)
    def _forward_segments(self, segment: rtc.TranscriptionSegment) -> None:
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


class TTSStdoutForwarder(TTSSegmentsForwarder):
    """Forwards synchronized TTS segments to stdout with real-time display."""

    def __init__(
        self,
        audio_sink: AudioSink,
        text_sink: Optional[TextSink] = None,
        sync_options: TranscriptionSyncOptions | None = None,
        show_timing: bool = False,
    ):
        super().__init__(audio_sink, text_sink, sync_options=sync_options)
        self._show_timing = show_timing
        self._start_time: Optional[float] = None
        self._last_text = ""

        self.on("segment_playout_started", self._reset)

    def _reset(self) -> None:
        self._start_time = None
        self._last_text = ""

    @log_exceptions(logger=logger)
    def _forward_segments(self, segment: rtc.TranscriptionSegment) -> None:
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
