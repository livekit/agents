from __future__ import annotations

import abc
import asyncio
import sys
from typing import Optional

from livekit import rtc

from ..log import logger
from ..pipeline.io import AudioSink, PlaybackFinishedEvent, TextSink
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
            self._forward_task = asyncio.create_task(self._forward_segment_stream())

    @abc.abstractmethod
    async def _forward_segments(self, segment: rtc.TranscriptionSegment) -> None:
        """Forward a single segment to the target destination."""
        pass

    async def _forward_segment_stream(self) -> None:
        """Process segments from the stream."""
        try:
            async for segment in self.segment_stream:
                await self._forward_segments(segment)
        except Exception:
            logger.exception("Error processing segment stream")

    async def run(self, audio_sink: AudioSink, text_sink: TextSink) -> None:
        """Run the forwarder, forwarding segments from the audio and text sinks."""

        if self._forward_task is None:
            self._forward_task = asyncio.create_task(self._forward_segment_stream())

        # Set up audio sink event handlers
        def on_audio_frame(frame: rtc.AudioFrame, capturing: bool) -> None:
            if not capturing:
                self.segment_playout_started()
            self.push_audio(frame)

        def on_audio_flush() -> None:
            self.mark_audio_segment_end()

        def on_audio_clear() -> None:
            self.mark_audio_segment_end()

        def on_playback_finished(ev: PlaybackFinishedEvent) -> None:
            self.segment_playout_finished()

        # Set up text sink event handlers
        def on_text(text: str) -> None:
            self.push_text(text)

        def on_text_flush() -> None:
            self.mark_text_segment_end()

        try:
            audio_sink.on("capture_frame", on_audio_frame)
            audio_sink.on("flush", on_audio_flush)
            audio_sink.on("clear", on_audio_clear)
            audio_sink.on("playback_finished", on_playback_finished)

            text_sink.on("capture_text", on_text)
            text_sink.on("flush", on_text_flush)

            # Wait for segment stream to complete
            if self._forward_task:
                await self._forward_task

        finally:
            # Clean up event handlers
            audio_sink.off("capture_frame", on_audio_frame)
            audio_sink.off("flush", on_audio_flush)
            audio_sink.off("clear", on_audio_clear)
            audio_sink.off("playback_finished", on_playback_finished)

            text_sink.off("capture_text", on_text)
            text_sink.off("flush", on_text_flush)

    async def aclose(self) -> None:
        """Close both syncer and forwarder."""
        await super().aclose()
        if self._forward_task:
            await self._forward_task


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
    async def _forward_segments(self, segment: rtc.TranscriptionSegment) -> None:
        """Forward transcription segment to LiveKit room."""
        if not self._room.isconnected():
            return

        transcription = rtc.Transcription(
            participant_identity=self._participant_identity,
            track_sid=self._track_id,
            segments=[segment],
        )

        try:
            await self._room.local_participant.publish_transcription(transcription)
        except Exception:
            logger.exception("Failed to publish transcription")


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
        self._last_text = ""

    def segment_playout_started(self) -> None:
        if self._show_timing:
            self._start_time = asyncio.get_running_loop().time()
        return super().segment_playout_started()

    @log_exceptions(logger=logger)
    async def _forward_segments(self, segment: rtc.TranscriptionSegment) -> None:
        """Forward transcription segment to console with real-time display."""
        text = segment.text
        if text == self._last_text:
            return

        # Clear the line and write the new text
        sys.stdout.write("\r" + " " * len(self._last_text) + "\r")

        if self._show_timing and self._start_time is not None:
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
