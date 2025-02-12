from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Literal, Optional

from livekit import rtc

from .. import utils
from ..transcription import (
    TranscriptionRoomForwarder,
    TranscriptionSyncIO,
    TranscriptSegment,
)
from .io import AudioSink, TextSink
from ..log import logger


@dataclass
class RoomInputOptions:
    audio_enabled: bool = True
    """Whether to subscribe to audio"""
    video_enabled: bool = False
    """Whether to subscribe to video"""
    audio_sample_rate: int = 24000
    """Sample rate of the input audio in Hz"""
    audio_num_channels: int = 1
    """Number of audio channels"""
    audio_queue_capacity: int = 0
    """Capacity of the internal audio queue, 0 means unlimited"""
    video_queue_capacity: int = 0
    """Capacity of the internal video queue, 0 means unlimited"""


@dataclass
class RoomOutputOptions:
    sample_rate: int = 24000
    num_channels: int = 1
    forward_transcription: bool = True
    """Whether to forward transcription segments to the room"""
    sync_transcription: bool = True
    """Whether to sync transcription segments with audio playback"""


DEFAULT_ROOM_INPUT_OPTIONS = RoomInputOptions()
DEFAULT_ROOM_OUTPUT_OPTIONS = RoomOutputOptions()


class RoomInput:
    """Creates video and audio streams from a remote participant in a LiveKit room"""

    def __init__(
        self,
        room: rtc.Room,
        participant_identity: Optional[str] = None,
        options: RoomInputOptions = DEFAULT_ROOM_INPUT_OPTIONS,
    ) -> None:
        """
        Args:
            room: The LiveKit room to get streams from
            participant_identity: Optional identity of the participant to get streams from.
                                If None, will use the first participant that joins.
            options: RoomInputOptions
        """
        self._options = options
        self._room = room
        self._expected_identity = participant_identity
        self._participant: rtc.RemoteParticipant | None = None
        self._closed = False

        # streams
        self._audio_stream: Optional[rtc.AudioStream] = None
        self._video_stream: Optional[rtc.VideoStream] = None

        self._participant_ready = asyncio.Event()
        self._room.on("participant_connected", self._on_participant_connected)

        # try to find participant
        if self._expected_identity is not None:
            participant = self._room.remote_participants.get(self._expected_identity)
            if participant is not None:
                self._link_participant(participant)
        else:
            for participant in self._room.remote_participants.values():
                self._link_participant(participant)
                if self._participant:
                    break

    async def wait_for_participant(self) -> rtc.RemoteParticipant:
        await self._participant_ready.wait()
        assert self._participant is not None
        return self._participant

    @property
    def audio(self) -> AsyncIterator[rtc.AudioFrame] | None:
        if self._audio_stream is None:
            return None

        async def _read_stream():
            async for event in self._audio_stream:
                yield event.frame

        return _read_stream()

    @property
    def video(self) -> AsyncIterator[rtc.VideoFrame] | None:
        if self._video_stream is None:
            return None

        async def _read_stream():
            async for event in self._video_stream:
                yield event.frame

        return _read_stream()

    def _link_participant(self, participant: rtc.RemoteParticipant) -> None:
        if (
            self._expected_identity is not None
            and participant.identity != self._expected_identity
        ):
            return

        if self._expected_identity is None and participant.metadata == "avatar_worker":
            # ignore the avatar worker participant
            return

        self._participant = participant

        # set up tracks
        if self._options.audio_enabled:
            self._audio_stream = rtc.AudioStream.from_participant(
                participant=participant,
                track_source=rtc.TrackSource.SOURCE_MICROPHONE,
                sample_rate=self._options.audio_sample_rate,
                num_channels=self._options.audio_num_channels,
                capacity=self._options.audio_queue_capacity,
            )
        if self._options.video_enabled:
            self._video_stream = rtc.VideoStream.from_participant(
                participant=participant,
                track_source=rtc.TrackSource.SOURCE_CAMERA,
                capacity=self._options.video_queue_capacity,
            )

        self._participant_ready.set()

    def _on_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        if self._participant is not None:
            return
        self._link_participant(participant)

    async def aclose(self) -> None:
        if self._closed:
            raise RuntimeError("RoomInput already closed")

        self._closed = True
        self._room.off("participant_connected", self._on_participant_connected)
        self._participant = None

        if self._audio_stream is not None:
            await self._audio_stream.aclose()
            self._audio_stream = None
        if self._video_stream is not None:
            await self._video_stream.aclose()
            self._video_stream = None


class RoomOutput(rtc.EventEmitter[Literal["agent_transcript_updated"]]):
    """Manages audio output to a LiveKit room"""

    def __init__(
        self, room: rtc.Room, options: RoomOutputOptions = DEFAULT_ROOM_OUTPUT_OPTIONS
    ) -> None:
        super().__init__()

        self._options = options
        self._room = room
        self._room_audio_sink = RoomAudioSink(
            room=room,
            sample_rate=self._options.sample_rate,
            num_channels=self._options.num_channels,
        )

        self._tr_forwarder: Optional[TranscriptionRoomForwarder] = None
        self._tr_sync: Optional[TranscriptionSyncIO] = None
        if not self._options.sync_transcription:
            self._audio_sink = self._room_audio_sink
            self._text_sink = _TranscriptionTextSink()
            self._text_sink.on(
                "transcription_updated",
                lambda ev: self.emit("agent_transcript_updated", ev),
            )
        else:
            self._tr_sync = TranscriptionSyncIO(self._room_audio_sink)
            self._audio_sink = self._tr_sync.audio_output
            self._text_sink = self._tr_sync.text_output
            self._tr_sync.on(
                "transcription_updated",
                lambda ev: self.emit("agent_transcript_updated", ev),
            )

    async def start(self) -> None:
        await self._room_audio_sink.start()

        if self._options.forward_transcription:
            self._tr_forwarder = TranscriptionRoomForwarder(
                room=self._room, participant=self._room.local_participant
            )
            self.on("agent_transcript_updated", self._tr_forwarder.update)

    @property
    def audio(self) -> AudioSink:
        return self._audio_sink

    @property
    def text(self) -> TextSink:
        return self._text_sink


class RoomAudioSink(AudioSink):
    """AudioSink implementation that publishes audio to a LiveKit room"""

    def __init__(
        self,
        room: rtc.Room,
        *,
        sample_rate: int = 24000,
        num_channels: int = 1,
        queue_size_ms: int = 100_000,
    ) -> None:
        """Initialize the RoomAudioSink

        Args:
            room: The LiveKit room to publish audio to
            sample_rate: Sample rate of the audio in Hz
            num_channels: Number of audio channels
            queue_size_ms: Size of the internal audio queue in ms.
                Default to 100s to capture as fast as possible.
        """
        super().__init__(sample_rate=sample_rate)
        self._room = room

        # buffer the audio frames as soon as they are captured
        self._audio_source = rtc.AudioSource(
            sample_rate=sample_rate,
            num_channels=num_channels,
            queue_size_ms=queue_size_ms,
        )

        self._publication: rtc.LocalTrackPublication | None = None
        self._pushed_duration: Optional[float] = None
        self._interrupted: bool = False
        self._flush_task: Optional[asyncio.Task[None]] = None

        def _on_reconnected(self) -> None:
            self._publication = None
            asyncio.create_task(self.start())

        self._room.on("reconnected", _on_reconnected)

    async def start(self) -> None:
        """Start publishing the audio track to the room"""
        if self._publication:
            return

        track = rtc.LocalAudioTrack.create_audio_track(
            "assistant_voice", self._audio_source
        )
        self._publication = await self._room.local_participant.publish_track(
            track=track,
            options=rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        await self._publication.wait_for_subscription()

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Capture an audio frame and publish it to the room"""
        await super().capture_frame(frame)

        if self._pushed_duration is None:
            self._pushed_duration = 0.0
            self._interrupted = False  # reset interrupted flag
        self._pushed_duration += frame.duration
        await self._audio_source.capture_frame(frame)

    def flush(self) -> None:
        """Flush the current audio segment and notify when complete"""
        super().flush()
        if self._pushed_duration is None:
            return
        if self._flush_task and not self._flush_task.done():
            # shouldn't happen if only one active speech handle at a time
            logger.error("flush called while playback is in progress")
            self._flush_task.cancel()
            self._flush_task = None

        def _playback_finished(task: asyncio.Task[None]) -> None:
            pushed_duration, interrupted = self._pushed_duration, self._interrupted
            self._pushed_duration = None
            self._interrupted = False
            self.on_playback_finished(
                playback_position=pushed_duration, interrupted=interrupted
            )

        self._flush_task = asyncio.create_task(self._audio_source.wait_for_playout())
        self._flush_task.add_done_callback(_playback_finished)

    def clear_buffer(self) -> None:
        """Clear the audio buffer immediately"""
        super().clear_buffer()
        if self._pushed_duration is None:
            return

        queued_duration = self._audio_source.queued_duration
        self._pushed_duration = max(0, self._pushed_duration - queued_duration)
        self._interrupted = True
        self._audio_source.clear_queue()


class _TranscriptionTextSink(
    TextSink, rtc.EventEmitter[Literal["transcription_updated"]]
):
    def __init__(self) -> None:
        super().__init__()
        self._current_id = utils.shortuuid("SG_")

    async def capture_text(self, text: str) -> None:
        self.emit(
            "transcription_updated",
            self._create_transcription_segment(text, final=False),
        )

    def flush(self) -> None:
        self.emit(
            "transcription_updated",
            self._create_transcription_segment("", final=True),
        )

    def _create_transcription_segment(
        self, text: str, final: bool
    ) -> TranscriptSegment:
        segment = TranscriptSegment(
            id=self._current_id,
            text=text,
            start_time=0,
            end_time=0,
            final=final,
            is_delta=True,
            language="",
        )
        if final:
            self._current_id = utils.shortuuid("SG_")
        return segment
