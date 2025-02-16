from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator, Optional

from livekit import rtc

from .. import stt, utils
from ..log import logger
from .io import AudioSink, TextSink
from .transcription import TextSynchronizer, find_micro_track_id

if TYPE_CHECKING:
    from ..pipeline import PipelineAgent


@dataclass(frozen=True)
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
    forward_user_transcript: bool = True
    """Whether to forward user transcript segments to the room"""


@dataclass(frozen=True)
class RoomOutputOptions:
    sample_rate: int = 24000
    num_channels: int = 1
    sync_agent_transcription: bool = True
    """Whether to synchronize agent transcription with audio playback"""
    forward_agent_transcription: bool = True
    """Whether to forward transcription segments to the room"""
    track_source: rtc.TrackSource.ValueType = rtc.TrackSource.SOURCE_MICROPHONE
    """Source of the audio track to publish"""


DEFAULT_ROOM_INPUT_OPTIONS = RoomInputOptions()
DEFAULT_ROOM_OUTPUT_OPTIONS = RoomOutputOptions()
LK_PUBLISH_FOR_ATTR = "lk.publish_for"


class BaseStreamHandler:
    """Base class for handling audio/video streams from a participant"""

    def __init__(self, room: rtc.Room, enabled: bool = True) -> None:
        self._room = room
        self._enabled = enabled
        self._participant: Optional[rtc.RemoteParticipant] = None
        self._stream: Optional[rtc.VideoStream | rtc.AudioStream] = None
        self._stream_connected = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        if not value:
            self.close()

    async def link_participant(self, participant: rtc.RemoteParticipant) -> None:
        """Start streaming from the participant"""
        if not self._enabled:
            return

        self.close()
        self._participant = participant
        self._stream = await self._create_stream(participant)
        self._stream_connected.set()

    async def _create_stream(
        self, participant: rtc.RemoteParticipant
    ) -> rtc.VideoStream | rtc.AudioStream:
        raise NotImplementedError()

    def close(self) -> None:
        """Close the current stream"""
        self._participant = None
        if self._stream is None:
            return

        # TODO(long): stream cannot be closed by aclose()?
        # self._stream._queue.put(None)
        task = asyncio.create_task(self._stream.aclose())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self._stream = None
        self._stream_connected.clear()

    def read_stream(self) -> AsyncIterator[rtc.AudioFrame | rtc.VideoFrame] | None:
        if not self._enabled:
            return None
        return self._read_stream()

    @utils.log_exceptions(logger=logger)
    async def _read_stream(self) -> AsyncIterator[rtc.AudioFrame | rtc.VideoFrame]:
        while True:
            if self._stream is None:
                await self._stream_connected.wait()
                continue
            stream = self._stream

            logger.info(
                "start reading stream",
                extra={
                    "participant": self._participant.identity,
                    "stream_type": self.__class__.__name__,
                },
            )
            try:
                async for event in stream:
                    yield event.frame
                if self._participant and stream is self._stream:
                    logger.warning(
                        "stream ended before participant was unlinked",
                        extra={
                            "participant": self._participant.identity,
                            "stream_type": self.__class__.__name__,
                        },
                    )
                    self.close()
            except Exception:
                logger.exception(f"Error reading {self.__class__.__name__} stream")

    async def _wait_for_track(
        self,
        participant: rtc.RemoteParticipant,
        track_source: rtc.TrackSource.ValueType,
    ) -> rtc.Track:
        for publication in participant.track_publications.values():
            if publication.source == track_source and publication.track:
                return publication.track

        track_subscribed = asyncio.Future[rtc.Track]()

        def _on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            p: rtc.RemoteParticipant,
        ) -> None:
            if (
                p.identity == participant.identity
                and publication.source == track_source
            ):
                track_subscribed.set_result(track)

        self._room.on("track_subscribed", _on_track_subscribed)
        track = await track_subscribed
        self._room.off("track_subscribed", _on_track_subscribed)
        return track


class AudioStreamHandler(BaseStreamHandler):
    def __init__(
        self,
        room: rtc.Room,
        enabled: bool = True,
        sample_rate: int = 24000,
        num_channels: int = 1,
        capacity: int = 0,
    ) -> None:
        super().__init__(room=room, enabled=enabled)
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.capacity = capacity

    async def _create_stream(
        self, participant: rtc.RemoteParticipant
    ) -> rtc.AudioStream:
        # return rtc.AudioStream.from_participant(
        #     participant=participant,
        #     track_source=rtc.TrackSource.SOURCE_MICROPHONE,
        #     sample_rate=self.sample_rate,
        #     num_channels=self.num_channels,
        #     capacity=self.capacity,
        # )

        track = await self._wait_for_track(
            participant, rtc.TrackSource.SOURCE_MICROPHONE
        )
        return rtc.AudioStream.from_track(
            track=track,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
            capacity=self.capacity,
        )


class VideoStreamHandler(BaseStreamHandler):
    def __init__(self, room: rtc.Room, enabled: bool = True, capacity: int = 0) -> None:
        super().__init__(room=room, enabled=enabled)
        self.capacity = capacity

    async def _create_stream(
        self, participant: rtc.RemoteParticipant
    ) -> rtc.VideoStream:
        return rtc.VideoStream.from_participant(
            participant=participant,
            track_source=rtc.TrackSource.SOURCE_CAMERA,
            capacity=self.capacity,
        )


class RoomInput:
    """Creates video and audio streams from remote participants in a LiveKit room"""

    def __init__(
        self,
        room: rtc.Room,
        participant: Optional[rtc.RemoteParticipant | str] = None,
        options: RoomInputOptions = DEFAULT_ROOM_INPUT_OPTIONS,
    ) -> None:
        """
        Args:
            room: The LiveKit room to get streams from
            participant_identity: Participant automatically linked to. If not specified,
                                will link the first participant without
                                {LK_PUBLISH_FOR_ATTR: <local_identity>} attribute.
            options: RoomInputOptions
        """
        self._options = options
        self._room = room
        self._closed = False
        self._tasks: set[asyncio.Task] = set()

        # target participant
        self._target_participant_identity: Optional[str] = (
            participant.identity
            if isinstance(participant, rtc.RemoteParticipant)
            else participant
        )
        self._active_participant: rtc.RemoteParticipant | None = None
        self._participant_connected = asyncio.Event()

        # transcription forwarder
        self._text_sink: Optional[RoomTranscriptEventSink] = None

        # streams
        self._audio_handler = AudioStreamHandler(
            room=room,
            enabled=options.audio_enabled,
            sample_rate=options.audio_sample_rate,
            num_channels=options.audio_num_channels,
            capacity=options.audio_queue_capacity,
        )
        self._video_handler = VideoStreamHandler(
            room=room,
            enabled=options.video_enabled,
            capacity=options.video_queue_capacity,
        )

        self._room.on("participant_connected", self._on_participant_connected)
        self._room.on("participant_disconnected", self._on_participant_disconnected)
        for participant in self._room.remote_participants.values():
            self._on_participant_connected(participant)

    async def start(self, agent: Optional["PipelineAgent"] = None) -> None:
        participant = await self.wait_for_participant()
        if not agent:
            return

        agent.input.audio = self.audio
        agent.input.video = self.video
        if self._options.forward_user_transcript:
            self._text_sink = RoomTranscriptEventSink(
                room=self._room, participant=participant, is_stream=False
            )
            agent.on("user_transcript_updated", self._on_user_transcript_updated)

    @property
    def audio(self) -> AsyncIterator[rtc.AudioFrame] | None:
        return self._audio_handler.read_stream()

    @property
    def video(self) -> AsyncIterator[rtc.VideoFrame] | None:
        return self._video_handler.read_stream()

    @property
    def active_participant(self) -> rtc.RemoteParticipant | None:
        """Get currently active participant"""
        return self._active_participant

    async def link_participant(
        self,
        participant: rtc.RemoteParticipant | str,
        *,
        wait_for_connection: bool = False,
    ) -> None:
        """Switch audio and video streams to specified participant"""
        target_identity = (
            participant.identity
            if isinstance(participant, rtc.RemoteParticipant)
            else participant
        )
        if (
            self._active_participant
            and self._active_participant.identity == target_identity
        ):
            return

        self.unlink_participant()
        self._target_participant_identity = target_identity
        if target_identity in self._room.remote_participants:
            participant = self._room.remote_participants[target_identity]
        else:
            if not wait_for_connection:
                logger.error(
                    "participant not connected",
                    extra={"participant_identity": target_identity},
                )
                raise ValueError(f"Participant {target_identity} not connected")
            participant = await self.wait_for_participant()

        self._active_participant = participant
        await self._video_handler.link_participant(participant)
        await self._audio_handler.link_participant(participant)
        if self._text_sink:
            self._text_sink.set_participant(participant)

        self._participant_connected.set()
        logger.debug(
            "linked participant",
            extra={"participant_identity": participant.identity},
        )

    def unlink_participant(self) -> None:
        """Unlink the current participant"""
        if self._active_participant is None:
            return

        self._target_participant_identity = None
        unlinked_identity = self._active_participant.identity
        self._active_participant = None
        self._participant_connected.clear()
        self._audio_handler.close()
        self._video_handler.close()
        logger.debug(
            "unlinked participant",
            extra={"participant_identity": unlinked_identity},
        )

    async def wait_for_participant(self) -> rtc.RemoteParticipant:
        await self._participant_connected.wait()
        assert self._active_participant is not None
        return self._active_participant

    def _on_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        logger.debug(
            "participant connected",
            extra={"participant_identity": participant.identity},
        )
        if self._active_participant:
            return

        if self._target_participant_identity is not None:
            if participant.identity != self._target_participant_identity:
                return
        # otherwise, skip participants that are marked as publishing for this agent
        elif (
            participant.attributes.get(LK_PUBLISH_FOR_ATTR)
            == self._room.local_participant.identity
        ):
            return

        task = asyncio.create_task(self.link_participant(participant))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        logger.debug(
            "participant disconnected",
            extra={"participant_identity": participant.identity},
        )
        if (
            not self._active_participant
            or self._active_participant.identity != participant.identity
        ):
            return

        self.unlink_participant()

    def _on_user_transcript_updated(self, ev: stt.SpeechEvent) -> None:
        if self._text_sink is None:
            return

        async def _capture_text():
            if ev.alternatives:
                data = ev.alternatives[0]
                await self._text_sink.capture_text(data.text)

            if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                self._text_sink.flush()

        task = asyncio.create_task(_capture_text())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        if self._closed:
            raise RuntimeError("RoomInput already closed")

        self._closed = True
        self._room.off("participant_connected", self._on_participant_connected)
        self._room.off("participant_disconnected", self._on_participant_disconnected)

        self._active_participant = None
        self._audio_handler.close()
        self._video_handler.close()

        # cancel and wait for all pending tasks
        await utils.aio.cancel_and_wait(*self._tasks)
        self._tasks.clear()


class RoomOutput:
    """Manages audio output to a LiveKit room"""

    def __init__(
        self, room: rtc.Room, options: RoomOutputOptions = DEFAULT_ROOM_OUTPUT_OPTIONS
    ) -> None:
        self._options = options
        self._room = room
        self._audio_sink = RoomAudioSink(
            room=room,
            sample_rate=self._options.sample_rate,
            num_channels=self._options.num_channels,
            track_source=self._options.track_source,
        )
        self._text_sink: Optional[RoomTranscriptEventSink] = None
        self._text_synchronizer: Optional[TextSynchronizer] = None

    async def start(self, agent: Optional["PipelineAgent"] = None) -> None:
        await self._audio_sink.start()

        if self._options.forward_agent_transcription:
            self._text_sink = RoomTranscriptEventSink(
                room=self._room, participant=self._room.local_participant
            )

        if self._options.sync_agent_transcription and self._text_sink:
            self._text_synchronizer = TextSynchronizer(
                audio_sink=self._audio_sink, text_sink=self._text_sink
            )

        if agent:
            agent.output.audio = self.audio
            agent.output.text = self.text

    @property
    def audio(self) -> AudioSink:
        if self._text_synchronizer:
            return self._text_synchronizer.audio_sink
        return self._audio_sink

    @property
    def text(self) -> TextSink | None:
        if self._text_synchronizer:
            return self._text_synchronizer.text_sink
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
        track_source: rtc.TrackSource.ValueType = rtc.TrackSource.SOURCE_MICROPHONE,
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
        self._track_source = track_source
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

        self._tasks: set[asyncio.Task] = set()

        def _on_reconnected(self) -> None:
            self._publication = None
            task = asyncio.create_task(self.start())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

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
            options=rtc.TrackPublishOptions(source=self._track_source),
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


class RoomTranscriptEventSink(TextSink):
    """TextSink implementation that publishes transcription segments to a LiveKit room"""

    def __init__(
        self,
        room: rtc.Room,
        participant: rtc.Participant | str,
        *,
        track: rtc.Track | rtc.TrackPublication | str | None = None,
        is_stream: bool = True,
    ):
        super().__init__()
        self._room = room
        self._tasks: set[asyncio.Task] = set()
        self._track_id: str | None = None
        self._is_stream = is_stream
        self.set_participant(participant, track)

    def set_participant(
        self,
        participant: rtc.Participant | str,
        track: rtc.Track | rtc.TrackPublication | str | None = None,
    ) -> None:
        identity = participant if isinstance(participant, str) else participant.identity
        self._participant_identity = identity

        if isinstance(track, (rtc.TrackPublication, rtc.Track)):
            track = track.sid
        else:
            try:
                track = find_micro_track_id(self._room, identity)
            except ValueError:
                track = None

        self._track_id = track
        if track is None:
            self._room.on("track_published", self._on_track_published)

        self._capturing = False
        self._pushed_text = ""
        self._current_id = utils.shortuuid("SG_")

    async def capture_text(self, text: str) -> None:
        if not self._capturing:
            self._capturing = True
            self._pushed_text = ""
            self._current_id = utils.shortuuid("SG_")

        if self._is_stream:
            self._pushed_text += text
        else:
            self._pushed_text = text
        await self._publish_transcription(
            self._current_id, self._pushed_text, final=False
        )

    def flush(self) -> None:
        if not self._capturing:
            return
        self._capturing = False
        task = asyncio.create_task(
            self._publish_transcription(self._current_id, self._pushed_text, final=True)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _publish_transcription(self, id: str, text: str, final: bool) -> None:
        if self._track_id is None:
            logger.warning(
                "track not found, skipping transcription publish",
                extra={
                    "participant_identity": self._participant_identity,
                    "text": text,
                    "final": final,
                },
            )
            return

        transcription = rtc.Transcription(
            participant_identity=self._participant_identity,
            track_sid=self._track_id,
            segments=[
                rtc.TranscriptionSegment(
                    id=id,
                    text=text,
                    start_time=0,
                    end_time=0,
                    final=final,
                    language="",
                )
            ],
        )
        try:
            await self._room.local_participant.publish_transcription(transcription)
        except Exception:
            logger.exception("Failed to publish transcription")

    def _on_track_published(
        self, track: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if self._track_id is not None:
            return
        if (
            participant.identity != self._participant_identity
            or track.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return
        self._track_id = track.sid
        self._room.off("track_published", self._on_track_published)
