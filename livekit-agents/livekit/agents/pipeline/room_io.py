from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator, Optional

from livekit import rtc

from .. import stt, utils
from ..log import logger
from .io import AudioSink, MultiTextSink, TextSink
from .transcription import TextSynchronizer, find_micro_track_id

if TYPE_CHECKING:
    from ..pipeline import PipelineAgent


@dataclass(frozen=True)
class RoomInputOptions:
    text_enabled: bool = True
    """Whether to subscribe to text input"""
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
LK_TEXT_INPUT_TOPIC = "lk.room_text_input"


class BaseStreamHandle:
    """Base class for handling audio/video streams from a participant"""

    def __init__(self, room: rtc.Room, name: str = "") -> None:
        self._room = room
        self._name = name

        self._participant_identity: Optional[str] = None
        self._stream: Optional[rtc.VideoStream | rtc.AudioStream] = None
        self._track: Optional[rtc.Track] = None
        self._stream_ready = asyncio.Event()

        self._tasks: set[asyncio.Task] = set()
        self._room.on("track_subscribed", self._on_track_subscribed)

    @property
    def stream(self) -> AsyncIterator[rtc.AudioFrame | rtc.VideoFrame] | None:
        return self._read_stream()

    def set_participant(self, participant_identity: str | None) -> None:
        """Start streaming from the participant"""
        if participant_identity is None:
            self._participant_identity = None
            self._close_stream()
            return

        if self._participant_identity == participant_identity:
            return

        if (
            self._participant_identity
            and self._participant_identity != participant_identity
        ):
            self._close_stream()

        self._participant_identity = participant_identity

        participant = self._room.remote_participants.get(participant_identity)
        if participant:
            for publication in participant.track_publications.values():
                if publication.track:
                    self._on_track_subscribed(
                        publication.track, publication, participant
                    )

    async def aclose(self) -> None:
        if self._stream:
            await self._stream.aclose()
            self._stream = None
            self._track = None
            self._stream_ready.clear()

    def _create_stream(
        self, track: rtc.Track
    ) -> Optional[rtc.VideoStream | rtc.AudioStream]:
        raise NotImplementedError()

    def _close_stream(self) -> None:
        """Close the current stream"""
        if self._stream is None:
            return

        # TODO(long): stream cannot be closed if it's created from participant?
        # self._stream._queue.put(None)
        task = asyncio.create_task(self._stream.aclose())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self._stream = None
        self._track = None
        self._stream_ready.clear()

    @utils.log_exceptions(logger=logger)
    async def _read_stream(self) -> AsyncIterator[rtc.AudioFrame | rtc.VideoFrame]:
        while True:
            if self._stream is None:
                await self._stream_ready.wait()
                continue

            stream = self._stream
            logger.debug(
                "start reading stream",
                extra={
                    "participant": self._participant_identity,
                    "stream_name": self._name,
                },
            )
            try:
                async for event in stream:
                    yield event.frame

                logger.debug(
                    "stream ended",
                    extra={
                        "participant": self._participant_identity,
                        "stream_name": self._name,
                    },
                )
                if stream is self._stream:
                    self._close_stream()
            except Exception:
                logger.exception(f"Error reading {self._name} stream")

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if (
            not self._participant_identity
            or self._participant_identity != participant.identity
        ):
            return

        if self._track and self._track.sid == track.sid:
            return

        new_stream = self._create_stream(track)
        if new_stream and self._stream:
            self._close_stream()
        self._stream = new_stream
        self._track = track
        self._stream_ready.set()


class AudioStreamHandle(BaseStreamHandle):
    def __init__(
        self,
        room: rtc.Room,
        *,
        sample_rate: int = 24000,
        num_channels: int = 1,
        capacity: int = 0,
    ) -> None:
        super().__init__(room=room, name="audio")
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.capacity = capacity

    def _create_stream(self, track: rtc.Track) -> rtc.AudioStream:
        return rtc.AudioStream.from_track(
            track=track,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
            capacity=self.capacity,
        )


class VideoStreamHandle(BaseStreamHandle):
    def __init__(self, room: rtc.Room, *, capacity: int = 0) -> None:
        super().__init__(room=room, name="video")
        self.capacity = capacity

    def _create_stream(self, track: rtc.Track) -> rtc.VideoStream:
        return rtc.VideoStream.from_track(track=track, capacity=self.capacity)


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
        self._agent: Optional["PipelineAgent"] = None
        self._tasks: set[asyncio.Task] = set()

        # target participant
        self._participant_identity: Optional[str] = (
            participant.identity
            if isinstance(participant, rtc.RemoteParticipant)
            else participant
        )
        self._participant_connected = asyncio.Future[rtc.RemoteParticipant]()

        # transcription forwarder
        self._text_sink: Optional[TextSink] = None

        # streams
        self._audio_handle: Optional[AudioStreamHandle] = None
        self._video_handle: Optional[VideoStreamHandle] = None
        if options.audio_enabled:
            self._audio_handle = AudioStreamHandle(
                room=room,
                sample_rate=options.audio_sample_rate,
                num_channels=options.audio_num_channels,
                capacity=options.audio_queue_capacity,
            )
            self._audio_handle.set_participant(self._participant_identity)

        if options.video_enabled:
            self._video_handle = VideoStreamHandle(
                room=room,
                capacity=options.video_queue_capacity,
            )
            self._video_handle.set_participant(self._participant_identity)

        self._room.on("participant_connected", self._on_participant_connected)
        self._room.on("participant_disconnected", self._on_participant_disconnected)
        for participant in self._room.remote_participants.values():
            self._on_participant_connected(participant)

        # text input from datastream
        if options.text_enabled:
            self._room.register_text_stream_handler(
                LK_TEXT_INPUT_TOPIC, self._on_text_input
            )

    @property
    def audio(self) -> AsyncIterator[rtc.AudioFrame] | None:
        if not self._audio_handle:
            return None
        return self._audio_handle.stream

    @property
    def video(self) -> AsyncIterator[rtc.VideoFrame] | None:
        if not self._video_handle:
            return None
        return self._video_handle.stream

    @property
    def linked_participant(self) -> rtc.RemoteParticipant | None:
        if not self._participant_connected.done():
            return None
        return self._participant_connected.result()

    async def start(self, agent: Optional["PipelineAgent"] = None) -> None:
        participant = await self.wait_for_participant()
        if self._participant_identity is None:
            # link to the first connected participant if not set
            self.set_participant(participant.identity)

        # TODO(long): should we force the agent to be set or provide a set_agent method?
        self._agent = agent
        if not self._agent:
            return

        agent.input.audio = self.audio
        agent.input.video = self.video
        if self._options.forward_user_transcript:
            # TODO: support multiple participants
            self._text_sink = MultiTextSink(
                [
                    RoomTranscriptEventSink(
                        room=self._room,
                        participant=self._participant,
                        is_delta_stream=False,
                    ),
                    DataStreamSink(
                        room=self._room,
                        participant=self._participant,
                        is_delta_stream=False,
                    ),
                ]
            )
            agent.on("user_transcript_updated", self._on_user_transcript_updated)

    def set_participant(self, participant_identity: str | None) -> None:
        """Switch audio and video streams to specified participant"""
        if participant_identity is None:
            self.unset_participant()
            return

        # reset future if switching to a different participant
        if (
            self._participant_identity is not None
            and self._participant_identity != participant_identity
        ):
            self._participant_connected = asyncio.Future[rtc.RemoteParticipant]()
            # check if new participant is already connected
            for participant in self._room.remote_participants.values():
                if participant.identity == participant_identity:
                    self._participant_connected.set_result(participant)
                    break

        # update participant identity and handlers
        self._participant_identity = participant_identity
        if self._audio_handle:
            self._audio_handle.set_participant(participant_identity)
        if self._video_handle:
            self._video_handle.set_participant(participant_identity)

        # update text sink if user transcript forwarding is enabled
        if self._options.forward_user_transcript:
            if self._text_sink:
                self._text_sink.set_participant(participant_identity)
            else:
                self._text_sink = RoomTranscriptEventSink(
                    room=self._room, participant=participant_identity, is_stream=False
                )

        logger.debug(
            "set participant",
            extra={"participant": participant_identity},
        )

    def unset_participant(self) -> None:
        self._participant_identity = None
        self._participant_connected = asyncio.Future[rtc.RemoteParticipant]()
        if self._audio_handle:
            self._audio_handle.set_participant(None)
        if self._video_handle:
            self._video_handle.set_participant(None)
        self._text_sink = None
        logger.debug("unset participant")

    async def wait_for_participant(self) -> rtc.RemoteParticipant:
        return await self._participant_connected

    def _on_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        logger.debug(
            "participant connected",
            extra={"participant": participant.identity},
        )
        if self._participant_connected.done():
            return

        if self._participant_identity is not None:
            if participant.identity != self._participant_identity:
                return
        # otherwise, skip participants that are marked as publishing for this agent
        elif (
            participant.attributes.get(LK_PUBLISH_FOR_ATTR)
            == self._room.local_participant.identity
        ):
            return

        self._participant_connected.set_result(participant)

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        logger.debug(
            "participant disconnected",
            extra={"participant": participant.identity},
        )
        if (
            self._participant_identity is None
            or self._participant_identity != participant.identity
        ):
            return

        self._participant_connected = asyncio.Future[rtc.RemoteParticipant]()

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

    def _on_text_input(
        self, reader: rtc.TextStreamReader, participant_identity: str
    ) -> None:
        if participant_identity != self._participant_identity:
            return

        async def _read_text():
            if not self._agent:
                return

            text = await reader.read_all()
            logger.debug(
                "received text input",
                extra={"text": text, "participant": self._participant_identity},
            )
            self._agent.interrupt()
            self._agent.generate_reply(user_input=text)

        task = asyncio.create_task(_read_text())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        self._room.off("participant_connected", self._on_participant_connected)
        self._room.off("participant_disconnected", self._on_participant_disconnected)

        if self._audio_handle:
            await self._audio_handle.aclose()
        if self._video_handle:
            await self._video_handle.aclose()

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
        self._text_sink: Optional[TextSink] = None
        self._text_synchronizer: Optional[TextSynchronizer] = None

    async def start(self, agent: Optional["PipelineAgent"] = None) -> None:
        await self._audio_sink.start()

        if self._options.forward_agent_transcription:
            self._text_sink = MultiTextSink(
                [
                    RoomTranscriptEventSink(
                        room=self._room, participant=self._room.local_participant
                    ),
                    DataStreamSink(
                        room=self._room,
                        participant=self._room.local_participant,
                        topic="lk.chat",
                    ),
                ]
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
        is_delta_stream: bool = True,
    ):
        super().__init__()
        self._room = room
        self._tasks: set[asyncio.Task] = set()
        self._track_id: str | None = None
        self._is_stream = is_delta_stream
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

    async def capture_text(self, text: str, *, segment_id: str | None = None) -> None:
        if not self._capturing:
            self._capturing = True
            self._pushed_text = ""
            self._current_id = segment_id or utils.shortuuid("SG_")

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


class DataStreamSink(TextSink):
    """TextSink implementation that publishes transcriptions as text streams to a LiveKit room"""

    def __init__(
        self,
        room: rtc.Room,
        participant: rtc.Participant | str,
        track: rtc.Track | rtc.TrackPublication | str | None = None,
        topic: str | None = None,
        is_delta_stream: bool = True,
    ):
        super().__init__()
        self._room = room
        self._tasks: set[asyncio.Task] = set()
        self.set_participant(participant, track)
        self._topic = topic or "lk.chat"
        self._is_delta_stream = is_delta_stream
        self._text_writer: rtc.TextStreamWriter | None = None
        self._is_capturing = False

    def set_participant(
        self,
        participant: rtc.Participant | str,
        track: rtc.Track | rtc.TrackPublication | str | None = None,
    ) -> None:
        identity = participant if isinstance(participant, str) else participant.identity
        self._participant_identity = identity
        self._latest_text = ""

    async def capture_text(self, text: str, *, segment_id: str | None = None) -> None:
        if segment_id is not None and segment_id != self._current_id:
            self.flush()

        self._latest_text = text
        if not self._is_capturing:
            self._current_id = segment_id or utils.shortuuid("SG_")
            self._is_capturing = True

        try:
            if not self._text_writer:
                self._is_capturing = True
                self._text_writer = await self._room.local_participant.stream_text(
                    topic=self._topic,
                    stream_id=self._current_id,
                    sender_identity=self._participant_identity,
                    attributes={
                        "lk.transcription_final": "false",
                    },
                )
            await self._text_writer.write(text)

            if not self._is_delta_stream:
                # close non-delta stream immediately after writing
                await self._text_writer.aclose()
                self._text_writer = None
        except Exception:
            logger.exception("Failed to publish transcription to stream")

    def flush(self) -> None:
        attributes = {
            "lk.transcription_final": "true",
        }

        self._is_capturing = False

        async def _close_writer():
            if not self._is_delta_stream:
                writer = await self._room.local_participant.stream_text(
                    topic=self._topic,
                    stream_id=self._current_id,
                    sender_identity=self._participant_identity,
                    attributes=attributes,
                )
                await writer.write(self._latest_text)
                await writer.aclose(attributes=attributes)
            else:
                if not self._text_writer:
                    return
                await self._text_writer.aclose(attributes=attributes)

        task = asyncio.create_task(_close_writer())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self._text_writer = None
