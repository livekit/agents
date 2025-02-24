from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator, Generic, Optional, TypeVar

from livekit import rtc

from .. import stt, utils
from ..log import logger
from ..types import ATTRIBUTE_AGENT_STATE, AgentState
from .io import AudioSink, ParallelTextSink, TextSink
from .transcription import TextSynchronizer, find_micro_track_id

if TYPE_CHECKING:
    from ..pipeline import PipelineAgent


ATTRIBUTE_PUBLISH_FOR = "lk.publish_for"
ATTRIBUTE_TRANSCRIPTION_FINAL = "lk.transcription_final"
TOPIC_CHAT = "lk.chat"


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
    text_input_topic: str | None = TOPIC_CHAT
    """Topic for text input"""
    _audio_queue_capacity: int = 0
    """Capacity of the internal audio queue, 0 means unlimited"""
    _video_queue_capacity: int = 0
    """Capacity of the internal video queue, 0 means unlimited"""


@dataclass(frozen=True)
class RoomOutputOptions:
    text_enabled: bool = True
    """Whether to publish text output"""
    audio_enabled: bool = True
    """Whether to publish audio output"""
    audio_sample_rate: int = 24000
    """Sample rate of the output audio in Hz"""
    audio_num_channels: int = 1
    """Number of audio channels"""
    text_output_topic: str | None = TOPIC_CHAT
    """Topic for text output"""
    agent_text_identity: str | None = None
    """Identity of the sender for text output, default to the local participant"""
    agent_text_sync_with_audio: bool = True
    """Whether to synchronize agent transcription with audio playback"""
    audio_track_source: rtc.TrackSource.ValueType = rtc.TrackSource.SOURCE_MICROPHONE
    """Source of the audio track to publish"""


DEFAULT_ROOM_INPUT_OPTIONS = RoomInputOptions()
DEFAULT_ROOM_OUTPUT_OPTIONS = RoomOutputOptions()


class RoomIO:
    def __init__(
        self,
        room: rtc.Room,
        agent: "PipelineAgent",
        *,
        link_to_participant: Optional[rtc.RemoteParticipant | str] = None,
        input_options: RoomInputOptions = DEFAULT_ROOM_INPUT_OPTIONS,
        output_options: RoomOutputOptions = DEFAULT_ROOM_OUTPUT_OPTIONS,
    ) -> None:
        self._room = room
        self._agent = agent
        self._in_opts = input_options
        self._out_opts = output_options

        # room input
        self._participant_identity: Optional[str] = (
            link_to_participant.identity
            if isinstance(link_to_participant, rtc.RemoteParticipant)
            else link_to_participant
        )
        self._participant_connected = asyncio.Future[rtc.RemoteParticipant]()
        self._audio_input_handle: Optional[AudioStreamHandle] = None
        self._video_input_handle: Optional[VideoStreamHandle] = None

        # room output
        self._audio_output: Optional[RoomAudioSink] = None
        self._user_text_output: Optional[TextSink] = None
        self._agent_text_output: Optional[TextSink] = None
        self._text_synchronizer: Optional[TextSynchronizer] = None

        self._tasks: set[asyncio.Task] = set()
        self._update_state_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._room.on("participant_connected", self._on_participant_connected)
        self._room.on("participant_disconnected", self._on_participant_disconnected)
        for participant in self._room.remote_participants.values():
            self._on_participant_connected(participant)

        # room input setup
        if self._in_opts.text_enabled:
            self._room.register_text_stream_handler(
                topic=TOPIC_CHAT, handler=self._on_user_text_input
            )
        if self._in_opts.audio_enabled:
            self._audio_input_handle = AudioStreamHandle(
                room=self._room,
                sample_rate=self._in_opts.audio_sample_rate,
                num_channels=self._in_opts.audio_num_channels,
                capacity=self._in_opts._audio_queue_capacity,
            )
        if self._in_opts.video_enabled:
            self._video_input_handle = VideoStreamHandle(
                room=self._room,
                capacity=self._in_opts._video_queue_capacity,
            )

        # room output setup
        if self._out_opts.audio_enabled:
            self._audio_output = RoomAudioSink(
                room=self._room,
                sample_rate=self._out_opts.audio_sample_rate,
                num_channels=self._out_opts.audio_num_channels,
                track_source=self._out_opts.audio_track_source,
            )

        if self._out_opts.text_enabled:
            self._user_text_output = self._create_text_sink(
                participant_identity=self._participant_identity,
                is_delta_stream=False,
                topic=self._out_opts.text_output_topic,
            )
            self._agent.on("user_transcript_updated", self._on_user_transcript_updated)

            self._agent_text_output = self._create_text_sink(
                participant_identity=(
                    self._out_opts.agent_text_identity
                    or self._room.local_participant.identity
                ),
                is_delta_stream=True,
                topic=self._out_opts.text_output_topic,
            )
            if self._out_opts.agent_text_sync_with_audio:
                audio_output = self._audio_output or self._agent.output.audio
                if not audio_output:
                    logger.warning(
                        "text sync with audio is enabled but audio output is not enabled, ignoring"
                    )
                else:
                    self._text_synchronizer = TextSynchronizer(
                        audio_sink=audio_output,
                        text_sink=self._agent_text_output,
                    )

        if self._audio_output:
            await self._audio_output.start()

        # wait for the specified participant or the first participant joined
        input_participant = await self.wait_for_participant()
        self.set_participant(input_participant.identity)

        if self.audio_input:
            self._agent.input.audio = self.audio_input
        if self.video_input:
            self._agent.input.video = self.video_input

        if self.audio_output:
            self._agent.output.audio = self.audio_output
        if self.text_output:
            self._agent.output.text = self.text_output

        self._agent.on("agent_state_changed", self._on_agent_state_changed)

        self._room.local_participant.register_rpc_method(
            "set_participant", self.on_set_participant
        )
        self._room.local_participant.register_rpc_method(
            "unset_participant", self.on_unset_participant
        )

    @property
    def audio_output(self) -> AudioSink | None:
        if self._text_synchronizer:
            return self._text_synchronizer.audio_sink
        return self._audio_output

    @property
    def text_output(self) -> TextSink | None:
        if self._text_synchronizer:
            return self._text_synchronizer.text_sink
        return self._agent_text_output

    @property
    def audio_input(self) -> AsyncIterator[rtc.AudioFrame] | None:
        if not self._audio_input_handle:
            return None
        return self._audio_input_handle.stream

    @property
    def video_input(self) -> AsyncIterator[rtc.VideoFrame] | None:
        if not self._video_input_handle:
            return None
        return self._video_input_handle.stream

    @property
    def linked_participant(self) -> rtc.RemoteParticipant | None:
        if not self._participant_connected.done():
            return None
        return self._participant_connected.result()

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
        if self._audio_input_handle:
            self._audio_input_handle.set_participant(participant_identity)
        if self._video_input_handle:
            self._video_input_handle.set_participant(participant_identity)

        self._update_user_text_sink(participant_identity)

        logger.debug(
            "set participant",
            extra={"participant": participant_identity},
        )

    def unset_participant(self) -> None:
        self._participant_identity = None
        self._participant_connected = asyncio.Future[rtc.RemoteParticipant]()
        if self._audio_input_handle:
            self._audio_input_handle.set_participant(None)
        if self._video_input_handle:
            self._video_input_handle.set_participant(None)
        self._update_user_text_sink(None)
        logger.debug("unset participant")

    async def wait_for_participant(self) -> rtc.RemoteParticipant:
        return await self._participant_connected

    # -- RPC methods --
    # user can override these methods to handle RPC calls from the room

    async def on_set_participant(self, data: rtc.RpcInvocationData) -> None:
        target_identity = data.payload or data.caller_identity
        logger.debug(
            "set participant called",
            extra={
                "caller_identity": data.caller_identity,
                "payload": data.payload,
                "target_identity": target_identity,
            },
        )

        self.set_participant(target_identity)

    async def on_unset_participant(self, data: rtc.RpcInvocationData) -> None:
        logger.debug(
            "unset participant called",
            extra={"caller_identity": data.caller_identity, "payload": data.payload},
        )
        self.unset_participant()

    # -- end of RPC methods --

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
            participant.attributes.get(ATTRIBUTE_PUBLISH_FOR)
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
        if self._user_text_output is None:
            return

        async def _capture_text():
            if ev.alternatives:
                data = ev.alternatives[0]
                await self._user_text_output.capture_text(data.text)

            if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                self._user_text_output.flush()

        task = asyncio.create_task(_capture_text())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_user_text_input(
        self, reader: rtc.TextStreamReader, participant_identity: str
    ) -> None:
        if participant_identity != self._participant_identity:
            return

        async def _read_text():
            text = await reader.read_all()
            logger.debug(
                "received text input",
                extra={"text": text, "participant": self._participant_identity},
            )
            if not self._agent._started:
                logger.warning("received text input but agent is not started, ignoring")
                return

            self._agent.interrupt()
            self._agent.generate_reply(user_input=text)

        task = asyncio.create_task(_read_text())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_agent_state_changed(self, state: AgentState):
        @utils.log_exceptions(logger=logger)
        async def _set_state() -> None:
            if self._room.isconnected():
                await self._room.local_participant.set_attributes({
                    ATTRIBUTE_AGENT_STATE: state
                })

        if self._update_state_task is not None:
            self._update_state_task.cancel()

        self._update_state_task = asyncio.create_task(_set_state())

    def _update_user_text_sink(self, participant_identity: str | None) -> None:
        if not self._user_text_output:
            return

        for sink in self._user_text_output._sinks:
            assert isinstance(sink, (DataStreamTextSink, RoomTranscriptEventSink))
            sink.set_participant(participant_identity)

    def _create_text_sink(
        self, participant_identity: str | None, is_delta_stream: bool, topic: str | None
    ) -> TextSink:
        return ParallelTextSink(
            RoomTranscriptEventSink(
                room=self._room,
                participant=participant_identity,
                is_delta_stream=is_delta_stream,
            ),
            DataStreamTextSink(
                room=self._room,
                participant=participant_identity,
                topic=topic,
                is_delta_stream=is_delta_stream,
            ),
        )

    async def aclose(self) -> None:
        self._room.off("participant_connected", self._on_participant_connected)
        self._room.off("participant_disconnected", self._on_participant_disconnected)

        if self._audio_input_handle:
            await self._audio_input_handle.aclose()
        if self._video_input_handle:
            await self._video_input_handle.aclose()

        # cancel and wait for all pending tasks
        await utils.aio.cancel_and_wait(*self._tasks)
        self._tasks.clear()


# -- Output Sinks --
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
        participant: rtc.Participant | str | None,
        *,
        track: rtc.Track | rtc.TrackPublication | str | None = None,
        is_delta_stream: bool = True,
    ):
        super().__init__()
        self._room = room
        self._is_delta_stream = is_delta_stream
        self._tasks: set[asyncio.Task] = set()
        self._track_id: str | None = None
        self._participant_identity: str | None = None

        self._room.on("track_published", self._on_track_published)
        self._room.on("local_track_published", self._on_local_track_published)
        self.set_participant(participant, track)

    def set_participant(
        self,
        participant: rtc.Participant | str | None = None,
        track: rtc.Track | rtc.TrackPublication | str | None = None,
    ) -> None:
        identity = (
            participant.identity
            if isinstance(participant, rtc.Participant)
            else participant
        )
        self._participant_identity = identity

        if identity is None:
            self._track_id = None
        else:
            if isinstance(track, (rtc.TrackPublication, rtc.Track)):
                track = track.sid
            else:
                try:
                    track = find_micro_track_id(self._room, identity)
                except ValueError:
                    track = None
            self._track_id = track

        self._capturing = False
        self._pushed_text = ""
        self._current_id = utils.shortuuid("SG_")

    async def capture_text(self, text: str) -> None:
        if self._participant_identity is None:
            return

        if not self._capturing:
            self._capturing = True
            self._pushed_text = ""
            self._current_id = utils.shortuuid("SG_")

        if self._is_delta_stream:
            self._pushed_text += text
        else:
            self._pushed_text = text
        await self._publish_transcription(
            self._current_id, self._pushed_text, final=False
        )

    def flush(self) -> None:
        if self._participant_identity is None or not self._capturing:
            return
        self._capturing = False
        task = asyncio.create_task(
            self._publish_transcription(self._current_id, self._pushed_text, final=True)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _publish_transcription(self, id: str, text: str, final: bool) -> None:
        if self._participant_identity is None:
            return

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
        if (
            self._track_id is not None
            or self._participant_identity is None
            or participant.identity != self._participant_identity
            or track.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return
        self._track_id = track.sid

    def _on_local_track_published(
        self, track: rtc.LocalTrackPublication, _: rtc.Track
    ) -> None:
        if (
            self._track_id is not None
            or self._participant_identity is None
            or self._participant_identity != self._room.local_participant.identity
            or track.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return
        self._track_id = track.sid


class DataStreamTextSink(TextSink):
    """TextSink implementation that publishes transcriptions as text streams to a LiveKit room"""

    def __init__(
        self,
        room: rtc.Room,
        participant: rtc.Participant | str | None,
        *,
        topic: str | None = None,
        is_delta_stream: bool = True,
    ):
        super().__init__()
        self._room = room
        self._is_delta_stream = is_delta_stream
        self._tasks: set[asyncio.Task] = set()

        self._participant_identity: str | None = None

        self._topic = topic or TOPIC_CHAT
        self._text_writer: rtc.TextStreamWriter | None = None
        self.set_participant(participant)

    def set_participant(
        self,
        participant: rtc.Participant | str | None,
        track: rtc.Track | rtc.TrackPublication | str | None = None,
    ) -> None:
        identity = (
            participant.identity
            if isinstance(participant, rtc.Participant)
            else participant
        )
        if identity != self._participant_identity and self._text_writer:
            # close the previous stream if the participant has changed
            current_writer = self._text_writer
            self._text_writer = None
            task = asyncio.create_task(current_writer.aclose())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        self._participant_identity = identity
        self._latest_text = ""
        self._current_id = utils.shortuuid("SG_")
        self._is_capturing = False

    async def capture_text(self, text: str) -> None:
        if self._participant_identity is None:
            return

        self._latest_text = text
        if not self._is_capturing:
            self._current_id = utils.shortuuid("SG_")
            self._is_capturing = True

        async def _create_writer() -> rtc.TextStreamWriter:
            return await self._room.local_participant.stream_text(
                topic=self._topic,
                stream_id=self._current_id,
                sender_identity=self._participant_identity,
                attributes={ATTRIBUTE_TRANSCRIPTION_FINAL: "false"},
            )

        try:
            if self._is_delta_stream:
                # reuse existing writer for delta streams
                writer = self._text_writer or await _create_writer()
                if not self._text_writer:
                    self._text_writer = writer
            else:
                # always create new writer for non-delta streams
                if self._text_writer:
                    logger.error("non-delta stream should not have an active writer")
                    await self._text_writer.aclose()
                    self._text_writer = None
                writer = await _create_writer()

            await writer.write(text)
            if writer is not self._text_writer:
                await writer.aclose()

        except Exception:
            logger.exception("Failed to publish transcription to stream")

    def flush(self) -> None:
        if not self._is_capturing or self._participant_identity is None:
            return

        self._is_capturing = False

        async def _close_writer(
            writer: rtc.TextStreamWriter | None,
            text: str,
            stream_id: str,
            participant_identity: str | None,
        ):
            attributes = {
                ATTRIBUTE_TRANSCRIPTION_FINAL: "true",
            }
            if not self._is_delta_stream:
                if writer:
                    logger.error("non-delta stream writer not closed")
                    await writer.aclose()
                writer = await self._room.local_participant.stream_text(
                    topic=self._topic,
                    stream_id=stream_id,
                    sender_identity=participant_identity,
                    attributes=attributes,
                )
                await writer.write(text)
                await writer.aclose(attributes=attributes)
            else:
                if not writer:
                    return
                await writer.aclose(attributes=attributes)

        current_writer = self._text_writer
        self._text_writer = None
        task = asyncio.create_task(
            _close_writer(
                current_writer,
                self._latest_text,
                self._current_id,
                self._participant_identity,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


# -- Input Streams --

T = TypeVar("T", bound=rtc.AudioFrame | rtc.VideoFrame)


class BaseStreamHandle(Generic[T]):
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
        self._continuous_stream = self._read_stream()

    @property
    def stream(self) -> AsyncIterator[T] | None:
        return self._continuous_stream

    def set_participant(self, participant_identity: str | None) -> None:
        """Start streaming from the participant"""
        if participant_identity is None:
            self._participant_identity = None
            self._close_current_stream()
            return

        if self._participant_identity == participant_identity:
            return

        if (
            self._participant_identity
            and self._participant_identity != participant_identity
        ):
            self._close_current_stream()

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

    def _close_current_stream(self) -> None:
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

    async def _read_stream(self) -> AsyncIterator[T]:
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
                    self._close_current_stream()
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
            self._close_current_stream()
        self._stream = new_stream
        self._track = track
        self._stream_ready.set()


class AudioStreamHandle(BaseStreamHandle[rtc.AudioFrame]):
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


class VideoStreamHandle(BaseStreamHandle[rtc.VideoFrame]):
    def __init__(self, room: rtc.Room, *, capacity: int = 0) -> None:
        super().__init__(room=room, name="video")
        self.capacity = capacity

    def _create_stream(self, track: rtc.Track) -> rtc.VideoStream:
        return rtc.VideoStream.from_track(track=track, capacity=self.capacity)
