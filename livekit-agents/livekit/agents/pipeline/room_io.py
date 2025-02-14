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

        # transcription forwarder
        self._text_sink: Optional[RoomTranscriptEventSink] = None

        # streams
        self._audio_stream: Optional[rtc.AudioStream] = None
        self._video_stream: Optional[rtc.VideoStream] = None

        self._participant_ready = asyncio.Event()
        self._room.on("participant_connected", self._on_participant_connected)

        # try to find participant
        # TODO: support link and unlink participants
        if self._expected_identity is not None:
            participant = self._room.remote_participants.get(self._expected_identity)
            if participant is not None:
                self._link_participant(participant)
        else:
            for participant in self._room.remote_participants.values():
                self._link_participant(participant)
                if self._participant:
                    break

        self._tasks: set[asyncio.Task] = set()

    async def start(self, agent: Optional["PipelineAgent"] = None) -> None:
        await self.wait_for_participant()
        if not agent:
            return

        agent.input.audio = self.audio
        agent.input.video = self.video
        if self._options.forward_user_transcript:
            # TODO: support multiple participants
            self._text_sink = RoomTranscriptEventSink(
                room=self._room, participant=self._participant, capture_delta=False
            )
            agent.on("user_transcript_updated", self._on_user_transcript_updated)

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
        should_ignore = (
            participant.attributes.get(LK_PUBLISH_FOR_ATTR)
            == self._room.local_participant.identity
        )
        if should_ignore or (
            self._expected_identity is not None
            and participant.identity != self._expected_identity
        ):
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

    async def wait_for_participant(self) -> rtc.RemoteParticipant:
        await self._participant_ready.wait()
        assert self._participant is not None
        return self._participant

    async def aclose(self) -> None:
        if self._closed:
            raise RuntimeError("RoomInput already closed")

        self._closed = True
        self._room.off("participant_connected", self._on_participant_connected)
        self._participant = None

        # cancel and wait for all pending tasks
        await utils.aio.cancel_and_wait(*self._tasks)
        self._tasks.clear()

        if self._audio_stream is not None:
            await self._audio_stream.aclose()
            self._audio_stream = None
        if self._video_stream is not None:
            await self._video_stream.aclose()
            self._video_stream = None


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
