from __future__ import annotations

import asyncio

from livekit import rtc

from ... import utils
from ...log import logger
from ...types import ATTRIBUTE_TRANSCRIPTION_FINAL, ATTRIBUTE_TRANSCRIPTION_TRACK_ID, TOPIC_CHAT
from .. import io
from ..transcription import find_micro_track_id


class _ParticipantAudioOutput(io.AudioOutput):
    def __init__(
        self,
        room: rtc.Room,
        *,
        sample_rate: int,
        num_channels: int,
        track_publish_options: rtc.TrackPublishOptions,
        queue_size_ms: int = 100_000,  # TODO(long): move buffer to python
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        self._room = room
        self._lock = asyncio.Lock()
        self._audio_source = rtc.AudioSource(sample_rate, num_channels, queue_size_ms)
        self._publish_options = track_publish_options
        self._publication: rtc.LocalTrackPublication | None = None
        self._republish_task: asyncio.Task | None = None  # used to republish track on reconnection
        self._flush_task: asyncio.Task | None = None

        self._pushed_duration: float = 0.0
        self._interrupted: bool = False

    async def _publish_track(self) -> None:
        async with self._lock:
            track = rtc.LocalAudioTrack.create_audio_track("roomio_audio", self._audio_source)
            self._publication = await self._room.local_participant.publish_track(
                track, self._publish_options
            )
            await self._publication.wait_for_subscription()

    async def start(self) -> None:
        await self._publish_track()

        def _on_reconnected() -> None:
            if self._republish_task:
                self._republish_task.cancel()
            self._republish_task = asyncio.create_task(self._publish_track())

        self._room.on("reconnected", _on_reconnected)

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

        if self._flush_task and not self._flush_task.done():
            logger.error("capture_frame called while flush is in progress")
            await self._flush_task

        self._pushed_duration += frame.duration
        await self._audio_source.capture_frame(frame)

    def flush(self) -> None:
        super().flush()

        if not self._pushed_duration:
            return

        if self._flush_task and not self._flush_task.done():
            # shouldn't happen if only one active speech handle at a time
            logger.error("flush called while playback is in progress")
            self._flush_task.cancel()

        def _playback_finished(task: asyncio.Task[None]) -> None:
            pushed_duration, interrupted = self._pushed_duration, self._interrupted
            self._pushed_duration = 0
            self._interrupted = False
            self.on_playback_finished(playback_position=pushed_duration, interrupted=interrupted)

        self._flush_task = asyncio.create_task(self._audio_source.wait_for_playout())
        self._flush_task.add_done_callback(_playback_finished)

    def clear_buffer(self) -> None:
        super().clear_buffer()

        if not self._pushed_duration:
            return

        queued_duration = self._audio_source.queued_duration
        self._pushed_duration = max(self._pushed_duration - queued_duration, 0)
        self._interrupted = True
        self._audio_source.clear_queue()


class _ParticipantLegacyTranscriptionOutput(io.TextOutput):
    def __init__(
        self,
        room: rtc.Room,
        *,
        is_delta_stream: bool = True,
        participant: rtc.Participant | str | None = None,
    ):
        super().__init__()
        self._room, self._is_delta_stream = room, is_delta_stream
        self._track_id: str | None = None
        self._participant_identity: str | None = None

        self._room.on("track_published", self._on_track_published)
        self._room.on("local_track_published", self._on_local_track_published)
        self._flush_task: asyncio.Task | None = None

        self._reset_state()
        self.set_participant(participant)

    def set_participant(
        self,
        participant: rtc.Participant | str | None,
    ) -> None:
        self._participant_identity = (
            participant.identity if isinstance(participant, rtc.Participant) else participant
        )
        if self._participant_identity is None:
            return

        try:
            self._track_id = find_micro_track_id(self._room, self._participant_identity)
        except ValueError:
            return

        self.flush()
        self._reset_state()

    def _reset_state(self) -> None:
        self._current_id = utils.shortuuid("SG_")
        self._capturing = False
        self._pushed_text = ""

    @utils.log_exceptions(logger=logger)
    async def capture_text(self, text: str) -> None:
        if self._participant_identity is None or self._track_id is None:
            return

        if self._flush_task and not self._flush_task.done():
            await self._flush_task

        if not self._capturing:
            self._reset_state()
            self._capturing = True

        if self._is_delta_stream:
            self._pushed_text += text
        else:
            self._pushed_text = text

        await self._publish_transcription(self._current_id, self._pushed_text, final=False)

    @utils.log_exceptions(logger=logger)
    def flush(self) -> None:
        if self._participant_identity is None or self._track_id is None or not self._capturing:
            return

        self._flush_task = asyncio.create_task(
            self._publish_transcription(self._current_id, self._pushed_text, final=True)
        )
        self._reset_state()

    async def _publish_transcription(self, id: str, text: str, final: bool) -> None:
        if self._participant_identity is None or self._track_id is None:
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
        except Exception as e:
            logger.warning("failed to publish transcription", exc_info=e)

    def _on_track_published(
        self, track: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if (
            self._participant_identity is None
            or participant.identity != self._participant_identity
            or track.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return

        self._track_id = track.sid

    def _on_local_track_published(self, track: rtc.LocalTrackPublication, _: rtc.Track) -> None:
        if (
            self._participant_identity is None
            or self._participant_identity != self._room.local_participant.identity
            or track.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return

        self._track_id = track.sid


class _ParticipantTranscriptionOutput(io.TextOutput):
    def __init__(
        self,
        room: rtc.Room,
        *,
        is_delta_stream: bool = True,
        participant: rtc.Participant | str | None = None,
    ):
        super().__init__()
        self._room, self._is_delta_stream = room, is_delta_stream
        self._track_id: str | None = None
        self._participant_identity: str | None = None

        self._writer: rtc.TextStreamWriter | None = None

        self._room.on("track_published", self._on_track_published)
        self._room.on("local_track_published", self._on_local_track_published)
        self._flush_atask: asyncio.Task | None = None

        self._reset_state()
        self.set_participant(participant)

    def set_participant(
        self,
        participant: rtc.Participant | str | None,
    ) -> None:
        self._participant_identity = (
            participant.identity if isinstance(participant, rtc.Participant) else participant
        )
        if self._participant_identity is None:
            return

        try:
            self._track_id = find_micro_track_id(self._room, self._participant_identity)
        except ValueError:
            # track id is optional for TextStream when audio is not published
            self._track_id = None

        self.flush()
        self._reset_state()

    def _reset_state(self) -> None:
        self._current_id = utils.shortuuid("SG_")
        self._capturing = False
        self._latest_text = ""

    async def _create_text_writer(
        self, attributes: dict[str, str] | None = None
    ) -> rtc.TextStreamWriter:
        assert self._participant_identity is not None, "participant_identity is not set"

        if not attributes:
            attributes = {
                ATTRIBUTE_TRANSCRIPTION_FINAL: "false",
            }
            if self._track_id:
                attributes[ATTRIBUTE_TRANSCRIPTION_TRACK_ID] = self._track_id

        return await self._room.local_participant.stream_text(
            topic=TOPIC_CHAT,
            stream_id=self._current_id,
            sender_identity=self._participant_identity,
            attributes=attributes,
        )

    @utils.log_exceptions(logger=logger)
    async def capture_text(self, text: str) -> None:
        if self._participant_identity is None:
            return

        if self._flush_atask and not self._flush_atask.done():
            await self._flush_atask

        if not self._capturing:
            self._reset_state()
            self._capturing = True

        self._latest_text = text

        if self._is_delta_stream:  # reuse the existing writer
            if self._writer is None:
                self._writer = await self._create_text_writer()

            await self._writer.write(text)
        else:  # always create a new writer
            tmp_writer = await self._create_text_writer()
            await tmp_writer.write(text)
            await tmp_writer.aclose()

    async def _flush_task(self, writer: rtc.TextStreamWriter | None):
        attributes = {
            ATTRIBUTE_TRANSCRIPTION_FINAL: "true",
        }
        if self._track_id:
            attributes[ATTRIBUTE_TRANSCRIPTION_TRACK_ID] = self._track_id

        if self._is_delta_stream:
            if writer:
                await writer.aclose(attributes=attributes)
        else:
            tmp_writer = await self._create_text_writer(attributes=attributes)
            await tmp_writer.write(self._latest_text)
            await tmp_writer.aclose()

    def flush(self) -> None:
        if self._participant_identity is None or not self._capturing:
            return

        self._capturing = False
        curr_writer = self._writer
        self._writer = None
        self._flush_atask = asyncio.create_task(self._flush_task(curr_writer))

    def _on_track_published(
        self, track: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if (
            self._participant_identity is None
            or participant.identity != self._participant_identity
            or track.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return

        self._track_id = track.sid

    def _on_local_track_published(self, track: rtc.LocalTrackPublication, _: rtc.Track) -> None:
        if (
            self._participant_identity is None
            or self._participant_identity != self._room.local_participant.identity
            or track.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return

        self._track_id = track.sid


# Keep this utility private for now
class _ParallelTextOutput(io.TextOutput):
    def __init__(self, *sinks: io.TextOutput) -> None:
        self._sinks = sinks

    async def capture_text(self, text: str) -> None:
        await asyncio.gather(*[sink.capture_text(text) for sink in self._sinks])

    def flush(self) -> None:
        for sink in self._sinks:
            sink.flush()
