from __future__ import annotations

import asyncio

from livekit import rtc

from ... import utils
from ...log import logger
from ...types import (
    ATTRIBUTE_PUBLISH_ON_BEHALF,
    ATTRIBUTE_TRANSCRIPTION_FINAL,
    ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID,
    ATTRIBUTE_TRANSCRIPTION_TRACK_ID,
    TOPIC_TRANSCRIPTION,
)
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
        super().__init__(next_in_chain=None, sample_rate=sample_rate)
        self._room = room
        self._lock = asyncio.Lock()
        self._audio_source = rtc.AudioSource(sample_rate, num_channels, queue_size_ms)
        self._publish_options = track_publish_options
        self._publication: rtc.LocalTrackPublication | None = None
        self._subscribed_fut = asyncio.Future[None]()

        # used to republish track on reconnection
        self._republish_task: asyncio.Task[None] | None = None
        self._flush_task: asyncio.Task[None] | None = None
        self._interrupted_event = asyncio.Event()

        self._pushed_duration: float = 0.0
        self._interrupted: bool = False

    async def _publish_track(self) -> None:
        async with self._lock:
            track = rtc.LocalAudioTrack.create_audio_track("roomio_audio", self._audio_source)
            self._publication = await self._room.local_participant.publish_track(
                track, self._publish_options
            )
            await self._publication.wait_for_subscription()
            if not self._subscribed_fut.done():
                self._subscribed_fut.set_result(None)

    @property
    def subscribed(self) -> asyncio.Future[None]:
        return self._subscribed_fut

    async def start(self) -> None:
        await self._publish_track()
        self._room.on("reconnected", self._on_reconnected)

    async def aclose(self) -> None:
        self._room.off("reconnected", self._on_reconnected)
        if self._republish_task:
            await utils.aio.cancel_and_wait(self._republish_task)
        if self._flush_task:
            await utils.aio.cancel_and_wait(self._flush_task)

        await self._audio_source.aclose()

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await self._subscribed_fut

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

        self._flush_task = asyncio.create_task(self._wait_for_playout())

    def clear_buffer(self) -> None:
        if not self._pushed_duration:
            return
        self._interrupted_event.set()

    async def _wait_for_playout(self) -> None:
        wait_for_interruption = asyncio.create_task(self._interrupted_event.wait())
        wait_for_playout = asyncio.create_task(self._audio_source.wait_for_playout())
        await asyncio.wait(
            [wait_for_playout, wait_for_interruption],
            return_when=asyncio.FIRST_COMPLETED,
        )

        interrupted = wait_for_interruption.done()
        pushed_duration = self._pushed_duration

        if interrupted:
            pushed_duration = max(pushed_duration - self._audio_source.queued_duration, 0)
            self._audio_source.clear_queue()
            wait_for_playout.cancel()
        else:
            wait_for_interruption.cancel()

        self._pushed_duration = 0
        self._interrupted_event.clear()
        self.on_playback_finished(playback_position=pushed_duration, interrupted=interrupted)

    def _on_reconnected(self) -> None:
        if self._republish_task:
            self._republish_task.cancel()
        self._republish_task = asyncio.create_task(self._publish_track())


class _ParticipantLegacyTranscriptionOutput(io.TextOutput):
    def __init__(
        self,
        room: rtc.Room,
        *,
        is_delta_stream: bool = True,
        participant: rtc.Participant | str | None = None,
    ):
        super().__init__(next_in_chain=None)
        self._room, self._is_delta_stream = room, is_delta_stream
        self._track_id: str | None = None
        self._participant_identity: str | None = None

        # identity of the participant that on behalf of the current participant
        self._represented_by: str | None = None

        self._room.on("track_published", self._on_track_published)
        self._room.on("local_track_published", self._on_local_track_published)
        self._flush_task: asyncio.Task[None] | None = None

        self._reset_state()
        self.set_participant(participant)

    def set_participant(
        self,
        participant: rtc.Participant | str | None,
    ) -> None:
        self._participant_identity = (
            participant.identity if isinstance(participant, rtc.Participant) else participant
        )
        self._represented_by = self._participant_identity
        if self._participant_identity is None:
            return

        # find track id from existing participants
        if self._participant_identity == self._room.local_participant.identity:
            for local_track in self._room.local_participant.track_publications.values():
                self._on_local_track_published(local_track, local_track.track)
                if self._track_id is not None:
                    break
        if not self._track_id:
            for p in self._room.remote_participants.values():
                if not self._is_local_proxy_participant(p):
                    continue
                for remote_track in p.track_publications.values():
                    self._on_track_published(remote_track, p)
                    if self._track_id is not None:
                        break

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
            participant_identity=self._represented_by or self._participant_identity,
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
            if self._room.isconnected():
                await self._room.local_participant.publish_transcription(transcription)
        except Exception as e:
            logger.warning("failed to publish transcription", exc_info=e)

    def _on_track_published(
        self, track: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if (
            not self._is_local_proxy_participant(participant)
            or track.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return

        self._track_id = track.sid
        self._represented_by = participant.identity

    def _on_local_track_published(
        self, track: rtc.LocalTrackPublication, _: rtc.Track | None
    ) -> None:
        if (
            self._participant_identity is None
            or self._participant_identity != self._room.local_participant.identity
            or track.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return

        self._track_id = track.sid

    def _is_local_proxy_participant(self, participant: rtc.Participant) -> bool:
        if not self._participant_identity:
            return False

        if participant.identity == self._participant_identity or (
            (on_behalf := participant.attributes.get(ATTRIBUTE_PUBLISH_ON_BEHALF)) is not None
            and on_behalf == self._participant_identity
        ):
            return True

        return False


class _ParticipantTranscriptionOutput(io.TextOutput):
    def __init__(
        self,
        room: rtc.Room,
        *,
        is_delta_stream: bool = True,
        participant: rtc.Participant | str | None = None,
    ):
        super().__init__(next_in_chain=None)
        self._room, self._is_delta_stream = room, is_delta_stream
        self._track_id: str | None = None
        self._participant_identity: str | None = None

        self._writer: rtc.TextStreamWriter | None = None

        self._room.on("track_published", self._on_track_published)
        self._room.on("local_track_published", self._on_local_track_published)
        self._flush_atask: asyncio.Task[None] | None = None

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
        attributes[ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID] = self._current_id

        return await self._room.local_participant.stream_text(
            topic=TOPIC_TRANSCRIPTION,
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

        try:
            if self._room.isconnected():
                if self._is_delta_stream:  # reuse the existing writer
                    if self._writer is None:
                        self._writer = await self._create_text_writer()

                    await self._writer.write(text)
                else:  # always create a new writer
                    tmp_writer = await self._create_text_writer()
                    await tmp_writer.write(text)
                    await tmp_writer.aclose()
        except Exception as e:
            logger.warning("failed to publish transcription", exc_info=e)

    async def _flush_task(self, writer: rtc.TextStreamWriter | None) -> None:
        attributes = {
            ATTRIBUTE_TRANSCRIPTION_FINAL: "true",
        }
        if self._track_id:
            attributes[ATTRIBUTE_TRANSCRIPTION_TRACK_ID] = self._track_id

        try:
            if self._room.isconnected():
                if self._is_delta_stream:
                    if writer:
                        await writer.aclose(attributes=attributes)
                else:
                    tmp_writer = await self._create_text_writer(attributes=attributes)
                    await tmp_writer.write(self._latest_text)
                    await tmp_writer.aclose()
        except Exception as e:
            logger.warning("failed to publish transcription", exc_info=e)

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
    def __init__(
        self, sinks: list[io.TextOutput], *, next_in_chain: io.TextOutput | None = None
    ) -> None:
        super().__init__(next_in_chain=next_in_chain)
        self._sinks = sinks

    async def capture_text(self, text: str) -> None:
        await asyncio.gather(*[sink.capture_text(text) for sink in self._sinks])

    def flush(self) -> None:
        for sink in self._sinks:
            sink.flush()
