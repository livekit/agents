from __future__ import annotations

import asyncio
import json
import re
import time

from google.protobuf.json_format import MessageToDict

from livekit import rtc
from livekit.protocol.agent_pb import agent_session as agent_pb

from ... import utils
from ...log import logger
from ...tts._provider_format import (
    ExpressiveTag,
    TranscriptMarkupStripper,
    expression_attribute,
    split_all_markup,
    strip_all_markup,
)
from ...types import (
    ATTRIBUTE_PUBLISH_ON_BEHALF,
    ATTRIBUTE_TRANSCRIPTION_FINAL,
    ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID,
    ATTRIBUTE_TRANSCRIPTION_TRACK_ID,
    TOPIC_TRANSCRIPTION,
    TimedString,
)
from .. import io
from ..transcription import find_micro_track_id

# a complete self-closing expressive marker (<expr/>, <expression/>, or <emotion/>)
_EXPR_MARKER_SPLIT_RE = re.compile(r"(<(?:expr|expression|emotion)\b[^>]*?/\s*>)")


class _ParticipantAudioOutput(io.AudioOutput):
    def __init__(
        self,
        room: rtc.Room,
        *,
        sample_rate: int,
        num_channels: int,
        track_publish_options: rtc.TrackPublishOptions,
        track_name: str = "roomio_audio",
    ) -> None:
        super().__init__(
            label="RoomIO",
            next_in_chain=None,
            sample_rate=sample_rate,
            capabilities=io.AudioOutputCapabilities(pause=True),
        )
        self._room = room
        self._track_name = track_name
        self._lock = asyncio.Lock()
        self._audio_source = rtc.AudioSource(sample_rate, num_channels, queue_size_ms=200)
        self._publish_options = track_publish_options
        self._publication: rtc.LocalTrackPublication | None = None
        self._subscribed_fut = asyncio.Future[None]()

        self._audio_buf = utils.aio.Chan[rtc.AudioFrame]()
        self._audio_bstream = utils.audio.AudioByteStream(
            sample_rate, num_channels, samples_per_channel=sample_rate // 20, progressive=True
        )

        self._flush_task: asyncio.Task[None] | None = None
        self._interrupted_event = asyncio.Event()
        self._forwarding_task: asyncio.Task[None] | None = None

        self._pushed_duration: float = 0.0

        self._playback_enabled = asyncio.Event()
        self._playback_enabled.set()
        self._first_frame_event = asyncio.Event()

    async def _publish_track(self) -> None:
        async with self._lock:
            track = rtc.LocalAudioTrack.create_audio_track(self._track_name, self._audio_source)
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
        self._forwarding_task = asyncio.create_task(self._forward_audio())
        await self._publish_track()

    async def aclose(self) -> None:
        if self._flush_task:
            await utils.aio.cancel_and_wait(self._flush_task)
        if self._forwarding_task:
            await utils.aio.cancel_and_wait(self._forwarding_task)

        await self._audio_source.aclose()

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await self._subscribed_fut

        await super().capture_frame(frame)

        if self._flush_task and not self._flush_task.done():
            logger.error("capture_frame called while flush is in progress")
            await self._flush_task

        for f in self._audio_bstream.push(frame.data):
            self._audio_buf.send_nowait(f)
            self._pushed_duration += f.duration

    def flush(self) -> None:
        super().flush()

        for f in self._audio_bstream.flush():
            self._audio_buf.send_nowait(f)
            self._pushed_duration += f.duration

        if not self._pushed_duration:
            return

        if self._flush_task and not self._flush_task.done():
            # shouldn't happen if only one active speech handle at a time
            logger.error("flush called while playback is in progress")
            self._flush_task.cancel()

        self._flush_task = asyncio.create_task(self._wait_for_playout())

    def clear_buffer(self) -> None:
        self._audio_bstream.clear()

        if not self._pushed_duration:
            return
        self._interrupted_event.set()

    def pause(self) -> None:
        super().pause()
        self._playback_enabled.clear()
        # self._audio_source.clear_queue()

    def resume(self) -> None:
        super().resume()
        self._playback_enabled.set()
        self._first_frame_event.clear()

    async def _wait_for_playout(self) -> None:
        wait_for_interruption = asyncio.create_task(self._interrupted_event.wait())

        async def _wait_buffered_audio() -> None:
            while not self._audio_buf.empty():
                if not self._playback_enabled.is_set():
                    await self._playback_enabled.wait()

                await self._audio_source.wait_for_playout()
                # avoid deadlock when clear_buffer called before capture_frame
                await asyncio.sleep(0)

        wait_for_playout = asyncio.create_task(_wait_buffered_audio())
        await asyncio.wait(
            [wait_for_playout, wait_for_interruption],
            return_when=asyncio.FIRST_COMPLETED,
        )

        interrupted = self._interrupted_event.is_set()
        pushed_duration = self._pushed_duration

        if interrupted:
            queued_duration = self._audio_source.queued_duration
            while not self._audio_buf.empty():
                queued_duration += self._audio_buf.recv_nowait().duration

            pushed_duration = max(pushed_duration - queued_duration, 0)
            self._audio_source.clear_queue()
            wait_for_playout.cancel()
        else:
            wait_for_interruption.cancel()

        self._pushed_duration = 0
        self._interrupted_event.clear()
        self._first_frame_event.clear()
        self.on_playback_finished(playback_position=pushed_duration, interrupted=interrupted)

    async def _forward_audio(self) -> None:
        async for frame in self._audio_buf:
            if not self._playback_enabled.is_set():
                self._audio_source.clear_queue()
                await self._playback_enabled.wait()
                # TODO(long): save the frames in the queue and play them later
                # TODO(long): ignore frames from previous syllable

            if self._interrupted_event.is_set() or self._pushed_duration == 0:
                if self._interrupted_event.is_set() and self._flush_task:
                    await self._flush_task

                # ignore frames if interrupted
                continue

            if not self._first_frame_event.is_set():
                self._first_frame_event.set()
                self.on_playback_started(created_at=time.time())
            await self._audio_source.capture_frame(frame)


class _ParticipantLegacyTranscriptionOutput:
    def __init__(
        self,
        room: rtc.Room,
        *,
        is_delta_stream: bool = True,
        participant: rtc.Participant | str | None = None,
    ):
        self._room, self._is_delta_stream = room, is_delta_stream
        self._track_id: str | None = None
        self._participant_identity: str | None = None

        # identity of the participant that on behalf of the current participant
        self._represented_by: str | None = None

        self._room.on("track_published", self._on_track_published)
        self._room.on("local_track_published", self._on_local_track_published)
        self._flush_task: asyncio.Task[None] | None = None
        self._closed = False

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

        # _pushed_text keeps the raw text (markup intact); publish the visible text only.
        # Stripping the whole accumulation each time avoids partial-tag edge cases; the
        # expression is dropped here — the deprecated rtc Transcription API has no
        # attribute channel (the stream-based output carries lk.expression instead).
        clean_text = strip_all_markup(self._pushed_text)
        await self._publish_transcription(self._current_id, clean_text, final=False)

    @utils.log_exceptions(logger=logger)
    def flush(self) -> None:
        if self._participant_identity is None or self._track_id is None or not self._capturing:
            return

        clean_text = strip_all_markup(self._pushed_text)
        self._flush_task = asyncio.create_task(
            self._publish_transcription(self._current_id, clean_text, final=True)
        )
        self._reset_state()

    async def aclose(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._room.off("track_published", self._on_track_published)
        self._room.off("local_track_published", self._on_local_track_published)
        if self._flush_task:
            await self._flush_task

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
            if self._room.isconnected():
                logger.warning("failed to publish agent transcription to room: %s", e)

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


class _ParticipantStreamTranscriptionOutput:
    def __init__(
        self,
        room: rtc.Room,
        *,
        is_delta_stream: bool = True,
        participant: rtc.Participant | str | None = None,
        attributes: dict[str, str] | None = None,
        json_format: bool = False,
    ):
        self._room, self._is_delta_stream = room, is_delta_stream
        self._track_id: str | None = None
        self._participant_identity: str | None = None
        self._additional_attributes = attributes or {}
        self._writer: rtc.TextStreamWriter | None = None
        self._json_format = json_format

        self._room.on("track_published", self._on_track_published)
        self._room.on("local_track_published", self._on_local_track_published)
        self._flush_atask: asyncio.Task[None] | None = None
        self._closed = False

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
        # per-turn markup stripping: delta streams strip incrementally (buffering a tag
        # split across chunks); non-delta streams re-strip the full text each time and keep
        # the latest tags here for the expression attribute (see TranscriptMarkupStripper)
        self._stripper = TranscriptMarkupStripper()
        self._segment_tags: list[ExpressiveTag] = []
        # delta-stream expression bookkeeping (drives mid-turn segment rotation)
        self._expr_consumed = 0
        self._writer_expression_sent = False
        self._writer_has_text = False

    def _encode(self, clean_text: str, timing_src: str | None = None) -> str:
        """Wrap visible text for the wire (JSON TimedString when json_format, else raw)."""
        if not self._json_format:
            return clean_text

        ts_pb = agent_pb.TimedString(text=clean_text)
        if isinstance(timing_src, TimedString):
            if utils.is_given(timing_src.start_time):
                ts_pb.start_time = timing_src.start_time
            if utils.is_given(timing_src.end_time):
                ts_pb.end_time = timing_src.end_time
            if utils.is_given(timing_src.confidence):
                ts_pb.confidence = timing_src.confidence
            if utils.is_given(timing_src.start_time_offset):
                ts_pb.start_time_offset = timing_src.start_time_offset
        return json.dumps(MessageToDict(ts_pb, preserving_proto_field_name=True)) + "\n"

    async def _create_text_writer(
        self,
        attributes: dict[str, str] | None = None,
        extra_attributes: dict[str, str] | None = None,
    ) -> rtc.TextStreamWriter:
        assert self._participant_identity is not None, "participant_identity is not set"

        if not attributes:
            attributes = {
                ATTRIBUTE_TRANSCRIPTION_FINAL: "false",
            }
            if self._track_id:
                attributes[ATTRIBUTE_TRANSCRIPTION_TRACK_ID] = self._track_id
        attributes[ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID] = self._current_id

        for key, val in (extra_attributes or {}).items():
            attributes.setdefault(key, val)

        for key, val in self._additional_attributes.items():
            if key not in attributes:
                attributes[key] = val

        return await self._room.local_participant.stream_text(
            topic=TOPIC_TRANSCRIPTION,
            sender_identity=self._participant_identity,
            attributes=attributes,
        )

    def _pending_expressions(self) -> list[ExpressiveTag]:
        """Expression tags stripped so far but not yet attached to a wire segment."""
        tags = [t for t in self._stripper.tags if t["type"] in ("expression", "emotion")]
        return tags[self._expr_consumed :]

    def _consume_expressions(self) -> None:
        self._expr_consumed = sum(
            1 for t in self._stripper.tags if t["type"] in ("expression", "emotion")
        )

    async def _rotate_writer(self, pending: list[ExpressiveTag]) -> None:
        """Finalize the current wire segment and open a new one led by the pending expression."""
        assert self._writer is not None
        attributes = {ATTRIBUTE_TRANSCRIPTION_FINAL: "true"}
        if self._track_id:
            attributes[ATTRIBUTE_TRANSCRIPTION_TRACK_ID] = self._track_id
        await self._writer.aclose(attributes=attributes)

        self._current_id = utils.shortuuid("SG_")
        self._writer = await self._create_text_writer(
            extra_attributes=expression_attribute(pending)
        )
        self._writer_expression_sent = True
        self._writer_has_text = False

    async def _capture_delta(self, piece: str, timing_src: str) -> None:
        clean_text = self._stripper.push(piece)
        if not self._room.isconnected():
            return

        pending = self._pending_expressions()
        if self._writer is None:
            if not clean_text and not pending:
                return
            # open the segment as soon as its leading expression (or first text) is
            # known, so lk.expression rides the opening header
            self._writer = await self._create_text_writer(
                extra_attributes=expression_attribute(pending)
            )
            self._writer_expression_sent = bool(pending)
            self._consume_expressions()
        elif pending and clean_text:
            if self._writer_has_text or not self._writer_expression_sent:
                # a new expression starts a new statement: rotate so the new segment's
                # opening header carries it
                await self._rotate_writer(pending)
            # else: markers stacked before the first word coalesce into the header
            # already sent (first tag wins)
            self._consume_expressions()

        if clean_text:
            payload = self._encode(clean_text, timing_src)
            self._latest_text = payload
            await self._writer.write(payload)
            if clean_text.strip():
                self._writer_has_text = True

    @utils.log_exceptions(logger=logger)
    async def capture_text(self, text: str) -> None:
        if self._participant_identity is None:
            return

        if self._flush_atask and not self._flush_atask.done():
            await self._flush_atask

        if not self._capturing:
            self._reset_state()
            self._capturing = True

        # the raw text (expressive markup intact) arrives here; publish only the visible
        # text. A chunk that strips to nothing (a partial tag still buffering) is held
        # back so the transcript cadence isn't disturbed.
        try:
            if self._is_delta_stream:
                # split at expression markers so text on each side of a marker lands
                # in the right wire segment
                for piece in _EXPR_MARKER_SPLIT_RE.split(text):
                    if piece:
                        await self._capture_delta(piece, text)
            else:  # always create a new writer
                clean_text, self._segment_tags = split_all_markup(text)
                if not clean_text or not self._room.isconnected():
                    return
                payload = self._encode(clean_text, text)
                self._latest_text = payload
                tmp_writer = await self._create_text_writer(
                    extra_attributes=expression_attribute(self._segment_tags)
                )
                await tmp_writer.write(payload)
                await tmp_writer.aclose()
        except Exception as e:
            logger.warning("failed to publish agent transcription to room: %s", e)

    async def _flush_task(
        self,
        writer: rtc.TextStreamWriter | None,
        extra_attributes: dict[str, str] | None = None,
        pending_text: str = "",
    ) -> None:
        attributes = {ATTRIBUTE_TRANSCRIPTION_FINAL: "true"}
        if self._track_id:
            attributes[ATTRIBUTE_TRANSCRIPTION_TRACK_ID] = self._track_id
        for key, val in (extra_attributes or {}).items():
            attributes.setdefault(key, val)

        try:
            if self._room.isconnected():
                if self._is_delta_stream:
                    if writer:
                        if pending_text:  # visible text left in the strip buffer
                            await writer.write(pending_text)
                        await writer.aclose(attributes=attributes)
                else:
                    tmp_writer = await self._create_text_writer(attributes=attributes)
                    await tmp_writer.write(self._latest_text)
                    await tmp_writer.aclose()
        except Exception as e:
            logger.warning("failed to publish agent transcription to room: %s", e)

    def flush(self) -> None:
        # only emit on a segment that captured text (keeps lk.transcription cadence intact).
        # The closing header carries the expression only as a fallback when the stream
        # never got one — e.g. the tag only completed in the flush remainder.
        if self._participant_identity is None or not self._capturing:
            return

        self._capturing = False
        curr_writer = self._writer
        self._writer = None

        extra_attributes: dict[str, str] | None = None
        if self._is_delta_stream:
            remaining = self._stripper.flush()
            if not self._writer_expression_sent:
                extra_attributes = expression_attribute(self._pending_expressions())
        else:
            remaining = ""
            extra_attributes = expression_attribute(self._segment_tags)

        pending_text = self._encode(remaining) if remaining else ""
        self._flush_atask = asyncio.create_task(
            self._flush_task(curr_writer, extra_attributes, pending_text)
        )

    async def aclose(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._room.off("track_published", self._on_track_published)
        self._room.off("local_track_published", self._on_local_track_published)

        if self._flush_atask:
            await self._flush_atask

        if self._writer:
            writer = self._writer
            self._writer = None
            await writer.aclose()

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
class _ParticipantTranscriptionOutput(io.TextOutput):
    def __init__(
        self,
        *,
        room: rtc.Room,
        is_delta_stream: bool = True,
        participant: rtc.Participant | str | None = None,
        next_in_chain: io.TextOutput | None = None,
        json_format: bool = False,
    ) -> None:
        super().__init__(label="RoomIO", next_in_chain=next_in_chain)

        self.__outputs: list[
            _ParticipantLegacyTranscriptionOutput | _ParticipantStreamTranscriptionOutput
        ] = [
            _ParticipantLegacyTranscriptionOutput(
                room=room,
                is_delta_stream=is_delta_stream,
                participant=participant,
            ),
            _ParticipantStreamTranscriptionOutput(
                room=room,
                is_delta_stream=is_delta_stream,
                participant=participant,
                json_format=json_format,
            ),
        ]
        self.__closed = False

    def set_participant(self, participant: rtc.Participant | str | None) -> None:
        for source in self.__outputs:
            source.set_participant(participant)

    async def capture_text(self, text: str) -> None:
        await asyncio.gather(*[sink.capture_text(text) for sink in self.__outputs])

        if self.next_in_chain:
            await self.next_in_chain.capture_text(text)

    def flush(self) -> None:
        for source in self.__outputs:
            source.flush()

        if self.next_in_chain:
            self.next_in_chain.flush()

    async def aclose(self) -> None:
        if self.__closed:
            return

        self.__closed = True
        await asyncio.gather(*[source.aclose() for source in self.__outputs])
