from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Literal

from livekit import rtc

from ...language import LanguageCode
from ...log import logger
from ...stt import STT, RecognizeStream, SpeechData, SpeechEventType
from ...types import (
    ATTRIBUTE_TRANSCRIPTION_FINAL,
    ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID,
    ATTRIBUTE_TRANSCRIPTION_TRACK_ID,
    TOPIC_TRANSCRIPTION,
    TimedString,
)
from ...utils import aio, log_exceptions, shortuuid
from ...utils.audio_gate import AdaptiveNoiseGate, AudioGate
from ..room_io._output import _legacy_transcription_needed
from .activity import ActivityTimeline
from .attribution import AttributionMethod, _Attributor, _Run
from .mixer import _Mixer, _TrackInput

_SAMPLE_RATE = 16000
_BLOCK_DURATION = 0.02

# speaker activity kept for attribution; longer than any STT takes to return a transcript
_RETENTION = 30.0

# the gate's floor never rises above -40 dBFS: no microphone's room tone is that loud once noise
# suppression has run, and a participant who joins mid-sentence still opens their gate
_MIC_MAX_FLOOR = 0.01

# a second participant at least this loud, relative to the loudest, marks a segment overlapping
_OVERLAP_RATIO = 0.5

# how long the mix waits past a block's due time for a participant whose audio is late
_JITTER = 0.06

# a participant can run this far ahead of the mix before its oldest audio is dropped
_MAX_BUFFER = 1.0

# the mix falls this far behind real time (a stalled loop) before it stops trying to catch up
_MAX_LAG = 1.0

# an STT that keeps failing is reopened at most this often
_MAX_REOPEN_DELAY = 30.0

# a participant's audio queued past this for all of _BACKLOG_WINDOW without shrinking (it came
# in late and never drained, unlike a long frame or a burst) has its oldest dropped
_MAX_BACKLOG = 0.1
_BACKLOG_WINDOW = 1.0

# on close, how long the STT gets to return what it heard in the last of the audio
_CLOSE_TIMEOUT = 2.0

# agents are left out, so an agent's own speech is never transcribed
_TRANSCRIBED_KINDS = {
    rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
    rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
    rtc.ParticipantKind.PARTICIPANT_KIND_INGRESS,
}


@dataclass
class TranscriptSegment:
    """A stretch of one participant's speech.

    An STT utterance is split into one segment per speaker turn inside it. Interim segments are
    replaced by later ones with the same ``id``; a final segment closes it. When a later
    transcript splits the utterance differently, a segment it no longer contains is closed as
    final with empty ``text``.
    """

    id: str
    participant_identity: str | None
    """Who spoke; None when nobody was audible for it."""
    track_sid: str | None
    text: str
    final: bool
    start_time: float
    """Seconds into the mixed stream."""
    end_time: float
    language: LanguageCode
    words: list[TimedString] | None
    confidence: float
    overlap: bool
    """Another participant was audibly speaking at the same time."""
    attribution: AttributionMethod


EventTypes = Literal["transcript_segment"]


@dataclass
class _StreamState:
    """One STT stream's share of the mix, which its results are attributed within.

    A stream that ended still returns results while the next one hears newer audio, so each
    keeps its own bounds and utterance.
    """

    start: float
    attributor: _Attributor
    """Its own, since a new connection numbers its diarization labels afresh."""
    end: float | None = None
    """Where the stream's audio ended on the mix's clock; None while it is open."""
    last_final_end: float = 0.0
    utterance_id: str = field(default_factory=lambda: shortuuid("SG_"))
    segments: dict[str, TranscriptSegment] = field(default_factory=dict)
    # a segment's id belongs to one speaker: a run that changes hands gets a new one
    segment_ids: dict[tuple[int, str | None], str] = field(default_factory=dict)


class TranscriptionSession(rtc.EventEmitter[EventTypes]):
    """Transcribes every participant in a room through a single STT stream.

    Participants' audio is mixed into one stream, and each word the STT returns is attributed to
    whoever was audible when it was spoken, using a per-participant energy gate. Any streaming
    STT works; word timings and diarization make the attribution finer when the STT has them.

    Emits ``transcript_segment`` with a :class:`TranscriptSegment`.
    """

    def __init__(
        self,
        *,
        stt: STT,
        gate: Callable[[], AudioGate] | None = None,
        participant_identities: Iterable[str] | None = None,
        publish_transcriptions: bool = True,
    ) -> None:
        """Create a transcription session.

        Args:
            stt: A streaming STT. Wrap a non-streaming one with ``stt.StreamAdapter``.
            gate: Builds the gate that decides when each participant is speaking. Defaults to
                :class:`~livekit.agents.utils.audio_gate.AdaptiveNoiseGate` with its floor
                capped for microphones.
            participant_identities: Only transcribe these participants, when given. Agents are
                never transcribed.
            publish_transcriptions: Publish segments to the room on ``lk.transcription``,
                attributed to the participant who spoke them.

        Raises:
            ValueError: If the STT does not stream.
        """
        super().__init__()
        if not stt.capabilities.streaming:
            raise ValueError(
                "TranscriptionSession needs a streaming STT, wrap it with stt.StreamAdapter"
            )

        self._stt = stt
        self._gate_factory = gate or (lambda: AdaptiveNoiseGate(max_floor=_MIC_MAX_FLOOR))
        self._block_size = int(_SAMPLE_RATE * _BLOCK_DURATION)
        self._participant_identities = (
            set(participant_identities) if participant_identities is not None else None
        )
        self._publish = publish_transcriptions

        self._timeline = ActivityTimeline(block_duration=_BLOCK_DURATION, retention=_RETENTION)
        self._mixer = _Mixer(
            sample_rate=_SAMPLE_RATE,
            block_size=self._block_size,
            timeline=self._timeline,
            max_backlog=round(_MAX_BACKLOG / _BLOCK_DURATION),
            backlog_window=round(_BACKLOG_WINDOW / _BLOCK_DURATION),
        )

        self._room: rtc.Room | None = None
        self._stream: RecognizeStream | None = None
        self._stream_state: _StreamState | None = None
        self._data_ev = asyncio.Event()
        self._input_atasks: dict[str, asyncio.Task[None]] = {}
        self._mix_atask: asyncio.Task[None] | None = None
        self._recv_atasks: set[asyncio.Task[None]] = set()
        self._publish_ch: aio.Chan[TranscriptSegment] | None = None
        self._publish_atask: asyncio.Task[None] | None = None
        self._started = False
        self._closing = False
        self._stt_failures = 0
        self._reopen_at = 0.0

        self._history: list[TranscriptSegment] = []
        # kept past a participant leaving: their last words are attributed after they've gone
        self._track_sids: dict[str, str] = {}

    @property
    def history(self) -> list[TranscriptSegment]:
        """Every final segment so far, in order."""
        return list(self._history)

    @property
    def participants(self) -> list[str]:
        """Identities currently being transcribed."""
        return list(self._mixer.inputs)

    async def start(self, *, room: rtc.Room | None = None) -> None:
        """Start transcribing.

        Args:
            room: Transcribe this room's participants as they come and go. Without it, audio
                comes only from :meth:`add_input` and :meth:`push_audio`.
        """
        if self._started:
            raise RuntimeError("TranscriptionSession already started")
        self._started = True

        self._mix_atask = asyncio.create_task(self._mix_task())

        if room is None:
            return

        self._room = room
        if self._publish:
            self._publish_ch = aio.Chan[TranscriptSegment]()
            self._publish_atask = asyncio.create_task(self._publish_task(room, self._publish_ch))

        room.on("track_published", self._on_track_published)
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_unsubscribed", self._on_track_unsubscribed)
        room.on("participant_disconnected", self._on_participant_disconnected)
        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                if isinstance(publication.track, rtc.RemoteAudioTrack) and publication.subscribed:
                    self._on_track_subscribed(publication.track, publication, participant)
                else:
                    self._on_track_published(publication, participant)

    def add_input(
        self,
        identity: str,
        frames: AsyncIterable[rtc.AudioFrame],
        *,
        track_sid: str | None = None,
    ) -> None:
        """Transcribe a participant from a stream of frames, replacing any input it had.

        The participant leaves the mix once the stream ends and its audio has played out.
        """
        self.remove_input(identity)
        track_input = self._new_input(identity, track_sid)
        task = asyncio.create_task(self._forward_input(track_input, frames))
        self._input_atasks[identity] = task

        def _on_done(_: asyncio.Task[None]) -> None:
            if self._input_atasks.get(identity) is task:
                del self._input_atasks[identity]

        task.add_done_callback(_on_done)

    def push_audio(
        self, identity: str, frame: rtc.AudioFrame, *, track_sid: str | None = None
    ) -> None:
        """Push one frame of a participant's audio, adding the participant if it is new."""
        if self._closing:
            return
        track_input = self._mixer.inputs.get(identity)
        if track_input is None or track_input.ended:
            track_input = self._new_input(identity, track_sid)
        track_input.push(frame)
        self._data_ev.set()

    def remove_input(self, identity: str) -> None:
        """Stop transcribing a participant, discarding audio of theirs not yet mixed."""
        if task := self._input_atasks.pop(identity, None):
            task.cancel()
        self._mixer.remove(identity)

    async def aclose(self) -> None:
        """Stop transcribing, after the STT has returned what it heard in the audio so far."""
        if self._closing:
            return
        self._closing = True

        if self._room is not None:
            self._room.off("track_published", self._on_track_published)
            self._room.off("track_subscribed", self._on_track_subscribed)
            self._room.off("track_unsubscribed", self._on_track_unsubscribed)
            self._room.off("participant_disconnected", self._on_participant_disconnected)

        await aio.cancel_and_wait(*self._input_atasks.values())
        self._input_atasks.clear()

        if self._mix_atask is not None:
            self._data_ev.set()
            await self._mix_atask
        self._end_stream()
        if self._recv_atasks:
            _, pending = await asyncio.wait(self._recv_atasks, timeout=_CLOSE_TIMEOUT)
            if pending:
                logger.debug("STT did not finish before the transcription session closed")
            await aio.cancel_and_wait(*pending)

        if self._publish_ch is not None and self._publish_atask is not None:
            self._publish_ch.close()
            await self._publish_atask

    def _new_input(self, identity: str, track_sid: str | None) -> _TrackInput:
        track_input = _TrackInput(
            identity,
            track_sid=track_sid,
            gate=self._gate_factory(),
            sample_rate=_SAMPLE_RATE,
            block_size=self._block_size,
            max_blocks=round(_MAX_BUFFER / _BLOCK_DURATION),
        )
        self._mixer.add(track_input)
        if track_sid is not None:
            self._track_sids[identity] = track_sid
        return track_input

    @log_exceptions(logger=logger)
    async def _forward_input(
        self, track_input: _TrackInput, frames: AsyncIterable[rtc.AudioFrame]
    ) -> None:
        try:
            async for frame in frames:
                track_input.push(frame)
                self._data_ev.set()
        finally:
            track_input.ended = True
            self._data_ev.set()

    def _open_stream(self, *, rewind: float = 0.0) -> RecognizeStream:
        stream = self._stt.stream()
        # the stream dates audio by the wall clock since this anchor, first frame and reconnects
        # alike; the mix never runs ahead of real time, so that stays on the mix's clock. The
        # anchor is a block on, since the stream dates a frame from its push minus its duration
        start = self._timeline.duration - rewind
        stream.start_time_offset = start + _BLOCK_DURATION
        state = _StreamState(
            start=start,
            attributor=_Attributor(self._timeline, overlap_ratio=_OVERLAP_RATIO),
            last_final_end=start,
        )
        task = asyncio.create_task(self._recv_task(stream, state))
        self._recv_atasks.add(task)
        task.add_done_callback(self._recv_atasks.discard)
        self._stream, self._stream_state = stream, state
        return stream

    def _end_stream(self) -> None:
        """End the open stream's input; its results still arrive, then it closes."""
        if self._stream is not None:
            with contextlib.suppress(RuntimeError):
                self._stream.end_input()
            self._stream = None
        if self._stream_state is not None:
            self._stream_state.end = self._timeline.duration
            self._stream_state = None

    @log_exceptions(logger=logger)
    async def _mix_task(self) -> None:
        # block k is mixed no earlier than anchor + k * block, so the mix never runs ahead of real
        # time, and once every participant has sent it. One who hasn't by anchor + (k + 1) *
        # block + jitter is left out of it
        anchor: float | None = None
        while True:
            self._data_ev.clear()
            self._mixer.prune()
            if self._closing and not self._mixer.has_pending():
                return

            if not self._mixer.inputs:
                # nobody to transcribe: the mix's clock stops rather than bill silence. The
                # stream ends too, since one left idle reconnects on a clock that kept running
                anchor = None
                self._end_stream()
                await self._data_ev.wait()
                continue

            now = time.monotonic()
            if anchor is None:
                anchor = now - self._mixer.index * _BLOCK_DURATION

            if not self._closing and not self._mixer.all_ready():
                due = anchor + (self._mixer.index + 1) * _BLOCK_DURATION + _JITTER
                if due > now:
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(self._data_ev.wait(), due - now)
                    continue

            # buffered audio (a drain on close, a participant ahead of the others) still goes
            # out in real time: the stream dates a burst as if it were captured just now
            if (slot := anchor + self._mixer.index * _BLOCK_DURATION) > now:
                await asyncio.sleep(slot - now)
                continue

            # after a failure the STT is left alone a while; the mix's clock runs on meanwhile
            stream = self._stream
            if stream is None and now >= self._reopen_at:
                stream = self._open_stream()
            frame = self._mixer.mix()
            if stream is not None:
                try:
                    stream.push_frame(frame)
                except RuntimeError:
                    # the STT failed past its retries since the last block. A first failure
                    # reconnects at once: a new stream, dated a block back, takes this one
                    self._end_stream()
                    if not self._stt_failures:
                        self._open_stream(rewind=_BLOCK_DURATION).push_frame(frame)

            behind = time.monotonic() - (anchor + self._mixer.index * _BLOCK_DURATION)
            if behind > _MAX_LAG:
                anchor = time.monotonic() - self._mixer.index * _BLOCK_DURATION

    async def _recv_task(self, stream: RecognizeStream, state: _StreamState) -> None:
        try:
            async for ev in stream:
                if not ev.alternatives:
                    continue
                if ev.type == SpeechEventType.INTERIM_TRANSCRIPT:
                    self._on_transcript(ev.alternatives[0], state, final=False)
                elif ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                    self._on_transcript(ev.alternatives[0], state, final=True)
                else:
                    continue
                # only a transcript shows the STT works; a speech-start or usage event may not
                self._stt_failures = 0
        except Exception:
            # the first failure reconnects at once, since the stream already retried what it
            # could; a failure that keeps coming back (a bad key) is retried less and less often
            self._stt_failures += 1
            delay = (
                min(2.0 ** (self._stt_failures - 2), _MAX_REOPEN_DELAY)
                if self._stt_failures > 1
                else 0.0
            )
            self._reopen_at = time.monotonic() + delay
            logger.exception("transcription STT stream failed, reopening in %.0fs", delay)
        finally:
            if self._stream is stream:
                # it failed while the mix was feeding it; the next block opens a new one
                self._end_stream()
            # what it last showed and never finalized stands as said
            open_segments = [replace(segment, final=True) for segment in state.segments.values()]
            state.segments = {}
            for segment in open_segments:
                self._emit(segment)
            self._history.extend(open_segments)
            await stream.aclose()

    def _on_transcript(self, alt: SpeechData, state: _StreamState, *, final: bool) -> None:
        if not alt.text.strip() and not final:
            return

        # untimed results are placed within what their own stream heard
        end = state.end if state.end is not None else self._timeline.duration
        runs = state.attributor.attribute(alt, fallback=(state.last_final_end, end))

        segments = [
            self._segment(_segment_id(state, i, run.identity), run, alt, final)
            for i, run in enumerate(runs)
            if run.text
        ]
        current = {segment.id for segment in segments}
        # an interim split or attributed the transcript differently; close what it showed that
        # this doesn't, under the speaker who showed it
        stale = [_cleared(segment) for id, segment in state.segments.items() if id not in current]

        for segment in [*stale, *segments]:
            self._emit(segment)

        if final:
            self._history.extend(segments)
            state.last_final_end = max(run.end_time for run in runs) if runs else end
            state.utterance_id = shortuuid("SG_")
            state.segments = {}
            state.segment_ids = {}
        else:
            state.segments = {segment.id: segment for segment in segments}

    def _emit(self, segment: TranscriptSegment) -> None:
        self.emit("transcript_segment", segment)
        if self._publish_ch is not None and segment.participant_identity is not None:
            self._publish_ch.send_nowait(segment)

    def _segment(self, id: str, run: _Run, alt: SpeechData, final: bool) -> TranscriptSegment:
        track_sid: str | None = None
        if run.identity is not None:
            # the track the words were spoken on, which a microphone switch since then hasn't
            # changed; the latest one known when the timeline no longer covers them
            track_sid = self._timeline.track_sid(
                run.identity, run.start_time, run.end_time
            ) or self._track_sids.get(run.identity)
        return TranscriptSegment(
            id=id,
            participant_identity=run.identity,
            track_sid=track_sid,
            text=run.text,
            final=final,
            start_time=run.start_time,
            end_time=run.end_time,
            language=alt.language,
            words=run.words,
            confidence=alt.confidence,
            overlap=run.overlap,
            attribution=run.method,
        )

    @log_exceptions(logger=logger)
    async def _publish_task(self, room: rtc.Room, ch: aio.Chan[TranscriptSegment]) -> None:
        async for segment in ch:
            assert segment.participant_identity is not None
            attributes = {
                ATTRIBUTE_TRANSCRIPTION_FINAL: str(segment.final).lower(),
                ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID: segment.id,
            }
            if segment.track_sid:
                attributes[ATTRIBUTE_TRANSCRIPTION_TRACK_ID] = segment.track_sid
            try:
                if not room.isconnected():
                    continue
                writer = await room.local_participant.stream_text(
                    topic=TOPIC_TRANSCRIPTION,
                    sender_identity=segment.participant_identity,
                    attributes=attributes,
                )
                await writer.write(segment.text)
                await writer.aclose()

                # clients too old to read lk.transcription still get the deprecated packet
                if segment.track_sid and _legacy_transcription_needed(room):
                    await room.local_participant.publish_transcription(
                        rtc.Transcription(
                            participant_identity=segment.participant_identity,
                            track_sid=segment.track_sid,
                            segments=[
                                rtc.TranscriptionSegment(
                                    id=segment.id,
                                    text=segment.text,
                                    start_time=0,
                                    end_time=0,
                                    final=segment.final,
                                    language=str(segment.language),
                                )
                            ],
                        )
                    )
            except Exception as e:
                logger.warning("failed to publish transcription to room: %s", e)

    def _accepts(self, participant: rtc.RemoteParticipant) -> bool:
        if participant.kind not in _TRANSCRIBED_KINDS:
            return False
        return (
            self._participant_identities is None
            or participant.identity in self._participant_identities
        )

    def _on_track_published(
        self, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if (
            publication.kind == rtc.TrackKind.KIND_AUDIO
            and publication.source == rtc.TrackSource.SOURCE_MICROPHONE
            and not publication.subscribed
            and self._accepts(participant)
        ):
            publication.set_subscribed(True)

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if (
            track.kind != rtc.TrackKind.KIND_AUDIO
            or publication.source != rtc.TrackSource.SOURCE_MICROPHONE
            or not self._accepts(participant)
            or self._closing
        ):
            return

        stream = rtc.AudioStream.from_track(track=track, sample_rate=_SAMPLE_RATE, num_channels=1)
        self.add_input(participant.identity, _track_frames(stream), track_sid=publication.sid)

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        track_input = self._mixer.inputs.get(participant.identity)
        if track_input is not None and track_input.track_sid == publication.sid:
            self._end_input(participant.identity)

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        self._end_input(participant.identity)

    def _end_input(self, identity: str) -> None:
        """Stop reading a participant's track; what they already sent is still mixed."""
        if task := self._input_atasks.pop(identity, None):
            task.cancel()


def _segment_id(state: _StreamState, index: int, identity: str | None) -> str:
    key = (index, identity)
    if key not in state.segment_ids:
        state.segment_ids[key] = f"{state.utterance_id}_{len(state.segment_ids)}"
    return state.segment_ids[key]


def _cleared(segment: TranscriptSegment) -> TranscriptSegment:
    return TranscriptSegment(
        id=segment.id,
        participant_identity=segment.participant_identity,
        track_sid=segment.track_sid,
        text="",
        final=True,
        start_time=segment.start_time,
        end_time=segment.end_time,
        language=segment.language,
        words=None,
        confidence=0.0,
        overlap=False,
        attribution=segment.attribution,
    )


async def _track_frames(stream: rtc.AudioStream) -> AsyncIterator[rtc.AudioFrame]:
    try:
        async for ev in stream:
            yield ev.frame
    finally:
        await stream.aclose()
