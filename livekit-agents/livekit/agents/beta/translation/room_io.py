from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from livekit import rtc

from ...job import DEFAULT_PARTICIPANT_KINDS
from ...log import logger
from ...telemetry import trace_types
from ...types import (
    ATTRIBUTE_PUBLISH_ON_BEHALF,
    ATTRIBUTE_TRANSCRIPTION_LANGUAGE,
    NOT_GIVEN,
    NotGivenOr,
)
from ...utils import aio, is_given, log_exceptions
from ...voice.events import CloseReason
from ...voice.io import AudioInput, AudioOutput, TextOutput
from ...voice.room_io import (
    AudioInputOptions,
    AudioOutputOptions,
    TextOutputOptions,
    _ParticipantAudioOutput,
    _ParticipantTranscriptionOutput,
)
from ...voice.room_io._input import _ParticipantAudioInputStream
from ...voice.room_io._pre_connect_audio import PreConnectAudioHandler
from ...voice.room_io.types import DEFAULT_CLOSE_ON_DISCONNECT_REASONS
from ...voice.transcription.synchronizer import TranscriptSynchronizer

if TYPE_CHECKING:
    from .translation_session import TranslationSession


def _translated_track_name(source_track_sid: str, language: str) -> str:
    """The name the translation of a track is published under, unless one is given.

    A track carries no attributes, only a name, so the name is what tells a listener whose
    speech a translated track carries and in which language.
    """
    return f"{source_track_sid}-{language}"


class _LinkedTrack(NamedTuple):
    publication: rtc.RemoteTrackPublication
    participant: rtc.RemoteParticipant


@dataclass
class TranslationRoomOptions:
    """How a :class:`~livekit.agents.beta.translation.TranslationSession` reaches a room.

    The subset of ``RoomOptions`` a translator can use: it hears one participant and speaks
    its translation back into the room, and there is no text input to answer.
    """

    audio_input: AudioInputOptions | bool = True
    """The speech to translate. Pass ``False`` to feed the session audio yourself."""
    audio_output: AudioOutputOptions | bool = True
    """The track the translation is spoken on. Pass ``False`` to take the audio yourself.

    Unless ``track_name`` is given, it is published as ``{source_track_sid}-{language}``,
    and republished under the new name when the source track or the language changes."""
    text_output: TextOutputOptions | bool = True
    """Transcripts published to the room: what was heard, and what was spoken back.

    What was heard is the speaker's, on their track; the translation is the agent's, on the
    translated track it transcribes, as a client expects of an agent's speech. Both carry
    ``lk.transcription_language`` where it is known, and the translated track's name links
    the translation back to the speaker's track."""
    track: NotGivenOr[str] = NOT_GIVEN
    """The sid of the audio track to translate. If not given, the first eligible one
    published. A track is the unit here rather than a participant: a participant can publish
    more than one, and only a track can be transcribed and attributed on the wire."""
    participant_identity: NotGivenOr[str] = NOT_GIVEN
    """Narrows the automatic search to one participant's tracks. Ignored when ``track`` names
    a publication outright."""
    participant_kinds: NotGivenOr[list[rtc.ParticipantKind.ValueType]] = NOT_GIVEN
    """Participant kinds accepted when searching automatically."""
    close_on_disconnect: bool = True
    """Close the session when the linked track's publisher leaves of their own accord."""

    def get_audio_input_options(self) -> AudioInputOptions | None:
        if isinstance(self.audio_input, AudioInputOptions):
            return self.audio_input
        return AudioInputOptions() if self.audio_input else None

    def get_audio_output_options(self) -> AudioOutputOptions | None:
        if isinstance(self.audio_output, AudioOutputOptions):
            return self.audio_output
        return AudioOutputOptions() if self.audio_output else None

    def get_text_output_options(self) -> TextOutputOptions | None:
        if isinstance(self.text_output, TextOutputOptions):
            return self.text_output
        return TextOutputOptions() if self.text_output else None


class TranslationRoomIO:
    """Wires a :class:`~livekit.agents.beta.translation.TranslationSession` to a room.

    Its own wiring rather than ``RoomIO``: that one serves an ``AgentSession``, reaching for
    turn timers, chat input and an away timer a translator has none of. What is left is the
    part a translator does use — link a track, take its audio, publish the translated track,
    and publish both transcripts.

    What it links is a *track*, not a participant: the transcript a translator publishes is
    attributed to a track on the wire, a participant may publish more than one, and the
    publisher's identity is read off the track rather than the other way round.
    """

    def __init__(
        self,
        session: TranslationSession,
        room: rtc.Room,
        *,
        options: NotGivenOr[TranslationRoomOptions] = NOT_GIVEN,
    ) -> None:
        self._session, self._room = session, room
        self._options = options if is_given(options) else TranslationRoomOptions()
        self._track_sid = self._options.track if is_given(self._options.track) else None
        self._participant_identity = (
            self._options.participant_identity
            if is_given(self._options.participant_identity)
            else None
        )

        self._audio_input: _ParticipantAudioInputStream | None = None
        self._audio_output: _ParticipantAudioOutput | None = None
        self._input_tr_output: _ParticipantTranscriptionOutput | None = None
        self._output_tr_output: _ParticipantTranscriptionOutput | None = None
        self._tr_synchronizer: TranscriptSynchronizer | None = None
        self._pre_connect_audio_handler: PreConnectAudioHandler | None = None

        self._track_available_fut = asyncio.Future[_LinkedTrack]()
        self._room_connected_fut = asyncio.Future[None]()
        self._ready_fut = asyncio.Future[None]()
        self._init_atask: asyncio.Task[None] | None = None
        self._input_transcript_ch: aio.Chan[tuple[str, bool, str | None]] | None = None
        self._input_transcript_atask: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        # the track whose speech is being translated, once linked
        self._source_track_sid: str | None = None
        # the language the translation is labelled with; the session moves it once the model
        # is actually speaking a new one, and it lands once the old one has played out
        self._output_language: str | None = None
        self._requested_output_language: str | None = None
        self._relabel_atask: asyncio.Task[None] | None = None
        # a name given outright is kept; otherwise it follows the source track and language
        audio_output_options = self._options.get_audio_output_options()
        self._fixed_track_name = audio_output_options is not None and is_given(
            audio_output_options.track_name
        )

    async def start(self) -> None:
        audio_input_options = self._options.get_audio_input_options()
        if audio_input_options:
            if audio_input_options.pre_connect_audio:
                self._pre_connect_audio_handler = PreConnectAudioHandler(
                    room=self._room, timeout=audio_input_options.pre_connect_audio_timeout
                )
                self._pre_connect_audio_handler.register()

            self._audio_input = _ParticipantAudioInputStream(
                self._room,
                track_source=[
                    rtc.TrackSource.SOURCE_MICROPHONE,
                    rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO,
                ],
                sample_rate=audio_input_options.sample_rate,
                num_channels=audio_input_options.num_channels,
                frame_size_ms=audio_input_options.frame_size_ms,
                noise_cancellation=audio_input_options.noise_cancellation,
                auto_gain_control=(
                    audio_input_options.auto_gain_control
                    if is_given(audio_input_options.auto_gain_control)
                    else (
                        audio_input_options.noise_cancellation is None
                        or callable(audio_input_options.noise_cancellation)
                    )
                ),
                pre_connect_audio_handler=self._pre_connect_audio_handler,
            )

        audio_output_options = self._options.get_audio_output_options()
        if audio_output_options:
            self._audio_output = _ParticipantAudioOutput(
                self._room,
                sample_rate=audio_output_options.sample_rate,
                num_channels=audio_output_options.num_channels,
                track_publish_options=audio_output_options.track_publish_options,
                track_name=(
                    audio_output_options.track_name
                    if is_given(audio_output_options.track_name)
                    else "roomio_audio"
                ),
            )

        text_output_options = self._options.get_text_output_options()
        if text_output_options:
            # _input_tr_output is built in _init_task, once there is a track to stamp it with
            self._input_transcript_ch = aio.Chan[tuple[str, bool, str | None]]()
            self._input_transcript_atask = asyncio.create_task(
                self._forward_input_transcript(self._input_transcript_ch)
            )

            self._output_tr_output = _ParticipantTranscriptionOutput(
                room=self._room,
                is_delta_stream=True,
                participant=None,
                next_in_chain=text_output_options.next_in_chain,
                json_format=text_output_options.json_format,
            )

            if text_output_options.sync_transcription is not False and (
                audio_output := self._audio_output or self._session.output.audio
            ):
                self._tr_synchronizer = TranscriptSynchronizer(
                    next_in_chain_audio=audio_output,
                    next_in_chain_text=self._output_tr_output,
                    speed=text_output_options.transcription_speed_factor,
                )

        self._room.on("track_published", self._on_track_published)
        self._room.on("participant_connected", self._on_participant_connected)
        self._room.on("participant_disconnected", self._on_participant_disconnected)
        self._room.on("connection_state_changed", self._on_connection_state_changed)
        if self._room.isconnected():
            self._on_connection_state_changed(rtc.ConnectionState.CONN_CONNECTED)

        self._init_atask = asyncio.create_task(self._init_task())

        if self.audio_input:
            self._session.input.audio = self.audio_input
        if self.audio_output:
            self._session.output.audio = self.audio_output
        if self.transcription_output:
            self._session.output.transcription = self.transcription_output

        self._session.on("user_input_transcribed", self._on_input_transcribed)

    async def aclose(self) -> None:
        self._room.off("track_published", self._on_track_published)
        self._room.off("participant_connected", self._on_participant_connected)
        self._room.off("participant_disconnected", self._on_participant_disconnected)
        self._room.off("connection_state_changed", self._on_connection_state_changed)
        self._session.off("user_input_transcribed", self._on_input_transcribed)

        if self._init_atask:
            await aio.cancel_and_wait(self._init_atask)

        if self._input_transcript_ch:
            self._input_transcript_ch.close()
        if self._input_transcript_atask:
            await aio.cancel_and_wait(self._input_transcript_atask)

        if self._pre_connect_audio_handler:
            await self._pre_connect_audio_handler.aclose()

        if self._audio_input:
            await self._audio_input.aclose()

        if self._tr_synchronizer:
            await self._tr_synchronizer.aclose()

        if self._input_tr_output:
            await self._input_tr_output.aclose()
        if self._output_tr_output:
            await self._output_tr_output.aclose()

        if self._audio_output:
            await self._audio_output.aclose()

        await aio.cancel_and_wait(*self._tasks)
        self._tasks.clear()

    @property
    def room(self) -> rtc.Room:
        return self._room

    @property
    def audio_input(self) -> AudioInput | None:
        return self._audio_input

    @property
    def audio_output(self) -> AudioOutput | None:
        if self._tr_synchronizer:
            return self._tr_synchronizer.audio_output

        return self._audio_output

    @property
    def transcription_output(self) -> TextOutput | None:
        if self._tr_synchronizer:
            return self._tr_synchronizer.text_output

        return self._output_tr_output

    @property
    def linked_track(self) -> _LinkedTrack | None:
        """The track being translated, with the participant publishing it."""
        if not self._track_available_fut.done():
            return None

        return self._track_available_fut.result()

    @property
    def linked_participant(self) -> rtc.RemoteParticipant | None:
        """The participant publishing the linked track."""
        linked = self.linked_track
        return linked.participant if linked else None

    @property
    def subscribed_fut(self) -> asyncio.Future[None] | None:
        if self._audio_output:
            return self._audio_output.subscribed
        return None

    async def wait_for_ready(self) -> None:
        """Wait until the source track is linked and the translated track is published."""
        await self._ready_fut

    def set_track(self, track_sid: str | None) -> None:
        """Translate another track.

        The publisher is read off the track, so nothing else has to be supplied. A track that
        has not been published yet is accepted: it is linked once it appears.

        Raises:
            ValueError: If ``track_sid`` names a publication that is not audio.
        """
        if track_sid is None:
            self.unset_track()
            return

        linked = self.linked_track
        if linked is not None and linked.publication.sid == track_sid:
            # already on it, but pin it all the same: unpinned, a later search is free to
            # settle on another of the publisher's microphones
            self._track_sid = track_sid
            if self._audio_input:
                self._audio_input.set_track_sid(track_sid)
            return

        found: _LinkedTrack | None = None
        for participant in self._room.remote_participants.values():
            if (publication := participant.track_publications.get(track_sid)) is not None:
                if publication.kind != rtc.TrackKind.KIND_AUDIO:
                    # the model is handed audio; linking a camera would only close what is
                    # playing and leave nothing in its place
                    raise ValueError(
                        f"track {track_sid} is {rtc.TrackKind.Name(publication.kind)}, not audio"
                    )

                found = _LinkedTrack(publication, participant)
                break

        if self._track_available_fut.done():
            self._track_available_fut = asyncio.Future[_LinkedTrack]()

        self._track_sid = track_sid
        if found is not None:
            self._track_available_fut.set_result(found)
            self._apply_link(found)
        elif self._audio_input:
            # pinned now, so whatever the publisher is currently sending is dropped until
            # the named track appears and _on_track_published links it
            self._audio_input.set_track_sid(track_sid)

    def _apply_link(self, linked: _LinkedTrack) -> None:
        """Point the input, both transcripts and the translated track at a linked track."""
        self._participant_identity = linked.participant.identity
        track_sid = linked.publication.sid
        self._source_track_sid = track_sid

        if self._audio_input:
            # pinned before the participant, so set_participant's scan of existing
            # publications cannot latch a different audio track of theirs first
            self._audio_input.set_track_sid(track_sid)
            self._audio_input.set_participant(self._participant_identity)
        if self._input_tr_output:
            self._input_tr_output.set_participant(self._participant_identity)
            # the transcript follows the track, or later speech keeps the old stamp
            self._input_tr_output.set_track(track_sid)
        if self._output_tr_output:
            # the translation stays the agent's, on its own track: clients file a transcript
            # under its sender, so sent as the speaker it would read as their own words
            self._output_tr_output.set_attributes(self._output_attributes())

        if self._init_atask is not None and self._init_atask.done():
            # startup names the track itself, ahead of publishing it
            self._rename_output_track_soon()

        logger.info(
            "TranslationRoomIO linked to track",
            extra={
                trace_types.ATTR_TRACK_SID: track_sid,
                trace_types.ATTR_PARTICIPANT_IDENTITY: self._participant_identity,
                trace_types.ATTR_ROOM_NAME: self._room.name,
            },
        )

    def unset_track(self) -> None:
        """Stop translating any track, leaving the session with nothing to hear."""
        self._track_sid = None
        self._participant_identity = None
        self._source_track_sid = None
        if self._track_available_fut.done():
            # a pending one is kept: startup may still be waiting on it, and would wait on a
            # replaced one forever, never publishing the translation
            self._track_available_fut = asyncio.Future[_LinkedTrack]()
        if self._audio_input:
            self._audio_input.set_track_sid(None)
            self._audio_input.set_participant(None)
        if self._input_tr_output:
            self._input_tr_output.set_participant(None)
            self._input_tr_output.set_track(None)

    def _on_output_language_changed(self, language: str) -> None:
        """Retag the translation with the language it is now spoken in, and rename its track.

        With transcripts paced to the audio, the synchronizer still holds words of the old
        language while its speech plays out, and releases them after. So the change lands only
        once that speech has played and the synchronizer has closed its segment; until then
        the old words, and the old audio, keep the old label.
        """
        if language == (self._requested_output_language or self._output_language):
            return

        self._requested_output_language = language
        pending = self._relabel_atask is not None and not self._relabel_atask.done()
        if self._tr_synchronizer is None and not pending:
            # nothing holds text back, so what is published already is the old language's
            self._apply_output_language(language)
            return

        self._relabel_atask = asyncio.create_task(
            self._relabel_after_playout(language, self._relabel_atask)
        )
        self._tasks.add(self._relabel_atask)
        self._relabel_atask.add_done_callback(self._tasks.discard)

    @log_exceptions(logger=logger)
    async def _relabel_after_playout(
        self, language: str, previous: asyncio.Task[None] | None
    ) -> None:
        """Apply ``language`` once the speech captured so far has played out.

        Waits for the change made before it, then for the synchronizer to release and close
        the old language's words, so none of them go out under the new label.
        """
        if previous is not None:
            # changes land in the order they were made
            with contextlib.suppress(Exception):
                await previous

        if (synchronizer := self._tr_synchronizer) is not None:
            await synchronizer.audio_output.wait_for_playout()
            await synchronizer.barrier()

        self._apply_output_language(language)

    def _apply_output_language(self, language: str) -> None:
        """Retag the translation transcript and rename its track for ``language``, now."""
        self._output_language = language
        if self._output_tr_output and self._source_track_sid is not None:
            self._output_tr_output.set_attributes(self._output_attributes())

        if self._init_atask is not None and self._init_atask.done():
            self._rename_output_track_soon()

    @property
    def _language(self) -> str:
        """The language the translation is labelled with in the room."""
        return self._output_language or self._session.model.target_language

    def _output_attributes(self) -> dict[str, str]:
        """The attributes stamped on the translation transcript."""
        return {ATTRIBUTE_TRANSCRIPTION_LANGUAGE: self._language}

    def _output_track_name(self) -> str | None:
        """The name the translated track should carry, or ``None`` to leave it as it is."""
        if self._fixed_track_name or self._source_track_sid is None:
            return None

        return _translated_track_name(self._source_track_sid, self._language)

    def _rename_output_track_soon(self) -> None:
        """Republish the translated track under its current name, in the background."""
        if self._audio_output is None or (name := self._output_track_name()) is None:
            return

        task = asyncio.create_task(self._audio_output.set_track_name(name))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @log_exceptions(logger=logger)
    async def _init_task(self) -> None:
        await self._room_connected_fut

        for participant in self._room.remote_participants.values():
            self._on_participant_connected(participant)

        linked = await self._track_available_fut

        if self._options.get_text_output_options() is not None:
            # stamped with the track itself, not whichever microphone track its publisher
            # happens to own, and attributed to that publisher on the wire
            self._input_tr_output = _ParticipantTranscriptionOutput(
                room=self._room,
                is_delta_stream=False,
                participant=linked.participant.identity,
                track=linked.publication.sid,
            )

        self._apply_link(linked)

        if self._output_tr_output:
            # stamped with the agent's own microphone track, which is the translated one, once
            # it is published, and again whenever it is republished under a new name
            self._output_tr_output.set_participant(self._room.local_participant.identity)

        if self._audio_output:
            if (name := self._output_track_name()) is not None:
                # named before the first publish, so it never goes out under the default
                await self._audio_output.set_track_name(name)
            await self._audio_output.start()

        if not self._ready_fut.done():
            self._ready_fut.set_result(None)

    @log_exceptions(logger=logger)
    async def _forward_input_transcript(
        self, event_ch: aio.Chan[tuple[str, bool, str | None]]
    ) -> None:
        async for transcript, is_final, language in event_ch:
            if self._input_tr_output is None:
                continue

            # each interim goes out whole, so the language it carries is always current
            self._input_tr_output.set_attributes(
                {ATTRIBUTE_TRANSCRIPTION_LANGUAGE: language} if language else {}
            )
            await self._input_tr_output.capture_text(transcript)
            if is_final:
                self._input_tr_output.flush()

    def _on_input_transcribed(self, ev: Any) -> None:
        if self._input_transcript_ch:
            self._input_transcript_ch.send_nowait((ev.transcript, ev.is_final, ev.language))

    def _on_connection_state_changed(self, state: rtc.ConnectionState.ValueType) -> None:
        if self._room.isconnected() and not self._room_connected_fut.done():
            self._room_connected_fut.set_result(None)

    def _on_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        for publication in participant.track_publications.values():
            self._on_track_published(publication, participant)

    def _on_track_published(
        self, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if self._track_available_fut.done() or publication.kind != rtc.TrackKind.KIND_AUDIO:
            return

        # a named track is taken whoever published it; otherwise the search is narrowed the
        # way RoomIO narrows its participant search
        if self._track_sid is not None:
            if publication.sid != self._track_sid:
                return
        else:
            if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
                return

            if self._participant_identity is not None:
                if participant.identity != self._participant_identity:
                    return
            elif (
                participant.attributes.get(ATTRIBUTE_PUBLISH_ON_BEHALF)
                == self._room.local_participant.identity
            ):
                return

            accepted_kinds = (
                self._options.participant_kinds
                if is_given(self._options.participant_kinds)
                else DEFAULT_PARTICIPANT_KINDS
            )
            if participant.kind not in accepted_kinds:
                return

        linked = _LinkedTrack(publication, participant)
        self._track_available_fut.set_result(linked)

        if self._init_atask is not None and self._init_atask.done():
            # startup has finished, so nothing else will pick this up
            self._apply_link(linked)

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if not (linked := self.linked_participant) or participant.identity != linked.identity:
            return

        self._track_available_fut = asyncio.Future[_LinkedTrack]()

        if (
            self._options.close_on_disconnect
            and participant.disconnect_reason in DEFAULT_CLOSE_ON_DISCONNECT_REASONS
        ):
            logger.info(
                "closing translation session due to participant disconnect "
                "(disable via `TranslationRoomOptions.close_on_disconnect=False`)",
                extra={
                    trace_types.ATTR_ROOM_NAME: self._room.name,
                    trace_types.ATTR_PARTICIPANT_IDENTITY: participant.identity,
                },
            )
            self._session._close_soon(reason=CloseReason.PARTICIPANT_DISCONNECTED)
