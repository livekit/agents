from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from livekit import rtc

from ...job import DEFAULT_PARTICIPANT_KINDS
from ...log import logger
from ...types import ATTRIBUTE_PUBLISH_ON_BEHALF, NOT_GIVEN, NotGivenOr
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


@dataclass
class TranslationRoomOptions:
    """How a :class:`~livekit.agents.beta.translation.TranslationSession` reaches a room.

    The subset of ``RoomOptions`` a translator can use: it hears one participant and speaks
    its translation back into the room, and there is no text input to answer.
    """

    audio_input: AudioInputOptions | bool = True
    """The speech to translate. Pass ``False`` to feed the session audio yourself."""
    audio_output: AudioOutputOptions | bool = True
    """The track the translation is spoken on. Pass ``False`` to take the audio yourself."""
    text_output: TextOutputOptions | bool = True
    """Transcripts published to the room: what was heard, and what was spoken back."""
    participant_identity: NotGivenOr[str] = NOT_GIVEN
    """The participant to listen to. If not given, the first eligible one to connect."""
    participant_kinds: NotGivenOr[list[rtc.ParticipantKind.ValueType]] = NOT_GIVEN
    """Participant kinds accepted when linking automatically."""
    close_on_disconnect: bool = True
    """Close the session when the linked participant leaves of their own accord."""

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
    part a translator does use — link a participant, take their microphone, publish the
    translated track, and publish both transcripts.
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

        self._participant_available_fut = asyncio.Future[rtc.RemoteParticipant]()
        self._room_connected_fut = asyncio.Future[None]()
        self._ready_fut = asyncio.Future[None]()
        self._init_atask: asyncio.Task[None] | None = None
        self._input_transcript_ch: aio.Chan[tuple[str, bool]] | None = None
        self._input_transcript_atask: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

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
            # what was heard is attributed to the speaker, what was spoken back to the agent
            self._input_tr_output = _ParticipantTranscriptionOutput(
                room=self._room, is_delta_stream=False, participant=self._participant_identity
            )
            self._input_transcript_ch = aio.Chan[tuple[str, bool]]()
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
    def linked_participant(self) -> rtc.RemoteParticipant | None:
        if not self._participant_available_fut.done():
            return None

        return self._participant_available_fut.result()

    @property
    def subscribed_fut(self) -> asyncio.Future[None] | None:
        if self._audio_output:
            return self._audio_output.subscribed
        return None

    async def wait_for_ready(self) -> None:
        """Wait until the participant is linked and the translated track is published."""
        await self._ready_fut

    def set_participant(self, participant_identity: str | None) -> None:
        """Listen to another participant."""
        if participant_identity is None:
            self.unset_participant()
            return

        linked = self.linked_participant
        if linked is None or linked.identity != participant_identity:
            if self._participant_available_fut.done():
                self._participant_available_fut = asyncio.Future[rtc.RemoteParticipant]()

            for participant in self._room.remote_participants.values():
                if participant.identity == participant_identity:
                    self._participant_available_fut.set_result(participant)
                    break

        self._participant_identity = participant_identity
        if self._audio_input:
            self._audio_input.set_participant(participant_identity)
        if self._input_tr_output:
            self._input_tr_output.set_participant(participant_identity)

        logger.info(
            "TranslationRoomIO linked to participant",
            extra={"participant": participant_identity, "room": self._room.name},
        )

    def unset_participant(self) -> None:
        self._participant_identity = None
        self._participant_available_fut = asyncio.Future[rtc.RemoteParticipant]()
        if self._audio_input:
            self._audio_input.set_participant(None)
        if self._input_tr_output:
            self._input_tr_output.set_participant(None)

    @log_exceptions(logger=logger)
    async def _init_task(self) -> None:
        await self._room_connected_fut

        for participant in self._room.remote_participants.values():
            self._on_participant_connected(participant)

        participant = await self._participant_available_fut
        self.set_participant(participant.identity)

        if self._output_tr_output:
            self._output_tr_output.set_participant(self._room.local_participant.identity)

        if self._audio_output:
            await self._audio_output.start()

        if not self._ready_fut.done():
            self._ready_fut.set_result(None)

    @log_exceptions(logger=logger)
    async def _forward_input_transcript(self, event_ch: aio.Chan[tuple[str, bool]]) -> None:
        async for transcript, is_final in event_ch:
            if self._input_tr_output is None:
                continue

            await self._input_tr_output.capture_text(transcript)
            if is_final:
                self._input_tr_output.flush()

    def _on_input_transcribed(self, ev: Any) -> None:
        if self._input_transcript_ch:
            self._input_transcript_ch.send_nowait((ev.transcript, ev.is_final))

    def _on_connection_state_changed(self, state: rtc.ConnectionState.ValueType) -> None:
        if self._room.isconnected() and not self._room_connected_fut.done():
            self._room_connected_fut.set_result(None)

    def _on_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        if self._participant_available_fut.done():
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

        self._participant_available_fut.set_result(participant)

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if not (linked := self.linked_participant) or participant.identity != linked.identity:
            return

        self._participant_available_fut = asyncio.Future[rtc.RemoteParticipant]()

        if (
            self._options.close_on_disconnect
            and participant.disconnect_reason in DEFAULT_CLOSE_ON_DISCONNECT_REASONS
        ):
            logger.info(
                "closing translation session due to participant disconnect "
                "(disable via `TranslationRoomOptions.close_on_disconnect=False`)",
                extra={"room": self._room.name, "participant": participant.identity},
            )
            self._session._close_soon(reason=CloseReason.PARTICIPANT_DISCONNECTED)
