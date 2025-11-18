from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from livekit import api, rtc

from ... import utils
from ...job import get_job_context
from ...log import logger
from ...types import (
    ATTRIBUTE_AGENT_STATE,
    ATTRIBUTE_PUBLISH_ON_BEHALF,
    NOT_GIVEN,
    TOPIC_CHAT,
    NotGivenOr,
)
from ..events import AgentStateChangedEvent, CloseEvent, CloseReason, UserInputTranscribedEvent
from ..io import AudioInput, AudioOutput, TextOutput, VideoInput
from ..transcription import TranscriptSynchronizer
from ._pre_connect_audio import PreConnectAudioHandler

if TYPE_CHECKING:
    from ..agent_session import AgentSession


from ._input import _ParticipantAudioInputStream, _ParticipantVideoInputStream
from ._output import _ParticipantAudioOutput, _ParticipantTranscriptionOutput
from .types import (
    DEFAULT_CLOSE_ON_DISCONNECT_REASONS,
    DEFAULT_PARTICIPANT_KINDS,
    RoomInputOptions,
    RoomOptions,
    RoomOutputOptions,
    TextInputCallback,
    TextInputEvent,
)


class RoomIO:
    def __init__(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        participant: rtc.RemoteParticipant | str | None = None,
        options: NotGivenOr[RoomOptions] = NOT_GIVEN,
        # deprecated
        input_options: NotGivenOr[RoomInputOptions] = NOT_GIVEN,
        output_options: NotGivenOr[RoomOutputOptions] = NOT_GIVEN,
    ) -> None:
        self._options = RoomOptions._ensure_options(
            options, room_input_options=input_options, room_output_options=output_options
        )
        self._text_input_cb: TextInputCallback | None = None

        self._agent_session, self._room = agent_session, room
        # self._input_options = input_options
        # self._output_options = output_options
        self._participant_identity = (
            participant.identity if isinstance(participant, rtc.RemoteParticipant) else participant
        )
        if self._participant_identity is None and utils.is_given(
            self._options.participant_identity
        ):
            self._participant_identity = self._options.participant_identity

        self._audio_input: _ParticipantAudioInputStream | None = None
        self._video_input: _ParticipantVideoInputStream | None = None
        self._audio_output: _ParticipantAudioOutput | None = None
        self._user_tr_output: _ParticipantTranscriptionOutput | None = None
        self._agent_tr_output: _ParticipantTranscriptionOutput | None = None
        self._tr_synchronizer: TranscriptSynchronizer | None = None

        self._participant_available_fut = asyncio.Future[rtc.RemoteParticipant]()
        self._room_connected_fut = asyncio.Future[None]()

        self._init_atask: asyncio.Task[None] | None = None
        self._user_transcript_ch = utils.aio.Chan[UserInputTranscribedEvent]()
        self._user_transcript_atask: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._update_state_atask: asyncio.Task[None] | None = None
        self._close_session_atask: asyncio.Task[None] | None = None
        self._delete_room_task: asyncio.Future[api.DeleteRoomResponse] | None = None

        self._pre_connect_audio_handler: PreConnectAudioHandler | None = None
        self._text_stream_handler_registered = False

    async def start(self) -> None:
        # -- create inputs --
        input_audio_options = self._options.get_audio_input_options()
        if input_audio_options and input_audio_options.pre_connect_audio:
            self._pre_connect_audio_handler = PreConnectAudioHandler(
                room=self._room,
                timeout=input_audio_options.pre_connect_audio_timeout,
            )
            self._pre_connect_audio_handler.register()

        input_text_options = self._options.get_text_input_options()
        if input_text_options:
            self._text_input_cb = input_text_options.text_input_cb
            try:
                self._room.register_text_stream_handler(TOPIC_CHAT, self._on_user_text_input)
                self._text_stream_handler_registered = True
            except ValueError:
                if utils.is_given(self._options.text_input):
                    logger.warning(
                        f"text stream handler for topic '{TOPIC_CHAT}' already set, ignoring"
                    )
        else:
            self._text_input_cb = None

        input_video_options = self._options.get_video_input_options()
        if input_video_options:
            self._video_input = _ParticipantVideoInputStream(self._room)

        if input_audio_options:
            self._audio_input = _ParticipantAudioInputStream(
                self._room,
                sample_rate=input_audio_options.sample_rate,
                num_channels=input_audio_options.num_channels,
                frame_size_ms=input_audio_options.frame_size_ms,
                noise_cancellation=input_audio_options.noise_cancellation,
                pre_connect_audio_handler=self._pre_connect_audio_handler,
            )

        # -- create outputs --
        output_audio_options = self._options.get_audio_output_options()
        if output_audio_options:
            self._audio_output = _ParticipantAudioOutput(
                self._room,
                sample_rate=output_audio_options.sample_rate,
                num_channels=output_audio_options.num_channels,
                track_publish_options=output_audio_options.track_publish_options,
                track_name=(
                    output_audio_options.track_name
                    if utils.is_given(output_audio_options.track_name)
                    else "roomio_audio"
                ),
            )

        output_text_options = self._options.get_text_output_options()
        if output_text_options:
            self._user_tr_output = _ParticipantTranscriptionOutput(
                room=self._room, is_delta_stream=False, participant=self._participant_identity
            )
            self._user_transcript_atask = asyncio.create_task(self._forward_user_transcript())

            # TODO(long): add next in the chain for session.output.transcription
            self._agent_tr_output = _ParticipantTranscriptionOutput(
                room=self._room, is_delta_stream=True, participant=None
            )

            # use the RoomIO's audio output if available, otherwise use the agent's audio output
            # (e.g the audio output isn't using RoomIO with our avatar datastream impl)
            if output_text_options.sync_transcription is not False and (
                audio_output := self._audio_output or self._agent_session.output.audio
            ):
                self._tr_synchronizer = TranscriptSynchronizer(
                    next_in_chain_audio=audio_output,
                    next_in_chain_text=self._agent_tr_output,
                    speed=output_text_options.transcription_speed_factor,
                )

        # -- set the room event handlers --
        self._room.on("participant_connected", self._on_participant_connected)
        self._room.on("connection_state_changed", self._on_connection_state_changed)
        self._room.on("participant_disconnected", self._on_participant_disconnected)
        if self._room.isconnected():
            self._on_connection_state_changed(rtc.ConnectionState.CONN_CONNECTED)

        self._init_atask = asyncio.create_task(self._init_task())

        # -- attach to the agent session --
        if self.audio_input:
            self._agent_session.input.audio = self.audio_input

        if self.video_input:
            self._agent_session.input.video = self.video_input

        if self.audio_output:
            self._agent_session.output.audio = self.audio_output

        if self.transcription_output:
            self._agent_session.output.transcription = self.transcription_output

        self._agent_session.on("agent_state_changed", self._on_agent_state_changed)
        self._agent_session.on("user_input_transcribed", self._on_user_input_transcribed)
        self._agent_session.on("close", self._on_agent_session_close)

    @property
    def room(self) -> rtc.Room:
        return self._room

    async def aclose(self) -> None:
        self._room.off("participant_connected", self._on_participant_connected)
        self._room.off("connection_state_changed", self._on_connection_state_changed)
        self._agent_session.off("agent_state_changed", self._on_agent_state_changed)
        self._agent_session.off("user_input_transcribed", self._on_user_input_transcribed)
        self._agent_session.off("close", self._on_agent_session_close)

        if self._text_stream_handler_registered:
            self._room.unregister_text_stream_handler(TOPIC_CHAT)
            self._text_stream_handler_registered = False

        if self._init_atask:
            await utils.aio.cancel_and_wait(self._init_atask)

        self._user_transcript_ch.close()
        if self._user_transcript_atask:
            await utils.aio.cancel_and_wait(self._user_transcript_atask)

        if self._update_state_atask:
            await utils.aio.cancel_and_wait(self._update_state_atask)

        if self._pre_connect_audio_handler:
            await self._pre_connect_audio_handler.aclose()

        if self._audio_input:
            await self._audio_input.aclose()
        if self._video_input:
            await self._video_input.aclose()

        if self._tr_synchronizer:
            await self._tr_synchronizer.aclose()

        if self._audio_output:
            await self._audio_output.aclose()

        # cancel and wait for all pending tasks
        await utils.aio.cancel_and_wait(*self._tasks)
        self._tasks.clear()

    @property
    def audio_output(self) -> AudioOutput | None:
        if self._tr_synchronizer:
            return self._tr_synchronizer.audio_output

        return self._audio_output

    @property
    def transcription_output(self) -> TextOutput | None:
        if self._tr_synchronizer:
            return self._tr_synchronizer.text_output

        return self._agent_tr_output

    @property
    def audio_input(self) -> AudioInput | None:
        return self._audio_input

    @property
    def video_input(self) -> VideoInput | None:
        return self._video_input

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

    def set_participant(self, participant_identity: str | None) -> None:
        """Switch audio and video streams to specified participant"""
        if participant_identity is None:
            self.unset_participant()
            return

        if (
            self._participant_identity is not None
            and self._participant_identity != participant_identity
        ):
            # reset future if switching to a different participant
            self._participant_available_fut = asyncio.Future[rtc.RemoteParticipant]()

            # check if new participant is already connected
            for participant in self._room.remote_participants.values():
                if participant.identity == participant_identity:
                    self._participant_available_fut.set_result(participant)
                    break

        # update participant identity and handlers
        self._participant_identity = participant_identity
        if self._audio_input:
            self._audio_input.set_participant(participant_identity)
        if self._video_input:
            self._video_input.set_participant(participant_identity)

        if self._user_tr_output:
            self._user_tr_output.set_participant(participant_identity)

    def unset_participant(self) -> None:
        self._participant_identity = None
        self._participant_available_fut = asyncio.Future[rtc.RemoteParticipant]()
        if self._audio_input:
            self._audio_input.set_participant(None)
        if self._video_input:
            self._video_input.set_participant(None)

        if self._user_tr_output:
            self._user_tr_output.set_participant(None)

    @utils.log_exceptions(logger=logger)
    async def _init_task(self) -> None:
        await self._room_connected_fut

        # check existing participants
        for participant in self._room.remote_participants.values():
            self._on_participant_connected(participant)

        participant = await self._participant_available_fut
        self.set_participant(participant.identity)

        # init outputs
        if self._agent_tr_output:
            self._agent_tr_output.set_participant(self._room.local_participant.identity)

        if self._audio_output:
            await self._audio_output.start()

    @utils.log_exceptions(logger=logger)
    async def _forward_user_transcript(self) -> None:
        async for ev in self._user_transcript_ch:
            if self._user_tr_output is None:
                continue

            await self._user_tr_output.capture_text(ev.transcript)
            if ev.is_final:
                self._user_tr_output.flush()

    def _on_connection_state_changed(self, state: rtc.ConnectionState.ValueType) -> None:
        if self._room.isconnected() and not self._room_connected_fut.done():
            self._room_connected_fut.set_result(None)

    def _on_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        if self._participant_available_fut.done():
            return

        if self._participant_identity is not None:
            if participant.identity != self._participant_identity:
                return
        # otherwise, skip participants that are marked as publishing for this agent
        elif (
            participant.attributes.get(ATTRIBUTE_PUBLISH_ON_BEHALF)
            == self._room.local_participant.identity
        ):
            return

        accepted_kinds = self._options.participant_kinds or DEFAULT_PARTICIPANT_KINDS
        if participant.kind not in accepted_kinds:
            # not an accepted participant kind, skip
            return

        self._participant_available_fut.set_result(participant)

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if not (linked := self.linked_participant) or participant.identity != linked.identity:
            return
        self._participant_available_fut = asyncio.Future[rtc.RemoteParticipant]()

        if (
            self._options.close_on_disconnect
            and participant.disconnect_reason in DEFAULT_CLOSE_ON_DISCONNECT_REASONS
            and not self._close_session_atask
            and not self._delete_room_task
        ):
            logger.info(
                "closing agent session due to participant disconnect "
                "(disable via `RoomInputOptions.close_on_disconnect=False`)",
                extra={
                    "participant": participant.identity,
                    "reason": rtc.DisconnectReason.Name(
                        participant.disconnect_reason or rtc.DisconnectReason.UNKNOWN_REASON
                    ),
                },
            )
            self._agent_session._close_soon(reason=CloseReason.PARTICIPANT_DISCONNECTED)

    def _on_user_input_transcribed(self, ev: UserInputTranscribedEvent) -> None:
        if self._user_transcript_atask:
            self._user_transcript_ch.send_nowait(ev)

    def _on_user_text_input(self, reader: rtc.TextStreamReader, participant_identity: str) -> None:
        if participant_identity != self._participant_identity:
            return

        participant = self._room.remote_participants.get(participant_identity)
        if not participant:
            logger.warning("participant not found, ignoring text input")
            return

        async def _read_text(text_input_cb: TextInputCallback) -> None:
            text = await reader.read_all()

            text_input_result = text_input_cb(
                self._agent_session,
                TextInputEvent(text=text, info=reader.info, participant=participant),
            )
            if asyncio.iscoroutine(text_input_result):
                await text_input_result

        if self._text_input_cb is None:
            logger.error("text input callback is not set, ignoring text input")
            return

        task = asyncio.create_task(_read_text(self._text_input_cb))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_agent_state_changed(self, ev: AgentStateChangedEvent) -> None:
        @utils.log_exceptions(logger=logger)
        async def _set_state() -> None:
            if self._room.isconnected():
                await self._room.local_participant.set_attributes(
                    {ATTRIBUTE_AGENT_STATE: ev.new_state}
                )

        if self._update_state_atask is not None:
            self._update_state_atask.cancel()

        self._update_state_atask = asyncio.create_task(_set_state())

    def _on_agent_session_close(self, ev: CloseEvent) -> None:
        def _on_delete_room_task_done(task: asyncio.Future[api.DeleteRoomResponse]) -> None:
            self._delete_room_task = None

        if self._options.delete_room_on_close and self._delete_room_task is None:
            job_ctx = get_job_context()
            logger.info(
                "deleting room on agent session close (disable via `RoomInputOptions.delete_room_on_close=False`)"
            )
            self._delete_room_task = job_ctx.delete_room()
            self._delete_room_task.add_done_callback(_on_delete_room_task_done)
