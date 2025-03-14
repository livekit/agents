from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    AsyncIterable,
    Callable,
    Coroutine,
    Optional,
)

from livekit import rtc

from ... import utils
from ...log import logger
from ...types import ATTRIBUTE_AGENT_STATE, TOPIC_CHAT
from ..events import AgentStateChangedEvent, UserInputTranscribedEvent
from ..io import AudioSink, TextSink
from ..transcription import TextSynchronizer

if TYPE_CHECKING:
    from ..voice_agent import VoiceAgent


from ._input import _ParticipantAudioInputStream, _ParticipantVideoInputStream
from ._output import (
    _ParallelTextSink,
    _ParticipantAudioSink,
    _ParticipantLegacyTranscriptionSink,
    _ParticipantTranscriptionSink,
)

ATTRIBUTE_PUBLISH_ON_BEHALF = "lk.publish_on_behalf"


@dataclass
class TextInputEvent:
    text: str
    info: rtc.TextStreamInfo
    participant: rtc.RemoteParticipant


TextInputCallback = Callable[["VoiceAgent", TextInputEvent], Optional[Coroutine[None, None, None]]]


def _default_text_input_cb(agent: "VoiceAgent", ev: TextInputEvent) -> None:
    agent.interrupt()
    agent.generate_reply(user_input=ev.text)


@dataclass(frozen=True)
class RoomInputOptions:
    text_enabled: bool = True
    audio_enabled: bool = True
    video_enabled: bool = False
    audio_sample_rate: int = 24000
    audio_num_channels: int = 1
    noise_cancellation: rtc.NoiseCancellationOptions | None = None
    text_input_cb: TextInputCallback = _default_text_input_cb


@dataclass(frozen=True)
class RoomOutputOptions:
    transcription_enabled: bool = True
    audio_enabled: bool = True
    audio_sample_rate: int = 24000
    audio_num_channels: int = 1
    audio_publish_options: rtc.TrackPublishOptions = field(
        default_factory=lambda: rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )


DEFAULT_ROOM_INPUT_OPTIONS = RoomInputOptions()
DEFAULT_ROOM_OUTPUT_OPTIONS = RoomOutputOptions()


class RoomIO:
    def __init__(
        self,
        agent: "VoiceAgent",
        room: rtc.Room,
        *,
        participant: rtc.RemoteParticipant | str | None = None,
        input_options: RoomInputOptions = DEFAULT_ROOM_INPUT_OPTIONS,
        output_options: RoomOutputOptions = DEFAULT_ROOM_OUTPUT_OPTIONS,
    ) -> None:
        self._agent, self._room = agent, room
        self._input_options = input_options
        self._output_options = output_options
        self._participant_identity = (
            participant.identity if isinstance(participant, rtc.RemoteParticipant) else participant
        )

        self._audio_input: _ParticipantAudioInputStream | None = None
        self._video_input: _ParticipantVideoInputStream | None = None
        self._audio_output: _ParticipantAudioSink | None = None
        self._user_tr_output: _ParallelTextSink | None = None
        self._agent_tr_output: _ParallelTextSink | None = None
        self._tr_output_synchronizer: TextSynchronizer | None = None

        self._participant_available_fut = asyncio.Future[rtc.RemoteParticipant]()

        self._tasks: set[asyncio.Task] = set()
        self._update_state_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._room.on("participant_connected", self._on_participant_connected)
        self._room.on("participant_disconnected", self._on_participant_disconnected)

        for participant in self._room.remote_participants.values():
            self._on_participant_connected(participant)

        if self._input_options.text_enabled:
            self._room.register_text_stream_handler(TOPIC_CHAT, self._on_user_text_input)

        if self._input_options.video_enabled:
            self._video_input = _ParticipantVideoInputStream(self._room)

        if self._input_options.audio_enabled:
            self._audio_input = _ParticipantAudioInputStream(
                self._room,
                sample_rate=self._input_options.audio_sample_rate,
                num_channels=self._input_options.audio_num_channels,
                noise_cancellation=self._input_options.noise_cancellation,
            )

        def _create_transcription_output(is_delta_stream: bool) -> _ParallelTextSink:
            return _ParallelTextSink(
                _ParticipantLegacyTranscriptionSink(room=self._room),
                _ParticipantTranscriptionSink(room=self._room, is_delta_stream=is_delta_stream),
            )

        if self._output_options.audio_enabled:
            self._audio_output = _ParticipantAudioSink(
                self._room,
                sample_rate=self._output_options.audio_sample_rate,
                num_channels=self._output_options.audio_num_channels,
                track_publish_options=self._output_options.audio_publish_options,
            )

        if self._output_options.transcription_enabled:
            self._user_tr_output = _create_transcription_output(is_delta_stream=False)
            self._agent_tr_output = _create_transcription_output(is_delta_stream=True)

            self._agent.on("user_input_transcribed", self._on_user_input_transcribed)

            audio_output = self._audio_output or self._agent.output.audio
            if audio_output:
                self._tr_synchronizer = TextSynchronizer(
                    audio_output, text_sink=self._agent_tr_output
                )
                self._agent.output.on("audio_changed", self._on_agent_output_changed)
                self._agent.output.on("transcription_changed", self._on_agent_output_changed)

        # TODO(theomonnom): ideally we're consistent and every input/output has a start method
        if self._audio_output:
            await self._audio_output.start()

        # wait for the specified participant or the first participant joined
        input_participant = await self._participant_available_fut
        self.set_participant(input_participant.identity)

        if self.audio_input:
            self._agent.input.audio = self.audio_input

        if self.video_input:
            self._agent.input.video = self.video_input

        if self.audio_output:
            self._agent.output.audio = self.audio_output

        if self.transcription_output:
            self._agent.output.transcription = self.transcription_output

        self._agent.on("agent_state_changed", self._on_agent_state_changed)

    async def aclose(self) -> None:
        self._room.off("participant_connected", self._on_participant_connected)
        self._room.off("participant_disconnected", self._on_participant_disconnected)

        if self._audio_input_handle:
            await self._audio_input_handle.aclose()
        if self._video_input_handle:
            await self._video_input_handle.aclose()

        if self._tr_synchronizer:
            self._agent.output.off("audio_changed", self._on_agent_output_changed)
            self._agent.output.off("transcription_changed", self._on_agent_output_changed)
            await self._tr_synchronizer.aclose()

        # cancel and wait for all pending tasks
        await utils.aio.cancel_and_wait(*self._tasks)
        self._tasks.clear()

    @property
    def audio_output(self) -> AudioSink | None:
        if self._tr_synchronizer:
            return self._tr_synchronizer.audio_sink
        return self._audio_output

    @property
    def transcription_output(self) -> TextSink | None:
        if self._tr_synchronizer:
            return self._tr_synchronizer.text_sink
        return self._agent_transcription

    @property
    def audio_input(self) -> AsyncIterable[rtc.AudioFrame] | None:
        if not self._audio_input_handle:
            return None
        return self._audio_input_handle.stream

    @property
    def video_input(self) -> AsyncIterable[rtc.VideoFrame] | None:
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
            # check if new participant is already connected
            for participant in self._room.remote_participants.values():
                if participant.identity == participant_identity:
                    self._participant_connected.set_result(participant)
                    break

        # update participant identity and handlers
        self._participant_identity = participant_identity
        if self._audio_input:
            self._audio_input.set_participant(participant_identity)
        if self._video_input:
            self._video_input.set_participant(participant_identity)

        self._update_user_transcription(participant_identity)

    def unset_participant(self) -> None:
        self._participant_identity = None
        if self._audio_input_handle:
            self._audio_input_handle.set_participant(None)
        if self._video_input_handle:
            self._video_input_handle.set_participant(None)
        self._update_user_transcription(None)
        logger.debug("unset participant")

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
        if self._participant_identity is None or self._participant_identity != participant.identity:
            return

    def _on_user_input_transcribed(self, ev: UserInputTranscribedEvent) -> None:
        if self._user_transcription is None:
            return

        async def _capture_text():
            await self._user_transcription.capture_text(ev.transcript)

            if ev.is_final:
                self._user_transcription.flush()

        task = asyncio.create_task(_capture_text())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_user_text_input(self, reader: rtc.TextStreamReader, participant_identity: str) -> None:
        if participant_identity != self._participant_identity:
            return

        participant = self._room.remote_participants.get(participant_identity)
        if not participant:
            logger.warning("participant not found, ignoring text input")
            return

        async def _read_text():
            text = await reader.read_all()

            if self._input_options.text_input_cb:
                text_input_result = self._input_options.text_input_cb(
                    self._agent,
                    TextInputEvent(text=text, info=reader.info, participant=participant),
                )
                if asyncio.iscoroutine(text_input_result):
                    await text_input_result

        task = asyncio.create_task(_read_text())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_agent_state_changed(self, ev: AgentStateChangedEvent):
        @utils.log_exceptions(logger=logger)
        async def _set_state() -> None:
            if self._room.isconnected():
                await self._room.local_participant.set_attributes({ATTRIBUTE_AGENT_STATE: ev.state})

        if self._update_state_task is not None:
            self._update_state_task.cancel()

        self._update_state_task = asyncio.create_task(_set_state())

    def _on_agent_output_changed(self, sink: AudioSink | TextSink | None) -> None:
        if not self._tr_synchronizer:
            return

        sync_enabled = (
            self._agent.output.audio is self.audio_output
            and self._agent.output.transcription is self.transcription_output
        )
        self._tr_synchronizer.set_sync_enabled(sync_enabled)

    def _update_user_transcription(self, participant_identity: str | None) -> None:
        if not self._user_transcription:
            return

        for sink in self._user_transcription._sinks:
            assert isinstance(sink, (TextStreamSink, RoomTranscriptEventSink))
            sink.set_participant(participant_identity)
