from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from livekit import api, rtc

from ... import llm, stt, tts, utils, vad
from ...job import get_job_context
from ...llm.tool_context import ToolError, ToolFlag, function_tool
from ...log import logger
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import is_given
from ...voice import room_io
from ...voice.agent import Agent, AgentTask
from ...voice.agent_session import AgentSession
from ...voice.background_audio import (
    AudioConfig,
    AudioSource,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    PlayHandle,
)

if TYPE_CHECKING:
    from ...voice.audio_recognition import TurnDetectionMode


BASE_INSTRUCTIONS = """
# Identity

You are an agent that is reaching out to a human supervisor for help. There has been a previous conversation
between you and a customer, the conversation history is included below.

# Goal

Your main goal is to give the supervisor sufficient context about why the customer had called in,
so that the supervisor could gain sufficient knowledge to help the customer directly.

# Context

In the conversation, user refers to the supervisor, customer refers to the person who's transcript is included.
Remember, you are not speaking to the customer right now, you are speaking to the supervisor.

Once the supervisor has confirmed, you should call the tool `connect_to_customer` to connect them to the customer.

Start by giving them a summary of the conversation so far, and answer any questions they might have.

## Conversation history with customer
{conversation_history}
## End of conversation history with customer

"""


@dataclass
class WarmTransferResult:
    supervisor_identity: str


class WarmTransferTask(AgentTask[WarmTransferResult]):
    def __init__(
        self,
        target_phone_number: str,
        *,
        hold_audio: NotGivenOr[AudioSource | AudioConfig | list[AudioConfig] | None] = NOT_GIVEN,
        sip_trunk_id: NotGivenOr[str] = NOT_GIVEN,
        extra_instructions: str = "",
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.FunctionTool | llm.RawFunctionTool]] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            instructions=self.get_instructions(
                chat_ctx=chat_ctx, extra_instructions=extra_instructions
            ),
            chat_ctx=NOT_GIVEN,  # don't pass the chat_ctx
            turn_detection=turn_detection,
            tools=tools or [],
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            allow_interruptions=allow_interruptions,
        )

        self._customer_room: rtc.Room | None = None
        self._supervisor_sess: AgentSession | None = None
        self._supervisor_failed_fut: asyncio.Future[None] = asyncio.Future()
        self._supervisor_identity = "supervisor-sip"

        self._target_phone_number = target_phone_number
        self._sip_trunk_id = (
            sip_trunk_id if is_given(sip_trunk_id) else os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK", "")
        )
        if not self._sip_trunk_id:
            raise ValueError(
                "`LIVEKIT_SIP_OUTBOUND_TRUNK` environment variable or `sip_trunk_id` argument must be set"
            )

        # background audio and io
        self._background_audio = BackgroundAudioPlayer()
        self._hold_audio_handle: PlayHandle | None = None
        self._hold_audio = (
            cast(AudioSource | AudioConfig | list[AudioConfig] | None, hold_audio)
            if is_given(hold_audio)
            else AudioConfig(BuiltinAudioClip.HOLD_MUSIC, volume=0.8)
        )

        self._original_io_state: dict[str, bool] = {}

    def get_instructions(
        self, *, chat_ctx: NotGivenOr[llm.ChatContext], extra_instructions: str = ""
    ) -> str:
        prev_convo = ""
        if chat_ctx:
            context_copy = chat_ctx.copy(
                exclude_empty_message=True, exclude_instructions=True, exclude_function_call=True
            )
            for msg in context_copy.items:
                if msg.type != "message":
                    continue
                role = "Customer" if msg.role == "user" else "Assistant"
                prev_convo += f"{role}: {msg.text_content}\n"
        return BASE_INSTRUCTIONS.format(conversation_history=prev_convo) + extra_instructions

    async def on_enter(self) -> None:
        job_ctx = get_job_context()
        self._customer_room = job_ctx.room

        # start the background audio
        if self._hold_audio is not None:
            await self._background_audio.start(room=self._customer_room)
            self._hold_audio_handle = self._background_audio.play(self._hold_audio, loop=True)

        self._set_io_enabled(False)

        try:
            dial_supervisor_task = asyncio.create_task(self._dial_supervisor())
            done, _ = await asyncio.wait(
                (dial_supervisor_task, self._supervisor_failed_fut),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if dial_supervisor_task not in done:
                raise RuntimeError()

            self._supervisor_sess = dial_supervisor_task.result()
            self._supervisor_sess.generate_reply(
                instructions="you are talking to the supervisor now. give a brief introduction of the conversation so far."
            )

        except Exception:
            logger.exception("could not dial supervisor")
            self._set_result(ToolError("could not dial supervisor"))
            return

        finally:
            await utils.aio.cancel_and_wait(dial_supervisor_task)

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def connect_to_customer(self) -> None:
        """Called when the supervisor wants to connect to the customer."""
        logger.debug("connecting to customer")
        assert self._customer_room is not None

        await self._merge_calls()
        self._set_result(WarmTransferResult(supervisor_identity=self._supervisor_identity))

        # when the customer or supervisor leaves the room, we'll delete the room
        self._customer_room.on("participant_connected", self._on_customer_participant_connected)

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def decline_transfer(self, reason: str) -> None:
        """Handles the case when the supervisor explicitly declines to connect to the customer.

        Args:
            reason: A short explanation of why the supervisor declined to connect to the customer
        """
        self._set_result(ToolError(f"supervisor declined to connect: {reason}"))

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def voicemail_detected(self) -> None:
        """Called when the call reaches voicemail. Use this tool AFTER you hear the voicemail greeting"""
        self._set_result(ToolError("voicemail detected"))

    def _on_supervisor_room_close(self, reason: rtc.DisconnectReason.ValueType) -> None:
        logger.debug(
            "supervisor's room closed",
            extra={"reason": rtc.DisconnectReason.Name(reason)},
        )
        with contextlib.suppress(asyncio.InvalidStateError):
            self._supervisor_failed_fut.set_result(None)

        self._set_result(ToolError(f"room closed: {rtc.DisconnectReason.Name(reason)}"))

    def _on_customer_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.kind not in (
            rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
            rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
        ):
            return

        logger.info(f"participant connected from customer room: {participant.identity}")

        assert self._customer_room is not None
        self._customer_room.off("participant_connected", self._on_customer_participant_connected)
        job_ctx = get_job_context()
        job_ctx.delete_room(room_name=self._customer_room.name)

    def _set_result(self, result: WarmTransferResult | Exception) -> None:
        if self.done():
            return

        if self._supervisor_sess:
            self._supervisor_sess.shutdown()
            self._supervisor_sess = None

        if self._hold_audio_handle:
            self._hold_audio_handle.stop()
            self._hold_audio_handle = None

        self._set_io_enabled(True)
        self.complete(result)

    async def _dial_supervisor(self) -> AgentSession:
        assert self._customer_room is not None

        job_ctx = get_job_context()
        ws_url = job_ctx._info.url

        # create a new room for the supervisor
        supervisor_room_name = self._customer_room.name + "-supervisor"
        room = rtc.Room()
        token = (
            api.AccessToken()
            .with_identity(self._customer_room.local_participant.identity)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=supervisor_room_name,
                    can_update_own_metadata=True,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .with_kind("agent")
        ).to_jwt()

        logger.debug(
            "connecting to supervisor room",
            extra={"ws_url": ws_url, "supervisor_room_name": supervisor_room_name},
        )
        await room.connect(ws_url, token)

        # if supervisor hung up for whatever reason, we'd resume the customer conversation
        room.on("disconnected", self._on_supervisor_room_close)

        supervisor_sess: AgentSession = AgentSession(
            vad=self.session.vad or NOT_GIVEN,
            llm=self.session.llm or NOT_GIVEN,
            stt=self.session.stt or NOT_GIVEN,
            tts=self.session.tts or NOT_GIVEN,
            turn_detection=self.session.turn_detection or NOT_GIVEN,
        )
        # create a copy of this AgentTask
        supervisor_agent = Agent(
            instructions=self.instructions,
            turn_detection=self.turn_detection,
            stt=self.stt,
            vad=self.vad,
            llm=self.llm,
            tts=self.tts,
            tools=self.tools,
            chat_ctx=self.chat_ctx,
            allow_interruptions=self.allow_interruptions,
        )
        await supervisor_sess.start(
            agent=supervisor_agent,
            room=room,
            room_options=room_io.RoomOptions(
                close_on_disconnect=True,
                delete_room_on_close=True,
                participant_identity=self._supervisor_identity,
            ),
            record=False,  # TODO: support recording on multiple sessions?
        )

        # dial the supervisor
        await job_ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=self._sip_trunk_id,
                sip_call_to=self._target_phone_number,
                room_name=supervisor_room_name,
                participant_identity=self._supervisor_identity,
                wait_until_answered=True,
            )
        )

        return supervisor_sess

    async def _merge_calls(self) -> None:
        assert self._customer_room is not None
        assert self._supervisor_sess is not None

        job_ctx = get_job_context()
        supervisor_room = self._supervisor_sess.room_io.room
        # we no longer care about the supervisor session. it's supposed to be over
        supervisor_room.off("disconnected", self._on_supervisor_room_close)

        logger.debug(
            f"moving {self._supervisor_identity} to customer room {self._customer_room.name}"
        )
        await job_ctx.api.room.move_participant(
            api.MoveParticipantRequest(
                room=supervisor_room.name,
                identity=self._supervisor_identity,
                destination_room=self._customer_room.name,
            )
        )

    def _set_io_enabled(self, enabled: bool) -> None:
        input = self.session.input
        output = self.session.output

        if not self._original_io_state:
            self._original_io_state = {
                "audio_input": input.audio_enabled,
                "video_input": input.video_enabled,
                "audio_output": output.audio_enabled,
                "transcription_output": output.transcription_enabled,
                "video_output": output.video_enabled,
            }

        if input.audio:
            input.set_audio_enabled(enabled and self._original_io_state["audio_input"])
        if input.video:
            input.set_video_enabled(enabled and self._original_io_state["video_input"])
        if output.audio:
            output.set_audio_enabled(enabled and self._original_io_state["audio_output"])
        if output.transcription:
            output.set_transcription_enabled(
                enabled and self._original_io_state["transcription_output"]
            )
        if output.video:
            output.set_video_enabled(enabled and self._original_io_state["video_output"])
