from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_type

from livekit import api, rtc

from ... import llm, stt, tts, vad
from ...job import get_job_context
from ...llm.tool_context import ToolError, ToolFlag, function_tool
from ...log import logger
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import is_given
from ...voice.agent import AgentTask
from ...voice.background_audio import (
    AudioConfig,
    AudioSource,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    PlayHandle,
)
from ...voice.events import RunContext

if TYPE_CHECKING:
    from ...voice.audio_recognition import TurnDetectionMode


BASE_INSTRUCTIONS = """
# Identity

You are an agent that is reaching out to a supervisor for help. There has been a previous conversation
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

"""


@dataclass
class WarmTransferResult:
    email_address: str


class WarmTransferTask(AgentTask[WarmTransferResult]):
    def __init__(
        self,
        *,
        transfer_to: str,
        extra_instructions: str = "",
        hold_audio: NotGivenOr[AudioSource | AudioConfig | list[AudioConfig] | None] = NOT_GIVEN,
        sip_trunk_id: NotGivenOr[str] = NOT_GIVEN,
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
        self._supervisor_room: rtc.Room | None = None
        self._sip_trunk_id = (
            sip_trunk_id if is_given(sip_trunk_id) else os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK", "")
        )
        if not self._sip_trunk_id:
            raise ValueError(
                "`LIVEKIT_SIP_OUTBOUND_TRUNK` environment variable or `sip_trunk_id` argument must be set"
            )

        self._transfer_to = transfer_to  # phone number of the supervisor
        self._background_audio = BackgroundAudioPlayer()
        self._hold_audio_handle: PlayHandle | None = None
        self._hold_audio = (
            hold_audio
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
        # # disable `delete_on_close` for RoomIO
        # if room_io := self.session.room_io:
        #     # backup the original value
        #     self._room_delete_on_close = room_io._options.delete_room_on_close
        #     room_io._options.delete_room_on_close = False

        job_ctx = get_job_context()
        self._customer_room = job_ctx.room

        await self._start_hold()

        try:
            self._supervisor_room = await self._dial_supervisor()
        except Exception:
            logger.exception("could not dial supervisor")
            await self._stop_hold()
            self.complete(ToolError("could not dial supervisor"))
            return

        logger.info("supervisor agent entered")

    async def _start_hold(self) -> None:
        assert self._customer_room is not None

        # start the background audio
        if self._hold_audio is not None:
            await self._background_audio.start(room=self._customer_room)
            self._hold_audio_handle = self._background_audio.play(self._hold_audio, loop=True)

        self._toggle_io(False)

    async def _stop_hold(self) -> None:
        if self._hold_audio_handle:
            self._hold_audio_handle.stop()
            self._hold_audio_handle = None

        self._toggle_io(False)
        await self.session.room_io.aclose()
        await self.session.room_io.start(room=self._customer_room)
        self._toggle_io(True)

    async def _dial_supervisor(self) -> rtc.Room:
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
        ).to_jwt()

        logger.info(
            f"connecting to supervisor room {supervisor_room_name}",
            extra={"token": f"{token[:10]}...", "url": ws_url},
        )
        await room.connect(ws_url, token)

        # if supervisor hung up for whatever reason, we'd resume the customer conversation
        room.on("disconnected", self._on_supervisor_room_close)

        # restart the RoomIO for the new room
        await self.session.room_io.aclose()
        await self.session.room_io.start(room=room)

        # dial the supervisor
        await job_ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=self._sip_trunk_id,
                sip_call_to=self._transfer_to,
                room_name=supervisor_room_name,
                participant_identity="supervisor-sip",
                wait_until_answered=True,
            )
        )
        return room

    def _on_supervisor_room_close(self, reason: rtc.DisconnectReason) -> None:
        pass

    def _toggle_io(self, enabled: bool | None) -> None:
        if not self._original_io_state:
            self._original_io_state = {
                "audio_input": self.session.input.audio_enabled,
                "video_input": self.session.input.video_enabled,
                "audio_output": self.session.output.audio_enabled,
                "transcription_output": self.session.output.transcription_enabled,
                "video_output": self.session.output.video_enabled,
            }

        if enabled is None:
            # reset back to the original values
            self.session.input.set_audio_enabled(self._original_io_state["audio_input"])
            self.session.input.set_video_enabled(self._original_io_state["video_input"])
            self.session.output.set_audio_enabled(self._original_io_state["audio_output"])
            self.session.output.set_transcription_enabled(
                self._original_io_state["transcription_output"]
            )
            self.session.output.set_video_enabled(self._original_io_state["video_output"])
        else:
            self.session.input.set_audio_enabled(enabled and self._original_io_state["audio_input"])
            self.session.input.set_video_enabled(enabled and self._original_io_state["video_input"])
            self.session.output.set_audio_enabled(
                enabled and self._original_io_state["audio_output"]
            )
            self.session.output.set_transcription_enabled(
                enabled and self._original_io_state["transcription_output"]
            )
            self.session.output.set_video_enabled(
                enabled and self._original_io_state["video_output"]
            )

    @function_tool
    async def update_email_address(self, email: str, ctx: RunContext) -> str:
        """Update the email address provided by the user.

        Args:
            email: The email address provided by the user
        """
        self._email_update_speech_handle = ctx.speech_handle
        email = email.strip()

        if not re.match(EMAIL_REGEX, email):
            raise ToolError(f"Invalid email address provided: {email}")

        self._current_email = email
        separated_email = " ".join(email)

        return (
            f"The email has been updated to {email}\n"
            f"Repeat the email character by character: {separated_email} if needed\n"
            f"Prompt the user for confirmation, do not call `confirm_email_address` directly"
        )

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def confirm_email_address(self, ctx: RunContext) -> None:
        """Validates/confirms the email address provided by the user."""
        await ctx.wait_for_playout()

        if ctx.speech_handle == self._email_update_speech_handle:
            raise ToolError("error: the user must confirm the email address explicitly")

        if not self._current_email.strip():
            raise ToolError(
                "error: no email address were provided, `update_email_address` must be called before"
            )

        if not self.done():
            self.complete(GetEmailResult(email_address=self._current_email))

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def decline_email_capture(self, reason: str) -> None:
        """Handles the case when the user explicitly declines to provide an email address.

        Args:
            reason: A short explanation of why the user declined to provide the email address
        """
        if not self.done():
            self.complete(ToolError(f"couldn't get the email address: {reason}"))
