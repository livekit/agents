from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union, cast

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

You are an agent that is reaching out to a human agent for help. There has been a previous conversation
between you and a caller, the conversation history is included below.

# Goal

Your main goal is to give the human agent sufficient context about why the caller had called in,
so that the human agent could gain sufficient knowledge to help the caller directly.

# Context

In the conversation, user refers to the human agent, caller refers to the person who's transcript is included.
Remember, you are not speaking to the caller right now, you are speaking to the human agent.

Once the human agent has confirmed, you should call the tool `connect_to_caller` to connect them to the caller.

Start by giving them a summary of the conversation so far, and answer any questions they might have.

## Conversation history with caller
{conversation_history}
## End of conversation history with caller

You are talking to the human agent now,
give a brief introduction of the conversation so far, and ask if they want to connect to the caller.
"""


@dataclass
class WarmTransferResult:
    human_agent_identity: str


class WarmTransferTask(AgentTask[WarmTransferResult]):
    def __init__(
        self,
        target_phone_number: str,
        *,
        hold_audio: NotGivenOr[AudioSource | AudioConfig | list[AudioConfig] | None] = NOT_GIVEN,
        sip_trunk_id: NotGivenOr[str] = NOT_GIVEN,
        sip_number: NotGivenOr[str] = NOT_GIVEN,
        extra_instructions: str = "",
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool | llm.Toolset]] = NOT_GIVEN,
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

        self._caller_room: rtc.Room | None = None
        self._human_agent_sess: AgentSession | None = None
        self._human_agent_failed_fut: asyncio.Future[None] = asyncio.Future()
        self._human_agent_identity = "human-agent-sip"

        self._target_phone_number = target_phone_number
        self._sip_trunk_id = (
            sip_trunk_id if is_given(sip_trunk_id) else os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK", "")
        )
        if not self._sip_trunk_id:
            raise ValueError(
                "`LIVEKIT_SIP_OUTBOUND_TRUNK` environment variable or `sip_trunk_id` argument must be set"
            )

        self._sip_number = (
            sip_number if is_given(sip_number) else os.getenv("LIVEKIT_SIP_NUMBER", "")
        )

        # background audio and io
        self._background_audio = BackgroundAudioPlayer()
        self._hold_audio_handle: PlayHandle | None = None
        self._hold_audio = (
            cast(Optional[Union[AudioSource, AudioConfig, list[AudioConfig]]], hold_audio)
            if is_given(hold_audio)
            else AudioConfig(BuiltinAudioClip.HOLD_MUSIC, volume=0.8)
        )

        self._original_io_state: dict[str, bool] = {}

    def get_instructions(
        self, *, chat_ctx: NotGivenOr[llm.ChatContext], extra_instructions: str = ""
    ) -> str:
        # users can override this method if they want to customize the entire instructions
        prev_convo = ""
        if chat_ctx:
            for msg in chat_ctx.messages():
                if msg.role not in ("user", "assistant"):
                    continue
                if not msg.text_content:
                    continue
                role = "Caller" if msg.role == "user" else "Assistant"
                prev_convo += f"{role}: {msg.text_content}\n"
        return BASE_INSTRUCTIONS.format(conversation_history=prev_convo) + extra_instructions

    async def on_enter(self) -> None:
        job_ctx = get_job_context()
        self._caller_room = job_ctx.room

        # start the background audio
        if self._hold_audio is not None:
            await self._background_audio.start(room=self._caller_room)
            self._hold_audio_handle = self._background_audio.play(self._hold_audio, loop=True)

        self._set_io_enabled(False)

        try:
            dial_human_agent_task = asyncio.create_task(self._dial_human_agent())
            done, _ = await asyncio.wait(
                (dial_human_agent_task, self._human_agent_failed_fut),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if dial_human_agent_task not in done:
                raise RuntimeError()

            self._human_agent_sess = dial_human_agent_task.result()
            # let the human speak first

        except Exception:
            logger.exception("could not dial human agent")
            self._set_result(ToolError("could not dial human agent"))
            return

        finally:
            await utils.aio.cancel_and_wait(dial_human_agent_task)

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def connect_to_caller(self) -> None:
        """Called when the human agent wants to connect to the caller."""
        logger.debug("connecting to caller")
        assert self._caller_room is not None

        await self._merge_calls()
        self._set_result(WarmTransferResult(human_agent_identity=self._human_agent_identity))

        # when the caller or human agent leaves the room, we'll delete the room
        self._caller_room.on("participant_disconnected", self._on_caller_participant_disconnected)

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def decline_transfer(self, reason: str) -> None:
        """Handles the case when the human agent explicitly declines to connect to the caller.

        Args:
            reason: A short explanation of why the human agent declined to connect to the caller
        """
        self._set_result(ToolError(f"human agent declined to connect: {reason}"))

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def voicemail_detected(self) -> None:
        """Called when the call reaches voicemail. Use this tool AFTER you hear the voicemail greeting"""
        self._set_result(ToolError("voicemail detected"))

    def _on_human_agent_room_close(self, reason: rtc.DisconnectReason.ValueType) -> None:
        logger.debug(
            "human agent's room closed",
            extra={"reason": rtc.DisconnectReason.Name(reason)},
        )
        with contextlib.suppress(asyncio.InvalidStateError):
            self._human_agent_failed_fut.set_result(None)

        self._set_result(ToolError(f"room closed: {rtc.DisconnectReason.Name(reason)}"))

    def _on_caller_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.kind not in (
            rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
            rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
        ):
            return

        logger.info(f"participant disconnected from caller room: {participant.identity}, closing")

        assert self._caller_room is not None
        self._caller_room.off("participant_disconnected", self._on_caller_participant_disconnected)
        job_ctx = get_job_context()
        job_ctx.delete_room(room_name=self._caller_room.name)

    def _set_result(self, result: WarmTransferResult | Exception) -> None:
        if self.done():
            return

        if self._human_agent_sess:
            self._human_agent_sess.shutdown()
            self._human_agent_sess = None

        if self._hold_audio_handle:
            self._hold_audio_handle.stop()
            self._hold_audio_handle = None

        self._set_io_enabled(True)
        self.complete(result)

    async def _dial_human_agent(self) -> AgentSession:
        assert self._caller_room is not None

        job_ctx = get_job_context()
        ws_url = job_ctx._info.url

        # create a new room for the human agent
        human_agent_room_name = self._caller_room.name + "-human-agent"
        room = rtc.Room()
        token = (
            api.AccessToken()
            .with_identity(self._caller_room.local_participant.identity)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=human_agent_room_name,
                    can_update_own_metadata=True,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .with_kind("agent")
        ).to_jwt()

        logger.debug(
            "connecting to human agent room",
            extra={"ws_url": ws_url, "human_agent_room_name": human_agent_room_name},
        )
        await room.connect(ws_url, token)

        # if human agent hung up for whatever reason, we'd resume the caller conversation
        room.on("disconnected", self._on_human_agent_room_close)

        human_agent_sess: AgentSession = AgentSession(
            vad=self.session.vad or NOT_GIVEN,
            llm=self.session.llm or NOT_GIVEN,
            stt=self.session.stt or NOT_GIVEN,
            tts=self.session.tts or NOT_GIVEN,
            turn_detection=self.session.turn_detection or NOT_GIVEN,
        )
        # create a copy of this AgentTask
        human_agent_agent = Agent(
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
        await human_agent_sess.start(
            agent=human_agent_agent,
            room=room,
            room_options=room_io.RoomOptions(
                close_on_disconnect=True,
                delete_room_on_close=True,
                participant_identity=self._human_agent_identity,
            ),
            record=False,  # TODO: support recording on multiple sessions?
        )

        # dial the human agent
        await job_ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=self._sip_trunk_id,
                sip_call_to=self._target_phone_number,
                room_name=human_agent_room_name,
                participant_identity=self._human_agent_identity,
                wait_until_answered=True,
                sip_number=self._sip_number or None,
            )
        )

        return human_agent_sess

    async def _merge_calls(self) -> None:
        assert self._caller_room is not None
        assert self._human_agent_sess is not None

        job_ctx = get_job_context()
        human_agent_room = self._human_agent_sess.room_io.room
        # we no longer care about the human agent session. it's supposed to be over
        human_agent_room.off("disconnected", self._on_human_agent_room_close)

        logger.debug(f"moving {self._human_agent_identity} to caller room {self._caller_room.name}")
        await job_ctx.api.room.move_participant(
            api.MoveParticipantRequest(
                room=human_agent_room.name,
                identity=self._human_agent_identity,
                destination_room=self._caller_room.name,
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
