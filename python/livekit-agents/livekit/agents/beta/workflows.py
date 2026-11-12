from __future__ import annotations

import os
from datetime import date
from typing import TYPE_CHECKING, Annotated, Any, cast

from pydantic import AfterValidator

import livekit.rtc as rtc
from livekit import api

from .. import _core, llm, stt, tts
from .._generated import public_beta_workflows as _public
from .._generated.public_beta_workflows import *  # noqa: F403
from .._generated.vad import VAD
from .._generated.voice_session import DtmfTaskOptions
from .._generated.workflow_prompts import (
    ASK_DIGITS_INSTRUCTIONS,
    ASK_EMAIL_INSTRUCTIONS,
)
from .._generated.workflows import IoEnableState, WorkflowInstructions as WorkflowInstructions
from ..job import JobContext, _server_credentials, get_job_context
from ..llm.tool_context import Completed, ToolError, ToolFlag, function_tool
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from ..voice import Agent, room_io
from ..voice.agent_session import VoiceSession, VoiceTask, _live_binding
from ..voice.background_audio import (
    AudioConfig,
    AudioSource,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    PlayHandle,
)
from ..voice.events import RunContext

if TYPE_CHECKING:
    from ..voice.io import TurnHandlingOptions

__all__ = [
    "GetDtmfTask",
    "GetEmailTask",
    "TwilioConnectorWarmTransfer",
    "WarmTransfer",
    "WorkflowInstructions",
    "is_expired",
    "normalize_date_of_birth",
    "receive_dtmf",
]
__all__ += _public.__all__


_HOLD_MUSIC = AudioConfig(BuiltinAudioClip.HOLD_MUSIC, volume=0.8)


def is_expired(month: int, year: int) -> bool:
    """Whether a two-digit card expiration year/month precedes this month."""
    return _core.card_expired(month, year, date.today())


def normalize_date_of_birth(year: int, month: int, day: int) -> date:
    """Normalize two-digit birth years and reject invalid or future dates."""
    return _core.normalize_date_of_birth(year, month, day, date.today())


async def receive_dtmf(
    room: rtc.Room,
    *,
    num_digits: int,
    timeout: float = 4.0,
    stop_event: DtmfEvent = DtmfEvent.POUND,
) -> GetDtmfResult:
    """Receive keypad digits; stop at the delimiter or after an inter-digit timeout."""
    digits = await _core.receive_dtmf(
        _core._core_create_room_session(room), num_digits, timeout, stop_event
    )
    return GetDtmfResult(user_input=digits)


async def _record_inputs(ctx: RunContext, inputs: list[DtmfEvent]) -> Completed[GetDtmfResult]:
    """Record the collected digit inputs without additional confirmation.

    Call this tool as soon as a valid sequence of digits has been provided by the user (via
    DTMF or spoken)."""
    return Completed(GetDtmfResult(user_input=format_dtmf([e.value for e in inputs])))


async def _ask_digits(session: VoiceSession) -> None:
    session.generate_reply(instructions=ASK_DIGITS_INSTRUCTIONS, tool_choice="none")


class GetDtmfTask(VoiceTask[GetDtmfResult]):
    """Collect digits the caller presses on the keypad, or says::

        result = await GetDtmfTask(num_digits=4).run()

    The keys are collected until ``num_digits`` arrived and ``dtmf_input_timeout`` seconds passed
    without another, or ``dtmf_stop_event`` was pressed. Too few keys fail the task
    (``VoiceTaskFailed``). With ``ask_for_confirmation`` the digits are read back and the caller
    confirms them first."""

    def __init__(
        self,
        *,
        num_digits: int,
        ask_for_confirmation: bool = False,
        dtmf_input_timeout: float = 4.0,
        dtmf_stop_event: DtmfEvent = DtmfEvent.POUND,
    ) -> None:
        plan = _core.dtmf_capture_plan(ask_for_confirmation)
        agent = Agent(
            name=plan.agent_name, instructions=plan.instructions, output_type=GetDtmfResult
        )
        agent.tool(name=plan.tool_name, description=plan.tool_description)(_record_inputs)
        agent.on_enter(_ask_digits)
        super().__init__(agent=agent)
        self._dtmf = DtmfTaskOptions(
            num_digits=num_digits,
            ask_for_confirmation=ask_for_confirmation,
            timeout=dtmf_input_timeout,
            stop_event=dtmf_stop_event,
        )

    async def run(self) -> GetDtmfResult:
        binding = _live_binding("GetDtmfTask.run()")
        session, agent = binding.session, self._agent
        value = await session._running.run_voice_task(
            session._frame(agent, task=True),
            origin=binding.origin,
            opens_with_reply=not agent._has_on_enter(),
            dtmf=self._dtmf,
        )
        return (
            GetDtmfResult(user_input=value)
            if isinstance(value, str)
            else cast(GetDtmfResult, value)
        )


def _email_address(email: str) -> str:
    try:
        return normalize_email_address(email)
    except ToolError as e:
        raise ValueError(e.message) from None


EmailAddress = Annotated[str, AfterValidator(_email_address)]
"""An email address as said, trimmed and checked."""


async def _ask_email(session: VoiceSession) -> None:
    await session.generate_reply(instructions=ASK_EMAIL_INSTRUCTIONS)


async def _read_back_email(ctx: RunContext, email_address: EmailAddress) -> str:
    return _core.email_read_back(email_address)


async def _confirm_email_address(
    ctx: RunContext, email_address: EmailAddress
) -> Completed[GetEmailResult]:
    """Record the caller's email address; they confirm it before it is kept.

    Args:
        email_address: the address the caller gave, in standard form
    """
    return Completed(GetEmailResult(email_address=email_address))


class GetEmailTask(VoiceTask[GetEmailResult]):
    """Collect and confirm the caller's email address on the call::

    result = await GetEmailTask().run()
    """

    def __init__(self) -> None:
        plan = _core.email_capture_plan()
        agent = Agent(
            name=plan.agent_name, instructions=plan.instructions, output_type=GetEmailResult
        )
        agent.tool(
            name=plan.tool_name, description=plan.tool_description, needs_approval=_read_back_email
        )(_confirm_email_address)
        agent.on_enter(_ask_email)
        super().__init__(agent=agent)


class WarmTransfer:
    """Hand the caller over to a human agent reached over SIP.

    The caller waits on hold, their session's audio off, while a second session in a room of its
    own dials the human agent and briefs them on the conversation so far. The human agent joins
    the caller's room once they agree (``connect_to_caller``); ``run`` raises ``ToolError`` when
    they decline, the call reaches voicemail or their room closes first."""

    def __init__(
        self,
        sip_call_to: str,
        *,
        sip_trunk_id: NotGivenOr[str | None] = NOT_GIVEN,
        sip_connection: api.SIPOutboundConfig | None = None,
        sip_number: NotGivenOr[str] = NOT_GIVEN,
        sip_headers: dict[str, str] | None = None,
        dtmf: str | None = None,
        ringing_timeout: float | None = None,
        hold_audio: AudioSource | AudioConfig | list[AudioConfig] | None = _HOLD_MUSIC,
        instructions: NotGivenOr[WorkflowInstructions | str] = NOT_GIVEN,
        chat_ctx: llm.ChatContext | None = None,
        turn_handling: NotGivenOr[TurnHandlingOptions] = NOT_GIVEN,
        tools: list[llm.Tool | llm.Toolset] | None = None,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            sip_call_to: The phone number or SIP URI to dial for the human agent
                (e.g. ``"+15105550123"`` or ``"sip:user@example.com"``).
            sip_trunk_id: The LiveKit SIP outbound trunk that originates the call; defaults to
                ``LIVEKIT_SIP_OUTBOUND_TRUNK``.
            sip_connection: Originate from a custom SIP domain (hostname, transport,
                credentials) instead of a saved trunk.
            sip_number: The caller ID to dial from; defaults to ``LIVEKIT_SIP_NUMBER``.
            dtmf: DTMF tones to send once the human agent's call is answered, e.g. to dial
                an extension or navigate an IVR menu (``"1234#"``). Insert ``w`` characters
                to pause ~0.5s each before/between digits (``"wwww1234#"`` waits ~2s, useful
                when the destination plays a greeting before accepting input).
            ringing_timeout: How long to wait, in seconds, for the human agent to answer
                before giving up on the call.
            hold_audio: Audio played to the caller while they are on hold during the
                transfer.
            instructions: The briefing agent's instructions; a ``WorkflowInstructions``
                overrides the persona and extra sections of the built-in briefing.
            chat_ctx: The caller conversation the briefing summarises.
            tools: More tools for the briefing agent.
        """
        self._hold_audio = hold_audio
        self._transfer = _core.WarmTransfer(
            sip_call_to=sip_call_to,
            sip_trunk_id=sip_trunk_id,
            sip_connection=sip_connection.SerializeToString() if sip_connection else None,
            sip_number=sip_number,
            sip_headers=sip_headers,
            dtmf=dtmf,
            ringing_timeout=ringing_timeout,
            instructions=instructions,
            conversation=chat_ctx.items if chat_ctx is not None else None,
            env_sip_trunk=os.environ.get("LIVEKIT_SIP_OUTBOUND_TRUNK"),
            env_sip_number=os.environ.get("LIVEKIT_SIP_NUMBER", ""),
        )
        self._agent = _briefing_agent(self._transfer, tools=tools, llm=llm)
        self._briefing_audio = _given(stt=stt, vad=vad, tts=tts, turn_handling=turn_handling)

    async def run(self) -> WarmTransferResult:
        """Dial and bridge a human participant. Await it inside a tool of the agent holding the
        call; the session is the one that tool runs in."""
        session = _live_binding("WarmTransfer.run()").session
        job_ctx = get_job_context()
        identity = await self._transfer.run(
            _TransferHost(self, session, job_ctx),
            _core._core_create_room_session(job_ctx.room),
        )
        return WarmTransferResult(human_agent_identity=identity)


class TwilioConnectorWarmTransfer(WarmTransfer):
    def __init__(
        self,
        phone_number: str,
        *,
        twilio_from_number: str,
        twilio_account_sid: NotGivenOr[str] = NOT_GIVEN,
        twilio_auth_token: NotGivenOr[str] = NOT_GIVEN,
        twilio_call_token: str | None = None,
        original_caller_number: str | None = None,
        ringing_timeout: float | None = _core.TWILIO_RINGING_TIMEOUT,
        hold_audio: AudioSource | AudioConfig | list[AudioConfig] | None = _HOLD_MUSIC,
        instructions: NotGivenOr[WorkflowInstructions | str] = NOT_GIVEN,
        chat_ctx: llm.ChatContext | None = None,
        turn_handling: NotGivenOr[TurnHandlingOptions] = NOT_GIVEN,
        tools: list[llm.Tool | llm.Toolset] | None = None,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
    ) -> None:
        """Dial a human agent using Twilio's Calls API and the LiveKit Twilio Connector.

        Arguments not listed have the same meaning as in ``WarmTransfer``.

        Args:
            phone_number: The human agent's phone number to dial.
            twilio_from_number: Agent/business Twilio number or verified caller ID.
                Used when no token is supplied and for the caller-ID rejection fallback.
            twilio_account_sid: Twilio account SID. Defaults to ``TWILIO_ACCOUNT_SID``.
            twilio_auth_token: Twilio auth token. Defaults to ``TWILIO_AUTH_TOKEN``.
            twilio_call_token: Nonempty ``CallToken`` from the inbound call's validated
                webhook, letting Twilio place the outbound call from the original
                caller's number. Requires ``original_caller_number``.
                Keep it in server-side state for that call, outside prompts, logs and
                participant attributes.
            original_caller_number: The inbound call's ``From``. Used as the outbound
                caller ID only together with a nonempty ``twilio_call_token``; otherwise
                ``twilio_from_number`` is used.
            ringing_timeout: How long to wait for the human agent to answer; Twilio reports
                no answer only to status webhooks, so 30 seconds unless set.

        If Twilio rejects the preserved caller ID (HTTP 400, error 21210 or 21212), the call is
        retried once from ``twilio_from_number`` without the token. Other failures are not
        retried, to avoid duplicate calls.
        """
        self._hold_audio = hold_audio
        self._transfer = _core.WarmTransfer(
            ringing_timeout=ringing_timeout,
            instructions=instructions,
            conversation=chat_ctx.items if chat_ctx is not None else None,
            twilio={
                "phone_number": phone_number,
                "from_number": twilio_from_number,
                "account_sid": twilio_account_sid
                if is_given(twilio_account_sid)
                else os.environ.get("TWILIO_ACCOUNT_SID", ""),
                "auth_token": twilio_auth_token
                if is_given(twilio_auth_token)
                else os.environ.get("TWILIO_AUTH_TOKEN", ""),
                "call_token": twilio_call_token,
                "original_caller_number": original_caller_number,
            },
        )
        self._agent = _briefing_agent(self._transfer, tools=tools, llm=llm)
        self._briefing_audio = _given(stt=stt, vad=vad, tts=tts, turn_handling=turn_handling)


def _briefing_agent(
    transfer: _core.WarmTransfer,
    *,
    tools: list[llm.Tool | llm.Toolset] | None,
    llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None],
) -> Agent:
    return Agent(
        instructions=transfer.instructions,
        tools=[*(tools or []), *(_transfer_tool(transfer, spec) for spec in transfer.tools())],
        llm=llm,
    )


def _given(**options: Any) -> dict[str, Any]:
    return {k: v for k, v in options.items() if is_given(v)}


def _transfer_tool(transfer: _core.WarmTransfer, spec: dict[str, Any]) -> llm.RawFunctionTool:
    async def call(raw_arguments: dict[str, object]) -> None:
        await transfer.call_tool(spec["name"], raw_arguments)

    return function_tool(call, raw_schema=spec, flags=ToolFlag.IGNORE_ON_ENTER)


class _TransferHost:
    def __init__(self, transfer: WarmTransfer, session: VoiceSession, job_ctx: JobContext) -> None:
        self._transfer = transfer
        self._session = session
        self._job = job_ctx
        self._room = rtc.Room()
        self._player = BackgroundAudioPlayer()
        # the briefing session's audio: the transfer's own, else the caller session's
        self._human: VoiceSession = VoiceSession(
            **{
                "vad": session.vad or NOT_GIVEN,
                "llm": session.llm or NOT_GIVEN,
                "stt": session.stt or NOT_GIVEN,
                "tts": session.tts or NOT_GIVEN,
                **transfer._briefing_audio,
            }
        )

    async def _core_start_hold(self) -> PlayHandle | None:
        if self._transfer._hold_audio is None:
            return None
        await self._player.start(room=self._job.room)
        return self._player.play(self._transfer._hold_audio, loop=True)

    def _core_io_state(self) -> IoEnableState:
        return {
            "audio_input": self._session.input.audio_enabled,
            "audio_output": self._session.output.audio_enabled,
            "transcription_output": self._session.output.transcription_enabled,
        }

    def _core_set_io(self, state: IoEnableState) -> None:
        self._session.input.set_audio_enabled(state.get("audio_input", True))
        self._session.output.set_audio_enabled(state.get("audio_output", True))
        self._session.output.set_transcription_enabled(state.get("transcription_output", True))

    async def _core_connect_human(self, room_name: str, identity: str) -> _core._core_RoomSession:
        _, api_key, api_secret = _server_credentials()
        token = _core.room_join_token(api_key, api_secret, identity, room_name, agent=True)
        await self._room.connect(self._job._info.url, token)
        return _core._core_create_room_session(self._room)

    async def _core_start_human(self) -> None:
        await self._human.start(
            agent=self._transfer._agent,
            chat_ctx=self._session.history.copy(),
            room=self._room,
            room_options=room_io.RoomOptions(
                close_on_disconnect=True,
                delete_room_on_close=True,
                participant_identity=_core.WARM_TRANSFER_IDENTITY,
            ),
        )
