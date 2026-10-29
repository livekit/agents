"""Building blocks for voice flows that collect or act on what a caller says: the result
records, the checks a capture tool runs on what the model heard, keypad (DTMF) reception, and the
warm transfer to a human agent. The checks, the reception and the transfer are the core's."""

from __future__ import annotations

import os
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import AfterValidator

import livekit.rtc as rtc
from livekit import api

from .. import _core, llm, stt, tts, utils
from .._generated.vad import VAD
from .._generated.workflows import (
    GetAddressResult,
    GetCardNumberResult,
    GetCreditCardResult,
    GetDOBResult,
    GetEmailResult,
    GetExpirationDateResult,
    GetNameResult,
    GetPhoneNumberResult,
    GetSecurityCodeResult,
    WarmTransferResult,
    WorkflowInstructions as WorkflowInstructions,
)
from ..job import JobContext, get_job_context
from ..llm.tool_context import Completed, ToolError, ToolFlag, function_tool
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from ..voice import room_io
from ..voice.agent import Agent
from ..voice.agent_session import VoiceSession, VoiceTask, _live_binding
from ..voice.background_audio import (
    AudioConfig,
    AudioSource,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    PlayHandle,
)
from ..voice.events import RunContext
from ..voice.room_io import Room

if TYPE_CHECKING:
    from ..voice.turn import TurnHandlingOptions

__all__ = [
    "DtmfEvent",
    "GetAddressResult",
    "GetCardNumberResult",
    "GetCreditCardResult",
    "GetDOBResult",
    "GetDtmfResult",
    "GetEmailResult",
    "GetEmailTask",
    "GetExpirationDateResult",
    "GetNameResult",
    "GetPhoneNumberResult",
    "GetSecurityCodeResult",
    "TwilioConnectorWarmTransfer",
    "WarmTransfer",
    "WarmTransferResult",
    "WorkflowInstructions",
    "dtmf_event_to_code",
    "format_dtmf",
    "is_expired",
    "normalize_date_of_birth",
    "normalize_email_address",
    "normalize_phone_number",
    "receive_dtmf",
    "validate_card_number",
]

GetDtmfResult = _core.GetDtmfResult
_HOLD_MUSIC = AudioConfig(BuiltinAudioClip.HOLD_MUSIC, volume=0.8)
validate_card_number = _core.validate_card_number
normalize_email_address = _core.normalize_email_address
normalize_phone_number = _core.normalize_phone_number
dtmf_event_to_code = _core.dtmf_code
format_dtmf = _core.format_dtmf


def is_expired(month: int, year: int) -> bool:
    """Whether a two-digit card expiration year/month precedes this month."""
    return _core.card_expired(month, year, date.today())


def normalize_date_of_birth(year: int, month: int, day: int) -> date:
    """Normalize two-digit birth years and reject invalid or future dates."""
    return _core.normalize_date_of_birth(year, month, day, date.today())


class DtmfEvent(str, Enum):
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    ZERO = "0"
    STAR = "*"
    POUND = "#"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


async def receive_dtmf(
    room: rtc.Room,
    *,
    num_digits: int,
    timeout: float = 4.0,
    stop_event: DtmfEvent = DtmfEvent.POUND,
) -> _core.GetDtmfResult:
    """Receive keypad digits; stop at the delimiter or after an inter-digit timeout."""
    return await _core.receive_dtmf(
        _core._core_create_room_session(Room.of(room)), num_digits, timeout, stop_event.value
    )


def _email_address(email: str) -> str:
    try:
        return normalize_email_address(email)
    except ToolError as e:
        raise ValueError(e.message) from None


EmailAddress = Annotated[str, AfterValidator(_email_address)]
"""An email address as said, trimmed and checked."""

_email_collector = Agent(
    name="get_email",
    instructions="""\
You only capture the caller's email address, one step in a broader system.
Treat what you hear as a noisy transcription of an address said aloud: turn 'dot', \
'underscore', 'dash', 'plus' and 'at' into `.`, `_`, `-`, `+` and `@`, join a word spelled \
after it ('john j o h n' is john), and drop filler words. Never invent an address. When only \
the part before the '@' was said, ask for the domain; when it stays unclear, ask for it in \
parts. Call confirm_email_address with the address as soon as you have one; its read-back \
asks the caller to confirm, and a correction means calling it again with the corrected \
address. No greetings, no markdown, nothing off topic.""",
    output_type=GetEmailResult,
)


@_email_collector.on_enter
async def _ask_email(session: VoiceSession) -> None:
    await session.generate_reply(
        instructions="Ask for their email address, or confirm one they already gave."
    )


async def _read_back_email(ctx: RunContext, email_address: EmailAddress) -> str:
    return f"Just to confirm, your email is {email_address}, right?"


@_email_collector.tool(name="confirm_email_address", needs_approval=_read_back_email)
async def _confirm_email_address(
    ctx: RunContext, email_address: EmailAddress
) -> Completed[GetEmailResult]:
    """Record the caller's email address; they confirm it before it is kept.

    Args:
        email_address: the address the caller gave, in standard form
    """
    return Completed(GetEmailResult(email_address=email_address))


class GetEmailTask(VoiceTask[GetEmailResult]):
    """Collect and confirm the caller's email address on the call (docs/voice-tasks.md)::

    result = await GetEmailTask().run()
    """

    def __init__(self) -> None:
        super().__init__(agent=_email_collector)


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
        """Initialize a WarmTransfer to dial a human agent via SIP.

        Args:
            sip_call_to: The phone number or SIP URI to dial for the human agent
                (e.g. ``"+15105550123"`` or ``"sip:user@example.com"``).
            sip_trunk_id: ID of a pre-configured LiveKit SIP outbound trunk used to
                originate the call. Falls back to the ``LIVEKIT_SIP_OUTBOUND_TRUNK``
                environment variable when not provided.
            sip_connection: Low-level SIP connection config (``api.SIPOutboundConfig``)
                for originating calls from a **custom SIP domain** instead of through a
                saved trunk. Use this when you need to specify a custom hostname,
                transport, or authentication credentials directly, bypassing the
                trunk-based configuration.
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
        self._sip_connection = sip_connection
        self._hold_audio = hold_audio
        self._transfer = _core.WarmTransfer(
            sip_call_to=sip_call_to,
            sip_trunk_id=sip_trunk_id,
            has_sip_connection=sip_connection is not None,
            sip_number=sip_number,
            sip_headers=sip_headers,
            dtmf=dtmf,
            ringing_timeout=ringing_timeout,
            instructions=instructions,
            conversation=_conversation(chat_ctx),
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
            _core._core_create_room_session(Room.of(job_ctx.room)),
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

        Note:
            If Twilio rejects the preserved caller ID (HTTP 400 with error 21210 or
            21212), the call is retried once from ``twilio_from_number`` without the
            token. Other failures propagate without a retry to avoid duplicate calls.
            Other arguments have the same meaning as in ``WarmTransfer``.
        """
        self._sip_connection = None
        self._hold_audio = hold_audio
        self._transfer = _core.WarmTransfer(
            ringing_timeout=ringing_timeout,
            instructions=instructions,
            conversation=_conversation(chat_ctx),
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
    """The agent that briefs the human agent: the transfer's instructions and tools."""
    return Agent(
        instructions=transfer.instructions,
        tools=[*(tools or []), *(_transfer_tool(transfer, spec) for spec in transfer.tools())],
        llm=llm,
    )


def _given(**options: Any) -> dict[str, Any]:
    return {k: v for k, v in options.items() if is_given(v)}


def _conversation(chat_ctx: llm.ChatContext | None) -> list[tuple[str, str]] | None:
    if chat_ctx is None:
        return None
    return [(m.role, m.text_content or "") for m in chat_ctx.messages()]


def _transfer_tool(transfer: _core.WarmTransfer, spec: dict[str, Any]) -> llm.RawFunctionTool:
    async def call(raw_arguments: dict[str, object]) -> None:
        await transfer.call_tool(spec["name"], raw_arguments)

    return function_tool(call, raw_schema=spec, flags=ToolFlag.IGNORE_ON_ENTER)


class _TransferHost:
    """What the core's transfer asks of the SDK: the rooms, the human agent's session, the hold
    music and the API requests. Each ``_core_<method>`` is one call or two."""

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

    def _core_stop_hold(self, hold: PlayHandle) -> None:
        hold.stop()

    def _core_io_state(self) -> dict[str, bool]:
        return {
            "audio_input": self._session.input.audio_enabled,
            "audio_output": self._session.output.audio_enabled,
            "transcription_output": self._session.output.transcription_enabled,
        }

    def _core_set_io(self, state: dict[str, bool]) -> None:
        self._session.input.set_audio_enabled(state["audio_input"])
        self._session.output.set_audio_enabled(state["audio_output"])
        self._session.output.set_transcription_enabled(state["transcription_output"])

    async def _core_connect_human(self, room_name: str, identity: str) -> _core._core_RoomSession:
        token = (
            api.AccessToken()
            .with_identity(identity)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_update_own_metadata=True,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .with_kind("agent")
            .to_jwt()
        )
        await self._room.connect(self._job._info.url, token)
        return _core._core_create_room_session(Room.of(self._room))

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

    def _core_close_human(self, drain: bool) -> None:
        self._human.shutdown(drain=drain)

    async def _core_disconnect_human(self) -> None:
        await self._room.disconnect()

    async def _core_create_sip_participant(self, request: dict[str, Any]) -> None:
        ringing_timeout = request.pop("ringing_timeout")
        sip = api.CreateSIPParticipantRequest(**request)
        if ringing_timeout is not None:
            sip.ringing_timeout.FromNanoseconds(int(ringing_timeout * 1e9))
        if self._transfer._sip_connection is not None:
            sip.trunk.CopyFrom(self._transfer._sip_connection)
        await self._job.api.sip.create_sip_participant(sip)

    async def _core_connect_twilio_call(self, room_name: str, identity: str) -> str:
        response = await self._job.api.connector.connect_twilio_call(
            api.ConnectTwilioCallRequest(
                twilio_call_direction=api.ConnectTwilioCallRequest.TwilioCallDirection.TWILIO_CALL_DIRECTION_OUTBOUND,
                room_name=room_name,
                participant_identity=identity,
            )
        )
        return response.connect_url

    async def _core_http_post(
        self, url: str, headers: dict[str, str], form: list[tuple[str, str]]
    ) -> tuple[int, str]:
        async with utils.http_context.http_session().post(
            url, headers=headers, data=form, allow_redirects=False
        ) as response:
            return response.status, await response.text()

    async def _core_move_participant(self, room: str, identity: str, destination: str) -> None:
        await self._job.api.room.move_participant(
            api.MoveParticipantRequest(room=room, identity=identity, destination_room=destination)
        )

    async def _core_delete_caller_room(self) -> None:
        await self._job.delete_room(room_name=self._job.room.name)
