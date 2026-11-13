from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from livekit import rtc

from ...job import get_job_context
from ...llm import ToolError, ToolFlag, ToolResult, Toolset, function_tool
from ...log import logger
from ...voice.events import RunContext

# Backward compatibility alias
ai_callable = function_tool

DEFAULT_DTMF_PUBLISH_DELAY: float = 0.3
DEFAULT_DTMF_TOPIC: str = "dtmf"

STANDARD_DTMF_DIGITS: frozenset[str] = frozenset("0123456789*#")
EXTENDED_DTMF_DIGITS: frozenset[str] = frozenset("ABCD")
VALID_DTMF_DIGITS: frozenset[str] = STANDARD_DTMF_DIGITS | EXTENDED_DTMF_DIGITS

DTMF_CODE_MAP: dict[str, int] = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "*": 10,
    "#": 11,
    "A": 12,
    "B": 13,
    "C": 14,
    "D": 15,
}

E164_REGEX = re.compile(r"^\+[1-9]\d{1,14}$")
PBX_EXTENSION_REGEX = re.compile(r"^\d{3,7}$")
SIP_URI_REGEX = re.compile(r"^sips?:[^@\s]+@[^@\s]+$", re.IGNORECASE)

DTMF_KEYPAD_DESCRIPTION = """
Sends DTMF (Dual-Tone Multi-Frequency) touch tones to navigate automated phone systems (IVR) or dial extensions.

Call when:
- The user or remote system prompts for touch tone inputs (e.g. "press 1 for sales", "enter your 4-digit PIN").
- Navigating interactive phone menus or entering numbers into an IVR keypad.

Do not call when:
- Normal verbal conversation is requested.
"""

CALL_TRANSFER_DESCRIPTION = """
Transfers the active SIP phone call to another phone number or SIP URI (blind / cold transfer).

Call when:
- The user requests to speak to another department, person, or external agent (e.g. "transfer me to support", "connect me to 555-0199").
- The call needs to be escalated or routed to a secondary destination.

Do not call when:
- The caller wants to remain on hold or requests information the current agent can provide.
- Intent to transfer is unclear.
"""


def dtmf_char_to_code(char: str) -> int:
    """Convert a single DTMF character to its RFC 4733 integer code."""
    if char in DTMF_CODE_MAP:
        return DTMF_CODE_MAP[char]
    raise ToolError(f"Invalid DTMF digit: '{char}'. Valid digits are 0-9, *, #, and uppercase A-D.")


def validate_dtmf_digits(digits: str) -> None:
    """Validate that a string contains only valid RFC 4733 DTMF digits.

    Valid digits are 0-9, *, #, and uppercase A-D.

    Raises:
        ToolError: If digits is empty or contains any invalid character.
    """
    if not digits:
        raise ToolError("DTMF digits string cannot be empty.")

    for ch in digits:
        if ch not in VALID_DTMF_DIGITS:
            raise ToolError(
                f"Invalid DTMF digit '{ch}' in '{digits}'. "
                f"Valid DTMF digits are 0-9, *, #, and uppercase A-D."
            )


def validate_transfer_destination(destination: str) -> str:
    """Validate and normalize a call transfer destination.

    Supports:
    - E.164 phone numbers (e.g. '+15551234567' or formatted '+1 (555) 123-4567')
    - PBX extensions (3 to 7 digits, e.g. '1001')
    - SIP URIs (e.g. 'sip:agent@trunk.example.com')

    Returns:
        The normalized destination string.

    Raises:
        ToolError: If destination is empty or invalid.
    """
    cleaned = destination.strip()
    if not cleaned:
        raise ToolError("Transfer destination cannot be empty.")

    # 1. Check SIP URI (RFC 3261)
    if cleaned.lower().startswith(("sip:", "sips:")):
        if SIP_URI_REGEX.match(cleaned):
            return cleaned
        raise ToolError(
            f"Invalid SIP URI: '{destination}'. Expected format 'sip:user@host' or 'sips:user@host'."
        )

    # 2. Check PBX Extension (3-7 digits)
    if PBX_EXTENSION_REGEX.match(cleaned):
        return cleaned

    # 3. Check E.164 phone number (allow stripping spaces, hyphens, parentheses, periods)
    normalized = re.sub(r"[\s\-\(\)\.]", "", cleaned)
    if E164_REGEX.match(normalized):
        return normalized

    raise ToolError(
        f"Invalid transfer destination: '{destination}'. Must be an E.164 phone number "
        f"(e.g. +15551234567), a 3-7 digit PBX extension (e.g. 1001), or a SIP URI (sip:user@host)."
    )


class DTMFKeypadTool(Toolset):
    """Telephony toolset for emitting and intercepting RFC 4733 DTMF touch tones."""

    def __init__(
        self,
        *,
        publish_delay: float = DEFAULT_DTMF_PUBLISH_DELAY,
        enable_data_packets: bool = True,
        data_packet_topic: str = DEFAULT_DTMF_TOPIC,
        room: rtc.Room | None = None,
        extra_description: str = "",
        ignore_on_enter: bool = False,
        on_tool_called: Callable[[Toolset.ToolCalledEvent], Awaitable[None]] | None = None,
        on_tool_completed: Callable[[Toolset.ToolCompletedEvent], Awaitable[None]] | None = None,
        on_dtmf_received: Callable[[rtc.SipDTMF], Any] | None = None,
    ) -> None:
        """
        Args:
            publish_delay: Inter-digit delay in seconds between consecutive DTMF tones. Defaults to 0.3s.
            enable_data_packets: Whether to publish WebRTC data packets on the topic alongside tones. Defaults to True.
            data_packet_topic: Topic name for WebRTC data packets. Defaults to "dtmf".
            room: Optional explicit Room instance. If None, resolved from RunContext or JobContext.
            extra_description: Additional instructions appended to the tool description for LLM prompting.
            ignore_on_enter: Hide the tool during ``on_enter`` so the model doesn't emit DTMF tones while greeting.
            on_tool_called: Callback called when the tool is invoked.
            on_tool_completed: Callback called when the tool completes.
            on_dtmf_received: Optional callback invoked when inbound SIP DTMF tones are received.
        """
        self._publish_delay = publish_delay
        self._enable_data_packets = enable_data_packets
        self._data_packet_topic = data_packet_topic
        self._room = room
        self._on_tool_called = on_tool_called
        self._on_tool_completed = on_tool_completed
        self._on_dtmf_received = on_dtmf_received
        self._subscribed_room: rtc.Room | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

        desc = f"{DTMF_KEYPAD_DESCRIPTION.strip()}\n{extra_description}".strip()
        dtmf_tool = function_tool(
            self.send_dtmf,
            name="send_dtmf",
            description=desc,
            flags=ToolFlag.IGNORE_ON_ENTER if ignore_on_enter else ToolFlag.NONE,
        )
        super().__init__(id="dtmf_keypad", tools=[dtmf_tool])

        if room is not None and on_dtmf_received is not None:
            self.listen(room)

    def _resolve_room(self, ctx: RunContext) -> rtc.Room:
        if self._room is not None:
            return self._room

        try:
            if hasattr(ctx, "session") and hasattr(ctx.session, "room_io") and ctx.session.room_io:
                room = ctx.session.room_io.room
                if room is not None:
                    return room
        except (AttributeError, RuntimeError):
            pass

        try:
            job_ctx = get_job_context()
            if job_ctx and job_ctx.room:
                return job_ctx.room
        except Exception:
            pass

        raise ToolError("No active Room connection found for DTMF emission.")

    async def send_dtmf(self, ctx: RunContext, digits: str) -> ToolResult:
        """Send DTMF touch tones to navigate an IVR or dial an extension.

        Args:
            digits: Sequence of DTMF touch tones to emit (0-9, *, #, A-D). For example: "1", "1234#", or "*67".
        """
        logger.debug(f"send_dtmf called with digits='{digits}'")
        validate_dtmf_digits(digits)

        room = self._resolve_room(ctx)
        if room.local_participant is None:
            raise ToolError("Room has no local participant connected.")

        if self._on_tool_called:
            await self._on_tool_called(
                Toolset.ToolCalledEvent(ctx=ctx, arguments={"digits": digits})
            )

        for digit in digits:
            code = dtmf_char_to_code(digit)
            try:
                await room.local_participant.publish_dtmf(code=code, digit=digit)
            except Exception as e:
                raise ToolError(f"Failed to publish DTMF digit '{digit}': {e}") from e

            if self._enable_data_packets:
                try:
                    payload = json.dumps({"event": "dtmf", "digit": digit, "code": code}).encode(
                        "utf-8"
                    )
                    await room.local_participant.publish_data(
                        payload,
                        topic=self._data_packet_topic,
                        reliable=True,
                    )
                except Exception as e:
                    logger.warning(f"Failed to publish DTMF data packet: {e}")

            if self._publish_delay > 0:
                await asyncio.sleep(self._publish_delay)

        output = f"Successfully sent DTMF digits: {digits}"
        if self._on_tool_completed:
            await self._on_tool_completed(Toolset.ToolCompletedEvent(ctx=ctx, output=output))

        return ToolResult(output=output, reply_required=False)

    def listen(self, room: rtc.Room) -> None:
        """Begin listening for inbound SIP DTMF events on the provided room."""
        if self._subscribed_room is not None:
            self.stop_listening()
        self._subscribed_room = room
        room.on("sip_dtmf_received", self._handle_sip_dtmf)

    def stop_listening(self) -> None:
        """Stop listening for inbound SIP DTMF events."""
        if self._subscribed_room is not None:
            self._subscribed_room.off("sip_dtmf_received", self._handle_sip_dtmf)
            self._subscribed_room = None

    def _handle_sip_dtmf(self, ev: rtc.SipDTMF) -> None:
        if self._on_dtmf_received:
            res = self._on_dtmf_received(ev)
            if asyncio.iscoroutine(res):
                task = asyncio.create_task(res)
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

    def register_listener(
        self,
        room: rtc.Room,
        callback: Callable[[rtc.SipDTMF], Any] | None = None,
    ) -> None:
        """Register a callback to intercept incoming DTMF events from SIP."""
        if callback is not None:
            self._on_dtmf_received = callback
        self.listen(room)

    def unregister_listener(self, room: rtc.Room | None = None) -> None:
        """Unregister DTMF interception listeners from the room."""
        self.stop_listening()

    async def aclose(self) -> None:
        self.stop_listening()
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
        await super().aclose()


class CallTransferTool(Toolset):
    """Telephony toolset for initiating cold/blind call transfers over SIP."""

    def __init__(
        self,
        *,
        play_dialtone: bool = False,
        extra_description: str = "",
        ignore_on_enter: bool = False,
        on_tool_called: Callable[[Toolset.ToolCalledEvent], Awaitable[None]] | None = None,
        on_tool_completed: Callable[[Toolset.ToolCompletedEvent], Awaitable[None]] | None = None,
    ) -> None:
        """
        Args:
            play_dialtone: Whether to play a dialtone to the caller during the transfer. Defaults to False.
            extra_description: Additional instructions appended to the LLM tool description.
            ignore_on_enter: Hide the tool during ``on_enter`` so the model doesn't transfer while greeting.
            on_tool_called: Callback called when the tool is invoked.
            on_tool_completed: Callback called when the tool completes.
        """
        self._play_dialtone = play_dialtone
        self._on_tool_called = on_tool_called
        self._on_tool_completed = on_tool_completed

        desc = f"{CALL_TRANSFER_DESCRIPTION.strip()}\n{extra_description}".strip()
        transfer_tool = function_tool(
            self.transfer_call,
            name="transfer_call",
            description=desc,
            flags=ToolFlag.IGNORE_ON_ENTER if ignore_on_enter else ToolFlag.NONE,
        )
        super().__init__(id="call_transfer", tools=[transfer_tool])

    async def transfer_call(
        self,
        ctx: RunContext,
        transfer_to: str,
        play_dialtone: bool | None = None,
    ) -> ToolResult:
        """Transfer the active SIP phone call to a secondary phone number, PBX extension, or SIP URI.

        Args:
            transfer_to: Destination phone number (E.164 e.g. "+15551234567"), PBX extension (3-7 digits), or SIP URI (e.g. "sip:agent@trunk.example.com").
            play_dialtone: Optional override for playing a dialtone during the transfer. Defaults to None (uses configured tool default).
        """
        logger.debug(f"transfer_call called with transfer_to='{transfer_to}'")
        clean_destination = validate_transfer_destination(transfer_to)
        dialtone = play_dialtone if play_dialtone is not None else self._play_dialtone

        # 1. Resolve active SIP participant
        sip_participant: rtc.RemoteParticipant | None = None

        try:
            if hasattr(ctx, "session") and hasattr(ctx.session, "room_io") and ctx.session.room_io:
                linked = ctx.session.room_io.linked_participant
                if (
                    linked is not None
                    and getattr(linked, "kind", None) == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                ):
                    sip_participant = linked
        except (AttributeError, RuntimeError):
            pass

        if sip_participant is None:
            room: rtc.Room | None = None
            try:
                if hasattr(ctx, "session") and hasattr(ctx.session, "room_io"):
                    room = ctx.session.room_io.room
            except (AttributeError, RuntimeError):
                pass

            if room is None:
                try:
                    fallback_job = get_job_context(required=False)
                    if fallback_job:
                        room = fallback_job.room
                except Exception:
                    pass

            if room and hasattr(room, "remote_participants") and room.remote_participants:
                for p in room.remote_participants.values():
                    if getattr(p, "kind", None) == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                        sip_participant = p
                        break

        if sip_participant is None:
            raise ToolError("No active SIP participant found to transfer.")

        # 2. Execute transfer via JobContext
        job_ctx = get_job_context(required=False)

        if job_ctx is None:
            raise ToolError("No active JobContext found to execute SIP participant transfer.")

        if self._on_tool_called:
            await self._on_tool_called(
                Toolset.ToolCalledEvent(
                    ctx=ctx,
                    arguments={"transfer_to": clean_destination, "play_dialtone": dialtone},
                )
            )

        try:
            transfer_fut = job_ctx.transfer_sip_participant(
                sip_participant,
                transfer_to=clean_destination,
                play_dialtone=dialtone,
            )
            if asyncio.isfuture(transfer_fut) or asyncio.iscoroutine(transfer_fut):
                await transfer_fut
        except ToolError:
            raise
        except Exception as e:
            logger.error(f"Failed to transfer SIP participant {sip_participant.identity}: {e}")
            raise ToolError(f"Call transfer failed: {e}") from e

        output = f"Successfully initiated call transfer to {clean_destination}"
        if self._on_tool_completed:
            await self._on_tool_completed(Toolset.ToolCompletedEvent(ctx=ctx, output=output))

        return ToolResult(output=output, reply_required=False)


def on_sip_dtmf(
    room: rtc.Room,
    callback: Callable[[rtc.SipDTMF], Any],
    *,
    participant: rtc.RemoteParticipant | str | None = None,
) -> Callable[[], None]:
    """Register a hook listening for inbound SIP DTMF events with optional participant filtering.

    Args:
        room: The LiveKit Room instance to listen on.
        callback: Function or coroutine called when a SIP DTMF event is received.
        participant: Optional participant or identity to filter events for.

    Returns:
        A callable that unregisters the listener when called.
    """
    participant_identity = (
        participant.identity if isinstance(participant, rtc.RemoteParticipant) else participant
    )
    tasks: set[asyncio.Task[Any]] = set()

    def _handler(ev: rtc.SipDTMF) -> None:
        if participant_identity and (
            ev.participant is None or ev.participant.identity != participant_identity
        ):
            return
        res = callback(ev)
        if asyncio.iscoroutine(res):
            task = asyncio.create_task(res)
            tasks.add(task)
            task.add_done_callback(tasks.discard)

    def _unsubscribe() -> None:
        room.off("sip_dtmf_received", _handler)
        for t in list(tasks):
            t.cancel()
        tasks.clear()

    room.on("sip_dtmf_received", _handler)
    return _unsubscribe


async def collect_sip_dtmf(
    room: rtc.Room,
    *,
    num_digits: int | None = None,
    stop_digit: str | None = "#",
    timeout: float = 10.0,
    participant: rtc.RemoteParticipant | str | None = None,
) -> str:
    """Collect an inbound sequence of SIP DTMF digits until count reached or stop digit received.

    Args:
        room: The LiveKit Room instance to listen on.
        num_digits: Number of digits to collect before completing.
        stop_digit: Terminator digit (e.g. '#') that immediately completes collection when received.
        timeout: Maximum duration in seconds to wait for digits.
        participant: Optional participant or identity to filter events for.

    Returns:
        The string of collected digits.
    """
    participant_identity = (
        participant.identity if isinstance(participant, rtc.RemoteParticipant) else participant
    )
    collected: list[str] = []
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[str] = loop.create_future()

    def _on_digit(ev: rtc.SipDTMF) -> None:
        if participant_identity and (
            ev.participant is None or ev.participant.identity != participant_identity
        ):
            return

        if stop_digit is not None and ev.digit == stop_digit:
            if not fut.done():
                fut.set_result("".join(collected))
            return

        collected.append(ev.digit)
        if num_digits is not None and len(collected) >= num_digits:
            if not fut.done():
                fut.set_result("".join(collected))

    room.on("sip_dtmf_received", _on_digit)
    try:
        return await asyncio.wait_for(fut, timeout=timeout)
    finally:
        room.off("sip_dtmf_received", _on_digit)
