from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from livekit.agents import NOT_GIVEN, NotGivenOr
from livekit.agents.llm import ChatContext
from livekit.agents.llm.tool_context import ToolError, ToolFlag, function_tool
from livekit.agents.voice.agent import AgentTask

from .hotel import RoomBooking
from .hotel_db import HotelDB
from .persona import common_instructions

_VERIFY_INSTRUCTIONS = """\
The caller wants to look up an existing reservation - verify them first.

Default path: ask for last name plus confirmation code (codes look like HTL-XXXX). Read the code back letter by letter to confirm before looking up the booking.

Fallback if they don't have the code: ask for last name plus the last four digits of the card on file, then look up the booking by card. Never accept just one of the two. Those are the ONLY two paths - email is not a verification field; never ask for it here.

If the caller already gave their last name or code earlier in the call, use it - don't make them repeat.

If no booking turns up and the caller wants to do something else instead (make a NEW booking, leave it, anything beyond verifying), call give_up right away - the tools for their request live outside this step, and the conversation continues there. Your only tools here are the two lookups and give_up: a call to anything else returns an error and does NOTHING - never tell the caller something was booked, recorded, or arranged after an error.
"""


@dataclass
class VerifyBookingResult:
    booking: RoomBooking


class VerifyBookingTask(AgentTask[VerifyBookingResult]):
    def __init__(
        self,
        db: HotelDB,
        today: date,
        *,
        allow_cancelled: bool = False,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
    ) -> None:
        self._db = db
        self._attempts = 0
        # Reinstating a cancelled booking is the one flow that must verify against a
        # cancelled record; every other flow only accepts a confirmed booking.
        self._allow_cancelled = allow_cancelled
        super().__init__(
            instructions=f"{common_instructions(today)}\n\n{_VERIFY_INSTRUCTIONS}",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Ask the caller for their last name and their confirmation code."
        )

    @function_tool()
    async def lookup_by_code(self, last_name: str, code: str) -> str | None:
        """Look up a booking by last name + confirmation code.

        Args:
            last_name: Caller's last name.
            code: Confirmation code, e.g. 'HTL-7A3K'.
        """
        self._attempts += 1
        code = code.replace(" ", "").upper()
        booking = await self._db.find_booking(last_name=last_name, confirmation_code=code)
        return self._handle(booking, "code")

    @function_tool()
    async def lookup_by_card(self, last_name: str, card_last4: str) -> str | None:
        """Look up a booking by last name + last 4 digits of the card on file.

        Args:
            last_name: Caller's last name.
            card_last4: Last 4 digits of the credit card on file.
        """
        self._attempts += 1
        digits = "".join(c for c in card_last4 if c.isdigit())
        if len(digits) != 4:
            raise ToolError("the last 4 digits should be exactly 4 digits")
        booking = await self._db.find_booking(last_name=last_name, card_last4=digits)
        return self._handle(booking, "card")

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def give_up(self, reason: str) -> None:
        """End the verification step: after repeated failures, OR the moment the caller pivots to something verification isn't needed for (a new booking, a general question, giving up on the lookup). The right tools for their request become available again once this returns.

        Args:
            reason: short explanation (e.g. "no booking found - caller wants a new booking instead")
        """
        if not self.done():
            self.complete(
                ToolError(
                    f"couldn't verify the booking: {reason} | verification is closed - your full "
                    "toolset is back: continue with what the caller actually wants (a new booking "
                    "-> start_room_booking; a followup -> record_followup). Nothing was booked or "
                    "recorded during verification - don't claim otherwise."
                )
            )

    def _handle(self, booking: RoomBooking | None, kind: str) -> str | None:
        if booking is not None and (
            booking.status == "confirmed"
            or (self._allow_cancelled and booking.status == "cancelled")
        ):
            if not self.done():
                self.complete(VerifyBookingResult(booking=booking))
            return None
        if self._attempts >= 3:
            if not self.done():
                self.complete(
                    ToolError(
                        "verification failed after 3 attempts - don't keep trying. "
                        "Apologize, then call record_followup with kind='verification_help' "
                        "so a manager can follow up."
                    )
                )
            return None
        if booking is None:
            return (
                f"No booking found via {kind}. Politely ask the caller to repeat, or offer the "
                "other verification path (code vs. card)."
            )
        return (
            "That booking was already cancelled. Ask if the caller meant a different reservation."
        )
