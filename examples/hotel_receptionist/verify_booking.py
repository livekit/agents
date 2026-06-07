from __future__ import annotations

from dataclasses import dataclass

from hotel_db import HotelDB, RoomBooking
from persona import COMMON_INSTRUCTIONS

from livekit.agents import NOT_GIVEN, NotGivenOr
from livekit.agents.llm import ChatContext
from livekit.agents.llm.tool_context import ToolError, ToolFlag, function_tool
from livekit.agents.voice.agent import AgentTask

_VERIFY_INSTRUCTIONS = """\
The caller wants to look up an existing reservation - verify them first.

Default path: ask for last name plus confirmation code (codes look like HTL-XXXX). Read the code back letter by letter to confirm before looking up the booking.

Fallback if they don't have the code: ask for last name plus the last four digits of the card on file, then look up the booking by card. Never accept just one of the two.

If the caller already gave their last name or code earlier in the call, use it - don't make them repeat.
"""


@dataclass
class VerifyBookingResult:
    booking: RoomBooking


class VerifyBookingTask(AgentTask[VerifyBookingResult]):
    def __init__(self, db: HotelDB, *, chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN) -> None:
        self._db = db
        self._attempts = 0
        super().__init__(
            instructions=f"{COMMON_INSTRUCTIONS}\n\n{_VERIFY_INSTRUCTIONS}",
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
        """Abandon verification after repeated failures.

        Args:
            reason: short explanation
        """
        if not self.done():
            self.complete(ToolError(f"couldn't verify the booking: {reason}"))

    def _handle(self, booking: RoomBooking | None, kind: str) -> str | None:
        if booking is not None and booking.status == "confirmed":
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
