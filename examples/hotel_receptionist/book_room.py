from __future__ import annotations

from datetime import date
from typing import Annotated

from hotel_db import (
    MAX_PARTY_SIZE,
    TODAY,
    HotelDB,
    RoomBooking,
    RoomExtra,
    RoomType,
    Unavailable,
    speak_usd,
)
from persona import COMMON_INSTRUCTIONS
from pydantic import Field

from livekit.agents import NOT_GIVEN, NotGivenOr, beta
from livekit.agents.llm import ChatContext
from livekit.agents.llm.tool_context import ToolError, ToolFlag, function_tool
from livekit.agents.voice.agent import AgentTask

_BOOK_ROOM_INSTRUCTIONS = """\
You're handling a room booking from start to finish. Collect details in whatever order the caller offers them - don't follow a fixed script, and never re-ask something already given.

Before asking anything, scan the conversation so far. If dates, room type, party size, or smoking preference were already discussed, call the matching recording tools (set_stay, choose_room) right away with those values - don't re-ask the caller for details they already gave.

Run set_stay before choose_room - available rooms depend on the dates. Before calling confirm, make sure you've collected the stay, the room choice, plus the caller's name, email, phone, and card.

Each tool's return ends with a directive for the next action (e.g. "next: call open_email_dialog"). Follow that directive immediately - don't narrate what the tool just did. When the directive says "call confirm() now", call it - the call IS the next action, no filler turn.

If the room sells out at the last second, just pick another - everything else stays captured.
"""


class BookRoomTask(AgentTask[RoomBooking]):
    """The entire room booking as one focused task. `set_stay` / `choose_room`
    handle the part with real coupling - dates <-> availability <-> room - and the
    `open_*_dialog` tools capture each independent detail the moment it's
    offered, storing it on the draft so a later hiccup never re-asks it.
    `confirm()` takes the card, writes the booking, and completes with it."""

    def __init__(self, db: HotelDB, *, chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN) -> None:
        self._db = db
        self._check_in: date | None = None
        self._check_out: date | None = None
        self._guests: int | None = None
        self._room_type: RoomType | None = None
        self._extras: list[RoomExtra] = []
        # Smoking defaults to non-smoking: it's industry-standard opt-in, not
        # a value the caller has to volunteer. choose_room flips it when the
        # caller actually asks for a smoking-permitted room.
        self._smoking: bool = False
        self._first_name: str | None = None
        self._last_name: str | None = None
        self._email: str | None = None
        self._phone: str | None = None
        self._card_last4: str | None = None
        super().__init__(
            instructions=f"{COMMON_INSTRUCTIONS}\n\n{_BOOK_ROOM_INSTRUCTIONS}",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Help the caller book a room. Record anything they've already mentioned - dates, "
                "party size, or room type - then ask only for what's still missing."
            )
        )

    def _status(self) -> str:
        # Action-oriented status, NOT a missing-field list. A "still need: card"
        # string gets parroted by the model as "What card should I use?" - the
        # field name leaks straight into the spoken question. Phrasing each
        # step as the next action avoids that.
        if self._check_in is None:
            return "no stay yet - ask the caller for dates and party size, then call set_stay"
        if self._room_type is None:
            return "stay captured - ask which room type, then call choose_room"
        if not (self._first_name and self._last_name):
            return "stay and room captured - next: call open_name_dialog"
        if not self._email:
            return "name captured - next: call open_email_dialog"
        if not self._phone:
            return "email captured - next: call open_phone_dialog"
        if not self._card_last4:
            return "phone captured - next: call open_credit_card_dialog"
        return "all required details captured - call confirm() now to finalize the booking"

    @function_tool()
    async def set_stay(
        self,
        check_in: date,
        check_out: date,
        guests: Annotated[int, Field(ge=1, le=MAX_PARTY_SIZE)],
    ) -> str:
        """Record the stay dates + party size; returns each available room type with rate and view, so you can answer "how much?" / "what's the cheapest?" without leaving the flow.

        Args:
            check_in: Check-in date in ISO YYYY-MM-DD format (e.g. "2026-01-20").
            check_out: Check-out date in ISO YYYY-MM-DD format.
            guests: Number of guests (must be >= 1; ask the caller if not specified).
        """
        if check_out <= check_in:
            raise ToolError("check-out must be after check-in")
        if (check_out - check_in).days > 30:
            raise ToolError("the max stay is 30 nights")
        if check_in < TODAY:
            raise ToolError("check-in can't be in the past")

        avail = await self._db.list_room_types_available(
            check_in=check_in, check_out=check_out, guests=guests
        )
        if not avail:
            # Don't persist sold-out dates as the active stay - if the model
            # drifts forward without re-setting, the booking would carry
            # invalid dates. The caller needs to pick different dates anyway.
            return f"sold out for {check_in} to {check_out}, {guests} guests - dates not recorded; ask for adjacent dates"

        self._check_in, self._check_out, self._guests = check_in, check_out, guests
        available_types = {a.type for a in avail}
        if self._room_type and self._room_type not in available_types:
            self._room_type = None  # prior choice no longer fits the new dates
        options = " | ".join(
            f"{a.type.replace('_', ' ')} ({speak_usd(a.nightly_rate)}/night, {a.sample_view})"
            for a in avail
        )
        return f"stay recorded ({check_in} to {check_out}, {guests} guests); options: {options} | {self._status()}"

    @function_tool()
    async def choose_room(
        self, room_type: RoomType, extras: list[RoomExtra], smoking_room: bool = False
    ) -> str:
        """Record the chosen room type, extras, and smoking preference.

        Args:
            room_type: One of the room types returned by set_stay.
            extras: Any of breakfast / valet / late_checkout / pets; empty list if none.
            smoking_room: True if the caller wants a smoking-permitted room.
        """
        if self._check_in is None or self._check_out is None or self._guests is None:
            raise ToolError("stay dates and guest count not yet recorded")
        # Re-check against availability filtered by the smoking preference: a
        # type may have rooms free, but not a smoking (or non-smoking) one.
        avail = await self._db.list_room_types_available(
            check_in=self._check_in,
            check_out=self._check_out,
            guests=self._guests,
            smoking=smoking_room,
        )
        chosen = next((a for a in avail if a.type == room_type), None)
        if chosen is None:
            kind = "smoking " if smoking_room else ""
            offer = ", ".join(sorted(a.type for a in avail)) or "nothing for those dates"
            raise ToolError(f"no {kind}{room_type} available; offer one of: {offer}")
        self._room_type = room_type
        self._extras = list(extras)
        self._smoking = smoking_room
        extras_part = f", extras: {', '.join(extras)}" if extras else ""
        return f"room recorded: {room_type.replace('_', ' ')}{extras_part} | {self._status()}"

    @function_tool()
    async def open_name_dialog(self) -> str:
        """Open the name dialog. It collects the guest's first and last name (read back and confirmed) from the caller."""
        r = await beta.workflows.GetNameTask(
            first_name=True,
            last_name=True,
            chat_ctx=self.chat_ctx,
            extra_instructions=COMMON_INSTRUCTIONS,
        )
        self._first_name, self._last_name = r.first_name or "", r.last_name or ""
        return f"name recorded: {self._first_name} {self._last_name} | {self._status()}"

    @function_tool()
    async def open_email_dialog(self) -> str:
        """Open the email dialog. It collects the guest's email address (read back and confirmed) from the caller."""
        r = await beta.workflows.GetEmailTask(
            chat_ctx=self.chat_ctx, extra_instructions=COMMON_INSTRUCTIONS
        )
        self._email = r.email_address
        return f"email recorded: {self._email} | {self._status()}"

    @function_tool()
    async def open_phone_dialog(self) -> str:
        """Open the phone dialog. It collects the guest's phone number (read back and confirmed) from the caller."""
        r = await beta.workflows.GetPhoneNumberTask(
            chat_ctx=self.chat_ctx, extra_instructions=COMMON_INSTRUCTIONS
        )
        self._phone = r.phone_number
        return f"phone recorded: {self._phone} | {self._status()}"

    @function_tool()
    async def open_credit_card_dialog(self) -> str:
        """Open the credit-card dialog. It collects the card number, expiry, security code, and cardholder name from the caller one at a time."""
        card = await beta.workflows.GetCreditCardTask(
            chat_ctx=self.chat_ctx, extra_instructions=COMMON_INSTRUCTIONS
        )
        self._card_last4 = card.card_number[-4:]
        return f"card recorded (ending {self._card_last4}) | {self._status()}"

    @function_tool()
    async def confirm(self) -> str | None:
        """Finalize the booking. All details (stay, room, name, email, phone, card) must already be captured."""
        check_in, check_out, guests, room_type = (
            self._check_in,
            self._check_out,
            self._guests,
            self._room_type,
        )
        first_name, last_name = self._first_name, self._last_name
        email, phone, card_last4 = self._email, self._phone, self._card_last4
        if not (
            check_in
            and check_out
            and guests
            and room_type
            and first_name
            and last_name
            and email
            and phone
            and card_last4
        ):
            raise ToolError(self._status())
        try:
            booking = await self._db.book_room(
                room_type=room_type,
                smoking=self._smoking,
                guests=guests,
                check_in=check_in,
                check_out=check_out,
                first_name=first_name,
                last_name=last_name,
                email=email,
                phone=phone,
                card_last4=card_last4,
                extras=self._extras,
            )
        except Unavailable:
            self._room_type = None
            return (
                "That room just got booked - pick another room or shift the dates; "
                "I've kept everything else."
            )
        if not self.done():
            self.complete(booking)
        return None

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def give_up(self, reason: str) -> None:
        """Caller wants to abandon the booking.

        Args:
            reason: short explanation.
        """
        if not self.done():
            self.complete(ToolError(f"booking abandoned: {reason}"))
