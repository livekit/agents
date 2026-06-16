from __future__ import annotations

from datetime import date
from typing import Annotated

from context import speech_only
from get_card import GetCardTask
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

Run set_stay before choose_room - available rooms depend on the dates. set_stay's options are for YOU to offer, not to act on: name the room types to the caller and let them pick (ask about any preference they've hinted at, like a view) before calling choose_room. Before calling confirm_booking, make sure you've collected the stay, the room choice, plus the caller's name, email, phone, and card - then read the whole booking back in one short sentence (dates, room type and extras, total, card last four) and let the caller say "go ahead" or correct something. confirm_booking only fires once they've agreed to the read-back.

Each tool's return ends with a directive for the next action (e.g. "next: call open_email_dialog"). Follow that directive immediately - don't narrate what the tool just did. When the directive says "call confirm_booking() now", call it - the call IS the next action, no filler turn.

If the room sells out at the last second, just pick another - everything else stays captured.

A booking is not complete unless "confirm_booking" is called. Bookings are only valid once you call "confirm_booking."

Never speak the same question twice in a row. If a field was just captured ("name recorded", "email recorded"), it is DONE - asking for it again stalls the call; the only valid next move is the directive in the last tool return.
"""


class BookRoomTask(AgentTask[RoomBooking]):
    """The entire room booking as one focused task. `set_stay` / `choose_room`
    handle the part with real coupling - dates <-> availability <-> room - and the
    `open_*_dialog` tools capture each independent detail the moment it's
    offered, storing it on the draft so a later hiccup never re-asks it.
    `confirm_booking()` takes the card, writes the booking, and completes with it."""

    def __init__(self, db: HotelDB, *, chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN) -> None:
        self._db = db
        self._check_in: date | None = None
        self._check_out: date | None = None
        self._guests: int | None = None
        self._room_type: RoomType | None = None
        self._view: str | None = None
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
        self._quoted_total: int | None = None
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
        total = (
            f"total {speak_usd(self._quoted_total)} including tax, " if self._quoted_total else ""
        )
        return (
            "all required details captured - read the booking back in one sentence "
            f"(dates, room and extras, {total}card ending {self._card_last4}) and call "
            "confirm_booking() the moment the caller agrees. Quote ONLY this total - "
            "never compute your own."
        )

    @function_tool()
    async def set_stay(
        self,
        check_in: date,
        check_out: date,
        guests: Annotated[int, Field(ge=1, le=MAX_PARTY_SIZE)],
    ) -> str:
        """Record the stay dates + party size. The return lists each available room type with rate and view - that is reference material for answering "how much?" / "what's the cheapest?" and for OFFERING the choice to the caller. Never act on it by picking a type yourself; the next step after this tool is a question, not another tool call.

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
            f"{a.type.replace('_', ' ')} ({speak_usd(a.nightly_rate)}/night, "
            f"{' or '.join(a.views)} view{'s' if len(a.views) > 1 else ''})"
            for a in avail
        )
        return f"stay recorded ({check_in} to {check_out}, {guests} guests); options: {options} | {self._status()}"

    @function_tool()
    async def choose_room(
        self,
        room_type: RoomType,
        extras: list[RoomExtra],
        smoking_room: bool = False,
        view: str | None = None,
    ) -> str:
        """Record the room type the caller chose from the options set_stay returned, plus any view they asked for.

        Call ONLY after the caller has named a room type (a stated view narrows WHICH room of that type they get - it doesn't pick the type). If the caller asks for a view, pass it here; if that view isn't available for the type, this errors with where the view IS available - relay that and let them choose. Never guess a type from a preference.

        Args:
            room_type: The room type exactly as the caller chose it.
            extras: Any of breakfast / valet / late_checkout / pets; empty list if none.
            smoking_room: True if the caller wants a smoking-permitted room.
            view: The view the caller asked for (city / garden / ocean), ONLY if they stated one - omit entirely otherwise.
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
        # Models sometimes send placeholder strings for optional args they
        # should omit - normalize those to "no view preference".
        if view is not None:
            view = view.strip().casefold()
            if view in ("", "null", "none", "any", "no preference", "unspecified"):
                view = None
        if view is not None and view not in chosen.views:
            where = ", ".join(f"{a.type.replace('_', ' ')} ({' or '.join(a.views)})" for a in avail)
            raise ToolError(
                f"no {view}-view {room_type.replace('_', ' ')} for those dates - "
                f"the views by room type are: {where}. Tell the caller and let them choose."
            )
        self._room_type = room_type
        self._view = view
        self._extras = list(extras)
        self._smoking = smoking_room
        # The exact total (with tax) for the room that will be booked - quoted
        # here so the read-back uses the real number, never per-night arithmetic.
        self._quoted_total = await self._db.peek_stay_total(
            room_type=room_type,
            smoking=smoking_room,
            guests=self._guests,
            check_in=self._check_in,
            check_out=self._check_out,
            view=view,
            extras=extras,
        )
        view_part = f" with a {view} view" if view else ""
        extras_part = f", extras: {', '.join(extras)}" if extras else ""
        total_part = (
            f"; total for the stay {speak_usd(self._quoted_total)} including tax"
            if self._quoted_total
            else ""
        )
        return f"room recorded: {room_type.replace('_', ' ')}{view_part}{extras_part}{total_part} | {self._status()}"

    @function_tool()
    async def open_name_dialog(self) -> str:
        """Open the name dialog. It collects the guest's first and last name (read back and confirmed) from the caller."""
        r = await beta.workflows.GetNameTask(
            first_name=True,
            last_name=True,
            chat_ctx=speech_only(self.chat_ctx),
            extra_instructions=COMMON_INSTRUCTIONS,
        )
        self._first_name, self._last_name = r.first_name or "", r.last_name or ""
        return f"name recorded: {self._first_name} {self._last_name} | {self._status()}"

    @function_tool()
    async def open_email_dialog(self) -> str:
        """Open the email dialog. It collects the guest's email address (read back and confirmed) from the caller."""
        r = await beta.workflows.GetEmailTask(
            chat_ctx=speech_only(self.chat_ctx), extra_instructions=COMMON_INSTRUCTIONS
        )
        self._email = r.email_address
        return f"email recorded: {self._email} | {self._status()}"

    @function_tool()
    async def open_phone_dialog(self) -> str:
        """Open the phone dialog. It collects the guest's phone number (read back and confirmed) from the caller."""
        r = await beta.workflows.GetPhoneNumberTask(
            chat_ctx=speech_only(self.chat_ctx), extra_instructions=COMMON_INSTRUCTIONS
        )
        self._phone = r.phone_number
        return f"phone recorded: {self._phone} | {self._status()}"

    @function_tool()
    async def open_credit_card_dialog(self) -> str:
        """Open the credit-card dialog. It collects the card number, expiry, security code, and cardholder name from the caller in one focused step."""
        card = await GetCardTask(chat_ctx=speech_only(self.chat_ctx))
        self._card_last4 = card.card_number[-4:]
        return f"card recorded (ending {self._card_last4}) | {self._status()}"

    @function_tool()
    async def confirm_booking(self) -> str | None:
        """Finalize the booking and charge the card. Call ONLY after every detail is captured AND the caller has agreed to your read-back (dates, room and extras, total, card last four). Returns the final confirmation - relay it to the caller; the booking flow ends with this call."""
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
                view=self._view,
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
