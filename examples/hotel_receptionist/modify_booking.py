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

from livekit.agents import NOT_GIVEN, NotGivenOr
from livekit.agents.llm import ChatContext
from livekit.agents.llm.tool_context import ToolError, ToolFlag, function_tool
from livekit.agents.voice.agent import AgentTask

_MODIFY_INSTRUCTIONS = """\
You're modifying an existing room booking. The caller has been verified and the booking is loaded - dates, room, extras, and party size are pre-filled with the current values. Your job is to apply ONLY the changes the caller asks for, then call confirm_changes().

Identity fields (name, email, phone, card) cannot be changed here. If the caller wants to change any of those, say so plainly and steer back to what this flow handles.

Run set_stay before choose_room when changing dates - new dates may make the current room unavailable, in which case set_stay will tell you which types are available for the new range.

For extras (breakfast, valet, late_checkout, pets) the caller adds and removes additively in conversation - merge their request with the current extras list and pass the full new list to choose_room.

Each tool returns a short status with what's pending. When the status says all set, call confirm_changes() - the call IS the next action, no filler turn.

If the caller decides they don't want to change anything after all - or needs anything this flow can't do (cancel the booking, a manager callback, a followup, a new booking) - call give_up with a short reason; the booking stays as it was and the right tools for their request become available again outside this flow. Your only tools here are set_stay, choose_room, confirm_changes, and give_up: a call to anything else returns an error and does NOTHING - never tell the caller something was logged, promised, or arranged after an error.
"""


class ModifyBookingTask(AgentTask[RoomBooking]):
    """Modify a confirmed booking. Pre-fills draft state from the existing
    booking; `set_stay` / `choose_room` mutate the draft; `confirm_changes()` writes
    the changes back via `HotelDB.update_booking()`. Identity fields (name,
    email, phone, card) are NOT touched."""

    def __init__(
        self,
        db: HotelDB,
        existing: RoomBooking,
        *,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
    ) -> None:
        self._db = db
        self._existing = existing
        # Draft state - starts as a copy of the booking; tools mutate it.
        self._check_in: date = existing.check_in
        self._check_out: date = existing.check_out
        self._guests: int = existing.guests
        self._room_type: RoomType = existing.room_type
        self._extras: list[RoomExtra] = list(existing.extras)
        self._smoking: bool = existing.smoking
        # Set of slot names that diverge from `existing` after a tool call.
        # `confirm_changes()` consults this both to decide if there's anything to do
        # and to produce a faithful summary of what changed.
        self._changed: set[str] = set()
        # The model is asked to read the booking back - so the booking's actual
        # facts must be IN its context, or it will invent dates and amounts.
        extras = ", ".join(existing.extras) if existing.extras else "none"
        booking_facts = (
            f"\nThe loaded booking: {existing.first_name} {existing.last_name}, "
            f"{existing.room_type.replace('_', ' ')}, "
            f"check-in {existing.check_in.strftime('%A, %B %-d')}, "
            f"check-out {existing.check_out.strftime('%A, %B %-d')}, "
            f"{existing.guests} guest{'s' if existing.guests != 1 else ''}, "
            f"extras: {extras}, total {speak_usd(existing.total)}. "
            "These are the ONLY facts to read back - never invent dates or amounts."
        )
        super().__init__(
            instructions=f"{COMMON_INSTRUCTIONS}\n\n{_MODIFY_INSTRUCTIONS}{booking_facts}",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Read the booking back briefly in one sentence (guest name, dates, room "
                "type, current extras if any) and ask what the caller wants to change. "
                "Do not list every column - just enough that the caller recognizes the "
                "booking and can say what to update."
            )
        )

    def _status(self) -> str:
        if not self._changed:
            return "draft unchanged so far - ask the caller what to update"
        parts = sorted(self._changed)
        return f"pending changes: {', '.join(parts)} | call confirm_changes() when the caller has nothing else to change"

    @function_tool()
    async def set_stay(
        self,
        check_in: date,
        check_out: date,
        guests: Annotated[int, Field(ge=1, le=MAX_PARTY_SIZE)],
    ) -> str:
        """Update the stay (check-in, check-out, party size) on the booking being modified.

        Pass the FULL new stay even if only one field is changing - omit nothing.

        Args:
            check_in: New check-in date in ISO YYYY-MM-DD format. May equal the
                existing check-in (e.g. when the caller is in-house and just
                wants to extend or shorten the check-out).
            check_out: New check-out date in ISO YYYY-MM-DD format.
            guests: New number of guests (must be >= 1).
        """
        if check_out <= check_in:
            raise ToolError("check-out must be after check-in")
        if (check_out - check_in).days > 30:
            raise ToolError("the max stay is 30 nights")
        # A future check-in can't be in the past, but if the booking is
        # in-house already its existing check_in IS in the past, and the
        # caller should be able to keep it while shifting check_out.
        if check_in < TODAY and check_in != self._existing.check_in:
            raise ToolError("check-in can't be in the past")

        avail = await self._db.list_room_types_available(
            check_in=check_in,
            check_out=check_out,
            guests=guests,
            smoking=self._smoking,
            exclude_booking_code=self._existing.code,
        )
        if not avail:
            return f"sold out for {check_in} to {check_out}, {guests} guests - dates not recorded; ask for adjacent dates"

        self._check_in, self._check_out, self._guests = check_in, check_out, guests
        existing = self._existing
        self._set_changed(
            "stay",
            (check_in, check_out, guests)
            != (existing.check_in, existing.check_out, existing.guests),
        )
        # If the current room type is no longer available for the new
        # dates, surface that so the model can re-pick. Don't silently drop
        # the choice - the caller needs to know.
        types = {a.type for a in avail}
        if self._room_type not in types:
            available_list = ", ".join(t.replace("_", " ") for t in sorted(types))
            return (
                f"stay updated ({check_in} to {check_out}, {guests} guests); "
                f"{self._room_type.replace('_', ' ')} is no longer available for those dates - "
                f"offer one of: {available_list}, then call choose_room | {self._status()}"
            )
        return f"stay updated ({check_in} to {check_out}, {guests} guests) | {self._status()}"

    @function_tool()
    async def choose_room(
        self,
        room_type: RoomType,
        extras: list[RoomExtra],
        smoking_room: bool = False,
    ) -> str:
        """Update the chosen room type, extras, and smoking preference on the booking being modified.

        Pass the FULL new extras list (e.g. caller asks to add breakfast on a booking that already has valet -> pass ["breakfast", "valet"]). To clear all extras, pass an empty list.

        Args:
            room_type: Room type for the booking (king / queen_2beds / double_queen / suite / penthouse).
            extras: Full new list of extras after the caller's change.
            smoking_room: True if the caller wants a smoking-permitted room.
        """
        avail = await self._db.list_room_types_available(
            check_in=self._check_in,
            check_out=self._check_out,
            guests=self._guests,
            smoking=smoking_room,
            exclude_booking_code=self._existing.code,
        )
        chosen = next((a for a in avail if a.type == room_type), None)
        if chosen is None:
            kind = "smoking " if smoking_room else ""
            offer = ", ".join(sorted(a.type for a in avail)) or "nothing for those dates"
            raise ToolError(f"no {kind}{room_type} available; offer one of: {offer}")
        self._room_type = room_type
        self._extras = list(extras)
        self._smoking = smoking_room
        self._set_changed("room", room_type != self._existing.room_type)
        self._set_changed("smoking", smoking_room != self._existing.smoking)
        self._set_changed("extras", sorted(extras) != sorted(self._existing.extras))
        extras_part = f", extras: {', '.join(extras)}" if extras else ", no extras"
        return f"room updated: {room_type.replace('_', ' ')}{extras_part} | {self._status()}"

    @function_tool()
    async def confirm_changes(self) -> str | None:
        """Write the pending changes back to the booking. Call this once the
        caller has nothing more to change."""
        if not self._changed:
            # Caller didn't end up changing anything - bail out cleanly.
            if not self.done():
                self.complete(self._existing)
            return None

        try:
            updated = await self._db.update_booking(
                booking_code=self._existing.code,
                room_type=self._room_type,
                smoking=self._smoking,
                guests=self._guests,
                check_in=self._check_in,
                check_out=self._check_out,
                extras=self._extras,
            )
        except Unavailable:
            raise ToolError(
                f"{self._room_type.replace('_', ' ')} just got taken for those dates - pick another room or adjust the dates"
            ) from None

        if not self.done():
            self.complete(updated)
        return None

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def give_up(self, reason: str) -> None:
        """End the modification flow, leaving the booking unchanged: the caller no longer wants changes, OR they need something this flow can't do (cancellation, manager callback, followup, anything else). The right tools for their request become available once this returns.

        Args:
            reason: short explanation (e.g. "guest wants a manager callback instead").
        """
        if not self.done():
            self.complete(self._existing)

    def _set_changed(self, slot: str, differs: bool) -> None:
        if differs:
            self._changed.add(slot)
        else:
            self._changed.discard(slot)
