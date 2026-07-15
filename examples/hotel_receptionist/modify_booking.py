from __future__ import annotations

from datetime import date
from typing import Annotated

from pydantic import Field

from livekit.agents import NOT_GIVEN, NotGivenOr
from livekit.agents.llm import ChatContext
from livekit.agents.llm.tool_context import ToolError, ToolFlag, function_tool
from livekit.agents.voice.agent import AgentTask

from .hotel import (
    MAX_PARTY_SIZE,
    RoomBooking,
    RoomExtra,
    RoomType,
    Unavailable,
    format_date,
    speak_usd,
)
from .hotel_db import HotelDB
from .persona import common_instructions

_MODIFY_INSTRUCTIONS = """\
You're modifying an existing room booking. The caller has been verified and the booking is loaded - dates, room, extras, and party size are pre-filled with the current values. Your job is to apply ONLY the changes the caller asks for, then call confirm_changes().

Identity fields (name, email, phone, card) cannot be changed here. If the caller wants to change any of those, say so plainly and steer back to what this flow handles.

Run set_stay before choose_room when changing dates - new dates may make the current room unavailable, in which case set_stay will tell you which types are available for the new range.

A caller unhappy with their room's view (e.g. "I booked a garden view but this room has none") is a room change you CAN make here: pass the view to choose_room and it moves them to a room with that view. If that view isn't available for their current type, choose_room tells you which type has it - offer that, don't fall back to a callback when a real move exists.

For extras (breakfast, valet, late_checkout, pets) the caller adds and removes additively in conversation - merge their request with the current extras list and pass the full new list to choose_room.

Each tool returns a short status with what's pending. When the status says all set, call confirm_changes() - the call IS the next action, no filler turn.

A caller moving rooms because of a view/type complaint often pushes back hard or demands a manager - that is NOT a reason to give up. As long as a room with what they want is available (choose_room tells you when it is, even under a different type), the fix is to complete the move here, not to hand off to a callback. Stay calm, re-offer the available room as the concrete fix, and only escalate if there is genuinely no matching room. A manager callback is a worse outcome than the move you can make right now.

If the caller decides they don't want to change anything after all - or needs anything this flow genuinely can't do (cancel the booking, a callback for something unrelated, a new booking) - call give_up with a short reason; the booking stays as it was and the right tools for their request become available again outside this flow. Your only tools here are set_stay, choose_room, confirm_changes, and give_up: a call to anything else returns an error and does NOTHING - never tell the caller something was logged, promised, or arranged after an error.
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
        today: date,
        *,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
    ) -> None:
        self._db = db
        self._existing = existing
        self._today = today
        # Draft state - starts as a copy of the booking; tools mutate it.
        self._check_in: date = existing.check_in
        self._check_out: date = existing.check_out
        self._guests: int = existing.guests
        self._room_type: RoomType = existing.room_type
        self._extras: list[RoomExtra] = list(existing.extras)
        self._smoking: bool = existing.smoking
        # None = no view change requested, so confirm_changes keeps the guest's
        # current room when it still fits. A stated view re-picks the room to one
        # with that view (e.g. moving an unhappy guest to a garden-view room).
        self._view: str | None = None
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
            f"check-in {format_date(existing.check_in)}, "
            f"check-out {format_date(existing.check_out)}, "
            f"{existing.guests} guest{'s' if existing.guests != 1 else ''}, "
            f"extras: {extras}, total {speak_usd(existing.total)}. "
            "These are the ONLY facts to read back - never invent dates or amounts."
        )
        super().__init__(
            instructions=f"{common_instructions(today)}\n\n{_MODIFY_INSTRUCTIONS}{booking_facts}",
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
        return (
            f"pending changes: {', '.join(parts)} - NOT saved yet. The caller already named what to "
            "change, so call confirm_changes() now to finalize it; don't ask 'anything else?' as a "
            "filler turn (that reads as done and the caller may hang up before it's saved). Only hold "
            "off if the caller themselves raised another change."
        )

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
        if check_in < self._today and check_in != self._existing.check_in:
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
        smoking_room: bool | None = None,
        view: str | None = None,
    ) -> str:
        """Update the chosen room type, extras, smoking preference, and view on the booking being modified.

        Pass the FULL new extras list (e.g. caller asks to add breakfast on a booking that already has valet -> pass ["breakfast", "valet"]). To clear all extras, pass an empty list.

        A stated view moves the guest to a room with that view (this is how you resolve "I booked a garden view but my room has none"). The view is a property of specific rooms, NOT a separate type - if the requested view isn't available for the chosen type, this errors with where that view IS available, so you can offer the right type. Omit view entirely unless the caller asks for one.

        Args:
            room_type: Room type for the booking (king / queen_2beds / double_queen / suite / penthouse).
            extras: Full new list of extras after the caller's change.
            smoking_room: True or false to change the smoking preference; omit to keep the
                booking's current preference.
            view: The view the caller asked for (city / garden / ocean), ONLY if they stated one - omit entirely otherwise.
        """
        smoking = self._smoking if smoking_room is None else smoking_room
        avail = await self._db.list_room_types_available(
            check_in=self._check_in,
            check_out=self._check_out,
            guests=self._guests,
            smoking=smoking,
            exclude_booking_code=self._existing.code,
        )
        chosen = next((a for a in avail if a.type == room_type), None)
        if chosen is None:
            kind = "smoking " if smoking else ""
            offer = ", ".join(sorted(a.type for a in avail)) or "nothing for those dates"
            raise ToolError(f"no {kind}{room_type} available; offer one of: {offer}")
        # Models sometimes send placeholder strings for optional args they
        # should omit - normalize those to "no view preference".
        if view is not None:
            view = view.strip().casefold()
            if view in ("", "null", "none", "any", "no preference", "unspecified"):
                view = None
        if view is not None and view not in chosen.views:
            matching = [a.type for a in avail if view in a.views]
            if matching:
                rec = " or ".join(t.replace("_", " ") for t in matching)
                raise ToolError(
                    f"no {view}-view {room_type.replace('_', ' ')} for those dates, but the "
                    f"{view} view IS open as a {rec} - that is the real fix here. Offer it warmly "
                    f'as "the {view}-view room available for your dates" (don\'t dwell on the type '
                    f"as a downgrade), then call choose_room again with that type and view to "
                    f"complete the move. Do NOT give up to a manager callback - this flow can do it."
                )
            where = ", ".join(f"{a.type.replace('_', ' ')} ({' or '.join(a.views)})" for a in avail)
            raise ToolError(
                f"no {view}-view room of any type for those dates - the views by room type are: "
                f"{where}. Be honest that the exact view isn't open, and offer the closest option."
            )
        self._room_type = room_type
        self._view = view
        self._extras = list(extras)
        self._smoking = smoking
        # A stated view re-picks the room, so it's a change even when the type
        # is unchanged (the whole point of moving an unhappy guest's room).
        self._set_changed("room", room_type != self._existing.room_type or view is not None)
        self._set_changed("smoking", smoking != self._existing.smoking)
        self._set_changed("extras", sorted(extras) != sorted(self._existing.extras))
        view_part = f" with a {view} view" if view else ""
        extras_part = f", extras: {', '.join(extras)}" if extras else ", no extras"
        return f"room updated: {room_type.replace('_', ' ')}{view_part}{extras_part} | {self._status()}"

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
                view=self._view,
            )
        except Unavailable:
            view_part = f"{self._view}-view " if self._view else ""
            raise ToolError(
                f"{view_part}{self._room_type.replace('_', ' ')} just got taken for those dates - pick another room or adjust the dates"
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
