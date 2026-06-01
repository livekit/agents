from __future__ import annotations

from datetime import date, time
from typing import Annotated

from hotel_db import MAX_PARTY_SIZE, TODAY, HotelDB, RestaurantReservation, Unavailable, speak_time
from pydantic import Field
from workflows import COMMON_INSTRUCTIONS, capture_name, capture_phone

from livekit.agents import NOT_GIVEN, NotGivenOr
from livekit.agents.llm import ChatContext
from livekit.agents.llm.tool_context import ToolError, ToolFlag, function_tool
from livekit.agents.voice.agent import AgentTask

_BOOK_RESTAURANT_INSTRUCTIONS = """\
You're handling a restaurant reservation from start to finish. Collect details in whatever order the caller offers them - don't follow a fixed script, and never re-ask something already given.

Before asking anything, scan the conversation so far. If date, party size, time, or special-request notes were already discussed, call the matching recording tools (set_party, choose_time) right away with those values - don't re-ask the caller for details they already gave.

Run set_party before choose_time - open slots depend on the date and party size. Before calling confirm, make sure you've collected the date, party, time, and the caller's name and phone.

Each tool returns a short status with the recorded value plus what's still missing. Use that to ask for the next missing piece naturally, without narrating what the tool just did. When a status says "all required details captured", call confirm() immediately - don't say "one moment" or any other filler; the call IS the next action.
"""


class BookRestaurantTask(AgentTask[RestaurantReservation]):
    """Restaurant booking as one focused task, mirroring BookRoomTask: `set_party`
    / `choose_time` handle the date <-> slot-availability coupling, the
    `open_*_dialog` tools capture each detail the moment it's offered (stored on
    the draft so a later hiccup never re-asks it), and `confirm()` books the
    table."""

    def __init__(self, db: HotelDB, *, chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN) -> None:
        self._db = db
        self._date: date | None = None
        self._party_size: int | None = None
        self._time: time | None = None
        self._notes: str | None = None
        self._open_times: set[time] = set()
        self._first_name: str | None = None
        self._last_name: str | None = None
        self._phone: str | None = None
        super().__init__(
            instructions=f"{COMMON_INSTRUCTIONS}\n\n{_BOOK_RESTAURANT_INSTRUCTIONS}",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Help the caller book a table. Record anything they've already mentioned - date, "
                "party size, or time - then ask only for what's still missing."
            )
        )

    def _missing(self) -> list[str]:
        missing: list[str] = []
        if self._date is None:
            missing.append("date")
        if self._time is None:
            missing.append("time")
        if not (self._first_name and self._last_name):
            missing.append("name")
        if not self._phone:
            missing.append("phone")
        return missing

    def _status(self) -> str:
        # See book_room.py - status describes the next *action*, not a list
        # of missing field names. A "still need: phone" string gets parroted
        # as "What's your phone number?".
        if self._date is None:
            return "no party yet - ask the caller for date and party size, then call set_party"
        if self._time is None:
            return "party captured - ask which time slot, then call choose_time"
        if not (self._first_name and self._last_name):
            return "party and time captured - next: call open_name_dialog"
        if not self._phone:
            return "name captured - next: call open_phone_dialog"
        return "all required details captured - call confirm() now to finalize the reservation"  # fmt: skip

    @function_tool()
    async def set_party(self, on_date: date, party_size: Annotated[int, Field(ge=1, le=MAX_PARTY_SIZE)]) -> str:  # fmt: skip
        """Record the date + party size; returns the open time slots for them.

        Args:
            on_date: Reservation date.
            party_size: Number of guests.
        """
        if on_date < TODAY:
            raise ToolError("the date can't be in the past")

        slots = await self._db.list_restaurant_availability(on_date=on_date, party_size=party_size)
        open_times = {s.time for s in slots if s.available_table_ids}
        if not open_times:
            # Same reasoning as BookRoomTask.set_stay: don't persist a
            # fully-booked date - the caller needs to pick another.
            return f"fully booked on {on_date.strftime('%A, %B %-d')} for {party_size} - date not recorded; ask for another date"  # fmt: skip

        self._date, self._party_size = on_date, party_size
        self._open_times = open_times
        if self._time and self._time not in self._open_times:
            self._time = None  # prior slot no longer open for the new date/party
        labels = ", ".join(speak_time(t) for t in sorted(self._open_times))
        return f"party recorded ({on_date.strftime('%A, %B %-d')}, {party_size} guests); open times: {labels} | {self._status()}"  # fmt: skip

    @function_tool()
    async def choose_time(self, at_time: time, notes: str | None = None) -> str:
        """Record the chosen time slot and any special request.

        Args:
            at_time: Slot time - one of the open slots returned by set_party.
            notes: Optional special request (allergy, anniversary...), or null.
        """
        if self._date is None:
            raise ToolError("date and party size not yet recorded")
        if at_time not in self._open_times:
            open_labels = ", ".join(speak_time(o) for o in sorted(self._open_times))
            raise ToolError(f"{speak_time(at_time)} isn't open; offer one of: {open_labels}")
        self._time = at_time
        self._notes = notes
        notes_part = f", notes: {notes}" if notes else ""
        return f"time recorded: {speak_time(at_time)}{notes_part} | {self._status()}"

    @function_tool()
    async def open_name_dialog(self) -> str:
        """Open the name dialog. It collects the guest's first and last name (read back and confirmed) from the caller."""
        self._first_name, self._last_name = await capture_name(self)
        return f"name recorded: {self._first_name} {self._last_name} | {self._status()}"

    @function_tool()
    async def open_phone_dialog(self) -> str:
        """Open the phone dialog. It collects the guest's phone number (read back and confirmed) from the caller."""
        self._phone = await capture_phone(self)
        return f"phone recorded: {self._phone} | {self._status()}"

    @function_tool()
    async def confirm(self) -> str | None:
        """Finalize once the date, party, time, and the caller's details are all captured: book
        the table."""
        missing = self._missing()
        if missing:
            raise ToolError("still need: " + ", ".join(missing))

        assert self._date and self._party_size and self._time and self._first_name and self._phone
        try:
            reservation = await self._db.book_restaurant(
                first_name=self._first_name,
                last_name=self._last_name or "",
                phone=self._phone,
                party_size=self._party_size,
                on_date=self._date,
                at_time=self._time,
                notes=self._notes,
            )
        except Unavailable:
            self._time = None
            return "That slot just filled up - pick another time; I've kept your details."
        if not self.done():
            self.complete(reservation)
        return None

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def give_up(self, reason: str) -> None:
        """Caller wants to abandon the reservation.

        Args:
            reason: short explanation.
        """
        if not self.done():
            self.complete(ToolError(f"reservation abandoned: {reason}"))
