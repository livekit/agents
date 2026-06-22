from __future__ import annotations

from datetime import date, time
from typing import Annotated

from context import speech_only
from hotel_db import MAX_PARTY_SIZE, TODAY, HotelDB, RestaurantReservation, Unavailable, speak_time
from persona import COMMON_INSTRUCTIONS
from pydantic import Field

from livekit.agents import NOT_GIVEN, NotGivenOr, beta
from livekit.agents.llm import ChatContext
from livekit.agents.llm.tool_context import ToolError, ToolFlag, function_tool
from livekit.agents.voice.agent import AgentTask

_BOOK_RESTAURANT_INSTRUCTIONS = """\
You're handling a restaurant reservation from start to finish. Collect details in whatever order the caller offers them - don't follow a fixed script, and never re-ask something already given.

Before asking anything, scan the conversation so far. If date, party size, time, or special-request notes were already discussed, call the matching recording tools (set_party, choose_time) right away with those values - don't re-ask the caller for details they already gave.

Run set_party before choose_time - open slots depend on the date and party size. Before calling confirm_reservation, make sure you've collected the date, party, time, and the caller's name and phone - then read the reservation back in one short sentence (date, time, party size, name) and let the caller agree. confirm_reservation only fires once they've agreed to the read-back.

Each tool's return ends with a directive for the next action (e.g. "next: call open_phone_dialog"). Follow that directive immediately - don't narrate what the tool just did. When the directive says "call confirm_reservation() now", call it - the call IS the next action, no filler turn.

Never speak the same question twice in a row. If a field was just captured ("name recorded", "time recorded"), it is DONE - asking for it again stalls the call; the only valid next move is the directive in the last tool return.
"""


class BookRestaurantTask(AgentTask[RestaurantReservation]):
    """Restaurant booking as one focused task, mirroring BookRoomTask: `set_party`
    / `choose_time` handle the date <-> slot-availability coupling, the
    `open_*_dialog` tools capture each detail the moment it's offered (stored on
    the draft so a later hiccup never re-asks it), and `confirm_reservation()` books the
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

    def _status(self) -> str:
        if self._date is None:
            return "no party yet - ask the caller for date and party size, then call set_party"
        if self._time is None:
            return "party captured - ask which time slot, then call choose_time"
        if not (self._first_name and self._last_name):
            return "party and time captured - next: call open_name_dialog"
        if not self._phone:
            return "name captured - next: call open_phone_dialog"
        return "all required details captured - call confirm_reservation() now to finalize the reservation"

    @function_tool()
    async def set_party(
        self, on_date: date, party_size: Annotated[int, Field(ge=1)]
    ) -> str:
        """Record the date + party size. The return lists the open time slots - offer them to the caller and let them pick; don't choose a slot yourself.

        Args:
            on_date: Reservation date in ISO YYYY-MM-DD format (e.g. "2026-01-20").
            party_size: Number of guests, exactly as the caller stated it - never shrink it to fit; if it's too big to seat, that's handled below.
        """
        if on_date < TODAY:
            raise ToolError("the date can't be in the past")
        if party_size > MAX_PARTY_SIZE:
            # The largest table seats MAX_PARTY_SIZE; a bigger party (and the
            # private-room / set-menu asks that come with it) is the restaurant's
            # to arrange, not a desk table booking. Bail out of this flow and
            # transfer rather than quietly booking a too-small table.
            raise ToolError(
                f"{party_size} guests is beyond a normal table - we seat up to {MAX_PARTY_SIZE}. "
                "Don't book it here and don't reduce the number to fit: this is a large-party / "
                "private-dining request the restaurant handles directly. Call give_up, then tell "
                "the caller you'll put them on hold to connect them and, once they agree, "
                "transfer_call(destination='restaurant') with a one-line summary."
            )

        slots = await self._db.list_restaurant_availability(on_date=on_date, party_size=party_size)
        open_times = {s.time for s in slots if s.available_table_ids}
        if not open_times:
            # Same reasoning as BookRoomTask.set_stay: don't persist a
            # fully-booked date - the caller needs to pick another.
            return f"fully booked on {on_date.strftime('%A, %B %-d')} for {party_size} - date not recorded; ask for another date"

        self._date, self._party_size = on_date, party_size
        self._open_times = open_times
        if self._time and self._time not in self._open_times:
            self._time = None  # prior slot no longer open for the new date/party
        labels = ", ".join(speak_time(t) for t in sorted(self._open_times))
        return f"party recorded ({on_date.strftime('%A, %B %-d')}, {party_size} guests); open times: {labels} | {self._status()}"

    @function_tool()
    async def choose_time(self, at_time: time, notes: str | None = None) -> str:
        """Record the chosen time slot and any special request.

        Args:
            at_time: The slot the CALLER picked, from the open times set_party returned.
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
        r = await beta.workflows.GetNameTask(
            first_name=True,
            last_name=True,
            chat_ctx=speech_only(self.chat_ctx),
            extra_instructions=COMMON_INSTRUCTIONS,
        )
        self._first_name, self._last_name = r.first_name or "", r.last_name or ""
        return f"name recorded: {self._first_name} {self._last_name} | {self._status()}"

    @function_tool()
    async def open_phone_dialog(self) -> str:
        """Open the phone dialog. It collects the guest's phone number (read back and confirmed) from the caller."""
        r = await beta.workflows.GetPhoneNumberTask(
            chat_ctx=speech_only(self.chat_ctx), extra_instructions=COMMON_INSTRUCTIONS
        )
        self._phone = r.phone_number
        return f"phone recorded: {self._phone} | {self._status()}"

    @function_tool()
    async def confirm_reservation(self) -> str | None:
        """Finalize once the date, party, time, and the caller's details are all captured: book
        the table."""
        on_date, party_size, at_time = self._date, self._party_size, self._time
        first_name, phone = self._first_name, self._phone
        if not (on_date and party_size and at_time and first_name and phone):
            raise ToolError(self._status())
        try:
            reservation = await self._db.book_restaurant(
                first_name=first_name,
                last_name=self._last_name or "",
                phone=phone,
                party_size=party_size,
                on_date=on_date,
                at_time=at_time,
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
