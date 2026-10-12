from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Annotated

from hotel import (
    CANCELLATION_WINDOW_HOURS,
    MAX_GUESTS,
    TODAY,
    Booking,
    Hotel,
    RoomType,
    Unavailable,
    View,
)
from pydantic import BaseModel, Field

from livekit.agents import (
    Agent,
    AgentHandler,
    AgentTask,
    Completed,
    Failed,
    RunContext,
    ToolError,
    function_tool,
)
from livekit.agents.a2a import A2ASessionContext


class Caller(BaseModel):
    """What the back office keeps about the caller across turns, on its lane."""

    verified_code: str | None = None
    last_booking_code: str | None = None


# ---------------------------------------------------------------------------------------------
# the guided flows
# ---------------------------------------------------------------------------------------------


class Attempts(BaseModel):
    count: int = 0


@dataclass
class VerifyBooking(AgentTask[str, Attempts]):
    """Finds the guest's booking from their last name and confirmation code; three attempts."""

    def __post_init__(self) -> None:
        super().__init__(
            instructions=(
                "Verify the guest: ask for their last name and confirmation code, HTL and four "
                "characters, and look the booking up. Use what they already said, ask for what is "
                "missing, one thing at a time, in one plain sentence. If they want something that "
                "needs no booking, give_up."
            ),
            llm="google/gemma-4-31b-it",
        )

    @function_tool
    async def lookup(
        self, ctx: RunContext[Hotel, Caller], last_name: str, code: str
    ) -> str | Completed[str] | Failed:
        """Look the booking up.

        Args:
            last_name: the guest's last name
            code: the confirmation code, like HTL-7A3K
        """
        self.state.count += 1
        booking = ctx.deps.find(last_name=last_name, code=code)
        if booking is not None:
            ctx.state.verified_code = booking.code
            return Completed(
                f"verified: {booking.code} for {booking.first_name} {booking.last_name}; now do what the guest asked"
            )
        if self.state.count >= 3:
            return Failed(
                "not verified after three attempts; apologize, say the code is in the confirmation "
                "email, and offer to call back with it or to make a new booking"
            )
        return "no booking found; ask the guest to check the name and the code"

    @function_tool
    async def give_up(self, ctx: RunContext[Hotel, Caller], reason: str) -> Failed:
        """End verification: the guest wants something else, or has no details.

        Args:
            reason: why, in a few words
        """
        return Failed(f"not verified: {reason}")


class RoomDraft(BaseModel):
    """The booking as it takes shape, the task's own state."""

    check_in: date | None = None
    check_out: date | None = None
    guests: int | None = None
    room_type: RoomType | None = None
    view: View | None = None
    quoted_total: int | None = None
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    card_last4: str | None = None


@dataclass
class BookRoom(AgentTask[str, RoomDraft]):
    """A room booking from the dates to the card; completes with the confirmation."""

    check_in: date | None = None  # what the guest already said, from the routing tool
    check_out: date | None = None
    guests: int | None = None

    def __post_init__(self) -> None:
        super().__init__(
            instructions=(
                f"Book a room for the guest; today is {TODAY} and dates are in the future. Record "
                "what they already said before asking for anything, ask only for what is still "
                "missing, one thing at a time, in one plain sentence, and never invent a value "
                "they did not give. set_stay first, since the rooms depend on the dates; offer the "
                "rooms it returns and let the guest pick, then choose_room; then record_guest and "
                "record_card. With everything recorded, confirm_booking reads the booking back "
                "and asks the guest to agree; nothing is booked before that."
            ),
            llm="google/gemma-4-31b-it",
        )
        self.state = RoomDraft(check_in=self.check_in, check_out=self.check_out, guests=self.guests)
        self.instructions(self._status)

    def _status(self, ctx: RunContext[Hotel, Caller]) -> str:
        d = self.state
        if d.check_in is None or d.check_out is None or d.guests is None:
            return "Status: no stay yet; ask for the dates and the party size, then set_stay."
        if d.room_type is None:
            return "Status: stay recorded; the guest picks a room, then choose_room."
        total = f"the total is ${d.quoted_total} with tax"
        if not (d.first_name and d.last_name and d.email):
            return f"Status: room recorded, {total}; next record_guest."
        if d.card_last4 is None:
            return f"Status: guest recorded, {total}; next record_card."
        return f"Status: everything recorded, {total} on the card ending {d.card_last4}; call confirm_booking."

    @function_tool
    async def set_stay(
        self,
        ctx: RunContext[Hotel, Caller],
        check_in: date,
        check_out: date,
        guests: Annotated[int, Field(ge=1)],
    ) -> str:
        """Record the dates and the party size; returns the rooms available with rates and views.

        Args:
            check_in: the check-in date, YYYY-MM-DD
            check_out: the check-out date, YYYY-MM-DD
            guests: how many guests
        """
        check_dates(check_in, check_out, guests)
        rooms = ctx.deps.available(check_in=check_in, check_out=check_out, guests=guests)
        if not rooms:
            return f"sold out {check_in} to {check_out} for {guests}; offer other dates"
        d = self.state
        d.check_in, d.check_out, d.guests, d.room_type = check_in, check_out, guests, None
        return "available: " + ", ".join(
            f"{r.type} with {r.view} view at ${r.nightly_rate} a night" for r in rooms
        )

    @function_tool
    async def choose_room(
        self, ctx: RunContext[Hotel, Caller], room_type: RoomType, view: View | None = None
    ) -> str:
        """Record the room the guest chose from what set_stay offered.

        Args:
            room_type: the type the guest picked
            view: the view they asked for, only if they named one
        """
        d = self.state
        if d.check_in is None or d.check_out is None or d.guests is None:
            raise ToolError("set_stay first")
        rooms = [
            r
            for r in ctx.deps.available(
                check_in=d.check_in, check_out=d.check_out, guests=d.guests, room_type=room_type
            )
            if view is None or r.view == view
        ]
        if not rooms:
            raise ToolError(
                f"no {view or ''} {room_type} for those dates; offer what set_stay returned"
            )
        d.room_type, d.view = room_type, view
        d.quoted_total = ctx.deps.quote(rooms[0], d.check_in, d.check_out)
        return f"room recorded; the stay totals ${d.quoted_total} with tax"

    @function_tool
    async def record_guest(
        self, ctx: RunContext[Hotel, Caller], first_name: str, last_name: str, email: str
    ) -> str:
        """Record the guest's name and email, as they gave them.

        Args:
            first_name: the guest's first name
            last_name: the guest's last name
            email: where the confirmation goes
        """
        d = self.state
        d.first_name, d.last_name, d.email = first_name, last_name, email
        return "guest recorded"

    @function_tool
    async def record_card(
        self, ctx: RunContext[Hotel, Caller], card_number: str, expiry: str, security_code: str
    ) -> str:
        """Record the card; only its last four digits are kept.

        Args:
            card_number: the card number
            expiry: the expiry, MM/YY
            security_code: the three or four digit code
        """
        digits = "".join(c for c in card_number if c.isdigit())
        if len(digits) not in (15, 16):
            raise ToolError("a card number is 15 or 16 digits; ask the guest to repeat it")
        self.state.card_last4 = digits[-4:]
        return f"card recorded, ending {digits[-4:]}"

    def _read_back(self, ctx: RunContext[Hotel, Caller]) -> str:
        """The booking as the guest approves it, once everything is recorded."""
        d = self.state
        if not (
            d.check_in
            and d.check_out
            and d.guests
            and d.room_type
            and d.first_name
            and d.last_name
            and d.email
            and d.card_last4
        ):
            raise ToolError(self._status(ctx))
        view = f" with {d.view} view" if d.view else ""
        return (
            f"Please confirm: a {d.room_type}{view}, {d.check_in} to {d.check_out}, "
            f"{d.guests} guests, under {d.first_name} {d.last_name}, confirmation to {d.email}, "
            f"${d.quoted_total} with tax charged to the card ending {d.card_last4}. Yes or no?"
        )

    @function_tool(requires_approval=_read_back)
    async def confirm_booking(self, ctx: RunContext[Hotel, Caller]) -> Completed[str] | str:
        """Write the booking and charge the card. It reads the booking back and asks the guest;
        their yes comes back as this call's output, and then it is called again the same way."""
        d = self.state
        assert d.check_in and d.check_out and d.guests and d.room_type  # the read-back checked
        assert d.first_name and d.last_name and d.email and d.card_last4
        try:
            booking = ctx.deps.book(
                room_type=d.room_type,
                view=d.view,
                check_in=d.check_in,
                check_out=d.check_out,
                guests=d.guests,
                first_name=d.first_name,
                last_name=d.last_name,
                email=d.email,
                card_last4=d.card_last4,
            )
        except Unavailable:
            d.room_type = None
            return "that room was just taken; offer another room or other dates, the rest is kept"
        ctx.state.last_booking_code = booking.code
        return Completed(describe(booking) + f"; confirmation emailed to {booking.email}")

    @function_tool
    async def give_up(self, ctx: RunContext[Hotel, Caller], reason: str) -> Failed:
        """The guest abandons the booking.

        Args:
            reason: why, in a few words
        """
        return Failed(f"booking abandoned: {reason}")


def check_dates(check_in: date, check_out: date, guests: int) -> None:
    if check_out <= check_in:
        raise ToolError("check-out must be after check-in")
    if check_in < TODAY:
        raise ToolError("check-in is in the past")
    if guests > MAX_GUESTS:
        raise ToolError(f"the rooms sleep up to {MAX_GUESTS}; a party of {guests} takes two rooms")


def describe(b: Booking) -> str:
    return (
        f"{b.code}: a {b.room.type} with {b.room.view} view, {b.check_in} to {b.check_out}, "
        f"{b.guests} guests, total ${b.total} with tax on the card ending {b.card_last4}, {b.status}"
    )


# ---------------------------------------------------------------------------------------------
# the back office: one routing agent, the hotel's operations as its tools
# ---------------------------------------------------------------------------------------------

backend = Agent(
    instructions=(
        f"You are the back office of The LiveKit Hotel, a small boutique hotel; today is {TODAY} "
        "and dates are in the future. You answer the guest's request in English, from your tools, in one or "
        "two plain sentences, and never claim anything a tool did not return. The rooms sleep up to "
        "four; a larger party takes two rooms. A guest who wants to book starts "
        "start_room_booking at once; the flow collects everything. To look up or cancel a "
        "booking, call the tool: it verifies the guest by last name and confirmation code "
        "itself, and nothing is looked up or cancelled without them. Never confirm or deny that "
        "anyone is staying at the hotel, and never give a room number; a message for a guest "
        "can be taken for them. A write reads the details back and asks the guest; "
        "their yes or no is the next request, and a yes calls the same tool again exactly "
        "the same way."
    ),
    llm="google/gemma-4-31b-it",
    deps_type=Hotel,
    state_type=Caller,
)


@backend.instructions
def known(ctx: RunContext[Hotel, Caller]) -> str:
    parts = []
    if ctx.state.verified_code:
        parts.append(f"The guest is verified on booking {ctx.state.verified_code}.")
    if ctx.state.last_booking_code:
        parts.append(f"Booking {ctx.state.last_booking_code} was made in this conversation.")
    return " ".join(parts)


def verified(ctx: RunContext[Hotel, Caller]) -> Booking | None:
    """The guest's booking from the lane's state, verified earlier in the conversation."""
    code = ctx.state.verified_code
    booking = next((b for b in ctx.deps.bookings if b.code == code), None) if code else None
    if booking is None or booking.status != "confirmed":
        ctx.state.verified_code = None
    return booking


@backend.tool
async def check_room_availability(
    ctx: RunContext[Hotel, Caller], check_in: date, check_out: date, guests: int
) -> str:
    """What is available for the dates, with rates and views; books nothing.

    Args:
        check_in: the check-in date, YYYY-MM-DD
        check_out: the check-out date, YYYY-MM-DD
        guests: how many guests
    """
    check_dates(check_in, check_out, guests)
    rooms = ctx.deps.available(check_in=check_in, check_out=check_out, guests=guests)
    if not rooms:
        return "nothing available for those dates; offer the nights either side"
    return ", ".join(f"{r.type} with {r.view} view at ${r.nightly_rate} a night" for r in rooms)


@backend.tool
async def start_room_booking(
    ctx: RunContext[Hotel, Caller],
    check_in: date | None = None,
    check_out: date | None = None,
    guests: int | None = None,
) -> str:
    """Book a room; the flow collects the stay, the room, the guest and the card, and reads the
    booking back before writing it. Pass what the guest already said.

    Args:
        check_in: the check-in date, if said
        check_out: the check-out date, if said
        guests: the party size, if said
    """
    # a tool's code after an awaited task does not run in text: the task's Completed text is
    # this call's output, and the turns of the booking pass in between
    return await BookRoom(check_in=check_in, check_out=check_out, guests=guests)


@backend.tool
async def lookup_booking(ctx: RunContext[Hotel, Caller]) -> str:
    """The guest's booking, read-only: dates, room, total. Verifies first."""
    booking = verified(ctx)
    if booking is None:
        return await VerifyBooking()  # its value tells the model to call this tool again
    return describe(booking)


async def cancellation(ctx: RunContext[Hotel, Caller]) -> str:
    """The cancellation and its refund as the guest approves them; verifies the guest first."""
    booking = verified(ctx)
    if booking is None:
        return await VerifyBooking()  # its value tells the model to call the tool again
    refund = ctx.deps.refund(booking)
    penalty = (
        f"inside the {CANCELLATION_WINDOW_HOURS}-hour window one night is forfeited and "
        if refund < booking.total
        else ""
    )
    return (
        f"Please confirm: cancel {booking.code}, {booking.check_in} to {booking.check_out}; "
        f"{penalty}${refund} goes back to the card ending {booking.card_last4}. Yes or no?"
    )


@backend.tool(requires_approval=cancellation)
async def cancel_room_booking(ctx: RunContext[Hotel, Caller]) -> str:
    """Cancel the guest's booking. Verifies first, then reads the booking and the refund back
    and asks the guest; their yes comes back as this call's output, and then it is called
    again the same way."""
    booking = verified(ctx)
    if booking is None:
        raise ToolError("the guest is not verified; call this tool again to verify them")
    refund = ctx.deps.cancel(booking)
    ctx.state.verified_code = None
    if refund < booking.total:
        return (
            f"cancelled; inside the {CANCELLATION_WINDOW_HOURS}-hour window one night is forfeited "
            f"and ${refund} goes back to the card"
        )
    return f"cancelled; no penalty, ${refund} back to the card in two to five business days"


# ---------------------------------------------------------------------------------------------
# the door
# ---------------------------------------------------------------------------------------------

hotel = Hotel.seeded()


back_office = AgentHandler(lane="backend")


@back_office.a2a_session(
    endpoint="backend",
    description="The LiveKit Hotel's back office: rooms and bookings.",
    public=False,
    inbox="queue",
    state=Caller,
)
async def text(ctx: A2ASessionContext[Caller]) -> str:
    result = await backend.run(
        ctx.message, message_history=ctx.message_history, state=ctx.state, deps=hotel
    )
    return result.input_required or str(result.output)
