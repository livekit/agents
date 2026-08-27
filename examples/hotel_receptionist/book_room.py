from __future__ import annotations

from datetime import date
from enum import Enum, auto
from typing import Annotated

from common import _count_caller_turns
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
    describe_extras,
    describe_room_options,
    speak_usd,
)
from persona import COMMON_INSTRUCTIONS
from pydantic import Field

from livekit.agents import NOT_GIVEN, NotGivenOr, beta
from livekit.agents.llm import ChatContext
from livekit.agents.llm.tool_context import ToolError, ToolFlag, function_tool
from livekit.agents.voice.agent import AgentTask

_BOOK_ROOM_INSTRUCTIONS = """\
You're handling a room booking from start to finish.

Before asking anything, scan the conversation so far. If dates, room type, party size, or smoking preference were already discussed, call the matching recording tools (set_stay, choose_room) right away with those values.
"""


class _Owed:
    """Speech the flow owes the caller before it may move on.

    Armed where the flow incurs the obligation and discharged by the caller's
    next turn, so the model speaks and the turn ends there - the caller's reply
    is what reopens the tools.

    Reading the caller's turns out of the history is what keeps the gate the same
    for a caller who types as for one who speaks: `on_user_turn_completed` runs
    only on the audio end-of-turn path, so an obligation cleared from that hook
    never clears for text input and the flow never leaves the step.
    """

    def __init__(self, *, armed: bool) -> None:
        self._armed = armed
        self._at: int | None = None

    def arm(self) -> None:
        self._armed, self._at = True, None

    def pending(self, turns: int) -> bool:
        if not self._armed:
            return False
        if self._at is None:
            # Stamped when the flow first sees the obligation pending, not when it
            # is armed: the read-back is incurred while details are still being
            # captured, and the caller turns spent capturing them are not answers
            # to it.
            self._at = turns
        if turns > self._at:
            self._armed = False
        return self._armed


class _Step(Enum):
    """Where the booking stands. Derived from the captured values on every read,
    so a correction lands the flow back on the right step with no rollback
    bookkeeping to keep in sync."""

    NEED_STAY = auto()
    OFFERING = auto()
    NEED_ROOM = auto()
    NEED_EXTRAS = auto()
    CAPTURE = auto()
    READ_BACK = auto()
    AWAIT_AGREEMENT = auto()


# What each step accepts. Every tool stays listed to the model - a call outside
# its step returns _closed() instead of running, so choose_room can't land
# before the options have been offered and confirm_booking can't land before the
# read-back has been answered. CAPTURE adds the dialogs for the details still
# missing, leaving a captured detail with no dialog to re-ask it with. set_stay
# and choose_room stay open past their own step so a caller can correct them.
_RECORDERS = frozenset({"set_stay", "choose_room", "set_extras"})
_ALLOWED: dict[_Step, frozenset[str]] = {
    _Step.NEED_STAY: frozenset({"set_stay"}),
    _Step.OFFERING: frozenset({"set_stay"}),
    _Step.NEED_ROOM: frozenset({"choose_room", "set_stay"}),
    _Step.NEED_EXTRAS: _RECORDERS,
    _Step.CAPTURE: _RECORDERS,
    _Step.READ_BACK: _RECORDERS,
    _Step.AWAIT_AGREEMENT: _RECORDERS | {"confirm_booking"},
}


class BookRoomTask(AgentTask[RoomBooking]):
    """The entire room booking as one focused task. `set_stay` / `choose_room`
    handle the part with real coupling - dates <-> availability <-> room - and the
    `open_*_dialog` tools capture each independent detail the moment it's
    offered, storing it on the draft so a later hiccup never re-asks it.
    `confirm_booking()` takes the card, writes the booking, and completes with it.

    Which of those will actually run is `_ALLOWED[self._step()]`, checked by
    each tool through `_closed()`: the sequencing is enforced where the call
    lands rather than requested in instructions the model has to remember."""

    def __init__(self, db: HotelDB, *, chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN) -> None:
        self._db = db
        self._check_in: date | None = None
        self._check_out: date | None = None
        self._guests: int | None = None
        self._room_type: RoomType | None = None
        self._view: str | None = None
        self._extras: list[RoomExtra] = []
        # An empty extras list is a real answer, indistinguishable from never having
        # asked - so the answer is tracked separately from the value. No total is
        # quoted until it's True, since every extra moves the total.
        self._extras_set: bool = False
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
        self._must_offer = _Owed(armed=False)
        self._must_read_back = _Owed(armed=True)
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

    def _missing(self) -> list[tuple[str, str]]:
        """The dialog tool and its directive for each detail still uncaptured, in ladder order."""
        return [
            (tool, directive)
            for captured, tool, directive in (
                (
                    bool(self._first_name and self._last_name),
                    "open_name_dialog",
                    "stay and room captured - next: call open_name_dialog",
                ),
                (
                    bool(self._email),
                    "open_email_dialog",
                    "name captured - next: call open_email_dialog",
                ),
                (
                    bool(self._phone),
                    "open_phone_dialog",
                    "email captured - next: call open_phone_dialog",
                ),
                (
                    bool(self._card_last4),
                    "open_credit_card_dialog",
                    "phone captured - next: call open_credit_card_dialog",
                ),
            )
            if not captured
        ]

    def _step(self) -> _Step:
        if self._check_in is None:
            return _Step.NEED_STAY
        turns = _count_caller_turns(self.session.history)
        if self._must_offer.pending(turns):
            return _Step.OFFERING
        if self._room_type is None:
            return _Step.NEED_ROOM
        if not self._extras_set:
            return _Step.NEED_EXTRAS
        if self._missing():
            return _Step.CAPTURE
        if self._must_read_back.pending(turns):
            return _Step.READ_BACK
        return _Step.AWAIT_AGREEMENT

    def _allowed(self, step: _Step) -> frozenset[str]:
        tools = _ALLOWED[step]
        if step is _Step.CAPTURE:
            tools |= {tool for tool, _ in self._missing()}
        return tools | {"give_up"}

    async def _requote(self) -> None:
        """Recompute the exact total (with tax) for the room that will be booked, so
        the read-back uses the real number rather than per-night arithmetic. Stays
        None until the extras are answered - a total quoted before then is one that
        changes the moment the caller says "and breakfast"."""
        if not (self._extras_set and self._room_type and self._check_in and self._check_out):
            self._quoted_total = None
            return
        assert self._guests is not None
        self._quoted_total = await self._db.peek_stay_total(
            room_type=self._room_type,
            smoking=self._smoking,
            guests=self._guests,
            check_in=self._check_in,
            check_out=self._check_out,
            view=self._view,
            extras=self._extras,
        )

    def _closed(self, tool: str) -> str | None:
        """The refusal for a tool called outside its step, or None when it's open.

        The refusal has to name the outcome. A bare "do X next" reads as a report
        of progress, and the model relays it to the caller as though the call had
        gone through - which for confirm_booking means speaking a confirmation
        code for a reservation that was never written.
        """
        allowed = self._allowed(self._step())
        if tool in allowed:
            return None
        return (
            f"{tool} did NOT run - nothing was recorded, no booking was made, and no "
            f"confirmation code exists. Available right now: {', '.join(sorted(allowed))}. "
            f"{self._status()}"
        )

    def _status(self) -> str:
        # Action-oriented status, NOT a missing-field list. A "still need: card"
        # string gets parroted by the model as "What card should I use?" - the
        # field name leaks straight into the spoken question. Phrasing each
        # step as the next action avoids that.
        step = self._step()
        if step is _Step.NEED_STAY:
            return "no stay yet - ask the caller for dates and party size, then call set_stay"
        if step is _Step.OFFERING:
            return (
                "offer these room types to the caller and ask which one they want - "
                "choose_room stays closed until they've answered"
            )
        if step is _Step.NEED_ROOM:
            return "stay captured - ask which room type, then call choose_room"
        if step is _Step.NEED_EXTRAS:
            return (
                "room captured, no total yet - offer the extras and ask which the caller "
                "wants, then call set_extras (empty list if they want none). Each extra "
                "moves the total, so the total only exists once this is answered"
            )
        if step is _Step.CAPTURE:
            return self._missing()[0][1]
        if step is _Step.AWAIT_AGREEMENT:
            return "the read-back is done - call confirm_booking() the moment the caller agrees"
        total = (
            f"total {speak_usd(self._quoted_total)} including tax, " if self._quoted_total else ""
        )
        return (
            "all required details captured - read the booking back in one sentence "
            f"(dates, {self._guests} guests, room and extras, {total}card ending "
            f"{self._card_last4}) and call confirm_booking() the moment the caller agrees. "
            "Quote ONLY this total - never compute your own."
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
        if closed := self._closed("set_stay"):
            return closed
        if check_out <= check_in:
            raise ToolError("check-out must be after check-in")
        if (check_out - check_in).days > 30:
            raise ToolError("the max stay is 30 nights")
        if check_in < TODAY:
            raise ToolError("check-in can't be in the past")

        avail = await self._db.list_room_options(
            check_in=check_in, check_out=check_out, guests=guests
        )
        if not avail:
            # Don't persist sold-out dates as the active stay - if the model
            # drifts forward without re-setting, the booking would carry
            # invalid dates. The caller needs to pick different dates anyway.
            return f"sold out for {check_in} to {check_out}, {guests} guests - dates not recorded; ask for adjacent dates"

        self._check_in, self._check_out, self._guests = check_in, check_out, guests
        # New dates invalidate any read-back already given.
        self._must_read_back.arm()
        # The type+view pairing is what was picked and what carries the rate, so both
        # have to still be open for the choice to survive the new dates.
        had_room = self._room_type is not None
        survives = had_room and any(
            a.type == self._room_type and (self._view is None or a.view == self._view)
            for a in avail
        )
        if not survives:
            self._room_type = self._view = None
            # Options have to be spoken before a room can be picked, so the offer is owed
            # only where a pick actually died:
            # - pick invalidated by the new dates: the caller has to choose again.
            # - pick that survives: already offered and picked, and re-arming over it
            #   rewinds a settled step - where a date corrected at the read-back lands.
            # - no pick yet: the flow's first set_stay has nothing to invalidate; the
            #   offer behind its values predates the flow, and recording them is what
            #   this call and choose_room are for.
            if had_room:
                self._must_offer.arm()
        # Per-night extras and the room both reprice with the night count.
        await self._requote()
        return (
            f"stay recorded ({check_in} to {check_out}, {guests} guests)\n"
            f"options (one line per room type + view - the price is that pairing's, so "
            f"the view is part of what the caller is picking):\n"
            f"{describe_room_options(avail)}\n{self._status()}"
        )

    @function_tool()
    async def choose_room(
        self,
        room_type: RoomType,
        smoking_room: bool = False,
        view: str | None = None,
    ) -> str:
        """Record the room type and view the caller picked from the options set_stay returned.

        Call ONLY after the caller has named a room type (a stated view narrows WHICH room of that type they get - it doesn't pick the type). If the caller asks for a view, pass it here; if that view isn't available for the type, this errors with where the view IS available - relay that and let them choose. Never guess a type from a preference.

        This does NOT produce a total - extras are still open, and each one moves it. The return lists the extras to offer; call set_extras once the caller has answered.

        Args:
            room_type: The room type exactly as the caller chose it.
            smoking_room: True if the caller wants a smoking-permitted room.
            view: The view the caller asked for (city / garden / ocean), ONLY if they stated one - omit entirely otherwise.
        """
        if closed := self._closed("choose_room"):
            return closed
        if self._check_in is None or self._check_out is None or self._guests is None:
            raise ToolError("stay dates and guest count not yet recorded")
        # Re-check against availability filtered by the smoking preference: a
        # type may have rooms free, but not a smoking (or non-smoking) one.
        avail = await self._db.list_room_options(
            check_in=self._check_in,
            check_out=self._check_out,
            guests=self._guests,
            smoking=smoking_room,
        )
        for_type = [a for a in avail if a.type == room_type]
        if not for_type:
            kind = "smoking " if smoking_room else ""
            offer = ", ".join(sorted({a.type for a in avail})) or "nothing for those dates"
            raise ToolError(f"no {kind}{room_type} available; offer one of: {offer}")
        # Models sometimes send placeholder strings for optional args they
        # should omit - normalize those to "no view preference".
        if view is not None:
            view = view.strip().casefold()
            if view in ("", "null", "none", "any", "no preference", "unspecified"):
                view = None
        if view is not None and view not in {a.view for a in for_type}:
            raise ToolError(
                f"no {view}-view {room_type.replace('_', ' ')} for those dates - "
                f"the pairings open are:\n{describe_room_options(avail)}\n"
                "Tell the caller and let them choose."
            )
        self._room_type = room_type
        self._view = view
        self._smoking = smoking_room
        self._must_read_back.arm()  # a different room means a different read-back
        await self._requote()
        rate = min(a.nightly_rate for a in for_type if view is None or a.view == view)
        view_part = f" with a {view} view" if view else ""
        nights = (self._check_out - self._check_in).days
        return (
            f"room recorded: {room_type.replace('_', ' ')}{view_part} at "
            f"{speak_usd(rate)}/night\nextras for this {nights}-night stay - offer these "
            f"and get an answer before any total, since each one moves it:\n"
            f"{describe_extras(nights)}\n{self._status()}"
        )

    @function_tool()
    async def set_extras(
        self, breakfast: bool, valet: bool, late_checkout: bool, pets: bool
    ) -> str:
        """Record the caller's answer on each extra, after you've offered them.

        Every extra takes an explicit true or false, so pass false for the ones the caller turned down - all four false is a real answer and settles the question. Don't call this to skip the offer.

        The stay's total is computed here, because every extra moves it. This return is where the number for the read-back comes from.

        Args:
            breakfast: True if the caller wants breakfast added.
            valet: True if the caller wants valet parking.
            late_checkout: True if the caller wants a late checkout.
            pets: True if the caller is bringing a pet.
        """
        if closed := self._closed("set_extras"):
            return closed
        # One boolean per extra rather than a list: the model has to answer for every
        # one, so an extra it never raised with the caller can't be quietly left out
        # of an array the way an omitted list member is.
        answers: tuple[tuple[RoomExtra, bool], ...] = (
            ("breakfast", breakfast),
            ("valet", valet),
            ("late_checkout", late_checkout),
            ("pets", pets),
        )
        extras = [name for name, wanted in answers if wanted]
        self._extras = extras
        self._extras_set = True
        self._must_read_back.arm()  # different extras mean a different read-back
        await self._requote()
        chosen = ", ".join(e.replace("_", " ") for e in extras) if extras else "none"
        total_part = (
            f"; total for the stay {speak_usd(self._quoted_total)} including tax"
            if self._quoted_total
            else ""
        )
        return f"extras recorded: {chosen}{total_part} | {self._status()}"

    @function_tool()
    async def open_name_dialog(self) -> str:
        """Open the name dialog. It collects the guest's first and last name (read back and confirmed) from the caller."""
        if closed := self._closed("open_name_dialog"):
            return closed
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
        if closed := self._closed("open_email_dialog"):
            return closed
        r = await beta.workflows.GetEmailTask(
            chat_ctx=speech_only(self.chat_ctx), extra_instructions=COMMON_INSTRUCTIONS
        )
        self._email = r.email_address
        return f"email recorded: {self._email} | {self._status()}"

    @function_tool()
    async def open_phone_dialog(self) -> str:
        """Open the phone dialog. It collects the guest's phone number (read back and confirmed) from the caller."""
        if closed := self._closed("open_phone_dialog"):
            return closed
        r = await beta.workflows.GetPhoneNumberTask(
            chat_ctx=speech_only(self.chat_ctx), extra_instructions=COMMON_INSTRUCTIONS
        )
        self._phone = r.phone_number
        return f"phone recorded: {self._phone} | {self._status()}"

    @function_tool()
    async def open_credit_card_dialog(self) -> str:
        """Open the credit-card dialog. It collects the card number, expiry, security code, and cardholder name from the caller in one focused step."""
        if closed := self._closed("open_credit_card_dialog"):
            return closed
        card = await GetCardTask(chat_ctx=speech_only(self.chat_ctx))
        self._card_last4 = card.card_number[-4:]
        return f"card recorded (ending {self._card_last4}) | {self._status()}"

    @function_tool()
    async def confirm_booking(self) -> str | None:
        """Finalize the booking and charge the card. Call ONLY after every detail is captured AND the caller has agreed to your read-back (dates, room and extras, total, card last four). Returns the final confirmation - relay it to the caller; the booking flow ends with this call."""
        if closed := self._closed("confirm_booking"):
            return closed
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
        """End the booking without making it: the caller no longer wants the room, OR they need something this flow can't do. The right tools for their request become available once this returns.

        Args:
            reason: short explanation.
        """
        if not self.done():
            self.complete(ToolError(f"booking abandoned: {reason}"))
