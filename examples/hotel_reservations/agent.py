from __future__ import annotations

import logging
import os
import sys
from datetime import date
from typing import Annotated, Literal

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# scenarios.yaml dates are literals against this date. hotel_db.TODAY freezes at
# import time, so the pin has to precede every import that reaches hotel_db.
if "--simulation" in sys.argv:
    os.environ.setdefault("HOTEL_TODAY", "2026-06-08")

from benchmark import build_expected, diff_databases
from book_room import BookRoomTask
from common import Userdata, _count_caller_turns, _speak_code, speech_only
from dotenv import load_dotenv
from get_card import GetCardTask
from hotel_db import (
    DISPUTE_POLICIES,
    MAX_PARTY_SIZE,
    PRICING,
    TODAY,
    DisputeCategory,
    DisputePolicy,
    FollowupKind,
    HotelDB,
    NotFound,
    RoomBooking,
    Unavailable,
    describe_room_options,
    speak_usd,
)
from instructions import INSTRUCTIONS
from modify_booking import ModifyBookingTask
from policies import build_lookup_policy_tool
from pydantic import Field
from seed import build_seed_bytes
from verify_booking import VerifyBookingTask

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    SimulationContext,
    ToolError,
    cli,
    function_tool,
    inference,
)
from livekit.agents.evals import (
    JudgeGroup,
    accuracy_judge,
    coherence_judge,
    conciseness_judge,
    handoff_judge,
    relevancy_judge,
    safety_judge,
    task_completion_judge,
    tool_use_judge,
)

load_dotenv(".env.local")

logger = logging.getLogger("hotel-reservations")


def _resolve_dispute_outcome(
    *,
    policy: DisputePolicy,
    amount_cents: int,
    line_item_label: str,
    invoice_line_items: list[tuple[str, int]],
    accepts: bool,
) -> tuple[str, int]:
    action = policy.action
    if action == "auto_refund_if_under_threshold":
        if amount_cents <= PRICING.minibar_auto_refund_threshold:
            return ("auto_refunded", amount_cents)
        return ("credit_offered", amount_cents) if accepts else ("escalated_to_manager", 0)
    if action == "verify_explain_then_offer_credit":
        return ("credit_offered", amount_cents) if accepts else ("escalated_to_manager", 0)
    if action == "explain_no_refund":
        return ("escalated_to_manager", 0) if not accepts else ("explained_no_action", 0)
    if action == "explain_policy_offer_goodwill":
        return ("goodwill_waived", amount_cents) if accepts else ("escalated_to_manager", 0)
    if action == "correct_immediately_or_open_ticket":
        same = sum(
            1
            for label, amt in invoice_line_items
            if label == line_item_label and amt == amount_cents
        )
        if same > 1:
            return ("auto_refunded", amount_cents)
        return ("accounting_ticket_opened", 0)
    return ("open", 0)


def _say_dispute_outcome(
    *,
    outcome: str,
    refund: int,
    case_number: str,
    line_item: str,
    escalation: str,
    policy_explanation: str,
) -> str:
    if outcome == "auto_refunded":
        return (
            f"I've removed the {line_item} charge - that's {speak_usd(refund)} back to the card. "
            f"Case number {_speak_code(case_number)} if you need to reference it."
        )
    if outcome == "credit_offered":
        return (
            f"Applied a {speak_usd(refund)} credit toward the {line_item}. "
            f"Case number {_speak_code(case_number)}."
        )
    if outcome == "goodwill_waived":
        return (
            f"Waived as a one-time courtesy - {speak_usd(refund)} back to the card. "
            f"Case number {_speak_code(case_number)}."
        )
    if outcome == "explained_no_action":
        return f"{policy_explanation}"
    if outcome == "escalated_to_manager":
        return (
            f"I've escalated this to the manager - they'll review and follow up by email. "
            f"Your case number is {_speak_code(case_number)}."
        )
    if outcome == "accounting_ticket_opened":
        return (
            f"I've opened an accounting ticket. They'll investigate and email you within two business days. "
            f"Case number {_speak_code(case_number)}."
        )
    return f"Logged. Case number {_speak_code(case_number)}."


@function_tool
async def record_followup(
    ctx: RunContext[Userdata],
    kind: FollowupKind,
    caller_name: str,
    caller_phone: str,
    summary: str,
) -> str:
    """Capture something for a human to follow up on - sales leads, identity-field change requests (email/phone/name), callback requests, abandoned bookings, verification-failed callers, and any other request you can't handle on this line. ALWAYS use this instead of saying "someone will follow up" with no record; otherwise the request vanishes.

    Args:
        kind: One of sales_lead, identity_change, callback, verification_help, abandoned_booking, other.
        caller_name: Caller's name (ask if you don't already have it).
        caller_phone: Caller's callback number - for an in-house guest, the room number works.
        summary: One sentence describing what they want, with enough detail for a human to act on it.
    """
    code = await ctx.userdata.db.record_followup(
        kind=kind, caller_name=caller_name, caller_phone=caller_phone, summary=summary
    )
    return (
        f"recorded; reference {_speak_code(code)} | read it back so the caller knows it's "
        f"actually on the list: who it's for ({caller_name}, {caller_phone}) and what's noted "
        f'("{summary}"). Don\'t just say "logged", and don\'t promise anyone will follow up or '
        "call back unless that's what was actually recorded."
    )


class ReservationsAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS, tools=[build_lookup_policy_tool()])

    async def on_enter(self) -> None:
        # The caller may have already said what they want before we speak -
        # pick up from there instead of re-asking "how can I help?".
        await self.session.generate_reply(
            instructions=(
                "Greet the caller in one short sentence. If they've already named a need "
                "(a room, a change, a cancellation...), move straight into helping; "
                "otherwise ask how you can help."
            )
        )

    @function_tool
    async def check_room_availability(
        self,
        ctx: RunContext[Userdata],
        check_in: date,
        check_out: date,
        guests: Annotated[int, Field(ge=1, le=MAX_PARTY_SIZE)],
        smoking: Literal["smoking", "non_smoking", "no_preference"],
        room_type: Literal["king", "queen_2beds", "suite", "penthouse", "any"],
    ) -> str:
        """Check what's available for a date range, with prices and views. One tool for every "what do you have?" / "how much?" / "any king available?" / "any smoking rooms?" question. Read-only browsing: it never books anything - when the caller wants to actually book, call start_room_booking instead. Surface the results progressively (types first, details after they narrow), don't recite the whole list.

        Args:
            check_in: Check-in date in ISO YYYY-MM-DD format (e.g. "2026-01-20").
            check_out: Check-out date in ISO YYYY-MM-DD format.
            guests: Number of guests in the room (must be >= 1; ask the caller if not specified).
            smoking: The caller's stated smoking preference, or "no_preference" if they haven't said.
            room_type: The room type the caller picked, or "any" to list everything.
        """
        if check_out <= check_in:
            raise ToolError("check-out must be after check-in")
        smoking_filter = {"smoking": True, "non_smoking": False, "no_preference": None}[smoking]
        avail = await ctx.userdata.db.list_room_options(
            check_in=check_in, check_out=check_out, guests=guests, smoking=smoking_filter
        )
        if room_type != "any":
            avail = [a for a in avail if a.type == room_type]
        if not avail:
            kind = (
                "smoking " if smoking_filter else "non-smoking " if smoking_filter is False else ""
            )
            what = f"{kind}{room_type.replace('_', ' ')}" if room_type != "any" else f"{kind}rooms"
            return f"no {what} available for those dates"
        return describe_room_options(avail)

    @function_tool
    async def start_room_booking(self, ctx: RunContext[Userdata]) -> str | None:
        """Start the room-booking flow. Call it the MOMENT the caller wants to book - never pre-collect name, email, phone, or card yourself first; the flow gathers everything. Its return is the FINAL result of the booking ("You're booked", code, total): relay that to the caller and move on - there is nothing further to confirm or call afterwards."""
        # Guard against a self-inflicted double booking: after a booking completes,
        # the model sometimes re-enters this flow on its own and re-fills every field
        # from the transcript - no caller input - committing a duplicate into another
        # room. If the caller hasn't said a word since the last booking, there is
        # nothing to book: don't start a second flow, point the model back at the one
        # just made. A real second room (a family's extra room, another night) is
        # always preceded by the caller asking, so the legitimate multi-room path
        # stays open. The check is at entry, before any flow-2 chatter can muddy it.
        prev = ctx.userdata.last_room_booking
        if (
            prev is not None
            and _count_caller_turns(self.session.history)
            <= ctx.userdata.caller_turns_at_last_booking
        ):
            logger.info(
                "suppressed duplicate room-booking re-entry (no caller turn since %s)", prev.code
            )
            return (
                f"This booking is already complete - confirmation {_speak_code(prev.code)} was "
                "issued moments ago and you've already given the caller the code and total. Do NOT "
                "book again or repeat the confirmation. If the caller actually wants an ADDITIONAL "
                "room, ask them to confirm that first; otherwise just ask if there's anything else."
            )

        booking = await BookRoomTask(db=ctx.userdata.db, chat_ctx=speech_only(self.chat_ctx))
        ctx.userdata.last_room_booking = booking
        ctx.userdata.caller_turns_at_last_booking = _count_caller_turns(self.session.history)
        logger.info("[stub] would email confirmation to %s for %s", booking.email, booking.code)
        return (
            f"You're booked. Your confirmation code is {_speak_code(booking.code)}. "
            f"Total is {speak_usd(booking.total)}, charged to the card ending in {booking.card_last4}. "
            f"A confirmation email is on its way to {booking.email}. "
            "| booking complete - relay the code and total to the caller; "
            "no further tool call is needed for this booking."
        )

    async def _verified_booking(self, ctx: RunContext[Userdata]) -> RoomBooking:
        """Verify the caller once per call. Tools that mutate the booking
        (modify, cancel) update or clear the cache themselves."""
        if ctx.userdata.verified_booking is None:
            verify = await VerifyBookingTask(
                db=ctx.userdata.db, chat_ctx=speech_only(self.chat_ctx)
            )
            ctx.userdata.verified_booking = verify.booking
        return ctx.userdata.verified_booking

    @function_tool
    async def lookup_booking(self, ctx: RunContext[Userdata]) -> str:
        """Read-only lookup of a confirmed room booking. Use this when the caller wants to
        check or recall their booking details (dates, room type, what they're paying, who
        it's under, check-in time) without changing anything. Verifies the caller first."""
        b = await self._verified_booking(ctx)
        nights = b.nights
        extras = ", ".join(b.extras) if b.extras else "no extras"
        smoking = "smoking-permitted" if b.smoking else "non-smoking"
        return (
            f"Booking for {b.first_name} {b.last_name}, {b.room_type.replace('_', ' ')} ({smoking}), "
            f"checking in {b.check_in.strftime('%A %B %-d')} and out {b.check_out.strftime('%A %B %-d')} "
            f"({nights} night{'s' if nights != 1 else ''}, {b.guests} guest{'s' if b.guests != 1 else ''}), "
            f"extras: {extras}. Total {speak_usd(b.total)} on card ending in {b.card_last4}."
        )

    @function_tool
    async def start_booking_modification(self, ctx: RunContext[Userdata]) -> str:
        """Start the booking-modification flow for an existing reservation. Verifies the caller, then hands off to a focused task that lets them change the stay dates, room type, room view, extras, and party size on the booking - this is the path for a guest unhappy that their room's view or type doesn't match what they booked (it moves them to a matching room). Identity fields (name, email, phone) are NOT modifiable through this flow - record_followup with kind="identity_change" covers those, and a new card goes through start_card_update. NOT for cancellations: if the caller wants to cancel - even after asking for a change first - call cancel_room_booking directly instead."""
        booking = await self._verified_booking(ctx)
        if booking.status != "confirmed":
            raise ToolError("that booking was cancelled - nothing to modify")
        if booking.check_out < TODAY:
            raise ToolError("that stay already ended - can't modify a past booking")

        updated = await ModifyBookingTask(
            db=ctx.userdata.db, existing=booking, chat_ctx=speech_only(self.chat_ctx)
        )
        # Cache the post-modify booking so subsequent tools (lookup, cancel)
        # don't re-verify and don't see the pre-modify state.
        ctx.userdata.verified_booking = updated
        # ModifyBookingTask returns the *same* booking object when the caller
        # made no changes (it completes with `self._existing`); identity check
        # is the cleanest signal that the modify flow was a no-op.
        if updated is booking:
            return (
                "Booking left unchanged | the modification flow is CLOSED - don't re-open it or "
                "re-ask for dates. If the caller pivoted to something else (most often: they "
                "decided to CANCEL instead), do that now with the right tool (cancel_room_booking)."
            )
        delta = updated.total - booking.total
        if delta == 0:
            money = f"total stays at {speak_usd(updated.total)}"
        else:
            direction = "added to" if delta > 0 else "refunded to"
            money = f"new total is {speak_usd(updated.total)}; {speak_usd(abs(delta))} {direction} the card ending in {updated.card_last4}"
        return (
            f"Your booking is updated; {money}. "
            "| modification complete - relay all of this information to the caller (what changed, "
            "the new total, and any amount added or refunded); no further tool call is needed."
        )

    @function_tool
    async def cancel_room_booking(self, ctx: RunContext[Userdata]) -> str:
        """Cancel the caller's room booking. The right tool the moment the caller wants to cancel - including when they pivot mid-modification (staged changes are simply abandoned). Verifies the caller first if not already verified. Returns the refund outcome - relay it exactly as returned; never guess or invent a refund amount or "deposit". When the caller asks "will I lose my deposit if I cancel?" while asking to cancel, this tool's return IS the answer: confirm they want to proceed and run it - don't quote refund policy as if the cancellation already happened and leave the booking standing."""
        # Idempotency: after a successful cancellation the model sometimes re-invokes this
        # with no new caller input. Re-verifying then finds the booking already cancelled and
        # dead-ends in a confusing "did you mean a different reservation?" - while the refund
        # answer it already produced never gets relayed. If a cancel just happened and the
        # caller hasn't spoken since, re-surface that outcome instead of cancelling again.
        # A genuine second cancellation (a different booking) always has a caller turn first.
        if (
            ctx.userdata.caller_turns_at_last_cancel >= 0
            and _count_caller_turns(self.session.history)
            <= ctx.userdata.caller_turns_at_last_cancel
        ):
            return (
                "you already cancelled this booking moments ago - do NOT cancel again or "
                "re-verify. Relay the outcome to the caller and answer their refund/deposit "
                f"question from it: {ctx.userdata.last_cancel_message}"
            )
        booking = await self._verified_booking(ctx)
        if booking.check_in < TODAY:
            raise ToolError("this booking's check-in has already passed; can't cancel a past stay")
        within = (booking.check_in - TODAY).days * 24 < PRICING.cancellation_window_hours
        forfeit = booking.nightly_rate if within else 0
        await ctx.userdata.db.cancel_room_booking(booking.code)
        # Booking is no longer confirmed; the next tool needing a verified
        # booking should re-prompt the caller (a different reservation, or
        # they're done).
        ctx.userdata.verified_booking = None
        if within:
            msg = (
                f"Cancelled. Because the booking's inside the {PRICING.cancellation_window_hours}-hour "
                f"window, one room-night ({speak_usd(forfeit)}) is forfeited; "
                f"I'll refund {speak_usd(booking.total - forfeit)} to the card on file."
            )
        else:
            msg = (
                f"Cancelled - well outside the {PRICING.cancellation_window_hours}-hour window, so "
                f"there's no penalty and no deposit is lost. I'll refund the full "
                f"{speak_usd(booking.total)} to the card on file - usually two to five business days."
            )
        # Remember the outcome + when it happened, so an immediate re-invocation (above)
        # relays this instead of re-verifying a now-cancelled booking.
        ctx.userdata.last_cancel_message = msg
        ctx.userdata.caller_turns_at_last_cancel = _count_caller_turns(self.session.history)
        return msg

    @function_tool
    async def reinstate_booking(self, ctx: RunContext[Userdata]) -> str:
        """Bring back a room booking the caller previously CANCELLED and now wants reactivated. Verifies the caller first - this is the one flow that verifies against a cancelled booking - then checks the booking's original room is still free for its dates and flips it back to confirmed. If the room's been taken since the cancellation, say so honestly and offer to look at other rooms/dates; never silently rebook a different room and call it reinstated. Not for editing a confirmed booking (start_booking_modification) or making a brand-new one (start_room_booking)."""
        verify = await VerifyBookingTask(
            db=ctx.userdata.db, allow_cancelled=True, chat_ctx=speech_only(self.chat_ctx)
        )
        booking = verify.booking
        if booking.status == "confirmed":
            return (
                f"That booking, {_speak_code(booking.code)}, is already active - nothing to "
                "reinstate. Reassure the caller it's all set."
            )
        if booking.check_in < TODAY:
            raise ToolError(
                "that stay's dates have already passed, so it can't be reinstated - offer a new booking"
            )
        try:
            await ctx.userdata.db.reinstate_booking(booking.code)
        except Unavailable:
            raise ToolError(
                "that room has been taken for those dates since the cancellation - tell the caller "
                "honestly and offer to check other rooms or dates (start_room_booking); do NOT claim "
                "it was reinstated"
            ) from None
        ctx.userdata.verified_booking = None
        return (
            f"Reinstated. Booking {_speak_code(booking.code)} is active again - "
            f"{booking.check_in.strftime('%A, %B %-d')} to "
            f"{booking.check_out.strftime('%A, %B %-d')}, total {speak_usd(booking.total)} "
            f"on the card ending {booking.card_last4}. | relay this and move on; nothing "
            "further to call."
        )

    @function_tool
    async def flag_late_arrival(self, ctx: RunContext[Userdata], note: str) -> str:
        """Flag a confirmed booking with an expected late-arrival note ("checking in around 1 AM", "redeye lands at 11 PM"). Verifies the caller first. The note goes onto the booking so the front desk holds the room and doesn't no-show it.

        Args:
            note: A short, concrete description of when the caller expects to arrive (e.g. "around 1 AM" or "after midnight, redeye flight").
        """
        booking = await self._verified_booking(ctx)
        await ctx.userdata.db.flag_late_arrival(booking_code=booking.code, note=note)
        return f"Noted on the booking - we'll hold the room. See you at {note}."

    @function_tool
    async def add_to_waitlist(
        self,
        ctx: RunContext[Userdata],
        first_name: str,
        last_name: str,
        phone: str,
        check_in: date,
        check_out: date,
        guests: Annotated[int, Field(ge=1, le=MAX_PARTY_SIZE)],
    ) -> str:
        """Put the caller on the waitlist for dates the hotel is SOLD OUT on. Use ONLY after check_room_availability has come back empty for their dates and the caller wants to be told if something opens up. Records their name, number, dates, and party size and returns a reference - it does NOT hold or promise a room; the desk reaches out only if a room frees up. Never use it when rooms ARE available (book those instead) and never imply it guarantees anything.

        Args:
            first_name: Caller's first name.
            last_name: Caller's last name.
            phone: Callback number.
            check_in: Requested check-in date in ISO YYYY-MM-DD format.
            check_out: Requested check-out date in ISO YYYY-MM-DD format.
            guests: Number of guests.
        """
        code = await ctx.userdata.db.add_to_waitlist(
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            check_in=check_in,
            check_out=check_out,
            guests=guests,
        )
        return (
            f"waitlisted; reference {_speak_code(code)} | tell the caller they're on the list "
            "for those dates and you'll reach out if something opens up - make clear nothing is "
            "held and it's not a guarantee."
        )

    @function_tool
    async def record_group_inquiry(
        self,
        ctx: RunContext[Userdata],
        company: str,
        contact_name: str,
        contact_phone: str,
        party_size: Annotated[int, Field(ge=15)],
        share_type: Literal["twin", "double", "single", "mixed"],
        check_in: date,
        nights: Annotated[int, Field(ge=1)],
    ) -> str:
        """Open a room-block inquiry for a group of 15 or more guests (tours, teams, conferences). This records the inquiry for the group desk - it does NOT confirm or hold rooms, and you cannot confirm a group on this call no matter how hard the caller pushes; a new sponsor needs credit approval first. Call this the MOMENT you have all the arguments - if the caller asks more questions while you're collecting, record the inquiry first and answer after; an unrecorded inquiry is lost when the call ends. For the terms to quote (group rate, tour-leader comp, cancellation), call lookup_policy with topic "group_bookings" first. Under 15 guests, use the normal booking flow instead.

        Args:
            company: The sponsoring company or organization (ask who the group is with).
            contact_name: Full name of the group's contact person.
            contact_phone: The contact's callback number, as the caller gave it.
            party_size: Total number of guests in the group (15 or more).
            share_type: The predominant room-share arrangement the caller described - "mostly twin-share" records as twin; use mixed only if no single arrangement dominates.
            check_in: Group arrival date in ISO YYYY-MM-DD format.
            nights: Number of nights the group stays.
        """
        code = await ctx.userdata.db.record_group_inquiry(
            company=company,
            contact_name=contact_name,
            contact_phone=contact_phone,
            party_size=party_size,
            share_type=share_type,
            check_in=check_in,
            nights=nights,
        )
        return (
            f"group inquiry recorded; reference {_speak_code(code)} | nothing is confirmed yet: "
            "tell the caller the group desk will call them back within two business days, "
            "after credit review, to confirm the block."
        )

    @function_tool
    async def lookup_guest_history(self, ctx: RunContext[Userdata], last_name: str) -> str:
        """Look up a returning guest's remembered preferences from past stays (floor/room preferences, bedding, known sensitivities). Use it when a caller presents as a repeat/returning guest ("booking another stay", "I've stayed before") or you otherwise recognize them, so you can proactively offer to set up what they've liked before. Returns their on-file preferences, or says there's no history. Only ever surface preferences this returns - never invent or assume preferences not on file - and only for the guest themselves.

        Args:
            last_name: The returning guest's last name.
        """
        prefs = await ctx.userdata.db.lookup_guest_history(last_name=last_name)
        if not prefs:
            return (
                "No guest history on file for that name - treat them as a new guest and don't "
                "invent past preferences."
            )
        return (
            f"On file: {prefs} | proactively offer to set these up again for the new stay, and "
            "apply or note the ones the guest confirms. Don't add any preference beyond these."
        )

    @function_tool
    async def start_card_update(self, ctx: RunContext[Userdata]) -> str:
        """Replace the card on file for an existing room booking - the path when a guest's card isn't going through or they want a different card charged. Verifies the caller first, then a focused sub-task collects the replacement card (number, expiry, security code, cardholder) with its own read-back - never collect card digits yourself. Call this as your response once the caller offers a new card. Keep the money talk discreet throughout: the card "isn't going through at the moment - possibly a technical issue", never "declined" or "rejected" (full policy: lookup_policy topic "payments_and_currency")."""
        booking = await self._verified_booking(ctx)
        if booking.status != "confirmed":
            raise ToolError(
                f"booking {booking.code} is {booking.status} - there's no active booking to update"
            )
        card = await GetCardTask(chat_ctx=speech_only(self.chat_ctx))
        await ctx.userdata.db.update_booking_card(
            booking_code=booking.code, card_last4=card.card_number[-4:]
        )
        return (
            f"card on file updated to the one ending {card.card_number[-4:]} | confirm to the "
            "caller that the new card is on the booking and everything is set for their stay; "
            "no further tool call is needed for this."
        )

    @function_tool
    async def lookup_invoice(self, ctx: RunContext[Userdata]) -> str:
        """Verify the caller, fetch their invoice, and read it back."""
        booking = await self._verified_booking(ctx)
        invoice = await ctx.userdata.db.get_invoice(booking.code)
        items = ", ".join(f"{li.label} {speak_usd(li.amount_cents)}" for li in invoice.line_items)
        return (
            f"That booking's total is {speak_usd(invoice.total)}, with line items: "
            f"{items}. I can email an itemized copy to the address on file, {booking.email}, if you'd like - just say the word."
        )

    @function_tool
    async def dispute_charge(
        self,
        ctx: RunContext[Userdata],
        category: DisputeCategory,
        line_item_label: str,
        caller_note: str,
        accepts_offered_resolution: bool,
    ) -> str:
        """Handle a guest dispute on a line item.

        Args:
            category: Pick the category that best matches what the caller is disputing.
            line_item_label: The label of the line item on the invoice, as it appears.
            caller_note: A short summary of what the caller said about the charge.
            accepts_offered_resolution: Required. Set true ONLY after the caller has actually
                accepted the policy outcome you offered (a goodwill waiver, a credit, etc.).
                Set false if they pushed back, asked for a manager, or haven't been offered
                anything yet. Never default to true to skip the conversation.
        """
        if category not in DISPUTE_POLICIES:
            raise ToolError(f"unknown dispute category: {category}")
        policy = DISPUTE_POLICIES[category]

        booking = await self._verified_booking(ctx)
        invoice = await ctx.userdata.db.get_invoice(booking.code)

        # Match labels case-insensitively so an LLM mistranscription like
        # "Late checkout" vs "late checkout" still resolves to the real line.
        target = line_item_label.casefold()
        item = next((li for li in invoice.line_items if li.label.casefold() == target), None)
        if item is None:
            raise ToolError(
                f"No line item labelled {line_item_label!r} on that invoice. "
                "Read the line items back and ask the caller to pick one."
            )

        amount = item.amount_cents
        outcome, refund = _resolve_dispute_outcome(
            policy=policy,
            amount_cents=amount,
            line_item_label=item.label,
            invoice_line_items=[(li.label, li.amount_cents) for li in invoice.line_items],
            accepts=accepts_offered_resolution,
        )

        case_number = await ctx.userdata.db.file_dispute(
            booking_code=booking.code,
            line_item=item.label,
            amount_cents=amount,
            category=category,
            caller_note=caller_note,
            outcome=outcome,
            refund_amount=refund,
        )

        return _say_dispute_outcome(
            outcome=outcome,
            refund=refund,
            case_number=case_number,
            line_item=item.label,
            escalation=policy.escalation,
            policy_explanation=policy.explanation,
        )

    @function_tool
    async def resend_confirmation(
        self,
        ctx: RunContext[Userdata],
        kind: Literal["booking_confirmation", "folio"],
    ) -> str:
        """Re-send a document for an existing booking to the email already on file for it - the booking confirmation, or an itemized folio of the stay. Verifies the caller first (this hits their account). It only ever goes to the address on record; there is no way to send it to a different address the caller reads out - if they want it somewhere else, their contact email on the booking has to be updated first (record_followup, kind="identity_change"). This actually sends - only tell the caller it's on its way after this returns.

        Args:
            kind: Which document to re-send.
        """
        booking = await self._verified_booking(ctx)
        await ctx.userdata.db.send_email(recipient=booking.email, kind=kind)
        return f"Sent to the address on file, {booking.email.strip().lower()}."

    @function_tool
    async def transfer_call(
        self,
        ctx: RunContext[Userdata],
        destination: Literal["restaurant", "duty_manager", "housekeeping"],
        summary: str,
    ) -> str:
        """Transfer the caller to a hotel DEPARTMENT - the restaurant, the duty manager, or housekeeping. NOT a guest's room (never connect a caller to a guest). Before calling this you must have told the caller you're putting them on hold to connect them to that department AND gotten their okay; only then transfer. Pass a one-line summary of what the caller needs so the department is briefed.

        Args:
            destination: The department to transfer to.
            summary: A one-line summary of what the caller needs.
        """
        # A transfer happens exactly once. If the agent re-calls this (the caller reacts
        # and it "re-confirms", or it retries after thinking the first failed), don't write
        # a second transfer row - the deterministic grader counts rows, and a duplicate
        # fails the run. Just reassure the caller they're being connected.
        if destination in ctx.userdata.transferred_to:
            return (
                f"already transferred to the {destination.replace('_', ' ')} on this call - do NOT "
                "transfer again. Just briefly reassure the caller they're being connected."
            )
        try:
            await ctx.userdata.db.transfer_call(destination=destination, summary=summary)
        except NotFound as e:
            raise ToolError(str(e)) from None
        ctx.userdata.transferred_to.add(destination)
        # Don't disconnect the session here: the caller may have a last reaction, and going
        # silent mid-call reads as a hang (the conversation ends when the caller is done, not
        # when we drop off). Close out briefly instead so the call can wrap up naturally.
        return (
            f"Transferred to the {destination.replace('_', ' ')} - your part of the call is done. "
            'Give ONE short closing hand-off ("You\'re all set - connecting you now"), NOT '
            '"anything else?", so the call can wrap up. Do NOT transfer again or take the request '
            "down as a followup; if the caller reacts, keep it to a brief acknowledgement and "
            "don't reopen the conversation."
        )


server = AgentServer()

_SEED_DB_BYTES = build_seed_bytes(TODAY)


def _expected_state_statements(userdata: dict[str, object]) -> list[str] | None:
    """No key means the DB isn't checked; null or [] means it must come back unchanged."""
    if "expected_state" not in userdata:
        return None
    statements = userdata["expected_state"]
    if statements is None:
        return []
    if not isinstance(statements, list) or not all(isinstance(s, str) for s in statements):
        raise TypeError("expected_state must be a list of SQL statements")
    return statements


async def on_simulation_end(ctx: SimulationContext) -> None:
    tagger = ctx.job_context.tagger
    failure: str | None = None
    graded = False
    try:
        expected_state = _expected_state_statements(ctx.userdata())
        graded = expected_state is not None
        if expected_state is not None:
            # Grade the run on final DB state: build the scenario's `expected_state` on a
            # fresh seed, then diff it against the agent's DB. The diff compares
            # agent-decided facts only (room type, dates, extras, status), so minted
            # codes / order / which-king don't matter and the agent need not reproduce the
            # statements — while collateral damage still surfaces.
            session = ctx.job_context.primary_session
            expected = await build_expected(_SEED_DB_BYTES, expected_state)
            try:
                diffs = diff_databases(expected.connection, session.userdata.db.connection)
            finally:
                await expected.aclose()
            if diffs:
                failure = "final DB diverges from expected: " + " | ".join(diffs[:8])
    except Exception as exc:
        # Grading that can't run is not a pass. The prefix tells a broken scenario
        # apart from a failing agent.
        logger.exception("expected-state grading failed")
        failure = f"expected-state grading failed: {exc}"

    # Most scenarios skip the DB check on purpose, so the run has to record
    # whether it ran at all.
    tagger.add("state:graded" if graded else "state:ungraded")

    # A run passes only if both the conversation and the DB check pass. A scenario
    # with no DB check is graded on the conversation alone.
    if failure:
        ctx.fail(reason=failure)
        tagger.fail(reason=failure)
    elif ctx.simulator_verdict.success:
        tagger.success(reason=ctx.simulator_verdict.reason)
    else:
        tagger.fail(reason=ctx.simulator_verdict.reason)


async def on_session_end(ctx: JobContext) -> None:
    try:
        report = ctx.make_session_report()
    except RuntimeError:
        return

    chat = report.chat_history.copy(exclude_function_call=True, exclude_instructions=True)
    if len(chat.items) < 3:
        return

    judges = JudgeGroup(
        llm="openai/gpt-4.1-mini",
        judges=[
            task_completion_judge(),
            accuracy_judge(),
            tool_use_judge(),
            handoff_judge(),
            safety_judge(),
            relevancy_judge(),
            coherence_judge(),
            conciseness_judge(),
        ],
    )
    await judges.evaluate(report.chat_history)

    userdata = ctx.primary_session.userdata

    logger.info("session tags: %s", ctx.tagger.tags)

    try:
        await userdata.db.aclose()
    except Exception:
        logger.exception("error closing hotel DB")


@server.rtc_session(on_session_end=on_session_end, on_simulation_end=on_simulation_end)
async def hotel_reservations_agent(ctx: JobContext) -> None:
    await ctx.connect()

    userdata = Userdata(db=HotelDB.from_bytes(_SEED_DB_BYTES))
    session = AgentSession[Userdata](
        userdata=userdata,
        # Session-scoped, so every agent and every AgentTask in the call can reach
        # it - including the name / email / phone dialogs. A caller can abandon
        # anywhere, and the alternative to recording the callback where they say so
        # is promising one that was never written.
        tools=[record_followup],
        # An explicit VAD is required (not the bundled default): without it the
        # speaking anchor falls back to the STT stream clock, which drifts into the
        # future across a long call / nested-task switch and makes the turn-commit
        # logic sleep for that offset (~the elapsed call time) before replying.
        vad=inference.VAD(model="silero"),
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("google/gemma-4-31b-it"),
        tts=inference.TTS("inworld/inworld-tts-2"),
        max_tool_steps=5,
    )

    await session.start(agent=ReservationsAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
