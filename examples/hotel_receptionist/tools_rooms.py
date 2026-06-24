from __future__ import annotations

import logging
import os
import sys
from datetime import date
from typing import Annotated, Literal

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from book_room import BookRoomTask
from common import Userdata, _count_caller_turns, _speak_code
from context import speech_only
from get_card import GetCardTask
from hotel_db import (
    DISPUTE_POLICIES,
    MAX_PARTY_SIZE,
    PRICING,
    TODAY,
    DisputeCategory,
    DisputePolicy,
    NotFound,
    RoomBooking,
    Unavailable,
    speak_usd,
)
from modify_booking import ModifyBookingTask
from pydantic import Field
from verify_booking import VerifyBookingTask

from livekit.agents import RunContext, ToolError, function_tool

logger = logging.getLogger("hotel-receptionist")

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

class RoomToolsMixin:
    @function_tool
    async def resolve_room_conflict(self, ctx: RunContext[Userdata]) -> str:
        """Fix a double-booked / no-room situation on the caller's verified booking - run this when lookup_booking warned the room is double-booked. It applies the house procedure in fixed order: move the guest to a free room of the same or better category (an upgrade is free), and only if nothing in the house fits, arrange the walk (partner hotel tonight on us, covered taxi, their room back from the return date). Returns the concrete facts; deliver them following the guest_walks policy (own the overbooking, explain plainly why it happened, "at no extra cost to you"). Full procedure: lookup_policy topic "guest_walks"."""
        booking = await self._verified_booking(ctx)
        try:
            r = await ctx.userdata.db.resolve_room_conflict(booking_code=booking.code)
        except (NotFound, Unavailable) as e:
            raise ToolError(str(e)) from None
        if r.moved_to:
            what = "an upgrade, free of charge" if r.upgraded else "same category, no charge"
            return (
                f"resolved: moved to {r.moved_to} - a {r.moved_to_view}-view "
                f"{r.moved_to_type.replace('_', ' ')} ({what}), same dates, total unchanged | "
                "deliver this per the guest_walks policy: own the overbooking and explain plainly "
                "why it happened, then confirm they still have a place for the whole stay, at the "
                "same total, at no extra cost. No further tool call is needed."
            )
        assert r.walk_return_date is not None
        return (
            f"no room in the house fits (every room was checked) - walk arranged at "
            f"{r.walk_partner} (two blocks away, room and taxi both on us), guest's room back "
            f"here {r.walk_return_date.strftime('%A, %B %-d')} | deliver this per the guest_walks "
            "policy: own the overbooking and explain plainly why it happened, then the plan above, "
            "all at no extra cost to them. The guest is angry and will interrupt - give it in short "
            "pieces and make sure every piece lands before the call ends, resuming any that got "
            'talked over. If still upset after the full plan, record a manager callback '
            '(record_followup, kind="callback") before wrapping up.'
        )

    @function_tool
    async def check_room_availability(
        self,
        ctx: RunContext[Userdata],
        check_in: date,
        check_out: date,
        guests: Annotated[int, Field(ge=1, le=MAX_PARTY_SIZE)],
        smoking: Literal["smoking", "non_smoking", "no_preference"],
        room_type: Literal["king", "queen_2beds", "double_queen", "suite", "penthouse", "any"],
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
        avail = await ctx.userdata.db.list_room_types_available(
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
        return " | ".join(
            f"{a.type.replace('_', ' ')}: {speak_usd(a.nightly_rate)} per night, "
            f"{' or '.join(a.views)} view{'s' if len(a.views) > 1 else ''}"
            for a in avail
        )

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
            logger.info("suppressed duplicate room-booking re-entry (no caller turn since %s)", prev.code)
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
    async def lookup_booking(self, ctx: RunContext[Userdata]) -> str:
        """Read-only lookup of a confirmed room booking. Use this when the caller wants to
        check or recall their booking details (dates, room type, what they're paying, who
        it's under, check-in time) without changing anything. Verifies the caller first."""
        b = await self._verified_booking(ctx)
        nights = b.nights
        extras = ", ".join(b.extras) if b.extras else "no extras"
        smoking = "smoking-permitted" if b.smoking else "non-smoking"
        info = (
            f"Booking for {b.first_name} {b.last_name}, {b.room_type.replace('_', ' ')} ({smoking}), "
            f"checking in {b.check_in.strftime('%A %B %-d')} and out {b.check_out.strftime('%A %B %-d')} "
            f"({nights} night{'s' if nights != 1 else ''}, {b.guests} guest{'s' if b.guests != 1 else ''}), "
            f"extras: {extras}. Total {speak_usd(b.total)} on card ending in {b.card_last4}."
        )
        if conflict := await ctx.userdata.db.room_conflict(booking_code=b.code):
            info += (
                f" | WARNING: the room is double-booked {conflict[0].strftime('%B %-d')} to "
                f"{conflict[1].strftime('%B %-d')} - no room is assigned to this booking for that "
                "period. Break the news with ownership and an apology, then run "
                'resolve_room_conflict to fix it (procedure: lookup_policy topic "guest_walks"). '
                "Don't pretend the booking is fine."
            )
        return info

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
            and _count_caller_turns(self.session.history) <= ctx.userdata.caller_turns_at_last_cancel
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
