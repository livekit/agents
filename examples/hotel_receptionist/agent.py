from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark import build_expected, diff_databases
from book_restaurant import BookRestaurantTask
from book_room import BookRoomTask
from dotenv import load_dotenv
from fake_data.seed import build_seed_bytes
from get_consent import GetRecordingConsentTask
from hotel_db import (
    DISPUTE_POLICIES,
    MAX_PARTY_SIZE,
    PRICING,
    TODAY,
    DisputeCategory,
    DisputePolicy,
    FollowupKind,
    HotelDB,
    RoomBooking,
    format_usd,
    speak_time,
    speak_usd,
)
from modify_booking import ModifyBookingTask
from persona import COMMON_INSTRUCTIONS
from pydantic import Field
from ui_view import UiView
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
from livekit.agents.voice import UserStateChangedEvent, presets
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

logger = logging.getLogger("hotel-receptionist")


_HOTEL_INFO = """\
Address: 100 LiveKit Way, San Francisco.
Airport: SFO is roughly 30 minutes by car. No hotel shuttle; the front desk will arrange a ride.
Getting around: nearest Muni stop is two blocks away; BART is a 10-minute walk. Cabs and rideshares pick up at the main entrance.
Neighborhood: a few coffee shops and a 24-hour pharmacy within two blocks. The nearest hospital is six blocks east; non-emergency urgent care five blocks south.
Things to do nearby: walkable to the waterfront and the main shopping street; the front desk keeps a list of dinner spots, museums, and tour operators for guests who ask.
Rooms: 55-inch TV, mini-fridge, safe, iron, hair dryer, Nespresso, blackout curtains. King beds in most rooms; suites have a separate sitting area.
Cribs and rollaway beds: free on request, subject to availability - mention it at booking or call ahead.
Accessibility: ADA-accessible rooms on every floor, roll-in showers in the suites. Mention at booking so we assign one.
Connecting rooms: available on request, subject to availability.
Laundry and dry-cleaning: drop at the front desk before 9 AM for same-day return, priced per item.
Lost-and-found: held at the front desk for 90 days.
Business center: 24/7 lobby workstations with printing.
Spa: not on-site. The front desk can recommend places nearby.
"""


_RESTAURANT_INFO = """\
Menu: standard dinner fare - starters and salads, mains (salmon, chicken, steak, pasta, burger, vegetarian risotto), sides, desserts, full bar. Specific dish prices rotate and I don't keep them memorized; if the caller asks about a particular dish or price I don't have, offer to note the question for the kitchen via record_followup (kind="other").
Dietary and allergies: vegetarian and most dietary needs handled. For severe or anaphylactic allergies, the kitchen needs to know at the reservation.
Dress code: smart casual. No jacket required.
Seating: indoor dining room, outdoor terrace, and a bar. Children welcome.
Reservations: bar walk-ins fine anytime; tables are reservation-only on weekends.
Private dining: separate room seats up to twelve. Advance reservation required.
Room service: same menu as the restaurant, 5:30 to 9:30 PM.
Takeout and delivery: not offered.
Celebrations: mention a birthday or anniversary at the reservation and the kitchen sends out a small dessert.
"""


@dataclass
class Userdata:
    db: HotelDB
    consent: bool | None = None
    booked_room_codes: list[str] = field(default_factory=list)
    booked_restaurant_codes: list[str] = field(default_factory=list)
    cancelled_codes: list[str] = field(default_factory=list)
    verified_booking: RoomBooking | None = None


def _instructions() -> str:
    return f"""\
{COMMON_INSTRUCTIONS}

You're the lead receptionist, holding the whole call and routing each request to the right tool. Help the caller with whatever they bring - if a request fits a tool, run it; if it's general (a policy, a fact, recalling their stay), answer from what you know.

# Quick facts (answer directly - no tool call needed)
- Check-in 3 PM, check-out 11 AM. Late checkout until 2 PM is {format_usd(PRICING.late_checkout)}, subject to availability. Early check-in is on a same-day, ask-housekeeping basis.
- Late arrival is fine; the room is held all night as long as the booking is confirmed. ID at check-in: a government-issued photo ID (driver's license or passport for international guests).
- Pets: pet-friendly rooms only, {format_usd(PRICING.pet_fee)} per stay. Service animals always welcome at no charge.
- Smoking: smoking-permitted rooms on request; {format_usd(PRICING.smoking_cleaning_fee)} cleaning fee for smoking in a non-smoking room.
- Self-parking free; valet {format_usd(PRICING.valet_per_night)} per night.
- Wi-Fi free. Pool, gym, sauna 6 AM to 10 PM, towels provided, free for guests.
- Cancellation: free up to {PRICING.cancellation_window_hours} hours before check-in; inside that window, one night is forfeited. Tax is {PRICING.tax_rate_pct}% on room and extras.
- Breakfast buffet in the restaurant, 6:30 to 10:30 AM, {format_usd(PRICING.breakfast_per_night)} a night when added as a room extra.
- Restaurant: on-site, dinner only, 5:30 to 9 PM last seating.
- Luggage hold at the front desk before check-in and after check-out, no charge.

# Routing the call
- Browse without booking: check_room_availability (rate + view + optional smoking/room_type filters), check_restaurant_availability, lookup_booking, lookup_restaurant_reservation. None of these change anything.
- Caller wants to book: start_room_booking or start_restaurant_booking - the call IS your response, not something after an acknowledgment. Don't ask the caller for name, email, phone, or card without one of these running - that's the only path that creates a booking.
- Existing booking changes: start_booking_modification (dates, room, extras, party size). Cancel via cancel_room_booking. Late arrival ("I'll be in past midnight") -> flag_late_arrival with a short note.
- Existing restaurant reservation: change isn't supported directly - with the caller's permission, cancel the old one and book a new slot. Be explicit about the two steps before doing them.
- Sold out: offer adjacent dates or another room type. One tool call per turn; finish each tool's flow before starting another.
- Detail beyond the quick facts: get_hotel_info (address, transport, amenities, accessibility, etc.) or get_restaurant_info (menu, dietary, dress, room service, etc.).

# Things you can't book directly - use record_followup
You don't actually have the power to do everything a guest might ask. When the caller wants any of these, call record_followup with the right kind so a human can follow up. NEVER say "someone will follow up" without making this tool call - that's how requests get lost.
- Group bookings, events, weddings, corporate rates -> kind="sales_lead". Take their name and number and a one-sentence summary.
- Changes to identity fields on an existing booking (email, phone, name, the card on file) -> kind="identity_change". Verify the booking first if not already verified.
- "Call me back later" / "I'll think about it" -> kind="callback". Note when they want the callback and what about.
- Verification failed three times -> kind="verification_help". A manager calls back.
- In-house guest wants to check out early / shorten a stay they've already started -> kind="early_checkout". Front desk handles in person.
- Anything else outside what your tools cover -> kind="other" with a clear summary.

# Multiple needs in one call
Callers commonly bring more than one thing - "I want to book a room AND a table" or "cancel my room and my dinner reservation." Hold every named need; complete one flow, then surface the next without prompting "anything else?" until they're all done. Don't drop a need just because you finished an unrelated one. If two flows conflict (e.g. caller wants to modify and cancel the same booking), confirm which one they actually want before acting.

# Multiple rooms in one call
Caller wants two (or more) rooms in one transaction - common for families. Call start_room_booking once per room. The booking sub-task auto-fills the guest's name, email, and phone from earlier in the conversation, so you don't re-collect identity between rooms. The card sub-task DOES re-ask the card for each booking (we don't carry the full number across bookings) - mention this once, then let the caller give it again. Confirm whether the rooms share dates or differ; ask just once and pass the right values into set_stay each time.

# Before you confirm a new booking
Before calling confirm() inside start_room_booking, read the whole booking back to the caller in one short sentence - dates, room type and extras, total, and the card last four - and let them say "go ahead" (or correct something). Same idea for restaurant: confirm date, time, party size, and the name on the reservation before calling confirm(). The tool call only fires once they've agreed to what you read back.

# Never invent a confirmation
A booking, reservation, cancellation, refund, modification, or invoice lookup is only real if a tool just returned it. Never tell the caller "you're booked", "you're confirmed", "your changes are saved", or read back a confirmation code, total, or refund amount unless the corresponding tool actually ran in this turn and returned it. If you catch yourself about to confirm something without a tool result in hand, you're hallucinating - stop and call the right tool first.
"""


class HotelReceptionistAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=_instructions())

    async def on_enter(self) -> None:
        consent = await GetRecordingConsentTask(chat_ctx=self.chat_ctx)
        self.session.userdata.consent = consent.consent
        # Don't greet with a canned line - the caller may have already said what
        # they want during the consent step. Let the model pick up from there so
        # we don't re-ask "how can I help?" right after they told us.
        ack = (
            "Briefly confirm the call will be recorded."
            if consent.consent
            else "Briefly confirm the call won't be recorded."
        )
        await self.session.generate_reply(
            instructions=(
                f"{ack} Then continue from whatever the caller has already said: if they've named "
                "a need (a room, a table, a cancellation...), move straight into helping; otherwise "
                "welcome them and ask how you can help."
            )
        )

    @function_tool
    async def get_hotel_info(self, ctx: RunContext[Userdata]) -> str:
        """Return hotel details the receptionist doesn't keep top-of-mind: address, airport and transport, room amenities, accessibility, cribs and rollaways, laundry, lost-and-found, business center. Call this when the caller asks about anything beyond the operational basics already in your instructions (check-in, pets, smoking, parking, Wi-Fi, pool, cancellation, tax)."""
        return _HOTEL_INFO

    @function_tool
    async def get_restaurant_info(self, ctx: RunContext[Userdata]) -> str:
        """Return restaurant details: menu shape (no per-item prices), dietary handling, dress code, private dining, room service, takeout/delivery policy, special-occasion handling. Call this whenever the caller asks about the restaurant beyond hours or wanting to book a table."""
        return _RESTAURANT_INFO

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
    async def record_followup(
        self,
        ctx: RunContext[Userdata],
        kind: FollowupKind,
        caller_name: str,
        caller_phone: str,
        summary: str,
    ) -> str:
        """Capture something for a human to follow up on - sales/group leads, identity-field change requests (email/phone/name/card), callback requests, verification-failed callers, in-house early-checkout requests, and any other request you can't handle on this line. ALWAYS use this instead of saying "someone will follow up" with no record; otherwise the request vanishes.

        Args:
            kind: One of sales_lead, identity_change, callback, verification_help, early_checkout, abandoned_booking, other.
            caller_name: Caller's name (ask if you don't already have it).
            caller_phone: Caller's callback number (ask if you don't already have it).
            summary: One sentence describing what they want, with enough detail for a human to act on it.
        """
        code = await ctx.userdata.db.record_followup(
            kind=kind, caller_name=caller_name, caller_phone=caller_phone, summary=summary
        )
        return f"recorded; reference {_speak_code(code)}. The right team will follow up."

    @function_tool
    async def check_room_availability(
        self,
        ctx: RunContext[Userdata],
        check_in: date,
        check_out: date,
        guests: Annotated[int, Field(ge=1, le=MAX_PARTY_SIZE)],
        smoking: bool | None = None,
        room_type: str | None = None,
    ) -> str:
        """Check what's available for a date range, with prices and views. One tool for every "what do you have?" / "how much?" / "any king available?" / "any smoking rooms?" question.

        Args:
            check_in: Check-in date in ISO YYYY-MM-DD format (e.g. "2026-01-20").
            check_out: Check-out date in ISO YYYY-MM-DD format.
            guests: Number of guests in the room (must be >= 1; ask the caller if not specified).
            smoking: Pass true for smoking-permitted rooms only, false to exclude smoking, or omit if the caller hasn't said.
            room_type: Narrow to a single type ("king", "queen_2beds", "double_queen", "suite", "penthouse") when the caller has already picked one; omit to list everything.
        """
        if check_out <= check_in:
            raise ToolError("check-out must be after check-in")
        avail = await ctx.userdata.db.list_room_types_available(
            check_in=check_in, check_out=check_out, guests=guests, smoking=smoking
        )
        if room_type is not None:
            avail = [a for a in avail if a.type == room_type]
        if not avail:
            kind = "smoking " if smoking else "non-smoking " if smoking is False else ""
            what = f"{kind}{room_type.replace('_', ' ')}" if room_type else f"{kind}rooms"
            return f"no {what} available for those dates"
        return " | ".join(
            f"{a.type.replace('_', ' ')}: {speak_usd(a.nightly_rate)} per night, {a.sample_view} view"
            for a in avail
        )

    @function_tool
    async def start_room_booking(self, ctx: RunContext[Userdata]) -> str | None:
        """Start the room-booking flow. This hands off to a focused booking task that collects stay, room choice, name, email, phone, and card, then finalizes the reservation."""
        booking = await BookRoomTask(db=ctx.userdata.db, chat_ctx=self.chat_ctx)
        ctx.userdata.booked_room_codes.append(booking.code)
        logger.info("[stub] would email confirmation to %s for %s", booking.email, booking.code)
        return (
            f"You're booked. Your confirmation code is {_speak_code(booking.code)}. "
            f"Total is {speak_usd(booking.total)}, charged to the card ending in {booking.card_last4}. "
            f"A confirmation email is on its way to {booking.email}."
        )

    @function_tool
    async def check_restaurant_availability(
        self,
        ctx: RunContext[Userdata],
        on_date: date,
        party_size: Annotated[int, Field(ge=1, le=MAX_PARTY_SIZE)],
    ) -> str:
        """Check restaurant time slots for a date.

        Args:
            on_date: The date to check, in ISO YYYY-MM-DD format (e.g. "2026-01-20").
            party_size: Number of guests (must be >= 1; ask the caller if not specified).
        """
        slots = await ctx.userdata.db.list_restaurant_availability(
            on_date=on_date, party_size=party_size
        )
        open_slots = [s for s in slots if s.available_table_ids]
        if not open_slots:
            return f"fully booked on {on_date.strftime('%A, %B %-d')}"
        return ", ".join(speak_time(s.time) for s in open_slots)

    @function_tool
    async def start_restaurant_booking(self, ctx: RunContext[Userdata]) -> str | None:
        """Start the restaurant-reservation flow. This hands off to a focused task that collects date, party size, time, name, and phone, then finalizes the reservation."""
        reservation = await BookRestaurantTask(db=ctx.userdata.db, chat_ctx=self.chat_ctx)
        ctx.userdata.booked_restaurant_codes.append(reservation.code)
        return (
            f"You're set for {speak_time(reservation.time)} on "
            f"{reservation.date.strftime('%A, %B %-d')} for "
            f"{reservation.party_size} guest{'s' if reservation.party_size != 1 else ''}. "
            f"Confirmation code: {_speak_code(reservation.code)}."
        )

    @function_tool
    async def lookup_restaurant_reservation(
        self,
        ctx: RunContext[Userdata],
        last_name: str,
        confirmation_code: str,
    ) -> str:
        """Read-only lookup of a confirmed restaurant reservation. Use this when the caller wants
        to check or recall their reservation details (date, time, party size, notes) without
        changing or cancelling it.

        Args:
            last_name: caller's last name.
            confirmation_code: confirmation code like 'RES-X9Y2'.
        """
        code = confirmation_code.replace(" ", "").upper()
        reservation = await ctx.userdata.db.find_restaurant_reservation(
            last_name=last_name, confirmation_code=code
        )
        if not reservation or reservation.status != "confirmed":
            raise ToolError("Couldn't find a matching confirmed reservation.")
        notes_part = f", note: {reservation.notes}" if reservation.notes else ""
        return (
            f"Reservation for {reservation.first_name} {reservation.last_name}, "
            f"{speak_time(reservation.time)} on {reservation.date.strftime('%A, %B %-d')}, "
            f"party of {reservation.party_size}{notes_part}."
        )

    async def _verified_booking(self, ctx: RunContext[Userdata]) -> RoomBooking:
        """Verify the caller once per call. Tools that mutate the booking
        (modify, cancel) update or clear the cache themselves."""
        if ctx.userdata.verified_booking is None:
            verify = await VerifyBookingTask(db=ctx.userdata.db, chat_ctx=self.chat_ctx)
            ctx.userdata.verified_booking = verify.booking
        return ctx.userdata.verified_booking

    @function_tool
    async def start_booking_modification(self, ctx: RunContext[Userdata]) -> str:
        """Start the booking-modification flow for an existing reservation. Verifies the caller, then hands off to a focused task that lets them change the stay dates, room type, extras, and party size on the booking. Identity fields (name, email, phone, card) are NOT modifiable through this flow."""
        booking = await self._verified_booking(ctx)
        if booking.status != "confirmed":
            raise ToolError("that booking was cancelled - nothing to modify")
        if booking.check_out < TODAY:
            raise ToolError("that stay already ended - can't modify a past booking")

        updated = await ModifyBookingTask(
            db=ctx.userdata.db, existing=booking, chat_ctx=self.chat_ctx
        )
        # Cache the post-modify booking so subsequent tools (lookup, cancel)
        # don't re-verify and don't see the pre-modify state.
        ctx.userdata.verified_booking = updated
        # ModifyBookingTask returns the *same* booking object when the caller
        # made no changes (it completes with `self._existing`); identity check
        # is the cleanest signal that the modify flow was a no-op.
        if updated is booking:
            return "Booking left unchanged."
        delta = updated.total - booking.total
        if delta == 0:
            money = f"total stays at {speak_usd(updated.total)}"
        else:
            direction = "added to" if delta > 0 else "refunded to"
            money = f"new total is {speak_usd(updated.total)}; {speak_usd(abs(delta))} {direction} the card ending in {updated.card_last4}"
        return f"Your booking is updated; {money}."

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
    async def cancel_room_booking(self, ctx: RunContext[Userdata]) -> str:
        """Cancel a room booking after verifying the caller."""
        booking = await self._verified_booking(ctx)
        if booking.check_in < TODAY:
            raise ToolError("this booking's check-in has already passed; can't cancel a past stay")
        within = (booking.check_in - TODAY).days * 24 < PRICING.cancellation_window_hours
        forfeit = booking.nightly_rate if within else 0
        await ctx.userdata.db.cancel_room_booking(booking.code)
        ctx.userdata.cancelled_codes.append(booking.code)
        # Booking is no longer confirmed; the next tool needing a verified
        # booking should re-prompt the caller (a different reservation, or
        # they're done).
        ctx.userdata.verified_booking = None
        if within:
            return (
                f"Cancelled. Because the booking's inside the {PRICING.cancellation_window_hours}-hour "
                f"window, one room-night ({speak_usd(forfeit)}) is forfeited; "
                f"I'll refund {speak_usd(booking.total - forfeit)} to the card on file."
            )
        return (
            f"Cancelled. I'll refund the full {speak_usd(booking.total)} "
            f"to the card on file - usually two to five business days."
        )

    @function_tool
    async def cancel_restaurant_reservation(
        self,
        ctx: RunContext[Userdata],
        last_name: str,
        confirmation_code: str,
    ) -> str:
        """Cancel a restaurant reservation. Restaurants verify with last name + confirmation code (no card, no email — the code is what we print when the table is booked).

        Args:
            last_name: caller's last name.
            confirmation_code: confirmation code like 'RES-X9Y2'.
        """
        code = confirmation_code.replace(" ", "").upper()
        reservation = await ctx.userdata.db.find_restaurant_reservation(
            last_name=last_name, confirmation_code=code
        )
        if not reservation or reservation.status != "confirmed":
            raise ToolError("Couldn't find a matching confirmed reservation.")
        await ctx.userdata.db.cancel_restaurant_reservation(reservation.code)
        ctx.userdata.cancelled_codes.append(reservation.code)
        return (
            f"Reservation for {speak_time(reservation.time)} on "
            f"{reservation.date.strftime('%A, %B %-d')} cancelled."
        )

    @function_tool
    async def lookup_invoice(self, ctx: RunContext[Userdata]) -> str:
        """Verify the caller, fetch their invoice, and read it back."""
        booking = await self._verified_booking(ctx)
        invoice = await ctx.userdata.db.get_invoice(booking.code)
        logger.info("[stub] would email invoice to %s", booking.email)
        items = ", ".join(f"{li.label} {speak_usd(li.amount_cents)}" for li in invoice.line_items)
        return (
            f"That booking's total is {speak_usd(invoice.total)}, with line items: "
            f"{items}. I've emailed a copy to {booking.email}."
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


def _speak_code(code: str) -> str:
    return ", ".join(code.replace("-", " dash ").upper())


server = AgentServer()

_SEED_DB_BYTES = build_seed_bytes(TODAY)


async def on_simulation_end(ctx: SimulationContext) -> None:
    # Grade the run on final DB state: build the scenario's `expected_state` on a
    # fresh seed, then diff it against the agent's DB. The diff compares
    # agent-decided facts only (room type, dates, extras, status), so minted
    # codes / order / which-king don't matter and the agent need not reproduce the
    # statements — while collateral damage still surfaces.
    expected_state = ctx.userdata().get("expected_state") or []
    if not expected_state:
        return

    session = ctx.job_context.primary_session
    expected = await build_expected(_SEED_DB_BYTES, expected_state)
    try:
        diffs = diff_databases(expected.connection, session.userdata.db.connection)
    finally:
        await expected.aclose()

    # Veto the run if the final DB state diverged. The effective result is the AND of
    # this check and the simulator's conversation judgment, so a mismatch fails a run
    # the simulator passed; a match simply leaves the simulator's verdict to stand.
    if diffs:
        ctx.fail(reason="final DB diverges from expected: " + " | ".join(diffs[:8]))


async def on_session_end(ctx: JobContext) -> None:
    try:
        report = ctx.make_session_report()
    except RuntimeError:
        return

    chat = report.chat_history.copy(exclude_function_call=True, exclude_instructions=True)
    if len(chat.items) < 3:
        return

    judges = JudgeGroup(
        llm="openai/gpt-4o-mini",
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
    if userdata.booked_room_codes or userdata.booked_restaurant_codes or userdata.cancelled_codes:
        ctx.tagger.success()
    else:
        ctx.tagger.fail(reason="No successful booking, cancellation, or invoice action.")

    logger.info("session tags: %s", ctx.tagger.tags)

    try:
        await userdata.db.aclose()
    except Exception:
        logger.exception("error closing hotel DB")


@server.rtc_session(on_session_end=on_session_end, on_simulation_end=on_simulation_end)
async def hotel_receptionist_agent(ctx: JobContext) -> None:
    await ctx.connect()

    db = HotelDB.from_bytes(_SEED_DB_BYTES)

    ui = UiView(ctx.room, db.connection)
    db.on_change = ui.on_change
    await ui.start()

    userdata = Userdata(db=db)
    session = AgentSession[Userdata](
        userdata=userdata,
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("google/gemma-4-31b-it"),
        tts=inference.TTS(
            "inworld/inworld-tts-2",
            voice="Ashley",
            extra_kwargs={"delivery_mode": "CREATIVE", "speaking_rate": 1.1},
        ),
        expressive=presets.CUSTOMER_SERVICE,
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
        max_tool_steps=5,
        # Flip user_state to "away" after 10s of mutual silence so we can
        # check whether they're still there (default is 15s).
        user_away_timeout=10.0,
    )

    idle_task: asyncio.Task[None] | None = None

    async def _nudge_while_idle() -> None:
        # Nudge every 10s until the user speaks again — speaking flips
        # user_state out of "away", which cancels this task below.
        while True:
            logger.info("user idle — checking if they're still there")
            await session.generate_reply(
                instructions="The user has been idle, see if they're still there"
            )
            await asyncio.sleep(10)

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: UserStateChangedEvent) -> None:
        nonlocal idle_task
        if ev.new_state == "away":
            if idle_task is None or idle_task.done():
                idle_task = asyncio.create_task(_nudge_while_idle())
        elif idle_task is not None:
            idle_task.cancel()
            idle_task = None

    await session.start(agent=HotelReceptionistAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
