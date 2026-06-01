from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from book_restaurant import BookRestaurantTask
from book_room import BookRoomTask
from dotenv import load_dotenv
from hotel_db import (
    DISPUTE_POLICIES,
    MAX_PARTY_SIZE,
    PRICING,
    TODAY,
    DisputeCategory,
    DisputePolicy,
    HotelDB,
    NotFound,
    format_usd,
    speak_time,
    speak_usd,
)
from pydantic import Field
from ui_view import UiView
from workflows import COMMON_INSTRUCTIONS, GetRecordingConsentTask, VerifyBookingTask

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
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
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

logger = logging.getLogger("hotel-receptionist")


@dataclass
class Userdata:
    db: HotelDB
    consent: bool | None = None
    booked_room_codes: list[str] = field(default_factory=list)
    booked_restaurant_codes: list[str] = field(default_factory=list)
    cancelled_codes: list[str] = field(default_factory=list)


def _instructions() -> str:
    return f"""\
{COMMON_INSTRUCTIONS}

You're the lead receptionist, holding the whole call and routing each request to the right tool.

# What you know (answer directly - no tool call needed)
- Check-in 3 PM, check-out 11 AM.
- Breakfast {format_usd(PRICING.breakfast_per_night)} per night, served 6:30 to 10:30 AM.
- Late checkout until 2 PM: {format_usd(PRICING.late_checkout)} flat, subject to availability.
- Pets: pet-friendly rooms only, {format_usd(PRICING.pet_fee)} per stay.
- Smoking: smoking-permitted rooms on request; smoking in a non-smoking room is a {format_usd(PRICING.smoking_cleaning_fee)} cleaning fee.
- Valet {format_usd(PRICING.valet_per_night)} per night, self-parking free.
- Wi-Fi free. Pool, gym, sauna 6 AM to 10 PM, free for guests.
- Cancellation: free up to {PRICING.cancellation_window_hours} hours before check-in; inside that window, the caller forfeits one night.
- Tax: {PRICING.tax_rate_pct}% on room and extras.

# Sold out
Offer adjacent dates or another room type. One tool call per turn; finish each tool's flow before starting another.

# Browse vs. book
check_room_availability, room_details, and check_restaurant_availability look things up but never create a booking. The moment the caller signals they want to proceed (says "can I book a room", "book me a table", picks a room, confirms a date), call start_room_booking or start_restaurant_booking - the call IS your response, not something you do after an acknowledgment. Don't say "absolutely, I can help" and stop; that leaves the caller in silence. If a brief acknowledgment feels natural, fold it into the same turn alongside the call. If you find yourself asking the caller for personal details (name, email, phone, card) without start_room_booking or start_restaurant_booking running, you've skipped the only path that actually creates a booking - call the right one instead.

# Never invent a confirmation
A booking, reservation, cancellation, refund, or invoice lookup is only real if it came from a tool call. Never tell the caller "you're booked", "you're confirmed", or read back a confirmation code unless the corresponding tool actually ran and returned one. If you catch yourself about to confirm something without a tool result in hand, you're hallucinating - stop and call the right tool first.
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
    async def check_room_availability(
        self,
        ctx: RunContext[Userdata],
        check_in: date,
        check_out: date,
        guests: Annotated[int, Field(ge=1, le=MAX_PARTY_SIZE)],
    ) -> str:
        """Check which room types are available for a date range. Returns only the room type names - use room_details for prices and views.

        Args:
            check_in: Check-in date.
            check_out: Check-out date.
            guests: Number of guests in the room.
        """
        if check_out <= check_in:
            raise ToolError("check-out must be after check-in")

        avail = await ctx.userdata.db.list_room_types_available(
            check_in=check_in, check_out=check_out, guests=guests
        )
        if not avail:
            return "sold out"
        return ", ".join(a.type.replace("_", " ") for a in avail)

    @function_tool
    async def room_details(
        self,
        ctx: RunContext[Userdata],
        check_in: date,
        check_out: date,
        guests: Annotated[int, Field(ge=1, le=MAX_PARTY_SIZE)],
        room_type: str,
    ) -> str:
        """Fetch the nightly rate and view for a specific room type. Call this after the caller picks a type from check_room_availability.

        Args:
            check_in: Check-in date.
            check_out: Check-out date.
            guests: Number of guests in the room.
            room_type: One of the types from check_room_availability.
        """
        avail = await ctx.userdata.db.list_room_types_available(
            check_in=check_in, check_out=check_out, guests=guests
        )
        match = next((a for a in avail if a.type == room_type), None)
        if match is None:
            raise ToolError(f"no {room_type} available for those dates")
        return f"{room_type.replace('_', ' ')}: {speak_usd(match.nightly_rate)} per night, {match.sample_view} view"  # fmt: skip

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
            on_date: The date to check.
            party_size: number of guests.
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
    async def cancel_room_booking(self, ctx: RunContext[Userdata]) -> str:
        """Cancel a room booking after verifying the caller."""
        verify = await VerifyBookingTask(db=ctx.userdata.db, chat_ctx=self.chat_ctx)
        booking = verify.booking
        if booking.check_in < TODAY:
            raise ToolError(
                "this booking's check-in has already passed; can't cancel a past stay"
            )
        within = (booking.check_in - TODAY).days * 24 < PRICING.cancellation_window_hours
        forfeit = booking.total // max(booking.nights, 1) if within else 0
        try:
            await ctx.userdata.db.cancel_room_booking(booking.code)
        except NotFound:
            raise ToolError("Couldn't find that booking when I tried to cancel.") from None
        ctx.userdata.cancelled_codes.append(booking.code)
        if within:
            return (
                f"Cancelled. Because the booking's inside the {PRICING.cancellation_window_hours}-hour "
                f"window, one night ({speak_usd(forfeit)}) is forfeited; "
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
        """Cancel a restaurant reservation, verified by last name + confirmation code.

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
        verify = await VerifyBookingTask(db=ctx.userdata.db, chat_ctx=self.chat_ctx)
        invoice = await ctx.userdata.db.get_invoice(verify.booking.code)
        logger.info("[stub] would email invoice to %s", verify.booking.email)
        items = ", ".join(f"{li.label} {speak_usd(li.amount_cents)}" for li in invoice.line_items)
        return (
            f"That booking's total is {speak_usd(invoice.total)}, with line items: "
            f"{items}. I've emailed a copy to {verify.booking.email}."
        )

    @function_tool
    async def dispute_charge(
        self,
        ctx: RunContext[Userdata],
        category: DisputeCategory,
        line_item_label: str,
        caller_note: str,
        accepts_offered_resolution: bool = True,
    ) -> str:
        """Handle a guest dispute on a line item.

        Args:
            category: Pick the category that best matches what the caller is disputing.
            line_item_label: The label of the line item on the invoice, as it appears.
            caller_note: A short summary of what the caller said about the charge.
            accepts_offered_resolution: True if the caller accepts the policy outcome offered
                (a goodwill waiver, a credit, etc.); False if they push back and the case
                should escalate.
        """
        if category not in DISPUTE_POLICIES:
            raise ToolError(f"unknown dispute category: {category}")
        policy = DISPUTE_POLICIES[category]

        verify = await VerifyBookingTask(db=ctx.userdata.db, chat_ctx=self.chat_ctx)
        invoice = await ctx.userdata.db.get_invoice(verify.booking.code)

        item = next((li for li in invoice.line_items if li.label == line_item_label), None)
        if item is None:
            raise ToolError(
                f"No line item labelled {line_item_label!r} on that invoice. "
                "Read the line items back and ask the caller to pick one."
            )

        amount = item.amount_cents
        outcome, refund = _resolve_dispute_outcome(
            policy=policy,
            amount_cents=amount,
            line_item_label=line_item_label,
            invoice_line_items=[(li.label, li.amount_cents) for li in invoice.line_items],
            accepts=accepts_offered_resolution,
        )

        case_number = await ctx.userdata.db.file_dispute(
            booking_code=verify.booking.code,
            line_item=line_item_label,
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
            line_item=line_item_label,
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


@server.rtc_session(on_session_end=on_session_end)
async def hotel_receptionist_agent(ctx: JobContext) -> None:
    await ctx.connect()

    db_path = os.getenv(
        "HOTEL_DB_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_data", "hotel.db"),
    )

    db = HotelDB(db_path)
    await db.initialize()

    ui = UiView(ctx.room, db.connection)
    db.on_change = ui.on_change
    await ui.start()

    userdata = Userdata(db=db)
    session = AgentSession[Userdata](
        userdata=userdata,
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("google/gemini-2.5-flash"),
        tts=inference.TTS("cartesia/sonic-3", voice="39b376fc-488e-4d0c-8b37-e00b72059fdd"),
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
        # Allow the model to chain a few tool calls in one turn - e.g. fill the
        # booking draft (set_stay -> choose_room -> set_contact) when the caller
        # volunteers several details at once, instead of one tool per turn.
        max_tool_steps=5,
    )

    await session.start(agent=HotelReceptionistAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
