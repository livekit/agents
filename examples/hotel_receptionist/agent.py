from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, time
from typing import Annotated, Literal

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark import build_expected, diff_databases
from book_restaurant import BookRestaurantTask
from book_room import BookRoomTask
from context import speech_only
from dotenv import load_dotenv
from fake_data.seed import build_seed_bytes
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
    format_usd,
    speak_time,
    speak_usd,
)
from modify_booking import ModifyBookingTask
from persona import COMMON_INSTRUCTIONS
from policies import build_lookup_policy_tool
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
    handoff_judge,
    safety_judge,
    task_completion_judge,
    tool_use_judge,
)

load_dotenv(".env.local")

logger = logging.getLogger("hotel-receptionist")


@dataclass
class Userdata:
    db: HotelDB
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
- EMERGENCY FIRST, above everything on this list: someone hurt, unresponsive, or in danger -> get the room number and call dispatch_emergency immediately (it alerts the desk and sends the manager and staff up). No verification, no other flow, no policy lookup. Then direct the caller to hang up and dial 911 themselves - the dispatcher needs them on the line and will coach them until help arrives. The hotel does not call 911 for them, and you never give medical instructions yourself.
- Browse without booking: check_room_availability (rate + view + optional smoking/room_type filters), check_restaurant_availability, lookup_booking, lookup_restaurant_reservation. None of these change anything.
- Caller wants to book: start_room_booking or start_restaurant_booking - the call IS your response, not something after an acknowledgment. Don't ask the caller for name, email, phone, or card without one of these running - that's the only path that creates a booking.
- Existing booking changes: start_booking_modification (dates, room, extras, party size). Cancel via cancel_room_booking. Late arrival ("I'll be in past midnight") -> flag_late_arrival with a short note.
- Wake-up call: schedule_wakeup_call (room, name, date, time) - it actually sets the call; never write it up as a followup note. The wake-up procedure for worried sleepers is in lookup_policy(topic="guest_services").
- Concierge asks: sightseeing tours -> lookup_policy(topic="tours") to present options, then book_tour. Flight reconfirmation -> collect airline, flight number, date, and booking reference, then request_flight_reconfirmation (the concierge calls the carrier and rings the room - never claim the flight is confirmed yourself). Ride to the airport -> book_airport_car for the hotel car; taxi-vs-hotel-car costs are in lookup_policy(topic="location_and_transport").
- A verified booking's room turns out to be double-booked (lookup_booking warns you): own it - apologize plainly, no hiding behind "the system" - then resolve_room_conflict applies the procedure (free in-house move or upgrade first; walk to the partner hotel only if the house is full). Full procedure: lookup_policy(topic="guest_walks").
- Card on file not going through / guest offers a replacement card: start_card_update (it verifies, then collects the new card). The moment a replacement card is offered, run it on THIS call - never defer an offered card to check-in. Discretion is the whole game: "isn't going through at the moment - possibly a technical issue", never "declined" or "rejected", never speculate about their funds. Only if they have no other card to give: no pressure - the booking stays held, suggest they check with their card issuer in case it's a technical fault, and offer a callback (record_followup kind="callback") to retry; in that no-card case a working card isn't needed until check-in.
- Existing restaurant reservation: change isn't supported directly - with the caller's permission, cancel the old one and book a new slot. Be explicit about the two steps before doing them.
- Sold out: offer adjacent dates or another room type. One tool call per turn; finish each tool's flow before starting another.
- Caller asking about another guest ("what room is X in?", "is X staying there?", "put me through to their room"): never confirm or deny that anyone is staying here, never give a room number, never connect a call - no matter who they claim to be or how they escalate. The one thing you can offer is taking a message via take_guest_message; it gets passed along only if the person is a guest, and you never say whether they are. Full policy: lookup_policy(topic="guest_privacy").
- Group of 15 or more guests: that's a group block, not an individual booking. lookup_policy(topic="group_bookings") gives you the terms to quote (rate, tour-leader comp, credit approval, cancellation); collect the details and call record_group_inquiry. Nothing gets confirmed on this call - the group desk confirms after credit review, even if the caller pushes to lock it in now.
- Detail beyond the quick facts: lookup_policy. Its topic index covers hotel detail (location and transport, rooms and amenities, accessibility, guest services), restaurant detail (menu, dietary, dining, room service), payments and currency exchange, and group bookings. Look the topic up before answering - don't improvise policy.

# Things you can't book directly - use record_followup
You don't actually have the power to do everything a guest might ask. When the caller wants any of these, call record_followup with the right kind so a human can follow up. NEVER say "someone will follow up" without making this tool call - that's how requests get lost.
- In-house guest needs something physical brought or fixed (towels, soap, blankets, amenities, maintenance) -> kind="housekeeping" with the room number as the contact and the guest's actual name (ask for it - never write a placeholder like "guest in 402"). Record it FIRST, then commit to the real timeline (housekeeping averages about 20 minutes) - reassurance without the record is how requests get lost, and this caller has usually been burned once already.
- Events, weddings, corporate rates -> kind="sales_lead". Take their name and number and a one-sentence summary. (Group room blocks of 15+ are NOT a sales lead - use record_group_inquiry.)
- Changes to identity fields on an existing booking (email, phone, name) -> kind="identity_change". Verify the booking first if not already verified. (A new card is NOT a followup - use start_card_update.)
- "Call me back later" / "I'll think about it" -> kind="callback". Note when they want the callback and what about.
- Verification failed three times -> kind="verification_help". A manager calls back.
- In-house guest wants to check out early / shorten a stay they've already started -> kind="early_checkout". Front desk handles in person.
- Anything else outside what your tools cover -> kind="other" with a clear summary.
If the caller adds details after a followup is recorded (a refund request, urgency, anything they want passed along), call record_followup again with the fuller summary - never claim the notes were updated without making the call.
A followup is a recorded request, not a dispatch: never promise that someone is physically on their way, will respond "immediately" or "right away", or will arrive by a specific time off the back of one. Say what's actually true - "I've logged this for the duty manager as urgent; they'll get to you as soon as they can."

# Multiple needs in one call
Callers commonly bring more than one thing - "I want to book a room AND a table" or "cancel my room and my dinner reservation." Hold every named need; complete one flow, then surface the next without prompting "anything else?" until they're all done. Don't drop a need just because you finished an unrelated one. If two flows conflict (e.g. caller wants to modify and cancel the same booking), confirm which one they actually want before acting.

# Multiple rooms in one call
Caller wants two (or more) rooms in one transaction - common for families. Call start_room_booking once per room. The booking sub-task auto-fills the guest's name, email, and phone from earlier in the conversation, so you don't re-collect identity between rooms. The card sub-task DOES re-ask the card for each booking (we don't carry the full number across bookings) - mention this once, then let the caller give it again. Confirm whether the rooms share dates or differ; ask just once and pass the right values into set_stay each time.

# When a booking flow returns
start_room_booking, start_restaurant_booking, and start_booking_modification return the FINAL result - "You're booked", "You're set", "Your booking is updated". That returned result IS the confirmation: relay the code and total to the caller and move on to their next need. The flow is closed at that point - the read-back already happened inside it, there is no card to take, and there is no tool to call to "re-confirm" anything. Never re-run the confirmation conversation after the flow has returned its result.

# Never invent a confirmation
A booking, reservation, cancellation, refund, modification, invoice lookup, logged message, or recorded followup is only real if a tool just returned it. "I've logged that for you" with no tool call is a lie the caller will act on - if you owe the caller a tool call (a message to log, an inquiry to record), make it before or while answering whatever they asked next; an interleaved question doesn't cancel the debt. Never tell the caller "you're booked", "you're confirmed", "your changes are saved", or read back a confirmation code, total, or refund amount unless the corresponding tool actually ran in this turn and returned it. A tool ERROR - including "Unknown function" - means nothing happened this turn: never announce success, a code, or a total off the back of an error; fix the call (the error names the available tools) or tell the caller you need a moment. If you catch yourself about to confirm something without a tool result in hand, you're hallucinating - stop and call the right tool first.
"""


class HotelReceptionistAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=_instructions(), tools=[build_lookup_policy_tool()])

    async def on_enter(self) -> None:
        # The caller may have already said what they want before we speak -
        # pick up from there instead of re-asking "how can I help?".
        await self.session.generate_reply(
            instructions=(
                "Greet the caller in one short sentence. If they've already named a need "
                "(a room, a table, a cancellation...), move straight into helping; "
                "otherwise ask how you can help."
            )
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
    async def record_followup(
        self,
        ctx: RunContext[Userdata],
        kind: FollowupKind,
        caller_name: str,
        caller_phone: str,
        summary: str,
    ) -> str:
        """Capture something for a human to follow up on - sales/group leads, identity-field change requests (email/phone/name), callback requests, verification-failed callers, in-house early-checkout requests, and any other request you can't handle on this line. ALWAYS use this instead of saying "someone will follow up" with no record; otherwise the request vanishes.

        Args:
            kind: One of housekeeping, sales_lead, identity_change, callback, verification_help, early_checkout, abandoned_booking, other.
            caller_name: Caller's name (ask if you don't already have it).
            caller_phone: Caller's callback number - for an in-house guest, the room number works.
            summary: One sentence describing what they want, with enough detail for a human to act on it.
        """
        code = await ctx.userdata.db.record_followup(
            kind=kind, caller_name=caller_name, caller_phone=caller_phone, summary=summary
        )
        return f"recorded; reference {_speak_code(code)}. The right team will follow up."

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
    async def schedule_wakeup_call(
        self,
        ctx: RunContext[Userdata],
        room: str,
        guest_name: str,
        call_date: date,
        call_time: time,
    ) -> str:
        """Schedule a wake-up call to a guest's room. This actually sets the call - never log a wake-up request as a followup note instead. Collect the room, the name, and the exact date and time from the caller, read them back, and call this once they've agreed. No booking verification needed.

        Args:
            room: The room number as the caller gave it (e.g. "304").
            guest_name: The guest's name.
            call_date: The date of the wake-up call in ISO YYYY-MM-DD format ("tomorrow morning" = tomorrow's date).
            call_time: The wake-up time in 24-hour HH:MM format (4:45 a.m. = "04:45").
        """
        try:
            code = await ctx.userdata.db.schedule_wakeup_call(
                room=room, guest_name=guest_name, call_date=call_date, call_time=call_time
            )
        except NotFound:
            raise ToolError(
                f"no room {room} exists - re-confirm the room number with the caller"
            ) from None
        except Unavailable as e:
            raise ToolError(f"can't schedule that: {e} - re-confirm the date") from None
        return (
            f"wake-up call set for room {room}, {call_date.strftime('%A, %B %-d')} at "
            f"{speak_time(call_time)}; reference {_speak_code(code)} | confirm it's set. If the "
            "caller worries about sleeping through: a second call comes about five minutes later "
            "if there's no answer, and no response to that sends staff up for an in-person room "
            "check - they will be woken."
        )

    @function_tool
    async def dispatch_emergency(
        self,
        ctx: RunContext[Userdata],
        room: str,
        situation: str,
    ) -> str:
        """EMERGENCY ONLY - someone hurt, unresponsive, in danger (collapse, not breathing, fire, violence). Alerts the front desk and sends the duty manager and staff to the room. Call this the MOMENT you have the room number and what's happening - no verification, no other questions first. The hotel does NOT call 911 - after this runs, the caller must be told to hang up and dial 911 themselves: the dispatcher needs to hear them directly and will coach them (CPR, what to check) until the ambulance arrives.

        Args:
            room: The room number (e.g. "206"). Get this first if you don't have it.
            situation: One short sentence: what's happening to whom (e.g. "guest's husband collapsed, not breathing").
        """
        try:
            code = await ctx.userdata.db.dispatch_emergency(room=room, situation=situation)
        except NotFound:
            raise ToolError(
                f"no room {room} exists - re-confirm the room number, calmly, right now"
            ) from None
        return (
            f"DISPATCHED (ref {code}): duty manager alerted, staff heading to room {room} now | "
            "tell the caller, short and calm, in this order: our manager and staff are on their "
            "way up right now - now hang up and dial 9-1-1; the dispatcher will stay on the line "
            "and tell you exactly what to do until the ambulance arrives. Then let her go - the "
            "911 dispatcher is the right person from here, not you. Don't give medical "
            "instructions yourself."
        )

    @function_tool
    async def book_tour(
        self,
        ctx: RunContext[Userdata],
        tour: Literal["half_day_city", "full_day_city", "private_city"],
        on_date: date,
        party_size: Annotated[int, Field(ge=1)],
        guest_name: str,
        guest_phone: str,
    ) -> str:
        """Book a sightseeing tour through the desk. The catalog (times, prices, what's included) is in lookup_policy topic "tours" - look it up first and narrow with the caller (group or private, half or full day, date, party size) before booking. The options are for the CALLER to pick from, never pick for them. Once they pick and agree, THIS CALL is the booking - saying "I'll get that set up" books nothing; nothing exists until this returns a reference.

        Args:
            tour: The tour the caller picked.
            on_date: Tour date in ISO YYYY-MM-DD format.
            party_size: How many people are going.
            guest_name: The caller's full name.
            guest_phone: The caller's phone number, in case the operator needs to reach them.
        """
        try:
            code, t, total = await ctx.userdata.db.book_tour(
                tour_id=tour,
                guest_name=guest_name,
                guest_phone=guest_phone,
                on_date=on_date,
                party_size=party_size,
            )
        except (NotFound, Unavailable) as e:
            raise ToolError(str(e)) from None
        return (
            f"{t.name} booked for {party_size} on {on_date.strftime('%A, %B %-d')}; reference "
            f"{_speak_code(code)}. Pickup {speak_time(t.pickup_time)} at the {t.pickup_location}; "
            f"total {speak_usd(total)} ({t.description}) | confirm the pickup time, spot, and "
            "total to the caller - these are fixed, give them as facts; no further tool call "
            "is needed for this tour."
        )

    @function_tool
    async def request_flight_reconfirmation(
        self,
        ctx: RunContext[Userdata],
        room: str,
        airline: str,
        flight_number: str,
        flight_date: date,
        booking_reference: str,
        seat_check: bool,
    ) -> str:
        """Log a flight-reconfirmation request for an in-house guest: the concierge calls the carrier and rings the guest's room with the result. Collect ALL the flight details first and read the booking reference back before calling - a wrong reference makes the whole request useless.

        Args:
            room: The guest's room number.
            airline: The carrier name (e.g. "Iberia").
            flight_number: Airline code and number as given (e.g. "IB 6174").
            flight_date: Flight date in ISO YYYY-MM-DD format. When the caller says a weekday ("Thursday"), resolve it against today and say the concrete date back ("Thursday - that's June eleventh?") BEFORE calling; a one-day slip sends the whole request to the wrong flight.
            booking_reference: The airline booking reference, letters and digits only.
            seat_check: True if the guest also wants their seat assignment checked - it's handled in the same carrier call.
        """
        try:
            code = await ctx.userdata.db.request_flight_reconfirmation(
                room=room,
                airline=airline,
                flight_number=flight_number,
                flight_date=flight_date,
                booking_reference=booking_reference,
                seat_check=seat_check,
            )
        except NotFound:
            raise ToolError(f"no room {room} exists - re-confirm the room number") from None
        return (
            f"reconfirmation request logged; reference {_speak_code(code)} | tell the caller the "
            "concierge will call the carrier and ring their room with the result within the hour"
            + (", including the seat check" if seat_check else "")
            + ". The flight is NOT confirmed yet - never say it is; promise the callback instead."
        )

    @function_tool
    async def book_airport_car(
        self,
        ctx: RunContext[Userdata],
        room: str,
        pickup_date: date,
        pickup_time: time,
        passengers: Annotated[int, Field(ge=1, le=4)],
    ) -> str:
        """Book the hotel car to the airport for an in-house guest: flat eighty-five dollars to SFO, seats up to four with luggage, charged to the room. (Taxis are hailed at the door, metered roughly fifty-five to seventy dollars, and can't be reserved ahead - cost comparison in lookup_policy topic "location_and_transport".) Sanity-check the pickup time against the flight when you know it - about three hours before departure is right for international.

        Args:
            room: The guest's room number.
            pickup_date: Pickup date in ISO YYYY-MM-DD format. Resolve a weekday against today and confirm the concrete date with the caller before booking.
            pickup_time: Pickup time in 24-hour HH:MM format (2:30 p.m. = "14:30").
            passengers: How many people are riding - ASK the caller; never assume one.
        """
        try:
            code = await ctx.userdata.db.book_airport_car(
                room=room,
                pickup_date=pickup_date,
                pickup_time=pickup_time,
                passengers=passengers,
            )
        except NotFound:
            raise ToolError(f"no room {room} exists - re-confirm the room number") from None
        except Unavailable as e:
            raise ToolError(f"can't book that: {e} - re-confirm the date") from None
        return (
            f"hotel car booked; reference {_speak_code(code)}. Pickup "
            f"{pickup_date.strftime('%A, %B %-d')} at {speak_time(pickup_time)}, front entrance, "
            f"{passengers} passenger{'s' if passengers != 1 else ''}, flat eighty-five dollars "
            "charged to the room | confirm the time, the front-entrance pickup, the cost, and "
            "the reference to the caller; no further tool call is needed for the car."
        )

    @function_tool
    async def resolve_room_conflict(self, ctx: RunContext[Userdata]) -> str:
        """Fix a double-booked / no-room situation on the caller's verified booking - run this when lookup_booking warned the room is double-booked. It applies the house procedure in fixed order: move the guest to a free room of the same or better category (an upgrade is free), and only if nothing in the house fits, arrange the walk (partner hotel tonight on us, covered taxi, their room back from the return date). Returns the concrete plan - relay it with ownership and an apology, and say "at no extra cost to you" explicitly. Procedure detail: lookup_policy topic "guest_walks"."""
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
                "confirm the WHOLE new arrangement out loud: they absolutely still have a place "
                "to stay - the new room and what it is, for the same dates and the whole stay, "
                "at the same total - it costs nothing extra. Apologize for the mix-up. No "
                "further tool call is needed."
            )
        assert r.walk_return_date is not None
        return_day = r.walk_return_date.strftime("%A")
        return (
            f"no room in the house fits (every room was checked) - walk arranged at "
            f"{r.walk_partner}, room back here {r.walk_return_date.strftime('%A, %B %-d')} | "
            "deliver the plan in SHORT pieces - the guest is angry and WILL interrupt; that's "
            "fine, just make sure every one of these gets said before the call ends, resuming "
            "after each interruption: (1) \"I've checked every room in the house - nothing is "
            f"free tonight, and I'm so sorry.\" (2) \"We've arranged a comparable room for you "
            f"tonight at {r.walk_partner}, two blocks away - the room and the taxi over are "
            f'both on us." (3) "Your room here is guaranteed from {return_day} - all at no '
            "extra cost to you.\" Track which pieces you've said; repeat any that got talked "
            "over. If the guest is still upset after hearing all three, don't argue - record a "
            'manager callback (record_followup, kind="callback") and tell them the manager '
            "will call; do this BEFORE wrapping up the call."
        )

    @function_tool
    async def take_guest_message(
        self,
        ctx: RunContext[Userdata],
        recipient: str,
        caller_name: str,
        caller_phone: str,
        message: str,
    ) -> str:
        """Take a message for someone the caller says is staying at the hotel. It gets delivered only if that person is in fact a guest - the result never tells you whether they are, and you must never tell the caller either: no confirming or denying anyone's presence, no room numbers, no connecting calls (see lookup_policy topic "guest_privacy"). Read the caller's name, number, and message back before calling this.

        Args:
            recipient: Full name of the person the message is for - first AND last. If the caller only gave a first name, ask for the last name before calling.
            caller_name: The caller's own name.
            caller_phone: The caller's callback number.
            message: The message, in the caller's words.
        """
        if len(recipient.split()) < 2:
            raise ToolError(
                f"'{recipient}' is only one name - a message needs the recipient's full name "
                "to reach the right person. Ask the caller for the last name, then call again."
            )
        code = await ctx.userdata.db.take_guest_message(
            recipient=recipient,
            caller_name=caller_name,
            caller_phone=caller_phone,
            message=message,
        )
        return (
            f"message recorded; reference {_speak_code(code)} | tell the caller it's logged and "
            "give the reference. You don't know whether the recipient is staying here and never "
            "say either way - but the general policy IS shareable: messages for in-house guests "
            "reach the room within about thirty minutes (message light, slip under the door). "
            "Promise delivery timing only, never that the person will read or act on it."
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
        booking = await BookRoomTask(db=ctx.userdata.db, chat_ctx=speech_only(self.chat_ctx))
        ctx.userdata.booked_room_codes.append(booking.code)
        logger.info("[stub] would email confirmation to %s for %s", booking.email, booking.code)
        return (
            f"You're booked. Your confirmation code is {_speak_code(booking.code)}. "
            f"Total is {speak_usd(booking.total)}, charged to the card ending in {booking.card_last4}. "
            f"A confirmation email is on its way to {booking.email}. "
            "| booking complete - relay the code and total to the caller; "
            "no further tool call is needed for this booking."
        )

    @function_tool
    async def check_restaurant_availability(
        self,
        ctx: RunContext[Userdata],
        on_date: date,
        party_size: Annotated[int, Field(ge=1, le=MAX_PARTY_SIZE)],
    ) -> str:
        """Check restaurant time slots for a date. Read-only browsing - to actually book a table, call start_restaurant_booking.

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
        """Start the restaurant-reservation flow. Call it the moment the caller wants a table - the flow collects date, party size, time, name, and phone itself. Its return is the FINAL result of the reservation: relay it and move on - nothing further to confirm or call afterwards."""
        reservation = await BookRestaurantTask(
            db=ctx.userdata.db, chat_ctx=speech_only(self.chat_ctx)
        )
        ctx.userdata.booked_restaurant_codes.append(reservation.code)
        return (
            f"You're set for {speak_time(reservation.time)} on "
            f"{reservation.date.strftime('%A, %B %-d')} for "
            f"{reservation.party_size} guest{'s' if reservation.party_size != 1 else ''}. "
            f"Confirmation code: {_speak_code(reservation.code)}. "
            "| reservation complete - relay this to the caller; no further tool call is needed."
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
        """Start the booking-modification flow for an existing reservation. Verifies the caller, then hands off to a focused task that lets them change the stay dates, room type, extras, and party size on the booking. Identity fields (name, email, phone) are NOT modifiable through this flow - record_followup with kind="identity_change" covers those, and a new card goes through start_card_update. NOT for cancellations: if the caller wants to cancel - even after asking for a change first - call cancel_room_booking directly instead."""
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
            "| modification complete - relay this to the caller; no further tool call is needed."
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
            f"Cancelled - well outside the {PRICING.cancellation_window_hours}-hour window, so "
            f"there's no penalty and no deposit is lost. I'll refund the full "
            f"{speak_usd(booking.total)} to the card on file - usually two to five business days."
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
    # Spell character by character, with "-" spoken as the single word "dash" -
    # NOT spelled D, A, S, H (that reads as four more code characters).
    return ", ".join("dash" if c == "-" else c for c in code.upper())


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
        llm="openai/gpt-4.1-mini",
        judges=[
            task_completion_judge(),
            accuracy_judge(),
            tool_use_judge(),
            handoff_judge(),
            safety_judge(),
            # relevancy_judge(),
            # coherence_judge(),
            # conciseness_judge(),
        ],
    )
    await judges.evaluate(report.chat_history)

    userdata = ctx.primary_session.userdata

    # The server-side finalize (-> on_simulation_end) is not always delivered:
    # some sims tear down via participant disconnect and the DB-diff veto
    # silently never runs. Run the same diff here too, so the verdict always
    # lands in the LOCAL session report even when the server never asks.
    db_diffs: list[str] = []
    try:
        sim_ctx = ctx.simulation_context()
        if sim_ctx is None:
            logger.info(
                "local expected-state diff skipped: no simulation context "
                "(job/room metadata carried no SimulationDispatch)"
            )
        expected_state = (sim_ctx.userdata().get("expected_state") if sim_ctx else None) or []
        if sim_ctx is not None and not expected_state:
            logger.info("local expected-state diff skipped: scenario has no expected_state")
        if expected_state:
            logger.info("running local expected-state diff (%d statement(s))", len(expected_state))
            expected = await build_expected(_SEED_DB_BYTES, expected_state)
            try:
                db_diffs = diff_databases(expected.connection, userdata.db.connection)
            finally:
                await expected.aclose()
    except Exception:
        logger.exception("error running local expected-state diff")

    # "Did the call do real work?" is a DB question, not per-tool bookkeeping:
    # compare the final DB against the untouched seed. Any change in the
    # transactional tables (booking, cancellation, modification, dispute,
    # followup, late-arrival note...) counts.
    try:
        seed_db = HotelDB.from_bytes(_SEED_DB_BYTES)
        try:
            state_changes = diff_databases(seed_db.connection, userdata.db.connection)
        finally:
            await seed_db.aclose()
    except Exception:
        logger.exception("error diffing final DB against seed")
        state_changes = []

    # Read-only calls (policy questions, availability checks, booking lookups)
    # are real work too - a Q&A call that answered from a successful read tool
    # shouldn't be tagged as having accomplished nothing.
    read_tools = {
        "lookup_policy",
        "lookup_booking",
        "lookup_invoice",
        "lookup_restaurant_reservation",
        "check_room_availability",
        "check_restaurant_availability",
    }
    call_names = {
        item.call_id: item.name
        for item in report.chat_history.items
        if item.type == "function_call"
    }
    served_reads = any(
        item.type == "function_call_output"
        and not item.is_error
        and call_names.get(item.call_id) in read_tools
        for item in report.chat_history.items
    )

    if db_diffs:
        ctx.tagger.fail(reason="final DB diverges from expected: " + " | ".join(db_diffs[:8]))
    elif state_changes or served_reads:
        ctx.tagger.success()
    else:
        ctx.tagger.fail(
            reason="The call accomplished nothing: no state was changed (booking, "
            "cancellation, modification, dispute, followup, message, wake-up call...) "
            "and no information was looked up for the caller."
        )

    logger.info("session tags: %s", ctx.tagger.tags)

    # Dump the final DB next to the session report so expected-state diffs can
    # run post-hoc (diff-run.py at the repo root). The in-room grading path is
    # unreliable in the current beta: the SimulationDispatch doesn't reach room
    # metadata and finalize_simulation is skipped on disconnect teardowns.
    if report_dir := os.environ.get("LIVEKIT_SESSION_REPORT_DIR"):
        try:
            path = os.path.join(report_dir, f"final_db-{ctx.room.name}.sqlite")
            with open(path, "wb") as f:
                f.write(userdata.db.serialize())
        except Exception:
            logger.exception("error dumping final DB state")

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
        tts=inference.TTS("cartesia/sonic-3", voice="39b376fc-488e-4d0c-8b37-e00b72059fdd"),
        max_tool_steps=5,
    )

    await session.start(agent=HotelReceptionistAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
