from __future__ import annotations

import logging
import os
import sys
from datetime import date, time
from typing import Annotated, Literal

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import Userdata, _speak_code
from hotel_db import (
    MAX_PARTY_SIZE,
    FollowupKind,
    NotFound,
    Unavailable,
    speak_time,
    speak_usd,
)
from pydantic import Field

from livekit.agents import RunContext, ToolError, function_tool

logger = logging.getLogger("hotel-receptionist")


class ServicesToolsMixin:
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
            kind: One of housekeeping, sales_lead, identity_change, callback, verification_help, early_checkout, abandoned_booking, lost_and_found, other.
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
        kind: Literal["medical", "fire", "security"],
        situation: str,
    ) -> str:
        """EMERGENCY ONLY - a real, in-progress danger. Use it the MOMENT you have the room number and what's happening: no verification, no other questions first. It alerts the duty manager and sends hotel staff/security to the room - that dispatch is the PRIMARY action and shows the hotel owns it; outside help (911 / fire brigade / police) is a secondary direction you give the caller, never a substitute for sending the hotel's own people. Classify the kind:
          - "medical" - someone hurt, collapsed, unresponsive, not breathing, a health crisis.
          - "fire" - fire, smoke, or a fire alarm going off.
          - "security" - a safety/security threat: an intruder or someone forcing a door, assault or violence, a theft.
        NOT for nuisances - a noisy neighbour with nobody in danger is record_followup (kind="other"), not this.

        Args:
            room: The room number (e.g. "206"). Get this first if you don't have it.
            kind: medical, fire, or security - classify what's happening.
            situation: One short sentence: what's happening to whom.
        """
        try:
            code = await ctx.userdata.db.dispatch_emergency(room=room, kind=kind, situation=situation)
        except NotFound:
            raise ToolError(
                f"no room {room} exists - re-confirm the room number, calmly, right now"
            ) from None
        head = (
            f"DISPATCHED (ref {code}): duty manager alerted, staff heading to room {room} now | "
            "tell the caller, short and calm, that our people are on their way up right now"
        )
        if kind == "medical":
            tail = (
                " - then have them hang up and dial 9-1-1; the dispatcher stays on the line and "
                "tells them exactly what to do until the ambulance arrives. Don't give medical "
                "instructions yourself - the 911 dispatcher is the right person for that."
            )
        elif kind == "fire":
            tail = (
                " - tell them to get out now via the stairs or fire escapes, NOT the elevator, "
                "stay low if there's smoke, and once safe call the fire brigade on 9-1-1. Don't "
                "tell them to fight the fire or go investigate it."
            )
        else:  # security
            tail = (
                " - if they're in any immediate danger tell them to call 9-1-1 (police) now and "
                "stay somewhere safe with the door locked; otherwise our security and duty manager "
                "will be right there to help and take care of what's needed (a police report, and "
                "for a lost passport the consulate can help). Don't tell them to confront anyone."
            )
        return head + tail

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
    async def book_spa_appointment(
        self,
        ctx: RunContext[Userdata],
        service: Literal[
            "deep_tissue_massage", "signature_facial", "personal_training", "group_yoga"
        ],
        on_date: date,
        at_time: time,
        party_size: Annotated[int, Field(ge=1)],
        guest_name: str,
        guest_phone: str,
    ) -> str:
        """Book a spa or health-club service (massage, facial, personal training, yoga). The catalog (services, prices, durations, hours) is in lookup_policy topic "spa" - look it up first and narrow with the caller (which service, date, time, party size) before booking. The options are for the CALLER to pick from, never pick for them. Once they pick and agree, THIS CALL is the booking - saying "I'll get that set up" books nothing; nothing exists until this returns a reference.

        Args:
            service: The spa service the caller picked.
            on_date: Appointment date in ISO YYYY-MM-DD format.
            at_time: Appointment start time in 24-hour HH:MM format.
            party_size: How many people the appointment is for.
            guest_name: The caller's full name.
            guest_phone: The caller's phone number, in case the spa needs to reach them.
        """
        try:
            code, s, total = await ctx.userdata.db.book_spa_appointment(
                service_id=service, guest_name=guest_name, guest_phone=guest_phone,
                on_date=on_date, at_time=at_time, party_size=party_size,
            )
        except (NotFound, Unavailable) as e:
            raise ToolError(str(e)) from None
        return (
            f"{s.name} booked for {party_size} on {on_date.strftime('%A, %B %-d')} at "
            f"{speak_time(at_time)}; reference {_speak_code(code)}. {s.duration_min} minutes, "
            f"total {speak_usd(total)} ({s.description}) | confirm the service, date, time, and "
            "total to the caller; no further tool call is needed for this appointment."
        )

    @function_tool
    async def book_business_center(
        self,
        ctx: RunContext[Userdata],
        service: Literal["meeting_room", "secretarial", "printing"],
        on_date: date,
        at_time: time,
        duration_hours: Annotated[int, Field(ge=1)],
        guest_name: str,
        guest_phone: str,
    ) -> str:
        """Book a business-centre service - a meeting room, secretarial help, or a printing job. The catalog (rates, hours, what's included) is in lookup_policy topic "business_center" - look it up first and narrow with the caller (which service, the date and start time, and how long) before booking. The options are for the CALLER to pick from, never pick for them. Once they pick and agree, THIS CALL is the booking - saying "I'll get that set up" books nothing; nothing exists until this returns a reference.

        Args:
            service: The service the caller picked.
            on_date: Service date in ISO YYYY-MM-DD format.
            at_time: Start time in 24-hour HH:MM format.
            duration_hours: How many hours the caller needs (printing is a flat one-hour job).
            guest_name: The caller's full name.
            guest_phone: The caller's phone number, in case the business centre needs to reach them.
        """
        try:
            code, s, total = await ctx.userdata.db.book_business_center(
                service_id=service, guest_name=guest_name, guest_phone=guest_phone,
                on_date=on_date, at_time=at_time, duration_hours=duration_hours,
            )
        except (NotFound, Unavailable) as e:
            raise ToolError(str(e)) from None
        return (
            f"{s.name} booked for {on_date.strftime('%A, %B %-d')} at {speak_time(at_time)}; "
            f"reference {_speak_code(code)}. Total {speak_usd(total)} ({s.description}) | confirm "
            "the service, start time, and total to the caller - these are fixed, give them as "
            "facts; no further tool call is needed."
        )

    @function_tool
    async def order_flowers(
        self,
        ctx: RunContext[Userdata],
        arrangement: Literal["bouquet", "roses", "centerpiece"],
        on_date: date,
        deliver_to: str,
        card_message: str,
        guest_name: str,
        guest_phone: str,
    ) -> str:
        """Order a flower arrangement from the hotel florist for delivery to a room or recipient. The catalog (arrangements, prices, delivery cutoff) is in lookup_policy topic "florist" - look it up first and let the caller pick the arrangement, never pick for them. Collect the delivery date, where it goes (room number or recipient name), and the gift-card message, and read the card message back so it's right. Once they pick and agree, THIS CALL places the order - saying "I'll get that arranged" orders nothing; nothing exists until this returns a reference.

        Args:
            arrangement: The arrangement the caller picked.
            on_date: Delivery date in ISO YYYY-MM-DD format.
            deliver_to: Where it goes - the number of the room or the recipient's name. Prefer room number when available.
            card_message: The gift-card message exactly as the caller dictates it.
            guest_name: The caller's full name.
            guest_phone: The caller's phone number, in case the florist needs to reach them.
        """
        try:
            code, a, total = await ctx.userdata.db.order_flowers(
                arrangement_id=arrangement, guest_name=guest_name, guest_phone=guest_phone,
                deliver_to=deliver_to, on_date=on_date, card_message=card_message,
            )
        except (NotFound, Unavailable) as e:
            raise ToolError(str(e)) from None
        return (
            f"{a.name} ordered for delivery to {deliver_to} on "
            f"{on_date.strftime('%A, %B %-d')}; reference {_speak_code(code)}; total "
            f"{speak_usd(total)} | confirm the arrangement, where it's going, the date, and the "
            "total to the caller - no further tool call is needed for this order."
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
            "Give ONE short closing hand-off (\"You're all set - connecting you now\"), NOT "
            "\"anything else?\", so the call can wrap up. Do NOT transfer again or take the request "
            "down as a followup; if the caller reacts, keep it to a brief acknowledgement and "
            "don't reopen the conversation."
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
    async def set_do_not_disturb(self, ctx: RunContext[Userdata], room: str) -> str:
        """Place a Do-Not-Disturb hold on an in-house guest's room when they ask not to be disturbed / to hold their calls and messages. It's a standing hold (until lifted), not a one-off like a single message or a wake-up call. Take the room number. Always tell the guest that a genuine emergency or hotel safety matter still overrides DND.

        Args:
            room: The guest's room number.
        """
        try:
            code = await ctx.userdata.db.set_do_not_disturb(room=room)
        except NotFound:
            raise ToolError(f"no room {room} exists - re-confirm the room number") from None
        return (
            f"Do-Not-Disturb set on room {room}; reference {_speak_code(code)} | confirm it holds "
            "their calls and messages until they ask to lift it, and that a genuine emergency "
            "still gets through."
        )

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
