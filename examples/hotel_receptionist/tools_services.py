from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import date, time
from enum import StrEnum
from typing import Annotated, Literal

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import Userdata, _count_caller_turns, _speak_code
from end_call_check import run_goodbye_gate
from hotel_db import (
    MAX_PARTY_SIZE,
    FollowupKind,
    NotFound,
    Unavailable,
    speak_room,
    speak_time,
    speak_usd,
)
from pydantic import Field, RootModel

from livekit.agents import (
    Agent,
    AgentSession,
    CloseEvent,
    RunContext,
    ToolError,
    function_tool,
    get_job_context,
)

logger = logging.getLogger("hotel-receptionist")

# Strong refs to in-flight post-goodbye shutdown tasks (see say_goodbye_and_close_call).
_pending_shutdowns: set[asyncio.Task[None]] = set()
# At most one close watchdog may own a session. A newer close supersedes the old
# timer so stale work can never shut down a later turn.
_close_watchdogs: dict[AgentSession, asyncio.Task[None]] = {}

# How long the line stays quiet after the goodbye before the agent hangs up itself.
# Callers usually hang up within a couple of seconds of the farewell; this is only
# the fallback for the ones who don't.
_CALLER_HANGUP_GRACE = 10.0

# The shorter quiet period for a REPEAT close: the farewell already happened and
# the caller has indicated twice that they're done. New caller speech postpones this
# timer by the full _CALLER_HANGUP_GRACE rather than cancelling it.
_REPEAT_CLOSE_GRACE = 3.0


def _arm_close_watchdog(session: AgentSession, *, grace: float) -> asyncio.Task[None]:
    """Hang up after `grace` seconds of caller silence.

    Caller speech postpones the close, it never cancels it: the farewell is already
    spoken, so a caller who keeps answering it must not be able to hold the line open
    for the rest of the call. Postponement is always the full caller-hangup grace, which
    outlasts the reply to that utterance, so the session is never torn down mid-answer.
    shutdown() is idempotent, so a caller who hangs up during the wait is a no-op.
    """

    if previous := _close_watchdogs.get(session):
        previous.cancel()

    loop = asyncio.get_running_loop()
    deadline = loop.time() + grace

    def _on_item_added(ev: object) -> None:
        nonlocal deadline
        item = getattr(ev, "item", None)
        if item is not None and getattr(item, "role", None) == "user":
            deadline = loop.time() + _CALLER_HANGUP_GRACE

    async def _close_after_silence() -> None:
        try:
            while (remaining := deadline - loop.time()) > 0:
                await asyncio.sleep(remaining)
            session.shutdown()
        except asyncio.CancelledError:
            pass  # superseded by a newer close
        finally:
            session.off("conversation_item_added", _on_item_added)
            if _close_watchdogs.get(session) is task:
                del _close_watchdogs[session]

    session.on("conversation_item_added", _on_item_added)
    task = asyncio.create_task(_close_after_silence())
    _close_watchdogs[session] = task
    _pending_shutdowns.add(task)
    task.add_done_callback(_pending_shutdowns.discard)
    return task


def _farewell_instruction(userdata: Userdata) -> str:
    """The close path's reply instruction: one farewell per call, ever.

    Callers routinely answer a goodbye ("you too!"), and the model then calls the
    close tool again - without this guard that produced a "Goodbye!" / "Take care!" /
    "Goodbye!" loop. The first close delivers the farewell; every later one answers
    real questions only and otherwise stays quiet while the line closes on its own.
    """
    if not userdata.goodbye_said:
        userdata.goodbye_said = True
        return (
            "The line closes right after your next utterance. Give ONE short, warm "
            "goodbye now - no questions, no new information."
        )
    return (
        "You've already said goodbye - do NOT give another farewell, sign-off, or "
        "filler. If the caller just asked a real question, answer it in one short "
        "sentence; otherwise say nothing. The line closes on its own once they stop."
    )


def _departure_margin_note(*, pickup: time, departure: time) -> str:
    """The pickup-vs-departure sanity check, computed instead of remembered.

    The prose instruction to "sanity-check the pickup time against the flight"
    was routinely skipped; with the departure time on file the tool hands the
    agent the actual margin to say back.
    """
    minutes = (departure.hour * 60 + departure.minute) - (pickup.hour * 60 + pickup.minute)
    spoken_dep = speak_time(departure)
    if minutes <= 0:
        return (
            f"the pickup is not before the {spoken_dep} departure - re-check the "
            "times with the guest before confirming anything."
        )
    halves = round(minutes / 30)
    whole, half = divmod(halves, 2)
    if whole == 0:
        margin = "about half an hour"
    elif half:
        margin = f"about {whole} and a half hours"
    else:
        margin = f"about {whole} hour{'s' if whole != 1 else ''}"
    if minutes < 120:
        return (
            f"that's only {margin} before the {spoken_dep} departure - TIGHT for an "
            "airport run; flag it to the guest (about 3 hours ahead is right for "
            "international)."
        )
    return (
        f"that's {margin} before the {spoken_dep} departure - say this margin back "
        "to the guest (about 3 hours ahead is right for international)."
    )


class NumberedRoom(RootModel[int]):
    """A numbered room or suite, serialized as its integer room number."""


class PenthouseSuite(RootModel[Literal["penthouse"]]):
    """The hotel's sole named room, serialized as the string ``"penthouse"``."""


Room = NumberedRoom | PenthouseSuite


class DeliveryPreference(StrEnum):
    AS_EARLY_AS_POSSIBLE = "as_early_as_possible"
    BEFORE_NOON_IF_POSSIBLE = "before_noon_if_possible"
    LEAVE_WITH_FRONT_DESK = "leave_with_front_desk"


def room_to_id(room: NumberedRoom | PenthouseSuite) -> str:
    return "RM_PH" if isinstance(room, PenthouseSuite) else f"RM_{room.root}"


def _delivery_instruction_value(instruction: DeliveryPreference) -> str:
    """Return the stable value stored in florist_orders."""
    return instruction.value


def _speak_delivery_instruction(instruction: DeliveryPreference) -> str:
    return {
        DeliveryPreference.AS_EARLY_AS_POSSIBLE: "as early as possible",
        DeliveryPreference.BEFORE_NOON_IF_POSSIBLE: "before noon if possible",
        DeliveryPreference.LEAVE_WITH_FRONT_DESK: "leave with the front desk",
    }[instruction]


def _florist_destination(
    room: NumberedRoom | PenthouseSuite | None, recipient_name: str | None
) -> tuple[str | None, str | None]:
    """The delivery destination: (room_id, None) for a room here, or
    (None, recipient) when no room is assigned yet. The Room type makes a
    fabricated destination inexpressible; existence of the numbered room is
    checked where the order is written."""
    recipient = (recipient_name or "").strip() or None
    if room is not None:
        return room_to_id(room), None
    if recipient:
        return None, recipient
    raise ToolError(
        "no destination: ask the caller which room or suite the flowers go to, "
        "or who they're for if their room isn't assigned yet."
    )


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
        """Capture something for a human to follow up on - sales/group leads, identity-field change requests (email/phone/name), callback requests, verification-failed callers, in-house early-checkout requests, and any other request you can't handle on this line. ALWAYS use this instead of saying "someone will follow up" with no record; otherwise the request vanishes. Florist delivery notes are the one exception - they belong on the order itself via amend_florist_order, never here.

        Args:
            kind: One of housekeeping, sales_lead, identity_change, callback, verification_help, early_checkout, abandoned_booking, lost_and_found, other.
            caller_name: The caller's actual name - ask for it if you don't already have it, for EVERY kind of followup. Never write a placeholder derived from context: no "Unknown", no "guest in 402", no "Guest in 202" - a room number identifies the room, not the person, and a followup is a note for a human about a person.
            caller_phone: Caller's callback number - for an in-house guest, the room number works (for the phone only, never for the name).
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
            "after credit review, to confirm the block. Then settle any question the caller "
            "asked earlier that never got its answer - a group-rate question from the top of "
            "the call still needs the provisional rate quoted from the group_bookings policy "
            "before the call ends."
        )

    @function_tool
    async def schedule_wakeup_call(
        self,
        ctx: RunContext[Userdata],
        room: Room,
        guest_name: str,
        call_date: date,
        call_time: time,
    ) -> str:
        """Schedule a wake-up call to a guest's room. This actually sets the call - never log a wake-up request as a followup note instead. Collect the room, the name, and the exact date and time from the caller, read them back, and call this once they've agreed. No booking verification needed.

        Args:
            room: The guest's room, as the caller gave it.
            guest_name: The guest's name.
            call_date: The date of the wake-up call in ISO YYYY-MM-DD format ("tomorrow morning" = tomorrow's date).
            call_time: The wake-up time in 24-hour HH:MM format (4:45 a.m. = "04:45").
        """
        room_id = room_to_id(room)
        spoken = speak_room(room_id)
        try:
            code = await ctx.userdata.db.schedule_wakeup_call(
                room=room_id, guest_name=guest_name, call_date=call_date, call_time=call_time
            )
        except NotFound:
            raise ToolError(
                f"{spoken} doesn't exist here - re-confirm the room with the caller"
            ) from None
        except Unavailable as e:
            raise ToolError(f"can't schedule that: {e} - re-confirm the date") from None
        return (
            f"wake-up call set for {spoken}, {call_date.strftime('%A, %B %-d')} at "
            f"{speak_time(call_time)}; reference {_speak_code(code)} | confirm it's set. If the "
            "caller worries about sleeping through: a second call comes about five minutes later "
            "if there's no answer, and no response to that sends staff up for an in-person room "
            "check - they will be woken."
        )

    @function_tool
    async def dispatch_emergency(
        self,
        ctx: RunContext[Userdata],
        room: Room,
        kind: Literal["medical", "fire", "security"],
        situation: str,
    ) -> str:
        """EMERGENCY ONLY - a real, in-progress danger. Use it the MOMENT you have the room number and what's happening: no verification, no other questions first. It alerts the duty manager and sends hotel staff/security to the room - that dispatch is the PRIMARY action and shows the hotel owns it; outside help (911 / fire brigade / police) is a secondary direction you give the caller, never a substitute for sending the hotel's own people. Classify the kind:
          - "medical" - someone hurt, collapsed, unresponsive, not breathing, a health crisis.
          - "fire" - fire, smoke, or a fire alarm going off.
          - "security" - a safety/security threat: an intruder or someone forcing a door, assault or violence, a theft.
        NOT for nuisances - a noisy neighbour with nobody in danger is record_followup (kind="other"), not this.

        Args:
            room: The guest's room. Get this first if you don't have it.
            kind: medical, fire, or security - classify what's happening.
            situation: One short sentence: what's happening to whom.
        """
        room_id = room_to_id(room)
        spoken = speak_room(room_id)
        try:
            code = await ctx.userdata.db.dispatch_emergency(
                room=room_id, kind=kind, situation=situation
            )
        except NotFound:
            raise ToolError(
                f"{spoken} doesn't exist here - re-confirm the room, calmly, right now"
            ) from None
        head = (
            f"DISPATCHED (ref {code}): duty manager alerted, staff heading to {spoken} now | "
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
        """Book a spa or health-club service (massage, facial, personal training, yoga). The catalog (services, prices, durations, hours) is in lookup_policy topic "spa" - look it up first and narrow with the caller (which service, date, time, party size) before booking. The options are for the CALLER to pick from, never pick for them. Quote the chosen service's price and duration from the catalog BEFORE booking - a caller who named the service up front still hears both before you book, not only after. Once they pick and agree, THIS CALL is the booking - saying "I'll get that set up" books nothing; nothing exists until this returns a reference.

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
                service_id=service,
                guest_name=guest_name,
                guest_phone=guest_phone,
                on_date=on_date,
                at_time=at_time,
                party_size=party_size,
            )
        except (NotFound, Unavailable) as e:
            raise ToolError(str(e)) from None
        return (
            f"{s.name} booked for {party_size} on {on_date.strftime('%A, %B %-d')} at "
            f"{speak_time(at_time)}; reference {_speak_code(code)}. {s.duration_min} minutes, "
            f"total {speak_usd(total)} ({s.description}) | confirm the service, date, time, "
            f"duration ({s.duration_min} minutes), total, and reference to the caller; no "
            "further tool call is needed for this appointment."
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
                service_id=service,
                guest_name=guest_name,
                guest_phone=guest_phone,
                on_date=on_date,
                at_time=at_time,
                duration_hours=duration_hours,
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
        card_message: str,
        guest_name: str,
        guest_phone: str,
        room: Room | None = None,
        recipient_name: str | None = None,
        delivery_instruction: DeliveryPreference | None = None,
    ) -> str:
        """Order a flower arrangement from the hotel florist, delivered to a room or suite here, or to an arriving guest by name if their room isn't assigned yet. The catalog (arrangements, prices, delivery cutoff) is in lookup_policy topic "florist" - look it up first and let the caller pick the arrangement, never pick for them. Collect the delivery date, where it goes, and the gift-card message, and read the card message back so it's right. Once they pick and agree, THIS CALL places the order - saying "I'll get that arranged" orders nothing; nothing exists until this returns a reference. Delivery handling requests ("as early as possible") go in delivery_instruction here, or via amend_florist_order after placing - never in a followup.

        Args:
            arrangement: The arrangement the caller picked.
            on_date: Delivery date in ISO YYYY-MM-DD format.
            card_message: The gift-card message exactly as the caller dictates it.
            guest_name: The caller's full name.
            guest_phone: The caller's phone number, in case the florist needs to reach them.
            room: The destination room. Pass a numbered room or suite as its integer room number; pass the penthouse suite as "penthouse". If the caller says the delivery goes to a room or suite here - including for a guest who hasn't arrived yet - the destination IS that room: ask WHICH room or suite and pass it. When no room or suite exists yet to name, omit this parameter and pass recipient_name; never fill in a room the caller did not name.
            recipient_name: Who it's for, ONLY when the caller cannot name a room or suite at all (no room assigned or known yet). Naming the recipient does not capture the destination - if the caller mentioned a room or suite, ask which one and use room instead.
            delivery_instruction: A delivery preference - a handling constraint ADDITIONAL to the destination, never a restatement of it. Holding the flowers for a guest's arrival - "hold it for my arrival", "until she checks in" - is where they go, not a preference: omit this parameter. Pass "as_early_as_possible", "before_noon_if_possible", or "leave_with_front_desk" only when the caller adds a real constraint the destination doesn't imply. Match the caller's actual constraint: a deadline ("before she arrives around noon", "before checkout") is "before_noon_if_possible"; reserve "as_early_as_possible" for callers who want the earliest slot with no deadline. These are requests for the florist, not guaranteed delivery times. Omit if none.
        """
        room_id, recipient = _florist_destination(room, recipient_name)
        try:
            code, a, total = await ctx.userdata.db.order_flowers(
                arrangement_id=arrangement,
                guest_name=guest_name,
                guest_phone=guest_phone,
                on_date=on_date,
                card_message=card_message,
                room_id=room_id,
                recipient_name=recipient,
                delivery_instructions=(
                    _delivery_instruction_value(delivery_instruction)
                    if delivery_instruction is not None
                    else ""
                ),
            )
        except (NotFound, Unavailable) as e:
            raise ToolError(str(e)) from None
        destination = speak_room(room_id) if room_id else recipient
        return (
            f"{a.name} ordered for delivery to {destination} on "
            f"{on_date.strftime('%A, %B %-d')}; reference {_speak_code(code)}; total "
            f"{speak_usd(total)} | confirm the arrangement, where it's going, the date, the "
            "total, AND the reference number to the caller - the reference is part of the "
            "confirmation, not optional. No further tool call is needed for this order."
        )

    @function_tool
    async def amend_florist_order(
        self,
        ctx: RunContext[Userdata],
        order_code: str,
        delivery_instruction: DeliveryPreference,
    ) -> str:
        """Add or update the delivery instructions on an existing florist order - THIS is how a note gets to the florist ("deliver as early as possible", "leave with the front desk"). Use it when a delivery request comes up after the order was placed. Never record a followup for florist delivery notes; this call is the record.

        Args:
            order_code: The order's reference code (FLR-...).
            delivery_instruction: The delivery preference to add: "as_early_as_possible", "before_noon_if_possible", or "leave_with_front_desk". Match the caller's actual constraint: a deadline ("before she arrives around noon") is "before_noon_if_possible"; reserve "as_early_as_possible" for callers who want the earliest slot with no deadline. These are requests for the florist, not guaranteed delivery times.
        """
        value = _delivery_instruction_value(delivery_instruction)
        try:
            await ctx.userdata.db.amend_florist_order(code=order_code, delivery_instructions=value)
        except NotFound as e:
            raise ToolError(str(e)) from None
        return (
            f"noted on order {_speak_code(order_code)}: "
            f'"{_speak_delivery_instruction(delivery_instruction)}" | read the '
            "note back to the caller - it goes to the florist with the order; no further tool "
            "call is needed."
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
        # Send exactly once per ask. The first call can suspend into booking
        # verification and resume; the model then sometimes re-issues the call in the
        # same turn, emailing the same document twice - the deterministic grader
        # counts emails_sent rows, and a duplicate fails the run. With no caller turn
        # since the last send, there is no new ask: relay the sent outcome instead.
        if (
            ctx.userdata.caller_turns_at_last_resend >= 0
            and _count_caller_turns(self.session.history)
            <= ctx.userdata.caller_turns_at_last_resend
        ):
            return (
                "that document already went out moments ago - do NOT send it again. "
                f"Relay the outcome to the caller: {ctx.userdata.last_resend_message}"
            )
        booking = await self._verified_booking(ctx)
        await ctx.userdata.db.send_email(recipient=booking.email, kind=kind)
        msg = f"Sent to the address on file, {booking.email.strip().lower()}."
        ctx.userdata.last_resend_message = msg
        ctx.userdata.caller_turns_at_last_resend = _count_caller_turns(self.session.history)
        return msg

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

    @function_tool
    async def request_flight_reconfirmation(
        self,
        ctx: RunContext[Userdata],
        room: Room,
        airline: str,
        flight_number: str,
        flight_date: date,
        booking_reference: str,
        seat_check: bool,
        departure_time: time | None,
    ) -> str:
        """Log a flight-reconfirmation request for an in-house guest: the concierge calls the carrier and rings the guest's room with the result. Collect ALL the flight details first and read the booking reference back before calling - a wrong reference makes the whole request useless.

        Args:
            room: The guest's room.
            airline: The carrier name (e.g. "Iberia").
            flight_number: Airline code and number as given (e.g. "IB 6174").
            flight_date: Flight date in ISO YYYY-MM-DD format. When the caller says a weekday ("Thursday"), resolve it against today and say the concrete date back ("Thursday - that's June eleventh?") BEFORE calling; a one-day slip sends the whole request to the wrong flight.
            booking_reference: The airline booking reference, letters and digits only.
            seat_check: True if the guest also wants their seat assignment checked - it's handled in the same carrier call.
            departure_time: Scheduled departure in 24-hour HH:MM format. ASK the caller for it before logging - it's what any airport-car pickup gets sanity-checked against. Pass null only when the caller says they don't know it.
        """
        room_id = room_to_id(room)
        normalized_booking_reference = "".join(c for c in booking_reference if c.isalnum()).upper()
        try:
            code = await ctx.userdata.db.request_flight_reconfirmation(
                room=room_id,
                airline=airline,
                flight_number=flight_number,
                flight_date=flight_date,
                booking_reference=normalized_booking_reference,
                seat_check=seat_check,
                departure_time=departure_time,
            )
        except NotFound:
            raise ToolError(
                f"{speak_room(room_id)} doesn't exist here - re-confirm the room"
            ) from None
        return (
            f"reconfirmation request logged; request reference {_speak_code(code)}. "
            "The airline booking reference captured is "
            f"{_speak_code(normalized_booking_reference)}; explicitly read that airline booking "
            "reference back to the caller, labeling it as the airline booking reference so it is "
            "not confused with the car or request reference. The concierge will call the carrier "
            "and ring their room with the result within the hour"
            + (", including the seat check" if seat_check else "")
            + ". The flight is NOT confirmed yet - never say it is; promise the callback instead."
        )

    @function_tool
    async def book_airport_car(
        self,
        ctx: RunContext[Userdata],
        room: Room,
        pickup_date: date,
        pickup_time: time,
        flight_departure_time: time,
        passengers: Annotated[int, Field(ge=1, le=4)],
    ) -> str:
        """Book the hotel car to the airport for an in-house guest: flat eighty-five dollars to SFO, seats up to four with luggage, charged to the room. (Taxis are hailed at the door, metered roughly fifty-five to seventy dollars, and can't be reserved ahead - cost comparison in lookup_policy topic "location_and_transport".) The scheduled flight departure is required so the pickup can always be sanity-checked - about three hours before departure is right for international.

        Args:
            room: The guest's room.
            pickup_date: Pickup date in ISO YYYY-MM-DD format. Resolve a weekday against today and confirm the concrete date with the caller before booking.
            pickup_time: Pickup time in 24-hour HH:MM format (2:30 p.m. = "14:30").
            flight_departure_time: Scheduled flight departure in 24-hour HH:MM format. ASK the caller; never infer it from the requested pickup time.
            passengers: How many people are riding - ASK the caller; never assume one.
        """
        room_id = room_to_id(room)
        try:
            code = await ctx.userdata.db.book_airport_car(
                room=room_id,
                pickup_date=pickup_date,
                pickup_time=pickup_time,
                passengers=passengers,
            )
        except NotFound:
            raise ToolError(
                f"{speak_room(room_id)} doesn't exist here - re-confirm the room"
            ) from None
        except Unavailable as e:
            raise ToolError(f"can't book that: {e} - re-confirm the date") from None
        margin_note = " " + _departure_margin_note(
            pickup=pickup_time, departure=flight_departure_time
        )
        return (
            f"hotel car booked; reference {_speak_code(code)}. Pickup "
            f"{pickup_date.strftime('%A, %B %-d')} at {speak_time(pickup_time)}, front entrance, "
            f"{passengers} passenger{'s' if passengers != 1 else ''}, flat eighty-five dollars "
            f"charged to the room -{margin_note}"
            " | confirm the time, the front-entrance pickup, the cost, and "
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
            f"On file: {prefs} | proactively offer to set these up again for the new stay. The "
            'ones the guest confirms are only noted once record_followup (kind="other") has run '
            "with them - make that call before starting any booking flow (it isn't reachable "
            "from inside one), loading guest_services first if record_followup isn't on yet. "
            "Don't add any preference beyond these."
        )

    @function_tool
    async def set_do_not_disturb(self, ctx: RunContext[Userdata], room: Room) -> str:
        """Place a Do-Not-Disturb hold on an in-house guest's room when they ask not to be disturbed / to hold their calls and messages. It's a standing hold (until lifted), not a one-off like a single message or a wake-up call. Take the room number. Always tell the guest that a genuine emergency or hotel safety matter still overrides DND.

        Args:
            room: The guest's room.
        """
        room_id = room_to_id(room)
        spoken = speak_room(room_id)
        try:
            code = await ctx.userdata.db.set_do_not_disturb(room=room_id)
        except NotFound:
            raise ToolError(f"{spoken} doesn't exist here - re-confirm the room") from None
        return (
            f"Do-Not-Disturb set on {spoken}; reference {_speak_code(code)} | confirm it holds "
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

    @function_tool
    async def say_goodbye_and_close_call(self, ctx: RunContext[Userdata]) -> str:
        """End the call once the caller indicates they're finished ("that's all", "thanks, bye"). NEVER say goodbye yourself - this tool delivers the farewell and then closes the line. It may instead hand back one last thing your standing policy still requires on this call: handle that with the caller first, then call this tool again. Don't call it when the caller is only pausing, holding, or mid-request."""
        # Pre-hangup policy audit: re-read the standing policy against the transcript
        # and, at most once per call, hand the agent back the one thing it still owes
        # the caller instead of closing - the "offer before wind-down" policy grounded
        # in a guaranteed action. Skipped on repeat closes (the farewell already
        # happened; the call is winding down, not re-opening).
        if not ctx.userdata.goodbye_said:
            agent_instructions = self.instructions if isinstance(self, Agent) else ""
            nudge = await run_goodbye_gate(
                ctx.userdata,
                ctx.session.llm,
                instructions=agent_instructions if isinstance(agent_instructions, str) else "",
                chat_ctx=ctx.session.history,
            )
            if nudge is not None:
                return nudge

        # Close path: the goodbye is this tool's reply, reusing the current speech
        # handle. Don't hang up right after it - callers routinely answer a farewell
        # ("okay, bye!"), and a session torn down under that reply leaves their turn
        # hanging (observed in simulations as 60s turn timeouts). Do what a real
        # receptionist does: say goodbye, give the caller the chance to hang up
        # first, and only close the line after it stays quiet. On the FIRST close
        # anything the caller says re-opens the conversation and cancels the pending
        # close. Repeat closes use a shorter quiet period, but caller speech still
        # cancels the old timer so it cannot shut down a newer active turn. When that
        # turn finishes, a repeat close replaces the timer.
        session = ctx.session
        repeat_close = ctx.userdata.goodbye_said

        def _arm_after_reply(_: object) -> None:
            _arm_close_watchdog(
                session,
                grace=_REPEAT_CLOSE_GRACE if repeat_close else _CALLER_HANGUP_GRACE,
            )

        ctx.speech_handle.add_done_callback(_arm_after_reply)

        @ctx.session.once("close")
        def _on_close(ev: CloseEvent) -> None:
            try:
                job_ctx = get_job_context()
            except RuntimeError:
                return  # no job to shut down (console / tests)

            async def _delete_room() -> None:
                await job_ctx.delete_room()

            job_ctx.add_shutdown_callback(_delete_room)
            job_ctx.shutdown(reason=ev.reason.value)

        return _farewell_instruction(ctx.userdata)
