from __future__ import annotations

import logging
import os
import sys
from datetime import date, time
from typing import Annotated, Literal

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# scenarios.yaml dates are literals against this date. hotel_db.TODAY freezes at
# import time, so the pin has to precede every import that reaches hotel_db.
if "--simulation" in sys.argv:
    os.environ.setdefault("HOTEL_TODAY", "2026-06-08")

from benchmark import build_expected, diff_databases
from book_restaurant import BookRestaurantTask
from common import Userdata, _speak_code, speech_only
from dotenv import load_dotenv
from hotel_db import (
    MAX_PARTY_SIZE,
    TODAY,
    FollowupKind,
    HotelDB,
    NotFound,
    Unavailable,
    speak_time,
    speak_usd,
)
from instructions import INSTRUCTIONS
from policies import build_lookup_policy_tool
from pydantic import Field
from seed import build_seed_bytes

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

logger = logging.getLogger("hotel-amenities")


@function_tool
async def record_followup(
    ctx: RunContext[Userdata],
    kind: FollowupKind,
    caller_name: str,
    caller_phone: str,
    summary: str,
) -> str:
    """Capture something for a human to follow up on - sales leads (events, weddings), callback requests, pre-arrival preferences, and any other request you can't handle on this line. ALWAYS use this instead of saying "someone will follow up" with no record; otherwise the request vanishes.

    Args:
        kind: One of sales_lead, callback, other.
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


class AmenitiesAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS, tools=[build_lookup_policy_tool()])

    async def on_enter(self) -> None:
        # The caller may have already said what they want before we speak -
        # pick up from there instead of re-asking "how can I help?".
        await self.session.generate_reply(
            instructions=(
                "Greet the caller in one short sentence. If they've already named a need "
                "(a table, a spa appointment, flowers...), move straight into helping; "
                "otherwise ask how you can help."
            )
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
        changing or cancelling it - and before a modification that keeps some details "the
        same", so you know the current values being kept.

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

    @function_tool
    async def modify_restaurant_reservation(
        self,
        ctx: RunContext[Userdata],
        last_name: str,
        confirmation_code: str,
        new_date: date,
        new_time: str,
        new_party_size: Annotated[int, Field(ge=1, le=MAX_PARTY_SIZE)] | None = None,
    ) -> str:
        """Move an existing confirmed restaurant reservation to a new date/time (and
        optionally a new party size), keeping the same confirmation code. Restaurants
        verify with last name + confirmation code (no card, no email). Read the new
        details back to the caller before calling this, and relay the party size from
        this tool's return when confirming - it's how a wrong count gets caught.

        Args:
            last_name: caller's last name.
            confirmation_code: confirmation code like 'RES-X9Y2'.
            new_date: the new date, in ISO YYYY-MM-DD format (e.g. "2026-01-20").
            new_time: the new time, in 24-hour HH:MM format (e.g. "18:00").
            new_party_size: new number of guests, ONLY when the caller states the new number. "Keep it the same" means OMIT this parameter - the reservation keeps its current size when omitted. Never fill it with a number the caller didn't say.
        """
        if new_date < TODAY:
            raise ToolError("the new date can't be in the past")
        code = confirmation_code.replace(" ", "").upper()
        reservation = await ctx.userdata.db.find_restaurant_reservation(
            last_name=last_name, confirmation_code=code
        )
        if not reservation or reservation.status != "confirmed":
            raise ToolError("Couldn't find a matching confirmed reservation.")
        try:
            at_time = time.fromisoformat(new_time)
        except ValueError:
            raise ToolError("Please give the new time as 24-hour HH:MM, e.g. 18:00.") from None
        try:
            updated = await ctx.userdata.db.modify_restaurant_reservation(
                code=reservation.code,
                on_date=new_date,
                at_time=at_time,
                party_size=new_party_size,
            )
        except Unavailable:
            raise ToolError(
                f"No table for a party of {new_party_size or reservation.party_size} "
                f"at {speak_time(at_time)} on {new_date.strftime('%A, %B %-d')}."
            ) from None
        return (
            f"Done - your reservation is now {speak_time(updated.time)} on "
            f"{updated.date.strftime('%A, %B %-d')} for "
            f"{updated.party_size} guest{'s' if updated.party_size != 1 else ''}, "
            f"under confirmation code {_speak_code(updated.code)}. "
            "| confirm the new date, time, AND the party size above to the caller - if the "
            "party size isn't what they expect, this is their chance to catch it."
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
        return (
            f"Reservation for {speak_time(reservation.time)} on "
            f"{reservation.date.strftime('%A, %B %-d')} cancelled."
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
                arrangement_id=arrangement,
                guest_name=guest_name,
                guest_phone=guest_phone,
                deliver_to=deliver_to,
                on_date=on_date,
                card_message=card_message,
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
async def hotel_amenities_agent(ctx: JobContext) -> None:
    await ctx.connect()

    userdata = Userdata(db=HotelDB.from_bytes(_SEED_DB_BYTES))
    session = AgentSession[Userdata](
        userdata=userdata,
        # Session-scoped, so every agent and every AgentTask in the call can reach
        # it - a caller can drop off mid-flow, and the followup has to be recordable there.
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

    await session.start(agent=AmenitiesAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
