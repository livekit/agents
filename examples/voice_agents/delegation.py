"""Fast brain / slow brain: a realtime model talks, a text model reasons.

Northwind Air's support line. The realtime model owns the conversation — turn-taking,
latency, barge-in — and is given exactly one tool, `lk_agents_delegate`. Every lookup,
fare rule, price and booking change goes to `delegation_llm`, which never speaks: it
returns facts the realtime model phrases.

The split is what makes this work. A realtime model asked to turn "next Monday" into a
date, identify a caller, read a disruption waiver off a fare table, price the change and
then call `rebook` with the right four arguments will get it wrong. A frontier text model
asked to hold a natural spoken conversation is slow and stilted. Here each does the half
it is good at, and none of the thirteen tools below reaches the model holding the
microphone — its list is exactly `[lk_agents_delegate]`. Nothing here is flagged
NO_DELEGATE; that is for tools too latency-critical to route through a frontier model,
like a DTMF digit that has to land while an IVR prompt is still playing.

Points worth reading:
  - the fare rules and the identification policy live in `policy()`, the delegation's
    instructions. The voice persona holds no domain knowledge at all, and is deliberately
    not told that a second model exists — call the other half "the fare desk" in its
    prompt and it starts deciding for itself that a weather question is out of scope.
    Everything goes to the tool; nothing is off-topic for it.
  - nothing here asks the voice model to say "let me check". `announce` defaults to off,
    which selects a `delegate` description that asks for one short line in the same turn
    as the call — free, in the same completion, and phrased so the caller never learns
    anyone was consulted. A model that already volunteers a line just keeps its own.
  - where the line falls between the two: the policy carries judgment — whether a caller
    has earned a goodwill discount at all — and DISCOUNT_TIERS carries entitlement.
    Putting the table in the prompt would only let a model read it wrong, and wrong
    inside the legal range is exactly what a ceiling check cannot catch. So
    `apply_discount` takes no percentage; it works one out and reports what it gave.
  - dates are the same shape of split. The timetable repeats every day, and both models
    are told what day it is — the desk so it can turn "next Monday" into the YYYY-MM-DD
    the tools accept, the voice model only so it can say "Friday" rather than read a date
    back. The arithmetic is the desk's, and the voice model is told to leave it alone.
  - `collect_email` is delegated even though it talks. It runs a GetEmailTask, which
    takes the conversation over to spell an address back and confirm it — so the desk
    reaches out and drives a piece of the conversation itself, then keeps reasoning with
    the answer in the same loop. `ctx.foreground()` is what makes that safe: it waits for
    the line to go quiet and holds it, so the spelling never collides with the voice
    model. The caller is never asked to read out a booking reference.
  - a delegation is stateless — built fresh from the conversation each time, and its own
    tool calls are recorded in neither history — so the desk cannot remember having asked
    for anything. A tool that talks to the caller therefore has to be idempotent about it:
    `collect_email` answers off `Userdata` when an address is already on file, which is
    what stops the second delegation asking all over again. Anything a tool says out loud
    needs that guard; anything it merely computes does not.
  - `rebook` and `book_flight` are CANCELLABLE with on_duplicate="replace", so "actually,
    make it Thursday" cancels the booking in flight instead of making a second one. Their
    `ctx.update()` reaches the caller as a progress line the voice model rephrases, and
    the delegation is given lk_agents_cancel_task automatically because of them.
  - the mock airline is per session, in Userdata. Seats are per departure and really do
    get decremented, and the seeded bookings sit a few days out from whenever you run it.

Try it with `python delegation.py console`:
  - "my flight to Tokyo is delayed, can you get me out tomorrow instead"
    → one delegation that asks for the address itself, then looks the caller up, checks
      the flight, checks the Haneda forecast — there is a typhoon warning — searches,
      quotes with the delay waiver applied, and rebooks. Eight seconds that narrate
      themselves.
  - "can I get a refund on my other trip"
    → two bookings, so the desk hands back the list and the voice model asks which;
      the answer for the BASIC one back from Paris is no, and it says so without
      offering to try.
  - "that's really disappointing, is there anything you can do on the price"
    → the desk decides they have earned one and asks for it; 15 for Gold on a SAVER
      fare, plus 5 because we delayed them, comes back from the tool rather than from it.
  - "I need to get to Beijing a week on Tuesday"
    → the desk does the calendar arithmetic and searches that day. Westbound out of SFO
      it also has to say you land the following afternoon.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    ToolError,
    ToolExecutionUpdatedEvent,
    cli,
    inference,
)
from livekit.agents.beta.workflows import GetEmailTask
from livekit.agents.llm import DELEGATE_TOOL_NAME, ToolFlag, function_tool
from livekit.plugins import openai

logger = logging.getLogger("delegation")

load_dotenv()


# cheapest to dearest — the order the rules compare buckets in
FARE_BUCKETS = ("BASIC", "SAVER", "FLEX", "BUSINESS")

# what a caller is entitled to, by loyalty status and the fare they hold. deliberately not
# in the policy: reading a number off a table is not reasoning, and a model that reads it
# wrong is wrong within the legal range. the desk decides whether to give one, not how much
# fmt: off
DISCOUNT_TIERS: dict[str, dict[str, int]] = {
    "Blue":   {"BASIC": 0,  "SAVER": 5,  "FLEX": 5,  "BUSINESS": 10},
    "Silver": {"BASIC": 5,  "SAVER": 10, "FLEX": 10, "BUSINESS": 15},
    "Gold":   {"BASIC": 10, "SAVER": 15, "FLEX": 15, "BUSINESS": 20},
}
# fmt: on
DISRUPTION_BONUS = 5
MAX_DISCOUNT = 25


def policy(today: date) -> str:
    """The desk's instructions. Dated, because the caller will say "next Monday"."""
    return f"""You are the fare desk for Northwind Air. You never speak to the caller; the
phone agent does. Work out what is true, hand back the facts and the numbers, and say
what the phone agent should tell them.

DATES
Today is {today:%Y-%m-%d}, a {today:%A}. Callers talk in "tomorrow", "next Monday", "the
week after next" — never in calendar dates, and the phone agent is told not to convert
them. That arithmetic is yours. Every tool takes YYYY-MM-DD and refuses a date in the
past. The timetable is the same every day, so a route that flies at all flies on any day
they ask for; if a day is no good it is because that flight is full, not because it does
not operate.

IDENTIFYING THE CALLER
Most of what people ring about needs no identity at all. The timetable, fares, seats,
whether a flight is running and the weather are public: search_flights, flight_status and
check_weather never want an address. Answer those and do not ask who is calling.
An address is needed only to reach an account — to see, price or change a booking, or to
sell one. Start with lookup_caller, which either comes back with the caller and their
bookings or tells you there is no address yet; only then run collect_email. Never open
with collect_email, and do not call it a second time to be sure: someone already
identified is handed straight back, and asking twice in one call is the rudest thing you
can do to them.
Bookings are found from that address, never from a reference read out loud, and never
from a guess off the transcript. A first-time caller needs one too — it is what
book_flight opens their account with — but collect it when they are ready to buy, not
while they are still looking.
If the caller has more than one booking, do not pick one for them. Hand back the list by
route, date and flight number — never the reference — and say what to ask.

FARE BUCKETS, cheapest to dearest: BASIC, SAVER, FLEX, BUSINESS.
  BASIC     no changes, no refund. 1 cabin bag, checked bags 45 USD each.
  SAVER     changes for 75 USD plus the fare difference; refund as travel credit.
            1 cabin bag, checked bags 35 USD each.
  FLEX      free changes, refundable to the original card, 1 checked bag included.
  BUSINESS  free changes, refundable, 2 checked bags included.

CHANGES
The change fee is waived when we cancelled the flight or delayed it more than three
hours, and on any fare for a Gold member. A fare difference is never waived.
Quote before you commit: quote_change first, rebook second. Never rebook into a cheaper
bucket than the one held, and never into a flight with no seats in that bucket.

REFUNDS
A BASIC ticket is never refundable — say so plainly rather than offering to try. A SAVER
refund is travel credit, unless we cancelled the flight or delayed it more than three
hours, in which case it goes back to the card.

GOODWILL DISCOUNTS
Only when the caller asks for one, or when we cancelled on them. Never volunteer one.
Whether they get one is your call; how much is not, and you do not know the figure — the
desk works it out from their status, their fare and whether we disrupted the flight, and
tells you what it gave. Never name a number before then. It comes off what they owe on a
change or a new booking, never off a segment already flown, so quote the change first.

DISRUPTION
When a flight is delayed or cancelled, check the weather at both ends before you
recommend an alternative — putting someone on the next flight into a storm is worse than
the delay. Say what you checked."""


@dataclass
class Route:
    """A line in the timetable. The same flight leaves at the same time every day."""

    flight_no: str
    origin: str
    destination: str
    departs: str  # local HH:MM
    arrives: str
    seats: dict[str, int]  # what a day starts with, before anyone books
    fares: dict[str, float]
    # westbound out of SFO lands the next calendar day; coming back east you arrive the
    # same day you left, and the desk has to be able to tell a caller which
    arrives_next_day: bool = False


@dataclass
class Departure:
    """One route on one day: the seats left on it, and whether we have broken it."""

    route: Route
    date: str
    seats: dict[str, int]
    delay_minutes: int = 0
    cancelled: bool = False

    @property
    def departs(self) -> str:
        return f"{self.date} {self.route.departs}"

    @property
    def arrives(self) -> str:
        landing = date.fromisoformat(self.date) + timedelta(days=int(self.route.arrives_next_day))
        return f"{landing} {self.route.arrives}"

    @property
    def disrupted(self) -> bool:
        """Airline fault as the fare rules define it: cancelled, or over three hours late."""
        return self.cancelled or self.delay_minutes > 180

    @property
    def status(self) -> str:
        if self.cancelled:
            return "cancelled"
        if self.delay_minutes:
            return f"delayed {self.delay_minutes} minutes"
        return "on time"


@dataclass
class Traveler:
    email: str
    name: str
    status: str  # Blue, Silver or Gold
    credit_usd: float = 0.0


@dataclass
class Booking:
    ref: str
    email: str
    flight_no: str
    date: str
    fare: str
    passengers: int
    checked_bags: int
    paid_usd: float
    state: str = "confirmed"
    # set by quote_change and apply_discount, consumed by rebook
    pending_quote_usd: float | None = None
    discount_pct: int = 0


@dataclass
class Airline:
    routes: dict[str, Route]
    travelers: dict[str, Traveler]
    bookings: dict[str, Booking]
    # every day of the timetable is identical until someone touches it, so a day's
    # inventory is opened lazily and only the days in play are ever held
    departures: dict[tuple[str, str], Departure] = field(default_factory=dict)

    def departure(self, route: Route, day: str) -> Departure:
        key = (route.flight_no, day)
        if key not in self.departures:
            self.departures[key] = Departure(route, day, dict(route.seats))
        return self.departures[key]


def seed_airline() -> Airline:
    """A fresh mock airline, dated off today. One per session, since the tools mutate it."""
    today = date.today()
    # long-haul out of San Francisco: Tokyo, Paris and Beijing, each with a return leg
    # fmt: off
    routes = [
        Route("NW808", "SFO", "HND", "11:30", "14:45",
              {"SAVER": 1, "FLEX": 3}, {"SAVER": 742.0, "FLEX": 1180.0},
              arrives_next_day=True),
        Route("NW812", "SFO", "HND", "17:40", "21:05",
              {"BASIC": 4, "SAVER": 6, "FLEX": 2},
              {"BASIC": 615.0, "SAVER": 698.0, "FLEX": 1120.0}, arrives_next_day=True),
        Route("NW809", "HND", "SFO", "17:00", "10:30",
              {"SAVER": 3, "FLEX": 2, "BUSINESS": 2},
              {"SAVER": 755.0, "FLEX": 1210.0, "BUSINESS": 3480.0}),
        Route("NW440", "SFO", "CDG", "15:20", "11:05",
              {"BASIC": 0, "SAVER": 5, "FLEX": 4, "BUSINESS": 2},
              {"BASIC": 588.0, "SAVER": 690.0, "FLEX": 1340.0, "BUSINESS": 3960.0},
              arrives_next_day=True),
        Route("NW441", "CDG", "SFO", "13:10", "15:40",
              {"BASIC": 2, "SAVER": 4, "FLEX": 3},
              {"BASIC": 566.0, "SAVER": 705.0, "FLEX": 1290.0}),
        Route("NW620", "SFO", "PEK", "13:45", "17:20",
              {"BASIC": 2, "SAVER": 5, "BUSINESS": 1},
              {"BASIC": 640.0, "SAVER": 735.0, "BUSINESS": 3620.0}, arrives_next_day=True),
        Route("NW621", "PEK", "SFO", "16:30", "12:15",
              {"SAVER": 4, "FLEX": 2}, {"SAVER": 748.0, "FLEX": 1265.0}),
    ]
    # fmt: on
    travelers = [
        Traveler("dana.whitfield@example.com", "Dana Whitfield", "Gold"),
        Traveler("m.ortiz@example.com", "Miguel Ortiz", "Blue"),
        Traveler("priya.raman@example.com", "Priya Raman", "Silver", credit_usd=120.0),
    ]

    def day(offset: int) -> str:
        return (today + timedelta(days=offset)).isoformat()

    bookings = [
        # two bookings on one address: the desk has to ask which one
        Booking("NW7Q2K", "dana.whitfield@example.com", "NW808", day(1), "SAVER", 1, 0, 742.0),
        Booking("NW3H8L", "dana.whitfield@example.com", "NW441", day(30), "BASIC", 1, 1, 566.0),
        Booking("NW5T4P", "m.ortiz@example.com", "NW812", day(5), "BASIC", 2, 0, 1230.0),
        Booking("NW8W6C", "priya.raman@example.com", "NW620", day(12), "BUSINESS", 1, 2, 3620.0),
    ]
    airline = Airline(
        routes={r.flight_no: r for r in routes},
        travelers={t.email: t for t in travelers},
        bookings={b.ref: b for b in bookings},
    )
    # the one broken flight in the schedule: tomorrow's Tokyo run, which Dana is on
    airline.departure(airline.routes["NW808"], day(1)).delay_minutes = 245
    return airline


# stand-in for a forecast service, by airport and days from today, so the storm that makes
# the disrupted flight interesting lands on the same day the caller is asking about
FORECASTS: dict[tuple[str, int], tuple[str, int, int, str]] = {
    ("HND", 1): ("typhoon warning", 28, 24, "high"),
    ("HND", 2): ("clearing, humid", 31, 25, "low"),
    ("SFO", 1): ("morning fog", 19, 13, "moderate"),
    ("CDG", 30): ("cool and drizzly", 16, 11, "moderate"),
    ("PEK", 12): ("hazy, light wind", 33, 23, "low"),
}
DEFAULT_FORECAST = ("clear", 24, 14, "low")


@dataclass
class Userdata:
    airline: Airline
    # confirmed by collect_email. the delegation is stateless and its tool calls land in
    # no conversation store, so the address lives here to survive into the next one
    email: str = ""
    events: list[str] = field(default_factory=list)


def _day(value: str) -> str:
    """Normalize a date the desk supplied, refusing anything but a future YYYY-MM-DD."""
    try:
        parsed = date.fromisoformat(value.strip())
    except ValueError:
        raise ToolError(f"{value!r} is not a date — resolve it to YYYY-MM-DD first") from None

    if parsed < date.today():
        raise ToolError(f"{parsed} has already gone; today is {date.today()}")
    return parsed.isoformat()


def _departure(userdata: Userdata, flight_no: str, day: str) -> Departure:
    route = userdata.airline.routes.get(flight_no.upper().replace(" ", ""))
    if route is None:
        raise ToolError(f"no flight {flight_no} in the timetable")
    return userdata.airline.departure(route, day)


def _booking(userdata: Userdata, booking_ref: str) -> Booking:
    booking = userdata.airline.bookings.get(booking_ref.upper().replace(" ", ""))
    if booking is None:
        raise ToolError(f"no booking {booking_ref} — find the caller's bookings with lookup_caller")
    return booking


def _booked_departure(userdata: Userdata, booking: Booking) -> Departure:
    """The flight a booking is on. Not date-checked — a booking may have already flown."""
    return userdata.airline.departure(userdata.airline.routes[booking.flight_no], booking.date)


def _short(text: str | None, limit: int = 90) -> str:
    """One clipped line. Tool payloads are long and it is the shape of the call that reads."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _discount_percent(traveler: Traveler, bucket: str, disrupted: bool) -> int:
    """The caller's tier, plus a bump when the flight was our fault."""
    percent = DISCOUNT_TIERS[traveler.status][bucket]
    if disrupted:
        percent += DISRUPTION_BONUS
    return min(percent, MAX_DISCOUNT)


def _describe(userdata: Userdata, booking: Booking) -> dict[str, Any]:
    departure = _booked_departure(userdata, booking)
    return {
        "booking_ref": booking.ref,
        "flight": booking.flight_no,
        "route": f"{departure.route.origin}-{departure.route.destination}",
        "departs": departure.departs,
        "fare_bucket": booking.fare,
        "passengers": booking.passengers,
        "state": booking.state,
        "flight_status": departure.status,
    }


class SupportAgent(Agent):
    def __init__(self) -> None:
        # both models are told what day it is: the desk to do the arithmetic, the voice
        # model so it can say "Friday" instead of reading a date back
        today = date.today()
        super().__init__(
            # purely conversational, and deliberately not told who does the rest. calling the
            # other half "the fare desk" here would invite it to decide a weather question is
            # out of scope and answer from memory instead of delegating
            instructions=(
                f"Today is {today:%A, %Y-%m-%d}. "
                "You are the voice of Northwind Air's support line. Keep every reply to one "
                "or two short sentences. You are on the phone, so no emojis, asterisks or "
                "markdown, and never read a booking reference out loud. "
                "You do not look anything up or work anything out yourself. Every request "
                "goes to the delegate tool — flights, prices, rules, bookings, dates, "
                "baggage, the weather, whatever the caller asks — and you say what comes "
                "back in your own words. Nothing is off-topic for it, so never decide by "
                "yourself that something is out of scope or cannot be done, and never ask "
                "the caller for personal details: if any are needed they will be asked for "
                "as part of the work. "
                "Say dates the way a person would, 'Friday the 31st', but never convert one "
                "yourself — pass on the caller's own words. If what comes back lists more "
                "than one booking, ask which flight they mean by route and date."
            ),
            # # the conversation model. its tool list is just lk_agents_delegate
            llm=openai.realtime.RealtimeModel(model="gpt-realtime-2"),
            # the reasoning model: every tool below belongs to it
            delegation_llm=inference.LLM("google/gemma-4-31b-it"),
            delegation_options={"instructions": policy(today), "announce": False},
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="greet the caller as Northwind Air and ask how you can help"
        )

    @function_tool
    async def collect_email(self, ctx: RunContext[Userdata], change: bool = False) -> str:
        """Ask the caller for their email address, reading it back to confirm it.

        This one talks: it takes the floor for as long as that exchange lasts. Reach for
        it only once a tool has told you it needs an address, never to open with, and
        never for a question that is not about a specific booking.

        Args:
            change: only when the caller wants a different address from the one already
                on file. Left alone, someone already identified is never asked twice.
        """
        userdata = ctx.userdata
        # a delegation cannot remember having asked: each one is built fresh from the
        # conversation, and the desk's own tool calls are recorded in neither history. so
        # the answer has to come off Userdata, or the next delegation asks all over again
        if userdata.email and not change:
            return f"already confirmed earlier in this call: {userdata.email}"

        # foreground() waits for the line to go quiet and then holds it, so the spelling
        # never collides with whatever the voice model is in the middle of saying
        async with ctx.foreground():
            result = await GetEmailTask(chat_ctx=self.chat_ctx, llm=self.llm)

        userdata.email = result.email_address.strip().lower()
        logger.info(f"caller identified as {userdata.email}")
        return f"confirmed with the caller: {userdata.email}"

    @function_tool
    async def lookup_caller(self, ctx: RunContext[Userdata]) -> dict[str, Any]:
        """Identify the caller and list every booking on their account.

        Takes no arguments on purpose: the caller is whoever owns the address collect_email
        confirmed with them, never a reference read out over a phone line.
        """
        userdata = ctx.userdata
        if not userdata.email:
            raise ToolError("no address on file — run collect_email first")

        traveler = userdata.airline.travelers.get(userdata.email)
        if traveler is None:
            # a first-time caller. book_flight opens an account for them
            return {"email": userdata.email, "known_traveler": False, "bookings": []}

        bookings = [
            _describe(userdata, b)
            for b in userdata.airline.bookings.values()
            if b.email == traveler.email
        ]
        return {
            "email": traveler.email,
            "known_traveler": True,
            "name": traveler.name,
            "loyalty_status": traveler.status,
            "travel_credit_usd": traveler.credit_usd,
            "bookings": bookings,
        }

    @function_tool
    async def search_flights(
        self,
        ctx: RunContext[Userdata],
        origin: str,
        destination: str,
        date: str,
        min_bucket: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find flights on a route and day, with the buckets that still have seats.

        Public: the timetable is the same for everyone, so do not identify the caller
        first.

        Args:
            origin: departure airport, three-letter code.
            destination: arrival airport, three-letter code.
            date: the departure day as YYYY-MM-DD. Work out the caller's "next Monday"
                yourself; this will not take it.
            min_bucket: only return flights with seats in this bucket or better.
        """
        userdata = ctx.userdata
        day = _day(date)
        logger.info(f"searching {origin}->{destination} on {day}")
        await asyncio.sleep(1.5)  # stand-in for the inventory system

        floor = FARE_BUCKETS.index(min_bucket.upper()) if min_bucket else 0
        results = []
        for route in userdata.airline.routes.values():
            if (route.origin, route.destination) != (origin.upper(), destination.upper()):
                continue

            departure = userdata.airline.departure(route, day)
            available = {
                bucket: {"seats": seats, "fare_usd": route.fares[bucket]}
                for bucket, seats in departure.seats.items()
                if seats > 0 and FARE_BUCKETS.index(bucket) >= floor
            }
            if available:
                results.append(
                    {
                        "flight": route.flight_no,
                        "departs": departure.departs,
                        "arrives": departure.arrives,
                        "status": departure.status,
                        "available": available,
                    }
                )
        return results

    @function_tool
    async def flight_status(
        self, ctx: RunContext[Userdata], flight_no: str, date: str
    ) -> dict[str, Any]:
        """Whether a flight is running on a given day, and how late it is.

        Public: anyone can ask about any flight, so do not identify the caller first.

        Args:
            flight_no: the flight to check.
            date: the departure day, YYYY-MM-DD.
        """
        userdata = ctx.userdata
        departure = _departure(userdata, flight_no, _day(date))
        return {
            "flight": departure.route.flight_no,
            "route": f"{departure.route.origin}-{departure.route.destination}",
            "scheduled_departure": departure.departs,
            "status": departure.status,
            "delay_minutes": departure.delay_minutes,
            "cancelled": departure.cancelled,
            "airline_at_fault": departure.disrupted,
        }

    @function_tool
    async def check_weather(
        self, ctx: RunContext[Userdata], airport: str, date: str
    ) -> dict[str, Any]:
        """The forecast at an airport on a day, with how likely it is to disrupt flying.

        Public: do not identify the caller first.

        Args:
            airport: three-letter code.
            date: YYYY-MM-DD.
        """
        day = _day(date)
        await asyncio.sleep(0.5)

        # datetime, not the date class: the `date` argument shadows it here
        today = datetime.now().date()
        offset = (datetime.strptime(day, "%Y-%m-%d").date() - today).days
        conditions, high_c, low_c, risk = FORECASTS.get((airport.upper(), offset), DEFAULT_FORECAST)
        return {
            "airport": airport.upper(),
            "date": day,
            "conditions": conditions,
            "high_c": high_c,
            "low_c": low_c,
            "disruption_risk": risk,
        }

    @function_tool
    async def quote_change(
        self,
        ctx: RunContext[Userdata],
        booking_ref: str,
        new_flight_no: str,
        new_date: str,
        bucket: str,
    ) -> dict[str, Any]:
        """Price a move to another flight without committing to it. Run this before rebook.

        Args:
            booking_ref: from lookup_caller, not from the caller.
            new_flight_no: the flight to move onto.
            new_date: the day to move onto, YYYY-MM-DD.
            bucket: the fare bucket to move into.
        """
        userdata = ctx.userdata
        booking = _booking(userdata, booking_ref)
        old = _booked_departure(userdata, booking)
        new = _departure(userdata, new_flight_no, _day(new_date))
        bucket = bucket.upper()

        if bucket not in new.route.fares:
            raise ToolError(f"{new.route.flight_no} does not sell {bucket}")

        traveler = userdata.airline.travelers[booking.email]
        waived = old.disrupted or traveler.status == "Gold"
        fee = 0.0 if booking.fare in ("FLEX", "BUSINESS") or waived else 75.0
        difference = max(0.0, new.route.fares[bucket] * booking.passengers - booking.paid_usd)

        booking.pending_quote_usd = round(fee + difference, 2)
        booking.discount_pct = 0
        return {
            "booking_ref": booking.ref,
            "new_flight": new.route.flight_no,
            "new_departure": new.departs,
            "bucket": bucket,
            "seats_left": new.seats.get(bucket, 0),
            "change_fee_usd": fee,
            "fee_waived": waived,
            "fare_difference_usd": round(difference, 2),
            "total_due_usd": booking.pending_quote_usd,
            "hold_expires_in_minutes": 20,
        }

    @function_tool
    async def apply_discount(
        self, ctx: RunContext[Userdata], booking_ref: str, reason: str
    ) -> dict[str, Any]:
        """Take the caller's goodwill discount off what a quoted change costs.

        How much is not yours to pick: it comes out of the caller's status, the fare they
        hold and whether we disrupted the flight. Whether to offer one at all is yours.
        Refuses until quote_change has run.

        Args:
            booking_ref: from lookup_caller.
            reason: what the caller is being compensated for. It prints on the invoice.
        """
        userdata = ctx.userdata
        booking = _booking(userdata, booking_ref)
        if booking.pending_quote_usd is None:
            raise ToolError("nothing is quoted on this booking yet — run quote_change first")

        traveler = userdata.airline.travelers[booking.email]
        disrupted = _booked_departure(userdata, booking).disrupted
        percent = _discount_percent(traveler, booking.fare, disrupted)
        if percent == 0:
            raise ToolError(
                f"a {traveler.status} member on a {booking.fare} fare has nothing to give"
            )

        booking.discount_pct = percent
        return {
            "booking_ref": booking.ref,
            "discount_percent": percent,
            "based_on": {
                "loyalty_status": traveler.status,
                "fare_bucket": booking.fare,
                "airline_disrupted": disrupted,
            },
            "was_due_usd": booking.pending_quote_usd,
            "now_due_usd": round(booking.pending_quote_usd * (1 - percent / 100), 2),
            "reason": reason,
        }

    # cancellable and replaceable: "actually, make it Thursday" halfway through cancels this
    # call rather than booking twice. the desk also gets lk_agents_cancel_task because of it
    @function_tool(flags=ToolFlag.CANCELLABLE, on_duplicate="replace")
    async def rebook(
        self,
        ctx: RunContext[Userdata],
        booking_ref: str,
        new_flight_no: str,
        new_date: str,
        bucket: str,
    ) -> dict[str, Any]:
        """Move a booking onto another flight and charge the quoted amount.

        Args:
            booking_ref: from lookup_caller.
            new_flight_no: the flight to move onto, quoted already.
            new_date: the day to move onto, YYYY-MM-DD.
            bucket: the fare bucket to move into. Never below the one held.
        """
        userdata = ctx.userdata
        booking = _booking(userdata, booking_ref)
        new = _departure(userdata, new_flight_no, _day(new_date))
        bucket = bucket.upper()

        if booking.state != "confirmed":
            raise ToolError(f"{booking.ref} is {booking.state} and cannot be changed")
        if new.seats.get(bucket, 0) < booking.passengers:
            raise ToolError(f"{new.route.flight_no} on {new.date} has no {bucket} seats left")

        # reaches the caller as a progress line, re-attributed to the delegate call
        await ctx.update(f"holding a seat on {new.route.flight_no}")
        await asyncio.sleep(8)  # inventory hold, reissue, payment capture

        old = _booked_departure(userdata, booking)
        old.seats[booking.fare] = old.seats.get(booking.fare, 0) + booking.passengers
        new.seats[bucket] -= booking.passengers

        charged = round((booking.pending_quote_usd or 0.0) * (1 - booking.discount_pct / 100), 2)
        booking.flight_no = new.route.flight_no
        booking.date = new.date
        booking.fare = bucket
        booking.paid_usd += charged
        booking.pending_quote_usd = None
        booking.discount_pct = 0

        userdata.events.append(f"{booking.ref} moved to {new.departs} for {charged} USD")
        return {
            "booking_ref": booking.ref,
            "flight": new.route.flight_no,
            "bucket": bucket,
            "departs": new.departs,
            "charged_usd": charged,
        }

    @function_tool(flags=ToolFlag.CANCELLABLE, on_duplicate="replace")
    async def book_flight(
        self,
        ctx: RunContext[Userdata],
        flight_no: str,
        date: str,
        bucket: str,
        passengers: int,
        goodwill_discount: bool = False,
    ) -> dict[str, Any]:
        """Sell the caller a new ticket, using up any travel credit they hold.

        Opens an account for a first-time caller. Needs an address on file.

        Args:
            flight_no: the flight to sell.
            date: the departure day, YYYY-MM-DD.
            bucket: the fare bucket to sell.
            passengers: how many seats.
            goodwill_discount: whether to take the caller's discount off this fare. How
                much comes from their status and the bucket, not from you.
        """
        userdata = ctx.userdata
        departure = _departure(userdata, flight_no, _day(date))
        bucket = bucket.upper()

        if not userdata.email:
            raise ToolError("no address on file — run collect_email first")
        if bucket not in departure.route.fares:
            raise ToolError(f"{departure.route.flight_no} does not sell {bucket}")
        if departure.seats.get(bucket, 0) < passengers:
            raise ToolError(
                f"{departure.departs} has {departure.seats.get(bucket, 0)} {bucket} seats left"
            )

        traveler = userdata.airline.travelers.setdefault(
            userdata.email, Traveler(userdata.email, "new customer", "Blue")
        )
        percent = (
            _discount_percent(traveler, bucket, departure.disrupted) if goodwill_discount else 0
        )

        await ctx.update(f"holding {passengers} on {departure.route.flight_no}")
        await asyncio.sleep(4)  # inventory hold, payment capture

        fare = departure.route.fares[bucket] * passengers * (1 - percent / 100)
        credit_used = min(traveler.credit_usd, fare)
        traveler.credit_usd -= credit_used
        departure.seats[bucket] -= passengers

        booking = Booking(
            ref=f"NW{uuid.uuid4().hex[:4].upper()}",
            email=traveler.email,
            flight_no=departure.route.flight_no,
            date=departure.date,
            fare=bucket,
            passengers=passengers,
            checked_bags=0,
            # what went on the card. credit spent here does not come back as cash on a refund
            paid_usd=round(fare - credit_used, 2),
        )
        userdata.airline.bookings[booking.ref] = booking
        userdata.events.append(f"{booking.ref} booked on {departure.departs}")
        return {
            "booking_ref": booking.ref,
            "flight": departure.route.flight_no,
            "departs": departure.departs,
            "bucket": bucket,
            "passengers": passengers,
            "discount_percent": percent,
            "charged_usd": round(fare - credit_used, 2),
            "travel_credit_used_usd": round(credit_used, 2),
        }

    @function_tool
    async def refund(
        self, ctx: RunContext[Userdata], booking_ref: str, reason: str
    ) -> dict[str, Any]:
        """Cancel a booking and refund it. Check the fare rules allow it before promising.

        Args:
            booking_ref: from lookup_caller.
            reason: why it is being refunded.
        """
        userdata = ctx.userdata
        booking = _booking(userdata, booking_ref)
        departure = _booked_departure(userdata, booking)
        traveler = userdata.airline.travelers[booking.email]

        if booking.state != "confirmed":
            raise ToolError(f"{booking.ref} is already {booking.state}")
        if booking.fare == "BASIC":
            raise ToolError("BASIC fares are not refundable, as credit or otherwise")

        as_credit = booking.fare == "SAVER" and not departure.disrupted
        amount = round(booking.paid_usd, 2)
        booking.state = "refunded"
        departure.seats[booking.fare] = departure.seats.get(booking.fare, 0) + booking.passengers
        if as_credit:
            traveler.credit_usd += amount

        userdata.events.append(f"{booking.ref} refunded ({'credit' if as_credit else 'card'})")
        return {
            "booking_ref": booking.ref,
            "amount_usd": amount,
            "as_travel_credit": as_credit,
            "travel_credit_balance_usd": round(traveler.credit_usd, 2),
            "settles_in_days": 1 if as_credit else 7,
            "reason": reason,
        }

    @function_tool
    async def baggage_allowance(
        self, ctx: RunContext[Userdata], booking_ref: str
    ) -> dict[str, Any]:
        """What a booking is allowed to carry, and what another bag would cost."""
        booking = _booking(ctx.userdata, booking_ref)
        included = {"BASIC": 0, "SAVER": 0, "FLEX": 1, "BUSINESS": 2}[booking.fare]
        return {
            "booking_ref": booking.ref,
            "fare_bucket": booking.fare,
            "cabin_bags": 1,
            "checked_bags_included": included * booking.passengers,
            "checked_bags_purchased": booking.checked_bags,
            "extra_bag_usd": 35.0 if booking.fare != "BASIC" else 45.0,
            "max_bag_weight_kg": 23,
        }

    @function_tool
    async def add_checked_bags(
        self, ctx: RunContext[Userdata], booking_ref: str, bags: int
    ) -> dict[str, Any]:
        """Buy checked bags on a booking, over and above whatever the fare includes.

        Args:
            booking_ref: from lookup_caller.
            bags: how many bags to add.
        """
        booking = _booking(ctx.userdata, booking_ref)
        if bags < 1:
            raise ToolError("add at least one bag")

        price = (35.0 if booking.fare != "BASIC" else 45.0) * bags
        booking.checked_bags += bags
        booking.paid_usd += price
        ctx.userdata.events.append(f"{booking.ref} added {bags} bag(s)")
        return {
            "booking_ref": booking.ref,
            "checked_bags_purchased": booking.checked_bags,
            "charged_usd": price,
        }

    @function_tool
    async def email_itinerary(self, ctx: RunContext[Userdata], booking_ref: str) -> str:
        """Send the current itinerary to the address on file."""
        booking = _booking(ctx.userdata, booking_ref)
        if not ctx.userdata.email:
            raise ToolError("no address on file — run collect_email first")

        await asyncio.sleep(1)
        return f"itinerary for {booking.flight_no} on {booking.date} sent to {ctx.userdata.email}"


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    userdata = Userdata(airline=seed_airline())
    session = AgentSession[Userdata](userdata=userdata, max_tool_steps=8)

    # Both levels come through this one event, so it reads as a trace of the hand-off:
    # what the voice model asked for, each step the desk took, and what went back. The
    # desk's own calls are here and nowhere else — they land in neither chat history.
    #
    #   ── delegated: caller's flight to Tokyo is delayed, find them something tomorrow
    #        desk → lookup_caller({})
    #             ← done: {'name': 'Dana Whitfield', 'loyalty_status': 'Gold', ...}
    #        desk → check_weather({"airport": "HND", "date": "2026-07-31"})
    #             ← done: {'conditions': 'typhoon warning', ...}
    #        desk → rebook({"booking_ref": "NW7Q2K", "new_flight_no": "NW812", ...})
    #             … rebook: holding a seat on NW812
    #             ← done: {'booking_ref': 'NW7Q2K', 'charged_usd': 302.4}
    #   ── answered: moved to NW812, 302.40 charged, the delay waived the fee
    tool_names: dict[str, str] = {}

    @session.on("tool_execution_updated")
    def _on_tool_execution_updated(ev: ToolExecutionUpdatedEvent) -> None:
        update = ev.update
        if update.type == "tool_call_started":
            call = update.function_call
            tool_names[call.call_id] = call.name
            if call.name != DELEGATE_TOOL_NAME:
                logger.info(f"     desk → {call.name}({_short(call.arguments)})")
                return
            try:
                task = json.loads(call.arguments or "{}").get("task", "")
            except ValueError:
                task = call.arguments
            logger.info(f"── delegated: {_short(task, 200)}")
        elif update.type == "tool_call_updated":
            name = tool_names.get(update.call_id, "?")
            # a delegated tool's progress is emitted twice: once under the tool's own call,
            # then again re-attributed to the delegate call, which is the copy the voice
            # model rephrases. the first names the tool, so the second only repeats it.
            # a silent update is recorded for the model but never spoken — the dispatch
            # note with `announce` off — and is not worth a line either way
            if update.silent or name == DELEGATE_TOOL_NAME:
                return
            logger.info(f"          … {name}: {_short(update.message)}")
        elif update.type == "tool_call_ended":
            if tool_names.pop(update.call_id, "") == DELEGATE_TOOL_NAME:
                logger.info(f"── answered: {_short(update.message, 200)}")
            else:
                logger.info(f"          ← {update.status}: {_short(update.message)}")

    async def log_usage() -> None:
        logger.info(f"Usage: {session.usage}, events: {userdata.events}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(agent=SupportAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
