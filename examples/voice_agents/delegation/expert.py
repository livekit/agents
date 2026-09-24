"""The fare desk: Northwind Air's expert, served over A2A with no voice of its own.

Run it alongside `voice.py`, which talks to it:

    python expert.py dev

It serves one endpoint on the agent server's HTTP app:

    curl localhost:8321/fare-desk/.well-known/agent-card.json

Anything that speaks A2A can use it, ours or not. To drive it by hand, with the version
header the binding requires:

    curl -X POST localhost:8321/fare-desk/v1/message:stream -N \
         -H 'content-type: application/json' -H 'A2A-Version: 1.0' \
         -d '{"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text":
              "what flights are there from SFO to Tokyo next Monday?"}]}}'

The desk never speaks to the caller. It works out what is true, hands back the facts and the
numbers, and says what the phone agent should tell them. It holds each context whole:
one session per contextId, so the second request sees what the first one did.

`ctx.update()` inside a tool reports while the work is still running and releases the turn,
so the phone agent can say "holding a seat" while the seat is being held. That report is
relayed as the tool wrote it rather than handed to a model to restate.

With LIVEKIT_AGENTDB_URL set, each context persists as a session in the conversation the caller
names and survives a restart of the desk; see the README's "Persistence" section.
"""

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    RunContext,
    ToolError,
    ToolExecutionUpdatedEvent,
    cli,
    inference,
    store,
)
from livekit.agents.a2a import REQUEST_ID_KEY, A2ASessionContext
from livekit.agents.llm import ToolFlag, function_tool

logger = logging.getLogger("fare-desk")
logging.getLogger("a2a").setLevel(logging.INFO)
logging.getLogger("sse_starlette").setLevel(logging.INFO)

load_dotenv()

# pinned, because voice.py needs a fixed address to reach: the port otherwise defaults to a
# random one in dev
server = AgentServer(port=8321)

# without agent-db the desk keeps contexts in memory; the short lease lets a desk restarted
# after a crash take one back within seconds
AGENTDB_URL = os.environ.get("LIVEKIT_AGENTDB_URL")
# devLocal serves its data plane on a port of its own, set as LIVEKIT_AGENTDB_WS_URL
# todo: devLocal should accept the project key; until then a local agent-db takes its own
LOCAL_KEY = (
    {"api_key": "devkey", "api_secret": "secret"}
    if AGENTDB_URL and "localhost" in AGENTDB_URL
    else {}
)
DB = (
    store.AgentDB(ws_url=os.environ.get("LIVEKIT_AGENTDB_WS_URL"), lease_ttl=10, **LOCAL_KEY)
    if AGENTDB_URL
    else None
)


# cheapest to dearest — the order the rules compare buckets in
FARE_BUCKETS = ("BASIC", "SAVER", "FLEX", "BUSINESS")

# what a caller is entitled to on a new fare, by loyalty status and the bucket they buy.
# deliberately not in the policy: reading a number off a table is not reasoning, and a model
# that reads it wrong is wrong within the legal range. the desk decides whether, not how much
# fmt: off
DISCOUNT_TIERS: dict[str, dict[str, int]] = {
    "Blue":   {"BASIC": 0,  "SAVER": 5,  "FLEX": 5,  "BUSINESS": 10},
    "Silver": {"BASIC": 5,  "SAVER": 10, "FLEX": 10, "BUSINESS": 15},
    "Gold":   {"BASIC": 10, "SAVER": 15, "FLEX": 15, "BUSINESS": 20},
}
# fmt: on
DISRUPTION_BONUS = 5
MAX_DISCOUNT = 25


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
    # a disruption on this booking has been made up for, so it stops earning goodwill
    compensated: bool = False
    # set by quote_change, consumed by rebook. a change is never discounted, so there is
    # nothing else for the desk to put on a booking between the quote and the reissue
    pending_quote_usd: float | None = None


@dataclass
class Airline:
    routes: dict[str, Route]
    travelers: dict[str, Traveler]
    bookings: dict[str, Booking]
    # every day of the timetable is identical until someone touches it, so a day's
    # inventory is opened lazily and only the days in play are ever held
    departures: dict[str, Departure] = field(default_factory=dict)

    def departure(self, route: Route, day: str) -> Departure:
        # a string key, so the airline persists as readable JSON
        key = f"{route.flight_no}@{day}"
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
        Traveler("dana@example.com", "Dana Whitfield", "Gold"),
        Traveler("ortiz@example.com", "Miguel Ortiz", "Blue"),
        Traveler("raman@example.com", "Priya Raman", "Silver", credit_usd=120.0),
    ]

    def day(offset: int) -> str:
        return (today + timedelta(days=offset)).isoformat()

    bookings = [
        # two bookings on one address: the desk has to ask which one
        Booking("NW7Q2K", "dana@example.com", "NW808", day(1), "SAVER", 1, 0, 742.0),
        Booking("NW3H8L", "dana@example.com", "NW441", day(30), "BASIC", 1, 1, 566.0),
        Booking("NW5T4P", "ortiz@example.com", "NW812", day(5), "BASIC", 2, 0, 1230.0),
        Booking("NW8W6C", "raman@example.com", "NW620", day(12), "BUSINESS", 1, 2, 3620.0),
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
    # remembered by lookup_caller; the phone agent collects it, since the desk is not on
    # the phone
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


def _trace(task_id: str, arrow: str, text: str | None, limit: int = 90) -> None:
    """One line of the trace: which task, which direction, and what was said."""
    logger.info(f"{_short(task_id, 12):<12} {arrow} {_short(text, limit)}")


def _discount_percent(traveler: Traveler, bucket: str, owed: bool) -> int:
    """The caller's tier, plus a bump when we have broken a flight of theirs."""
    percent = DISCOUNT_TIERS[traveler.status][bucket]
    if owed:
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


def policy(today: date) -> str:
    """The desk's instructions. Dated, because the caller will say "next Monday"."""
    return f"""You are the fare desk for Northwind Air. You never speak to the caller; the
phone agent does. Work out what is true, hand back the facts and the numbers, and say
what the phone agent should tell them. Keep it to a few sentences — it has to say this
out loud.

THE NETWORK
Northwind flies San Francisco (SFO) to Tokyo Haneda (HND), Paris (CDG) and Beijing (PEK),
and back. Those four codes are the whole airline: a caller who says "Tokyo" means HND, and
nowhere else is served at all. Never search a code that is not one of them.

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
lookup_caller is what puts an address on file, so run it before pricing or selling
anything even when the phone agent has just read one out.
An address is needed only to reach an account — to see, price or change a booking, or to
sell one. You cannot ask for one yourself, because you are not on the phone: end your
answer by asking the phone agent to get it, and it will come back in the next request.
Once lookup_caller has seen an address it is remembered for the rest of the call, so ask
only when nothing has been confirmed yet, and only when they are ready to buy rather than
while they are still looking.
If the caller has more than one booking, do not pick one for them. Hand back the list by
route, date and flight number — never the reference — and say what to ask.

FARE BUCKETS, cheapest to dearest: BASIC, SAVER, FLEX, BUSINESS.
  BASIC     no changes unless we are at fault, and never refundable — say so plainly
            rather than offering to try. 1 cabin bag, checked bags 45 USD each.
  SAVER     changes for 75 USD plus the fare difference; refunds as travel credit.
            1 cabin bag, checked bags 35 USD each.
  FLEX      free changes, refundable to the original card, 1 checked bag included.
  BUSINESS  free changes, refundable, 2 checked bags included.

DISRUPTION
A flight we cancelled, or delayed more than three hours, is our fault, and we put that right
in kind rather than with a discount: the change fee is waived on any fare, the seat they hold
moves across for nothing however the new flight is priced — only a dearer bucket is charged,
and only for the step up — and a SAVER refund goes back to the card instead of out as credit.
Check the weather at both ends before you recommend an alternative — putting someone on the
next flight into a storm is worse than the delay. Say what you checked.

CHANGES
A Gold member's change fee is waived on any fare too; a fare difference falls away only when
we are at fault. Quote before you commit: quote_change first, rebook second. Never rebook
into a cheaper bucket than the one held, and never into a flight with no seats in that bucket.

GOODWILL DISCOUNTS
Only on a new booking, and only when the caller asks or when we broke a flight of theirs;
never volunteered. Whether they get one is your call; how much is not, and you do not know
the figure — book_flight works it out from their status, the bucket and what we still owe them
for a disruption, refuses the flag when there is nothing to give, and tells you what it gave.
Never name a number before then. A change is never discounted, however they ask and whatever
we did to them: not the fee, not the fare difference, not for someone who had one on the fare
they are leaving — no tool of yours will do it, and after a disruption the seat they hold
already moves for nothing. Where the fare is refundable, refunding and booking again is the
route that carries one; price both and say which leaves them better off.

ENDING THE CALL
When the caller has everything they came for and says goodbye, call end_of_call. It does not
hang up; it tells the phone agent the call can end once it has said your last answer."""


class FareDesk(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=policy(date.today()))

    @function_tool
    async def lookup_caller(self, ctx: RunContext[Userdata], email: str) -> dict[str, Any]:
        """Identify the caller by their address and list every booking on their account.

        The phone agent collects the address, because it is the one on the phone. Once this
        has seen one it is remembered for the rest of the call.

        Args:
            email: the address the phone agent confirmed with the caller.
        """
        userdata = ctx.userdata
        userdata.email = email.strip().lower()
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
        if old.disrupted:
            # our fault, so the seat they hold moves for nothing however this flight is
            # priced; only a dearer bucket is charged, and only for the step up. a bucket the
            # new flight does not sell prices at the one they are taking, which comes to zero
            held = new.route.fares.get(booking.fare, new.route.fares[bucket])
            difference = max(0.0, (new.route.fares[bucket] - held) * booking.passengers)
        else:
            difference = max(0.0, new.route.fares[bucket] * booking.passengers - booking.paid_usd)

        booking.pending_quote_usd = round(fee + difference, 2)
        return {
            "booking_ref": booking.ref,
            "new_flight": new.route.flight_no,
            "new_departure": new.departs,
            "bucket": bucket,
            "seats_left": new.seats.get(bucket, 0),
            "change_fee_usd": fee,
            "fee_waived": waived,
            # why it was waived, since a Gold member gets that anyway
            "airline_at_fault": old.disrupted,
            "fare_difference_usd": round(difference, 2),
            "total_due_usd": booking.pending_quote_usd,
            "hold_expires_in_minutes": 20,
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

        # what quote_change said, in full: a change carries no discount, whoever is asking
        charged = round(booking.pending_quote_usd or 0.0, 2)
        booking.flight_no = new.route.flight_no
        booking.date = new.date
        booking.fare = bucket
        booking.paid_usd += charged
        booking.pending_quote_usd = None

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
                much comes from their status, the bucket and what we still owe them for a
                disruption, not from you. Refuses if there is nothing to give.
        """
        userdata = ctx.userdata
        departure = _departure(userdata, flight_no, _day(date))
        bucket = bucket.upper()

        if not userdata.email:
            raise ToolError(
                "no address on file — run lookup_caller with it first, and ask the phone "
                "agent to collect one if the caller has not given it yet"
            )
        if bucket not in departure.route.fares:
            raise ToolError(f"{departure.route.flight_no} does not sell {bucket}")
        if departure.seats.get(bucket, 0) < passengers:
            raise ToolError(
                f"{departure.departs} has {departure.seats.get(bucket, 0)} {bucket} seats left"
            )

        traveler = userdata.airline.travelers.setdefault(
            userdata.email, Traveler(userdata.email, "new customer", "Blue")
        )
        percent = 0
        owed: list[Booking] = []
        if goodwill_discount:
            # the bump is for a flight of theirs we broke, not for the one they are buying, and
            # it is paid once. a refunded booking still counts: being cancelled on is why they
            # are buying again
            owed = [
                b
                for b in userdata.airline.bookings.values()
                if b.email == traveler.email
                and not b.compensated
                and _booked_departure(userdata, b).disrupted
            ]
            percent = _discount_percent(traveler, bucket, bool(owed))
            # before the seat is sold, so the desk can offer full price instead of promising
            if percent == 0:
                raise ToolError(f"a {traveler.status} member buying {bucket} has nothing to give")

        await ctx.update(f"holding {passengers} on {departure.route.flight_no}")
        await asyncio.sleep(4)  # inventory hold, payment capture

        fare = departure.route.fares[bucket] * passengers * (1 - percent / 100)
        credit_used = min(traveler.credit_usd, fare)
        traveler.credit_usd -= credit_used
        departure.seats[bucket] -= passengers
        for settled in owed:
            settled.compensated = True

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
            raise ToolError(
                "no address on file — run lookup_caller with it first, and ask the phone "
                "agent to collect one if the caller has not given it yet"
            )

        await asyncio.sleep(1)
        return f"itinerary for {booking.flight_no} on {booking.date} sent to {ctx.userdata.email}"

    @function_tool
    async def end_of_call(self, ctx: RunContext[Userdata]) -> str:
        """Called when the caller has everything they came for and is done."""
        # advice to whoever asked; in an ordinary session nobody is waiting on an answer
        if (request := ctx.request) is not None:
            request.set_directive("end_session", reason="caller_done")
        return "nothing outstanding"


@server.a2a_session(
    endpoint="fare-desk",
    description="Fares, seats, bookings, changes, refunds and baggage for Northwind Air.",
)
async def fare_desk(ctx: A2ASessionContext) -> None:
    userdata = Userdata(airline=seed_airline())
    session = AgentSession[Userdata](
        llm=inference.LLM("google/gemma-4-31b-it"), userdata=userdata, max_tool_steps=8
    )

    # the desk's own half of the trace, one line per event under the task it belongs to, so
    # two requests in flight stay apart: ▶ what came in, → a tool call, … a report while it
    # runs, ← its result, ◀ what went back. The phone agent's half is in the other terminal.
    #
    #                ▶ my flight to Tokyo is delayed, what else can you put me on?
    #   5517e27c-a3… → rebook({"booking_ref": "NW7Q2K", "new_flight_no": "NW812", ...})
    #   5517e27c-a3… … rebook: holding a seat on NW812
    #   5517e27c-a3… ← done: {'booking_ref': 'NW7Q2K', 'charged_usd': 302.4}
    #   5517e27c-a3… ◀ moved to NW812, 302.40 charged, the delay waived the fee
    # call id -> the task that made the call, and the name of the tool it calls
    calls: dict[str, tuple[str, str]] = {}

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
        if ev.item.type != "message":
            return
        # messages frame the work, so they take no task id: what came in at the top,
        # what went back at the bottom, and the task's own lines in between
        _trace("", "▶" if ev.item.role == "user" else "◀", ev.item.text_content, limit=200)

    @session.on("tool_execution_updated")
    def _on_tool_execution_updated(ev: ToolExecutionUpdatedEvent) -> None:
        update = ev.update
        if update.type == "tool_call_started":
            call = update.function_call
            task_id = call.extra.get(REQUEST_ID_KEY, "")
            calls[call.call_id] = (task_id, call.name)
            _trace(task_id, "→", f"{call.name}({call.arguments})")
            return

        # a deferred reply's update names several calls and belongs to none of them, so
        # only the two that name one read the table
        if update.type == "tool_call_updated" and not update.silent:
            task_id, name = calls.get(update.call_id, ("", "?"))
            _trace(task_id, "…", f"{name}: {update.message}")
        elif update.type == "tool_call_ended":
            task_id, name = calls.pop(update.call_id, ("", "?"))
            _trace(task_id, "←", f"{update.status}: {update.message}")

    persisted = None
    if DB is not None and ctx.conversation_id:
        # the caller names the conversation; this context is one session in it, under the caller's
        persisted = DB.session(
            ctx.conversation_id, ctx.context_id, parent=ctx.caller_session_id, endpoint="fare-desk"
        )
    await session.start(agent=FareDesk(), persist=persisted)
    if persisted is not None and (messages := session.history.messages()):
        # a fresh session has said nothing yet, so any message here came back from the store
        _trace("", "↺", f"rehydrated {ctx.context_id}: {len(messages)} messages back", limit=200)
    # todo: the expert runs in the server process; a job process per context is planned
    ctx.attach(session)


if __name__ == "__main__":
    cli.run_app(server)
