from __future__ import annotations

import asyncio
import datetime
import os

from travel import AIRLINES, PaymentMissing, Travel, spoken, travel

from livekit.agents import Agent, AgentHandler, InputRequired, RunContext, RunResult, ToolError
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.delegation import A2ADelegate
from livekit.agents.llm import ToolFlag

# the payments desk on the project's own host: a chain, concierge to flights to payments
PAYMENTS = os.environ["LIVEKIT_URL"].replace("ws", "http", 1) + "/agents/getaway/default/payments"

SEARCH_SECONDS = (10, 8)
"""How long each step of a flight search takes: it stands in for the airlines' systems."""
TICKETING_SECONDS = 15
"""How long issuing the ticket takes, once the fare is paid."""

flights = Agent(
    instructions=(
        "You are Getaway's flight desk. What you say reaches the traveler through the concierge. "
        "Answer in English, from your tools, in one or two plain sentences, and never name a "
        "flight or a fare a tool did not list. A new request starts with search_flights, with "
        "only what the traveler said, never a guess; it asks for the rest, never you, and you "
        "offer the cheapest two or three. Once they pick one, payments charges its total, "
        "with the flight and its dates; a question payments asks is answered by calling it again "
        "with the traveler's answer, as said. Then book_flight with the payment code it gave. "
        "Only its booked means booked; a flight the traveler no longer wants, you cancel."
    ),
    llm="google/gemma-4-31b-it",
    deps_type=Travel,
    tools=[A2ADelegate(PAYMENTS, name="payments")],
)


@flights.instructions
def today(ctx: RunContext[Travel, None]) -> str:
    return f"Today is {ctx.deps.today():%A, %B %d, %Y}; dates are in the future."


@flights.tool(flags=ToolFlag.CANCELLABLE)
async def search_flights(
    ctx: RunContext[Travel, None],
    destination: str,
    origin: str | None = None,
    leave: datetime.date | None = None,
    back: datetime.date | None = None,
    travelers: int | None = None,
) -> str | InputRequired:
    """Round-trip flights, one per line, as `<id> - <airline>, out <when>, back <when>, <fare>`.
    It takes a while, and it asks for what the traveler has not said.

    Args:
        destination: the city they fly to
        origin: the city they fly from; left out when they named none
        leave: the day they fly out, YYYY-MM-DD
        back: the day they fly back, YYYY-MM-DD
        travelers: how many fly; left out when they did not say
    """
    if not origin:
        return InputRequired("Which city are you flying from?")
    if leave is not None and leave < ctx.deps.today():
        raise ToolError("that day is past; check the year")
    if leave is None or back is None:
        return InputRequired("Which days do you fly out and back?")
    if not travelers:
        return InputRequired("How many travelers?")
    await ctx.update(f"Searching {', '.join(AIRLINES)} from {origin} to {destination}.")
    await asyncio.sleep(SEARCH_SECONDS[0])
    found = ctx.deps.search_flights(origin, destination, leave, back)
    await ctx.update(f"Comparing {len(found)} fares.")
    await asyncio.sleep(SEARCH_SECONDS[1])
    return "\n".join(
        f"{f.id} - {f.airline}, out {spoken(f.out)}, back {spoken(f.back)}, "
        f"${f.fare} a traveler, ${f.fare * travelers:,} for {travelers}"
        for f in found
    )


@flights.tool
async def book_flight(
    ctx: RunContext[Travel, None], flight_id: str, travelers: int, payment_code: str
) -> str:
    """Book a flight once payments charged its total.

    Args:
        flight_id: the flight's id, as search_flights wrote it
        travelers: how many fly
        payment_code: the code payments gave for the charge
    """
    flight = ctx.deps.flights.get(flight_id.strip().upper())
    if flight is None:
        raise ToolError("no such flight; search again")
    try:
        ctx.deps.redeem(payment_code, flight.fare * travelers)
    except PaymentMissing:
        raise ToolError("no charge with that code covers this flight; charge it first") from None
    await ctx.update(f"Issuing the ticket with {flight.airline}.")
    await asyncio.sleep(TICKETING_SECONDS)
    ctx.deps.bookings.append(f"{flight.id} for {travelers}")
    return (
        f"booked {flight.airline} {flight.id} for {travelers}, out {spoken(flight.out)}, "
        f"back {spoken(flight.back)}"
    )


desk = AgentHandler(lane="flights")


@desk.a2a_session(
    endpoint="flights",
    description="Getaway's flight desk: round-trip fares, and booking one.",
    public=False,
    inbox="live",
)
async def text(ctx: A2ASessionContext[None]) -> RunResult[str]:
    return await flights.run(ctx.message, message_history=ctx.message_history, deps=travel)
