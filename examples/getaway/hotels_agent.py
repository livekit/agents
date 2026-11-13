from __future__ import annotations

import asyncio
import datetime
import os

from travel import HOTELS, PaymentMissing, Travel, spoken, travel

from livekit.agents import Agent, AgentHandler, InputRequired, RunContext, RunResult, ToolError
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.delegation import A2ADelegate
from livekit.agents.llm import ToolFlag

# the same payments desk the flight desk charges through
PAYMENTS = os.environ["LIVEKIT_URL"].replace("ws", "http", 1) + "/agents/getaway/default/payments"

SEARCH_SECONDS = (8, 6)
"""How long each step of a hotel search takes: it stands in for the hotels' systems."""
RESERVING_SECONDS = 10
"""How long the hotel takes to confirm a room, once it is paid."""
GUESTS_A_ROOM = 2
"""How many guests one room sleeps."""

hotels = Agent(
    instructions=(
        "You are Getaway's hotel desk. What you say reaches the traveler through the concierge. "
        "Answer in English, from your tools, in one or two plain sentences, and never name a "
        "hotel or a price a tool did not list. Every hotel: check-in at 15:00, check-out at 11:00, "
        "breakfast included. A new request starts with search_hotels, with only what the "
        "traveler said, never a guess; it asks for the rest, never you, and you offer the "
        "cheapest two or three. Once they pick one, payments charges its total, with the hotel, "
        "nights and guests; a question payments asks is answered by calling it again with the "
        "traveler's answer, as said. Then book_hotel with the payment code it gave. Only its "
        "booked means booked; a search the traveler no longer wants, you cancel."
    ),
    llm="google/gemma-4-31b-it",
    deps_type=Travel,
    tools=[A2ADelegate(PAYMENTS, name="payments")],
)


@hotels.instructions
def today(ctx: RunContext[Travel, None]) -> str:
    return f"Today is {ctx.deps.today():%A, %B %d, %Y}; dates are in the future."


@hotels.tool(flags=ToolFlag.CANCELLABLE)
async def search_hotels(
    ctx: RunContext[Travel, None],
    city: str,
    check_in: datetime.date | None = None,
    nights: int | None = None,
    guests: int | None = None,
) -> str | InputRequired:
    """Hotels with rooms free, one per line, as `<name>, <stars>, <price a room a night>, <total>`.
    It takes a while, and it asks for what the traveler has not said.

    Args:
        city: the city they stay in
        check_in: the first night, YYYY-MM-DD
        nights: how many nights; left out when they did not say
        guests: how many stay; left out when they did not say
    """
    if check_in is None:
        return InputRequired("Which night do you check in?")
    if check_in < ctx.deps.today():
        raise ToolError("that night is past; check the year")
    if not nights:
        return InputRequired("How many nights?")
    if not guests:
        return InputRequired("How many guests?")
    rooms = -(-guests // GUESTS_A_ROOM)
    await ctx.update(f"Checking {len(HOTELS)} hotels in {city} from {spoken(check_in)}.")
    await asyncio.sleep(SEARCH_SECONDS[0])
    found = ctx.deps.search_hotels(city, check_in)
    await ctx.update(f"Rooms at {len(found)}; sorting by price.")
    await asyncio.sleep(SEARCH_SECONDS[1])
    return (
        "\n".join(
            f"{h.name}, {h.stars} stars, ${h.nightly} a room a night, "
            f"${h.nightly * nights * rooms:,} for {nights} nights, {rooms} room(s)"
            for h in found
        )
        or "every hotel is full that night"
    )


@hotels.tool
async def book_hotel(
    ctx: RunContext[Travel, None],
    hotel: str,
    city: str,
    check_in: datetime.date,
    nights: int,
    guests: int,
    payment_code: str,
) -> str:
    """Book a hotel once payments charged its total.

    Args:
        hotel: the hotel's name, as search_hotels wrote it
        city: the city it is in
        check_in: the first night, YYYY-MM-DD
        nights: how many nights
        guests: how many stay
        payment_code: the code payments gave for the charge
    """
    if check_in < ctx.deps.today():
        raise ToolError("that night is past; check the year")
    found = next(
        (h for h in ctx.deps.search_hotels(city, check_in) if h.name.lower() == hotel.lower()),
        None,
    )
    if found is None:
        raise ToolError("that hotel has no room that night; search again")
    rooms = -(-guests // GUESTS_A_ROOM)
    try:
        ctx.deps.redeem(payment_code, found.nightly * nights * rooms)
    except PaymentMissing:
        raise ToolError("no charge with that code covers this stay; charge it first") from None
    await ctx.update(f"Reserving the room at {found.name}.")
    await asyncio.sleep(RESERVING_SECONDS)
    ctx.deps.bookings.append(f"{found.name}, {nights} nights from {check_in}, {rooms} room(s)")
    return f"booked {found.name}, {rooms} room(s), {nights} nights from {spoken(check_in)}"


desk = AgentHandler(lane="hotels")


@desk.a2a_session(
    endpoint="hotels",
    description="Getaway's hotel desk: rooms and prices in any city, and booking one.",
    public=False,
    inbox="live",
)
async def text(ctx: A2ASessionContext[None]) -> RunResult[str]:
    return await hotels.run(ctx.message, message_history=ctx.message_history, deps=travel)
