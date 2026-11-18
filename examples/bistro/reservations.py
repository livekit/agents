import asyncio
import random
from datetime import timedelta

from livekit.agents import (
    Agent,
    AgentHandler,
    DelegationOptions,
    InputRequired,
    RunContext,
    RunResult,
    TextSession,
    inference,
)
from livekit.agents.a2a import A2ASessionContext

MODEL = "google/gemma-4-31b-it"

# The reservation book is slow: each step stands for a real system the slow brain waits on.
SEARCH_SECONDS = 8
CHECK_SECONDS = 2
HOLD_SECONDS = 3

book = Agent(
    instructions=(
        "You are the reservations desk of Bistro Lumen. What you say reaches the guest through "
        "the host on the phone. Find tables with find_table and hold them with hold_table, "
        "never inventing availability. Seating is inside or terrace: pass it only when the "
        "guest chose; find_table asks otherwise. A guest's answer to a question is the next "
        "request: call the tool again with it. Once find_table reports a free table, call "
        "hold_table right away: it reads the hold back to the guest itself, so never ask for "
        "that confirmation yourself. Reply in one plain sentence."
    ),
    llm=MODEL,
)


@book.tool
async def find_table(
    ctx: RunContext, party_size: int, time: str, seating: str | None = None
) -> str | InputRequired:
    """Look for a table in the book.

    Args:
        party_size: how many guests
        time: when, as the guest said it, e.g. "tonight at 8"
        seating: "inside" or "terrace", only when the guest chose
    """
    if seating is None:
        # Progress streams to the host while this keeps working; it never ends the call's turn.
        ctx.update(f"Looking through the book for {party_size} at {time}.")
        await asyncio.sleep(SEARCH_SECONDS / 2)
        ctx.update("Checking with the floor manager which sections are free.")
        await asyncio.sleep(SEARCH_SECONDS / 2)
        # A question in the middle of the work: the task waits for the guest's answer, which
        # the host's routing model recognizes and sends back to this exact task.
        return InputRequired("We have room inside and on the terrace. Which would they like?")
    ctx.update(f"Checking the {seating} for {party_size}.")
    await asyncio.sleep(CHECK_SECONDS)
    return f"A {seating} table for {party_size} is free {time}; hold it with hold_table now."


def read_back(ctx: RunContext, party_size: int, time: str, seating: str) -> str:
    return f"Shall I hold a {seating} table for {party_size} {time}, with a 20 euro deposit?"


@book.tool(needs_approval=read_back)
async def hold_table(ctx: RunContext, party_size: int, time: str, seating: str) -> str:
    """Hold the table and take the deposit. Runs only after the guest agreed to the read-back.

    Args:
        party_size: how many guests
        time: when, as found by find_table
        seating: "inside" or "terrace"
    """
    ctx.update("Holding the table and taking the deposit.")
    await asyncio.sleep(HOLD_SECONDS)
    code = f"LUMEN-{random.randint(100, 999)}"
    # Exact words: the host speaks the code as written, across the A2A hop, never rephrased.
    await ctx.say(f"Your confirmation code is {code}.")
    return f"held: {seating}, {party_size} guests, {time}, code {code}"


reservations = AgentHandler()


# "batch": messages that arrive while a turn runs (a guest changing their mind mid-search) are
# merged into one next turn; the earlier tasks complete pointing at it with livekit/mergedInto.
@reservations.a2a_session(
    partition_key="reservations",
    endpoint="reservations",
    description="Bistro Lumen's reservations: finds and holds tables.",
    inbox="batch",
    debounce=timedelta(seconds=1),
)
async def desk(ctx: A2ASessionContext) -> RunResult[str]:
    async with inference.LLM(MODEL) as routing:
        return await TextSession(
            agent=book, delegation=DelegationOptions(routing_model=routing)
        ).run(ctx.message, context=ctx.store.bind())
