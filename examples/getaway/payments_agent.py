from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel
from travel import CARDS, Travel, travel

from livekit.agents import Agent, AgentHandler, InputRequired, RunContext, RunResult, ToolError
from livekit.agents.a2a import A2ASessionContext

Card = Literal["visa", "amex"]


class Wallet(BaseModel):
    """The card the traveler picked, kept on the payments lane for the trip's next charge."""

    card: Card | None = None


AUTHORIZING_SECONDS = 5
"""How long the card network takes to authorize a charge."""

payments = Agent(
    instructions=(
        "You are Getaway's payments desk. The flight desk and the hotel desk ask you to charge a "
        "booking; what you say, and the questions you ask, reach the traveler through them. A "
        "charge is charge with the amount and what it is for, as the desk wrote them; it asks "
        "which card, then reads the charge back for the traveler's yes. Their answer is the next "
        "request: a card or a yes, call it again with it; a no, charge nothing. Answer with what "
        "charge returned, in one plain sentence."
    ),
    llm="google/gemma-4-31b-it",
    deps_type=Travel,
    state_type=Wallet,
)


def read_back(
    ctx: RunContext[Travel, Wallet], amount: int, description: str, card: Card | None = None
) -> str | InputRequired:
    """What the traveler approves before anything is charged; the card first, when unpicked."""
    card = card or ctx.state.card
    if card is None:
        return InputRequired(f"Which card should pay: {CARDS['visa']} or {CARDS['amex']}?")
    return f"Please confirm: ${amount:,} for {description}, on {CARDS[card]}. Yes or no?"


@payments.tool(requires_approval=read_back)
async def charge(
    ctx: RunContext[Travel, Wallet], amount: int, description: str, card: Card | None = None
) -> str:
    """Charge a booking. It asks which card when the traveler has not picked one, then reads the
    charge back; each answer comes back as this call's output, and then it is called again with it.

    Args:
        amount: the total, in whole dollars
        description: what it pays for, as the desk wrote it
        card: the card the traveler picked
    """
    card = card or ctx.state.card
    if card is None:
        raise ToolError("no card yet; call again with the traveler's pick")
    await ctx.update(f"Authorizing ${amount:,} on {CARDS[card]}.")
    await asyncio.sleep(AUTHORIZING_SECONDS)
    code = ctx.deps.charge(amount, description, card)
    ctx.state.card = card
    # an amount and a receipt reach the traveler as written, through every desk on the way
    await ctx.say(f"Charged ${amount:,} to {CARDS[card]}, receipt {code}.")
    return f"charged; the payment code is {code}"


desk = AgentHandler(lane="payments")


@desk.a2a_session(
    endpoint="payments",
    description="Getaway's payments desk: charges a booking to the traveler's card.",
    public=False,
    inbox="live",
    state=Wallet,
)
async def text(ctx: A2ASessionContext[Wallet]) -> RunResult[str]:
    return await payments.run(
        ctx.message, message_history=ctx.message_history, state=ctx.state, deps=travel
    )
