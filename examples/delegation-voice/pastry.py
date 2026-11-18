import asyncio

from livekit.agents import Agent, AgentHandler, RunContext, RunResult
from livekit.agents.a2a import A2ASessionContext

MODEL = "google/gemma-4-31b-it"

chef = Agent(
    instructions=(
        "You are Bistro Lumen's pastry kitchen. What you say reaches the guest through the host "
        "on the phone. Plan cakes with plan_cake and reply in one plain sentence."
    ),
    llm=MODEL,
)


@chef.tool
async def plan_cake(ctx: RunContext, occasion: str, guests: int, message: str = "") -> str:
    """Plan a cake for the table.

    Args:
        occasion: what it celebrates
        guests: how many it serves
        message: what to write on it, if anything
    """
    ctx.update("The pastry chef is checking what she can bake for tonight.")
    await asyncio.sleep(5)
    flavour = "dark chocolate and raspberry" if guests > 4 else "lemon and almond"
    written = f', with "{message}" on top' if message else ""
    return f"a {flavour} {occasion} cake for {guests}{written}, 35 euros, ready for dessert"


pastry = AgentHandler()


@pastry.a2a_session(
    partition_key="pastry", endpoint="pastry", description="Bistro Lumen's pastry kitchen: cakes."
)
async def kitchen(ctx: A2ASessionContext) -> RunResult[str]:
    return await chef.run(ctx.message, context=ctx.store.bind())
