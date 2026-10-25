"""
- Larkfield Outfitters' order line: order status, and a watch for an item that is out of stock.
- `notify_when_back` submits `CheckStock`, a pure code `AgentTask` that checks again with `delay=`
  or hands over to `Offer`.
- `Offer` asks the shopper with `InputRequired(question)`; a question has no timeout, the offer
  lapses by the task's own `timeout=` on `submit`; the task's model reads the reply and calls
  `decide`.
- Every hook and tool returns the A2A transition: `Completed`, `InputRequired`, or a plain tool
  result.
- `on_timeout` completes the task when the offer lapses.
"""

from dataclasses import dataclass, field
from datetime import timedelta

import httpx

from livekit.agents import (
    Agent,
    AgentServer,
    AgentTask,
    Completed,
    InputRequired,
    RunContext,
    function_tool,
)
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage

SHOP = "https://api.larkfield.example"


@dataclass
class Shopper:
    """State: saved with the lane after every run, shared by the agent and its tasks."""

    holds: list[str] = field(default_factory=list)  # hold ids placed for this shopper


@dataclass
class Deps:
    """What the tools and the tasks need for one run. Never saved."""

    shop: httpx.AsyncClient


shop = Agent(
    instructions="Larkfield Outfitters customer line: orders and stock.",
    llm="google/gemma-4-31b-it",
    deps_type=Deps,
    state_type=Shopper,
)


async def in_stock(api: httpx.AsyncClient, sku: str) -> int:
    r = await api.get(f"/inventory/{sku}")
    return int(r.json()["available"])


# Background work is an AgentTask submitted instead of awaited: it runs as its own A2A task beside
# the conversation. A pure code task has on_enter and no model. A task that must talk to the
# shopper returns InputRequired with the question in the code's words; its model reads the answer.
# Every return value is the A2A state the task moves to; a plain value is a tool result.


@dataclass
class CheckStock(AgentTask[None]):  # pure code: check again later, or hand to Offer
    sku: str

    def __post_init__(self) -> None:  # a task builds its deps like a door does
        super().__init__(deps=Deps(shop=httpx.AsyncClient(base_url=SHOP)))

    async def on_enter(self, ctx: RunContext[Deps, Shopper]) -> Completed[None]:
        if await in_stock(ctx.deps.shop, self.sku) == 0:
            await ctx.update(f"{self.sku} still out of stock.")
            await ctx.submit(CheckStock(self.sku), delay=timedelta(hours=6))
        else:
            await ctx.submit(Offer(self.sku), timeout=timedelta(days=2))
        return Completed(None)


@dataclass
class Offer(AgentTask[str]):  # talks to the shopper: offers the hold, places it when they say yes
    sku: str

    def __post_init__(self) -> None:
        super().__init__(
            instructions="When the shopper answers, call decide.",
            llm="google/gemma-4-31b-it",
            deps=Deps(shop=httpx.AsyncClient(base_url=SHOP)),
        )

    async def on_enter(self, ctx: RunContext[Deps, Shopper]) -> InputRequired:
        # the question is the code's, word for word; the model answers through decide
        return InputRequired(
            f"Good news: {self.sku} is back in stock. Want me to hold one for 48 hours?",
            verbatim=True,
        )

    @function_tool
    async def decide(self, ctx: RunContext[Deps, Shopper], hold: bool) -> Completed[str]:
        """Whether the shopper wants the item held."""
        if not hold:
            return Completed("No problem, I won't hold it.")
        r = await ctx.deps.shop.post(
            "/holds", json={"sku": self.sku}, headers={"Idempotency-Key": ctx.task_id}
        )
        ctx.state.holds.append(r.json()["hold_id"])
        return Completed(f"Held for 48 hours, hold {r.json()['hold_id']}.")

    async def on_timeout(self, ctx: RunContext[Deps, Shopper]) -> Completed[str]:
        return Completed(f"The hold offer on {self.sku} lapsed; it is on general sale again.")


@shop.tool
async def order_status(ctx: RunContext[Deps, Shopper], order_id: str) -> dict:
    """Where an order is. Read-only, and the most common question a store line gets."""
    r = await ctx.deps.shop.get(f"/orders/{order_id}")
    return r.json()


@shop.tool
async def notify_when_back(ctx: RunContext[Deps, Shopper], sku: str) -> str:
    handle = await ctx.submit(CheckStock(sku))  # an A2A task in this thread; its id is its stream
    return f"I'll watch it and let you know. ({handle.id})"


server = AgentServer(storage=AgentDBStorage())


@server.a2a_session(endpoint="shop", description="Larkfield Outfitters.", state=Shopper)
async def text(ctx: A2ASessionContext[Shopper]) -> str:
    deps = Deps(shop=httpx.AsyncClient(base_url=SHOP))
    result = await shop.run(
        ctx.message, message_history=ctx.message_history, state=ctx.state, deps=deps
    )
    return result.output


# A task's constructor arguments are JSON in its row; its progress is its own A2A history; new
# code runs old rows, so an upgrade is a deploy. The shopper's "yes please" arrives as a turn of
# shop, which sees Offer waiting and hands the conversation to it; Offer's model reads it and
# calls decide.
# The Completed value is the task's last message to the shopper.
