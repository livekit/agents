"""An email returns desk opens RMAs from typed photo references and reads return status."""

from dataclasses import dataclass

import httpx
from pydantic import BaseModel

from livekit.agents import Agent, AgentServer, RunContext, RunResult, persisted
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage

SHOP = "https://api.larkfield.example"
WAREHOUSE = "https://api.larkfield-returns.example"


@dataclass
class Return:
    """Application data retained through explicit context saves."""

    order_id: str | None = None
    rma_id: str | None = None


class Photo(BaseModel):
    """An attachment on the email. The item keeps a reference; the bytes are not in the thread."""

    ref: str
    name: str


@dataclass
class Deps:
    shop: httpx.AsyncClient
    warehouse: httpx.AsyncClient


@dataclass
class AppContext:
    services: Deps
    returned: Return = persisted(default_factory=Return)


support = Agent[AppContext](
    instructions="Larkfield Outfitters returns desk. Replies are email: a greeting, short paragraphs, every question in the message answered, a sign-off. Markdown is rendered.",
    llm="google/gemma-4-31b-it",
)


@support.tool
async def start_return(
    ctx: RunContext[AppContext], order_id: str, reason: str, photos: list[Photo]
) -> str:
    """Open a return for an order. Damaged items need the photos the customer attached."""
    body = {"order": order_id, "reason": reason, "photos": [p.ref for p in photos]}
    r = await ctx.context.services.shop.post(
        "/returns", json=body, headers={"Idempotency-Key": ctx.function_call.call_id}
    )
    rma = r.json()
    ctx.context.returned.order_id, ctx.context.returned.rma_id = (order_id, rma["rma_id"])
    await ctx.store.save_context(ctx.context)
    return f"RMA {rma['rma_id']} opened. Prepaid label: {rma['label_url']}. Refund on receipt."


@support.tool
async def return_status(ctx: RunContext[AppContext]) -> dict:
    """Read the current warehouse status of the customer's return."""
    if ctx.context.returned.rma_id is None:
        return {"status": "no return started"}
    response = await ctx.context.services.warehouse.get(f"/returns/{ctx.context.returned.rma_id}")
    response.raise_for_status()
    return response.json()


server = AgentServer(storage=AgentDBStorage())


@server.a2a_session(endpoint="returns", description="Larkfield Outfitters returns.")
async def inbox(ctx: A2ASessionContext) -> RunResult[str]:
    deps = Deps(
        shop=httpx.AsyncClient(base_url=SHOP), warehouse=httpx.AsyncClient(base_url=WAREHOUSE)
    )
    app = AppContext(services=deps)
    await ctx.store.load_context(app)
    result = await support.run(ctx.message, message_history=ctx.message_history, context=app)
    await ctx.store.save_context(app)
    return result
