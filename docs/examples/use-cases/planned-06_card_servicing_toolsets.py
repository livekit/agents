"""Harbor Credit Union card servicing: cards, payments and rewards as toolsets."""

from dataclasses import dataclass

import httpx

from livekit.agents import (
    Agent,
    AgentServer,
    RunContext,
    RunResult,
    ToolError,
    function_tool,
    persisted,
)
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.llm import Toolset
from livekit.agents.llm.mcp import MCPServerHTTP, MCPToolset
from livekit.agents.storage import AgentDBStorage

BANK = "https://api.harborcu.example"


@dataclass
class Member:
    """Application data retained through explicit context saves."""

    member_id: str | None = None
    verified: bool = False


@dataclass
class Deps:
    """What the toolsets need for one run. Never saved."""

    bank: httpx.AsyncClient


@dataclass
class AppContext:
    services: Deps
    member: Member = persisted(default_factory=Member)


def verified_member_id(ctx: RunContext[AppContext]) -> str:
    member = ctx.context.member
    if not member.verified or member.member_id is None:
        raise ToolError("Verify the member before accessing account details.")
    return member.member_id


@function_tool
async def card_status(ctx: RunContext[AppContext], last_four: str) -> dict:
    member_id = verified_member_id(ctx)
    r = await ctx.context.services.bank.get(f"/members/{member_id}/cards/{last_four}")
    return r.json()


@function_tool(needs_approval=True)
async def block_card(ctx: RunContext[AppContext], last_four: str, date_of_birth: str) -> str:
    """Block a card before full verification: the last four digits and the date of birth."""
    await ctx.context.services.bank.post(f"/cards/{last_four}/block", json={"dob": date_of_birth})
    return f"card ending {last_four} blocked; a replacement ships in 3 to 5 days"


@function_tool
async def balance(ctx: RunContext[AppContext]) -> dict:
    member_id = verified_member_id(ctx)
    r = await ctx.context.services.bank.get(f"/members/{member_id}/balance")
    return r.json()


@function_tool(needs_approval=True)
async def pay_bill(ctx: RunContext[AppContext], payee: str, amount_cents: int) -> str:
    """Set up a bill payment. No card number crosses the voice channel: the member gets a link."""
    body = {"member": verified_member_id(ctx), "payee": payee, "amount": amount_cents}
    r = await ctx.context.services.bank.post("/payments/links", json=body)
    return f"give the member this link to pay {payee} securely: {r.json()['url']}"


cards = Toolset(id="cards", tools=[card_status, block_card])
payments = Toolset(id="payments", tools=[balance, pay_bill])
rewards = MCPToolset(id="rewards", mcp_server=MCPServerHTTP("https://mcp.rewards-partner.example/"))


member_services = Agent[AppContext](
    instructions="Harbor Credit Union member services.",
    llm="google/gemma-4-31b-it",
    tools=[cards, payments, rewards],
)


@member_services.tool
async def verify(ctx: RunContext[AppContext], last_four: str, date_of_birth: str) -> str:
    details = {"last4": last_four, "dob": date_of_birth}
    r = await ctx.context.services.bank.post("/members/verify", json=details)
    if r.status_code != 200:
        await ctx.store.save_context(ctx.context)
        return "those details do not match"
    ctx.context.member.member_id = r.json()["member_id"]
    ctx.context.member.verified = True
    await ctx.store.save_context(ctx.context)
    return "verified; account details can now be accessed"


server = AgentServer(storage=AgentDBStorage())


@server.a2a_session(endpoint="cards", description="Harbor Credit Union cards.")
async def text(ctx: A2ASessionContext) -> RunResult[str]:
    async with httpx.AsyncClient(base_url=BANK) as bank:
        app = AppContext(services=Deps(bank=bank))
        await ctx.store.load_context(app)
        result = await member_services.run(
            ctx.message, message_history=ctx.message_history, context=app
        )
        await ctx.store.save_context(app)
        return result
