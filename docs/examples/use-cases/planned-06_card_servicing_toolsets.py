"""
- Harbor Credit Union card servicing: cards, payments and rewards as toolsets.
- `Toolset` groups tools; `FilteredToolset` opens account tools only once the `Member` state says
  verified, except blocking a lost card.
- `PrefixedToolset` mounts the rewards partner's MCP server without name clashes.
- `requires_approval=True` and `InputRequired` from `pay_bill` make the member confirm; the next
  message becomes the call's output.
- No card number crosses the channel: payments go through a link the reply carries.
"""

from dataclasses import dataclass

import httpx

from livekit.agents import Agent, AgentServer, InputRequired, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.llm import FilteredToolset, PrefixedToolset, Toolset, mcp
from livekit.agents.storage import AgentDBStorage

BANK = "https://api.harborcu.example"


@dataclass
class Member:
    """State: saved with the lane after every run."""

    member_id: str | None = None
    verified: bool = False


@dataclass
class Deps:
    """What the toolsets need for one run. Never saved."""

    bank: httpx.AsyncClient


cards = Toolset(name="cards")
payments = Toolset(name="payments")


@cards.tool
async def card_status(ctx: RunContext[Deps, Member], last_four: str) -> dict:
    r = await ctx.deps.bank.get(f"/members/{ctx.state.member_id}/cards/{last_four}")
    return r.json()


@cards.tool(requires_approval=True)  # the member's next message answers yes or no
async def block_card(ctx: RunContext[Deps, Member], last_four: str, date_of_birth: str) -> str:
    """Block a card before full verification: the last four digits and the date of birth."""
    await ctx.deps.bank.post(f"/cards/{last_four}/block", json={"dob": date_of_birth})
    return f"card ending {last_four} blocked; a replacement ships in 3 to 5 days"


@payments.tool
async def balance(ctx: RunContext[Deps, Member]) -> dict:
    r = await ctx.deps.bank.get(f"/members/{ctx.state.member_id}/balance")
    return r.json()


@payments.tool
async def pay_bill(
    ctx: RunContext[Deps, Member], payee: str, amount_cents: int, confirmed: bool = False
) -> str | InputRequired:
    """Set up a bill payment. No card number crosses the voice channel: the member gets a link."""
    if amount_cents > 500_000 and not confirmed:
        amount = f"{amount_cents / 100:,.2f}"
        # the turn waits in input-required; the member's answer becomes this call's result and the
        # model calls again with confirmed=True
        question = f"Payments over 5,000 need your confirmation. Send {amount} to {payee}?"
        return InputRequired(question)
    body = {"member": ctx.state.member_id, "payee": payee, "amount": amount_cents}
    r = await ctx.deps.bank.post("/payments/links", json=body)
    return f"give the member this link to pay {payee} securely: {r.json()['url']}"


# The rewards partner's catalog and redemptions, as tools.
rewards = mcp.MCPServerHTTP("https://mcp.rewards-partner.example/")


def verified_only(ctx: RunContext[Deps, Member], tool) -> bool:
    # blocking a lost card must not wait for full verification
    return ctx.state.verified or tool.name == "block_card"


member_services = Agent(
    instructions="Harbor Credit Union member services.",
    llm="google/gemma-4-31b-it",
    deps_type=Deps,
    state_type=Member,
    toolsets=[
        FilteredToolset(cards, verified_only),
        FilteredToolset(payments, verified_only),
        PrefixedToolset(rewards, "rewards_"),  # rewards_balance, rewards_redeem: no name clashes
    ],
)


@member_services.tool
async def verify(ctx: RunContext[Deps, Member], last_four: str, date_of_birth: str) -> str:
    details = {"last4": last_four, "dob": date_of_birth}
    r = await ctx.deps.bank.post("/members/verify", json=details)
    if r.status_code != 200:
        return "those details do not match"
    ctx.state.member_id = r.json()["member_id"]
    ctx.state.verified = True
    return "verified; account tools are now open"


server = AgentServer(storage=AgentDBStorage())


@server.a2a_session(endpoint="cards", description="Harbor Credit Union cards.", state=Member)
async def text(ctx: A2ASessionContext[Member]) -> str:
    deps = Deps(bank=httpx.AsyncClient(base_url=BANK))
    result = await member_services.run(
        ctx.message, message_history=ctx.message_history, state=ctx.state, deps=deps
    )
    return result.output


# An approval or an InputRequired is a pending call in the history: the member answers later on
# any worker, by conversation or from a form, the answer becomes the call's output, the model
# continues. The account lives behind the bank's API; what this member said and confirmed, in the
# thread.
